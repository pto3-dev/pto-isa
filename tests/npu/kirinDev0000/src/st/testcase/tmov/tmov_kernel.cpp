/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>

using namespace pto;

template <
    typename cType, typename aType, typename bType, typename biasType, int M, int K, int N, int ValidM, int ValidK,
    int ValidN>
__global__ AICORE void runTMovL12Bias(__gm__ cType* out, __gm__ aType* src0, __gm__ bType* src1, __gm__ biasType* src2)
{
    // static shape
    using GlobalDataSrc0 = GlobalTensor<
        aType, pto::Shape<1, 1, 1, ValidM, ValidK>,
        pto::Stride<ValidM * ValidK, ValidM * ValidK, ValidM * ValidK, ValidK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<
        bType, pto::Shape<1, 1, 1, ValidK, ValidN>,
        pto::Stride<ValidK * ValidN, ValidK * ValidN, ValidK * ValidN, ValidN, 1>>;
    using GlobalDataSrc2 =
        GlobalTensor<biasType, pto::Shape<1, 1, 1, 1, ValidN>, pto::Stride<ValidN, ValidN, ValidN, ValidN, 1>>;
    using GlobalDataOut = GlobalTensor<
        cType, pto::Shape<1, 1, 1, ValidM, ValidN>,
        pto::Stride<ValidM * ValidN, ValidM * ValidN, ValidM * ValidN, ValidN, 1>>;

    constexpr int alignN = ((N * sizeof(biasType) + 63) / 64) * 64 / sizeof(biasType); // bias aligned to 64 bits

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, aType, M, K, BLayout::ColMajor, ValidM, ValidK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, bType, K, N, BLayout::ColMajor, ValidK, ValidN, SLayout::RowMajor, 512>;
    using TileMatBiasData =
        Tile<TileType::Mat, biasType, 1, alignN, BLayout::RowMajor, 1, ValidN, SLayout::NoneBox, 512>;

    using RightTile = TileRight<bType, K, N, ValidK, ValidN>;
    using AccTile = TileAcc<cType, M, N, ValidM, ValidN>;
    using BiasTile = Tile<TileType::Bias, cType, 1, alignN, BLayout::RowMajor, 1, ValidN, SLayout::NoneBox, 512>;
    using VecTile = Tile<TileType::Vec, cType, M, N, BLayout::RowMajor, ValidM, ValidN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileMatBiasData biasMatTile;
    TASSIGN<0x0>(aMatTile);
    TASSIGN<M * K * sizeof(aType)>(bMatTile);
    TASSIGN<M * K * sizeof(aType) + K * N * sizeof(bType)>(biasMatTile);

    RightTile bTile;
    AccTile cTile;
    BiasTile biasTile;
    VecTile dstTile;
    TASSIGN<0x0>(bTile);
    TASSIGN<M * K * sizeof(aType) + K * N * sizeof(bType) + alignN * sizeof(biasType)>(cTile);
    TASSIGN<0x0>(biasTile);
    TASSIGN<0x0>(dstTile);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    TLOAD(biasMatTile, src2Global);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    TMOV(bTile, bMatTile);
    TMOV(biasTile, biasMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL_BIAS(cTile, aMatTile, bTile, biasTile);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TMOV(dstTile, cTile);

    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <int32_t tilingKey>
void launchTMovL12Bias(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* bias, void* stream)
{
    if constexpr (tilingKey == 4) {
        runTMovL12Bias<int32_t, int8_t, int8_t, int32_t, 128, 96, 64, 128, 96, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<int32_t*>(out), reinterpret_cast<int8_t*>(src0), reinterpret_cast<int8_t*>(src1),
            reinterpret_cast<int32_t*>(bias));
    } else if constexpr (tilingKey == 5) {
        runTMovL12Bias<int32_t, int8_t, int8_t, int32_t, 32, 32, 64, 31, 32, 63><<<1, nullptr, stream>>>(
            reinterpret_cast<int32_t*>(out), reinterpret_cast<int8_t*>(src0), reinterpret_cast<int8_t*>(src1),
            reinterpret_cast<int32_t*>(bias));
    }
}

template void launchTMovL12Bias<4>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream);
template void launchTMovL12Bias<5>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream);

template <
    typename cType, typename aType, typename bType, int M, int K, int N, int ValidM, int ValidK, int ValidN,
    int Block = 1>
__global__ AICORE void runTMovAcc2Vec(__gm__ cType* out, __gm__ aType* src0, __gm__ bType* src1)
{
    // static shape
    using GlobalDataSrc0 = GlobalTensor<
        aType, pto::Shape<1, 1, 1, ValidM, ValidK>,
        pto::Stride<ValidM * ValidK, ValidM * ValidK, ValidM * ValidK, ValidK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<
        bType, pto::Shape<1, 1, 1, ValidK, ValidN>,
        pto::Stride<ValidK * ValidN, ValidK * ValidN, ValidK * ValidN, ValidN, 1>>;
    using GlobalDataOutNd = GlobalTensor<
        cType, pto::Shape<1, 1, 1, ValidM, ValidN>,
        pto::Stride<ValidM * ValidN, ValidM * ValidN, ValidM * ValidN, ValidN, 1>>;
    using GlobalDataOutNz = GlobalTensor<
        cType, pto::Shape<1, ValidM / Block, ValidN / Block, Block, Block>,
        pto::Stride<ValidM * ValidN, ValidN * Block, Block * Block, Block, 1>, pto::Layout::NZ>;
    using GlobalDataOut = std::conditional_t<Block == 1, GlobalDataOutNd, GlobalDataOutNz>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, aType, M, K, BLayout::ColMajor, ValidM, ValidK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, bType, K, N, BLayout::ColMajor, ValidK, ValidN, SLayout::RowMajor, 512>;

    using RightTile = TileRight<bType, K, N, ValidK, ValidN>;
    using AccTile = TileAcc<cType, M, N, ValidM, ValidN>;

    using VecTileNd = Tile<TileType::Vec, cType, M, N, BLayout::RowMajor, ValidM, ValidN>;
    using VecTileNz = Tile<TileType::Vec, cType, M, N, BLayout::ColMajor, ValidM, ValidN, SLayout::RowMajor, 1024>;
    using VecTile = std::conditional_t<Block == 1, VecTileNd, VecTileNz>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN<0x0>(aMatTile);
    TASSIGN<M * K * sizeof(aType)>(bMatTile);

    RightTile bTile;
    AccTile cTile;
    VecTile dstTile;

    TASSIGN<0x0>(bTile);
    TASSIGN<M * K * sizeof(aType) + K * N * sizeof(bType)>(cTile);
    TASSIGN<0x0>(dstTile);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    TMOV(bTile, bMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL(cTile, aMatTile, bTile);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TMOV(dstTile, cTile);

    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <int32_t tilingKey>
void launchTMovAcc2Vec(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream)
{
    if constexpr (tilingKey == 1) {
        runTMovAcc2Vec<half, half, half, 64, 64, 64, 64, 64, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
    } else if constexpr (tilingKey == 2) {
        runTMovAcc2Vec<half, half, half, 64, 64, 64, 64, 64, 64, 16><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
    }
}

template void launchTMovAcc2Vec<1>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTMovAcc2Vec<2>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
