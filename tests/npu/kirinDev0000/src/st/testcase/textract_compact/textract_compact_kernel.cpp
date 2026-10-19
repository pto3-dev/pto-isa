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
    int M, int K, int N, uint16_t indexM, uint16_t indexK, uint16_t indexN, int baseM, int baseK, int baseN,
    bool isAtranspose, bool isBtranspose, typename T, typename U, typename S>
AICORE inline void runTEXTRACT_UNALIGN(__gm__ T* out, __gm__ U* src0, __gm__ S* src1)
{
    constexpr int mValid = M - indexM;
    constexpr int kValid = K - indexK;
    constexpr int nValid = N - indexN;

    constexpr int c0SizeU = BLOCK_BYTE_SIZE / sizeof(U);
    constexpr int aRowsAlign = (mValid + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW;
    constexpr int aColsAlign = (kValid + c0SizeU - 1) / c0SizeU * c0SizeU;
    constexpr int cRowsAlign = (mValid + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW;
    constexpr int cColsAlign = (nValid + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW;

    using GlobalDataSrc0 = std::conditional_t<
        isAtranspose,
        GlobalTensor<
            U, pto::Shape<1, 1, 1, aRowsAlign, aColsAlign>,
            pto::Stride<1 * aRowsAlign * aColsAlign, 1 * aRowsAlign * aColsAlign, aRowsAlign * aColsAlign, 1, baseM>,
            Layout::DN>,
        GlobalTensor<
            U, pto::Shape<1, 1, 1, aRowsAlign, aColsAlign>,
            pto::Stride<1 * aRowsAlign * aColsAlign, 1 * aRowsAlign * aColsAlign, aRowsAlign * aColsAlign, baseK, 1>,
            Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t<
        isBtranspose,
        GlobalTensor<
            S, pto::Shape<1, 1, 1, baseK, baseN>,
            pto::Stride<1 * baseK * baseN, 1 * baseK * baseN, baseK * baseN, baseN, 1>, Layout::ND>,
        GlobalTensor<
            S, pto::Shape<1, 1, 1, baseK, baseN>,
            pto::Stride<1 * baseK * baseN, 1 * baseK * baseN, baseK * baseN, 1, baseK>, Layout::DN>>;
    using GlobalDataOut = GlobalTensor<
        T, pto::Shape<1, 1, 1, mValid, nValid>,
        pto::Stride<1 * mValid * nValid, 1 * mValid * nValid, mValid * nValid, nValid, 1>>;

    constexpr uint32_t src0Offset = isAtranspose ? (indexK * baseM + indexM) : (indexM * baseK + indexK);
    GlobalDataSrc0 src0Global(src0 + src0Offset);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        Tile<TileType::Mat, U, aRowsAlign, aColsAlign, BLayout::ColMajor, mValid, kValid, SLayout::RowMajor, 512>;
    using TileMatBData = std::conditional_t<
        isBtranspose, Tile<TileType::Mat, S, baseK, baseN, BLayout::ColMajor, baseK, baseN, SLayout::RowMajor, 512>,
        Tile<TileType::Mat, S, baseK, baseN, BLayout::RowMajor, baseK, baseN, SLayout::ColMajor, 512>>;

    using RightTile = TileRightCompact<S, baseK, baseN, kValid, nValid>;
    using ResTile =
        Tile<TileType::Mat, T, cRowsAlign, cColsAlign, BLayout::ColMajor, mValid, nValid, SLayout::RowMajor, 1024>;
    using VecTile = Tile<TileType::Vec, T, cRowsAlign, cColsAlign, BLayout::RowMajor, mValid, nValid>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN<0x0>(aMatTile);
    TASSIGN<aRowsAlign * aColsAlign * sizeof(U)>(bMatTile);

    RightTile bTile;
    ResTile cTile;
    VecTile cVecTile;
    TASSIGN<0x0>(bTile);
    TASSIGN<aRowsAlign * aColsAlign * sizeof(U) + baseK * baseN * sizeof(S)>(cTile);
    TASSIGN<0x0>(cVecTile);

    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**********************************TEXTRACT**********************************/
    TEXTRACT(bTile, bMatTile, indexK, indexN);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL(cTile, aMatTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    /****************************************TMOV + TSTORE*****************************************/
    TMOV(cVecTile, cTile);
    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, cVecTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTEXTRACT_11(__gm__ half* out, __gm__ half* src0, __gm__ half* src1)
{
    runTEXTRACT_UNALIGN<63, 48, 66, 0, 0, 0, 128, 64, 256, false, false>(out, src0, src1);
}
extern "C" __global__ AICORE void launchTEXTRACT_12(__gm__ half* out, __gm__ half* src0, __gm__ half* src1)
{
    runTEXTRACT_UNALIGN<68, 93, 97, 0, 0, 0, 128, 128, 128, true, true>(out, src0, src1);
}
extern "C" __global__ AICORE void launchTEXTRACT_14(__gm__ half* out, __gm__ half* src0, __gm__ half* src1)
{
    runTEXTRACT_UNALIGN<59, 232, 61, 16, 16, 16, 64, 256, 64, true, true>(out, src0, src1);
}
extern "C" __global__ AICORE void launchTEXTRACT_21(__gm__ int32_t* out, __gm__ int8_t* src0, __gm__ int8_t* src1)
{
    runTEXTRACT_UNALIGN<97, 231, 83, 0, 0, 0, 128, 256, 128, false, false>(out, src0, src1);
}
extern "C" __global__ AICORE void launchTEXTRACT_22(__gm__ int32_t* out, __gm__ int8_t* src0, __gm__ int8_t* src1)
{
    runTEXTRACT_UNALIGN<71, 188, 82, 0, 0, 0, 128, 256, 128, true, true>(out, src0, src1);
}
extern "C" __global__ AICORE void launchTEXTRACT_23(__gm__ int32_t* out, __gm__ int8_t* src0, __gm__ int8_t* src1)
{
    runTEXTRACT_UNALIGN<63, 112, 98, 32, 32, 32, 64, 128, 128, false, false>(out, src0, src1);
}
extern "C" __global__ AICORE void launchTEXTRACT_24(__gm__ int32_t* out, __gm__ int8_t* src0, __gm__ int8_t* src1)
{
    runTEXTRACT_UNALIGN<106, 125, 60, 32, 32, 32, 128, 128, 64, true, true>(out, src0, src1);
}

template <int32_t tilingKey>
void launchTEXTRACT(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream)
{
    if constexpr (tilingKey == 11) {
        launchTEXTRACT_11<<<1, nullptr, stream>>>((half*)out, (half*)src0, (half*)src1);
    } else if constexpr (tilingKey == 12) {
        launchTEXTRACT_12<<<1, nullptr, stream>>>((half*)out, (half*)src0, (half*)src1);
    } else if constexpr (tilingKey == 14) {
        launchTEXTRACT_14<<<1, nullptr, stream>>>((half*)out, (half*)src0, (half*)src1);
    } else if constexpr (tilingKey == 21) {
        launchTEXTRACT_21<<<1, nullptr, stream>>>((int32_t*)out, (int8_t*)src0, (int8_t*)src1);
    } else if constexpr (tilingKey == 22) {
        launchTEXTRACT_22<<<1, nullptr, stream>>>((int32_t*)out, (int8_t*)src0, (int8_t*)src1);
    } else if constexpr (tilingKey == 23) {
        launchTEXTRACT_23<<<1, nullptr, stream>>>((int32_t*)out, (int8_t*)src0, (int8_t*)src1);
    } else if constexpr (tilingKey == 24) {
        launchTEXTRACT_24<<<1, nullptr, stream>>>((int32_t*)out, (int8_t*)src0, (int8_t*)src1);
    }
}

template void launchTEXTRACT<11>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTEXTRACT<12>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTEXTRACT<14>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTEXTRACT<21>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTEXTRACT<22>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTEXTRACT<23>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTEXTRACT<24>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
