/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>

using namespace pto;

AICORE inline void test_copy_cbuf_to_ubuf(
    __ubuf__ void* dst, __cbuf__ void* src, uint8_t sid, uint16_t nBurst, uint16_t lenBurst, uint16_t srcGap,
    uint16_t dstGap)
{
    constexpr uint32_t CBUF_UB_BURST_UNIT = 32;
    uint16_t loop2SrcStride = static_cast<uint16_t>(lenBurst + srcGap);
    uint64_t fixpNzPara =
        static_cast<uint64_t>(1) | (static_cast<uint64_t>(loop2SrcStride) << 16) | (static_cast<uint64_t>(1) << 32);
    set_fixp_nz_para(fixpNzPara);
    __ubuf__ uint8_t* dstP = reinterpret_cast<__ubuf__ uint8_t*>(dst);
    __cbuf__ uint8_t* srcP = reinterpret_cast<__cbuf__ uint8_t*>(src);
    uint32_t srcStep = (lenBurst + srcGap) * CBUF_UB_BURST_UNIT;
    uint32_t dstStep = (lenBurst + dstGap) * CBUF_UB_BURST_UNIT;
    for (uint16_t i = 0; i < nBurst; ++i) {
        pto_copy_cbuf_to_ubuf(
            reinterpret_cast<__ubuf__ void*>(dstP + i * dstStep), reinterpret_cast<__cbuf__ void*>(srcP + i * srcStep),
            sid, 1, lenBurst, 0, 0);
    }
    set_fixp_nz_para(0);
}

template <
    typename TileMatAData, typename TileMatBData, typename RightTile, typename AccTile, typename GlobalDataSrc0,
    typename GlobalDataSrc1>
AICORE inline void LoadMatmulOnly(
    TileMatAData& aMatTile, TileMatBData& bMatTile, RightTile& bTile, AccTile& cTile, GlobalDataSrc0& src0Global,
    GlobalDataSrc1& src1Global)
{
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
}

template <typename AType, typename CType, int M, int K, int N>
__global__ AICORE void RunTInsertAcc2Mat(__gm__ CType* out, __gm__ AType* src0, __gm__ AType* src1)
{
    using GlobalDataSrc0 = GlobalTensor<AType, pto::Shape<1, 1, 1, M, K>, pto::Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<AType, pto::Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, N, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);

    using TileMatAData = Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, AType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using DstMatTile = Tile<TileType::Mat, CType, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    TileMatAData aMatTile;
    TileMatBData bMatTile;
    DstMatTile dstMatTile;
    TASSIGN<0x0>(aMatTile);
    TASSIGN<M * K * sizeof(AType)>(bMatTile);
    TASSIGN<(M * K + K * N) * sizeof(AType)>(dstMatTile);

    using RightTile = TileRight<AType, K, N, K, N>;
    using AccTile = TileAcc<CType, M, N, M, N>;
    RightTile bTile;
    AccTile cTile;
    TASSIGN<0x0>(bTile);
    TASSIGN<(M * K + K * N + M * N) * sizeof(AType)>(cTile);

    using DstVecTile = Tile<TileType::Vec, CType, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    DstVecTile dstVecTile;
    TASSIGN<0x0>(dstVecTile);

    static constexpr uint16_t sGRows = 16;
    static constexpr uint16_t sGCols = 512 / (sGRows * sizeof(CType));
    static constexpr uint16_t kGRows = (M + sGRows - 1) / sGRows;
    static constexpr uint16_t kGCols = (N + sGCols - 1) / sGCols;
    using ShapeDim5 = Shape<1, kGCols, kGRows, sGRows, sGCols>;
    using StridDim5 =
        pto::Stride<kGCols * kGRows * sGCols * sGRows, kGRows * sGCols * sGRows, sGCols * sGRows, sGCols, 1>;
    using NZOutputGlobalData = GlobalTensor<CType, ShapeDim5, StridDim5, Layout::NZ>;
    NZOutputGlobalData dstGlobal(out);

    LoadMatmulOnly(aMatTile, bMatTile, bTile, cTile, src0Global, src1Global);

    TINSERT(dstMatTile, cTile, static_cast<uint16_t>(0), static_cast<uint16_t>(0));

    set_flag(PIPE_FIX, PIPE_FIX, EVENT_ID1);
    wait_flag(PIPE_FIX, PIPE_FIX, EVENT_ID1);

    constexpr uint32_t c0Size = 512 / (16 * sizeof(CType));
    constexpr uint16_t burstLen = M * c0Size * sizeof(CType) / 32;
    constexpr uint16_t burstNum = N / c0Size;
    __ubuf__ CType* dstUbAddr = dstVecTile.data();
    __cbuf__ CType* srcMatAddr = dstMatTile.data();
    test_copy_cbuf_to_ubuf((__ubuf__ void*)dstUbAddr, (__cbuf__ void*)srcMatAddr, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstVecTile);
}

template <typename AType, typename CType, int M, int K, int N>
__global__ AICORE void RunTInsertMat2Mat(__gm__ CType* out, __gm__ AType* src0, __gm__ AType* src1)
{
    using GlobalDataSrc0 = GlobalTensor<AType, pto::Shape<1, 1, 1, M, K>, pto::Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<AType, pto::Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, N, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);

    using TileMatAData = Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, AType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using DstMatTile = Tile<TileType::Mat, CType, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    TileMatAData aMatTile;
    TileMatBData bMatTile;
    DstMatTile dstMatTile;
    TASSIGN<0x0>(aMatTile);
    TASSIGN<M * K * sizeof(AType)>(bMatTile);
    TASSIGN<(M * K + K * N) * sizeof(AType)>(dstMatTile);

    using RightTile = TileRight<AType, K, N, K, N>;
    using SrcMatTile = Tile<TileType::Mat, CType, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 1024>;
    RightTile bTile;
    SrcMatTile cMatTile;
    TASSIGN<0x0>(bTile);
    TASSIGN<(M * K + K * N + M * N) * sizeof(AType)>(cMatTile);

    using DstVecTile = Tile<TileType::Vec, CType, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    DstVecTile dstVecTile;
    TASSIGN<0x0>(dstVecTile);

    static constexpr uint16_t sGRows = 16;
    static constexpr uint16_t sGCols = 512 / (sGRows * sizeof(CType));
    static constexpr uint16_t kGRows = (M + sGRows - 1) / sGRows;
    static constexpr uint16_t kGCols = (N + sGCols - 1) / sGCols;
    using ShapeDim5 = Shape<1, kGCols, kGRows, sGRows, sGCols>;
    using StridDim5 =
        pto::Stride<kGCols * kGRows * sGCols * sGRows, kGRows * sGCols * sGRows, sGCols * sGRows, sGCols, 1>;
    using NZOutputGlobalData = GlobalTensor<CType, ShapeDim5, StridDim5, Layout::NZ>;
    NZOutputGlobalData dstGlobal(out);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    TMOV(bTile, bMatTile);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL(cMatTile, aMatTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TINSERT(dstMatTile, cMatTile, static_cast<uint16_t>(0), static_cast<uint16_t>(0));

    pipe_barrier(PIPE_FIX);

    constexpr uint32_t c0Size = 512 / (16 * sizeof(CType));
    constexpr uint16_t burstLen = M * c0Size * sizeof(CType) / 32;
    constexpr uint16_t burstNum = N / c0Size;
    __ubuf__ CType* dstUbAddr = dstVecTile.data();
    __cbuf__ CType* srcMatAddr = dstMatTile.data();
    test_copy_cbuf_to_ubuf((__ubuf__ void*)dstUbAddr, (__cbuf__ void*)srcMatAddr, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstVecTile);
}

template <typename T, int M, int N, bool useFixpNzPara>
__global__ AICORE void RunCbufToCbufTest(__gm__ T* out, __gm__ T* src)
{
    constexpr int c0 = 32 / sizeof(T);
    using SrcShape = pto::Shape<1, N / c0, M / 16, 16, c0>;
    using SrcStride = pto::Stride<M * N, 16 * c0, 16 * c0, c0, 1>;
    using SrcGlobal = GlobalTensor<T, SrcShape, SrcStride, Layout::NZ>;
    using DstGlobal = GlobalTensor<T, SrcShape, SrcStride, Layout::NZ>;
    SrcGlobal srcGlobal(src);
    DstGlobal dstGlobal(out);

    using SrcMatTile = Tile<TileType::Mat, T, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    using DstMatTile = Tile<TileType::Mat, T, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    SrcMatTile srcMatTile;
    DstMatTile dstMatTile;
    TASSIGN<0x0>(srcMatTile);
    TASSIGN<M * N * sizeof(T)>(dstMatTile);

    using DstVecTile = Tile<TileType::Vec, T, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    DstVecTile dstVecTile;
    TASSIGN<0x0>(dstVecTile);

    TLOAD(srcMatTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);

    __cbuf__ T* srcAddr = srcMatTile.data();
    __cbuf__ T* dstAddr = dstMatTile.data();
    uint32_t totalBytes = M * N * sizeof(T);
    uint32_t totalBlocks = totalBytes / 32;
    if constexpr (useFixpNzPara) {
        SetFixpNzPara(1, static_cast<uint16_t>(totalBlocks), 1, 0);
    }
    fix_cbuf_to_cbuf(
        dstAddr, srcAddr, static_cast<uint64_t>(totalBytes), totalBytes, fixp_trans_mode_t::NORMAL_DMA, 0, 1);
    if constexpr (useFixpNzPara) {
        SetFixpNzPara(0, 0, 0, 0);
    }

    pipe_barrier(PIPE_FIX);

    constexpr uint32_t c0Size = 512 / (16 * sizeof(T));
    constexpr uint16_t burstLen = M * c0Size * sizeof(T) / 32;
    constexpr uint16_t burstNum = N / c0Size;
    __ubuf__ T* dstUbAddr = dstVecTile.data();
    __cbuf__ T* srcCbAddr = dstMatTile.data();
    test_copy_cbuf_to_ubuf((__ubuf__ void*)dstUbAddr, (__cbuf__ void*)srcCbAddr, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstVecTile);
}

template <typename T, int M, int N, bool withTinsert>
__global__ AICORE void RunTInsertMat2MatLoad(__gm__ T* out, __gm__ T* src)
{
    constexpr int c0 = 32 / sizeof(T);
    using Shape5 = pto::Shape<1, N / c0, M / 16, 16, c0>;
    using Stride5 = pto::Stride<M * N, 16 * c0, 16 * c0, c0, 1>;
    using GlobalData = GlobalTensor<T, Shape5, Stride5, Layout::NZ>;
    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    using SrcMatTile = Tile<TileType::Mat, T, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    using DstMatTile = Tile<TileType::Mat, T, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    SrcMatTile srcMatTile;
    DstMatTile dstMatTile;
    TASSIGN<0x0>(srcMatTile);
    TASSIGN<M * N * sizeof(T)>(dstMatTile);

    using DstVecTile = Tile<TileType::Vec, T, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    DstVecTile dstVecTile;
    TASSIGN<0x0>(dstVecTile);

    TLOAD(srcMatTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    constexpr uint32_t c0Size = 512 / (16 * sizeof(T));
    constexpr uint16_t burstLen = M * c0Size * sizeof(T) / 32;
    constexpr uint16_t burstNum = N / c0Size;
    __ubuf__ T* dstUbAddr = dstVecTile.data();

    if constexpr (withTinsert) {
        TINSERT(dstMatTile, srcMatTile, static_cast<uint16_t>(0), static_cast<uint16_t>(0));
        pipe_barrier(PIPE_FIX);
        test_copy_cbuf_to_ubuf(
            (__ubuf__ void*)dstUbAddr, (__cbuf__ void*)dstMatTile.data(), 0, burstNum, burstLen, 0, 0);
    } else {
        test_copy_cbuf_to_ubuf(
            (__ubuf__ void*)dstUbAddr, (__cbuf__ void*)srcMatTile.data(), 0, burstNum, burstLen, 0, 0);
    }

    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstVecTile);
}

template <int32_t testKey>
void launchTInsertAcc2Mat(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream)
{
    if constexpr (testKey == 1) {
        RunTInsertAcc2Mat<half, half, 16, 16, 16><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
    } else if constexpr (testKey == 2) {
        RunTInsertAcc2Mat<half, half, 32, 32, 32><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
    } else if constexpr (testKey == 3) {
        RunTInsertMat2Mat<half, half, 16, 16, 16><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
    } else if constexpr (testKey == 4) {
        RunTInsertMat2Mat<half, half, 32, 32, 32><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
    } else if constexpr (testKey == 5) {
        RunCbufToCbufTest<half, 16, 16, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0));
    } else if constexpr (testKey == 6) {
        RunCbufToCbufTest<half, 16, 16, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0));
    } else if constexpr (testKey == 7) {
        RunTInsertMat2MatLoad<half, 16, 16, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0));
    } else if constexpr (testKey == 8) {
        RunTInsertMat2MatLoad<half, 16, 16, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0));
    }
}

template void launchTInsertAcc2Mat<1>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTInsertAcc2Mat<2>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTInsertAcc2Mat<3>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTInsertAcc2Mat<4>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTInsertAcc2Mat<5>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTInsertAcc2Mat<6>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTInsertAcc2Mat<7>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTInsertAcc2Mat<8>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
