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

// Variant A: GM(ND) → UB(ND) → UB(NZ) → UB(NZ) → GM(NZ)
//            Tests UB-internal ND→NZ only (no L1 round-trip)
template <typename T, int Rows, int Cols>
AICORE void runTmovUb2Ub(__gm__ T* out, __gm__ T* src)
{
    using SrcGlobalData =
        GlobalTensor<T, pto::Shape<1, 1, 1, Rows, Cols>, pto::Stride<Rows * Cols, Rows * Cols, Rows * Cols, Cols, 1>>;
    SrcGlobalData srcGlobal(src);

    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));
    using DstShapeDim5 = pto::Shape<1, Cols / c0Size, Rows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using DstStridDim5 = pto::Stride<Rows * Cols, Rows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using DstGlobalData = GlobalTensor<T, DstShapeDim5, DstStridDim5, Layout::NZ>;
    DstGlobalData dstGlobal(out);

    using SrcTileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, Rows, Cols>;
    using DstTileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::ColMajor, Rows, Cols, SLayout::RowMajor>;

    SrcTileData srcTile;
    DstTileData dstTile;
    TASSIGN<0x0>(srcTile);
    TASSIGN<SrcTileData::Numel * sizeof(T)>(dstTile);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TMOV(dstTile, srcTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstTile);
}

// Variant B: GM(ND) → UB(ND) → UB(NZ) → L1(NZ) → UB(NZ) → GM(NZ)
//            Adds L1 round-trip: TMOV(UB→L1) + pto_copy_cbuf_to_ubuf(L1→UB)
//            Same as tmov_ub2l1 but with pto_copy_cbuf_to_ubuf for L1→UB
template <typename T, int Rows, int Cols>
AICORE void runTmovUb2L1Raw(__gm__ T* out, __gm__ T* src)
{
    using SrcGlobalData =
        GlobalTensor<T, pto::Shape<1, 1, 1, Rows, Cols>, pto::Stride<Rows * Cols, Rows * Cols, Rows * Cols, Cols, 1>>;
    SrcGlobalData srcGlobal(src);

    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));
    using DstShapeDim5 = pto::Shape<1, Cols / c0Size, Rows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using DstStridDim5 = pto::Stride<Rows * Cols, Rows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using DstGlobalData = GlobalTensor<T, DstShapeDim5, DstStridDim5, Layout::NZ>;
    DstGlobalData dstGlobal(out);

    using SrcTileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, Rows, Cols>;
    using TmpTileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::ColMajor, Rows, Cols, SLayout::RowMajor>;
    using DstTileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::ColMajor, Rows, Cols, SLayout::RowMajor>;
    using MatTileData = Tile<TileType::Mat, T, Rows, Cols, BLayout::ColMajor, Rows, Cols, SLayout::RowMajor>;

    SrcTileData srcTile;
    TmpTileData tmpTile;
    DstTileData dstTile;
    MatTileData matTile;
    TASSIGN<0x0>(srcTile);
    TASSIGN<SrcTileData::Numel * sizeof(T)>(tmpTile);
    TASSIGN<(SrcTileData::Numel + TmpTileData::Numel) * sizeof(T)>(dstTile);
    TASSIGN<0x0>(matTile);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TMOV(tmpTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TMOV(matTile, tmpTile);
    set_flag(PIPE_MTE3, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_FIX, EVENT_ID0);

    // L1 → UB: raw pto_copy_cbuf_to_ubuf (same as tmov_ub2l1's TMOVMat2Vec)
    {
        __ubuf__ T* dstAddr = dstTile.data();
        __cbuf__ T* srcAddr = matTile.data();
        uint16_t nBurst = 1;
        uint16_t lenBurst = Rows * Cols * sizeof(T) / 32;
        uint64_t fixpNzPara =
            static_cast<uint64_t>(1) | (static_cast<uint64_t>(lenBurst) << 16) | (static_cast<uint64_t>(1) << 32);
        set_fixp_nz_para(fixpNzPara);
        pto_copy_cbuf_to_ubuf(dstAddr, srcAddr, 0, nBurst, lenBurst, 0, 0);
        set_fixp_nz_para(0);
    }

    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstTile);
}

// Variant C: GM(ND) → UB(ND) → UB(NZ) → L1(NZ) → UB(NZ) → GM(NZ)
//            Adds L1 round-trip: TMOV(UB→L1) + TMOV(L1→UB via PTO TMOV instruction)
template <typename T, int Rows, int Cols>
AICORE void runTmovUb2L1Pto(__gm__ T* out, __gm__ T* src)
{
    using SrcGlobalData =
        GlobalTensor<T, pto::Shape<1, 1, 1, Rows, Cols>, pto::Stride<Rows * Cols, Rows * Cols, Rows * Cols, Cols, 1>>;
    SrcGlobalData srcGlobal(src);

    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));
    using DstShapeDim5 = pto::Shape<1, Cols / c0Size, Rows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using DstStridDim5 = pto::Stride<Rows * Cols, Rows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using DstGlobalData = GlobalTensor<T, DstShapeDim5, DstStridDim5, Layout::NZ>;
    DstGlobalData dstGlobal(out);

    using SrcTileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, Rows, Cols>;
    using TmpTileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::ColMajor, Rows, Cols, SLayout::RowMajor>;
    using DstTileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::ColMajor, Rows, Cols, SLayout::RowMajor>;
    using MatTileData = Tile<TileType::Mat, T, Rows, Cols, BLayout::ColMajor, Rows, Cols, SLayout::RowMajor>;

    SrcTileData srcTile;
    TmpTileData tmpTile;
    DstTileData dstTile;
    MatTileData matTile;
    TASSIGN<0x0>(srcTile);
    TASSIGN<SrcTileData::Numel * sizeof(T)>(tmpTile);
    TASSIGN<(SrcTileData::Numel + TmpTileData::Numel) * sizeof(T)>(dstTile);
    TASSIGN<0x0>(matTile);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TMOV(tmpTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TMOV(matTile, tmpTile);
    set_flag(PIPE_MTE3, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_FIX, EVENT_ID0);

    // L1 → UB: PTO TMOV instruction (TMovCbufToUb path)
    TMOV(dstTile, matTile);

    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstTile);
}

template <typename T, int Rows, int Cols>
__global__ AICORE void launchTmovUb2Ub(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTmovUb2Ub<T, Rows, Cols>(reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src));
}

template <typename T, int Rows, int Cols>
__global__ AICORE void launchTmovUb2L1RawKernel(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTmovUb2L1Raw<T, Rows, Cols>(reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src));
}

template <typename T, int Rows, int Cols>
__global__ AICORE void launchTmovUb2L1PtoKernel(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTmovUb2L1Pto<T, Rows, Cols>(reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src));
}

template <int32_t testKey>
void launchTmovUb2Ub(uint64_t* out, uint64_t* src, void* stream)
{
    if constexpr (testKey == 1) {
        launchTmovUb2Ub<half, 16, 32><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 2) {
        launchTmovUb2Ub<half, 64, 256><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 3) {
        launchTmovUb2Ub<int32_t, 48, 72><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 4) {
        launchTmovUb2Ub<int8_t, 32, 512><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 5) {
        launchTmovUb2Ub<int8_t, 64, 96><<<1, nullptr, stream>>>(out, src);
    }
}

template <int32_t testKey>
void launchTmovUb2L1Raw(uint64_t* out, uint64_t* src, void* stream)
{
    if constexpr (testKey == 1) {
        launchTmovUb2L1RawKernel<half, 16, 32><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 2) {
        launchTmovUb2L1RawKernel<half, 64, 256><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 3) {
        launchTmovUb2L1RawKernel<int32_t, 48, 72><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 4) {
        launchTmovUb2L1RawKernel<int8_t, 32, 512><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 5) {
        launchTmovUb2L1RawKernel<int8_t, 64, 96><<<1, nullptr, stream>>>(out, src);
    }
}

template <int32_t testKey>
void launchTmovUb2L1Pto(uint64_t* out, uint64_t* src, void* stream)
{
    if constexpr (testKey == 1) {
        launchTmovUb2L1PtoKernel<half, 16, 32><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 2) {
        launchTmovUb2L1PtoKernel<half, 64, 256><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 3) {
        launchTmovUb2L1PtoKernel<int32_t, 48, 72><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 4) {
        launchTmovUb2L1PtoKernel<int8_t, 32, 512><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 5) {
        launchTmovUb2L1PtoKernel<int8_t, 64, 96><<<1, nullptr, stream>>>(out, src);
    }
}

template void launchTmovUb2Ub<1>(uint64_t* out, uint64_t* src, void* stream);
template void launchTmovUb2Ub<2>(uint64_t* out, uint64_t* src, void* stream);
template void launchTmovUb2Ub<3>(uint64_t* out, uint64_t* src, void* stream);
template void launchTmovUb2Ub<4>(uint64_t* out, uint64_t* src, void* stream);
template void launchTmovUb2Ub<5>(uint64_t* out, uint64_t* src, void* stream);

template void launchTmovUb2L1Raw<1>(uint64_t* out, uint64_t* src, void* stream);
template void launchTmovUb2L1Raw<2>(uint64_t* out, uint64_t* src, void* stream);
template void launchTmovUb2L1Raw<3>(uint64_t* out, uint64_t* src, void* stream);
template void launchTmovUb2L1Raw<4>(uint64_t* out, uint64_t* src, void* stream);
template void launchTmovUb2L1Raw<5>(uint64_t* out, uint64_t* src, void* stream);

template void launchTmovUb2L1Pto<1>(uint64_t* out, uint64_t* src, void* stream);
template void launchTmovUb2L1Pto<2>(uint64_t* out, uint64_t* src, void* stream);
template void launchTmovUb2L1Pto<3>(uint64_t* out, uint64_t* src, void* stream);
template void launchTmovUb2L1Pto<4>(uint64_t* out, uint64_t* src, void* stream);
template void launchTmovUb2L1Pto<5>(uint64_t* out, uint64_t* src, void* stream);
