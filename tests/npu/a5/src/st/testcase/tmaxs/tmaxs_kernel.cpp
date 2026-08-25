/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <type_traits>
#include <pto/pto-inst.hpp>
#include "acl/acl.h"

#define PAD_VALUE_NULL (-100)
#define PAD_VALUE_MAX (1)

using namespace pto;

#ifdef __CCE_AICORE__
template <typename T, int dstRow, int dstCol, int srcRow, int srcCol, int kVRows_, int kVCols_, int kPadValue_>
struct GenericDataSelector {};

template <typename T, int dstRow, int dstCol, int srcRow, int srcCol, int kVRows_, int kVCols_>
struct GenericDataSelector<T, dstRow, dstCol, srcRow, srcCol, kVRows_, kVCols_, PAD_VALUE_NULL> {
    using srcTileType = Tile<
        TileType::Vec, T, srcRow, srcCol, BLayout::RowMajor, kVRows_, kVCols_, SLayout::NoneBox, 512, PadValue::Null>;
    using dstTileType = Tile<
        TileType::Vec, T, dstRow, dstCol, BLayout::RowMajor, kVRows_, kVCols_, SLayout::NoneBox, 512, PadValue::Null>;
};

template <typename T, int dstRow, int dstCol, int srcRow, int srcCol, int kVRows_, int kVCols_>
struct GenericDataSelector<T, dstRow, dstCol, srcRow, srcCol, kVRows_, kVCols_, PAD_VALUE_MAX> {
    using srcTileType = Tile<
        TileType::Vec, T, srcRow, srcCol, BLayout::RowMajor, kVRows_, kVCols_, SLayout::NoneBox, 512, PadValue::Max>;
    using dstTileType = Tile<
        TileType::Vec, T, dstRow, dstCol, BLayout::RowMajor, kVRows_, kVCols_, SLayout::NoneBox, 512, PadValue::Max>;
};
#endif

template <typename T, int dstRow, int dstCol, int srcRow, int srcCol, int kVRows_, int kVCols_, int kPadValue_>
__global__ AICORE void runTMAXS(__gm__ T __out__* out, __gm__ T __in__* src0, __gm__ T __in__* scalar)
{
    using GDS = GenericDataSelector<T, dstRow, dstCol, srcRow, srcCol, kVRows_, kVCols_, kPadValue_>;

    using srcDynShapeDim5 = Shape<1, 1, 1, kVRows_, kVCols_>;
    using srcDynStridDim5 = pto::Stride<1, 1, 1, srcCol, 1>;
    using srcGlobalType = GlobalTensor<T, srcDynShapeDim5, srcDynStridDim5>;

    using dstDynShapeDim5 = Shape<1, 1, 1, kVRows_, kVCols_>;
    using dstDynStridDim5 = pto::Stride<1, 1, 1, dstCol, 1>;
    using dstGlobalType = GlobalTensor<T, dstDynShapeDim5, dstDynStridDim5>;

    using dstTileData = typename GDS::dstTileType;
    using srcTileData = typename GDS::srcTileType;
    srcTileData src0Tile;
    dstTileData dstTile;
    TASSIGN(src0Tile, 0x0 + 0x400 * block_idx);
    TASSIGN(dstTile, 0x8000 + 0x400 * block_idx);

    int offset = 0;
    srcGlobalType src0Global(src0 + offset);
    dstGlobalType dstGlobal(out + offset);

    Event<Op::TLOAD, Op::TMAXS> event0;
    Event<Op::TMAXS, Op::TSTORE_VEC> event1;

    event0 = TLOAD(src0Tile, src0Global);
    event1 = TMAXS(dstTile, src0Tile, scalar[0], event0);
    TSTORE(dstGlobal, dstTile, event1);
    out = dstGlobal.data();
}

template <typename T, int cols>
__global__ AICORE void runTMAXSWideInt64(__gm__ T __out__* out, __gm__ T __in__* src0, __gm__ T __in__* scalar)
{
    constexpr int tileRows = 1;
    constexpr int tileCols = 64;

    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using DynStridDim5 = pto::Stride<1, 1, 1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, tileRows, tileCols, BLayout::RowMajor, -1, -1>;

    for (int col = 0; col < cols; col += tileCols) {
        int validCols = (col + tileCols <= cols) ? tileCols : (cols - col);
        GlobalData src0Global(src0 + col, DynShapeDim5(tileRows, validCols), DynStridDim5(cols, 1));
        GlobalData dstGlobal(out + col, DynShapeDim5(tileRows, validCols), DynStridDim5(cols, 1));
        TileData src0Tile(tileRows, validCols);
        TileData dstTile(tileRows, validCols);
        TASSIGN(src0Tile, 0x0);
        TASSIGN(dstTile, 0x2000);

        Event<Op::TLOAD, Op::TMAXS> event0 = TLOAD(src0Tile, src0Global);
        Event<Op::TMAXS, Op::TSTORE_VEC> event1 = TMAXS(dstTile, src0Tile, scalar[0], event0);
        TSTORE(dstGlobal, dstTile, event1);
        pipe_barrier(PIPE_ALL);
    }
}

template <typename T, int dstRow, int dstCol, int srcRow, int srcCol, int kVRows_, int kVCols_, int kPadValue_>
void LaunchTMaxs(T* out, T* src0, T* scalar, void* stream)
{
    constexpr int wideTileCols = 64;
    if constexpr (std::is_same_v<T, aclFloat16> && kPadValue_ == PAD_VALUE_MAX) {
        runTMAXS<half, dstRow, dstCol, srcRow, srcCol, kVRows_, kVCols_, kPadValue_>
            <<<1, nullptr, stream>>>((half*)(out), (half*)(src0), (half*)(scalar));
    } else if constexpr (
        (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) && dstRow == 1 && srcRow == 1 && dstCol == srcCol &&
        dstCol == kVCols_ && kVRows_ == 1 && kVCols_ > wideTileCols && kPadValue_ == PAD_VALUE_NULL) {
        runTMAXSWideInt64<T, kVCols_><<<1, nullptr, stream>>>(out, src0, scalar);
    } else {
        runTMAXS<T, dstRow, dstCol, srcRow, srcCol, kVRows_, kVCols_, kPadValue_>
            <<<1, nullptr, stream>>>(out, src0, scalar);
    }
}

template void LaunchTMaxs<float, 64, 64, 32, 32, 32, 32, PAD_VALUE_NULL>(
    float* out, float* src0, float* scalar, void* stream);
template void LaunchTMaxs<float, 128, 128, 64, 64, 64, 64, PAD_VALUE_NULL>(
    float* out, float* src0, float* scalar, void* stream);

template void LaunchTMaxs<float, 60, 128, 64, 64, 60, 60, PAD_VALUE_MAX>(
    float* out, float* src0, float* scalar, void* stream);
template void LaunchTMaxs<float, 16, 200, 20, 512, 16, 200, PAD_VALUE_MAX>(
    float* out, float* src0, float* scalar, void* stream);
template void LaunchTMaxs<float, 1, 3600, 2, 4096, 1, 3600, PAD_VALUE_MAX>(
    float* out, float* src0, float* scalar, void* stream);
template void LaunchTMaxs<aclFloat16, 16, 256, 20, 224, 16, 200, PAD_VALUE_MAX>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* scalar, void* stream);
template void LaunchTMaxs<aclFloat16, 1, 256, 1, 224, 1, 200, PAD_VALUE_MAX>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* scalar, void* stream);

template void LaunchTMaxs<int32_t, 32, 32, 32, 32, 32, 32, PAD_VALUE_NULL>(
    int32_t* out, int32_t* src0, int32_t* scalar, void* stream);
template void LaunchTMaxs<uint32_t, 32, 32, 32, 32, 32, 32, PAD_VALUE_NULL>(
    uint32_t* out, uint32_t* src0, uint32_t* scalar, void* stream);
template void LaunchTMaxs<int16_t, 32, 128, 32, 128, 32, 128, PAD_VALUE_NULL>(
    int16_t* out, int16_t* src0, int16_t* scalar, void* stream);
template void LaunchTMaxs<uint16_t, 32, 128, 32, 128, 32, 128, PAD_VALUE_NULL>(
    uint16_t* out, uint16_t* src0, uint16_t* scalar, void* stream);

template void LaunchTMaxs<int8_t, 32, 128, 32, 128, 32, 128, PAD_VALUE_NULL>(
    int8_t* out, int8_t* src0, int8_t* scalar, void* stream);
template void LaunchTMaxs<uint8_t, 32, 128, 32, 128, 32, 128, PAD_VALUE_NULL>(
    uint8_t* out, uint8_t* src0, uint8_t* scalar, void* stream);
template void LaunchTMaxs<int64_t, 4, 16, 4, 16, 4, 16, PAD_VALUE_NULL>(
    int64_t* out, int64_t* src0, int64_t* scalar, void* stream);
template void LaunchTMaxs<uint64_t, 4, 16, 4, 16, 4, 16, PAD_VALUE_NULL>(
    uint64_t* out, uint64_t* src0, uint64_t* scalar, void* stream);
template void LaunchTMaxs<int64_t, 1, 16364, 1, 16364, 1, 16364, PAD_VALUE_NULL>(
    int64_t* out, int64_t* src0, int64_t* scalar, void* stream);
template void LaunchTMaxs<uint64_t, 1, 16364, 1, 16364, 1, 16364, PAD_VALUE_NULL>(
    uint64_t* out, uint64_t* src0, uint64_t* scalar, void* stream);
template void LaunchTMaxs<int64_t, 1, 16368, 1, 16368, 1, 16368, PAD_VALUE_NULL>(
    int64_t* out, int64_t* src0, int64_t* scalar, void* stream);
template void LaunchTMaxs<uint64_t, 1, 16368, 1, 16368, 1, 16368, PAD_VALUE_NULL>(
    uint64_t* out, uint64_t* src0, uint64_t* scalar, void* stream);

template <typename T, int rows, int stride, int validCols>
__global__ AICORE void runTMAXSInplaceWideInt64(__gm__ T* out, __gm__ T* src, T scalar)
{
    constexpr int tileRows = 1;
    constexpr int tileCols = 64;
    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using DynStridDim5 = pto::Stride<1, 1, 1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, tileRows, tileCols, BLayout::RowMajor, -1, -1>;

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < validCols; col += tileCols) {
            int curValidCols = (col + tileCols <= validCols) ? tileCols : (validCols - col);
            int offset = row * stride + col;
            GlobalData srcGlobal(src + offset, DynShapeDim5(tileRows, curValidCols), DynStridDim5(stride, 1));
            GlobalData dstGlobal(out + offset, DynShapeDim5(tileRows, curValidCols), DynStridDim5(stride, 1));
            TileData src0Tile(tileRows, curValidCols);
            TileData dstTile(tileRows, curValidCols);
            TASSIGN(src0Tile, 0x0);
            TASSIGN(dstTile, 0x0);
            Event<Op::TLOAD, Op::TMAXS> event0 = TLOAD(src0Tile, srcGlobal);
            Event<Op::TMAXS, Op::TSTORE_VEC> event1 = TMAXS(dstTile, src0Tile, scalar, event0);
            TSTORE(dstGlobal, dstTile, event1);
            pipe_barrier(PIPE_ALL);
        }
    }
}

template <typename T, int tileH, int tileW, int vRows, int vCols>
__global__ AICORE void runTMAXSInplace(__gm__ T* out, __gm__ T* src, T scalar)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = pto::Stride<1, 1, 1, tileW, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, tileH, tileW, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(dstTile, 0x0);
    GlobalData dstGlobal(out);
    GlobalData srcGlobal(src);
    TLOAD(src0Tile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TMAXS(dstTile, src0Tile, scalar);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int tileH, int tileW, int vRows, int vCols>
void LaunchTMaxsInplace(T* out, T* src, T scalar, void* stream)
{
    if constexpr ((std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) && tileW > 32) {
        runTMAXSInplaceWideInt64<T, vRows, tileW, vCols><<<1, nullptr, stream>>>(out, src, scalar);
    } else {
        runTMAXSInplace<T, tileH, tileW, vRows, vCols><<<1, nullptr, stream>>>(out, src, scalar);
    }
}

template void LaunchTMaxsInplace<int64_t, 4, 32, 4, 32>(int64_t* out, int64_t* src, int64_t scalar, void* stream);
template void LaunchTMaxsInplace<uint64_t, 4, 32, 4, 32>(uint64_t* out, uint64_t* src, uint64_t scalar, void* stream);
template void LaunchTMaxsInplace<int64_t, 1, 1024, 1, 1024>(int64_t* out, int64_t* src, int64_t scalar, void* stream);
template void LaunchTMaxsInplace<int64_t, 4, 64, 4, 40>(int64_t* out, int64_t* src, int64_t scalar, void* stream);
template void LaunchTMaxsInplace<int64_t, 1, 2048, 1, 2045>(int64_t* out, int64_t* src, int64_t scalar, void* stream);
