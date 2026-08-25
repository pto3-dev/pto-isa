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
#include <pto/common/constants.hpp>
#include "acl/acl.h"

using namespace pto;

template <typename T, int kTRows_, int kTCols_, int kGRows_, int kGCols_>
__global__ AICORE void runTMax(__gm__ T __out__* out, __gm__ T __in__* src0, __gm__ T __in__* src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(kTRows_, kTCols_);
    TileData src1Tile(kTRows_, kTCols_);
    TileData dstTile(kTRows_, kTCols_);
    TASSIGN(src0Tile, 0x0 + 0x400 * block_idx);
    TASSIGN(src1Tile, 0x4000 + 0x400 * block_idx);
    TASSIGN(dstTile, 0x8000 + 0x400 * block_idx);

    int offset = (block_idx / 4) * (64 * 16) + (block_idx % 4) * 16;
    GlobalData src0Global(src0 + offset);
    GlobalData src1Global(src1 + offset);
    GlobalData dstGlobal(out + offset);

    Event<Op::TLOAD, Op::TMAX> event0;
    Event<Op::TMAX, Op::TSTORE_VEC> event1;

    TLOAD(src0Tile, src0Global);
    event0 = TLOAD(src1Tile, src1Global);
    event1 = TMAX(dstTile, src0Tile, src1Tile, event0);
    TSTORE(dstGlobal, dstTile, event1);
    out = dstGlobal.data();
}

template <
    typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
    int vCols>
__global__ AICORE void runTMax(__gm__ T __out__* out, __gm__ T __in__* src0, __gm__ T __in__* src1)
{
    using DynShape = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    GlobalData dstGlobal(
        out, pto::Shape(1, 1, 1, vRows, vCols),
        pto::Stride(dstTileH * dstTileW, dstTileH * dstTileW, dstTileH * dstTileW, dstTileW, 1));
    GlobalData src0Global(
        src0, pto::Shape(1, 1, 1, vRows, vCols),
        pto::Stride(src0TileH * src0TileW, src0TileH * src0TileW, src0TileH * src0TileW, src0TileW, 1));
    GlobalData src1Global(
        src1, pto::Shape(1, 1, 1, vRows, vCols),
        pto::Stride(src1TileH * src1TileW, src1TileH * src1TileW, src1TileH * src1TileW, src1TileW, 1));

    using TileDataDst = Tile<TileType::Vec, T, dstTileH, dstTileW, BLayout::RowMajor, -1, -1>;
    using TileDataSrc0 = Tile<TileType::Vec, T, src0TileH, src0TileW, BLayout::RowMajor, -1, -1>;
    using TileDataSrc1 = Tile<TileType::Vec, T, src1TileH, src1TileW, BLayout::RowMajor, -1, -1>;
    TileDataDst dstTile(vRows, vCols);
    TileDataSrc0 src0Tile(vRows, vCols);
    TileDataSrc1 src1Tile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    Event<Op::TLOAD, Op::TMAX> event0;
    Event<Op::TMAX, Op::TSTORE_VEC> event1;

    TLOAD(src0Tile, src0Global);
    event0 = TLOAD(src1Tile, src1Global);
    event1 = TMAX<TileDataDst, TileDataSrc0, TileDataSrc1>(dstTile, src0Tile, src1Tile, event0);
    TSTORE(dstGlobal, dstTile, event1);
    out = dstGlobal.data();
}

template <typename T, int cols>
__global__ AICORE void runTMaxWideInt64(__gm__ T __out__* out, __gm__ T __in__* src0, __gm__ T __in__* src1)
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
        GlobalData src1Global(src1 + col, DynShapeDim5(tileRows, validCols), DynStridDim5(cols, 1));
        GlobalData dstGlobal(out + col, DynShapeDim5(tileRows, validCols), DynStridDim5(cols, 1));
        TileData src0Tile(tileRows, validCols);
        TileData src1Tile(tileRows, validCols);
        TileData dstTile(tileRows, validCols);
        TASSIGN(src0Tile, 0x0);
        TASSIGN(src1Tile, 0x2000);
        TASSIGN(dstTile, 0x4000);

        Event<Op::TLOAD, Op::TMAX> event0;
        Event<Op::TMAX, Op::TSTORE_VEC> event1;
        TLOAD(src0Tile, src0Global);
        event0 = TLOAD(src1Tile, src1Global);
        event1 = TMAX(dstTile, src0Tile, src1Tile, event0);
        TSTORE(dstGlobal, dstTile, event1);
        pipe_barrier(PIPE_ALL);
    }
}

template <
    typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
    int vCols, bool sameTile>
void LaunchTMax(T* out, T* src0, T* src1, void* stream)
{
    constexpr int wideTileCols = 64;
    if constexpr (
        (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) && dstTileH == 1 && src0TileH == 1 &&
        src1TileH == 1 && dstTileW == src0TileW && dstTileW == src1TileW && dstTileW == vCols && vRows == 1 &&
        vCols > wideTileCols) {
        runTMaxWideInt64<T, vCols><<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (sameTile) {
        runTMax<T, dstTileH, dstTileW, vRows, vCols><<<1, nullptr, stream>>>(out, src0, src1);
    } else {
        runTMax<T, dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW, vRows, vCols>
            <<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template <
    int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows, int vCols,
    bool sameTile>
void LaunchTMaxHalf(aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream)
{
    if constexpr (sameTile) {
        runTMax<half, dstTileH, dstTileW, vRows, vCols>
            <<<1, nullptr, stream>>>((half*)(out), (half*)(src0), (half*)(src1));
    } else {
        runTMax<half, dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW, vRows, vCols>
            <<<1, nullptr, stream>>>((half*)(out), (half*)(src0), (half*)(src1));
    }
}

template void LaunchTMax<float, 64, 64, 64, 64, 64, 64, 64, 64, true>(
    float* out, float* src0, float* src1, void* stream);
template void LaunchTMax<int32_t, 64, 64, 64, 64, 64, 64, 64, 64, true>(
    int32_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTMax<int16_t, 64, 64, 64, 64, 64, 64, 64, 64, true>(
    int16_t* out, int16_t* src0, int16_t* src1, void* stream);
template void LaunchTMaxHalf<16, 256, 16, 256, 16, 256, 16, 256, true>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTMaxHalf<16, 64, 16, 128, 16, 128, 16, 64, false>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTMax<float, 16, 32, 16, 64, 16, 32, 16, 32, false>(
    float* out, float* src0, float* src1, void* stream);
template void LaunchTMax<int16_t, 32, 128, 32, 128, 32, 256, 32, 128, false>(
    int16_t* out, int16_t* src0, int16_t* src1, void* stream);
template void LaunchTMax<int32_t, 16, 32, 16, 64, 16, 32, 16, 32, false>(
    int32_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTMaxHalf<16, 64, 16, 128, 16, 128, 16, 63, false>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTMax<float, 16, 32, 16, 64, 16, 32, 16, 31, false>(
    float* out, float* src0, float* src1, void* stream);
template void LaunchTMax<int16_t, 32, 128, 32, 128, 32, 256, 32, 127, false>(
    int16_t* out, int16_t* src0, int16_t* src1, void* stream);
template void LaunchTMax<int32_t, 16, 32, 16, 64, 16, 32, 16, 31, false>(
    int32_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTMax<int64_t, 4, 16, 4, 16, 4, 16, 4, 16, true>(
    int64_t* out, int64_t* src0, int64_t* src1, void* stream);
template void LaunchTMax<uint64_t, 4, 16, 4, 16, 4, 16, 4, 16, true>(
    uint64_t* out, uint64_t* src0, uint64_t* src1, void* stream);
template void LaunchTMax<int64_t, 1, 16364, 1, 16364, 1, 16364, 1, 16364, true>(
    int64_t* out, int64_t* src0, int64_t* src1, void* stream);
template void LaunchTMax<uint64_t, 1, 16364, 1, 16364, 1, 16364, 1, 16364, true>(
    uint64_t* out, uint64_t* src0, uint64_t* src1, void* stream);

template <typename T, int tileH, int tileW, int vRows, int vCols>
__global__ AICORE void runTMaxInplace(__gm__ T* out, __gm__ T* src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = pto::Stride<tileH * tileW, tileH * tileW, tileH * tileW, tileW, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, tileH, tileW, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TileData src1Tile(vRows, vCols);
    constexpr unsigned tileBytes = tileH * tileW * sizeof(T);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(dstTile, 0x0);
    TASSIGN(src1Tile, tileBytes);

    GlobalData src0Global((__gm__ T*)out);
    GlobalData dstGlobal(out);
    GlobalData src1Global(src1);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TMAX(dstTile, src0Tile, src1Tile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int tileH, int tileW, int vRows, int vCols>
void LaunchTMaxInplace(T* out, T* src1, void* stream)
{
    runTMaxInplace<T, tileH, tileW, vRows, vCols><<<1, nullptr, stream>>>(out, src1);
}

template void LaunchTMaxInplace<int64_t, 4, 32, 4, 32>(int64_t* out, int64_t* src1, void* stream);
template void LaunchTMaxInplace<uint64_t, 4, 32, 4, 32>(uint64_t* out, uint64_t* src1, void* stream);
template void LaunchTMaxInplace<int64_t, 1, 1024, 1, 1024>(int64_t* out, int64_t* src1, void* stream);
template void LaunchTMaxInplace<int64_t, 1, 2048, 1, 2045>(int64_t* out, int64_t* src1, void* stream);
template void LaunchTMaxInplace<int64_t, 4, 64, 4, 40>(int64_t* out, int64_t* src1, void* stream);
