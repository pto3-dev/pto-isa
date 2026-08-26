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
#include <pto/common/constants.hpp>
#include <acl/acl.h>

using namespace std;
using namespace pto;

template <
    typename T, int dstTileRow, int dstTileCol, int srcTileRow, int srcTileCol, int validRow, int validCol,
    bool highPrecision = false>
__global__ AICORE void runTDIVS(__gm__ T* out, __gm__ T* src, T scalar)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    GlobalData srcGlobal(src, DynDim2Shape(validRow, validCol), DynDim2Stride(srcTileRow, srcTileCol));
    GlobalData dstGlobal(out, DynDim2Shape(validRow, validCol), DynDim2Stride(dstTileRow, dstTileCol));
    using srcTileData = Tile<TileType::Vec, T, srcTileRow, srcTileCol, BLayout::RowMajor, -1, -1>;
    using dstTileData = Tile<TileType::Vec, T, dstTileRow, dstTileCol, BLayout::RowMajor, -1, -1>;
    srcTileData srcTile(validRow, validCol);
    dstTileData dstTile(validRow, validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x26000);
    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    constexpr auto precisionType = highPrecision ? DivAlgorithm::HIGH_PRECISION : DivAlgorithm::DEFAULT;
    TDIVS<precisionType>(dstTile, srcTile, scalar);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int rows, int cols>
__global__ AICORE void runTDIVSWideInt64(__gm__ T* out, __gm__ T* src, T scalar)
{
    constexpr int tileRows = 1;
    constexpr int tileCols = 64;
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, 1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    using TileData = Tile<TileType::Vec, T, tileRows, tileCols, BLayout::RowMajor, -1, -1>;

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; col += tileCols) {
            int validCols = (col + tileCols <= cols) ? tileCols : (cols - col);
            int offset = row * cols + col;
            GlobalData srcGlobal(src + offset, DynDim2Shape(tileRows, validCols), DynDim2Stride(cols, 1));
            GlobalData dstGlobal(out + offset, DynDim2Shape(tileRows, validCols), DynDim2Stride(cols, 1));
            TileData srcTile(tileRows, validCols);
            TileData dstTile(tileRows, validCols);
            TASSIGN(srcTile, 0x0);
            TASSIGN(dstTile, 0x2000);
            Event<Op::TLOAD, Op::TDIVS> event0 = TLOAD(srcTile, srcGlobal);
            Event<Op::TDIVS, Op::TSTORE_VEC> event1 = TDIVS(dstTile, srcTile, scalar, event0);
            TSTORE(dstGlobal, dstTile, event1);
            pipe_barrier(PIPE_ALL);
        }
    }
}

template <
    typename T, int dstTileRow, int dstTileCol, int srcTileRow, int srcTileCol, int validRow, int validCol,
    bool highPrecision = false>
void LaunchTDivS(T* out, T* src, T scalar, void* stream)
{
    if constexpr (
        (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) && dstTileRow == srcTileRow &&
        dstTileCol == srcTileCol && dstTileRow == validRow && dstTileCol == validCol &&
        (validCol > 64 || validRow > 64)) {
        runTDIVSWideInt64<T, validRow, validCol><<<1, nullptr, stream>>>(out, src, scalar);
    } else {
        runTDIVS<T, dstTileRow, dstTileCol, srcTileRow, srcTileCol, validRow, validCol, highPrecision>
            <<<1, nullptr, stream>>>(out, src, scalar);
    }
}

template <
    int dstTileRow, int dstTileCol, int srcTileRow, int srcTileCol, int validRow, int validCol,
    bool highPrecision = false>
void LaunchTDivSHalf(aclFloat16* out, aclFloat16* src, aclFloat16 scalar, void* stream)
{
    runTDIVS<half, dstTileRow, dstTileCol, srcTileRow, srcTileCol, validRow, validCol, highPrecision>
        <<<1, nullptr, stream>>>((half*)out, (half*)src, *(half*)&scalar);
}

template void LaunchTDivS<float, 32, 128, 32, 64, 32, 64>(float* out, float* src, float scalar, void* stream);
template void LaunchTDivSHalf<63, 128, 63, 64, 63, 64>(
    aclFloat16* out, aclFloat16* src, aclFloat16 scalar, void* stream);
template void LaunchTDivS<int32_t, 31, 256, 31, 128, 31, 128>(int32_t* out, int32_t* src, int32_t scalar, void* stream);
template void LaunchTDivS<int16_t, 15, 192, 15, 192, 15, 192>(int16_t* out, int16_t* src, int16_t scalar, void* stream);
template void LaunchTDivS<float, 7, 512, 7, 448, 7, 448>(float* out, float* src, float scalar, void* stream);
template void LaunchTDivS<float, 256, 32, 256, 16, 256, 16>(float* out, float* src, float scalar, void* stream);
template void LaunchTDivS<float, 1, 32, 1, 16, 1, 16>(float* out, float* src, float scalar, void* stream);
template void LaunchTDivS<float, 2, 16, 2, 16, 2, 16, true>(float* out, float* src, float scalar, void* stream);
template void LaunchTDivSHalf<2, 32, 2, 32, 2, 32, true>(
    aclFloat16* out, aclFloat16* src, aclFloat16 scalar, void* stream);
template void LaunchTDivS<int64_t, 4, 16, 4, 16, 4, 16>(int64_t* out, int64_t* src, int64_t scalar, void* stream);
template void LaunchTDivS<uint64_t, 4, 16, 4, 16, 4, 16>(uint64_t* out, uint64_t* src, uint64_t scalar, void* stream);
template void LaunchTDivS<int64_t, 4, 64, 4, 64, 4, 64>(int64_t* out, int64_t* src, int64_t scalar, void* stream);
template void LaunchTDivS<uint64_t, 4, 64, 4, 64, 4, 64>(uint64_t* out, uint64_t* src, uint64_t scalar, void* stream);
template void LaunchTDivS<int64_t, 1, 16364, 1, 16364, 1, 16364>(
    int64_t* out, int64_t* src, int64_t scalar, void* stream);
template void LaunchTDivS<uint64_t, 1, 16364, 1, 16364, 1, 16364>(
    uint64_t* out, uint64_t* src, uint64_t scalar, void* stream);
template void LaunchTDivS<int64_t, 1, 32732, 1, 32732, 1, 32732>(
    int64_t* out, int64_t* src, int64_t scalar, void* stream);
template void LaunchTDivS<int64_t, 4091, 4, 4091, 4, 4091, 4>(int64_t* out, int64_t* src, int64_t scalar, void* stream);
template void LaunchTDivS<uint64_t, 4091, 4, 4091, 4, 4091, 4>(
    uint64_t* out, uint64_t* src, uint64_t scalar, void* stream);

template <typename T, int rows, int stride, int validCols>
__global__ AICORE void runTDIVSInplaceWideInt64(__gm__ T* out, __gm__ T* src, T scalar)
{
    constexpr int tileRows = 1;
    constexpr int tileCols = 64;
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, 1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    using TileData = Tile<TileType::Vec, T, tileRows, tileCols, BLayout::RowMajor, -1, -1>;

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < validCols; col += tileCols) {
            int curValidCols = (col + tileCols <= validCols) ? tileCols : (validCols - col);
            int offset = row * stride + col;
            GlobalData srcGlobal(src + offset, DynDim2Shape(tileRows, curValidCols), DynDim2Stride(stride, 1));
            GlobalData dstGlobal(out + offset, DynDim2Shape(tileRows, curValidCols), DynDim2Stride(stride, 1));
            TileData src0Tile(tileRows, curValidCols);
            TileData dstTile(tileRows, curValidCols);
            TASSIGN(src0Tile, 0x0);
            TASSIGN(dstTile, 0x0);
            Event<Op::TLOAD, Op::TDIVS> event0 = TLOAD(src0Tile, srcGlobal);
            Event<Op::TDIVS, Op::TSTORE_VEC> event1 = TDIVS(dstTile, src0Tile, scalar, event0);
            TSTORE(dstGlobal, dstTile, event1);
            pipe_barrier(PIPE_ALL);
        }
    }
}

template <typename T, int tileH, int tileW, int vRows, int vCols>
__global__ AICORE void runTDIVSInplace(__gm__ T* out, __gm__ T* src, T scalar)
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
    TDIVS<DivAlgorithm::DEFAULT>(dstTile, src0Tile, scalar);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int tileH, int tileW, int vRows, int vCols>
void LaunchTDivSInplace(T* out, T* src, T scalar, void* stream)
{
    if constexpr ((std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) && tileW > 32) {
        runTDIVSInplaceWideInt64<T, vRows, tileW, vCols><<<1, nullptr, stream>>>(out, src, scalar);
    } else {
        runTDIVSInplace<T, tileH, tileW, vRows, vCols><<<1, nullptr, stream>>>(out, src, scalar);
    }
}

template void LaunchTDivSInplace<int64_t, 4, 32, 4, 32>(int64_t* out, int64_t* src, int64_t scalar, void* stream);
template void LaunchTDivSInplace<uint64_t, 4, 32, 4, 32>(uint64_t* out, uint64_t* src, uint64_t scalar, void* stream);
template void LaunchTDivSInplace<int64_t, 1, 1024, 1, 1024>(int64_t* out, int64_t* src, int64_t scalar, void* stream);
template void LaunchTDivSInplace<int64_t, 4, 64, 4, 40>(int64_t* out, int64_t* src, int64_t scalar, void* stream);
template void LaunchTDivSInplace<int64_t, 1, 2048, 1, 2045>(int64_t* out, int64_t* src, int64_t scalar, void* stream);
