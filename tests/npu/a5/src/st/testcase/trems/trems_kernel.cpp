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
    bool highPrecision>
__global__ AICORE void runTRemS(__gm__ T* out, __gm__ T* src, T scalar)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    using srcTileData = Tile<TileType::Vec, T, srcTileRow, srcTileCol, BLayout::RowMajor, -1, -1>;
    using dstTileData = Tile<TileType::Vec, T, dstTileRow, dstTileCol, BLayout::RowMajor, -1, -1>;
    using tmpTileData = Tile<TileType::Vec, T, 1, dstTileCol, BLayout::RowMajor, -1, -1>;

    GlobalData dstGlobal(out, DynDim2Shape(validRow, validCol), DynDim2Stride(dstTileRow, dstTileCol));
    GlobalData srcGlobal(src, DynDim2Shape(validRow, validCol), DynDim2Stride(srcTileRow, srcTileCol));
    dstTileData dstTile(validRow, validCol);
    srcTileData srcTile(validRow, validCol);
    tmpTileData tmpTile(1, validCol);

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, srcTileRow * srcTileCol * sizeof(T));
    TASSIGN(tmpTile, srcTileRow * srcTileCol * sizeof(T) + dstTileRow * dstTileCol * sizeof(T));
    constexpr auto precisionType = highPrecision ? RemSAlgorithm::HIGH_PRECISION : RemSAlgorithm::DEFAULT;

    Event<Op::TLOAD, Op::TREMS> event0;
    Event<Op::TREMS, Op::TSTORE_VEC> event1;

    event0 = TLOAD(srcTile, srcGlobal);
    event1 = TREMS<precisionType>(dstTile, srcTile, scalar, tmpTile, event0);
    TSTORE(dstGlobal, dstTile, event1);
    out = dstGlobal.data();
}

template <typename T, int rows, int cols>
__global__ AICORE void runTRemSWideInt64(__gm__ T* out, __gm__ T* src, T scalar)
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
            TileData tmpTile(tileRows, validCols);
            TASSIGN(srcTile, 0x0);
            TASSIGN(dstTile, 0x2000);
            TASSIGN(tmpTile, 0x4000);
            Event<Op::TLOAD, Op::TREMS> event0 = TLOAD(srcTile, srcGlobal);
            Event<Op::TREMS, Op::TSTORE_VEC> event1 = TREMS(dstTile, srcTile, scalar, tmpTile, event0);
            TSTORE(dstGlobal, dstTile, event1);
            pipe_barrier(PIPE_ALL);
        }
    }
}

template <
    typename T, int dstTileRow, int dstTileCol, int srcTileRow, int srcTileCol, int validRow, int validCol,
    bool highPrecision = false>
void LaunchTRemS(T* out, T* src, T scalar, void* stream)
{
    if constexpr (
        (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) && dstTileRow == srcTileRow &&
        dstTileCol == srcTileCol && dstTileRow == validRow && dstTileCol == validCol &&
        (validCol > 64 || validRow > 64)) {
        runTRemSWideInt64<T, validRow, validCol><<<1, nullptr, stream>>>(out, src, scalar);
    } else {
        runTRemS<T, dstTileRow, dstTileCol, srcTileRow, srcTileCol, validRow, validCol, highPrecision>
            <<<1, nullptr, stream>>>(out, src, scalar);
    }
}

template <
    int dstTileRow, int dstTileCol, int srcTileRow, int srcTileCol, int validRow, int validCol,
    bool highPrecision = false>
void LaunchTRemSHalf(aclFloat16* out, aclFloat16* src, aclFloat16 scalar, void* stream)
{
    runTRemS<half, dstTileRow, dstTileCol, srcTileRow, srcTileCol, validRow, validCol, highPrecision>
        <<<1, nullptr, stream>>>((half*)out, (half*)src, *(half*)&scalar);
}

template void LaunchTRemS<float, 32, 128, 32, 128, 32, 64>(float* out, float* src, float scalar, void* stream);
template void LaunchTRemSHalf<63, 128, 63, 128, 63, 64>(
    aclFloat16* out, aclFloat16* src, aclFloat16 scalar, void* stream);
template void LaunchTRemS<int32_t, 31, 256, 31, 256, 31, 128>(int32_t* out, int32_t* src, int32_t scalar, void* stream);
template void LaunchTRemS<int16_t, 15, 192, 15, 192, 15, 192>(int16_t* out, int16_t* src, int16_t scalar, void* stream);
template void LaunchTRemS<float, 7, 512, 7, 512, 7, 448>(float* out, float* src, float scalar, void* stream);
template void LaunchTRemS<float, 256, 32, 256, 32, 256, 31>(float* out, float* src, float scalar, void* stream);
template void LaunchTRemS<float, 64, 64, 64, 64, 64, 64, true>(float* out, float* src, float scalar, void* stream);
template void LaunchTRemS<float, 64, 64, 64, 64, 64, 61, true>(float* out, float* src, float scalar, void* stream);
template void LaunchTRemS<int64_t, 4, 16, 4, 16, 4, 16>(int64_t* out, int64_t* src, int64_t scalar, void* stream);
template void LaunchTRemS<uint64_t, 4, 16, 4, 16, 4, 16>(uint64_t* out, uint64_t* src, uint64_t scalar, void* stream);
template void LaunchTRemS<int64_t, 4, 64, 4, 64, 4, 64>(int64_t* out, int64_t* src, int64_t scalar, void* stream);
template void LaunchTRemS<uint64_t, 4, 64, 4, 64, 4, 64>(uint64_t* out, uint64_t* src, uint64_t scalar, void* stream);
template void LaunchTRemS<int64_t, 1, 10912, 1, 10912, 1, 10912>(
    int64_t* out, int64_t* src, int64_t scalar, void* stream);
template void LaunchTRemS<uint64_t, 1, 10912, 1, 10912, 1, 10912>(
    uint64_t* out, uint64_t* src, uint64_t scalar, void* stream);

template <typename T, int rows, int stride, int validCols>
__global__ AICORE void runTRemSInplaceWideInt64(__gm__ T* out, __gm__ T* src, T scalar)
{
    constexpr int tileRows = 1;
    constexpr int tileCols = 64;
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, 1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    using TileData = Tile<TileType::Vec, T, tileRows, tileCols, BLayout::RowMajor, -1, -1>;
    using TmpTileData = Tile<TileType::Vec, T, tileRows, tileCols, BLayout::RowMajor, -1, -1>;

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < validCols; col += tileCols) {
            int curValidCols = (col + tileCols <= validCols) ? tileCols : (validCols - col);
            int offset = row * stride + col;
            GlobalData srcGlobal(src + offset, DynDim2Shape(tileRows, curValidCols), DynDim2Stride(stride, 1));
            GlobalData dstGlobal(out + offset, DynDim2Shape(tileRows, curValidCols), DynDim2Stride(stride, 1));
            TileData src0Tile(tileRows, curValidCols);
            TileData dstTile(tileRows, curValidCols);
            TmpTileData tmpTile(tileRows, curValidCols);
            constexpr unsigned tileBytes = tileRows * tileCols * sizeof(T);
            TASSIGN(src0Tile, 0x0);
            TASSIGN(dstTile, 0x0);
            TASSIGN(tmpTile, tileBytes * 2);
            Event<Op::TLOAD, Op::TREMS> event0 = TLOAD(src0Tile, srcGlobal);
            Event<Op::TREMS, Op::TSTORE_VEC> event1 = TREMS(dstTile, src0Tile, scalar, tmpTile, event0);
            TSTORE(dstGlobal, dstTile, event1);
            pipe_barrier(PIPE_ALL);
        }
    }
}

template <typename T, int tileH, int tileW, int vRows, int vCols>
__global__ AICORE void runTRemSInplace(__gm__ T* out, __gm__ T* src, T scalar)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = pto::Stride<1, 1, 1, tileW, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, tileH, tileW, BLayout::RowMajor, -1, -1>;
    using TmpTileData = Tile<TileType::Vec, T, 1, tileW, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TmpTileData tmpTile(1, vCols);
    constexpr unsigned tileBytes = tileH * tileW * sizeof(T);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(dstTile, 0x0);
    TASSIGN(tmpTile, tileBytes * 2);
    GlobalData dstGlobal(out);
    GlobalData srcGlobal(src);
    TLOAD(src0Tile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TREMS<RemSAlgorithm::DEFAULT>(dstTile, src0Tile, scalar, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int tileH, int tileW, int vRows, int vCols>
void LaunchTRemSInplace(T* out, T* src, T scalar, void* stream)
{
    if constexpr ((std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) && tileW > 32) {
        runTRemSInplaceWideInt64<T, vRows, tileW, vCols><<<1, nullptr, stream>>>(out, src, scalar);
    } else {
        runTRemSInplace<T, tileH, tileW, vRows, vCols><<<1, nullptr, stream>>>(out, src, scalar);
    }
}

template void LaunchTRemSInplace<int64_t, 4, 32, 4, 32>(int64_t* out, int64_t* src, int64_t scalar, void* stream);
template void LaunchTRemSInplace<uint64_t, 4, 32, 4, 32>(uint64_t* out, uint64_t* src, uint64_t scalar, void* stream);
template void LaunchTRemSInplace<int64_t, 1, 1024, 1, 1024>(int64_t* out, int64_t* src, int64_t scalar, void* stream);
template void LaunchTRemSInplace<int64_t, 4, 64, 4, 40>(int64_t* out, int64_t* src, int64_t scalar, void* stream);
template void LaunchTRemSInplace<int64_t, 1, 2048, 1, 2045>(int64_t* out, int64_t* src, int64_t scalar, void* stream);
