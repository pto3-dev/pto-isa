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

#define PTO_CEIL(x, y) ((((x) + (y) - 1) / (y)) * (y))

template <
    typename T, typename TMask, int dstTileH, int dstTileW, int maskTileH, int maskTileW, int srcTileH, int srcTileW,
    int vRows, int vCols>
__global__ AICORE void runTSELS(__gm__ T __out__* out, __gm__ TMask* mask, __gm__ T __in__* src, T __in__ scalar)
{
    using DynShape = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    using GlobalDataMask = GlobalTensor<TMask, DynShape, DynStride>;
    GlobalData dstGlobal(
        out, pto::Shape(1, 1, 1, vRows, vCols),
        pto::Stride(dstTileH * dstTileW, dstTileH * dstTileW, dstTileH * dstTileW, dstTileW, 1));
    GlobalDataMask maskGlobal(
        mask, pto::Shape(1, 1, 1, vRows, maskTileW),
        pto::Stride(maskTileH * maskTileW, maskTileH * maskTileW, maskTileH * maskTileW, maskTileW, 1));
    GlobalData srcGlobal(
        src, pto::Shape(1, 1, 1, vRows, vCols),
        pto::Stride(srcTileH * srcTileW, srcTileH * srcTileW, srcTileH * srcTileW, srcTileW, 1));

    using TileDataDst = Tile<TileType::Vec, T, dstTileH, dstTileW, BLayout::RowMajor, -1, -1>;
    using TileDataMask = Tile<TileType::Vec, TMask, maskTileH, maskTileW, BLayout::RowMajor, -1, -1>;
    using TileDataSrc = Tile<TileType::Vec, T, srcTileH, srcTileW, BLayout::RowMajor, -1, -1>;
    using TmpTile = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, -1, -1>;
    TileDataDst dstTile(vRows, vCols);
    TileDataMask maskTile(vRows, maskTileW);
    TileDataSrc srcTile(vRows, vCols);
    TmpTile tmpTile(1, 32);
    size_t dstSize = sizeof(T) * dstTileH * dstTileW;
    size_t srcSize = sizeof(T) * srcTileH * srcTileW;
    size_t maskSize = sizeof(TMask) * maskTileH * maskTileW;
    size_t tmpOffset = PTO_CEIL(dstSize + srcSize + maskSize, 512);
    size_t totalSize = tmpOffset + 32;
    size_t dstOffset = totalSize * block_idx;
    size_t srcOffset = totalSize * block_idx + dstSize;
    size_t maskOffset = totalSize * block_idx + dstSize + srcSize;
    size_t tmpBaseOffset = totalSize * block_idx + tmpOffset;
    TASSIGN(dstTile, dstOffset);
    TASSIGN(maskTile, maskOffset);
    TASSIGN(srcTile, srcOffset);
    TASSIGN(tmpTile, tmpBaseOffset);

    Event<Op::TLOAD, Op::TSELS> event0;
    Event<Op::TSELS, Op::TSTORE_VEC> event1;

    TLOAD(maskTile, maskGlobal);
    event0 = TLOAD(srcTile, srcGlobal);
    event1 = TSELS(dstTile, maskTile, srcTile, tmpTile, scalar, event0);
    TSTORE(dstGlobal, dstTile, event1);
    out = dstGlobal.data();
}

template <typename T, int cols>
__global__ AICORE void runTSELSWideInt64(__gm__ T __out__* out, __gm__ uint8_t* mask, __gm__ T __in__* src, T scalar)
{
    constexpr int maskCols = (cols + 7) / 8;
    constexpr int tileRows = 1;
    constexpr int tileCols = 32;
    constexpr int maskTileCols = 32;

    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using DynStrideDim5 = pto::Stride<1, 1, 1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStrideDim5>;
    using GlobalDataMask = GlobalTensor<uint8_t, DynShapeDim5, DynStrideDim5>;
    using TileData = Tile<TileType::Vec, T, tileRows, tileCols, BLayout::RowMajor, -1, -1>;
    using TileDataMask = Tile<TileType::Vec, uint8_t, tileRows, maskTileCols, BLayout::RowMajor, -1, -1>;
    using TmpTile = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, -1, -1>;

    for (int col = 0; col < cols; col += tileCols) {
        int validCols = (col + tileCols <= cols) ? tileCols : (cols - col);
        int validMaskCols = (validCols + 7) / 8;
        GlobalData dstGlobal(out + col, DynShapeDim5(tileRows, validCols), DynStrideDim5(cols, 1));
        GlobalData srcGlobal(src + col, DynShapeDim5(tileRows, validCols), DynStrideDim5(cols, 1));
        GlobalDataMask maskGlobal(mask + col / 8, DynShapeDim5(tileRows, validMaskCols), DynStrideDim5(maskCols, 1));
        TileData dstTile(tileRows, validCols);
        TileData srcTile(tileRows, validCols);
        TileDataMask maskTile(tileRows, validMaskCols);
        TmpTile tmpTile(1, 32);
        TASSIGN(dstTile, 0x0);
        TASSIGN(srcTile, 0x2000);
        TASSIGN(maskTile, 0x4000);
        TASSIGN(tmpTile, 0x6000);

        TLOAD(maskTile, maskGlobal);
        Event<Op::TLOAD, Op::TSELS> event0 = TLOAD(srcTile, srcGlobal);
        Event<Op::TSELS, Op::TSTORE_VEC> event1 = TSELS(dstTile, maskTile, srcTile, tmpTile, scalar, event0);
        TSTORE(dstGlobal, dstTile, event1);
        pipe_barrier(PIPE_ALL);
    }
}

template <
    typename T, typename TMask, int dstTileH, int dstTileW, int maskTileH, int maskTileW, int srcTileH, int srcTileW,
    int vRows, int vCols>
__global__ AICORE void runTSELSInplace(__gm__ T __out__* out, __gm__ TMask* mask, T __in__ scalar)
{
    using DynShape = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    using GlobalDataMask = GlobalTensor<TMask, DynShape, DynStride>;
    GlobalData dstGlobal(
        out, pto::Shape(1, 1, 1, vRows, vCols),
        pto::Stride(dstTileH * dstTileW, dstTileH * dstTileW, dstTileH * dstTileW, dstTileW, 1));
    GlobalDataMask maskGlobal(
        mask, pto::Shape(1, 1, 1, vRows, maskTileW),
        pto::Stride(maskTileH * maskTileW, maskTileH * maskTileW, maskTileH * maskTileW, maskTileW, 1));
    GlobalData srcGlobal(
        out, pto::Shape(1, 1, 1, vRows, vCols),
        pto::Stride(srcTileH * srcTileW, srcTileH * srcTileW, srcTileH * srcTileW, srcTileW, 1));

    using TileDataDst = Tile<TileType::Vec, T, dstTileH, dstTileW, BLayout::RowMajor, -1, -1>;
    using TileDataMask = Tile<TileType::Vec, TMask, maskTileH, maskTileW, BLayout::RowMajor, -1, -1>;
    using TileDataSrc = Tile<TileType::Vec, T, srcTileH, srcTileW, BLayout::RowMajor, -1, -1>;
    using TmpTile = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, -1, -1>;
    TileDataDst dstTile(vRows, vCols);
    TileDataMask maskTile(vRows, maskTileW);
    TileDataSrc srcTile(vRows, vCols);
    TmpTile tmpTile(1, 32);
    size_t dstSize = sizeof(T) * dstTileH * dstTileW;
    size_t maskSize = sizeof(TMask) * maskTileH * maskTileW;
    size_t totalSize = dstSize + maskSize;
    size_t dstOffset = totalSize * block_idx;
    size_t maskOffset = totalSize * block_idx + dstSize;
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0);
    TASSIGN(maskTile, dstSize);
    TASSIGN(tmpTile, dstSize * 2);

    TLOAD(maskTile, maskGlobal);
    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TSELS(dstTile, maskTile, srcTile, tmpTile, scalar);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int vRows, int vCols>
__global__ AICORE void runTSELSInplaceWideInt64(__gm__ T __out__* out, __gm__ uint8_t* mask, T __in__ scalar)
{
    constexpr int maskCols = (vCols + 7) / 8;
    constexpr int tileCols = 32;
    constexpr int maskTileCols = 32;

    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using DynStrideDim5 = pto::Stride<1, 1, 1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStrideDim5>;
    using GlobalDataMask = GlobalTensor<uint8_t, DynShapeDim5, DynStrideDim5>;
    using TileData = Tile<TileType::Vec, T, vRows, tileCols, BLayout::RowMajor, -1, -1>;
    using TileDataMask = Tile<TileType::Vec, uint8_t, vRows, maskTileCols, BLayout::RowMajor, -1, -1>;
    using TmpTile = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, -1, -1>;

    for (int col = 0; col < vCols; col += tileCols) {
        int validCols = (col + tileCols <= vCols) ? tileCols : (vCols - col);
        int validMaskCols = (validCols + 7) / 8;
        GlobalData dstGlobal(out + col, DynShapeDim5(vRows, validCols), DynStrideDim5(vCols, 1));
        GlobalData srcGlobal(out + col, DynShapeDim5(vRows, validCols), DynStrideDim5(vCols, 1));
        GlobalDataMask maskGlobal(mask + col / 8, DynShapeDim5(vRows, validMaskCols), DynStrideDim5(maskCols, 1));
        TileData srcTile(vRows, validCols);
        TileData dstTile(vRows, validCols);
        TileDataMask maskTile(vRows, maskTileCols);
        TmpTile tmpTile(1, 32);
        constexpr uint64_t tileSize = vRows * tileCols * sizeof(T);
        constexpr uint64_t maskSize = vRows * maskTileCols;
        TASSIGN(srcTile, 0x0);
        TASSIGN(dstTile, 0x0);
        TASSIGN(maskTile, tileSize);
        TASSIGN(tmpTile, tileSize * 2);

        TLOAD(maskTile, maskGlobal);
        TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
        TSELS(dstTile, maskTile, srcTile, tmpTile, scalar);
#ifndef __PTO_AUTO__
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
        TSTORE(dstGlobal, dstTile);
        pipe_barrier(PIPE_ALL);
    }
}

template <
    typename T, typename TMask, int dstTileH, int dstTileW, int maskTileH, int maskTileW, int srcTileH, int srcTileW,
    int vRows, int vCols>
void LaunchTSelsInplace(T* out, TMask* mask, T scalar, void* stream)
{
    constexpr int wideTileCols = 32;
    if constexpr (
        (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) && std::is_same_v<TMask, uint8_t> &&
        dstTileW == srcTileW && vCols > wideTileCols && (dstTileW == vCols || vRows == 1)) {
        runTSELSInplaceWideInt64<T, vRows, vCols><<<1, nullptr, stream>>>(out, mask, scalar);
    } else {
        runTSELSInplace<T, TMask, dstTileH, dstTileW, maskTileH, maskTileW, srcTileH, srcTileW, vRows, vCols>
            <<<1, nullptr, stream>>>(out, mask, scalar);
    }
}

template <
    typename T, typename TMask, int dstTileH, int dstTileW, int maskTileH, int maskTileW, int srcTileH, int srcTileW,
    int vRows, int vCols>
void LaunchTSels(T* out, TMask* mask, T* src, T scalar, void* stream)
{
    constexpr int wideTileCols = 32;
    if constexpr (
        (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) && std::is_same_v<TMask, uint8_t> &&
        dstTileH == 1 && srcTileH == 1 && maskTileH == 1 && dstTileW == srcTileW && dstTileW == vCols &&
        maskTileW >= (vCols + 7) / 8 && vRows == 1 && vCols > wideTileCols) {
        runTSELSWideInt64<T, vCols><<<1, nullptr, stream>>>(out, mask, src, scalar);
    } else {
        runTSELS<T, TMask, dstTileH, dstTileW, maskTileH, maskTileW, srcTileH, srcTileW, vRows, vCols>
            <<<1, nullptr, stream>>>(out, mask, src, scalar);
    }
}

template <
    typename TMask, int dstTileH, int dstTileW, int maskTileH, int maskTileW, int srcTileH, int srcTileW, int vRows,
    int vCols>
void LaunchTSelsHalf(aclFloat16* out, TMask* mask, aclFloat16* src, aclFloat16 scalar, void* stream)
{
    runTSELS<half, TMask, dstTileH, dstTileW, maskTileH, maskTileW, srcTileH, srcTileW, vRows, vCols>
        <<<1, nullptr, stream>>>((half*)out, mask, (half*)src, *(half*)&scalar);
}

template void LaunchTSels<uint8_t, uint8_t, 2, 32, 2, 32, 2, 32, 2, 32>(
    uint8_t* out, uint8_t* mask, uint8_t* src, uint8_t scalar, void* stream);
template void LaunchTSels<uint8_t, uint16_t, 2, 32, 2, 16, 2, 32, 2, 32>(
    uint8_t* out, uint16_t* mask, uint8_t* src, uint8_t scalar, void* stream);
template void LaunchTSels<uint8_t, uint32_t, 2, 32, 2, 8, 2, 32, 2, 32>(
    uint8_t* out, uint32_t* mask, uint8_t* src, uint8_t scalar, void* stream);

template void LaunchTSels<uint16_t, uint8_t, 2, 16, 2, 32, 2, 16, 2, 16>(
    uint16_t* out, uint8_t* mask, uint16_t* src, uint16_t scalar, void* stream);
template void LaunchTSels<uint16_t, uint16_t, 2, 16, 2, 16, 2, 16, 2, 16>(
    uint16_t* out, uint16_t* mask, uint16_t* src, uint16_t scalar, void* stream);
template void LaunchTSels<uint16_t, uint32_t, 2, 16, 2, 8, 2, 16, 2, 16>(
    uint16_t* out, uint32_t* mask, uint16_t* src, uint16_t scalar, void* stream);

template void LaunchTSels<uint32_t, uint8_t, 2, 8, 2, 32, 2, 8, 2, 8>(
    uint32_t* out, uint8_t* mask, uint32_t* src, uint32_t scalar, void* stream);
template void LaunchTSels<uint32_t, uint16_t, 2, 8, 2, 16, 2, 8, 2, 8>(
    uint32_t* out, uint16_t* mask, uint32_t* src, uint32_t scalar, void* stream);
template void LaunchTSels<uint32_t, uint32_t, 2, 8, 2, 8, 2, 8, 2, 8>(
    uint32_t* out, uint32_t* mask, uint32_t* src, uint32_t scalar, void* stream);

template void LaunchTSelsHalf<uint8_t, 2, 16, 2, 32, 2, 16, 2, 16>(
    aclFloat16* out, uint8_t* mask, aclFloat16* src, aclFloat16 scalar, void* stream);
template void LaunchTSelsHalf<uint16_t, 2, 16, 2, 16, 2, 16, 2, 16>(
    aclFloat16* out, uint16_t* mask, aclFloat16* src, aclFloat16 scalar, void* stream);
template void LaunchTSelsHalf<uint32_t, 2, 16, 2, 8, 2, 16, 2, 16>(
    aclFloat16* out, uint32_t* mask, aclFloat16* src, aclFloat16 scalar, void* stream);

template void LaunchTSels<float, uint8_t, 2, 8, 2, 32, 2, 8, 2, 8>(
    float* out, uint8_t* mask, float* src, float scalar, void* stream);
template void LaunchTSels<float, uint16_t, 2, 8, 2, 16, 2, 8, 2, 8>(
    float* out, uint16_t* mask, float* src, float scalar, void* stream);
template void LaunchTSels<float, uint32_t, 2, 8, 2, 8, 2, 8, 2, 8>(
    float* out, uint32_t* mask, float* src, float scalar, void* stream);

template void LaunchTSels<uint8_t, uint8_t, 2, 32, 2, 64, 2, 128, 2, 31>(
    uint8_t* out, uint8_t* mask, uint8_t* src, uint8_t scalar, void* stream);
template void LaunchTSels<uint16_t, uint8_t, 2, 32, 2, 64, 2, 128, 2, 31>(
    uint16_t* out, uint8_t* mask, uint16_t* src, uint16_t scalar, void* stream);
template void LaunchTSels<float, uint8_t, 2, 32, 2, 64, 2, 128, 2, 31>(
    float* out, uint8_t* mask, float* src, float scalar, void* stream);

template void LaunchTSels<uint8_t, uint8_t, 32, 672, 32, 96, 32, 672, 32, 666>(
    uint8_t* out, uint8_t* mask, uint8_t* src, uint8_t scalar, void* stream);
template void LaunchTSelsHalf<uint8_t, 32, 672, 32, 96, 32, 672, 32, 666>(
    aclFloat16* out, uint8_t* mask, aclFloat16* src, aclFloat16 scalar, void* stream);
template void LaunchTSels<float, uint8_t, 32, 672, 32, 96, 32, 672, 32, 666>(
    float* out, uint8_t* mask, float* src, float scalar, void* stream);

template void LaunchTSels<float, uint8_t, 1, 8192, 1, 4096, 1, 8192, 1, 8192>(
    float* out, uint8_t* mask, float* src, float scalar, void* stream);
template void LaunchTSels<int64_t, uint8_t, 4, 16, 4, 32, 4, 16, 4, 16>(
    int64_t* out, uint8_t* mask, int64_t* src, int64_t scalar, void* stream);
template void LaunchTSels<uint64_t, uint8_t, 4, 16, 4, 32, 4, 16, 4, 16>(
    uint64_t* out, uint8_t* mask, uint64_t* src, uint64_t scalar, void* stream);
template void LaunchTSels<int64_t, uint8_t, 49, 160, 49, 32, 49, 160, 49, 160>(
    int64_t* out, uint8_t* mask, int64_t* src, int64_t scalar, void* stream);
template void LaunchTSels<uint64_t, uint8_t, 49, 160, 49, 32, 49, 160, 49, 160>(
    uint64_t* out, uint8_t* mask, uint64_t* src, uint64_t scalar, void* stream);
template void LaunchTSels<int64_t, uint8_t, 1, 16364, 1, 2048, 1, 16364, 1, 16364>(
    int64_t* out, uint8_t* mask, int64_t* src, int64_t scalar, void* stream);
template void LaunchTSels<uint64_t, uint8_t, 1, 16364, 1, 2048, 1, 16364, 1, 16364>(
    uint64_t* out, uint8_t* mask, uint64_t* src, uint64_t scalar, void* stream);
template void LaunchTSels<int64_t, uint8_t, 1, 16368, 1, 2046, 1, 16368, 1, 16368>(
    int64_t* out, uint8_t* mask, int64_t* src, int64_t scalar, void* stream);
template void LaunchTSels<uint64_t, uint8_t, 1, 16368, 1, 2046, 1, 16368, 1, 16368>(
    uint64_t* out, uint8_t* mask, uint64_t* src, uint64_t scalar, void* stream);
template void LaunchTSelsInplace<int64_t, uint8_t, 4, 32, 4, 32, 4, 32, 4, 32>(
    int64_t* out, uint8_t* mask, int64_t scalar, void* stream);
template void LaunchTSelsInplace<uint64_t, uint8_t, 4, 32, 4, 32, 4, 32, 4, 32>(
    uint64_t* out, uint8_t* mask, uint64_t scalar, void* stream);
template void LaunchTSelsInplace<int64_t, uint8_t, 1, 1024, 1, 128, 1, 1024, 1, 1024>(
    int64_t* out, uint8_t* mask, int64_t scalar, void* stream);
template void LaunchTSelsInplace<int64_t, uint8_t, 4, 40, 4, 5, 4, 40, 4, 40>(
    int64_t* out, uint8_t* mask, int64_t scalar, void* stream);
template void LaunchTSelsInplace<int64_t, uint8_t, 1, 2048, 1, 256, 1, 2048, 1, 2045>(
    int64_t* out, uint8_t* mask, int64_t scalar, void* stream);
