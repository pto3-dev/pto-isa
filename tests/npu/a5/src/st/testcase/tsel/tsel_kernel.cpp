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
#include "acl/acl.h"

using namespace pto;

#define PTO_DIV_ROUNDUP(x, y) ((((x) + (y) - 1) / (y)))
#define PTO_CEIL(x, y) ((((x) + (y) - 1) / (y)) * (y))

template <typename T, int Rows, int Cols, int ValidRows, int ValidCols>
__global__ AICORE void runTSel(
    __gm__ T __out__* out, __gm__ uint8_t __in__* mask, __gm__ T __in__* src0, __gm__ T __in__* src1)
{
    constexpr unsigned maskRow = Rows;
    constexpr unsigned maskCol = ((((Cols + 7) / 8) + 31) / 32) * 32;
    constexpr unsigned maskVRow = ValidRows;
    constexpr unsigned maskVCol = (ValidCols + 7) / 8;

    using DynShapeDim5 = Shape<1, 1, 1, ValidRows, ValidCols>;
    using DynStridDim5 = pto::Stride<1, 1, 1, ValidCols, 1>;

    using DynShapeDim5m = Shape<1, 1, 1, maskVRow, maskVCol>;
    using DynStridDim5m = pto::Stride<1, 1, 1, maskVCol, 1>;

    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, -1, -1>;
    using MaskGlobal = GlobalTensor<uint8_t, DynShapeDim5m, DynStridDim5m>;

    using MaskTile = Tile<TileType::Vec, uint8_t, maskRow, maskCol, BLayout::RowMajor, -1, -1>;
    using TmpTile = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, -1, -1>;

    TileData src0Tile(ValidRows, ValidCols);
    TileData src1Tile(ValidRows, ValidCols);
    TileData dstTile(ValidRows, ValidCols);
    MaskTile maskTile(maskVRow, maskVCol);
    TmpTile tmpTile(1, 32);
    constexpr uint64_t tileSize = Rows * Cols * sizeof(T);
    constexpr uint64_t maskSize = maskRow * maskCol;
    constexpr uint64_t tmpOffset = PTO_CEIL(tileSize * 3 + maskSize, 512);
    constexpr uint64_t totalSize = tmpOffset + 32;
    static_assert(totalSize <= 192 * 1024, "UB size overflow, should be less than 192KB.");

    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, (uint64_t)(tileSize));
    TASSIGN(dstTile, (uint64_t)(tileSize * 2));
    TASSIGN(maskTile, (uint64_t)(tileSize * 3));
    TASSIGN(tmpTile, (uint64_t)(tmpOffset));

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);
    MaskGlobal maskGlobal(mask);

    Event<Op::TLOAD, Op::TSEL> event0;
    Event<Op::TSEL, Op::TSTORE_VEC> event1;

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    event0 = TLOAD(maskTile, maskGlobal);
    event1 = TSEL(dstTile, maskTile, src0Tile, src1Tile, tmpTile, event0);
    TSTORE(dstGlobal, dstTile, event1);
    out = dstGlobal.data();
}

template <typename T, int cols>
__global__ AICORE void runTSelWideInt64(
    __gm__ T __out__* out, __gm__ uint8_t __in__* mask, __gm__ T __in__* src0, __gm__ T __in__* src1)
{
    constexpr int maskCols = (cols + 7) / 8;
    constexpr int tileRows = 1;
    constexpr int tileCols = 32;
    constexpr int maskTileCols = 32;

    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using DynStrideDim5 = pto::Stride<1, 1, 1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStrideDim5>;
    using MaskGlobal = GlobalTensor<uint8_t, DynShapeDim5, DynStrideDim5>;
    using TileData = Tile<TileType::Vec, T, tileRows, tileCols, BLayout::RowMajor, -1, -1>;
    using MaskTile = Tile<TileType::Vec, uint8_t, tileRows, maskTileCols, BLayout::RowMajor, -1, -1>;
    using TmpTile = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, -1, -1>;

    for (int col = 0; col < cols; col += tileCols) {
        int validCols = (col + tileCols <= cols) ? tileCols : (cols - col);
        int validMaskCols = (validCols + 7) / 8;
        GlobalData dstGlobal(out + col, DynShapeDim5(tileRows, validCols), DynStrideDim5(cols, 1));
        GlobalData src0Global(src0 + col, DynShapeDim5(tileRows, validCols), DynStrideDim5(cols, 1));
        GlobalData src1Global(src1 + col, DynShapeDim5(tileRows, validCols), DynStrideDim5(cols, 1));
        MaskGlobal maskGlobal(mask + col / 8, DynShapeDim5(tileRows, validMaskCols), DynStrideDim5(maskCols, 1));
        TileData dstTile(tileRows, validCols);
        TileData src0Tile(tileRows, validCols);
        TileData src1Tile(tileRows, validCols);
        MaskTile maskTile(tileRows, validMaskCols);
        TmpTile tmpTile(1, 32);
        TASSIGN(dstTile, 0x0);
        TASSIGN(src0Tile, 0x2000);
        TASSIGN(src1Tile, 0x4000);
        TASSIGN(maskTile, 0x6000);
        TASSIGN(tmpTile, 0x8000);

        TLOAD(src0Tile, src0Global);
        TLOAD(src1Tile, src1Global);
        Event<Op::TLOAD, Op::TSEL> event0 = TLOAD(maskTile, maskGlobal);
        Event<Op::TSEL, Op::TSTORE_VEC> event1 = TSEL(dstTile, maskTile, src0Tile, src1Tile, tmpTile, event0);
        TSTORE(dstGlobal, dstTile, event1);
        pipe_barrier(PIPE_ALL);
    }
}

template <typename T, int Rows, int Cols, int ValidRows, int ValidCols>
__global__ AICORE void runTSelInplace(__gm__ T __out__* out, __gm__ uint8_t __in__* mask, __gm__ T __in__* src1)
{
    constexpr unsigned maskRow = Rows;
    constexpr unsigned maskCol = ((((Cols + 7) / 8) + 31) / 32) * 32;
    constexpr unsigned maskVRow = ValidRows;
    constexpr unsigned maskVCol = (ValidCols + 7) / 8;

    using DynShapeDim5 = Shape<1, 1, 1, ValidRows, ValidCols>;
    using DynStridDim5 = pto::Stride<1, 1, 1, ValidCols, 1>;
    using DynShapeDim5m = Shape<1, 1, 1, maskVRow, maskVCol>;
    using DynStridDim5m = pto::Stride<1, 1, 1, maskVCol, 1>;

    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using MaskGlobal = GlobalTensor<uint8_t, DynShapeDim5m, DynStridDim5m>;
    using TileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, -1, -1>;
    using MaskTile = Tile<TileType::Vec, uint8_t, maskRow, maskCol, BLayout::RowMajor, -1, -1>;
    using TmpTile = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, -1, -1>;

    TileData src0Tile(ValidRows, ValidCols);
    TileData dstTile(ValidRows, ValidCols);
    TileData src1Tile(ValidRows, ValidCols);
    MaskTile maskTile(maskVRow, maskVCol);
    TmpTile tmpTile(1, 32);
    constexpr uint64_t tileSize = Rows * Cols * sizeof(T);
    constexpr uint64_t maskSize = maskRow * maskCol;
    constexpr uint64_t totalSize = tileSize * 2 + maskSize;
    static_assert(totalSize <= 192 * 1024, "UB size overflow, should be less than 192KB.");

    TASSIGN(src0Tile, 0x0);
    TASSIGN(dstTile, 0x0);
    TASSIGN(src1Tile, tileSize);
    TASSIGN(maskTile, tileSize * 2);
    TASSIGN(tmpTile, tileSize * 2 + maskSize);

    GlobalData src0Global(out);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);
    MaskGlobal maskGlobal(mask);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    TLOAD(maskTile, maskGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TSEL(dstTile, maskTile, src0Tile, src1Tile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int Rows, int Cols, int ValidRows, int ValidCols>
void LaunchTSel(T* out, uint8_t* mask, T* src0, T* src1, void* stream)
{
    constexpr int wideTileCols = 32;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTSel<half, Rows, Cols, ValidRows, ValidCols>
            <<<1, nullptr, stream>>>((half*)(out), mask, (half*)(src0), (half*)(src1));
    } else if constexpr (
        (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) && Rows == 1 && Cols == ValidCols &&
        ValidRows == 1 && ValidCols > wideTileCols) {
        runTSelWideInt64<T, ValidCols><<<1, nullptr, stream>>>(out, mask, src0, src1);
    } else {
        runTSel<T, Rows, Cols, ValidRows, ValidCols><<<1, nullptr, stream>>>(out, mask, src0, src1);
    }
}

template <typename T, int vRows, int vCols>
__global__ AICORE void runTSelInplaceWideInt64(
    __gm__ T __out__* out, __gm__ uint8_t __in__* mask, __gm__ T __in__* src1)
{
    constexpr int maskCols = (vCols + 7) / 8;
    constexpr int tileCols = 32;
    constexpr int maskTileCols = 32;

    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using DynStrideDim5 = pto::Stride<1, 1, 1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStrideDim5>;
    using MaskGlobal = GlobalTensor<uint8_t, DynShapeDim5, DynStrideDim5>;
    using TileData = Tile<TileType::Vec, T, vRows, tileCols, BLayout::RowMajor, -1, -1>;
    using MaskTile = Tile<TileType::Vec, uint8_t, vRows, maskTileCols, BLayout::RowMajor, -1, -1>;
    using TmpTile = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, -1, -1>;

    for (int col = 0; col < vCols; col += tileCols) {
        int validCols = (col + tileCols <= vCols) ? tileCols : (vCols - col);
        int validMaskCols = (validCols + 7) / 8;
        GlobalData dstGlobal(out + col, DynShapeDim5(vRows, validCols), DynStrideDim5(vCols, 1));
        GlobalData src0Global(out + col, DynShapeDim5(vRows, validCols), DynStrideDim5(vCols, 1));
        GlobalData src1Global(src1 + col, DynShapeDim5(vRows, validCols), DynStrideDim5(vCols, 1));
        MaskGlobal maskGlobal(mask + col / 8, DynShapeDim5(vRows, validMaskCols), DynStrideDim5(maskCols, 1));
        TileData src0Tile(vRows, validCols);
        TileData dstTile(vRows, validCols);
        TileData src1Tile(vRows, validCols);
        MaskTile maskTile(vRows, maskTileCols);
        TmpTile tmpTile(1, 32);
        constexpr uint64_t tileSize = vRows * tileCols * sizeof(T);
        constexpr uint64_t maskSize = vRows * maskTileCols;
        TASSIGN(src0Tile, 0x0);
        TASSIGN(dstTile, 0x0);
        TASSIGN(src1Tile, tileSize);
        TASSIGN(maskTile, tileSize * 2);
        TASSIGN(tmpTile, tileSize * 2 + maskSize);

        TLOAD(src0Tile, src0Global);
        TLOAD(src1Tile, src1Global);
        TLOAD(maskTile, maskGlobal);
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
        TSEL(dstTile, maskTile, src0Tile, src1Tile, tmpTile);
#ifndef __PTO_AUTO__
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
        TSTORE(dstGlobal, dstTile);
        pipe_barrier(PIPE_ALL);
    }
}

template <typename T, int Rows, int Cols, int ValidRows, int ValidCols>
void LaunchTSelInplace(T* out, uint8_t* mask, T* src1, void* stream)
{
    constexpr int wideTileCols = 32;
    if constexpr ((std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) && ValidCols > wideTileCols) {
        runTSelInplaceWideInt64<T, ValidRows, ValidCols><<<1, nullptr, stream>>>(out, mask, src1);
    } else if constexpr (std::is_same_v<T, aclFloat16>) {
        runTSelInplace<half, Rows, Cols, ValidRows, ValidCols>
            <<<1, nullptr, stream>>>((half*)(out), mask, (half*)(src1));
    } else {
        runTSelInplace<T, Rows, Cols, ValidRows, ValidCols><<<1, nullptr, stream>>>(out, mask, src1);
    }
}

template void LaunchTSel<float, 2, 128, 2, 128>(float* out, uint8_t* mask, float* src0, float* src1, void* stream);
template void LaunchTSel<float, 2, 32, 2, 32>(float* out, uint8_t* mask, float* src0, float* src1, void* stream);
template void LaunchTSel<float, 2, 160, 2, 160>(float* out, uint8_t* mask, float* src0, float* src1, void* stream);
template void LaunchTSel<aclFloat16, 2, 128, 2, 128>(
    aclFloat16* out, uint8_t* mask, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTSel<aclFloat16, 2, 32, 2, 32>(
    aclFloat16* out, uint8_t* mask, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTSel<aclFloat16, 2, 160, 2, 160>(
    aclFloat16* out, uint8_t* mask, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTSel<int8_t, 2, 128, 2, 128>(int8_t* out, uint8_t* mask, int8_t* src0, int8_t* src1, void* stream);
template void LaunchTSel<int8_t, 2, 32, 2, 32>(int8_t* out, uint8_t* mask, int8_t* src0, int8_t* src1, void* stream);
template void LaunchTSel<int8_t, 2, 160, 2, 160>(int8_t* out, uint8_t* mask, int8_t* src0, int8_t* src1, void* stream);
template void LaunchTSel<float, 2, 512, 2, 512>(float* out, uint8_t* mask, float* src0, float* src1, void* stream);
template void LaunchTSel<int64_t, 4, 16, 4, 16>(
    int64_t* out, uint8_t* mask, int64_t* src0, int64_t* src1, void* stream);
template void LaunchTSel<uint64_t, 4, 16, 4, 16>(
    uint64_t* out, uint8_t* mask, uint64_t* src0, uint64_t* src1, void* stream);
template void LaunchTSel<int64_t, 49, 160, 49, 160>(
    int64_t* out, uint8_t* mask, int64_t* src0, int64_t* src1, void* stream);
template void LaunchTSel<uint64_t, 49, 160, 49, 160>(
    uint64_t* out, uint8_t* mask, uint64_t* src0, uint64_t* src1, void* stream);
template void LaunchTSel<int64_t, 1, 16364, 1, 16364>(
    int64_t* out, uint8_t* mask, int64_t* src0, int64_t* src1, void* stream);
template void LaunchTSel<uint64_t, 1, 16364, 1, 16364>(
    uint64_t* out, uint8_t* mask, uint64_t* src0, uint64_t* src1, void* stream);
template void LaunchTSel<int64_t, 1, 16368, 1, 16368>(
    int64_t* out, uint8_t* mask, int64_t* src0, int64_t* src1, void* stream);
template void LaunchTSel<uint64_t, 1, 16368, 1, 16368>(
    uint64_t* out, uint8_t* mask, uint64_t* src0, uint64_t* src1, void* stream);
template void LaunchTSelInplace<int64_t, 4, 32, 4, 32>(int64_t* out, uint8_t* mask, int64_t* src1, void* stream);
template void LaunchTSelInplace<uint64_t, 4, 32, 4, 32>(uint64_t* out, uint8_t* mask, uint64_t* src1, void* stream);
template void LaunchTSelInplace<int64_t, 1, 1024, 1, 1024>(int64_t* out, uint8_t* mask, int64_t* src1, void* stream);
template void LaunchTSelInplace<int64_t, 4, 64, 4, 40>(int64_t* out, uint8_t* mask, int64_t* src1, void* stream);
template void LaunchTSelInplace<int64_t, 1, 2048, 1, 2045>(int64_t* out, uint8_t* mask, int64_t* src1, void* stream);
