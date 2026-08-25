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

template <typename T, int dstTileRow, int dstTileCol, int row, int validRow, int col, int validCol>
PTO_INTERNAL void runTAddS(__gm__ T* out, __gm__ T* src, T scalar)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    using srcTileData = Tile<TileType::Vec, T, row, col, BLayout::RowMajor, -1, -1>;
    using dstTileData = Tile<TileType::Vec, T, dstTileRow, dstTileCol, BLayout::RowMajor, -1, -1>;
    GlobalData srcGlobal(src, DynDim2Shape(validRow, validCol), DynDim2Stride(row, col));
    GlobalData dstGlobal(out, DynDim2Shape(validRow, validCol), DynDim2Stride(dstTileRow, dstTileCol));
    srcTileData srcTile(validRow, validCol);
    dstTileData dstTile(validRow, validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x28000);
    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TADDS(dstTile, srcTile, scalar);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <
    typename T, int srcGmRow, int srcGmCol, int dstGmRow, int dstGmCol, int tileRow, int tileCol, int validRow,
    int validCol>
PTO_INTERNAL void runTAddSByColTile(__gm__ T* out, __gm__ T* src, T scalar)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    using SrcTileData = Tile<TileType::Vec, T, tileRow, tileCol, BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<TileType::Vec, T, tileRow, tileCol, BLayout::RowMajor, -1, -1>;

    for (int colOffset = 0; colOffset < validCol; colOffset += tileCol) {
        int curCol = (validCol - colOffset) < tileCol ? (validCol - colOffset) : tileCol;
        GlobalData srcGlobal(src + colOffset, DynDim2Shape(validRow, curCol), DynDim2Stride(srcGmRow, srcGmCol));
        GlobalData dstGlobal(out + colOffset, DynDim2Shape(validRow, curCol), DynDim2Stride(dstGmRow, dstGmCol));
        SrcTileData srcTile(validRow, curCol);
        DstTileData dstTile(validRow, curCol);
        TASSIGN(srcTile, 0x0);
        TASSIGN(dstTile, 0x28000);
        TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
        TADDS(dstTile, srcTile, scalar);
#ifndef __PTO_AUTO__
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
        TSTORE(dstGlobal, dstTile);
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        pipe_barrier(PIPE_ALL);
#endif
    }
}

extern "C" __global__ AICORE void launchTADDSCase1(__gm__ float* out, __gm__ float* src, float scalar)
{
    runTAddS<float, 32, 128, 32, 32, 64, 64>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTADDSCase2(__gm__ aclFloat16* out, __gm__ aclFloat16* src, float scalar)
{
    runTAddS<half, 63, 128, 63, 63, 64, 64>((__gm__ half*)out, (__gm__ half*)src, (half)scalar);
}
extern "C" __global__ AICORE void launchTADDSCase3(__gm__ int32_t* out, __gm__ int32_t* src, int32_t scalar)
{
    runTAddS<int32_t, 31, 256, 31, 31, 128, 128>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTADDSCase4(__gm__ int16_t* out, __gm__ int16_t* src, int16_t scalar)
{
    runTAddS<int16_t, 15, 192, 15, 15, 192, 192>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTADDSCase5(__gm__ float* out, __gm__ float* src, float scalar)
{
    runTAddS<float, 7, 512, 7, 7, 448, 448>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTADDSCase6(__gm__ float* out, __gm__ float* src, float scalar)
{
    runTAddS<float, 256, 32, 256, 256, 16, 16>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTADDSCase7(__gm__ uint32_t* out, __gm__ uint32_t* src, uint32_t scalar)
{
    runTAddS<uint32_t, 256, 32, 256, 256, 16, 16>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTADDSCase8(__gm__ uint16_t* out, __gm__ uint16_t* src, uint16_t scalar)
{
    runTAddS<uint16_t, 256, 32, 256, 256, 16, 16>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTADDSCase9(__gm__ int8_t* out, __gm__ int8_t* src, int8_t scalar)
{
    runTAddS<int8_t, 256, 64, 256, 256, 32, 32>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTADDSCase10(__gm__ uint8_t* out, __gm__ uint8_t* src, uint8_t scalar)
{
    runTAddS<uint8_t, 256, 64, 256, 256, 32, 32>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTADDSCase11(__gm__ uint8_t* out, __gm__ uint8_t* src, uint8_t scalar)
{
    runTAddS<uint8_t, 1, 64, 1, 1, 32, 32>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTADDSCase12(__gm__ int64_t* out, __gm__ int64_t* src, int64_t scalar)
{
    runTAddS<int64_t, 4, 16, 4, 4, 16, 16>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTADDSCase13(__gm__ uint64_t* out, __gm__ uint64_t* src, uint64_t scalar)
{
    runTAddS<uint64_t, 4, 16, 4, 4, 16, 16>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTADDSCase14(__gm__ int64_t* out, __gm__ int64_t* src, int64_t scalar)
{
    runTAddSByColTile<int64_t, 96, 32768, 32, 1024, 32, 128, 32, 1024>(out, src, scalar);
}

template <typename T, int cols>
PTO_INTERNAL void runTAddSWideInt64(__gm__ T* out, __gm__ T* src, T scalar)
{
    constexpr int tileRows = 1;
    constexpr int tileCols = 64;

    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using DynStridDim5 = pto::Stride<1, 1, 1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, tileRows, tileCols, BLayout::RowMajor, -1, -1>;

    for (int col = 0; col < cols; col += tileCols) {
        int validCols = (col + tileCols <= cols) ? tileCols : (cols - col);
        GlobalData srcGlobal(src + col, DynShapeDim5(tileRows, validCols), DynStridDim5(cols, 1));
        GlobalData dstGlobal(out + col, DynShapeDim5(tileRows, validCols), DynStridDim5(cols, 1));
        TileData srcTile(tileRows, validCols);
        TileData dstTile(tileRows, validCols);
        TASSIGN(srcTile, 0x0);
        TASSIGN(dstTile, 0x2000);

        Event<Op::TLOAD, Op::TADDS> event0 = TLOAD(srcTile, srcGlobal);
        Event<Op::TADDS, Op::TSTORE_VEC> event1 = TADDS(dstTile, srcTile, scalar, event0);
        TSTORE(dstGlobal, dstTile, event1);
        pipe_barrier(PIPE_ALL);
    }
}

extern "C" __global__ AICORE void launchTADDSCase15(__gm__ int64_t* out, __gm__ int64_t* src, int64_t scalar)
{
    runTAddSWideInt64<int64_t, 16364>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTADDSCase16(__gm__ uint64_t* out, __gm__ uint64_t* src, uint64_t scalar)
{
    runTAddSWideInt64<uint64_t, 16364>(out, src, scalar);
}

template <typename T, int rows, int stride, int validCols>
PTO_INTERNAL void runTAddSInplaceWideInt64(__gm__ T* out, __gm__ T* src, T scalar)
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

            Event<Op::TLOAD, Op::TADDS> event0 = TLOAD(src0Tile, srcGlobal);
            Event<Op::TADDS, Op::TSTORE_VEC> event1 = TADDS(dstTile, src0Tile, scalar, event0);
            TSTORE(dstGlobal, dstTile, event1);
            pipe_barrier(PIPE_ALL);
        }
    }
}

template <typename T, int tileH, int tileW, int vRows, int vCols>
PTO_INTERNAL void runTAddSInplace(__gm__ T* out, __gm__ T* src, T scalar)
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
    TADDS(dstTile, src0Tile, scalar);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTADDSCase17(__gm__ int64_t* out, __gm__ int64_t* src, int64_t scalar)
{
    runTAddSInplace<int64_t, 4, 32, 4, 32>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTADDSCase18(__gm__ int64_t* out, __gm__ int64_t* src, int64_t scalar)
{
    runTAddSInplaceWideInt64<int64_t, 1, 1024, 1024>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTADDSCase19(__gm__ int64_t* out, __gm__ int64_t* src, int64_t scalar)
{
    runTAddSInplaceWideInt64<int64_t, 4, 64, 40>(out, src, scalar);
}

extern "C" __global__ AICORE void launchTADDSCase20(__gm__ int64_t* out, __gm__ int64_t* src, int64_t scalar) {}

extern "C" __global__ AICORE void launchTADDSCase21(__gm__ int64_t* out, __gm__ int64_t* src, int64_t scalar)
{
    runTAddSInplaceWideInt64<int64_t, 1, 2048, 2045>(out, src, scalar);
}

extern "C" __global__ AICORE void launchTADDSCase22(__gm__ uint64_t* out, __gm__ uint64_t* src, uint64_t scalar)
{
    runTAddSInplace<uint64_t, 4, 32, 4, 32>(out, src, scalar);
}

template <uint32_t caseId, typename T>
void launchTADDSTestCase(void* out, void* src, T scalar, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTADDSCase1<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar);
            break;
        }
        case 2: {
            launchTADDSCase2<<<1, nullptr, stream>>>((aclFloat16*)out, (aclFloat16*)src, scalar);
            break;
        }
        case 3: {
            launchTADDSCase3<<<1, nullptr, stream>>>((int32_t*)out, (int32_t*)src, scalar);
            break;
        }
        case 4: {
            launchTADDSCase4<<<1, nullptr, stream>>>((int16_t*)out, (int16_t*)src, scalar);
            break;
        }
        case 5: {
            launchTADDSCase5<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar);
            break;
        }
        case 6: {
            launchTADDSCase6<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar);
            break;
        }
        case 7: {
            launchTADDSCase7<<<1, nullptr, stream>>>((uint32_t*)out, (uint32_t*)src, scalar);
            break;
        }
        case 8: {
            launchTADDSCase8<<<1, nullptr, stream>>>((uint16_t*)out, (uint16_t*)src, scalar);
            break;
        }
        case 9: {
            launchTADDSCase9<<<1, nullptr, stream>>>((int8_t*)out, (int8_t*)src, scalar);
            break;
        }
        case 10: {
            launchTADDSCase10<<<1, nullptr, stream>>>((uint8_t*)out, (uint8_t*)src, scalar);
            break;
        }
        case 11: {
            launchTADDSCase11<<<1, nullptr, stream>>>((uint8_t*)out, (uint8_t*)src, scalar);
            break;
        }
        case 12: {
            launchTADDSCase12<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src, (int64_t)scalar);
            break;
        }
        case 13: {
            launchTADDSCase13<<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src, (uint64_t)scalar);
            break;
        }
        case 14: {
            launchTADDSCase14<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src, (int64_t)scalar);
            break;
        }
        case 15: {
            launchTADDSCase15<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src, (int64_t)scalar);
            break;
        }
        case 16: {
            launchTADDSCase16<<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src, (uint64_t)scalar);
            break;
        }
        case 17: {
            launchTADDSCase17<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src, (int64_t)scalar);
            break;
        }
        case 18: {
            launchTADDSCase18<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src, (int64_t)scalar);
            break;
        }
        case 19: {
            launchTADDSCase19<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src, (int64_t)scalar);
            break;
        }
        case 20: {
            launchTADDSCase20<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src, (int64_t)scalar);
            break;
        }
        case 21: {
            launchTADDSCase21<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src, (int64_t)scalar);
            break;
        }
        case 22: {
            launchTADDSCase22<<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src, (uint64_t)scalar);
            break;
        }
        default: {
        }
    }
}

template <uint32_t caseId>
void launchTADDSTestCase(void* out, void* src, float scalar, aclrtStream stream)
{
    launchTADDSTestCase<caseId, float>(out, src, scalar, stream);
}

template void launchTADDSTestCase<1, float>(void* out, void* src, float scalar, aclrtStream stream);
template void launchTADDSTestCase<2, aclFloat16>(void* out, void* src, aclFloat16 scalar, aclrtStream stream);
template void launchTADDSTestCase<3, int32_t>(void* out, void* src, int32_t scalar, aclrtStream stream);
template void launchTADDSTestCase<4, int16_t>(void* out, void* src, int16_t scalar, aclrtStream stream);
template void launchTADDSTestCase<5, float>(void* out, void* src, float scalar, aclrtStream stream);
template void launchTADDSTestCase<6, float>(void* out, void* src, float scalar, aclrtStream stream);
template void launchTADDSTestCase<7, uint32_t>(void* out, void* src, uint32_t scalar, aclrtStream stream);
template void launchTADDSTestCase<8, uint16_t>(void* out, void* src, uint16_t scalar, aclrtStream stream);
template void launchTADDSTestCase<9, int8_t>(void* out, void* src, int8_t scalar, aclrtStream stream);
template void launchTADDSTestCase<10, uint8_t>(void* out, void* src, uint8_t scalar, aclrtStream stream);
template void launchTADDSTestCase<11, uint8_t>(void* out, void* src, uint8_t scalar, aclrtStream stream);
template void launchTADDSTestCase<12, int64_t>(void* out, void* src, int64_t scalar, aclrtStream stream);
template void launchTADDSTestCase<13, uint64_t>(void* out, void* src, uint64_t scalar, aclrtStream stream);
template void launchTADDSTestCase<14, int64_t>(void* out, void* src, int64_t scalar, aclrtStream stream);
template void launchTADDSTestCase<15, int64_t>(void* out, void* src, int64_t scalar, aclrtStream stream);
template void launchTADDSTestCase<16, uint64_t>(void* out, void* src, uint64_t scalar, aclrtStream stream);
template void launchTADDSTestCase<17, int64_t>(void* out, void* src, int64_t scalar, aclrtStream stream);
template void launchTADDSTestCase<18, int64_t>(void* out, void* src, int64_t scalar, aclrtStream stream);
template void launchTADDSTestCase<19, int64_t>(void* out, void* src, int64_t scalar, aclrtStream stream);
template void launchTADDSTestCase<20, int64_t>(void* out, void* src, int64_t scalar, aclrtStream stream);
template void launchTADDSTestCase<21, int64_t>(void* out, void* src, int64_t scalar, aclrtStream stream);
template void launchTADDSTestCase<1>(void*, void*, float, aclrtStream);
template void launchTADDSTestCase<2>(void*, void*, float, aclrtStream);
template void launchTADDSTestCase<3>(void*, void*, float, aclrtStream);
template void launchTADDSTestCase<4>(void*, void*, float, aclrtStream);
template void launchTADDSTestCase<5>(void*, void*, float, aclrtStream);
template void launchTADDSTestCase<6>(void*, void*, float, aclrtStream);
template void launchTADDSTestCase<7>(void*, void*, float, aclrtStream);
template void launchTADDSTestCase<8>(void*, void*, float, aclrtStream);
template void launchTADDSTestCase<9>(void*, void*, float, aclrtStream);
template void launchTADDSTestCase<10>(void*, void*, float, aclrtStream);
template void launchTADDSTestCase<11>(void*, void*, float, aclrtStream);
template void launchTADDSTestCase<22, uint64_t>(void*, void*, uint64_t, aclrtStream);
