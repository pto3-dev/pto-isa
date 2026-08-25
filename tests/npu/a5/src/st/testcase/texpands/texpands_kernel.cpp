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

using namespace pto;

#define PAD_VALUE_NULL (-100)
#define PAD_VALUE_MAX (1)

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kVRows_, int kVCols_, int padValueType>
__global__ AICORE void runTEXPANDS(__gm__ T* out, __gm__ T* scalar)
{
    T src = *scalar;
    constexpr bool isColMajor = ((kTRows_ * sizeof(T) % 32) == 0);
    constexpr int stride3 = isColMajor ? 1 : kGCols_;
    constexpr int stride4 = isColMajor ? kGRows_ : 1;
    constexpr Layout lay = isColMajor ? Layout::DN : Layout::ND;
    constexpr BLayout bLay = isColMajor ? BLayout::ColMajor : BLayout::RowMajor;
    constexpr PadValue padType = (padValueType == PAD_VALUE_NULL) ? PadValue::Null : PadValue::Max;

    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<kGRows_ * kGCols_, kGRows_ * kGCols_, kGRows_ * kGCols_, stride3, stride4>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5, lay>;
    GlobalData dstGlobal(out);

    using TileData =
        Tile<TileType::Vec, T, kTRows_, kTCols_, bLay, -1, -1, SLayout::NoneBox, TileConfig::fractalABSize, padType>;

    TileData dstTile(kVRows_, kVCols_);
    TASSIGN(dstTile, 0x0);

    Event<Op::TEXPANDS, Op::TSTORE_VEC> event = TEXPANDS(dstTile, src);
    TSTORE(dstGlobal, dstTile, event);
}

template <typename T, int cols, int padValueType>
__global__ AICORE void runTEXPANDSWideInt64(__gm__ T* out, __gm__ T* scalar)
{
    constexpr int tileRows = 1;
    constexpr int tileCols = 64;
    constexpr PadValue padType = (padValueType == PAD_VALUE_NULL) ? PadValue::Null : PadValue::Max;
    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using DynStridDim5 = pto::Stride<1, 1, 1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<
        TileType::Vec, T, tileRows, tileCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, TileConfig::fractalABSize,
        padType>;

    T src = *scalar;
    for (int col = 0; col < cols; col += tileCols) {
        int validCols = (col + tileCols <= cols) ? tileCols : (cols - col);
        GlobalData dstGlobal(out + col, DynShapeDim5(tileRows, validCols), DynStridDim5(cols, 1));
        TileData dstTile(tileRows, validCols);
        TASSIGN(dstTile, 0x0);
        Event<Op::TEXPANDS, Op::TSTORE_VEC> event = TEXPANDS(dstTile, src);
        TSTORE(dstGlobal, dstTile, event);
        pipe_barrier(PIPE_ALL);
    }
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kVRows_, int kVCols_, int padValueType>
__global__ AICORE void runTEXPANDSInplace(__gm__ T* out, __gm__ T* scalar)
{
    T src = *scalar;
    constexpr int stride3 = kGCols_;
    constexpr int stride4 = 1;
    constexpr PadValue padType = (padValueType == PAD_VALUE_NULL) ? PadValue::Null : PadValue::Max;

    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<kGRows_ * kGCols_, kGRows_ * kGCols_, kGRows_ * kGCols_, stride3, stride4>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    GlobalData dstGlobal(out);
    GlobalData srcGlobal(out);

    using TileData = Tile<
        TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1, SLayout::NoneBox, TileConfig::fractalABSize,
        padType>;

    TileData src0Tile(kVRows_, kVCols_);
    TileData dstTile(kVRows_, kVCols_);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(dstTile, 0x0);

    TLOAD(src0Tile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TEXPANDS(dstTile, src);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
}

template <typename T, int cols, int padValueType>
__global__ AICORE void runTEXPANDSInplaceWideInt64(__gm__ T* out, __gm__ T* scalar)
{
    constexpr int tileRows = 1;
    constexpr int tileCols = 64;
    constexpr PadValue padType = (padValueType == PAD_VALUE_NULL) ? PadValue::Null : PadValue::Max;
    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using DynStridDim5 = pto::Stride<1, 1, 1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<
        TileType::Vec, T, tileRows, tileCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, TileConfig::fractalABSize,
        padType>;

    T src = *scalar;
    for (int col = 0; col < cols; col += tileCols) {
        int validCols = (col + tileCols <= cols) ? tileCols : (cols - col);
        GlobalData dstGlobal(out + col, DynShapeDim5(tileRows, validCols), DynStridDim5(cols, 1));
        GlobalData srcGlobal(out + col, DynShapeDim5(tileRows, validCols), DynStridDim5(cols, 1));
        TileData src0Tile(tileRows, validCols);
        TileData dstTile(tileRows, validCols);
        TASSIGN(src0Tile, 0x0);
        TASSIGN(dstTile, 0x0);

        TLOAD(src0Tile, srcGlobal);
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
        TEXPANDS(dstTile, src);
#ifndef __PTO_AUTO__
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
        TSTORE(dstGlobal, dstTile);
        pipe_barrier(PIPE_ALL);
    }
}

template <
    typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kVRows_, int kVCols_, int padValueType,
    bool isBf16>
void LaunchTExpandSInplace(void* out, void* scalar, void* stream)
{
    constexpr int wideTileCols = 64;
    if constexpr (
        (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) && kGRows_ == 1 && kTRows_ == 1 && kVRows_ == 1 &&
        kGCols_ == kVCols_ && kTCols_ == kVCols_ && kVCols_ > wideTileCols) {
        runTEXPANDSInplaceWideInt64<T, kVCols_, padValueType><<<1, nullptr, stream>>>((T*)out, (T*)scalar);
    } else if constexpr (isBf16) {
        runTEXPANDSInplace<bfloat16_t, kGRows_, kGCols_, kTRows_, kTCols_, kVRows_, kVCols_, padValueType>
            <<<1, nullptr, stream>>>((bfloat16_t*)out, (bfloat16_t*)scalar);
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        runTEXPANDSInplace<half, kGRows_, kGCols_, kTRows_, kTCols_, kVRows_, kVCols_, padValueType>
            <<<1, nullptr, stream>>>((half*)out, (half*)scalar);
    } else {
        runTEXPANDSInplace<T, kGRows_, kGCols_, kTRows_, kTCols_, kVRows_, kVCols_, padValueType>
            <<<1, nullptr, stream>>>((T*)out, (T*)scalar);
    }
}

template <
    typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kVRows_, int kVCols_, int padValueType,
    bool isBf16>
void LaunchTExpandS(void* out, void* scalar, void* stream)
{
    constexpr int wideTileCols = 64;
    if constexpr (
        (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) && kGRows_ == 1 && kTRows_ == 1 && kVRows_ == 1 &&
        kGCols_ == kVCols_ && kTCols_ == kVCols_ && kVCols_ > wideTileCols) {
        runTEXPANDSWideInt64<T, kVCols_, padValueType><<<1, nullptr, stream>>>((T*)out, (T*)scalar);
    } else if constexpr (isBf16) {
        runTEXPANDS<bfloat16_t, kGRows_, kGCols_, kTRows_, kTCols_, kVRows_, kVCols_, padValueType>
            <<<1, nullptr, stream>>>((bfloat16_t*)out, (bfloat16_t*)scalar);
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        runTEXPANDS<half, kGRows_, kGCols_, kTRows_, kTCols_, kVRows_, kVCols_, padValueType>
            <<<1, nullptr, stream>>>((half*)out, (half*)scalar);
    } else {
        runTEXPANDS<T, kGRows_, kGCols_, kTRows_, kTCols_, kVRows_, kVCols_, padValueType>
            <<<1, nullptr, stream>>>((T*)out, (T*)scalar);
    }
}

template void LaunchTExpandS<float, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL, false>(
    void* out, void* scalar, void* stream);
template void LaunchTExpandS<int32_t, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL, false>(
    void* out, void* scalar, void* stream);
template void LaunchTExpandS<uint16_t, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL, false>(
    void* out, void* scalar, void* stream);
template void LaunchTExpandS<uint16_t, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL, true>(
    void* out, void* scalar, void* stream);
template void LaunchTExpandS<int16_t, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL, false>(
    void* out, void* scalar, void* stream);

template void LaunchTExpandS<float, 60, 60, 64, 64, 60, 60, PAD_VALUE_MAX, false>(
    void* out, void* scalar, void* stream);
template void LaunchTExpandS<int32_t, 60, 60, 64, 64, 60, 60, PAD_VALUE_MAX, false>(
    void* out, void* scalar, void* stream);
template void LaunchTExpandS<uint16_t, 1, 3600, 2, 4096, 1, 3600, PAD_VALUE_MAX, false>(
    void* out, void* scalar, void* stream);
template void LaunchTExpandS<uint16_t, 1, 3600, 2, 4096, 1, 3600, PAD_VALUE_MAX, true>(
    void* out, void* scalar, void* stream);
template void LaunchTExpandS<int16_t, 16, 200, 20, 512, 16, 200, PAD_VALUE_MAX, false>(
    void* out, void* scalar, void* stream);
template void LaunchTExpandS<int16_t, 1, 200, 1, 512, 1, 200, PAD_VALUE_MAX, false>(
    void* out, void* scalar, void* stream);
template void LaunchTExpandS<int64_t, 5, 16, 5, 16, 5, 16, PAD_VALUE_NULL, false>(
    void* out, void* scalar, void* stream);
template void LaunchTExpandS<uint64_t, 5, 16, 5, 16, 5, 16, PAD_VALUE_NULL, false>(
    void* out, void* scalar, void* stream);
template void LaunchTExpandS<int64_t, 5, 64, 5, 64, 5, 64, PAD_VALUE_NULL, false>(
    void* out, void* scalar, void* stream);
template void LaunchTExpandS<uint64_t, 5, 64, 5, 64, 5, 64, PAD_VALUE_NULL, false>(
    void* out, void* scalar, void* stream);
template void LaunchTExpandS<int64_t, 1, 32732, 1, 32732, 1, 32732, PAD_VALUE_NULL, false>(
    void* out, void* scalar, void* stream);
template void LaunchTExpandS<uint64_t, 1, 32732, 1, 32732, 1, 32732, PAD_VALUE_NULL, false>(
    void* out, void* scalar, void* stream);
template void LaunchTExpandSInplace<int64_t, 4, 32, 4, 32, 4, 32, PAD_VALUE_NULL, false>(
    void* out, void* scalar, void* stream);
template void LaunchTExpandSInplace<uint64_t, 4, 32, 4, 32, 4, 32, PAD_VALUE_NULL, false>(
    void* out, void* scalar, void* stream);
template void LaunchTExpandSInplace<int64_t, 1, 1024, 1, 1024, 1, 1024, PAD_VALUE_NULL, false>(
    void* out, void* scalar, void* stream);
template void LaunchTExpandSInplace<int64_t, 4, 64, 4, 64, 4, 40, PAD_VALUE_NULL, false>(
    void* out, void* scalar, void* stream);
template void LaunchTExpandSInplace<int64_t, 1, 2048, 1, 2048, 1, 2045, PAD_VALUE_NULL, false>(
    void* out, void* scalar, void* stream);
