/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

using namespace pto;

template <typename T, int kRows, int kCols>
__global__ AICORE void runTMOV_nd2zn(__gm__ T __out__* out, __gm__ T __in__* src)
{
    constexpr int k0 = BLOCK_BYTE_SIZE / sizeof(T);

    using SrcShape = Shape<1, 1, 1, kRows, kCols>;
    using SrcStride = pto::Stride<1, 1, 1, kCols, 1>;
    using SrcGlobal = GlobalTensor<T, SrcShape, SrcStride>;
    using DstShape = Shape<1, 1, 1, kRows, kCols>;
    using DstStride = pto::Stride<1, 1, 1, kCols, 1>;
    using DstGlobal = GlobalTensor<T, DstShape, DstStride>;

    using SrcTile = Tile<TileType::Vec, T, kRows, kCols, BLayout::RowMajor, -1, -1>;
    using ZnTile = Tile<TileType::Vec, T, kRows, kCols, BLayout::RowMajor, -1, -1, SLayout::ColMajor>;
    using NdTile = Tile<TileType::Vec, T, kRows, kCols, BLayout::RowMajor, -1, -1>;

    SrcTile srcTile(kRows, kCols);
    ZnTile znTile(kRows, kCols);
    NdTile ndTile(kRows, kCols);
    TASSIGN(srcTile, 0x0);
    TASSIGN(znTile, 0x10000);
    TASSIGN(ndTile, 0x10000);

    SrcGlobal srcGlobal(src);
    DstGlobal dstGlobal(out);

    TLOAD(srcTile, srcGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    TMOV<ZnTile, SrcTile>(znTile, srcTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(dstGlobal, ndTile);
    out = dstGlobal.data();
}

template <int kRows, int kCols>
void launchTMOV_nd2zn_hif8(uint8_t* out, uint8_t* src, void* stream)
{
    runTMOV_nd2zn<hifloat8_t, kRows, kCols><<<1, nullptr, stream>>>((hifloat8_t*)out, (hifloat8_t*)src);
}

template <int kRows, int kCols>
void launchTMOV_nd2zn_half(uint16_t* out, uint16_t* src, void* stream)
{
    runTMOV_nd2zn<half, kRows, kCols><<<1, nullptr, stream>>>((half*)out, (half*)src);
}

template <int kRows, int kCols>
void launchTMOV_nd2zn_b32(uint32_t* out, uint32_t* src, void* stream)
{
    runTMOV_nd2zn<int32_t, kRows, kCols><<<1, nullptr, stream>>>((int32_t*)out, (int32_t*)src);
}

template void launchTMOV_nd2zn_hif8<32, 32>(uint8_t*, uint8_t*, void*);
template void launchTMOV_nd2zn_hif8<32, 64>(uint8_t*, uint8_t*, void*);
template void launchTMOV_nd2zn_hif8<64, 64>(uint8_t*, uint8_t*, void*);
template void launchTMOV_nd2zn_hif8<128, 128>(uint8_t*, uint8_t*, void*);
template void launchTMOV_nd2zn_half<32, 32>(uint16_t*, uint16_t*, void*);
template void launchTMOV_nd2zn_half<32, 64>(uint16_t*, uint16_t*, void*);
template void launchTMOV_nd2zn_half<64, 64>(uint16_t*, uint16_t*, void*);
template void launchTMOV_nd2zn_half<128, 128>(uint16_t*, uint16_t*, void*);
template void launchTMOV_nd2zn_b32<32, 32>(uint32_t*, uint32_t*, void*);
template void launchTMOV_nd2zn_b32<32, 64>(uint32_t*, uint32_t*, void*);
template void launchTMOV_nd2zn_b32<64, 64>(uint32_t*, uint32_t*, void*);
template void launchTMOV_nd2zn_b32<128, 128>(uint32_t*, uint32_t*, void*);
