/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T, int M, int N>
AICORE inline void runTLoadDn2Nz(__gm__ T* out, __gm__ T* src)
{
    using GlobalDataSrc =
        GlobalTensor<T, pto::Shape<1, 1, 1, M, N>, pto::Stride<1 * M * N, 1 * M * N, M * N, 1, M>, Layout::DN>;
    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, M, N>, pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;

    GlobalDataSrc srcGlobal(src);
    GlobalDataOut dstGlobal(out);

    using TileMatData = Tile<TileType::Mat, T, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    using VecTile = Tile<TileType::Vec, T, M, N, BLayout::RowMajor, M, N>;

    TileMatData matTile;
    VecTile vecTile;
    TASSIGN<0x0>(matTile);
    TASSIGN<0x0>(vecTile);

    TLOAD(matTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);

    TMOV(vecTile, matTile);
    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, vecTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTLoadDn2Nz_1(__gm__ uint8_t* out, __gm__ uint8_t* src)
{
    runTLoadDn2Nz<half, 16, 16>(reinterpret_cast<__gm__ half*>(out), reinterpret_cast<__gm__ half*>(src));
}

extern "C" __global__ AICORE void launchTLoadDn2Nz_2(__gm__ uint8_t* out, __gm__ uint8_t* src)
{
    runTLoadDn2Nz<half, 32, 32>(reinterpret_cast<__gm__ half*>(out), reinterpret_cast<__gm__ half*>(src));
}

extern "C" __global__ AICORE void launchTLoadDn2Nz_3(__gm__ uint8_t* out, __gm__ uint8_t* src)
{
    runTLoadDn2Nz<half, 32, 64>(reinterpret_cast<__gm__ half*>(out), reinterpret_cast<__gm__ half*>(src));
}

extern "C" __global__ AICORE void launchTLoadDn2Nz_4(__gm__ uint8_t* out, __gm__ uint8_t* src)
{
    runTLoadDn2Nz<half, 64, 96>(reinterpret_cast<__gm__ half*>(out), reinterpret_cast<__gm__ half*>(src));
}

template <int32_t tilingKey>
void launchTLoadDn2Nz(uint8_t* out, uint8_t* src, void* stream)
{
    if constexpr (tilingKey == 1) {
        launchTLoadDn2Nz_1<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == 2) {
        launchTLoadDn2Nz_2<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == 3) {
        launchTLoadDn2Nz_3<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == 4) {
        launchTLoadDn2Nz_4<<<1, nullptr, stream>>>(out, src);
    }
}

template void launchTLoadDn2Nz<1>(uint8_t* out, uint8_t* src, void* stream);
template void launchTLoadDn2Nz<2>(uint8_t* out, uint8_t* src, void* stream);
template void launchTLoadDn2Nz<3>(uint8_t* out, uint8_t* src, void* stream);
template void launchTLoadDn2Nz<4>(uint8_t* out, uint8_t* src, void* stream);
