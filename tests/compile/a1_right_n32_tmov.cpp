/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <cstdint>
#include <pto/pto-inst.hpp>
#if defined(__CPU_SIM)
#include <vector>
#endif
#ifndef TEST_M
#define TEST_M 32
#endif
#ifndef TEST_K
#define TEST_K 64
#endif
#ifndef TEST_GAP
#define TEST_GAP 0
#endif
#ifdef TEST_ZZ
constexpr auto BLayout = pto::BLayout::RowMajor;
#else
constexpr auto BLayout = pto::BLayout::ColMajor;
#endif
constexpr int M = TEST_M, K = TEST_K, Stride = 32 + TEST_GAP;
using MA = pto::Tile<pto::TileType::Mat, half, M, K, pto::BLayout::RowMajor, M, K, pto::SLayout::RowMajor, 512>;
using MB = pto::Tile<pto::TileType::Mat, half, K, 32, BLayout, K, 32, pto::SLayout::RowMajor, 512>;
using LA = pto::Tile<pto::TileType::Left, half, M, K, pto::BLayout::RowMajor, M, K, pto::SLayout::RowMajor, 512>;
using RB = pto::TileRight<half, K, 32>;
using ACC = pto::TileAcc<float, M, 32>;
using VEC = pto::Tile<pto::TileType::Vec, float, M, 16>;
using GA = pto::GlobalTensor<half, pto::Shape<1, 1, 1, M, K>, pto::Stride<1, 1, 1, K, 1>>;
using GB = pto::GlobalTensor<half, pto::Shape<1, 1, 1, K, 32>, pto::Stride<1, 1, 1, Stride, 1>>;
using GC = pto::GlobalTensor<float, pto::Shape<1, 1, 1, M, 16>, pto::Stride<1, 1, 1, 32, 1>>;

AICORE void TestRightN32(__gm__ float* out, __gm__ half* a, __gm__ half* b)
{
    MA ma;
    MB mb;
    LA left;
    RB right;
    ACC acc;
    VEC first, second;
    pto::TASSIGN<0>(ma);
    pto::TASSIGN<M * K * sizeof(half)>(mb);
    pto::TASSIGN<0>(left);
    pto::TASSIGN<0>(right);
    pto::TASSIGN<0>(acc);
    pto::TASSIGN<0>(first);
    pto::TASSIGN<M * 16 * sizeof(float)>(second);
    GA ga(a);
    GB gb(b);
    pto::TLOAD(ma, ga);
    pto::TLOAD(mb, gb);
#if !defined(__CPU_SIM)
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
    pto::TMOV(left, ma);
    pto::TMOV(right, mb);
#if !defined(__CPU_SIM)
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#if defined(TEST_PUBLIC_N32)
    pto::TMATMUL(acc, left, right);
#if defined(TEST_ACCUMULATE)
    pto::TMATMUL_ACC(acc, acc, left, right);
#endif
#else
    mad(acc.data(), left.data(), right.data(), M, K, 32, true);
#endif
    set_flag(PIPE_M, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_V, EVENT_ID0);
    copy_matrix_cc_to_ubuf(first.data(), acc.data(), 0, 1, (M / 16) * 2, 0, 0, CRMODE_NONE);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    GC out0(out), out1(out + 16);
    pto::TSTORE(out0, first);
    pto::TSTORE(out1, second);
#else
    pto::TMATMUL(acc, left, right);
#if defined(TEST_ACCUMULATE)
    pto::TMATMUL_ACC(acc, acc, left, right);
#endif
    for (int row = 0; row < M; ++row)
        for (int col = 0; col < 32; ++col)
            out[row * 32 + col] = acc.data()[pto::GetTileElementOffset<ACC>(row, col)];
#endif
}
#if defined(__CPU_SIM)
int main()
{
    std::vector<half> a(M * K + 32, half(7)), b(K * Stride + 32, half(7));
    std::vector<float> out(M * 32 + 32, 7);
    for (int row = 0; row < M; ++row)
        for (int col = 0; col < K; ++col)
            a[16 + row * K + col] = half(((row * 11 + col * 7) % 17 - 8) / 16.0f);
    for (int row = 0; row < K; ++row)
        for (int col = 0; col < 32; ++col)
            b[16 + row * Stride + col] = half(((row * 5 + col * 3) % 17 - 8) / 16.0f);
    TestRightN32(out.data() + 16, a.data() + 16, b.data() + 16);
    for (int row = 0; row < M; ++row)
        for (int col = 0; col < 32; ++col) {
            float expected = 0;
            for (int k = 0; k < K; ++k)
                expected += float(a[16 + row * K + k]) * float(b[16 + k * Stride + col]);
#if defined(TEST_ACCUMULATE)
            expected *= 2;
#endif
            if (out[16 + row * 32 + col] != expected)
                return 1;
        }
    for (int i = 0; i < 16; ++i)
        if (out[i] != 7 || out[16 + M * 32 + i] != 7)
            return 2;
    return 0;
}
#else
#include "tensor.h"
extern "C" __global__ AICORE void NativeRightN32(__gm__ int64_t* args)
{
    auto* a = reinterpret_cast<__gm__ Tensor*>(args[0]);
    auto* b = reinterpret_cast<__gm__ Tensor*>(args[1]);
    auto* out = reinterpret_cast<__gm__ Tensor*>(args[2]);
    TestRightN32(
        reinterpret_cast<__gm__ float*>(out->buffer.addr) + 16, reinterpret_cast<__gm__ half*>(a->buffer.addr) + 16,
        reinterpret_cast<__gm__ half*>(b->buffer.addr) + 16);
}

#endif
