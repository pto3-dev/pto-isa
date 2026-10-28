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
#ifdef __CPU_SIM
#include <vector>
#endif
#ifndef PTO_TEST_M
#define PTO_TEST_M 32
#endif
#ifndef PTO_TEST_K
#define PTO_TEST_K 144
#endif
#ifndef PTO_TEST_GAP
#define PTO_TEST_GAP 0
#endif
constexpr int Rows = PTO_TEST_M, Reduction = PTO_TEST_K, AStride = Reduction + PTO_TEST_GAP;
constexpr int BStride = 16 + PTO_TEST_GAP;
#ifdef PTO_TEST_ZZ
constexpr auto InputLayout = pto::BLayout::RowMajor;
#else
constexpr auto InputLayout = pto::BLayout::ColMajor;
#endif
#ifdef PTO_TEST_LEGACY_LEFT
constexpr auto LeftLayout = pto::BLayout::ColMajor;
#else
constexpr auto LeftLayout = pto::BLayout::RowMajor;
#endif
using MatA =
    pto::Tile<pto::TileType::Mat, half, Rows, Reduction, InputLayout, Rows, Reduction, pto::SLayout::RowMajor, 512>;
using MatB =
    pto::Tile<pto::TileType::Mat, half, Reduction, 16, InputLayout, Reduction, 16, pto::SLayout::RowMajor, 512>;
using Left =
    pto::Tile<pto::TileType::Left, half, Rows, Reduction, LeftLayout, Rows, Reduction, pto::SLayout::RowMajor, 512>;
using Right = pto::TileRight<half, Reduction, 16>;
using Acc = pto::TileAcc<float, Rows, 16>;
using Vec = pto::Tile<pto::TileType::Vec, float, Rows, 16>;
using GA = pto::GlobalTensor<half, pto::Shape<1, 1, 1, Rows, Reduction>, pto::Stride<1, 1, 1, AStride, 1>>;
using GB = pto::GlobalTensor<half, pto::Shape<1, 1, 1, Reduction, 16>, pto::Stride<1, 1, 1, BStride, 1>>;
using GC = pto::GlobalTensor<float, pto::Shape<1, 1, 1, Rows, 16>, pto::Stride<1, 1, 1, 16, 1>>;
extern "C" __global__ AICORE void A1TiledGemm(__gm__ float* output, __gm__ half* a, __gm__ half* b)
{
    MatA ma;
    MatB mb;
    Left left;
    Right right;
    Acc acc;
    Vec result;
    pto::TASSIGN<0>(ma);
    pto::TASSIGN<Rows * Reduction * sizeof(half)>(mb);
    pto::TASSIGN<0>(left);
    pto::TASSIGN<0>(right);
    pto::TASSIGN<0>(acc);
    pto::TASSIGN<0>(result);
    GA ga(a);
    GB gb(b);
    GC gc(output);
#ifdef PTO_TEST_UB_INPUT
    // Build the operands in UB first. Doubling A makes the V->MTE3 edge
    // observable; the result reuses UB only after the Cube pipeline completes.
    using InputA = pto::Tile<pto::TileType::Vec, half, Rows, Reduction>;
    using InputB = pto::Tile<pto::TileType::Vec, half, Reduction, 16>;
    InputA va;
    InputB vb;
    pto::TASSIGN<0>(va);
    pto::TASSIGN<Rows * Reduction * sizeof(half)>(vb);
    pto::TLOAD(va, ga);
    pto::TLOAD(vb, gb);
#ifndef __CPU_SIM
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    pto::TADD(va, va, va);
#ifndef __CPU_SIM
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    pto::TMOV(ma, va);
    pto::TMOV(mb, vb);
#ifndef __CPU_SIM
    set_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
#endif
#else
    pto::TLOAD(ma, ga);
    pto::TLOAD(mb, gb);
#endif
#ifndef __CPU_SIM
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
    pto::TMOV(left, ma);
    pto::TMOV(right, mb);
#ifndef __CPU_SIM
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif
    pto::TMATMUL(acc, left, right);
#ifdef PTO_TEST_ACCUMULATE
    pto::TMATMUL_ACC(acc, acc, left, right);
#endif
#ifndef __CPU_SIM
    set_flag(PIPE_M, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_V, EVENT_ID0);
    pto::TMOV<Vec, Acc, pto::AccToVecMode::SingleModeVec0>(result, acc);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pto::TSTORE(gc, result);
#else
    for (int row = 0; row < Rows; ++row)
        for (int col = 0; col < 16; ++col)
            output[row * 16 + col] = acc.data()[pto::GetTileElementOffset<Acc>(row, col)];
#endif
}
#ifdef __CPU_SIM
int main()
{
    std::vector<half> a(Rows * AStride + 32, half(7)), b(Reduction * BStride + 32, half(7));
    std::vector<float> output(Rows * 16 + 32, 7);
    for (int row = 0; row < Rows; ++row)
        for (int k = 0; k < Reduction; ++k)
            a[16 + row * AStride + k] = half(((row * 11 + k * 7) % 17 - 8) / 16.0f);
    for (int k = 0; k < Reduction; ++k)
        for (int col = 0; col < 16; ++col)
            b[16 + k * BStride + col] = half(((k * 5 + col * 3) % 17 - 8) / 16.0f);
    A1TiledGemm(output.data() + 16, a.data() + 16, b.data() + 16);
    for (int row = 0; row < Rows; ++row) {
        for (int col = 0; col < 16; ++col) {
            float expected = 0;
            for (int k = 0; k < Reduction; ++k)
                expected += float(a[16 + row * AStride + k]) * float(b[16 + k * BStride + col]);
#ifdef PTO_TEST_UB_INPUT
            expected *= 2;
#endif
#ifdef PTO_TEST_ACCUMULATE
            expected *= 2;
#endif
            if (output[16 + row * 16 + col] != expected)
                return 1;
        }
    }
    for (int i = 0; i < 16; ++i)
        if (output[i] != 7 || output[16 + Rows * 16 + i] != 7)
            return 2;
    return 0;
}
#endif
