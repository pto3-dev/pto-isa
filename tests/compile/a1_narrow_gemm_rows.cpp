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
#if defined(__CPU_SIM)
#include <vector>
#endif
#include <pto/pto-inst.hpp>
using namespace pto;
#ifndef PTO_A1_TEST_ROWS
#define PTO_A1_TEST_ROWS 32
#endif
constexpr int Rows = PTO_A1_TEST_ROWS;
using MatA = Tile<TileType::Mat, half, Rows, 16, BLayout::ColMajor, Rows, 16, SLayout::RowMajor>;
using MatB = Tile<TileType::Mat, half, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor>;
using Left = TileLeft<half, Rows, 16>;
using Right = TileRight<half, 16, 16>;
using Acc = TileAcc<float, Rows, 16>;
using Vec = Tile<TileType::Vec, float, Rows, 16>;
using InputA = GlobalTensor<half, Shape<1, 1, 1, Rows, 16>, Stride<1, 1, 1, 16, 1>>;
using InputB = GlobalTensor<half, Shape<1, 1, 1, 16, 16>, Stride<1, 1, 1, 16, 1>>;
using Output = GlobalTensor<float, Shape<1, 1, 1, Rows, 16>, Stride<1, 1, 1, 16, 1>>;
extern "C" __global__ AICORE void A1NarrowGemm(__gm__ float* output, __gm__ half* lhs, __gm__ half* rhs)
{
    MatA ma;
    MatB mb;
    Left a;
    Right b;
    Acc c;
    Vec v;
    TASSIGN<0>(ma);
    TASSIGN<Rows * 16 * sizeof(half)>(mb);
    TASSIGN<0>(a);
    TASSIGN<0>(b);
    TASSIGN<0>(c);
    TASSIGN<0>(v);
    InputA ga(lhs);
    InputB gb(rhs);
    Output gc(output);
    TLOAD(ma, ga);
#if defined(__CPU_SIM)
    TLOAD(mb, gb);
    TMOV(a, ma);
    TMOV(b, mb);
    TMATMUL(c, a, b);
    TMOV<Vec, Acc, AccToVecMode::SingleModeVec0>(v, c);
    TSTORE(gc, v);
#else
    Event<Op::TLOAD, Op::TMOV_M2L> loaded = TLOAD(mb, gb);
    TMOV(a, ma, loaded);
    Event<Op::TMOV_M2R, Op::TMATMUL> moved = TMOV(b, mb);
    Event<Op::TMATMUL, Op::TMOV_A2V> computed = TMATMUL(c, a, b, moved);
    Event<Op::TMOV_A2V, Op::TSTORE_VEC> written = TMOV<Vec, Acc, AccToVecMode::SingleModeVec0>(v, c, computed);
    TSTORE(gc, v, written);
#endif
}
#if defined(__CPU_SIM)
int main()
{
    std::vector<half> lhs(Rows * 16), rhs(256);
    std::vector<float> output(Rows * 16);
    for (int i = 0; i < Rows * 16; ++i)
        lhs[i] = i % 9 - 4;
    for (int i = 0; i < 256; ++i)
        rhs[i] = (i * 3) % 11 - 5;
    A1NarrowGemm(output.data(), lhs.data(), rhs.data());
    for (int row = 0; row < Rows; ++row)
        for (int col = 0; col < 16; ++col) {
            float expected = 0;
            for (int k = 0; k < 16; ++k)
                expected += float(lhs[row * 16 + k]) * float(rhs[k * 16 + col]);
            if (output[row * 16 + col] != expected)
                return 1;
        }
    return 0;
}
#endif
