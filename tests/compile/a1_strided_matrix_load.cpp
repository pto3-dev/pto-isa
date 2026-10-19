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
#ifndef PTO_A1_TEST_ROW_STRIDE
#define PTO_A1_TEST_ROW_STRIDE 32
#endif
constexpr int RowStride = PTO_A1_TEST_ROW_STRIDE;
constexpr int ColOffset = 16;
using MatA = Tile<TileType::Mat, half, Rows, 16, BLayout::ColMajor, Rows, 16, SLayout::RowMajor>;
using MatB = Tile<TileType::Mat, half, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor>;
using Left = TileLeft<half, Rows, 16>;
using Right = TileRight<half, 16, 16>;
using Acc = TileAcc<float, Rows, 16>;
using Vec = Tile<TileType::Vec, float, Rows, 16>;
using InputA = GlobalTensor<half, Shape<1, 1, 1, Rows, 16>, Stride<1, 1, 1, RowStride, 1>>;
using InputB = GlobalTensor<half, Shape<1, 1, 1, 16, 16>, Stride<1, 1, 1, RowStride, 1>>;
using Output = GlobalTensor<float, Shape<1, 1, 1, Rows, 16>, Stride<1, 1, 1, 16, 1>>;
extern "C" __global__ AICORE void A1StridedMatrixGemm(__gm__ float* output, __gm__ half* lhs, __gm__ half* rhs)
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
    InputA ga(lhs + ColOffset);
    InputB gb(rhs + ColOffset);
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
    std::vector<half> lhs(Rows * RowStride), rhs(16 * RowStride);
    std::vector<float> output(Rows * 16);
    for (int i = 0; i < Rows * RowStride; ++i)
        lhs[i] = i % 9 - 4;
    for (int i = 0; i < 16 * RowStride; ++i)
        rhs[i] = (i * 3) % 11 - 5;
    A1StridedMatrixGemm(output.data(), lhs.data(), rhs.data());
    for (int row = 0; row < Rows; ++row)
        for (int col = 0; col < 16; ++col) {
            float expected = 0;
            for (int k = 0; k < 16; ++k)
                expected += float(lhs[row * RowStride + ColOffset + k]) * float(rhs[k * RowStride + ColOffset + col]);
            if (output[row * 16 + col] != expected)
                return 1;
        }
    return 0;
}
#endif
