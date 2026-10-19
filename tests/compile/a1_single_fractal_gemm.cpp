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
using namespace pto;
#if defined(PTO_A1_TEST_MATRIX_TYPE)
using MatrixData = float;
#else
using MatrixData = half;
#endif
#if defined(PTO_A1_TEST_MATRIX_NZ_INPUT)
constexpr BLayout InputBlockLayout = BLayout::ColMajor;
#else
constexpr BLayout InputBlockLayout = BLayout::RowMajor;
#endif
using Mat = Tile<TileType::Mat, MatrixData, 16, 16, InputBlockLayout, 16, 16, SLayout::RowMajor>;
using Left = TileLeft<half, 16, 16>;
#if defined(PTO_A1_TEST_MATRIX_SHAPE)
using Right = TileRight<half, 16, 32>;
#elif defined(PTO_A1_TEST_MATRIX_LAYOUT)
using Right = Tile<TileType::Right, half, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor>;
#else
using Right = TileRight<half, 16, 16>;
#endif
#if defined(PTO_A1_TEST_DUAL_MODE)
constexpr AccToVecMode WritebackMode = AccToVecMode::DualModeSplitM;
#else
constexpr AccToVecMode WritebackMode = AccToVecMode::SingleModeVec0;
#endif
using Acc = TileAcc<float, 16, 16>;
using Vec = Tile<TileType::Vec, float, 16, 16>;
using Input = GlobalTensor<half, Shape<1, 1, 1, 16, 16>, Stride<1, 1, 1, 16, 1>>;
using Output = GlobalTensor<float, Shape<1, 1, 1, 16, 16>, Stride<1, 1, 1, 16, 1>>;

extern "C" __global__ AICORE void A1SingleFractalGemm(__gm__ float* out, __gm__ half* lhs, __gm__ half* rhs)
{
    Mat ma, mb;
    Left a;
    Right b;
    Acc c;
    Vec v;
    TASSIGN<0>(ma);
    TASSIGN<512>(mb);
    TASSIGN<0>(a);
    TASSIGN<0>(b);
    TASSIGN<0>(c);
    TASSIGN<0>(v);
    Input ga(lhs), gb(rhs);
    Output gc(out);
    TLOAD(ma, ga);
#if defined(__CPU_SIM)
    TLOAD(mb, gb);
    TMOV(a, ma);
    TMOV(b, mb);
    TMATMUL(c, a, b);
    TMOV<Vec, Acc, WritebackMode>(v, c);
    TSTORE(gc, v);
#else
    Event<Op::TLOAD, Op::TMOV_M2L> loaded = TLOAD(mb, gb);
    TMOV(a, ma, loaded);
    Event<Op::TMOV_M2R, Op::TMATMUL> operands = TMOV(b, mb);
    Event<Op::TMATMUL, Op::TMOV_A2V> computed = TMATMUL(c, a, b, operands);
    Event<Op::TMOV_A2V, Op::TSTORE_VEC> moved = TMOV<Vec, Acc, WritebackMode>(v, c, computed);
#if defined(PTO_A1_TEST_DIRECT_ACC_STORE)
    TSTORE(gc, c, moved);
#else
    TSTORE(gc, v, moved);
#endif
#endif
}

#if defined(__CPU_SIM)
int main()
{
    std::vector<half> a(256), b(256);
    std::vector<float> out(256);
    for (int i = 0; i < 256; ++i) {
        a[i] = i % 9 - 4;
        b[i] = (i * 3) % 11 - 5;
    }
    A1SingleFractalGemm(out.data(), a.data(), b.data());
    for (int row = 0; row < 16; ++row)
        for (int col = 0; col < 16; ++col) {
            float expected = 0;
            for (int k = 0; k < 16; ++k)
                expected += float(a[row * 16 + k]) * float(b[k * 16 + col]);
            if (out[row * 16 + col] != expected)
                return 1;
        }
    return 0;
}
#endif
