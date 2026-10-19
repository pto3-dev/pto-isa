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

#if defined(PTO_A1_TEST_FP32)
using Data = float;
#else
using Data = half;
#endif
#if defined(PTO_A1_TEST_LARGE)
constexpr int Cols = (255 * 256 + 32) / sizeof(Data);
#elif defined(PTO_A1_TEST_HIGH_MASK)
constexpr int Cols = 160 / sizeof(Data);
#else
constexpr int Cols = 32 / sizeof(Data);
#endif
using TileData = pto::Tile<pto::TileType::Vec, Data, 1, Cols>;
using GlobalData = pto::GlobalTensor<Data, pto::Shape<1, 1, 1, 1, Cols>, pto::Stride<1, 1, 1, Cols, 1>>;

extern "C" __global__ AICORE void A1MaskedAdd(__gm__ Data* out, __gm__ Data* lhs, __gm__ Data* rhs)
{
    TileData a, b;
    pto::TASSIGN<0>(a);
    pto::TASSIGN<Cols * sizeof(Data)>(b);
    GlobalData ga(lhs), gb(rhs), gc(out), guard(out + Cols);
    pto::TLOAD(a, ga);
#if defined(__CPU_SIM)
    pto::TLOAD(b, gb);
    pto::TADD(a, a, b);
    pto::TSTORE(gc, a);
    pto::TSTORE(guard, b);
#else
    pto::Event<pto::Op::TLOAD, pto::Op::TADD> loaded = pto::TLOAD(b, gb);
    pto::Event<pto::Op::TADD, pto::Op::TSTORE_VEC> added = pto::TADD(a, a, b, loaded);
    pto::TSTORE(gc, a, added);
    pto::TSTORE(guard, b);
#endif
}

#if defined(__CPU_SIM)
int main()
{
    std::vector<Data> lhs(Cols), rhs(Cols), out(2 * Cols);
    for (int i = 0; i < Cols; ++i) {
        lhs[i] = i % 17 - 8;
        rhs[i] = (i * 3) % 17 - 8;
    }
    A1MaskedAdd(out.data(), lhs.data(), rhs.data());
    for (int i = 0; i < Cols; ++i)
        if (out[i] != Data(lhs[i] + rhs[i]) || out[Cols + i] != rhs[i])
            return 1;
    return 0;
}
#endif
