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

using Data = float;
constexpr int Rows = 16;
constexpr int Cols = 128;
constexpr bool InPlace = false;
using Tile = pto::Tile<pto::TileType::Vec, Data, Rows, Cols>;
using Global = pto::GlobalTensor<Data, pto::Shape<1, 1, 1, Rows, Cols>, pto::Stride<1, 1, 1, Cols, 1>>;

extern "C" __global__ AICORE void A1Binary(__gm__ Data* out, __gm__ Data* lhs, __gm__ Data* rhs)
{
    Tile a, b, c;
    pto::TASSIGN<0>(a);
    pto::TASSIGN<Rows * Cols * sizeof(Data)>(b);
    pto::TASSIGN<InPlace ? 0 : 2 * Rows * Cols * sizeof(Data)>(c);
    Global ga(lhs), gb(rhs), gc(out);
    pto::TLOAD(a, ga);
#if defined(__CPU_SIM)
    pto::TLOAD(b, gb);
    pto::TMUL(c, a, b);
    pto::TSTORE(gc, c);
#else
    pto::Event<pto::Op::TLOAD, pto::Op::TMUL> loaded = pto::TLOAD(b, gb);
    pto::Event<pto::Op::TMUL, pto::Op::TSTORE_VEC> computed = pto::TMUL(c, a, b, loaded);
    pto::TSTORE(gc, c, computed);
#endif
}

#if defined(__CPU_SIM)
#include <iostream>
#include <vector>
int main()
{
    std::vector<Data> lhs(Rows * Cols), rhs(Rows * Cols), out(Rows * Cols);
    for (int i = 0; i < Rows * Cols; ++i) {
        lhs[i] = (i % 17) - 8;
        rhs[i] = ((i * 3) % 17) - 8;
    }
    A1Binary(out.data(), lhs.data(), rhs.data());
    for (int i = 0; i < Rows * Cols; ++i)
        if (out[i] != static_cast<Data>(lhs[i] * rhs[i]))
            return 1;
    std::cout << "CPU reference passed\n";
}
#endif
