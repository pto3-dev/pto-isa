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
#include <cmath>
#include <vector>
#endif
#include <pto/pto-inst.hpp>

#if defined(PTO_A1_TEST_FP16)
using Element = half;
#else
using Element = float;
#endif
#if !defined(PTO_A1_TEST_ELEMENTS)
#define PTO_A1_TEST_ELEMENTS 16
#endif
constexpr int Count = PTO_A1_TEST_ELEMENTS;
using Tile = pto::Tile<pto::TileType::Vec, Element, 1, Count>;
using Global = pto::GlobalTensor<Element, pto::Shape<1, 1, 1, 1, Count>, pto::Stride<1, 1, 1, Count, 1>>;

extern "C" __global__ AICORE void A1ExpDiv(__gm__ Element* out, __gm__ Element* lhs, __gm__ Element* rhs)
{
    Tile a, b, c;
    pto::TASSIGN<0>(a);
    pto::TASSIGN<Count * sizeof(Element)>(b);
#if defined(PTO_A1_TEST_INPLACE)
    pto::TASSIGN<0>(c);
#else
    pto::TASSIGN<2 * Count * sizeof(Element)>(c);
#endif
    Global ga(lhs), gb(rhs), gc(out);
#if !defined(PTO_A1_TEST_INPLACE)
    using GuardTile = pto::Tile<pto::TileType::Vec, Element, 1, 16>;
    using GuardGlobal = pto::GlobalTensor<Element, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<1, 1, 1, 16, 1>>;
    GuardTile guard;
    GuardGlobal gg(out + Count);
    pto::TASSIGN<3 * Count * sizeof(Element)>(guard);
    pto::TLOAD(guard, gg);
#endif
    pto::TLOAD(a, ga);
    pto::TLOAD(b, gb);
#if !defined(__CPU_SIM)
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
#if defined(PTO_A1_TEST_DIV)
#if defined(PTO_A1_TEST_HIGH_PRECISION)
    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(c, a, b);
#else
    pto::TDIV(c, a, b);
#endif
#else
#if defined(PTO_A1_TEST_HIGH_PRECISION)
    pto::TEXP<pto::ExpAlgorithm::HIGH_PRECISION>(c, a);
#else
    pto::TEXP(c, a);
#endif
#endif
#if !defined(__CPU_SIM)
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    pto::TSTORE(gc, c);
#if !defined(PTO_A1_TEST_INPLACE)
    pto::TSTORE(gg, guard);
#endif
}

#if defined(__CPU_SIM)
int main()
{
    std::vector<Element> lhs(Count), rhs(Count), out(Count + 16, static_cast<Element>(7));
    for (int i = 0; i < Count; ++i) {
        lhs[i] = static_cast<Element>((i % 33 - 16) / 8.0f);
        rhs[i] = static_cast<Element>((i % 11 + 1) / 8.0f);
    }
    A1ExpDiv(out.data(), lhs.data(), rhs.data());
    for (int i = Count; i < Count + 16; ++i)
        if (out[i] != static_cast<Element>(7))
            return 1;
    for (int i = 0; i < Count; ++i) {
#if defined(PTO_A1_TEST_DIV)
        const float expected = static_cast<float>(lhs[i]) / static_cast<float>(rhs[i]);
#else
        const float expected = std::exp(static_cast<float>(lhs[i]));
#endif
        if (std::abs(static_cast<float>(out[i]) - expected) > 0.002f * std::abs(expected) + 1e-6f)
            return 1;
    }
}
#endif
