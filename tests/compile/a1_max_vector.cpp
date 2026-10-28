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
#ifndef PTO_MAX_ELEMENTS
#define PTO_MAX_ELEMENTS 80
#endif
#ifndef PTO_MAX_ALIAS
#define PTO_MAX_ALIAS 2
#endif
#ifdef PTO_MAX_FP16
using Element = half;
#elif defined(PTO_MAX_INT)
using Element = int32_t;
#else
using Element = float;
#endif
constexpr unsigned Count = PTO_MAX_ELEMENTS, Span = (Count + 32) * sizeof(Element);
constexpr unsigned Destination = PTO_MAX_ALIAS * Span + 16 * sizeof(Element);
using Tile = pto::Tile<pto::TileType::Vec, Element, 1, Count>;
using Guard = pto::Tile<pto::TileType::Vec, Element, 1, 16>;
using Global = pto::GlobalTensor<Element, pto::Shape<1, 1, 1, 1, Count>, pto::Stride<1, 1, 1, Count, 1>>;
using GuardGlobal = pto::GlobalTensor<Element, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<1, 1, 1, 16, 1>>;
extern "C" __global__ AICORE void A1Maximum(__gm__ Element* output, __gm__ Element* lhs, __gm__ Element* rhs)
{
    Tile a, b, result;
    Guard before, after;
    pto::TASSIGN<16 * sizeof(Element)>(a);
#ifdef PTO_MAX_SAME_INPUT
    pto::TASSIGN<16 * sizeof(Element)>(b);
#else
    pto::TASSIGN<Span + 16 * sizeof(Element)>(b);
#endif
    pto::TASSIGN<Destination>(result);
    pto::TASSIGN<Destination - 16 * sizeof(Element)>(before);
    pto::TASSIGN<Destination + Count * sizeof(Element)>(after);
    Global ga(lhs), gb(rhs), gc(output);
    GuardGlobal prefix(output - 16), suffix(output + Count);
    pto::TLOAD(a, ga);
#ifndef PTO_MAX_SAME_INPUT
    pto::TLOAD(b, gb);
#endif
    pto::TLOAD(before, prefix);
#ifdef __CPU_SIM
    pto::TLOAD(after, suffix);
    pto::TMAX(result, a, b);
    pto::TSTORE(gc, result);
#else
    pto::Event<pto::Op::TLOAD, pto::Op::TMAX> loaded = pto::TLOAD(after, suffix);
    pto::Event<pto::Op::TMAX, pto::Op::TSTORE_VEC> calculated = pto::TMAX(result, a, b, loaded);
    pto::TSTORE(gc, result, calculated);
#endif
    pto::TSTORE(prefix, before);
    pto::TSTORE(suffix, after);
}
#ifdef __CPU_SIM
int main()
{
    std::vector<Element> lhs(Count), rhs(Count), output(Count + 32, Element(7));
    for (unsigned i = 0; i < Count; ++i) {
        lhs[i] = Element((int(i % 17) - 8) / 8.0f);
#ifdef PTO_MAX_SAME_INPUT
        rhs[i] = lhs[i];
#else
        rhs[i] = Element((int((i * 5) % 13) - 6) / 8.0f);
#endif
    }
    A1Maximum(output.data() + 16, lhs.data(), rhs.data());
    for (unsigned i = 0; i < Count; ++i)
        if (output[16 + i] != (lhs[i] > rhs[i] ? lhs[i] : rhs[i]))
            return 1;
    for (unsigned i = 0; i < 16; ++i)
        if (output[i] != Element(7) || output[16 + Count + i] != Element(7))
            return 2;
    return 0;
}
#endif
