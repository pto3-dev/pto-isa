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
#ifndef PTO_A1_TEST_ELEMENTS
#define PTO_A1_TEST_ELEMENTS 80
#endif
#ifndef PTO_A1_TEST_VALUE
#define PTO_A1_TEST_VALUE -2.5f
#endif
constexpr int Count = PTO_A1_TEST_ELEMENTS;
using Tile = pto::Tile<pto::TileType::Vec, Element, 1, Count>;
using Global = pto::GlobalTensor<Element, pto::Shape<1, 1, 1, 1, Count>, pto::Stride<1, 1, 1, Count, 1>>;
extern "C" __global__ AICORE void A1ScalarFill(__gm__ Element* out)
{
    Tile dst;
    pto::Tile<pto::TileType::Vec, Element, 1, 16> guard;
    Global global(out);
    pto::GlobalTensor<Element, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<1, 1, 1, 16, 1>> guardGlobal(out + Count);
    pto::TASSIGN<0>(dst);
    pto::TASSIGN<Count * sizeof(Element)>(guard);
    pto::TLOAD(guard, guardGlobal);
#if !defined(__CPU_SIM)
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    pto::TEXPANDS(dst, static_cast<Element>(PTO_A1_TEST_VALUE));
#if !defined(__CPU_SIM)
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    pto::TSTORE(global, dst);
    pto::TSTORE(guardGlobal, guard);
}
#if defined(__CPU_SIM)
int main()
{
    std::vector<Element> out(Count + 16, static_cast<Element>(7));
    A1ScalarFill(out.data());
    for (int i = 0; i < Count; ++i) {
        const float expected = static_cast<float>(static_cast<Element>(PTO_A1_TEST_VALUE));
        if (static_cast<float>(out[i]) != expected ||
            std::signbit(static_cast<float>(out[i])) != std::signbit(expected))
            return 1;
    }
    for (int i = Count; i < Count + 16; ++i)
        if (out[i] != static_cast<Element>(7))
            return 1;
}
#endif
