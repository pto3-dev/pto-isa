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
#ifndef PTO_A1_TEST_ROWS
#define PTO_A1_TEST_ROWS 16
#endif
#if defined(PTO_A1_TEST_FP16)
using Element = half;
#else
using Element = float;
#endif
constexpr int Rows = PTO_A1_TEST_ROWS;
using Column = pto::Tile<pto::TileType::Vec, Element, Rows, 1, pto::BLayout::ColMajor>;
using Shape = pto::Shape<1, 1, 1, Rows, 1>;
using Stride = pto::Stride<1, 1, 1, 1, 1>;
using Input = pto::GlobalTensor<Element, Shape, Stride>;
#if defined(PTO_A1_TEST_DN)
using Output = pto::GlobalTensor<Element, Shape, Stride, pto::Layout::DN>;
#else
using Output = pto::GlobalTensor<Element, Shape, Stride>;
#endif
extern "C" __global__ AICORE void A1PackedColumnDMA(__gm__ Element* out, __gm__ Element* input)
{
    Column column;
    pto::TASSIGN<0>(column);
    Input gi(input);
    Output go(out + 16);
    pto::TLOAD(column, gi);
#if !defined(__CPU_SIM)
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
#endif
    pto::TSTORE(go, column);
}
#if defined(__CPU_SIM)
int main()
{
    std::vector<Element> input(Rows), output(Rows + 32, static_cast<Element>(7));
    for (int i = 0; i < Rows; ++i)
        input[i] = static_cast<Element>((i % 17 - 8) / 16.0f);
    A1PackedColumnDMA(output.data(), input.data());
    for (int i = 0; i < Rows + 32; ++i) {
        const auto expected = i >= 16 && i < Rows + 16 ? input[i - 16] : static_cast<Element>(7);
        if (output[i] != expected)
            return 1;
    }
}
#endif
