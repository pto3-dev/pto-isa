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
#include <algorithm>
#include <vector>
#endif
#include <pto/pto-inst.hpp>
#ifndef PTO_A1_TEST_ROWS
#define PTO_A1_TEST_ROWS 16
#endif
#ifndef PTO_A1_TEST_COLS
#define PTO_A1_TEST_COLS 80
#endif
#ifndef PTO_A1_TEST_EXPAND
#define PTO_A1_TEST_EXPAND 0
#endif
#if defined(PTO_A1_TEST_FP16)
using Element = half;
#else
using Element = float;
#endif
constexpr int Rows = PTO_A1_TEST_ROWS;
constexpr int Cols = PTO_A1_TEST_COLS;
constexpr bool Row = PTO_A1_TEST_EXPAND == 0;
constexpr unsigned Bytes = Rows * Cols * sizeof(Element);
using Matrix = pto::Tile<pto::TileType::Vec, Element, Rows, Cols>;
using Reduced = pto::Tile<
    pto::TileType::Vec, Element, Row ? Rows : 1, Row ? 1 : Cols, Row ? pto::BLayout::ColMajor : pto::BLayout::RowMajor>;
using Global = pto::GlobalTensor<Element, pto::Shape<1, 1, 1, Rows, Cols>, pto::Stride<1, 1, 1, Cols, 1>>;
extern "C" __global__ AICORE void A1ReductionBroadcast(__gm__ Element* out, __gm__ Element* input)
{
    Matrix src, dst, tmp;
    Reduced reduced;
    pto::TASSIGN<0>(src);
    pto::TASSIGN<Bytes>(tmp);
    constexpr unsigned reducedAddress = Row ? 2 * Bytes : Bytes;
    constexpr unsigned reducedBytes = (Row ? Rows : Cols) * sizeof(Element);
    constexpr unsigned outputAddress = PTO_A1_TEST_EXPAND == 2 ? reducedAddress : reducedAddress + reducedBytes;
    pto::TASSIGN<reducedAddress>(reduced);
    pto::TASSIGN<outputAddress>(dst);
    using Guard = pto::Tile<pto::TileType::Vec, Element, 1, 16>;
    using GuardGlobal = pto::GlobalTensor<Element, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<1, 1, 1, 16, 1>>;
    Guard guard;
    pto::TASSIGN<outputAddress + Bytes>(guard);
    Global gs(input), gd(out);
    GuardGlobal gg(out + Rows * Cols);
    pto::TLOAD(src, gs);
    pto::TLOAD(guard, gg);
#if !defined(__CPU_SIM)
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
#if PTO_A1_TEST_EXPAND == 0
#if defined(PTO_A1_TEST_SUM)
    pto::TROWSUM(reduced, src, tmp);
#else
    pto::TROWMAX(reduced, src, tmp);
#endif
#else
#if defined(PTO_A1_TEST_SUM)
    pto::TCOLSUM(reduced, src);
#else
    pto::TCOLMAX(reduced, src);
#endif
#endif
#if !defined(__CPU_SIM)
    pipe_barrier(PIPE_V);
#endif
#if PTO_A1_TEST_EXPAND == 0
    pto::TROWEXPAND(dst, reduced);
#else
    pto::TCOLEXPAND(dst, reduced);
#endif
#if !defined(__CPU_SIM)
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    pto::TSTORE(gd, dst);
    pto::TSTORE(gg, guard);
}
#if defined(__CPU_SIM)
int main()
{
    std::vector<Element> input(Rows * Cols), out(Rows * Cols + 16, static_cast<Element>(7));
    for (int i = 0; i < Rows * Cols; ++i)
        input[i] = static_cast<Element>((i % 17 - 8) / 16.0f);
    A1ReductionBroadcast(out.data(), input.data());
    for (int r = 0; r < Rows; ++r) {
        for (int c = 0; c < Cols; ++c) {
            Element expected = static_cast<Element>(0);
            for (int j = 0; j < (Row ? Cols : Rows); ++j) {
                const auto value = input[Row ? r * Cols + j : j * Cols + c];
#if defined(PTO_A1_TEST_SUM)
                expected = static_cast<Element>(expected + value);
#else
                expected = j == 0 || value > expected ? value : expected;
#endif
            }
            if (out[r * Cols + c] != expected)
                return 1;
        }
    }
    for (int i = Rows * Cols; i < Rows * Cols + 16; ++i)
        if (out[i] != static_cast<Element>(7))
            return 1;
}
#endif
