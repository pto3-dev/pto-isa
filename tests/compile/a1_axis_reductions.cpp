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
#include <cmath>
#include <vector>
#endif
#include <pto/pto-inst.hpp>
#ifndef PTO_A1_TEST_ROWS
#define PTO_A1_TEST_ROWS 16
#endif
#ifndef PTO_A1_TEST_COLS
#define PTO_A1_TEST_COLS 80
#endif
#ifndef PTO_A1_TEST_REDUCTION
#define PTO_A1_TEST_REDUCTION 0
#endif
#if defined(PTO_A1_TEST_FP16)
using Element = half;
#else
using Element = float;
#endif
constexpr int Rows = PTO_A1_TEST_ROWS;
constexpr int Cols = PTO_A1_TEST_COLS;
constexpr bool RowReduction = PTO_A1_TEST_REDUCTION < 2;
constexpr bool Maximum = PTO_A1_TEST_REDUCTION % 2 == 1;
constexpr int OutRows = RowReduction ? Rows : 1;
constexpr int OutCols = RowReduction ? 1 : Cols;
constexpr int OutCount = OutRows * OutCols;
constexpr unsigned SourceBytes = Rows * Cols * sizeof(Element);
using Src = pto::Tile<pto::TileType::Vec, Element, Rows, Cols>;
using Dst = pto::Tile<
    pto::TileType::Vec, Element, OutRows, OutCols, RowReduction ? pto::BLayout::ColMajor : pto::BLayout::RowMajor>;
using SrcGlobal = pto::GlobalTensor<Element, pto::Shape<1, 1, 1, Rows, Cols>, pto::Stride<1, 1, 1, Cols, 1>>;
using DstGlobal = pto::GlobalTensor<Element, pto::Shape<1, 1, 1, OutRows, OutCols>, pto::Stride<1, 1, 1, OutCols, 1>>;
extern "C" __global__ AICORE void A1AxisReduce(__gm__ Element* out, __gm__ Element* input)
{
    Src src;
    Dst dst;
    pto::TASSIGN<0>(src);
#if defined(PTO_A1_TEST_COLUMN_ALIAS)
    static_assert(!RowReduction, "Column alias testing requires a column reduction.");
    constexpr unsigned outputAddress = 0;
    constexpr unsigned guardAddress = SourceBytes;
#else
    constexpr unsigned outputAddress = RowReduction ? 2 * SourceBytes : SourceBytes;
    constexpr unsigned guardAddress = outputAddress + OutCount * sizeof(Element);
#endif
    pto::TASSIGN<outputAddress>(dst);
    SrcGlobal gs(input);
    DstGlobal gd(out);
    using Guard = pto::Tile<pto::TileType::Vec, Element, 1, 16>;
    using GuardGlobal = pto::GlobalTensor<Element, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<1, 1, 1, 16, 1>>;
    Guard guard;
    GuardGlobal gg(out + OutCount);
    pto::TASSIGN<guardAddress>(guard);
    pto::TLOAD(src, gs);
    pto::TLOAD(guard, gg);
#if !defined(__CPU_SIM)
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
#if PTO_A1_TEST_REDUCTION < 2
    Src tmp;
    pto::TASSIGN<SourceBytes>(tmp);
#if PTO_A1_TEST_REDUCTION == 0
    pto::TROWSUM(dst, src, tmp);
#else
    pto::TROWMAX(dst, src, tmp);
#endif
#elif PTO_A1_TEST_REDUCTION == 2
    pto::TCOLSUM(dst, src);
#else
    pto::TCOLMAX(dst, src);
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
    std::vector<Element> input(Rows * Cols), out(OutCount + 16, static_cast<Element>(7));
    for (int i = 0; i < Rows * Cols; ++i)
        input[i] = static_cast<Element>((i % 17 - 8) / 16.0f);
    A1AxisReduce(out.data(), input.data());
    for (int index = 0; index < OutCount; ++index) {
        Element expected = static_cast<Element>(0);
        const int reduced = RowReduction ? Cols : Rows;
        for (int j = 0; j < reduced; ++j) {
            const auto value = input[RowReduction ? index * Cols + j : j * Cols + index];
            expected =
                Maximum ? (j == 0 || value > expected ? value : expected) : static_cast<Element>(expected + value);
        }
        if (std::abs(static_cast<float>(out[index]) - static_cast<float>(expected)) > 1e-4f)
            return 1;
    }
    for (int i = OutCount; i < OutCount + 16; ++i)
        if (out[i] != static_cast<Element>(7))
            return 1;
}
#endif
