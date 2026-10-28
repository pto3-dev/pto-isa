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
#include <cstring>
#include <vector>
#endif
#ifndef PTO_ROW_ROWS
#define PTO_ROW_ROWS 16
#endif
#ifndef PTO_ROW_COLS
#define PTO_ROW_COLS 400
#endif
#ifdef PTO_ROW_FP16
using Element = half;
#else
using Element = float;
#endif
constexpr unsigned Rows = PTO_ROW_ROWS, Cols = PTO_ROW_COLS;
constexpr unsigned SourceBytes = Rows * sizeof(Element), GuardBytes = 16 * sizeof(Element);
constexpr unsigned OutputAddress = SourceBytes + GuardBytes;
using Column = pto::Tile<pto::TileType::Vec, Element, Rows, 1, pto::BLayout::ColMajor>;
using Matrix = pto::Tile<pto::TileType::Vec, Element, Rows, Cols>;
using Guard = pto::Tile<pto::TileType::Vec, Element, 1, 16>;
using GS = pto::GlobalTensor<Element, pto::Shape<1, 1, 1, Rows, 1>, pto::Stride<1, 1, 1, 1, 1>>;
using GD = pto::GlobalTensor<Element, pto::Shape<1, 1, 1, Rows, Cols>, pto::Stride<1, 1, 1, Cols, 1>>;
using GG = pto::GlobalTensor<Element, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<1, 1, 1, 16, 1>>;
extern "C" __global__ AICORE void A1RowBroadcast(__gm__ Element* output, __gm__ Element* input)
{
    Column column;
    Matrix matrix;
    Guard before, after;
    pto::TASSIGN<0>(column);
    pto::TASSIGN<SourceBytes>(before);
    pto::TASSIGN<OutputAddress>(matrix);
    pto::TASSIGN<OutputAddress + Rows * Cols * sizeof(Element)>(after);
    GS source(input);
    GD target(output);
    GG gb(output - 16), ga(output + Rows * Cols);
    pto::TLOAD(column, source);
    pto::TLOAD(before, gb);
    pto::TLOAD(after, ga);
#ifndef __CPU_SIM
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    pto::TROWEXPAND(matrix, column);
#ifndef __CPU_SIM
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    pto::TSTORE(target, matrix);
    pto::TSTORE(gb, before);
    pto::TSTORE(ga, after);
}
#ifdef __CPU_SIM
int main()
{
#ifdef PTO_ROW_FP16
    const uint32_t patterns[] = {0,      0x8000, 0x3c00, 0xbc00, 1,      0x8001, 0x7bff,
                                 0xfbff, 0x7c00, 0xfc00, 0x7e01, 0xfe05, 0x7c01, 0xfc01};
#else
    const uint32_t patterns[] = {0,          0x80000000, 0x3f800000, 0xbf800000, 1,          0x80000001, 0x7f7fffff,
                                 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00001, 0xffc00005, 0x7f800001, 0xff800001};
#endif
    std::vector<Element> input(Rows), output(Rows * Cols + 32, Element(7));
    for (unsigned row = 0; row < Rows; ++row) {
        auto bits = patterns[row % 14];
        std::memcpy(&input[row], &bits, sizeof(Element));
    }
    A1RowBroadcast(output.data() + 16, input.data());
    for (unsigned row = 0; row < Rows; ++row)
        for (unsigned col = 0; col < Cols; ++col)
            if (std::memcmp(&input[row], &output[16 + row * Cols + col], sizeof(Element)))
                return 1;
    for (unsigned i = 0; i < 16; ++i)
        if (output[i] != Element(7) || output[16 + Rows * Cols + i] != Element(7))
            return 2;
    return 0;
}
#endif
