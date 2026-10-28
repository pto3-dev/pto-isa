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
#ifndef PTO_TEST_M
#define PTO_TEST_M 32
#endif
#ifndef PTO_TEST_K
#define PTO_TEST_K 64
#endif
#ifndef PTO_TEST_PANEL_ROWS
#define PTO_TEST_PANEL_ROWS 16
#endif
#ifndef PTO_TEST_GAP
#define PTO_TEST_GAP 0
#endif
constexpr int Rows = PTO_TEST_M, Reduction = PTO_TEST_K, PanelRows = PTO_TEST_PANEL_ROWS;
#ifdef PTO_TEST_REPLICATE
static_assert(Reduction % 64 == 0);
constexpr int InputK = Reduction / 4;
#else
constexpr int InputK = Reduction;
#endif
constexpr int InputStride = InputK + PTO_TEST_GAP;
using Full = pto::Tile<
    pto::TileType::Mat, half, Rows, Reduction, pto::BLayout::ColMajor, Rows, Reduction, pto::SLayout::RowMajor, 512>;
using Panel = pto::Tile<
    pto::TileType::Mat, half, PanelRows, 16, pto::BLayout::ColMajor, PanelRows, 16, pto::SLayout::RowMajor, 512>;
using MatB = pto::Tile<
    pto::TileType::Mat, half, Reduction, 16, pto::BLayout::ColMajor, Reduction, 16, pto::SLayout::RowMajor, 512>;
using Left = pto::Tile<
    pto::TileType::Left, half, Rows, Reduction, pto::BLayout::RowMajor, Rows, Reduction, pto::SLayout::RowMajor, 512>;
using Right = pto::TileRight<half, Reduction, 16>;
using Acc = pto::TileAcc<float, Rows, 16>;
using Vec = pto::Tile<pto::TileType::Vec, float, Rows, 16>;
using GP = pto::GlobalTensor<half, pto::Shape<1, 1, 1, PanelRows, 16>, pto::Stride<1, 1, 1, InputStride, 1>>;
using GB = pto::GlobalTensor<half, pto::Shape<1, 1, 1, Reduction, 16>, pto::Stride<1, 1, 1, 16, 1>>;
using GC = pto::GlobalTensor<float, pto::Shape<1, 1, 1, Rows, 16>, pto::Stride<1, 1, 1, 16, 1>>;
extern "C" __global__ AICORE void A1PanelInsert(__gm__ float* output, __gm__ half* a, __gm__ half* b)
{
    Full full;
    Panel panel;
    MatB mb;
    Left left;
    Right right;
    Acc acc;
    Vec result;
    pto::TASSIGN<0>(full);
    pto::TASSIGN<0>(panel);
    pto::TASSIGN<Rows * Reduction * sizeof(half)>(mb);
    pto::TASSIGN<0>(left);
    pto::TASSIGN<0>(right);
    pto::TASSIGN<0>(acc);
    pto::TASSIGN<0>(result);
#ifdef PTO_TEST_PATCH
    using GA = pto::GlobalTensor<half, pto::Shape<1, 1, 1, Rows, Reduction>, pto::Stride<1, 1, 1, InputStride, 1>>;
    GA ga(a);
    pto::TLOAD(full, ga);
#ifndef __CPU_SIM
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
    pto::TMOV(left, full);
#ifndef __CPU_SIM
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
#endif
    GP source(a);
    pto::TLOAD(panel, source);
#ifndef __CPU_SIM
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
    // A strict sub-region update must preserve all other rows and columns.
    pto::TINSERT(left, panel, 16, 16);
#else
    for (int row = 0; row < Rows; row += PanelRows) {
        for (int column = 0; column < Reduction; column += 16) {
#ifdef PTO_TEST_REPLICATE
            // Match the convolution loader: each C16 panel feeds four adjacent
            // K16 blocks in L0A before moving to the next input-channel panel.
            const int sourceColumn = column / 64 * 16;
#else
            const int sourceColumn = column;
#endif
            GP source(a + row * InputStride + sourceColumn);
            pto::TLOAD(panel, source);
#ifndef __CPU_SIM
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
            pto::TINSERT(left, panel, row, column);
#ifndef __CPU_SIM
            // L1 can be overwritten after MTE1 consumes it; L0A stays live.
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
#endif
        }
    }
#endif
    GB gb(b);
    pto::TLOAD(mb, gb);
#ifndef __CPU_SIM
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
    pto::TMOV(right, mb);
#ifndef __CPU_SIM
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif
    pto::TMATMUL(acc, left, right);
#ifndef __CPU_SIM
    set_flag(PIPE_M, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_V, EVENT_ID0);
    pto::TMOV<Vec, Acc, pto::AccToVecMode::SingleModeVec0>(result, acc);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    GC gc(output);
    pto::TSTORE(gc, result);
#else
    for (int row = 0; row < Rows; ++row)
        for (int col = 0; col < 16; ++col)
            output[row * 16 + col] = acc.GetElement(row, col);
#endif
}
#ifdef __CPU_SIM
int main()
{
    std::vector<half> a(Rows * InputStride + 32, half(7)), b(Reduction * 16 + 32, half(7));
    std::vector<float> output(Rows * 16 + 32, 7);
    for (int row = 0; row < Rows; ++row)
        for (int k = 0; k < InputK; ++k)
            a[16 + row * InputStride + k] = half(((row * 11 + k * 7) % 17 - 8) / 16.0f);
    for (int k = 0; k < Reduction; ++k)
        for (int col = 0; col < 16; ++col)
            b[16 + k * 16 + col] = half(((k * 5 + col * 3) % 17 - 8) / 16.0f);
    A1PanelInsert(output.data() + 16, a.data() + 16, b.data() + 16);
    for (int row = 0; row < Rows; ++row) {
        for (int col = 0; col < 16; ++col) {
            float expected = 0;
            for (int k = 0; k < Reduction; ++k) {
                int sourceRow = row, sourceCol = k;
#ifdef PTO_TEST_REPLICATE
                sourceCol = k / 64 * 16 + k % 16;
#endif
#ifdef PTO_TEST_PATCH
                if (row >= 16 && row < 16 + PanelRows && k >= 16 && k < 32) {
                    sourceRow = row - 16;
                    sourceCol = k - 16;
                }
#endif
                expected += float(a[16 + sourceRow * InputStride + sourceCol]) * float(b[16 + k * 16 + col]);
            }
            if (output[16 + row * 16 + col] != expected)
                return 1;
        }
    }
    for (int i = 0; i < 16; ++i)
        if (output[i] != 7 || output[16 + Rows * 16 + i] != 7)
            return 2;
    return 0;
}
#endif
