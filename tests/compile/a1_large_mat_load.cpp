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
#ifdef __CPU_SIM
#include <cstring>
#include <vector>
#endif
#include <pto/pto-inst.hpp>

#ifndef MAT_ROWS
#define MAT_ROWS 4608
#endif
#ifndef MAT_COLS
#define MAT_COLS 32
#endif
#ifndef MAT_STRIDE
#define MAT_STRIDE (MAT_COLS + 16)
#endif
#ifndef MAT_STAGE_MODE
#define MAT_STAGE_MODE 0
#endif
#ifndef MAT_ROW_MAJOR
#define MAT_ROW_MAJOR 0
#endif

using namespace pto;
constexpr int Rows = MAT_ROWS, Columns = MAT_COLS;
constexpr uint64_t RowStride = MAT_STRIDE;
using Matrix = Tile<
    TileType::Mat, half, Rows, Columns, MAT_ROW_MAJOR ? BLayout::RowMajor : BLayout::ColMajor, Rows, Columns,
    SLayout::RowMajor, 512>;
using Input = GlobalTensor<half, Shape<1, 1, 1, Rows, Columns>, Stride<1, 1, 1, RowStride, 1>>;

extern "C" __global__ AICORE void A1LargeMatLoad(__gm__ half* input)
{
    Matrix matrix;
    TASSIGN<0>(matrix);
    Input source(input + 16);
#if MAT_STAGE_MODE == 1
    using Staged = Tile<TileType::Vec, half, Rows, Columns>;
    Staged staged;
    TASSIGN<0>(staged);
    TLOAD(staged, source);
#ifndef __CPU_SIM
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
#endif
    TMOV(matrix, staged);
#else
    TLOAD(matrix, source);
#if MAT_STAGE_MODE == 2
    using Patch = Tile<TileType::Vec, half, 16, Columns>;
    using PatchGM = GlobalTensor<half, Shape<1, 1, 1, 16, Columns>, Stride<1, 1, 1, RowStride, 1>>;
    Patch patch;
    TASSIGN<0>(patch);
    PatchGM gm(input + 16);
    TLOAD(patch, gm);
#ifndef __CPU_SIM
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
#endif
    TINSERT(matrix, patch, Rows - 16, 0);
#endif
#endif
}

#ifdef __CPU_SIM
static uint16_t bits(half value)
{
    uint16_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

int main()
{
    std::vector<uint16_t> original(Rows * RowStride + 32, 0x3555);
    for (int row = 0; row < Rows; ++row)
        for (int col = 0; col < Columns; ++col)
            original[16 + row * RowStride + col] = uint16_t(row * 251u + col * 4093u);
    std::vector<half> input(original.size());
    std::memcpy(input.data(), original.data(), original.size() * sizeof(uint16_t));
    // Reuse the exact physical L1 allocation: poison its complete payload first,
    // then verify every logical element, including both sides of DMA split rows.
    Matrix matrix;
    TASSIGN<0>(matrix);
    for (int row = 0; row < Rows; ++row)
        for (int col = 0; col < Columns; ++col)
            matrix.SetElement(row, col, half(-7));
    A1LargeMatLoad(input.data());
    for (int row = 0; row < Rows; ++row)
        for (int col = 0; col < Columns; ++col)
            if (bits(matrix.GetElement(row, col)) !=
                original[16 + (MAT_STAGE_MODE == 2 && row >= Rows - 16 ? row - (Rows - 16) : row) * RowStride + col])
                return 1;
    if (std::memcmp(input.data(), original.data(), original.size() * sizeof(uint16_t)) != 0)
        return 2;
    return 0;
}
#endif
