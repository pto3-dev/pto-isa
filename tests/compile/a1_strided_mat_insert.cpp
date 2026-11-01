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
#include <cstdlib>
#include <cstring>
#include <vector>
#endif
#include <pto/pto-inst.hpp>

#ifndef SI_ROWS
#define SI_ROWS 16
#endif
#ifndef SI_COLS
#define SI_COLS 32
#endif
#ifndef SI_DST_ROWS
#define SI_DST_ROWS 2304
#endif
#ifndef SI_DST_COLS
#define SI_DST_COLS 64
#endif
#ifndef SI_STEP
#define SI_STEP 145
#endif
#ifndef SI_ROW
#define SI_ROW 128
#endif
#ifndef SI_COL
#define SI_COL 16
#endif

using namespace pto;
using Source = Tile<TileType::Vec, half, SI_ROWS, SI_COLS>;
using Matrix = Tile<
    TileType::Mat, half, SI_DST_ROWS, SI_DST_COLS, BLayout::ColMajor, SI_DST_ROWS, SI_DST_COLS, SLayout::RowMajor, 512>;
using SourceGM = GlobalTensor<half, Shape<1, 1, 1, SI_ROWS, SI_COLS>, Stride<1, 1, 1, SI_COLS + 16, 1>>;
using MatrixGM = GlobalTensor<half, Shape<1, 1, 1, SI_DST_ROWS, SI_DST_COLS>, Stride<1, 1, 1, SI_DST_COLS, 1>>;

PTO_INTERNAL void RunStridedInsert(__gm__ half* input, __gm__ half* initial, uint16_t row, uint16_t column)
{
    Matrix matrix;
    Source source;
    TASSIGN<512>(matrix);
    TASSIGN<0>(source);
    MatrixGM gm(initial + 16);
    SourceGM gs(input + 16);
    TLOAD(matrix, gm);
#ifndef __CPU_SIM
    // Covers both the input load and the initial Mat write before MTE3.
    Event<Op::TLOAD, Op::TINSERT_V2M, false, EVENT_ID0> loaded = TLOAD(source, gs);
    TINSERT<SI_STEP>(matrix, source, row, column, loaded);
    pipe_barrier(PIPE_ALL);
#else
    TLOAD(source, gs);
    TINSERT<SI_STEP>(matrix, source, row, column);
#endif
}

extern "C" __global__ AICORE void A1StridedMatInsert(__gm__ half* input, __gm__ half* initial)
{
    RunStridedInsert(input, initial, SI_ROW, SI_COL);
}

extern "C" __global__ AICORE void A1StridedMatInsertDynamic(
    __gm__ half* input, __gm__ half* initial, uint16_t row, uint16_t column)
{
    RunStridedInsert(input, initial, row, column);
}

#ifdef __CPU_SIM
static half fromBits(uint16_t bits)
{
    half value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
static uint16_t toBits(half value)
{
    uint16_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
using Guard = Tile<TileType::Mat, half, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 512>;
static void initializeGuard(Guard& guard)
{
    for (int row = 0; row < 16; ++row)
        for (int col = 0; col < 16; ++col)
            guard.SetElement(row, col, fromBits(0x3555));
}
static bool guardIntact(Guard& guard)
{
    for (int row = 0; row < 16; ++row)
        for (int col = 0; col < 16; ++col)
            if (toBits(guard.GetElement(row, col)) != 0x3555)
                return false;
    return true;
}

static int checkWindow(uint16_t rowOffset, uint16_t colOffset)
{
    Matrix matrix;
    Source source;
    Guard before, after;
    TASSIGN<512>(matrix);
    TASSIGN<0>(source);
    TASSIGN<0>(before);
    TASSIGN<512 + SI_DST_ROWS * SI_DST_COLS * 2>(after);
    for (int dataset : {0, 1, 2}) {
        initializeGuard(before);
        initializeGuard(after);
        std::vector<uint16_t> original(SI_ROWS * (SI_COLS + 16) + 32, 0x3555);
        std::vector<uint16_t> initial(SI_DST_ROWS * SI_DST_COLS + 32, 0x3955);
        for (size_t i = 0; i < original.size(); ++i)
            original[i] = uint16_t(i * 4093 + dataset * 251);
        for (size_t i = 0; i < initial.size(); ++i)
            initial[i] = uint16_t(i * 277 + dataset * 53);
        std::vector<half> input(original.size()), gm(initial.size());
        std::memcpy(input.data(), original.data(), original.size() * 2);
        std::memcpy(gm.data(), initial.data(), initial.size() * 2);
        // Every untouched element differs from a default zero-initialized L1.
        for (int row = 0; row < SI_DST_ROWS; ++row)
            for (int col = 0; col < SI_DST_COLS; ++col)
                matrix.SetElement(row, col, fromBits(0x3555));
        A1StridedMatInsertDynamic(input.data(), gm.data(), rowOffset, colOffset);
        for (int row = 0; row < SI_DST_ROWS; ++row)
            for (int col = 0; col < SI_DST_COLS; ++col) {
                uint16_t expected = initial[16 + row * SI_DST_COLS + col];
                const int delta = row - rowOffset;
                if (delta >= 0 && delta % SI_STEP == 0 && delta / SI_STEP < SI_ROWS && col >= colOffset &&
                    col - colOffset < SI_COLS)
                    expected = original[16 + delta / SI_STEP * (SI_COLS + 16) + col - colOffset];
                if (toBits(matrix.GetElement(row, col)) != expected)
                    return 1;
            }
        for (int row = 0; row < SI_ROWS; ++row)
            for (int col = 0; col < SI_COLS; ++col)
                if (toBits(source.GetElement(row, col)) != original[16 + row * (SI_COLS + 16) + col])
                    return 2;
        if (!guardIntact(before) || !guardIntact(after))
            return 3;
        if (std::memcmp(input.data(), original.data(), original.size() * 2) ||
            std::memcmp(gm.data(), initial.data(), initial.size() * 2))
            return 4;
    }
    return 0;
}

// Independent compact CHW fixture: zero L1 once through ordinary insertion,
// then reload nine taps repeatedly from compact GM weights using two UB banks.
template <int Channels, int Groups, int Columns>
static int checkSparseWeights()
{
    using Sparse = Tile<
        TileType::Mat, half, Groups * 2304, Columns, BLayout::ColMajor, Groups * 2304, Columns, SLayout::RowMajor, 512>;
    using Patch = Tile<TileType::Vec, half, Channels, Columns>;
    using Zero = Tile<TileType::Vec, half, 16, Columns>;
    using Weights = GlobalTensor<half, Shape<1, 1, 1, Channels, Columns>, Stride<1, 1, 1, 9 * Columns, 1>>;
    Sparse matrix;
    Patch banks[2];
    Zero zero;
    Guard before, after;
    TASSIGN<512>(matrix);
    TASSIGN<0>(before);
    TASSIGN<512 + Groups * 2304 * Columns * 2>(after);
    TASSIGN<0>(banks[0]);
    TASSIGN<Channels * Columns * 2>(banks[1]);
    TASSIGN<2 * Channels * Columns * 2>(zero);
    initializeGuard(before);
    initializeGuard(after);
    for (int row = 0; row < Sparse::Rows; ++row)
        for (int col = 0; col < Columns; ++col)
            matrix.SetElement(row, col, fromBits(0x3555));
    for (int row = 0; row < 16; ++row)
        for (int col = 0; col < Columns; ++col)
            zero.SetElement(row, col, half(0));
    for (int row = 0; row < Sparse::Rows; row += 16)
        TINSERT(matrix, zero, row, 0);
    for (int reload : {0, 1, 2}) {
        std::vector<uint16_t> raw(16 + Groups * Channels * 9 * Columns + 16, 0x3555);
        for (int c = 0; c < Groups * Channels; ++c)
            for (int tap = 0; tap < 9; ++tap)
                for (int n = 0; n < Columns; ++n)
                    raw[16 + (c * 9 + tap) * Columns + n] = uint16_t(c * 4093 + tap * 277 + n * 251 + reload * 53);
        std::vector<half> weights(raw.size());
        std::memcpy(weights.data(), raw.data(), raw.size() * 2);
        for (int group = 0; group < Groups; ++group)
            for (int tap = 0; tap < 9; ++tap) {
                auto& bank = banks[(group * 9 + tap) % 2];
                Weights gm(weights.data() + 16 + (group * Channels * 9 + tap) * Columns);
                TLOAD(bank, gm);
                TINSERT<145>(matrix, bank, group * 2304 + tap * 16, 0);
            }
        std::vector<uint16_t> expected(Sparse::Rows * Columns, 0);
        for (int c = 0; c < Groups * Channels; ++c)
            for (int tap = 0; tap < 9; ++tap)
                for (int n = 0; n < Columns; ++n) {
                    const int dstRow = c / Channels * 2304 + tap * 16 + (c % Channels) * 145;
                    expected[dstRow * Columns + n] = raw[16 + (c * 9 + tap) * Columns + n];
                }
        for (int row = 0; row < Sparse::Rows; ++row)
            for (int n = 0; n < Columns; ++n)
                if (toBits(matrix.GetElement(row, n)) != expected[row * Columns + n])
                    return 5;
        if (std::memcmp(weights.data(), raw.data(), raw.size() * 2) || !guardIntact(before) || !guardIntact(after))
            return 6;
    }
    return 0;
}

template <TInsertMode Mode>
static int checkLegacySplit()
{
    using Target = Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512>;
    using Input = Tile<TileType::Vec, half, 16, 32, BLayout::ColMajor, 16, 32, SLayout::RowMajor, 512>;
    Target target;
    Input input;
    TASSIGN<0>(target);
    TASSIGN<0>(input);
    for (int row = 0; row < 64; ++row)
        for (int col = 0; col < 64; ++col)
            target.SetElement(row, col, half(0));
    for (int row = 0; row < 16; ++row)
        for (int col = 0; col < 32; ++col)
            input.SetElement(row, col, fromBits(uint16_t(row * 251 + col * 4093)));
    TINSERT<Mode>(target, input, 16, 16);
    for (int row = 0; row < 64; ++row)
        for (int col = 0; col < 64; ++col) {
            const uint16_t expected =
                row >= 16 && row < 32 && col >= 16 && col < 48 ? uint16_t((row - 16) * 251 + (col - 16) * 4093) : 0;
            if (toBits(target.GetElement(row, col)) != expected)
                return 7;
        }
    return 0;
}

int main(int argc, char** argv)
{
    if (argc == 2 && std::strcmp(argv[1], "sparse") == 0)
        return checkSparseWeights<3, 1, 16>() || checkSparseWeights<3, 1, 32>() || checkSparseWeights<16, 1, 16>() ||
               checkSparseWeights<16, 1, 32>() || checkSparseWeights<16, 2, 16>() || checkSparseWeights<16, 2, 32>();
    if (argc == 2 && std::strcmp(argv[1], "split") == 0)
        return checkLegacySplit<TInsertMode::SPLIT2>() || checkLegacySplit<TInsertMode::SPLIT4>();
    if (argc == 3)
        return checkWindow(uint16_t(std::atoi(argv[1])), uint16_t(std::atoi(argv[2])));
    return checkWindow(SI_ROW, SI_COL);
}
#endif
