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
#include <bit>
#include <charconv>
#include <iostream>
#include <string_view>
#endif
#include <pto/pto-inst.hpp>

#ifndef PTO_TEST_EXTRACT_SRC_ROWS
#define PTO_TEST_EXTRACT_SRC_ROWS 2304
#endif
#ifndef PTO_TEST_EXTRACT_SRC_COLS
#define PTO_TEST_EXTRACT_SRC_COLS 32
#endif
#ifndef PTO_TEST_EXTRACT_DST_ROWS
#define PTO_TEST_EXTRACT_DST_ROWS 144
#endif
#ifndef PTO_TEST_EXTRACT_DST_COLS
#define PTO_TEST_EXTRACT_DST_COLS 32
#endif
#ifndef PTO_TEST_EXTRACT_STEP
#define PTO_TEST_EXTRACT_STEP 4
#endif
#ifndef PTO_TEST_EXTRACT_ROW
#define PTO_TEST_EXTRACT_ROW 1776
#endif
#ifndef PTO_TEST_EXTRACT_COL
#define PTO_TEST_EXTRACT_COL 0
#endif
#ifndef PTO_TEST_EXTRACT_ROW_MAJOR
#define PTO_TEST_EXTRACT_ROW_MAJOR 0
#endif
#ifndef PTO_TEST_EXTRACT_ORDINARY
#define PTO_TEST_EXTRACT_ORDINARY 0
#endif

using namespace pto;
constexpr unsigned sourceRows = PTO_TEST_EXTRACT_SRC_ROWS;
constexpr unsigned sourceColumns = PTO_TEST_EXTRACT_SRC_COLS;
constexpr unsigned targetRows = PTO_TEST_EXTRACT_DST_ROWS;
constexpr unsigned targetColumns = PTO_TEST_EXTRACT_DST_COLS;
constexpr unsigned blockStep = PTO_TEST_EXTRACT_STEP;
constexpr unsigned firstRow = PTO_TEST_EXTRACT_ROW;
constexpr unsigned firstColumn = PTO_TEST_EXTRACT_COL;
constexpr unsigned sourceBytes = sourceRows * sourceColumns * sizeof(half);
constexpr unsigned targetBytes = targetRows * targetColumns * sizeof(half);
constexpr unsigned sourceAddress = sourceBytes == 1048576 ? 0 : 512;
constexpr unsigned targetAddress = targetBytes == 65536 ? 0 : 512;
constexpr BLayout sourceLayout = PTO_TEST_EXTRACT_ROW_MAJOR ? BLayout::RowMajor : BLayout::ColMajor;
using Source = Tile<
    TileType::Mat, half, sourceRows, sourceColumns, sourceLayout, sourceRows, sourceColumns, SLayout::RowMajor, 512>;
using Target = Tile<
    TileType::Right, half, targetRows, targetColumns, BLayout::RowMajor, targetRows, targetColumns, SLayout::ColMajor,
    512>;

PTO_INTERNAL void RunBlockExtract(uint16_t row, uint16_t column)
{
    Source source;
    Target target;
    TASSIGN(source, sourceAddress);
    TASSIGN(target, targetAddress);
    if constexpr (PTO_TEST_EXTRACT_ORDINARY) {
        TEXTRACT(target, source, row, column);
    } else {
        TEXTRACT<blockStep>(target, source, row, column);
    }
}

extern "C" __global__ AICORE void A1BlockExtract() { RunBlockExtract(firstRow, firstColumn); }

extern "C" __global__ AICORE void A1BlockExtractDynamic(uint16_t row, uint16_t column) { RunBlockExtract(row, column); }

#if PTO_TEST_EXTRACT_ORDINARY
// A full-window contiguous extraction must preserve the existing TMOV encoding.
extern "C" __global__ AICORE void A1BlockExtractMoveControl()
{
    Source source;
    Target target;
    TASSIGN(source, sourceAddress);
    TASSIGN(target, targetAddress);
    TMOV(target, source);
}
#endif

#ifdef __CPU_SIM
using SourceGuard = Tile<TileType::Mat, half, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 512>;
using TargetGuard = Tile<TileType::Right, half, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor, 512>;

uint16_t SourceBits(unsigned row, unsigned column, unsigned dataset)
{
    return static_cast<uint16_t>(row * 257 + column * 4093 + dataset * 73);
}

uint16_t Bits(half value) { return std::bit_cast<uint16_t>(value); }
half FromBits(uint16_t bits) { return std::bit_cast<half>(bits); }

template <typename Guard>
void Paint(Guard& guard)
{
    for (unsigned row = 0; row < 16; ++row) {
        for (unsigned column = 0; column < 16; ++column) {
            guard.SetElement(row, column, FromBits(0x3555));
        }
    }
}

template <typename Guard>
bool Intact(Guard& guard)
{
    for (unsigned row = 0; row < 16; ++row) {
        for (unsigned column = 0; column < 16; ++column) {
            if (Bits(guard.GetElement(row, column)) != 0x3555) {
                return false;
            }
        }
    }
    return true;
}

bool CheckTarget(Target& target, unsigned dataset)
{
    for (unsigned row = 0; row < targetRows; ++row) {
        const unsigned inputRow = firstRow + (row / 16) * blockStep * 16 + row % 16;
        for (unsigned column = 0; column < targetColumns; ++column) {
            if (Bits(target.GetElement(row, column)) != SourceBits(inputRow, firstColumn + column, dataset)) {
                std::cerr << "target mismatch at " << row << ',' << column << '\n';
                return false;
            }
        }
    }
    return true;
}

bool CheckDataset(unsigned dataset)
{
    Source source;
    Target target;
    SourceGuard lowerSource, upperSource;
    TargetGuard lowerTarget, upperTarget;
    TASSIGN(source, sourceAddress);
    TASSIGN(target, targetAddress);
    if constexpr (sourceAddress >= 512) {
        TASSIGN(lowerSource, sourceAddress - 512);
        Paint(lowerSource);
    }
    if constexpr (sourceAddress + sourceBytes + 512 <= 1048576) {
        TASSIGN(upperSource, sourceAddress + sourceBytes);
        Paint(upperSource);
    }
    if constexpr (targetAddress >= 512) {
        TASSIGN(lowerTarget, targetAddress - 512);
        Paint(lowerTarget);
    }
    if constexpr (targetAddress + targetBytes + 512 <= 65536) {
        TASSIGN(upperTarget, targetAddress + targetBytes);
        Paint(upperTarget);
    }
    for (unsigned row = 0; row < sourceRows; ++row) {
        for (unsigned column = 0; column < sourceColumns; ++column) {
            source.SetElement(row, column, FromBits(SourceBits(row, column, dataset)));
        }
    }
    for (unsigned row = 0; row < targetRows; ++row) {
        for (unsigned column = 0; column < targetColumns; ++column) {
            target.SetElement(row, column, FromBits(0x3555));
        }
    }
    A1BlockExtract();
    if (!CheckTarget(target, dataset)) {
        return false;
    }
    for (unsigned row = 0; row < sourceRows; ++row) {
        for (unsigned column = 0; column < sourceColumns; ++column) {
            if (Bits(source.GetElement(row, column)) != SourceBits(row, column, dataset)) {
                return false;
            }
        }
    }
    if constexpr (sourceAddress >= 512) {
        if (!Intact(lowerSource)) {
            return false;
        }
    }
    if constexpr (sourceAddress + sourceBytes + 512 <= 1048576) {
        if (!Intact(upperSource)) {
            return false;
        }
    }
    if constexpr (targetAddress >= 512) {
        if (!Intact(lowerTarget)) {
            return false;
        }
    }
    if constexpr (targetAddress + targetBytes + 512 <= 65536) {
        if (!Intact(upperTarget)) {
            return false;
        }
    }
    return true;
}

bool ParseOffset(const char* argument, uint16_t& offset)
{
    std::string_view text(argument);
    unsigned value = 0;
    auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value > UINT16_MAX) {
        return false;
    }
    offset = static_cast<uint16_t>(value);
    return true;
}

int main(int argc, char** argv)
{
    if (argc == 3) {
        uint16_t row = 0, column = 0;
        if (!ParseOffset(argv[1], row) || !ParseOffset(argv[2], column)) {
            return 4;
        }
        A1BlockExtractDynamic(row, column);
        return 0;
    }
    if (argc != 1) {
        return 4;
    }
    for (unsigned dataset = 0; dataset < 3; ++dataset) {
        if (!CheckDataset(dataset)) {
            return 1;
        }
    }
    std::cout << "PASS " << 3 * uint64_t(targetRows) * targetColumns
              << " bitwise FP16 target values and readonly source/guards\n";
    return 0;
}
#endif
