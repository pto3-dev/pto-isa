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
#include <cstdlib>
#include <cstring>
#include <vector>

#include <pto/pto-inst.hpp>

using namespace pto;
using GuardMat = Tile<TileType::Mat, half, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 512>;
using GuardVec = Tile<TileType::Vec, half, 1, 16>;

static half fromBits(uint16_t bits)
{
    half value;
    std::memcpy(&value, &bits, sizeof(bits));
    return value;
}

static uint16_t toBits(half value)
{
    uint16_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

template <typename T>
static void fillGuard(T& tile)
{
    for (int row = 0; row < T::Rows; ++row)
        for (int col = 0; col < T::Cols; ++col)
            tile.SetElement(row, col, fromBits(0x3555));
}

template <typename T>
static bool guardIntact(T& tile)
{
    for (int row = 0; row < T::Rows; ++row)
        for (int col = 0; col < T::Cols; ++col)
            if (toBits(tile.GetElement(row, col)) != 0x3555)
                return false;
    return true;
}

template <int Rows, int Cols, int PanelRows, int PanelCols>
static int checkInsert(int rowOffset, int colOffset, int parts)
{
    using Matrix = Tile<TileType::Mat, half, Rows, Cols, BLayout::ColMajor, Rows, Cols, SLayout::RowMajor, 512>;
    using Source = Tile<TileType::Vec, half, PanelRows, PanelCols>;
    Matrix matrix;
    Source banks[2];
    GuardMat before, after;
    GuardVec ubBefore[2], ubAfter[2];
    TASSIGN<512>(matrix);
    TASSIGN<0>(before);
    TASSIGN<512 + Rows * Cols * 2>(after);
    // Keep guards in the actual simulated Mat/UB address spaces so an incorrect
    // physical window can corrupt them even when the logical result looks right.
    for (int bank = 0; bank < 2; ++bank) {
        TASSIGN(banks[bank], 4096 * bank + 32);
        TASSIGN(ubBefore[bank], 4096 * bank);
        TASSIGN(ubAfter[bank], 4096 * bank + 32 + PanelRows * PanelCols * 2);
    }
    for (int dataset : {0, 1}) {
        fillGuard(before);
        fillGuard(after);
        std::vector<uint16_t> expected(Rows * Cols);
        for (int row = 0; row < Rows; ++row)
            for (int col = 0; col < Cols; ++col) {
                const auto bits = uint16_t((row * Cols + col) * 251u + dataset * 103u);
                expected[row * Cols + col] = bits;
                matrix.SetElement(row, col, fromBits(bits));
            }
        for (int part = 0; part < parts; ++part) {
            const int bank = part % 2;
            auto& source = banks[bank];
            fillGuard(ubBefore[bank]);
            fillGuard(ubAfter[bank]);
            std::vector<uint16_t> original(PanelRows * PanelCols);
            for (int row = 0; row < PanelRows; ++row)
                for (int col = 0; col < PanelCols; ++col) {
                    const auto bits = uint16_t((row * PanelCols + col) * 4093u + part * 277u + dataset * 53u);
                    original[row * PanelCols + col] = bits;
                    source.SetElement(row, col, fromBits(bits));
                    expected[(rowOffset + part * PanelRows + row) * Cols + colOffset + col] = bits;
                }
            TINSERT(matrix, source, uint16_t(rowOffset + part * PanelRows), uint16_t(colOffset));
            for (int row = 0; row < PanelRows; ++row)
                for (int col = 0; col < PanelCols; ++col)
                    if (toBits(source.GetElement(row, col)) != original[row * PanelCols + col])
                        return 1;
            if (!guardIntact(ubBefore[bank]) || !guardIntact(ubAfter[bank]))
                return 2;
        }
        for (int row = 0; row < Rows; ++row)
            for (int col = 0; col < Cols; ++col)
                if (toBits(matrix.GetElement(row, col)) != expected[row * Cols + col])
                    return 3;
        if (!guardIntact(before) || !guardIntact(after))
            return 4;
    }
    return 0;
}

int main(int argc, char** argv)
{
    if (argc == 2 && std::strcmp(argv[1], "c16") == 0)
        return checkInsert<96, 96, 16, 16>(0, 32, 5);
    if (argc == 3)
        return checkInsert<64, 64, 32, 32>(std::atoi(argv[1]), std::atoi(argv[2]), 1);
    if (argc == 1)
        return checkInsert<96, 96, 16, 32>(0, 32, 5);
    return 10;
}
