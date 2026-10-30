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
#include <array>
#include <bit>
#include <vector>
#endif
#include <pto/pto-inst.hpp>

#ifndef DMA_ROWS
#define DMA_ROWS 16
#endif
#ifndef DMA_COLS
#define DMA_COLS 32
#endif
#ifndef DMA_SOURCE_STRIDE
#define DMA_SOURCE_STRIDE DMA_COLS
#endif
#ifndef DMA_DESTINATION_STRIDE
#define DMA_DESTINATION_STRIDE DMA_COLS
#endif
#ifdef DMA_UNSIGNED
using Data = uint32_t;
#else
using Data = int32_t;
#endif
using TileData = pto::Tile<pto::TileType::Vec, Data, DMA_ROWS, DMA_COLS>;
using Shape = pto::Shape<1, 1, 1, DMA_ROWS, DMA_COLS>;
using Input = pto::GlobalTensor<Data, Shape, pto::Stride<1, 1, 1, DMA_SOURCE_STRIDE, 1>>;
using Output = pto::GlobalTensor<Data, Shape, pto::Stride<1, 1, 1, DMA_DESTINATION_STRIDE, 1>>;

extern "C" __global__ AICORE void A1IntegerCopy(__gm__ Data* destination, __gm__ Data* source)
{
    TileData tile;
    pto::TASSIGN<0>(tile);
    Input input(source);
    Output output(destination);
#if defined(__CPU_SIM)
    pto::TLOAD(tile, input);
    pto::TSTORE(output, tile);
#else
    pto::Event<pto::Op::TLOAD, pto::Op::TSTORE_VEC> loaded = pto::TLOAD(tile, input);
    pto::TSTORE(output, tile, loaded);
#endif
}

#if defined(__CPU_SIM)
int main()
{
    constexpr size_t guard = 16;
    constexpr size_t sourceElements = size_t(DMA_ROWS) * DMA_SOURCE_STRIDE;
    constexpr size_t destinationElements = size_t(DMA_ROWS) * DMA_DESTINATION_STRIDE;
    constexpr uint32_t sentinel = 0xa55ac33cU;
    constexpr std::array<uint32_t, 8> edges = {0U,          1U,          0x7fffffffU, 0x80000000U,
                                               0xffffffffU, 0x01020304U, 0x7f800000U, 0x7fc01234U};
    for (uint32_t dataset : {0U, 1U}) {
        std::vector<Data> source(sourceElements + 2 * guard);
        std::vector<Data> destination(destinationElements + 2 * guard, std::bit_cast<Data>(sentinel));
        for (size_t i = 0; i < source.size(); ++i) {
            const uint32_t bits = i < edges.size() ? edges[i] : uint32_t(i) * 1664525U + 1013904223U + dataset;
            source[i] = std::bit_cast<Data>(bits);
        }
        // Exercise signed extrema and bit patterns inside the actual transfer,
        // not only in the prefix guard.
        for (size_t i = 0; i < edges.size(); ++i)
            source[guard + i] = std::bit_cast<Data>(edges[(i + dataset) % edges.size()]);
        const auto original = source;
        A1IntegerCopy(destination.data() + guard, source.data() + guard);
        if (source != original)
            return 1;
        for (size_t i = 0; i < destination.size(); ++i) {
            uint32_t expected = sentinel;
            if (i >= guard && i < guard + destinationElements) {
                const size_t index = i - guard;
                const size_t row = index / DMA_DESTINATION_STRIDE;
                const size_t col = index % DMA_DESTINATION_STRIDE;
                if (col < DMA_COLS)
                    expected = std::bit_cast<uint32_t>(original[guard + row * DMA_SOURCE_STRIDE + col]);
            }
            if (std::bit_cast<uint32_t>(destination[i]) != expected)
                return 2;
        }
    }
    return 0;
}
#endif
