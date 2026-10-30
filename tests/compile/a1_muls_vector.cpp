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
#include <array>
#include <bit>
#include <cstring>
#include <vector>
#endif

#ifndef MULS_ROWS
#define MULS_ROWS 1
#endif
#ifndef MULS_COLS
#define MULS_COLS 80
#endif
#ifdef MULS_FP16
using Element = half;
#elif defined(MULS_INT)
using Element = int32_t;
#else
using Element = float;
#endif
constexpr unsigned Rows = MULS_ROWS, Cols = MULS_COLS, Count = Rows * Cols;
constexpr unsigned Bytes = Count * sizeof(Element), GuardCount = 32 / sizeof(Element);
#ifdef MULS_INPLACE
constexpr unsigned Destination = 32;
#else
constexpr unsigned Destination = 64 + Bytes;
#endif
#ifdef MULS_PARTIAL
using Tile = pto::Tile<pto::TileType::Vec, Element, Rows, Cols, pto::BLayout::RowMajor, 0, Cols>;
#else
using Tile = pto::Tile<pto::TileType::Vec, Element, Rows, Cols>;
#endif
using Guard = pto::Tile<pto::TileType::Vec, Element, 1, GuardCount>;
using Global = pto::GlobalTensor<Element, pto::Shape<1, 1, 1, Rows, Cols>, pto::Stride<1, 1, 1, Cols, 1>>;
using GuardGlobal = pto::GlobalTensor<Element, pto::Shape<1, 1, 1, 1, GuardCount>, pto::Stride<1, 1, 1, GuardCount, 1>>;

extern "C" __global__ AICORE void A1ScalarMultiply(__gm__ Element* output, __gm__ Element* input, float factor)
{
    Tile src, dst;
    Guard guard;
    pto::TASSIGN<32>(src);
    pto::TASSIGN<Destination>(dst);
    pto::TASSIGN<Destination + Bytes>(guard);
    Global source(input), destination(output);
    GuardGlobal suffix(output + Count);
    pto::TLOAD(src, source);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pto::TEXPANDS(guard, static_cast<Element>(9));
    pipe_barrier(PIPE_V);
    pto::TMULS(dst, src, static_cast<Element>(factor));
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pto::TSTORE(destination, dst);
    // An adjacent UB guard makes a missing final vector mask observable in GM.
    pto::TSTORE(suffix, guard);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
}

#ifdef __CPU_SIM
int main()
{
    const std::array<float, 6> factors = {0.1767766922712326f, -2.0f, 0.0f, -0.0f, 0x1p-30f, 1.0003f};
    for (float factor : factors) {
        for (int dataset : {0, 1}) {
            std::vector<Element> input(Count + 32, Element(9)), output(Count + 32, Element(9));
            for (unsigned i = 0; i < Count; ++i) {
                input[16 + i] = Element((static_cast<int>((i * 17 + dataset * 7) % 97) - 48) / 31.0f);
                if (dataset == 1 && i < 4) {
                    input[16 + i] = i % 2 ? Element(-0.0f) : Element(0.0f);
                }
            }
            const auto original = input;
#ifdef MULS_INPLACE
            auto* result = input.data();
#else
            auto* result = output.data();
#endif
            A1ScalarMultiply(result + 16, input.data() + 16, factor);
            const Element scalar = static_cast<Element>(factor);
            for (unsigned i = 0; i < Count; ++i) {
                const Element expected = Element(static_cast<float>(original[16 + i]) * static_cast<float>(scalar));
                if (std::memcmp(&result[16 + i], &expected, sizeof(Element)))
                    return 1;
            }
            for (unsigned i = 0; i < 16; ++i) {
                if (result[i] != Element(9) || result[16 + Count + i] != Element(9))
                    return 2;
            }
#ifndef MULS_INPLACE
            if (std::memcmp(input.data(), original.data(), input.size() * sizeof(Element)))
                return 3;
#endif
        }
    }
    return 0;
}
#endif
