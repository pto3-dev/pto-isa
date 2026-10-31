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
#if defined(__CPU_SIM) && defined(A1_RESHAPE_NATIVE_MODEL)
// Exercise the unmodified production helper with CPU tile storage/assignment.
// Skip only the generic reshape implementation to avoid duplicate definitions.
#define __PTO_RESHAPE__
#endif
#include <pto/pto-inst.hpp>
#if defined(__CPU_SIM) && defined(A1_RESHAPE_NATIVE_MODEL)
#include <pto/npu/a1/TReshape.hpp>
#endif
#ifdef __CPU_SIM
#include <array>
#include <bit>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>
#endif

#ifndef RESHAPE_TYPE
#define RESHAPE_TYPE uint32_t
#endif
#ifndef RESHAPE_ROWS
#define RESHAPE_ROWS 8
#define RESHAPE_COLS 32
#define RESHAPE_TARGET_COLS 256
#endif
using Element = RESHAPE_TYPE;
constexpr unsigned Rows = RESHAPE_ROWS, Cols = RESHAPE_COLS, TargetCols = RESHAPE_TARGET_COLS;
constexpr unsigned Count = Rows * Cols;
using Source = pto::Tile<pto::TileType::Vec, Element, Rows, Cols>;
using View = pto::Tile<pto::TileType::Vec, Element, Count / TargetCols, TargetCols>;
using SourceGlobal = pto::GlobalTensor<Element, pto::Shape<1, 1, 1, Rows, Cols>, pto::Stride<1, 1, 1, Cols, 1>>;
using ViewGlobal = pto::GlobalTensor<
    Element, pto::Shape<1, 1, 1, Count / TargetCols, TargetCols>, pto::Stride<1, 1, 1, TargetCols, 1>>;

extern "C" __global__ AICORE void ReshapeStorage(
    __gm__ Element* output, __gm__ Element* input, __gm__ Element* replacement)
{
    Source source;
    View view;
    pto::TASSIGN<32>(source);
    pto::TASSIGN<0>(view);
    SourceGlobal original(input), updated(output + Count);
    ViewGlobal flattened(output), overwrite(replacement);
    pto::TLOAD(source, original);
    pto::TRESHAPE(view, source);
#ifdef __CPU_SIM
    if (reinterpret_cast<uintptr_t>(source.data()) != reinterpret_cast<uintptr_t>(view.data()))
        __builtin_trap();
#else
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
#endif
    pto::TSTORE(flattened, view);
#ifndef __CPU_SIM
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
#endif
    // A DMA write through the new shape must be visible through the source.
    pto::TLOAD(view, overwrite);
#ifndef __CPU_SIM
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
#endif
    pto::TSTORE(updated, source);
#ifndef __CPU_SIM
    pipe_barrier(PIPE_ALL);
#endif
}

#ifdef __CPU_SIM
int main()
{
    using Bits = std::conditional_t<sizeof(Element) == 2, uint16_t, uint32_t>;
    constexpr unsigned guard = 16;
    constexpr uint32_t sentinel = 0xa55ac33cU;
    constexpr std::array<uint32_t, 12> patterns = {0U,          0x80000000U, 0x7fc01234U, 0xffc05678U,
                                                   0x7f800001U, 0x7f800000U, 0xffffffffU, 0x7fffffffU,
                                                   0x80000001U, 0x01020304U, 0x7e017c01U, 0xfc017e55U};
    for (unsigned dataset = 0; dataset < 6; ++dataset) {
        std::vector<Element> input(Count + 2 * guard), replacement(Count + 2 * guard);
        std::vector<Element> output(2 * Count + 2 * guard, std::bit_cast<Element>(static_cast<Bits>(sentinel)));
        for (unsigned i = 0; i < input.size(); ++i) {
            const auto a = patterns[(i + dataset) % patterns.size()] ^ (dataset == 5 ? i * 1664525U : 0);
            const auto b = patterns[(i * 7 + dataset) % patterns.size()] ^ (dataset == 5 ? i * 22695477U : 0x5a5a5a5aU);
            input[i] = std::bit_cast<Element>(static_cast<Bits>(a));
            replacement[i] = std::bit_cast<Element>(static_cast<Bits>(b));
        }
        const auto original = input, savedReplacement = replacement;
        ReshapeStorage(output.data() + guard, input.data() + guard, replacement.data() + guard);
        if (std::memcmp(input.data(), original.data(), input.size() * sizeof(Element)) ||
            std::memcmp(replacement.data(), savedReplacement.data(), replacement.size() * sizeof(Element)))
            return 1;
        for (unsigned i = 0; i < output.size(); ++i) {
            const Bits expected = i < guard || i >= output.size() - guard ? static_cast<Bits>(sentinel) :
                                  i < guard + Count                       ? std::bit_cast<Bits>(original[i]) :
                                                      std::bit_cast<Bits>(savedReplacement[i - Count]);
            if (std::bit_cast<Bits>(output[i]) != expected) {
                std::fprintf(stderr, "reshape storage dataset %u element %u mismatch\n", dataset, i);
                return 2;
            }
        }
    }
    return 0;
}
#endif
