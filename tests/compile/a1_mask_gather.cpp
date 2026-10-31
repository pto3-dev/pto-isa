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
#include <cstring>
#include <vector>
#endif
#ifndef MASK_INPUT
#define MASK_INPUT float
#endif
#ifndef MASK_OUTPUT
#define MASK_OUTPUT uint32_t
#endif
#ifndef MASK_PATTERN
#define MASK_PATTERN 2
#endif
#ifndef MASK_ROWS
#define MASK_ROWS 3
#endif
#ifndef MASK_COLS
#define MASK_COLS 128
#endif
using Input = MASK_INPUT;
using Output = MASK_OUTPUT;
constexpr unsigned Pattern = MASK_PATTERN, Rows = MASK_ROWS, Cols = MASK_COLS;
constexpr unsigned Factor = Pattern <= 2 ? 2 : Pattern <= 6 ? 4 : 1;
constexpr unsigned Offset = Pattern <= 2 ? Pattern - 1 : Pattern <= 6 ? Pattern - 3 : 0;
constexpr unsigned SourceN = Rows * Cols, N = SourceN / Factor;
constexpr unsigned SourceBytes = SourceN * sizeof(Input), DestinationBytes = N * sizeof(Output);
constexpr unsigned DestinationAddress = SourceBytes + 192;
static_assert(SourceBytes + DestinationBytes + 256 <= 128 * 1024, "Test tiles and guards must fit UB");
using Source = pto::Tile<pto::TileType::Vec, Input, Rows, Cols>;
using Result = pto::Tile<pto::TileType::Vec, Output, Rows, Cols / Factor>;
using Guard = pto::Tile<pto::TileType::Vec, uint32_t, 1, 16>;
template <typename T, unsigned R, unsigned C>
using GM = pto::GlobalTensor<T, pto::Shape<1, 1, 1, R, C>, pto::Stride<1, 1, 1, C, 1>>;

#ifdef A1_MASK_NATIVE_MODEL
// Model the m200 intrinsic contract, while executing the production mask,
// chunk, pointer and tail logic unchanged. Generic CPU TGATHER is separate.
static bool modelCountMode;
static uint64_t modelHigh = UINT64_MAX, modelLow = UINT64_MAX;
static unsigned modelCalls, modelBarriers, modelCopiedBytes;
static uintptr_t modelSource, modelDestination;
static void modelMask(uint64_t high, uint64_t low)
{
    modelHigh = high;
    modelLow = low;
}
static void modelNorm() { modelCountMode = false; }
static void modelCount() { modelCountMode = true; }
static void trap() { std::exit(86); }
static void modelBarrier(pipe_t pipe)
{
    if (pipe != PIPE_V)
        std::abort();
    ++modelBarriers;
}
static void copyWord(void* destination, const void* source, unsigned bytes)
{
    const uintptr_t input = reinterpret_cast<uintptr_t>(source), output = reinterpret_cast<uintptr_t>(destination);
    if (input < modelSource || input + bytes > modelSource + SourceBytes || output < modelDestination ||
        output + bytes > modelDestination + DestinationBytes)
        std::abort();
    std::memcpy(destination, source, bytes);
    modelCopiedBytes += bytes;
}
template <typename Word>
static void vreduce(
    Word* dst, Word* src, Word* ignored, uint8_t repeats, int db, int sb, int pattern, int dr, int sr, int mr)
{
    if (ignored != src || !repeats || db != 1 || sb != 1 || dr != 8 || sr != 8 || mr != 0 || pattern < 1 || pattern > 6)
        std::abort();
    constexpr unsigned lanes = 256 / sizeof(Word);
    const unsigned factor = pattern <= 2 ? 2 : 4, offset = pattern <= 2 ? pattern - 1 : pattern - 3;
    if (modelCountMode && (repeats != 1 || modelHigh || modelLow == 0 || modelLow >= lanes))
        std::abort();
    if (!modelCountMode && (modelHigh != UINT64_MAX || modelLow != UINT64_MAX))
        std::abort();
    const unsigned elements = modelCountMode ? modelLow : repeats * lanes;
    unsigned written = 0;
    for (unsigned i = 0; i < elements; ++i)
        if (i % factor == offset)
            copyWord(dst + written++, src + i, sizeof(Word));
    ++modelCalls;
}
static void vor(
    uint16_t* dst, uint16_t* left, uint16_t* right, uint8_t repeats, int db, int lb, int rb, int dr, int lr, int rr)
{
    if (left != right || !repeats || db != 1 || lb != 1 || rb != 1 || lr != rr ||
        (repeats > 1 && (dr != 8 || lr != 8)) || modelCountMode)
        std::abort();
    for (unsigned r = 0; r < repeats; ++r)
        for (unsigned lane = 0; lane < 128; ++lane)
            if (((lane < 64 ? modelLow : modelHigh) >> (lane % 64)) & 1)
                copyWord(dst + r * dr * 16 + lane, left + r * lr * 16 + lane, sizeof(uint16_t));
    ++modelCalls;
}
// The header also defines indexed gather. It is not instantiated by this
// model; keep its nondependent integer-add declaration available for parsing.
static void vadd(int32_t*, int32_t*, int32_t*, uint8_t, int, int, int, int, int, int);
#undef set_mask_norm
#undef set_mask_count
#undef set_vector_mask
#define set_vector_mask modelMask
#define set_mask_norm modelNorm
#define set_mask_count modelCount
#define pipe_barrier modelBarrier
#define PTO_NPU_ARCH_A1
#define TGATHER_IMPL A1NativeMaskGather
#include <pto/npu/a1/TGather.hpp>
#undef TGATHER_IMPL
#undef pipe_barrier
#undef set_mask_count
#undef set_mask_norm
#undef set_vector_mask
#endif

extern "C" __global__ AICORE void A1MaskGather(__gm__ Output* output, __gm__ Input* input, __gm__ uint32_t* guards)
{
    Source source;
    Result result;
    Guard sourceBefore, sourceAfter, outputBefore, outputAfter;
    pto::TASSIGN<64>(source);
    pto::TASSIGN<DestinationAddress>(result);
    pto::TASSIGN<0>(sourceBefore);
    pto::TASSIGN<64 + SourceBytes>(sourceAfter);
    pto::TASSIGN<DestinationAddress - 64>(outputBefore);
    pto::TASSIGN<DestinationAddress + DestinationBytes>(outputAfter);
    GM<Input, Rows, Cols> sourceGM(input);
    GM<Output, Rows, Cols / Factor> outputGM(output);
    GM<uint32_t, 1, 16> sbGM(guards), saGM(guards + 16), obGM(guards + 32), oaGM(guards + 48);
    pto::TLOAD(source, sourceGM);
    pto::TLOAD(sourceBefore, sbGM);
    pto::TLOAD(sourceAfter, saGM);
    pto::TLOAD(outputBefore, obGM);
    pto::TLOAD(outputAfter, oaGM);
#ifndef __CPU_SIM
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
#ifdef A1_MASK_NATIVE_MODEL
    modelSource = reinterpret_cast<uintptr_t>(source.data());
    modelDestination = reinterpret_cast<uintptr_t>(result.data());
    pto::A1NativeMaskGather<Result, Source, static_cast<pto::MaskPattern>(Pattern)>(result, source);
    if (std::memcmp(source.data(), input, SourceBytes) || modelCountMode || modelHigh != UINT64_MAX ||
        modelLow != UINT64_MAX)
        std::abort();
#else
    pto::TGATHER<Result, Source, static_cast<pto::MaskPattern>(Pattern)>(result, source);
#endif
#ifndef __CPU_SIM
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    pto::TSTORE(outputGM, result);
    pto::TSTORE(sbGM, sourceBefore);
    pto::TSTORE(saGM, sourceAfter);
    pto::TSTORE(obGM, outputBefore);
    pto::TSTORE(oaGM, outputAfter);
}

#ifdef __CPU_SIM
int main(int argc, char** argv)
{
#ifdef A1_MASK_NATIVE_MODEL
    if (argc > 1) {
        Source source;
        Result result;
        pto::TASSIGN<64>(source);
        const unsigned address = argv[1][0] == '0' ? 64 : argv[1][0] == '1' ? 32 : 96;
        pto::TASSIGN(result, address);
        pto::A1NativeMaskGather<Result, Source, static_cast<pto::MaskPattern>(Pattern)>(result, source);
        return 88;
    }
#else
    (void)argc;
    (void)argv;
#endif
    constexpr unsigned guard = 16;
    constexpr uint32_t sentinel = 0xa55ac33cU;
    constexpr std::array<uint32_t, 16> patterns = {
        0U,          0x80000000U, 0x7fc01234U, 0xffc05678U, 0x7f800001U, 0x7f800000U, 0xffffffffU, 0x80000001U,
        0x01020304U, 0x00008000U, 0x00007c00U, 0x0000fc00U, 0x00007e13U, 0x0000fe55U, 0x00000001U, 0x00007bffU};
    for (unsigned dataset = 0; dataset < 6; ++dataset) {
        std::vector<Input> source(SourceN + 2 * guard);
        std::vector<Output> result(N + 2 * guard);
        std::vector<uint32_t> guards(64, sentinel);
        for (unsigned i = 0; i < source.size(); ++i) {
            const uint32_t bits = dataset < 2        ? patterns[(i + dataset) % patterns.size()] :
                                  sizeof(Input) == 2 ? (i + (dataset - 2) * SourceN) & 0xffffU :
                                                       patterns[(i + dataset) % patterns.size()] ^ i * 1664525U;
            std::memcpy(&source[i], &bits, sizeof(Input));
        }
        for (auto& value : result)
            std::memcpy(&value, &sentinel, sizeof(Output));
        const auto original = source;
        A1MaskGather(result.data() + guard, source.data() + guard, guards.data());
        if (std::memcmp(source.data(), original.data(), source.size() * sizeof(Input)))
            return 1;
        for (unsigned i = 0; i < result.size(); ++i) {
            const void* expected = i >= guard && i < guard + N ?
                                       static_cast<const void*>(&source[guard + (i - guard) * Factor + Offset]) :
                                       static_cast<const void*>(&sentinel);
            if (std::memcmp(&result[i], expected, sizeof(Output)))
                return 2;
        }
        for (auto value : guards)
            if (value != sentinel)
                return 3;
    }
#ifdef A1_MASK_NATIVE_MODEL
    constexpr unsigned lanes = 256 / sizeof(Input);
    constexpr unsigned full = SourceN / lanes, tail = SourceN % lanes;
    if (modelCalls != ((full + 254) / 255 + (tail != 0)) * 6 || modelBarriers != 6 ||
        modelCopiedBytes != DestinationBytes * 6)
        return 4;
#endif
    return 0;
}
#endif
