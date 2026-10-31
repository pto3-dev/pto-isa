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
#include <algorithm>
#include <array>
#include <cstring>
#include <vector>
#endif
#ifndef GATHER_TYPE
#define GATHER_TYPE 1
#endif
#ifndef GATHER_ROWS
#define GATHER_ROWS 8
#endif
#ifndef GATHER_COLS
#define GATHER_COLS 96
#endif
#ifndef GATHER_SOURCE_COLS
#define GATHER_SOURCE_COLS 96
#endif
#ifndef GATHER_SOURCE_ROWS
#define GATHER_SOURCE_ROWS 8
#endif
#if GATHER_TYPE == 0
using Data = half;
#elif GATHER_TYPE == 1
using Data = float;
#elif GATHER_TYPE == 2
using Data = int32_t;
#else
using Data = uint32_t;
#endif
#ifdef GATHER_UNSIGNED_INDEX
using Index = uint32_t;
#else
using Index = int32_t;
#endif
constexpr unsigned Rows = GATHER_ROWS, Cols = GATHER_COLS;
constexpr unsigned SourceRows = GATHER_SOURCE_ROWS, SourceCols = GATHER_SOURCE_COLS;
constexpr unsigned N = Rows * Cols, SourceN = SourceRows * SourceCols;
constexpr unsigned Halves = sizeof(Data) / sizeof(uint16_t);
using Source = pto::Tile<pto::TileType::Vec, Data, SourceRows, SourceCols>;
using Result = pto::Tile<pto::TileType::Vec, Data, Rows, Cols>;
using Indices = pto::Tile<pto::TileType::Vec, Index, Rows, Cols>;
// Extra scratch capacity must remain untouched past the required offset prefix.
using Scratch = pto::Tile<pto::TileType::Vec, Index, Rows, Cols * Halves + 16>;
using Guard = pto::Tile<pto::TileType::Vec, uint32_t, 1, 16>;
template <typename T, unsigned R, unsigned C>
using GM = pto::GlobalTensor<T, pto::Shape<1, 1, 1, R, C>, pto::Stride<1, 1, 1, C, 1>>;

#ifdef A1_GATHER_NATIVE_MODEL
// Execute the production A1 offset/mask/repeat implementation on the host.
// Model integer VADD plus raw 16/32-bit vgather. Numeric and guard
// checks are shared with the independent generic CPU reference.
// The source address field is the low 32 bits of the UB pointer on m200.
static const uint8_t* modelSource;
static unsigned modelSourceBytes, modelRepeats, modelErrors, modelSets, modelAdds, modelBarriers;
static unsigned modelPending[2];
static void modelSetFlag(pipe_t from, pipe_t to, event_t id)
{
    const unsigned slot = from == PIPE_S ? 0 : 1;
    if (id != EVENT_ID7 || (from != PIPE_S && from != PIPE_V) || to != (from == PIPE_S ? PIPE_V : PIPE_S) ||
        modelPending[slot])
        std::abort();
    ++modelPending[slot];
    ++modelSets;
}
static void modelWaitFlag(pipe_t from, pipe_t to, event_t id)
{
    const unsigned slot = from == PIPE_S ? 0 : 1;
    if (id != EVENT_ID7 || (from != PIPE_S && from != PIPE_V) || to != (from == PIPE_S ? PIPE_V : PIPE_S) ||
        !modelPending[slot])
        std::abort();
    --modelPending[slot];
}
static uint64_t modelHigh = UINT64_MAX, modelLow = UINT64_MAX;
static void modelMask(uint64_t high, uint64_t low)
{
    modelHigh = high;
    modelLow = low;
}
static void trap() { std::exit(86); }
static void modelBarrier(pipe_t pipe)
{
    if (pipe != PIPE_V)
        std::abort();
    ++modelBarriers;
}
static void vadd(
    int32_t* dst, int32_t* left, int32_t* right, uint8_t repeats, int db, int lb, int rb, int dr, int lr, int rr)
{
    if (modelHigh || db != 1 || lb != 1 || rb != 1 || dr != 8 || lr != 8 || rr != 8 || !repeats)
        std::abort();
    for (unsigned repeat = 0; repeat < repeats; ++repeat)
        for (unsigned lane = 0; lane < 64; ++lane)
            if ((modelLow >> lane) & 1) {
                const unsigned i = repeat * 64 + lane;
                const uint32_t value = uint32_t(left[i]) + uint32_t(right[i]);
                std::memcpy(dst + i, &value, sizeof(value));
            }
    modelAdds += repeats;
}
template <typename Word>
static void vgather(Word* dst, uint32_t* offsets, uint64_t config)
{
    if (uint32_t(config) != uint32_t(reinterpret_cast<uintptr_t>(modelSource)) || ((config >> 32) & 0xff) != 8 ||
        ((config >> 56) & 0xff) == 0)
        std::abort();
    const unsigned repeats = config >> 56;
    for (unsigned repeat = 0; repeat < repeats; ++repeat) {
        for (unsigned lane = 0; lane < 256 / sizeof(Word); ++lane) {
            if (((lane < 64 ? modelLow : modelHigh) >> (lane % 64)) & 1) {
                const unsigned index = repeat * (256 / sizeof(Word)) + lane;
                if (offsets[index] + sizeof(Word) > modelSourceBytes)
                    std::abort();
                std::memcpy(dst + index, modelSource + offsets[index], sizeof(Word));
            }
        }
    }
    modelRepeats += repeats;
}
#undef set_vector_mask
#define set_vector_mask(high, low) modelMask(high, low)
// Keep the CPU reference's public TGATHER separate from the native backend.
#define PTO_NPU_ARCH_A1
#define set_flag modelSetFlag
#define wait_flag modelWaitFlag
#define pipe_barrier modelBarrier
#define TGATHER_IMPL A1NativeIndexedGather
#include <pto/npu/a1/TGather.hpp>
#undef TGATHER_IMPL
#undef pipe_barrier
#undef set_flag
#undef wait_flag
#undef set_vector_mask
#endif

extern "C" __global__ AICORE void A1IndexedGather(
    __gm__ Data* output, __gm__ Data* input, __gm__ Index* positions, __gm__ uint32_t* scratchGuards)
{
    Source source;
    Result result;
    Indices indices;
    Scratch scratch;
    Guard before, after, outputBefore, outputAfter;
    pto::TASSIGN<0>(source);
    pto::TASSIGN<32768>(indices);
    pto::TASSIGN<65536>(scratch);
    pto::TASSIGN<106496>(result);
    pto::TASSIGN<65536 - 64>(before);
    pto::TASSIGN<65536 + N * Halves * sizeof(Index)>(after);
    pto::TASSIGN<106496 - 64>(outputBefore);
    pto::TASSIGN<106496 + N * sizeof(Data)>(outputAfter);
    GM<Data, SourceRows, SourceCols> sourceGM(input);
    GM<Index, Rows, Cols> indexGM(positions);
    GM<Data, Rows, Cols> outputGM(output);
    GM<uint32_t, 1, 16> beforeGM(scratchGuards), afterGM(scratchGuards + 16);
    GM<uint32_t, 1, 16> outputBeforeGM(scratchGuards + 32), outputAfterGM(scratchGuards + 48);
    pto::TLOAD(source, sourceGM);
    pto::TLOAD(indices, indexGM);
    pto::TLOAD(before, beforeGM);
    pto::TLOAD(after, afterGM);
    pto::TLOAD(outputBefore, outputBeforeGM);
    pto::TLOAD(outputAfter, outputAfterGM);
#ifndef __CPU_SIM
    // Keep conservative DMA publication to both pipes for manual callers.
    // The optimized offset generation and value gather both execute on V.
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
#ifdef A1_GATHER_NATIVE_MODEL
    std::fill_n(scratch.data(), N * Halves, Index(0x13579bdf));
    modelSource = reinterpret_cast<const uint8_t*>(source.data());
    modelSourceBytes = SourceN * sizeof(Data);
    pto::A1NativeIndexedGather(result, source, indices, scratch);
    for (unsigned i = 0; i < N; ++i)
        if (uint32_t(scratch.data()[i]) != uint32_t(positions[i]) * sizeof(Data))
            ++modelErrors;
    for (unsigned i = N; i < N * Halves; ++i)
        if (scratch.data()[i] != Index(0x13579bdf))
            ++modelErrors;
    if (std::memcmp(source.data(), input, SourceN * sizeof(Data)) ||
        std::memcmp(indices.data(), positions, N * sizeof(Index)) || modelHigh != UINT64_MAX || modelLow != UINT64_MAX)
        ++modelErrors;
#else
    pto::TGATHER(result, source, indices, scratch);
#endif
#ifndef __CPU_SIM
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    pto::TSTORE(outputGM, result);
    pto::TSTORE(beforeGM, before);
    pto::TSTORE(afterGM, after);
    pto::TSTORE(outputBeforeGM, outputBefore);
    pto::TSTORE(outputAfterGM, outputAfter);
}

#ifdef __CPU_SIM
int main(int argc, char** argv)
{
#ifdef A1_GATHER_NATIVE_MODEL
    if (argc > 1) {
        Source source;
        Result result;
        Indices indices;
        Scratch scratch;
        pto::TASSIGN<0>(source);
        pto::TASSIGN<32768>(indices);
        pto::TASSIGN<65536>(scratch);
        pto::TASSIGN<106496>(result);
        switch (argv[1][0]) {
            case '0':
                pto::TASSIGN<0>(result);
                break;
            case '1':
                pto::TASSIGN<32768>(result);
                break;
            case '2':
                pto::TASSIGN<0>(scratch);
                break;
            case '3':
                pto::TASSIGN<32768>(scratch);
                break;
            case '4':
                pto::TASSIGN<106496>(scratch);
                break;
            default:
                return 87;
        }
        pto::A1NativeIndexedGather(result, source, indices, scratch);
        return 88;
    }
#else
    (void)argc;
    (void)argv;
#endif
    constexpr unsigned guard = 16;
    constexpr uint32_t sentinel = 0xa55ac33cU;
    constexpr std::array<uint32_t, 14> patterns = {0U,          0x80000000U, 0x7fc01234U, 0xffc05678U, 0x7f800001U,
                                                   0x7f800000U, 0xffffffffU, 0x7fffffffU, 0x80000001U, 0x01020304U,
                                                   0x7e017c01U, 0xfc017e55U, 0x00008000U, 0x00003c00U};
    for (unsigned dataset = 0; dataset < 4; ++dataset) {
        std::vector<Data> source(SourceN + 2 * guard), result(N + 2 * guard);
        std::vector<Index> indices(N + 2 * guard, Index(7));
        std::vector<uint32_t> scratchGuards(64, sentinel);
        for (unsigned i = 0; i < source.size(); ++i) {
            const uint32_t bits = patterns[(i + dataset) % patterns.size()] ^ (dataset == 3 ? i * 1664525U : 0);
            std::memcpy(&source[i], &bits, sizeof(Data));
        }
        for (auto& value : result)
            std::memcpy(&value, &sentinel, sizeof(Data));
        for (unsigned i = 0; i < N; ++i) {
            // Identity/rotation, reverse, duplicates and a cross-row permutation.
            const unsigned index = dataset == 0 ? i % SourceN :
                                   dataset == 1 ? SourceN - 1 - i % SourceN :
                                   dataset == 2 ? (i % 7 == 0 ? 0 : SourceN - 1) :
                                                  (i * 37 + 19) % SourceN;
            indices[guard + i] = static_cast<Index>(index);
        }
        const auto original = source;
        const auto originalIndices = indices;
        A1IndexedGather(result.data() + guard, source.data() + guard, indices.data() + guard, scratchGuards.data());
        if (std::memcmp(source.data(), original.data(), source.size() * sizeof(Data)) || indices != originalIndices)
            return 1;
        for (unsigned i = 0; i < result.size(); ++i) {
            const void* expected = i >= guard && i < guard + N ? static_cast<const void*>(&source[guard + indices[i]]) :
                                                                 static_cast<const void*>(&sentinel);
            if (std::memcmp(&result[i], expected, sizeof(Data)))
                return 2;
        }
        for (auto value : scratchGuards)
            if (value != sentinel)
                return 3;
    }
#ifdef A1_GATHER_NATIVE_MODEL
    constexpr unsigned repeats = (N * Halves + 127) / 128;
    if (modelErrors || modelRepeats != repeats * 4 || modelSets != 8 || modelPending[0] || modelPending[1])
        return 4;
    constexpr unsigned passes = sizeof(Data) == 4 ? 2 : 1;
    if (modelAdds != ((N + 63) / 64) * passes * 4 || modelBarriers != (passes + 1) * 4)
        return 5;
#endif
    return 0;
}
#endif
