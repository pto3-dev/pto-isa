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
#include <bit>
#include <cstring>
#include <vector>
#endif
#ifndef SORT_ROWS
#define SORT_ROWS 3
#endif
#ifndef SORT_COLS
#define SORT_COLS 160
#endif
#ifndef SORT_SCRATCH
#define SORT_SCRATCH 1040
#endif
constexpr unsigned Rows = SORT_ROWS, Cols = SORT_COLS, N = Rows * Cols, ScratchWords = SORT_SCRATCH;
#ifdef SORT_BROADCAST
constexpr unsigned IndexRows = 1;
#else
constexpr unsigned IndexRows = Rows;
#endif
using Source = pto::Tile<pto::TileType::Vec, float, Rows, Cols>;
using Indices = pto::Tile<pto::TileType::Vec, uint32_t, IndexRows, Cols>;
using Result = pto::Tile<pto::TileType::Vec, float, Rows, 2 * Cols>;
using Scratch = pto::Tile<pto::TileType::Vec, float, 1, ScratchWords>;
using Guard = pto::Tile<pto::TileType::Vec, uint32_t, 1, 16>;
constexpr unsigned SourceAddress = 64, IndexAddress = SourceAddress + N * 4 + 128;
constexpr unsigned OutputAddress = IndexAddress + IndexRows * Cols * 4 + 128;
constexpr unsigned ScratchAddress = OutputAddress + N * 8 + 128;
constexpr unsigned UsedScratch = ScratchWords / 512 * 512;
static_assert(ScratchAddress + ScratchWords * 4 + 64 <= 128 * 1024);
template <typename T, unsigned R, unsigned C>
using GM = pto::GlobalTensor<T, pto::Shape<1, 1, 1, R, C>, pto::Stride<1, 1, 1, C, 1>>;

#ifdef A1_SORT_NATIVE_MODEL
// Model only the native proposal intrinsics. The production A1 helper performs
// batching, scratch addressing, compaction, unsigned ties and synchronization.
static std::array<uint64_t, 8> modelFields;
static unsigned modelVbsRepeats, modelConcatRepeats, modelMerges, modelSets;
static unsigned modelPending[2];
static uint64_t modelHigh = UINT64_MAX, modelLow = UINT64_MAX;
static bool modelReadonlyAlias;
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
static void modelMask(uint64_t high, uint64_t low)
{
    modelHigh = high;
    modelLow = low;
}
static void trap() { std::exit(86); }
static void set_va_reg_sb(unsigned va, uint64_t* fields)
{
    if (va != 0)
        std::abort();
    std::copy(fields, fields + 8, modelFields.begin());
}
static void scatter_vconcat_f32(float* dst, unsigned va, unsigned repeat, unsigned fields)
{
    if (va != 0 || repeat == 0 || repeat > 255 || fields != 0b110000 || modelHigh != UINT64_MAX ||
        modelLow != UINT64_MAX)
        std::abort();
    for (unsigned i = 0; i < repeat * 16; ++i) {
        for (unsigned field : {4U, 5U}) {
            const auto* source = reinterpret_cast<const uint8_t*>(modelFields[field]);
            std::memcpy(dst + i * 8 + field, source + i * 4, 4);
        }
    }
    modelConcatRepeats += repeat;
}
using Proposal = std::array<uint32_t, 8>;
static void vbitsort(float* dst, float* src, unsigned repeat)
{
    if (repeat == 0 || repeat > 255)
        std::abort();
    for (unsigned b = 0; b < repeat; ++b) {
        std::array<Proposal, 16> values;
        std::memcpy(values.data(), src + b * 128, 128 * sizeof(float));
        // Reverse equal scores deliberately; hardware does not promise the
        // compact interface's ascending unsigned index tie order.
        std::reverse(values.begin(), values.end());
        std::stable_sort(values.begin(), values.end(), [](const auto& a, const auto& z) {
            return std::bit_cast<float>(a[4]) > std::bit_cast<float>(z[4]);
        });
        std::memcpy(dst + b * 128, values.data(), 128 * sizeof(float));
    }
    modelVbsRepeats += repeat;
}
static void vmrgsort4(
    float* dst, float** lists, unsigned repeat, unsigned a, unsigned b, unsigned c, unsigned d, bool exhausted,
    unsigned mask)
{
    if (repeat != 1 || a != 16 || b != 16 || c || d || exhausted || mask != 0b0011)
        std::abort();
    unsigned left = 0, right = 0;
    for (unsigned i = 0; i < 32; ++i) {
        const bool takeLeft = right == 16 || (left < 16 && lists[0][left * 8 + 4] > lists[1][right * 8 + 4]);
        auto* value = takeLeft ? lists[0] + left++ * 8 : lists[1] + right++ * 8;
        std::memcpy(dst + i * 8, value, 8 * sizeof(float));
    }
    ++modelMerges;
}
#undef set_vector_mask
#define set_vector_mask(high, low) modelMask(high, low)
#define PTO_NPU_ARCH_A1
#define VA0 0
#define set_flag modelSetFlag
#define wait_flag modelWaitFlag
#define TSORT32_IMPL A1NativeSort32
#include <pto/npu/a1/TSort32.hpp>
#undef TSORT32_IMPL
#undef set_flag
#undef wait_flag
#undef set_vector_mask
#undef VA0
#endif

extern "C" __global__ AICORE void A1Sort32(
    __gm__ float* output, __gm__ float* input, __gm__ uint32_t* positions, __gm__ uint32_t* guards)
{
    Source source;
    Indices indices;
    Result result;
    Scratch scratch;
    Guard before, after, outputBefore, outputAfter;
    pto::TASSIGN<SourceAddress>(source);
    pto::TASSIGN<IndexAddress>(indices);
    pto::TASSIGN<OutputAddress>(result);
    pto::TASSIGN<ScratchAddress>(scratch);
    pto::TASSIGN<ScratchAddress - 64>(before);
    pto::TASSIGN<ScratchAddress + UsedScratch * 4>(after);
    pto::TASSIGN<OutputAddress - 64>(outputBefore);
    pto::TASSIGN<OutputAddress + N * 8>(outputAfter);
    GM<float, Rows, Cols> sourceGM(input);
    GM<uint32_t, IndexRows, Cols> indexGM(positions);
    GM<float, Rows, 2 * Cols> outputGM(output);
    GM<uint32_t, 1, 16> beforeGM(guards), afterGM(guards + 16);
    GM<uint32_t, 1, 16> outputBeforeGM(guards + 32), outputAfterGM(guards + 48);
    pto::TLOAD(source, sourceGM);
    pto::TLOAD(indices, indexGM);
    pto::TLOAD(before, beforeGM);
    pto::TLOAD(after, afterGM);
    pto::TLOAD(outputBefore, outputBeforeGM);
    pto::TLOAD(outputAfter, outputAfterGM);
#ifndef __CPU_SIM
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
#ifdef A1_SORT_NATIVE_MODEL
    if (modelReadonlyAlias)
        pto::TASSIGN<SourceAddress>(indices);
    pto::A1NativeSort32(result, source, indices, scratch);
    if (std::memcmp(source.data(), input, N * 4) || std::memcmp(indices.data(), positions, IndexRows * Cols * 4))
        std::abort();
#else
    pto::TSORT32(result, source, indices, scratch);
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
#ifdef A1_SORT_NATIVE_MODEL
    if (argc > 1 && argv[1][0] != 'r') {
        Source source;
        Indices indices;
        Result result;
        Scratch scratch;
        pto::TASSIGN<SourceAddress>(source);
        pto::TASSIGN<IndexAddress>(indices);
        pto::TASSIGN<OutputAddress>(result);
        pto::TASSIGN<ScratchAddress>(scratch);
        switch (argv[1][0]) {
            case '0':
                pto::TASSIGN(result, SourceAddress + 32);
                break;
            case '1':
                pto::TASSIGN(result, IndexAddress);
                break;
            case '2':
                pto::TASSIGN(scratch, SourceAddress + 32);
                break;
            case '3':
                pto::TASSIGN(scratch, IndexAddress);
                break;
            case '4':
                pto::TASSIGN(scratch, OutputAddress + 32);
                break;
            default:
                return 87;
        }
        pto::A1NativeSort32(result, source, indices, scratch);
        return 88;
    }
    modelReadonlyAlias = argc > 1;
#else
    (void)argc;
    (void)argv;
#endif
    constexpr unsigned guard = 16;
    constexpr uint32_t sentinel = 0xa55ac33cU;
    constexpr std::array<uint32_t, 12> edges = {0U, 0x80000000U, 0x7f800000U, 0xff800000U, 0x7f7fffffU, 0xff7fffffU,
                                                1U, 0x80000001U, 0x00800000U, 0x80800000U, 0x3f800000U, 0xbf800000U};
    for (unsigned dataset = 0; dataset < 5; ++dataset) {
        std::vector<float> input(N + 2 * guard), output(2 * N + 2 * guard);
        std::vector<uint32_t> indices(IndexRows * Cols + 2 * guard, sentinel), guards(64, sentinel);
        for (auto& value : output)
            value = std::bit_cast<float>(sentinel);
        for (auto& value : input)
            value = std::bit_cast<float>(sentinel);
        for (unsigned i = 0; i < IndexRows * Cols; ++i)
            indices[guard + i] = dataset == 4 ? i % 17 : i * 1664525U + 1013904223U;
        for (unsigned i = 0; i < N; ++i) {
            input[guard + i] = dataset == 0 ? float((i * 23 + 17) % 127) - 63.0f :
                               dataset == 1 ? float(i % 7) - 3.0f :
                               dataset == 2 ? std::bit_cast<float>(i % 2 ? 0x80000000U : 0U) :
                               dataset == 3 ? std::bit_cast<float>(edges[i % edges.size()]) :
                                              float(i % 3 + 1);
#ifdef A1_SORT_NATIVE_MODEL
            if (modelReadonlyAlias) {
                const unsigned id = 0x3f000000U + i % Cols;
                input[guard + i] = std::bit_cast<float>(id);
                indices[guard + i % (IndexRows * Cols)] = id;
            }
#endif
        }
        const auto original = input;
        const auto originalIndices = indices;
#ifdef A1_SORT_NATIVE_MODEL
        // A preceding masked operation must not suppress any sort lanes.
        modelHigh = 0;
        modelLow = 1;
#endif
        A1Sort32(output.data() + guard, input.data() + guard, indices.data() + guard, guards.data());
        if (std::memcmp(input.data(), original.data(), input.size() * 4) || indices != originalIndices)
            return 1;
        for (unsigned row = 0; row < Rows; ++row) {
            for (unsigned block = 0; block < Cols; block += 32) {
                std::array<std::pair<uint32_t, uint32_t>, 32> expected;
                for (unsigned j = 0; j < 32; ++j)
                    expected[j] = {
                        std::bit_cast<uint32_t>(input[guard + row * Cols + block + j]),
                        indices[guard + (IndexRows == 1 ? 0 : row * Cols) + block + j]};
                std::stable_sort(expected.begin(), expected.end(), [](auto a, auto b) {
                    const float x = std::bit_cast<float>(a.first), y = std::bit_cast<float>(b.first);
                    return x != y ? x > y : a.second < b.second;
                });
                for (unsigned j = 0; j < 32; ++j) {
                    const unsigned index = guard + 2 * (row * Cols + block + j);
                    if (std::bit_cast<uint32_t>(output[index]) != expected[j].first ||
                        std::bit_cast<uint32_t>(output[index + 1]) != expected[j].second)
                        return 2;
                }
            }
        }
        for (unsigned i = 0; i < guard; ++i)
            if (std::bit_cast<uint32_t>(output[i]) != sentinel ||
                std::bit_cast<uint32_t>(output[guard + 2 * N + i]) != sentinel)
                return 3;
        for (auto value : guards)
            if (value != sentinel)
                return 4;
    }
#ifdef A1_SORT_NATIVE_MODEL
    constexpr unsigned batch = ScratchWords / 512;
    constexpr unsigned batches = Rows * ((Cols / 32 + batch - 1) / batch);
    if (modelVbsRepeats != 2 * N / 32 * 5 || modelConcatRepeats != modelVbsRepeats || modelMerges != N / 32 * 5 ||
        modelSets != batches * 2 * 5 || modelPending[0] || modelPending[1] || modelHigh != UINT64_MAX ||
        modelLow != UINT64_MAX)
        return 5;
#endif
    return 0;
}
#endif
