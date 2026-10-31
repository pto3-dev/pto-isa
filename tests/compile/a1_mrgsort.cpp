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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#endif
#ifndef MERGE_WAYS
#define MERGE_WAYS 4
#endif
#ifndef MERGE_P0
#define MERGE_P0 32
#endif
#ifndef MERGE_P1
#define MERGE_P1 48
#endif
#ifndef MERGE_P2
#define MERGE_P2 16
#endif
#ifndef MERGE_P3
#define MERGE_P3 64
#endif
#ifndef MERGE_EXHAUSTED
#define MERGE_EXHAUSTED false
#endif
constexpr unsigned Ways = MERGE_WAYS;
constexpr unsigned P0 = MERGE_P0, P1 = MERGE_P1, P2 = MERGE_P2, P3 = MERGE_P3;
constexpr unsigned Total = P0 + P1 + (Ways > 2 ? P2 : 0) + (Ways > 3 ? P3 : 0);
#ifndef MERGE_OUTPUT
#define MERGE_OUTPUT Total
#endif
constexpr unsigned OutputPairs = MERGE_OUTPUT;
template <unsigned Pairs>
using Compact = pto::Tile<pto::TileType::Vec, float, 1, Pairs * 2>;
using Src0 = Compact<P0>;
using Src1 = Compact<P1>;
using Src2 = Compact<P2>;
using Src3 = Compact<P3>;
using Result = Compact<OutputPairs>;
using Scratch = pto::Tile<pto::TileType::Vec, float, 1, Total * 16 + 16>;
using Guard = pto::Tile<pto::TileType::Vec, uint32_t, 1, 16>;
using Metadata = pto::Tile<pto::TileType::Vec, uint32_t, 1, 8>;
template <typename T, unsigned C>
using GM = pto::GlobalTensor<T, pto::Shape<1, 1, 1, 1, C>, pto::Stride<1, 1, 1, C, 1>>;
constexpr unsigned A0 = 64, A1 = A0 + P0 * 8 + 64, A2 = A1 + P1 * 8 + 64;
constexpr unsigned A3 = A2 + P2 * 8 + 64, AD = A3 + P3 * 8 + 64, AT = AD + OutputPairs * 8 + 64;
constexpr unsigned AM = AT + Total * 64 + 128;
static_assert(AM + 32 <= 128 * 1024);

#ifdef A1_MERGE_NATIVE_MODEL
// Only native vector operations are modeled; production code performs all
// packing, tie repair, bounds, compaction and consumed-count handling.
static std::array<const uint8_t*, 4> modelSources;
static std::array<unsigned, 4> modelBytes;
static unsigned modelPending[2], modelSets, modelGathers, modelDuplicates, modelMerges, modelBarriers;
static uint64_t modelHigh, modelLow;
static bool modelNormal, modelReadonlyAlias;
static void modelMergeSet(pipe_t from, pipe_t to, event_t id)
{
    const unsigned slot = from == PIPE_S ? 0 : 1;
    if (id != EVENT_ID7 || (from != PIPE_S && from != PIPE_V) || to != (from == PIPE_S ? PIPE_V : PIPE_S) ||
        modelPending[slot])
        std::abort();
    ++modelPending[slot];
    ++modelSets;
}
static void modelMergeWait(pipe_t from, pipe_t to, event_t id)
{
    const unsigned slot = from == PIPE_S ? 0 : 1;
    if (id != EVENT_ID7 || (from != PIPE_S && from != PIPE_V) || to != (from == PIPE_S ? PIPE_V : PIPE_S) ||
        !modelPending[slot])
        std::abort();
    --modelPending[slot];
}
static void modelMergeMask(uint64_t high, uint64_t low)
{
    modelHigh = high;
    modelLow = low;
}
static void modelMergeNormal() { modelNormal = true; }
static void modelMergeBarrier(pipe_t pipe)
{
    if (pipe != PIPE_V)
        std::abort();
    ++modelBarriers;
}
static void trap() { std::exit(86); }
static void vgather(uint32_t* dst, uint32_t* offsets, uint64_t config)
{
    if (!modelNormal || modelHigh != 0 || modelLow != UINT64_C(0x3030303030303030) || ((config >> 32) & 0xff) != 8 ||
        (config >> 56) != 2)
        std::abort();
    const uint8_t* source = nullptr;
    unsigned available = 0;
    for (unsigned list = 0; list < Ways; ++list) {
        const uint32_t delta = uint32_t(config) - uint32_t(reinterpret_cast<uintptr_t>(modelSources[list]));
        if (delta < modelBytes[list] && modelBytes[list] - delta > available) {
            source = modelSources[list] + delta;
            available = modelBytes[list] - delta;
        }
    }
    if (!source)
        std::abort();
    for (unsigned repeat = 0; repeat < 2; ++repeat)
        for (unsigned lane = 0; lane < 64; ++lane)
            if ((modelLow >> lane) & 1) {
                const unsigned i = repeat * 64 + lane;
                if (offsets[i] + 4 > available)
                    std::abort();
                std::memcpy(dst + i, source + offsets[i], 4);
            }
    ++modelGathers;
}
static void vector_dup(int32_t* dst, int32_t scalar, unsigned repeats, unsigned a, unsigned b, unsigned c, unsigned d)
{
    if (!modelNormal || modelHigh || modelLow != UINT64_C(0x0101010101010101) || (repeats == 0 || repeats > 255) ||
        a != 1 || b != 1 || c != 8 || d != 8)
        std::abort();
    for (unsigned i = 0; i < repeats * 64; ++i)
        if ((modelLow >> (i % 64)) & 1)
            dst[i] = scalar;
    modelDuplicates += repeats;
}
static void vmrgsort4(
    float* dst, float** lists, unsigned repeats, unsigned n0, unsigned n1, unsigned n2, unsigned n3, bool exhausted,
    unsigned mask)
{
    if (repeats != 1 || exhausted || mask != (1U << Ways) - 1 || modelLow != UINT64_MAX || modelHigh != UINT64_MAX ||
        modelBarriers == 0)
        std::abort();
    const unsigned lengths[4] = {n0, n1, n2, n3};
    unsigned consumed[4] = {0, 0, 0, 0};
    for (unsigned i = 0; i < n0 + n1 + n2 + n3; ++i) {
        int chosen = -1;
        for (unsigned list = 0; list < Ways; ++list) {
            if (consumed[list] < lengths[list] &&
                (chosen < 0 || lists[list][consumed[list] * 8 + 4] >= lists[chosen][consumed[chosen] * 8 + 4]))
                chosen = static_cast<int>(list);
        }
        if (chosen < 0)
            std::abort();
        // Deliberately prefer later lists for ties; preserve order within each
        // list. A production scalar repair must restore public earlier-list order.
        std::memcpy(dst + i * 8, lists[chosen] + consumed[chosen] * 8, 8 * sizeof(float));
        ++consumed[chosen];
    }
    ++modelMerges;
}
#undef set_vector_mask
#define set_vector_mask modelMergeMask
#undef set_mask_norm
#define set_mask_norm modelMergeNormal
#define pipe_barrier modelMergeBarrier
#define set_flag modelMergeSet
#define wait_flag modelMergeWait
#define PTO_NPU_ARCH_A1
#include <pto/npu/a1/TMrgSort.hpp>
#undef set_vector_mask
#undef set_mask_norm
#undef pipe_barrier
#undef set_flag
#undef wait_flag
#endif

// Use the exact public template argument order for CPU reference and native
// compilation. The native model calls the production implementation directly.
template <typename D, typename T, typename S0, typename S1, typename S2, typename S3>
AICORE void CallMerge(D& dst, pto::MrgSortExecutedNumList& executed, T& tmp, S0& a, S1& b, S2& c, S3& d)
{
#ifdef A1_MERGE_NATIVE_MODEL
#if MERGE_WAYS == 2
    pto::detail::A1MergeLists<MERGE_EXHAUSTED>(dst, executed, tmp, a, b);
#elif MERGE_WAYS == 3
    pto::detail::A1MergeLists<MERGE_EXHAUSTED>(dst, executed, tmp, a, b, c);
#else
    pto::detail::A1MergeLists<MERGE_EXHAUSTED>(dst, executed, tmp, a, b, c, d);
#endif
#else
#if MERGE_WAYS == 2
    pto::TMRGSORT<D, T, S0, S1, MERGE_EXHAUSTED>(dst, executed, tmp, a, b);
#elif MERGE_WAYS == 3
    pto::TMRGSORT<D, T, S0, S1, S2, MERGE_EXHAUSTED>(dst, executed, tmp, a, b, c);
#else
    pto::TMRGSORT<D, T, S0, S1, S2, S3, MERGE_EXHAUSTED>(dst, executed, tmp, a, b, c, d);
#endif
#endif
}
extern "C" __global__ AICORE void A1Merge(
    __gm__ float* output, __gm__ float* input0, __gm__ float* input1, __gm__ float* input2, __gm__ float* input3,
    __gm__ uint32_t* metadata, __gm__ uint32_t* guards)
{
    Src0 src0;
    Src1 src1;
    Src2 src2;
    Src3 src3;
    Result dst;
    Scratch tmp;
    Metadata counts;
    Guard before, after, dstBefore, dstAfter;
    pto::MrgSortExecutedNumList executed{};
    pto::TASSIGN<A0>(src0);
    pto::TASSIGN<A1>(src1);
    pto::TASSIGN<A2>(src2);
    pto::TASSIGN<A3>(src3);
    pto::TASSIGN<AD>(dst);
    pto::TASSIGN<AT>(tmp);
    pto::TASSIGN<AM>(counts);
    pto::TASSIGN<AT - 64>(before);
    pto::TASSIGN<AT + Total * 64>(after);
    pto::TASSIGN<AD - 64>(dstBefore);
    pto::TASSIGN<AD + OutputPairs * 8>(dstAfter);
#ifdef A1_MERGE_NATIVE_MODEL
    if (modelReadonlyAlias) {
        pto::TASSIGN<A0>(src1);
        pto::TASSIGN<A0>(src2);
        pto::TASSIGN<A0>(src3);
    }
#endif
    GM<float, P0 * 2> gm0(input0);
    GM<float, P1 * 2> gm1(input1);
    GM<float, P2 * 2> gm2(input2);
    GM<float, P3 * 2> gm3(input3);
    GM<float, OutputPairs * 2> gmOutput(output);
    GM<uint32_t, 8> gmCounts(metadata);
    GM<uint32_t, 16> g0(guards), g1(guards + 16), g2(guards + 32), g3(guards + 48);
    pto::TLOAD(src0, gm0);
    pto::TLOAD(src1, gm1);
    if constexpr (Ways > 2)
        pto::TLOAD(src2, gm2);
    if constexpr (Ways > 3)
        pto::TLOAD(src3, gm3);
    pto::TLOAD(dst, gmOutput);
    pto::TLOAD(counts, gmCounts);
    pto::TLOAD(before, g0);
    pto::TLOAD(after, g1);
    pto::TLOAD(dstBefore, g2);
    pto::TLOAD(dstAfter, g3);
#ifndef __CPU_SIM
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
#ifdef A1_MERGE_NATIVE_MODEL
    modelSources = {
        reinterpret_cast<const uint8_t*>(src0.data()), reinterpret_cast<const uint8_t*>(src1.data()),
        reinterpret_cast<const uint8_t*>(src2.data()), reinterpret_cast<const uint8_t*>(src3.data())};
    modelBytes = {P0 * 8, P1 * 8, P2 * 8, P3 * 8};
#endif
    CallMerge(dst, executed, tmp, src0, src1, src2, src3);
    counts.data()[0] = executed.mrgSortList0;
    counts.data()[1] = executed.mrgSortList1;
    counts.data()[2] = executed.mrgSortList2;
    counts.data()[3] = executed.mrgSortList3;
#ifdef __CPU_SIM
    if (std::memcmp(src0.data(), input0, P0 * 8) || std::memcmp(src1.data(), input1, P1 * 8) ||
        (Ways > 2 && std::memcmp(src2.data(), input2, P2 * 8)) ||
        (Ways > 3 && std::memcmp(src3.data(), input3, P3 * 8)))
        std::abort();
#else
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
#endif
    pto::TSTORE(gmOutput, dst);
    pto::TSTORE(gmCounts, counts);
    pto::TSTORE(g0, before);
    pto::TSTORE(g1, after);
    pto::TSTORE(g2, dstBefore);
    pto::TSTORE(g3, dstAfter);
}

#ifdef __CPU_SIM
struct ReferencePair {
    uint32_t score;
    uint32_t index;
    unsigned list;
};
int main(int argc, char** argv)
{
    constexpr unsigned guard = 16;
    constexpr uint32_t sentinel = 0xa55ac33cU;
    const std::array<unsigned, 4> lengths = {P0, P1, P2, P3};
#ifdef A1_MERGE_NATIVE_MODEL
    if (argc > 1 && argv[1][0] != 'r') {
        const unsigned alias = static_cast<unsigned>(std::strtoul(argv[1], nullptr, 10));
        Src0 a;
        Src1 b;
        Src2 c;
        Src3 d;
        Result dst;
        Scratch tmp;
        pto::TASSIGN<A0>(a);
        pto::TASSIGN<A1>(b);
        pto::TASSIGN<A2>(c);
        pto::TASSIGN<A3>(d);
        pto::TASSIGN<AD>(dst);
        pto::TASSIGN<AT>(tmp);
        const unsigned addresses[4] = {A0, A1, A2, A3};
        if (alias < Ways)
            pto::TASSIGN(dst, addresses[alias] + (alias == 0 ? 32 : 0));
        else if (alias < 2 * Ways)
            pto::TASSIGN(tmp, addresses[alias - Ways]);
        else if (alias == 2 * Ways)
            pto::TASSIGN(tmp, AD);
        else
            return 87;
        pto::MrgSortExecutedNumList executed{};
        CallMerge(dst, executed, tmp, a, b, c, d);
        return 88;
    }
    modelReadonlyAlias = argc > 1;
#else
    (void)argc;
    (void)argv;
#endif
    constexpr std::array<uint32_t, 8> edges = {0U, 0x80000000U, 0x7f800000U, 0xff800000U,
                                               1U, 0x80000001U, 0x7f7fffffU, 0xff7fffffU};
    for (unsigned dataset = 0; dataset < 6; ++dataset) {
        std::array<std::vector<float>, 4> inputs;
        std::vector<ReferencePair> merged;
        for (unsigned list = 0; list < 4; ++list) {
            inputs[list].resize(lengths[list] * 2 + 2 * guard, std::bit_cast<float>(sentinel));
            std::vector<ReferencePair> values;
            for (unsigned i = 0; i < lengths[list]; ++i) {
                const float score = dataset == 0 ? static_cast<float>((i * 13 + list * 7) % 41) - 20.0f :
                                    dataset == 1 ? 0.5f :
                                    dataset == 2 ? std::bit_cast<float>(i % 2 ? 0x80000000U : 0U) :
                                    dataset == 3 ? std::bit_cast<float>(edges[(i + list) % edges.size()]) :
                                    dataset == 4 ? static_cast<float>(list * 1000 + i) :
                                                   static_cast<float>((3 - list) * 1000 + i);
                const unsigned index = (i * 1664525U + 1013904223U) ^ (list * 0x80000001U);
#ifdef A1_MERGE_NATIVE_MODEL
                values.push_back({std::bit_cast<uint32_t>(modelReadonlyAlias ? 0.5f : score), index, list});
                if (modelReadonlyAlias)
                    values.back().index = i * 1664525U + 1013904223U;
#else
                values.push_back({std::bit_cast<uint32_t>(score), index, list});
#endif
            }
            std::stable_sort(values.begin(), values.end(), [](const auto& a, const auto& b) {
                return std::bit_cast<float>(a.score) > std::bit_cast<float>(b.score);
            });
            for (unsigned i = 0; i < lengths[list]; ++i) {
                inputs[list][guard + 2 * i] = std::bit_cast<float>(values[i].score);
                inputs[list][guard + 2 * i + 1] = std::bit_cast<float>(values[i].index);
            }
            if (list < Ways)
                merged.insert(merged.end(), values.begin(), values.end());
        }
        std::stable_sort(merged.begin(), merged.end(), [](const auto& a, const auto& b) {
            return std::bit_cast<float>(a.score) > std::bit_cast<float>(b.score);
        });
        std::vector<float> output(OutputPairs * 2 + 2 * guard, std::bit_cast<float>(sentinel));
        std::vector<float> expected = output;
        std::vector<uint32_t> counts(8 + 2 * guard, sentinel), guards(64, sentinel);
        std::array<unsigned, 4> consumed{};
        unsigned produced = 0;
        for (const auto& pair : merged) {
            expected[guard + produced * 2] = std::bit_cast<float>(pair.score);
            expected[guard + produced * 2 + 1] = std::bit_cast<float>(pair.index);
            ++produced;
            ++consumed[pair.list];
            if (produced == OutputPairs || (MERGE_EXHAUSTED && consumed[pair.list] == lengths[pair.list]))
                break;
        }
        const auto original = inputs;
#ifdef A1_MERGE_NATIVE_MODEL
        modelPending[0] = modelPending[1] = modelSets = modelGathers = modelDuplicates = modelMerges = modelBarriers =
            0;
        modelNormal = false;
        modelHigh = 0;
        modelLow = 1;
#endif
        A1Merge(
            output.data() + guard, inputs[0].data() + guard, inputs[1].data() + guard, inputs[2].data() + guard,
            inputs[3].data() + guard, counts.data() + guard, guards.data());
        for (unsigned list = 0; list < 4; ++list)
            if (std::memcmp(inputs[list].data(), original[list].data(), inputs[list].size() * 4))
                return 1;
        for (unsigned i = 0; i < output.size(); ++i)
            if (std::bit_cast<uint32_t>(output[i]) != std::bit_cast<uint32_t>(expected[i])) {
                std::fprintf(
                    stderr, "merge mismatch: dataset=%u word=%u expected=%08x actual=%08x\n", dataset, i,
                    std::bit_cast<uint32_t>(expected[i]), std::bit_cast<uint32_t>(output[i]));
                return 2;
            }
        for (unsigned i = 0; i < counts.size(); ++i) {
            const unsigned value = i >= guard && i < guard + 4 ? consumed[i - guard] : sentinel;
            if (counts[i] != value)
                return 3;
        }
        for (auto value : guards)
            if (value != sentinel)
                return 4;
#ifdef A1_MERGE_NATIVE_MODEL
        if (modelSets != 4 || modelPending[0] || modelPending[1] || modelMerges != 1 || modelGathers != Total / 16 ||
            modelDuplicates != Total / 8 || modelBarriers != Ways + 1 || modelHigh != UINT64_MAX ||
            modelLow != UINT64_MAX || !modelNormal)
            return 5;
#endif
    }
    return 0;
}
#endif
