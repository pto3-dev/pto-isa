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
#include <vector>
#endif

#ifndef PIPELINE_WAYS
#define PIPELINE_WAYS 4
#endif
#ifndef PIPELINE_OUTPUT
#define PIPELINE_OUTPUT (PIPELINE_WAYS * 32)
#endif
constexpr unsigned Ways = PIPELINE_WAYS, Values = Ways * 32, OutputPairs = PIPELINE_OUTPUT;
static_assert(Ways >= 2 && Ways <= 4 && OutputPairs > 0 && OutputPairs <= Values && OutputPairs % 4 == 0);
using Table = pto::Tile<pto::TileType::Vec, float, 3, 32>;
using Index = pto::Tile<pto::TileType::Vec, uint32_t, 1, Values>;
using Gathered = pto::Tile<pto::TileType::Vec, float, 1, Values>;
using Sorted = pto::Tile<pto::TileType::Vec, float, 1, Values * 2>;
using Run = pto::Tile<pto::TileType::Vec, float, 1, 64>;
using Result = pto::Tile<pto::TileType::Vec, float, 1, OutputPairs * 2>;
using GatherTmp = pto::Tile<pto::TileType::Vec, uint32_t, 1, Values * 2>;
using SortTmp = pto::Tile<pto::TileType::Vec, float, 1, 512>;
using MergeTmp = pto::Tile<pto::TileType::Vec, float, 1, Values * 16>;
using Counts = pto::Tile<pto::TileType::Vec, uint32_t, 1, 8>;
template <typename T, unsigned R, unsigned C>
using GM = pto::GlobalTensor<T, pto::Shape<1, 1, 1, R, C>, pto::Stride<1, 1, 1, C, 1>>;

// Leave a full block between live allocations. Run views below are read-only
// aliases of the compact sort output; no extra GM roundtrip splits the stages.
constexpr unsigned TableAddr = 64, PositionAddr = TableAddr + 96 * 4 + 64;
constexpr unsigned OrdinalAddr = PositionAddr + Values * 4 + 64;
constexpr unsigned GatherAddr = OrdinalAddr + Values * 4 + 64;
constexpr unsigned SortedAddr = GatherAddr + Values * 4 + 64;
constexpr unsigned GatherTmpAddr = SortedAddr + Values * 8 + 64;
constexpr unsigned SortTmpAddr = GatherTmpAddr + Values * 8 + 64;
constexpr unsigned MergeTmpAddr = SortTmpAddr + 512 * 4 + 64;
constexpr unsigned ResultAddr = MergeTmpAddr + Values * 64 + 64;
constexpr unsigned CountsAddr = ResultAddr + OutputPairs * 8 + 64;
static_assert(CountsAddr + 32 <= 128 * 1024);

extern "C" __global__ AICORE void GatherSortMerge(
    __gm__ float* output, __gm__ float* source, __gm__ uint32_t* positions, __gm__ uint32_t* ordinals,
    __gm__ uint32_t* consumed)
{
    Table table;
    Index index, ordinal;
    Gathered values;
    Sorted sorted;
    Run run0, run1, run2, run3;
    Result result;
    GatherTmp gatherTmp;
    SortTmp sortTmp;
    MergeTmp mergeTmp;
    Counts counts;
    pto::TASSIGN(table, TableAddr);
    pto::TASSIGN(index, PositionAddr);
    pto::TASSIGN(ordinal, OrdinalAddr);
    pto::TASSIGN(values, GatherAddr);
    pto::TASSIGN(sorted, SortedAddr);
    pto::TASSIGN(run0, SortedAddr);
    pto::TASSIGN(run1, SortedAddr + 64 * 4);
    if constexpr (Ways > 2)
        pto::TASSIGN(run2, SortedAddr + 2 * 64 * 4);
    if constexpr (Ways > 3)
        pto::TASSIGN(run3, SortedAddr + 3 * 64 * 4);
    pto::TASSIGN(result, ResultAddr);
    pto::TASSIGN(gatherTmp, GatherTmpAddr);
    pto::TASSIGN(sortTmp, SortTmpAddr);
    pto::TASSIGN(mergeTmp, MergeTmpAddr);
    pto::TASSIGN(counts, CountsAddr);
    GM<float, 3, 32> tableGM(source);
    GM<uint32_t, 1, Values> positionGM(positions), ordinalGM(ordinals);
    GM<float, 1, OutputPairs * 2> outputGM(output);
    GM<uint32_t, 1, 8> countsGM(consumed);
    pto::TLOAD(table, tableGM);
    pto::TLOAD(index, positionGM);
    pto::TLOAD(ordinal, ordinalGM);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pto::TGATHER(values, table, index, gatherTmp);
    pto::TSORT32(sorted, values, ordinal, sortTmp);
    pto::MrgSortExecutedNumList executed{};
    if constexpr (Ways == 2) {
        pto::TMRGSORT<Result, MergeTmp, Run, Run, false>(result, executed, mergeTmp, run0, run1);
    } else if constexpr (Ways == 3) {
        pto::TMRGSORT<Result, MergeTmp, Run, Run, Run, false>(result, executed, mergeTmp, run0, run1, run2);
    } else {
        pto::TMRGSORT<Result, MergeTmp, Run, Run, Run, Run, false>(result, executed, mergeTmp, run0, run1, run2, run3);
    }
    const uint32_t copied[8] = {
        executed.mrgSortList0, executed.mrgSortList1, executed.mrgSortList2, executed.mrgSortList3, 0, 0, 0, 0};
    for (unsigned i = 0; i < 8; ++i)
        counts.data()[i] = copied[i];
    // Both final compact pairs and count metadata are scalar-written. Compiler
    // users get equivalent dependencies from the compound S/V scheduling model.
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    pto::TSTORE(outputGM, result);
    pto::TSTORE(countsGM, counts);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

#ifdef __CPU_SIM
namespace {
constexpr unsigned Guard = 16;
constexpr uint32_t Sentinel = 0xa55ac33cU;
struct Pair {
    float score;
    uint32_t ordinal;
    unsigned source;
};

float scoreFor(unsigned dataset, unsigned i)
{
    constexpr std::array<uint32_t, 8> patterns = {0U,          0x80000000U, 0x7f800000U, 0xff800000U,
                                                  0x3f800000U, 0xbf800000U, 1U,          0x80000001U};
    if (dataset == 0)
        return static_cast<float>((i * 17) % 97) - 48.0f;
    if (dataset == 1)
        return 0.5f;
    if (dataset == 2)
        return std::bit_cast<float>(patterns[i % patterns.size()]);
    return static_cast<float>((i * (dataset + 7)) % (dataset * 11));
}

bool equalBits(const std::vector<float>& x, const std::vector<float>& y)
{
    for (unsigned i = 0; i < x.size(); ++i) {
        if (std::bit_cast<uint32_t>(x[i]) != std::bit_cast<uint32_t>(y[i])) {
            std::fprintf(stderr, "pipeline score/payload/guard mismatch at %u\n", i);
            return false;
        }
    }
    return true;
}

bool checkDataset(unsigned dataset)
{
    std::vector<float> table(96 + Guard * 2, std::bit_cast<float>(Sentinel));
    std::vector<uint32_t> positions(Values + Guard * 2, Sentinel), ordinals(Values + Guard * 2, Sentinel);
    std::vector<Pair> expected;
    for (unsigned i = 0; i < 96; ++i)
        table[Guard + i] = scoreFor(dataset, i);
    for (unsigned i = 0; i < Values; ++i) {
        positions[Guard + i] = (i * 37 + dataset * 13) % 96;
        // Ordinals increase across runs so stable list precedence also gives
        // exact unsigned-index tie ordering after the multi-list merge.
        ordinals[Guard + i] = 0x80000000U + i * 65537U;
        expected.push_back({table[Guard + positions[Guard + i]], ordinals[Guard + i], i / 32});
    }
    std::stable_sort(expected.begin(), expected.end(), [](const Pair& a, const Pair& b) {
        return a.score != b.score ? a.score > b.score : a.ordinal < b.ordinal;
    });
    const auto oldTable = table;
    const auto oldPositions = positions, oldOrdinals = ordinals;
    std::vector<float> reference(OutputPairs * 2 + Guard * 2, std::bit_cast<float>(Sentinel));
    std::vector<uint32_t> expectedCounts(8 + Guard * 2, Sentinel);
    for (unsigned i = 0; i < 8; ++i)
        expectedCounts[Guard + i] = 0;
    for (unsigned i = 0; i < OutputPairs; ++i) {
        reference[Guard + i * 2] = expected[i].score;
        reference[Guard + i * 2 + 1] = std::bit_cast<float>(expected[i].ordinal);
        ++expectedCounts[Guard + expected[i].source];
    }
    // Reuse the same UB addresses without clearing the memory model between calls.
    for (unsigned replay = 0; replay < 2; ++replay) {
        std::vector<float> output(OutputPairs * 2 + Guard * 2, std::bit_cast<float>(Sentinel));
        std::vector<uint32_t> counts(8 + Guard * 2, Sentinel);
        GatherSortMerge(
            output.data() + Guard, table.data() + Guard, positions.data() + Guard, ordinals.data() + Guard,
            counts.data() + Guard);
        if (!equalBits(output, reference) || !equalBits(table, oldTable) || positions != oldPositions ||
            ordinals != oldOrdinals || counts != expectedCounts)
            return false;
    }
    return true;
}
} // namespace

int main()
{
    for (unsigned dataset = 0; dataset < 6; ++dataset) {
        if (!checkDataset(dataset))
            return 1;
    }
    return 0;
}
#endif
