/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_A1_TMRGSORT_HPP
#define PTO_NPU_A1_TMRGSORT_HPP

#include <pto/npu/a1/TVectorUtils.hpp>

namespace pto {
namespace detail {

template <typename TileData>
PTO_INTERNAL void CheckA1MergeTile()
{
    PTO_STATIC_ASSERT(
        (TileData::Loc == TileType::Vec && TileData::Rows == 1 && std::is_same<typename TileData::DType, float>::value),
        "A1 TMRGSORT requires one-row FP32 Vec tiles.");
    CheckA1VectorStorageLayout<TileData>();
}

// The temporary's first bank holds eight-field input proposals. Its second
// bank is first a reusable 128-word gather-offset table, then the merged list.
// Source compact pairs remain read-only; no fixed/hidden UB address is used.
PTO_INTERNAL void StageA1MergeList(
    __ubuf__ float* proposals, __ubuf__ float* source, __ubuf__ uint32_t* offsets, unsigned pairs, unsigned list)
{
    constexpr unsigned proposalFields = 8, gatherPairs = 16;
    constexpr uint64_t pairFieldsMask = UINT64_C(0x3030303030303030);
    constexpr uint64_t listFieldMask = UINT64_C(0x0101010101010101);
    set_vector_mask(0, listFieldMask);
    constexpr unsigned proposalsPerRepeat = 8, maxRepeats = 255;
    for (unsigned first = 0; first < pairs; first += proposalsPerRepeat * maxRepeats) {
        const unsigned remaining = (pairs - first) / proposalsPerRepeat;
        const unsigned repeats = remaining < maxRepeats ? remaining : maxRepeats;
        vector_dup(
            reinterpret_cast<__ubuf__ int32_t*>(proposals + first * proposalFields), static_cast<int32_t>(list),
            repeats, 1, 1, 8, 8);
    }
    // Tags and gathered fields occupy the same UB blocks. Complete tag writes
    // before the masked gather, without adding a barrier for every 16 pairs.
    pipe_barrier(PIPE_V);
    set_vector_mask(0, pairFieldsMask);
    for (unsigned first = 0; first < pairs; first += gatherPairs) {
        auto* destination = proposals + first * proposalFields;
        const uint64_t sourceAddress = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(source + first * 2));
        vgather(
            reinterpret_cast<__ubuf__ uint32_t*>(destination), offsets,
            sourceAddress | (uint64_t{8} << 32) | (uint64_t{2} << 56));
    }
}

template <bool Exhausted, typename DstTile>
PTO_INTERNAL void CompactA1Merge(
    DstTile& dst, __ubuf__ float* merged, const unsigned* lengths, unsigned ways, unsigned total,
    MrgSortExecutedNumList& executed)
{
    constexpr unsigned proposalFields = 8, scoreField = 4, indexField = 5;
    auto* words = reinterpret_cast<__ubuf__ uint32_t*>(merged);
    auto* output = reinterpret_cast<__ubuf__ uint32_t*>(dst.data());
    unsigned consumed[4] = {0, 0, 0, 0};
    unsigned written = 0;
    bool complete = false;
    for (unsigned first = 0; first < total && !complete;) {
        unsigned last = first + 1;
        const float score = merged[first * proposalFields + scoreField];
        while (last < total && merged[last * proposalFields + scoreField] == score)
            ++last;
        // Native merge compares scores only. Restore earlier-list tie priority
        // before truncation/exhaustion, retaining each input's relative order.
        // At most four scans per equal-score run; no quadratic insertion sort.
        for (unsigned list = 0; list < ways && !complete; ++list) {
            for (unsigned item = first; item < last; ++item) {
                const unsigned base = item * proposalFields;
                if (words[base] != list)
                    continue;
                output[2 * written] = words[base + scoreField];
                output[2 * written + 1] = words[base + indexField];
                ++written;
                ++consumed[list];
                complete = written == DstTile::Numel / 2 || (Exhausted && consumed[list] == lengths[list]);
                if (complete)
                    break;
            }
        }
        first = last;
    }
    executed = {
        static_cast<uint16_t>(consumed[0]), static_cast<uint16_t>(consumed[1]), static_cast<uint16_t>(consumed[2]),
        static_cast<uint16_t>(consumed[3])};
}

template <bool Exhausted, typename DstTile, typename TmpTile, typename... SrcTiles>
PTO_INTERNAL void A1MergeLists(DstTile& dst, MrgSortExecutedNumList& executed, TmpTile& tmp, SrcTiles&... sources)
{
    constexpr unsigned ways = sizeof...(SrcTiles);
    PTO_STATIC_ASSERT(
        ((static_cast<uint64_t>(SrcTiles::Numel) <= PTO_UBUF_SIZE_BYTES / sizeof(float)) && ...),
        "A1 TMRGSORT source capacity exceeds the configured UB budget.");
    constexpr uint64_t sourceWords = (uint64_t{0} + ... + static_cast<uint64_t>(SrcTiles::Numel));
    PTO_STATIC_ASSERT(
        sourceWords <= PTO_UBUF_SIZE_BYTES / (8 * sizeof(float)),
        "A1 TMRGSORT proposal scratch exceeds the configured UB budget.");
    constexpr unsigned total = static_cast<unsigned>(sourceWords / 2);
    constexpr unsigned proposalFields = 8, offsetWords = 128;
    PTO_STATIC_ASSERT(ways >= 2 && ways <= 4, "A1 TMRGSORT requires two to four source lists.");
    CheckA1MergeTile<DstTile>();
    CheckA1MergeTile<TmpTile>();
    (CheckA1MergeTile<SrcTiles>(), ...);
    PTO_STATIC_ASSERT(
        ((SrcTiles::Numel >= 32 && SrcTiles::Numel % 32 == 0) && ...),
        "A1 TMRGSORT source lists require complete 16-pair groups.");
    PTO_STATIC_ASSERT(
        DstTile::Numel >= 8 && DstTile::Numel / 2 <= total,
        "A1 TMRGSORT destination cannot exceed the total input pair count.");
    PTO_STATIC_ASSERT(
        TmpTile::Numel >= total * 2 * proposalFields,
        "A1 TMRGSORT scratch needs two eight-field proposal banks for all input pairs.");
    const bool overlap = A1BuffersOverlap(dst, tmp) || (A1BuffersOverlap(dst, sources) || ...) ||
                         (A1BuffersOverlap(tmp, sources) || ...);
    if (overlap) {
        trap();
        return;
    }
    auto* proposals = tmp.data();
    auto* merged = proposals + total * proposalFields;
    auto* offsets = reinterpret_cast<__ubuf__ uint32_t*>(merged);
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    for (unsigned word = 0; word < offsetWords; ++word)
        offsets[word] = (word / proposalFields) * 8 + (word % proposalFields == 5 ? 4 : 0);
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
    __ubuf__ float* inputs[ways] = {sources.data()...};
    unsigned lengths[4] = {static_cast<unsigned>(SrcTiles::Numel / 2)...};
    __ubuf__ float* lists[4] = {proposals, proposals, proposals, proposals};
    unsigned first = 0;
    set_mask_norm();
    for (unsigned list = 0; list < ways; ++list) {
        lists[list] = proposals + first * proposalFields;
        StageA1MergeList(lists[list], inputs[list], offsets, lengths[list], list);
        first += lengths[list];
    }
    set_vector_mask(UINT64_MAX, UINT64_MAX);
    pipe_barrier(PIPE_V);
    // Always materialize the full native merge so software can repair equal
    // scores before applying the public destination/exhaustion boundary.
    vmrgsort4(merged, lists, 1, lengths[0], lengths[1], lengths[2], lengths[3], false, (1U << ways) - 1);
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    CompactA1Merge<Exhausted>(dst, merged, lengths, ways, total, executed);
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

} // namespace detail
} // namespace pto

#endif // PTO_NPU_A1_TMRGSORT_HPP
