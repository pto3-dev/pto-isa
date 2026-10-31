/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_A1_TSORT32_HPP
#define PTO_NPU_A1_TSORT32_HPP

#include <pto/npu/a1/TVectorUtils.hpp>

namespace pto {

// m200 sorts 16 eight-field proposals, whereas TSORT32 exposes 32 compact
// [float score, uint32 index] pairs. Two caller-owned banks avoid hidden UB
// allocation; each 32-value block needs 2 * 32 * 8 float slots. Larger scratch
// batches blocks and amortizes the scalar/vector handoffs.
template <typename DstTile, typename SrcTile, typename IndexTile, typename TmpTile>
PTO_INTERNAL void TSORT32_IMPL(DstTile& dst, SrcTile& src, IndexTile& indices, TmpTile& tmp)
{
    constexpr unsigned blockValues = 32;
    constexpr unsigned proposalFields = 8;
    constexpr unsigned scoreField = 4;
    constexpr unsigned indexField = 5;
    constexpr unsigned wordsPerBlock = blockValues * proposalFields;
    constexpr unsigned scratchPerBlock = 2 * wordsPerBlock;
    constexpr unsigned repeatsPerBlock = 2; // Native VBS16 produces the two sorted halves.
    constexpr unsigned maxBlocksPerCall = 255 / repeatsPerBlock;
    PTO_STATIC_ASSERT(
        (std::is_same<typename SrcTile::DType, float>::value && std::is_same<typename DstTile::DType, float>::value &&
         std::is_same<typename TmpTile::DType, float>::value &&
         std::is_same<typename IndexTile::DType, uint32_t>::value),
        "A1 TSORT32 requires FP32 values/destination/scratch and UINT32 indices.");
    PTO_STATIC_ASSERT(
        DstTile::Loc == TileType::Vec && SrcTile::Loc == TileType::Vec && IndexTile::Loc == TileType::Vec &&
            TmpTile::Loc == TileType::Vec,
        "A1 TSORT32 requires Vec storage.");
    detail::CheckA1VectorStorageLayout<DstTile>();
    detail::CheckA1VectorStorageLayout<SrcTile>();
    detail::CheckA1VectorStorageLayout<IndexTile>();
    detail::CheckA1VectorStorageLayout<TmpTile>();
    PTO_STATIC_ASSERT(
        SrcTile::Cols % blockValues == 0 && DstTile::Rows == SrcTile::Rows && DstTile::Cols == 2 * SrcTile::Cols &&
            IndexTile::Cols == SrcTile::Cols && (IndexTile::Rows == 1 || IndexTile::Rows == SrcTile::Rows),
        "A1 TSORT32 requires full 32-value blocks, expanded destination and matching/broadcast indices.");
    PTO_STATIC_ASSERT(
        TmpTile::Numel >= scratchPerBlock,
        "A1 TSORT32 scratch needs at least 512 FP32 elements for two proposal banks.");
    // A destination or scratch overlap can destroy an unread later block.
    // The two read-only inputs may alias; all writable overlaps are rejected.
    if (detail::A1BuffersOverlap(dst, src) || detail::A1BuffersOverlap(dst, indices) ||
        detail::A1BuffersOverlap(tmp, src) || detail::A1BuffersOverlap(tmp, indices) ||
        detail::A1BuffersOverlap(tmp, dst)) {
        trap();
        return;
    }
    constexpr unsigned scratchBlocks = TmpTile::Numel / scratchPerBlock;
    constexpr unsigned batchBlocks = scratchBlocks < maxBlocksPerCall ? scratchBlocks : maxBlocksPerCall;
    constexpr unsigned rowBlocks = SrcTile::Cols / blockValues;
    auto* proposals = tmp.data();
    auto* sorted = proposals + batchBlocks * wordsPerBlock;
    auto* proposalBits = reinterpret_cast<__ubuf__ uint32_t*>(proposals);
    auto* outputBits = reinterpret_cast<__ubuf__ uint32_t*>(dst.data());
    set_mask_norm();
    set_vector_mask(UINT64_MAX, UINT64_MAX);
    for (unsigned row = 0; row < SrcTile::Rows; ++row) {
        for (unsigned block = 0; block < rowBlocks; block += batchBlocks) {
            const unsigned count = rowBlocks - block < batchBlocks ? rowBlocks - block : batchBlocks;
            const unsigned sourceOffset = row * SrcTile::Cols + block * blockValues;
            const unsigned indexOffset = (IndexTile::Rows == 1 ? 0 : row * IndexTile::Cols) + block * blockValues;
            uint64_t fields[proposalFields] = {
                0,
                0,
                0,
                0,
                reinterpret_cast<uint64_t>(src.data() + sourceOffset),
                reinterpret_cast<uint64_t>(indices.data() + indexOffset),
                0,
                0};
            set_va_reg_sb(VA0, fields);
            // Only fields 4/5 are consumed; all other proposal fields are opaque
            // payload that is discarded when compact pairs are restored.
            scatter_vconcat_f32(proposals, VA0, count * repeatsPerBlock, 0b110000);
            pipe_barrier(PIPE_V);
            vbitsort(sorted, proposals, count * repeatsPerBlock);
            pipe_barrier(PIPE_V);
            for (unsigned b = 0; b < count; ++b) {
                auto* base = sorted + b * wordsPerBlock;
                __ubuf__ float* lists[4] = {base, base + wordsPerBlock / 2, base, base};
                vmrgsort4(proposals + b * wordsPerBlock, lists, 1, 16, 16, 0, 0, false, 0b0011);
            }
            // Internal event 7 is reserved by the A1 compiler and drained here.
            // Native proposal instructions compare scores only. Scalar extraction
            // restores unsigned index order within equal-score runs, preserving
            // raw score bits (including signed zero) and exact 32-bit payloads.
            set_flag(PIPE_V, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
            for (unsigned b = 0; b < count; ++b) {
                const unsigned destination = 2 * (sourceOffset + b * blockValues);
                for (unsigned i = 0; i < blockValues; ++i) {
                    const unsigned proposal = b * wordsPerBlock + i * proposalFields;
                    const uint32_t scoreBits = proposalBits[proposal + scoreField];
                    const uint32_t index = proposalBits[proposal + indexField];
                    const float score = proposals[proposal + scoreField];
                    unsigned position = i;
                    while (position > 0 && dst.data()[destination + 2 * (position - 1)] == score &&
                           outputBits[destination + 2 * (position - 1) + 1] > index) {
                        outputBits[destination + 2 * position] = outputBits[destination + 2 * (position - 1)];
                        outputBits[destination + 2 * position + 1] = outputBits[destination + 2 * (position - 1) + 1];
                        --position;
                    }
                    outputBits[destination + 2 * position] = scoreBits;
                    outputBits[destination + 2 * position + 1] = index;
                }
            }
            // Publish scalar-written pairs to the outward PIPE_V contract, and
            // finish all scalar scratch reads before the next batch reuses it.
            set_flag(PIPE_S, PIPE_V, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
        }
    }
}

} // namespace pto

#endif // PTO_NPU_A1_TSORT32_HPP
