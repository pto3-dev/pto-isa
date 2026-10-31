/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_A1_TGATHER_HPP
#define PTO_NPU_A1_TGATHER_HPP

#include <pto/npu/a1/TVectorUtils.hpp>

namespace pto {

// Mask gather selects a fixed sub-sequence from each source row. Full aligned
// rows contain complete pattern groups, so flattening preserves row semantics.
// m200 uses the ten-argument VREDUCE overload, not VREDUCEv2. No index/scratch
// bank is needed; raw 16/32-bit words also permit same-size dtype extraction.
template <typename DstTile, typename SrcTile, MaskPattern Pattern, auto Axis = GatherAxis::GATHER_ROW>
PTO_INTERNAL void TGATHER_IMPL(DstTile& dst, SrcTile& src)
{
    using Input = typename SrcTile::DType;
    using Output = typename DstTile::DType;
    constexpr unsigned pattern = static_cast<unsigned>(Pattern);
    constexpr unsigned factor = pattern <= 2 ? 2 : pattern <= 6 ? 4 : 1;
    constexpr unsigned lanes = 256 / sizeof(Input);
    constexpr unsigned full = SrcTile::Numel / lanes * lanes;
    constexpr unsigned tail = SrcTile::Numel % lanes;
    PTO_STATIC_ASSERT(pattern >= 1 && pattern <= 7, "A1 mask gather requires a fixed pattern in 1..7.");
    PTO_STATIC_ASSERT(Axis == GatherAxis::GATHER_ROW, "A1 mask gather supports the row axis only.");
    PTO_STATIC_ASSERT(
        (std::is_same<Input, half>::value || std::is_same<Input, float>::value || std::is_same<Input, int32_t>::value ||
         std::is_same<Input, uint32_t>::value) &&
            (std::is_same<Output, half>::value || std::is_same<Output, float>::value ||
             std::is_same<Output, int32_t>::value || std::is_same<Output, uint32_t>::value),
        "A1 mask gather requires FP16, FP32, INT32 or UINT32 storage.");
    PTO_STATIC_ASSERT(sizeof(Input) == sizeof(Output), "A1 mask gather requires equal element widths.");
    PTO_STATIC_ASSERT(
        DstTile::Loc == TileType::Vec && SrcTile::Loc == TileType::Vec, "A1 mask gather requires Vec storage.");
    detail::CheckA1VectorStorageLayout<DstTile>();
    detail::CheckA1VectorStorageLayout<SrcTile>();
    PTO_STATIC_ASSERT(
        DstTile::Numel == SrcTile::Numel / factor, "A1 mask gather destination must match the selected element count.");
    // Compact writes can clobber later reads. Keep all source/destination
    // overlaps fail-closed, including exact aliases, until separately qualified.
    if (detail::A1BuffersOverlap(dst, src)) {
        trap();
        return;
    }
    pipe_barrier(PIPE_V);
    set_mask_norm();
    if constexpr (Pattern == MaskPattern::P1111) {
        auto* output = reinterpret_cast<__ubuf__ uint16_t*>(dst.data());
        auto* input = reinterpret_cast<__ubuf__ uint16_t*>(src.data());
        constexpr unsigned halfwords = SrcTile::Numel * sizeof(Input) / 2;
        constexpr unsigned whole = halfwords / 128 * 128, remaining = halfwords % 128;
        for (unsigned offset = 0; offset < whole; offset += 255 * 128) {
            const unsigned count = (whole - offset) / 128;
            const uint8_t repeats = count > 255 ? 255 : count;
            detail::A1CopyBitRows<128>(output + offset, input + offset, repeats, 8, 8);
        }
        if constexpr (remaining != 0) {
            detail::A1CopyBitRows<remaining>(output + whole, input + whole, 1, 0);
        }
    } else {
        using Word = typename std::conditional<sizeof(Input) == 4, uint32_t, uint16_t>::type;
        auto* output = reinterpret_cast<__ubuf__ Word*>(dst.data());
        auto* input = reinterpret_cast<__ubuf__ Word*>(src.data());
        set_vector_mask(UINT64_MAX, UINT64_MAX);
        // Fixed patterns ignore source1; keep its pointer inside the source.
        // m200 packs output contiguously with the mandated dst repeat stride 8.
        for (unsigned offset = 0; offset < full; offset += 255 * lanes) {
            const unsigned count = (full - offset) / lanes;
            const uint8_t repeats = count > 255 ? 255 : count;
            vreduce(output + offset / factor, input + offset, input + offset, repeats, 1, 1, pattern, 8, 8, 0);
        }
        if constexpr (tail != 0) {
            // Counter mode represents the final partial source repeat. All
            // pattern groups are complete and no masked source word is read.
            set_mask_count();
            set_vector_mask(0, tail);
            vreduce(output + full / factor, input + full, input + full, 1, 1, 1, pattern, 8, 8, 0);
        }
    }
    set_mask_norm();
    set_vector_mask(UINT64_MAX, UINT64_MAX);
}

// Public indices are flat element offsets. Generate byte offsets on V and
// gather raw 16/32-bit words without converting floating-point or integer bits.
// Keep the established scratch capacity contract for existing callers.
template <typename DstTile, typename SrcTile, typename IndexTile, typename TmpTile>
PTO_INTERNAL void TGATHER_IMPL(DstTile& dst, SrcTile& src, IndexTile& indices, TmpTile& tmp)
{
    using Element = typename SrcTile::DType;
    using Index = typename IndexTile::DType;
    constexpr unsigned halves = sizeof(Element) / sizeof(uint16_t);
    constexpr unsigned indexLanes = 64;
    constexpr unsigned gatherLanes = 256 / sizeof(Element);
    constexpr unsigned elements = DstTile::Numel;
    constexpr unsigned gatheredHalves = elements * halves;
    PTO_STATIC_ASSERT(
        (std::is_same<Element, half>::value || std::is_same<Element, float>::value ||
         std::is_same<Element, int32_t>::value || std::is_same<Element, uint32_t>::value),
        "A1 indexed gather requires FP16, FP32, INT32 or UINT32 data.");
    PTO_STATIC_ASSERT(
        (std::is_same<Element, typename DstTile::DType>::value), "A1 indexed gather requires matching data types.");
    PTO_STATIC_ASSERT(
        (std::is_same<Index, int32_t>::value || std::is_same<Index, uint32_t>::value) &&
            (std::is_same<Index, typename TmpTile::DType>::value),
        "A1 indexed gather requires matching INT32/UINT32 indices and scratch.");
    PTO_STATIC_ASSERT(
        DstTile::Loc == TileType::Vec && SrcTile::Loc == TileType::Vec && IndexTile::Loc == TileType::Vec &&
            TmpTile::Loc == TileType::Vec,
        "A1 indexed gather requires Vec storage.");
    detail::CheckA1VectorStorageLayout<DstTile>();
    detail::CheckA1VectorStorageLayout<SrcTile>();
    detail::CheckA1VectorStorageLayout<IndexTile>();
    detail::CheckA1VectorStorageLayout<TmpTile>();
    PTO_STATIC_ASSERT(
        DstTile::Rows == IndexTile::Rows && DstTile::Cols == IndexTile::Cols,
        "A1 indexed gather requires destination and index shapes to match.");
    PTO_STATIC_ASSERT(
        TmpTile::Numel >= gatheredHalves, "A1 indexed gather scratch needs one 32-bit offset per 16-bit half.");
    // Even an exact source/destination alias can destroy a later indexed read.
    // Only the two read-only inputs may alias each other.
    if (detail::A1BuffersOverlap(dst, src) || detail::A1BuffersOverlap(dst, indices) ||
        detail::A1BuffersOverlap(tmp, src) || detail::A1BuffersOverlap(tmp, indices) ||
        detail::A1BuffersOverlap(tmp, dst)) {
        trap();
        return;
    }
    auto* offsets = reinterpret_cast<__ubuf__ uint32_t*>(tmp.data());
    auto* indexWords = reinterpret_cast<__ubuf__ int32_t*>(indices.data());
    auto* offsetWords = reinterpret_cast<__ubuf__ int32_t*>(offsets);
    // Preserve ordering for manual callers that filled indices on S.
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
    // Finish earlier vector readers/producers before recycling scratch.
    pipe_barrier(PIPE_V);
    set_mask_norm();
    // Valid source indices fit the UB budget, so their byte offsets fit INT32.
    // m200 supports integer VADD but not the 32-bit VSHL scalar overload.
    // One doubling forms FP16 offsets; two form FP32/INT32/UINT32 offsets.
    constexpr unsigned fullIndices = elements / indexLanes * indexLanes;
    constexpr unsigned tailIndices = elements % indexLanes;
    set_vector_mask(0, UINT64_MAX);
    for (unsigned offset = 0; offset < fullIndices; offset += 255 * indexLanes) {
        const unsigned remaining = (fullIndices - offset) / indexLanes;
        const uint8_t repeats = remaining > 255 ? 255 : remaining;
        vadd(offsetWords + offset, indexWords + offset, indexWords + offset, repeats, 1, 1, 1, 8, 8, 8);
    }
    if constexpr (tailIndices != 0) {
        set_vector_mask(0, (uint64_t{1} << tailIndices) - 1);
        vadd(offsetWords + fullIndices, indexWords + fullIndices, indexWords + fullIndices, 1, 1, 1, 1, 8, 8, 8);
    }
    pipe_barrier(PIPE_V);
    if constexpr (sizeof(Element) == 4) {
        set_vector_mask(0, UINT64_MAX);
        for (unsigned offset = 0; offset < fullIndices; offset += 255 * indexLanes) {
            const unsigned remaining = (fullIndices - offset) / indexLanes;
            const uint8_t repeats = remaining > 255 ? 255 : remaining;
            vadd(offsetWords + offset, offsetWords + offset, offsetWords + offset, repeats, 1, 1, 1, 8, 8, 8);
        }
        if constexpr (tailIndices != 0) {
            set_vector_mask(0, (uint64_t{1} << tailIndices) - 1);
            vadd(offsetWords + fullIndices, offsetWords + fullIndices, offsetWords + fullIndices, 1, 1, 1, 1, 8, 8, 8);
        }
        pipe_barrier(PIPE_V);
    }
    using Word = typename std::conditional<sizeof(Element) == 4, uint32_t, uint16_t>::type;
    auto* output = reinterpret_cast<__ubuf__ Word*>(dst.data());
    const uint64_t sourceAddress = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(src.data()));
    constexpr unsigned full = elements / gatherLanes * gatherLanes;
    constexpr unsigned tail = elements % gatherLanes;
    set_mask_norm();
    set_vector_mask(UINT64_MAX, UINT64_MAX);
    // A repeat writes 256 contiguous bytes and consumes one offset per word.
    // Split at the 8-bit repeat-count limit; restore the full mask on return.
    for (unsigned offset = 0; offset < full; offset += 255 * gatherLanes) {
        const unsigned remaining = (full - offset) / gatherLanes;
        const uint64_t repeats = remaining > 255 ? 255 : remaining;
        const uint64_t config = sourceAddress | (uint64_t{8} << 32) | (repeats << 56);
        vgather(output + offset, offsets + offset, config);
    }
    if constexpr (tail != 0) {
        constexpr uint64_t low = tail >= 64 ? UINT64_MAX : (uint64_t{1} << (tail % 64)) - 1;
        constexpr uint64_t high = tail <= 64 ? 0 : (uint64_t{1} << (tail > 64 ? tail - 64 : 0)) - 1;
        set_vector_mask(high, low);
        vgather(output + full, offsets + full, sourceAddress | (uint64_t{8} << 32) | (uint64_t{1} << 56));
    }
    set_vector_mask(UINT64_MAX, UINT64_MAX);
    // The next scalar user may reuse tmp immediately after this call.
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
}

} // namespace pto

#endif // PTO_NPU_A1_TGATHER_HPP
