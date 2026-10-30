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

// The public indices are flat element offsets. m200 only has a 16-bit
// vgather, so a 32-bit source is copied as two adjacent halves, preserving
// NaN payloads and integer bits. The caller owns the explicit offset scratch.
template <typename DstTile, typename SrcTile, typename IndexTile, typename TmpTile>
PTO_INTERNAL void TGATHER_IMPL(DstTile& dst, SrcTile& src, IndexTile& indices, TmpTile& tmp)
{
    using Element = typename SrcTile::DType;
    using Index = typename IndexTile::DType;
    constexpr unsigned halves = sizeof(Element) / sizeof(uint16_t);
    constexpr unsigned halfLanes = 128;
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
    // A prior vector producer may have filled the indices, source or recycled
    // scratch. ID 7 is reserved for local ISA helpers and drained before return.
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    for (unsigned i = 0; i < elements; ++i) {
        const uint32_t byteOffset = static_cast<uint32_t>(indices.data()[i]) * sizeof(Element);
        for (unsigned h = 0; h < halves; ++h)
            offsets[i * halves + h] = byteOffset + h * sizeof(uint16_t);
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
    auto* output = reinterpret_cast<__ubuf__ uint16_t*>(dst.data());
    const uint64_t sourceAddress = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(src.data()));
    constexpr unsigned full = gatheredHalves / halfLanes * halfLanes;
    constexpr unsigned tail = gatheredHalves % halfLanes;
    set_mask_norm();
    set_vector_mask(UINT64_MAX, UINT64_MAX);
    // A gather repeat writes 256 contiguous bytes (dst stride 8 * 32B) and
    // consumes 128 uint32 offsets. Split at the 8-bit repeat-count limit.
    for (unsigned offset = 0; offset < full; offset += 255 * halfLanes) {
        const unsigned remaining = (full - offset) / halfLanes;
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
