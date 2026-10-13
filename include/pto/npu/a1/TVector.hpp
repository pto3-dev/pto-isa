/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_A1_TVECTOR_HPP
#define PTO_NPU_A1_TVECTOR_HPP

#include <pto/common/buffer_limits.hpp>
#include <pto/common/pto_tile.hpp>

#ifdef __PTO_AUTO__
#error "A1 FP16 vector instructions currently require manual allocation and synchronization."
#endif

namespace pto {
namespace detail {

template <typename TileData>
PTO_INTERNAL void CheckA1VectorTile()
{
    PTO_STATIC_ASSERT(TileData::Loc == TileType::Vec, "A1 vector instructions require a Vec tile.");
    PTO_STATIC_ASSERT((std::is_same<typename TileData::DType, half>::value), "A1 vector instructions require FP16.");
    PTO_STATIC_ASSERT(
        TileData::isRowMajor && TileData::SFractal == SLayout::NoneBox,
        "A1 vector instructions require contiguous row-major tiles.");
    PTO_STATIC_ASSERT(
        TileData::ValidRow == TileData::Rows && TileData::ValidCol == TileData::Cols,
        "A1 vector instructions require a static full valid shape.");
    PTO_STATIC_ASSERT(
        TileData::RowStride == TileData::Cols && TileData::Cols % 16 == 0,
        "A1 vector rows must be aligned to 32 bytes without padding.");
    PTO_STATIC_ASSERT(
        TileData::Numel * sizeof(half) <= PTO_UBUF_SIZE_BYTES, "A1 vector tile exceeds the configured UB budget.");
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void CheckA1VectorTransfer()
{
    CheckA1VectorTile<TileData>();
    PTO_STATIC_ASSERT((std::is_same<typename GlobalData::RawDType, half>::value), "A1 DMA requires FP16 tensors.");
    PTO_STATIC_ASSERT(GlobalData::layout == Layout::ND, "A1 vector DMA requires ND tensors.");
    PTO_STATIC_ASSERT(
        GlobalData::staticShape[0] == 1 && GlobalData::staticShape[1] == 1 && GlobalData::staticShape[2] == 1 &&
            GlobalData::staticShape[3] == TileData::Rows && GlobalData::staticShape[4] == TileData::Cols,
        "A1 vector DMA requires matching static two-dimensional tensor and tile shapes.");
    PTO_STATIC_ASSERT(
        GlobalData::staticStride[3] == TileData::Cols && GlobalData::staticStride[4] == 1,
        "A1 vector DMA requires contiguous tensor rows.");
}

} // namespace detail

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLOAD_IMPL(TileData& dst, GlobalData& src)
{
    detail::CheckA1VectorTransfer<TileData, GlobalData>();
    copy_gm_to_ubuf(dst.data(), src.data(), 0, 1, TileData::Numel * sizeof(half) / 32, 0, 0);
}

template <
    typename TileData, typename GlobalData, AtomicType Atomic = AtomicType::AtomicNone,
    STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TSTORE_IMPL(GlobalData& dst, TileData& src)
{
    detail::CheckA1VectorTransfer<TileData, GlobalData>();
    PTO_STATIC_ASSERT(
        Atomic == AtomicType::AtomicNone && Phase == STPhase::Unspecified,
        "A1 vector stores do not support atomic or unit-flag modes.");
    set_atomic_none();
    copy_ubuf_to_gm(dst.data(), src.data(), 0, 1, TileData::Numel * sizeof(half) / 32, 0, 0);
}

template <typename DstTile, typename Src0Tile, typename Src1Tile>
PTO_INTERNAL void TADD_IMPL(DstTile& dst, Src0Tile& src0, Src1Tile& src1)
{
    detail::CheckA1VectorTile<DstTile>();
    detail::CheckA1VectorTile<Src0Tile>();
    detail::CheckA1VectorTile<Src1Tile>();
    PTO_STATIC_ASSERT(
        DstTile::Rows == Src0Tile::Rows && DstTile::Rows == Src1Tile::Rows && DstTile::Cols == Src0Tile::Cols &&
            DstTile::Cols == Src1Tile::Cols,
        "A1 TADD requires matching tile shapes.");
    PTO_STATIC_ASSERT(DstTile::Numel % 128 == 0, "A1 TADD requires complete 128-element vector repeats.");
    set_mask_norm();
    set_vector_mask(UINT64_MAX, UINT64_MAX);
    // The instruction repeat field has eight bits. Every chunk covers full
    // 256-byte repeats, including tiles with more than 255 repeats.
    for (unsigned offset = 0; offset < DstTile::Numel; offset += 255 * 128) {
        const unsigned remaining = (DstTile::Numel - offset) / 128;
        const uint8_t repeats = remaining > 255 ? 255 : remaining;
        vadd(dst.data() + offset, src0.data() + offset, src1.data() + offset, repeats, 1, 1, 1, 8, 8, 8);
    }
}

} // namespace pto
#endif // PTO_NPU_A1_TVECTOR_HPP
