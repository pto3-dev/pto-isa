/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMMON_STRIDED_EXTRACT_HPP
#define PTO_COMMON_STRIDED_EXTRACT_HPP

#include <pto/common/pto_tile.hpp>

namespace pto {
namespace detail {
// Logical rows advance within one C16 block, then jump complete source blocks.
// Clamp malformed destination dimensions before computing a diagnostic span.
template <unsigned SrcBlockStep, typename DstTile>
PTO_INTERNAL constexpr uint64_t A1BlockExtractSpanRows()
{
    constexpr uint64_t blocks = DstTile::Rows >= 16 && DstTile::Rows <= 2048 ? DstTile::Rows / 16 : 0;
    return blocks == 0 ? 0 : ((blocks - 1) * uint64_t(SrcBlockStep) + 1) * 16;
}

template <unsigned SrcBlockStep, typename DstTile, typename SrcTile>
PTO_INTERNAL void CheckA1BlockStridedExtract()
{
    PTO_STATIC_ASSERT(SrcBlockStep > 0, "A1 block extraction requires a positive source block step.");
    PTO_STATIC_ASSERT(
        (SrcTile::Loc == TileType::Mat && DstTile::Loc == TileType::Right &&
         std::is_same_v<typename SrcTile::DType, half> && std::is_same_v<typename DstTile::DType, half>),
        "A1 block extraction requires FP16 Mat source and Right destination.");
    PTO_STATIC_ASSERT(
        (SrcTile::SFractal == SLayout::RowMajor && SrcTile::SFractalSize == 512 && SrcTile::Rows >= 16 &&
         SrcTile::Rows % 16 == 0 && SrcTile::Cols >= 16 && SrcTile::Cols <= 2048 && SrcTile::Cols % 16 == 0 &&
         SrcTile::ValidRow == SrcTile::Rows && SrcTile::ValidCol == SrcTile::Cols &&
         uint64_t(SrcTile::Rows) * SrcTile::Cols * sizeof(half) <= 1048576),
        "A1 block extraction requires a full FP16 ZZ/NZ source within the configured 1 MiB L1 budget.");
    PTO_STATIC_ASSERT(
        (DstTile::isRowMajor && DstTile::SFractal == SLayout::ColMajor && DstTile::SFractalSize == 512 &&
         DstTile::Rows >= 16 && DstTile::Rows <= 2048 && DstTile::Rows % 16 == 0 &&
         (DstTile::Cols == 16 || DstTile::Cols == 32) && DstTile::ValidRow == DstTile::Rows &&
         DstTile::ValidCol == DstTile::Cols && uint64_t(DstTile::Rows) * DstTile::Cols * sizeof(half) <= 65536),
        "A1 block extraction requires a full N16/N32 ZN destination within the configured 64 KiB L0B budget.");
    // Ordinary CPU headers do not select an NPU architecture or define these
    // build-time budgets. Honor explicit/native budgets without changing that
    // portable include contract; the physical A1 limits above always apply.
#ifdef PTO_CBUF_SIZE_BYTES
    PTO_STATIC_ASSERT(
        (uint64_t(SrcTile::Rows) * SrcTile::Cols * sizeof(half) <= PTO_CBUF_SIZE_BYTES),
        "A1 block extraction exceeds the configured L1 budget.");
#endif
#ifdef PTO_L0B_SIZE_BYTES
    PTO_STATIC_ASSERT(
        (uint64_t(DstTile::Rows) * DstTile::Cols * sizeof(half) <= PTO_L0B_SIZE_BYTES),
        "A1 block extraction exceeds the configured L0B budget.");
#endif
    PTO_STATIC_ASSERT(
        (A1BlockExtractSpanRows<SrcBlockStep, DstTile>() <= SrcTile::Rows && DstTile::Cols <= SrcTile::Cols),
        "A1 block extraction selected source span exceeds its source.");
}

template <unsigned SrcBlockStep, typename DstTile, typename SrcTile>
PTO_INTERNAL bool ValidA1BlockExtractWindow(uint16_t indexRow, uint16_t indexCol)
{
    return indexRow % 16 == 0 && indexCol % 16 == 0 &&
           uint64_t(indexRow) + A1BlockExtractSpanRows<SrcBlockStep, DstTile>() <= SrcTile::Rows &&
           uint64_t(indexCol) + DstTile::Cols <= SrcTile::Cols;
}
} // namespace detail
} // namespace pto
#endif // PTO_COMMON_STRIDED_EXTRACT_HPP
