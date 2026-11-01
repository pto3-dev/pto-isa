/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMMON_STRIDED_INSERT_HPP
#define PTO_COMMON_STRIDED_INSERT_HPP

#include <pto/common/pto_tile.hpp>

namespace pto {
namespace detail {
// Logical rows are the free dimension of an NZ Mat plane. A row step
// scatters C16 payloads without changing their within-row element order.
template <unsigned DstRowStep, typename DstTile, typename SrcTile>
PTO_INTERNAL void CheckA1RowStridedInsert()
{
    PTO_STATIC_ASSERT(DstRowStep > 0 && DstRowStep <= 65536, "A1 strided insertion requires a row step in [1, 65536].");
    PTO_STATIC_ASSERT(
        (SrcTile::Loc == TileType::Vec && DstTile::Loc == TileType::Mat &&
         std::is_same_v<typename SrcTile::DType, half> && std::is_same_v<typename DstTile::DType, half>),
        "A1 strided insertion requires FP16 Vec source and Mat destination.");
    PTO_STATIC_ASSERT(
        (SrcTile::isRowMajor && SrcTile::SFractal == SLayout::NoneBox && SrcTile::Rows > 0 && SrcTile::Cols >= 16 &&
         SrcTile::Cols % 16 == 0 && SrcTile::ValidRow == SrcTile::Rows && SrcTile::ValidCol == SrcTile::Cols &&
         uint64_t(SrcTile::Rows) * SrcTile::Cols * sizeof(half) <= 131072),
        "A1 strided insertion requires a full C16-aligned ND source within 128 KiB UB.");
    PTO_STATIC_ASSERT(
        (!DstTile::isRowMajor && DstTile::SFractal == SLayout::RowMajor && DstTile::SFractalSize == 512 &&
         DstTile::Rows >= 16 && DstTile::Rows % 16 == 0 && DstTile::Cols >= 16 && DstTile::Cols <= 2048 &&
         DstTile::Cols % 16 == 0 && DstTile::ValidRow == DstTile::Rows && DstTile::ValidCol == DstTile::Cols &&
         uint64_t(DstTile::Rows) * DstTile::Cols * sizeof(half) <= 1048576),
        "A1 strided insertion requires a full FP16 NZ destination within 1 MiB L1.");
    PTO_STATIC_ASSERT(
        (uint64_t(SrcTile::Rows - 1) * DstRowStep + 1 <= DstTile::Rows && SrcTile::Cols <= DstTile::Cols),
        "A1 strided insertion source span exceeds its destination.");
}

template <unsigned DstRowStep, typename DstTile, typename SrcTile>
PTO_INTERNAL bool ValidA1RowStridedInsertWindow(uint16_t indexRow, uint16_t indexCol)
{
    constexpr uint64_t rowSpan = uint64_t(SrcTile::Rows - 1) * DstRowStep + 1;
    return indexCol % 16 == 0 && uint64_t(indexRow) + rowSpan <= DstTile::Rows &&
           uint64_t(indexCol) + SrcTile::Cols <= DstTile::Cols;
}
} // namespace detail
} // namespace pto
#endif // PTO_COMMON_STRIDED_INSERT_HPP
