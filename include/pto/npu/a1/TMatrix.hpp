/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_A1_TMATRIX_HPP
#define PTO_NPU_A1_TMATRIX_HPP

#include <pto/npu/a1/TVector.hpp>

namespace pto {
namespace detail {
template <typename TileData>
PTO_INTERNAL void CheckA1SingleFractal()
{
    PTO_STATIC_ASSERT(
        (TileData::Rows == 16 && TileData::Cols == 16 && TileData::ValidRow == 16 && TileData::ValidCol == 16),
        "A1 matrix instructions require a static full 16x16 tile.");
}

template <typename TileData>
PTO_INTERNAL void CheckA1MatrixInput()
{
    CheckA1SingleFractal<TileData>();
    PTO_STATIC_ASSERT(
        (TileData::Loc == TileType::Mat && std::is_same<typename TileData::DType, half>::value &&
         TileData::SFractal == SLayout::RowMajor && TileData::SFractalSize == 512),
        "A1 matrix input requires an FP16 ZZ or NZ Mat tile.");
}
} // namespace detail

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLOAD_IMPL(TileData& dst, GlobalData& src)
{
    if constexpr (TileData::Loc == TileType::Vec) {
        TLOAD_VEC_IMPL(dst, src);
    } else {
        detail::CheckA1MatrixInput<TileData>();
        PTO_STATIC_ASSERT(
            ((std::is_same<typename GlobalData::RawDType, half>::value && GlobalData::layout == Layout::ND &&
              GlobalData::staticShape[0] == 1 && GlobalData::staticShape[1] == 1 && GlobalData::staticShape[2] == 1 &&
              GlobalData::staticShape[3] == 16 && GlobalData::staticShape[4] == 16 &&
              GlobalData::staticStride[3] == 16 && GlobalData::staticStride[4] == 1)),
            "A1 matrix load requires a contiguous static FP16 16x16 ND tensor.");
        // A single 16x16 ZZ or NZ fractal has the same element order as ND.
        copy_gm_to_cbuf(dst.data(), src.data(), 0, 1, 16, 0, 0, (pad_t)0);
    }
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TMOV_IMPL(DstTile& dst, SrcTile& src)
{
    detail::CheckA1MatrixInput<SrcTile>();
    detail::CheckA1SingleFractal<DstTile>();
    PTO_STATIC_ASSERT(
        (std::is_same<typename DstTile::DType, half>::value && DstTile::SFractalSize == 512),
        "A1 matrix operand moves require FP16 512-byte fractals.");
    if constexpr (DstTile::Loc == TileType::Left) {
        PTO_STATIC_ASSERT(
            (!DstTile::isRowMajor && DstTile::SFractal == SLayout::RowMajor),
            "A1 left operand requires an NZ Left tile.");
        load_cbuf_to_ca(dst.data(), src.data(), 0, 1, 1, 0, 0, false, addr_cal_mode_t(0));
    } else {
        PTO_STATIC_ASSERT(
            (DstTile::Loc == TileType::Right && DstTile::isRowMajor && DstTile::SFractal == SLayout::ColMajor),
            "A1 right operand requires a ZN Right tile.");
        load_cbuf_to_cb(dst.data(), src.data(), 0, 1, 1, 0, 0, true, addr_cal_mode_t(0));
    }
}

template <typename AccTile, typename LeftTile, typename RightTile>
PTO_INTERNAL void TMATMUL_IMPL(AccTile& dst, LeftTile& lhs, RightTile& rhs)
{
    detail::CheckA1SingleFractal<AccTile>();
    detail::CheckA1SingleFractal<LeftTile>();
    detail::CheckA1SingleFractal<RightTile>();
    PTO_STATIC_ASSERT(
        (AccTile::Loc == TileType::Acc && std::is_same<typename AccTile::DType, float>::value && !AccTile::isRowMajor &&
         AccTile::SFractal == SLayout::RowMajor && AccTile::SFractalSize == 1024),
        "A1 matrix multiplication requires an FP32 NZ accumulator.");
    PTO_STATIC_ASSERT(
        (LeftTile::Loc == TileType::Left && RightTile::Loc == TileType::Right &&
         std::is_same<typename LeftTile::DType, half>::value && std::is_same<typename RightTile::DType, half>::value &&
         !LeftTile::isRowMajor && LeftTile::SFractal == SLayout::RowMajor && LeftTile::SFractalSize == 512 &&
         RightTile::isRowMajor && RightTile::SFractal == SLayout::ColMajor && RightTile::SFractalSize == 512),
        "A1 matrix multiplication requires FP16 NZ Left and ZN Right operands.");
    mad(dst.data(), lhs.data(), rhs.data(), 16, 16, 16, true);
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void A1AccumulatorToVector(DstTile& dst, SrcTile& src)
{
    detail::CheckA1SingleFractal<SrcTile>();
    detail::CheckA1SingleFractal<DstTile>();
    detail::CheckA1VectorTile<DstTile>();
    PTO_STATIC_ASSERT(
        (SrcTile::Loc == TileType::Acc && std::is_same<typename SrcTile::DType, float>::value && !SrcTile::isRowMajor &&
         SrcTile::SFractal == SLayout::RowMajor && SrcTile::SFractalSize == 1024 &&
         std::is_same<typename DstTile::DType, float>::value),
        "A1 accumulator writeback requires FP32 NZ Acc and FP32 ND Vec tiles.");
    // Only a single NZ fractal has the same order as the ND destination.
    copy_matrix_cc_to_ubuf(dst.data(), src.data(), 0, 1, 1, 0, 0, CRMODE_NONE);
}
} // namespace pto
#endif // PTO_NPU_A1_TMATRIX_HPP
