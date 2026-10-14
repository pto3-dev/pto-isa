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
PTO_INTERNAL void CheckA1NarrowMatrixTile()
{
    PTO_STATIC_ASSERT(
        (TileData::Rows >= 16 && TileData::Rows <= 2048 && TileData::Rows % 16 == 0 && TileData::Cols == 16 &&
         TileData::ValidRow == TileData::Rows && TileData::ValidCol == 16),
        "A1 narrow matrix instructions require full aligned rows in [16, 2048] and 16 columns.");
}

template <typename TileData>
PTO_INTERNAL void CheckA1MatrixInput()
{
    CheckA1NarrowMatrixTile<TileData>();
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
              GlobalData::staticShape[3] == TileData::Rows && GlobalData::staticShape[4] == 16 &&
              GlobalData::staticStride[3] == 16 && GlobalData::staticStride[4] == 1)),
            "A1 matrix load requires a matching contiguous static FP16 Nx16 ND tensor.");
        // One outer column of ZZ/NZ fractals has the same element order as ND.
        copy_gm_to_cbuf(dst.data(), src.data(), 0, 1, TileData::Rows, 0, 0, (pad_t)0);
    }
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TMOV_IMPL(DstTile& dst, SrcTile& src)
{
    detail::CheckA1MatrixInput<SrcTile>();
    detail::CheckA1NarrowMatrixTile<DstTile>();
    PTO_STATIC_ASSERT(DstTile::Rows == SrcTile::Rows, "A1 matrix operand moves require matching shapes.");
    PTO_STATIC_ASSERT(
        (std::is_same<typename DstTile::DType, half>::value && DstTile::SFractalSize == 512),
        "A1 matrix operand moves require FP16 512-byte fractals.");
    if constexpr (DstTile::Loc == TileType::Left) {
        PTO_STATIC_ASSERT(
            (!DstTile::isRowMajor && DstTile::SFractal == SLayout::RowMajor),
            "A1 left operand requires an NZ Left tile.");
        load_cbuf_to_ca(dst.data(), src.data(), 0, SrcTile::Rows / 16, 1, 0, 0, false, addr_cal_mode_t(0));
    } else {
        PTO_STATIC_ASSERT(
            (DstTile::Loc == TileType::Right && DstTile::Rows == 16 && DstTile::isRowMajor &&
             DstTile::SFractal == SLayout::ColMajor),
            "A1 right operand requires a ZN Right tile.");
        load_cbuf_to_cb(dst.data(), src.data(), 0, 1, 1, 0, 0, true, addr_cal_mode_t(0));
    }
}

template <bool Initialize, typename AccTile, typename LeftTile, typename RightTile>
PTO_INTERNAL void A1MatrixMultiply(AccTile& dst, LeftTile& lhs, RightTile& rhs)
{
    detail::CheckA1NarrowMatrixTile<AccTile>();
    detail::CheckA1NarrowMatrixTile<LeftTile>();
    detail::CheckA1NarrowMatrixTile<RightTile>();
    PTO_STATIC_ASSERT(
        AccTile::Rows == LeftTile::Rows && RightTile::Rows == 16,
        "A1 narrow GEMM requires matching M rows with K=N=16.");
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
    mad(dst.data(), lhs.data(), rhs.data(), AccTile::Rows, 16, 16, Initialize);
}

template <typename AccTile, typename LeftTile, typename RightTile>
PTO_INTERNAL void TMATMUL_IMPL(AccTile& dst, LeftTile& lhs, RightTile& rhs)
{
    A1MatrixMultiply<true>(dst, lhs, rhs);
}

template <typename AccTile, typename LeftTile, typename RightTile>
PTO_INTERNAL void TMATMUL_ACC_IMPL(AccTile& dst, AccTile& acc, LeftTile& lhs, RightTile& rhs)
{
    // DAV-M200 MAD accumulates in its destination; it cannot read a separate L0C buffer.
    // Keep this contract active in release builds, where PTO_ASSERT is disabled.
    if (dst.data() != acc.data()) {
        trap();
        return;
    }
    // Consecutive MADs need the previous L0C write to complete before reading it.
    pipe_barrier(PIPE_M);
    A1MatrixMultiply<false>(dst, lhs, rhs);
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void A1AccumulatorToVector(DstTile& dst, SrcTile& src)
{
    detail::CheckA1NarrowMatrixTile<SrcTile>();
    detail::CheckA1NarrowMatrixTile<DstTile>();
    detail::CheckA1VectorTile<DstTile>();
    PTO_STATIC_ASSERT(SrcTile::Rows == DstTile::Rows, "A1 matrix writeback requires matching shapes.");
    PTO_STATIC_ASSERT(
        (SrcTile::Loc == TileType::Acc && std::is_same<typename SrcTile::DType, float>::value && !SrcTile::isRowMajor &&
         SrcTile::SFractal == SLayout::RowMajor && SrcTile::SFractalSize == 1024 &&
         std::is_same<typename DstTile::DType, float>::value),
        "A1 accumulator writeback requires FP32 NZ Acc and FP32 ND Vec tiles.");
    // The single outer column of NZ fractals matches the Nx16 ND destination.
    copy_matrix_cc_to_ubuf(dst.data(), src.data(), 0, 1, SrcTile::Rows / 16, 0, 0, CRMODE_NONE);
}
} // namespace pto
#endif // PTO_NPU_A1_TMATRIX_HPP
