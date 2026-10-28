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

#include <pto/npu/a1/TImg2col.hpp>
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
    PTO_STATIC_ASSERT(
        (TileData::Rows >= 16 && TileData::Rows <= 2048 && TileData::Rows % 16 == 0 && TileData::Cols >= 16 &&
         TileData::Cols <= 2048 && TileData::Cols % 16 == 0 && TileData::ValidRow == TileData::Rows &&
         TileData::ValidCol == TileData::Cols && uint64_t(TileData::Rows) * TileData::Cols * sizeof(half) <= 1048576),
        "A1 matrix inputs require full aligned M/K dimensions and at most 1 MiB L1.");
    PTO_STATIC_ASSERT(
        (TileData::Loc == TileType::Mat && std::is_same<typename TileData::DType, half>::value &&
         TileData::SFractal == SLayout::RowMajor && TileData::SFractalSize == 512),
        "A1 matrix input requires an FP16 ZZ or NZ Mat tile.");
}
} // namespace detail

// Each panel is contiguous in L1. Wide ND rows are split into C16 panels;
// ZZ stores [M1,K1,M0,K0], while NZ stores [K1,M1,M0,K0].
template <unsigned Rows, uint64_t GlobalStride>
PTO_INTERNAL void A1LoadMatrixPanel(__cbuf__ half* dst, __gm__ half* src)
{
    constexpr uint64_t gap = (GlobalStride - 16) / 16;
    if constexpr (gap == 0) {
        copy_gm_to_cbuf(dst, src, 0, 1, Rows, 0, 0, (pad_t)0);
    } else if constexpr (gap <= UINT16_MAX) {
        copy_gm_to_cbuf(dst, src, 0, Rows, 1, gap, 0, (pad_t)0);
    } else {
        // Preserve full GM strides rather than truncating the DMA field.
        for (unsigned row = 0; row < Rows; ++row)
            copy_gm_to_cbuf(dst + row * 16, src + row * GlobalStride, 0, 1, 1, 0, 0, (pad_t)0);
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLOAD_IMPL(TileData& dst, GlobalData& src)
{
    if constexpr (is_conv_tile_v<TileData>) {
        A1LoadConvImage(dst, src);
    } else if constexpr (TileData::Loc == TileType::Vec) {
        TLOAD_VEC_IMPL(dst, src);
    } else {
        detail::CheckA1MatrixInput<TileData>();
        PTO_STATIC_ASSERT(
            ((std::is_same<typename GlobalData::RawDType, half>::value && GlobalData::layout == Layout::ND &&
              GlobalData::staticShape[0] == 1 && GlobalData::staticShape[1] == 1 && GlobalData::staticShape[2] == 1 &&
              GlobalData::staticShape[3] == TileData::Rows && GlobalData::staticShape[4] == TileData::Cols &&
              GlobalData::staticStride[3] >= TileData::Cols && GlobalData::staticStride[3] % 16 == 0 &&
              GlobalData::staticStride[4] == 1)),
            "A1 matrix load requires a matching static FP16 ND tensor with aligned rows.");
        constexpr uint64_t stride = GlobalData::staticStride[3];
        for (unsigned column = 0; column < TileData::Cols; column += 16) {
            if constexpr (TileData::isRowMajor && TileData::Cols > 16) {
                for (unsigned row = 0; row < TileData::Rows; row += 16)
                    A1LoadMatrixPanel<16, stride>(
                        dst.data() + row * TileData::Cols + column * 16, src.data() + row * stride + column);
            } else {
                A1LoadMatrixPanel<TileData::Rows, stride>(dst.data() + column * TileData::Rows, src.data() + column);
            }
        }
    }
}

// UB is ND, whereas Cube Mat operands are boxed into C16 panels. Repack
// through MTE3 without scalar reads or a GM round trip. The caller fences
// V/MTE2 -> MTE3, MTE3 -> MTE1, and MTE1 -> MTE3 before reusing L1.
template <typename DstTile, typename SrcTile>
PTO_INTERNAL void A1VectorToMatrix(DstTile& dst, SrcTile& src)
{
    detail::CheckA1VectorTile<SrcTile>();
    detail::CheckA1MatrixInput<DstTile>();
    PTO_STATIC_ASSERT(
        (std::is_same<typename SrcTile::DType, half>::value), "A1 Vec-to-Mat moves require FP16 source data.");
    PTO_STATIC_ASSERT(
        (DstTile::Rows == SrcTile::Rows && DstTile::Cols == SrcTile::Cols),
        "A1 Vec-to-Mat moves require matching full shapes.");
    if constexpr (DstTile::Cols == 16) {
        // One C16 panel is already contiguous in both representations.
        copy_ubuf_to_cbuf(dst.data(), src.data(), 0, 1, DstTile::Rows, 0, 0);
    } else if constexpr (DstTile::isRowMajor) {
        // ZZ: [M1,K1,M0,K0]. Each DMA boxes sixteen ND rows.
        for (unsigned row = 0; row < DstTile::Rows; row += 16)
            for (unsigned column = 0; column < DstTile::Cols; column += 16)
                copy_ubuf_to_cbuf(
                    dst.data() + row * DstTile::Cols + column * 16, src.data() + row * SrcTile::Cols + column, 0, 16, 1,
                    SrcTile::Cols / 16 - 1, 0);
    } else {
        // NZ: [K1,M1,M0,K0]. Each DMA collects one complete C16 panel.
        for (unsigned column = 0; column < DstTile::Cols; column += 16)
            copy_ubuf_to_cbuf(
                dst.data() + column * DstTile::Rows, src.data() + column, 0, DstTile::Rows, 1, SrcTile::Cols / 16 - 1,
                0);
    }
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void A1MatrixToOperand(DstTile& dst, SrcTile& src)
{
    detail::CheckA1MatrixInput<SrcTile>();
    PTO_STATIC_ASSERT(
        (DstTile::Rows == SrcTile::Rows && DstTile::Cols == SrcTile::Cols && DstTile::ValidRow == DstTile::Rows &&
         DstTile::ValidCol == DstTile::Cols),
        "A1 matrix operand moves require matching full shapes.");
    PTO_STATIC_ASSERT(
        (std::is_same<typename DstTile::DType, half>::value && DstTile::SFractalSize == 512 &&
         uint64_t(DstTile::Rows) * DstTile::Cols * sizeof(half) <= 65536),
        "A1 matrix operand moves require FP16 fractals within 64 KiB L0A/L0B.");
    if constexpr (DstTile::Loc == TileType::Left) {
        PTO_STATIC_ASSERT(
            ((DstTile::isRowMajor || DstTile::Cols == 16) && DstTile::SFractal == SLayout::RowMajor),
            "A1 multi-K left operand requires ZZ; the legacy NZ layout is valid for K16 only.");
        if constexpr (SrcTile::isRowMajor || SrcTile::Cols == 16) {
            load_cbuf_to_ca(
                dst.data(), src.data(), 0, SrcTile::Rows * SrcTile::Cols / 256, 1, 0, 0, false, addr_cal_mode_t(0));
        } else {
            for (unsigned row = 0; row < SrcTile::Rows / 16; ++row)
                load_cbuf_to_ca(
                    dst.data() + row * SrcTile::Cols * 16, src.data() + row * 256, 0, SrcTile::Cols / 16,
                    SrcTile::Rows / 16, 0, 0, false, addr_cal_mode_t(0));
        }
    } else {
        PTO_STATIC_ASSERT(
            (DstTile::Loc == TileType::Right && DstTile::Cols == 16 && DstTile::isRowMajor &&
             DstTile::SFractal == SLayout::ColMajor),
            "A1 right operand requires a Kx16 ZN Right tile.");
        load_cbuf_to_cb(dst.data(), src.data(), 0, SrcTile::Rows / 16, 1, 0, 0, true, addr_cal_mode_t(0));
    }
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TMOV_IMPL(DstTile& dst, SrcTile& src)
{
    if constexpr (SrcTile::Loc == TileType::Vec && DstTile::Loc == TileType::Mat) {
        A1VectorToMatrix(dst, src);
    } else {
        A1MatrixToOperand(dst, src);
    }
}

// One C16 L1 panel can fill any aligned rectangle of a ZZ L0A tile.
// MTE1's destination fractal stride preserves all other rows/columns and
// allows repeated insertion of the same panel without additional GM reads.
template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TINSERT_IMPL(DstTile& dst, SrcTile& src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    detail::CheckA1MatrixInput<SrcTile>();
    PTO_STATIC_ASSERT(SrcTile::Cols == 16, "A1 Left insertion requires full C16 Mat panels.");
    PTO_STATIC_ASSERT(
        (DstTile::Loc == TileType::Left && std::is_same<typename DstTile::DType, half>::value && DstTile::Rows >= 16 &&
         DstTile::Rows <= 2048 && DstTile::Rows % 16 == 0 && DstTile::Cols >= 16 && DstTile::Cols <= 2048 &&
         DstTile::Cols % 16 == 0 && DstTile::Rows == DstTile::ValidRow && DstTile::Cols == DstTile::ValidCol &&
         SrcTile::Rows <= DstTile::Rows),
        "A1 Left insertion requires a matching full aligned FP16 Left destination.");
    PTO_STATIC_ASSERT(
        ((DstTile::isRowMajor || DstTile::Cols == 16) && DstTile::SFractal == SLayout::RowMajor &&
         DstTile::SFractalSize == 512),
        "A1 Left insertion requires ZZ layout; legacy NZ is valid for K16 only.");
    PTO_STATIC_ASSERT(
        uint64_t(DstTile::Rows) * DstTile::Cols * sizeof(half) <= 65536,
        "A1 Left insertion destination exceeds the 64 KiB L0A budget.");
    // Native callers can pass runtime offsets. Keep release builds safe even
    // when they bypass the compiler's aligned-offset and bounds proof.
    if (indexRow % 16 || indexCol % 16 || indexRow + SrcTile::Rows > DstTile::Rows || indexCol + 16 > DstTile::Cols) {
        trap();
        return;
    }
    load_cbuf_to_ca(
        dst.data() + uint32_t(indexRow) * DstTile::Cols + uint32_t(indexCol) * 16, src.data(), 0, SrcTile::Rows / 16, 1,
        DstTile::Cols / 16 - 1, 0, false, addr_cal_mode_t(0));
}

template <bool Initialize, typename AccTile, typename LeftTile, typename RightTile>
PTO_INTERNAL void A1MatrixMultiply(AccTile& dst, LeftTile& lhs, RightTile& rhs)
{
    detail::CheckA1NarrowMatrixTile<AccTile>();
    PTO_STATIC_ASSERT(
        (LeftTile::Rows >= 16 && LeftTile::Rows <= 2048 && LeftTile::Rows % 16 == 0 && LeftTile::Cols >= 16 &&
         LeftTile::Cols <= 2048 && LeftTile::Cols % 16 == 0 && LeftTile::ValidRow == LeftTile::Rows &&
         LeftTile::ValidCol == LeftTile::Cols && uint64_t(LeftTile::Rows) * LeftTile::Cols * sizeof(half) <= 65536 &&
         RightTile::Rows == LeftTile::Cols && RightTile::Cols == 16 && RightTile::ValidRow == RightTile::Rows &&
         RightTile::ValidCol == 16 && uint64_t(RightTile::Rows) * 16 * sizeof(half) <= 65536),
        "A1 GEMM requires aligned full M/K, N16 and operands within 64 KiB L0A/L0B.");
    PTO_STATIC_ASSERT(AccTile::Rows == LeftTile::Rows, "A1 GEMM requires matching M rows.");
    PTO_STATIC_ASSERT(
        (AccTile::Loc == TileType::Acc && std::is_same<typename AccTile::DType, float>::value && !AccTile::isRowMajor &&
         AccTile::SFractal == SLayout::RowMajor && AccTile::SFractalSize == 1024),
        "A1 matrix multiplication requires an FP32 NZ accumulator.");
    PTO_STATIC_ASSERT(
        (LeftTile::Loc == TileType::Left && RightTile::Loc == TileType::Right &&
         std::is_same<typename LeftTile::DType, half>::value && std::is_same<typename RightTile::DType, half>::value &&
         (LeftTile::isRowMajor || LeftTile::Cols == 16) && LeftTile::SFractal == SLayout::RowMajor &&
         LeftTile::SFractalSize == 512 && RightTile::isRowMajor && RightTile::SFractal == SLayout::ColMajor &&
         RightTile::SFractalSize == 512),
        "A1 GEMM requires FP16 ZZ Left (legacy NZ at K16) and ZN Right operands.");
    mad(dst.data(), lhs.data(), rhs.data(), AccTile::Rows, LeftTile::Cols, 16, Initialize);
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
