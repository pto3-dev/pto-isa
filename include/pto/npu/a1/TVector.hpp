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
#include <pto/common/event.hpp>
#include <pto/common/pto_tile.hpp>

#ifdef __PTO_AUTO__
#error "A1 vector instructions currently require manual allocation and synchronization."
#endif

namespace pto {
namespace detail {

template <typename First, typename Second>
PTO_INTERNAL bool A1BuffersOverlap(First& first, Second& second)
{
    const uintptr_t a = reinterpret_cast<uintptr_t>(first.data());
    const uintptr_t b = reinterpret_cast<uintptr_t>(second.data());
    return a < b + Second::Numel * sizeof(typename Second::DType) &&
           b < a + First::Numel * sizeof(typename First::DType);
}

template <typename TileData>
PTO_INTERNAL void CheckA1VectorTile()
{
    PTO_STATIC_ASSERT(TileData::Loc == TileType::Vec, "A1 vector instructions require a Vec tile.");
    PTO_STATIC_ASSERT(
        (std::is_same<typename TileData::DType, half>::value || std::is_same<typename TileData::DType, float>::value),
        "A1 vector instructions require FP16 or FP32.");
    PTO_STATIC_ASSERT(
        TileData::isRowMajor && TileData::SFractal == SLayout::NoneBox,
        "A1 vector instructions require contiguous row-major tiles.");
    PTO_STATIC_ASSERT(
        TileData::ValidRow == TileData::Rows && TileData::ValidCol == TileData::Cols,
        "A1 vector instructions require a static full valid shape.");
    PTO_STATIC_ASSERT(
        TileData::RowStride == TileData::Cols && TileData::Cols % (32 / sizeof(typename TileData::DType)) == 0,
        "A1 vector rows must be aligned to 32 bytes without padding.");
    PTO_STATIC_ASSERT(
        TileData::Numel * sizeof(typename TileData::DType) <= PTO_UBUF_SIZE_BYTES,
        "A1 vector tile exceeds the configured UB budget.");
}

template <typename TileData>
PTO_INTERNAL void CheckA1PackedColumn()
{
    PTO_STATIC_ASSERT(TileData::Loc == TileType::Vec, "A1 packed columns require Vec storage.");
    PTO_STATIC_ASSERT(
        (std::is_same<typename TileData::DType, half>::value || std::is_same<typename TileData::DType, float>::value),
        "A1 packed columns require FP16 or FP32.");
    PTO_STATIC_ASSERT(
        !TileData::isRowMajor && TileData::Rows > 0 && TileData::Cols == 1 && TileData::SFractal == SLayout::NoneBox &&
            TileData::ValidRow == TileData::Rows && TileData::ValidCol == 1 &&
            TileData::Rows % (32 / sizeof(typename TileData::DType)) == 0,
        "A1 packed columns require full static aligned Mx1 column-major tiles.");
    PTO_STATIC_ASSERT(
        TileData::Numel * sizeof(typename TileData::DType) <= PTO_UBUF_SIZE_BYTES,
        "A1 packed column exceeds the configured UB budget.");
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void CheckA1PackedColumnTransfer()
{
    CheckA1PackedColumn<TileData>();
    PTO_STATIC_ASSERT(
        (std::is_same<typename GlobalData::RawDType, typename TileData::DType>::value),
        "A1 packed DMA requires matching data types.");
    PTO_STATIC_ASSERT(
        (GlobalData::layout == Layout::ND || GlobalData::layout == Layout::DN) && GlobalData::staticShape[0] == 1 &&
            GlobalData::staticShape[1] == 1 && GlobalData::staticShape[2] == 1 &&
            GlobalData::staticShape[3] == TileData::Rows && GlobalData::staticShape[4] == 1 &&
            GlobalData::staticStride[3] == 1 && GlobalData::staticStride[4] > 0,
        "A1 packed DMA requires a matching contiguous ND Mx1 tensor.");
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void CheckA1VectorTransfer()
{
    CheckA1VectorTile<TileData>();
    PTO_STATIC_ASSERT(
        (std::is_same<typename GlobalData::RawDType, typename TileData::DType>::value),
        "A1 DMA requires tensor and tile data types to match.");
    PTO_STATIC_ASSERT(GlobalData::layout == Layout::ND, "A1 vector DMA requires ND tensors.");
    PTO_STATIC_ASSERT(
        GlobalData::staticShape[0] == 1 && GlobalData::staticShape[1] == 1 && GlobalData::staticShape[2] == 1 &&
            GlobalData::staticShape[3] == TileData::Rows && GlobalData::staticShape[4] == TileData::Cols,
        "A1 vector DMA requires matching static two-dimensional tensor and tile shapes.");
    PTO_STATIC_ASSERT(
        GlobalData::staticStride[3] >= TileData::Cols && GlobalData::staticStride[4] == 1 &&
            GlobalData::staticStride[3] % (32 / sizeof(typename TileData::DType)) == 0,
        "A1 vector DMA requires static non-overlapping 32-byte-aligned tensor rows.");
}

template <Op Operation, typename Element>
PTO_INTERNAL void A1VectorRepeat(
    __ubuf__ Element* dst, __ubuf__ Element* src0, __ubuf__ Element* src1, uint8_t repeats, Element scalar = 0)
{
    if constexpr (Operation == Op::TADD) {
        vadd(dst, src0, src1, repeats, 1, 1, 1, 8, 8, 8);
    } else if constexpr (Operation == Op::TSUB) {
        vsub(dst, src0, src1, repeats, 1, 1, 1, 8, 8, 8);
    } else if constexpr (Operation == Op::TDIV) {
        vdiv(dst, src0, src1, repeats, 1, 1, 1, 8, 8, 8);
    } else if constexpr (Operation == Op::TMAX) {
        vmax(dst, src0, src1, repeats, 1, 1, 1, 8, 8, 8);
    } else if constexpr (Operation == Op::TEXPANDS) {
        vector_dup(dst, scalar, repeats, 1, 1, 8, 8);
    } else if constexpr (Operation == Op::TEXP) {
        vexp(dst, src0, repeats, 1, 1, 8, 8);
    } else {
        static_assert(Operation == Op::TMUL);
        vmul(dst, src0, src1, repeats, 1, 1, 1, 8, 8, 8);
    }
}

template <Op Operation, unsigned Elements, typename Element>
PTO_INTERNAL void A1VectorInstruction(
    __ubuf__ Element* dst, __ubuf__ Element* src0, __ubuf__ Element* src1, Element scalar = 0)
{
    constexpr unsigned elementsPerRepeat = 256 / sizeof(Element);
    constexpr unsigned fullElements = Elements / elementsPerRepeat * elementsPerRepeat;
    constexpr unsigned tailElements = Elements % elementsPerRepeat;
    set_mask_norm();
    if constexpr (elementsPerRepeat == 128)
        set_vector_mask(UINT64_MAX, UINT64_MAX);
    else
        set_vector_mask(0, UINT64_MAX);
    for (unsigned offset = 0; offset < fullElements; offset += 255 * elementsPerRepeat) {
        const unsigned remaining = (fullElements - offset) / elementsPerRepeat;
        const uint8_t repeats = remaining > 255 ? 255 : remaining;
        A1VectorRepeat<Operation>(dst + offset, src0 + offset, src1 + offset, repeats, scalar);
    }
    if constexpr (tailElements != 0) {
        constexpr uint64_t lowMask = tailElements >= 64 ? UINT64_MAX : (UINT64_C(1) << (tailElements % 64)) - 1;
        constexpr uint64_t highMask =
            tailElements <= 64 ? 0 : (UINT64_C(1) << (tailElements > 64 ? tailElements - 64 : 0)) - 1;
        // The inactive lanes may belong to an adjacent tile or alias source.
        set_vector_mask(highMask, lowMask);
        A1VectorRepeat<Operation>(dst + fullElements, src0 + fullElements, src1 + fullElements, 1, scalar);
        if constexpr (elementsPerRepeat == 128)
            set_vector_mask(UINT64_MAX, UINT64_MAX);
        else
            set_vector_mask(0, UINT64_MAX);
    }
}

template <typename DstElement, typename SrcElement>
PTO_INTERNAL void A1FloatingConvertRepeat(__ubuf__ DstElement* dst, __ubuf__ SrcElement* src, uint8_t repeats)
{
    if constexpr (sizeof(DstElement) == 2)
        vconv_f322f16(dst, src, repeats, 1, 1, 4, 8);
    else
        vconv_f162f32(dst, src, repeats, 1, 1, 8, 4);
}

} // namespace detail

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLOAD_VEC_IMPL(TileData& dst, GlobalData& src)
{
    if constexpr (!TileData::isRowMajor) {
        detail::CheckA1PackedColumnTransfer<TileData, GlobalData>();
        copy_gm_to_ubuf(dst.data(), src.data(), 0, 1, TileData::Numel * sizeof(typename TileData::DType) / 32, 0, 0);
    } else {
        detail::CheckA1VectorTransfer<TileData, GlobalData>();
        constexpr unsigned rowBlocks = TileData::Cols * sizeof(typename TileData::DType) / 32;
        constexpr uint64_t globalGap =
            (GlobalData::staticStride[3] - TileData::Cols) / (32 / sizeof(typename TileData::DType));
        if constexpr (globalGap == 0) {
            copy_gm_to_ubuf(dst.data(), src.data(), 0, 1, TileData::Rows * rowBlocks, 0, 0);
        } else if constexpr (globalGap <= UINT16_MAX) {
            // Multi-burst DMA has a 12-bit burst count and a 16-bit GM gap.
            for (unsigned row = 0; row < TileData::Rows; row += 4095) {
                const unsigned remaining = TileData::Rows - row;
                const unsigned bursts = remaining > 4095 ? 4095 : remaining;
                copy_gm_to_ubuf(
                    dst.data() + row * TileData::Cols, src.data() + row * GlobalData::staticStride[3], 0, bursts,
                    rowBlocks, globalGap, 0);
            }
        } else {
            for (unsigned row = 0; row < TileData::Rows; ++row)
                copy_gm_to_ubuf(
                    dst.data() + row * TileData::Cols, src.data() + row * GlobalData::staticStride[3], 0, 1, rowBlocks,
                    0, 0);
        }
    }
}

template <
    typename TileData, typename GlobalData, AtomicType Atomic = AtomicType::AtomicNone,
    STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TSTORE_IMPL(GlobalData& dst, TileData& src)
{
    if constexpr (!TileData::isRowMajor) {
        detail::CheckA1PackedColumnTransfer<TileData, GlobalData>();
        PTO_STATIC_ASSERT(
            Atomic == AtomicType::AtomicNone && Phase == STPhase::Unspecified,
            "A1 packed stores require ordinary store mode.");
        set_atomic_none();
        copy_ubuf_to_gm(dst.data(), src.data(), 0, 1, TileData::Numel * sizeof(typename TileData::DType) / 32, 0, 0);
    } else {
        detail::CheckA1VectorTransfer<TileData, GlobalData>();
        PTO_STATIC_ASSERT(
            Atomic == AtomicType::AtomicNone && Phase == STPhase::Unspecified,
            "A1 vector stores do not support atomic or unit-flag modes.");
        set_atomic_none();
        constexpr unsigned rowBlocks = TileData::Cols * sizeof(typename TileData::DType) / 32;
        constexpr uint64_t globalGap =
            (GlobalData::staticStride[3] - TileData::Cols) / (32 / sizeof(typename TileData::DType));
        if constexpr (globalGap == 0) {
            copy_ubuf_to_gm(dst.data(), src.data(), 0, 1, TileData::Rows * rowBlocks, 0, 0);
        } else if constexpr (globalGap <= UINT16_MAX) {
            for (unsigned row = 0; row < TileData::Rows; row += 4095) {
                const unsigned remaining = TileData::Rows - row;
                const unsigned bursts = remaining > 4095 ? 4095 : remaining;
                copy_ubuf_to_gm(
                    dst.data() + row * GlobalData::staticStride[3], src.data() + row * TileData::Cols, 0, bursts,
                    rowBlocks, 0, globalGap);
            }
        } else {
            for (unsigned row = 0; row < TileData::Rows; ++row)
                copy_ubuf_to_gm(
                    dst.data() + row * GlobalData::staticStride[3], src.data() + row * TileData::Cols, 0, 1, rowBlocks,
                    0, 0);
        }
    }
}

template <Op Operation, typename DstTile, typename Src0Tile, typename Src1Tile>
PTO_INTERNAL void A1ElementwiseVector(DstTile& dst, Src0Tile& src0, Src1Tile& src1, typename DstTile::DType scalar = 0)
{
    detail::CheckA1VectorTile<DstTile>();
    detail::CheckA1VectorTile<Src0Tile>();
    detail::CheckA1VectorTile<Src1Tile>();
    PTO_STATIC_ASSERT(
        DstTile::Rows == Src0Tile::Rows && DstTile::Rows == Src1Tile::Rows && DstTile::Cols == Src0Tile::Cols &&
            DstTile::Cols == Src1Tile::Cols,
        "A1 binary vector operations require matching tile shapes.");
    PTO_STATIC_ASSERT(
        (std::is_same<typename DstTile::DType, typename Src0Tile::DType>::value &&
         std::is_same<typename DstTile::DType, typename Src1Tile::DType>::value),
        "A1 binary vector operations require matching data types.");
    detail::A1VectorInstruction<Operation, DstTile::Numel>(dst.data(), src0.data(), src1.data(), scalar);
}

template <typename DstTile, typename Src0Tile, typename Src1Tile>
PTO_INTERNAL void TADD_IMPL(DstTile& dst, Src0Tile& src0, Src1Tile& src1)
{
    A1ElementwiseVector<Op::TADD>(dst, src0, src1);
}

template <typename DstTile, typename Src0Tile, typename Src1Tile>
PTO_INTERNAL void TSUB_IMPL(DstTile& dst, Src0Tile& src0, Src1Tile& src1)
{
    A1ElementwiseVector<Op::TSUB>(dst, src0, src1);
}

template <typename DstTile, typename Src0Tile, typename Src1Tile>
PTO_INTERNAL void TMUL_IMPL(DstTile& dst, Src0Tile& src0, Src1Tile& src1)
{
    A1ElementwiseVector<Op::TMUL>(dst, src0, src1);
}

template <typename DstTile, typename Src0Tile, typename Src1Tile>
PTO_INTERNAL void TMAX_IMPL(DstTile& dst, Src0Tile& src0, Src1Tile& src1)
{
    // Exact operand aliases are safe; shifted aliases can overwrite values
    // that a later vector repeat has not consumed yet.
    const auto address = reinterpret_cast<uintptr_t>(dst.data());
    if ((detail::A1BuffersOverlap(dst, src0) && address != reinterpret_cast<uintptr_t>(src0.data())) ||
        (detail::A1BuffersOverlap(dst, src1) && address != reinterpret_cast<uintptr_t>(src1.data()))) {
        trap();
        return;
    }
    A1ElementwiseVector<Op::TMAX>(dst, src0, src1);
}

template <typename TileData>
PTO_INTERNAL void TEXPANDS_IMPL(TileData& dst, typename TileData::DType scalar)
{
    // The scalar specialization does not read either source pointer.
    A1ElementwiseVector<Op::TEXPANDS>(dst, dst, dst, scalar);
}

template <auto PrecisionType = DivAlgorithm::DEFAULT, typename DstTile, typename Src0Tile, typename Src1Tile>
PTO_INTERNAL void TDIV_IMPL(DstTile& dst, Src0Tile& src0, Src1Tile& src1)
{
    PTO_STATIC_ASSERT(PrecisionType == DivAlgorithm::DEFAULT, "A1 division supports default precision only.");
    A1ElementwiseVector<Op::TDIV>(dst, src0, src1);
}

template <auto PrecisionType = ExpAlgorithm::DEFAULT, typename DstTile, typename SrcTile>
PTO_INTERNAL void TEXP_IMPL(DstTile& dst, SrcTile& src)
{
    PTO_STATIC_ASSERT(PrecisionType == ExpAlgorithm::DEFAULT, "A1 exponential supports default precision only.");
    A1ElementwiseVector<Op::TEXP>(dst, src, src);
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void A1ConvertVector(DstTile& dst, SrcTile& src, RoundMode mode, SaturationMode saturation)
{
    detail::CheckA1VectorTile<DstTile>();
    detail::CheckA1VectorTile<SrcTile>();
    PTO_STATIC_ASSERT(
        DstTile::Rows == SrcTile::Rows && DstTile::Cols == SrcTile::Cols,
        "A1 floating conversion requires matching tile shapes.");
    PTO_STATIC_ASSERT(
        (!std::is_same<typename DstTile::DType, typename SrcTile::DType>::value),
        "A1 floating conversion requires an FP16/FP32 dtype pair.");
    // Always enforce unsupported dynamic options; assertions cannot disappear
    // in release builds and silently select another rounding operation.
    if (mode != RoundMode::CAST_NONE || saturation != SaturationMode::OFF) {
        trap();
        return;
    }
    const uintptr_t dstBegin = reinterpret_cast<uintptr_t>(dst.data());
    const uintptr_t srcBegin = reinterpret_cast<uintptr_t>(src.data());
    constexpr unsigned dstBytes = DstTile::Numel * sizeof(typename DstTile::DType);
    constexpr unsigned srcBytes = SrcTile::Numel * sizeof(typename SrcTile::DType);
    if (!(dstBegin + dstBytes <= srcBegin || srcBegin + srcBytes <= dstBegin ||
          (sizeof(typename DstTile::DType) == 2 && dstBegin == srcBegin))) {
        trap();
        return;
    }
    constexpr unsigned elementsPerRepeat = 64;
    constexpr unsigned fullElements = DstTile::Numel / elementsPerRepeat * elementsPerRepeat;
    constexpr unsigned tailElements = DstTile::Numel % elementsPerRepeat;
    set_mask_norm();
    set_vector_mask(0, UINT64_MAX);
    for (unsigned offset = 0; offset < fullElements; offset += 255 * elementsPerRepeat) {
        const unsigned remaining = (fullElements - offset) / elementsPerRepeat;
        detail::A1FloatingConvertRepeat(dst.data() + offset, src.data() + offset, remaining > 255 ? 255 : remaining);
    }
    if constexpr (tailElements != 0) {
        set_vector_mask(0, (uint64_t{1} << tailElements) - 1);
        detail::A1FloatingConvertRepeat(dst.data() + fullElements, src.data() + fullElements, 1);
    }
    set_vector_mask(UINT64_MAX, UINT64_MAX);
}

} // namespace pto
#endif // PTO_NPU_A1_TVECTOR_HPP
