/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_A1_TEXPAND_HPP
#define PTO_NPU_A1_TEXPAND_HPP

#include <pto/npu/a1/TVector.hpp>

namespace pto {
namespace detail {
// Seed each row without scalar UB loads, then copy its first block across
// the remaining columns. All operations move raw bits, including
// signed zero and NaN payloads. No temporary UB allocation is required.
template <typename DstTile, typename SrcTile>
PTO_INTERNAL void A1VectorRowExpand(DstTile& dst, SrcTile& src)
{
    using Element = typename DstTile::DType;
    constexpr unsigned blocks = DstTile::Cols * sizeof(Element) / 32;
    constexpr unsigned group = sizeof(Element) == 2 ? 16 : 8;
    for (unsigned row = 0; row < DstTile::Rows; row += 255 * group) {
        const unsigned remaining = (DstTile::Rows - row) / group;
        const uint8_t repeats = remaining < 255 ? remaining : 255;
        uint64_t inputAddresses[16], outputAddresses[16];
        for (unsigned lane = 0; lane < 16; ++lane) {
            // Transposing identical source rows produces a constant row for
            // each input element. The b32 instruction writes two blocks/row.
            inputAddresses[lane] = reinterpret_cast<uint64_t>(src.data() + row);
            if constexpr (sizeof(Element) == 2)
                outputAddresses[lane] = reinterpret_cast<uint64_t>(dst.data() + (row + lane) * DstTile::Cols);
            else
                outputAddresses[lane] =
                    reinterpret_cast<uint64_t>(dst.data() + (row + lane / 2) * DstTile::Cols + lane % 2 * 8);
        }
        set_va_reg_sb(VA2, inputAddresses);
        set_va_reg_sb(VA3, inputAddresses + 8);
        set_va_reg_sb(VA0, outputAddresses);
        set_va_reg_sb(VA1, outputAddresses + 8);
        if constexpr (sizeof(Element) == 2)
            scatter_vnchwconv_b16(VA0, VA2, repeats, repeats == 1 ? 0 : DstTile::Cols, repeats == 1 ? 0 : 1);
        else
            scatter_vnchwconv_b32(VA0, VA2, repeats, repeats == 1 ? 0 : DstTile::Cols, repeats == 1 ? 0 : 1);
    }
    pipe_barrier(PIPE_V);
    constexpr unsigned rowHalfwords = DstTile::Cols * sizeof(Element) / 2;
    constexpr unsigned seedHalfwords = sizeof(Element) == 2 ? 16 : 32;
    constexpr unsigned copiedHalfwords = rowHalfwords - seedHalfwords;
    constexpr unsigned full = copiedHalfwords / 128;
    constexpr unsigned tail = copiedHalfwords % 128;
    auto* words = reinterpret_cast<__ubuf__ uint16_t*>(dst.data());
    set_mask_norm();
    for (unsigned row = 0; row < DstTile::Rows; row += 255) {
        const unsigned repeat = DstTile::Rows - row < 255 ? DstTile::Rows - row : 255;
        auto* source = words + row * rowHalfwords;
        if constexpr (full > 0) {
            set_vector_mask(-1, -1);
            for (unsigned chunk = 0; chunk < full; ++chunk)
                vor(source + seedHalfwords + chunk * 128, source, source, repeat, 1, 0, 0, blocks, blocks, blocks);
        }
        if constexpr (tail > 0) {
            constexpr uint64_t low = tail >= 64 ? UINT64_MAX : (uint64_t(1) << (tail % 64)) - 1;
            constexpr uint64_t high = tail > 64 ? (uint64_t(1) << (tail % 64)) - 1 : 0;
            set_vector_mask(high, low);
            vor(source + seedHalfwords + full * 128, source, source, repeat, 1, 0, 0, blocks, blocks, blocks);
        }
    }
    if constexpr (sizeof(Element) == 4)
        set_vector_mask(0, UINT64_MAX);
    else
        set_vector_mask(UINT64_MAX, UINT64_MAX);
}

} // namespace detail

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TROWEXPAND_IMPL(DstTile& dst, SrcTile& src)
{
    detail::CheckA1VectorTile<DstTile>();
    detail::CheckA1PackedColumn<SrcTile>();
    PTO_STATIC_ASSERT(
        DstTile::Rows == SrcTile::Rows && (std::is_same<typename DstTile::DType, typename SrcTile::DType>::value),
        "A1 row expansion requires a matching same-dtype packed Mx1 source.");
    if (detail::A1BuffersOverlap(dst, src)) {
        trap();
        return;
    }
    constexpr unsigned blocks = DstTile::Cols * sizeof(typename DstTile::DType) / 32;
    // Bound the copy's 8-bit repeat strides. b32 transposition seeds 16
    // elements per row; retain the scalar path for narrower FP32 rows.
    if constexpr (blocks <= 255 && DstTile::Cols >= 16) {
        detail::A1VectorRowExpand(dst, src);
        return;
    }
    // Each row consumes scalar UB data produced by DMA or a vector reduction.
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    for (unsigned row = 0; row < DstTile::Rows; ++row) {
        const auto scalar = src.data()[row];
        auto* output = dst.data() + row * DstTile::Cols;
        detail::A1VectorInstruction<Op::TEXPANDS, DstTile::Cols>(output, output, output, scalar);
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TCOLEXPAND_IMPL(DstTile& dst, SrcTile& src)
{
    detail::CheckA1VectorTile<DstTile>();
    detail::CheckA1VectorTile<SrcTile>();
    PTO_STATIC_ASSERT(
        SrcTile::Rows == 1 && DstTile::Cols == SrcTile::Cols &&
            (std::is_same<typename DstTile::DType, typename SrcTile::DType>::value),
        "A1 column expansion requires a matching same-dtype 1xN source.");
    if (detail::A1BuffersOverlap(dst, src) && dst.data() != src.data()) {
        trap();
        return;
    }
    using Element = typename DstTile::DType;
    constexpr unsigned blocks = DstTile::Cols * sizeof(Element) / 32;
    constexpr unsigned halfwords = DstTile::Cols * sizeof(Element) / 2;
    auto* output = reinterpret_cast<__ubuf__ uint16_t*>(dst.data());
    auto* input = reinterpret_cast<__ubuf__ uint16_t*>(src.data());
    // Same-base aliases keep their first row immutable while later rows read it.
    const unsigned first = dst.data() == src.data() ? 1 : 0;
    set_mask_norm();
    if constexpr (blocks <= 255) {
        for (unsigned row = first; row < DstTile::Rows; row += 255) {
            const unsigned remaining = DstTile::Rows - row;
            const uint8_t repeat = remaining < 255 ? remaining : 255;
            detail::A1CopyBitRows<halfwords>(output + row * halfwords, input, repeat, blocks);
        }
    } else {
        // Wide rows cannot encode an 8-bit repeat stride; copy each row with
        // a single repeat so no stride truncation or out-of-window DMA occurs.
        for (unsigned row = first; row < DstTile::Rows; ++row)
            detail::A1CopyBitRows<halfwords>(output + row * halfwords, input, 1, 0);
    }
    if constexpr (sizeof(Element) == 4)
        set_vector_mask(0, UINT64_MAX);
    else
        set_vector_mask(UINT64_MAX, UINT64_MAX);
}
} // namespace pto
#endif // PTO_NPU_A1_TEXPAND_HPP
