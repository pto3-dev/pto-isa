/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_A1_TREDUCE_HPP
#define PTO_NPU_A1_TREDUCE_HPP

#include <pto/npu/a1/TVector.hpp>

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

template <bool Maximum, typename DstTile, typename SrcTile>
PTO_INTERNAL void A1ColumnReduction(DstTile& dst, SrcTile& src)
{
    CheckA1VectorTile<DstTile>();
    CheckA1VectorTile<SrcTile>();
    PTO_STATIC_ASSERT(
        DstTile::Rows == 1 && DstTile::Cols == SrcTile::Cols &&
            (std::is_same<typename DstTile::DType, typename SrcTile::DType>::value),
        "A1 column reductions require matching FP16/FP32 1xN row results.");
    if (A1BuffersOverlap(dst, src)) {
        trap();
        return;
    }
    constexpr Op operation = Maximum ? Op::TMAX : Op::TADD;
    // vmax(x,x) copies a row without adding a rounding step for finite inputs.
    A1VectorInstruction<Op::TMAX, DstTile::Numel>(dst.data(), src.data(), src.data());
    for (unsigned row = 1; row < SrcTile::Rows; ++row) {
        pipe_barrier(PIPE_V);
        A1VectorInstruction<operation, DstTile::Numel>(dst.data(), dst.data(), src.data() + row * SrcTile::Cols);
    }
}

template <bool Maximum, typename DstTile, typename SrcTile, typename TmpTile>
PTO_INTERNAL void A1RowReduction(DstTile& dst, SrcTile& src, TmpTile& tmp)
{
    CheckA1PackedColumn<DstTile>();
    CheckA1VectorTile<SrcTile>();
    CheckA1VectorTile<TmpTile>();
    PTO_STATIC_ASSERT(
        DstTile::Rows == SrcTile::Rows && TmpTile::Rows >= SrcTile::Rows && TmpTile::Cols >= SrcTile::Cols &&
            (std::is_same<typename DstTile::DType, typename SrcTile::DType>::value) &&
            (std::is_same<typename TmpTile::DType, typename SrcTile::DType>::value),
        "A1 row reductions require matching packed Mx1 results and full-size same-dtype scratch.");
    if (A1BuffersOverlap(dst, src) || A1BuffersOverlap(dst, tmp) || A1BuffersOverlap(src, tmp)) {
        trap();
        return;
    }
    using Element = typename SrcTile::DType;
    constexpr unsigned lanes = 256 / sizeof(Element);
    constexpr unsigned reducedLanes = SrcTile::Cols < lanes ? SrcTile::Cols : lanes;
    constexpr unsigned tail = SrcTile::Cols % lanes;
    for (unsigned row = 0; row < SrcTile::Rows; ++row) {
        auto* workspace = tmp.data() + row * TmpTile::Cols;
        auto* source = src.data() + row * SrcTile::Cols;
        A1VectorInstruction<Op::TMAX, reducedLanes>(workspace, source, source);
        for (unsigned offset = lanes; offset < SrcTile::Cols; offset += lanes) {
            pipe_barrier(PIPE_V);
            const bool partial = offset + lanes > SrcTile::Cols;
            if constexpr (Maximum) {
                if (partial)
                    A1VectorInstruction<Op::TMAX, tail>(workspace, workspace, source + offset);
                else
                    A1VectorInstruction<Op::TMAX, lanes>(workspace, workspace, source + offset);
            } else {
                if (partial)
                    A1VectorInstruction<Op::TADD, tail>(workspace, workspace, source + offset);
                else
                    A1VectorInstruction<Op::TADD, lanes>(workspace, workspace, source + offset);
            }
        }
        pipe_barrier(PIPE_V);
        set_mask_norm();
        constexpr uint64_t low = reducedLanes >= 64 ? UINT64_MAX : (UINT64_C(1) << (reducedLanes % 64)) - 1;
        constexpr uint64_t high = reducedLanes == 128 ? UINT64_MAX :
                                  reducedLanes <= 64  ? 0 :
                                                        (UINT64_C(1) << (reducedLanes > 64 ? reducedLanes - 64 : 0)) - 1;
        set_vector_mask(high, low);
        if constexpr (Maximum)
            vcmax(workspace, workspace, 1, 1, 1, 8, false, false, false);
        else
            vcadd(workspace, workspace, 1, 1, 1, 8);
    }
    // m200 vcmax writes value/index pairs. Pack only each row's value;
    // explicit scalar/vector handoffs also order subsequent vector consumers.
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    for (unsigned row = 0; row < SrcTile::Rows; ++row)
        dst.data()[row] = tmp.data()[row * TmpTile::Cols];
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
    set_vector_mask(UINT64_MAX, UINT64_MAX);
}
} // namespace detail

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TCOLSUM_IMPL(DstTile& dst, SrcTile& src)
{
    detail::A1ColumnReduction<false>(dst, src);
}
template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TCOLMAX_IMPL(DstTile& dst, SrcTile& src)
{
    detail::A1ColumnReduction<true>(dst, src);
}
template <typename DstTile, typename SrcTile, typename TmpTile>
PTO_INTERNAL void TROWSUM_IMPL(DstTile& dst, SrcTile& src, TmpTile& tmp)
{
    detail::A1RowReduction<false>(dst, src, tmp);
}
template <typename DstTile, typename SrcTile, typename TmpTile>
PTO_INTERNAL void TROWMAX_IMPL(DstTile& dst, SrcTile& src, TmpTile& tmp)
{
    detail::A1RowReduction<true>(dst, src, tmp);
}
} // namespace pto
#endif // PTO_NPU_A1_TREDUCE_HPP
