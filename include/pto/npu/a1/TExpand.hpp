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
    for (unsigned row = 0; row < DstTile::Rows; ++row) {
        if (row == 0 && dst.data() == src.data())
            continue;
        auto* output = dst.data() + row * DstTile::Cols;
        detail::A1VectorInstruction<Op::TMAX, DstTile::Cols>(output, src.data(), src.data());
    }
}
} // namespace pto
#endif // PTO_NPU_A1_TEXPAND_HPP
