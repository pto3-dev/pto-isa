/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_A1_TRESHAPE_HPP
#define PTO_NPU_A1_TRESHAPE_HPP

#include <pto/npu/a1/TVector.hpp>
#include <pto/npu/a2a3/TAssign.hpp>

namespace pto {

// Full row-major Vec tiles have address (base + (row * Cols + col) * sizeof(T)).
// Equal element counts therefore allow shape changes without moving any bytes.
template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TRESHAPE_IMPL(DstTile& dst, SrcTile& src)
{
    detail::CheckA1VectorTile<SrcTile>();
    detail::CheckA1VectorTile<DstTile>();
    PTO_STATIC_ASSERT(
        (std::is_same<typename DstTile::DType, typename SrcTile::DType>::value),
        "A1 TRESHAPE requires matching element types.");
    PTO_STATIC_ASSERT(DstTile::Numel == SrcTile::Numel, "A1 TRESHAPE requires equal element counts.");
    TASSIGN_IMPL(dst, reinterpret_cast<uintptr_t>(src.data()));
}

} // namespace pto
#endif // PTO_NPU_A1_TRESHAPE_HPP
