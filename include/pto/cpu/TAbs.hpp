/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TABS_HPP
#define TABS_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"

namespace pto {

template <typename TileDst, typename TileSrc>
void TAbs_Impl(TileDst& dst, TileSrc& src, unsigned validRow, unsigned validCol)
{
    for (size_t c = 0; c < validCol; c++) {
        for (size_t r = 0; r < validRow; r++) {
            const auto val = src.GetElement(r, c);
            dst.SetElement(r, c, val < 0 ? -val : val);
        }
    }
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TABS_IMPL(TileDst& dst, TileSrc& src)
{
    static_assert(
        std::is_same_v<typename TileDst::DType, typename TileSrc::DType>,
        "Fix: TABS the data type of dst must be consistent with src.");
    static_assert(
        std::is_same<typename TileDst::DType, int32_t>::value || std::is_same<typename TileDst::DType, int>::value ||
            std::is_same<typename TileDst::DType, int16_t>::value ||
            std::is_same<typename TileDst::DType, int8_t>::value ||
            std::is_same<typename TileDst::DType, half>::value ||
            std::is_same<typename TileDst::DType, bfloat16_t>::value ||
            std::is_same<typename TileDst::DType, float>::value,
        "TABS: Invalid data type");
    unsigned row = dst.GetValidRow();
    unsigned col = dst.GetValidCol();
    PTO_ASSERT(
        src.GetValidRow() == row && src.GetValidCol() == col,
        "Fix: TABS input tile src valid shape mismatch with output tile dst shape.");
    TAbs_Impl(dst, src, row, col);
}
} // namespace pto
#endif // TABS_HPP
