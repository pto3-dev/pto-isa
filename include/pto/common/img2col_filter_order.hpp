/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMMON_IMG2COL_FILTER_ORDER_HPP
#define PTO_COMMON_IMG2COL_FILTER_ORDER_HPP

#include <utility>

#include <pto/common/pto_tile.hpp>

namespace pto {
namespace detail {
template <typename Conv>
PTO_INTERNAL void CheckA1ConvImage()
{
    PTO_STATIC_ASSERT(
        (Conv::Loc == TileType::Mat && std::is_same_v<typename Conv::DType, half> && Conv::layout == Layout::NC1HWC0 &&
         Conv::totalDimCount == 5),
        "A1 im2col requires an FP16 NC1HWC0 Mat image.");
    PTO_STATIC_ASSERT(
        (Conv::dynamicDimCount == 0 && Conv::validDynamicDimCount == 0 && Conv::staticShape[0] == 1 &&
         Conv::staticShape[1] > 0 && Conv::staticShape[1] <= 4095 && Conv::staticShape[2] > 0 &&
         Conv::staticShape[2] <= 65535 && Conv::staticShape[3] > 0 && Conv::staticShape[3] <= 65535 &&
         Conv::staticShape[4] == 16),
        "A1 im2col requires one static C16 image with 16-bit spatial dimensions.");
    constexpr uint64_t elements = uint64_t(Conv::staticShape[1]) * Conv::staticShape[2] * Conv::staticShape[3] * 16;
    PTO_STATIC_ASSERT(
        (elements * sizeof(half) <= 1048576 && Conv::bufferSize >= elements),
        "A1 im2col image must fit its allocation and the 1 MiB L1 budget.");
    PTO_STATIC_ASSERT(
        (Conv::staticValidShape[0] == Conv::staticShape[0] && Conv::staticValidShape[1] == Conv::staticShape[1] &&
         Conv::staticValidShape[2] == Conv::staticShape[2] && Conv::staticValidShape[3] == Conv::staticShape[3] &&
         Conv::staticValidShape[4] == Conv::staticShape[4]),
        "A1 im2col requires the complete static image valid shape.");
}

template <typename Conv>
PTO_INTERNAL bool ValidA1ConvGeometry(Conv& src)
{
    // A1 has one FMATRIX bank. Both paths use a complete image and default
    // V2 metadata; filter ordering derives its V1 M-repeat operands directly.
    return src.GetFmapH() == Conv::staticShape[2] && src.GetFmapW() == Conv::staticShape[3] && src.GetFilterH() > 0 &&
           src.GetFilterH() <= 255 && src.GetFilterW() > 0 && src.GetFilterW() <= 255 && src.GetStrideH() > 0 &&
           src.GetStrideH() <= 63 && src.GetStrideW() > 0 && src.GetStrideW() <= 63 && src.GetDilationH() > 0 &&
           src.GetDilationW() > 0 && src.GetChannelSize() == Conv::staticShape[1] * 16 && !src.GetTranspose() &&
           src.GetRepeatStride() == 0 && src.GetRepeatTime() == 1 && src.GetRepeatMode() == 0 &&
           src.GetDstStride() == 0 && src.GetDstMposition() == 0;
}

// Selection/order is functional: each destination C16 block is written by
// its selected filter tap. This also represents smaller K96/K48 subsets
// without allocating an unused full K144 patch or a partial-write alias.
template <int... Order>
PTO_INTERNAL constexpr bool IsImg2colFilterOrderValid()
{
    constexpr size_t count = sizeof...(Order);
    if constexpr (count == 0 || count > 127) {
        return false;
    } else {
        constexpr int taps[] = {Order...};
        // At most127 compile-time entries; this emits no runtime device loop.
        // A filter dimension is byte-sized, so its largest index is65024.
        for (size_t i = 0; i < count; ++i) {
            if (taps[i] < 0 || taps[i] >= 255 * 255)
                return false;
            for (size_t j = 0; j < i; ++j)
                if (taps[i] == taps[j])
                    return false;
        }
        return true;
    }
}

template <typename Left, typename Conv, int... Order>
PTO_INTERNAL void CheckA1Img2colFilterOrder(std::integer_sequence<int, Order...>)
{
    CheckA1ConvImage<Conv>();
    PTO_STATIC_ASSERT(
        (IsImg2colFilterOrderValid<Order...>()),
        "A1 im2col filter order requires 1 to 127 unique indices in [0, 65025).");
    PTO_STATIC_ASSERT(
        (Left::Loc == TileType::Left && std::is_same_v<typename Left::DType, half> && Left::isRowMajor &&
         Left::SFractal == SLayout::RowMajor && Left::SFractalSize == 512),
        "A1 ordered im2col requires an FP16 ZZ Left tile.");
    PTO_STATIC_ASSERT(
        (Left::Rows > 0 && Left::Rows <= 2048 && Left::Rows % 16 == 0 && Left::ValidRow > 0 &&
         Left::ValidRow <= Left::Rows && Left::ValidRow % 16 == 0 && Left::Cols == 16 * sizeof...(Order) &&
         Left::ValidCol == Left::Cols && uint64_t(Left::Rows) * Left::Cols * sizeof(half) <= 65536),
        "A1 ordered im2col requires aligned M and one C16 output block per selected filter within 64 KiB L0A.");
}

template <typename Left, typename Conv, int... Order>
PTO_INTERNAL bool ValidA1Img2colFilterWindow(
    const Conv& src, uint16_t posM, uint16_t posK, std::integer_sequence<int, Order...>)
{
    if (!ValidA1ConvGeometry(src) || src.GetFmapH() > 32767 || src.GetFmapW() > 32767)
        return false;
    const uint64_t filters = uint64_t(src.GetFilterH()) * src.GetFilterW();
    // K names a source C16 group in the original kernel, independently of
    // the number of selected output taps. Tap bounds depend on runtime config.
    if (posK % (filters * 16) || posK / (filters * 16) >= Conv::staticShape[1] || !((uint64_t(Order) < filters) && ...))
        return false;
    const int spanH =
        src.GetFmapH() + src.GetPadList(2) + src.GetPadList(3) - src.GetDilationH() * (src.GetFilterH() - 1) - 1;
    const int spanW =
        src.GetFmapW() + src.GetPadList(0) + src.GetPadList(1) - src.GetDilationW() * (src.GetFilterW() - 1) - 1;
    if (spanH < 0 || spanW < 0)
        return false;
    const int outH = spanH / src.GetStrideH() + 1, outW = spanW / src.GetStrideW() + 1;
    // V1 takes signed16-bit starting coordinates. Bound the complete grid so
    // implicit M-repeat increments cannot cross that coordinate limit either.
    return uint64_t(posM) + Left::ValidRow <= uint64_t(outH) * outW &&
           (outH - 1) * src.GetStrideH() - src.GetPadList(2) <= 32767 &&
           (outW - 1) * src.GetStrideW() - src.GetPadList(0) <= 32767;
}
} // namespace detail
} // namespace pto
#endif // PTO_COMMON_IMG2COL_FILTER_ORDER_HPP
