/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_A1_TIMG2COL_HPP
#define PTO_NPU_A1_TIMG2COL_HPP

#include <pto/common/img2col_filter_order.hpp>
#include <pto/npu/a1/TVector.hpp>

namespace pto {
namespace detail {
template <typename Conv>
PTO_INTERNAL void SetA1Fmatrix(Conv& src)
{
    uint64_t value = src.GetFmapW() | (uint64_t(src.GetFmapH()) << 16);
    for (unsigned i = 0; i < 4; ++i)
        value |= uint64_t(src.GetPadList(i)) << (32 + i * 8);
    set_fmatrix(value);
    const half padding = src.GetPadValue();
    set_padding(*reinterpret_cast<const uint16_t*>(&padding));
    // dav-m200 has no SET_L3D_RPT target feature. Its native V2
    // instruction uses the explicit M/K window; only default repeat
    // metadata is accepted by ValidA1ConvGeometry.
}
} // namespace detail

template <typename Conv, typename Global>
PTO_INTERNAL void A1LoadConvImage(Conv& dst, Global& src)
{
    detail::CheckA1ConvImage<Conv>();
    PTO_STATIC_ASSERT(
        (std::is_same_v<typename Global::RawDType, half> && Global::layout == Layout::NC1HWC0),
        "A1 image loads require an FP16 NC1HWC0 global tensor.");
    constexpr uint64_t elements = uint64_t(Conv::staticShape[1]) * Conv::staticShape[2] * Conv::staticShape[3] * 16;
    for (unsigned dim = 0; dim < 5; ++dim)
        if (Global::staticShape[dim] != Conv::staticShape[dim]) {
            trap();
            return;
        }
    // The descriptor always describes logical NC1HWC0. Its strides can
    // additionally represent a C16-aligned channel slice of a physical NHWC
    // tensor, avoiding a temporary global layout conversion.
    uint64_t expected = 1;
    bool contiguous = true;
    for (int dim = 4; dim >= 0; --dim) {
        if (Global::staticStride[dim] != expected)
            contiguous = false;
        expected *= Global::staticShape[dim];
    }
    if (contiguous) {
        // A 1 MiB image has at most 32768 contiguous 32-byte blocks.
        copy_gm_to_cbuf(dst.data(), src.data(), 0, 1, elements / 16, 0, 0, (pad_t)0);
        return;
    }
    constexpr int64_t rowStride = Global::staticStride[3];
    constexpr int64_t pixels = Conv::staticShape[2] * Conv::staticShape[3];
    if (rowStride < Conv::staticShape[1] * 16 || rowStride % 16 != 0 || rowStride / 16 - 1 > 65535 ||
        Global::staticStride[4] != 1 || Global::staticStride[1] != 16 ||
        Global::staticStride[2] != Conv::staticShape[3] * rowStride || Global::staticStride[0] < pixels * rowStride) {
        trap();
        return;
    }
    // Each C0 plane is stored contiguously in L1. Split the source gather
    // into at most 4095 bursts, matching the native DMA descriptor limit.
    for (int64_t channel = 0; channel < Conv::staticShape[1]; ++channel) {
        for (int64_t first = 0; first < pixels; first += 4095) {
            const int64_t count = pixels - first < 4095 ? pixels - first : 4095;
            copy_gm_to_cbuf(
                dst.data() + (channel * pixels + first) * 16, src.data() + first * rowStride + channel * 16, 0, count,
                1, rowStride / 16 - 1, 0, (pad_t)0);
        }
    }
}

template <typename Left, typename Conv, SetFmatrixMode Mode>
PTO_INTERNAL void A1Img2col(Left& dst, Conv& src, uint16_t posM, uint16_t posK)
{
    detail::CheckA1ConvImage<Conv>();
    PTO_STATIC_ASSERT(
        (Mode == SetFmatrixMode::FMATRIX_A_AUTO || Mode == SetFmatrixMode::FMATRIX_A_MANUAL),
        "A1 im2col supports the FMATRIX A bank only.");
    PTO_STATIC_ASSERT(
        (Left::Loc == TileType::Left && std::is_same_v<typename Left::DType, half> && Left::isRowMajor &&
         Left::SFractal == SLayout::RowMajor && Left::SFractalSize == 512),
        "A1 im2col requires an FP16 ZZ Left tile.");
    PTO_STATIC_ASSERT(
        (Left::Rows > 0 && Left::Cols > 0 && Left::Rows % 16 == 0 && Left::Cols % 16 == 0 && Left::ValidRow > 0 &&
         Left::ValidRow <= Left::Rows && Left::ValidRow % 16 == 0 && Left::ValidCol == Left::Cols &&
         Left::Numel * sizeof(half) <= 65536),
        "A1 im2col requires full K and aligned M rows within 64 KiB L0A.");
    if (!detail::ValidA1ConvGeometry(src)) {
        trap();
        return;
    }
    const int spanH =
        src.GetFmapH() + src.GetPadList(2) + src.GetPadList(3) - src.GetDilationH() * (src.GetFilterH() - 1) - 1;
    const int spanW =
        src.GetFmapW() + src.GetPadList(0) + src.GetPadList(1) - src.GetDilationW() * (src.GetFilterW() - 1) - 1;
    // Reject empty outputs before signed division, which truncates toward zero.
    if (spanH < 0 || spanW < 0) {
        trap();
        return;
    }
    const int outH = spanH / src.GetStrideH() + 1;
    const int outW = spanW / src.GetStrideW() + 1;
    if (outH <= 0 || outW <= 0 || posK % 16 || uint64_t(posM) + Left::ValidRow > uint64_t(outH) * outW ||
        uint64_t(posK) + Left::Cols > Conv::staticShape[1] * 16ULL * src.GetFilterH() * src.GetFilterW()) {
        trap();
        return;
    }
    if constexpr (Mode == SetFmatrixMode::FMATRIX_A_AUTO)
        detail::SetA1Fmatrix(src);
    // dav-m200 uses the native fourteen-argument overload, unlike A2/A5's
    // additional high-filter and FMATRIX-bank control arguments.
    img2colv2_cbuf_to_ca(
        dst.data(), src.data(), Left::Cols, Left::ValidRow, posK, posM, src.GetStrideW(), src.GetStrideH(),
        src.GetFilterW(), src.GetFilterH(), src.GetDilationW(), src.GetDilationH(), false, src.GetChannelSize());
}

namespace detail {
template <size_t Column, int Tap, size_t Count, typename Left, typename Conv>
PTO_INTERNAL void A1Img2colFilterTap(Left& dst, Conv& src, int firstW, int firstH, uint16_t c1)
{
    // ZZ address = (M16 * Count + filter-column) *256 + M0 *16 + C0.
    // V1 repeats over M16 fragments while retaining this fixed filter tap.
    img2col_cbuf_to_ca(
        dst.data() + Column * 256, src.data(), Tap % src.GetFilterW(), Tap / src.GetFilterW(), firstW, firstH, c1,
        src.GetStrideW(), src.GetStrideH(), src.GetFilterW(), src.GetFilterH(), src.GetDilationW(), src.GetDilationH(),
        Count, true, Left::ValidRow / 16, CSIZE0);
}

template <typename Left, typename Conv, int... Order, size_t... Column>
PTO_INTERNAL void A1Img2colFilterTaps(
    Left& dst, Conv& src, uint16_t posM, uint16_t posK, std::integer_sequence<int, Order...>,
    std::index_sequence<Column...>)
{
    const int outW =
        (src.GetFmapW() + src.GetPadList(0) + src.GetPadList(1) - src.GetDilationW() * (src.GetFilterW() - 1) - 1) /
            src.GetStrideW() +
        1;
    const int firstW = posM % outW * src.GetStrideW() - src.GetPadList(0);
    const int firstH = posM / outW * src.GetStrideH() - src.GetPadList(2);
    const uint16_t c1 = posK / (uint64_t(src.GetFilterH()) * src.GetFilterW() * 16);
    (A1Img2colFilterTap<Column, Order, sizeof...(Order)>(dst, src, firstW, firstH, c1), ...);
}
} // namespace detail

template <typename Left, typename Conv, SetFmatrixMode Mode, int... Order>
PTO_INTERNAL void A1Img2colOrdered(
    Left& dst, Conv& src, uint16_t posM, uint16_t posK, std::integer_sequence<int, Order...> order)
{
    detail::CheckA1Img2colFilterOrder<Left, Conv>(order);
    PTO_STATIC_ASSERT(
        (Mode == SetFmatrixMode::FMATRIX_A_AUTO || Mode == SetFmatrixMode::FMATRIX_A_MANUAL),
        "A1 ordered im2col supports the FMATRIX A bank only.");
    if (!detail::ValidA1Img2colFilterWindow<Left, Conv>(src, posM, posK, order)) {
        trap();
        return;
    }
    if constexpr (Mode == SetFmatrixMode::FMATRIX_A_AUTO)
        detail::SetA1Fmatrix(src);
    // A compile-time selection produces one V1 M-repeat instruction per
    // tap, matching phase-major native extraction without a temporary patch.
    detail::A1Img2colFilterTaps(dst, src, posM, posK, order, std::make_index_sequence<sizeof...(Order)>{});
}
} // namespace pto
#endif // PTO_NPU_A1_TIMG2COL_HPP
