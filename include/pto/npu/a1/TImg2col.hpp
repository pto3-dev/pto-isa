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

#include <pto/npu/a1/TVector.hpp>

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
    // A1 has one FMATRIX bank. Restrict the first public profile to a single
    // complete image and the ordinary, untransposed LoadData3DV2 operation.
    return src.GetFmapH() == Conv::staticShape[2] && src.GetFmapW() == Conv::staticShape[3] && src.GetFilterH() > 0 &&
           src.GetFilterH() <= 255 && src.GetFilterW() > 0 && src.GetFilterW() <= 255 && src.GetStrideH() > 0 &&
           src.GetStrideH() <= 63 && src.GetStrideW() > 0 && src.GetStrideW() <= 63 && src.GetDilationH() > 0 &&
           src.GetDilationW() > 0 && src.GetChannelSize() == Conv::staticShape[1] * 16 && !src.GetTranspose() &&
           src.GetRepeatStride() == 0 && src.GetRepeatTime() == 1 && src.GetRepeatMode() == 0 &&
           src.GetDstStride() == 0 && src.GetDstMposition() == 0;
}

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
    uint64_t expected = 1;
    for (int dim = 4; dim >= 0; --dim) {
        if (Global::staticStride[dim] != expected) {
            trap();
            return;
        }
        expected *= Global::staticShape[dim];
    }
    // A 1 MiB image has at most 32768 contiguous 32-byte blocks, below the
    // 16-bit burst-length limit. No image transpose or UB staging is issued.
    copy_gm_to_cbuf(dst.data(), src.data(), 0, 1, elements / 16, 0, 0, (pad_t)0);
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
} // namespace pto
#endif // PTO_NPU_A1_TIMG2COL_HPP
