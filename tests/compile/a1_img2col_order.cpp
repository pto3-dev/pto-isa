/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>
#include <pto/pto-inst.hpp>
#if defined(__CPU_SIM)
#include <bit>
#include <cmath>
#include <cstring>
#include <vector>
#endif

#ifndef ORDER_ROWS
#define ORDER_ROWS 32
#endif
#ifndef ORDER_HEIGHT
#define ORDER_HEIGHT 8
#endif
#ifndef ORDER_WIDTH
#define ORDER_WIDTH 16
#endif
#ifndef ORDER_C1
#define ORDER_C1 2
#endif
#ifndef ORDER_FH
#define ORDER_FH 3
#endif
#ifndef ORDER_FW
#define ORDER_FW 3
#endif
#ifndef ORDER_STRIDE
#define ORDER_STRIDE 1
#endif
#ifndef ORDER_DILATION
#define ORDER_DILATION 1
#endif
#ifndef ORDER_PAD
#define ORDER_PAD 1
#endif
#ifndef ORDER_M
#define ORDER_M 0
#endif
#ifndef ORDER_K
#define ORDER_K 0
#endif
#ifndef ORDER_COUNT
#define ORDER_COUNT (ORDER_FH * ORDER_FW)
#endif
#ifndef ORDER_PHASE
#define ORDER_PHASE 0
#endif
#ifndef ORDER_STEP
#define ORDER_STEP 4
#endif
#ifndef ORDER_MANUAL
#define ORDER_MANUAL 0
#endif
#ifndef ORDER_LVALUE
#define ORDER_LVALUE 0
#endif
constexpr int Rows = ORDER_ROWS, Height = ORDER_HEIGHT, Width = ORDER_WIDTH, C1 = ORDER_C1;
constexpr int FH = ORDER_FH, FW = ORDER_FW, FilterTaps = FH * FW;
constexpr int Taps = ORDER_COUNT, Columns = Taps * 16;
constexpr int ImageElements = Height * Width * C1 * 16;
constexpr int WeightAddress = (ImageElements * 2 + 511) / 512 * 512;
constexpr int LeftAddress = 512, LeftEnd = LeftAddress + Rows * Columns * 2;
using Image = pto::ConvTile<
    pto::TileType::Mat, half, ImageElements, pto::Layout::NC1HWC0, pto::ConvTileShape<1, C1, Height, Width, 16>>;
using Left = pto::Tile<
    pto::TileType::Left, half, Rows, Columns, pto::BLayout::RowMajor, Rows, Columns, pto::SLayout::RowMajor, 512>;
using Mat =
    pto::Tile<pto::TileType::Mat, half, Columns, 16, pto::BLayout::RowMajor, Columns, 16, pto::SLayout::RowMajor, 512>;
using Right = pto::Tile<
    pto::TileType::Right, half, Columns, 16, pto::BLayout::RowMajor, Columns, 16, pto::SLayout::ColMajor, 512>;
using Acc =
    pto::Tile<pto::TileType::Acc, float, Rows, 16, pto::BLayout::ColMajor, Rows, 16, pto::SLayout::RowMajor, 1024>;
using Vec = pto::Tile<pto::TileType::Vec, float, Rows, 16, pto::BLayout::RowMajor>;
template <size_t... Index>
auto MakeFilterOrder(std::index_sequence<Index...>)
    -> std::integer_sequence<int, (ORDER_PHASE + static_cast<int>(Index) * ORDER_STEP) % FilterTaps...>;
#ifndef ORDER_TYPE
using Order = decltype(MakeFilterOrder(std::make_index_sequence<Taps>{}));
#else
using Order = ORDER_TYPE;
#endif
constexpr auto Mode = ORDER_MANUAL ? pto::SetFmatrixMode::FMATRIX_A_MANUAL : pto::SetFmatrixMode::FMATRIX_A_AUTO;

extern "C" __global__ AICORE void OrderedImage(__gm__ half* source, __gm__ half* weights, __gm__ float* output)
{
    Image image;
    Left left;
    Mat mat;
    Right right;
    Acc acc;
    Vec vec;
    pto::TASSIGN<0>(image);
    pto::TASSIGN<LeftAddress>(left);
    pto::TASSIGN<WeightAddress>(mat);
    pto::TASSIGN<0>(right);
    pto::TASSIGN<0>(acc);
    pto::TASSIGN<0>(vec);
    using Input = pto::GlobalTensor<
        half, pto::Shape<1, C1, Height, Width, 16>, pto::Stride<ImageElements, Height * Width * 16, Width * 16, 16, 1>,
        pto::Layout::NC1HWC0>;
    using Weights = pto::GlobalTensor<
        half, pto::Shape<1, 1, 1, Columns, 16>, pto::Stride<Columns * 16, Columns * 16, Columns * 16, 16, 1>>;
    using Output =
        pto::GlobalTensor<float, pto::Shape<1, 1, 1, Rows, 16>, pto::Stride<Rows * 16, Rows * 16, Rows * 16, 16, 1>>;
    Input input(source);
    Weights weight(weights);
    Output out(output);
    pto::TLOAD(image, input);
    pto::TLOAD(mat, weight);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    image.SetFmapH(Height);
    image.SetFmapW(Width);
    image.SetFilterH(FH);
    image.SetFilterW(FW);
    image.SetStrideH(ORDER_STRIDE);
    image.SetStrideW(ORDER_STRIDE);
    image.SetDilationH(ORDER_DILATION);
    image.SetDilationW(ORDER_DILATION);
    image.SetChannelSize(C1 * 16);
    for (int p = 0; p < 4; ++p)
        image.SetPadList(p, ORDER_PAD);
    image.SetPadValue(half(-0.25f));
    image.SetRepeatStride(0);
    image.SetRepeatTime(1);
    image.SetRepeatMode(0);
#if defined(ORDER_INVALID_METADATA)
    image.SetRepeatTime(2);
#endif
    image.SetDstStride(0);
    image.SetDstMposition(0);
    if constexpr (ORDER_MANUAL)
        pto::SETFMATRIX<Image, pto::SetFmatrixMode::FMATRIX_A_AUTO>(image);
#if defined(ORDER_NATIVE_CONTROL)
    // Independent direct-intrinsic control follows the existing native
    // spatial_repeat path; compare its optimized instruction operands.
    if constexpr (!ORDER_MANUAL)
        pto::SETFMATRIX<Image, pto::SetFmatrixMode::FMATRIX_A_AUTO>(image);
    constexpr int ow = (Width + 2 * ORDER_PAD - ORDER_DILATION * (FW - 1) - 1) / ORDER_STRIDE + 1;
#pragma unroll
    for (int j = 0; j < Taps; ++j) {
        const int tap = (ORDER_PHASE + j * ORDER_STEP) % FilterTaps;
        img2col_cbuf_to_ca(
            left.data() + j * 256, image.data(), tap % FW, tap / FW, ORDER_M % ow * ORDER_STRIDE - ORDER_PAD,
            ORDER_M / ow * ORDER_STRIDE - ORDER_PAD, ORDER_K / (FilterTaps * 16), ORDER_STRIDE, ORDER_STRIDE, FW, FH,
            ORDER_DILATION, ORDER_DILATION, Taps, true, Rows / 16, CSIZE0);
    }
#else
    if constexpr (ORDER_LVALUE) {
        Order order;
        pto::TIMG2COL<Left, Image, Mode>(left, image, ORDER_M, ORDER_K, order);
    } else {
        pto::TIMG2COL<Left, Image, Mode>(left, image, ORDER_M, ORDER_K, Order{});
    }
#endif
    pto::TMOV(right, mat);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    pto::TMATMUL(acc, left, right);
    set_flag(PIPE_M, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_V, EVENT_ID0);
    pto::TMOV<Vec, Acc, pto::AccToVecMode::SingleModeVec0>(vec, acc);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pto::TSTORE(out, vec);
}

#if defined(__CPU_SIM)
int main()
{
    static_assert(LeftEnd + 512 <= 65536);
    constexpr int OutputWidth = (Width + 2 * ORDER_PAD - ORDER_DILATION * (FW - 1) - 1) / ORDER_STRIDE + 1;
    constexpr int OutputHeight = (Height + 2 * ORDER_PAD - ORDER_DILATION * (FH - 1) - 1) / ORDER_STRIDE + 1;
    static_assert(ORDER_M + Rows <= OutputHeight * OutputWidth && ORDER_K % (FilterTaps * 16) == 0);
    static_assert(ORDER_K / (FilterTaps * 16) < C1);
    using Guard =
        pto::Tile<pto::TileType::Left, half, 16, 16, pto::BLayout::RowMajor, 16, 16, pto::SLayout::RowMajor, 512>;
    Guard before, after;
    Left check;
    pto::TASSIGN<0>(before);
    pto::TASSIGN<LeftEnd>(after);
    pto::TASSIGN<LeftAddress>(check);
    for (int dataset = 0; dataset < 3; ++dataset) {
        std::vector<half> source(ImageElements + 32, half(7)), weights(Columns * 16 + 32, half(11));
        std::vector<float> output(Rows * 16 + 32, 9.0f), expected = output;
        for (int i = 0; i < ImageElements; ++i) {
            const unsigned pattern = unsigned(i) * 73 % 63488;
            const uint16_t bits = pattern < 31744 ? pattern : (pattern - 31744) | 0x8000;
            source[16 + i] = dataset == 2 ? std::bit_cast<half>(bits) : half((i % 97 - 48) / (dataset ? 37.0f : 16.0f));
        }
        for (int i = 0; i < Columns * 16; ++i)
            weights[16 + i] = half(((i * 17) % 101 - 50) / 43.0f);
        const auto originalSource = source, originalWeights = weights;
        for (int i = 0; i < 256; ++i)
            before.data()[i] = after.data()[i] = half(9);
        for (int i = 0; i < Left::Numel; ++i)
            check.data()[i] = half(13);
        // Independent NHWC-coordinate reference never calls an ISA layout or
        // permutation helper. Input storage itself is NC1HWC0.
        auto expectedPatch = [&](int row, int col) {
            const int tap = (ORDER_PHASE + col / 16 * ORDER_STEP) % FilterTaps;
            const int pixel = ORDER_M + row;
            const int h = pixel / OutputWidth * ORDER_STRIDE + tap / FW * ORDER_DILATION - ORDER_PAD;
            const int w = pixel % OutputWidth * ORDER_STRIDE + tap % FW * ORDER_DILATION - ORDER_PAD;
            const int c1 = ORDER_K / (FilterTaps * 16);
            return h < 0 || h >= Height || w < 0 || w >= Width ?
                       half(-0.25f) :
                       source[16 + ((c1 * Height + h) * Width + w) * 16 + col % 16];
        };
        for (int row = 0; row < Rows; ++row)
            for (int n = 0; n < 16; ++n) {
                float sum = 0;
                for (int k = 0; k < Columns; ++k)
                    sum = std::fma(float(expectedPatch(row, k)), float(weights[16 + k * 16 + n]), sum);
                expected[16 + row * 16 + n] = sum;
            }
        OrderedImage(source.data() + 16, weights.data() + 16, output.data() + 16);
        if (std::memcmp(source.data(), originalSource.data(), source.size() * sizeof(half)) ||
            std::memcmp(weights.data(), originalWeights.data(), weights.size() * sizeof(half)) ||
            std::memcmp(output.data(), expected.data(), output.size() * sizeof(float)))
            return 1;
        for (int i = 0; i < 256; ++i)
            if (before.data()[i] != half(9) || after.data()[i] != half(9))
                return 2;
        for (int row = 0; row < Rows; ++row)
            for (int col = 0; col < Columns; ++col) {
                // ZZ's physical K stride is what the V1 M-repeat jump must
                // reproduce; a contiguous C16 copy would fail for M>16.
                const int offset = ((row / 16) * Taps + col / 16) * 256 + row % 16 * 16 + col % 16;
                if (std::bit_cast<uint16_t>(check.data()[offset]) != std::bit_cast<uint16_t>(expectedPatch(row, col)))
                    return 3;
            }
    }
}
#endif
