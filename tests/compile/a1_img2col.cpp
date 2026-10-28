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
#include <vector>
#endif

#ifndef PTO_A1_IMG_FILTER
#define PTO_A1_IMG_FILTER 3
#endif
#ifndef PTO_A1_IMG_STRIDE
#define PTO_A1_IMG_STRIDE 1
#endif
#ifndef PTO_A1_IMG_PAD
#define PTO_A1_IMG_PAD 1
#endif
#ifndef PTO_A1_IMG_K
#define PTO_A1_IMG_K 144
#endif
#ifndef PTO_A1_IMG_POS_M
#define PTO_A1_IMG_POS_M 0
#endif
#ifndef PTO_A1_IMG_POS_K
#define PTO_A1_IMG_POS_K 0
#endif
#ifndef PTO_A1_IMG_M
#define PTO_A1_IMG_M 32
#endif
#ifndef PTO_A1_IMG_DILATION
#define PTO_A1_IMG_DILATION 1
#endif
#ifndef PTO_A1_IMG_PAD_VALUE
#define PTO_A1_IMG_PAD_VALUE 0.0f
#endif
constexpr int Height = 8, Width = 8, C1 = 2, Rows = PTO_A1_IMG_M, Cols = PTO_A1_IMG_K;
constexpr int ImageElements = C1 * Height * Width * 16;
constexpr int Filter = PTO_A1_IMG_FILTER, Stride = PTO_A1_IMG_STRIDE, Pad = PTO_A1_IMG_PAD;
constexpr int PosM = PTO_A1_IMG_POS_M, PosK = PTO_A1_IMG_POS_K;
using Image = pto::ConvTile<
    pto::TileType::Mat, half, ImageElements, pto::Layout::NC1HWC0, pto::ConvTileShape<1, C1, Height, Width, 16>>;
// dav-m200 writes M-major ZZ fractals. A5's NZ Left contract is different;
// a single-K16 identity case cannot distinguish those physical layouts.
using Left =
    pto::Tile<pto::TileType::Left, half, Rows, Cols, pto::BLayout::RowMajor, Rows, Cols, pto::SLayout::RowMajor, 512>;
using Input = pto::GlobalTensor<
    half, pto::Shape<1, C1, Height, Width, 16>, pto::Stride<ImageElements, Height * Width * 16, Width * 16, 16, 1>,
    pto::Layout::NC1HWC0>;

extern "C" __global__ AICORE void A1Img2col(__gm__ float* output, __gm__ half* input, __gm__ half* identity)
{
    Image image;
    Left left;
#if defined(__CPU_SIM)
    image.data() = input;
#else
    pto::TASSIGN<0>(image);
    Input global(input);
    pto::TLOAD(image, global);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
    pto::TASSIGN<0>(left);
    image.SetFmapH(Height);
    image.SetFmapW(Width);
    image.SetFilterH(Filter);
    image.SetFilterW(Filter);
    image.SetDilationH(PTO_A1_IMG_DILATION);
    image.SetDilationW(PTO_A1_IMG_DILATION);
    image.SetPadValue(half(PTO_A1_IMG_PAD_VALUE));
    image.SetStrideH(Stride);
    image.SetStrideW(Stride);
    image.SetChannelSize(C1 * 16);
    const uint8_t pads[4] = {Pad, Pad, Pad, Pad};
    image.SetPadListArray(pads);
#if defined(PTO_A1_IMG_MANUAL)
    pto::SETFMATRIX<Image, pto::SetFmatrixMode::FMATRIX_A_AUTO>(image);
    pto::TIMG2COL(left, image, PosM, PosK);
#else
    pto::TIMG2COL<Left, Image, pto::SetFmatrixMode::FMATRIX_A_AUTO>(left, image, PosM, PosK);
#endif
#if defined(__CPU_SIM)
    for (int row = 0; row < Rows; ++row)
        for (int col = 0; col < Cols; ++col)
            output[row * Cols + col] = float(left.data()[pto::GetTileElementOffset<Left>(row, col)]);
#else
    // L0A has no direct GM readback. Read every M16/K16 fragment through an
    // exact FP16 identity MAD; random finite data exposes layout mistakes.
    using Mat =
        pto::Tile<pto::TileType::Mat, half, 16, 16, pto::BLayout::RowMajor, 16, 16, pto::SLayout::RowMajor, 512>;
    using NarrowLeft = pto::TileLeft<half, 16, 16>;
    using Right = pto::TileRight<half, 16, 16>;
    using Acc = pto::TileAcc<float, 16, 16>;
    using Vec = pto::Tile<pto::TileType::Vec, float, 16, 16>;
    using Identity = pto::GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<1, 1, 1, 16, 1>>;
    using Output = pto::GlobalTensor<float, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<1, 1, 1, Cols, 1>>;
    Mat mat;
    Right right;
    Acc acc;
    Vec vec;
    pto::TASSIGN<ImageElements * sizeof(half)>(mat);
    pto::TASSIGN<0>(right);
    pto::TASSIGN<0>(acc);
    pto::TASSIGN<0>(vec);
    Identity global_identity(identity);
    pto::TLOAD(mat, global_identity);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    pto::TMOV(right, mat);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    for (int mf = 0; mf < Rows / 16; ++mf) {
        for (int kf = 0; kf < Cols / 16; ++kf) {
            NarrowLeft fragment;
            pto::TASSIGN(fragment, reinterpret_cast<uint64_t>(left.data() + (mf * (Cols / 16) + kf) * 256));
            pto::TMATMUL(acc, fragment, right);
            set_flag(PIPE_M, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_V, EVENT_ID0);
            pto::TMOV<Vec, Acc, pto::AccToVecMode::SingleModeVec0>(vec, acc);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            Output destination(output + mf * 16 * Cols + kf * 16);
            pto::TSTORE(destination, vec);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            set_flag(PIPE_V, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_M, EVENT_ID0);
        }
    }
#endif
}

#if defined(__CPU_SIM)
int main()
{
    std::vector<half> input(ImageElements), identity(256);
    std::vector<float> output(Rows * Cols + 16, 7.0f);
    for (int i = 0; i < ImageElements; ++i)
        input[i] = half((i % 127 - 63) * 0.125f);
    A1Img2col(output.data(), input.data(), identity.data());
    constexpr int outWidth = (Width + 2 * Pad - PTO_A1_IMG_DILATION * (Filter - 1) - 1) / Stride + 1;
    for (int r = 0; r < Rows; ++r) {
        for (int c = 0; c < Cols; ++c) {
            const int m = PosM + r, k = PosK + c;
            const int channel = k / (Filter * Filter * 16), pos = (k / 16) % (Filter * Filter);
            const int inputRow = m / outWidth * Stride - Pad + pos / Filter * PTO_A1_IMG_DILATION;
            const int inputCol = m % outWidth * Stride - Pad + pos % Filter * PTO_A1_IMG_DILATION;
            const float expected = inputRow < 0 || inputRow >= Height || inputCol < 0 || inputCol >= Width ?
                                       float(half(PTO_A1_IMG_PAD_VALUE)) :
                                       float(input[((channel * Height + inputRow) * Width + inputCol) * 16 + k % 16]);
            if (output[r * Cols + c] != expected)
                return 1;
        }
    }
    for (int i = Rows * Cols; i < Rows * Cols + 16; ++i)
        if (output[i] != 7.0f)
            return 2;
    return 0;
}
#endif
