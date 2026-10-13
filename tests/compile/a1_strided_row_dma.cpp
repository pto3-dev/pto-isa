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
#if defined(__CPU_SIM)
#include <vector>
#endif
#include <pto/pto-inst.hpp>

#if defined(PTO_A1_TEST_FP32)
using Data = float;
#else
using Data = half;
#endif
#if defined(PTO_A1_TEST_BURST_LIMIT)
constexpr int Rows = 4096;
constexpr int Cols = 32 / sizeof(Data);
constexpr int RowStride = 2 * Cols;
#elif defined(PTO_A1_TEST_WIDE_GAP)
constexpr int Rows = 8;
constexpr int Cols = 32 / sizeof(Data);
constexpr int RowStride = Cols + 65536 * (32 / sizeof(Data));
#else
constexpr int Rows = 16;
constexpr int Cols = 128;
constexpr int RowStride = 256;
#endif
using TileData = pto::Tile<pto::TileType::Vec, Data, Rows, Cols>;
using GlobalData = pto::GlobalTensor<Data, pto::Shape<1, 1, 1, Rows, Cols>, pto::Stride<1, 1, 1, RowStride, 1>>;

extern "C" __global__ AICORE void A1StridedCopy(__gm__ Data* out, __gm__ Data* src, __gm__ Data*)
{
    TileData tile;
    pto::TASSIGN<0>(tile);
    GlobalData input(src), output(out);
#if defined(__CPU_SIM)
    pto::TLOAD(tile, input);
    pto::TSTORE(output, tile);
#else
    pto::Event<pto::Op::TLOAD, pto::Op::TSTORE_VEC> loaded = pto::TLOAD(tile, input);
    pto::TSTORE(output, tile, loaded);
#endif
}

#if defined(__CPU_SIM)
int main()
{
    std::vector<Data> source(Rows * RowStride), output(Rows * RowStride, -123);
    for (int i = 0; i < Rows * RowStride; ++i)
        source[i] = i % 17 - 8;
    A1StridedCopy(output.data(), source.data(), nullptr);
    for (int row = 0; row < Rows; ++row)
        for (int col = 0; col < RowStride; ++col)
            if (output[row * RowStride + col] != (col < Cols ? source[row * RowStride + col] : Data(-123)))
                return 1;
    return 0;
}
#endif
