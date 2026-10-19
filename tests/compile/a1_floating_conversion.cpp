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
#include <cmath>
#include <vector>
#endif
#include <pto/pto-inst.hpp>
using namespace pto;
#if defined(PTO_A1_TEST_WIDEN)
using Source = half;
using Destination = float;
#else
using Source = float;
using Destination = half;
#endif
#if defined(PTO_A1_TEST_NON_FLOAT)
using Converted = int16_t;
#else
using Converted = Destination;
#endif
#if defined(PTO_A1_TEST_LARGE)
constexpr int Cols = 255 * 64 + 16;
#elif defined(PTO_A1_TEST_MULTI_REPEAT)
constexpr int Cols = 80;
#else
constexpr int Cols = 16;
#endif
#if defined(PTO_A1_TEST_DYNAMIC_VALID)
using SrcTile = Tile<TileType::Vec, Source, 1, Cols, BLayout::RowMajor, 1, -1>;
#else
using SrcTile = Tile<TileType::Vec, Source, 1, Cols>;
#endif
#if defined(PTO_A1_TEST_SHAPE_MISMATCH)
using DstTile = Tile<TileType::Vec, Converted, 1, Cols + 16>;
#else
using DstTile = Tile<TileType::Vec, Converted, 1, Cols>;
#endif
using SrcGlobal = GlobalTensor<Source, Shape<1, 1, 1, 1, Cols>, Stride<1, 1, 1, Cols, 1>>;
using DstGlobal = GlobalTensor<Converted, Shape<1, 1, 1, 1, Cols>, Stride<1, 1, 1, Cols, 1>>;

extern "C" __global__ AICORE void A1FloatingConvert(__gm__ Converted* out, __gm__ Source* input)
{
    SrcTile src;
    DstTile dst;
    TASSIGN<0>(src);
#if defined(PTO_A1_TEST_NARROW_ALIAS)
    TASSIGN<0>(dst);
#else
    TASSIGN<Cols * sizeof(Source)>(dst);
#endif
    SrcGlobal gs(input);
    DstGlobal gd(out);
#if defined(PTO_A1_TEST_NEIGHBOR_GUARD) && !defined(PTO_A1_TEST_NARROW_ALIAS)
    using GuardTile = Tile<TileType::Vec, Converted, 1, 16>;
    using GuardGlobal = GlobalTensor<Converted, Shape<1, 1, 1, 1, 16>, Stride<1, 1, 1, 16, 1>>;
    GuardTile guard;
    GuardGlobal gg(out + Cols);
    TASSIGN<Cols * (sizeof(Source) + sizeof(Converted))>(guard);
    TLOAD(guard, gg);
#endif
#if defined(__CPU_SIM)
    TLOAD(src, gs);
    TCVT(dst, src, RoundMode::CAST_NONE, SaturationMode::OFF);
    TSTORE(gd, dst);
#else
    Event<Op::TLOAD, Op::TCVT> loaded = TLOAD(src, gs);
    Event<Op::TCVT, Op::TSTORE_VEC> converted = TCVT(dst, src, RoundMode::CAST_NONE, SaturationMode::OFF, loaded);
    TSTORE(gd, dst, converted);
#endif
#if defined(PTO_A1_TEST_NEIGHBOR_GUARD) && !defined(PTO_A1_TEST_NARROW_ALIAS)
    TSTORE(gg, guard);
#endif
}

#if defined(__CPU_SIM)
int main()
{
    std::vector<Source> input(Cols);
#if defined(PTO_A1_TEST_NEIGHBOR_GUARD)
    std::vector<Converted> output(Cols + 16, Converted(123));
#else
    std::vector<Converted> output(Cols);
#endif
    for (int i = 0; i < Cols; ++i)
        input[i] = Source((i % 99 - 49) * 0.125f);
    input[0] = Source(-0.0f);
    A1FloatingConvert(output.data(), input.data());
    for (int i = 0; i < Cols; ++i) {
        const Converted expected = Converted(input[i]);
        if (output[i] != expected || (expected == 0 && std::signbit(float(output[i])) != std::signbit(float(expected))))
            return 1;
    }
#if defined(PTO_A1_TEST_NEIGHBOR_GUARD)
    for (int i = Cols; i < Cols + 16; ++i)
        if (output[i] != Converted(123))
            return 1;
#endif
    return 0;
}
#endif
