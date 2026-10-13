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

#if defined(PTO_A1_TEST_PARTIAL_REPEAT)
constexpr int A1Rows = 1;
constexpr int A1Cols = 8;
#elif defined(PTO_A1_TEST_LARGE)
constexpr int A1Rows = 128;
constexpr int A1Cols = 128;
#else
constexpr int A1Rows = 16;
constexpr int A1Cols = 128;
#endif
using A1Element = float;
constexpr int A1Numel = A1Rows * A1Cols;

#if defined(PTO_A1_TEST_DYNAMIC_VALID)
using A1Tile = pto::Tile<pto::TileType::Vec, A1Element, A1Rows, A1Cols, pto::BLayout::RowMajor, -1, -1>;
#else
using A1Tile = pto::Tile<pto::TileType::Vec, A1Element, A1Rows, A1Cols>;
#endif
#if defined(PTO_A1_TEST_STRIDED_DMA)
using A1Stride = pto::Stride<1, 1, 1, A1Cols + 1, 1>;
#else
using A1Stride = pto::Stride<1, 1, 1, A1Cols, 1>;
#endif
using A1Global = pto::GlobalTensor<A1Element, pto::Shape<1, 1, 1, A1Rows, A1Cols>, A1Stride>;

extern "C" __global__ AICORE void A1Fp32Add(__gm__ A1Element* out, __gm__ A1Element* lhs, __gm__ A1Element* rhs)
{
#if defined(PTO_A1_TEST_DYNAMIC_VALID)
    A1Tile a(A1Rows, A1Cols), b(A1Rows, A1Cols), c(A1Rows, A1Cols);
#else
    A1Tile a, b, c;
#endif
    pto::TASSIGN<0>(a);
    pto::TASSIGN<A1Numel * sizeof(A1Element)>(b);
#if defined(PTO_A1_TEST_LARGE)
    pto::TASSIGN<0>(c);
#else
    pto::TASSIGN<2 * A1Numel * sizeof(A1Element)>(c);
#endif
    A1Global ga(lhs), gb(rhs), gc(out);
    pto::TLOAD(a, ga);
#if defined(__CPU_SIM)
    pto::TLOAD(b, gb);
    pto::TADD(c, a, b);
    pto::TSTORE(gc, c);
#else
    pto::Event<pto::Op::TLOAD, pto::Op::TADD> loaded = pto::TLOAD(b, gb);
#if defined(PTO_A1_TEST_MIXED_ADD)
    pto::Tile<pto::TileType::Vec, half, A1Rows, A1Cols> other;
    pto::TASSIGN<65536>(other);
    pto::Event<pto::Op::TADD, pto::Op::TSTORE_VEC> added = pto::TADD(c, a, other, loaded);
#else
    pto::Event<pto::Op::TADD, pto::Op::TSTORE_VEC> added = pto::TADD(c, a, b, loaded);
#endif
#if defined(PTO_A1_TEST_ATOMIC_STORE)
    pto::TSTORE<A1Tile, A1Global, pto::AtomicType::AtomicAdd>(gc, c, added);
#else
    pto::TSTORE(gc, c, added);
#endif
#endif
}

#if defined(__CPU_SIM)
int main()
{
    std::vector<A1Element> lhs(A1Numel), rhs(A1Numel), out(A1Numel);
    for (int i = 0; i < A1Numel; ++i) {
        lhs[i] = i % 17 - 8;
        rhs[i] = (i * 3) % 17 - 8;
    }
    A1Fp32Add(out.data(), lhs.data(), rhs.data());
    for (int i = 0; i < A1Numel; ++i) {
        if (out[i] != static_cast<A1Element>(lhs[i] + rhs[i]))
            return 1;
    }
    return 0;
}
#endif
