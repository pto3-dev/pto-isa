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
#ifdef __CPU_SIM
#include <cmath>
#include <vector>
#else
#include "tensor.h"
#endif
#ifndef TEST_COUNT
#define TEST_COUNT 80
#endif
#ifndef TEST_ALIAS
#define TEST_ALIAS 0
#endif
using Tile = pto::Tile<pto::TileType::Vec, float, 1, TEST_COUNT>;
using Global = pto::GlobalTensor<float, pto::Shape<1, 1, 1, 1, TEST_COUNT>, pto::Stride<1, 1, 1, TEST_COUNT, 1>>;
using Guard = pto::Tile<pto::TileType::Vec, float, 1, 16>;
using GuardGlobal = pto::GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<1, 1, 1, 16, 1>>;
AICORE void a1_mula(__gm__ float* a, __gm__ float* b, __gm__ float* initial, __gm__ float* output)
{
    Tile x, y, dst;
    Guard before, after;
    pto::TASSIGN<TEST_ALIAS == 1 ? 65536 : 0>(x);
    pto::TASSIGN<TEST_ALIAS == 2 ? 65536 : 32768>(y);
    pto::TASSIGN<65536>(dst);
    pto::TASSIGN<65536 - 64>(before);
    pto::TASSIGN<65536 + TEST_COUNT * 4>(after);
    Global ga(a), gb(b), gd(initial), go(output);
    GuardGlobal prefix(output - 16), suffix(output + TEST_COUNT);
    pto::TLOAD(x, ga);
    pto::TLOAD(y, gb);
    pto::TLOAD(before, prefix);
#ifdef __CPU_SIM
    pto::TLOAD(dst, gd);
    pto::TLOAD(after, suffix);
    pto::TMULA(dst, x, y);
    pto::TSTORE(go, dst);
#else
    pto::TLOAD(dst, gd);
    pto::Event<pto::Op::TLOAD, pto::Op::TMULA> guards_loaded = pto::TLOAD(after, suffix);
    pto::Event<pto::Op::TMULA, pto::Op::TSTORE_VEC> computed = pto::TMULA(dst, x, y, guards_loaded);
    pto::TSTORE(go, dst, computed);
#endif
    pto::TSTORE(prefix, before);
    pto::TSTORE(suffix, after);
}
#ifdef __CPU_SIM
int main()
{
    std::vector<float> a(TEST_COUNT), b(TEST_COUNT), d(TEST_COUNT), out(TEST_COUNT + 32, 7.0f);
    for (unsigned i = 0; i < TEST_COUNT; ++i) {
        a[i] = 1.0f + float(i % 11 + 1) / 8388608.0f;
        b[i] = 1.0f + float(i % 13 + 1) / 8388608.0f;
        d[i] = -1.0f;
        if constexpr (TEST_ALIAS == 1)
            a[i] = d[i];
        if constexpr (TEST_ALIAS == 2)
            b[i] = d[i];
    }
    a1_mula(a.data(), b.data(), d.data(), out.data() + 16);
    for (unsigned i = 0; i < TEST_COUNT; ++i)
        if (out[i + 16] != std::fma(a[i], b[i], d[i]))
            return 1;
    for (unsigned i = 0; i < 16; ++i)
        if (out[i] != 7.0f || out[16 + TEST_COUNT + i] != 7.0f)
            return 2;
    return 0;
}
#else
extern "C" __global__ AICORE void A1Mula(__gm__ int64_t* args)
{
    auto* a = reinterpret_cast<__gm__ Tensor*>(args[0]);
    auto* b = reinterpret_cast<__gm__ Tensor*>(args[1]);
    auto* d = reinterpret_cast<__gm__ Tensor*>(args[2]);
    auto* out = reinterpret_cast<__gm__ Tensor*>(args[3]);
    a1_mula(
        reinterpret_cast<__gm__ float*>(a->buffer.addr) + 16, reinterpret_cast<__gm__ float*>(b->buffer.addr) + 16,
        reinterpret_cast<__gm__ float*>(d->buffer.addr) + 16, reinterpret_cast<__gm__ float*>(out->buffer.addr) + 16);
}
#endif
