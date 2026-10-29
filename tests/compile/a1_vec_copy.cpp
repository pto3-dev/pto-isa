/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <pto/pto-inst.hpp>
#include "tensor.h"
#ifndef TEST_COUNT
#define TEST_COUNT 80
#endif
#ifndef TEST_ALIAS
#define TEST_ALIAS 0
#endif
#ifndef TEST_TYPE
#define TEST_TYPE float
#endif
using T = TEST_TYPE;
using Tile = pto::Tile<pto::TileType::Vec, T, 1, TEST_COUNT>;
using Guard = pto::Tile<pto::TileType::Vec, T, 1, 16>;
using Global = pto::GlobalTensor<T, pto::Shape<1, 1, 1, 1, TEST_COUNT>, pto::Stride<1, 1, 1, TEST_COUNT, 1>>;
using GuardGlobal = pto::GlobalTensor<T, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<1, 1, 1, 16, 1>>;
extern "C" __global__ AICORE void A1VecCopy(__gm__ int64_t* args)
{
    auto* in = reinterpret_cast<__gm__ Tensor*>(args[0]);
    auto* out = reinterpret_cast<__gm__ Tensor*>(args[1]);
    auto* src_ptr = reinterpret_cast<__gm__ T*>(in->buffer.addr) + 16;
    auto* dst_ptr = reinterpret_cast<__gm__ T*>(out->buffer.addr) + 16;
    Tile src, dst;
    Guard before, after;
    pto::TASSIGN<0>(src);
    pto::TASSIGN<TEST_ALIAS ? 0 : 32768>(dst);
    pto::TASSIGN<32768 - 16 * sizeof(T)>(before);
    pto::TASSIGN<32768 + TEST_COUNT * sizeof(T)>(after);
    Global gs(src_ptr), gd(dst_ptr);
    GuardGlobal gb(dst_ptr - 16), ga(dst_ptr + TEST_COUNT);
    pto::TLOAD(before, gb);
    pto::TLOAD(after, ga);
    pto::Event<pto::Op::TLOAD, pto::Op::TMOV_V2V> loaded = pto::TLOAD(src, gs);
    pto::Event<pto::Op::TMOV_V2V, pto::Op::TSTORE_VEC> copied = pto::TMOV(dst, src, loaded);
    pto::TSTORE(gd, dst, copied);
    pto::TSTORE(gb, before);
    pto::TSTORE(ga, after);
}
