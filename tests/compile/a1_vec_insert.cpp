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
#ifndef TEST_TYPE
#define TEST_TYPE float
#endif
#ifndef TEST_SRC_ROWS
#define TEST_SRC_ROWS 128
#endif
#ifndef TEST_DST_ROWS
#define TEST_DST_ROWS 128
#endif
#ifndef TEST_ROW
#define TEST_ROW 0
#endif
#ifndef TEST_COL
#define TEST_COL 0
#endif
using T = TEST_TYPE;
using S = pto::Tile<pto::TileType::Vec, T, TEST_SRC_ROWS, 16>;
using D = pto::Tile<pto::TileType::Vec, T, TEST_DST_ROWS, 32>;
using G = pto::Tile<pto::TileType::Vec, T, 1, 16>;
using SG = pto::GlobalTensor<T, pto::Shape<1, 1, 1, TEST_SRC_ROWS, 16>, pto::Stride<1, 1, 1, 16, 1>>;
using DG = pto::GlobalTensor<T, pto::Shape<1, 1, 1, TEST_DST_ROWS, 32>, pto::Stride<1, 1, 1, 32, 1>>;
using GG = pto::GlobalTensor<T, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<1, 1, 1, 16, 1>>;
extern "C" __global__ AICORE void A1Insert(__gm__ int64_t* args)
{
    auto* in = reinterpret_cast<__gm__ Tensor*>(args[0]);
    auto* output = reinterpret_cast<__gm__ Tensor*>(args[1]);
    auto* ip = reinterpret_cast<__gm__ T*>(in->buffer.addr);
    auto* op = reinterpret_cast<__gm__ T*>(output->buffer.addr) + 16;
    S src;
    D dst;
    G before, after;
    pto::TASSIGN<0>(src);
    pto::TASSIGN<32768>(dst);
    pto::TASSIGN<32768 - 16 * sizeof(T)>(before);
    pto::TASSIGN<32768 + TEST_DST_ROWS * 32 * sizeof(T)>(after);
    SG gi(ip);
    DG go(op);
    GG gb(op - 16), ga(op + TEST_DST_ROWS * 32);
    pto::TLOAD(before, gb);
    pto::TLOAD(after, ga);
    pto::TLOAD(dst, go);
    pto::Event<pto::Op::TLOAD, pto::Op::TINSERT_V2V> loaded = pto::TLOAD(src, gi);
    pto::Event<pto::Op::TINSERT_V2V, pto::Op::TSTORE_VEC> inserted = pto::TINSERT(dst, src, TEST_ROW, TEST_COL, loaded);
    pto::TSTORE(go, dst, inserted);
    pto::TSTORE(gb, before);
    pto::TSTORE(ga, after);
}
