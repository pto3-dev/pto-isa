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
#include <vector>
#endif
#ifndef PTO_RESHAPE_ROWS
#define PTO_RESHAPE_ROWS 160
#endif
#ifndef PTO_RESHAPE_COLS
#define PTO_RESHAPE_COLS 16
#endif
#ifndef PTO_RESHAPE_TARGET_COLS
#define PTO_RESHAPE_TARGET_COLS (PTO_RESHAPE_ROWS * PTO_RESHAPE_COLS)
#endif
#ifdef PTO_RESHAPE_FP16
using Element = half;
#else
using Element = float;
#endif
constexpr unsigned Rows = PTO_RESHAPE_ROWS, Cols = PTO_RESHAPE_COLS;
constexpr unsigned Count = Rows * Cols, TargetCols = PTO_RESHAPE_TARGET_COLS;
constexpr unsigned Span = Count * sizeof(Element), Offset = 32;
using Source = pto::Tile<pto::TileType::Vec, Element, Rows, Cols>;
using View = pto::Tile<pto::TileType::Vec, Element, Count / TargetCols, TargetCols>;
using SourceGlobal = pto::GlobalTensor<Element, pto::Shape<1, 1, 1, Rows, Cols>, pto::Stride<1, 1, 1, Cols, 1>>;
using ViewGlobal = pto::GlobalTensor<
    Element, pto::Shape<1, 1, 1, Count / TargetCols, TargetCols>, pto::Stride<1, 1, 1, TargetCols, 1>>;
extern "C" __global__ AICORE void A1Reshape(__gm__ Element* output, __gm__ Element* input)
{
    Source source;
    View view;
    pto::TASSIGN<Offset>(source);
    // A different assigned destination must be rebound, not copied into.
    pto::TASSIGN<0>(view);
    SourceGlobal original(input), modified(output + Count);
    ViewGlobal flattened(output);
#ifdef __CPU_SIM
    pto::TLOAD(source, original);
    pto::TRESHAPE(view, source);
    pto::TSTORE(flattened, view);
    pto::TADD(view, view, view);
    pto::TSTORE(modified, source);
#else
    pto::Event<pto::Op::TLOAD, pto::Op::TSTORE_VEC> loaded = pto::TLOAD(source, original);
    pto::TRESHAPE(view, source);
    pto::Event<pto::Op::TSTORE_VEC, pto::Op::TADD> saved = pto::TSTORE(flattened, view, loaded);
    // Writing through the view must change the original tile as well.
    pto::Event<pto::Op::TADD, pto::Op::TSTORE_VEC> changed = pto::TADD(view, view, view, saved);
    pto::TSTORE(modified, source, changed);
#endif
}
#ifdef __CPU_SIM
int main()
{
    std::vector<Element> input(Count), output(2 * Count + 32, Element(7));
    for (unsigned i = 0; i < Count; ++i)
        input[i] = Element((int(i % 257) - 128) / 8.0f);
    A1Reshape(output.data() + 16, input.data());
    for (unsigned i = 0; i < Count; ++i)
        if (output[16 + i] != input[i] || output[16 + Count + i] != Element(float(input[i]) * 2))
            return 1;
    for (unsigned i = 0; i < 16; ++i)
        if (output[i] != Element(7) || output[16 + 2 * Count + i] != Element(7))
            return 2;
    return 0;
}
#endif
