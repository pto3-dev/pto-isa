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

#ifdef A1_TRANSPOSE_INVALID_SHAPE
constexpr unsigned kRows = 32;
#else
constexpr unsigned kRows = 16;
#endif
#ifdef A1_TRANSPOSE_INVALID_VALID_SHAPE
constexpr unsigned kValidRows = 8;
#else
constexpr unsigned kValidRows = kRows;
#endif
#ifdef A1_TRANSPOSE_INVALID_TYPE
using SourceType = float;
#else
using SourceType = half;
#endif
using Source = pto::Tile<
    pto::TileType::Mat, SourceType, kRows, 16, pto::BLayout::RowMajor, kValidRows, 16, pto::SLayout::ColMajor, 512>;
#ifdef A1_TRANSPOSE_INVALID_DEST
using Operand =
    pto::Tile<pto::TileType::Left, half, kRows, 16, pto::BLayout::ColMajor, kRows, 16, pto::SLayout::RowMajor, 512>;
#else
using Operand =
    pto::Tile<pto::TileType::Right, half, kRows, 16, pto::BLayout::RowMajor, kRows, 16, pto::SLayout::ColMajor, 512>;
#endif

extern "C" AICORE void test_transposed_right()
{
    Source source;
    Operand operand;
    pto::TASSIGN<0>(source);
    pto::TASSIGN<0>(operand);
    pto::TMOV(operand, source);
}
