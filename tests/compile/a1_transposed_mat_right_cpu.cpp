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

using Original =
    pto::Tile<pto::TileType::Mat, half, 16, 16, pto::BLayout::ColMajor, 16, 16, pto::SLayout::RowMajor, 512>;
using Transposed =
    pto::Tile<pto::TileType::Mat, half, 16, 16, pto::BLayout::RowMajor, 16, 16, pto::SLayout::ColMajor, 512>;
using Right =
    pto::Tile<pto::TileType::Right, half, 16, 16, pto::BLayout::RowMajor, 16, 16, pto::SLayout::ColMajor, 512>;
using Global = pto::GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<1, 1, 1, 16, 1>>;

int main()
{
    half input[256];
    for (int i = 0; i < 256; ++i)
        input[i] = half(i - 128);
    Original original;
    Transposed view;
    Right right;
    pto::TASSIGN<0>(original);
    pto::TASSIGN<0>(right);
    Global src(input);
    pto::TLOAD(original, src);
    pto::TRESHAPE(view, original);
    pto::TMOV(right, view);
    for (int row = 0; row < 16; ++row)
        for (int column = 0; column < 16; ++column)
            if (right.data()[pto::GetTileElementOffset<Right>(row, column)] != input[column * 16 + row])
                return 1;
    return 0;
}
