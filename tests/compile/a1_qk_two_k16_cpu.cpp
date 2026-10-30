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
#include <cstring>
#include <pto/pto-inst.hpp>

using Mat = pto::Tile<pto::TileType::Mat, half, 16, 16, pto::BLayout::ColMajor, 16, 16, pto::SLayout::RowMajor, 512>;
using View = pto::Tile<pto::TileType::Mat, half, 16, 16, pto::BLayout::RowMajor, 16, 16, pto::SLayout::ColMajor, 512>;
using Left = pto::Tile<pto::TileType::Left, half, 16, 16, pto::BLayout::RowMajor, 16, 16, pto::SLayout::RowMajor, 512>;
using Right =
    pto::Tile<pto::TileType::Right, half, 16, 16, pto::BLayout::RowMajor, 16, 16, pto::SLayout::ColMajor, 512>;
using Acc = pto::Tile<pto::TileType::Acc, float, 16, 16, pto::BLayout::ColMajor, 16, 16, pto::SLayout::RowMajor, 1024>;
using Global = pto::GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<1, 1, 1, 32, 1>>;

int main()
{
    half q[16 * 32], k[16 * 32];
    for (int i = 0; i < 16 * 32; ++i) {
        q[i] = half(((i * 11) % 31 - 15) / 16.0f);
        k[i] = half(((i * 7) % 29 - 14) / 16.0f);
    }
    Mat query, key;
    View keyTranspose;
    Left lhs;
    Right rhs;
    Acc acc;
    pto::TASSIGN<0>(query);
    pto::TASSIGN<512>(key);
    pto::TASSIGN<0>(lhs);
    pto::TASSIGN<0>(rhs);
    pto::TASSIGN<0>(acc);
    for (int tile = 0; tile < 2; ++tile) {
        Global qGlobal(q + tile * 16), kGlobal(k + tile * 16);
        pto::TLOAD(query, qGlobal);
        pto::TLOAD(key, kGlobal);
        pto::TRESHAPE(keyTranspose, key);
        pto::TMOV(lhs, query);
        pto::TMOV(rhs, keyTranspose);
        if (tile == 0)
            pto::TMATMUL(acc, lhs, rhs);
        else
            pto::TMATMUL_ACC(acc, acc, lhs, rhs);
    }
    for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 16; ++col) {
            float expected = 0.0f;
            for (int j = 0; j < 32; ++j)
                expected += float(q[row * 32 + j]) * float(k[col * 32 + j]);
            float actual = acc.data()[pto::GetTileElementOffset<Acc>(row, col)];
            if (std::memcmp(&actual, &expected, sizeof(float)) != 0)
                return 1;
        }
    }
    return 0;
}
