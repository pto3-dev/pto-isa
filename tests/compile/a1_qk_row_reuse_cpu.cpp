/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>
#include <pto/pto-inst.hpp>

using Mat = pto::Tile<pto::TileType::Mat, half, 16, 16, pto::BLayout::ColMajor, 16, 16, pto::SLayout::RowMajor, 512>;
using View = pto::Tile<pto::TileType::Mat, half, 16, 16, pto::BLayout::RowMajor, 16, 16, pto::SLayout::ColMajor, 512>;
using Left = pto::Tile<pto::TileType::Left, half, 16, 16, pto::BLayout::RowMajor, 16, 16, pto::SLayout::RowMajor, 512>;
using Right =
    pto::Tile<pto::TileType::Right, half, 16, 16, pto::BLayout::RowMajor, 16, 16, pto::SLayout::ColMajor, 512>;
using Acc = pto::Tile<pto::TileType::Acc, float, 16, 16, pto::BLayout::ColMajor, 16, 16, pto::SLayout::RowMajor, 1024>;
using Vec = pto::Tile<pto::TileType::Vec, float, 16, 16>;
using HVec = pto::Tile<pto::TileType::Vec, half, 16, 16>;
using Global = pto::GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<1, 1, 1, 32, 1>>;

static int check(int dataset)
{
    constexpr int n = 400, k = 32, tile = 16, guards = 16;
    std::vector<half> query(n * k), key(n * k);
    for (int i = 0; i < n * k; ++i) {
        if (dataset == 0) {
            query[i] = half(((i * 11 + i / 32) % 31 - 15) / 16.0f);
            key[i] = half(((i * 7 + i / 32) % 29 - 14) / 16.0f);
        } else {
            const uint16_t qbits =
                uint16_t(0x2c00 + ((i * 73 + i / 32) % 0xc00)) | ((i % 3 == 0) ? uint16_t(0x8000) : uint16_t(0));
            const uint16_t kbits =
                uint16_t(0x2800 + ((i * 97 + i / 32) % 0x1000)) | ((i % 5 < 2) ? uint16_t(0x8000) : uint16_t(0));
            std::memcpy(&query[i], &qbits, sizeof(qbits));
            std::memcpy(&key[i], &kbits, sizeof(kbits));
        }
    }
    auto originalQuery = query, originalKey = key;
    std::vector<float> reference(n * n);
    std::vector<half> referenceHalf(n * n);
    for (int row = 0; row < n; ++row)
        for (int col = 0; col < n; ++col) {
            float expected = 0.0f;
            for (int j = 0; j < k; ++j)
                expected += float(query[row * k + j]) * float(key[col * k + j]);
            reference[row * n + col] = expected;
            referenceHalf[row * n + col] = half(expected);
        }
    for (int cores : {1, 3, 8, 32}) {
        std::vector<float> guarded(n * n + 2 * guards, -9876.0f);
        std::vector<half> narrowed(n * n + 2 * guards, half(-123.0f));
        std::array<int, n / tile> visits{};
        Mat q0, q1, k0, k1;
        View v0, v1;
        Left lhs;
        Right rhs;
        Acc acc;
        Vec values;
        HVec rounded;
        pto::TASSIGN<0>(q0);
        pto::TASSIGN<512>(q1);
        pto::TASSIGN<1024>(k0);
        pto::TASSIGN<1536>(k1);
        pto::TASSIGN<0>(lhs);
        pto::TASSIGN<0>(rhs);
        pto::TASSIGN<0>(acc);
        pto::TASSIGN<0>(values);
        pto::TASSIGN<0>(rounded);
        for (int core = 0; core < cores; ++core) {
            for (int rowTile = core; rowTile < n / tile; rowTile += cores) {
                ++visits[rowTile];
                Global qGlobal0(query.data() + rowTile * tile * k);
                Global qGlobal1(query.data() + rowTile * tile * k + tile);
                pto::TLOAD(q0, qGlobal0);
                pto::TLOAD(q1, qGlobal1);
                for (int colTile = 0; colTile < n / tile; ++colTile) {
                    Global kGlobal0(key.data() + colTile * tile * k);
                    Global kGlobal1(key.data() + colTile * tile * k + tile);
                    pto::TLOAD(k0, kGlobal0);
                    pto::TLOAD(k1, kGlobal1);
                    pto::TRESHAPE(v0, k0);
                    pto::TRESHAPE(v1, k1);
                    pto::TMOV(lhs, q0);
                    pto::TMOV(rhs, v0);
                    pto::TMATMUL(acc, lhs, rhs);
                    pto::TMOV(lhs, q1);
                    pto::TMOV(rhs, v1);
                    pto::TMATMUL_ACC(acc, acc, lhs, rhs);
                    for (int r = 0; r < tile; ++r)
                        for (int c = 0; c < tile; ++c)
                            guarded[guards + (rowTile * tile + r) * n + colTile * tile + c] =
                                acc.data()[pto::GetTileElementOffset<Acc>(r, c)];
                    pto::TMOV<Vec, Acc, pto::AccToVecMode::SingleModeVec0>(values, acc);
                    pto::TCVT(rounded, values, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
                    for (int r = 0; r < tile; ++r)
                        for (int c = 0; c < tile; ++c)
                            narrowed[guards + (rowTile * tile + r) * n + colTile * tile + c] =
                                rounded.data()[pto::GetTileElementOffset<HVec>(r, c)];
                }
            }
        }
        for (int count : visits)
            if (count != 1)
                return 1;
        if (std::memcmp(reference.data(), guarded.data() + guards, reference.size() * sizeof(float)) != 0)
            return 2;
        if (std::memcmp(referenceHalf.data(), narrowed.data() + guards, referenceHalf.size() * sizeof(half)) != 0)
            return 5;
        for (int i = 0; i < guards; ++i)
            if (guarded[i] != -9876.0f || guarded[guards + n * n + i] != -9876.0f || float(narrowed[i]) != -123.0f ||
                float(narrowed[guards + n * n + i]) != -123.0f)
                return 3;
        if (std::memcmp(query.data(), originalQuery.data(), query.size() * sizeof(half)) != 0 ||
            std::memcmp(key.data(), originalKey.data(), key.size() * sizeof(half)) != 0)
            return 4;
    }
    return 0;
}

int main()
{
    for (int dataset : {0, 1}) {
        int result = check(dataset);
        if (result)
            return result;
    }
    return 0;
}
