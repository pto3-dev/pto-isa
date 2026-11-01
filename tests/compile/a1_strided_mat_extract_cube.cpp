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
#ifdef __CPU_SIM
#include <bit>
#include <iostream>
#include <vector>
#endif
#include <pto/pto-inst.hpp>
#ifndef PTO_TEST_EXTRACT_ROW_MAJOR
#define PTO_TEST_EXTRACT_ROW_MAJOR 0
#endif
using namespace pto;
constexpr unsigned M = 16, K = 144, N = 32, residentRows = 2304;
constexpr BLayout sourceLayout = PTO_TEST_EXTRACT_ROW_MAJOR ? BLayout::RowMajor : BLayout::ColMajor;
using Resident = Tile<TileType::Mat, half, residentRows, N, sourceLayout, residentRows, N, SLayout::RowMajor, 512>;
using LeftMat = Tile<TileType::Mat, half, M, K, BLayout::RowMajor, M, K, SLayout::RowMajor, 512>;
using Left = Tile<TileType::Left, half, M, K, BLayout::RowMajor, M, K, SLayout::RowMajor, 512>;
using Right = TileRight<half, K, N>;
using Acc = TileAcc<float, M, N>;
using Weights = GlobalTensor<half, Shape<1, 1, 1, residentRows, N>, Stride<1, 1, 1, N, 1>>;
using Input = GlobalTensor<half, Shape<1, 1, 1, M, K>, Stride<1, 1, 1, K, 1>>;

extern "C" __global__ AICORE void A1BlockExtractCube(__gm__ half* input, __gm__ half* weights)
{
    Resident resident;
    LeftMat leftMat;
    Left left;
    Right bank0, bank1;
    Acc acc;
    TASSIGN<512>(resident);
    TASSIGN<524288>(leftMat);
    TASSIGN<0>(left);
    TASSIGN<0>(bank0);
    TASSIGN<16384>(bank1);
    TASSIGN<0>(acc);
    Input ga(input);
    Weights gb(weights);
    TLOAD(leftMat, ga);
#ifdef __CPU_SIM
    TLOAD(resident, gb);
    TMOV(left, leftMat);
    TEXTRACT<4>(bank0, resident, 0, 0);
    TEXTRACT<4>(bank1, resident, 16, 0);
    TMATMUL(acc, left, bank0);
    TMATMUL_ACC(acc, acc, left, bank1);
    TEXTRACT<4>(bank0, resident, 32, 0);
    TMATMUL_ACC(acc, acc, left, bank0);
    TEXTRACT<4>(bank1, resident, 48, 0);
    TMATMUL_ACC(acc, acc, left, bank1);
    TEXTRACT<4>(bank0, resident, 576, 0);
    TMATMUL_ACC(acc, acc, left, bank0);
#else
    Event<Op::TLOAD, Op::TMOV_M2L, false, EVENT_ID0> loaded = TLOAD(resident, gb);
    Event<Op::TMOV_M2L, Op::TMATMUL, false, EVENT_ID2> leftReady = TMOV(left, leftMat, loaded);
    Event<Op::TEXTRACT_M2LR, Op::TMATMUL, false, EVENT_ID0> ready0 = TEXTRACT<4>(bank0, resident, 0, 0);
    Event<Op::TEXTRACT_M2LR, Op::TMATMUL, false, EVENT_ID1> ready1 = TEXTRACT<4>(bank1, resident, 16, 0);
    Event<Op::TMATMUL, Op::TEXTRACT_M2LR, false, EVENT_ID0> free0 = TMATMUL(acc, left, bank0, leftReady, ready0);
    Event<Op::TMATMUL, Op::TEXTRACT_M2LR, false, EVENT_ID1> free1 = TMATMUL_ACC(acc, acc, left, bank1, ready1);
    Event<Op::TEXTRACT_M2LR, Op::TMATMUL, false, EVENT_ID0> ready2 = TEXTRACT<4>(bank0, resident, 32, 0, free0);
    Event<Op::TMATMUL, Op::TEXTRACT_M2LR, false, EVENT_ID0> free2 = TMATMUL_ACC(acc, acc, left, bank0, ready2);
    Event<Op::TEXTRACT_M2LR, Op::TMATMUL, false, EVENT_ID1> ready3 = TEXTRACT<4>(bank1, resident, 48, 0, free1);
    TMATMUL_ACC(acc, acc, left, bank1, ready3);
    Event<Op::TEXTRACT_M2LR, Op::TMATMUL, false, EVENT_ID0> ready4 = TEXTRACT<4>(bank0, resident, 576, 0, free2);
    TMATMUL_ACC(acc, acc, left, bank0, ready4);
    pipe_barrier(PIPE_ALL);
#endif
}
#ifdef __CPU_SIM
int main()
{
    for (int dataset = 0; dataset < 3; ++dataset) {
        std::vector<half> input(M * K), weights(residentRows * N);
        for (unsigned i = 0; i < input.size(); ++i)
            input[i] = int((i * 3 + dataset) % 5) - 2;
        for (unsigned i = 0; i < weights.size(); ++i)
            weights[i] = int((i * 5 + dataset * 3) % 7) - 3;
        const auto originalInput = input, originalWeights = weights;
        A1BlockExtractCube(input.data(), weights.data());
        if (input != originalInput || weights != originalWeights)
            return 1;
        Acc acc;
        TASSIGN<0>(acc);
        for (unsigned row = 0; row < M; ++row) {
            for (unsigned col = 0; col < N; ++col) {
                float expected = 0;
                // Integer products and total sum fit exactly in FP32; no tolerance.
                for (unsigned start : {0, 16, 32, 48, 576}) {
                    for (unsigned k = 0; k < K; ++k) {
                        const unsigned wr = start + (k / 16) * 64 + k % 16;
                        expected += float(input[row * K + k]) * float(weights[wr * N + col]);
                    }
                }
                if (std::bit_cast<uint32_t>(acc.GetElement(row, col)) != std::bit_cast<uint32_t>(expected)) {
                    std::cerr << "Cube mismatch " << row << ',' << col << '\n';
                    return 2;
                }
            }
        }
        Right bank0, bank1;
        TASSIGN<0>(bank0);
        TASSIGN<16384>(bank1);
        Resident resident;
        TASSIGN<512>(resident);
        for (unsigned row = 0; row < residentRows; ++row)
            for (unsigned col = 0; col < N; ++col)
                if (resident.GetElement(row, col) != weights[row * N + col])
                    return 3;
        for (unsigned k = 0; k < K; ++k)
            for (unsigned col = 0; col < N; ++col) {
                const unsigned delta = (k / 16) * 64 + k % 16;
                if (bank0.GetElement(k, col) != weights[(576 + delta) * N + col] ||
                    bank1.GetElement(k, col) != weights[(48 + delta) * N + col])
                    return 4;
            }
    }
    std::cout << "PASS 1536 bitwise FP32 Cube values, readonly GM/L1 and both reused Right banks\n";
}
#endif
