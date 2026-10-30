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

using namespace pto;
static_assert(OpPipeEntry<Op::TINSERT_V2M>::pipe == PIPE_MTE3);
static_assert(OpPipeEntry<Op::TINSERT_V2V>::pipe == PIPE_V);
static_assert(OpPipeEntry<Op::TMOV_V2M>::pipe == PIPE_MTE3);
using Matrix = Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512>;
using Weight = Tile<TileType::Mat, half, 64, 16, BLayout::ColMajor, 64, 16, SLayout::RowMajor, 512>;
using Left = Tile<TileType::Left, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::RowMajor, 512>;
using Right = TileRight<half, 64, 16>;
using Acc = TileAcc<float, 64, 16>;
using Result = Tile<TileType::Vec, float, 64, 16>;
using MatrixGM = GlobalTensor<half, Shape<1, 1, 1, 64, 64>, Stride<1, 1, 1, 64, 1>>;
using WeightGM = GlobalTensor<half, Shape<1, 1, 1, 64, 16>, Stride<1, 1, 1, 16, 1>>;
using OutputGM = GlobalTensor<float, Shape<1, 1, 1, 64, 16>, Stride<1, 1, 1, 16, 1>>;

// Compilation-only entry: instantiate native DMA, MTE3 event chains, UB reuse
// and the downstream Cube consumer without querying or opening a device.
template <int Columns>
PTO_INTERNAL void RunVecMatInsertPipeline(
    __gm__ half* panels, __gm__ half* initial, __gm__ half* weights, __gm__ float* output)
{
    using Source = Tile<TileType::Vec, half, 16, Columns>;
    using SourceGM = GlobalTensor<half, Shape<1, 1, 1, 16, Columns>, Stride<1, 1, 1, Columns, 1>>;
    Source s0, s1;
    Matrix matrix;
    Weight weight;
    Left left;
    Right right;
    Acc acc;
    Result result;
    TASSIGN<0>(s0);
    TASSIGN<4096>(s1);
    TASSIGN<0>(matrix);
    TASSIGN<8192>(weight);
    TASSIGN<0>(left);
    TASSIGN<0>(right);
    TASSIGN<0>(acc);
    TASSIGN<8192>(result);
    MatrixGM gm(initial);
    WeightGM gb(weights);
    SourceGM g0(panels), g1(panels + 16 * Columns), g2(panels + 32 * Columns), g3(panels + 48 * Columns);
    OutputGM go(output);
    Event<Op::TLOAD, Op::TINSERT_V2M> initialReady = TLOAD(matrix, gm);
    Event<Op::TLOAD, Op::TMOV_M2R> weightReady = TLOAD(weight, gb);
    Event<Op::TLOAD, Op::TINSERT_V2M> ready0 = TLOAD(s0, g0);
    Event<Op::TLOAD, Op::TINSERT_V2M> ready1 = TLOAD(s1, g1);
    Event<Op::TINSERT_V2M, Op::TLOAD> free0 = TINSERT(matrix, s0, 0, 16, initialReady, ready0);
    Event<Op::TINSERT_V2M, Op::TLOAD> free1 = TINSERT(matrix, s1, 16, 16, ready1);
    Event<Op::TLOAD, Op::TINSERT_V2M> ready2 = TLOAD(s0, g2, free0);
    TINSERT(matrix, s0, 32, 16, ready2);
    Event<Op::TLOAD, Op::TINSERT_V2M> ready3 = TLOAD(s1, g3, free1);
    Event<Op::TINSERT_V2M, Op::TMOV_M2L> inserted = TINSERT(matrix, s1, 48, 16, ready3);
    TMOV(left, matrix, inserted);
    TMOV(right, weight, weightReady);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(acc, left, right);
    set_flag(PIPE_M, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_V, EVENT_ID0);
    TMOV<Result, Acc, AccToVecMode::SingleModeVec0>(result, acc);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(go, result);
}

extern "C" __global__ AICORE void A1VecMatInsertPipelineWide(
    __gm__ half* panels, __gm__ half* initial, __gm__ half* weights, __gm__ float* output)
{
    RunVecMatInsertPipeline<32>(panels, initial, weights, output);
}

extern "C" __global__ AICORE void A1VecMatInsertPipelineC16(
    __gm__ half* panels, __gm__ half* initial, __gm__ half* weights, __gm__ float* output)
{
    RunVecMatInsertPipeline<16>(panels, initial, weights, output);
}
