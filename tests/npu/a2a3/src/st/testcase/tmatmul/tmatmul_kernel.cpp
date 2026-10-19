/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>

using namespace pto;

constexpr uint16_t BIAS_ALIGN = 64;

template <typename T>
AICORE constexpr inline T CeilAlign(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

template <typename T, typename U, typename S, typename B, int validM, int validK, int validN, bool isBias>
__global__ AICORE void RunTMATMUL_GEMV_CLOSE(__gm__ T* out, __gm__ U* src0, __gm__ S* src1, __gm__ B* src2)
{
    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(U);
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);

    using GlobalDataSrc0 = GlobalTensor<
        U, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<
        S, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    using GlobalDataOut = GlobalTensor<
        T, pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using GlobalDataSrc2 =
        GlobalTensor<B, pto::Shape<1, 1, 1, 1, validN>, pto::Stride<validN, validN, validN, validN, 1>>;
    GlobalDataSrc2 src2Global(src2);

    using TileMatAData = Tile<TileType::Mat, U, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, S, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using TileBiasData = Tile<TileType::Mat, B, 1, N, BLayout::RowMajor, 1, validN>;

    using LeftTile = TileLeft<U, M, K, validM, validK>;
    using RightTile = TileRight<S, K, N, validK, validN>;
    using AccTile = TileAcc<T, M, N, validM, validN>;

    using BiasTile = Tile<TileType::Bias, B, 1, N, BLayout::RowMajor, 1, validN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileBiasData biasDataTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x20000);
    TASSIGN(biasDataTile, 0x40000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    BiasTile biasTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);

    /******************************TLOAD*****************************/
    TLOAD(aMatTile, src0Global);

#ifndef __PTO_AUTO__
    // clear l1 buffer which exceed the valid shape
    TFILLPAD(aMatTile, aMatTile);
#endif

    TLOAD(bMatTile, src1Global);

    if constexpr (isBias) {
        TLOAD(biasDataTile, src2Global);
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    /**************************TMOV && TEXTRACT**************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

    if constexpr (isBias) {
        TMOV(biasTile, biasDataTile);
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    if constexpr (isBias) {
        TMATMUL_BIAS(cTile, aTile, bTile, biasTile);
    } else {
        TMATMUL(cTile, aTile, bTile);
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    /********************************TSTORE****************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, typename B, int validM, int validK, int validN, bool isBias>
__global__ AICORE void RunTMATMUL(__gm__ T* out, __gm__ U* src0, __gm__ S* src1, __gm__ B* src2)
{
    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(U);
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);

    using GlobalDataSrc0 = GlobalTensor<
        U, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<
        S, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    using GlobalDataOut = GlobalTensor<
        T, pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using GlobalDataSrc2 =
        GlobalTensor<B, pto::Shape<1, 1, 1, 1, validN>, pto::Stride<validN, validN, validN, validN, 1>>;
    GlobalDataSrc2 src2Global(src2);

    using TileMatAData = Tile<TileType::Mat, U, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, S, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using TileBiasData = Tile<TileType::Mat, B, 1, N, BLayout::RowMajor, 1, validN>;

    using LeftTile = TileLeft<U, M, K, validM, validK>;
    using RightTile = TileRight<S, K, N, validK, validN>;
    using AccTile = TileAcc<T, M, N, validM, validN>;
    using BiasTile = Tile<TileType::Bias, B, 1, N, BLayout::RowMajor, 1, validN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileBiasData biasDataTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x20000);
    TASSIGN(biasDataTile, 0x40000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    BiasTile biasTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);

    /******************************TLOAD*****************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

    if constexpr (isBias) {
        TLOAD(biasDataTile, src2Global);
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    /**************************TMOV && TEXTRACT**************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

    if constexpr (isBias) {
        TMOV(biasTile, biasDataTile);
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    if constexpr (isBias) {
        TMATMUL_BIAS(cTile, aTile, bTile, biasTile);
    } else {
        TMATMUL(cTile, aTile, bTile);
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    /********************************TSTORE****************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, typename B, int validM, int validK, int validN, bool isBias>
__global__ AICORE void RunTMATMULSplitK(__gm__ T* out, __gm__ U* src0, __gm__ S* src1, __gm__ B* src2)
{
    constexpr int BASEK = 32;

    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(U);
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, BASEK);

    using GlobalDataSrc0 = GlobalTensor<
        U, pto::Shape<1, 1, 1, validM, BASEK>,
        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<
        S, pto::Shape<1, 1, 1, BASEK, validN>,
        pto::Stride<1 * BASEK * validN, 1 * BASEK * validN, BASEK * validN, validN, 1>>;
    using GlobalDataSrc2 =
        GlobalTensor<B, pto::Shape<1, 1, 1, 1, validN>, pto::Stride<validN, validN, validN, validN, 1>>;
    using GlobalDataOut = GlobalTensor<
        T, pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, U, M, BASEK, BLayout::ColMajor, validM, BASEK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, S, BASEK, N, BLayout::ColMajor, BASEK, validN, SLayout::RowMajor, 512>;
    using TileBiasData = Tile<TileType::Mat, B, 1, N, BLayout::RowMajor, 1, validN>;

    using LeftTile = TileLeft<U, M, BASEK, validM, BASEK>;
    using RightTile = TileRight<S, BASEK, N, BASEK, validN>;
    using AccTile = TileAcc<T, M, N, validM, validN>;
    using BiasTile = Tile<TileType::Bias, B, 1, N, BLayout::RowMajor, 1, validN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileBiasData biasDataTile;

    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x20000);
    TASSIGN(biasDataTile, 0x40000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    BiasTile biasTile;

    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);

    constexpr int iter = K / BASEK;

    for (int i = 0; i < iter; i++) {
        GlobalDataSrc0 src0Global(src0 + i * BASEK);
        GlobalDataSrc1 src1Global(src1 + validN * i * BASEK);

        /******************************TLOAD*****************************/
        TFILLPAD(aMatTile, aMatTile);
        TFILLPAD(bMatTile, bMatTile);
        TLOAD(aMatTile, src0Global);
        TLOAD(bMatTile, src1Global);

        if constexpr (isBias) {
            TLOAD(biasDataTile, src2Global);
        }

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

        /**************************TMOV && TEXTRACT**************************/
        TMOV(aTile, aMatTile);
        TMOV(bTile, bMatTile);

        if constexpr (isBias) {
            TMOV(biasTile, biasDataTile);
        }

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

        if (i == 0) {
            if constexpr (isBias) {
                TMATMUL_BIAS(cTile, aTile, bTile, biasTile);
            } else {
                TMATMUL(cTile, aTile, bTile);
            }
        } else {
            TMATMUL_ACC(cTile, cTile, aTile, bTile);
        }
#ifndef __PTO_AUTO__
        set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
#endif
    }
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, typename B, int validM, int validK, int validN, bool isBias>
__global__ AICORE void RunTGEMV(__gm__ T* out, __gm__ U* src0, __gm__ S* src1, __gm__ B* src2)
{
    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(U);
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);

    using GlobalDataSrc0 = GlobalTensor<
        U, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<
        S, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    using GlobalDataOut = GlobalTensor<
        T, pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using GlobalDataSrc2 =
        GlobalTensor<B, pto::Shape<1, 1, 1, 1, validN>, pto::Stride<validN, validN, validN, validN, 1>>;
    GlobalDataSrc2 src2Global(src2);
    constexpr int blockLeft = CUBE_BLOCK_SIZE / sizeof(U);
    constexpr int KLeft = CeilAlign<int>(validK, blockLeft);
    using TileMatADataGemv = Tile<TileType::Mat, U, 1, KLeft, BLayout::RowMajor, 1, validK>;
    using TileMatBData = Tile<TileType::Mat, S, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using TileBiasData = Tile<TileType::Mat, B, 1, N, BLayout::RowMajor, 1, validN>;

    using LeftTile = TileLeft<U, 1, KLeft, 1, validK>;
    using RightTile = TileRight<S, K, N, validK, validN>;
    using AccTile = TileAcc<T, M, N, validM, validN>;

    using BiasTile = Tile<TileType::Bias, B, 1, N, BLayout::RowMajor, 1, validN>;

    TileMatADataGemv aMatTileGemv;
    TileMatBData bMatTile;
    TileBiasData biasDataTile;
    TASSIGN(aMatTileGemv, 0x0);
    TASSIGN(bMatTile, 0x20000);
    TASSIGN(biasDataTile, 0x40000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    BiasTile biasTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);

    /******************************TLOAD*****************************/
    TLOAD(aMatTileGemv, src0Global);
    TLOAD(bMatTile, src1Global);

    if constexpr (isBias) {
        TLOAD(biasDataTile, src2Global);
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    /**************************TMOV && TEXTRACT**************************/
    TEXTRACT(aTile, aMatTileGemv, 0, 0);
    TMOV(bTile, bMatTile);

    if constexpr (isBias) {
        TMOV(biasTile, biasDataTile);
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    if constexpr (isBias) {
        TGEMV_BIAS(cTile, aTile, bTile, biasTile);
    } else {
        TGEMV(cTile, aTile, bTile);
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    /********************************TSTORE****************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, typename B, int validM, int validK, int validN, RoundMode tf32TransMode>
__global__ AICORE void RunTMATMUL_TF32(__gm__ T* out, __gm__ U* src0, __gm__ S* src1, __gm__ B* src2)
{
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, 16);
    constexpr int K = CeilAlign<int>(validK, 16);

    using GlobalDataSrc0 = GlobalTensor<
        U, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<
        S, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    using GlobalDataOut = GlobalTensor<
        T, pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, U, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, S, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;

    using LeftTile = TileLeftCompact<U, M, K, validM, validK>;
    using RightTile = TileRightCompact<S, K, N, validK, validN>;
    using AccTile = TileAcc<T, M, N, validM, validN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x20000);

    LeftTile aTile;
    aTile.SetMadTF32Mode(tf32TransMode);
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    /******************************TLOAD*****************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    /**************************TMOV && TEXTRACT**************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif
    TMATMUL(cTile, aTile, bTile);
    aTile.ResetMadMode();
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    /********************************TSTORE****************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, typename B, int validM, int validK, int validN, bool isBias>
__global__ AICORE void RunTGEMVSplitK(__gm__ T* out, __gm__ U* src0, __gm__ S* src1, __gm__ B* src2)
{
    constexpr int BASEK = 256;

    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(U);
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, BASEK);

    using GlobalDataSrc0 = GlobalTensor<
        U, pto::Shape<1, 1, 1, validM, BASEK>,
        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<
        S, pto::Shape<1, 1, 1, BASEK, validN>,
        pto::Stride<1 * BASEK * validN, 1 * BASEK * validN, BASEK * validN, validN, 1>>;
    using GlobalDataSrc2 =
        GlobalTensor<B, pto::Shape<1, 1, 1, 1, validN>, pto::Stride<validN, validN, validN, validN, 1>>;
    using GlobalDataOut = GlobalTensor<
        T, pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatADataGemv = Tile<TileType::Mat, U, 1, BASEK, BLayout::RowMajor, 1, BASEK>;
    using TileMatBData = Tile<TileType::Mat, S, BASEK, N, BLayout::ColMajor, BASEK, validN, SLayout::RowMajor, 512>;
    using TileBiasData = Tile<TileType::Mat, B, 1, N, BLayout::RowMajor, 1, validN>;

    using LeftTile = TileLeft<U, 1, BASEK, 1, BASEK>;
    using RightTile = TileRight<S, BASEK, N, BASEK, validN>;
    using AccTile = TileAcc<T, M, N, validM, validN>;
    using BiasTile = Tile<TileType::Bias, B, 1, N, BLayout::RowMajor, 1, validN>;

    TileMatADataGemv aMatTile;
    TileMatBData bMatTile;
    TileBiasData biasDataTile;

    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x20000);
    TASSIGN(biasDataTile, 0x40000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    BiasTile biasTile;

    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);

    constexpr int iter = K / BASEK;

    for (int i = 0; i < iter; i++) {
        GlobalDataSrc0 src0Global(src0 + i * BASEK);
        GlobalDataSrc1 src1Global(src1 + validN * i * BASEK);

        /******************************TLOAD*****************************/
        TLOAD(aMatTile, src0Global);
        TLOAD(bMatTile, src1Global);

        if constexpr (isBias) {
            TLOAD(biasDataTile, src2Global);
        }

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

        /**************************TMOV && TEXTRACT**************************/
        TMOV(aTile, aMatTile);
        TMOV(bTile, bMatTile);

        if constexpr (isBias) {
            TMOV(biasTile, biasDataTile);
        }

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

        if (i == 0) {
            if constexpr (isBias) {
                TGEMV_BIAS(cTile, aTile, bTile, biasTile);
            } else {
                TGEMV(cTile, aTile, bTile);
            }
        } else {
            TGEMV_ACC(cTile, cTile, aTile, bTile);
        }
#ifndef __PTO_AUTO__
        set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
#endif
    }
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <int32_t tilingKey>
void LaunchTMATMUL(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream)
{
    if constexpr (tilingKey == 1) {
        RunTMATMUL<float, half, half, float, 31, 120, 58, false><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1), nullptr);
    } else if constexpr (tilingKey == 2) {
        RunTMATMUL<int32_t, int8_t, int8_t, int32_t, 65, 90, 89, false><<<1, nullptr, stream>>>(
            reinterpret_cast<int32_t*>(out), reinterpret_cast<int8_t*>(src0), reinterpret_cast<int8_t*>(src1), nullptr);
    } else if constexpr (tilingKey == 3) {
        RunTMATMULSplitK<float, half, half, float, 5, 75, 11, false><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1), nullptr);
    } else if constexpr (tilingKey == 4) {
        RunTMATMUL_GEMV_CLOSE<float, half, half, float, 1, 256, 64, false><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1), nullptr);
    } else if constexpr (tilingKey == 5) {
        RunTGEMV<float, half, half, float, 1, 16, 32, false><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1), nullptr);
    } else if constexpr (tilingKey == 6) {
        RunTGEMV<float, half, half, float, 1, 200, 32, false><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1), nullptr);
    } else if constexpr (tilingKey == 7) {
        RunTMATMUL_TF32<float, float, float, float, 16, 32, 64, RoundMode::CAST_RINT><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(src0), reinterpret_cast<float*>(src1), nullptr);
    } else if constexpr (tilingKey == 8) {
        RunTMATMUL_TF32<float, float, float, float, 5, 75, 11, RoundMode::CAST_ROUND><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(src0), reinterpret_cast<float*>(src1), nullptr);
    }
}

template <int32_t tilingKey>
void LaunchTMATMULBIAS(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream)
{
    if constexpr (tilingKey == 1) {
        RunTMATMUL<float, half, half, float, 26, 100, 94, true><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<float*>(src2));
    } else if constexpr (tilingKey == 2) {
        RunTMATMUL<float, half, half, float, 101, 288, 67, true><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<float*>(src2));
    } else if constexpr (tilingKey == 3) {
        RunTMATMUL<float, float, float, float, 15, 16, 15, true><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(src0), reinterpret_cast<float*>(src1),
            reinterpret_cast<float*>(src2));
    } else if constexpr (tilingKey == 4) {
        RunTMATMUL<int32_t, int8_t, int8_t, int32_t, 55, 127, 29, true><<<1, nullptr, stream>>>(
            reinterpret_cast<int32_t*>(out), reinterpret_cast<int8_t*>(src0), reinterpret_cast<int8_t*>(src1),
            reinterpret_cast<int32_t*>(src2));
    } else if constexpr (tilingKey == 5) {
        RunTMATMUL<float, bfloat16_t, bfloat16_t, float, 11, 402, 30, true><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<bfloat16_t*>(src0), reinterpret_cast<bfloat16_t*>(src1),
            reinterpret_cast<float*>(src2));
    } else if constexpr (tilingKey == 6) {
        RunTMATMUL<int32_t, int8_t, int8_t, int32_t, 150, 89, 50, true><<<1, nullptr, stream>>>(
            reinterpret_cast<int32_t*>(out), reinterpret_cast<int8_t*>(src0), reinterpret_cast<int8_t*>(src1),
            reinterpret_cast<int32_t*>(src2));
    } else if constexpr (tilingKey == 7) {
        RunTMATMULSplitK<int32_t, int8_t, int8_t, int32_t, 135, 64, 88, true><<<1, nullptr, stream>>>(
            reinterpret_cast<int32_t*>(out), reinterpret_cast<int8_t*>(src0), reinterpret_cast<int8_t*>(src1),
            reinterpret_cast<int32_t*>(src2));
    } else if constexpr (tilingKey == 8) {
        RunTGEMVSplitK<float, half, half, float, 1, 512, 32, true><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<float*>(src2));
    }
}

template void LaunchTMATMUL<1>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void LaunchTMATMUL<2>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void LaunchTMATMUL<3>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void LaunchTMATMUL<4>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void LaunchTMATMUL<5>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void LaunchTMATMUL<6>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void LaunchTMATMUL<7>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void LaunchTMATMUL<8>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

template void LaunchTMATMULBIAS<1>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream);
template void LaunchTMATMULBIAS<2>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream);
template void LaunchTMATMULBIAS<3>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream);
template void LaunchTMATMULBIAS<4>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream);
template void LaunchTMATMULBIAS<5>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream);
template void LaunchTMATMULBIAS<6>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream);
template void LaunchTMATMULBIAS<7>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream);
template void LaunchTMATMULBIAS<8>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream);

constexpr int kFractal = 16; // L0C row fractal
constexpr int kK = 32;

template <event_t EVENT>
AICORE inline void SyncMte2ToMte1()
{
#ifndef __PTO_AUTO__
    event_t event = EVENT;
    set_flag(PIPE_MTE2, PIPE_MTE1, event);
    wait_flag(PIPE_MTE2, PIPE_MTE1, event);
#endif
}

template <event_t EVENT>
AICORE inline void SyncMte1ToM()
{
#ifndef __PTO_AUTO__
    event_t event = EVENT;
    set_flag(PIPE_MTE1, PIPE_M, event);
    wait_flag(PIPE_MTE1, PIPE_M, event);
#endif
}

template <event_t EVENT>
AICORE inline void SyncMToMte2()
{
#ifndef __PTO_AUTO__
    event_t event = EVENT;
    set_flag(PIPE_M, PIPE_MTE2, event);
    wait_flag(PIPE_M, PIPE_MTE2, event);
#endif
}

AICORE inline void SyncMToFix()
{
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
}

template <event_t EVENT, typename MatTile, typename GlobalTile, typename CubeTile>
AICORE inline void LoadMove(MatTile& matTile, GlobalTile& globalTile, CubeTile& cubeTile)
{
    TLOAD(matTile, globalTile);
    SyncMte2ToMte1<EVENT>();
    TMOV(cubeTile, matTile);
    SyncMte1ToM<EVENT>();
}

template <typename AccTile, typename GlobalOut>
AICORE inline void StoreWindow(GlobalOut& dstGlobal, AccTile& cTile)
{
    SyncMToMte2<EVENT_ID0>();
    SyncMToFix();
    TSTORE(dstGlobal, cTile);
}

template <typename T, typename U, int WINDOWS, int N>
__global__ AICORE void RunAccRowWindow(__gm__ T* out, __gm__ U* src0, __gm__ U* src1)
{
    constexpr int kTileM = kFractal;

    using GlobalA =
        GlobalTensor<U, pto::Shape<1, 1, 1, kTileM, kK>, pto::Stride<kTileM * kK, kTileM * kK, kTileM * kK, kK, 1>>;
    using GlobalB = GlobalTensor<U, pto::Shape<1, 1, 1, kK, N>, pto::Stride<kK * N, kK * N, kK * N, N, 1>>;
    using GlobalOut = GlobalTensor<
        T, pto::Shape<1, 1, 1, WINDOWS * kTileM, N>,
        pto::Stride<WINDOWS * kTileM * N, WINDOWS * kTileM * N, WINDOWS * kTileM * N, N, 1>>;
    GlobalB src1Global(src1);
    GlobalOut dstGlobal(out);

    using TileMatA = Tile<TileType::Mat, U, kTileM, kK, BLayout::ColMajor, kTileM, kK, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, U, kK, N, BLayout::ColMajor, kK, N, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<U, kTileM, kK, kTileM, kK>;
    using RightTile = TileRight<U, kK, N, kK, N>;
    using AccTile = TileAcc<T, kTileM, N, kTileM, N>;

    TileMatA aMatTile;
    TileMatB bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x20000);

    LeftTile aTile;
    RightTile bTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);

    LoadMove<EVENT_ID0>(bMatTile, src1Global, bTile);

    for (int t = 0; t < WINDOWS; ++t) {
        GlobalA src0Global(src0 + t * kTileM * kK);
        LoadMove<EVENT_ID0>(aMatTile, src0Global, aTile);

        AccTile cTile;
        TASSIGN(cTile, 0x0);
        TMATMUL(cTile, aTile, bTile);

        GlobalOut dstWindowGlobal(out + t * kFractal * N);
        StoreWindow(dstWindowGlobal, cTile);
    }
}

template <typename T, typename U, int WINDOWS, int N>
__global__ AICORE void RunAccColumnWindow(__gm__ T* out, __gm__ U* src0, __gm__ U* src1)
{
    constexpr int kTileM = kFractal;

    using GlobalA =
        GlobalTensor<U, pto::Shape<1, 1, 1, kTileM, kK>, pto::Stride<kTileM * kK, kTileM * kK, kTileM * kK, kK, 1>>;
    using GlobalB = GlobalTensor<
        U, pto::Shape<1, 1, 1, kK, N>,
        pto::Stride<kK * WINDOWS * N, kK * WINDOWS * N, kK * WINDOWS * N, WINDOWS * N, 1>>;
    using GlobalOut = GlobalTensor<
        T, pto::Shape<1, 1, 1, kTileM, WINDOWS * N>,
        pto::Stride<kTileM * WINDOWS * N, kTileM * WINDOWS * N, kTileM * WINDOWS * N, WINDOWS * N, 1>>;
    GlobalA src0Global(src0);
    GlobalOut dstGlobal(out);

    using TileMatA = Tile<TileType::Mat, U, kTileM, kK, BLayout::ColMajor, kTileM, kK, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, U, kK, N, BLayout::ColMajor, kK, N, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<U, kTileM, kK, kTileM, kK>;
    using RightTile = TileRight<U, kK, N, kK, N>;
    using AccWindow = TileAcc<T, kTileM, WINDOWS * N, kTileM, N>;
    using AccFull = TileAcc<T, kTileM, WINDOWS * N, kTileM, WINDOWS * N>;

    TileMatA aMatTile;
    TileMatB bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x20000);

    LeftTile aTile;
    RightTile bTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);

    LoadMove<EVENT_ID0>(aMatTile, src0Global, aTile);

    for (int t = 0; t < WINDOWS; ++t) {
        GlobalB src1Global(src1 + t * N);
        LoadMove<EVENT_ID1>(bMatTile, src1Global, bTile);

        AccWindow cWindow;
        TASSIGN(cWindow, static_cast<size_t>(t) * N * kTileM * sizeof(T));
        TMATMUL(cWindow, aTile, bTile);

        SyncMToMte2<EVENT_ID0>();
    }

    SyncMToFix();
    AccFull cFull;
    TASSIGN(cFull, 0x0);
    TSTORE(dstGlobal, cFull);
    out = dstGlobal.data();
}

template <int32_t tilingKey>
void LaunchTMATMULWINDOW(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream)
{
    if constexpr (tilingKey == 9) {
        RunAccRowWindow<float, half, 2, 32><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
    } else if constexpr (tilingKey == 10) {
        RunAccColumnWindow<float, half, 2, 32><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
    }
}

template void LaunchTMATMULWINDOW<9>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void LaunchTMATMULWINDOW<10>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
