/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TROWPROD_HPP
#define TROWPROD_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "TRowReduceOps.hpp"

namespace pto {

template <typename T, typename TileDataOut, typename TileDataIn, typename TileDataTmp>
__tf__ PTO_INTERNAL void TRowProd(
    typename TileDataOut::TileDType __out__ dst, typename TileDataIn::TileDType __in__ src,
    typename TileDataTmp::TileDType __in__ tmp, int validRow, int validCol)
{
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    __ubuf__ T* tmpPtr = (__ubuf__ T*)__cce_get_tile_ptr(tmp);

    constexpr unsigned dstRowStride = TileDataOut::RowStride;
    constexpr unsigned srcRowStride = TileDataIn::RowStride;

    constexpr unsigned elemsPerBlock = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elemsPerRepeat = REPEAT_BYTE / sizeof(T);
    unsigned repeatsNum = validCol / elemsPerRepeat;
    unsigned repeatRemain = validCol % elemsPerRepeat;
    unsigned tmpRepeatsNum = CeilDivision(TileDataTmp::RowStride, elemsPerRepeat);

    set_mask_norm();
    set_vector_mask(-1, -1);
    for (unsigned row = 0; row < validRow; ++row, dstPtr += dstRowStride, srcPtr += srcRowStride) {
        vector_dup(tmpPtr, (T)1.0f, tmpRepeatsNum, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        vmul(tmpPtr, srcPtr, srcPtr + elemsPerRepeat, repeatsNum / 2, 1, 1, 1, 8, 16, 16);
        pipe_barrier(PIPE_V);

        if (repeatsNum % 2 != 0) {
            vmul(tmpPtr, tmpPtr, srcPtr + (repeatsNum - 1) * elemsPerRepeat, 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
        }

        if (repeatRemain) {
            set_mask_count();
            set_vector_mask(0, repeatRemain);
            vmul(tmpPtr, tmpPtr, srcPtr + repeatsNum * elemsPerRepeat, 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            set_mask_norm();
            set_vector_mask(-1, -1);
        }

        unsigned reduceRepeat = repeatsNum / 2;
        while (reduceRepeat > 0) {
            vmul(tmpPtr, tmpPtr, tmpPtr + elemsPerRepeat, reduceRepeat / 2, 1, 1, 1, 8, 16, 16);
            reduceRepeat /= 2;
            pipe_barrier(PIPE_V);
        }

        if constexpr (sizeof(T) == sizeof(int32_t)) {
            set_mask_count();
            set_vector_mask(0, elemsPerRepeat / 2);
            vmul(tmpPtr, tmpPtr, tmpPtr + elemsPerRepeat / 2, 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            set_vector_mask(0, elemsPerRepeat / 4);
            vmul(tmpPtr, tmpPtr, tmpPtr + elemsPerRepeat / 4, 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            set_vector_mask(0, elemsPerRepeat / 8);
            vmul(tmpPtr, tmpPtr, tmpPtr + elemsPerRepeat / 8, 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);

            PtoSetWaitFlag<PIPE_V, PIPE_S>();
            for (int32_t i = elemsPerRepeat / 16; i < elemsPerRepeat / 8; i++) {
                tmpPtr[elemsPerBlock + i - elemsPerRepeat / 16] = tmpPtr[i];
            }
            PtoSetWaitFlag<PIPE_S, PIPE_V>();

            set_vector_mask(0, elemsPerRepeat / 16);
            vmul(tmpPtr, tmpPtr, tmpPtr + elemsPerBlock, 1, 1, 1, 1, 8, 8, 8);

            PtoSetWaitFlag<PIPE_V, PIPE_S>();
            for (int32_t i = elemsPerRepeat / 32; i < elemsPerRepeat / 16; i++) {
                tmpPtr[elemsPerBlock + i - elemsPerRepeat / 32] = tmpPtr[i];
            }
            PtoSetWaitFlag<PIPE_S, PIPE_V>();

            set_vector_mask(0, elemsPerRepeat / 32);
            vmul(tmpPtr, tmpPtr, tmpPtr + elemsPerBlock, 1, 1, 1, 1, 8, 8, 8);

            PtoSetWaitFlag<PIPE_V, PIPE_S>();
            dstPtr[0] = tmpPtr[0] * tmpPtr[1];
            PtoSetWaitFlag<PIPE_S, PIPE_V>();
        } else {
            set_mask_count();
            set_vector_mask(0, elemsPerRepeat / 2);
            vmul(tmpPtr, tmpPtr, tmpPtr + elemsPerRepeat / 2, 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);

            set_vector_mask(0, elemsPerRepeat / 4);
            vmul(tmpPtr, tmpPtr, tmpPtr + elemsPerRepeat / 4, 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);

            set_vector_mask(0, elemsPerRepeat / 8);
            vmul(tmpPtr, tmpPtr, tmpPtr + elemsPerRepeat / 8, 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);

            PtoSetWaitFlag<PIPE_V, PIPE_S>();
            for (int32_t i = elemsPerRepeat / 16; i < elemsPerRepeat / 8; i++) {
                tmpPtr[elemsPerBlock + i - elemsPerRepeat / 16] = tmpPtr[i];
            }
            PtoSetWaitFlag<PIPE_S, PIPE_V>();

            set_vector_mask(0, elemsPerRepeat / 16);
            vmul(tmpPtr, tmpPtr, tmpPtr + elemsPerBlock, 1, 1, 1, 1, 8, 8, 8);

            PtoSetWaitFlag<PIPE_V, PIPE_S>();
            for (int32_t i = elemsPerRepeat / 32; i < elemsPerRepeat / 16; i++) {
                tmpPtr[elemsPerBlock + i - elemsPerRepeat / 32] = tmpPtr[i];
            }
            PtoSetWaitFlag<PIPE_S, PIPE_V>();

            set_vector_mask(0, elemsPerRepeat / 32);
            vmul(tmpPtr, tmpPtr, tmpPtr + elemsPerBlock, 1, 1, 1, 1, 8, 8, 8);

            if constexpr (elemsPerRepeat > 64) {
                PtoSetWaitFlag<PIPE_V, PIPE_S>();
                for (int32_t i = elemsPerRepeat / 64; i < elemsPerRepeat / 32; i++) {
                    tmpPtr[elemsPerBlock + i - elemsPerRepeat / 64] = tmpPtr[i];
                }
                PtoSetWaitFlag<PIPE_S, PIPE_V>();

                set_vector_mask(0, elemsPerRepeat / 64);
                vmul(tmpPtr, tmpPtr, tmpPtr + elemsPerBlock, 1, 1, 1, 1, 8, 8, 8);
            }

            PtoSetWaitFlag<PIPE_V, PIPE_S>();
            if constexpr (std::is_same_v<T, half>) {
                dstPtr[0] = (half)((float)(tmpPtr[0]) * (float)(tmpPtr[1]));
            } else {
                dstPtr[0] = tmpPtr[0] * tmpPtr[1];
            }
            PtoSetWaitFlag<PIPE_S, PIPE_V>();
        }
        set_mask_norm();
        set_vector_mask(-1, -1);
    }
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWPROD_IMPL(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp)
{
    using T = typename TileDataIn::DType;
    int validCol = src.GetValidCol();
    int validRow = src.GetValidRow();
    TRowReduceCheck<TileDataOut, TileDataIn>(validRow, validCol, dst.GetValidRow());

    TRowProd<T, TileDataOut, TileDataIn, TileDataTmp>(dst.data(), src.data(), tmp.data(), validRow, validCol);
}

} // namespace pto
#endif
