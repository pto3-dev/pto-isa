/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TGATHERB_HPP
#define TGATHERB_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>

namespace pto {

// On kirinDev0000 (dav-l510), CCE_VL=64 (vs a5's 256). The a5 TGatherB hardcodes
// the offset stride as 8 uint32 elements (32 bytes) per vgatherb call, which is
// naturally 32-byte aligned. On kirinDev0000, offsetsPerRepeat = CCE_VL/BLOCK_BYTE_SIZE = 2
// (8 bytes per vgatherb), so the vlds offset address is NOT 32-byte aligned.
//
// Fix: always load 8 uint32 offsets (32 bytes, aligned) per vlds, then use vselr
// with an index vector to extract the correct offsetsPerRepeat entries to position 0
// for vgatherb. We process 4 repeats per aligned load (4 * offsetsPerRepeat = 8).

template <typename T, typename SrcT, typename DistT>
__tf__ PTO_INTERNAL void TGatherBSelectGather(
    RegTensor<uint32_t>& vregOffset, RegTensor<uint32_t>& vregOffsetSel, RegTensor<uint32_t>& selIdx,
    RegTensor<T>& vregDst, __ubuf__ SrcT* srcAddr, __ubuf__ T* dstPtr, uint32_t subIdx, uint32_t dstOffset,
    DistT distMode, MaskReg& preg)
{
    vci((RegTensor<int32_t>&)selIdx, (int32_t)subIdx, INC_ORDER);
    vselr((RegTensor<uint32_t>&)vregOffsetSel, (RegTensor<uint32_t>&)vregOffset, (RegTensor<uint32_t>&)selIdx);
    pto_vgatherb(vregDst, srcAddr, vregOffsetSel, preg);
    vsts(vregDst, dstPtr, dstOffset, distMode, preg);
}

template <
    typename TileDataDst, typename TileDataSrc, typename TileDataOffset, unsigned elementsPerRepeat,
    unsigned blockSizeElem, unsigned dstRowStride, unsigned offsetRowStride>
__tf__ PTO_INTERNAL void TGatherBRowWise(
    typename TileDataDst::TileDType __out__ dst, typename TileDataSrc::TileDType __in__ src,
    typename TileDataOffset::TileDType __in__ offset, unsigned validRow, unsigned validCol, uint16_t repeatTimes,
    uint32_t remainEleNum)
{
    using T = typename TileDataDst::DType;
    constexpr unsigned offsetsPerRepeat = elementsPerRepeat / blockSizeElem;
    constexpr unsigned alignOffsets = BLOCK_BYTE_SIZE / sizeof(uint32_t);
    constexpr unsigned repeatsPerAlign = alignOffsets / offsetsPerRepeat;
    __ubuf__ typename TileDataSrc::DType* srcAddr = (__ubuf__ typename TileDataSrc::DType*)__cce_get_tile_ptr(src);
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ uint32_t* offsetPtr = (__ubuf__ uint32_t*)__cce_get_tile_ptr(offset);
    __VEC_SCOPE__
    {
        uint32_t countFull = elementsPerRepeat;
        uint32_t countRem = remainEleNum;
        MaskReg preg0 = CreatePredicate<T>(countFull);
        MaskReg preg1 = CreatePredicate<T>(countRem);
        RegTensor<uint32_t> vregOffset, vregOffsetSel, selIdx;
        RegTensor<T> vregDst;
        uint16_t alignedRepeatTimes = repeatTimes / repeatsPerAlign * repeatsPerAlign;
        for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
            uint32_t perRowOffset = (uint32_t)i * offsetRowStride;
            uint32_t perRowDstOffset = (uint32_t)i * dstRowStride;
            for (uint16_t j = 0; j < alignedRepeatTimes; j += (uint16_t)repeatsPerAlign) {
                vlds(vregOffset, offsetPtr, perRowOffset + (uint32_t)j * offsetsPerRepeat, NORM);
                for (uint16_t k = 0; k < (uint16_t)repeatsPerAlign; ++k) {
                    uint16_t repIdx = j + k;
                    MaskReg& preg = (repIdx == repeatTimes - 1) ? preg1 : preg0;
                    TGatherBSelectGather<T, typename TileDataSrc::DType>(
                        vregOffset, vregOffsetSel, selIdx, vregDst, srcAddr, dstPtr, (uint32_t)k * offsetsPerRepeat,
                        perRowDstOffset + (uint32_t)repIdx * elementsPerRepeat, NORM_B32, preg);
                }
            }
            for (uint16_t j = alignedRepeatTimes; j < repeatTimes; ++j) {
                uint32_t fullOffset = perRowOffset + (uint32_t)j * offsetsPerRepeat;
                vlds(vregOffset, offsetPtr, fullOffset / alignOffsets * alignOffsets, NORM);
                MaskReg& preg = (j == repeatTimes - 1) ? preg1 : preg0;
                TGatherBSelectGather<T, typename TileDataSrc::DType>(
                    vregOffset, vregOffsetSel, selIdx, vregDst, srcAddr, dstPtr, fullOffset % alignOffsets,
                    perRowDstOffset + (uint32_t)j * elementsPerRepeat, NORM_B32, preg);
            }
        }
    }
}

template <
    typename TileDataDst, typename TileDataSrc, typename TileDataOffset, unsigned elementsPerRepeat,
    unsigned blockSizeElem, unsigned dstRowStride, unsigned offsetRowStride>
__tf__ PTO_INTERNAL void TGatherBColWise(
    typename TileDataDst::TileDType __out__ dst, typename TileDataSrc::TileDType __in__ src,
    typename TileDataOffset::TileDType __in__ offset, unsigned validRow, unsigned validCol, uint16_t repeatTimes,
    uint32_t remainEleNum)
{
    using T = typename TileDataDst::DType;
    constexpr unsigned offsetsPerRepeat = elementsPerRepeat / blockSizeElem;
    constexpr unsigned alignOffsets = BLOCK_BYTE_SIZE / sizeof(uint32_t);
    __ubuf__ uint32_t* offsetPtr = (__ubuf__ uint32_t*)__cce_get_tile_ptr(offset);
    __ubuf__ typename TileDataSrc::DType* srcAddr = (__ubuf__ typename TileDataSrc::DType*)__cce_get_tile_ptr(src);
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    __VEC_SCOPE__
    {
        uint32_t countFull = elementsPerRepeat;
        uint32_t countRem = remainEleNum;
        MaskReg preg0 = CreatePredicate<T>(countFull);
        MaskReg preg1 = CreatePredicate<T>(countRem);
        RegTensor<uint32_t> vregOffset, vregOffsetSel, selIdx;
        RegTensor<T> vregDst;
        uint16_t fullRepeatLimit = static_cast<uint16_t>(repeatTimes - 1);
        for (uint16_t i = 0; i < fullRepeatLimit; i++) {
            uint32_t perRowOffset = (uint32_t)i * offsetsPerRepeat;
            uint32_t perRowDstOffset = (uint32_t)i * elementsPerRepeat;
            for (uint16_t j = 0; j < (uint16_t)validRow; j++) {
                uint32_t fullOffset = perRowOffset + (uint32_t)j * offsetRowStride;
                vlds(vregOffset, offsetPtr, fullOffset / alignOffsets * alignOffsets, NORM);
                TGatherBSelectGather<T, typename TileDataSrc::DType>(
                    vregOffset, vregOffsetSel, selIdx, vregDst, srcAddr, dstPtr, fullOffset % alignOffsets,
                    perRowDstOffset + (uint32_t)j * dstRowStride, distValue, preg0);
            }
        }
        uint16_t lastRepeat = repeatTimes - 1;
        for (uint16_t j = 0; j < (uint16_t)validRow; j++) {
            uint32_t fullOffset = (uint32_t)lastRepeat * offsetsPerRepeat + (uint32_t)j * offsetRowStride;
            vlds(vregOffset, offsetPtr, fullOffset / alignOffsets * alignOffsets, NORM);
            TGatherBSelectGather<T, typename TileDataSrc::DType>(
                vregOffset, vregOffsetSel, selIdx, vregDst, srcAddr, dstPtr, fullOffset % alignOffsets,
                (uint32_t)lastRepeat * elementsPerRepeat + (uint32_t)j * dstRowStride, distValue, preg1);
        }
    }
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataOffset>
PTO_INTERNAL void TGATHERB_IMPL(TileDataDst& dst, TileDataSrc& src, TileDataOffset& offset)
{
    static_assert(
        sizeof(typename TileDataDst::DType) == 4 || sizeof(typename TileDataDst::DType) == 2 ||
            sizeof(typename TileDataDst::DType) == 1,
        "Fix: TGATHERB has invalid data type.");
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileDataDst::DType);
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(typename TileDataDst::DType);
    constexpr unsigned staticRepeatTimes = (TileDataDst::Cols + elementsPerRepeat - 1) / elementsPerRepeat;
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned offsetRowStride = TileDataOffset::RowStride;
    unsigned validRow = dst.GetValidRow();
    if constexpr (TileDataDst::ValidCol > 0) {
        constexpr unsigned validCol = static_cast<unsigned>(TileDataDst::ValidCol);
        constexpr uint16_t repeatTimes = static_cast<uint16_t>((validCol + elementsPerRepeat - 1) / elementsPerRepeat);
        constexpr uint32_t remainEleNum =
            (validCol % elementsPerRepeat) == 0 ? elementsPerRepeat : (validCol % elementsPerRepeat);
        if constexpr (staticRepeatTimes > TileDataDst::Rows) {
            TGatherBRowWise<
                TileDataDst, TileDataSrc, TileDataOffset, elementsPerRepeat, blockSizeElem, dstRowStride,
                offsetRowStride>(dst.data(), src.data(), offset.data(), validRow, validCol, repeatTimes, remainEleNum);
        } else {
            TGatherBColWise<
                TileDataDst, TileDataSrc, TileDataOffset, elementsPerRepeat, blockSizeElem, dstRowStride,
                offsetRowStride>(dst.data(), src.data(), offset.data(), validRow, validCol, repeatTimes, remainEleNum);
        }
    } else {
        unsigned validCol = dst.GetValidCol();
        uint16_t repeatTimes = CeilDivision(validCol, elementsPerRepeat);
        uint32_t remainEleNum = validCol % elementsPerRepeat ?: elementsPerRepeat;
        if constexpr (staticRepeatTimes > TileDataDst::Rows) {
            TGatherBRowWise<
                TileDataDst, TileDataSrc, TileDataOffset, elementsPerRepeat, blockSizeElem, dstRowStride,
                offsetRowStride>(dst.data(), src.data(), offset.data(), validRow, validCol, repeatTimes, remainEleNum);
        } else {
            TGatherBColWise<
                TileDataDst, TileDataSrc, TileDataOffset, elementsPerRepeat, blockSizeElem, dstRowStride,
                offsetRowStride>(dst.data(), src.data(), offset.data(), validRow, validCol, repeatTimes, remainEleNum);
        }
    }
}
} // namespace pto
#endif
