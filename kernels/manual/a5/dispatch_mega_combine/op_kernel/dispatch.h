/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_DISPATCH_H
#define DISPATCH_MEGA_COMBINE_DISPATCH_H

#include "kernel_operator.h"

#include <pto/pto-inst.hpp>

#include "dispatch_mega_combine_tiling.h"
#include "utils/const_args.hpp"
#include "utils/hccl_window.hpp"
#include "utils/mega_expert_sync.hpp"
#include "utils/pto_vector.hpp"

constexpr uint32_t kDispatchGatherPackedTileCols = 8192U;
constexpr uint32_t kDispatchGatherPackedWordTileCols = kDispatchGatherPackedTileCols / sizeof(uint32_t);
constexpr uint32_t kDispatchMaxBufferCount = 6U;
constexpr uint32_t kDispatchMetaSlotBytes = 32U;
constexpr event_t kDispatchMaskLoadEvent = EVENT_ID6;
constexpr event_t kDispatchMaskVectorEvent = EVENT_ID0;
constexpr event_t kDispatchReadyCountEvent = EVENT_ID7;

static_assert(kDispatchGatherPackedTileCols % sizeof(uint32_t) == 0);
static_assert(kDispatchGatherPackedWordTileCols <= 4095U);
static_assert(kDispatchMaxBufferCount <= 6U);

__tf__ AICORE void CompressDispatchRouteMask(
    uint32_t routeIndexUbOffset, uint32_t maskUbOffset, uint32_t routeBegin, uint32_t validRouteCount,
    uint32_t routeCountUbOffset)
{
    constexpr uint32_t kRoutesPerRepeat = 64U;
    __ubuf__ uint8_t* ubBase = (__ubuf__ uint8_t*)0;
    __ubuf__ uint32_t* routeIndexPtr = (__ubuf__ uint32_t*)(ubBase + routeIndexUbOffset);
    __ubuf__ uint32_t* maskPtr = (__ubuf__ uint32_t*)(ubBase + maskUbOffset);
    __ubuf__ uint32_t* routeCountPtr = (__ubuf__ uint32_t*)(ubBase + routeCountUbOffset);
    const uint16_t repeatTimes = static_cast<uint16_t>((validRouteCount + kRoutesPerRepeat - 1U) / kRoutesPerRepeat);

    __VEC_SCOPE__
    {
        vector_s32 routeIndex;
        vector_s32 selectedRouteIndex;
        vector_u32 packedMask;
        vector_bool routeMask;
        vector_bool validMask;
        vector_bool executeMask;
        vector_align routeIndexAlign;
        vector_align maskAlign;
        uint32_t remainingRoutes = validRouteCount;

        sprclr(SPR_AR);
        for (uint16_t repeat = 0U; repeat < repeatTimes; ++repeat) {
            vci(routeIndex, routeBegin + static_cast<uint32_t>(repeat) * kRoutesPerRepeat, INC_ORDER);
            __ubuf__ uint32_t* currentMask = maskPtr + static_cast<uint32_t>(repeat) * 2U;
            vldas(maskAlign, currentMask);
            vldus(packedMask, maskAlign, currentMask);
            movvp(routeMask, packedMask, 0);
            validMask = plt_b32(remainingRoutes, POST_UPDATE);
            pand(executeMask, routeMask, validMask, validMask);
            vsqz(selectedRouteIndex, routeIndex, executeMask, MODE_STORED);
            vstur(routeIndexAlign, (vector_u32)selectedRouteIndex, routeIndexPtr, POST_UPDATE);
        }
        vstar(routeIndexAlign, routeIndexPtr);
        sprsts(SPR_AR, routeCountPtr, 0);
    }
}

class DispatchGather {
public:
    AICORE inline void Init(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData)
    {
        workspaceGM_ = workspaceGM;
        tilingData_ = tilingData;

        const auto& info = tilingData_->megaMoeInfo;
        problemK_ = info.K;
        topK_ = info.topK;
        expertPerRank_ = info.expertPerRank;
        routeElems_ = tilingData_->frontReorderTiling.routeElems;
        rankSize_ = tilingData_->runtimeInfo.rankSize;

        gmAPtr_ = reinterpret_cast<__gm__ int8_t*>(workspaceGM + tilingData_->dispatchTiling.gmAOffset);
        gmAScalePtr_ = reinterpret_cast<__gm__ uint8_t*>(workspaceGM + tilingData_->dispatchTiling.gmAScaleOffset);
        routeMetaPtr_ = reinterpret_cast<__gm__ int32_t*>(workspaceGM + tilingData_->dispatchTiling.routeMetaOffset);
        cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t*>(workspaceGM + tilingData_->frontReorderTiling.cumsumMMOffset);

        remoteWindow_.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
        peerMemoryLayout_.Init(tilingData_->frontReorderTiling);
        localRouteMaskSlots_ =
            reinterpret_cast<__gm__ uint8_t*>(remoteWindow_.LocalBase() + peerMemoryLayout_.routeMaskSlots);
    }

    AICORE inline void ProcessFixed(uint32_t groupLocalId, uint32_t groupSize)
    {
        coreIdx_ = groupLocalId;
        coreNum_ = groupSize;
        ProcessRankSplitCopy();
    }

private:
    AICORE inline uint32_t PackedRowStride() const { return tilingData_->frontReorderTiling.packedRowStride; }

    AICORE inline uint64_t DispatchGatherPackedUbOffset(uint32_t bufferId) const
    {
        return static_cast<uint64_t>(bufferId) * tilingData_->dispatchTiling.copyBufferBytes;
    }

    AICORE inline event_t DispatchGatherBufferEvent(uint32_t bufferId) const { return static_cast<event_t>(bufferId); }

    AICORE inline uint64_t DispatchGatherMetaUbOffset(uint32_t bufferId) const
    {
        return static_cast<uint64_t>(tilingData_->dispatchTiling.bufferCount) *
                   tilingData_->dispatchTiling.copyBufferBytes +
               static_cast<uint64_t>(bufferId) * kDispatchMetaSlotBytes;
    }

    AICORE inline uint32_t RawRowsForLocalGroup(uint32_t srcRank, uint32_t groupIdx) const
    {
        const uint32_t cumulative =
            static_cast<uint32_t>(cumsumMMPtr_[static_cast<uint64_t>(srcRank) * expertPerRank_ + groupIdx]);
        return cumulative - CopyCumsumBeforeSource(srcRank, groupIdx);
    }

    AICORE inline uint32_t DispatchShardRowBegin(uint32_t rows, uint32_t shardIdx, uint32_t shardCount) const
    {
        return static_cast<uint32_t>((static_cast<uint64_t>(rows) * shardIdx) / shardCount);
    }

    AICORE inline __gm__ int32_t* ReadyCountSlot(uint32_t groupIdx, uint32_t tileIdx) const
    {
        const __gm__ MegaMoeDispatchTiling& dispatch = tilingData_->dispatchTiling;
        const uint64_t byteOffset =
            dispatch.readyCountOffset +
            static_cast<uint64_t>(groupIdx) * dispatch.readyCountMaxTilesPerExpert * kMegaMoeReadyCountSlotBytes +
            static_cast<uint64_t>(tileIdx) * kMegaMoeReadyCountSlotBytes;
        return reinterpret_cast<__gm__ int32_t*>(workspaceGM_ + byteOffset);
    }

    AICORE inline void PublishDispatchReadyRange(
        uint32_t groupIdx, uint32_t expertRowBegin, uint32_t expertRowEnd) const
    {
        if (expertRowBegin >= expertRowEnd) {
            return;
        }
        constexpr uint32_t tileM = kMegaMoeGmmTileM;
        const uint32_t firstTile = expertRowBegin / tileM;
        const uint32_t lastTile = (expertRowEnd - 1U) / tileM;
        const uint64_t scratchUbOffset = tilingData_->dispatchTiling.routeCountUbOffset;
        for (uint32_t tileIdx = firstTile; tileIdx <= lastTile; ++tileIdx) {
            const uint32_t tileBegin = tileIdx * tileM;
            const uint32_t tileEnd = tileBegin + tileM;
            const uint32_t overlapBegin = expertRowBegin > tileBegin ? expertRowBegin : tileBegin;
            const uint32_t overlapEnd = expertRowEnd < tileEnd ? expertRowEnd : tileEnd;
            const uint32_t overlapRows = overlapEnd - overlapBegin;
            PtoSetValue<int32_t, 8U>(scratchUbOffset, 0U, static_cast<int32_t>(overlapRows));
            pto::PtoSetWaitFlag<PIPE_S, PIPE_MTE3>(kDispatchReadyCountEvent, kDispatchReadyCountEvent);
            PtoStoreAtomicAddVector<int32_t, 8U>(ReadyCountSlot(groupIdx, tileIdx), scratchUbOffset, 1U);
            pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>(kDispatchReadyCountEvent, kDispatchReadyCountEvent);
        }
    }

    AICORE inline uint32_t CopyCumsumBeforeSource(uint32_t srcRank, uint32_t groupIdx) const
    {
        if (srcRank == 0U) {
            return 0U;
        }
        return static_cast<uint32_t>(cumsumMMPtr_[static_cast<uint64_t>(srcRank - 1U) * expertPerRank_ + groupIdx]);
    }

    AICORE inline void StoreDispatchPayload(uint64_t ubOffsetBytes, uint32_t dstRow) const
    {
        using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using PayloadWordGlobal = pto::GlobalTensor<uint32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;
        using PayloadWordTile = pto::Tile<
            pto::TileType::Vec, uint32_t, 1, kDispatchGatherPackedWordTileCols, pto::BLayout::RowMajor, -1, -1>;

        const uint32_t wordCols = problemK_ / sizeof(uint32_t);
        ShapeDyn payloadShape(1, 1, 1, 1, wordCols);
        StrideDyn payloadStride(wordCols, wordCols, wordCols, wordCols, 1);
        PayloadWordGlobal payloadDst(
            reinterpret_cast<__gm__ uint32_t*>(gmAPtr_ + static_cast<uint64_t>(dstRow) * problemK_), payloadShape,
            payloadStride);
        PayloadWordTile payloadTile(1, wordCols);
        pto::TASSIGN(payloadTile, ubOffsetBytes);
        pto::TSTORE(payloadDst, payloadTile);
    }

    AICORE inline void StoreDispatchScale(uint64_t ubOffsetBytes, uint32_t dstRow) const
    {
        using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using ScaleWordGlobal = pto::GlobalTensor<uint32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;
        using ScaleWordTile = pto::Tile<
            pto::TileType::Vec, uint32_t, 1, kDispatchGatherPackedWordTileCols, pto::BLayout::RowMajor, -1, -1>;

        const uint32_t scaleCols = tilingData_->frontReorderTiling.quantScaleCols;
        const uint32_t scaleWordCols = scaleCols / sizeof(uint32_t);
        ShapeDyn scaleShape(1, 1, 1, 1, scaleWordCols);
        StrideDyn scaleStride(scaleWordCols, scaleWordCols, scaleWordCols, scaleWordCols, 1);
        ScaleWordGlobal scaleDst(
            reinterpret_cast<__gm__ uint32_t*>(gmAScalePtr_ + static_cast<uint64_t>(dstRow) * scaleCols), scaleShape,
            scaleStride);
        ScaleWordTile scaleTile(1, scaleWordCols);
        pto::TASSIGN(scaleTile, ubOffsetBytes + tilingData_->frontReorderTiling.quantDataStorageBytes);
        pto::TSTORE(scaleDst, scaleTile);
    }

    AICORE inline void StoreDispatchRingToken(uint32_t bufferId, uint32_t dstRow) const
    {
        const event_t event = DispatchGatherBufferEvent(bufferId);
        const uint64_t ubOffsetBytes = DispatchGatherPackedUbOffset(bufferId);
        const uint64_t metaUbOffsetBytes = DispatchGatherMetaUbOffset(bufferId);
        wait_flag(PIPE_MTE2, PIPE_MTE3, event);
        StoreDispatchPayload(ubOffsetBytes, dstRow);
        StoreDispatchScale(ubOffsetBytes, dstRow);
        wait_flag(PIPE_S, PIPE_MTE3, event);
        PtoStoreVector<int32_t, 8U>(
            routeMetaPtr_ + static_cast<uint64_t>(dstRow) * kMegaMoeRouteMetaFields, metaUbOffsetBytes,
            kMegaMoeRouteMetaFields);
        set_flag(PIPE_MTE3, PIPE_MTE2, event);
        set_flag(PIPE_MTE3, PIPE_S, event);
    }

    template <bool IsBufferReuse>
    AICORE inline void IssueRemoteRouteToken(
        __gm__ int8_t* remoteRecords, uint32_t srcRank, uint32_t routeSlot, uint32_t bufferId) const
    {
        const uint32_t packedStride = PackedRowStride();
        const event_t event = DispatchGatherBufferEvent(bufferId);
        const uint64_t ubOffsetBytes = DispatchGatherPackedUbOffset(bufferId);
        const uint64_t metaUbOffsetBytes = DispatchGatherMetaUbOffset(bufferId);

        if constexpr (IsBufferReuse) {
            wait_flag(PIPE_MTE3, PIPE_MTE2, event);
        }
        const uint32_t token = routeSlot / topK_;
        PtoLoadVector<int8_t, kDispatchGatherPackedTileCols>(
            ubOffsetBytes, remoteRecords + static_cast<uint64_t>(token) * packedStride, packedStride);
        set_flag(PIPE_MTE2, PIPE_MTE3, event);

        if constexpr (IsBufferReuse) {
            wait_flag(PIPE_MTE3, PIPE_S, event);
        }
        PtoSetValue<int32_t, 8U>(metaUbOffsetBytes, 0U, static_cast<int32_t>(srcRank));
        PtoSetValue<int32_t, 8U>(metaUbOffsetBytes, 1U, static_cast<int32_t>(routeSlot));
        for (uint32_t field = 2U; field < kMegaMoeRouteMetaFields; ++field) {
            PtoSetValue<int32_t, 8U>(metaUbOffsetBytes, field, 0);
        }
        set_flag(PIPE_S, PIPE_MTE3, event);
    }

    AICORE inline void CopyRemoteRouteRange(
        uint32_t srcRank, uint32_t dstRowBegin, uint32_t routeIndexBegin, uint32_t routeCount) const
    {
        const uint32_t bufferCount = tilingData_->dispatchTiling.bufferCount;
        const uint64_t routeIndexUbOffset = tilingData_->dispatchTiling.routeIndexUbOffset;
        __gm__ int8_t* remoteRecords =
            reinterpret_cast<__gm__ int8_t*>(remoteWindow_.RemoteBase(0, static_cast<int32_t>(srcRank)));

        const uint32_t firstRouteSlot = PtoGetValue<uint32_t>(routeIndexUbOffset, routeIndexBegin);
        IssueRemoteRouteToken<false>(remoteRecords, srcRank, firstRouteSlot, 0U);

        const uint32_t firstUseEnd = routeCount < bufferCount ? routeCount : bufferCount;
        for (uint32_t issueIdx = 1U; issueIdx < firstUseEnd; ++issueIdx) {
            const uint32_t routeSlot = PtoGetValue<uint32_t>(routeIndexUbOffset, routeIndexBegin + issueIdx);
            IssueRemoteRouteToken<false>(remoteRecords, srcRank, routeSlot, issueIdx);
            StoreDispatchRingToken(issueIdx - 1U, dstRowBegin + issueIdx - 1U);
        }
        for (uint32_t issueIdx = bufferCount; issueIdx < routeCount; ++issueIdx) {
            const uint32_t copyIdx = issueIdx - 1U;
            const uint32_t routeSlot = PtoGetValue<uint32_t>(routeIndexUbOffset, routeIndexBegin + issueIdx);
            IssueRemoteRouteToken<true>(remoteRecords, srcRank, routeSlot, issueIdx % bufferCount);
            StoreDispatchRingToken(copyIdx % bufferCount, dstRowBegin + copyIdx);
        }
        StoreDispatchRingToken((routeCount - 1U) % bufferCount, dstRowBegin + routeCount - 1U);

        for (uint32_t bufferId = 0U; bufferId < firstUseEnd; ++bufferId) {
            const event_t event = DispatchGatherBufferEvent(bufferId);
            wait_flag(PIPE_MTE3, PIPE_MTE2, event);
            wait_flag(PIPE_MTE3, PIPE_S, event);
        }
    }

    AICORE inline __gm__ uint8_t* LocalMaskSlot(uint32_t groupIdx, uint32_t srcRank) const
    {
        const uint64_t slot = static_cast<uint64_t>(groupIdx) * rankSize_ + srcRank;
        return localRouteMaskSlots_ + slot * tilingData_->frontReorderTiling.maskSlotBytes;
    }

    AICORE inline void FetchRankGroupRows(
        uint32_t srcRank, uint32_t groupIdx, uint32_t dstRowBase, uint32_t rowBegin, uint32_t rowEnd) const
    {
        if (rowBegin == rowEnd) {
            return;
        }

        const __gm__ MegaMoeDispatchTiling& dispatch = tilingData_->dispatchTiling;
        const uint32_t batchRoutes = dispatch.routeItemsPerBatch;
        const uint32_t routeBatchCount = (routeElems_ + batchRoutes - 1U) / batchRoutes;

        __gm__ uint8_t* maskSlot = LocalMaskSlot(groupIdx, srcRank);
        uint32_t matchedOrdinal = 0U;
        for (uint32_t batchIdx = 0U; batchIdx < routeBatchCount && matchedOrdinal < rowEnd; ++batchIdx) {
            const uint32_t batchStart = batchIdx * batchRoutes;
            const uint32_t validRoutes =
                routeElems_ - batchStart > batchRoutes ? batchRoutes : routeElems_ - batchStart;
            const uint32_t maskBytes = (validRoutes + 7U) / 8U;
            PtoLoadVector<uint8_t>(dispatch.maskBufferUbOffset, maskSlot + batchStart / 8U, maskBytes);
            pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_V>(kDispatchMaskLoadEvent, kDispatchMaskLoadEvent);
            CompressDispatchRouteMask(
                dispatch.routeIndexUbOffset, dispatch.maskBufferUbOffset, batchStart, validRoutes,
                dispatch.routeCountUbOffset);
            pto::PtoSetWaitFlag<PIPE_V, PIPE_S>(kDispatchMaskVectorEvent, kDispatchMaskVectorEvent);

            const uint32_t batchMatchedRouteCount =
                PtoGetValue<uint32_t, 8U>(dispatch.routeCountUbOffset, 0U) / sizeof(uint32_t);
            const uint32_t batchMatchOrdinalBegin = matchedOrdinal;
            const uint32_t batchMatchOrdinalEnd = matchedOrdinal + batchMatchedRouteCount;
            const uint32_t dispatchMatchOrdinalBegin =
                batchMatchOrdinalBegin > rowBegin ? batchMatchOrdinalBegin : rowBegin;
            const uint32_t dispatchMatchOrdinalEnd = batchMatchOrdinalEnd < rowEnd ? batchMatchOrdinalEnd : rowEnd;
            if (dispatchMatchOrdinalEnd > dispatchMatchOrdinalBegin) {
                const uint32_t batchLocalMatchBegin = dispatchMatchOrdinalBegin - batchMatchOrdinalBegin;
                const uint32_t dispatchRowCount = dispatchMatchOrdinalEnd - dispatchMatchOrdinalBegin;
                CopyRemoteRouteRange(
                    srcRank, dstRowBase + dispatchMatchOrdinalBegin, batchLocalMatchBegin, dispatchRowCount);
            }
            matchedOrdinal = batchMatchOrdinalEnd;
        }
    }

    AICORE inline void ProcessRankSplitCopy() const
    {
        const uint32_t lanesPerRank = coreNum_ / rankSize_;
        const uint32_t srcRank = coreIdx_ / lanesPerRank;
        const uint32_t shardIdx = coreIdx_ % lanesPerRank;
        uint32_t prevGroupSum = 0U;
        for (uint32_t groupIdx = 0U; groupIdx < expertPerRank_; ++groupIdx) {
            const uint32_t currentM =
                static_cast<uint32_t>(cumsumMMPtr_[static_cast<uint64_t>(rankSize_ - 1U) * expertPerRank_ + groupIdx]);
            const uint32_t rawRows = RawRowsForLocalGroup(srcRank, groupIdx);
            const uint32_t shardRowBegin = DispatchShardRowBegin(rawRows, shardIdx, lanesPerRank);
            const uint32_t shardRowEnd = DispatchShardRowBegin(rawRows, shardIdx + 1U, lanesPerRank);
            const uint32_t sourceRowBase = CopyCumsumBeforeSource(srcRank, groupIdx);
            FetchRankGroupRows(srcRank, groupIdx, prevGroupSum + sourceRowBase, shardRowBegin, shardRowEnd);
            PublishDispatchReadyRange(groupIdx, sourceRowBase + shardRowBegin, sourceRowBase + shardRowEnd);
            prevGroupSum += currentM;
        }
    }

    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;
    __gm__ int8_t* gmAPtr_ = nullptr;
    __gm__ uint8_t* gmAScalePtr_ = nullptr;
    __gm__ int32_t* routeMetaPtr_ = nullptr;
    __gm__ uint8_t* localRouteMaskSlots_ = nullptr;
    __gm__ int32_t* cumsumMMPtr_ = nullptr;

    PtoRemoteWindow remoteWindow_;
    MegaMoePeerMemoryLayout peerMemoryLayout_;

    uint32_t problemK_ = 0;
    uint32_t topK_ = 0;
    uint32_t routeElems_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t rankSize_ = 0;
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
};

#endif // DISPATCH_MEGA_COMBINE_DISPATCH_H
