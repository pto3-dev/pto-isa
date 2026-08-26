/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_FRONT_REORDER_H
#define DISPATCH_MEGA_COMBINE_FRONT_REORDER_H

#include "kernel_operator.h"

#include <type_traits>

#include <pto/pto-inst.hpp>

#include "dispatch_mega_combine_tiling.h"
#include "utils/common_helpers.hpp"
#include "utils/const_args.hpp"
#include "utils/hccl_window.hpp"
#include "utils/mega_expert_sync.hpp"
#include "utils/pto_vector.hpp"

constexpr uint32_t kFrontMaskMaxBatchRoutes = 2048U;
constexpr uint32_t kFrontMaskMaxBatchBytes = kFrontMaskMaxBatchRoutes / 8U;
constexpr uint32_t kFrontBufferNum = 2U;
constexpr uint32_t kFrontQuantMaxCols = 8192U;
constexpr uint32_t kFrontQuantMaxScaleCols = kFrontQuantMaxCols / kMegaMoeMxGroupSize;
constexpr uint32_t kFrontMaskRoutesPerStoreBlock = kMegaMoeFrontMaskCountRecordBytes * 8U;
constexpr uint32_t kFrontMaskCountRecordElems = kMegaMoeFrontMaskCountRecordBytes / sizeof(int32_t);
constexpr uint64_t kFrontMaskSrcUbBase = 0U;
constexpr uint64_t kFrontMaskSrcUbStride = static_cast<uint64_t>(kFrontMaskMaxBatchRoutes) * sizeof(int32_t);
constexpr uint64_t kFrontMaskUbBase = kFrontBufferNum * kFrontMaskSrcUbStride;
constexpr uint64_t kFrontMaskUbStride = kFrontMaskMaxBatchBytes;
constexpr uint64_t kFrontMaskCountUb = kFrontMaskUbBase + kFrontBufferNum * kFrontMaskUbStride;

using FrontMaskSrcTile =
    pto::Tile<pto::TileType::Vec, int32_t, 1, kFrontMaskMaxBatchRoutes, pto::BLayout::RowMajor, -1, -1>;
using FrontMaskDstTile =
    pto::Tile<pto::TileType::Vec, uint8_t, 1, kFrontMaskMaxBatchBytes, pto::BLayout::RowMajor, -1, -1>;
using FrontQuantSrcTile = pto::Tile<
    pto::TileType::Vec, bfloat16_t, 1, kFrontQuantMaxCols, pto::BLayout::RowMajor, -1, -1, pto::SLayout::NoneBox, 512,
    pto::PadValue::Zero>;
using FrontQuantFp8Tile = pto::Tile<
    pto::TileType::Vec, int8_t, 1, kFrontQuantMaxCols, pto::BLayout::RowMajor, -1, -1, pto::SLayout::NoneBox, 512,
    pto::PadValue::Zero>;
using FrontQuantE8Tile = pto::Tile<
    pto::TileType::Vec, uint8_t, 1, kFrontQuantMaxScaleCols, pto::BLayout::RowMajor, -1, -1, pto::SLayout::NoneBox, 512,
    pto::PadValue::Zero>;
using FrontQuantBf16ScaleTile =
    pto::Tile<pto::TileType::Vec, bfloat16_t, 1, kFrontQuantMaxScaleCols, pto::BLayout::RowMajor, -1, -1>;

static_assert(kMegaMoeFrontMaskCountRecordBytes == UB_ALIGN);
static_assert(kFrontMaskMaxBatchRoutes % kFrontMaskRoutesPerStoreBlock == 0U);

struct FrontQuantUbPlan {
    uint64_t raw = 0U;
    uint64_t fp8 = 0U;
    uint64_t e8 = 0U;
    uint64_t max = 0U;
    uint64_t scaling = 0U;
};

AICORE inline uint32_t FrontMaskPopcount(uint8_t value)
{
    uint32_t count = 0U;
    while (value != 0U) {
        count += static_cast<uint32_t>(value & 1U);
        value >>= 1U;
    }
    return count;
}

template <typename InputElement>
class FrontMaskPull {
public:
    AICORE inline void Init(
        GM_ADDR xGM, GM_ADDR expertIdGM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM,
        const __gm__ MegaMoeTilingData* tilingData)
    {
        static_assert(std::is_same_v<InputElement, bfloat16_t>, "MXFP8 Front requires BF16 input");
        xPtr_ = reinterpret_cast<__gm__ InputElement*>(xGM);
        expertIdPtr_ = reinterpret_cast<__gm__ int32_t*>(expertIdGM);
        expertTokenNumsPtr_ = reinterpret_cast<__gm__ int32_t*>(expertTokenNumsGM);
        workspaceGM_ = workspaceGM;
        tilingData_ = tilingData;

        const __gm__ MegaMoeInfo& info = tilingData_->megaMoeInfo;
        problemM_ = info.M;
        problemK_ = info.K;
        expertPerRank_ = info.expertPerRank;
        expertNum_ = tilingData_->frontReorderTiling.expertNum;
        routeElems_ = tilingData_->frontReorderTiling.routeElems;
        rank_ = tilingData_->runtimeInfo.rank;
        rankSize_ = tilingData_->runtimeInfo.rankSize;
        coreIdx_ = get_block_idx() + get_subblockid() * get_block_num();
        coreNum_ = get_block_num() * get_subblockdim();

        remoteWindow_.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
        peerMemoryLayout_.Init(tilingData_->frontReorderTiling);
        sourceTokenRecords_ = reinterpret_cast<__gm__ int8_t*>(remoteWindow_.LocalBase());
        localRouteMaskSlots_ =
            reinterpret_cast<__gm__ uint8_t*>(remoteWindow_.LocalBase() + peerMemoryLayout_.routeMaskSlots);
        cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t*>(workspaceGM_ + tilingData_->frontReorderTiling.cumsumMMOffset);
    }

    AICORE inline void Process()
    {
        if (coreIdx_ == 0U) {
            remoteWindow_.PrepareFrontReadyEpoch();
        }
        pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
        const int32_t frontReadyEpoch = remoteWindow_.FrontReadyEpoch();

        QuantizeSourceTokens();
        pto::SYNCALL<pto::SyncCoreType::AIVOnly>();

        BuildAndPushRouteMasks();
        dsb(DSB_DDR);
        pto::SYNCALL<pto::SyncCoreType::AIVOnly>();

        if (coreIdx_ == 0U) {
            for (uint32_t dstRank = 0U; dstRank < rankSize_; ++dstRank) {
                remoteWindow_.PublishFrontReady(static_cast<int32_t>(dstRank), frontReadyEpoch);
            }
            remoteWindow_.WaitAllFrontReady(frontReadyEpoch);
            BuildCumsumAndExpertTokenNums();
            dsb(DSB_DDR);
            PublishScalarEpoch(
                FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncFrontMetadataReadySlot),
                kMegaMoeFixedFrontMetadataReadyMarker);
        }
        pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
    }

private:
    AICORE inline uint32_t PackedRowStride() const { return tilingData_->frontReorderTiling.packedRowStride; }

    AICORE inline event_t FrontBufferEvent(uint32_t bufferId) const { return static_cast<event_t>(bufferId); }

    AICORE inline __gm__ uint8_t* LocalMaskSlot(uint32_t localExpert, uint32_t sourceRank) const
    {
        const uint64_t slot = static_cast<uint64_t>(localExpert) * rankSize_ + sourceRank;
        return localRouteMaskSlots_ + slot * tilingData_->frontReorderTiling.maskSlotBytes;
    }

    AICORE inline __gm__ uint8_t* RemoteMaskSlot(uint32_t globalExpert) const
    {
        const uint32_t dstRank = globalExpert / expertPerRank_;
        const uint32_t localExpert = globalExpert - dstRank * expertPerRank_;
        __gm__ uint8_t* remoteBase = reinterpret_cast<__gm__ uint8_t*>(
            remoteWindow_.RemoteBase(peerMemoryLayout_.routeMaskSlots, static_cast<int32_t>(dstRank)));
        const uint64_t slot = static_cast<uint64_t>(localExpert) * rankSize_ + rank_;
        return remoteBase + slot * tilingData_->frontReorderTiling.maskSlotBytes;
    }

    AICORE inline uint32_t FrontExpertCoreBegin(uint32_t globalExpert) const
    {
        return static_cast<uint32_t>((static_cast<uint64_t>(globalExpert) * coreNum_ + expertNum_ - 1U) / expertNum_);
    }

    AICORE inline uint32_t FrontExpertCoreEnd(uint32_t globalExpert) const
    {
        return static_cast<uint32_t>(
            (static_cast<uint64_t>(globalExpert + 1U) * coreNum_ + expertNum_ - 1U) / expertNum_);
    }

    AICORE inline uint32_t AllocatedMaskLaneCount(uint32_t globalExpert) const
    {
        if (coreNum_ < expertNum_) {
            return 1U;
        }
        return FrontExpertCoreEnd(globalExpert) - FrontExpertCoreBegin(globalExpert);
    }

    AICORE inline uint32_t ActiveMaskLaneCount(uint32_t globalExpert) const
    {
        const uint32_t allocatedLanes = AllocatedMaskLaneCount(globalExpert);
        const uint32_t maskBlockCount = tilingData_->frontReorderTiling.maskBytes / kMegaMoeFrontMaskCountRecordBytes;
        return allocatedLanes < maskBlockCount ? allocatedLanes : maskBlockCount;
    }

    AICORE inline uint64_t FrontQuantUbBytesPerBuffer() const
    {
        const uint64_t scaleCols = tilingData_->frontReorderTiling.quantScaleCols;
        uint64_t bytes = alignUp(static_cast<uint64_t>(problemK_) * sizeof(InputElement), UB_ALIGN);
        bytes += alignUp(static_cast<uint64_t>(problemK_), UB_ALIGN);
        bytes += alignUp(scaleCols, UB_ALIGN);
        bytes += 2U * alignUp(scaleCols * sizeof(bfloat16_t), UB_ALIGN);
        return bytes;
    }

    AICORE inline FrontQuantUbPlan FrontQuantUbPlanForInput(uint32_t bufferId) const
    {
        const uint64_t base = static_cast<uint64_t>(bufferId) * FrontQuantUbBytesPerBuffer();
        const uint64_t scaleCols = tilingData_->frontReorderTiling.quantScaleCols;
        FrontQuantUbPlan plan;
        plan.raw = base;
        plan.fp8 = plan.raw + alignUp(static_cast<uint64_t>(problemK_) * sizeof(InputElement), UB_ALIGN);
        plan.e8 = plan.fp8 + alignUp(static_cast<uint64_t>(problemK_), UB_ALIGN);
        plan.max = plan.e8 + alignUp(scaleCols, UB_ALIGN);
        plan.scaling = plan.max + alignUp(scaleCols * sizeof(bfloat16_t), UB_ALIGN);
        return plan;
    }

    AICORE inline void PrepareFrontQuantEvents() const
    {
        for (uint32_t bufferId = 0U; bufferId < kFrontBufferNum; ++bufferId) {
            const event_t event = FrontBufferEvent(bufferId);
            set_flag(PIPE_V, PIPE_MTE2, event);
            set_flag(PIPE_MTE3, PIPE_V, event);
        }
    }

    AICORE inline void PrefetchSourceToken(uint32_t token, uint32_t bufferId) const
    {
        const FrontQuantUbPlan plan = FrontQuantUbPlanForInput(bufferId);
        const event_t event = FrontBufferEvent(bufferId);
        wait_flag(PIPE_V, PIPE_MTE2, event);
        PtoLoadVector<InputElement>(plan.raw, xPtr_ + static_cast<uint64_t>(token) * problemK_, problemK_);
        set_flag(PIPE_MTE2, PIPE_V, event);
    }

    AICORE inline void QuantizeAndStoreSourceToken(uint32_t token, uint32_t bufferId) const
    {
        const FrontQuantUbPlan plan = FrontQuantUbPlanForInput(bufferId);
        const event_t event = FrontBufferEvent(bufferId);
        wait_flag(PIPE_MTE2, PIPE_V, event);
        wait_flag(PIPE_MTE3, PIPE_V, event);

        const uint32_t scaleCols = tilingData_->frontReorderTiling.quantScaleCols;
        FrontQuantSrcTile srcTile(1U, problemK_);
        FrontQuantFp8Tile fp8Tile(1U, problemK_);
        FrontQuantE8Tile e8Tile(1U, scaleCols);
        FrontQuantBf16ScaleTile maxTile(1U, scaleCols);
        FrontQuantBf16ScaleTile scalingTile(1U, scaleCols);
        pto::TASSIGN(srcTile, plan.raw);
        pto::TASSIGN(fp8Tile, plan.fp8);
        pto::TASSIGN(e8Tile, plan.e8);
        pto::TASSIGN(maxTile, plan.max);
        pto::TASSIGN(scalingTile, plan.scaling);
        pto::TQUANT<
            pto::QuantType::MXFP8, FrontQuantFp8Tile, FrontQuantSrcTile, FrontQuantE8Tile, FrontQuantBf16ScaleTile,
            FrontQuantBf16ScaleTile, pto::QuantScaleAlg::OCP>(fp8Tile, srcTile, &e8Tile, &maxTile, &scalingTile);

        set_flag(PIPE_V, PIPE_MTE2, event);
        set_flag(PIPE_V, PIPE_MTE3, event);
        wait_flag(PIPE_V, PIPE_MTE3, event);
        __gm__ int8_t* record = sourceTokenRecords_ + static_cast<uint64_t>(token) * PackedRowStride();
        PtoStoreVector<int8_t>(record, plan.fp8, problemK_);
        PtoStoreVector<uint8_t>(
            reinterpret_cast<__gm__ uint8_t*>(record) + tilingData_->frontReorderTiling.quantDataStorageBytes, plan.e8,
            scaleCols);
        set_flag(PIPE_MTE3, PIPE_V, event);
    }

    AICORE inline void FinalizeFrontQuantEvents() const
    {
        for (uint32_t bufferId = 0U; bufferId < kFrontBufferNum; ++bufferId) {
            const event_t event = FrontBufferEvent(bufferId);
            wait_flag(PIPE_V, PIPE_MTE2, event);
            wait_flag(PIPE_MTE3, PIPE_V, event);
        }
    }

    AICORE inline void QuantizeSourceTokens() const
    {
        const uint32_t tokenBegin = static_cast<uint32_t>((static_cast<uint64_t>(problemM_) * coreIdx_) / coreNum_);
        const uint32_t tokenEnd =
            static_cast<uint32_t>((static_cast<uint64_t>(problemM_) * (coreIdx_ + 1U)) / coreNum_);
        const uint32_t tokenCount = tokenEnd - tokenBegin;
        PrepareFrontQuantEvents();
        const uint32_t preloadCount = tokenCount < kFrontBufferNum ? tokenCount : kFrontBufferNum;
        for (uint32_t offset = 0U; offset < preloadCount; ++offset) {
            PrefetchSourceToken(tokenBegin + offset, offset);
        }
        for (uint32_t offset = 0U; offset < tokenCount; ++offset) {
            const uint32_t bufferId = offset % kFrontBufferNum;
            QuantizeAndStoreSourceToken(tokenBegin + offset, bufferId);
            const uint32_t nextOffset = offset + kFrontBufferNum;
            if (nextOffset < tokenCount) {
                PrefetchSourceToken(tokenBegin + nextOffset, bufferId);
            }
        }
        FinalizeFrontQuantEvents();
    }

    AICORE inline uint64_t FrontMaskSrcUb(uint32_t bufferId) const
    {
        return kFrontMaskSrcUbBase + static_cast<uint64_t>(bufferId) * kFrontMaskSrcUbStride;
    }

    AICORE inline uint64_t FrontMaskUb(uint32_t bufferId) const
    {
        return kFrontMaskUbBase + static_cast<uint64_t>(bufferId) * kFrontMaskUbStride;
    }

    AICORE inline void PrepareFrontMaskEvents() const
    {
        for (uint32_t bufferId = 0U; bufferId < kFrontBufferNum; ++bufferId) {
            const event_t event = FrontBufferEvent(bufferId);
            set_flag(PIPE_V, PIPE_MTE2, event);
            set_flag(PIPE_MTE3, PIPE_V, event);
        }
    }

    AICORE inline void PrefetchFrontMaskSource(uint32_t batchStart, uint32_t validRoutes, uint32_t bufferId) const
    {
        const event_t event = FrontBufferEvent(bufferId);
        wait_flag(PIPE_V, PIPE_MTE2, event);
        PtoLoadVector<int32_t, kFrontMaskMaxBatchRoutes>(
            FrontMaskSrcUb(bufferId), expertIdPtr_ + batchStart, validRoutes);
        set_flag(PIPE_MTE2, PIPE_V, event);
    }

    AICORE inline uint32_t CompareAndPushFrontMask(
        uint32_t globalExpert, __gm__ uint8_t* remoteSlot, uint32_t batchStart, uint32_t validRoutes,
        uint32_t storeMaskBytes, uint32_t bufferId) const
    {
        const event_t event = FrontBufferEvent(bufferId);
        const uint64_t srcUb = FrontMaskSrcUb(bufferId);
        const uint64_t maskUb = FrontMaskUb(bufferId);
        const uint32_t validMaskBytes = (validRoutes + 7U) / 8U;
        wait_flag(PIPE_MTE2, PIPE_V, event);
        wait_flag(PIPE_MTE3, PIPE_V, event);
        PtoFillUb<uint8_t, kFrontMaskMaxBatchBytes>(maskUb, 0U, storeMaskBytes);

        FrontMaskSrcTile srcTile(1U, validRoutes);
        FrontMaskDstTile dstTile(1U, validMaskBytes);
        pto::TASSIGN(srcTile, srcUb);
        pto::TASSIGN(dstTile, maskUb);
        pto::TCMPS(dstTile, srcTile, static_cast<int32_t>(globalExpert), pto::CmpMode::EQ);
        set_flag(PIPE_V, PIPE_MTE2, event);
        set_flag(PIPE_V, PIPE_S, event);
        set_flag(PIPE_V, PIPE_MTE3, event);
        wait_flag(PIPE_V, PIPE_MTE3, event);
        PtoStoreVector<uint8_t, kFrontMaskMaxBatchBytes>(remoteSlot + batchStart / 8U, maskUb, storeMaskBytes);
        set_flag(PIPE_MTE3, PIPE_V, event);

        wait_flag(PIPE_V, PIPE_S, event);
        uint32_t matchedCount = 0U;
        for (uint32_t byte = 0U; byte < validMaskBytes; ++byte) {
            matchedCount += FrontMaskPopcount(PtoGetValue<uint8_t, kFrontMaskMaxBatchBytes>(maskUb, byte));
        }
        return matchedCount;
    }

    AICORE inline void FinalizeFrontMaskEvents() const
    {
        for (uint32_t bufferId = 0U; bufferId < kFrontBufferNum; ++bufferId) {
            const event_t event = FrontBufferEvent(bufferId);
            wait_flag(PIPE_V, PIPE_MTE2, event);
            wait_flag(PIPE_MTE3, PIPE_V, event);
        }
    }

    AICORE inline void StoreFrontMaskLaneCount(
        __gm__ uint8_t* remoteSlot, uint32_t laneIdx, uint32_t matchedCount) const
    {
        PtoFillUb<int32_t, kFrontMaskCountRecordElems>(kFrontMaskCountUb, 0, kFrontMaskCountRecordElems);
        pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();
        PtoSetValue<int32_t, kFrontMaskCountRecordElems>(kFrontMaskCountUb, 0U, static_cast<int32_t>(matchedCount));
        pto::PtoSetWaitFlag<PIPE_S, PIPE_MTE3>();
        __gm__ int32_t* countRecord = reinterpret_cast<__gm__ int32_t*>(
            remoteSlot + tilingData_->frontReorderTiling.maskBytes +
            static_cast<uint64_t>(laneIdx) * kMegaMoeFrontMaskCountRecordBytes);
        PtoStoreVector<int32_t, kFrontMaskCountRecordElems>(countRecord, kFrontMaskCountUb, kFrontMaskCountRecordElems);
        pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_V>();
    }

    AICORE inline void BuildAndPushRouteMaskLane(uint32_t globalExpert, uint32_t laneIdx, uint32_t laneCount) const
    {
        const uint32_t maskBlockCount = tilingData_->frontReorderTiling.maskBytes / kMegaMoeFrontMaskCountRecordBytes;
        const uint32_t laneBlockBegin =
            static_cast<uint32_t>((static_cast<uint64_t>(maskBlockCount) * laneIdx) / laneCount);
        const uint32_t laneBlockEnd =
            static_cast<uint32_t>((static_cast<uint64_t>(maskBlockCount) * (laneIdx + 1U)) / laneCount);
        const uint32_t laneRouteBegin = laneBlockBegin * kFrontMaskRoutesPerStoreBlock;
        const uint32_t unboundedRouteEnd = laneBlockEnd * kFrontMaskRoutesPerStoreBlock;
        const uint32_t laneRouteEnd = unboundedRouteEnd < routeElems_ ? unboundedRouteEnd : routeElems_;
        const uint32_t laneMaskByteEnd = laneBlockEnd * kMegaMoeFrontMaskCountRecordBytes;
        constexpr uint32_t batchRoutes = kMegaMoeFrontMaskRouteItemsPerBatch;
        const uint32_t laneRouteCount = laneRouteEnd - laneRouteBegin;
        const uint32_t batchCount = static_cast<uint32_t>(ceilDiv(laneRouteCount, batchRoutes));
        __gm__ uint8_t* remoteSlot = RemoteMaskSlot(globalExpert);

        uint32_t matchedCount = 0U;
        PrepareFrontMaskEvents();
        const uint32_t preloadCount = batchCount < kFrontBufferNum ? batchCount : kFrontBufferNum;
        for (uint32_t batchIdx = 0U; batchIdx < preloadCount; ++batchIdx) {
            const uint32_t batchStart = laneRouteBegin + batchIdx * batchRoutes;
            const uint32_t validRoutes =
                laneRouteEnd - batchStart > batchRoutes ? batchRoutes : laneRouteEnd - batchStart;
            PrefetchFrontMaskSource(batchStart, validRoutes, batchIdx);
        }
        for (uint32_t batchIdx = 0U; batchIdx < batchCount; ++batchIdx) {
            const uint32_t bufferId = batchIdx % kFrontBufferNum;
            const uint32_t batchStart = laneRouteBegin + batchIdx * batchRoutes;
            const uint32_t validRoutes =
                laneRouteEnd - batchStart > batchRoutes ? batchRoutes : laneRouteEnd - batchStart;
            const uint32_t batchMaskByteBegin = batchStart / 8U;
            const uint32_t remainingMaskBytes = laneMaskByteEnd - batchMaskByteBegin;
            const uint32_t fullBatchMaskBytes = batchRoutes / 8U;
            const uint32_t storeMaskBytes =
                remainingMaskBytes < fullBatchMaskBytes ? remainingMaskBytes : fullBatchMaskBytes;
            matchedCount +=
                CompareAndPushFrontMask(globalExpert, remoteSlot, batchStart, validRoutes, storeMaskBytes, bufferId);
            const uint32_t nextBatchIdx = batchIdx + kFrontBufferNum;
            if (nextBatchIdx < batchCount) {
                const uint32_t nextBatchStart = laneRouteBegin + nextBatchIdx * batchRoutes;
                const uint32_t nextValidRoutes =
                    laneRouteEnd - nextBatchStart > batchRoutes ? batchRoutes : laneRouteEnd - nextBatchStart;
                PrefetchFrontMaskSource(nextBatchStart, nextValidRoutes, bufferId);
            }
        }
        FinalizeFrontMaskEvents();
        StoreFrontMaskLaneCount(remoteSlot, laneIdx, matchedCount);
    }

    AICORE inline void BuildAndPushRouteMasks() const
    {
        if (coreNum_ >= expertNum_) {
            const uint32_t globalExpert =
                static_cast<uint32_t>((static_cast<uint64_t>(coreIdx_) * expertNum_) / coreNum_);
            const uint32_t expertCoreBegin = FrontExpertCoreBegin(globalExpert);
            const uint32_t laneIdx = coreIdx_ - expertCoreBegin;
            const uint32_t laneCount = ActiveMaskLaneCount(globalExpert);
            if (laneIdx < laneCount) {
                BuildAndPushRouteMaskLane(globalExpert, laneIdx, laneCount);
            }
            return;
        }

        for (uint32_t globalExpert = coreIdx_; globalExpert < expertNum_; globalExpert += coreNum_) {
            BuildAndPushRouteMaskLane(globalExpert, 0U, 1U);
        }
    }

    AICORE inline void BuildCumsumAndExpertTokenNums() const
    {
        constexpr uint32_t maxCountRecordElems =
            kMegaMoeExpertProgressMaxRanks * kMegaMoeFixedPhysicalAivNum * kFrontMaskCountRecordElems;
        constexpr uint32_t maxCumsumElems = kMegaMoeExpertProgressMaxRanks * kMegaMoeFixedMaxExperts;
        constexpr uint32_t expertTokenNumsTileElems = (kMegaMoeFixedMaxExperts + kFrontMaskCountRecordElems - 1U) /
                                                      kFrontMaskCountRecordElems * kFrontMaskCountRecordElems;
        constexpr uint64_t countRecordsUb = 0U;
        constexpr uint64_t cumsumUb = alignUp(static_cast<uint64_t>(maxCountRecordElems) * sizeof(int32_t), UB_ALIGN);
        constexpr uint64_t expertTokenNumsUb =
            cumsumUb + alignUp(static_cast<uint64_t>(maxCumsumElems) * sizeof(int32_t), UB_ALIGN);
        constexpr uint64_t cumsumEndUb =
            expertTokenNumsUb + static_cast<uint64_t>(expertTokenNumsTileElems) * sizeof(int32_t);
        static_assert(cumsumEndUb <= A5_MAIN_UB_SIZE);

        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        for (uint32_t localExpert = 0U; localExpert < expertPerRank_; ++localExpert) {
            wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
            const uint32_t globalExpert = rank_ * expertPerRank_ + localExpert;
            const uint32_t laneCount = ActiveMaskLaneCount(globalExpert);
            for (uint32_t sourceRank = 0U; sourceRank < rankSize_; ++sourceRank) {
                for (uint32_t laneIdx = 0U; laneIdx < laneCount; ++laneIdx) {
                    const uint32_t recordIdx = sourceRank * kMegaMoeFixedPhysicalAivNum + laneIdx;
                    __gm__ int32_t* countPtr = reinterpret_cast<__gm__ int32_t*>(
                        LocalMaskSlot(localExpert, sourceRank) + tilingData_->frontReorderTiling.maskBytes +
                        static_cast<uint64_t>(laneIdx) * kMegaMoeFrontMaskCountRecordBytes);
                    PtoLoadVector<int32_t, kFrontMaskCountRecordElems>(
                        countRecordsUb + static_cast<uint64_t>(recordIdx) * kMegaMoeFrontMaskCountRecordBytes, countPtr,
                        kFrontMaskCountRecordElems);
                }
            }
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            uint32_t cumulative = 0U;
            for (uint32_t sourceRank = 0U; sourceRank < rankSize_; ++sourceRank) {
                uint32_t sourceCount = 0U;
                for (uint32_t laneIdx = 0U; laneIdx < laneCount; ++laneIdx) {
                    const uint32_t recordIdx = sourceRank * kMegaMoeFixedPhysicalAivNum + laneIdx;
                    sourceCount += static_cast<uint32_t>(PtoGetValue<int32_t, kFrontMaskCountRecordElems>(
                        countRecordsUb + static_cast<uint64_t>(recordIdx) * kMegaMoeFrontMaskCountRecordBytes, 0U));
                }
                cumulative += sourceCount;
                PtoSetValue<int32_t, maxCumsumElems>(
                    cumsumUb, sourceRank * expertPerRank_ + localExpert, static_cast<int32_t>(cumulative));
            }
            PtoSetValue<int32_t, expertTokenNumsTileElems>(
                expertTokenNumsUb, localExpert, static_cast<int32_t>(cumulative));
            set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        }
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        pto::PtoSetWaitFlag<PIPE_S, PIPE_MTE3>();
        PtoStoreVector<int32_t, maxCumsumElems>(cumsumMMPtr_, cumsumUb, rankSize_ * expertPerRank_);
        PtoStoreVector<int32_t, expertTokenNumsTileElems>(expertTokenNumsPtr_, expertTokenNumsUb, expertPerRank_);
        pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
    }

    __gm__ InputElement* xPtr_ = nullptr;
    __gm__ int32_t* expertIdPtr_ = nullptr;
    __gm__ int32_t* expertTokenNumsPtr_ = nullptr;
    __gm__ int8_t* sourceTokenRecords_ = nullptr;
    __gm__ uint8_t* localRouteMaskSlots_ = nullptr;
    __gm__ int32_t* cumsumMMPtr_ = nullptr;
    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;
    PtoRemoteWindow remoteWindow_;
    MegaMoePeerMemoryLayout peerMemoryLayout_;

    uint32_t problemM_ = 0U;
    uint32_t problemK_ = 0U;
    uint32_t expertPerRank_ = 0U;
    uint32_t expertNum_ = 0U;
    uint32_t routeElems_ = 0U;
    uint32_t rank_ = 0U;
    uint32_t rankSize_ = 0U;
    uint32_t coreIdx_ = 0U;
    uint32_t coreNum_ = 1U;
};

template <typename InputElement>
AICORE inline void FrontReorderProcess(
    GM_ADDR xGM, GM_ADDR expertIdGM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM,
    const __gm__ MegaMoeTilingData* tilingData)
{
    FrontMaskPull<InputElement> front;
    front.Init(xGM, expertIdGM, expertTokenNumsGM, workspaceGM, tilingData);
    front.Process();
}

#endif // DISPATCH_MEGA_COMBINE_FRONT_REORDER_H
