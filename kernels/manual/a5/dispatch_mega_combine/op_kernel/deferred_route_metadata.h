/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_DEFERRED_ROUTE_METADATA_H
#define DISPATCH_MEGA_COMBINE_DEFERRED_ROUTE_METADATA_H

#include "kernel_operator.h"

#include <pto/pto-inst.hpp>

#include "dispatch_mega_combine_tiling.h"
#include "front_metadata_sort.h"
#include "gmm_task_descriptor_builder.h"
#include "utils/const_args.hpp"
#include "utils/hccl_window.hpp"
#include "utils/mega_expert_sync.hpp"
#include "utils/pto_vector.hpp"

namespace deferred_route_metadata {

constexpr uint32_t kCountRecordElems = kMegaMoeFrontMaskCountRecordBytes / sizeof(int32_t);
constexpr uint32_t kInverseBlockElems = UB_ALIGN / sizeof(int32_t);
constexpr uint32_t kInverseRowsPerLoop = 1024U;
constexpr uint32_t kInverseAssistElems = 256U;
constexpr uint32_t kInverseRowsPerAssist = 32U;
constexpr uint64_t kPreSumPrefixUb = 0U;
constexpr uint64_t kPreSumCountUb = 2U * UB_ALIGN;
constexpr uint64_t kInverseInputUb = 0U;
constexpr uint64_t kInverseOutputUb = kInverseInputUb + static_cast<uint64_t>(kInverseRowsPerLoop) * sizeof(int32_t);
constexpr uint64_t kInverseAssistUb =
    kInverseOutputUb + static_cast<uint64_t>(kInverseRowsPerLoop) * kInverseBlockElems * sizeof(int32_t);
constexpr uint64_t kInverseUbEnd = kInverseAssistUb + static_cast<uint64_t>(kInverseAssistElems) * sizeof(int32_t);

static_assert(kCountRecordElems == kInverseBlockElems);
static_assert(kInverseUbEnd <= A5_MAIN_UB_SIZE);

class GroupBarrier {
public:
    AICORE inline void Init(
        GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData, uint32_t workerIdx, uint32_t workerCount)
    {
        workspaceGM_ = workspaceGM;
        tilingData_ = tilingData;
        workerIdx_ = workerIdx;
        workerCount_ = workerCount;
    }

    AICORE inline void Sync()
    {
        const uint32_t phase = phase_++;
        NotifyGroupConsumersMte(
            workspaceGM_, tilingData_, kMegaMoeFixedSyncDeferredMetadataArrivalBase,
            kMegaMoeFixedSyncDeferredMetadataReadyBase, workerCount_, workerCount_, workerIdx_, 0U, phase);
        WaitEpochAcquire(
            FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncDeferredMetadataReadyBase + workerIdx_),
            static_cast<int32_t>(phase + 1U));
    }

private:
    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;
    uint32_t workerIdx_ = 0U;
    uint32_t workerCount_ = 0U;
    uint32_t phase_ = 0U;
};

class DeferredRouteMetadata {
public:
    AICORE inline void Init(
        GM_ADDR expertIdGM, GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData, uint32_t workerIdx,
        uint32_t workerCount)
    {
        expertIdPtr_ = reinterpret_cast<__gm__ int32_t*>(expertIdGM);
        workspaceGM_ = workspaceGM;
        tilingData_ = tilingData;
        workerIdx_ = workerIdx;
        workerCount_ = workerCount;

        const __gm__ MegaMoeFrontReorderTiling& front = tilingData_->frontReorderTiling;
        routeElems_ = front.routeElems;
        expertNum_ = front.expertNum;
        rank_ = tilingData_->runtimeInfo.rank;
        rankSize_ = tilingData_->runtimeInfo.rankSize;
        expertPerRank_ = tilingData_->megaMoeInfo.expertPerRank;
        sortedRouteSlotPtr_ = reinterpret_cast<__gm__ int32_t*>(workspaceGM_ + front.sortedRouteSlotOffset);
        expandedRowIdxPtr_ = reinterpret_cast<__gm__ int32_t*>(workspaceGM_ + front.expandedRowIdxOffset);

        remoteWindow_.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
        peerMemoryLayout_.Init(front);
        barrier_.Init(workspaceGM_, tilingData_, workerIdx_, workerCount_);
    }

    AICORE inline void Run()
    {
        const bool gmm1Mailbox =
            tilingData_->gmmSchedulerTiling.gmm1ScheduleMode == kMegaMoeGmm1ScheduleWave0MailboxSuffix;
        ParallelGmmTaskDescriptorBuilder descriptorBuilder;
        descriptorBuilder.Init(workspaceGM_, tilingData_, workerIdx_, workerCount_);
        if (gmm1Mailbox) {
            const uint32_t gmm1TaskCount = descriptorBuilder.BuildGmm1();
            barrier_.Sync();
            if (workerIdx_ == 0U) {
                descriptorBuilder.PublishGeneratedTail(tilingData_->gmmSchedulerTiling.gmm1, gmm1TaskCount);
            }
        }
        const uint32_t gmm2TaskCount = descriptorBuilder.BuildGmm2();
        barrier_.Sync();
        if (workerIdx_ == 0U) {
            descriptorBuilder.PublishGeneratedTail(tilingData_->gmmSchedulerTiling.gmm2, gmm2TaskCount);
        }
        // Core 0 owns the single-run sort used by small-M cases. Run preSum on
        // the last worker so both metadata paths overlap at the sort barrier.
        const uint32_t preSumWorker = workerCount_ - 1U;
        if (workerIdx_ == preSumWorker) {
            BuildAndPublishPreSum();
        }

        front_metadata_sort::Sorter sorter;
        sorter.Init(expertIdPtr_, workspaceGM_, &tilingData_->frontReorderTiling, workerIdx_, workerCount_);
        sorter.Run(barrier_);

        BuildExpandedRowIdx();
        barrier_.Sync();
        if (workerIdx_ == 0U) {
            PublishScalarEpoch(
                FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncDeferredExpandedReadySlot),
                kMegaMoeFixedDeferredExpandedReadyMarker);
        }
    }

private:
    AICORE inline uint32_t FrontAivCount() const
    {
        return tilingData_->fixedGroupTiling.physicalAicNum * kMegaMoeFixedAivSubblocksPerPhysicalBlock;
    }

    AICORE inline uint32_t FrontExpertCoreBegin(uint32_t globalExpert) const
    {
        return static_cast<uint32_t>(
            (static_cast<uint64_t>(globalExpert) * FrontAivCount() + expertNum_ - 1U) / expertNum_);
    }

    AICORE inline uint32_t FrontExpertCoreEnd(uint32_t globalExpert) const
    {
        return static_cast<uint32_t>(
            (static_cast<uint64_t>(globalExpert + 1U) * FrontAivCount() + expertNum_ - 1U) / expertNum_);
    }

    AICORE inline uint32_t ActiveMaskLaneCount(uint32_t globalExpert) const
    {
        uint32_t allocatedLanes = 1U;
        if (FrontAivCount() >= expertNum_) {
            allocatedLanes = FrontExpertCoreEnd(globalExpert) - FrontExpertCoreBegin(globalExpert);
        }
        const uint32_t maskBlockCount = tilingData_->frontReorderTiling.maskBytes / kMegaMoeFrontMaskCountRecordBytes;
        return allocatedLanes < maskBlockCount ? allocatedLanes : maskBlockCount;
    }

    AICORE inline __gm__ int32_t* RemoteMaskCount(uint32_t globalExpert, uint32_t laneIdx) const
    {
        const uint32_t dstRank = globalExpert / expertPerRank_;
        const uint32_t localExpert = globalExpert - dstRank * expertPerRank_;
        __gm__ uint8_t* remoteMaskBase = reinterpret_cast<__gm__ uint8_t*>(
            remoteWindow_.RemoteBase(peerMemoryLayout_.routeMaskSlots, static_cast<int32_t>(dstRank)));
        const uint64_t slot = static_cast<uint64_t>(localExpert) * rankSize_ + rank_;
        return reinterpret_cast<__gm__ int32_t*>(
            remoteMaskBase + slot * tilingData_->frontReorderTiling.maskSlotBytes +
            tilingData_->frontReorderTiling.maskBytes +
            static_cast<uint64_t>(laneIdx) * kMegaMoeFrontMaskCountRecordBytes);
    }

    AICORE inline __gm__ int32_t* RemotePreSumRow(uint32_t dstRank) const
    {
        __gm__ int32_t* remotePreSum = reinterpret_cast<__gm__ int32_t*>(
            remoteWindow_.RemoteBase(peerMemoryLayout_.preSumBeforeRank, static_cast<int32_t>(dstRank)));
        return remotePreSum + static_cast<uint64_t>(rank_) * expertPerRank_;
    }

    AICORE inline uint32_t ReadRouteCount(uint32_t globalExpert) const
    {
        uint32_t count = 0U;
        const uint32_t laneCount = ActiveMaskLaneCount(globalExpert);
        for (uint32_t laneIdx = 0U; laneIdx < laneCount; ++laneIdx) {
            PtoLoadVector<int32_t, kCountRecordElems>(
                kPreSumCountUb, RemoteMaskCount(globalExpert, laneIdx), kCountRecordElems);
            pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
            count += static_cast<uint32_t>(PtoGetValue<int32_t, kCountRecordElems>(kPreSumCountUb, 0U));
        }
        return count;
    }

    AICORE inline void LoadSingleLaneRouteCountsForRank(uint32_t dstRank) const
    {
        using CountTile = pto::Tile<
            pto::TileType::Vec, int32_t, kMegaMoeFixedMaxExperts, kCountRecordElems, pto::BLayout::RowMajor, -1, -1>;
        using CountShape = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using CountStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using CountGlobal = pto::GlobalTensor<int32_t, CountShape, CountStride, pto::Layout::ND>;

        const int64_t rowStride =
            static_cast<int64_t>(rankSize_) * tilingData_->frontReorderTiling.maskSlotBytes / sizeof(int32_t);
        const int64_t outerStride = static_cast<int64_t>(expertPerRank_) * rowStride;
        CountTile countTile(expertPerRank_, kCountRecordElems);
        pto::TASSIGN(countTile, kPreSumCountUb);
        CountShape countShape(1, 1, 1, expertPerRank_, kCountRecordElems);
        CountStride countStride(outerStride, outerStride, outerStride, rowStride, 1);
        CountGlobal countGlobal(RemoteMaskCount(dstRank * expertPerRank_, 0U), countShape, countStride);
        pto::TLOAD(countTile, countGlobal);
        pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
    }

    AICORE inline void BuildAndPublishPreSum() const
    {
        uint32_t prefix = 0U;
        const bool singleMaskLane = tilingData_->frontReorderTiling.maskLaneCapacity == 1U;
        for (uint32_t dstRank = 0U; dstRank < rankSize_; ++dstRank) {
            if (singleMaskLane) {
                LoadSingleLaneRouteCountsForRank(dstRank);
            }
            for (uint32_t localExpert = 0U; localExpert < expertPerRank_; ++localExpert) {
                const uint32_t globalExpert = dstRank * expertPerRank_ + localExpert;
                PtoSetValue<int32_t>(kPreSumPrefixUb, localExpert, static_cast<int32_t>(prefix));
                if (globalExpert < expertNum_) {
                    prefix +=
                        singleMaskLane ?
                            static_cast<uint32_t>(PtoGetValue<int32_t, kCountRecordElems>(
                                kPreSumCountUb + static_cast<uint64_t>(localExpert) * kMegaMoeFrontMaskCountRecordBytes,
                                0U)) :
                            ReadRouteCount(globalExpert);
                }
            }
            pto::PtoSetWaitFlag<PIPE_S, PIPE_MTE3>();
            PtoStoreVector<int32_t, kMegaMoeFixedMaxExperts>(RemotePreSumRow(dstRank), kPreSumPrefixUb, expertPerRank_);
            pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
        }
        dsb(DSB_DDR);
        const int32_t epoch = remoteWindow_.FrontReadyEpoch();
        for (uint32_t dstRank = 0U; dstRank < rankSize_; ++dstRank) {
            remoteWindow_.PublishPreSumReady(static_cast<int32_t>(dstRank), epoch);
        }
    }

    AICORE inline uint32_t RouteBegin() const
    {
        return static_cast<uint32_t>((static_cast<uint64_t>(routeElems_) * workerIdx_) / workerCount_);
    }

    AICORE inline uint32_t RouteEnd() const
    {
        return static_cast<uint32_t>((static_cast<uint64_t>(routeElems_) * (workerIdx_ + 1U)) / workerCount_);
    }

    AICORE inline void PrepareInverseRowAssist(uint32_t routeBegin) const
    {
        using AssistTile = PtoVecTile<int32_t, kInverseAssistElems>;
        AssistTile assistTile(1, kInverseAssistElems);
        pto::TASSIGN(assistTile, kInverseAssistUb);
        for (uint32_t idx = 0U; idx < kInverseAssistElems; ++idx) {
            const int32_t row =
                idx % kInverseBlockElems == 0U ? static_cast<int32_t>(routeBegin + idx / kInverseBlockElems) : 0;
            assistTile.SetValue(idx, row);
        }
        pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();
    }

    AICORE inline void BuildExpandedRowIdx() const
    {
        const uint32_t routeBegin = RouteBegin();
        const uint32_t routeEnd = RouteEnd();
        if (routeBegin >= routeEnd) {
            return;
        }

        PrepareInverseRowAssist(routeBegin);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        uint32_t loop = 0U;
        for (uint32_t chunkBegin = routeBegin; chunkBegin < routeEnd; chunkBegin += kInverseRowsPerLoop, ++loop) {
            const uint32_t routes =
                routeEnd - chunkBegin > kInverseRowsPerLoop ? kInverseRowsPerLoop : routeEnd - chunkBegin;
            const bool hasNext = chunkBegin + routes < routeEnd;
            PtoLoadVector<int32_t, kInverseRowsPerLoop>(kInverseInputUb, sortedRouteSlotPtr_ + chunkBegin, routes);

            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            const uint32_t assistLoops = (routes + kInverseRowsPerAssist - 1U) / kInverseRowsPerAssist;
            for (uint32_t assist = 0U; assist < assistLoops; ++assist) {
                PtoAddScalarUb<int32_t, kInverseAssistElems>(
                    kInverseOutputUb + static_cast<uint64_t>(assist) * kInverseAssistElems * sizeof(int32_t),
                    kInverseAssistUb, kInverseAssistElems,
                    static_cast<int32_t>(loop * kInverseRowsPerLoop + assist * kInverseRowsPerAssist));
            }
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
            for (uint32_t idx = 0U; idx < routes; ++idx) {
                const uint32_t srcRoute = static_cast<uint32_t>(PtoGetValue<int32_t>(kInverseInputUb, idx));
                if (srcRoute < routeElems_) {
                    PtoStoreVector<int32_t>(
                        expandedRowIdxPtr_ + srcRoute,
                        kInverseOutputUb + static_cast<uint64_t>(idx) * kInverseBlockElems * sizeof(int32_t), 1U);
                }
            }
            if (hasNext) {
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            }
        }
        pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
    }

    __gm__ int32_t* expertIdPtr_ = nullptr;
    __gm__ int32_t* sortedRouteSlotPtr_ = nullptr;
    __gm__ int32_t* expandedRowIdxPtr_ = nullptr;
    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;
    PtoRemoteWindow remoteWindow_;
    MegaMoePeerMemoryLayout peerMemoryLayout_;
    GroupBarrier barrier_;
    uint32_t routeElems_ = 0U;
    uint32_t expertNum_ = 0U;
    uint32_t rank_ = 0U;
    uint32_t rankSize_ = 0U;
    uint32_t expertPerRank_ = 0U;
    uint32_t workerIdx_ = 0U;
    uint32_t workerCount_ = 0U;
};

} // namespace deferred_route_metadata

#endif // DISPATCH_MEGA_COMBINE_DEFERRED_ROUTE_METADATA_H
