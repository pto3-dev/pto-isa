/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_GMM_TASK_PRODUCER_H
#define DISPATCH_MEGA_COMBINE_GMM_TASK_PRODUCER_H

#include "kernel_operator.h"

#include "dispatch_mega_combine_tiling.h"
#include "gmm_common.h"
#include "gmm_task_queue_device.h"
#include "utils/mega_wave_schedule.hpp"
#include "utils/mega_expert_sync.hpp"
#include "utils/pto_vector.hpp"

constexpr uint32_t kGmmProducerMaxAicCount = kMegaMoeFixedPhysicalAicNum;
constexpr uint32_t kGmmProducerProgressSnapshotSlots = 2U;
constexpr uint32_t kGmmProducerProgressWordsPerLane = sizeof(MegaMoeGmmC2pSlot) / sizeof(uint32_t);
constexpr uint32_t kGmmProducerProgressValueWords = kGmmProducerMaxAicCount * kGmmProducerProgressWordsPerLane;
constexpr uint32_t kGmmProducerProgressSnapshotWords = (kGmmProducerProgressValueWords + 7U) / 8U * 8U;
constexpr uint64_t kGmmProducerProgressSnapshotBytes =
    static_cast<uint64_t>(kGmmProducerProgressSnapshotWords) * sizeof(uint32_t);
using GmmProducerProgressSnapshotTile = PtoVecTile<uint32_t, kGmmProducerProgressSnapshotWords>;
constexpr uint32_t kGmmProducerP2cWordsPerLane = sizeof(MegaMoeGmmP2cSlot) / sizeof(uint32_t);
constexpr uint32_t kGmmProducerP2cShadowWords =
    static_cast<uint32_t>(GmmMailboxP2cStorageBytes(kGmmProducerMaxAicCount) / sizeof(uint32_t));
constexpr uint64_t kGmmProducerP2cShadowBytes = static_cast<uint64_t>(kGmmProducerP2cShadowWords) * sizeof(uint32_t);
using GmmProducerP2cShadowTile = PtoVecTile<uint32_t, kGmmProducerP2cShadowWords>;
constexpr uint32_t kGmmProducerP2cTicketWord = offsetof(MegaMoeGmmP2cSlot, ticket) / sizeof(uint32_t);
constexpr uint64_t kGmmProducerReadyProbeIntervalTicks = 1000U;
constexpr uint64_t kGmmProducerMailboxTargetPeriodTicks = 50U;
constexpr uint64_t kGmmProducerProgressSnapshotUbBase = 0U;
constexpr uint64_t kGmmProducerP2cShadowUbBase =
    kGmmProducerProgressSnapshotUbBase + kGmmProducerProgressSnapshotSlots * kGmmProducerProgressSnapshotBytes;
constexpr uint32_t kGmmProducerMailboxStageCount = 2U;
constexpr uint32_t kGmmProducerReadySnapshotMaxBlockM = kGmmTaskBlockMMask + 1U;
constexpr uint64_t kGmmProducerReadySnapshotBytes =
    static_cast<uint64_t>(kGmmProducerReadySnapshotMaxBlockM) * sizeof(uint32_t);
constexpr uint64_t kGmmProducerReadySnapshotUbBase = kGmmProducerP2cShadowUbBase + kGmmProducerP2cShadowBytes;
constexpr uint64_t kGmmProducerStateUbEnd = kGmmProducerReadySnapshotUbBase + kGmmProducerReadySnapshotBytes;

enum class GmmMailboxLanePhase : uint32_t {
    kAwaitGmm1Wave0End = 0U,
    kGmm1Pc = 1U,
    kAwaitGmm2Wave0End = 2U,
    kGmm2Pc = 3U,
    kTerminalReady = 4U,
    kDone = 5U,
};

AICORE inline GmmMailboxLanePhase ObserveGmmMailboxWave0End(
    GmmMailboxLanePhase phase, bool group2Lane, bool gmm1Wave0End, bool gmm2Wave0End)
{
    if (phase == GmmMailboxLanePhase::kAwaitGmm1Wave0End) {
        if (group2Lane && gmm2Wave0End) {
            return GmmMailboxLanePhase::kGmm2Pc;
        }
        if (gmm1Wave0End) {
            return group2Lane ? GmmMailboxLanePhase::kAwaitGmm2Wave0End : GmmMailboxLanePhase::kGmm1Pc;
        }
    } else if (phase == GmmMailboxLanePhase::kAwaitGmm2Wave0End && gmm2Wave0End) {
        return GmmMailboxLanePhase::kGmm2Pc;
    }
    return phase;
}

AICORE inline GmmMailboxLanePhase AdvanceGmmMailboxLanePhase(
    GmmMailboxLanePhase phase, bool gmm1SuffixPublished, bool gmm2GateReady, bool gmm2SuffixPublished)
{
    if (phase == GmmMailboxLanePhase::kGmm1Pc && gmm1SuffixPublished) {
        return gmm2GateReady ? GmmMailboxLanePhase::kGmm2Pc : phase;
    }
    if (phase == GmmMailboxLanePhase::kGmm2Pc && gmm2SuffixPublished) {
        return GmmMailboxLanePhase::kTerminalReady;
    }
    return phase;
}

static_assert(kGmmProducerProgressValueWords * sizeof(uint32_t) == kGmmProducerMaxAicCount * sizeof(MegaMoeGmmC2pSlot));
static_assert(kGmmProducerP2cShadowWords * sizeof(uint32_t) == GmmMailboxP2cStorageBytes(kGmmProducerMaxAicCount));
static_assert(kGmmProducerStateUbEnd <= A5_MAIN_UB_SIZE);

template <typename TileData>
__tf__ AICORE void LoadGmmSnapshot(
    typename TileData::TileDType __out__ dstTile, __gm__ uint8_t __in__* src, uint16_t rowCount, uint32_t rowBytes,
    uint64_t srcStride)
{
    __ubuf__ uint8_t* dst = reinterpret_cast<__ubuf__ uint8_t*>(__cce_get_tile_ptr(dstTile));
    pto_copy_gm_to_ubuf_align_v2(dst, src, 0, rowCount, rowBytes, 0, 0, false, 0, srcStride, rowBytes);
}

class MegaMoeGmmTaskProducer {
public:
    AICORE inline void Init(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData)
    {
        workspaceGM_ = workspaceGM;
        tilingData_ = tilingData;
        cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t*>(workspaceGM + tilingData->frontReorderTiling.cumsumMMOffset);
        rankSize_ = tilingData->runtimeInfo.rankSize;
        expertPerRank_ = tilingData->megaMoeInfo.expertPerRank;
    }

    AICORE inline void Process()
    {
        WaitEpochAcquire(
            FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncFrontMetadataReadySlot),
            kMegaMoeFixedFrontMetadataReadyMarker);
        ProcessMailbox();
    }

private:
    using MailboxStage = GmmTaskStage;

    struct MailboxLaneState {
        uint32_t publishedTicket = kGmmMailboxEmptyTicket;
        GmmMailboxLanePhase phase = GmmMailboxLanePhase::kAwaitGmm1Wave0End;
    };

    struct ExpertTaskLayout {
        uint32_t currentM[kMegaMoeFixedMaxExperts] = {};
        uint32_t gmm1TileM[kMegaMoeFixedMaxExperts] = {};
        uint32_t taskBase[kGmmProducerMailboxStageCount][kMegaMoeFixedMaxExperts + 1U] = {};
        uint32_t tileN[kGmmProducerMailboxStageCount] = {};
    };

    struct ReadyTracker {
        uint32_t scanExpert = 0U;
        uint32_t publishExpert = 0U;
        uint32_t readyTail = 0U;
        uint32_t publishTail = 0U;
    };

    AICORE inline volatile __gm__ int32_t* DispatchReadySlot(uint32_t expert, uint32_t blockM) const
    {
        const __gm__ MegaMoeDispatchTiling& dispatch = tilingData_->dispatchTiling;
        const uint64_t offset =
            dispatch.readyCountOffset +
            static_cast<uint64_t>(expert) * dispatch.readyCountMaxTilesPerExpert * kMegaMoeReadyCountSlotBytes +
            static_cast<uint64_t>(blockM) * kMegaMoeReadyCountSlotBytes;
        return reinterpret_cast<volatile __gm__ int32_t*>(workspaceGM_ + offset);
    }

    AICORE inline bool Gmm2EntryGateReady() const { return Gmm2EntryReady(workspaceGM_, tilingData_); }

    AICORE inline event_t MailboxP2cEvent() const { return static_cast<event_t>(0U); }

    AICORE inline void AdvanceMailboxLane(uint32_t& physicalBlockId, uint32_t aicCount) const
    {
        ++physicalBlockId;
        if (physicalBlockId == aicCount) {
            physicalBlockId = 0U;
        }
    }

    AICORE inline void AcquireMailboxP2cWrite()
    {
        if (p2cWriteBusy_) {
            wait_flag(PIPE_MTE3, PIPE_S, MailboxP2cEvent());
            p2cWriteBusy_ = false;
        }
    }

    AICORE inline void StageMailboxTicket(
        uint32_t physicalBlockId, uint32_t ticket, MailboxLaneState& lane, GmmProducerP2cShadowTile& p2cShadow,
        bool& p2cDirty)
    {
        AcquireMailboxP2cWrite();
        const uint32_t ticketWord = physicalBlockId * kGmmProducerP2cWordsPerLane + kGmmProducerP2cTicketWord;
        p2cShadow.SetValue(ticketWord, ticket);
        p2cDirty = true;
        lane.publishedTicket = ticket;
    }

    AICORE inline void StageMailboxWork(
        uint32_t physicalBlockId, MailboxStage stage, uint32_t localTicket, MailboxLaneState& lane,
        GmmProducerP2cShadowTile& p2cShadow, bool& p2cDirty)
    {
        const __gm__ MegaMoeGmmMailboxTiling& mailbox = tilingData_->gmmSchedulerTiling.mailbox;
        const uint32_t globalTicket = stage == MailboxStage::kGmm1 ? kGmmMailboxFirstTaskTicket + localTicket :
                                                                     mailbox.gmm2TicketBase + localTicket;
        StageMailboxTicket(physicalBlockId, globalTicket, lane, p2cShadow, p2cDirty);
    }

    AICORE inline void InitExpertTaskLayout(ExpertTaskLayout& layout) const
    {
        constexpr uint32_t gmm1Stage = static_cast<uint32_t>(MailboxStage::kGmm1);
        constexpr uint32_t gmm2Stage = static_cast<uint32_t>(MailboxStage::kGmm2);
        layout.tileN[gmm1Stage] = GmmCommonTileN(tilingData_->megaMoeInfo.N / 2U);
        layout.tileN[gmm2Stage] = GmmCommonTileN(tilingData_->megaMoeInfo.K);
        layout.taskBase[gmm1Stage][0U] = 0U;
        layout.taskBase[gmm2Stage][0U] = 0U;
    }

    AICORE inline void AppendExpertTaskLayout(ExpertTaskLayout& layout, uint32_t expert) const
    {
        constexpr uint32_t gmm1Stage = static_cast<uint32_t>(MailboxStage::kGmm1);
        constexpr uint32_t gmm2Stage = static_cast<uint32_t>(MailboxStage::kGmm2);
        const uint32_t currentM = CurrentM(expert);
        const GmmCommonTaskShape gmm1Shape = GmmCommonBuildTaskShape(currentM, tilingData_->megaMoeInfo.N / 2U);
        const GmmCommonTaskShape gmm2Shape = GmmCommonBuildTaskShape(currentM, tilingData_->megaMoeInfo.K);
        layout.currentM[expert] = currentM;
        layout.gmm1TileM[expert] = gmm1Shape.tileM;
        layout.taskBase[gmm1Stage][expert + 1U] = layout.taskBase[gmm1Stage][expert] + gmm1Shape.taskCount;
        layout.taskBase[gmm2Stage][expert + 1U] = layout.taskBase[gmm2Stage][expert] + gmm2Shape.taskCount;
    }

    AICORE inline event_t ReadySnapshotEvent() const { return static_cast<event_t>(2U); }

    AICORE inline void AdvanceExpert(
        uint32_t& expert, uint32_t tail, uint32_t stage, const ExpertTaskLayout& layout) const
    {
        while (expert < expertPerRank_ && tail >= layout.taskBase[stage][expert + 1U]) {
            ++expert;
        }
    }

    AICORE inline __gm__ int32_t* ReadyCounterSlot(uint32_t stage, uint32_t expert, uint32_t blockM) const
    {
        if (stage == static_cast<uint32_t>(MailboxStage::kGmm1)) {
            return const_cast<__gm__ int32_t*>(DispatchReadySlot(expert, blockM));
        }
        return GmmTaskDependencySlot(
            workspaceGM_, tilingData_->gmmSchedulerTiling.gmm2, tilingData_->dispatchTiling.readyCountMaxTilesPerExpert,
            expert, blockM);
    }

    AICORE inline uint32_t ReadyCounterExpected(
        uint32_t stage, const ExpertTaskLayout& layout, uint32_t expert, uint32_t blockM) const
    {
        if (stage == static_cast<uint32_t>(MailboxStage::kGmm2)) {
            constexpr uint32_t gmm1Stage = static_cast<uint32_t>(MailboxStage::kGmm1);
            return layout.taskBase[gmm1Stage][expert + 1U] - layout.taskBase[gmm1Stage][expert];
        }
        const uint32_t rowBegin = blockM * kMegaMoeGmmTileM;
        const uint32_t remaining = layout.currentM[expert] - rowBegin;
        return remaining < kMegaMoeGmmTileM ? remaining : kMegaMoeGmmTileM;
    }

    AICORE inline void LoadReadyExpertSnapshot(uint32_t stage, uint32_t expert, uint32_t blockMCount) const
    {
        using SnapshotTile = PtoVecTile<uint32_t, kGmmProducerReadySnapshotMaxBlockM>;
        SnapshotTile snapshot(1, blockMCount);
        pto::TASSIGN(snapshot, kGmmProducerReadySnapshotUbBase);
        __gm__ uint8_t* src = reinterpret_cast<__gm__ uint8_t*>(ReadyCounterSlot(stage, expert, 0U));
        const uint64_t srcStride = stage == static_cast<uint32_t>(MailboxStage::kGmm1) ? kMegaMoeReadyCountSlotBytes :
                                                                                         kMegaMoeFixedSyncSlotBytes;
        LoadGmmSnapshot<SnapshotTile>(
            snapshot.data(), src, static_cast<uint16_t>(blockMCount), sizeof(uint32_t), srcStride);
        set_flag(PIPE_MTE2, PIPE_S, ReadySnapshotEvent());
        wait_flag(PIPE_MTE2, PIPE_S, ReadySnapshotEvent());
    }

    AICORE inline bool DiscoverReadyTickets(
        uint32_t stage, ReadyTracker& tracker, const ExpertTaskLayout& layout, uint32_t totalTasks) const
    {
        if (tracker.readyTail >= totalTasks) {
            return false;
        }
        AdvanceExpert(tracker.scanExpert, tracker.readyTail, stage, layout);
        const uint32_t expert = tracker.scanExpert;
        const uint32_t blockMCount =
            stage == static_cast<uint32_t>(MailboxStage::kGmm2) ? 1U : layout.gmm1TileM[expert];
        LoadReadyExpertSnapshot(stage, expert, blockMCount);

        using SnapshotTile = PtoVecTile<uint32_t, kGmmProducerReadySnapshotMaxBlockM>;
        SnapshotTile snapshot(1, blockMCount);
        pto::TASSIGN(snapshot, kGmmProducerReadySnapshotUbBase);
        const uint32_t previousReadyTail = tracker.readyTail;
        const uint32_t expertTaskBase = layout.taskBase[stage][expert];
        const uint32_t expertTaskEnd = layout.taskBase[stage][expert + 1U];
        if (blockMCount == 1U) {
            if (snapshot.GetValue(0U) >= ReadyCounterExpected(stage, layout, expert, 0U)) {
                tracker.readyTail = expertTaskEnd;
            }
        } else {
            while (tracker.readyTail < expertTaskEnd) {
                const uint32_t expertLoop = tracker.readyTail - expertTaskBase;
                uint32_t blockM = 0U;
                uint32_t blockN = 0U;
                GmmCommonGetBlockCoordMN(expertLoop, blockMCount, layout.tileN[stage], blockM, blockN);
                (void)blockN;
                const uint32_t observed = snapshot.GetValue(blockM);
                if (observed < ReadyCounterExpected(stage, layout, expert, blockM)) {
                    break;
                }
                ++tracker.readyTail;
            }
        }
        if (tracker.readyTail >= expertTaskEnd) {
            ++tracker.scanExpert;
        }
        return tracker.readyTail != previousReadyTail;
    }

    AICORE inline bool TakeReadyTicket(
        uint32_t stage, ReadyTracker& tracker, const ExpertTaskLayout& layout, uint32_t& localTicket) const
    {
        if (tracker.publishTail >= tracker.readyTail) {
            return false;
        }
        AdvanceExpert(tracker.publishExpert, tracker.publishTail, stage, layout);
        localTicket = tracker.publishTail++;
        AdvanceExpert(tracker.publishExpert, tracker.publishTail, stage, layout);
        return true;
    }

    AICORE inline uint64_t ProgressSnapshotUbOffset(uint32_t slot) const
    {
        return kGmmProducerProgressSnapshotUbBase + static_cast<uint64_t>(slot) * kGmmProducerProgressSnapshotBytes;
    }

    AICORE inline event_t ProgressSnapshotEvent(uint32_t slot) const { return static_cast<event_t>(slot); }

    AICORE inline void FlushMailboxP2c(bool& p2cDirty)
    {
        if (!p2cDirty) {
            return;
        }

        __gm__ uint32_t* p2cWords =
            reinterpret_cast<__gm__ uint32_t*>(workspaceGM_ + tilingData_->gmmSchedulerTiling.mailbox.p2cOffset);
        const event_t event = MailboxP2cEvent();
        pto::PtoSetWaitFlag<PIPE_S, PIPE_MTE3>(event, event);
        PtoStoreVector<uint32_t, kGmmProducerP2cShadowWords>(
            p2cWords, kGmmProducerP2cShadowUbBase, kGmmProducerP2cShadowWords);
        set_flag(PIPE_MTE3, PIPE_S, event);
        p2cWriteBusy_ = true;
        p2cDirty = false;
    }

    AICORE inline void IssueProgressSnapshot(uint32_t slot, uint32_t aicCount) const
    {
        GmmProducerProgressSnapshotTile tile(1, aicCount * kGmmProducerProgressWordsPerLane);
        pto::TASSIGN(tile, ProgressSnapshotUbOffset(slot));
        __gm__ uint8_t* src =
            reinterpret_cast<__gm__ uint8_t*>(workspaceGM_ + tilingData_->gmmSchedulerTiling.mailbox.c2pOffset);
        const uint32_t bytes = aicCount * kGmmProducerProgressWordsPerLane * sizeof(uint32_t);
        LoadGmmSnapshot<GmmProducerProgressSnapshotTile>(tile.data(), src, 1U, bytes, bytes);
        set_flag(PIPE_MTE2, PIPE_S, ProgressSnapshotEvent(slot));
    }

    AICORE inline void ProcessMailbox()
    {
        PtoFillUb<uint32_t, kGmmProducerP2cShadowWords>(kGmmProducerP2cShadowUbBase, 0U, kGmmProducerP2cShadowWords);
        GmmProducerP2cShadowTile p2cShadow(1, kGmmProducerP2cShadowWords);
        pto::TASSIGN(p2cShadow, kGmmProducerP2cShadowUbBase);
        pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();

        const uint32_t aicCount = tilingData_->fixedGroupTiling.physicalAicNum;
        const bool gmm1Mailbox =
            tilingData_->gmmSchedulerTiling.gmm1ScheduleMode == kMegaMoeGmm1ScheduleWave0MailboxSuffix;

        MailboxLaneState lanes[kGmmProducerMaxAicCount] = {};
        ExpertTaskLayout taskLayout;
        InitExpertTaskLayout(taskLayout);
        constexpr uint32_t gmm1Stage = static_cast<uint32_t>(MailboxStage::kGmm1);
        constexpr uint32_t gmm2Stage = static_cast<uint32_t>(MailboxStage::kGmm2);
        const __gm__ MegaMoeFixedGroupTiling& fixed = tilingData_->fixedGroupTiling;
        const MegaMoeExpertWaveRange wave0 = GetExpertWaveRange(
            0U, expertPerRank_, fixed.fullAicExpertsPerWave, fixed.expertsPerWave, fixed.fullAicGmm1WaveCount);
        const uint32_t wave0ExpertEnd = wave0.end;
        uint32_t layoutExpert = 0U;
        while (layoutExpert < wave0ExpertEnd) {
            AppendExpertTaskLayout(taskLayout, layoutExpert);
            ++layoutExpert;
        }
        const uint32_t gmm1Wave0TaskCount = taskLayout.taskBase[gmm1Stage][wave0ExpertEnd];
        const uint32_t gmm2Wave0TaskCount = taskLayout.taskBase[gmm2Stage][wave0ExpertEnd];
        while (layoutExpert < expertPerRank_) {
            AppendExpertTaskLayout(taskLayout, layoutExpert);
            ++layoutExpert;
        }
        const uint32_t initialGmm1TaskCount = gmm1Mailbox ? gmm1Wave0TaskCount : 0U;
        const uint32_t initialGmm2TaskCount = gmm2Wave0TaskCount;
        ReadyTracker readyTrackers[kGmmProducerMailboxStageCount];
        // Ready discovery is independent of lane phases. Both shared suffix
        // pools start immediately after their stage-local direct wave0 prefix.
        readyTrackers[gmm1Stage].scanExpert = gmm1Mailbox ? wave0ExpertEnd : expertPerRank_;
        readyTrackers[gmm1Stage].publishExpert = gmm1Mailbox ? wave0ExpertEnd : expertPerRank_;
        readyTrackers[gmm1Stage].readyTail = initialGmm1TaskCount;
        readyTrackers[gmm1Stage].publishTail = initialGmm1TaskCount;
        readyTrackers[gmm2Stage].scanExpert = wave0ExpertEnd;
        readyTrackers[gmm2Stage].publishExpert = wave0ExpertEnd;
        readyTrackers[gmm2Stage].readyTail = initialGmm2TaskCount;
        readyTrackers[gmm2Stage].publishTail = initialGmm2TaskCount;
        uint64_t nextReadyTaskTick = 0U;
        uint32_t progressSnapshotSlot = 0U;
        IssueProgressSnapshot(progressSnapshotSlot, aicCount);

        const uint32_t gmm1TaskCount = gmm1Mailbox ? taskLayout.taskBase[gmm1Stage][expertPerRank_] : 0U;
        const uint32_t gmm2TaskCount = taskLayout.taskBase[gmm2Stage][expertPerRank_];
        const uint32_t gmm1GroupSize = tilingData_->fixedGroupTiling.gmm1GroupSize;
        uint32_t doneLanes = 0U;
        uint32_t laneCursor = 0U;
        uint32_t readyStageCursor = gmm1Stage;
        bool gmm2Gate = false;
        uint64_t nextGateCheckTick = 0U;
        uint64_t nextMailboxTaskTick = 0U;

        while (doneLanes < aicCount) {
            bool progressed = false;

            // Task A: snapshot exactly one current expert every 1 us and
            // advance only the contiguous swizzled ready-ticket prefix.
            const uint64_t readyNow = get_sys_cnt();
            const bool gmm1ReadyIncomplete = gmm1Mailbox && readyTrackers[0].readyTail < gmm1TaskCount;
            const bool gmm2ReadyIncomplete = readyTrackers[1].readyTail < gmm2TaskCount;
            if ((gmm1ReadyIncomplete || gmm2ReadyIncomplete) && readyNow >= nextReadyTaskTick) {
                nextReadyTaskTick = readyNow + kGmmProducerReadyProbeIntervalTicks;
                uint32_t readyStage = readyStageCursor;
                if ((readyStage == static_cast<uint32_t>(MailboxStage::kGmm1) && !gmm1ReadyIncomplete) ||
                    (readyStage == static_cast<uint32_t>(MailboxStage::kGmm2) && !gmm2ReadyIncomplete)) {
                    readyStage ^= 1U;
                }
                const uint32_t readyTaskCount =
                    readyStage == static_cast<uint32_t>(MailboxStage::kGmm1) ? gmm1TaskCount : gmm2TaskCount;
                progressed = DiscoverReadyTickets(readyStage, readyTrackers[readyStage], taskLayout, readyTaskCount) ||
                             progressed;
                readyStageCursor = readyStage ^ 1U;
            }

            // Task B: observe all 36 C2P progress words with one MTE load,
            // then refill empty single-ticket P2C slots. It never scans ready.
            const uint64_t mailboxNow = get_sys_cnt();
            if (mailboxNow >= nextMailboxTaskTick) {
                const uint32_t completedProgressSlot = progressSnapshotSlot;
                wait_flag(PIPE_MTE2, PIPE_S, ProgressSnapshotEvent(completedProgressSlot));
                const uint32_t nextProgressSlot = completedProgressSlot ^ 1U;
                IssueProgressSnapshot(nextProgressSlot, aicCount);
                progressSnapshotSlot = nextProgressSlot;

                const uint64_t gateNow = get_sys_cnt();
                if (!gmm2Gate && gateNow >= nextGateCheckTick) {
                    if (Gmm2EntryGateReady()) {
                        gmm2Gate = true;
                    }
                    nextGateCheckTick = gateNow + kGmmProducerReadyProbeIntervalTicks;
                }

                bool p2cDirty = false;
                GmmProducerProgressSnapshotTile completedProgress(1, aicCount * kGmmProducerProgressWordsPerLane);
                pto::TASSIGN(completedProgress, ProgressSnapshotUbOffset(completedProgressSlot));
                uint32_t physicalBlockId = laneCursor;
                for (uint32_t visit = 0U; visit < aicCount; ++visit) {
                    MailboxLaneState& lane = lanes[physicalBlockId];
                    const uint32_t progress =
                        completedProgress.GetValue(physicalBlockId * kGmmProducerProgressWordsPerLane);
                    const bool group2Lane = physicalBlockId >= gmm1GroupSize;
                    const GmmMailboxLanePhase observedPhase = ObserveGmmMailboxWave0End(
                        lane.phase, group2Lane, progress == kGmmMailboxGmm1Wave0EndTicket,
                        progress == kGmmMailboxGmm2Wave0EndTicket);
                    if (observedPhase != lane.phase) {
                        lane.phase = observedPhase;
                        progressed = true;
                    }
                    if (lane.phase != GmmMailboxLanePhase::kDone && lane.publishedTicket != kGmmMailboxEmptyTicket) {
                        if (progress == lane.publishedTicket) {
                            const uint32_t ticket = lane.publishedTicket;
                            lane.publishedTicket = kGmmMailboxEmptyTicket;
                            if (ticket == kGmmMailboxTerminalTicket) {
                                lane.phase = GmmMailboxLanePhase::kDone;
                                ++doneLanes;
                            }
                            progressed = true;
                        }
                    }
                    if (lane.phase != GmmMailboxLanePhase::kDone && lane.publishedTicket == kGmmMailboxEmptyTicket) {
                        for (uint32_t phaseAdvance = 0U; phaseAdvance < 2U; ++phaseAdvance) {
                            const GmmMailboxLanePhase advancedPhase = AdvanceGmmMailboxLanePhase(
                                lane.phase, readyTrackers[gmm1Stage].publishTail >= gmm1TaskCount, gmm2Gate,
                                readyTrackers[gmm2Stage].publishTail >= gmm2TaskCount);
                            if (advancedPhase == lane.phase) {
                                break;
                            }
                            lane.phase = advancedPhase;
                            progressed = true;
                        }
                        uint32_t localTicket = 0U;
                        bool published = false;
                        if (lane.phase == GmmMailboxLanePhase::kGmm1Pc) {
                            if (TakeReadyTicket(gmm1Stage, readyTrackers[gmm1Stage], taskLayout, localTicket)) {
                                StageMailboxWork(
                                    physicalBlockId, MailboxStage::kGmm1, localTicket, lane, p2cShadow, p2cDirty);
                                published = true;
                            }
                        } else if (lane.phase == GmmMailboxLanePhase::kGmm2Pc && gmm2Gate) {
                            if (TakeReadyTicket(gmm2Stage, readyTrackers[gmm2Stage], taskLayout, localTicket)) {
                                StageMailboxWork(
                                    physicalBlockId, MailboxStage::kGmm2, localTicket, lane, p2cShadow, p2cDirty);
                                published = true;
                            }
                        } else if (lane.phase == GmmMailboxLanePhase::kTerminalReady) {
                            StageMailboxTicket(physicalBlockId, kGmmMailboxTerminalTicket, lane, p2cShadow, p2cDirty);
                            published = true;
                        }
                        if (published) {
                            progressed = true;
                        }
                    }
                    AdvanceMailboxLane(physicalBlockId, aicCount);
                }
                if (doneLanes >= aicCount) {
                    wait_flag(PIPE_MTE2, PIPE_S, ProgressSnapshotEvent(progressSnapshotSlot));
                    break;
                }
                FlushMailboxP2c(p2cDirty);
                AdvanceMailboxLane(laneCursor, aicCount);
                nextMailboxTaskTick = mailboxNow + kGmmProducerMailboxTargetPeriodTicks;
            }

            if (!progressed) {
                GmmPollBackoff();
            }
        }
        AcquireMailboxP2cWrite();
    }

    AICORE inline uint32_t CurrentM(uint32_t expert) const
    {
        return MoeCurrentMRaw(cumsumMMPtr_, rankSize_, expertPerRank_, expert);
    }

    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;
    __gm__ int32_t* cumsumMMPtr_ = nullptr;
    uint32_t rankSize_ = 0U;
    uint32_t expertPerRank_ = 0U;
    bool p2cWriteBusy_ = false;
};

#endif // DISPATCH_MEGA_COMBINE_GMM_TASK_PRODUCER_H
