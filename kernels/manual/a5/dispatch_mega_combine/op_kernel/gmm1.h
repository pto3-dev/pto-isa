/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_GMM1_H
#define DISPATCH_MEGA_COMBINE_GMM1_H

#include "kernel_operator.h"

#include "dispatch_mega_combine_tiling.h"
#include "gmm_common.h"
#include "gmm1_swiglu_cv_pipe.h"
#include "gmm_task_queue_device.h"
#include "utils/const_args.hpp"
#include "utils/mega_expert_sync.hpp"
#include "utils/mega_wave_schedule.hpp"

using Gmm1Pipeline = GmmCommonPipeline;

AICORE inline void PublishGmm1SwigluControl(Gmm1SwigluCvPipe& pipe, const MegaMoeGmmTask& task)
{
    Gmm1SwigluControlProducerAllocate(pipe);
    WriteGmmCvTaskControl(pipe.prod.controlIndex, kGmm1SwigluControlFifoDepth, task);
    Gmm1SwigluControlProducerRecord(pipe);
}

AICORE inline MegaMoeGmmTask Gmm1SwigluStageEndControl()
{
    MegaMoeGmmTask task;
    task.flags = kGmmTaskFlagStageEnd;
    return task;
}

class Gmm1 {
public:
    AICORE inline void Init(
        GM_ADDR weight1GM, GM_ADDR weightScale1GM, GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData);
    AICORE inline GmmClaimedTask ProcessFixed(uint32_t groupLocalId, uint32_t groupSize, Gmm1Pipeline* gmmPipeline);

private:
    AICORE inline uint32_t CoreLoops(uint32_t currentM) const { return GmmCommonCoreLoops(currentM, outputN_); }
    AICORE inline uint32_t StartLoopIdx(uint32_t startCoreIdx) const
    {
        return GmmCommonStartLoopIdx(coreIdx_, coreNum_, startCoreIdx);
    }
    struct DirectWaveAssignment {
        MegaMoeGmmTask task;
        GmmCommonTileInfo tileInfo;
    };

    class DirectWaveCursor {
    public:
        AICORE inline void Init(Gmm1* owner, uint32_t waveEnd)
        {
            owner_ = owner;
            waveEnd_ = waveEnd;
            SetCoreTileBalancerRange(tileBalancer_, 0U, owner_->coreNum_);
        }

        AICORE inline bool Advance()
        {
            while (expertLoaded_ || expert_ < waveEnd_) {
                if (!expertLoaded_) {
                    currentM_ =
                        MoeCurrentMRaw(owner_->cumsumMMPtr_, owner_->rankSize_, owner_->expertPerRank_, expert_);
                    coreLoops_ = owner_->CoreLoops(currentM_);
                    startCoreIdx_ = SelectCoreTileStart(tileBalancer_, coreLoops_);
                    nextLoopIdx_ = owner_->StartLoopIdx(startCoreIdx_);
                    expertLoaded_ = true;
                }
                if (nextLoopIdx_ < coreLoops_) {
                    const uint32_t loopIdx = nextLoopIdx_;
                    nextLoopIdx_ += owner_->coreNum_;
                    current_.tileInfo = owner_->BuildTileInfo(currentM_, loopIdx);
                    current_.task.flags = kGmmTaskFlagNormal;
                    current_.task.expert = expert_;
                    current_.task.expertBase = expertBase_;
                    current_.task.currentM = currentM_;
                    current_.task.blockM = current_.tileInfo.blockM;
                    current_.task.blockN = current_.tileInfo.blockN;
                    return true;
                }
                CommitCoreTileAssignment(tileBalancer_, startCoreIdx_, coreLoops_);
                expertBase_ += currentM_;
                ++expert_;
                expertLoaded_ = false;
            }
            return false;
        }

        AICORE inline const DirectWaveAssignment& Current() const { return current_; }

    private:
        Gmm1* owner_ = nullptr;
        MegaMoeCoreTileBalancer tileBalancer_;
        DirectWaveAssignment current_;
        uint32_t waveEnd_ = 0U;
        uint32_t expert_ = 0U;
        uint32_t expertBase_ = 0U;
        uint32_t currentM_ = 0U;
        uint32_t coreLoops_ = 0U;
        uint32_t startCoreIdx_ = 0U;
        uint32_t nextLoopIdx_ = 0U;
        bool expertLoaded_ = false;
    };

    AICORE inline volatile __gm__ int32_t* DispatchReadyCountSlot(uint32_t groupIdx, uint32_t blockM) const
    {
        const __gm__ MegaMoeDispatchTiling& dispatch = tilingData_->dispatchTiling;
        const uint64_t byteOffset =
            dispatch.readyCountOffset +
            static_cast<uint64_t>(groupIdx) * dispatch.readyCountMaxTilesPerExpert * kMegaMoeReadyCountSlotBytes +
            static_cast<uint64_t>(blockM) * kMegaMoeReadyCountSlotBytes;
        return reinterpret_cast<volatile __gm__ int32_t*>(workspaceGM_ + byteOffset);
    }
    AICORE inline void WaitDispatchTileReady(uint32_t groupIdx, const GmmCommonTileInfo& tileInfo) const
    {
        volatile __gm__ int32_t* slot = DispatchReadyCountSlot(groupIdx, tileInfo.blockM);
        const int32_t targetRows = static_cast<int32_t>(tileInfo.actualM);
        int32_t observedRows = 0;
        do {
            dcci((__gm__ void*)slot, SINGLE_CACHE_LINE);
            dsb(DSB_DDR);
            observedRows = *slot;
            if (observedRows != targetRows) {
                EpochPollBackoff();
            }
        } while (observedRows != targetRows);
        dsb(DSB_DDR);
    }
    AICORE inline void BuildGmmPairRuns(
        const MegaMoeGmmTask& task, const GmmCommonTileInfo& tileInfo, uint32_t dataSlotBase, uint32_t scaleSlotBase,
        Gmm1Pipeline::TileRun& xRun, Gmm1Pipeline::TileRun& gateRun) const
    {
        const uint32_t scaleLeadingDim = problemK_ / kMegaMoeMxGroupSize;
        const uint64_t gmOffsetA = static_cast<uint64_t>(task.expertBase + tileInfo.blockRowStart) * problemK_;
        const uint64_t gmOffsetAScale =
            static_cast<uint64_t>(task.expertBase + tileInfo.blockRowStart) * scaleLeadingDim;
        const uint64_t weightExpertStride = static_cast<uint64_t>(problemN_) * problemK_;
        const uint64_t weightScaleExpertStride = static_cast<uint64_t>(problemN_) * scaleLeadingDim;
        const uint64_t weightExpertBase = static_cast<uint64_t>(task.expert) * weightExpertStride;
        const uint64_t weightScaleExpertBase = static_cast<uint64_t>(task.expert) * weightScaleExpertStride;
        const uint64_t xOffset = weightExpertBase + static_cast<uint64_t>(tileInfo.blockColStart) * problemK_;
        const uint64_t gateOffset =
            weightExpertBase + static_cast<uint64_t>(outputN_ + tileInfo.blockColStart) * problemK_;
        const uint64_t xScaleOffset =
            weightScaleExpertBase + static_cast<uint64_t>(tileInfo.blockColStart) * scaleLeadingDim;
        const uint64_t gateScaleOffset =
            weightScaleExpertBase + static_cast<uint64_t>(outputN_ + tileInfo.blockColStart) * scaleLeadingDim;
        const uint32_t gateDataSlotBase = Gmm1Pipeline::AdvanceDataSlotBase(dataSlotBase, problemK_);
        const uint32_t gateScaleSlotBase = Gmm1Pipeline::AdvanceScaleSlotBase(scaleSlotBase, problemK_);
        xRun = Gmm1Pipeline::TileRun{
            gmAPtr_ + gmOffsetA,
            gmAScalePtr_ + gmOffsetAScale,
            weight1Ptr_ + xOffset,
            weightScale1Ptr_ + xScaleOffset,
            tileInfo.actualM,
            tileInfo.actualN,
            problemK_,
            problemK_,
            scaleLeadingDim,
            problemK_,
            scaleLeadingDim,
            dataSlotBase,
            scaleSlotBase};
        gateRun = Gmm1Pipeline::TileRun{
            gmAPtr_ + gmOffsetA,
            gmAScalePtr_ + gmOffsetAScale,
            weight1Ptr_ + gateOffset,
            weightScale1Ptr_ + gateScaleOffset,
            tileInfo.actualM,
            tileInfo.actualN,
            problemK_,
            problemK_,
            scaleLeadingDim,
            problemK_,
            scaleLeadingDim,
            gateDataSlotBase,
            gateScaleSlotBase};
    }
    AICORE inline GmmCommonTileInfo BuildTileInfo(uint32_t currentM, uint32_t loopIdx) const
    {
        return GmmCommonBuildTileInfo(currentM, outputN_, loopIdx);
    }
    AICORE inline void RunGmmPair(
        Gmm1Pipeline& gmmPipeline, Gmm1SwigluCvPipe& cvPipe, uint32_t groupIdx, uint32_t groupBase,
        const GmmCommonTileInfo& tileInfo) const
    {
        const uint32_t scaleLeadingDim = problemK_ / kMegaMoeMxGroupSize;
        const uint64_t gmOffsetA = static_cast<uint64_t>(groupBase + tileInfo.blockRowStart) * problemK_;
        const uint64_t gmOffsetAScale = static_cast<uint64_t>(groupBase + tileInfo.blockRowStart) * scaleLeadingDim;
        const uint64_t weightExpertStride = static_cast<uint64_t>(problemN_) * problemK_;
        const uint64_t weightScaleExpertStride = static_cast<uint64_t>(problemN_) * scaleLeadingDim;
        const uint64_t weightExpertBase = static_cast<uint64_t>(groupIdx) * weightExpertStride;
        const uint64_t weightScaleExpertBase = static_cast<uint64_t>(groupIdx) * weightScaleExpertStride;
        const uint64_t xOffset = weightExpertBase + static_cast<uint64_t>(tileInfo.blockColStart) * problemK_;
        const uint64_t gateOffset =
            weightExpertBase + static_cast<uint64_t>(outputN_ + tileInfo.blockColStart) * problemK_;
        const uint64_t xScaleOffset =
            weightScaleExpertBase + static_cast<uint64_t>(tileInfo.blockColStart) * scaleLeadingDim;
        const uint64_t gateScaleOffset =
            weightScaleExpertBase + static_cast<uint64_t>(outputN_ + tileInfo.blockColStart) * scaleLeadingDim;

        gmmPipeline.RunPairDirect(
            cvPipe, gmAPtr_ + gmOffsetA, gmAScalePtr_ + gmOffsetAScale, weight1Ptr_ + xOffset,
            weightScale1Ptr_ + xScaleOffset, weight1Ptr_ + gateOffset, weightScale1Ptr_ + gateScaleOffset,
            tileInfo.actualM, tileInfo.actualN, problemK_, problemK_, scaleLeadingDim, problemK_, scaleLeadingDim);
    }
    AICORE inline void ProcessFixedWave();
    AICORE inline GmmClaimedTask ProcessDirectWave0(Gmm1SwigluCvPipe& cvPipe, Gmm1Pipeline& gmmPipeline);
    AICORE inline GmmClaimedTask WaitGmm1Successor(
        GmmMailboxConsumerCursor& mailboxCursor, GmmMailboxTicketProbe* successorProbe, uint32_t dataSlotBase,
        uint32_t scaleSlotBase);
    AICORE inline GmmClaimedTask ProcessMailbox(
        Gmm1Pipeline& gmmPipeline, Gmm1SwigluCvPipe& cvPipe, const GmmClaimedTask& initialTask);

    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;

    __gm__ float8_e4m3_t* gmAPtr_ = nullptr;
    __gm__ float8_e8m0_t* gmAScalePtr_ = nullptr;
    __gm__ float8_e4m3_t* weight1Ptr_ = nullptr;
    __gm__ float8_e8m0_t* weightScale1Ptr_ = nullptr;
    __gm__ int32_t* cumsumMMPtr_ = nullptr;

    uint32_t problemK_ = 0;
    uint32_t problemN_ = 0;
    uint32_t outputN_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t rankSize_ = 0;
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
};

AICORE inline void Gmm1::Init(
    GM_ADDR weight1GM, GM_ADDR weightScale1GM, GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData)
{
    workspaceGM_ = workspaceGM;
    tilingData_ = tilingData;

    problemK_ = tilingData_->megaMoeInfo.K;
    problemN_ = tilingData_->megaMoeInfo.N;
    outputN_ = problemN_ / 2U;
    expertPerRank_ = tilingData_->megaMoeInfo.expertPerRank;
    rankSize_ = tilingData_->runtimeInfo.rankSize;

    gmAPtr_ = reinterpret_cast<__gm__ float8_e4m3_t*>(workspaceGM + tilingData_->dispatchTiling.gmAOffset);
    gmAScalePtr_ = reinterpret_cast<__gm__ float8_e8m0_t*>(workspaceGM + tilingData_->dispatchTiling.gmAScaleOffset);
    weight1Ptr_ = reinterpret_cast<__gm__ float8_e4m3_t*>(weight1GM);
    weightScale1Ptr_ = reinterpret_cast<__gm__ float8_e8m0_t*>(weightScale1GM);
    cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t*>(workspaceGM + tilingData_->frontReorderTiling.cumsumMMOffset);
}
AICORE inline GmmClaimedTask Gmm1::ProcessFixed(uint32_t groupLocalId, uint32_t groupSize, Gmm1Pipeline* gmmPipeline)
{
    coreIdx_ = groupLocalId;
    coreNum_ = groupSize;
    GmmClaimedTask controlResult;
    if (tilingData_->gmmSchedulerTiling.gmm1ScheduleMode == kMegaMoeGmm1ScheduleFixedWave) {
        ProcessFixedWave();
        return controlResult;
    }

    WaitEpochAcquire(
        FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncFrontMetadataReadySlot),
        kMegaMoeFixedFrontMetadataReadyMarker);
    Gmm1SwigluCvPipe cvPipe;
    const GmmClaimedTask initialTask = ProcessDirectWave0(cvPipe, *gmmPipeline);
    const bool group2DirectGmm2 = coreIdx_ >= tilingData_->fixedGroupTiling.gmm1GroupSize;
    if (group2DirectGmm2) {
        // GMM2 reuses the intra-block event ids from the GMM1->SwiGLU pipe.
        // Consume the final payload credit before this physical block enters GMM2.
        Gmm1SwigluProducerDrain(cvPipe);
        return initialTask;
    }
    return ProcessMailbox(*gmmPipeline, cvPipe, initialTask);
}

AICORE inline void Gmm1::ProcessFixedWave()
{
    WaitEpochAcquire(
        FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncFrontMetadataReadySlot),
        kMegaMoeFixedFrontMetadataReadyMarker);
    Gmm1SwigluCvPipe cvPipe;
    Gmm1Pipeline gmmPipeline;
    uint32_t groupBase = 0;
    MegaMoeCoreTileBalancer tileBalancer;
    const __gm__ MegaMoeFixedGroupTiling& fixed = tilingData_->fixedGroupTiling;
    for (uint32_t waveIdx = 0U; waveIdx < fixed.totalWaveCount; ++waveIdx) {
        coreNum_ = waveIdx < fixed.fullAicGmm1WaveCount ? fixed.physicalAicNum : fixed.gmm1GroupSize;
        if (coreIdx_ >= coreNum_) {
            break;
        }
        SetCoreTileBalancerRange(tileBalancer, 0U, coreNum_);
        const MegaMoeExpertWaveRange wave = GetExpertWaveRange(
            waveIdx, expertPerRank_, fixed.fullAicExpertsPerWave, fixed.expertsPerWave, fixed.fullAicGmm1WaveCount);
        for (uint32_t groupIdx = wave.begin; groupIdx < wave.end; ++groupIdx) {
            const uint32_t currentM = MoeCurrentMRaw(cumsumMMPtr_, rankSize_, expertPerRank_, groupIdx);
            const uint32_t coreLoops = CoreLoops(currentM);
            const uint32_t startCoreIdx = SelectCoreTileStart(tileBalancer, coreLoops);
            const uint32_t startLoopIdx = StartLoopIdx(startCoreIdx);
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum_) {
                const GmmCommonTileInfo tileInfo = BuildTileInfo(currentM, loopIdx);
                WaitDispatchTileReady(groupIdx, tileInfo);
                RunGmmPair(gmmPipeline, cvPipe, groupIdx, groupBase, tileInfo);
            }
            groupBase += currentM;
            CommitCoreTileAssignment(tileBalancer, startCoreIdx, coreLoops);
        }
        // The CV FIFO carries tile completion; retain a per-AIC wave marker for
        // zero-tile participants and the existing wave-to-GMM2 control path.
        pipe_barrier(PIPE_FIX);
        PublishGroupArrival(workspaceGM_, tilingData_, kMegaMoeFixedSyncGmm1ArrivalBase, coreIdx_, waveIdx);
    }
    pipe_barrier(PIPE_FIX);
    Gmm1SwigluProducerDrain(cvPipe);
    if (coreIdx_ < fixed.gmm1GroupSize) {
        pipe_barrier(PIPE_ALL);
        PublishScalarEpoch(
            FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncGmm1ArrivalBase + coreIdx_),
            static_cast<int32_t>(fixed.totalWaveCount + kMegaMoeFixedGmm1GroupDoneEpochOffset));
    }
}

AICORE inline GmmClaimedTask Gmm1::WaitGmm1Successor(
    GmmMailboxConsumerCursor& mailboxCursor, GmmMailboxTicketProbe* successorProbe, uint32_t dataSlotBase,
    uint32_t scaleSlotBase)
{
    const __gm__ MegaMoeGmmMailboxTiling& mailbox = tilingData_->gmmSchedulerTiling.mailbox;
    GmmClaimedTask next = WaitGmmMailboxTask(
        workspaceGM_, mailbox, tilingData_->gmmSchedulerTiling.gmm1, kGmmMailboxFirstTaskTicket, coreIdx_,
        mailboxCursor, GmmTaskStage::kGmm1, successorProbe);
    const bool nextIsGmm2 = next.ticket >= mailbox.gmm2TicketBase && next.ticket < kGmmMailboxTerminalTicket;
    if (next.stageTransition) {
        next = ResolveGmmMailboxStageTransition(
            workspaceGM_, tilingData_->gmmSchedulerTiling.gmm2, mailbox.gmm2TicketBase, next);
    }
    if (!next.valid) {
        return next;
    }

    next.dataSlotBase = dataSlotBase;
    next.scaleSlotBase = scaleSlotBase;
    return next;
}

AICORE inline GmmClaimedTask Gmm1::ProcessDirectWave0(Gmm1SwigluCvPipe& cvPipe, Gmm1Pipeline& gmmPipeline)
{
    GmmClaimedTask successor;
    const __gm__ MegaMoeFixedGroupTiling& fixed = tilingData_->fixedGroupTiling;
    const MegaMoeExpertWaveRange wave = GetExpertWaveRange(
        0U, expertPerRank_, fixed.fullAicExpertsPerWave, fixed.expertsPerWave, fixed.fullAicGmm1WaveCount);
    coreNum_ = fixed.physicalAicNum;
    DirectWaveCursor directCursor;
    directCursor.Init(this, wave.end);
    const bool hasDirectAssignment = directCursor.Advance();

    const __gm__ MegaMoeGmmMailboxTiling& mailbox = tilingData_->gmmSchedulerTiling.mailbox;
    GmmMailboxConsumerCursor mailboxCursor;
    const bool group2DirectGmm2 = coreIdx_ >= fixed.gmm1GroupSize;
    if (!hasDirectAssignment) {
        PublishGmmMailboxProgress(workspaceGM_, mailbox, coreIdx_, kGmmMailboxGmm1Wave0EndTicket);
        if (group2DirectGmm2) {
            return successor;
        }
        GmmMailboxTicketProbe successorProbe;
        ProbeGmmMailboxSuccessorTicket(workspaceGM_, mailbox, coreIdx_, kGmmMailboxGmm1Wave0EndTicket, successorProbe);
        return WaitGmm1Successor(mailboxCursor, &successorProbe, 0U, 0U);
    }

    while (true) {
        const DirectWaveAssignment assignment = directCursor.Current();
        const MegaMoeGmmTask task = assignment.task;
        const GmmCommonTileInfo tileInfo = assignment.tileInfo;
        WaitDispatchTileReady(task.expert, tileInfo);
        Gmm1Pipeline::TileRun xRun;
        Gmm1Pipeline::TileRun gateRun;
        BuildGmmPairRuns(task, tileInfo, 0U, 0U, xRun, gateRun);
        GmmDirectWaveNextAssignmentProbe<DirectWaveCursor> panelProbe(
            workspaceGM_, mailbox, coreIdx_, kGmmMailboxGmm1Wave0EndTicket, directCursor, GmmTaskStage::kGmm1);
        gmmPipeline.ComputePairDirect(cvPipe, xRun, gateRun, panelProbe);
        const bool finalLocalTile = !panelProbe.HasNextAssignment();
        const uint32_t nextDataSlotBase = Gmm1Pipeline::AdvanceDataSlotBase(gateRun.dataSlotBase, problemK_);
        const uint32_t nextScaleSlotBase = Gmm1Pipeline::AdvanceScaleSlotBase(gateRun.scaleSlotBase, problemK_);
        GmmMailboxTicketProbe* successorProbe = panelProbe.SuccessorProbe();
        if (finalLocalTile && !group2DirectGmm2 && successorProbe->ready) {
            successor = WaitGmm1Successor(mailboxCursor, successorProbe, nextDataSlotBase, nextScaleSlotBase);
        }
        gmmPipeline.EnqueuePairDirect(cvPipe, tileInfo.actualM, tileInfo.actualN);
        gmmPipeline.RecordPairDirect(cvPipe);
        if (finalLocalTile && !group2DirectGmm2 && successor.ticket == kGmmMailboxEmptyTicket) {
            successor = WaitGmm1Successor(mailboxCursor, successorProbe, nextDataSlotBase, nextScaleSlotBase);
        }
        if (finalLocalTile) {
            gmmPipeline.DrainPairDirectStore();
        }
        if (finalLocalTile) {
            break;
        }
    }
    return successor;
}

AICORE inline GmmClaimedTask Gmm1::ProcessMailbox(
    Gmm1Pipeline& gmmPipeline, Gmm1SwigluCvPipe& cvPipe, const GmmClaimedTask& initialTask)
{
    GmmClaimedTask controlResult;
    const __gm__ MegaMoeGmmMailboxTiling& mailbox = tilingData_->gmmSchedulerTiling.mailbox;
    uint32_t dataSlotBase = initialTask.dataSlotBase;
    uint32_t scaleSlotBase = initialTask.scaleSlotBase;

    GmmMailboxConsumerCursor mailboxCursor = initialTask.mailboxCursor;
    GmmClaimedTask current;
    if (initialTask.ticket != kGmmMailboxEmptyTicket) {
        current = initialTask;
    } else {
        current = WaitGmmMailboxTask(
            workspaceGM_, mailbox, tilingData_->gmmSchedulerTiling.gmm1, kGmmMailboxFirstTaskTicket, coreIdx_,
            mailboxCursor, GmmTaskStage::kGmm1, nullptr);
    }
    if (current.stageTransition) {
        current = ResolveGmmMailboxStageTransition(
            workspaceGM_, tilingData_->gmmSchedulerTiling.gmm2, mailbox.gmm2TicketBase, current);
    }
    const bool initialIsGmm2 = current.ticket >= mailbox.gmm2TicketBase && current.ticket < kGmmMailboxTerminalTicket;
    if (current.valid && !initialIsGmm2) {
        PublishGmm1SwigluControl(cvPipe, current.task);
    }
    while (current.valid && !initialIsGmm2) {
        const MegaMoeGmmTask currentTask = current.task;
        const GmmCommonTileInfo tileInfo =
            GmmCommonBuildTileInfoFromCoord(currentTask.currentM, outputN_, currentTask.blockM, currentTask.blockN);
        Gmm1Pipeline::TileRun xRun;
        Gmm1Pipeline::TileRun gateRun;
        BuildGmmPairRuns(currentTask, tileInfo, dataSlotBase, scaleSlotBase, xRun, gateRun);
        GmmMailboxPanelProbe panelProbe(workspaceGM_, mailbox, coreIdx_, current.ticket, GmmTaskStage::kGmm1);
        gmmPipeline.ComputePairDirect(cvPipe, xRun, gateRun, panelProbe);
        const uint32_t nextDataSlotBase = Gmm1Pipeline::AdvanceDataSlotBase(gateRun.dataSlotBase, problemK_);
        const uint32_t nextScaleSlotBase = Gmm1Pipeline::AdvanceScaleSlotBase(gateRun.scaleSlotBase, problemK_);
        GmmMailboxTicketProbe* successorProbe = panelProbe.SuccessorProbe();
        GmmClaimedTask next;
        if (successorProbe->ready) {
            next = WaitGmm1Successor(mailboxCursor, successorProbe, nextDataSlotBase, nextScaleSlotBase);
        }
        gmmPipeline.EnqueuePairDirect(cvPipe, tileInfo.actualM, tileInfo.actualN);
        gmmPipeline.RecordPairDirect(cvPipe);
        if (next.ticket == kGmmMailboxEmptyTicket) {
            next = WaitGmm1Successor(mailboxCursor, successorProbe, nextDataSlotBase, nextScaleSlotBase);
        }
        const bool nextIsGmm2 = next.ticket >= mailbox.gmm2TicketBase && next.ticket < kGmmMailboxTerminalTicket;
        if (next.valid && !nextIsGmm2) {
            PublishGmm1SwigluControl(cvPipe, next.task);
        }
        gmmPipeline.DrainPairDirectStore();
        dataSlotBase = nextDataSlotBase;
        scaleSlotBase = nextScaleSlotBase;
        current = next;
        if (nextIsGmm2) {
            break;
        }
    }
    const bool terminalTicketContinuesToGmm2 = (current.task.flags & kGmmTaskFlagTerminal) != 0U;
    const bool nextTicketIsGmm2 =
        current.ticket >= mailbox.gmm2TicketBase && current.ticket < kGmmMailboxTerminalTicket;
    const bool continuesWithGmm2 = nextTicketIsGmm2 || terminalTicketContinuesToGmm2;

    // GMM2 reuses the event ids from both GMM1->SwiGLU rings. Always drain
    // the outstanding payload credit, including the mailbox transition path.
    Gmm1SwigluProducerDrain(cvPipe);
    PublishGmm1SwigluControl(cvPipe, Gmm1SwigluStageEndControl());
    // Drain the stage-end control acknowledgement before GMM2 can reuse the
    // control event ids as free credits.
    Gmm1SwigluControlProducerDrain(cvPipe);
    pipe_barrier(PIPE_FIX);
    if (current.ticket != kGmmMailboxEmptyTicket && !nextTicketIsGmm2 && !terminalTicketContinuesToGmm2) {
        PublishGmmMailboxProgress(workspaceGM_, mailbox, coreIdx_, current.ticket);
    }
    controlResult = current;
    // Group1 reaches this point only after resolving its GMM1-to-GMM2 handoff.
    if (!continuesWithGmm2) {
        pipe_barrier(PIPE_ALL);
    }
    PublishScalarEpoch(
        FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncGmm1ArrivalBase + coreIdx_),
        kMegaMoeFixedGmm1DoneMarker);
    return controlResult;
}

#endif // DISPATCH_MEGA_COMBINE_GMM1_H
