/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_COMBINE_H
#define DISPATCH_MEGA_COMBINE_COMBINE_H

#include <type_traits>

#include <pto/pto-inst.hpp>

#include "dispatch_mega_combine_tiling.h"
#include "gmm2_combine_cv_pipe.h"
#include "gmm_common.h"
#include "gmm_task_queue_device.h"
#include "kernel_operator.h"
#include "utils/const_args.hpp"
#include "utils/mega_wave_schedule.hpp"
#include "utils/hccl_window.hpp"
#include "utils/mega_expert_sync.hpp"
#include "utils/pto_vector.hpp"

constexpr uint32_t kDirectCombineMetadataMaxElems = kMegaMoeExpertProgressMaxRanks * kMegaMoeFixedMaxExperts;
constexpr uint64_t kDirectCombineCumsumUbOffset = kGmm2CombineMetadataOffset;
constexpr uint64_t kDirectCombinePreSumUbOffset =
    kDirectCombineCumsumUbOffset + static_cast<uint64_t>(kDirectCombineMetadataMaxElems) * sizeof(int32_t);
constexpr uint64_t kDirectCombineMetadataUbEnd =
    kDirectCombinePreSumUbOffset + static_cast<uint64_t>(kDirectCombineMetadataMaxElems) * sizeof(int32_t);
constexpr uint64_t kDirectCombineCompletionUbOffset = REMOTE_WINDOW_PROGRESS_SIGNAL_UB_OFFSET;
constexpr uint64_t kDirectCombineCompletionUbBytes =
    static_cast<uint64_t>(kMegaMoeFixedMaxExperts) * REMOTE_WINDOW_READY_SIGNAL_SLOT_BYTES;

static_assert(kDirectCombineMetadataUbEnd <= kGmm2CombineMetadataOffset + kGmm2CombineMetadataBytes);
static_assert(
    kDirectCombineCompletionUbOffset + kDirectCombineCompletionUbBytes <= REMOTE_WINDOW_FINAL_SIGNAL_UB_OFFSET);

template <typename CvTile>
__tf__ AICORE inline void DirectCombineStrideStore(
    __gm__ bfloat16_t* dst, typename CvTile::TileDType __in__ srcTile, uint32_t rows, uint32_t cols,
    uint32_t dstLeadingDim)
{
    __ubuf__ bfloat16_t* src = reinterpret_cast<__ubuf__ bfloat16_t*>(__cce_get_tile_ptr(srcTile));
    copy_ubuf_to_gm_align_v2(
        dst, src, 0, rows, cols * sizeof(bfloat16_t), 0, static_cast<uint64_t>(dstLeadingDim) * sizeof(bfloat16_t),
        CvTile::Cols * sizeof(bfloat16_t));
}

template <typename OutputElement>
class Combine {
public:
    AICORE inline void Init(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData);
    AICORE inline void ProcessFixed(uint32_t physicalBlockId);

private:
    static_assert(std::is_same_v<OutputElement, bfloat16_t>, "MXFP8 combine output must be BF16");

    AICORE inline void PublishCompletedExpertTiles(
        const __gm__ MegaMoeGmmQueueTiling& queue, uint32_t expert, uint32_t tileCount) const;
    AICORE inline void PrefetchDirectMetadata();
    AICORE inline void PrepareDirectExpert(uint32_t groupIdx);
    AICORE inline void StoreDirectTile(const GmmCommonTileInfo& tileInfo) const;
    AICORE inline void ConsumeDirectTile(const GmmCommonTileInfo& tileInfo);
    AICORE inline void ConsumeDirectWave0();
    AICORE inline void ProcessImpl();

    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;
    Gmm2CombineCvPipe cvPipe_;
    __gm__ int32_t* cumsumMMPtr_ = nullptr;
    __gm__ int32_t* preSumBeforeRankPtr_ = nullptr;
    __gm__ OutputElement* remoteOutputBase_[kMegaMoeExpertProgressMaxRanks] = {nullptr};

    uint32_t problemK_ = 0U;
    uint32_t routeElems_ = 0U;
    uint32_t expertPerRank_ = 0U;
    uint32_t rankSize_ = 0U;
    uint32_t physicalBlockId_ = 0U;
    uint32_t directRankRowBegin_[kMegaMoeExpertProgressMaxRanks] = {0U};
    uint32_t directRankRowEnd_[kMegaMoeExpertProgressMaxRanks] = {0U};
    uint32_t directCompactRowBegin_[kMegaMoeExpertProgressMaxRanks] = {0U};
};

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::Init(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData)
{
    workspaceGM_ = workspaceGM;
    tilingData_ = tilingData;

    problemK_ = tilingData_->megaMoeInfo.K;
    routeElems_ = tilingData_->frontReorderTiling.routeElems;
    expertPerRank_ = tilingData_->megaMoeInfo.expertPerRank;
    rankSize_ = tilingData_->runtimeInfo.rankSize;

    PtoRemoteWindow remoteWindow;
    remoteWindow.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
    MegaMoePeerMemoryLayout peerMemoryLayout;
    peerMemoryLayout.Init(tilingData_->frontReorderTiling);
    cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t*>(workspaceGM + tilingData_->frontReorderTiling.cumsumMMOffset);
    preSumBeforeRankPtr_ =
        reinterpret_cast<__gm__ int32_t*>(remoteWindow.LocalBase() + peerMemoryLayout.preSumBeforeRank);
    for (uint32_t srcRank = 0U; srcRank < rankSize_; ++srcRank) {
        remoteOutputBase_[srcRank] = reinterpret_cast<__gm__ OutputElement*>(
            remoteWindow.RemoteBase(peerMemoryLayout.combineOutputByRouteSlot, static_cast<int32_t>(srcRank)));
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::PublishCompletedExpertTiles(
    const __gm__ MegaMoeGmmQueueTiling& queue, uint32_t expert, uint32_t tileCount) const
{
    if (tileCount == 0U || expert >= expertPerRank_) {
        return;
    }
    // Expert progress is an acquire/release boundary for live Unpermute.
    // Drain this consumer's remote output stores before incrementing the
    // completion counter; merely ordering both operations on MTE3 allows the
    // local counter update to become visible before a remote store arrives.
    pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
    const uint64_t scratchUbOffset =
        kDirectCombineCompletionUbOffset + static_cast<uint64_t>(expert) * REMOTE_WINDOW_READY_SIGNAL_SLOT_BYTES;
    PtoSetValue<int32_t, REMOTE_WINDOW_SYNC_VALUES_PER_SLOT>(scratchUbOffset, 0U, static_cast<int32_t>(tileCount));
    pto::PtoSetWaitFlag<PIPE_S, PIPE_MTE3>();
    PtoStoreAtomicAddVector<int32_t, REMOTE_WINDOW_SYNC_VALUES_PER_SLOT>(
        GmmExpertCompletionSlot(workspaceGM_, queue, expert), scratchUbOffset, 1U);
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::PrefetchDirectMetadata()
{
    const uint32_t metadataElems = rankSize_ * expertPerRank_;
    PtoLoadVector<int32_t, kDirectCombineMetadataMaxElems>(kDirectCombineCumsumUbOffset, cumsumMMPtr_, metadataElems);
    PtoLoadVector<int32_t, kDirectCombineMetadataMaxElems>(
        kDirectCombinePreSumUbOffset, preSumBeforeRankPtr_, metadataElems);
    pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::PrepareDirectExpert(uint32_t groupIdx)
{
    uint32_t rankRowBegin = 0U;
    for (uint32_t srcRank = 0U; srcRank < rankSize_; ++srcRank) {
        const uint32_t metadataIdx = srcRank * expertPerRank_ + groupIdx;
        const uint32_t rankRowEnd = static_cast<uint32_t>(
            PtoGetValue<int32_t, kDirectCombineMetadataMaxElems>(kDirectCombineCumsumUbOffset, metadataIdx));
        directRankRowBegin_[srcRank] = rankRowBegin;
        directRankRowEnd_[srcRank] = rankRowEnd;
        directCompactRowBegin_[srcRank] = static_cast<uint32_t>(
            PtoGetValue<int32_t, kDirectCombineMetadataMaxElems>(kDirectCombinePreSumUbOffset, metadataIdx));
        rankRowBegin = rankRowEnd;
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::StoreDirectTile(const GmmCommonTileInfo& tileInfo) const
{
    const uint32_t tileRowBegin = tileInfo.blockRowStart;
    const uint32_t tileRowEnd = tileRowBegin + tileInfo.actualM;
    for (uint32_t srcRank = 0U; srcRank < rankSize_; ++srcRank) {
        const uint32_t rankRowBegin = directRankRowBegin_[srcRank];
        const uint32_t rankRowEnd = directRankRowEnd_[srcRank];
        if (rankRowBegin >= tileRowEnd) {
            break;
        }
        if (rankRowEnd <= tileRowBegin) {
            continue;
        }
        const uint32_t intersectionBegin = rankRowBegin > tileRowBegin ? rankRowBegin : tileRowBegin;
        const uint32_t intersectionEnd = rankRowEnd < tileRowEnd ? rankRowEnd : tileRowEnd;
        if (intersectionEnd <= intersectionBegin) {
            continue;
        }
        const uint32_t rows = intersectionEnd - intersectionBegin;
        const uint32_t srcTileRow = intersectionBegin - tileRowBegin;
        const uint32_t compactRow = directCompactRowBegin_[srcRank] + intersectionBegin - rankRowBegin;
        if (compactRow >= routeElems_ || rows > routeElems_ - compactRow) {
            continue;
        }
        __gm__ OutputElement* dstBase = remoteOutputBase_[srcRank];
        if (dstBase == nullptr) {
            continue;
        }
        __gm__ OutputElement* dst = dstBase + static_cast<uint64_t>(compactRow) * problemK_ + tileInfo.blockColStart;
        const uint64_t srcOffset = kGmm2CombineCvBufferOffset +
                                   static_cast<uint64_t>(srcTileRow) * kGmm2CombineCvTileCols * sizeof(bfloat16_t);
        Gmm2CombineCvTile srcTile(rows, tileInfo.actualN);
        pto::TASSIGN(srcTile, srcOffset);
        DirectCombineStrideStore<Gmm2CombineCvTile>(dst, srcTile.data(), rows, tileInfo.actualN, problemK_);
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::ConsumeDirectTile(const GmmCommonTileInfo& tileInfo)
{
    // Keep descriptor/expert preparation on Scalar ahead of this wait. The
    // ready event still orders all payload reads behind the AIC FIX TMOV.
    Gmm2CombineConsumerEnqueueReadyWait(cvPipe_);
    StoreDirectTile(tileInfo);
    Gmm2CombineConsumerRelease(cvPipe_);
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::ConsumeDirectWave0()
{
    const __gm__ MegaMoeFixedGroupTiling& fixed = tilingData_->fixedGroupTiling;
    if (physicalBlockId_ < fixed.gmm1GroupSize) {
        return;
    }
    const uint32_t group2LocalId = physicalBlockId_ - fixed.gmm1GroupSize;
    const MegaMoeExpertWaveRange wave = GetExpertWaveRange(
        0U, expertPerRank_, fixed.fullAicExpertsPerWave, fixed.expertsPerWave, fixed.fullAicGmm1WaveCount);
    const __gm__ MegaMoeGmmQueueTiling& queue = tilingData_->gmmSchedulerTiling.gmm2;
    MegaMoeCoreTileBalancer tileBalancer;
    SetCoreTileBalancerRange(tileBalancer, fixed.gmm1GroupSize, fixed.gmm2GroupSize);
    for (uint32_t expert = wave.begin; expert < wave.end; ++expert) {
        const uint32_t currentM = MoeCurrentMRaw(cumsumMMPtr_, rankSize_, expertPerRank_, expert);
        const uint32_t coreLoops = GmmCommonCoreLoops(currentM, problemK_);
        const uint32_t startCoreIdx = SelectCoreTileStart(tileBalancer, coreLoops);
        const uint32_t startLoopIdx = GmmCommonStartLoopIdx(group2LocalId, fixed.gmm2GroupSize, startCoreIdx);
        uint32_t assignedTileCount = 0U;
        if (startLoopIdx < coreLoops) {
            PrepareDirectExpert(expert);
        }
        for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += fixed.gmm2GroupSize) {
            const GmmCommonTileInfo tileInfo =
                GmmCommonBuildTileInfoWithOffset<kGmm2CombineSwizzleOffset>(currentM, problemK_, loopIdx);
            ConsumeDirectTile(tileInfo);
            ++assignedTileCount;
        }
        PublishCompletedExpertTiles(queue, expert, assignedTileCount);
        CommitCoreTileAssignment(tileBalancer, startCoreIdx, coreLoops);
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::ProcessImpl()
{
    const __gm__ MegaMoeGmmQueueTiling& queue = tilingData_->gmmSchedulerTiling.gmm2;
    ConsumeDirectWave0();

    uint32_t preparedExpert = expertPerRank_;
    uint32_t completionExpert = expertPerRank_;
    uint32_t completionTileCount = 0U;
    GmmCvTaskInferenceCache inferenceCache;
    while (true) {
        // Both rings use the same logical task sequence. Copy the packed
        // control first; payload ownership remains unchanged until MTE3
        // finishes consuming the matching CV slot.
        Gmm2CombineControlConsumerWait(cvPipe_);
        const uint32_t control = ReadGmmCvTaskControl(0U, kGmm2CombineControlFifoDepth);
        Gmm2CombineControlConsumerRelease(cvPipe_);
        if (IsGmmStageEndControl(control)) {
            break;
        }
        const MegaMoeGmmTask task = InferGmmCvTask(control, cumsumMMPtr_, rankSize_, expertPerRank_, inferenceCache);

        const GmmCommonTileInfo tileInfo =
            GmmCommonBuildTileInfoFromCoord(task.currentM, problemK_, task.blockM, task.blockN);
        if (completionExpert != task.expert) {
            PublishCompletedExpertTiles(queue, completionExpert, completionTileCount);
            completionExpert = task.expert;
            completionTileCount = 0U;
        }
        if (preparedExpert != task.expert) {
            PrepareDirectExpert(task.expert);
            preparedExpert = task.expert;
        }
        ConsumeDirectTile(tileInfo);
        ++completionTileCount;
    }
    PublishCompletedExpertTiles(queue, completionExpert, completionTileCount);
    // Drain only this consumer's ordered data/completion MTE3 chain before
    // the paired AIV can continue into Unpermute. Global readiness is
    // published once by Gmm2ExpertProgressCoordinator.
    pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::ProcessFixed(uint32_t physicalBlockId)
{
    physicalBlockId_ = physicalBlockId;
    // The Group2 AIV0 coordinator publishes the common start marker only
    // after local Dispatch and all direct-route metadata are ready.
    PrefetchDirectMetadata();
    ProcessImpl();
}

#endif // DISPATCH_MEGA_COMBINE_COMBINE_H
