/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_SWIGLU_H
#define DISPATCH_MEGA_COMBINE_SWIGLU_H

#include "kernel_operator.h"

#include <type_traits>

#include <pto/pto-inst.hpp>

#include "dispatch_mega_combine_tiling.h"
#include "gmm_common.h"
#include "gmm1_swiglu_cv_pipe.h"
#include "gmm_task_queue_device.h"
#include "utils/const_args.hpp"
#include "utils/mega_expert_sync.hpp"
#include "utils/mega_wave_schedule.hpp"
#include "utils/pto_vector.hpp"

constexpr uint32_t kDirectSwigluTileRows = kGmm1SwigluCvTileRows;
constexpr uint32_t kDirectSwigluOutputCols = kGmm1SwigluCvOutputCols;
constexpr uint32_t kDirectSwigluTileElems = kDirectSwigluTileRows * kDirectSwigluOutputCols;
constexpr uint32_t kDirectSwigluScaleElems = kDirectSwigluTileElems / kMegaMoeMxGroupSize;
constexpr uint32_t kDirectSwigluWorkBytes = kDirectSwigluTileElems * sizeof(bfloat16_t);
constexpr uint32_t kDirectSwigluFp8Bytes = kDirectSwigluTileElems * sizeof(int8_t);
constexpr uint32_t kDirectSwigluE8Bytes = kDirectSwigluScaleElems * sizeof(uint8_t);
constexpr uint32_t kDirectSwigluMaxBytes = kDirectSwigluScaleElems * sizeof(bfloat16_t);
constexpr uint32_t kDirectSwigluScalingBytes = kDirectSwigluScaleElems * sizeof(bfloat16_t);
// Activation overwrites x, then quantization scratch reuses the consumed gate region.
constexpr uint32_t kDirectSwigluWorkOffset = kGmm1SwigluCvXOffset;
constexpr uint32_t kDirectSwigluFp8Offset = kGmm1SwigluCvGateOffset;
constexpr uint32_t kDirectSwigluE8Offset = kDirectSwigluFp8Offset + kDirectSwigluFp8Bytes;
constexpr uint32_t kDirectSwigluMaxOffset = kDirectSwigluE8Offset + kDirectSwigluE8Bytes;
constexpr uint32_t kDirectSwigluScalingOffset = kDirectSwigluMaxOffset + kDirectSwigluMaxBytes;
constexpr uint32_t kDirectSwigluRequiredUbBytes = kDirectSwigluScalingOffset + kDirectSwigluScalingBytes;
constexpr uint32_t kDirectSwigluStoreEvent = 0U;

using DirectSwigluBf16Tile = pto::Tile<
    pto::TileType::Vec, bfloat16_t, kDirectSwigluTileRows, kDirectSwigluOutputCols, pto::BLayout::RowMajor,
    pto::DYNAMIC, pto::DYNAMIC>;
using DirectSwigluQuantSrcTile = pto::Tile<
    pto::TileType::Vec, bfloat16_t, kDirectSwigluTileRows, kDirectSwigluOutputCols, pto::BLayout::RowMajor,
    pto::DYNAMIC, pto::DYNAMIC, pto::SLayout::NoneBox, 512, pto::PadValue::Zero>;
using DirectSwigluQuantFp8Tile = pto::Tile<
    pto::TileType::Vec, int8_t, kDirectSwigluTileRows, kDirectSwigluOutputCols, pto::BLayout::RowMajor, pto::DYNAMIC,
    pto::DYNAMIC, pto::SLayout::NoneBox, 512, pto::PadValue::Zero>;
using DirectSwigluQuantE8Tile = pto::Tile<
    pto::TileType::Vec, uint8_t, 1, kDirectSwigluScaleElems, pto::BLayout::RowMajor, pto::DYNAMIC, pto::DYNAMIC,
    pto::SLayout::NoneBox, 512, pto::PadValue::Zero>;
using DirectSwigluQuantScaleTile = pto::Tile<
    pto::TileType::Vec, bfloat16_t, 1, kDirectSwigluScaleElems, pto::BLayout::RowMajor, pto::DYNAMIC, pto::DYNAMIC>;

static_assert(kDirectSwigluRequiredUbBytes <= A5_MAIN_UB_SIZE);
static_assert(kGmm1SwigluCvSlotBytes == AtlasA5::UB_SIZE);

template <typename DstTile, typename SrcTile>
__tf__ AICORE inline void DirectSwigluActivation(
    typename DstTile::TileDType __out__ dstData, typename SrcTile::TileDType __in__ xData,
    typename SrcTile::TileDType __in__ gateData, uint32_t rows, uint32_t outputCols)
{
    __ubuf__ bfloat16_t* dst = reinterpret_cast<__ubuf__ bfloat16_t*>(__cce_get_tile_ptr(dstData));
    __ubuf__ bfloat16_t* x = reinterpret_cast<__ubuf__ bfloat16_t*>(__cce_get_tile_ptr(xData));
    __ubuf__ bfloat16_t* gate = reinterpret_cast<__ubuf__ bfloat16_t*>(__cce_get_tile_ptr(gateData));
    constexpr uint32_t kFp32VectorElems = 64U;
    const uint16_t repeatCount = static_cast<uint16_t>((outputCols + kFp32VectorElems - 1U) / kFp32VectorElems);
    __VEC_SCOPE__
    {
        pto::RegTensor<bfloat16_t> xBf16;
        pto::RegTensor<bfloat16_t> gateBf16;
        pto::RegTensor<bfloat16_t> outBf16;
        pto::RegTensor<float> xFp32;
        pto::RegTensor<float> gateFp32;
        pto::RegTensor<float> negFp32;
        pto::RegTensor<float> expFp32;
        pto::RegTensor<float> denominatorFp32;
        pto::RegTensor<float> siluFp32;
        pto::RegTensor<float> outFp32;
        uint32_t fullBf16Count = 2U * kFp32VectorElems;
        pto::MaskReg fullBf16Mask = pto::CreatePredicate<bfloat16_t>(fullBf16Count);
        for (uint16_t row = 0U; row < static_cast<uint16_t>(rows); ++row) {
            uint32_t remaining = outputCols;
            for (uint16_t repeat = 0U; repeat < repeatCount; ++repeat) {
                const uint32_t offset = static_cast<uint32_t>(row) * kDirectSwigluOutputCols +
                                        static_cast<uint32_t>(repeat) * kFp32VectorElems;
                pto::MaskReg mask = pto::CreatePredicate<float>(remaining);
                vlds(xBf16, x, offset, UNPK_B16);
                vlds(gateBf16, gate, offset, UNPK_B16);
                vcvt(xFp32, xBf16, fullBf16Mask, PART_EVEN);
                vcvt(gateFp32, gateBf16, fullBf16Mask, PART_EVEN);
                vmuls(negFp32, xFp32, -1.0f, mask, MODE_ZEROING);
                vexp(expFp32, negFp32, mask, MODE_ZEROING);
                vadds(denominatorFp32, expFp32, 1.0f, mask, MODE_ZEROING);
                vdiv(siluFp32, xFp32, denominatorFp32, mask, MODE_ZEROING);
                vmul(outFp32, siluFp32, gateFp32, mask, MODE_ZEROING);
                vcvt(outBf16, outFp32, mask, ROUND_R, RS_DISABLE, PART_EVEN);
                vsts(outBf16, dst, offset, PK_B32, mask);
            }
        }
    }
}

template <typename DataTile, typename ScaleTile>
__tf__ AICORE inline void DirectSwigluStore(
    __gm__ int8_t* dataDst, __gm__ uint8_t* scaleDst, typename DataTile::TileDType __in__ dataTile,
    typename ScaleTile::TileDType __in__ scaleTile, uint32_t rows, uint32_t dataCols, uint32_t dataLeadingDim,
    uint32_t scaleCols, uint32_t scaleLeadingDim)
{
    __ubuf__ int8_t* dataSrc = reinterpret_cast<__ubuf__ int8_t*>(__cce_get_tile_ptr(dataTile));
    __ubuf__ uint8_t* scaleSrc = reinterpret_cast<__ubuf__ uint8_t*>(__cce_get_tile_ptr(scaleTile));
    copy_ubuf_to_gm_align_v2(dataDst, dataSrc, 0, rows, dataCols, 0, dataLeadingDim, kDirectSwigluOutputCols);
    copy_ubuf_to_gm_align_v2(scaleDst, scaleSrc, 0, rows, scaleCols, 0, scaleLeadingDim, scaleCols);
}

template <typename InputElement>
class Swiglu {
public:
    AICORE inline void Init(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData);
    AICORE inline void ProcessFixed(uint32_t groupLocalId, uint32_t groupSize);

private:
    AICORE inline uint32_t CoreLoops(uint32_t currentM) const { return GmmCommonCoreLoops(currentM, outputN_); }

    AICORE inline uint32_t StartLoopIdx(uint32_t startCoreIdx) const
    {
        return GmmCommonStartLoopIdx(coreIdx_, coreNum_, startCoreIdx);
    }

    AICORE inline GmmCommonTileInfo BuildPairTileInfo(uint32_t currentM, uint32_t loopIdx) const
    {
        return GmmCommonBuildTileInfo(currentM, outputN_, loopIdx);
    }

    AICORE inline void ComputeDirectPayload(
        Gmm1SwigluCvPipe& cvPipe, uint32_t globalRow, uint32_t pairN, uint32_t pairCol, uint32_t rows) const
    {
        constexpr uint64_t slotBase = kGmm1SwigluCvBufferOffset;
        Gmm1SwigluConsumerWait(cvPipe);

        DirectSwigluBf16Tile xTile(rows, pairN);
        DirectSwigluBf16Tile gateTile(rows, pairN);
        DirectSwigluQuantSrcTile srcTile(rows, pairN);
        pto::TASSIGN(xTile, slotBase + kGmm1SwigluCvXOffset);
        pto::TASSIGN(gateTile, slotBase + kGmm1SwigluCvGateOffset);
        pto::TASSIGN(srcTile, kDirectSwigluWorkOffset);
        DirectSwigluActivation<DirectSwigluQuantSrcTile, DirectSwigluBf16Tile>(
            srcTile.data(), xTile.data(), gateTile.data(), rows, pairN);

        const uint32_t scaleCols = pairN / kMegaMoeMxGroupSize;
        DirectSwigluQuantFp8Tile fp8Tile(rows, pairN);
        DirectSwigluQuantE8Tile e8Tile(1U, rows * kDirectSwigluOutputCols / kMegaMoeMxGroupSize);
        DirectSwigluQuantScaleTile maxTile(1U, rows * kDirectSwigluOutputCols / kMegaMoeMxGroupSize);
        DirectSwigluQuantScaleTile scalingTile(1U, rows * kDirectSwigluOutputCols / kMegaMoeMxGroupSize);
        pto::TASSIGN(fp8Tile, kDirectSwigluFp8Offset);
        pto::TASSIGN(e8Tile, kDirectSwigluE8Offset);
        pto::TASSIGN(maxTile, kDirectSwigluMaxOffset);
        pto::TASSIGN(scalingTile, kDirectSwigluScalingOffset);
        pto::TQUANT<
            pto::QuantType::MXFP8, DirectSwigluQuantFp8Tile, DirectSwigluQuantSrcTile, DirectSwigluQuantE8Tile,
            DirectSwigluQuantScaleTile, DirectSwigluQuantScaleTile, pto::QuantScaleAlg::OCP>(
            fp8Tile, srcTile, &e8Tile, &maxTile, &scalingTile);

        set_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(kDirectSwigluStoreEvent));
        wait_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(kDirectSwigluStoreEvent));
        __gm__ int8_t* dataDst =
            reinterpret_cast<__gm__ int8_t*>(gmSwigluAPtr_) + static_cast<uint64_t>(globalRow) * outputN_ + pairCol;
        __gm__ uint8_t* scaleDst = reinterpret_cast<__gm__ uint8_t*>(gmSwigluScalePtr_) +
                                   static_cast<uint64_t>(globalRow) * ScaleCols() + pairCol / kMegaMoeMxGroupSize;
        DirectSwigluStore<DirectSwigluQuantFp8Tile, DirectSwigluQuantE8Tile>(
            dataDst, scaleDst, fp8Tile.data(), e8Tile.data(), rows, pairN, outputN_, scaleCols, ScaleCols());
        set_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(kDirectSwigluStoreEvent));
        wait_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(kDirectSwigluStoreEvent));
        // Quant output aliases the producer slot, so it is reusable only after both GM stores complete.
        Gmm1SwigluConsumerRelease(cvPipe);
    }

    AICORE inline void ConsumeDirectWave0(Gmm1SwigluCvPipe& cvPipe) const
    {
        const __gm__ MegaMoeFixedGroupTiling& fixed = tilingData_->fixedGroupTiling;
        const MegaMoeExpertWaveRange wave = GetExpertWaveRange(
            0U, expertPerRank_, fixed.fullAicExpertsPerWave, fixed.expertsPerWave, fixed.fullAicGmm1WaveCount);
        const uint32_t participantCount = fixed.physicalAicNum;
        uint32_t groupBase = 0U;
        MegaMoeCoreTileBalancer tileBalancer;
        SetCoreTileBalancerRange(tileBalancer, 0U, participantCount);
        for (uint32_t expert = 0U; expert < wave.end; ++expert) {
            const uint32_t currentM = MoeCurrentMRaw(cumsumMMPtr_, rankSize_, expertPerRank_, expert);
            const uint32_t coreLoops = CoreLoops(currentM);
            const uint32_t startCoreIdx = SelectCoreTileStart(tileBalancer, coreLoops);
            const uint32_t startLoopIdx = StartLoopIdx(startCoreIdx);
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += participantCount) {
                const GmmCommonTileInfo tileInfo = BuildPairTileInfo(currentM, loopIdx);
                ComputeDirectPayload(
                    cvPipe, groupBase + tileInfo.blockRowStart, tileInfo.actualN, tileInfo.blockColStart,
                    tileInfo.actualM);
                PublishGmm2InputExpertTileReady(expert);
            }
            groupBase += currentM;
            CommitCoreTileAssignment(tileBalancer, startCoreIdx, coreLoops);
        }
        dsb(DSB_DDR);
    }

    AICORE inline uint32_t ScaleCols() const { return outputN_ / kMegaMoeMxGroupSize; }

    AICORE inline void PublishGmm2InputExpertTileReady(uint32_t expert) const
    {
        const __gm__ MegaMoeGmmQueueTiling& queue = tilingData_->gmmSchedulerTiling.gmm2;
        __gm__ int32_t* slot = GmmTaskDependencySlot(
            workspaceGM_, queue, tilingData_->dispatchTiling.readyCountMaxTilesPerExpert, expert, 0U);
        // Do not let the scalar counter overtake the SwiGLU GM stores.
        pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
        dsb(DSB_DDR);
        (void)atomicAdd(slot, 1);
    }

    AICORE inline void ProcessFixedWave();
    AICORE inline void ProcessHybrid();

    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;
    __gm__ float8_e4m3_t* gmSwigluAPtr_ = nullptr;
    __gm__ float8_e8m0_t* gmSwigluScalePtr_ = nullptr;
    __gm__ int32_t* cumsumMMPtr_ = nullptr;
    uint32_t outputN_ = 0U;
    uint32_t expertPerRank_ = 0U;
    uint32_t rankSize_ = 0U;
    uint32_t coreIdx_ = 0U;
    uint32_t coreNum_ = 1U;
};

template <typename InputElement>
AICORE inline void Swiglu<InputElement>::Init(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData)
{
    static_assert(std::is_same_v<InputElement, bfloat16_t>, "MXFP8 SwiGLU requires BF16 GMM output");
    workspaceGM_ = workspaceGM;
    tilingData_ = tilingData;
    outputN_ = tilingData_->megaMoeInfo.N / 2U;
    expertPerRank_ = tilingData_->megaMoeInfo.expertPerRank;
    rankSize_ = tilingData_->runtimeInfo.rankSize;

    gmSwigluAPtr_ = reinterpret_cast<__gm__ float8_e4m3_t*>(workspaceGM_ + tilingData_->swigluTiling.gmSwigluAOffset);
    gmSwigluScalePtr_ =
        reinterpret_cast<__gm__ float8_e8m0_t*>(workspaceGM_ + tilingData_->swigluTiling.gmSwigluScaleOffset);
    cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t*>(workspaceGM_ + tilingData_->frontReorderTiling.cumsumMMOffset);
}

template <typename InputElement>
AICORE inline void Swiglu<InputElement>::ProcessFixed(uint32_t groupLocalId, uint32_t groupSize)
{
    coreIdx_ = groupLocalId;
    coreNum_ = groupSize;
    if (tilingData_->gmmSchedulerTiling.gmm1ScheduleMode == kMegaMoeGmm1ScheduleWave0MailboxSuffix) {
        ProcessHybrid();
    } else {
        ProcessFixedWave();
    }
}

template <typename InputElement>
AICORE inline void Swiglu<InputElement>::ProcessFixedWave()
{
    WaitEpochAcquire(
        FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncFrontMetadataReadySlot),
        kMegaMoeFixedFrontMetadataReadyMarker);

    const __gm__ MegaMoeFixedGroupTiling& fixed = tilingData_->fixedGroupTiling;
    Gmm1SwigluCvPipe cvPipe;
    uint32_t groupBase = 0U;
    MegaMoeCoreTileBalancer tileBalancer;
    for (uint32_t waveIdx = 0U; waveIdx < fixed.totalWaveCount; ++waveIdx) {
        const uint32_t participantCount =
            waveIdx < fixed.fullAicGmm1WaveCount ? fixed.physicalAicNum : fixed.gmm1GroupSize;
        coreNum_ = participantCount;
        if (coreIdx_ >= participantCount) {
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
                const GmmCommonTileInfo tileInfo = BuildPairTileInfo(currentM, loopIdx);
                ComputeDirectPayload(
                    cvPipe, groupBase + tileInfo.blockRowStart, tileInfo.actualN, tileInfo.blockColStart,
                    tileInfo.actualM);
                PublishGmm2InputExpertTileReady(groupIdx);
            }
            groupBase += currentM;
            CommitCoreTileAssignment(tileBalancer, startCoreIdx, coreLoops);
        }

        dsb(DSB_DDR);

        // Tile consumption has already overlapped the paired AIC producer.
        // This final per-core marker is only the wave-boundary control path;
        // it also gives zero-tile participants a deterministic rendezvous.
        const uint32_t readyEpoch = waveIdx + 1U;
        (void)WaitEpochAcquire(
            FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncGmm1ArrivalBase + coreIdx_),
            static_cast<int32_t>(readyEpoch));
    }
    if (fixed.fullAicGmm1WaveCount > kMegaMoeFullAicGmm1WaveCount && coreIdx_ >= fixed.gmm1GroupSize) {
        // Only the configured multi-wave split (M2048 today) hands Group2 to
        // GMM2 locally. Default one-wave fixed schedules arm in the outer path.
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
        PublishCombineConsumerArmed(workspaceGM_, tilingData_, coreIdx_);
    }
}

template <typename InputElement>
AICORE inline void Swiglu<InputElement>::ProcessHybrid()
{
    WaitEpochAcquire(
        FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncFrontMetadataReadySlot),
        kMegaMoeFixedFrontMetadataReadyMarker);

    Gmm1SwigluCvPipe cvPipe;
    ConsumeDirectWave0(cvPipe);
    const bool group2DirectGmm2 = coreIdx_ >= tilingData_->fixedGroupTiling.gmm1GroupSize;
    if (group2DirectGmm2) {
        dsb(DSB_DDR);
        return;
    }

    GmmCvTaskInferenceCache inferenceCache;
    while (true) {
        Gmm1SwigluControlConsumerWait(cvPipe);
        const uint32_t control = ReadGmmCvTaskControl(cvPipe.cons.controlIndex, kGmm1SwigluControlFifoDepth);
        Gmm1SwigluControlConsumerRelease(cvPipe);
        if (IsGmmStageEndControl(control)) {
            break;
        }
        const MegaMoeGmmTask task = InferGmmCvTask(control, cumsumMMPtr_, rankSize_, expertPerRank_, inferenceCache);
        const GmmCommonTileInfo tileInfo =
            GmmCommonBuildTileInfoFromCoord(task.currentM, outputN_, task.blockM, task.blockN);
        ComputeDirectPayload(
            cvPipe, task.expertBase + tileInfo.blockRowStart, tileInfo.actualN, tileInfo.blockColStart,
            tileInfo.actualM);
        PublishGmm2InputExpertTileReady(task.expert);
    }
    dsb(DSB_DDR);
}

#endif // DISPATCH_MEGA_COMBINE_SWIGLU_H
