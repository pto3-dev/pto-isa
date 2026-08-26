/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file dispatch_mega_combine_tiling.h
 * \brief
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "gmm_task_queue.h"
#include "utils/const_args.hpp"

constexpr uint32_t kMegaMoeGmmTileM = 256U;
constexpr uint32_t kMegaMoeGmmTileN = 256U;
constexpr uint32_t kMegaMoeFrontMaskRouteItemsPerBatch = 2048U;
constexpr uint32_t kMegaMoeFrontMetadataSortOutLoopElems = 2040U;
constexpr uint32_t kMegaMoeReadyCountSlotBytes = 64U;

struct MegaMoeInfo {
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t expertPerRank;
    uint32_t topK;
};

struct MegaMoeRuntimeInfo {
    uint64_t remoteWindowContext = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
};

struct MegaMoeFrontReorderTiling {
    uint32_t routeElems = 0;
    uint32_t expertNum = 0;
    uint32_t quantDataStorageBytes = 0;
    uint32_t quantScaleCols = 0;
    uint32_t packedRowStride = 0;
    uint32_t maskBytes = 0;
    uint32_t maskSlotBytes = 0;
    uint32_t maskLaneCapacity = 0;
    uint64_t routeMaskOffset = 0;
    // Source ranks publish one ExpertPerRank prefix slice to each destination.
    uint64_t preSumBeforeRankPeerOffset = 0;
    uint64_t combineOutputOffset = 0;
    uint64_t cumsumMMOffset = 0;
    uint64_t expandedRowIdxOffset = 0;
    // Only (expertId, routeSlot) records are sorted. Token payload stays in
    // the source-token storage used by Dispatch.
    uint32_t sortRunElems = 0;
    uint32_t sortRunCount = 0;
    uint64_t sortedRouteSlotOffset = 0;
    uint64_t sortWorkspace0Offset = 0;
    uint64_t sortWorkspace1Offset = 0;
};

struct MegaMoeDispatchTiling {
    uint64_t gmAOffset = 0;
    uint64_t gmAScaleOffset = 0;
    uint64_t routeMetaOffset = 0;
    uint64_t readyCountOffset = 0;
    uint32_t readyCountMaxTilesPerExpert = 0;
    uint32_t routeItemsPerBatch = 0;
    uint32_t bufferCount = 0;
    uint32_t copyBufferBytes = 0;
    uint32_t routeIndexUbOffset = 0;
    uint32_t routeCountUbOffset = 0;
    uint32_t maskBufferUbOffset = 0;
};

struct MegaMoeSwigluTiling {
    uint64_t gmSwigluAOffset = 0;
    uint64_t gmSwigluScaleOffset = 0;
};

struct MegaMoeGmmQueueTiling {
    uint64_t controlOffset = 0;
    uint64_t taskOffset = 0;
    uint64_t dependencyOffset = 0;
    uint64_t completionOffset = 0;
};

constexpr uint32_t kMegaMoeGmm1ScheduleFixedWave = 0U;
constexpr uint32_t kMegaMoeGmm1ScheduleWave0MailboxSuffix = 1U;

constexpr uint32_t MegaMoeResolveAutoGmm1ScheduleMode(uint32_t m)
{
    return m <= 512U ? kMegaMoeGmm1ScheduleWave0MailboxSuffix : kMegaMoeGmm1ScheduleFixedWave;
}

struct MegaMoeGmmMailboxTiling {
    uint64_t p2cOffset = 0;
    uint64_t c2pOffset = 0;
    uint32_t gmm2TicketBase = kGmmMailboxFirstTaskTicket;
};

struct MegaMoeGmmSchedulerTiling {
    MegaMoeGmmQueueTiling gmm1;
    MegaMoeGmmQueueTiling gmm2;
    MegaMoeGmmMailboxTiling mailbox;
    uint32_t gmm1ScheduleMode = kMegaMoeGmm1ScheduleFixedWave;
};

struct MegaMoeUnpermuteTiling {
    uint32_t unpermuteTileCols = 1024;
    uint32_t unpermuteTokenBatch = 256;
    uint32_t unpermuteInputBufferCount = 2;
    uint32_t rankStreamingInitialWorkerCount = 0;
};

constexpr uint32_t kMegaMoeUnpermuteMinInputBufferCount = 2U;
constexpr uint32_t kMegaMoeUnpermuteMaxInputBufferCount = 6U;
constexpr uint32_t kMegaMoeRankStreamingMaxTokensPerWorker = 256U;
constexpr uint32_t kMegaMoeFixedMaxExperts = 32U;
constexpr uint32_t kMegaMoeFullAicGmm1WaveCount = 1U;

constexpr uint32_t kMegaMoeFixedSyncSlotBytes = kMegaMoeReadyCountSlotBytes;
constexpr uint32_t kMegaMoeFixedSyncHeadCanarySlot = 0U;
constexpr uint32_t kMegaMoeFixedSyncFrontMetadataReadySlot = kMegaMoeFixedSyncHeadCanarySlot + 1U;
constexpr uint32_t kMegaMoeFixedSyncGmm1ArrivalBase = kMegaMoeFixedSyncFrontMetadataReadySlot + 1U;
constexpr uint32_t kMegaMoeFixedSyncGmm1ArrivalSlotCount = kMegaMoeFixedPhysicalAicNum;
constexpr uint32_t kMegaMoeFixedSyncCombineConsumerArmedBase =
    kMegaMoeFixedSyncGmm1ArrivalBase + kMegaMoeFixedSyncGmm1ArrivalSlotCount;
constexpr uint32_t kMegaMoeFixedSyncCombineConsumerArmedSlotCount = kMegaMoeFixedPhysicalAicNum;
constexpr uint32_t kMegaMoeFixedSyncDeferredMetadataArrivalBase =
    kMegaMoeFixedSyncCombineConsumerArmedBase + kMegaMoeFixedSyncCombineConsumerArmedSlotCount;
constexpr uint32_t kMegaMoeFixedSyncDeferredMetadataReadyBase =
    kMegaMoeFixedSyncDeferredMetadataArrivalBase + kMegaMoeFixedGmm2GroupSize;
constexpr uint32_t kMegaMoeFixedSyncDeferredExpandedReadySlot =
    kMegaMoeFixedSyncDeferredMetadataReadyBase + kMegaMoeFixedGmm2GroupSize;
constexpr uint32_t kMegaMoeFixedSyncDispatchDoneSlot = kMegaMoeFixedSyncDeferredExpandedReadySlot + 1U;
constexpr uint32_t kMegaMoeFixedSyncGmm2EntryReadySlot = kMegaMoeFixedSyncDispatchDoneSlot + 1U;
constexpr uint32_t kMegaMoeFixedSyncTailCanarySlot = kMegaMoeFixedSyncGmm2EntryReadySlot + 1U;
constexpr uint32_t kMegaMoeFixedSyncSlotCount = kMegaMoeFixedSyncTailCanarySlot + 1U;
constexpr uint32_t kMegaMoeFixedSyncBytes =
    (kMegaMoeFixedSyncSlotCount * kMegaMoeFixedSyncSlotBytes + 511U) / 512U * 512U;
constexpr uint32_t kMegaMoeFixedCompletionSlotBytes = 64U;
constexpr uint32_t kMegaMoeFixedCompletionMarker = 1U;

constexpr uint64_t MegaMoeFixedCompletionBytes(uint32_t physicalAicNum)
{
    const uint64_t rawBytes = static_cast<uint64_t>(physicalAicNum) * kMegaMoeFixedCompletionSlotBytes;
    return (rawBytes + 511U) / 512U * 512U;
}

constexpr uint32_t kMegaMoeFixedHeadCanary = 0x13579BDFU;
constexpr uint32_t kMegaMoeFixedTailCanary = 0x2468ACE0U;
constexpr int32_t kMegaMoeFixedFrontMetadataReadyMarker = 1;
constexpr int32_t kMegaMoeFixedDeferredExpandedReadyMarker = 1;
constexpr int32_t kMegaMoeFixedGmm1DoneMarker = 1;
constexpr int32_t kMegaMoeFixedDispatchDoneMarker = 1;
constexpr int32_t kMegaMoeFixedCombineConsumerArmedMarker = 1;
constexpr int32_t kMegaMoeFixedGmm2EntryReadyMarker = 1;
constexpr uint32_t kMegaMoeFixedGmm1GroupDoneEpochOffset = 1U;

static_assert(kMegaMoeFixedSyncSlotCount == 110U);
static_assert(kMegaMoeFixedSyncBytes == 7168U);
static_assert(kMegaMoeFixedSyncSlotCount * kMegaMoeFixedSyncSlotBytes <= kMegaMoeFixedSyncBytes);

struct MegaMoeFixedGroupTiling {
    uint64_t syncOffset = 0;
    // A scalar completion transaction keeps the mixed-core kernel ordered on A5.
    uint64_t completionOffset = 0;
    uint32_t physicalAicNum = 0;
    uint32_t dispatchGroupSize = 0;
    uint32_t gmm1GroupSize = 0;
    uint32_t gmm2GroupSize = 0;
    uint32_t fullAicExpertsPerWave = 1U;
    uint32_t expertsPerWave = 1U;
    uint32_t totalWaveCount = 1U;
    uint32_t fullAicGmm1WaveCount = kMegaMoeFullAicGmm1WaveCount;
};

static_assert(sizeof(MegaMoeSwigluTiling) == 16);
static_assert(sizeof(MegaMoeDispatchTiling) == 64);
static_assert(sizeof(MegaMoeGmmQueueTiling) == 32);
static_assert(sizeof(MegaMoeGmmMailboxTiling) == 24);
static_assert(sizeof(MegaMoeGmmSchedulerTiling) == 96);
static_assert(sizeof(MegaMoeUnpermuteTiling) == 16);
static_assert(sizeof(MegaMoeFixedGroupTiling) == 48);

struct MegaMoeTilingData {
    MegaMoeInfo megaMoeInfo;
    MegaMoeRuntimeInfo runtimeInfo;
    MegaMoeFrontReorderTiling frontReorderTiling;
    MegaMoeDispatchTiling dispatchTiling;
    MegaMoeSwigluTiling swigluTiling;
    MegaMoeGmmSchedulerTiling gmmSchedulerTiling;
    MegaMoeUnpermuteTiling unpermuteTiling;
    MegaMoeFixedGroupTiling fixedGroupTiling;
};
