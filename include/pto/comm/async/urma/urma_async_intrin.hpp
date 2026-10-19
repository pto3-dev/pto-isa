/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_INTRIN_HPP
#define PTO_COMM_ASYNC_URMA_INTRIN_HPP

#ifdef PTO_URMA_SUPPORTED

#include "pto/common/debug.h"
#include "pto/comm/async_common/async_types.hpp"
#include "pto/comm/async/urma/urma_types.hpp"

namespace pto {
namespace comm {
namespace urma {

AICORE inline void DcciCachelines(__gm__ uint8_t* addr, uint64_t length)
{
    if (length == 0U) {
        return;
    }
    __gm__ uint8_t* start =
        reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uint64_t>(addr) / kCacheLineSize * kCacheLineSize);
    __gm__ uint8_t* end = reinterpret_cast<__gm__ uint8_t*>(
        (reinterpret_cast<uint64_t>(addr) + length - 1U) / kCacheLineSize * kCacheLineSize);
    for (uint64_t offset = 0; offset <= static_cast<uint64_t>(end - start); offset += kCacheLineSize) {
        __asm__ __volatile__("");
        dcci(reinterpret_cast<__gm__ void*>(start + offset), cache_line_t::SINGLE_CACHE_LINE);
        __asm__ __volatile__("");
    }
}

namespace detail {

struct UrmaPostResult {
    uint64_t handle;
    uint32_t targetCqe;
};

AICORE inline uint64_t EncodeHandle(uint32_t peer, uint32_t targetBb)
{
    return (static_cast<uint64_t>(peer) << 32U) | targetBb;
}

AICORE inline void DecodeHandle(uint64_t handle, uint32_t& peer, uint32_t& targetBb)
{
    peer = static_cast<uint32_t>(handle >> 32U);
    targetBb = static_cast<uint32_t>(handle);
}

AICORE inline bool SequenceReached(uint32_t current, uint32_t target)
{
    return static_cast<int32_t>(current - target) >= 0;
}

AICORE inline void UrmaPostSendUpdateInfo(uint32_t head, __gm__ UrmaWQCtx* wq)
{
    st_dev(head, reinterpret_cast<__gm__ uint32_t*>(wq->dbAddr), 0);
    st_dev(head, reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
}

AICORE inline __gm__ UrmaWQCtx* GetWqContext(const AsyncSession& session, uint32_t peer)
{
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    return reinterpret_cast<__gm__ UrmaWQCtx*>(info->sqPtr + (peer * info->qpNum + session.qpIdx) * sizeof(UrmaWQCtx));
}

AICORE inline __gm__ UrmaCqCtx* GetCqContext(const AsyncSession& session, uint32_t peer)
{
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    return reinterpret_cast<__gm__ UrmaCqCtx*>(info->scqPtr + (peer * info->qpNum + session.qpIdx) * sizeof(UrmaCqCtx));
}

AICORE inline bool ValidateUrmaSession(const AsyncSession& session, uint32_t peer)
{
    if (!session.valid || session.engine != DmaEngine::URMA || session.contextGm == nullptr) {
        return false;
    }
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    if (peer >= info->rankCount) {
        return false;
    }
    if (session.qpIdx >= info->qpNum) {
        return false;
    }
    __gm__ UrmaWQCtx* wq = GetWqContext(session, peer);
    __gm__ UrmaCqCtx* cq = GetCqContext(session, peer);
    return wq->depth != 0U && cq->depth != 0U;
}

AICORE inline __gm__ UrmaNotifyResourceRegion* GetUrmaNotifyResourceRegion(const AsyncSession& session, uint32_t peer)
{
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    __gm__ UrmaNotifyResourceRegion* regions = reinterpret_cast<__gm__ UrmaNotifyResourceRegion*>(info->notifyPoolPtr);
    return regions + peer * info->qpNum + session.qpIdx;
}

AICORE inline void PublishCqAndWqTails(
    uint32_t targetBb, uint32_t targetCqe, __gm__ UrmaCqCtx* cq, __gm__ UrmaWQCtx* wq)
{
    st_dev(targetCqe, reinterpret_cast<__gm__ uint32_t*>(cq->tailAddr), 0);
    st_dev(targetCqe & 0xFFFFFFU, reinterpret_cast<__gm__ uint32_t*>(cq->dbAddr), 0);
    st_dev(targetBb, reinterpret_cast<__gm__ uint32_t*>(wq->tailAddr), 0);
}

AICORE inline __gm__ UrmaJfcCqeCtx* GetUrmaCqe(__gm__ UrmaCqCtx* cq, uint32_t sequence, uint32_t cqeSize)
{
    return reinterpret_cast<__gm__ UrmaJfcCqeCtx*>(cq->bufAddr + cqeSize * (sequence & (cq->depth - 1U)));
}

AICORE inline bool IsUrmaCqePrefixReady(__gm__ UrmaCqCtx* cq, uint32_t begin, uint32_t end, uint32_t cqeSize)
{
    for (uint32_t sequence = begin; sequence != end; ++sequence) {
        __gm__ UrmaJfcCqeCtx* cqe = GetUrmaCqe(cq, sequence, cqeSize);
        const bool validOwner = (sequence / cq->depth) & 1U;
        DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(cqe), sizeof(UrmaJfcCqeCtx));
        if ((validOwner ^ cqe->owner) == 0U) {
            return false;
        }
    }
    return true;
}

AICORE inline bool WaitForUrmaCqe(__gm__ UrmaCqCtx* cq, uint32_t sequence, uint32_t cqeSize, __gm__ UrmaJfcCqeCtx*& cqe)
{
    cqe = GetUrmaCqe(cq, sequence, cqeSize);
    const bool validOwner = (sequence / cq->depth) & 1U;
    uint32_t polls = 0U;
    DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(cqe), sizeof(UrmaJfcCqeCtx));
    while ((validOwner ^ cqe->owner) == 0U) {
        if (++polls >= kUrmaMaxPollTimes) {
            return false;
        }
        DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(cqe), sizeof(UrmaJfcCqeCtx));
    }
    return true;
}

AICORE inline bool RetireUrmaCqePrefix(__gm__ UrmaCqCtx* cq, uint32_t beginCqe, uint32_t targetCqe, uint32_t cqeSize)
{
    for (uint32_t sequence = beginCqe; sequence != targetCqe; ++sequence) {
        __gm__ UrmaJfcCqeCtx* cqe = nullptr;
        if (!WaitForUrmaCqe(cq, sequence, cqeSize, cqe)) {
            return false;
        }
        constexpr uint32_t kStatusShift = 8U;
        const uint32_t cqeError =
            (static_cast<uint32_t>(cqe->status) << kStatusShift) | static_cast<uint32_t>(cqe->substatus);
        if (cqeError != 0U) {
            return false;
        }
    }
    return true;
}

AICORE inline bool UrmaPollCq(
    const AsyncSession& session, uint32_t peer, uint32_t targetBb, uint32_t targetCqe, bool blocking, bool& completed)
{
    completed = false;
    if (!ValidateUrmaSession(session, peer)) {
        return false;
    }
    __gm__ UrmaCqCtx* cq = GetCqContext(session, peer);
    __gm__ UrmaWQCtx* wq = GetWqContext(session, peer);
    const uint32_t completedCqe = ld_dev(reinterpret_cast<__gm__ uint32_t*>(cq->tailAddr), 0);
    if (SequenceReached(completedCqe, targetCqe)) {
        const uint32_t completedBb = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->tailAddr), 0);
        completed = SequenceReached(completedBb, targetBb);
        return completed;
    }
    const uint32_t cqeSize = 1U << cq->cqeShiftSize;
    if (!blocking && !IsUrmaCqePrefixReady(cq, completedCqe, targetCqe, cqeSize)) {
        return true;
    }
    if (!RetireUrmaCqePrefix(cq, completedCqe, targetCqe, cqeSize)) {
        return false;
    }
    PublishCqAndWqTails(targetBb, targetCqe, cq, wq);
    completed = true;
    return true;
}

AICORE inline bool EnsureUrmaCapacity(
    const AsyncSession& session, uint32_t peer, uint32_t requiredBb, uint32_t requiredWqeCount)
{
    __gm__ UrmaWQCtx* wq = GetWqContext(session, peer);
    __gm__ UrmaCqCtx* cq = GetCqContext(session, peer);
    if (requiredBb > wq->depth || requiredWqeCount > cq->depth) {
        return false;
    }

    const uint32_t completedBb = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->tailAddr), 0);
    const uint32_t completedCqe = ld_dev(reinterpret_cast<__gm__ uint32_t*>(cq->tailAddr), 0);
    const uint32_t submittedBb = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
    if (submittedBb - completedBb + requiredBb > wq->depth ||
        wq->submittedWqeCount - completedCqe + requiredWqeCount > cq->depth) {
        bool completed = false;
        if (!UrmaPollCq(session, peer, submittedBb, wq->submittedWqeCount, true, completed) || !completed) {
            return false;
        }
    }
    return true;
}

AICORE inline void FillSqeCtx(
    __gm__ UrmaSqeCtx* sqe, __gm__ UrmaMemInfo* remoteMem, __gm__ uint8_t* remoteAddr, UrmaOpcode opcode,
    uint32_t bbSequence, uint32_t depth)
{
    sqe->sqeBbIdx = static_cast<uint16_t>(bbSequence % depth);
    sqe->opcode = static_cast<uint32_t>(opcode);
    sqe->flag = 0x20U;
    sqe->rsv0 = 0U;
    sqe->nf = 0U;
    sqe->tokenEn = remoteMem->tokenValueValid ? 1U : 0U;
    sqe->rmtJettyType = remoteMem->rmtJettyType;
    sqe->owner = (bbSequence & depth) == 0U ? 1U : 0U;
    sqe->targetHint = remoteMem->targetHint;
    sqe->rsv1 = 0U;
    sqe->inlineMsgLen = 0U;
    sqe->tpId = remoteMem->tpn;
    sqe->sgeNum = 1U;
    sqe->rmtJettyOrSegId = remoteMem->tid;
    sqe->rsv2 = 0U;
    sqe->rmtTokenValue = remoteMem->rmtTokenValue;
    sqe->udfType = 0U;
    sqe->reduceDataType = 0U;
    sqe->reduceOpcode = 0U;
    sqe->rsv3 = 0U;

    const uint64_t remoteAddress = reinterpret_cast<uint64_t>(remoteAddr);
    __gm__ uint8_t* bytes = reinterpret_cast<__gm__ uint8_t*>(sqe);
    __gm__ uint64_t* eid = reinterpret_cast<__gm__ uint64_t*>(remoteMem->eidAddr);
    *reinterpret_cast<__gm__ uint64_t*>(bytes + kUrmaSqeRmtEidLOffset) = eid[0];
    *reinterpret_cast<__gm__ uint64_t*>(bytes + kUrmaSqeRmtEidHOffset) = eid[1];
    *reinterpret_cast<__gm__ uint32_t*>(bytes + kUrmaSqeRmtAddrLOffset) = static_cast<uint32_t>(remoteAddress);
    *reinterpret_cast<__gm__ uint32_t*>(bytes + kUrmaSqeRmtAddrHOffset) = static_cast<uint32_t>(remoteAddress >> 32U);
}

AICORE inline void FillTransferWqe(
    __gm__ uint8_t* wqe, __gm__ UrmaMemInfo* remoteMem, __gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr,
    uint64_t messageLen, UrmaOpcode opcode, uint32_t bbSequence, uint32_t depth, uint32_t localTokenId)
{
    FillSqeCtx(reinterpret_cast<__gm__ UrmaSqeCtx*>(wqe), remoteMem, remoteAddr, opcode, bbSequence, depth);
    __gm__ UrmaSgeCtx* sge = reinterpret_cast<__gm__ UrmaSgeCtx*>(wqe + kUrmaSqeSizeBytes);
    sge->len = static_cast<uint32_t>(messageLen);
    sge->tokenId = localTokenId;
    sge->va = reinterpret_cast<uint64_t>(localAddr);
}

AICORE inline void FillFaaWqe(
    __gm__ uint8_t* firstBb, __gm__ uint8_t* secondBb, __gm__ UrmaMemInfo* remoteMem, __gm__ int32_t* remoteSignal,
    __gm__ int32_t* resultSink, int32_t addValue, uint32_t bbSequence, uint32_t depth, uint32_t localTokenId)
{
    FillSqeCtx(
        reinterpret_cast<__gm__ UrmaSqeCtx*>(firstBb), remoteMem, reinterpret_cast<__gm__ uint8_t*>(remoteSignal),
        UrmaOpcode::FAA, bbSequence, depth);
    reinterpret_cast<__gm__ UrmaSqeCtx*>(firstBb)->flag = 0x22U;
    __gm__ UrmaSgeCtx* sge = reinterpret_cast<__gm__ UrmaSgeCtx*>(firstBb + kUrmaSqeSizeBytes);
    sge->len = sizeof(int32_t);
    sge->tokenId = localTokenId;
    sge->va = reinterpret_cast<uint64_t>(resultSink);
    *reinterpret_cast<__gm__ int32_t*>(secondBb) = addValue;
}

AICORE inline UrmaPostResult UrmaPostSend(
    __gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr, uint64_t messageLen, UrmaOpcode opcode,
    const AsyncSession& session, uint32_t peer)
{
    PTO_ASSERT(ValidateUrmaSession(session, peer), "UrmaPostSend: invalid URMA session, peer or QP.");
    PTO_ASSERT(
        opcode == UrmaOpcode::WRITE || opcode == UrmaOpcode::READ, "UrmaPostSend: opcode must be WRITE or READ.");
    PTO_ASSERT(
        messageLen > 0U && messageLen <= kUrmaMaxWqeTransferBytes, "UrmaPostSend: messageLen must be in (0, 256MB].");
    const bool hasCapacity = EnsureUrmaCapacity(session, peer, 1U, 1U);
    PTO_ASSERT(hasCapacity, "UrmaPostSend: failed to acquire URMA queue capacity.");

    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    __gm__ UrmaWQCtx* wq = GetWqContext(session, peer);
    __gm__ UrmaMemInfo* remoteMem = reinterpret_cast<__gm__ UrmaMemInfo*>(info->memPtr + peer * sizeof(UrmaMemInfo));
    const uint32_t bbSize = 1U << wq->wqeShiftSize;
    const uint32_t startBb = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
    __gm__ uint8_t* wqe = reinterpret_cast<__gm__ uint8_t*>(wq->bufAddr + bbSize * (startBb % wq->depth));
    FillTransferWqe(wqe, remoteMem, remoteAddr, localAddr, messageLen, opcode, startBb, wq->depth, info->localTokenId);

    const uint32_t targetBb = startBb + 1U;
    const uint32_t targetCqe = wq->submittedWqeCount + 1U;
    wq->submittedWqeCount = targetCqe;
    pipe_barrier(PIPE_ALL);
    DcciCachelines(wqe, kUrmaSqeSizeBytes + kUrmaSgeSizeBytes);
    DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(&wq->submittedWqeCount), sizeof(wq->submittedWqeCount));
    dsb(DSB_DDR);
    UrmaPostSendUpdateInfo(targetBb, wq);
    return {EncodeHandle(peer, targetBb), targetCqe};
}

struct UrmaNotifyWqeBuildState {
    __gm__ UrmaInfo* info;
    __gm__ UrmaWQCtx* wq;
    __gm__ UrmaMemInfo* remoteMem;
    __gm__ uint8_t* payloadWqe;
    __gm__ uint8_t* signalFirstBb;
    __gm__ uint8_t* signalSecondBb;
    uint32_t startBb;
    uint32_t signalBb;
};

AICORE inline bool PrepareNotifyResources(
    const AsyncSession& session, uint32_t peer, __gm__ UrmaNotifyResourceRegion* region, bool isAdd,
    uint32_t requiredBb, uint32_t& setSlotIndex)
{
    PTO_ASSERT(
        isAdd || region->nextSetSlot < kUrmaNotifySetSlotCount,
        "PrepareNotifyResources: next SET slot is out of range.");
    setSlotIndex = isAdd ? 0U : region->nextSetSlot;
    const bool reusingSetSlotZero = !isAdd && region->setRingStarted != 0U && setSlotIndex == 0U;
    __gm__ UrmaCqCtx* cq = GetCqContext(session, peer);
    __gm__ UrmaWQCtx* wq = GetWqContext(session, peer);
    const uint32_t completedCqe = ld_dev(reinterpret_cast<__gm__ uint32_t*>(cq->tailAddr), 0);
    if (reusingSetSlotZero && wq->submittedWqeCount != completedCqe) {
        bool completed = false;
        const uint32_t submittedBb = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
        if (!UrmaPollCq(session, peer, submittedBb, wq->submittedWqeCount, true, completed) || !completed) {
            return false;
        }
    }
    return EnsureUrmaCapacity(session, peer, requiredBb, 2U);
}

AICORE inline UrmaNotifyWqeBuildState BuildNotifyWqeState(const AsyncSession& session, uint32_t peer)
{
    UrmaNotifyWqeBuildState buildState{};
    buildState.info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    buildState.wq = GetWqContext(session, peer);
    buildState.remoteMem = reinterpret_cast<__gm__ UrmaMemInfo*>(buildState.info->memPtr + peer * sizeof(UrmaMemInfo));
    const uint32_t bbSize = 1U << buildState.wq->wqeShiftSize;
    buildState.startBb = ld_dev(reinterpret_cast<__gm__ uint32_t*>(buildState.wq->headAddr), 0);
    buildState.signalBb = buildState.startBb + 1U;
    buildState.payloadWqe = reinterpret_cast<__gm__ uint8_t*>(
        buildState.wq->bufAddr + bbSize * (buildState.startBb % buildState.wq->depth));
    buildState.signalFirstBb = reinterpret_cast<__gm__ uint8_t*>(
        buildState.wq->bufAddr + bbSize * (buildState.signalBb % buildState.wq->depth));
    buildState.signalSecondBb = reinterpret_cast<__gm__ uint8_t*>(
        buildState.wq->bufAddr + bbSize * ((buildState.signalBb + 1U) % buildState.wq->depth));
    return buildState;
}

AICORE inline void FillNotifySignalWqe(
    const UrmaNotifyWqeBuildState& buildState, __gm__ UrmaNotifyResourceRegion* region, __gm__ int32_t* remoteSignal,
    int32_t signalValue, bool isAdd, uint32_t setSlotIndex)
{
    if (isAdd) {
        FillFaaWqe(
            buildState.signalFirstBb, buildState.signalSecondBb, buildState.remoteMem, remoteSignal, &region->faaResult,
            signalValue, buildState.signalBb, buildState.wq->depth, buildState.info->notifyPoolTokenId);
        return;
    }
    region->setValues[setSlotIndex] = signalValue;
    FillTransferWqe(
        buildState.signalFirstBb, buildState.remoteMem, reinterpret_cast<__gm__ uint8_t*>(remoteSignal),
        reinterpret_cast<__gm__ uint8_t*>(&region->setValues[setSlotIndex]), sizeof(int32_t), UrmaOpcode::WRITE,
        buildState.signalBb, buildState.wq->depth, buildState.info->notifyPoolTokenId);
}

AICORE inline UrmaPostResult CommitNotifyWqes(
    const UrmaNotifyWqeBuildState& buildState, __gm__ UrmaNotifyResourceRegion* region, bool isAdd, uint32_t requiredBb,
    uint32_t peer)
{
    const uint32_t targetBb = buildState.startBb + requiredBb;
    const uint32_t targetCqe = buildState.wq->submittedWqeCount + 2U;
    buildState.wq->submittedWqeCount = targetCqe;
    if (!isAdd) {
        region->nextSetSlot = (region->nextSetSlot + 1U) % kUrmaNotifySetSlotCount;
        region->setRingStarted = 1U;
    }
    pipe_barrier(PIPE_ALL);
    DcciCachelines(buildState.payloadWqe, kUrmaSqeSizeBytes + kUrmaSgeSizeBytes);
    DcciCachelines(buildState.signalFirstBb, kUrmaSqeSizeBytes + kUrmaSgeSizeBytes);
    if (isAdd) {
        DcciCachelines(buildState.signalSecondBb, sizeof(int32_t));
        DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(&region->faaResult), sizeof(region->faaResult));
    } else {
        DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(region), kUrmaNotifyResourceCachelineBytes);
    }
    DcciCachelines(
        reinterpret_cast<__gm__ uint8_t*>(&buildState.wq->submittedWqeCount), sizeof(buildState.wq->submittedWqeCount));
    dsb(DSB_DDR);
    UrmaPostSendUpdateInfo(targetBb, buildState.wq);
    return {EncodeHandle(peer, targetBb), targetCqe};
}

AICORE inline UrmaPostResult UrmaPostNotify(
    __gm__ uint8_t* remotePayload, __gm__ uint8_t* localPayload, uint64_t messageLen, __gm__ int32_t* remoteSignal,
    int32_t signalValue, NotifyOp notifyOp, const AsyncSession& session, uint32_t peer)
{
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    PTO_ASSERT(info->notifyPoolPtr != 0U, "UrmaPostNotify: notify resource pool is unavailable.");

    const bool isAtomicAdd = notifyOp == NotifyOp::AtomicAdd;
    const uint32_t requiredWqeBb = isAtomicAdd ? 3U : 2U;
    __gm__ UrmaNotifyResourceRegion* region = GetUrmaNotifyResourceRegion(session, peer);

    uint32_t setSlotIndex = 0U;
    const bool resourcesReady = PrepareNotifyResources(session, peer, region, isAtomicAdd, requiredWqeBb, setSlotIndex);
    PTO_ASSERT(resourcesReady, "UrmaPostNotify: failed to acquire notify resources or URMA queue capacity.");

    UrmaNotifyWqeBuildState buildState = BuildNotifyWqeState(session, peer);
    FillTransferWqe(
        buildState.payloadWqe, buildState.remoteMem, remotePayload, localPayload, messageLen, UrmaOpcode::WRITE,
        buildState.startBb, buildState.wq->depth, buildState.info->localTokenId);
    FillNotifySignalWqe(buildState, region, remoteSignal, signalValue, isAtomicAdd, setSlotIndex);

    return CommitNotifyWqes(buildState, region, isAtomicAdd, requiredWqeBb, peer);
}

AICORE inline bool UrmaWaitEvent(uint64_t handle, uint32_t targetCqe, const AsyncSession& session)
{
    uint32_t peer = 0U;
    uint32_t targetBb = 0U;
    DecodeHandle(handle, peer, targetBb);
    if (!ValidateUrmaSession(session, peer)) {
        return false;
    }
    bool completed = false;
    return UrmaPollCq(session, peer, targetBb, targetCqe, true, completed) && completed;
}

AICORE inline bool UrmaTestEvent(uint64_t handle, uint32_t targetCqe, const AsyncSession& session)
{
    uint32_t peer = 0U;
    uint32_t targetBb = 0U;
    DecodeHandle(handle, peer, targetBb);
    if (!ValidateUrmaSession(session, peer)) {
        return false;
    }
    bool completed = false;
    return UrmaPollCq(session, peer, targetBb, targetCqe, false, completed) && completed;
}

} // namespace detail

AICORE inline detail::UrmaPostResult __urma_put_async(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint64_t transferSize, const AsyncSession& session, uint32_t peer)
{
    return detail::UrmaPostSend(dst, src, transferSize, UrmaOpcode::WRITE, session, peer);
}

AICORE inline detail::UrmaPostResult __urma_get_async(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint64_t transferSize, const AsyncSession& session, uint32_t peer)
{
    return detail::UrmaPostSend(src, dst, transferSize, UrmaOpcode::READ, session, peer);
}

AICORE inline detail::UrmaPostResult __urma_put_async(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint64_t transferSize, const AsyncSession& session)
{
    return __urma_put_async(dst, src, transferSize, session, session.destRankId);
}

AICORE inline detail::UrmaPostResult __urma_get_async(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint64_t transferSize, const AsyncSession& session)
{
    return __urma_get_async(dst, src, transferSize, session, session.destRankId);
}

AICORE inline detail::UrmaPostResult __urma_put_async_notify(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint64_t transferSize, __gm__ int32_t* signal, int32_t signalValue,
    NotifyOp notifyOp, const AsyncSession& session, uint32_t peer)
{
    return detail::UrmaPostNotify(dst, src, transferSize, signal, signalValue, notifyOp, session, peer);
}

AICORE inline uint64_t UrmaPeerMrBaseAddr(__gm__ uint8_t* urmaWorkspace, uint32_t peerRank)
{
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(urmaWorkspace);
    PTO_ASSERT(peerRank < info->rankCount, "UrmaPeerMrBaseAddr: peerRank out of range");
    __gm__ UrmaMemInfo* memRow = reinterpret_cast<__gm__ UrmaMemInfo*>(info->memPtr) + peerRank;
    return memRow->addr;
}

} // namespace urma
} // namespace comm
} // namespace pto

#endif // PTO_URMA_SUPPORTED
#endif // PTO_COMM_ASYNC_URMA_INTRIN_HPP
