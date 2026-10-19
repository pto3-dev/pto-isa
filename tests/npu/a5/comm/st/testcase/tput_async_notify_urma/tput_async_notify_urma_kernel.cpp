/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstddef>
#include <cstdint>
#include <vector>

#include <pto/pto-inst.hpp>

#include "../common.hpp"
#include "pto/common/pto_tile.hpp"
#include "pto/comm/async_common/async_types.hpp"
#include "tput_async_notify_urma_kernel.h"

namespace {

constexpr uint32_t kTargetPeer = 1U;
constexpr uint32_t kPayloadElems = 4U;
constexpr uint32_t kConsumeElems = 8U;
constexpr uint32_t kNotifyCount = 65U;
constexpr uint32_t kTotalPayloadElems = kPayloadElems * kNotifyCount;
constexpr size_t kCanaryBeforeOffset = 0U;
constexpr size_t kSignalOffset = 4U;
constexpr size_t kCanaryAfterOffset = 8U;
constexpr size_t kStatusOffset = 64U;
constexpr size_t kPrimeSinkOffset = 96U;
constexpr size_t kSourceOffset = 128U;
constexpr size_t kTargetOffset = 1216U;
constexpr size_t kGetSourceOffset = 2304U;
constexpr size_t kGetDestOffset = 2368U;
constexpr size_t kConsumeOffset = 2432U;
constexpr size_t kCommBytes = kConsumeOffset + kConsumeElems * sizeof(int32_t);
constexpr int32_t kCanaryBefore = 0x13572468;
constexpr int32_t kCanaryAfter = 0x24681357;
constexpr int32_t kPoison = -777777;
constexpr int32_t kStatusInitial = 0;
constexpr int32_t kStatusSuccess = 1;
constexpr int32_t kSetValue = 37;
constexpr int32_t kAddInitial = 11;
constexpr int32_t kAddValue = 5;
constexpr int32_t kMixedSetValue = 100;
constexpr int32_t kMixedAddValue = 7;
constexpr int32_t kConsumeAdd = 100;
constexpr uint32_t kSignalPollLimit = 10000000U;

using Global4 = pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 4>, pto::Stride<4, 4, 4, 4, 1>, pto::Layout::ND>;
using ConsumeGlobal8 =
    pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<8, 8, 8, 8, 1>, pto::Layout::ND>;
using ConsumeTile8 = pto::Tile<pto::TileType::Vec, int32_t, 1, kConsumeElems>;

AICORE inline void StoreStatus(__gm__ int32_t* status, int32_t value)
{
    *status = value;
    pipe_barrier(PIPE_ALL);
    pto::comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(status), sizeof(int32_t));
    pipe_barrier(PIPE_ALL);
}

AICORE inline void ConsumePayload(__gm__ int32_t* input, __gm__ int32_t* output)
{
    ConsumeGlobal8 inputGlobal(input);
    ConsumeGlobal8 outputGlobal(output);
    ConsumeTile8 inputTile;
    ConsumeTile8 outputTile;
    TASSIGN(inputTile, 0x1000);
    TASSIGN(outputTile, 0x2000);
    TLOAD(inputTile, inputGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADDS(outputTile, inputTile, kConsumeAdd);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(outputGlobal, outputTile);
    pipe_barrier(PIPE_ALL);
}

AICORE inline void ConsumeNotifyOnReceiver(__gm__ uint8_t* localBase)
{
    __gm__ int32_t* status = reinterpret_cast<__gm__ int32_t*>(localBase + kStatusOffset);
    pto::comm::Signal signal(reinterpret_cast<__gm__ int32_t*>(localBase + kSignalOffset));
    bool signaled = false;
    for (uint32_t poll = 0U; poll < kSignalPollLimit; ++poll) {
        if (pto::comm::TTEST(signal, kSetValue, pto::comm::WaitCmp::GE)) {
            signaled = true;
            break;
        }
    }
    if (!signaled) {
        StoreStatus(status, -8);
        return;
    }
    __gm__ int32_t* target = reinterpret_cast<__gm__ int32_t*>(localBase + kTargetOffset);
    __gm__ int32_t* consumed = reinterpret_cast<__gm__ int32_t*>(localBase + kConsumeOffset);
    pto::comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(target), kConsumeElems * sizeof(int32_t));
    dsb(DSB_DDR);
    ConsumePayload(target, consumed);
    StoreStatus(status, kStatusSuccess);
}

__global__ AICORE void PrimeNotifyPayloadCache(__gm__ uint8_t* localBase, uint32_t rankId, uint32_t mode)
{
#ifdef PTO_URMA_SUPPORTED
    if (rankId != kTargetPeer || static_cast<UrmaNotifyStMode>(mode) != UrmaNotifyStMode::ReceiverConsumeSet) {
        return;
    }
    __gm__ int32_t* target = reinterpret_cast<__gm__ int32_t*>(localBase + kTargetOffset);
    __gm__ int32_t* primeSink = reinterpret_cast<__gm__ int32_t*>(localBase + kPrimeSinkOffset);
    pto::comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(target), kConsumeElems * sizeof(int32_t));
    dsb(DSB_DDR);
    volatile __gm__ int32_t* input = target;
    int32_t sum = 0;
    for (uint32_t i = 0U; i < kConsumeElems; ++i) {
        sum += input[i];
    }
    primeSink[0] = sum;
    pipe_barrier(PIPE_ALL);
    pto::comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(primeSink), sizeof(int32_t));
    dsb(DSB_DDR);
#else
    (void)localBase;
    (void)rankId;
    (void)mode;
#endif
}

AICORE inline pto::comm::AsyncEvent SubmitSingleNotify(
    __gm__ int32_t* sourcePtr, __gm__ int32_t* targetPtr, pto::comm::Signal& signal, int32_t value,
    pto::comm::NotifyOp op, const pto::comm::AsyncSession& session, uint32_t peer)
{
    Global4 source(sourcePtr);
    Global4 target(targetPtr);
    return pto::comm::TPUT_ASYNC_NOTIFY<pto::comm::DmaEngine::URMA>(target, source, signal, value, op, session, peer);
}

AICORE inline int32_t SubmitNotifyBatch(
    __gm__ int32_t* localSource, __gm__ int32_t* remoteTarget, pto::comm::Signal& signal, pto::comm::NotifyOp op,
    const pto::comm::AsyncSession& session, uint32_t peer, pto::comm::AsyncEvent& finalEvent)
{
    const bool isSet = op == pto::comm::NotifyOp::Set;
    for (uint32_t i = 0; i < kNotifyCount; ++i) {
        const int32_t value = isSet ? static_cast<int32_t>(i + 1U) : 1;
        pto::comm::AsyncEvent event = SubmitSingleNotify(
            localSource + i * kPayloadElems, remoteTarget + i * kPayloadElems, signal, value, op, session, peer);
        if (!event.valid()) {
            if (finalEvent.valid()) {
                (void)finalEvent.Wait(session);
            }
            return isSet ? -2 : -7;
        }
        finalEvent = event;
    }
    return 0;
}

AICORE inline int32_t SubmitMixedNotifyFlow(
    __gm__ uint8_t* localBase, uint64_t peerBase, __gm__ int32_t* localSource, __gm__ int32_t* remoteTarget,
    pto::comm::Signal& signal, const pto::comm::AsyncSession& session, uint32_t peer, pto::comm::AsyncEvent& finalEvent)
{
    Global4 putSource(localSource);
    Global4 putTarget(remoteTarget);
    Global4 setSource(localSource + kPayloadElems);
    Global4 setTarget(remoteTarget + kPayloadElems);
    Global4 addSource(localSource + 2U * kPayloadElems);
    Global4 addTarget(remoteTarget + 2U * kPayloadElems);
    Global4 getSource(reinterpret_cast<__gm__ int32_t*>(peerBase + kGetSourceOffset));
    Global4 getTarget(reinterpret_cast<__gm__ int32_t*>(localBase + kGetDestOffset));
    const pto::comm::AsyncEvent putEvent =
        pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(putTarget, putSource, session, peer);
    const pto::comm::AsyncEvent setEvent = pto::comm::TPUT_ASYNC_NOTIFY<pto::comm::DmaEngine::URMA>(
        setTarget, setSource, signal, kMixedSetValue, pto::comm::NotifyOp::Set, session, peer);
    const pto::comm::AsyncEvent getEvent =
        pto::comm::TGET_ASYNC<pto::comm::DmaEngine::URMA>(getTarget, getSource, session, peer);
    finalEvent = pto::comm::TPUT_ASYNC_NOTIFY<pto::comm::DmaEngine::URMA>(
        addTarget, addSource, signal, kMixedAddValue, pto::comm::NotifyOp::AtomicAdd, session, peer);
    if (putEvent.valid() && setEvent.valid() && getEvent.valid() && finalEvent.valid()) {
        return 0;
    }
    (void)finalEvent.Wait(session);
    return -3;
}

AICORE inline int32_t SubmitNotifyMode(
    UrmaNotifyStMode mode, __gm__ uint8_t* localBase, uint64_t peerBase, __gm__ int32_t* localSource,
    __gm__ int32_t* remoteTarget, pto::comm::Signal& signal, const pto::comm::AsyncSession& session, uint32_t peer,
    pto::comm::AsyncEvent& finalEvent)
{
    switch (mode) {
        case UrmaNotifyStMode::Set:
        case UrmaNotifyStMode::ReceiverConsumeSet:
            finalEvent = SubmitSingleNotify(
                localSource, remoteTarget, signal, kSetValue, pto::comm::NotifyOp::Set, session, peer);
            return 0;
        case UrmaNotifyStMode::AtomicAdd:
            finalEvent = SubmitSingleNotify(
                localSource, remoteTarget, signal, kAddValue, pto::comm::NotifyOp::AtomicAdd, session, peer);
            return 0;
        case UrmaNotifyStMode::SetBatchDrain65:
            return SubmitNotifyBatch(
                localSource, remoteTarget, signal, pto::comm::NotifyOp::Set, session, peer, finalEvent);
        case UrmaNotifyStMode::FaaSharedSink65:
            return SubmitNotifyBatch(
                localSource, remoteTarget, signal, pto::comm::NotifyOp::AtomicAdd, session, peer, finalEvent);
        case UrmaNotifyStMode::MixedPutSetGetAdd:
            return SubmitMixedNotifyFlow(
                localBase, peerBase, localSource, remoteTarget, signal, session, peer, finalEvent);
        default:
            return -5;
    }
}

AICORE inline void CompleteNotifyMode(
    __gm__ int32_t* status, const pto::comm::AsyncSession& session, const pto::comm::AsyncEvent& finalEvent)
{
    const bool submitted = finalEvent.valid();
    const bool waited = submitted && finalEvent.Wait(session);
    const bool completed = waited && finalEvent.Test(session);
    StoreStatus(status, completed ? kStatusSuccess : -6);
}

__global__ AICORE void TPutAsyncNotifyUrmaKernel(
    __gm__ uint8_t* localBase, __gm__ uint8_t* urmaWorkspace, uint32_t rankId, uint32_t firstRankId, uint32_t mode)
{
#ifdef PTO_URMA_SUPPORTED
    const UrmaNotifyStMode notifyMode = static_cast<UrmaNotifyStMode>(mode);
    if (rankId != firstRankId) {
        if (rankId == kTargetPeer && notifyMode == UrmaNotifyStMode::ReceiverConsumeSet) {
            ConsumeNotifyOnReceiver(localBase);
        }
        return;
    }
    __gm__ int32_t* status = reinterpret_cast<__gm__ int32_t*>(localBase + kStatusOffset);
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(urmaWorkspace, session)) {
        StoreStatus(status, -1);
        return;
    }
    const uint64_t peerBase = pto::comm::urma::UrmaPeerMrBaseAddr(urmaWorkspace, kTargetPeer);
    __gm__ int32_t* localSource = reinterpret_cast<__gm__ int32_t*>(localBase + kSourceOffset);
    __gm__ int32_t* remoteTarget = reinterpret_cast<__gm__ int32_t*>(peerBase + kTargetOffset);
    pto::comm::Signal remoteSignal(reinterpret_cast<__gm__ int32_t*>(peerBase + kSignalOffset));
    pto::comm::AsyncEvent finalEvent;
    const int32_t submitStatus = SubmitNotifyMode(
        notifyMode, localBase, peerBase, localSource, remoteTarget, remoteSignal, session, kTargetPeer, finalEvent);
    if (submitStatus != 0) {
        StoreStatus(status, submitStatus);
        return;
    }
    CompleteNotifyMode(status, session, finalEvent);
#else
    (void)localBase;
    (void)urmaWorkspace;
    (void)rankId;
    (void)firstRankId;
    (void)mode;
#endif
}

bool ValidateAllRanks(bool localOk, int nRanks)
{
    const uint8_t localState = localOk ? 1U : 0U;
    std::vector<uint8_t> states(static_cast<size_t>(nRanks), 0U);
    if (CommMpiAllgather(&localState, 1, states.data(), 1) != 0) {
        return false;
    }
    for (uint8_t state : states) {
        if (state == 0U) {
            return false;
        }
    }
    return true;
}

UrmaTestContext gUrmaTestContext;
bool gUrmaTestContextInitialized = false;

int32_t InitialSignal(UrmaNotifyStMode mode)
{
    return mode == UrmaNotifyStMode::AtomicAdd || mode == UrmaNotifyStMode::FaaSharedSink65 ? kAddInitial : 0;
}

struct NotifyHostData {
    std::vector<int32_t> source;
    std::vector<int32_t> target;
    int32_t getSource[kPayloadElems] = {9000, 9001, 9002, 9003};
    int32_t getDest[kPayloadElems] = {kPoison, kPoison, kPoison, kPoison};
    int32_t consumed[kConsumeElems] = {kPoison, kPoison, kPoison, kPoison, kPoison, kPoison, kPoison, kPoison};
    int32_t signal;
    int32_t status{kStatusInitial};
    int32_t primeSink{kPoison};

    explicit NotifyHostData(UrmaNotifyStMode mode)
        : source(kTotalPayloadElems), target(kTotalPayloadElems, kPoison), signal(InitialSignal(mode))
    {
        for (uint32_t i = 0; i < kTotalPayloadElems; ++i) {
            source[i] = 1000 + static_cast<int32_t>(i);
        }
    }
};

bool EnsureNotifyTestContext(int rankId, int nRanks, int nDevices, int firstDeviceId, int rootRank)
{
    if (gUrmaTestContextInitialized) {
        return true;
    }
    if (!gUrmaTestContext.Setup(rankId, nRanks, nDevices, firstDeviceId, rootRank, kCommBytes)) {
        return false;
    }
    gUrmaTestContextInitialized = true;
    return true;
}

bool CopyNotifyInputs(uint8_t* base, const NotifyHostData& data)
{
    return aclrtMemcpy(
               base + kCanaryBeforeOffset, sizeof(kCanaryBefore), &kCanaryBefore, sizeof(kCanaryBefore),
               ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
           aclrtMemcpy(
               base + kSignalOffset, sizeof(data.signal), &data.signal, sizeof(data.signal),
               ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
           aclrtMemcpy(
               base + kCanaryAfterOffset, sizeof(kCanaryAfter), &kCanaryAfter, sizeof(kCanaryAfter),
               ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
           aclrtMemcpy(
               base + kStatusOffset, sizeof(data.status), &data.status, sizeof(data.status),
               ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
           aclrtMemcpy(
               base + kPrimeSinkOffset, sizeof(data.primeSink), &data.primeSink, sizeof(data.primeSink),
               ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
           aclrtMemcpy(
               base + kSourceOffset, data.source.size() * sizeof(int32_t), data.source.data(),
               data.source.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
           aclrtMemcpy(
               base + kTargetOffset, data.target.size() * sizeof(int32_t), data.target.data(),
               data.target.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
           aclrtMemcpy(
               base + kGetSourceOffset, sizeof(data.getSource), data.getSource, sizeof(data.getSource),
               ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
           aclrtMemcpy(
               base + kGetDestOffset, sizeof(data.getDest), data.getDest, sizeof(data.getDest),
               ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
           aclrtMemcpy(
               base + kConsumeOffset, sizeof(data.consumed), data.consumed, sizeof(data.consumed),
               ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
}

bool ValidateRootResult(uint8_t* base, UrmaNotifyStMode mode, const NotifyHostData& data)
{
    int32_t actualStatus = 0;
    aclrtMemcpy(
        &actualStatus, sizeof(actualStatus), base + kStatusOffset, sizeof(actualStatus), ACL_MEMCPY_DEVICE_TO_HOST);
    if (actualStatus != kStatusSuccess) {
        return false;
    }
    if (mode != UrmaNotifyStMode::MixedPutSetGetAdd) {
        return true;
    }
    int32_t actualGet[kPayloadElems] = {};
    aclrtMemcpy(actualGet, sizeof(actualGet), base + kGetDestOffset, sizeof(actualGet), ACL_MEMCPY_DEVICE_TO_HOST);
    for (uint32_t i = 0; i < kPayloadElems; ++i) {
        if (actualGet[i] != data.getSource[i]) {
            return false;
        }
    }
    return true;
}

void ExpectedPeerResult(UrmaNotifyStMode mode, uint32_t& payloadsToCheck, int32_t& expectedSignal)
{
    payloadsToCheck = 1U;
    expectedSignal = kSetValue;
    if (mode == UrmaNotifyStMode::AtomicAdd) {
        expectedSignal = kAddInitial + kAddValue;
    } else if (mode == UrmaNotifyStMode::SetBatchDrain65) {
        payloadsToCheck = kNotifyCount;
        expectedSignal = static_cast<int32_t>(kNotifyCount);
    } else if (mode == UrmaNotifyStMode::FaaSharedSink65) {
        payloadsToCheck = kNotifyCount;
        expectedSignal = kAddInitial + static_cast<int32_t>(kNotifyCount);
    } else if (mode == UrmaNotifyStMode::MixedPutSetGetAdd) {
        payloadsToCheck = 3U;
        expectedSignal = kMixedSetValue + kMixedAddValue;
    }
}

bool ValidatePeerResult(uint8_t* base, UrmaNotifyStMode mode, const NotifyHostData& data)
{
    int32_t actualBefore = 0;
    int32_t actualSignal = 0;
    int32_t actualAfter = 0;
    aclrtMemcpy(
        &actualBefore, sizeof(actualBefore), base + kCanaryBeforeOffset, sizeof(actualBefore),
        ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(
        &actualSignal, sizeof(actualSignal), base + kSignalOffset, sizeof(actualSignal), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(
        &actualAfter, sizeof(actualAfter), base + kCanaryAfterOffset, sizeof(actualAfter), ACL_MEMCPY_DEVICE_TO_HOST);
    uint32_t payloadsToCheck = 0U;
    int32_t expectedSignal = 0;
    ExpectedPeerResult(mode, payloadsToCheck, expectedSignal);
    if (actualBefore != kCanaryBefore || actualAfter != kCanaryAfter || actualSignal != expectedSignal) {
        return false;
    }
    std::vector<int32_t> actual(payloadsToCheck * kPayloadElems);
    aclrtMemcpy(
        actual.data(), actual.size() * sizeof(int32_t), base + kTargetOffset, actual.size() * sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != data.source[i]) {
            return false;
        }
    }
    if (mode == UrmaNotifyStMode::ReceiverConsumeSet) {
        int32_t actualStatus = 0;
        int32_t consumed[kConsumeElems] = {};
        aclrtMemcpy(
            &actualStatus, sizeof(actualStatus), base + kStatusOffset, sizeof(actualStatus), ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(consumed, sizeof(consumed), base + kConsumeOffset, sizeof(consumed), ACL_MEMCPY_DEVICE_TO_HOST);
        if (actualStatus != kStatusSuccess) {
            return false;
        }
        for (uint32_t i = 0U; i < kConsumeElems; ++i) {
            const int32_t expected = (i < kPayloadElems ? data.source[i] : kPoison) + kConsumeAdd;
            if (consumed[i] != expected) {
                return false;
            }
        }
    }
    return true;
}

bool RunTPutAsyncNotifyUrmaKernel(
    int rankId, int nRanks, int nDevices, int firstDeviceId, int firstRankId, int rootRank, UrmaNotifyStMode mode)
{
    if (!EnsureNotifyTestContext(rankId, nRanks, nDevices, firstDeviceId, rootRank)) {
        return false;
    }
    UrmaTestContext& ctx = gUrmaTestContext;
    NotifyHostData data(mode);
    uint8_t* base = reinterpret_cast<uint8_t*>(ctx.devBuf);
    if (!ValidateAllRanks(CopyNotifyInputs(base, data), nRanks)) {
        return false;
    }
    if (mode == UrmaNotifyStMode::ReceiverConsumeSet) {
        PrimeNotifyPayloadCache<<<1, nullptr, ctx.stream>>>(
            base, static_cast<uint32_t>(rankId), static_cast<uint32_t>(mode));
        const bool primed = aclrtSynchronizeStream(ctx.stream) == ACL_SUCCESS;
        if (!ValidateAllRanks(primed, nRanks)) {
            return false;
        }
    }
    CommMpiBarrier();
    TPutAsyncNotifyUrmaKernel<<<1, nullptr, ctx.stream>>>(
        base, reinterpret_cast<uint8_t*>(ctx.urmaMgr.GetWorkspaceAddr()), static_cast<uint32_t>(rankId),
        static_cast<uint32_t>(firstRankId), static_cast<uint32_t>(mode));
    const int syncRet = aclrtSynchronizeStream(ctx.stream);
    CommMpiBarrier();
    bool localOk = syncRet == ACL_SUCCESS;
    if (rankId == rootRank) {
        localOk = localOk && ValidateRootResult(base, mode, data);
    } else if (rankId == rootRank + 1) {
        localOk = localOk && ValidatePeerResult(base, mode, data);
    }
    return ValidateAllRanks(localOk, nRanks);
}

UrmaNotifyStMode gUrmaNotifyStMode = UrmaNotifyStMode::Set;

bool RunTPutAsyncNotifyUrmaEntry(int rankId, int nRanks, int nDevices, int firstDeviceId, int firstRankId, int rootRank)
{
    return RunTPutAsyncNotifyUrmaKernel(
        rankId, nRanks, nDevices, firstDeviceId, firstRankId, rootRank, gUrmaNotifyStMode);
}

} // namespace

bool RunTPutAsyncNotifyUrma(int nRanks, int nDevices, int firstRankId, int firstDeviceId, UrmaNotifyStMode mode)
{
    gUrmaNotifyStMode = mode;
    return RunUrmaTestMpiLaunch(nRanks, nDevices, firstRankId, firstDeviceId, RunTPutAsyncNotifyUrmaEntry);
}

void FinalizeTPutAsyncNotifyUrma()
{
    if (gUrmaTestContextInitialized) {
        gUrmaTestContext.Cleanup();
        gUrmaTestContextInitialized = false;
    }
}
