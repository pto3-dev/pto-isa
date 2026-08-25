/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <pto/pto-inst.hpp>

#include "../common.hpp"
#include "tput_async_rdma_kernel.h"
#include "pto/common/pto_tile.hpp"
#ifdef PTO_RDMA_SUPPORTED
#include "pto/comm/async/rdma/rdma_async_intrin.hpp"
#include "pto/comm/async/rdma/rdma_workspace_manager.hpp"
#include "backends/rdma_test_backend.hpp"
#endif

#ifdef PTO_RDMA_SUPPORTED
template <typename... Args>
static void RdmaTrace(uint32_t caseId, int rankId, Args&&... args)
{
    if (!pto::comm::rdma::test::BackendVerboseEnabled()) {
        return;
    }
    std::ostringstream os;
    os << "[RDMA][" << pto::comm::rdma::RdmaWorkspaceManager::ConfiguredBackendName() << "][case " << caseId
       << "][rank " << rankId << "] ";
    (os << ... << std::forward<Args>(args));
    os << '\n';
    const std::string line = os.str();
    (void)std::fwrite(line.data(), 1, line.size(), stderr);
    std::fflush(stderr);
}

static uint64_t RdmaElapsedUs(const std::chrono::steady_clock::time_point& start)
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
}

static std::string RdmaHex(uint64_t value)
{
    std::ostringstream os;
    os << "0x" << std::hex << value;
    return os.str();
}

static uint32_t gRdmaCaseSequence = 0;
#endif

// ============================================================================
// TPUT_ASYNC via RDMA.
//
// Data plane is a true one-sided remote write posted from AIV: root rank writes
// its send buffer into every peer's recv buffer. The symmetric communication
// buffer layout matches the URMA test:
//   [64 x int32 header][sendBuf: count x T][recvBuf: count x T]
// The remote target VA is computed from the peer's registered MR base VA
// (PeerMrBaseAddr) plus the recv-region offset.
// ============================================================================

#ifdef PTO_RDMA_SUPPORTED
constexpr uint32_t kRdmaPublicEventWaitError = 0x30000;
constexpr uint32_t kRdmaPublicEventTestError = 0x30001;
constexpr size_t kRdmaTestDataOffset = 64 * sizeof(int32_t);
constexpr size_t kRdmaNotifyCanaryBeforeOffset = sizeof(uint32_t);
constexpr size_t kRdmaNotifySignalOffset = 2U * sizeof(uint32_t);
constexpr size_t kRdmaNotifyCanaryAfterOffset = 3U * sizeof(uint32_t);
constexpr int32_t kRdmaNotifySignalValue = 37;
constexpr int32_t kRdmaNotifyCanaryBefore = 0x13572468;
constexpr int32_t kRdmaNotifyCanaryAfter = 0x24681357;
constexpr uint32_t kRdmaNotifyPollLimit = 10000000U;
using RdmaTestShape = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using RdmaTestStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using RdmaScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::rdma::kRdmaScratchBytes>;

template <typename T>
using RdmaTestGlobal = pto::GlobalTensor<T, RdmaTestShape, RdmaTestStride, pto::Layout::ND>;

AICORE inline uint32_t CompleteRdmaEvent(
    pto::comm::AsyncEvent& event, pto::comm::AsyncSession& session, RdmaCompletionMode completionMode)
{
    if (completionMode != RdmaCompletionMode::PUBLIC_EVENT_WAIT_TEST) {
        return pto::comm::rdma::WaitEventStatus(event.handle, session);
    }

    // The first Test may legitimately be false. Wait must complete the event,
    // and Test must then observe the already-consumed target index as complete.
    (void)event.Test(session);
    if (!event.Wait(session)) {
        return kRdmaPublicEventWaitError;
    }
    return event.Test(session) ? 0 : kRdmaPublicEventTestError;
}

AICORE inline uint32_t CompleteRdmaOperation(
    pto::comm::AsyncEvent& event, pto::comm::AsyncSession& session, RdmaCompletionMode completionMode)
{
    return completionMode == RdmaCompletionMode::STATUS_WAIT_EACH ? CompleteRdmaEvent(event, session, completionMode) :
                                                                    0;
}

AICORE inline uint32_t CompleteRdmaPeer(
    pto::comm::AsyncEvent& lastEvent, pto::comm::AsyncSession& session, RdmaCompletionMode completionMode,
    uint32_t currentStatus)
{
    if (currentStatus != 0 || completionMode == RdmaCompletionMode::STATUS_WAIT_EACH) {
        return currentStatus;
    }
    return CompleteRdmaEvent(lastEvent, session, completionMode);
}

struct RdmaCompletionState {
    pto::comm::AsyncEvent lastEvent;
    uint32_t status{0};
};

AICORE inline bool BuildRdmaTestSession(
    RdmaScratchTile& scratchTile, __gm__ uint8_t* rdmaWorkspace, uint32_t myPeer, pto::comm::AsyncSession& session,
    uint32_t syncId, __gm__ uint32_t* deviceStatus)
{
    if (pto::comm::BuildAsyncSession<pto::comm::DmaEngine::RDMA>(scratchTile, rdmaWorkspace, myPeer, session, syncId)) {
        return true;
    }
    *deviceStatus = pto::comm::rdma::kRdmaSessionBuildError;
    pipe_barrier(PIPE_ALL);
    return false;
}

template <typename T, size_t count>
AICORE inline uint32_t PostPutOperations(
    __gm__ T* sendBuf, uint64_t peerBase, uint32_t targetPeer, int elemOffset, int elemCount, int operationCount,
    const RdmaTestShape& shape, const RdmaTestStride& stride, pto::comm::AsyncSession& session,
    RdmaCompletionMode completionMode)
{
    RdmaCompletionState completion;
    for (int operation = 0; operation < operationCount; ++operation) {
        const int operationOffset = elemOffset + operation * elemCount;
        RdmaTestGlobal<T> sendGlobal(sendBuf + operationOffset, shape, stride);
        __gm__ T* remoteRecv = reinterpret_cast<__gm__ T*>(peerBase + kRdmaTestDataOffset) + count + operationOffset;
        RdmaTestGlobal<T> remoteRecvGlobal(remoteRecv, shape, stride);
        completion.lastEvent =
            pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::RDMA>(remoteRecvGlobal, sendGlobal, session, targetPeer);
        completion.status = CompleteRdmaOperation(completion.lastEvent, session, completionMode);
        if (completion.status != 0) {
            break;
        }
    }
    return CompleteRdmaPeer(completion.lastEvent, session, completionMode, completion.status);
}

template <typename T, size_t count>
AICORE inline uint32_t PostGetOperations(
    __gm__ T* recvBuf, uint64_t peerBase, uint32_t sourcePeer, int elemOffset, int elemCount, int operationCount,
    const RdmaTestShape& shape, const RdmaTestStride& stride, pto::comm::AsyncSession& session,
    RdmaCompletionMode completionMode)
{
    RdmaCompletionState completion;
    for (int operation = 0; operation < operationCount; ++operation) {
        const int operationOffset = elemOffset + operation * elemCount;
        __gm__ T* remoteSend = reinterpret_cast<__gm__ T*>(peerBase + kRdmaTestDataOffset) + operationOffset;
        __gm__ T* localRecv = recvBuf + static_cast<size_t>(sourcePeer) * count + operationOffset;
        RdmaTestGlobal<T> localRecvGlobal(localRecv, shape, stride);
        RdmaTestGlobal<T> remoteSendGlobal(remoteSend, shape, stride);
        completion.lastEvent =
            pto::comm::TGET_ASYNC<pto::comm::DmaEngine::RDMA>(localRecvGlobal, remoteSendGlobal, session, sourcePeer);
        completion.status = CompleteRdmaOperation(completion.lastEvent, session, completionMode);
        if (completion.status != 0) {
            break;
        }
    }
    return CompleteRdmaPeer(completion.lastEvent, session, completionMode, completion.status);
}

template <typename T, size_t count>
AICORE inline void ExecutePutRdma(
    __gm__ T* localBuf, int nranks, int myRank, int firstRankId, int rootRank, int elemOffset, int elemCount,
    int operationCount, RdmaCompletionMode completionMode, __gm__ uint8_t* rdmaWorkspace, uint32_t syncId)
{
    __gm__ uint32_t* deviceStatus = reinterpret_cast<__gm__ uint32_t*>(localBuf);
    *deviceStatus = 0;
    if (myRank != rootRank) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    const RdmaTestShape shape(1, 1, 1, 1, elemCount);
    const RdmaTestStride stride(elemCount, elemCount, elemCount, elemCount, 1);
    __gm__ T* sendBuf = reinterpret_cast<__gm__ T*>(reinterpret_cast<__gm__ uint8_t*>(localBuf) + kRdmaTestDataOffset);
    pipe_barrier(PIPE_ALL);

    const uint32_t myPeer = static_cast<uint32_t>(myRank - firstRankId);
    RdmaScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    if (!BuildRdmaTestSession(scratchTile, rdmaWorkspace, myPeer, session, syncId, deviceStatus)) {
        return;
    }
    for (uint32_t targetPeer = 0; targetPeer < static_cast<uint32_t>(nranks); ++targetPeer) {
        if (targetPeer == myPeer) {
            continue;
        }
        const uint64_t peerBase = pto::comm::rdma::PeerMrBaseAddr(rdmaWorkspace, targetPeer);
        const uint32_t completionStatus = PostPutOperations<T, count>(
            sendBuf, peerBase, targetPeer, elemOffset, elemCount, operationCount, shape, stride, session,
            completionMode);
        *deviceStatus = completionStatus;
        if (completionStatus != 0) {
            break;
        }
    }
    pipe_barrier(PIPE_ALL);
}

#ifdef PTO_RDMA_GET_TEST
template <typename T, size_t count>
AICORE inline void ExecuteGetRdma(
    __gm__ T* localBuf, int nranks, int myRank, int firstRankId, int rootRank, int elemOffset, int elemCount,
    int operationCount, RdmaCompletionMode completionMode, __gm__ uint8_t* rdmaWorkspace, uint32_t syncId)
{
    __gm__ uint32_t* deviceStatus = reinterpret_cast<__gm__ uint32_t*>(localBuf);
    *deviceStatus = 0;
    const RdmaTestShape shape(1, 1, 1, 1, elemCount);
    const RdmaTestStride stride(elemCount, elemCount, elemCount, elemCount, 1);
    __gm__ T* sendBuf = reinterpret_cast<__gm__ T*>(reinterpret_cast<__gm__ uint8_t*>(localBuf) + kRdmaTestDataOffset);
    __gm__ T* recvBuf = sendBuf + count;
    pipe_barrier(PIPE_ALL);
    if (myRank != rootRank) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    const uint32_t myPeer = static_cast<uint32_t>(myRank - firstRankId);
    RdmaScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    if (!BuildRdmaTestSession(scratchTile, rdmaWorkspace, myPeer, session, syncId, deviceStatus)) {
        return;
    }
    for (uint32_t sourcePeer = 0; sourcePeer < static_cast<uint32_t>(nranks); ++sourcePeer) {
        if (sourcePeer == myPeer) {
            continue;
        }
        const uint64_t peerBase = pto::comm::rdma::PeerMrBaseAddr(rdmaWorkspace, sourcePeer);
        const uint32_t completionStatus = PostGetOperations<T, count>(
            recvBuf, peerBase, sourcePeer, elemOffset, elemCount, operationCount, shape, stride, session,
            completionMode);
        *deviceStatus = completionStatus;
        if (completionStatus != 0) {
            break;
        }
    }
    pipe_barrier(PIPE_ALL);
}
#endif // PTO_RDMA_GET_TEST
#endif

template <typename T, size_t count>
[[bisheng::core_ratio(0, 1)]] __global__ AICORE void TPutAsyncRdmaKernelImpl(
    __gm__ T* localBuf, int nranks, int my_rank, int first_rank_id, int root_rank, int elem_offset, int elem_count,
    int operation_count, RdmaCompletionMode completionMode, __gm__ uint8_t* rdmaWorkspace, uint32_t syncId)
{
#ifdef PTO_RDMA_SUPPORTED
    ExecutePutRdma<T, count>(
        localBuf, nranks, my_rank, first_rank_id, root_rank, elem_offset, elem_count, operation_count, completionMode,
        rdmaWorkspace, syncId);
#else
    *reinterpret_cast<__gm__ uint32_t*>(localBuf) = 0;
    (void)nranks;
    (void)first_rank_id;
    (void)elem_offset;
    (void)elem_count;
    (void)operation_count;
    (void)completionMode;
    (void)rdmaWorkspace;
    (void)syncId;
    if (my_rank != root_rank) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    pipe_barrier(PIPE_ALL);
    pipe_barrier(PIPE_ALL);
#endif
}

#if defined(PTO_RDMA_SUPPORTED) && defined(PTO_RDMA_GET_TEST)
// Build the GET entry points for the tget_async_rdma target.
template <typename T, size_t count>
[[bisheng::core_ratio(0, 1)]] __global__ AICORE void TGetAsyncRdmaKernelImpl(
    __gm__ T* localBuf, int nranks, int my_rank, int first_rank_id, int root_rank, int elem_offset, int elem_count,
    int operation_count, RdmaCompletionMode completionMode, __gm__ uint8_t* rdmaWorkspace, uint32_t syncId)
{
    ExecuteGetRdma<T, count>(
        localBuf, nranks, my_rank, first_rank_id, root_rank, elem_offset, elem_count, operation_count, completionMode,
        rdmaWorkspace, syncId);
}
#endif // PTO_RDMA_SUPPORTED && PTO_RDMA_GET_TEST

template <size_t count>
[[bisheng::core_ratio(0, 1)]] __global__ AICORE void TPutAsyncNotifyRdmaKernelImpl(
    __gm__ int32_t* localBuf, int myRank, int firstRankId, int rootRank, __gm__ uint8_t* rdmaWorkspace, uint32_t syncId)
{
#ifdef PTO_RDMA_SUPPORTED
    __gm__ uint8_t* localBytes = reinterpret_cast<__gm__ uint8_t*>(localBuf);
    __gm__ uint32_t* deviceStatus = reinterpret_cast<__gm__ uint32_t*>(localBytes);
    __gm__ int32_t* localSignal = reinterpret_cast<__gm__ int32_t*>(localBytes + kRdmaNotifySignalOffset);
    __gm__ int32_t* sendBuf = reinterpret_cast<__gm__ int32_t*>(localBytes + kRdmaTestDataOffset);
    __gm__ int32_t* recvBuf = sendBuf + count;
    *deviceStatus = 0U;

    if (myRank != rootRank) {
        pto::comm::Signal signal(localSignal);
        bool signaled = false;
        for (uint32_t poll = 0U; poll < kRdmaNotifyPollLimit; ++poll) {
            if (pto::comm::TTEST(signal, kRdmaNotifySignalValue, pto::comm::WaitCmp::EQ)) {
                signaled = true;
                break;
            }
        }
        if (!signaled) {
            *deviceStatus = kRdmaPublicEventWaitError;
            pipe_barrier(PIPE_ALL);
            return;
        }

        __asm__ __volatile__("");
        dcci(static_cast<__gm__ void*>(0), cache_line_t::ENTIRE_DATA_CACHE);
        __asm__ __volatile__("");
        for (uint32_t index = 0U; index < count; ++index) {
            if (recvBuf[index] != static_cast<int32_t>(index + static_cast<uint32_t>(rootRank) * 10000U)) {
                *deviceStatus = kRdmaPublicEventTestError;
                pipe_barrier(PIPE_ALL);
                return;
            }
        }
        pipe_barrier(PIPE_ALL);
        return;
    }

    const uint32_t myPeer = static_cast<uint32_t>(myRank - firstRankId);
    const uint32_t targetPeer = myPeer + 1U;
    RdmaScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    if (!BuildRdmaTestSession(scratchTile, rdmaWorkspace, myPeer, session, syncId, deviceStatus)) {
        return;
    }

    const uint64_t peerBase = pto::comm::rdma::PeerMrBaseAddr(rdmaWorkspace, targetPeer);
    const RdmaTestShape shape(1, 1, 1, 1, static_cast<int>(count));
    const RdmaTestStride stride(
        static_cast<int>(count), static_cast<int>(count), static_cast<int>(count), static_cast<int>(count), 1);
    RdmaTestGlobal<int32_t> source(sendBuf, shape, stride);
    RdmaTestGlobal<int32_t> destination(
        reinterpret_cast<__gm__ int32_t*>(peerBase + kRdmaTestDataOffset) + count, shape, stride);
    pto::comm::Signal remoteSignal(reinterpret_cast<__gm__ int32_t*>(peerBase + kRdmaNotifySignalOffset));
    pto::comm::AsyncEvent event = pto::comm::TPUT_ASYNC_NOTIFY<pto::comm::DmaEngine::RDMA>(
        destination, source, remoteSignal, kRdmaNotifySignalValue, pto::comm::NotifyOp::Set, session, targetPeer);
    *deviceStatus = CompleteRdmaEvent(event, session, RdmaCompletionMode::PUBLIC_EVENT_WAIT_TEST);
    pipe_barrier(PIPE_ALL);
#else
    (void)localBuf;
    (void)myRank;
    (void)firstRankId;
    (void)rootRank;
    (void)rdmaWorkspace;
    (void)syncId;
#endif
}

static bool AllRanksReady(bool localReady, int nRanks, const char* stage, bool* anyRankNotReady = nullptr)
{
    if (anyRankNotReady != nullptr) {
        *anyRankNotReady = false;
    }
    uint8_t local = localReady ? 1 : 0;
    std::vector<uint8_t> all(static_cast<size_t>(nRanks), 0);
    if (CommMpiAllgather(&local, 1, all.data(), 1) != 0) {
        std::cerr << "[ERROR] MPI_Allgather failed while agreeing on " << stage << std::endl;
        return false;
    }
    for (int rank = 0; rank < nRanks; ++rank) {
        if (all[rank] == 0) {
            if (anyRankNotReady != nullptr) {
                *anyRankNotReady = true;
            }
            std::cerr << "[RDMA] rank " << rank << " is not ready at stage: " << stage << std::endl;
        }
    }
    return anyRankNotReady == nullptr ? std::all_of(all.begin(), all.end(), [](uint8_t ready) { return ready != 0; }) :
                                        !*anyRankNotReady;
}

#ifdef PTO_RDMA_SUPPORTED
static pto::comm::rdma::WorkspaceInitResult AgreeOnRdmaPreflight(int nRanks)
{
    using pto::comm::rdma::WorkspaceInitResult;

    const auto localPreflight = pto::comm::rdma::RdmaWorkspaceManager::Preflight();
    const uint8_t localCode = static_cast<uint8_t>(localPreflight);
    std::vector<uint8_t> allCodes(static_cast<size_t>(nRanks), 0);
    if (CommMpiAllgather(&localCode, 1, allCodes.data(), 1) != 0) {
        std::cerr << "[ERROR] MPI_Allgather failed while agreeing on RDMA backend selection" << std::endl;
        return WorkspaceInitResult::ERROR;
    }

    const bool allDisabled = std::all_of(allCodes.begin(), allCodes.end(), [](uint8_t value) {
        return value == static_cast<uint8_t>(WorkspaceInitResult::DISABLED);
    });
    const bool allReady = std::all_of(allCodes.begin(), allCodes.end(), [](uint8_t value) {
        return value == static_cast<uint8_t>(WorkspaceInitResult::READY);
    });
    if (allDisabled) {
        if (CommMpiRank() == 0) {
            std::cerr << "[SKIP] no RDMA backend was compiled into this binary (compiled backend="
                      << pto::comm::rdma::RdmaWorkspaceManager::ConfiguredBackendName() << ")" << std::endl;
        }
        return WorkspaceInitResult::DISABLED;
    }
    if (!allReady) {
        if (CommMpiRank() == 0) {
            std::cerr << "[ERROR] inconsistent or unsupported RDMA backend selection across ranks" << std::endl;
        }
        return WorkspaceInitResult::ERROR;
    }
    return WorkspaceInitResult::READY;
}

// ============================================================================
// RdmaTestContext: device, registered communication buffer, and RDMA workspace manager.
//
// Each case owns a complete backend-channel lifecycle. Reusing the same port
// across the suite validates channel destruction followed by reconnection.
// ============================================================================
struct RdmaTestContext {
    int deviceId{-1};
    int rankId{-1};
    int nRanks{0};
    rtStream_t stream{nullptr};
    void* devBuf{nullptr};
    size_t allocSize{0};
    pto::comm::rdma::RdmaWorkspaceManager rdmaMgr;
    pto::comm::rdma::test::BackendBootstrap boot;
    uint32_t traceId{0};

    enum class SetupResult {
        READY,
        FAILED,
        SKIPPED,
    };

    bool SetupLocalResources(size_t commBytesNeeded)
    {
        bool localReady = aclrtSetDevice(deviceId) == ACL_SUCCESS;
        if (!localReady) {
            std::cerr << "[ERROR] aclrtSetDevice(" << deviceId << ") failed" << std::endl;
        }
        if (localReady && rtStreamCreate(&stream, RT_STREAM_PRIORITY_DEFAULT) != 0) {
            std::cerr << "[ERROR] rtStreamCreate failed" << std::endl;
            localReady = false;
        }
        allocSize = commBytesNeeded;
        if (localReady &&
            (aclrtMalloc(&devBuf, allocSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS || devBuf == nullptr)) {
            std::cerr << "[ERROR] aclrtMalloc(" << allocSize << ") failed" << std::endl;
            localReady = false;
        }
        if (localReady && aclrtMemset(devBuf, allocSize, 0, allocSize) != ACL_SUCCESS) {
            std::cerr << "[ERROR] aclrtMemset failed" << std::endl;
            localReady = false;
        }
        return AllRanksReady(localReady, nRanks, "local ACL resource setup");
    }

    std::string DescribePeers() const
    {
        std::ostringstream peers;
        for (int peer = 0; peer < nRanks; ++peer) {
            if (peer != 0) {
                peers << ',';
            }
            peers << peer << ':' << boot.peerIps[peer] << "/phy" << boot.peerPhyIds[peer] << "/sym"
                  << RdmaHex(boot.peerSymAddrs[peer]);
        }
        return peers.str();
    }

    SetupResult SetupBootstrap()
    {
        if (!boot.Init(rankId, nRanks, deviceId, devBuf, AllRanksReady)) {
            return boot.skipped ? SetupResult::SKIPPED : SetupResult::FAILED;
        }
        RdmaTrace(traceId, rankId, "SETUP bootstrap ready basePort=", boot.basePort, " peers=[", DescribePeers(), ']');
        return SetupResult::READY;
    }

    bool ConnectRdma()
    {
        pto::comm::rdma::WorkspaceConfig config{};
        config.rankId = static_cast<uint32_t>(rankId);
        config.rankCount = static_cast<uint32_t>(nRanks);
        config.phyId = boot.phyId;
        config.localIp = boot.peerIps[rankId];
        config.basePort = boot.basePort;
        config.peerIps = boot.peerIps;
        config.peerPhyIds = boot.peerPhyIds;
        config.peerSymAddrs = boot.peerSymAddrs;
        config.symmetricAddr = devBuf;
        config.symmetricSize = allocSize;
        const bool localConnected = rdmaMgr.Init(config) == pto::comm::rdma::WorkspaceInitResult::READY;
        if (!localConnected) {
            std::cerr << "[ERROR] RDMA workspace initialization failed" << std::endl;
        }
        return AllRanksReady(localConnected, nRanks, "RDMA backend channel initialization");
    }

    SetupResult Setup(
        uint32_t caseId, int rank_id, int n_ranks, int n_devices, int first_device_id, size_t commBytesNeeded)
    {
        const auto setupStart = std::chrono::steady_clock::now();
        traceId = caseId;
        rankId = rank_id;
        nRanks = n_ranks;
        deviceId = rank_id % n_devices + first_device_id;
        rdmaMgr.SetTraceId(traceId);
        RdmaTrace(traceId, rankId, "SETUP begin device=", deviceId, " commBytes=", commBytesNeeded);
        if (!SetupLocalResources(commBytesNeeded)) {
            return SetupResult::FAILED;
        }
        RdmaTrace(
            traceId, rankId, "SETUP local resources ready stream=", stream, " devBuf=", devBuf,
            " allocSize=", allocSize, " elapsed_us=", RdmaElapsedUs(setupStart));
        const SetupResult bootstrapResult = SetupBootstrap();
        if (bootstrapResult != SetupResult::READY) {
            return bootstrapResult;
        }
        CommMpiBarrier();
        const auto connectStart = std::chrono::steady_clock::now();
        if (!ConnectRdma()) {
            return SetupResult::FAILED;
        }
        RdmaTrace(
            traceId, rankId, "SETUP RDMA backend ready connect_us=", RdmaElapsedUs(connectStart),
            " total_us=", RdmaElapsedUs(setupStart));
        return SetupResult::READY;
    }

    bool Cleanup()
    {
        const auto cleanupStart = std::chrono::steady_clock::now();
        RdmaTrace(traceId, rankId, "CLEANUP begin devBuf=", devBuf, " stream=", stream);
        CommMpiBarrier();
        bool localOk = rdmaMgr.Finalize();
        if (devBuf != nullptr) {
            aclError ret = aclrtFree(devBuf);
            if (ret != ACL_SUCCESS) {
                std::cerr << "[ERROR] aclrtFree(symmetric buffer) failed: " << static_cast<int>(ret) << std::endl;
                localOk = false;
            } else {
                devBuf = nullptr;
            }
        }
        if (stream != nullptr) {
            rtError_t ret = rtStreamDestroy(stream);
            if (ret != 0) {
                std::cerr << "[ERROR] rtStreamDestroy failed: " << ret << std::endl;
                localOk = false;
            }
            stream = nullptr;
        }
        const bool allOk = AllRanksReady(localOk, nRanks, "RDMA test resource cleanup");
        CommMpiBarrier();
        RdmaTrace(
            traceId, rankId, "CLEANUP end localOk=", localOk, " allOk=", allOk,
            " elapsed_us=", RdmaElapsedUs(cleanupStart));
        return allOk;
    }
};
#endif // PTO_RDMA_SUPPORTED

template <typename T>
static T RdmaInputValue(size_t index, int rankId)
{
    if constexpr (sizeof(T) == sizeof(uint8_t)) {
        return static_cast<T>(((index * 131U) ^ (index >> 8U) ^ (static_cast<size_t>(rankId) * 17U)) & 0xffU);
    }
    return static_cast<T>(index + static_cast<size_t>(rankId) * 10000U);
}

template <typename T>
static T RdmaSentinelValue(size_t index)
{
    if constexpr (sizeof(T) == sizeof(uint8_t)) {
        return static_cast<T>((0xa5U ^ (index * 29U) ^ (index >> 8U)) & 0xffU);
    }
    return static_cast<T>(-1);
}

static const char* RdmaCompletionModeName(RdmaCompletionMode mode)
{
    switch (mode) {
        case RdmaCompletionMode::STATUS_WAIT_EACH:
            return "status-wait-each";
        case RdmaCompletionMode::STATUS_WAIT_LAST:
            return "status-wait-last";
        case RdmaCompletionMode::PUBLIC_EVENT_WAIT_TEST:
            return "public-event-wait-test";
        default:
            return "unknown";
    }
}

#ifdef PTO_RDMA_SUPPORTED
static void PrintRdmaDeviceStatus(const char* operation, int rankId, int deviceId, int syncRet, uint32_t devStatus)
{
    std::cerr << "[RDMA][" << pto::comm::rdma::RdmaWorkspaceManager::ConfiguredBackendName() << "] " << operation
              << " Rank " << rankId << " Device " << deviceId << " SyncRet " << syncRet << " DevStatus 0x" << std::hex
              << devStatus << std::dec;
    if (devStatus == pto::comm::rdma::kRdmaSessionBuildError) {
        std::cerr << " (session_build_fail)";
    } else if (devStatus == kRdmaPublicEventWaitError) {
        std::cerr << " (public_event_wait_failed)";
    } else if (devStatus == kRdmaPublicEventTestError) {
        std::cerr << " (public_event_test_after_wait_failed)";
    } else if (devStatus != 0) {
        const char* backendStatus = pto::comm::rdma::test::DescribeBackendCompletionStatus(devStatus);
        std::cerr << " (" << (backendStatus == nullptr ? "cqe_syndrome_or_error" : backendStatus) << ')';
    }
    std::cerr << std::endl;
}

struct RdmaCaseConfig {
    int rankId;
    int nRanks;
    int nDevices;
    int firstDeviceId;
    int firstRankId;
    int rootRank;
    uint32_t caseId;
    int elemOffset;
    int elemCount;
    int operationCount;
    RdmaCompletionMode completionMode;
};

template <typename T>
struct RdmaHostStaging {
    T* input{nullptr};
    T* output{nullptr};

    bool Allocate(size_t inputElements, size_t outputElements, int nRanks)
    {
        bool localOk = aclrtMallocHost(reinterpret_cast<void**>(&input), inputElements * sizeof(T)) == ACL_SUCCESS &&
                       input != nullptr;
        localOk = (aclrtMallocHost(reinterpret_cast<void**>(&output), outputElements * sizeof(T)) == ACL_SUCCESS &&
                   output != nullptr) &&
                  localOk;
        if (!localOk) {
            std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        }
        if (AllRanksReady(localOk, nRanks, "host staging allocation")) {
            return true;
        }
        ReleaseLocal();
        return false;
    }

    void ReleaseLocal()
    {
        if (input != nullptr) {
            (void)aclrtFreeHost(input);
            input = nullptr;
        }
        if (output != nullptr) {
            (void)aclrtFreeHost(output);
            output = nullptr;
        }
    }

    bool Release(int nRanks)
    {
        bool localOk = true;
        if (input != nullptr) {
            localOk = aclrtFreeHost(input) == ACL_SUCCESS;
            input = nullptr;
        }
        if (output != nullptr) {
            localOk = (aclrtFreeHost(output) == ACL_SUCCESS) && localOk;
            output = nullptr;
        }
        return AllRanksReady(localOk, nRanks, "host staging release");
    }
};

struct RdmaKernelResult {
    int syncResult;
    uint32_t deviceStatus;
    bool copied;
};

template <typename T, size_t count>
static RdmaTestResult PrepareRdmaCase(
    const char* operation, const RdmaCaseConfig& config, size_t communicationBytes, RdmaTestContext& context)
{
    CommMpiBarrier();
    RdmaTrace(
        config.caseId, config.rankId, "CASE begin op=", operation, " elemSize=", sizeof(T), " count=", count,
        " bytes=", count * sizeof(T), " offset=", config.elemOffset, " elemsPerOp=", config.elemCount,
        " operations=", config.operationCount, " completion=", RdmaCompletionModeName(config.completionMode));
    const bool localPlanValid =
        config.elemOffset >= 0 && config.elemCount > 0 && config.operationCount > 0 &&
        static_cast<int64_t>(config.elemOffset) + static_cast<int64_t>(config.elemCount) * config.operationCount <=
            static_cast<int64_t>(count);
    const std::string validationStage = std::string(operation) + " transfer plan validation";
    if (!AllRanksReady(localPlanValid, config.nRanks, validationStage.c_str())) {
        return RdmaTestResult::FAILED;
    }

    const auto setupStart = std::chrono::steady_clock::now();
    const RdmaTestContext::SetupResult setup = context.Setup(
        config.caseId, config.rankId - config.firstRankId, config.nRanks, config.nDevices, config.firstDeviceId,
        communicationBytes);
    RdmaTrace(
        config.caseId, config.rankId, "CASE setup result=", static_cast<int>(setup),
        " elapsed_us=", RdmaElapsedUs(setupStart));
    if (setup == RdmaTestContext::SetupResult::READY) {
        return RdmaTestResult::PASSED;
    }
    const bool cleanupOk = context.Cleanup();
    return cleanupOk && setup == RdmaTestContext::SetupResult::SKIPPED ? RdmaTestResult::SKIPPED :
                                                                         RdmaTestResult::FAILED;
}

template <bool IsGet, typename T, size_t count>
static bool InitializeRdmaBuffers(
    const RdmaCaseConfig& config, RdmaTestContext& context, RdmaHostStaging<T>& staging, T*& sendBuffer, T*& recvBuffer)
{
    const size_t outputElements = IsGet ? static_cast<size_t>(config.nRanks) * count : count;
    if (!staging.Allocate(count, outputElements, config.nRanks)) {
        return false;
    }
    for (size_t index = 0; index < count; ++index) {
        staging.input[index] = RdmaInputValue<T>(index, config.rankId);
    }
    for (size_t index = 0; index < outputElements; ++index) {
        staging.output[index] = RdmaSentinelValue<T>(index % count);
    }

    constexpr size_t kDataOffset = 64 * sizeof(int32_t);
    sendBuffer = reinterpret_cast<T*>(static_cast<uint8_t*>(context.devBuf) + kDataOffset);
    recvBuffer = sendBuffer + count;
    bool localOk =
        aclrtMemcpy(sendBuffer, count * sizeof(T), staging.input, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE) ==
        ACL_SUCCESS;
    localOk = (aclrtMemcpy(
                   recvBuffer, outputElements * sizeof(T), staging.output, outputElements * sizeof(T),
                   ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS) &&
              localOk;
    const uint32_t zero = 0;
    localOk =
        (aclrtMemcpy(context.devBuf, sizeof(zero), &zero, sizeof(zero), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS) &&
        localOk;
    const char* stage = IsGet ? "GET input initialization" : "PUT input initialization";
    return AllRanksReady(localOk, config.nRanks, stage);
}

template <typename T>
static RdmaKernelResult CollectRdmaKernelResult(
    const char* operation, const RdmaCaseConfig& config, RdmaTestContext& context, int syncResult,
    const std::chrono::steady_clock::time_point& kernelStart, T* output, size_t outputElements, T* deviceOutput)
{
    CommMpiBarrier();
    RdmaTrace(
        config.caseId, config.rankId, "CASE ", operation, " kernel synchronized syncRet=", syncResult,
        " elapsed_us=", RdmaElapsedUs(kernelStart));
    uint32_t deviceStatus = 0;
    bool copied = aclrtMemcpy(
                      &deviceStatus, sizeof(deviceStatus), context.devBuf, sizeof(deviceStatus),
                      ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
    PrintRdmaDeviceStatus(operation, config.rankId, context.deviceId, syncResult, deviceStatus);
    copied = (aclrtMemcpy(
                  output, outputElements * sizeof(T), deviceOutput, outputElements * sizeof(T),
                  ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS) &&
             copied;
    return {syncResult, deviceStatus, copied};
}

template <typename T, size_t count>
static RdmaKernelResult LaunchPutRdmaKernel(
    const RdmaCaseConfig& config, RdmaTestContext& context, T* output, T* deviceOutput)
{
    CommMpiBarrier();
    const auto kernelStart = std::chrono::steady_clock::now();
    // clang-format off
    TPutAsyncRdmaKernelImpl<T, count><<<1, nullptr, context.stream>>>(
        reinterpret_cast<T*>(context.devBuf), config.nRanks, config.rankId, config.firstRankId, config.rootRank,
        config.elemOffset, config.elemCount, config.operationCount, config.completionMode,
        reinterpret_cast<uint8_t*>(context.rdmaMgr.GetWorkspaceAddr()), 0);
    // clang-format on
    const int syncResult = aclrtSynchronizeStream(context.stream);
    return CollectRdmaKernelResult("PUT", config, context, syncResult, kernelStart, output, count, deviceOutput);
}

#ifdef PTO_RDMA_GET_TEST
template <typename T, size_t count>
static RdmaKernelResult LaunchGetRdmaKernel(
    const RdmaCaseConfig& config, RdmaTestContext& context, T* output, size_t outputElements, T* deviceOutput)
{
    CommMpiBarrier();
    const auto kernelStart = std::chrono::steady_clock::now();
    // clang-format off
    TGetAsyncRdmaKernelImpl<T, count><<<1, nullptr, context.stream>>>(
        reinterpret_cast<T*>(context.devBuf), config.nRanks, config.rankId, config.firstRankId, config.rootRank,
        config.elemOffset, config.elemCount, config.operationCount, config.completionMode,
        reinterpret_cast<uint8_t*>(context.rdmaMgr.GetWorkspaceAddr()), 0);
    // clang-format on
    const int syncResult = aclrtSynchronizeStream(context.stream);
    return CollectRdmaKernelResult(
        "GET", config, context, syncResult, kernelStart, output, outputElements, deviceOutput);
}
#endif

template <typename T, size_t count>
static bool VerifyPutResult(
    const RdmaCaseConfig& config, const RdmaTestContext& context, const RdmaHostStaging<T>& staging,
    const RdmaKernelResult& kernelResult)
{
    bool valid = kernelResult.copied && kernelResult.syncResult == 0 && kernelResult.deviceStatus == 0;
    const size_t writeBegin = static_cast<size_t>(config.elemOffset);
    const size_t writeEnd = writeBegin + static_cast<size_t>(config.elemCount) * config.operationCount;
    for (size_t index = 0; index < count && valid; ++index) {
        const bool shouldReceive = config.rankId != config.rootRank && index >= writeBegin && index < writeEnd;
        const T expected = shouldReceive ? RdmaInputValue<T>(index, config.rootRank) : RdmaSentinelValue<T>(index);
        if (staging.output[index] != expected) {
            std::cerr << "PUT Rank " << config.rankId << " Device " << context.deviceId << " SyncRet "
                      << kernelResult.syncResult << " Index " << index << " Expected: " << static_cast<float>(expected)
                      << " Actual: " << static_cast<float>(staging.output[index]) << std::endl;
            valid = false;
        }
    }
    return AllRanksReady(valid, config.nRanks, "PUT result verification");
}

#ifdef PTO_RDMA_GET_TEST
template <typename T, size_t count>
static bool VerifyGetResult(
    const RdmaCaseConfig& config, const RdmaTestContext& context, const RdmaHostStaging<T>& staging,
    const RdmaKernelResult& kernelResult)
{
    bool valid = kernelResult.copied && kernelResult.syncResult == 0 && kernelResult.deviceStatus == 0;
    const size_t readBegin = static_cast<size_t>(config.elemOffset);
    const size_t readEnd = readBegin + static_cast<size_t>(config.elemCount) * config.operationCount;
    const int rootPeer = config.rootRank - config.firstRankId;
    for (int sourcePeer = 0; sourcePeer < config.nRanks && valid; ++sourcePeer) {
        for (size_t index = 0; index < count; ++index) {
            const bool shouldReceive =
                config.rankId == config.rootRank && sourcePeer != rootPeer && index >= readBegin && index < readEnd;
            const T expected =
                shouldReceive ? RdmaInputValue<T>(index, config.firstRankId + sourcePeer) : RdmaSentinelValue<T>(index);
            const T value = staging.output[static_cast<size_t>(sourcePeer) * count + index];
            if (value != expected) {
                std::cerr << "GET Rank " << config.rankId << " Device " << context.deviceId << " SourcePeer "
                          << sourcePeer << " Index " << index << " Expected: " << static_cast<float>(expected)
                          << " Actual: " << static_cast<float>(value) << std::endl;
                valid = false;
                break;
            }
        }
    }
    return AllRanksReady(valid, config.nRanks, "GET result verification");
}
#endif

template <typename T>
static RdmaTestResult FinishRdmaCase(
    const char* operation, const RdmaCaseConfig& config, RdmaTestContext& context, RdmaHostStaging<T>& staging,
    bool dataValid, const std::chrono::steady_clock::time_point& caseStart)
{
    dataValid = staging.Release(config.nRanks) && dataValid;
    const auto cleanupStart = std::chrono::steady_clock::now();
    const bool cleanupOk = context.Cleanup();
    RdmaTrace(
        config.caseId, config.rankId, "CASE end op=", operation, " dataOk=", dataValid, " cleanupOk=", cleanupOk,
        " cleanup_us=", RdmaElapsedUs(cleanupStart), " total_us=", RdmaElapsedUs(caseStart));
    return dataValid && cleanupOk ? RdmaTestResult::PASSED : RdmaTestResult::FAILED;
}

template <typename T>
static RdmaTestResult AbortRdmaCase(const RdmaCaseConfig& config, RdmaTestContext& context, RdmaHostStaging<T>& staging)
{
    // Initialization failures used to free local host staging directly before
    // the collectively ordered context cleanup. Preserve that sequence here.
    staging.ReleaseLocal();
    (void)context.Cleanup();
    return RdmaTestResult::FAILED;
}
#endif

template <typename... Args>
static RdmaTestResult SkipUnsupportedRdmaKernel(Args&&...)
{
    std::cerr << "[SKIP] built without PTO_RDMA_SUPPORTED" << std::endl;
    return RdmaTestResult::SKIPPED;
}

// ============================================================================
// Host-side runner.
// ============================================================================
template <bool IsGet, typename T, size_t count>
static RdmaTestResult RunAsyncRdmaRootKernel(
    int rank_id, int n_ranks, int n_devices, int first_device_id, int first_rank_id, int root_rank, uint32_t caseId,
    int elemOffset, int elemCount, int operationCount, RdmaCompletionMode completionMode)
{
#ifndef PTO_RDMA_SUPPORTED
    return SkipUnsupportedRdmaKernel(
        rank_id, n_ranks, n_devices, first_device_id, first_rank_id, root_rank, caseId, elemOffset, elemCount,
        operationCount, completionMode);
#else
    const auto caseStart = std::chrono::steady_clock::now();
    const char* operation = IsGet ? "GET" : "PUT";
    const RdmaCaseConfig config{rank_id, n_ranks,    n_devices, first_device_id, first_rank_id, root_rank,
                                caseId,  elemOffset, elemCount, operationCount,  completionMode};
    const size_t recvElements = IsGet ? static_cast<size_t>(n_ranks) * count : count;
    const size_t commBytesNeeded = 64 * sizeof(int32_t) + (recvElements + count) * sizeof(T);
    RdmaTestContext context;
    const RdmaTestResult preparation = PrepareRdmaCase<T, count>(operation, config, commBytesNeeded, context);
    if (preparation != RdmaTestResult::PASSED) {
        return preparation;
    }
    RdmaHostStaging<T> staging;
    T* sendBuffer = nullptr;
    T* recvBuffer = nullptr;
    if (!InitializeRdmaBuffers<IsGet, T, count>(config, context, staging, sendBuffer, recvBuffer)) {
        return AbortRdmaCase(config, context, staging);
    }
    if constexpr (IsGet) {
#ifdef PTO_RDMA_GET_TEST
        const RdmaKernelResult kernelResult =
            LaunchGetRdmaKernel<T, count>(config, context, staging.output, recvElements, recvBuffer);
        const bool dataValid = VerifyGetResult<T, count>(config, context, staging, kernelResult);
        return FinishRdmaCase(operation, config, context, staging, dataValid, caseStart);
#endif
    } else {
        const RdmaKernelResult kernelResult =
            LaunchPutRdmaKernel<T, count>(config, context, staging.output, recvBuffer);
        const bool dataValid = VerifyPutResult<T, count>(config, context, staging, kernelResult);
        return FinishRdmaCase(operation, config, context, staging, dataValid, caseStart);
    }
    return AbortRdmaCase(config, context, staging);
#endif // PTO_RDMA_SUPPORTED
}

template <typename T, size_t count>
RdmaTestResult RunPutAsyncRdmaRootPutKernel(
    int rank_id, int n_ranks, int n_devices, int first_device_id, int first_rank_id, int root_rank, uint32_t caseId,
    int elemOffset, int elemCount, int operationCount, RdmaCompletionMode completionMode)
{
    return RunAsyncRdmaRootKernel<false, T, count>(
        rank_id, n_ranks, n_devices, first_device_id, first_rank_id, root_rank, caseId, elemOffset, elemCount,
        operationCount, completionMode);
}

static RdmaTestResult RunPutAsyncNotifyRdmaSetKernel(
    int rankId, int nRanks, int nDevices, int firstDeviceId, int firstRankId, int rootRank, uint32_t caseId)
{
#ifndef PTO_RDMA_SUPPORTED
    return SkipUnsupportedRdmaKernel(rankId, nRanks, nDevices, firstDeviceId, firstRankId, rootRank, caseId);
#else
    constexpr size_t kCount = 256U;
    const auto caseStart = std::chrono::steady_clock::now();
    const RdmaCaseConfig config{
        rankId,
        nRanks,
        nDevices,
        firstDeviceId,
        firstRankId,
        rootRank,
        caseId,
        0,
        static_cast<int>(kCount),
        1,
        RdmaCompletionMode::PUBLIC_EVENT_WAIT_TEST};
    const size_t communicationBytes = kRdmaTestDataOffset + 2U * kCount * sizeof(int32_t);
    RdmaTestContext context;
    const RdmaTestResult preparation =
        PrepareRdmaCase<int32_t, kCount>("PUT_NOTIFY", config, communicationBytes, context);
    if (preparation != RdmaTestResult::PASSED) {
        return preparation;
    }

    RdmaHostStaging<int32_t> staging;
    int32_t* sendBuffer = nullptr;
    int32_t* receiveBuffer = nullptr;
    if (!InitializeRdmaBuffers<false, int32_t, kCount>(config, context, staging, sendBuffer, receiveBuffer)) {
        return AbortRdmaCase(config, context, staging);
    }

    auto* base = static_cast<uint8_t*>(context.devBuf);
    const int32_t initialSignal = 0;
    bool headerReady =
        aclrtMemcpy(
            base + kRdmaNotifySignalOffset, sizeof(initialSignal), &initialSignal, sizeof(initialSignal),
            ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
        aclrtMemcpy(
            base + kRdmaNotifyCanaryBeforeOffset, sizeof(kRdmaNotifyCanaryBefore), &kRdmaNotifyCanaryBefore,
            sizeof(kRdmaNotifyCanaryBefore), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
        aclrtMemcpy(
            base + kRdmaNotifyCanaryAfterOffset, sizeof(kRdmaNotifyCanaryAfter), &kRdmaNotifyCanaryAfter,
            sizeof(kRdmaNotifyCanaryAfter), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
    if (!AllRanksReady(headerReady, nRanks, "RDMA notify header initialization")) {
        return AbortRdmaCase(config, context, staging);
    }

    CommMpiBarrier();
    const auto kernelStart = std::chrono::steady_clock::now();
    // clang-format off
    TPutAsyncNotifyRdmaKernelImpl<kCount><<<1, nullptr, context.stream>>>(
        reinterpret_cast<int32_t*>(context.devBuf), rankId, firstRankId, rootRank,
        reinterpret_cast<uint8_t*>(context.rdmaMgr.GetWorkspaceAddr()), 0U);
    // clang-format on
    const int synchronizeResult = aclrtSynchronizeStream(context.stream);
    const RdmaKernelResult kernelResult = CollectRdmaKernelResult(
        "PUT_NOTIFY", config, context, synchronizeResult, kernelStart, staging.output, kCount, receiveBuffer);
    bool dataValid = VerifyPutResult<int32_t, kCount>(config, context, staging, kernelResult);

    int32_t signal = 0;
    int32_t canaryBefore = 0;
    int32_t canaryAfter = 0;
    bool notifyValid = aclrtMemcpy(
                           &signal, sizeof(signal), base + kRdmaNotifySignalOffset, sizeof(signal),
                           ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS &&
                       aclrtMemcpy(
                           &canaryBefore, sizeof(canaryBefore), base + kRdmaNotifyCanaryBeforeOffset,
                           sizeof(canaryBefore), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS &&
                       aclrtMemcpy(
                           &canaryAfter, sizeof(canaryAfter), base + kRdmaNotifyCanaryAfterOffset, sizeof(canaryAfter),
                           ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
    const int32_t expectedSignal = rankId == rootRank ? initialSignal : kRdmaNotifySignalValue;
    notifyValid = notifyValid && signal == expectedSignal && canaryBefore == kRdmaNotifyCanaryBefore &&
                  canaryAfter == kRdmaNotifyCanaryAfter;
    if (!notifyValid) {
        std::cerr << "RDMA notify Rank " << rankId << " Device " << context.deviceId << " Signal " << signal
                  << " Expected " << expectedSignal << " CanaryBefore " << canaryBefore << " CanaryAfter "
                  << canaryAfter << std::endl;
    }
    dataValid = AllRanksReady(notifyValid, nRanks, "RDMA notify signal verification") && dataValid;
    return FinishRdmaCase("PUT_NOTIFY", config, context, staging, dataValid, caseStart);
#endif
}

#ifdef PTO_RDMA_GET_TEST
template <typename T, size_t count>
RdmaTestResult RunGetAsyncRdmaRootGetKernel(
    int rank_id, int n_ranks, int n_devices, int first_device_id, int first_rank_id, int root_rank, uint32_t caseId,
    int elemOffset, int elemCount, int operationCount, RdmaCompletionMode completionMode)
{
    return RunAsyncRdmaRootKernel<true, T, count>(
        rank_id, n_ranks, n_devices, first_device_id, first_rank_id, root_rank, caseId, elemOffset, elemCount,
        operationCount, completionMode);
}
#endif // PTO_RDMA_GET_TEST

// ============================================================================
// MPI-based multi-rank launch with explicit pass/fail/skip propagation.
// ============================================================================
struct RdmaLaunchPreparation {
    RdmaTestResult result;
    int mpiRank;
};

static bool ValidateRdmaLaunchConfiguration(int nRanks, int nDevices, int mpiRank, int mpiSize)
{
    const bool valid = nRanks > 0 && nDevices > 0 && mpiSize == nRanks;
    if (!valid && mpiRank == 0) {
        std::cerr << "[ERROR] invalid launch configuration: mpiSize=" << mpiSize << " nRanks=" << nRanks
                  << " nDevices=" << nDevices << std::endl;
    }
    return valid;
}

#ifdef PTO_RDMA_SUPPORTED
static RdmaTestResult CheckRdmaDeviceAvailability(int nRanks, int nDevices, int firstDeviceId, int mpiRank)
{
    const int deviceCount = GetAvailableDeviceCount();
    const bool localDevicesReady = deviceCount >= nDevices + firstDeviceId;
    bool anyDeviceMissing = false;
    if (AllRanksReady(localDevicesReady, nRanks, "device availability", &anyDeviceMissing)) {
        return RdmaTestResult::PASSED;
    }
    if (mpiRank == 0) {
        std::cerr << "[SKIP] Need " << (nDevices + firstDeviceId) << " NPU(s), have " << deviceCount << std::endl;
    }
    return anyDeviceMissing ? RdmaTestResult::SKIPPED : RdmaTestResult::FAILED;
}

static bool InitializeAclForRdma(int nRanks)
{
    constexpr int kAclRepeatInit = 100002;
    const aclError aclRet = aclInit(nullptr);
    const bool aclReady = aclRet == ACL_SUCCESS || static_cast<int>(aclRet) == kAclRepeatInit;
    if (!aclReady) {
        std::cerr << "[ERROR] aclInit failed: " << static_cast<int>(aclRet) << std::endl;
    }
    return AllRanksReady(aclReady, nRanks, "ACL initialization");
}
#endif

static RdmaLaunchPreparation PrepareRdmaLaunch(int nRanks, int nDevices, int firstDeviceId)
{
    const int mpiRank = CommMpiRank();
    if (!ValidateRdmaLaunchConfiguration(nRanks, nDevices, mpiRank, CommMpiSize())) {
        return {RdmaTestResult::FAILED, mpiRank};
    }
#ifndef PTO_RDMA_SUPPORTED
    (void)firstDeviceId;
    if (mpiRank == 0) {
        std::cerr << "[SKIP] built without PTO_RDMA_SUPPORTED" << std::endl;
    }
    return {RdmaTestResult::SKIPPED, mpiRank};
#else
    const auto preflight = AgreeOnRdmaPreflight(nRanks);
    if (preflight != pto::comm::rdma::WorkspaceInitResult::READY) {
        const RdmaTestResult result = preflight == pto::comm::rdma::WorkspaceInitResult::DISABLED ?
                                          RdmaTestResult::SKIPPED :
                                          RdmaTestResult::FAILED;
        return {result, mpiRank};
    }
    const RdmaTestResult deviceResult = CheckRdmaDeviceAvailability(nRanks, nDevices, firstDeviceId, mpiRank);
    if (deviceResult != RdmaTestResult::PASSED) {
        return {deviceResult, mpiRank};
    }
    return {InitializeAclForRdma(nRanks) ? RdmaTestResult::PASSED : RdmaTestResult::FAILED, mpiRank};
#endif
}

template <typename T, size_t count>
RdmaTestResult RunPutAsyncRdmaRootPut(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunPutAsyncRdmaRootPutPlan<T, count>(
        n_ranks, n_devices, first_rank_id, first_device_id, 0, static_cast<int>(count), 1,
        RdmaCompletionMode::STATUS_WAIT_EACH);
}

template <bool IsGet, typename T, size_t count>
static RdmaTestResult RunAsyncRdmaPlan(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, int elem_offset, int elem_count,
    int operation_count, RdmaCompletionMode completion_mode)
{
    const RdmaLaunchPreparation preparation = PrepareRdmaLaunch(n_ranks, n_devices, first_device_id);
    if (preparation.result != RdmaTestResult::PASSED) {
        return preparation.result;
    }
#ifdef PTO_RDMA_SUPPORTED
    const int rankId = first_rank_id + preparation.mpiRank;
    const int rootRank = first_rank_id;
    const uint32_t caseId = ++gRdmaCaseSequence;
    if constexpr (IsGet) {
#ifdef PTO_RDMA_GET_TEST
        return RunGetAsyncRdmaRootGetKernel<T, count>(
            rankId, n_ranks, n_devices, first_device_id, first_rank_id, rootRank, caseId, elem_offset, elem_count,
            operation_count, completion_mode);
#endif
    } else {
        return RunPutAsyncRdmaRootPutKernel<T, count>(
            rankId, n_ranks, n_devices, first_device_id, first_rank_id, rootRank, caseId, elem_offset, elem_count,
            operation_count, completion_mode);
    }
    return RdmaTestResult::FAILED;
#else
    return RdmaTestResult::SKIPPED;
#endif
}

RdmaTestResult RunPutAsyncNotifyRdmaSet(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    const RdmaLaunchPreparation preparation = PrepareRdmaLaunch(n_ranks, n_devices, first_device_id);
    if (preparation.result != RdmaTestResult::PASSED) {
        return preparation.result;
    }
#ifdef PTO_RDMA_SUPPORTED
    const int rankId = first_rank_id + preparation.mpiRank;
    const int rootRank = first_rank_id;
    const uint32_t caseId = ++gRdmaCaseSequence;
    return RunPutAsyncNotifyRdmaSetKernel(rankId, n_ranks, n_devices, first_device_id, first_rank_id, rootRank, caseId);
#else
    return RdmaTestResult::SKIPPED;
#endif
}

template <typename T, size_t count>
RdmaTestResult RunPutAsyncRdmaRootPutPlan(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, int elem_offset, int elem_count,
    int operation_count, RdmaCompletionMode completion_mode)
{
    return RunAsyncRdmaPlan<false, T, count>(
        n_ranks, n_devices, first_rank_id, first_device_id, elem_offset, elem_count, operation_count, completion_mode);
}

#ifdef PTO_RDMA_GET_TEST
template <typename T, size_t count>
RdmaTestResult RunGetAsyncRdmaRootGetPlan(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, int elem_offset, int elem_count,
    int operation_count, RdmaCompletionMode completion_mode)
{
    return RunAsyncRdmaPlan<true, T, count>(
        n_ranks, n_devices, first_rank_id, first_device_id, elem_offset, elem_count, operation_count, completion_mode);
}
#endif // PTO_RDMA_GET_TEST

// Explicit instantiations
template RdmaTestResult RunPutAsyncRdmaRootPut<float, 256>(int, int, int, int);
template RdmaTestResult RunPutAsyncRdmaRootPut<int32_t, 4096>(int, int, int, int);
template RdmaTestResult RunPutAsyncRdmaRootPut<uint8_t, 512>(int, int, int, int);
template RdmaTestResult RunPutAsyncRdmaRootPut<uint8_t, 64>(int, int, int, int);
template RdmaTestResult RunPutAsyncRdmaRootPut<float, 64>(int, int, int, int);
template RdmaTestResult RunPutAsyncRdmaRootPut<float, 524288>(int, int, int, int);

template RdmaTestResult RunPutAsyncRdmaRootPutPlan<float, 256>(int, int, int, int, int, int, int, RdmaCompletionMode);
template RdmaTestResult RunPutAsyncRdmaRootPutPlan<int32_t, 4096>(
    int, int, int, int, int, int, int, RdmaCompletionMode);
template RdmaTestResult RunPutAsyncRdmaRootPutPlan<uint8_t, 512>(int, int, int, int, int, int, int, RdmaCompletionMode);
template RdmaTestResult RunPutAsyncRdmaRootPutPlan<uint8_t, 64>(int, int, int, int, int, int, int, RdmaCompletionMode);
template RdmaTestResult RunPutAsyncRdmaRootPutPlan<float, 64>(int, int, int, int, int, int, int, RdmaCompletionMode);
template RdmaTestResult RunPutAsyncRdmaRootPutPlan<float, 524288>(
    int, int, int, int, int, int, int, RdmaCompletionMode);

#ifdef PTO_RDMA_GET_TEST
template RdmaTestResult RunGetAsyncRdmaRootGetPlan<float, 256>(int, int, int, int, int, int, int, RdmaCompletionMode);
template RdmaTestResult RunGetAsyncRdmaRootGetPlan<int32_t, 4096>(
    int, int, int, int, int, int, int, RdmaCompletionMode);
template RdmaTestResult RunGetAsyncRdmaRootGetPlan<uint8_t, 512>(int, int, int, int, int, int, int, RdmaCompletionMode);
template RdmaTestResult RunGetAsyncRdmaRootGetPlan<uint8_t, 64>(int, int, int, int, int, int, int, RdmaCompletionMode);
template RdmaTestResult RunGetAsyncRdmaRootGetPlan<float, 64>(int, int, int, int, int, int, int, RdmaCompletionMode);
template RdmaTestResult RunGetAsyncRdmaRootGetPlan<float, 524288>(
    int, int, int, int, int, int, int, RdmaCompletionMode);
#endif // PTO_RDMA_GET_TEST
