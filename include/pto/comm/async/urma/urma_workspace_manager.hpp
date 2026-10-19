/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_WORKSPACE_MANAGER_HPP
#define PTO_COMM_ASYNC_URMA_WORKSPACE_MANAGER_HPP

#if defined(__CCE_KT_TEST__)
#error "urma_workspace_manager.hpp is a host-only header and cannot be included in device code."
#endif

#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "securec.h"

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_rank_graph.h"

#include "pto/comm/async/urma/urma_types.hpp"
#include "pto/comm/async/urma/urma_hccl_defs.hpp"
#include "pto/comm/async/urma/urma_channel_helper.hpp"

namespace pto {
namespace comm {
namespace urma {

// ============================================================================
// UrmaWorkspaceManager: HCCL-based URMA workspace initialization.
//
// Uses HCCL public APIs for connection establishment (HcclCommMemReg,
// HcclRankGraphGetLinks, HcclChannelAcquire), then reads ChannelEntity from
// device/host and fills UrmaInfo for AIV-side urma_async_intrin.hpp.
// Flow:
//   1. Allocate the notify pool and register symmetric + notify memory
//   2. HcclRankGraphGetLinks (find UBC_TP/CTP links per peer)
//   3. HcclChannelAcquire(COMM_ENGINE_AIV) — returns device ChannelEntity* handles
//      (requires CANN with ConvertAivChannelHandlesToDevicePtrs in HcclChannelAcquire)
//   4. Read back ChannelEntity + sub-structures from device
//   5. Convert to UrmaInfo format and copy to device
// ============================================================================
class UrmaWorkspaceManager {
public:
    UrmaWorkspaceManager() = default;
    ~UrmaWorkspaceManager() { Finalize(); }

    UrmaWorkspaceManager(const UrmaWorkspaceManager&) = delete;
    UrmaWorkspaceManager& operator=(const UrmaWorkspaceManager&) = delete;

    bool Init(HcclComm comm, uint32_t rankId, uint32_t rankCount, void* symmetricAddr, uint64_t symmetricSize)
    {
        comm_ = comm;
        rankId_ = rankId;
        rankCount_ = rankCount;
        symmetricAddr_ = symmetricAddr;
        symmetricSize_ = symmetricSize;

        if (!AllocateNotifyPool()) {
            return false;
        }
        if (!RegisterMemory()) {
            return false;
        }
        if (!BuildChannels()) {
            return false;
        }
        if (!ExtractAndFillUrmaInfo()) {
            return false;
        }

        initialized_ = true;
        return true;
    }

    // The owning communication context must stop kernels, drain QPs, and
    // destroy HCCL communication resources before calling Finalize().
    void Finalize()
    {
        FreeDeviceAddr(urmaInfoDevice_);
        FreeDeviceAddr(eidDevice_);
        FreeDeviceAddr(notifyPoolDevice_);
        channelHandles_.clear();
        peerBaseAddrs_.clear();
        initialized_ = false;
    }

    void* GetWorkspaceAddr() const { return urmaInfoDevice_; }

    // Per-peer symmetric MR base address (self = symmetricAddr_). Valid after Init().
    uint64_t PeerBaseAddr(uint32_t peer) const
    {
        if (peer >= peerBaseAddrs_.size()) {
            return 0;
        }
        return peerBaseAddrs_[peer];
    }

private:
    bool AllocateNotifyPool()
    {
        const uint64_t regionCount = static_cast<uint64_t>(rankCount_) * kUrmaQpNum;
        if (regionCount == 0U ||
            regionCount > std::numeric_limits<uint64_t>::max() / sizeof(UrmaNotifyResourceRegion)) {
            std::cerr << "[URMA] invalid notify pool region count=" << regionCount << std::endl;
            return false;
        }
        notifyPoolSize_ = regionCount * sizeof(UrmaNotifyResourceRegion);
        aclError err = aclrtMalloc(&notifyPoolDevice_, notifyPoolSize_, ACL_MEM_MALLOC_HUGE_FIRST);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMalloc(notifyPool) failed: " << err << " size=" << notifyPoolSize_ << std::endl;
            return false;
        }
        err = aclrtMemset(notifyPoolDevice_, notifyPoolSize_, 0, notifyPoolSize_);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMemset(notifyPool) failed: " << err << std::endl;
            return false;
        }
        return true;
    }

    bool RegisterMemory()
    {
        CommMem mem{};
        mem.type = COMM_MEM_TYPE_DEVICE;
        mem.addr = symmetricAddr_;
        mem.size = symmetricSize_;

        HcclResult ret = HcclCommMemReg(comm_, kUrmaSymMemTag, &mem, &memHandle_);
        if (ret != HCCL_SUCCESS) {
            std::cerr << "[URMA] HcclCommMemReg failed: " << static_cast<int>(ret) << std::endl;
            return false;
        }

        CommMem notifyMem{};
        notifyMem.type = COMM_MEM_TYPE_DEVICE;
        notifyMem.addr = notifyPoolDevice_;
        notifyMem.size = notifyPoolSize_;
        ret = HcclCommMemReg(comm_, kUrmaNotifyMemTag, &notifyMem, &notifyMemHandle_);
        if (ret != HCCL_SUCCESS) {
            std::cerr << "[URMA] HcclCommMemReg(notifyPool) failed: " << static_cast<int>(ret) << std::endl;
            return false;
        }
        memHandles_[0] = memHandle_;
        memHandles_[1] = notifyMemHandle_;
        return true;
    }

    bool BuildChannelDesc(uint32_t peer, HcclChannelDesc& desc)
    {
        uint32_t linkNum = 0;
        CommLink* linkList = nullptr;
        HcclResult rc = HcclRankGraphGetLinks(comm_, 0, rankId_, peer, &linkList, &linkNum);
        if (rc != HCCL_SUCCESS) {
            std::cerr << "[URMA] HcclRankGraphGetLinks peer=" << peer << " ret=" << static_cast<int>(rc) << std::endl;
            return false;
        }
        for (uint32_t i = 0; i < linkNum; ++i) {
            CommProtocol proto = linkList[i].linkAttr.linkProtocol;
            if (proto != kCommProtocolUbcCtp && proto != kCommProtocolUbcTp) {
                continue;
            }
            HcclChannelDescInit(&desc, 1);
            desc.remoteRank = peer;
            desc.notifyNum = 0;
            desc.channelProtocol = proto;
            desc.localEndpoint = linkList[i].srcEndpointDesc;
            desc.remoteEndpoint = linkList[i].dstEndpointDesc;
            desc.memHandles = memHandles_;
            desc.memHandleNum = kUrmaRegisteredMemCount;
            return true;
        }
        std::cerr << "[URMA] rank=" << rankId_ << " no UBC_TP/CTP link to peer=" << peer << std::endl;
        return false;
    }

    bool BuildChannels()
    {
        std::vector<HcclChannelDesc> descs;
        descs.reserve(rankCount_ - 1);
        for (uint32_t peer = 0; peer < rankCount_; ++peer) {
            if (peer == rankId_) {
                continue;
            }
            HcclChannelDesc desc{};
            if (!BuildChannelDesc(peer, desc)) {
                return false;
            }
            descs.push_back(desc);
        }
        channelHandles_.resize(descs.size());
        HcclResult rc = HcclChannelAcquire(
            comm_, COMM_ENGINE_AIV, descs.data(), static_cast<uint32_t>(descs.size()), channelHandles_.data());
        if (rc != HCCL_SUCCESS) {
            std::cerr << "[URMA] HcclChannelAcquire failed: " << static_cast<int>(rc) << std::endl;
            return false;
        }
        return true;
    }

    struct UrmaPeerInfoTables {
        explicit UrmaPeerInfoTables(uint32_t rankCount)
            : wqList(rankCount), cqList(rankCount), memList(rankCount), eidTable(rankCount * kUrmaEidBytes, 0)
        {}

        std::vector<UrmaWQCtx> wqList;
        std::vector<UrmaCqCtx> cqList;
        std::vector<UrmaMemInfo> memList;
        std::vector<uint8_t> eidTable;
    };

    struct UrmaRegistrationTokenState {
        uint32_t localTokenId{0};
        uint32_t notifyPoolTokenId{0};
        bool notifyPoolTokenInitialized{false};
    };

    bool ExtractAndFillUrmaInfo()
    {
        UrmaPeerInfoTables urmaPeerInfoTables(rankCount_);
        UrmaRegistrationTokenState urmaRegistrationTokenState{};

        if (!ExtractPerPeerInfo(urmaPeerInfoTables, urmaRegistrationTokenState)) {
            return false;
        }
        if (!InitializeAsyncQueueState(urmaPeerInfoTables.wqList, urmaPeerInfoTables.cqList)) {
            return false;
        }
        peerBaseAddrs_.resize(rankCount_);
        for (uint32_t peer = 0; peer < rankCount_; ++peer) {
            peerBaseAddrs_[peer] = urmaPeerInfoTables.memList[peer].addr;
        }
        if (!AllocAndCopyEidTable(urmaPeerInfoTables.eidTable, urmaPeerInfoTables.memList)) {
            return false;
        }
        if (!BuildAndCopyUrmaInfoTable(urmaPeerInfoTables, urmaRegistrationTokenState)) {
            return false;
        }

        std::cerr << "[URMA] UrmaInfo OK rank=" << rankId_ << " localTokenId=0x" << std::hex
                  << urmaRegistrationTokenState.localTokenId << std::dec << " notifyPoolTokenId=0x" << std::hex
                  << urmaRegistrationTokenState.notifyPoolTokenId << std::dec << " notifyPoolBytes=" << notifyPoolSize_
                  << std::endl;
        return true;
    }

    bool InitializeAsyncQueueState(std::vector<UrmaWQCtx>& wqList, const std::vector<UrmaCqCtx>& cqList)
    {
        for (uint32_t peer = 0; peer < rankCount_; ++peer) {
            if (peer == rankId_) {
                continue;
            }
            uint32_t sqHead = 0U;
            uint32_t sqTail = 0U;
            uint32_t cqTail = 0U;
            if (aclrtMemcpy(
                    &sqHead, sizeof(sqHead), reinterpret_cast<void*>(wqList[peer].headAddr), sizeof(sqHead),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
                aclrtMemcpy(
                    &sqTail, sizeof(sqTail), reinterpret_cast<void*>(wqList[peer].tailAddr), sizeof(sqTail),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
                aclrtMemcpy(
                    &cqTail, sizeof(cqTail), reinterpret_cast<void*>(cqList[peer].tailAddr), sizeof(cqTail),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
                sqHead != sqTail) {
                std::cerr << "[URMA] peer=" << peer << " async queue initialization requires an empty SQ" << std::endl;
                return false;
            }
            wqList[peer].submittedWqeCount = cqTail;
        }
        return true;
    }

    bool ExtractPerPeerInfo(
        UrmaPeerInfoTables& urmaPeerInfoTables, UrmaRegistrationTokenState& urmaRegistrationTokenState)
    {
        uint32_t channelIdx = 0;
        for (uint32_t peer = 0; peer < rankCount_; ++peer) {
            if (peer == rankId_) {
                urmaPeerInfoTables.wqList[peer] = UrmaWQCtx{};
                urmaPeerInfoTables.cqList[peer] = UrmaCqCtx{};
                urmaPeerInfoTables.memList[peer] = UrmaMemInfo{};
                urmaPeerInfoTables.memList[peer].addr = reinterpret_cast<uint64_t>(symmetricAddr_);
                urmaPeerInfoTables.memList[peer].len = static_cast<uint32_t>(symmetricSize_);
                continue;
            }
            if (!ExtractSinglePeer(peer, channelIdx, urmaPeerInfoTables, urmaRegistrationTokenState)) {
                return false;
            }
            ++channelIdx;
        }
        return true;
    }

    struct PeerChannelState {
        ChannelHandle handle{};
        ChannelEntity entity{};
        SqContext sq{};
        CqContext cq{};
    };

    bool ReadPeerChannel(uint32_t peer, uint32_t channelIdx, PeerChannelState& state)
    {
        state.handle = channelHandles_[channelIdx];
        if (state.handle != 0 && static_cast<uint64_t>(state.handle) < kDeviceVaThreshold) {
            std::cerr << "[URMA] ChannelHandle looks like host pointer (0x" << std::hex << state.handle << std::dec
                      << ") for peer=" << peer
                      << ". Upgrade CANN: HcclChannelAcquire must return device ChannelEntity pointers for AIV URMA."
                      << std::endl;
            return false;
        }
        RegedBufferEntity remoteBuf{};
        RegedBufferEntity localBuf{};
        if (state.handle == 0 || !UrmaChannelHelper::TryReadChannelEntity(
                                     state.handle, peer, state.entity, state.sq, state.cq, remoteBuf, localBuf)) {
            std::cerr << "[URMA] Cannot read ChannelEntity for peer=" << peer << " handle=0x" << std::hex
                      << static_cast<uint64_t>(state.handle) << std::dec << std::endl;
            return false;
        }
        return true;
    }

    bool ExtractPeerRegistrations(
        uint32_t peer, const PeerChannelState& state, RegedBufferEntity& symRemoteBuf, uint64_t& symRmaAddr,
        uint32_t& symRmaSize, UrmaRegistrationTokenState& urmaRegistrationTokenState)
    {
        if (!UrmaChannelHelper::SelectSymmetricRemoteBuffer(
                comm_, kUrmaSymMemTag, symmetricSize_, state.handle, peer, state.entity, symRemoteBuf, symRmaAddr,
                symRmaSize)) {
            return false;
        }
        RegedBufferEntity symLocalBuf{};
        if (!UrmaChannelHelper::FindLocalRmaRegistration(
                reinterpret_cast<uint64_t>(symmetricAddr_), symmetricSize_, state.entity, peer, symLocalBuf)) {
            std::cerr << "[URMA] peer=" << peer << " no local symmetric registration found" << std::endl;
            return false;
        }
        urmaRegistrationTokenState.localTokenId = symLocalBuf.bufferInfo.rma.protectionInfo.memInfo.ub.tokenId;
        RegedBufferEntity notifyLocalBuf{};
        if (!UrmaChannelHelper::FindLocalRmaRegistration(
                reinterpret_cast<uint64_t>(notifyPoolDevice_), notifyPoolSize_, state.entity, peer, notifyLocalBuf)) {
            std::cerr << "[URMA] peer=" << peer << " no local notify pool registration found" << std::endl;
            return false;
        }
        const uint32_t peerNotifyTokenId = notifyLocalBuf.bufferInfo.rma.protectionInfo.memInfo.ub.tokenId;
        if (urmaRegistrationTokenState.notifyPoolTokenInitialized &&
            urmaRegistrationTokenState.notifyPoolTokenId != peerNotifyTokenId) {
            std::cerr << "[URMA] inconsistent notify pool token peer=" << peer << " expected=0x" << std::hex
                      << urmaRegistrationTokenState.notifyPoolTokenId << " actual=0x" << peerNotifyTokenId << std::dec
                      << std::endl;
            return false;
        }
        urmaRegistrationTokenState.notifyPoolTokenId = peerNotifyTokenId;
        urmaRegistrationTokenState.notifyPoolTokenInitialized = true;
        return true;
    }

    static void LogPeerInfo(uint32_t peer, const UrmaWQCtx& wq, const UrmaMemInfo& mem)
    {
        std::cerr << "[URMA] peer=" << peer << " tpId=" << mem.tpn << " rmtAddr=0x" << std::hex << mem.addr
                  << " sqVa=0x" << wq.bufAddr << " dbAddr=0x" << wq.dbAddr << std::dec << std::endl;
    }

    bool ExtractSinglePeer(
        uint32_t peer, uint32_t channelIdx, UrmaPeerInfoTables& urmaPeerInfoTables,
        UrmaRegistrationTokenState& urmaRegistrationTokenState)
    {
        PeerChannelState state{};
        RegedBufferEntity symRemoteBuf{};
        uint64_t symRmaAddr = 0;
        uint32_t symRmaSize = 0;
        if (!ReadPeerChannel(peer, channelIdx, state) ||
            !ExtractPeerRegistrations(peer, state, symRemoteBuf, symRmaAddr, symRmaSize, urmaRegistrationTokenState)) {
            return false;
        }
        FillWqCtx(urmaPeerInfoTables.wqList[peer], state.sq);
        FillCqCtx(urmaPeerInfoTables.cqList[peer], state.cq);
        FillMemInfo(urmaPeerInfoTables.memList[peer], state.sq, symRemoteBuf, symRmaAddr, symRmaSize);
        (void)memcpy_s(
            &urmaPeerInfoTables.eidTable[peer * kUrmaEidBytes], kUrmaEidBytes, state.sq.contextInfo.ubJfs.remoteEID,
            kUrmaEidBytes);
        LogPeerInfo(peer, urmaPeerInfoTables.wqList[peer], urmaPeerInfoTables.memList[peer]);
        return true;
    }

    static void FillWqCtx(UrmaWQCtx& wq, const SqContext& sq)
    {
        wq.wqn = sq.contextInfo.ubJfs.jfsID;
        wq.bufAddr = sq.contextInfo.ubJfs.sqVa;
        wq.wqeShiftSize = Log2U32(sq.contextInfo.ubJfs.wqeSize);
        wq.depth = sq.contextInfo.ubJfs.sqDepth;
        wq.headAddr = sq.contextInfo.ubJfs.headAddr;
        wq.tailAddr = sq.contextInfo.ubJfs.tailAddr;
        wq.dbMode = UrmaDbMode::SW_DB;
        wq.dbAddr = sq.contextInfo.ubJfs.dbVa;
        wq.sl = 0;
    }

    static void FillCqCtx(UrmaCqCtx& cqCtx, const CqContext& cq)
    {
        cqCtx.cqn = cq.contextInfo.ubJfc.jfcID;
        cqCtx.bufAddr = cq.contextInfo.ubJfc.scqVa;
        cqCtx.cqeShiftSize = Log2U32(cq.contextInfo.ubJfc.cqeSize);
        cqCtx.depth = cq.contextInfo.ubJfc.cqDepth;
        cqCtx.headAddr = cq.contextInfo.ubJfc.headAddr;
        cqCtx.tailAddr = cq.contextInfo.ubJfc.tailAddr;
        cqCtx.dbMode = UrmaDbMode::SW_DB;
        cqCtx.dbAddr = cq.contextInfo.ubJfc.dbVa;
    }

    static void FillMemInfo(
        UrmaMemInfo& mem, const SqContext& sq, const RegedBufferEntity& symRemoteBuf, uint64_t symRmaAddr,
        uint32_t symRmaSize)
    {
        mem.tokenValueValid = true;
        mem.rmtJettyType = 1;
        mem.targetHint = 0;
        mem.tpn = sq.contextInfo.ubJfs.tpID;
        mem.tid = symRemoteBuf.bufferInfo.rma.protectionInfo.memInfo.ub.tokenId;
        mem.rmtTokenValue = symRemoteBuf.bufferInfo.rma.protectionInfo.memInfo.ub.tokenValue;
        mem.len = symRmaSize;
        mem.addr = symRmaAddr;
    }

    bool AllocAndCopyEidTable(const std::vector<uint8_t>& eidTable, std::vector<UrmaMemInfo>& memList)
    {
        size_t eidDevSize = rankCount_ * kUrmaEidBytes;
        aclError err = aclrtMalloc(&eidDevice_, eidDevSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMalloc(eidTable) failed: " << err << std::endl;
            return false;
        }
        err = aclrtMemcpy(eidDevice_, eidDevSize, eidTable.data(), eidDevSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMemcpy(eidTable) failed: " << err << std::endl;
            return false;
        }
        for (uint32_t peer = 0; peer < rankCount_; ++peer) {
            memList[peer].eidAddr =
                reinterpret_cast<uint64_t>(static_cast<uint8_t*>(eidDevice_) + peer * kUrmaEidBytes);
        }
        return true;
    }

    bool BuildAndCopyUrmaInfoTable(
        const UrmaPeerInfoTables& urmaPeerInfoTables, const UrmaRegistrationTokenState& urmaRegistrationTokenState)
    {
        size_t totalSize =
            sizeof(UrmaInfo) + rankCount_ * (2U * sizeof(UrmaWQCtx) * kUrmaQpNum + 2U * sizeof(UrmaCqCtx) * kUrmaQpNum +
                                             sizeof(UrmaMemInfo) * kUrmaQpNum);

        aclError err = aclrtMalloc(&urmaInfoDevice_, totalSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMalloc(urmaInfo) failed: " << err << std::endl;
            return false;
        }

        std::vector<uint8_t> hostBuf(totalSize, 0);
        FillUrmaInfoLayout(hostBuf, urmaPeerInfoTables, urmaRegistrationTokenState);

        err = aclrtMemcpy(urmaInfoDevice_, totalSize, hostBuf.data(), totalSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMemcpy(urmaInfo) failed: " << err << std::endl;
            return false;
        }
        return true;
    }

    void FillUrmaInfoLayout(
        std::vector<uint8_t>& hostBuf, const UrmaPeerInfoTables& urmaPeerInfoTables,
        const UrmaRegistrationTokenState& urmaRegistrationTokenState)
    {
        auto* info = reinterpret_cast<UrmaInfo*>(hostBuf.data());
        info->qpNum = kUrmaQpNum;
        info->localTokenId = urmaRegistrationTokenState.localTokenId;
        info->notifyPoolTokenId = urmaRegistrationTokenState.notifyPoolTokenId;
        info->rankCount = rankCount_;
        info->notifyPoolPtr = reinterpret_cast<uint64_t>(notifyPoolDevice_);

        uint8_t* devAddr = static_cast<uint8_t*>(urmaInfoDevice_) + sizeof(UrmaInfo);
        info->sqPtr = reinterpret_cast<uint64_t>(devAddr);
        devAddr += sizeof(UrmaWQCtx) * rankCount_ * kUrmaQpNum;
        info->rqPtr = reinterpret_cast<uint64_t>(devAddr);
        devAddr += sizeof(UrmaWQCtx) * rankCount_ * kUrmaQpNum;
        info->scqPtr = reinterpret_cast<uint64_t>(devAddr);
        devAddr += sizeof(UrmaCqCtx) * rankCount_ * kUrmaQpNum;
        info->rcqPtr = reinterpret_cast<uint64_t>(devAddr);
        devAddr += sizeof(UrmaCqCtx) * rankCount_ * kUrmaQpNum;
        info->memPtr = reinterpret_cast<uint64_t>(devAddr);

        uint8_t* hostAddr = hostBuf.data() + sizeof(UrmaInfo);
        auto* sqArr = reinterpret_cast<UrmaWQCtx*>(hostAddr);
        hostAddr += sizeof(UrmaWQCtx) * rankCount_ * kUrmaQpNum;
        auto* rqArr = reinterpret_cast<UrmaWQCtx*>(hostAddr);
        hostAddr += sizeof(UrmaWQCtx) * rankCount_ * kUrmaQpNum;
        auto* scqArr = reinterpret_cast<UrmaCqCtx*>(hostAddr);
        hostAddr += sizeof(UrmaCqCtx) * rankCount_ * kUrmaQpNum;
        auto* rcqArr = reinterpret_cast<UrmaCqCtx*>(hostAddr);
        hostAddr += sizeof(UrmaCqCtx) * rankCount_ * kUrmaQpNum;
        auto* memArr = reinterpret_cast<UrmaMemInfo*>(hostAddr);

        for (uint32_t rank = 0; rank < rankCount_; ++rank) {
            sqArr[rank] = urmaPeerInfoTables.wqList[rank];
            rqArr[rank] = urmaPeerInfoTables.wqList[rank];
            scqArr[rank] = urmaPeerInfoTables.cqList[rank];
            rcqArr[rank] = urmaPeerInfoTables.cqList[rank];
            memArr[rank] = urmaPeerInfoTables.memList[rank];
        }
    }

    static uint32_t Log2U32(uint32_t n) { return (n <= 1) ? 0 : __builtin_ctz(n); }

    static void FreeDeviceAddr(void*& addr)
    {
        if (addr) {
            aclrtFree(addr);
            addr = nullptr;
        }
    }

    static constexpr const char* kUrmaSymMemTag = "pto_urma_sym";
    static constexpr const char* kUrmaNotifyMemTag = "pto_urma_notify";
    static constexpr uint32_t kUrmaQpNum = 1U;
    static constexpr uint32_t kUrmaRegisteredMemCount = 2U;
    static constexpr uint64_t kDeviceVaThreshold = 0x100000000000ULL;
    static constexpr CommProtocol kCommProtocolUbcCtp = static_cast<CommProtocol>(4);
    static constexpr CommProtocol kCommProtocolUbcTp = static_cast<CommProtocol>(5);

    HcclComm comm_{nullptr};
    uint32_t rankId_{0};
    uint32_t rankCount_{0};
    void* symmetricAddr_{nullptr};
    uint64_t symmetricSize_{0};
    HcclMemHandle memHandle_{nullptr};
    HcclMemHandle notifyMemHandle_{nullptr};
    HcclMemHandle memHandles_[kUrmaRegisteredMemCount]{};

    std::vector<ChannelHandle> channelHandles_;

    void* urmaInfoDevice_{nullptr};
    void* eidDevice_{nullptr};
    void* notifyPoolDevice_{nullptr};
    uint64_t notifyPoolSize_{0};
    std::vector<uint64_t> peerBaseAddrs_;

    bool initialized_{false};
};

} // namespace urma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_URMA_WORKSPACE_MANAGER_HPP
