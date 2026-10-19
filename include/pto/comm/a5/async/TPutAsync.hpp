/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TPUT_ASYNC_HPP
#define PTO_COMM_TPUT_ASYNC_HPP

#include "pto/comm/async_common/TPutAsyncCommonDetail.hpp"
#ifdef PTO_URMA_SUPPORTED
#include "pto/comm/async/urma/urma_async_intrin.hpp"
#endif
#ifdef PTO_RDMA_SUPPORTED
#include "pto/comm/async/rdma/rdma_async_intrin.hpp"
#endif

namespace pto {
namespace comm {
namespace detail {

// ============================================================================
// TPUT_ASYNC_MTE_FALLBACK: Synchronous MTE fallback for A5 platforms where
// SDMA does not support PUT direction.
//
// Uses the session's UB scratch buffer (tmpBuf) as staging to perform a
// chunked GM -> UB -> GM transfer via MTE2/MTE3 pipelines. The operation
// completes synchronously; the returned AsyncEvent has handle=0 (already done).
// ============================================================================

template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent
TPUT_ASYNC_MTE_FALLBACK(GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, const AsyncSession& session)
{
    const uint64_t totalBytes = TPutAsyncValidatePayload(dstGlobalData, srcGlobalData);
    if (totalBytes == 0) {
        return AsyncEvent(0, DmaEngine::SDMA);
    }

    __ubuf__ uint8_t* ubBuf = session.tmpBufAddr;
    const uint32_t ubSize = session.tmpBufSize;
    PTO_ASSERT(ubBuf != nullptr && ubSize > 0, "TPUT_ASYNC MTE fallback: tmpBuf is invalid");

    __gm__ uint8_t* srcPtr = reinterpret_cast<__gm__ uint8_t*>(srcGlobalData.data());
    __gm__ uint8_t* dstPtr = reinterpret_cast<__gm__ uint8_t*>(dstGlobalData.data());

    uint64_t offset = 0;
    while (offset < totalBytes) {
        const uint64_t remaining = totalBytes - offset;
        const uint32_t chunkBytes = static_cast<uint32_t>((remaining < ubSize) ? remaining : ubSize);

        copy_gm_to_ubuf_align_v2(
            reinterpret_cast<__ubuf__ uint8_t*>(ubBuf), reinterpret_cast<__gm__ uint8_t*>(srcPtr + offset), 0, 1,
            chunkBytes, 0, 0, false, 0, chunkBytes, chunkBytes);
        set_flag(PIPE_MTE2, PIPE_MTE3, session.syncId);
        wait_flag(PIPE_MTE2, PIPE_MTE3, session.syncId);

        copy_ubuf_to_gm_align_v2(
            reinterpret_cast<__gm__ uint8_t*>(dstPtr + offset), reinterpret_cast<__ubuf__ uint8_t*>(ubBuf), 0, 1,
            chunkBytes, 0, chunkBytes, chunkBytes);
        set_flag(PIPE_MTE3, PIPE_MTE2, session.syncId);
        wait_flag(PIPE_MTE3, PIPE_MTE2, session.syncId);

        offset += chunkBytes;
    }

    return AsyncEvent(0, DmaEngine::SDMA);
}

#ifdef PTO_URMA_SUPPORTED
template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL uint64_t TPutAsyncCheckUrmaPayload(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, const AsyncSession& session, uint32_t peer)
{
    PTO_ASSERT(urma::detail::ValidateUrmaSession(session, peer), "TPUT_ASYNC URMA: invalid session, peer or QP.");
    const uint64_t transferSize = TPutAsyncValidatePayload(dstGlobalData, srcGlobalData);
    PTO_ASSERT(
        transferSize > 0U && transferSize <= urma::kUrmaMaxWqeTransferBytes,
        "TPUT_ASYNC URMA: transfer size must be in (0, 256MB] per single WQE.");
    return transferSize;
}

template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_URMA_IMPL(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, const AsyncSession& session, uint32_t peer)
{
    const uint64_t transferSize = TPutAsyncCheckUrmaPayload(dstGlobalData, srcGlobalData, session, peer);

    const urma::detail::UrmaPostResult result = urma::__urma_put_async(
        reinterpret_cast<__gm__ uint8_t*>(dstGlobalData.data()),
        reinterpret_cast<__gm__ uint8_t*>(srcGlobalData.data()), transferSize, session, peer);
    return AsyncEvent(result.handle, DmaEngine::URMA, result.targetCqe);
}

template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent
TPUT_ASYNC_URMA_IMPL(GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, const AsyncSession& session)
{
    return TPUT_ASYNC_URMA_IMPL(dstGlobalData, srcGlobalData, session, session.destRankId);
}
#endif

#ifdef PTO_RDMA_SUPPORTED
// RDMA PUT: async remote write. dst is remote, src is local.
template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_RDMA_IMPL(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, const AsyncSession& session, uint32_t peer)
{
    const uint64_t transferSize = TPutAsyncValidatePayload(dstGlobalData, srcGlobalData);
    if (transferSize == 0) {
        return AsyncEvent(0, DmaEngine::RDMA);
    }

    const uint64_t eventHandle = rdma::Write(
        session, reinterpret_cast<__gm__ uint8_t*>(dstGlobalData.data()),
        reinterpret_cast<__gm__ uint8_t*>(srcGlobalData.data()), transferSize, peer);
    return AsyncEvent(eventHandle, DmaEngine::RDMA);
}

template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent
TPUT_ASYNC_RDMA_IMPL(GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, const AsyncSession& session)
{
    return TPUT_ASYNC_RDMA_IMPL(dstGlobalData, srcGlobalData, session, session.destRankId);
}
#endif

} // namespace detail

// ============================================================================
// Main TPUT_ASYNC_IMPL with DmaEngine template parameter
// A5: SDMA uses MTE fallback for PUT direction; URMA and RDMA are also supported
// ============================================================================

template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent
TPUT_ASYNC_IMPL(GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, const AsyncSession& session)
{
    if constexpr (engine == DmaEngine::SDMA) {
        return detail::TPUT_ASYNC_MTE_FALLBACK(dstGlobalData, srcGlobalData, session);
    } else if constexpr (engine == DmaEngine::URMA) {
#ifdef PTO_URMA_SUPPORTED
        return detail::TPUT_ASYNC_URMA_IMPL(dstGlobalData, srcGlobalData, session);
#else
        static_assert(engine != DmaEngine::URMA, "TPUT_ASYNC: URMA engine requires NPU_ARCH 3510");
        return AsyncEvent(0, engine);
#endif
    } else if constexpr (engine == DmaEngine::RDMA) {
#ifdef PTO_RDMA_SUPPORTED
        return detail::TPUT_ASYNC_RDMA_IMPL(dstGlobalData, srcGlobalData, session);
#else
        static_assert(engine != DmaEngine::RDMA, "TPUT_ASYNC: RDMA support is not enabled for this build");
        return AsyncEvent(0, engine);
#endif
    } else {
        PTO_ASSERT(false, "TPUT_ASYNC: unsupported engine");
        return AsyncEvent(0, engine);
    }
}

// peer overload: URMA and RDMA use peer for queue/memory selection; SDMA ignores peer.
template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent
TPUT_ASYNC_IMPL(GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, const AsyncSession& session, uint32_t peer)
{
    if constexpr (engine == DmaEngine::SDMA) {
        return detail::TPUT_ASYNC_MTE_FALLBACK(dstGlobalData, srcGlobalData, session);
    } else if constexpr (engine == DmaEngine::URMA) {
#ifdef PTO_URMA_SUPPORTED
        return detail::TPUT_ASYNC_URMA_IMPL(dstGlobalData, srcGlobalData, session, peer);
#else
        static_assert(engine != DmaEngine::URMA, "TPUT_ASYNC: URMA engine requires NPU_ARCH 3510");
        return AsyncEvent(0, engine);
#endif
    } else if constexpr (engine == DmaEngine::RDMA) {
#ifdef PTO_RDMA_SUPPORTED
        return detail::TPUT_ASYNC_RDMA_IMPL(dstGlobalData, srcGlobalData, session, peer);
#else
        static_assert(engine != DmaEngine::RDMA, "TPUT_ASYNC: RDMA support is not enabled for this build");
        return AsyncEvent(0, engine);
#endif
    } else {
        PTO_ASSERT(false, "TPUT_ASYNC: unsupported engine");
        return AsyncEvent(0, engine);
    }
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TPUT_ASYNC_HPP
