/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_A5_TPUT_ASYNC_NOTIFY_HPP
#define PTO_COMM_A5_TPUT_ASYNC_NOTIFY_HPP

#include "pto/comm/a5/TNotify.hpp"
#include "pto/comm/a5/async/TPutAsync.hpp"

namespace pto {
namespace comm {
namespace detail {

// A5 keeps DmaEngine::SDMA as an API-compatible name for the synchronous MTE
// fallback. Complete every payload chunk before issuing the Scalar notification,
// then return handle 0 to represent an already-completed operation.
//
// Intended ordering and concurrency semantics:
// - Repeated calls from one AICore are fully serialized.
// - AtomicAdd to one signal may aggregate completions from multiple AICores or
//   ranks, provided every payload uses a non-conflicting destination range.
// - Set does not define a winner when multiple AICores or ranks write one signal;
//   use one signal slot per producer or AtomicAdd for completion counting.
// Receiver-side visibility and multi-producer cases require dedicated hardware
// stress tests; the current ST covers sequential calls from one sender AICore.
template <typename GlobalDstData, typename GlobalSrcData, typename GlobalSignalData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_NOTIFY_MTE_FALLBACK(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, GlobalSignalData& dstSignalData, int32_t signalValue,
    NotifyOp notifyOp, const AsyncSession& session)
{
    PTO_ASSERT(
        session.valid && session.engine == DmaEngine::SDMA,
        "TPUT_ASYNC_NOTIFY: A5 MTE fallback requires a valid SDMA session.");
    TPutAsyncValidateNotifySignal(dstSignalData, notifyOp);

    // TPUT_ASYNC_MTE_FALLBACK uses MTE3 -> MTE2 events for UB reuse. Ensure the
    // payload reaches DDR, then order all pipelines before the Scalar notification.
    (void)TPUT_ASYNC_MTE_FALLBACK(dstGlobalData, srcGlobalData, session);
    dsb(DSB_DDR);
    pipe_barrier(PIPE_ALL);
    TNOTIFY_IMPL(dstSignalData, signalValue, notifyOp);
    return AsyncEvent(0, DmaEngine::SDMA);
}

#ifdef PTO_URMA_SUPPORTED
template <typename GlobalDstData, typename GlobalSrcData, typename GlobalSignalData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_NOTIFY_URMA(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, GlobalSignalData& dstSignalData, int32_t signalValue,
    NotifyOp notifyOp, const AsyncSession& session, uint32_t peer)
{
    const uint64_t transferSize = TPutAsyncCheckUrmaPayload(dstGlobalData, srcGlobalData, session, peer);
    TPutAsyncValidateNotifySignal(dstSignalData, notifyOp);

    const urma::detail::UrmaPostResult result = urma::__urma_put_async_notify(
        reinterpret_cast<__gm__ uint8_t*>(dstGlobalData.data()),
        reinterpret_cast<__gm__ uint8_t*>(srcGlobalData.data()), transferSize,
        reinterpret_cast<__gm__ int32_t*>(dstSignalData.data()), signalValue, notifyOp, session, peer);
    return AsyncEvent(result.handle, DmaEngine::URMA, result.targetCqe);
}
#endif

#ifdef PTO_RDMA_SUPPORTED
template <typename GlobalDstData, typename GlobalSrcData, typename GlobalSignalData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_NOTIFY_RDMA(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, GlobalSignalData& dstSignalData, int32_t signalValue,
    NotifyOp notifyOp, const AsyncSession& session, uint32_t peer)
{
    PTO_ASSERT(
        session.valid && session.engine == DmaEngine::RDMA, "TPUT_ASYNC_NOTIFY: RDMA requires a valid RDMA session.");
    const uint64_t transferSize = TPutAsyncValidatePayload(dstGlobalData, srcGlobalData);
    TPutAsyncValidateNotifySignal(dstSignalData, notifyOp);
    if (notifyOp != NotifyOp::Set) {
        PTO_ASSERT(false, "TPUT_ASYNC_NOTIFY: RDMA currently supports NotifyOp::Set only.");
        return AsyncEvent(rdma::EncodeErrorHandle(rdma::kRdmaUnsupportedOperationError), DmaEngine::RDMA);
    }
    PTO_ASSERT(transferSize <= 0x7fffffffULL, "TPUT_ASYNC_NOTIFY: RDMA payload must not exceed 0x7fffffff bytes.");

    const uint64_t eventHandle = rdma::WriteNotify(
        session, reinterpret_cast<__gm__ uint8_t*>(dstGlobalData.data()),
        reinterpret_cast<__gm__ uint8_t*>(srcGlobalData.data()), transferSize,
        reinterpret_cast<__gm__ int32_t*>(dstSignalData.data()), signalValue, peer);
    return AsyncEvent(eventHandle, DmaEngine::RDMA);
}
#endif

} // namespace detail

template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData, typename GlobalSignalData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_NOTIFY_IMPL(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, GlobalSignalData& dstSignalData, int32_t signalValue,
    NotifyOp notifyOp, const AsyncSession& session, uint32_t peer)
{
    if constexpr (engine == DmaEngine::SDMA) {
        (void)peer;
        return detail::TPUT_ASYNC_NOTIFY_MTE_FALLBACK(
            dstGlobalData, srcGlobalData, dstSignalData, signalValue, notifyOp, session);
    } else if constexpr (engine == DmaEngine::URMA) {
#ifdef PTO_URMA_SUPPORTED
        return detail::TPUT_ASYNC_NOTIFY_URMA(
            dstGlobalData, srcGlobalData, dstSignalData, signalValue, notifyOp, session, peer);
#else
        static_assert(engine != DmaEngine::URMA, "TPUT_ASYNC_NOTIFY: A5 URMA requires NPU_ARCH 3510");
        return {};
#endif
    } else if constexpr (engine == DmaEngine::RDMA) {
#ifdef PTO_RDMA_SUPPORTED
        return detail::TPUT_ASYNC_NOTIFY_RDMA(
            dstGlobalData, srcGlobalData, dstSignalData, signalValue, notifyOp, session, peer);
#else
        static_assert(engine != DmaEngine::RDMA, "TPUT_ASYNC_NOTIFY: RDMA support is not enabled for this build");
        return AsyncEvent(0, engine);
#endif
    } else {
        static_assert(engine == DmaEngine::SDMA, "TPUT_ASYNC_NOTIFY: unsupported DMA engine");
        return AsyncEvent(0, engine);
    }
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_A5_TPUT_ASYNC_NOTIFY_HPP
