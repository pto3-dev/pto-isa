/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TPUT_ASYNC_NOTIFY_HPP
#define PTO_COMM_TPUT_ASYNC_NOTIFY_HPP

#include "pto/comm/async_common/TPutAsyncCommonDetail.hpp"

namespace pto {
namespace comm {
namespace detail {

template <typename GlobalDstData, typename GlobalSrcData, typename GlobalSignalData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_NOTIFY_SDMA_IMPL(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, GlobalSignalData& dstSignalData, int32_t signalValue,
    NotifyOp notifyOp, const AsyncSession& session)
{
    PTO_ASSERT(
        session.valid && session.engine == DmaEngine::SDMA, "TPUT_ASYNC_NOTIFY: A2/A3 requires a valid SDMA session.");

    TPutAsyncValidateNotifySignal(dstSignalData, notifyOp);
    const uint64_t transferSize = TPutAsyncValidatePayload(dstGlobalData, srcGlobalData);
    const uint64_t eventHandle = sdma::__sdma_put_async_notify(
        dstGlobalData.data(), srcGlobalData.data(), dstSignalData.data(), signalValue, notifyOp, transferSize, session);
    return AsyncEvent(eventHandle, DmaEngine::SDMA);
}

} // namespace detail

template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData, typename GlobalSignalData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_NOTIFY_IMPL(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, GlobalSignalData& dstSignalData, int32_t signalValue,
    NotifyOp notifyOp, const AsyncSession& session, uint32_t peer)
{
    (void)peer;
    if constexpr (engine == DmaEngine::SDMA) {
        return detail::TPUT_ASYNC_NOTIFY_SDMA_IMPL(
            dstGlobalData, srcGlobalData, dstSignalData, signalValue, notifyOp, session);
    } else {
        static_assert(engine == DmaEngine::SDMA, "TPUT_ASYNC_NOTIFY: only SDMA is supported on A2/A3");
        return {};
    }
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TPUT_ASYNC_NOTIFY_HPP
