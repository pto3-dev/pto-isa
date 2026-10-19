/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_A1_TSYNC_HPP
#define PTO_NPU_A1_TSYNC_HPP

#include <pto/common/event.hpp>

namespace pto {

template <Op OpCode>
PTO_INTERNAL void TSYNC_IMPL()
{
#ifndef __PTO_AUTO__
    constexpr pipe_t pipe = OpPipeEntry<OpCode>::pipe;
    PTO_STATIC_ASSERT(pipe != PIPE_ALL, "A1 TSYNC requires a supported local operation.");
    pipe_barrier((pipe_t)pipe);
#endif
}

template <Op SrcOp, Op DstOp, bool AutoToken = true, event_t EventID = EVENT_ID0>
struct Event : EventBase<Event<SrcOp, DstOp, AutoToken, EventID>, SrcOp, DstOp, AutoToken, EventID> {
    using Base = EventBase<Event, SrcOp, DstOp, AutoToken, EventID>;
    using Base::operator=;
    static constexpr bool IsCrossCore = false;

    template <Op op>
    PTO_INTERNAL static constexpr pipe_t GetPipeByOp()
    {
        return OpPipeEntry<op>::pipe;
    }

#ifndef __PTO_AUTO__
    PTO_STATIC_ASSERT(
        Base::srcPipe != PIPE_ALL && Base::dstPipe != PIPE_ALL,
        "A1 Event requires supported local source and destination operations.");
    PTO_STATIC_ASSERT(AutoToken || EventID != EVENT_ID7, "A1 event ID 7 is reserved for ISA helpers.");
#endif

    PTO_INTERNAL Event& InitAddrImpl(uint64_t)
    {
        PTO_STATIC_ASSERT(SrcOp != SrcOp, "A1 events do not support cross-core FFTS addresses.");
        return *this;
    }

    template <uint8_t CrossCoreId = 0xff>
    PTO_INTERNAL Event& InitImpl()
    {
        PTO_STATIC_ASSERT(CrossCoreId == 0xff, "A1 events do not support cross-core IDs.");
#ifndef __PTO_AUTO__
        if constexpr (!Base::isSamePipe) {
            set_flag((pipe_t)Base::srcPipe, (pipe_t)Base::dstPipe, Base::token);
        }
#endif
        return *this;
    }

    template <uint8_t CrossCoreId = 0xff>
    PTO_INTERNAL Event& WaitImpl()
    {
        PTO_STATIC_ASSERT(CrossCoreId == 0xff, "A1 events do not support cross-core IDs.");
#ifndef __PTO_AUTO__
        if constexpr (Base::isSamePipe) {
            pipe_barrier((pipe_t)Base::srcPipe);
        } else {
            wait_flag((pipe_t)Base::srcPipe, (pipe_t)Base::dstPipe, Base::token);
        }
#if defined(__CPU_SIM) || defined(__COSTMODEL)
        EventIdCounter<Base::srcPipe, Base::dstPipe>::MarkFree(Base::token);
#endif
#endif
        return *this;
    }

    PTO_INTERNAL Event() = default;
    PTO_INTERNAL Event(RecordEvent record) : Base(record) {}
};

} // namespace pto
#endif // PTO_NPU_A1_TSYNC_HPP
