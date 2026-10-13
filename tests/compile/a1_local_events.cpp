/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>
#if defined(__CPU_SIM)
#include <pto/common/cpu_stub.hpp>
#endif
#include <pto/common/arch_capability.hpp>
#include <pto/npu/a1/TSync.hpp>

static_assert(pto::OpPipeEntry<pto::Op::TLOAD>::pipe == PIPE_MTE2);
static_assert(pto::OpPipeEntry<pto::Op::TADD>::pipe == PIPE_V);
static_assert(pto::OpPipeEntry<pto::Op::TSTORE_VEC>::pipe == PIPE_MTE3);
static_assert(pto::OpPipeEntry<pto::Op::TMOV_A2V>::pipe == PIPE_V);
static_assert(pto::OpPipeEntry<pto::Op::TSTORE_ACC>::pipe == PIPE_ALL);

AICORE void CheckA1LocalEvents()
{
#if defined(PTO_A1_TEST_RESERVED_ID)
    pto::Event<pto::Op::TLOAD, pto::Op::TADD, false, EVENT_ID7> loaded;
#else
    pto::Event<pto::Op::TLOAD, pto::Op::TADD, false, EVENT_ID0> loaded;
#endif
#if defined(PTO_A1_TEST_CROSS_CORE)
    loaded.Init<0>();
#else
    loaded.Init();
#endif
    loaded.Wait();
    pto::Event<pto::Op::TADD, pto::Op::TSTORE_VEC, false, EVENT_ID0> added;
    added.Init().Wait();
    pto::TSYNC_IMPL<pto::Op::TADD>();
    pto::Event<pto::Op::TLOAD, pto::Op::TADD> automatic;
    automatic.Init().Wait();
}

#if defined(__CPU_SIM)
int main()
{
    CheckA1LocalEvents();
    using Counter = pto::EventIdCounter<PIPE_MTE2, PIPE_V>;
    Counter::Reset();
    for (int i = 0; i < 100; ++i) {
        auto id = Counter::GetNextId();
        if (id == EVENT_ID7)
            return 1;
        Counter::MarkFree(id);
    }
    return 0;
}
#endif
