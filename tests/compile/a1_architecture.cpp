/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Compile with __NPU_ARCH__=2002 on a host, or with CCEC --cce-aicore-arch=dav-m200.
#include <cstdint>
#include <pto/common/arch_capability.hpp>

#ifndef PTO_NPU_ARCH_A1
#error "dav-m200 must select A1"
#endif
#if defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_A5)
#error "A1 must not alias a split Cube/Vector architecture"
#endif
#ifndef PTO_COMM_NOT_SUPPORTED
#error "A1 communication instructions have not been enabled"
#endif

static_assert(pto::GetCurrentArch() == pto::ChipArch::A1);
static_assert(pto::CurrArch::AccSupportsFloat);
static_assert(pto::CurrArch::AccSupportsInt32);
static_assert(!pto::CurrArch::SupportsBf16);
static_assert(!pto::CurrArch::SupportsFp8);
static_assert(!pto::CurrArch::SupportsFp4);
static_assert(!pto::CurrArch::SupportsSyncAll);
static_assert(!pto::CurrArch::SupportsComm);
static_assert(!pto::CurrArch::SupportsTQuant);
static_assert(!pto::CurrArch::SupportsTHistogram);
static_assert(!pto::CurrArch::SupportsMxLayout);
static_assert(pto::caps::IsFP16<half>());
static_assert(!pto::caps::IsBF16<half>());
static_assert(static_cast<unsigned>(pto::ChipArch::A2A3) == 0);
static_assert(static_cast<unsigned>(pto::ChipArch::A5) == 1);
static_assert(static_cast<unsigned>(pto::ChipArch::A6) == 4);

int main() { return 0; }
