# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

pto_vec_st(tfillpad)
# Host main.cpp is compiled with -xc++ and does not get --cce-aicore-arch, so
# PTO_NPU_ARCH_A5 is not defined via arch_macro.hpp. Define it for A5 ST so
# the fp8/hif8/fp4 cases (23-39) are registered. A2A3 SOC_VERSION does not match.
if(SOC_VERSION MATCHES "9599")
    target_compile_definitions(tfillpad PRIVATE PTO_NPU_ARCH_A5)
    target_compile_definitions(tfillpad_kernel PRIVATE PTO_NPU_ARCH_A5)
endif()
