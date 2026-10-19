/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#pragma once

#include <cstdint>

bool RunTPutAsyncNotifySet(int nRanks, int nDevices, int firstRankId, int firstDeviceId, uint32_t queueNum);
bool RunTPutAsyncNotifyConsumeSet(int nRanks, int nDevices, int firstRankId, int firstDeviceId, uint32_t queueNum);
bool RunTPutAsyncNotifyAdd(int nRanks, int nDevices, int firstRankId, int firstDeviceId, uint32_t queueNum);

bool RunTPutAsyncNotifyAddRingReuse(int nRanks, int nDevices, int firstRankId, int firstDeviceId, uint32_t queueNum);

bool RunTPutAsyncNotifyInterleavedRingReuse(
    int nRanks, int nDevices, int firstRankId, int firstDeviceId, uint32_t queueNum);

bool RunTPutAsyncNotifyNonZeroGroup(int nRanks, int nDevices, int firstRankId, int firstDeviceId, uint32_t queueNum);

bool RunTPutAsyncNotifyConcurrentSet(int nRanks, int nDevices, int firstRankId, int firstDeviceId, uint32_t queueNum);

bool RunTPutAsyncNotifyConcurrentAdd(int nRanks, int nDevices, int firstRankId, int firstDeviceId, uint32_t queueNum);
