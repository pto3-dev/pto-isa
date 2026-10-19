/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "acl/acl.h"
#include "test_common.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <int32_t tilingKey>
void LaunchTPushPopSubBlockDispatchV2C(uint8_t* out, uint8_t* srcA, uint8_t* srcB, void* stream);

template <int32_t tilingKey>
void LaunchTPushPopSubBlockDispatchC2VGm(uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* fifoMem, void* stream);

class TPushPopSubBlockDispatchTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    return "../" + suiteName + "." + caseName;
}

template <typename T, int32_t key>
void TPushPopSubBlockDispatchV2CTestFunc(uint32_t m, uint32_t k, uint32_t n)
{
    size_t aFileSize = m * k * sizeof(T);
    size_t bFileSize = k * n * sizeof(T);
    size_t cElementCount = m * n;
    size_t cFileSize = cElementCount * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *srcAHost, *srcBHost;
    uint8_t *dstDevice, *srcADevice, *srcBDevice;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&srcAHost), aFileSize);
    aclrtMallocHost((void**)(&srcBHost), bFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcADevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcBDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, srcAHost, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, srcBHost, bFileSize);

    aclrtMemcpy(srcADevice, aFileSize, srcAHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcBDevice, bFileSize, srcBHost, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTPushPopSubBlockDispatchV2C<key>(dstDevice, srcADevice, srcBDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcADevice);
    aclrtFree(srcBDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcAHost);
    aclrtFreeHost(srcBHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(cElementCount);
    std::vector<T> devFinal(cElementCount);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

template <typename T, int32_t key>
void TPushPopSubBlockDispatchC2VGmTestFunc(uint32_t m, uint32_t k, uint32_t n)
{
    size_t aFileSize = m * k * sizeof(T);
    size_t bFileSize = k * n * sizeof(T);
    size_t cElementCount = m * n;
    size_t cFileSize = cElementCount * sizeof(T);
    // FIFO_DEPTH slots of one [m, n] accumulator tile each.
    size_t fifoFileSize = 2 * cFileSize;

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *srcAHost, *srcBHost;
    uint8_t *dstDevice, *srcADevice, *srcBDevice, *fifoMemDevice;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&srcAHost), aFileSize);
    aclrtMallocHost((void**)(&srcBHost), bFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcADevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcBDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&fifoMemDevice, fifoFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, srcAHost, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, srcBHost, bFileSize);

    aclrtMemcpy(srcADevice, aFileSize, srcAHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcBDevice, bFileSize, srcBHost, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTPushPopSubBlockDispatchC2VGm<key>(dstDevice, srcADevice, srcBDevice, fifoMemDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcADevice);
    aclrtFree(srcBDevice);
    aclrtFree(fifoMemDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcAHost);
    aclrtFreeHost(srcBHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(cElementCount);
    std::vector<T> devFinal(cElementCount);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

// V2C, TILE_NO_SPLIT, two-argument TPUSH/TPOP: the single logical lane must be
// addressed without reading the hardware sub-block ID.
TEST_F(TPushPopSubBlockDispatchTest, case1_v2c_nosplit_implicit_id)
{
    TPushPopSubBlockDispatchV2CTestFunc<float, 1>(16, 64, 32);
}

// V2C, TILE_NO_SPLIT, three-argument TPUSH/TPOP with a non-zero ID: the payload
// must land in the same place as case1.
TEST_F(TPushPopSubBlockDispatchTest, case2_v2c_nosplit_explicit_id)
{
    TPushPopSubBlockDispatchV2CTestFunc<float, 2>(16, 64, 32);
}

// V2C, TILE_UP_DOWN, two-argument TPUSH: each vector core must keep writing its
// own row window through the hardware sub-block ID.
TEST_F(TPushPopSubBlockDispatchTest, case3_v2c_split_implicit_id)
{
    TPushPopSubBlockDispatchV2CTestFunc<float, 3>(16, 64, 32);
}

// V2C, TILE_UP_DOWN, three-argument TPUSH with the peer's ID: the caller
// supplied ID wins, so the two row windows swap.
TEST_F(TPushPopSubBlockDispatchTest, case4_v2c_split_explicit_swapped_id)
{
    TPushPopSubBlockDispatchV2CTestFunc<float, 4>(16, 64, 32);
}

// C2V GM FIFO, TILE_NO_SPLIT, two-argument TPOP: the slot is read from offset 0
// whatever the hardware sub-block ID is.
TEST_F(TPushPopSubBlockDispatchTest, case5_c2v_gm_nosplit_implicit_id)
{
    TPushPopSubBlockDispatchC2VGmTestFunc<float, 5>(32, 32, 64);
}

// C2V GM FIFO, TILE_UP_DOWN, two-argument TPOP: each vector core must keep
// reading its own sub-range through the hardware sub-block ID.
TEST_F(TPushPopSubBlockDispatchTest, case6_c2v_gm_split_implicit_id)
{
    TPushPopSubBlockDispatchC2VGmTestFunc<float, 6>(32, 32, 64);
}

// C2V GM FIFO, TILE_UP_DOWN, three-argument TPOP with the peer's ID: the caller
// supplied ID wins, so the two sub-ranges swap.
TEST_F(TPushPopSubBlockDispatchTest, case7_c2v_gm_split_explicit_swapped_id)
{
    TPushPopSubBlockDispatchC2VGmTestFunc<float, 7>(32, 32, 64);
}
