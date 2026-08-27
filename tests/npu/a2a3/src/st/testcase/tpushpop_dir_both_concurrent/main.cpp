/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include "acl/acl.h"
#include "runtime/rt.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <int32_t tilingKey>
void LaunchTPushPopDirBoth(
    uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* srcD, uint8_t* srcF, uint8_t* fifoMem,
    uint8_t* outCube, void* stream);

class TPushPopDirBothConcurrentTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

// Host and device buffers for one run, grouped so setup / teardown / verify can be separate
// functions instead of one long test body.
struct DirBothBuffers {
    size_t aSize, bSize, dSize, fSize, outSize, fifoSize;
    uint8_t *outHost, *outCubeHost, *srcAHost, *srcBHost, *srcDHost, *srcFHost;
    uint8_t *outDevice, *outCubeDevice, *srcADevice, *srcBDevice, *srcDDevice, *srcFDevice, *fifoMemDevice;
};

template <typename T>
void SetupBuffers(DirBothBuffers& b, uint32_t M, uint32_t K, uint32_t N)
{
    b.aSize = M * K * sizeof(T);
    b.bSize = M * K * sizeof(T);
    b.dSize = K * N * sizeof(T);
    b.fSize = M * N * sizeof(T);
    b.outSize = M * N * sizeof(T);
    // A DIR_BOTH pipe is two rings in GM: C2V at offset 0 and V2C at SLOT_NUM * SLOT_SIZE.
    // SLOT_SIZE = M * N * sizeof(T) and SLOT_NUM = FIFO_DEPTH = 2 in the kernel, so the
    // buffer must be 2 * SLOT_NUM * SLOT_SIZE.
    b.fifoSize = 4 * M * N * sizeof(T);

    aclrtMallocHost((void**)(&b.outHost), b.outSize);
    aclrtMallocHost((void**)(&b.outCubeHost), b.outSize);
    aclrtMallocHost((void**)(&b.srcAHost), b.aSize);
    aclrtMallocHost((void**)(&b.srcBHost), b.bSize);
    aclrtMallocHost((void**)(&b.srcDHost), b.dSize);
    aclrtMallocHost((void**)(&b.srcFHost), b.fSize);

    aclrtMalloc((void**)&b.outDevice, b.outSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&b.outCubeDevice, b.outSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&b.srcADevice, b.aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&b.srcBDevice, b.bSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&b.srcDDevice, b.dSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&b.srcFDevice, b.fSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&b.fifoMemDevice, b.fifoSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/srcA_gm.bin", b.aSize, b.srcAHost, b.aSize);
    ReadFile(GetGoldenDir() + "/srcB_gm.bin", b.bSize, b.srcBHost, b.bSize);
    ReadFile(GetGoldenDir() + "/srcD_gm.bin", b.dSize, b.srcDHost, b.dSize);
    ReadFile(GetGoldenDir() + "/srcF_gm.bin", b.fSize, b.srcFHost, b.fSize);

    aclrtMemcpy(b.srcADevice, b.aSize, b.srcAHost, b.aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(b.srcBDevice, b.bSize, b.srcBHost, b.bSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(b.srcDDevice, b.dSize, b.srcDHost, b.dSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(b.srcFDevice, b.fSize, b.srcFHost, b.fSize, ACL_MEMCPY_HOST_TO_DEVICE);
}

void TeardownBuffers(DirBothBuffers& b)
{
    aclrtFree(b.outDevice);
    aclrtFree(b.outCubeDevice);
    aclrtFree(b.srcADevice);
    aclrtFree(b.srcBDevice);
    aclrtFree(b.srcDDevice);
    aclrtFree(b.srcFDevice);
    aclrtFree(b.fifoMemDevice);

    aclrtFreeHost(b.outHost);
    aclrtFreeHost(b.outCubeHost);
    aclrtFreeHost(b.srcAHost);
    aclrtFreeHost(b.srcBHost);
    aclrtFreeHost(b.srcDHost);
    aclrtFreeHost(b.srcFHost);
}

// Check BOTH directions. The two payloads are different sizes, so whichever push lands
// second overwrites only part of the other tile and a single-sided check can still come
// back clean. On an affected build both comparisons fail.
template <typename T>
void VerifyOutputs(size_t outFileSize)
{
    std::vector<T> golden(outFileSize / sizeof(T));
    std::vector<T> devFinal(outFileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", outFileSize, golden.data(), outFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", outFileSize, devFinal.data(), outFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret) << "vector side (C2V) mismatch";

    std::vector<T> goldenCube(outFileSize / sizeof(T));
    std::vector<T> devCube(outFileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden_cube.bin", outFileSize, goldenCube.data(), outFileSize);
    ReadFile(GetGoldenDir() + "/output_cube.bin", outFileSize, devCube.data(), outFileSize);

    bool retCube = ResultCmp(goldenCube, devCube, 0.001f);
    EXPECT_TRUE(retCube) << "cube side (V2C) mismatch -- C2V and V2C aliased the same GM slot";
}

template <typename T, int32_t key>
void TPushPopDirBothConcurrentTestFunc(uint32_t M, uint32_t K, uint32_t N)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    DirBothBuffers b{};
    SetupBuffers<T>(b, M, K, N);

    uint64_t ffts{0};
    uint32_t fftsLen{0};
    rtGetC2cCtrlAddr(&ffts, &fftsLen);

    LaunchTPushPopDirBoth<key>(
        (uint8_t*)ffts, b.outDevice, b.srcADevice, b.srcBDevice, b.srcDDevice, b.srcFDevice, b.fifoMemDevice,
        b.outCubeDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(b.outHost, b.outSize, b.outDevice, b.outSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(b.outCubeHost, b.outSize, b.outCubeDevice, b.outSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", b.outHost, b.outSize);
    WriteFile(GetGoldenDir() + "/output_cube.bin", b.outCubeHost, b.outSize);

    const size_t outFileSize = b.outSize;
    TeardownBuffers(b);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    VerifyOutputs<T>(outFileSize);
}

TEST_F(TPushPopDirBothConcurrentTest, case1_float_dir_both_concurrent)
{
    TPushPopDirBothConcurrentTestFunc<float, 1>(128, 64, 128);
}

TEST_F(TPushPopDirBothConcurrentTest, case2_float_dir_both_concurrent_left_right)
{
    TPushPopDirBothConcurrentTestFunc<float, 2>(128, 64, 128);
}
