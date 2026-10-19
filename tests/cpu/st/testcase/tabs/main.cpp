/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <int32_t tilingKey>
void launchTABS_demo(uint8_t* out, uint8_t* src, void* stream);

class TABSTest : public testing::Test {
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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchTAbs(T* out, T* src, void* stream);

template <int kRows, int kCols>
void LaunchTAbsMixedValidShape(int32_t* out, int32_t* src, void* stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, bool mixedValidShape = false>
void test_tabs(const std::string& goldenDir = GetGoldenDir())
{
    size_t fileSize = kGRows_ * kGCols_ * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), fileSize);
    aclrtMallocHost((void**)(&srcHost), fileSize);

    aclrtMalloc((void**)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(dstDevice, fileSize, 0, fileSize);

    CHECK_RESULT_GTEST(ReadFile(goldenDir + "/input1.bin", fileSize, srcHost, fileSize));

    aclrtMemcpy(srcDevice, fileSize, srcHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if constexpr (mixedValidShape) {
        LaunchTAbsMixedValidShape<kTRows_, kTCols_>(
            reinterpret_cast<int32_t*>(dstDevice), reinterpret_cast<int32_t*>(srcDevice), stream);
    } else {
        LaunchTAbs<T, kGRows_, kGCols_, kTRows_, kTCols_>(dstDevice, srcDevice, stream);
    }

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(goldenDir + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSize / sizeof(T));
    std::vector<T> devFinal(fileSize / sizeof(T));
    CHECK_RESULT_GTEST(ReadFile(goldenDir + "/golden.bin", fileSize, golden.data(), fileSize));
    CHECK_RESULT_GTEST(ReadFile(goldenDir + "/output.bin", fileSize, devFinal.data(), fileSize));

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TABSTest, case_float_64x64_64x64_64x64) { test_tabs<float, 64, 64, 64, 64>(); }
TEST_F(TABSTest, case_int32_64x64_64x64_64x64) { test_tabs<int32_t, 64, 64, 64, 64>(); }
TEST_F(TABSTest, case_int16_64x64_64x64_64x64) { test_tabs<int16_t, 64, 64, 64, 64>(); }
TEST_F(TABSTest, case_int8_64x64_64x64_64x64) { test_tabs<int8_t, 64, 64, 64, 64>(); }
TEST_F(TABSTest, case_half_16x256_16x256_16x256) { test_tabs<aclFloat16, 16, 256, 16, 256>(); }

TEST_F(TABSTest, case_int32_64x64_mixed_static_valid)
{
    test_tabs<int32_t, 64, 64, 64, 64, true>("../TABSTest.case_int32_64x64_64x64_64x64");
}
#ifdef CPU_SIM_BFLOAT_ENABLED
TEST_F(TABSTest, case_bf16_16x256_16x256_16x256) { test_tabs<bfloat16_t, 16, 256, 16, 256>(); }
#endif
