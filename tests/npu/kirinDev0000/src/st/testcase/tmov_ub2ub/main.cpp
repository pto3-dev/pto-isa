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
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <int32_t testKey>
void launchTmovUb2Ub(uint64_t* out, uint64_t* src, void* stream);

template <int32_t testKey>
void launchTmovUb2L1Raw(uint64_t* out, uint64_t* src, void* stream);

template <int32_t testKey>
void launchTmovUb2L1Pto(uint64_t* out, uint64_t* src, void* stream);

class TMovUb2UbTest : public testing::Test {
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

template <int32_t testKey, typename dType, int32_t rows, int32_t cols, int variant = 0>
void testTMovUb2Ub()
{
    size_t byteSize = rows * cols * sizeof(dType);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    dType *dstHost, *srcHost, *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), byteSize);
    aclrtMallocHost((void**)(&srcHost), byteSize);

    aclrtMalloc((void**)&dstDevice, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", byteSize, srcHost, byteSize);
    aclrtMemcpy(srcDevice, byteSize, srcHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    if constexpr (variant == 0) {
        launchTmovUb2Ub<testKey>(
            reinterpret_cast<uint64_t*>(dstDevice), reinterpret_cast<uint64_t*>(srcDevice), stream);
    } else if constexpr (variant == 1) {
        launchTmovUb2L1Raw<testKey>(
            reinterpret_cast<uint64_t*>(dstDevice), reinterpret_cast<uint64_t*>(srcDevice), stream);
    } else if constexpr (variant == 2) {
        launchTmovUb2L1Pto<testKey>(
            reinterpret_cast<uint64_t*>(dstDevice), reinterpret_cast<uint64_t*>(srcDevice), stream);
    }

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, byteSize, dstDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", dstHost, byteSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    size_t elementCount = byteSize / sizeof(dType);
    std::vector<dType> golden(elementCount);
    std::vector<dType> devFinal(elementCount);
    ReadFile(GetGoldenDir() + "/golden.bin", byteSize, golden.data(), byteSize);
    ReadFile(GetGoldenDir() + "/output.bin", byteSize, devFinal.data(), byteSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

// Variant A: UB-internal ND→NZ only (no L1 round-trip)
TEST_F(TMovUb2UbTest, caseA1_half_16x32) { testTMovUb2Ub<1, uint16_t, 16, 32>(); }
TEST_F(TMovUb2UbTest, caseA2_half_64x256) { testTMovUb2Ub<2, uint16_t, 64, 256>(); }
TEST_F(TMovUb2UbTest, caseA3_int32_48x72) { testTMovUb2Ub<3, int32_t, 48, 72>(); }
TEST_F(TMovUb2UbTest, caseA4_int8_32x512) { testTMovUb2Ub<4, int8_t, 32, 512>(); }
TEST_F(TMovUb2UbTest, caseA5_int8_64x96) { testTMovUb2Ub<5, int8_t, 64, 96>(); }

// Variant B: GM→UB(ND)→UB(NZ)→L1(NZ)→UB(NZ)→GM(NZ), L1→UB uses raw pto_copy_cbuf_to_ubuf
TEST_F(TMovUb2UbTest, caseB1_half_16x32_raw) { testTMovUb2Ub<1, uint16_t, 16, 32, 1>(); }
TEST_F(TMovUb2UbTest, caseB2_half_64x256_raw) { testTMovUb2Ub<2, uint16_t, 64, 256, 1>(); }
TEST_F(TMovUb2UbTest, caseB3_int32_48x72_raw) { testTMovUb2Ub<3, int32_t, 48, 72, 1>(); }
TEST_F(TMovUb2UbTest, caseB4_int8_32x512_raw) { testTMovUb2Ub<4, int8_t, 32, 512, 1>(); }
TEST_F(TMovUb2UbTest, caseB5_int8_64x96_raw) { testTMovUb2Ub<5, int8_t, 64, 96, 1>(); }

// Variant C: GM→UB(ND)→UB(NZ)→L1(NZ)→UB(NZ)→GM(NZ), L1→UB uses PTO TMOV instruction
TEST_F(TMovUb2UbTest, caseC1_half_16x32_pto) { testTMovUb2Ub<1, uint16_t, 16, 32, 2>(); }
TEST_F(TMovUb2UbTest, caseC2_half_64x256_pto) { testTMovUb2Ub<2, uint16_t, 64, 256, 2>(); }
TEST_F(TMovUb2UbTest, caseC3_int32_48x72_pto) { testTMovUb2Ub<3, int32_t, 48, 72, 2>(); }
TEST_F(TMovUb2UbTest, caseC4_int8_32x512_pto) { testTMovUb2Ub<4, int8_t, 32, 512, 2>(); }
TEST_F(TMovUb2UbTest, caseC5_int8_64x96_pto) { testTMovUb2Ub<5, int8_t, 64, 96, 2>(); }
