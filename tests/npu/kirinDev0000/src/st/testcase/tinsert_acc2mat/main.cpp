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
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <int32_t testKey>
void launchTInsertAcc2Mat(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

class TInsertAcc2MatTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    return "../" + std::string(testInfo->test_suite_name()) + "." + std::string(testInfo->name());
}

template <int32_t testKey, typename AType, typename CType>
void testTInsertAcc2Mat(int32_t m, int32_t k, int32_t n)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t aFileSize = m * k * sizeof(AType);
    size_t bFileSize = k * n * sizeof(AType);
    size_t cFileSize = m * n * sizeof(CType);
    uint8_t *outHost, *src0Host, *src1Host, *outDevice, *src0Device, *src1Device;

    aclrtMallocHost((void**)(&outHost), cFileSize);
    aclrtMallocHost((void**)(&src0Host), aFileSize);
    aclrtMallocHost((void**)(&src1Host), bFileSize);
    aclrtMalloc((void**)&outDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launchTInsertAcc2Mat<testKey>(outDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outHost, cFileSize, outDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output_z.bin", outHost, cFileSize);

    aclrtFree(outDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFreeHost(outHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<CType> golden(m * n);
    std::vector<CType> devFinal(m * n);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);
    EXPECT_TRUE(ResultCmp(golden, devFinal, 0.001f));
}

TEST_F(TInsertAcc2MatTest, case_acc2mat_1) { testTInsertAcc2Mat<1, uint16_t, uint16_t>(16, 16, 16); }
TEST_F(TInsertAcc2MatTest, case_acc2mat_2) { testTInsertAcc2Mat<2, uint16_t, uint16_t>(32, 32, 32); }
TEST_F(TInsertAcc2MatTest, case_mat2mat_1) { testTInsertAcc2Mat<3, uint16_t, uint16_t>(16, 16, 16); }
TEST_F(TInsertAcc2MatTest, case_mat2mat_2) { testTInsertAcc2Mat<4, uint16_t, uint16_t>(32, 32, 32); }

template <int32_t testKey, typename CType>
void testCbufToCbuf(int32_t m, int32_t n)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t cFileSize = m * n * sizeof(CType);
    size_t aFileSize = cFileSize;
    uint8_t *outHost, *src0Host, *outDevice, *src0Device;

    aclrtMallocHost((void**)(&outHost), cFileSize);
    aclrtMallocHost((void**)(&src0Host), aFileSize);
    aclrtMalloc((void**)&outDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launchTInsertAcc2Mat<testKey>(outDevice, src0Device, nullptr, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outHost, cFileSize, outDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output_z.bin", outHost, cFileSize);

    aclrtFree(outDevice);
    aclrtFree(src0Device);
    aclrtFreeHost(outHost);
    aclrtFreeHost(src0Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<CType> golden(m * n);
    std::vector<CType> devFinal(m * n);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);
    EXPECT_TRUE(ResultCmp(golden, devFinal, 0.001f));
}

TEST_F(TInsertAcc2MatTest, case_cbuf2cbuf_fixp_16) { testCbufToCbuf<5, uint16_t>(16, 16); }
TEST_F(TInsertAcc2MatTest, case_cbuf2cbuf_nofixp_16) { testCbufToCbuf<6, uint16_t>(16, 16); }
TEST_F(TInsertAcc2MatTest, case_mat2mat_load_16) { testCbufToCbuf<7, uint16_t>(16, 16); }
TEST_F(TInsertAcc2MatTest, case_mat2mat_ctrl_16) { testCbufToCbuf<8, uint16_t>(16, 16); }
