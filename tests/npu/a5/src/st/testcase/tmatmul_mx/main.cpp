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

template <int32_t tilingKey>
void LaunchTMATMUL_MX(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* src3, void* stream);

template <int32_t tilingKey>
void LaunchTMATMUL_MX_BIAS(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* src3, uint8_t* src4, void* stream);

template <int32_t tilingKey>
void LaunchTMATMUL_MX_PINGPONG(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* src3, uint8_t* src4, uint8_t* src5,
    void* stream);

class TMATMULMXTest : public testing::Test {
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

template <typename T>
constexpr T CeilDiv(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2;
}

template <typename T>
const T CeilAlign(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

template <typename T, typename U, typename S, bool isBias, bool isFp4, int32_t key>
void TmatmulMXTest(uint32_t M, uint32_t K, uint32_t N, uint32_t validM, uint32_t validK, uint32_t validN)
{
    uint32_t kAlign = CeilAlign<uint32_t>(validK, 64);
    size_t aFileSize = isFp4 ? CeilDiv<uint32_t>(validM * validK, 2) : validM * validK * sizeof(U);
    size_t bFileSize = isFp4 ? CeilDiv<uint32_t>(validK * validN, 2) : validK * validN * sizeof(S);
    size_t aScaleFileSize = M * CeilDiv<uint32_t>(kAlign, 32);
    size_t bScaleFileSize = N * CeilDiv<uint32_t>(kAlign, 32);
    size_t cFileSize = validM * validN * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host, *src2Host, *src3Host;
    uint8_t *dstDevice, *src0Device, *src1Device, *src2Device, *src3Device;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&src0Host), aFileSize);
    aclrtMallocHost((void**)(&src1Host), bFileSize);
    aclrtMallocHost((void**)(&src2Host), aScaleFileSize);
    aclrtMallocHost((void**)(&src3Host), bScaleFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src2Device, aScaleFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src3Device, bScaleFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    ReadFile(GetGoldenDir() + "/x1_mx_gm.bin", aScaleFileSize, src2Host, aScaleFileSize);
    ReadFile(GetGoldenDir() + "/x2_mx_gm.bin", bScaleFileSize, src3Host, bScaleFileSize);
    aclrtMemset(dstDevice, cFileSize, 0, cFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src2Device, aScaleFileSize, src2Host, aScaleFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src3Device, bScaleFileSize, src3Host, bScaleFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    size_t biasFileSize = isBias ? (1 * validN * sizeof(T)) : 0;
    uint8_t* src4Host = nullptr;
    uint8_t* src4Device = nullptr;

    if (isBias) {
        aclrtMallocHost((void**)(&src4Host), biasFileSize);
        aclrtMalloc((void**)&src4Device, biasFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

        ReadFile(GetGoldenDir() + "/bias_gm.bin", biasFileSize, src4Host, biasFileSize);
        aclrtMemcpy(src4Device, biasFileSize, src4Host, biasFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        LaunchTMATMUL_MX_BIAS<key>(dstDevice, src0Device, src1Device, src2Device, src3Device, src4Device, stream);
    } else {
        LaunchTMATMUL_MX<key>(dstDevice, src0Device, src1Device, src2Device, src3Device, stream);
    }

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(src2Device);
    aclrtFree(src3Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(src2Host);
    aclrtFreeHost(src3Host);

    if (isBias) {
        aclrtFree(src4Device);
        aclrtFreeHost(src4Host);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(cFileSize / sizeof(T));
    std::vector<T> devFinal(cFileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TMATMULMXTest, case1)
{
    uint32_t M = 128;
    uint32_t K = 64;
    uint32_t N = 64;

    TmatmulMXTest<float, uint8_t, uint8_t, false, false, 1>(M, K, N, M, K, N);
}

TEST_F(TMATMULMXTest, case2)
{
    uint32_t M = 127;
    uint32_t K = 72;
    uint32_t N = 64;

    TmatmulMXTest<float, uint8_t, uint8_t, false, false, 2>(128, 128, 64, M, K, N);
}

TEST_F(TMATMULMXTest, case3)
{
    uint32_t M = 128;
    uint32_t K = 110;
    uint32_t N = 63;

    TmatmulMXTest<float, uint8_t, uint8_t, false, false, 3>(128, 128, 64, M, K, N);
}

TEST_F(TMATMULMXTest, case4)
{
    uint32_t M = 128;
    uint32_t K = 64;
    uint32_t N = 64;

    TmatmulMXTest<float, uint8_t, uint8_t, false, true, 4>(128, 64, 64, M, K, N);
}

TEST_F(TMATMULMXTest, case5)
{
    uint32_t M = 117;
    uint32_t K = 64;
    uint32_t N = 60;

    TmatmulMXTest<float, uint8_t, uint8_t, false, true, 5>(128, 64, 64, M, K, N);
}

TEST_F(TMATMULMXTest, case6)
{
    uint32_t M = 128;
    uint32_t K = 118;
    uint32_t N = 64;

    TmatmulMXTest<float, uint8_t, uint8_t, false, true, 6>(128, 128, 64, M, K, N);
}

TEST_F(TMATMULMXTest, case7)
{
    uint32_t M = 115;
    uint32_t K = 64;
    uint32_t N = 30;

    TmatmulMXTest<float, uint8_t, uint8_t, false, true, 7>(128, 64, 64, M, K, N);
}

TEST_F(TMATMULMXTest, case8)
{
    uint32_t M = 16;
    uint32_t K = 32;
    uint32_t N = 16;

    TmatmulMXTest<float, uint8_t, uint8_t, false, false, 8>(16, 64, 32, M, K, N);
}

TEST_F(TMATMULMXTest, case9)
{
    uint32_t M = 10;
    uint32_t K = 50;
    uint32_t N = 54;

    TmatmulMXTest<float, uint8_t, uint8_t, false, false, 9>(16, 64, 64, M, K, N);
}

TEST_F(TMATMULMXTest, case10)
{
    uint32_t M = 4;
    uint32_t K = 30;
    uint32_t N = 8;

    TmatmulMXTest<float, uint8_t, uint8_t, false, true, 10>(16, 64, 64, M, K, N);
}

TEST_F(TMATMULMXTest, case11)
{
    uint32_t M = 1;
    uint32_t K = 128;
    uint32_t N = 62;

    TmatmulMXTest<float, uint8_t, uint8_t, false, true, 11>(16, 128, 64, M, K, N);
}

TEST_F(TMATMULMXTest, case12)
{
    uint32_t M = 1;
    uint32_t K = 256;
    uint32_t N = 20;

    TmatmulMXTest<float, uint8_t, uint8_t, false, false, 12>(16, 256, 64, M, K, N);
}

TEST_F(TMATMULMXTest, case13)
{
    uint32_t M = 115;
    uint32_t K = 64;
    uint32_t N = 30;

    TmatmulMXTest<float, uint8_t, uint8_t, true, false, 1>(128, 64, 32, M, K, N);
}

TEST_F(TMATMULMXTest, case14)
{
    uint32_t M = 200;
    uint32_t K = 192;
    uint32_t N = 95;

    TmatmulMXTest<float, uint8_t, uint8_t, true, false, 2>(208, 192, 128, M, K, N);
}

TEST_F(TMATMULMXTest, case15)
{
    uint32_t M = 35;
    uint32_t K = 128;
    uint32_t N = 56;

    TmatmulMXTest<float, uint8_t, uint8_t, true, true, 3>(48, 128, 64, M, K, N);
}

TEST_F(TMATMULMXTest, case16)
{
    uint32_t M = 47;
    uint32_t K = 128;
    uint32_t N = 62;

    TmatmulMXTest<float, uint8_t, uint8_t, true, true, 4>(48, 128, 64, M, K, N);
}

TEST_F(TMATMULMXTest, case17)
{
    uint32_t M = 64;
    uint32_t K = 192;
    uint32_t N = 64;

    TmatmulMXTest<float, uint8_t, uint8_t, true, false, 5>(64, 192, 64, M, K, N);
}

TEST_F(TMATMULMXTest, case18)
{
    uint32_t M = 1;
    uint32_t K = 64;
    uint32_t N = 62;

    TmatmulMXTest<float, uint8_t, uint8_t, true, true, 6>(16, 64, 64, M, K, N);
}

TEST_F(TMATMULMXTest, case19)
{
    uint32_t M = 1;
    uint32_t K = 2048;
    uint32_t N = 64;

    TmatmulMXTest<float, uint8_t, uint8_t, true, true, 7>(16, 2048, 64, M, K, N);
}

// PingPong: two back-to-back TMATMUL_MX with different scale addresses.
// out0 uses scale0 (1x), out1 uses scale1 (2x) -> out1 == 4*out0.
template <typename T, typename U, typename S, bool isFp4, int32_t key>
void TmatmulMXPingPongTest(uint32_t M, uint32_t K, uint32_t N)
{
    uint32_t kAlign = CeilAlign<uint32_t>(K, 64);
    size_t aFileSize = isFp4 ? CeilDiv<uint32_t>(M * K, 2) : M * K * sizeof(U);
    size_t bFileSize = isFp4 ? CeilDiv<uint32_t>(K * N, 2) : K * N * sizeof(S);
    size_t aScaleFileSize = M * CeilDiv<uint32_t>(kAlign, 32);
    size_t bScaleFileSize = N * CeilDiv<uint32_t>(kAlign, 32);
    size_t cFileSize = M * N * sizeof(T);
    size_t outFileSize = 2 * cFileSize; // out0 || out1

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host, *sA0Host, *sA1Host, *sB0Host, *sB1Host;
    uint8_t *dstDevice, *src0Device, *src1Device, *sA0Device, *sA1Device, *sB0Device, *sB1Device;

    aclrtMallocHost((void**)(&dstHost), outFileSize);
    aclrtMallocHost((void**)(&src0Host), aFileSize);
    aclrtMallocHost((void**)(&src1Host), bFileSize);
    aclrtMallocHost((void**)(&sA0Host), aScaleFileSize);
    aclrtMallocHost((void**)(&sA1Host), aScaleFileSize);
    aclrtMallocHost((void**)(&sB0Host), bScaleFileSize);
    aclrtMallocHost((void**)(&sB1Host), bScaleFileSize);

    aclrtMalloc((void**)&dstDevice, outFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&sA0Device, aScaleFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&sA1Device, aScaleFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&sB0Device, bScaleFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&sB1Device, bScaleFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    ReadFile(GetGoldenDir() + "/x1_mx0_gm.bin", aScaleFileSize, sA0Host, aScaleFileSize);
    ReadFile(GetGoldenDir() + "/x1_mx1_gm.bin", aScaleFileSize, sA1Host, aScaleFileSize);
    ReadFile(GetGoldenDir() + "/x2_mx0_gm.bin", bScaleFileSize, sB0Host, bScaleFileSize);
    ReadFile(GetGoldenDir() + "/x2_mx1_gm.bin", bScaleFileSize, sB1Host, bScaleFileSize);
    aclrtMemset(dstDevice, outFileSize, 0, outFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(sA0Device, aScaleFileSize, sA0Host, aScaleFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(sA1Device, aScaleFileSize, sA1Host, aScaleFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(sB0Device, bScaleFileSize, sB0Host, bScaleFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(sB1Device, bScaleFileSize, sB1Host, bScaleFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTMATMUL_MX_PINGPONG<key>(
        dstDevice, src0Device, src1Device, sA0Device, sA1Device, sB0Device, sB1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outFileSize, dstDevice, outFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, outFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(sA0Device);
    aclrtFree(sA1Device);
    aclrtFree(sB0Device);
    aclrtFree(sB1Device);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(sA0Host);
    aclrtFreeHost(sA1Host);
    aclrtFreeHost(sB0Host);
    aclrtFreeHost(sB1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    // Compare both halves (out0, out1) against the concatenated golden.
    std::vector<T> golden(outFileSize / sizeof(T));
    std::vector<T> devFinal(outFileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", outFileSize, golden.data(), outFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", outFileSize, devFinal.data(), outFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

TEST_F(TMATMULMXTest, case20_pingpong)
{
    // fp8_e4m3 x fp8_e4m3, 16x64x16. scale0=127(1x), scale1=128(2x) => out1=4*out0.
    TmatmulMXPingPongTest<float, uint8_t, uint8_t, false, 1>(16, 64, 16);
}
