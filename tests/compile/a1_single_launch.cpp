/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <acl/acl.h>
#include <runtime/rt.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

void Check(int status, const char* operation)
{
    if (status != 0)
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(status));
}

struct Session {
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    bool initialized = false;
    ~Session()
    {
        if (stream) {
            aclrtSynchronizeStreamWithTimeout(stream, 5000);
            aclrtDestroyStream(stream);
        }
        if (context)
            aclrtDestroyContext(context);
        if (initialized)
            aclFinalize();
    }
};

struct DeviceBuffer {
    void* address = nullptr;
    explicit DeviceBuffer(size_t bytes)
    {
        Check(aclrtMalloc(&address, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc");
    }
    ~DeviceBuffer()
    {
        if (address)
            aclrtFree(address);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct StreamDrain {
    aclrtStream stream;
    ~StreamDrain() { aclrtSynchronizeStreamWithTimeout(stream, 5000); }
};

struct Kernel {
    void* handle = nullptr;
    ~Kernel()
    {
        if (handle)
            rtDevBinaryUnRegister(handle);
    }
};

uint16_t HalfInteger(int value)
{
    if (!value)
        return 0;
    unsigned magnitude = value < 0 ? -value : value;
    unsigned exponent = 0;
    for (unsigned n = magnitude; n > 1; n >>= 1)
        ++exponent;
    return (value < 0 ? 0x8000 : 0) | ((15 + exponent) << 10) | ((magnitude << (10 - exponent)) & 0x3ff);
}

int main(int argc, char** argv)
{
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: a1_single_launch KERNEL.o LOGICAL_DEVICE [ELEMENTS]\n";
        return 2;
    }
    try {
        std::ifstream input(argv[1], std::ios::binary);
        if (!input)
            throw std::runtime_error("cannot open kernel object");
        std::vector<char> object{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        if (object.empty())
            throw std::runtime_error("empty kernel object");
        Session session;
        Check(aclInit(nullptr), "aclInit");
        session.initialized = true;
        const int device = std::stoi(argv[2]);
        Check(aclrtCreateContext(&session.context, device), "aclrtCreateContext");
        Check(aclrtCreateStream(&session.stream), "aclrtCreateStream");
        const size_t count = argc == 4 ? std::stoul(argv[3]) : 16 * 128;
        if (count == 0 || count > 65536)
            throw std::runtime_error("element count must be in 1..65536");
        const size_t bytes = count * sizeof(uint16_t);
        std::vector<uint16_t> lhs(count), rhs(count), result(count, 0x7e00), expected(count);
        for (size_t i = 0; i < count; ++i) {
            int a = static_cast<int>(i % 17) - 8;
            int b = static_cast<int>((i * 3) % 17) - 8;
            lhs[i] = HalfInteger(a);
            rhs[i] = HalfInteger(b);
            expected[i] = HalfInteger(a + b);
        }
        DeviceBuffer a(bytes), b(bytes), c(bytes);
        Check(aclrtMemcpy(a.address, bytes, lhs.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy lhs");
        Check(aclrtMemcpy(b.address, bytes, rhs.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy rhs");
        Check(aclrtMemcpy(c.address, bytes, result.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "initialize result");
        Kernel kernel;
        rtDevBinary_t binary{};
        binary.magic = RT_DEV_BINARY_MAGIC_ELF;
        binary.data = object.data();
        binary.length = object.size();
        Check(rtRegisterAllKernel(&binary, &kernel.handle), "rtRegisterAllKernel");
        StreamDrain drain{session.stream};
        struct Args {
            uint64_t out, lhs, rhs;
        } args{
            reinterpret_cast<uint64_t>(c.address), reinterpret_cast<uint64_t>(a.address),
            reinterpret_cast<uint64_t>(b.address)};
        static_assert(std::is_standard_layout_v<Args> && std::is_trivially_copyable_v<Args>);
        rtArgsEx_t launchArgs{};
        launchArgs.args = &args;
        launchArgs.argsSize = sizeof(args);
        rtTaskCfgInfo_t config{};
        config.schemMode = RT_SCHEM_MODE_BATCH;
        Check(
            rtKernelLaunchWithHandleV2(kernel.handle, 0, 1, &launchArgs, nullptr, session.stream, &config),
            "kernel launch");
        Check(aclrtSynchronizeStreamWithTimeout(session.stream, 5000), "stream synchronize");
        Check(aclrtMemcpy(result.data(), bytes, c.address, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy result");
        size_t mismatches = 0;
        for (size_t i = 0; i < count; ++i) {
            if (result[i] != expected[i]) {
                if (mismatches < 8)
                    std::cerr << "index=" << i << " actual=" << result[i] << " expected=" << expected[i] << '\n';
                ++mismatches;
            }
        }
        std::cout << "device=" << device << " elements=" << count << " dtype=fp16 mismatches=" << mismatches << '/'
                  << count << '\n';
        return mismatches ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
