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
#include <cstring>
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

template <typename Word>
Word IntegerBits(int value)
{
    if constexpr (std::is_same_v<Word, uint16_t>) {
        return HalfInteger(value);
    } else {
        static_assert(std::is_same_v<Word, uint32_t>);
        float scalar = static_cast<float>(value);
        Word bits;
        static_assert(sizeof(bits) == sizeof(scalar));
        std::memcpy(&bits, &scalar, sizeof(bits));
        return bits;
    }
}

template <typename Word>
int RunVector(int argc, char** argv, const char* dtype, const std::string& operation)
{
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
        const size_t count = argc >= 4 ? std::stoul(argv[3]) : 16 * 128;
        if (count == 0 || count > 65536)
            throw std::runtime_error("element count must be in 1..65536");
        const size_t rows = argc == 9 ? std::stoul(argv[6]) : 1;
        const size_t cols = argc == 9 ? std::stoul(argv[7]) : count;
        const size_t rowStride = argc == 9 ? std::stoul(argv[8]) : cols;
        if (!rows || !cols || rowStride < cols || rows > 16777216 / rowStride || count != rows * cols)
            throw std::runtime_error("invalid row-transfer geometry or storage exceeds 16M elements");
        const size_t storageCount = rows * rowStride;
        const size_t bytes = storageCount * sizeof(Word);
        constexpr Word poison = sizeof(Word) == 2 ? 0x7e00 : 0x7fc00000;
        std::vector<Word> lhs(storageCount), rhs(storageCount), result(storageCount, poison),
            expected(storageCount, poison);
        for (size_t i = 0; i < storageCount; ++i) {
            int a = static_cast<int>(i % 17) - 8;
            int b = static_cast<int>((i * 3) % 17) - 8;
            lhs[i] = IntegerBits<Word>(a);
            rhs[i] = IntegerBits<Word>(b);
            if (i % rowStride >= cols)
                continue;
            const int value =
                operation == "copy" ? a : (operation == "sub" ? a - b : (operation == "mul" ? a * b : a + b));
            expected[i] = IntegerBits<Word>(value);
            if (operation == "mul" && value == 0 && ((a < 0) != (b < 0)))
                expected[i] |= Word(1) << (sizeof(Word) * 8 - 1);
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
        for (size_t i = 0; i < storageCount; ++i) {
            if (result[i] != expected[i]) {
                if (mismatches < 8)
                    std::cerr << "index=" << i << " actual=" << result[i] << " expected=" << expected[i] << '\n';
                ++mismatches;
            }
        }
        std::cout << "device=" << device << " elements=" << count << " dtype=" << dtype << " mismatches=" << mismatches
                  << '/' << storageCount << " operation=" << operation << " rows=" << rows << " cols=" << cols
                  << " row_stride=" << rowStride << '\n';
        return mismatches ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

int main(int argc, char** argv)
{
    if (argc < 3 || (argc > 6 && argc != 9)) {
        std::cerr << "usage: a1_single_launch KERNEL.o LOGICAL_DEVICE [ELEMENTS [fp16|fp32 [add|sub|mul|copy [ROWS "
                     "COLS ROW_STRIDE]]]]\n";
        return 2;
    }
    const std::string dtype = argc >= 5 ? argv[4] : "fp16";
    const std::string operation = argc >= 6 ? argv[5] : "add";
    if (operation != "add" && operation != "sub" && operation != "mul" && operation != "copy") {
        std::cerr << "operation must be add, sub, mul or copy\n";
        return 2;
    }
    if (dtype == "fp16")
        return RunVector<uint16_t>(argc, argv, "fp16", operation);
    if (dtype == "fp32")
        return RunVector<uint32_t>(argc, argv, "fp32", operation);
    std::cerr << "dtype must be fp16 or fp32\n";
    return 2;
}
