/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl_types.h"

#include "comm_mpi.h"
#include "data_utils.hpp"
#include "kernel_launch.hpp"
#include "op_kernel/utils/const_args.hpp"
#include "runtime_context.hpp"
#include "tiling_builder.hpp"

extern "C" rtError_t rtSetDevice(int32_t device);
extern "C" rtError_t rtGetC2cCtrlAddr(uint64_t* addr, uint32_t* len);

namespace {

constexpr uint32_t kMxDataCacheVersion = 5U;
constexpr int kDefaultWarmupIters = 3;
constexpr int kDefaultMeasureIters = 5;
constexpr double kMicrosecondsPerSecond = 1000.0 * 1000.0;
constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;
constexpr double kNanosecondsPerSysCntTick = 1.0;
constexpr double kNanosecondsPerMicrosecond = 1000.0;

struct RunOptions {
    int warmupIters = kDefaultWarmupIters;
    int measureIters = kDefaultMeasureIters;
};

struct PerfStats {
    double avg = 0.0;
    double min = 0.0;
    double max = 0.0;
    double stddev = 0.0;
};

struct DeviceBuffer {
    void* ptr = nullptr;
    size_t bytes = 0U;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr), bytes(other.bytes)
    {
        other.ptr = nullptr;
        other.bytes = 0U;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other) {
            if (ptr != nullptr) {
                aclrtFree(ptr);
            }
            ptr = other.ptr;
            bytes = other.bytes;
            other.ptr = nullptr;
            other.bytes = 0U;
        }
        return *this;
    }

    ~DeviceBuffer()
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
};

struct RankHostInputs {
    std::vector<uint8_t> x;
    std::vector<uint8_t> weight1;
    std::vector<uint8_t> weight2;
    std::vector<uint8_t> expertIdx;
    std::vector<uint8_t> weightScale1;
    std::vector<uint8_t> weightScale2;
    std::vector<uint8_t> probs;
    std::vector<uint16_t> expectedOut;
};

struct RankDeviceBuffers {
    DeviceBuffer x;
    DeviceBuffer weight1;
    DeviceBuffer weight2;
    DeviceBuffer expertIdx;
    DeviceBuffer weightScale1;
    DeviceBuffer weightScale2;
    DeviceBuffer probs;
    DeviceBuffer out;
    DeviceBuffer expertTokenNums;
    DeviceBuffer workspace;
    DeviceBuffer tiling;
    DeviceBuffer kernelTiming;
};

DeviceBuffer MakeDeviceBuffer(size_t bytes, const void* hostSrc = nullptr)
{
    DeviceBuffer buffer;
    buffer.bytes = bytes;
    if (bytes == 0U) {
        return buffer;
    }
    if (aclrtMalloc(&buffer.ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        throw std::runtime_error("aclrtMalloc failed");
    }
    if (hostSrc != nullptr &&
        aclrtMemcpy(buffer.ptr, bytes, hostSrc, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        throw std::runtime_error("aclrtMemcpy host->device failed");
    }
    return buffer;
}

std::vector<uint16_t> BytesToU16(const std::vector<uint8_t>& bytes)
{
    if (bytes.size() % sizeof(uint16_t) != 0U) {
        throw std::runtime_error("BF16 file size is not aligned");
    }
    std::vector<uint16_t> out(bytes.size() / sizeof(uint16_t));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

int ParseEnvInt(const char* name, int defaultValue)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    try {
        return std::stoi(value);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid integer in env: ") + name);
    }
}

RunOptions LoadRunOptions()
{
    RunOptions options;
    options.warmupIters = ParseEnvInt("DISPATCH_MEGA_COMBINE_WARMUP_ITERS", kDefaultWarmupIters);
    options.measureIters = ParseEnvInt("DISPATCH_MEGA_COMBINE_MEASURE_ITERS", kDefaultMeasureIters);
    if (options.warmupIters < 0 || options.measureIters < 0) {
        throw std::runtime_error("warmup/measure iterations must be non-negative");
    }
    return options;
}

bool ParseFirstDevice(int argc, char** argv, int worldSize, int& firstDevice)
{
    firstDevice = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--first-device") != 0) {
            continue;
        }
        if (i + 1 >= argc) {
            std::cerr << "--first-device requires a non-negative device ID" << std::endl;
            return false;
        }
        const std::string text = argv[++i];
        size_t parsed = 0U;
        long long value = -1;
        try {
            value = std::stoll(text, &parsed);
        } catch (const std::exception&) {
            parsed = 0U;
        }
        if (parsed != text.size() || value < 0 || value > std::numeric_limits<int>::max()) {
            std::cerr << "invalid --first-device value: " << text << std::endl;
            return false;
        }
        firstDevice = static_cast<int>(value);
    }
    if (worldSize <= 0 || firstDevice > std::numeric_limits<int>::max() - (worldSize - 1)) {
        std::cerr << "device range overflows int: first_device=" << firstDevice << " world_size=" << worldSize
                  << std::endl;
        return false;
    }
    return true;
}

uint64_t CheckedMulU64(uint64_t lhs, uint64_t rhs, const char* name)
{
    if (lhs != 0U && rhs > UINT64_MAX / lhs) {
        throw std::runtime_error(std::string(name) + " byte size overflows uint64");
    }
    return lhs * rhs;
}

void ValidateFullPathConstraints(const CaseConfig& cfg)
{
    if (cfg.data_cache_version != kMxDataCacheVersion) {
        throw std::runtime_error("case data cache version does not match the MXFP8 contract");
    }
    if (cfg.m == 0U || cfg.k == 0U || cfg.n == 0U || cfg.topk == 0U || cfg.world_size == 0U ||
        cfg.world_size > PTO_HCCL_MAX_RANKS || cfg.max_output_size == 0U) {
        throw std::runtime_error("invalid M/K/N/topK/worldSize/maxOutputSize");
    }
    if (cfg.expert_per_rank != 4U && cfg.expert_per_rank != 8U && cfg.expert_per_rank != 16U &&
        cfg.expert_per_rank != 32U) {
        throw std::runtime_error("expert_per_rank must be one of 4, 8, 16 or 32");
    }
}

struct RankInputByteSizes {
    uint64_t x = 0U;
    uint64_t weight1 = 0U;
    uint64_t weight2 = 0U;
    uint64_t expertIdx = 0U;
    uint64_t weightScale1 = 0U;
    uint64_t weightScale2 = 0U;
    uint64_t probs = 0U;
    uint64_t outputElems = 0U;
};

RankInputByteSizes ExpectedRankInputByteSizes(const CaseConfig& cfg)
{
    const uint64_t hidden = cfg.n / 2U;
    const uint64_t xElems = CheckedMulU64(cfg.m, cfg.k, "x");
    const uint64_t routeElems = CheckedMulU64(cfg.m, cfg.topk, "route metadata");
    const uint64_t weight1Elems = CheckedMulU64(CheckedMulU64(cfg.expert_per_rank, cfg.n, "weight1"), cfg.k, "weight1");
    const uint64_t weight2Elems =
        CheckedMulU64(CheckedMulU64(cfg.expert_per_rank, cfg.k, "weight2"), hidden, "weight2");
    const uint64_t weightScale1Elems = CheckedMulU64(
        CheckedMulU64(cfg.expert_per_rank, cfg.n, "weightScale1"), cfg.k / kMegaMoeMxGroupSize, "weightScale1");
    const uint64_t weightScale2Elems = CheckedMulU64(
        CheckedMulU64(cfg.expert_per_rank, cfg.k, "weightScale2"), hidden / kMegaMoeMxGroupSize, "weightScale2");
    return {
        .x = CheckedMulU64(xElems, sizeof(uint16_t), "x"),
        .weight1 = weight1Elems,
        .weight2 = weight2Elems,
        .expertIdx = CheckedMulU64(routeElems, sizeof(int32_t), "expert_idx"),
        .weightScale1 = weightScale1Elems,
        .weightScale2 = weightScale2Elems,
        .probs = CheckedMulU64(routeElems, sizeof(float), "probs"),
        .outputElems = xElems,
    };
}

void RequireExactBytes(const char* name, size_t observed, uint64_t expected)
{
    if (expected > SIZE_MAX || observed != static_cast<size_t>(expected)) {
        throw std::runtime_error(
            std::string(name) + " byte size mismatch: observed=" + std::to_string(observed) +
            " expected=" + std::to_string(expected));
    }
}

void ValidateRankHostInputSizes(const CaseConfig& cfg, const RankHostInputs& inputs)
{
    const RankInputByteSizes sizes = ExpectedRankInputByteSizes(cfg);
    RequireExactBytes("x", inputs.x.size(), sizes.x);
    RequireExactBytes("weight1 E4M3 DN", inputs.weight1.size(), sizes.weight1);
    RequireExactBytes("weight2 E4M3 DN", inputs.weight2.size(), sizes.weight2);
    RequireExactBytes("expert_idx", inputs.expertIdx.size(), sizes.expertIdx);
    RequireExactBytes("weightScale1 E8M0 ScaleBDN", inputs.weightScale1.size(), sizes.weightScale1);
    RequireExactBytes("weightScale2 E8M0 ScaleBDN", inputs.weightScale2.size(), sizes.weightScale2);
    RequireExactBytes("probs", inputs.probs.size(), sizes.probs);
    if (inputs.expectedOut.size() != sizes.outputElems) {
        throw std::runtime_error(
            "expected_out BF16 element count mismatch: observed=" + std::to_string(inputs.expectedOut.size()) +
            " expected=" + std::to_string(sizes.outputElems));
    }
}

int32_t LoadI32(const std::vector<uint8_t>& bytes, size_t index)
{
    int32_t value = 0;
    std::memcpy(&value, bytes.data() + index * sizeof(value), sizeof(value));
    return value;
}

enum class MaskPullValidationCode : uint32_t {
    Ok = 0U,
    InvalidExpertBuffer = 1U,
    InvalidExpertId = 2U,
    ReceiveCapacity = 3U,
};

struct MaskPullLocalValidation {
    uint64_t recvRows[PTO_HCCL_MAX_RANKS] = {};
    uint64_t invalidExpertCount = 0U;
    uint64_t expertBufferBytes = 0U;
    uint32_t firstInvalidSlot = UINT32_MAX;
    int32_t firstInvalidExpert = 0;
};

struct MaskPullValidationResult {
    uint32_t code = static_cast<uint32_t>(MaskPullValidationCode::Ok);
    uint32_t rank = 0U;
    uint32_t slotOrDst = 0U;
    int32_t expert = 0;
    uint64_t observed = 0U;
    uint64_t limit = 0U;
};

void ValidateMaskPullRoutes(const CaseConfig& cfg, const RankHostInputs& inputs, int rankId, int worldSize)
{
    MaskPullLocalValidation local;
    local.expertBufferBytes = inputs.expertIdx.size();
    const uint64_t routeCount = static_cast<uint64_t>(cfg.m) * cfg.topk;
    const uint64_t expectedExpertBytes = routeCount * sizeof(int32_t);
    const uint64_t globalExpertCount = static_cast<uint64_t>(cfg.world_size) * cfg.expert_per_rank;

    if (local.expertBufferBytes == expectedExpertBytes) {
        for (uint32_t slot = 0U; slot < routeCount; ++slot) {
            const int32_t expert = LoadI32(inputs.expertIdx, slot);
            if (expert < 0 || static_cast<uint64_t>(expert) >= globalExpertCount) {
                if (local.firstInvalidSlot == UINT32_MAX) {
                    local.firstInvalidSlot = slot;
                    local.firstInvalidExpert = expert;
                }
                ++local.invalidExpertCount;
                continue;
            }
            ++local.recvRows[static_cast<uint32_t>(expert) / cfg.expert_per_rank];
        }
    }

    std::vector<MaskPullLocalValidation> gathered(rankId == 0 ? static_cast<size_t>(worldSize) : 0U);
    CommMpiGather(
        &local, static_cast<int>(sizeof(local)), COMM_MPI_CHAR,
        rankId == 0 ? static_cast<void*>(gathered.data()) : nullptr, static_cast<int>(sizeof(local)), COMM_MPI_CHAR, 0);

    MaskPullValidationResult result;
    if (rankId == 0) {
        uint64_t recvRows[PTO_HCCL_MAX_RANKS] = {};
        for (int sourceRank = 0; sourceRank < worldSize; ++sourceRank) {
            const MaskPullLocalValidation& rankLocal = gathered[static_cast<size_t>(sourceRank)];
            if (result.code == static_cast<uint32_t>(MaskPullValidationCode::Ok) &&
                rankLocal.expertBufferBytes != expectedExpertBytes) {
                result.code = static_cast<uint32_t>(MaskPullValidationCode::InvalidExpertBuffer);
                result.rank = static_cast<uint32_t>(sourceRank);
                result.observed = rankLocal.expertBufferBytes;
                result.limit = expectedExpertBytes;
            }
            if (result.code == static_cast<uint32_t>(MaskPullValidationCode::Ok) &&
                rankLocal.invalidExpertCount != 0U) {
                result.code = static_cast<uint32_t>(MaskPullValidationCode::InvalidExpertId);
                result.rank = static_cast<uint32_t>(sourceRank);
                result.slotOrDst = rankLocal.firstInvalidSlot;
                result.expert = rankLocal.firstInvalidExpert;
            }
            for (int dstRank = 0; dstRank < worldSize; ++dstRank) {
                recvRows[dstRank] += rankLocal.recvRows[dstRank];
            }
        }
        if (result.code == static_cast<uint32_t>(MaskPullValidationCode::Ok)) {
            for (int dstRank = 0; dstRank < worldSize; ++dstRank) {
                if (recvRows[dstRank] > cfg.max_output_size) {
                    result.code = static_cast<uint32_t>(MaskPullValidationCode::ReceiveCapacity);
                    result.slotOrDst = static_cast<uint32_t>(dstRank);
                    result.observed = recvRows[dstRank];
                    result.limit = cfg.max_output_size;
                    break;
                }
            }
        }
    }
    CommMpiBcast(&result, static_cast<int>(sizeof(result)), COMM_MPI_CHAR, 0);

    switch (static_cast<MaskPullValidationCode>(result.code)) {
        case MaskPullValidationCode::Ok:
            return;
        case MaskPullValidationCode::InvalidExpertBuffer:
            throw std::runtime_error("Mask Pull expert_idx byte size mismatch on rank " + std::to_string(result.rank));
        case MaskPullValidationCode::InvalidExpertId:
            throw std::runtime_error(
                "Mask Pull invalid expert id on rank " + std::to_string(result.rank) +
                " slot=" + std::to_string(result.slotOrDst) + " expert=" + std::to_string(result.expert));
        case MaskPullValidationCode::ReceiveCapacity:
            throw std::runtime_error(
                "Mask Pull receive capacity exceeded on dst rank " + std::to_string(result.slotOrDst) +
                ": rows=" + std::to_string(result.observed) + " maxOutputSize=" + std::to_string(result.limit));
    }
    throw std::runtime_error("unknown Mask Pull validation result");
}

bool ZeroWindowMemory(const StandaloneRankRuntime& runtime)
{
    const uint64_t bytes = runtime.hccl.WindowClearBytes();
    void* window = runtime.hccl.WindowClearBase();
    return aclrtMemset(window, bytes, 0, bytes) == ACL_SUCCESS;
}

void ZeroDeviceBuffer(const DeviceBuffer& buffer, const char* name)
{
    if (buffer.bytes != 0U && aclrtMemset(buffer.ptr, buffer.bytes, 0, buffer.bytes) != ACL_SUCCESS) {
        throw std::runtime_error(std::string("failed to zero ") + name);
    }
}

void ZeroWorkspaceForLaunch(const DeviceBuffer& workspace, const MegaMoeTilingData& tiling)
{
    const MegaMoeFixedGroupTiling& fixed = tiling.fixedGroupTiling;
    if (fixed.syncOffset > workspace.bytes || kMegaMoeFixedSyncBytes > workspace.bytes - fixed.syncOffset) {
        throw std::runtime_error("fixed-group sync range exceeds workspace");
    }
    const uint64_t completionBytes = MegaMoeFixedCompletionBytes(fixed.physicalAicNum);
    if (fixed.completionOffset > workspace.bytes || completionBytes > workspace.bytes - fixed.completionOffset) {
        throw std::runtime_error("completion doorbell range exceeds workspace");
    }
    auto* bytes = static_cast<uint8_t*>(workspace.ptr);
    if (fixed.syncOffset != 0U && aclrtMemset(bytes, fixed.syncOffset, 0, fixed.syncOffset) != ACL_SUCCESS) {
        throw std::runtime_error("failed to zero workspace prefix");
    }
    const uint64_t suffixOffset = fixed.syncOffset + kMegaMoeFixedSyncBytes;
    const uint64_t suffixBytes = workspace.bytes - suffixOffset;
    if (suffixBytes != 0U && aclrtMemset(bytes + suffixOffset, suffixBytes, 0, suffixBytes) != ACL_SUCCESS) {
        throw std::runtime_error("failed to zero workspace suffix");
    }
}

void PrepareLaunchState(
    const StandaloneRankRuntime& runtime, const RankDeviceBuffers& buffers, const MegaMoeTilingData& tiling)
{
    if (!ZeroWindowMemory(runtime)) {
        throw std::runtime_error("failed to zero HCCL window");
    }
    ZeroDeviceBuffer(buffers.out, "out buffer");
    ZeroDeviceBuffer(buffers.expertTokenNums, "expert_token_nums");
    ZeroDeviceBuffer(buffers.kernelTiming, "kernel timing");
    ZeroWorkspaceForLaunch(buffers.workspace, tiling);
}

std::string BuildAccuracyReport(int rankId, const AccuracyReport& report)
{
    std::ostringstream os;
    os << std::setprecision(6) << "rank=" << rankId << " max_diff=" << report.max_abs_err
       << " max_ratio=" << report.max_rel_err << " err=" << report.mismatch_count << "/" << report.err_threshold
       << " -> " << (report.pass ? "PASS" : "FAIL");
    return os.str();
}

void PrintOrderedByRank(int rankId, int worldSize, const std::string& text)
{
    for (int turn = 0; turn < worldSize; ++turn) {
        CommMpiBarrier();
        if (turn == rankId) {
            std::cout << text << std::endl;
        }
    }
    CommMpiBarrier();
}

PerfStats CalculatePerfStats(const std::vector<double>& samples)
{
    PerfStats stats;
    if (samples.empty()) {
        return stats;
    }
    stats.min = *std::min_element(samples.begin(), samples.end());
    stats.max = *std::max_element(samples.begin(), samples.end());
    stats.avg = std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
    double variance = 0.0;
    for (double sample : samples) {
        const double delta = sample - stats.avg;
        variance += delta * delta;
    }
    stats.stddev = std::sqrt(variance / static_cast<double>(samples.size()));
    return stats;
}

double ToTokensPerSecond(double tokens, double microseconds)
{
    return microseconds > 0.0 ? tokens * kMicrosecondsPerSecond / microseconds : 0.0;
}

double ToTflops(double flops, double microseconds)
{
    return microseconds > 0.0 ? flops * kMicrosecondsPerSecond / microseconds / 1.0e12 : 0.0;
}

double ToGbs(double bytes, double microseconds)
{
    return microseconds > 0.0 ? bytes * kMicrosecondsPerSecond / microseconds / kBytesPerGiB : 0.0;
}

std::vector<double> GatherMaxSamplesToRoot(const std::vector<double>& localSamples, int rankId, int worldSize)
{
    if (localSamples.empty()) {
        return {};
    }
    const int bytesPerRank = static_cast<int>(localSamples.size() * sizeof(double));
    std::vector<double> gathered(rankId == 0 ? localSamples.size() * static_cast<size_t>(worldSize) : 0U);
    CommMpiGather(
        localSamples.data(), bytesPerRank, COMM_MPI_CHAR, rankId == 0 ? gathered.data() : nullptr, bytesPerRank,
        COMM_MPI_CHAR, 0);
    if (rankId != 0) {
        return {};
    }

    std::vector<double> maxSamples(localSamples.size(), 0.0);
    for (size_t sampleIdx = 0; sampleIdx < localSamples.size(); ++sampleIdx) {
        double maxValue = gathered[sampleIdx];
        for (int rank = 1; rank < worldSize; ++rank) {
            maxValue = std::max(maxValue, gathered[static_cast<size_t>(rank) * localSamples.size() + sampleIdx]);
        }
        maxSamples[sampleIdx] = maxValue;
    }
    return maxSamples;
}

void PrintPerfSummary(
    const CaseConfig& cfg, const MegaMoeBuildResult& build, const RunOptions& options,
    const std::vector<double>& kernelSamplesUs)
{
    if (kernelSamplesUs.empty()) {
        return;
    }
    const PerfStats stats = CalculatePerfStats(kernelSamplesUs);
    const MegaMoeFixedGroupTiling& fixed = build.tiling.fixedGroupTiling;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n===============================================================\n";
    std::cout << "[KERNEL_PERF] dispatch_mega_combine\n";
    std::cout << "  shape: m=" << cfg.m << " k=" << cfg.k << " n=" << cfg.n << " topk=" << cfg.topk
              << " expert_per_rank=" << cfg.expert_per_rank << " world_size=" << cfg.world_size << '\n';
    std::cout << "  schedule: aic_num=" << cfg.aic_num << " aiv_num=" << cfg.aiv_num
              << " launch_block_dim=" << build.block_dim << " dispatch_aic=" << fixed.dispatchGroupSize
              << " gmm1_aic=" << fixed.gmm1GroupSize << " gmm2_aic=" << fixed.gmm2GroupSize << '\n';
    std::cout << "  iterations: warmup=" << options.warmupIters << " measure=" << options.measureIters << '\n';
    std::cout << "  logical work(all ranks): input_tokens=" << cfg.input_tokens_all_ranks
              << " routed_tokens=" << cfg.routed_tokens_all_ranks
              << " remote_routed_tokens=" << cfg.remote_routed_tokens_all_ranks
              << " compute_flops=" << cfg.compute_flops_all_ranks << " comm_bytes=" << cfg.comm_bytes_all_ranks << '\n';
    std::cout << "  kernel(syscnt max rank per iteration): avg=" << stats.avg << " us"
              << " min=" << stats.min << " us"
              << " max=" << stats.max << " us"
              << " std=" << stats.stddev << " us\n";
    std::cout << "    input_tokens/s=" << ToTokensPerSecond(cfg.input_tokens_all_ranks, stats.avg)
              << " routed_tokens/s=" << ToTokensPerSecond(cfg.routed_tokens_all_ranks, stats.avg)
              << " eq_compute=" << ToTflops(cfg.compute_flops_all_ranks, stats.avg) << " TFLOPS"
              << " eq_comm=" << ToGbs(cfg.comm_bytes_all_ranks, stats.avg) << " GiB/s\n";
    std::cout << "  note: equivalent compute/comm use case.json workload estimates, not hardware counters.\n";
    std::cout << "===============================================================\n" << std::endl;
}

RankHostInputs LoadRankHostInputs(const RankFileSet& files)
{
    RankHostInputs inputs;
    inputs.x = ReadBinaryFile(files.x);
    inputs.weight1 = ReadBinaryFile(files.weight1);
    inputs.weight2 = ReadBinaryFile(files.weight2);
    inputs.expertIdx = ReadBinaryFile(files.expert_idx);
    inputs.weightScale1 = ReadBinaryFile(files.weight_scale1);
    inputs.weightScale2 = ReadBinaryFile(files.weight_scale2);
    inputs.probs = ReadBinaryFile(files.probs);
    inputs.expectedOut = BytesToU16(ReadBinaryFile(files.expected_out));
    return inputs;
}

MegaMoeBuildResult BuildAndValidateTiling(const CaseConfig& cfg, const StandaloneRankRuntime& runtime, int rankId)
{
    MegaMoeBuildResult build = BuildMegaMoeTiling(cfg, runtime);
    const MegaMoeFixedGroupTiling& fixed = build.tiling.fixedGroupTiling;
    std::cout << "rank=" << rankId << " selectedTopology=aic:" << fixed.physicalAicNum
              << ",gmm1:" << fixed.gmm1GroupSize << ",gmm2:" << fixed.gmm2GroupSize
              << ",dispatch_aiv0:" << fixed.dispatchGroupSize
              << ",aiv:" << fixed.physicalAicNum * kMegaMoeFixedAivSubblocksPerPhysicalBlock << std::endl;
    return build;
}

RankDeviceBuffers AllocateRankDeviceBuffers(
    const CaseConfig& cfg, const MegaMoeBuildResult& build, const RankHostInputs& inputs)
{
    RankDeviceBuffers buffers;
    buffers.x = MakeDeviceBuffer(inputs.x.size(), inputs.x.data());
    buffers.weight1 = MakeDeviceBuffer(inputs.weight1.size(), inputs.weight1.data());
    buffers.weight2 = MakeDeviceBuffer(inputs.weight2.size(), inputs.weight2.data());
    buffers.expertIdx = MakeDeviceBuffer(inputs.expertIdx.size(), inputs.expertIdx.data());
    buffers.weightScale1 = MakeDeviceBuffer(inputs.weightScale1.size(), inputs.weightScale1.data());
    buffers.weightScale2 = MakeDeviceBuffer(inputs.weightScale2.size(), inputs.weightScale2.data());
    buffers.probs = MakeDeviceBuffer(inputs.probs.size(), inputs.probs.data());
    buffers.out = MakeDeviceBuffer(static_cast<size_t>(cfg.m) * cfg.k * sizeof(uint16_t));
    buffers.expertTokenNums = MakeDeviceBuffer(static_cast<size_t>(cfg.expert_per_rank) * sizeof(int32_t));
    buffers.workspace = MakeDeviceBuffer(build.workspace_bytes);
    buffers.tiling = MakeDeviceBuffer(sizeof(build.tiling), &build.tiling);
    buffers.kernelTiming = MakeDeviceBuffer(MegaMoeKernelTimingBytes(build.block_dim));
    return buffers;
}

MegaMoeLaunchArgs BuildLaunchArgs(const MegaMoeBuildResult& build, const RankDeviceBuffers& buffers, int rankId)
{
    uint64_t fftsAddr = 0U;
    uint32_t fftsLen = 0U;
    const rtError_t result = rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    if (result != 0) {
        std::cerr << "rank=" << rankId << " warning: rtGetC2cCtrlAddr failed ret=" << result
                  << ", continuing with ffts=0" << std::endl;
        fftsAddr = 0U;
    }

    MegaMoeLaunchArgs args;
    args.ffts = reinterpret_cast<void*>(fftsAddr);
    args.x = buffers.x.ptr;
    args.weight1 = buffers.weight1.ptr;
    args.weight2 = buffers.weight2.ptr;
    args.expert_idx = buffers.expertIdx.ptr;
    args.scale1 = buffers.weightScale1.ptr;
    args.scale2 = buffers.weightScale2.ptr;
    args.probs = buffers.probs.ptr;
    args.out = buffers.out.ptr;
    args.expert_token_nums = buffers.expertTokenNums.ptr;
    args.workspace = buffers.workspace.ptr;
    args.tiling = buffers.tiling.ptr;
    args.kernel_timing = buffers.kernelTiming.ptr;
    args.block_dim = build.block_dim;
    args.start_sync = ParseEnvInt("DISPATCH_MEGA_COMBINE_START_SYNC", 0) != 0 ? 1U : 0U;
    return args;
}

void LaunchAndSynchronize(const MegaMoeLaunchArgs& args, aclrtStream stream)
{
    launchMegaMoe(args, stream);
    const aclError result = aclrtSynchronizeStream(stream);
    if (result != ACL_SUCCESS) {
        throw std::runtime_error("stream sync failed, acl error=" + std::to_string(result));
    }
}

void RunWarmupIterations(
    const StandaloneRankRuntime& runtime, const RankDeviceBuffers& buffers, const MegaMoeTilingData& tiling,
    const MegaMoeLaunchArgs& args, int warmupIters)
{
    for (int iter = 0; iter < warmupIters; ++iter) {
        PrepareLaunchState(runtime, buffers, tiling);
        CommMpiBarrier();
        LaunchAndSynchronize(args, runtime.compute_stream);
        CommMpiBarrier();
    }
}

double ReadKernelTimingUs(const DeviceBuffer& kernelTiming, uint32_t blockDim)
{
    const size_t expectedBytes = MegaMoeKernelTimingBytes(blockDim);
    if (kernelTiming.ptr == nullptr || kernelTiming.bytes != expectedBytes || expectedBytes == 0U) {
        throw std::runtime_error("invalid kernel timing buffer");
    }
    std::vector<uint8_t> timingHost(expectedBytes);
    if (aclrtMemcpy(timingHost.data(), expectedBytes, kernelTiming.ptr, expectedBytes, ACL_MEMCPY_DEVICE_TO_HOST) !=
        ACL_SUCCESS) {
        throw std::runtime_error("failed to copy kernel timing data");
    }

    uint64_t startMin = std::numeric_limits<uint64_t>::max();
    uint64_t endMax = 0U;
    for (uint32_t block = 0U; block < blockDim; ++block) {
        for (size_t entryIdx = 0U; entryIdx < kMegaMoeKernelTimingEntriesPerBlock; ++entryIdx) {
            const size_t byteOffset = static_cast<size_t>(block) * kMegaMoeKernelTimingBytesPerBlock +
                                      entryIdx * kMegaMoeKernelTimingEntryBytes;
            MegaMoeKernelTimingEntry entry{};
            std::memcpy(&entry, timingHost.data() + byteOffset, sizeof(entry));
            if (entry.start == 0U || entry.end < entry.start) {
                throw std::runtime_error("invalid kernel timing entry");
            }
            startMin = std::min(startMin, entry.start);
            endMax = std::max(endMax, entry.end);
        }
    }
    if (startMin == std::numeric_limits<uint64_t>::max() || endMax < startMin) {
        throw std::runtime_error("kernel timing range is empty");
    }
    return static_cast<double>(endMax - startMin) * kNanosecondsPerSysCntTick / kNanosecondsPerMicrosecond;
}

double RunMeasuredIteration(
    const StandaloneRankRuntime& runtime, const RankDeviceBuffers& buffers, const MegaMoeTilingData& tiling,
    const MegaMoeLaunchArgs& args)
{
    PrepareLaunchState(runtime, buffers, tiling);
    CommMpiBarrier();
    LaunchAndSynchronize(args, runtime.compute_stream);
    CommMpiBarrier();
    return ReadKernelTimingUs(buffers.kernelTiming, args.block_dim);
}

std::vector<double> RunMeasureIterations(
    const StandaloneRankRuntime& runtime, const RankDeviceBuffers& buffers, const MegaMoeTilingData& tiling,
    const MegaMoeLaunchArgs& args, int measureIters)
{
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(measureIters));
    for (int iter = 0; iter < measureIters; ++iter) {
        samples.push_back(RunMeasuredIteration(runtime, buffers, tiling, args));
    }
    return samples;
}

std::vector<uint16_t> CopyOutputToHost(const CaseConfig& cfg, const DeviceBuffer& out)
{
    std::vector<uint16_t> actual(static_cast<size_t>(cfg.m) * cfg.k);
    const size_t bytes = actual.size() * sizeof(uint16_t);
    if (aclrtMemcpy(actual.data(), bytes, out.ptr, bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host output copy failed");
    }
    return actual;
}

bool RunOneRank(
    int rankId, int worldSize, int deviceId, uint32_t aicoreNum, const std::string& caseDir,
    const HcclRootInfo& rootInfo)
{
    StandaloneRankRuntime runtime;
    if (!InitStandaloneRankRuntime(runtime, rankId, worldSize, deviceId, rootInfo)) {
        return false;
    }

    bool ok = false;
    try {
        const RunOptions options = LoadRunOptions();
        CaseConfig cfg = LoadCaseConfig(caseDir + "/case.json");
        if (cfg.world_size != static_cast<uint32_t>(worldSize)) {
            throw std::runtime_error("case world_size does not match MPI world size");
        }
        cfg.aic_num = aicoreNum;
        cfg.aiv_num = aicoreNum * kMegaMoeFixedAivSubblocksPerPhysicalBlock;
        ValidateFullPathConstraints(cfg);

        const RankHostInputs inputs = LoadRankHostInputs(BuildRankFileSet(caseDir, rankId));
        ValidateRankHostInputSizes(cfg, inputs);
        ValidateMaskPullRoutes(cfg, inputs, rankId, worldSize);
        const MegaMoeBuildResult build = BuildAndValidateTiling(cfg, runtime, rankId);
        RankDeviceBuffers buffers = AllocateRankDeviceBuffers(cfg, build, inputs);
        const MegaMoeLaunchArgs args = BuildLaunchArgs(build, buffers, rankId);

        RunWarmupIterations(runtime, buffers, build.tiling, args, options.warmupIters);
        std::vector<double> localSamples;
        if (options.measureIters > 0) {
            localSamples = RunMeasureIterations(runtime, buffers, build.tiling, args, options.measureIters);
        }
        const std::vector<double> kernelSamples = GatherMaxSamplesToRoot(localSamples, rankId, worldSize);
        if (rankId == 0) {
            PrintPerfSummary(cfg, build, options, kernelSamples);
        }

        PrepareLaunchState(runtime, buffers, build.tiling);
        CommMpiBarrier();
        LaunchAndSynchronize(args, runtime.compute_stream);
        CommMpiBarrier();

        const std::vector<uint16_t> actual = CopyOutputToHost(cfg, buffers.out);
        WriteBinaryFile(
            caseDir + "/output_rank" + std::to_string(rankId) + ".bin", actual.data(),
            actual.size() * sizeof(uint16_t));
        const AccuracyReport report = CompareBf16File(inputs.expectedOut, actual, cfg.compare_atol, cfg.compare_rtol);
        ok = report.pass;
        PrintOrderedByRank(rankId, worldSize, BuildAccuracyReport(rankId, report));
    } catch (const std::exception& ex) {
        std::cerr << "rank=" << rankId << " error: " << ex.what() << std::endl;
        ok = false;
    }

    DestroyStandaloneRankRuntime(runtime);
    return ok;
}

} // namespace

int main(int argc, char** argv)
{
    if (!CommMpiInit(&argc, &argv)) {
        std::cerr << "MPI_Init failed" << std::endl;
        return 1;
    }

    const int rankId = CommMpiRank();
    const int worldSize = CommMpiSize();
    int firstDevice = 0;
    if (!ParseFirstDevice(argc, argv, worldSize, firstDevice)) {
        CommMpiFinalize();
        return 1;
    }
    const int deviceId = firstDevice + rankId;
    const char* caseDirEnv = std::getenv("DISPATCH_MEGA_COMBINE_CASE_DIR");
    const std::string caseDir = caseDirEnv == nullptr ? "../out" : caseDirEnv;

    if (rankId == 0) {
        std::cout << "rank/device mapping: ranks=[0," << worldSize << ") physical_devices=[" << firstDevice << ","
                  << firstDevice + worldSize << ")" << std::endl;
    }

    const aclError aclInitResult = aclInit(nullptr);
    if (aclInitResult != ACL_SUCCESS) {
        std::cerr << "rank=" << rankId << " aclInit failed ret=" << aclInitResult << std::endl;
        CommMpiFinalize();
        return 1;
    }
    if (rtSetDevice(deviceId) != 0 || aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        std::cerr << "rank=" << rankId << " failed to set device " << deviceId << std::endl;
        aclFinalize();
        CommMpiFinalize();
        return 1;
    }

    int64_t queriedAicoreNum = 0;
    const aclError queryResult =
        aclrtGetDeviceInfo(static_cast<uint32_t>(deviceId), ACL_DEV_ATTR_AICORE_CORE_NUM, &queriedAicoreNum);
    if (queryResult != ACL_SUCCESS || queriedAicoreNum <= 0 ||
        queriedAicoreNum > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        std::cerr << "rank=" << rankId << " failed to query AICore count, ret=" << queryResult
                  << " value=" << queriedAicoreNum << std::endl;
        aclrtResetDevice(deviceId);
        aclFinalize();
        CommMpiFinalize();
        return 1;
    }

    const uint32_t runtimeAicoreNum = static_cast<uint32_t>(queriedAicoreNum);
    int requestedAicoreNum = 0;
    try {
        requestedAicoreNum = ParseEnvInt("DISPATCH_MEGA_COMBINE_AICORE_NUM", 0);
    } catch (const std::exception& ex) {
        std::cerr << "rank=" << rankId << " " << ex.what() << std::endl;
        aclrtResetDevice(deviceId);
        aclFinalize();
        CommMpiFinalize();
        return 1;
    }
    if (requestedAicoreNum < 0 ||
        (requestedAicoreNum > 0 && static_cast<uint32_t>(requestedAicoreNum) > runtimeAicoreNum)) {
        std::cerr << "rank=" << rankId << " invalid requested AICore count=" << requestedAicoreNum
                  << " runtime=" << runtimeAicoreNum << std::endl;
        aclrtResetDevice(deviceId);
        aclFinalize();
        CommMpiFinalize();
        return 1;
    }
    const uint32_t effectiveAicoreNum =
        requestedAicoreNum == 0 ? runtimeAicoreNum : static_cast<uint32_t>(requestedAicoreNum);
    const A5FixedScheduleConfig* defaultSchedule = FindA5DefaultSchedule(effectiveAicoreNum);
    if (defaultSchedule == nullptr) {
        std::cerr << "rank=" << rankId << " unsupported effective A5 AICore count=" << effectiveAicoreNum
                  << "; supported counts are 28, 32, and 36" << std::endl;
        aclrtResetDevice(deviceId);
        aclFinalize();
        CommMpiFinalize();
        return 1;
    }
    std::cout << "rank=" << rankId << " runtimeAicoreNum=" << runtimeAicoreNum
              << " effectiveAicoreNum=" << effectiveAicoreNum
              << " defaultSchedule=dispatch_aiv0:" << defaultSchedule->dispatchGroupSize
              << ",gmm1:" << defaultSchedule->gmm1GroupSize << ",gmm2:" << defaultSchedule->gmm2GroupSize
              << ",aiv:" << defaultSchedule->PhysicalAivNum() << std::endl;

    HcclRootInfo rootInfo{};
    const HcclResult rootResult = rankId == 0 ? HcclGetRootInfo(&rootInfo) : HCCL_SUCCESS;
    if (rootResult != HCCL_SUCCESS) {
        std::cerr << "rank=" << rankId << " HcclGetRootInfo failed ret=" << rootResult << std::endl;
        aclrtResetDevice(deviceId);
        aclFinalize();
        CommMpiFinalize();
        return 1;
    }
    CommMpiBcast(&rootInfo, HCCL_ROOT_INFO_BYTES, COMM_MPI_CHAR, 0);
    CommMpiBarrier();

    const bool ok = RunOneRank(rankId, worldSize, deviceId, effectiveAicoreNum, caseDir, rootInfo);

    CommMpiBarrier();
    aclrtResetDevice(deviceId);
    aclFinalize();
    CommMpiFinalize();
    return ok ? 0 : 1;
}
