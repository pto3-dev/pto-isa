/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_A1_TVECTOR_UTILS_HPP
#define PTO_NPU_A1_TVECTOR_UTILS_HPP

#include <pto/common/buffer_limits.hpp>
#include <pto/common/pto_tile.hpp>

namespace pto {
namespace detail {

template <typename First, typename Second>
PTO_INTERNAL bool A1BuffersOverlap(First& first, Second& second)
{
    const uintptr_t a = reinterpret_cast<uintptr_t>(first.data());
    const uintptr_t b = reinterpret_cast<uintptr_t>(second.data());
    return a < b + Second::Numel * sizeof(typename Second::DType) &&
           b < a + First::Numel * sizeof(typename First::DType);
}

// Layout and byte capacity are shared by floating arithmetic and raw DMA.
// Integer index buffers need identical storage rules, without granting them
// access to floating vector instructions.
template <typename TileData>
PTO_INTERNAL void CheckA1VectorStorageLayout()
{
    PTO_STATIC_ASSERT(
        TileData::isRowMajor && TileData::SFractal == SLayout::NoneBox,
        "A1 vector instructions require contiguous row-major tiles.");
    PTO_STATIC_ASSERT(
        TileData::ValidRow == TileData::Rows && TileData::ValidCol == TileData::Cols,
        "A1 vector instructions require a static full valid shape.");
    PTO_STATIC_ASSERT(
        TileData::RowStride == TileData::Cols && TileData::Cols % (32 / sizeof(typename TileData::DType)) == 0,
        "A1 vector rows must be aligned to 32 bytes without padding.");
    PTO_STATIC_ASSERT(
        TileData::Numel * sizeof(typename TileData::DType) <= PTO_UBUF_SIZE_BYTES,
        "A1 vector tile exceeds the configured UB budget.");
}

template <typename TileData>
PTO_INTERNAL void CheckA1VectorTile()
{
    PTO_STATIC_ASSERT(TileData::Loc == TileType::Vec, "A1 vector instructions require a Vec tile.");
    PTO_STATIC_ASSERT(
        (std::is_same<typename TileData::DType, half>::value || std::is_same<typename TileData::DType, float>::value),
        "A1 vector instructions require FP16 or FP32.");
    CheckA1VectorStorageLayout<TileData>();
}

// Copy raw halfword rows with explicit repeat strides (in 32-byte blocks).
// A zero input stride broadcasts one row; eight blocks copy contiguous repeats.
// Shared by column expansion and identity mask-gather, without float arithmetic.
template <unsigned Halfwords, typename Word>
PTO_INTERNAL void A1CopyBitRows(
    __ubuf__ Word* output, __ubuf__ Word* input, uint8_t repeat, uint8_t outputStride, uint8_t inputStride = 0)
{
    PTO_STATIC_ASSERT((std::is_same<Word, uint16_t>::value), "A1 bit rows require raw halfword storage.");
    constexpr unsigned full = Halfwords / 128, tail = Halfwords % 128;
    if constexpr (full > 0) {
        set_vector_mask(UINT64_MAX, UINT64_MAX);
        for (unsigned chunk = 0; chunk < full; ++chunk)
            vor(output + chunk * 128, input + chunk * 128, input + chunk * 128, repeat, 1, 1, 1, outputStride,
                inputStride, inputStride);
    }
    if constexpr (tail > 0) {
        constexpr uint64_t low = tail >= 64 ? UINT64_MAX : (uint64_t(1) << (tail % 64)) - 1;
        constexpr uint64_t high = tail > 64 ? (uint64_t(1) << (tail % 64)) - 1 : 0;
        set_vector_mask(high, low);
        vor(output + full * 128, input + full * 128, input + full * 128, repeat, 1, 1, 1, outputStride, inputStride,
            inputStride);
    }
}

} // namespace detail
} // namespace pto

#endif // PTO_NPU_A1_TVECTOR_UTILS_HPP
