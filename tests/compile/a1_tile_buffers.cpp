/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>
#if defined(__CPU_SIM)
#include <pto/common/cpu_stub.hpp>
#endif
#include <pto/common/arch_capability.hpp>
#include <pto/common/buffer_limits.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/common/tassign_check.hpp>

static_assert(PTO_UBUF_SIZE_BYTES == 128 * 1024);
static_assert(PTO_CBUF_SIZE_BYTES == 512 * 1024);
static_assert(PTO_L0A_SIZE_BYTES == 64 * 1024);
static_assert(PTO_L0B_SIZE_BYTES == 64 * 1024);
static_assert(PTO_L0C_SIZE_BYTES == 256 * 1024);
static_assert(PTO_BIAS_SIZE_BYTES == 0);
static_assert(PTO_FBUF_SIZE_BYTES == 0);
static_assert(PTO_SCALELEFT_SIZE_BYTES == 0);
static_assert(PTO_SCALERIGHT_SIZE_BYTES == 0);

using A1VectorTile = pto::Tile<pto::TileType::Vec, half, 16, 128>;
static_assert(A1VectorTile::Rows == 16);
static_assert(A1VectorTile::Cols == 128);
static_assert(A1VectorTile::RowStride == 128);

// The last aligned 4 KiB allocation must fit, while moving it one DMA block
// further must fail the real TASSIGN bound checker on the device compiler.
#if defined(PTO_A1_TEST_OVERFLOW)
using A1BoundaryAllocation = pto::detail::tassign_static_check<A1VectorTile, 124 * 1024 + 32>;
#else
using A1BoundaryAllocation = pto::detail::tassign_static_check<A1VectorTile, 124 * 1024>;
#endif
static_assert(sizeof(A1BoundaryAllocation) > 0);

AICORE int CheckA1TileMetadata()
{
    A1VectorTile tile;
    return tile.GetValidRow() == 16 && tile.GetValidCol() == 128 ? 0 : 1;
}

#if defined(__CPU_SIM)
int main() { return CheckA1TileMetadata(); }
#endif
