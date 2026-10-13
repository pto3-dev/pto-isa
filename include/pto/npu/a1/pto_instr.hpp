/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_A1_PTO_INSTR_HPP
#define PTO_NPU_A1_PTO_INSTR_HPP

#include <pto/common/arch_cce_intrinsic.hpp>
#include <pto/common/tassign_check.hpp>
#include <pto/npu/a1/TSync.hpp>
#include <pto/npu/a1/TVector.hpp>
// This assignment helper only assigns typed addresses and has no A2/A3 instructions.
#include <pto/npu/a2a3/TAssign.hpp>

namespace pto {

template <typename T, typename AddrType>
PTO_INST void TASSIGN(T& obj, AddrType addr)
{
    TASSIGN_IMPL(obj, addr);
}

template <std::size_t Addr, typename T>
PTO_INST std::enable_if_t<is_tile_data_v<T> || is_conv_tile_v<T>> TASSIGN(T& obj)
{
    (void)detail::tassign_static_check<std::remove_cv_t<T>, Addr>{};
    TASSIGN(obj, Addr);
}

template <Op OpCode>
PTO_INST void TSYNC()
{
    TSYNC_IMPL<OpCode>();
}

template <typename... WaitEvents>
PTO_INST void TSYNC(WaitEvents&... events)
{
    WaitAllEvents(events...);
}

template <typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TLOAD(TileData& dst, GlobalData& src, WaitEvents&... events)
{
    TSYNC(events...);
    TLOAD_IMPL(dst, src);
    return {};
}

template <typename DstTile, typename Src0Tile, typename Src1Tile, typename... WaitEvents>
PTO_INST RecordEvent TADD(DstTile& dst, Src0Tile& src0, Src1Tile& src1, WaitEvents&... events)
{
    TSYNC(events...);
    TADD_IMPL(dst, src0, src1);
    return {};
}
template <typename DstTile, typename Src0Tile, typename Src1Tile, typename... WaitEvents>
PTO_INST RecordEvent TSUB(DstTile& dst, Src0Tile& src0, Src1Tile& src1, WaitEvents&... events)
{
    TSYNC(events...);
    TSUB_IMPL(dst, src0, src1);
    return {};
}
template <typename DstTile, typename Src0Tile, typename Src1Tile, typename... WaitEvents>
PTO_INST RecordEvent TMUL(DstTile& dst, Src0Tile& src0, Src1Tile& src1, WaitEvents&... events)
{
    TSYNC(events...);
    TMUL_IMPL(dst, src0, src1);
    return {};
}

template <typename TileData, typename GlobalData, AtomicType Atomic = AtomicType::AtomicNone, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events)
{
    TSYNC(events...);
    TSTORE_IMPL<TileData, GlobalData, Atomic>(dst, src);
    return {};
}

} // namespace pto
#endif // PTO_NPU_A1_PTO_INSTR_HPP
