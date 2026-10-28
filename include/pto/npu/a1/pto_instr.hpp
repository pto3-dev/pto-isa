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
#include <pto/npu/a1/TExpand.hpp>
#include <pto/npu/a1/TImg2col.hpp>
#include <pto/npu/a1/TMatrix.hpp>
#include <pto/npu/a1/TReduce.hpp>
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

template <typename DstTile, typename SrcTile, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTile& dst, SrcTile& src, WaitEvents&... events)
{
    TSYNC(events...);
    TMOV_IMPL(dst, src);
    return {};
}

template <
    typename DstTile, typename SrcTile, AccToVecMode Mode, ReluPreMode Relu = ReluPreMode::NoRelu,
    typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTile& dst, SrcTile& src, WaitEvents&... events)
{
    PTO_STATIC_ASSERT(
        Mode == AccToVecMode::SingleModeVec0 && Relu == ReluPreMode::NoRelu,
        "A1 accumulator moves require single coupled-worker mode without fused relu.");
    TSYNC(events...);
    A1AccumulatorToVector(dst, src);
    return {};
}

template <typename AccTile, typename LeftTile, typename RightTile, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL(AccTile& dst, LeftTile& lhs, RightTile& rhs, WaitEvents&... events)
{
    TSYNC(events...);
    TMATMUL_IMPL(dst, lhs, rhs);
    return {};
}

template <typename AccTile, typename LeftTile, typename RightTile, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_ACC(AccTile& dst, AccTile& acc, LeftTile& lhs, RightTile& rhs, WaitEvents&... events)
{
    TSYNC(events...);
    TMATMUL_ACC_IMPL(dst, acc, lhs, rhs);
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

template <
    auto PrecisionType = DivAlgorithm::DEFAULT, typename DstTile, typename Src0Tile, typename Src1Tile,
    typename... WaitEvents>
PTO_INST RecordEvent TDIV(DstTile& dst, Src0Tile& src0, Src1Tile& src1, WaitEvents&... events)
{
    TSYNC(events...);
    TDIV_IMPL<PrecisionType>(dst, src0, src1);
    return {};
}

template <auto PrecisionType = ExpAlgorithm::DEFAULT, typename DstTile, typename SrcTile, typename... WaitEvents>
PTO_INST RecordEvent TEXP(DstTile& dst, SrcTile& src, WaitEvents&... events)
{
    TSYNC(events...);
    TEXP_IMPL<PrecisionType>(dst, src);
    return {};
}

template <typename DstTile, typename SrcTile, typename... WaitEvents>
PTO_INST RecordEvent TCOLSUM(DstTile& dst, SrcTile& src, WaitEvents&... events)
{
    TSYNC(events...);
    TCOLSUM_IMPL(dst, src);
    return {};
}

template <typename DstTile, typename SrcTile, typename... WaitEvents>
PTO_INST RecordEvent TCOLMAX(DstTile& dst, SrcTile& src, WaitEvents&... events)
{
    TSYNC(events...);
    TCOLMAX_IMPL(dst, src);
    return {};
}

template <typename DstTile, typename SrcTile, typename TmpTile, typename... WaitEvents>
PTO_INST RecordEvent TROWSUM(DstTile& dst, SrcTile& src, TmpTile& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    TROWSUM_IMPL(dst, src, tmp);
    return {};
}

template <typename DstTile, typename SrcTile, typename TmpTile, typename... WaitEvents>
PTO_INST RecordEvent TROWMAX(DstTile& dst, SrcTile& src, TmpTile& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    TROWMAX_IMPL(dst, src, tmp);
    return {};
}

template <typename DstTile, typename SrcTile, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPAND(DstTile& dst, SrcTile& src, WaitEvents&... events)
{
    TSYNC(events...);
    TROWEXPAND_IMPL(dst, src);
    return {};
}

template <typename DstTile, typename SrcTile, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPAND(DstTile& dst, SrcTile& src, WaitEvents&... events)
{
    TSYNC(events...);
    TCOLEXPAND_IMPL(dst, src);
    return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TEXPANDS(TileData& dst, typename TileData::DType scalar, WaitEvents&... events)
{
    TSYNC(events...);
    TEXPANDS_IMPL(dst, scalar);
    return {};
}

template <typename TileData, typename GlobalData, AtomicType Atomic = AtomicType::AtomicNone, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events)
{
    TSYNC(events...);
    TSTORE_IMPL<TileData, GlobalData, Atomic>(dst, src);
    return {};
}

template <bool NeedSetCtrl = true, typename DstTile, typename SrcTile, typename... WaitEvents>
PTO_INST RecordEvent TCVT(DstTile& dst, SrcTile& src, RoundMode mode, SaturationMode saturation, WaitEvents&... events)
{
    TSYNC(events...);
    A1ConvertVector(dst, src, mode, saturation);
    return {};
}

template <bool NeedSetCtrl = true, typename DstTile, typename SrcTile, typename... WaitEvents>
PTO_INST RecordEvent TCVT(DstTile& dst, SrcTile& src, RoundMode mode, WaitEvents&... events)
{
    return TCVT<NeedSetCtrl>(dst, src, mode, SaturationMode::OFF, events...);
}

// Same public image instruction surface as the other architectures; A1's
// implementation keeps coupled-core geometry and native LoadData3DV2 ABI.
template <typename Left, typename Conv, SetFmatrixMode Mode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent TIMG2COL(Left& dst, Conv& src, uint16_t posM = 0, uint16_t posK = 0, WaitEvents&... events)
{
    TSYNC(events...);
    A1Img2col<Left, Conv, Mode>(dst, src, posM, posK);
    return {};
}

template <typename Conv, SetFmatrixMode Mode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent SETFMATRIX(Conv& src, WaitEvents&... events)
{
    TSYNC(events...);
    detail::CheckA1ConvImage<Conv>();
    static_assert(Mode == SetFmatrixMode::FMATRIX_A_AUTO || Mode == SetFmatrixMode::FMATRIX_A_MANUAL);
    if (!detail::ValidA1ConvGeometry(src)) {
        trap();
        return {};
    }
    if constexpr (Mode == SetFmatrixMode::FMATRIX_A_AUTO)
        detail::SetA1Fmatrix(src);
    return {};
}

} // namespace pto
#endif // PTO_NPU_A1_PTO_INSTR_HPP
