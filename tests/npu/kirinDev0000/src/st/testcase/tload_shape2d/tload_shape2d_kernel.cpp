/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE inline void ZeroUbUnaligned(__ubuf__ T* dst, uint32_t elemCount)
{
    constexpr uint32_t elementsPerRepeat = CCE_VL / sizeof(T);
    __VEC_SCOPE__
    {
        RegTensor<T> vregZero;
        UnalignReg ureg;
        MaskReg pgAll = PSetTyped<T>(PAT_ALL);
        vdup(vregZero, (T)0, pgAll, MODE_ZEROING);
        uint32_t remaining = elemCount;
        __ubuf__ T* pdst = dst;
        uint16_t repeats = CeilDivision(elemCount, elementsPerRepeat);
        for (uint16_t j = 0; j < repeats; ++j) {
            uint32_t sreg = remaining > elementsPerRepeat ? elementsPerRepeat : remaining;
            vstus(ureg, sreg, vregZero, pdst, POST_UPDATE);
            remaining -= sreg;
        }
        vstas(ureg, pdst, 0, POST_UPDATE);
    }
}

template <typename T, int M, int K, int baseM, int baseK>
AICORE inline void ZeroUbNzPadding(__ubuf__ T* ubAddr)
{
    constexpr uint32_t c0 = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr uint32_t fractalNzRow = FRACTAL_NZ_ROW;
    constexpr uint32_t numMBlocks = baseM / fractalNzRow;
    constexpr uint32_t numKBlocks = baseK / c0;
    constexpr uint32_t blockSizeElems = fractalNzRow * c0;

    __ubuf__ T* base = ubAddr;

    for (uint32_t kBlock = 0; kBlock < numKBlocks; ++kBlock) {
        for (uint32_t mBlock = 0; mBlock < numMBlocks; ++mBlock) {
            __ubuf__ T* blockBase = base + (kBlock * numMBlocks + mBlock) * blockSizeElems;
            uint32_t validRow = (mBlock + 1) * fractalNzRow <= static_cast<uint32_t>(M) ?
                                    fractalNzRow :
                                    (static_cast<uint32_t>(M) > mBlock * fractalNzRow ?
                                         static_cast<uint32_t>(M) - mBlock * fractalNzRow :
                                         0);
            uint32_t validCol =
                (kBlock + 1) * c0 <= static_cast<uint32_t>(K) ?
                    c0 :
                    (static_cast<uint32_t>(K) > kBlock * c0 ? static_cast<uint32_t>(K) - kBlock * c0 : 0);

            if (validRow == 0 || validCol == 0) {
                ZeroUbUnaligned<T>(blockBase, blockSizeElems);
            } else {
                if (validCol < c0) {
                    uint32_t colPadElems = c0 - validCol;
                    for (uint32_t row = 0; row < validRow; ++row) {
                        ZeroUbUnaligned<T>(blockBase + row * c0 + validCol, colPadElems);
                    }
                }
                if (validRow < fractalNzRow) {
                    for (uint32_t row = validRow; row < fractalNzRow; ++row) {
                        ZeroUbUnaligned<T>(blockBase + row * c0, c0);
                    }
                }
            }
        }
    }
}

template <typename T, int baseM, int baseK>
AICORE inline void CopyL1ToUbNoPad(__cbuf__ T* srcMatAddr, __ubuf__ T* srcUbAddr)
{
    uint16_t blockCount = 1;
    uint16_t blockLen = baseM * baseK * sizeof(T) / 32;
    set_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);
    uint64_t fixpNzPara =
        static_cast<uint64_t>(1) | (static_cast<uint64_t>(blockLen) << 16) | (static_cast<uint64_t>(1) << 32);
    set_fixp_nz_para(fixpNzPara);
    pto_copy_cbuf_to_ubuf((__ubuf__ void*)srcUbAddr, (__cbuf__ void*)srcMatAddr, 0, blockCount, blockLen, 0, 0);
    set_fixp_nz_para(0);
    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
}

template <typename T, int M, int K, int baseM, int baseK>
AICORE inline void CopyL1ToUb(__cbuf__ T* srcMatAddr, __ubuf__ T* srcUbAddr)
{
    uint16_t blockCount = 1;
    uint16_t blockLen = baseM * baseK * sizeof(T) / 32;
    set_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);
    uint64_t fixpNzPara =
        static_cast<uint64_t>(1) | (static_cast<uint64_t>(blockLen) << 16) | (static_cast<uint64_t>(1) << 32);
    set_fixp_nz_para(fixpNzPara);
    pto_copy_cbuf_to_ubuf((__ubuf__ void*)srcUbAddr, (__cbuf__ void*)srcMatAddr, 0, blockCount, blockLen, 0, 0);
    set_fixp_nz_para(0);

    set_flag(PIPE_FIX, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_V, EVENT_ID0);
    ZeroUbNzPadding<T, M, K, baseM, baseK>(srcUbAddr);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
}

template <
    typename T, int N1, int N2, int N3, int M, int K, int WN1, int WN2, int WN3, int WN4, int WN5, int baseM, int baseK>
AICORE inline void runTLOAD_MIX_ND2NZ(__gm__ T* out, __gm__ T* src0, __gm__ T* src1)
{
    // 静态写法
    using NDValidShape = TileShape2D<T, M, K, Layout::ND>;
    using NDWholeShape = BaseShape2D<T, WN4, WN5, Layout::ND>;
    using GlobalDataSrc0 = GlobalTensor<T, NDValidShape, NDWholeShape, Layout::ND>;
    GlobalDataSrc0 src0Global(src0);

    using GlobalDataOut = GlobalTensor<
        T, pto::Shape<1, 1, 1, baseM, baseK>,
        pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, baseK, 1>, Layout::ND>;

    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        Tile<TileType::Mat, T, baseM, baseK, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>; // 大N小Z
    using TileUBData = Tile<TileType::Vec, T, baseM, baseK, BLayout::RowMajor, -1, -1>;
    TileUBData srcTile(baseM, baseK);
    TASSIGN<0x0>(srcTile);

    TileMatAData aMatTile;
    TASSIGN<0x0>(aMatTile);

    __cbuf__ T* srcMatAddr = aMatTile.data();
    __ubuf__ T* srcUbAddr = srcTile.data();
    __gm__ T* outAddr = dstGlobal.data();

    /*************************************TLOAD****************************************/
    TLOAD<TileMatAData, GlobalDataSrc0>(aMatTile, src0Global);

    CopyL1ToUb<T, M, K, baseM, baseK>(srcMatAddr, srcUbAddr);

    TSTORE(dstGlobal, srcTile); // UB -> GM : AIV
}
template <
    typename T, int N1, int N2, int N3, int M, int K, int WN1, int WN2, int WN3, int WN4, int WN5, int baseM, int baseK>
AICORE inline void runTLOAD_MIX_DN2NZ(__gm__ T* out, __gm__ T* src0, __gm__ T* src1)
{
    // 静态写法
    using DNValidShape = TileShape2D<T, M, K, Layout::DN>;
    using DNWholeShape = BaseShape2D<T, WN4, WN5, Layout::DN>;
    using GlobalDataSrc0 = GlobalTensor<T, DNValidShape, DNWholeShape, Layout::DN>;
    GlobalDataSrc0 src0Global(src0);

    using GlobalDataOut = GlobalTensor<
        T, pto::Shape<1, 1, 1, baseM, baseK>,
        pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, baseK, 1>, Layout::ND>;

    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        Tile<TileType::Mat, T, baseM, baseK, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>; // 大N小Z
    using TileUBData = Tile<TileType::Vec, T, baseM, baseK, BLayout::RowMajor, -1, -1>;
    TileUBData srcTile(baseM, baseK);
    TASSIGN<0x0>(srcTile);

    TileMatAData aMatTile;
    TASSIGN<0x0>(aMatTile);

    __cbuf__ T* srcMatAddr = aMatTile.data();
    __ubuf__ T* srcUbAddr = srcTile.data();
    __gm__ T* outAddr = dstGlobal.data();

    /*************************************TLOAD****************************************/
    TLOAD<TileMatAData, GlobalDataSrc0>(aMatTile, src0Global);

    CopyL1ToUb<T, M, K, baseM, baseK>(srcMatAddr, srcUbAddr);

    TSTORE(dstGlobal, srcTile); // UB -> GM : AIV
}

template <
    typename T, int N1, int N2, int N3, int M, int K, int WN1, int WN2, int WN3, int WN4, int WN5, int baseM, int baseK>
AICORE inline void runTLOAD_MIX_ND2ND(__gm__ T* out, __gm__ T* src0, __gm__ T* src1)
{
    // 动态写法
    using NDValidShape = TileShape2D<T, -1, -1, Layout::ND>;
    using NDWholeShape = BaseShape2D<T, -1, -1, Layout::ND>;
    NDValidShape ndValidShape(M, K);
    NDWholeShape ndWholeShape(WN4, WN5);
    using GlobalDataSrc0 = GlobalTensor<T, NDValidShape, NDWholeShape, Layout::ND>;
    GlobalDataSrc0 src0Global(src0, ndValidShape, ndWholeShape);

    using GlobalDataOut = GlobalTensor<
        T, pto::Shape<1, 1, 1, baseM, baseK>,
        pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, baseK, 1>, Layout::ND>;

    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, T, baseM, baseK, BLayout::RowMajor, M, K, SLayout::NoneBox>; // 大N小Z
    using TileUBData = Tile<TileType::Vec, T, baseM, baseK, BLayout::RowMajor, -1, -1>;
    TileUBData srcTile(baseM, baseK);
    TASSIGN<0x0>(srcTile);

    TileMatAData aMatTile;
    TASSIGN<0x0>(aMatTile);

    __cbuf__ T* srcMatAddr = aMatTile.data();
    __ubuf__ T* srcUbAddr = srcTile.data();
    __gm__ T* outAddr = dstGlobal.data();

    /*************************************TLOAD****************************************/
    TLOAD<TileMatAData, GlobalDataSrc0>(aMatTile, src0Global);

    CopyL1ToUbNoPad<T, baseM, baseK>(srcMatAddr, srcUbAddr);

    TSTORE(dstGlobal, srcTile); // UB -> GM : AIV
}

template <
    typename T, int N1, int N2, int N3, int M, int K, int WN1, int WN2, int WN3, int WN4, int WN5, int baseM, int baseK>
AICORE inline void runTLOAD_MIX_DN2DN(__gm__ T* out, __gm__ T* src0, __gm__ T* src1)
{
    // 动态写法
    using DNValidShape = TileShape2D<T, -1, -1, Layout::DN>;
    using DNWholeShape = BaseShape2D<T, -1, -1, Layout::DN>;
    DNValidShape dnValidShape(M, K);
    DNWholeShape dnWholeShape(WN4, WN5);
    using GlobalDataSrc0 = GlobalTensor<T, DNValidShape, DNWholeShape, Layout::DN>;
    GlobalDataSrc0 src0Global(src0, dnValidShape, dnWholeShape);

    using GlobalDataOut = GlobalTensor<
        T, pto::Shape<1, 1, 1, baseK, baseM>,
        pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, baseM, 1>,
        Layout::ND>; // actually is DN
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, T, baseM, baseK, BLayout::ColMajor, M, K, SLayout::NoneBox>; // 大N小Z
    using TileUBData = Tile<TileType::Vec, T, baseK, baseM, BLayout::RowMajor, -1, -1>; // DN：baseM need 32Byte aligned
    TileUBData srcTile(baseK, baseM);
    TASSIGN<0x0>(srcTile);

    TileMatAData aMatTile;
    TASSIGN<0x0>(aMatTile);

    __cbuf__ T* srcMatAddr = aMatTile.data();
    __ubuf__ T* srcUbAddr = srcTile.data();
    __gm__ T* outAddr = dstGlobal.data();

    /*************************************TLOAD****************************************/
    TLOAD<TileMatAData, GlobalDataSrc0>(aMatTile, src0Global);

    CopyL1ToUbNoPad<T, baseM, baseK>(srcMatAddr, srcUbAddr);

    TSTORE(dstGlobal, srcTile); // UB -> GM : AIV
}

template <
    typename T, int N1, int N2, int N3, int M, int K, int WN1, int WN2, int WN3, int WN4, int WN5, int baseM, int baseK>
AICORE inline void runTLOAD_MIX_NZ2NZ(__gm__ T* out, __gm__ T* src0, __gm__ T* src1)
{
    // 动态写法
    using NZValidShape = TileShape2D<T, -1, -1, Layout::NZ>;
    using NZWholeShape = BaseShape2D<T, -1, -1, Layout::NZ>;
    NZValidShape nzValidShape(N3 * M, N2 * K);
    NZWholeShape nzWholeShape(WN3 * WN4, WN2 * WN5);
    using GlobalDataSrc0 = GlobalTensor<T, NZValidShape, NZWholeShape, Layout::NZ>;
    GlobalDataSrc0 src0Global(src0, nzValidShape, nzWholeShape);

    using GlobalDataOut = GlobalTensor<
        T, pto::Shape<1, 1, 1, baseM, baseK>,
        pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, baseK, 1>, Layout::ND>;

    GlobalDataOut dstGlobal(out);
    using TileMatAData =
        Tile<TileType::Mat, T, baseM, baseK, BLayout::ColMajor, N3 * M, N2 * K, SLayout::RowMajor, 512>; // [80,48]
                                                                                                         // valid is
                                                                                                         // [N3 * M, N2
                                                                                                         // * K]

    using TileUBData = Tile<TileType::Vec, T, baseM, baseK, BLayout::RowMajor, -1, -1>;
    TileUBData srcTile(baseM, baseK);
    TASSIGN<0x0>(srcTile);

    TileMatAData aMatTile;
    TASSIGN<0x0>(aMatTile);

    __cbuf__ T* srcMatAddr = aMatTile.data();
    __ubuf__ T* srcUbAddr = srcTile.data();
    __gm__ T* outAddr = dstGlobal.data();

    /*************************************TLOAD****************************************/
    TLOAD<TileMatAData, GlobalDataSrc0>(aMatTile, src0Global);

    CopyL1ToUb<T, N3 * M, N2 * K, baseM, baseK>(srcMatAddr, srcUbAddr);

    TSTORE(dstGlobal, srcTile); // UB -> GM : AIV
}

template <
    typename T, int format, int N1, int N2, int N3, int N4, int N5, int WN1, int WN2, int WN3, int WN4, int WN5,
    int BASEM, int BASEK>
__global__ AICORE void TLOAD_MIX_KERNEL(__gm__ uint8_t* out, __gm__ uint8_t* src0, __gm__ uint8_t* src1)
{
    if constexpr (format == 0) { // ND2NZ
        runTLOAD_MIX_ND2NZ<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
            reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src0), reinterpret_cast<__gm__ T*>(src1));
    } else if constexpr (format == 1) { // DN2NZ
        runTLOAD_MIX_DN2NZ<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
            reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src0), reinterpret_cast<__gm__ T*>(src1));
    } else if constexpr (format == 2) { // ND2ND
        runTLOAD_MIX_ND2ND<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
            reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src0), reinterpret_cast<__gm__ T*>(src1));
    } else if constexpr (format == 3) { // DN2DN
        runTLOAD_MIX_DN2DN<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
            reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src0), reinterpret_cast<__gm__ T*>(src1));
    } else if constexpr (format == 4) { // NZ2NZ
        runTLOAD_MIX_NZ2NZ<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
            reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src0), reinterpret_cast<__gm__ T*>(src1));
    }
}

template <
    typename T, int format, int N1, int N2, int N3, int N4, int N5, int WN1, int WN2, int WN3, int WN4, int WN5,
    int BASEM, int BASEK>
void launchTLOADMIX(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream)
{
    TLOAD_MIX_KERNEL<T, format, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>
        <<<1, nullptr, stream>>>(out, src0, src1);
}

/********************format 0:ND2NZ 1:DN2NZ 2:ND2ND 3:DN2DN 4 NZ2NZ*****************************/
// 2:ND2ND
template void launchTLOADMIX<int8_t, 2, 1, 1, 1, 37, 126, 1, 1, 1, 37, 126, 37, 128>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<float, 2, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

// 0:ND2NZ
template void launchTLOADMIX<uint16_t, 0, 1, 1, 1, 63, 127, 1, 1, 1, 63, 127, 64, 128>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<float, 0, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<int8_t, 0, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<uint16_t, 0, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<uint16_t, 0, 1, 1, 1, 33, 99, 1, 1, 1, 64, 128, 48, 112>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<int8_t, 0, 1, 1, 1, 59, 119, 1, 1, 1, 64, 128, 64, 128>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

// 3:DN2DN
template void launchTLOADMIX<float, 3, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<int8_t, 3, 1, 1, 1, 37, 126, 1, 1, 1, 37, 126, 64, 126>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

// 4.NZ2NZ
template void launchTLOADMIX<uint16_t, 4, 1, 10, 8, 16, 16, 1, 11, 9, 16, 16, 128, 160>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<int8_t, 4, 1, 8, 4, 16, 32, 1, 9, 4, 16, 32, 80, 256>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

template void launchTLOADMIX<int64_t, 2, 1, 1, 1, 59, 119, 1, 1, 1, 59, 124, 59, 120>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
