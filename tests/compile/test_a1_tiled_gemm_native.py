# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU-first exact checks for A1 multi-fractal K and native GEMM accumulation."""

import ctypes
import shutil
import subprocess
from pathlib import Path
import pytest

torch = pytest.importorskip("torch")
simpler = pytest.importorskip("simpler_setup")


@pytest.fixture(scope="module", autouse=True)
def require_310p():
    try:
        acl = ctypes.CDLL("libascendcl.so")
    except OSError:
        pytest.skip("Ascend runtime unavailable")
    acl.aclrtGetSocName.restype = ctypes.c_char_p
    acl.aclrtGetSocName.argtypes = []
    soc = acl.aclrtGetSocName()
    if not soc or not soc.startswith(b"Ascend310P"):
        pytest.skip("Requires Ascend310P hardware")


def _run_cpu_oracle(root, tmp_path, source):
    compiler = shutil.which("clang++-19") or shutil.which("g++-15") or shutil.which("g++-14")
    if compiler is None:
        pytest.skip("CPU oracle requires Clang19 or GCC>=14")
    cpu = tmp_path / "cpu.cpp"
    cpu.write_text(source)
    exe = tmp_path / "cpu-gemm"
    subprocess.run(
        [compiler, "-std=c++20", "-O1", "-D__CPU_SIM", f"-I{root / 'include'}", str(cpu), "-o", str(exe)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(exe)], check=True, capture_output=True, text=True)


@pytest.mark.parametrize(
    "rows,reduction,zz,gap,accumulate,legacy,ub_input",
    [
        (16, 16, False, 0, False, True, False),
        (16, 16, True, 0, True, False, False),
        (32, 32, True, 32, False, False, False),
        (32, 64, False, 0, True, False, False),
        (112, 144, True, 0, False, False, False),
        (224, 144, False, 32, True, False, False),
        (32, 512, True, 32, True, False, False),
        (16, 1024, False, 0, False, False, False),
        (16, 2048, False, 0, True, False, False),
        (64, 256, False, 0, False, False, False),
        (2048, 16, False, 0, True, True, False),
        (16, 32, False, 1048576, False, False, False),
        (16, 16, False, 0, False, True, True),
        (32, 32, True, 32, False, False, True),
        (32, 64, False, 0, True, False, True),
        (112, 144, True, 0, False, False, True),
        (224, 144, False, 32, True, False, True),
        (16, 2048, False, 0, False, False, True),
    ],
)
def test_a1_tiled_gemm_native(tmp_path, rows, reduction, zz, gap, accumulate, legacy, ub_input):
    root = Path(__file__).resolve().parents[2]
    source = (root / "tests/compile/a1_tiled_gemm.cpp").read_text()
    definitions = f"#define PTO_TEST_M {rows}\n#define PTO_TEST_K {reduction}\n#define PTO_TEST_GAP {gap}\n"
    for name, enabled in [("ZZ", zz), ("ACCUMULATE", accumulate), ("LEGACY_LEFT", legacy), ("UB_INPUT", ub_input)]:
        if enabled:
            definitions += f"#define PTO_TEST_{name}\n"
    _run_cpu_oracle(root, tmp_path, definitions + source)
    source = source.replace('extern "C" __global__ AICORE void A1TiledGemm', "AICORE void A1TiledGemm")
    wrapper = r"""
#include "tensor.h"
extern "C" __global__ AICORE void NativeTiledGemm(__gm__ int64_t* args) {
    auto* a=reinterpret_cast<__gm__ Tensor*>(args[0]);
    auto* b=reinterpret_cast<__gm__ Tensor*>(args[1]);
    auto* out=reinterpret_cast<__gm__ Tensor*>(args[2]);
    A1TiledGemm(reinterpret_cast<__gm__ float*>(out->buffer.addr)+16,
                reinterpret_cast<__gm__ half*>(a->buffer.addr)+16,
                reinterpret_cast<__gm__ half*>(b->buffer.addr)+16);
}
"""
    path = tmp_path / "native.cpp"
    path.write_text(definitions + source + wrapper)
    image = simpler.KernelCompiler("a1").compile_single_launch(str(path), str(root))
    with simpler.A1SingleLaunch(image, device_id=0) as runner:
        for seed in (853, 857):
            generator = torch.Generator().manual_seed(seed)
            a = (torch.randint(-8, 9, (rows, reduction), generator=generator).float() / 16).half()
            b = (torch.randint(-8, 9, (reduction, 16), generator=generator).float() / 16).half()

            def pack(x, stride):
                guarded = torch.full((x.shape[0] * stride + 32,), 7, dtype=torch.float16)
                guarded[16:-16].reshape(x.shape[0], stride)[:, : x.shape[1]] = x
                return guarded

            ga = pack(a, reduction + gap)
            gb = pack(b, 16 + gap)
            output = torch.full((rows * 16 + 32,), 7, dtype=torch.float32)
            runner(ga, gb, output)
            # Binary fractions keep every product and partial exactly representable;
            # this checks every output without a tolerance or a reduction-order assumption.
            expected = a.float() @ b.float()
            if ub_input:
                expected *= 2
            if accumulate:
                expected *= 2
            assert torch.equal(output[16:-16].reshape(rows, 16), expected)
            assert torch.all(output[:16] == 7) and torch.all(output[-16:] == 7)
            assert torch.equal(ga, pack(a, reduction + gap)) and torch.equal(gb, pack(b, 16 + gap))


@pytest.mark.parametrize(
    "rows,reduction,legacy,diagnostic", [(224, 160, False, "within 64 KiB"), (32, 32, True, "legacy NZ layout")]
)
def test_a1_tiled_gemm_rejects_unsupported(tmp_path, rows, reduction, legacy, diagnostic):
    root = Path(__file__).resolve().parents[2]
    source = (root / "tests/compile/a1_tiled_gemm.cpp").read_text()
    definitions = f"#define PTO_TEST_M {rows}\n#define PTO_TEST_K {reduction}\n"
    if legacy:
        definitions += "#define PTO_TEST_LEGACY_LEFT\n"
    path = tmp_path / "unsupported.cpp"
    path.write_text(definitions + source)
    with pytest.raises(RuntimeError, match=diagnostic):
        simpler.KernelCompiler("a1").compile_single_launch(str(path), str(root))


@pytest.mark.parametrize("pos_m,accumulate", [(0, False), (16, True)])
def test_a1_img2col_to_tiled_gemm(tmp_path, pos_m, accumulate):
    root = Path(__file__).resolve().parents[2]
    source = (root / "tests/compile/a1_tiled_gemm.cpp").read_text()
    definitions = "#define PTO_TEST_M 32\n#define PTO_TEST_K 144\n"
    if accumulate:
        definitions += "#define PTO_TEST_ACCUMULATE\n"
    image_types = """
using Image = pto::ConvTile<pto::TileType::Mat, half, 1024, pto::Layout::NC1HWC0,
                           pto::ConvTileShape<1,1,8,8,16>>;
using ImageGlobal = pto::GlobalTensor<half,pto::Shape<1,1,8,8,16>,
                                    pto::Stride<1024,1024,128,16,1>,pto::Layout::NC1HWC0>;
"""
    source = source.replace("#include <pto/pto-inst.hpp>", "#include <pto/pto-inst.hpp>\n" + image_types)
    source = source.replace(
        "pto::TLOAD(ma, ga);",
        """
    Image image;
    pto::TASSIGN<0>(image);
#ifdef __CPU_SIM
    image.data()=a;
#else
    ImageGlobal globalImage(a);
    pto::TLOAD(image,globalImage);
#endif
""",
    )
    source = source.replace(
        "pto::TMOV(left, ma);",
        f"""
    image.SetFmapH(8); image.SetFmapW(8);
    image.SetFilterH(3); image.SetFilterW(3);
    image.SetStrideH(1); image.SetStrideW(1); image.SetChannelSize(16);
    const uint8_t pads[4]={{1,1,1,1}};
    image.SetPadListArray(pads);
    pto::TIMG2COL<Left,Image,pto::SetFmatrixMode::FMATRIX_A_AUTO>(left,image,{pos_m},0);
""",
    )
    source = source[: source.index("#ifdef __CPU_SIM\nint main()")]
    cpu_main = r"""
#ifdef __CPU_SIM
int main() {
    std::vector<half> a(1024),b(144*16);
    std::vector<float> output(32*16+16,7);
    for(int i=0;i<1024;++i) a[i]=half((i%17-8)/16.0f);
    for(int i=0;i<144*16;++i) b[i]=half((i%13-6)/16.0f);
    A1TiledGemm(output.data(),a.data(),b.data());
    for(int row=0;row<32;++row) {
        for(int col=0;col<16;++col) {
            float expected=0;
            for(int tap=0;tap<9;++tap) {
                int m=row+IMAGE_POS_M;
                int inputRow=m/8+tap/3-1,inputCol=m%8+tap%3-1;
                if(inputRow<0||inputRow>=8||inputCol<0||inputCol>=8) continue;
                for(int c=0;c<16;++c)
                    expected+=float(a[(inputRow*8+inputCol)*16+c])*float(b[(tap*16+c)*16+col]);
            }
#ifdef PTO_TEST_ACCUMULATE
            expected*=2;
#endif
            if(output[row*16+col]!=expected) return 1;
        }
    }
    for(int i=32*16;i<32*16+16;++i) if(output[i]!=7) return 2;
    return 0;
}
#endif
""".replace("IMAGE_POS_M", str(pos_m))
    _run_cpu_oracle(root, tmp_path, definitions + source + cpu_main)
    source = source.replace('extern "C" __global__ AICORE void A1TiledGemm', "AICORE void A1TiledGemm")
    wrapper = r"""
#include "tensor.h"
extern "C" __global__ AICORE void NativeImageGemm(__gm__ int64_t* args) {
    auto* a=reinterpret_cast<__gm__ Tensor*>(args[0]);
    auto* b=reinterpret_cast<__gm__ Tensor*>(args[1]);
    auto* out=reinterpret_cast<__gm__ Tensor*>(args[2]);
    A1TiledGemm(reinterpret_cast<__gm__ float*>(out->buffer.addr)+16,
                reinterpret_cast<__gm__ half*>(a->buffer.addr)+16,
                reinterpret_cast<__gm__ half*>(b->buffer.addr)+16);
}
"""
    path = tmp_path / "native_image_gemm.cpp"
    path.write_text(definitions + source + wrapper)
    image = simpler.KernelCompiler("a1").compile_single_launch(str(path), str(root))
    with simpler.A1SingleLaunch(image, device_id=0) as runner:
        for seed in (877, 879):
            generator = torch.Generator().manual_seed(seed)
            x = (torch.randint(-8, 9, (1, 16, 8, 8), generator=generator).float() / 16).half()
            b = (torch.randint(-8, 9, (144, 16), generator=generator).float() / 16).half()
            ga = torch.full((1024 + 32,), 7, dtype=torch.float16)
            ga[16:-16] = x.permute(0, 2, 3, 1).contiguous().flatten()
            gb = torch.full((144 * 16 + 32,), 7, dtype=torch.float16)
            gb[16:-16] = b.flatten()
            output = torch.full((32 * 16 + 32,), 7, dtype=torch.float32)
            patches = torch.nn.functional.unfold(x.float(), 3, padding=1)
            patches = patches.reshape(1, 16, 3, 3, 64).permute(0, 4, 2, 3, 1).contiguous().reshape(64, 144)
            expected = patches[pos_m : pos_m + 32] @ b.float()
            if accumulate:
                expected *= 2
            runner(ga, gb, output)
            assert torch.equal(output[16:-16].reshape(32, 16), expected)
            assert torch.all(output[:16] == 7) and torch.all(output[-16:] == 7)
            assert torch.equal(ga[16:-16], x.permute(0, 2, 3, 1).contiguous().flatten())
            assert torch.equal(gb[16:-16], b.flatten())
            assert torch.all(ga[:16] == 7) and torch.all(ga[-16:] == 7)
            assert torch.all(gb[:16] == 7) and torch.all(gb[-16:] == 7)


@pytest.mark.parametrize(
    "dtype,rows,columns,valid_rows,dst_rows,diagnostic",
    [
        ("float", 16, 16, 16, 16, "FP16 source data"),
        ("half", 32, 32, 16, 32, "static full valid shape"),
        ("half", 16, 16, 16, 32, "matching full shapes"),
        ("half", 64, 2048, 64, 64, "UB budget"),
    ],
)
def test_a1_vec_to_mat_rejects_unsupported(tmp_path, dtype, rows, columns, valid_rows, dst_rows, diagnostic):
    root = Path(__file__).resolve().parents[2]
    source = f"""
#include <pto/pto-inst.hpp>
using Source = pto::Tile<pto::TileType::Vec, {dtype}, {rows}, {columns},
                         pto::BLayout::RowMajor, {valid_rows}, {columns}>;
using Destination = pto::Tile<pto::TileType::Mat, half, {dst_rows}, {columns},
                              pto::BLayout::RowMajor, {dst_rows}, {columns},
                              pto::SLayout::RowMajor, 512>;
extern "C" __global__ AICORE void Unsupported(__gm__ int64_t* args) {{
    Source source;
    Destination destination;
    pto::TASSIGN<0>(source);
    pto::TASSIGN<0>(destination);
    pto::TMOV(destination, source);
}}
"""
    path = tmp_path / "unsupported.cpp"
    path.write_text(source)
    with pytest.raises(RuntimeError, match=diagnostic):
        simpler.KernelCompiler("a1").compile_single_launch(str(path), str(root))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
