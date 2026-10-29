# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""A1 public NC1HWC0 image DMA and LoadData3DV2 im2col layout checks."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
simpler = pytest.importorskip("simpler_setup")


@pytest.fixture(scope="module", autouse=True)
def require_310p_hardware():
    try:
        acl = ctypes.CDLL("libascendcl.so")
    except OSError:
        pytest.skip("Ascend runtime is unavailable")
    acl.aclrtGetSocName.restype = ctypes.c_char_p
    acl.aclrtGetSocName.argtypes = []
    soc = acl.aclrtGetSocName()
    if not soc or not soc.startswith(b"Ascend310P"):
        pytest.skip("Native im2col requires Ascend310P hardware")


@pytest.mark.parametrize(
    "filter_size,stride,pad,dilation,rows,cols,pos_m,pos_k,pad_value,manual,height,width,row_stride,channel_offset",
    [
        (1, 1, 0, 1, 32, 16, 0, 0, 0.0, False, 8, 8, 0, 0),
        (1, 1, 0, 1, 32, 16, 16, 16, 0.0, True, 8, 8, 0, 0),
        (3, 1, 1, 1, 32, 144, 0, 0, 0.0, False, 8, 8, 0, 0),
        (3, 1, 1, 1, 32, 144, 16, 144, 0.25, True, 8, 8, 0, 0),
        (3, 2, 1, 1, 16, 16, 0, 64, 0.0, False, 8, 8, 0, 0),
        (3, 1, 1, 2, 32, 144, 0, 0, -0.5, False, 8, 8, 0, 0),
        (3, 1, 1, 1, 32, 144, 16, 144, 0.25, True, 8, 8, 64, 16),
        (3, 1, 1, 1, 32, 144, 0, 0, 0.0, False, 64, 80, 64, 16),
    ],
)
def test_a1_img2col_native(
    tmp_path,
    filter_size,
    stride,
    pad,
    dilation,
    rows,
    cols,
    pos_m,
    pos_k,
    pad_value,
    manual,
    height,
    width,
    row_stride,
    channel_offset,
):
    root = Path(__file__).resolve().parents[2]
    original = root / "tests/compile/a1_img2col.cpp"
    settings = {
        "FILTER": filter_size,
        "STRIDE": stride,
        "PAD": pad,
        "DILATION": dilation,
        "M": rows,
        "K": cols,
        "POS_M": pos_m,
        "POS_K": pos_k,
        "PAD_VALUE": pad_value,
        "HEIGHT": height,
        "WIDTH": width,
        "ROW_STRIDE": row_stride,
        "CHANNEL_OFFSET": channel_offset,
    }
    definitions = "".join(f"#define PTO_A1_IMG_{name} {value}\n" for name, value in settings.items())
    if manual:
        definitions += "#define PTO_A1_IMG_MANUAL\n"
    # Run the shared CPU instruction oracle first, with the exact same shape
    # and metadata as the device case. Older AArch64 GCC lacks _Float16.
    compiler = shutil.which("clang++-19") or shutil.which("g++-15") or shutil.which("g++-14")
    if compiler is None:
        pytest.skip("CPU im2col oracle requires Clang 19 or GCC >=14")
    cpu_source = tmp_path / "cpu.cpp"
    cpu_source.write_text(definitions + original.read_text())
    executable = tmp_path / "cpu-im2col"
    subprocess.run(
        [compiler, "-std=c++20", "-O1", "-D__CPU_SIM", f"-I{root / 'include'}", str(cpu_source), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    source = original.read_text().replace('extern "C" __global__ AICORE void A1Img2col', "AICORE void A1Img2col")
    wrapper = r"""
#include "tensor.h"
extern "C" __global__ AICORE void NativeImageColumns(__gm__ int64_t* args) {
    auto* src = reinterpret_cast<__gm__ Tensor*>(args[0]);
    auto* weights = reinterpret_cast<__gm__ Tensor*>(args[1]);
    auto* dst = reinterpret_cast<__gm__ Tensor*>(args[2]);
    A1Img2col(reinterpret_cast<__gm__ float*>(dst->buffer.addr) + 16,
              reinterpret_cast<__gm__ half*>(src->buffer.addr) + 16 + PTO_A1_IMG_CHANNEL_OFFSET,
              reinterpret_cast<__gm__ half*>(weights->buffer.addr));
}
"""
    path = tmp_path / "native.cpp"
    path.write_text(definitions + source + wrapper)
    image = simpler.KernelCompiler("a1").compile_single_launch(str(path), str(root))
    with simpler.A1SingleLaunch(image, device_id=0) as runner:
        for seed in (829, 839):
            x = (torch.randn((1, 32, height, width), generator=torch.Generator().manual_seed(seed)) * 3).half()
            if row_stride:
                packed = torch.full((height, width, row_stride), -6.0, dtype=torch.float16)
                packed[:, :, channel_offset : channel_offset + 32] = x[0].permute(1, 2, 0)
            else:
                packed = x.reshape(1, 2, 16, height, width).permute(0, 1, 3, 4, 2).contiguous()
            guarded = torch.full((packed.numel() + 32,), 7.0, dtype=torch.float16)
            guarded[16:-16] = packed.reshape(-1)
            original_input = guarded.clone()
            output = torch.full((rows * cols + 32,), 7.0, dtype=torch.float32)
            runner(guarded, torch.eye(16, dtype=torch.float16), output)
            padded = torch.nn.functional.pad(x.float(), (pad, pad, pad, pad), value=pad_value)
            patches = torch.nn.functional.unfold(padded, filter_size, dilation=dilation, stride=stride)
            # Public K is [C1, filterH, filterW, C0], not NCHW's C0-before-filter order.
            expected = patches.reshape(1, 2, 16, filter_size, filter_size, -1)
            expected = expected.permute(0, 5, 1, 3, 4, 2).contiguous().reshape(-1, 32 * filter_size * filter_size)
            expected = expected[pos_m : pos_m + rows, pos_k : pos_k + cols]
            assert torch.equal(output[16:-16].reshape(rows, cols), expected)
            assert torch.all(output[:16] == 7) and torch.all(output[-16:] == 7)
            assert torch.equal(guarded, original_input)


@pytest.mark.parametrize(
    "left_layout,mode,rows,cols,diagnostic",
    [
        ("ColMajor", "FMATRIX_A_AUTO", 32, 144, "FP16 ZZ Left"),
        ("RowMajor", "FMATRIX_B_AUTO", 32, 144, "FMATRIX A bank only"),
        ("RowMajor", "FMATRIX_A_AUTO", 32, 145, "Layout cols must be divisible"),
        ("RowMajor", "FMATRIX_A_AUTO", 240, 144, "full K and aligned M"),
    ],
)
def test_a1_img2col_rejects_unsupported_tiles(tmp_path, left_layout, mode, rows, cols, diagnostic):
    root = Path(__file__).resolve().parents[2]
    source = f"""
#include <cstdint>
#include <pto/pto-inst.hpp>
using Image = pto::ConvTile<pto::TileType::Mat, half, 2048, pto::Layout::NC1HWC0,
                           pto::ConvTileShape<1, 2, 8, 8, 16>>;
using Left = pto::Tile<pto::TileType::Left, half, {rows}, {cols}, pto::BLayout::{left_layout},
                      {rows}, {cols}, pto::SLayout::RowMajor, 512>;
extern "C" __global__ AICORE void InvalidImage(__gm__ int64_t* args) {{
    Image image;
    Left left;
    pto::TIMG2COL<Left, Image, pto::SetFmatrixMode::{mode}>(left, image);
}}
"""
    path = tmp_path / "unsupported.cpp"
    path.write_text(source)
    # Compile only: invalid profiles must never be launched on a device.
    with pytest.raises(RuntimeError, match=diagnostic):
        simpler.KernelCompiler("a1").compile_single_launch(str(path), str(root))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
