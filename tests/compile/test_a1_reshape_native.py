# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU-first full ND vector reshape: views, write-through, reload and guards."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
simpler = pytest.importorskip("simpler_setup")


@pytest.fixture(scope="module", autouse=True)
def require_310p():
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclrtGetSocName.restype = ctypes.c_char_p
    acl.aclrtGetSocName.argtypes = []
    soc = acl.aclrtGetSocName()
    if not soc or not soc.startswith(b"Ascend310P"):
        pytest.skip("Requires Ascend310P hardware")


@pytest.mark.parametrize("fp16", [False, True])
@pytest.mark.parametrize("shape", [(160, 16, 2560), (1, 2560, 16), (64, 32, 128), (1, 32736, 32)])
def test_a1_reshape_native(tmp_path, fp16, shape):
    rows, cols, target_cols = shape
    root = Path(__file__).resolve().parents[2]
    source = (root / "tests/compile/a1_reshape_vector.cpp").read_text()
    definitions = (
        f"#define PTO_RESHAPE_ROWS {rows}\n#define PTO_RESHAPE_COLS {cols}\n"
        f"#define PTO_RESHAPE_TARGET_COLS {target_cols}\n"
    )
    if fp16:
        definitions += "#define PTO_RESHAPE_FP16\n"
    compiler = shutil.which("clang++-19")
    assert compiler, "CPU oracle requires Clang19"
    cpu = tmp_path / "cpu.cpp"
    cpu.write_text(definitions + source)
    binary = tmp_path / "cpu-reshape"
    subprocess.run(
        [compiler, "-std=c++20", "-O1", "-D__CPU_SIM", f"-I{root / 'include'}", str(cpu), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    source = source.replace('extern "C" __global__ AICORE void A1Reshape', "AICORE void A1Reshape")
    wrapper = r"""
#include "tensor.h"
extern "C" __global__ AICORE void NativeReshape(__gm__ int64_t* args) {
    auto* src = reinterpret_cast<__gm__ Tensor*>(args[0]);
    auto* dst = reinterpret_cast<__gm__ Tensor*>(args[1]);
    A1Reshape(reinterpret_cast<__gm__ Element*>(dst->buffer.addr) + 16,
              reinterpret_cast<__gm__ Element*>(src->buffer.addr));
}
"""
    path = tmp_path / "native.cpp"
    path.write_text(definitions + source + wrapper)
    image = simpler.KernelCompiler("a1").compile_single_launch(str(path), str(root))
    dtype = torch.float16 if fp16 else torch.float32
    count = rows * cols
    with simpler.A1SingleLaunch(image, device_id=0) as runner:
        for seed in [1013, 1019]:
            generator = torch.Generator().manual_seed(seed)
            values = torch.randint(-256, 256, (rows, cols), generator=generator).to(dtype) / 8
            saved = values.clone()
            out = torch.full((1, 2 * count + 32), 7, dtype=dtype)
            runner(values, out)
            assert torch.equal(out[0, 16 : 16 + count], saved.reshape(-1))
            assert torch.equal(out[0, 16 + count : -16], (saved * 2).reshape(-1))
            assert torch.equal(values, saved)
            assert torch.all(out[:, :16] == 7) and torch.all(out[:, -16:] == 7)


@pytest.mark.parametrize(
    ("src", "dst", "diagnostic"),
    [
        ("float, 2, 16", "float, 1, 16", "equal element counts"),
        ("half, 2, 16", "float, 1, 32", "matching element types"),
        ("float, 2, 16", "float, 8, 4, pto::BLayout::ColMajor", "row-major"),
        ("float, 2, 16", "float, 4, 8, pto::BLayout::RowMajor, 3, 8", "full valid shape"),
        ("float, 2, 16", "float, 1, 31", "32 bytes align"),
        ("int32_t, 2, 16", "int32_t, 1, 32", "FP16 or FP32"),
    ],
)
def test_a1_reshape_rejects_unsupported_profile(tmp_path, src, dst, diagnostic):
    root = Path(__file__).resolve().parents[2]
    path = tmp_path / "invalid.cpp"
    path.write_text(f"""
#include <pto/pto-inst.hpp>
extern "C" __global__ AICORE void Invalid() {{
    pto::Tile<pto::TileType::Vec, {src}> src;
    pto::Tile<pto::TileType::Vec, {dst}> dst;
    pto::TRESHAPE(dst, src);
}}
""")
    with pytest.raises(RuntimeError, match=diagnostic):
        simpler.KernelCompiler("a1").compile_single_launch(str(path), str(root))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
