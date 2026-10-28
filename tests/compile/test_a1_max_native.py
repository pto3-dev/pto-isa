# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU-first finite A1 maximum, repeat tails, operand aliases and output guards."""

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


def _cpu_first(tmp_path, definitions):
    root = Path(__file__).resolve().parents[2]
    source = (root / "tests/compile/a1_max_vector.cpp").read_text()
    compiler = shutil.which("clang++-19") or shutil.which("g++-15") or shutil.which("g++-14")
    if compiler is None:
        pytest.skip("CPU oracle requires Clang19 or GCC>=14")
    cpu = tmp_path / "cpu.cpp"
    cpu.write_text(definitions + source)
    binary = tmp_path / "cpu-max"
    subprocess.run(
        [compiler, "-std=c++20", "-O1", "-D__CPU_SIM", f"-I{root / 'include'}", str(cpu), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    return root, source


@pytest.mark.parametrize("fp16", [False, True])
@pytest.mark.parametrize(
    ("size", "alias"), [(size, alias) for size in [16, 80, "large"] for alias in [0, 1, 2]] + [("repeat", 0)]
)
def test_a1_maximum_native(tmp_path, fp16, size, alias):
    if size == "repeat":
        count = 32784 if fp16 else 16400
    elif size == "large":
        count = (21808 if fp16 else 10880) if alias == 2 else (32720 if fp16 else 16336)
    else:
        count = size
    definitions = f"#define PTO_MAX_ELEMENTS {count}\n#define PTO_MAX_ALIAS {alias}\n"
    if fp16:
        definitions += "#define PTO_MAX_FP16\n"
    if size == "repeat":
        definitions += "#define PTO_MAX_SAME_INPUT\n"
    root, source = _cpu_first(tmp_path, definitions)
    source = source.replace('extern "C" __global__ AICORE void A1Maximum', "AICORE void A1Maximum")
    wrapper = r"""
#include "tensor.h"
extern "C" __global__ AICORE void NativeMaximum(__gm__ int64_t* args) {
    auto* lhs = reinterpret_cast<__gm__ Tensor*>(args[0]);
    auto* rhs = reinterpret_cast<__gm__ Tensor*>(args[1]);
    auto* out = reinterpret_cast<__gm__ Tensor*>(args[2]);
    A1Maximum(reinterpret_cast<__gm__ Element*>(out->buffer.addr) + 16,
              reinterpret_cast<__gm__ Element*>(lhs->buffer.addr),
              reinterpret_cast<__gm__ Element*>(rhs->buffer.addr));
}
"""
    path = tmp_path / "native.cpp"
    path.write_text(definitions + source + wrapper)
    image = simpler.KernelCompiler("a1").compile_single_launch(str(path), str(root))
    dtype = torch.float16 if fp16 else torch.float32
    with simpler.A1SingleLaunch(image, device_id=0) as runner:
        for seed in [947, 953]:
            generator = torch.Generator().manual_seed(seed)
            lhs = torch.randn((1, count), generator=generator).to(dtype)
            rhs = torch.randn((1, count), generator=generator).to(dtype)
            # Include finite subnormals, normal limits, zeros and equal operands.
            bits = torch.tensor([0, 0x8000, 1, 0x8001, 0x7BFF, 0xFBFF, 0x0400, 0x8400], dtype=torch.int64)
            if fp16:
                special = bits.to(torch.int16).view(dtype)
            else:
                special = (
                    torch.tensor(
                        [0, 0x80000000, 1, 0x80000001, 0x7F7FFFFF, 0xFF7FFFFF, 0x00800000, 0x80800000],
                        dtype=torch.int64,
                    )
                    .to(torch.int32)
                    .view(dtype)
                )
            lhs[0, :8], rhs[0, :8] = special, special.flip(0)
            lhs[0, 8:16], rhs[0, 8:16] = special, 0
            if size == "repeat":
                rhs = lhs.clone()
            original_lhs, original_rhs = lhs.clone(), rhs.clone()
            output = torch.full((1, count + 32), 7.0, dtype=dtype)
            runner(lhs, rhs, output)
            torch.testing.assert_close(output[:, 16:-16], torch.maximum(original_lhs, original_rhs), rtol=0, atol=0)
            assert torch.equal(lhs, original_lhs) and torch.equal(rhs, original_rhs)
            assert torch.all(output[:, :16] == 7) and torch.all(output[:, -16:] == 7)


@pytest.mark.parametrize(
    ("definitions", "diagnostic"),
    [("#define PTO_MAX_ELEMENTS 15\n", "aligned to 32 bytes"), ("#define PTO_MAX_INT\n", "FP16 or FP32")],
)
def test_a1_maximum_rejects_unsupported_profile(tmp_path, definitions, diagnostic):
    # These cases only compile intentionally invalid profiles; no device entry runs.
    root = Path(__file__).resolve().parents[2]
    source = (root / "tests/compile/a1_max_vector.cpp").read_text()
    path = tmp_path / "invalid.cpp"
    path.write_text(definitions + source)
    with pytest.raises(RuntimeError, match=diagnostic):
        simpler.KernelCompiler("a1").compile_single_launch(str(path), str(root))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
