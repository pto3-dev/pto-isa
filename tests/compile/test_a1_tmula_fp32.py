# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Verify A1 TMULA against fused FP32 and exact FP16-product oracles."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
simpler = pytest.importorskip("simpler_setup")


@pytest.mark.parametrize("count,alias", [(64, 0), (80, 0), (960, 0), (4096, 0), (80, 1), (80, 2)])
def test_a1_tmula_fp32(tmp_path, count, alias):
    try:
        acl = ctypes.CDLL("libascendcl.so")
    except OSError:
        pytest.skip("Ascend runtime unavailable")
    acl.aclrtGetSocName.restype = ctypes.c_char_p
    soc = acl.aclrtGetSocName()
    if not soc or not soc.startswith(b"Ascend310P"):
        pytest.skip("requires Ascend310P")
    root = Path(__file__).resolve().parents[2]
    source = (
        f"#define TEST_COUNT {count}\n#define TEST_ALIAS {alias}\n"
        + (root / "tests/compile/a1_tmula_fp32.cpp").read_text()
    )
    compiler = shutil.which("clang++-19")
    assert compiler
    cpp = tmp_path / "mula.cpp"
    cpp.write_text(source)
    executable = tmp_path / "cpu"
    subprocess.run(
        [compiler, "-std=c++20", "-O1", "-D__CPU_SIM", f"-I{root / 'include'}", str(cpp), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(executable)], check=True)
    image = simpler.KernelCompiler("a1").compile_single_launch(str(cpp), str(root))
    libm = ctypes.CDLL("libm.so.6")
    libm.fmaf.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    libm.fmaf.restype = ctypes.c_float
    with simpler.A1SingleLaunch(image, device_id=0) as runner:
        for seed in (1949, 1951):
            generator = torch.Generator().manual_seed(seed)
            a, b, d = [torch.randn(count, generator=generator) for _ in range(3)]
            if alias == 1:
                a = d.clone()
            if alias == 2:
                b = d.clone()
            inputs = [torch.full((count + 32,), 7.0, dtype=torch.float32) for _ in range(3)]
            for target, values in zip(inputs, (a, b, d), strict=True):
                target[16:-16] = values
            originals = [x.clone() for x in inputs]
            result = torch.full((count + 32,), 7.0, dtype=torch.float32)
            runner(*inputs, result)
            expected = torch.tensor(
                [libm.fmaf(x, y, z) for x, y, z in zip(a.tolist(), b.tolist(), d.tolist(), strict=True)]
            )
            assert torch.equal(result[16:-16].view(torch.int32), expected.view(torch.int32))
            assert torch.all(result[:16] == 7) and torch.all(result[-16:] == 7)
            assert all(torch.equal(x, y) for x, y in zip(inputs, originals, strict=True))


@pytest.mark.parametrize(
    "dst_type, src_type, message",
    [("half", "half", "A1 TMULA requires FP32 tiles"), ("float", "half", "matching data types")],
)
def test_a1_tmula_rejects_unsupported_types(tmp_path, dst_type, src_type, message):
    root = Path(__file__).resolve().parents[2]
    source = f"""
#include <pto/pto-inst.hpp>
using D = pto::Tile<pto::TileType::Vec, {dst_type}, 1, 64>;
using S = pto::Tile<pto::TileType::Vec, {src_type}, 1, 64>;
extern "C" __global__ AICORE void InvalidMula(__gm__ int64_t*) {{
 D d; S a, b; pto::TASSIGN<0>(d); pto::TASSIGN<4096>(a); pto::TASSIGN<8192>(b); pto::TMULA(d, a, b);
}}
"""
    path = tmp_path / "invalid.cpp"
    path.write_text(source)
    with pytest.raises(RuntimeError, match=message):
        simpler.KernelCompiler("a1").compile_single_launch(str(path), str(root))
