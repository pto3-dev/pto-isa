# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check public A1 N32 matrix multiplication and accumulation."""

import ctypes
import shutil
import subprocess
from pathlib import Path
import pytest

torch = pytest.importorskip("torch")
simpler = pytest.importorskip("simpler_setup")


@pytest.mark.parametrize(
    "rows,reduction,zz,gap,accumulate",
    [(16, 16, False, 0, False), (32, 64, True, 32, True), (128, 128, False, 0, False), (16, 1024, True, 0, True)],
)
def test_a1_n32_matmul(tmp_path, rows, reduction, zz, gap, accumulate):
    try:
        acl = ctypes.CDLL("libascendcl.so")
    except OSError:
        pytest.skip("Ascend runtime unavailable")
    acl.aclrtGetSocName.restype = ctypes.c_char_p
    acl.aclrtGetSocName.argtypes = []
    soc = acl.aclrtGetSocName()
    if not soc or not soc.startswith(b"Ascend310P"):
        pytest.skip("requires Ascend310P")
    root = Path(__file__).resolve().parents[2]
    source = (root / "tests/compile/a1_right_n32_tmov.cpp").read_text()
    defs = f"#define TEST_M {rows}\n#define TEST_K {reduction}\n#define TEST_GAP {gap}\n"
    defs += "#define TEST_PUBLIC_N32\n"
    if accumulate:
        defs += "#define TEST_ACCUMULATE\n"
    if zz:
        defs += "#define TEST_ZZ\n"
    compiler = shutil.which("clang++-19") or shutil.which("g++-15") or shutil.which("g++-14")
    if compiler is None:
        pytest.skip("CPU oracle requires Clang 19 or GCC >=14")
    cpu = tmp_path / "cpu.cpp"
    cpu.write_text(defs + source)
    executable = tmp_path / "cpu-n32"
    subprocess.run(
        [compiler, "-std=c++20", "-O1", "-D__CPU_SIM", f"-I{root / 'include'}", str(cpu), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    path = tmp_path / "right.cpp"
    path.write_text(defs + source)
    image = simpler.KernelCompiler("a1").compile_single_launch(str(path), str(root))
    with simpler.A1SingleLaunch(image, device_id=0) as runner:
        for seed in (919, 929):
            gen = torch.Generator().manual_seed(seed)
            a = (torch.randint(-8, 9, (rows, reduction), generator=gen).float() / 16).half()
            b = (torch.randint(-8, 9, (reduction, 32), generator=gen).float() / 16).half()
            ga = torch.full((a.numel() + 32,), 7.0, dtype=torch.float16)
            ga[16:-16] = a.reshape(-1)
            gb = torch.full((reduction * (32 + gap) + 32,), 7.0, dtype=torch.float16)
            gb[16:-16].reshape(reduction, 32 + gap)[:, :32] = b
            ga_before, gb_before = ga.clone(), gb.clone()
            out = torch.full((rows * 32 + 32,), 7.0, dtype=torch.float32)
            runner(ga, gb, out)
            expected = a.float() @ b.float()
            if accumulate:
                expected *= 2
            assert torch.equal(out[16:-16].reshape(rows, 32), expected)
            assert torch.all(out[:16] == 7) and torch.all(out[-16:] == 7)
            assert torch.equal(ga, ga_before) and torch.equal(gb, gb_before)


@pytest.mark.parametrize("rows,reduction", [(256, 144), (16, 1040)])
def test_a1_n32_matmul_rejects_operand_overflow(tmp_path, rows, reduction):
    root = Path(__file__).resolve().parents[2]
    source = (root / "tests/compile/a1_right_n32_tmov.cpp").read_text()
    path = tmp_path / "oversized.cpp"
    path.write_text(f"#define TEST_M {rows}\n#define TEST_K {reduction}\n#define TEST_PUBLIC_N32\n" + source)
    with pytest.raises(RuntimeError, match="within 64 KiB"):
        simpler.KernelCompiler("a1").compile_single_launch(str(path), str(root))
