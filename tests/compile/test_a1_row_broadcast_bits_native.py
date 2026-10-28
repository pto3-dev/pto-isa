# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Exact A1 row-broadcast bit patterns, guards, stride limits and repeat tails."""

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


@pytest.mark.parametrize(
    "fp16,rows,cols,exhaustive",
    [
        (False, 16, 400, False),
        (True, 16, 400, False),
        (False, 8, 16, False),
        (True, 16, 16, False),
        (False, 16, 24, False),
        (True, 16, 32, False),
        (False, 16, 64, False),
        (True, 16, 80, False),
        (False, 264, 80, False),
        (True, 272, 80, False),
        (False, 8, 2040, False),
        (True, 16, 4080, False),
        (False, 8, 8, False),
        (False, 8, 2056, False),
        (True, 256, 16, True),
    ],
)
def test_a1_row_broadcast_bits(tmp_path, fp16, rows, cols, exhaustive):
    root = Path(__file__).resolve().parents[2]
    source = (root / "tests/compile/a1_row_broadcast_bits.cpp").read_text()
    definitions = f"#define PTO_ROW_ROWS {rows}\n#define PTO_ROW_COLS {cols}\n"
    if fp16:
        definitions += "#define PTO_ROW_FP16\n"
    compiler = shutil.which("clang++-19") or shutil.which("g++-15") or shutil.which("g++-14")
    if compiler is None:
        pytest.skip("CPU oracle requires Clang19 or GCC>=14")
    cpu = tmp_path / "cpu.cpp"
    cpu.write_text(definitions + source)
    exe = tmp_path / "cpu-row"
    subprocess.run(
        [compiler, "-std=c++20", "-O1", "-D__CPU_SIM", f"-I{root / 'include'}", str(cpu), "-o", str(exe)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    source = source.replace('extern "C" __global__ AICORE void A1RowBroadcast', "AICORE void A1RowBroadcast")
    wrapper = r"""
#include "tensor.h"
extern "C" __global__ AICORE void NativeRowBits(__gm__ int64_t* args) {
    auto* src=reinterpret_cast<__gm__ Tensor*>(args[0]);
    auto* dst=reinterpret_cast<__gm__ Tensor*>(args[1]);
    A1RowBroadcast(reinterpret_cast<__gm__ Element*>(dst->buffer.addr)+16,
                   reinterpret_cast<__gm__ Element*>(src->buffer.addr)+16);
}
"""
    path = tmp_path / "native.cpp"
    path.write_text(definitions + source + wrapper)
    image = simpler.KernelCompiler("a1").compile_single_launch(str(path), str(root))
    integers = torch.int16 if fp16 else torch.int32
    dtype = torch.float16 if fp16 else torch.float32
    patterns = (
        [0, 0x8000, 0x3C00, 0xBC00, 1, 0x8001, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E01, 0xFE05, 0x7C01, 0xFC01]
        if fp16
        else [
            0,
            0x80000000,
            0x3F800000,
            0xBF800000,
            1,
            0x80000001,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC00001,
            0xFFC00005,
            0x7F800001,
            0xFF800001,
        ]
    )
    with simpler.A1SingleLaunch(image, device_id=0) as runner:
        starts = range(0, 65536, rows) if exhaustive else range(2)
        for start in starts:
            if exhaustive:
                bits = torch.arange(start, start + rows, dtype=torch.int32).to(integers)
            elif start == 0:
                bits = torch.tensor([patterns[i % len(patterns)] for i in range(rows)], dtype=torch.int64).to(integers)
            else:
                bits = torch.randint(
                    0,
                    65536 if fp16 else 0x100000000,
                    (rows,),
                    generator=torch.Generator().manual_seed(919),
                    dtype=torch.int64,
                ).to(integers)
            src = torch.full((rows + 32,), 7, dtype=dtype)
            src.view(integers)[16:-16] = bits
            original = src.view(integers).clone()
            output = torch.full((rows * cols + 32,), 7, dtype=dtype)
            runner(src, output)
            assert torch.equal(output.view(integers)[16:-16].reshape(rows, cols), bits[:, None].expand(rows, cols))
            assert torch.all(output[:16] == 7) and torch.all(output[-16:] == 7)
            assert torch.equal(src.view(integers), original)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
