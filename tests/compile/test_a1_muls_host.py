# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU and offline A1 scalar multiply checks; never queries or executes NPU."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/compile/a1_muls_vector.cpp"
CASES = [(fp16, 1, cols, alias) for fp16 in (False, True) for cols in (16, 80, 256) for alias in (False, True)] + [
    (False, 41, 400, True),
    (True, 41, 800, True),
]


def _definitions(fp16, rows, cols, alias):
    result = [f"-DMULS_ROWS={rows}", f"-DMULS_COLS={cols}"]
    if fp16:
        result.append("-DMULS_FP16")
    if alias:
        result.append("-DMULS_INPLACE")
    return result


@pytest.mark.parametrize("fp16,rows,cols,alias", CASES)
def test_a1_muls_cpu_and_offline(tmp_path, fp16, rows, cols, alias):
    compiler = shutil.which("clang++-19")
    native = shutil.which("ccec")
    if not compiler or not native:
        pytest.skip("requires clang++-19 and CCEC for CPU/offline checks")
    flags = _definitions(fp16, rows, cols, alias)
    binary = tmp_path / "cpu"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            "-ffp-contract=off",
            "-D__CPU_SIM",
            f"-I{ROOT / 'include'}",
            *flags,
            str(SOURCE),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    subprocess.run(
        [
            native,
            "-c",
            "-O2",
            "-x",
            "cce",
            "-std=c++17",
            "--cce-aicore-only",
            "--cce-aicore-arch=dav-m200",
            f"-I{ROOT / 'include'}",
            *flags,
            str(SOURCE),
            "-o",
            str(tmp_path / "native.o"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "definition,diagnostic", [("-DMULS_INT", "FP16 or FP32"), ("-DMULS_PARTIAL", "static full valid shape")]
)
def test_a1_muls_rejects_unsupported_tile(tmp_path, definition, diagnostic):
    native = shutil.which("ccec")
    if not native:
        pytest.skip("requires CCEC for A1 template rejection")
    result = subprocess.run(
        [
            native,
            "-c",
            "-O2",
            "-x",
            "cce",
            "-std=c++17",
            "--cce-aicore-only",
            "--cce-aicore-arch=dav-m200",
            f"-I{ROOT / 'include'}",
            definition,
            str(SOURCE),
            "-o",
            str(tmp_path / "invalid.o"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert diagnostic in result.stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
