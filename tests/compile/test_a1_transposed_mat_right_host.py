# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Offline compiler guards for A1 16x16 transposed Mat-to-Right moves."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_transposed_mat_right_cpu(tmp_path):
    compiler = shutil.which("clang++-19") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("requires a C++20 compiler")
    binary = tmp_path / "cpu"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            "-D__CPU_SIM",
            f"-I{ROOT / 'include'}",
            str(ROOT / "tests/compile/a1_transposed_mat_right_cpu.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)


def test_qk_two_k16_cpu(tmp_path):
    compiler = shutil.which("clang++-19") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("requires a C++20 compiler")
    binary = tmp_path / "qk_cpu"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            "-ffp-contract=off",
            "-D__CPU_SIM",
            f"-I{ROOT / 'include'}",
            str(ROOT / "tests/compile/a1_qk_two_k16_cpu.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        "A1_TRANSPOSE_INVALID_SHAPE",
        "A1_TRANSPOSE_INVALID_DEST",
        "A1_TRANSPOSE_INVALID_VALID_SHAPE",
        "A1_TRANSPOSE_INVALID_TYPE",
    ],
)
def test_a1_transposed_mat_right_compile(tmp_path, invalid):
    compiler = shutil.which("ccec")
    if compiler is None:
        pytest.skip("requires CCEC for offline A1 compilation")
    command = [
        compiler,
        "-c",
        "-O2",
        "-x",
        "cce",
        "-std=c++17",
        "--cce-aicore-only",
        "--cce-aicore-arch=dav-m200",
        f"-I{ROOT / 'include'}",
    ]
    if invalid:
        command.append(f"-D{invalid}")
    command += [str(ROOT / "tests/compile/a1_transposed_mat_right.cpp"), "-o", str(tmp_path / "move.o")]
    result = subprocess.run(command, capture_output=True, text=True)
    if invalid:
        assert result.returncode != 0
        assert "A1 transposed Mat view requires" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
