# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Full-size QK row partition and precision checks on the CPU simulator."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_qk_row_reuse_cpu(tmp_path):
    compiler = shutil.which("clang++-19") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("requires a C++20 compiler")
    binary = tmp_path / "qk_row_reuse_cpu"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            "-ffp-contract=off",
            "-D__CPU_SIM",
            f"-I{ROOT / 'include'}",
            str(ROOT / "tests/compile/a1_qk_row_reuse_cpu.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
