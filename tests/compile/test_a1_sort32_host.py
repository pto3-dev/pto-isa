# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A1 proposal-backed public TSORT32, exact compact pairs and unsigned ties."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/compile/a1_sort32.cpp"
GEOMETRIES = [
    (1, 32, 512),
    (1, 96, 512),
    (3, 160, 1040),
    (8, 96, 1536),
    (2, 512, 4096),
    (1, 4096, 4096),
    (4, 256, 3600),
    (1, 1024, 10256),
]


@pytest.mark.parametrize("rows,cols,scratch", GEOMETRIES)
def test_a1_sort32_cpu_native_model_and_offline(tmp_path, rows, cols, scratch):
    _validate(tmp_path, rows, cols, scratch, broadcast=False)


def test_a1_sort32_broadcast_indices(tmp_path):
    # The generic CPU helper currently returns early for broadcast rows. The
    # independent pair oracle still validates the production A1 implementation.
    _validate(tmp_path, 3, 96, 512, broadcast=True)


def _validate(tmp_path, rows, cols, scratch, *, broadcast):
    compiler, native = shutil.which("clang++-19"), shutil.which("ccec")
    if not compiler or not native:
        pytest.skip("requires clang++-19 and CCEC for CPU/offline checks")
    definitions = [f"-DSORT_ROWS={rows}", f"-DSORT_COLS={cols}", f"-DSORT_SCRATCH={scratch}"]
    if broadcast:
        definitions.append("-DSORT_BROADCAST")
    for model in (False, True):
        if broadcast and not model:
            continue
        binary = tmp_path / ("native-model" if model else "cpu-reference")
        subprocess.run(
            [
                compiler,
                "-std=c++20",
                "-O1",
                "-D__CPU_SIM",
                f"-I{ROOT / 'include'}",
                *definitions,
                *(["-DA1_SORT_NATIVE_MODEL"] if model else []),
                str(SOURCE),
                "-o",
                str(binary),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run([str(binary)], check=True, capture_output=True, text=True)
        if model:
            for alias in range(5):
                result = subprocess.run([str(binary), str(alias)], capture_output=True, text=True)
                assert result.returncode == 86
            subprocess.run([str(binary), "readonly-alias"], check=True, capture_output=True, text=True)
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
            *definitions,
            str(SOURCE),
            "-o",
            str(tmp_path / "native.o"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "source,destination,index,scratch,diagnostic",
    [
        ("float,1,32", "float,1,64", "uint32_t,1,32", "float,1,256", "at least 512"),
        ("half,1,32", "half,1,128", "uint32_t,1,32", "half,1,512", "requires FP32"),
        ("float,1,32", "float,1,64", "int32_t,1,32", "float,1,512", "UINT32 indices"),
        ("float,1,32", "float,1,32", "uint32_t,1,32", "float,1,512", "expanded destination"),
        ("float,2,32", "float,2,64", "uint32_t,3,32", "float,1,512", "matching/broadcast indices"),
        ("float,1,48", "float,1,96", "uint32_t,1,48", "float,1,512", "full 32-value blocks"),
        (
            "float,1,32,pto::BLayout::RowMajor,1,16",
            "float,1,64",
            "uint32_t,1,32",
            "float,1,512",
            "static full valid shape",
        ),
        ("float,1,32", "float,1,64", "uint32_t,1,32", "float,1,32776", "UB budget"),
        ("float,1,32", "float,1,64", "uint32_t,1,32", None, "requires explicit proposal scratch"),
    ],
)
def test_a1_sort32_rejects_unsupported_profile(tmp_path, source, destination, index, scratch, diagnostic):
    native = shutil.which("ccec")
    if not native:
        pytest.skip("requires CCEC for A1 template rejection")
    cpp = tmp_path / "invalid.cpp"
    temporary = f"pto::Tile<pto::TileType::Vec,{scratch}> t;" if scratch else ""
    argument = ",t" if scratch else ""
    cpp.write_text(f"""
#include <pto/pto-inst.hpp>
extern "C" __global__ AICORE void Invalid(__gm__ float*) {{
    pto::Tile<pto::TileType::Vec,{source}> s;
    pto::Tile<pto::TileType::Vec,{destination}> d;
    pto::Tile<pto::TileType::Vec,{index}> i;
    {temporary}
    pto::TSORT32(d,s,i{argument});
}}
""")
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
            str(cpp),
            "-o",
            str(tmp_path / "invalid.o"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert diagnostic in result.stderr
