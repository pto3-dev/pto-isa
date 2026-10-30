# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A1 indexed gather: CPU reference, production native model and offline CCEC."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/compile/a1_indexed_gather.cpp"
# 32B tails, the 80-class/96-record geometry, unequal source/output shapes,
# cross-row indices and a large panel. Together the live tiles fit 128KiB UB.
GEOMETRIES = [(1, 16, 1, 32), (1, 80, 1, 96), (8, 96, 8, 80), (3, 80, 8, 96), (32, 128, 64, 128)]


@pytest.mark.parametrize("dtype", range(4), ids=["fp16", "fp32", "int32", "uint32"])
@pytest.mark.parametrize("unsigned_index", [False, True])
@pytest.mark.parametrize("rows,cols,source_rows,source_cols", GEOMETRIES)
def test_a1_indexed_gather_cpu_and_offline(tmp_path, dtype, unsigned_index, rows, cols, source_rows, source_cols):
    compiler, native = shutil.which("clang++-19"), shutil.which("ccec")
    if not compiler or not native:
        pytest.skip("requires clang++-19 and CCEC for CPU/offline checks")
    definitions = [
        f"-DGATHER_TYPE={dtype}",
        f"-DGATHER_ROWS={rows}",
        f"-DGATHER_COLS={cols}",
        f"-DGATHER_SOURCE_ROWS={source_rows}",
        f"-DGATHER_SOURCE_COLS={source_cols}",
    ]
    if unsigned_index:
        definitions.append("-DGATHER_UNSIGNED_INDEX")
    for model in (False, True):
        binary = tmp_path / ("native-model" if model else "cpu-reference")
        subprocess.run(
            [
                compiler,
                "-std=c++20",
                "-O1",
                "-D__CPU_SIM",
                f"-I{ROOT / 'include'}",
                *definitions,
                *(["-DA1_GATHER_NATIVE_MODEL"] if model else []),
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
            # All five writable/read-only or writable/writable overlap pairs
            # must trap before gathering from uninitialized alias-test input.
            for alias in range(5):
                rejected = subprocess.run([str(binary), str(alias)], capture_output=True, text=True)
                assert rejected.returncode == 86
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
    "source,destination,indices,scratch,diagnostic",
    [
        ("float,1,80", "float,1,80", "int32_t,1,80", "int32_t,1,80", "scratch needs"),
        ("float,1,80", "float,1,80", "int16_t,1,80", "int16_t,1,160", "matching INT32/UINT32"),
        ("float,1,80", "float,1,80", "int32_t,1,80", "uint32_t,1,160", "matching INT32/UINT32"),
        ("float,1,80", "int32_t,1,80", "int32_t,1,80", "int32_t,1,160", "matching data types"),
        ("float,1,80", "float,1,80", "int32_t,2,40", "int32_t,1,160", "shapes to match"),
        (
            "float,1,80,pto::BLayout::RowMajor,1,64",
            "float,1,80",
            "int32_t,1,80",
            "int32_t,1,160",
            "static full valid shape",
        ),
        ("float,1,81", "float,1,80", "int32_t,1,80", "int32_t,1,160", "Rows must be 32 bytes align"),
        ("int16_t,1,80", "int16_t,1,80", "int32_t,1,80", "int32_t,1,160", "FP16, FP32, INT32 or UINT32"),
        ("float,8,80,pto::BLayout::ColMajor", "float,1,80", "int32_t,1,80", "int32_t,1,160", "contiguous row-major"),
        ("float,1,32776", "float,1,80", "int32_t,1,80", "int32_t,1,160", "UB budget"),
    ],
)
def test_a1_indexed_gather_rejects_unsupported_tiles(tmp_path, source, destination, indices, scratch, diagnostic):
    native = shutil.which("ccec")
    if not native:
        pytest.skip("requires CCEC for A1 template rejection")
    cpp = tmp_path / "invalid.cpp"
    cpp.write_text(f"""
#include <pto/pto-inst.hpp>
using Source = pto::Tile<pto::TileType::Vec,{source}>;
using Destination = pto::Tile<pto::TileType::Vec,{destination}>;
using Indices = pto::Tile<pto::TileType::Vec,{indices}>;
using Scratch = pto::Tile<pto::TileType::Vec,{scratch}>;
extern "C" __global__ AICORE void Invalid(__gm__ float*) {{
    Source s; Destination d; Indices i; Scratch t;
    pto::TGATHER(d,s,i,t);
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
