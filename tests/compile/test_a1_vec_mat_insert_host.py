# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU semantics and offline A1 compilation for UB-to-L1 insertion."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "arguments",
    [["0", "0"], ["16", "16"], ["32", "32"], [], ["c16"]],
    ids=["upper-left", "middle", "bottom-right", "reuse-odd-tail", "c16-reuse"],
)
def test_vec_mat_insert_cpu(tmp_path, arguments):
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
            str(ROOT / "tests/compile/a1_vec_mat_insert_cpu.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary), *arguments], check=True)


def test_vec_mat_pipeline_compile(tmp_path):
    compiler = shutil.which("ccec")
    if compiler is None:
        pytest.skip("requires CCEC for offline A1 compilation")
    subprocess.run(
        [
            compiler,
            "-c",
            "-O2",
            "-x",
            "cce",
            "-std=c++17",
            "--cce-aicore-only",
            "--cce-aicore-arch=dav-m200",
            f"-I{ROOT / 'include'}",
            str(ROOT / "tests/compile/a1_vec_mat_insert_host.cpp"),
            "-o",
            str(tmp_path / "pipeline.o"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "source,destination,message",
    [
        (
            "Tile<TileType::Vec, float, 32, 32>",
            "Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512>",
            "A1 Vec-to-Mat insertion",
        ),
        (
            "Tile<TileType::Vec, half, 32, 32>",
            "Tile<TileType::Mat, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::RowMajor, 512>",
            "A1 Vec-to-Mat insertion",
        ),
        (
            "Tile<TileType::Vec, half, 32, 96>",
            "Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512>",
            "A1 Vec-to-Mat insertion",
        ),
        (
            "Tile<TileType::Vec, half, 8, 32>",
            "Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512>",
            "A1 Vec-to-Mat insertion",
        ),
        (
            "Tile<TileType::Vec, half, 32, 32, BLayout::RowMajor, 16, 32>",
            "Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512>",
            "static full valid shape",
        ),
    ],
)
def test_vec_mat_insert_rejects_native_types(tmp_path, source, destination, message):
    compiler = shutil.which("ccec")
    if compiler is None:
        pytest.skip("requires CCEC for offline A1 compilation")
    cpp = tmp_path / "invalid.cpp"
    cpp.write_text(f"""
#include <pto/pto-inst.hpp>
using namespace pto;
extern "C" __global__ AICORE void Invalid(__gm__ half*) {{
 {source} source; {destination} destination;
 TASSIGN<0>(source); TASSIGN<0>(destination);
 TINSERT(destination, source, 0, 0);
}}
""")
    result = subprocess.run(
        [
            compiler,
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
    assert message in result.stderr
