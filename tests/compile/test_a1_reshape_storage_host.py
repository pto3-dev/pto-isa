# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A1 raw-storage reshape, alias write-through, generic CPU and native offline only."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SHAPES = [(8, 32, 32), (8, 32, 256), (1, 256, 32), (3, 160, 96), (1, 32736, 32)]


@pytest.mark.parametrize("dtype", ["int32_t", "uint32_t", "float", "half"])
@pytest.mark.parametrize("rows,columns,target_columns", SHAPES)
def test_a1_reshape_storage_cpu_and_offline(tmp_path, dtype, rows, columns, target_columns):
    compiler, native = shutil.which("clang++-19"), shutil.which("ccec")
    assert compiler and native, "requires clang++-19 and CCEC for CPU/offline validation"
    definitions = [
        f"-DRESHAPE_TYPE={dtype}",
        f"-DRESHAPE_ROWS={rows}",
        f"-DRESHAPE_COLS={columns}",
        f"-DRESHAPE_TARGET_COLS={target_columns}",
    ]
    source = ROOT / "tests/compile/a1_reshape_storage.cpp"
    for native_model in (False, True):
        binary = tmp_path / ("production-model" if native_model else "cpu-reference")
        subprocess.run(
            [
                compiler,
                "-std=c++20",
                "-O1",
                "-D__CPU_SIM",
                "-DPTO_NPU_ARCH_A1",
                f"-I{ROOT / 'include'}",
                *definitions,
                *(["-DA1_RESHAPE_NATIVE_MODEL"] if native_model else []),
                str(source),
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
            *definitions,
            str(source),
            "-o",
            str(tmp_path / "native.o"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "source,destination,diagnostic",
    [
        ("int32_t,2,16", "int32_t,1,16", "equal element counts"),
        ("int32_t,2,16", "uint32_t,1,32", "matching element types"),
        ("half,2,16", "float,1,32", "matching element types"),
        ("int64_t,2,16", "int64_t,1,32", "FP16/FP32/INT32/UINT32"),
        ("uint32_t,2,16", "uint32_t,4,8,pto::BLayout::RowMajor,3,8", "full valid shape"),
        ("uint32_t,2,16", "uint32_t,8,4,pto::BLayout::ColMajor", "row-major"),
        ("uint32_t,2,16", "uint32_t,1,31", "32 bytes"),
        ("uint32_t,1,32776", "uint32_t,1,32776", "UB budget"),
    ],
)
def test_a1_reshape_storage_rejects_invalid_profile(tmp_path, source, destination, diagnostic):
    native = shutil.which("ccec")
    assert native, "requires CCEC for native rejection"
    cpp = tmp_path / "invalid.cpp"
    cpp.write_text(
        "#include <pto/pto-inst.hpp>\n"
        + f'extern "C" __global__ AICORE void Invalid() {{ pto::Tile<pto::TileType::Vec,{source}> src; pto::Tile<pto::TileType::Vec,{destination}> dst; pto::TRESHAPE(dst,src); }}\n'
    )
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
    assert diagnostic in result.stdout + result.stderr
