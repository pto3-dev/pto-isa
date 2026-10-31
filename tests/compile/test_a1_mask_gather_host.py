# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Driver-free fixed-mask gather: bitwise CPU/native-model and m200 compilation."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/compile/a1_mask_gather.cpp"
TYPES = [
    ("half", "half"),
    ("float", "float"),
    ("float", "uint32_t"),
    ("uint32_t", "float"),
    ("uint32_t", "int32_t"),
    ("int32_t", "uint32_t"),
]
# Base cases cover every mask and same-size bit interpretation. Extra cases
# exercise partial repeats, the padded 304-pair prefix and the 255-repeat boundary.
CASES = [(src, dst, pattern, 3, 128) for src, dst in TYPES for pattern in range(1, 8)]
CASES += [
    ("float", "uint32_t", 1, 1, 608),
    ("float", "uint32_t", 2, 1, 608),
    ("float", "float", 3, 1, 608),
    ("float", "uint32_t", 2, 5, 96),
    ("half", "half", 1, 3, 160),
    ("half", "half", 4, 3, 192),
    ("float", "uint32_t", 1, 1, 16352),
    ("half", "half", 2, 1, 32704),
    ("float", "uint32_t", 2, 1, 16416),
    ("half", "half", 1, 1, 32832),
    ("float", "uint32_t", 5, 1, 24608),
    ("half", "half", 6, 1, 49216),
    ("float", "uint32_t", 7, 1, 16352),
    ("half", "half", 7, 1, 32704),
]


def native_command(source: Path, output: Path, definitions: list[str]) -> list[str]:
    compiler = shutil.which("ccec")
    assert compiler, "requires CCEC for m200 validation"
    return [
        compiler,
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
        str(output),
    ]


@pytest.mark.parametrize("input_type,output_type,pattern,rows,columns", CASES)
def test_a1_mask_gather_cpu_and_offline(tmp_path, input_type, output_type, pattern, rows, columns):
    compiler = shutil.which("clang++-19")
    assert compiler, "requires clang++-19 for bitwise CPU validation"
    definitions = [
        f"-DMASK_INPUT={input_type}",
        f"-DMASK_OUTPUT={output_type}",
        f"-DMASK_PATTERN={pattern}",
        f"-DMASK_ROWS={rows}",
        f"-DMASK_COLS={columns}",
    ]
    for native_model in (False, True):
        binary = tmp_path / ("production-model" if native_model else "cpu-reference")
        subprocess.run(
            [
                compiler,
                "-std=c++20",
                "-O1",
                "-D__CPU_SIM",
                f"-I{ROOT / 'include'}",
                *definitions,
                *(["-DA1_MASK_NATIVE_MODEL"] if native_model else []),
                str(SOURCE),
                "-o",
                str(binary),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run([str(binary)], check=True, capture_output=True, text=True)
        if native_model:
            for overlap in range(3):
                result = subprocess.run([str(binary), str(overlap)], capture_output=True, text=True)
                assert result.returncode == 86
    subprocess.run(
        native_command(SOURCE, tmp_path / "native.o", definitions), check=True, capture_output=True, text=True
    )


@pytest.mark.parametrize(
    "source,destination,pattern,axis,diagnostic",
    [
        ("float,1,128", "float,1,64", 0, "GATHER_ROW", "fixed pattern in 1..7"),
        ("float,1,128", "float,1,128", 8, "GATHER_ROW", "fixed pattern in 1..7"),
        ("float,4,128", "float,1,128", 3, "GATHER_COL", "row axis only"),
        ("int16_t,1,128", "int16_t,1,64", 1, "GATHER_ROW", "FP16, FP32, INT32 or UINT32"),
        ("half,1,128", "float,1,64", 1, "GATHER_ROW", "equal element widths"),
        ("float,1,128", "float,1,72", 1, "GATHER_ROW", "selected element count"),
        ("float,1,128,pto::BLayout::RowMajor,1,96", "float,1,64", 1, "GATHER_ROW", "static full valid shape"),
        ("float,8,128,pto::BLayout::ColMajor", "float,8,64", 1, "GATHER_ROW", "contiguous row-major"),
        ("float,1,32776", "float,1,16384", 1, "GATHER_ROW", "UB budget"),
    ],
)
def test_a1_mask_gather_rejects_invalid_profile(tmp_path, source, destination, pattern, axis, diagnostic):
    cpp = tmp_path / "invalid.cpp"
    cpp.write_text(
        "#include <pto/pto-inst.hpp>\n"
        + f"using Source = pto::Tile<pto::TileType::Vec,{source}>;\n"
        + f"using Result = pto::Tile<pto::TileType::Vec,{destination}>;\n"
        + 'extern "C" __global__ AICORE void Invalid() { Source src; Result dst; '
        + f"pto::TGATHER<Result,Source,static_cast<pto::MaskPattern>({pattern}),pto::GatherAxis::{axis}>(dst,src); }}\n"
    )
    result = subprocess.run(native_command(cpp, tmp_path / "invalid.o", []), capture_output=True, text=True)
    assert result.returncode != 0
    assert diagnostic in result.stdout + result.stderr


@pytest.mark.parametrize(
    "fp16,rows,columns,alias",
    [
        (False, 80, 16, False),
        (False, 80, 16, True),
        (False, 1, 16, False),
        (False, 1, 16, True),
        (True, 16, 16, False),
        (True, 16, 16, True),
        (False, 264, 24, False),
        (True, 272, 80, True),
        (False, 2, 2040, False),
        (False, 2, 2048, True),
        (False, 2, 2056, False),
        (True, 2, 4080, False),
        (False, 16, 400, False),
        (True, 16, 400, False),
        (True, 2, 4096, True),
    ],
)
def test_a1_shared_copy_column_broadcast_cpu_offline(tmp_path, fp16, rows, columns, alias):
    # Reuse the established bitwise oracle without importing its NPU fixture.
    # This protects source-repeat stride zero while mask identity uses stride 8.
    compiler = shutil.which("clang++-19")
    assert compiler, "requires clang++-19 for column broadcast CPU reference"
    source = ROOT / "tests/compile/a1_col_broadcast_bits.cpp"
    definitions = [f"-DPTO_COL_ROWS={rows}", f"-DPTO_COL_COLS={columns}"]
    if fp16:
        definitions.append("-DPTO_COL_FP16")
    if alias:
        definitions.append("-DPTO_COL_ALIAS")
    binary = tmp_path / "column-oracle"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            "-D__CPU_SIM",
            f"-I{ROOT / 'include'}",
            *definitions,
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
        native_command(source, tmp_path / "column-native.o", definitions), check=True, capture_output=True, text=True
    )
