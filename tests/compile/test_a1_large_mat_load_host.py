# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Driver-free large resident Mat storage, exact DMA addresses and unchanged L0 bounds."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from test_a1_img2col_order_host import assert_no_executable_relocations, native_command, optimized_target_ir

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/compile/a1_large_mat_load.cpp"
CASES = [
    (2048, 32, 48, 0),
    (2304, 16, 16, 0),
    (2304, 32, 48, 0),
    (4096, 16, 32, 0),
    (4608, 32, 48, 0),
    (8192, 32, 48, 0),
    (32768, 16, 16, 0),
    (16384, 32, 48, 0),
    (2304, 32, 48, 1),
    (16, 16, 1048576, 0),
    (16, 16, 1048592, 0),
]


def check_native_nz_addresses(code, rows, columns, stride):
    """Decode actual V1 descriptors; check every copied lane against NZ algebra."""
    offsets = dict(re.findall(r"(%\w+) = getelementptr inbounds half, ptr addrspace\(1\) %0, i64 (\d+)", code))
    calls = re.findall(
        r"call void @llvm\.hivm\.MOV\.OUT\.TO\.L1\(ptr addrspace\(2\) (.*?), "
        r"ptr addrspace\(1\) (%\w+), i64 (\d+), i64 (\d+)\)",
        code,
    )
    assert calls, code
    written = bytearray(rows * columns)
    for destination, source, encoded, padding in calls:
        base = 0 if destination == "null" else int(re.search(r"inttoptr \(i64 (\d+)", destination).group(1)) // 2
        # SID:4, nburst:12, length:16, source gap:16, destination gap:16.
        descriptor = int(encoded)
        count = (descriptor >> 4) & 0xFFF
        length = (descriptor >> 16) & 0xFFFF
        source_gap = (descriptor >> 32) & 0xFFFF
        destination_gap = (descriptor >> 48) & 0xFFFF
        assert descriptor & 0xF == 0 and padding == "0"
        assert 1 <= count <= 4095 and length >= 1
        origin = int(offsets[source]) - 16
        for burst in range(count):
            dst = base + burst * (length + destination_gap) * 16
            src = origin + burst * (length + source_gap) * 16
            for lane in range(length * 16):
                position = dst + lane
                assert 0 <= position < len(written) and not written[position]
                plane, within = divmod(position, rows * 16)
                row, col = divmod(within, 16)
                assert src + lane == row * stride + plane * 16 + col
                written[position] = 1
    assert all(written), "a burst split or plane transition left resident data unwritten"


@pytest.mark.parametrize(
    "rows,columns,stride,row_major,mode",
    [(*case, 0) for case in CASES] + [(4096, 16, 32, 0, 1), (4608, 32, 48, 0, 2), (32768, 16, 16, 0, 2)],
)
def test_large_mat_load_cpu_and_native(tmp_path, rows, columns, stride, row_major, mode):
    compiler = shutil.which("clang++-19")
    assert compiler, "requires Clang19 for the independent logical CPU oracle"
    definitions = [
        f"-DMAT_ROWS={rows}",
        f"-DMAT_COLS={columns}",
        f"-DMAT_STRIDE={stride}",
        f"-DMAT_ROW_MAJOR={row_major}",
        f"-DMAT_STAGE_MODE={mode}",
    ]
    binary = tmp_path / "oracle"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            "-D__CPU_SIM",
            f"-I{ROOT / 'include'}",
            *definitions,
            str(SOURCE),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [str(binary)],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PTO_CPU_SIM_L1_BYTES": "1048576",
            "PTO_CPU_SIM_NUM_CORES": "1",
            "PTO_CPU_SIM_TRACE_ENABLE": "0",
            "OMP_NUM_THREADS": "1",
        },
    )
    target = tmp_path / "load.o"
    subprocess.run(native_command(SOURCE, target, definitions), check=True, capture_output=True, text=True)
    assert_no_executable_relocations(target)
    # The native compiler keeps larger ZZ loops rolled. NZ profiles below
    # instantiate literal V1 descriptors, permitting an independent address oracle.
    if not row_major and mode == 0:
        code = optimized_target_ir(SOURCE, tmp_path, "load", definitions)
        check_native_nz_addresses(code, rows, columns, stride)


@pytest.mark.parametrize("operation", ["left", "right", "insert"])
def test_large_mat_preserves_l0_consumer_limits(tmp_path, operation):
    destination = (
        "Tile<TileType::Left, half, 2304, 16, BLayout::RowMajor, 2304, 16, SLayout::RowMajor, 512>"
        if operation != "right"
        else "TileRight<half,2304,16>"
    )
    statement = "TINSERT(destination, source);" if operation == "insert" else "TMOV(destination, source);"
    cpp = tmp_path / "invalid.cpp"
    cpp.write_text(f"""
#include <pto/pto-inst.hpp>
using namespace pto;
extern "C" __global__ AICORE void Invalid() {{
 Tile<TileType::Mat, half, 2304, 16, BLayout::ColMajor, 2304, 16, SLayout::RowMajor, 512> source;
 {destination} destination;
 {statement}
}}
""")
    result = subprocess.run(native_command(cpp, tmp_path / "invalid.o", []), capture_output=True, text=True)
    assert result.returncode != 0, "resident storage must not enlarge L0 operands"
    assert "64 KiB L0" in result.stderr, result.stderr


@pytest.mark.parametrize(
    "rows,columns,definitions", [(32784, 16, []), (16400, 32, []), (32768, 16, ["-DPTO_CBUF_SIZE_BYTES=524288"])]
)
def test_large_mat_rejects_capacity_overflow(tmp_path, rows, columns, definitions):
    result = subprocess.run(
        native_command(SOURCE, tmp_path / "overflow.o", [*definitions, f"-DMAT_ROWS={rows}", f"-DMAT_COLS={columns}"]),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "A1 matrix inputs require" in result.stderr, result.stderr
