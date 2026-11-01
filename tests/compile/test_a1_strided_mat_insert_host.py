# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exact CPU, emitted DMA geometry and release guards for row-strided insertion."""

import os
import re
import shutil
import signal
import subprocess
from pathlib import Path

import pytest

from test_a1_img2col_order_host import assert_no_executable_relocations, native_command, optimized_target_ir

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/compile/a1_strided_mat_insert.cpp"
# Source rows/columns, destination rows/columns, logical row step, row/column origin.
CASES = [
    (16, 32, 2304, 64, 145, 128, 16),
    (3, 16, 2304, 16, 145, 128, 0),
    (3, 32, 2304, 64, 145, 128, 16),
    (8, 16, 64, 64, 7, 14, 32),
    (3, 32, 64, 64, 1, 5, 16),
    (64, 16, 128, 32, 1, 31, 16),
    (4096, 16, 8192, 16, 1, 4096, 0),
    (4096, 16, 8192, 16, 2, 1, 0),
    (16, 16, 4608, 32, 145, 2432, 16),
    (16, 32, 4112, 64, 257, 256, 16),
    (1, 16, 16, 16, 65536, 15, 0),
    (16, 16, 32, 16, 1, 0, 0),
]
ENV = {
    **os.environ,
    "PTO_CPU_SIM_L1_BYTES": "1048576",
    "PTO_CPU_SIM_UB_BYTES": "131072",
    "PTO_CPU_SIM_NUM_CORES": "1",
    "PTO_CPU_SIM_TRACE_ENABLE": "0",
    "OMP_NUM_THREADS": "1",
}


def cpu_binary(directory, definitions=()):
    compiler = shutil.which("clang++-19")
    assert compiler, "requires Clang19 for independent raw-bit CPU controls"
    binary = directory / "oracle"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            "-DNDEBUG",
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
    return binary


def kernel_body(code, name):
    result = re.search(rf"define [^\n]*@{name}\([^\n]*\{{\n(.*?)^\}}", code, re.M | re.S)
    assert result, code
    return result.group(1)


def check_descriptors(body, settings):
    rows, cols, dst_rows, dst_cols, step, row_offset, col_offset = settings
    calls = re.findall(
        r"call void @llvm\.hivm\.MOV\.UB\.TO\.L1\(ptr addrspace\(2\) (.*?), "
        r"ptr addrspace\(6\) (.*?), i64 (-?\d+)\)",
        body,
    )
    assert len(calls) == (1 if step == 1 and cols == 16 else ((rows + 4094) // 4095) * (cols // 16))
    copied = {}

    def origin(pointer):
        return 0 if pointer == "null" else int(re.search(r"inttoptr \(i64 (\d+)", pointer).group(1)) // 2

    for dst, src, encoded in calls:
        descriptor = int(encoded)
        count, length = descriptor >> 4 & 0xFFF, descriptor >> 16 & 0xFFFF
        src_gap, dst_gap = descriptor >> 32 & 0xFFFF, descriptor >> 48 & 0xFFFF
        assert 1 <= count <= 4095 and length >= 1 and descriptor & 15 == 0
        for burst in range(count):
            for lane in range(length * 16):
                destination = origin(dst) - 256 + burst * (length + dst_gap) * 16 + lane
                source = origin(src) + burst * (length + src_gap) * 16 + lane
                assert 0 <= destination < dst_rows * dst_cols and 0 <= source < rows * cols
                assert destination not in copied
                copied[destination] = source
    expected = {}
    for row in range(rows):
        for column in range(cols):
            n1, n0 = divmod(col_offset + column, 16)
            destination = n1 * dst_rows * 16 + (row_offset + row * step) * 16 + n0
            expected[destination] = row * cols + column
    assert copied == expected
    assert (
        body.index("@llvm.hivm.SET.FLAG") < body.index("@llvm.hivm.WAIT.FLAG") < body.index("@llvm.hivm.MOV.UB.TO.L1")
    )


@pytest.mark.parametrize("settings", CASES)
def test_strided_insert_cpu_and_native(tmp_path, settings):
    names = ("ROWS", "COLS", "DST_ROWS", "DST_COLS", "STEP", "ROW", "COL")
    definitions = [f"-DSI_{key}={value}" for key, value in zip(names, settings)]
    binary = cpu_binary(tmp_path, definitions)
    subprocess.run([str(binary)], env=ENV, check=True, capture_output=True, text=True)
    target = tmp_path / "insert.o"
    subprocess.run(native_command(SOURCE, target, definitions), check=True, capture_output=True, text=True)
    assert_no_executable_relocations(target)
    code = optimized_target_ir(SOURCE, tmp_path, "insert", definitions)
    check_descriptors(kernel_body(code, "A1StridedMatInsert"), settings)
    assert "@llvm.hivm.TRAP()" in kernel_body(code, "A1StridedMatInsertDynamic")


@pytest.fixture(scope="module")
def default_binary(tmp_path_factory):
    return cpu_binary(tmp_path_factory.mktemp("strided-insert-release"))


def test_compact_sparse_weights(default_binary):
    # Six RGB/C16/N16/N32 geometries, two UB banks, three reloads after one clear.
    subprocess.run([str(default_binary), "sparse"], env=ENV, check=True, capture_output=True, text=True)


@pytest.mark.parametrize("row,col", [(129, 16), (128, 1), (128, 48), (65535, 0), (0, 65535)])
def test_release_invalid_window_aborts(default_binary, row, col):
    result = subprocess.run([str(default_binary), str(row), str(col)], env=ENV, capture_output=True, text=True)
    assert result.returncode == -signal.SIGABRT
    assert "A1 strided insertion requires a bounded window" in result.stderr


@pytest.mark.parametrize(
    "source,destination,step,diagnostic",
    [
        (
            "Tile<TileType::Vec, half,16,16>",
            "Tile<TileType::Mat,half,2304,32,BLayout::ColMajor,2304,32,SLayout::RowMajor,512>",
            0,
            "row step",
        ),
        (
            "Tile<TileType::Vec, half,16,16>",
            "Tile<TileType::Mat,half,2304,32,BLayout::ColMajor,2304,32,SLayout::RowMajor,512>",
            65537,
            "row step",
        ),
        (
            "Tile<TileType::Vec, float,16,16>",
            "Tile<TileType::Mat,half,2304,32,BLayout::ColMajor,2304,32,SLayout::RowMajor,512>",
            145,
            "FP16 Vec source",
        ),
        ("Tile<TileType::Vec, half,16,16>", "Tile<TileType::Vec,half,32,32>", 145, "FP16 Vec source"),
        (
            "Tile<TileType::Vec, half,16,16>",
            "Tile<TileType::Mat,half,2304,32,BLayout::RowMajor,2304,32,SLayout::RowMajor,512>",
            145,
            "full FP16 NZ destination",
        ),
        (
            "Tile<TileType::Vec, half,16,16,BLayout::RowMajor,8,16>",
            "Tile<TileType::Mat,half,2304,32,BLayout::ColMajor,2304,32,SLayout::RowMajor,512>",
            145,
            "full C16-aligned ND source",
        ),
        (
            "Tile<TileType::Vec, half,16,24>",
            "Tile<TileType::Mat,half,2304,32,BLayout::ColMajor,2304,32,SLayout::RowMajor,512>",
            145,
            "BFractal_ is RowMajor",
        ),
        (
            "Tile<TileType::Vec, half,17,16>",
            "Tile<TileType::Mat,half,2304,32,BLayout::ColMajor,2304,32,SLayout::RowMajor,512>",
            145,
            "source span exceeds",
        ),
        (
            "Tile<TileType::Vec, half,16,64>",
            "Tile<TileType::Mat,half,2304,32,BLayout::ColMajor,2304,32,SLayout::RowMajor,512>",
            145,
            "source span exceeds",
        ),
    ],
)
def test_strided_insert_rejects_types(tmp_path, source, destination, step, diagnostic):
    cpp = tmp_path / "invalid.cpp"
    cpp.write_text(f"""
#include <pto/pto-inst.hpp>
using namespace pto;
extern "C" __global__ AICORE void Invalid() {{
 {source} source; {destination} destination;
 TINSERT<{step}>(destination,source,0,0);
}}
""")
    result = subprocess.run(native_command(cpp, tmp_path / "invalid.o", []), capture_output=True, text=True)
    assert result.returncode != 0 and diagnostic in result.stderr, result.stderr


def test_legacy_split_modes_remain_unambiguous(default_binary):
    subprocess.run([str(default_binary), "split"], env=ENV, check=True, capture_output=True, text=True)


@pytest.mark.parametrize("architecture,available", [("dav-m200", True), ("dav-c220", False), ("dav-c310", False)])
def test_row_step_overload_backend_gate(tmp_path, architecture, available):
    cpp = tmp_path / "profile.cpp"
    cpp.write_text("#include <pto/pto-inst.hpp>\n")
    command = native_command(cpp, tmp_path / "unused.o", [])
    command[command.index("-c")] = "-E"
    command[command.index("--cce-aicore-arch=dav-m200")] = f"--cce-aicore-arch={architecture}"
    # A2/A5 preprocessing emits separate Cube and vector outputs. Keep
    # both on stdout instead of incorrectly specifying a single -o path.
    result = subprocess.run(command[:-2], check=True, capture_output=True, text=True)
    declaration = "TINSERT_IMPL<DstRowStep>(dst, src, indexRow, indexCol);"
    assert (declaration in result.stdout) is available


def test_strided_insert_double_buffer_events(tmp_path):
    cpp = tmp_path / "pipeline.cpp"
    cpp.write_text("""
#include <pto/pto-inst.hpp>
using namespace pto;
using M = Tile<TileType::Mat,half,2304,64,BLayout::ColMajor,2304,64,SLayout::RowMajor,512>;
using V = Tile<TileType::Vec,half,16,32>;
using GM = GlobalTensor<half,Shape<1,1,1,2304,64>,Stride<1,1,1,64,1>>;
using GV = GlobalTensor<half,Shape<1,1,1,16,32>,Stride<1,1,1,32,1>>;
extern "C" __global__ AICORE void DoubleBuffer(__gm__ half* input,__gm__ half* initial) {
 M matrix; V b0,b1;
 TASSIGN<0>(matrix); TASSIGN<0>(b0); TASSIGN<1024>(b1);
 GM gi(initial); TLOAD(matrix,gi);
 GV g0(input),g1(input+512),g2(input+1024),g3(input+1536),g4(input+2048);
 Event<Op::TLOAD,Op::TINSERT_V2M,false,EVENT_ID0> r0=TLOAD(b0,g0);
 Event<Op::TLOAD,Op::TINSERT_V2M,false,EVENT_ID1> r1=TLOAD(b1,g1);
 Event<Op::TINSERT_V2M,Op::TLOAD,false,EVENT_ID0> f0=TINSERT<145>(matrix,b0,128,16,r0);
 Event<Op::TINSERT_V2M,Op::TLOAD,false,EVENT_ID1> f1=TINSERT<145>(matrix,b1,128,16,r1);
 Event<Op::TLOAD,Op::TINSERT_V2M,false,EVENT_ID0> r2=TLOAD(b0,g2,f0);
 Event<Op::TINSERT_V2M,Op::TLOAD,false,EVENT_ID0> f2=TINSERT<145>(matrix,b0,128,16,r2);
 Event<Op::TLOAD,Op::TINSERT_V2M,false,EVENT_ID1> r3=TLOAD(b1,g3,f1);
 TINSERT<145>(matrix,b1,128,16,r3);
 Event<Op::TLOAD,Op::TINSERT_V2M,false,EVENT_ID0> r4=TLOAD(b0,g4,f2);
 TINSERT<145>(matrix,b0,128,16,r4);
 pipe_barrier(PIPE_ALL);
}
""")
    target = tmp_path / "pipeline.o"
    subprocess.run(native_command(cpp, target, []), check=True, capture_output=True, text=True)
    assert_no_executable_relocations(target)
    body = kernel_body(optimized_target_ir(cpp, tmp_path, "pipeline", []), "DoubleBuffer")
    assert body.count("call void @llvm.hivm.MOV.UB.TO.L1(") == 10
    for instruction in ("SET.FLAG", "WAIT.FLAG"):
        instruction = re.escape(instruction + ".REG")
        assert len(re.findall(rf"call void @llvm\.hivm\.{instruction}\(i64 4, i64 5, i64 [01]\)", body)) == 5
        assert len(re.findall(rf"call void @llvm\.hivm\.{instruction}\(i64 5, i64 4, i64 [01]\)", body)) == 3
