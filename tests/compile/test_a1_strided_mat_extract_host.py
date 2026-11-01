# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exact CPU, emitted DMA geometry and release guards for block-strided extraction."""

import os
import re
import shutil
import signal
import subprocess
from pathlib import Path

import pytest

from test_a1_img2col_order_host import assert_no_executable_relocations, native_command, optimized_target_ir

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/compile/a1_strided_mat_extract.cpp"
DEFAULT = dict(SRC_ROWS=2304, SRC_COLS=32, DST_ROWS=144, DST_COLS=32, STEP=4, ROW=1776, COL=0, ORDINARY=0)
GEOMETRIES = (
    {},
    dict(DST_COLS=16, ROW=0, COL=16),
    dict(SRC_ROWS=4608, SRC_COLS=64, ROW=4080, COL=16),
    dict(SRC_ROWS=256, SRC_COLS=64, DST_ROWS=128, STEP=1, ROW=128, COL=32),
    dict(SRC_ROWS=32768, SRC_COLS=16, DST_ROWS=32, DST_COLS=16, STEP=2047, ROW=0),
    dict(SRC_ROWS=256, SRC_COLS=2048, DST_ROWS=32, STEP=1, ROW=224, COL=2016),
    dict(SRC_ROWS=16, SRC_COLS=16, DST_ROWS=16, DST_COLS=16, STEP=4294967295, ROW=0),
    dict(SRC_ROWS=4096, SRC_COLS=16, DST_ROWS=2048, DST_COLS=16, STEP=2, ROW=16),
    dict(SRC_ROWS=2048, DST_ROWS=1024, STEP=2, ROW=16),
    dict(SRC_ROWS=128, DST_ROWS=128, STEP=1, ROW=0, ORDINARY=1),
)
PROFILES = [{**DEFAULT, **geometry, "ROW_MAJOR": layout} for geometry in GEOMETRIES for layout in (0, 1)]
ENVIRONMENT = {
    **os.environ,
    "PTO_CPU_SIM_ARCH": "a2a3",
    "PTO_CPU_SIM_L1_BYTES": "1048576",
    "PTO_CPU_SIM_L0B_BYTES": "65536",
    "PTO_CPU_SIM_NUM_CORES": "1",
    "PTO_CPU_SIM_TRACE_ENABLE": "0",
    "OMP_NUM_THREADS": "1",
}


def definitions(profile):
    return [f"-DPTO_TEST_EXTRACT_{key}={value}" for key, value in profile.items()]


def cpu_command(binary, profile, release=False, source=SOURCE, l1=1048576, l0b=65536):
    compiler = shutil.which("clang++-19")
    assert compiler, "requires Clang19 CPU-SIM compilation"
    return [
        compiler,
        "-std=c++20",
        "-O1",
        "-D__CPU_SIM",
        f"-DPTO_CBUF_SIZE_BYTES={l1}",
        f"-DPTO_L0B_SIZE_BYTES={l0b}",
        f"-I{ROOT / 'include'}",
        *definitions(profile),
        *(["-DNDEBUG"] if release else []),
        str(source),
        "-o",
        str(binary),
    ]


def kernel_body(code, name):
    match = re.search(rf"^define[^\n]*@{name}\([^\n]*\)[^{{\n]*\{{\n(.*?)^\}}", code, re.MULTILINE | re.DOTALL)
    assert match, f"missing actual native kernel {name}"
    return match.group(1)


def pointer(argument):
    match = re.search(r"\bi64 (\d+)", argument)
    if match:
        return int(match.group(1))
    assert argument.rstrip().endswith("null"), "native panel address must be constant"
    return 0


def source_offset(row, column, profile):
    block = (
        ((row // 16) * (profile["SRC_COLS"] // 16) + column // 16)
        if profile["ROW_MAJOR"]
        else ((column // 16) * (profile["SRC_ROWS"] // 16) + row // 16)
    )
    return block * 256 + row % 16 * 16 + column % 16


def target_offset(row, column, columns):
    return (row // 16 * (columns // 16) + column // 16) * 256 + column % 16 * 16 + row % 16


def check_native_lanes(code, profile):
    body = kernel_body(code, "A1BlockExtract")
    calls = re.findall(
        r"call void @llvm\.hivm\.LOAD\.L1\.TO\.L0B\.2D\.f16\.addr\("
        r"(ptr addrspace\(4\).*?), (ptr addrspace\(2\).*?), i64 (\d+), i64 (\d+), i64 (\d+)\)",
        body,
    )
    if profile["ORDINARY"]:
        move = kernel_body(code, "A1BlockExtractMoveControl")
        move_calls = re.findall(
            r"call void @llvm\.hivm\.LOAD\.L1\.TO\.L0B\.2D\.f16\.addr\("
            r"(ptr addrspace\(4\).*?), (ptr addrspace\(2\).*?), i64 (\d+), i64 (\d+), i64 (\d+)\)",
            move,
        )
        assert sorted(calls) == sorted(move_calls)
    panels = profile["DST_COLS"] // 16
    assert len(calls) == panels
    assert "@llvm.hivm.TRAP()" not in body
    assert "@llvm.hivm.TRAP()" in kernel_body(code, "A1BlockExtractDynamic")
    src_bytes = profile["SRC_ROWS"] * profile["SRC_COLS"] * 2
    dst_bytes = profile["DST_ROWS"] * profile["DST_COLS"] * 2
    src_base, dst_base = (0 if src_bytes == 1048576 else 512), (0 if dst_bytes == 65536 else 512)
    repeat = profile["DST_ROWS"] // 16
    step = 1 if repeat == 1 else profile["STEP"] * (profile["SRC_COLS"] // 16 if profile["ROW_MAJOR"] else 1)
    expected_control = repeat << 16 | step << 24 | (panels - 1) << 44
    seen = set()
    count = 0
    for destination, source, control, transpose, mode in calls:
        dst, src = pointer(destination), pointer(source)
        panel = (dst - dst_base) // 512
        assert dst == dst_base + panel * 512 and 0 <= panel < panels and panel not in seen
        seen.add(panel)
        assert int(control) == expected_control and int(transpose) == 1 and int(mode) == 0
        assert src == src_base + 2 * source_offset(profile["ROW"], profile["COL"] + panel * 16, profile)
        for block in range(repeat):
            for row in range(16):
                for column in range(16):
                    actual_source = src + 2 * (block * step * 256 + row * 16 + column)
                    actual_target = dst + 2 * (block * panels * 256 + column * 16 + row)
                    logical_row, logical_col = block * 16 + row, panel * 16 + column
                    wanted_row = profile["ROW"] + block * profile["STEP"] * 16 + row
                    assert actual_source == src_base + 2 * source_offset(
                        wanted_row, profile["COL"] + logical_col, profile
                    )
                    assert actual_target == dst_base + 2 * target_offset(logical_row, logical_col, profile["DST_COLS"])
                    assert src_base <= actual_source < src_base + src_bytes
                    assert dst_base <= actual_target < dst_base + dst_bytes
                    count += 1
    assert count == profile["DST_ROWS"] * profile["DST_COLS"]
    return count


@pytest.mark.parametrize("profile", PROFILES)
def test_cpu_and_native(tmp_path, profile):
    for release in (False, True):
        binary = tmp_path / f"oracle-{release}"
        subprocess.run(cpu_command(binary, profile, release), check=True, capture_output=True, text=True, timeout=120)
        result = subprocess.run([str(binary)], env=ENVIRONMENT, check=True, capture_output=True, text=True, timeout=120)
        assert f"PASS {3 * profile['DST_ROWS'] * profile['DST_COLS']} bitwise FP16" in result.stdout
    target = tmp_path / "native.o"
    subprocess.run(
        native_command(SOURCE, target, definitions(profile)), check=True, capture_output=True, text=True, timeout=120
    )
    assert_no_executable_relocations(target)
    check_native_lanes(optimized_target_ir(SOURCE, tmp_path, "extract", definitions(profile)), profile)


@pytest.mark.parametrize("release", [False, True])
@pytest.mark.parametrize("row,column", [(1, 0), (0, 1), (1792, 0), (0, 16), (65520, 0), (0, 65520)])
def test_runtime_window_rejected(tmp_path, release, row, column):
    binary = tmp_path / "invalid-window"
    subprocess.run(
        cpu_command(binary, {**DEFAULT, "ROW_MAJOR": 0}, release),
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    result = subprocess.run(
        [str(binary), str(row), str(column)], env=ENVIRONMENT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == -signal.SIGABRT
    assert "A1 block extraction requires a bounded window" in result.stderr


@pytest.mark.parametrize(
    "settings",
    [
        {"STEP": 0},
        {"STEP": 8192},
        {"SRC_ROWS": 17},
        {"SRC_ROWS": 32784, "SRC_COLS": 16, "DST_COLS": 16},
        {"DST_ROWS": 2304, "DST_COLS": 16, "STEP": 1, "ROW": 0},
        {"DST_COLS": 48},
        {"DST_ROWS": 2048, "DST_COLS": 32, "STEP": 1, "ROW": 0},
    ],
)
def test_static_contract_rejected(tmp_path, settings):
    profile = {**DEFAULT, "ROW_MAJOR": 0, **settings}
    commands = [
        cpu_command(tmp_path / "invalid-cpu", profile),
        native_command(SOURCE, tmp_path / "invalid.o", definitions(profile)),
    ]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        assert result.returncode != 0
        diagnostic = (
            "Layout rows must be divisible by inner box rows"
            if settings.get("SRC_ROWS") == 17
            else "A1 block extraction"
        )
        assert diagnostic in result.stderr


@pytest.mark.parametrize("l1,l0b", [(512, 65536), (1048576, 512)])
def test_configured_capacity_rejected(tmp_path, l1, l0b):
    result = subprocess.run(
        cpu_command(tmp_path / "small-budget", {**DEFAULT, "ROW_MAJOR": 0}, l1=l1, l0b=l0b),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0
    assert "configured" in result.stderr


@pytest.mark.parametrize("architecture,available", [("dav-m200", True), ("dav-c220", False), ("dav-c310", False)])
def test_block_step_backend_gate(tmp_path, architecture, available):
    cpp = tmp_path / "profile.cpp"
    cpp.write_text("#include <pto/pto-inst.hpp>\n")
    command = native_command(cpp, tmp_path / "unused.o", [])
    command[command.index("-c")] = "-E"
    command[command.index("--cce-aicore-arch=dav-m200")] = f"--cce-aicore-arch={architecture}"
    result = subprocess.run(command[:-2], check=True, capture_output=True, text=True, timeout=120)
    declaration = "TEXTRACT_IMPL<SrcBlockStep>(dst, src, indexRow, indexCol);"
    assert (declaration in result.stdout) is available


@pytest.mark.parametrize("layout", [0, 1])
def test_two_right_banks_feed_cube(tmp_path, layout):
    source = ROOT / "tests/compile/a1_strided_mat_extract_cube.cpp"
    profile = {"ROW_MAJOR": layout}
    for release in (False, True):
        binary = tmp_path / f"cube-{release}"
        subprocess.run(
            cpu_command(binary, profile, release, source=source),
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        result = subprocess.run([str(binary)], env=ENVIRONMENT, check=True, capture_output=True, text=True, timeout=120)
        assert "PASS 1536 bitwise FP32 Cube values" in result.stdout
    target = tmp_path / "cube.o"
    subprocess.run(
        native_command(source, target, definitions(profile)), check=True, capture_output=True, text=True, timeout=120
    )
    assert_no_executable_relocations(target)
    code = optimized_target_ir(source, tmp_path, "cube", definitions(profile))
    body = kernel_body(code, "A1BlockExtractCube")
    assert len(re.findall(r"call void @llvm.hivm.LOAD.L1.TO.L0B.2D.f16.addr", body)) == 10
    assert "@llvm.hivm.TRAP()" not in body
    # Check exact bank handoffs; ID2 is the independent Left-ready event.
    for operation in ("SET", "WAIT"):
        flags = re.findall(rf"call void @llvm\.hivm\.{operation}\.FLAG\.REG\(i64 (\d+), i64 (\d+), i64 (\d+)\)", body)
        assert flags, "requires native pipe events, not source-level placeholders"
        assert [(int(s), int(d), int(i)) for s, d, i in flags if (int(s), int(d)) == (3, 2) and int(i) < 2] == [
            (3, 2, 0),
            (3, 2, 1),
            (3, 2, 0),
            (3, 2, 1),
            (3, 2, 0),
        ]
        assert [(int(s), int(d), int(i)) for s, d, i in flags if (int(s), int(d)) == (2, 3)] == [
            (2, 3, 0),
            (2, 3, 1),
            (2, 3, 0),
        ]

    loads = list(
        re.finditer(
            r"call void @llvm\.hivm\.LOAD\.L1\.TO\.L0B\.2D\.f16\.addr\("
            r"(ptr addrspace\(4\).*?), (ptr addrspace\(2\).*?), i64 (\d+), i64 (\d+), i64 (\d+)\)",
            body,
        )
    )
    products = list(
        re.finditer(
            r"call void @llvm\.hivm\.MAD\.f162f32\(ptr addrspace\(5\).*?, ptr addrspace\(3\).*?, "
            r"(ptr addrspace\(4\).*?), i64 (-?\d+)\)",
            body,
        )
    )
    assert len(products) == 5
    rows = (0, 16, 32, 48, 576)
    for index, row in enumerate(rows):
        bank = index % 2
        assert pointer(products[index].group(1)) == bank * 16384
        for panel in (0, 1):
            call = loads[index * 2 + panel]
            assert pointer(call.group(1)) == bank * 16384 + panel * 512
            profile = {"SRC_ROWS": 2304, "SRC_COLS": 32, "ROW_MAJOR": layout}
            assert pointer(call.group(2)) == 512 + 2 * source_offset(row, panel * 16, profile)
            assert tuple(map(int, call.groups()[2:])) == (9 << 16 | (8 if layout else 4) << 24 | 1 << 44, 1, 0)
        ready_set = list(re.finditer(rf"call void @llvm\.hivm\.SET\.FLAG\.REG\(i64 3, i64 2, i64 {bank}\)", body))[
            index // 2
        ]
        ready_wait = list(re.finditer(rf"call void @llvm\.hivm\.WAIT\.FLAG\.REG\(i64 3, i64 2, i64 {bank}\)", body))[
            index // 2
        ]
        assert loads[index * 2 + 1].start() < ready_set.start() < ready_wait.start() < products[index].start()
        if index >= 2:
            released = list(re.finditer(rf"call void @llvm\.hivm\.SET\.FLAG\.REG\(i64 2, i64 3, i64 {bank}\)", body))[
                index // 2 - 1
            ]
            reacquired = list(
                re.finditer(rf"call void @llvm\.hivm\.WAIT\.FLAG\.REG\(i64 2, i64 3, i64 {bank}\)", body)
            )[index // 2 - 1]
            assert products[index - 2].start() < released.start() < reacquired.start() < loads[index * 2].start()


@pytest.mark.parametrize("release", [False, True])
def test_portable_cpu_headers_without_native_budgets(tmp_path, release):
    binary = tmp_path / "portable-cpu"
    command = [
        arg
        for arg in cpu_command(binary, {**DEFAULT, "ROW_MAJOR": 0}, release)
        if not arg.startswith(("-DPTO_CBUF_SIZE_BYTES=", "-DPTO_L0B_SIZE_BYTES="))
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    result = subprocess.run([str(binary)], env=ENVIRONMENT, check=True, capture_output=True, text=True, timeout=120)
    assert "PASS 13824 bitwise FP16" in result.stdout
