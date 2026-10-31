# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Driver-free filter selection/order: independent CPU oracle and V1 m200 codegen."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/compile/a1_img2col_order.cpp"
CASES = [{"PHASE": p, "MANUAL": p % 2, "LVALUE": (p + 1) % 2} for p in range(4)]
CASES += [{"ROWS": m, "HEIGHT": 3, "WIDTH": 640, "M": 640, "K": 144, "PHASE": 2} for m in (16, 64, 112, 128, 224)]
CASES += [
    {"FH": 1, "FW": 1, "STEP": 1, "ROWS": 64, "K": 16},
    {"FH": 2, "FW": 3, "STEP": 5, "PHASE": 1, "ROWS": 16, "STRIDE": 2, "M": 16, "K": 96},
    {"FH": 5, "FW": 1, "STEP": 2, "PHASE": 3, "DILATION": 2, "PAD": 2, "HEIGHT": 16, "M": 16},
    {"ROWS": 64, "HEIGHT": 3, "WIDTH": 640, "C1": 16, "M": 640, "K": 2160, "PHASE": 3},
]

# Native RGB stem retains K96 per phase and a separate K48 tail. Selection
# must not pad these back to K144; an extra C1 case distinguishes the source
# group's original kernel extent from the smaller destination extent.
CASES += [{"COUNT": 6, "PHASE": p, "ROWS": 128, "HEIGHT": 3, "WIDTH": 640, "C1": 1, "STRIDE": 2} for p in range(4)]
CASES += [
    {"COUNT": 3, "PHASE": 6, "STEP": 1, "ROWS": 128, "HEIGHT": 3, "WIDTH": 640, "C1": 1, "STRIDE": 2},
    {"COUNT": 6, "PHASE": 3, "C1": 3, "K": 288},
]


def native_command(source, target, definitions):
    compiler = shutil.which("ccec")
    assert compiler, "requires driver-free m200 compilation"
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
        str(target),
    ]


def assert_no_executable_relocations(path):
    readelf = shutil.which("readelf")
    assert readelf, "requires native-object relocation inspection"
    report = subprocess.run([readelf, "-rW", str(path)], check=True, capture_output=True, text=True).stdout
    # A1 single-launch consumes this object without a device-link step. An
    # outlined helper call must not leave a relocation against executable text.
    assert "Relocation section '.rela.text" not in report
    assert "Relocation section '.rel.text" not in report


def optimized_target_ir(source, directory, name, definitions):
    llvm_dis = shutil.which("llvm-dis-19") or shutil.which("llvm-dis")
    assert llvm_dis, "requires LLVM bitcode inspection"
    bitcode, text = directory / f"{name}.bc", directory / f"{name}.ll"
    subprocess.run(
        native_command(source, bitcode, [*definitions, "-emit-llvm"]), check=True, capture_output=True, text=True
    )
    subprocess.run([llvm_dis, str(bitcode), "-o", str(text)], check=True, capture_output=True, text=True)
    return text.read_text()


@pytest.mark.parametrize("settings", CASES)
def test_a1_img2col_filter_order_cpu_and_offline(tmp_path, settings):
    compiler = shutil.which("clang++-19")
    assert compiler, "requires Clang19 CPU instruction semantics"
    definitions = [f"-DORDER_{k}={v}" for k, v in settings.items()]
    binary = tmp_path / "oracle"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            "-ffp-contract=off",
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
            "PTO_CPU_SIM_UB_BYTES": "131072",
            "PTO_CPU_SIM_L0A_BYTES": "65536",
            "PTO_CPU_SIM_L0C_BYTES": "262144",
            "PTO_CPU_SIM_NUM_CORES": "1",
            "PTO_CPU_SIM_TRACE_ENABLE": "0",
            "OMP_NUM_THREADS": "1",
        },
    )
    subprocess.run(
        native_command(SOURCE, tmp_path / "kernel.o", definitions), check=True, capture_output=True, text=True
    )
    assert_no_executable_relocations(tmp_path / "kernel.o")
    code = optimized_target_ir(SOURCE, tmp_path, "kernel", definitions)
    # The installed object disassembler reports <not available> for m200.
    # Inspect CCEC optimized target IR, retaining the exact V1 call-count gate.
    taps = settings.get("COUNT", settings.get("FH", 3) * settings.get("FW", 3))
    loads = [line for line in code.splitlines() if "call void @llvm.hivm.LOAD.L1.TO.L0A" in line]
    assert len(loads) == taps
    assert all("@llvm.hivm.LOAD.L1.TO.L0A.3D.f16(" in line for line in loads)
    assert "L3D" not in code and " = phi " not in code
    control = optimized_target_ir(SOURCE, tmp_path, "control", [*definitions, "-DORDER_NATIVE_CONTROL"])
    control_loads = [line for line in control.splitlines() if "call void @llvm.hivm.LOAD.L1.TO.L0A" in line]
    # Strip only LLVM alias-metadata IDs. Call order, destination addresses,
    # filter coordinates, source C1, M-repeat count and jump remain exact.
    assert [line.split(", !")[0] for line in loads] == [line.split(", !")[0] for line in control_loads]


@pytest.mark.parametrize(
    "order,diagnostic",
    [
        ("std::integer_sequence<int,0,4,8,3,7,2,6,1,1>", "filter order requires 1 to 127 unique indices"),
        ("std::integer_sequence<int,-1,1,2,3,4,5,6,7,8>", "filter order requires 1 to 127 unique indices"),
        ("std::integer_sequence<int,0,1,2,3,4,5,6,7,65025>", "filter order requires 1 to 127 unique indices"),
        ("std::integer_sequence<int>", "filter order requires 1 to 127 unique indices"),
        ("std::make_integer_sequence<int,128>", "filter order requires 1 to 127 unique indices"),
        ("std::integer_sequence<int,0,1,2>", "one C16 output block per selected filter"),
    ],
)
def test_a1_img2col_filter_order_rejects_invalid_profiles(tmp_path, order, diagnostic):
    result = subprocess.run(
        native_command(SOURCE, tmp_path / "invalid.o", [f"-DORDER_TYPE={order}"]), capture_output=True, text=True
    )
    assert result.returncode != 0 and diagnostic in result.stderr, result.stderr


@pytest.mark.parametrize(
    "definitions", [["-DORDER_INVALID_METADATA"], ["-DORDER_TYPE=std::integer_sequence<int,0,1,2,3,4,5,6,7,9>"]]
)
def test_a1_img2col_filter_order_rejects_runtime_configuration(tmp_path, definitions):
    compiler = shutil.which("clang++-19")
    assert compiler
    binary = tmp_path / "invalid-metadata"
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
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode != 0 and "bounded filter selection" in result.stderr
    code = optimized_target_ir(SOURCE, tmp_path, "invalid", definitions)
    assert "trap" in code.lower() and "call void @llvm.hivm.LOAD.L1.TO.L0A" not in code


@pytest.mark.parametrize(
    "definitions",
    [
        [],
        ["-DPTO_A1_IMG_FILTER=1", "-DPTO_A1_IMG_K=16", "-DPTO_A1_IMG_POS_K=16"],
        ["-DPTO_A1_IMG_FILTER=3", "-DPTO_A1_IMG_DILATION=2", "-DPTO_A1_IMG_MANUAL"],
    ],
)
def test_a1_existing_contiguous_v2_profile_is_unchanged(tmp_path, definitions):
    source = ROOT / "tests/compile/a1_img2col.cpp"
    compiler = shutil.which("clang++-19")
    assert compiler
    binary = tmp_path / "contiguous"
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
        native_command(source, tmp_path / "contiguous.o", definitions), check=True, capture_output=True, text=True
    )
    assert_no_executable_relocations(tmp_path / "contiguous.o")
    code = optimized_target_ir(source, tmp_path, "contiguous", definitions)
    calls = [line for line in code.splitlines() if "call void @llvm.hivm.LOAD.L1.TO.L0A.3D" in line]
    assert len(calls) == 1 and "@llvm.hivm.LOAD.L1.TO.L0A.3D.f16(" not in calls[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
