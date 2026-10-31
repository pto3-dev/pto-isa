# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Driver-free A1 static full-L1 admission and unchanged resource rejection."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def native_command(source, target, definitions=()):
    compiler = shutil.which("ccec")
    assert compiler, "requires CCEC for driver-free A1 checks"
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


@pytest.mark.parametrize(
    "kind,rows,columns,address,definitions,diagnostic",
    [
        ("Mat", 2048, 256, 0, (), None),
        ("Conv", 2048, 256, 0, (), None),
        ("Mat", 1920, 256, 0, (), None),
        ("Mat", 16, 16, 1024 * 1024 - 512, (), None),
        ("Mat", 16, 16, 1024 * 1024, (), "SA-0353"),
        ("Mat", 2064, 256, 0, (), "SA-0352"),
        ("Mat", 16, 16, 512 * 1024 + 16, (), "SA-0354"),
        ("Mat", 2048, 256, 0, ("-DPTO_CBUF_SIZE_BYTES=524288",), "SA-0352"),
    ],
)
def test_a1_static_l1_boundaries(tmp_path, kind, rows, columns, address, definitions, diagnostic):
    source = tmp_path / "allocation.cpp"
    tile = (
        f"pto::Tile<pto::TileType::Mat, half, {rows}, {columns}, pto::BLayout::ColMajor>"
        if kind == "Mat"
        else "pto::ConvTile<pto::TileType::Mat, half, 524288, pto::Layout::NC1HWC0, pto::ConvTileShape<1,16,32,64,16>>"
    )
    source.write_text(f"""
#include <cstdint>
#include <pto/pto-inst.hpp>
using Allocation = {tile};
extern "C" __global__ AICORE void Bind() {{
    Allocation tile;
    pto::TASSIGN<{address}>(tile);
}}
""")
    result = subprocess.run(
        native_command(source, tmp_path / "allocation.o", definitions), capture_output=True, text=True
    )
    if diagnostic is None:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0 and diagnostic in result.stderr, result.stderr


@pytest.mark.parametrize("overflow", [False, True])
def test_a1_existing_tile_metadata_and_ub_limit(tmp_path, overflow):
    definitions = ("-DPTO_A1_TEST_OVERFLOW",) if overflow else ()
    result = subprocess.run(
        native_command(ROOT / "tests/compile/a1_tile_buffers.cpp", tmp_path / "metadata.o", definitions),
        capture_output=True,
        text=True,
    )
    if overflow:
        assert result.returncode != 0 and "SA-0353" in result.stderr, result.stderr
    else:
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "architecture,capacity",
    [
        ("A1", 1024),
        ("A2A3", 512),
        ("A5", 512),
        ("A6", 512),
        ("KIRIN9030", 512),
        ("KIRIN9050", 1536),
        ("KIRINX90", 1024),
    ],
)
def test_static_l1_profile_preserves_other_architectures(tmp_path, architecture, capacity):
    compiler = shutil.which("clang++-19")
    assert compiler, "requires Clang19 for resource-profile checks"
    source = tmp_path / "profile.cpp"
    source.write_text(f"""
#include <pto/common/buffer_limits.hpp>
static_assert(PTO_CBUF_SIZE_BYTES == {capacity} *1024);
""")
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-fsyntax-only",
            f"-DPTO_NPU_ARCH_{architecture}",
            f"-I{ROOT / 'include'}",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
