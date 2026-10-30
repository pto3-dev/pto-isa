# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Raw INT32/UINT32 A1 DMA; CPU correctness and offline CCEC only."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/compile/a1_integer_dma.cpp"
CASES = [
    (16, 32, 32, 32),
    (16, 32, 96, 128),
    (3, 8, 24, 40),
    (4096, 8, 16, 24),
    (4, 8, 8 + 65536 * 8, 24),
    (4, 8, 24, 8 + 65536 * 8),
    (4, 8, 8 + 65535 * 8, 8 + 65535 * 8),
]


@pytest.mark.parametrize("unsigned", [False, True])
@pytest.mark.parametrize("rows,cols,source_stride,destination_stride", CASES)
def test_a1_integer_dma_cpu_and_offline(tmp_path, unsigned, rows, cols, source_stride, destination_stride):
    compiler = shutil.which("clang++-19")
    native = shutil.which("ccec")
    if not compiler or not native:
        pytest.skip("requires clang++-19 and CCEC for CPU/offline checks")
    definitions = [
        f"-DDMA_ROWS={rows}",
        f"-DDMA_COLS={cols}",
        f"-DDMA_SOURCE_STRIDE={source_stride}",
        f"-DDMA_DESTINATION_STRIDE={destination_stride}",
    ]
    if unsigned:
        definitions.append("-DDMA_UNSIGNED")
    binary = tmp_path / "cpu"
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
            str(SOURCE),
            "-o",
            str(tmp_path / "native.o"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "tile,tensor,diagnostic",
    [
        (
            "pto::Tile<pto::TileType::Vec, int16_t, 1, 16>",
            "pto::GlobalTensor<int16_t, pto::Shape<1,1,1,1,16>, pto::Stride<1,1,1,16,1>>",
            "A1 vector DMA requires FP16, FP32, INT32 or UINT32",
        ),
        (
            "pto::Tile<pto::TileType::Vec, uint32_t, 1, 8>",
            "pto::GlobalTensor<int32_t, pto::Shape<1,1,1,1,8>, pto::Stride<1,1,1,8,1>>",
            "A1 DMA requires tensor and tile data types to match",
        ),
        (
            "pto::Tile<pto::TileType::Vec, uint32_t, 16, 1, pto::BLayout::ColMajor>",
            "pto::GlobalTensor<uint32_t, pto::Shape<1,1,1,16,1>, pto::Stride<1,1,1,1,16>>",
            "A1 packed columns require FP16 or FP32",
        ),
        (
            "pto::Tile<pto::TileType::Vec, int32_t, 1, 16, pto::BLayout::RowMajor, 1, 8>",
            "pto::GlobalTensor<int32_t, pto::Shape<1,1,1,1,16>, pto::Stride<1,1,1,16,1>>",
            "A1 vector instructions require a static full valid shape",
        ),
    ],
)
def test_a1_integer_dma_rejects_unsupported_storage(tmp_path, tile, tensor, diagnostic):
    native = shutil.which("ccec")
    if not native:
        pytest.skip("requires CCEC for A1 template rejection")
    source = tmp_path / "invalid.cpp"
    source.write_text(f"""
#include <cstdint>
#include <pto/pto-inst.hpp>
using Tile = {tile};
using Input = {tensor};
using Element = typename Input::RawDType;
extern "C" __global__ AICORE void Invalid(__gm__ int32_t* address) {{
    Tile value;
    pto::TASSIGN<0>(value);
    Input input(reinterpret_cast<__gm__ Element*>(address));
    pto::TLOAD(value, input);
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
            str(source),
            "-o",
            str(tmp_path / "invalid.o"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert diagnostic in result.stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
