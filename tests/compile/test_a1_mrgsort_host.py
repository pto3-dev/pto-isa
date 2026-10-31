# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A1 public explicit-scratch merge: stable compact pairs and exact prefixes."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/compile/a1_mrgsort.cpp"
PROFILES = [
    (2, (16, 32, 16, 16), 48),
    (3, (16, 48, 32, 16), 96),
    (4, (32, 48, 16, 64), 160),
    (2, (512, 512, 16, 16), 1024),
    (2, (512, 512, 16, 16), 300),
    (3, (256, 512, 128, 16), 300),
    (4, (512, 512, 256, 256), 1536),
    (4, (32, 48, 16, 64), 4),
]


@pytest.mark.parametrize("ways,lengths,output", PROFILES)
@pytest.mark.parametrize("exhausted", [False, True])
def test_a1_mrgsort_exact_cpu_native_model_and_offline(tmp_path, ways, lengths, output, exhausted):
    compiler, native = shutil.which("clang++-19"), shutil.which("ccec")
    if not compiler or not native:
        pytest.skip("requires clang++-19 and CCEC for CPU/offline validation")
    definitions = [f"-DMERGE_WAYS={ways}", f"-DMERGE_OUTPUT={output}", f"-DMERGE_EXHAUSTED={str(exhausted).lower()}"]
    definitions += [f"-DMERGE_P{i}={length}" for i, length in enumerate(lengths)]
    for model in [False, True]:
        binary = tmp_path / ("production-model" if model else "cpu-reference")
        subprocess.run(
            [
                compiler,
                "-std=c++20",
                "-O1",
                "-Werror",
                "-D__CPU_SIM",
                f"-I{ROOT / 'include'}",
                *definitions,
                *(["-DA1_MERGE_NATIVE_MODEL"] if model else []),
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
            for alias in range(2 * ways + 1):
                result = subprocess.run([str(binary), str(alias)], capture_output=True, text=True)
                assert result.returncode == 86
            subprocess.run([str(binary), "readonly-alias"], check=True, capture_output=True, text=True)
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
    "sources,destination,scratch,diagnostic",
    [
        (["half,1,32", "float,1,32"], "float,1,64", "float,1,512", "one-row FP32 Vec"),
        (["float,2,32", "float,1,32"], "float,1,64", "float,1,768", "one-row FP32 Vec"),
        (["float,1,24", "float,1,32"], "float,1,56", "float,1,512", "complete 16-pair groups"),
        (["float,1,32", "float,1,32"], "float,1,72", "float,1,512", "cannot exceed"),
        (["float,1,32", "float,1,32"], "float,1,64", "float,1,504", "two eight-field proposal banks"),
        (["float,1,32", "float,1,32"], "float,1,64", "half,1,1024", "one-row FP32 Vec"),
        (["float,1,32", "float,1,32"], "float,1,64", "float,1,32776", "UB budget"),
        (["float,1,4096", "float,1,32"], "float,1,64", "float,1,32768", "proposal scratch exceeds"),
        (["float,8,32,pto::BLayout::ColMajor", "float,1,32"], "float,1,288", "float,1,2304", "one-row FP32 Vec"),
        (
            ["float,1,32,pto::BLayout::RowMajor,1,16", "float,1,32"],
            "float,1,64",
            "float,1,512",
            "static full valid shape",
        ),
        (["float,1,32", "float,1,32", "half,1,32"], "float,1,96", "float,1,768", "one-row FP32 Vec"),
        (["float,1,32", "float,1,32", "float,1,32", "float,2,16"], "float,1,128", "float,1,1024", "one-row FP32 Vec"),
    ],
)
def test_a1_mrgsort_rejects_invalid_profile(tmp_path, sources, destination, scratch, diagnostic):
    native = shutil.which("ccec")
    if not native:
        pytest.skip("requires CCEC for A1 template rejection")
    typedefs = [f"using S{i}=pto::Tile<pto::TileType::Vec,{shape}>;" for i, shape in enumerate(sources)]
    declarations = [f"S{i} s{i};" for i in range(len(sources))]
    names = [f"S{i}" for i in range(len(sources))]
    arguments = [f"s{i}" for i in range(len(sources))]
    cpp = tmp_path / "invalid.cpp"
    cpp.write_text(
        "#include <pto/pto-inst.hpp>\n"
        + "\n".join(typedefs)
        + f"""
using D=pto::Tile<pto::TileType::Vec,{destination}>;
using T=pto::Tile<pto::TileType::Vec,{scratch}>;
extern "C" __global__ AICORE void Invalid(__gm__ float*) {{
    D dst; T tmp; {" ".join(declarations)}
    pto::MrgSortExecutedNumList executed{{}};
    pto::TMRGSORT<D,T,{",".join(names)},false>(dst,executed,tmp,{",".join(arguments)});
}}
"""
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
    assert diagnostic in result.stderr


def test_a1_mrgsort_size_query_and_no_implicit_scratch(tmp_path):
    native = shutil.which("ccec")
    if not native:
        pytest.skip("requires CCEC for A1 signature validation")
    cpp = tmp_path / "size.cpp"
    cpp.write_text("""
#include <pto/pto-inst.hpp>
using S=pto::Tile<pto::TileType::Vec,float,1,32>;
static_assert(pto::GETMRGSORTTMPSIZE<S,S>()==512);
static_assert(pto::GETMRGSORTTMPSIZE<S,S,S>()==768);
static_assert(pto::GETMRGSORTTMPSIZE<S,S,S,S>()==1024);
extern "C" __global__ AICORE void Check(__gm__ float*) {}
""")
    command = [
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
        str(tmp_path / "size.o"),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    cpp.write_text("""
#include <pto/pto-inst.hpp>
extern "C" __global__ AICORE void Invalid(__gm__ float*) {
    pto::Tile<pto::TileType::Vec,float,1,256> src,dst;
    pto::TMRGSORT(dst,src,64);
}
""")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "requires the existing explicit-scratch multi-list form" in result.stderr
