# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Verify bitwise A1 Vec panel insertion, pipeline events and guards."""

import ctypes
from pathlib import Path
import pytest

torch = pytest.importorskip("torch")
simpler = pytest.importorskip("simpler_setup")


@pytest.mark.parametrize(
    "dtype,srows,drows,row,col",
    [("float", 128, 128, 0, 0), ("float", 128, 128, 0, 16), ("float", 4, 8, 2, 16), ("half", 128, 128, 0, 16)],
)
def test_a1_vec_insert(tmp_path, dtype, srows, drows, row, col):
    try:
        acl = ctypes.CDLL("libascendcl.so")
    except OSError:
        pytest.skip("Ascend runtime unavailable")
    acl.aclrtGetSocName.restype = ctypes.c_char_p
    if not (acl.aclrtGetSocName() or b"").startswith(b"Ascend310P"):
        pytest.skip("requires Ascend310P")
    root = Path(__file__).resolve().parents[2]
    cpp = tmp_path / "insert.cpp"
    cpp.write_text(
        f"#define TEST_TYPE {dtype}\n#define TEST_SRC_ROWS {srows}\n#define TEST_DST_ROWS {drows}\n#define TEST_ROW {row}\n#define TEST_COL {col}\n"
        + (root / "tests/compile/a1_vec_insert.cpp").read_text()
    )
    image = simpler.KernelCompiler("a1").compile_single_launch(str(cpp), str(root))
    itype = torch.int16 if dtype == "half" else torch.int32
    ftype = torch.float16 if dtype == "half" else torch.float32
    with simpler.A1SingleLaunch(image, device_id=0) as runner:
        for seed in (3389, 3391):
            g = torch.Generator().manual_seed(seed)
            source = torch.randint(-32768, 32768, (srows, 16), generator=g, dtype=torch.int32).to(itype)
            if dtype == "float":
                source = source | (torch.randint(-32768, 32768, (srows, 16), generator=g, dtype=torch.int32) << 16)
            initial = torch.full((drows * 32 + 32,), 7, dtype=ftype)
            original = initial.view(itype).clone()
            expected = original.clone()
            expected[16:-16].view(drows, 32)[row : row + srows, col : col + 16] = source
            source_before = source.clone()
            runner(source.view(ftype), initial)
            assert torch.equal(initial.view(itype), expected)
            assert torch.equal(source, source_before)


@pytest.mark.parametrize("source,destination", [("float, 8, 32", "float, 8, 16"), ("half, 8, 16", "float, 8, 32")])
def test_a1_vec_insert_rejects_incompatible_tiles(tmp_path, source, destination):
    root = Path(__file__).resolve().parents[2]
    cpp = tmp_path / "invalid.cpp"
    cpp.write_text(f"""
#include <pto/pto-inst.hpp>
using S=pto::Tile<pto::TileType::Vec, {source}>;
using D=pto::Tile<pto::TileType::Vec, {destination}>;
extern "C" __global__ AICORE void Invalid(__gm__ int64_t*) {{
 S s; D d; pto::TASSIGN<0>(s); pto::TASSIGN<32768>(d); pto::TINSERT(d,s,0,0);
}}
""")
    with pytest.raises(RuntimeError, match="A1 Vec insertion requires matching data types and a bounded source tile"):
        simpler.KernelCompiler("a1").compile_single_launch(str(cpp), str(root))
