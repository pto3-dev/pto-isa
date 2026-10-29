# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""310P Vec-to-Vec TMOV preserves all bit patterns and adjacent UB storage."""

import ctypes
from pathlib import Path
import pytest

torch = pytest.importorskip("torch")
simpler = pytest.importorskip("simpler_setup")


@pytest.mark.parametrize(
    "count,dtype,alias",
    [(16, "half", 0), (80, "half", 0), (960, "half", 0), (80, "float", 0), (960, "float", 0), (80, "float", 1)],
)
def test_a1_vec_copy(tmp_path, count, dtype, alias):
    try:
        acl = ctypes.CDLL("libascendcl.so")
    except OSError:
        pytest.skip("Ascend runtime unavailable")
    acl.aclrtGetSocName.restype = ctypes.c_char_p
    if not (acl.aclrtGetSocName() or b"").startswith(b"Ascend310P"):
        pytest.skip("requires Ascend310P")
    root = Path(__file__).resolve().parents[2]
    cpp = tmp_path / "copy.cpp"
    cpp.write_text(
        f"#define TEST_COUNT {count}\n#define TEST_TYPE {dtype}\n#define TEST_ALIAS {alias}\n"
        + (root / "tests/compile/a1_vec_copy.cpp").read_text()
    )
    image = simpler.KernelCompiler("a1").compile_single_launch(str(cpp), str(root))
    itype = torch.int16 if dtype == "half" else torch.int32
    ftype = torch.float16 if dtype == "half" else torch.float32
    with simpler.A1SingleLaunch(image, device_id=0) as runner:
        for seed in (3203, 3209, 3217):
            g = torch.Generator().manual_seed(seed)
            bits = torch.randint(-32768, 32768, (count + 32,), generator=g, dtype=torch.int32).to(itype)
            if dtype == "float":
                bits = bits | (torch.randint(-32768, 32768, (count + 32,), generator=g, dtype=torch.int32) << 16)
            bits[16:20] = torch.tensor(
                [0, -32768, 0x7C00, 0x7E01] if dtype == "half" else [0, -2147483648, 0x7F800000, 0x7FC01234],
                dtype=itype,
            )
            inp = bits.view(ftype)
            original = bits.clone()
            guard = torch.full((count + 32,), 7, dtype=ftype)
            before = guard.view(itype).clone()
            runner(inp, guard)
            result = guard.view(itype)
            assert torch.equal(result[16:-16], original[16:-16])
            assert torch.equal(inp.view(itype), original)
            assert torch.equal(result[:16], before[:16]) and torch.equal(result[-16:], before[-16:])


@pytest.mark.parametrize("src,dst", [("float, 1, 80", "float, 1, 64"), ("half, 1, 80", "float, 1, 80")])
def test_a1_vec_copy_rejects_mismatch(tmp_path, src, dst):
    root = Path(__file__).resolve().parents[2]
    cpp = tmp_path / "invalid.cpp"
    cpp.write_text(f"""
#include <pto/pto-inst.hpp>
using S = pto::Tile<pto::TileType::Vec, {src}>;
using D = pto::Tile<pto::TileType::Vec, {dst}>;
extern "C" __global__ AICORE void Invalid(__gm__ int64_t*) {{
 S s; D d; pto::TASSIGN<0>(s); pto::TASSIGN<32768>(d); pto::TMOV(d,s);
}}
""")
    with pytest.raises(RuntimeError, match="A1 Vec-to-Vec moves require matching"):
        simpler.KernelCompiler("a1").compile_single_launch(str(cpp), str(root))
