# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Native A1 instruction checks; requires the matching Simpler A1 runtime."""

import ctypes
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
simpler = pytest.importorskip("simpler_setup")


@pytest.fixture(scope="module", autouse=True)
def require_310p_hardware():
    try:
        acl = ctypes.CDLL("libascendcl.so")
    except OSError:
        pytest.skip("Ascend runtime is unavailable")
    acl.aclrtGetSocName.restype = ctypes.c_char_p
    acl.aclrtGetSocName.argtypes = []
    soc = acl.aclrtGetSocName()
    if not soc or not soc.startswith(b"Ascend310P"):
        pytest.skip("This native regression requires Ascend310P hardware")


@pytest.mark.parametrize("division", [False, True])
@pytest.mark.parametrize("fp16", [False, True])
@pytest.mark.parametrize(
    ("size", "inplace"),
    [(16, False), (16, True), (80, False), (80, True), (8192, False), (8192, True), ("large", True)],
)
def test_a1_exp_div_native(tmp_path, division, fp16, size, inplace):
    count = (32720 if fp16 else 16336) if size == "large" else size
    root = Path(__file__).resolve().parents[2]
    source = (root / "tests/compile/a1_exp_div_vector.cpp").read_text()
    source = source.replace('extern "C" __global__ AICORE void A1ExpDiv', "AICORE void A1ExpDiv")
    definitions = f"#define PTO_A1_TEST_ELEMENTS {count}\n"
    if division:
        definitions += "#define PTO_A1_TEST_DIV\n"
    if fp16:
        definitions += "#define PTO_A1_TEST_FP16\n"
    if inplace:
        definitions += "#define PTO_A1_TEST_INPLACE\n"
    wrapper = r"""
#include "tensor.h"
extern "C" __global__ AICORE void NativeExpDiv(__gm__ int64_t* args) {
    auto* lhs = reinterpret_cast<__gm__ Tensor*>(args[0]);
    auto* rhs = reinterpret_cast<__gm__ Tensor*>(args[1]);
    auto* out = reinterpret_cast<__gm__ Tensor*>(args[2]);
    A1ExpDiv(reinterpret_cast<__gm__ Element*>(out->buffer.addr),
             reinterpret_cast<__gm__ Element*>(lhs->buffer.addr),
             reinterpret_cast<__gm__ Element*>(rhs->buffer.addr));
}
"""
    path = tmp_path / "native_exp_div.cpp"
    path.write_text(definitions + source + wrapper)
    image = simpler.KernelCompiler("a1").compile_single_launch(str(path), str(root))
    dtype = torch.float16 if fp16 else torch.float32
    with simpler.A1SingleLaunch(image, device_id=0) as runner:
        for seed in [113, 127]:
            generator = torch.Generator().manual_seed(seed)
            lhs = (torch.rand((1, count), generator=generator) * 16 - 8).to(dtype)
            rhs = (torch.rand((1, count), generator=generator) * 4 + 0.25).to(dtype)
            rhs[:, ::2] *= -1
            out = torch.full((1, count + 16), 7.0, dtype=dtype)
            expected = lhs / rhs if division else lhs.exp()
            runner(lhs, rhs, out)
            torch.testing.assert_close(
                out[:, :count], expected, rtol=0.003 if fp16 else 0.0001, atol=0.00002 if fp16 else 0.000002
            )
            assert torch.equal(out[:, count:], torch.full((1, 16), 7.0, dtype=dtype))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
