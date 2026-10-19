# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Native A1 scalar-fill checks; requires the matching Simpler A1 runtime."""

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


@pytest.mark.parametrize("fp16", [False, True])
@pytest.mark.parametrize("size", [16, 80, "large"])
@pytest.mark.parametrize("value", [-0.0, -2.5, 3.25])
def test_a1_scalar_broadcast_native(tmp_path, fp16, size, value):
    count = (32720 if fp16 else 16336) if size == "large" else size
    root = Path(__file__).resolve().parents[2]
    source = (root / "tests/compile/a1_scalar_broadcast.cpp").read_text()
    source = source.replace('extern "C" __global__ AICORE void A1ScalarFill', "AICORE void A1ScalarFill")
    definitions = f"#define PTO_A1_TEST_ELEMENTS {count}\n#define PTO_A1_TEST_VALUE {value}f\n"
    if fp16:
        definitions += "#define PTO_A1_TEST_FP16\n"
    wrapper = r"""
#include "tensor.h"
extern "C" __global__ AICORE void NativeScalarFill(__gm__ int64_t* args) {
    auto* out = reinterpret_cast<__gm__ Tensor*>(args[0]);
    A1ScalarFill(reinterpret_cast<__gm__ Element*>(out->buffer.addr));
}
"""
    path = tmp_path / "native_scalar_fill.cpp"
    path.write_text(definitions + source + wrapper)
    image = simpler.KernelCompiler("a1").compile_single_launch(str(path), str(root))
    dtype = torch.float16 if fp16 else torch.float32
    integer = torch.int16 if fp16 else torch.int32
    with simpler.A1SingleLaunch(image, device_id=0) as runner:
        for prefill in [7.0, 9.0]:
            out = torch.full((1, count + 16), prefill, dtype=dtype)
            runner(out)
            expected = torch.full((1, count), value, dtype=dtype)
            assert torch.equal(out[:, :count].view(integer), expected.view(integer))
            assert torch.equal(out[:, count:], torch.full((1, 16), prefill, dtype=dtype))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
