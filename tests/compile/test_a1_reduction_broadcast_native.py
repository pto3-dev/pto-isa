# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Native A1 axis-reduction checks; requires the matching Simpler A1 runtime."""

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
@pytest.mark.parametrize("kind", [0, 1, 2])
@pytest.mark.parametrize("cols", [16, 80, 400])
@pytest.mark.parametrize("sum_values", [False, True])
def test_a1_reduction_broadcast_native(tmp_path, fp16, kind, cols, sum_values):
    root = Path(__file__).resolve().parents[2]
    source = (root / "tests/compile/a1_reduction_broadcast.cpp").read_text()
    source = source.replace(
        'extern "C" __global__ AICORE void A1ReductionBroadcast', "AICORE void A1ReductionBroadcast"
    )
    definitions = f"#define PTO_A1_TEST_COLS {cols}\n#define PTO_A1_TEST_EXPAND {kind}\n"
    if fp16:
        definitions += "#define PTO_A1_TEST_FP16\n"
    if sum_values:
        definitions += "#define PTO_A1_TEST_SUM\n"
    wrapper = r"""
#include "tensor.h"
extern "C" __global__ AICORE void NativeReductionBroadcast(__gm__ int64_t* args) {
    auto* src = reinterpret_cast<__gm__ Tensor*>(args[0]);
    auto* out = reinterpret_cast<__gm__ Tensor*>(args[1]);
    A1ReductionBroadcast(reinterpret_cast<__gm__ Element*>(out->buffer.addr),
                         reinterpret_cast<__gm__ Element*>(src->buffer.addr));
}
"""
    path = tmp_path / "native_reduction_broadcast.cpp"
    path.write_text(definitions + source + wrapper)
    image = simpler.KernelCompiler("a1").compile_single_launch(str(path), str(root))
    dtype = torch.float16 if fp16 else torch.float32
    with simpler.A1SingleLaunch(image, device_id=0) as runner:
        for seed in [173, 179]:
            generator = torch.Generator().manual_seed(seed)
            src = (torch.randint(-16, 17, (16, cols), generator=generator).float() / 32).to(dtype)
            axis = 1 if kind == 0 else 0
            expected = (
                src.float().sum(dim=axis, keepdim=True).to(dtype) if sum_values else src.amax(dim=axis, keepdim=True)
            )
            out = torch.full((1, 16 * cols + 16), 7.0, dtype=dtype)
            runner(src, out)
            assert torch.equal(out[0, : 16 * cols].reshape(16, cols), expected.expand(16, cols))
            assert torch.equal(out[0, 16 * cols :], torch.full((16,), 7.0, dtype=dtype))
