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
@pytest.mark.parametrize("kind", [0, 1, 2, 3])
@pytest.mark.parametrize(("rows", "cols"), [(16, 16), (16, 80), (32, 256), (128, 16)])
def test_a1_axis_reductions_native(tmp_path, fp16, kind, rows, cols):
    _check_reduction(tmp_path, fp16, kind, rows, cols)


@pytest.mark.parametrize("fp16", [False, True])
@pytest.mark.parametrize("kind", [2, 3])
@pytest.mark.parametrize("wide", [False, True])
def test_a1_column_reduction_limits(tmp_path, fp16, kind, wide):
    rows, cols = (1, 32720 if fp16 else 16336) if wide else (3, 80)
    _check_reduction(tmp_path, fp16, kind, rows, cols)


def _check_reduction(tmp_path, fp16, kind, rows, cols):
    root = Path(__file__).resolve().parents[2]
    source = (root / "tests/compile/a1_axis_reductions.cpp").read_text()
    source = source.replace('extern "C" __global__ AICORE void A1AxisReduce', "AICORE void A1AxisReduce")
    definitions = (
        f"#define PTO_A1_TEST_ROWS {rows}\n#define PTO_A1_TEST_COLS {cols}\n#define PTO_A1_TEST_REDUCTION {kind}\n"
    )
    if fp16:
        definitions += "#define PTO_A1_TEST_FP16\n"
    wrapper = r"""
#include "tensor.h"
extern "C" __global__ AICORE void NativeAxisReduce(__gm__ int64_t* args) {
    auto* src = reinterpret_cast<__gm__ Tensor*>(args[0]);
    auto* out = reinterpret_cast<__gm__ Tensor*>(args[1]);
    A1AxisReduce(reinterpret_cast<__gm__ Element*>(out->buffer.addr),
                 reinterpret_cast<__gm__ Element*>(src->buffer.addr));
}
"""
    path = tmp_path / "native_axis_reduce.cpp"
    path.write_text(definitions + source + wrapper)
    image = simpler.KernelCompiler("a1").compile_single_launch(str(path), str(root))
    dtype = torch.float16 if fp16 else torch.float32
    axis = 1 if kind < 2 else 0
    count = rows if kind < 2 else cols
    with simpler.A1SingleLaunch(image, device_id=0) as runner:
        for seed in [149, 151]:
            generator = torch.Generator().manual_seed(seed)
            low, high = (-8, 0) if seed == 149 else (-8, 9)
            src = (torch.randint(low, high, (rows, cols), generator=generator).float() / 16).to(dtype)
            if seed == 151 and not fp16:
                src = torch.randn((rows, cols), generator=generator)
            out = torch.full((1, count + 16), 7.0, dtype=dtype)
            expected = src.amax(dim=axis) if kind % 2 else src.float().sum(dim=axis).to(dtype)
            runner(src, out)
            torch.testing.assert_close(
                out[0, :count], expected, rtol=0 if fp16 else 0.00001, atol=0 if fp16 else 0.00002
            )
            assert torch.equal(out[0, count:], torch.full((16,), 7.0, dtype=dtype))


@pytest.mark.parametrize("fp16", [False, True])
@pytest.mark.parametrize("rows", [16, 128])
@pytest.mark.parametrize("dn", [False, True])
def test_a1_packed_column_dma(tmp_path, fp16, rows, dn):
    root = Path(__file__).resolve().parents[2]
    source = (root / "tests/compile/a1_packed_column_dma.cpp").read_text()
    source = source.replace('extern "C" __global__ AICORE void A1PackedColumnDMA', "AICORE void A1PackedColumnDMA")
    definitions = f"#define PTO_A1_TEST_ROWS {rows}\n"
    if fp16:
        definitions += "#define PTO_A1_TEST_FP16\n"
    if dn:
        definitions += "#define PTO_A1_TEST_DN\n"
    wrapper = r"""
#include "tensor.h"
extern "C" __global__ AICORE void NativePackedColumnDMA(__gm__ int64_t* args) {
    auto* src = reinterpret_cast<__gm__ Tensor*>(args[0]);
    auto* out = reinterpret_cast<__gm__ Tensor*>(args[1]);
    A1PackedColumnDMA(reinterpret_cast<__gm__ Element*>(out->buffer.addr),
                      reinterpret_cast<__gm__ Element*>(src->buffer.addr));
}
"""
    path = tmp_path / "native_packed_dma.cpp"
    path.write_text(definitions + source + wrapper)
    image = simpler.KernelCompiler("a1").compile_single_launch(str(path), str(root))
    dtype = torch.float16 if fp16 else torch.float32
    src = (torch.arange(rows).float() - rows / 2).to(dtype).reshape(rows, 1)
    out = torch.full((rows + 32, 1), 7.0, dtype=dtype)
    with simpler.A1SingleLaunch(image, device_id=0) as runner:
        runner(src, out)
    assert torch.equal(out[16 : rows + 16], src)
    assert torch.equal(out[:16], torch.full((16, 1), 7.0, dtype=dtype))
    assert torch.equal(out[rows + 16 :], torch.full((16, 1), 7.0, dtype=dtype))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
