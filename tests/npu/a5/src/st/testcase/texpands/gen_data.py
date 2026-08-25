#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os
import struct
import numpy as np

try:
    import ml_dtypes

    bfloat16 = ml_dtypes.bfloat16
    BFLOAT16_STORAGE_DTYPE = bfloat16
except ModuleNotFoundError:
    bfloat16 = object()
    BFLOAT16_STORAGE_DTYPE = np.float16
np.random.seed(19)

PAD_VALUE_NULL = "PAD_VALUE_NULL"
PAD_VALUE_MAX = "PAD_VALUE_MAX"
PAD_VALUE_MIN = "PAD_VALUE_MIN"


def gen_golden_data(param):
    dtype = param.dtype
    storage_dtype = BFLOAT16_STORAGE_DTYPE if dtype == bfloat16 else dtype

    height, width = [param.global_row, param.global_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]

    # Generate random input arrays
    scalar = 0
    if dtype == np.int16:
        scalar = np.random.randint(-30_000, 30_000, size=1).astype(storage_dtype)
    elif dtype == np.int32:
        scalar = np.random.randint(-2_000_000_000, 2_000_000_000, size=1).astype(storage_dtype)
    elif dtype == np.int64:
        scalar = np.array([-4_000_000_007], dtype=storage_dtype)
    elif dtype == np.uint64:
        scalar = np.array([10_000_000_019], dtype=storage_dtype)
    elif dtype == np.float16:
        scalar = np.random.uniform(-8, 8, size=1).astype(storage_dtype)
    elif dtype == bfloat16:
        scalar = np.random.uniform(-8, 8, size=1).astype(storage_dtype)
    elif dtype == np.float32:
        scalar = np.random.uniform(-8, 8, size=1).astype(storage_dtype)

    golden = np.full((height, width), 0).astype(storage_dtype)
    golden[:h_valid, :w_valid] = scalar[0]

    if getattr(param, "is_inplace", False):
        if dtype in (np.int64, np.uint64):
            input1 = np.random.randint(1, 100, size=height * width).astype(storage_dtype)
        else:
            input1 = np.random.uniform(-10, 10, size=height * width).astype(storage_dtype)
        input1_2d = input1.reshape(height, width)
        if h_valid < height:
            golden[h_valid:, :] = input1_2d[h_valid:, :]
        if w_valid < width:
            golden[:h_valid, w_valid:] = input1_2d[:h_valid, w_valid:]
        input1.tofile("input1.bin")

    # Save the golden data to binary files
    golden.tofile("golden.bin")
    scalar.tofile("scalar.bin")


class TestParams:
    def __init__(
        self,
        dtype,
        global_row,
        global_col,
        tile_row,
        tile_col,
        valid_row,
        valid_col,
        pad_value_type=PAD_VALUE_NULL,
        is_inplace=False,
        custom_name=None,
    ):
        self.dtype = dtype
        self.custom_name = custom_name
        self.global_row = global_row
        self.global_col = global_col
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col
        self.pad_value_type = pad_value_type
        self.is_inplace = is_inplace


def generate_case_name(param):
    if hasattr(param, 'custom_name') and param.custom_name:
        return param.custom_name
    dtype_str = {
        np.float32: "float",
        np.float16: "half",
        np.int8: "int8",
        np.int32: "int32",
        bfloat16: "bfloat16",
        np.int16: "int16",
        np.int64: "int64",
        np.uint64: "uint64",
    }[param.dtype]
    inplace_suffix = "_inplace" if getattr(param, "is_inplace", False) else ""
    return (
        f"TEXPANDSTest.case_{dtype_str}_"
        f"{param.global_row}x{param.global_col}_"
        f"{param.tile_row}x{param.tile_col}_"
        f"{param.valid_row}x{param.valid_col}_"
        f"{param.pad_value_type}{inplace_suffix}"
    )


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TestParams(np.float32, 64, 64, 64, 64, 64, 64),
        TestParams(np.int32, 64, 64, 64, 64, 64, 64),
        TestParams(np.int16, 64, 64, 64, 64, 64, 64),
        TestParams(np.float16, 64, 64, 64, 64, 64, 64),
        TestParams(bfloat16, 64, 64, 64, 64, 64, 64),
        TestParams(np.float32, 60, 60, 64, 64, 60, 60, PAD_VALUE_MAX),
        TestParams(np.int32, 60, 60, 64, 64, 60, 60, PAD_VALUE_MAX),
        TestParams(bfloat16, 1, 3600, 2, 4096, 1, 3600, PAD_VALUE_MAX),
        TestParams(np.float16, 1, 3600, 2, 4096, 1, 3600, PAD_VALUE_MAX),
        TestParams(np.int16, 16, 200, 20, 512, 16, 200, PAD_VALUE_MAX),
        TestParams(np.int16, 1, 200, 1, 512, 1, 200, PAD_VALUE_MAX),
        TestParams(np.int64, 5, 16, 5, 16, 5, 16),
        TestParams(np.uint64, 5, 16, 5, 16, 5, 16),
        TestParams(np.int64, 5, 64, 5, 64, 5, 64),
        TestParams(np.uint64, 5, 64, 5, 64, 5, 64),
        TestParams(np.int64, 1, 32732, 1, 32732, 1, 32732),
        TestParams(np.uint64, 1, 32732, 1, 32732, 1, 32732),
        TestParams(np.int64, 4, 32, 4, 32, 4, 32, is_inplace=True),
TestParams(np.uint64, 4, 32, 4, 32, 4, 32, is_inplace=True, custom_name="TEXPANDSTest.case_uint64_4x32_4x32_4x32_PAD_VALUE_NULL_inplace"),
        TestParams(np.int64, 1, 1024, 1, 1024, 1, 1024, is_inplace=True),
        TestParams(np.int64, 4, 64, 4, 64, 4, 40, is_inplace=True),
        TestParams(np.int64, 1, 2048, 1, 2048, 1, 2045, is_inplace=True),
    ]

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(param)
        os.chdir(original_dir)
