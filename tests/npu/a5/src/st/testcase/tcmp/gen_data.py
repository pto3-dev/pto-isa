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
import numpy as np

try:
    import ml_dtypes

    bfloat16 = ml_dtypes.bfloat16
    BFLOAT16_STORAGE_DTYPE = bfloat16
except ModuleNotFoundError:
    bfloat16 = object()
    BFLOAT16_STORAGE_DTYPE = np.float16
np.random.seed(19)


def gen_tcmp_input(param, storage_dtype):
    row, col = param.row, param.col
    if np.issubdtype(storage_dtype, np.integer):
        input1 = np.random.randint(-10, 10, size=[row, col]).astype(storage_dtype)
        input2 = np.random.randint(-10, 10, size=[row, col]).astype(storage_dtype)
        if param.dtype == np.int64:
            input1.flat[:5] = [np.iinfo(param.dtype).min, -1, 0, 1, np.iinfo(param.dtype).max]
            input2.flat[:5] = [np.iinfo(param.dtype).max, -1, 0, -1, np.iinfo(param.dtype).min]
        elif param.dtype == np.uint64:
            input1.flat[:5] = [0, 1, 1 << 63, np.iinfo(param.dtype).max, np.iinfo(param.dtype).max]
            input2.flat[:5] = [np.iinfo(param.dtype).max, 1, 1 << 63, 0, np.iinfo(param.dtype).max]
    else:
        input1 = np.random.uniform(-10, 10, size=[row, col]).astype(storage_dtype)
        input2 = np.random.uniform(-10, 10, size=[row, col]).astype(storage_dtype)
    if getattr(param, "is_nan", False):
        input1[:] = np.nan
        input2[:] = np.nan
    return input1, input2


def get_tcmp_bool_result(param, input1, input2, storage_dtype):
    if param.mode == "EQ":
        if np.issubdtype(storage_dtype, np.integer):
            return input1 == input2
        elif param.dtype == bfloat16:
            input1_f32 = input1.astype(np.float32)
            input2_f32 = input2.astype(np.float32)
            return np.isclose(input1_f32, input2_f32, rtol=0, atol=1e-9)
        return np.isclose(input1, input2, rtol=0, atol=1e-9)
    elif param.mode == "NE":
        if np.issubdtype(storage_dtype, np.integer):
            return input1 != input2
        elif param.dtype == bfloat16:
            input1_f32 = input1.astype(np.float32)
            input2_f32 = input2.astype(np.float32)
            return ~np.isclose(input1_f32, input2_f32, rtol=0, atol=1e-9)
        return ~np.isclose(input1, input2, rtol=0, atol=1e-9)
    elif param.mode == "LT":
        return input1 < input2
    elif param.mode == "GT":
        return input1 > input2
    elif param.mode == "GE":
        return input1 >= input2
    elif param.mode == "LE":
        return input1 <= input2
    raise ValueError(f"Unsupported compare mode: {param.mode}")


def gen_golden_data_tcmp(param):
    storage_dtype = BFLOAT16_STORAGE_DTYPE if param.dtype == bfloat16 else param.dtype
    row, col = param.row, param.col
    valid_row, valid_col = param.valid_row, param.valid_col
    input1, input2 = gen_tcmp_input(param, storage_dtype)
    bool_result = get_tcmp_bool_result(param, input1, input2, storage_dtype)
    output = np.zeros((row, col), dtype=np.uint8)
    output[:valid_row, :valid_col] = bool_result[:valid_row, :valid_col]
    golden = np.packbits(output, axis=1, bitorder="little")
    if getattr(param, "is_inplace", False) and valid_col < col:
        dst_valid_cols = (valid_col + 7) // 8
        dst_cols = (col + 7) // 8
        input1_bytes = input1.reshape(-1).view(np.uint8)
        golden_flat = golden.flatten()
        for r in range(row):
            for b in range(dst_valid_cols, dst_cols):
                idx = r * dst_cols + b
                if idx < len(input1_bytes):
                    golden_flat[idx] = input1_bytes[idx]
        golden = golden_flat.reshape(golden.shape)
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")


class TcmpParams:
    def __init__(self, dtype, row, col, valid_row, valid_col, cmp_mode, is_nan=False, is_inplace=False):
        self.dtype = dtype
        self.row = row
        self.col = col
        self.valid_row = valid_row
        self.valid_col = valid_col
        self.mode = cmp_mode
        self.is_nan = is_nan
        self.is_inplace = is_inplace


def generate_case_name(param):
    dtype_str = {
        np.float32: "float",
        np.float16: "half",
        np.int32: "int32",
        np.int16: "int16",
        np.int64: "int64",
        np.uint64: "uint64",
        bfloat16: "bfloat16",
    }[param.dtype]
    nan_suffix = "_nan" if getattr(param, "is_nan", False) else ""
    inplace_suffix = "_inplace" if getattr(param, "is_inplace", False) else ""
    mode_suffix = f"_{param.mode}" if param.dtype in (np.int64, np.uint64) else ""
    return (
        f"TCMPTest.case_{dtype_str}_{param.row}x{param.col}_"
        f"{param.valid_row}x{param.valid_col}{mode_suffix}{inplace_suffix}{nan_suffix}"
    )


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [  # Comment out test cases that do not handle size correctly
        TcmpParams(np.float16, 32, 32, 32, 32, "EQ"),
        TcmpParams(np.float32, 8, 64, 8, 64, "GT"),
        TcmpParams(np.int32, 4, 64, 4, 64, "NE"),
        TcmpParams(np.int32, 128, 128, 64, 64, "LT"),
        TcmpParams(np.int32, 64, 64, 32, 32, "EQ"),
        TcmpParams(np.int32, 16, 32, 16, 32, "EQ"),
        TcmpParams(np.float32, 128, 128, 64, 64, "LE"),
        TcmpParams(np.int32, 77, 80, 32, 32, "EQ"),
        TcmpParams(np.int32, 32, 32, 32, 32, "EQ"),
        TcmpParams(np.int16, 32, 32, 16, 32, "EQ"),
        TcmpParams(np.int16, 77, 80, 32, 32, "LE"),
        TcmpParams(bfloat16, 32, 32, 16, 32, "EQ"),
        TcmpParams(bfloat16, 77, 80, 32, 32, "LE"),
        TcmpParams(np.float32, 32, 32, 32, 32, "NE", is_nan=True),
        *[
            TcmpParams(dtype, 4, 16, 4, 15, mode)
            for dtype in (np.int64, np.uint64)
            for mode in ("EQ", "NE", "LT", "GT", "GE", "LE")
        ],
        *[
            TcmpParams(dtype, 4, 64, 4, 64, mode)
            for dtype in (np.int64, np.uint64)
            for mode in ("EQ", "NE", "LT", "GT", "GE", "LE")
        ],
        TcmpParams(np.int64, 4, 32, 4, 32, "EQ", is_inplace=True),
TcmpParams(np.uint64, 4, 32, 4, 32, "EQ", is_inplace=True),
        TcmpParams(np.int64, 1, 1024, 1, 1024, "EQ", is_inplace=True),
        TcmpParams(np.int64, 4, 64, 4, 40, "EQ", is_inplace=True),
        TcmpParams(np.int64, 1, 2048, 1, 2045, "EQ", is_inplace=True),
    ]

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tcmp(param)
        os.chdir(original_dir)
