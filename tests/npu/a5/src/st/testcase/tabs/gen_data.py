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
np.random.seed(19)

def gen_golden_data(case_name, param):
    dtype = param.dtype

    H, W = [param.tile_row, param.tile_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]

    golden = np.zeros([H, W]).astype(dtype)
    if dtype == np.int64:
        # Use large values so the high 32-bit half of each 64-bit element is
        # exercised. Keep magnitudes below INT64_MIN's magnitude so abs() does
        # not overflow.
        rng = np.random.default_rng(5)
        input1 = rng.integers(-(2 ** 31), 2 ** 31 - 1, size=(H, W)).astype(dtype)
    else:
        input1 = np.random.random(size=(H, W)).astype(dtype)
    golden[:h_valid, :w_valid] = np.abs(input1[:h_valid, :w_valid])
    input1.tofile("input1.bin")
    golden.tofile("golden.bin")


class tunaryParams:
    def __init__(self, dtype, global_row, global_col, tile_row, tile_col, valid_row, valid_col, in_place = False):
        self.dtype = dtype
        self.global_row = global_row
        self.global_col = global_col
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col
        self.in_place = in_place

def generate_case_name(param):
    dtype_str = {
        np.float32: 'float',
        np.float16: 'half',
        np.int8: 'int8',
        np.int32: 'int32',
        np.int16: 'int16',
        np.int64: 'int64'
    }[param.dtype]
    str_inplace = '_InPlace' if param.in_place else ''
    return f"TABSTest.case_{dtype_str}_{param.global_row}x{param.global_col}_{param.tile_row}x{param.tile_col}"\
        f"_{param.valid_row}x{param.valid_col}{str_inplace}"

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        tunaryParams(np.float32, 64, 64, 64, 64, 64, 64, True),
        tunaryParams(np.float32, 64, 64, 64, 64, 64, 64, False),
        tunaryParams(np.float16, 64, 64, 64, 64, 64, 64, True),
        tunaryParams(np.float16, 64, 64, 64, 64, 64, 64, False),
        tunaryParams(np.int8, 64, 64, 64, 64, 64, 64, True),
        tunaryParams(np.int8, 64, 64, 64, 64, 64, 64, False),
        tunaryParams(np.int16, 64, 64, 64, 64, 64, 64, True),
        tunaryParams(np.int16, 64, 64, 64, 64, 64, 64, False),
        tunaryParams(np.int32, 64, 64, 64, 64, 64, 64, True),
        tunaryParams(np.int32, 64, 64, 64, 64, 64, 64, False),
        tunaryParams(np.int64, 64, 64, 64, 64, 64, 64, False),
    ]

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, param)
        os.chdir(original_dir)
