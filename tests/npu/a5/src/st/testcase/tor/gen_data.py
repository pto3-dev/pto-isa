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


def gen_golden_data_tor(case_name, param):
    dtype = param.dtype

    tile_row, tile_col = param.tile_row, param.tile_col
    h_valid, w_valid = [param.valid_row, param.valid_col]

    if dtype in (np.int64, np.uint64):
        # Full-tile arrays so the written binary sizes match main.cpp's
        # kTRows_*kTCols_ read. Use large values so the high 32-bit half of
        # each 64-bit element is exercised.
        rng = np.random.default_rng(5)
        size = tile_row * tile_col
        if dtype == np.int64:
            input1 = rng.integers(-(2 ** 31), 2 ** 31 - 1, size=size, dtype=np.int64)
            input2 = rng.integers(-(2 ** 31), 2 ** 31 - 1, size=size, dtype=np.int64)
        else:
            input1 = rng.integers(0, 2 ** 32 - 1, size=size, dtype=np.uint64)
            input2 = rng.integers(0, 2 ** 32 - 1, size=size, dtype=np.uint64)

        golden = np.zeros(size).astype(dtype)
        golden[0:h_valid * w_valid] = (
            input1[0:h_valid * w_valid] | input2[0:h_valid * w_valid])
    else:
        input1 = np.random.randint(1, 16383, size=h_valid * w_valid).astype(dtype)
        input2 = np.random.randint(1, 16383, size=h_valid * w_valid).astype(dtype)
        golden = input1 | input2

    # Save the input and golden data to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")

    return input1, input2, golden


class TOrParams:
    def __init__(self, name, dtype, tile_row, tile_col, valid_row, valid_col):
        self.name = name
        self.dtype = dtype
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TOrParams("TORTest.case1", np.uint16, 64, 64, 64, 64),
        TOrParams("TORTest.case2", np.uint16, 64, 64, 63, 63),
        TOrParams("TORTest.case3", np.uint16, 1, 16384, 1, 16384),
        TOrParams("TORTest.case4", np.uint16, 2048, 16, 2048, 16),
        TOrParams("TORTest.case5", np.uint8, 32, 32, 32, 32),
        TOrParams("TORTest.case6", np.uint32, 8, 8, 8, 8),
        TOrParams("TORTest.case7", np.int8, 32, 32, 32, 32),
        TOrParams("TORTest.case8", np.int16, 16, 16, 16, 16),
        TOrParams("TORTest.case9", np.int32, 8, 8, 8, 8),
        TOrParams("TORTest.case_int64_4x16_4x15", np.int64, 4, 16, 4, 15),
        TOrParams("TORTest.case_uint64_4x16_4x15", np.uint64, 4, 16, 4, 15),
        TOrParams("TORTest.case_int64_32x32_32x32", np.int64, 32, 32, 32, 32),
        TOrParams("TORTest.case_int64_1x1024_1x1024", np.int64, 1, 1024, 1, 1024),
    ]

    for param in case_params_list:
        case_name = param.name
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tor(case_name, param)
        os.chdir(original_dir)