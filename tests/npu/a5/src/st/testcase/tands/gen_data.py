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
import ctypes
import numpy as np
np.random.seed(19)


def gen_golden_data_tand(case_name, param):
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
            input2 = rng.integers(-(2 ** 31), 2 ** 31 - 1, size=(1, 1), dtype=np.int64)
        else:
            input1 = rng.integers(0, 2 ** 32 - 1, size=size, dtype=np.uint64)
            input2 = rng.integers(0, 2 ** 32 - 1, size=(1, 1), dtype=np.uint64)

        golden = np.zeros(size).astype(dtype)
        golden[0:h_valid * w_valid] = (
            input1[0:h_valid * w_valid] & input2[0, 0])
    else:
        input1 = np.random.randint(1, 16383, size=(h_valid, w_valid)).astype(dtype)
        input2 = np.random.randint(1, 16383, size=(1, 1)).astype(dtype)
        golden = np.zeros((h_valid, w_valid), dtype=dtype)
        for i in range(h_valid):
            for j in range(w_valid):
                golden[i, j] = input1[i, j] & input2[0, 0]

    with open("input2.bin", 'wb') as f:
        dtype_map = {
            np.int8: 'b',
            np.uint8: 'B',
            np.int16: 'h',
            np.uint16: 'H',
            np.int32: 'i',
            np.uint32: 'I',
            np.int64: 'q',
            np.uint64: 'Q'
        }
        format_char = dtype_map.get(dtype)
        if format_char is not None:
            f.write(struct.pack(format_char, input2[0, 0]))

    input1.tofile("input1.bin")
    golden.tofile("golden.bin")
    return input1, input2, golden


class TAndSParams:
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
        TAndSParams("TANDSTest.case1", np.uint16, 64, 64, 64, 64),
        TAndSParams("TANDSTest.case2", np.uint16, 64, 64, 63, 63),
        TAndSParams("TANDSTest.case3", np.uint16, 1, 16384, 1, 16384),
        TAndSParams("TANDSTest.case4", np.uint16, 2048, 16, 2048, 16),
        TAndSParams("TANDSTest.case5", np.uint8, 32, 32, 32, 32),
        TAndSParams("TANDSTest.case6", np.uint32, 8, 8, 8, 8),
        TAndSParams("TANDSTest.case7", np.int8, 32, 32, 32, 32),
        TAndSParams("TANDSTest.case8", np.int16, 16, 16, 16, 16),
        TAndSParams("TANDSTest.case9", np.int32, 8, 8, 8, 8),
        TAndSParams("TANDSTest.case10", np.int32, 1, 8, 1, 8),
        TAndSParams("TANDSTest.case_int64_4x16_4x15", np.int64, 4, 16, 4, 15),
        TAndSParams("TANDSTest.case_uint64_4x16_4x15", np.uint64, 4, 16, 4, 15),
        TAndSParams("TANDSTest.case_int64_1x1024_1x1024", np.int64, 1, 1024, 1, 1024),
    ]

    for param in case_params_list:
        case_name = param.name
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tand(case_name, param)
        os.chdir(original_dir)