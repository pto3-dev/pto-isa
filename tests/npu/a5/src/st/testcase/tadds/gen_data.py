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
np.random.seed(23)


def gen_golden_data(param):
    data_type = param.data_type
    rows = param.row
    cols = param.col
    dst_tile_row = param.dst_tile_row
    dst_tile_col = param.dst_tile_col
    valid_row = param.valid_row
    valid_col = param.valid_col
    if np.issubdtype(data_type, np.integer):
        value_max = np.iinfo(data_type).max
        value_min = np.iinfo(data_type).min
    else:
        value_max = np.finfo(data_type).max
        value_min = np.finfo(data_type).min

    if data_type in (np.int64, np.uint64):
        low = param.int_low if param.int_low is not None else 1
        high = param.int_high if param.int_high is not None else 1000
        input_arr = np.random.randint(low, high, size=(rows, cols)).astype(data_type)
        divider_value = param.scalar if param.scalar is not None else 17
        divider = np.array([[divider_value]], dtype=data_type)
    else:
        input_arr = np.random.uniform(low=value_min, high=value_max, size=(rows, cols)).astype(data_type)
        divider = np.random.uniform(low=value_min, high=value_max, size=(1, 1)).astype(data_type)
    output_arr = np.zeros((dst_tile_row, dst_tile_col), dtype=data_type)
    if param.inplace:
        output_arr[:rows, :cols] = input_arr[:rows, :cols]
    for i in range(valid_row):
        for j in range(valid_col):
            output_arr[i, j] = input_arr[i, j] + divider[0, 0]

    input_arr.tofile('input.bin')
    if data_type in (np.int64, np.uint64):
        divider.tofile("divider.bin")
    else:
        with open("divider.bin", 'wb') as f:
            f.write(struct.pack('f', np.float32(divider[0, 0])))
    output_arr.tofile('golden.bin')


class taddsParams:
    def __init__(
        self, name, data_type, dst_tile_row, dst_tile_col, row, col,
        valid_row=None, valid_col=None, scalar=None, int_low=None, int_high=None, inplace=False
    ):
        self.name = name
        self.data_type = data_type
        self.dst_tile_row = dst_tile_row
        self.dst_tile_col = dst_tile_col
        self.row = row
        self.col = col
        self.valid_row = row if valid_row is None else valid_row
        self.valid_col = col if valid_col is None else valid_col
        self.scalar = scalar
        self.int_low = int_low
        self.int_high = int_high
        self.inplace = inplace

if __name__ == "__main__":
    case_params_list = [
        taddsParams("TADDSTest.case1", np.float32, 32, 128, 32, 64),
        taddsParams("TADDSTest.case2", np.float16, 63, 128, 63, 64),
        taddsParams("TADDSTest.case3", np.int32, 31, 256, 31, 128),
        taddsParams("TADDSTest.case4", np.int16, 15, 192, 15, 64 * 3),
        taddsParams("TADDSTest.case5", np.float32, 7, 512, 7, 64 * 7),
        taddsParams("TADDSTest.case6", np.float32, 256, 32, 256, 16),
        taddsParams("TADDSTest.case7", np.uint32, 256, 32, 256, 16),
        taddsParams("TADDSTest.case8", np.uint16, 256, 32, 256, 16),
        taddsParams("TADDSTest.case9", np.int8, 256, 64, 256, 32),
        taddsParams("TADDSTest.case10", np.uint8, 256, 64, 256, 32),
        taddsParams("TADDSTest.case11", np.uint8, 1, 64, 1, 32),
        taddsParams("TADDSTest.case_int64_4x16", np.int64, 4, 16, 4, 16),
        taddsParams("TADDSTest.case_uint64_4x16", np.uint64, 4, 16, 4, 16),
        taddsParams(
            "TADDSTest.case_int64_96x32768_32x1024_32x128",
            np.int64, 32, 1024, 32, 32768, 32, 1024, 1, -10, 11
        ),
        taddsParams("TADDSTest.case_int64_1x16364", np.int64, 1, 16364, 1, 16364),
        taddsParams("TADDSTest.case_uint64_1x16364", np.uint64, 1, 16364, 1, 16364),
        taddsParams("TADDSTest.case_int64_4x32_inplace", np.int64, 4, 32, 4, 32, inplace=True),
taddsParams("TADDSTest.case_uint64_4x32_inplace", np.uint64, 4, 32, 4, 32, inplace=True),
        taddsParams("TADDSTest.case_int64_1x1024_inplace", np.int64, 1, 1024, 1, 1024, inplace=True),
        taddsParams("TADDSTest.case_int64_4x64_40_inplace", np.int64, 4, 64, 4, 64, valid_col=40, inplace=True),
        taddsParams("TADDSTest.case_int64_1x2048_2045_inplace", np.int64, 1, 2048, 1, 2048, valid_col=2045, inplace=True),
    ]

    for _, case in enumerate(case_params_list):
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)
