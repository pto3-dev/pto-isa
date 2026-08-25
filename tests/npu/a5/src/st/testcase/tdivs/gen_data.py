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


def gen_golden_data(param):
    dtype = param.data_type
    rows = param.row
    cols = param.col
    dst_tile_row = param.dst_tile_row
    dst_tile_col = param.dst_tile_col
    valid_row = param.valid_row
    valid_col = param.valid_col

    if dtype in (np.int64, np.uint64):
        input_arr = np.random.randint(1, 1000, size=[rows, cols]).astype(dtype)
        divider_value = param.scalar if param.scalar is not None else 7
        divider = np.array([[divider_value]], dtype=dtype)
    elif dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32):
        dtype_info = np.iinfo(dtype)
        input_arr = np.random.randint(dtype_info.min, dtype_info.max, size=[rows, cols]).astype(dtype)
        divider = np.random.randint(dtype_info.min, dtype_info.max, size=[1, 1]).astype(dtype)
    else:
        dtype_info = np.finfo(dtype)
        input_arr = np.random.uniform(low=dtype_info.min, high=dtype_info.max, size=[rows, cols]).astype(dtype)
        divider = np.random.uniform(low=dtype_info.min, high=dtype_info.max, size=[1, 1]).astype(dtype)
    
    output_arr = np.zeros((dst_tile_row, dst_tile_col), dtype=dtype)
    if param.inplace:
        output_arr[:rows, :cols] = input_arr[:rows, :cols]
    if dtype in (np.int64, np.uint64):
        output_arr[0:valid_row, 0:valid_col] = input_arr[0:valid_row, 0:valid_col] // divider[0, 0]
    else:
        output_arr[0:valid_row, 0:valid_col] = input_arr[0:valid_row, 0:valid_col] / divider[0, 0]

    input_arr.tofile('input.bin')
    divider.tofile('divider.bin')
    output_arr.tofile('golden.bin')


class TDivsParams:
    def __init__(self, name, data_type, dst_tile_row, dst_tile_col, row, col, scalar=None,
                 valid_row=None, valid_col=None, inplace=False):
        self.name = name
        self.data_type = data_type
        self.dst_tile_row = dst_tile_row
        self.dst_tile_col = dst_tile_col
        self.row = row
        self.col = col
        self.valid_row = row if valid_row is None else valid_row
        self.valid_col = col if valid_col is None else valid_col
        self.scalar = scalar
        self.inplace = inplace


if __name__ == "__main__":
    case_params_list = [
        TDivsParams("TDIVSTest.case1", np.float32, 32, 128, 32, 64),
        TDivsParams("TDIVSTest.case2", np.float16, 63, 128, 63, 64),
        TDivsParams("TDIVSTest.case3", np.int32, 31, 256, 31, 128),
        TDivsParams("TDIVSTest.case4", np.int16, 15, 192, 15, 64 * 3),
        TDivsParams("TDIVSTest.case5", np.float32, 7, 512, 7, 64 * 7),
        TDivsParams("TDIVSTest.case6", np.float32, 256, 32, 256, 16),
        TDivsParams("TDIVSTest.case7", np.float32, 1, 32, 1, 16),
        TDivsParams("TDIVSTest.caseHP1", np.float32, 2, 16, 2, 16),
        TDivsParams("TDIVSTest.caseHP2", np.float16, 2, 32, 2, 32),
        TDivsParams("TDIVSTest.case_int64_4x16", np.int64, 4, 16, 4, 16),
        TDivsParams("TDIVSTest.case_uint64_4x16", np.uint64, 4, 16, 4, 16),
        TDivsParams("TDIVSTest.case_int64_4x64", np.int64, 4, 64, 4, 64),
        TDivsParams("TDIVSTest.case_uint64_4x64", np.uint64, 4, 64, 4, 64),
        TDivsParams("TDIVSTest.case_int64_1x16364", np.int64, 1, 16364, 1, 16364),
        TDivsParams("TDIVSTest.case_uint64_1x16364", np.uint64, 1, 16364, 1, 16364),
        TDivsParams("TDIVSTest.case_int64_4091x4", np.int64, 4091, 4, 4091, 4),
        TDivsParams("TDIVSTest.case_uint64_4091x4", np.uint64, 4091, 4, 4091, 4),
        TDivsParams("TDIVSTest.case_int64_4x32_inplace", np.int64, 4, 32, 4, 32, scalar=17, inplace=True),
TDivsParams("TDIVSTest.case_uint64_4x32_inplace", np.uint64, 4, 32, 4, 32, scalar=17, inplace=True),
        TDivsParams("TDIVSTest.case_int64_1x1024_inplace", np.int64, 1, 1024, 1, 1024, scalar=17, inplace=True),
        TDivsParams("TDIVSTest.case_int64_4x64_40_inplace", np.int64, 4, 64, 4, 64, scalar=17,
                    valid_col=40, inplace=True),
        TDivsParams("TDIVSTest.case_int64_1x2048_2045_inplace", np.int64, 1, 2048, 1, 2048,
                    scalar=17, valid_col=2045, inplace=True),
    ]

    for _, case in enumerate(case_params_list):
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)
