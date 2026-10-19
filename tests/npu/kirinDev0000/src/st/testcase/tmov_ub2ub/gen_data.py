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


def gen_golden_data(test_type, rows, cols):
    np.random.seed(42)
    input_arr = np.random.uniform(low=-10, high=10, size=(rows, cols)).astype(test_type)
    input_arr.tofile("input.bin")

    nz_block_row = 16
    c0_size = 16
    if test_type == np.int8:
        c0_size = 32
    elif test_type == np.int32:
        c0_size = 8

    output_arr = (
        input_arr.reshape(rows // nz_block_row, nz_block_row, cols // c0_size, c0_size)
        .transpose(2, 0, 1, 3)
        .astype(test_type)
    )
    output_arr.tofile("golden.bin")


if __name__ == "__main__":
    # Variant A/B/C share same golden data (ND→NZ transposed)
    case_list = [
        ("TMovUb2UbTest.caseA1_half_16x32", np.float16, 16, 32),
        ("TMovUb2UbTest.caseA2_half_64x256", np.float16, 64, 256),
        ("TMovUb2UbTest.caseA3_int32_48x72", np.int32, 48, 72),
        ("TMovUb2UbTest.caseA4_int8_32x512", np.int8, 32, 512),
        ("TMovUb2UbTest.caseA5_int8_64x96", np.int8, 64, 96),
        ("TMovUb2UbTest.caseB1_half_16x32_raw", np.float16, 16, 32),
        ("TMovUb2UbTest.caseB2_half_64x256_raw", np.float16, 64, 256),
        ("TMovUb2UbTest.caseB3_int32_48x72_raw", np.int32, 48, 72),
        ("TMovUb2UbTest.caseB4_int8_32x512_raw", np.int8, 32, 512),
        ("TMovUb2UbTest.caseB5_int8_64x96_raw", np.int8, 64, 96),
        ("TMovUb2UbTest.caseC1_half_16x32_pto", np.float16, 16, 32),
        ("TMovUb2UbTest.caseC2_half_64x256_pto", np.float16, 64, 256),
        ("TMovUb2UbTest.caseC3_int32_48x72_pto", np.int32, 48, 72),
        ("TMovUb2UbTest.caseC4_int8_32x512_pto", np.int8, 32, 512),
        ("TMovUb2UbTest.caseC5_int8_64x96_pto", np.int8, 64, 96),
    ]

    for case_name, test_type, rows, cols in case_list:
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(test_type, rows, cols)
        os.chdir(original_dir)
