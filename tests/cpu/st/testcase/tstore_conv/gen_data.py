#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

# Define the hardware-specific block size (e.g., 32 bytes)
BLOCK_SIZE_BYTES = 32


def get_c0_size(data_type):
    """Calculates C0 dimension size based on hardware alignment."""
    elem_size = np.dtype(data_type).itemsize
    return BLOCK_SIZE_BYTES // elem_size


def gen_golden_data(g_info):
    data_type = g_info.datatype
    c0 = get_c0_size(data_type)
    
    # 6D shape: [N, D, C1, H, W, C0]
    # Note: C0 is automatically fixed by the data_type
    shape = (g_info.g_whole_shape0, g_info.g_whole_shape1, g_info.g_whole_shape2, 
             g_info.g_whole_shape3, g_info.g_whole_shape4, c0)
    
    input_arr = np.random.randint(-5, 5, size=shape).astype(data_type)
    output_arr = np.zeros(shape=shape, dtype=data_type)
    
    # Slice using the fixed C0
    output_arr[0:g_info.g_shape0, 0:g_info.g_shape1, 0:g_info.g_shape2, 
               0:g_info.g_shape3, 0:g_info.g_shape4, 0:c0] = \
    input_arr[0:g_info.g_shape0, 0:g_info.g_shape1, 0:g_info.g_shape2, 
              0:g_info.g_shape3, 0:g_info.g_shape4, 0:c0]

    input_arr.tofile("./input.bin")
    output_arr.tofile("./golden.bin")


class GlobalTensorInfo:
    def __init__(self, datatype, s0, s1, s2, s3, s4, ws0, ws1, ws2, ws3, ws4):
        self.datatype = datatype
        self.g_shape0, self.g_shape1, self.g_shape2, self.g_shape3, self.g_shape4 = s0, s1, s2, s3, s4
        self.g_whole_shape0, self.g_whole_shape1 = ws0, ws1
        self.g_whole_shape2, self.g_whole_shape3, self.g_whole_shape4 = ws2, ws3, ws4

if __name__ == "__main__":
    # Test cases: [N, D, C1, H, W] (C0 is implicit)
    case_name_list = ["TStoreConvTest.NDC1HWC0_1", "TStoreConvTest.NDC1HWC0_2", "TStoreConvTest.NDC1HWC0_3"]
    case_params_list = [
        GlobalTensorInfo(np.float32, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2),
        GlobalTensorInfo(np.float32, 2, 3, 4, 1, 7, 2, 3, 4, 1, 7),
        GlobalTensorInfo(np.int32, 1, 2, 4, 16, 8, 1, 2, 4, 16, 8)
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name): 
            os.makedirs(case_name)
        os.chdir(case_name)
        gen_golden_data(case_params_list[i])
        os.chdir("..")