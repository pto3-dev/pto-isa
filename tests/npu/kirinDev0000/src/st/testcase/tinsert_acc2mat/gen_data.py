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


def get_c0(dtype_size):
    if dtype_size >= 4:
        return 8
    if dtype_size == 1:
        return 32
    return 16


def nd_to_nz(arr, rows, cols, dtype_size):
    c0 = get_c0(dtype_size)
    return arr.reshape(rows // 16, 16, cols // c0, c0).transpose(2, 0, 1, 3)


def gen_acc2mat(m, k, n):
    x1 = np.random.randint(-2, 3, size=(m, k)).astype(np.float16)
    x2 = np.random.randint(-2, 3, size=(k, n)).astype(np.float16)
    x1.tofile("x1_gm.bin")
    x2.tofile("x2_gm.bin")
    golden = np.matmul(x1, x2).astype(np.float16)
    nd_to_nz(golden, m, n, 2).tofile("golden.bin")


def gen_cbuf2cbuf(m, n):
    arr = np.random.randint(1, 100, size=(m, n)).astype(np.float16)
    arr_nz = nd_to_nz(arr, m, n, 2)
    arr_nz.tofile("x1_gm.bin")
    arr_nz.tofile("golden.bin")


if __name__ == "__main__":
    cases = [
        ("TInsertAcc2MatTest.case_acc2mat_1", gen_acc2mat, 16, 16, 16),
        ("TInsertAcc2MatTest.case_acc2mat_2", gen_acc2mat, 32, 32, 32),
        ("TInsertAcc2MatTest.case_mat2mat_1", gen_acc2mat, 16, 16, 16),
        ("TInsertAcc2MatTest.case_mat2mat_2", gen_acc2mat, 32, 32, 32),
        ("TInsertAcc2MatTest.case_cbuf2cbuf_fixp_16", gen_cbuf2cbuf, 16, 16),
        ("TInsertAcc2MatTest.case_cbuf2cbuf_nofixp_16", gen_cbuf2cbuf, 16, 16),
        ("TInsertAcc2MatTest.case_mat2mat_load_16", gen_cbuf2cbuf, 16, 16),
        ("TInsertAcc2MatTest.case_mat2mat_ctrl_16", gen_cbuf2cbuf, 16, 16),
    ]

    for name, gen_fn, *args in cases:
        os.makedirs(name, exist_ok=True)
        orig = os.getcwd()
        os.chdir(name)
        gen_fn(*args)
        os.chdir(orig)
