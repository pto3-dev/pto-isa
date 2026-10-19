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

"""
Generate golden data for the tpushpop_subblock_dispatch tests.

Every case computes out[M, N] = A[M, K] x B[K, N]; the cases differ only in how
the tile travels through the FIFO:

    V2C  : each vector core pushes a K row window of B into the L1 slot.
    C2V  : each vector core pops an M row window of A x B out of the GM slot.

A case that hands each vector core its peer's sub-block ID must end up with the
two row windows swapped, which is what swap_halves() encodes.
"""

import os

import numpy as np

np.random.seed(19)

VEC_CORES = 2


def swap_halves(mat):
    """Exchange the two row windows the vector cores own."""
    half = mat.shape[0] // VEC_CORES
    return np.concatenate([mat[half:], mat[:half]], axis=0)


def gen_v2c_golden(case_params):
    m, k, n, dtype, swapped = case_params
    x1_gm = np.random.uniform(-1, 1, [m, k]).astype(dtype)
    x2_gm = np.random.uniform(-1, 1, [k, n]).astype(dtype)

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")

    # A swapped push places the K row windows of B the other way round in L1.
    b_in_fifo = swap_halves(x2_gm) if swapped else x2_gm
    golden = np.matmul(x1_gm, b_in_fifo).astype(dtype)
    golden.tofile("./golden.bin")


def gen_c2v_gm_golden(case_params):
    m, k, n, dtype, swapped = case_params
    x1_gm = np.random.uniform(-1, 1, [m, k]).astype(dtype)
    x2_gm = np.random.uniform(-1, 1, [k, n]).astype(dtype)

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")

    # A swapped pop makes each vector core store its peer's M row window.
    matmul = np.matmul(x1_gm, x2_gm).astype(dtype)
    golden = swap_halves(matmul) if swapped else matmul
    golden.tofile("./golden.bin")


if __name__ == "__main__":
    # Case names must match the test suite and test case names in main.cpp so
    # that GetGoldenDir() resolves to the correct directory.
    case_name_list = [
        "TPushPopSubBlockDispatchTest.case1_v2c_nosplit_implicit_id",
        "TPushPopSubBlockDispatchTest.case2_v2c_nosplit_explicit_id",
        "TPushPopSubBlockDispatchTest.case3_v2c_split_implicit_id",
        "TPushPopSubBlockDispatchTest.case4_v2c_split_explicit_swapped_id",
        "TPushPopSubBlockDispatchTest.case5_c2v_gm_nosplit_implicit_id",
        "TPushPopSubBlockDispatchTest.case6_c2v_gm_split_implicit_id",
        "TPushPopSubBlockDispatchTest.case7_c2v_gm_split_explicit_swapped_id",
    ]

    # Parameters mirror the kernel instantiations: (M, K, N, dtype, swapped).
    case_params_list = [
        (16, 64, 32, np.float32, False),
        (16, 64, 32, np.float32, False),
        (16, 64, 32, np.float32, False),
        (16, 64, 32, np.float32, True),
        (32, 32, 64, np.float32, False),
        (32, 32, 64, np.float32, False),
        (32, 32, 64, np.float32, True),
    ]

    gen_func_list = [
        gen_v2c_golden,
        gen_v2c_golden,
        gen_v2c_golden,
        gen_v2c_golden,
        gen_c2v_gm_golden,
        gen_c2v_gm_golden,
        gen_c2v_gm_golden,
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_func_list[i](case_params_list[i])
        os.chdir(original_dir)
