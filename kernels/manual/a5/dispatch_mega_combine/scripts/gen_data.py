#!/usr/bin/env python3
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

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DATA_CACHE_VERSION = 5
MX_GROUP_SIZE = 32
WEIGHT_PERIOD = 16
WEIGHT_WRITE_ROWS = 256
GOLDEN_CHUNK_ROWS = 512
E4M3_MAX = np.float32(448.0)

# The period is zero-mean, BF16-exact, and contains several magnitudes. A power-of-two
# row amplitude changes the E8M0 code without changing the periodic structure used by
# the large-shape golden fast path.
WEIGHT_PATTERN = np.array(
    [0.0, 0.125, -0.125, 0.25, -0.25, 0.375, -0.375, 0.5, -0.5, 0.625, -0.625, 0.75, -0.75, 0.875, -0.875, 0.0],
    dtype=np.float32,
)


def fp32_to_bf16_bits(arr: np.ndarray) -> np.ndarray:
    """Convert FP32 to BF16 with round-to-nearest-even and return raw uint16."""
    src = np.asarray(arr, dtype=np.float32)
    bits = src.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def bf16_bits_to_fp32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def bf16_round(arr: np.ndarray) -> np.ndarray:
    return bf16_bits_to_fp32(fp32_to_bf16_bits(arr))


def encode_e4m3fn(values: np.ndarray) -> np.ndarray:
    """Vectorized RNE+SAT encoder matching pto::cpu_quant::EncodeE4M3Fn."""
    src = np.asarray(values, dtype=np.float32)
    clipped = np.clip(src, -E4M3_MAX, E4M3_MAX)
    magnitude = np.abs(clipped)
    sign = np.where(clipped < 0.0, np.uint8(0x80), np.uint8(0))
    code = np.zeros(src.shape, dtype=np.uint8)

    subnormal = (magnitude != 0.0) & (magnitude < np.float32(2.0**-6))
    if np.any(subnormal):
        mantissa = np.rint(magnitude[subnormal] * np.float32(512.0)).astype(np.int32)
        promoted = mantissa >= 8
        sub_code = mantissa.astype(np.uint8)
        sub_code[promoted] = np.uint8(0x08)
        code[subnormal] = sub_code

    normal = magnitude >= np.float32(2.0**-6)
    if np.any(normal):
        normal_values = magnitude[normal]
        exponent = np.floor(np.log2(normal_values)).astype(np.int32)
        power = np.ldexp(np.ones(exponent.shape, dtype=np.float32), exponent)
        mantissa = np.rint((normal_values / power - np.float32(1.0)) * np.float32(8.0)).astype(np.int32)
        carry = mantissa == 8
        exponent[carry] += 1
        mantissa[carry] = 0
        exponent_field = exponent + 7
        code[normal] = ((exponent_field << 3) | mantissa).astype(np.uint8)

    nonzero = code != 0
    code[nonzero] |= sign[nonzero]
    code[np.isnan(src)] = np.uint8(0x7F)
    return code


def decode_e4m3fn(raw: np.ndarray) -> np.ndarray:
    code = np.asarray(raw, dtype=np.uint8)
    sign = np.where((code & np.uint8(0x80)) != 0, np.float32(-1.0), np.float32(1.0))
    exponent = ((code >> np.uint8(3)) & np.uint8(0x0F)).astype(np.int32)
    mantissa = (code & np.uint8(0x07)).astype(np.float32)
    values = np.empty(code.shape, dtype=np.float32)
    subnormal = exponent == 0
    values[subnormal] = sign[subnormal] * np.ldexp(mantissa[subnormal], -9)
    normal = ~subnormal
    values[normal] = sign[normal] * np.ldexp(np.float32(1.0) + mantissa[normal] / np.float32(8.0), exponent[normal] - 7)
    values[(exponent == 0x0F) & (mantissa == 7.0)] = np.nan
    return values


def mx_quantize_bf16_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OCP MXFP8 quantization over independent 32-column groups."""
    src = bf16_round(values)
    if src.ndim != 2 or src.shape[1] % MX_GROUP_SIZE != 0:
        raise ValueError("MXFP8 source must be a 2-D matrix with columns divisible by 32")
    rows, cols = src.shape
    grouped = src.reshape(rows, cols // MX_GROUP_SIZE, MX_GROUP_SIZE)
    max_abs = np.max(np.abs(grouped), axis=2)
    max_bits = max_abs.view(np.uint32)
    exponent = ((max_bits >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int32)
    e8m0 = np.where(exponent <= 8, 0, exponent - 8).astype(np.uint8)
    scaling = np.ldexp(np.ones(e8m0.shape, dtype=np.float32), 127 - e8m0.astype(np.int32))
    scaled = grouped * scaling[:, :, None]
    fp8 = encode_e4m3fn(scaled).reshape(rows, cols)
    return fp8, e8m0


def mx_dequantize_rows(fp8: np.ndarray, e8m0: np.ndarray) -> np.ndarray:
    data = np.asarray(fp8, dtype=np.uint8)
    scales = np.asarray(e8m0, dtype=np.uint8)
    if data.ndim != 2 or scales.shape != (data.shape[0], data.shape[1] // MX_GROUP_SIZE):
        raise ValueError("MXFP8 data/scale shapes do not match")
    decoded = decode_e4m3fn(data).reshape(data.shape[0], scales.shape[1], MX_GROUP_SIZE)
    dequant_scale = np.ldexp(np.ones(scales.shape, dtype=np.float32), scales.astype(np.int32) - 127)
    return (decoded * dequant_scale[:, :, None]).reshape(data.shape).astype(np.float32)


def write(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(arr).tofile(path)


def make_x(rank: int, args: argparse.Namespace) -> np.ndarray:
    base = np.arange(args.m * args.k, dtype=np.float32).reshape(args.m, args.k)
    seed_phase = np.float32(args.seed % 97)
    x = np.float32(0.75) * np.sin((base + rank * np.float32(17.0) + seed_phase) / np.float32(23.0))
    x += np.float32(0.05) * np.cos((np.mod(base, np.float32(29.0)) + seed_phase) / np.float32(11.0))
    return bf16_round(x)


def make_probs(rank: int, args: argparse.Namespace) -> np.ndarray:
    topk_weights = np.arange(args.topk, 0, -1, dtype=np.float32)[None, :]
    token_offset = (np.arange(args.m, dtype=np.float32)[:, None] % max(args.topk, 1)) * np.float32(0.05)
    probs = topk_weights + token_offset + np.float32(rank * 0.01)
    probs /= probs.sum(axis=1, keepdims=True, dtype=np.float32)
    return probs.astype(np.float32)


def make_expert_idx(rank: int, args: argparse.Namespace) -> np.ndarray:
    total_experts = args.world_size * args.experts
    global_token = rank * args.m + np.arange(args.m, dtype=np.int32)
    base = global_token[:, None] * args.topk + np.arange(args.topk, dtype=np.int32)[None, :]
    return (base % total_experts).astype(np.int32)


def make_weight_rows(
    rank: int, expert: int, row_begin: int, row_count: int, reduction: int, kind: int, args: argparse.Namespace
) -> np.ndarray:
    rows = row_begin + np.arange(row_count, dtype=np.int32)
    seed_phase = args.seed % WEIGHT_PERIOD
    phase = (rows * 3 + expert * 5 + rank * 7 + kind * 11 + seed_phase) % WEIGHT_PERIOD
    amplitude_exp = -4 + ((rows + expert * 2 + rank + kind + seed_phase) % 4)
    amplitude = np.ldexp(np.ones(row_count, dtype=np.float32), amplitude_exp)
    reduction_idx = np.arange(reduction, dtype=np.int32)
    pattern_idx = (reduction_idx[None, :] + phase[:, None]) % WEIGHT_PERIOD
    return bf16_round(WEIGHT_PATTERN[pattern_idx] * amplitude[:, None])


def weight_dimensions(kind: int, args: argparse.Namespace) -> tuple[int, int]:
    if kind == 1:
        return args.n, args.k
    return args.k, args.n // 2


def make_quantized_weight_write_chunk(
    rank: int, expert: int, reduction: int, kind: int, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray]:
    if MX_GROUP_SIZE % WEIGHT_PERIOD != 0 or WEIGHT_WRITE_ROWS % WEIGHT_PERIOD != 0:
        raise ValueError("weight write and MX group sizes must be divisible by the weight period")
    if reduction % MX_GROUP_SIZE != 0:
        raise ValueError("weight reduction dimension must be divisible by the MX group size")

    # The deterministic weights repeat every 16 rows and columns; one 32-column MX group captures a full cycle.
    periodic_rows = make_weight_rows(rank, expert, 0, WEIGHT_PERIOD, MX_GROUP_SIZE, kind, args)
    periodic_fp8, periodic_e8m0 = mx_quantize_bf16_rows(periodic_rows)
    row_repeats = WEIGHT_WRITE_ROWS // WEIGHT_PERIOD
    column_repeats = reduction // MX_GROUP_SIZE
    fp8 = np.tile(periodic_fp8, (row_repeats, column_repeats))
    e8m0 = np.tile(periodic_e8m0, (row_repeats, column_repeats))
    return fp8, e8m0


def write_weight_pair(
    rank: int, kind: int, args: argparse.Namespace, out_dir: Path, expected_sizes: dict[str, int], reuse_static: bool
) -> None:
    weight_name = f"weight{kind}"
    scale_name = f"scale{kind}"
    weight_path = out_dir / f"rank{rank}_{weight_name}.bin"
    scale_path = out_dir / f"rank{rank}_{scale_name}.bin"
    if (
        reuse_static
        and weight_path.exists()
        and scale_path.exists()
        and weight_path.stat().st_size == expected_sizes[weight_name]
        and scale_path.stat().st_size == expected_sizes[scale_name]
    ):
        return

    output_dim, reduction = weight_dimensions(kind, args)
    with weight_path.open("wb") as weight_file, scale_path.open("wb") as scale_file:
        for expert in range(args.experts):
            fp8, e8m0 = make_quantized_weight_write_chunk(rank, expert, reduction, kind, args)
            for row_begin in range(0, output_dim, WEIGHT_WRITE_ROWS):
                row_count = min(WEIGHT_WRITE_ROWS, output_dim - row_begin)
                fp8[:row_count].tofile(weight_file)
                e8m0[:row_count].tofile(scale_file)
    if weight_path.stat().st_size != expected_sizes[weight_name]:
        raise RuntimeError(f"{weight_path.name} generated with an unexpected size")
    if scale_path.stat().st_size != expected_sizes[scale_name]:
        raise RuntimeError(f"{scale_path.name} generated with an unexpected size")


@dataclass
class GoldenInputs:
    xs: list[np.ndarray]
    expert_idx_list: list[np.ndarray]
    probs_list: list[np.ndarray]


@dataclass
class BatchGoldenContext:
    data: GoldenInputs
    args: argparse.Namespace
    route_groups: list[list[list[tuple[int, int, int]]]]
    x_dequant_by_rank: list[np.ndarray]
    outputs: list[np.ndarray]
    chunk_rows: int


def build_route_groups(expert_idx_list: list[np.ndarray], args: argparse.Namespace):
    route_groups: list[list[list[tuple[int, int, int]]]] = [
        [[] for _ in range(args.experts)] for _ in range(args.world_size)
    ]
    recv_rows = [0 for _ in range(args.world_size)]
    remote_routes = 0
    for src_rank, expert_idx in enumerate(expert_idx_list):
        for token in range(args.m):
            for topk_idx in range(args.topk):
                expert = int(expert_idx[token, topk_idx])
                if expert < 0 or expert >= args.world_size * args.experts:
                    raise ValueError(f"invalid expert id rank={src_rank} token={token} topk={topk_idx}: {expert}")
                dst_rank = expert // args.experts
                local_expert = expert % args.experts
                route_groups[dst_rank][local_expert].append((src_rank, token, topk_idx))
                recv_rows[dst_rank] += 1
                remote_routes += int(src_rank != dst_rank)
    for dst_rank, rows in enumerate(recv_rows):
        if rows > args.max_output_size:
            raise ValueError(
                f"dropless receive capacity exceeded for rank {dst_rank}: rows={rows} "
                f"max_output_size={args.max_output_size}"
            )
    routed_routes = args.world_size * args.m * args.topk
    workload = {
        "input_tokens_all_ranks": float(args.world_size * args.m),
        "routed_tokens_all_ranks": float(routed_routes),
        "remote_routed_tokens_all_ranks": float(remote_routes),
        "compute_flops_all_ranks": float(routed_routes) * 3.0 * args.k * args.n,
        "comm_bytes_all_ranks": float(remote_routes) * (3 * args.k + args.k // MX_GROUP_SIZE),
    }
    return route_groups, workload


def prequantize_inputs(xs: list[np.ndarray]) -> list[np.ndarray]:
    dequantized = []
    for x in xs:
        fp8, e8m0 = mx_quantize_bf16_rows(x)
        dequantized.append(mx_dequantize_rows(fp8, e8m0))
    return dequantized


_WEIGHT_TABLE_CACHE: dict[tuple[int, int, int, int, int], np.ndarray] = {}


def periodic_weight_table(rank: int, expert: int, output_dim: int, kind: int, args: argparse.Namespace) -> np.ndarray:
    key = (rank, expert, output_dim, kind, args.seed)
    cached = _WEIGHT_TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    signature = make_weight_rows(rank, expert, 0, output_dim, MX_GROUP_SIZE, kind, args)
    fp8, e8m0 = mx_quantize_bf16_rows(signature)
    dequant = mx_dequantize_rows(fp8, e8m0)
    table = np.ascontiguousarray(dequant[:, :WEIGHT_PERIOD].T, dtype=np.float32)
    _WEIGHT_TABLE_CACHE[key] = table
    return table


def matmul_periodic_weight(lhs: np.ndarray, table: np.ndarray) -> np.ndarray:
    if lhs.shape[1] % WEIGHT_PERIOD != 0:
        raise ValueError("periodic golden requires reduction dimension divisible by the weight period")
    residue_sums = lhs.reshape(lhs.shape[0], lhs.shape[1] // WEIGHT_PERIOD, WEIGHT_PERIOD).sum(axis=1, dtype=np.float32)
    return np.asarray(residue_sums @ table, dtype=np.float32)


def collect_batch_inputs(ctx: BatchGoldenContext, chunk: list[tuple[int, int, int]]):
    rows = len(chunk)
    src_ranks = np.fromiter((route[0] for route in chunk), dtype=np.int32, count=rows)
    token_indices = np.fromiter((route[1] for route in chunk), dtype=np.int32, count=rows)
    topk_indices = np.fromiter((route[2] for route in chunk), dtype=np.int32, count=rows)
    x = np.empty((rows, ctx.args.k), dtype=np.float32)
    probs = np.empty((rows,), dtype=np.float32)
    for src_rank in range(ctx.args.world_size):
        src_mask = src_ranks == src_rank
        if not np.any(src_mask):
            continue
        src_tokens = token_indices[src_mask]
        x[src_mask] = ctx.x_dequant_by_rank[src_rank][src_tokens]
        probs[src_mask] = ctx.data.probs_list[src_rank][src_tokens, topk_indices[src_mask]]
    return src_ranks, token_indices, x, probs


def swiglu_bf16_then_mx(gmm1_output: np.ndarray) -> np.ndarray:
    x, gate = np.split(gmm1_output.astype(np.float32, copy=False), 2, axis=1)
    negative_x = np.multiply(x, np.float32(-1.0), dtype=np.float32)
    with np.errstate(over="ignore"):
        exp_negative_x = np.exp(negative_x).astype(np.float32)
    denominator = np.add(exp_negative_x, np.float32(1.0), dtype=np.float32)
    silu = np.divide(x, denominator, dtype=np.float32)
    output = np.multiply(silu, gate, dtype=np.float32)
    output_bf16 = bf16_round(output)
    fp8, e8m0 = mx_quantize_bf16_rows(output_bf16)
    return mx_dequantize_rows(fp8, e8m0)


def run_batch_chunk(
    ctx: BatchGoldenContext, dst_rank: int, local_expert: int, chunk: list[tuple[int, int, int]]
) -> None:
    src_ranks, token_indices, x, probs = collect_batch_inputs(ctx, chunk)
    table1 = periodic_weight_table(dst_rank, local_expert, ctx.args.n, 1, ctx.args)
    gmm1_output = bf16_round(matmul_periodic_weight(x, table1))
    swiglu_dequant = swiglu_bf16_then_mx(gmm1_output)
    table2 = periodic_weight_table(dst_rank, local_expert, ctx.args.k, 2, ctx.args)
    gmm2_output = bf16_round(matmul_periodic_weight(swiglu_dequant, table2))
    weighted = np.multiply(gmm2_output, probs[:, None], dtype=np.float32)
    for src_rank in range(ctx.args.world_size):
        src_mask = src_ranks == src_rank
        if np.any(src_mask):
            np.add.at(ctx.outputs[src_rank], token_indices[src_mask], weighted[src_mask])


def compute_outputs_and_workload(data: GoldenInputs, args: argparse.Namespace):
    route_groups, workload = build_route_groups(data.expert_idx_list, args)
    x_dequant_by_rank = prequantize_inputs(data.xs)
    outputs = [np.zeros((args.m, args.k), dtype=np.float32) for _ in range(args.world_size)]
    ctx = BatchGoldenContext(data, args, route_groups, x_dequant_by_rank, outputs, GOLDEN_CHUNK_ROWS)

    for dst_rank in range(args.world_size):
        for local_expert in range(args.experts):
            routes = route_groups[dst_rank][local_expert]
            for start in range(0, len(routes), ctx.chunk_rows):
                chunk = routes[start : start + ctx.chunk_rows]
                if chunk:
                    run_batch_chunk(ctx, dst_rank, local_expert, chunk)
    return [bf16_round(output) for output in outputs], workload


def build_case_metadata(args: argparse.Namespace) -> dict[str, int | float | str]:
    return {
        "data_cache_version": DATA_CACHE_VERSION,
        "data_type": "bf16_e4m3_e8m0",
        "world_size": args.world_size,
        "m": args.m,
        "k": args.k,
        "n": args.n,
        "topk": args.topk,
        "expert_per_rank": args.experts,
        "max_output_size": args.max_output_size,
        "seed": args.seed,
        "compare_atol": args.atol,
        "compare_rtol": args.rtol,
    }


def expected_rank_file_sizes(args: argparse.Namespace) -> dict[str, int]:
    hidden = args.n // 2
    return {
        "x": args.m * args.k * np.dtype(np.uint16).itemsize,
        "weight1": args.experts * args.n * args.k,
        "weight2": args.experts * args.k * hidden,
        "expert_idx": args.m * args.topk * np.dtype(np.int32).itemsize,
        "scale1": args.experts * args.n * (args.k // MX_GROUP_SIZE),
        "scale2": args.experts * args.k * (hidden // MX_GROUP_SIZE),
        "probs": args.m * args.topk * np.dtype(np.float32).itemsize,
        "expected_out": args.m * args.k * np.dtype(np.uint16).itemsize,
    }


def reusable_data_mismatch(out_dir: Path, args: argparse.Namespace) -> str | None:
    case_path = out_dir / "case.json"
    if not case_path.exists():
        return "case.json missing"
    try:
        case_json = json.loads(case_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"case.json invalid: {exc}"
    for key, expected in build_case_metadata(args).items():
        if case_json.get(key) != expected:
            return f"case.json mismatch for {key}: cached={case_json.get(key)!r} requested={expected!r}"
    sizes = expected_rank_file_sizes(args)
    for rank in range(args.world_size):
        for name, expected_size in sizes.items():
            path = out_dir / f"rank{rank}_{name}.bin"
            if not path.exists() or path.stat().st_size != expected_size:
                return f"{path.name} missing or has an unexpected size"
    return None


def can_reuse_static_rank_files(out_dir: Path, args: argparse.Namespace) -> bool:
    if not args.reuse_data:
        return False
    case_path = out_dir / "case.json"
    if not case_path.exists():
        return False
    try:
        cached = json.loads(case_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    desired = build_case_metadata(args)
    for key in ("data_cache_version", "data_type", "world_size", "k", "n", "expert_per_rank", "seed"):
        if cached.get(key) != desired[key]:
            return False
    sizes = expected_rank_file_sizes(args)
    for rank in range(args.world_size):
        for name in ("weight1", "weight2", "scale1", "scale2"):
            path = out_dir / f"rank{rank}_{name}.bin"
            if not path.exists() or path.stat().st_size != sizes[name]:
                return False
    return True


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    for name in ("world_size", "m", "k", "n", "topk", "experts", "max_output_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.n % 2 != 0 or args.k % 128 != 0 or (args.n // 2) % 128 != 0:
        parser.error("MXFP8 path requires even N, K % 128 == 0, and (N/2) % 128 == 0")
    if args.experts not in (4, 8, 16, 32):
        parser.error("--experts must be one of 4, 8, 16, or 32")


def self_check_numeric_helpers() -> None:
    values = np.array([0.0, 1.0, -1.0, 448.0, -448.0], dtype=np.float32)
    expected = np.array([0x00, 0x38, 0xB8, 0x7E, 0xFE], dtype=np.uint8)
    actual = encode_e4m3fn(values)
    if not np.array_equal(actual, expected):
        raise RuntimeError(f"E4M3 encoder self-check failed: {actual.tolist()}")
    if not np.array_equal(fp32_to_bf16_bits(bf16_bits_to_fp32(fp32_to_bf16_bits(values))), fp32_to_bf16_bits(values)):
        raise RuntimeError("BF16 conversion self-check failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--max-output-size", type=int, default=32)
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    args = parser.parse_args()
    validate_args(args, parser)
    self_check_numeric_helpers()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_data:
        mismatch = reusable_data_mismatch(out_dir, args)
        if mismatch is None:
            print(f"[GEN_DATA] reuse existing data: {out_dir}", flush=True)
            return
        print(f"[GEN_DATA] regenerate data: {mismatch}", flush=True)
    reuse_static = can_reuse_static_rank_files(out_dir, args)
    if reuse_static:
        print("[GEN_DATA] reuse static rank E4M3/E8M0 weight files", flush=True)

    xs = [make_x(rank, args) for rank in range(args.world_size)]
    probs = [make_probs(rank, args) for rank in range(args.world_size)]
    expert_idx = [make_expert_idx(rank, args) for rank in range(args.world_size)]
    data = GoldenInputs(xs, expert_idx, probs)
    expected_out, workload = compute_outputs_and_workload(data, args)
    sizes = expected_rank_file_sizes(args)

    for rank in range(args.world_size):
        write(out_dir / f"rank{rank}_x.bin", fp32_to_bf16_bits(xs[rank]))
        write(out_dir / f"rank{rank}_expert_idx.bin", expert_idx[rank].astype(np.int32))
        write(out_dir / f"rank{rank}_probs.bin", probs[rank].astype(np.float32))
        write(out_dir / f"rank{rank}_expected_out.bin", fp32_to_bf16_bits(expected_out[rank]))
        write_weight_pair(rank, 1, args, out_dir, sizes, reuse_static)
        write_weight_pair(rank, 2, args, out_dir, sizes, reuse_static)

    case_json = {**build_case_metadata(args), **workload}
    (out_dir / "case.json").write_text(json.dumps(case_json, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
