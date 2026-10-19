# PTO MegaMoE Dispatch + Combine Fusion Example

[中文文档](README_zh.md)

## Overview

This example implements an end-to-end MegaMoE fused operator with PTO Manual kernels. It combines token quantization,
route-mask exchange, grouped FFN compute, combine, and unpermute in one mixed-core kernel. The current implementation
uses MXFP8 data, direct AIC-to-AIV CV pipes, dynamic GMM task mailboxes, and token-ready RankStreaming Unpermute to
overlap communication, metadata preparation, AIC compute, and AIV processing.

The main device pipeline is:

```text
FrontReorder -> Dispatch -> GMM1 -> SwiGLU -> GMM2 -> Combine -> Unpermute
```

## Supported AI Processors

- Ascend950PR (A5, arch35 / DAV_3510 family)
- Toolchain-compatible alias `Ascend910_9599`

This project always builds the mixed-core kernel with `dav-c310` and `PTO_NPU_ARCH_A5`. `Ascend910B` is an A3 target and is not a valid `--soc` value for this directory.

## Directory Layout

```text
kernels/manual/a5/dispatch_mega_combine/
├── CMakeLists.txt                  # Build configuration for host executable and device kernel shared library
├── run.sh                          # One-click data generation, build, and mpirun entry
├── main.cpp                        # Host entry: case loading, ACL/HCCL/MPI setup, launch, verification, and timing
├── kernel_launch.cpp               # Device kernel launch wrapper
├── runtime_context.*               # Single-rank runtime, HCCL window, device/context management
├── tiling_builder.*                # Host tiling and workspace planning
├── data_utils.*                    # Case file IO and validation helpers
├── comm_mpi.h                      # MPI dynamic loading wrapper
├── scripts/
│   └── gen_data.py                 # Synthetic input, weight, and golden generation
├── op_kernel/
│   ├── dispatch_mega_combine.h     # MegaMoE device pipeline entry
│   ├── front_reorder.h             # Token-order MXFP8 quantization and route-mask publication
│   ├── front_metadata_sort.h       # Expert-major route sort used by deferred metadata
│   ├── deferred_route_metadata.h   # Task descriptors, preSum, and inverse-route metadata
│   ├── dispatch.h                  # Mask compaction, remote token pull, and GMM1 input build
│   ├── gmm_common.h                # Shared GMM tile scheduling and helpers
│   ├── gmm_task_producer.h         # GMM1/GMM2 producer-consumer ticket scheduler
│   ├── gmm_expert_progress.h       # Combine completion to rank-progress coordinator
│   ├── gmm1.h                      # First grouped matmul and GMM1 mailbox consumer
│   ├── swiglu.h                    # SwiGLU activation and dynamic quantization
│   ├── gmm2.h                      # Direct-wave0 plus mailbox GMM2 consumer
│   ├── combine.h                   # CV-tile consumption and remote route-output writeback
│   ├── unpermute.h                 # Token-ready TopK reduction in original token order
│   └── utils/                      # PTO vector, sync, HCCL window, and GMM pipeline helpers
```

## Operator Description

### Functionality

The operator implements the multi-rank MoE FFN main path:

```text
x[rank, M, K] + expertId[rank, M, topK] + probs[rank, M, topK]
  -> route-mask pull into destination expert-major rows
  -> grouped GMM1
  -> SwiGLU activation + dynamic quantization
  -> grouped GMM2
  -> combine back to source ranks
  -> TopK weighted reduction
  -> out[rank, M, K]
```

Conceptually:

```text
for each rank, token:
  out[token] = sum_{topK route} probs[token, route] * FFN_expert(x[token])
```

`FFN_expert` consists of MXFP8 GMM1, SwiGLU, and MXFP8 GMM2. Cross-rank data exchange uses the HCCL RDMA window and PTO communication/synchronization helpers.

### Specification

| Item | Value |
| --- | --- |
| OpType | `MegaMoE Dispatch + FFN + Combine` |
| Input | `x`: `[M, K]`, `bfloat16`; `expertId`: `[M, topK]`, `int32`; `probs`: `[M, topK]`, `float32`; `weight1/weight2`: per-local-expert E4M3 data; `scale1/scale2`: E8M0 scales |
| Output | `out`: `[M, K]`, `bfloat16` |
| Kernel name | `dispatch_mega_combine_kernel` |
| Host executable | `dispatch_mega_combine` |
| Default script case | `worldSize=2, M=2048, K=7168, N=4096, topK=8, expertPerRank=16, maxOutputSize=81940` |

## Optimization Notes

- **Mask Pull Front**: source tokens are quantized once to E4M3/E8M0 records; PTO `TCMPS` builds per-expert route
  masks, and Dispatch pulls only matching records.
- **Deferred metadata**: reserved AIV0 workers build GMM task descriptors, source-rank `preSum`, expert-major route
  order, and the inverse `expandedRowIdx` while Dispatch and the first GMM stage are progressing.
- **Direct CV handoff**: GMM1 sends paired BF16 x/gate tiles directly to SwiGLU through a one-slot CV FIFO; GMM2
  sends BF16 result tiles directly to Combine through a one-slot CV FIFO. Neither GMM output is materialized in GM.
- **Adaptive vector pipelines**: Front uses ping-pong quant buffers, Dispatch chooses a 2-6-slot remote-token ring,
  and Unpermute chooses a 2-6-slot BF16/FP32 input ring within the 216 KiB main UB budget.
- **Dual-mode GMM1 scheduling**: `M <= 512` uses an all-AIC direct wave0 followed by a mailbox P/C suffix. Larger
  shapes use fixed-wave GMM1; Group2 switches to GMM2 after the configured full-AIC wave prefix, without a global barrier.
- **Dynamic GMM2 scheduling**: the initial GMM2 group executes wave0 directly; all remaining GMM2 tiles use producer-
  assigned mailbox tickets, allowing Group1 AICs to join as soon as their GMM1 handoff is complete.
- **Token-ready Unpermute**: Combine publishes ordered expert-prefix progress after its remote stores are complete.
  An Unpermute worker consumes a token as soon as all of that token's TopK routes are ready.

The host queries the AICore count with `aclrtGetDeviceInfo(..., ACL_DEV_ATTR_AICORE_CORE_NUM, ...)` and selects a
validated mixed-core mapping. Every physical block contains one AIC and two AIV subblocks. The default mappings are:

| AIC | AIV | Default Dispatch AIV0 / steady GMM1 AIC | Initial GMM2 AIC | Two-phase Unpermute AIV0 |
| ---: | ---: | --- | --- | ---: |
| 28 | 56 | `0..19` (20) | `20..27` (8) | 16 |
| 32 | 64 | `0..20` (21) | `21..31` (11) | 16 |
| 36 | 72 | `0..23` (24) | `24..35` (12) | 16 |

Fixed-wave GMM1 starts with a configured full-AIC prefix. For the tuned 36-AIC canonical shapes, the steady split is
overridden to `22:14` for `(worldSize=8, M=1024/2048, expertPerRank=16)` and `(worldSize=16, M=1024,
expertPerRank=16)`; the eight-rank `M=2048` case keeps four full-AIC waves before that split, while other listed cases use the default full-AIC prefix and split. Every AIV1 is the paired SwiGLU/Combine consumer for its
physical AIC. All AIVs participate in final-phase Unpermute, while `M >= 512` additionally enables 16 AIV0 phase-1
workers from a frozen expert-progress snapshot.

`run.sh --aicore-num 0|28|32|36` selects the effective launch count (`0` uses the runtime-reported count). A requested
count must not exceed the physical runtime count. Dispatch width grows to at least `worldSize`; the rank limit is
`min(32, aicNum - 2)`, leaving AIV0 workers for deferred metadata and the mailbox producer.

## Tiling Parameters

| Parameter | Default / Description |
| --- | --- |
| `M` | Set by `run.sh --m` or `case.json` |
| `K` | Hidden size for GMM input; must satisfy packed-row and GMM tile alignment |
| `N` | FFN intermediate size; GMM1 output and `N/2` after SwiGLU |
| `topK` | Number of routed experts per token |
| `expertPerRank` | Runtime local-expert count: `4`, `8`, `16`, or `32`; all values share one kernel |
| `worldSize` | MPI/HCCL rank count, limited by `min(32, aicNum - 2)` |
| `maxOutputSize` | Per-rank routed-row workspace limit |
| `aicNum` | Effective AICore launch count; validated values are 28, 32, and 36 |
| `aivNum` | Derived as `2 * aicNum`: 56, 64, or 72 |
| `GMM baseM/baseN` | Main output tile shape is `256 x 256` |
| `Front Mask Pull` | The only Front implementation; host rejects clipping, invalid experts, inactive tokens, and insufficient receive capacity |
| `Fixed-role AIV UB` | Dispatch, SwiGLU, Combine, and Unpermute use a 216 KiB main region; the final 40 KiB is reserved for synchronization snapshots |
| `Dispatch / Unpermute tiles` | Dispatch uses an adaptive 2-6-slot packed-token ring; Unpermute keeps a full K row when possible, supports up to 8192 columns per tile, and uses an adaptive 2-6-slot input ring |
| `GMM / AIV CV tiles` | GMM1 sends paired `256 x 256` BF16 half-tiles through one CV slot; GMM2 sends `256 x 256` BF16 tiles through one CV slot |

## Supported Cases

The tuned eight-rank performance cases keep all major parameters fixed except `M`. The same shapes can be used as
two-rank smoke/correctness cases; generic tiling also supports other legal rank counts and runtime local-expert counts.

```text
worldSize=8
K=7168
N=4096
topK=8
expertPerRank=16
aicNum=runtime (28, 32, or 36), or `run.sh --aicore-num 28|32|36`
aivNum=2 * aicNum
```

| M | maxOutputSize | Command |
| --- | --- | --- |
| 16 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 16 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 32 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 32 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 64 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 64 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 128 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 128 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 512 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 512 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 1024 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 1024 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 2048 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 2048 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |

## Overall Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ Front Mask Pull (AIV)                                                       │
│   BF16 -> MXFP8 source records + per-expert route masks/counts               │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │ front metadata ready
┌──────────────────────────────▼───────────────────────────────────────────────┐
│ Dispatch + deferred metadata + dynamic GMM task producer                    │
│                                                                              │
│ AIV0: mask compact/pull + route sort/inverse + mailbox descriptors           │
│ AIC : GMM1 direct/fixed/mailbox -> released AICs join GMM2 mailbox           │
│ AIV1:       CV SwiGLU                    CV Combine -> remote compact rows    │
│                                                                              │
│ Hybrid GMM1/GMM2 use direct wave0 + mailbox suffix; fixed GMM1 uses waves    │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │ ordered expert-prefix progress
┌──────────────────────────────▼───────────────────────────────────────────────┐
│ RankStreaming Unpermute (all AIVs)                                          │
│   expandedRowIdx + compact output + probs -> TopK reduce -> out[M, K]        │
└──────────────────────────────────────────────────────────────────────────────┘
```

## FrontReorder Stage

Front keeps one MXFP8 record per source token and publishes one bitmask slot per global expert/source rank:

```text
x[M, K] + expertId[M, topK]
  -> sourceTokenRecords[M, aligned E4M3 data + aligned E8M0 scales]
  -> routeMaskSlots[localExpert, srcRank, mask + laneCapacity * 32B partial counts]
  -> cumsumMM[srcRank, localExpert] / expertTokenNums[localExpert]
```

Tokens are split across all AIVs without splitting K. Global experts receive balanced AIV lane intervals. Each lane
writes disjoint 32B mask blocks (256 route slots per block) plus an aligned partial-count record. Front quantization
uses two PTO UB buffers. After every source publishes its front-ready epoch, one coordinator reduces lane counts and
builds source-major `cumsumMM` and `expertTokenNums`.

In parallel, reserved AIV0 workers sort route slots into expert-major order, build `expandedRowIdx`, publish
source-rank `preSumBeforeRank`, and generate GMM1/GMM2 mailbox descriptors. GMM2 entry is released only after deferred
metadata, local Dispatch, and all source-rank preSum rows are visible.

## Dispatch Stage

Dispatch runs on the destination rank. Each source rank gets `dispatchGroupSize / worldSize` AIV0 lanes; every lane
scans the source mask but consumes only its match-ordinal shard:

```text
srcRank.sourceTokenRecords[routeSlot / topK]
  -> workspace.gmA[dstRow, 0:K]                  (E4M3)
  -> workspace.gmAScale[dstRow, 0:K/32]          (E8M0)
  -> workspace.routeMeta[dstRow] = {srcRank, routeSlot, 0...}
```

PTO vector mask compaction produces matching route indices. Each source rank receives
`floor(dispatchGroupSize / worldSize)` active AIV0 lanes, with any trailing Dispatch roles left idle. An adaptive
2-6-slot ring overlaps remote packed-record loads with E4M3/E8M0 and metadata stores. Dispatch publishes cache-line-
isolated ready counts per expert and 256-row M tile, so GMM1 waits only for the input tile it is about to consume.

## GMM1 / SwiGLU / GMM2 Stages

### GMM1

GMM1 runs on AIC and performs the first MXFP8 grouped matmul:

```text
gmA[E4M3] / gmAScale[E8M0] x weight1[E4M3] / scale1[E8M0]
  -> int32 accumulator
  -> fixpipe MX scaling and BF16 cast
  -> paired x/gate CV tiles
```

Each local expert is split into `256 x 256` half-output tiles; x and gate are computed as a pair and sent directly to
the paired AIV1 through a one-slot CV FIFO. Linear tile ids are swizzled for B-side L1 reuse. For `M <= 512`, all AICs
execute direct wave0 and the steady GMM1 group consumes the suffix from the mailbox. For larger M, fixed-wave GMM1
uses all AICs for the configured full-AIC wave prefix and the configured GMM1 group afterwards.

### SwiGLU

SwiGLU runs on the paired AIV1 and consumes the BF16 x/gate CV tile:

```text
x/gate BF16 CV tile
  -> silu(x) * gate
  -> MXFP8 dynamic quantization
  -> gmSwigluA[E4M3] + gmSwigluScale[E8M0]
```

The CV control stream carries the exact GMM task sequence. SwiGLU drains each E4M3/E8M0 store before incrementing the
GMM2 dependency counter for that expert, and GMM2 starts the expert after the required SwiGLU tile count is complete.

### GMM2

GMM2 runs on AIC and performs the second MXFP8 grouped matmul:

```text
gmSwigluA[E4M3] / gmSwigluScale[E8M0] x weight2[E4M3] / scale2[E8M0]
  -> int32 accumulator
  -> fixpipe MX scaling and BF16 cast
  -> GMM2/Combine CV tile
```

The configured GMM2 group executes wave0 directly. All suffix tiles are producer-assigned mailbox tasks. Group1 AICs
enter this mailbox immediately after their mode-specific GMM1 handoff, without a global GMM1/GMM2 barrier. Each AIC
streams BF16 `256 x 256` results to its paired Combine AIV1 through a one-slot CV FIFO.

## Combine / Unpermute Stages

Combine runs on the paired AIV1 and writes each BF16 GMM2 CV tile directly to source-rank compact output rows:

```text
GMM2 BF16 CV tile
  -> intersect expert rows with each source-rank cumsum range
  -> srcRank.combineOutputByRouteSlot[compactRow, 0:K]
```

Combine preloads `cumsumMM` and `preSumBeforeRank`, then uses strided UB-to-GM stores for each source-rank
intersection. Before publishing an expert completion count, it drains prior remote MTE3 stores; the progress
coordinator therefore exposes an ordered expert prefix to each source rank.

Unpermute restores the source-rank token order:

```text
expandedRowIdx[token * topK + topk]
  -> combineOutputByRouteSlot[compactRow] * probs[token, topk]
  -> TopK weighted accumulation
  -> out[M, K]
```

For `M < 512`, all AIVs run the single live phase. For `M >= 512`, 16 AIV0 workers first consume tokens admitted by a
frozen progress snapshot, then all AIVs consume the remaining tokens from live expert-prefix progress. RankStreaming
is the only Unpermute implementation; unsupported shapes fail host validation instead of falling back to a barrier
kernel.

## Memory Layout and HCCL Window

The HCCL remote window carries cross-rank visible data:

| Buffer | Location | Purpose |
| --- | --- | --- |
| `sourceTokenRecords` | HCCL window | One packed E4M3/E8M0 record per source token; Dispatch pulls by `routeSlot / topK` |
| `routeMaskSlots` | HCCL window | Per-local-expert/source-rank mask plus aligned count record |
| `combineOutputByRouteSlot` | HCCL window | Expert-major compact BF16 rows written by Combine and indexed through `expandedRowIdx` |
| `preSumBeforeRank` | HCCL window | Source-rank compact-row base for each global expert |
| signal tail | HCCL window | Front/preSum/data-ready epochs and rank-progress slots |
| `gmA` / `gmAScale` | workspace GM | E4M3/E8M0 GMM1 input generated by Dispatch |
| `gmSwigluA` / `gmSwigluScale` | workspace GM | E4M3/E8M0 GMM2 input generated by SwiGLU |
| `routeMeta` | workspace GM | 32B Dispatch row record containing `{srcRank, routeSlot}` |
| `cumsumMM` | workspace GM | source-major cumulative rows for Dispatch destination offsets |
| `sortedRouteSlot` / `expandedRowIdx` | workspace GM | Expert-major route order and its route-slot-to-compact-row inverse |
| GMM queues/mailbox | workspace GM | Runtime task descriptors, tickets, dependency counters, and completion counters |

`run.sh` estimates the HCCL window from `M`, `topK`, `K`, expert topology, and mask-lane capacity, then raises
`HCCL_BUFFSIZE` when needed.

## Build and Run

Configure the Ascend CANN environment:

```bash
source ~/zy/set_env.sh
```

Build the A5 mixed-core kernel and host without generating case data:

```bash
cd ${git_clone_path}/kernels/manual/a5/dispatch_mega_combine
bash run.sh --build-only
```

Run on contiguous available A5 devices. The environment must provide MPICH; OpenMPI is not supported by the MPI
compatibility wrapper.

```bash
bash run.sh --soc Ascend910_9599 --world-size 2 --first-device 2 --m 2048 --k 7168 --n 4096 \
  --topk 8 --experts 16 --max-output-size 81940 --reuse-data
```

This maps ranks 0 and 1 to physical devices 2 and 3. `Ascend910_9599` is the A5 toolchain alias here; it must not be
replaced with the A3 target `Ascend910B`.

For another `M` value, keep the initial two-rank setup, for example:

```bash
bash run.sh --soc Ascend910_9599 --world-size 2 --first-device 2 --m 512 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data
```

### Environment Variables

| Environment Variable | Purpose | Default Behavior |
| --- | --- | --- |
| `ASCEND_HOME_PATH` | CANN installation path | Must be set before running |
| `CMAKE_COMPILER` | Compiler used by CMake | `bisheng` |
| `FIRST_DEVICE` | First physical device in the contiguous rank mapping | `0`; overridden by `--first-device` |
| `MPI_LIB_PATH` | Optional absolute path to the MPICH `libmpi.so` | Otherwise resolved from `LD_LIBRARY_PATH` |
| `MPI_RUNNER` | MPICH launch command | `mpirun` from the sourced environment |
| `HCCL_BUFFSIZE` | HCCL RDMA window size | Raised automatically by `run.sh` when needed |
| `DISPATCH_MEGA_COMBINE_AICORE_NUM` | Effective AIC count | `0`, which uses the runtime-reported count |
| `DISPATCH_MEGA_COMBINE_REUSE_DATA` | Reuse compatible generated case data | Disabled; a nonzero value is equivalent to `--reuse-data` |
| `DISPATCH_MEGA_COMBINE_WARMUP_ITERS` | Warmup launches before timing | `3` |
| `DISPATCH_MEGA_COMBINE_MEASURE_ITERS` | Timed launches used for the kernel summary | `5` |

### Kernel Performance

After the timed launches, rank 0 prints one `[KERNEL_PERF]` summary for the complete kernel. Each AIC/AIV records only
its overall start/end system-counter values; the summary reports the maximum rank duration per iteration together with
average, minimum, maximum, standard deviation, token throughput, equivalent compute TFLOPS, and equivalent
communication bandwidth.

## Changing Case Parameters

When changing a case, keep its MXFP8 dimensions, rank topology, and receive capacity within host-validated limits.

```bash
bash run.sh --world-size 8 --first-device 0 --m 512 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data
```

Common constraints:

- `K` must be a multiple of 128 and satisfy the Dispatch packed-row capacity.
- `N` must be even and `N / 2` must be a multiple of 128.
- `topK` must be in `1..32`.
- `expertPerRank` must be `4`, `8`, `16`, or `32`; all values use the same runtime kernel with capacity for 32 local
  experts.
- `worldSize` must not exceed `min(32, aicNum - 2)` for the selected 28/32/36-AIC topology.
- `maxOutputSize` must cover the per-rank routed-row workspace limit.
- Each RankStreaming phase is limited to 256 tokens per worker; unsupported shapes are rejected by host tiling.
- Synthetic `expert_idx` uses global token round-robin so small-M cases still cover global experts.

## FAQ

| Problem | Cause and Fix |
| --- | --- |
| `ASCEND_HOME_PATH must be set` | Source the CANN environment and export `ASCEND_HOME_PATH` before running `run.sh` |
| HCCL window too small | The manually set `HCCL_BUFFSIZE` is below the case requirement; unset it or increase it |
| MPI launch fails | Source the project environment and verify `mpirun --version` reports MPICH/HYDRA; OpenMPI is unsupported |
| Golden generation is slow | Reuse the generated files with `--reuse-data` after the first run; the chunk size is fixed internally |
| Shape or rank topology is rejected | Check the 28/32/36-AIC selection, rank limit, MXFP8 alignment, UB capacity, and RankStreaming token-per-worker limit |
| Result diff is abnormal | Check whether old generated data was reused; do not reuse stale `out/` after changing expert distribution or key case parameters |

## Build System

- **Compiler**: `bisheng`
- **Device kernel flags**: `-xcce --cce-aicore-arch=dav-c310` plus the A5 address-transform and explicit DCCI policy
- **Host executable**: `-xc++ -std=c++17`
- **Targets**: `dispatch_mega_combine_kernel`, `dispatch_mega_combine`
- **Linked libraries**: `stdc++`, `ascendcl`, `hcomm`, `runtime`, `tiling_api`, `platform`, `nnopbase`, `pthread`, and others
- **PTO include**: repository root `include/` is added to the include path for PTO tile/communication helpers

## Changelog

| Date | Change |
| --- | --- |
| 2026-06-26 | Added `dispatch_mega_combine` README covering the MegaMoE operator, stage flow, build/run, and FAQ |
| 2026-07-27 | Ported the optimized A3 schedule to the A5 backend and completed compile-only validation |
| 2026-08-14 | Removed development instrumentation and prepared the A5 target for production use |
| 2026-08-20 | Updated the production path with runtime expert counts, deferred metadata, hybrid GMM1, mailbox GMM2, CV handoff, RankStreaming-only Unpermute, and ordered Combine progress publication |
