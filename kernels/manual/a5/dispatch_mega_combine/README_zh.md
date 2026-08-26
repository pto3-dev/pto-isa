# PTO MegaMoE Dispatch + Combine 融合算子示例

## 概览

本示例使用 PTO Manual kernel 实现 MegaMoE 的端到端融合流程，把 token 量化、route mask 交换、
grouped FFN、Combine 和 Unpermute 合并到一个 mixed-core kernel。当前实现采用 MXFP8 数据、AIC 到 AIV
的 CV 直传、动态 GMM mailbox 和 token-ready RankStreaming Unpermute，重叠通信、元数据准备、AIC
计算和 AIV 处理。

## 支持的 AI 处理器

- Ascend950PR（A5，arch35 / DAV_3510 系列）
- 工具链兼容别名 `Ascend910_9599`

本项目固定使用 `dav-c310` 和 `PTO_NPU_ARCH_A5` 构建 mixed-core kernel。`Ascend910B` 属于 A3，不能作为本目录的 `--soc` 参数。

## 目录结构

```text
kernels/manual/a5/dispatch_mega_combine/
├── CMakeLists.txt                  # 构建配置，生成 host 可执行文件和 device kernel so
├── run.sh                          # 数据生成、构建、mpirun 执行的一键脚本
├── main.cpp                        # Host 入口：加载 case、初始化 ACL/HCCL/MPI、启动、校验和计时
├── kernel_launch.cpp               # Device kernel launch 包装
├── runtime_context.*               # 单 rank runtime、HCCL window、device/context 管理
├── tiling_builder.*                # Host 侧 tiling 构造和 workspace 规划
├── data_utils.*                    # case 数据文件读写和校验辅助
├── comm_mpi.h                      # MPI 动态加载包装
├── scripts/
│   └── gen_data.py                 # synthetic 输入、权重、golden 生成
├── op_kernel/
│   ├── dispatch_mega_combine.h     # MegaMoe device 主流程入口
│   ├── front_reorder.h             # token-order MXFP8 量化、route mask 和 partial count 发布
│   ├── front_metadata_sort.h       # deferred metadata 使用的 expert-major route 排序
│   ├── deferred_route_metadata.h   # task descriptor、preSum 和逆向 route metadata
│   ├── dispatch.h                  # mask 压缩、远端 token 拉取和 GMM1 输入构造
│   ├── gmm_common.h                # GMM1/GMM2 共享 tile 调度和 helper
│   ├── gmm_task_producer.h         # GMM1/GMM2 P/C ticket 调度
│   ├── gmm_expert_progress.h       # Combine completion 到 rank progress 的协调
│   ├── gmm1.h                      # 第一层 grouped matmul 和 GMM1 mailbox 消费
│   ├── swiglu.h                    # SwiGLU + dynamic quant
│   ├── gmm2.h                      # wave0 直通与 mailbox GMM2 消费
│   ├── combine.h                   # CV tile 消费和远端 route output 写回
│   ├── unpermute.h                 # token-ready topK reduce 和原 token 顺序还原
│   └── utils/                      # PTO vector、sync、HCCL window、GMM pipeline helper
```

## 算子说明

### 计算功能

本示例实现多 rank MoE FFN 主体流程：

```text
x[rank, M, K] + expertId[rank, M, topK] + probs[rank, M, topK]
  -> route mask 拉取为目标侧 expert-major token rows
  -> grouped GMM1
  -> SwiGLU activation + dynamic quant
  -> grouped GMM2
  -> combine 回源 rank
  -> topK weighted reduce
  -> out[rank, M, K]
```

逻辑公式可以理解为：

```text
for each rank, token:
  out[token] = sum_{topK route} probs[token, route] * FFN_expert(x[token])
```

其中 `FFN_expert` 由 MXFP8 GMM1、SwiGLU、MXFP8 GMM2 组成；前后通信通过 HCCL RDMA window 和 PTO 通信/同步 helper 完成。

### 规格

| 项目 | 值 |
| --- | --- |
| OpType | `MegaMoE Dispatch + FFN + Combine` |
| 输入 | `x`: `[M, K]`, `bfloat16`; `expertId`: `[M, topK]`, `int32`; `probs`: `[M, topK]`, `float32`; `weight1/weight2`: 每个本地 expert 的 E4M3 数据；`scale1/scale2`: E8M0 scale |
| 输出 | `out`: `[M, K]`, `bfloat16` |
| Kernel 名称 | `dispatch_mega_combine_kernel` |
| Host 可执行文件 | `dispatch_mega_combine` |
| 默认脚本 case | `worldSize=2, M=2048, K=7168, N=4096, topK=8, expertPerRank=16, maxOutputSize=81940` |

## 优化说明

- **Mask Pull Front**：源 token 只量化一次，形成 E4M3/E8M0 记录；PTO `TCMPS` 生成逐 expert route
  mask，Dispatch 只拉取命中记录。
- **Deferred metadata**：预留 AIV0 在 Dispatch 和首段 GMM 推进期间并行构造 GMM task descriptor、
  source-rank `preSum`、expert-major route 顺序及逆映射 `expandedRowIdx`。
- **CV 直传**：GMM1 通过单槽 CV FIFO 把 BF16 x/gate tile 直接交给 SwiGLU；GMM2 通过单槽 CV FIFO
  把 BF16 结果直接交给 Combine，两层 GMM 输出均不落 GM 中间结果。
- **自适应向量流水**：Front 使用 quant ping-pong；Dispatch 和 Unpermute 在 216 KiB 主 UB 内分别选择
  2～6 槽的远端 token ring 和 BF16/FP32 输入 ring。
- **GMM1 双模式调度**：`M <= 512` 使用全 AIC wave0 直通和 mailbox P/C 后缀；更大场景使用
  fixed-wave GMM1，Group2 在配置的全 AIC wave 前缀结束后无全局 barrier 地切换到 GMM2。
- **GMM2 动态调度**：初始 GMM2 组直通执行 wave0，其余 GMM2 tile 均由 producer 分配 mailbox ticket；
  Group1 AIC 完成 GMM1 handoff 后即可加入消费。
- **Token-ready Unpermute**：Combine 在远端写完成后发布有序 expert-prefix progress；一个 token 的全部
  TopK route ready 后，Unpermute worker 即可消费。

Host 通过 `aclrtGetDeviceInfo(..., ACL_DEV_ATTR_AICORE_CORE_NUM, ...)` 查询 AICore 数并选择已验证的
mixed-core 分配。每个 physical block 包含一个 AIC 和两个 AIV subblock。默认分配如下：

| AIC | AIV | 默认 Dispatch AIV0 / 稳态 GMM1 AIC | 初始 GMM2 AIC | 两阶段 Unpermute AIV0 |
| ---: | ---: | --- | --- | ---: |
| 28 | 56 | `0..19`（20） | `20..27`（8） | 16 |
| 32 | 64 | `0..20`（21） | `21..31`（11） | 16 |
| 36 | 72 | `0..23`（24） | `24..35`（12） | 16 |

fixed-wave GMM1 先执行配置的全 AIC wave 前缀。36 AIC canonical 场景中，
`(worldSize=8, M=1024/2048, expertPerRank=16)` 和
`(worldSize=16, M=1024, expertPerRank=16)` 的稳态分组覆盖为 `22:14`；其中八卡 `M=2048` 在切分前保持 4 个全 AIC wave，其它已列场景使用默认全 AIC 前缀和比例。
每个 AIV1 都是配对 AIC 的 SwiGLU/Combine consumer；所有 AIV 参与 Unpermute 最终阶段，`M >= 512`
时额外启用 16 个 AIV0 phase1 worker 消费冻结的 progress 快照。

`run.sh --aicore-num 0|28|32|36` 选择有效 launch 核数（`0` 表示采用 runtime 查询值）。请求值不能超过
实际物理核数。Dispatch 宽度至少扩展到 `worldSize`；rank 上限为 `min(32, aicNum - 2)`，以保留
deferred metadata worker 和 mailbox producer。

## Tiling 参数

| 参数 | 默认值 / 说明 |
| --- | --- |
| `M` | 由 `run.sh --m` 或 case.json 指定 |
| `K` | GMM 输入 hidden size；要求满足 packed row 和 GMM tile 对齐 |
| `N` | FFN 中间维度；GMM1 输出，SwiGLU 后进入 `N/2` |
| `topK` | 每 token 路由专家数 |
| `expertPerRank` | runtime 本地 expert 数：`4`、`8`、`16` 或 `32`，共用同一 kernel |
| `worldSize` | MPI/HCCL rank 数，上限为 `min(32, aicNum - 2)` |
| `maxOutputSize` | 每 rank routed row workspace 上限 |
| `aicNum` | 有效 AICore launch 核数；已验证 28、32 和 36 |
| `aivNum` | 按 `2 * aicNum` 推导：56、64 或 72 |
| `GMM baseM/baseN` | 主要 tile 口径为 `256 x 256` output tile |
| `Front Mask Pull` | 唯一 Front 实现；host 拒绝 clipping、非法 expert、inactive token 和接收容量不足 |
| `固定角色 AIV UB` | Dispatch、SwiGLU、Combine、Unpermute 使用 216 KiB 主区；尾部 40 KiB 保留给同步快照 |
| `Dispatch / Unpermute tile` | Dispatch 使用自适应 2～6 槽 packed-token ring；Unpermute 尽量保留完整 K 行、单 tile 最多 8192 列，并使用自适应 2～6 槽输入 ring |
| `GMM / AIV CV tile` | GMM1 通过一个 CV 槽传递成对的 `256 x 256` BF16 half-tile；GMM2 通过一个 CV 槽传递 `256 x 256` BF16 tile |

## 支持 Case

八卡调优性能 case 固定除 `M` 外的其它主参数。同一组 shape 也可用于两卡冒烟和正确性验证；
generic tiling 还支持其它满足约束的 rank 数和 runtime 本地 expert 数。

```text
worldSize=8
K=7168
N=4096
topK=8
expertPerRank=16
aicNum=runtime（28、32 或 36），或通过 `run.sh --aicore-num 28|32|36` 指定
aivNum=2 * aicNum
```

典型 case 列表：

| M | maxOutputSize | 命令 |
| --- | --- | --- |
| 16 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 16 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 32 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 32 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 64 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 64 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 128 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 128 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 512 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 512 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 1024 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 1024 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 2048 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 2048 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |

## 整体架构

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ Front Mask Pull (AIV)                                                       │
│   BF16 -> MXFP8 source record + per-expert route mask/partial count          │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │ front metadata ready
┌──────────────────────────────▼───────────────────────────────────────────────┐
│ Dispatch + deferred metadata + dynamic GMM task producer                    │
│                                                                              │
│ AIV0: mask compact/pull + route sort/inverse + mailbox descriptor            │
│ AIC : GMM1 direct/fixed/mailbox -> released AIC joins GMM2 mailbox           │
│ AIV1:       CV SwiGLU                    CV Combine -> remote compact row     │
│                                                                              │
│ hybrid GMM1/GMM2 使用 wave0 直通 + mailbox 后缀；fixed GMM1 按 wave 调度       │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │ 有序 expert-prefix progress
┌──────────────────────────────▼───────────────────────────────────────────────┐
│ RankStreaming Unpermute（全部 AIV）                                         │
│   expandedRowIdx + compact output + probs -> topK reduce -> out[M, K]        │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Front Mask Pull 阶段

Front 在源 rank 保留每个 token 的一份 MXFP8 记录，并向目标 rank 发布逐 global expert 的 route mask：

```text
x[M, K] + expertId[M, topK]
  -> sourceTokenRecords[M, 对齐的 E4M3 数据 + 对齐的 E8M0 scale]
  -> routeMaskSlots[localExpert, srcRank, mask + laneCapacity * 32B partial counts]
  -> cumsumMM[srcRank, localExpert] / expertTokenNums[localExpert]
```

所有 AIV 按 token 维均分量化任务，不切 K。同一个 expert 可由多个 AIV lane 并行生成 mask；lane 只写
完整的 32B mask block，每个 block 覆盖 256 个 route slot，并写对齐的 partial-count record。Front
量化使用两组 PTO UB。所有 source rank 发布 front-ready epoch 后，协调核归约 lane count，并构造
source-major `cumsumMM` 和 `expertTokenNums`。

与此同时，预留 AIV0 把 route slot 排成 expert-major 顺序，构造 `expandedRowIdx`，发布 source-rank
`preSumBeforeRank`，并生成 GMM1/GMM2 mailbox descriptor。只有 deferred metadata、本地 Dispatch 和
全部 source-rank preSum 可见后，才放行 GMM2 入口。

## Dispatch 阶段

Dispatch 在目标 rank 运行。每个 source rank 分到 `dispatchGroupSize / worldSize` 个 AIV0 lane；所有 lane
扫描同一 mask，但只消费各自的命中序号区间：

```text
srcRank.sourceTokenRecords[routeSlot / topK]
  -> workspace.gmA[dstRow, 0:K]                  (E4M3)
  -> workspace.gmAScale[dstRow, 0:K/32]          (E8M0)
  -> workspace.routeMeta[dstRow] = {srcRank, routeSlot, 0...}
```

PTO vector mask 压缩生成命中 route index。每个 source rank 获得
`floor(dispatchGroupSize / worldSize)` 个 active AIV0 lane，多余 Dispatch 角色保持空闲。自适应
2～6 槽 ring 重叠远端 packed record 搬入与 E4M3/E8M0、metadata 写回。Dispatch 按 expert 和
256-row M tile 发布 cache-line 隔离的 ready count，GMM1 只等待当前将消费的输入 tile。

## GMM1 / SwiGLU / GMM2 阶段

### GMM1

GMM1 在 AIC 上执行第一层 MXFP8 grouped matmul：

```text
gmA[E4M3] / gmAScale[E8M0] x weight1[E4M3] / scale1[E8M0]
  -> int32 accumulator
  -> fixpipe MX scale 和 BF16 cast
  -> 成对的 x/gate CV tile
```

每个 local expert 按 `256 x 256` 半输出 tile 切分，x 和 gate 成对计算，并通过单槽 CV FIFO 直接交给
配对 AIV1。线性 tile id 使用 swizzle 改善 B 侧 L1 复用。`M <= 512` 时全部 AIC 直通执行 wave0，
稳态 GMM1 组从 mailbox 消费后缀；更大 M 使用 fixed-wave GMM1，先执行配置的全 AIC wave 前缀，再按配置分组。

### SwiGLU

SwiGLU 在配对 AIV1 上消费 BF16 x/gate CV tile：

```text
x/gate BF16 CV tile
  -> silu(x) * gate
  -> MXFP8 dynamic quant
  -> gmSwigluA[E4M3] + gmSwigluScale[E8M0]
```

CV control stream 携带准确的 GMM task 顺序。SwiGLU 完成每个 E4M3/E8M0 写回后，再增加对应
expert 的 GMM2 dependency counter；达到该 expert 所需的 SwiGLU tile 数后，GMM2 才开始处理该 expert。

### GMM2

GMM2 在 AIC 上执行第二层 MXFP8 grouped matmul：

```text
gmSwigluA[E4M3] / gmSwigluScale[E8M0] x weight2[E4M3] / scale2[E8M0]
  -> int32 accumulator
  -> fixpipe MX scale 和 BF16 cast
  -> GMM2/Combine CV tile
```

配置的 GMM2 组直通执行 wave0，其余 tile 都是 producer 分配的 mailbox 任务。Group1 AIC 完成各自
调度模式对应的 GMM1 handoff 后，立即进入 GMM2 mailbox，不经过全局 GMM1/GMM2 barrier。每个 AIC
通过单槽 CV FIFO 把 BF16 `256 x 256` 结果流式交给配对 Combine AIV1。

## Combine / Unpermute 阶段

Combine 在配对 AIV1 上把每个 BF16 GMM2 CV tile 直接写到源 rank 的 compact output row：

```text
GMM2 BF16 CV tile
  -> 与各 source-rank cumsum row 区间求交
  -> srcRank.combineOutputByRouteSlot[compactRow, 0:K]
```

Combine 预取 `cumsumMM` 和 `preSumBeforeRank`，然后对每个 source-rank 交集执行带 stride 的
UB-to-GM 写。发布 expert completion count 前会先排空此前的远端 MTE3 写，因此 progress coordinator
向各源 rank 暴露的是有序 expert prefix。

Unpermute 是最后的源 rank 还原阶段：

```text
expandedRowIdx[token * topK + topk]
  -> combineOutputByRouteSlot[compactRow] * probs[token, topk]
  -> 按原 token/topK 累加
  -> out[M, K]
```

`M < 512` 时全部 AIV 直接执行单个 live 阶段；`M >= 512` 时先由 16 个 AIV0 worker 消费冻结 progress
快照允许的 token，再由全部 AIV 根据 live expert-prefix progress 消费剩余 token。RankStreaming 是唯一
Unpermute 实现，不支持的 shape 由 Host 校验直接拒绝，不回退到 barrier kernel。

## 内存布局与 HCCL 窗口

HCCL remote window 主要承载跨 rank 可见的数据：

| Buffer | 位置 | 用途 |
| --- | --- | --- |
| `sourceTokenRecords` | HCCL window | 每个源 token 一份 packed E4M3/E8M0 记录；Dispatch 按 `routeSlot / topK` 拉取 |
| `routeMaskSlots` | HCCL window | 每个 local expert/source rank 的 mask 和对齐 lane partial count |
| `combineOutputByRouteSlot` | HCCL window | Combine 写入的 expert-major compact BF16 row，由 `expandedRowIdx` 索引 |
| `preSumBeforeRank` | HCCL window | 每个 global expert 的 source-rank compact-row 基址 |
| signal tail | HCCL window | Front/preSum/data-ready epoch 和 rank-progress slot |
| `gmA` / `gmAScale` | workspace GM | Dispatch 生成的 E4M3/E8M0 GMM1 输入 |
| `gmSwigluA` / `gmSwigluScale` | workspace GM | SwiGLU 生成的 E4M3/E8M0 GMM2 输入 |
| `routeMeta` | workspace GM | 包含 `{srcRank, routeSlot}` 的 32B Dispatch row record |
| `cumsumMM` | workspace GM | Dispatch 目标行地址使用的 source-major 累计 row |
| `sortedRouteSlot` / `expandedRowIdx` | workspace GM | expert-major route 顺序及其 route-slot 到 compact-row 的逆映射 |
| GMM queue/mailbox | workspace GM | runtime task descriptor、ticket、dependency counter 和 completion counter |

`run.sh` 根据 `M`、`topK`、`K`、expert 拓扑和 mask lane 容量估算 HCCL window，并在需要时自动提高
`HCCL_BUFFSIZE`。

## 构建与运行

配置 Ascend CANN 环境：

```bash
source ~/zy/set_env.sh
```

只编译 A5 mixed-core kernel 和 host：

```bash
cd ${git_clone_path}/kernels/manual/a5/dispatch_mega_combine
bash run.sh --build-only
```

请使用连续且可用的 A5 设备运行。环境需要提供 MPICH，MPI 兼容包装不支持 OpenMPI。

```bash
bash run.sh --soc Ascend910_9599 --world-size 2 --first-device 2 --m 2048 --k 7168 --n 4096 \
  --topk 8 --experts 16 --max-output-size 81940 --reuse-data
```

该命令将 rank 0、1 映射到物理卡 2、3。上面的 `Ascend910_9599` 是 A5 工具链兼容别名，不是 A3 的
`Ascend910B`。

切换其它 `M` 档位时可保持两卡首验配置，例如：

```bash
bash run.sh --soc Ascend910_9599 --world-size 2 --first-device 2 --m 512 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data
```

### 环境变量说明

| 环境变量 | 用途 | 默认行为 |
| --- | --- | --- |
| `ASCEND_HOME_PATH` | CANN 安装目录 | 必须提前设置 |
| `CMAKE_COMPILER` | CMake 使用的编译器 | `bisheng` |
| `FIRST_DEVICE` | 连续 rank 映射中的第一张物理卡 | 默认 `0`，可由 `--first-device` 覆盖 |
| `MPI_LIB_PATH` | 可选的 MPICH `libmpi.so` 绝对路径 | 默认从 `LD_LIBRARY_PATH` 解析 |
| `MPI_RUNNER` | MPICH 启动命令 | 使用已 source 环境中的 `mpirun` |
| `HCCL_BUFFSIZE` | HCCL RDMA window 大小 | `run.sh` 按 case 自动抬高到安全值 |
| `DISPATCH_MEGA_COMBINE_AICORE_NUM` | 有效 AIC 数 | 默认 `0`，使用 runtime 查询值 |
| `DISPATCH_MEGA_COMBINE_REUSE_DATA` | 复用兼容的已生成 case 数据 | 默认关闭；非零值等价于 `--reuse-data` |
| `DISPATCH_MEGA_COMBINE_WARMUP_ITERS` | 计时前的 warmup 启动次数 | `3` |
| `DISPATCH_MEGA_COMBINE_MEASURE_ITERS` | 用于整体 kernel 统计的计时启动次数 | `5` |

### 整体 Kernel 性能统计

计时启动完成后，rank 0 输出一份 `[KERNEL_PERF]` 整体 kernel 统计。每个 AIC/AIV 只记录整体起止 syscnt，统计以
每轮各 rank 最大耗时为口径，输出平均值、最小值、最大值、标准差、token 吞吐、等效计算 TFLOPS 和等效通信带宽。

## 修改 Case 参数

修改 case 时，MXFP8 维度、rank 拓扑和接收容量都需要满足 Host 校验。

```bash
bash run.sh --world-size 8 --first-device 0 --m 512 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data
```

常用约束：

- `K` 必须是 128 的倍数，并满足 Dispatch packed-row 容量。
- `N` 必须为偶数，且 `N / 2` 必须是 128 的倍数。
- `topK` 必须位于 `1..32`。
- `expertPerRank` 必须是 `4`、`8`、`16` 或 `32`，全部共用一份 runtime kernel，device 容量为
  32 个本地 expert。
- `worldSize` 不能超过所选 28/32/36-AIC 拓扑的 `min(32, aicNum - 2)`。
- `maxOutputSize` 必须覆盖单 rank 接收的 routed rows 上限。
- 每个 RankStreaming 阶段每 worker 最多处理 256 个 token；超出能力的 shape 会被 Host tiling 拒绝。
- synthetic `expert_idx` 默认使用 global token round-robin，使小 M case 也能覆盖全局 expert。

## 常见问题

| 问题 | 原因与解决 |
| --- | --- |
| `ASCEND_HOME_PATH must be set` | 运行 `run.sh` 前需要 source CANN 环境并导出 `ASCEND_HOME_PATH` |
| HCCL window too small | 手动设置的 `HCCL_BUFFSIZE` 低于 case 需求；取消覆盖或调大该变量 |
| MPI 启动失败 | source 项目环境，并确认 `mpirun --version` 显示 MPICH/HYDRA；不支持 OpenMPI |
| golden 生成很慢 | 首次生成后使用 `--reuse-data` 复用文件；chunk 大小由程序内部固定 |
| shape 或 rank 拓扑被拒绝 | 检查 28/32/36-AIC 选择、rank 上限、MXFP8 对齐、UB 容量和 RankStreaming 每 worker token 上限 |
| 结果 diff 异常 | 先检查 data cache 是否复用旧分布；改变 expert 分布或 case 关键参数后不要使用旧 `out/` |

## 构建系统

- **编译器**：`bisheng`
- **Device kernel flags**：`-xcce --cce-aicore-arch=dav-c310`，并启用 A5 address transform 和显式 DCCI 策略
- **Host executable**：`-xc++ -std=c++17`
- **输出 target**：`dispatch_mega_combine_kernel`、`dispatch_mega_combine`
- **链接库**：`stdc++`、`ascendcl`、`hcomm`、`runtime`、`tiling_api`、`platform`、`nnopbase`、`pthread` 等
- **PTO include**：仓库根目录 `include/` 会被放入 include path，用于 PTO tile/comm helper

## 变更记录

| 日期 | 变更 |
| --- | --- |
| 2026-06-26 | 新增 `dispatch_mega_combine` README，整理 MegaMoE 算子说明、阶段流程、构建运行和 FAQ |
| 2026-07-27 | 将 A3 优化调度迁移到 A5 后端，并完成仅编译验证 |
| 2026-08-14 | 删除开发期观测代码，完成 A5 生产化整理 |
| 2026-08-20 | 同步 runtime expert、deferred metadata、hybrid GMM1、mailbox GMM2、CV 直传、纯 RankStreaming Unpermute 和 Combine progress 有序发布 |
