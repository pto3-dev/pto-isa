# TSORT32

## 指令示意图

![TSORT32 tile operation](../../../../figures/isa/TSORT32.svg)

## 简介

对 `src` 的每个 32 元素块，与 `idx` 中对应的索引一起进行排序，并将排序后的值-索引对写入 `dst`。底层 SFU 指令为 **VBS32**（`vbitsort`），单次调用可排序一个或多个独立的 32 元素列表。

## Ascend310P (A1) 提案排序支持范围

A1 复用既有四操作数 `TSORT32(dst, src, idx, tmp)`：先调用 m200 的 16 提案
`vbitsort`，再对每个 32 值块执行双列表 `vmrgsort4`。原生记录包含八个 FP32
字段（字段 4 为分数，字段 5 为索引原始位）；公共结果仍为紧凑的 8 字节值-索引对。
提案布局不暴露到接口外，也不隐式分配 UB。

- 源值、结果与 scratch 为 FP32，索引为 UINT32。
- 所有操作数必须是完整静态、连续行主序、无分形的 Vec Tile，行按 32 字节对齐，
  并满足既有 UB 容量。源为 `R × C`、`C % 32 == 0`，结果为 `R × 2C`；
  索引为 `R × C` 或支持行广播的 `1 × C`。
- 包括对齐输入在内，scratch 始终至少为 **512 个 FP32 元素（2048 字节）**。
  每个块有两组 `32 × 8` 字段缓冲区。每批处理
  `B = min(127, floor(tmp.Numel / 512))` 个块，并跨批次和行复用缓冲区。
  增大 scratch 可以减少标量/向量交接次数；末批不足 B 时仍不超出预留范围。
- 本 A1 支持范围尚不含三操作数形式、FP16 值、部分有效形状及非 32 对齐尾块。
  这些情况会被拒绝，不会隐式申请 scratch 或改变公共值-索引对格式。
- 结果/scratch 与其他操作数的任何可写重叠会在开始前拒绝；两个只读输入可别名。
- 原生提案指令只比较分数。压紧结果时，在相等分数内恢复**无符号**索引升序，
  不把索引转换为 FP32，不改变分数原始位，包括符号零。支持有限分数与无穷大；
  NaN 分数顺序未定义。当值与索引都相等时，不额外保证遍历顺序。
- 调用方必须先完成输入 DMA 到 `PIPE_V` 的同步。局部事件 7 负责 V→S 提取及
  S→V 发布，并在返回前排空。事件 7 为局部 ISA helper 保留，调用方不能在其上
  留有未完成事务。操作对外仍属于 `PIPE_V`，消费方按通常规则同步，
  后续存储包括 V→MTE3 同步。helper 恢复 normal/full 向量掩码。

CPU 参考、执行生产 A1 helper 的宿主指令模型与 CCEC 编译用于验证语义和原生
模板实例化，不证明设备时延、异步 DMA 正确性或性能。A1 编译器与公共 PyPTO
接入仍需提供显式 scratch，完成后才能替换模型的原生排序 kernel。

## 硬件：VBS32（`vbitsort`）

VBS32 运行在 **SFU**（非向量流水线）上。一次调用排序 `repeat` 个连续的 32 元素块，每个块由 32 个值 + 32 个索引组成，打包为值-索引对：

```cpp
void vbitsort(__ubuf__ T *dst,        // 排序后的值-索引对输出
              __ubuf__ T *src0,        // 每块 32 个值 × repeat
              __ubuf__ uint32_t *src1, // 每块 32 个索引 × repeat
              uint8_t repeat);         // 32 元素块的数量（1..255）
```

- `repeat`（上限 `REPEAT_MAX = 255`）打包到 `config[63:56]`。
- 块在内存中**连续分布，步长为 32 个元素**：块 `b` 读取 `src0[b*32 : b*32+32]` 和 `src1[b*32 : b*32+32]`，写入 `dst[b*32*coef : ...]`，其中 `coef` = 2（float）或 4（half）——值-索引对的扩展因子。
- 排序顺序：按值**降序**；相同值时索引小者优先。

## 数学语义

对每一行 `r`，`src` 按独立的 32 元素块处理。设块 `b` 覆盖列 `32b … 32b+31`，`n_b = min(32, C - 32b)` 为其有效元素数。

$$
(v_k, i_k) = (\mathrm{src}_{r,32b+k},\; \mathrm{idx}_{r,32b+k}), \quad 0 \le k < n_b
$$

按值降序排序，输出重排后的序列：

$$
[(v_{\pi(0)}, i_{\pi(0)}),\; (v_{\pi(1)}, i_{\pi(1)}),\; \ldots]
$$

其中 `π` 为该块的排序置换。

注：
- `idx` 是输入 Tile（索引随值一起被重排），不是输出。
- `dst` 存储排序后的值-索引对，而非仅排序后的值。

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
// 3 参数：src 必须 32 对齐（validCol % 32 == 0）
template <typename DstTileData, typename SrcTileData, typename IdxTileData>
PTO_INST RecordEvent TSORT32(DstTileData &dst, SrcTileData &src, IdxTileData &idx);

// 4 参数：支持非 32 对齐尾部（validCol % 32 != 0），通过 tmp 填充
template <typename DstTileData, typename SrcTileData, typename IdxTileData, typename TmpTileData>
PTO_INST RecordEvent TSORT32(DstTileData &dst, SrcTileData &src, IdxTileData &idx, TmpTileData &tmp);
```

## Tile 尺寸与数据类型

对于 `src` 形状为 $R \times C$（有效区域）、块大小 32：

| Tile | dtype | 尺寸（元素数） | 说明 |
|------|-------|----------------|-------|
| `src` | `half` 或 `float`（$T$） | $R \times C$ | 待排序的值 |
| `idx` | `uint32_t` | $R \times C$（或 $1 \times C$ 广播） | 随值重排的索引 |
| `dst` | $T$ | $R \times (2C)$ float，$R \times (4C)$ half | 排序后的值-索引对（见下方扩展因子） |
| `tmp`（仅 4 参数） | $T$ | 见下方 tmp 尺寸公式 | 尾部填充 scratch |

**`dst` 扩展因子**（`typeCoef`）：每个输入元素生成一个 8 字节的 tuple `[value (4 B), index (4 B)]`——`float` 的 value 占满 4 B；`half` 的 2 B value 零扩展至 4 B。因此 `dst` 恒为 $C \times 8$ 字节。

| dtype | 每个 `src` 列对应的 `dst` 列数（dtype 单位） | tuple 布局 | 字节/tuple |
|-------|-------------------------------------------|--------------|------------|
| `float` | ×2（2 个 float 槽位） | `[value_f32, index_u32]` | 8 |
| `half` | ×4（4 个 half 槽位） | `[value_f16, 0x0000, index_u32]` | 8 |

## 约束

| 约束 | 原因 |
|------------|------|
| `dst`/`src` dtype = `half` 或 `float`（须一致）；`idx` = `uint32_t` | VBS32 类型分派 |
| 所有 Tile 为 `TileType::Vec`、`BLayout::RowMajor` | SFU 寻址 |
| `validCol % 32 == 0`（3 参数） | 每块恰为 32 个元素 |
| `validCol` 任意（4 参数） | 尾块通过 `tmp` 填充至 32，填充值为 $-\infty$ |
| `repeat = validCol/32`（3 参数）或 `ceil(validCol/32)`（4 参数） | VBS32 repeat 计数，每次调用 ≤ 255；更大的 `validCol` 拆分为多次 `vbitsort` 调用 |
| `tmp`（4 参数）≥ `tmpSize` 元素（见下方公式） | 保存填充后的行/尾块副本 |
| 无 `WaitEvents&...` / 无内部 `TSYNC` | 如需同步须显式调用 |

### `tmp` 尺寸公式（4 参数）

设 $C$ = `validCol`，$b$ = `sizeof(T)` 字节数，$G$ = 32（块大小）。实现根据整行（字节数）是否满足 `MAX_UB_TMP = 8160` 进行分支：

$$
\mathrm{tmpSize} =
\begin{cases}
\mathrm{ceil}_{G}(C) & \text{若 } C \cdot b \le 8160 \quad \text{（整行复制到 tmp + 填充最后一块）} \\
G = 32 & \text{若 } C \cdot b > 8160 \quad \text{（仅复制尾块 + 填充）}
\end{cases}
$$

- `ceil_G(C)` = $C$ 向上取整到 32 的倍数。
- `8160` 阈值单位为**字节**（与 `validCol * sizeof(T)` 比较）：float → $C \le 2040$，half → $C \le 4080$。该阈值为 `pto_copy_ubuf_to_ubuf`（MOV_UB_TO_UB）的 repeat 上限 = 255 块 × 32 B。
- 尾块 = $t = C \bmod G$ 个元素（末尾不完整块），扩展至 $G$ 并以 $-\infty$ 填充。
- Path A（$C \cdot b \le 8160$，小行）：从行首**整行**复制到 tmp，然后原地填充最后 32 个元素。
- Path B（$C \cdot b > 8160$，大行）：仅复制**尾块**到 tmp；完整块直接从 `src` 排序。
- VBS32 硬件上限：每次调用 `repeat ≤ REPEAT_MAX = 255` 块（≤ 8160 元素）；超过 255 块的行拆分为多次 `vbitsort` 调用。
- **UB 布局：** `tmp` 应放置在 `dst` 之后（32 B 对齐），大小为 `ceil(ALIGN_C·b, 32)` 字节——不应使用固定的 8 KB 偏移，因为 Path A 在接近阈值时对 float 需要最多 ~32 KB。

### 4 参数尾部处理

当 `validCol % 32 != 0` 时，末尾不完整块（$t = C \bmod 32$ 个元素）须填充为完整的 32 元素块后才能送入 `vbitsort`。两条路径：

- **$C \cdot b \le 8160$**（小行）：**整行**复制到 `tmp`，然后通过 `vdup` 原地覆盖最后 32 个元素为 $-\infty$ 填充；从 `tmp` 排序整行。
- **$C \cdot b > 8160$**（大行）：仅复制**尾块**到 `tmp` 并填充；完整块直接从 `src` 排序，仅尾块从 `tmp` 排序。

填充值（$-\infty$ = `-(0.0/0.0)`）落在降序排序的底部。若 `validCol > 32 × 255`，行按 `REPEAT_MAX` 大小的组拆分，每组通过独立的 `vbitsort` 调用排序。

## 汇编语法

### AS Level 1（SSA）

```text
%dst = pto.tsort32 %src, %idx : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tsort32 ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## 示例

```cpp
#include <cstdint>
#include <pto/pto-inst.hpp>
using namespace pto;

// 32 对齐：每行单个块
using SrcT = Tile<TileType::Vec, float, 1, 32>;
using IdxT = Tile<TileType::Vec, uint32_t, 1, 32>;
using DstT = Tile<TileType::Vec, float, 1, 64>;   // 2× src 列数（float）
SrcT src; IdxT idx; DstT dst;
TSORT32(dst, src, idx);

// 非 32 对齐尾部：4 参数 + tmp
using SrcT2 = Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 100>;
using IdxT2 = Tile<TileType::Vec, uint32_t, 1, 128, BLayout::RowMajor, 1, 100>;
using DstT2 = Tile<TileType::Vec, half, 1, 512, BLayout::RowMajor, 1, 400>;  // 4× src 列数（half）
using TmpT  = Tile<TileType::Vec, half, 1, 128>;  // ≥ ceil32(100)=128
SrcT2 src2; IdxT2 idx2; DstT2 dst2; TmpT tmp;
TSORT32(dst2, src2, idx2, tmp);
```

## ASM 形式示例

### Auto 模式

```text
%dst = pto.tsort32 %src, %idx : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual 模式

```text
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
# pto.tassign %arg2, @tile(0x3000)
%dst = pto.tsort32 %src, %idx : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = tsort32 %src, %idx : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
# AS Level 2（DPS）
pto.tsort32 ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
