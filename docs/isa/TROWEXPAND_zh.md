# TROWEXPAND

## 指令示意图

![TROWEXPAND tile operation](../figures/isa/TROWEXPAND.svg)

## 简介

将每个源行的第一个元素广播到目标行中。

## 数学语义

设 `R = dst.GetValidRow()`，`C = dst.GetValidCol()`。对 `0 <= i < R` 且 `0 <= j < C`：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,0} $$

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPAND(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

## 约束

实现检查 (NPU):

- Tile类型：`dst` 和 `src` 必须是 `TileType::Vec`。
- Tile布局：`dst` 为非分形 RowMajor；`src` 可为非分形 RowMajor 或 ColMajor（`SLayout::NoneBox`）。输入输出元素类型必须相同。
- A5 的源静态 `ValidCol` 必须为 1 或动态值 -1；单标量行广播应使用有效列数 1。
- 数据类型（Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品）：元素类型必须是以下之一：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half`、`bfloat16_t`、`float`。
- 数据类型（Ascend 950PR/Ascend 950DT）：元素类型必须是以下之一：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`int64_t`、`uint64_t`、`half`、`bfloat16_t`、`float`。
- 运行期有效区域检查：
    - Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品：若 `dstValidRow`、`dstValidCol`、`srcValidRow`、`srcValidCol` 中任意一个为零则提前返回。
    - Ascend 950PR/Ascend 950DT：断言 `srcValidRow == dstValidRow`，且断言 `srcValidRow != 0 && srcValidCol != 0`。

- 64 位输入（Ascend 950PR/Ascend 950DT）：ND 源可使用物理 `[64,4]`、有效 `[64,1]`，DN 源可使用紧凑 `[64,1]`。每行广播 `src[i,0]`，地址使用物理 `RowStride`。
- 与 [TROWEXPANDADD](TROWEXPANDADD_zh.md) 等二元行广播不同，本指令的 ND 源仍提供每行单标量，不重复广播 32 字节块。

## Ascend 310P（A1）

A1 支持同类型 FP16/FP32 Vec，源为完整静态列主序紧凑 `M x 1`，目标为行主序
`M x N`，遵循 A1 的 32 字节对齐和 UB 容量检查，源与目标必须互不重叠。

当 N >= 16 且行步长不超过 255 个 32 字节块时，原生 b16/b32 转置将输入值复制到
各输出行的前若干块，再通过按位 OR 复制到该行剩余列。该路径无需标量流水读取 UB
数值，也不额外分配 UB 临时空间；包括正负零和 NaN payload 在内，所有位模式均保持。
单次转置 repeat 使用零 stride；复制的 repeat 数与掩码受限，包含末组 repeat 和列尾。
更窄的 FP32 行及更大行步长继续使用既有标量实现。生产者与消费者的流水同步仍由调用方负责。

`tests/compile/test_a1_row_broadcast_bits_native.py` 先运行 CPU oracle，再检查设备
输出及输入的原始位模式、相邻 UB/GM 守卫、全部 65,536 种 FP16 模式、FP32 特殊及
随机模式，以及 stride/repeat 边界。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, 16, 1>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TROWEXPAND(dst, src);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, 16, 1>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TROWEXPAND(dst, src);
}
```
