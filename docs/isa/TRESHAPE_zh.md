# TRESHAPE

## 指令示意图

![TRESHAPE tile operation](../figures/isa/TRESHAPE.svg)

## 简介

将Tile重新解释为另一种Tile类型/形状，同时保留底层字节。

## 数学语义

除非另有说明，语义定义在有效区域上，目标相关行为标记为实现定义。

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TRESHAPE(TileDataOut &dst, TileDataIn &src, WaitEvents &... events);
```

## 约束

由 `TRESHAPE_IMPL` 强制执行：

- **Tile类型必须匹配**：`TileDataIn::Loc == TileDataOut::Loc`。
- **总字节大小必须匹配**：`sizeof(InElem) * InNumel == sizeof(OutElem) * OutNumel`。
- **不允许boxed/non-boxed转换**：
    - 不能在 `SLayout::NoneBox` 与boxed布局之间进行reshape。

## 备注

- **CPU模拟**（`__CPU_SIM`）：目的视图引用同一底层字节，无载荷复制。
- **Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品**：实现为别名（`TASSIGN_IMPL(dst, src.data())`），因此 `dst` 和 `src` 引用同一底层存储。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using Src = Tile<TileType::Vec, float, 16, 16>;
  using Dst = Tile<TileType::Vec, float, 8, 32>;
  static_assert(Src::Numel == Dst::Numel);

  Src src;
  Dst dst;
  TRESHAPE(dst, src);
}
```

## Ascend310P (A1) 存储视图

A1支持FP16、FP32、INT32、UINT32的完整行主序、非分形Vec视图，源和目的的
元素类型及数量必须一致。两种形状的行均连续且按32字节对齐，每个视图适合
128 KiB UB。TRESHAPE将目的地址重新绑定到源，无scratch、无载荷搬运；随后
经任一形状DMA写入的数据在另一形状可见。整数视图不开放整数运算或转换；
有符号/无符号转换及FP32/整数重解释仍不属于本A1支持范围。

CPU/离线回归覆盖展平、行重组、同形状视图、NaN/索引原始位、DMA写穿、输入
与GM守卫及接近容量边界的存储。分别执行通用CPU实现和使用CPU存储的生产helper，
并编译原生dav-m200对象。设备并发和时延尚未验证。
