# TRESHAPE


## Tile Operation Diagram

![TRESHAPE tile operation](../figures/isa/TRESHAPE.svg)

## Introduction

Reinterpret a tile as another tile type/shape while preserving the underlying bytes.

This is a *bitwise* reshape: it does not change values, it only changes how the same byte buffer is viewed.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TRESHAPE(TileDataOut &dst, TileDataIn &src, WaitEvents &... events);
```

## Constraints

Enforced by `TRESHAPE_IMPL`:

- **Tile type must match**: `TileDataIn::Loc == TileDataOut::Loc`.
- **Total byte size must match**: `sizeof(InElem) * InNumel == sizeof(OutElem) * OutNumel`.
- **No boxed/non-boxed conversion**:
    - cannot reshape between `SLayout::NoneBox` and boxed layouts.

## Notes

- **CPU simulation** (`__CPU_SIM`): the destination aliases the same underlying bytes; no payload copy.
- **A2/A3**: implemented as an alias (`TASSIGN_IMPL(dst, src.data())`), so `dst` and `src` refer to the same underlying storage.

## Examples

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

## Math Interpretation

Unless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

## Ascend310P (A1) storage views

A1 supports FP16, FP32, INT32 and UINT32 full row-major, non-boxed Vec
views with identical source/destination element types and counts. Both rows
must be contiguous and 32-byte aligned, and each view must fit the 128 KiB UB.
TRESHAPE rebinds the destination address to the source; it allocates no scratch
and moves no payload bytes. Later DMA writes through either shape are visible
through the other. Integer views do not enable integer arithmetic or conversions;
signedness and FP32/integer reinterpretation remain outside this A1 profile.

CPU/offline regressions cover flattening, row regrouping, identity views, raw
NaN/index bits, DMA write-through, input/GM guards and near-capacity storage.
They execute both generic CPU and the production helper with CPU storage and
compile native dav-m200 objects. Device concurrency and timing remain unqualified.
