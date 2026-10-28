# pto.tmax

`pto.tmax` is part of the [Elementwise Tile Tile](../../elementwise-tile-tile.md) instruction set.

## Summary

Elementwise maximum of two tiles.

## Mechanism

Elementwise maximum of two tiles.

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \max(\mathrm{src0}_{i,j}, \mathrm{src1}_{i,j}) $$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = tmax %src0, %src1 : !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tmax %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TMAX(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## Inputs

| Operand | Role | Description |
|---------|------|-------------|
| `%src0` | Left tile | First source tile; read at `(i, j)` for each `(i, j)` in `dst` valid region |
| `%src1` | Right tile | Second source tile; read at `(i, j)` for each `(i, j)` in `dst` valid region |
| `WaitEvents...` | Optional synchronisation | `RecordEvent` tokens to wait on before issuing the operation |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%dst` | `!pto.tile<...>` | Destination tile; all `(i, j)` in its valid region contain `max(src0[i,j], src1[i,j])` after the operation |

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

!!! warning "Constraints"
    - **Valid region**:
        - The op uses `dst.GetValidRow()` / `dst.GetValidCol()` as the iteration domain; `src0/src1` are assumed to be compatible (not validated by explicit runtime checks in this op).

## Exceptions

!!! danger "Exceptions"
    - Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
    - Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - **Implementation checks (A1)**:
        - Full static row-major FP16/FP32 Vec tiles with matching shapes/types, 32-byte-aligned rows and bounded UB storage.
        - Exact destination aliases of either input are supported; shifted overlaps are rejected before execution.
        - Repeat groups are limited to 255 and masked tails preserve adjacent data. CPU/native tests qualify finite values including subnormals and normal limits; NaN selection is not qualified.

    - **Implementation checks (A2A3)**:
        - `TileData::DType` must be one of: `int32_t`, `int16_t`, `half`, `float`.
        - Tile layout must be row-major (`TileData::isRowMajor`).
        - Tile location must be vector (`TileData::Loc == TileType::Vec`).
        - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
        - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.

    - **Implementation checks (A5)**:
        - `TileData::DType` must be one of: `uint32_t`, `int32_t`, `uint16_t`, `int16_t`, `uint8_t`,  `int8_t`, `float`, `half`.
        - Tile layout must be row-major (`TileData::isRowMajor`).
        - Tile location must be vector (`TileData::Loc == TileType::Vec`).
        - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
        - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.

## Performance

### A2/A3 Throughput

`TMAX` compiles to CCE vector instructions via the `TBinOp.hpp` performance model. The throughput is identical to `TADD` (binary arithmetic):

| Metric | Value (FP) | Value (INT) |
|--------|-------------|-------------|
| Startup latency | 14 | 14 |
| Completion latency | 19 | 17 |
| Per-repeat throughput | 2 | 2 |
| Pipeline interval | 18 | 18 |

---

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TMAX(dst, src0, src1);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TASSIGN(src0, 0x1000);
  TASSIGN(src1, 0x2000);
  TASSIGN(dst,  0x3000);
  TMAX(dst, src0, src1);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tmax %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tmax %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tmax %src0, %src1 : !pto.tile<...>
# AS Level 2 (DPS)
pto.tmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Elementwise Tile Tile](../../elementwise-tile-tile.md)
- Previous op in instruction set: [pto.tmin](./tmin.md)
- Next op in instruction set: [pto.tcmp](./tcmp.md)
