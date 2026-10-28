# TMATMUL


## Tile Operation Diagram

![TMATMUL tile operation](../figures/isa/TMATMUL.svg)

## Introduction

Matrix multiply (GEMM) producing an accumulator/output tile.

## Math Interpretation

Let:

- `M = aMatrix.GetValidRow()`
- `K = aMatrix.GetValidCol()`
- `N = bMatrix.GetValidCol()`

For `0 <= i < M` and `0 <= j < N` (output elements in the effective matmul domain):

$$ \mathrm{C}_{i,j} = \sum_{k=0}^{K-1} \mathrm{A}_{i,k} \cdot \mathrm{B}_{k,j} $$

Exact accumulator behavior and datatype promotion are target/implementation-defined.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, WaitEvents &... events);

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, WaitEvents &... events);
```

## Constraints

- **Accumulator pitch constraint (CPU, A2A3, A5, A6, and Kirin9030)**: `mad` writes the
  accumulator's block columns at a pitch of `ceil16(m)` rows, where `m` is the row count the
  instruction receives, while every reader of that region takes its pitch from the tile's
  `Rows`. The two must agree:
    - `((m + 15) / 16) * 16 == TileRes::Rows`, where `m` is `TileLeft::ValidRow` for
      `TMATMUL` and 1 for `TGEMV`. Outside `TGEMV`, `m == 1` is promoted to 16.
    - Not enforced when `TileRes::Cols <= FRACTAL_NZ_ROW` (a single block column needs no
      pitch), when `TileRes::Compact != CompactMode::Null`, or when either value is
      `DYNAMIC`. `FRACTAL_NZ_ROW` is 16.
    - `TileRes::ValidRow` does not enter the condition. A tile whose `ValidRow` is smaller
      than its `Rows` is fine as long as the pitch matches: `Rows` is then just the padded
      allocation size.
- **Implementation checks (A2A3)**:
    - Supported `(CType, AType, BType)` triples:
    - `(int32_t, int8_t, int8_t)`
    - `(float, half, half)`
    - `(float, float, float)`
    - `(float, bfloat16_t, bfloat16_t)`
    - Static shape constraints: `TileLeft::Rows == TileRes::Rows`, `TileLeft::Cols == TileRight::Rows`, `TileRight::Cols == TileRes::Cols`.
    - Tile locations: `TileLeft::Loc == Left`, `TileRight::Loc == Right`, `TileRes::Loc == Acc`.
    - Runtime: `m/k/n` (taken from `aMatrix.GetValidRow()`, `aMatrix.GetValidCol()`, `bMatrix.GetValidCol()`) must be in `[1, 4095]`.
- **Implementation checks (A5)**:
    - Accumulator type must be `int32_t` or `float`.
    - If `int32_t`: `AType == int8_t` and `BType == int8_t`.
    - If `float`: supports `half/bfloat16_t/float`, selected fp8 pairs, and `hifloat8_t/hifloat8_t` (target-defined).
    - Static shape constraints: `TileLeft::Rows == TileRes::Rows`, `TileLeft::Cols == TileRight::Rows`, `TileRight::Cols == TileRes::Cols`.
    - Fractal/layout constraints are enforced:
    - Left: `Loc == Left`, `!isRowMajor`, `SFractal == RowMajor`
    - Right: `Loc == Right`, `isRowMajor`, `SFractal == ColMajor`
    - Acc: `Loc == Acc`, `!isRowMajor`, `SFractal == RowMajor`
    - Runtime: `m/k/n` (taken from `aMatrix.GetValidRow()`, `aMatrix.GetValidCol()`, `bMatrix.GetValidCol()`) must be in `[1, 4095]`.

- **Implementation checks (CPU_SIM, FP8/HIF8)**:
    - The accumulator type must be `float`. Each input may independently be `float8_e4m3_t` or
      `float8_e5m2_t` (all four combinations); `hifloat8_t` supports only `hifloat8_t` as the other input.
    - A5 simulation mode is required. Select the target architecture before assigning tiles; see
      [Selecting the simulated architecture](../coding/cpu_sim.md#selecting-the-simulated-architecture).
    - Inputs are converted to `float` and accumulated using fused multiply-add; results are not
      guaranteed to match NPU hardware bit for bit.

## Ascend 310P (A1) multi-K profile

A1 supports FP16 `M x K` Left times FP16 `K x 16` Right with an FP32
`M x 16` NZ accumulator. M and K are static multiples of 16 in [16, 2048],
with full valid shapes; each operand must fit its 64 KiB L0A/L0B allocation.
N remains 16. Wider N and partial tiles are not part of this profile.

- Multi-K Left uses **ZZ** (row-major outer and inner fractals, 512 bytes).
  The legacy NZ Left encoding remains accepted at K=16, where outer orders
  coincide. Right uses ZN; Acc uses NZ with 1024-byte FP32 fractals.
- Public `TLOAD` accepts matching, aligned static FP16 ND matrix rows and
  constructs ZZ or NZ L1 Mat panels. `TMOV` converts either to the required
  Left order and transposes the Right inner fractals. Live L1 allocations
  must remain disjoint and fit the 1 MiB budget.
- `TMATMUL_ACC` requires the same input/output accumulator address. It keeps
  the existing matrix-pipeline barrier before the dependent MAD.
- FP32 Acc-to-Vec writeback remains the N16 path, with explicit
  `AccToVecMode::SingleModeVec0`. DMA, operand and writeback synchronization
  remain the caller's responsibility.

Hardware `TIMG2COL` produces the same ZZ Left format. The composed im2col →
GEMM regression covers a padded 3x3 C16 convolution using a K144 operand,
including an M-window offset and dependent accumulation. Binary-fraction
inputs make CPU and device checks exact without a tolerance. These checks
establish the primitive contract, not whole-model accuracy or a speedup.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using A = TileLeft<half, 16, 16>;
  using B = TileRight<half, 16, 16>;
  using C = TileAcc<float, 16, 16>;
  A a;
  B b;
  C c;
  TMATMUL(c, a, b);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using A = TileLeft<half, 16, 16>;
  using B = TileRight<half, 16, 16>;
  using C = TileAcc<float, 16, 16>;
  A a;
  B b;
  C c;
  TASSIGN(a, 0x1000);
  TASSIGN(b, 0x2000);
  TASSIGN(c, 0x3000);
  TMATMUL(c, a, b);
}
```
