# pto.tsort32

`pto.tsort32` is part of the [Irregular And Complex](../../irregular-and-complex.md) instruction set.


## Tile Operation Diagram

![TSORT32 tile operation](../../../../figures/isa/TSORT32.svg)

## Introduction

Sort each 32-element block of `src` together with the corresponding indices from `idx`, and write the sorted value-index pairs into `dst`. The underlying SFU instruction is **VBS32** (`vbitsort`), which sorts one or more independent 32-element lists in a single call.

## Ascend310P (A1) proposal-backed profile

A1 implements the existing four-operand `TSORT32(dst, src, idx, tmp)` using
m200's 16-proposal `vbitsort`, followed by a two-list `vmrgsort4` for each
32-value block. Native records have eight FP32 fields (score at field 4,
index bits at field 5); the public destination remains compact 8-byte pairs.
No proposal layout escapes the helper and no implicit UB memory is allocated.

- Values, destination and scratch are FP32; indices are UINT32.
- All operands are full static contiguous row-major, unboxed Vec tiles with
  32-byte-aligned rows and the existing UB budget. `src` is `R × C`, with
  `C % 32 == 0`; `dst` is `R × 2C`. Index geometry is `R × C` or `1 × C`
  (row broadcast).
- Scratch always needs **at least 512 FP32 elements (2048 bytes)**, including
  aligned inputs. It holds two banks of `32 × 8` fields per block. One call
  batches `B = min(127, floor(tmp.Numel / 512))` blocks, reusing the banks
  across batches and rows. Extra scratch can amortize scalar/vector handoffs;
  an incomplete final batch stays within the same reserved banks.
- The three-operand overload, FP16 values, partial valid shapes and non-32
  tails are not implemented by this A1 profile. They are rejected rather than
  allocating hidden scratch or changing the public pair format.
- All writable overlaps involving destination/scratch are rejected before
  work starts. The read-only source and indices may alias.
- Native proposals compare scores only. Compaction restores ascending
  **unsigned** indices within each equal-score run, without converting indices
  to FP32 or changing score bits, including signed zero. Finite scores and
  infinities are supported; NaN score ordering is unspecified. Sorting equal
  value/index keys does not add a traversal-order guarantee.
- Callers must complete input DMA on `PIPE_V` before the call. Local event 7
  orders V-to-S extraction and S-to-V publication, and is drained before
  return. Event 7 is reserved for local ISA helpers; callers must not leave
  outstanding transactions on it. The outward operation remains `PIPE_V`:
  synchronize its consumer normally, including V-to-MTE3 for a subsequent store.
  The helper restores
  the normal full vector mask.

CPU reference checks, a host model executing the production A1 helper, and
CCEC compilation validate semantics and native template instantiation. They
do not establish device timing, asynchronous DMA correctness or performance.
A1 compiler/public-PyPTO integration must supply the explicit scratch before
this profile can replace a model's native sort kernel.

## Hardware: VBS32 (`vbitsort`)

VBS32 runs on the **SFU** (not the vector pipeline). One invocation sorts `repeat` consecutive 32-element blocks, each consisting of 32 values + 32 indices packed as value-index pairs:

```cpp
void vbitsort(__ubuf__ T *dst,        // sorted value-index pairs out
              __ubuf__ T *src0,        // 32 values per block × repeat
              __ubuf__ uint32_t *src1, // 32 indices per block × repeat
              uint8_t repeat);         // number of 32-element blocks (1..255)
```

- `repeat` (cap `REPEAT_MAX = 255`) is packed into `config[63:56]`.
- Blocks are **contiguous, strided by 32 elements**: block `b` reads `src0[b*32 : b*32+32]` and `src1[b*32 : b*32+32]`, writes `dst[b*32*coef : ...]` where `coef` = 2 (float) or 4 (half) — the value-index pair expansion factor.
- Sort order: **descending** by value; ties broken by smaller index first.

## Math Interpretation

For each row `r`, `src` is processed in independent 32-element blocks. Let block `b` cover columns `32b … 32b+31`, and `n_b = min(32, C - 32b)` be its valid count.

$$
(v_k, i_k) = (\mathrm{src}_{r,32b+k},\; \mathrm{idx}_{r,32b+k}), \quad 0 \le k < n_b
$$

Sort the pairs by value (descending); output the reordered sequence:

$$
[(v_{\pi(0)}, i_{\pi(0)}),\; (v_{\pi(1)}, i_{\pi(1)}),\; \ldots]
$$

where `π` is the sort permutation for that block.

Notes:
- `idx` is an input tile (indices permuted with values), not an output.
- `dst` stores sorted value-index pairs, not just sorted values.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
// 3-arg: src must be 32-aligned (validCol % 32 == 0)
template <typename DstTileData, typename SrcTileData, typename IdxTileData>
PTO_INST RecordEvent TSORT32(DstTileData &dst, SrcTileData &src, IdxTileData &idx);

// 4-arg: supports non-32-aligned tails (validCol % 32 != 0) via tmp padding
template <typename DstTileData, typename SrcTileData, typename IdxTileData, typename TmpTileData>
PTO_INST RecordEvent TSORT32(DstTileData &dst, SrcTileData &src, IdxTileData &idx, TmpTileData &tmp);
```

## Tile Sizes & Data Types

For `src` of shape $R \times C$ (valid region), block size 32:

| Tile | dtype | Size (elements) | Notes |
|------|-------|-----------------|-------|
| `src` | `half` or `float` ($T$) | $R \times C$ | values to sort |
| `idx` | `uint32_t` | $R \times C$ (or $1 \times C$ broadcast) | indices permuted with values |
| `dst` | $T$ | $R \times (2C)$ float, $R \times (4C)$ half | sorted value-index pairs (expansion below) |
| `tmp` (4-arg only) | $T$ | see tmp-size equation below | tail-padding scratch |

**`dst` expansion factor** (`typeCoef`): each input element becomes an 8-byte tuple `[value (4 B), index (4 B)]` — for `float` the value fills 4 B; for `half` the 2-B value is zero-padded to 4 B. So `dst` is always `C × 8` bytes.

| dtype | `dst` cols per `src` col (in dtype units) | tuple layout | bytes/tuple |
|-------|-------------------------------------------|--------------|------------|
| `float` | ×2 (2 float slots) | `[value_f32, index_u32]` | 8 |
| `half` | ×4 (4 half slots) | `[value_f16, 0x0000, index_u32]` | 8 |

## Constraints

| Constraint | Reason |
|------------|--------|
| `dst`/`src` dtype = `half` or `float` (must match); `idx` = `uint32_t` | VBS32 type dispatch |
| All tiles `TileType::Vec`, `BLayout::RowMajor` | SFU addressing |
| `validCol % 32 == 0` (3-arg) | each block exactly 32 elements |
| `validCol` arbitrary (4-arg) | tail block padded to 32 with $-\infty$ via `tmp` |
| `repeat = validCol/32` (3-arg) or `ceil(validCol/32)` (4-arg) | VBS32 repeat count, ≤ 255 per call; larger `validCol` splits into multiple `vbitsort` calls |
| `tmp` (4-arg) ≥ `tmpSize` elements (equation below) | holds the padded copy of the tail/row |
| No `WaitEvents&...` / no internal `TSYNC` | synchronize explicitly if needed |

### `tmp` size equation (4-arg)

Let $C$ = `validCol`, $b$ = `sizeof(T)` bytes, $G$ = 32 (block size). The implementation branches on whether the whole row (in bytes) fits `MAX_UB_TMP = 8160`:

$$
\mathrm{tmpSize} =
\begin{cases}
\mathrm{ceil}_{G}(C) & \text{if } C \cdot b \le 8160 \quad \text{(whole row copied to tmp + last block padded)} \\
G = 32 & \text{if } C \cdot b > 8160 \quad \text{(only the tail block copied + padded)}
\end{cases}
$$

- `ceil_G(C)` = $C$ rounded up to the next multiple of 32.
- The `8160` threshold is in **bytes**: it is the `pto_copy_ubuf_to_ubuf` (MOV_UB_TO_UB) repeat cap = 255 blocks × 32 B. Path A copies the whole row in one such call, so the row must be ≤ 8160 B (float → $C \le 2040$, half → $C \le 4080$). This is independent of the VBS32 `repeat` cap (which is in elements, ≤ 8160).
- Tail block = $t = C \bmod G$ elements (the trailing partial block), extended to $G$ with $-\infty$.
- Path A ($C \cdot b \le 8160$) copies the **entire row** from its start into tmp, then pads the last 32 elements in place.
- Path B ($C \cdot b > 8160$) copies **only the tail block** into tmp; full blocks are sorted directly from `src`.
- VBS32 hard cap: `repeat ≤ REPEAT_MAX = 255` blocks per call (≤ 8160 elements); rows longer than 255 blocks are split across multiple `vbitsort` calls.
- **UB placement:** `tmp` should be placed right after `dst` (32-B aligned), sized `ceil(ALIGN_C·b, 32)` bytes — not at a fixed 8 KB offset, since Path A needs up to ~32 KB for float near the threshold.

### 4-arg tail handling

When `validCol % 32 != 0`, the trailing partial block ($t = C \bmod 32$ elements) must be padded to a full 32-element block before `vbitsort`. Two paths:

- **$C \cdot b \le 8160$** (small row): the **entire row** is copied to `tmp`, then the last 32 elements are overwritten in place with $-\infty$ padding via `vdup`; the row is sorted from `tmp`.
- **$C \cdot b > 8160$** (large row): only the **tail block** is copied to `tmp` and padded; full blocks are sorted directly from `src`, only the tail is sorted from `tmp`.

Padding values ($-\infty$ = `-(0.0/0.0)`) land at the bottom of the descending order. If `validCol > 32 × 255`, the row is chunked into `REPEAT_MAX`-sized groups, each sorted via a separate `vbitsort` call.

## Assembly Syntax

### AS Level 1 (SSA)

```text
%dst = pto.tsort32 %src, %idx : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tsort32 ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Examples

```cpp
#include <cstdint>
#include <pto/pto-inst.hpp>
using namespace pto;

// 32-aligned: single block per row
using SrcT = Tile<TileType::Vec, float, 1, 32>;
using IdxT = Tile<TileType::Vec, uint32_t, 1, 32>;
using DstT = Tile<TileType::Vec, float, 1, 64>;   // 2× src cols (float)
SrcT src; IdxT idx; DstT dst;
TSORT32(dst, src, idx);

// Non-32-aligned tail: 4-arg with tmp
using SrcT2 = Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 100>;
using IdxT2 = Tile<TileType::Vec, uint32_t, 1, 128, BLayout::RowMajor, 1, 100>;
using DstT2 = Tile<TileType::Vec, half, 1, 512, BLayout::RowMajor, 1, 400>;  // 4× src cols (half)
using TmpT  = Tile<TileType::Vec, half, 1, 128>;  // ≥ ceil32(100)=128
SrcT2 src2; IdxT2 idx2; DstT2 dst2; TmpT tmp;
TSORT32(dst2, src2, idx2, tmp);
```

## ASM Form Examples

### Auto Mode

```text
%dst = pto.tsort32 %src, %idx : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
# pto.tassign %arg2, @tile(0x3000)
%dst = pto.tsort32 %src, %idx : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tsort32 %src, %idx : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tsort32 ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
