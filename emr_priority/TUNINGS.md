# OpenMP tunings of the priority Apps kernels

This page describes the OpenMP variant tunings that the vectorized branch adds on top
of `develop` (`d3ea45cc`). The Seq, GPU and SYCL variants are unchanged. Select a tuning
with RAJAPerf's `--tunings <name>` option; in result tables a row is labeled
`<Variant>/<tuning>`, and the bare variant name (for example `Base_OpenMP`) is the
`default` tuning.

Reports compare every tuning against the **`develop` default tuning** of the same
variant. Measurements are in [size_sweep/SIZE_SWEEP.md](size_sweep/SIZE_SWEEP.md).

## Common building blocks

| Term | Meaning |
|---|---|
| SoA batch | `FEM_VL` elements (8 in double precision, one 512-bit vector) are gathered from the AoS layout into element-innermost arrays, computed lane by lane, and scattered back. Tail lanes repeat the last valid element; only valid lanes are written. |
| AVX-512 path | Explicit intrinsics from `src/apps/FEM_SIMD.hpp` (forced FMA, 8x8 in-register transposes). Enabled when `__AVX512F__`, `__AVX512DQ__` and double precision are present, unless `-DRAJAPERF_FEM_NO_INTRINSICS` is set. Otherwise the code falls back to `omp simd` lane loops. |
| `_nt` | Streaming (non-temporal) stores: output is staged in an L1 tile and written with `_mm512_stream_pd`, bypassing the caches. This pays off only when the output does not fit in the LLC. **AVX-512 only.** |
| `pfN` / `_pfN` | Software prefetch of the input data N batches (or elements) ahead. |
| `_x2` / `_x4` | 2 or 4 vectors per value (16 or 32 elements per batch) for more instruction-level parallelism. **AVX-512 only.** |

Tunings marked **AVX-512 only** below are registered only when the AVX-512 path is
compiled in.

## Kernels

### EDGE3D (Base, Lambda, RAJA)

| Tuning | Description |
|---|---|
| `default` | Original algorithm on SoA batches with the AVX-512 batch kernel. |
| `metric` | Reformulated with the metric tensors G = JᵀJ / detJ and H = J⁻¹J⁻ᵀ · detJ. Each matrix entry becomes 3 FMAs; quadrature loops are fully unrolled. Gives the same result with fewer FLOPs than RAJAPerf counts. |
| `metric_rb` | `metric` computed as a GEMM-like, register-blocked contraction. Phase 1 computes the 12 metric terms per quadrature point; phase 2 accumulates 4 x 4 tiles of the upper triangle in registers over all quadrature points, so the element matrix no longer goes through memory once per point. Element-independent quadrature data is built once per run (`Edge3DQpTable`). |
| `metric_x2`, `metric_x4` | `metric` with 16 or 32 elements per batch. **AVX-512 only.** |
| `metric_rb_x2`, `metric_rb_x4` | `metric_rb` with 16 or 32 elements per batch. **AVX-512 only.** |

### DIFFUSION3DPA (Base, RAJA)

| Tuning | Description |
|---|---|
| `default` | All sum-factorization stages on SoA batches of elements (`omp simd`). |
| `transpose` | AVX-512 batches; X, D and Y move between AoS and SoA through 8x8 in-register transposes instead of gathers/scatters, and the next batch's D, X and Y are prefetched. **AVX-512 only.** |

### ENERGY (Base, Lambda, RAJA)

| Tuning | Description |
|---|---|
| `default` | Unchanged reference code (six separate loops). |
| `fused` | All six loops fused into one pass per element. Bit-exact against the reference; moves fewer bytes than RAJAPerf counts, so its GB/s is overstated. |
| `fused_nt` | `fused` with results staged in a 256-entry L1 tile and written with streaming stores. **AVX-512 only.** |
| `fused_skip` | `fused` with the `delvc <= 0` work skipped where no element needs it. That branch reads 6 more arrays and needs a division and two square roots, and the compiler's if-converted vector loop executes it for every element. The AVX-512 path computes all six bodies in registers, 8 elements at a time, and enters the branch only when the lane mask is non-empty. The portable path checks each 256-element tile and runs a branch-free loop when possible. For the suite's inputs (`delvc > 0` everywhere) it reads 7 arrays instead of 13. The branch is checked against the six scalar bodies in a standalone test with mixed-sign `delvc`. |
| `fused_skip_nt` | `fused_skip` writing `e_new` and `q_new` from registers with streaming stores. **AVX-512 only.** |

### FEMSWEEP (Base, RAJA)

All tunings batch 8 energy groups per SIMD vector, use an in-place Crout LU solve with
one reciprocal per diagonal, and run each (angle, group batch) sweep on a team of threads
that synchronizes through a sense-reversing barrier.

| Tuning | Description |
|---|---|
| `default` | Reference SoA sweep. |
| `prefetch` | Per-element routine `femsweep_element` with software prefetch 2 elements ahead. |
| `prefetch_pf1`, `prefetch_pf4` | `prefetch` with a distance of 1 or 4 elements. |
| `prefetch_mixed` | `prefetch`, but the last, partial round of tasks is run by larger teams so all threads stay busy. |
| `prefetch_mixed_ft` | `prefetch_mixed` reading face data (interior face/side, upwind element, local dofs) from a 16-byte-per-face table built once in `setUp`, instead of the dependent index chain `elem_to_faces -> F_g2l -> idx1/idx2`. Falls back to the index chain if the mesh does not fit the table. |

### INTSC_HEXRECT (Base, Lambda, RAJA)

| Tuning | Description |
|---|---|
| `default` | Unchanged reference: each donor facet is clipped separately against each of the 8 target boxes. |
| `split` | Each donor hex is processed once against its 2x2x2 targets; facets are split by the shared x/y/z planes. Less work than RAJAPerf counts. |
| `split_lean` | `split` clipping into ping-pong buffers that write only the kept side, and integrating without staging copies. |
| `split_simd` | 8 donor zones per batch, one per AVX-512 lane, on one shared control path. When the lanes disagree on a clipping decision, the facet is recomputed lane by lane with the scalar `split_lean` code. **AVX-512 only.** |

### MASS3DEA (Base, RAJA)

| Tuning | Description |
|---|---|
| `default` | Register-blocked AVX-512 path producing contiguous 64-entry blocks; parallel over (element batch, j-block). |
| `sumfact` | Full sum factorization of the element matrix (T3 -> T2 -> 64-entry blocks), about 17x fewer FLOPs than the reference. Its GFLOP/s are therefore far above hardware peak. |
| `sumfact_nt` | `sumfact` with a transpose and streaming stores of the output. **AVX-512 only.** |
| `sumfact_nt_seq` | `sumfact_nt`, but the batch's (element x j3) slices are staged in a 64 KB buffer, and each element's contiguous 8 KB slice is then streamed in one go: one sequential write stream per thread instead of 8 interleaved ones. **AVX-512 only.** |

### MASS3DPA_ATOMIC (Base, RAJA)

All tunings remove the atomics: (y, z) element rows are grouped into 4x4 blocks colored
like a checkerboard (4 colors), so blocks of one color share no dofs. Each SIMD batch is
scattered in two passes (even lanes, then odd lanes).

| Tuning | Description |
|---|---|
| `default` | AVX-512 batch kernel (gather, transpose, masked scatter) without prefetch. |
| `pf1`, `pf2`, `pf4` | `default` with software prefetch of D and ElemToDoF 1, 2 or 4 batches ahead. **AVX-512 only.** |
| `row` | Whole x-rows per thread in 4 row colors; 8 consecutive elements of a row per batch (lane = element). On the structured mesh each node row of a batch is 17 contiguous dofs, so X is loaded with plain loads + even/odd permutes and Y is updated with plain read-modify-writes, the dof shared by neighbouring lanes summed in registers: no gathers, scatters or atomics. `setUp` verifies the structured form of ElemToDoF; otherwise the default path runs. **AVX-512 only.** |
| `row_pf` | `row` with the X and Y node rows of the next block prefetched. **AVX-512 only.** |

### MASSVEC3DPA (Base, RAJA)

| Tuning | Description |
|---|---|
| `default` | AVX-512 batch kernel. |
| `nt` | The batch's contiguous 8 x 81 Y block is staged and written with streaming stores when it is full and aligned. **AVX-512 only.** |
| `nt_pf1`, `nt_pf2`, `nt_pf4` | `nt` with software prefetch 1, 2 or 4 batches ahead. **AVX-512 only.** |

### NODAL_ACCUMULATION_3D (Base, Lambda, RAJA)

| Tuning | Description |
|---|---|
| `default` | Replaces the zone scatter with atomics by a node gather: each node pulls 0.125 · vol from its up to 8 adjacent zones, parallel over (j, k) node rows with a SIMD loop over i. There are no atomics and no write conflicts, and the result is bit-exact against Base_Seq. |
| `blocked` | `default` with cache blocking: work units (about one per thread) are (block of 64 node rows in j) x (slab of k-planes) and walk the planes of a slab in order, so the vol rows of plane k-1 are reused from L2 instead of being read again from memory. |

### VOL3D (Base, Lambda, RAJA)

| Tuning | Description |
|---|---|
| `default` | Unchanged reference loop (kept as a plain member function so the lambda loop is inlined and vectorized). |
| `blocked` | Cache blocking: work units are (tile of rows) x (slab of k-planes), walking the planes of a slab innermost, so the k+1 plane rows of x, y, z are reused from L2. |
| `nt` | Each thread computes one 64-byte-aligned contiguous chunk of vol in 256-entry L1 tiles that are streamed to memory. **AVX-512 only.** |
| `strip` | Plane strips: thread s owns the same contiguous range of offsets in every k-plane and walks the planes in order, so x, y, z of plane k+1 are still in L2 at the next k. |
| `strip_nt` | `strip` with vol written by streaming stores from 256-entry L1 tiles (blocking and streaming stores combined). **AVX-512 only.** |

## Which tuning to pick

Best tuning per problem-size regime on a 64-core Xeon 8592+ (see
[size_sweep/SIZE_SWEEP.md](size_sweep/SIZE_SWEEP.md); the `fused_skip*`, `strip*`,
NODAL `blocked`, `sumfact_nt_seq` and MASS3DPA_ATOMIC `row*` tunings were measured later at the same sizes,
see [compare_other/COMPARISON.md](compare_other/COMPARISON.md)). `fit` is a working set of about
0.25x LLC; `exceed` is about 8x LLC.

| Kernel | in cache (default / fit) | out of cache (exceed) |
|---|---|---|
| EDGE3D | `metric_rb` | `metric_rb` |
| DIFFUSION3DPA | `default` | `transpose` |
| ENERGY | `fused_skip` | `fused_skip` / `fused_skip_nt` |
| FEMSWEEP | `prefetch_mixed_ft` | `prefetch_mixed_ft` |
| INTSC_HEXRECT | `split_simd` | `split_simd` |
| MASS3DEA | `sumfact` | `sumfact_nt_seq` |
| MASS3DPA_ATOMIC | `row_pf` | `row_pf` |
| MASSVEC3DPA | `default` | `nt_pf1` |
| NODAL_ACCUMULATION_3D | `default` | `blocked` |
| VOL3D | `default` / `strip` | `strip_nt` |

Streaming-store (`_nt`) and cache-blocking (`blocked`, `strip`) tunings are slower when the data
fits in the LLC and only pay off at the `exceed` size.
