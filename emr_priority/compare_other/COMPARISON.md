# Other agent's optimizations vs ours (EMR, 64 cores)

Comparison of two independent optimizations of the same 10 RAJAPerf Apps kernels:

- **other**: `RAJAPerf_other` (branch `DIFFUSION3DPA` at
  `41d0e580`, which contains all per-kernel branches). Based on upstream `e3c6197d`.
  One optimized implementation per OpenMP variant; no extra tunings.
- **ours**: the vectorized branch at `60cb8ce8` (same binary as
  [../size_sweep/SIZE_SWEEP.md](../size_sweep/SIZE_SWEEP.md)), based on upstream
  `d3ea45cc`, with the tunings described in [../TUNINGS.md](../TUNINGS.md).

## Setup

- Slurm job  (`--exclusive`, the same node as the develop-vs-ours
  sweep), Xeon Platinum 8592+, 64 threads pinned to socket 0 (`GOMP_CPU_AFFINITY=0-63`).
- Both trees built with clang 21.1.8,
  `-O3 -march=sapphirerapids -mprefer-vector-width=512 -ffast-math`. All AVX-512 paths of
  the other tree are enabled under clang. It was developed with icx and `-funroll-loops`,
  which were not used here.
- `run_emr_priority.sh sweep-run` with `REUSE_SIZES=1`: exactly the problem sizes of the
  develop-vs-ours sweep (`default`, `fit` ~0.25x LLC, `exceed` ~8x LLC). Median of 3
  interleaved runs x 3 passes per build.
- Correctness: every variant and tuning of both builds passes against its own Base_Seq
  reference in all three regimes. The builds use different inputs (ours randomizes
  them), so their checksums are not compared with each other.
- Ours was within noise of the earlier sweep (median time ratio 0.995 to 0.997).

### Comparing time, not GB/s or GFLOP/s

The two upstream bases differ in some nominal FLOP counts (DIFFUSION3DPA, MASS3DEA), so
compare by time per rep. Problem size, iterations and bytes per rep were checked to be
identical for 9 kernels in every regime. **MASSVEC3DPA is the exception**: upstream changed
its size-to-element mapping between the two bases, so at the same `--size` the other tree
processes 2.37x fewer elements (and bytes). Bytes per element are identical, so its time is
scaled by 2.370 to our problem (equal to comparing GB/s). The raw per-variant speedups in
[SIZE_SWEEP.md](SIZE_SWEEP.md) and `<regime>/summary.md` are **not** corrected for this.

## Result

Each tree at its fastest variant/tuning; other t/rep divided by ours (>1 = ours faster):

| Kernel | default | fit | exceed | Faster |
|---|---:|---:|---:|---|
| INTSC_HEXRECT | 3.72x | 4.00x | 4.89x | ours (`split_simd`) |
| EDGE3D | 3.33x | 3.78x | 3.15x | ours (`metric_rb`) |
| FEMSWEEP | 1.08x | 1.24x | 1.66x | ours (`prefetch_mixed_ft`) |
| DIFFUSION3DPA | 0.89x | 0.88x | 1.03x | other in cache, tie at DRAM |
| MASSVEC3DPA \* | 1.71x | 0.89x | 0.91x | ours at default, other otherwise |
| MASS3DEA | 1.10x | 4.17x | 0.86x | ours in cache (`sumfact`), other at DRAM |
| VOL3D | 1.24x | 1.43x | 0.87x | ours in cache, other at DRAM |
| NODAL_ACCUMULATION_3D | 1.35x | 1.34x | 0.78x | ours in cache, other at DRAM |
| MASS3DPA_ATOMIC | 0.61x | 0.62x | 0.84x | other |
| ENERGY | 0.44x | 0.48x | 0.67x | other |
| **geomean, ours fastest tuning** | **1.26x** | **1.43x** | **1.23x** | **ours** |
| geomean, ours `default` tuning only | 0.82x | 0.78x | 0.65x | other |

\* time scaled to the same number of elements (see above).

Full tables (fastest variant and tuning of each tree, times, run counts):
[COMPARISON_TABLES.md](COMPARISON_TABLES.md). Every variant and tuning, with GB/s and
GFLOP/s: [SIZE_SWEEP.md](SIZE_SWEEP.md).

## Findings

1. **Overall ours is faster (1.23x to 1.43x geomean), but only because of the extra
   tunings.** With our `default` tunings only, the other tree is faster in every regime
   (0.65x to 0.82x).
2. **Ours wins clearly where the algorithm was changed**: INTSC_HEXRECT `split_simd`
   (3.7x to 4.9x), EDGE3D `metric_rb` (3.2x to 3.8x) and FEMSWEEP `prefetch_mixed_ft`
   (1.1x to 1.7x). The other tree vectorizes the original algorithms of these kernels.
   MASS3DEA `sumfact` is 4.2x faster at `fit`, when its output stays in the LLC.
3. **The other tree reaches higher bandwidth out of cache.** At `exceed` it is faster on
   VOL3D (286 vs 248 GB/s), NODAL_ACCUMULATION_3D (391 vs 304), MASS3DPA_ATOMIC (323 vs
   271), MASS3DEA (205 vs 176) and MASSVEC3DPA (318 vs 290). These are nominal byte counts;
   DRAM peak measured with them is about 265 GB/s. According to its commits and skill
   files, the other tree uses plane-strip blocking plus streaming stores for VOL3D, a
   strip/halo decomposition with `j` blocking for NODAL_ACCUMULATION_3D, and streaming
   stores for MASS3DEA and MASSVEC3DPA. Ours has no streaming-store tuning for
   NODAL_ACCUMULATION_3D or MASS3DPA_ATOMIC, and our fastest VOL3D tuning at `exceed` is
   `blocked`.
4. **ENERGY is the other tree's biggest win** (2.1x to 2.3x in cache, 1.5x at `exceed`,
   804 vs 542 nominal GB/s): all six loops are fused and hand-vectorized with AVX-512 and
   streaming stores. Our `fused` is plain C++ left to the compiler, and `fused_nt` only
   adds streaming stores.
5. **MASS3DPA_ATOMIC and MASSVEC3DPA** are also faster in the other tree in cache (1.6x and
   1.1x per element); ours wins MASSVEC3DPA only at the small default size (1.7x).
6. **Next steps for ours:** port the other tree's ENERGY intrinsics, VOL3D/NODAL
   decompositions and MASS3DPA_ATOMIC/MASS3DEA store paths as additional tunings; the
   algorithmic wins in finding 2 do not overlap with these, so the two sets combine.

## Files

- `make_report.py`: writes `COMPARISON_TABLES.md` from this directory
  (`python3 make_report.py .`), including the bytes/rep normalization.
- `SIZE_SWEEP.md`, `<regime>/summary.{md,csv}`: generated by `run_emr_priority.sh`
  (`MAIN_LABEL=other VEC_LABEL=ours`); `main` = other, `vec` = ours.
- `<regime>/perf/{main,vec}/<kernel>/run1/RAJAPerf-kernel-details.csv`: problem
  definitions used for the normalization check.
- `sizes*.csv/txt`, `<regime>/{build,node}-info.txt`, `sweep-392590.log`: inputs and
  provenance. The remote work directory is
  `/data/nfs_home/aheineck/Projects/RAJA_workspace/emr_cmp_other`.

## Follow-up: closing findings 3 and 4

New tunings (see [../TUNINGS.md](../TUNINGS.md)), measured on the same node type,
compiler, flags and sizes. Both binaries ran interleaved, 3 runs x 3 passes, in Slurm
job 392909. Remote directory: `/data/nfs_home/aheineck/Projects/RAJA_workspace/emr_vec2`.
All checksums pass. Ratio = other's fastest variant / ours (>1 = ours faster):

| Kernel | Tuning | default | fit | exceed | Before (exceed) |
|---|---|---:|---:|---:|---:|
| ENERGY | `fused_skip` | 1.55x | 1.23x | 1.19x | 0.67x (`fused`) |
| ENERGY | `fused_skip_nt` | 0.50x | 0.55x | 1.20x | |
| VOL3D | `strip` | 1.23x | 1.20x | 0.94x | |
| VOL3D | `strip_nt` | 0.99x | 0.96x | 1.00x | 0.87x (`blocked`) |
| NODAL_ACCUMULATION_3D | `blocked` | 1.10x | 0.73x | 1.03x | 0.78x (`default`) |
| MASS3DEA | `sumfact_nt_seq` | 1.07x | 1.00x | 1.01x | 0.86x (`sumfact_nt`) |

- **ENERGY:** the other tree's lead did not come from streaming stores (its ENERGY has
  none). It came from hand-written AVX-512 that tests the rarely taken `delvc <= 0`
  branch for a whole vector. The suite's inputs never take that branch, so the 6 arrays,
  the division and the square roots it needs are skipped; our compiler-vectorized
  `fused` if-converted the branch and paid for all of it on every element. `fused_skip`
  reads 7 arrays instead of 13. It is now faster than the other tree in every regime.
- **VOL3D:** the other tree combines plane strips with streaming stores. We had blocking
  (`blocked`) and streaming stores (`nt`) only as separate tunings; `strip_nt` combines
  them.
- **NODAL_ACCUMULATION_3D:** cache blocking of our gather (64-row j blocks x k slabs,
  about one unit per thread) keeps plane k-1 in L2. With 16-row blocks the halo row
  re-reads left it 4% behind (0.96x).
- **MASS3DEA:** our `sumfact_nt` streamed 8 interleaved element rows per thread (lane =
  element). Staging the batch and streaming each element's 8 KB slice contiguously
  reaches the other tree's pure streaming-write rate.
