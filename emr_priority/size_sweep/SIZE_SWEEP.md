# Problem-size sweep: develop vs vectorized

LLC = 320 MiB. OMP_NUM_THREADS=64; median of 3 interleaved runs x 3 passes. Speedup = t/rep of the develop default tuning divided by t/rep of the vectorized variant/tuning (>1 is faster). GB/s uses RAJAPerf's nominal byte counts: values above the measured DRAM peak mean the data was served from cache, not that DRAM was beaten. Per-regime full tables: `<regime>/summary.md`.

```
pcl-sprh01.sc.intel.com
CPU(s):                                  256
Model name:                              INTEL(R) XEON(R) PLATINUM 8592+
CPU(s) scaling MHz:                      29%
NUMA node0 CPU(s):                       0-63,128-191
NUMA node1 CPU(s):                       64-127,192-255
```

Slurm job 392170 on a dedicated (`--exclusive`) EMR node, 64 threads pinned to socket 0
(GOMP_CPU_AFFINITY=0-63). develop = `d3ea45cc`, vectorized = `60cb8ce8` (only
`run_emr_priority.sh` modified). clang 21.1.8, `-O3 -march=sapphirerapids
-mprefer-vector-width=512 -ffast-math`. All checksums PASSED in all three regimes
(Base_Seq reference, plus a develop-vs-vectorized cross check).

## Key findings

1. **Every suite-default size fits in the 320 MiB LLC** (0.01x to 0.36x). The default
   numbers are therefore cache numbers, like the `fit` column, not DRAM numbers.
2. **The speedups hold up out of cache.** The geomean of the best tuning per kernel is
   4.6x at the default size, 6.4x at `fit` and 4.8x at `exceed`. Compute- or
   latency-bound kernels keep or increase their gain at 8x LLC: EDGE3D 18x,
   FEMSWEEP 11.5x, INTSC_HEXRECT 11.4x and MASS3DEA 12.3x.
3. **Memory-bound speedups shrink once data leaves the LLC**, which is the expected
   signature: NODAL_ACCUMULATION_3D falls from 36x to 8.4x, MASSVEC3DPA from 4.7x
   to 2.1x and DIFFUSION3DPA from 2.4x to 1.5x. Their vectorized GB/s at `exceed`
   (290 to 305) is at or slightly above the ~265 GB/s socket DRAM peak measured in
   round 6. That peak uses nominal byte counts, so these kernels are now
   DRAM-bandwidth-bound.
4. **Streaming stores behave as SKILL.md predicts.** MASS3DEA `sumfact_nt` is 4.5x
   *slower* than `sumfact` when the output fits in the LLC (121 vs 545 GB/s) but is
   the fastest tuning at 8x LLC (176 vs 166 GB/s). MASSVEC3DPA `nt_pf*` wins only at
   `exceed`. ENERGY `fused_nt` is about equal to `fused` at `exceed` and slower in
   cache.
5. **Cache blocking only pays off out of cache.** VOL3D `blocked` is 0.64x to 0.81x
   in cache but 1.54x at 8x LLC (develop drops from 900 to 162 GB/s; `blocked`
   reaches 249 GB/s). The untuned VOL3D variants are unchanged (about 1.0x) in
   every regime.
6. **Default sizes that are too small understate the gain.** MASS3DEA at its 4 MiB
   default runs for 87 us and reaches only 4.1x; at 80 MiB `sumfact` reaches 38x.
7. **ENERGY's nominal GB/s exceeds the DRAM peak (542 GB/s) at `exceed`.** Its
   per-loop byte counts double-count arrays that `fused` reuses across loops, so
   compare ENERGY by time (2.77x), not by GB/s.

Best tuning per regime (Base/Lambda/RAJA give equivalent results; one shown):

| Kernel | default | fit | exceed |
|---|---|---|---|
| EDGE3D | metric_rb | metric_rb | metric_rb |
| DIFFUSION3DPA | default | default | transpose |
| ENERGY | fused | fused | fused / fused_nt |
| FEMSWEEP | prefetch_mixed_ft | prefetch_mixed_ft | prefetch_mixed_ft |
| INTSC_HEXRECT | split_simd | split_simd | split_simd |
| MASS3DEA | sumfact | sumfact | sumfact_nt |
| MASS3DPA_ATOMIC | pf1 | default | pf1 |
| MASSVEC3DPA | default | default | nt_pf1 |
| NODAL_ACCUMULATION_3D | default | default | default |
| VOL3D | default | default | blocked |

## Regimes

| Kernel | default size | default WS MiB (x LLC) | fit size | fit WS MiB (x LLC) | exceed size | exceed WS MiB (x LLC) |
|---|---:|---:|---:|---:|---:|---:|
| EDGE3D | 1000000 | 34.3 (0.11x) | 2406104 | 80.2 (0.25x) | 81746504 | 2564.3 (8.01x) |
| DIFFUSION3DPA | 421875 | 52.2 (0.16x) | 646380 | 80.0 (0.25x) | 20684241 | 2560.0 (8.00x) |
| ENERGY | 1000000 | 114.4 (0.36x) | 699051 | 80.0 (0.25x) | 22369621 | 2560.0 (8.00x) |
| FEMSWEEP | 144000 | 19.6 (0.06x) | 589824 | 82.3 (0.26x) | 18000000 | 2582.6 (8.07x) |
| INTSC_HEXRECT | 1000000 | 45.0 (0.14x) | 1728000 | 77.7 (0.24x) | 56623104 | 2540.5 (7.94x) |
| MASS3DEA | 512000 | 4.0 (0.01x) | 10174464 | 80.0 (0.25x) | 325607424 | 2560.0 (8.00x) |
| MASS3DPA_ATOMIC | 1030301 | 102.5 (0.32x) | 804357 | 79.9 (0.25x) | 25153757 | 2544.5 (7.95x) |
| MASSVEC3DPA | 421848 | 9.0 (0.03x) | 3758157 | 80.0 (0.25x) | 120261429 | 2560.0 (8.00x) |
| NODAL_ACCUMULATION_3D | 1000000 | 24.7 (0.08x) | 3307949 | 79.7 (0.25x) | 109902239 | 2556.0 (7.99x) |
| VOL3D | 1000000 | 34.3 (0.11x) | 2406104 | 80.2 (0.25x) | 81746504 | 2564.3 (8.01x) |

**default**: suite default size; **fit**: working set ~0.25 x LLC (cache resident); **exceed**: working set ~8 x LLC (DRAM bound)

## Best-tuning speedup per regime

| Kernel | default | fit | exceed |
|---|---:|---:|---:|
| EDGE3D | 19.89x | 19.93x | 17.94x |
| DIFFUSION3DPA | 2.32x | 2.38x | 1.47x |
| ENERGY | 1.57x | 1.80x | 2.77x |
| FEMSWEEP | 8.41x | 10.50x | 11.54x |
| INTSC_HEXRECT | 7.62x | 8.05x | 11.36x |
| MASS3DEA | 4.09x | 38.35x | 12.34x |
| MASS3DPA_ATOMIC | 2.54x | 2.55x | 2.00x |
| MASSVEC3DPA | 3.07x | 4.68x | 2.06x |
| NODAL_ACCUMULATION_3D | 29.35x | 36.44x | 8.43x |
| VOL3D | 1.02x | 1.00x | 1.54x |
| **geomean** | **4.62x** | **6.43x** | **4.79x** |

## Per-kernel results (every variant and tuning)

### EDGE3D

| Variant | default: develop GB/s | default: vectorized GB/s | default: speedup | fit: develop GB/s | fit: vectorized GB/s | fit: speedup | exceed: develop GB/s | exceed: vectorized GB/s | exceed: speedup | Checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| Base_OpenMP | 0.9 | 5.9 | 6.64x | 0.9 | 5.9 | 6.65x | 0.9 | 5.6 | 6.40x | PASS |
| Base_OpenMP/metric | 0.9 | 13.9 | 15.73x | 0.9 | 13.2 | 14.96x | 0.9 | 12.2 | 14.03x | PASS |
| Base_OpenMP/metric_rb | 0.9 | 16.8 | 18.98x | 0.9 | 16.5 | 18.76x | 0.9 | 15.0 | 17.34x | PASS |
| Base_OpenMP/metric_rb_x2 | 0.9 | 15.1 | 17.15x | 0.9 | 14.7 | 16.69x | 0.9 | 14.3 | 16.45x | PASS |
| Base_OpenMP/metric_rb_x4 | 0.9 | 11.8 | 13.41x | 0.9 | 12.2 | 13.82x | 0.9 | 11.5 | 13.31x | PASS |
| Base_OpenMP/metric_x2 | 0.9 | 12.6 | 14.32x | 0.9 | 12.8 | 14.55x | 0.9 | 12.2 | 14.04x | PASS |
| Base_OpenMP/metric_x4 | 0.9 | 12.1 | 13.75x | 0.9 | 11.4 | 13.00x | 0.9 | 11.7 | 13.53x | PASS |
| Lambda_OpenMP | 0.8 | 5.7 | 6.73x | 0.8 | 5.8 | 6.95x | 0.8 | 5.4 | 6.43x | PASS |
| Lambda_OpenMP/metric | 0.8 | 13.7 | 16.24x | 0.8 | 13.1 | 15.62x | 0.8 | 12.2 | 14.62x | PASS |
| Lambda_OpenMP/metric_rb | 0.8 | 16.7 | 19.89x | 0.8 | 16.7 | 19.93x | 0.8 | 15.0 | 17.94x | PASS |
| Lambda_OpenMP/metric_rb_x2 | 0.8 | 14.9 | 17.72x | 0.8 | 15.3 | 18.24x | 0.8 | 14.4 | 17.17x | PASS |
| Lambda_OpenMP/metric_rb_x4 | 0.8 | 12.3 | 14.60x | 0.8 | 11.9 | 14.28x | 0.8 | 11.6 | 13.87x | PASS |
| Lambda_OpenMP/metric_x2 | 0.8 | 12.4 | 14.74x | 0.8 | 13.0 | 15.54x | 0.8 | 12.2 | 14.62x | PASS |
| Lambda_OpenMP/metric_x4 | 0.8 | 12.1 | 14.43x | 0.8 | 11.5 | 13.79x | 0.8 | 11.5 | 13.76x | PASS |
| RAJA_OpenMP | 0.8 | 5.6 | 6.65x | 0.8 | 5.8 | 6.96x | 0.8 | 5.4 | 6.43x | PASS |
| RAJA_OpenMP/metric | 0.8 | 13.6 | 16.15x | 0.8 | 13.4 | 16.00x | 0.8 | 12.1 | 14.52x | PASS |
| RAJA_OpenMP/metric_rb | 0.8 | 16.7 | 19.83x | 0.8 | 16.3 | 19.43x | 0.8 | 14.9 | 17.85x | PASS |
| RAJA_OpenMP/metric_rb_x2 | 0.8 | 14.6 | 17.38x | 0.8 | 15.2 | 18.09x | 0.8 | 14.4 | 17.23x | PASS |
| RAJA_OpenMP/metric_rb_x4 | 0.8 | 12.3 | 14.65x | 0.8 | 12.2 | 14.52x | 0.8 | 11.7 | 13.99x | PASS |
| RAJA_OpenMP/metric_x2 | 0.8 | 12.5 | 14.85x | 0.8 | 12.7 | 15.12x | 0.8 | 12.4 | 14.82x | PASS |
| RAJA_OpenMP/metric_x4 | 0.8 | 11.7 | 13.94x | 0.8 | 11.9 | 14.14x | 0.8 | 11.5 | 13.74x | PASS |

### DIFFUSION3DPA

| Variant | default: develop GB/s | default: vectorized GB/s | default: speedup | fit: develop GB/s | fit: vectorized GB/s | fit: speedup | exceed: develop GB/s | exceed: vectorized GB/s | exceed: speedup | Checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| Base_OpenMP | 422.4 | 978.5 | 2.32x | 427.0 | 1014.9 | 2.38x | 221.3 | 290.7 | 1.31x | PASS |
| Base_OpenMP/transpose | 422.4 | 673.1 | 1.59x | 427.0 | 689.1 | 1.61x | 221.3 | 302.5 | 1.37x | PASS |
| RAJA_OpenMP | 409.2 | 517.0 | 1.26x | 415.5 | 533.6 | 1.28x | 206.5 | 262.6 | 1.27x | PASS |
| RAJA_OpenMP/transpose | 409.2 | 652.6 | 1.59x | 415.5 | 681.2 | 1.64x | 206.5 | 302.7 | 1.47x | PASS |

### ENERGY

| Variant | default: develop GB/s | default: vectorized GB/s | default: speedup | fit: develop GB/s | fit: vectorized GB/s | fit: speedup | exceed: develop GB/s | exceed: vectorized GB/s | exceed: speedup | Checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| Base_OpenMP | 1810.0 | 1819.8 | 1.01x | 1490.3 | 1502.9 | 1.01x | 195.5 | 195.9 | 1.00x | PASS |
| Base_OpenMP/fused | 1810.0 | 2533.4 | 1.40x | 1490.3 | 2339.2 | 1.57x | 195.5 | 542.0 | 2.77x | PASS |
| Base_OpenMP/fused_nt | 1810.0 | 2398.4 | 1.33x | 1490.3 | 2159.2 | 1.45x | 195.5 | 541.4 | 2.77x | PASS |
| Lambda_OpenMP | 1614.1 | 928.3 | 0.58x | 1302.2 | 1303.9 | 1.00x | 196.1 | 196.7 | 1.00x | PASS |
| Lambda_OpenMP/fused | 1614.1 | 2533.0 | 1.57x | 1302.2 | 2344.9 | 1.80x | 196.1 | 542.0 | 2.76x | PASS |
| Lambda_OpenMP/fused_nt | 1614.1 | 2397.5 | 1.49x | 1302.2 | 2169.3 | 1.67x | 196.1 | 541.3 | 2.76x | PASS |
| RAJA_OpenMP | 1954.2 | 1938.9 | 0.99x | 1514.9 | 1524.0 | 1.01x | 195.9 | 196.6 | 1.00x | PASS |
| RAJA_OpenMP/fused | 1954.2 | 2566.4 | 1.31x | 1514.9 | 2408.8 | 1.59x | 195.9 | 542.3 | 2.77x | PASS |
| RAJA_OpenMP/fused_nt | 1954.2 | 2412.6 | 1.23x | 1514.9 | 2231.4 | 1.47x | 195.9 | 541.2 | 2.76x | PASS |

### FEMSWEEP

| Variant | default: develop GB/s | default: vectorized GB/s | default: speedup | fit: develop GB/s | fit: vectorized GB/s | fit: speedup | exceed: develop GB/s | exceed: vectorized GB/s | exceed: speedup | Checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| Base_OpenMP | 23.9 | 88.4 | 3.70x | 24.4 | 110.5 | 4.53x | 16.9 | 97.0 | 5.73x | PASS |
| Base_OpenMP/prefetch | 23.9 | 186.1 | 7.80x | 24.4 | 211.7 | 8.68x | 16.9 | 153.1 | 9.05x | PASS |
| Base_OpenMP/prefetch_mixed | 23.9 | 191.2 | 8.01x | 24.4 | 245.3 | 10.05x | 16.9 | 183.6 | 10.86x | PASS |
| Base_OpenMP/prefetch_mixed_ft | 23.9 | 200.8 | 8.41x | 24.4 | 256.2 | 10.50x | 16.9 | 195.2 | 11.54x | PASS |
| Base_OpenMP/prefetch_pf1 | 23.9 | 188.5 | 7.90x | 24.4 | 210.5 | 8.63x | 16.9 | 152.0 | 8.99x | PASS |
| Base_OpenMP/prefetch_pf4 | 23.9 | 183.6 | 7.69x | 24.4 | 210.3 | 8.62x | 16.9 | 149.7 | 8.85x | PASS |
| RAJA_OpenMP | 24.6 | 84.6 | 3.43x | 25.2 | 106.1 | 4.21x | 17.8 | 96.5 | 5.44x | PASS |
| RAJA_OpenMP/prefetch | 24.6 | 181.1 | 7.35x | 25.2 | 207.5 | 8.23x | 17.8 | 151.6 | 8.54x | PASS |
| RAJA_OpenMP/prefetch_mixed | 24.6 | 183.5 | 7.45x | 25.2 | 242.5 | 9.62x | 17.8 | 182.3 | 10.27x | PASS |
| RAJA_OpenMP/prefetch_mixed_ft | 24.6 | 191.2 | 7.76x | 25.2 | 256.8 | 10.19x | 17.8 | 197.2 | 11.11x | PASS |
| RAJA_OpenMP/prefetch_pf1 | 24.6 | 155.6 | 6.32x | 25.2 | 207.4 | 8.23x | 17.8 | 152.4 | 8.58x | PASS |
| RAJA_OpenMP/prefetch_pf4 | 24.6 | 179.2 | 7.27x | 25.2 | 207.4 | 8.23x | 17.8 | 149.5 | 8.42x | PASS |

### INTSC_HEXRECT

| Variant | default: develop GB/s | default: vectorized GB/s | default: speedup | fit: develop GB/s | fit: vectorized GB/s | fit: speedup | exceed: develop GB/s | exceed: vectorized GB/s | exceed: speedup | Checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| Base_OpenMP | 2.8 | 2.9 | 1.03x | 2.8 | 2.8 | 1.00x | 2.6 | 2.7 | 1.03x | PASS |
| Base_OpenMP/split | 2.8 | 6.4 | 2.26x | 2.8 | 6.6 | 2.38x | 2.6 | 6.7 | 2.54x | PASS |
| Base_OpenMP/split_lean | 2.8 | 7.6 | 2.68x | 2.8 | 7.6 | 2.76x | 2.6 | 7.8 | 2.96x | PASS |
| Base_OpenMP/split_simd | 2.8 | 21.6 | 7.62x | 2.8 | 22.2 | 8.05x | 2.6 | 30.0 | 11.36x | PASS |
| Lambda_OpenMP | 3.1 | 3.1 | 0.99x | 3.0 | 3.0 | 1.00x | 2.9 | 2.9 | 1.01x | PASS |
| Lambda_OpenMP/split | 3.1 | 6.3 | 2.01x | 3.0 | 6.4 | 2.15x | 2.9 | 6.4 | 2.23x | PASS |
| Lambda_OpenMP/split_lean | 3.1 | 7.5 | 2.38x | 3.0 | 7.5 | 2.53x | 2.9 | 7.6 | 2.67x | PASS |
| Lambda_OpenMP/split_simd | 3.1 | 21.8 | 6.95x | 3.0 | 22.0 | 7.42x | 2.9 | 30.3 | 10.57x | PASS |
| RAJA_OpenMP | 3.1 | 3.1 | 1.00x | 2.9 | 3.0 | 1.01x | 2.8 | 2.9 | 1.01x | PASS |
| RAJA_OpenMP/split | 3.1 | 6.5 | 2.10x | 2.9 | 6.5 | 2.22x | 2.8 | 6.7 | 2.34x | PASS |
| RAJA_OpenMP/split_lean | 3.1 | 7.4 | 2.40x | 2.9 | 7.5 | 2.56x | 2.8 | 7.7 | 2.69x | PASS |
| RAJA_OpenMP/split_simd | 3.1 | 21.6 | 6.97x | 2.9 | 22.0 | 7.47x | 2.8 | 30.2 | 10.61x | PASS |

### MASS3DEA

| Variant | default: develop GB/s | default: vectorized GB/s | default: speedup | fit: develop GB/s | fit: vectorized GB/s | fit: speedup | exceed: develop GB/s | exceed: vectorized GB/s | exceed: speedup | Checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| Base_OpenMP | 11.9 | 42.8 | 3.61x | 14.2 | 114.2 | 8.03x | 14.3 | 103.5 | 7.26x | PASS |
| Base_OpenMP/sumfact | 11.9 | 48.5 | 4.09x | 14.2 | 545.3 | 38.35x | 14.3 | 166.0 | 11.64x | PASS |
| Base_OpenMP/sumfact_nt | 11.9 | 44.4 | 3.74x | 14.2 | 120.9 | 8.50x | 14.3 | 176.0 | 12.34x | PASS |
| RAJA_OpenMP | 15.3 | 40.7 | 2.66x | 19.9 | 89.8 | 4.52x | 19.9 | 71.6 | 3.60x | PASS |
| RAJA_OpenMP/sumfact | 15.3 | 48.9 | 3.20x | 19.9 | 559.5 | 28.16x | 19.9 | 166.1 | 8.36x | PASS |
| RAJA_OpenMP/sumfact_nt | 15.3 | 44.1 | 2.88x | 19.9 | 122.7 | 6.18x | 19.9 | 176.2 | 8.87x | PASS |

### MASS3DPA_ATOMIC

| Variant | default: develop GB/s | default: vectorized GB/s | default: speedup | fit: develop GB/s | fit: vectorized GB/s | fit: speedup | exceed: develop GB/s | exceed: vectorized GB/s | exceed: speedup | Checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| Base_OpenMP | 134.4 | 334.9 | 2.49x | 140.0 | 300.6 | 2.15x | 136.1 | 266.5 | 1.96x | PASS |
| Base_OpenMP/pf1 | 134.4 | 341.3 | 2.54x | 140.0 | 299.9 | 2.14x | 136.1 | 272.6 | 2.00x | PASS |
| Base_OpenMP/pf2 | 134.4 | 328.5 | 2.44x | 140.0 | 291.7 | 2.08x | 136.1 | 268.3 | 1.97x | PASS |
| Base_OpenMP/pf4 | 134.4 | 319.9 | 2.38x | 140.0 | 283.6 | 2.03x | 136.1 | 262.8 | 1.93x | PASS |
| RAJA_OpenMP | 143.9 | 360.1 | 2.50x | 135.0 | 344.7 | 2.55x | 139.5 | 265.4 | 1.90x | PASS |
| RAJA_OpenMP/pf1 | 143.9 | 340.6 | 2.37x | 135.0 | 332.0 | 2.46x | 139.5 | 269.1 | 1.93x | PASS |
| RAJA_OpenMP/pf2 | 143.9 | 326.8 | 2.27x | 135.0 | 320.7 | 2.38x | 139.5 | 265.7 | 1.91x | PASS |
| RAJA_OpenMP/pf4 | 143.9 | 309.6 | 2.15x | 135.0 | 306.6 | 2.27x | 139.5 | 261.0 | 1.87x | PASS |

### MASSVEC3DPA

| Variant | default: develop GB/s | default: vectorized GB/s | default: speedup | fit: develop GB/s | fit: vectorized GB/s | fit: speedup | exceed: develop GB/s | exceed: vectorized GB/s | exceed: speedup | Checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| Base_OpenMP | 142.9 | 438.9 | 3.07x | 157.3 | 736.7 | 4.68x | 140.6 | 214.2 | 1.52x | PASS |
| Base_OpenMP/nt | 142.9 | 386.7 | 2.71x | 157.3 | 475.6 | 3.02x | 140.6 | 277.3 | 1.97x | PASS |
| Base_OpenMP/nt_pf1 | 142.9 | 387.1 | 2.71x | 157.3 | 464.0 | 2.95x | 140.6 | 290.1 | 2.06x | PASS |
| Base_OpenMP/nt_pf2 | 142.9 | 386.7 | 2.71x | 157.3 | 465.8 | 2.96x | 140.6 | 289.8 | 2.06x | PASS |
| Base_OpenMP/nt_pf4 | 142.9 | 386.9 | 2.71x | 157.3 | 466.7 | 2.97x | 140.6 | 289.3 | 2.06x | PASS |
| RAJA_OpenMP | 138.1 | 379.0 | 2.74x | 159.1 | 731.1 | 4.59x | 150.4 | 214.6 | 1.43x | PASS |
| RAJA_OpenMP/nt | 138.1 | 392.2 | 2.84x | 159.1 | 477.8 | 3.00x | 150.4 | 276.9 | 1.84x | PASS |
| RAJA_OpenMP/nt_pf1 | 138.1 | 389.4 | 2.82x | 159.1 | 484.9 | 3.05x | 150.4 | 290.1 | 1.93x | PASS |
| RAJA_OpenMP/nt_pf2 | 138.1 | 380.4 | 2.75x | 159.1 | 467.9 | 2.94x | 150.4 | 289.4 | 1.92x | PASS |
| RAJA_OpenMP/nt_pf4 | 138.1 | 384.7 | 2.78x | 159.1 | 483.6 | 3.04x | 150.4 | 289.4 | 1.92x | PASS |

### NODAL_ACCUMULATION_3D

| Variant | default: develop GB/s | default: vectorized GB/s | default: speedup | fit: develop GB/s | fit: vectorized GB/s | fit: speedup | exceed: develop GB/s | exceed: vectorized GB/s | exceed: speedup | Checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| Base_OpenMP | 37.0 | 1086.9 | 29.35x | 37.0 | 1336.5 | 36.15x | 36.2 | 305.0 | 8.43x | PASS |
| Lambda_OpenMP | 37.1 | 1084.7 | 29.26x | 37.0 | 1347.7 | 36.44x | 36.2 | 305.2 | 8.43x | PASS |
| RAJA_OpenMP | 37.2 | 990.2 | 26.60x | 37.2 | 713.1 | 19.17x | 36.7 | 304.1 | 8.30x | PASS |

### VOL3D

| Variant | default: develop GB/s | default: vectorized GB/s | default: speedup | fit: develop GB/s | fit: vectorized GB/s | fit: speedup | exceed: develop GB/s | exceed: vectorized GB/s | exceed: speedup | Checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| Base_OpenMP | 816.3 | 817.9 | 1.00x | 900.3 | 900.9 | 1.00x | 161.8 | 161.8 | 1.00x | PASS |
| Base_OpenMP/blocked | 816.3 | 663.3 | 0.81x | 900.3 | 575.8 | 0.64x | 161.8 | 248.9 | 1.54x | PASS |
| Base_OpenMP/nt | 816.3 | 629.4 | 0.77x | 900.3 | 559.0 | 0.62x | 161.8 | 169.6 | 1.05x | PASS |
| Lambda_OpenMP | 818.8 | 811.3 | 0.99x | 884.1 | 888.0 | 1.00x | 161.4 | 161.6 | 1.00x | PASS |
| Lambda_OpenMP/blocked | 818.8 | 651.2 | 0.80x | 884.1 | 568.1 | 0.64x | 161.4 | 248.2 | 1.54x | PASS |
| Lambda_OpenMP/nt | 818.8 | 626.2 | 0.76x | 884.1 | 549.6 | 0.62x | 161.4 | 169.6 | 1.05x | PASS |
| RAJA_OpenMP | 762.7 | 779.5 | 1.02x | 897.8 | 876.6 | 0.98x | 161.9 | 162.0 | 1.00x | PASS |
| RAJA_OpenMP/blocked | 762.7 | 626.4 | 0.82x | 897.8 | 584.4 | 0.65x | 161.9 | 247.5 | 1.53x | PASS |
| RAJA_OpenMP/nt | 762.7 | 585.8 | 0.77x | 897.8 | 560.1 | 0.62x | 161.9 | 170.1 | 1.05x | PASS |

