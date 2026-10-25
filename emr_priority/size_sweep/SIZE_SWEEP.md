# Problem-size sweep: develop vs vectorized

LLC = 320 MiB. OMP_NUM_THREADS=64; median of 3 interleaved runs x 3 passes. Speedup = t/rep of the develop default tuning divided by t/rep of the vectorized variant/tuning (>1 is faster). GB/s uses RAJAPerf's nominal byte counts: values above the measured DRAM peak mean the data was served from cache, not that DRAM was beaten. GFLOP/s likewise uses RAJAPerf's nominal per-kernel FLOP counts. Tunings are described in [TUNINGS.md](../TUNINGS.md). Per-regime full tables: `<regime>/summary.md`.

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
8. **Vectorized GFLOP/s for EDGE3D and MASS3DEA exceed the FP64 peak of one socket
   (~4 TFLOP/s).** EDGE3D `metric*` reaches up to 6.9 TFLOP/s and MASS3DEA `sumfact` up
   to 59 TFLOP/s. RAJAPerf's FLOP counts assume the original algorithm, but these
   tunings do less arithmetic (precomputed metric terms, sum factorization). Their
   GFLOP/s is therefore "effective" throughput; compare these kernels by time.

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

| Variant | default: develop t/rep | default: vectorized t/rep | default: develop GB/s | default: vectorized GB/s | default: develop GFLOP/s | default: vectorized GFLOP/s | default: speedup | fit: develop t/rep | fit: vectorized t/rep | fit: develop GB/s | fit: vectorized GB/s | fit: develop GFLOP/s | fit: vectorized GFLOP/s | fit: speedup | exceed: develop t/rep | exceed: vectorized t/rep | exceed: develop GB/s | exceed: vectorized GB/s | exceed: develop GFLOP/s | exceed: vectorized GFLOP/s | exceed: speedup | Checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| Base_OpenMP | 40.262 ms | 6.059 ms | 0.9 | 5.9 | 362.0 | 2405.4 | 6.64x | 94.648 ms | 14.237 ms | 0.9 | 5.9 | 361.6 | 2403.6 | 6.65x | 3090.500 ms | 482.734 ms | 0.9 | 5.6 | 357.6 | 2289.6 | 6.40x | PASS |
| Base_OpenMP/metric | 40.262 ms | 2.560 ms | 0.9 | 13.9 | 362.0 | 5693.6 | 15.73x | 94.648 ms | 6.327 ms | 0.9 | 13.2 | 361.6 | 5408.9 | 14.96x | 3090.500 ms | 220.341 ms | 0.9 | 12.2 | 357.6 | 5016.1 | 14.03x | PASS |
| Base_OpenMP/metric_rb | 40.262 ms | 2.122 ms | 0.9 | 16.8 | 362.0 | 6869.5 | 18.98x | 94.648 ms | 5.044 ms | 0.9 | 16.5 | 361.6 | 6784.4 | 18.76x | 3090.500 ms | 178.256 ms | 0.9 | 15.0 | 357.6 | 6200.4 | 17.34x | PASS |
| Base_OpenMP/metric_rb_x2 | 40.262 ms | 2.347 ms | 0.9 | 15.1 | 362.0 | 6209.3 | 17.15x | 94.648 ms | 5.673 ms | 0.9 | 14.7 | 361.6 | 6032.6 | 16.69x | 3090.500 ms | 187.827 ms | 0.9 | 14.3 | 357.6 | 5884.5 | 16.45x | PASS |
| Base_OpenMP/metric_rb_x4 | 40.262 ms | 3.001 ms | 0.9 | 11.8 | 362.0 | 4856.1 | 13.41x | 94.648 ms | 6.849 ms | 0.9 | 12.2 | 361.6 | 4996.2 | 13.82x | 3090.500 ms | 232.142 ms | 0.9 | 11.5 | 357.6 | 4761.1 | 13.31x | PASS |
| Base_OpenMP/metric_x2 | 40.262 ms | 2.811 ms | 0.9 | 12.6 | 362.0 | 5184.2 | 14.32x | 94.648 ms | 6.505 ms | 0.9 | 12.8 | 361.6 | 5260.7 | 14.55x | 3090.500 ms | 220.126 ms | 0.9 | 12.2 | 357.6 | 5021.0 | 14.04x | PASS |
| Base_OpenMP/metric_x4 | 40.262 ms | 2.928 ms | 0.9 | 12.1 | 362.0 | 4977.6 | 13.75x | 94.648 ms | 7.279 ms | 0.9 | 11.4 | 361.6 | 4700.9 | 13.00x | 3090.500 ms | 228.347 ms | 0.9 | 11.7 | 357.6 | 4840.3 | 13.53x | PASS |
| Lambda_OpenMP | 42.249 ms | 6.279 ms | 0.8 | 5.7 | 345.0 | 2321.2 | 6.73x | 99.597 ms | 14.328 ms | 0.8 | 5.8 | 343.6 | 2388.4 | 6.95x | 3208.030 ms | 498.708 ms | 0.8 | 5.4 | 344.5 | 2216.2 | 6.43x | PASS |
| Lambda_OpenMP/metric | 42.249 ms | 2.602 ms | 0.8 | 13.7 | 345.0 | 5602.5 | 16.24x | 99.597 ms | 6.378 ms | 0.8 | 13.1 | 343.6 | 5365.8 | 15.62x | 3208.030 ms | 219.443 ms | 0.8 | 12.2 | 344.5 | 5036.6 | 14.62x | PASS |
| Lambda_OpenMP/metric_rb | 42.249 ms | 2.125 ms | 0.8 | 16.7 | 345.0 | 6860.4 | 19.89x | 99.597 ms | 4.997 ms | 0.8 | 16.7 | 343.6 | 6848.7 | 19.93x | 3208.030 ms | 178.817 ms | 0.8 | 15.0 | 344.5 | 6181.0 | 17.94x | PASS |
| Lambda_OpenMP/metric_rb_x2 | 42.249 ms | 2.385 ms | 0.8 | 14.9 | 345.0 | 6112.2 | 17.72x | 99.597 ms | 5.461 ms | 0.8 | 15.3 | 343.6 | 6266.3 | 18.24x | 3208.030 ms | 186.813 ms | 0.8 | 14.4 | 344.5 | 5916.4 | 17.17x | PASS |
| Lambda_OpenMP/metric_rb_x4 | 42.249 ms | 2.895 ms | 0.8 | 12.3 | 345.0 | 5035.3 | 14.60x | 99.597 ms | 6.975 ms | 0.8 | 11.9 | 343.6 | 4906.2 | 14.28x | 3208.030 ms | 231.287 ms | 0.8 | 11.6 | 344.5 | 4778.7 | 13.87x | PASS |
| Lambda_OpenMP/metric_x2 | 42.249 ms | 2.867 ms | 0.8 | 12.4 | 345.0 | 5083.8 | 14.74x | 99.597 ms | 6.408 ms | 0.8 | 13.0 | 343.6 | 5340.2 | 15.54x | 3208.030 ms | 219.389 ms | 0.8 | 12.2 | 344.5 | 5037.9 | 14.62x | PASS |
| Lambda_OpenMP/metric_x4 | 42.249 ms | 2.928 ms | 0.8 | 12.1 | 345.0 | 4977.6 | 14.43x | 99.597 ms | 7.222 ms | 0.8 | 11.5 | 343.6 | 4738.2 | 13.79x | 3208.030 ms | 233.165 ms | 0.8 | 11.5 | 344.5 | 4740.2 | 13.76x | PASS |
| RAJA_OpenMP | 42.249 ms | 6.351 ms | 0.8 | 5.6 | 345.0 | 2295.0 | 6.65x | 99.351 ms | 14.271 ms | 0.8 | 5.8 | 344.4 | 2397.9 | 6.96x | 3206.550 ms | 499.014 ms | 0.8 | 5.4 | 344.7 | 2214.9 | 6.43x | PASS |
| RAJA_OpenMP/metric | 42.249 ms | 2.616 ms | 0.8 | 13.6 | 345.0 | 5571.3 | 16.15x | 99.351 ms | 6.208 ms | 0.8 | 13.4 | 344.4 | 5512.7 | 16.00x | 3206.550 ms | 220.774 ms | 0.8 | 12.1 | 344.7 | 5006.3 | 14.52x | PASS |
| RAJA_OpenMP/metric_rb | 42.249 ms | 2.131 ms | 0.8 | 16.7 | 345.0 | 6839.7 | 19.83x | 99.351 ms | 5.113 ms | 0.8 | 16.3 | 344.4 | 6693.1 | 19.43x | 3206.550 ms | 179.669 ms | 0.8 | 14.9 | 344.7 | 6151.6 | 17.85x | PASS |
| RAJA_OpenMP/metric_rb_x2 | 42.249 ms | 2.431 ms | 0.8 | 14.6 | 345.0 | 5994.4 | 17.38x | 99.351 ms | 5.494 ms | 0.8 | 15.2 | 344.4 | 6229.2 | 18.09x | 3206.550 ms | 186.098 ms | 0.8 | 14.4 | 344.7 | 5939.1 | 17.23x | PASS |
| RAJA_OpenMP/metric_rb_x4 | 42.249 ms | 2.884 ms | 0.8 | 12.3 | 345.0 | 5053.4 | 14.65x | 99.351 ms | 6.843 ms | 0.8 | 12.2 | 344.4 | 5000.7 | 14.52x | 3206.550 ms | 229.156 ms | 0.8 | 11.7 | 344.7 | 4823.2 | 13.99x | PASS |
| RAJA_OpenMP/metric_x2 | 42.249 ms | 2.845 ms | 0.8 | 12.5 | 345.0 | 5122.9 | 14.85x | 99.351 ms | 6.571 ms | 0.8 | 12.7 | 344.4 | 5207.7 | 15.12x | 3206.550 ms | 216.340 ms | 0.8 | 12.4 | 344.7 | 5108.9 | 14.82x | PASS |
| RAJA_OpenMP/metric_x4 | 42.249 ms | 3.030 ms | 0.8 | 11.7 | 345.0 | 4810.7 | 13.94x | 99.351 ms | 7.027 ms | 0.8 | 11.9 | 344.4 | 4869.8 | 14.14x | 3206.550 ms | 233.308 ms | 0.8 | 11.5 | 344.7 | 4737.3 | 13.74x | PASS |

### DIFFUSION3DPA

| Variant | default: develop t/rep | default: vectorized t/rep | default: develop GB/s | default: vectorized GB/s | default: develop GFLOP/s | default: vectorized GFLOP/s | default: speedup | fit: develop t/rep | fit: vectorized t/rep | fit: develop GB/s | fit: vectorized GB/s | fit: develop GFLOP/s | fit: vectorized GFLOP/s | fit: speedup | exceed: develop t/rep | exceed: vectorized t/rep | exceed: develop GB/s | exceed: vectorized GB/s | exceed: develop GFLOP/s | exceed: vectorized GFLOP/s | exceed: speedup | Checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| Base_OpenMP | 137.6 us | 59.4 us | 422.4 | 978.5 | 698.7 | 1618.5 | 2.32x | 208.5 us | 87.8 us | 427.0 | 1014.9 | 706.3 | 1678.6 | 2.38x | 12.879 ms | 9.805 ms | 221.3 | 290.7 | 366.0 | 480.8 | 1.31x | PASS |
| Base_OpenMP/transpose | 137.6 us | 86.4 us | 422.4 | 673.1 | 698.7 | 1113.2 | 1.59x | 208.5 us | 129.2 us | 427.0 | 689.1 | 706.3 | 1139.8 | 1.61x | 12.879 ms | 9.420 ms | 221.3 | 302.5 | 366.0 | 500.4 | 1.37x | PASS |
| RAJA_OpenMP | 142.1 us | 112.4 us | 409.2 | 517.0 | 676.8 | 855.2 | 1.26x | 214.3 us | 166.9 us | 415.5 | 533.6 | 687.3 | 882.5 | 1.28x | 13.801 ms | 10.852 ms | 206.5 | 262.6 | 341.5 | 434.4 | 1.27x | PASS |
| RAJA_OpenMP/transpose | 142.1 us | 89.1 us | 409.2 | 652.6 | 676.8 | 1079.5 | 1.59x | 214.3 us | 130.7 us | 415.5 | 681.2 | 687.3 | 1126.8 | 1.64x | 13.801 ms | 9.414 ms | 206.5 | 302.7 | 341.5 | 500.7 | 1.47x | PASS |

### ENERGY

| Variant | default: develop t/rep | default: vectorized t/rep | default: develop GB/s | default: vectorized GB/s | default: develop GFLOP/s | default: vectorized GFLOP/s | default: speedup | fit: develop t/rep | fit: vectorized t/rep | fit: develop GB/s | fit: vectorized GB/s | fit: develop GFLOP/s | fit: vectorized GFLOP/s | fit: speedup | exceed: develop t/rep | exceed: vectorized t/rep | exceed: develop GB/s | exceed: vectorized GB/s | exceed: develop GFLOP/s | exceed: vectorized GFLOP/s | exceed: speedup | Checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| Base_OpenMP | 119.3 us | 118.7 us | 1810.0 | 1819.8 | 460.9 | 463.4 | 1.01x | 101.3 us | 100.5 us | 1490.3 | 1502.9 | 379.5 | 382.7 | 1.01x | 24.710 ms | 24.661 ms | 195.5 | 195.9 | 49.8 | 49.9 | 1.00x | PASS |
| Base_OpenMP/fused | 119.3 us | 85.3 us | 1810.0 | 2533.4 | 460.9 | 645.1 | 1.40x | 101.3 us | 64.5 us | 1490.3 | 2339.2 | 379.5 | 595.6 | 1.57x | 24.710 ms | 8.914 ms | 195.5 | 542.0 | 49.8 | 138.0 | 2.77x | PASS |
| Base_OpenMP/fused_nt | 119.3 us | 90.1 us | 1810.0 | 2398.4 | 460.9 | 610.7 | 1.33x | 101.3 us | 69.9 us | 1490.3 | 2159.2 | 379.5 | 549.8 | 1.45x | 24.710 ms | 8.924 ms | 195.5 | 541.4 | 49.8 | 137.9 | 2.77x | PASS |
| Lambda_OpenMP | 133.8 us | 232.7 us | 1614.1 | 928.3 | 411.0 | 236.4 | 0.58x | 116.0 us | 115.8 us | 1302.2 | 1303.9 | 331.6 | 332.0 | 1.00x | 24.636 ms | 24.561 ms | 196.1 | 196.7 | 49.9 | 50.1 | 1.00x | PASS |
| Lambda_OpenMP/fused | 133.8 us | 85.3 us | 1614.1 | 2533.0 | 411.0 | 645.0 | 1.57x | 116.0 us | 64.4 us | 1302.2 | 2344.9 | 331.6 | 597.1 | 1.80x | 24.636 ms | 8.915 ms | 196.1 | 542.0 | 49.9 | 138.0 | 2.76x | PASS |
| Lambda_OpenMP/fused_nt | 133.8 us | 90.1 us | 1614.1 | 2397.5 | 411.0 | 610.5 | 1.49x | 116.0 us | 69.6 us | 1302.2 | 2169.3 | 331.6 | 552.4 | 1.67x | 24.636 ms | 8.926 ms | 196.1 | 541.3 | 49.9 | 137.8 | 2.76x | PASS |
| RAJA_OpenMP | 110.5 us | 111.4 us | 1954.2 | 1938.9 | 497.6 | 493.7 | 0.99x | 99.7 us | 99.1 us | 1514.9 | 1524.0 | 385.7 | 388.0 | 1.01x | 24.663 ms | 24.581 ms | 195.9 | 196.6 | 49.9 | 50.1 | 1.00x | PASS |
| RAJA_OpenMP/fused | 110.5 us | 84.2 us | 1954.2 | 2566.4 | 497.6 | 653.5 | 1.31x | 99.7 us | 62.7 us | 1514.9 | 2408.8 | 385.7 | 613.4 | 1.59x | 24.663 ms | 8.910 ms | 195.9 | 542.3 | 49.9 | 138.1 | 2.77x | PASS |
| RAJA_OpenMP/fused_nt | 110.5 us | 89.5 us | 1954.2 | 2412.6 | 497.6 | 614.3 | 1.23x | 99.7 us | 67.7 us | 1514.9 | 2231.4 | 385.7 | 568.2 | 1.47x | 24.663 ms | 8.928 ms | 195.9 | 541.2 | 49.9 | 137.8 | 2.76x | PASS |

### FEMSWEEP

| Variant | default: develop t/rep | default: vectorized t/rep | default: develop GB/s | default: vectorized GB/s | default: develop GFLOP/s | default: vectorized GFLOP/s | default: speedup | fit: develop t/rep | fit: vectorized t/rep | fit: develop GB/s | fit: vectorized GB/s | fit: develop GFLOP/s | fit: vectorized GFLOP/s | fit: speedup | exceed: develop t/rep | exceed: vectorized t/rep | exceed: develop GB/s | exceed: vectorized GB/s | exceed: develop GFLOP/s | exceed: vectorized GFLOP/s | exceed: speedup | Checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| Base_OpenMP | 1.040 ms | 281.1 us | 23.9 | 88.4 | 118.6 | 438.9 | 3.70x | 4.324 ms | 954.9 us | 24.4 | 110.5 | 118.5 | 536.5 | 4.53x | 198.001 ms | 34.534 ms | 16.9 | 97.0 | 80.2 | 459.6 | 5.73x | PASS |
| Base_OpenMP/prefetch | 1.040 ms | 133.4 us | 23.9 | 186.1 | 118.6 | 924.7 | 7.80x | 4.324 ms | 498.2 us | 24.4 | 211.7 | 118.5 | 1028.3 | 8.68x | 198.001 ms | 21.882 ms | 16.9 | 153.1 | 80.2 | 725.3 | 9.05x | PASS |
| Base_OpenMP/prefetch_mixed | 1.040 ms | 129.9 us | 23.9 | 191.2 | 118.6 | 949.7 | 8.01x | 4.324 ms | 430.1 us | 24.4 | 245.3 | 118.5 | 1191.0 | 10.05x | 198.001 ms | 18.238 ms | 16.9 | 183.6 | 80.2 | 870.3 | 10.86x | PASS |
| Base_OpenMP/prefetch_mixed_ft | 1.040 ms | 123.7 us | 23.9 | 200.8 | 118.6 | 997.4 | 8.41x | 4.324 ms | 411.7 us | 24.4 | 256.2 | 118.5 | 1244.2 | 10.50x | 198.001 ms | 17.162 ms | 16.9 | 195.2 | 80.2 | 924.8 | 11.54x | PASS |
| Base_OpenMP/prefetch_pf1 | 1.040 ms | 131.7 us | 23.9 | 188.5 | 118.6 | 936.5 | 7.90x | 4.324 ms | 501.1 us | 24.4 | 210.5 | 118.5 | 1022.2 | 8.63x | 198.001 ms | 22.034 ms | 16.9 | 152.0 | 80.2 | 720.3 | 8.99x | PASS |
| Base_OpenMP/prefetch_pf4 | 1.040 ms | 135.3 us | 23.9 | 183.6 | 118.6 | 912.1 | 7.69x | 4.324 ms | 501.7 us | 24.4 | 210.3 | 118.5 | 1021.1 | 8.62x | 198.001 ms | 22.380 ms | 16.9 | 149.7 | 80.2 | 709.2 | 8.85x | PASS |
| RAJA_OpenMP | 1.008 ms | 293.4 us | 24.6 | 84.6 | 122.4 | 420.4 | 3.43x | 4.185 ms | 994.0 us | 25.2 | 106.1 | 122.4 | 515.3 | 4.21x | 188.636 ms | 34.692 ms | 17.8 | 96.5 | 84.1 | 457.5 | 5.44x | PASS |
| RAJA_OpenMP/prefetch | 1.008 ms | 137.2 us | 24.6 | 181.1 | 122.4 | 899.4 | 7.35x | 4.185 ms | 508.4 us | 25.2 | 207.5 | 122.4 | 1007.7 | 8.23x | 188.636 ms | 22.088 ms | 17.8 | 151.6 | 84.1 | 718.6 | 8.54x | PASS |
| RAJA_OpenMP/prefetch_mixed | 1.008 ms | 135.3 us | 24.6 | 183.5 | 122.4 | 911.7 | 7.45x | 4.185 ms | 435.0 us | 25.2 | 242.5 | 122.4 | 1177.6 | 9.62x | 188.636 ms | 18.370 ms | 17.8 | 182.3 | 84.1 | 864.0 | 10.27x | PASS |
| RAJA_OpenMP/prefetch_mixed_ft | 1.008 ms | 129.9 us | 24.6 | 191.2 | 122.4 | 949.7 | 7.76x | 4.185 ms | 410.8 us | 25.2 | 256.8 | 122.4 | 1247.1 | 10.19x | 188.636 ms | 16.983 ms | 17.8 | 197.2 | 84.1 | 934.6 | 11.11x | PASS |
| RAJA_OpenMP/prefetch_pf1 | 1.008 ms | 159.6 us | 24.6 | 155.6 | 122.4 | 773.2 | 6.32x | 4.185 ms | 508.7 us | 25.2 | 207.4 | 122.4 | 1007.0 | 8.23x | 188.636 ms | 21.983 ms | 17.8 | 152.4 | 84.1 | 722.0 | 8.58x | PASS |
| RAJA_OpenMP/prefetch_pf4 | 1.008 ms | 138.6 us | 24.6 | 179.2 | 122.4 | 890.0 | 7.27x | 4.185 ms | 508.5 us | 25.2 | 207.4 | 122.4 | 1007.3 | 8.23x | 188.636 ms | 22.403 ms | 17.8 | 149.5 | 84.1 | 708.5 | 8.42x | PASS |

### INTSC_HEXRECT

| Variant | default: develop t/rep | default: vectorized t/rep | default: develop GB/s | default: vectorized GB/s | default: develop GFLOP/s | default: vectorized GFLOP/s | default: speedup | fit: develop t/rep | fit: vectorized t/rep | fit: develop GB/s | fit: vectorized GB/s | fit: develop GFLOP/s | fit: vectorized GFLOP/s | fit: speedup | exceed: develop t/rep | exceed: vectorized t/rep | exceed: develop GB/s | exceed: vectorized GB/s | exceed: develop GFLOP/s | exceed: vectorized GFLOP/s | exceed: speedup | Checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| Base_OpenMP | 16.654 ms | 16.174 ms | 2.8 | 2.9 | 216.2 | 222.6 | 1.03x | 29.494 ms | 29.493 ms | 2.8 | 2.8 | 210.9 | 210.9 | 1.00x | 1008.530 ms | 979.422 ms | 2.6 | 2.7 | 202.1 | 208.1 | 1.03x | PASS |
| Base_OpenMP/split | 16.654 ms | 7.373 ms | 2.8 | 6.4 | 216.2 | 488.3 | 2.26x | 29.494 ms | 12.389 ms | 2.8 | 6.6 | 210.9 | 502.1 | 2.38x | 1008.530 ms | 396.755 ms | 2.6 | 6.7 | 202.1 | 513.8 | 2.54x | PASS |
| Base_OpenMP/split_lean | 16.654 ms | 6.215 ms | 2.8 | 7.6 | 216.2 | 579.2 | 2.68x | 29.494 ms | 10.674 ms | 2.8 | 7.6 | 210.9 | 582.8 | 2.76x | 1008.530 ms | 341.072 ms | 2.6 | 7.8 | 202.1 | 597.7 | 2.96x | PASS |
| Base_OpenMP/split_simd | 16.654 ms | 2.186 ms | 2.8 | 21.6 | 216.2 | 1646.6 | 7.62x | 29.494 ms | 3.664 ms | 2.8 | 22.2 | 210.9 | 1697.7 | 8.05x | 1008.530 ms | 88.745 ms | 2.6 | 30.0 | 202.1 | 2297.0 | 11.36x | PASS |
| Lambda_OpenMP | 15.047 ms | 15.168 ms | 3.1 | 3.1 | 239.3 | 237.3 | 0.99x | 27.453 ms | 27.521 ms | 3.0 | 3.0 | 226.6 | 226.0 | 1.00x | 930.125 ms | 925.063 ms | 2.9 | 2.9 | 219.2 | 220.4 | 1.01x | PASS |
| Lambda_OpenMP/split | 15.047 ms | 7.481 ms | 3.1 | 6.3 | 239.3 | 481.2 | 2.01x | 27.453 ms | 12.788 ms | 3.0 | 6.4 | 226.6 | 486.5 | 2.15x | 930.125 ms | 416.582 ms | 2.9 | 6.4 | 219.2 | 489.3 | 2.23x | PASS |
| Lambda_OpenMP/split_lean | 15.047 ms | 6.326 ms | 3.1 | 7.5 | 239.3 | 569.1 | 2.38x | 27.453 ms | 10.852 ms | 3.0 | 7.5 | 226.6 | 573.2 | 2.53x | 930.125 ms | 348.404 ms | 2.9 | 7.6 | 219.2 | 585.1 | 2.67x | PASS |
| Lambda_OpenMP/split_simd | 15.047 ms | 2.166 ms | 3.1 | 21.8 | 239.3 | 1661.8 | 6.95x | 27.453 ms | 3.702 ms | 3.0 | 22.0 | 226.6 | 1680.4 | 7.42x | 930.125 ms | 88.027 ms | 2.9 | 30.3 | 219.2 | 2315.7 | 10.57x | PASS |
| RAJA_OpenMP | 15.241 ms | 15.239 ms | 3.1 | 3.1 | 236.2 | 236.2 | 1.00x | 27.667 ms | 27.311 ms | 2.9 | 3.0 | 224.8 | 227.8 | 1.01x | 935.753 ms | 925.787 ms | 2.8 | 2.9 | 217.8 | 220.2 | 1.01x | PASS |
| RAJA_OpenMP/split | 15.241 ms | 7.262 ms | 3.1 | 6.5 | 236.2 | 495.7 | 2.10x | 27.667 ms | 12.450 ms | 2.9 | 6.5 | 224.8 | 499.6 | 2.22x | 935.753 ms | 399.371 ms | 2.8 | 6.7 | 217.8 | 510.4 | 2.34x | PASS |
| RAJA_OpenMP/split_lean | 15.241 ms | 6.345 ms | 3.1 | 7.4 | 236.2 | 567.4 | 2.40x | 27.667 ms | 10.820 ms | 2.9 | 7.5 | 224.8 | 574.9 | 2.56x | 935.753 ms | 348.099 ms | 2.8 | 7.7 | 217.8 | 585.6 | 2.69x | PASS |
| RAJA_OpenMP/split_simd | 15.241 ms | 2.188 ms | 3.1 | 21.6 | 236.2 | 1645.1 | 6.97x | 27.667 ms | 3.704 ms | 2.9 | 22.0 | 224.8 | 1679.4 | 7.47x | 935.753 ms | 88.182 ms | 2.8 | 30.2 | 217.8 | 2311.6 | 10.61x | PASS |

### MASS3DEA

| Variant | default: develop t/rep | default: vectorized t/rep | default: develop GB/s | default: vectorized GB/s | default: develop GFLOP/s | default: vectorized GFLOP/s | default: speedup | fit: develop t/rep | fit: vectorized t/rep | fit: develop GB/s | fit: vectorized GB/s | fit: develop GFLOP/s | fit: vectorized GFLOP/s | fit: speedup | exceed: develop t/rep | exceed: vectorized t/rep | exceed: develop GB/s | exceed: vectorized GB/s | exceed: develop GFLOP/s | exceed: vectorized GFLOP/s | exceed: speedup | Checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| Base_OpenMP | 355.7 us | 98.6 us | 11.9 | 42.8 | 1259.4 | 4541.9 | 3.61x | 5.898 ms | 734.7 us | 14.2 | 114.2 | 1509.4 | 12117.9 | 8.03x | 188.210 ms | 25.931 ms | 14.3 | 103.5 | 1513.8 | 10987.0 | 7.26x | PASS |
| Base_OpenMP/sumfact | 355.7 us | 87.1 us | 11.9 | 48.5 | 1259.4 | 5145.1 | 4.09x | 5.898 ms | 153.8 us | 14.2 | 545.3 | 1509.4 | 57881.1 | 38.35x | 188.210 ms | 16.171 ms | 14.3 | 166.0 | 1513.8 | 17618.7 | 11.64x | PASS |
| Base_OpenMP/sumfact_nt | 355.7 us | 95.2 us | 11.9 | 44.4 | 1259.4 | 4708.3 | 3.74x | 5.898 ms | 693.6 us | 14.2 | 120.9 | 1509.4 | 12835.9 | 8.50x | 188.210 ms | 15.248 ms | 14.3 | 176.0 | 1513.8 | 18684.6 | 12.34x | PASS |
| RAJA_OpenMP | 275.8 us | 103.6 us | 15.3 | 40.7 | 1624.3 | 4324.8 | 2.66x | 4.222 ms | 933.9 us | 19.9 | 89.8 | 2108.8 | 9533.2 | 4.52x | 135.096 ms | 37.486 ms | 19.9 | 71.6 | 2108.9 | 7600.4 | 3.60x | PASS |
| RAJA_OpenMP/sumfact | 275.8 us | 86.3 us | 15.3 | 48.9 | 1624.3 | 5192.1 | 3.20x | 4.222 ms | 149.9 us | 19.9 | 559.5 | 2108.8 | 59380.6 | 28.16x | 135.096 ms | 16.158 ms | 19.9 | 166.1 | 2108.9 | 17632.9 | 8.36x | PASS |
| RAJA_OpenMP/sumfact_nt | 275.8 us | 95.7 us | 15.3 | 44.1 | 1624.3 | 4680.6 | 2.88x | 4.222 ms | 683.5 us | 19.9 | 122.7 | 2108.8 | 13025.1 | 6.18x | 135.096 ms | 15.239 ms | 19.9 | 176.2 | 2108.9 | 18696.4 | 8.87x | PASS |

### MASS3DPA_ATOMIC

| Variant | default: develop t/rep | default: vectorized t/rep | default: develop GB/s | default: vectorized GB/s | default: develop GFLOP/s | default: vectorized GFLOP/s | default: speedup | fit: develop t/rep | fit: vectorized t/rep | fit: develop GB/s | fit: vectorized GB/s | fit: develop GFLOP/s | fit: vectorized GFLOP/s | fit: speedup | exceed: develop t/rep | exceed: vectorized t/rep | exceed: develop GB/s | exceed: vectorized GB/s | exceed: develop GFLOP/s | exceed: vectorized GFLOP/s | exceed: speedup | Checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| Base_OpenMP | 860.9 us | 345.5 us | 134.4 | 334.9 | 271.1 | 675.4 | 2.49x | 643.8 us | 299.9 us | 140.0 | 300.6 | 282.3 | 605.9 | 2.15x | 21.075 ms | 10.768 ms | 136.1 | 266.5 | 275.7 | 539.6 | 1.96x | PASS |
| Base_OpenMP/pf1 | 860.9 us | 339.1 us | 134.4 | 341.3 | 271.1 | 688.3 | 2.54x | 643.8 us | 300.6 us | 140.0 | 299.9 | 282.3 | 604.5 | 2.14x | 21.075 ms | 10.527 ms | 136.1 | 272.6 | 275.7 | 551.9 | 2.00x | PASS |
| Base_OpenMP/pf2 | 860.9 us | 352.2 us | 134.4 | 328.5 | 271.1 | 662.6 | 2.44x | 643.8 us | 309.1 us | 140.0 | 291.7 | 282.3 | 587.9 | 2.08x | 21.075 ms | 10.693 ms | 136.1 | 268.3 | 275.7 | 543.4 | 1.97x | PASS |
| Base_OpenMP/pf4 | 860.9 us | 361.8 us | 134.4 | 319.9 | 271.1 | 645.1 | 2.38x | 643.8 us | 317.9 us | 140.0 | 283.6 | 282.3 | 571.6 | 2.03x | 21.075 ms | 10.919 ms | 136.1 | 262.8 | 275.7 | 532.1 | 1.93x | PASS |
| RAJA_OpenMP | 804.0 us | 321.4 us | 143.9 | 360.1 | 290.3 | 726.1 | 2.50x | 668.0 us | 261.6 us | 135.0 | 344.7 | 272.0 | 694.7 | 2.55x | 20.573 ms | 10.812 ms | 139.5 | 265.4 | 282.4 | 537.4 | 1.90x | PASS |
| RAJA_OpenMP/pf1 | 804.0 us | 339.7 us | 143.9 | 340.6 | 290.3 | 686.9 | 2.37x | 668.0 us | 271.6 us | 135.0 | 332.0 | 272.0 | 669.1 | 2.46x | 20.573 ms | 10.665 ms | 139.5 | 269.1 | 282.4 | 544.8 | 1.93x | PASS |
| RAJA_OpenMP/pf2 | 804.0 us | 354.2 us | 143.9 | 326.8 | 290.3 | 658.9 | 2.27x | 668.0 us | 281.2 us | 135.0 | 320.7 | 272.0 | 646.3 | 2.38x | 20.573 ms | 10.798 ms | 139.5 | 265.7 | 282.4 | 538.1 | 1.91x | PASS |
| RAJA_OpenMP/pf4 | 804.0 us | 373.8 us | 143.9 | 309.6 | 290.3 | 624.4 | 2.15x | 668.0 us | 294.1 us | 135.0 | 306.6 | 272.0 | 618.0 | 2.27x | 20.573 ms | 10.992 ms | 139.5 | 261.0 | 282.4 | 528.6 | 1.87x | PASS |

### MASSVEC3DPA

| Variant | default: develop t/rep | default: vectorized t/rep | default: develop GB/s | default: vectorized GB/s | default: develop GFLOP/s | default: vectorized GFLOP/s | default: speedup | fit: develop t/rep | fit: vectorized t/rep | fit: develop GB/s | fit: vectorized GB/s | fit: develop GFLOP/s | fit: vectorized GFLOP/s | fit: speedup | exceed: develop t/rep | exceed: vectorized t/rep | exceed: develop GB/s | exceed: vectorized GB/s | exceed: develop GFLOP/s | exceed: vectorized GFLOP/s | exceed: speedup | Checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| Base_OpenMP | 65.9 us | 21.5 us | 142.9 | 438.9 | 436.4 | 1339.9 | 3.07x | 533.4 us | 113.9 us | 157.3 | 736.7 | 480.1 | 2249.3 | 4.68x | 19.097 ms | 12.532 ms | 140.6 | 214.2 | 429.2 | 654.0 | 1.52x | PASS |
| Base_OpenMP/nt | 65.9 us | 24.4 us | 142.9 | 386.7 | 436.4 | 1180.6 | 2.71x | 533.4 us | 176.4 us | 157.3 | 475.6 | 480.1 | 1452.1 | 3.02x | 19.097 ms | 9.680 ms | 140.6 | 277.3 | 429.2 | 846.6 | 1.97x | PASS |
| Base_OpenMP/nt_pf1 | 65.9 us | 24.3 us | 142.9 | 387.1 | 436.4 | 1182.0 | 2.71x | 533.4 us | 180.8 us | 157.3 | 464.0 | 480.1 | 1416.8 | 2.95x | 19.097 ms | 9.255 ms | 140.6 | 290.1 | 429.2 | 885.6 | 2.06x | PASS |
| Base_OpenMP/nt_pf2 | 65.9 us | 24.3 us | 142.9 | 386.7 | 436.4 | 1180.7 | 2.71x | 533.4 us | 180.1 us | 157.3 | 465.8 | 480.1 | 1422.2 | 2.96x | 19.097 ms | 9.264 ms | 140.6 | 289.8 | 429.2 | 884.6 | 2.06x | PASS |
| Base_OpenMP/nt_pf4 | 65.9 us | 24.3 us | 142.9 | 386.9 | 436.4 | 1181.1 | 2.71x | 533.4 us | 179.7 us | 157.3 | 466.7 | 480.1 | 1425.0 | 2.97x | 19.097 ms | 9.279 ms | 140.6 | 289.3 | 429.2 | 883.2 | 2.06x | PASS |
| RAJA_OpenMP | 68.2 us | 24.8 us | 138.1 | 379.0 | 421.8 | 1157.1 | 2.74x | 527.1 us | 114.7 us | 159.1 | 731.1 | 485.9 | 2232.0 | 4.59x | 17.844 ms | 12.511 ms | 150.4 | 214.6 | 459.3 | 655.1 | 1.43x | PASS |
| RAJA_OpenMP/nt | 68.2 us | 24.0 us | 138.1 | 392.2 | 421.8 | 1197.3 | 2.84x | 527.1 us | 175.6 us | 159.1 | 477.8 | 485.9 | 1458.7 | 3.00x | 17.844 ms | 9.696 ms | 150.4 | 276.9 | 459.3 | 845.3 | 1.84x | PASS |
| RAJA_OpenMP/nt_pf1 | 68.2 us | 24.2 us | 138.1 | 389.4 | 421.8 | 1188.8 | 2.82x | 527.1 us | 173.0 us | 159.1 | 484.9 | 485.9 | 1480.6 | 3.05x | 17.844 ms | 9.253 ms | 150.4 | 290.1 | 459.3 | 885.8 | 1.93x | PASS |
| RAJA_OpenMP/nt_pf2 | 68.2 us | 24.8 us | 138.1 | 380.4 | 421.8 | 1161.5 | 2.75x | 527.1 us | 179.3 us | 159.1 | 467.9 | 485.9 | 1428.4 | 2.94x | 17.844 ms | 9.277 ms | 150.4 | 289.4 | 459.3 | 883.4 | 1.92x | PASS |
| RAJA_OpenMP/nt_pf4 | 68.2 us | 24.5 us | 138.1 | 384.7 | 421.8 | 1174.4 | 2.78x | 527.1 us | 173.5 us | 159.1 | 483.6 | 485.9 | 1476.5 | 3.04x | 17.844 ms | 9.276 ms | 150.4 | 289.4 | 459.3 | 883.6 | 1.92x | PASS |

### NODAL_ACCUMULATION_3D

| Variant | default: develop t/rep | default: vectorized t/rep | default: develop GB/s | default: vectorized GB/s | default: develop GFLOP/s | default: vectorized GFLOP/s | default: speedup | fit: develop t/rep | fit: vectorized t/rep | fit: develop GB/s | fit: vectorized GB/s | fit: develop GFLOP/s | fit: vectorized GFLOP/s | fit: speedup | exceed: develop t/rep | exceed: vectorized t/rep | exceed: develop GB/s | exceed: vectorized GB/s | exceed: develop GFLOP/s | exceed: vectorized GFLOP/s | exceed: speedup | Checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| Base_OpenMP | 877.2 us | 29.9 us | 37.0 | 1086.9 | 10.3 | 301.1 | 29.35x | 2.892 ms | 80.0 us | 37.0 | 1336.5 | 10.3 | 372.1 | 36.15x | 97.546 ms | 11.568 ms | 36.2 | 305.0 | 10.1 | 85.5 | 8.43x | PASS |
| Lambda_OpenMP | 876.4 us | 29.9 us | 37.1 | 1084.7 | 10.3 | 300.5 | 29.26x | 2.891 ms | 79.3 us | 37.0 | 1347.7 | 10.3 | 375.2 | 36.44x | 97.513 ms | 11.561 ms | 36.2 | 305.2 | 10.1 | 85.6 | 8.43x | PASS |
| RAJA_OpenMP | 872.7 us | 32.8 us | 37.2 | 990.2 | 10.3 | 274.3 | 26.60x | 2.874 ms | 149.9 us | 37.2 | 713.1 | 10.4 | 198.6 | 19.17x | 96.225 ms | 11.600 ms | 36.7 | 304.1 | 10.3 | 85.3 | 8.30x | PASS |

### VOL3D

| Variant | default: develop t/rep | default: vectorized t/rep | default: develop GB/s | default: vectorized GB/s | default: develop GFLOP/s | default: vectorized GFLOP/s | default: speedup | fit: develop t/rep | fit: vectorized t/rep | fit: develop GB/s | fit: vectorized GB/s | fit: develop GFLOP/s | fit: vectorized GFLOP/s | fit: speedup | exceed: develop t/rep | exceed: vectorized t/rep | exceed: develop GB/s | exceed: vectorized GB/s | exceed: develop GFLOP/s | exceed: vectorized GFLOP/s | exceed: speedup | Checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| Base_OpenMP | 43.6 us | 43.5 us | 816.3 | 817.9 | 1823.2 | 1826.8 | 1.00x | 92.6 us | 92.5 us | 900.3 | 900.9 | 2014.4 | 2015.8 | 1.00x | 16.567 ms | 16.575 ms | 161.8 | 161.8 | 363.5 | 363.3 | 1.00x | PASS |
| Base_OpenMP/blocked | 43.6 us | 53.6 us | 816.3 | 663.3 | 1823.2 | 1481.5 | 0.81x | 92.6 us | 144.7 us | 900.3 | 575.8 | 2014.4 | 1288.4 | 0.64x | 16.567 ms | 10.774 ms | 161.8 | 248.9 | 363.5 | 559.0 | 1.54x | PASS |
| Base_OpenMP/nt | 43.6 us | 56.5 us | 816.3 | 629.4 | 1823.2 | 1405.8 | 0.77x | 92.6 us | 149.1 us | 900.3 | 559.0 | 2014.4 | 1250.7 | 0.62x | 16.567 ms | 15.808 ms | 161.8 | 169.6 | 363.5 | 381.0 | 1.05x | PASS |
| Lambda_OpenMP | 43.4 us | 43.8 us | 818.8 | 811.3 | 1828.7 | 1811.9 | 0.99x | 94.3 us | 93.8 us | 884.1 | 888.0 | 1978.3 | 1987.0 | 1.00x | 16.608 ms | 16.593 ms | 161.4 | 161.6 | 362.6 | 362.9 | 1.00x | PASS |
| Lambda_OpenMP/blocked | 43.4 us | 54.6 us | 818.8 | 651.2 | 1828.7 | 1454.3 | 0.80x | 94.3 us | 146.7 us | 884.1 | 568.1 | 1978.3 | 1271.2 | 0.64x | 16.608 ms | 10.804 ms | 161.4 | 248.2 | 362.6 | 557.4 | 1.54x | PASS |
| Lambda_OpenMP/nt | 43.4 us | 56.8 us | 818.8 | 626.2 | 1828.7 | 1398.6 | 0.76x | 94.3 us | 151.6 us | 884.1 | 549.6 | 1978.3 | 1229.7 | 0.62x | 16.608 ms | 15.812 ms | 161.4 | 169.6 | 362.6 | 380.9 | 1.05x | PASS |
| RAJA_OpenMP | 46.6 us | 45.6 us | 762.7 | 779.5 | 1703.5 | 1741.0 | 1.02x | 92.8 us | 95.1 us | 897.8 | 876.6 | 2009.0 | 1961.4 | 0.98x | 16.565 ms | 16.551 ms | 161.9 | 162.0 | 363.6 | 363.9 | 1.00x | PASS |
| RAJA_OpenMP/blocked | 46.6 us | 56.8 us | 762.7 | 626.4 | 1703.5 | 1399.1 | 0.82x | 92.8 us | 142.6 us | 897.8 | 584.4 | 2009.0 | 1307.7 | 0.65x | 16.565 ms | 10.835 ms | 161.9 | 247.5 | 363.6 | 555.8 | 1.53x | PASS |
| RAJA_OpenMP/nt | 46.6 us | 60.7 us | 762.7 | 585.8 | 1703.5 | 1308.2 | 0.77x | 92.8 us | 148.8 us | 897.8 | 560.1 | 2009.0 | 1253.2 | 0.62x | 16.565 ms | 15.758 ms | 161.9 | 170.1 | 363.6 | 382.2 | 1.05x | PASS |

