---
name: hpc-kernel-optimization
description: 'Staged CPU optimization workflow for HPC loop kernels (finite element, stencil, streaming, geometry, many small dense solves). Use when: optimizing or vectorizing a compute kernel; deciding if a kernel is compute, memory, latency or cache bound; roofline analysis; AoS to SoA conversion or element batching; removing atomics; loop fusion; sum factorization; algebraic reformulation; SIMD intrinsics (AVX-512, AVX2, SVE, NEON); forcing FMA; streaming or non-temporal stores; checking whether GFLOP/s or GB/s above peak is real; comparing an optimized branch against a baseline.'
---

# HPC Kernel Optimization Strategy

Work through the stages **in order**. Each stage is more complex and less
portable than the one before, so only move on when the earlier stages have
been exhausted or clearly cannot help. Measure after every stage. Keep the
reference implementation runnable so every change can be compared against it.

| Stage | What | Typical gain | Cost / risk |
|---|---|---|---|
| 0 | Baseline, sensitive correctness checks, measurement setup | enables the rest | low |
| 1 | Classify: compute, memory, latency or cache bound | decides the path | low |
| 2 | Data layout for vectorization, remove serialization | 1.3x to over 30x | medium |
| 3 | Better algorithm: same result, less work or traffic | 2x to over 30x | medium to high |
| 4 | Intrinsics (FMA, register blocking, transposes), streaming stores | 1.0x to 2.3x on top of stages 2 and 3 | high, ISA-specific |

Rules of thumb from past use:
- **The large wins come from stages 2 and 3.** Intrinsics on top of
  well-vectorized compiler code gave 1.0x–1.2x for most kernels and at most
  about 2.3x for register-heavy dense contractions.
- **Memory-bound kernels already at DRAM peak gain nothing** from better
  vectorization. Only moving fewer bytes helps.
- **Removing atomics or serialization** was the single largest win (over 30x)
  when it applied.

---

## Stage 0: Baseline and measurement setup

1. **Pin the baseline.** Build the untouched upstream commit in a separate
   tree with identical compiler and flags.
2. **Build flags.**
   - Use `-O3 -march=<target> -ffast-math` (or the project's equivalent).
   - On AVX-512 parts add `-mprefer-vector-width=512`; clang and gcc
     otherwise default to 256-bit.
3. **Run environment.**
   - Set threads to one socket's cores and pin them (`OMP_PLACES=cores
     OMP_PROC_BIND=close` or an explicit CPU list).
   - Report the median of several *interleaved* runs (A B A B ...) rather
     than back-to-back runs of one build.
4. **Make correctness checks sensitive before optimizing.**
   - Constant inputs (all 1.0) and regular meshes hide indexing bugs: the
     checksum does not change under permutation. Use random inputs, and
     perturb geometry (for example ±15% via a hash of the node id).
   - Prove the check works with a **mutation test**: deliberately swap two
     indices and confirm the check fails.
   - Keep a standalone harness that compares the new code against the
     reference on random data. Cover tail sizes (N mod VL != 0) and both
     sides of every data-dependent branch.
5. **Problem size.**
   - Compute the working set explicitly and compare it to the last-level
     cache (LLC).
   - Choose sizes that are deliberately above or below the LLC, and state
     which.
   - Default benchmark sizes are often tiny or cache-resident.

## Stage 1: Classify the kernel

Collect these numbers for every kernel before changing code:

| Measurement | How | Interpretation |
|---|---|---|
| Achieved GB/s | bytes moved / time | ≥ ~80% of *measured* socket DRAM bandwidth means memory bound |
| Achieved FLOP/s | FLOPs / time | compare to cores × FLOP/cycle × sustained SIMD clock |
| **Hardware FLOPs** | PMU counters, e.g. `perf stat -e fp_arith_inst_retired.{scalar,128b_packed,256b_packed,512b_packed}_double` (weights 1, 2, 4, 8; FMA already counts 2) | a nominal/hardware ratio above 1 means the nominal model overstates the real work |
| Vector share | wide-vector fraction of hardware FLOPs | low share means the code did not vectorize |
| Working set vs LLC | sum of touched array sizes | GB/s above DRAM peak means cache-resident, not a bandwidth win |
| Thread scaling | t(1) / t(N) | far below N points to bandwidth or serialization (atomics, critical sections) |
| Size sweep | 3–4 sizes spanning the LLC | a speedup that collapses once data leaves the LLC is the memory-bound signature |
| Compiler remarks | `-Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize` (clang), `-fopt-info-vec-all` (gcc) | shows which loops did not vectorize and why |
| Assembly | compile with `-S` for the target ISA; look for vector registers, FMA, gathers, spills | shows what really executes |

Classes and what to do next:
- **Memory bound at DRAM peak:** reduce traffic.
  - Fuse passes.
  - Avoid write-allocate with streaming stores.
  - Recompute instead of loading.
  - Use smaller types if allowed.
- **Memory bound below peak:** fix the access pattern.
  - Unit stride and SoA layout.
  - Gather instead of scatter.
  - No atomics.
  - Fewer indirections.
- **Compute bound:**
  - First look for an algorithm with less work (stage 3).
  - Then SIMD layout (stage 2) and intrinsics (stage 4).
- **Latency or IPC bound** (small dependent solves, branchy geometry):
  - Batch independent problems across SIMD lanes.
  - Fully unroll fixed trip counts.
  - Remove redundant work.
- **Cache resident:** say so.
  - Bandwidth above DRAM peak is expected here.
  - DRAM-oriented tricks such as streaming stores will *hurt*.

## Stage 2: Data structures for vectorization

Goal: every hot inner loop is unit stride, has the SIMD lane as the innermost
index, and has no cross-lane dependencies or serialization.

1. **AoS → SoA batching across independent items.** Per-item work with small
   inner loops (element quadrature, local DoFs, small matrices) vectorizes
   poorly *within* one item. Vectorize *across* items instead:
   - Process VL = vector bytes / sizeof(real) items per batch.
   - Stage each item's data into `buf[k][VL]` with the lane innermost.
   - Compute with `omp simd` lane loops, then write back.
   - Tails: replicate the last valid item into unused lanes so all lanes
     compute on valid data, and mask or skip their stores.
2. **Remove serialization.**
   - Replace scatter plus atomics (many-to-one accumulation) with a
     **gather**: loop over outputs and read the contributing inputs through
     an inverse map.
   - If a scatter must remain, do one masked, conflict-free scatter per batch
     after an in-register transpose. Do not use per-lane atomics.
3. **Batch many small independent solves** (small LU, Newton or quadratic
   solves) across lanes. Each lane solves a different system with identical
   control flow.
4. **Compiler pitfalls (clang, observed):**
   - Short lane loops (trip count = VL) get fully unrolled *before* the loop
     vectorizer runs, which leads to poor SLP code. Put
     `clang loop unroll(disable)` on the lane loop.
   - Conversely, fully unroll fixed-trip *outer* loops
     (`clang loop unroll(full)`) so table indices fold to constants.
   - **No stack realignment for wide-vector spills** in OpenMP outlined
     regions. This causes misaligned spill slots that split cache lines, and
     25–60% run-to-run differences between otherwise identical variants.
     Fix: move the batch body into a `noinline` function containing an
     `alignas(64)` local kept alive with `asm volatile("" :: "r"(ptr))`.
     This applies to plain C `omp simd` code with large `[..][VL]` arrays
     too.
   - Index macros that do not parenthesize their arguments break when passed
     expressions like `e0 + l`. Pass a plain variable.
5. **Verify.** Run the random-input harness, check the vectorization
   remarks, and inspect the assembly.

## Stage 3: More efficient algorithms (same result, less work)

Always look for this before writing intrinsics. For compute-bound kernels it
usually beats everything else. Patterns that generalize:

| Pattern | Recognize it by | Effect |
|---|---|---|
| **Loop fusion** | a chain of element-wise loops over the same index range, communicating through temporary arrays | one pass, temporaries stay in registers, traffic drops by the number of eliminated passes; keeping the operation order can make it bit-exact |
| **Sum factorization** | tensor-product structure, e.g. $\sum_q \prod_d B_{i_d q_d} B_{j_d q_d}\, D_q$ evaluated naively | contract one dimension at a time; work goes from $O(p^{2d} q^d)$ to roughly $O(p^{d+1} q^d)$, observed about 17x fewer FLOPs |
| **Metric / algebraic reformulation** | every basis function or vector is transformed by the same Jacobian at each quadrature point | transform the *metric* once per point ($G = J^T J/\det J$, $H = \det J\, J^{-1}J^{-T}$) and contract with the untransformed reference basis; exploit its sparsity; each entry becomes a few FMAs |
| **Share work across outputs** | the same input is processed separately against several neighbouring targets (e.g. clipping one cell against each of its $2^d$ overlapping cells) | split once by the shared planes or boundaries so each piece goes to exactly one target; skip copies when a piece lies entirely on one side |
| **Gather instead of scatter** | many-to-one accumulation with atomics | no atomics, full parallelism, vectorizable |
| **Batch independent problems** | many tiny identical solves | SIMD across problems |
| Symmetry | $M_{ij} = M_{ji}$ and similar | compute half |
| Hoisting / precompute | loop-invariant tables or factors | fewer operations per item |
| Recompute vs load | memory-bound kernel reading cheaply derivable data | less traffic, but this changes the benchmark input, so flag it |

Rules:
- **Keep the reference algorithm as the default** and add the new algorithm
  as a separately named variant or tuning, so baselines stay comparable.
- **Write the new algorithm in plain C first** and measure it. This separates
  the algorithmic gain from the intrinsics gain. Typically the algorithm
  gives most of it (for example 1.9x from the algorithm, then 1.2x from
  intrinsics).
- Nominal FLOP and byte models describe the *reference* algorithm.
  Reformulated variants will report GFLOP/s or GB/s *above hardware peak*.
  Report speedup instead, and use hardware counters for real efficiency (for
  example real FLOP/cycle out of the theoretical maximum).
- When a number exceeds peak, **verify before believing it**: random inputs,
  the mutation test, and hardware FLOP counts.
- Floating point: reassociation changes the last bits. Use a relative
  tolerance (about 1e-12 for double), and state whether the result is
  bit-exact.
- Do not silently change the benchmark's inputs or semantics.

## Stage 4: Intrinsics and streaming stores

Only for kernels that are still compute bound, or memory bound with a clear
store or gather inefficiency, after stages 2 and 3.

### 4a. Structure
- **One thin wrapper header** with `vd`, `vload`/`vstore` (plus masked
  forms), `vfma`/`vfms`/`vfnma`, `lane_mask(n)`, an in-register transpose,
  `load_transpose<N>` and `transpose_store<N>`.
- **Guard it with feature macros** (ISA flags, precision) plus an explicit
  `-D..._NO_INTRINSICS` switch. The compiler-vectorized path builds from the
  same source, so the two can be compared directly.
- **Always keep a portable fallback** (plain C lane loops). The development
  machine may lack the target ISA: test the fallback locally and the
  intrinsics path on the target.

### 4b. Techniques that pay off
- **Force FMA.** Write FMAs explicitly and rearrange formulas into FMA
  chains. For example, prescale weights so `w*x*x + acc` becomes
  `fma(w*x, x, acc)`.
- **Register blocking.** Keep an output tile of accumulators in registers
  across the whole reduction, and broadcast shared coefficients. Stay under
  the architectural register count (32 on AVX-512) to avoid spills.
- **Full unrolling** of fixed-size loops, so every table index is a
  compile-time constant. Without it, kernels become IPC bound on address
  arithmetic.
- **Transpose in registers instead of gather/scatter.**
  - AoS → SoA: load VL contiguous rows (one per item) and transpose them
    in registers.
  - Stores: transpose back and write contiguous rows.
  - Partial batches: masked loads and stores.
- **Large batch bodies in `noinline` functions**, shared by all front-end
  variants, with an aligned stack frame (see stage 2).
- **Check the assembly:** no spills in the hot loop, a high FMA share, and
  no unexpected gathers.

### 4c. Streaming (non-temporal) stores
Use them only when the output is **write-only, written in full cache lines,
and larger than the LLC**. They avoid the read-for-ownership and cache
pollution.

- Stage results in an aligned L1 tile (for example 256 doubles). Stream-copy
  full, 64-byte-aligned tiles, and use regular stores for tails and
  unaligned cases.
- One store fence (`sfence` on x86) per thread at the end of the parallel
  region.
- Expose it as a separate `*_nt` variant, compiled only when the ISA
  supports it.

Observed behaviour:

| Situation | Effect |
|---|---|
| Output ~2x LLC, write-only, few input streams | **~1.3x faster**, reached measured DRAM peak |
| Output large, but many input streams dominate traffic | ~1.0x |
| Output fits in the LLC | **~4x slower**: every write is forced to DRAM |

**Never use streaming stores when the output fits in the LLC, or when it is
re-read soon afterwards.**

## Stage 5: Validate and report

1. All correctness checks pass for every variant. Compare checksums across
   builds only when the reference checksums match; otherwise the inputs
   differ.
2. Report one row per `kernel / variant / tuning` with baseline time, new
   time, speedup, GB/s and GFLOP/s. Compare non-default variants against the
   baseline's *default*.
3. State explicitly:
   - the algorithmic gain versus the intrinsics gain;
   - which cases are cache resident;
   - which throughput numbers are nominal and exceed peak because of a
     reformulation.
4. Record lessons (pitfalls, flags, machine facts) in persistent notes for
   the next agent.

## Integrating into a benchmark suite

- Add optimizations as **named variants or tunings** next to an untouched
  reference default. Use compile-time dispatch (`template <size_t idx>` +
  `if constexpr`) so the reference code stays byte-for-byte unchanged.
- Register ISA-specific variants only when the ISA is available.
- Put shared SIMD helpers in one header, not copied per kernel.
- The comparison driver builds the baseline and the optimized tree with
  identical flags, runs them interleaved, and emits a markdown/CSV table.
  Bash arrays are not exported to child processes, so pass lists as strings.
- Useful commands:
  ```bash
  # hardware FLOPs on one pinned core; subtract a 1x-reps run from a 3x-reps run to remove setup cost
  OMP_NUM_THREADS=1 taskset -c 7 perf stat -x, -e cycles,fp_arith_inst_retired.512b_packed_double,... <exe> <args>
  # vectorization remarks / asm for the target ISA, from any host
  clang++ <includes> -O3 -march=<target> -mprefer-vector-width=512 -ffast-math -Rpass-missed=loop-vectorize -S file.cpp
  ```
