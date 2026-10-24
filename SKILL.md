---
name: hpc-kernel-optimization
description: 'Staged CPU optimization workflow for HPC loop kernels (finite element, stencil, streaming, geometry, many small dense solves). Use when: optimizing or vectorizing a compute kernel; deciding if a kernel is compute, memory, latency or cache bound; roofline analysis; AoS to SoA conversion or element batching; removing atomics; loop fusion; sum factorization; algebraic reformulation; GEMM-like register blocking of small matrix assembly; multi-vector batches for ILP; SIMD intrinsics (AVX-512, AVX2, SVE, NEON); forcing FMA; streaming or non-temporal stores; checking whether GFLOP/s or GB/s above peak is real; comparing an optimized branch against a baseline.'
---

# HPC Kernel Optimization Strategy

Work through the stages **in order**. Each stage is more complex and less
portable than the one before, so only move on when the earlier stages have
been exhausted or clearly cannot help. Measure after every stage. Keep the
reference implementation runnable so every change can be compared against it.

| Stage | What | Pays off most when | Cost / risk |
|---|---|---|---|
| 0 | Baseline, sensitive correctness checks, measurement setup | always (prerequisite) | low |
| 1 | Classify: compute, memory, latency or cache bound | always (decides the path) | low |
| 2 | Data layout for vectorization, remove serialization | code does not vectorize, uses strided/indirect access, or serializes on atomics | medium |
| 3 | Better algorithm: same result, less work or traffic | the kernel does redundant work or makes redundant passes over memory | medium to high |
| 4 | Intrinsics (FMA, register blocking, transposes), streaming stores | still compute bound after stages 2–3, or a store/gather inefficiency remains | high, ISA-specific |

Rules of thumb:
- **Expect the largest wins from stages 2 and 3.** Intrinsics on top of
  code the compiler already vectorizes well usually add a modest gain. They
  matter most for register-heavy dense contractions, where the compiler
  spills or misses FMA opportunities.
- **Memory-bound kernels already at DRAM peak gain nothing** from better
  vectorization. Only moving fewer bytes helps.
- **Removing atomics or serialization** can give order-of-magnitude gains
  when it applies, because it restores full parallelism.

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
| **Real DRAM traffic** | memory-controller counters (e.g. `perf stat -a -e uncore_imc/cas_count_read/,uncore_imc/cas_count_write/`), 3x-reps minus 1x-reps run, divided by reps | real bytes far above nominal means re-reads (cache blocking helps); near zero means cache resident. Measure the DRAM peak the same way with a STREAM triad that is many times the LLC: a triad only a few times the LLC can overstate it by ~30% |
| Achieved FLOP/s | FLOPs / time | compare to cores × FLOP/cycle × sustained SIMD clock |
| **Hardware FLOPs** | PMU counters, e.g. `perf stat -e fp_arith_inst_retired.{scalar,128b_packed,256b_packed,512b_packed}_double` (weights 1, 2, 4, 8; FMA already counts 2) | a nominal/hardware ratio above 1 means the nominal model overstates the real work |
| Vector share | wide-vector fraction of hardware FLOPs | low share means the code did not vectorize |
| Working set vs LLC | sum of touched array sizes | GB/s above DRAM peak means cache-resident, not a bandwidth win |
| Instruction mix per item | perf counts of loads, stores, FP instructions, divider cycles and per-port uops on one pinned core; subtract a 1x-reps run from a 3x-reps run and divide by items or batches | a high store/FP ratio means intermediates go through memory (register-block them); saturated FP ports with few loads/stores means only less work helps |
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
  - Stencils over 3D grids: if real reads exceed nominal, the neighbouring
    plane is re-read from DRAM. Tile the in-plane rows and walk the planes
    innermost so the next plane stays in L2.
  - Scattered per-item data (sweeps, unstructured meshes): software-prefetch
    the item a few steps ahead.
- **Load imbalance:** with T tasks on N threads and T mod N small, the last
  round leaves most threads idle. Run full rounds with one thread per task
  and the remainder with teams of threads per task.
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
   - **Branchy geometry (clipping, intersections) on regular meshes:**
     neighbouring items usually take the same branches (check: near-zero
     branch misses in the scalar code). Batch them across lanes with one
     shared control path: turn every branch condition into a lane mask and
     take the branch only if the mask is all-true or all-false. On a mixed
     mask, abandon the smallest unit of work (one facet, not the whole
     item), recompute it with the scalar code lane by lane, and add the
     results. Accumulate each unit into local sums that are merged only
     when the unit succeeds, so a partial SIMD result is never counted
     twice. Measure the fallback rate; coarse fallback (whole batch) can
     waste much of the gain.
4. **Compiler pitfalls to check for (seen with clang):**
   - Short lane loops (trip count = VL) get fully unrolled *before* the loop
     vectorizer runs, which leads to poor SLP code. Put
     `clang loop unroll(disable)` on the lane loop.
   - Conversely, fully unroll fixed-trip *outer* loops
     (`clang loop unroll(full)`) so table indices fold to constants.
   - **No stack realignment for wide-vector spills** in OpenMP outlined
     regions. This causes misaligned spill slots that split cache lines, and
     large timing differences between otherwise identical variants.
     Fix: move the batch body into a `noinline` function containing an
     `alignas(64)` local kept alive with `asm volatile("" :: "r"(ptr))`.
     This applies to plain C `omp simd` code with large `[..][VL]` arrays
     too.
   - Index macros that do not parenthesize their arguments break when passed
     expressions like `e0 + l`. Pass a plain variable.
   - `omp simd` lane loops must not store into escaped memory (a buffer
     passed to `asm volatile`, or a function parameter): the vectorizer
     gives up. Compute into a local temporary and copy it out in a
     separate loop.
   - Compile-time loops via lambdas (`static_for`): intrinsics code needs
     the lambdas forced inline (`__attribute__((always_inline))`), otherwise
     they are emitted out of line. In plain-C `omp simd` lane loops, forcing
     them inline can *break* vectorization ("too many memory checks"); leave
     normal inlining there.
5. **Verify.** Run the random-input harness, check the vectorization
   remarks, and inspect the assembly.

## Stage 3: More efficient algorithms (same result, less work)

Always look for this before writing intrinsics. For compute-bound kernels it
usually beats everything else. Patterns that generalize:

| Pattern | Recognize it by | Effect |
|---|---|---|
| **Loop fusion** | a chain of element-wise loops over the same index range, communicating through temporary arrays | one pass, temporaries stay in registers, traffic drops by the number of eliminated passes; keeping the operation order can make it bit-exact |
| **Sum factorization** | tensor-product structure, e.g. $\sum_q \prod_d B_{i_d q_d} B_{j_d q_d}\, D_q$ evaluated naively | contract one dimension at a time; work goes from $O(p^{2d} q^d)$ to roughly $O(p^{d+1} q^d)$; the saving grows with order and dimension |
| **Metric / algebraic reformulation** | every basis function or vector is transformed by the same Jacobian at each quadrature point | transform the *metric* once per point ($G = J^T J/\det J$, $H = \det J\, J^{-1}J^{-T}$) and contract with the untransformed reference basis; exploit its sparsity; each entry becomes a few FMAs |
| **Share work across outputs** | the same input is processed separately against several neighbouring targets (e.g. clipping one cell against each of its $2^d$ overlapping cells) | split once by the shared planes or boundaries so each piece goes to exactly one target; skip copies when a piece lies entirely on one side |
| **GEMM-like reordering of small matrix assembly** | a per-item matrix accumulated as $M_{pm} \mathrel{+}= \sum_q f(q,p,m)$ with the $q$ loop outermost, so every entry is loaded and stored once per $q$ | two phases: compute the few per-$q$ coefficients for all $q$ first, then contract over $q$ tile by tile with accumulators in registers (see 4c); stores drop from once per $q$ to once per entry |
| **Gather instead of scatter** | many-to-one accumulation with atomics | no atomics, full parallelism, vectorizable |
| **Batch independent problems** | many tiny identical solves | SIMD across problems |
| Symmetry | $M_{ij} = M_{ji}$ and similar | compute half |
| Hoisting / precompute | loop-invariant tables or factors | fewer operations per item |
| **Hoist into setup (mesh / data-structure precomputation)** | the kernel re-derives the same per-entity data on every sweep, repetition or outer index (angle, group, time step), e.g. a chain of dependent index loads `a[b[c[i]]]` or small geometric factors | build a compact table once at initialization and read it directly; shorter dependency chains and fewer instructions per item; see below |
| Recompute vs load | memory-bound kernel reading cheaply derivable data | less traffic, but this changes the benchmark input, so flag it |

Rules:
- **Keep the reference algorithm as the default** and add the new algorithm
  as a separately named variant or tuning, so baselines stay comparable.
- **Write the new algorithm in plain C first** and measure it. This separates
  the algorithmic gain from the intrinsics gain; the algorithm usually
  accounts for most of it.
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

### Hoisting dynamic work into initialization

Ask for every value the kernel computes per item: **does it depend on
anything that changes between sweeps?** Work that depends only on the
mesh or the data structures, not on the loop over angles, groups, time steps
or repetitions, can be computed once when the grid, mesh or data
structures are set up, and stored in a small table the kernel reads.

1. **Find the candidates.** For each per-item computation, list the indices
   it actually depends on. Typical finds:
   - chains of dependent index loads (connectivity → local numbering →
     global index), recomputed for every angle or group;
   - branch conditions and side/orientation flags derived from connectivity;
   - geometric factors (normals, Jacobians, face areas) of a static mesh.
   Profiling signs: L1-bound or latency-bound loads feeding a gather, and an
   instruction count per item far above the arithmetic it needs.
2. **Size the table before building it.** It has one entry per invariant
   entity (e.g. element × local face), not per item of the full iteration
   space. Compare it with the total footprint and the LLC: aim for well
   under 1% of the footprint, and ideally a size that stays cache-resident.
   - Pack it: narrow integer types, local (small-range) indices instead of
     global ones, flags folded into spare bits, and a record size that
     divides the cache line (e.g. 16 bytes) so an item's entries share few
     lines.
   - Store only what the kernel needs after the derivation (e.g. the final
     offsets), not the intermediate indices.
   - If the honest table would be large (per angle, per group), it is a
     traffic trade, not a hoist: on a memory-bound kernel it can be slower.
3. **Keep it general and safe.** The builder checks the assumptions the
   compact form relies on (index ranges fit, entries of one face belong to
   one neighbour, ...) and reports failure; the kernel then falls back to the
   original derivation. The original arrays stay the benchmark's inputs.
4. **Account for it.** Build it in setup, outside the timed region, and add
   its size to the reported allocated bytes. It changes the data structures
   the kernel reads, so make it a separately named tuning and state the
   footprint increase. Ask before doing it when the user owns the benchmark
   definition.
5. **Verify and measure.**
   - Checksums should be identical to the derivation it replaces.
   - Confirm the table path actually runs, since a fallback gives the same
     checksum. Use instruction counts per item or a debug counter.
   - Expect the gain to be largest on one core and to shrink at full thread
     count once the kernel reaches the bandwidth limit.

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
  the architectural register count (32 on AVX-512) to avoid spills. For
  small per-item matrices see 4c.
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

### 4c. GEMM-style register-blocked contractions

For kernels that build a small dense matrix per item (element matrices,
local operators) as a sum over quadrature points or samples $q$:

$$M_{pm} = \sum_q \Big( a_p(q)\, a_m(q)\, g(q) + b_p(q)^T H(q)\, b_m(q) \Big)$$

The natural loop order (for each $q$: update every $M_{pm}$) streams the
whole matrix through L1 once per $q$ and is store bound. Reorder it like a
GEMM, with $q$ as the reduction dimension $K$ and the batch of items in the
SIMD lanes:

1. **Phase 1, per-$q$ coefficients.** For every $q$, compute the few
   item-dependent terms ($g$, the entries of $H$, ...) for the whole batch
   and store them in a small aligned stack buffer `W[q][k][VL]`. The $q$
   iterations are independent, so this phase has plenty of ILP.
2. **Phase 2, tiled contraction.** Loop over output tiles; for each tile,
   zero the accumulators, loop over all $q$, and write the tile once at the
   end.
   - Per $q$ and tile, load only the coefficient terms that tile needs.
   - Item-independent factors ($a_p$, $b_p$, quadrature weights) come from
     a table and are **broadcast**, like the B operand of a GEMM
     micro-kernel.
   - Precompute per-column products (e.g. $H b_m$) once per $q$ and tile,
     then each entry update is a short FMA chain.
3. **Size the tile to the register file.** Accumulators plus per-column
   operands must fit (e.g. 4×4 = 16 accumulators + 8 operands on 32 zmm).
   With NV vectors per value (see 4d), shrink the tile to keep the register
   count constant, e.g. 4×(4/NV).
4. **Exploit symmetry** at compile time: visit only upper-triangle tiles and
   drop lower entries of diagonal tiles via constant-folded conditions
   (`if constexpr`, fully unrolled tile loops).
5. **Build item-independent tables once per run** (a struct constructed in
   setup), not per batch.
6. **Write it in plain C first** (`[k][VL]` arrays, `omp simd` lane loops),
   then in intrinsics, with the same structure.

Verify with the instruction-mix counters from stage 1: the store/FP ratio
should drop sharply and the kernel should end up FP-throughput bound.

### 4d. Multi-vector batches for ILP

When long dependency chains (divides, determinants, short reductions) leave
the FP units idle, hold each value as NV independent vector registers
(a small `vdx<NV>` struct with overloaded ops) and process VL×NV items per
batch. Each instruction then has NV independent copies in flight.

- Shrink register tiles by NV (see 4c) to avoid spills.
- **Do not fully unroll large loops at NV > 1.** The code size can exceed
  the L1i / decoded-uop cache and the kernel becomes front-end bound.
  Keep outer loops (e.g. over quadrature points) rolled, or make the unroll
  count depend on NV (`clang loop unroll_count(N)` accepts a constexpr).
- Share the batch prelude (gather with tail replication, common geometry)
  between kernels as an inlined helper templated on NV.
- Register as separate tunings (`_x2`, `_x4`) and measure; the best NV
  depends on the kernel and machine.
- Check the top-down breakdown first (e.g. `perf stat -M
  tma_memory_bound,tma_core_bound,tma_l1_bound,tma_divider`). When
  consecutive items are already independent, out-of-order execution overlaps
  them and NV > 1 only adds spills. If the kernel is L1-bound on chains of
  dependent index loads (`a[b[c[i]]]` feeding a gather), neither
  interleaving nor replacing the gather helps; only shortening the index
  chain does (see "Hoisting dynamic work into initialization").

### 4e. Streaming (non-temporal) stores
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

Expected behaviour:

| Situation | Effect |
|---|---|
| Output well above LLC size, write-only, few input streams | faster; can approach measured DRAM peak |
| Output large, but input streams dominate traffic | little or no change |
| Output fits in the LLC | **much slower**: every write is forced out to DRAM |

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
  - Turning the reference function into a template can still change its
    code: lambdas inside it lose internal linkage and with it the inliner's
    bonus, so a lambda loop may stop vectorizing (seen as a 2x slowdown).
    Check the vectorization remarks of the default, or keep the reference
    as a plain member function and put the new tunings in a separate
    template.
- Sense-reversing barriers that persist across repetitions must start each
  repetition from the barrier's current sense, not from 0; otherwise an odd
  number of waits per repetition lets threads pass or deadlock later.
- Register ISA-specific variants only when the ISA is available.
- Put shared SIMD helpers in one header, not copied per kernel.
- The comparison driver builds the baseline and the optimized tree with
  identical flags, runs them interleaved, and emits a markdown/CSV table.
  Bash arrays are not exported to child processes, so pass lists as strings.
- Submit long measurements as batch jobs writing to a log file; interactive
  `ssh ... srun` sessions with long silent stretches can drop and cancel the
  job. Do not edit a shell script while a job is executing it.
- Useful commands:
  ```bash
  # hardware FLOPs on one pinned core; subtract a 1x-reps run from a 3x-reps run to remove setup cost
  OMP_NUM_THREADS=1 taskset -c 7 perf stat -x, -e cycles,fp_arith_inst_retired.512b_packed_double,... <exe> <args>
  # vectorization remarks / asm for the target ISA, from any host
  clang++ <includes> -O3 -march=<target> -mprefer-vector-width=512 -ffast-math -Rpass-missed=loop-vectorize -S file.cpp
  ```
