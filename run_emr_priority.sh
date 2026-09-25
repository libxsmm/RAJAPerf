#!/bin/bash
#
# Compare a baseline branch (default: develop) against the vectorized branch
# for the 10 priority Apps kernels on one 64-core Intel Xeon (Emerald Rapids)
# node, and write a bandwidth (GB/s) / compute (GFLOP/s) summary.
#
# Usage:
#   ./run_emr_priority.sh all      # build, allocate node (salloc), run, report
#   ./run_emr_priority.sh build    # configure + build both source trees
#   ./run_emr_priority.sh run      # run; must be called inside an allocation
#   ./run_emr_priority.sh report   # write $WORK/results/summary.{md,csv}
#
# Environment overrides:
#   WORK         work directory                 (default: <repo>/emr_priority)
#   MAIN_REF     baseline git ref               (default: develop)
#   VEC_REF      vectorized git ref             (default: vectorized)
#   MAIN_SRC     baseline source tree           (default: git worktree of MAIN_REF)
#   VEC_SRC      vectorized source tree         (default: git worktree of VEC_REF)
#   CXX, CC      compilers                      (default: clang++, clang)
#   CXX_FLAGS    compile flags  (default: -O3 -march=sapphirerapids -mprefer-vector-width=512 -ffast-math)
#   MAIN_EXTRA_FLAGS, VEC_EXTRA_FLAGS  flags appended for one build only (default: none),
#                e.g. MAIN_SRC=\$VEC_SRC MAIN_EXTRA_FLAGS=-DRAJAPERF_FEM_NO_INTRINSICS
#                compares the compiler-vectorized and intrinsics code of one tree
#   MAIN_LABEL, VEC_LABEL  column labels in the report (default: main, vec)
#   SALLOC_ARGS  allocation     (default: --partition=emr --nodes=1 --time=03:59:00 --constraint=ddr5600)
#   NTHREADS     OpenMP threads (default: 64, pinned to cores 0..NTHREADS-1)
#   NREP         interleaved main/vec repetitions (default: 3)
#   NPASSES      RAJAPerf --npasses per run       (default: 3)
#   FORCE_BUILD  rebuild even if binaries exist: 1, main or vec (default: 0)
#   KERNELS      space-separated subset of the kernels below (default: all 10)
#   RESULTS      results directory                (default: $WORK/results)
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$REPO/$(basename "${BASH_SOURCE[0]}")"
WORK="${WORK:-$REPO/emr_priority}"
MAIN_REF="${MAIN_REF:-develop}"
VEC_REF="${VEC_REF:-vectorized}"
MAIN_SRC="${MAIN_SRC:-$WORK/src-main}"
VEC_SRC="${VEC_SRC:-$WORK/src-vec}"
CXX="${CXX:-clang++}"
CC="${CC:-clang}"
CXX_FLAGS="${CXX_FLAGS:--O3 -march=sapphirerapids -mprefer-vector-width=512 -ffast-math}"
MAIN_EXTRA_FLAGS="${MAIN_EXTRA_FLAGS:-}"
VEC_EXTRA_FLAGS="${VEC_EXTRA_FLAGS:-}"
MAIN_LABEL="${MAIN_LABEL:-main}"
VEC_LABEL="${VEC_LABEL:-vec}"
SALLOC_ARGS="${SALLOC_ARGS:---partition=emr --nodes=1 --time=03:59:00 --constraint=ddr5600}"
NTHREADS="${NTHREADS:-64}"
NREP="${NREP:-3}"
NPASSES="${NPASSES:-3}"
FORCE_BUILD="${FORCE_BUILD:-0}"
RESULTS="${RESULTS:-$WORK/results}"

ALL_KERNELS="EDGE3D DIFFUSION3DPA ENERGY FEMSWEEP INTSC_HEXRECT MASS3DEA
             MASS3DPA_ATOMIC MASSVEC3DPA NODAL_ACCUMULATION_3D VOL3D"
# a separate array name keeps the exported KERNELS string visible to 'salloc ... run'
read -r -a KERNEL_LIST <<< "$(echo ${KERNELS:-$ALL_KERNELS})"

# Problem sizes: every OpenMP run takes >= ~0.5 ms per rep on 64 cores, and the
# streaming FEM/hydro kernels (DIFFUSION3DPA, MASSVEC3DPA, MASS3DPA_ATOMIC,
# ENERGY) have working sets well beyond the 320 MB L3 of a socket.
declare -A SIZE=(
  [EDGE3D]=8000000                 # 200^3 zones
  [DIFFUSION3DPA]=27000000         # 1M elements (27 dofs each)
  [ENERGY]=16000000
  [FEMSWEEP]=10368000              # 72 angles x 16 groups x 9261 elements
  [INTSC_HEXRECT]=4000000
  [MASS3DEA]=33554432              # 8192 elements (4096 matrix entries each)
  [MASS3DPA_ATOMIC]=27000000       # DOFs
  [MASSVEC3DPA]=81000000           # 1M elements (81 dofs each)
  [NODAL_ACCUMULATION_3D]=8000000
  [VOL3D]=8000000
)

# Kernels whose OpenMP variants differ between the two builds.
CHANGED="${CHANGED:-EDGE3D DIFFUSION3DPA FEMSWEEP MASS3DEA MASS3DPA_ATOMIC MASSVEC3DPA NODAL_ACCUMULATION_3D}"

PERF_VARIANTS="Base_OpenMP Lambda_OpenMP RAJA_OpenMP"
CHECK_VARIANTS="Base_Seq $PERF_VARIANTS"

make_src() {
  local dir=$1 ref=$2
  if [[ ! -d "$dir" ]]; then
    git -C "$REPO" worktree add --detach "$dir" "$ref"
    git -C "$dir" submodule update --init --recursive
  fi
}

build_one() {
  local name=$1 src=$2 extra=$3
  local bdir="$WORK/build-$name"
  if [[ "$FORCE_BUILD" != "1" && "$FORCE_BUILD" != "$name" && -x "$bdir/bin/raja-perf.exe" ]]; then
    echo "[$name] using existing $bdir/bin/raja-perf.exe"
    return
  fi
  mkdir -p "$bdir"
  echo "[$name] configuring $src ($(git -C "$src" rev-parse --short HEAD))"
  cmake -S "$src" -B "$bdir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_CXX_FLAGS="$CXX_FLAGS${extra:+ $extra}" \
    -DENABLE_OPENMP=On -DENABLE_TESTS=Off \
    -DRAJA_ENABLE_TESTS=Off -DRAJA_ENABLE_EXAMPLES=Off -DRAJA_ENABLE_EXERCISES=Off \
    > "$bdir/cmake.log" 2>&1
  echo "[$name] building"
  cmake --build "$bdir" --target raja-perf.exe -j "$(nproc)" > "$bdir/make.log" 2>&1
}

do_build() {
  mkdir -p "$WORK"
  make_src "$MAIN_SRC" "$MAIN_REF"
  make_src "$VEC_SRC" "$VEC_REF"
  build_one main "$MAIN_SRC" "$MAIN_EXTRA_FLAGS"
  build_one vec "$VEC_SRC" "$VEC_EXTRA_FLAGS"
  {
    echo "main: $MAIN_SRC $(git -C "$MAIN_SRC" describe --always --dirty --abbrev=12) $MAIN_EXTRA_FLAGS"
    echo "vec:  $VEC_SRC $(git -C "$VEC_SRC" describe --always --dirty --abbrev=12) $VEC_EXTRA_FLAGS"
    echo "CXX:  $("$CXX" --version | head -1)"
    echo "CXX_FLAGS: $CXX_FLAGS"
  } > "$WORK/build-info.txt"
}

do_run() {
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "error: 'run' must be executed inside a Slurm allocation" >&2
    exit 1
  fi
  export OMP_NUM_THREADS="$NTHREADS"
  export GOMP_CPU_AFFINITY="0-$((NTHREADS-1))"
  mkdir -p "$RESULTS"
  cp "$WORK/build-info.txt" "$RESULTS/" 2>/dev/null || true
  srun bash -c 'hostname; lscpu | grep -E "Model name|^CPU\(s\)|NUMA node[0-9]"' \
    > "$RESULTS/node-info.txt"
  cat "$RESULTS/node-info.txt"

  # Correctness: one rep per kernel with Base_Seq as checksum reference.
  for b in main vec; do
    for k in "${KERNEL_LIST[@]}"; do
      local out="$RESULTS/check/$b/$k"
      mkdir -p "$out"
      srun "$WORK/build-$b/bin/raja-perf.exe" --kernels "$k" --variants $CHECK_VARIANTS \
        --size "${SIZE[$k]}" --checkrun 1 --outdir "$out" > "$out/stdout.txt" 2>&1 \
        || echo "check $b $k: raja-perf.exe exited with $?"
      # informational only (files may lag on NFS); 'report' does the real check
      echo "check $b $k: $(grep -sc PASSED "$out/RAJAPerf-checksum.txt" || true) passed," \
           "$(grep -sc FAILED "$out/RAJAPerf-checksum.txt" || true) failed"
    done
  done

  # Performance: main and vec interleaved to spread node noise over both.
  for rep in $(seq 1 "$NREP"); do
    for b in main vec; do
      for k in "${KERNEL_LIST[@]}"; do
        local out="$RESULTS/perf/$b/$k/run$rep"
        mkdir -p "$out"
        local st=0
        srun "$WORK/build-$b/bin/raja-perf.exe" --kernels "$k" --variants $PERF_VARIANTS \
          --size "${SIZE[$k]}" --npasses "$NPASSES" --outdir "$out" > "$out/stdout.txt" 2>&1 || st=$?
        echo "perf run $rep $b $k: exit $st"
      done
    done
  done
}

do_report() {
  RESULTS="$RESULTS" CHANGED="$CHANGED" KERNELS="${KERNEL_LIST[*]}" \
  MAIN_LABEL="$MAIN_LABEL" VEC_LABEL="$VEC_LABEL" \
  NTHREADS="$NTHREADS" NREP="$NREP" NPASSES="$NPASSES" python3 - << 'EOF'
import csv, glob, os, re, statistics

R = os.environ["RESULTS"]
kernels = os.environ["KERNELS"].split()
changed = set(os.environ["CHANGED"].split())
GIB_TO_GB = 1024**3 / 1e9
LM, LV = os.environ["MAIN_LABEL"], os.environ["VEC_LABEL"]

def vkey(variant, tuning):
    return variant if tuning == "default" else f"{variant}/{tuning}"

def run_data(path):
    d = {}
    for r in csv.reader(open(path)):
        r = [c.strip() for c in r]
        if len(r) >= 8 and r[0].startswith("Apps_"):
            try:
                d[vkey(r[1], r[2])] = dict(size=int(r[3]), status=r[4], t=float(r[5]),
                                           gbs=float(r[6]) * GIB_TO_GB, gflops=float(r[7]))
            except ValueError:
                pass
    return d

def checksums(path):
    d = {}
    for line in open(path):
        m = re.match(r"^(\w+)-(\S+)\s+(PASSED|FAILED)\s+(\S+)\s+(\S+)", line)
        if m:
            d[vkey(m.group(1), m.group(2))] = (m.group(3), float(m.group(4)), float(m.group(5)))
    return d

rows, problems = [], []
for k in kernels:
    ok = True
    for b in ("main", "vec"):
        for f in glob.glob(f"{R}/check/{b}/{k}/RAJAPerf-checksum.txt") + \
                 glob.glob(f"{R}/perf/{b}/{k}/run*/RAJAPerf-checksum.txt"):
            for var, (st, _, _) in checksums(f).items():
                if st != "PASSED":
                    ok = False
                    problems.append(f"{k} {b} {var}: {st} ({f})")
    cm = checksums(f"{R}/check/main/{k}/RAJAPerf-checksum.txt")
    cv = checksums(f"{R}/check/vec/{k}/RAJAPerf-checksum.txt")
    rel = None
    # cross-build comparison is only meaningful if both builds use the same inputs
    same_ref = ("Base_Seq" in cm and "Base_Seq" in cv and
                abs(cm["Base_Seq"][2] - cv["Base_Seq"][2]) <= cm["Base_Seq"][1] * abs(cm["Base_Seq"][2]))
    if same_ref and "Base_OpenMP" in cm and "Base_OpenMP" in cv:
        tol, c0, c1 = cm["Base_OpenMP"][1], cm["Base_OpenMP"][2], cv["Base_OpenMP"][2]
        rel = abs(c1 - c0) / abs(c0) if c0 != 0 else abs(c1)
        if rel > tol:
            ok = False
            problems.append(f"{k}: main vs vec Base_OpenMP checksum rel diff {rel:.2e} > tol {tol:.1e}")
    per = {b: [run_data(f) for f in sorted(glob.glob(f"{R}/perf/{b}/{k}/run*/RAJAPerf-kernel-run-data.csv"))]
           for b in ("main", "vec")}
    vec_keys = set().union(*per["vec"]) if per["vec"] else set()
    for base in ("Base_OpenMP", "Lambda_OpenMP", "RAJA_OpenMP"):
      # non-default tunings are compared against the baseline's default tuning
      for var in [base] + sorted(x for x in vec_keys if x.startswith(base + "/")):
        mvar = var if any(var in d for d in per["main"]) else base
        m = [d[mvar] for d in per["main"] if mvar in d]
        v = [d[var] for d in per["vec"] if var in d]
        if not m or not v:
            continue
        med = lambda xs, key: statistics.median(x[key] for x in xs)
        rows.append(dict(kernel=k, variant=var, size=m[0]["size"],
                         changed="yes" if k in changed else "no",
                         t_main=med(m, "t"), t_vec=med(v, "t"),
                         gbs_main=med(m, "gbs"), gbs_vec=med(v, "gbs"),
                         gf_main=med(m, "gflops"), gf_vec=med(v, "gflops"),
                         speedup=med(m, "t") / med(v, "t"),
                         checks="PASS" if ok else "FAIL",
                         xdiff=(f"{rel:.1e}" if rel is not None else
                                "-" if same_ref else "inputs differ"),
                         n=f"{len(m)}/{len(v)}"))

fields = ["kernel", "variant", "size", "changed", "t_main", "t_vec", "gbs_main", "gbs_vec",
          "gf_main", "gf_vec", "speedup", "checks", "xdiff", "n"]
with open(f"{R}/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

def ftime(t):
    return f"{t*1e3:.3f} ms" if t >= 1e-3 else f"{t*1e6:.1f} us"

node = open(f"{R}/node-info.txt").read().strip() if os.path.exists(f"{R}/node-info.txt") else ""
build = open(f"{R}/build-info.txt").read().strip() if os.path.exists(f"{R}/build-info.txt") else ""
with open(f"{R}/summary.md", "w") as f:
    f.write(f"# RAJAPerf priority Apps kernels: {LM} vs {LV}\n\n")
    f.write(f"OMP_NUM_THREADS={os.environ['NTHREADS']}, GOMP_CPU_AFFINITY=0-{int(os.environ['NTHREADS'])-1}; "
            f"median of {os.environ['NREP']} interleaved runs x {os.environ['NPASSES']} passes. "
            "Bandwidth in GB/s (1e9 B/s) and GFLOP/s as computed by RAJAPerf from its "
            f"per-kernel byte and FLOP counts. 'changed' = OpenMP code differs between {LM} and {LV}.\n\n")
    f.write("```\n" + node + "\n" + build + "\n```\n\n")
    f.write(f"| Kernel | Variant | Size | changed | t/rep {LM} | t/rep {LV} | GB/s {LM} | GB/s {LV} "
            f"| GFLOP/s {LM} | GFLOP/s {LV} | Speedup | Checks | {LM}-vs-{LV} rel diff |\n")
    f.write("|---|---|---:|:-:|---:|---:|---:|---:|---:|---:|---:|:-:|---:|\n")
    for r in rows:
        f.write(f"| {r['kernel']} | {r['variant']} | {r['size']} | {r['changed']} "
                f"| {ftime(r['t_main'])} | {ftime(r['t_vec'])} "
                f"| {r['gbs_main']:.1f} | {r['gbs_vec']:.1f} "
                f"| {r['gf_main']:.1f} | {r['gf_vec']:.1f} "
                f"| {r['speedup']:.2f}x | {r['checks']} | {r['xdiff']} |\n")
    f.write("\n" + ("All checks passed.\n" if not problems else
                    "Problems:\n" + "\n".join(f"- {p}" for p in problems) + "\n"))
print(open(f"{R}/summary.md").read())
EOF
}

case "${1:-all}" in
  build)  do_build ;;
  run)    do_run ;;
  report) do_report ;;
  all)
    do_build
    # shellcheck disable=SC2086
    salloc $SALLOC_ARGS "$SCRIPT" run
    do_report
    ;;
  *) echo "usage: $0 [all|build|run|report]" >&2; exit 1 ;;
esac
