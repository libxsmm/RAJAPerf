#!/bin/bash
# Per-batch execution port, load/store and divider counts of EDGE3D OpenMP
# tunings on one pinned core (Intel Sapphire/Emerald Rapids perf events).
# Counts are the difference of --repfact 3 and 1 runs, i.e. without setup.
#
# usage: edge_ports.sh SIZE tuning...
#   e.g. RAJAPERF_EXE=build/bin/raja-perf.exe edge_ports.sh 1000000 metric metric_rb
# env:   RAJAPERF_EXE (default ./bin/raja-perf.exe), CPU (default 7)
S=$1; shift
X=$(realpath "${RAJAPERF_EXE:-./bin/raja-perf.exe}")
CPU=${CPU:-7}
G1=cycles,instructions,uops_dispatched.port_0,uops_dispatched.port_1,uops_dispatched.port_5_11,uops_dispatched.port_6,arith.div_active,fp_arith_inst_retired.512b_packed_double
G2=cycles,uops_issued.any,uops_dispatched.port_2_3_10,uops_dispatched.port_4_9,uops_dispatched.port_7_8,mem_inst_retired.all_loads,mem_inst_retired.all_stores,exe_activity.bound_on_loads
WD=$(mktemp -d)
trap 'rm -rf "$WD"' EXIT
cd "$WD"
for t in "$@"; do
  nv=1; [[ $t == *_x2 ]] && nv=2; [[ $t == *_x4 ]] && nv=4
  for g in 1 2; do
    ev=G$g
    for rf in 1 3; do
      OMP_NUM_THREADS=1 taskset -c "$CPU" perf stat -x, -e "${!ev}" -o ep.$g.$rf.perf -- \
        "$X" --kernels EDGE3D --variants Base_OpenMP --tunings "$t" --size "$S" \
        --repfact $rf --warmup-disable --outdir ep_$rf > /dev/null 2>&1
    done
  done
  python3 - "$t" $nv << 'EOF'
import sys, csv
t, nv = sys.argv[1], int(sys.argv[2])
def rd(f):
    out = {}
    for l in open(f):
        if l.strip() and not l.startswith('#'):
            p = l.split(',')
            out[p[2]] = float(p[0])
    return out
def det(d):
    rows = [r for r in csv.reader(open(d + '/RAJAPerf-kernel-details.csv')) if len(r) > 3]
    h = [c.strip() for c in rows[0]]
    r = [x for x in rows if x[0].strip() == 'Apps_EDGE3D'][0]
    return dict(zip(h, [c.strip() for c in r]))
d1, d3 = det('ep_1'), det('ep_3')
reps = int(d3['Reps']) - int(d1['Reps'])
nb = reps * int(d1['Iterations/rep']) / (8 * nv)
c = {}
for g in (1, 2):
    a, b = rd('ep.%d.1.perf' % g), rd('ep.%d.3.perf' % g)
    for k in a:
        c[k + ('' if g == 1 or k != 'cycles' else '2')] = (b[k] - a[k]) / nb
cyc = c['cycles']
print("== %s: per batch of %d elements (%d batches)" % (t, 8 * nv, nb))
for k, v in c.items():
    print("  %-40s %9.1f  %5.2f /cyc" % (k, v, v / cyc))
EOF
done
