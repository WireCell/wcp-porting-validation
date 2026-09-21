#!/bin/bash
# doc sbnd_xin/119 round 0 -- gperftools CPU / jemalloc heap profile of the SBND PR chain
# AS PRODUCTION ACTUALLY RUNS IT.
#
# Forked BY DUPLICATION from scripts/perf/profile_pr11.sh, which stays untouched (M10, M13:
# it is the record of doc pr/11).  The one thing that differs is the one thing that matters:
#
#   profile_pr11.sh HARDCODES pipeline_names as 13 stages.  Production runs 15 -- the list
#   predates ClusteringProtectBundle:pr and CreateSteinerGraph:prrefresh, and it predates the
#   doc-118 flip entirely.  Profiling with it would attribute a job that nobody runs.
#
# Instead this script profiles the arm's OWN precompiled config, `.wct-cfg-evt<ID>.json`, which
# run_pr_chain_batch.sh leaves in every PR dir.  That file IS production by construction: it is
# the exact JSON wire-cell was handed when the arm was produced.  No TLA list to keep in sync.
#
# ---------------------------------------------------------------------------------------------
# THE DESTRUCTIVE TRAP THIS SCRIPT EXISTS TO AVOID
# ---------------------------------------------------------------------------------------------
# That config carries ABSOLUTE output paths back into the arm directory it was compiled for:
#   SbndPrMagnifyTrackingVisitor:pr  output_filename -> <armdir>/tracking-pr.root
#   UbooneTaggerOutputVisitor:pr     output_filename -> <armdir>/tracking-pr.root
#   PrDisplayDump:pr                 output_filename -> <armdir>/calib-pr-evt<ID>.json
#   TensorFileSink:clus_pr           outname         -> <armdir>/pctree-pr-evt<ID>.tar.gz
#   MultiAlgBlobClustering:clus_pr   (bee zip)       -> <armdir>/mabc-pr.zip
# Running `wire-cell -c` on it in place would OVERWRITE the products that doc 118's G1 output
# gate compared -- for a flip that is already pushed and in production.  That is exactly the
# M13 scientific-record overwrite.  So: copy the JSON to scratch, rewrite the arm prefix to the
# scratch dir, and REFUSE to run if any write path still points inside the arm.
#
# The TensorFileSource `inname` also lives under a work- dir (the stage-A pctree).  That one is
# a READ and must be preserved -- hence the guard is "no arm-dir path survives", not "no work-
# path survives", and it prints the surviving input for the record.
# ---------------------------------------------------------------------------------------------
#
# Usage:
#   PRDIR=<abs path to a pr_evt<ID> dir> OUTDIR=<scratch dir> [MODE=cpu|heap|glibc] \
#     ./profile_pr118.sh
#
#   MODE=cpu    (default) gperftools CPUPROFILE under libtcmalloc_and_profiler
#   MODE=heap   jemalloc sampling (MALLOC_CONF prof) -- doc pdvd/28 sec 0 measured the
#               gperftools HEAPPROFILE path ~25x slower; every heap study in this tree uses
#               jemalloc + pdhd/stm/perf/je2pprof.py.  NOTE: jemalloc REPLACES tcmalloc, so a
#               heap run is not the production allocator either; it is a heap-composition
#               instrument, not a wall-clock one.
#   MODE=tcmalloc  libtcmalloc_minimal only -- the clean A/B arm for the round-1 lever.
#   MODE=glibc  no allocator preload at all -- the control that says what tcmalloc is worth
#               on this chain.  doc sbnd_xin/118 sec 6: SBND's PR runners do NOT preload
#               tcmalloc (they assign LD_PRELOAD="$PYLIB"), so MODE=glibc is the arm's REAL
#               allocator and MODE=cpu is the counterfactual.  Read every cpu-mode number with
#               that caveat.
#
# Env: PROFLIB, FREQ (CPUPROFILE_FREQUENCY, default 250), DL (default on: the arm's config
#      already carries its dl_weights, we do not second-guess it).
#
# M17: the config is ALREADY compiled, so gojsonnet never runs under SIGPROF.  Never run this
# under `setarch -R` -- the profiler dies at startup with ASLR off.
set -eu

PRDIR=${PRDIR:?set PRDIR to an absolute pr_evt<ID> directory}
OD=${OUTDIR:?set OUTDIR to a scratch directory}
MODE=${MODE:-cpu}

PRDIR=$(cd "$PRDIR" && pwd -P)
case "$PRDIR" in
    */pr_evt*) : ;;
    *) echo "ERROR: PRDIR does not look like a pr_evt<ID> dir: $PRDIR" >&2; exit 1 ;;
esac
EVT=${PRDIR##*/pr_evt}
CFG_IN="$PRDIR/.wct-cfg-evt$EVT.json"
[ -s "$CFG_IN" ] || { echo "ERROR: no precompiled config at $CFG_IN" >&2; exit 1; }

mkdir -p "$OD"
OD=$(cd "$OD" && pwd -P)
CFG="$OD/prof_evt$EVT.json"

# --- rewrite every arm-dir reference to the scratch dir -------------------------------------
python3 - "$CFG_IN" "$CFG" "$PRDIR" "$OD" <<'PYEOF'
import json, sys
src, dst, armdir, scratch = sys.argv[1:5]
text = open(src).read()
n = text.count(armdir)
text = text.replace(armdir, scratch)
open(dst, 'w').write(text)

# Guard: no write path may still point into the arm.  Walk the parsed config and fail loudly.
cfg = json.loads(text)
bad = []
inputs = []
WRITE_HINTS = ('out', 'dump', 'save', 'zip', 'sink')
for i, node in enumerate(cfg):
    data = node.get('data')
    if not isinstance(data, dict):
        continue
    for k, v in data.items():
        if not isinstance(v, str) or armdir not in v:
            continue
        bad.append(f"[{i}] {node.get('type','?')}:{node.get('name','')} {k} = {v}")
    for k, v in data.items():
        if isinstance(v, str) and '/work-' in v and 'sbnd_xin' in v:
            kind = 'WRITE' if any(h in k.lower() for h in WRITE_HINTS) else 'read'
            inputs.append(f"  {kind}  [{i}] {node.get('type','?')}:{node.get('name','')} {k} = {v}")
if bad:
    print("ERROR: arm-dir path survived the rewrite -- refusing to run:", file=sys.stderr)
    for b in bad:
        print("   " + b, file=sys.stderr)
    sys.exit(2)
print(f"rewrote {n} arm-dir reference(s) -> {scratch}")
print("paths still under sbnd_xin/work- (these must all be reads):")
for s in inputs or ["  <none>"]:
    print(s)
# A WRITE under work- after the rewrite means some OTHER arm would be clobbered.
if any(s.strip().startswith('WRITE') for s in inputs):
    print("ERROR: a write path still targets a work- arm -- refusing to run", file=sys.stderr)
    sys.exit(2)
PYEOF

PROFLIB=${PROFLIB:-/usr/lib/x86_64-linux-gnu/libtcmalloc_and_profiler.so.4}
PYLIB=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3.11.so.1.0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export WIRECELL_PATH=${WIRECELL_PATH:-/nfs/data/1/xqian/toolkit-dev/toolkit/cfg:/nfs/data/1/xqian/toolkit-dev/wire-cell-data}
LOG="$OD/prof_evt$EVT.$MODE.log"
TIMECMD=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/abtest/timecmd.py
# ORDERING IS LOAD-BEARING: timecmd.py OUTSIDE, the profiling env INSIDE, on wire-cell alone.
# The first version of this script had them the other way round.  timecmd.py is a python process
# that fork/execs the child, so with LD_PRELOAD + CPUPROFILE in its own environment it linked
# libtcmalloc_and_profiler too, profiled ITSELF, and -- exiting after its child -- wrote its own
# profile over wire-cell's at the same CPUPROFILE path.  The symptom is two "PROFILE:
# interrupts/evictions/bytes" lines in the log, the second tiny, and a .prof holding ~5 samples of
# python import machinery.  A profile that small is obviously wrong; one merely truncated would
# not have been.  Same hazard for MALLOC_CONF in heap mode.

# The arm's own runner preloads libpython (RTLD_GLOBAL) so the SCN vertex net can import.
# Keep it in every mode, or the job silently falls back to the geometric vertex and profiles a
# different pipeline (doc sbnd_xin/118 sec 8).
cd "$OD"
case "$MODE" in
  cpu)
    OUT="$OD/pr118_evt$EVT.prof"
    rc=0
    python3 "$TIMECMD" "$OD/prof_evt$EVT.$MODE.time.meta" \
        env LD_PRELOAD="$PYLIB:$PROFLIB" CPUPROFILE="$OUT" CPUPROFILE_FREQUENCY="${FREQ:-250}" \
        wire-cell -l stderr -l "$LOG:debug" -L debug -c "$CFG" || rc=$?
    echo "rc=$rc"
    echo "profile -> $OUT"
    echo "view: google-pprof --text --cum \$(which wire-cell) $OUT | head -60"
    ;;
  heap)
    OUT="$OD/pr118_evt$EVT.je"
    rc=0
    python3 "$TIMECMD" "$OD/prof_evt$EVT.$MODE.time.meta" \
        env LD_PRELOAD="$PYLIB:/usr/lib/x86_64-linux-gnu/libjemalloc.so.2" \
        MALLOC_CONF="prof:true,prof_active:true,lg_prof_sample:19,prof_prefix:$OUT,prof_final:true" \
        wire-cell -l stderr -l "$LOG:debug" -L debug -c "$CFG" || rc=$?
    echo "rc=$rc"
    echo "heap -> ${OUT}.*.heap"
    echo "view: je2pprof.py then google-pprof --text --inuse_space"
    ;;
  tcmalloc)
    # The CLEAN allocator A/B arm.  MODE=cpu preloads libtcmalloc_AND_PROFILER and arms SIGPROF,
    # which costs wall time of its own -- so a cpu-vs-glibc wall comparison measures the profiler
    # as much as the allocator.  This mode preloads libtcmalloc_minimal only: exactly what PDHD,
    # PDVD and SBND's own run_clus_evt.sh do, and exactly what the round-1 lever would ship.
    rc=0
    python3 "$TIMECMD" "$OD/prof_evt$EVT.$MODE.time.meta" \
        env LD_PRELOAD="$PYLIB:/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4" \
        wire-cell -l stderr -l "$LOG:debug" -L debug -c "$CFG" || rc=$?
    echo "rc=$rc"
    echo "tcmalloc arm -> $OD/prof_evt$EVT.$MODE.time.meta"
    ;;
  glibc)
    OUT="$OD/pr118_evt$EVT.glibc"
    rc=0
    python3 "$TIMECMD" "$OD/prof_evt$EVT.$MODE.time.meta" \
        env LD_PRELOAD="$PYLIB" \
        wire-cell -l stderr -l "$LOG:debug" -L debug -c "$CFG" || rc=$?
    echo "rc=$rc"
    echo "glibc control run (no allocator preload) -> $OD/prof_evt$EVT.$MODE.time"
    ;;
  *) echo "ERROR: unknown MODE=$MODE (cpu|heap|tcmalloc|glibc)" >&2; exit 1 ;;
esac
