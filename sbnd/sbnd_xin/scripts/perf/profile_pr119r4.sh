#!/bin/bash
# doc sbnd_xin/119 round 4 -- jemalloc HEAP profile of the SBND PR chain, at the PEAK and at exit.
#
# Fork by duplication (M10) of scripts/perf/profile_pr118.sh, which stays untouched: it is the
# record of round 0's CPU profiles (docs/119_figs/119_prof_*.txt).  Three things differ, and each
# one is why this file exists rather than an edit.
#
# 1. IT PINS THE BINARY.  profile_pr118.sh sets no LD_LIBRARY_PATH, so it profiles whatever
#    local/lib happens to hold.  For round 0 that was harmless -- one binary existed.  Round 3
#    built a second one, so an unpinned heap profile in round 4 would attribute memory to a
#    binary nobody can name afterwards (M1).  This script asserts the round-3 pin and records the
#    md5 the profiled PROCESS actually mapped, read back out of /proc/<pid>/maps, not just the
#    md5 of the file it intended to load.
#
# 2. IT DUMPS AT THE PEAK, NOT ONLY AT EXIT.  profile_pr118.sh's heap mode passes prof_final:true
#    alone, which dumps once, at exit.  That answers only half the question this round asks:
#
#      The MEM ladder shows CreateSteinerGraph:pr adding +1.2 to +1.4 GiB on the SBND tail events
#      and never giving it back.  "Never gives it back" in an RSS ladder has TWO readings -- the
#      memory is still LIVE, or it was freed and glibc kept the arena.  An exit-time dump
#      distinguishes them only if the answer is "live"; if the peak is transient, --inuse_space at
#      exit shows nothing and the round would conclude Steiner holds nothing, which is the inverse
#      error.
#
#    So a dump SERIES is taken rather than one dump, via lg_prof_interval (a dump every 2^N bytes
#    of cumulative allocation), and the series' MAX-live dump is read as the peak.  Peak dump vs
#    final dump is the live-vs-arena answer, and the peak dump carries the call sites worth
#    reading.
#
#    DO NOT USE prof_gdump HERE.  It is the obvious choice -- "dump on every VM high-water rise"
#    is exactly the trigger this round wants -- and it is unusable at this scale: on the 10-second
#    SBND MEDIAN event it produced 29 470 dumps and 5.1 GB, because in a job whose heap grows
#    monotonically nearly every extending allocation IS a new high-water mark.  It also silently
#    broke the script's own reporting, since `ls "$PFX".*.heap` then exceeded ARG_MAX and reported
#    zero dumps while 5 GB sat on disk.  lg_prof_interval bounds the series by construction; the
#    counting below uses find, not a glob, so a large series can never read as an empty one again.
#
# 3. IT IS HEAP-ONLY.  Nothing here measures time.  jemalloc REPLACES the process allocator, so
#    this run is neither the glibc that SBND production actually uses nor the tcmalloc PDHD/PDVD
#    use.  It is a heap-COMPOSITION instrument.  Never quote a wall or RSS number from it, and
#    never compare its RSS across detectors -- the allocator differs (doc 119 sec 9.3 records the
#    round-3 RSS claim that this exact confound already broke once).
#
# Usage:
#   PRDIR=<abs path to a pr_evt<ID> dir> OUTDIR=<scratch dir> ./profile_pr119r4.sh
#
# Env: LIBSNAP (default the round-3 pin), LG_SAMPLE (lg_prof_sample, default 19 = 512 KiB),
#      LG_INTERVAL (lg_prof_interval, default 28 = a dump per 256 MiB allocated).
#
# M17: the config is ALREADY compiled, so gojsonnet never runs under a profiling signal.  Never
# run this under `setarch -R`.
set -eu

PRDIR=${PRDIR:?set PRDIR to an absolute pr_evt<ID> directory}
OD=${OUTDIR:?set OUTDIR to a scratch directory}

PRDIR=$(cd "$PRDIR" && pwd -P)
case "$PRDIR" in
    */pr_evt*) : ;;
    *) echo "ERROR: PRDIR does not look like a pr_evt<ID> dir: $PRDIR" >&2; exit 1 ;;
esac
EVT=${PRDIR##*/pr_evt}
CFG_IN="$PRDIR/.wct-cfg-evt$EVT.json"
[ -s "$CFG_IN" ] || { echo "ERROR: no precompiled config at $CFG_IN" >&2; exit 1; }

R3_CLUS_MD5=4ff75274e43e766eaec4eef8b6107ae7
LIBSNAP=${LIBSNAP:-$HOME/tmp/d119r3-libpin}
[ -e "$LIBSNAP/libWireCellClus.so" ] || { echo "ERROR: no pin at $LIBSNAP" >&2; exit 1; }
CLUS_MD5=$(md5sum "$LIBSNAP/libWireCellClus.so" | cut -d' ' -f1)
[ "$CLUS_MD5" = "$R3_CLUS_MD5" ] \
    || { echo "REFUSING: pin is $CLUS_MD5, round 3 was measured on $R3_CLUS_MD5 (M1)." >&2; exit 2; }
export LD_LIBRARY_PATH=$LIBSNAP:${LD_LIBRARY_PATH:-}

mkdir -p "$OD"
OD=$(cd "$OD" && pwd -P)
CFG="$OD/heap_evt$EVT.json"

# --- rewrite every arm-dir reference to the scratch dir (carried over verbatim) --------------
# WITHOUT THIS the run would write its outputs back over the arm it was compiled for -- the M13
# scientific-record overwrite.  The guard refuses if any arm path survives, and prints the
# surviving work- paths so a read can be told from a write.
python3 - "$CFG_IN" "$CFG" "$PRDIR" "$OD" <<'PYEOF'
import json, sys
src, dst, armdir, scratch = sys.argv[1:5]
text = open(src).read()
n = text.count(armdir)
text = text.replace(armdir, scratch)
open(dst, 'w').write(text)

cfg = json.loads(text)
bad, inputs = [], []
WRITE_HINTS = ('out', 'dump', 'save', 'zip', 'sink')
for i, node in enumerate(cfg):
    data = node.get('data')
    if not isinstance(data, dict):
        continue
    for k, v in data.items():
        if isinstance(v, str) and armdir in v:
            bad.append(f"[{i}] {node.get('type','?')}:{node.get('name','')} {k} = {v}")
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
if any(s.strip().startswith('WRITE') for s in inputs):
    print("ERROR: a write path still targets a work- arm -- refusing to run", file=sys.stderr)
    sys.exit(2)
PYEOF

PYLIB=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3.11.so.1.0
JEMALLOC=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
[ -e "$JEMALLOC" ] || { echo "ERROR: no jemalloc at $JEMALLOC" >&2; exit 1; }
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export WIRECELL_PATH=${WIRECELL_PATH:-/nfs/data/1/xqian/toolkit-dev/toolkit/cfg:/nfs/data/1/xqian/toolkit-dev/wire-cell-data}
LOG="$OD/heap_evt$EVT.log"
PFX="$OD/je_evt$EVT"

echo "=== d119 round 4 heap profile: evt $EVT"
echo "=== pin: $LIBSNAP  clus md5 $CLUS_MD5"
echo "=== prefix: $PFX"

cd "$OD"
# The arm's own runner preloads libpython (RTLD_GLOBAL) so the SCN vertex net can import.  Keep
# it, or the job silently falls back to the geometric vertex and profiles a different pipeline.
# jemalloc goes AFTER libpython in LD_PRELOAD for the same reason the CPU script puts the
# allocator second: libpython must win the symbol lookups it needs, the allocator interposes
# malloc regardless of order.
#
# NOTE the ordering rule this script's parent learned the hard way: no profiling env may leak
# into a wrapper process.  There is no timecmd.py here at all -- wire-cell is exec'd directly and
# the MALLOC_CONF is set only in ITS environment, so nothing else can dump over the prefix.
rc=0
env LD_PRELOAD="$PYLIB:$JEMALLOC" \
    MALLOC_CONF="prof:true,prof_active:true,lg_prof_sample:${LG_SAMPLE:-19},prof_prefix:$PFX,prof_final:true,lg_prof_interval:${LG_INTERVAL:-28}" \
    wire-cell -l stderr -l "$LOG:debug" -L debug -c "$CFG" > "$OD/heap_evt$EVT.stdout" 2>&1 &
WCPID=$!
# Record what the PROCESS actually mapped, not what we asked it to map.  This is the difference
# between a freshness claim and a freshness proof.
MAPPED=""
for _ in $(seq 1 60); do
    MAPPED=$(sed -n 's#.* \(/.*libWireCellClus\.so\)$#\1#p' "/proc/$WCPID/maps" 2>/dev/null | head -1) || true
    [ -n "$MAPPED" ] && break
    sleep 1
done
if [ -n "$MAPPED" ]; then
    echo "=== process mapped: $MAPPED  md5 $(md5sum "$MAPPED" | cut -d' ' -f1)" | tee "$OD/heap_evt$EVT.libproof"
else
    echo "=== WARNING: could not read libWireCellClus.so from /proc/$WCPID/maps" | tee "$OD/heap_evt$EVT.libproof"
fi
wait $WCPID || rc=$?

echo "rc=$rc"
echo "=== dumps:"
echo "=== count: $(find "$(dirname "$PFX")" -maxdepth 1 -name "$(basename "$PFX").*.heap" | wc -l)  (.i* = per-interval, .f = final/exit)"
echo "view: scripts/d119/heap_rank.py $OD $EVT"
exit $rc
