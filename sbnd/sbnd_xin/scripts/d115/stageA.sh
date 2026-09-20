#!/bin/bash
# doc sbnd_xin/115 stage A: reco1 -> imaging -> clustering + Q/L, for one round-3 sample.
#
# ONE out_root PER RECO1 FILE.  run_chain_group.sh --layout perevt keys its products by the art
# event NUMBER alone (ql_evt<ID>/), and SBND MC event numbers repeat across files -- mc-cv has 50
# distinct numbers over its 2017 events (scripts/d115/rse_census.sh).  A single out_root would
# overwrite most of the sample in silence.  Per-file roots are sufficient because the census
# proved every file internally unique (154/154 and 225/225 files, one subrun each), and they cost
# nothing downstream: the analysis joins on (run, subrun, event) read from T_tagger's own
# branches, not on a directory name.  --gbase is not needed either -- each root has its own g0.
#
# --size 1000 makes each file exactly ONE group (6-18 events), so one dump + one imaging + one
# Q/L process per file.
#
# Usage: [JOBS=n] stageA.sh <cv|nuecc|off>
# Env:   JOBS   concurrent FILES (default 8; each wire-cell process is itself multi-threaded,
#               ~1.5 GiB per concurrent job -- CLAUDE.md M5, check /proc/loadavg after launch)
#        LIBSNAP  pinned library dir (default ~/tmp/d115-libsnap)
set -u
cd -P "$(dirname "$0")/../.." || exit 1
SX=$PWD
S=${1:?usage: [JOBS=n] stageA.sh <cv|nuecc|off>}
J=${JOBS:-8}
export LIBSNAP=${LIBSNAP:-$HOME/tmp/d115-libsnap}
export LD_LIBRARY_PATH=$LIBSNAP:${LD_LIBRARY_PATH:-}
# doc 87: the imaging -> Q/L handoff npz are read by nothing once Q/L has succeeded (the PR job's
# compiled config contains no icluster reference at all), and they are 4.8 GB per 1000 events.
export SBND_QL_KEEP_ICLUSTER=0

# The two MC samples are many small reco1 files, so parallelism is over FILES and each file is
# one group (--size 1000 >= any file's entry count).  beam-off is ONE 1000-entry file, so there
# is no file-level parallelism to have: it takes the ordinary --size 16 grouping and hands the
# concurrency to run_chain_group.sh's own SBND_MAX_JOBS, exactly as the d102m data arms did.
case "$S" in
    cv)    OUT=$SX/work-r3cv-d115;  REALITY=sim;  MCFLAG=(--mc); GSIZE=1000; PERFILE=1 ;;
    nuecc) OUT=$SX/work-r3nue-d115; REALITY=sim;  MCFLAG=(--mc); GSIZE=1000; PERFILE=1 ;;
    off)   OUT=$SX/work-r3off-d115; REALITY=data; MCFLAG=();     GSIZE=16;   PERFILE=0 ;;
    *) echo "unknown sample: $S (cv|nuecc|off)" >&2; exit 2 ;;
esac
LST=$SX/products/d115/$S/files.lst
[ -s "$LST" ] || { echo "ERROR: no $LST -- run scripts/d115/rse_census.sh $S first" >&2; exit 1; }
NF=$(wc -l < "$LST")
LOGD=$HOME/tmp/d115-stageA-$S; mkdir -p "$LOGD" "$OUT"

echo "=== d115 stage A: sample=$S files=$NF jobs=$J reality=$REALITY mc=${#MCFLAG[@]}"
echo "=== libsnap: $LIBSNAP (toolkit $(cat "$LIBSNAP/TOOLKIT_HEAD" 2>/dev/null))"
echo "=== toolkit HEAD before: $(git -C /nfs/data/1/xqian/toolkit-dev/toolkit rev-parse HEAD)"
echo "=== out: $OUT"
t0=$(date +%s)

one() {  # one <fileidx> <path>
    local i=$1 f=$2 g="$OUTT/f$1"
    # Resume guard: a group that already produced its pctrees is not re-run.  run_chain_group.sh
    # skips completed stages on its own, but this avoids even starting a process for a done file.
    if [ -s "$g/g0/events.txt" ] \
       && [ "$(wc -l < "$g/g0/events.txt")" -eq "$(ls -d "$g"/ql_evt* 2>/dev/null | wc -l)" ] \
       && [ "$(ls -d "$g"/ql_evt* 2>/dev/null | wc -l)" -gt 0 ] \
       && ! find "$g"/ql_evt*/pctree-evt*.tar.gz -size 0 2>/dev/null | grep -q .; then
        echo "[f$i] already complete -- skipped"; return 0
    fi
    SBND_MAX_JOBS=1 "$SXX/run_chain_group.sh" "$f" "$g" "$REAL" \
        "${MCF[@]}" --size "$GSZ" --layout perevt > "$LOGDD/f$i.log" 2>&1
    local rc=$?
    echo "[f$i] rc=$rc evt=$(ls -d "$g"/ql_evt* 2>/dev/null | wc -l)"
    return 0        # never abort the batch on one file; the completeness gate is the verdict
}
export -f one
export SXX=$SX OUTT=$OUT LOGDD=$LOGD REAL=$REALITY GSZ=$GSIZE
# bash arrays do not survive `export`; pass --mc (or nothing) through a plain string instead.
export MCF_STR="${MCFLAG[*]-}"

if [ "$PERFILE" = 1 ]; then
    # shellcheck disable=SC2016
    awk -F'\t' '{print $1"\t"$2}' "$LST" \
      | xargs -P "$J" -I{} bash -c 'IFS=$(printf "\t"); set -- {}; read -r -a MCF <<< "$MCF_STR"; one "$1" "$2"'
else
    # One out_root, many groups: hand the concurrency to run_chain_group.sh itself.
    f=$(awk -F'\t' 'NR==1{print $2; exit}' "$LST")
    SBND_MAX_JOBS=$J "$SX/run_chain_group.sh" "$f" "$OUT" "$REALITY" \
        "${MCFLAG[@]+"${MCFLAG[@]}"}" --size "$GSIZE" --layout perevt > "$LOGD/all.log" 2>&1
    echo "[all] rc=$? evt=$(ls -d "$OUT"/ql_evt* 2>/dev/null | wc -l)"
fi

t1=$(date +%s)
echo "=== toolkit HEAD after : $(git -C /nfs/data/1/xqian/toolkit-dev/toolkit rev-parse HEAD)"
echo "=== stage A $S finished in $((t1-t0)) s"
echo "=== ql_evt dirs: $(ls -d "$OUT"/f*/ql_evt* "$OUT"/ql_evt* 2>/dev/null | wc -l) (expect $(( $(wc -l < "$SX/products/d115/$S/file_rse.tsv") - 1 )))"
exit 0
