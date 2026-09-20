#!/bin/bash
# doc sbnd_xin/115 stage B: the 15-stage PR + tagger + BDT chain on a d115 stage-A arm.
#
# Nothing but the per-event arguments is passed: the SBND production operating point IS the
# jsonnet TLA defaults (doc 92 sec 0), and the doc-109 ROOT trees (T_tagger, T_kine, T_bundle,
# T_flash, with run/subrun/event on each) are on by default at HEAD.  PR_EXTRA_STAGES=pr_display
# is the calib dump only -- no physics effect (doc pr/3).
#
# One stage-B out_root per stage-A out_root, for the same event-ID reason (see stageA.sh).
#
# Usage: [JOBS=n] stageB.sh <cv|nuecc|off>
set -u
cd -P "$(dirname "$0")/../.." || exit 1
SX=$PWD
S=${1:?usage: [JOBS=n] stageB.sh <cv|nuecc|off>}
J=${JOBS:-16}
export LIBSNAP=${LIBSNAP:-$HOME/tmp/d115-libsnap}
export LD_LIBRARY_PATH=$LIBSNAP:${LD_LIBRARY_PATH:-}
export PR_EXTRA_STAGES=${PR_EXTRA_STAGES:-pr_display}

case "$S" in
    cv)    A=$SX/work-r3cv-d115;  B=$SX/work-r3cv-d115pr;  REALITY=sim ;;
    nuecc) A=$SX/work-r3nue-d115; B=$SX/work-r3nue-d115pr; REALITY=sim ;;
    off)   A=$SX/work-r3off-d115; B=$SX/work-r3off-d115pr; REALITY=data ;;
    *) echo "unknown sample: $S" >&2; exit 2 ;;
esac
[ -d "$A" ] || { echo "ERROR: no stage-A arm $A" >&2; exit 1; }
LOGD=$HOME/tmp/d115-stageB-$S; mkdir -p "$LOGD" "$B"

echo "=== d115 stage B: sample=$S reality=$REALITY PR_JOBS=$J"
echo "=== libsnap: $LIBSNAP (toolkit $(cat "$LIBSNAP/TOOLKIT_HEAD" 2>/dev/null))"
echo "=== toolkit HEAD before: $(git -C /nfs/data/1/xqian/toolkit-dev/toolkit rev-parse HEAD)"
t0=$(date +%s)

if [ "$S" = off ]; then
    SUBS=("")                       # one root
else
    mapfile -t SUBS < <(ls -d "$A"/f* 2>/dev/null | xargs -n1 basename | sort)
fi
echo "=== stage-A sub-roots: ${#SUBS[@]}"

for sub in "${SUBS[@]}"; do
    qa=$A${sub:+/$sub}; qb=$B${sub:+/$sub}
    nql=$(ls -d "$qa"/ql_evt* 2>/dev/null | wc -l)
    npr=$(ls -d "$qb"/pr_evt* 2>/dev/null | wc -l)
    if [ "$nql" -gt 0 ] && [ "$npr" -eq "$nql" ]; then
        echo "[${sub:-.}] already complete ($npr/$nql) -- skipped"; continue
    fi
    PR_JOBS=$J "$SX/run_pr_chain_batch.sh" "$qa" "$qb" "$REALITY" \
        > "$LOGD/${sub:-all}.log" 2>&1
    rc=$?
    echo "[${sub:-.}] rc=$rc pr_evt=$(ls -d "$qb"/pr_evt* 2>/dev/null | wc -l)/$nql"
done

# run_pr_chain_batch.sh merges nusel-table.tsv / nusel-events.tsv per OUT_ROOT, and this
# campaign has one out_root per reco1 file (see stageA.sh), so a MC sample ends with 154 or 225
# table pairs.  Merge them into one pair per sample with the native tool before anything reads
# them.  (The candidate/truth analysis does not use these -- it joins on T_tagger's own RSE
# branches -- but the per-bundle label table is what doc 92's event_label numbers come from.)
echo "=== merging the per-file nusel tables"
mapfile -t _NS < <(ls "$B"/f*/pr_evt*/nusel-evt*.tsv "$B"/pr_evt*/nusel-evt*.tsv 2>/dev/null)
if [ "${#_NS[@]}" -gt 0 ]; then
    python3 "$SX/nusel_extract.py" --merge "${_NS[@]}" \
        --out "$B/nusel-table.tsv" --events-out "$B/nusel-events.tsv" \
        > "$LOGD/merge.log" 2>&1
    echo "  merge rc=$? rows=$(( $(wc -l < "$B/nusel-table.tsv" 2>/dev/null || echo 1) - 1 ))" \
         "events=$(( $(wc -l < "$B/nusel-events.tsv" 2>/dev/null || echo 1) - 1 ))"
else
    echo "  no per-event nusel tables found"
fi

t1=$(date +%s)
echo "=== toolkit HEAD after : $(git -C /nfs/data/1/xqian/toolkit-dev/toolkit rev-parse HEAD)"
echo "=== stage B $S finished in $((t1-t0)) s"
echo "=== pr_evt dirs : $(ls -d "$B"/*/pr_evt* "$B"/pr_evt* 2>/dev/null | wc -l)"
echo "=== rc!=0 events: $(grep -L 'rc=0' "$B"/*/pr_evt*/rc.txt "$B"/pr_evt*/rc.txt 2>/dev/null | wc -l)"
exit 0
