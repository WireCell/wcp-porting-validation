#!/bin/bash
# doc sbnd_xin/123: stage B (the 15-stage PR chain, run_pr_chain_batch.sh) on a d123 stage-A arm,
# data or MC, one out_root per stage-A out_root (per-file sub-roots for cv/nuecc), at the d123 pin.
#
# Usage: [JOBS=n] scripts/d123/stageB.sh <stageA_root> <data|sim>      -> <stageA_root>pr
set -u
cd -P "$(dirname "$0")/../.." || exit 1
SX=$PWD
A=${1:?usage: stageB.sh <stageA_root> <data|sim>}; REALITY=${2:?}
case "$A" in /*) ;; *) A=$SX/$A;; esac
B=${A}pr
J=${JOBS:-12}
P=${PIN:-$HOME/tmp/d123-libpin}
export LD_LIBRARY_PATH=$P:$P/reco1/lib:${LD_LIBRARY_PATH:-}
export PR_EXTRA_STAGES=${PR_EXTRA_STAGES:-pr_display}
[ -d "$A" ] || { echo "ERROR: no stage-A arm $A" >&2; exit 1; }
LOGD=$HOME/tmp/d123-stageB-$(basename "$A"); mkdir -p "$LOGD" "$B"
echo "=== d123 stage B: $A -> $B reality=$REALITY PR_JOBS=$J pin=$P"
t0=$(date +%s)
mapfile -t SUBS < <(ls -d "$A"/f[0-9]* 2>/dev/null | xargs -n1 basename 2>/dev/null | sort)
[ "${#SUBS[@]}" -gt 0 ] || SUBS=("")
for sub in "${SUBS[@]}"; do
    qa=$A${sub:+/$sub}; qb=$B${sub:+/$sub}
    nql=$(ls -d "$qa"/ql_evt* 2>/dev/null | wc -l)
    npr=$(ls -d "$qb"/pr_evt* 2>/dev/null | wc -l)
    if [ "$nql" -gt 0 ] && [ "$npr" -eq "$nql" ]; then
        echo "[${sub:-.}] already complete ($npr/$nql) -- skipped"; continue
    fi
    PR_JOBS=$J "$SX/run_pr_chain_batch.sh" "$qa" "$qb" "$REALITY" > "$LOGD/${sub:-all}.log" 2>&1
    echo "[${sub:-.}] rc=$? pr_evt=$(ls -d "$qb"/pr_evt* 2>/dev/null | wc -l)/$nql"
done
if [ -n "${SUBS[0]}" ]; then
    mapfile -t _NS < <(ls "$B"/f*/pr_evt*/nusel-evt*.tsv 2>/dev/null)
    [ "${#_NS[@]}" -gt 0 ] && python3 "$SX/nusel_extract.py" --merge "${_NS[@]}" \
        --out "$B/nusel-table.tsv" --events-out "$B/nusel-events.tsv" > "$LOGD/merge.log" 2>&1
fi
echo "=== stage B finished in $(( $(date +%s) - t0 )) s; pr_evt dirs: $(ls -d "$B"/f*/pr_evt* "$B"/pr_evt* 2>/dev/null | wc -l)"
