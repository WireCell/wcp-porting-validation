#!/bin/bash
# doc sbnd_xin/118 gate G1: re-run stage B at the FLIPPED PRODUCTION DEFAULT and prove it reproduces
# the doc-116 `tfull` arm.  Fork by duplication (CLAUDE.md M10) of scripts/d117/stageB_cell.sh,
# which stays untouched.
#
# The whole point is the invocation.  Doc 116's `tfull` arm reached the PDHD/PDVD trajectory with
#   PR_EXTRA_TLA=docs/116_figs/tla/tfull.tla   (4 --tla-code lines)
#   SBND_TRACKFIT_JSON=docs/pr/149_figs/149_tf_sbnd_kf.json
# on top of run_pr_chain_batch.sh's own base TLA set (the 15-stage pipeline_names, save_tensors,
# dl_weights, reality, run/subrun/event, the doc-109 provenance TLAs, PR_EXTRA_STAGES=pr_display).
# This script is that invocation MINUS EXACTLY THOSE TWO THINGS and nothing else, so the only
# difference between this arm and the `tfull` arm is where the configuration came from: the flipped
# jsonnet call sites and the in-tree fit JSON, instead of a TLA file and a runtime JSON path.
# Running it "with no TLAs at all" would be a DIFFERENT job (no pipeline, no display stage) and
# would make the comparison meaningless -- hence the explicit unset of only the two, at :NN below.
#
# Unlike the d117 fork, a missing .tla is the REQUIRED state here, not a refusal.
#
# Same pin (LIBSNAP, default ~/tmp/d115-libsnap = toolkit 0a2807f4 -- cfg is read from the toolkit
# tree through WIRECELL_PATH, so the flip needs no rebuild and the binary is bit-for-bit the one
# doc 116 ran), same PR_EXTRA_STAGES=pr_display, same per-sub-root layout, same output set.
#
# Usage: [JOBS=n] [OFF_N=k] stageB_flip.sh <cv|nuecc|off> [f000 f001 ...]
#   f*      optional stage-A sub-root subset (MC only); default = every sub-root
#   ARM_SUFFIX  e.g. "rep" -> the determinism control arm work-r3<s>-d118fliprep
#   OFF_N   beam-off has no sub-roots (one flat root of 1000 gates): take the first k event ids
#           (default 20).  Gate, not census -- the census is doc 116's.
set -u
cd -P "$(dirname "$0")/../.." || exit 1
SX=$PWD
S=${1:?usage: [JOBS=n] [OFF_N=k] stageB_flip.sh <cv|nuecc|off> [f000 ...]}
shift
J=${JOBS:-16}
export LIBSNAP=${LIBSNAP:-$HOME/tmp/d115-libsnap}
export LD_LIBRARY_PATH=$LIBSNAP:${LD_LIBRARY_PATH:-}
export PR_EXTRA_STAGES=${PR_EXTRA_STAGES:-pr_display}
# THE FLIP: these two are what the production default now supplies from jsonnet + the in-tree JSON.
unset PR_EXTRA_TLA SBND_TRACKFIT_JSON PR_GROUP_SIZE SBND_NO_DL PR_CFG_TREE

# ARM_SUFFIX=rep writes work-r3<s>-d118fliprep instead: the SAME configuration a second time.
# That control is not optional here.  The G1 comparison is at the byte level, and no byte-level
# instrument in this arc has ever been run on two runs of ONE configuration -- doc 116 sec 4's
# "noise floor is zero" is a statement about METRICS, not about archive members or ROOT
# containers.  Without it, run-to-run non-reproducibility would be read as the flip's doing.
SUF=${ARM_SUFFIX:-}
case "$S" in
    cv)    A=$SX/work-r3cv-d115;  B=$SX/work-r3cv-d118flip$SUF;  T=$SX/work-r3cv-d116tfull;  REALITY=sim ;;
    nuecc) A=$SX/work-r3nue-d115; B=$SX/work-r3nue-d118flip$SUF; T=$SX/work-r3nue-d116tfull; REALITY=sim ;;
    off)   A=$SX/work-r3off-d115; B=$SX/work-r3off-d118flip$SUF; T=$SX/work-r3off-d116tfull; REALITY=data ;;
    *) echo "unknown sample: $S" >&2; exit 2 ;;
esac
[ -d "$A" ] || { echo "ERROR: no stage-A arm $A" >&2; exit 1; }
[ -d "$T" ] || { echo "ERROR: no doc-116 tfull arm $T to compare against" >&2; exit 1; }

if [ -e "$B" ]; then
    [ "$(cat "$B/.d118_arm" 2>/dev/null)" = "flip" ] \
        || { echo "REFUSING: $B exists and is not a d118 flip arm (M13)" >&2; exit 2; }
    echo "=== resuming existing $B"
else
    mkdir -p "$B" && echo "flip" > "$B/.d118_arm"
fi
LOGD=$HOME/tmp/d118-stageB-$S$SUF; mkdir -p "$LOGD"

echo "=== d118 stage B (flipped production default): sample=$S reality=$REALITY PR_JOBS=$J  $(date +%F_%H:%M:%S)"
echo "=== libsnap: $LIBSNAP (toolkit $(cat "$LIBSNAP/TOOLKIT_HEAD" 2>/dev/null))"
echo "=== toolkit HEAD before: $(git -C /nfs/data/1/xqian/toolkit-dev/toolkit rev-parse HEAD) (cfg WORKING TREE, flipped)"
echo "=== PR_EXTRA_TLA=${PR_EXTRA_TLA:-<none, by design>}  SBND_TRACKFIT_JSON=${SBND_TRACKFIT_JSON:-<none, by design>}  PR_EXTRA_STAGES=$PR_EXTRA_STAGES"
echo "=== pin start: $(md5sum "$LIBSNAP/libWireCellClus.so")"
t0=$(date +%s)

if [ "$S" = off ]; then
    SUBS=("")
elif [ $# -gt 0 ]; then
    SUBS=("$@")
else
    mapfile -t SUBS < <(ls -d "$A"/f* 2>/dev/null | xargs -n1 basename | sort)
fi
echo "=== stage-A sub-roots: ${#SUBS[@]}"

for sub in "${SUBS[@]}"; do
    qa=$A${sub:+/$sub}; qb=$B${sub:+/$sub}
    [ -d "$qa" ] || { echo "[${sub:-.}] ERROR: no stage-A sub-root $qa" >&2; continue; }
    mkdir -p "$qb"
    EVTS=()
    if [ "$S" = off ]; then
        # gate subset: the first OFF_N event ids that the tfull arm also ran
        mapfile -t EVTS < <(ls -d "$T"/pr_evt* 2>/dev/null | sed -E 's#.*/pr_evt([0-9]+)/?$#\1#' \
                            | sort -n | head -n "${OFF_N:-20}")
        echo "[.] beam-off gate subset: ${#EVTS[@]} events"
    fi
    PR_JOBS=$J "$SX/run_pr_chain_batch.sh" "$qa" "$qb" "$REALITY" "${EVTS[@]}" \
        > "$LOGD/${sub:-all}.log" 2>&1
    rc=$?
    nql=$(ls -d "$qa"/ql_evt* 2>/dev/null | wc -l)
    echo "[${sub:-.}] rc=$rc pr_evt=$(ls -d "$qb"/pr_evt* 2>/dev/null | wc -l)/$(( ${#EVTS[@]} > 0 ? ${#EVTS[@]} : nql )) opcfg=$(sha256sum "$qb/.d109-opcfg.json" 2>/dev/null | cut -c1-16)"
done

t1=$(date +%s)
echo "=== pin end  : $(md5sum "$LIBSNAP/libWireCellClus.so")"
echo "=== distinct op_config_sha256 in the arm: $(sha256sum "$B"/f*/.d109-opcfg.json "$B"/.d109-opcfg.json 2>/dev/null | cut -c1-16 | sort -u | tr '\n' ' ')"
echo "=== tfull arm op_config_sha256          : $(sha256sum "$T"/f*/.d109-opcfg.json "$T"/.d109-opcfg.json 2>/dev/null | cut -c1-16 | sort -u | tr '\n' ' ')"
echo "=== stage B flip/$S finished in $((t1-t0)) s"
echo "=== pr_evt dirs : $(ls -d "$B"/*/pr_evt* "$B"/pr_evt* 2>/dev/null | wc -l)"
echo "=== rc!=0 events: $(grep -L 'rc=0' "$B"/*/pr_evt*/rc.txt "$B"/pr_evt*/rc.txt 2>/dev/null | wc -l)"
exit 0
