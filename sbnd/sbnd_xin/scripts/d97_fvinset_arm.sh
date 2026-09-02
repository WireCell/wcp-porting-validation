#!/bin/bash
# doc 97 -- the SECOND separation case, 105-23-21, taken to a VERDICT.
#
# doc 96 sec 8.3 measured that 105-23-21 separates only under the 15 cm y/z FV
# inset AND the far_point knobs together, and that the second piece then lands
# inside a different 28389-point host -- "not a clean two-track split".  What
# that probe could not say is whether the EVENT gets better: its TGM verdict
# exists only because the two tracks are one cluster (doc 96 sec 3.1).  This
# arm answers that by running the same operating point through the PR chain.
#
# It is a FEASIBILITY arm, not a proposal.  The inset is applied to the shared
# DetectorVolumes metadata, so it also moves clustering_neutrino and the
# containment taggers (doc 96 sec 9 step 1): it cannot ship as a config change.
# A shippable version needs a separation-scoped inset knob in C++.
#
# Usage: [D97_JOBS=6] ./scripts/d97_fvinset_arm.sh [evt ...]   (default: all of group a)
set -u
cd -P "$(dirname "$0")/.." || exit 1
BASE=$PWD
export LD_LIBRARY_PATH=$HOME/tmp/doc94r3b-libsnap:${LD_LIBRARY_PATH:-}
export PR_CFG_TREE=$HOME/tmp/dbg25-cfgsnap
export WIRECELL_PATH=/nfs/data/1/xqian/toolkit-dev/toolkit/cfg:/nfs/data/1/xqian/toolkit-dev/wire-cell-data:/nfs/data/1/xqian/toolkit-dev/wire-cell-data/sbnd/photodet
export PR_JOBS=${D97_JOBS:-6}
export PR_EXTRA_STAGES=pr_display
SRC=$BASE/work-dbg25a-ql
OUTP=/home/xqian/tmp/d97/fvinset
QL=$OUTP/pdhdlike
PR=$BASE/work-dbg25a-d97fvpr
LOGD=/home/xqian/tmp/d97; mkdir -p "$LOGD" "$OUTP"
[ -e "$PR" ] && { echo "REFUSE: $PR exists (M13)" >&2; exit 1; }

evts=("$@")
if [ ${#evts[@]} -eq 0 ]; then
    while read -r d; do evts+=("${d#ql_evt}"); done < <(cd "$SRC" && ls -d ql_evt* 2>/dev/null)
fi
echo "=== fvinset arm: ${#evts[@]} events  $(date -Is)"
python3 scripts/d96_fvinset_probe.py --src "$SRC" --out "$OUTP" "${evts[@]}" \
    > "$LOGD/fvinset-patch.log" 2>&1
echo "  patch rc=$?  configs=$(ls "$OUTP"/pdhdlike/evt*.json 2>/dev/null | wc -l)"

run_one() {
    local e=$1
    setarch x86_64 -R wire-cell -l "$QL/ql_evt$e/wct_ql_evt$e.log:debug" -L debug \
        -c "$QL/evt$e.json" > "$QL/ql_evt$e/stdout.log" 2>&1
    echo "rc=$? evt=$e fired=$(grep -ac 'Separate track_recarve' "$QL/ql_evt$e/stdout.log" 2>/dev/null)" \
        > "$QL/.status-$e"
}
export -f run_one
export QL
printf '%s\n' "${evts[@]}" | xargs -P "${D97_JOBS:-6}" -I{} bash -c 'run_one {}'
echo "  ql done $(date -Is)  ok=$(grep -hc '^rc=0 ' "$QL"/.status-* 2>/dev/null | paste -sd+ | bc)"
grep -h . "$QL"/.status-* | grep -v '^rc=0 ' | sed 's/^/  FAIL /'

./run_pr_chain_batch.sh "$QL" "$PR" sim > "$LOGD/fvinset-pr.log" 2>&1
echo "  pr rc=$?  pr_evt=$(find "$PR" -maxdepth 1 -type d -name 'pr_evt*' | wc -l)  $(date -Is)"
