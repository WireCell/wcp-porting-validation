#!/bin/bash
# doc pr/149 -- one PR (stage-B) arm of the charge_stepped retile study.
#
# Fork by duplication (CLAUDE.md M10) of pdvd/docs/nf_sp_img_clus/scripts/d101_sbnd_arm.sh
# (that script stays untouched).  Differences: the arm is parameterised (samples, manifest,
# TLA file, fit JSON, geometric-vertex switch, cfg-tree overlay) instead of hard-coding one
# gate list, and the output is work-<sample>-pr149<TAG>.  Stage A is always the production
# work-<sample>-d102m.  Refuses an existing output tree (M13).
#
# Env:
#   TAG       arm label (required), e.g. s0 / s0rep / cs / kf / cskf / csq2000 / goff
#   PIN       whole-local/lib snapshot to prepend to LD_LIBRARY_PATH (required)
#   SAMPLES   default "nuecc48 ncpi0"
#   MANIFEST  optional dir holding <sample>.txt (one event id per line); absent => every event
#   TLA_FILE  optional PR_EXTRA_TLA file (key=value jsonnet code per line)
#   TFJSON    optional ABSOLUTE TrackFitting JSON (SBND_TRACKFIT_JSON)
#   NO_DL     1 => geometric vertex (SBND_NO_DL=1); default 0 = production DL vertex
#   CFG_TREE  optional PR_CFG_TREE (an older cfg tree, for the knob-OFF gate)
#   JOBS      PR_JOBS per sample, default 8
#
# Usage: TAG=s0 PIN=/home/xqian/tmp/pr149/libpin bash scripts/pr149_arm.sh \
#            > /home/xqian/tmp/pr149/arm_s0.log 2>&1
set -u
TAG=${TAG:?set TAG}
PIN=${PIN:?set PIN}
SAMPLES=${SAMPLES:-"nuecc48 ncpi0"}
JOBS=${JOBS:-8}
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
cd "$SX" || exit 2

export LD_LIBRARY_PATH="$PIN:${LD_LIBRARY_PATH:-}"
export PR_EXTRA_STAGES=pr_display
unset PR_GROUP_SIZE PR_EXTRA_TLA SBND_TRACKFIT_JSON SBND_NO_DL PR_CFG_TREE
[ -n "${TLA_FILE:-}" ] && export PR_EXTRA_TLA="$TLA_FILE"
[ -n "${TFJSON:-}" ] && export SBND_TRACKFIT_JSON="$TFJSON"
[ "${NO_DL:-0}" = 1 ] && export SBND_NO_DL=1
[ -n "${CFG_TREE:-}" ] && export PR_CFG_TREE="$CFG_TREE"

echo "=== pr149 arm $TAG  samples=[$SAMPLES] jobs=$JOBS pin=$PIN  $(date +%F_%H:%M:%S)"
echo "    TLA_FILE=${TLA_FILE:-} TFJSON=${TFJSON:-} NO_DL=${NO_DL:-0} CFG_TREE=${CFG_TREE:-} MANIFEST=${MANIFEST:-}"
[ -n "${TLA_FILE:-}" ] && sed 's/^/    tla: /' "$TLA_FILE"
md5sum "$PIN/libWireCellClus.so"
for s in $SAMPLES; do
  OUT="work-$s-pr149$TAG"
  [ -e "$OUT" ] && { echo "REFUSING: $OUT exists (M13)"; exit 2; }
done
rc_all=0
for s in $SAMPLES; do
  OUT="work-$s-pr149$TAG"
  EVTS=()
  if [ -n "${MANIFEST:-}" ]; then
    [ -r "$MANIFEST/$s.txt" ] || { echo "--- $s: no manifest $MANIFEST/$s.txt, skipped"; continue; }
    mapfile -t EVTS < <(grep -v '^#' "$MANIFEST/$s.txt" | awk 'NF{print $1}')
  fi
  echo "--- $s -> $OUT  (${#EVTS[@]} events listed; 0 = all)  $(date +%H:%M:%S)"
  PR_JOBS=$JOBS ./run_pr_chain_batch.sh "work-$s-d102m" "$OUT" data "${EVTS[@]}"
  rc=$?
  [ $rc = 0 ] || rc_all=1
  n=$(ls -d $OUT/pr_evt*/ 2>/dev/null | wc -l)
  nbad=0
  for r in $OUT/pr_evt*/rc.txt; do [ "$(cat $r 2>/dev/null)" = rc=0 ] || nbad=$((nbad+1)); done   # rc.txt holds "rc=N"
  echo "--- $s rc=$rc events=$n nonzero_rc=$nbad  $(date +%H:%M:%S) loadavg=$(cut -d' ' -f1 /proc/loadavg)"
done
md5sum "$PIN/libWireCellClus.so"
echo "=== pr149 arm $TAG DONE rc_all=$rc_all $(date +%F_%H:%M:%S)"
exit $rc_all
