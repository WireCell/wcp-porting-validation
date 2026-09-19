#!/bin/bash
# doc sbnd_xin/pr/150 -- one PR (stage-B) arm of the PDHD/PDVD-trajectory-on-SBND study.
#
# Fork by duplication (CLAUDE.md M10) of scripts/pr149_arm.sh (that script stays untouched).
# Differences: output work-<sample>-pr150<TAG>; STM_FIT=1 appends the stm_magnify stage and sets
# save_stm_fit=true (tracking-stm.root per event: T_rec_charge / T_stm_pass / T_stm_eval / T_proj_data;
# verdict-neutral, gated in doc pr/150 sec 1); TRACE_ENV passes the Steiner / STM dump env vars to the
# jobs; a scoped placeholder grep on the TLA file before launch; the pin md5 at start and end.
# Stage A is always the production work-<sample>-d102m.  Refuses an existing output tree (M13).
#
# Env:
#   TAG        arm label (required), e.g. s0 / cs / p3bw / csp3bw / tfull / g16old / r1
#   PIN        whole-local/lib snapshot to prepend to LD_LIBRARY_PATH (required)
#   SAMPLES    default "nuecc48 ncpi0"
#   MANIFEST   optional dir holding <sample>.txt (one event id per line); absent => every event
#   TLA_FILE   optional PR_EXTRA_TLA file (key=value jsonnet code per line; '__' placeholders refused)
#   TFJSON     optional ABSOLUTE TrackFitting JSON (SBND_TRACKFIT_JSON)
#   NO_DL      1 => geometric vertex (SBND_NO_DL=1); default 0 = production DL vertex
#   CFG_TREE   optional PR_CFG_TREE (an older cfg tree, for the knob-OFF gate)
#   STM_FIT    1 => PR_EXTRA_STAGES=pr_display,stm_magnify + save_stm_fit=true (default 0)
#   TRACE_ENV  optional "K=V K=V" exported to the jobs (WCT_STEINER_GRAPH_DUMP=1 WCT_STM_PATH_DEBUG=1 ...)
#   JOBS       PR_JOBS per sample, default 8
#
# Usage: TAG=s0 PIN=/home/xqian/tmp/pr150/libpin STM_FIT=1 bash scripts/pr150_arm.sh > /home/xqian/tmp/pr150/logs/arm_s0.log 2>&1
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
TLA_TMP=""
if [ "${STM_FIT:-0}" = 1 ]; then
  export PR_EXTRA_STAGES=pr_display,stm_magnify
  TLA_TMP=$(mktemp /home/xqian/tmp/pr150/tla_${TAG}.XXXX)
  { [ -n "${TLA_FILE:-}" ] && cat "$TLA_FILE"; echo "save_stm_fit=true"; } > "$TLA_TMP"
  export PR_EXTRA_TLA="$TLA_TMP"
elif [ -n "${TLA_FILE:-}" ]; then
  export PR_EXTRA_TLA="$TLA_FILE"
fi
if [ -n "${PR_EXTRA_TLA:-}" ] && grep -q '__' "$PR_EXTRA_TLA"; then echo "REFUSING: placeholder '__' in $PR_EXTRA_TLA"; exit 2; fi
[ -n "${TFJSON:-}" ] && export SBND_TRACKFIT_JSON="$TFJSON"
[ "${NO_DL:-0}" = 1 ] && export SBND_NO_DL=1
[ -n "${CFG_TREE:-}" ] && export PR_CFG_TREE="$CFG_TREE"
for kv in ${TRACE_ENV:-}; do export "$kv"; done

echo "=== pr150 arm $TAG  samples=[$SAMPLES] jobs=$JOBS pin=$PIN stm_fit=${STM_FIT:-0}  $(date +%F_%H:%M:%S)"
echo "    TLA_FILE=${TLA_FILE:-} TFJSON=${TFJSON:-} NO_DL=${NO_DL:-0} CFG_TREE=${CFG_TREE:-} MANIFEST=${MANIFEST:-} TRACE_ENV=${TRACE_ENV:-} PR_EXTRA_STAGES=$PR_EXTRA_STAGES"
[ -n "${PR_EXTRA_TLA:-}" ] && sed 's/^/    tla: /' "$PR_EXTRA_TLA"
md5sum "$PIN/libWireCellClus.so"
for s in $SAMPLES; do
  OUT="work-$s-pr150$TAG"
  [ -e "$OUT" ] && { echo "REFUSING: $OUT exists (M13)"; exit 2; }
done
rc_all=0
for s in $SAMPLES; do
  OUT="work-$s-pr150$TAG"
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
  nstm=$(ls $OUT/pr_evt*/tracking-stm.root 2>/dev/null | wc -l)
  ndl=$(grep -l 'DL vertex failed' $OUT/pr_evt*/wct_pr_evt*.log 2>/dev/null | wc -l)
  echo "--- $s rc=$rc events=$n nonzero_rc=$nbad stm_root=$nstm dl_failed=$ndl  $(date +%H:%M:%S) loadavg=$(cut -d' ' -f1 /proc/loadavg)"
done
md5sum "$PIN/libWireCellClus.so"
echo "=== pr150 arm $TAG DONE rc_all=$rc_all $(date +%F_%H:%M:%S)"
exit $rc_all
