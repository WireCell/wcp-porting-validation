#!/bin/bash
# doc pdvd/36 -- the 120-event PDVD PR manifest under the anisotropic ctpc
# metric knob, one binary (a pinned local/lib), fresh tags only (M13), inputs
# symlinked from d27fresh by scripts/stage_pr_tag.sh.
#
#   d36refpr  no TLA, run on the OLD pin (PIN=pin_d36ref): production config on
#                                the pre-change binary -- the knob-OFF reference.
#                                (d32p035 is NOT a valid reference any more: doc 35
#                                changed the production tagger FV at 07:49, after it ran.)
#   d36off    no TLA, NEW pin   -> production config (good_point_pitch_frac 0.35,
#                                metric OFF): the knob-OFF byte-identity arm,
#                                compared member-by-member against d36refpr.
#   d36p000   metric OFF + the knob-off track-fitting JSON (frac 0), NEW pin:
#                                the legacy reference under TODAY's config (d32p000
#                                predates doc 35's tagger-FV change).
#   d36on     metric ON + the knob-off track-fitting JSON (frac 0): the
#                                recommendation -- the metric SUBSUMES the floor.
#   d36on035  metric ON + production JSON (frac 0.35): the stacked arm, what a
#                                bare TLA flip would give without retiring the floor.
#
# Usage:
#   PIN=<dir with libpin/ and binpin/>  TFOFF=<knob-off pdvd_track_fitting.json> \
#   [JOBS=16] [ARMS="d36off d36on d36on035"] [OUT=<log dir>] \
#   ./docs/nf_sp_img_clus/scripts/run_d36_arms.sh
#
# TFOFF must be the 49-key file WITHOUT good_point_pitch_frac, i.e.
#   git show bec1bd75:cfg/pgrapher/experiment/protodunevd/pdvd_track_fitting.json
# (HEAD's copy carries the 0.35 floor, so `git show HEAD:` is the knob-ON file).
set -u
PIN=${PIN:?PIN=<dir with libpin/ and binpin/>}
TFOFF=${TFOFF:?TFOFF=<knob-off pdvd_track_fitting.json>}
JOBS=${JOBS:-16}
ARMS=${ARMS:-"d36off d36on d36on035"}
OUT=${OUT:-/home/xqian/tmp/d36_arms}
export LD_LIBRARY_PATH="$PIN/libpin:${LD_LIBRARY_PATH:-}"
export PATH="$PIN/binpin:$PATH"
cd "$(dirname "$0")/../../.." || exit 9      # pdvd/
MAN=scripts/perf_manifest.tsv
mkdir -p "$OUT"

stage_and_run() {
    local run=$1 idx=$2 tag=$3 extra=$4
    local rp; rp=$(printf '%06d' "$((10#$run))")
    [ -d "work/${rp}_${idx}_${tag}" ] || ./scripts/stage_pr_tag.sh "$run" "$idx" "$tag" d27fresh >/dev/null 2>&1
    if [ -n "$extra" ]; then
        PDVD_PR_TLA="$extra" ./run_pr_evt.sh -s "$tag" -stm-fit "$run" "$idx" \
            > "$OUT/${tag}_${rp}_${idx}.log" 2>&1
    else
        ./run_pr_evt.sh -s "$tag" -stm-fit "$run" "$idx" \
            > "$OUT/${tag}_${rp}_${idx}.log" 2>&1
    fi
    echo "$run $idx $tag $?"
}
export -f stage_and_run
export OUT

extra_for() {
    case "$1" in
        d36refpr) echo "" ;;
        d36off)   echo "" ;;
        d36p000)  echo "-A trackfitting_config=$TFOFF" ;;
        d36on)    echo "-S ctpc_aniso_metric=true -A trackfitting_config=$TFOFF" ;;
        d36on035) echo "-S ctpc_aniso_metric=true" ;;
        *) echo "unknown arm $1" >&2; exit 2 ;;
    esac
}

for tag in $ARMS; do
    extra=$(extra_for "$tag")
    echo "=== arm $tag  (extra='${extra}')  jobs=$JOBS  $(date +%H:%M:%S)"
    awk 'NR>1{print $1, $2}' "$MAN" \
      | xargs -P "$JOBS" -n 2 bash -c 'stage_and_run "$0" "$1" '"$tag"' "'"$extra"'"' \
      > "$OUT/rc_${tag}.txt" 2>&1
    bad=$(awk '$4!=0' "$OUT/rc_${tag}.txt" | wc -l)
    echo "    $tag: $(wc -l < "$OUT/rc_${tag}.txt") events, nonzero rc: $bad  $(date +%H:%M:%S)"
done
echo "ALL DONE"
