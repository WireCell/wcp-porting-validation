#!/bin/bash
# doc pdvd/38 -- the 120-event PDVD PR manifest under the GAP-AWARE end trim
# (TrackFitting parameter `end_trim_gap_len`, C++ default 0 = off), one binary
# per pin, fresh tags only (M13), inputs symlinked from d27fresh by
# scripts/stage_pr_tag.sh.
#
# `end_trim_gap_len` is a TrackFitting parameter, not a jsonnet TLA, so each arm
# gets its own trackfitting JSON generated from the production file below.
#
#   d38ref    OLD pin (no gap-trim code), production config      -- the knob-OFF
#                                reference for the byte-identity gate.
#   d38off    NEW pin, production config, no end_trim_gap_len key -- must be
#                                member-identical to d38ref.
#   d38g<N>   NEW pin, end_trim_gap_len = <N> cm.  Offline simulation on the
#                                d36on arm (doc 38 sec 2) predicts N = 3 cm is
#                                the operating point: on 039252/2 cluster 109 it
#                                takes the fit from 39 % to 0 % of points >2 cm
#                                from charge at a 3-point coverage cost, and on
#                                039349/48 cluster 53 from 38 % to 0 %.
#
# Usage:
#   PIN=<dir with libpin/ and binpin/> [TFPROD=<production pdvd_track_fitting.json>]
#   [ANISO=1] [JOBS=16] [ARMS="d38off d38g2 d38g3 d38g5"] [OUT=<log dir>] \
#   ./docs/nf_sp_img_clus/scripts/run_d38_arms.sh
#
# ANISO=1 adds -S ctpc_aniso_metric=true to every arm.  Needed only BEFORE the
# doc-36 sec 11 flip lands in pdvd/wct-pr-perevt.jsonnet; after it, leave unset.
set -u
PIN=${PIN:?PIN=<dir with libpin/ and binpin/>}
JOBS=${JOBS:-16}
ARMS=${ARMS:-"d38off d38g2 d38g3 d38g5"}
OUT=${OUT:-/home/xqian/tmp/d38_arms}
ANISO=${ANISO:-}
export LD_LIBRARY_PATH="$PIN/libpin:${LD_LIBRARY_PATH:-}"
export PATH="$PIN/binpin:$PATH"
cd "$(dirname "$0")/../../.." || exit 9      # pdvd/
TFPROD=${TFPROD:-../../toolkit/cfg/pgrapher/experiment/protodunevd/pdvd_track_fitting.json}
MAN=scripts/perf_manifest.tsv
mkdir -p "$OUT/cfg"

# One trackfitting JSON per gap length, generated from the production file so a
# later change to any other fitting constant is picked up automatically.
mk_tf() {   # mk_tf <cm> -> path
    # NOTE: cm and dst must be SEPARATE `local` statements.  Bash expands every
    # word of a `local a=... b=...$a...` line before the assignments take
    # effect, so the one-line form expanded ${cm} while it was still unset and,
    # under `set -u`, aborted the function -- leaving extra_for to emit a bare
    # `-A trackfitting_config=`.  That is not inert: an empty value makes
    # TaggerCheckSTM skip load_trackfitting_config entirely and run on the C++
    # default TrackFitting parameters instead of PDVD's, which changed the
    # output of 111 of 111 events in the arm it silently ruined (tag d38g2,
    # abandoned).  The guard below turns any such failure into a hard stop.
    local cm=$1
    local dst="$OUT/cfg/pdvd_track_fitting_gap${cm}.json"
    python3 - "$TFPROD" "$dst" "$cm" <<'PY'
import json, sys
src, dst, cm = sys.argv[1], sys.argv[2], float(sys.argv[3])
d = json.load(open(src))
d["_comment_end_trim_gap_len"] = ("doc pdvd/38 arm: gap-aware end trim, %g cm in "
                                  "WCT internal units (mm). C++ default 0 = off." % cm)
d["end_trim_gap_len"] = cm * 10.0            # cm -> mm
json.dump(d, open(dst, "w"), indent=4)
PY
    [ -s "$dst" ] || { echo "mk_tf: failed to write $dst" >&2; return 3; }
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); \
        sys.exit(0 if d.get('end_trim_gap_len',0)>0 and len(d)>50 else 4)" "$dst" \
        || { echo "mk_tf: $dst has no end_trim_gap_len or is truncated" >&2; return 4; }
    echo "$dst"
}

extra_for() {
    local base=""
    [ -n "$ANISO" ] && base="-S ctpc_aniso_metric=true"
    case "$1" in
        d38ref|d38off|d38base) echo "$base" ;;
        d38g*|d38h*)   local tf; tf=$(mk_tf "${1#d38?}") || return 3
                       [ -n "$tf" ] || { echo "extra_for: empty trackfitting path for $1" >&2; return 3; }
                       echo "$base -A trackfitting_config=$tf" ;;
        *) echo "unknown arm $1" >&2; exit 2 ;;
    esac
}

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

for tag in $ARMS; do
    extra=$(extra_for "$tag") || { echo "extra_for failed for $tag -- refusing to run" >&2; exit 2; }
    case "$extra" in *"trackfitting_config="|*"trackfitting_config= "*)
        echo "REFUSING $tag: empty trackfitting_config would drop the PDVD fitting parameters" >&2; exit 2;; esac
    echo "=== arm $tag  (extra='${extra}')  jobs=$JOBS  $(date +%H:%M:%S)"
    awk 'NR>1{print $1, $2}' "$MAN" \
      | xargs -P "$JOBS" -n 2 bash -c 'stage_and_run "$0" "$1" '"$tag"' "'"$extra"'"' \
      > "$OUT/rc_${tag}.txt" 2>&1
    bad=$(awk '$4!=0' "$OUT/rc_${tag}.txt" | wc -l)
    echo "    $tag: $(wc -l < "$OUT/rc_${tag}.txt") events, nonzero rc: $bad  $(date +%H:%M:%S)"
done
echo "ALL DONE"
