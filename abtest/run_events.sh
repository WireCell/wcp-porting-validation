#!/bin/bash
# Run img and/or clus for the manifest events and snapshot outputs+metrics.
#
# Usage: ./run_events.sh <label> [img|clus|both] [manifest]
#   label    snapshot name -> snap/<label>/<det>_<run>_<evt>/
#   stage    default both (img then clus, per event)
#   manifest default events.txt
#
# Events run in parallel (one slot each); within an event stages are
# sequential.  Wall time + peak RSS captured via /usr/bin/time -v.
# Extra args to run_img_evt.sh can be passed via IMG_ARGS (e.g. IMG_ARGS="-P").

set -u
AB_DIR=$(cd "$(dirname "$0")" && pwd)
BASE_DIR=$(dirname "$AB_DIR")

LABEL=${1:?usage: run_events.sh <label> [img|clus|both] [manifest]}
STAGE=${2:-both}
MANIFEST=${3:-$AB_DIR/events.txt}
IMG_ARGS=${IMG_ARGS:-}

SNAP="$AB_DIR/snap/$LABEL"
mkdir -p "$SNAP"

run_one() {
    local det=$1 run=$2 evt=$3
    local run_padded workdir dest dflag rc t0 t1
    run_padded=$(printf '%06d' "$((10#$run))")
    workdir="$BASE_DIR/$det/work/${run_padded}_${evt}"
    dest="$SNAP/${det}_${run_padded}_${evt}"
    mkdir -p "$dest"

    dflag=""
    [ "$det" = "pdhd" ] && dflag="-d on"

    if [ "$STAGE" = "img" ] || [ "$STAGE" = "both" ]; then
        python3 "$AB_DIR/timecmd.py" "$dest/img_meta.txt" \
            "$BASE_DIR/$det/run_img_evt.sh" $dflag $IMG_ARGS "$run" "$evt" \
            > "$dest/img_stdout.log" 2>&1
        rc=$?
        cp -f "$workdir"/clusters-apa-*.tar.gz "$dest/" 2>/dev/null
        # per-anode logs (and per-anode time files if produced by -P mode)
        cp -f "$workdir"/wct_img_*.log "$dest/" 2>/dev/null
        cp -f "$workdir"/time_img_*.txt "$dest/" 2>/dev/null
        [ $rc -ne 0 ] && { echo "[$det $run $evt] IMG FAILED rc=$rc" >&2; return $rc; }
    fi

    if [ "$STAGE" = "clus" ] || [ "$STAGE" = "both" ]; then
        python3 "$AB_DIR/timecmd.py" "$dest/clus_meta.txt" \
            "$BASE_DIR/$det/run_clus_evt.sh" "$run" "$evt" \
            > "$dest/clus_stdout.log" 2>&1
        rc=$?
        cp -f "$workdir"/mabc-*.zip "$dest/" 2>/dev/null
        cp -f "$workdir"/wct_clus_*.log "$dest/" 2>/dev/null
        [ $rc -ne 0 ] && { echo "[$det $run $evt] CLUS FAILED rc=$rc" >&2; return $rc; }
    fi
    echo "[$det $run $evt] done"
    return 0
}

pids=()
while read -r det run evt; do
    case "$det" in ''|\#*) continue ;; esac
    run_one "$det" "$run" "$evt" &
    pids+=($!)
done < "$MANIFEST"

fail=0
for p in "${pids[@]}"; do
    wait "$p" || fail=1
done

echo "=== snapshot $LABEL complete (fail=$fail) ==="
exit $fail
