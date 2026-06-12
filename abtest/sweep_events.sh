#!/bin/bash
# Run img+clus over a manifest of events with bounded concurrency and
# record per-stage wall/VmHWM/rc, one TSV row per event.
#
# Usage: ./sweep_events.sh <label> [concurrency] [manifest]
#   label        -> sweep/<label>/{manifest.txt,results.tsv,<det>_<run>_<evt>/}
#   concurrency  default 16 (events in flight; matches the original
#                resource-profile run conditions)
#   manifest     rows "<det> <run> <evt>"; default = every <run>_<evt>
#                work dir of pdhd and pdvd (selection-tagged dirs skipped)
#
# Unlike run_events.sh this does NOT snapshot output archives -- it is for
# population metrics (and refreshes the work-dir outputs as a side effect).
set -u
AB_DIR=$(cd "$(dirname "$0")" && pwd)
BASE_DIR=$(dirname "$AB_DIR")

LABEL=${1:?usage: sweep_events.sh <label> [concurrency] [manifest]}
PJOBS=${2:-16}
MANIFEST=${3:-}

OUT="$AB_DIR/sweep/$LABEL"
mkdir -p "$OUT"
TSV="$OUT/results.tsv"
printf 'det\trun\tevt\timg_s\timg_kb\timg_rc\tclus_s\tclus_kb\tclus_rc\n' > "$TSV"

if [ -z "$MANIFEST" ]; then
    MANIFEST="$OUT/manifest.txt"
    : > "$MANIFEST"
    for det in pdhd pdvd; do
        for d in "$BASE_DIR/$det/work"/*/; do
            b=$(basename "$d")
            [[ "$b" =~ ^([0-9]{6})_([0-9]+)$ ]] || continue
            echo "$det ${BASH_REMATCH[1]} ${BASH_REMATCH[2]}" >> "$MANIFEST"
        done
    done
fi

run_one() {
    local det=$1 run=$2 evt=$3
    local dest="$OUT/${det}_${run}_${evt}"
    mkdir -p "$dest"
    local dflag=""
    [ "$det" = "pdhd" ] && dflag="-d on"

    python3 "$AB_DIR/timecmd.py" "$dest/img_meta.txt" \
        "$BASE_DIR/$det/run_img_evt.sh" $dflag "$run" "$evt" \
        > "$dest/img_stdout.log" 2>&1
    python3 "$AB_DIR/timecmd.py" "$dest/clus_meta.txt" \
        "$BASE_DIR/$det/run_clus_evt.sh" "$run" "$evt" \
        > "$dest/clus_stdout.log" 2>&1

    local iw ik ir cw ck cr
    ir=$(awk -F= '/^rc=/{print $2}' "$dest/img_meta.txt")
    iw=$(awk -F= '/^wall_s=/{print $2}' "$dest/img_meta.txt")
    ik=$(awk -F= '/^maxrss_kb=/{print $2}' "$dest/img_meta.txt")
    cr=$(awk -F= '/^rc=/{print $2}' "$dest/clus_meta.txt")
    cw=$(awk -F= '/^wall_s=/{print $2}' "$dest/clus_meta.txt")
    ck=$(awk -F= '/^maxrss_kb=/{print $2}' "$dest/clus_meta.txt")
    # single short O_APPEND writes are atomic
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$det" "$run" "$evt" "$iw" "$ik" "$ir" "$cw" "$ck" "$cr" >> "$TSV"
    echo "[$det $run $evt] img=${iw}s/${ik}kB rc=$ir  clus=${cw}s/${ck}kB rc=$cr"
}

running=0
while read -r det run evt; do
    case "$det" in ''|\#*) continue ;; esac
    run_one "$det" "$run" "$evt" &
    running=$((running+1))
    if [ "$running" -ge "$PJOBS" ]; then
        wait -n
        running=$((running-1))
    fi
done < "$MANIFEST"
wait

n=$(($(wc -l < "$TSV") - 1))
nfail=$(awk -F'\t' 'NR>1 && ($6!=0 || $9!=0)' "$TSV" | wc -l)
echo "=== sweep $LABEL complete: $n events, $nfail with failures -> $TSV ==="
