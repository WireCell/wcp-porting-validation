#!/bin/bash
# doc pdvd/40 sec 15.10 — build one Bee set per event holding BOTH arms as two
# Bee "events" on the same geometry, so the knob toggles in one tab:
#     event 0 = OFF arm, event 1 = ON arm
#
# Usage:  ./d40r3_bee_compare.sh [-n] <run>_<evt> [<run>_<evt> ...]
#   -n         build the zips but do not upload
#   OFF_ARM=   arm tag for event 0 (default d41base2, knob OFF)
#   ON_ARM=    arm tag for event 1 (default d41fix20x, bad_blob_max_run=20cm)
#   OUTDIR=    where to build (default /home/xqian/tmp/d40r3_bee)
#
# Each input is work/<run>_<evt>_<arm>/mabc-pr.zip, whose instances live under
# data/0/0-*.json; they are re-indexed to data/<i>/<i>-*.json exactly the way
# run_bee_combined_evt.sh does it, then zipped and posted with upload-to-bee.sh.
# Verify every returned UUID before quoting it (curl the set URL, expect 200 and
# both events listed) — the uploader builds the URL from whatever came back.
set -e

PDVD_DIR=/home/xqian/toolkit-dev/wcp-porting-img/pdvd
OFF_ARM="${OFF_ARM:-d41base2}"
ON_ARM="${ON_ARM:-d41fix20x}"
OUTDIR="${OUTDIR:-/home/xqian/tmp/d40r3_bee}"
DRY=0
[ "$1" = "-n" ] && { DRY=1; shift; }
mkdir -p "$OUTDIR"
cd "$OUTDIR"

for ev in "$@"; do
    work="$OUTDIR/w_$ev"; out="$OUTDIR/beecmp_${ev}.zip"
    mkdir -p "$work"; cd "$work"; mkdir -p data/0 data/1
    i=0
    for arm in "$OFF_ARM" "$ON_ARM"; do
        src="$PDVD_DIR/work/${ev}_${arm}/mabc-pr.zip"
        [ -s "$src" ] || { echo "missing $src" >&2; exit 1; }
        t="$work/x_$arm"; mkdir -p "$t"
        unzip -q -o "$src" -d "$t"
        for f in "$t"/data/0/0-*.json; do
            s=$(basename "$f"); s=${s#0-}
            cp "$f" "data/${i}/${i}-${s}"
        done
        i=$((i+1))
    done
    rm -f "$out"; zip -rq "$out" data
    echo "built $out  (0=$OFF_ARM 1=$ON_ARM)"
    cd "$OUTDIR"
    if [ "$DRY" = 0 ]; then
        url=$(bash "$PDVD_DIR/upload-to-bee.sh" "$out" | tail -1)
        code=$(curl -k -s -o /dev/null -w '%{http_code}' "$url")
        echo "$ev $url http=$code"
    fi
done
