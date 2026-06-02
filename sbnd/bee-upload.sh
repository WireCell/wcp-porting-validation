#!/bin/bash
# Build a single BEE event-display zip from both lar output streams and
# upload it. The two streams contribute different algorithm JSONs into the
# same `data/<evt-idx>/` folder so each event in BEE shows the union.
#
# Inputs (in $PWD):
#   mabc-*.zip          each containing data/<n>/<n>-<alg>.json
#                       (e.g. mabc-apa0-face0.zip has data/0/0-clustering-apa0-face0.json;
#                        mabc-all-apa.zip has 0-clustering-global.json + 0-img-global.json)
#   data-sep/<n>/<n>-img-apa{0,1}.json   to be merged across APAs
#   data-sep/<n>/<n>-op-apa{0,1}.json    to be merged across APAs
#
# Output:
#   data/<n>/<n>-<alg>.json (union of the above)
#   combined.zip            zip of data/, uploaded to BEE
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPLOAD="${UPLOAD_TO_BEE:-$HERE/../upload-to-bee.sh}"
MERGE_APA="$HERE/merge-apa.py"
OUT_ZIP="combined.zip"
WORK="data"

if [[ ! -x "$UPLOAD" ]]; then
    echo "ERROR: upload helper not found: $UPLOAD" >&2
    exit 1
fi

# Fresh workspace.
rm -rf "$WORK" "$OUT_ZIP"
mkdir -p "$WORK"

# --- Stream A: drop mabc-*.zip contents into data/<n>/  (cluster JSONs) -----
shopt -s nullglob
mabc_zips=( mabc-*.zip )
if (( ${#mabc_zips[@]} )); then
    echo "[mabc]   extracting ${#mabc_zips[@]} zip(s) -> data/"
    for z in "${mabc_zips[@]}"; do
        # Each zip's internal layout is already "data/<n>/<n>-<alg>.json",
        # so extracting at $PWD lays files in the right place without
        # clobbering files contributed by other zips (different <alg>).
        # python3 -m zipfile is used in place of unzip(1) so this script
        # works inside the SL7 apptainer image (no unzip there).
        python3 -m zipfile -e "$z" .
    done
else
    echo "[mabc]   no mabc-*.zip in $PWD"
fi
shopt -u nullglob

# --- Stream B: merge per-event APA pairs into data/<n>/<n>-img.json/op.json -
if [[ -d data-sep ]]; then
    for event_dir in data-sep/*; do
        event_no=$(basename "$event_dir")
        if [[ -d "$event_dir" && "$event_no" =~ ^[0-9]+$ ]]; then
            echo "[apa]    event $event_no: merge-apa.py img/op apa0+apa1"
            python3 "$MERGE_APA" --inpath=data-sep --outpath="$WORK" --eventNo="$event_no" > /dev/null
        fi
    done
else
    echo "[apa]    no data-sep/ in $PWD"
fi

echo
echo "=== combined event tree ==="
find "$WORK" -maxdepth 2 -type f | sort | sed 's,^,  ,'

# --- Pack + upload ---------------------------------------------------------
echo
echo "[zip]    packaging $WORK -> $OUT_ZIP"
# python3 -m zipfile (instead of zip(1)) for SL7 apptainer compatibility.
python3 -m zipfile -c "$OUT_ZIP" "$WORK"

echo "[upload] $OUT_ZIP"
URL=$(BROWSER=echo bash "$UPLOAD" "$OUT_ZIP" | tail -1)

echo
echo "=== BEE event display ==="
echo "  $URL"
