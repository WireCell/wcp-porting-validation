#!/bin/bash
# Original author: Haiwang Yu <hyu@bnl.gov>
# Updated by: Prabhjot Singh <prabhjot@fnal.gov>
# Changes made on 2024-06-21
# Changes include:
# Include truth cluster JSONs (tru-apa*.json) if they exist, renaming them to match the event structure in data/<n>/<n>-tru-apa{0,1}.json. This allows truth clusters to be included in the BEE event display alongside the algorithm outputs.

# Build a single BEE event-display zip from both lar output streams,
# truth clusters, and upload it. The streams contribute different algorithm
# JSONs into the same `data/<evt-idx>/` folder so each event in BEE shows
# the union.
#
# Inputs (in $PWD):
#   mabc-*.zip          each containing data/<n>/<n>-<alg>.json
#                       (e.g. mabc-apa0-face0.zip has data/0/0-clustering-apa0-face0.json;
#                        mabc-all-apa.zip has 0-clustering-global.json + 0-img-global.json)
#   data-sep/<n>/<n>-img-apa{0,1}.json   to be merged across APAs
#   data-sep/<n>/<n>-op-apa{0,1}.json    to be merged across APAs
#   tru-apa{0,1}-<n>.json                truth cluster data (optional)
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
        # so unzip at $PWD lays files in the right place without clobbering
        # files contributed by other zips (different <alg>).
        unzip -oq "$z"
    done
else
    echo "[mabc]   no mabc-*.zip in $PWD"
fi

# --- Stream B: merge per-event APA pairs into data/<n>/<n>-img.json/op.json -
if [[ -d data-sep ]]; then
    for event_dir in data-sep/*; do
        event_no=$(basename "$event_dir")
        if [[ -d "$event_dir" && "$event_no" =~ ^[0-9]+$ ]]; then
            echo "[apa]    event $event_no: merge-apa.py img/op apa0+apa1"
            python "$MERGE_APA" --inpath=data-sep --outpath="$WORK" --eventNo="$event_no" > /dev/null
        fi
    done
else
    echo "[apa]    no data-sep/ in $PWD"
fi

# --- Stream C: include truth cluster JSONs (tru-apa*.json) if they exist ----
tru_files=( tru-apa*.json )
if (( ${#tru_files[@]} )); then
    echo "[truth]  found ${#tru_files[@]} truth json file(s), processing..."
    for tru_file in "${tru_files[@]}"; do
        # Extract event number from filename: tru-apa{0,1}-<event_num>.json
        # Example: tru-apa0-5.json -> event_num=5
        if [[ $tru_file =~ ^tru-apa[0-1]-([0-9]+)\.json$ ]]; then
            event_no="${BASH_REMATCH[1]}"
            mkdir -p "$WORK/$event_no"
            # Copy truth file, renaming to match event structure
            # tru-apa{0,1}-<n>.json -> <n>-tru-apa{0,1}.json
            apa_num=$(echo "$tru_file" | sed 's/tru-apa\([0-1]\)-.*/\1/')
            cp "$tru_file" "$WORK/$event_no/${event_no}-tru-apa${apa_num}.json"
            echo "  $tru_file -> $WORK/$event_no/${event_no}-tru-apa${apa_num}.json"
        else
            echo "  WARNING: skipping $tru_file (unexpected filename format)" >&2
        fi
    done
else
    echo "[truth]  no tru-apa*.json files found, skipping truth data"
fi
shopt -u nullglob

echo
echo "=== combined event tree ==="
find "$WORK" -maxdepth 2 -type f | sort | sed 's,^,  ,'

# --- Pack + upload ---------------------------------------------------------
echo
echo "[zip]    packaging $WORK -> $OUT_ZIP"
zip -rq "$OUT_ZIP" "$WORK"

echo "[upload] $OUT_ZIP"
URL=$(BROWSER=echo bash "$UPLOAD" "$OUT_ZIP" | tail -1)

echo
echo "=== BEE event display ==="
echo "  $URL"
