#!/bin/bash
# Step 1: map RSE (run,subRun,event) -> reco1 artROOT files via a samweb definition.
#
# For each event in the RSE CSV, query the samweb definition for the reco1 file
# whose sbnd.event_number_list metadata contains that event (matched as an
# underscore-delimited token: %_<event>_%). Emit the unique list of file paths
# that cover the RSEs, plus a list of RSEs not found in the definition.
#
# Run INSIDE the SL7 apptainer with sam_web_client set up.
#   samweb has NO event_number dimension; sbnd.event_number_list is the key.
#
# Usage: find_reco1_files.sh <DEF> <RSE_CSV> <OUT_FILES_LST> <OUT_MISSING_LST>

set -u
DEF="$1"; CSV="$2"; OUT_FILES="$3"; OUT_MISSING="$4"
export SAM_EXPERIMENT=sbnd

TMP=$(mktemp -d)
RAW="$TMP/raw_files.txt"       # every file hit (may repeat)
MAP="$TMP/rse_to_file.txt"     # rse -> file mapping (audit trail)
: > "$RAW"; : > "$MAP"; : > "$OUT_MISSING"

n=0
tail -n +2 "$CSV" | while IFS=, read -r RUN SUB EVT; do
    EVT="${EVT//[$'\r\n ']/}"; [ -z "$EVT" ] && continue
    n=$((n+1))
    hits=$(samweb -e sbnd list-files \
        "defname:$DEF and run_number $RUN and sbnd.event_number_list %_${EVT}_%" 2>/dev/null)
    if [ -z "$hits" ]; then
        echo "$RUN,$SUB,$EVT" >> "$OUT_MISSING"
        echo "MISS  $RUN,$SUB,$EVT" >&2
    else
        while read -r f; do
            [ -z "$f" ] && continue
            echo "$f" >> "$RAW"
            echo "$RUN,$SUB,$EVT -> $f" >> "$MAP"
        done <<< "$hits"
        echo "OK    $RUN,$SUB,$EVT -> $(echo "$hits" | head -1)" >&2
    fi
done

# Resolve unique files to physical /pnfs paths.
: > "$OUT_FILES"
sort -u "$RAW" | while read -r f; do
    loc=$(samweb -e sbnd locate-file "$f" 2>/dev/null | head -1)
    dir=$(echo "$loc" | grep -oE '/pnfs[^ ()]*' | head -1)
    if [ -n "$dir" ]; then
        echo "$dir/$f" >> "$OUT_FILES"
    else
        echo "NOLOC $f" >&2
    fi
done

cp "$MAP" "${OUT_FILES%.lst}.map.txt" 2>/dev/null

NRSE=$(tail -n +2 "$CSV" | grep -cve '^[[:space:]]*$')
NMISS=$(grep -cve '^[[:space:]]*$' "$OUT_MISSING")
NFILES=$(grep -cve '^[[:space:]]*$' "$OUT_FILES")
echo "======================================================" >&2
echo "RSE total:      $NRSE" >&2
echo "RSE found:      $((NRSE-NMISS))" >&2
echo "RSE missing:    $NMISS   -> $OUT_MISSING" >&2
echo "unique files:   $NFILES  -> $OUT_FILES" >&2
echo "======================================================" >&2
rm -rf "$TMP"
