#!/bin/bash
# BlobDepoFill time_offset calibration (wcfm/docs/02 sec 4): image ONE event with the truth
# tiers at a ladder of time offsets and print the captured true-charge fraction and the
# ghost fractions per offset (iso_baseline.py --quick).  The maximum of "captured" is the
# offset; it lands in wcfm_params.jsonnet depofill_time_offset.
# Usage: scripts/scan_time_offset.sh <run> <evt> [offsets_us...]   (default ladder 0..125 by 12.5)
set -u
WCFM_DIR=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:?run}; EVT=${2:?evt}; shift 2
OFFS=("$@"); [ ${#OFFS[@]} -eq 0 ] && OFFS=(0 12.5 25 37.5 50 56.25 62.5 68.75 75 87.5 100 112.5 125)
RUN6=$(printf '%06d' "$((10#$RUN))")
for us in "${OFFS[@]}"; do
    tag=$(echo "$us" | tr '.' 'p')
    "$WCFM_DIR/run_img_evt.sh" -O "_toff${tag}" -t "$us" "$RUN" "$EVT" > "$WCFM_DIR/work/.scan_toff_${tag}.log" 2>&1 \
        && echo "offset ${us} us: done" || echo "offset ${us} us: FAILED (see work/.scan_toff_${tag}.log)"
done
python3 "$WCFM_DIR/scripts/iso_baseline.py" --quick "$WCFM_DIR"/work/${RUN6}_${EVT}_toff*
