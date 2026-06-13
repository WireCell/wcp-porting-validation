#!/bin/bash
# Combined imaging + clustering Bee upload for one run (all events), one link.
#
# Per event, the Bee instance set (the "7 image situation") is:
#   imaging-group0123 / imaging-group4567     after imaging step      (bee-blobs)
#   clustering-group0123 / clustering-group4567  after per-anode clustering
#   clustering-global                         after all-anode clustering
#   channel-deadarea-group0123 / -group4567   dead area
#
# Imaging instances come from clusters-apa-anode<N>-ms-active.tar.gz via
# wirecell-img bee-blobs (anodes 0-3 and 4-7 merged by drift side).
# Clustering + dead instances are taken from work/<run>_<evt>/mabc-all-apa.zip,
# so run ./run_clus_evt.sh <run> all first.
#
# Usage: ./run_bee_combined_evt.sh <run> [subrun]
#        ./run_bee_combined_evt.sh            # list available runs

set -e

PDVD_DIR=$(cd "$(dirname "$0")" && pwd)
. "$PDVD_DIR/_runlib.sh"

if [ $# -eq 0 ]; then
    list_runs; exit 0
fi
RUN=$1
SUBRUN=${2:-0}

RUN_STRIPPED=$(echo "$RUN" | sed 's/^0*//')
[ -z "$RUN_STRIPPED" ] && RUN_STRIPPED=0
RUN_PADDED=$(printf '%06d' "$RUN_STRIPPED")

# Anodes 0-3 bottom-drift (-x); anodes 4-7 top-drift (+x).  Must match
# wct-img-2-bee.py / run_bee_img_evt.sh.
bee_anode_args() {
    if [ "$1" -le 3 ]; then
        echo '--speed "-1.57*mm/us" --t0 "0*us" --x0 "-341.5*cm"'
    else
        echo '--speed "1.57*mm/us" --t0 "0*us" --x0 "341.5*cm"'
    fi
}

find_clus_input() {
    local evt=$1
    local wd="$PDVD_DIR/work/${RUN_PADDED}_${evt}"
    if ls "$wd/clusters-apa-anode"*"-ms-active.tar.gz" >/dev/null 2>&1; then
        echo "$wd"; return 0
    fi
    for rn in "run${RUN}" "run${RUN_PADDED}" "run${RUN_STRIPPED}"; do
        local rdir="$PDVD_DIR/input_data/$rn"
        [ -d "$rdir" ] || continue
        for en in "evt${evt}" "evt_${evt}"; do
            local c="$rdir/$en"
            if [ -d "$c" ] && ls "$c/clusters-apa-anode"*"-ms-active.tar.gz" >/dev/null 2>&1; then
                echo "$c"; return 0
            fi
        done
    done
    return 1
}

mapfile -t _events < <(discover_events "$RUN" "$RUN_PADDED")
if [ ${#_events[@]} -eq 0 ]; then
    echo "no events found for run=$RUN under input_data/ or work/" >&2; exit 1
fi
echo "Found ${#_events[@]} event(s) for run=$RUN: ${_events[*]}"

cd "$PDVD_DIR"
rm -rf data
mkdir -p data

_idx=0
for _e in "${_events[@]}"; do
    _clus_input=$(find_clus_input "$_e") || { echo "[skip] evt=$_e: no active tarballs" >&2; continue; }
    _apa0=$(ls "$_clus_input/clusters-apa-anode"*"-ms-active.tar.gz" 2>/dev/null | head -1)
    _event_no=$(tar tzf "$_apa0" | head -1 | sed -E 's/.*cluster_([0-9]+)_.*/\1/')
    echo "$_event_no" | grep -qE '^[0-9]+$' || { echo "[skip] evt=$_e: cannot parse art event" >&2; continue; }

    echo "  [evt $_e -> bee index $_idx]  art_event=$_event_no  clus: $_clus_input"
    mkdir -p "data/$_idx"

    # --- imaging instances (active blobs, merged by drift side) ---
    for _spec in "imaging-group0123 0 0 1 2 3" "imaging-group4567 4 4 5 6 7"; do
        # shellcheck disable=SC2086
        set -- $_spec
        _gname=$1; _geoidx=$2; shift 2
        _files=()
        for _j in "$@"; do
            _f="$_clus_input/clusters-apa-anode${_j}-ms-active.tar.gz"
            [ -s "$_f" ] && _files+=("$_f")
        done
        [ ${#_files[@]} -gt 0 ] || continue
        # shellcheck disable=SC2046
        eval wirecell-img bee-blobs -g protodunevd -s uniform -d 1 \
            --rse "$RUN_STRIPPED" "$SUBRUN" "$_event_no" \
            $(bee_anode_args "$_geoidx") \
            -o "data/${_idx}/${_idx}-${_gname}.json" \
            "${_files[@]}"
    done

    # --- clustering + dead instances (from mabc-all-apa.zip, re-indexed) ---
    _mabc="$PDVD_DIR/work/${RUN_PADDED}_${_e}/mabc-all-apa.zip"
    if [ -s "$_mabc" ]; then
        _tmp=$(mktemp -d /home/xqian/tmp/beecomb.XXXXXX)
        unzip -q -o "$_mabc" -d "$_tmp"
        for _f in "$_tmp"/data/0/0-*.json; do
            [ -e "$_f" ] || continue
            _suffix=$(basename "$_f"); _suffix=${_suffix#0-}   # strip leading "0-"
            cp "$_f" "data/${_idx}/${_idx}-${_suffix}"
        done
        rm -rf "$_tmp"
    else
        echo "  [warn] evt=$_e: no mabc-all-apa.zip (run run_clus_evt.sh first) -> no clustering/dead" >&2
    fi

    _idx=$((_idx + 1))
done

[ "$_idx" -gt 0 ] || { echo "ERROR: no events produced Bee data" >&2; exit 1; }

_zip="upload-combined-run${RUN_PADDED}.zip"
rm -f "$_zip"
zip -rq "$_zip" data
echo "Uploading $_zip ($_idx event(s)) ..."
./upload-to-bee.sh "$_zip"
