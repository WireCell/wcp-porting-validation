#!/bin/bash
# Combined Bee upload for one run (all events), one link.
#
# Every Bee instance is taken straight from work/<run>_<evt>/mabc-all-apa.zip,
# which MultiAlgBlobClustering dumps with all charge points already in it:
#   img-global                              raw imaged charge (pre-pipeline);
#                                           its cluster ids are the SAME
#                                           enumeration as the op dump's
#                                           op_cluster_ids -> check the
#                                           flash<->cluster pairing HERE
#   clustering-global                       all-anode clustered result
#   op                                      optical flashes + Q/L predicted PE
#   channel-deadarea-group0123 / -group4567 dead area
# so just run ./run_clus_evt.sh <run> all first (no separate imaging pass —
# a wirecell-img bee-blobs image would carry its own unrelated cluster
# numbering and NOT match the op instance; PDHD parity).
#
# Usage: ./run_bee_combined_evt.sh [-e evt,evt,...] <run>
#        ./run_bee_combined_evt.sh            # list available runs
#
# -e: restrict the link to a comma-separated subset of event indices (in the
#     order they appear); default (no -e) uploads every discovered event.

set -e

PDVD_DIR=$(cd "$(dirname "$0")" && pwd)
. "$PDVD_DIR/_runlib.sh"

EVT_SUBSET=""
_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        -e) EVT_SUBSET="$2"; shift 2 ;;
        -e*) EVT_SUBSET="${1#-e}"; shift ;;
        *) _args+=("$1"); shift ;;
    esac
done
set -- "${_args[@]}"

if [ $# -eq 0 ]; then
    list_runs; exit 0
fi
RUN=$1

RUN_STRIPPED=$(echo "$RUN" | sed 's/^0*//')
[ -z "$RUN_STRIPPED" ] && RUN_STRIPPED=0
RUN_PADDED=$(printf '%06d' "$RUN_STRIPPED")

mapfile -t _events < <(discover_events "$RUN" "$RUN_PADDED")
if [ ${#_events[@]} -eq 0 ]; then
    echo "no events found for run=$RUN under input_data/ or work/" >&2; exit 1
fi
if [ -n "$EVT_SUBSET" ]; then
    # keep only the requested indices, in the order given on -e
    _want=$(echo "$EVT_SUBSET" | tr ',' ' ')
    _sel=()
    for _w in $_want; do
        for _e in "${_events[@]}"; do
            [ "$_e" = "$_w" ] && { _sel+=("$_w"); break; }
        done
    done
    [ ${#_sel[@]} -gt 0 ] || { echo "no requested events (-e $EVT_SUBSET) found for run=$RUN" >&2; exit 1; }
    _events=("${_sel[@]}")
fi
echo "Found ${#_events[@]} event(s) for run=$RUN: ${_events[*]}"

cd "$PDVD_DIR"
rm -rf data
mkdir -p data

# ── Optional per-CRP imaging instances (PDVD_BEE_PER_CRP=1) ───────────────────
# Default OFF => output byte-identical to legacy (global instances only).
# When on, each event additionally carries img-crp0..img-crp7 (raw per-CRP
# imaged charge via `wirecell-img bee-blobs`, same geometry as run_bee_img_evt.sh),
# so the one Bee viewer holds the global instances plus a per-CRP breakdown.
# Geometry must stay in sync with run_bee_img_evt.sh:bee_anode_args / wct-img-2-bee.py.
PER_CRP="${PDVD_BEE_PER_CRP:-0}"
bee_anode_args() {
    local idx=$1
    if [ "$idx" -le 3 ]; then
        echo '--speed "-1.57*mm/us" --t0 "0*us" --x0 "-341.5*cm"'
    else
        echo '--speed "1.57*mm/us" --t0 "0*us" --x0 "341.5*cm"'
    fi
}

_idx=0
for _e in "${_events[@]}"; do
    # All Bee instances (img-global, clustering-global, op, dead area) come from
    # the all-anode MABC dump; just re-index its data/0/0-*.json to this event.
    _mabc="$PDVD_DIR/work/${RUN_PADDED}_${_e}/mabc-all-apa.zip"
    if [ ! -s "$_mabc" ]; then
        echo "[skip] evt=$_e: no mabc-all-apa.zip (run run_clus_evt.sh first)" >&2
        continue
    fi

    echo "  [evt $_e -> bee index $_idx]  mabc: $_mabc"
    mkdir -p "data/$_idx"

    _tmp=$(mktemp -d /home/xqian/tmp/beecomb.XXXXXX)
    unzip -q -o "$_mabc" -d "$_tmp"
    for _f in "$_tmp"/data/0/0-*.json; do
        [ -e "$_f" ] || continue
        _suffix=$(basename "$_f"); _suffix=${_suffix#0-}   # strip leading "0-"
        cp "$_f" "data/${_idx}/${_idx}-${_suffix}"
    done
    rm -rf "$_tmp"

    # Per-CRP imaging instances (raw charge per anode/CRP), from the same
    # clustering-input tarballs the workdir was built from.
    if [ "$PER_CRP" = 1 ]; then
        _wd="$PDVD_DIR/work/${RUN_PADDED}_${_e}"
        _a0=$(ls "$_wd"/clusters-apa-anode*-ms-active.tar.gz 2>/dev/null | head -1)
        if [ -z "$_a0" ]; then
            echo "  [evt $_e] PER_CRP: no clusters-apa tarballs in $_wd; skipping per-CRP" >&2
        else
            _evno=$(tar tzf "$_a0" 2>/dev/null | head -1 | sed -E 's/.*cluster_([0-9]+)_.*/\1/')
            if echo "$_evno" | grep -qE '^[0-9]+$'; then
                for _n in 0 1 2 3 4 5 6 7; do
                    _ct="$_wd/clusters-apa-anode${_n}-ms-active.tar.gz"
                    [ -s "$_ct" ] || continue
                    _ga=$(bee_anode_args "$_n")
                    # shellcheck disable=SC2086
                    eval wirecell-img bee-blobs \
                        -g protodunevd -s uniform -d 1 \
                        --rse "$RUN_STRIPPED" 0 "$_evno" \
                        $_ga \
                        -o "data/${_idx}/${_idx}-img-crp${_n}.json" \
                        "$_ct" \
                        || echo "  [evt $_e] PER_CRP: bee-blobs failed for crp${_n}" >&2
                done
            else
                echo "  [evt $_e] PER_CRP: cannot parse event number from $_a0" >&2
            fi
        fi
    fi

    _idx=$((_idx + 1))
done

[ "$_idx" -gt 0 ] || { echo "ERROR: no events produced Bee data" >&2; exit 1; }

_zip="upload-combined-run${RUN_PADDED}.zip"
rm -f "$_zip"
zip -rq "$_zip" data
echo "Uploading $_zip ($_idx event(s)) ..."
./upload-to-bee.sh "$_zip"
