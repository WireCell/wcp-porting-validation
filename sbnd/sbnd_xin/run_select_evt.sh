#!/bin/bash
# Select a region of interest from SBND SP frames using the Woodpecker GUI.  -h for help.
# Usage: ./run_select_evt.sh [mc|data] [-N n] [-a anode] <idx> <sel_tag>
#   mode:    mc (default) | data — selects the event list (input-<N>evt-<mode>)
#   -N:      event-sample size (default 10); e.g. -N 100 uses input-100evt-<mode>
#   idx:     1-based event index into the chosen sample/mode (see no-arg listing)
#   sel_tag: short label for this selection (e.g. sel1, tight, track5)
#   -a:      restrict to one anode (0 or 1)
#
# Requires run_sp_to_magnify_evt.sh to have been run first (creates per-anode
# sbnd-sp-frames-anode<N>.tar.bz2 in work/evt<ID>/).
#
# Output: work/evt<ID>_<SEL_TAG>/input/
#             sbnd-sp-frames-anode<N>.tar.bz2   (masked, per-anode, gauss<N> tag — woodpecker output)
#             selection-anode<N>.json            (tick/channel sidecar)
#             sp-frames.tar.bz2                  (combined dnnsp-tagged archive for downstream pipeline)
#
# After selection, pass -s <SEL_TAG> to the pipeline scripts:
#   ./run_sp_to_magnify_evt.sh <idx> -s <SEL_TAG>
#   ./run_img_evt.sh           <idx> -s <SEL_TAG>
#   ./run_clus_evt.sh          <idx> -s <SEL_TAG>
#   ./run_bee_img_evt.sh       <idx> -s <SEL_TAG>

set -e

SBND_DIR=$(cd "$(dirname "$0")" && pwd)

. "$SBND_DIR/_runlib.sh"

usage() {
    cat <<EOF
Select a region of interest from one SBND event's SP frames (Woodpecker GUI).

Usage: $(basename "$0") [mc|data] [-N n] [-a anode] <idx> <sel_tag>
       $(basename "$0") [mc|data] [-N n]            # list available events

  mc|data   input set (default mc)
  idx       1-based event index into the chosen sample/mode (see no-arg listing)
  sel_tag   short label for this selection (e.g. sel1, tight, track5)
  -a        restrict to one anode (0 or 1)

Requires: run_sp_to_magnify_evt.sh first (work/evt<ID>/sbnd-sp-frames-anode*.tar.bz2).
Output:   work/evt<EVT_ID>_<sel_tag>/input/  (then pass -s <sel_tag> to the pipeline)
EOF
    sbnd_common_help
}

MODE=mc
ANODE=""
_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -N) SBND_SAMPLE="$2"; shift 2 ;;
        -N*) SBND_SAMPLE="${1#-N}"; shift ;;
        mc|data) MODE="$1"; shift ;;
        -a) ANODE="$2"; shift 2 ;;
        -a*) ANODE="${1#-a}"; shift ;;
        *) _args+=("$1"); shift ;;
    esac
done
set -- "${_args[@]}"

sbnd_check_sample "$MODE" || exit 1
load_events "$MODE" || exit 1

if [ $# -eq 0 ]; then
    list_events; exit 0
fi

if [ $# -lt 2 ]; then
    usage >&2
    exit 1
fi

IDX=$1
SEL_TAG=$2

EVT_ID=$(lookup_evt_id "$IDX") || exit 1
WORKDIR="$SBND_DIR/work/evt${EVT_ID}"

# Find per-anode SP frame archives (produced by run_sp_to_magnify_evt.sh).
if [ -n "$ANODE" ]; then
    ARCHIVES=("$WORKDIR/sbnd-sp-frames-anode${ANODE}.tar.bz2")
else
    mapfile -t ARCHIVES < <(ls "$WORKDIR/sbnd-sp-frames-anode"*.tar.bz2 2>/dev/null)
fi

if [ ${#ARCHIVES[@]} -eq 0 ]; then
    echo "ERROR: no sbnd-sp-frames-anode*.tar.bz2 found in $WORKDIR" >&2
    echo "  Run: ./run_sp_to_magnify_evt.sh $IDX" >&2
    exit 1
fi

SELDIR="$SBND_DIR/work/evt${EVT_ID}_${SEL_TAG}/input"
mkdir -p "$SELDIR"

echo "Event index:      $IDX → EVT_ID=$EVT_ID"
echo "Source dir:       $WORKDIR"
echo "Found ${#ARCHIVES[@]} anode archive(s)."
echo "Selection output: $SELDIR"

export MPLBACKEND=WebAgg
echo ""
echo "Using browser-based GUI (WebAgg)."
echo "Once woodpecker prints its URL (e.g. http://127.0.0.1:8988),"
echo "forward the port from your local machine:"
echo "  ssh -L 8988:localhost:8988 $USER@$(hostname)"
echo "then open http://127.0.0.1:8988 in your browser."
echo ""
echo "GUI instructions (repeated for each anode):"
echo "  1. Drag vertically on any plane → tick range, press ENTER"
echo "  2. Drag horizontally on U plane → U channel range, press ENTER"
echo "  3. Drag horizontally on V plane → V channel range, press ENTER"
echo "  4. Drag horizontally on W plane → W channel range, press ENTER"
echo "  5. Click 'Save selection'"
echo ""

for archive in "${ARCHIVES[@]}"; do
    [ -s "$archive" ] || { echo "Skipping missing/empty: $archive"; continue; }
    echo "--- Opening GUI: $archive"
    woodpecker select "$archive" \
        --detector sbnd \
        --outdir "$SELDIR" \
        --prefix "sbnd-sp-frames"
done

# Merge per-anode masked archives back into a combined dnnsp-tagged sp-frames.tar.bz2
# that downstream scripts (run_sp_to_magnify_evt.sh, run_img_evt.sh, run_clus_evt.sh,
# run_bee_img_evt.sh) consume via -s <SEL_TAG>.
ORIG_COMBINED="$WORKDIR/sp-frames.tar.bz2"
COMBINED_OUT="$SELDIR/sp-frames.tar.bz2"
mapfile -t MASKED < <(ls "$SELDIR/sbnd-sp-frames-anode"*.tar.bz2 2>/dev/null)
if [ ${#MASKED[@]} -gt 0 ] && [ -s "$ORIG_COMBINED" ]; then
    echo ""
    echo "--- Merging masked anode archives into $COMBINED_OUT"
    python3 "$SBND_DIR/merge_sel_archives.py" \
        "$ORIG_COMBINED" "$COMBINED_OUT" "$EVT_ID" "${MASKED[@]}"
else
    echo ""
    echo "WARNING: skipped combined-archive merge"
    echo "  ORIG=$ORIG_COMBINED (exists: $([ -s "$ORIG_COMBINED" ] && echo yes || echo no))"
    echo "  masked anode count: ${#MASKED[@]}"
fi

echo ""
echo "Selection complete -> $SELDIR"
echo ""
echo "Next steps:"
echo "  ./run_sp_to_magnify_evt.sh $IDX -s $SEL_TAG"
echo "  ./run_img_evt.sh           $IDX -s $SEL_TAG"
echo "  ./run_clus_evt.sh          $IDX -s $SEL_TAG"
echo "  ./run_bee_img_evt.sh       $IDX -s $SEL_TAG"
