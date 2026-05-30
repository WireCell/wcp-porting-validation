#!/bin/bash
# Standalone SBND charge-light (Q/L) matching chain — no LArSoft.
# Reproduces yuhw's wct-clus-matching-standalone.jsonnet run + BEE packaging.
#
# Usage: ./run_clust_QL_evt.sh [mc|data] [--upload]
#   mc   (default): input-10evt-mc,   reality=sim
#   data:           input-10evt-data, reality=data
#   --upload:       also upload combined.zip to the BNL BEE server
#                   (default: build combined.zip only, no network)
#
# Input  (read-only, yuhw's): input_files/input-10evt-<mode>/
#          icluster-apa{0,1}-{active,masked}.npz  opflash_apa{0,1}.tar.gz
# Output (writable):          work/ql_<mode>/
#          mabc-all-apa.zip   (single self-contained BEE zip: per event the
#          img/clustering charge layers, the dead-area patches, AND the optical
#          op/flash + Q/L-matching layer — all dumped by the all-APA MABC. No
#          per-APA data-sep JSON and no combine step any more.)

set -e

SBND_DIR=$(cd "$(dirname "$0")" && pwd)
WCP_DIR=$(cd "$SBND_DIR/.." && pwd)                 # wcp-porting-img/sbnd
WCT_BASE=/nfs/data/1/xqian/toolkit-dev

export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WCT_BASE}/wire-cell-data/sbnd/photodet:${WIRECELL_PATH}

# --- Drift / lar params (documented sim values for this jsonnet; edit as needed) ---
# NOTE: drift_speed is NOT passed here; it comes from the common SBND config
# (pgrapher/experiment/sbnd/simparams.jsonnet: 1.563 mm/us) so charge and light
# matching share one value.
DL=6.2            # cm^2/s
DT=9.8            # cm^2/s
LIFETIME=6        # ms
SEMIMODEL=semi-analytical-sbnd.json

# sbnd_xin standalone chain (imports the in-tree pre-tagging clus.jsonnet)
JSONNET="$SBND_DIR/wct-clus-matching-standalone.jsonnet"
# Direct BEE uploader (the all-APA MABC already writes one self-contained zip,
# so there is no longer a separate combine step).
BEE_UPLOADER="$WCP_DIR/upload-to-bee.sh"

# --- Args ---
MODE=mc
DO_UPLOAD=0
for arg in "$@"; do
    case "$arg" in
        mc|data) MODE="$arg" ;;
        --upload) DO_UPLOAD=1 ;;
        *) echo "ERROR: unknown argument '$arg' (use: [mc|data] [--upload])" >&2; exit 1 ;;
    esac
done

case "$MODE" in
    mc)   REALITY=sim ;;
    data) REALITY=data ;;
esac

INPUT_DIR="$SBND_DIR/input_files/input-10evt-$MODE"
WORKDIR="$SBND_DIR/work/ql_$MODE"
LOG="$WORKDIR/wct_clus_QL_$MODE.log"

# Dead (masked) clusters are no longer read from a file: the matching graph images
# them in-toolkit from the SP frames (2-view dead area).  Only the live (active)
# clusters and the opflash archives are staged from yuhw's inputs.
INPUTS=(
    icluster-apa0-active.npz
    icluster-apa1-active.npz
    opflash_apa0.tar.gz      opflash_apa1.tar.gz
)

# --- Sanity ---
[ -f "$JSONNET" ]   || { echo "ERROR: missing jsonnet: $JSONNET" >&2; exit 1; }
[ -d "$INPUT_DIR" ] || { echo "ERROR: missing input dir: $INPUT_DIR" >&2; exit 1; }
for f in "${INPUTS[@]}"; do
    [ -f "$INPUT_DIR/$f" ] || { echo "ERROR: missing input: $INPUT_DIR/$f" >&2; exit 1; }
done

echo "Mode:         $MODE  (reality=$REALITY)"
echo "Input:        $INPUT_DIR"
echo "Work dir:     $WORKDIR"
echo "Drift params: DL=$DL DT=$DT lifetime=$LIFETIME (drift_speed from common config)"
echo "Upload:       $([ "$DO_UPLOAD" = 1 ] && echo 'yes (BNL BEE)' || echo 'no (build only)')"
echo "Log:          $LOG"

# --- Fresh writable workdir; stage read-only inputs as symlinks ---
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
for f in "${INPUTS[@]}"; do
    ln -s "$INPUT_DIR/$f" "$WORKDIR/$f"
done

# --- Assemble the combined SP-frame archive for in-graph dead-area imaging ---
# The matching graph images the 2-view dead/masked clusters from the SP frames.
# FrameFileSource reads a single archive, so we concatenate the per-event
# work/evt<ID>/sp-frames.tar.bz2 (each member uniquely suffixed by event id) in
# the SAME event order the active-cluster npz streams them, so PointTreeBuilding
# pairs /live (active) and /dead (imaged) by event.
FRAMES="sp-frames-10evt.tar.bz2"
EVENT_IDS=$(python3 -c "
import numpy as np, re, sys
z = np.load('$INPUT_DIR/icluster-apa0-active.npz')
seen = []
for k in z.files:
    m = re.match(r'cluster_(\d+)_', k)
    if m and m.group(1) not in seen:
        seen.append(m.group(1))
print(' '.join(seen))
") || { echo "ERROR: could not read event order from $INPUT_DIR/icluster-apa0-active.npz" >&2; exit 1; }
echo "Event order:  $EVENT_IDS"

FRAMES_STAGE="$WORKDIR/frames_stage"
mkdir -p "$FRAMES_STAGE"
FRAME_MEMBERS=()
for e in $EVENT_IDS; do
    SPF="$SBND_DIR/work/evt$e/sp-frames.tar.bz2"
    if [ ! -s "$SPF" ]; then
        echo "ERROR: missing per-event SP frames: $SPF" >&2
        echo "  Generate it first: ./run_sp_to_magnify_evt.sh <idx-for-evt$e>" >&2
        exit 1
    fi
    tar xjf "$SPF" -C "$FRAMES_STAGE"
    for m in chanmask_bad channels_dnnsp frame_dnnsp summary_dnnsp tickinfo_dnnsp; do
        FRAME_MEMBERS+=("${m}_${e}.npy")
    done
done
tar cjf "$WORKDIR/$FRAMES" -C "$FRAMES_STAGE" "${FRAME_MEMBERS[@]}"
echo "[frames]      assembled $WORKDIR/$FRAMES ($(echo "$EVENT_IDS" | wc -w) events)"

cd "$WORKDIR"

# --- Run the standalone matching graph (outputs land in $WORKDIR) ---
echo "[wire-cell] running matching graph ..."
wire-cell \
    -l stderr \
    -l "${LOG}:debug" \
    -L debug \
    -V "reality=$REALITY" \
    -V "input=." \
    -V "frames=$FRAMES" \
    -V "semimodel_file=$SEMIMODEL" \
    -C "DL=$DL" \
    -C "DT=$DT" \
    -C "lifetime=$LIFETIME" \
    -c "$JSONNET"

echo "[wire-cell] done -> mabc-all-apa.zip (self-contained BEE zip)"

OUT_ZIP="$WORKDIR/mabc-all-apa.zip"
[ -f "$OUT_ZIP" ] || { echo "ERROR: expected BEE zip not produced: $OUT_ZIP" >&2; exit 1; }

# --- Upload (optional) ---------------------------------------------------------
# mabc-all-apa.zip is already the complete event-display zip (charge img/
# clustering + dead area + optical op/flash), so we upload it directly — no
# merge-apa.py / bee-upload.sh combine step.
if [ "$DO_UPLOAD" = 1 ]; then
    echo "[bee] uploading mabc-all-apa.zip ..."
    URL=$(BROWSER=echo bash "$BEE_UPLOADER" "$OUT_ZIP" | tail -1)
    echo "[bee] $URL"
else
    echo "[bee] upload skipped (re-run with --upload to publish)"
fi

echo
echo "=== done ==="
echo "  BEE zip:   $OUT_ZIP"
[ "$DO_UPLOAD" = 1 ] || echo "  upload:    re-run with --upload to publish to BNL BEE"
