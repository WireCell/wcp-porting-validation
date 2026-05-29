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
#          data-sep/<n>/<n>-{img,op}-apa{0,1}.json   mabc-all-apa.zip   combined.zip

set -e

SBND_DIR=$(cd "$(dirname "$0")" && pwd)
WCP_DIR=$(cd "$SBND_DIR/.." && pwd)                 # wcp-porting-img/sbnd
WCT_BASE=/nfs/data/1/xqian/toolkit-dev

export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WCT_BASE}/wire-cell-data/sbnd/photodet:${WIRECELL_PATH}

# --- Drift / lar params (documented sim values for this jsonnet; edit as needed) ---
DL=6.2            # cm^2/s
DT=9.8            # cm^2/s
LIFETIME=6        # ms
DRIFTSPEED=1.565  # mm/us
SEMIMODEL=semi-analytical-sbnd.json

# sbnd_xin standalone chain (imports the in-tree pre-tagging clus.jsonnet)
JSONNET="$SBND_DIR/wct-clus-matching-standalone.jsonnet"
BEE_UPLOAD="$WCP_DIR/bee-upload.sh"

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

INPUTS=(
    icluster-apa0-active.npz icluster-apa0-masked.npz
    icluster-apa1-active.npz icluster-apa1-masked.npz
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
echo "Drift params: DL=$DL DT=$DT lifetime=$LIFETIME driftSpeed=$DRIFTSPEED"
echo "Upload:       $([ "$DO_UPLOAD" = 1 ] && echo 'yes (BNL BEE)' || echo 'no (build only)')"
echo "Log:          $LOG"

# --- Fresh writable workdir; stage read-only inputs as symlinks ---
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
for f in "${INPUTS[@]}"; do
    ln -s "$INPUT_DIR/$f" "$WORKDIR/$f"
done

cd "$WORKDIR"

# --- Run the standalone matching graph (outputs land in $WORKDIR) ---
echo "[wire-cell] running matching graph ..."
wire-cell \
    -l stderr \
    -l "${LOG}:debug" \
    -L debug \
    -V "reality=$REALITY" \
    -V "input=." \
    -V "semimodel_file=$SEMIMODEL" \
    -C "DL=$DL" \
    -C "DT=$DT" \
    -C "lifetime=$LIFETIME" \
    -C "driftSpeed=$DRIFTSPEED" \
    -c "$JSONNET"

echo "[wire-cell] done -> data-sep/ + mabc-all-apa.zip"

# --- Package via bee-upload.sh (single-sourced merge logic) ---
# Default: stub out the uploader so combined.zip is built but nothing is sent.
if [ "$DO_UPLOAD" = 1 ]; then
    echo "[bee] building + uploading combined.zip ..."
    bash "$BEE_UPLOAD"
else
    STUB="$WORKDIR/.noupload.sh"
    cat > "$STUB" <<'EOF'
#!/bin/bash
echo "[upload] SKIPPED (build-only). Re-run with --upload to publish: $1"
EOF
    chmod +x "$STUB"
    echo "[bee] building combined.zip (upload skipped) ..."
    UPLOAD_TO_BEE="$STUB" bash "$BEE_UPLOAD"
fi

echo
echo "=== done ==="
echo "  combined.zip: $WORKDIR/combined.zip"
echo "  reference:    diff data-sep/ against"
echo "                $INPUT_DIR/archive-runs/wct-standalone-10ev/  (mc only)"
[ "$DO_UPLOAD" = 1 ] || echo "  upload:       re-run with --upload to publish to BNL BEE"
