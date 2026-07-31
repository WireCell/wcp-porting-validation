#!/bin/bash
# Extract an SBND reco1 art/LArSoft ROOT file into a standalone-chain
# sample dir -- directly with the toolkit, no LArSoft.  Run with -h for help.
#
# Usage: ./run_reco1_dump.sh [-caf none|product|auto|override:<ns>] [-mc] [-t tag] [reco1.root]
#   reco1.root  input art file; default: the single .root under input_files_reco1/
#   -caf        frame_apply_at_caf mode for the opflash tensor-set metadata
#               (default auto = ported FrameShift derivation, ~0.26 us low;
#                product = authoritative FrameShiftInfo::fFrameApplyAtCaf,
#                needs a *_frameshift.root input; none = omit key;
#                override:<ns> = fixed value)
#   -mc         read the SBND *MC* reco1 product names (simtpc2d/dnnsp under the
#               DetSim process) instead of the data ones (sptpc2d, Reco1).  MC
#               files carry no PTB/TDC/DAQ-header products, so use with -caf none
#               (the default).  See docs/67_round2-patrec-10evt.md.
#   -t          output tag; sample dir becomes input_files_reco1/extracted-<tag>/
#               (default: input file basename up to the first '-')
#
# Output: input_files_reco1/extracted-<tag>/{frames-dnn.tar.bz2,opflash_apa{0,1}.tar.gz}
#         (yuhw standalone-sample layout; ALL events of the art file)
#
# Then run the unchanged chain against it:
#   export SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-<tag>
#   ./run_img_evt.sh data            # list events
#   ./run_img_evt.sh data <idx>      # imaging -> work/evt<ID>/icluster-*.npz
#   ./run_ql_evt.sh  data <idx>      # clustering + Q/L matching (iterate freely)

set -e

SBND_DIR=$(cd "$(dirname "$0")" && pwd)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WIRECELL_PATH}

CAF_MODE=auto
CAF_OVERRIDE=""
TAG=""
INPUT=""
# Empty => the jsonnet omits the keys => C++ defaults = the data product names.
WIRE_PRODUCT=""
BADMASK_PRODUCT=""
SUMMARY_PRODUCT=""

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -mc|--mc)
            WIRE_PRODUCT='recob::Wires_simtpc2d_dnnsp_DetSim.'
            BADMASK_PRODUCT='ints_simtpc2d_badmasks_DetSim.'
            SUMMARY_PRODUCT='doubles_simtpc2d_wienersummary_DetSim.'
            shift ;;
        -caf) CAF_MODE="$2"; shift 2 ;;
        -caf*) CAF_MODE="${1#-caf}"; shift ;;
        -t) TAG="$2"; shift 2 ;;
        -t*) TAG="${1#-t}"; shift ;;
        *) INPUT="$1"; shift ;;
    esac
done

case "$CAF_MODE" in
    override:*) CAF_OVERRIDE="${CAF_MODE#override:}"; CAF_MODE=override ;;
    none|product|auto) ;;
    *) echo "ERROR: bad -caf mode '$CAF_MODE' (none|product|auto|override:<ns>)" >&2; exit 1 ;;
esac

if [ -z "$INPUT" ]; then
    INPUT=$(ls "$SBND_DIR"/input_files_reco1/*.root 2>/dev/null | head -1)
    if [ -z "$INPUT" ]; then
        echo "ERROR: no input given and no .root under input_files_reco1/" >&2
        exit 1
    fi
fi
INPUT=$(readlink -f "$INPUT")
[ -f "$INPUT" ] || { echo "ERROR: no such file: $INPUT" >&2; exit 1; }

if [ -z "$TAG" ]; then
    TAG=$(basename "$INPUT" .root)
    TAG=${TAG%%-*}
fi

OUTDIR="$SBND_DIR/input_files_reco1/extracted-${TAG}"
if [ -e "$OUTDIR/frames-dnn.tar.bz2" ]; then
    echo "ERROR: $OUTDIR already holds an extraction; pick a new -t tag" >&2
    exit 1
fi
mkdir -p "$OUTDIR"
LOG="$OUTDIR/reco1-dump.log"

echo "Input:   $INPUT"
echo "Output:  $OUTDIR"
echo "caf:     $CAF_MODE${CAF_OVERRIDE:+ ($CAF_OVERRIDE ns)}"
echo "Log:     $LOG"

cd "$SBND_DIR"
rm -f "$LOG"
wire-cell \
    -l stderr \
    -l "${LOG}:debug" \
    -L info \
    --tla-str "input=${INPUT}" \
    --tla-str "output_dir=${OUTDIR}" \
    --tla-str "caf_offset_mode=${CAF_MODE}" \
    --tla-str "caf_offset_override=${CAF_OVERRIDE:-0}" \
    --tla-str "wire_product=${WIRE_PRODUCT}" \
    --tla-str "badmask_product=${BADMASK_PRODUCT}" \
    --tla-str "summary_product=${SUMMARY_PRODUCT}" \
    -c wct-reco1-dump.jsonnet

echo
echo "Extraction done -> $OUTDIR"
echo "Events:"
tar tjf "$OUTDIR/frames-dnn.tar.bz2" | grep '^frame_dnnsp_' | sed 's/frame_dnnsp_/  /;s/\.npy//'
echo
echo "Next:"
echo "  export SBND_INPUT_DIR=$OUTDIR"
echo "  ./run_img_evt.sh data           # list events"
echo "  ./run_img_evt.sh data <idx>     # imaging"
echo "  ./run_ql_evt.sh  data <idx>     # clustering + Q/L matching"
