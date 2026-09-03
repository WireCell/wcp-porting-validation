#!/bin/bash
# doc pdvd/28 §7: re-image AND re-sample one PDVD event against an alternate
# wires-geometry file, WITHOUT touching any tracked config. A partial swap
# (only imaging or only sampling under the new geometry) is geometrically
# inconsistent -- see §7's discarded first attempt -- so this always redoes
# both stages.
#
# Method: make a scratch copy of toolkit/cfg (this run's snapshot only, not
# reused across runs -- cheap, ~8 MB), patch protodunevd/params.jsonnet's
# `wires:` line to the requested file, then compile+run imaging (per anode,
# DNN-ROI SP frames reused from the existing production work dir) and
# clustering (-calib -save-pctree) against that scratch cfg.
#
# Usage:
#   cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
#   ./docs/nf_sp_img_clus/scripts/rerun_with_wires_geometry.sh \
#       <wires-file-basename> <tag> <run> <evt>
# e.g.
#   ./docs/nf_sp_img_clus/scripts/rerun_with_wires_geometry.sh \
#       protodunevd-wires-larsoft-v7-uvwfit.json.bz2 v7img 039252 2
#
# Requires: work/<run6>_<evt>_keep/protodune-sp-dnnroi-frames-anode{0..7}.tar.bz2
# already staged (the standard production DNN-ROI SP output for that event).
# Output: work/<run6>_<evt>_<tag>/pctree-evt<ID>.tar.gz (+ the usual clustering
# outputs), built entirely from the alternate geometry.
set -e

WIRES_FILE=$1
TAG=$2
RUN=$3
EVT=$4
[ -n "$EVT" ] || { echo "usage: $0 <wires-file-basename> <tag> <run> <evt>" >&2; exit 1; }

WCT_BASE=/nfs/data/1/xqian/toolkit-dev
PDVD_DIR=$(cd "$(dirname "$0")/../../.." && pwd)
RUN_STRIPPED=$(echo "$RUN" | sed 's/^0*//'); [ -z "$RUN_STRIPPED" ] && RUN_STRIPPED=0
RUN_PADDED=$(printf '%06d' "$RUN_STRIPPED")

[ -s "$WCT_BASE/wire-cell-data/$WIRES_FILE" ] || { echo "ERROR: $WIRES_FILE not in $WCT_BASE/wire-cell-data/" >&2; exit 1; }
SRC_KEEP="$PDVD_DIR/work/${RUN_PADDED}_${EVT}_keep"
ls "$SRC_KEEP"/protodune-sp-dnnroi-frames-anode*.tar.bz2 >/dev/null 2>&1 || {
    echo "ERROR: no protodune-sp-dnnroi-frames-anode*.tar.bz2 in $SRC_KEEP" >&2; exit 1; }

SCRATCH=$(mktemp -d /home/xqian/tmp/wires_geom_test.XXXXXX)
cp -a "$WCT_BASE/toolkit/cfg" "$SCRATCH/cfg"
sed -i "s/protodunevd-wires-larsoft-v6\.json\.bz2/$WIRES_FILE/" \
    "$SCRATCH/cfg/pgrapher/experiment/protodunevd/params.jsonnet"
grep -q "wires: \"$WIRES_FILE\"" "$SCRATCH/cfg/pgrapher/experiment/protodunevd/params.jsonnet" \
    || { echo "ERROR: patch did not take" >&2; exit 1; }
export WIRECELL_PATH="$SCRATCH/cfg:$WCT_BASE/wire-cell-data"

WORKDIR="$PDVD_DIR/work/${RUN_PADDED}_${EVT}_${TAG}"
mkdir -p "$WORKDIR"
cd "$PDVD_DIR"

echo "[1/2] imaging (8 anodes, DNN-ROI frames from $SRC_KEEP) -> $WORKDIR"
for ai in 0 1 2 3 4 5 6 7; do
    wcsonnet -A "input_prefix=$SRC_KEEP/protodune-sp-dnnroi-frames" \
        -S "anode_indices=[$ai]" -A "output_dir=$WORKDIR" \
        -o "$WORKDIR/.wct-img-a${ai}.json" wct-img-all.jsonnet &
done
wait
for ai in 0 1 2 3 4 5 6 7; do
    [ -s "$WORKDIR/.wct-img-a${ai}.json" ] || { echo "ERROR: imaging compile failed for anode$ai" >&2; exit 1; }
done
for ai in 0 1 2 3 4 5 6 7; do
    env GOGC=off wire-cell -l stderr -l "$WORKDIR/wct_img_a${ai}.log:debug" -L debug \
        -c "$WORKDIR/.wct-img-a${ai}.json"
    rm -f "$WORKDIR/.wct-img-a${ai}.json"
done

echo "[2/2] clustering (-calib -save-pctree) -> $WORKDIR"
PDVD_LIGHT_SUFFIX=_keep PDVD_CLUS_COMPILE_ONLY=1 PDVD_KEEP_CFG=1 \
    ./run_clus_evt.sh -s "$TAG" -calib -save-pctree "$RUN" "$EVT"
sed -i "s/protodunevd-wires-larsoft-v6\.json\.bz2/$WIRES_FILE/" "$WORKDIR/.wct-clus.json"
env GOGC=off wire-cell -l stderr -l "$WORKDIR/wct_clus_${RUN_PADDED}_${EVT}.log:debug" -L debug \
    -c "$WORKDIR/.wct-clus.json"

rm -rf "$SCRATCH"
echo "done -> $WORKDIR/pctree-evt*.tar.gz"
