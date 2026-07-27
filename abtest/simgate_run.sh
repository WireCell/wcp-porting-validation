#!/bin/bash
# Master-merge validation: gen/ (simulation) arm for PDHD + PDVD.
#
# Replicates pd{hd,vd}_sim/run_sim_{track,noise}.sh's wire-cell invocation but
# redirects output_prefix into scratch, so the existing 728 MB of sim output
# under pd*_sim/work/ is never overwritten (M13).
#
# Usage: ./simgate_run.sh <arm>       e.g. ./simgate_run.sh pre
set -u
ARM=${1:?usage: simgate_run.sh <arm>}
W=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
WCT=/nfs/data/1/xqian/toolkit-dev
DEST=/home/xqian/tmp/mergegate/sim-$ARM
mkdir -p "$DEST"

# anode/plane combos: one per detector is enough to catch a gen/ change; the
# track sim exercises drift+transform+digitize, the noise sim exercises the
# noise model + digitizer with no signal.
run_hd() {
    local d=$W/pdhd_sim
    export WIRECELL_PATH="$d:$WCT/toolkit/cfg:$WCT/wire-cell-data"
    cd "$d" || return 1
    setarch x86_64 -R wire-cell -l stderr -l "$DEST/hd-track.log:debug" -L debug \
        -V "elecGain=14" \
        --tla-code "tracks_json=$(cat "$d/tracks/tracks-hd-anode0-U.json")" \
        --tla-str "output_prefix=$DEST/hd-track" \
        --tla-code "anode_indices=[0]" \
        -c wct-sim-check-track.jsonnet > "$DEST/hd-track.out" 2>&1
    echo "  hd-track rc=$?"
    setarch x86_64 -R wire-cell -l stderr -l "$DEST/hd-noise.log:debug" -L debug \
        -V "elecGain=14" \
        --tla-str "output_prefix=$DEST/hd-noise" \
        --tla-code "anode_indices=[0,1,2,3]" \
        -c wct-sim-noise-only.jsonnet > "$DEST/hd-noise.out" 2>&1
    echo "  hd-noise rc=$?"
}
run_vd() {
    local d=$W/pdvd_sim
    export WIRECELL_PATH="$d:$WCT/toolkit/cfg:$WCT/wire-cell-data"
    cd "$d" || return 1
    setarch x86_64 -R wire-cell -l stderr -l "$DEST/vd-track.log:debug" -L debug \
        --tla-code "tracks_json=$(cat "$d/tracks/tracks-vd-anode0-U.json")" \
        --tla-str "output_prefix=$DEST/vd-track" \
        --tla-code "anode_indices=[0]" \
        -c wct-sim-check-track.jsonnet > "$DEST/vd-track.out" 2>&1
    echo "  vd-track rc=$?"
    setarch x86_64 -R wire-cell -l stderr -l "$DEST/vd-noise.log:debug" -L debug \
        --tla-str "output_prefix=$DEST/vd-noise" \
        --tla-code "anode_indices=[0,1]" \
        -c wct-sim-noise-only.jsonnet > "$DEST/vd-noise.out" 2>&1
    echo "  vd-noise rc=$?"
}

echo "== PDHD sim =="; run_hd
echo "== PDVD sim =="; run_vd
echo "=== sim arm '$ARM' -> $DEST ==="
ls -la "$DEST"/*.tar.bz2 2>/dev/null | awk '{print "  ", $5, $9}'
