#!/bin/bash
# Compile every live SBND/PDHD/PDVD job config with the exact TLA set its runner
# uses, into $1.  Run identically before and after the master merge; compare
# with cmp_cfg.sh (which normalizes away master's new `_pnode` key).
#
# Usage: ./compile_all_cfg.sh <outdir>
set -u
OUT=${1:?usage: compile_all_cfg.sh <outdir>}
mkdir -p "$OUT"

WCT=/nfs/data/1/xqian/toolkit-dev
DATA=$WCT/wire-cell-data
# CFGROOT lets us compile against an alternate cfg/ tree (e.g. one extracted
# from the merged git tree) without touching the repo.
CFG=${CFGROOT:-$WCT/toolkit/cfg}
W=$WCT/wcp-porting-img
PD=$W/pdhd
VD=$W/pdvd
SB=$W/sbnd/sbnd_xin
PDS=$W/pdhd_sim
VDS=$W/pdvd_sim

fails=0
# one() <tag> <extra WIRECELL_PATH prefix or -> <wcsonnet args...>
one() {
    local tag=$1 pre=$2; shift 2
    local wp="$CFG:$DATA"
    [ "$pre" != "-" ] && wp="$pre:$CFG:$DATA"
    if WIRECELL_PATH="$wp" wcsonnet "$@" -o "$OUT/$tag.json" \
            > "$OUT/$tag.log" 2>&1; then
        printf '  %-16s ok  %s elements\n' "$tag" "$(jq 'length' "$OUT/$tag.json" 2>/dev/null)"
    else
        printf '  %-16s FAILED (see %s.log)\n' "$tag" "$tag"; fails=$((fails+1))
    fi
}

echo "== SBND =="
cd "$SB"
one sbnd_pr - \
  -A "input=/x/pctree-evt287517.tar.gz" -S "anode_indices=[0,1]" -A "output_dir=/x" \
  -S "run=18255" -S "subrun=1" -S "event=287517" -A "reality=data" \
  -S "DL=6.5781" -S "DT=13.1349" -S "lifetime=35" -S "driftSpeed=1.563" \
  -S "pipeline_names=['switch_scope','unmerge_bundle','unmerge_assoc','steiner','fiducialutils','tagger_check_tgm','tagger_check_stm','tagger_check_fc','stm_magnify']" \
  -A "trackfitting_config=$SB/sbnd_track_fitting.json" \
  -A "save_tensors=/x/pctree-pr.tar.gz" -A "dl_weights=" \
  -S "beam_window_us=[0.2,2.2]" -S "beam_window_only=true" \
  -S "tgm_neutrino_candidate=true" -S "tgm_chord_charge=true" -A "tgm_chord_mode=path" \
  -S "tgm_component_extremes=true" -S "tgm_component_rescue=true" -S "tgm_rescue_chord=true" \
  -S "tgm_main_pair=true" -A "tgm_main_pair_mode=real" \
  -S "tgm_fv_zmax_margin=5" -S "tgm_fv_zmax_margin_interior=3" \
  -S "tgm_fv_x_margin=2.5" -S "tgm_fv_y_margin=3" \
  -S "mip_dqdx=56000" -A "unmerge_bundle_mode=real" \
  -S "save_stm_fit=true" -S "stm_consistent_fv=true" \
  -S "stm_accept_guards=true" -S "stm_proton_muon_guard=true" \
  -S "stm_cathode_guard=true" -S "stm_anode_dist_fix=true" \
  -S "stm_second_track_guard=true" -S "stm_deficit_guard=true" -S "stm_vertex_kink_guard=true" \
  wct-pr-perevt.jsonnet
one sbnd_img - -A "input=/x/sp.tar.gz" -S "anode_indices=[0,1]" -A "output_dir=/x" wct-img-all.jsonnet
one sbnd_clus - -A "input=/x/clusters-apa.tar.gz" -S "anode_indices=[0,1]" -A "output_dir=/x" \
  -S "run=18255" -S "subrun=1" -S "event=287517" wct-clustering.jsonnet
one sbnd_ql - -A "input=/x/clusters-apa.tar.gz" -S "anode_indices=[0,1]" -A "output_dir=/x" \
  -S "run=18255" -S "subrun=1" -S "event=287517" wct-clus-matching-perevt.jsonnet

echo "== PDHD =="
cd "$PD"
one pdhd_nfsp - -V "elecGain=14" -A "orig_prefix=/x/orig" -A "raw_prefix=/x/raw" \
  -A "sp_prefix=/x/sp" -A "reality=data" -S "anode_indices=[0,1,2,3]" wct-nf-sp.jsonnet
one pdhd_img - -A "input_prefix=/x/in" -S "anode_indices=[0]" -A "output_dir=/x" -S "nticks=6000" wct-img-all.jsonnet
one pdhd_clus - -A "input=/x/clusters-apa.tar.gz" -S "anode_indices=[0,1,2,3]" -A "output_dir=/x" \
  -S "run=27305" -S "subrun=0" -S "event=0" -S "do_qlmatch=false" -S "calib=false" \
  -S "save_opflash=false" -S "trigger_offset_us=0" -S "readout_window_ticks=6000" wct-clustering.jsonnet

echo "== PDVD =="
cd "$VD"
one pdvd_nfsp - -A "orig_prefix=/x/orig" -A "raw_prefix=/x/raw" -A "sp_prefix=/x/sp" \
  -A "reality=data" -S "anode_indices=[0,1,2,3,4,5,6,7]" wct-nf-sp.jsonnet
one pdvd_img - -A "input_prefix=/x/in" -S "anode_indices=[0]" -A "output_dir=/x" wct-img-all.jsonnet
one pdvd_clus - -A "input=/x/clusters-apa.tar.gz" -S "anode_indices=[0,1,2,3,4,5,6,7]" -A "output_dir=/x" \
  -S "run=39252" -S "subrun=0" -S "event=5" wct-clustering.jsonnet

echo "== sim-check (in-tree, gen/ coverage) =="
cd "$WCT/toolkit"
one pdhd_simcheck - -V "elecGain=14.0" -V "elecShaping=2.2" -V "elecAdcPerVolt=1" -V "elecBaseline=0" \
  cfg/pgrapher/experiment/pdhd/wct-sim-check.jsonnet
one pdvd_simcheck - cfg/pgrapher/experiment/protodunevd/wct-sim-check.jsonnet
one sbnd_simcheck - cfg/pgrapher/experiment/sbnd/wct-sim-check.jsonnet

echo "== sim-track (pdhd_sim / pdvd_sim runners) =="
cd "$PDS"
one pdhd_simtrack "$PDS" -V "elecGain=14" \
  -S "tracks_json=$(cat "$PDS/tracks/tracks-hd-anode0-U.json")" \
  -A "output_prefix=/x/hd" -S "anode_indices=[0]" wct-sim-check-track.jsonnet
one pdhd_simnoise "$PDS" -V "elecGain=14" -A "output_prefix=/x/hdn" \
  -S "anode_indices=[0,1,2,3]" wct-sim-noise-only.jsonnet
cd "$VDS"
one pdvd_simtrack "$VDS" \
  -S "tracks_json=$(cat "$VDS/tracks/tracks-vd-anode0-U.json")" \
  -A "output_prefix=/x/vd" -S "anode_indices=[0]" wct-sim-check-track.jsonnet

echo "=== $fails failure(s); outputs in $OUT ==="
exit $fails
