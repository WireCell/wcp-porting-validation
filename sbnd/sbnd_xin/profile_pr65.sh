#!/bin/bash
# Doc 65: gperftools CPU-profile of the PR tagger job on evt 287517 (the
# heaviest STM event of the doc-54 30-event scan), current binary, production
# flag set (doc-63 guards ON).  Mirrors run_nusel_evt.sh's wire-cell call;
# config pre-compiled with wcsonnet (M17 -- SIGPROF kills gojsonnet GC).
# NB: must NOT run under `setarch -R` (profiler dies at startup); outputs to
# scratch, never into a work-* label (M13).
set -e
SBND=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
EVT=${EVT:-287517}; RUN_NO=18255; SUBRUN_NO=1
ROOT=${ROOT:-work-mcp1000-p65fin}
PCT=$SBND/$ROOT/ql_evt$EVT/pctree-evt$EVT.tar.gz
OD=${OUTDIR:?set OUTDIR}
mkdir -p "$OD"; cd "$OD"

CFG=$OD/pr65_evt$EVT.json
wcsonnet \
  -A "input=$PCT" -S "anode_indices=[0,1]" -A "output_dir=$OD" \
  -S "run=$RUN_NO" -S "subrun=$SUBRUN_NO" -S "event=$EVT" -A "reality=data" \
  -S "DL=6.5781" -S "DT=13.1349" -S "lifetime=35" -S "driftSpeed=1.563" \
  -S "pipeline_names=['switch_scope','unmerge_bundle','unmerge_assoc','steiner','fiducialutils','tagger_check_tgm','tagger_check_stm','tagger_check_fc','stm_magnify']" \
  -A "trackfitting_config=$SBND/sbnd_track_fitting.json" \
  -A "save_tensors=$OD/pctree-pr-evt$EVT.tar.gz" \
  -A "dl_weights=" \
  -S "beam_window_us=[0.2,2.2]" -S "beam_window_only=true" \
  -S "tgm_neutrino_candidate=true" \
  -S "tgm_chord_charge=true" -A "tgm_chord_mode=path" \
  -S "tgm_component_extremes=true" -S "tgm_component_rescue=true" \
  -S "tgm_rescue_chord=true" \
  -S "tgm_main_pair=true" -A "tgm_main_pair_mode=real" \
  -S "tgm_fv_zmax_margin=5" -S "tgm_fv_zmax_margin_interior=3" \
  -S "tgm_fv_x_margin=2.5" -S "tgm_fv_y_margin=3" \
  -S "mip_dqdx=56000" -A "unmerge_bundle_mode=real" \
  -S "save_stm_fit=true" -S "stm_consistent_fv=true" \
  -S "stm_accept_guards=true" -S "stm_proton_muon_guard=true" \
  -S "stm_cathode_guard=true" -S "stm_anode_dist_fix=true" \
  -S "stm_second_track_guard=true" -S "stm_deficit_guard=true" \
  -S "stm_vertex_kink_guard=true" \
  -o "$CFG" "$SBND/wct-pr-perevt.jsonnet"

OUT=${1:-$OD/pr65_evt$EVT.prof}
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_and_profiler.so.4 \
CPUPROFILE="$OUT" CPUPROFILE_FREQUENCY=250 \
wire-cell -l stderr -L info -c "$CFG"
echo "profile -> $OUT"
