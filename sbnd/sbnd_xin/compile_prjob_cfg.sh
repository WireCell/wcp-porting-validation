#!/bin/bash
# usage: compile_prjob_f3c.sh <cfgroot> <outjson>
set -u
CFG=$1; OUT=$2
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
export WIRECELL_PATH=$CFG:/nfs/data/1/xqian/toolkit-dev/wire-cell-data:/nfs/data/1/xqian/toolkit-dev/wire-cell-data/sbnd/photodet
/nfs/data/1/xqian/toolkit-dev/local/bin/wcsonnet \
  -A "input=in.tar.gz" \
  --tla-code 'anode_indices=[0,1]' \
  -A "output_dir=out" \
  --tla-code run=18253 --tla-code subrun=1 --tla-code event=172230 \
  -A reality=data --tla-code DL=4.0 --tla-code DT=8.8 \
  --tla-code lifetime=35 --tla-code driftSpeed=1.563 \
  --tla-code "pipeline_names=['switch_scope','unmerge_bundle','unmerge_assoc','steiner','fiducialutils','tagger_check_tgm','tagger_check_stm','tagger_check_fc','tagger_check_neutrino','numu_bdt_scorer','nue_bdt_scorer','tracking_visitor','tagger_output']" \
  -A "trackfitting_config=$SX/sbnd_track_fitting.json" \
  -A "save_tensors=out.tar.gz" \
  -A "dl_weights=" --tla-code 'beam_window_us=[0.2,2.2]' \
  $CFG/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet > $OUT
