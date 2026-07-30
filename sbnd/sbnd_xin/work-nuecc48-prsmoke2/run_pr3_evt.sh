#!/bin/bash
# doc pr/3 validation runner: production tail + neutrino PR + BDT scorers +
# Magnify-tracking + T_tagger/T_kine on one nueCC48 event.
# Usage: ./run_pr3_evt.sh <EVT>
set -u
EVT=${1:?usage: run_pr3_evt.sh <EVT>}
SX=$(cd "$(dirname "$0")/.." && pwd)          # sbnd_xin
OUT=$(cd "$(dirname "$0")" && pwd)/nupr_evt$EVT
mkdir -p "$OUT"
TK=/nfs/data/1/xqian/toolkit-dev/toolkit
export WIRECELL_PATH=$TK/cfg:/nfs/data/1/xqian/toolkit-dev/wire-cell-data:/nfs/data/1/xqian/toolkit-dev/wire-cell-data/sbnd/photodet
setarch x86_64 -R /nfs/data/1/xqian/toolkit-dev/local/bin/wire-cell \
  -l stderr -l "$OUT/wct_nupr_evt$EVT.log:debug" -L debug \
  --tla-str "input=$SX/work-nuecc48-nuf/ql_evt$EVT/pctree-evt$EVT.tar.gz" \
  --tla-code 'anode_indices=[0,1]' --tla-str "output_dir=$OUT" \
  --tla-code run=18253 --tla-code subrun=1 --tla-code event=$EVT \
  --tla-str reality=data --tla-code DL=4.0 --tla-code DT=8.8 \
  --tla-code lifetime=35 --tla-code driftSpeed=1.563 \
  --tla-code "pipeline_names=['switch_scope','unmerge_bundle','unmerge_assoc','steiner','fiducialutils','tagger_check_tgm','tagger_check_stm','tagger_check_fc','tagger_check_neutrino','numu_bdt_scorer','nue_bdt_scorer','tracking_visitor','tagger_output']" \
  --tla-str "trackfitting_config=$SX/sbnd_track_fitting.json" \
  --tla-str "save_tensors=$OUT/pctree-pr-evt$EVT.tar.gz" \
  --tla-str "dl_weights=" --tla-code 'beam_window_us=[0.2,2.2]' \
  -c $TK/cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet \
  > "$OUT/stdout.log" 2>&1
echo "rc=$?" | tee "$OUT/rc.txt"
