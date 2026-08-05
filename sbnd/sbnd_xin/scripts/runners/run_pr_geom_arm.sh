#!/bin/bash
# doc pr/2 sec 2e(iv) A/B: same binary, two arms of the detector-extent knobs.
#   ARM=on  -> SBND defaults (cosmic_y 183/185/163/133 cm, vertex_z_prior 100 cm)
#   ARM=off -> uBooNE literals via TLAs (pr_y_top=117, vertex_z_prior_scale=200)
# Geometric vertex only (dl_weights=""): the DL vertex is not bit-stable (M4).
# Usage: PROUT=<dir> ARM=on|off ./run_pr_geom_arm.sh <EVT>
set -u
EVT=${1:?usage: run_pr_geom_arm.sh <EVT>}
ARM=${ARM:?ARM=on|off required}
OUT=${PROUT:?PROUT required}
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
TK=/nfs/data/1/xqian/toolkit-dev/toolkit
mkdir -p "$OUT"
export WIRECELL_PATH=$TK/cfg:/nfs/data/1/xqian/toolkit-dev/wire-cell-data:/nfs/data/1/xqian/toolkit-dev/wire-cell-data/sbnd/photodet
export PYTHONPATH=$TK/pyutil/python:/nfs/data/1/xqian/toolkit-dev/local/python:/nfs/data/1/xqian/toolkit-dev/wire-cell-python:${PYTHONPATH:-}
GEOM=()
if [ "$ARM" = off ]; then
  GEOM=(--tla-code pr_y_top=117 --tla-code vertex_z_prior_scale=200)
fi
setarch x86_64 -R /nfs/data/1/xqian/toolkit-dev/local/bin/wire-cell \
  -l stderr -l "$OUT/wct_nupr_evt$EVT.log:trace" -L trace \
  --tla-str "input=$SX/work-nuecc48-cb0805/ql_evt$EVT/pctree-evt$EVT.tar.gz" \
  --tla-code 'anode_indices=[0,1]' --tla-str "output_dir=$OUT" \
  --tla-code run=18253 --tla-code subrun=1 --tla-code event=$EVT \
  --tla-str reality=data --tla-code DL=4.0 --tla-code DT=8.8 \
  --tla-code lifetime=35 --tla-code driftSpeed=1.563 \
  --tla-code "pipeline_names=['switch_scope','unmerge_bundle','unmerge_assoc','steiner','fiducialutils','tagger_check_tgm','tagger_check_stm','tagger_check_fc','tagger_check_neutrino','numu_bdt_scorer','nue_bdt_scorer','tracking_visitor','tagger_output']" \
  --tla-str "trackfitting_config=$SX/sbnd_track_fitting.json" \
  --tla-str "dl_weights=" \
  --tla-code 'beam_window_us=[0.2,2.2]' \
  "${GEOM[@]}" \
  -c $TK/cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet \
  > "$OUT/stdout.log" 2>&1
echo "rc=$?" | tee "$OUT/rc.txt"
