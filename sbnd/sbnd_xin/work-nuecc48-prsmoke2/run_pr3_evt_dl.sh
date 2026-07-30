#!/bin/bash
# doc pr/3 diagnostic: same pipeline as run_pr3_evt.sh but with the DL (SCN)
# neutrino-vertex finder ENABLED, using the uBooNE-trained weights.
# The SCN voxelizer subtracts the point-cloud min (SCN_Vertex.py voxelize()),
# so the net is translation-invariant -- applying it to SBND coordinates is
# meaningful, but the training is uBooNE (doc pr/2 gap G3): DIAGNOSTIC ONLY,
# never a gate arm (CLAUDE.md M4: DL vertex is not bit-stable).
# Output goes to a SEPARATE dir (nupr_evt<EVT>_dl) so the geometric-vertex
# results stay untouched (M13).
# Usage: ./run_pr3_evt_dl.sh <EVT>
set -u
EVT=${1:?usage: run_pr3_evt_dl.sh <EVT>}
SX=$(cd "$(dirname "$0")/.." && pwd)          # sbnd_xin
# DL_SUFFIX lets a repeat run land in its own dir (determinism check) without
# touching the first one (M13).
OUT=$(cd "$(dirname "$0")" && pwd)/nupr_evt${EVT}${DL_SUFFIX:-_dl}
mkdir -p "$OUT"
TK=/nfs/data/1/xqian/toolkit-dev/toolkit
export WIRECELL_PATH=$TK/cfg:/nfs/data/1/xqian/toolkit-dev/wire-cell-data:/nfs/data/1/xqian/toolkit-dev/wire-cell-data/sbnd/photodet
export PYTHONPATH=$TK/pyutil/python:/nfs/data/1/xqian/toolkit-dev/local/python:/nfs/data/1/xqian/toolkit-dev/wire-cell-python:${PYTHONPATH:-}
# The embedded interpreter needs libpython loaded RTLD_GLOBAL, else the
# extension modules fail with "undefined symbol: PyTuple_Type" and the DL
# vertex silently falls back to the geometric one (same idiom as run_pr_evt.sh:180).
PYLIB=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3.11.so.1.0
export LD_PRELOAD="$PYLIB"
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
  --tla-str "dl_weights=uboone/scn_vtx/t48k-m16-l5-lr5d-res0.5-CP24.pth" \
  --tla-code 'beam_window_us=[0.2,2.2]' \
  -c $TK/cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet \
  > "$OUT/stdout.log" 2>&1
echo "rc=$?" | tee "$OUT/rc.txt"
