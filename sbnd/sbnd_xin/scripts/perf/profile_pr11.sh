#!/bin/bash
# doc pr/11: gperftools CPU/heap-profile of the FULL 13-stage PR chain (the
# neutrino-PR + BDT + tracking/tagger_output tail, DL vertex ON) on one event.
# Fork of profile_pr65.sh (which profiles only the TGM/STM/FC chain, geometric
# vertex): same wcsonnet-precompile discipline (M17 -- SIGPROF corrupts
# gojsonnet's GC) and same "never under setarch -R" rule (the profiler dies at
# startup under ASLR-off), extended with:
#   - the 5 neutrino-PR stages, DL vertex on (no dl_weights override) --
#     matches run_pr_chain_batch.sh's production knob set exactly.
#   - LD_PRELOAD carries BOTH libpython (RTLD_GLOBAL, for the SCN import) and
#     libtcmalloc_and_profiler, colon-joined. If that combination is unstable,
#     rerun with DL_WEIGHTS= (geometric only) and report the profile as
#     geometric-arm, noting the DL cost separately from wall/core (already
#     measured, unprofiled, in the main run) -- doc pr/11 sec 5.
#
# Usage:
#   EVT=<id> ROOT=<ql_root dir, relative to sbnd_xin> OUTDIR=<scratch dir> \
#     ./profile_pr11.sh [out.prof]
#   HEAPOUT=<path> ... ./profile_pr11.sh     # tcmalloc heap profile instead
# Env: PROFLIB (default libtcmalloc_and_profiler.so.4), DL_WEIGHTS (default
#      the SCN production weights; pass empty string for the geometric arm).
set -e
SBND=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
EVT=${EVT:?set EVT}
ROOT=${ROOT:?set ROOT (ql_root under sbnd_xin, e.g. work-mcp1kall-d59k)}
OD=${OUTDIR:?set OUTDIR}
QLDIR="$SBND/$ROOT/ql_evt$EVT"
PCT="$QLDIR/pctree-evt$EVT.tar.gz"
[ -s "$PCT" ] || { echo "ERROR: no pctree: $PCT" >&2; exit 1; }
mkdir -p "$OD"; cd "$OD"

# RSE from the Q/L job's own opflash metadata (same source as
# run_pr_chain_batch.sh -- correct across every sample, not hardcoded).
RUN_NO=0; SUBRUN_NO=0
_md=$(tar xzOf "$QLDIR/opflash_apa0.tar.gz" "opflash_tensorset_${EVT}_metadata.json" 2>/dev/null) || _md=''
if [ -n "$_md" ]; then
    _rse=$(printf '%s' "$_md" | python3 -c \
        'import json,sys; d=json.load(sys.stdin); print(int(d.get("run",0)), int(d.get("subrun",0)))' \
        2>/dev/null) && [ -n "$_rse" ] && read -r RUN_NO SUBRUN_NO <<< "$_rse"
fi
REALITY=${REALITY:-data}
DL_WEIGHTS_TLA=${DL_WEIGHTS-uboone/scn_vtx/t48k-m16-l5-lr5d-res0.5-CP24.pth}  # unset var => default; empty string => geometric

CFG=$OD/pr11_evt$EVT.json
wcsonnet \
  -A "input=$PCT" -S "anode_indices=[0,1]" -A "output_dir=$OD" \
  -S "run=$RUN_NO" -S "subrun=$SUBRUN_NO" -S "event=$EVT" -A "reality=$REALITY" \
  -S "DL=4.0" -S "DT=8.8" -S "lifetime=35" -S "driftSpeed=1.563" \
  -S "pipeline_names=['switch_scope','unmerge_bundle','unmerge_assoc','steiner','fiducialutils','tagger_check_tgm','tagger_check_stm','tagger_check_fc','tagger_check_neutrino','numu_bdt_scorer','nue_bdt_scorer','tracking_visitor','tagger_output']" \
  -A "trackfitting_config=$SBND/sbnd_track_fitting.json" \
  -A "save_tensors=$OD/pctree-pr-evt$EVT.tar.gz" \
  -A "dl_weights=$DL_WEIGHTS_TLA" \
  -S "beam_window_us=[0.2,2.2]" -S "beam_window_only=true" \
  -S "tgm_neutrino_candidate=true" \
  -S "tgm_chord_charge=true" -A "tgm_chord_mode=path" \
  -S "tgm_component_extremes=true" -S "tgm_component_rescue=true" \
  -S "tgm_rescue_chord=true" \
  -S "tgm_main_pair=true" -A "tgm_main_pair_mode=real" \
  -S "tgm_fv_zmax_margin=5" -S "tgm_fv_zmax_margin_interior=3" \
  -S "tgm_fv_x_margin=2.5" -S "tgm_fv_y_margin=3" \
  -S "mip_dqdx=56000" -A "unmerge_bundle_mode=real" \
  -S "save_stm_fit=false" -S "stm_consistent_fv=true" \
  -S "stm_accept_guards=true" -S "stm_proton_muon_guard=true" \
  -S "stm_cathode_guard=true" -S "stm_anode_dist_fix=true" \
  -S "stm_second_track_guard=true" -S "stm_deficit_guard=true" \
  -S "stm_vertex_kink_guard=true" -S "stm_d66_cuts=true" \
  -o "$CFG" "$SBND/wct-pr-perevt.jsonnet"

PROFLIB=${PROFLIB:-/usr/lib/x86_64-linux-gnu/libtcmalloc_and_profiler.so.4}
PRELOAD="$PROFLIB"
if [ -n "$DL_WEIGHTS_TLA" ]; then
    PYLIB=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3.11.so.1.0
    PRELOAD="$PYLIB:$PROFLIB"
fi

if [ -n "${HEAPOUT:-}" ]; then
    LD_PRELOAD="$PRELOAD" HEAPPROFILE="$HEAPOUT" \
    wire-cell -l stderr -L info -c "$CFG"
    echo "heap profile -> ${HEAPOUT}.<N>.heap"
    echo "view: google-pprof --text --inuse_space \$(which wire-cell) ${HEAPOUT}.<N>.heap | head -40"
else
    OUT=${1:-$OD/pr11_evt$EVT.prof}
    LD_PRELOAD="$PRELOAD" CPUPROFILE="$OUT" CPUPROFILE_FREQUENCY="${FREQ:-250}" \
    wire-cell -l stderr -L info -c "$CFG"
    echo "profile -> $OUT"
    echo "view: google-pprof --text --cum \$(which wire-cell) $OUT | head -60"
fi
