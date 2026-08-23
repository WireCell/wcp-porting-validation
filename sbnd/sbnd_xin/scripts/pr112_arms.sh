#!/bin/bash
# doc pr/112 -- regenerate the DL-vertex study arms deleted by the 2026-08-23
# retire campaign (memory: "sbnd_xin retire 2026-08-23 MINIMAL - 203G->57G,
# 380 arms").  pr/106 sec 9/10 and pr/111 were measured on work-vtx105-* /
# work-vtx106-*, of which only vtx105-base and four vtx106 nueCC48 arms
# survive; mcp1k/mcp2k harvest arms and EVERY DL-off (trad) arm are gone.
# Owner authorised the reproduction (2026-08-23).
#
# NEW arm names (M13 -- never write into a retired label's name).  Arms:
#   harv    exclusion ON  + dl_vtx_harvest      == old work-vtx106-harv-base
#   trad    exclusion ON  + DL disabled         == old work-vtx105-trad
#   cne     harv + dl_vtx_cloud_no_exclusion    == old work-vtx106-cne-on
#   nofitx  fit_exclusion=false + harvest       == old work-vtx106-harv-nofitx
# The `harv` arm doubles as the REPRODUCTION GATE: its scoreboards must match
# the surviving work-vtx106-harv-base-nuecc48 (pr112_repro_gate.py).
#
# Usage: ./pr112_arms.sh <arm> <sample> [PR_JOBS]
set -eu
ARM=${1:?arm: harv|trad|cne|nofitx}
SAMP=${2:?sample: nuecc48|ncpi0|mcp1k|mcp2k}
JOBS=${3:-16}
cd "$(dirname "$0")/.."
QL="work-${SAMP}-ql0819"
OUT="work-pr112-${ARM}-${SAMP}"
[ -d "$QL" ] || { echo "no ql root $QL" >&2; exit 1; }
export PR_JOBS="$JOBS"
# calib-pr-evt<ID>.json (the scoreboard + hv_cloud payload every pr112 script
# reads) is written by the PrDisplayDump stage; without it the arm completes
# rc=0 and is silently useless.
export PR_EXTRA_STAGES=pr_display
case "$ARM" in
  harv)   export SBND_DL_VTX_HARVEST=true ;;
  cne)    export SBND_DL_VTX_HARVEST=true SBND_DL_VTX_CLOUD_NO_EXCLUSION=true ;;
  nofitx) export SBND_DL_VTX_HARVEST=true SBND_FIT_EXCLUSION=false ;;
  trad)   export SBND_VERTEX_SCOREBOARD=true SBND_DL_WEIGHTS='' ;;
  *) echo "unknown arm $ARM" >&2; exit 1 ;;
esac
echo "=== $OUT  ql=$QL  jobs=$JOBS  $(date +%T)"
./run_pr_chain_batch.sh "$QL" "$OUT" data
echo "=== DONE $OUT $(date +%T)"
