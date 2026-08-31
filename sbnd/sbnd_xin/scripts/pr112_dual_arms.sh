#!/bin/bash
# doc pr/112 sec 11 -- dual-chain arms.  Forked from pr112_arms.sh (same ql
# roots, same harvest-on convention so every arm can be scored on the pr/106
# target metric against the vtxscan-harv3-* hand-scan labels).
#
# Arm names: work-pr112i-<arm>-<sample>  (M13: never reuse a name)
#   off        new binary, every dual-chain key absent  (knob-off gate)
#   nofitx     SBND_FIT_EXCLUSION=false, no dual chain (sec 11.6 ceiling rung 1)
#   probe      dl_vtx_dual_chain=true, transfer unset   (leakage gate, snap mode)
#   probev     same, mode=voxels                        (leakage gate, voxels stop)
#   fid        probe + WCT_DUAL_CHAIN_OFF_EXCL=1        (duplicate-fidelity gate)
#   snapD<x>   snap transfer, D=<x> cm                  e.g. snapD2
#   vox        voxels transfer
#   uniW<w>    union transfer, vtx_weight=<w>, D=2      e.g. uniW0, uniW2
#   *-noswap   suffix: dual_chain_allow_cluster_swap=false
#
# Usage: ./pr112_dual_arms.sh <arm> <sample> [PR_JOBS] [evt ...]
#   sample nuecc48|ncpi0|mcp1k|mcp2k|numu100 (numu100 = mcp1k root, the 100
#   labeled events in scripts/pr112_numu100.txt)
set -eu
ARM=${1:?arm}; SAMP=${2:?sample}; JOBS=${3:-16}; shift 3 || shift $#
cd "$(dirname "$0")/.."
ROOT=$SAMP; EVTS=("$@")
if [ "$SAMP" = numu100 ]; then ROOT=mcp1k; [ ${#EVTS[@]} -eq 0 ] && mapfile -t EVTS < scripts/pr112_numu100.txt; fi
QL="work-${ROOT}-ql0819"; OUT="work-pr112i-${ARM}-${SAMP}"
[ -d "$QL" ] || { echo "no ql root $QL" >&2; exit 1; }
export PR_JOBS="$JOBS" PR_EXTRA_STAGES=pr_display SBND_DL_VTX_HARVEST=true
base=${ARM%-noswap}
[ "$base" != "$ARM" ] && export SBND_DUAL_CHAIN_ALLOW_CLUSTER_SWAP=false
case "$base" in
  off)    ;;
  nofitx) export SBND_FIT_EXCLUSION=false ;;
  probe)  export SBND_DL_VTX_DUAL_CHAIN=true ;;
  probev) export SBND_DL_VTX_DUAL_CHAIN=true SBND_DUAL_CHAIN_MODE=voxels ;;
  fid)    export SBND_DL_VTX_DUAL_CHAIN=true WCT_DUAL_CHAIN_OFF_EXCL=1 ;;
  snapD*) export SBND_DL_VTX_DUAL_CHAIN=true SBND_DUAL_CHAIN_TRANSFER=true SBND_DUAL_CHAIN_TRANSFER_MAX=${base#snapD} ;;
  vox)    export SBND_DL_VTX_DUAL_CHAIN=true SBND_DUAL_CHAIN_TRANSFER=true SBND_DUAL_CHAIN_MODE=voxels ;;
  uniW*)  export SBND_DL_VTX_DUAL_CHAIN=true SBND_DUAL_CHAIN_TRANSFER=true SBND_DUAL_CHAIN_MODE=union SBND_DUAL_CHAIN_VTX_WEIGHT=${base#uniW} SBND_DUAL_CHAIN_TRANSFER_MAX=2.0 ;;
  *) echo "unknown arm $ARM" >&2; exit 1 ;;
esac
echo "=== $OUT  ql=$QL  jobs=$JOBS  evts=${#EVTS[@]}  $(date +%T)"
env | grep -E '^(SBND_DUAL|SBND_DL_VTX_DUAL|WCT_DUAL)' || true
./run_pr_chain_batch.sh "$QL" "$OUT" data "${EVTS[@]}"
echo "=== DONE $OUT $(date +%T)"
