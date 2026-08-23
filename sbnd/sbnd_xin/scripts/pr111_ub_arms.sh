#!/bin/bash
# doc pr/111 sec 12 -- the owner's uBooNE 2x2: toolkit vs prototype, exclusion ON vs OFF,
# BOTH WITH THE DL VERTEX ON.  Forked from pr108_wct.sh + run_wcp.sh so the pr/108 arms and
# the stored prototype references (prototype_base/nue_5384_*.root) stay untouched (M13).
#
#   toolkit   : qlport/scripts/run_one.sh, DL_WEIGHTS set, QL_FIT_EXCLUSION={true,false}
#               NOTE the uBooNE TLA compares == "true", so =1 silently means FALSE.
#   prototype : wire-cell-prod-nue-port in an isolated scratch dir; DL is ON by default
#               (no -l0); WCP_FIT_EXCLUSION=0 turns exclusion off.
#
# usage: pr111_ub_arms.sh wct|wcp   (run the two arms of that code, all 35 events)
set -u
WHICH=${1:?wct|wcp}
JOBS=${PR111_JOBS:-6}
W=uboone/scn_vtx/t48k-m16-l5-lr5d-res0.5-CP24.pth
PB=/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base
QL=/nfs/data/1/xqian/toolkit-dev/toolkit/qlport

if [ "$WHICH" = wct ]; then
  cd /nfs/data/1/xqian/toolkit-dev && exec direnv exec . bash -c '
    cd toolkit/qlport/scripts
    for arm in on off; do
      case $arm in on) E="QL_FIT_EXCLUSION=true";; off) E="QL_FIT_EXCLUSION=false";; esac
      echo "=== $(date +%T) WCT arm=$arm"
      seq 0 34 | xargs -P '"$JOBS"' -I{} env $E DL_WEIGHTS='"$W"' ./run_one.sh {} pr111_wct_${arm}_dl
    done; echo "=== ALL DONE $(date +%T)"'
fi

# ---- prototype: isolated scratch, references never touched
run_wcp_one () {
  local ARM=$1 ev=$2
  local D=/home/xqian/tmp/pr111_wcp/$ARM/$ev; mkdir -p "$D"; cd "$D" || return 1
  ln -sfn $PB/input_data_files input_data_files
  [ "$ARM" = off ] && export WCP_FIT_EXCLUSION=0
  local f; f=$(ls $QL/rootfiles/nuselEval_5384_*_${ev}.root 2>/dev/null | head -1)
  [ -n "$f" ] || { echo "ev=$ev NO INPUT"; return 1; }
  setarch x86_64 -R $PB/build/pid/wire-cell-prod-nue-port \
      ./input_data_files/ChannelWireGeometry_v2.txt "$f" 0 -d0 -o1 -gfind_other_segments \
      > 5384_$ev.log 2>&1
  echo "arm=$ARM ev=$ev rc=$?"
}
export -f run_wcp_one; export PB QL

if [ "$WHICH" = wcp ]; then
  export LD_LIBRARY_PATH="$(ls -d $PB/build/*/ | tr '\n' ':')/nfs/data/1/xqian/prototype-dev/install/lib64:${LD_LIBRARY_PATH:-}"
  EVTS=$(sed -E 's#.*_([0-9]+)\.root#\1#' $QL/filelist)
  for arm in on off; do
    echo "=== $(date +%T) WCP arm=$arm (DL on by default)"
    echo "$EVTS" | xargs -P $JOBS -I{} bash -c "run_wcp_one $arm {}"
  done
  echo "=== ALL DONE $(date +%T)"
fi
