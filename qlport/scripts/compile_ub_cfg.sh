#!/bin/bash
set -u
CFG=$1; OUT=$2
QL=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/qlport
export WIRECELL_PATH=$CFG:/nfs/data/1/xqian/toolkit-dev/wire-cell-data
/nfs/data/1/xqian/toolkit-dev/local/bin/wcsonnet \
  -A kind=both -A "beezip=mabc_0.zip" -A "initial_index=0" \
  -A "initial_runNo=5384" -A "initial_subRunNo=22" -A "initial_eventNo=6805" \
  -A "dl_weights=" -A "dir_weak_use_score=true" \
  -A "infiles=$QL/rootfiles/nuselEval_5384_22_6805.root" \
  $QL/uboone-mabc.jsonnet > $OUT
