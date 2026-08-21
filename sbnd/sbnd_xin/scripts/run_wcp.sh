#!/bin/bash
# doc pr/108 Test B: prototype arms in a scratch dir (the stored references in prototype_base/nue_5384_*.root are never touched).
# usage: run_wcp.sh <arm: on|off> <evt...>    (WCP_FIT_EXCLUSION=0 for off)
set -u
ARM=$1; shift
PB=/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base
D=/home/xqian/tmp/pr108_wcp/$ARM; mkdir -p $D; cd $D
ln -sfn $PB/input_data_files input_data_files
export LD_LIBRARY_PATH=/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/build/2dtoy/:/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/build/3dst/:/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/build/c4che/:/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/build/data/:/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/build/dune_app/:/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/build/examples/:/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/build/graph/:/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/build/icarus_app/:/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/build/lsp/:/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/build/matrix/:/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/build/mcs/:/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/build/nanoflann/:/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/build/nav/:/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/build/paal/:/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/build/pid/:/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/build/pyutil/:/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/build/quickhull/:/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/build/ress/:/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/build/rootvis/:/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/build/signal/:/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/build/sst/:/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/build/tiling/:/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/build/uboone_bdt_app/:/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/build/uboone_eval_app/:/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/build/uboone_light_app/:/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/build/uboone_nusel_app/:/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/build/uboone_sp_app/:/nfs/data/1/xqian/prototype-dev/install/lib64:${LD_LIBRARY_PATH:-}
BIN=/nfs/data/1/xqian/toolkit-dev/toolkit/prototype_base/build/pid/wire-cell-prod-nue-port
[ "$ARM" = off ] && export WCP_FIT_EXCLUSION=0
for ev in "$@"; do
  f=$(ls /nfs/data/1/xqian/toolkit-dev/toolkit/qlport/rootfiles/nuselEval_5384_*_${ev}.root | head -1)
  echo "=== $(date +%T) arm=$ARM ev=$ev $f"
  setarch x86_64 -R $BIN ./input_data_files/ChannelWireGeometry_v2.txt $f 0 -d0 -o1 -gfind_other_segments > 5384_$ev.log 2>&1; echo "rc=$? $(ls nue_5384_*_$ev.root 2>/dev/null)"
done
