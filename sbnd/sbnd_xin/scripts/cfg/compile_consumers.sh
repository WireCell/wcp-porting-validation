#!/bin/bash
# usage: compile_consumers.sh <cfgroot> <outdir>
#
# doc 77 round 2.  Compile EVERY live consumer of the SBND/common clustering
# config against <cfgroot>.  Run it once against a pristine tree
# (git archive HEAD cfg | tar -x -C <dir>) and once against the working tree,
# then cmp_consumers.sh the two output dirs: a cfg refactor is only "no
# behavior change" if all of them are byte-identical.
set -u
CFG=${1:?}; OUT=${2:?}
mkdir -p "$OUT"
AB=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/abtest
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
QL=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/qlport
DATA=/nfs/data/1/xqian/toolkit-dev/wire-cell-data
W=/nfs/data/1/xqian/toolkit-dev/local/bin/wcsonnet

# (a) the broad harness: sbnd pr/img/clus/ql + pdhd + pdvd + sim checks
CFGROOT=$CFG $AB/compile_all_cfg.sh "$OUT" > "$OUT/_compile_all.log" 2>&1
echo "compile_all rc=$?"

# (b) SBND PR job at the PRODUCTION operating point (full PR pipeline + BDTs)
$SX/scripts/cfg/compile_prjob_cfg.sh "$CFG" "$OUT/prod_prjob.json" 2> "$OUT/prod_prjob.err"
echo "prod_prjob rc=$?"

# (c) SBND wcls imaging+clustering and the legacy standalone Q/L job
$SX/scripts/cfg/compile_sbnd_prod.sh "$CFG" "$OUT/prod" > "$OUT/_prod.log" 2>&1
echo "prod_sbnd rc=$?"

# (d) uBooNE MABC (the other caller of common/clus.jsonnet tagger_check_neutrino)
$QL/scripts/compile_ub_cfg.sh "$CFG" "$OUT/uboone.json" 2> "$OUT/uboone.err"
echo "uboone rc=$?"

# (e) SBND PR job bare (default pipeline, default operating point)
export WIRECELL_PATH=$CFG:$DATA:$DATA/sbnd/photodet
$W -A input=in.tar.gz -A output_dir=out -S run=1 -S subrun=1 -S event=1 -A reality=data \
   "$CFG/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet" > "$OUT/bare_prjob.json" 2> "$OUT/bare_prjob.err"
echo "bare_prjob rc=$?"
