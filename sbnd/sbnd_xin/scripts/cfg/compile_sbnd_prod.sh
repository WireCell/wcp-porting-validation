#!/bin/bash
# compiled-config proof for the SBND PRODUCTION entry points, which import
# pgrapher/experiment/sbnd/clus.jsonnet: the LArSoft wcls imaging+clustering
# chain and the standalone clustering+matching job.
set -u
CFG=$1; OUT=$2
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
export WIRECELL_PATH=$CFG:/nfs/data/1/xqian/toolkit-dev/wire-cell-data:/nfs/data/1/xqian/toolkit-dev/wire-cell-data/sbnd/photodet
WCS=/nfs/data/1/xqian/toolkit-dev/local/bin/wcsonnet
$WCS -V reality=data -V DL=4.0 -V DT=8.8 -V lifetime=35 -V driftSpeed=1.563 \
     -V 'input_mask_tags=[]' -V 'output_mask_tags=[]' -V 'recobwire_tags=["gauss"]' \
     -V 'summary_tags=[]' -V 'trace_tags=["gauss"]' \
     $CFG/pgrapher/experiment/sbnd/wcls-img-clus.jsonnet > $OUT.wcls 2> $OUT.wcls.err
echo "wcls rc=$?"
$WCS -V reality=data -V DL=4.0 -V DT=8.8 -V lifetime=35 -V driftSpeed=1.563 \
     -V semimodel_file="" --ext-code joint=false --ext-code pmt_nl=true \
     -V input=in.tar.gz -V 'frames=[]' -V output_dir=out \
     $SX/wct-clus-matching-standalone.jsonnet > $OUT.standalone 2> $OUT.standalone.err
echo "standalone rc=$?"
