#!/bin/bash
# usage: compile_prjob_cfg.sh <cfgroot> <outjson>
#
# Compiled-config proof for the SBND PR job at the PRODUCTION operating point.
# Since doc 68 that operating point IS the job's own TLA defaults, so this
# compiles almost bare: only what is per-event, plus the PR-chain pipeline
# (which adds the neutrino taggers + BDT scorers on top of the default list)
# and the dl_weights / save_tensors choices run_pr_chain_batch.sh makes.
set -u
CFG=$1; OUT=$2
export WIRECELL_PATH=$CFG:/nfs/data/1/xqian/toolkit-dev/wire-cell-data:/nfs/data/1/xqian/toolkit-dev/wire-cell-data/sbnd/photodet
/nfs/data/1/xqian/toolkit-dev/local/bin/wcsonnet \
  -A "input=in.tar.gz" \
  -A "output_dir=out" \
  --tla-code run=18253 --tla-code subrun=1 --tla-code event=172230 \
  -A reality=data \
  --tla-code "pipeline_names=['switch_scope','unmerge_bundle','unmerge_assoc','steiner','fiducialutils','tagger_check_tgm','tagger_check_stm','tagger_check_fc','protect_bundle','steiner_refresh','tagger_check_neutrino','numu_bdt_scorer','nue_bdt_scorer','tracking_visitor','tagger_output']" \
  -A "save_tensors=out.tar.gz" \
  -A "dl_weights=" \
  $CFG/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet > $OUT
