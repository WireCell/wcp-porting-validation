#!/bin/bash
# doc sbnd_xin/118 gate G3: what the flip changes in the compiled config, node by node.
# Fork by duplication (M10) of scripts/d117/cfg_proof.sh, which stays untouched.
#
# Compiles the SBND PR job in run_pr_chain_batch.sh's operating-point form (reality=sim, the
# production pipeline + pr_display) with NO cell TLAs, against two cfg trees:
#   PRE  = `git archive HEAD cfg` -- the tree before this round's edit
#   POST = the working tree
# and prints the node-level diff.  Expected, from docs/116_figs/116_cfg_proof.txt's null->csp3bw
# block: exactly 3 changed nodes (CreateSteinerGraph:pr, CreateSteinerGraph:prrefresh,
# ImproveCluster_2:pr) plus the two BlobSampler renames live-apa{0,1}-0 -> live-cs-apa{0,1}-0.
# trackfitting_config_file must NOT move: the fit keys are a change of that file's CONTENTS, and
# this instrument is blind to them by construction (that is what G2 is for).
#
# Usage: bash scripts/d118/cfg_diff.sh > docs/118_figs/118_cfg_diff.txt
set -u
cd -P "$(dirname "$0")/../.." || exit 1
SX=$PWD
TK=/nfs/data/1/xqian/toolkit-dev/toolkit
DATA=/nfs/data/1/xqian/toolkit-dev/wire-cell-data
W=/nfs/data/1/xqian/toolkit-dev/local/bin/wcsonnet
T=$(mktemp -d /home/xqian/tmp/d118_cfgdiff.XXXX)
echo "# d118 cfg diff  toolkit=$(git -C $TK rev-parse --short HEAD)(dirty)  wcp=$(git -C $SX rev-parse --short HEAD)  $(date +%F_%H:%M)"

mkdir -p "$T/pre"
git -C $TK archive HEAD cfg | tar -x -C "$T/pre" || { echo "git archive failed"; exit 1; }
echo "PRE  cfg tree: $T/pre/cfg   (git HEAD, pristine)"
echo "POST cfg tree: $TK/cfg      (working tree)"

PIPE="switch_scope,unmerge_bundle,unmerge_assoc,steiner,fiducialutils,tagger_check_tgm,tagger_check_stm,tagger_check_fc,protect_bundle,steiner_refresh,tagger_check_neutrino,numu_bdt_scorer,nue_bdt_scorer,tracking_visitor,tagger_output,pr_display"
prjob() {  # cfgroot out
  local cfg=$1 out=$2
  WIRECELL_PATH=$cfg:$DATA:$DATA/sbnd/photodet $W \
    --tla-str input=/placeholder/pctree.tar.gz --tla-code "anode_indices=[0,1]" \
    --tla-str output_dir=/placeholder --tla-code run=0 --tla-code subrun=0 --tla-code event=0 \
    --tla-str "reality=sim" \
    --tla-code "pipeline_names=[$(echo "$PIPE" | sed "s/[^,]\+/'&'/g")]" \
    $cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet > $out 2> $out.err
  echo "rc=$?"
}
echo "pre   $(prjob "$T/pre/cfg" $T/pre.json)  sha=$(sha256sum < $T/pre.json | cut -c1-16)"
echo "post  $(prjob "$TK/cfg"    $T/post.json) sha=$(sha256sum < $T/post.json | cut -c1-16)"

python3 - "$T" <<'PY'
import json, sys
T = sys.argv[1]
def load(f):
    return {f"{n['type']}:{n.get('name','')}": n for n in json.load(open(f))}
a, b = load(f'{T}/pre.json'), load(f'{T}/post.json')
print(f'-- node diff pre -> post: only pre {sorted(set(a)-set(b))}  only post {sorted(set(b)-set(a))}')
n = 0
for k in sorted(set(a) & set(b)):
    if a[k] != b[k]:
        n += 1
        da, db = a[k].get('data', {}), b[k].get('data', {})
        keys = sorted(set(da) | set(db))
        moved = [x for x in keys if da.get(x) != db.get(x)]
        print(f'   changed node {k}: keys {moved}')
        for x in moved:
            print(f'       {x}: {str(da.get(x))[:120]}  ->  {str(db.get(x))[:120]}')
print(f'   changed nodes: {n}')
tf = [k for k in set(a) & set(b)
      if a[k].get('data', {}).get('trackfitting_config_file') != b[k].get('data', {}).get('trackfitting_config_file')]
print(f'-- trackfitting_config_file moved on: {sorted(tf)}  (expected: [])')
PY
rm -rf "$T/pre"
