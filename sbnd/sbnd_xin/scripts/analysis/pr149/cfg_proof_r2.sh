#!/bin/bash
# doc pr/149 round 2 -- compiled-config proof of the resample_live_strategy knob
# (fork of cfg_proof.sh, which stays untouched).
#
# round-1 header, for the shared parts:
#
#   (a) the SBND PR job at the production operating point, compiled from the PRE-knob tree
#       (git archive BASE cfg) and from the working tree with the knob null / 'stepped' /
#       'charge_stepped' / charge_stepped + both numeric overrides / amendment 1's
#       steiner_terminal_min_separation (0 and 0.5 with charge_stepped); prints sha256 and the
#       node-level diff against PRE (expected: only BlobSampler live-apa* -> live-cs-apa* and
#       the ImproveCluster_2:pr sampler refs);
#   (b) the LArSoft one-step chain (wcp sbnd/wcls-img-clus-matching-xin.jsonnet), which
#       compiles per_apa() and pr() into ONE config -- the sampler-name clash site -- in
#       modes sync/bare x tracking-root true/false, PRE vs working tree (expected identical).
#       ('preflip' mode aborts in wcsonnet on the PRE tree too: pre-existing, not tested.)
#   (c) scripts/cfg/prod_cfg_gate.py on the working tree (expected PASS 21/21).
#
# Usage: BASE=06fd9e22 bash scripts/analysis/pr149/cfg_proof_r2.sh > docs/pr/149_figs/149_r2_cfg_proof.txt
set -u
BASE=${BASE:-06fd9e22}
TK=/nfs/data/1/xqian/toolkit-dev/toolkit
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
DATA=/nfs/data/1/xqian/toolkit-dev/wire-cell-data
W=/nfs/data/1/xqian/toolkit-dev/local/bin/wcsonnet
T=$(mktemp -d /home/xqian/tmp/pr149r2_cfgproof.XXXX)
git -C $TK archive $BASE cfg | tar x -C $T
PRE=$T/cfg; POST=$TK/cfg
echo "# pr149 round-2 cfg proof  base=$BASE  post=$(git -C $TK rev-parse --short HEAD) (+ working tree)  $(date +%F_%H:%M)"

PIPE="pipeline_names=['switch_scope','unmerge_bundle','unmerge_assoc','steiner','fiducialutils','tagger_check_tgm','tagger_check_stm','tagger_check_fc','protect_bundle','steiner_refresh','tagger_check_neutrino','numu_bdt_scorer','nue_bdt_scorer','tracking_visitor','tagger_output']"
prjob() {  # cfgroot out [extra TLA...]
  local cfg=$1 out=$2; shift 2
  WIRECELL_PATH=$cfg:$DATA:$DATA/sbnd/photodet $W -A input=in.tar.gz -A output_dir=out \
    --tla-code run=18253 --tla-code subrun=1 --tla-code event=172230 -A reality=data \
    --tla-code "$PIPE" -A save_tensors=out.tar.gz -A dl_weights= "$@" \
    $cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet > $out 2> $out.err
  echo "rc=$?"
}
echo "## (a) PR job"
echo "pre          $(prjob $PRE $T/pre.json)  sha=$(sha256sum < $T/pre.json | cut -c1-16)"
echo "post null    $(prjob $POST $T/null.json)  sha=$(sha256sum < $T/null.json | cut -c1-16)"
echo "post rs cs   $(prjob $POST $T/rscs.json --tla-code "resample_live_strategy='charge_stepped'")  sha=$(sha256sum < $T/rscs.json | cut -c1-16)"
echo "post rs st   $(prjob $POST $T/rsst.json --tla-code "resample_live_strategy='stepped'")  sha=$(sha256sum < $T/rsst.json | cut -c1-16)"
echo "post rs+cs   $(prjob $POST $T/rscsr.json --tla-code "resample_live_strategy='charge_stepped'" --tla-code "retile_sampler_strategy='charge_stepped'")  sha=$(sha256sum < $T/rscsr.json | cut -c1-16)"
echo "post typo    $(prjob $POST $T/typo.json --tla-code "resample_live_strategy='chargestepped'")  (expected nonzero: assert)"
python3 - $T <<'PYEOF'
import json, sys
T = sys.argv[1]
def load(f):
    return {f"{n['type']}:{n.get('name','')}": n for n in json.load(open(f))}
pre = load(f'{T}/pre.json')
for tag in ('rscs', 'rsst', 'rscsr'):
    b = load(f'{T}/{tag}.json')
    print(f'-- node diff pre -> {tag}: only pre {sorted(set(pre)-set(b))}  only post {sorted(set(b)-set(pre))}')
    for k in sorted(set(b) - set(pre)):
        d = b[k].get('data', {})
        if k.startswith('BlobSampler'):
            print(f'   new node {k}: strategy={json.dumps(d["strategy"])} extra={json.dumps(d["extra"])}')
        else:
            print(f'   new node {k}: {json.dumps(d, sort_keys=True)}')
    for k in sorted(set(pre) & set(b)):
        if pre[k] != b[k]:
            da, db = pre[k].get('data', {}), b[k].get('data', {})
            ks = [kk for kk in sorted(set(da) | set(db)) if da.get(kk) != db.get(kk)]
            print(f'   changed node {k}: keys {ks}')
            if 'pipeline' in ks:
                print(f'      pipeline head pre {da["pipeline"][:2]}  post {db["pipeline"][:3]}')
PYEOF

echo "## (b) LArSoft one-step chain"
for op in sync bare; do for tr in true false; do
  for side in PRE POST; do
    cfg=$([ $side = PRE ] && echo $PRE || echo $POST)
    (cd $SX/.. && WIRECELL_PATH=$cfg:$DATA:$DATA/sbnd/photodet $W -V input_mask_tags=x -V output_mask_tags=x \
       -V recobwire_tags=x -V summary_tags=x -V trace_tags=x -V opflash0_input_label=x -V opflash1_input_label=x \
       -V reality=data -V pr_operating_point=$op -V enable_tracking_root=$tr \
       wcls-img-clus-matching-xin.jsonnet > $T/onestep_${op}_${tr}_$side.json 2>/dev/null)
  done
  a=$(sha256sum < $T/onestep_${op}_${tr}_PRE.json | cut -c1-16); b=$(sha256sum < $T/onestep_${op}_${tr}_POST.json | cut -c1-16)
  echo "onestep op=$op tracking_root=$tr  pre=$a post=$b  $([ $a = $b ] && echo IDENTICAL || echo DIFFER)"
done; done

echo "## (c) prod_cfg_gate"
(cd $SX && python3 scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-14 2>&1 | tail -2; echo "gate rc=${PIPESTATUS[0]}")
rm -rf $T
