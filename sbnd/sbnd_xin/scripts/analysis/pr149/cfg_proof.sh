#!/bin/bash
# doc pr/149 sec 3 -- compiled-config proof of the retile_sampler_strategy knob.
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
# Usage: bash scripts/analysis/pr149/cfg_proof.sh [BASE=9d8fd892] > docs/pr/149_figs/149_cfg_proof.txt
set -u
BASE=${BASE:-9d8fd892}
TK=/nfs/data/1/xqian/toolkit-dev/toolkit
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
DATA=/nfs/data/1/xqian/toolkit-dev/wire-cell-data
W=/nfs/data/1/xqian/toolkit-dev/local/bin/wcsonnet
T=$(mktemp -d /home/xqian/tmp/pr149_cfgproof.XXXX)
git -C $TK archive $BASE cfg | tar x -C $T
PRE=$T/cfg; POST=$TK/cfg
echo "# pr149 cfg proof  base=$BASE  post=$(git -C $TK rev-parse --short HEAD) (+ working tree)  $(date +%F_%H:%M)"

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
echo "post stepped $(prjob $POST $T/stepped.json --tla-code "retile_sampler_strategy='stepped'")  sha=$(sha256sum < $T/stepped.json | cut -c1-16)"
echo "post cs      $(prjob $POST $T/cs.json --tla-code "retile_sampler_strategy='charge_stepped'")  sha=$(sha256sum < $T/cs.json | cut -c1-16)"
echo "post cs+num  $(prjob $POST $T/csn.json --tla-code "retile_sampler_strategy='charge_stepped'" --tla-code retile_sampler_charge_threshold=2000 --tla-code retile_sampler_wire_product=10000)  sha=$(sha256sum < $T/csn.json | cut -c1-16)"
echo "post typo    $(prjob $POST $T/typo.json --tla-code "retile_sampler_strategy='chargestepped'")  (expected nonzero: assert)"
echo "post sep0    $(prjob $POST $T/sep0.json --tla-code steiner_terminal_min_separation=0)  sha=$(sha256sum < $T/sep0.json | cut -c1-16)"
echo "post cs+sep  $(prjob $POST $T/cssep.json --tla-code "retile_sampler_strategy='charge_stepped'" --tla-code steiner_terminal_min_separation=0.5)  sha=$(sha256sum < $T/cssep.json | cut -c1-16)"
python3 - $T <<'EOF'
import json, sys
T = sys.argv[1]
def load(f):
    return {f"{n['type']}:{n.get('name','')}": n for n in json.load(open(f))}
pre = load(f'{T}/pre.json')
for tag in ('cs', 'csn', 'cssep'):
    b = load(f'{T}/{tag}.json')
    print(f'-- node diff pre -> {tag}: only pre {sorted(set(pre)-set(b))}  only post {sorted(set(b)-set(pre))}')
    for k in sorted(set(pre) & set(b)):
        if pre[k] != b[k]:
            da, db = pre[k].get('data', {}), b[k].get('data', {})
            ks = [kk for kk in sorted(set(da) | set(db)) if da.get(kk) != db.get(kk)]
            print(f'   changed node {k}: keys {ks}')
    for k in sorted(set(b) - set(pre)):
        print(f'   new node {k}: strategy={json.dumps(b[k]["data"]["strategy"])}')
    for k in sorted(set(pre) & set(b)):
        if pre[k] != b[k] and k.startswith('CreateSteinerGraph'):
            print(f'   {k}: terminal_min_separation={b[k]["data"].get("terminal_min_separation")}')
    others = [k for k in (set(pre) & set(b)) if pre[k] != b[k] and k != 'ImproveCluster_2:pr' and not k.startswith('CreateSteinerGraph')]
    print(f'   other changed nodes: {len(others)}')
EOF

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
