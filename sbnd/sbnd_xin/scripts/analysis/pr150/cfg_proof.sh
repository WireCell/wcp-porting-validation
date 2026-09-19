#!/bin/bash
# doc sbnd_xin/pr/150 sec 1 -- compiled-config proof of the four Steiner seed TLAs threaded into the
# SBND PR job (steiner_blank_plane_mode / _radius, steiner_base_weight_blank_alpha / _scope).
# Fork by duplication (CLAUDE.md M10) of scripts/analysis/pr149/cfg_proof.sh (untouched).
#
#   (a) the PR job at the production operating point from the PRE-knob tree (git archive BASE cfg)
#       and from the working tree with the knobs null / the PDHD-PDVD values / a misspelled mode
#       (expected abort) / the doc-116 trajectory (cs + p3bw) / + the fit-key TrackFitting JSON is not a
#       config matter (read at runtime); prints sha256 and the node-level diff against PRE
#       (expected: null identical; ON changes only CreateSteinerGraph:pr and :prrefresh);
#   (b) the LArSoft one-step chain (per_apa()+pr() in ONE config), sync/bare x tracking-root, PRE vs
#       working tree (expected identical);
#   (c) scripts/cfg/prod_cfg_gate.py on the working tree (expected PASS 21/21);
#   (d) scripts/analysis/pr150/tla_probe_added.py PRE vs working tree: every PRE TLA probes identically on
#       both trees, and every gained TLA changes POST's compiled output (threaded, not dropped).
#
# Usage: bash scripts/analysis/pr150/cfg_proof.sh [BASE=f9665bea] > docs/pr/150_figs/150_cfg_proof.txt
set -u
BASE=${BASE:-f9665bea}
TK=/nfs/data/1/xqian/toolkit-dev/toolkit
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
DATA=/nfs/data/1/xqian/toolkit-dev/wire-cell-data
W=/nfs/data/1/xqian/toolkit-dev/local/bin/wcsonnet
T=$(mktemp -d /home/xqian/tmp/pr150_cfgproof.XXXX)
git -C $TK archive $BASE cfg | tar x -C $T
PRE=$T/cfg; POST=$TK/cfg
echo "# pr150 cfg proof  base=$BASE  post=$(git -C $TK rev-parse --short HEAD) (+ working tree)  $(date +%F_%H:%M)"

PIPE="pipeline_names=['switch_scope','unmerge_bundle','unmerge_assoc','steiner','fiducialutils','tagger_check_tgm','tagger_check_stm','tagger_check_fc','protect_bundle','steiner_refresh','tagger_check_neutrino','numu_bdt_scorer','nue_bdt_scorer','tracking_visitor','tagger_output']"
prjob() {  # cfgroot out [extra TLA...]
  local cfg=$1 out=$2; shift 2
  WIRECELL_PATH=$cfg:$DATA:$DATA/sbnd/photodet $W -A input=in.tar.gz -A output_dir=out \
    --tla-code run=18253 --tla-code subrun=1 --tla-code event=172230 -A reality=data \
    --tla-code "$PIPE" -A save_tensors=out.tar.gz -A dl_weights= "$@" \
    $cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet > $out 2> $out.err
  echo "rc=$?"
}
P3BW=(--tla-code "steiner_blank_plane_mode='prefer3'" --tla-code steiner_base_weight_blank_alpha=0.5 --tla-code "steiner_base_weight_scope='tree+path'")
echo "## (a) PR job"
echo "pre          $(prjob $PRE $T/pre.json)  sha=$(sha256sum < $T/pre.json | cut -c1-16)"
echo "post null    $(prjob $POST $T/null.json)  sha=$(sha256sum < $T/null.json | cut -c1-16)"
echo "post wcp/0/tree (explicit C++ defaults) $(prjob $POST $T/dflt.json --tla-code "steiner_blank_plane_mode='wcp'" --tla-code steiner_base_weight_blank_alpha=0 --tla-code "steiner_base_weight_scope='tree'")  sha=$(sha256sum < $T/dflt.json | cut -c1-16)"
echo "post p3bw    $(prjob $POST $T/p3bw.json "${P3BW[@]}")  sha=$(sha256sum < $T/p3bw.json | cut -c1-16)"
echo "post p3bw+rad $(prjob $POST $T/p3bwr.json "${P3BW[@]}" --tla-code steiner_blank_plane_radius=10)  sha=$(sha256sum < $T/p3bwr.json | cut -c1-16)"
echo "post csp3bw  $(prjob $POST $T/csp3bw.json "${P3BW[@]}" --tla-code "retile_sampler_strategy='charge_stepped'")  sha=$(sha256sum < $T/csp3bw.json | cut -c1-16)"
echo "post typo    $(prjob $POST $T/typo.json --tla-code "steiner_blank_plane_mode='prefer4'")  (expected nonzero: the C++ parser would reject it; the jsonnet passes strings through)"
head -c 300 $T/typo.json.err | tr '\n' ' '; echo
python3 - $T <<'PY'
import json, sys
T = sys.argv[1]
def load(f):
    return {f"{n['type']}:{n.get('name','')}": n for n in json.load(open(f))}
pre = load(f'{T}/pre.json')
for tag in ('null', 'dflt', 'p3bw', 'p3bwr', 'csp3bw'):
    try: b = load(f'{T}/{tag}.json')
    except Exception as e: print(f'-- {tag}: no json ({e})'); continue
    print(f'-- node diff pre -> {tag}: only pre {sorted(set(pre)-set(b))}  only post {sorted(set(b)-set(pre))}')
    for k in sorted(set(pre) & set(b)):
        if pre[k] != b[k]:
            da, db = pre[k].get('data', {}), b[k].get('data', {})
            ks = [kk for kk in sorted(set(da) | set(db)) if da.get(kk) != db.get(kk)]
            print(f'   changed node {k}: keys {ks} -> {[db.get(kk) for kk in ks]}')
    others = [k for k in (set(pre) & set(b)) if pre[k] != b[k] and not k.startswith('CreateSteinerGraph') and k != 'ImproveCluster_2:pr']
    print(f'   other changed nodes: {len(others)}')
PY

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
(cd $SX && python3 scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-17b 2>&1 | tail -2; echo "gate rc=${PIPESTATUS[0]}")
echo "## (d) per-TLA plumbing: scripts/analysis/pr150/tla_probe_added.py PRE vs POST"
(cd $SX && python3 scripts/analysis/pr150/tla_probe_added.py $PRE $POST --jobs 16 2>&1 | tail -8; echo "probe rc=${PIPESTATUS[0]}")
rm -rf $T
