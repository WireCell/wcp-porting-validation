#!/bin/bash
# doc sbnd_xin/116: compiled-config proof of every study cell, two ways.
# Fork by duplication (CLAUDE.md M10) of scripts/analysis/pr150/cfg_proof.sh (untouched).
#
#  (a) standalone: compile the PR job exactly as run_pr_chain_batch.sh's operating-point compile does
#      (its `_optla` block, reality=sim, the production pipeline + pr_display) with each cell's TLA
#      file (+ the tfull TrackFitting JSON) and print the node-level diff against the no-TLA compile.
#      Expected (doc pr/150 150_cell_cfg.txt): cs -> only the live-cs-* sampler swap; p3bw -> only the
#      three keys on CreateSteinerGraph:pr and :prrefresh; csp3bw = both; tfull = csp3bw + the
#      trackfitting_config path (that is the ONLY place the fit keys leave a trace in the config);
#      s0rep -> identical.
#  (b) on the arms: the runner writes the config it hashed into Trun.op_config_sha256 as
#      <arm>/<sub>/.d109-opcfg.json.  Diff each cell arm's file against the doc-115 baseline arm's,
#      so the proof is of what actually ran, not of a re-derivation.  Skipped for arms not yet run.
#
# Usage: bash scripts/d116/cfg_proof.sh [cells...] > docs/116_figs/116_cfg_proof.txt
set -u
cd -P "$(dirname "$0")/../.." || exit 1
SX=$PWD
TK=/nfs/data/1/xqian/toolkit-dev/toolkit
DATA=/nfs/data/1/xqian/toolkit-dev/wire-cell-data
W=/nfs/data/1/xqian/toolkit-dev/local/bin/wcsonnet
CELLS=${*:-s0rep cs p3bw csp3bw tfull}
T=$(mktemp -d /home/xqian/tmp/d116_cfgproof.XXXX)
echo "# d116 cfg proof  toolkit=$(git -C $TK rev-parse --short HEAD)  wcp=$(git -C $SX rev-parse --short HEAD)  $(date +%F_%H:%M)"

PIPE="switch_scope,unmerge_bundle,unmerge_assoc,steiner,fiducialutils,tagger_check_tgm,tagger_check_stm,tagger_check_fc,protect_bundle,steiner_refresh,tagger_check_neutrino,numu_bdt_scorer,nue_bdt_scorer,tracking_visitor,tagger_output,pr_display"
prjob() {  # out reality [extra TLA...]
  local out=$1 reality=$2; shift 2
  WIRECELL_PATH=$TK/cfg:$DATA:$DATA/sbnd/photodet $W \
    --tla-str input=/placeholder/pctree.tar.gz --tla-code "anode_indices=[0,1]" \
    --tla-str output_dir=/placeholder --tla-code run=0 --tla-code subrun=0 --tla-code event=0 \
    --tla-str "reality=$reality" \
    --tla-code "pipeline_names=[$(echo "$PIPE" | sed "s/[^,]\+/'&'/g")]" "$@" \
    $SX/wct-pr-perevt.jsonnet > $out 2> $out.err
  echo "rc=$?"
}
cell_tla() {  # cell -> the --tla-code / --tla-str list, as stageB_cell.sh exports them
  local c=$1 f=$SX/docs/116_figs/tla/$1.tla j=$SX/docs/116_figs/tla/$1.tfjson
  [ -s "$j" ] && printf '%s\n' --tla-str "trackfitting_config=$(head -1 "$j")"
  [ -s "$f" ] && grep -v '^\s*#' "$f" | grep -v '^\s*$' | while IFS= read -r l; do printf '%s\n' --tla-code "$l"; done
  return 0
}

echo "## (a) standalone compile, reality=sim, runner _optla form"
echo "null    $(prjob $T/null.json sim)  sha=$(sha256sum < $T/null.json | cut -c1-16)"
for c in $CELLS; do
  mapfile -t X < <(cell_tla $c)
  echo "$c  $(prjob $T/$c.json sim "${X[@]}")  sha=$(sha256sum < $T/$c.json | cut -c1-16)  tla=[${X[*]}]"
done
python3 - $T "$CELLS" <<'PY'
import json, sys
T, cells = sys.argv[1], sys.argv[2].split()
def load(f): return {f"{n['type']}:{n.get('name','')}": n for n in json.load(open(f))}
pre = load(f'{T}/null.json')
def diff(pre, b, tag):
    print(f'-- node diff null -> {tag}: only null {sorted(set(pre)-set(b))}  only {tag} {sorted(set(b)-set(pre))}')
    n = 0
    for k in sorted(set(pre) & set(b)):
        if pre[k] != b[k]:
            n += 1
            da, db = pre[k].get('data', {}), b[k].get('data', {})
            ks = [kk for kk in sorted(set(da) | set(db)) if da.get(kk) != db.get(kk)]
            vals = [db.get(kk) for kk in ks]
            vals = [v if len(json.dumps(v)) < 120 else json.dumps(v)[:117] + '...' for v in vals]
            print(f'   changed node {k}: keys {ks} -> {vals}')
    print(f'   changed nodes: {n}  IDENTICAL' if not n and set(pre) == set(b) else f'   changed nodes: {n}')
for c in cells:
    try: b = load(f'{T}/{c}.json')
    except Exception as e: print(f'-- {c}: no json ({e})'); continue
    diff(pre, b, c)
PY

echo "## (b) the arms' own .d109-opcfg.json vs the doc-115 baseline arm's"
python3 - $SX "$CELLS" <<'PY'
import json, sys, glob, hashlib, os
SX, cells = sys.argv[1], sys.argv[2].split()
def load(f): return {f"{n['type']}:{n.get('name','')}": n for n in json.load(open(f))}
def sha(f): return hashlib.sha256(open(f,'rb').read()).hexdigest()[:16]
for s, base, pat in (('cv', 'work-r3cv-d115pr/f000', 'work-r3cv-d116%s/f*'),
                     ('nuecc', 'work-r3nue-d115pr/f000', 'work-r3nue-d116%s/f*'),
                     ('off', 'work-r3off-d115pr', 'work-r3off-d116%s')):
    bf = f'{SX}/{base}/.d109-opcfg.json'
    if not os.path.exists(bf): print(f'-- {s}: no baseline opcfg {bf}'); continue
    pre = load(bf); print(f'-- {s} baseline {base} sha={sha(bf)}')
    for c in cells:
        fs = sorted(glob.glob(f'{SX}/{pat % c}/.d109-opcfg.json'))
        if not fs: print(f'   {c}: arm not run'); continue
        shas = sorted({sha(f) for f in fs})
        b = load(fs[0])
        changed = [k for k in sorted(set(pre) & set(b)) if pre[k] != b[k]]
        only = (sorted(set(pre)-set(b)), sorted(set(b)-set(pre)))
        det = []
        for k in changed:
            da, db = pre[k].get('data', {}), b[k].get('data', {})
            ks = [kk for kk in sorted(set(da) | set(db)) if da.get(kk) != db.get(kk)]
            det.append((k, ks))
        print(f'   {c}: {len(fs)} sub-root(s), distinct sha {shas}; only base {only[0]} only cell {only[1]}; changed {det}')
PY
rm -rf $T
