#!/bin/bash
# doc pr/100 -- the neutrino-vertex label-refresh runbook, as one script.
#
# Once the PR chain is declared final, this is the single command that:
#   1. builds current-epoch base+topo arms over the LABELED universe only
#      (not every event in a sample -- the label pool is a fixed subset);
#   2. carries every hand-scan label onto the base arm by position (TOL=1.0),
#      holding declined events back for a possible rescan rather than
#      dropping them;
#   3. seals a fresh lockbox (recorded, never read here);
#   4. prints the report card (vtx_report.py) and the rerank closure/search
#      (rerank_tune.py), both gated on the new epoch's own arms.
#
# It does NOT decide whether to tune anything -- that reads the printed
# report and is a separate, human step.  Nothing here writes toolkit code.
#
# Usage:
#   ./vtx_prep.sh <tag>
#     <tag>  names the arms: work-<tag>-base-<sample>, work-<tag>-topo-<sample>,
#            label carry tag vtxscan-<tag>-*, lockbox runs/lockbox-<tag>.txt
#
# Env: PR_JOBS (default 6, M5 -- CLAUDE.md caps interactive batches ~6-8;
#      raise only with explicit authorization), SBND_DL_VTX_TOPO_WEIGHT for
#      the topo arm (default 3.0, the pr/89 sec 11.5 offline-selected value --
#      only the topo_frac/topo_votes rows it exposes matter; its own
#      decisions are not scored as production).
set -euo pipefail

SX=$(cd "$(dirname "$0")" && pwd -P)
TAG=${1:?usage: ./vtx_prep.sh <tag>}
PR_JOBS=${PR_JOBS:-6}
TOPO_W=${SBND_DL_VTX_TOPO_WEIGHT:-3.0}
SAMPLES=(nuecc48 ncpi0 mcp1k mcp2k)

echo "== M1 freshness proof =="
git -C "$SX/../../../toolkit" log --oneline -1
ls -la "$SX/../../../toolkit/build/clus/libWireCellClus.so"
echo "(verify this mtime is newer than your last source edit before trusting anything below)"

echo
echo "== step 1: labeled-universe event lists per sample =="
python3 - "$SX" <<'PYEOF'
import sys, collections
sys.path.insert(0, sys.argv[1] + '/vtx_rules')
import vtx_io as V

def sample_of(lab):
    tag = lab['tag']
    if tag in ('vtxscan-mcp2k', 'vtxscan-mcp2k-auto', 'vtxscan-mcp2k-ragree'):
        return 'mcp2k'
    for s in ('nuecc48', 'ncpi0', 'mcp1k'):
        if s in tag:
            return s
    arm = lab.get('arm') or ''
    for s in ('nuecc48', 'ncpi0', 'mcp1k', 'mcp2k'):
        if s in arm:
            return s
    return None

tags = V.TAGS_HARV3 + V.TAGS_MCP2K + V.TAGS_MCP2K_AUTO + ['vtxscan-mcp2k-ragree']
L = V.load_labels(tags=tags)
by_sample = collections.defaultdict(set)
for r in L:
    s = sample_of(r)
    assert s is not None, ('unresolved sample', r['tag'], r['eventNo'])
    by_sample[s].add(r['eventNo'])
total = 0
for s, evts in sorted(by_sample.items()):
    with open('/home/xqian/tmp/vtxprep-evts-%s.txt' % s, 'w') as fh:
        fh.write(' '.join(str(e) for e in sorted(evts)))
    print('%-8s %d' % (s, len(evts)))
    total += len(evts)
print('total %d' % total)
PYEOF

echo
echo "== step 2: base + topo arms (labeled universe only) =="
for s in "${SAMPLES[@]}"; do
    evts=$(cat /home/xqian/tmp/vtxprep-evts-${s}.txt)
    [ -z "$evts" ] && { echo "skip $s: no labeled events"; continue; }
    echo "-- $s base --"
    PR_JOBS=$PR_JOBS PR_EXTRA_STAGES=pr_display \
        "$SX/run_pr_chain_batch.sh" "$SX/work-${s}-ql0819" \
        "$SX/work-${TAG}-base-${s}" data $evts
    echo "-- $s topo (SBND_DL_VTX_TOPO_WEIGHT=$TOPO_W) --"
    PR_JOBS=$PR_JOBS PR_EXTRA_STAGES=pr_display SBND_DL_VTX_TOPO_WEIGHT=$TOPO_W \
        "$SX/run_pr_chain_batch.sh" "$SX/work-${s}-ql0819" \
        "$SX/work-${TAG}-topo-${s}" data $evts
done

echo
echo "== step 3: carry labels onto the base arm (position join, TOL=1.0) =="
mkdir -p "$SX/dl_vtx_training/runs/${TAG}"
python3 "$SX/vtx_rules/carry_labels.py" --write \
    --delta-list "$SX/dl_vtx_training/runs/${TAG}/delta.txt" \
    --tsv "$SX/dl_vtx_training/runs/${TAG}/carry.tsv" \
    --arms \
        vtxscan-harv3-nuecc48=work-${TAG}-base-nuecc48:vtxscan-${TAG}-nuecc48 \
        vtxscan-harv3-ncpi0=work-${TAG}-base-ncpi0:vtxscan-${TAG}-ncpi0 \
        vtxscan-harv3-mcp1k=work-${TAG}-base-mcp1k:vtxscan-${TAG}-mcp1k \
        vtxscan-harv3-delta=work-${TAG}-base-{sample}:vtxscan-${TAG}-delta \
        vtxscan-mcp2k=work-${TAG}-base-mcp2k:vtxscan-${TAG}-mcp2k \
        vtxscan-mcp2k-auto=work-${TAG}-base-mcp2k:vtxscan-${TAG}-mcp2k-auto \
        vtxscan-mcp2k-ragree=work-${TAG}-base-mcp2k:vtxscan-${TAG}-mcp2k-ragree
echo "declined events (held back, not rescanned): $SX/dl_vtx_training/runs/${TAG}/delta.txt"

echo
echo "== step 4: seal a fresh lockbox (recorded, NOT read) =="
mkdir -p "$SX/dl_vtx_training/runs/${TAG}"
python3 - "$SX" "$TAG" <<'PYEOF'
import sys, random, zlib
sys.path.insert(0, sys.argv[1] + '/vtx_rules')
import vtx_io as V
sx, tag = sys.argv[1], sys.argv[2]
tags = ['vtxscan-%s-%s' % (tag, s) for s in
        ('nuecc48', 'ncpi0', 'mcp1k', 'delta', 'mcp2k', 'mcp2k-auto', 'mcp2k-ragree')]
try:
    L = V.load_labels(tags=tags)
except FileNotFoundError as e:
    print('WARN: %s -- run step 3 first' % e); sys.exit(0)
evts = sorted({r['eventNo'] for r in L})
# seed is a function of the TAG, not today's date -- every re-run of this
# script for the SAME tag draws the SAME lockbox (idempotent); a fresh round
# uses a fresh tag and therefore a fresh seed automatically.
seed = zlib.crc32(tag.encode())
rnd = random.Random(seed)
rnd.shuffle(evts)
n = max(1, len(evts) // 4)
lockbox = sorted(evts[:n])
path = '%s/dl_vtx_training/runs/%s/lockbox.txt' % (sx, tag)
with open(path, 'w') as fh:
    fh.write('\n'.join(str(e) for e in lockbox) + '\n')
print('sealed %d/%d events (seed=crc32(%r)=%d) -> %s -- NOT read'
      % (n, len(evts), tag, seed, path))
PYEOF

echo
echo "== step 4b: IPW weights over the mcp2k scan tiers =="
python3 "$SX/dl_vtx_training/ipw_weights.py" \
    --runs "$SX/vtx_rules/runs/mcp2k-20260816" \
    --tags vtxscan-${TAG}-mcp2k vtxscan-${TAG}-mcp2k-auto vtxscan-${TAG}-mcp2k-ragree \
    --tsv "$SX/dl_vtx_training/runs/ipw-${TAG}.tsv"

echo
echo "== step 5: report card (excludes the lockbox from any read here) =="
python3 "$SX/dl_vtx_training/vtx_report.py" \
    --tags vtxscan-${TAG}-nuecc48 vtxscan-${TAG}-ncpi0 vtxscan-${TAG}-mcp1k \
           vtxscan-${TAG}-delta vtxscan-${TAG}-mcp2k vtxscan-${TAG}-mcp2k-auto \
           vtxscan-${TAG}-mcp2k-ragree \
    --arm-roots nuecc=work-${TAG}-base-nuecc48 ncpi0=work-${TAG}-base-ncpi0 \
                mcp1k=work-${TAG}-base-mcp1k mcp2k=work-${TAG}-base-mcp2k \
    --ipw-tsv "$SX/dl_vtx_training/runs/ipw-${TAG}.tsv" \
    --tsv "$SX/dl_vtx_training/runs/vtx-report-${TAG}.tsv"

echo
echo "== step 6: rerank closure + coordinate ascent at the new epoch =="
echo "doc pr/100 sec 4: run closure FIRST and read it before trusting --search."
echo "'production still optimal -> stop' is a legitimate outcome, not a failure."
python3 "$SX/dl_vtx_training/rerank_tune.py" --search \
    --arm-template "$SX/work-${TAG}-{arm}-{sample}/pr_evt{evt}/calib-pr-evt{evt}.json" \
    --ipw-tsv "$SX/dl_vtx_training/runs/ipw-${TAG}.tsv" \
    --tags vtxscan-harv3-nuecc48 vtxscan-harv3-ncpi0 vtxscan-harv3-mcp1k vtxscan-harv3-delta \
           vtxscan-mcp2k vtxscan-mcp2k-auto vtxscan-mcp2k-ragree \
    --tsv "$SX/dl_vtx_training/runs/rerank-search-${TAG}.tsv"
