#!/bin/bash
# doc sbnd_xin/113 sec 6 -- evaluate a doc-113 fix arm against production, in the order of 113_pred.txt.
#   scripts/d113_eval.sh <stageA tag, e.g. d113on> <stageB tag, e.g. d113onpr> [samples...]
# Writes docs/113_figs/eval/<stageB>/{metrics.log, compare.log, sentinels.log, census_summary.txt, blobs.txt}.
# Reuses, untouched: scripts/analysis/pr150/pr150_metrics.py (per-event metrics + the pr/149-150 Q1/Q2 census:
# labelled vertex, event_label / nu_evaluated migrations, working points, |dEnu|, TGM/STM/FC flips, pi0,
# resources), scripts/analysis/pr149/sentinels_tolerant.py, and this doc's census on the ON products.
set -u
SX=/home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
cd $SX
A=$1; B=$2; shift 2
SAMPLES=${*:-"nuecc48 ncpi0 mcp1k mcp2k"}
OUT=docs/113_figs/eval/$B; mkdir -p $OUT
echo "== metrics extract $B ($SAMPLES)  $(date +%T)"
python3 scripts/analysis/pr150/pr150_metrics.py extract --arm $B --samples $SAMPLES --jobs 8 > $OUT/metrics.log 2>&1; echo rc=$?
[ -f docs/pr/150_figs/metrics/pr150s0-mcp2k.tsv ] || python3 scripts/analysis/pr150/pr150_metrics.py extract --arm pr150s0 --samples $SAMPLES --jobs 8 >> $OUT/metrics.log 2>&1
echo "== compare pr150s0 -> $B  $(date +%T)"
python3 scripts/analysis/pr150/pr150_metrics.py compare --a pr150s0 --b $B --samples $SAMPLES --ref pr150s0 --movers $OUT/movers.tsv > $OUT/compare.log 2>&1; echo rc=$?
echo "== sentinels on $B  $(date +%T)"
python3 scripts/analysis/pr149/sentinels_tolerant.py --arms "work-*-$B" > $OUT/sentinels.log 2>&1; echo rc=$?
echo "== the doc-113 census on the ON products  $(date +%T)"
mkdir -p $OUT/census
for s in $SAMPLES; do
  python3 scripts/analysis/d113/d113_missing2d.py extract --sample $s --jobs 8 --outdir $OUT/census --stageA $A --stageB $B >> $OUT/census.log 2>&1
done
python3 scripts/analysis/d113/d113_summary.py --census $OUT/census --out $OUT/census_summary.txt --flags $OUT/flags.tsv --exhibits $OUT/exhibits.tsv > /dev/null 2>&1; echo rc=$?
echo "== blob counts / stage-A cost  $(date +%T)"
python3 - "$A" "$OUT" <<'EOF' > $OUT/blobs.txt 2>&1
import json, glob, sys, numpy as np, os, re
A, OUT = sys.argv[1], sys.argv[2]
SX = '/home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
def ev(census):
    d = {}
    for f in glob.glob(f'{census}/*.events.json'):
        for e in json.load(open(f)):
            if 'error' not in e: d[(e['sample'], e['event'])] = e
    return d
off = ev(f'{SX}/docs/113_figs/census'); on = ev(f'{OUT}/census')
common = sorted(set(off) & set(on))
nb = np.array([(off[k]['nblob'], on[k]['nblob']) for k in common], float)
print(f'events common {len(common)}; stage-A live blobs per event: OFF median {np.median(nb[:,0]):.0f}, ON median {np.median(nb[:,1]):.0f}, ON/OFF median {np.median(nb[:,1]/np.maximum(1,nb[:,0])):.3f}, mean {np.mean(nb[:,1]/np.maximum(1,nb[:,0])):.3f}, p90 {np.percentile(nb[:,1]/np.maximum(1,nb[:,0]),90):.3f}')
u2 = np.array([(sum(off[k]['Qunc2'].values())/max(1,sum(off[k]['Q0'].values())), sum(on[k]['Qunc2'].values())/max(1,sum(on[k]['Q0'].values()))) for k in common])
print(f'whole-event uncovered-charge fraction: OFF median {100*np.median(u2[:,0]):.2f} %, ON median {100*np.median(u2[:,1]):.2f} %; total OFF {100*u2[:,0].mean():.2f} % ON {100*u2[:,1].mean():.2f} %')
# stage-A wall / rss per group from .img.time.meta / .ql.time.meta
for stage in ('img', 'ql'):
    w_off, w_on, r_off, r_on = [], [], [], []
    for s in ('nuecc48', 'ncpi0', 'mcp1k', 'mcp2k'):
        for g in glob.glob(f'{SX}/work-{s}-{A}/g*/.{stage}.time.meta'):
            gid = os.path.basename(os.path.dirname(g))
            o = f'{SX}/work-{s}-d102m/{gid}/.{stage}.time.meta'
            if not os.path.exists(o): continue
            def rd(p):
                t = open(p).read(); return float(re.search(r'wall_s=(\d+)', t).group(1)), float(re.search(r'maxrss_kb=(\d+)', t).group(1))
            a = rd(o); b = rd(g); w_off.append(a[0]); w_on.append(b[0]); r_off.append(a[1]); r_on.append(b[1])
    if w_off:
        print(f'{stage}: groups {len(w_off)}; wall ON/OFF median {np.median(np.array(w_on)/np.array(w_off)):.3f} (OFF median {np.median(w_off):.0f} s, ON {np.median(w_on):.0f} s); maxrss ON/OFF median {np.median(np.array(r_on)/np.array(r_off)):.3f}')
EOF
echo "== done $(date +%T)"
