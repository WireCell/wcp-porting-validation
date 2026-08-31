#!/bin/bash
# doc pr/107 -- evaluation after /home/xqian/tmp/pr107_arms.sh: OFF gate, ON target metric (own cloud), movers, scores, point-drop census.
set -u
cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin
O="--orig-tags vtxscan-harv3-nuecc48 vtxscan-harv3-ncpi0 vtxscan-harv3-mcp1k vtxscan-harv3-delta vtxscan-mcp2k vtxscan-mcp2k-auto vtxscan-mcp2k-ragree"
mkdir -p docs/pr/107_on
for s in nuecc48 ncpi0; do
  echo "=== OFF gate $s (new binary no env vs old binary cne-off)"
  python3 scripts/pr85_hash_gate.py work-vtx106-cne-off-$s work-pr107-off-$s > /home/xqian/tmp/pr107_gate_$s.log 2>&1; echo gate_rc=$?; tail -1 /home/xqian/tmp/pr107_gate_$s.log
  echo "=== ON vs OFF archives (expected to differ) $s"
  python3 scripts/pr85_hash_gate.py work-pr107-off-$s work-pr107-on-$s > /home/xqian/tmp/pr107_onoff_$s.log 2>&1; echo onoff_rc=$?; tail -1 /home/xqian/tmp/pr107_onoff_$s.log
  ev=/home/xqian/tmp/pr106_nue_evts.txt; [ $s = ncpi0 ] && ev=/home/xqian/tmp/pr106_ncpi0_evts.txt
  echo "=== target metric ON (own pre-DL cloud, orig labels) $s"
  python3 dl_vtx_training/vtx_target_eval.py $O --harv-base "work-pr107-on-{sample}" --harv-rows "work-pr107-on-{sample}" \
     --live-arms base=work-pr107-on-{sample} trad=work-vtx105-trad-{sample} --only-events $ev --closure --table \
     --events-tsv docs/pr/107_on/events-$s.tsv > /home/xqian/tmp/pr107_eval_$s.log 2>&1; echo eval_rc=$?
  grep -E "^  (nuecc48|ncpi0) |closure|mismatch|M1|M3|DL-alone|production" /home/xqian/tmp/pr107_eval_$s.log | head -30
  echo "=== movers $s"
  python3 scripts/pr90_movers.py work-pr107-off-$s work-pr107-on-$s --tags vtx105 --tsv docs/pr/107_on/movers-$s.tsv > /home/xqian/tmp/pr107_mov_$s.log 2>&1; echo mov_rc=$?; tail -4 /home/xqian/tmp/pr107_mov_$s.log
  echo "=== scores $s"
  python3 scripts/pr83r3_scores_ab.py work-pr107-off-$s work-pr107-on-$s --tsv docs/pr/107_on/scores-$s.tsv > /home/xqian/tmp/pr107_sc_$s.log 2>&1; echo sc_rc=$?; tail -8 /home/xqian/tmp/pr107_sc_$s.log
  echo "=== pre-dQ/dx drop census $s (OFF arm logs)"
  for a in off on; do grep -h "pre-dQ/dx form_map_graph dropped" work-pr107-$a-$s/pr_evt*/wct_pr_evt*.log 2>/dev/null | awk -v a=$a '{for(i=1;i<=NF;i++) if($i=="dropped"){d+=$(i+1)}} END{print a, "dropped points:",d+0, "drop lines:",NR}'; done
  echo "=== wall $s"; for a in off on; do python3 - "$a" "$s" <<'PY'
import sys,glob,re
a,s=sys.argv[1:]; t=[]
for f in glob.glob(f"work-pr107-{a}-{s}/pr_evt*/time.txt")+glob.glob(f"work-pr107-{a}-{s}/pr_evt*/*time*"):
    try: t.append(float(open(f).read().split()[0]))
    except Exception: pass
print(a, "n", len(t), "mean %.1f sum %.0f"%(sum(t)/len(t), sum(t)) if t else "")
PY
  done
done
echo "=== EVAL DONE $(date +%T)"
