#!/usr/bin/env python3
"""doc sbnd_xin/123 round 3 -- what the Q/L moves go to.

Joins r3_ql_compare.py's per-cluster table (A = reco1-flash arm, B = hit-flash arm) with
r1_census.py's per-flash table of the hit-flash arm, and classes every MOVED cluster by the hit
flash it lands on: 'restored' (a flash reco1 did not have: absorbed / vetoed / dropped / prepulse),
'both-had' (the flash has a reco1 match), and by its PE; plus the beam-window flow (clusters
leaving / entering +0.3..1.9 us) with their predicted light.  Anode n == TPC n.

usage: r3_move_classes.py <r3.tsv> <r1.tsv> [--win-us 0.05]
"""
import argparse, csv, json
from collections import defaultdict, Counter
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("r3"); ap.add_argument("r1"); ap.add_argument("--win-us", type=float, default=0.05)
a = ap.parse_args()
fl = defaultdict(list)   # (event, tpc) -> [(t_us, pe, class)]
for r in csv.DictReader(open(a.r1), delimiter="\t"):
    if r["ours_idx"] != "-1":
        fl[(int(r["event"]), int(r["tpc"]))].append((float(r["ours_t_us"]), float(r["ours_pe"]), r["class"]))
def cls_of(ev, tpc, t):
    best = None
    for x in fl.get((ev, tpc), []):
        if abs(x[0] - t) <= a.win_us and (best is None or abs(x[0] - t) < abs(best[0] - t)):
            best = x
    return best[2] if best else "unknown"
rows = list(csv.DictReader(open(a.r3), delimiter="\t"))
mv = [r for r in rows if r["class"] == "moved"]
dest = Counter(); dest_big = Counter(); flow = Counter(); pred_out, pred_in = [], []
dt = []
for r in mv:
    ev, an = int(r["event"]), int(r["anode"])
    tb, pb, predb = float(r["B_t_us"]), float(r["B_pe"]), float(r["B_pred"])
    ta, pa, preda = float(r["A_t_us"]), float(r["A_pe"]), float(r["A_pred"])
    c = cls_of(ev, an, tb)
    grp = "restored" if c in ("absorbed", "vetoed", "dropped", "prepulse", "piece") else ("both-had" if c in ("match", "bugged") else c)
    dest[(grp, c)] += 1
    if pb >= 100: dest_big[grp] += 1
    dt.append(tb - ta)
    ab, bb = r["A_beam"] == "1", r["B_beam"] == "1"
    if ab and not bb: flow["leaves beam"] += 1; pred_out.append(preda)
    elif bb and not ab: flow["enters beam"] += 1; pred_in.append(predb)
    elif ab and bb: flow["beam->beam"] += 1
dt = np.abs(np.array(dt))
out = dict(moved=len(mv), dest=dict(("%s/%s" % k, v) for k, v in dest.items()), dest_ge100pe=dict(dest_big),
           abs_dt_lt8us=int((dt < 8).sum()), abs_dt_8_100us=int(((dt >= 8) & (dt < 100)).sum()), abs_dt_ge100us=int((dt >= 100).sum()),
           beam_flow=dict(flow),
           pred_leaving_beam_q=[round(float(x), 1) for x in np.percentile(pred_out, [0, 25, 50, 75, 100])] if pred_out else [],
           pred_entering_beam_q=[round(float(x), 1) for x in np.percentile(pred_in, [0, 25, 50, 75, 100])] if pred_in else [],
           leaving_beam_pred_ge500=int(sum(1 for x in pred_out if x >= 500)), entering_beam_pred_ge500=int(sum(1 for x in pred_in if x >= 500)))
print(json.dumps(out, indent=1))
