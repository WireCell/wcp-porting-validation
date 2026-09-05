#!/usr/bin/env python3
"""Doc pdvd/41 sec 10 -- the curved-FV A/B: TGM / STM / FC verdicts, arm vs arm.

Counts are per (event, cluster), read from the PR log's own per-object verdict
lines, NOT from a line count (doc pdvd/36: a log-line count is not an object
count -- TaggerCheckSTM emits a second shape, "already TGM; skipping", for a
cluster it never evaluates).

  TaggerCheckTGM: cluster N -> TGM=true|false
  TaggerCheckSTM: cluster N -> STM=0|1 TGM=0|1   |   cluster N already TGM; skipping
  TaggerCheckFC:  cluster N -> FC=true|false

Both arms must carry the SAME cluster id set per event (identical pctree input),
which is asserted rather than assumed.

Repro:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 docs/nf_sp_img_clus/scripts/fv_curved_ab.py d41fvoff d41fvon \
      --out /home/xqian/tmp/doc41/ab
"""
import argparse, json, os, re, sys, glob
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_curved_load import bee_points
from fv_curved_map import XW, YW, ZLO, ZHI, CATH, WALLS, wall_dist

RE_TGM = re.compile(r"TaggerCheckTGM: cluster (\d+) . TGM=(true|false)")
RE_STM = re.compile(r"TaggerCheckSTM: cluster (\d+) . STM=(\d) TGM=(\d)")
RE_STM_SKIP = re.compile(r"TaggerCheckSTM: cluster (\d+) already TGM; skipping")
RE_FC = re.compile(r"TaggerCheckFC: cluster (\d+) . FC=(true|false)")


def read_arm(pdvd, tag):
    """{(run, idx): {cluster: {'tgm':b,'stm':b,'fc':b,'stm_skipped':b}}}"""
    out = {}
    for d in sorted(glob.glob(os.path.join(pdvd, "work", f"*_{tag}"))):
        base = os.path.basename(d)
        run, idx, _ = base.split("_", 2)
        logs = glob.glob(os.path.join(d, "wct_pr_*.log"))
        if not logs:
            continue
        ev = defaultdict(lambda: dict(tgm=None, stm=None, fc=None, stm_skipped=False))
        with open(logs[0], errors="replace") as f:
            for line in f:
                if "TaggerCheck" not in line:
                    continue
                m = RE_TGM.search(line)
                if m:
                    ev[int(m.group(1))]["tgm"] = (m.group(2) == "true"); continue
                m = RE_STM.search(line)
                if m:
                    ev[int(m.group(1))]["stm"] = (m.group(2) == "1"); continue
                m = RE_STM_SKIP.search(line)
                if m:
                    ev[int(m.group(1))]["stm_skipped"] = True; continue
                m = RE_FC.search(line)
                if m:
                    ev[int(m.group(1))]["fc"] = (m.group(2) == "true")
        out[(run, idx)] = dict(ev)
    return out


def tally(arm, key):
    n = 0
    for ev in arm.values():
        for c in ev.values():
            if c[key]:
                n += 1
    return n


def geometry(pdvd, tag, cats):
    """Where do the clusters live?  cats maps (run, idx, cid) -> category label.

    Per cluster, from the arm's own Bee clustering-global layer (t0-corrected x):
    the two PCA ends, each end's distance to the NEAREST boundary surface of the
    volume (the four transverse walls and the anode faces -- the cathode slab is
    spanned by both fiducials, so a cathode end is not a boundary contact), and
    the WORST of the two.  TGM needs BOTH ends at a boundary, so the worst end is
    the one that decides, and its distance is what separates a real exiter from a
    track the flat 15 cm inset only called one."""
    byev = defaultdict(list)
    for (run, idx, cid) in cats:
        byev[(run, idx)].append(cid)
    rows = []
    for (run, idx), cids in sorted(byev.items()):
        d = os.path.join(pdvd, "work", f"{run}_{idx}_{tag}")
        try:
            P, q, cluster_id, _, _ = bee_points(d)
        except Exception as ex:
            print("  geom: cannot read", d, ex); continue
        keep = np.abs(P[:, 0]) < 1e4                      # drop the no-t0 sentinel
        for cid in cids:
            cat = cats[(run, idx, cid)]
            m = (cluster_id == cid) & keep
            if m.sum() < 5:
                rows.append(dict(run=run, idx=idx, cid=int(cid), cat=cat, n=int(m.sum()),
                                 wall=None, no_t0=True))
                continue
            Q = P[m]
            c = Q - Q.mean(0)
            axis = np.linalg.svd(c, full_matrices=False)[2][0]
            t = c @ axis
            ends = [Q[int(np.argmax(t))], Q[int(np.argmin(t))]]
            edist, ewall = [], []
            for e in ends:
                dd = {w: float(wall_dist(w, e[1], e[2])) for w in WALLS}
                dd["anode"] = float(XW - abs(e[0]))
                w = min(dd, key=dd.get)
                edist.append(dd[w]); ewall.append(w)
            iw = int(np.argmax(edist))                    # the deciding (worst) end
            rows.append(dict(run=run, idx=idx, cid=int(cid), cat=cat, n=int(m.sum()),
                             no_t0=False,
                             wall=ewall[iw], worst_end_cm=round(edist[iw], 2),
                             best_end_cm=round(edist[1 - iw], 2),
                             x_cm=round(float(ends[iw][0]), 1),
                             half=("cathode" if abs(ends[iw][0]) < 170 else "anode"),
                             len_cm=round(float(t.max() - t.min()), 1)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("off_tag"); ap.add_argument("on_tag")
    ap.add_argument("--pdvd", default="/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd")
    ap.add_argument("--out", default="/home/xqian/tmp/doc41/ab")
    ap.add_argument("--geom", action="store_true", help="locate the flipped clusters")
    a = ap.parse_args()

    A = read_arm(a.pdvd, a.off_tag)
    B = read_arm(a.pdvd, a.on_tag)
    evs = sorted(set(A) & set(B))
    rep = dict(off_tag=a.off_tag, on_tag=a.on_tag,
               events_off=len(A), events_on=len(B), events_common=len(evs))
    print(f"events: {a.off_tag} {len(A)}, {a.on_tag} {len(B)}, common {len(evs)}")
    if set(A) - set(B) or set(B) - set(A):
        print("  WARNING: event sets differ:", sorted(set(A) ^ set(B))[:10])

    # gate: identical cluster id sets per event
    bad = [e for e in evs if set(A[e]) != set(B[e])]
    rep["events_with_different_cluster_sets"] = len(bad)
    print(f"cluster-id gate: {len(bad)} of {len(evs)} events differ", bad[:5])

    for key, label in (("tgm", "TGM"), ("stm", "STM"), ("fc", "fully-contained")):
        nA = sum(1 for e in evs for c, v in A[e].items() if v[key])
        nB = sum(1 for e in evs for c, v in B[e].items() if v[key])
        nEvalA = sum(1 for e in evs for c, v in A[e].items() if v[key] is not None)
        nEvalB = sum(1 for e in evs for c, v in B[e].items() if v[key] is not None)
        gain = [(e, c) for e in evs for c in set(A[e]) & set(B[e])
                if (not A[e][c][key]) and B[e][c][key]]
        loss = [(e, c) for e in evs for c in set(A[e]) & set(B[e])
                if A[e][c][key] and not B[e][c][key]]
        nev_changed = len({e for e, _ in gain + loss})
        rep[key] = dict(off=nA, on=nB, evaluated_off=nEvalA, evaluated_on=nEvalB,
                        gained=len(gain), lost=len(loss), events_changed=nev_changed,
                        gained_list=[[e[0], e[1], c] for e, c in gain],
                        lost_list=[[e[0], e[1], c] for e, c in loss])
        print(f"{label:16s} off {nA:5d}  on {nB:5d}  delta {nB - nA:+5d}   "
              f"(+{len(gain)} / -{len(loss)}; {nev_changed} of {len(evs)} events changed; "
              f"evaluated {nEvalA}/{nEvalB})")

    if a.geom:
        cats = {}
        for e in evs:
            for c in set(A[e]) & set(B[e]):
                ta, tb = A[e][c]["tgm"], B[e][c]["tgm"]
                cats[(e[0], e[1], c)] = ("tgm_lost" if ta and not tb else
                                         "tgm_gained" if tb and not ta else
                                         "tgm_kept" if ta and tb else "tgm_never")
        rows = geometry(a.pdvd, a.off_tag, cats)
        rep["geometry"] = rows
        import collections
        bins = [0, 3, 5, 8, 12, 15, 18, 25, 40, 1e9]
        print("\n  the deciding (worst) end's distance to the nearest boundary, per category")
        print("  category      n   no_t0  " + " ".join(f"{bins[i]:>4g}-{bins[i+1]:<4g}"
                                                       for i in range(len(bins) - 1)).replace("1e+09", "inf"))
        for cat in ("tgm_kept", "tgm_lost", "tgm_gained", "tgm_never"):
            R = [r for r in rows if r["cat"] == cat]
            nt0 = sum(1 for r in R if r["no_t0"])
            h = [0] * (len(bins) - 1)
            for r in R:
                if r["no_t0"]:
                    continue
                for i in range(len(bins) - 1):
                    if bins[i] <= r["worst_end_cm"] < bins[i + 1]:
                        h[i] += 1; break
            print(f"  {cat:12s} {len(R):5d} {nt0:6d}  " + " ".join(f"{v:9d}" for v in h))
        print("\n  tgm_lost by wall of the deciding end:")
        hist = collections.Counter((r["wall"], r.get("half")) for r in rows if r["cat"] == "tgm_lost")
        for k, v in sorted(hist.items(), key=lambda kv: -kv[1]):
            print(f"    {str(k):28s} {v}")

    with open(a.out + "_verdicts.json", "w") as f:
        json.dump(rep, f, indent=1)
    print("wrote", a.out + "_verdicts.json")


if __name__ == "__main__":
    main()
