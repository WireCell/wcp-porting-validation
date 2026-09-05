#!/usr/bin/env python3
"""Doc pdvd/43 -- grade the fiducial arms on the existing tests, side by side.

Arms are PR-stage tags on the same 99 pctrees (identical cluster sets asserted).
Tests:
  1. TGM / STM / FC counts per arm, and the flip sets against the reference arm
     (lost / gained), split by cluster length;
  2. the labelled cases: the doc 41 sec 12 Bee scan set (figs/41_bee_longloss_manifest.json)
     -- the owner's scan called indices 1, 4, 5, 6 through-going -- TGM per arm;
  3. the doc 41 sec 11 classes of the long (> 2 m) TGM losses of the d50 arm
     (figs/41_longloss_ends.json: readout-clipped / wall-parallel / stops-short ...)
     -- how many of each class every arm tags.

Repro:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 docs/nf_sp_img_clus/scripts/fv_quantile_grade.py d43fvoff d43fvd50 d43p80c3 d43p90c3 d43p90c5 \
      --old d41fvoff d41fvon --out /home/xqian/tmp/doc43/grade
"""
import argparse, json, os, sys
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_curved_ab import read_arm

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+"); ap.add_argument("--old", nargs="*", default=[])
    ap.add_argument("--pdvd", default="/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd")
    ap.add_argument("--geom", default="/home/xqian/tmp/doc41/ab_verdicts.json", help="per-cluster length (doc 41 sec 10 geometry)")
    ap.add_argument("--out", default="/home/xqian/tmp/doc43/grade")
    a = ap.parse_args()

    arms = {t: read_arm(a.pdvd, t) for t in a.tags + a.old}
    ref = a.tags[0]
    events = sorted(arms[ref])
    for t, arm in arms.items():
        assert sorted(arm) == events, (t, len(arm), len(events))
        for ev in events:
            assert set(arm[ev]) == set(arms[ref][ev]), (t, ev)
    print(f"{len(events)} events, {sum(len(arms[ref][e]) for e in events)} clusters, identical id sets in every arm\n")

    G = json.load(open(a.geom))["geometry"]
    length = {(r["run"], r["idx"], r["cid"]): r.get("len_cm", -1) for r in G}
    lbins = [(0, 50), (50, 100), (100, 200), (200, 1e9)]

    out = {"events": len(events), "arms": {}}
    print(f"{'arm':10s} {'TGM':>5s} {'STM':>5s} {'FC':>5s} | vs {ref}: {'TGM lost':>8s} {'gained':>6s} {'STM lost':>8s} {'gained':>6s} | TGM by length "
          + " ".join(f"{lo:.0f}-{hi if hi < 1e8 else 'inf':>3}" for lo, hi in lbins))
    for t in a.tags + a.old:
        arm = arms[t]
        cnt = Counter(); lost = Counter(); gained = Counter(); bylen = Counter()
        for ev in events:
            for cid, v in arm[ev].items():
                for k in ("tgm", "stm", "fc"):
                    if v[k]:
                        cnt[k] += 1
                    r = arms[ref][ev][cid][k]
                    if r and not v[k]:
                        lost[k] += 1
                    if v[k] and not r:
                        gained[k] += 1
                if v["tgm"]:
                    L = length.get((ev[0], ev[1], cid), -1)
                    for lo, hi in lbins:
                        if lo <= L < hi:
                            bylen[(lo, hi)] += 1
        out["arms"][t] = dict(tgm=cnt["tgm"], stm=cnt["stm"], fc=cnt["fc"],
                              tgm_lost=lost["tgm"], tgm_gained=gained["tgm"], stm_lost=lost["stm"], stm_gained=gained["stm"],
                              tgm_by_length={f"{lo:.0f}-{hi:.0f}": bylen[(lo, hi)] for lo, hi in lbins})
        print(f"{t:10s} {cnt['tgm']:5d} {cnt['stm']:5d} {cnt['fc']:5d} | {'':>3}{'':10s} {lost['tgm']:8d} {gained['tgm']:6d} {lost['stm']:8d} {gained['stm']:6d} | "
              + " ".join(f"{bylen[(lo, hi)]:6d}" for lo, hi in lbins))

    # 2. the labelled cases
    man = json.load(open(os.path.join(DOCS, "figs", "41_bee_longloss_manifest.json")))
    print("\nThe doc 41 sec 12 Bee scan set (owner: 1, 4, 5, 6 look like TGM).  TGM/STM per arm:")
    print(f"{'bee':>3s} {'run':>6s} {'idx':>3s} {'cid':>4s} {'len':>6s} {'wall':>5s} " + " ".join(f"{t:>10s}" for t in a.tags + a.old))
    out["bee"] = []
    for m in man:
        ev = (m["run"], m["work_idx"]); cid = m["cluster"]
        cells = []
        rec = dict(bee=m["bee_index"], run=m["run"], idx=m["work_idx"], cid=cid, verdict={})
        for t in a.tags + a.old:
            v = arms[t][ev][cid]
            s = "TGM" if v["tgm"] else ("STM" if v["stm"] else ("FC" if v["fc"] else "-"))
            rec["verdict"][t] = s; cells.append(f"{s:>10s}")
        out["bee"].append(rec)
        print(f"{m['bee_index']:3d} {m['run']:>6s} {m['work_idx']:>3s} {cid:4d} {m['length_cm']:6.0f} {m['wall']:>5s} " + " ".join(cells))

    # 3. the doc 41 sec 11 classes of the 140 long d50-arm losses
    p = os.path.join(DOCS, "figs", "41_longloss_ends.json")
    if os.path.exists(p):
        LL = json.load(open(p))
        recs = LL["lost"] if isinstance(LL, dict) and "lost" in LL else LL
        if isinstance(recs, dict):
            recs = list(recs.values())
        # the doc 41 sec 11.4 classes, mutually exclusive in priority order, from the
        # per-cluster flags of fv_curved_longloss.py (the d41fvoff -> d41fvon losses)
        def klass(r):
            if r["d_late_cm"] < 5 or (r["d_early_cm"] < 5 and abs(r["x_cm"]) < 330):
                return "readout-clipped"
            if r["cos_to_wall"] < 0.2:
                return "runs along the wall"
            if r["n_other_beyond"] > 0:
                return "charge beyond (attribution)"
            return "points at the wall, stops short"
        cls = defaultdict(list)
        for r in recs:
            if not isinstance(r, dict) or "cid" not in r or r.get("cat") != "tgm_lost":
                continue
            cls[klass(r)].append(((r["run"], r["idx"]), r["cid"]))
        if cls:
            print("\nThe 140 long TGM losses of the d50 arm by doc 41 sec 11 class: TGM-tagged per arm")
            print(f"{'class':32s} {'n':>4s} " + " ".join(f"{t:>13s}" for t in a.tags + a.old))
            out["longloss_classes"] = {}
            for c, lst in sorted(cls.items(), key=lambda kv: -len(kv[1])):
                cells = []
                out["longloss_classes"][c] = {}
                for t in a.tags + a.old:
                    n = sum(1 for ev, cid in lst if arms[t].get(ev, {}).get(cid, {}).get("tgm"))
                    ns = sum(1 for ev, cid in lst if arms[t].get(ev, {}).get(cid, {}).get("stm"))
                    out["longloss_classes"][c][t] = dict(tgm=n, stm=ns); cells.append(f"{n:4d} (stm {ns:2d})")
                print(f"{str(c):32s} {len(lst):4d} " + " ".join(cells))
    json.dump(out, open(a.out + ".json", "w"), indent=1)
    print("\nwrote", a.out + ".json")


if __name__ == "__main__":
    main()
