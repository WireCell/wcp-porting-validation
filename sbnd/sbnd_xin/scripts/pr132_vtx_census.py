#!/usr/bin/env python3
"""doc pr/132 round 5 (Phase A) -- the wrong-vertex population census.

New script (borrows vtx_io label loading like pr90_movers.py; not a fork of a
census: this one classifies the MAIN-VERTEX miss mechanism, not pi0 pairing).
For every vtx105-labeled event with a dump in the given arms, classify:

  CORRECT      |click - main| <= 1.0 cm      (the pr/78/79 bar)
  NEAR         1.0 < d <= 4.0 cm             (scan min_accept was 4.0)
  WRONG+CAND   d > 4.0 and a candidate vertex sits within 2 cm of the click
               -> a RANKING failure (the right answer existed and lost)
  WRONG+NOCAND d > 4.0 and no vertex within 2 cm of the click
               -> a GENERATION failure (the right answer never existed)

For WRONG+CAND also report whether the winning main sits in a DIFFERENT
cluster than the click-nearest candidate (the 76346 cross-cluster shape) and
whether that candidate carried main_candidate=True.
"""
import json, math, os, sys, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vtx_rules import vtx_io

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="arm dirs, e.g. work-pr132-r4off-mcp2k")
    ap.add_argument("--tsv")
    args = ap.parse_args()

    truth = {}
    for doc in vtx_io.load_labels(tags=vtx_io.TAGS_VTX105):
        ev = doc.get("eventNo")
        if doc["truth"] is None: continue
        truth.setdefault(ev, doc["truth"])

    rows = []
    for arm in args.arms:
        for ev in sorted(truth):
            p = os.path.join(arm, "pr_evt%d" % ev, "calib-pr-evt%d.json" % ev)
            if not os.path.exists(p): continue
            d = json.load(open(p))
            mv = d["main_vertex"]; click = truth[ev]
            dm = math.dist((mv["x"], mv["y"], mv["z"]), click)
            best = None
            for v in d.get("vertices") or []:
                f = v.get("fit")
                if not f: continue
                dd = math.dist((f["x"], f["y"], f["z"]), click)
                if best is None or dd < best[0]:
                    best = (dd, v)
            dc, cv = best if best else (None, None)
            if dm <= 1.0: klass = "CORRECT"
            elif dm <= 4.0: klass = "NEAR"
            elif dc is not None and dc <= 2.0: klass = "WRONG+CAND"
            else: klass = "WRONG+NOCAND"
            xclu = (klass == "WRONG+CAND" and cv["cluster_id"] != mv["cluster_id"])
            rows.append(dict(arm=arm.split("-")[-1], event=ev, klass=klass,
                             d_main=round(dm, 2),
                             d_cand=round(dc, 2) if dc is not None else -1,
                             cand_vid=cv["id"] if cv else -1,
                             cand_is_maincand=int(bool(cv and cv.get("main_candidate"))) if cv else -1,
                             cross_cluster=int(xclu)))
    from collections import Counter
    cnt = Counter(r["klass"] for r in rows)
    n = len(rows)
    print("=== wrong-vertex census (vtx105 clicks, %d labeled events with dumps) ===" % n)
    for k in ("CORRECT", "NEAR", "WRONG+CAND", "WRONG+NOCAND"):
        print("  %-13s %3d  %.1f%%" % (k, cnt[k], 100.0 * cnt[k] / n if n else 0))
    print("\n=== the WRONG population ===")
    for r in rows:
        if r["klass"].startswith("WRONG"):
            print("  %s %-7d %-13s d_main=%6.1f d_cand=%5.1f cand_vid=%-6d maincand=%d cross_cluster=%d"
                  % (r["arm"], r["event"], r["klass"], r["d_main"], r["d_cand"],
                     r["cand_vid"], r["cand_is_maincand"], r["cross_cluster"]))
    if args.tsv:
        import csv
        with open(args.tsv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
            w.writeheader(); w.writerows(rows)
        print("\nwrote %s (%d rows)" % (args.tsv, len(rows)))

if __name__ == "__main__":
    main()
