#!/usr/bin/env python3
"""doc pr/124 front C -- pass3 absorber census (fork of pr123_pass4_census.py
in role; parser written fresh for the pass3 probe vocabulary).

pass3 absorbs at three sites (NeutrinoShowerClustering.cxx): pass3_proximity,
pass3_cone (the 20-OUT-mark absorber on the 141-set), pass3_cluster_map.
Existing probes under WCT_SHOWER_ABSORB_DEBUG:
  SHOWER_ABSORB DIRECT site=pass3_* shower_start_seg=A seg=B pdg=P   (absorb)
  SHOWER_ABSORB P120_P3CONE seg=B ... site_ang= dist_cm= ang15= ang60= ao=
    (pr/120 admission census for the winning cone pair, printed pre-guard)

For every pass3 absorb: label verdict of the absorbed seg (segment-level:
IN-marked anywhere / OUT-marked anywhere / unlabeled), whether the seg is
still owned by the absorbing shower in the FINAL dump (the pr/123 prune and
later passes may have moved it -- the moved share is already handled), and
for cone absorbs the P120 geometry features.  The question (owner, pr/124):
does a feature threshold separate the OUT cone absorbs from IN with zero
collateral, on the CONTIGUOUS share?

Repro:
  ./scripts/pr124_pass3_census.py 'work-pr124r1-dbg-*' --tsv docs/pr/pr124-pass3-census.tsv
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)
LABEL_DIRS = [os.path.join(SX, "em_labels", t)
              for t in ("emscan-0827", "emscan-0828-agent5")]

RE_DIRECT = re.compile(
    r"SHOWER_ABSORB DIRECT site=(pass3_\w+) shower_start_seg=(-?\d+) seg=(-?\d+) pdg=(-?\d+)")
RE_P120 = re.compile(
    r"SHOWER_ABSORB P120_P3CONE seg=(-?\d+) pdg=(-?\d+) len_cm=([\d.eE+-]+) "
    r"shower_start_seg=(-?\d+) site_ang=([\d.eE+-]+) dist_cm=([\d.eE+-]+) "
    r"ang15=([\d.eE+-]+) ang60=([\d.eE+-]+) ao=([\d.eE+-]+)")


def load_marks(ev):
    for ld in LABEL_DIRS:
        p = os.path.join(ld, "labels-evt%d.json" % ev)
        if os.path.exists(p):
            em = json.load(open(p)).get("em") or {}
            marks = em.get("marks_by_shower") or {}
            ins, outs = set(), set()
            for mm in marks.values():
                for s, v in mm.items():
                    (ins if v == "in" else outs).add(int(s))
            return ins, outs
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--tsv")
    args = ap.parse_args()
    roots = [r for g in args.roots for r in sorted(glob.glob(g))]
    rows = []
    seen_ev = set()
    for root in roots:
        for ed in sorted(glob.glob(os.path.join(root, "pr_evt*"))):
            ev = int(os.path.basename(ed)[len("pr_evt"):])
            if ev in seen_ev:
                continue
            log = os.path.join(ed, "stdout.log")
            if not os.path.exists(log):
                continue
            lab = load_marks(ev)
            seen_ev.add(ev)
            p120 = {}   # seg -> latest features
            absorbs = []
            for line in open(log, errors="replace"):
                m = RE_P120.search(line)
                if m:
                    p120[int(m.group(1))] = dict(
                        len_cm=float(m.group(3)), site_ang=float(m.group(5)),
                        dist_cm=float(m.group(6)), ang15=float(m.group(7)),
                        ang60=float(m.group(8)), ao=float(m.group(9)))
                    continue
                m = RE_DIRECT.search(line)
                if m:
                    absorbs.append((m.group(1), int(m.group(2)), int(m.group(3)),
                                    int(m.group(4)), dict(p120.get(int(m.group(3)), {}))))
            if not absorbs:
                continue
            # final ownership from the dump
            fown = {}
            djs = glob.glob(os.path.join(ed, "calib-pr-evt*.json"))
            if djs:
                j = json.load(open(djs[0]))
                fown = {s["id"]: s.get("shower_id") for s in j.get("segments", [])}
            for site, shw, seg, pdg, feat in absorbs:
                verdict = "NOLAB"
                if lab:
                    ins, outs = lab
                    verdict = ("IN" if seg in ins else
                               "OUT" if seg in outs else "UNL")
                final = fown.get(seg)
                rows.append(dict(ev=ev, site=site, shw=shw, seg=seg, pdg=pdg,
                                 verdict=verdict,
                                 final_owner=final if final is not None else -99,
                                 kept=int(final == shw) if final is not None else -1,
                                 **{k: feat.get(k, -1.0) for k in
                                    ("len_cm", "site_ang", "dist_cm", "ang15", "ang60", "ao")}))

    cols = ["ev", "site", "shw", "seg", "pdg", "verdict", "final_owner", "kept",
            "len_cm", "site_ang", "dist_cm", "ang15", "ang60", "ao"]
    if args.tsv:
        with open(args.tsv, "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")
        print("wrote %d absorb rows -> %s" % (len(rows), args.tsv))

    print("\npass3 absorbs by site x verdict (labeled events only):")
    sites = sorted({r["site"] for r in rows})
    print("  %-20s %5s %5s %5s" % ("site", "OUT", "IN", "UNL"))
    for s in sites:
        c = {v: sum(1 for r in rows if r["site"] == s and r["verdict"] == v)
             for v in ("OUT", "IN", "UNL")}
        print("  %-20s %5d %5d %5d" % (s, c["OUT"], c["IN"], c["UNL"]))

    print("\ncontiguous share (kept=1: seg still owned by the absorbing shower):")
    for s in sites:
        for v in ("OUT", "IN"):
            k1 = sum(1 for r in rows if r["site"] == s and r["verdict"] == v and r["kept"] == 1)
            k0 = sum(1 for r in rows if r["site"] == s and r["verdict"] == v and r["kept"] == 0)
            if k1 or k0:
                print("  %-20s %-3s kept=%d moved=%d" % (s, v, k1, k0))

    print("\npass3_cone kept OUT vs IN features:")
    for v in ("OUT", "IN"):
        sel = [r for r in rows if r["site"] == "pass3_cone" and r["verdict"] == v
               and r["kept"] == 1]
        for r in sorted(sel, key=lambda r: -r["dist_cm"]):
            print("  %-3s evt%-7d seg=%-7d pdg=%-6d len=%-7.2f site_ang=%-7.2f "
                  "dist=%-7.2f ang15=%-7.2f ang60=%-7.2f ao=%.1f"
                  % (v, r["ev"], r["seg"], r["pdg"], r["len_cm"], r["site_ang"],
                     r["dist_cm"], r["ang15"], r["ang60"], r["ao"]))


if __name__ == "__main__":
    main()
