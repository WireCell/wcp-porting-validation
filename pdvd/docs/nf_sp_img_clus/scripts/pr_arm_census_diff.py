#!/usr/bin/env python3
"""doc pdvd/32 round 3b: does a knob flip still work across the PR manifest?

Compares two PR arms event by event straight from what the runner leaves on
disk -- no rerun, no extra instrumentation.  Answers the three questions a
production flip has to answer beyond "the metric I aimed at moved":

  1. did anything stop working (non-zero rc, missing output, a crash)?
  2. did the tagger census move, and where -- per event, not just in total?
  3. did cost move (wall, peak RSS)?

Counts come from the PR log's own verdict lines, so they mean exactly what the
`pr_<tag>_events.tsv` columns mean:

    mains      "flag_mains: ... N main"          clusters promoted to main
    tgm/stm    TGM=true / STM=1 verdicts
    stm_eval   clusters TaggerCheckSTM evaluated (persist_stm_fit records)
    nu_cands   neutrino candidates offered to the tagger

Usage:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 docs/nf_sp_img_clus/scripts/pr_arm_census_diff.py d32p000 d32p035
"""
import argparse
import glob
import os
import re
import sys
from collections import OrderedDict

PDVD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

PATS = OrderedDict([
    ("tgm",      re.compile(r"TGM=true")),
    ("stm",      re.compile(r"STM=1")),
    ("stm_eval", re.compile(r"persist_stm_fit:")),
    ("nu_cands", re.compile(r"neutrino candidate")),
])
RSS = re.compile(r"peak_rss_gb[=\s]+([\d.]+)")
WALL = re.compile(r"wall_s[=\s]+([\d.]+)")


def arm(tag):
    """{(run, idx): {metric: count, ...}} for every event dir of this tag."""
    out = {}
    for d in sorted(glob.glob(os.path.join(PDVD, "work", "*_%s" % tag))):
        base = os.path.basename(d)[: -(len(tag) + 1)]
        try:
            run, idx = base.rsplit("_", 1)
        except ValueError:
            continue
        logs = glob.glob(os.path.join(d, "wct_pr_*.log"))
        if not logs:
            continue
        text = open(logs[0], errors="replace").read()
        rec = {k: len(p.findall(text)) for k, p in PATS.items()}
        rec["zip"] = os.path.exists(os.path.join(d, "mabc-pr.zip"))
        for res in glob.glob(os.path.join(d, "pr_resource_*.txt")):
            body = open(res, errors="replace").read()
            m = WALL.search(body)
            if m:
                rec["wall"] = float(m.group(1))
            m = RSS.search(body)
            if m:
                rec["rss"] = float(m.group(1))
        out[(run, idx)] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag_a")
    ap.add_argument("tag_b")
    ap.add_argument("--show", type=int, default=12, help="events to list per metric")
    args = ap.parse_args()

    A, B = arm(args.tag_a), arm(args.tag_b)
    common = sorted(set(A) & set(B))
    print(f"{args.tag_a}: {len(A)} events   {args.tag_b}: {len(B)} events   common: {len(common)}")
    only = sorted(set(A) ^ set(B))
    if only:
        print(f"  !! present in only one arm: {only[:8]}")
    missing = [e for e in common if not (A[e]["zip"] and B[e]["zip"])]
    print(f"  events missing mabc-pr.zip in either arm: {len(missing)}"
          + (f"  {missing[:6]}" if missing else "  -> nothing stopped working"))

    print("\n  metric      total A    total B      delta   events changed   worst event (delta)")
    for k in PATS:
        ta = sum(A[e][k] for e in common)
        tb = sum(B[e][k] for e in common)
        diffs = [(B[e][k] - A[e][k], e) for e in common if B[e][k] != A[e][k]]
        worst = max(diffs, key=lambda t: abs(t[0])) if diffs else (0, None)
        w = f"{worst[1][0]}/{worst[1][1]} {worst[0]:+d}" if worst[1] else "-"
        print(f"   {k:10s} {ta:8d}   {tb:8d}   {tb-ta:+8d}   {len(diffs):8d}         {w}")

    for k, lbl, unit in (("wall", "wall", "s"), ("rss", "peak RSS", "GB")):
        pairs = [(A[e].get(k), B[e].get(k)) for e in common
                 if A[e].get(k) is not None and B[e].get(k) is not None]
        if not pairs:
            continue
        sa = sum(p[0] for p in pairs)
        sb = sum(p[1] for p in pairs)
        print(f"\n  {lbl}: {sa:.1f} -> {sb:.1f} {unit} over {len(pairs)} events "
              f"({100*(sb-sa)/sa:+.1f} %)")


if __name__ == "__main__":
    main()
