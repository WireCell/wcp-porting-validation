#!/usr/bin/env python3
"""doc pr/124 front B -- seed-time re-classification census.

Parses the pr/122 SHOWER_SEED probe (in_main_cluster BFS seeder,
NeutrinoShowerClustering.cxx:735, env WCT_SHOWER_ABSORB_DEBUG):

  SHOWER_SEED site=in_main_cluster seg=N gidx=G pdg=P traj=T topo=T pdg11=T
      long_muon=L len_cm=X med_dqdx_mip=Y straight=S

and joins each ACCEPTED seed to the labels' fake-root signal: a labels
marks_by_shower entry keyed by the seed segment with the seed itself marked
'out' of its own shower means the scanner judged the ROOT wrong (the
54332-16014 class).  A seed whose entry (or membership) carries IN marks is a
good root.  Everything else is unlabeled.

Question (owner, pr/124): does any seed-time feature -- which disjunct fired
(traj/topo/pdg11), length, median dQ/dx, straightness -- separate fake roots
from good roots with zero collateral, i.e. can a seeder-side re-validation
(or an existing knob: shower_topo_dqdx_guard / shower_topo_demote_len /
shower_traj_straight_guard) kill the 489327/69232/54332-16014 class?

Repro:
  ./scripts/pr124_seed_census.py 'work-pr124r1-dbg-*' --tsv docs/pr/pr124-seed-census.tsv
"""
import argparse
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)
LABEL_DIRS = [os.path.join(SX, "em_labels", t)
              for t in ("emscan-0827", "emscan-0828-agent5")]
FOCUS = {489327, 69232, 54332, 171143, 277298}

RE_SEED = re.compile(
    r"SHOWER_SEED site=in_main_cluster seg=(-?\d+) gidx=(\d+) pdg=(-?\d+) "
    r"traj=(\d) topo=(\d) pdg11=(\d) long_muon=(\d) len_cm=([\d.eE+-]+) "
    r"med_dqdx_mip=([\d.eE+-]+) straight=(\d)")


def load_em(ev):
    for ld in LABEL_DIRS:
        p = os.path.join(ld, "labels-evt%d.json" % ev)
        if os.path.exists(p):
            j = json.load(open(p))
            return (j.get("em") or {}), (j.get("note") or "")
    return None, ""


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
            seen_ev.add(ev)
            em, note = load_em(ev)
            marks = (em or {}).get("marks_by_shower") or {}
            for line in open(log, errors="replace"):
                m = RE_SEED.search(line)
                if not m:
                    continue
                seg = int(m.group(1))
                verdict = "UNL"
                if em is not None:
                    mm = marks.get(str(seg))
                    if mm is not None:
                        v_self = mm.get(str(seg))
                        n_in = sum(1 for v in mm.values() if v == "in")
                        if v_self == "out":
                            verdict = "FAKE_ROOT"
                        elif v_self == "in" or n_in:
                            verdict = "GOOD_ROOT"
                        else:
                            verdict = "MARKED_OTHER"
                    elif not marks:
                        verdict = "NOLAB_EVT" if em == {} else "UNL"
                rows.append(dict(
                    ev=ev, seg=seg, pdg=int(m.group(3)), traj=int(m.group(4)),
                    topo=int(m.group(5)), pdg11=int(m.group(6)),
                    long_muon=int(m.group(7)), len_cm=float(m.group(8)),
                    med_dqdx_mip=float(m.group(9)), straight=int(m.group(10)),
                    verdict=verdict, note=note.replace("\t", " ")[:80]))

    cols = ["ev", "seg", "pdg", "traj", "topo", "pdg11", "long_muon",
            "len_cm", "med_dqdx_mip", "straight", "verdict", "note"]
    if args.tsv:
        with open(args.tsv, "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")
        print("wrote %d seed rows -> %s" % (len(rows), args.tsv))

    print("\naccepted seeds by disjunct x verdict:")
    print("  %-16s %-10s %5s" % ("disjunct", "verdict", "n"))
    from collections import Counter
    cnt = Counter()
    for r in rows:
        dis = "+".join(k for k in ("traj", "topo", "pdg11") if r[k])
        cnt[(dis or "none", r["verdict"])] += 1
    for (dis, v), n in sorted(cnt.items()):
        print("  %-16s %-10s %5d" % (dis, v, n))

    print("\nFAKE_ROOT seeds (all) and FOCUS-event seeds:")
    for r in rows:
        if r["verdict"] == "FAKE_ROOT" or r["ev"] in FOCUS:
            print("  evt%-7d seg=%-7d %-10s pdg=%-6d traj/topo/pdg11=%d%d%d "
                  "len=%-7.2f mdqdx=%-6.3f straight=%d  %s"
                  % (r["ev"], r["seg"], r["verdict"], r["pdg"], r["traj"],
                     r["topo"], r["pdg11"], r["len_cm"], r["med_dqdx_mip"],
                     r["straight"], r["note"]))

    print("\nfeature quantiles by verdict (len_cm / med_dqdx_mip):")
    import statistics
    for v in ("FAKE_ROOT", "GOOD_ROOT"):
        sel = [r for r in rows if r["verdict"] == v]
        if not sel:
            continue
        ls = sorted(r["len_cm"] for r in sel)
        ds = sorted(r["med_dqdx_mip"] for r in sel)
        print("  %-10s n=%-4d len med=%.1f p90=%.1f  mdqdx med=%.2f p10=%.2f"
              % (v, len(sel), statistics.median(ls), ls[int(0.9 * (len(ls) - 1))],
                 statistics.median(ds), ds[int(0.1 * (len(ds) - 1))]))


if __name__ == "__main__":
    main()
