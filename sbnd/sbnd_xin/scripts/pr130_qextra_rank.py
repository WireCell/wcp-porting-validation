#!/usr/bin/env python3
"""doc pr/130 -- rank the SAME scored rows by q_extra, and split BOTH error
sides into the part a scanner affirmatively judged and the part nobody judged.

Why this exists.  Item 1 (scripts/pr130_qmiss_rank.py) asked "is a q_miss
hand-scan worth a scanner's time", answered GO on concentration, and found
the premise fails out of sample: on the 141-set q_extra (2.514e7) is the
LARGER half.  That raises a question item 1 did not ask -- is the other half
concentrated too, and is it even a physics quantity?

The second half of that question is the one that matters, because the two
sides are NOT symmetric in em117_score.py:

    target = (members | ins) - outs      # members = shower membership AT SCAN TIME
    miss   = target - have               # reco dropped it
    extra  = have  - target              # reco holds it

`miss` is anchored on things the scanner saw.  `extra` is a COMPLEMENT: a
segment lands in it either because the scanner marked it `out` (an
affirmative over-clustering complaint) or merely because it was not in the
shower when the scanner looked -- which is also true of every segment a
LATER, CORRECT merge added.  Ranking on raw q_extra would therefore reward
this campaign's own shipped merges as if they were errors.

So this script reports the decomposition, and holds q_miss to the matching
standard (`miss` that carries an explicit `in` mark) so the comparison is
like-for-like rather than strict on one side only.

Reads item 1's two score tables, the label store (read-only, M13) and the
calib dumps.  No arms, no knobs, no re-scoring, nothing shipped.

Repro:
  cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
  scripts/pr130_qextra_rank.py > docs/pr/pr130-qextra-rank.txt
"""
import collections
import csv
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)
EMD = os.path.join(SX, "em_display")

# Owner-ruled rows leave the pool -- identical list to pr130_qmiss_rank.py so
# the two rankings stay strictly comparable.
ADJUDICATED = {318769: "pr/129 owner reject",
               415278: "pr/124 declined trade-off",
               283515: "pr/130 Part 4 -- owner 'ON better'",
               179369: "pr/130 Part 4 -- owner 'OFF better'"}

SETS = [("141", "em114c-pr130q141-manifest.tsv", "emscan-0828-agent5",
         "docs/pr/pr130-141-score-prod.tsv", "emprep-pr130q141"),
        ("98", "em117-pr130q98-manifest.tsv", "emscan-0827",
         "docs/pr/pr130-98-score-prod.tsv", "emprep-pr130q98")]


def load_scorer():
    """Reuse em117_score.py's own dump digest -- re-deriving segment charge by
    hand is how you get a table that disagrees with the one it is ranking."""
    cwd = os.getcwd()
    os.chdir(EMD)
    try:
        spec = importlib.util.spec_from_file_location("s117", "em117_score.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        os.chdir(cwd)


def habit(tag, labs):
    """Marking habit of one scan.  The two label sets were produced by
    different scans and mark very differently, so the CROSS-SET comparison of
    the affirmative split is partly a statement about scanners, not about the
    reconstruction.  Printed so no reader can miss it."""
    shw = ins = outs = 0
    for rec in labs.values():
        for det in ((rec.get("em") or {}).get("marks_detail") or {}).values():
            m = det.get("marked") or {}
            if not m:
                continue
            shw += 1
            ins += sum(1 for v in m.values() if v.get("kind") == "in")
            outs += sum(1 for v in m.values() if v.get("kind") == "out")
    frac = 100.0 * outs / (ins + outs) if ins + outs else 0.0
    print("  MARKING HABIT of %s: %d marked showers, IN=%d OUT=%d -> %.0f%% of marks are OUT"
          % (tag, shw, ins, outs, frac))
    return frac


def marks(labs):
    """(event, scanned shower) -> (explicit IN ids, explicit OUT ids)."""
    ins, outs = collections.defaultdict(set), collections.defaultdict(set)
    for ev, rec in labs.items():
        for shw, det in ((rec.get("em") or {}).get("marks_detail") or {}).items():
            m = det.get("marked") or {}
            ins[(ev, int(shw))] = {int(s) for s, v in m.items() if v.get("kind") == "in"}
            outs[(ev, int(shw))] = {int(s): v for s, v in m.items() if v.get("kind") == "out"}
    return ins, outs


def main():
    s117 = load_scorer()
    for tag, manf, labtag, tsv, prepdir in SETS:
        cwd = os.getcwd()
        os.chdir(EMD)
        try:
            man = s117.load_manifest(manf)
            labs = s117.load_labels(labtag)
            habit(labtag, labs)
            rows = list(csv.DictReader(open(os.path.join(SX, tsv)), delimiter="\t"))
            ins_by, outs_by = marks(labs)
            MA = MC = XA = XC = 0.0
            nMA = nMC = nXA = nXC = 0
            aff_ev = collections.defaultdict(float)
            by_absorber = collections.defaultdict(lambda: [0.0, 0])
            detail = []
            for r in rows:
                ev = int(r["event"])
                if ev in ADJUDICATED:
                    continue
                shw = int(r["shower"])
                dump = s117.load_dump(man[ev]["dump"])
                if dump is None:
                    continue
                _, seginfo, _ = s117.digest_dump(dump, s117.load_prep(ev, prepdir))
                Q = lambda s: seginfo.get(s, {}).get("charge", 0.0)
                ins, outs = ins_by.get((ev, shw), set()), outs_by.get((ev, shw), {})
                for s in [int(x) for x in (r["miss"] or "").split(",") if x.strip()]:
                    if s in ins:
                        MA += Q(s); nMA += 1
                    else:
                        MC += Q(s); nMC += 1
                aff = []
                for s in [int(x) for x in (r["extra"] or "").split(",") if x.strip()]:
                    if s in outs:
                        XA += Q(s); nXA += 1; aff.append(s); aff_ev[ev] += Q(s)
                        # The label store records WHICH absorber placed the
                        # segment -- the mechanism is on disk, not a guess.
                        who = outs[s].get("absorbed_by") or "(none - shower own root/extent)"
                        by_absorber[who][0] += Q(s); by_absorber[who][1] += 1
                    else:
                        XC += Q(s); nXC += 1
                if aff:
                    shw_cl = seginfo.get(shw, {}).get("cluster")
                    detail.append((ev, shw, shw_cl, sum(Q(s) for s in aff),
                                   [(s, seginfo.get(s, {}), outs[s]) for s in aff]))
        finally:
            os.chdir(cwd)

        print("=" * 78)
        print("%s-set, kept pool (%d adjudicated events removed)" % (tag, len(ADJUDICATED)))
        print("=" * 78)
        print("  BOTH SIDES HELD TO THE SAME AFFIRMATIVE STANDARD")
        print("    q_miss  affirmative (explicit IN  mark, reco dropped it) : %.3e  %3d segs" % (MA, nMA))
        print("    q_miss  weak        (scan-time member, reco dropped it)  : %.3e  %3d segs" % (MC, nMC))
        print("    q_extra affirmative (explicit OUT mark, reco still holds): %.3e  %3d segs" % (XA, nXA))
        print("    q_extra weak        (never judged, absent at scan time)  : %.3e  %3d segs" % (XC, nXC))
        if MA + XA:
            print("    --> affirmative-only split:  q_miss %.1f%%   q_extra %.1f%%"
                  % (100 * MA / (MA + XA), 100 * XA / (MA + XA)))
        tot = sum(aff_ev.values())
        if tot:
            rank = sorted(aff_ev.items(), key=lambda kv: -kv[1])
            # NOT "top-10 holds 100%" -- there are only ~10 events with any
            # affirmative q_extra, so that figure is true by construction.
            print("\n  AFFIRMATIVE q_extra: %.3e over %d events; top-4 holds %.1f%%"
                  % (tot, len(aff_ev), 100 * sum(v for _, v in rank[:4]) / tot))
            same = cross = 0
            q_same = q_cross = 0.0
            print("  %8s %8s %7s %10s %5s  %s"
                  % ("event", "shower", "shw_cl", "q_aff", "nseg", "condemned segment(cluster, len, pdg)"))
            for ev, shw, shw_cl, q, segs in sorted(detail, key=lambda d: -d[3]):
                txt = []
                for s, i, mk in segs:
                    txt.append("%d(cl%s,%.0fcm,pdg%s,%s,d=%.0f,a=%.0f)"
                               % (s, i.get("cluster"), i.get("length", 0.0), i.get("pdg"),
                                  mk.get("absorbed_by") or "own-root",
                                  float(mk.get("dist") or 0.0), float(mk.get("angle") or 0.0)))
                    if i.get("cluster") == shw_cl:
                        same += 1; q_same += i.get("charge", 0.0)
                    else:
                        cross += 1; q_cross += i.get("charge", 0.0)
                print("  %8d %8d %7s %10.3e %5d  %s" % (ev, shw, shw_cl, q, len(segs), "  ".join(txt)))
            qt = q_same + q_cross
            print("\n  condemned segment sits in the SHOWER'S OWN cluster: %d segs / %.3e (%.0f%% of charge)"
                  % (same, q_same, 100 * q_same / qt if qt else 0))
            print("  ... in a DIFFERENT cluster                        : %d segs / %.3e (%.0f%% of charge)"
                  % (cross, q_cross, 100 * q_cross / qt if qt else 0))
            print("\n  WHICH ABSORBER PLACED IT (label store 'absorbed_by' -- measured, not inferred):")
            for who, (q, n) in sorted(by_absorber.items(), key=lambda kv: -kv[1][0]):
                print("    %-34s %.3e  (%4.1f%% of charge, %2d segs)" % (who, q, 100 * q / tot, n))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
