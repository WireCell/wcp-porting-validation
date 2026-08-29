#!/usr/bin/env python3
"""doc pr/127 -- SCCC candidate census from the byte-neutral WCT_SCCC_DEBUG tape.

Owner (2026-08-29, scanning the pr/125 production Bee pair): "there is one
issue 137238, the electron 89 MeV should connect to some thing, which is
missing from the PF tree, should check and fix."

Measured root cause (doc pr/127 sec 2): the pr/93 round-4 fix for THIS event
(straight_cont_cross_cluster, SBND ON since 2026-08-18) stopped firing when
the Q/L era changed cb0805 -> grp0825.  The muon-body candidate that measured
g=5.68cm K=17.0deg then -- and set the production base tier at 6cm/18deg --
now measures g=8.00cm K=14.0deg, outside the base tier's gap and outside the
aligned tier's kink (12cm/7.5deg).  Nothing else about the event changed.

This script reads the stderr tape an arm leaves in pr_evt*/stdout.log:

  SCCC pass:      main_vtx_gidx=.. gap=A/B cm kink=C/D deg bridge=..
  SCCC stem-cand: seg=.. len=..cm traj=.. topo=..        (per main-vertex seg)
  SCCC seg=.. vtx_gidx=.. deg=..                          (per stem endpoint)
  SCCC seg=.. cand=.. g=..cm K=.. k_tan=.. tier_ok=..     (per cross-cluster cand,
                                                           printed when g <= aligned gap)

and reports, per event, every cross-cluster candidate together with the
verdict under the current tiers and under each proposed gate.  Segment ids
print as -1 in this era (PR ids unassigned at this pass), so candidates are
attributed positionally to the preceding stem/vertex lines and identified by
their measured (g, K) -- the ON-arm `sccc demote:` log line carries the
cluster ids for adjudication.

Repro:
  ./scripts/pr127_sccc_census.py work-pr125r1-flipK598-* work-pr125r1-flipK5141-* \
      --tsv docs/pr/pr127-sccc-census.tsv
"""
import argparse
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)

RE_PASS = re.compile(r"SCCC pass: main_vtx_gidx=(\d+) gap=([\d.]+)/([\d.]+)cm kink=([\d.]+)/([\d.]+)deg")
RE_STEM = re.compile(r"SCCC stem-cand: seg=(-?\d+) len=([\d.]+)cm traj=(\d) topo=(\d)")
RE_VTX = re.compile(r"SCCC seg=(-?\d+) vtx_gidx=(\d+) deg=(\d+)")
RE_CAND = re.compile(r"SCCC seg=(-?\d+) cand=(-?\d+) g=([\d.]+)cm K=([\d.]+) k_tan=([\d.]+) tier_ok=(\d)")
RE_DEMOTE = re.compile(r"sccc demote: seg id=(-?\d+) cluster=(-?\d+) len_cm=([\d.]+) pdg (-?\d+) -> (-?\d+) "
                       r"sib id=(-?\d+) sib_cluster=(-?\d+) sib_len_cm=([\d.]+)")

# Proposed gates, evaluated against every candidate the tape recorded.
# (label, list of (max_gap_cm, max_kink_deg) tiers)
GATES = [
    ("prod6_18", [(6.0, 18.0), (12.0, 7.5)]),
    ("gap9", [(9.0, 18.0), (12.0, 7.5)]),
    ("gap10", [(10.0, 18.0), (12.0, 7.5)]),
    ("gap12", [(12.0, 18.0), (12.0, 7.5)]),
    ("mid9_15", [(6.0, 18.0), (9.0, 15.0), (12.0, 7.5)]),
    ("mid10_15", [(6.0, 18.0), (10.0, 15.0), (12.0, 7.5)]),
    # linear taper between the two production tier corners (6,18) -> (12,7.5)
    ("taper", None),
]


def taper_ok(g, k):
    if g <= 6.0:
        return k <= 18.0
    if g > 12.0:
        return False
    return k <= 18.0 - 10.5 * (g - 6.0) / 6.0


def gate_ok(label, tiers, g, k):
    if label == "taper":
        return taper_ok(g, k)
    return any(g <= mg and k <= mk for mg, mk in tiers)


def parse_event(path):
    """-> (pass_seen, [cands], [stems], [demotes]) for one event log."""
    cands, stems, demotes = [], [], []
    pass_seen = None
    cur_stem = None
    with open(path, errors="replace") as fh:
        for line in fh:
            if "SCCC" not in line and "sccc " not in line:
                continue
            m = RE_PASS.search(line)
            if m:
                pass_seen = tuple(float(x) for x in m.groups()[1:])
                continue
            m = RE_STEM.search(line)
            if m:
                cur_stem = (float(m.group(2)), int(m.group(3)), int(m.group(4)))
                stems.append(cur_stem)
                continue
            m = RE_CAND.search(line)
            if m:
                cands.append({
                    "stem_len": cur_stem[0] if cur_stem else -1.0,
                    "g": float(m.group(3)),
                    "K": float(m.group(4)),
                    "k_tan": float(m.group(5)),
                    "tier_ok": int(m.group(6)),
                })
                continue
            m = RE_DEMOTE.search(line)
            if m:
                demotes.append({
                    "cluster": int(m.group(2)), "len": float(m.group(3)),
                    "pdg_old": int(m.group(4)), "pdg_new": int(m.group(5)),
                    "sib_cluster": int(m.group(7)), "sib_len": float(m.group(8)),
                })
    return pass_seen, cands, stems, demotes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="arm roots (globs ok)")
    ap.add_argument("--tsv")
    args = ap.parse_args()

    rows = []
    seen = set()
    for pat in args.roots:
        for root in sorted(glob.glob(os.path.join(SX, pat)) or glob.glob(pat)):
            for log in sorted(glob.glob(os.path.join(root, "pr_evt*", "stdout.log"))):
                ev = int(os.path.basename(os.path.dirname(log))[len("pr_evt"):])
                if ev in seen:
                    continue
                seen.add(ev)
                pass_seen, cands, stems, demotes = parse_event(log)
                for c in cands:
                    c.update(event=ev, arm=os.path.basename(root))
                    rows.append(c)
                for d in demotes:
                    print("FIRE evt %d: demote cluster %d len %.1fcm pdg %d->%d "
                          "sib cluster %d len %.1fcm" %
                          (ev, d["cluster"], d["len"], d["pdg_old"], d["pdg_new"],
                           d["sib_cluster"], d["sib_len"]))

    print("\n%d events scanned, %d cross-cluster candidate(s) recorded "
          "(tape prints g <= aligned gap only)" % (len(seen), len(rows)))

    # Per-gate: how many candidates / events pass that would not pass today.
    base = GATES[0]
    print("\n%-10s %8s %8s %8s %8s" % ("gate", "cands", "events", "new_cand", "new_evt"))
    for label, tiers in GATES:
        ok = [r for r in rows if gate_ok(label, tiers, r["g"], r["K"])]
        ok_base = [r for r in rows if gate_ok(base[0], base[1], r["g"], r["K"])]
        base_evts = {r["event"] for r in ok_base}
        new = [r for r in ok if not gate_ok(base[0], base[1], r["g"], r["K"])]
        print("%-10s %8d %8d %8d %8d" % (label, len(ok), len({r['event'] for r in ok}),
                                         len(new), len({r['event'] for r in new} - base_evts)))

    # The new-admission table (what a widening would buy and cost).
    print("\nCandidates admitted by gap12 but not by the production tiers:")
    print("%-8s %-10s %8s %7s %7s %9s" % ("event", "arm", "stem_cm", "g_cm", "K_deg", "k_tan"))
    for r in sorted(rows, key=lambda r: (r["g"], r["K"])):
        if gate_ok("gap12", [(12.0, 18.0), (12.0, 7.5)], r["g"], r["K"]) and \
           not gate_ok(base[0], base[1], r["g"], r["K"]):
            print("%-8d %-10s %8.1f %7.2f %7.1f %9.1f" %
                  (r["event"], r["arm"], r["stem_len"], r["g"], r["K"], r["k_tan"]))

    if args.tsv:
        out = args.tsv if os.path.isabs(args.tsv) else os.path.join(SX, args.tsv)
        with open(out, "w") as fh:
            fh.write("event\tarm\tstem_len_cm\tg_cm\tK_deg\tk_tan_deg\ttier_ok\t" +
                     "\t".join(l for l, _ in GATES) + "\n")
            for r in sorted(rows, key=lambda r: (r["event"], r["g"])):
                fh.write("%d\t%s\t%.1f\t%.2f\t%.1f\t%.1f\t%d\t%s\n" %
                         (r["event"], r["arm"], r["stem_len"], r["g"], r["K"], r["k_tan"],
                          r["tier_ok"],
                          "\t".join("1" if gate_ok(l, t, r["g"], r["K"]) else "0"
                                    for l, t in GATES)))
        print("\nwrote %s (%d rows)" % (out, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
