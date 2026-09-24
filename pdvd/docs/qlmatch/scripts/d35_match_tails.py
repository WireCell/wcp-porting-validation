#!/usr/bin/env python3
"""doc qlmatch/35 check C3 -- crashes and match-rate tails on the events without a hand-scan record.

Per event of pdvd/stm/events.txt, control arm vs candidate arm calib dumps (no truth involved):
  long   = the scorer's long-cluster filter (d32_scan_items.long_uids), in both dumps;
  matched = a long cluster with >= 1 auto-selected bundle;
  mover  = a long cluster in both dumps whose auto-selected flash-time set differs beyond 1.5 us (the ToT light moves
           a bright railed flash earlier by ~1 us; 1.5 us reproduces the time-mapped count on run 039252, 162 vs 161).
Bars (d35/prereg.md sec 2 C3), applied to the 102 events of runs 039253 + 039349; run 039252 (the scanned run, spent)
is printed as the reference:
  STOP  an event with complete control outputs and incomplete candidate outputs;
  STOP  an event (control >= 10 matched) keeping < 75 % of the control's matched long clusters;
  STOP  pooled 102-event mover fraction > 0.20;
  FLAG  per-event matched change < -10 % or mover fraction > 0.30.

    python3 d35_match_tails.py --ctl q31ctl --cand q34tk1 [--pr] > ../d35/c3_tails_pinned.txt
"""
import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from d32_scan_items import load, long_uids, auto_times   # noqa: E402

PDVD = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
TOL = 1.5


def events():
    for ln in open(os.path.join(PDVD, "stm", "events.txt")):
        f = ln.split()
        if len(f) >= 2 and not ln.startswith("#"):
            yield "%06d_%s" % (int(f[0]), f[1])


def outputs(d, pr):
    ok = bool(glob.glob(d + "/calib-evt*.json")) and bool(glob.glob(d + "/mabc-*.zip"))
    if pr:
        ok = ok and bool(glob.glob(d + "/pctree-evt*.tar.gz"))
    return ok


def pr_state(d, zero_ok=False):
    """'ok' (tracking-pr.root + a CheckSTM line), 'nostm' (tree but no candidate line), 'missing'.
    zero_ok: CheckSTM_Michel's zero-candidate end line ("no STM-tagged main cluster; nothing to reconstruct") also
    counts as a completed CheckSTM -> 'ok0'.  Added after the literal reading fired on 039349_55 (d35 sec 4); the
    literal reading (default) is kept alongside."""
    p = d + "/tracking-pr.root"
    if not (os.path.exists(p) and os.path.getsize(p) > 0):
        return "missing"
    zero = False
    for lg in glob.glob(d + "/wct_pr_*.log"):
        with open(lg, errors="replace") as fh:
            for line in fh:
                if "CheckSTM_Michel: " in line and "candidate(s)" in line:
                    return "ok"
                if "CheckSTM_Michel: no STM-tagged main cluster" in line:
                    zero = True
    return "ok0" if (zero and zero_ok) else "nostm"


def same(a, b):
    a, b = sorted(a), sorted(b)
    return len(a) == len(b) and all(abs(x - y) <= TOL for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctl", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--pr", action="store_true", help="also require pctree + PR outputs (the q35 arms)")
    ap.add_argument("--zero-candidate-ok", action="store_true",
                    help="count CheckSTM_Michel's zero-candidate end line as a completed PR (non-literal reading)")
    a = ap.parse_args()
    W = os.path.join(PDVD, "work")
    stops, flags = [], []
    pool = {"ref": [0, 0, 0, 0, 0], "new": [0, 0, 0, 0, 0]}     # events, long, matched ctl, matched cand, movers
    print(f"# doc qlmatch/35 C3 -- {a.ctl} -> {a.cand}; mover tolerance {TOL} us; ref = run 039252 (scanned), "
          f"new = runs 039253 + 039349")
    print("# event            set  long  m_ctl  m_cand  d_match  movers  mov_frac  pr_ctl/pr_cand")
    for e in events():
        dc, dt = f"{W}/{e}_{a.ctl}", f"{W}/{e}_{a.cand}"
        s = "ref" if e.startswith("039252_") else "new"
        oc, ot = outputs(dc, a.pr), outputs(dt, a.pr)
        prs = ""
        if a.pr:
            pc, pt = ((pr_state(dc, a.zero_candidate_ok) if oc else "missing"),
                      (pr_state(dt, a.zero_candidate_ok) if ot else "missing"))
            prs = f"{pc}/{pt}"
            if pc != "missing" and pt == "missing":
                stops.append(f"{e}: control PR {pc}, candidate PR missing")
            elif pc in ("ok", "ok0") and pt == "nostm":
                stops.append(f"{e}: candidate PR has no CheckSTM candidate line, control does")
        if oc and not ot:
            stops.append(f"{e}: control outputs complete, candidate incomplete")
            print(f"{e:<18} {s}  CANDIDATE OUTPUTS MISSING")
            continue
        if not oc:
            print(f"{e:<18} {s}  control outputs missing (not a candidate failure)")
            continue
        A = load(glob.glob(dc + "/calib-evt*.json")[0])
        B = load(glob.glob(dt + "/calib-evt*.json")[0])
        aa, bb = auto_times(A), auto_times(B)
        L = long_uids(A) & long_uids(B)
        mc = sum(1 for u in L if aa.get(u))
        mt = sum(1 for u in L if bb.get(u))
        mv = sum(1 for u in L if not same(aa.get(u, []), bb.get(u, [])))
        dm = (mt - mc) / mc if mc else 0.0
        fr = mv / len(L) if L else 0.0
        p = pool[s]
        for i, v in enumerate((1, len(L), mc, mt, mv)):
            p[i] += v
        mark = ""
        if s == "new":
            if mc >= 10 and mt < 0.75 * mc:
                stops.append(f"{e}: keeps {mt}/{mc} matched long clusters (< 75 %)"); mark = " STOP"
            elif dm < -0.10 or fr > 0.30:
                flags.append(f"{e}: matched {mc}->{mt} ({dm:+.1%}), mover fraction {fr:.3f}"); mark = " FLAG"
        print(f"{e:<18} {s}  {len(L):>4}  {mc:>5}  {mt:>6}  {dm:>+7.1%}  {mv:>6}  {fr:>8.3f}  {prs}{mark}")
    print()
    for s in ("ref", "new"):
        n, nl, mc, mt, mv = pool[s]
        print(f"POOLED {s}: events {n}  long {nl}  matched {mc} -> {mt} ({(mt - mc) / max(mc, 1):+.2%})  "
              f"movers {mv}  fraction {mv / max(nl, 1):.3f}")
    n, nl, mc, mt, mv = pool["new"]
    if mv / max(nl, 1) > 0.20:
        stops.append(f"pooled 102-event mover fraction {mv / nl:.3f} > 0.20")
    print(f"\nFLAGS ({len(flags)}):")
    for f in flags:
        print("  " + f)
    print(f"STOPS ({len(stops)}):")
    for f in stops:
        print("  " + f)
    print(f"VERDICT C3 {a.ctl} -> {a.cand}: {'PASS' if not stops else 'STOP'}")


if __name__ == "__main__":
    main()
