#!/usr/bin/env python3
"""doc qlmatch/34 step E -- selection on odd events and the owner's non-inferiority margin on even events
(rules: d34/prereg.md).

Inputs are the outputs of d33_paired_score.py (run on the UNION truth = round-2 target + round-3 top-up, and again on
the top-up alone = the fresh part) and the owner+target secondary (ql_agree_score.py --override-truth, scores.json per
arm).

  selection: the ToT ladder rung with the largest ODD net vs q31ctl, provided it exceeds the reference (q33tu) odd
             net; ties -> fewer phantoms on the resolved movers (odd), then the smaller tolerance (ladder order).
  margin   : EVEN half, one-sided 95 % lower bound of net = net - 1.645 sqrt(W+L) >= -MARGIN (6.46 = 2 % of 323), AND
             owner+target phantoms <= q31ctl's.

    python3 d34_margin.py --union ../d34/scan_r3/union/paired_score.txt --fresh ../d34/scan_r3/topup/paired_score.txt \
        --scores-root /home/xqian/tmp/p34/scores_target --ladder q34tk1,q34tk2,q34tk3 --ref q33tu \
        --twins q34tk1=q34ck1,q34tk2=q34ck2,q34tk3=q34ck3
"""
import argparse
import json
import math
import os

MARGIN = 6.46
Z = 1.645


def parse(p):
    """paired_score.txt -> {(arm, half): dict(n, wins, losses, net, p)}, {(arm, half): (agree, phantom, missed)}"""
    prim, sec = {}, {}
    for ln in open(p):
        x = ln.split()
        if len(x) == 9 and x[1] in ("odd", "even", "all"):
            prim[(x[0], x[1])] = dict(n=int(x[2]), wins=int(x[5]), losses=int(x[6]), net=int(x[7]), p=float(x[8]))
        elif ln.strip().endswith(tuple("0123456789")) and ":" in ln and "odd" in ln:
            arm, rest = ln.split(": ", 1)
            y = rest.split()
            for h, v in zip(y[0::2], y[1::2]):
                sec[(arm.strip(), h)] = tuple(int(t) for t in v.split("/"))
    return prim, sec


def bound(r):
    return r["net"] - Z * math.sqrt(r["wins"] + r["losses"])


def total(root, arm):
    d = json.load(open(os.path.join(root, arm, "scores.json")))["total"]
    return d["agree"], d["phantom"], d["missed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union", required=True)
    ap.add_argument("--fresh", required=True)
    ap.add_argument("--scores-root", required=True)
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--ref", default="q33tu")
    ap.add_argument("--ctl", default="q31ctl")
    ap.add_argument("--twins", default="")
    a = ap.parse_args()
    U, Us = parse(a.union)
    F, _ = parse(a.fresh)
    ladder = a.ladder.split(",")
    twins = dict(x.split("=") for x in a.twins.split(",") if x)
    out = []
    P = out.append
    P(f"# doc 34 margin read-out (union: {a.union}; fresh: {a.fresh})")
    P(f"{'arm':>8} {'half':>5} {'n':>4} {'W':>4} {'L':>4} {'net':>4} {'p':>6} {'bound95':>8}")
    for arm in [a.ref] + ladder + [twins[x] for x in ladder if x in twins]:
        for h in ("odd", "even", "all"):
            r = U.get((arm, h))
            if r:
                P(f"{arm:>8} {h:>5} {r['n']:>4} {r['wins']:>4} {r['losses']:>4} {r['net']:>+4} {r['p']:>6.3f} "
                  f"{bound(r):>+8.2f}")
    P("\n# fresh part (top-up movers only)")
    for arm in [a.ref] + ladder:
        for h in ("odd", "even"):
            r = F.get((arm, h))
            if r:
                P(f"{arm:>8} {h:>5} {r['n']:>4} {r['wins']:>4} {r['losses']:>4} {r['net']:>+4} {r['p']:>6.3f} "
                  f"{bound(r):>+8.2f}")
    # ---- selection on odd ----
    ref_odd = U[(a.ref, "odd")]["net"]
    ph = lambda arm: Us.get((arm, "odd"), (0, 10 ** 9, 0))[1]
    ranked = sorted(ladder, key=lambda arm: (-U[(arm, "odd")]["net"], ph(arm), ladder.index(arm)))
    best = ranked[0]
    nets = ", ".join(f"{x} {U[(x, 'odd')]['net']:+d}" for x in ladder)
    P(f"\n# SELECTION (odd): nets {nets}; reference {a.ref} {ref_odd:+d}")
    if U[(best, "odd")]["net"] > ref_odd:
        cand = best
        P(f"   selected {cand} (odd net {U[(cand, 'odd')]['net']:+d} > {a.ref} {ref_odd:+d})")
    else:
        cand = a.ref
        P(f"   no rung exceeds {a.ref} on odd events -> NO IMPROVEMENT OVER L3; candidate stays {a.ref} "
          f"(its margin pass is post hoc)")
    # ---- margin on even ----
    r = U[(cand, "even")]
    b = bound(r)
    tc, tk = total(a.scores_root, a.ctl), total(a.scores_root, cand)
    ok1, ok2 = b >= -MARGIN, tk[1] <= tc[1]
    P(f"\n# MARGIN ({cand}, even half): net {r['net']:+d} ({r['wins']}-{r['losses']}), bound {b:+.2f} vs -{MARGIN} "
      f"-> {'OK' if ok1 else 'FAIL'}")
    P(f"   owner+target phantoms {cand} {tk[1]} vs {a.ctl} {tc[1]} -> {'OK' if ok2 else 'FAIL'}   "
      f"(agree/phantom/missed {tk} vs {tc})")
    P(f"   superiority (even, two-sided sign test): p = {r['p']:.3f}")
    if cand in twins:
        t = U.get((twins[cand], "even"))
        if t:
            P(f"   twin {twins[cand]} even net {t['net']:+d} ({t['wins']}-{t['losses']}) -> "
              f"{'gain NOT ToT-specific' if t['net'] >= r['net'] else 'gain ToT-specific'}")
    P(f"\n# VERDICT: {cand} {'PASSES' if ok1 and ok2 else 'FAILS'} the owner's margin"
      f"{' (post hoc: the reference was not improved on)' if cand == a.ref else ''}")
    for arm in [a.ctl, a.ref] + ladder + [twins[x] for x in ladder if x in twins]:
        try:
            P(f"   owner+target {arm:>8}: {' / '.join(map(str, total(a.scores_root, arm)))}")
        except FileNotFoundError:
            pass
    print("\n".join(out))


if __name__ == "__main__":
    main()
