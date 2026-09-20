#!/usr/bin/env python3
"""doc sbnd_xin/117: the vertex-chain grid, paired per true interaction against a named base.

doc 116's d116_compare.py compares every cell against ONE base (the doc-115 baseline).  This round
is a 2 x 3 factorial -- trajectory (current / PDHD-PDVD) x vertex chain (2-step dual snapD2 /
1-step exclusion-ON / 1-step exclusion-free) -- of which two corners already exist, so a cell has
TWO relevant bases: the shipping configuration (c2) and its own row corner.  Rather than fork 400
lines, this script IMPORTS d116_compare (unmodified) and reuses its doc-107 selection logic, its
Arm class, its metric() and its exact sign test, so both rounds score with the same code.

What it adds:
  * arbitrary (label -> products path) arms, so c2 = products/d115/{s} and t2 =
    products/d116/{s}-tfull sit beside products/d117/{s}-<cell>;
  * arbitrary pairs, printed as "<cell> vs <base>";
  * the VERTEX THRESHOLD table (1 / 2 / 3 / 5 cm and the median distance), pre-registered in
    117_pred.txt as this round's secondary: a chain can place the vertex better without crossing
    the 5 cm line of M7/M8, and doc 116 sec 16.3 showed that is exactly where the trajectory's
    gain sits.
  * the floor is 0 by construction this round (doc 116 sec 4 measured it; there is no s0rep cell),
    so the rule is |delta| > 0.5 pp AND p < 0.05.

Usage:
  python3 d117_compare.py \
      --arm c2=products/d115/{s} --arm t2=products/d116/{s}-tfull \
      --arm c1e=products/d117/{s}-c1e --arm c1x=products/d117/{s}-c1x \
      --arm t1e=products/d117/{s}-t1e --arm t1x=products/d117/{s}-t1x \
      --pair c1e:c2 --pair c1x:c2 --pair t2:c2 --pair t1e:c2 --pair t1x:c2 \
      --pair t1e:t2 --pair t1x:t2 --pair c1x:c1e --pair t1x:t1e \
      --out docs/117_figs/117_compare
"""
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d116_compare as C  # noqa: E402  (its main() is guarded)

THRESH = [1.0, 2.0, 3.0, 5.0]


def vtx_at(arm, T, cls=None):
    """keys of true in-FV interactions whose nearest candidate vertex is within T cm"""
    out = set()
    for t in arm.fvT:
        if cls and t["cls"] != cls:
            continue
        d = float(t["min_cand_dist_cm"])
        if 0 <= d < T:
            out.add(t["key"])
    return out


def all_keys(arm, cls=None):
    return {t["key"] for t in arm.fvT if not cls or t["cls"] == cls}


def dist_map(arm):
    return {t["key"]: float(t["min_cand_dist_cm"]) for t in arm.fvT}


def verdict(delta, p, floor=0.0):
    lim = max(2 * floor, 0.5)
    if abs(delta) > lim and p < 0.05:
        return "IMPROVED" if delta > 0 else "DEGRADED"
    return "not separable"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", default=[], metavar="LABEL=PATH",
                    help="products path with {s} for the sample; off arms use {s}=off")
    ap.add_argument("--pair", action="append", default=[], metavar="CELL:BASE")
    ap.add_argument("--out", default="docs/117_figs/117_compare")
    a = ap.parse_args()

    spec = dict(x.split("=", 1) for x in a.arm)
    arms, off = {}, {}
    for lab, pat in spec.items():
        arms[lab] = {}
        for s in ("cv", "nuecc"):
            d = pat.format(s=s)
            if os.path.exists(os.path.join(d, "candidates.tsv")):
                arms[lab][s] = C.Arm(d)
        d = pat.format(s="off")
        if os.path.exists(os.path.join(d, "candidates.tsv")):
            off[lab] = C.load_off(d)

    out = []
    p = out.append
    p("# doc 117 -- the vertex-chain grid, paired per true interaction (d117_compare.py)")
    p("# selection logic, Arm and the exact sign test are d116_compare.py's, imported unmodified.")
    p("# floor = 0 (doc 116 sec 4, same pin/box); rule: |delta| > 0.5 pp AND p < 0.05.")
    p("arms: " + "; ".join(f"{k}[{','.join(sorted(v))}{',off' if k in off else ''}]" for k, v in arms.items()))

    # the doc-115 baseline must still reproduce docs/115_sel (V6 of doc 116)
    if "c2" in arms:
        for (s, flav, cut), (k, n, kp, np_) in C.BASELINE_CHECK.items():
            if s not in arms["c2"]:
                continue
            sel, sig, bkg = arms["c2"][s].selected(flav, cut)
            got = (len({x for x in arms["c2"][s].signal_keys(flav + "CC") if x in sig}),
                   len(arms["c2"][s].signal_keys(flav + "CC")), len(sel) - len(bkg), len(sel))
            assert got == (k, n, kp, np_), f"c2 {s} {flav}>{cut}: {got} != {k, n, kp, np_}"
        p("c2 reproduces docs/115_sel (387/557, 387/448; 618/1511, 618/632; 734/1511, 734/781)  OK")

    pairs = [tuple(x.split(":", 1)) for x in a.pair]
    tsv = ["metric\tcell\tbase\tsample\tk_base\tn_base\tk_cell\tn_cell\tdelta_pp\tlost\tgained\tp\tverdict"]

    p("")
    p("## primary metrics (doc 107 definitions, doc 116 M1-M8)")
    p("  id  metric                                            cell   vs base           base           cell"
      "    delta -lost/+gain       p  verdict")
    for mid, s, kind, flav, cut, label in C.PRIMARY:
        for cell, bas in pairs:
            if s not in arms.get(cell, {}) or s not in arms.get(bas, {}):
                continue
            kb, nb, kc, nc, ao, bo, _ = C.metric(arms[bas][s], arms[cell][s], kind, flav, cut)
            d = 100.0 * kc / nc - 100.0 * kb / nb
            pv = C.sign_test(ao, bo)
            v = verdict(d, pv)
            p(f"  {mid:3s} {label:50s} {cell:6s} vs {bas:4s} {kb:5d}/{nb:<5d}={100.0*kb/nb:5.1f}% "
              f"{kc:5d}/{nc:<5d}={100.0*kc/nc:5.1f}% {d:+7.2f} {-ao:4d}/+{bo:<4d} {pv:7.3f}  {v}")
            tsv.append(f"{mid}\t{cell}\t{bas}\t{s}\t{kb}\t{nb}\t{kc}\t{nc}\t{d:.3f}\t{ao}\t{bo}\t{pv:.4g}\t{v}")

    p("")
    p("## the vertex at four thresholds (117_pred.txt secondary; doc 116 sec 16.3 construction)")
    for s in ("cv", "nuecc"):
        p(f"### {s}")
        p("  T(cm)  cell   vs base      base             cell        delta -lost/+gain       p  verdict")
        for T in THRESH:
            for cell, bas in pairs:
                if s not in arms.get(cell, {}) or s not in arms.get(bas, {}):
                    continue
                A, B = arms[bas][s], arms[cell][s]
                keys = all_keys(A) & all_keys(B)
                ma, mb = vtx_at(A, T) & keys, vtx_at(B, T) & keys
                lost, gain = len(ma - mb), len(mb - ma)
                d = 100.0 * (len(mb) - len(ma)) / len(keys)
                pv = C.sign_test(lost, gain)
                v = verdict(d, pv)
                p(f"  {T:5.0f}  {cell:6s} vs {bas:4s} {len(ma):5d}/{len(keys):<5d}={100.0*len(ma)/len(keys):5.1f}% "
                  f"{len(mb):5d}/{len(keys):<5d}={100.0*len(mb)/len(keys):5.1f}% {d:+7.2f} {-lost:4d}/+{gain:<4d} "
                  f"{pv:8.4g}  {v}")
                tsv.append(f"VTX{T:.0f}\t{cell}\t{bas}\t{s}\t{len(ma)}\t{len(keys)}\t{len(mb)}\t{len(keys)}"
                           f"\t{d:.3f}\t{lost}\t{gain}\t{pv:.4g}\t{v}")
        p("  median nearest-candidate distance (cm) on the interactions both arms place within 10 cm:")
        for cell, bas in pairs:
            if s not in arms.get(cell, {}) or s not in arms.get(bas, {}):
                continue
            DA, DB = dist_map(arms[bas][s]), dist_map(arms[cell][s])
            ks = [k for k in set(DA) & set(DB) if 0 <= DA[k] <= 10 and 0 <= DB[k] <= 10]
            if not ks:
                continue
            da = sorted(DA[k] for k in ks)
            db = sorted(DB[k] for k in ks)
            closer = sum(1 for k in ks if DB[k] < DA[k] - 0.01)
            further = sum(1 for k in ks if DB[k] > DA[k] + 0.01)
            pv = min(1.0, 2.0 * sum(math.comb(closer + further, i)
                                    for i in range(min(closer, further) + 1)) / 2.0 ** (closer + further)) \
                if closer + further else 1.0
            p(f"    {cell:6s} vs {bas:4s}  n={len(ks):5d}  {da[len(ks)//2]:.3f} -> {db[len(ks)//2]:.3f}"
              f"   closer/further {closer}/{further}   sign p {pv:.3g}")

    # M7/M8 and the threshold table measure the NEAREST candidate vertex, so they move only when the
    # candidate SET moves -- they are nearly blind to which candidate the chooser picks.  This round
    # is about the chooser, so the selected candidate's own vertex is scored too: for every signal
    # interaction selected in BOTH arms, the distance from its selected candidate's vertex to the
    # true vertex (candidates.tsv t_dist_cm, the same quantity doc 107 matches on).
    p("")
    p("## chosen-vertex accuracy: the SELECTED candidate's vertex, signal selected in both arms")
    for s, flav, cut in (("cv", "numu", 0.9), ("nuecc", "nue", 7.0), ("nuecc", "nue", 4.0)):
        for cell, bas in pairs:
            if s not in arms.get(cell, {}) or s not in arms.get(bas, {}):
                continue
            def sel_dist(arm):
                sel, sig, _ = arm.selected(flav, cut)
                out = {}
                for r in sel:
                    k = (C.ev(r), r["t_idx"])
                    if k not in sig:
                        continue
                    try:
                        d = float(r["t_dist_cm"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if d >= 0 and (k not in out or d < out[k]):
                        out[k] = d
                return out
            DA, DB = sel_dist(arms[bas][s]), sel_dist(arms[cell][s])
            ks = sorted(set(DA) & set(DB))
            if not ks:
                continue
            closer = sum(1 for k in ks if DB[k] < DA[k] - 0.01)
            further = sum(1 for k in ks if DB[k] > DA[k] + 0.01)
            da = sorted(DA[k] for k in ks)
            db = sorted(DB[k] for k in ks)
            n = len(ks)
            w1a = sum(1 for k in ks if DA[k] < 1.0); w1b = sum(1 for k in ks if DB[k] < 1.0)
            w3a = sum(1 for k in ks if DA[k] < 3.0); w3b = sum(1 for k in ks if DB[k] < 3.0)
            p(f"  {s:5s} {flav}>{cut:<4} {cell:6s} vs {bas:4s}  n={n:4d}  median {da[n//2]:.3f} -> {db[n//2]:.3f} cm"
              f"   <1cm {w1a}->{w1b}   <3cm {w3a}->{w3b}   closer/further {closer}/{further}"
              f"   sign p {C.sign_test(further, closer):.3g}")
            tsv.append(f"SELVTX_{flav}{cut}\t{cell}\t{bas}\t{s}\t{w1a}\t{n}\t{w1b}\t{n}"
                       f"\t{100.0*(w1b-w1a)/n:.3f}\t{further}\t{closer}\t{C.sign_test(further, closer):.4g}\t-")

    if off:
        p("")
        p("## beam-off gates (paired by gate; the pre-registration counts a change only from 5 gates)")
        for cell, bas in pairs:
            if cell not in off or bas not in off:
                continue
            gb, sb = off[bas]
            gc, sc = off[cell]
            for key in (("numu", 0.9), ("nue", 7.0), ("nue", 4.0)):
                A, B = sb.get(key, set()), sc.get(key, set())
                p(f"  {cell:6s} vs {bas:4s}  {key[0]} > {key[1]:<4} : {len(A)}/{len(gb)} -> {len(B)}/{len(gc)}"
                  f"   -{len(A-B)}/+{len(B-A)} gates   sign p {C.sign_test(len(A-B), len(B-A)):.3g}")

    txt = "\n".join(out) + "\n"
    print(txt)
    with open(a.out + ".txt", "w") as fh:
        fh.write(txt)
    with open(a.out + ".tsv", "w") as fh:
        fh.write("\n".join(tsv) + "\n")


if __name__ == "__main__":
    main()
