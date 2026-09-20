#!/usr/bin/env python3
"""doc sbnd_xin/116: every study cell against the doc-115 baseline, paired per interaction.

d107_selection.py (reused unmodified by scripts/d116/analyze_cell.sh) prints each arm's numbers on
its own.  This script puts two arms side by side on the SAME truth rows, which is what decides
whether a delta is real:

  for each primary metric of docs/116_figs/116_pred.txt (M1..M10), per sample and cell:
    baseline k/n, cell k/n, delta in percentage points,
    the paired exchange -- signal interactions selected in the baseline only ("lost") / in the
    cell only ("gained"); for a purity, the per-event background counts; for the vertex, the
    interactions matched (< 5 cm) in one arm only --
    the exact two-sided sign test on that exchange,
    the noise floor = |s0rep - baseline| for that metric, and the frozen verdict rule:
      IMPROVED / DEGRADED iff |delta| > 2 x floor (> 0.5 pp when the floor is 0) AND p < 0.05,
      NOT SEPARABLE otherwise.
  plus, per cell: the vertex mover bands (where the baseline's matched interactions went, and the
  reverse), the beam-off gate exchange, and the doc-115 sec 13 exposure-weighted purities
  (constants imported from scripts/d115/normalisation.py, not re-typed).

Selection definitions are d107_selection.py's, duplicated here verbatim (in_fv, tclass, the
5 cm match, "selection = score cut AND reco vertex in FV", sign-blind flavour) -- that script is
the doc-107 record and is not edited.  Every number here for the baseline must reproduce
docs/115_sel*/; the script asserts the four headline counts (V6).

Usage:
  python3 d116_compare.py --cells s0rep cs p3bw csp3bw tfull --out docs/116_figs/116_compare
  -> <out>.txt (report), <out>.tsv (long form), <out>.md (the headline table for the doc)
"""
import argparse
import collections
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "scripts", "d115"))
import normalisation as N  # noqa: E402  (doc 115 sec 13 constants; main() guarded)

MATCH_CM = 5.0
SCORE = {"numu": "numu_score", "nue": "nue_score"}
f = float


def in_fv(x, y, z):
    return 5.0 < abs(x) < 190.0 and abs(y) < 190.0 and 10.0 < z < 450.0


def tclass(flav, ccnc):
    if ccnc == "NC":
        return "NC"
    return {"numu": "numuCC", "nue": "nueCC"}.get(flav, "CC-" + flav)


def wilson(k, n, z=1.0):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def sign_test(a, b):
    """Exact two-sided sign test on a discordant pair count (a, b)."""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


ev = lambda r: (r["run"], r["subrun"], r["event"])


class Arm:
    """One (sample, arm) pair: candidates + truth, with the doc-107 derived fields."""

    def __init__(self, src):
        self.src = src
        self.C = list(csv.DictReader(open(os.path.join(src, "candidates.tsv")), delimiter="\t"))
        self.T = list(csv.DictReader(open(os.path.join(src, "truth.tsv")), delimiter="\t"))
        self.truth = {}
        for t in self.T:
            t["fv"] = in_fv(f(t["vx"]), f(t["vy"]), f(t["vz"]))
            t["cls"] = tclass(t["flav"], t["ccnc"])
            t["key"] = (ev(t), t["idx"])
            self.truth[t["key"]] = t
        for r in self.C:
            r["reco_fv"] = r["vertex_default"] == "0" and in_fv(f(r["nu_x"]), f(r["nu_y"]), f(r["nu_z"]))
            d = f(r["t_dist_cm"])
            r["matched"] = r["vertex_default"] == "0" and 0 <= d < MATCH_CM
            r["truth"] = self.truth.get((ev(r), r["t_idx"])) if r["matched"] else None
        self.fvT = [t for t in self.T if t["fv"]]

    def signal_keys(self, sig):
        return [t["key"] for t in self.fvT if t["cls"] == sig]

    def selected(self, flav, cut):
        """Selected candidates and the set of signal keys they cover."""
        col, sig = SCORE[flav], flav + "CC"
        sel = [r for r in self.C if r["reco_fv"] and f(r[col]) > cut]
        sigsel = {(ev(r), r["t_idx"]) for r in sel
                  if r["matched"] and r["truth"] is not None and r["truth"]["fv"] and r["truth"]["cls"] == sig}
        bkg = [r for r in sel if (ev(r), r["t_idx"]) not in sigsel or not r["matched"]]
        return sel, sigsel, bkg

    def matched_keys(self):
        """Q1: true interactions in FV whose NEAREST candidate vertex is within 5 cm."""
        return {t["key"] for t in self.fvT if 0 <= f(t["min_cand_dist_cm"]) < MATCH_CM}

    def dist(self, key):
        t = self.truth[key]
        if t["event_has_candidate"] != "1":
            return None
        return f(t["min_cand_dist_cm"])


def band(d):
    if d is None or d < 0:
        return "no cand"
    return "<5" if d < 5 else "5-20" if d < 20 else "20-50" if d < 50 else ">50"


def load_off(src):
    """Beam-off arm: gates (from the census) and the selected gate sets per cut."""
    gates = set()
    with open(os.path.join(src, "file_rse.tsv")) as fh:
        fh.readline()
        for line in fh:
            x = line.rstrip("\n").split("\t")
            if len(x) >= 6:
                gates.add((x[3], x[4], x[5]))
    rows = list(csv.DictReader(open(os.path.join(src, "candidates.tsv")), delimiter="\t"))
    sel = collections.defaultdict(set)
    for r in rows:
        fv = r["vertex_default"] == "0" and in_fv(f(r["nu_x"]), f(r["nu_y"]), f(r["nu_z"]))
        if not fv:
            continue
        if f(r["numu_score"]) > 0.9:
            sel[("numu", 0.9)].add(ev(r))
        es = f(r["nue_score"])
        if es != -15.0:
            if es > 7.0:
                sel[("nue", 7.0)].add(ev(r))
            if es > 4.0:
                sel[("nue", 4.0)].add(ev(r))
    return gates, sel


PRIMARY = [  # (id, sample, kind, flav, cut, label)
    ("M1", "cv", "eff", "numu", 0.9, "numuCC efficiency (cv, numu > 0.9)"),
    ("M2", "cv", "pur", "numu", 0.9, "numuCC in-sample purity (cv)"),
    ("M3", "nuecc", "eff", "nue", 7.0, "nueCC efficiency (nuecc, nue > 7)"),
    ("M4", "nuecc", "pur", "nue", 7.0, "nueCC in-sample purity (nuecc, > 7)"),
    ("M5", "nuecc", "eff", "nue", 4.0, "nueCC efficiency (nuecc, nue > 4)"),
    ("M6", "nuecc", "pur", "nue", 4.0, "nueCC in-sample purity (nuecc, > 4)"),
    ("M7", "cv", "vtx", None, None, "vertex < 5 cm, all true nu in FV (cv)"),
    ("M8", "nuecc", "vtx", None, None, "vertex < 5 cm, all true nu in FV (nuecc)"),
]
SECONDARY = [
    ("S1", "nuecc", "eff", "numu", 0.9, "numuCC efficiency on the nuecc arm (90 signal)"),
    ("S2", "cv", "eff", "nue", 7.0, "nueCC efficiency on the cv arm (5 signal)"),
    ("S3", "cv", "vtxc", "numuCC", None, "vertex < 5 cm, true numuCC in FV (cv)"),
    ("S4", "cv", "vtxc", "NC", None, "vertex < 5 cm, true NC in FV (cv)"),
    ("S5", "nuecc", "vtxc", "nueCC", None, "vertex < 5 cm, true nueCC in FV (nuecc)"),
]
BASELINE_CHECK = {("cv", "numu", 0.9): (387, 557, 387, 448), ("nuecc", "nue", 7.0): (618, 1511, 618, 632),
                  ("nuecc", "nue", 4.0): (734, 1511, 734, 781)}


def metric(A, B, kind, flav, cut):
    """-> (kb, nb, kc, nc, a_only, b_only) for arms A (baseline) and B (cell)."""
    if kind == "eff":
        keys = A.signal_keys(flav + "CC")
        assert set(keys) == set(B.signal_keys(flav + "CC")), "truth rows differ between arms"
        _, sa, _ = A.selected(flav, cut)
        _, sb, _ = B.selected(flav, cut)
        ka, kb = {k for k in keys if k in sa}, {k for k in keys if k in sb}
        return len(ka), len(keys), len(kb), len(keys), len(ka - kb), len(kb - ka), (ka - kb, kb - ka)
    if kind == "pur":
        sela, sa, ba = A.selected(flav, cut)
        selb, sb, bb = B.selected(flav, cut)
        ca, cb = collections.Counter(ev(r) for r in ba), collections.Counter(ev(r) for r in bb)
        evs = set(ca) | set(cb)
        more_a = sum(1 for e in evs if ca[e] > cb[e])
        more_b = sum(1 for e in evs if cb[e] > ca[e])
        # a purity DROPS when the cell has more background: "a_only" = events where the baseline
        # is better (fewer bkg) so the sign convention matches the efficiency case (a_only = lost)
        return (len(sela) - len(ba), len(sela), len(selb) - len(bb), len(selb), more_b, more_a,
                ({e for e in evs if cb[e] > ca[e]}, {e for e in evs if ca[e] > cb[e]}))
    if kind in ("vtx", "vtxc"):
        keys = [t["key"] for t in A.fvT if kind == "vtx" or t["cls"] == flav]
        ma, mb = A.matched_keys(), B.matched_keys()
        ka, kb = {k for k in keys if k in ma}, {k for k in keys if k in mb}
        return len(ka), len(keys), len(kb), len(keys), len(ka - kb), len(kb - ka), (ka - kb, kb - ka)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="products/d115")
    ap.add_argument("--products", default="products/d116")
    ap.add_argument("--cells", nargs="+", default=["s0rep", "cs", "p3bw", "csp3bw", "tfull"])
    ap.add_argument("--floor-cell", default="s0rep", help="the cell whose |delta| is the noise floor")
    ap.add_argument("--out", default="docs/116_figs/116_compare")
    args = ap.parse_args()

    out, tsv = [], []
    p = out.append
    p("# doc 116 -- every study cell vs the doc-115 baseline, paired per true interaction (d116_compare.py)")
    p(f"FV 5<|x|<190, |y|<190, 10<z<450 cm; match = reco vertex within {MATCH_CM:g} cm; selection = score cut"
      " AND reco vertex in FV; sign test = exact two-sided on the discordant pairs; floor = |%s - baseline|"
      % args.floor_cell)

    base = {}
    for s in ("cv", "nuecc"):
        base[s] = Arm(os.path.join(args.base, s))
    for (s, flav, cut), (k, n, kp, np_) in BASELINE_CHECK.items():
        sel, sig, bkg = base[s].selected(flav, cut)
        got = (len({x for x in base[s].signal_keys(flav + "CC") if x in sig}), len(base[s].signal_keys(flav + "CC")),
               len(sel) - len(bkg), len(sel))
        assert got == (k, n, kp, np_), f"baseline {s} {flav}>{cut}: {got} != docs/115_sel {k, n, kp, np_}"
    p("baseline reproduces docs/115_sel: numuCC 387/557, 387/448; nueCC 618/1511, 618/632; >4 734/1511, 734/781  OK")

    cells = {}
    for c in args.cells:
        cells[c] = {}
        for s in ("cv", "nuecc"):
            d = os.path.join(args.products, f"{s}-{c}")
            if os.path.exists(os.path.join(d, "candidates.tsv")):
                cells[c][s] = Arm(d)
        d = os.path.join(args.products, f"off-{c}")
        if os.path.exists(os.path.join(d, "candidates.tsv")):
            cells[c]["off"] = load_off(d)
    have = {c: sorted(cells[c]) for c in cells}
    p("arms found: " + "; ".join(f"{c}: {','.join(v) or 'none'}" for c, v in have.items()))

    # ---- the noise floor
    floor = {}
    fc = args.floor_cell
    for mid, s, kind, flav, cut, label in PRIMARY + SECONDARY:
        if fc in cells and s in cells[fc]:
            kb, nb, kc, nc, ao, bo, _ = metric(base[s], cells[fc][s], kind, flav, cut)
            floor[mid] = abs(100 * kc / nc - 100 * kb / nb) if nb and nc else 0.0
    p("")
    p(f"## noise floor: {fc} vs baseline (same config, same binary, same input; DL vertex not bit-stable)")
    if not floor:
        p(f"  (no {fc} arm yet -- floor = 0, verdicts use the 0.5 pp fallback)")
    for mid, s, kind, flav, cut, label in PRIMARY + SECONDARY:
        if mid in floor:
            kb, nb, kc, nc, ao, bo, _ = metric(base[s], cells[fc][s], kind, flav, cut)
            p(f"  {mid:3s} {label:48s} {kb}/{nb} -> {kc}/{nc}  delta {100*kc/nc-100*kb/nb:+.2f} pp  exchange -{ao}/+{bo}")

    # ---- the verdict table
    verdict_rows = []
    for tier, rows in (("primary", PRIMARY), ("secondary (reported, not graded)", SECONDARY)):
        p("")
        p(f"## {tier}")
        p("  %-3s %-48s %-7s %14s %14s %8s %9s %7s %6s  %s"
          % ("id", "metric", "cell", "baseline", "cell", "delta", "-lost/+gain", "p", "floor", "verdict"))
        for mid, s, kind, flav, cut, label in rows:
            for c in args.cells:
                if s not in cells.get(c, {}):
                    continue
                kb, nb, kc, nc, ao, bo, _ = metric(base[s], cells[c][s], kind, flav, cut)
                if not nb or not nc:
                    continue
                d = 100 * kc / nc - 100 * kb / nb
                pv = sign_test(ao, bo)
                fl = floor.get(mid, 0.0)
                thr = 2 * fl if fl > 0 else 0.5
                if c == fc:
                    v = "(floor)"
                elif tier != "primary":
                    v = "--"
                elif abs(d) > thr and pv < 0.05:
                    v = "IMPROVED" if d > 0 else "DEGRADED"
                else:
                    v = "not separable"
                p("  %-3s %-48s %-7s %14s %14s %+7.2f %9s %7.3f %6.2f  %s"
                  % (mid, label, c, f"{kb}/{nb}={100*kb/nb:.1f}%", f"{kc}/{nc}={100*kc/nc:.1f}%",
                     d, f"-{ao}/+{bo}", pv, fl, v))
                tsv.append((tier.split()[0], mid, s, c, label, kb, nb, kc, nc, "%.3f" % d, ao, bo, "%.4f" % pv,
                            "%.3f" % fl, v))
                if tier == "primary":
                    verdict_rows.append((mid, label, c, kb, nb, kc, nc, d, ao, bo, pv, v))

    # ---- vertex movers: where the baseline's matched interactions went, by class and band
    p("")
    p("## vertex movers (true nu in FV; nearest-candidate distance): baseline matched -> cell band, and the reverse")
    for c in args.cells:
        if c == fc:
            continue
        for s in ("cv", "nuecc"):
            if s not in cells.get(c, {}):
                continue
            A, B = base[s], cells[c][s]
            ma, mb = A.matched_keys(), B.matched_keys()
            lost = collections.Counter((A.truth[k]["cls"], band(B.dist(k))) for k in ma - mb)
            gain = collections.Counter((A.truth[k]["cls"], band(A.dist(k))) for k in mb - ma)
            p(f"  {c}/{s}: matched {len(ma)} -> {len(mb)}  lost {len(ma-mb)}  gained {len(mb-ma)}")
            for cl in ("numuCC", "nueCC", "NC"):
                lo = "  ".join(f"{b}:{lost[(cl, b)]}" for b in ("5-20", "20-50", ">50", "no cand") if lost[(cl, b)])
                ga = "  ".join(f"{b}:{gain[(cl, b)]}" for b in ("5-20", "20-50", ">50", "no cand") if gain[(cl, b)])
                if lo or ga:
                    p(f"     {cl:7s} lost -> [{lo}]   gained from [{ga}]")

    # ---- beam-off
    p("")
    p("## beam-off: gates with a selected candidate (reco vertex in FV), paired by gate")
    offb = load_off(os.path.join(args.base, "off")) if os.path.exists(os.path.join(args.base, "off", "candidates.tsv")) else None
    if offb:
        gb, sb = offb
        for c in args.cells:
            if "off" not in cells.get(c, {}):
                continue
            gc, sc = cells[c]["off"]
            assert gb == gc, "beam-off gate census differs"
            for cut in (("numu", 0.9), ("nue", 7.0), ("nue", 4.0)):
                a, b = sb[cut], sc[cut]
                lo, hi = wilson(len(b), len(gc))
                p(f"  {c:7s} {cut[0]} > {cut[1]:<4g}: {len(a)}/{len(gb)} -> {len(b)}/{len(gc)} = {100*len(b)/len(gc):.2f} %"
                  f" [{100*lo:.2f}, {100*hi:.2f}]   -{len(a-b)}/+{len(b-a)} gates")
                tsv.append(("off", "M9" if cut[0] == "numu" else "M10", "off", c, f"beam-off {cut[0]}>{cut[1]}",
                            len(a), len(gb), len(b), len(gc), "%.3f" % (100 * (len(b) - len(a)) / len(gc)),
                            len(a - b), len(b - a), "", "", "rate"))

    # ---- doc 115 sec 13 exposure-weighted purities, per cell
    p("")
    p("## exposure-weighted purities (doc 115 sec 13 method; POT and gate constants from scripts/d115/normalisation.py)")
    w = N.POT_CV / N.POT_NUE
    gates = N.POT_CV / N.POT_PER_SPILL
    p(f"  w(nuecc->cv) = {w:.4e}; beam gates at {N.POT_PER_SPILL:.0e} POT/spill = {gates:.0f}; f = 1 (off-beam events per gate, unmeasured)")
    p("  %-8s %26s %26s %26s" % ("cell", "numuCC pur in-sample->f=1", "nueCC>7 common-exposure", "nueCC>4 common-exposure"))
    for c in ["baseline"] + args.cells:
        arms = {"cv": base["cv"], "nuecc": base["nuecc"]} if c == "baseline" else cells.get(c, {})
        if "cv" not in arms or "nuecc" not in arms:
            continue
        sel, sig, bkg = arms["cv"].selected("numu", 0.9)
        n_off = len(offb[1][("numu", 0.9)]) if c == "baseline" else (len(arms["off"][1][("numu", 0.9)]) if "off" in arms else None)
        if n_off is None:
            numu = "no off arm"
        else:
            cos = n_off * gates / N.OFF_GATES
            numu = f"{100*(len(sel)-len(bkg))/len(sel):.1f} -> {100*(len(sel)-len(bkg))/(len(sel)+cos):.1f} % (cosmic {cos:.1f})"
        cols = [numu]
        for cut in (7.0, 4.0):
            seln, sign, bkgn = arms["nuecc"].selected("nue", cut)
            selc, sigc, bkgc = arms["cv"].selected("nue", cut)
            S, Bn = len(sign) * w, len(bkgn) * w
            # cv contributes only its NON-nue background (its nueCC signal would double count)
            Bc = sum(1 for r in bkgc if not (r["matched"] and r["truth"] is not None and r["truth"]["cls"] == "nueCC"))
            cols.append(f"{100*S/(S+Bn+Bc):.1f} % (S {S:.2f}, cv bkg {Bc})")
        p("  %-8s %26s %26s %26s" % (c, *cols))

    txt = "\n".join(out) + "\n"
    print(txt)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out + ".txt", "w").write(txt)
    with open(args.out + ".tsv", "w") as fo:
        fo.write("tier\tid\tsample\tcell\tmetric\tbase_k\tbase_n\tcell_k\tcell_n\tdelta_pp\tlost\tgained\tp\tfloor_pp\tverdict\n")
        for r in tsv:
            fo.write("\t".join(str(x) for x in r) + "\n")
    # the headline table for the doc: metric rows x cell columns
    md = []
    cl = [c for c in args.cells if any(s in cells.get(c, {}) for s in ("cv", "nuecc"))]
    md.append("| metric | baseline | " + " | ".join(cl) + " |")
    md.append("|---|---:|" + "|".join("---:" for _ in cl) + "|")
    for mid, s, kind, flav, cut, label in PRIMARY:
        row = None
        vals = {}
        for (m2, lab, c, kb, nb, kc, nc, d, ao, bo, pv, v) in verdict_rows:
            if m2 == mid:
                row = f"{kb}/{nb} = {100*kb/nb:.1f} %"
                tag = "" if v in ("(floor)", "not separable") else f" **{v}**"
                vals[c] = f"{kc}/{nc} = {100*kc/nc:.1f} % ({d:+.1f}; −{ao}/+{bo}, p {pv:.2f}){tag}"
        if row:
            md.append(f"| {mid} {label} | {row} | " + " | ".join(vals.get(c, "—") for c in cl) + " |")
    open(args.out + ".md", "w").write("\n".join(md) + "\n")
    print(f"-> {args.out}.txt / .tsv / .md")


if __name__ == "__main__":
    main()
