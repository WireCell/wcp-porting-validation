#!/usr/bin/env python3
"""doc pdvd/115 -- fork of d114_verdict.py (doc 114): read every clause of the frozen rule figs/115_pred.txt off the
committed output files and print the verdict per level.  No number is computed here that a clause script did not
print; this only parses and compares.

Inputs (figs/):
  115_steiner_<arm>.json            d113_steiner_census.py          S1, Gb, W, P5 (seed), terminal classes (reported)
  115_steiner_d115?off.json         the same on this round's OFF arm  the base of S1 / Gb / W / P5
  115_eval_<det>_<lvl>.txt          d111_eval.py --base d115?off    P1, P2, P3, P4, P6', R
  115_compare_<det>_<lvl>.txt       d113_compare.py --logd-base     Ga, Gc, Gd
  115_support_<det>_<lvl>.txt       d114_support_census.py          U1 (DETOUR length), with 115_support_<det>_off.txt
  115_resid_<det>_<lvl>.json        d115_proj_resid.py              R2D (+ the exact-slice OFF row share, reported)
  115_dqdx_<det>_<lvl>.json         d115_dqdx_compare.py            D1 f_low, D2 shape, D3 k_pop, D4 Bee holes
  115_grade_<det>.txt               d113_grade.py                   T (REPORTED in this round, not gating)
  115_spot.tsv                      d113_spot_figs.py               P5f
  115_gate_off_<det>.txt            d111_identity_gate.py           G2 (d115?off == d114?base)
  115_gate_p3_<det>.txt             d111_identity_gate.py           G3 (d115?p3 == d114?p3)
  /home/xqian/tmp/d115/arm_<arm>.out + run logs + work dirs        C

Usage: d115_verdict.py [--figs DIR] > figs/115_verdict.txt
"""
import argparse, glob, json, os, re, sys

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
ARMD = "/home/xqian/tmp/d115"

LEVELS = [("P3", "p3"), ("BW2", "bw2"), ("BWP1", "bwp1"), ("P3BW2", "p3bw2")]
DETS = {"pdhd": "h", "pdvd": "v"}
SPOT = {"pdhd": "h1", "pdvd": "v3"}
GA_REF = {"pdhd": 4.48, "pdvd": 3.37}       # the accepted charge_stepped flip's truncation share (doc 113 / 114)
GD_REF = {"pdhd": 19, "pdvd": 9}
GC_FLIP = {"pdhd": 30.2, "pdvd": 25.1}
BARS = dict(S1=-0.20, P1=-0.15, U1=-0.30, R2D=-0.10, P5=1.2, P5F_PAD=0.2, W=1.10, GB=1.10, D2_PAD=0.01, D3_TOL=0.02, R=1.20)


def completeness(det, arm):
    summ = open(f"{ARMD}/arm_{arm}.out").read()
    m = re.search(rf"\[{arm}\] complete (\d+), incomplete (\d+), run rc=(\d+)", summ)
    runner_incomplete = re.findall(rf"\[{arm}\] INCOMPLETE (\S+)", summ)
    failed = sum(open(p, errors="replace").read().count("wire-cell PR failed") for p in glob.glob(f"{ARMD}/arm_{arm}/run_*.log"))
    dirs = glob.glob(f"{IMG}/{det}/work/*_{arm}")
    missing = [os.path.basename(d) for d in dirs
               if not all(os.path.exists(f"{d}/{f}") for f in ("tracking-pr.root", "tracking-stm.root", "mabc-pr.zip"))]
    ok = m is not None and int(m.group(3)) == 0 and failed == 0 and not missing
    txt = (f"runner: complete {m.group(1)}, incomplete {m.group(2)} {runner_incomplete}, rc {m.group(3)}; " if m else "no runner summary; ") + \
          f"events {len(dirs)}, failure lines {failed}, events missing an output {len(missing)} {missing}"
    return ok, txt


def num(s):
    return float(re.search(r"[-+]?\d+(?:\.\d+)?", s).group(0))


def eval_rows(path):
    rows = {}
    for line in open(path):
        if not line.startswith("| "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows[cells[0]] = cells[1:]
    return rows


def find(rows, prefix):
    for k, v in rows.items():
        if k.startswith(prefix):
            return v
    raise KeyError(prefix)


def detour_len(path):
    """the DETOUR roll-up row of a d114_support_census.py census: (stretches, length m)."""
    for line in open(path):
        if line.startswith("| DETOUR |"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            return int(c[1]), float(c[3])
    raise KeyError("DETOUR")


def gate_line(path):
    g = open(path).read()
    m = re.search(r"RESULT \w+:[^\n]*", g)
    return "RESULT PASS" in g, (m.group(0) if m else "no result line")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs"))
    a = ap.parse_args()
    F = a.figs
    comp_lines = []
    spot = {}
    if os.path.exists(f"{F}/115_spot.tsv"):
        for line in open(f"{F}/115_spot.tsv").read().splitlines()[1:]:
            f = line.split("\t")
            spot[(f[0], f[1])] = float(f[4])
    out = []
    verdict = {lvl: {} for lvl, _ in LEVELS}
    reported = {lvl: {} for lvl, _ in LEVELS}
    for det, x in DETS.items():
        off = f"d115{x}off"
        base = json.load(open(f"{F}/115_steiner_{off}.json"))["summary"]
        p5f_base = spot.get((SPOT[det], off), float("nan"))
        grade = open(f"{F}/115_grade_{det}.txt").read() if os.path.exists(f"{F}/115_grade_{det}.txt") else ""
        n_det, l_det = detour_len(f"{F}/115_support_{det}_off.txt")
        out.append(f"## {det}: base {off} -- S1 off {100*base['off']:.2f} %, far {100*base['far']:.2f} %, wig {100*base['wig']:.2f} %, "
                   f"{SPOT[det]} seed max {max(base['spots'].get(SPOT[det], [float('nan')])):.2f} cm, fit max {p5f_base:.2f} cm; "
                   f"DETOUR {n_det} stretches {l_det:.1f} m")
        for lvl, suf in LEVELS:
            arm = f"d115{x}{suf}"
            if not os.path.exists(f"{F}/115_steiner_{arm}.json"):
                out.append(f"### {det} {lvl} ({arm}): NOT RUN")
                verdict[lvl][det] = {"C": (False, "arm not run")}
                continue
            c = {}
            s = json.load(open(f"{F}/115_steiner_{arm}.json"))["summary"]
            rel = (s["off"] - base["off"]) / base["off"]
            c["S1"] = (rel <= BARS["S1"], f"seed > 1 cm off {100*base['off']:.2f} -> {100*s['off']:.2f} % ({100*rel:+.1f} %, bar {100*BARS['S1']:.0f} %)")
            ev = eval_rows(f"{F}/115_eval_{det}_{suf}.txt")
            p1 = find(ev, "P1"); v1, v2 = num(p1[0]), num(p1[1]); r1 = (v2 - v1) / v1 if v1 else float("nan")
            c["P1"] = (r1 <= BARS["P1"], f"fit rows > 1 cm {v1:.2f} -> {v2:.2f} % ({100*r1:+.1f} %, bar {100*BARS['P1']:.0f} %)")
            n_arm, l_arm = detour_len(f"{F}/115_support_{det}_{suf}.txt")
            ru = (l_arm - l_det) / l_det if l_det else float("nan")
            c["U1"] = (ru <= BARS["U1"], f"DETOUR length {l_det:.1f} -> {l_arm:.1f} m ({100*ru:+.1f} %, bar {100*BARS['U1']:.0f} %); stretches {n_det} -> {n_arm}")
            rj = json.load(open(f"{F}/115_resid_{det}_{suf}.json"))
            r2a, r2b = rj["A"]["R2D"], rj["B"]["R2D"]; rr2 = (r2b - r2a) / r2a if r2a else float("nan")
            c["R2D"] = (rr2 <= BARS["R2D"], f"W-plane rows > 1 wire off the measured cell {100*r2a:.2f} -> {100*r2b:.2f} % ({100*rr2:+.1f} %, bar {100*BARS['R2D']:.0f} %); "
                        f"rows off in >= 1 live plane (exact slice) {100*rj['A']['offrow']:.2f} -> {100*rj['B']['offrow']:.2f} %")
            sv = s["spots"].get(SPOT[det], [])
            smax = max(sv) if sv else float("nan")
            c["P5"] = (bool(sv) and smax <= BARS["P5"], f"{SPOT[det]} seed max {smax:.2f} cm (bar {BARS['P5']})")
            fm = spot.get((SPOT[det], arm), float("nan"))
            c["P5f"] = (fm <= p5f_base + BARS["P5F_PAD"], f"{SPOT[det]} fit max {p5f_base:.2f} -> {fm:.2f} cm (bar base + {BARS['P5F_PAD']})")
            dj = json.load(open(f"{F}/115_dqdx_{det}_{suf}.json"))
            DA, DB = dj["A"], dj["B"]
            c["D1"] = (DB["f_low_med"] <= DA["f_low_med"] + 1e-9, f"median f_low {DA['f_low_med']:.4f} -> {DB['f_low_med']:.4f} (bar <= base); paired better {dj['paired']['f_low']['better']} worse {dj['paired']['f_low']['worse']}")
            c["D2"] = (DB["shape_med"] <= DA["shape_med"] + BARS["D2_PAD"], f"median shape rms {DA['shape_med']:.4f} -> {DB['shape_med']:.4f} (bar base + {BARS['D2_PAD']}); paired better {dj['paired']['shape']['better']} worse {dj['paired']['shape']['worse']}")
            dk = (DB["k_pop"] - DA["k_pop"]) / DA["k_pop"]
            c["D3"] = (abs(dk) <= BARS["D3_TOL"], f"k_pop {DA['k_pop']:.4f} -> {DB['k_pop']:.4f} ({100*dk:+.2f} %, bar +-{100*BARS['D3_TOL']:.0f} %); chi2 {DA['chi2']:.1f} -> {DB['chi2']:.1f}")
            p3 = find(ev, "P3"); c["D4"] = (num(p3[1]) <= num(p3[0]), f"Bee holes / 10 m {num(p3[0]):.2f} -> {num(p3[1]):.2f} (bar <= base)")
            c["W"] = (s["wig"] <= BARS["W"] * base["wig"], f"seed wiggle {100*base['wig']:.2f} -> {100*s['wig']:.2f} % (bar x{BARS['W']})")
            p2 = find(ev, "P2"); c["P2"] = (num(p2[1]) <= num(p2[0]), f"off-charge rows (+-1 wire) {num(p2[0]):.2f} -> {num(p2[1]):.2f} %")
            p4 = find(ev, "P4"); c["P4"] = (num(p4[1]) >= num(p4[0]) - 1.0, f"coverage {num(p4[0]):.2f} -> {num(p4[1]):.2f} %")
            p6 = find(ev, "P6'"); c["P6'"] = (num(p6[1]) <= 1.0, f"clean-record rows > 1 cm {num(p6[1]):.2f} %")
            cmp_ = open(f"{F}/115_compare_{det}_{suf}.txt").read()
            ga = float(re.search(r"truncated \(< 0.9\) ([\d.]+) %", cmp_).group(1))
            c["Ga"] = (ga <= GA_REF[det], f"truncated fits {ga:.2f} % (bar {GA_REF[det]:.2f} %)")
            c["Gb"] = (s["far"] <= BARS["GB"] * base["far"], f"seed > 5 cm off {100*base['far']:.2f} -> {100*s['far']:.2f} % (bar x{BARS['GB']})")
            md = re.search(r"base tags dead\s+(\d+); lost in the arm\s+(\d+) \(([\d.]+) %\)", cmp_)
            mn = re.search(r"base tags nodead\s+(\d+); lost in the arm\s+(\d+) \(([\d.]+) %\)", cmp_)
            ld, ln = float(md.group(3)), float(mn.group(3))
            c["Gc"] = (ld <= ln + 10.0 and ld <= GC_FLIP[det] + 5.0,
                       f"dead-touching tags lost {md.group(2)}/{md.group(1)} = {ld:.1f} % vs other {mn.group(2)}/{mn.group(1)} = {ln:.1f} % "
                       f"(bars <= {ln + 10:.1f} and <= {GC_FLIP[det] + 5:.1f})")
            gd = int(re.search(r"clusters with fits only in base (\d+)", cmp_).group(1))
            c["Gd"] = (gd <= GD_REF[det], f"clusters losing every fit {gd} (bar {GD_REF[det]})")
            c["C"] = completeness(det, arm)
            comp_lines.append(f"{arm}: {c['C'][1]}")
            rr = num(find(ev, "R")[1])
            c["R"] = (rr <= BARS["R"], f"median wall ratio vs the OFF arm {rr:.3f} (bar {BARS['R']})")
            c["G2"] = gate_line(f"{F}/115_gate_off_{det}.txt")
            if os.path.exists(f"{F}/115_gate_p3_{det}.txt"):
                c["G3"] = gate_line(f"{F}/115_gate_p3_{det}.txt")
            else:
                c["G3"] = (None, "p3 gate not run")
            gs = re.search(rf"^\s+{lvl}\s+T -> (\w+)", grade, re.M)
            tv = gs.group(1) if gs else "MISSING"
            reported[lvl][det] = f"tags (d113_grade, NOT gating this round): {tv}"
            out.append(f"### {det} {lvl} ({arm})")
            for k in ("S1", "P1", "U1", "R2D", "P5", "P5f", "D1", "D2", "D3", "D4", "W", "P2", "P4", "P6'", "Ga", "Gb", "Gc", "Gd", "C", "R", "G2", "G3"):
                ok, txt = c[k]
                out.append(f"  {k:4s} {'PASS' if ok else 'UNDECIDED' if ok is None else 'FAIL':9s} {txt}")
            out.append(f"  T    reported  {reported[lvl][det]}")
            verdict[lvl][det] = c
    out.append("\n## verdict per level (both detectors; FIX = S1 P1 U1 R2D P5 P5f D1 D2 D3 D4, NO NEW ISSUES = the rest)")
    for lvl, suf in LEVELS:
        fails = [f"{det}:{k}" for det in DETS for k, (ok, _) in verdict[lvl].get(det, {}).items() if ok is False]
        und = [f"{det}:{k}" for det in DETS for k, (ok, _) in verdict[lvl].get(det, {}).items() if ok is None]
        v = "FAIL" if fails else "UNDECIDED" if und else "PASS"
        out.append(f"  {lvl}: {v}" + (f"  failing {', '.join(fails)}" if fails else "") + (f"  undecided {', '.join(und)}" if und else ""))
    open(f"{F}/115_arms_complete.txt", "w").write("\n".join(comp_lines) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
