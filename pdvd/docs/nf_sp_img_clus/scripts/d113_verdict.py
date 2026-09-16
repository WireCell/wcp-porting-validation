#!/usr/bin/env python3
"""doc pdvd/113 -- read every clause of the frozen rule figs/113_pred.txt off the committed output files and print the
verdict per level.  No number is computed here that a clause script did not print; this only parses and compares.

Inputs (figs/):
  113_steiner_<arm>.json            d113_steiner_census.py       S1, Gb, P5 (seed), terminal classes (reported)
  113_eval_<det>_<lvl>.txt          d111_eval.py --base <base>   P1, P2, P3, P4, P6'
  113_evalR_<det>_<lvl>.txt         d111_eval.py --base <base2>  R
  113_compare_<det>_<lvl>.txt       d113_compare.py --logd-base  Ga, Gc, Gd
  113_grade_<det>.txt               d113_grade.py                T
  113_spot.tsv                      d113_spot_figs.py            P5f
  /home/xqian/tmp/d113/arm_<arm>.out + run logs + work dirs     C (amendment 1: rc 0 and the three output files;
                                    the runner's summary and its INCOMPLETE events are written to 113_arms_complete.txt)

Usage: d113_verdict.py [--figs DIR] > figs/113_verdict.txt
"""
import argparse, glob, json, os, re, sys

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
ARMD = "/home/xqian/tmp/d113"


def completeness(det, arm):
    """figs/113_pred_amend1.txt: complete = no 'wire-cell PR failed' line in the arm's run logs, and every event dir
    holds tracking-pr.root, tracking-stm.root and mabc-pr.zip.  Returns (ok, text)."""
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

LEVELS = [("L3", "none"), ("L2", "foot"), ("L1", "nopaint")]
DETS = {"pdhd": "h", "pdvd": "v"}
SPOT = {"pdhd": "h1", "pdvd": "v3"}
GA_REF = {"pdhd": 4.48, "pdvd": 3.37}
GD_REF = {"pdhd": 19, "pdvd": 9}
GC_FLIP = {"pdhd": 30.2, "pdvd": 25.1}
P5F_BASE = {"pdhd": 1.769, "pdvd": 0.912}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs"))
    a = ap.parse_args()
    F = a.figs
    comp_lines = []
    spot = {}
    for line in open(f"{F}/113_spot.tsv").read().splitlines()[1:]:
        f = line.split("\t")
        spot[(f[0], f[1])] = float(f[4])
    out = []
    verdict = {lvl: {} for lvl, _ in LEVELS}
    for det, x in DETS.items():
        base = json.load(open(f"{F}/113_steiner_d113{x}base.json"))["summary"]
        grade = open(f"{F}/113_grade_{det}.txt").read()
        out.append(f"## {det}: base d113{x}base -- S1 off {100*base['off']:.2f} %, far {100*base['far']:.2f} %, "
                   f"{SPOT[det]} seed max {max(base['spots'].get(SPOT[det], [float('nan')])):.2f} cm")
        for lvl, suf in LEVELS:
            arm = f"d113{x}{suf}"
            c = {}
            s = json.load(open(f"{F}/113_steiner_{arm}.json"))["summary"]
            rel = (s["off"] - base["off"]) / base["off"]
            c["S1"] = (rel <= -0.25, f"seed > 1 cm off {100*base['off']:.2f} -> {100*s['off']:.2f} % ({100*rel:+.1f} %, bar -25 %)")
            ev = eval_rows(f"{F}/113_eval_{det}_{suf}.txt")
            p1 = find(ev, "P1"); v1, v2 = num(p1[0]), num(p1[1]); r1 = (v2 - v1) / v1 if v1 else float("nan")
            c["P1"] = (r1 <= -0.20, f"fit rows > 1 cm {v1:.2f} -> {v2:.2f} % ({100*r1:+.1f} %, bar -20 %)")
            sv = s["spots"].get(SPOT[det], [])
            smax = max(sv) if sv else float("nan")
            c["P5"] = (bool(sv) and smax <= 1.2, f"{SPOT[det]} seed max {smax:.2f} cm (bar 1.2)")
            gs = re.search(rf"^\s+{lvl}\s+T -> (\w+)", grade, re.M)
            tv = gs.group(1) if gs else "MISSING"
            c["T"] = (tv == "PASS" if tv != "UNDECIDED" else None, f"grade {tv}")
            p2 = find(ev, "P2"); c["P2"] = (num(p2[1]) <= num(p2[0]), f"off-charge rows {num(p2[0]):.2f} -> {num(p2[1]):.2f} %")
            p3 = find(ev, "P3"); c["P3"] = (num(p3[1]) <= num(p3[0]), f"Bee holes / 10 m {num(p3[0]):.2f} -> {num(p3[1]):.2f}")
            p4 = find(ev, "P4"); c["P4"] = (num(p4[1]) >= num(p4[0]) - 1.0, f"coverage {num(p4[0]):.2f} -> {num(p4[1]):.2f} %")
            p6 = find(ev, "P6'"); c["P6'"] = (num(p6[1]) <= 1.0, f"clean-record rows > 1 cm {num(p6[1]):.2f} %")
            fm = spot.get((SPOT[det], arm), float("nan"))
            c["P5f"] = (fm <= P5F_BASE[det] + 0.2, f"{SPOT[det]} fit max {P5F_BASE[det]:.2f} -> {fm:.2f} cm")
            cmp_ = open(f"{F}/113_compare_{det}_{suf}.txt").read()
            ga = float(re.search(r"truncated \(< 0.9\) ([\d.]+) %", cmp_).group(1))
            c["Ga"] = (ga <= GA_REF[det], f"truncated fits {ga:.2f} % (bar {GA_REF[det]:.2f} %)")
            c["Gb"] = (s["far"] <= 1.10 * base["far"], f"seed > 5 cm off {100*base['far']:.2f} -> {100*s['far']:.2f} % (bar x1.10)")
            md = re.search(r"base tags dead\s+(\d+); lost in the arm\s+(\d+) \(([\d.]+) %\)", cmp_)
            mn = re.search(r"base tags nodead\s+(\d+); lost in the arm\s+(\d+) \(([\d.]+) %\)", cmp_)
            ld, ln = float(md.group(3)), float(mn.group(3))
            c["Gc"] = (ld <= ln + 10.0 and ld <= GC_FLIP[det] + 5.0,
                       f"dead-touching tags lost {md.group(2)}/{md.group(1)} = {ld:.1f} % vs other {mn.group(2)}/{mn.group(1)} "
                       f"= {ln:.1f} % (bars <= {ln + 10:.1f} and <= {GC_FLIP[det] + 5:.1f})")
            gd = int(re.search(r"clusters with fits only in base (\d+)", cmp_).group(1))
            c["Gd"] = (gd <= GD_REF[det], f"clusters losing every fit {gd} (bar {GD_REF[det]})")
            c["C"] = completeness(det, arm)
            comp_lines.append(f"{arm}: {c['C'][1]}")
            er = eval_rows(f"{F}/113_evalR_{det}_{suf}.txt")
            rr = num(find(er, "R")[1])
            c["R"] = (rr <= 1.20, f"median wall ratio vs base2 {rr:.3f}")
            out.append(f"### {det} {lvl} ({arm})")
            for k in ("S1", "P1", "P5", "T", "P2", "P3", "P4", "P6'", "P5f", "Ga", "Gb", "Gc", "Gd", "C", "R"):
                ok, txt = c[k]
                out.append(f"  {k:4s} {'PASS' if ok else 'UNDECIDED' if ok is None else 'FAIL':9s} {txt}")
            verdict[lvl][det] = c
    out.append("\n## verdict per level (both detectors)")
    for lvl, suf in LEVELS:
        fails = [f"{det}:{k}" for det in DETS for k, (ok, _) in verdict[lvl][det].items() if ok is False]
        und = [f"{det}:{k}" for det in DETS for k, (ok, _) in verdict[lvl][det].items() if ok is None]
        v = "FAIL" if fails else "UNDECIDED" if und else "PASS"
        out.append(f"  {lvl}: {v}" + (f"  failing {', '.join(fails)}" if fails else "") + (f"  undecided {', '.join(und)}" if und else ""))
    open(f"{F}/113_arms_complete.txt", "w").write("\n".join(comp_lines) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
