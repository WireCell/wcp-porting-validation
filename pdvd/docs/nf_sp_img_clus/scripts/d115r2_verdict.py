#!/usr/bin/env python3
"""doc pdvd/115 round 2 -- read every clause of the frozen rule figs/115r2_pred.txt off the committed output files and
print the verdict per level, the FIX (detour-scoped) group and the NO-REGRESSION (arm-wide) group apart, and the
recommendation.  Fork of d115_verdict.py (round 1).  No number is computed here that a clause script did not print;
this only parses and compares.

Inputs (figs/), per level with prefix P (115r2_ for the round-2 arms; 115_ for the round-1 anchors under --anchors):
  P_steiner_<arm>.json              d113_steiner_census.py          S1, Gb, W, seed spot max (reported)
  115_steiner_d115?off.json         round 1's OFF census             the base of S1 / Gb / W
  P_eval_<det>_<lvl>.txt            d111_eval.py --base d115?off    P1, P2, P3 (D4), P4, P6', R
  P_compare_<det>_<lvl>.txt         d113_compare.py --logd-base     Ga, Gc, Gd
  P_support_<det>_<lvl>.txt         d114_support_census.py          U1 DETOUR, U1b DETOUR+FIT, U2 blank-carried, U3 crawl
  115_support_<det>_off.txt         round 1's OFF support census     their base
  P_resid_<det>_<lvl>.json          d115_proj_resid.py              R2D
  P_dqdx_<det>_<lvl>.json           d115_dqdx_compare.py            D1 f_low, D2 shape, D3 k_pop
  115r2_grade_<det>.txt             d113_grade.py                   T (REPORTED)
  115r2_spot.tsv                    d113_spot_figs.py               P5f (five spots), seed max (reported)
  /home/xqian/tmp/d115/arm_<arm>.out + run logs + work dirs        C, G0 (pin md5)

Usage: d115r2_verdict.py [--figs DIR] [--anchors] > figs/115r2_verdict.txt
"""
import argparse, glob, json, os, re

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
ARMD = "/home/xqian/tmp/d115"
PIN_MD5 = "2efa7fa09325"

# (label, arm suffix, alpha, figs prefix)
LEVELS = [("P3BWP025", "p3bwp025", 0.25, "115r2_"), ("P3BWP05", "p3bwp05", 0.5, "115r2_"), ("P3BWP1", "p3bwp1", 1.0, "115r2_")]
ANCHORS = [("P3", "p3", 0.0, "115_"), ("BWP1", "bwp1", 1.0, "115_"), ("P3BW2", "p3bw2", 2.0, "115_")]
DETS = {"pdhd": "h", "pdvd": "v"}
SPOTS = {"pdhd": ["h1", "h2"], "pdvd": ["v1", "v2", "v3"]}
SPOT_ABS = {"h1": 1.0, "h2": 1.0, "v3": 0.7}          # the fits round 1 showed the levers fix
GA_REF = {"pdhd": 4.48, "pdvd": 3.37}
GD_REF = {"pdhd": 19, "pdvd": 9}
GC_FLIP = {"pdhd": 30.2, "pdvd": 25.1}
D1_FACTOR = {"pdhd": 0.90, "pdvd": 1.0}
BARS = dict(U1=-0.30, U1B=-0.20, U2=-0.40, P5F_PAD=0.2, W=1.10, GB=1.10, D3_TOL=0.02, R=1.20)
FIX = ("U1", "U1b", "U2", "U3", "P5f", "D1", "D2", "D3", "D4")
NOREG = ("S1", "P1", "R2D", "W", "P2", "P4", "P6'", "Ga", "Gb", "Gc", "Gd", "C", "R", "G0")


def completeness(det, arm):
    p = f"{ARMD}/arm_{arm}.out"
    summ = open(p).read() if os.path.exists(p) else ""
    m = re.search(rf"\[{arm}\] complete (\d+), incomplete (\d+), run rc=(\d+)", summ)
    runner_incomplete = re.findall(rf"\[{arm}\] INCOMPLETE (\S+)", summ)
    failed = sum(open(q, errors="replace").read().count("wire-cell PR failed") for q in glob.glob(f"{ARMD}/arm_{arm}/run_*.log"))
    dirs = glob.glob(f"{IMG}/{det}/work/*_{arm}")
    missing = [os.path.basename(d) for d in dirs
               if not all(os.path.exists(f"{d}/{f}") for f in ("tracking-pr.root", "tracking-stm.root", "mabc-pr.zip"))]
    ok = m is not None and int(m.group(3)) == 0 and failed == 0 and not missing
    txt = (f"runner: complete {m.group(1)}, incomplete {m.group(2)} {runner_incomplete}, rc {m.group(3)}; " if m else "no runner summary; ") + \
          f"events {len(dirs)}, failure lines {failed}, events missing an output {len(missing)} {missing}"
    pm = re.search(r"pin md5 after (\w+)", summ)
    g0 = (pm is not None and pm.group(1).startswith(PIN_MD5), f"pin md5 after the arm {pm.group(1) if pm else 'MISSING'} (bar {PIN_MD5})")
    return (ok, txt), g0


def num(s):
    return float(re.search(r"[-+]?\d+(?:\.\d+)?", s).group(0))


def eval_rows(path):
    rows = {}
    for line in open(path):
        if line.startswith("| "):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows[cells[0]] = cells[1:]
    return rows


def find(rows, prefix):
    for k, v in rows.items():
        if k.startswith(prefix):
            return v
    raise KeyError(prefix)


def support(path):
    """(stretches, length m) per class row of a d114_support_census.py census: the per-class table (10 columns, length
    in column 4) and the roll-up (5 columns, length in column 3); a class absent from the table is (0, 0)."""
    cls, roll = {}, {}
    for line in open(path):
        if not line.startswith("| "):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) >= 9 and re.fullmatch(r"\d+", c[1]):
            cls[c[0]] = (int(c[1]), float(c[4]))
        elif len(c) == 5 and re.fullmatch(r"\d+", c[1]):
            roll[c[0]] = (int(c[1]), float(c[3]))
    return cls, roll


def spots_table(path):
    spot = {}
    if os.path.exists(path):
        for line in open(path).read().splitlines()[1:]:
            f = line.split("\t")
            spot[(f[0], f[1])] = (float(f[3]), float(f[4]))
    return spot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs"))
    ap.add_argument("--anchors", action="store_true", help="also grade the round-1 arms P3 / BWP1 / P3BW2 from their 115_ files under the round-2 clauses")
    # doc pdvd/116 Part 1 (additive): grade other levels under the same clauses, e.g. --levels BWP05:bwp05:0.5:115r2_ --spot 116_spot.tsv
    ap.add_argument("--levels", default=None, help="comma list LABEL:tag:alpha:prefix replacing the round-2 level table")
    ap.add_argument("--spot", default=None, help="an extra spot table (figs/<name>) consulted before 115r2_spot.tsv")
    a = ap.parse_args()
    F = a.figs
    levels = LEVELS + (ANCHORS if a.anchors else [])
    if a.levels:
        levels = [(t[0], t[1], float(t[2]), t[3]) for t in (x.split(":") for x in a.levels.split(","))]
    spot = spots_table(f"{F}/115r2_spot.tsv")
    if a.spot and os.path.exists(f"{F}/{a.spot}"):
        spot = {**spot, **spots_table(f"{F}/{a.spot}")}
    spot1 = spots_table(f"{F}/115_spot.tsv")
    comp_lines, out = [], []
    verdict = {lvl: {} for lvl, *_ in levels}
    for det, x in DETS.items():
        off = f"d115{x}off"
        base = json.load(open(f"{F}/115_steiner_{off}.json"))["summary"]
        bcls, broll = support(f"{F}/115_support_{det}_off.txt")
        grade = open(f"{F}/115r2_grade_{det}.txt").read() if os.path.exists(f"{F}/115r2_grade_{det}.txt") else ""
        grade1 = open(f"{F}/115_grade_{det}.txt").read() if os.path.exists(f"{F}/115_grade_{det}.txt") else ""
        b_det, b_fit = broll["DETOUR"], broll["FIT"]
        b_blank = bcls.get("D-blank-term", (0, 0))[0] + bcls.get("D-blank-int", (0, 0))[0]
        b_crawl = bcls.get("D-crawl", (0, 0))[0]
        sp_base = {s: (spot.get((s, off)) or spot1.get((s, off)) or (float("nan"), float("nan"))) for s in SPOTS[det]}
        out.append(f"## {det}: base {off} -- DETOUR {b_det[0]} stretches {b_det[1]:.1f} m, FIT {b_fit[0]} / {b_fit[1]:.1f} m, blank-carried "
                   f"(D-blank-term + D-blank-int) {b_blank}, D-crawl {b_crawl}; S1 off {100*base['off']:.2f} %, wig {100*base['wig']:.2f} %; "
                   f"spot fit max " + ", ".join(f"{s} {sp_base[s][1]:.2f}" for s in SPOTS[det]) + " cm")
        for lvl, suf, alpha, P in levels:
            arm = f"d115{x}{suf}"
            need = [f"{P}steiner_{arm}.json", f"{P}support_{det}_{suf}.txt", f"{P}eval_{det}_{suf}.txt", f"{P}compare_{det}_{suf}.txt",
                    f"{P}resid_{det}_{suf}.json", f"{P}dqdx_{det}_{suf}.json"]
            missing = [n for n in need if not os.path.exists(f"{F}/{n}")]
            if missing:
                out.append(f"### {det} {lvl} ({arm}): NOT RUN (missing {', '.join(missing)})")
                verdict[lvl][det] = {"C": (False, "arm not run")}
                continue
            c = {}
            s = json.load(open(f"{F}/{P}steiner_{arm}.json"))["summary"]
            cls, roll = support(f"{F}/{P}support_{det}_{suf}.txt")
            a_det, a_fit = roll["DETOUR"], roll["FIT"]
            r = (a_det[1] - b_det[1]) / b_det[1]
            c["U1"] = (r <= BARS["U1"], f"DETOUR length {b_det[1]:.1f} -> {a_det[1]:.1f} m ({100*r:+.1f} %, bar {100*BARS['U1']:.0f} %); stretches {b_det[0]} -> {a_det[0]}")
            bl, al = b_det[1] + b_fit[1], a_det[1] + a_fit[1]
            r = (al - bl) / bl
            c["U1b"] = (r <= BARS["U1B"], f"DETOUR + FIT length {bl:.1f} -> {al:.1f} m ({100*r:+.1f} %, bar {100*BARS['U1B']:.0f} %); FIT {b_fit[0]} / {b_fit[1]:.1f} m -> {a_fit[0]} / {a_fit[1]:.1f} m")
            a_blank = cls.get("D-blank-term", (0, 0))[0] + cls.get("D-blank-int", (0, 0))[0]
            r = (a_blank - b_blank) / b_blank if b_blank else float("nan")
            c["U2"] = (r <= BARS["U2"], f"blank-carried detour stretches {b_blank} -> {a_blank} ({100*r:+.1f} %, bar {100*BARS['U2']:.0f} %); "
                       f"D-blank-term {bcls.get('D-blank-term', (0, 0))[0]} -> {cls.get('D-blank-term', (0, 0))[0]}, D-blank-int {bcls.get('D-blank-int', (0, 0))[0]} -> {cls.get('D-blank-int', (0, 0))[0]}")
            a_crawl = cls.get("D-crawl", (0, 0))[0]
            c["U3"] = (a_crawl <= b_crawl, f"D-crawl stretches {b_crawl} -> {a_crawl} (bar <= base); D-3live {bcls.get('D-3live', (0, 0))[0]} -> {cls.get('D-3live', (0, 0))[0]} (reported)")
            parts, ok5 = [], True
            for sname in SPOTS[det]:
                sv = spot.get((sname, arm)) or spot1.get((sname, arm))
                if sv is None:
                    ok5 = False; parts.append(f"{sname} MISSING"); continue
                fit = sv[1]; bfit = sp_base[sname][1]
                ok_s = fit <= bfit + BARS["P5F_PAD"] and (sname not in SPOT_ABS or fit <= SPOT_ABS[sname])
                ok5 &= ok_s
                parts.append(f"{sname} fit {bfit:.2f} -> {fit:.2f}" + (f" (bar <= {SPOT_ABS[sname]})" if sname in SPOT_ABS else "") + ("" if ok_s else " FAIL"))
            c["P5f"] = (ok5, "spot fit max (every spot <= base + 0.2): " + "; ".join(parts))
            dj = json.load(open(f"{F}/{P}dqdx_{det}_{suf}.json"))
            DA, DB = dj["A"], dj["B"]; pf, ps = dj["paired"]["f_low"], dj["paired"]["shape"]
            c["D1"] = (DB["f_low_med"] <= D1_FACTOR[det] * DA["f_low_med"] + 1e-9 and pf["better"] >= pf["worse"],
                       f"median f_low {DA['f_low_med']:.4f} -> {DB['f_low_med']:.4f} ({100*(DB['f_low_med']/DA['f_low_med']-1) if DA['f_low_med'] else float('nan'):+.1f} %, bar <= {D1_FACTOR[det]:.2f} x base); paired better {pf['better']} worse {pf['worse']} (bar better >= worse)")
            c["D2"] = (DB["shape_med"] <= DA["shape_med"] + 1e-9 and ps["better"] >= ps["worse"],
                       f"median shape rms {DA['shape_med']:.4f} -> {DB['shape_med']:.4f} (bar <= base); paired better {ps['better']} worse {ps['worse']} (bar better >= worse)")
            dk = (DB["k_pop"] - DA["k_pop"]) / DA["k_pop"]
            c["D3"] = (abs(dk) <= BARS["D3_TOL"], f"k_pop {DA['k_pop']:.4f} -> {DB['k_pop']:.4f} ({100*dk:+.2f} %, bar +-{100*BARS['D3_TOL']:.0f} %); chi2 {DA['chi2']:.1f} -> {DB['chi2']:.1f}")
            ev = eval_rows(f"{F}/{P}eval_{det}_{suf}.txt")
            p3 = find(ev, "P3"); c["D4"] = (num(p3[1]) <= num(p3[0]), f"Bee holes / 10 m {num(p3[0]):.2f} -> {num(p3[1]):.2f} (bar <= base)")
            # --- no regression, arm-wide
            rel = (s["off"] - base["off"]) / base["off"]
            c["S1"] = (s["off"] <= base["off"], f"seed > 1 cm off {100*base['off']:.2f} -> {100*s['off']:.2f} % ({100*rel:+.1f} %, bar <= base)")
            p1 = find(ev, "P1"); v1, v2 = num(p1[0]), num(p1[1])
            c["P1"] = (v2 <= v1, f"fit rows > 1 cm {v1:.2f} -> {v2:.2f} % ({100*(v2-v1)/v1:+.1f} %, bar <= base)")
            rj = json.load(open(f"{F}/{P}resid_{det}_{suf}.json"))
            r2a, r2b = rj["A"]["R2D"], rj["B"]["R2D"]
            c["R2D"] = (r2b <= r2a, f"W-plane rows > 1 wire off the measured cell {100*r2a:.2f} -> {100*r2b:.2f} % ({100*(r2b-r2a)/r2a:+.1f} %, bar <= base); "
                        f"rows off in >= 1 live plane (exact slice) {100*rj['A']['offrow']:.2f} -> {100*rj['B']['offrow']:.2f} %")
            c["W"] = (s["wig"] <= BARS["W"] * base["wig"], f"seed wiggle {100*base['wig']:.2f} -> {100*s['wig']:.2f} % ({100*(s['wig']/base['wig']-1):+.1f} %, bar x{BARS['W']})")
            p2 = find(ev, "P2"); c["P2"] = (num(p2[1]) <= num(p2[0]), f"off-charge rows (+-1 wire) {num(p2[0]):.2f} -> {num(p2[1]):.2f} %")
            p4 = find(ev, "P4"); c["P4"] = (num(p4[1]) >= num(p4[0]) - 1.0, f"coverage {num(p4[0]):.2f} -> {num(p4[1]):.2f} %")
            p6 = find(ev, "P6'"); c["P6'"] = (num(p6[1]) <= 1.0, f"clean-record rows > 1 cm {num(p6[1]):.2f} %")
            cmp_ = open(f"{F}/{P}compare_{det}_{suf}.txt").read()
            ga = float(re.search(r"truncated \(< 0.9\) ([\d.]+) %", cmp_).group(1))
            c["Ga"] = (ga <= GA_REF[det], f"truncated fits {ga:.2f} % (bar {GA_REF[det]:.2f} %)")
            c["Gb"] = (s["far"] <= BARS["GB"] * base["far"], f"seed > 5 cm off {100*base['far']:.2f} -> {100*s['far']:.2f} % (bar x{BARS['GB']})")
            md = re.search(r"base tags dead\s+(\d+); lost in the arm\s+(\d+) \(([\d.]+) %\)", cmp_)
            mn = re.search(r"base tags nodead\s+(\d+); lost in the arm\s+(\d+) \(([\d.]+) %\)", cmp_)
            ld, ln = float(md.group(3)), float(mn.group(3))
            c["Gc"] = (ld <= ln + 10.0 and ld <= GC_FLIP[det] + 5.0,
                       f"dead-touching tags lost {md.group(2)}/{md.group(1)} = {ld:.1f} % vs other {mn.group(2)}/{mn.group(1)} = {ln:.1f} % (bars <= {ln + 10:.1f} and <= {GC_FLIP[det] + 5:.1f})")
            gd = int(re.search(r"clusters with fits only in base (\d+)", cmp_).group(1))
            c["Gd"] = (gd <= GD_REF[det], f"clusters losing every fit {gd} (bar {GD_REF[det]})")
            c["C"], c["G0"] = completeness(det, arm)
            comp_lines.append(f"{arm}: {c['C'][1]}")
            rr = num(find(ev, "R")[1])
            c["R"] = (rr <= BARS["R"], f"median wall ratio vs the OFF arm {rr:.3f} (bar {BARS['R']})")
            # --- reported
            seeds = "; ".join(f"{sn} seed {((spot.get((sn, arm)) or spot1.get((sn, arm)) or (float('nan'),))[0]):.2f}" for sn in SPOTS[det])
            g = grade if P == "115r2_" else grade1
            gs = re.search(rf"^\s+{lvl}\s+T -> (\w+)", g, re.M)
            rep = [f"spot seed max (not gating): {seeds} cm", f"tags (d113_grade, NOT gating): {gs.group(1) if gs else 'MISSING'}",
                   f"interiors / far (not gating): far {100*s['far']:.2f} %, cov {100*s['cov']:.2f} %"]
            out.append(f"### {det} {lvl} ({arm}){' [round-1 anchor under the round-2 clauses]' if P == '115_' else ''}")
            out.append("  FIX (detour-scoped):")
            for k in FIX:
                ok, txt = c[k]; out.append(f"    {k:4s} {'PASS' if ok else 'FAIL':5s} {txt}")
            out.append("  NO REGRESSION (arm-wide):")
            for k in NOREG:
                ok, txt = c[k]; out.append(f"    {k:4s} {'PASS' if ok else 'FAIL':5s} {txt}")
            for t in rep:
                out.append(f"    T    reported  {t}")
            verdict[lvl][det] = c
    out.append("\n## verdict per level (both detectors; FIX = " + " ".join(FIX) + "; NO REGRESSION = " + " ".join(NOREG) + ")")
    summary = {}
    for lvl, suf, alpha, P in levels:
        fails = [f"{det}:{k}" for det in DETS for k, (ok, _) in verdict[lvl].get(det, {}).items() if ok is False]
        ffix = [f for f in fails if f.split(":")[1] in FIX]
        summary[lvl] = (fails, alpha)
        out.append(f"  {lvl}{' (anchor)' if P == '115_' else ''}: {'PASS' if not fails else 'FAIL'}"
                   + (f"  failing FIX {', '.join(ffix) or '-'}; NO-REGRESSION {', '.join(f for f in fails if f not in ffix) or '-'}" if fails else ""))
    out.append("\n## recommendation (figs/115r2_pred.txt): the smallest-alpha PASSING round-2 level; a larger passing alpha is named as the "
               "alternative when it beats it by > 10 points on U1 on both detectors; if none passes, the level with the fewest failing clauses, unqualified")
    r2 = [(lvl, alpha) for lvl, suf, alpha, P in (LEVELS if not a.levels else levels)]
    passing = [(lvl, al) for lvl, al in r2 if not summary[lvl][0]]
    if passing:
        pick = min(passing, key=lambda t: t[1])
        out.append(f"  RECOMMENDED: {pick[0]} (alpha {pick[1]:g}, scope tree+path, with prefer3)")
        for lvl, al in passing:
            if al > pick[1]:
                gain = [num(verdict[lvl][det]["U1"][1].split("(")[1]) - num(verdict[pick[0]][det]["U1"][1].split("(")[1]) for det in DETS]
                if all(g < -10 for g in gain):
                    out.append(f"  ALTERNATIVE: {lvl} (alpha {al:g}) beats it on U1 by {', '.join(f'{-g:.0f}' for g in gain)} points")
                else:
                    out.append(f"  also passing: {lvl} (alpha {al:g}), U1 gain vs the pick {', '.join(f'{-g:+.0f}' for g in gain)} points (not > 10 on both)")
    else:
        pick = min(r2, key=lambda t: (len(summary[t[0]][0]), t[1]))
        out.append(f"  NO level passes.  Fewest failing clauses: {pick[0]} (alpha {pick[1]:g}), failing {', '.join(summary[pick[0]][0])} -> UNQUALIFIED candidate")
    os.makedirs(F, exist_ok=True)
    open(f"{F}/115r2_arms_complete.txt", "w").write("\n".join(comp_lines) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
