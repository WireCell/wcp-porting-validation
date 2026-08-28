#!/usr/bin/env python3
"""doc pr/118 -- offline census of the two byte-neutral probes.

Joins the SHOWER_MERGE tag=ex_shower1_p2dis and tag=cont_probe lines from a
knobs-OFF debug arm against the emscan-0827 hand-scan labels, and reports:

  1. P2 census: candidates the body-distance gate would admit that the legacy
     start-segment gate rejects, split by whether they would then survive the
     downstream angle test (predicts the shower_ex1_conn3_body_dis yield).
  2. Continuity separation: for every probed shower pair, a truth class from
     the scan marks (MERGE = the scanner wants the fragment's segments inside
     the absorber; DISTINCT = the scanner reviewed the absorber and does not,
     or marked them OUT, or the event carries no marks at all so any merge is
     churn), then the (cont_frac, badrun, gap_exact) distributions per class
     and a threshold grid -- this is what pins shower_merge_relax_cont_*.
  3. Connected-components dry run over the admitted pair relation, counting
     gamma-gamma guard violations (two main-vertex conn<=2 showers in one
     component) -- the measurement that decides whether the CC formulation is
     ever promoted to a knob (pr/118 plan B3: measure only this round).

The debug arm runs with all pr/118 knobs OFF, so its reconstruction equals
the scan-time prod0825 one (pr/117 sec 4: bare-arm diffstat 0/98) and label
shower ids join reconstruction node ids directly.

Repro:
  ./scripts/pr118_probe_census.py --prepdir em_display/emprep-118dbg \
      --pairs-tsv docs/pr/pr118-cont-pairs.tsv \
      work-pr118r1-dbg-mcp1k work-pr118r1-dbg-mcp2k \
      work-pr118r1-dbg-ncpi0 work-pr118r1-dbg-nuecc48
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)
LABEL_DIR = os.path.join(SX, "em_labels", "emscan-0827")

KV = re.compile(r"(\w+)=([^\s]+)")


def parse_line(line):
    d = {}
    for k, v in KV.findall(line):
        v = v.rstrip("cm")
        try:
            d[k] = float(v) if ("." in v or "e" in v) else int(v)
        except ValueError:
            d[k] = v
    return d


def collect(roots):
    """event -> {'p2': [line-dicts], 'cont': [line-dicts]}"""
    out = {}
    for root in roots:
        for log in sorted(glob.glob(os.path.join(root, "pr_evt*", "stdout.log"))):
            ev = int(os.path.basename(os.path.dirname(log))[len("pr_evt"):])
            rec = out.setdefault(ev, {"p2": [], "cont": []})
            with open(log, errors="replace") as fh:
                for line in fh:
                    if "tag=ex_shower1_p2dis" in line:
                        rec["p2"].append(parse_line(line))
                    elif "tag=cont_probe" in line:
                        rec["cont"].append(parse_line(line))
    return out


def load_marks(ev):
    p = os.path.join(LABEL_DIR, "labels-evt%d.json" % ev)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        em = json.load(fh).get("em") or {}
    marks = {}
    for shw, mm in (em.get("marks_by_shower") or {}).items():
        ins = {int(s) for s, v in mm.items() if v == "in"}
        outs = {int(s) for s, v in mm.items() if v == "out"}
        marks[int(shw)] = (ins, outs)
    return marks


def load_members(prepdir, ev):
    p = os.path.join(prepdir, "emprep-evt%d.json" % ev)
    if not os.path.exists(p):
        return {}
    with open(p) as fh:
        prep = json.load(fh)
    return {int(n): {int(m["seg"]) for m in (e.get("members") or ())}
            for n, e in (prep.get("showers") or {}).items()}


def classify(ev, pair, marks, members):
    """MERGE / DISTINCT / UNKNOWN for one cont_probe pair."""
    K, C = pair["keep_node"], pair["cand_node"]
    F = members.get(C) or {C}
    if marks is None:
        return "UNKNOWN", 0.0        # unscanned event
    if not marks:
        return "DISTINCT", 0.0       # scanned, no marks: any merge is churn
    for L, (ins, outs) in marks.items():
        if K != L:
            continue
        if F & outs:
            return "DISTINCT", 0.0   # explicitly marked OUT of this shower
        got = len(F & ins)
        if got:
            return "MERGE", got / len(F)
        return "DISTINCT", 0.0       # absorber was scanned; fragment unwanted
    return "UNKNOWN", 0.0            # absorber not a scanned shower


def p2_census(events):
    tot = admit = survive = 0
    per_evt = {}
    for ev, rec in sorted(events.items()):
        for d in rec["p2"]:
            tot += 1
            if d.get("legacy_gate") == "FAIL" and d.get("body_gate") == "PASS":
                admit += 1
                per_evt.setdefault(ev, [0, 0])[0] += 1
                if d.get("angles_pass") == 1:
                    survive += 1
                    per_evt[ev][1] += 1
    print("== P2 census (tag=ex_shower1_p2dis) ==")
    print("candidates probed: %d" % tot)
    print("body PASS & legacy FAIL (newly admitted): %d" % admit)
    print("  ... of which survive the angle gate (predicted mergers): %d" % survive)
    for ev, (a, s) in sorted(per_evt.items()):
        print("  evt%-8d newly_admitted=%d angle_survivors=%d" % (ev, a, s))
    print()


GRID_FRAC = (0.70, 0.80, 0.85, 0.90, 0.95, 1.00)
GRID_GAP = (6, 10, 15, 20, 25, 30)
GRID_BAD = (0, 1, 2)


def qf(p):
    """Charge-presence continuity: nstep==0 (touching) counts as continuous."""
    return 1.0 if p.get("nstep", 0) == 0 else p.get("qfrac", -1)


def legacy_fires(p):
    """Approximation of the production merge_relax predicate on this pair."""
    af = p.get("angle_fold", -1)
    return (p.get("len2", 0) >= 5.0 and p.get("gap", 1e9) <= 6.0
            and 0 <= af < 15.0)


def admitted(p, frac, gap, bad, angle=15.0, min_len=5.0):
    if p.get("gap_exact", 1e9) > gap:
        return False
    if p.get("cont_frac", -1) < frac or p.get("badrun", 99) > bad:
        return False
    if p.get("len2", 0) >= min_len:
        af = p.get("angle_fold", -1)
        if af < 0 or af >= angle:
            return False
    return True


def evaluate(pairs, admit_fn):
    """Model the knob's ACTUAL firing set: gamma-gamma guard, per-fragment
    argmin (each fragment merges into at most its best admissible absorber by
    gap_exact), and net-new relative to the production merge_relax predicate.
    Returns (counts-by-truth, admitted pair list)."""
    best = {}
    for p in pairs:
        if p.get("mv1") == 1 and p.get("mv2") == 1:
            continue                       # hard gamma-gamma guard
        if not admit_fn(p):
            continue
        k = (p["event"], p["cand_node"])
        if k not in best or p["gap_exact"] < best[k]["gap_exact"]:
            best[k] = p
    cnt = {"MERGE": 0, "DISTINCT": 0, "UNKNOWN": 0}
    out = []
    for p in best.values():
        if legacy_fires(p):
            continue                       # production already merges this
        cnt[p["truth"]] += 1
        out.append(p)
    return cnt, out


def cont_census(pairs, tsv_path):
    if tsv_path:
        cols = ["event", "truth", "frac_wanted", "keep_node", "cand_node",
                "conn1", "conn2", "mv1", "mv2", "len1", "len2", "gap",
                "gap_exact", "angle_fold", "nstep", "ngood", "nbad", "badrun",
                "cont_frac", "qmed", "qfrac", "dqdx_frag", "dqdx_abs",
                "t0_frag", "t0_abs", "walked",
                "ax15_ang", "ax100_ang", "ax_d", "jx15_ang", "jx100_ang", "jx_d"]
        with open(tsv_path, "w") as fh:
            fh.write("\t".join(cols) + "\n")
            for p in pairs:
                fh.write("\t".join(str(p.get(c, "")) for c in cols) + "\n")
        print("pairs tsv: %s (%d rows)" % (tsv_path, len(pairs)))

    print("== continuity separation (tag=cont_probe) ==")
    for cls in ("MERGE", "DISTINCT", "UNKNOWN"):
        sel = [p for p in pairs if p["truth"] == cls]
        if not sel:
            print("%-9s n=0" % cls)
            continue
        for name, key in (("cont_frac", "cont_frac"), ("badrun", "badrun"),
                          ("gap_exact", "gap_exact"), ("qfrac", "qfrac")):
            vs = sorted(float(p.get(key, -1)) for p in sel)
            q = lambda f: vs[min(len(vs) - 1, int(f * len(vs)))]
            print("%-9s n=%-4d %-9s min=%.2f q25=%.2f med=%.2f q75=%.2f max=%.2f"
                  % (cls, len(sel), name, vs[0], q(.25), q(.5), q(.75), vs[-1]))
    print()

    print("== is_good_point continuity grid (guarded, argmin, net-new) ==")
    print("%-6s %-5s %-4s | net-new M / D / U" % ("frac", "gap", "bad"))
    for frac in GRID_FRAC:
        for gap in GRID_GAP:
            for bad in GRID_BAD:
                cnt, _ = evaluate(pairs, lambda p: admitted(p, frac, gap, bad))
                if sum(cnt.values()) == 0:
                    continue
                print("%-6.2f %-5d %-4d | %d / %d / %d"
                      % (frac, gap, bad, cnt["MERGE"], cnt["DISTINCT"], cnt["UNKNOWN"]))
    print()

    print("== charge-presence (qfrac) grid (guarded, argmin, net-new) ==")
    print("%-6s %-5s | net-new M / D / U   [stub-only M / D / U]" % ("qf>=", "gap<="))
    for q0 in (0.80, 0.90, 0.95, 1.00):
        for gap in (6, 8, 10, 15, 20):
            def adm(p, q0=q0, gap=gap):
                if p["gap_exact"] > gap or qf(p) < q0:
                    return False
                if p["len2"] >= 5.0:
                    return 0 <= p["angle_fold"] < 15.0
                return True
            cnt, adm_pairs = evaluate(pairs, adm)
            s = {"MERGE": 0, "DISTINCT": 0, "UNKNOWN": 0}
            for p in adm_pairs:
                if p["len2"] < 5.0:
                    s[p["truth"]] += 1
            print("%-6.2f %-5d | %d / %d / %d   [%d / %d / %d]"
                  % (q0, gap, cnt["MERGE"], cnt["DISTINCT"], cnt["UNKNOWN"],
                     s["MERGE"], s["DISTINCT"], s["UNKNOWN"]))
    print()

    if any("jx100_ang" in p for p in pairs):
        print("== absorber-AXIS cone grid (guarded, argmin, net-new) ==")
        print("axis angle = min(jx15_ang, jx100_ang) valid; -1 treated as fail")
        print("%-6s %-6s %-5s | net-new M / D / U  [stub M/D/U]  events(M)" % ("ang<", "d<", "gap<="))
        for ang in (7.5, 10, 12.5, 15, 20, 25):
            for dmax in (80, 120, 200):
                for gap in (10, 15, 30):
                    def adm(p, ang=ang, dmax=dmax, gap=gap):
                        if p["gap_exact"] > gap:
                            return False
                        aa = [a for a in (p.get("jx15_ang", -1), p.get("jx100_ang", -1)) if a >= 0]
                        if not aa or min(aa) >= ang:
                            return False
                        return 0 < p.get("jx_d", -1) < dmax
                    cnt, adm_pairs = evaluate(pairs, adm)
                    if sum(cnt.values()) == 0:
                        continue
                    s = {"MERGE": 0, "DISTINCT": 0, "UNKNOWN": 0}
                    mev = set()
                    for p in adm_pairs:
                        if p["len2"] < 5.0:
                            s[p["truth"]] += 1
                        if p["truth"] == "MERGE":
                            mev.add(p["event"])
                    print("%-6s %-6d %-5d | %d / %d / %d  [%d/%d/%d]  %s"
                          % (ang, dmax, gap, cnt["MERGE"], cnt["DISTINCT"],
                             cnt["UNKNOWN"], s["MERGE"], s["DISTINCT"],
                             s["UNKNOWN"], sorted(mev)))
        print()


def cc_dry_run(pairs, frac, gap, bad):
    """Union-find over the admitted relation; count gamma-gamma violations."""
    print("== connected-components dry run (frac>=%.2f gap<=%g badrun<=%d) =="
          % (frac, gap, bad))
    by_evt = {}
    for p in pairs:
        if admitted(p, frac, gap, bad):
            by_evt.setdefault(p["event"], []).append(p)
    viol = comps = 0
    for ev, ps in sorted(by_evt.items()):
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        mv = {}
        for p in sorted(ps, key=lambda p: (p["gap_exact"], p["keep_node"], p["cand_node"])):
            mv[p["keep_node"]] = mv.get(p["keep_node"], 0) or p["mv1"]
            mv[p["cand_node"]] = mv.get(p["cand_node"], 0) or p["mv2"]
            a, b = find(p["keep_node"]), find(p["cand_node"])
            if a != b:
                parent[a] = b
        groups = {}
        for n in mv:
            groups.setdefault(find(n), []).append(n)
        for root, ns in sorted(groups.items()):
            if len(ns) < 2:
                continue
            comps += 1
            nmv = sum(1 for n in ns if mv.get(n))
            if nmv >= 2:
                viol += 1
                print("  evt%-8d GAMMA-GAMMA VIOLATION: component %s has %d "
                      "main-vertex conn<=2 showers" % (ev, sorted(ns), nmv))
    print("multi-shower components: %d; gamma-gamma violations: %d" % (comps, viol))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepdir", required=True,
                    help="the dbg arm's own sidecars (em_display/emprep-<tag>)")
    ap.add_argument("--pairs-tsv", default=None)
    ap.add_argument("--cc", nargs=3, type=float, metavar=("FRAC", "GAP", "BAD"),
                    default=(0.90, 15, 1), help="CC dry-run operating point")
    ap.add_argument("roots", nargs="+")
    args = ap.parse_args()

    roots = [r if os.path.isabs(r) else os.path.join(SX, r) for r in args.roots]
    for r in roots:
        if not os.path.isdir(r):
            sys.exit("no such arm root: %s" % r)
    prepdir = args.prepdir if os.path.isabs(args.prepdir) else os.path.join(SX, args.prepdir)

    events = collect(roots)
    print("events with probe lines: %d (p2 lines %d, cont lines %d)\n"
          % (len(events),
             sum(len(r["p2"]) for r in events.values()),
             sum(len(r["cont"]) for r in events.values())))
    p2_census(events)

    pairs = []
    for ev, rec in sorted(events.items()):
        if not rec["cont"]:
            continue
        marks = load_marks(ev)
        members = load_members(prepdir, ev)
        for p in rec["cont"]:
            p["event"] = ev
            p["truth"], p["frac_wanted"] = classify(ev, p, marks, members)
            pairs.append(p)
    cont_census(pairs, args.pairs_tsv)
    cc_dry_run(pairs, *args.cc)


if __name__ == "__main__":
    main()
