#!/usr/bin/env python3
"""doc pdvd/115 -- apply the frozen sizing bar (figs/115_sizing_rule.txt + amendment 1) to the two replay tables
(figs/115_replay_{pdhd,pdvd}.json) and print the level selection.  Parses and compares only.

Usage: d115_sizing_verdict.py [--figs DIR] > figs/115_sizing_verdict.txt
"""
import argparse, json, os

Q1, Q2_FAR, Q2_GAP, Q3, Q4 = -0.20, 1.10, 0.05, 1.10, 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs"))
    a = ap.parse_args()
    R = {det: json.load(open(f"{a.figs}/115_replay_{det}.json")) for det in ("pdhd", "pdvd")}
    levels = [l for l in R["pdhd"]["levels"] if l != "0:tree"]
    out = []
    for det in R:
        ck = R[det]["check"]
        out.append(f"## {det}: events {R[det]['events']}; CHECK alpha=0: tree edges {100*ck['common_edges']/max(1,ck['dumped_edges']):.2f} % of dumped "
                   f"reproduced (amendment 1: judged against the tie floor, PDHD test event 90.2 %), walk cost within 0.5 % on "
                   f"{100*ck['walk_cost_close']/max(1,ck['walks']):.1f} % of {ck['walks']} walks (bar 95 %)")
    res = {}
    out.append("\n| level | det | Q1 S1 rel (bar -20 %) | Q2 far ratio (<= 1.10) | Q2 GAP changed (<= 5 %) | Q3 wig ratio (<= 1.10) | Q4 cov diff (>= -1 pt) | verdict |")
    out.append("|---|---|---|---|---|---|---|---|")
    for lab in levels:
        ok_all = True
        for det in R:
            b = R[det]["levels"]["0:tree"]; r = R[det]["levels"][lab]
            q1 = r["off"] / b["off"] - 1 if b["off"] else float("nan")
            q2f = r["far"] / b["far"] if b["far"] else float("nan")
            q2g = r["gap_changed"] / max(1, r["gap_n"])
            q3 = r["wig"] / b["wig"] if b["wig"] else float("nan")
            q4 = 100 * (r["cov"] - b["cov"])
            ok = dict(Q1=q1 <= Q1, Q2=(q2f <= Q2_FAR and q2g <= Q2_GAP), Q3=q3 <= Q3, Q4=q4 >= -Q4)
            res[(lab, det)] = dict(q1=q1, ok=ok)
            ok_all &= all(ok.values())
            out.append(f"| {lab} | {det} | {100*q1:+.1f} % {'ok' if ok['Q1'] else 'FAIL'} | {q2f:.3f} | {100*q2g:.1f} % {'ok' if ok['Q2'] else 'FAIL'} | "
                       f"{q3:.3f} {'ok' if ok['Q3'] else 'FAIL'} | {q4:+.2f} {'ok' if ok['Q4'] else 'FAIL'} | {'PASS' if all(ok.values()) else 'FAIL'} |")
        out.append(f"| {lab} | both | | | | | | {'QUALIFIES' if ok_all else 'no'} |")
    out.append("\n## selection (figs/115_sizing_rule.txt): per scope the smallest alpha passing Q1-Q4 on both detectors; else the best alpha "
               "by Q1 (mean of the two detectors) among those passing Q2 on both, named as unqualified")
    for scope in ("tree", "tree+path"):
        cands = [l for l in levels if l.split(":")[1] == scope]
        passing = [l for l in cands if all(all(res[(l, d)]["ok"].values()) for d in R)]
        if passing:
            pick = min(passing, key=lambda l: float(l.split(":")[0]))
            out.append(f"  scope {scope}: alpha {pick.split(':')[0]} QUALIFIES (smallest passing)")
        else:
            q2ok = [l for l in cands if all(res[(l, d)]["ok"]["Q2"] for d in R)]
            if not q2ok:
                out.append(f"  scope {scope}: no level passes Q2 on both detectors -> nothing built for this scope")
                continue
            pick = min(q2ok, key=lambda l: sum(res[(l, d)]["q1"] for d in R) / len(R))
            fails = sorted({k for d in R for k, v in res[(pick, d)]["ok"].items() if not v})
            out.append(f"  scope {scope}: NO level qualifies; best by Q1 among Q2-passing = alpha {pick.split(':')[0]} "
                       f"(mean Q1 {100*sum(res[(pick, d)]['q1'] for d in R)/len(R):+.1f} %; fails {', '.join(fails)}) -> built UNQUALIFIED")
    print("\n".join(out))


if __name__ == "__main__":
    main()
