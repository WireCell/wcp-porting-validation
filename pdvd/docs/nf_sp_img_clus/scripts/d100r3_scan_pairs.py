#!/usr/bin/env python3
"""doc pdvd/100 round 3 -- prereg_round3.md Q2: which hand-scan truth pairs move between two scored QL arms.

    python3 d100r3_scan_pairs.py --base p100flip --arm p101q > ../d100/qtol_scan_pairs.txt

Reads pdvd/work/ql_scores/<tag>/scores.json (ql_agree_score.py, --truth-uid-map-tag keep).  Valid only for two arms whose
imaging is the same (QtoL moves QLMatching's picks and what clusters after them); the per-event uid-map counts of the two
score logs are compared when given (counts, not map content: each arm is scored against truth mapped onto its own clustering).  Per list (missed / phantom / unknown), the pairs
(event, cluster uid, flash time to 0.1 us) only in the base, only in the arm, and common; the newly missed and recovered
pairs by drift volume (uid >= 4000000 = top); and the exact two-sided sign test on newly missed vs recovered.
"""
import argparse, json, math, re

WORK = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work/ql_scores"


def sets(tag):
    d = json.load(open(f"{WORK}/{tag}/scores.json"))
    out = {}
    for k in ("missed_list", "phantom_list", "unknown_list"):
        out[k] = {(evt, e["uid"], round(e["time"], 1)) for evt, v in d["detail"].items() for e in v.get(k, [])}
    return d["total"], out


def umap(log):
    return [tuple(map(int, r)) for r in re.findall(r"(\d+) clusters mapped, (\d+) truth entries unmapped", open(log).read())]


def sign_p(k, n):
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--base-log")
    ap.add_argument("--arm-log")
    a = ap.parse_args()
    print(f"# doc pdvd/100 round 3 -- truth pairs moving {a.base} -> {a.arm} (work/ql_scores, uid map keep, no time map/shift)")
    if a.base_log and a.arm_log:
        mb, ma = umap(a.base_log), umap(a.arm_log)
        print(f"uid-map counts (mapped clusters, unmapped truth) per event equal in both score logs: {mb == ma} ({sum(x for x, _ in ma)} clusters mapped, "
              f"{sum(y for _, y in ma)} truth entries unmapped)")
    tb, sb = sets(a.base)
    ta, sa = sets(a.arm)
    for k in ("agree", "phantom", "missed", "unknown", "covered"):
        print(f"  total {k:8s} {a.base} {tb[k]:4d}  {a.arm} {ta[k]:4d}  diff {ta[k] - tb[k]:+d}")
    for k in ("missed_list", "phantom_list", "unknown_list"):
        B, A = sb[k], sa[k]
        print(f"{k}: distinct pairs {a.base} {len(B)} {a.arm} {len(A)}; only {a.base} {len(B - A)}, only {a.arm} {len(A - B)}, common {len(A & B)}")
    new = sorted(sa["missed_list"] - sb["missed_list"])
    rec = sorted(sb["missed_list"] - sa["missed_list"])
    vol = lambda s: (sum(1 for _, u, _ in s if u >= 4000000), sum(1 for _, u, _ in s if u < 4000000))
    print(f"newly missed on {a.arm}: {len(new)} (top {vol(new)[0]}, bottom {vol(new)[1]})")
    for x in new:
        print("   ", x)
    print(f"recovered on {a.arm}: {len(rec)} (top {vol(rec)[0]}, bottom {vol(rec)[1]})")
    for x in rec:
        print("   ", x)
    n = len(new) + len(rec)
    if n:
        print(f"sign test newly missed {len(new)} vs recovered {len(rec)} of {n} moving pairs: two-sided p = {sign_p(len(rec), n):.3f}")


if __name__ == "__main__":
    main()
