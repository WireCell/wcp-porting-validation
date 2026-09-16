#!/usr/bin/env python3
"""doc pdvd/104 sec 5 -- fold and read the symmetric check own103h2 exactly as figs/104_pred_amend7.txt sec 5 and 6.
Written before any own103h2 label existed.

    D103_PDHD_RECORD=<smx27> python3 d104_tpmover_score.py --new-record <smx28> --owner-record <own103h> \
        --set /home/xqian/tmp/d104/own/set_pdhd_tpmover \
        --labels $IMG/pdhd/work/stm_michel_labels/own103h2/labels.json --record-out <own103h2 record>

Fork by duplication of d103_tpmover_score.py.  Two things change beyond the tag, and both come from PDHD's different
Michel truth: on PDVD a Michel is the VERDICT STM_MICHEL, on PDHD it is a stopper whose michel_kind is attached or
both, with amendment 5's rule that an owner STM_ONLY carrying no kind is Michel-NEGATIVE rather than excluded
(d103_union_grade.population, the det == "pdhd" branch).

  * Rather than re-implement those rules, this script folds first and then asks the grader: an item is
    relabel-NEGATIVE if it is still in the post-fold Michel population and its m_truth is False, and REMOVED if it
    left that population at all -- whether by MESSY / UNCLEAR, by ceasing to be a stopper, or by the exclusion rule.
    One definition, the grader's, so the rates and the grade can never disagree.
  * is_stm is reported but not projected: amendment 7 sec 2 draws no is_stm mover, so there is no rate to project.
    Both is_stm metrics still have to clear the bar for the reading to be D1 (amendment 7 sec 6).

Refuses an existing record (M13).
"""
import argparse, csv, hashlib, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d103_union_grade as U
import d104_tpmover_set as TS

LIMIT = -0.020
UNSET = (None, "", "None", "— not set —")
STOP = ("STM_MICHEL", "STM_ONLY")


def base(v):
    return (v or "").replace("FRAG_", "")


def cp68(x, n):
    """68 % Clopper-Pearson interval by bisection on the binomial tail (no scipy dependency)"""
    if n == 0:
        return (0.0, 1.0)
    def tail_ge(p):
        return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(x, n + 1))
    def tail_le(p):
        return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(0, x + 1))
    def solve(f, target, increasing):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if (f(mid) < target) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    lower = 0.0 if x == 0 else solve(tail_ge, 0.16, True)
    upper = 1.0 if x == n else solve(tail_le, 0.16, False)
    return lower, upper


def pe(tp, fp, fn):
    return tp / max(1e-12, tp + fp), tp / max(1e-12, tp + fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-record", required=True)
    ap.add_argument("--owner-record", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--record-out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.record_out):
        sys.exit(f"REFUSING: {a.record_out} exists (M13)")
    raw = open(a.labels, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    J = json.loads(raw)
    L = J["labels"]
    rows = {r["key"]: r for r in csv.DictReader(open(a.set + "/items.tsv"), delimiter="\t")}
    if set(L) - set(rows):
        sys.exit(f"labels name keys outside the set: {sorted(set(L) - set(rows))}")
    nrev = sum(1 for x in L.values() if x.get("revealed_before_label"))
    print(f"# doc pdvd/104 {J.get('tag')} (owner, pdhd): {len(L)} of {len(rows)} items labelled; labels sha256 "
          f"{sha[:12]}; revealed before labelling {nrev} of {len(L)}; rule figs/104_pred_amend7.txt")

    # pre-fold context: the strata are defined on own103h > smx27 > smx28 (amendment 7 sec 3)
    T, R, G, conf, owner = TS.context(a.new_record, a.owner_record)
    M = TS.movers(T, R, G, owner)
    S = TS.strata(M)
    cls = lambda v: "stopper" if base(v) in STOP else ("unjudged" if base(v) in ("MESSY", "UNCLEAR", "") else "non-stopper")

    print("\n== 1. each item")
    for k, r in sorted(rows.items(), key=lambda kv: int(kv[1]["scan_id"])):
        x = L.get(k)
        o = f"{x['label']}/{x.get('michel_kind')}" if x else "UNLABELLED"
        print(f"  {r['scan_id']:>2s} {k:14s} {r['role']:7s} movers [{r['movers']}] A0 [{r['A0']}] A1 [{r['A1']}] | "
              f"label {r['label_verdict']}/{r['label_michel_kind']} ({r['label_source']}, {r['label_confidence']}) -> OWNER {o}")

    # fold: own103h2 above everything, then ask the grader what the fold did
    rec = []
    for k, x in L.items():
        r = rows[k]
        omk = None if x.get("michel_kind") in UNSET else x.get("michel_kind")
        rec.append(dict(key=k, verdict=x["label"], michel_kind=omk, confidence="owner", source=f"{J.get('tag')} (owner)",
                        owner_review=dict(verdict=x["label"], michel_kind=omk, notes=x.get("notes") or "",
                                          tag=J.get("tag"), question="doc pdvd/104 figs/104_pred_amend7.txt: symmetric check"),
                        scan_id=int(r["scan_id"]), role=r["role"], movers=r["movers"],
                        prior_label=f"{r['label_verdict']}/{r['label_michel_kind']} ({r['label_source']}, {r['label_confidence']})",
                        revealed_before_label=bool(x.get("revealed_before_label")),
                        pin=x.get("pin") if (x.get("pin") or {}).get("placed") else None, labels_sha256=sha))
    rec.sort(key=lambda it: it["scan_id"])
    T2 = dict(T)
    for r in rec:
        T2[r["key"]] = U.row_truth(r, "own103h2")
    G2 = U.population("pdhd", T2, R, stm_only_unset_negative=True)
    inm, mt2 = set(G2["mpop"]), G2["m_truth"]

    print("\n== 2. relabel rates per stratum (labelled Michel TP-movers), on the grader's own post-fold definitions")
    rate = {}
    for s in ("A1", "A0"):
        ks = [k for k, r in rows.items() if r["role"] == s and f"{s}:michel" in r["movers"].split(",") and k in L]
        neg = [k for k in ks if k in inm and not mt2[k]]
        rm = [k for k in ks if k not in inm]
        n = len(ks)
        rate[s] = (len(neg) / n if n else 0.0, len(rm) / n if n else 0.0)
        print(f"  stratum {s}: labelled {n}; relabel-negative {len(neg)} ({rate[s][0]:.2f}, 68 % "
              f"[{cp68(len(neg), n)[0]:.2f}, {cp68(len(neg), n)[1]:.2f}]); removal {len(rm)} ({rate[s][1]:.2f}, "
              f"68 % [{cp68(len(rm), n)[0]:.2f}, {cp68(len(rm), n)[1]:.2f}]); TP-movers in the stratum {len(S[s])}")
        if neg:
            print(f"    relabel-negative: {sorted(neg)}")
        if rm:
            print(f"    removed:          {sorted(rm)}")

    print("\n== 3. controls (amendment 7 sec 6: >= 2 of 4 stopper-class changes -> not settled)")
    ctl = [k for k, r in rows.items() if r["role"] == "control" and k in L]
    sc = [k for k in ctl if cls(rows[k]["label_verdict"]) != cls(L[k]["label"])]
    mc = [k for k in ctl if k not in sc and base(rows[k]["label_verdict"]) != base(L[k]["label"])]
    ctl_ok = len(sc) < 2
    print(f"  stopper class changed {len(sc)} of {len(ctl)} {sorted(sc)}; "
          f"Michel call changed with the class held {len(mc)} {sorted(mc)}")

    print("\n== 4. folded grade and projection (amendment 7 sec 5, 6)")
    ok_fold = ok_proj = True
    for i, (m, P, TR) in enumerate((("is_stm", G2["pop"], G2["stm_truth"]), ("michel", G2["mpop"], G2["m_truth"]))):
        c = {}
        for lab in ("A0", "A1"):
            ch = {k: v[i] for k, v in R[lab].items()}
            c[lab] = [sum(1 for k in P if TR[k] and ch.get(k, 0)), sum(1 for k in P if not TR[k] and ch.get(k, 0)),
                      sum(1 for k in P if TR[k] and not ch.get(k, 0))]
        f = {lab: pe(*c[lab]) for lab in c}
        d_fold = (f["A1"][0] - f["A0"][0], f["A1"][1] - f["A0"][1])
        ok_fold &= min(d_fold) >= LIMIT
        print(f"  {m:6s} folded:    A0 TP {c['A0'][0]} FP {c['A0'][1]} FN {c['A0'][2]} purity {f['A0'][0]:.3f} eff "
              f"{f['A0'][1]:.3f} | A1 TP {c['A1'][0]} FP {c['A1'][1]} FN {c['A1'][2]} purity {f['A1'][0]:.3f} eff "
              f"{f['A1'][1]:.3f} | A1 - A0 purity {d_fold[0]:+.3f} efficiency {d_fold[1]:+.3f}")
        if m == "is_stm":
            print("  is_stm projected: not applicable -- amendment 7 sec 2 draws no is_stm mover; "
                  "the folded numbers above are the ones that must clear the bar")
            continue
        p = {lab: [float(x) for x in c[lab]] for lab in c}
        for s, other in (("A1", "A0"), ("A0", "A1")):
            unrev = [k for k in S[s] if f"{s}:michel" in M[k] and k not in L]
            r_neg, r_rm = rate[s]
            n_neg, n_rm = r_neg * len(unrev), r_rm * len(unrev)
            p[s][0] -= n_neg + n_rm
            p[s][1] += n_neg
            p[other][2] -= n_neg + n_rm
            print(f"  michel stratum {s}: unreviewed TP-movers {len(unrev)}; projected relabel-negative {n_neg:.1f}, "
                  f"removal {n_rm:.1f}")
        q = {lab: pe(*p[lab]) for lab in p}
        d_proj = (q["A1"][0] - q["A0"][0], q["A1"][1] - q["A0"][1])
        ok_proj &= min(d_proj) >= LIMIT
        print(f"  {m:6s} projected: A0 purity {q['A0'][0]:.3f} eff {q['A0'][1]:.3f} | A1 purity {q['A1'][0]:.3f} eff "
              f"{q['A1'][1]:.3f} | A1 - A0 purity {d_proj[0]:+.3f} efficiency {d_proj[1]:+.3f}")
    reading = ("not settled (controls)" if not ctl_ok else "D1" if ok_fold and ok_proj else
               "not settled (projection fails; propose a wider review of the failing stratum)" if ok_fold else "D2")
    print(f"\n== reading: {reading}")
    print("   (a pass is reported to the owner with the numbers; amendment 7 sec 6: it is not itself authority to "
          "change a production default)")

    json.dump(rec, open(a.record_out, "w"), indent=1)
    print(f"\n== wrote {a.record_out}: {len(rec)} items")


if __name__ == "__main__":
    main()
