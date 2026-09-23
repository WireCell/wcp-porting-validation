#!/usr/bin/env python3
"""doc qlmatch/33 sec 5 -- merge the round-2 blind verdicts into the TARGET (rules: d33/prereg.md).

Per sheet decision: a letter at med/high = ('pos', t); tie:X,Y at med/high = ('tie', {t..}); none at med/high =
('none',); anything at low confidence, or unsure = ('unsure',).  Times are control-frame flash times (every sheet is
drawn from the control dump).

Per mover (its LOOKS primary sheets, plus an adjudication sheet if one was issued; duplicates never enter the truth):
  - resolved  if at least one decision is committed and no two committed decisions contradict;
      compatible pairs: pos t & pos t; pos t & tie S with t in S (-> pos t); tie S & tie S (-> tie S); none & none.
  - contradiction: with an adjudication look, the side the adjudicator is compatible with wins if it is exactly one
    side; otherwise unresolved.
  - ties are resolved-NEUTRAL: reported separately, dropped from the paired score.
Outputs (--out): truth.jsonl (evt, idx, uid, kind pos|none|tie, times), merge_report.txt, and, with --adjudicate-out,
the list of contradicted movers to send to an adjudication wave.

    python3 d33_merge_record.py --key /home/xqian/tmp/p33/scan_key_r2/key.json --scan-root /home/xqian/tmp/p33/scan/r2 \
        --out ../d33/scan_r2 [--r1-key /home/xqian/tmp/p32/scan_key_r1q/key.json --r1-scan-root /home/xqian/tmp/p32/scan/r1q \
        --time-map ../d32/time_map_ctl_to_q32ti.json]
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict

TOL = 0.5


def load_verdicts(root):
    ver = {}
    for p in sorted(glob.glob(os.path.join(root, "wave*", "verdicts.jsonl"))):
        for ln in open(p):
            if ln.strip():
                d = json.loads(ln)
                ver[d["id"]] = d
    return ver


def decision(v, letters, to_ctl=lambda t: [t]):
    """-> ('pos', t) | ('tie', frozenset) | ('none',) | ('unsure',); times mapped to the control frame."""
    if v is None or v["pick"] == "unsure" or v["conf"] == "low":
        return ("unsure",)
    if v["pick"] == "none":
        return ("none",)
    if v["pick"].startswith("tie:"):
        return ("tie", frozenset(round(x, 3) for L in v["pick"][4:].split(",") for x in to_ctl(letters[L]["time"])))
    ts = to_ctl(letters[v["pick"]]["time"])
    return ("pos", round(ts[0], 3)) if len(ts) == 1 else ("tie", frozenset(round(x, 3) for x in ts))


def near(t, S):
    return any(abs(t - x) <= TOL for x in S)


def combine(d1, d2):
    """two committed decisions -> merged decision, or None if they contradict."""
    k1, k2 = d1[0], d2[0]
    if k1 == "none" or k2 == "none":
        return d1 if k1 == k2 else None
    if k1 == "pos" and k2 == "pos":
        return d1 if abs(d1[1] - d2[1]) <= TOL else None
    if k1 == "tie" and k2 == "tie":
        same = all(near(t, d2[1]) for t in d1[1]) and all(near(t, d1[1]) for t in d2[1])
        return d1 if same else None
    p, t = (d1, d2) if k1 == "pos" else (d2, d1)
    return p if near(p[1], t[1]) else None


def resolve(decs, adj=None):
    """-> (status, decision) status in resolved | unresolved:unsure | unresolved:contradiction"""
    com = [d for d in decs if d[0] != "unsure"]
    if not com:
        return "unresolved:unsure", None
    m = com[0]
    for d in com[1:]:
        m = combine(m, d)
        if m is None:
            break
    if m is not None:
        return "resolved", m
    if adj is not None and adj[0] != "unsure":
        sides = [d for d in com if combine(d, adj) is not None]
        if len({str(s) for s in sides}) == 1:
            return "resolved", combine(sides[0], adj)
    return "unresolved:contradiction", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--scan-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adjudicate-out")
    ap.add_argument("--key-extra", help="adjudication key (d33_adjudicate_items.py)")
    ap.add_argument("--r1-key")
    ap.add_argument("--r1-scan-root")
    ap.add_argument("--time-map")
    a = ap.parse_args()
    key = json.load(open(a.key))["sheets"]
    if a.key_extra:
        key += json.load(open(a.key_extra))["sheets"]
    ver = load_verdicts(a.scan_root)
    os.makedirs(a.out, exist_ok=True)
    rep = []
    P = lambda *x: rep.append(" ".join(str(y) for y in x))
    shown = [s for s in key if s["letters"]]
    P(f"# sheets {len(key)} (rendered {len(shown)}); verdicts {len(ver)}; missing "
      f"{sum(1 for s in shown if s['id'] not in ver)}")
    kinds = Counter()
    for s in shown:
        v = ver.get(s["id"])
        if v:
            kinds[(v["pick"].split(":")[0] if v["pick"] in ("none", "unsure") or v["pick"].startswith("tie")
                   else "letter", v["conf"])] += 1
    P("# picks:", dict(sorted(kinds.items())))
    dec = {s["id"]: decision(ver.get(s["id"]), s["letters"]) for s in shown}

    # ---- calibration vs the owner record ----
    cal = defaultdict(Counter)
    rows = []
    for s in shown:
        if s["kind"] != "calib":
            continue
        d = dec[s["id"]]
        drawn = [x["time"] for x in s["letters"].values()]
        pos = [o["time"] for o in s["owner"] if o["positive"] and near(o["time"], drawn)]
        neg = [o["time"] for o in s["owner"] if not o["positive"]]
        if d[0] == "unsure":
            res = "unsure"
        elif d[0] == "none":
            res = "agree" if not pos else "disagree"
        elif d[0] == "tie":
            res = "tie"
        elif pos:
            res = "agree" if near(d[1], pos) else "disagree"
        elif near(d[1], neg):
            res = "disagree"
        else:
            res = "not-comparable"
        for k in (s["owner_source"], "all"):
            cal[k][res] += 1
        rows.append((s["id"], s["evt"], s["uid"], s["owner_source"], d, pos, neg, res))
    P("\n## scanner calibration vs the owner record (committed verdicts on owner-judged candidates)")
    for src in sorted(cal):
        c = cal[src]
        n = c["agree"] + c["disagree"]
        P(f"{src:>18}: agree {c['agree']} disagree {c['disagree']} -> "
          f"{100 * c['agree'] / n if n else float('nan'):.1f} %  (unsure {c['unsure']}, tie {c['tie']}, "
          f"not-comparable {c['not-comparable']})")
    for r in rows:
        if r[-1] == "disagree":
            P(f"   disagree: {r[0]} evt{r[1]} uid{r[2]} {r[3]}: scanner {r[4]} | owner pos {r[5]} neg {r[6]}")

    # ---- duplicate consistency ----
    looks = defaultdict(dict)
    adj = {}
    for s in shown:
        if s["kind"] == "mover":
            looks[(s["evt"], s["uid"])][s["look"]] = s
        elif s["kind"] == "adj":
            adj[(s["evt"], s["uid"])] = s
    dc = Counter()
    for s in shown:
        if s["kind"] != "dup":
            continue
        o = looks[(s["evt"], s["uid"])].get(s["dup_of_look"])
        if o is None:
            continue
        d1, d2 = dec[o["id"]], dec[s["id"]]
        if "unsure" in (d1[0], d2[0]):
            dc["one unsure"] += 1
        else:
            dc["compatible" if combine(d1, d2) is not None else "contradict"] += 1
    P("\n## duplicate-sheet consistency (different scanners, same sheet):", dict(dc))

    # ---- the target ----
    for s in key:
        if s["kind"] == "mover":
            looks.setdefault((s["evt"], s["uid"]), {})
    out = Counter()
    truth, contra = [], []
    per_parity = defaultdict(Counter)
    for (evt, uid), L in sorted(looks.items()):
        decs = [dec[s["id"]] for s in L.values() if s["id"] in dec]
        ad = dec.get(adj[(evt, uid)]["id"]) if (evt, uid) in adj else None
        st, d = resolve(decs, ad)
        tag = st if st != "resolved" else "resolved:" + d[0]
        out[tag] += 1
        idx = next(iter(L.values()))["idx"] if L else None
        per_parity["odd" if idx is not None and idx % 2 else "even"][tag] += 1
        if st == "resolved":
            truth.append(dict(evt=evt, idx=idx, uid=uid, kind=d[0],
                              times=[d[1]] if d[0] == "pos" else sorted(d[1]) if d[0] == "tie" else []))
        elif st == "unresolved:contradiction":
            contra.append(dict(evt=evt, uid=uid, looks=[s["id"] for s in L.values()]))
    nm = len(looks)
    nres = sum(v for k, v in out.items() if k.startswith("resolved"))
    P(f"\n## target over {nm} movers: {dict(sorted(out.items()))}")
    P(f"   resolved {nres} = {100 * nres / max(nm, 1):.1f} % (prereg minimum 60 %); of which tie "
      f"{out['resolved:tie']} ({100 * out['resolved:tie'] / max(nm, 1):.1f} %, neutral)")
    for par in ("odd", "even"):
        P(f"   {par} events: {dict(sorted(per_parity[par].items()))}")
    with open(os.path.join(a.out, "truth.jsonl"), "w") as fh:
        for e in truth:
            fh.write(json.dumps(e) + "\n")
    if a.adjudicate_out:
        with open(a.adjudicate_out, "w") as fh:
            json.dump(contra, fh, indent=1)
        P(f"   contradicted movers for adjudication: {len(contra)} -> {a.adjudicate_out}")

    # ---- post hoc: round-1 verdicts re-merged under this rule (no flip weight) ----
    if a.r1_key:
        tmap = json.load(open(a.time_map))["events"]
        k1 = json.load(open(a.r1_key))["sheets"]
        v1 = load_verdicts(a.r1_scan_root)
        r1 = defaultdict(list)
        for s in k1:
            if s["kind"] != "mover" or not s["letters"]:
                continue
            pairs = tmap.get(str(s["evt"]), [])
            to_ctl = (lambda t, pairs=pairs: [t0 for t0, t1 in pairs if abs(t - t1) <= 1e-3] or [t]) \
                if s["light"] == "cand" else (lambda t: [t])
            r1[(s["evt"], s["uid"])].append(decision(v1.get(s["id"]), s["letters"], to_ctl))
        o1 = Counter()
        agree = Counter()
        tr2 = {(e["evt"], e["uid"]): e for e in truth}
        for k, decs in r1.items():
            st, d = resolve(decs)
            o1[st if st != "resolved" else "resolved:" + d[0]] += 1
            if st == "resolved" and d[0] == "pos" and k in tr2 and tr2[k]["kind"] == "pos":
                agree["same" if abs(tr2[k]["times"][0] - d[1]) <= TOL else "different"] += 1
        n1 = sum(v for kk, v in o1.items() if kk.startswith("resolved"))
        P(f"\n## POST HOC: round-1 verdicts under the round-2 rule: {dict(sorted(o1.items()))}; resolved {n1}/"
          f"{len(r1)} = {100 * n1 / max(len(r1), 1):.1f} %")
        P(f"   round-1 vs round-2 positive truth on the same mover (reproducibility): {dict(agree)}")
    with open(os.path.join(a.out, "merge_report.txt"), "w") as fh:
        fh.write("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
