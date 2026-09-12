#!/usr/bin/env python3
"""doc pdhd/25 sec 6 -- score the smx25 blind two-scanner re-judge against its key, build reading B, and
re-grade production and the three arms on the re-judged record with an INDEPENDENT recount.

    python3 d25_score_smx25.py --key smx25/key_smx25.tsv --round /home/xqian/tmp/h25r \
        --record smx25_record.json --reading-b-out smx25/record_readingB.json \
        [--arms h25base,h25k,h25r,h25kr]

The record is mkowner_record.py --stopper-split's output (smx23 + a review_v5 block on every tranche item);
this script never writes it.

1. Integrity: every key item has exactly two scan records, from exactly the two scanners the key assigned.
2. Outcomes per group x stratum, judged on stopper-or-not (the fold's own rule): confirmed / adopted /
   split against the smx23 agent verdict, AND against the truth in force (owner precedence), because an
   owner-labelled item cannot be overridden by an agent pass -- those are reported as calibration.
   Controls: a THRU control HOLDS when neither scan calls it a stopper; a stopper control HOLDS when both do.
3. Reading B = the worst case for the levers, fixed in preregistered.txt before any arm ran: a DECISION item
   whose two scans split is set to THRU (unless an owner call governs it); a split THRU control stays as the
   record has it.  Written to --reading-b-out.
4. An independent recount (its own reader and counting, not d23_* / d25_bragg_michel's) of is_stm and
   michel_found on the 303 population, APA0 strict / majority / all, per arm, on smx23 (cross-checked against
   the committed numbers before anything else is printed), on smx25 (reading A) and on reading B; every new
   false positive against h25base is named.  The D25_GATES strings for d23_grade.py / d25_bragg_michel.py on
   each record are printed, derived from this recount.
5. The pre-registered decision line per arm: "free on the re-judged record" iff 0 new FP on APA0 strict under
   reading B.
"""
import argparse, collections, csv, glob, json, os, sys
import uproot

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
X = IMG + "/pdhd/docs/scan"
SMX23 = X + "/pdhd_stm_michel_smx23_verdicts.json"
KEY303 = X + "/smx18/pdhd_stm_michel_scan_key_p82bhoff.tsv"
STOP = ("STM_MICHEL", "STM_ONLY")
base = lambda v: v[5:] if v and v.startswith("FRAG_") else v
# committed numbers the recount must reproduce on smx23 (doc pdhd/23 sec 6-7, doc pdhd/25 sec 0)
XCHECK = {("p82bhoff", "all"): ((61, 0, 87, 108), None),
          ("h25base", "all"): ((96, 1, 52, 107), (69, 2, 18, 56)),
          ("h25base", "majority"): ((79, 1, 30, 70), (52, 2, 12, 42)),
          ("h25base", "strict"): ((77, 0, 28, 69), (51, 2, 12, 39))}


def read_arm(tag):
    out = {}
    for f in sorted(glob.glob(f"{IMG}/pdhd/work/*_{tag}/tracking-pr.root")):
        evt = os.path.basename(os.path.dirname(f)).replace("_" + tag, "")
        u = uproot.open(f)
        if "T_stm_michel" not in [k.split(";")[0] for k in u.keys()]:
            continue
        s = u["T_stm_michel"].arrays(["cluster_id", "is_stm", "michel_found"], library="np")
        p = u["T_stm_michel_pts"].arrays(["cluster_id", "role", "x", "z"], library="np")
        per = collections.defaultdict(collections.Counter)
        for c, x, z, r in zip(p["cluster_id"], p["x"], p["z"], p["role"]):
            if r == 1:
                per[int(c)][(0 if x < 0 else 1) + (0 if z < 231.0 else 2)] += 1
        for i, c in enumerate(s["cluster_id"]):
            h = per.get(int(c))
            if not h:
                continue
            out[f"{evt}/{int(c)}"] = (int(s["is_stm"][i]), int(s["michel_found"][i]), h.most_common(1)[0][0], 0 in h)
    return out


def truth(r):
    """-> (verdict, michel_kind, owner_governed)"""
    if r.get("owner_review"):
        o = r["owner_review"]; return base(o["verdict"]), o.get("michel_kind"), "owner_review"
    if r.get("owner_smx1"):
        o = r["owner_smx1"]; return base(o.get("choice") or o.get("label")), o.get("michel_kind"), "owner_smx1"
    return base(r["verdict"]), r.get("michel_kind"), None


def recount(rec, A, pop, POP):
    c, m = collections.Counter(), collections.Counter()
    fps = []
    for r in rec:
        k = r["key"]
        if k not in POP or k not in A:
            continue
        is_stm, mf, maj, any0 = A[k]
        if (pop == "strict" and any0) or (pop == "majority" and maj == 0):
            continue
        v, kind, src = truth(r)
        if v in ("MESSY", "UNCLEAR"):
            continue
        hs = v in STOP
        cls = "TP" if hs and is_stm else "FN" if hs else "FP" if is_stm else "TN"
        c[cls] += 1
        if cls == "FP":
            fps.append(k)
        if hs and not (src == "owner_review" and kind in (None, "", "— not set —")):
            hm = (kind or "").lower() in ("attached", "both")
            m["TP" if hm and mf else "FN" if hm else "FP" if mf else "TN"] += 1
    t = lambda x: (x["TP"], x["FP"], x["FN"], x["TN"])
    return t(c), t(m), sorted(fps)


def pe(t):
    TP, FP, FN, TN = t
    return f"{TP:3d}/{FP:2d}/{FN:3d}/{TN:3d}  purity {TP/max(1,TP+FP):.3f} eff {TP/max(1,TP+FN):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--round", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--reading-b-out", required=True)
    ap.add_argument("--arms", default="h25base,h25k,h25r,h25kr")
    a = ap.parse_args()
    if os.path.exists(a.reading_b_out):
        sys.exit(f"REFUSING: {a.reading_b_out} exists (M13)")
    K = list(csv.DictReader([l for l in open(a.key) if not l.startswith("#")], delimiter="\t"))
    key = {r["key"]: r for r in K}

    # ---- 1. integrity -------------------------------------------------------------------------------
    scans = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(a.round, "v_parts", "rv5_a*", "*.json"))):
        r = json.load(open(f))
        scans[r["key"]].append(r)
    bad = []
    for k, row in key.items():
        want = set(row["scanners"].split(","))
        got = [s.get("scanner") for s in scans.get(k, [])]
        if len(got) != 2 or set(got) != want:
            bad.append(f"{k}: want {sorted(want)} got {got}")
    extra = sorted(set(scans) - set(key))
    print("==== 1. integrity ====")
    print(f"  key items {len(key)}; scanned items {len(scans)}; items not in key {extra or '-'}")
    if bad or extra:
        print("\n".join("  BAD " + b for b in bad))
        sys.exit("integrity FAILED")
    shas = collections.Counter(s.get("rubric_sha") for ss in scans.values() for s in ss)
    print(f"  every item has exactly two scans from its two assigned scanners; rubric shas {dict(shas)}")

    rec23 = {r["key"]: r for r in json.load(open(SMX23))}
    rec25 = json.load(open(a.record))
    R25 = {r["key"]: r for r in rec25}
    for k in key:
        rv = R25[k].get("review_v5")
        if not rv:
            sys.exit(f"{k}: the smx25 record carries no review_v5 block -- was it folded from this round?")

    # ---- 2. outcomes --------------------------------------------------------------------------------
    stop = lambda v: base(v) in STOP
    print("\n==== 2. outcomes (stopper-or-not) ====")
    tab = collections.defaultdict(collections.Counter)
    rows = []
    for k in sorted(key, key=lambda k: (key[k]["group"], key[k]["stratum"], k)):
        row = key[k]
        s = sorted(scans[k], key=lambda r: r["scanner"])
        c = [stop(x["verdict"]) for x in s]
        agent_prev = stop(rec23[k]["verdict"])
        tv, tkind, towner = truth(rec23[k])
        in_force = tv in STOP
        out_rec = "split" if c[0] != c[1] else ("confirmed" if c[0] == agent_prev else "adopted")
        out_truth = "split" if c[0] != c[1] else ("agrees" if c[0] == in_force else "disagrees")
        g = row["group"]
        held = None
        if g == "control_thru": held = not c[0] and not c[1]
        if g == "control_stop": held = c[0] and c[1]
        if R25[k]["review_v5"]["outcome"] != out_rec:
            sys.exit(f"{k}: the record's fold says {R25[k]['review_v5']['outcome']}, this script says {out_rec}")
        tab[(g, row["stratum"])][out_truth] += 1
        if held is not None:
            tab[(g, row["stratum"])]["held" if held else "MOVED"] += 1
        rows.append((k, g, row["stratum"], tv, towner or "agent", rec23[k].get("confidence"),
                     [f"{x['scanner']}:{x['verdict']}/{x['michel_kind']}/{x['confidence']}" for x in s], out_rec, out_truth))
    for (g, st), cn in sorted(tab.items()):
        print(f"  {g:13s} stratum {st}: " + ", ".join(f"{o} {n}" for o, n in sorted(cn.items())))
    print("\n  item                  group         st  truth(in force, source, record conf)       scans                                                       fold      vs truth")
    for k, g, st, tv, src, conf, sc, o1, o2 in rows:
        print(f"  {k:20s}  {g:13s} {st}   {tv:10s} {src:12s} {str(conf):7s}  {'  '.join(sc):70s} {o1:9s} {o2}")
    own = [(k, tv, o2) for k, g, st, tv, src, conf, sc, o1, o2 in rows if src != "agent"]
    print(f"\n  owner-governed items in the tranche (calibration; the pass cannot override them): "
          + (", ".join(f"{k} owner {tv}: {o}" for k, tv, o in own) or "none"))

    # ---- 3. reading B -------------------------------------------------------------------------------
    recB, setB = [], []
    for r in rec25:
        k = r["key"]
        if k in key and key[k]["group"] == "decision" and r["review_v5"]["outcome"] == "split" and not truth(r)[2]:
            r = dict(r, verdict="THRU", michel_kind="none", reading_b="split decision item set to THRU (doc pdhd/25 sec 6)")
            setB.append(k)
        recB.append(r)
    os.makedirs(os.path.dirname(os.path.abspath(a.reading_b_out)), exist_ok=True)
    json.dump(recB, open(a.reading_b_out, "w"), indent=1, ensure_ascii=False)
    print(f"\n==== 3. reading B: split decision items set to THRU: {' '.join(setB) or 'none'} -> {a.reading_b_out}")

    # ---- 4. recount ---------------------------------------------------------------------------------
    POP = {"%s/%s" % (r["event"], r["cluster"]) for r in
           csv.DictReader([l for l in open(KEY303) if not l.startswith("#")], delimiter="\t")}
    arms = a.arms.split(",")
    AR = {t: read_arm(t) for t in set(arms) | {"p82bhoff"}}
    rec23l = list(rec23.values())
    print("\n==== 4. independent recount ====")
    for (t, pop), (want, wantm) in XCHECK.items():
        got, gotm, _ = recount(rec23l, AR[t], pop, POP)
        ok = got == want and (wantm is None or gotm == wantm)
        print(f"  cross-check smx23 {t} {pop:8s} is_stm {got} michel {gotm} -> {'PASS' if ok else 'FAIL, want ' + str((want, wantm))}")
        if not ok:
            sys.exit("recount does not reproduce the committed smx23 numbers")
    base_fp = {}
    for lab, rec in (("smx23", rec23l), ("smx25 reading A", rec25), ("smx25 reading B", recB)):
        print(f"\n  --- {lab} ---")
        gates = []
        for pop in ("strict", "majority", "all"):
            for t in arms:
                c, m, fps = recount(rec, AR[t], pop, POP)
                if t == arms[0]:
                    base_fp[(lab, pop)] = set(fps)
                new = sorted(set(fps) - base_fp[(lab, pop)])
                print(f"    {pop:8s} {t:8s} is_stm {pe(c)}   michel {m}   new FP vs {arms[0]}: {' '.join(new) or '-'}")
                if t == arms[0]:
                    gates.append(f"h23conf:{pop}=" + "/".join(map(str, c)) + "," + "/".join(map(str, m)))
        p82 = recount(rec, AR["p82bhoff"], "all", POP)[0]
        print(f"    D25_GATES=\"p82bhoff={'/'.join(map(str, p82))};{';'.join(gates)}\"")

    # ---- 5. decision line ---------------------------------------------------------------------------
    print("\n==== 5. pre-registered decision: free on the re-judged record iff 0 new FP on APA0 strict under reading B ====")
    for t in arms[1:]:
        c, m, fps = recount(recB, AR[t], "strict", POP)
        new = sorted(set(fps) - base_fp[("smx25 reading B", "strict")])
        print(f"  {t:8s} strict reading B {pe(c)}; new FP {' '.join(new) or 'none'} -> "
              f"{'FREE on the re-judged record' if not new else 'NOT free'}")


if __name__ == "__main__":
    main()
