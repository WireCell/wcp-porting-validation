#!/usr/bin/env python3
"""doc pdhd/25 sec 2 -- how many true stoppers does the chain accept on its OWN Bragg reading,
and then also find the Michel?  PDHD (APA0 excluded) against PDVD, on each detector's hand record.

    python3 d25_bragg_michel.py [--pdhd h23conf,h25va,...] [--pdvd p96vprod]

Definitions (owner's choice, 2026-09-12: "chain path, scored on truth"):
  hand stopper      truth verdict STM_MICHEL or STM_ONLY (FRAG_ stripped; MESSY/UNCLEAR unscored)
  hand Michel       hand stopper whose michel_kind is attached or both
  accept            is_stm == 1
  Bragg-path accept is_stm == 1 AND topology_cleared_bits == 0
                    -- the shape tests (no_bragg / shape_flat) passed on their own; P1
                    (topology_stop_evidence) did not have to clear them on the strength of the Michel
  topology-rescued  is_stm == 1 AND topology_cleared_bits != 0
  anchor-helped     Bragg-path accept with bragg_anchor_fallback or bragg_wide_fired set
  GOLDEN            hand Michel AND Bragg-path accept AND michel_found == 1

Truth: PDHD = owner_review > owner_smx1 > agent (d18_census / d23_grade precedence) on smx23,
population = the committed 303-item key.  PDVD = the merged smx1a..smx9 record's flat verdict,
population = items that HAVE a candidate in the arm (doc pdvd/93's "with a candidate" row).
An owner_review stopper with no michel_kind is excluded from Michel scoring (d21_michel_census rule).

PDHD APA assignment is doc pdhd/23 sec 7's: role-1 fit points, APA0 = x<0 AND z<231 cm;
"strict" drops any candidate with a role-1 point in APA0, "majority" drops the APA0-majority ones.

The script REFUSES to print any number unless the production arms reproduce the committed
censuses (doc pdhd/23 sec 7, doc pdvd/93 / 96).
"""
import argparse, collections, csv, glob, json, math, os, sys
import numpy as np
import uproot

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
HD_REC = os.environ.get("STM_SCAN_RECORD", IMG + "/pdhd/docs/scan/pdhd_stm_michel_smx23_verdicts.json")  # doc pdhd/25 sec 6
HD_KEY = IMG + "/pdhd/docs/scan/smx18/pdhd_stm_michel_scan_key_p82bhoff.tsv"
VD_REC = IMG + "/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_smx7_smx8_smx9_verdicts.json"
BR = ["cluster_id", "is_stm", "michel_found", "reject_bits", "topology_cleared_bits",
      "bragg_anchor_fallback", "bragg_wide_fired", "contrast", "contrast_expected", "ks_mu", "ks_flat",
      "michel_ke_best", "michel_len", "michel_conn_type", "muon_len", "plateau_med",
      "n_dead_pts", "n_profile_pts", "n_live_pts", "stop_x", "stop_y", "stop_z"]
base = lambda v: v[5:] if v and v.startswith("FRAG_") else v
STOP = ("STM_MICHEL", "STM_ONLY")
MICHEL_KINDS = ("attached", "both")


def read_arm(det, tag):
    out = {}
    for f in sorted(glob.glob(f"{IMG}/{det}/work/*_{tag}/tracking-pr.root")):
        evt = os.path.basename(os.path.dirname(f)).replace("_" + tag, "")
        u = uproot.open(f)
        if "T_stm_michel" not in [k.split(";")[0] for k in u.keys()]:
            continue                                   # an event with no candidate at all
        t = u["T_stm_michel"]
        s = t.arrays([b for b in BR if b in t.keys()], library="np")
        apas = collections.defaultdict(collections.Counter)
        if det == "pdhd":
            p = u["T_stm_michel_pts"].arrays(["cluster_id", "role", "x", "z"], library="np")
            m = p["role"] == 1
            for c, x, z in zip(p["cluster_id"][m], p["x"][m], p["z"][m]):
                apas[int(c)][(0 if x < 0 else 1) + (0 if z < 231.0 else 2)] += 1
        for i in range(len(s["cluster_id"])):
            c = int(s["cluster_id"][i])
            d = {k: s[k][i] for k in s}
            d["wide"] = int(d.get("bragg_wide_fired", 0))
            if det == "pdhd":
                h = apas.get(c)
                if not h:
                    continue                           # d23_apa.py skips these too
                d["apa_major"] = h.most_common(1)[0][0]
                d["apa_any0"] = 0 in h
            out[f"{evt}/{c}"] = d
    return out


def hd_truth(r):
    """-> (verdict, michel_kind, source)"""
    if r.get("owner_review"):
        o = r["owner_review"]; return base(o["verdict"]), o.get("michel_kind"), "owner_review"
    if r.get("owner_smx1"):
        o = r["owner_smx1"]; return base(o.get("choice") or o.get("label")), o.get("michel_kind"), "owner"
    return base(r["verdict"]), r.get("michel_kind"), "agent"


def vd_truth(r):
    return base(r["verdict"]), r.get("michel_kind"), "flat"


def items(det, arm, pop):
    """-> list of (key, verdict, kind, source, chain-dict) over the scored population"""
    A = read_arm(det, arm)
    if det == "pdhd":
        rec, tr = json.load(open(HD_REC)), hd_truth
        rd = [l for l in open(HD_KEY) if not l.startswith("#")]
        POP = {"%s/%s" % (r["event"], r["cluster"]) for r in csv.DictReader(rd, delimiter="\t")}
    else:
        rec, tr, POP = json.load(open(VD_REC)), vd_truth, None
    out, missing = [], 0
    for r in rec:
        k = r["key"]
        if POP is not None and k not in POP:
            continue
        d = A.get(k)
        if d is None:
            missing += 1
            continue
        v, kind, src = tr(r)
        if v in ("MESSY", "UNCLEAR"):
            continue
        if det == "pdhd":
            if pop == "strict" and d["apa_any0"]:
                continue
            if pop == "majority" and d["apa_major"] == 0:
                continue
        out.append((k, v, kind, src, d))
    return out, missing


def census(its):
    """-> (is_stm, michel on hand stoppers [d21_michel_census], michel on all judged [census_score 14.2])

    The two Michel definitions differ and both are in use: PDHD's graders score michel_found on
    hand STOPPERS against michel_kind (attached/both); PDVD's census_score.py sec 14.2 -- the source
    of doc pdvd/93's 144/12/20 -- scores it on ALL judged items against verdict == STM_MICHEL."""
    c = collections.Counter(); m = collections.Counter(); m2 = collections.Counter()
    for k, v, kind, src, d in its:
        hs = v in STOP; cs = int(d["is_stm"]) == 1; cm = int(d["michel_found"]) == 1
        c["TP" if hs and cs else "FN" if hs else "FP" if cs else "TN"] += 1
        hm2 = v == "STM_MICHEL"
        m2["TP" if hm2 and cm else "FN" if hm2 else "FP" if cm else "TN"] += 1
        if hs:
            if src == "owner_review" and kind is None:
                continue
            hm = kind in MICHEL_KINDS
            m["TP" if hm and cm else "FN" if hm else "FP" if cm else "TN"] += 1
    t = lambda x: (x["TP"], x["FP"], x["FN"], x["TN"])
    return t(c), t(m), t(m2)


def frac(a, n):
    if n == 0:
        return "   -   "
    p = a / n
    return f"{a:3d}/{n:3d} = {p:.3f} +- {math.sqrt(p * (1 - p) / n):.3f}"


def report(label, its):
    S = [x for x in its if x[1] in STOP]
    SM = [x for x in S if x[2] in MICHEL_KINDS and not (x[3] == "owner_review" and x[2] is None)]
    SO = [x for x in S if x[1] == "STM_ONLY"]
    acc = lambda x: int(x[4]["is_stm"]) == 1
    bragg = lambda x: acc(x) and int(x[4]["topology_cleared_bits"]) == 0
    rescued = lambda x: acc(x) and int(x[4]["topology_cleared_bits"]) != 0
    helped = lambda x: bragg(x) and (int(x[4]["bragg_anchor_fallback"]) or x[4]["wide"])
    mf = lambda x: int(x[4]["michel_found"]) == 1
    golden = lambda x: bragg(x) and mf(x)
    (TP, FP, FN, TN), (mTP, mFP, mFN, mTN), (nTP, nFP, nFN, nTN) = census(its)
    print(f"\n=== {label}")
    print(f"  is_stm census     {TP}/{FP}/{FN}/{TN}   purity {TP/max(1,TP+FP):.3f} eff {TP/max(1,TP+FN):.3f}")
    print(f"  michel census, hand stoppers vs michel_kind (PDHD graders)   "
          f"{mTP}/{mFP}/{mFN}/{mTN}   purity {mTP/max(1,mTP+mFP):.3f} eff {mTP/max(1,mTP+mFN):.3f}")
    print(f"  michel census, all judged vs verdict STM_MICHEL (census_score 14.2)   "
          f"{nTP}/{nFP}/{nFN}/{nTN}   purity {nTP/max(1,nTP+nFP):.3f} eff {nTP/max(1,nTP+nFN):.3f}")
    print(f"  hand stoppers {len(S)}  (hand Michel {len(SM)}, STM_ONLY {len(SO)})")
    print(f"  over ALL hand stoppers:")
    print(f"    accepted                       {frac(sum(map(acc, S)), len(S))}")
    print(f"    Bragg-path accept              {frac(sum(map(bragg, S)), len(S))}")
    print(f"      of which anchor-helped       {frac(sum(map(helped, S)), len(S))}")
    print(f"    topology-rescued               {frac(sum(map(rescued, S)), len(S))}")
    print(f"    GOLDEN (Bragg-path + Michel)   {frac(sum(1 for x in S if x in SM and golden(x)), len(S))}")
    print(f"  over hand STM_ONLY:")
    print(f"    Bragg-path accept              {frac(sum(map(bragg, SO)), len(SO))}")
    print(f"  over hand MICHEL items (the GOLDEN denominator):")
    print(f"    accepted                       {frac(sum(map(acc, SM)), len(SM))}")
    print(f"    Bragg-path accept              {frac(sum(map(bragg, SM)), len(SM))}")
    print(f"    topology-rescued               {frac(sum(map(rescued, SM)), len(SM))}")
    print(f"    michel_found (any accept)      {frac(sum(1 for x in SM if mf(x)), len(SM))}")
    print(f"    accepted AND michel_found      {frac(sum(1 for x in SM if acc(x) and mf(x)), len(SM))}")
    print(f"    GOLDEN                         {frac(sum(map(golden, SM)), len(SM))}")
    print(f"    GOLDEN, no anchor help         {frac(sum(1 for x in SM if golden(x) and not helped(x)), len(SM))}")
    print(f"    Michel found | Bragg-path      {frac(sum(map(golden, SM)), sum(map(bragg, SM)))}")
    # the chain's golden selection on its own: what else lands in it
    sel = [x for x in its if int(x[4]["is_stm"]) == 1 and int(x[4]["topology_cleared_bits"]) == 0 and mf(x)]
    cls = collections.Counter("hand Michel" if x in SM else x[1] for x in sel)
    print(f"  chain's own golden selection: {len(sel)} items -> {dict(cls)}")
    # the shape numbers on a common footing
    print("  shape quantities (p25 / p50 / p75):")
    for name, grp in (("hand STM_MICHEL", [x for x in S if x[1] == "STM_MICHEL"]), ("hand STM_ONLY", SO),
                      ("hand THRU", [x for x in its if x[1] == "THRU"])):
        if not grp:
            continue
        row = []
        for q in ("contrast", "ks_mu"):
            a = np.array([float(x[4][q]) for x in grp])
            row.append(f"{q} {np.percentile(a,25):.3f}/{np.percentile(a,50):.3f}/{np.percentile(a,75):.3f}")
        print(f"    {name:16s} n={len(grp):3d}  " + "   ".join(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdhd", default="h23conf")
    ap.add_argument("--pdvd", default="p96vprod")
    a = ap.parse_args()
    # ---- self-gates -------------------------------------------------------------
    # PDHD michel wants = doc pdhd/23 sec 7 (census_apa.txt, hand-stopper definition);
    # PDVD michel want = doc pdvd/93 / 96 (census_score 14.2, all-judged definition, TN not quoted)
    GATES = [("pdhd", "h23conf", "all", (96, 1, 52, 107), (69, 2, 18, 56), None),
             ("pdhd", "h23conf", "majority", (79, 1, 30, 70), (52, 2, 12, 42), None),
             ("pdhd", "h23conf", "strict", (77, 0, 28, 69), (51, 2, 12, 39), None),
             ("pdvd", "p96vprod", "all", (242, 8, 34, 262), None, (144, 12, 20))]
    # doc pdhd/25 sec 6: a later PDHD record (STM_SCAN_RECORD) moves the PDHD gates by construction.
    # D25_GATES="h23conf:strict=a/b/c/d,e/f/g/h;..." names is_stm[,michel] values derived for that record
    # by an independent census (d25_score_smx25.py).  Unset => the gates above, unchanged.
    for g in filter(None, os.environ.get("D25_GATES", "").split(";")):
        k, _, v = g.partition("=")
        if ":" not in k:
            continue                                   # e.g. d23_grade's p82bhoff entry
        garm, gpop = k.split(":")
        parts = [tuple(int(x) for x in p.split("/")) for p in v.split(",")]
        GATES = [(d, ar, po, parts[0], parts[1] if len(parts) > 1 else wm, wm2)
                 if (d == "pdhd" and ar == garm and po == gpop) else (d, ar, po, w, wm, wm2)
                 for d, ar, po, w, wm, wm2 in GATES]
    for det, arm, pop, want, wantm, wantm2 in GATES:
        its, _ = items(det, arm, pop)
        got, gotm, gotm2 = census(its)
        ok = (got == want and (wantm is None or gotm == wantm)
              and (wantm2 is None or gotm2[:3] == wantm2))
        print(f"GATE {det} {arm} {pop:8s} is_stm {got} want {want}; michel(stoppers) {gotm}"
              f"{'' if wantm is None else ' want ' + str(wantm)}; michel(all judged) {gotm2}"
              f"{'' if wantm2 is None else ' want ' + str(wantm2)} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit("self-gate FAILED -- refusing to print any number")
    # ---- report ------------------------------------------------------------------
    for arm in a.pdhd.split(","):
        for pop in ("strict", "majority", "all"):
            its, miss = items("pdhd", arm, pop)
            report(f"PDHD {arm}  APA0 {pop if pop != 'all' else 'INCLUDED (all four APAs)'}  "
                   f"(POP items with no candidate in this arm: {miss})", its)
    for arm in a.pdvd.split(","):
        its, miss = items("pdvd", arm, "all")
        report(f"PDVD {arm}  with a candidate  (record items with no candidate: {miss})", its)


if __name__ == "__main__":
    main()
