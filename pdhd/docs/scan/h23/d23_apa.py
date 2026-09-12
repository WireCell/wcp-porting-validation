#!/usr/bin/env python3
"""doc pdhd/23 sec 7 -- purity / efficiency per APA, and with APA0 excluded.

    python3 d23_apa.py <arm> [<arm> ...]        (default: p82bhoff h23conf)

APA0 on PDHD has a known hardware problem, so the owner asked for the numbers on
the three healthy APAs.

APA convention is doc pdhd/04 sec 6.2's, not invented here:
    face 0 = x < 0  (APA 0, 2)        face 1 = x > 0  (APA 1, 3)
    z < 231 cm      (APA 0, 1)        z > 231 cm      (APA 2, 3)
  =>  APA0 = x < 0 AND z < 231 cm.
This matches smgeom.py's sensvol for apa0 (x -352..0 cm, z 0..230 cm).

A candidate is assigned to the APA holding MOST of its role-1 (muon chain) points
-- doc 04's own rule.  Crucially this reads the FIT, not the verdict: classifying
by the chain's stop_x/y/z would be circular for the misses, where no stop was
declared.  A strict variant (drop any candidate with ANY point in APA0) is
reported alongside, because a muon may cross APAs.
"""
import uproot, glob, os, json, csv, sys, collections
import numpy as np
IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"; X = IMG + "/pdhd/docs/scan"
base = lambda v: v[5:] if v and v.startswith("FRAG_") else v
ARMS = sys.argv[1:] or ["p82bhoff", "h23conf"]

def apa_of(x, z):
    return (0 if x < 0 else 1) + (0 if z < 231.0 else 2)

def arm(tag):
    """-> {(evt,cluster): (is_stm, majority_apa, set_of_apas, michel_found)}"""
    out = {}
    for f in sorted(glob.glob(f"{IMG}/pdhd/work/*_{tag}/tracking-pr.root")):
        evt = os.path.basename(os.path.dirname(f)).replace("_" + tag, "")
        r = uproot.open(f)
        s = r["T_stm_michel"].arrays(["cluster_id", "is_stm", "michel_found"], library="np")
        p = r["T_stm_michel_pts"].arrays(["cluster_id", "role", "x", "z"], library="np")
        m = p["role"] == 1
        cid, px, pz = p["cluster_id"][m], p["x"][m], p["z"][m]
        per = collections.defaultdict(collections.Counter)
        for c, xx, zz in zip(cid, px, pz):
            per[int(c)][apa_of(xx, zz)] += 1
        for i, c in enumerate(s["cluster_id"]):
            h = per.get(int(c))
            if not h: continue
            out[(evt, str(int(c)))] = (int(s["is_stm"][i]), h.most_common(1)[0][0], set(h),
                                       int(s["michel_found"][i]))
    return out

rec = json.load(open(os.environ.get("STM_SCAN_RECORD", f"{X}/pdhd_stm_michel_smx23_verdicts.json")))  # doc pdhd/25 sec 6: env names a later record
def truth(r):
    if r.get("owner_review"): return base(r["owner_review"]["verdict"])
    if r.get("owner_smx1"):
        o = r["owner_smx1"]; return base(o.get("choice") or o.get("label"))
    return base(r["verdict"])
rd = lambda p: list(csv.DictReader([l for l in open(p) if not l.startswith("#")], delimiter="\t"))
POP = {"%s/%s" % (r["event"], r["cluster"]) for r in rd(f"{X}/smx18/pdhd_stm_michel_scan_key_p82bhoff.tsv")}

def fmt(c):
    TP, FP, FN, TN = c["TP"], c["FP"], c["FN"], c["TN"]
    n = TP + FP + FN + TN
    return (f"{TP:3d}/{FP:2d}/{FN:3d}/{TN:3d}  n={n:3d}  "
            f"purity {TP/max(1,TP+FP):.3f}  efficiency {TP/max(1,TP+FN):.3f}")

for tag in ARMS:
    A = arm(tag)
    buckets = {k: collections.Counter() for k in ("all", "no_apa0", "strict", 0, 1, 2, 3)}
    mixed = 0
    for r in rec:
        k = r["key"]
        if k not in POP: continue
        d = A.get(tuple(k.split("/")))
        if d is None: continue
        v = truth(r)
        if v in ("MESSY", "UNCLEAR"): continue
        is_stm, maj, apaset, _mf = d
        if len(apaset) > 1: mixed += 1
        hs = v in ("STM_MICHEL", "STM_ONLY"); cs = is_stm == 1
        cls = "TP" if (hs and cs) else "FN" if hs else "FP" if cs else "TN"
        buckets["all"][cls] += 1
        buckets[maj][cls] += 1
        if maj != 0: buckets["no_apa0"][cls] += 1
        if 0 not in apaset: buckets["strict"][cls] += 1
    print(f"=== {tag} ===   (candidates spanning >1 APA: {mixed})")
    print(f"  all four APAs            {fmt(buckets['all'])}")
    print(f"  APA0 EXCLUDED (majority) {fmt(buckets['no_apa0'])}")
    print(f"  APA0 EXCLUDED (strict)   {fmt(buckets['strict'])}")
    for a in (0, 1, 2, 3):
        print(f"    APA{a} only             {fmt(buckets[a])}")
    print()


# ---- the Michel side, same APA buckets -------------------------------------
# michel_found vs the hand michel_kind, on HAND STOPPERS only; an owner stopper
# carrying no kind is EXCLUDED, never scored "no Michel" (d21_michel_census.py's rule).
def kind_of(r):
    if r.get("owner_review"): return r["owner_review"].get("michel_kind")
    if r.get("owner_smx1"): return r["owner_smx1"].get("michel_kind")
    return r.get("michel_kind")

print("=" * 66)
print("MICHEL SIDE (michel_found vs hand michel_kind, hand stoppers only)")
for tag in ARMS:
    A = arm(tag)
    B = {k: collections.Counter() for k in ("all", "no_apa0", "strict", 0, 1, 2, 3)}
    excl = 0
    for r in rec:
        k = r["key"]
        if k not in POP: continue
        d = A.get(tuple(k.split("/")))
        if d is None: continue
        v = truth(r)
        if v in ("MESSY", "UNCLEAR"): continue
        if v not in ("STM_MICHEL", "STM_ONLY"): continue      # hand stoppers only
        kind = kind_of(r)
        if r.get("owner_review") and kind in (None, "", "\u2014 not set \u2014"):
            excl += 1; continue
        is_stm, maj, apaset, mf = d
        hm = (kind or "").lower() in ("attached", "both"); cm = mf == 1
        cls = "TP" if (hm and cm) else "FN" if hm else "FP" if cm else "TN"
        B["all"][cls] += 1; B[maj][cls] += 1
        if maj != 0: B["no_apa0"][cls] += 1
        if 0 not in apaset: B["strict"][cls] += 1
    print(f"\n=== {tag} ===   (owner stoppers with no kind, excluded: {excl})")
    print(f"  all four APAs            {fmt(B['all'])}")
    print(f"  APA0 EXCLUDED (majority) {fmt(B['no_apa0'])}")
    print(f"  APA0 EXCLUDED (strict)   {fmt(B['strict'])}")
    for a_ in (0, 1, 2, 3):
        print(f"    APA{a_} only             {fmt(B[a_])}")
