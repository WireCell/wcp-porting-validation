#!/usr/bin/env python3
"""doc pdhd/22 -- the SECOND truth reading the owner asked for.

    python3 d22_truth2.py <arm> [<arm> ...]

Reading A (all truth, what d21_grade.py uses): owner_review > owner_smx1 > agent verdict.
Reading B (this script, "owner + high confidence"): owner_review > owner_smx1 > the agent's
verdict ONLY when its confidence is "high"; every other agent item is EXCLUDED, not guessed.

Both are reported wherever they differ -- the owner's 2026-09-11 ruling.  It matters: in
round h21 reading B turned P1 from +13/+1 into +7/0 FP.
"""
import glob, os, json, csv, sys, collections
import uproot
IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"; X = IMG + "/pdhd/docs/scan"
base = lambda v: v[5:] if v and v.startswith("FRAG_") else v

def arm(tag):
    out = {}
    for f in sorted(glob.glob(f"{IMG}/pdhd/work/*_{tag}/tracking-pr.root")):
        evt = os.path.basename(os.path.dirname(f)).replace("_" + tag, "")
        c = uproot.open(f)["T_stm_michel"].arrays(["cluster_id", "is_stm"], library="np")
        for i in range(len(c["cluster_id"])):
            out[(evt, str(int(c["cluster_id"][i])))] = int(c["is_stm"][i])
    return out

rec = json.load(open(f"{X}/pdhd_stm_michel_smx22_verdicts.json"))
rd = lambda p: list(csv.DictReader([l for l in open(p) if not l.startswith("#")], delimiter="\t"))
POP = {"%s/%s" % (r["event"], r["cluster"]) for r in
       rd(f"{X}/smx18/pdhd_stm_michel_scan_key_p82bhoff.tsv")}

def truth(r, reading):
    if r.get("owner_review"): return base(r["owner_review"]["verdict"])
    if r.get("owner_smx1"):
        o = r["owner_smx1"]; return base(o.get("choice") or o.get("label"))
    if reading == "B" and (r.get("confidence") or "").lower() != "high":
        return None
    return base(r["verdict"])

for reading, name in (("A", "all truth (owner > smx1 > agent)"),
                      ("B", "owner + HIGH-confidence agent only")):
    print(f"\n=== reading {reading}: {name} ===")
    for tag in sys.argv[1:]:
        A = arm(tag); c = collections.Counter()
        for r in rec:
            k = r["key"]
            if k not in POP: continue
            d = A.get(tuple(k.split("/")))
            if d is None: continue
            v = truth(r, reading)
            if v is None or v in ("MESSY", "UNCLEAR"): continue
            hs = v in ("STM_MICHEL", "STM_ONLY"); cs = d == 1
            c["TP" if (hs and cs) else "FN" if hs else "FP" if cs else "TN"] += 1
        n = sum(c.values()); p = c["TP"]/max(1, c["TP"]+c["FP"]); e = c["TP"]/max(1, c["TP"]+c["FN"])
        print(f"  {tag:9s} TP {c['TP']:3d} FP {c['FP']:3d} FN {c['FN']:3d} TN {c['TN']:3d}"
              f" | purity {p:.3f} efficiency {e:.3f}  (scored {n})")
