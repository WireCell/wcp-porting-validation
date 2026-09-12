#!/usr/bin/env python3
"""doc pdhd/23 sec 6 -- grade an arm against the smx23 record (smx22 + the owner's own23).

    python3 d23_grade.py <arm> [<arm> ...]

Why not d21_grade.py: that script hardcodes the smx22 path and self-gates on
p82bhoff == 61/0/86/110.  smx23 DELIBERATELY breaks that gate -- the owner's
ruling turned 028084_3/72 from THRU into a stopper (so p82bhoff now misses one
more) and 028084_12/46 into MESSY (so it leaves the scored population).  A
scorer that still passed the smx22 gate would be reading the wrong record.

Truth precedence is unchanged: owner_review > owner_smx1 > agent, FRAG_ stripped,
MESSY/UNCLEAR unscored.  Reading B additionally keeps an agent verdict only when
its own confidence field is "high".
"""
import glob, os, json, csv, sys, collections
import uproot
IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"; X = IMG + "/pdhd/docs/scan"
REC = os.environ.get("STM_SCAN_RECORD", X + "/pdhd_stm_michel_smx23_verdicts.json")
base = lambda v: v[5:] if v and v.startswith("FRAG_") else v
GATE = {"p82bhoff": (61, 0, 87, 108)}          # smx23's value, NOT smx22's 61/0/86/110

def arm(tag):
    out = {}
    for f in sorted(glob.glob(f"{IMG}/pdhd/work/*_{tag}/tracking-pr.root")):
        evt = os.path.basename(os.path.dirname(f)).replace("_" + tag, "")
        c = uproot.open(f)["T_stm_michel"].arrays(
            ["cluster_id", "is_stm", "michel_found"], library="np")
        for i in range(len(c["cluster_id"])):
            out[(evt, str(int(c["cluster_id"][i])))] = dict(
                is_stm=int(c["is_stm"][i]), michel=int(c["michel_found"][i]))
    return out

rec = json.load(open(REC))
def truth(r, reading):
    if r.get("owner_review"): return base(r["owner_review"]["verdict"])
    if r.get("owner_smx1"):
        o = r["owner_smx1"]; return base(o.get("choice") or o.get("label"))
    if reading == "B" and (r.get("confidence") or "").lower() != "high": return None
    return base(r["verdict"])
rd = lambda p: list(csv.DictReader([l for l in open(p) if not l.startswith("#")], delimiter="\t"))
POP = {"%s/%s" % (r["event"], r["cluster"]) for r in
       rd(f"{X}/smx18/pdhd_stm_michel_scan_key_p82bhoff.tsv")}

def census(A, reading):
    c = collections.Counter()
    for r in rec:
        k = r["key"]
        if k not in POP: continue
        d = A.get(tuple(k.split("/")))
        if d is None: continue
        v = truth(r, reading)
        if v is None or v in ("MESSY", "UNCLEAR"): continue
        hs = v in ("STM_MICHEL", "STM_ONLY"); cs = d["is_stm"] == 1
        c["TP" if (hs and cs) else "FN" if hs else "FP" if cs else "TN"] += 1
    return c

print(f"record: {os.path.basename(REC)}")
B = census(arm("p82bhoff"), "A")
got = (B["TP"], B["FP"], B["FN"], B["TN"])
if got != GATE["p82bhoff"]:
    sys.exit(f"INSTRUMENT FAILED: p82bhoff reads {got}, smx23 says {GATE['p82bhoff']}")
print(f"instrument validated: p82bhoff reproduces smx23's {GATE['p82bhoff']}\n")
for reading in ("A", "B"):
    print(f"=== reading {reading} ===")
    for tag in sys.argv[1:]:
        c = census(arm(tag), reading)
        TP, FP, FN, TN = c["TP"], c["FP"], c["FN"], c["TN"]
        print(f"  {tag:10s} TP {TP:3d} FP {FP:2d} FN {FN:3d} TN {TN:3d} | n {TP+FP+FN+TN}"
              f" | purity {TP/max(1,TP+FP):.3f} efficiency {TP/max(1,TP+FN):.3f}")
