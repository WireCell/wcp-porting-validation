#!/usr/bin/env python3
"""doc pdhd/22 -- what is LEFT on the post-flip baseline, and what P1 could act on.

    python3 d22_bits.py [<arm>]        (default h22base)

Round h21 sized P1 against the PRE-flip baseline (86 missed stoppers).  The flip
recovered 19 of those, so P1's eligible population and its null floor have BOTH
moved.  Re-sizing on the wrong pool is the error that inverted this campaign's
P1 conclusion once already, so this recomputes from the primary source.

Self-gate: refuses to print unless the arm reproduces its committed census.
Truth precedence is d21_grade.py's: owner_review > owner_smx1 > agent.
"""
import glob, os, json, csv, sys, collections
import uproot

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
X = IMG + "/pdhd/docs/scan"
BITS = ["no_chain", "stop_unmatched", "no_bragg", "shape_flat", "not_muon_pid",
        "continuation", "stop_near_boundary", "vertex_hadron", "short",
        "profile_sparse", "plateau_off_mip", "stop_into_dead", "cluster_not_track",
        "profile_geometry"]
base = lambda v: v[5:] if v and v.startswith("FRAG_") else v
bset = lambda rb: {n for i, n in enumerate(BITS) if int(rb) >> i & 1}

ARM = sys.argv[1] if len(sys.argv) > 1 else "h22base"
EXPECT = {"h22base": (80, 0, 67, 110), "h22g": (82, 1, 65, 109), "h22c": (83, 2, 64, 108)}


def arm(tag):
    out = {}
    for f in sorted(glob.glob(f"{IMG}/pdhd/work/*_{tag}/tracking-pr.root")):
        evt = os.path.basename(os.path.dirname(f)).replace("_" + tag, "")
        c = uproot.open(f)["T_stm_michel"].arrays(
            ["cluster_id", "is_stm", "michel_found", "reject_bits"], library="np")
        for i in range(len(c["cluster_id"])):
            out[(evt, str(int(c["cluster_id"][i])))] = dict(
                is_stm=int(c["is_stm"][i]), michel=int(c["michel_found"][i]),
                bits=bset(c["reject_bits"][i]))
    return out


rec = json.load(open(f"{X}/pdhd_stm_michel_smx22_verdicts.json"))
def truth(r):
    if r.get("owner_review"):
        return base(r["owner_review"]["verdict"])
    if r.get("owner_smx1"):
        o = r["owner_smx1"]; return base(o.get("choice") or o.get("label"))
    return base(r["verdict"])

rd = lambda p: list(csv.DictReader([l for l in open(p) if not l.startswith("#")], delimiter="\t"))
POP = {"%s/%s" % (r["event"], r["cluster"]) for r in
       rd(f"{X}/smx18/pdhd_stm_michel_scan_key_p82bhoff.tsv")}

A = arm(ARM)
c = collections.Counter(); FN = []; TN = []
for r in rec:
    k = r["key"]
    if k not in POP: continue
    d = A.get(tuple(k.split("/")))
    if d is None: continue
    v = truth(r)
    if v in ("MESSY", "UNCLEAR"): continue
    hs = v in ("STM_MICHEL", "STM_ONLY"); cs = d["is_stm"] == 1
    cls = "TP" if (hs and cs) else "FN" if hs else "FP" if cs else "TN"
    c[cls] += 1
    if cls == "FN": FN.append((k, d))
    if cls == "TN": TN.append((k, d))

got = (c["TP"], c["FP"], c["FN"], c["TN"])
if ARM in EXPECT and got != EXPECT[ARM]:
    sys.exit(f"SELF-GATE FAILED: {ARM} reads {got}, expected {EXPECT[ARM]}")
print(f"{ARM}: TP {c['TP']} FP {c['FP']} FN {c['FN']} TN {c['TN']}   (self-gate OK)\n")

print(f"reject bits on the {len(FN)} REMAINING missed hand stoppers:")
tal = collections.Counter()
for k, d in FN:
    for b in (d["bits"] or {"<none>"}): tal[b] += 1
for b, n in tal.most_common():
    print(f"   {b:20s} {n:3d}")

CLEAR = {"shape_flat", "no_bragg"}
for name, extra in (("P1", set()), ("P1+clears_sparse", {"profile_sparse"})):
    ok = CLEAR | extra
    efn = [(k, d) for k, d in FN if d["bits"] and d["bits"] <= ok]
    etn = [(k, d) for k, d in TN if d["bits"] and d["bits"] <= ok]
    rfn = sum(1 for k, d in efn if d["michel"] == 1)
    rtn = sum(1 for k, d in etn if d["michel"] == 1)
    print(f"\n{name}: eligible misses {len(efn)}/{len(FN)}, eligible THRU (FP pool) {len(etn)}/{len(TN)}")
    print(f"   michel_found=1 among eligible misses : {rfn}/{len(efn)}"
          f" = {rfn/max(1,len(efn)):.3f}   <- recoverable")
    print(f"   michel_found=1 among eligible THRU   : {rtn}/{len(etn)}"
          f" = {rtn/max(1,len(etn)):.3f}   <- NULL FLOOR / FP risk")
    if rfn: print("   recoverable items:", " ".join(k for k, d in efn if d["michel"] == 1))
    if rtn: print("   FP-risk items    :", " ".join(k for k, d in etn if d["michel"] == 1))
