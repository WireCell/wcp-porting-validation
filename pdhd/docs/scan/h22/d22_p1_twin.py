#!/usr/bin/env python3
"""doc pdhd/22 -- the P1 offline twin, using P1's EXACT admission test.

    python3 d22_p1_twin.py <baseline-arm>        (h22g = this round's production candidate)

The gate, from CheckSTM_Michel.cxx:4444 and its cfg[] documentation block:
  michel_found AND michel_conn_type in {1 attached, 2 bridged}
  AND michel_ke_best >= topology_michel_ke_min   (default 10 MeV)
  AND michel_len     >= topology_michel_len_min_cm (default 3 cm)
  -> clears R_NO_BRAGG and R_SHAPE_FLAT (and R_PROFILE_SPARSE with
     topology_clears_sparse), AFTER every other bit is set, so is_stm moves
     only 0 -> 1 and only when nothing else rejects.

NOTE this is stricter than a michel_found-only proxy, which over-predicts:
michel_found alone gives 23 candidates on h22g; the real gate gives fewer.
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
ARM = sys.argv[1] if len(sys.argv) > 1 else "h22g"
EXPECT = {"h22g": (82, 1, 65, 109), "h22base": (80, 0, 67, 110)}
V = ["cluster_id","is_stm","michel_found","reject_bits","michel_conn_type",
     "michel_ke_best","michel_len"]

def arm(tag):
    out = {}
    for f in sorted(glob.glob(f"{IMG}/pdhd/work/*_{tag}/tracking-pr.root")):
        evt = os.path.basename(os.path.dirname(f)).replace("_" + tag, "")
        c = uproot.open(f)["T_stm_michel"].arrays(V, library="np")
        for i in range(len(c["cluster_id"])):
            out[(evt, str(int(c["cluster_id"][i])))] = dict(
                is_stm=int(c["is_stm"][i]), michel=int(c["michel_found"][i]),
                bits=bset(c["reject_bits"][i]), conn=int(c["michel_conn_type"][i]),
                ke=float(c["michel_ke_best"][i]), mlen=float(c["michel_len"][i]))
    return out

rec = json.load(open(f"{X}/pdhd_stm_michel_smx22_verdicts.json"))
def truth(r):
    if r.get("owner_review"): return base(r["owner_review"]["verdict"])
    if r.get("owner_smx1"):
        o = r["owner_smx1"]; return base(o.get("choice") or o.get("label"))
    return base(r["verdict"])
rd = lambda p: list(csv.DictReader([l for l in open(p) if not l.startswith("#")], delimiter="\t"))
POP = {"%s/%s" % (r["event"], r["cluster"]) for r in
       rd(f"{X}/smx18/pdhd_stm_michel_scan_key_p82bhoff.tsv")}

A = arm(ARM); c = collections.Counter(); FN = []; TN = []
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
print(f"baseline {ARM}: TP {c['TP']} FP {c['FP']} FN {c['FN']} TN {c['TN']}  (self-gate OK)\n")

def qualifies(d, ke_min):
    return (d["michel"] == 1 and d["conn"] in (1, 2)
            and d["ke"] >= ke_min and d["mlen"] >= 3.0)

for name, sparse, ke_min in (("h22p1  P1", False, 10.0),
                             ("h22p2  +clears_sparse", True, 10.0),
                             ("h22p3  +ke_min 3", True, 3.0)):
    ok = {"shape_flat", "no_bragg"} | ({"profile_sparse"} if sparse else set())
    rec_items = sorted(k for k, d in FN if d["bits"] and d["bits"] <= ok and qualifies(d, ke_min))
    fp_items = sorted(k for k, d in TN if d["bits"] and d["bits"] <= ok and qualifies(d, ke_min))
    tp, fp = c["TP"] + len(rec_items), c["FP"] + len(fp_items)
    fn, tn = c["FN"] - len(rec_items), c["TN"] - len(fp_items)
    print(f"== {name} ==  +{len(rec_items)} stoppers / +{len(fp_items)} FP")
    print(f"   predicted census {tp}/{fp}/{fn}/{tn}  purity {tp/max(1,tp+fp):.3f}"
          f"  efficiency {tp/max(1,tp+fn):.3f}")
    print(f"   recovered ({len(rec_items)}): {' '.join(rec_items)}")
    print(f"   new FP    ({len(fp_items)}): {' '.join(fp_items) or 'NONE'}\n")
