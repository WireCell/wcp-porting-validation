#!/usr/bin/env python3
"""doc pdvd/101 sec 6.5 -- PDHD STM/Michel grade of the knob arms on a FIXED denominator.

Fork by duplication (CLAUDE.md M10) of pdhd/docs/scan/h21/d21_grade.py (untouched): same arm
reader, same smx22 truth precedence, same 303-item population (smx18 key).  The one difference:
d21_grade.py skips a hand-scanned item whose cluster is not a CheckSTM_Michel candidate in the arm,
so an arm that loses candidates loses denominator.  Here such an item is kept and counted
'absent': a hand stopper that is absent is a miss (FN), a hand non-stopper that is absent is TN.

Usage: d101_stm_grade_pdhd.py <arm> [<arm> ...]
"""
import collections, csv, glob, json, os, sys
import uproot

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
X = IMG + "/pdhd/docs/scan"
base = lambda v: v[5:] if v and v.startswith("FRAG_") else v


def arm(tag):
    out = {}
    for f in sorted(glob.glob(f"{IMG}/pdhd/work/*_{tag}/tracking-pr.root")):
        evt = os.path.basename(os.path.dirname(f)).replace("_" + tag, "")
        c = uproot.open(f)["T_stm_michel"].arrays(["cluster_id", "is_stm", "michel_found"], library="np")
        for i in range(len(c["cluster_id"])):
            out[(evt, str(int(c["cluster_id"][i])))] = (int(c["is_stm"][i]), int(c["michel_found"][i]))
    return out


rec = json.load(open(f"{X}/pdhd_stm_michel_smx22_verdicts.json"))


def truth(r):
    if r.get("owner_review"):
        return base(r["owner_review"]["verdict"])
    if r.get("owner_smx1"):
        o = r["owner_smx1"]; return base(o.get("choice") or o.get("label"))
    return base(r["verdict"])


def michel_truth(r):
    """d21_michel_census.py's precedence: (michel_kind, source)."""
    if r.get("owner_review"):
        return r["owner_review"].get("michel_kind"), "owner_review"
    if r.get("owner_smx1"):
        return r["owner_smx1"].get("michel_kind"), "owner"
    return r.get("michel_kind"), "agent"


rd = lambda p: list(csv.DictReader([l for l in open(p) if not l.startswith("#")], delimiter="\t"))
POP = {"%s/%s" % (r["event"], r["cluster"]) for r in rd(f"{X}/smx18/pdhd_stm_michel_scan_key_p82bhoff.tsv")}

for tag in sys.argv[1:]:
    A = arm(tag); c = collections.Counter()
    for r in rec:
        k = r["key"]
        if k not in POP:
            continue
        v = truth(r)
        if v in ("MESSY", "UNCLEAR"):
            continue
        hs = v in ("STM_MICHEL", "STM_ONLY")
        d = A.get(tuple(k.split("/")))
        if d is None:
            c["absent_stopper" if hs else "absent_other"] += 1
            c["FN" if hs else "TN"] += 1
            continue
        cs = d[0] == 1
        c["TP" if (hs and cs) else "FN" if hs else "FP" if cs else "TN"] += 1
    n = c["TP"] + c["FP"] + c["FN"] + c["TN"]
    p = c["TP"] / max(1, c["TP"] + c["FP"]); e = c["TP"] / max(1, c["TP"] + c["FN"])
    print(f"  {tag:10s} scored {n}: TP {c['TP']:3d} FP {c['FP']:3d} FN {c['FN']:3d} TN {c['TN']:3d} | purity {p:.3f} "
          f"efficiency {e:.3f} | absent from the arm: {c['absent_stopper']} hand stoppers, {c['absent_other']} others")

# michel_found, the population of d21_michel_census.py (hand stoppers in POP; an owner_review stopper with
# no michel_kind is excluded), but an absent stopper is kept: absent hand Michel = FN, otherwise TN.
print("\nmichel_found on hand stoppers, fixed denominator (truth = michel_kind in attached/both):")
for tag in sys.argv[1:]:
    A = arm(tag); c = collections.Counter()
    for r in rec:
        k = r["key"]
        if k not in POP or truth(r) not in ("STM_MICHEL", "STM_ONLY"):
            continue
        kind, src = michel_truth(r)
        if src == "owner_review" and kind is None:
            c["excluded"] += 1
            continue
        hm = kind in ("attached", "both")
        d = A.get(tuple(k.split("/")))
        if d is None:
            c["absent_michel" if hm else "absent_other"] += 1
            c["FN" if hm else "TN"] += 1
            continue
        cm = d[1] == 1
        c["TP" if (hm and cm) else "FN" if hm else "FP" if cm else "TN"] += 1
    p = c["TP"] / max(1, c["TP"] + c["FP"]); e = c["TP"] / max(1, c["TP"] + c["FN"])
    print(f"  {tag:10s} TP {c['TP']:3d} FP {c['FP']:3d} FN {c['FN']:3d} TN {c['TN']:3d} | purity {p:.3f} efficiency {e:.3f} "
          f"(excluded {c['excluded']}) | absent from the arm: {c['absent_michel']} hand Michels, {c['absent_other']} other stoppers")
