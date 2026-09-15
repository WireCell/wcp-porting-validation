#!/usr/bin/env python3
"""doc pdvd/102 round 2 -- list the PDHD hand-scan items whose michel_found verdict changes between two arms.

Fork by duplication of the michel_found block of d101_stm_grade_pdhd.py (untouched): the same smx22 record,
the same truth precedence, the same smx18 population and exclusions.  Instead of counts it prints, for every
hand stopper in the population, the items that are
  NEW_FP    hand non-Michel, michel_found 0 (or absent) in BASE, 1 in ARM
  LOST_TP   hand Michel, 1 in BASE, 0 (or absent) in ARM
  GONE_FP / NEW_TP  the reverse moves
with the event/cluster key, the truth source and whether the candidate is absent from either arm -- the list
the owner needs to adjudicate a purity change (feedback: one mislabel faked a cost).

Usage: d102_michel_fp_list.py BASE ARM     eg  d102_michel_fp_list.py d101hkf d102hcs
"""
import csv, glob, json, os, sys
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
    if r.get("owner_review"):
        return r["owner_review"].get("michel_kind"), "owner_review"
    if r.get("owner_smx1"):
        return r["owner_smx1"].get("michel_kind"), "owner"
    return r.get("michel_kind"), "agent"


rd = lambda p: list(csv.DictReader([l for l in open(p) if not l.startswith("#")], delimiter="\t"))
POP = {"%s/%s" % (r["event"], r["cluster"]) for r in rd(f"{X}/smx18/pdhd_stm_michel_scan_key_p82bhoff.tsv")}


def main():
    b_tag, a_tag = sys.argv[1], sys.argv[2]
    B, A = arm(b_tag), arm(a_tag)
    rows = []
    for r in rec:
        k = r["key"]
        if k not in POP or truth(r) not in ("STM_MICHEL", "STM_ONLY"):
            continue
        kind, src = michel_truth(r)
        if src == "owner_review" and kind is None:
            continue
        hm = kind in ("attached", "both")
        key = tuple(k.split("/"))
        mb = B.get(key); ma = A.get(key)
        fb = mb is not None and mb[1] == 1
        fa = ma is not None and ma[1] == 1
        if fb == fa:
            continue
        move = ("NEW_TP" if fa else "LOST_TP") if hm else ("NEW_FP" if fa else "GONE_FP")
        rows.append((move, k, kind, src, "absent" if mb is None else f"is_stm {mb[0]} michel {mb[1]}",
                     "absent" if ma is None else f"is_stm {ma[0]} michel {ma[1]}"))
    order = {"NEW_FP": 0, "LOST_TP": 1, "GONE_FP": 2, "NEW_TP": 3}
    rows.sort(key=lambda x: (order[x[0]], x[1]))
    print(f"michel_found moves {b_tag} -> {a_tag} on the hand stoppers of the fixed population:")
    for m in order:
        print(f"  {m}: {sum(1 for x in rows if x[0] == m)}")
    print("move\tkey\ttruth_michel_kind\ttruth_source\tbase\tarm")
    for x in rows:
        print("\t".join(str(v) for v in x))


if __name__ == "__main__":
    main()
