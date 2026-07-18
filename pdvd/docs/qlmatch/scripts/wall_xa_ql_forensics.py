#!/usr/bin/env python3
"""Doc 25 §8: why wall-XA inclusion loses matches — gate forensics + raw
non-matched long cluster counts.

(a) wx0 -> wx1 selected-pair churn (same/moved/lost/new) and, for the lost
    pairs, the wx1 bundle metrics at the wx0 flash time (KS / chi2/ndf /
    LASSO strength) vs the wx0 selected bundle.
(b) Raw non-matched LONG clusters per tag (long = len >= 25 cm or >= 100 pts,
    same convention as ql_agree_score.py; non-matched = not main/other of any
    auto_selected bundle).
"""
import glob
import json
import math
import sys
import numpy as np

PDVD = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd"


def dump(idx, tag):
    return json.load(open(glob.glob(
        f"{PDVD}/work/039252_{idx}_{tag}/calib-evt*.json")[0]))


def cluster_len_cm(c):
    if not c.get("x"):
        return 0.0
    return math.sqrt((max(c["x"]) - min(c["x"])) ** 2 +
                     (max(c["y"]) - min(c["y"])) ** 2 +
                     (max(c["z"]) - min(c["z"])) ** 2)


def churn(tag0="wx0", tag1="wx1"):
    tot = dict(same=0, moved=0, gone=0, new=0)
    rows = []
    for idx in range(18):
        d0, d1 = dump(idx, tag0), dump(idx, tag1)
        f0 = {f["gid"]: f["time"] for f in d0["flashes"]}
        f1 = {f["gid"]: f["time"] for f in d1["flashes"]}
        sel0 = {}
        for b in d0["bundles"]:
            if b.get("auto_selected"):
                sel0.setdefault(b["main_cluster"], b)
        sel1 = {b["main_cluster"]: b for b in d1["bundles"] if b.get("auto_selected")}
        by1 = {}
        for b in d1["bundles"]:
            by1.setdefault(b["main_cluster"], []).append(b)
        for uid, b0 in sel0.items():
            if uid in sel1:
                t0, t1 = f0[b0["flash_gid"]], f1[sel1[uid]["flash_gid"]]
                tot["same" if abs(t1 - t0) < 0.1 else "moved"] += 1
            else:
                tot["gone"] += 1
                t0 = f0[b0["flash_gid"]]
                cands = [b for b in by1.get(uid, [])
                         if abs(f1[b["flash_gid"]] - t0) < 0.5]
                if cands:
                    b1 = cands[0]
                    rows.append((b0["ks_dis"], b1["ks_dis"],
                                 b0["chi2"] / max(b0["ndf"], 1),
                                 b1["chi2"] / max(b1["ndf"], 1),
                                 b0["strength"], b1["strength"]))
        tot["new"] += sum(1 for uid in sel1 if uid not in sel0)
    print(f"{tag0} -> {tag1} selected main clusters: {tot}")
    R = np.array(rows)
    print(f"lost-with-sametime-bundle n={len(R)}: "
          f"KS med {np.median(R[:,0]):.3f}->{np.median(R[:,1]):.3f} "
          f"(cross 0.10: {int(((R[:,0]<=0.10)&(R[:,1]>0.10)).sum())}); "
          f"chi2/ndf med {np.median(R[:,2]):.1f}->{np.median(R[:,3]):.1f} "
          f"(cross 35: {int(((R[:,2]<=35)&(R[:,3]>35)).sum())}); "
          f"strength med {np.median(R[:,4]):.2f}->{np.median(R[:,5]):.2f} "
          f"(collapse <0.05: {int(((R[:,4]>=0.05)&(R[:,5]<0.05)).sum())})")


def nonmatch(tags):
    print("tag\tevts\tlong\tmatched\tnonmatch")
    for tag in tags:
        tl = tm = nev = 0
        for p in sorted(glob.glob(f"{PDVD}/work/039252_*_{tag}/calib-evt*.json")):
            if "group" in p.split("/")[-1]:
                continue
            d = json.load(open(p))
            lu = {c["uid"] for c in d.get("clusters", [])
                  if cluster_len_cm(c) >= 25.0 or c.get("npoints", 0) >= 100}
            mu = set()
            for b in d.get("bundles", []):
                if not b.get("auto_selected"):
                    continue
                mu.add(b["main_cluster"])
                mu.update(b.get("other_clusters", []))
            tl += len(lu)
            tm += len(lu & mu)
            nev += 1
        print(f"{tag}\t{nev}\t{tl}\t{tm}\t{tl - tm}")


if __name__ == "__main__":
    churn()
    nonmatch(sys.argv[1:] or ["ac3", "wx0", "wx1", "wx2", "wx3"])
