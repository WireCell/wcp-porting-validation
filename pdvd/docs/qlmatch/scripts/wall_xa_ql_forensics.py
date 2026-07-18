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




def attribution(tag0="wx0", tag1="wx1"):
    """Doc 25 §8.3 attribution: split the wall |pred-meas| budget of the lost
    pairs into unpredicted-light / dark / responding-off, and check how much
    of the unpredicted light other candidate bundles on the same flash
    predict."""
    WALL = [0, 1, 3, 12, 18, 19]
    budget = np.zeros(3)  # dark, responding-off, unpredicted
    dom = np.zeros(3, int)
    dark_cov = [0, 0]     # covered, total
    unpred = []           # (meas, pred_this, pred_flash_sel, pred_flash_all)
    for idx in range(18):
        d0, d1 = dump(idx, tag0), dump(idx, tag1)
        f0 = {f["gid"]: f["time"] for f in d0["flashes"]}
        fl1 = {f["gid"]: f for f in d1["flashes"]}
        sel0 = {}
        for b in d0["bundles"]:
            if b.get("auto_selected"):
                sel0.setdefault(b["main_cluster"], b)
        sel1uid = {b["main_cluster"] for b in d1["bundles"] if b.get("auto_selected")}
        by1, pfs, pfa = {}, {}, {}
        for b in d1["bundles"]:
            by1.setdefault(b["main_cluster"], []).append(b)
            pp = np.array(b["pred_pe"], float)
            pfa[b["flash_gid"]] = pfa.get(b["flash_gid"], np.zeros(40)) + pp
            if b.get("auto_selected"):
                pfs[b["flash_gid"]] = pfs.get(b["flash_gid"], np.zeros(40)) + pp
        for uid, b0 in sel0.items():
            if uid in sel1uid:
                continue
            t0 = f0[b0["flash_gid"]]
            cands = [b for b in by1.get(uid, [])
                     if abs(fl1[b["flash_gid"]]["time"] - t0) < 0.5]
            if not cands:
                continue
            b1 = cands[0]
            g = b1["flash_gid"]
            pred = np.array(b1["pred_pe"], float)
            meas = np.array(fl1[g]["pe"], float)
            cov = np.array(fl1[g].get("cov", [1] * 40), float)
            here = np.zeros(3)
            for c in WALL:
                p, m = pred[c], meas[c]
                if p >= 2 and m < 0.5:
                    here[0] += p - m
                    dark_cov[0] += cov[c] >= 1
                    dark_cov[1] += 1
                elif p >= 2:
                    here[1] += abs(p - m)
                elif m >= 2:
                    here[2] += m - p
                    unpred.append((m, p,
                                   pfs.get(g, np.zeros(40))[c],
                                   pfa.get(g, np.zeros(40))[c]))
            budget += here
            if here.sum() > 0:
                dom[np.argmax(here)] += 1
    tot = budget.sum()
    for i, lab in enumerate(["dark(pred>=2,meas<0.5)", "responding-off",
                             "unpredicted(pred<2,meas>=2)"]):
        print(f"{lab}: {budget[i]:.0f} PE ({100*budget[i]/tot:.0f}%), "
              f"dominant in {dom[i]} pairs")
    print(f"dark wall channels: {dark_cov[1]}, fully covered: {dark_cov[0]}")
    U = np.array(unpred)
    if len(U):
        m, ps, pa = U[:, 0], U[:, 2], U[:, 3]
        print(f"unpredicted cases n={len(U)}: covered by other SELECTED "
              f"bundles {100*np.mean(ps >= 0.5*m):.0f}%, by ALL candidate "
              f"bundles {100*np.mean(pa >= 0.5*m):.0f}%")


if __name__ == "__main__":
    churn()
    attribution()
    nonmatch(sys.argv[1:] or ["ac3", "wx0", "wx1", "wx2", "wx3"])
