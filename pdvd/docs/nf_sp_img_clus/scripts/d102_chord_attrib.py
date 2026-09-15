#!/usr/bin/env python3
"""doc pdvd/102 sec 5.3 -- what sits under an off-image chord of the STM fit.

A chord is >= 10 consecutive STM-fit rows (tracking-stm.root T_rec_charge, one (cluster, pass)) that are
each > 3 cm from every clustering-image point of their own cluster (Bee clustering layer) -- the same
definition as d102_census.py.  For each chord, the median distance of its rows to the cluster's Steiner
points (calib-pr-evt*.json 'steiner', the retiled cloud the STM rough path runs on):
  steiner_absent_along   median > 3 cm: no retiled point under the chord, i.e. the rough path crossed it on
                         a long Steiner-graph edge (an MST bridge) and the fit filled it with a straight line
  steiner_present_along  median <= 3 cm: the retiled cloud does extend under the chord (the image and the
                         cloud disagree, not the graph)
  no_steiner_for_cluster the calib dump holds no Steiner points for this cluster

Usage: d102_chord_attrib.py [--max-events 30] det:ARM [det:ARM ...]
   eg  d102_chord_attrib.py pdhd:d101hkf pdvd:d101vkf pdhd:d102hcs pdvd:d102vcsall
"""
import argparse, collections, glob, json, os, zipfile
import numpy as np
import uproot
from scipy.spatial import cKDTree

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-events", type=int, default=30)
    ap.add_argument("arms", nargs="+")
    a = ap.parse_args()
    for spec in a.arms:
        det, tag = spec.split(":")
        works = sorted(glob.glob(f"{IMG}/{det}/work/*_{tag}"))
        if a.max_events:
            works = works[:a.max_events]
        res = collections.Counter(); ex = []; nev = 0; skipped = []
        for w in works:
            try:
                t = uproot.open(f"{w}/tracking-stm.root")["T_rec_charge"].arrays(["x", "y", "z", "cluster_id", "pass"], library="np")
                cal = json.load(open(glob.glob(f"{w}/calib-pr-evt*.json")[0]))
                img = json.loads(zipfile.ZipFile(f"{w}/mabc-pr.zip").read("data/0/0-clustering-global.json"))
            except Exception as e:
                skipped.append(os.path.basename(w)); continue
            nev += 1
            I = np.c_[img["x"], img["y"], img["z"]]; ic = np.array(img["cluster_id"])
            st = {}
            for s in cal.get("steiner", []):
                st.setdefault(s["cluster_id"], []).append(np.c_[s["x"], s["y"], s["z"]])
            X = np.c_[t["x"], t["y"], t["z"]]
            for cl in np.unique(t["cluster_id"]):
                Ic = I[ic == cl]
                if len(Ic) == 0:
                    continue
                ti = cKDTree(Ic)
                S = np.concatenate(st[cl]) if cl in st else None
                ts = cKDTree(S) if S is not None and len(S) else None
                for pa in np.unique(t["pass"][t["cluster_id"] == cl]):
                    idx = np.where((t["cluster_id"] == cl) & (t["pass"] == pa))[0]
                    P = X[idx]; off = ti.query(P)[0] > 3
                    j = 0
                    while j < len(idx):
                        if not off[j]:
                            j += 1; continue
                        k = j
                        while k < len(idx) and off[k]:
                            k += 1
                        if k - j >= 10:
                            arc = float(np.sum(np.linalg.norm(np.diff(P[j:k], axis=0), axis=1)))
                            straight = float(np.linalg.norm(P[k - 1] - P[j])) / max(arc, 1e-6)
                            if ts is None:
                                cat = "no_steiner_for_cluster"
                            else:
                                cat = "steiner_absent_along" if np.median(ts.query(P[j:k])[0]) > 3 else "steiner_present_along"
                            res[(cat, ">20cm" if arc > 20 else "<=20cm")] += 1
                            if len(ex) < 6 and cat == "steiner_present_along" and arc > 20:
                                ex.append((os.path.basename(w), int(cl), int(pa), round(arc, 1), round(straight, 2)))
                        j = k
        tot = sum(res.values())
        absent = res[("steiner_absent_along", ">20cm")] + res[("steiner_absent_along", "<=20cm")]
        print(f"[{det}:{tag}] events {nev} (skipped {len(skipped)}: {' '.join(skipped) or '-'}) chords {tot}; "
              f"steiner absent along {absent}/{tot} = {absent / max(tot, 1):.3f}")
        for key in sorted(res):
            print(f"    {key[0]:24s} {key[1]:7s} {res[key]}")
        if ex:
            print("    long chords WITH steiner along:", ex)


if __name__ == "__main__":
    main()
