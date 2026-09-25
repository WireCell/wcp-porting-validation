#!/usr/bin/env python3
"""doc sbnd_xin/124 -- read the flash-time nudge: per event, the verdict of arm A (production), arm A with its
beam clusters' t0 moved to arm B's flash time (the nudge), and arm B (production).

The candidate is the highest-numu one of the event (T_tagger rows of tracking-pr.root; numu_score from the
UbooneNumuBDTScorer output 'numu_score' branch), as r3_pr_compare.py picks it.

usage: nudge_cmp.py <pr_A> <pr_nudged> <pr_B> <anatomy.tsv>
"""
import csv, os, sys
import numpy as np, uproot

PDG = {11: "e", -11: "e", 13: "mu", -13: "mu", 211: "pi", -211: "pi", 2212: "p", 22: "g", 111: "pi0", 321: "K", 2112: "n", 0: "?"}


def verdict(d):
    f = uproot.open(os.path.join(d, "tracking-pr.root"))
    if "T_tagger" not in f:
        return None
    t = f["T_tagger"].arrays(["numu_score", "nu_x", "nu_y", "nu_z", "cluster_id"], library="np")
    i = int(np.argmax(t["numu_score"]))
    k = f["T_kine"].arrays(["kine_particle_type", "kine_energy_particle", "kine_reco_Enu", "cluster_id"], library="np")
    j = [x for x in range(len(k["cluster_id"])) if int(k["cluster_id"][x]) == int(t["cluster_id"][i])]
    pid = ""; enu = 0.0
    if j:
        j = j[0]
        pid = " ".join("%s%.0f" % (PDG.get(int(a), str(int(a))), e) for a, e in zip(k["kine_particle_type"][j], k["kine_energy_particle"][j]))
        enu = float(k["kine_reco_Enu"][j])
    return dict(numu=float(t["numu_score"][i]), v=np.array([t["nu_x"][i], t["nu_y"][i], t["nu_z"][i]], dtype=float), enu=enu, pid=pid)


def main():
    A, N, B, anat = sys.argv[1:5]
    cls = {r["event"]: r["class"] for r in csv.DictReader(open(anat), delimiter="\t")}
    tally = {}
    for e in sorted(os.listdir(N)):
        if not e.startswith("pr_evt"):
            continue
        ev = e[6:]
        a, n, b = verdict(os.path.join(A, e)), verdict(os.path.join(N, e)), verdict(os.path.join(B, e))
        if a is None or n is None or b is None:
            print("%s: a missing verdict" % ev); continue
        flip_n = (n["numu"] > 0.9) != (a["numu"] > 0.9)
        same_as_b = (n["numu"] > 0.9) == (b["numu"] > 0.9)
        exact_a = abs(n["numu"] - a["numu"]) < 1e-6 and n["pid"] == a["pid"]
        exact_b = abs(n["numu"] - b["numu"]) < 1e-6 and n["pid"] == b["pid"]
        tag = "=A" if exact_a else ("=B" if exact_b else "new")
        key = (cls.get(ev, "?"), "nudge flips" if flip_n else "nudge keeps A")
        tally[key] = tally.get(key, 0) + 1
        print("%-8s %-9s A %6.2f | nudged %6.2f (%s, dv %.1f cm from A) | B %6.2f  %s\n           A: %s\n           N: %s\n           B: %s" % (
            ev, cls.get(ev, "?"), a["numu"], n["numu"], tag, float(np.linalg.norm(n["v"] - a["v"])), b["numu"],
            "VERDICT FOLLOWS B" if flip_n and same_as_b else ("verdict flips" if flip_n else "verdict stays A"),
            a["pid"][:90], n["pid"][:90], b["pid"][:90]))
    print("tally:", ", ".join("%s/%s %d" % (k[0], k[1], v) for k, v in sorted(tally.items())))


if __name__ == "__main__":
    main()
