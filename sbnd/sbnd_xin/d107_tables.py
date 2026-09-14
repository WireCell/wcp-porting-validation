#!/usr/bin/env python3
"""doc 107 sec 4-5 tables, re-derived from products/d107/{candidates,truth}.tsv.

Usage:
  python3 d107_tables.py products/d107
Every count and fraction quoted in doc 107 sections 4.2, 4.5, 4.6 and 5 is
printed here; summary.txt (written by d107_truth_tagger_consistency.py) holds
the section 2 and 4.1/4.3/4.4 census counts.
"""
import collections
import csv
import os
import statistics as st
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "products/d107"
C = list(csv.DictReader(open(os.path.join(d, "candidates.tsv")), delimiter="\t"))
T = list(csv.DictReader(open(os.path.join(d, "truth.tsv")), delimiter="\t"))
f = float


def dstats(rows, label):
    x = [f(r["t_dist_cm"]) for r in rows
         if r["vertex_default"] == "0" and f(r["t_dist_cm"]) >= 0]
    n = len(x)
    if not n:
        print(f"{label:40s} rows={len(rows)} with_vertex=0")
        return
    print(f"{label:40s} rows={len(rows):5d} with_vertex={n:5d} "
          f"<3cm {sum(v < 3 for v in x)/n:.3f} <5cm {sum(v < 5 for v in x)/n:.3f} "
          f"<20cm {sum(v < 20 for v in x)/n:.3f} >50cm {sum(v > 50 for v in x)/n:.3f} "
          f"median {st.median(x):.1f}")


def cls(r):
    if r["t_ccnc"] == "CC" and r["t_flav"] == "numu":
        return "numuCC"
    if r["t_ccnc"] == "CC" and r["t_flav"] == "nue":
        return "nueCC"
    if r["t_ccnc"] == "CC":
        return "CC-" + r["t_flav"]
    return "NC"


print("== rows", len(C), "events", len({(r["run"], r["subrun"], r["event"]) for r in C}))
print("== sec 4.2 / 4.5 / 5.1 vertex association")
dstats(C, "all")
dstats([r for r in C if r["bundle_has_vetoed"] == "0"], "no vetoed activity")
dstats([r for r in C if r["bundle_has_vetoed"] == "1" and r["sel_demoted"] == "0"],
       "vetoed -> other main")
dstats([r for r in C if r["bundle_has_vetoed"] == "1" and r["sel_demoted"] == "1"],
       "vetoed -> demoted fallback")
dstats([r for r in C if r["bundle_has_vetoed"] == "0" and r["sel_demoted"] == "1"],
       "not vetoed but demoted selected")
for v in ("same", "in_act_not_selected", "not_in_act"):
    dstats([r for r in C if r["cid_relation"] == v], f"cid_relation={v}")
for v in ("0", "1"):
    dstats([r for r in C if r["nu_index"] == v], f"nu_index={v}")
A = [r for r in C if r["vertex_default"] == "0" and 0 <= f(r["t_dist_cm"]) < 5]
print("assoc<5cm", len(A), "with t_Edep>0:", sum(f(r["t_Edep"]) > 0 for r in A))

print("== sec 4.6 neutrino_type / cosmict_flag")
print("neutrino_type", collections.Counter(r["neutrino_type"] for r in C).most_common())
print("vertex_default rows", sum(r["vertex_default"] == "1" for r in C),
      "their neutrino_type", collections.Counter(r["neutrino_type"] for r in C
                                                 if r["vertex_default"] == "1"))
print("cosmict_flag == nt_cosmic", sum(r["cosmict_flag"] == r["nt_cosmic"] for r in C), "/", len(C))
print("(bundle_has_vetoed, cosmict_flag)",
      sorted(collections.Counter((r["bundle_has_vetoed"], r["cosmict_flag"]) for r in C).items()))

print("== sec 5.2 scores by true class (assoc < 5 cm; last line: >= 20 cm)")
groups = collections.defaultdict(list)
for r in A:
    groups[cls(r)].append(r)
groups["unassoc>=20cm"] = [r for r in C if r["vertex_default"] == "0" and f(r["t_dist_cm"]) >= 20]
for k, v in groups.items():
    nm = [f(r["numu_score"]) for r in v]
    ne = [f(r["nue_score"]) for r in v]
    n = len(v)
    print(f"{k:14s} n={n:5d} numu med {st.median(nm):.2f} >0 {sum(x > 0 for x in nm)/n:.3f} | "
          f"nue =-15 {sum(x == -15 for x in ne)/n:.3f} >0 {sum(x > 0 for x in ne)/n:.3f} | "
          f"bit numucc {sum(r['nt_numucc'] == '1' for r in v)/n:.3f} "
          f"nue {sum(r['nt_nue'] == '1' for r in v)/n:.3f} "
          f"cosmic {sum(r['nt_cosmic'] == '1' for r in v)/n:.3f}")

print("== sec 5.3 reco Enu ratios (assoc < 5 cm, reco_Enu > 0)")
for k in ("numuCC", "NC", "nueCC"):
    v = [r for r in A if cls(r) == k and f(r["reco_Enu"]) > 0]
    r1 = [f(r["reco_Enu"]) / f(r["t_Etot"]) for r in v if f(r["t_Etot"]) > 0]
    r2 = [f(r["reco_Enu"]) / f(r["t_Edep"]) for r in v if f(r["t_Edep"]) > 0]
    q1 = [round(x, 3) for x in st.quantiles(r1, n=4)]
    q2 = [round(x, 3) for x in st.quantiles(r2, n=4)]
    print(f"{k:7s} n={len(v)} reco/Etot [Q1,med,Q3] {q1} | reco/Edep {q2}")

print("== sec 5.4 candidate presence vs event max Edep")
ev = {}
for t in T:
    k = (t["run"], t["subrun"], t["event"])
    m, h = ev.get(k, (0.0, 0))
    ev[k] = (max(m, f(t["Edep"])), int(t["event_has_candidate"]))
print("events with truth nodes", len(ev))
for lo, hi in [(0, 1e-3), (1e-3, 20), (20, 50), (50, 100), (100, 200), (200, 500), (500, 1e12)]:
    s = [v for v in ev.values() if lo <= v[0] < hi]
    c = sum(x[1] for x in s)
    print(f"max Edep [{lo},{hi}) events {len(s):5d} with candidate {c:5d} "
          f"without {len(s)-c:5d} frac {c/max(1, len(s)):.3f}")
for lo, hi in [(100, 200), (200, 500), (500, 1e12)]:
    s = [t for t in T if lo <= f(t["Edep"]) < hi]
    print(f"truth Edep [{lo},{hi}) n={len(s)} candidate vertex <5cm "
          f"{sum(0 <= f(t['min_cand_dist_cm']) < 5 for t in s)/len(s):.3f}")
print("truth flavours", collections.Counter(t["flav"] for t in T).most_common(),
      "modes", collections.Counter(t["mode"] for t in T).most_common())
print("truth T range", min(f(t["T"]) for t in T), max(f(t["T"]) for t in T))
