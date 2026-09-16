#!/usr/bin/env python3
"""doc pdvd/113 -- two reported (non-gating) breakdowns behind the verdict.  Read-only.

  1. the Steiner seed's off-ridge share split into near (1-5 cm) and far (> 5 cm), with the cloud / terminal / bridge
     sizes, per arm (from figs/113_steiner_<arm>.json);
  2. the tag movers of L3 (none) against base, by truth label source (d113_grade.truth): new false positives (with keys
     and verdicts), false positives gone, true positives lost and gained -- which labels the T clause rests on.

Usage: d113_nearfar_movers.py > figs/113_nearfar_movers.txt
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d113_grade as G
import d103_union_grade as U

F = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")
print("## seed off-ridge share split: near (1-5 cm) = off - far, far (> 5 cm)")
for det, x in (("pdhd", "h"), ("pdvd", "v")):
    b = json.load(open(f"{F}/113_steiner_d113{x}base.json"))["summary"]
    for arm in ("base", "none", "foot", "nopaint"):
        s = json.load(open(f"{F}/113_steiner_d113{x}{arm}.json"))["summary"]
        near, far = s["off"] - s["far"], s["far"]
        bn, bf = b["off"] - b["far"], b["far"]
        print(f"{det} {arm:8s} off {100*s['off']:.2f} % | near {100*near:.2f} % ({100*(near-bn)/bn:+.1f} %) | far {100*far:.2f} % ({100*(far-bf)/bf:+.1f} %)"
              f" | seed {s['seed_len_m']:.0f} m | cloud {s['g_nret']} | vertices {s['g_nsv']} | terminals {s['terminals']} (off {s['terminals_off']})"
              f" | 1blank {s['term_class']['1blank']} 1dead {s['term_class']['1dead']} | bridges {s['bridge_edges']} ({s['bridge_len_m']:.0f} m)")
print("\n## L3 (none) tag movers vs base with label source")
for det, x in (("pdhd", "h"), ("pdvd", "v")):
    T, _ = G.truth(det)
    R = {"A0": U.cell_rows(det, f"d113{x}base"), "L3": U.cell_rows(det, f"d113{x}none")}
    Gp, _ = G.grade(det, T, R)
    for title, P, TR, idx in (("is_stm", Gp["pop"], Gp["stm_truth"], 0), ("michel_found", Gp["mpop"], Gp["m_truth"], 1)):
        g0 = lambda k: R["A0"].get(k, (0, 0))[idx]
        g1 = lambda k: R["L3"].get(k, (0, 0))[idx]
        fp_new = [k for k in P if not TR[k] and g1(k) and not g0(k)]
        fp_gone = [k for k in P if not TR[k] and g0(k) and not g1(k)]
        tp_lost = [k for k in P if TR[k] and g0(k) and not g1(k)]
        tp_new = [k for k in P if TR[k] and g1(k) and not g0(k)]
        src = lambda ks: {s: sum(1 for k in ks if T[k][2] == s) for s in sorted({T[k][2] for k in ks})}
        print(f"{det} {title}: new FP {len(fp_new)} {src(fp_new)}; FP gone {len(fp_gone)} {src(fp_gone)}; "
              f"TP lost {len(tp_lost)} {src(tp_lost)}; TP new {len(tp_new)} {src(tp_new)}")
        print(f"   new FP keys: {sorted(fp_new)}")
        print(f"   new FP verdicts: {sorted(T[k][0] + '/' + str(T[k][1]) for k in fp_new)}")

# 3. resources: per-event PR wall and peak RSS (pr_resource_*.txt) of each arm over the concurrent base2 (same load)
import glob, re
import numpy as np
print("\n## resources vs base2 (arms run concurrently): median per-event ratio, and totals")
for det, x in (("pdhd", "h"), ("pdvd", "v")):
    def res(arm):
        out = {}
        for p in glob.glob(f"{U.IMG}/{det}/work/*_{arm}/pr_resource_*.txt"):
            t = open(p).read()
            ev = os.path.basename(os.path.dirname(p))[:-len(arm) - 1]
            out[ev] = (float(re.search(r"wall_s=(\S+)", t).group(1)), float(re.search(r"peak_rss_gb=(\S+)", t).group(1)))
        return out
    b = res(f"d113{x}base2")
    for arm in ("none", "foot", "nopaint"):
        a = res(f"d113{x}{arm}")
        ev = sorted(set(a) & set(b))
        w = np.array([a[e][0] / b[e][0] for e in ev]); r = np.array([a[e][1] / b[e][1] for e in ev])
        print(f"{det} {arm:8s} events {len(ev)}: wall median ratio {np.median(w):.3f} (total {sum(a[e][0] for e in ev):.0f} s vs "
              f"{sum(b[e][0] for e in ev):.0f} s); peak RSS median ratio {np.median(r):.3f} (max {max(a[e][1] for e in ev):.2f} GB vs "
              f"{max(b[e][1] for e in ev):.2f} GB)")
