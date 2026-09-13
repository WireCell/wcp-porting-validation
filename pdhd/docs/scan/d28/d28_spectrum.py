#!/usr/bin/env python3
"""doc pdhd/28 sec 4 -- what the wire lookup does to PDHD's region energy, and the corrected spectrum against PDVD.

    STM_SCAN_RECORD=<smx27 record> python3 d28_spectrum.py [--off h28off --on h28wl --pdvd p96vprod] > spectrum.txt

Population: hand Michel items (hand stopper, michel_kind attached/both) that the chain calls is_stm & michel_found,
PDHD APA0 strict (smx27), PDVD with a candidate (smx1a..smx9) -- doc pdhd/26 sec 3's population.
"""
import argparse, collections, glob, os, sys
import numpy as np, uproot
from scipy import stats
IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
sys.path.insert(0, IMG + "/pdhd/docs/scan/h25")
import d25_bragg_michel as BM
ap = argparse.ArgumentParser()
ap.add_argument("--off", default="h28off"); ap.add_argument("--on", default="h28wl"); ap.add_argument("--pdvd", default="p96vprod")
a = ap.parse_args()
ENDPOINT = 52.83
KMEV = {"pdhd": 41.3035e-6, "pdvd": 42.7058e-6}
BR = ["cluster_id", "is_stm", "michel_found", "michel_ke_q2d_region", "michel_ke_q2d_ctl", "michel_ke_best",
      "michel_q2d_region_dropped_plane", "michel_q2d_ctl_dropped_plane", "michel_q2d_n_role0"] + \
     ["michel_q2d_region_%s" % p for p in "uvw"] + ["michel_q2d_region_n_%s" % p for p in "uvw"] + ["michel_q2d_ctl_%s" % p for p in "uvw"]


def load(det, arm, pop):
    its, _ = BM.items(det, arm, pop); J = {x[0]: x for x in its}
    out = {}
    for fn in sorted(glob.glob(f"{IMG}/{det}/work/*_{arm}/tracking-pr.root")):
        ev = os.path.basename(os.path.dirname(fn))[: -len(arm) - 1]
        u = uproot.open(fn)
        if "T_stm_michel" not in u: continue
        t = u["T_stm_michel"].arrays([b for b in BR if b in u["T_stm_michel"].keys()], library="np")
        C = u["T_stm_michel_2d"].arrays(["cluster_id", "apa", "plane", "role"], library="np") if "T_stm_michel_2d" in u else None
        for i in range(len(t["cluster_id"])):
            k = f"{ev}/{int(t['cluster_id'][i])}"; x = J.get(k)
            if not (x and x[1] in BM.STOP and x[2] in BM.MICHEL_KINDS and int(t["is_stm"][i]) == 1 and int(t["michel_found"][i]) == 1): continue
            d = {b: t[b][i] for b in t}
            if C is not None:
                m = (C["cluster_id"] == t["cluster_id"][i]) & (C["plane"] == 2) & (C["role"] == 1)
                d["apa"] = collections.Counter(C["apa"][m].tolist()).most_common(1)[0][0] if m.any() else -1
            out[k] = d
    return out


def bmed(v, n=2000, seed=28):
    v = np.asarray(v, float); r = np.random.default_rng(seed)
    m = np.median(v[r.integers(0, len(v), size=(n, len(v)))], axis=1)
    return np.median(v), np.percentile(m, 16), np.percentile(m, 84)


OFF, ON, VD = load("pdhd", a.off, "strict"), load("pdhd", a.on, "strict"), load("pdvd", a.pdvd, "all")
keys = sorted(set(OFF) & set(ON))
print(f"hand Michel items: PDHD {len(keys)} (off {len(OFF)}, on {len(ON)}), PDVD {len(VD)}")
print("\n=== 1. PDHD, knob off vs on, median over items")
f = lambda D, b: np.array([float(D[k][b]) for k in keys])
for lab, b in (("region MeV", "michel_ke_q2d_region"), ("control MeV", "michel_ke_q2d_ctl"), ("michel_ke_best", "michel_ke_best"),
               ("role-0 cells in the region sum (michel_q2d_n_role0)", "michel_q2d_n_role0")):
    print(f"  {lab:52s} off {np.median(f(OFF, b)):7.1f}  on {np.median(f(ON, b)):7.1f}")
for p in "uvw":
    print(f"  region plane {p.upper()}: sum MeV-eq off {np.median(f(OFF, 'michel_q2d_region_' + p)) * KMEV['pdhd']:6.1f} on {np.median(f(ON, 'michel_q2d_region_' + p)) * KMEV['pdhd']:6.1f}"
          f" | cells off {np.median(f(OFF, 'michel_q2d_region_n_' + p)):5.0f} on {np.median(f(ON, 'michel_q2d_region_n_' + p)):5.0f}"
          f" | control sum off {np.median(f(OFF, 'michel_q2d_ctl_' + p)) * KMEV['pdhd']:6.1f} on {np.median(f(ON, 'michel_q2d_ctl_' + p)) * KMEV['pdhd']:6.1f}")
for nm, b in (("region", "michel_q2d_region_dropped_plane"), ("control", "michel_q2d_ctl_dropped_plane")):
    print(f"  {nm} dropped plane (none/U/V/W): off {[int((f(OFF, b) == v).sum()) for v in (-1, 0, 1, 2)]} on {[int((f(ON, b) == v).sum()) for v in (-1, 0, 1, 2)]}")
side = {k: ("APA1/3" if OFF[k].get("apa") in (1, 3) else "APA2") for k in keys}
for s in ("APA2", "APA1/3"):
    kk = [k for k in keys if side[k] == s]
    if not kk: continue
    g = lambda D, b: np.median([float(D[k][b]) for k in kk])
    print(f"  {s} (n {len(kk)}): region off {g(OFF, 'michel_ke_q2d_region'):.1f} on {g(ON, 'michel_ke_q2d_region'):.1f} | control off {g(OFF, 'michel_ke_q2d_ctl'):.1f} on {g(ON, 'michel_ke_q2d_ctl'):.1f}")

print("\n=== 2. the corrected PDHD spectrum against PDVD production")
for lab, D, kk in (("PDHD off (was production)", OFF, keys), ("PDHD on  (new production)", ON, keys), ("PDVD p96vprod", VD, sorted(VD))):
    r = np.array([float(D[k]["michel_ke_q2d_region"]) for k in kk]); c = np.array([float(D[k]["michel_ke_q2d_ctl"]) for k in kk])
    m = bmed(r); mc = bmed(c)
    print(f"  {lab:28s} n {len(r):3d} region median {m[0]:5.1f} [{m[1]:5.1f},{m[2]:5.1f}] p90 {np.percentile(r, 90):5.1f} above 52.8 {int((r > ENDPOINT).sum()):3d} ({(r > ENDPOINT).mean():.3f})"
          f" | control median {mc[0]:5.1f} [{mc[1]:5.1f},{mc[2]:5.1f}] above 10 {(c > 10).mean():.3f}")
hon = [float(ON[k]["michel_ke_q2d_region"]) for k in keys]; vd = [float(VD[k]["michel_ke_q2d_region"]) for k in VD]
print(f"  PDHD on vs PDVD region: KS p {stats.ks_2samp(hon, vd).pvalue:.4f}, Mann-Whitney p {stats.mannwhitneyu(hon, vd).pvalue:.4f}")
d = np.array([float(ON[k]["michel_ke_q2d_region"]) - float(OFF[k]["michel_ke_q2d_region"]) for k in keys])
print(f"  per-item region shift on - off: p10/p50/p90 {np.percentile(d, 10):+.1f}/{np.median(d):+.1f}/{np.percentile(d, 90):+.1f}; unchanged {int((np.abs(d) < 1e-9).sum())}")
print("  largest shifts:")
for k in sorted(keys, key=lambda k: -abs(float(ON[k]["michel_ke_q2d_region"]) - float(OFF[k]["michel_ke_q2d_region"])))[:8]:
    print(f"    {k:15s} {side[k]:6s} region {float(OFF[k]['michel_ke_q2d_region']):6.1f} -> {float(ON[k]['michel_ke_q2d_region']):6.1f}"
          f"  control {float(OFF[k]['michel_ke_q2d_ctl']):5.1f} -> {float(ON[k]['michel_ke_q2d_ctl']):5.1f}  role-0 cells {int(OFF[k]['michel_q2d_n_role0'])} -> {int(ON[k]['michel_q2d_n_role0'])}")
