#!/usr/bin/env python3
"""doc pdvd/25 sec 13.6 follow-up: is the "flat stop end" measured where the TAGGER
puts the stop?

contrast_census.py measures median dQ/dx(rr<2 cm) / median(20-40 cm) with rr taken
from persist_stm_fit, i.e. from the fit PATH END (TaggerCheckSTM.cxx:931
`rr = Ltot - cumL[i]`).  eval_stm_core_impl re-anchors: it hunts the peak in a
window of width peak_range ending at the kink/path end, sets end_L = L[max_bin]+0.2cm,
and everything past that is `res_length` -- the residual it has already set aside.
So whenever res_length > 0 the census window sits in the tail, not on the tagger's
Bragg peak.

This script joins T_rec_charge (points), T_stm_pass (status) and T_stm_eval
(the accepting eval record) and reports, per accepted pass:
  contrast_end   = median dQ/dx(rr<2)         / median(20<=rr<40)     [the census number]
  contrast_peak  = median dQ/dx(res<=rr<res+2) / median(res+20<=rr<res+40)
plus the tagger's own flatness numbers (ks1, ks2, ratio1, ratio2) for the
accepting eval call, and how many accepted passes the EXISTING doc-63 ratio2 cap
(guard_ratio2_max = 2.0, part of stm_accept_guards, OFF in the PDVD job) would veto.

Repro:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 stm/stm_eval_anchor.py --tag stm2 --out stm/stm_eval_anchor_stm2.tsv
"""
import argparse, os, sys, glob, re
import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_stm_sample as C

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="stm2")
ap.add_argument("--out", default=None)
a = ap.parse_args()

def med_ratio(rr, dq, lo0, hi0, lo1, hi1, nmin0=3, nmin1=5):
    n = (rr >= lo0) & (rr < hi0) & (dq > 0)
    m = (rr >= lo1) & (rr < hi1) & (dq > 0)
    if n.sum() < nmin0 or m.sum() < nmin1:
        return float("nan"), int(n.sum()), int(m.sum())
    return float(np.median(dq[n]) / np.median(dq[m])), int(n.sum()), int(m.sum())

rows = []
for wd in sorted(glob.glob(os.path.join(C.PDVD, "work", f"*_{a.tag}"))):
    fp = os.path.join(wd, "tracking-stm.root")
    if not os.path.exists(fp):
        continue
    m = re.match(r"^(\d{6})_(\d+)", os.path.basename(wd))
    if not m:
        continue
    run, idx = m.group(1), m.group(2)
    ev = f"{run}_{idx}"
    try:
        f = uproot.open(fp)
    except Exception:
        continue
    if "T_rec_charge" not in f or "T_stm_eval" not in f:
        continue
    t = f["T_rec_charge"].arrays(["x", "q", "nq", "rr", "ndf", "status"], library="np")
    if len(t["ndf"]) == 0:
        continue
    tr = f["Trun"].arrays(["dQdx_scale", "dQdx_offset"], library="np")
    ee = f["T_stm_eval"].arrays(library="np")
    verdicts = C.read_verdicts(os.path.join(wd, f"wct_pr_{run}_{idx}.log"))
    for b in sorted(set(t["ndf"].tolist())):
        mk = t["ndf"] == b
        cid, pss = int(b) // 10, int(b) % 10
        v = verdicts.get(cid, {})
        if v.get("stm") != 1 or v.get("tgm") == 1 or int(t["status"][mk][0]) != 0:
            continue
        dQ = (t["q"][mk] - tr["dQdx_offset"][0]) / tr["dQdx_scale"][0]
        dx = np.where(np.abs(t["x"][mk]) > C.MAX_ABS_X, 0.0, t["nq"][mk])
        dq = np.where(dx > 0, dQ / np.maximum(dx, 1e-9), 0.0)
        rr = t["rr"][mk]
        # the accepting eval record for this (cluster, pass): the eval chain
        # short-circuits, so the LAST verdict==1 entry is the accepting call.
        sel = (ee["cluster_id"] == cid) & (ee["pass"] == pss) & (ee["verdict"] == 1)
        if not sel.any():
            continue
        j = np.where(sel)[0][-1]
        res = float(ee["res_length"][j])
        c_end, n0e, n1e = med_ratio(rr, dq, 0, 2, 20, 40)
        c_pk, n0p, n1p = med_ratio(rr, dq, res, res + 2, res + 20, res + 40)
        rows.append(dict(event=ev, cluster=cid, pss=pss, npts=int(mk.sum()), L=float(rr.max()),
                         res=res, com=float(ee["com_range"][j]), peak=float(ee["peak_range"][j]),
                         ks1=float(ee["ks1"][j]), ks2=float(ee["ks2"][j]),
                         r1=float(ee["ratio1"][j]), r2=float(ee["ratio2"][j]),
                         c_end=c_end, c_pk=c_pk, n0e=n0e, n1e=n1e, n0p=n0p, n1p=n1p))

n = len(rows)
print(f"accepted passes with an accepting eval record (tag {a.tag}): {n}")
res = np.array([r["res"] for r in rows])
ce = np.array([r["c_end"] for r in rows]); cp = np.array([r["c_pk"] for r in rows])
ks1 = np.array([r["ks1"] for r in rows]); ks2 = np.array([r["ks2"] for r in rows])
r2 = np.array([r["r2"] for r in rows]); r1 = np.array([r["r1"] for r in rows])
print("\n-- residual length past the tagger's peak (rr window the census actually used) --")
for lo, hi in [(0, 0.5), (0.5, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)]:
    print(f"  res_length [{lo},{hi}) cm: {int(np.sum((res>=lo)&(res<hi)))}")
print(f"  median res_length {np.median(res):.2f} cm; res > 2 cm: {int(np.sum(res>2))} ({100*np.mean(res>2):.0f} %)")

def hist(v, name):
    ok = np.isfinite(v)
    print(f"\n-- {name} (defined {int(ok.sum())}/{n}) --")
    for lo, hi in [(0, 0.8), (0.8, 1.2), (1.2, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 1e9)]:
        print(f"  [{lo},{hi}): {int(np.sum(ok & (v>=lo) & (v<hi)))}")
    if ok.sum():
        print(f"  median {np.median(v[ok]):.2f}; >= 2.0: {int(np.sum(ok & (v>=2)))} ({100*np.mean(v[ok]>=2):.0f} % of defined)")
hist(ce, "contrast anchored at the fit PATH END (the sec 13.6 census number)")
hist(cp, "contrast anchored at the TAGGER'S peak (res_length offset)")

both = np.isfinite(ce) & np.isfinite(cp)
if both.sum():
    print(f"\n  both defined: {int(both.sum())}; peak-anchored >= 2 while end-anchored < 2: "
          f"{int(np.sum(both & (cp>=2) & (ce<2)))}; end >= 2 while peak < 2: {int(np.sum(both & (ce>=2) & (cp<2)))}")

print("\n-- the tagger's OWN flatness numbers on the accepting call --")
print(f"  ks1 (muon hyp) median {np.median(ks1):.3f}; ks2 (flat MIP) median {np.median(ks2):.3f}")
print(f"  ks1 - ks2 median {np.median(ks1-ks2):.3f}; accepted with ks1-ks2 >= -0.02 (marginal branch only): "
      f"{int(np.sum(ks1-ks2 >= -0.02))} ({100*np.mean(ks1-ks2 >= -0.02):.0f} %)")
print(f"  |ratio1-1| median {np.median(np.abs(r1-1)):.3f}; |ratio2-1| median {np.median(np.abs(r2-1)):.3f}")
print(f"  ratio2 > 2.0 (the doc-63 guard_ratio2_max cap, OFF in the PDVD job): {int(np.sum(r2>2))} ({100*np.mean(r2>2):.0f} %)")
print(f"  com_range 15 cm: {int(np.sum([r['com']<20 for r in rows]))}; 35 cm: {int(np.sum([r['com']>=20 for r in rows]))}")

if a.out:
    with open(a.out, "w") as fo:
        fo.write("event\tcluster\tpass\tnpts\tL_cm\tres_len_cm\tcom_cm\tpeak_cm\tks1\tks2\tratio1\tratio2\tcontrast_end\tcontrast_peak\tn_end_near\tn_end_mid\tn_peak_near\tn_peak_mid\n")
        for r in rows:
            fo.write(f"{r['event']}\t{r['cluster']}\t{r['pss']}\t{r['npts']}\t{r['L']:.1f}\t{r['res']:.2f}\t{r['com']:.0f}\t{r['peak']:.0f}\t"
                     f"{r['ks1']:.4f}\t{r['ks2']:.4f}\t{r['r1']:.3f}\t{r['r2']:.3f}\t{r['c_end']:.3f}\t{r['c_pk']:.3f}\t"
                     f"{r['n0e']}\t{r['n1e']}\t{r['n0p']}\t{r['n1p']}\n")
    print("\nwrote", a.out)
