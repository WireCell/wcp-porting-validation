#!/usr/bin/env python3
"""doc pdvd/101 sec 4 -- grade the full-chain PR fit of the simulated single muons against truth,
with the SAME metrics d101_toy.metrics() computes, so toy and C++ are read with one ruler.

Input per event: the truth record (d101_make_muons.py: tail_cm/head_cm) and
<det>/work/<RUN6>_<k>_<ARM>/tracking-pr.root  T_rec_charge (the multi-track PR fit; sub_cluster_id
splits segments) and Trun (dQdx_scale/offset: dQ/dx = ((q - off)/scale)/nq, e/cm).

Metrics (3 cm trimmed at each truth end, like the toy):
  res_t_med/p90   transverse residual (mm) in the direction perpendicular to both the track and x
  x_bias          signed median drift offset (mm)
  dev_med/p90     +-3 cm chord deviation of the fitted points (mm)
  snap_u/v/w      share of pu/pv/pw within +-0.15 of an integer (= a wire centre) over 0.30
  dqdx_med        median dQ/dx over the truth charge 5000 e/mm (no recombination in the sim)
  dqdx_cv, dip (< 0.5 x running median), dip_truth (< 0.6 x truth), acf1, r_res_q
  len_ratio       sum(dx) over the true span of the kept points

Usage:
  d101_sim_fit_eval.py --det pdvd --arm d101b [--truth /home/xqian/tmp/d101/sim/truth_pdvd.json] --out FILE.tsv
"""
import argparse, json, os, sys
import numpy as np
import uproot

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
RUN = {"pdvd": 900101, "pdhd": 900102}


def running_median(a, k=8):
    n = len(a)
    return np.array([np.median(a[max(0, i - k):min(n, i + k + 1)]) for i in range(n)])


def chord_dev(P, half=30.0):
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    dev = np.full(len(P), np.nan)
    for i in range(len(P)):
        j0 = np.searchsorted(L, L[i] - half); j1 = min(len(P) - 1, np.searchsorted(L, L[i] + half))
        if j0 >= i or j1 <= i:
            continue
        c = P[j1] - P[j0]; nc = np.linalg.norm(c)
        if nc < 1e-6:
            continue
        u = c / nc; v = P[i] - P[j0]
        dev[i] = np.linalg.norm(v - (v @ u) * u)
    return dev


def grade(ev, root, dqdx_true_per_cm=50000.0, end_cut=30.0):
    f = uproot.open(root)
    if "T_rec_charge" not in [k.split(";")[0] for k in f.keys()]:
        return None
    t = f["T_rec_charge"].arrays(library="np")
    if len(t["x"]) < 10:
        return None
    run = f["Trun"].arrays(["dQdx_scale", "dQdx_offset"], library="np")
    scale, off = float(run["dQdx_scale"][0]), float(run["dQdx_offset"][0])
    p0 = np.array(ev["tail_cm"]) * 10.0; p1 = np.array(ev["head_cm"]) * 10.0
    d = (p1 - p0) / np.linalg.norm(p1 - p0); L = np.linalg.norm(p1 - p0)
    P = np.c_[t["x"], t["y"], t["z"]] * 10.0                   # mm
    # order along the truth direction (segments of a single muon; the multi-track fit may split)
    s = (P - p0) @ d
    order = np.argsort(s, kind="stable")
    P, s = P[order], s[order]
    q, nq = t["q"][order], t["nq"][order]
    pu, pv, pw = t["pu"][order], t["pv"][order], t["pw"][order]
    inner = (s > end_cut) & (s < L - end_cut)
    if inner.sum() < 10:
        return None
    perp = (P - p0) - np.outer(s, d)
    xhat = np.array([1.0, 0, 0]); t1 = np.cross(d, xhat)
    if np.linalg.norm(t1) < 1e-6:
        t1 = np.array([0, 1.0, 0])
    t1 /= np.linalg.norm(t1)
    a1 = perp @ t1
    out = dict(n=int(inner.sum()), nfit=int(len(P)),
               cover=float((s[inner].max() - s[inner].min()) / (L - 2 * end_cut)),
               res_t_med=float(np.median(np.abs(a1[inner]))), res_t_p90=float(np.percentile(np.abs(a1[inner]), 90)),
               res_med=float(np.median(np.linalg.norm(perp[inner], axis=1))),
               x_bias=float(np.median(perp[inner, 0])))
    dev = chord_dev(P)
    out["dev_med"] = float(np.nanmedian(dev[inner])); out["dev_p90"] = float(np.nanpercentile(dev[inner], 90))
    for nm, arr in (("u", pu), ("v", pv), ("w", pw)):
        fr = np.mod(arr[inner], 1.0)
        out["snap_" + nm] = float(np.mean(np.minimum(fr, 1 - fr) < 0.15) / 0.30)
    with np.errstate(divide="ignore", invalid="ignore"):
        dqdx = ((q - off) / scale) / nq
    ok = inner & np.isfinite(dqdx) & (nq > 0)
    r = dqdx[ok] / dqdx_true_per_cm
    if len(r) > 10:
        med = np.median(r)
        out["dqdx_med"] = float(med)
        out["dqdx_cv"] = float(1.4826 * np.median(np.abs(r - med)) / med) if med > 0 else np.nan
        rm = running_median(r)
        out["dip"] = float(np.mean(r < 0.5 * rm)); out["dip_truth"] = float(np.mean(r < 0.6))
        x = r - r.mean()
        out["acf1"] = float(np.sum(x[1:] * x[:-1]) / np.sum(x * x)) if np.sum(x * x) > 0 else np.nan
        dd = np.linalg.norm(perp[ok], axis=1)
        out["r_res_q"] = float(np.corrcoef(dd, r)[0, 1]) if np.std(dd) > 0 and np.std(r) > 0 else np.nan
        sk = s[ok]
        out["len_ratio"] = float(np.sum(nq[ok] * 10.0) / (sk.max() - sk.min() + 1e-9))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, choices=RUN)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--truth", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    truth = json.load(open(a.truth or f"/home/xqian/tmp/d101/sim/truth_{a.det}.json"))
    evs = truth["events"]
    cols = ["k", "theta", "phi", "n", "nfit", "cover", "res_t_med", "res_t_p90", "res_med", "x_bias", "dev_med", "dev_p90",
            "snap_u", "snap_v", "snap_w", "dqdx_med", "dqdx_cv", "dip", "dip_truth", "acf1", "r_res_q", "len_ratio"]
    rows = []
    for ev in evs:
        k = ev["k"]
        root = f"{IMG}/{a.det}/work/{RUN[a.det]}_{k}_{a.arm}/tracking-pr.root"
        if not os.path.exists(root):
            print(f"k={k}: missing {root}", file=sys.stderr); continue
        g = grade(ev, root)
        if g is None:
            print(f"k={k}: no gradable fit", file=sys.stderr); continue
        g.update(k=k, theta=ev["theta_deg"], phi=ev["phi_deg"])
        rows.append(g)
    with open(a.out, "w") as fo:
        fo.write("\t".join(cols) + "\n")
        for r in rows:
            fo.write("\t".join((f"{r[c]:.5g}" if isinstance(r.get(c), float) else str(r.get(c, "nan"))) for c in cols) + "\n")
    base = [r for r in rows if r["k"] < 16]
    summ = {c: float(np.nanmedian([r.get(c, np.nan) for r in base])) for c in cols[3:]}
    print(f"{a.det} {a.arm}: graded {len(rows)} events ({len(base)} grid); medians over the 16-track grid:")
    print("  " + "  ".join(f"{c}={summ[c]:.4g}" for c in cols[3:]))


if __name__ == "__main__":
    main()
