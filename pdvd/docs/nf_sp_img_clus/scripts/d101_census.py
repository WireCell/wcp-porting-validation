#!/usr/bin/env python3
"""doc pdvd/101 Phase 0 -- read-only census of the fitted trajectory and its dQ/dx on
existing production arms of PDHD, PDVD and SBND.

For every fitted track run it measures what the owner described on the Bee display
(PDHD 028084_21 cluster 55): a zig-zag trajectory and a dQ/dx that oscillates and drops
where the trajectory leaves the charge.  Nothing here is fitted; every quantity is read
from T_rec_charge (tracking-stm.root = TaggerCheckSTM single-track fit, tracking-pr.root
= PR multi-track fit):

  snap_<p>   lattice snap index of the fitted points in plane p: the share of points
             whose projected wire coordinate p* lies within +-0.15 wire of an integer
             (a wire centre: TrackFitting.cxx:829-857 builds p* so that integer = the
             ctpc wire centre of PointTreeBuilding.cxx:313), divided by the 0.30 a
             uniform phase gives.  1.0 = no snapping; pt is the drift-slice control.
  dev        transverse distance of a point from the chord of its own +-3 cm
             neighbourhood (cm) -- the zig-zag amplitude.
  cv         dQ/dx robust spread (1.4826*MAD/median) on the interior.
  dip        share of interior points with dQ/dx < 0.5 x the running median (+-8 pts).
  acf1       lag-1 autocorrelation of dQ/dx / running median.
  r_dev_q    Pearson correlation of dev with dQ/dx / running median.

dQ/dx = ((q - dQdx_offset)/dQdx_scale)/nq  (e/cm; d42_dqdx_rr.py:7).  STM rows: status 0
only.  Runs are split where consecutive points are > 2 cm apart (PR trees mix segments),
need >= 40 points and >= 20 cm, and lose 3 cm at each end (end trims are a different
question, doc pdvd/32/38).  PR shower segments (flag_shower) are dropped.

Usage:
  d101_census.py --out DIR  [--max-files N]
Writes DIR/101_census_runs.tsv (one row per run) and DIR/101_census_summary.txt.
"""
import argparse, glob, os, sys
import numpy as np
import uproot

WCP = "/home/xqian/toolkit-dev/wcp-porting-img"
ARMS = [
    # label, detector, glob, kind
    ("pdhd_stm", "pdhd", WCP + "/pdhd/work/*_h28prod/tracking-stm.root", "stm"),
    ("pdhd_pr", "pdhd", WCP + "/pdhd/work/*_h28prod/tracking-pr.root", "pr"),
    ("pdvd_stm", "pdvd", WCP + "/pdvd/work/*_p96vprod/tracking-stm.root", "stm"),
    ("pdvd_pr", "pdvd", WCP + "/pdvd/work/*_p96vprod/tracking-pr.root", "pr"),
    ("sbnd_pr", "sbnd", WCP + "/sbnd/sbnd_xin/work-mcp2k-d146sv25/*/tracking-pr.root", "pr"),
    ("sbnd_pr1k", "sbnd", WCP + "/sbnd/sbnd_xin/work-mcp1k-d146sv25/*/tracking-pr.root", "pr"),
]
PLANES = ("pu", "pv", "pw", "pt")


def chord_dev(P, L, half=3.0):
    n = len(P)
    dev = np.full(n, np.nan)
    for i in range(n):
        j0 = np.searchsorted(L, L[i] - half)
        j1 = min(n - 1, np.searchsorted(L, L[i] + half))
        if j0 >= i or j1 <= i:
            continue
        d = P[j1] - P[j0]
        nd = np.linalg.norm(d)
        if nd < 1e-6:
            continue
        u = d / nd
        v = P[i] - P[j0]
        dev[i] = np.linalg.norm(v - np.dot(v, u) * u)
    return dev


def running_median(a, k=8):
    n = len(a)
    return np.array([np.median(a[max(0, i - k):min(n, i + k + 1)]) for i in range(n)])


def runs_of(idx, P, maxgap=2.0):
    """Split an index array (in file order) where consecutive points jump > maxgap cm."""
    if len(idx) == 0:
        return []
    st = np.linalg.norm(np.diff(P[idx], axis=0), axis=1)
    cuts = np.where(st > maxgap)[0]
    out, s = [], 0
    for c in cuts:
        out.append(idx[s:c + 1]); s = c + 1
    out.append(idx[s:])
    return out


def measure(t, scale, offset, sel, label, fname):
    P = np.c_[t["x"][sel], t["y"][sel], t["z"][sel]]
    rows = []
    idx_all = np.arange(len(P))
    for r in runs_of(idx_all, P):
        if len(r) < 40:
            continue
        Pr = P[r]
        st = np.linalg.norm(np.diff(Pr, axis=0), axis=1)
        L = np.r_[0, np.cumsum(st)]
        if L[-1] < 20.0:
            continue
        inner = (L > 3.0) & (L < L[-1] - 3.0)
        if inner.sum() < 25:
            continue
        q = t["q"][sel][r]; nq = t["nq"][sel][r]
        with np.errstate(divide="ignore", invalid="ignore"):
            dqdx = ((q - offset) / scale) / nq
        ok = inner & np.isfinite(dqdx) & (nq > 0)
        if ok.sum() < 25:
            continue
        dev = chord_dev(Pr, L)
        dq = dqdx[ok]
        rm = running_median(dq)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = dq / rm
        good = np.isfinite(rel) & (rm > 0)
        med = np.median(dq)
        row = dict(label=label, file=fname, n=int(ok.sum()), L=float(L[-1]),
                   step=float(np.median(st)),
                   dev_med=float(np.nanmedian(dev[ok])), dev_p90=float(np.nanpercentile(dev[ok], 90)),
                   cv=float(1.4826 * np.median(np.abs(dq - med)) / med) if med > 0 else np.nan,
                   dip=float(np.mean(rel[good] < 0.5)) if good.any() else np.nan,
                   acf1=np.nan, r_dev_q=np.nan)
        if good.sum() > 10:
            x = rel[good] - rel[good].mean()
            if x.std() > 0:
                row["acf1"] = float(np.sum(x[1:] * x[:-1]) / np.sum(x * x))
            d = dev[ok][good]
            m2 = np.isfinite(d)
            if m2.sum() > 10 and np.std(d[m2]) > 0:
                row["r_dev_q"] = float(np.corrcoef(d[m2], rel[good][m2])[0, 1])
        for p in PLANES:
            fr = np.mod(t[p][sel][r][ok], 1.0)
            near = np.minimum(fr, 1.0 - fr) < 0.15
            row["snap_" + p] = float(np.mean(near) / 0.30)
        # wire advance per cm, per plane (stratifies snapping by crossing rate)
        for p in ("pu", "pv", "pw"):
            dp = np.abs(np.diff(t[p][sel][r]))
            dp = dp[dp < 5]
            row["adv_" + p] = float(np.sum(dp) / L[-1]) if L[-1] > 0 else np.nan
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-files", type=int, default=0)
    # sec 6: grade other arms with the same ruler, e.g. --arm pdvd_pr_new:pdvd:d101vnew:pr
    ap.add_argument("--arm", action="append", default=[],
                    help="label:det:ARMTAG:kind (kind pr|stm); replaces the default production arm list")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    allrows = []
    arms = ARMS
    if a.arm:
        arms = []
        for spec in a.arm:
            label, det, tag, kind = spec.split(":")
            arms.append((label, det, "%s/%s/work/*_%s/tracking-%s.root" % (WCP, det, tag, kind), kind))
    for label, det, pat, kind in arms:
        files = sorted(glob.glob(pat))
        if a.max_files:
            files = files[:a.max_files]
        nf = 0
        for f in files:
            try:
                ff = uproot.open(f)
                if "T_rec_charge" not in [k.split(";")[0] for k in ff.keys()]:
                    continue
                br = ["x", "y", "z", "q", "nq", "pu", "pv", "pw", "pt", "cluster_id"]
                tr = ff["T_rec_charge"]
                have = set(tr.keys())
                for extra in ("status", "pass", "flag_shower", "sub_cluster_id"):
                    if extra in have:
                        br.append(extra)
                t = tr.arrays(br, library="np")
                run = ff["Trun"].arrays(["dQdx_scale", "dQdx_offset"], library="np")
            except Exception as e:  # noqa: BLE001 -- a broken file is reported, not fatal
                print("skip", f, e, file=sys.stderr)
                continue
            scale = float(run["dQdx_scale"][0]); offset = float(run["dQdx_offset"][0])
            nf += 1
            cid = t["cluster_id"]
            keys = [cid]
            if kind == "stm":
                keys.append(t["pass"])
            elif "sub_cluster_id" in t:
                keys.append(t["sub_cluster_id"])
            combo = np.stack(keys, axis=1)
            for key in np.unique(combo, axis=0):
                sel = np.all(combo == key, axis=1)
                if kind == "stm" and "status" in t and np.any(t["status"][sel] != 0):
                    continue
                if kind == "pr" and "flag_shower" in t and np.any(t["flag_shower"][sel] == 1):
                    continue
                allrows.extend(measure(t, scale, offset, sel, label, os.path.relpath(f, WCP)))
        print(f"{label}: {nf} files", file=sys.stderr)
    cols = ["label", "file", "n", "L", "step", "dev_med", "dev_p90", "cv", "dip", "acf1", "r_dev_q",
            "snap_pu", "snap_pv", "snap_pw", "snap_pt", "adv_pu", "adv_pv", "adv_pw"]
    with open(os.path.join(a.out, "101_census_runs.tsv"), "w") as fo:
        fo.write("\t".join(cols) + "\n")
        for r in allrows:
            fo.write("\t".join(str(r[c]) if not isinstance(r[c], float) else f"{r[c]:.5g}" for c in cols) + "\n")
    lines = ["# doc pdvd/101 Phase 0 census -- per-arm medians over runs (n-weighted means for snap/dip)",
             "label\truns\tpoints\tstep_cm\tdev_med_cm\tdev_p90_cm\tcv\tdip\tacf1\tr_dev_q\tsnap_pu\tsnap_pv\tsnap_pw\tsnap_pt"]
    for label, *_ in arms:
        rr = [r for r in allrows if r["label"] == label]
        if not rr:
            continue
        w = np.array([r["n"] for r in rr], float)
        def med(k):
            v = np.array([r[k] for r in rr], float); return np.nanmedian(v)
        def wmean(k):
            v = np.array([r[k] for r in rr], float); m = np.isfinite(v); return np.sum(v[m] * w[m]) / np.sum(w[m])
        lines.append("\t".join([label, str(len(rr)), str(int(w.sum())), f"{med('step'):.3f}", f"{med('dev_med'):.3f}",
                               f"{med('dev_p90'):.3f}", f"{med('cv'):.3f}", f"{wmean('dip'):.4f}", f"{med('acf1'):.3f}",
                               f"{med('r_dev_q'):.3f}"] + [f"{wmean('snap_' + p):.3f}" for p in PLANES]))
    txt = "\n".join(lines)
    open(os.path.join(a.out, "101_census_summary.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
