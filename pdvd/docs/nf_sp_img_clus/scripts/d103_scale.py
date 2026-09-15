#!/usr/bin/env python3
"""doc pdvd/103 sec 4.4 -- does a trajectory change move the dQ/dx scale the taggers' MIP references were set on?  Read-only.

Per CheckSTM_Michel candidate present in both arms (T_stm_michel, same (event, cluster_id)):
  plateau_med ratio arm/base, tail_med ratio, muon_len ratio, n_profile_pts ratio, and from T_stm_michel_pts (role 1 =
  the muon profile rows) the summed q and median q ratios.  The per-candidate correlation of log(plateau ratio) with
  -log(muon_len ratio) separates "shorter dx" (correlated) from "more collected charge" (uncorrelated).
  The all-candidate plateau_med median per arm is printed against the taggers' references (TaggerCheckSTM mip_dqdx,
  CheckSTM_Michel mip_dqdx_median; compiled PR config).
  Stop / entry shift along the base track direction on candidates whose is_stm did not change: an end that moved would
  change the Bragg window; one that did not leaves only the profile.

Usage: d103_scale.py --det pdhd --base d101hnew --arms d101hkf,d102hocs,d102hcs
"""
import argparse, glob, os
import numpy as np
import uproot

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
REF = {"pdhd": (56000, 48000), "pdvd": (55000, 47000)}
BR = ["cluster_id", "is_stm", "plateau_med", "tail_med", "muon_len", "n_profile_pts",
      "entry_x", "entry_y", "entry_z", "stop_x", "stop_y", "stop_z"]


def load(det, arm):
    rows, prof = {}, {}
    for d in glob.glob(f"{IMG}/{det}/work/*_{arm}"):
        ev = os.path.basename(d)[:-len(arm) - 1]
        try:
            f = uproot.open(f"{d}/tracking-pr.root")
            t = f["T_stm_michel"].arrays(BR, library="np")
            p = f["T_stm_michel_pts"].arrays(["cluster_id", "q", "role"], library="np")
        except Exception:
            continue
        for i in range(len(t["cluster_id"])):
            rows[(ev, int(t["cluster_id"][i]))] = {b: float(t[b][i]) for b in BR}
        cid = p["cluster_id"].astype(int)
        for c in np.unique(cid):
            m = (cid == c) & (p["role"] == 1)
            if m.sum():
                prof[(ev, int(c))] = (float(np.nansum(p["q"][m])), float(np.nanmedian(p["q"][m])))
    return rows, prof


def s(x):
    x = np.asarray(x, float)
    return f"median {np.median(x):.3f} (p25 {np.percentile(x, 25):.3f}, p75 {np.percentile(x, 75):.3f})"


def v(r, w):
    return np.array([r[w + "_x"], r[w + "_y"], r[w + "_z"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, choices=["pdhd", "pdvd"])
    ap.add_argument("--base", required=True)
    ap.add_argument("--arms", required=True)
    a = ap.parse_args()
    B, PB = load(a.det, a.base)
    mip, mipmed = REF[a.det]
    print(f"[{a.det}] references: TaggerCheckSTM mip_dqdx {mip}, CheckSTM_Michel mip_dqdx_median {mipmed} (e/cm)")
    print(f"  {a.base:11s} all-candidate plateau_med median {np.median([r['plateau_med'] for r in B.values() if r['plateau_med'] > 0]):.0f}")
    for arm in a.arms.split(","):
        A, PA = load(a.det, arm)
        print(f"  {arm:11s} all-candidate plateau_med median {np.median([r['plateau_med'] for r in A.values() if r['plateau_med'] > 0]):.0f}")
        K = [k for k in B.keys() & A.keys() if B[k]["plateau_med"] > 0 and A[k]["plateau_med"] > 0 and B[k]["muon_len"] > 0]
        pr = np.array([A[k]["plateau_med"] / B[k]["plateau_med"] for k in K])
        lr = np.array([A[k]["muon_len"] / B[k]["muon_len"] for k in K])
        tr = [A[k]["tail_med"] / B[k]["tail_med"] for k in K if B[k]["tail_med"] > 0 and A[k]["tail_med"] > 0]
        npr = [A[k]["n_profile_pts"] / max(1, B[k]["n_profile_pts"]) for k in K]
        Kq = [k for k in K if k in PB and k in PA and PB[k][0] > 0 and PB[k][1] > 0]
        print(f"    paired candidates n {len(K)}")
        print(f"      plateau_med ratio   {s(pr)}")
        print(f"      tail_med ratio      {s(tr)}")
        print(f"      muon_len ratio      {s(lr)}")
        print(f"      n_profile_pts ratio {s(npr)}")
        print(f"      summed profile q ratio {s([PA[k][0] / PB[k][0] for k in Kq])}; median q ratio {s([PA[k][1] / PB[k][1] for k in Kq])}")
        print(f"      corr(log plateau ratio, -log muon_len ratio) {np.corrcoef(np.log(pr), -np.log(lr))[0, 1]:+.2f}")
        ax, ae = [], []
        for k in B.keys() & A.keys():
            if B[k]["is_stm"] != A[k]["is_stm"]:
                continue
            d = v(B[k], "stop") - v(B[k], "entry")
            n = np.linalg.norm(d)
            if n < 1:
                continue
            u = d / n
            ax.append(np.dot(v(A[k], "stop") - v(B[k], "stop"), u))
            ae.append(np.dot(v(A[k], "entry") - v(B[k], "entry"), u))
        print(f"      same-verdict candidates n {len(ax)}: stop shift along track (cm) {s(ax)}; entry shift {s(ae)}")


if __name__ == "__main__":
    main()
