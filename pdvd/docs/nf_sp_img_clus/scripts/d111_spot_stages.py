#!/usr/bin/env python3
"""doc pdvd/111 -- the owner's spots stage by stage: image, Steiner cloud, the Steiner Dijkstra seed of the persisted
pass, the first-pass fit, the final trajectory, and the points the area smoothing / skip step undid, on the 2x2 lever
arms (A0 neither, K fit keys, S charge_stepped sampler, A1 both = production), in one fit-independent image frame.

Inputs: the d111 trace arms (d111_run_arms.sh with WCT_STM_PATH_DEBUG=1 WCT_TRAJ_ASSOC_DEBUG=1) and their Bee zips /
tracking-stm.root.  The frame is d111_spot_frame.frame() of the first arm's image (every arm shares the pctree).

Outputs: <out>_<spot>.png (rows: transverse e2, e3; columns: arms) and <out>.tsv (per spot, arm: ridge offset p90/max
of the seed, fit1, fit2 and final within the window, the solve's recoveries and how many the post-solve steps undid).

Usage:
  d111_spot_stages.py --out figs/111_stages
"""
import argparse, os, sys
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d111_stage_attrib as A      # noqa: E402
from d111_spot_frame import frame   # noqa: E402

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
SPOTS = [
    ("h1", "pdhd", "029107_16", 108, (75.3, 438.3, 450.1), 12.0, ["d111hA0", "d111hK", "d111hS", "d111hA1"]),
    ("h2", "pdhd", "029107_16", 106, (182.0, 537.4, 401.8), 24.0, ["d111hA0", "d111hK", "d111hS", "d111hA1"]),
    ("v1", "pdvd", "039349_20", 80, (313.7, 95.9, 124.9), 12.0, ["d111vA0", "d111vK", "d111vS", "d111vA1"]),
    ("v2", "pdvd", "039349_20", 80, (94.5, 30.7, 165.0), 12.0, ["d111vA0", "d111vK", "d111vS", "d111vA1"]),
    ("v3", "pdvd", "039349_20", 25, (-270.2, -243.7, 36.0), 12.0, ["d111vA0", "d111vK", "d111vS", "d111vA1"]),
]
LABEL = {"A0": "A0 stepped, no fit keys", "K": "K stepped + fit keys", "S": "S charge_stepped, no fit keys",
         "A1": "A1 charge_stepped + fit keys (production)"}


def load(det, ev, arm, cl, click, R):
    d = f"{IMG}/{det}/work/{ev}_{arm}"
    blocks, assoc = A.parse_trace(f"/home/xqian/tmp/d111/arm_{arm}/evt_{ev}.log.gz")
    t = uproot.open(f"{d}/tracking-stm.root")["T_rec_charge"].arrays(library="np")
    img = A.bee_layer(f"{d}/mabc-pr.zip", "clustering-global")
    stc = A.bee_layer(f"{d}/mabc-pr.zip", "steiner_graph-global")
    mi = img[1] == cl
    out = dict(img=(img[0][mi], np.clip(img[2][mi], 1, None)), ridge=A.Ridge(img[0][mi], img[2][mi]))
    out["cloud"] = stc[0][stc[1] == cl] if stc is not None and len(stc[0]) else np.zeros((0, 3))
    best = None
    for ps in np.unique(t["pass"][t["cluster_id"] == cl]):
        m = (t["cluster_id"] == cl) & (t["pass"] == ps)
        F = np.c_[t["x"][m], t["y"][m], t["z"][m]]
        if np.min(np.linalg.norm(F - click, axis=1)) > R:
            continue
        bl = [b for b in blocks if b["cl"] == cl and b["rnd"] == "r2" and "final" in b["stages"]
              and len(b["stages"]["final"]) == len(F) and np.max(np.abs(b["stages"]["final"] - F)) < 0.01]
        if bl:
            best = (ps, F, t["q"][m], bl[-1])
            break
    out["rec"] = best
    out["assoc"] = assoc
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lines = ["spot\tarm\tpass\tseed_p90\tseed_max\tfit1_p90\tfit1_max\tfit2_p90\tfinal_p90\tfinal_max\t"
             "solve_recovered\tundone_revert\tundone_skip"]
    for spot, det, ev, cl, click, R, arms in SPOTS:
        click = np.array(click)
        D = {arm: load(det, ev, arm, cl, click, R) for arm in arms}
        P0, w0 = D[arms[0]]["img"]
        near = np.linalg.norm(P0 - click, axis=1) < R
        c, e1, e2, e3 = frame(P0[near], w0[near])
        proj = lambda X: ((X - c) @ e1, np.c_[(X - c) @ e2, (X - c) @ e3])
        fig, axs = plt.subplots(2, len(arms), figsize=(4.3 * len(arms), 6.6), sharex=True, sharey="row", squeeze=False)
        for j, arm in enumerate(arms):
            dd = D[arm]; lab = arm.replace("d111h", "").replace("d111v", "")
            s_i, T_i = proj(P0[near])
            cloud = dd["cloud"][np.linalg.norm(dd["cloud"] - click, axis=1) < R] if len(dd["cloud"]) else dd["cloud"]
            rec = dd["rec"]
            for r in range(2):
                ax = axs[r, j]
                ax.scatter(s_i, T_i[:, r], s=4, c="0.78", label="image")
                if len(cloud):
                    s_c, T_c = proj(cloud); ax.scatter(s_c, T_c[:, r], s=4, c="tab:green", alpha=0.5, label="Steiner cloud")
                if rec is not None:
                    ps, F, q, b = rec
                    for st, sty, col in (("seed", "-", "tab:orange"), ("fit1", ":", "tab:cyan"), ("final", "-", "tab:blue")):
                        Pst = b["stages"].get(st)
                        if Pst is None or len(Pst) == 0:
                            continue
                        k = np.linalg.norm(Pst - click, axis=1) < 1.3 * R
                        s_p, T_p = proj(Pst[k])
                        ax.plot(s_p, T_p[:, r], sty, color=col, lw=1.4 if st != "fit1" else 1.0,
                                marker="o" if st == "seed" else ("." if st == "final" else None), ms=3, label=st)
                    X = dd["assoc"].get(b["calls"].get(2))
                    if X is not None and len(X):
                        k = (np.linalg.norm(X[:, A.A_INIT:A.A_INIT + 3] - click, axis=1) < R) & (X[:, A.A_REV] == 1)
                        if k.any():
                            s_x, T_x = proj(X[k][:, A.A_SOL:A.A_SOL + 3])
                            ax.plot(s_x, T_x[:, r], "x", color="red", ms=6, label="fit2 solve, reverted")
                if r == 0:
                    ax.set_title(f"{spot} cl{cl}: {LABEL.get(lab, lab)}", fontsize=9)
                ax.set_ylabel(["e2 (cm)", "e3 (cm, drift side)"][r])
                if r == 1:
                    ax.set_xlabel("s along image PCA (cm)")
            # table row
            if rec is not None:
                ps, F, q, b = rec
                w = np.linalg.norm(F - click, axis=1) < R
                Fw = F[w]; Rg = dd["ridge"]; vals = []
                for st in ("seed", "fit1", "fit2", "final"):
                    Q, _ = A.closest_on_polyline(b["stages"][st], Fw); d_ = Rg.offset(Q)
                    vals.append((np.percentile(d_, 90), d_.max()))
                rec_n = rev_n = skip_n = 0
                for meth in (1, 2):
                    X = dd["assoc"].get(b["calls"].get(meth))
                    if X is None:
                        continue
                    k = np.linalg.norm(X[:, A.A_INIT:A.A_INIT + 3] - click, axis=1) < R
                    Y = X[k]
                    di = Rg.offset(Y[:, A.A_INIT:A.A_INIT + 3]); ds = Rg.offset(Y[:, A.A_SOL:A.A_SOL + 3])
                    ok = (di > 1) & (ds <= 1)
                    rec_n += int(ok.sum()); rev_n += int((ok & (Y[:, A.A_REV] == 1)).sum()); skip_n += int((ok & (Y[:, A.A_SKIP] == 1)).sum())
                lines.append(f"{spot}\t{arm}\t{ps}\t{vals[0][0]:.2f}\t{vals[0][1]:.2f}\t{vals[1][0]:.2f}\t{vals[1][1]:.2f}\t"
                             f"{vals[2][0]:.2f}\t{vals[3][0]:.2f}\t{vals[3][1]:.2f}\t{rec_n}\t{rev_n}\t{skip_n}")
        axs[0, 0].legend(fontsize=7, loc="best")
        fig.tight_layout(); fig.savefig(f"{a.out}_{spot}.png", dpi=110); plt.close(fig)
    open(f"{a.out}.tsv", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
