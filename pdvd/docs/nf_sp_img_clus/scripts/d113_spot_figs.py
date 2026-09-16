#!/usr/bin/env python3
"""doc pdvd/113 -- the owner's spots under each retile_mode, side by side.  Read-only.

Per spot (h1, h2 on PDHD 029107_16; v1, v2, v3 on PDVD 039349_20; windows as doc 111) and per arm (production = full
retile, then no_paint, footprint, none), from the arm's dump and outputs:
  * image     -- the cluster's clustering-global points (mabc-pr.zip), grey;
  * cloud     -- the Steiner stage's cloud (STGR: retiled for full / no_paint / footprint, re-sampled for none), light;
  * terminals -- non-extreme Steiner terminals, coloured by plane class (d112: 3live / 1blank / 1dead / 2blank+);
  * seed      -- the cluster's first STM rough walk replayed on its dumped steiner_graph (the Steiner seed, doc 111 r2);
  * fit       -- the persisted STM fit rows (T_rec_charge), the pass with the most rows in the window.
All drawn in one image-only frame per spot (d111_spot_frame.frame: charge-weighted PCA of the image within the window,
s along the track, e2 / e3 across it), the same frame for every arm.  Third column: ridge offset (d111_stage_attrib.Ridge)
of the seed and of the fit against s.

Outputs: <out>_<spot>.png and <out>.tsv (spot, arm, seed max offset, fit max offset, terminals / 1blank terminals / 1blank
terminals > 1 cm off in the window).

Usage: d113_spot_figs.py --out figs/113_spot [--arms pdhd=d113hbase:full,d113hnopaint:no_paint,... --arms pdvd=...]
"""
import argparse, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import d111s_common as C
from d111_spot_frame import frame
from d111s_replay import SPOTS
from d112_terminal_planes import vertex_class, vertex_role, CLASSES
A = C.A

CLS_COL = ("tab:green", "tab:red", "tab:blue", "tab:orange")
DEFAULT_ARMS = {
    "pdhd": [("d113hbase", "production (full retile)"), ("d113hnopaint", "no_paint"), ("d113hfoot", "footprint"),
             ("d113hnone", "none (no retile)")],
    "pdvd": [("d113vbase", "production (full retile)"), ("d113vnopaint", "no_paint"), ("d113vfoot", "footprint"),
             ("d113vnone", "none (no retile)")],
}


def load(det, arm, logd, ev, cl, click, win):
    d = f"{C.IMG}/{det}/work/{ev}_{arm}"
    img = A.bee_layer(f"{d}/mabc-pr.zip", "clustering-global")
    mi = img[1] == cl
    Pimg, qimg = img[0][mi], np.clip(img[2][mi], 0, None)
    R = A.Ridge(Pimg, img[2][mi])
    tr = C.parse_trace(f"{logd}/evt_{ev}.log.gz")
    out = dict(Pimg=Pimg, qimg=qimg, R=R, sb=None, seed=None, fit=None)
    for order, c, kind, src, dst, npath in tr["rp"]:
        if c != cl or kind != "rough":
            continue
        sb = C.graph_for(tr, cl, order)
        if sb is None or max(src, dst) >= sb.nv:
            continue
        p, _ = sb.walk(src, dst)
        if p is None:
            continue
        seed = sb.V[p]
        if np.min(np.linalg.norm(seed - click, axis=1)) > win:
            continue
        out["sb"], out["seed"] = sb, seed
        break
    t = uproot.open(f"{d}/tracking-stm.root")["T_rec_charge"].arrays(["cluster_id", "pass", "x", "y", "z"], library="np")
    best, nbest = None, 0
    for ps in sorted(set(t["pass"][t["cluster_id"] == cl].tolist())):
        m = (t["cluster_id"] == cl) & (t["pass"] == ps)
        F = np.c_[t["x"][m], t["y"][m], t["z"][m]]
        n = int((np.linalg.norm(F - click, axis=1) <= win).sum())
        if n > nbest:
            best, nbest = F, n
    out["fit"] = best
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", action="append", default=[], help="det=arm:label,arm:label,...")
    ap.add_argument("--logd-root", default="/home/xqian/tmp/d113")
    ap.add_argument("--spots", default="h1,h2,v1,v2,v3")
    a = ap.parse_args()
    arms = dict(DEFAULT_ARMS)
    for spec in a.arms:
        det, rest = spec.split("=")
        arms[det] = [tuple(x.split(":", 1)) for x in rest.split(",")]
    rows = ["spot\tarm\tlabel\tseed_max_off_cm\tfit_max_off_cm\tterminals\tterm_1blank\tterm_1blank_off"]
    for name in a.spots.split(","):
        det, ev, cl, click, win = SPOTS[name]
        click = np.array(click)
        L = arms[det]
        fig, axs = plt.subplots(len(L), 3, figsize=(18, 3.6 * len(L)), squeeze=False)
        frame_of = None
        for r, (arm, label) in enumerate(L):
            logd = f"{a.logd_root}/arm_{arm}"
            if not os.path.exists(f"{logd}/evt_{ev}.log.gz"):
                logd = f"/home/xqian/tmp/d111/arm_{arm}"
            X = load(det, arm, logd, ev, cl, click, win)
            if frame_of is None:
                near = np.linalg.norm(X["Pimg"] - click, axis=1) < win
                frame_of = frame(X["Pimg"][near], X["qimg"][near])
            c, e1, e2, e3 = frame_of
            pr = lambda P: ((P - c) @ e1, (P - c) @ e2, (P - c) @ e3)
            R = X["R"]
            near = np.linalg.norm(X["Pimg"] - click, axis=1) < win
            si, ai, bi = pr(X["Pimg"][near])
            sm = fm = np.nan
            nt = nb = nbo = 0
            for k, ax in enumerate(axs[r, :2]):
                ax.scatter(si, (ai, bi)[k], s=6, c="0.72", zorder=1)
            sb = X["sb"]
            if sb is not None:
                cls, _, _ = vertex_class(sb)
                role = vertex_role(sb)
                vin = np.linalg.norm(sb.V - click, axis=1) < win
                rin = np.linalg.norm(sb.R - click, axis=1) < win
                sr, ar, br = pr(sb.R[rin])
                for k, ax in enumerate(axs[r, :2]):
                    ax.scatter(sr, (ar, br)[k], s=2, c="tab:cyan", alpha=0.35, zorder=2)
                off = R.offset(sb.V) if R.ok else np.zeros(sb.nv)
                for ci in range(4):
                    m = vin & (role == 0) & (cls == ci)
                    sv, av, bv = pr(sb.V[m])
                    for k, ax in enumerate(axs[r, :2]):
                        ax.scatter(sv, (av, bv)[k], s=46, marker="^", color=CLS_COL[ci], edgecolor="k", lw=0.4, zorder=4)
                nt = int((vin & (role == 0)).sum())
                nb = int((vin & (role == 0) & (cls == 1)).sum())
                nbo = int((vin & (role == 0) & (cls == 1) & (off > 1.0)).sum())
                seed = X["seed"]
                ss, sa, sbb = pr(seed)
                keep = np.abs(ss) < 1.3 * win
                for k, ax in enumerate(axs[r, :2]):
                    ax.plot(ss[keep], (sa, sbb)[k][keep], "-", color="k", lw=1.2, zorder=3)
                Xs, _ = C.polyline_samples(seed, 0.3)
                ms = np.linalg.norm(Xs - click, axis=1) <= win
                if ms.any() and R.ok:
                    sm = float(np.max(R.offset(Xs[ms])))
                    axs[r, 2].plot(((Xs[ms] - c) @ e1), R.offset(Xs[ms]), "-", color="k", lw=1.2, label="seed")
            F = X["fit"]
            if F is not None:
                fs, fa, fb = pr(F)
                keep = np.abs(fs) < 1.3 * win
                for k, ax in enumerate(axs[r, :2]):
                    ax.plot(fs[keep], (fa, fb)[k][keep], ".", color="tab:purple", ms=4, zorder=5)
                mf = np.linalg.norm(F - click, axis=1) <= win
                if mf.any() and R.ok:
                    fm = float(np.max(R.offset(F[mf])))
                    o = np.argsort(fs[mf])
                    axs[r, 2].plot(fs[mf][o], R.offset(F[mf])[o], ".-", color="tab:purple", lw=0.8, ms=4, label="fit")
            for k, ax in enumerate(axs[r, :2]):
                ax.set_xlim(-1.3 * win, 1.3 * win)
                ax.set_ylim(-4, 4)
                ax.set_ylabel(f"{'e2' if k == 0 else 'e3'} (cm)")
            axs[r, 0].set_title(f"{label} [{arm}]: seed max {sm:.2f} cm, fit max {fm:.2f} cm; terminals {nt}, "
                                f"1blank {nb} ({nbo} > 1 cm off)", fontsize=9, loc="left")
            axs[r, 2].axhline(1.0, color="r", lw=0.8, ls="--")
            axs[r, 2].set_xlim(-1.3 * win, 1.3 * win)
            axs[r, 2].set_ylim(0, max(4.0, np.nan_to_num(max(sm, fm), nan=0) + 0.5))
            axs[r, 2].set_ylabel("ridge offset (cm)")
            rows.append(f"{name}\t{arm}\t{label}\t{sm:.3f}\t{fm:.3f}\t{nt}\t{nb}\t{nbo}")
        for ax in axs[-1]:
            ax.set_xlabel("s along the track (cm, image frame)")
        h = [Line2D([], [], ls="", marker="o", color="0.72", label="image (clustering)"),
             Line2D([], [], ls="", marker="o", color="tab:cyan", label="Steiner-stage cloud"),
             Line2D([], [], color="k", label="Steiner seed (rough walk)"),
             Line2D([], [], ls="", marker=".", color="tab:purple", label="STM fit rows")]
        h += [Line2D([], [], ls="", marker="^", color=CLS_COL[i], markeredgecolor="k", label=f"terminal {CLASSES[i]}")
              for i in range(4)]
        axs[0, 1].legend(handles=h, fontsize=7, loc="upper right", ncol=2)
        fig.suptitle(f"{name}: {det} {ev} cluster {cl}, click ({', '.join('%.1f' % v for v in click)}), window {win:.0f} cm -- the same image "
                     f"frame in every row", fontsize=11)
        fig.tight_layout()
        fig.savefig(f"{a.out}_{name}.png", dpi=90)
        plt.close(fig)
        print(f"wrote {a.out}_{name}.png")
    open(a.out + ".tsv", "w").write("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
