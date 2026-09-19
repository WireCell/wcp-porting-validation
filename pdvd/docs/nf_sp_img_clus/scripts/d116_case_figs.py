#!/usr/bin/env python3
"""doc pdvd/116 sec 2 -- case figures of tag movers: for one (event, cluster) the tagger's own record in two arms,
side by side.  Top row: the muon chain's dQ/dx profile (T_stm_michel_pts role 1, q = dQ/dx in e/cm against the
residual range rr) with the plateau / mip scale; bottom row: the chain and the Michel / delta members (roles 3 / 2)
in (x, y) and (x, z) with the image (mabc-pr.zip clustering-global) around the stop.  The panel titles carry the
persisted verdict fields (is_stm, reject bits, michel_found, conn, KE).  Read-only.

Usage: d116_case_figs.py --det pdvd --a d115voff --b d115vp3bwp05 --keys 039349_10/66,039252_12/117 --out figs/116_case_pdvd
Writes <out>_<event>_<cluster>.png per key.
"""
import argparse, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import d111s_common as C
from d116_mechanism import bits
A_ = C.A
MIP = {"pdhd": 56000.0, "pdvd": 55000.0}


def rec(det, arm, ev, cid):
    d = f"{C.IMG}/{det}/work/{ev}_{arm}"
    f = uproot.open(f"{d}/tracking-pr.root")
    t = f["T_stm_michel"].arrays(["cluster_id", "is_stm", "reject_bits", "michel_found", "michel_conn_type", "michel_ke_best", "michel_len",
                                  "plateau_med", "ks_mu", "ks_flat", "stop_x", "stop_y", "stop_z", "tagger_stop_x", "tagger_stop_y", "tagger_stop_z", "muon_len"], library="np")
    m = t["cluster_id"] == cid
    r = {k: float(t[k][m][0]) for k in t if k != "cluster_id"} if m.any() else None
    p = f["T_stm_michel_pts"].arrays(["cluster_id", "role", "x", "y", "z", "q", "rr"], library="np")
    mm = p["cluster_id"] == cid
    P = {role: np.c_[p["x"][mm & (p["role"] == role)], p["y"][mm & (p["role"] == role)], p["z"][mm & (p["role"] == role)],
                     p["q"][mm & (p["role"] == role)], p["rr"][mm & (p["role"] == role)]] for role in (1, 2, 3)}
    img = A_.bee_layer(f"{d}/mabc-pr.zip", "clustering-global")
    return r, P, img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--keys", required=True); ap.add_argument("--out", required=True); ap.add_argument("--win", type=float, default=40.0)
    a = ap.parse_args()
    for key in a.keys.split(","):
        ev, cid = key.split("/"); cid = int(cid)
        fig, axs = plt.subplots(3, 2, figsize=(14, 12))
        for j, arm in enumerate((a.a, a.b)):
            r, P, img = rec(a.det, arm, ev, cid)
            ax = axs[0, j]
            if r is None:
                ax.set_title(f"{arm}: not a candidate (no T_stm_michel row)")
                continue
            ch = P[1]
            if len(ch):
                o = np.argsort(ch[:, 4])
                ax.plot(ch[o, 4], ch[o, 3] / MIP[a.det], ".-", ms=3, lw=0.7, c="k", label="muon chain (role 1)")
            ax.axhline(1.0, c="grey", lw=0.5); ax.set_xlim(-2, 60); ax.set_ylim(0, 6)
            ax.set_xlabel("residual range rr (cm)"); ax.set_ylabel("dQ/dx / MIP")
            ax.set_title(f"{arm}: is_stm {int(r['is_stm'])} [{bits(r['reject_bits'])}] michel {int(r['michel_found'])} conn {int(r['michel_conn_type'])} "
                         f"KE {r['michel_ke_best']:.1f} MeV len {r['michel_len']:.1f} cm\nmuon_len {r['muon_len']:.0f} cm, plateau {r['plateau_med'] / MIP[a.det]:.2f} MIP, "
                         f"ks_mu {r['ks_mu']:.3f} ks_flat {r['ks_flat']:.3f}", fontsize=8)
            ax.legend(fontsize=7)
            s = np.array([r["stop_x"], r["stop_y"], r["stop_z"]])
            for ax, (i, k, nm) in zip(axs[1:, j], ((0, 1, "y"), (0, 2, "z"))):
                if img is not None and len(img[0]):
                    I = img[0]; near = np.linalg.norm(I - s, axis=1) < a.win
                    ax.scatter(I[near, i], I[near, k], s=2, c="lightgrey", label="image")
                if len(ch):
                    ax.plot(ch[:, i], ch[:, k], ".", ms=2, c="k", label="chain")
                if len(P[2]):
                    ax.plot(P[2][:, i], P[2][:, k], "s", ms=3, c="tab:orange", label="delta (role 2)")
                if len(P[3]):
                    ax.plot(P[3][:, i], P[3][:, k], "o", ms=4, c="tab:red", label="Michel (role 3)")
                ax.plot([s[i]], [s[k]], "*", ms=12, c="tab:blue", label="stop")
                ax.plot([r["tagger_stop_x"], r["tagger_stop_y"], r["tagger_stop_z"]][i:i + 1], [r["tagger_stop_x"], r["tagger_stop_y"], r["tagger_stop_z"]][k:k + 1], "x", ms=8, c="tab:green", label="tagger stop")
                ax.set_xlim(s[i] - a.win, s[i] + a.win); ax.set_ylim(s[k] - a.win, s[k] + a.win)
                ax.set_xlabel("x (cm)"); ax.set_ylabel(f"{nm} (cm)"); ax.legend(fontsize=6, loc="best")
        fig.suptitle(f"{a.det} {key}: {a.a} (left) vs {a.b} (right)")
        fig.tight_layout()
        fig.savefig(f"{a.out}_{ev}_{cid}.png", dpi=100)
        plt.close(fig)
        print(f"wrote {a.out}_{ev}_{cid}.png")


if __name__ == "__main__":
    main()
