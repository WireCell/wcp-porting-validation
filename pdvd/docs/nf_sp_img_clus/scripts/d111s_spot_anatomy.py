#!/usr/bin/env python3
"""doc pdvd/111 round 2 -- the owner's spots at the Steiner level: image, retiled cloud, Steiner cloud, terminals,
Steiner edges, and the Dijkstra seed, next to the image-oracle route and two re-priced replays of the same walks.

Inputs: the dump arms (d111hst / d111vst: WCT_STM_PATH_DEBUG=1 WCT_STEINER_GRAPH_DUMP=1), their Bee image and
tracking-stm.root.  The frame is d111_spot_frame.frame() of the image around the click (as in doc 111 sec 1).
The persisted pass through the window is the round-2 record; its seed is replayed from the STMRP walks that built it (the
two crawl walks when adjust_rough_path crawled, else the rough walk), with the crawl point kept.

Outputs: <out>_<spot>.png (rows e2 / e3; columns: clouds, graph + seed, seed vs replays) and <out>.tsv (per spot: point
counts in the window and the share of each > 1 cm off the ridge; the seed's max offset and longest edge; the carrying
edge's source / base / 3-plane support at the seed's worst point; the replays' max offsets).

Usage: d111s_spot_anatomy.py --out figs/111s_spot
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
from d111_spot_frame import frame
from d111s_replay import weights
A = C.A

SPOTS = [
    ("h1", "pdhd", "029107_16", 108, (75.3, 438.3, 450.1), 12.0, "d111hst"),
    ("h2", "pdhd", "029107_16", 106, (182.0, 537.4, 401.8), 24.0, "d111hst"),
    ("v1", "pdvd", "039349_20", 80, (313.7, 95.9, 124.9), 12.0, "d111vst"),
    ("v2", "pdvd", "039349_20", 80, (94.5, 30.7, 165.0), 12.0, "d111vst"),
    ("v3", "pdvd", "039349_20", 25, (-270.2, -243.7, 36.0), 12.0, "d111vst"),
]
REPLAYS = [("oracle", None, "m--"), ("G_3pl", 2, "C4:"), ("G_2pl", 2, "C8-.")]
SRC_COL = {0: "tab:olive", 1: "tab:red", 2: "tab:purple"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lines = ["spot\tcluster\tpass\tn_img\tn_ret\tn_steiner\tn_term\timg_off1\tret_off1\tsteiner_off1\tterm_off1\t"
             "seed_max\tseed_longest_edge\tworst_edge_src\tworst_edge_base\tworst_edge_len\tworst_edge_bad3\t"
             "oracle_max\tG_3pl2_max\tG_2pl2_max\twalks"]
    cache = {}
    for spot, det, ev, cl, click, R, arm in SPOTS:
        click = np.array(click)
        key = (det, ev, arm)
        if key not in cache:
            d = f"{C.IMG}/{det}/work/{ev}_{arm}"
            tr = C.parse_trace(f"/home/xqian/tmp/d111/arm_{arm}/evt_{ev}.log.gz")
            t = uproot.open(f"{d}/tracking-stm.root")["T_rec_charge"].arrays(["x", "y", "z", "cluster_id", "pass"], library="np")
            img = A.bee_layer(f"{d}/mabc-pr.zip", "clustering-global")
            cache[key] = (tr, t, img)
        tr, t, img = cache[key]
        mi = img[1] == cl
        P0, w0 = img[0][mi], np.clip(img[2][mi], 1, None)
        Rg = A.Ridge(img[0][mi], img[2][mi])
        blocks = [b for b in tr["blocks"] if b["cl"] == cl and "seed" in b["st"]]
        rec = None
        for ps in np.unique(t["pass"][t["cluster_id"] == cl]):
            m = (t["cluster_id"] == cl) & (t["pass"] == ps)
            F = np.c_[t["x"][m], t["y"][m], t["z"][m]]
            if np.min(np.linalg.norm(F - click, axis=1)) > R:
                continue
            bl = [b for b in blocks if b["rnd"] == "r2" and "final" in b["st"] and len(b["st"]["final"]) == len(F)
                  and np.max(np.abs(b["st"]["final"] - F)) < 0.01]
            if bl:
                rec = (ps, F, bl[-1]); break
        if rec is None:
            print(f"{spot}: no record through the window"); continue
        ps, F, b2 = rec
        sb = C.graph_for(tr, cl, b2["order"])
        r1 = [b for b in blocks if b["rnd"] == "r1" and b["dir"] == b2["dir"] and b["order"] < b2["order"]][-1]
        crawl = [r for r in C.walks_for(tr, cl, b2["order"], after=r1["order"]) if r[2] in ("crawl1", "crawl2")]
        rough = [r for r in C.walks_for(tr, cl, r1["order"]) if r[2] == "rough"]
        walks = crawl[-2:] if len(crawl) >= 2 else rough[-1:]
        seed = b2["st"]["seed"]
        # oracle weights
        from d111s_graph_census import edge_offfrac
        w_or = sb.E_w * (1.0 + 100.0 * edge_offfrac(sb, Rg))
        rep = {}
        for name, k, _ in REPLAYS:
            w = w_or if name == "oracle" else weights(sb, name, k)
            Ps = []
            for wk in walks:
                p, _ = sb.walk(wk[3], wk[4], w)
                if p is not None:
                    Ps.append(sb.V[p])
            rep[name] = np.concatenate(Ps) if Ps else np.zeros((0, 3))
        near = np.linalg.norm(P0 - click, axis=1) < R
        c, e1, e2, e3 = frame(P0[near], w0[near])
        proj = lambda X: ((X - c) @ e1, np.c_[(X - c) @ e2, (X - c) @ e3])
        inwin = lambda X, f=1.0: np.linalg.norm(X - click, axis=1) < f * R
        ret = sb.R[inwin(sb.R)]; Vw = inwin(sb.V); V = sb.V[Vw]; T = sb.V[Vw & sb.V_term]
        fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey="row")
        for r in range(2):
            s_i, T_i = proj(P0[near])
            ax = axs[r, 0]
            s_r, T_r = proj(ret); ax.scatter(s_r, T_r[:, r], s=2, c="tab:cyan", alpha=0.35, label="retiled cloud")
            ax.scatter(s_i, T_i[:, r], s=5, c="0.55", label="image")
            s_v, T_v = proj(V); ax.scatter(s_v, T_v[:, r], s=6, c="tab:green", label="Steiner vertex")
            s_t, T_t = proj(T); ax.scatter(s_t, T_t[:, r], s=18, marker="^", c="darkgreen", label="terminal")
            ax = axs[r, 1]
            ax.scatter(s_i, T_i[:, r], s=5, c="0.8")
            ew = inwin(sb.V[sb.E_s]) & inwin(sb.V[sb.E_t])
            for kk in np.where(ew)[0]:
                XY = np.vstack([sb.V[sb.E_s[kk]], sb.V[sb.E_t[kk]]])
                s_e, T_e = proj(XY)
                ax.plot(s_e, T_e[:, r], "-", color=SRC_COL[sb.E_src[kk]], lw=0.5, alpha=0.6)
            ks = inwin(seed, 1.3)
            s_s, T_s = proj(seed[ks]); ax.plot(s_s, T_s[:, r], "-o", color="tab:orange", ms=3, lw=1.6, label="seed")
            ax = axs[r, 2]
            ax.scatter(s_i, T_i[:, r], s=5, c="0.8")
            ax.plot(s_s, T_s[:, r], "-o", color="tab:orange", ms=3, lw=1.6, label="seed (production)")
            for name, k, sty in REPLAYS:
                X = rep[name]
                if len(X):
                    kk = inwin(X, 1.3); s_x, T_x = proj(X[kk])
                    ax.plot(s_x, T_x[:, r], sty, lw=1.3, label=name if k is None else f"{name} k={k}")
            for j in range(3):
                axs[r, j].set_ylabel(["e2 (cm)", "e3 (cm, drift side)"][r])
                if r == 1:
                    axs[r, j].set_xlabel("s along image PCA (cm)")
        axs[0, 0].set_title(f"{spot} {det} {ev} cl{cl}: image / retiled / Steiner / terminals", fontsize=9)
        axs[0, 1].set_title("steiner_graph edges (olive path, red connect, purple same-blob) + seed", fontsize=9)
        axs[0, 2].set_title("seed vs image oracle and re-priced replays of the same walks", fontsize=9)
        for j in range(3):
            axs[0, j].legend(fontsize=7, loc="best")
        fig.tight_layout(); fig.savefig(f"{a.out}_{spot}.png", dpi=110); plt.close(fig)
        # table
        off1 = lambda X: 100 * np.mean(Rg.offset(X) > 1) if len(X) else np.nan
        Xs, _ = C.polyline_samples(seed[inwin(seed)], 0.3)
        so = Rg.offset(Xs) if len(Xs) else np.array([np.nan])
        j2 = C.join_vertices(sb, seed)
        worst = int(np.nanargmax(so)) if len(Xs) else 0
        Q = Xs[worst] if len(Xs) else seed[0]
        kk_seg = int(np.argmin(np.linalg.norm(seed - Q, axis=1)))
        best = None
        for kk in range(max(0, kk_seg - 1), min(len(seed) - 1, kk_seg + 1)):
            A_, B_ = seed[kk], seed[kk + 1]; AB = B_ - A_; L2 = AB @ AB
            tt = np.clip(((Q - A_) @ AB) / L2, 0, 1) if L2 > 0 else 0.0
            dq = np.linalg.norm(A_ + tt * AB - Q)
            if best is None or dq < best[0]:
                best = (dq, kk)
        va, vb = int(j2[best[1]]), int(j2[best[1] + 1])
        e = sb.edge_index.get((min(va, vb), max(va, vb)))
        src = C.SRC_NAME[sb.E_src[e]] if e is not None else "-"
        base = C.BASE_NAME[sb.E_base[e]] if e is not None else "-"
        elen = sb.E_len[e] if e is not None else np.nan
        bad3 = 1 - sb.S02[e, 1] / max(sb.S02[e, 0], 1) if e is not None and sb.S02[e, 0] else np.nan
        seg = np.linalg.norm(np.diff(seed[inwin(seed, 1.3)], axis=0), axis=1)
        def repmax(X):
            X = X[inwin(X)] if len(X) else X
            Y, _ = C.polyline_samples(X, 0.3) if len(X) >= 2 else (np.zeros((0, 3)), None)
            return np.nanmax(Rg.offset(Y)) if len(Y) else np.nan
        lines.append(f"{spot}\t{cl}\t{ps}\t{int(near.sum())}\t{len(ret)}\t{len(V)}\t{len(T)}\t{off1(P0[near]):.1f}\t{off1(ret):.1f}\t"
                     f"{off1(V):.1f}\t{off1(T):.1f}\t{np.nanmax(so):.2f}\t{(seg.max() if len(seg) else np.nan):.2f}\t{src}\t{base}\t"
                     f"{elen:.2f}\t{bad3:.2f}\t{repmax(rep['oracle']):.2f}\t{repmax(rep['G_3pl']):.2f}\t{repmax(rep['G_2pl']):.2f}\t"
                     f"{'+'.join(w[2] for w in walks)}")
    open(f"{a.out}.tsv", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
