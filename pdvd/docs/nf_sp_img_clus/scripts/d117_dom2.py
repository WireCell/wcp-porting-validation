#!/usr/bin/env python3
"""doc pdvd/117 round 6 -- three checks on round 5's Michel-domain result.

    CUDA_VISIBLE_DEVICES=1 python3 d117_dom2.py > ../../scan/d117/dom2/dom2.txt

Setup = d117_dom.py's (S1's sim set and the same SEED / N_DRAW matched draws J; the data Michels' masks from
d117_dom_masks.py).  "base_<e>" is round 5's full emulation zbkg + trunc at Michel i's frac_lost from end e.
  1. Does the DATA respond to truncation as the sim does?  Data Z crops (in range, production SP) lose a further 20 %
     of their charge, from the MUON-SIDE end (the principal-axis end nearer the muon-mask centroid, i.e. the
     Michel's start, known on data) and, separately, from the far end.  The sim twin: base_<e> loses a further 20 %
     from the same end e.  Data readout = paired bootstrap of the data slope; sim = the same draws.
  2. Joint emulation: base_<e> + a 20 ms lifetime (tau20, S1's arm) + the D_L excess the data measured over the
     training equivalent (S6: +0.3 top / +1.8 all, cm^2/s; S1's dL|sp blur built from the sim crop's own label),
     compared with the data after the wire filter (S4, 0.525).
  3. Energy: slopes restricted to the data Michels with ke_best >= 20 MeV and < 20 MeV (their draws only), sim none /
     base_a / base_b, against the data's own split (mu_Z of scores.tsv, bootstrap).
"""
import csv, sys
import numpy as np
from scipy import stats
import d98_predict as PR
import d117_s1 as S1
from d117_dom import axis_order, trunc

SCAN = f"{PR.SCAN}/../d117"
EXTRA = 0.2


def main():
    import torch
    assert torch.cuda.is_available(), "the published function is the bf16 CUDA one"
    net, ep = PR.score_val.load_model(PR.RUN, "cuda")
    rng = np.random.default_rng(S1.SEED)
    rows = {r["key"]: r for r in csv.DictReader((l for l in open(SCAN + "/latest/scores.tsv") if not l.startswith("#")),
                                                delimiter="\t") if r["det"] == "pdvd"}
    z = np.load("/home/xqian/tmp/d117/latest/crops_pdvd.npz")
    keys = [str(k) for k in z["keys"]]
    xd = np.array([float(rows[k]["drift_tick"]) for k in keys])
    ked = np.array([float(rows[k]["ke_best"]) for k in keys])
    inr = np.array([rows[k]["inrange"] in ("1", "True") for k in keys])
    mz = np.load("/home/xqian/tmp/d117/dom/masks_pdvd.npz")
    mk = {str(k): i for i, k in enumerate(mz["keys"])}
    ikeys = [k for k, s in zip(keys, inr) if s]
    MU = [mz["muon"][mk[k]] for k in ikeys]
    fl = np.array([float(rows[k]["frac_lost"]) for k in ikeys])

    # ---- d117_dom.py's (= S1's) sim selection and draws, verbatim
    sp = [r for r in csv.DictReader(open(PR.RUN + "/predictions_test.csv")) if r["species"] == "e"]
    ys = np.array([float(r["y_true_crop_cm"]) for r in sp]); Es = np.array([float(r["E_MeV"]) for r in sp])
    kec = np.clip(ked[inr], 5, 50)
    nb = [np.nonzero((np.abs(ys - xi) < S1.SIM_DX) & (np.abs(Es - ki) < S1.SIM_DE))[0] for xi, ki in zip(xd[inr], kec)]
    uni = np.unique(np.concatenate(nb))
    pos = {j: i for i, j in enumerate(uni)}
    nbl = [np.array([pos[j] for j in a_]) for a_ in nb]
    store = PR.sparse_store.SparseCrops(PR.DATASET)
    ZS = np.stack([store.row(int(sp[j]["row"]), view=2).astype(np.float32) for j in uni])
    xsl = ys[uni]
    xi = xd[inr]; ni = len(xi)
    J = np.stack([np.array([rng.choice(b) for b in nbl]) for _ in range(S1.N_DRAW)])
    # ---- end of the copied block
    B = rng.integers(0, ni, (S1.N_DRAW, ni))                     # data bootstrap, fixed once
    ZD = z["Z"][inr]
    mu0 = np.array([float(rows[k]["mu_Z"]) for k in ikeys])
    hiE = ked[inr] >= 20
    print(f"# doc pdvd/117 round 6 (d117_dom2.py); model epoch {ep}; in-range data Michels {ni} (>= 20 MeV: {hiE.sum()}); "
          f"sim crops {len(uni)}; draws {S1.N_DRAW}; extra truncation {EXTRA}")

    def predict(imgs):
        return np.concatenate([PR.score_val.predict(net, np.ascontiguousarray(imgs[i:i + 64]), "cuda")[0]
                               for i in range(0, len(imgs), 64)])

    def ols_rows(X, Y):
        xm, ym = X.mean(1, keepdims=True), Y.mean(1, keepdims=True)
        sl = ((X - xm) * (Y - ym)).sum(1) / ((X - xm) ** 2).sum(1)
        return sl, ym[:, 0] - sl * xm[:, 0]

    q3 = lambda v: np.quantile(v, [0.16, 0.5, 0.84])
    fmt = lambda d: f"{d[1]:+.3f} [{d[0]:+.3f}, {d[2]:+.3f}]"
    cols = np.arange(ni)[None, :]
    zb = lambda im: S1.zbkg(im[None])[0]

    def paired(fn):
        P = np.zeros((ni, len(uni))); F = np.zeros((ni, len(uni)))
        pairs = [(i, j) for i in range(ni) for j in nbl[i]]
        for s in range(0, len(pairs), 256):
            ch = pairs[s:s + 256]
            res = [fn(ZS[j], i, j) for i, j in ch]
            m_ = predict(np.stack([r[0] for r in res]))
            for (i, j), v, r in zip(ch, m_, res):
                P[i, j] = v; F[i, j] = r[1]
        return P[cols, J], F[cols, J]

    def base(im, i, e):
        z0 = zb(im)
        return trunc(z0, fl[i], e)[0]

    def dLsp(im, x_cm, D):
        s = np.sqrt(2 * D * S1.t_sec(x_cm, S1.V_SIM)) / (S1.V_SIM * 0.1) / S1.TICK_US
        return (S1.blur(im[None], np.array([s]), 2)[0] * (im > 0)).astype(np.float32)

    def tau(im, x_cm):
        return (im * np.exp(-S1.t_sec(x_cm, S1.V_SIM) / 0.020)).astype(np.float32)

    frac = lambda a, b: 1.0 - float(b.sum()) / float(a.sum())
    arms = {"none": lambda im, i, j: (im, 0.0)}
    for e in "ab":
        arms[f"base_{e}"] = (lambda e: lambda im, i, j: (lambda o: (o, frac(im, o)))(base(im, i, e)))(e)
        arms[f"base_{e}+extra"] = (lambda e: lambda im, i, j: (lambda o: (o, frac(im, o)))(
            trunc(base(im, i, e), EXTRA, e)[0]))(e)
        arms[f"base_{e}+tau20"] = (lambda e: lambda im, i, j: (lambda o: (o, frac(im, o)))(tau(base(im, i, e), xsl[j])))(e)
        for D in (0.3, 1.8):
            arms[f"base_{e}+tau20+dL{D}"] = (lambda e, D: lambda im, i, j: (lambda o: (o, frac(im, o)))(
                dLsp(tau(base(im, i, e), xsl[j]), xsl[j], D)))(e, D)

    XS = np.broadcast_to(xi, J.shape)
    res = {}
    print("\n== sim arms, paired on the same draws; d = arm - none (and, for +..., arm - base of the same end)")
    print(f"{'arm':24s} {'slope':>7s} {'d vs none':>26s} {'d vs base':>26s} {'removed':>8s}")
    for name, fn in arms.items():
        Y, Fr = paired(fn)
        sl, _ = ols_rows(XS, Y)
        res[name] = (Y, sl)
        dn = q3(sl - res["none"][1])
        bname = name.split("+")[0]
        db = fmt(q3(sl - res[bname][1])) if "+" in name else "-"
        print(f"{name:24s} {np.median(sl):7.3f} {fmt(dn):>26s} {db:>26s} {np.median(Fr.mean(1)):8.3f}")
        sys.stdout.flush()

    print("\n== 1. data: a further 20 % truncation of the in-range production Z crops.  muon side = the principal-axis end "
          "nearer the muon-mask centroid (the Michel's start); d = arm - none on the same bootstrap resamples")
    ends, dist = [], []
    for im, mu_m in zip(ZD, MU):
        c, t, q = axis_order(im)
        P = np.c_[c * 5.1, t * 0.5 * 1.48073]
        mc, mt = np.nonzero(mu_m)
        if not len(mc):
            ends.append(None); continue
        M = np.array([mc.mean() * 5.1, mt.mean() * 0.5 * 1.48073])
        lo_end, hi_end = P[0], P[-1]                                  # ordered by axis_order
        ends.append("a" if np.linalg.norm(M - lo_end) < np.linalg.norm(M - hi_end) else "b")
        dist.append(min(np.linalg.norm(M - lo_end), np.linalg.norm(M - hi_end)))
    ok = np.array([e is not None for e in ends])
    print(f"  muon-side end found for {ok.sum()}/{ni} crops (no muon pixels in the crop: {ni - ok.sum()}); "
          f"end a {sum(e == 'a' for e in ends)}, b {sum(e == 'b' for e in ends)}; median muon-centroid to end {np.median(dist):.0f} mm")
    other = {"a": "b", "b": "a"}
    XB = xi[B]
    sD0, _ = ols_rows(XB, mu0[B])
    for lab, pick in (("muon side (start)", lambda e: e), ("far side", lambda e: other[e])):
        imgs, fr = [], []
        for im, e in zip(ZD, ends):
            o, f_ = trunc(im, EXTRA, pick(e) if e else "a")
            imgs.append(o); fr.append(f_)
        md = predict(np.stack(imgs))
        sD, _ = ols_rows(XB, md[B])
        print(f"  data extra {EXTRA} {lab:18s}: slope {np.polyfit(xi, md, 1)[0]:.3f} (none {np.polyfit(xi, mu0, 1)[0]:.3f}) "
              f"d {fmt(q3(sD - sD0))}  d mu median {np.median(md - mu0):+.1f}  removed {np.median(fr):.3f}")
    print(f"  (closure: the data none predictions are scores.tsv's mu_Z; max|re-scored - stored| "
          f"{np.abs(predict(ZD) - mu0).max():.4f} cm)")

    print("\n== 3. energy split: slope over the draws of the data Michels in each class; data: bootstrap of mu_Z")
    for lab, sel in ((">= 20 MeV", hiE), ("< 20 MeV", ~hiE)):
        idx = np.nonzero(sel)[0]
        Xs = np.broadcast_to(xi[idx], (S1.N_DRAW, len(idx)))
        parts = []
        for nm in ("none", "base_a", "base_b"):
            s_, _ = ols_rows(Xs, res[nm][0][:, idx])
            parts.append(f"{nm} {np.median(s_):.3f}")
        s0, _ = ols_rows(Xs, res["none"][0][:, idx])
        dA = q3(ols_rows(Xs, res["base_a"][0][:, idx])[0] - s0); dB = q3(ols_rows(Xs, res["base_b"][0][:, idx])[0] - s0)
        Bs = rng.integers(0, len(idx), (S1.N_DRAW, len(idx)))
        sd, _ = ols_rows(xi[idx][Bs], mu0[idx][Bs])
        print(f"  {lab:9s} n {len(idx):2d} | sim {', '.join(parts)} | d base a {fmt(dA)}, b {fmt(dB)} | "
              f"data slope {np.polyfit(xi[idx], mu0[idx], 1)[0]:.3f} [{np.quantile(sd, .16):.3f}, {np.quantile(sd, .84):.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
