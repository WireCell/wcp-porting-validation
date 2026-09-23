#!/usr/bin/env python3
"""doc pdvd/117 round 5 -- the Michel-domain test: does the data's muon-removal geometry, applied to the model's own
simulated electrons, compress their drift slope?

    CUDA_VISIBLE_DEVICES=1 python3 d117_dom.py > ../../scan/d117/dom/dom.txt

Inputs: S1's sim set (the 3,399 test-split electrons neighbouring an in-range data Michel, the same SEED / N_DRAW
matched draws J and data labels xi -- the selection block is d117_s1.main()'s, copied) and each data Michel's masks in
crop coordinates (d117_dom_masks.py, gate 80/80).  Arms are applied to the SIM crops; every sim readout regresses the
predictions on the data labels xi, exactly as S1 does, so the draws and the x-axis are shared by every arm.

Per-Michel arms are PAIRED: a draw that stands in for data Michel i gets Michel i's mask or removed fraction, so any
drift correlation of the removal is carried into the sim.
  mask_lit      sim * ~muon_i: Michel i's muon mask (dilated +-1 ch / +-4 ticks, as in the data) at the same crop
                coordinates -- both crops are centred on their charge centroid; orientation is random w.r.t. the electron;
  trunc_<f>_<e> charge removed from one end of the charge-weighted principal axis (mm: 5.1 per channel, 0.5 us x
                1.60563 mm/us per tick) until the fraction f is gone; f = fl (Michel i's frac_lost, a lower bound on the
                Michel's own loss) or fo (frac_overlap, an upper bound: it counts muon charge too) or a fixed dose;
                e = a / b, the two ends.  The stored crops carry no start point (the meta holds the centroid and the
                angles; the start would need the truth depos re-rasterised), so the two ends are a bracket;
  zbkg+...      S1's zbkg first (exact zeros outside the component that holds the peak), then the removal.
Data side: the in-range data slope split at the median of frac_lost / frac_overlap (mu_Z of scores.tsv, no re-scoring).
"""
import csv, sys
import numpy as np
from scipy import stats
import d98_predict as PR
import d117_s1 as S1

MM_CH, MM_TK = 5.1, 0.5 * 1.60563
SCAN = f"{PR.SCAN}/../d117"


def axis_order(im):
    """nonzero pixels ordered along the charge-weighted principal axis; (flat indices, charges) in that order"""
    c, t = np.nonzero(im > 0)
    q = im[c, t].astype(np.float64)
    P = np.c_[c * MM_CH, t * MM_TK]
    m = (P * q[:, None]).sum(0) / q.sum()
    D = P - m
    cov = (D * q[:, None]).T @ D / q.sum()
    u = np.linalg.eigh(cov)[1][:, -1]
    o = np.argsort(D @ u, kind="stable")
    return c[o], t[o], q[o]


def trunc(im, f, end):
    """remove charge from end 'a' (low projection) or 'b' until the removed fraction would exceed f"""
    out = im.copy()
    if f <= 0 or not (im > 0).any():
        return out, 0.0
    c, t, q = axis_order(im)
    if end == "b":
        c, t, q = c[::-1], t[::-1], q[::-1]
    cs = np.cumsum(q)
    n = int(np.searchsorted(cs, f * cs[-1], side="right"))
    out[c[:n], t[:n]] = 0
    return out, float(cs[n - 1] / cs[-1]) if n else 0.0


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
    fo = np.array([float(rows[k]["frac_overlap"]) for k in ikeys])

    # ---- S1's sim selection and draws, verbatim (d117_s1.main)
    sp = [r for r in csv.DictReader(open(PR.RUN + "/predictions_test.csv")) if r["species"] == "e"]
    ys = np.array([float(r["y_true_crop_cm"]) for r in sp]); Es = np.array([float(r["E_MeV"]) for r in sp])
    kec = np.clip(ked[inr], 5, 50)
    nb = [np.nonzero((np.abs(ys - xi) < S1.SIM_DX) & (np.abs(Es - ki) < S1.SIM_DE))[0] for xi, ki in zip(xd[inr], kec)]
    uni = np.unique(np.concatenate(nb))
    pos = {j: i for i, j in enumerate(uni)}
    nbl = [np.array([pos[j] for j in a_]) for a_ in nb]
    store = PR.sparse_store.SparseCrops(PR.DATASET)
    ZS = np.stack([store.row(int(sp[j]["row"]), view=2).astype(np.float32) for j in uni])
    mus_pub = np.array([float(sp[j]["mu_cm"]) for j in uni])
    xi = xd[inr]; ni = len(xi)
    J = np.stack([np.array([rng.choice(b) for b in nbl]) for _ in range(S1.N_DRAW)])
    # ---- end of the copied block
    npair = sum(len(b) for b in nbl)
    print(f"# doc pdvd/117 round 5, the Michel-domain test (d117_dom.py); model epoch {ep}; in-range data Michels {ni}; "
          f"sim crops {len(uni)}, (Michel, neighbour) pairs {npair}; draws {S1.N_DRAW}")
    print(f"# data removal: frac_lost q10/50/90 {np.quantile(fl, [.1, .5, .9]).round(3).tolist()}, frac_overlap "
          f"{np.quantile(fo, [.1, .5, .9]).round(3).tolist()}; spearman(drift, fl) {stats.spearmanr(xi, fl)[0]:+.2f}, "
          f"(drift, fo) {stats.spearmanr(xi, fo)[0]:+.2f}")

    def predict(imgs):
        return np.concatenate([PR.score_val.predict(net, np.ascontiguousarray(imgs[i:i + 64]), "cuda")[0]
                               for i in range(0, len(imgs), 64)])

    def ols_rows(X, Y):
        xm, ym = X.mean(1, keepdims=True), Y.mean(1, keepdims=True)
        sl = ((X - xm) * (Y - ym)).sum(1) / ((X - xm) ** 2).sum(1)
        return sl, ym[:, 0] - sl * xm[:, 0]

    XS = np.broadcast_to(xi, J.shape)
    q3 = lambda v: np.quantile(v, [0.16, 0.5, 0.84])
    cols = np.arange(ni)[None, :]

    # ---- arms.  A per-crop arm gives one prediction per sim crop; a paired arm one per (Michel i, neighbour) pair.
    def per_crop(fn):
        out, fr = np.empty(len(ZS)), np.empty(len(ZS))
        for s in range(0, len(ZS), 256):
            res = [fn(im) for im in ZS[s:s + 256]]
            out[s:s + 256] = predict(np.stack([r[0] for r in res])); fr[s:s + 256] = [r[1] for r in res]
        return out[J], fr[J]

    def paired(fn):
        """fn(im, i) -> (image, removed fraction); predictions for every (i, neighbour) pair, then indexed by J"""
        P = np.zeros((ni, len(uni))); F = np.zeros((ni, len(uni)))
        pairs = [(i, j) for i in range(ni) for j in nbl[i]]
        for s in range(0, len(pairs), 256):
            ch = pairs[s:s + 256]
            res = [fn(ZS[j], i) for i, j in ch]
            mu = predict(np.stack([r[0] for r in res]))
            for (i, j), m_, r in zip(ch, mu, res):
                P[i, j] = m_; F[i, j] = r[1]
        return P[cols, J], F[cols, J]

    def frac_of(im0, im1):
        return 1.0 - float(im1.sum()) / float(im0.sum())

    zb = lambda im: S1.zbkg(im[None])[0]
    arms = {"none": ("crop", lambda im: (im, 0.0)),
            "zbkg": ("crop", lambda im: (lambda o: (o, frac_of(im, o)))(zb(im))),
            "mask_lit": ("pair", lambda im, i: (lambda o: (o, frac_of(im, o)))(im * ~MU[i]))}
    for f in (0.2, 0.4, 0.6):
        for e in "ab":
            arms[f"trunc{f}_{e}"] = ("crop", (lambda f, e: lambda im: trunc(im, f, e))(f, e))
    for nm, F_ in (("fl", fl), ("fo", fo)):
        for e in "ab":
            arms[f"trunc_{nm}_{e}"] = ("pair", (lambda F_, e: lambda im, i: trunc(im, F_[i], e))(F_, e))
    for e in "ab":
        arms[f"zbkg+trunc_fl_{e}"] = ("pair", (lambda e: lambda im, i: (lambda o: (o, frac_of(im, o)))(
            trunc(zb(im), fl[i], e)[0]))(e))
    arms["zbkg+mask_lit"] = ("pair", lambda im, i: (lambda o: (o, frac_of(im, o)))(zb(im) * ~MU[i]))

    print("\n== A. sim, paired on the same draws.  d = arm - none, median [16, 84]; removed = charge fraction the arm "
          "took from the sim crop (median over draws of the per-draw mean)")
    print(f"{'arm':20s} {'slope':>7s} {'d slope [16,84]':>24s} {'d icpt [16,84]':>22s} {'d mu':>7s} {'rho':>6s} {'removed':>8s}")
    base = None
    for name, (kind, fn) in arms.items():
        Y, Fr = per_crop(fn) if kind == "crop" else paired(fn)
        sl, ic = ols_rows(XS, Y)
        rho = np.median([stats.spearmanr(xi, Y[d])[0] for d in range(0, S1.N_DRAW, 10)])
        if name == "none":
            base = (Y, sl, ic)
            print(f"# closure: max|mu(none) - predictions_test.csv| over the drawn crops "
                  f"{np.abs(Y - mus_pub[J]).max():.4f} cm; sim slope median {np.median(sl):.3f} (S1: 0.912)")
        dS, dI = q3(sl - base[1]), q3(ic - base[2])
        print(f"{name:20s} {np.median(sl):7.3f} {dS[1]:+8.3f} [{dS[0]:+.3f},{dS[2]:+.3f}] {dI[1]:+8.1f} [{dI[0]:+.0f},{dI[2]:+.0f}] "
              f"{np.median(Y - base[0]):+7.1f} {rho:+6.2f} {np.median(Fr.mean(1)):8.3f}")
        sys.stdout.flush()

    print("\n== B. data (no re-scoring): the in-range slope of mu_Z split at the median removed fraction; "
          "1000 bootstraps within each half, d = high - low")
    mu = np.array([float(rows[k]["mu_Z"]) for k in ikeys])
    for nm, F_ in (("frac_lost", fl), ("frac_overlap", fo)):
        lo = F_ <= np.median(F_); hi = ~lo
        s_lo, s_hi = np.polyfit(xi[lo], mu[lo], 1)[0], np.polyfit(xi[hi], mu[hi], 1)[0]
        d = []
        for _ in range(1000):
            a_ = rng.choice(np.nonzero(lo)[0], lo.sum()); b_ = rng.choice(np.nonzero(hi)[0], hi.sum())
            d.append(np.polyfit(xi[b_], mu[b_], 1)[0] - np.polyfit(xi[a_], mu[a_], 1)[0])
        dq = q3(np.array(d))
        print(f"  {nm:12s} low half n {lo.sum():2d} (<= {np.median(F_):.3f}) slope {s_lo:.3f} | high half n {hi.sum():2d} "
              f"slope {s_hi:.3f} | d {dq[1]:+.3f} [{dq[0]:+.3f}, {dq[2]:+.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
