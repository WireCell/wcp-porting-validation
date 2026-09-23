#!/usr/bin/env python3
"""doc pdvd/117 study S1 -- in/out transforms on existing images: which image differences move the regressor's
intercept and which move its slope?

    CUDA_VISIBLE_DEVICES=1 python3 d117_s1.py --run r1regen > ../../scan/d117/r1regen/s1.txt

Two image sets, each re-scored under every transform with the published bf16 CUDA function (d98_predict path):
  data -- the PDVD Michel-only crops (variant Z) of d98_michel_crops.py --run ../d117/<run>, in-range tier B;
  sim  -- the model's own test-split electrons that neighbour a data Michel (|d drift| < 15 cm, |d E| < 5 MeV,
          doc 98 sec 6c), read from the sparse store; readouts are the matched draws of d98_plots.sim_expect.
Transforms (sec 2.0 of doc 117: a drift-independent width should move the intercept only; a drift-growing one the slope):
  none         -- closure against scores.tsv / predictions_test.csv;
  floor1e-3    -- 1e-3 electrons added to every pixel: the model's sensitivity to a non-zero floor;
  filt3/10     -- the ratio of the training to the data Wire_col filter, H3/H10, along the channel axis (a blur:
                  emulates re-SP with Wire_col 3.0 on data, study S4, without the ROI stage; +-3-channel real-space kernel);
  gw<s>        -- a constant Gaussian blur along channels, s in wires (0.21 wire = the 1.05 mm data-only W excess);
  gt<s>        -- a constant Gaussian blur along ticks, s in ticks;
  dT<D>        -- a drift-proportional channel blur sigma^2 = 2 D t (extra transverse diffusion D cm^2/s);
  dL<D>        -- a drift-proportional tick blur sigma_t^2 = 2 D t / v^2 (extra longitudinal diffusion);
  tau<ms>      -- attenuation exp(-t/tau) applied (sim) / undone (data: exp(+t/tau));
  zbkg         -- sim only: exact zeros outside the dilated (+-2 ch, +-10 ticks) connected component that holds the
                  peak pixel -- the data crop's "Michel mask, zeros elsewhere" background.
Part B (model-free): per-crop image features against drift, data vs the matched sim draws.
Nothing is written except stdout; no record of doc 98 is read for writing.
"""
import argparse, csv, json, sys
import numpy as np
from scipy import ndimage, stats
import d98_plots as P
import d98_predict as PR

V_DATA, V_SIM = 1.48073, 1.60563          # mm/us: PDVD production .tlas, training sim
PITCH_MM, TICK_US = 5.100, 0.5
SEED = 20260923
N_DRAW = 1000
SIM_DX, SIM_DE = 15.0, 5.0


def t_sec(x_cm, v):
    return x_cm / (v * 0.1) * 1e-6                          # drift time, s


def hf(n, X):
    f = np.arange(n) / n * 2.0
    f = np.where(f > 1, f - 2, f)
    return np.exp(-0.5 * (np.abs(f) / (X / np.sqrt(np.pi))) ** 2)


def filt_kernel(n=480, half=3):
    """real-space kernel of H3/H10 (inverse DFT on n wires), truncated to +-half channels and renormalised.  Applied
    as a finite convolution: an FFT round trip leaves ~1e-3 e in every zero pixel, and the model reacts to that floor
    by hundreds of cm (transform `floor`), which is not the filter."""
    h = np.real(np.fft.ifft(hf(n, 3.0) / hf(n, 10.0)))
    k = np.concatenate([h[-half:], h[:half + 1]])
    return k / k.sum()


def filt_ratio(imgs, *_):
    return np.clip(ndimage.convolve1d(imgs, filt_kernel(), axis=1, mode="constant"), 0, None).astype(np.float32)


def blur(imgs, sig, axis):
    """per-image Gaussian blur along axis (1 = channels, 2 = ticks); sig array per image"""
    out = np.empty_like(imgs)
    for i, s in enumerate(sig):
        out[i] = ndimage.gaussian_filter1d(imgs[i], s, axis=axis - 1, mode="constant") if s > 1e-3 else imgs[i]
    return out


def grow(imgs, axis, k, eps):
    """support-only: pixels within k ticks (axis 2) or channels (axis 1) of the non-zero support, and zero, get eps;
    the charge itself is untouched"""
    st = np.zeros((2 * k + 1, 1), bool) if axis == 1 else np.zeros((1, 2 * k + 1), bool)
    st[:] = True
    out = imgs.copy()
    for i, im in enumerate(imgs):
        ring = ndimage.binary_dilation(im > 0, structure=st) & (im <= 0)
        out[i][ring] = eps
    return out


def zbkg(imgs, *_):
    out = np.zeros_like(imgs)
    st = np.ones((5, 21), bool)
    for i, im in enumerate(imgs):
        dil = ndimage.binary_dilation(im > 0, structure=st)
        lab, _ = ndimage.label(dil, structure=np.ones((3, 3), bool))
        pk = np.unravel_index(np.argmax(im), im.shape)
        keep = lab == lab[pk]
        out[i] = np.where(keep, im, 0)
    return out


def transforms():
    T = {"none": lambda im, x, v, s: im, "filt3/10": lambda im, x, v, s: filt_ratio(im)}
    for s in (0.21, 0.5):
        T[f"gw{s}"] = (lambda s: lambda im, x, v, sg: blur(im, np.full(len(im), s), 1))(s)
    for s in (1.0, 2.0):
        T[f"gt{s}"] = (lambda s: lambda im, x, v, sg: blur(im, np.full(len(im), s), 2))(s)
    for D in (2.0, 4.0, 8.0):
        T[f"dT{D}"] = (lambda D: lambda im, x, v, sg: blur(
            im, np.sqrt(2 * D * t_sec(x, v)) * 10.0 / PITCH_MM, 1))(D)          # cm -> mm -> wires
        T[f"dL{D}"] = (lambda D: lambda im, x, v, sg: blur(
            im, np.sqrt(2 * D * t_sec(x, v)) / (v * 0.1) / TICK_US, 2))(D)      # cm -> us -> ticks
    T["floor1e-3"] = lambda im, x, v, sg: (im + 1e-3).astype(np.float32)          # the FFT-artifact size, every pixel
    for nm in ("filt3/10", "gw0.5", "gt1.0", "gt2.0", "dL2.0", "dL4.0", "dL8.0"):
        T[nm + "|sp"] = (lambda f: lambda im, x, v, sg: (f(im, x, v, sg) * (im > 0)).astype(np.float32))(T[nm])
    for ax, k, eps in ((2, 1, 1e-3), (2, 3, 1e-3), (1, 1, 1e-3), (2, 2, 50.0)):
        T[f"sup_{'t' if ax == 2 else 'c'}{k}_{eps:g}"] = (lambda ax, k, eps: lambda im, x, v, sg: grow(im, ax, k, eps))(ax, k, eps)
    T["tau20"] = lambda im, x, v, sg: (im * np.exp(sg * t_sec(x, v) / 0.020)[:, None, None]).astype(np.float32)
    T["zbkg"] = lambda im, x, v, sg: zbkg(im)
    return T


def fit(x, y):
    s, c = np.polyfit(x, y, 1)
    return s, c, stats.spearmanr(x, y)[0]


def features(imgs):
    """model-free per-crop features: total charge, peak, pixels > 0, charge-weighted tick rms per channel (median
    over channels holding >= 5 % of the charge), charge-weighted channel rms per tick (same rule)"""
    out = []
    tk, ch = np.arange(imgs.shape[2]), np.arange(imgs.shape[1])
    for im in imgs.astype(np.float64):
        q = im.sum(); qc = im.sum(1); qt = im.sum(0)
        sel = qc >= 0.05 * q
        trms = [np.sqrt(np.average((tk - np.average(tk, weights=im[c])) ** 2, weights=im[c])) for c in np.nonzero(sel)[0]]
        selt = qt >= 0.05 * q
        crms = [np.sqrt(np.average((ch - np.average(ch, weights=im[:, t])) ** 2, weights=im[:, t])) for t in np.nonzero(selt)[0]]
        out.append((q, im.max(), (im > 0).sum(), np.median(trms) if trms else np.nan, np.median(crms) if crms else np.nan,
                    p25_window_rms2(im, sel, 12, axis=1), p25_window_rms2(im.T, selt, 3, axis=1),
                    np.median((im[sel] > 0).sum(1)) if sel.any() else np.nan,
                    np.quantile((im[sel] > 0).sum(1), 0.25) if sel.any() else np.nan,
                    np.median((im.T[selt] > 0).sum(1)) if selt.any() else np.nan))
    return np.array(out)


def w2_d98(im, win=12):
    """doc 98 d98_plots.py:289-310 verbatim: channels with q > 5 % of the largest channel's q; per channel the tick
    RMS^2 within +-win of the channel's peak; 25th percentile; None with fewer than 3 channels"""
    q = im.sum(axis=1); cols = np.nonzero(q > 0.05 * q.max())[0]
    v = []
    for c in cols:
        pk = int(np.argmax(im[c])); lo, hi = max(0, pk - win), min(im.shape[1], pk + win + 1)
        p = im[c, lo:hi]; t = np.arange(lo, hi); mu_ = (p * t).sum() / p.sum()
        v.append(((p * (t - mu_) ** 2).sum() / p.sum()))
    return float(np.quantile(v, 0.25)) if len(v) >= 3 else np.nan


def p25_window_rms2(im, sel, hw, axis):
    """doc 98 sec 6c's width estimator: per selected row, the charge-weighted rms^2 within +-hw of the row's peak;
    the 25th percentile over rows (the narrowest, least topology-broadened rows)"""
    v = []
    for r in np.nonzero(sel)[0]:
        w = im[r]; p = int(np.argmax(w)); lo, hi = max(p - hw, 0), min(p + hw + 1, len(w))
        ww = w[lo:hi]; x = np.arange(lo, hi)
        if ww.sum() > 0:
            m = np.average(x, weights=ww); v.append(np.average((x - m) ** 2, weights=ww))
    return np.quantile(v, 0.25) if v else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r1regen")
    ap.add_argument("--only-b", action="store_true", help="skip part A (the model re-scoring)")
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "the published function is the bf16 CUDA one"
    net, ep = PR.score_val.load_model(PR.RUN, dev)
    rng = np.random.default_rng(SEED)
    scan = f"{PR.SCAN}/../d117/{a.run}"
    rows = {r["key"]: r for r in csv.DictReader((l for l in open(scan + "/scores.tsv") if not l.startswith("#")),
                                                delimiter="\t") if r["det"] == "pdvd"}
    z = np.load(f"/home/xqian/tmp/d117/{a.run}/crops_pdvd.npz")
    keys = [str(k) for k in z["keys"]]
    ZD = z["Z"]
    xd = np.array([float(rows[k]["drift_tick"]) for k in keys])
    ked = np.array([float(rows[k]["ke_best"]) for k in keys])
    inr = np.array([rows[k]["inrange"] in ("1", "True") for k in keys])
    mu0 = np.array([float(rows[k]["mu_Z"]) for k in keys])
    print(f"# doc pdvd/117 S1 (d117_s1.py --run {a.run}); model epoch {ep}; data crops {len(keys)}, in-range {inr.sum()}")

    # sim neighbours of the in-range data Michels
    sp = [r for r in csv.DictReader(open(PR.RUN + "/predictions_test.csv")) if r["species"] == "e"]
    ys = np.array([float(r["y_true_crop_cm"]) for r in sp]); Es = np.array([float(r["E_MeV"]) for r in sp])
    kec = np.clip(ked[inr], 5, 50)
    nb = [np.nonzero((np.abs(ys - xi) < SIM_DX) & (np.abs(Es - ki) < SIM_DE))[0] for xi, ki in zip(xd[inr], kec)]
    uni = np.unique(np.concatenate(nb))
    pos = {j: i for i, j in enumerate(uni)}
    nbl = [np.array([pos[j] for j in a_]) for a_ in nb]
    store = PR.sparse_store.SparseCrops(PR.DATASET)
    ZS = np.stack([store.row(int(sp[j]["row"]), view=2).astype(np.float32) for j in uni])
    xs = ys[uni]; mus_pub = np.array([float(sp[j]["mu_cm"]) for j in uni])
    print(f"# sim: {len(sp)} test electrons, {len(uni)} neighbour crops (per Michel min {min(map(len, nb))}, "
          f"median {int(np.median([len(b) for b in nb]))}); matched draws {N_DRAW}")

    xi = xd[inr]; ni = len(xi)
    J = np.stack([np.array([rng.choice(b) for b in nbl]) for _ in range(N_DRAW)])      # paired sim draws, fixed once
    B = rng.integers(0, ni, (N_DRAW, ni))                                               # paired data bootstrap

    def ols_rows(X, Y):
        xm, ym = X.mean(1, keepdims=True), Y.mean(1, keepdims=True)
        sl = ((X - xm) * (Y - ym)).sum(1) / ((X - xm) ** 2).sum(1)
        return sl, ym[:, 0] - sl * xm[:, 0]

    q3 = lambda v: np.quantile(v, [0.16, 0.5, 0.84])
    XS = np.broadcast_to(xi, J.shape)
    XB = xi[B]
    if not a.only_b:
        print("\n== A. paired readouts per transform.  sim: the same N_DRAW matched draws for every transform; data: the "
              "same bootstrap resamples.  d = transform - none, median [16, 84].  * = data blur built from the drift label "
              "(its slope change is injected, not measured)")
        print(f"{'transform':16s} | {'data slope':>10s} {'d slope [16,84]':>22s} {'d mu':>6s} | "
              f"{'sim slope':>9s} {'d slope [16,84]':>22s} {'d icpt [16,84]':>20s} {'d mu':>6s}")
    for name, fn in ({} if a.only_b else transforms()).items():
        md = PR.score_val.predict(net, fn(ZD, xd, V_DATA, +1.0), dev)[0] if name != "zbkg" else None
        mS = np.concatenate([PR.score_val.predict(net, fn(ZS[i:i + 256], xs[i:i + 256], V_SIM, -1.0), dev)[0]
                             for i in range(0, len(ZS), 256)])
        slS, icS = ols_rows(XS, mS[J])
        if name == "none":
            print(f"# closure: data max|mu - scores.tsv| = {np.abs(md - mu0).max():.4f} cm; "
                  f"sim max|mu - predictions_test.csv| = {np.abs(mS - mus_pub).max():.4f} cm")
            md0, mS0, slS0, icS0 = md, mS, slS, icS
            slD0, _ = ols_rows(XB, md0[inr][B])
        dS, dI = q3(slS - slS0), q3(icS - icS0)
        if md is not None:
            s_, c_, r_ = fit(xi, md[inr]); slD, _ = ols_rows(XB, md[inr][B]); dD = q3(slD - slD0)
            dstr = f"{s_:10.3f} {dD[1]:+7.3f} [{dD[0]:+.3f},{dD[2]:+.3f}] {np.median(md[inr] - md0[inr]):+6.1f}"
        else:
            dstr = f"{'-':>10s} {'-':>22s} {'-':>6s}"
        star = "*" if name.startswith(("dL", "dT", "tau")) else " "
        print(f"{name:15s}{star} | {dstr} | {np.median(slS):9.3f} {dS[1]:+7.3f} [{dS[0]:+.3f},{dS[2]:+.3f}] "
              f"{dI[1]:+7.1f} [{dI[0]:+.0f},{dI[2]:+.0f}] {np.median(mS - mS0):+6.1f}")

    print("\n== C. doc 98's width estimator (d98_plots.py:289-310, w2 = p25 of per-channel tick RMS^2 within +-12 ticks, "
          "channels with q > 5 % of the largest) [ticks^2 per 100 cm, 68 % bootstrap]")
    allx = xd; ZW = z["Zw"]
    for lab, imgs, sel in (("data Z, all crops", ZD, np.ones(len(keys), bool)), ("data Z, in-range", ZD, inr),
                           ("data Zw, all crops", ZW, np.ones(len(keys), bool)), ("data Zw, in-range", ZW, inr)):
        w = np.array([w2_d98(im) for im in imgs[sel]]); x_ = allx[sel]; ok = np.isfinite(w)
        sl = np.polyfit(x_[ok], w[ok], 1)[0]
        bs = [np.polyfit(x_[ok][i_], w[ok][i_], 1)[0] for i_ in (rng.integers(0, ok.sum(), ok.sum()) for _ in range(1000))]
        print(f"  {lab:20s} n {ok.sum():3d}  slope {100 * sl:+.3f} [{100 * np.quantile(bs, .16):+.3f}, {100 * np.quantile(bs, .84):+.3f}]"
              f"  r {stats.pearsonr(x_[ok], w[ok])[0]:+.2f}  median w2 {np.median(w[ok]):.2f}")
    wS = np.array([w2_d98(im) for im in ZS])
    Ys = wS[J]; okS = np.isfinite(Ys).all(1)
    slw, _ = ols_rows(XS[okS], Ys[okS])
    print(f"  {'sim, matched draws':20s} n {ni:3d}  slope {100 * np.median(slw):+.3f} [{100 * np.quantile(slw, .16):+.3f}, "
          f"{100 * np.quantile(slw, .84):+.3f}]  median w2 {np.nanmedian(wS):.2f}   (naive 2 D_L/v^3: training "
          f"{100 * 2 * 4.0e-6 / 0.160563 ** 3 / 0.25:.3f}, PDVD data params {100 * 2 * 4.1307e-6 / 0.148073 ** 3 / 0.25:.3f})")

    print("\n== B. model-free image features against drift (OLS slope per 100 cm, and value at 200 cm)")
    FD, FS = features(ZD[inr]), features(ZS)
    names = ("Q_tot [ke]", "peak [e]", "n_pix", "tick rms/ch", "chan rms/tick", "tick rms2 p25", "chan rms2 p25",
             "nz ticks/ch med", "nz ticks/ch p25", "nz chans/tick med")
    scale = (1e-3, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    for k, (nm, sc) in enumerate(zip(names, scale)):
        fd = FD[:, k] * sc; ok = np.isfinite(fd)
        s, c = np.polyfit(xd[inr][ok], fd[ok], 1)
        xo, fo = xd[inr][ok], fd[ok]; bs = []
        for _ in range(1000):
            i_ = rng.integers(0, len(xo), len(xo)); bs.append(np.polyfit(xo[i_], fo[i_], 1)[0])
        blo, bhi = np.quantile(bs, [0.16, 0.84])
        S, V = [], []
        for _ in range(300):
            j = np.array([rng.choice(b) for b in nbl]); f_ = FS[j, k] * sc; o = np.isfinite(f_)
            ss, cc = np.polyfit(xd[inr][o], f_[o], 1); S.append(ss); V.append(ss * 200 + cc)
        print(f"  {nm:16s} data slope {100 * s:+9.3f} [{100 * blo:+.3f}, {100 * bhi:+.3f}] @200 {200 * s + c:9.2f} | sim slope {100 * np.median(S):+9.3f} "
              f"[{100 * np.quantile(S, .16):+.3f}, {100 * np.quantile(S, .84):+.3f}] @200 {np.median(V):9.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
