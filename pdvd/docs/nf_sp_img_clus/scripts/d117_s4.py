#!/usr/bin/env python3
"""doc pdvd/117 study S4 -- the real re-SP with the training Wire_col (3.0/sqrt(pi)) against production SP, on the
same Michels, masks and labels; and against S1's post-hoc emulation of it.

    CUDA_VISIBLE_DEVICES=1 python3 d117_s4.py > ../../scan/d117/w3/s4.txt

Inputs: scan/d117/latest/scores.tsv + /home/xqian/tmp/d117/latest/crops_pdvd.npz (production SP, d117sp) and
scan/d117/w3/scores.tsv + /home/xqian/tmp/d117/w3/crops_pdvd.npz (SP_ARM=d117w3sp, the --wire-col-x 3.0 arm).  Both
crop sets are cut from p98vonq's masks and labels, so only the SP differs.
Pre-registered (doc 117 sec 6.3 item 4): data slope +0.11 [0.07, 0.15], mu +19 cm from the emulation; the S1 frame
reading predicted "mostly intercept".
Readouts (in-range tier B, variant Z, and M): slope / intercept / rho per arm; the paired delta over 2000 shared
bootstrap resamples; the emulation (filt3/10 on the production crops, support kept) re-scored on the same crops; and
per crop the pixel L1 distance of the real re-SP and of the emulation from production, and between the two.
"""
import csv, sys
import numpy as np
from scipy import stats
import d98_predict as PR
import d117_s1 as S1

SEED = 20260924
NB = 2000


def load(run):
    rows = {r["key"]: r for r in csv.DictReader((l for l in open(f"{PR.SCAN}/../d117/{run}/scores.tsv")
                                                 if not l.startswith("#")), delimiter="\t") if r["det"] == "pdvd"}
    z = np.load(f"/home/xqian/tmp/d117/{run}/crops_pdvd.npz")
    return rows, z, [str(k) for k in z["keys"]]


def main():
    import torch
    assert torch.cuda.is_available(), "the published function is the bf16 CUDA one"
    net, ep = PR.score_val.load_model(PR.RUN, "cuda")
    rng = np.random.default_rng(SEED)
    rA, zA, kA = load("latest")
    rB, zB, kB = load("w3")
    keys = [k for k in kA if k in set(kB) and rA[k]["inrange"] in ("1", "True")]
    iA = {k: i for i, k in enumerate(kA)}; iB = {k: i for i, k in enumerate(kB)}
    x = np.array([float(rA[k]["drift_tick"]) for k in keys])
    dlab = max(abs(float(rA[k]["drift_tick"]) - float(rB[k]["drift_tick"])) for k in keys)
    top = np.array([int(rA[k]["apa"]) >= 4 for k in keys])
    print(f"# doc pdvd/117 S4 (d117_s4.py); model epoch {ep}; in-range Michels on both arms {len(keys)} "
          f"(latest {len(kA)} crops, w3 {len(kB)}); max |d drift label| between arms {dlab:.3f} cm")
    ZA = np.stack([zA["Z"][iA[k]] for k in keys]); ZB = np.stack([zB["Z"][iB[k]] for k in keys])
    mu = {"prod": np.array([float(rA[k]["mu_Z"]) for k in keys]), "w3": np.array([float(rB[k]["mu_Z"]) for k in keys])}
    mu["emul"] = PR.score_val.predict(net, (S1.filt_ratio(ZA) * (ZA > 0)).astype(np.float32), "cuda")[0]
    muM = {"prod": np.array([float(rA[k]["mu_M"]) for k in keys]), "w3": np.array([float(rB[k]["mu_M"]) for k in keys])}
    B = rng.integers(0, len(x), (NB, len(x)))
    XB = x[B]

    def ols(X, Y):
        xm, ym = X.mean(1, keepdims=True), Y.mean(1, keepdims=True)
        sl = ((X - xm) * (Y - ym)).sum(1) / ((X - xm) ** 2).sum(1)
        return sl, ym[:, 0] - sl * xm[:, 0]
    q = lambda v: np.quantile(v, [0.16, 0.5, 0.84])

    print("\n== A. readouts (in-range tier B).  d = arm - production on the same bootstrap resamples, median [16, 84]")
    for lab, M, sel in (("Z all", mu, np.ones(len(x), bool)), ("Z top", mu, top), ("Z bottom", mu, ~top),
                        ("M all", muM, np.ones(len(x), bool))):
        if sel.sum() < 4:
            continue
        xs = x[sel]; Bs = rng.integers(0, sel.sum(), (NB, sel.sum())); Xs = xs[Bs]
        s0, i0 = ols(Xs, M["prod"][sel][Bs])
        for arm in [a for a in ("w3", "emul") if a in M]:
            sl, ic = np.polyfit(xs, M[arm][sel], 1); rho = stats.spearmanr(xs, M[arm][sel])[0]
            s1, i1 = ols(Xs, M[arm][sel][Bs]); ds, di = q(s1 - s0), q(i1 - i0)
            print(f"  {lab:9s} n {sel.sum():2d} {arm:5s}: slope {sl:.3f} icpt {ic:6.1f} rho {rho:+.2f} | d slope {ds[1]:+.3f} "
                  f"[{ds[0]:+.3f}, {ds[2]:+.3f}]  d icpt {di[1]:+.1f} [{di[0]:+.1f}, {di[2]:+.1f}]  "
                  f"d mu median {np.median(M[arm][sel] - M['prod'][sel]):+.1f}")
        sl, ic = np.polyfit(xs, M["prod"][sel], 1)
        print(f"  {lab:9s} n {sel.sum():2d} prod : slope {sl:.3f} icpt {ic:6.1f} rho {stats.spearmanr(xs, M['prod'][sel])[0]:+.2f}")
    r = stats.pearsonr(mu["w3"] - mu["prod"], mu["emul"] - mu["prod"])[0]
    print(f"  per-crop d mu, real re-SP vs emulation: Pearson r {r:+.2f}; median |w3 - emul| {np.median(np.abs(mu['w3'] - mu['emul'])):.1f} cm")

    print("\n== B. pixels (Z crops, same origin only): L1 distance / production charge, median [16, 84]")
    import json
    mA = json.loads(str(zA["meta"])); mB = json.loads(str(zB["meta"]))
    same = np.array([(mA[iA[k]]["c0"], mA[iA[k]]["t0"]) == (mB[iB[k]]["c0"], mB[iB[k]]["t0"]) for k in keys])
    E = (S1.filt_ratio(ZA) * (ZA > 0)).astype(np.float32)
    for lab, (P, Q) in (("real re-SP vs production", (ZB, ZA)), ("emulation vs production", (E, ZA)),
                        ("real re-SP vs emulation", (ZB, E))):
        d = np.array([np.abs(P[i] - Q[i]).sum() / ZA[i].sum() for i in np.nonzero(same)[0]])
        print(f"  {lab:26s} n {same.sum():2d}: {np.median(d):.4f} [{np.quantile(d, .16):.4f}, {np.quantile(d, .84):.4f}]")
    qa, qb = ZA.sum((1, 2)), ZB.sum((1, 2)); na, nb = (ZA > 0).sum((1, 2)), (ZB > 0).sum((1, 2))
    print(f"  crop charge w3/prod median {np.median(qb / qa):.4f} [{np.quantile(qb / qa, .16):.4f}, {np.quantile(qb / qa, .84):.4f}]; "
          f"non-zero pixels w3/prod median {np.median(nb / na):.4f}")
    fA, fB = S1.features(ZA), S1.features(ZB)
    for j, nm in ((4, "chan rms/tick"), (6, "chan rms2 p25"), (3, "tick rms/ch"), (5, "tick rms2 p25")):
        print(f"  {nm:14s} prod median {np.nanmedian(fA[:, j]):.3f}  w3 median {np.nanmedian(fB[:, j]):.3f}  "
              f"paired d median {np.nanmedian(fB[:, j] - fA[:, j]):+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
