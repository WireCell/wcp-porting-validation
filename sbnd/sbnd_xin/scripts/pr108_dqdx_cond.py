#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/108 sec 9/10 -- conditioning of the dQ/dx system near a junction.

Reads a WCT_DQDX_DUMP / WCP_DQDX_DUMP file (same record layout on both sides):
  <call> dqdx call n3d= n2d=u/v/w iters= err=
  <call> A <row> <col> <value>            (the full sparse normal matrix)
  <call> b <i> <b_i> <x0_i>               (rhs and the initial guess)
  <call> dqdx <i> x y z local_dx regU regV regW cU sU bU cV sV bV cW sW bW Fdiag pos nconn [j:ou/ov/ow:Fij ...]
Picks the LAST call that has a position within --rj of the junction --j (the same
rule as pr108_dqdx_diff.py) and reports, for the charge summed over the positions
within --r of the junction:
  - the dumped solution vs a direct (dense LU) solve of the dumped A,b  -> is the solve converged?
  - cond(A) (dense 2-norm)
  - the spread of the near-J charge when A's stored entries are multiplied by
    (1 + N(0, eps)), symmetrically unless --asym, and the system re-solved
    (--eps, --draws, --seed; the seed is fixed so the numbers are reproducible): the
    discriminator for an ill-conditioned split of overlapped charge, together with
    the deterministic first-order sensitivity kappa (sigma of the relative change of
    the near-J charge per unit relative change of A's entries).
  - the reg flags (1 = plane cannot constrain this position: dead channel, or the
    projected wire/time carries no measured 2-D point) and the connected-vertex
    overlaps of the near-J positions.
Optional --swap OTHER.dump: solve this arm's A with the other arm's b and vice
versa, matching positions by 3-D position (--match cm), to locate a cross-arm
factor in A rather than in the data.

Usage:
  pr108_dqdx_cond.py --j x y z --dump LABEL=file [--dump LABEL2=file2 ...]
                     [--r 1.5] [--rj 2] [--eps 0.01 --eps 0.03] [--draws 200] [--seed 20260821]
"""
import argparse
import numpy as np


def parse(path):
    calls = {}
    for line in open(path):
        f = line.split()
        if len(f) < 3:
            continue
        c = int(f[0])
        d = calls.setdefault(c, {"hdr": "", "A": [], "b": {}, "rows": []})
        if f[1] == "dqdx" and f[2] == "call":
            d["hdr"] = line.strip()
        elif f[1] == "A":
            d["A"].append((int(f[2]), int(f[3]), float(f[4])))
        elif f[1] == "b":
            d["b"][int(f[2])] = (float(f[3]), float(f[4]))
        elif f[1] == "dqdx":
            d["rows"].append(dict(i=int(f[2]), p=np.array([float(f[3]), float(f[4]), float(f[5])]),
                                  dx=float(f[6]), reg=(int(f[7]), int(f[8]), int(f[9])),
                                  u=(int(f[10]), float(f[11]), float(f[12])),
                                  v=(int(f[13]), float(f[14]), float(f[15])),
                                  w=(int(f[16]), float(f[17]), float(f[18])),
                                  fd=float(f[19]), pos=float(f[20]), nconn=int(f[21]), conn=f[22:]))
    return calls


def pick(calls, J, rj):
    best = None
    for c in sorted(calls):
        rows = calls[c]["rows"]
        if not rows:
            continue
        P = np.array([r["p"] for r in rows])
        if np.linalg.norm(P - J, axis=1).min() <= rj:
            best = c
    return best


def build(d):
    n = len(d["rows"])
    idx = {r["i"]: k for k, r in enumerate(d["rows"])}
    A = np.zeros((n, n))
    for i, j, v in d["A"]:
        if i in idx and j in idx:
            A[idx[i], idx[j]] = v
    b = np.zeros(n)
    for i, (bi, _) in d["b"].items():
        if i in idx:
            b[idx[i]] = bi
    x = np.array([r["pos"] for r in d["rows"]])
    P = np.array([r["p"] for r in d["rows"]])
    return A, b, x, P


def report(lab, d, J, r, eps_list, draws, seed, sym=True):
    A, b, xd, P = build(d)
    n = len(b)
    dist = np.linalg.norm(P - J, axis=1)
    sel = dist <= r
    xs = np.linalg.solve(A, b)
    q_dump = xd[sel].sum()
    q_solve = xs[sel].sum()
    cond = np.linalg.cond(A)
    asym_meas = np.abs(A - A.T).max() / max(np.abs(A).max(), 1e-30)
    print(f"== {lab}  {d['hdr']}")
    print(f"   n3d={n}  near-J(|dx|<={r}cm) {sel.sum()} positions   dumped sum dQ {q_dump:.0f} ; "
          f"direct-solve sum dQ {q_solve:.0f} (rel diff {abs(q_solve-q_dump)/max(abs(q_dump),1e-9):.2e}) ; "
          f"max|x_direct-x_dump| {np.abs(xs-xd).max():.3g} ; cond(A) {cond:.3g} ; asym {asym_meas:.1e}")
    dead = [(f"{dist[k]:.2f}", d["rows"][k]["reg"]) for k in np.where(sel)[0]]
    nreg = sum(1 for _, g in dead if any(g))
    print(f"   reg flags near J: {nreg}/{sel.sum()} positions have a plane that cannot constrain them; "
          + " ".join(f"d={dd}:{g[0]}{g[1]}{g[2]}" for dd, g in dead))
    ov = []
    for k in np.where(sel)[0]:
        for c in d["rows"][k]["conn"]:
            try:
                j, o, _f = c.split(":")
                ou, ov_, ow = (float(t) for t in o.split("/"))
                ov.append(max(ou, ov_, ow))
            except ValueError:
                pass
    if ov:
        ov = np.array(ov)
        print(f"   connected overlaps near J: n={len(ov)} max {ov.max():.2f} med {np.median(ov):.2f} "
              f"(#>0.5 {(ov > 0.5).sum()}, #>0.9 {(ov > 0.9).sum()})")
    # First-order sensitivity of the near-J charge sum s = u^T x to a relative
    # perturbation of A's entries (dA_ij = eps * A_ij * g_ij, g ~ N(0,1), symmetric):
    #   ds = -y^T dA x with y = A^-1 u,  so sigma(ds)/s per unit eps is deterministic.
    # kappa is the amplification factor: a 1 % change of the couplings moves the
    # junction charge by kappa % (1 sigma).  No Monte-Carlo tail to argue about.
    u = sel.astype(float)
    y = np.linalg.solve(A, u)
    C = np.outer(y, xs) * A                      # c_ij = y_i A_ij x_j
    if sym:
        M = C + C.T
        var = (np.triu(M, 1) ** 2).sum() + (np.diag(C) ** 2).sum()
    else:
        var = (C ** 2).sum()
    kappa = np.sqrt(var) / abs(q_solve)
    spd = True
    try:
        np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        spd = False
    print(f"   sensitivity: kappa = {kappa:.1f}  (1 sigma relative change of the near-J charge per 1 unit"
          f" relative change of A's entries; a 1 % coupling change -> {kappa*0.01:.0%} 1-sigma); A pos-def: {spd}")
    rng = np.random.default_rng(seed)
    for eps in eps_list:
        vals = []
        nspd = 0
        for _ in range(draws):
            g = rng.normal(0.0, eps, A.shape)
            if sym:  # A is symmetric (R^T M R + F); keep the perturbed system symmetric too
                g = np.triu(g) + np.triu(g, 1).T
            Ap = A * (1.0 + g)
            try:
                np.linalg.cholesky(Ap)          # stay inside the physical (pos-def) regime
            except np.linalg.LinAlgError:
                continue
            nspd += 1
            try:
                v = np.linalg.solve(Ap, b)
            except np.linalg.LinAlgError:
                continue
            vals.append(v[sel].sum())
        v = np.array(vals) / q_solve
        if len(v) == 0:
            print(f"   perturb eps={eps:.0%}: no draw stayed positive-definite ({draws} tried)")
            continue
        print(f"   perturb eps={eps:.0%} ({len(v)}/{draws} draws stayed pos-def): near-J charge / nominal  "
              f"min {v.min():.2f} p10 {np.percentile(v,10):.2f} med {np.median(v):.2f} "
              f"p90 {np.percentile(v,90):.2f} max {v.max():.2f}  (p10-p90 spread {np.percentile(v,90)-np.percentile(v,10):.2f}, "
              f"rms {v.std():.2f})")
    return A, b, xs, P, sel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--j", type=float, nargs=3, required=True)
    ap.add_argument("--dump", action="append", default=[])
    ap.add_argument("--r", type=float, default=1.5)
    ap.add_argument("--rj", type=float, default=2.0)
    ap.add_argument("--eps", type=float, action="append", default=[])
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--asym", action="store_true", help="perturb A's entries independently (default: keep the perturbed A symmetric)")
    ap.add_argument("--swap", action="store_true", help="cross-solve the first two dumps (A of one with b of the other)")
    ap.add_argument("--match", type=float, default=0.5)
    a = ap.parse_args()
    J = np.array(a.j)
    eps = a.eps or [0.01, 0.03]
    keep = []
    for spec in a.dump:
        lab, path = spec.split("=", 1)
        calls = parse(path)
        c = pick(calls, J, a.rj)
        if c is None:
            print(f"== {lab}: no call with a position within {a.rj} cm of J")
            continue
        keep.append((lab,) + report(lab, calls[c], J, a.r, eps, a.draws, a.seed, not a.asym))
    if a.swap and len(keep) >= 2:
        (la, Aa, ba, xa, Pa, sa), (lb, Ab, bb, xb, Pb, sb) = keep[0], keep[1]
        # match positions of b onto a
        m = np.full(len(Pa), -1)
        for i, p in enumerate(Pa):
            dd = np.linalg.norm(Pb - p, axis=1)
            k = dd.argmin()
            if dd[k] <= a.match:
                m[i] = k
        ok = m >= 0
        print(f"-- swap {la}/{lb}: {ok.sum()}/{len(Pa)} positions matched within {a.match} cm")
        if ok.sum() >= 3:
            bmix = ba.copy()
            bmix[ok] = bb[m[ok]]
            x = np.linalg.solve(Aa, bmix)
            print(f"   A({la}) with b({lb}) -> near-J sum {x[sa].sum():.0f}   "
                  f"[{la} nominal {xa[sa].sum():.0f}, {lb} nominal {xb[sb].sum():.0f}]")
            amix = Ab.copy()
            sub = np.ix_(m[ok], m[ok])
            amix[sub] = Aa[np.ix_(np.where(ok)[0], np.where(ok)[0])]
            x2 = np.linalg.solve(amix, bb)
            print(f"   A({la}) block inside {lb} with b({lb}) -> near-J sum {x2[sb].sum():.0f}")


if __name__ == "__main__":
    main()
