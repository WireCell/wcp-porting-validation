#!/usr/bin/env python3
"""doc pdvd/117 study S6 (data half) -- the time-axis width of collection-plane pulses against drift, on PDVD data.

    python3 d117_s6_data.py -j 8 [--arm d117sp] [--tag s6]      # -> scan/d117/<tag>/s6_data_points.tsv.gz + stdout summary

Sample: every fitted trajectory point (T_rec_charge, sub_cluster_id >= 0) of p98vonq's PR tree; every fitted cluster
there carries a flash t0 (checked per event below), and the fit's x is t0-corrected, so drift = 341.55 - |x| cm
(x_anode(W) of d44_sigma_fit.py; doc 98 sec 5 agrees with the tick route to 0.3 cm).
Frames: the tick-level production SP frames of work/<evt>_<arm> (d117sp == p98von, doc 117 sec 6.1), W plane.
Per point:
  - local ticks-per-wire (tpw) from the neighbouring fit points of the same sub-cluster (+-4 points, >= 1 wire apart);
    kept if < MAX_TPW = 3.  The track's own extent on one wire is a box of tpw ticks, variance tpw^2/12, which the
    summary subtracts (vc = var - tpw^2/12) and checks by tpw class.  (PDVD drifts vertically, so cosmic tracks are
    mostly steep in x: tpw < 1 kept only 15 points per event.)
  - channel = W ident of floor(pw); one measurement per (sub-cluster, channel); bad channels (T_bad_ch) skipped;
  - the pulse: the peak within [4 pt - 8, 4 pt + 18]; the contiguous non-zero run holding it (the ROI segment the SP
    left); ISOLATED if nothing else above 10 % of the peak lies within +-25 ticks outside that run, and the run is
    <= 40 ticks;
  - sigma_t^2 = charge-weighted tick variance over the run; also the peak, the run length, and a +-6-tick windowed
    variance (less tail-sensitive).
Summary: sigma_t^2 against drift time, per-event bootstrap of a binned-median line; D_L,eff = k v^2 (0.5 us)^2 / 2.
The estimator's response to a KNOWN D_L comes from the simulation half (d117_s6_sim.py); this number alone is not D_L.
"""
import argparse, collections, gzip, multiprocessing as mp, os, sys
import numpy as np
import uproot
import d98_michel_crops as C

WORK = C.IMG + "/pdvd/work"
X_ANODE = 341.55
V = 1.48073                       # mm/us, production .tlas both volumes
TICK = 0.5
MAX_TPW = 3.0
EVENTS = None


def load_w_frames(evt, arm):
    """{ident: (apa, row)} and the 8 frames"""
    fr, where = {}, {}
    for apa in range(8):
        chans, frame, tbin = C.load_gauss(f"{WORK}/{evt}_{arm}/" + C.FRAME_PFX["pdvd"] % apa)
        assert tbin == 0, (evt, apa, tbin)
        fr[apa] = frame
        for i, ch in enumerate(chans):
            where[int(ch)] = (apa, i)
    return fr, where


def measure(row, tc):
    lo, hi = max(tc - 8, 0), min(tc + 19, len(row))
    if hi <= lo or row[lo:hi].max() <= 0:
        return None
    pk = lo + int(np.argmax(row[lo:hi])); a = row[pk]
    s = pk
    while s > 0 and row[s - 1] > 0:
        s -= 1
    e = pk
    while e < len(row) - 1 and row[e + 1] > 0:
        e += 1
    run = e - s + 1
    w0, w1 = max(s - 25, 0), min(e + 26, len(row))
    outside = np.r_[row[w0:s], row[e + 1:w1]]
    iso = run <= 40 and (outside.max() if len(outside) else 0) < 0.1 * a
    t = np.arange(s, e + 1); q = row[s:e + 1]
    m = (q * t).sum() / q.sum(); v = (q * (t - m) ** 2).sum() / q.sum()
    ws, we = max(pk - 6, s), min(pk + 7, e + 1)
    tw = np.arange(ws, we); qw = row[ws:we]; mw = (qw * tw).sum() / qw.sum()
    vw = (qw * (tw - mw) ** 2).sum() / qw.sum()
    return dict(peak=float(a), run=run, var=float(v), var6=float(vw), q=float(q.sum()), iso=bool(iso), tpk=pk)


def one(evt, arm="d117sp"):
    f = uproot.open(f"{WORK}/{evt}_p98vonq/tracking-pr.root")
    r = f["T_rec_charge"].arrays(["x", "pw", "pt", "q", "cluster_id", "sub_cluster_id"], library="np")
    tcl = f["T_cluster"].arrays(["cluster_id", "flash_id"], library="np")
    fl = dict(zip(tcl["cluster_id"], tcl["flash_id"]))
    bad = set(int(c) for c in f["T_bad_ch"].arrays(["chid"], library="np")["chid"])
    ident = C.w_ident("pdvd")
    fr, where = load_w_frames(evt, arm)
    out, n = [], collections.Counter()
    for sub in np.unique(r["sub_cluster_id"]):
        if sub < 0:
            continue
        s = np.nonzero(r["sub_cluster_id"] == sub)[0]
        cid = int(r["cluster_id"][s[0]])
        if fl.get(cid, -1) < 0:
            n["no flash"] += len(s); continue
        x, pw, pt = r["x"][s], r["pw"][s], r["pt"][s]
        seen = set()
        for k in range(len(s)):
            n["points"] += 1
            j1, j2 = max(k - 4, 0), min(k + 4, len(s) - 1)
            dw = pw[j2] - pw[j1]
            if abs(dw) < 1.0:
                n["< 1 wire span"] += 1; continue
            tpw = abs((x[j2] - x[j1]) / dw) / (V * 0.1) / TICK
            if tpw >= MAX_TPW:
                n["tpw cut"] += 1; continue
            ch = ident.get(int(np.floor(pw[k])))
            if ch is None or ch in bad or ch not in where:
                n["no/bad channel"] += 1; continue
            if (sub, ch) in seen:
                n["dup channel"] += 1; continue
            seen.add((sub, ch))
            apa, i = where[ch]
            m = measure(fr[apa][i], int(pt[k]) * 4 + 2)
            if m is None:
                n["no pulse"] += 1; continue
            n["measured"] += 1; n["isolated"] += m["iso"]
            drift = X_ANODE - abs(x[k])
            out.append(dict(evt=evt, cid=cid, sub=int(sub), apa=apa, ch=ch, x=float(x[k]), drift=float(drift),
                            t_us=float(drift / (V * 0.1)), tpw=float(tpw), **m))
    return out, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", type=int, default=8)
    ap.add_argument("--arm", default="d117sp")
    ap.add_argument("--tag", default="s6")
    a = ap.parse_args()
    evs = sorted({d.rsplit("_", 1)[0] for d in os.listdir(WORK) if d.endswith("_" + a.arm)})
    with mp.Pool(a.j) as pool:
        res = pool.starmap(one, [(e, a.arm) for e in evs])
    rows = [x for o, _ in res for x in o]
    n = sum((c for _, c in res), collections.Counter())
    outd = f"{C.IMG}/pdvd/docs/scan/d117/{a.tag}"
    os.makedirs(outd, exist_ok=True)
    cols = ["evt", "cid", "sub", "apa", "ch", "x", "drift", "t_us", "tpw", "peak", "run", "var", "var6", "q", "iso", "tpk"]
    with gzip.open(f"{outd}/s6_data_points_{a.arm}.tsv.gz", "wt") as fo:
        fo.write("# doc pdvd/117 S6 data points (d117_s6_data.py --arm %s); one row per (sub-cluster, W channel)\n" % a.arm)
        fo.write("\t".join(cols) + "\n")
        for r_ in rows:
            fo.write("\t".join(f"{r_[c]:.5g}" if isinstance(r_[c], float) else str(r_[c]) for c in cols) + "\n")
    print(f"# doc pdvd/117 S6 data, arm {a.arm}: {len(evs)} events; counts {dict(n)}")
    summarize(rows)
    return 0


def summarize(rows, label="data"):
    rng = np.random.default_rng(20260925)
    R = [r for r in rows if r["iso"]]
    ev = np.array([r["evt"] for r in R]); t = np.array([r["t_us"] for r in R])
    pk = np.array([r["peak"] for r in R])
    tp = np.array([r["tpw"] for r in R])
    for vname in ("vc", "vc6", "var"):
        y = np.array([r["var" if vname == "vc" else "var6" if vname == "vc6" else "var"] for r in R])
        if vname != "var":
            y = y - tp ** 2 / 12
        for alab, asel in (("all", pk > 0), ("peak >= 3000 e", pk >= 3000), ("peak >= 8000 e", pk >= 8000),
                           ("tpw < 1", tp < 1), ("tpw 1-2", (tp >= 1) & (tp < 2)), ("tpw 2-3", tp >= 2)):
            sel = asel & (t > 60)
            if sel.sum() < 40:
                print(f"  {label} {vname:5s} {alab:15s} n {sel.sum():6d}: too few"); continue
            k, c = binned_line(t[sel], y[sel])
            evu = np.unique(ev[sel]); ks = []
            for _ in range(300):
                pick = rng.choice(evu, len(evu))
                idx = np.concatenate([np.nonzero((ev == e) & sel)[0] for e in pick])
                ks.append(binned_line(t[idx], y[idx])[0])
            klo, khi = np.quantile(ks, [0.16, 0.84])
            conv = (V * 0.1) ** 2 * TICK ** 2 / 2 * 1e6          # (ticks^2/us) -> cm^2/s
            print(f"  {label} {vname:5s} {alab:15s} n {sel.sum():6d}: sigma_t^2 = {c:6.3f} + {k * 1000:.4f}e-3 t[us] "
                  f"[{klo * 1000:.4f}, {khi * 1000:.4f}]  -> D_L,eff {k * conv:.2f} [{klo * conv:.2f}, {khi * conv:.2f}] cm^2/s")
    edges = np.arange(0, 2400, 300)
    y = np.array([r["var"] for r in R])
    print("  binned median sigma_t^2 (var, all isolated): " + "  ".join(
        f"{lo}-{hi}us:{np.median(y[(t >= lo) & (t < hi)]):.3f}(n{((t >= lo) & (t < hi)).sum()})"
        for lo, hi in zip(edges[:-1], edges[1:]) if ((t >= lo) & (t < hi)).sum() > 20))


def binned_line(t, y, nb=8):
    """robust line: medians of y in equal-population bins of t, then OLS through the bin medians"""
    if len(t) < nb * 5:
        return np.nan, np.nan
    q = np.quantile(t, np.linspace(0, 1, nb + 1))
    tm, ym = [], []
    for lo, hi in zip(q[:-1], q[1:]):
        b = (t >= lo) & (t <= hi)
        tm.append(np.median(t[b])); ym.append(np.median(y[b]))
    return np.polyfit(tm, ym, 1)


if __name__ == "__main__":
    sys.exit(main())
