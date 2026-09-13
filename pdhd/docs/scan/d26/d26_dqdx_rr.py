#!/usr/bin/env python3
"""doc pdhd/26 sec 4 -- dQ/dx vs residual range on the stopping muons with a GOOD dQ/dx-vs-RR, against the
expectation, PDHD vs PDVD.

    python3 d26_dqdx_rr.py [--pdhd h26q2dprod] [--pdvd p96vprod] [--figdir ../../figs] > dqdx_rr.txt

SELECTION ("good dQ/dx vs RR"): a HAND stopper (STM_MICHEL / STM_ONLY on the record, owner precedence) that
the chain accepted on its OWN Bragg reading -- is_stm == 1 AND topology_cleared_bits == 0 (doc pdhd/25 sec 2's
Bragg-path accept: no_bragg / shape_flat never set, P1 did not have to rescue it).  PDHD headline = APA0
strict (no role-1 point in APA0), the per-APA split uses all four; PDVD = every record item with a candidate.
Populations and truth come from d25_bragg_michel.items, so they are exactly doc 25's.

POINTS: T_stm_michel_pts role 1 (the muon chain the verdict read), q in e/cm, rr in cm from the chain's stop.
Rows with rr < 0 or q <= 0 are dropped (no profile / dead).  A vertex row is written once per segment, so a
row identical in (rr, q) to another of the same candidate is kept once; the count is printed.

EXPECTATION: each detector's OWN table, read from the chain's calib-pr-evt*.json `dqdx_ref` block (ParticleDataSet,
401 points, 0-100 cm in 0.25 cm, e/cm): the muon dQ/dx the reconstruction itself compares to (Modified Box at
that detector's field, x 0.85).  Cross-checked at 0.5 + i cm against pdhd/stm/pdhd_ref_dqdx.json (0.4959 kV/cm)
and pdvd/stm/pdvd_ref_dqdx_045.json (0.45 kV/cm).  Beyond 100 cm the last value is held (minimum ionising).
NO FREE SCALE anywhere in the headline ratio.  A plateau-normalised SHAPE ratio (k = the population median of
the per-track plateau ratio) is printed and drawn alongside.

PER-TRACK PLATEAU: doc pdvd/50 round 2's definition (d51_dqdx_rr_apa.py, PLATEAU_RR 40-60): median(q / ref(rr))
over rr 40-60 cm, tracks with >= 10 such points -- so the number compares to doc 50 sec 13.1's APA1 1.013 /
APA2 1.012 / APA3 1.033.  Drift distance = x_anode - |x| (PDHD 352.1 cm, PDVD 339.91 cm), mean over those points.

ERRORS: points on one track are correlated, so every binned ratio's error is a bootstrap over TRACKS (1000).
"""
import argparse, collections, glob, json, os, sys
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
sys.path.insert(0, IMG + "/pdhd/docs/scan/h25")
import d25_bragg_michel as BM

EDGES = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50, 60, 80, 100])
PLATEAU = (40.0, 60.0)
ANODE = {"pdhd": 352.1, "pdvd": 339.91}
REFJSON = {"pdhd": IMG + "/pdhd/stm/pdhd_ref_dqdx.json", "pdvd": IMG + "/pdvd/stm/pdvd_ref_dqdx_045.json"}
COL = {"pdhd": "C0", "pdvd": "C3"}
NBOOT = 1000


def ref_table(det, arm):
    js = sorted(glob.glob(f"{IMG}/{det}/work/*_{arm}/calib-pr-evt*.json"))
    r = json.load(open(js[0]))["dqdx_ref"]
    g = r["grid"]
    x = g["start"] + g["step"] * np.arange(g["n"])
    y = np.asarray(r["muon"], float)
    # every event carries the same block; check a handful rather than trusting the first
    for j in js[1::max(1, len(js) // 5)]:
        rr = json.load(open(j))["dqdx_ref"]
        assert np.array_equal(np.asarray(rr["muon"], float), y), f"dqdx_ref differs in {j}"
    t = json.load(open(REFJSON[det]))["MuonDeDx"]
    xs = t["start"] + t["step"] * np.arange(len(t["values"]))
    rel = np.abs(np.interp(xs, x, y) / np.asarray(t["values"]) - 1)
    return x, y, (len(js), float(rel.max()), os.path.basename(REFJSON[det]))


def points(det, arm, keys):
    """-> {key: (rr, q, x)} role-1 rows, deduplicated"""
    want = collections.defaultdict(set)
    for k in keys:
        ev, c = k.rsplit("/", 1); want[ev].add(int(c))
    out, ndup = {}, 0
    for ev, cs in want.items():
        u = uproot.open(f"{IMG}/{det}/work/{ev}_{arm}/tracking-pr.root")
        p = u["T_stm_michel_pts"].arrays(["cluster_id", "role", "rr", "q", "x"], library="np")
        for c in cs:
            m = (p["cluster_id"] == c) & (p["role"] == 1) & (p["rr"] >= 0) & (p["q"] > 0)
            rr, q, x = p["rr"][m], p["q"][m], p["x"][m]
            _, ix = np.unique(np.round(np.c_[rr, q], 6), axis=0, return_index=True)
            ix = np.sort(ix); ndup += len(rr) - len(ix)
            out[f"{ev}/{c}"] = (rr[ix], q[ix], x[ix])
    return out, ndup


def binned(P, rx, ry, k=1.0, boot=True, seed=26):
    """per RR bin: n pts, n tracks, mean q, sem, trimmed mean (10-90 %), median q, mean expectation,
    mean(q)/mean(ref) with a track-bootstrap error, median(q/ref)"""
    keys = sorted(P)
    per = []                                            # per track, per bin: (sum q, sum ref, n)
    allpts = [[] for _ in range(len(EDGES) - 1)]
    for key in keys:
        rr, q, _ = P[key]
        ref = np.interp(rr, rx, ry)
        b = np.digitize(rr, EDGES) - 1
        row = np.zeros((len(EDGES) - 1, 3))
        for i in range(len(EDGES) - 1):
            s = b == i
            if s.any():
                row[i] = (q[s].sum(), ref[s].sum(), s.sum())
                allpts[i].append(np.c_[q[s], ref[s]])
        per.append(row)
    per = np.array(per)
    rows = []
    rng = np.random.default_rng(seed)
    bidx = rng.integers(0, len(keys), size=(NBOOT, len(keys))) if boot else None
    for i in range(len(EDGES) - 1):
        if not allpts[i]:
            rows.append(None); continue
        a = np.vstack(allpts[i]); q, ref = a[:, 0], a[:, 1]
        lo, hi = np.percentile(q, [10, 90])
        tm = q[(q >= lo) & (q <= hi)].mean()
        ratio = q.mean() / (k * ref.mean())
        err = np.nan
        if boot:
            sq, sr = per[bidx, i, 0].sum(1), per[bidx, i, 1].sum(1)
            ok = sr > 0
            err = np.std(sq[ok] / (k * sr[ok]))
        rows.append(dict(lo=EDGES[i], hi=EDGES[i + 1], n=len(q), ntrk=int((per[:, i, 2] > 0).sum()),
                         mean=q.mean(), sem=q.std(ddof=1) / np.sqrt(len(q)) if len(q) > 1 else np.nan,
                         tmean=tm, med=np.median(q), ref=ref.mean(), ratio=ratio, err=err,
                         medratio=np.median(q / (k * ref)), tratio=tm / (k * ref.mean())))
    return rows


def plateau(P, rx, ry, anode):
    """-> list of (key, ratio_p, drift_cm, npts) per track with >= 10 plateau points"""
    out = []
    for key, (rr, q, x) in P.items():
        s = (rr >= PLATEAU[0]) & (rr < PLATEAU[1])
        if s.sum() >= 10:
            out.append((key, float(np.median(q[s] / np.interp(rr[s], rx, ry))), float(np.mean(anode - np.abs(x[s]))), int(s.sum())))
    return out


def boot_median(v, seed=26):
    v = np.asarray(v); rng = np.random.default_rng(seed)
    m = np.median(v[rng.integers(0, len(v), size=(NBOOT, len(v)))], axis=1)
    return float(np.median(v)), float(np.percentile(m, 16)), float(np.percentile(m, 84))


def select(det, arm, pop):
    its, miss = BM.items(det, arm, pop)
    S = [x for x in its if x[1] in BM.STOP]
    good = [x for x in S if int(x[4]["is_stm"]) == 1 and int(x[4]["topology_cleared_bits"]) == 0]
    return its, S, good, miss


def table(label, rows):
    print(f"\n  {label}")
    print("    RR bin [cm]    pts trks   mean q    +-sem   trim-mean   median  expected   mean/exp  +-boot  trim/exp  med(q/exp)")
    for r in rows:
        if r is None: continue
        print("    %5.1f-%5.1f %6d %4d %9.0f %7.0f %10.0f %9.0f %9.0f %9.3f %7.3f %9.3f %10.3f" % (
            r["lo"], r["hi"], r["n"], r["ntrk"], r["mean"], r["sem"], r["tmean"], r["med"], r["ref"],
            r["ratio"], r["err"], r["tratio"], r["medratio"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdhd", default="h26q2dprod")
    ap.add_argument("--pdvd", default="p96vprod")
    ap.add_argument("--figdir", default=IMG + "/pdhd/docs/figs")
    a = ap.parse_args()
    R = {}
    for det, arm, pop in (("pdhd", a.pdhd, "strict"), ("pdvd", a.pdvd, "all")):
        rx, ry, chk = ref_table(det, arm)
        its, S, good, miss = select(det, arm, pop)
        P, ndup = points(det, arm, [x[0] for x in good])
        npts = sum(len(v[0]) for v in P.values())
        pl = plateau(P, rx, ry, ANODE[det])
        k = boot_median([p[1] for p in pl])
        rows = binned(P, rx, ry)
        rows_k = binned(P, rx, ry, k=k[0], boot=False)
        R[det] = dict(arm=arm, rx=rx, ry=ry, P=P, rows=rows, rows_k=rows_k, pl=pl, k=k, its=its, good=good)
        print(f"=== {det.upper()} {arm}  ({'APA0 strict' if det == 'pdhd' else 'with a candidate'}; record items with no candidate {miss})")
        print(f"  expectation: calib dqdx_ref muon, identical on the {chk[0]} events' calib json checked; "
              f"max |ratio-1| vs {chk[2]} at 0.5+i cm = {chk[1]:.2e}; ref(0.5 cm) {np.interp(0.5, rx, ry):.0f}, "
              f"ref(59.5 cm) {np.interp(59.5, rx, ry):.0f}, ref(100 cm) {ry[-1]:.0f} e/cm")
        print(f"  judged items {len(its)}, hand stoppers {len(S)}, GOOD (Bragg-path accept) {len(good)} = {len(good)/len(S):.3f}")
        print(f"  role-1 points kept {npts} (duplicate vertex rows dropped {ndup}); muon length p50 "
              f"{np.median([float(x[4]['muon_len']) for x in good]):.1f} cm")
        print(f"  per-track plateau ratio (rr 40-60, >= 10 pts): n {len(pl)}, median {k[0]:.3f} [{k[1]:.3f}, {k[2]:.3f}] (bootstrap 68 %), "
              f"p16 {np.percentile([p[1] for p in pl], 16):.3f} p84 {np.percentile([p[1] for p in pl], 84):.3f}")
        table("absolute (no free scale)", rows)
        print(f"  shape: mean(q) / ({k[0]:.3f} x expected) -- plateau-normalised")
        for r in rows_k:
            if r is not None:
                print("    %5.1f-%5.1f  %.3f" % (r["lo"], r["hi"], r["ratio"]))
        # peak and near-stop summaries
        s = [r for r in rows if r and r["hi"] <= 5]
        print("  rr < 5 cm pooled mean/exp: %.3f ; rr 5-30: %.3f ; rr 30-100: %.3f" % tuple(
            sum(r["mean"] * r["n"] for r in grp) / sum(r["ref"] * r["n"] for r in grp)
            for grp in ([r for r in rows if r and r["hi"] <= 5], [r for r in rows if r and 5 <= r["lo"] and r["hi"] <= 30],
                        [r for r in rows if r and r["lo"] >= 30])))
    # ---- splits ------------------------------------------------------------------------------
    print("\n=== SPLITS")
    hd_all = select("pdhd", a.pdhd, "all")[2]
    rxh, ryh = R["pdhd"]["rx"], R["pdhd"]["ry"]
    splits = {}
    for apa in range(4):
        g = [x for x in hd_all if x[4]["apa_major"] == apa]
        P, _ = points("pdhd", a.pdhd, [x[0] for x in g])
        pl = plateau(P, rxh, ryh, ANODE["pdhd"])
        splits[f"PDHD APA{apa}"] = ("pdhd", binned(P, rxh, ryh), pl)
    rxv, ryv = R["pdvd"]["rx"], R["pdvd"]["ry"]
    for name, sgn in (("x<0", -1), ("x>0", 1)):
        P = {kk: v for kk, v in R["pdvd"]["P"].items() if np.sign(np.median(v[2])) == sgn}
        splits[f"PDVD {name}"] = ("pdvd", binned(P, rxv, ryv), plateau(P, rxv, ryv, ANODE["pdvd"]))
    for name, (det, rows, pl) in splits.items():
        m = boot_median([p[1] for p in pl]) if pl else (np.nan, np.nan, np.nan)
        ntr = max((r["ntrk"] for r in rows if r), default=0)
        print("  %-10s tracks %3d  plateau ratio median %.3f [%.3f, %.3f] (n %d)  | mean/exp rr<5 %s" % (
            name, ntr, m[0], m[1], m[2], len(pl),
            "%.3f" % (sum(r["mean"] * r["n"] for r in rows if r and r["hi"] <= 5) / max(1e-9, sum(r["ref"] * r["n"] for r in rows if r and r["hi"] <= 5)))))
    # ---- drift ---------------------------------------------------------------------------------
    print("\n=== PLATEAU RATIO vs DRIFT DISTANCE (headline populations)")
    DE = np.array([0, 50, 100, 150, 200, 250, 300, 360])
    # a drift trend pooled over two drift volumes can be a volume difference (the volumes do not sample drift
    # distance alike), so each side of the cathode is also fitted on its own
    side = lambda det, key: ("x<0" if np.median(R[det]["P"][key][2]) < 0 else "x>0")
    for det in ("pdhd", "pdvd"):
        for which in ("both", "x<0", "x>0"):
            rows_ = [p for p in R[det]["pl"] if which == "both" or side(det, p[0]) == which]
            if len(rows_) < 5:
                continue
            pl = np.array([(p[1], p[2]) for p in rows_])
            b = np.digitize(pl[:, 1], DE) - 1
            cells = []
            for i in range(len(DE) - 1):
                s = b == i
                cells.append("%d-%d: %s" % (DE[i], DE[i + 1], "%.3f (n %d)" % (np.median(pl[s, 0]), s.sum()) if s.sum() >= 3 else "n %d" % s.sum()))
            sl = np.polyfit(pl[:, 1], pl[:, 0], 1)
            print("  %s %-4s n %3d  %s | linear fit slope %+.2e /cm (x 300 cm = %+.3f)" % (
                det.upper(), which, len(pl), " ; ".join(cells), sl[0], 300 * sl[0]))

    # ---- figures -------------------------------------------------------------------------------
    os.makedirs(a.figdir, exist_ok=True)
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw=dict(height_ratios=[2, 1.2]))
    for det in ("pdhd", "pdvd"):
        D = R[det]; rows = [r for r in D["rows"] if r]
        xc = np.array([(r["lo"] + r["hi"]) / 2 for r in rows]); xe = np.array([(r["hi"] - r["lo"]) / 2 for r in rows])
        lab = "PDHD %s, APA0 strict" % D["arm"] if det == "pdhd" else "PDVD %s" % D["arm"]
        ntr = len(D["P"])
        ax.errorbar(xc, [r["mean"] / 1e3 for r in rows], xerr=xe, yerr=[r["sem"] / 1e3 for r in rows], fmt="o", ms=4,
                    color=COL[det], label=f"{lab}: mean ({ntr} tracks)")
        ax.plot(xc, [r["med"] / 1e3 for r in rows], "s", mfc="none", ms=5, color=COL[det], label=f"{lab}: median")
        xx = np.linspace(0.25, 100, 400)
        ax.plot(xx, np.interp(xx, D["rx"], D["ry"]) / 1e3, "-", color=COL[det], lw=1.2,
                label="expected muon, %s field (chain's dqdx_ref)" % ("0.4959 kV/cm" if det == "pdhd" else "0.45 kV/cm"))
        bx.errorbar(xc, [r["ratio"] for r in rows], xerr=xe, yerr=[r["err"] for r in rows], fmt="o", ms=4, color=COL[det],
                    label=f"{det.upper()} mean / expected (no free scale; track bootstrap)")
        bx.plot(xc, [r["ratio"] for r in D["rows_k"] if r], "--", color=COL[det], lw=1,
                label=f"{det.upper()} mean / ({D['k'][0]:.3f} x expected) = shape")
    ax.set_xscale("log"); ax.set_xlim(0.4, 110); ax.set_ylabel("dQ/dx [ke/cm]")
    ax.set_title("Stopping muons with a good dQ/dx vs RR (hand stopper AND chain Bragg-path accept)")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3, which="both")
    bx.axhline(1, color="k", lw=0.8); bx.axhspan(0.95, 1.05, color="0.9")
    bx.set_ylim(0.6, 1.4); bx.set_xlabel("residual range from the chain's stop [cm]"); bx.set_ylabel("measured / expected")
    bx.legend(fontsize=7.5, loc="upper right"); bx.grid(alpha=0.3, which="both")
    fig.tight_layout(); f1 = os.path.join(a.figdir, "26_dqdx_rr.png"); fig.savefig(f1, dpi=130); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, det in zip(axs, ("pdhd", "pdvd")):
        for i, (name, (d, rows, pl)) in enumerate(n for n in splits.items() if n[1][0] == det):
            rows = [r for r in rows if r]
            if not rows: continue
            xc = np.array([(r["lo"] + r["hi"]) / 2 for r in rows]) * (1 + 0.03 * i)
            ax.errorbar(xc, [r["ratio"] for r in rows], yerr=[r["err"] for r in rows], fmt="o-", ms=3, lw=0.8,
                        label="%s (%d tracks)%s" % (name, max(r["ntrk"] for r in rows), "  -- not in the headline" if name.endswith("APA0") else ""))
        ax.axhline(1, color="k", lw=0.8); ax.axhspan(0.95, 1.05, color="0.9")
        ax.set_xscale("log"); ax.set_xlim(0.4, 110); ax.set_ylim(0.4, 1.4); ax.grid(alpha=0.3, which="both")
        ax.set_xlabel("residual range [cm]"); ax.legend(fontsize=8)
        ax.set_title("PDHD by APA (candidate's majority APA)" if det == "pdhd" else "PDVD by drift volume")
    axs[0].set_ylabel("mean dQ/dx / expected (no free scale)")
    fig.tight_layout(); f2 = os.path.join(a.figdir, "26_dqdx_rr_split.png"); fig.savefig(f2, dpi=130); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, det in zip(axs, ("pdhd", "pdvd")):
        for which, mk, shade in (("x<0", "o", 0.55), ("x>0", "^", 1.0)):
            rows_ = [p for p in R[det]["pl"] if side(det, p[0]) == which]
            if not rows_:
                continue
            pl = np.array([(p[1], p[2]) for p in rows_])
            c = matplotlib.colors.to_rgba(COL[det]); c = (c[0] * shade, c[1] * shade, c[2] * shade, 1)
            ax.plot(pl[:, 1], pl[:, 0], mk, color=c, alpha=0.35, ms=3.5, label=f"{which}: per track ({len(pl)})")
            b = np.digitize(pl[:, 1], DE) - 1
            xs, ys, es = [], [], []
            for i in range(len(DE) - 1):
                s = b == i
                if s.sum() >= 3:
                    m = boot_median(pl[s, 0]); xs.append((DE[i] + DE[i + 1]) / 2); ys.append(m[0]); es.append([m[0] - m[1], m[2] - m[0]])
            if xs:
                ax.errorbar(xs, ys, yerr=np.array(es).T, fmt=mk + "-", color=c, ms=6, lw=1.5, label=f"{which}: median in 50 cm bins (n >= 3)")
        ax.axhline(1, color="k", lw=0.8); ax.set_ylim(0.5, 1.5); ax.grid(alpha=0.3); ax.legend(fontsize=7.5)
        ax.set_xlabel("drift distance = x_anode - |x|  [cm]  (anode |x| %.2f cm)" % ANODE[det])
        ax.set_title(("PDHD %s, APA0 strict" % a.pdhd) if det == "pdhd" else ("PDVD %s" % a.pdvd), fontsize=10)
    axs[0].set_ylabel("per-track plateau dQ/dx / expected (rr 40-60 cm)")
    fig.suptitle("Plateau ratio vs drift distance, each side of the cathode separately (no lifetime correction exists)", fontsize=11)
    fig.tight_layout(); f3 = os.path.join(a.figdir, "26_plateau_vs_drift.png"); fig.savefig(f3, dpi=130); plt.close(fig)
    print("\nfigures: %s %s %s" % (f1, f2, f3))


if __name__ == "__main__":
    main()
