#!/usr/bin/env python3
"""doc pdhd/26 sec 3 -- the energy of the Michels the chain identifies, PDHD against PDVD, on the region estimator.

    STM_SCAN_RECORD=<smx27 record> python3 d26_michel_energy.py [--pdhd h26q2dprod] [--pdvd p96vprod] [--figdir ...]

SELECTION: the chain's STM+Michel -- is_stm == 1 AND michel_found == 1 -- on the judged populations of doc pdhd/25
(d25_bragg_michel.items: PDHD APA0 strict, PDVD with a candidate), split by the record into
  hand Michel  : hand stopper with michel_kind attached/both (the selection is right about both)
  other        : everything else the chain calls STM+Michel (STM_ONLY, other kinds, THRU)
The counts over ALL of the arm's candidates (no truth, APA0 included) are printed too.

ESTIMATORS, read from T_stm_michel:
  michel_ke_q2d_region  PRIMARY, now in production on both detectors (doc pdvd/95+96, doc pdhd/26 part A): the 2-D
                        charge in a 10 cm region around the stop, own cells (main cluster + admitted companions),
                        minus the muon fit's prediction there, through the bound recombination model.  Floored at 0.
  michel_ke_q2d_ctl     the SAME sum on a region 35 cm back up the muon body: the per-item phantom floor.
  michel_ke_best        the chain's association energy (fitted dQ/dx path + unfitted charge), for reference.

CROSS-SHARED CELLS (doc pdvd/81 sec 8a, the reason PDHD was held OFF): on a cell whose charge_err is at the
fitter's share sentinel the measurement cannot be attributed, so the sum takes the fit's non-muon prediction
max(pred_all - pred_mu, 0) there.  Re-derived per candidate from T_stm_michel_2d with the C++ rule
(CheckSTM_Michel.cxx:2116-2151: in_region = 0 <= d_stop_cm <= 10, scope 1 = own_blob != 0, pmu clamped at 0);
the offline per-plane sums are checked against michel_q2d_region_{u,v,w} (TWIN line), and the share of the region
sum carried by cross-shared cells is the split variable.

NO ENERGY TRUTH EXISTS on either detector: the record has verdicts, kinds, pins -- no energy.  The 52.8 MeV
endpoint is the only absolute anchor.  The free-muon-decay shape x^2 (3 - 2x) is drawn as a GUIDE only: it ignores
radiative loss, the bound mu- spectrum in argon, the energy deposited below threshold and the detector response.
"""
import argparse, collections, glob, json, math, os, sys
import numpy as np
import uproot
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
sys.path.insert(0, IMG + "/pdhd/docs/scan/h25")
import d25_bragg_michel as BM

ENDPOINT = 52.83
BR = ["cluster_id", "is_stm", "michel_found", "michel_conn_type", "michel_len", "michel_ke_best", "michel_ke_q2d",
      "michel_ke_q2d_region", "michel_ke_q2d_ctl", "michel_q2d_region_u", "michel_q2d_region_v", "michel_q2d_region_w",
      "michel_q2d_region_n_u", "michel_q2d_region_n_v", "michel_q2d_region_n_w", "michel_q2d_region_dropped_plane"]
REGION_CM = 10.0
COL = {"pdhd": "C0", "pdvd": "C3"}
CONN = {1: "attached (conn 1)", 2: "bridged (conn 2)", 3: "charge-only (conn 3)"}


def read(det, arm):
    """-> {key: dict} for every candidate with michel_found == 1, with the offline region re-derivation"""
    out = {}
    for f in sorted(glob.glob(f"{IMG}/{det}/work/*_{arm}/tracking-pr.root")):
        ev = os.path.basename(os.path.dirname(f))[: -len(arm) - 1]
        u = uproot.open(f)
        if "T_stm_michel" not in u:
            continue
        t = u["T_stm_michel"].arrays(BR, library="np")
        C = u["T_stm_michel_2d"].arrays(["cluster_id", "plane", "charge", "pred_mu", "pred_all", "xshared", "own_blob", "d_stop_cm"], library="np")
        for i in range(len(t["cluster_id"])):
            if int(t["michel_found"][i]) != 1:
                continue
            c = int(t["cluster_id"][i])
            d = {k: t[k][i] for k in t}
            s = (C["cluster_id"] == c) & (C["d_stop_cm"] >= 0) & (C["d_stop_cm"] <= REGION_CM) & (C["own_blob"] != 0)
            pmu = np.maximum(C["pred_mu"][s], 0.0)
            xs = C["xshared"][s] == 1
            contrib = np.where(xs, np.maximum(C["pred_all"][s] - pmu, 0.0), C["charge"][s] - pmu)
            pl = C["plane"][s]
            d["off_uvw"] = [float(contrib[pl == p].sum()) for p in range(3)]
            d["off_n"] = [int((pl == p).sum()) for p in range(3)]
            tot = float(contrib.sum()); shx = float(contrib[xs].sum())
            d["xshare"] = shx / tot if tot > 0 else 0.0
            d["n_xshared"] = int(xs.sum()); d["n_cells"] = int(s.sum())
            out[f"{ev}/{c}"] = d
    return out


def bmed(v, seed=26, n=2000):
    v = np.asarray(v, float)
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = np.median(v[rng.integers(0, len(v), size=(n, len(v)))], axis=1)
    return float(np.median(v)), float(np.percentile(m, 16)), float(np.percentile(m, 84))


def summary(label, v, ctl=None):
    v = np.asarray(v, float)
    if len(v) == 0:
        print(f"    {label:34s} n   0"); return
    m = bmed(v)
    s = (f"    {label:34s} n {len(v):3d}  median {m[0]:5.1f} [{m[1]:5.1f},{m[2]:5.1f}]  mean {v.mean():5.1f}  "
         f"p10 {np.percentile(v,10):5.1f}  p90 {np.percentile(v,90):5.1f}  max {v.max():6.1f}  "
         f">52.8 MeV {int((v > ENDPOINT).sum()):2d} ({(v > ENDPOINT).mean():.3f})  zero {int((v <= 0).sum())}")
    if ctl is not None and len(ctl):
        s += f"  | body ctl median {np.median(ctl):4.1f}"
    print(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdhd", default="h26q2dprod")
    ap.add_argument("--pdvd", default="p96vprod")
    ap.add_argument("--figdir", default=IMG + "/pdhd/docs/figs")
    a = ap.parse_args()
    print("PDHD record:", BM.HD_REC)
    S = {}
    for det, arm, pop in (("pdhd", a.pdhd, "strict"), ("pdvd", a.pdvd, "all")):
        D = read(det, arm)
        # the twin: the offline region sum must reproduce the C++ per plane
        ok = bad = 0; worst = 0.0
        for k, d in D.items():
            cpp = [float(d["michel_q2d_region_%s" % p]) for p in "uvw"]
            ncpp = [int(d["michel_q2d_region_n_%s" % p]) for p in "uvw"]
            dev = max(abs(x - y) / max(1.0, abs(y)) for x, y in zip(d["off_uvw"], cpp))
            worst = max(worst, dev)
            if dev < 1e-6 and d["off_n"] == ncpp: ok += 1
            else: bad += 1
        its, miss = BM.items(det, arm, pop)
        judged = {x[0]: x for x in its}
        sel = [(k, d, judged[k]) for k, d in D.items() if k in judged and int(d["is_stm"]) == 1]
        hm = [(k, d, x) for k, d, x in sel if x[1] in BM.STOP and x[2] in BM.MICHEL_KINDS]
        oth = [(k, d, x) for k, d, x in sel if not (x[1] in BM.STOP and x[2] in BM.MICHEL_KINDS)]
        allsel = [d for d in D.values() if int(d["is_stm"]) == 1]
        S[det] = dict(arm=arm, D=D, sel=sel, hm=hm, oth=oth, allsel=allsel)
        print(f"\n=== {det.upper()} {arm} ({'APA0 strict' if det == 'pdhd' else 'with a candidate'})")
        print(f"  TWIN: offline region sum (C++ rule from T_stm_michel_2d) reproduces michel_q2d_region_u/v/w and the cell counts on "
              f"{ok} of {ok + bad} michel_found candidates (worst relative deviation {worst:.2e})")
        print(f"  michel_found candidates in the arm {len(D)}; of them is_stm == 1: {len(allsel)} (no truth, every APA)")
        print(f"  judged STM+Michel selection {len(sel)}: hand Michel {len(hm)}, other {len(oth)} "
              f"{dict(collections.Counter(x[1] + ('/' + str(x[2]) if x[1] in BM.STOP else '') for _, _, x in oth))}")
        for name, grp in (("hand Michel", hm), ("other", oth), ("all judged selection", sel)):
            print(f"  [{name}]")
            summary("region (michel_ke_q2d_region)", [d["michel_ke_q2d_region"] for _, d, _ in grp], [d["michel_ke_q2d_ctl"] for _, d, _ in grp])
            summary("association (michel_ke_best)", [d["michel_ke_best"] for _, d, _ in grp])
        summary("region, every is_stm Michel (no truth)", [d["michel_ke_q2d_region"] for d in allsel], [d["michel_ke_q2d_ctl"] for d in allsel])
        summary("[hand Michel] region - body control", [d["michel_ke_q2d_region"] - d["michel_ke_q2d_ctl"] for _, d, _ in hm])
        print("    (region - control is a DIAGNOSTIC, not a published quantity: the control sits on the muon, where the fit's "
              "prediction is best, so it bounds the floor from below)")
        r = np.array([d["michel_ke_q2d_region"] for _, d, _ in hm]); b = np.array([d["michel_ke_best"] for _, d, _ in hm])
        okb = b > 0
        print(f"  hand Michel: region / best median {np.median(r[okb] / b[okb]):.3f} (n {okb.sum()}); region >= best on {int((r >= b).sum())} of {len(r)}")
        xsh = np.array([d["xshare"] for _, d, _ in sel]); nx = np.array([d["n_xshared"] for _, d, _ in sel])
        print(f"  cross-shared cells in the region sum (judged selection): candidates with any {int((nx > 0).sum())} of {len(nx)} "
              f"({(nx > 0).mean():.3f}); share of the region sum from them p50 {np.median(xsh):.3f} p90 {np.percentile(xsh, 90):.3f}; "
              f"share >= 0.20 on {int((xsh >= 0.2).sum())}")
        print("  by connection type (judged selection):")
        for ct in sorted(set(int(d["michel_conn_type"]) for _, d, _ in sel)):
            g = [(k, d, x) for k, d, x in sel if int(d["michel_conn_type"]) == ct]
            summary(CONN.get(ct, f"conn {ct}") + f" [hand Michel {sum(1 for _,_,x in g if x[1] in BM.STOP and x[2] in BM.MICHEL_KINDS)}]",
                    [d["michel_ke_q2d_region"] for _, d, _ in g])
        print("  by cross-shared share (judged selection):")
        summary("share < 0.20", [d["michel_ke_q2d_region"] for _, d, _ in sel if d["xshare"] < 0.2])
        summary("share >= 0.20", [d["michel_ke_q2d_region"] for _, d, _ in sel if d["xshare"] >= 0.2])
        print("  above the endpoint or at zero, by name (judged selection):")
        for k, d, x in sorted(sel, key=lambda t: -t[1]["michel_ke_q2d_region"]):
            e = d["michel_ke_q2d_region"]
            if e > ENDPOINT or e <= 0:
                print(f"    {k:15s} {x[1]:10s} {str(x[2]):14s} region {e:6.1f}  best {d['michel_ke_best']:6.1f}  ctl {d['michel_ke_q2d_ctl']:5.1f}  "
                      f"conn {int(d['michel_conn_type'])} len {d['michel_len']:5.1f} cm  xshare {d['xshare']:.2f}")
    # PDHD vs PDVD
    print("\n=== PDHD vs PDVD on hand Michel items")
    for est in ("michel_ke_q2d_region", "michel_ke_best"):
        h = [d[est] for _, d, _ in S["pdhd"]["hm"]]; v = [d[est] for _, d, _ in S["pdvd"]["hm"]]
        ks = stats.ks_2samp(h, v); mw = stats.mannwhitneyu(h, v)
        mh, mv = bmed(h), bmed(v)
        print(f"  {est:22s} PDHD n {len(h)} median {mh[0]:.1f} [{mh[1]:.1f},{mh[2]:.1f}] | PDVD n {len(v)} median {mv[0]:.1f} [{mv[1]:.1f},{mv[2]:.1f}] "
              f"| ratio of medians {mh[0]/mv[0]:.3f} | KS D {ks.statistic:.3f} p {ks.pvalue:.3f} | Mann-Whitney p {mw.pvalue:.3f}")
    h = [d["michel_ke_q2d_region"] for _, d, _ in S["pdhd"]["hm"] if d["xshare"] < 0.2]
    v = [d["michel_ke_q2d_region"] for _, d, _ in S["pdvd"]["hm"] if d["xshare"] < 0.2]
    print(f"  region, cross-shared share < 0.20 only: PDHD n {len(h)} median {np.median(h):.1f} | PDVD n {len(v)} median {np.median(v):.1f} "
          f"| KS p {stats.ks_2samp(h, v).pvalue:.3f}")
    print("  (KS / Mann-Whitney on these sample sizes have little power: a large p is NOT evidence of the same shape.)")

    # ---- figures ----------------------------------------------------------------------------------
    os.makedirs(a.figdir, exist_ok=True)
    bins = np.arange(0, 82.5, 2.5)
    xg = np.linspace(0, ENDPOINT, 200); yg = (xg / ENDPOINT) ** 2 * (3 - 2 * xg / ENDPOINT); yg /= (yg.sum() * (xg[1] - xg[0]))

    def hist(ax, det, vals, b=bins, **kw):
        v = np.clip(np.asarray(vals, float), b[0], b[-1] - 1e-6)
        ax.hist(v, bins=b, density=True, histtype="step", lw=kw.pop("lw", 1.8), color=COL[det], **kw)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, est, title in ((axs[0], "michel_ke_q2d_region", "region estimator michel_ke_q2d_region (production on both)"),
                           (axs[1], "michel_ke_best", "association estimator michel_ke_best (reference)")):
        for det in ("pdhd", "pdvd"):
            G = S[det]; v = [d[est] for _, d, _ in G["hm"]]
            hist(ax, det, v, label=f"{det.upper()} {G['arm']}: hand Michel (n {len(v)}, median {np.median(v):.1f})")
        ax.plot(xg, yg, "k--", lw=1, label="free mu decay x^2(3-2x): guide only, not a prediction")
        ax.axvline(ENDPOINT, color="0.4", lw=1)
        ax.set_xlabel("Michel energy [MeV] (last bin = overflow)"); ax.set_title(title, fontsize=9.5); ax.grid(alpha=0.3)
        ax.set_ylim(0, 0.075); ax.legend(fontsize=7, loc="upper left")
    axs[0].set_ylabel("normalised to unit area [1/MeV]")
    bc = np.arange(-10, 42.5, 2.5)
    for det in ("pdhd", "pdvd"):
        G = S[det]
        c = [d["michel_ke_q2d_ctl"] for _, d, _ in G["hm"]]
        hist(axs[2], det, c, b=bc, ls=":", lw=1.5, label=f"{det.upper()}: body control, 35 cm up the muon (median {np.median(c):.1f})")
    axs[2].set_xlabel("body-control energy [MeV] (last bin = overflow)")
    axs[2].set_title("the per-item non-Michel floor of the region estimator", fontsize=9.5)
    axs[2].grid(alpha=0.3); axs[2].legend(fontsize=7.5)
    fig.suptitle("Energy of the Michels the chain identifies (is_stm AND michel_found) on hand-scanned Michel items: "
                 "PDHD (APA0 strict) vs PDVD", fontsize=11)
    fig.tight_layout(); f1 = os.path.join(a.figdir, "26_michel_energy.png"); fig.savefig(f1, dpi=130); plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(12, 7.4), sharex=True)
    panels = [(CONN[1], lambda d: int(d["michel_conn_type"]) == 1),
              ("bridged (conn 2)", lambda d: int(d["michel_conn_type"]) == 2),
              ("no cross-shared cell in the region", lambda d: d["n_xshared"] == 0),
              ("at least one cross-shared cell in the region", lambda d: d["n_xshared"] > 0)]
    for ax, (name, pred) in zip(axs.flat, panels):
        for det in ("pdhd", "pdvd"):
            vals = [d["michel_ke_q2d_region"] for _, d, _ in S[det]["hm"] if pred(d)]
            if vals:
                hist(ax, det, vals, label=f"{det.upper()} (n {len(vals)}, median {np.median(vals):.1f})")
        ax.axvline(ENDPOINT, color="0.4", lw=1); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_title(name, fontsize=9.5)
    for ax in axs[1]: ax.set_xlabel("region Michel energy [MeV] (last bin = overflow)")
    fig.suptitle("Region Michel energy on hand Michel items, split so a pooled difference is not read as physics", fontsize=11)
    fig.tight_layout(); f2 = os.path.join(a.figdir, "26_michel_energy_split.png"); fig.savefig(f2, dpi=130); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    for det in ("pdhd", "pdvd"):
        for est, ls in (("michel_ke_q2d_region", "-"), ("michel_ke_best", "--")):
            v = np.sort([d[est] for _, d, _ in S[det]["hm"]])
            axs[0].step(v, np.arange(1, len(v) + 1) / len(v), where="post", ls=ls, color=COL[det],
                        label=f"{det.upper()} {'region' if est.endswith('region') else 'best'} (n {len(v)})")
        r = [d["michel_ke_q2d_region"] for _, d, _ in S[det]["hm"]]; b = [d["michel_ke_best"] for _, d, _ in S[det]["hm"]]
        axs[1].plot(b, r, "o", ms=3.5, alpha=0.6, color=COL[det], label=det.upper())
    axs[0].axvline(ENDPOINT, color="0.4", lw=1); axs[0].set_xlim(0, 90); axs[0].grid(alpha=0.3); axs[0].legend(fontsize=8)
    axs[0].set_xlabel("Michel energy [MeV]"); axs[0].set_ylabel("cumulative fraction"); axs[0].set_title("CDF, hand Michel items", fontsize=10)
    axs[1].plot([0, 90], [0, 90], "k:", lw=1); axs[1].set_xlim(0, 90); axs[1].set_ylim(0, 90); axs[1].grid(alpha=0.3)
    axs[1].axhline(ENDPOINT, color="0.6", lw=0.8); axs[1].axvline(ENDPOINT, color="0.6", lw=0.8)
    axs[1].set_xlabel("michel_ke_best [MeV]"); axs[1].set_ylabel("michel_ke_q2d_region [MeV]"); axs[1].legend(fontsize=8)
    axs[1].set_title("region vs association, per item (clipped at 90)", fontsize=10)
    fig.tight_layout(); f3 = os.path.join(a.figdir, "26_michel_energy_cdf.png"); fig.savefig(f3, dpi=130); plt.close(fig)
    print("\nfigures: %s %s %s" % (f1, f2, f3))


if __name__ == "__main__":
    main()
