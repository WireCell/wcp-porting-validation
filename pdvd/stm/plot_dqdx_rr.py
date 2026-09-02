#!/usr/bin/env python3
"""dQ/dx vs residual range of the PDVD STM sample against the expectation
curves, and the field check (doc pdvd/25 secs 7c, 12; owner 2026-09-02: "the
dQ/dx comparison with data is the key confirmation for the E field").

Reads stm/sample_points.tsv + stm/sample_index.tsv (collect_stm_sample.py) and
stm/pdvd_ref_dqdx.json.  Produces:
  * left: per-track binned medians of the muons over the 0.44 kV/cm muon table,
    the 0.50 kV/cm muon table and the proton table;
  * right: the SAMPLE median ratio data/table per rr bin for the two fields,
    after removing ONE free overall scale per field (gain / lifetime /
    0.85 fudge are uncalibrated) -- the SHAPE is what discriminates the field:
    the Bragg bins move ~4-6 % between 0.44 and 0.50, the plateau ~1.6 %;
  * a TSV with the per-bin ratios, chi2 per field, and the per-track k_muon
    distribution (median, rms), plus dQ/dx vs drift distance at the plateau
    (lifetime handle).

Usage:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 stm/plot_dqdx_rr.py -o docs/pics/pdvd_stm_dqdx_rr.png --tsv stm/dqdx_rr_field_check.tsv
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BINS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 7), (7, 10), (10, 15), (15, 20), (20, 30), (30, 40), (40, 60)]


def load_ref(key, suffix=""):
    d = json.load(open(os.path.join(HERE, "pdvd_ref_dqdx.json")))
    t = d[key + suffix]; x = t["start"] + t["step"] * np.arange(len(t["values"]))
    return x, np.asarray(t["values"], float)


def read_points(path):
    rows = [l.split("\t") for l in open(path) if not l.startswith("#") and not l.startswith("particle\t")]
    part = np.array([r[0] for r in rows]); ev = np.array([r[1] for r in rows]); blk = np.array([int(r[2]) for r in rows])
    a = np.array([[float(v) for v in r[3:]] for r in rows])   # rr dqdx dx x y z drift_cm drift_us
    return part, ev, blk, a


def binned_median(rr, dq):
    cen, med, err, n = [], [], [], []
    for lo, hi in BINS:
        s = (rr >= lo) & (rr < hi) & (dq > 0)
        if s.sum() >= 5:
            v = np.log(dq[s]); cen.append(0.5 * (lo + hi)); med.append(float(np.exp(np.median(v))))
            err.append(float(np.exp(np.median(v)) * 1.2533 * np.std(v) / np.sqrt(s.sum()))); n.append(int(s.sum()))
    return np.array(cen), np.array(med), np.array(err), np.array(n)


def field_check(cen, med, err, refs, rr_max=60.0, sys_floor=0.03):
    out = {}
    for name, (rx, rv) in refs.items():
        s = (cen <= rr_max)
        model = np.interp(cen[s], rx, rv); r = med[s] / model
        e = np.sqrt((err[s] / model) ** 2 + sys_floor ** 2)
        k = float(np.exp(np.sum(np.log(r) / e ** 2) / np.sum(1 / e ** 2)))   # one free scale
        chi2 = float(np.sum(((r / k - 1) / e) ** 2)); out[name] = dict(k=k, chi2=chi2, ndf=int(s.sum()) - 1, ratio=r / k, err=e, cen=cen[s])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default=os.path.join(HERE, "sample_points.tsv"))
    ap.add_argument("--index", default=os.path.join(HERE, "sample_index.tsv"))
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--tsv")
    ap.add_argument("--mip", type=float, default=55000.0)
    args = ap.parse_args()
    part, ev, blk, a = read_points(args.points)
    mu = part == "muon"
    rr, dq, drift = a[:, 0], a[:, 1], a[:, 6]
    refs = {"muon 0.44 kV/cm (config)": load_ref("MuonDeDx"), "muon 0.50 kV/cm": load_ref("MuonDeDx", "_E050")}
    prx, prv = load_ref("ProtonDeDx")
    cen, med, err, n = binned_median(rr[mu], dq[mu])
    fc = field_check(cen, med, err, refs)
    # per-track k_muon from the index
    ks = [float(l.split("\t")[10]) for l in open(args.index) if not l.startswith("#") and not l.startswith("particle\t") and l.startswith("muon")]
    ntracks = len(ks)
    # plateau dQ/dx vs drift (lifetime handle): points with 20 <= rr < 60
    pl = mu & (rr >= 20) & (rr < 60) & (dq > 0)
    dbins = np.arange(0, 340.01, 40.0); dcen, dmed = [], []
    for lo, hi in zip(dbins[:-1], dbins[1:]):
        s = pl & (drift >= lo) & (drift < hi)
        if s.sum() >= 20: dcen.append(0.5 * (lo + hi)); dmed.append(float(np.exp(np.median(np.log(dq[s])))))
    dcen, dmed = np.array(dcen), np.array(dmed)
    slope = None
    if len(dcen) >= 3:
        p = np.polyfit(dcen, np.log(dmed), 1); slope = p[0]   # 1/cm; tau = -1/(slope * v)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))
    ax = axes[0]
    for (name, (rx, rv)), ls in zip(refs.items(), ("-", "--")):
        ax.plot(rx, rv / 1e3, ls, color="#0b0b0b", lw=1.8, label=name, zorder=6)
    ax.plot(prx, prv / 1e3, ":", color="#52514e", lw=1.6, label="proton 0.44 kV/cm", zorder=5)
    ax.axhline(args.mip / 1e3, ls=":", color="#a3a29b", lw=1.2, label=f"flat MIP {args.mip/1e3:.0f} ke/cm")
    for (e_, b_) in sorted(set(zip(ev[mu], blk[mu]))):
        s = mu & (ev == e_) & (blk == b_); c, m, _, _ = binned_median(rr[s], dq[s])
        ax.plot(c, m / 1e3, "-", color="#2a78d6", lw=0.8, alpha=0.35)
    ax.errorbar(cen, med / 1e3, yerr=err / 1e3, fmt="o", color="#e34948", ms=5, zorder=8, label=f"sample median ({ntracks} muons)")
    ax.set_xlim(0, 60); ax.set_ylim(0, 260); ax.set_xlabel("residual range [cm]"); ax.set_ylabel("dQ/dx [e/cm x 1e3]"); ax.grid(alpha=0.2); ax.legend(fontsize=8)
    ax.set_title("PDVD STM-tagged muons vs expectation (uncalibrated data)", fontsize=10)
    ax = axes[1]
    for (name, f), col, mk in zip(fc.items(), ("#2a78d6", "#e34948"), ("o", "s")):
        ax.errorbar(f["cen"], f["ratio"], yerr=f["err"], fmt=mk + "-", color=col, ms=5, lw=1.2, label=f"{name}: k={f['k']:.3f}, chi2/ndf={f['chi2']:.1f}/{f['ndf']}")
    ax.axhline(1, ls=":", color="#a3a29b"); ax.set_xlim(0, 60); ax.set_ylim(0.8, 1.2); ax.set_xlabel("residual range [cm]"); ax.set_ylabel("data / table, one free scale removed")
    ax.set_title("field check: which table's SHAPE fits (Bragg bins vs plateau)", fontsize=10); ax.grid(alpha=0.2); ax.legend(fontsize=8)
    ax = axes[2]
    ax.hist(ks, bins=np.arange(0.7, 1.4, 0.025), color="#2a78d6", alpha=0.8)
    ax.set_xlabel("per-track free scale k vs the 0.44 kV/cm muon table"); ax.set_ylabel("tracks"); ax.grid(alpha=0.2)
    ax.set_title(f"k_muon: median {np.median(ks):.3f}, rms {np.std(ks):.3f} (n={ntracks})" + (f"\nplateau dQ/dx vs drift slope {slope*1e3:.2f}e-3/cm" if slope is not None else ""), fontsize=10)
    fig.tight_layout(); fig.savefig(args.out, dpi=140); print("wrote", args.out)
    if args.tsv:
        with open(args.tsv, "w") as fh:
            fh.write("# PDVD STM sample: binned-median data/table ratios (one free scale k per table removed); sys floor 3%/bin\n")
            for name, f in fc.items():
                fh.write(f"# {name}: k={f['k']:.4f} chi2={f['chi2']:.2f} ndf={f['ndf']}\n")
            fh.write("rr_cen\tn\tmedian_dqdx\t" + "\t".join("ratio[%s]" % n for n in fc) + "\n")
            for i, c in enumerate(cen):
                fh.write(f"{c:.1f}\t{n[i]}\t{med[i]:.1f}\t" + "\t".join(f"{f['ratio'][i]:.4f}" for f in fc.values()) + "\n")
            fh.write(f"# k_muon per track: n={ntracks} median={np.median(ks):.4f} rms={np.std(ks):.4f}\n")
            fh.write("# plateau (20<=rr<60) median dQ/dx vs drift distance [cm]:\n")
            for c, m in zip(dcen, dmed): fh.write(f"#   drift {c:.0f}: {m:.1f}\n")
            if slope is not None: fh.write(f"# ln(dQ/dx) vs drift slope = {slope:.3e} /cm  => tau_eff = {-1/(slope*0.148073):.0f} us if pure attenuation (v=0.148073 cm/us)\n")
        print("wrote", args.tsv)


if __name__ == "__main__":
    main()
