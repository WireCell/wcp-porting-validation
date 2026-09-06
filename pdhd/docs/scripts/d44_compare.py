#!/usr/bin/env python3
"""doc pdhd/02 -- PDHD copy (BY DUPLICATION, unchanged logic) of pdvd/docs/nf_sp_img_clus/scripts/d44_compare.py;
the PDVD script is untouched.  It is detector-agnostic: it only reads the d42_* analysis products.

Original header follows.

doc pdvd/44 -- before/after tables and figures for the effective-smearing arms.

Reads, per arm, the doc-42 analysis products (unchanged scripts, run on the new arms):
  <pre>_blocks.tsv   d42_proj2d_resid.py   per (block, plane): B_foot U_foot chi2N_foot uncov_foot pull_rms f_off ...
  <pre>_window.tsv   d42_shape_diag.py     pooled B / U inside Chebyshev radii per plane
  <pre>_summary.tsv  d42_dqdx_rr.py        per tier: k_pop, chi2, per-rr-bin median ratio to the reference table
and writes
  <out>_compare.tsv / markdown to stdout      the pre-registered metrics per arm and plane
  <out>_compare_2d.png                        B_foot and U_foot distributions per plane, arms overlaid
  <out>_compare_dqdx.png                      dQ/dx ratio vs rr per arm (doc-55 tier and contrast tier)
  <out>_verdicts.tsv                          the STM status census: counts per status and the flips vs the reference

Usage:
  d44_compare.py --ref name=prefix --arm name=prefix [--arm ...] --out figs/44_pdvd
  where prefix is what d42_proj2d_resid.py / d42_shape_diag.py / d42_dqdx_rr.py were given as --out
  (the three may differ: pass  name=resid_prefix,diag_prefix,dqdx_prefix)
"""
import argparse, csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RR_BINS = ["0_1", "1_2", "2_3", "3_5", "5_7", "7_10", "10_15", "15_20", "20_30", "30_40", "40_60"]
RR_MID = [0.5, 1.5, 2.5, 4, 6, 8.5, 12.5, 17.5, 25, 35, 50]


def read_tsv(path, comment="#"):
    rows = []
    with open(path) as f:
        lines = [l for l in f if not l.startswith(comment)]
    rd = csv.DictReader(lines, delimiter="\t")
    for r in rd:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                pass
        rows.append(r)
    return rows


def parse_arm(spec):
    name, pre = spec.split("=", 1)
    parts = pre.split(",")
    if len(parts) == 1:
        parts = parts * 3
    return name, dict(resid=parts[0], diag=parts[1], dqdx=parts[2])


def wmed(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


def iqr(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return (float(np.percentile(x, 25)), float(np.percentile(x, 75))) if len(x) else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--arm", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--status", type=int, default=0)
    ap.add_argument("--tier", default="doc55_muon")
    a = ap.parse_args()
    arms = [parse_arm(a.ref)] + [parse_arm(s) for s in a.arm]
    data = {}
    for name, pre in arms:
        d = {}
        d["blocks"] = read_tsv(pre["resid"] + "_blocks.tsv")
        d["window"] = read_tsv(pre["diag"] + "_window.tsv") if os.path.exists(pre["diag"] + "_window.tsv") else []
        d["summary"] = read_tsv(pre["dqdx"] + "_summary.tsv") if os.path.exists(pre["dqdx"] + "_summary.tsv") else []
        data[name] = d

    # ---------------- the 2-D metrics
    metrics = ["B_foot", "U_foot", "chi2N_foot", "pull_rms", "pull_tail3", "uncov_foot", "f_off", "f_off_far"]
    out_rows = []
    print("## 2-D pixel charge, accepted passes (status %d), medians over blocks [IQR]\n" % a.status)
    hdr = "| plane | metric | " + " | ".join(n for n, _ in arms) + " |"
    print(hdr); print("|" + "---|" * (2 + len(arms)))
    for p in "UVW":
        for m in metrics:
            cells = []
            for name, _ in arms:
                b = [r for r in data[name]["blocks"] if r["plane"] == p and r["status"] == a.status]
                v = [r[m] for r in b]
                med = wmed(v); lo, hi = iqr(v)
                cells.append("%.3f [%.3f, %.3f]" % (med, lo, hi))
                out_rows.append(dict(arm=name, plane=p, metric=m, median=med, q25=lo, q75=hi, n=len(v)))
            print("| %s | %s | %s |" % (p, m, " | ".join(cells)))
        # pooled (charge-weighted) B and U on the footprint, and the Chebyshev<=2 window from shape_diag
        cells_b, cells_u, cells_w = [], [], []
        for name, _ in arms:
            b = [r for r in data[name]["blocks"] if r["plane"] == p and r["status"] == a.status]
            q = np.array([r["Q_foot"] for r in b]); B = np.array([r["B_foot"] for r in b]); U = np.array([r["U_foot"] for r in b])
            ok = np.isfinite(B) & np.isfinite(U) & (q > 0)
            pb = (B[ok] * q[ok]).sum() / q[ok].sum(); pu = (U[ok] * q[ok]).sum() / q[ok].sum()
            cells_b.append("%.3f" % pb); cells_u.append("%.3f" % pu)
            out_rows.append(dict(arm=name, plane=p, metric="B_foot_pooled", median=pb, q25=np.nan, q75=np.nan, n=int(ok.sum())))
            out_rows.append(dict(arm=name, plane=p, metric="U_foot_pooled", median=pu, q25=np.nan, q75=np.nan, n=int(ok.sum())))
            w = [r for r in data[name]["window"] if r["plane"] == p and r["kind"] == "cheb" and r["radius"] == 2]
            cells_w.append("B %.3f / U %.3f" % (w[0]["B"], w[0]["U"]) if w else "n/a")
            if w:
                out_rows.append(dict(arm=name, plane=p, metric="B_cheb2", median=w[0]["B"], q25=np.nan, q75=np.nan, n=0))
                out_rows.append(dict(arm=name, plane=p, metric="U_cheb2", median=w[0]["U"], q25=np.nan, q75=np.nan, n=0))
        print("| %s | B_foot pooled (charge-weighted) | %s |" % (p, " | ".join(cells_b)))
        print("| %s | U_foot pooled (charge-weighted) | %s |" % (p, " | ".join(cells_u)))
        print("| %s | Chebyshev<=2 window (all blocks) | %s |" % (p, " | ".join(cells_w)))

    # ---------------- dQ/dx
    print("\n## dQ/dx vs residual range, per tier\n")
    print("| tier | quantity | " + " | ".join(n for n, _ in arms) + " |"); print("|" + "---|" * (2 + len(arms)))
    tiers = []
    for name, _ in arms:
        for r in data[name]["summary"]:
            if r["tier"] not in tiers:
                tiers.append(r["tier"])
    for t in tiers:
        for qn in ("ntracks", "k_pop", "chi2", "k_med", "k_rms", "contrast_med"):
            cells = []
            for name, _ in arms:
                s = [r for r in data[name]["summary"] if r["tier"] == t]
                cells.append(("%.4g" % s[0][qn]) if s else "n/a")
                if s:
                    out_rows.append(dict(arm=name, plane="-", metric="dqdx:%s:%s" % (t, qn), median=s[0][qn], q25=np.nan, q75=np.nan, n=int(s[0]["ntracks"])))
            print("| %s | %s | %s |" % (t, qn, " | ".join(cells)))
        cells = []
        for name, _ in arms:
            s = [r for r in data[name]["summary"] if r["tier"] == t]
            if s:
                hump = np.mean([s[0]["r_%s" % b] for b in ("3_5", "5_7", "7_10", "10_15", "15_20")])
                cells.append("%.3f" % hump)
                out_rows.append(dict(arm=name, plane="-", metric="dqdx:%s:hump_3_20" % t, median=hump, q25=np.nan, q75=np.nan, n=0))
            else:
                cells.append("n/a")
        print("| %s | mean ratio 3-20 cm (the doc-42 hump) | %s |" % (t, " | ".join(cells)))

    # ---------------- verdict census (status per block) vs the reference
    refname = arms[0][0]
    ref_st = {}
    for r in data[refname]["blocks"]:
        if r["plane"] == "U":
            ref_st[(r["event"], int(r["block"]))] = int(r["status"])
    with open(a.out + "_verdicts.tsv", "w") as fo:
        fo.write("arm\tstatus\tn_ref\tn_arm\tflips_in\tflips_out\n")
        print("\n## STM verdict census (status per (event, block)) vs %s\n" % refname)
        print("| arm | status | n_ref | n_arm | flipped in | flipped out |"); print("|---|---|---|---|---|---|")
        for name, _ in arms[1:]:
            st = {}
            for r in data[name]["blocks"]:
                if r["plane"] == "U":
                    st[(r["event"].replace("_" + name, "_" + refname), int(r["block"]))] = int(r["status"])
            allst = sorted(set(ref_st.values()) | set(st.values()))
            for s_ in allst:
                nref = sum(1 for v in ref_st.values() if v == s_); narm = sum(1 for v in st.values() if v == s_)
                fin = sum(1 for k, v in st.items() if v == s_ and ref_st.get(k, None) not in (None, s_))
                fout = sum(1 for k, v in ref_st.items() if v == s_ and st.get(k, None) not in (None, s_))
                fo.write("%s\t%d\t%d\t%d\t%d\t%d\n" % (name, s_, nref, narm, fin, fout))
                print("| %s | %d | %d | %d | %d | %d |" % (name, s_, nref, narm, fin, fout))
            common = set(ref_st) & set(st)
            print("| %s | blocks matched %d, status unchanged %d |  |  |  |  |" % (
                name, len(common), sum(1 for k in common if ref_st[k] == st[k])))

    with open(a.out + "_compare.tsv", "w") as fo:
        fo.write("arm\tplane\tmetric\tmedian\tq25\tq75\tn\n")
        for r in out_rows:
            fo.write("%s\t%s\t%s\t%.6g\t%.6g\t%.6g\t%d\n" % (r["arm"], r["plane"], r["metric"], r["median"], r["q25"], r["q75"], r["n"]))

    # ---------------- figures
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for j, p in enumerate("UVW"):
        for i, m in enumerate(("B_foot", "U_foot")):
            ax = axes[i, j]
            rng = (-0.6, 0.4) if m == "B_foot" else (0, 1.0)
            for k, (name, _) in enumerate(arms):
                b = [r for r in data[name]["blocks"] if r["plane"] == p and r["status"] == a.status]
                v = np.array([r[m] for r in b], float); v = v[np.isfinite(v)]
                ax.hist(v, bins=40, range=rng, histtype="step", lw=1.5 if k == 0 else 1.2, color="k" if k == 0 else "C%d" % k,
                        label="%s: med %.3f" % (name, np.median(v)))
            ax.axvline(0, color="grey", lw=0.8)
            ax.set_title("%s %s (accepted passes)" % (p, m)); ax.set_xlabel(m); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("2-D pixel charge on the footprint: bias B = (sum yhat - sum y)/sum y and U = sum|y-yhat|/sum y, per block")
    fig.tight_layout(); fig.savefig(a.out + "_compare_2d.png", dpi=110); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, t in zip(axes, [a.tier, "contrast_ge2"]):
        for k, (name, _) in enumerate(arms):
            s = [r for r in data[name]["summary"] if r["tier"] == t]
            if not s:
                continue
            y = [s[0]["r_%s" % b] for b in RR_BINS]; e = [s[0]["e_%s" % b] for b in RR_BINS]
            ax.errorbar(RR_MID, y, yerr=e, fmt="o-", ms=4, capsize=2, color="k" if k == 0 else "C%d" % k,
                        label="%s: k=%.3f chi2/%d=%.1f n=%d" % (name, s[0]["k_pop"], int(s[0]["nbins"]), s[0]["chi2"], int(s[0]["ntracks"])))
        ax.axhline(1, color="grey", lw=0.8); ax.set_xscale("log"); ax.set_xlabel("residual range [cm]")
        ax.set_ylabel("median dQ/dx / (k * reference)"); ax.set_title("tier %s" % t); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_ylim(0.6, 1.4)
    fig.tight_layout(); fig.savefig(a.out + "_compare_dqdx.png", dpi=110); plt.close(fig)
    print("\n->", a.out + "_compare.tsv", a.out + "_verdicts.tsv", a.out + "_compare_{2d,dqdx}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
