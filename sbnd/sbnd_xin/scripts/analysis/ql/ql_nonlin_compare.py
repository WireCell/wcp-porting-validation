#!/usr/bin/env python3
"""Compare QLMatching predicted-vs-measured PE with the PMT non-linearity OFF vs ON.

Reads two BEE op.json archives (mabc.zip) produced by run_clust_QL_evt.sh with PMT_NL
off and on, and plots median pred/meas vs predicted-PE brightness per sample. The
non-linearity correction lowers the predicted PE above the knee (~700), so it can only
move the bright-channel points; below the knee the two curves coincide.

  python3 ql_nonlin_compare.py --mc-off A.zip --mc-on B.zip --data-off C.zip --data-on D.zip

See match/docs/sbnd-opdetsim-chain.md and sbnd_xin/scripts/analysis/light/pmt_nonlinearity_curve.py.
"""
import argparse, json, zipfile
import numpy as np


def load(path):
    """matched-flash per-channel (measured, predicted) PE, keyed by (event, apa, t)."""
    z = zipfile.ZipFile(path)
    out = {}
    for n in z.namelist():
        if not n.endswith("-op.json"):
            continue
        d = json.loads(z.read(n))
        ev = d["eventNo"]
        for i in range(len(d["op_t"])):
            if not d["op_cluster_ids"][i]:
                continue
            pr = np.array(d["op_pes_pred"][i], float)
            pe = np.array(d["op_pes"][i], float)
            if pr.size == 0:
                continue
            out[(ev, str(d["apa"][i]), round(float(d["op_t"][i]), 3))] = (pe, pr)
    return out


BINS = [(0, 700), (700, 1500), (1500, 3000), (3000, 8000), (8000, 1e9)]


def ratios(off, on):
    common = sorted(set(off) & set(on))
    pe, pro, prn = [], [], []
    for k in common:
        pe_o, pr_o = off[k]
        _, pr_n = on[k]
        m = pr_o > 0
        pe += list(pe_o[m]); pro += list(pr_o[m]); prn += list(pr_n[m])
    return np.array(pe), np.array(pro), np.array(prn), len(common)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mc-off", required=True); ap.add_argument("--mc-on", required=True)
    ap.add_argument("--data-off", required=True); ap.add_argument("--data-on", required=True)
    ap.add_argument("--out", default="pics/ql_pmt_nonlin_compare.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, tag, (poff, pon) in zip(
            axes, ["MC", "data"],
            [(args.mc_off, args.mc_on), (args.data_off, args.data_on)]):
        pe, pro, prn, ncom = ratios(load(poff), load(pon))
        xs, roff, ron, ns = [], [], [], []
        for lo, hi in BINS:
            m = (pro >= lo) & (pro < hi) & (pe > 0)
            if m.sum() < 3:
                continue
            xs.append(np.sqrt(lo * max(hi, lo + 1)) if hi < 1e9 else lo * 1.6)
            roff.append(np.median(pro[m] / pe[m]))
            ron.append(np.median(prn[m] / pe[m]))
            ns.append(int(m.sum()))
        ax.plot(xs, roff, "C0o-", label="non-linearity OFF")
        ax.plot(xs, ron, "C3s--", label="non-linearity ON")
        for x, n in zip(xs, ns):
            ax.annotate(f"n={n}", (x, ax.get_ylim()[0]), fontsize=6, color="0.5",
                        ha="center", va="bottom")
        ax.axhline(1.0, color="k", ls=":", lw=1)
        ax.axvline(700, color="0.6", ls="--", lw=1, label="knee (700 PE)")
        ax.set_xscale("log")
        ax.set_xlabel("predicted PE per channel (OFF)")
        ax.set_title(f"{tag}  ({ncom} matched flashes)")
        ax.grid(alpha=0.3)
        if tag == "MC":
            ax.set_ylabel("median predicted / measured PE")
            ax.legend(fontsize=8)
    fig.suptitle("QLMatching predicted/measured PE vs brightness — PMT non-linearity OFF vs ON\n"
                 "MC: rising trend (mild saturation) is flattened by the correction;  "
                 "data: falling trend = more light than charge explains (not PMT saturation)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.out, dpi=130)
    print(f"[ok] wrote {args.out}")


if __name__ == "__main__":
    main()
