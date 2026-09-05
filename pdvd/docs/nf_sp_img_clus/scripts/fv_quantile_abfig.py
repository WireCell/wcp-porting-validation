#!/usr/bin/env python3
"""Doc pdvd/43 -- the A/B figure: TGM / STM / FC per arm, TGM by cluster length, and
the TGM flips against the flat control by the deciding end's distance to its wall.

Inputs: fv_quantile_grade.py's JSON and one fv_curved_ab.py --geom verdict JSON per
arm (vs the flat control), whose geometry rows carry cat in {tgm_kept, tgm_lost,
tgm_gained, ...}, worst_end_cm (the deciding end) and len_cm.

Repro:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 docs/nf_sp_img_clus/scripts/fv_quantile_abfig.py /home/xqian/tmp/doc43/grade.json \
      --ab d43fvd50=/home/xqian/tmp/doc43/ab_fvd50_verdicts.json d43p80c3=... d43p90c3=... d43p90c5=... \
      --out docs/nf_sp_img_clus/figs/43_ab.png
"""
import argparse, json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grade"); ap.add_argument("--ab", nargs="+", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    G = json.load(open(a.grade))
    arms = list(G["arms"])
    AB = {}
    for s in a.ab:
        t, p = s.split("=", 1); AB[t] = json.load(open(p))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    label = {"d43fvoff": "flat 15+2.5/3\n(today)", "d43fvd50": "d50 + 3\n(doc 41)", "d43p80c3": "p80 + 3",
             "d43p90c3": "p90 + 3", "d43p90c5": "p90 + 5", "d41fvoff": "flat\n(old bin.)", "d41fvon": "d50 + 3\n(old bin.)"}
    x = np.arange(len(arms))
    # 1. counts
    ax = axes[0]
    for k, off, c in (("tgm", -0.27, "tab:red"), ("stm", 0.0, "tab:blue"), ("fc", 0.27, "tab:green")):
        v = [G["arms"][t][k] for t in arms]
        ax.bar(x + off, v, 0.26, color=c, label=k.upper())
        for xi, vi in zip(x + off, v):
            ax.text(xi, vi + 15, str(vi), ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([label.get(t, t) for t in arms], fontsize=8)
    ax.set_ylabel("tagged clusters, 99 events"); ax.set_title("tagger verdicts per arm"); ax.legend(fontsize=8)
    # 2. TGM by length
    ax = axes[1]
    lb = list(G["arms"][arms[0]]["tgm_by_length"])
    w = 0.8 / len(arms)
    for i, t in enumerate(arms):
        v = [G["arms"][t]["tgm_by_length"][b] for b in lb]
        ax.bar(np.arange(len(lb)) + (i - len(arms) / 2 + 0.5) * w, v, w, label=label.get(t, t).replace("\n", " "))
    ax.set_xticks(np.arange(len(lb))); ax.set_xticklabels([b.replace("-1000000000", "+") + " cm" for b in lb])
    ax.set_ylabel("TGM-tagged clusters"); ax.set_title("TGM by cluster length"); ax.legend(fontsize=7)
    # 3. flips vs flat by the deciding end's distance
    ax = axes[2]
    bands = [(0, 3), (3, 8), (8, 18), (18, 30), (30, 1e9)]
    bl = ["0-3", "3-8", "8-18", "18-30", "30+"]
    w = 0.8 / max(len(AB), 1)
    for i, (t, D) in enumerate(AB.items()):
        rows = [r for r in D["geometry"] if not r.get("no_t0")]
        lost = np.array([sum(1 for r in rows if r["cat"] == "tgm_lost" and lo <= r["worst_end_cm"] < hi) for lo, hi in bands])
        gain = np.array([sum(1 for r in rows if r["cat"] == "tgm_gained" and lo <= r["worst_end_cm"] < hi) for lo, hi in bands])
        xx = np.arange(len(bands)) + (i - len(AB) / 2 + 0.5) * w
        ax.bar(xx, gain, w, color=f"C{i}", label=f"{label.get(t, t).replace(chr(10), ' ')}: +{gain.sum()} / -{lost.sum()}")
        ax.bar(xx, -lost, w, color=f"C{i}", alpha=0.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(bands))); ax.set_xticklabels([b + " cm" for b in bl])
    ax.set_xlabel("deciding (farther) end's distance to its nearest wall, flat-arm PCA ends")
    ax.set_ylabel("TGM gained (up) / lost (down) vs the flat control"); ax.set_title("TGM flips vs today, by where the deciding end is")
    ax.legend(fontsize=7)
    fig.suptitle("PDVD doc 43 -- the fiducial arms on the 99-event production set (same binary, same pctrees)")
    fig.tight_layout(); fig.savefig(a.out, dpi=130); print("wrote", a.out)


if __name__ == "__main__":
    main()
