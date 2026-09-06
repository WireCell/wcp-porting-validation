#!/usr/bin/env python3
"""doc pdvd/47 sec 9 -- the figure for the >= 2-wire tail study.

Row 1, one panel per detector: the beyond-+-2 share in excess of the Gaussian at that
selection's OWN fitted sigma, against the selection ladder (sec 8.6's two cuts plus the
width-insensitive isolation tag and their combination).  A tail made of delta rays or a
second track in the window must fall along the ladder; an instrumental one must not.
Row 2: the concentration of the tail relative to the proportional null (data vs the
simulation's own tail), and the tail share against the profile charge in deciles.

  d47_tail_plot.py --figs figs --out figs/47_tail.png
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DETS = ("pdhd", "pdvd", "sbnd")
PLC = {"U": "tab:blue", "V": "tab:orange", "W": "tab:green"}


def read(path):
    if not os.path.exists(path):
        return []
    hdr = open(path).readline().rstrip("\n").split("\t")
    out = []
    for ln in open(path).read().splitlines()[1:]:
        v = ln.split("\t")
        d = {}
        for k, x in zip(hdr, v):
            try:
                d[k] = float(x)
            except ValueError:
                d[k] = x
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    F = a.figs
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.4))

    order = None
    for j, det in enumerate(DETS):
        rows = read(os.path.join(F, "47_tail_%s_cuts.tsv" % det))
        A = ax[0, j]
        if rows:
            order = []
            for r in rows:
                if r["sel"] not in order:
                    order.append(r["sel"])
            for pl in "UVW":
                xs, ys, ns = [], [], []
                for k, sel in enumerate(order):
                    m = [r for r in rows if r["sel"] == sel and r["plane"] == pl]
                    if m:
                        xs.append(k); ys.append(max(m[0]["exc_beyond"], 1e-5)); ns.append(m[0]["n"])
                A.plot(xs, ys, "o-", color=PLC[pl], label=pl)
                for x_, y_, n_ in zip(xs, ys, ns):
                    if n_ < 300:
                        A.plot([x_], [y_], "x", color="k", ms=9, mew=1.6)
            A.set_xticks(range(len(order)))
            A.set_xticklabels([s.replace("iso+straight+rr", "all three") for s in order],
                              rotation=30, ha="right", fontsize=8)
        A.set_yscale("log"); A.set_ylim(1e-4, 5e-2)
        A.axhline(1e-4, color="0.6", lw=0.8, ls=":")
        A.set_title("%s: beyond-+-2 share above its own Gaussian" % det.upper(), fontsize=10)
        A.set_ylabel("meas - gaus (charge fraction)" if j == 0 else "")
        A.grid(alpha=0.3); A.legend(fontsize=8, ncol=3)
        note = "x = fewer than 300 profiles"
        if order is not None and not any(o.startswith("foff") for o in order):
            note += "\nsec 8.6's foff<0.15 keeps <20 profiles here"
        A.text(0.02, 0.03, note, transform=A.transAxes, fontsize=7, color="0.35")

    # --- concentration, data vs sim
    A = ax[1, 0]
    labels, dv, sv = [], [], []
    for det in DETS:
        d = {(r["sel"], r["plane"]): r for r in read(os.path.join(F, "47_tail_%s_anatomy.tsv" % det))}
        s = {r["plane"]: r for r in read(os.path.join(F, "47_tail_sim_%s_anatomy.tsv" % det))
             if r["sel"] == "S1_gauss"}
        for pl in "UVW":
            rd = d.get(("base", pl)); rs = s.get(pl)
            labels.append("%s %s" % (det.upper(), pl))
            dv.append(rd["r1_t2p"] if rd else np.nan)
            sv.append(rs["r1_t2p"] if rs else np.nan)
    x = np.arange(len(labels))
    A.bar(x - 0.2, dv, 0.4, label="data", color="tab:red")
    A.bar(x + 0.2, sv, 0.4, label="simulation", color="tab:blue")
    A.axhline(1.0, color="k", lw=1, ls="--")
    A.set_xticks(x); A.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    A.set_ylabel("top 1 % of profiles / proportional null")
    A.set_title("concentration of the >=2-wire charge (1 = proportional)", fontsize=10)
    A.grid(alpha=0.3, axis="y"); A.legend(fontsize=8)

    # --- charge dependence
    for j, pls in enumerate((("U", "V"), ("W",))):
        A = ax[1, 1 + j]
        for det, c in zip(DETS, ("tab:purple", "tab:brown", "tab:cyan")):
            d = {(r["sel"], r["plane"]): r for r in read(os.path.join(F, "47_tail_%s_anatomy.tsv" % det))}
            s = {r["plane"]: r for r in read(os.path.join(F, "47_tail_sim_%s_anatomy.tsv" % det))
                 if r["sel"] == "S1_gauss"}
            for src, tab, ls in (("data", d, "-"), ("sim", s, "--")):
                ys = []
                for k in range(10):
                    v = [tab[key]["dec%d_t2p" % k] for key in tab
                         if (key[1] if src == "data" else key) in pls
                         and (src == "sim" or key[0] == "base")]
                    ys.append(np.nanmean(v) if v else np.nan)
                A.plot(range(10), ys, ls, color=c, label="%s %s" % (det.upper(), src))
        A.set_yscale("log"); A.set_xlabel("decile of profile charge")
        A.set_ylabel(">=2-wire share" if j == 0 else "")
        A.set_title("%s planes" % ("induction" if j == 0 else "collection"), fontsize=10)
        A.grid(alpha=0.3); A.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(a.out, dpi=125)
    print("->", a.out)


if __name__ == "__main__":
    main()
