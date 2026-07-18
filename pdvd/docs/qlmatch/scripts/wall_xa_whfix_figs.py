#!/usr/bin/env python3
"""Doc 25 §7 figure + numbers: wall-XA detection and library matching with
the wide-hit booking fixed (wide_hit_mode='start' on mem+pmt, _whfix round).

Reads wall_xa_flash_channel.tsv (doc 25 §1 sample) + wall_xa_whfix_join.tsv.
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = os.environ.get("WALLXA_DIR", os.path.dirname(os.path.abspath(__file__)))
PICS = os.path.join(os.path.dirname(SP), "pics") if os.path.basename(SP) == "scripts" \
    else "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/qlmatch/pics"

pairs = list(csv.DictReader(open(f"{SP}/wall_xa_flash_channel.tsv"), delimiter="\t"))
new = list(csv.DictReader(open(f"{SP}/wall_xa_whfix_join.tsv"), delimiter="\t"))
f = lambda r, k: float(r[k])
nm = {(r["run"], r["idx"], r["gid"], r["ch"]): r for r in new if r["matched"] == "1"}
LIVE = [0, 1, 3, 12, 18, 19]
CATH = list(range(4, 12))
YW = {0: +1, 1: -1, 3: -1, 12: +1, 18: +1, 19: -1}


def collect(chs):
    E, Mo, Mn, Cn, D = [], [], [], [], []
    for r in pairs:
        ch = int(r["ch"])
        if ch not in chs:
            continue
        exp = f(r, "pred") * f(r, "r_cath")
        if exp < 5:
            continue
        k = (r["run"], r["idx"], r["gid"], r["ch"])
        if k not in nm:
            continue
        rn = nm[k]
        E.append(exp)
        Mo.append(f(r, "meas"))
        Mn.append(f(rn, "meas"))
        Cn.append(f(rn, "cov"))
        D.append(417.6 - YW.get(ch, 1) * f(r, "bary_y"))
    return map(np.array, (E, Mo, Mn, Cn, D))


E, Mo, Mn, Cn, D = collect(LIVE)
Ec, Moc, Mnc, _, _ = collect(CATH)
bins = np.array([5, 10, 20, 50, 100, 300, 3000])
bc = np.sqrt(bins[:-1] * bins[1:])
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))


def curve(EE, MM, cond=None):
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (EE >= lo) & (EE < hi)
        if cond is not None:
            m &= cond
        out.append(100 * (MM[m] > 0.5).mean() if m.sum() > 10 else np.nan)
    return out


ax[0].plot(bc, curve(Ec, Mnc), "o-", color="tab:green", label="cathode XA (control)")
ax[0].plot(bc, curve(E, Mo), "o-", color="tab:red",
           label="wall XA, BEFORE (peak-time booking)")
ax[0].plot(bc, curve(E, Mn), "s-", color="tab:blue",
           label="wall XA, AFTER (start-time booking)")
ax[0].plot(bc, curve(E, Mn, Cn >= 1), "^--", color="tab:purple",
           label="AFTER, given readout covered (cov=1)")
ax[0].set_xscale("log")
ax[0].set_ylim(0, 104)
ax[0].set_xlabel("expected PE (library pred x cathode ruler)")
ax[0].set_ylabel("detection: flash PE > 0.5 on the matched flash [%]")
ax[0].set_title("wide_hit_mode='start': wall-XA detection 28->47%,\n"
                "and 83% wherever the self-trigger provided readout", fontsize=10)
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)

db = [(0, 150), (150, 250), (250, 350), (350, 500), (500, 900)]
x = [np.mean(b) for b in db]
for MM, lab, c in ((Mo, "BEFORE", "tab:red"), (Mn, "AFTER", "tab:blue")):
    med, q16, q84 = [], [], []
    for lo, hi in db:
        m = (D >= lo) & (D < hi) & (MM > 0.5) & (E >= 20)
        q = MM[m] / E[m]
        med.append(np.median(q))
        q16.append(np.percentile(q, 16))
        q84.append(np.percentile(q, 84))
    ax[1].errorbar(x, med, yerr=[np.array(med) - q16, np.array(q84) - np.array(med)],
                   fmt="o-", color=c, capsize=4, label=lab)
ax[1].axhline(1, color="k", lw=0.8)
ax[1].set_yscale("log")
ax[1].set_xlabel("source charge-barycenter distance from the XA's wall [cm]")
ax[1].set_ylabel("measured / expected (responding, median, 16-84%)")
ax[1].set_title("with booking fixed, the residual is a clean monotone\n"
                "library shape error: x0.7 near wall -> x2.1 far", fontsize=10)
ax[1].legend(fontsize=9)
ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{PICS}/25_wallxa_whfix.png", dpi=110)
print("wrote 25_wallxa_whfix.png")

m20 = E >= 20
cov1 = Cn >= 1
print(f"wall exp>=20 n={m20.sum()}: det {100*(Mo[m20]>0.5).mean():.1f}% -> "
      f"{100*(Mn[m20]>0.5).mean():.1f}%; det|covered {100*(Mn[m20&cov1]>0.5).mean():.0f}%")
print("det|covered by exp bin:", [f"{v:.0f}" for v in curve(E, Mn, Cn >= 1) if v == v])
