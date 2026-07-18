#!/usr/bin/env python3
"""Figures + headline numbers for doc 25 (wall-XA usability study)."""
import csv
import os
import glob
import io
import json
import tarfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = os.environ.get("WALLXA_DIR", os.path.dirname(os.path.abspath(__file__)))
PICS = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/qlmatch/pics"
BASE = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work"
f = lambda r, k: float(r[k])

pairs = list(csv.DictReader(open(f"{SP}/wall_xa_flash_channel.tsv"), delimiter="\t"))
join = list(csv.DictReader(open(f"{SP}/wall_xa_ophit_join.tsv"), delimiter="\t"))
LIVE = [0, 1, 3, 12, 18, 19]
TOP = {0, 1, 3}
CATH = list(range(4, 12))

# ---------------- Fig 1: turn-on (the family never turns on) ----------------
def fam_curves(chs):
    exp, meas, cov = [], [], []
    for r in pairs:
        if int(r["ch"]) in chs:
            exp.append(f(r, "pred") * f(r, "r_cath"))
            meas.append(f(r, "meas"))
            cov.append(int(r["cov"]))
    return np.array(exp), np.array(meas), np.array(cov)

bins = np.array([5, 10, 20, 50, 100, 300, 3000])
bc = np.sqrt(bins[:-1] * bins[1:])
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for chs, lab, c in ((CATH, "cathode XA (control)", "tab:green"),
                    (LIVE, "wall XA (live 6)", "tab:red")):
    E, M, C = fam_curves(chs)
    det, covf = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (E >= lo) & (E < hi)
        det.append(100 * ((M[m] > 0.5) & (C[m] == 1)).mean() if m.sum() > 10 else np.nan)
        covf.append(100 * (C[m] == 1).mean() if m.sum() > 10 else np.nan)
    ax[0].plot(bc, det, "o-", color=c, label=lab)
    ax[1].plot(bc, covf, "o-", color=c, label=lab)
# ophit-level availability for the wall
EJ = np.array([f(r, "exp") for r in join])
AV = np.array([f(r, "avail") for r in join])
det = [100 * (AV[(EJ >= lo) & (EJ < hi)] > 0.5).mean()
       if ((EJ >= lo) & (EJ < hi)).sum() > 10 else np.nan
       for lo, hi in zip(bins[:-1], bins[1:])]
ax[0].plot(bc, det, "s--", color="tab:orange",
           label="wall XA, ophit level (booked anywhere)")
for a, t in zip(ax, ("detection: flash PE > 0.5 on the matched flash",
                     "readout coverage of the flash window (cov = 1)")):
    a.set_xscale("log")
    a.set_xlabel("expected PE on channel (library pred x cathode ruler)")
    a.set_ylabel("%")
    a.set_ylim(0, 104)
    a.set_title(t, fontsize=10)
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
fig.suptitle("PDVD wall (membrane) X-ARAPUCAs never turn on at flash level -- "
             "cathode control reaches ~100% by 20 PE", fontsize=11)
fig.tight_layout()
fig.savefig(f"{PICS}/25_wallxa_turnon.png", dpi=110)
print("wrote 25_wallxa_turnon.png")

# ---------------- Fig 2: per-channel summary ----------------
fig, ax = plt.subplots(figsize=(10, 5))
X = np.arange(len(LIVE))
covs, fdet, adet = [], [], []
for ch in LIVE:
    E, M, C = fam_curves([ch])
    m = E >= 20
    covs.append(100 * (C[m] == 1).mean())
    fdet.append(100 * ((M[m] > 0.5) & (C[m] == 1)).mean())
    mj = np.array([int(r["ch"]) == ch and f(r, "exp") >= 20 for r in join])
    adet.append(100 * (AV[mj] > 0.5).mean())
w = 0.27
ax.bar(X - w, covs, w, color="0.6", label="readout coverage (cov=1)")
ax.bar(X, adet, w, color="tab:orange", label="ophit exists at flash time (any booking)")
ax.bar(X + w, fdet, w, color="tab:red", label="PE booked on the matched flash")
ax.set_xticks(X)
ax.set_xticklabels([f"ch {c}\n{'top' if c in TOP else 'bottom'} wall"
                    + ("\n(half-ganged)" if c == 1 else "") for c in LIVE])
ax.set_ylabel("% of matched flashes with expected >= 20 PE")
ax.set_title("per-channel wall-XA availability, 1885 matched flashes / 120 events\n"
             "(ch 2 top: dim, masked;  ch 13 bottom: no WLS, masked -- not shown)",
             fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{PICS}/25_wallxa_per_channel.png", dpi=110)
print("wrote 25_wallxa_per_channel.png")

# ---------------- Fig 3: where the reconstructed PE goes + example ----------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
mm = (EJ >= 20) & (AV > 0)
PH = np.array([f(r, "pe_here") for r in join])[mm]
PO = np.array([f(r, "pe_other") for r in join])[mm]
PN = np.array([f(r, "pe_none") for r in join])[mm]
tot = AV[mm].sum()
ax[0].bar(["matched flash", "other flash", "no flash\n(unassigned)"],
          [100 * PH.sum() / tot, 100 * PO.sum() / tot, 100 * PN.sum() / tot],
          color=["tab:green", "tab:orange", "tab:red"])
ax[0].set_ylabel("% of reconstructed wall-XA ophit PE")
ax[0].set_title("where the ophit PE near matched-flash times ends up\n"
                f"({tot:.0f} PE in {mm.sum()} (flash,ch) cases, exp >= 20)", fontsize=10)
ax[0].grid(alpha=0.3, axis="y")
# example: evt49746 flash 81 ch18 -- 1451 PE booked +14 us late
d = json.load(open(f"{BASE}/039253_3_keep/calib-evt49746.json"))
off0 = d["trigger_offsets_us"][0]
with tarfile.open(f"{SP}/membrane-frames-39253-49746.tar.bz2") as tf:
    F = {nm.rsplit("_", 1)[0]: np.load(io.BytesIO(tf.extractfile(nm).read()))
         for nm in tf.getnames()}
Fd, Cd = F["frame_decon"], list(F["channels_decon"])
w = Fd[Cd.index(2060)] + Fd[Cd.index(2061)]
TICK = 0.016
fl81 = [x for x in d["flashes"] if x["gid"] == 81][0]
t81 = fl81["time"] - off0
i0, i1 = int((t81 - 20) / TICK), int((t81 + 40) / TICK)
tt = np.arange(i0, i1) * TICK - t81
ax[1].plot(tt, w[i0:i1], lw=0.9, color="k")
ax[1].axvline(0, color="tab:green", ls="--", lw=1.5, label="matched flash 81 (Q/L pair)")
ax[1].axvline(2881.59 - t81, color="tab:orange", ls="--", lw=1.5,
              label="flash 165: got the 1451 PE ophit")
ax[1].annotate("ophit peak time\n(booked here)", xy=(14.2, 10.0),
               fontsize=8, xytext=(35, -20), textcoords="offset points",
               arrowprops=dict(arrowstyle="->"))
ax[1].set_xlabel("t - matched flash time [us]")
ax[1].set_ylabel("decon amplitude [PE/tick]")
ax[1].set_title("evt 49746, opdet 18 (2060+2061): slow 16-us pulse starts AT the\n"
                "matched flash but peaks +14 us late -> whole snippet booked to the "
                "wrong flash", fontsize=9)
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{PICS}/25_wallxa_booking.png", dpi=110)
print("wrote 25_wallxa_booking.png")

# ---------------- Fig 4: response ratio quality ----------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
# flash-level ratio, wall vs cathode
for chs, lab, c in ((CATH, "cathode XA", "tab:green"), (LIVE, "wall XA", "tab:red")):
    E, M, C = fam_curves(chs)
    m = (E >= 20) & (C == 1)
    q = M[m] / E[m]
    ax[0].hist(np.log10(np.clip(q, 1e-3, 100)), bins=np.arange(-3, 2.05, 0.125),
               histtype="step", lw=1.8, density=True, color=c, label=lab)
ax[0].axvline(0, color="k", lw=0.8)
ax[0].set_xlabel("log10( measured / expected )   [flash level]")
ax[0].set_ylabel("density")
ax[0].set_title("flash-level response: wall XAs bimodal (zero spike + broad "
                "over-unity lobe)", fontsize=10)
ax[0].legend(fontsize=9)
ax[0].grid(alpha=0.3)
# ophit-level ratio vs distance to wall
YW = {0: +1, 1: -1, 3: -1, 12: +1, 18: +1, 19: -1}
D, Q = [], []
pmap = {(r["run"], r["idx"], r["gid"], r["ch"]): r for r in pairs}
for r in join:
    if f(r, "exp") < 20 or f(r, "avail") < 0.5:
        continue
    p = pmap.get((r["run"], r["idx"], r["gid"], r["ch"]))
    if p is None:
        continue
    D.append(417.6 - YW[int(r["ch"])] * f(p, "bary_y"))
    Q.append(f(r, "avail") / f(r, "exp"))
D, Q = np.array(D), np.array(Q)
db = [(0, 150), (150, 250), (250, 350), (350, 500), (500, 900)]
med = [np.median(Q[(D >= lo) & (D < hi)]) for lo, hi in db]
q16 = [np.percentile(Q[(D >= lo) & (D < hi)], 16) for lo, hi in db]
q84 = [np.percentile(Q[(D >= lo) & (D < hi)], 84) for lo, hi in db]
x = [np.mean(b) for b in db]
ax[1].errorbar(x, med, yerr=[np.array(med) - q16, np.array(q84) - med],
               fmt="o-", color="tab:red", capsize=4)
ax[1].axhline(1, color="k", lw=0.8)
ax[1].set_xlabel("source charge-barycenter distance from the XA's wall [cm]")
ax[1].set_ylabel("ophit-available PE / expected (median, 16-84%)")
ax[1].set_title("library shape error: response rises x2-3 with distance\n"
                "(v5 library falls off too fast for the wall XAs)", fontsize=10)
ax[1].set_yscale("log")
ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{PICS}/25_wallxa_ratio.png", dpi=110)
print("wrote 25_wallxa_ratio.png")

# ---------------- headline numbers ----------------
E, M, C = fam_curves(LIVE)
m = E >= 20
print(f"\nwall pooled exp>=20: n={m.sum()}, cov {100*(C[m]==1).mean():.0f}%, "
      f"flash-det {100*((M[m]>0.5)&(C[m]==1)).mean():.0f}%")
mj = EJ >= 20
print(f"ophit-avail-det {100*(AV[mj]>0.5).mean():.0f}%")
mm2 = mj & (AV > 0.5)
print(f"avail/exp median {np.median(AV[mm2]/EJ[mm2]):.2f} "
      f"16-84% {np.percentile(AV[mm2]/EJ[mm2],16):.2f}-{np.percentile(AV[mm2]/EJ[mm2],84):.2f}")
