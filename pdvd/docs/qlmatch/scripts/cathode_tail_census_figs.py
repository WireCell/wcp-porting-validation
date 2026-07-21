#!/usr/bin/env python3
"""Final figures + numbers for doc 16 &11 (120-event cathode-tail census)."""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = os.path.dirname(os.path.abspath(__file__))
PICS = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/qlmatch/pics"

pairs = list(csv.DictReader(open(f"{SP}/cathode_tail_pairs.tsv"), delimiter="\t"))
anat = list(csv.DictReader(open(f"{SP}/cathode_tail_anatomy.tsv"), delimiter="\t"))
f = lambda r, k: float(r[k])


def classify(r):
    if f(r, "pad_contig") > 0:
        return "pad"
    if f(r, "nticks") - f(r, "tick_face") < 300:
        return "edge"
    return "supported"


# ---------------- Fig 1: R distribution ----------------
R = np.array([f(r, "R") for r in pairs])
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ax[0].hist(R, bins=np.arange(-25, 25.5, 1), color="tab:blue", alpha=0.8)
ax[0].axvline(-8.27, color="red", ls="--", lw=1.5,
              label="evt298567 top97+bot139 (R=-8.3)")
ax[0].set_xlabel("sum-test residual R = X$_{top\\,end}$ + X$_{bot\\,end}$ [cm]  (T0-free)")
ax[0].set_ylabel("crosser pairs")
ax[0].set_title(f"{len(R)} cathode-crossing pairs, 120 events\n"
                f"median {np.median(R):+.2f} cm, rms {R.std():.2f} cm -- no global systematic",
                fontsize=10)
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)
ax[0].set_yscale("log")
# per-run
for run, c in (("039252", "tab:blue"), ("039253", "tab:green"), ("039349", "tab:red")):
    Rr = [f(r, "R") for r in pairs if r["run"] == run]
    ax[1].hist(Rr, bins=np.arange(-15, 15.5, 1.5), histtype="step", lw=1.6,
               color=c, label=f"{run} (n={len(Rr)}, med {np.median(Rr):+.2f})", density=True)
ax[1].set_xlabel("R [cm]")
ax[1].set_ylabel("density")
ax[1].set_title("per run: consistent, centered at 0", fontsize=10)
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{PICS}/16_census_sumtest_R.png", dpi=110)
print("wrote 16_census_sumtest_R.png")

# ---------------- Fig 2: contiguous penetration ----------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
pen_all, pen_sup, pen_pad = [], [], []
for r in anat:
    p = f(r, "pen_contig")
    pen_all.append(p)
    (pen_pad if classify(r) != "supported" else pen_sup).append(p)
bins = np.arange(-4, 13, 0.5)
ax[0].hist(pen_sup, bins=bins, color="tab:blue", alpha=0.8,
           label="ADC-supported (readout covers the end)")
ax[0].hist(pen_pad, bins=bins, color="tab:orange", alpha=0.8,
           label="face at/past BDE readout end (pad-artifact zone)")
ax[0].axvspan(0, 6, color="0.5", alpha=0.25, label="inside the 6 cm cathode slab")
ax[0].axvline(5.33, color="red", ls="--", lw=1.4, label="evt298567 top 97 (5.3 cm)")
ax[0].set_xlabel("contiguous penetration past own cathode face [cm]")
ax[0].set_ylabel("crosser halves")
ax[0].set_yscale("log")
ax[0].set_title("junk-robust, drift-connected endpoint of each crosser half", fontsize=10)
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)
# side split of supported tails
sup = [r for r in anat if classify(r) == "supported"]
pt = [f(r, "pen_contig") for r in sup if r["side"] == "T"]
pb = [f(r, "pen_contig") for r in sup if r["side"] == "B"]
ax[1].hist([pt, pb], bins=np.arange(-4, 13, 1.0), stacked=False,
           color=["tab:purple", "tab:green"], label=[f"top halves (n={len(pt)})",
                                                     f"bottom halves (n={len(pb)})"],
           histtype="step", lw=1.8)
ax[1].axvspan(0, 6, color="0.5", alpha=0.25)
ax[1].set_xlabel("contiguous penetration [cm] (ADC-supported only)")
ax[1].set_yscale("log")
ax[1].set_title("both drift volumes show the tail population", fontsize=10)
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{PICS}/16_census_penetration.png", dpi=110)
print("wrote 16_census_penetration.png")

# ---------------- Fig 3: junction map ----------------
fig, ax = plt.subplots(figsize=(8.5, 7))
yz_all = np.array([(f(r, "end_y"), f(r, "end_z")) for r in pairs
                   if r["t0src"] != "none"])
ax.scatter(yz_all[:, 0], yz_all[:, 1], s=8, c="0.75", label="all crosser junctions")
# supported contiguous tails >3
pmap = {}
for r in pairs:
    pmap[(r["run"], r["idx"], r["uid_t"], str(r["gid"]))] = r
    pmap[(r["run"], r["idx"], r["uid_b"], str(r["gid"]))] = r
done = set()
for r in anat:
    if f(r, "pen_contig") > 3 and classify(r) == "supported":
        p = pmap.get((r["run"], r["idx"], r["uid"], r["gid"]))
        if p:
            k = (r["run"], r["idx"], r["uid"])
            if k in done:
                continue
            done.add(k)
            ax.scatter([f(p, "end_y")], [f(p, "end_z")], s=90, marker="^",
                       c="red", zorder=5)
ax.scatter([], [], s=90, marker="^", c="red", label="in-cathode tail >3 cm (ADC-supported)")
for r in anat:
    if f(r, "det_pen") > 3 and f(r, "det_djunc") < 9:
        p = pmap.get((r["run"], r["idx"], r["uid"], r["gid"]))
        if p:
            ax.scatter([f(p, "end_y")], [f(p, "end_z")], s=80, marker="s",
                       facecolors="none", edgecolors="tab:orange", lw=1.8, zorder=5)
ax.scatter([], [], s=80, marker="s", facecolors="none", edgecolors="tab:orange",
           label="detached late clump at junction (>3 cm)")
for yy in (336.4, -336.4):
    ax.axvline(yy, color="k", lw=1)
for zz in (1.9, 297.7):
    ax.axhline(zz, color="k", lw=1)
ax.set_xlabel("junction y [cm]")
ax.set_ylabel("junction z [cm]")
ax.set_title("crossing-point (y,z) of tails vs all crossers:\n"
             "tails cluster toward the detector edges (field-cage proximity)", fontsize=10)
ax.legend(fontsize=8, loc="upper center")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{PICS}/16_census_junction_map.png", dpi=110)
print("wrote 16_census_junction_map.png")

# ---------------- numbers ----------------
sup3 = [r for r in anat if f(r, "pen_contig") > 3 and classify(r) == "supported"]
print(f"\nsupported contig>3: {len(sup3)} sides "
      f"(T {sum(1 for r in sup3 if r['side']=='T')} / B {sum(1 for r in sup3 if r['side']=='B')}) "
      f"in {len(set((r['run'],r['idx']) for r in sup3))} events")


def edge_dist(y, z):
    return min(336.4 - abs(y), z - 1.9, 297.7 - z)


ed_t, ed_all = [], []
for r in sup3:
    p = pmap.get((r["run"], r["idx"], r["uid"], r["gid"]))
    if p:
        ed_t.append(edge_dist(f(p, "end_y"), f(p, "end_z")))
for r in pairs:
    if r["t0src"] != "none":
        ed_all.append(edge_dist(f(r, "end_y"), f(r, "end_z")))
ed_t, ed_all = np.array(ed_t), np.array(ed_all)
print(f"edge dist: supported tails median {np.median(ed_t):.1f} vs all {np.median(ed_all):.1f}; "
      f"<30 cm: {100*np.mean(ed_t<30):.0f}% vs {100*np.mean(ed_all<30):.0f}%")
