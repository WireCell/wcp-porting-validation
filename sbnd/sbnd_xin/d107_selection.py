#!/usr/bin/env python3
"""doc 107 sec 8: vertex quality, numuCC / nueCC selection efficiency and purity,
and the stacked reconstructed-Enu spectra, from products/d107/{candidates,truth}.tsv.

Definitions (owner's choices, 2026-09-14):
  FV        5 < |x| < 190, |y| < 190, 10 < z < 450 cm  (standard analysis FV,
            cathode excluded); true vertex = generator vertex from mc.json,
            reco vertex = T_tagger nu_x/y/z.
  match     a candidate's reco vertex within 5 cm of the true vertex (its nearest
            true interaction in the event).
  selection score cut (numu_score > 0.9 / nue_score > 7.0) AND reco vertex in FV.
  signal    true CC of that flavour with its true vertex in the FV.
  efficiency = true signal interactions with >= 1 matched selected candidate
               / true signal interactions.
  purity     = selected candidates matched to a signal interaction
               / selected candidates.
  Flavour names in mc.json are sign-blind (TensorSetLabeler pdg_name maps +-14 to
  "numu", +-12 to "nue"), so "numuCC" includes anti-numu CC.

Usage:
  python3 d107_selection.py products/d107 docs/107_sel
"""
import collections
import csv
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else "products/d107"
figdir = sys.argv[2] if len(sys.argv) > 2 else "docs/107_sel"
C = list(csv.DictReader(open(os.path.join(src, "candidates.tsv")), delimiter="\t"))
T = list(csv.DictReader(open(os.path.join(src, "truth.tsv")), delimiter="\t"))
f = float
MATCH_CM = 5.0
CUTS = {"numu": ("numu_score", 0.9), "nue": ("nue_score", 7.0)}


def in_fv(x, y, z):
    return 5.0 < abs(x) < 190.0 and abs(y) < 190.0 and 10.0 < z < 450.0


def tclass(flav, ccnc):
    if ccnc == "NC":
        return "NC"
    return {"numu": "numuCC", "nue": "nueCC"}.get(flav, "CC-" + flav)


def wilson(k, n, z=1.0):
    """68% Wilson interval."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def frac(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100*k/n:.1f}% [{100*lo:.1f}, {100*hi:.1f}]" if n else f"{k}/0"


ev = lambda r: (r["run"], r["subrun"], r["event"])
truth = {}
for t in T:
    t["fv"] = in_fv(f(t["vx"]), f(t["vy"]), f(t["vz"]))
    t["cls"] = tclass(t["flav"], t["ccnc"])
    truth[(ev(t), t["idx"])] = t
for r in C:
    r["reco_fv"] = r["vertex_default"] == "0" and in_fv(f(r["nu_x"]), f(r["nu_y"]), f(r["nu_z"]))
    d = f(r["t_dist_cm"])
    r["matched"] = r["vertex_default"] == "0" and 0 <= d < MATCH_CM
    r["truth"] = truth.get((ev(r), r["t_idx"])) if r["matched"] else None

out = []
p = out.append
p("# doc 107 sec 8 -- selection numbers (d107_selection.py)")
p(f"FV 5<|x|<190, |y|<190, 10<z<450 cm; match = reco vertex within {MATCH_CM} cm; "
  "intervals are 68% Wilson")
p(f"truth interactions {len(T)}, in FV {sum(t['fv'] for t in T)}; candidates {len(C)}, "
  f"reco vertex in FV {sum(r['reco_fv'] for r in C)}")

# ---- Q1: vertex quality for true interactions in the FV
p("\n## Q1 vertex within 5 cm, true vertex in FV")
fvT = [t for t in T if t["fv"]]
hit = lambda t: 0 <= f(t["min_cand_dist_cm"]) < MATCH_CM
p("all true nu in FV:               " + frac(sum(hit(t) for t in fvT), len(fvT)))
wc = [t for t in fvT if t["event_has_candidate"] == "1"]
p("  ... in events with a candidate: " + frac(sum(hit(t) for t in wc), len(wc)))
for cl in ("numuCC", "NC", "nueCC"):
    s = [t for t in fvT if t["cls"] == cl]
    p(f"  {cl:7s}:                       " + frac(sum(hit(t) for t in s), len(s)))
for lo, hi in [(0, 100), (100, 300), (300, 1e9)]:
    s = [t for t in fvT if lo <= f(t["Edep"]) < hi]
    p(f"  Edep [{lo},{hi if hi < 1e9 else 'inf'}) MeV:          "
      + frac(sum(hit(t) for t in s), len(s)))


# ---- Q2-Q4: efficiency and purity
def category(r, flav):
    """Signal/background category of a selected candidate."""
    if not r["matched"]:
        return "no true vertex within 5 cm"
    t = r["truth"]
    if not t["fv"]:
        return "true vertex outside FV"
    if t["cls"] == "NC":
        return "NC in FV"
    if t["cls"] != flav + "CC":
        return ("nueCC" if flav == "numu" else "numuCC") + " in FV"
    return "signal"


summary = {}
for flav, (col, cut) in CUTS.items():
    sig = flav + "CC"
    p(f"\n## {sig}: {col} > {cut} and reco vertex in FV")
    sigT = [t for t in fvT if t["cls"] == sig]
    matched_any = set()
    matched_fv = set()
    matched_sel = set()
    for r in C:
        if r["matched"] and r["truth"] is not None and r["truth"]["fv"] and r["truth"]["cls"] == sig:
            key = (ev(r), r["t_idx"])
            matched_any.add(key)
            if r["reco_fv"]:
                matched_fv.add(key)
                if f(r[col]) > cut:
                    matched_sel.add(key)
    keys = [(ev(t), t["idx"]) for t in sigT]
    n = len(keys)
    k_any = sum(k in matched_any for k in keys)
    k_fv = sum(k in matched_fv for k in keys)
    k_sel = sum(k in matched_sel for k in keys)
    p(f"true {sig} in FV: {n}")
    p("  candidate vertex within 5 cm:              " + frac(k_any, n))
    p("  ... and reco vertex in FV:                 " + frac(k_fv, n))
    p(f"  ... and {col} > {cut} (EFFICIENCY):      " + frac(k_sel, n))
    p(f"  score-cut efficiency given a matched FV candidate: " + frac(k_sel, k_fv))
    sel = [r for r in C if r["reco_fv"] and f(r[col]) > cut]
    cats = collections.Counter(category(r, flav) for r in sel)
    p(f"selected candidates: {len(sel)} (in {len({ev(r) for r in sel})} events)")
    p("  PURITY (signal):                           " + frac(cats["signal"], len(sel)))
    for c, v in cats.most_common():
        if c != "signal":
            p(f"  background {c:28s} {v:5d} ({100*v/max(1,len(sel)):.1f}%)")
    # split the unmatched background by the distance to (and class of) the
    # nearest true interaction: 5-20 cm from a true signal-flavour CC is a
    # misplaced vertex on a real neutrino, > 50 cm is cosmic / other activity
    band = collections.Counter()
    for r in sel:
        if category(r, flav) != "no true vertex within 5 cm" or f(r["t_dist_cm"]) < 0:
            continue
        d = f(r["t_dist_cm"])
        cl = tclass(r["t_flav"], r["t_ccnc"])
        band[("5-20 cm" if d < 20 else "20-50 cm" if d < 50 else ">50 cm", cl)] += 1
    for (b, cl), v in sorted(band.items()):
        p(f"    unmatched: nearest true vertex {b:8s} is {cl:7s} {v:5d}")
    sel_all = [r for r in C if f(r[col]) > cut]
    p(f"  (no reco-FV requirement: {len(sel_all)} candidates, signal "
      + frac(sum(category(r, flav) == 'signal' for r in sel_all), len(sel_all)) + ")")
    summary[flav] = (sel, cats)

txt = "\n".join(out) + "\n"
print(txt)
os.makedirs(figdir, exist_ok=True)
with open(os.path.join(figdir, "d107_selection.txt"), "w") as fo:
    fo.write(txt)

# ---- plots: stacked reco-Enu spectra, signal vs background categories
SURFACE, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
# categorical slots 1-5 of the validated reference palette, fixed order
ORDER = {
    "numu": ["signal", "no true vertex within 5 cm", "NC in FV", "nueCC in FV",
             "true vertex outside FV"],
    "nue": ["signal", "no true vertex within 5 cm", "NC in FV", "numuCC in FV",
            "true vertex outside FV"],
}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
LABEL = {"signal": "signal: true {s} in FV (vertex within 5 cm)",
         "no true vertex within 5 cm": "no true vertex within 5 cm (cosmic / misplaced)",
         "NC in FV": "true NC in FV", "nueCC in FV": "true nueCC in FV",
         "numuCC in FV": "true numuCC in FV",
         "true vertex outside FV": "true vertex outside FV"}
BINS = {"numu": [i * 100 for i in range(31)], "nue": [i * 200 for i in range(16)]}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
for flav, (col, cut) in CUTS.items():
    sel, cats = summary[flav]
    sig = flav + "CC"
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    data, labels, colors = [], [], []
    for cat, colr in zip(ORDER[flav], COLORS):
        e = [min(f(r["reco_Enu"]), BINS[flav][-1] - 1e-6) for r in sel if category(r, flav) == cat]
        data.append(e)
        labels.append(f"{LABEL[cat].format(s=sig)}  ({len(e)})")
        colors.append(colr)
    ax.hist(data, bins=BINS[flav], stacked=True, color=colors, label=labels,
            edgecolor=SURFACE, linewidth=1.0)
    ax.set_xlim(BINS[flav][0], BINS[flav][-1])
    ax.set_xlabel("reconstructed neutrino energy kine_reco_Enu (MeV; last bin = overflow)", color=INK2)
    ax.set_ylabel(f"candidates / {BINS[flav][1]} MeV", color=INK2)
    k = cats["signal"]
    ax.set_title(f"{sig} selection: {col} > {cut}, reco vertex in FV -- "
                 f"{len(sel)} candidates, purity {100*k/max(1,len(sel)):.1f}%",
                 color=INK, fontsize=10, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2)
    leg = ax.legend(frameon=False, fontsize=8, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"d107_enu_{flav}cc.png"), facecolor=SURFACE)
    plt.close(fig)
print("wrote", figdir)
