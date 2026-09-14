#!/usr/bin/env python3
"""doc 107 sec 5.5-5.6: vertex quality, numuCC / nueCC selection efficiency and
purity, and the stacked reconstructed-Enu spectra, from
products/d107/{candidates,truth}.tsv.

Definitions (owner's choices, 2026-09-14):
  FV        5 < |x| < 190, |y| < 190, 10 < z < 450 cm  (standard analysis FV,
            cathode excluded); true vertex = generator vertex from mc.json,
            reco vertex = T_tagger nu_x/y/z.
  match     a candidate's reco vertex within 5 cm of the true vertex (its nearest
            true interaction in the event).
  selection score cut (e.g. numu_score > 0.9 / nue_score > 7.0) AND reco vertex in FV.
  signal    true CC of that flavour with its true vertex in the FV and, with
            --edep-min E, true deposited energy Edep > E MeV (sec 5.6: E = 100).
  efficiency = true signal interactions with >= 1 matched selected candidate
               / true signal interactions.
  purity     = selected candidates matched to a signal interaction
               / selected candidates.
  Flavour names in mc.json are sign-blind (TensorSetLabeler pdg_name maps +-14 to
  "numu", +-12 to "nue"), so "numuCC" includes anti-numu CC.

Usage:
  python3 d107_selection.py products/d107 docs/107_sel                  # sec 5.5
  python3 d107_selection.py products/d107 docs/107_sel_edep100 \
      --edep-min 100 --cuts numu:0.9,nue:7.0,nue:4.0                    # sec 5.6
"""
import argparse
import collections
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("src", nargs="?", default="products/d107")
ap.add_argument("figdir", nargs="?", default="docs/107_sel")
ap.add_argument("--edep-min", type=float, default=0.0,
                help="signal also needs true Edep > this (MeV); 0 = no Edep requirement")
ap.add_argument("--cuts", default="numu:0.9,nue:7.0",
                help="comma list of flavour:score_cut")
args = ap.parse_args()
src, figdir, EDEP_MIN = args.src, args.figdir, args.edep_min
C = list(csv.DictReader(open(os.path.join(src, "candidates.tsv")), delimiter="\t"))
T = list(csv.DictReader(open(os.path.join(src, "truth.tsv")), delimiter="\t"))
f = float
MATCH_CM = 5.0
SCORE = {"numu": "numu_score", "nue": "nue_score"}
DEFAULT_CUT = {"numu": 0.9, "nue": 7.0}
CUTS = [(fl, float(c)) for fl, c in (x.split(":") for x in args.cuts.split(","))]
EDEP_TXT = f", Edep > {EDEP_MIN:g} MeV" if EDEP_MIN > 0 else ""


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
    # no Edep requirement at all when --edep-min is 0 (23 FV interactions have Edep = 0)
    t["edep_ok"] = EDEP_MIN <= 0 or f(t["Edep"]) > EDEP_MIN
    truth[(ev(t), t["idx"])] = t
by_event = collections.defaultdict(list)
for t in T:
    by_event[ev(t)].append(t)
for r in C:
    r["reco_fv"] = r["vertex_default"] == "0" and in_fv(f(r["nu_x"]), f(r["nu_y"]), f(r["nu_z"]))
    d = f(r["t_dist_cm"])
    r["matched"] = r["vertex_default"] == "0" and 0 <= d < MATCH_CM
    r["truth"] = truth.get((ev(r), r["t_idx"])) if r["matched"] else None

out = []
p = out.append
p("# doc 107 sec 5.5 -- selection numbers (d107_selection.py)" if EDEP_MIN <= 0 else
  f"# doc 107 sec 5.6 -- selection numbers, signal Edep > {EDEP_MIN:g} MeV (d107_selection.py)")
p(f"FV 5<|x|<190, |y|<190, 10<z<450 cm; match = reco vertex within {MATCH_CM} cm; "
  "intervals are 68% Wilson")
p(f"truth interactions {len(T)}, in FV {sum(t['fv'] for t in T)}; candidates {len(C)}, "
  f"reco vertex in FV {sum(r['reco_fv'] for r in C)}")

# ---- Q1: vertex quality for true interactions in the FV
p(f"\n## Q1 vertex within 5 cm, true vertex in FV{EDEP_TXT}")
fvT = [t for t in T if t["fv"] and t["edep_ok"]]
hit = lambda t: 0 <= f(t["min_cand_dist_cm"]) < MATCH_CM
p("all true nu in FV:               " + frac(sum(hit(t) for t in fvT), len(fvT)))
wc = [t for t in fvT if t["event_has_candidate"] == "1"]
p("  ... in events with a candidate: " + frac(sum(hit(t) for t in wc), len(wc)))
for cl in ("numuCC", "NC", "nueCC"):
    s = [t for t in fvT if t["cls"] == cl]
    p(f"  {cl:7s}:                       " + frac(sum(hit(t) for t in s), len(s)))
for lo, hi in [(0, 100), (100, 300), (300, 1e9)]:
    s = [t for t in fvT if lo <= f(t["Edep"]) < hi]
    if not s:
        continue
    p(f"  Edep [{lo},{hi if hi < 1e9 else 'inf'}) MeV:          "
      + frac(sum(hit(t) for t in s), len(s)))

LOW_EDEP = "same CC in FV, Edep <= {e:g} MeV"


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
    if not t["edep_ok"]:
        return LOW_EDEP
    return "signal"


summary = {}
for flav, cut in CUTS:
    col = SCORE[flav]
    sig = flav + "CC"
    p(f"\n## {sig}: {col} > {cut:g} and reco vertex in FV; signal = true {sig} in FV{EDEP_TXT}")
    sigT = [t for t in fvT if t["cls"] == sig]
    matched_any = set()
    matched_fv = set()
    matched_sel = set()
    for r in C:
        t = r["truth"]
        if r["matched"] and t is not None and t["fv"] and t["edep_ok"] and t["cls"] == sig:
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
    p(f"true {sig} in FV{EDEP_TXT}: {n}")
    p("  candidate vertex within 5 cm:              " + frac(k_any, n))
    p("  ... and reco vertex in FV:                 " + frac(k_fv, n))
    p(f"  ... and {col} > {cut:g} (EFFICIENCY):      " + frac(k_sel, n))
    p(f"  score-cut efficiency given a matched FV candidate: " + frac(k_sel, k_fv))
    sel = [r for r in C if r["reco_fv"] and f(r[col]) > cut]
    cats = collections.Counter(category(r, flav) for r in sel)
    p(f"selected candidates: {len(sel)} (in {len({ev(r) for r in sel})} events)")
    p("  PURITY (signal):                           " + frac(cats["signal"], len(sel)))
    for c, v in cats.most_common():
        if c != "signal":
            p(f"  background {c.format(e=EDEP_MIN):28s} {v:5d} ({100*v/max(1,len(sel)):.1f}%)")
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
    # sec 5.7: the same selection with NO true-vs-reco vertex requirement.
    #  event level : a signal interaction is found if its event has >= 1 selected
    #                candidate; a candidate is signal if its event holds >= 1 signal
    #                interaction (pileup/cosmics can be credited -> loose bound)
    #  nearest     : candidate -> its nearest true interaction at ANY distance
    sigkeys = set(keys)
    sigev = {k[0] for k in keys}
    selev = {ev(r) for r in sel}
    nearest = {(ev(r), r["t_idx"]) for r in sel if r["vertex_default"] == "0"}
    p("  -- no vertex requirement (sec 5.7) --")
    p("  event-level EFFICIENCY:                    " + frac(sum(k[0] in selev for k in keys), n))
    p("  event-level PURITY:                        " + frac(sum(ev(r) in sigev for r in sel), len(sel)))
    p("  nearest-truth (any distance) EFFICIENCY:   " + frac(sum(k in nearest for k in keys), n))
    p("  nearest-truth (any distance) PURITY:       "
      + frac(sum((ev(r), r["t_idx"]) in sigkeys for r in sel), len(sel)))
    ebk = collections.Counter()
    for r in sel:
        if ev(r) in sigev:
            continue
        infv = [t for t in by_event[ev(r)] if t["fv"]]
        if any(t["cls"] == sig for t in infv):
            ebk[f"{sig} in FV failing the Edep cut"] += 1
        elif any(t["ccnc"] == "CC" for t in infv):
            ebk["other-flavour CC in FV"] += 1
        elif infv:
            ebk["only NC in FV"] += 1
        else:
            ebk["no true nu in FV (out-of-FV nu / cosmic)"] += 1
    for c, v in ebk.most_common():
        p(f"    event-level background: event has {c:40s} {v:5d}")
    # what the event-level count gains over the 5 cm match: distance band to the
    # nearest true interaction, and whether that nearest interaction is signal
    gain = collections.Counter()
    for r in sel:
        if ev(r) not in sigev or category(r, flav) == "signal":
            continue
        d = f(r["t_dist_cm"])
        b = "<5 cm" if d < 5 else "5-20 cm" if d < 20 else "20-50 cm" if d < 50 else ">50 cm"
        gain[(b, "is" if (ev(r), r["t_idx"]) in sigkeys else "is not")] += 1
    for (b, s), v in sorted(gain.items()):
        p(f"    event-level gain over 5 cm match: nearest true vertex {b:8s} {s:6s} signal {v:5d}")
    summary[(flav, cut)] = (sel, cats)

txt = "\n".join(out) + "\n"
print(txt)
os.makedirs(figdir, exist_ok=True)
with open(os.path.join(figdir, "d107_selection.txt"), "w") as fo:
    fo.write(txt)

# ---- plots: stacked reco-Enu spectra, signal vs background categories
SURFACE, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
# categorical slots 1-6 of the validated reference palette, fixed order
ORDER = {
    "numu": ["signal", "no true vertex within 5 cm", "NC in FV", "nueCC in FV",
             "true vertex outside FV"],
    "nue": ["signal", "no true vertex within 5 cm", "NC in FV", "numuCC in FV",
            "true vertex outside FV"],
}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
LABEL = {"signal": "signal: true {s} in FV{x} (vertex within 5 cm)",
         "no true vertex within 5 cm": "no true vertex within 5 cm (cosmic / misplaced)",
         "NC in FV": "true NC in FV", "nueCC in FV": "true nueCC in FV",
         "numuCC in FV": "true numuCC in FV",
         "true vertex outside FV": "true vertex outside FV",
         LOW_EDEP: "true {s} in FV, Edep <= {e:g} MeV"}
BINS = {"numu": [i * 100 for i in range(31)], "nue": [i * 200 for i in range(16)]}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
for flav, cut in CUTS:
    col = SCORE[flav]
    sel, cats = summary[(flav, cut)]
    sig = flav + "CC"
    order = ORDER[flav] + ([LOW_EDEP] if EDEP_MIN > 0 else [])
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    data, labels, colors = [], [], []
    for cat, colr in zip(order, COLORS):
        e = [min(f(r["reco_Enu"]), BINS[flav][-1] - 1e-6) for r in sel if category(r, flav) == cat]
        data.append(e)
        labels.append(f"{LABEL[cat].format(s=sig, x=EDEP_TXT, e=EDEP_MIN)}  ({len(e)})")
        colors.append(colr)
    counts, _, _ = ax.hist(data, bins=BINS[flav], stacked=True, color=colors, label=labels,
                           edgecolor=SURFACE, linewidth=1.0)
    ax.set_xlim(BINS[flav][0], BINS[flav][-1])
    # headroom so the legend never sits on a bar
    ax.set_ylim(0, 1.75 * max(1.0, float(max(counts[-1]))))
    ax.set_xlabel("reconstructed neutrino energy kine_reco_Enu (MeV; last bin = overflow)", color=INK2)
    ax.set_ylabel(f"candidates / {BINS[flav][1]} MeV", color=INK2)
    k = cats["signal"]
    ax.set_title(f"{sig} selection: {col} > {cut:g}, reco vertex in FV -- "
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
    suffix = "" if cut == DEFAULT_CUT[flav] else f"_cut{cut:g}"
    fig.savefig(os.path.join(figdir, f"d107_enu_{flav}cc{suffix}.png"), facecolor=SURFACE)
    plt.close(fig)
print("wrote", figdir)
