#!/usr/bin/env python3
"""Doc pdvd/43 -- a fiducial surface from EXIT-GAP QUANTILES.

Doc 41 sec 13 showed that the d50 (charge-density median) surface reproduces the
MEDIAN closest approach of long tracks to every wall, while a tagger tests the
TAIL: a boundary at the median calls one wall approach in six "contained".  So
the tagger's boundary is built here from the endpoint distribution itself --
per wall, per drift volume, per drift bin -- at a stated quantile (p80 / p90),
and the cushion goes on top through the taggers' fv_tolerance, exactly as
MicroBooNE probes its uncushioned SCB surface with a tolerance band (doc 41
sec 9.1).

INPUT: the rows of fv_exit_census.py -- one per END of a long cluster, assigned
to the boundary surface its own outward direction reaches first, with the
signed perpendicular gap to that surface.  (Doc 41 sec 13's "closest approach
within a cap" sample is contaminated by tracks passing a wall on their way out
through another one; raising its cap from 25 to 40 cm moved the anode-half p90
from 15 to 26 cm.  fv_curved_zapproach.py rows are still accepted, for that
comparison.)

SELECTION and BACKGROUND:
  - ends at a readout-window edge are excluded (d_late < 5 cm, or d_early < 5 cm
    with |x| < 330: the tick-0 plane lying in the bulk) -- those are truncated by
    the readout, not by imaging (doc 41 sec 11.2);
  - the ends assigned to a wall are exits (the pile-up at 0-30 cm) plus a floor
    of non-exits -- stopping muons whose stop end happens to point at that wall,
    fragment tips of over-clustered objects -- that is flat in the gap.  The
    floor density is taken per (wall, volume, bin) from --bg-range (40-150 cm by
    default), subtracted from the 0-40 cm histogram, and the quantiles are read
    off the excess.  The unsubtracted quantiles are kept alongside.

REGULARIZATION: with 4 bins per volume and 20-90 exits per bin a per-bin p90 is
the 2nd-3rd largest value, so each profile is forced NON-INCREASING toward the
anode (weighted pool-adjacent-violators, weights = n): the space-charge
displacement can only grow with drift and the instrumental stop-short tail (doc
41 sec 13.2) does not depend on x.  Raw and regularized numbers are both written.

Outputs (--out prefix):
  <out>_table.json        per (wall, vol, bin): n, floor, median/p80/p90 (+ bootstrap sigma), raw and PAV
  <out>_profiles.jsonnet  the knot lists in curved_fiducial.jsonnet's `profile` form
  <out>_surface.png       the eight profiles against the d50 surface and today's flat inset
  <out>_tail.png          the exit-gap distribution per wall, with the anode control

Repro:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 docs/nf_sp_img_clus/scripts/fv_exit_census.py --tag d41fvoff --out /home/xqian/tmp/doc43/exits
  python3 docs/nf_sp_img_clus/scripts/fv_quantile_surface.py /home/xqian/tmp/doc43/exits_rows.json \
      --d50 docs/nf_sp_img_clus/figs/41_fv_surface.json --out /home/xqian/tmp/doc43/q
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_curved_map import XW, YW, ZLO, ZHI, CATH, WALLS
from fv_curved_surface import inset as d50_inset, BOX_MARGIN

VOLS = ("bot", "top")
BIN_EDGES = [3.0, 80.0, 160.0, 240.0, 340.0]
QUANTILES = (50, 80, 90)
GAP_MAX = 40.0            # the exit window; quantiles are read inside it
JNAME = {("y+", "bot"): "yp_bot", ("y+", "top"): "yp_top",
         ("y-", "bot"): "ym_bot", ("y-", "top"): "ym_top",
         ("z-", "bot"): "zm_bot", ("z-", "top"): "zm_top",
         ("z+", "bot"): "zp_bot", ("z+", "top"): "zp_top"}


def readout_clipped(r):
    if "d_late" not in r:
        return False
    return r["d_late"] < 5 or (r["d_early"] < 5 and abs(r["x"]) < 330)


def pav_nonincreasing(y, w):
    """Weighted pool-adjacent-violators: the closest (weighted L2) sequence to y that
    is non-increasing in index order.  y, w indexed cathode -> anode."""
    blocks = [[float(v), float(wt), 1] for v, wt in zip(y, w)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0] - 1e-12:
            v = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (blocks[i][1] + blocks[i + 1][1])
            blocks[i] = [v, blocks[i][1] + blocks[i + 1][1], blocks[i][2] + blocks[i + 1][2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for v, _, c in blocks:
        out += [v] * c
    return np.array(out)


def excess_quantiles(g, quantiles, floor_per_cm, gap_max=GAP_MAX):
    """Quantiles of the gap distribution inside [0, gap_max] after subtracting a flat
    floor.  Negative gaps (charge past the wall) count at 0.  Returns {q: value}."""
    gg = np.clip(g[g < gap_max], 0.0, None)
    if len(gg) == 0:
        return {q: float("nan") for q in quantiles}
    edges = np.arange(0.0, gap_max + 0.5, 0.5)
    h, _ = np.histogram(gg, edges)
    ex = np.clip(h - floor_per_cm * 0.5, 0.0, None)
    tot = ex.sum()
    if tot <= 0:
        return {q: float("nan") for q in quantiles}
    cum = np.cumsum(ex) / tot
    out = {}
    for q in quantiles:
        i = int(np.searchsorted(cum, q / 100.0))
        i = min(i, len(edges) - 2)
        # linear within the bin
        c0 = cum[i - 1] if i > 0 else 0.0
        frac = (q / 100.0 - c0) / max(cum[i] - c0, 1e-12)
        out[q] = float(edges[i] + 0.5 * np.clip(frac, 0, 1))
    return out


def quantile_table(rows, edges, quantiles, boot, rng, bg_range, mincos=0.0, subtract=True):
    T = {}
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    for w in WALLS:
        T[w] = {}
        for vol in VOLS:
            sg = -1 if vol == "bot" else 1
            rec = {"n": [], "n_exit_window": [], "floor_per_cm": [], "center": centers, "edges": edges}
            for q in quantiles:
                rec[f"q{q}"] = []; rec[f"q{q}_err"] = []; rec[f"q{q}_raw"] = []
            for i in range(len(edges) - 1):
                sel = [r for r in rows if r["wall"] == w and np.sign(r["x"]) == sg
                       and edges[i] <= abs(r["x"]) < edges[i + 1] and r["cos"] >= mincos
                       and not readout_clipped(r)]
                g = np.array([r["dmin"] for r in sel], float)
                inwin = g < GAP_MAX
                nbg = int(((g >= bg_range[0]) & (g < bg_range[1])).sum())
                floor = (nbg / (bg_range[1] - bg_range[0])) if subtract else 0.0
                rec["n"].append(int(len(g))); rec["n_exit_window"].append(int(inwin.sum()))
                rec["floor_per_cm"].append(float(floor))
                est = excess_quantiles(g, quantiles, floor)
                raw = excess_quantiles(g, quantiles, 0.0)
                bs = {q: [] for q in quantiles}
                if len(g):
                    for _ in range(boot):
                        gb = rng.choice(g, len(g))
                        nb = int(((gb >= bg_range[0]) & (gb < bg_range[1])).sum())
                        fb = (nb / (bg_range[1] - bg_range[0])) if subtract else 0.0
                        eb = excess_quantiles(gb, quantiles, fb)
                        for q in quantiles:
                            bs[q].append(eb[q])
                for q in quantiles:
                    rec[f"q{q}"].append(est[q]); rec[f"q{q}_raw"].append(raw[q])
                    rec[f"q{q}_err"].append(float(np.nanstd(bs[q])) if bs[q] else float("nan"))
            for q in quantiles:
                rawv = np.array(rec[f"q{q}"], float); n = np.array(rec["n_exit_window"], float)
                ok = np.isfinite(rawv)
                sm = rawv.copy()
                if ok.sum() >= 2:
                    sm[ok] = pav_nonincreasing(rawv[ok], np.maximum(n[ok], 1))
                rec[f"q{q}_pav"] = [float(np.clip(x, 0.0, None)) if np.isfinite(x) else float("nan") for x in sm]
            T[w][vol] = rec
    return T


def knots_from_profile(centers, values):
    """[|x|, inset] knots anode face -> cathode face, held flat outside the bin centres."""
    c = list(centers); v = [round(float(x), 2) for x in values]
    k = [[XW, v[-1]]]
    for ci, vi in zip(reversed(c), reversed(v)):
        k.append([round(float(ci), 2), vi])
    k.append([CATH, v[0]])
    return k


def profiles_jsonnet(T, quantiles, meta):
    lines = ["// PDVD exit-gap QUANTILE fiducial profiles, MEASURED -- doc pdvd/43.",
             "//",
             "// Each entry is one wall of one drift volume as [|x|, inset] knots in cm from the",
             "// anode face (339.91) to the cathode face (3.0), the form curved_fiducial.jsonnet's",
             "// `profile` argument takes.  The inset at a bin is the p<q> of the perpendicular gap",
             "// between the end of a long (> 2 m) Q/L-matched cosmic track and the wall its own",
             "// direction exits through, per 80 cm of drift, after subtracting the flat non-exit",
             "// floor, regularized non-increasing toward the anode, cushion 0 -- the cushion is",
             "// the taggers' fv_tolerance (pr.jsonnet curved_fv_margin_y/z).",
             "//",
             "// GENERATED by docs/nf_sp_img_clus/scripts/fv_quantile_surface.py (wcp-porting-img);",
             "// do not edit by hand -- re-run the doc 43 Repro block.",
             "// " + meta,
             "{"]
    for q in quantiles:
        if q == 50:
            continue
        lines.append(f"  p{q}: {{")
        for w in WALLS:
            for vol in VOLS:
                rec = T[w][vol]
                k = knots_from_profile(rec["center"], rec[f"q{q}_pav"])
                ks = ", ".join(f"[{a:.2f}, {b:.2f}]" for a, b in k)
                lines.append(f"    {JNAME[(w, vol)]}: [{ks}],")
        lines.append("  },")
    lines.append("}")
    return "\n".join(lines) + "\n"


def surface_figure(rows, T, T_cos, d50, quantiles, cushion, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True, sharey=True)
    col = {50: "0.35", 80: "tab:blue", 90: "tab:red"}
    for j, w in enumerate(WALLS):
        for i, vol in enumerate(VOLS):
            ax = axes[i, j]
            sg = -1 if vol == "bot" else 1
            pts = [(abs(r["x"]), r["dmin"]) for r in rows
                   if r["wall"] == w and np.sign(r["x"]) == sg and not readout_clipped(r)]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=5, color="0.75", alpha=0.6,
                       label=f"exit gap of an end, n={len(pts)}", zorder=1)
            rec = T[w][vol]; c = np.array(rec["center"])
            for q in quantiles:
                off = {50: -6, 80: 0, 90: 6}[q]
                ax.errorbar(c + off, rec[f"q{q}"], yerr=rec[f"q{q}_err"], fmt="o", ms=4, color=col[q],
                            lw=1, capsize=2, label=f"p{q} per bin (floor subtracted)", zorder=3)
                if q != 50:
                    k = knots_from_profile(rec["center"], rec[f"q{q}_pav"])
                    ax.plot([p[0] for p in k], [p[1] for p in k], color=col[q], lw=1.6,
                            label=f"p{q} profile (non-increasing)", zorder=4)
                    ax.plot([p[0] for p in k], [p[1] + cushion for p in k], color=col[q], lw=1.2, ls="--",
                            label=f"p{q} + {cushion:g} cm cushion = tagger boundary", zorder=4)
            rc = T_cos[w][vol]
            ax.plot(c + 9, rc["q90"], marker="v", ms=4, ls="none", color="tab:red", mfc="none",
                    label="p90, |cos| to the wall >= 0.3 only", zorder=3)
            xx = np.linspace(CATH, XW, 300)
            s = d50["surface"][w][vol]["fv"]
            ax.plot(xx, d50_inset(s, xx), color="tab:green", lw=1.4, ls=":", label="doc 41 sec 9 d50 surface")
            ax.axhline(BOX_MARGIN["y"] if w[0] == "y" else BOX_MARGIN["z"], color="tab:brown", ls="-.", lw=1.2,
                       label="today: flat 15 cm + cushion")
            ax.axhline(0, color="k", lw=0.8)
            ax.set_title(f"{w} wall, {vol} volume")
            ax.set_xlim(0, XW); ax.set_ylim(-4, 42)
            if i == 1:
                ax.set_xlabel("|x| of the end [cm]  (cathode 3, anode 339.9)")
            if j == 0:
                ax.set_ylabel("gap between the end and its exit wall [cm]")
    axes[0, 0].legend(fontsize=6.5, loc="upper right", ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("wrote", path)


def tail_figure(rows, bg_range, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    bins = np.arange(-10, 151, 2.0)
    for ax, half in zip(axes[:2], ("anode", "cathode")):
        for w, c in zip(WALLS, ("tab:orange", "tab:purple", "tab:cyan", "tab:olive")):
            v = np.array([r["dmin"] for r in rows if r["half"] == half and r["wall"] == w and not readout_clipped(r)])
            fl = ((v >= bg_range[0]) & (v < bg_range[1])).sum() / (bg_range[1] - bg_range[0]) * 2.0
            ax.hist(v, bins=bins, histtype="step", color=c, lw=1.4,
                    label=f"{w}: n={len(v)}, floor {fl:.1f}/bin, in 0-40: {(v < 40).sum()}")
        ax.axvspan(bg_range[0], bg_range[1], color="0.9", zorder=0, label="floor estimate range")
        ax.axvline(GAP_MAX, color="k", lw=0.8, ls=":")
        ax.set_yscale("log"); ax.set_ylim(0.5, None)
        ax.set_title(f"{half} half (|x| {'> 170' if half == 'anode' else '< 170'} cm), side walls")
        ax.set_xlabel("gap between the end and its exit wall [cm]"); ax.set_ylabel("ends / 2 cm")
        ax.legend(fontsize=7)
    ax = axes[2]
    v = np.array([r["dmin"] for r in rows if r["wall"] == "anode" and not readout_clipped(r)])
    ax.hist(v, bins=bins, histtype="step", color="k", lw=1.4, label=f"anode faces: n={len(v)}, median {np.median(v):.1f}")
    ax.set_yscale("log"); ax.set_ylim(0.5, None)
    ax.set_title("CONTROL: ends exiting through an anode face")
    ax.set_xlabel("gap between the end and the anode plane [cm]  (< 0 = past it)"); ax.set_ylabel("ends / 2 cm")
    ax.legend(fontsize=7)
    fig.suptitle("PDVD doc 43 -- the exit-gap distribution a tagger has to cover (long Q/L-matched cosmic tracks, 99 events)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", help="fv_exit_census.py rows (or fv_curved_zapproach.py rows)")
    ap.add_argument("--d50", required=True, help="figs/41_fv_surface.json (the doc 41 sec 9 surface, for the figure)")
    ap.add_argument("--out", default="/home/xqian/tmp/doc43/q")
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--cushion", type=float, default=3.0)
    ap.add_argument("--mincos", type=float, default=0.3, help="the wall-pointing systematic check")
    ap.add_argument("--bg-range", type=float, nargs=2, default=(40.0, 150.0))
    ap.add_argument("--no-subtract", action="store_true")
    ap.add_argument("--title", default="PDVD doc 43 -- the exit-gap quantile surface (long Q/L-matched cosmic tracks, 99 events)")
    a = ap.parse_args()

    rows = json.load(open(a.rows))
    d50 = json.load(open(a.d50))
    rng = np.random.default_rng(43)
    sub = not a.no_subtract
    T = quantile_table(rows, BIN_EDGES, QUANTILES, a.boot, rng, a.bg_range, subtract=sub)
    T_cos = quantile_table(rows, BIN_EDGES, QUANTILES, 0, rng, a.bg_range, mincos=a.mincos, subtract=sub)

    nev = len({(r["run"], r["idx"]) for r in rows})
    ncl = len({(r["run"], r["idx"], r["cid"]) for r in rows})
    nro = sum(readout_clipped(r) for r in rows)
    meta = (f"{len(rows)} ends of {ncl} long clusters in {nev} events ({nro} at a readout edge excluded), "
            f"exit window {GAP_MAX:.0f} cm, floor from {a.bg_range[0]:.0f}-{a.bg_range[1]:.0f} cm"
            f"{'' if sub else ' (NOT subtracted)'}, bins {BIN_EDGES}, {a.boot} bootstraps.")
    print(meta)

    print(f"\n{'wall':4s} {'vol':3s} {'|x| bin':9s} {'n':>4s} {'n<40':>4s} {'floor':>5s}  {'median':>6s}  "
          f"{'p80':>12s}  {'p90':>12s}  {'p80raw':>6s} {'p90raw':>6s}  {'p80pav':>6s} {'p90pav':>6s}  {'p90 cos>=.3':>11s}  {'d50 surf':>8s}")
    for w in WALLS:
        for vol in VOLS:
            rec = T[w][vol]; rc = T_cos[w][vol]
            for i, c in enumerate(rec["center"]):
                lo, hi = rec["edges"][i], rec["edges"][i + 1]
                print(f"{w:4s} {vol:3s} {lo:3.0f}-{hi:3.0f}   {rec['n'][i]:4d} {rec['n_exit_window'][i]:4d} "
                      f"{rec['floor_per_cm'][i]*40:5.1f}  {rec['q50'][i]:6.1f}  "
                      f"{rec['q80'][i]:5.1f} +- {rec['q80_err'][i]:4.1f}  {rec['q90'][i]:5.1f} +- {rec['q90_err'][i]:4.1f}  "
                      f"{rec['q80_raw'][i]:6.1f} {rec['q90_raw'][i]:6.1f}  "
                      f"{rec['q80_pav'][i]:6.1f} {rec['q90_pav'][i]:6.1f}  {rc['q90'][i]:11.1f}  "
                      f"{float(d50_inset(d50['surface'][w][vol]['fv'], c)):8.1f}")
            print()

    pooled = {}
    for half in ("anode", "cathode"):
        g = np.array([r["dmin"] for r in rows if r["half"] == half and r["wall"] in WALLS and not readout_clipped(r)])
        nbg = int(((g >= a.bg_range[0]) & (g < a.bg_range[1])).sum())
        fl = nbg / (a.bg_range[1] - a.bg_range[0])
        e = excess_quantiles(g, (50, 80, 90, 95), fl)
        r0 = excess_quantiles(g, (50, 80, 90, 95), 0.0)
        pooled[half] = dict(n=int(len(g)), n_exit_window=int((g < GAP_MAX).sum()), floor_in_window=float(fl * GAP_MAX),
                            p50=e[50], p80=e[80], p90=e[90], p95=e[95],
                            p80_raw=r0[80], p90_raw=r0[90],
                            frac_gt8=float(np.mean(np.clip(g[g < GAP_MAX], 0, None) > 8)),
                            frac_gt18=float(np.mean(np.clip(g[g < GAP_MAX], 0, None) > 18)))
        print(f"{half} half, side walls: n {len(g)} (in window {pooled[half]['n_exit_window']}, floor {fl*GAP_MAX:.0f})  "
              f"p50 {e[50]:.1f}  p80 {e[80]:.1f}  p90 {e[90]:.1f}  p95 {e[95]:.1f}   (raw p80 {r0[80]:.1f} p90 {r0[90]:.1f})  "
              f"in-window >8 cm {100*pooled[half]['frac_gt8']:.1f} %  >18 cm {100*pooled[half]['frac_gt18']:.1f} %")
    ga = np.array([r["dmin"] for r in rows if r["wall"] == "anode" and not readout_clipped(r)])
    if len(ga):
        pooled["anode_control"] = dict(n=int(len(ga)), p50=float(np.median(ga)), p80=float(np.percentile(ga, 80)),
                                       p90=float(np.percentile(ga, 90)), frac_gt8=float(np.mean(ga > 8)))
        print(f"anode control: n {len(ga)}  median {np.median(ga):.1f}  p80 {np.percentile(ga, 80):.1f}  "
              f"p90 {np.percentile(ga, 90):.1f}  >8 cm {100*np.mean(ga > 8):.1f} %")

    knots = {f"p{q}": {JNAME[(w, vol)]: knots_from_profile(T[w][vol]["center"], T[w][vol][f"q{q}_pav"])
                       for w in WALLS for vol in VOLS} for q in QUANTILES if q != 50}
    json.dump(dict(meta=meta, bin_edges=BIN_EDGES, quantiles=list(QUANTILES), cushion_cm=a.cushion,
                   gap_max=GAP_MAX, bg_range=list(a.bg_range), subtract=sub,
                   table=T, table_wallpointing=T_cos, pooled=pooled, knots=knots),
              open(a.out + "_table.json", "w"), indent=1)
    open(a.out + "_profiles.jsonnet", "w").write(profiles_jsonnet(T, QUANTILES, meta))
    print("wrote", a.out + "_table.json", a.out + "_profiles.jsonnet")
    surface_figure(rows, T, T_cos, d50, QUANTILES, a.cushion, a.out + "_surface.png", a.title)
    tail_figure(rows, a.bg_range, a.out + "_tail.png")


if __name__ == "__main__":
    main()
