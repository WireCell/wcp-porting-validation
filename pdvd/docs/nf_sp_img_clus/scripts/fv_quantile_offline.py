#!/usr/bin/env python3
"""Doc pdvd/43 -- what each candidate boundary does to the exit ends, offline.

For each boundary (today's flat 15 cm + cushion, the doc 41 sec 9 d50 surface + 3,
the exit-gap p80 / p90 surfaces + cushion) and each exit end of a long cluster
(fv_exit_census.py rows, side walls, readout-clipped ends excluded, gap < 40 cm):
is the end INSIDE the boundary -- i.e. would a tagger call that exit "contained"?
The per-end miss rate is the containment cost of the boundary, per drift half.

Two honesty checks the arms cannot give:
  - the exit sample includes non-exits (the flat floor of fv_quantile_surface.py),
    so the rate is quoted with the floor's expected share subtracted as well;
  - the surfaces are built on the same 99 events the arms grade, so the p80 / p90
    surfaces are ALSO rebuilt from run 039349 alone (71 events) and scored on runs
    039252 + 039253 (28 events), and vice versa: the cross-validated miss rate is
    the one to believe.

Also draws the four boundaries at true scale in the X-Y and X-Z planes.

Repro:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 docs/nf_sp_img_clus/scripts/fv_quantile_offline.py /home/xqian/tmp/doc43/exits_rows.json \
      --table /home/xqian/tmp/doc43/q_table.json --d50 docs/nf_sp_img_clus/figs/41_fv_surface.json \
      --out /home/xqian/tmp/doc43/off
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_curved_map import XW, YW, ZLO, ZHI, CATH, WALLS
from fv_curved_surface import inset as d50_inset, BOX_MARGIN
from fv_quantile_surface import (readout_clipped, quantile_table, knots_from_profile, BIN_EDGES,
                                 QUANTILES, GAP_MAX, JNAME)


def knot_inset(knots, xabs):
    """piecewise-linear inset at |x| from [|x|, inset] knots (anode -> cathode)."""
    xs = np.array([k[0] for k in knots])[::-1]; ys = np.array([k[1] for k in knots])[::-1]
    return float(np.interp(xabs, xs, ys))


class Boundary:
    def __init__(self, name, kind, cushion, knots=None, d50=None):
        self.name, self.kind, self.cushion, self.knots, self.d50 = name, kind, cushion, knots, d50

    def inset(self, wall, vol, xabs):
        if self.kind == "flat":
            return (BOX_MARGIN["y"] if wall[0] == "y" else BOX_MARGIN["z"])
        if self.kind == "d50":
            return float(d50_inset(self.d50["surface"][wall][vol]["fv"], xabs)) + self.cushion
        return knot_inset(self.knots[JNAME[(wall, vol)]], xabs) + self.cushion


def miss_rate(rows, B, half=None):
    """(n, misses, expected floor misses): the exit ends the boundary would call
    contained (gap > inset), and how many of those the flat non-exit floor
    accounts for -- per (wall, vol) the floor density from 40-150 cm times the
    part of the 0-40 cm window beyond the inset."""
    n = 0; miss = 0; fl = 0.0
    sel = [r for r in rows if r["wall"] in WALLS and not readout_clipped(r) and (not half or r["half"] == half)]
    for r in sel:
        if r["dmin"] >= GAP_MAX:
            continue
        vol = "bot" if r["x"] < 0 else "top"
        n += 1
        if max(r["dmin"], 0.0) > B.inset(r["wall"], vol, abs(r["x"])):
            miss += 1
    for w in WALLS:
        for vol in ("bot", "top"):
            g = np.array([r["dmin"] for r in sel if r["wall"] == w and (r["x"] < 0) == (vol == "bot")])
            if not len(g):
                continue
            dens = ((g >= 40.0) & (g < 150.0)).sum() / 110.0
            xs = np.array([abs(r["x"]) for r in sel if r["wall"] == w and (r["x"] < 0) == (vol == "bot") and r["dmin"] < GAP_MAX])
            if len(xs):
                fl += dens * float(np.mean([max(GAP_MAX - B.inset(w, vol, x), 0.0) for x in xs])) * (len(xs) / len(xs))
    return n, miss, fl


def build_knots(rows, q, rng):
    T = quantile_table(rows, BIN_EDGES, QUANTILES, 0, rng, (40.0, 150.0))
    return {JNAME[(w, v)]: knots_from_profile(T[w][v]["center"], T[w][v][f"q{q}_pav"]) for w in WALLS for v in ("bot", "top")}


def polygon_figure(bounds, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    xs = np.concatenate([np.linspace(-XW, -CATH, 200), np.linspace(CATH, XW, 200)])
    for ax, (plane, lo, hi, wlo, whi) in zip(axes, (("y", -YW, YW, "y-", "y+"), ("z", ZLO, ZHI, "z-", "z+"))):
        ax.add_patch(plt.Rectangle((-XW, lo), 2 * XW, hi - lo, fill=False, ec="k", lw=1.2, label="sensitive volume"))
        for B, c, ls in bounds:
            low = [lo + B.inset(wlo, "bot" if x < 0 else "top", abs(x)) for x in xs]
            up = [hi - B.inset(whi, "bot" if x < 0 else "top", abs(x)) for x in xs]
            ax.plot(xs, low, color=c, ls=ls, lw=1.6, label=B.name); ax.plot(xs, up, color=c, ls=ls, lw=1.6)
        ax.axvspan(-CATH, CATH, color="0.85", label="cathode")
        ax.set_xlabel("x [cm]  (bottom drift < 0, top drift > 0)"); ax.set_ylabel(f"{plane} [cm]")
        ax.set_title(f"X-{plane.upper()} plane, true scale: the tagger boundaries")
        ax.set_xlim(-XW - 10, XW + 10); ax.set_ylim(lo - 12, hi + 12)
        ax.legend(fontsize=7, loc="center")
    fig.suptitle("PDVD doc 43 -- flat 15 cm inset vs d50 surface vs exit-gap quantile surfaces (cushion included)")
    fig.tight_layout(); fig.savefig(path, dpi=130); print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows"); ap.add_argument("--table", required=True); ap.add_argument("--d50", required=True)
    ap.add_argument("--out", default="/home/xqian/tmp/doc43/off")
    a = ap.parse_args()
    rows = json.load(open(a.rows)); Q = json.load(open(a.table)); d50 = json.load(open(a.d50))
    bounds = [
        (Boundary("today: flat 15 + 2.5/3", "flat", 0.0), "tab:brown", "-."),
        (Boundary("d50 + 3 (doc 41, arm d41fvon)", "d50", 3.0, d50=d50), "tab:green", ":"),
        (Boundary("p80 + 3 (arm d43p80c3)", "knots", 3.0, knots=Q["knots"]["p80"]), "tab:blue", "-"),
        (Boundary("p90 + 3 (arm d43p90c3)", "knots", 3.0, knots=Q["knots"]["p90"]), "tab:red", "-"),
        (Boundary("p90 + 5 (arm d43p90c5)", "knots", 5.0, knots=Q["knots"]["p90"]), "tab:red", "--"),
    ]
    out = {"in_sample": {}, "cross_validated": {}}
    print("IN-SAMPLE per-end miss rate (exit end called contained), side walls, gap < 40, readout-clipped excluded")
    print("  (miss / n  rate ; floor-corrected rate = (miss - expected non-exit floor beyond the boundary) / n)")
    print(f"{'boundary':34s} {'all':>24s} {'anode half':>24s} {'cathode half':>24s}")
    fmt = lambda rec, k: f"{rec[k]['miss']:4d}/{rec[k]['n']:4d} {100*rec[k]['rate']:5.1f}% ({100*rec[k]['rate_fc']:4.1f}%)"
    for B, _, _ in bounds:
        rec = {}
        for h in (None, "anode", "cathode"):
            n, m, fl = miss_rate(rows, B, h)
            rec[h or "all"] = dict(n=n, miss=m, rate=m / max(n, 1), floor=fl, rate_fc=max(m - fl, 0) / max(n, 1))
        out["in_sample"][B.name] = rec
        print(f"{B.name:34s} " + " ".join(fmt(rec, k) for k in ("all", "anode", "cathode")))

    # cross-validation: build on one run set, score on the other
    rng = np.random.default_rng(7)
    A = [r for r in rows if r["run"] == "039349"]; Bset = [r for r in rows if r["run"] != "039349"]
    print(f"\nCROSS-VALIDATED: build on {len({(r['run'], r['idx']) for r in A})} events of 039349, score on "
          f"{len({(r['run'], r['idx']) for r in Bset})} events of 039252+039253, and the reverse")
    for q in (80, 90):
        for cush in ((3.0,) if q == 80 else (3.0, 5.0)):
            tot = {"all": [0, 0, 0.0], "anode": [0, 0, 0.0], "cathode": [0, 0, 0.0]}
            for build, score in ((A, Bset), (Bset, A)):
                Bq = Boundary(f"p{q} + {cush:g}", "knots", cush, knots=build_knots(build, q, rng))
                for h in (None, "anode", "cathode"):
                    n, m, fl = miss_rate(score, Bq, h); tot[h or "all"][0] += n; tot[h or "all"][1] += m; tot[h or "all"][2] += fl
            rec = {k: dict(n=v[0], miss=v[1], rate=v[1] / max(v[0], 1), floor=v[2], rate_fc=max(v[1] - v[2], 0) / max(v[0], 1)) for k, v in tot.items()}
            out["cross_validated"][f"p{q} + {cush:g}"] = rec
            print(f"{'p%d + %g (cross-validated)' % (q, cush):34s} " + " ".join(fmt(rec, k) for k in ("all", "anode", "cathode")))
    json.dump(out, open(a.out + "_miss.json", "w"), indent=1)
    polygon_figure(bounds, a.out + "_polygons.png")


if __name__ == "__main__":
    main()
