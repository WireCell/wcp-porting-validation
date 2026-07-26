#!/usr/bin/env python3
"""Build the five SBND dQ/dx-vs-residual-range reference tables under the
free-power recombination model, and write them into the viewer's reference json.

The shipped SBND tables (`energy_loss/pion_travel/stopping_ave_dQ_dx_sbnd.root`,
mirrored into `sbnd/particle_dataset.jsonnet`) come from convert_field.C:

    dQ/dx = <R(dE/dx) * dE/dx> / W_ion * C,   R = ln(A + xi)/xi,
    xi = (B/rho E) dE/dx,  A = 0.93, B = 0.212, C = 0.85

Doc 55 section 7g fitted the curated stopping-track sample and found that form
cannot hold the proton shape: the free parameter has to sit on the POWER, not on
B.  The better description, best of a twelve-family zoo in both fitting planes,
is the same Box logarithm with

    R = ln(A + u)/u,   u = k (dE/dx / 2.1 MeV/cm)^p,   A = 0.93

That is what this script tabulates, for all five particles, on exactly
convert_field.C's grid and with exactly its bin-averaging recipe (recombination
applied pointwise on the fine dE/dx graph, THEN averaged over the 1 cm bin --
never the other way round: R is concave, so the two differ).

k, p and C are not hard-coded here.  They are re-fitted at run time by importing
`fit_recombination.py` and calling it in the residual-range plane, the same way
`plot_muon_proton_models.py` does, so the tables cannot drift away from the fit
they claim to come from.  The values that went into the committed json are
recorded in its `_meta` block.

Two deliberate carry-overs from convert_field.C, both electron-only:

  * everything above 15 cm is a CLAMP, not physics -- ele1.dat stops at 15.3 cm
    / 60 MeV (energy_loss/docs/energy_loss_overview.md sec 5);
  * no rise into the stopping end.  convert.C had two hand-set points there
    ("// hack .."); the 0.5 cm bin is instead held at the 1.5 cm value, so the
    curve goes flat into rr -> 0.  Toolkit LinterpFunction clamps below its
    first node, so no synthetic negative-rr anchor is needed.

What this does NOT touch, deliberately (doc 55 sec 9 item 1): the shipped
`stopping_ave_dQ_dx_sbnd.root` and `sbnd/particle_dataset.jsonnet`.  The tagger
still runs on the Box tables.  Both sets therefore live in the json: the five
canonical `*DeDx` keys carry the free-power expectation, the five `*DeDxBox`
keys carry the tables the running config actually uses.  Consumers that report
on the tagger's decision (the STM panel's curve, MIP_DQDX) must read the `*Box`
keys; consumers that show the physics expectation read the canonical ones.

Repro:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
  python3 dqdx_rr_sample/make_ref_tables.py \
      --json nusel_display/stm_ref_dqdx.json \
      -o dqdx_rr_sample/ref_tables_free_power.png
  python3 dqdx_rr_sample/make_ref_tables.py --dry-run     # print, write nothing
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import uproot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SBND_DQDX = os.path.join("/nfs/data/1/xqian/toolkit-dev/energy_loss/pion_travel",
                         "stopping_ave_dQ_dx_sbnd.root")

PARTS = ["muon", "electron", "pion", "kaon", "proton"]
KEY = {"muon": "MuonDeDx", "electron": "ElectronDeDx", "pion": "PionDeDx",
       "kaon": "KaonDeDx", "proton": "ProtonDeDx"}

# convert_field.C's grid and its sampling of it
NBIN, RR0, DRR, NSUB = 60, 0.5, 1.0, 10
ELECTRON_CLAMP = 15.0     # ele1.dat's last point; above this the table is a clamp

BOX_FUDGE = 0.85          # convert_field.C's C, i.e. the shipped tables' scale

# dataviz categorical slots 1-2 of the validated 3-slot subset; the "current"
# curve wears ink, not a hue, and the two are separated by dash pattern as well
INK, NEW = "#0b0b0b", "#2a78d6"
GRID = "#d9d7d2"


def load_fr():
    spec = importlib.util.spec_from_file_location(
        "fr", os.path.join(HERE, "fit_recombination.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def table(fr, graphs, part, name, p, C):
    """convert_field.C's recipe for one particle: pointwise R, then bin average.

    The sampling window is the full [rr-0.5, rr+0.5] with NO low clip -- the
    first bin really does sample 0.05 ... 0.95 cm, singularity and all, exactly
    as the macro does.  That is what makes the Box case reproduce the shipped
    table to 1e-3 rather than to 1e-1.
    """
    rr = RR0 + DRR * np.arange(NBIN)
    lo, hi = rr - DRR / 2.0, rr + DRR / 2.0
    frac = (np.arange(NSUB) + 0.5) / NSUB
    s = lo[:, None] + (hi - lo)[:, None] * frac[None, :]
    if part == "electron":
        s = np.minimum(s, ELECTRON_CLAMP)
    gx, gy = graphs[part]
    sub = np.interp(s, gx, gy)
    y = C * np.mean(fr.MODELS[name][4](sub, *p) * sub, axis=1) / fr.W_ION
    if part == "electron":
        y[0] = y[1]        # flat end: no rise into rr -> 0
    return rr, y


def check(part, y):
    """Cheap guards.  The Bragg end of the kaon and proton tables runs an
    empirical form with p > 1 past its fit domain, so bound it here by
    construction rather than trusting it."""
    assert np.all(np.isfinite(y)), f"{part}: non-finite entry"
    assert np.all(y > 0), f"{part}: non-positive entry"
    assert np.all(y < 1e6), f"{part}: entry above 1e6 e/cm"
    if part != "electron":
        # dE/dx falls monotonically with rr and R*dE/dx is monotone in dE/dx,
        # so the table must fall monotonically too
        assert np.all(np.diff(y) < 0), f"{part}: not monotone decreasing in rr"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(
        HERE, os.pardir, "nusel_display", "stm_ref_dqdx.json"))
    ap.add_argument("-o", "--out", help="figure path (optional)")
    ap.add_argument("--points", default=os.path.join(HERE, "sample_points.tsv"))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the tables, write neither json nor figure")
    args = ap.parse_args()

    fr = load_fr()
    graphs = fr.dedx_graphs(PARTS)

    # ---- the Box tables ARE the shipped ones, read straight off the ROOT ----
    # Not recomputed: the `*Box` keys claim to be the tables the running config
    # holds, so they must be those numbers, not a reimplementation that agrees
    # to 1e-3.  The reimplementation is the GATE, below -- if it stops
    # reproducing the shipped table then this script's recipe has drifted and
    # the free-power tables it builds with that same recipe are not trustworthy.
    f = uproot.open(SBND_DQDX)
    box = {part: np.asarray(f[part].values("y"), float) for part in PARTS}

    print(f"recipe check -- rebuilt Box tables (A=0.93, B=0.212, C={BOX_FUDGE}) "
          f"vs {os.path.basename(SBND_DQDX)}:")
    worst = 0.0
    for part in PARTS:
        _, y = table(fr, graphs, part, "box_fixed", [], BOX_FUDGE)
        rel = float(np.max(np.abs(y / box[part] - 1)))
        worst = max(worst, rel)
        print(f"  {part:>9s}: max relative deviation {rel:.2e}  "
              f"({'PASS' if rel < 2e-3 else 'FAIL'})")
    if worst >= 2e-3:
        raise SystemExit("regression FAILED -- refusing to write tables")

    # ---- the fit, in the residual-range plane -------------------------------
    d = fr.read_tsv(args.points)
    part, rr_d, dq, dx = d["particle"], d["rr"], d["dqdx"], d["dx"]
    tid = np.array([f"{int(a)}_{int(b)}" for a, b in zip(d["event"], d["block"])])
    sub = np.zeros((len(rr_d), fr.NSUB))
    for p in ("muon", "proton"):
        s = part == p
        sub[s] = fr.dedx_samples(graphs, p, rr_d[s], dx[s])
    de = np.mean(sub, axis=1)
    keep = (rr_d >= fr.RR_MIN) & (de <= fr.DEDX_MAX) & (dq > 0)
    rows = fr.bin_data_rr(part[keep], rr_d[keep], de[keep], dq[keep],
                          tid[keep], 0.03)
    res = fr.fit("box_p", rows, sub[keep])
    k, pw = [float(v) for v in res["p"]]
    C = float(res["C"])
    print(f"\nfree power, residual-range plane: k = {k:.4f}  p = {pw:.4f}  "
          f"C = {C:.4f}   chi2/ndf = {res['chi2ndf']:.2f}   "
          f"(C/{BOX_FUDGE} = {C/BOX_FUDGE:.3f})")
    print(f"fit domain: rr >= {fr.RR_MIN} cm, dE/dx <= {fr.DEDX_MAX} MeV/cm, "
          f"{len(rows)} bins from {len(set(tid[keep]))} tracks")

    # ---- the new tables -----------------------------------------------------
    new = {}
    for p in PARTS:
        rr, y = table(fr, graphs, p, "box_p", [k, pw], C)
        check(p, y)
        new[p] = y
    rr = RR0 + DRR * np.arange(NBIN)

    print("\n            rr=0.5 cm            rr=59.5 cm        peak/plateau")
    print("          box     free  ratio   box     free  ratio   box   free")
    for p in PARTS:
        b, n = box[p], new[p]
        print(f"  {p:>8s} {b[0]/1e3:7.1f} {n[0]/1e3:7.1f} {n[0]/b[0]:6.3f} "
              f"{b[-1]/1e3:7.1f} {n[-1]/1e3:7.1f} {n[-1]/b[-1]:6.3f} "
              f"{b[0]/b[-1]:6.2f} {n[0]/n[-1]:6.2f}")

    # dE/dx actually needed by each table's first bin, vs what the fit saw
    print(f"\n  fit constrained to dE/dx <= {fr.DEDX_MAX:g} MeV/cm.  Top dE/dx each "
          "table's innermost bin actually samples (at rr = 0.05 cm):")
    print("  " + ", ".join(
        f"{q} {np.max(np.interp(np.linspace(0.05, 0.95, NSUB), *graphs[q])):.0f}"
        for q in PARTS) + " MeV/cm  -- kaon and proton extrapolate ~2x, "
          "the rest are inside the domain (doc 55 sec 10.3)")

    # ---- write the json -----------------------------------------------------
    def block(y):
        return {"start": RR0, "step": DRR,
                "values": [float(f"{v:.6g}") for v in y]}

    out = {"_meta": {
        "what": "SBND stopping-particle dQ/dx (e/cm) vs residual range (cm).",
        "generator": "sbnd_xin/dqdx_rr_sample/make_ref_tables.py",
        "writeup": "sbnd_xin/docs/55_dqdx-vs-rr-three-bundles.md sec 7g, sec 10",
        "dedx_source": fr.STOPPING,
        "grid": f"{NBIN} bins, centres {RR0} .. {RR0 + DRR*(NBIN-1)} cm, "
                f"{NSUB}-point average of the pointwise dQ/dx inside each bin",
        "E_field_kVcm": fr.E_FIELD, "W_ion_MeV": fr.W_ION,
        "canonical_keys": {
            "model": "modified box, free power: R = ln(A + u)/u, "
                     "u = k (dEdx / 2.1 MeV/cm)^p",
            "A": fr.BOX_UB[0], "k": round(k, 6), "p": round(pw, 6),
            "C": round(C, 6), "chi2_per_ndf": round(float(res["chi2ndf"]), 4),
            "fit": "doc 55 sec 7g, residual-range plane, "
                   "dqdx_rr_sample/sample_points.tsv",
            "fit_domain": f"rr >= {fr.RR_MIN} cm, dE/dx <= {fr.DEDX_MAX} MeV/cm",
        },
        "Box_keys": {
            "model": "modified box: R = ln(A + xi)/xi, xi = (B/rho E) dEdx",
            "A": fr.BOX_UB[0], "B": fr.BOX_UB[1], "rho": fr.RHO, "C": BOX_FUDGE,
            "note": "the tables the running config uses, copied VERBATIM from "
                    "energy_loss/pion_travel/stopping_ave_dQ_dx_sbnd.root "
                    "(= sbnd/particle_dataset.jsonnet, generated by "
                    "convert_field.C).  This script's own implementation of "
                    f"that recipe reproduces them to {worst:.1e} relative, which "
                    "is the gate it applies before writing anything.  MIP_DQDX "
                    "and anything reporting on a TaggerCheckSTM decision must "
                    "read THESE keys.",
        },
        "electron": "above 15 cm the curve is a clamp (ele1.dat ends at "
                    "15.3 cm), not physics; the 0.5 cm bin is held at the "
                    "1.5 cm value so there is no rise into the stopping end.",
        "caveat": "the kaon and proton innermost bins sample dE/dx up to 50 "
                  "and 66 MeV/cm, ~2x beyond the fit domain; muon, pion and "
                  "electron stay inside it.",
    }}
    for p in PARTS:
        out[KEY[p]] = block(new[p])
    for p in PARTS:
        out[KEY[p] + "Box"] = block(box[p])

    if args.dry_run:
        print("\n--dry-run: nothing written")
    else:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)
            fh.write("\n")
        print(f"\nwrote {args.json}  ({len(out)-1} curves)")

    if args.out and not args.dry_run:
        figure(args.out, rr, box, new, rows, k, pw, C, res)


def figure(path, rr, box, new, rows, k, pw, C, res):
    """Small multiples: one column per particle, dQ/dx above, ratio below.

    Five hues would need a validated 5-slot palette; two series per panel need
    only the documented pair, and the ratio row is where the actual change --
    the Bragg end moving in opposite directions per particle -- is legible.
    """
    fig, axes = plt.subplots(2, 5, figsize=(16.5, 6.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.1, 1],
                                              hspace=0.10, wspace=0.28))
    for j, p in enumerate(PARTS):
        ax, rx = axes[0, j], axes[1, j]
        ax.plot(rr, box[p] / 1e3, color=INK, ls="--", lw=2.0,
                label="current (Box, C=0.85)")
        ax.plot(rr, new[p] / 1e3, color=NEW, ls="-", lw=2.0,
                label="free power")
        if p in ("muon", "proton"):
            sel = [r for r in rows if r["part"] == p]
            ax.errorbar([r["rr"] for r in sel],
                        [r["dqdx"] / 1e3 for r in sel],
                        yerr=[r["dqdx"] * r["sig"] / 1e3 for r in sel],
                        fmt="o", ms=5.5, color=NEW, mfc="white", mew=1.6,
                        ecolor=NEW, elinewidth=1.4, capsize=2.5, zorder=5,
                        label="curated sample")
        ax.set_title(p, fontsize=12, color="#0b0b0b")
        ax.set_ylim(0, max(box[p].max(), new[p].max()) / 1e3 * 1.08)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
            rx.spines[sp].set_visible(False)
        if j == 0:
            ax.set_ylabel("dQ/dx  [ke/cm]", fontsize=11)
            rx.set_ylabel("free power / current", fontsize=10)
            ax.legend(fontsize=8.5, frameon=False, loc="upper right")

        rx.axhline(1.0, color=INK, ls="--", lw=1.4)
        rx.plot(rr, new[p] / box[p], color=NEW, lw=2.0)
        rx.set_ylim(0.80, 1.20)
        rx.set_xlim(0, 60)
        rx.set_xlabel("residual range  [cm]", fontsize=10)
        rx.grid(True, color=GRID, lw=0.7)
        rx.set_axisbelow(True)
        rx.annotate(f"{new[p][0]/box[p][0]:.2f} at 0.5 cm\n"
                    f"{new[p][-1]/box[p][-1]:.2f} at 59.5 cm",
                    xy=(0.96, 0.06), xycoords="axes fraction", ha="right",
                    fontsize=8.5, color="#52514e")

    fig.suptitle("SBND stopping-particle dQ/dx reference tables at E = 0.5 kV/cm: "
                 "the shipped Modified-Box tables vs the free-power fit",
                 fontsize=13.5, y=0.985)
    fig.text(0.5, 0.925,
             f"free power:  R = ln(0.93 + u)/u,  u = k (dE/dx / 2.1)$^p$   "
             f"with  k = {k:.4f},  p = {pw:.3f},  C = {C:.4f}   "
             f"($\\chi^2$/ndf = {res['chi2ndf']:.2f}, doc 55 §7g)      "
             f"current:  R = ln(0.93 + $\\xi$)/$\\xi$,  "
             f"$\\xi$ = (0.212/$\\rho$E) dE/dx,  C = 0.85",
             ha="center", fontsize=9.5, color="#52514e")
    fig.subplots_adjust(top=0.87, bottom=0.10, left=0.055, right=0.99)
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
