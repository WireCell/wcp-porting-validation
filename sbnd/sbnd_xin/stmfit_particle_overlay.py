#!/usr/bin/env python3
"""Overlay several fitted STM dQ/dx-vs-residual-range tracks on the SBND
per-particle expectation curves.

Companion to `stmfit_showcase.py`, which does ONE fit against the muon curve
only.  This one takes several (work root, event, block) specs and overlays them
all on curves from the SBND reference json (`nusel_display/stm_ref_dqdx.json`),
plus the flat `mip_dqdx` MIP line the STM tagger's not-stopping hypothesis uses
(56000 e/cm for SBND, doc 48 section 6).

Since doc 55 section 10 that json holds TWO five-particle sets: the
Modified-Box tables the running config actually contains (`*DeDxBox`, doc 48)
and the free-power fit of doc 55 section 7g (`*DeDx`).  `--ref-set` picks;
the default is `box` so every ratio doc 55 sections 2-3 quotes reproduces.
`--particles` picks which of the five to draw (default muon + proton).

`stmfit_showcase.py` is left untouched -- doc 42's Repro block cites it with
specific arguments.

Repro:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
  python3 stmfit_particle_overlay.py -o pics/stmfit_dqdx_particle_overlay.png \\
      work-mcp1000b-d55ton:289343:90:'evt 289343 grp 5 main 9 (proton cand.)' \\
      work-mcp10-d55ton:285999:220:'evt 285999 grp 12 main 22 (muon)' \\
      work-mcp10-d55ton:286065:30:'evt 286065 grp 8 main 3 (muon)'

Block id = cluster_id * 10 + pass, and cluster_id is the bundle's `main_id`
column in `nusel_evt<ID>/nusel-evt<ID>.tsv` (the same number the nusel scan
viewer prints as "main N").

Caveat that governs every number this prints: MCP2025C reco1 is *data* with no
gain and no electron-lifetime calibration applied (doc 42 section 0), so the
absolute dQ/dx level is not calibrated.  The discriminating quantity is the
residual-range dependence, not the normalization.
"""
import argparse
import json
import os

import numpy as np
import uproot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.gridspec import GridSpec   # noqa: E402

# Profile bins, in cm of residual range from the candidate stopping end.
BINS = [(0, 2), (2, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40),
        (40, 60), (60, 100), (100, 1e9)]

# dataviz categorical slots 1-3 (blue / orange / aqua), the documented
# all-pairs-validated subset -- scatter needs the all-pairs pairlist.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]

# The reference curves wear ink, not a series hue, and are separated by dash
# pattern as well as shade so identity is never colour-alone.
REF_STYLE = {"MuonDeDx":   dict(color="#0b0b0b", ls="-",  lw=1.8),
             "ProtonDeDx": dict(color="#52514e", ls="--", lw=1.8),
             "PionDeDx":   dict(color="#7a7873", ls="-.", lw=1.5),
             "KaonDeDx":   dict(color="#9b9892", ls=(0, (4, 1, 1, 1)), lw=1.5),
             "ElectronDeDx": dict(color="#b8b5ae", ls=":", lw=1.5)}
REF_LABEL = {"MuonDeDx": "muon expectation (SBND)",
             "ProtonDeDx": "proton expectation (SBND)",
             "PionDeDx": "pion expectation (SBND)",
             "KaonDeDx": "kaon expectation (SBND)",
             "ElectronDeDx": "electron expectation (SBND)"}

# Which of the json's two five-curve sets to draw.  `box` is the DEFAULT and
# must stay so: doc 55 sections 2-3 (and doc 42) quote every ratio in their
# tables against the Modified-Box curves the running config holds, and a figure
# that silently switched to the free-power fit would invalidate those numbers
# without touching a single digit of them.  `fit` draws doc 55 section 7g's
# free-power tables; `both` draws all ten.
REF_SETS = {"box": lambda k: k.endswith("Box"),
            "fit": lambda k: not k.endswith("Box"),
            "both": lambda k: True}


def load_refs(path, which="box", only=None):
    """Return {key: (rr, dqdx)} for the selected LinterpFunction blocks.

    Keys beginning with `_` are metadata, not curves (the json's `_meta`
    provenance block), and are skipped.  `*Box` keys are returned under their
    canonical name so REF_STYLE/REF_LABEL and every caller keep working.
    """
    d = {k: v for k, v in json.load(open(path)).items() if not k.startswith("_")}
    keep = REF_SETS[which]
    if not any(k.endswith("Box") for k in d):
        # a pre-doc-55 json carries one set under the canonical names; every
        # --ref-set then means the same thing.  Keeps the section-4a repro
        # (`--ref` at an old revision of the json) working.
        keep = REF_SETS["both"]
    out = {}
    for k, v in d.items():
        if not keep(k):
            continue
        if only and k.replace("Box", "") not in only:
            continue
        x = v["start"] + v["step"] * np.arange(len(v["values"]))
        out[k.replace("Box", "")] = (x, np.array(v["values"], dtype=float))
    if not out:
        raise SystemExit(f"no curves in {path} for --ref-set {which} "
                         f"and --particles {sorted(only or [])}")
    return out


def load_track(work_root, event, block):
    """Decode one (cluster, pass) block's fitted dQ/dx and residual range."""
    fp = os.path.join(work_root, f"nusel_evt{event}", "tracking-stm.root")
    f = uproot.open(fp)
    t = f["T_rec_charge"].arrays(["x", "q", "nq", "rr", "ndf", "status",
                                 "reduced_chi2"], library="np")
    tr = f["Trun"].arrays(["dQdx_scale", "dQdx_offset"], library="np")
    m = t["ndf"] == block
    if not m.any():
        raise SystemExit(f"no block {block} in {fp}; have "
                         f"{sorted(set(t['ndf'].tolist()))}")
    # The converter packs dQ as q = dQ*scale + offset (the uBooNE track_fit
    # convention Bee's colour ramp expects); undo it before dividing by dx.
    dQ = (t["q"][m] - tr["dQdx_offset"][0]) / tr["dQdx_scale"][0]
    dx = t["nq"][m]
    return dict(fp=fp, event=event, block=block,
                rr=t["rr"][m],
                dqdx=np.where(dx > 0, dQ / np.maximum(dx, 1e-9), 0.0),
                dx=dx,
                status=int(t["status"][m][0]),
                chi2=float(np.median(t["reduced_chi2"][m])),
                npts=int(m.sum()))


def profile(rr, dqdx):
    """Binned median dQ/dx; returns (bin centres, medians, counts, edges)."""
    cen, med, cnt, edg = [], [], [], []
    for lo, hi in BINS:
        s = (rr >= lo) & (rr < hi)
        if not s.sum():
            continue
        top = min(hi, rr.max())
        cen.append(0.5 * (lo + top))
        med.append(np.median(dqdx[s]))
        cnt.append(int(s.sum()))
        edg.append((lo, hi))
    return np.array(cen), np.array(med), cnt, edg


def report(tk, refs, mip):
    """Print the binned profile with the ratio to every reference curve."""
    print(f"\n=== {tk['fp']}  block {tk['block']} "
          f"(cluster {tk['block'] // 10}, pass {tk['block'] % 10}) ===")
    print(f"  npts={tk['npts']}  fit path L={tk['rr'].max():.1f} cm  "
          f"median dx={np.median(tk['dx']):.3f} cm  "
          f"median reduced_chi2={tk['chi2']:.2f}  status={tk['status']}"
          f"  ({'accepted STM' if tk['status'] == 0 else 'not accepted'})")
    keys = list(refs)
    dom = min(refs[k][0].max() for k in keys)
    print(f"  reference domain: rr 0.5 - {dom:.1f} cm  "
          f"({', '.join(keys)});  flat MIP {mip / 1e3:.0f} ke/cm")
    hdr = f"  {'rr bin (cm)':>13s} {'n':>4s} {'fit ke/cm':>10s}"
    for k in keys:
        hdr += f" {k.replace('DeDx', ''):>9s} {'ratio':>6s}"
    hdr += f" {'/MIP':>6s}"
    print(hdr)
    for lo, hi in BINS:
        s = (tk['rr'] >= lo) & (tk['rr'] < hi)
        if not s.sum():
            continue
        fit = np.median(tk['dqdx'][s])
        hi_s = "inf" if hi > 1e8 else f"{hi:.0f}"
        row = f"  {lo:5.0f} - {hi_s:>5s} {s.sum():4d} {fit / 1e3:10.1f}"
        for k in keys:
            rx, rv = refs[k]
            if lo >= rx.max():
                # np.interp CLAMPS above the last node, which would manufacture
                # a ratio against a flat line; report out-of-domain instead.
                row += f" {'--':>9s} {'--':>6s}"
            else:
                ex = np.median(np.interp(tk['rr'][s], rx, rv))
                mark = "*" if hi > rx.max() else ""
                row += f" {ex / 1e3:9.1f} {fit / ex:5.2f}{mark:1s}"
        row += f" {fit / mip:6.2f}"
        print(row)
    print("  (* bin extends past the reference domain; ratio uses only the "
          "in-domain part of the curve)")
    # One summary number per curve: the median point-by-point ratio over the
    # part of the fit path the reference actually covers.  This is the quantity
    # that can be compared BETWEEN tracks in the same uncalibrated sample --
    # a common gain/lifetime offset cancels in the comparison of two such
    # ratios, which the absolute level does not.
    for k in keys:
        rx, rv = refs[k]
        s = (tk['rr'] >= rx.min()) & (tk['rr'] <= rx.max()) & (tk['dqdx'] > 0)
        if not s.sum():
            continue
        r = tk['dqdx'][s] / np.interp(tk['rr'][s], rx, rv)
        print(f"  median fit/{k.replace('DeDx', ''):<6s} over rr "
              f"{rx.min():.1f}-{rx.max():.1f} cm (n={s.sum():4d}): "
              f"{np.median(r):.2f}   [16-84%: {np.percentile(r, 16):.2f} - "
              f"{np.percentile(r, 84):.2f}]")


def main():
    ap = argparse.ArgumentParser(
        description="Overlay fitted STM dQ/dx vs residual range on the SBND "
                    "per-particle expectation curves.")
    ap.add_argument("specs", nargs="+",
                    help="WORK_ROOT:EVENT:BLOCK[:LABEL], block = cluster*10+pass")
    ap.add_argument("-o", "--out", help="output PNG")
    ap.add_argument("--ref", default="nusel_display/stm_ref_dqdx.json",
                    help="SBND per-particle dQ/dx json (default: %(default)s)")
    ap.add_argument("--mip", type=float, default=56000.0,
                    help="flat MIP dQ/dx in e/cm (default: %(default)s, "
                         "the SBND mip_dqdx of doc 48)")
    ap.add_argument("--xmax", type=float, default=70.0,
                    help="x range of the combined panel in cm (default: %(default)s)")
    ap.add_argument("--ref-set", choices=sorted(REF_SETS), default="box",
                    help="which recombination model's curves to draw: `box` = "
                         "the tables the config holds (default, and what doc 55 "
                         "sections 2-3 quote), `fit` = doc 55 section 7g's "
                         "free-power tables, `both`")
    ap.add_argument("--particles", nargs="*", default=["MuonDeDx", "ProtonDeDx"],
                    help="reference curves to draw (default: %(default)s)")
    args = ap.parse_args()

    refs = load_refs(args.ref, args.ref_set, set(args.particles))
    tracks, labels = [], []
    for sp in args.specs:
        parts = sp.split(":")
        if len(parts) < 3:
            raise SystemExit(f"bad spec {sp!r}, want WORK_ROOT:EVENT:BLOCK[:LABEL]")
        wr, ev, blk = parts[0], parts[1], int(parts[2])
        tracks.append(load_track(wr, ev, blk))
        labels.append(parts[3] if len(parts) > 3 else f"evt {ev} blk {blk}")

    print(f"reference json: {args.ref}  set: {args.ref_set}  "
          f"curves: {', '.join(refs)}")
    for tk in tracks:
        report(tk, refs, args.mip)

    if not args.out:
        return

    n = len(tracks)
    fig = plt.figure(figsize=(4.6 * max(n, 3), 8.2))
    gs = GridSpec(2, max(n, 3), figure=fig, height_ratios=[1.25, 1.0],
                  hspace=0.42, wspace=0.20)
    top = fig.add_subplot(gs[0, :])

    def draw_refs(ax, with_labels):
        for k, (rx, rv) in refs.items():
            ax.plot(rx, rv / 1e3, label=REF_LABEL.get(k, k) if with_labels else None,
                    zorder=2, **REF_STYLE.get(k, dict(color="#52514e", ls=":", lw=1.6)))
        ax.axhline(args.mip / 1e3, ls=":", color="#a3a29b", lw=1.4, zorder=1,
                   label=(f"flat MIP {args.mip / 1e3:.0f} ke/cm "
                          "(not-stopping ref)") if with_labels else None)

    # --- combined overlay: binned medians only, so three tracks stay legible.
    draw_refs(top, True)
    for i, (tk, lab) in enumerate(zip(tracks, labels)):
        cen, med, cnt, _ = profile(tk['rr'], tk['dqdx'])
        top.plot(cen, med / 1e3, MARKERS[i % len(MARKERS)] + "-",
                 color=SERIES[i % len(SERIES)], ms=7, lw=1.8, mew=1.4,
                 mec="white", zorder=3, label=lab)
    top.set_xlim(0, args.xmax)
    top.set_ylim(0, 290)
    top.set_xlabel("residual range from the candidate stopping end  [cm]")
    top.set_ylabel("dQ/dx  [ke/cm]")
    top.set_title("SBND fitted STM dQ/dx vs the per-particle expectation "
                  f"(binned medians; references tabulated to rr = "
                  f"{min(refs[k][0].max() for k in refs):.1f} cm)")
    top.legend(fontsize=9, loc="upper right", framealpha=0.95)
    top.grid(alpha=0.2, lw=0.6)

    # --- per-track panels: every fitted point, full fit path.
    for i, (tk, lab) in enumerate(zip(tracks, labels)):
        ax = fig.add_subplot(gs[1, i])
        draw_refs(ax, False)
        ax.plot(tk['rr'], tk['dqdx'] / 1e3, ".", ms=4,
                color=SERIES[i % len(SERIES)], alpha=0.45, zorder=3,
                label=f"fitted points (n={tk['npts']})")
        cen, med, cnt, _ = profile(tk['rr'], tk['dqdx'])
        ax.plot(cen, med / 1e3, MARKERS[i % len(MARKERS)] + "-",
                color=SERIES[i % len(SERIES)], ms=6.5, lw=1.8, mew=1.4,
                mec="white", zorder=4, label="binned median")
        ax.set_xlim(0, max(tk['rr'].max() * 1.02, 30))
        ax.set_ylim(0, 290)
        ax.set_xlabel("residual range  [cm]")
        if i == 0:
            ax.set_ylabel("dQ/dx  [ke/cm]")
        ax.set_title(f"{lab}\nL = {tk['rr'].max():.1f} cm, "
                     f"median $\\chi^2$/ndf = {tk['chi2']:.2f}", fontsize=9)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
        ax.grid(alpha=0.2, lw=0.6)

    fig.suptitle("MCP2025C reco1 DATA, uncalibrated (no gain, no electron "
                 "lifetime): the absolute level carries a common unknown "
                 "offset -- compare the tracks against each other, not to "
                 "unity", fontsize=9, color="#52514e", y=0.995)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
