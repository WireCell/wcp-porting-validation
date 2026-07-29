#!/usr/bin/env python3
"""Collect the clean stopping-track dQ/dx-vs-residual-range sample out of the
30-event MCP2025C `d55ton` set.

Sweeps every `T_rec_charge` (cluster, pass) block in
`work-{mcp10,mcp1000,mcp1000b}-d55ton/nusel_evt*/tracking-stm.root`, keeps only
the blocks whose dQ/dx-vs-rr *shape* is highly consistent with one of the SBND
stopping-particle curves, and writes the surviving tracks out as plain text so
the sample can be re-used without going back to the ROOT files.

Selection (all cuts are printed with the per-block numbers by `--verbose`, so a
rejected track can be traced):

  1. >= 40 fitted points in the block.
  2. >= 6 populated profile bins (>= 3 points each), the profile reaching
     rr < 2 cm at the stopping end and rr >= 22 cm at the far end.  Without both
     ends there is no Bragg peak *and* no plateau to constrain a shape.
  3. Bragg contrast = median dQ/dx(rr < 2) / median dQ/dx(20 <= rr < 40) >= 2.0.
     This is the cut that removes through-going passes; they sit at 0.4-1.5.
  4. median reduced_chi2 <= 2.5 (trajectory-fit quality).
  5. free-scale shape residual <= 10 % against at least one of the four
     hadron/muon curves.  For each hypothesis the binned profile is divided by
     the reference, the best single scale k is taken out (geometric mean), and
     the RMS fractional deviation of what is left is the shape residual.  The
     electron curve is EXCLUDED as a hypothesis: it is 44 identical entries
     above 15 cm (doc 48 section 5), so it is a flat line that every
     through-going track matches trivially.

Particle assignment is by the free scale k, not by the shape residual --
the muon/pion/kaon/proton shapes are nearly degenerate over 0.5-60 cm (they
differ by < 1.5 % in shape residual on most tracks) and separate almost entirely
in overall scale.  The muon population defines the charge scale empirically
(see the doc); a track needing k_muon ~ 1.9 is proton-like.

Usage:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
  python3 dqdx_rr_sample/collect_dqdx_rr_sample.py            # writes into dqdx_rr_sample/
  python3 dqdx_rr_sample/collect_dqdx_rr_sample.py --verbose   # every block, kept or not

Doc 55 section 13 re-collects the same sample out of the 4.0/8.8 diffusion arm.
`--roots`/`--events` point the sweep somewhere else, `--suffix` keeps the output
beside the published files instead of over them, and `--force-list` replays an
existing `sample_index.tsv`'s (event, block) set with the cuts NOT applied, so a
diffusion A/B runs on identical tracks:

  python3 dqdx_rr_sample/collect_dqdx_rr_sample.py --suffix _d66 \\
      --roots work-stmcamp-d66new --events dqdx_rr_sample/d55ton_events.txt \\
      --force-list dqdx_rr_sample/sample_index.tsv
"""
import argparse
import glob
import os

import numpy as np
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
SBND_DQDX = ("/nfs/data/1/xqian/toolkit-dev/energy_loss/pion_travel/"
             "stopping_ave_dQ_dx_sbnd.root")
ROOTS = ["work-mcp10-d55ton", "work-mcp1000-d55ton", "work-mcp1000b-d55ton"]
HYPS = ["muon", "pion", "kaon", "proton"]
BINS = [(0, 2), (2, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40), (40, 60)]

# Cuts
MIN_NPTS = 40
MIN_BINS = 6
MAX_RRMIN = 2.0
MIN_RRMAX = 22.0
MIN_CONTRAST = 2.0
MAX_CHI2 = 2.5
MAX_SHAPE_RMS = 0.10

# Assignment windows on the free scale (see module docstring)
MUON_K = (0.85, 1.25)
PROTON_K = (0.85, 1.35)

# SBND geometry / drift, cfg/pgrapher/experiment/sbnd/clus.jsonnet:22,82-83
ANODE_ABS_X = 202.05      # cm, W plane
DRIFT_SPEED = 0.1563      # cm/us


def load_refs():
    f = uproot.open(SBND_DQDX)
    return {n: (np.asarray(f[n].values("x"), float),
                np.asarray(f[n].values("y"), float)) for n in HYPS}


def read_index(path):
    """(event, block) -> particle out of a previously written sample_index.tsv."""
    out = {}
    for line in open(path):
        if line.startswith("#") or line.startswith("particle\t"):
            continue
        f = line.rstrip("\n").split("\t")
        out[(f[2], int(f[3]))] = f[0]
    return out


def iter_blocks(roots=None, events=None):
    for wr in (roots or ROOTS):
        for fp in sorted(glob.glob(os.path.join(wr, "nusel_evt*", "tracking-stm.root"))):
            ev = os.path.basename(os.path.dirname(fp)).replace("nusel_evt", "")
            if events is not None and ev not in events:
                continue
            f = uproot.open(fp)
            t = f["T_rec_charge"].arrays(["x", "y", "z", "q", "nq", "rr", "ndf",
                                          "status", "reduced_chi2"], library="np")
            tr = f["Trun"].arrays(["dQdx_scale", "dQdx_offset"], library="np")
            for b in sorted(set(t["ndf"].tolist())):
                m = t["ndf"] == b
                dQ = (t["q"][m] - tr["dQdx_offset"][0]) / tr["dQdx_scale"][0]
                dx = t["nq"][m]
                yield dict(root=wr, event=ev, block=int(b),
                           rr=t["rr"][m], dx=dx,
                           dqdx=np.where(dx > 0, dQ / np.maximum(dx, 1e-9), 0.0),
                           x=t["x"][m], y=t["y"][m], z=t["z"][m],
                           status=int(t["status"][m][0]), npts=int(m.sum()),
                           chi2=float(np.median(t["reduced_chi2"][m])))


def profile(rr, dq):
    cen, med, cnt = [], [], []
    for lo, hi in BINS:
        s = (rr >= lo) & (rr < hi) & (dq > 0)
        if s.sum() >= 3:
            cen.append(float(np.median(rr[s])))
            med.append(float(np.median(dq[s])))
            cnt.append(int(s.sum()))
    return np.array(cen), np.array(med), np.array(cnt)


def shape_scores(cen, med, refs):
    out = {}
    for n, (rx, rv) in refs.items():
        s = (cen >= rx.min()) & (cen <= rx.max())
        if s.sum() < MIN_BINS:
            continue
        r = med[s] / np.interp(cen[s], rx, rv)
        k = float(np.exp(np.mean(np.log(r))))
        out[n] = (k, float(np.sqrt(np.mean((r / k - 1) ** 2))))
    return out


def evaluate(bk, refs):
    """Return (kept, particle, diagnostics dict). Never raises."""
    d = dict(npts=bk["npts"], chi2=bk["chi2"], L=float(bk["rr"].max()))
    if bk["npts"] < MIN_NPTS:
        return False, None, dict(d, cut="npts")
    cen, med, cnt = profile(bk["rr"], bk["dqdx"])
    d.update(nbin=len(cen))
    if len(cen) < MIN_BINS:
        return False, None, dict(d, cut="nbin")
    d.update(rrmin=float(cen.min()), rrmax=float(cen.max()))
    if cen.min() > MAX_RRMIN or cen.max() < MIN_RRMAX:
        return False, None, dict(d, cut="rr coverage")
    near = (bk["rr"] < 2) & (bk["dqdx"] > 0)
    mid = (bk["rr"] >= 20) & (bk["rr"] < 40) & (bk["dqdx"] > 0)
    if near.sum() < 3 or mid.sum() < 5:
        return False, None, dict(d, cut="contrast stats")
    d["contrast"] = float(np.median(bk["dqdx"][near]) / np.median(bk["dqdx"][mid]))
    if d["contrast"] < MIN_CONTRAST:
        return False, None, dict(d, cut="contrast")
    if bk["chi2"] > MAX_CHI2:
        return False, None, dict(d, cut="chi2")
    sc = shape_scores(cen, med, refs)
    if not sc:
        return False, None, dict(d, cut="no hypothesis in domain")
    d["scores"] = sc
    best = min(sc, key=lambda n: sc[n][1])
    d["best_shape"] = best
    d["shape_rms"] = sc[best][1]
    if sc[best][1] > MAX_SHAPE_RMS:
        return False, None, dict(d, cut="shape rms")
    # assignment by scale
    part = None
    if MUON_K[0] <= sc["muon"][0] <= MUON_K[1]:
        part = "muon"
    elif "proton" in sc and PROTON_K[0] <= sc["proton"][0] <= PROTON_K[1]:
        part = "proton"
    if part is None:
        return False, None, dict(d, cut="scale in no window")
    d["particle"] = part
    d["k"] = sc[part][0]
    return True, part, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="print every block")
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--plot", help="also write an overlay PNG of the whole sample")
    ap.add_argument("--roots", help="comma-separated work roots to sweep "
                                    f"(default: {','.join(ROOTS)})")
    ap.add_argument("--events", help="restrict the sweep to these events: a "
                                     "comma-separated list, or a path to a file "
                                     "with one event id per line")
    ap.add_argument("--force-list", metavar="INDEX_TSV",
                    help="keep exactly the (event, block) set of this "
                         "sample_index.tsv with its particle assignment, and do "
                         "NOT apply the cuts -- for an A/B on identical tracks")
    ap.add_argument("--suffix", default="",
                    help="appended to the output basenames, e.g. _d66")
    args = ap.parse_args()

    roots = args.roots.split(",") if args.roots else list(ROOTS)
    forced = read_index(args.force_list) if args.force_list else None
    events = None
    if args.events:
        events = (set(open(args.events).read().split())
                  if os.path.exists(args.events) else set(args.events.split(",")))
    print(f"roots: {' '.join(roots)}"
          + (f"\nevents: {len(events)} from {args.events}" if events else "")
          + (f"\nforced track list: {len(forced)} blocks from {args.force_list} "
             "(cuts NOT applied)" if forced else ""))

    refs = load_refs()
    kept, seen = [], 0
    for bk in iter_blocks(roots, events):
        seen += 1
        ok, part, d = evaluate(bk, refs)
        if forced is not None:
            want = forced.get((bk["event"], bk["block"]))
            if want is None:
                continue
            # replay the published set: take the track whatever the cuts say, but
            # recompute k / rms / contrast on THIS arm's charge.
            ok, part = True, want
            cen, med, _ = profile(bk["rr"], bk["dqdx"])
            near = (bk["rr"] < 2) & (bk["dqdx"] > 0)
            mid = (bk["rr"] >= 20) & (bk["rr"] < 40) & (bk["dqdx"] > 0)
            d["scores"] = shape_scores(cen, med, refs)
            d["nbin"] = len(cen)
            d["contrast"] = (float(np.median(bk["dqdx"][near])
                                   / np.median(bk["dqdx"][mid]))
                             if near.sum() >= 3 and mid.sum() >= 3 else float("nan"))
            d["particle"] = part
            d["k"] = d["scores"][part][0]
            d["shape_rms"] = d["scores"][part][1]
            d.pop("cut", None)
        if args.verbose:
            print(f"{'KEEP' if ok else 'drop':>4s} {bk['event']} blk{bk['block']:<4d} "
                  f"st{bk['status']} " +
                  (f"{part:>6s} k={d['k']:.2f} rms={d['shape_rms']*100:.1f}% "
                   f"contrast={d['contrast']:.2f}" if ok
                   else f"cut={d.get('cut')}  " +
                        " ".join(f"{k}={v:.3g}" for k, v in d.items()
                                 if k not in ("cut", "scores") and isinstance(v, float))))
        if ok:
            kept.append((bk, d))

    kept.sort(key=lambda kb: (kb[1]["particle"], kb[0]["event"], kb[0]["block"]))

    idx = os.path.join(args.outdir, f"sample_index{args.suffix}.tsv")
    pts = os.path.join(args.outdir, f"sample_points{args.suffix}.tsv")
    with open(idx, "w") as fi, open(pts, "w") as fp:
        fi.write("# clean stopping-track dQ/dx-vs-rr sample, MCP2025C reco1\n")
        fi.write(f"# roots: {' '.join(roots)}\n")
        if forced:
            fi.write(f"# track set REPLAYED from {args.force_list}; the cuts below "
                     "were NOT applied\n")
        fi.write(f"# cuts: npts>={MIN_NPTS} nbin>={MIN_BINS} rrmin<={MAX_RRMIN} "
                 f"rrmax>={MIN_RRMAX} contrast>={MIN_CONTRAST} chi2<={MAX_CHI2} "
                 f"shape_rms<={MAX_SHAPE_RMS}\n")
        fi.write("# k_<h> = free scale needed against hypothesis h; rms_<h> = shape "
                 "residual after taking k out\n")
        fi.write("particle\troot\tevent\tblock\tstatus\tnpts\tL_cm\tcontrast\tchi2\t"
                 "k_muon\trms_muon\tk_proton\trms_proton\tmean_absx_cm\n")
        fp.write("# per-point sample; dqdx in e/cm, rr/x/y/z/dx in cm\n")
        fp.write("# drift_cm = 202.05 - |x| (anode at |x|=202.05); "
                 "drift_us = drift_cm / 0.1563\n")
        fp.write("particle\tevent\tblock\trr\tdqdx\tdx\tx\ty\tz\tdrift_cm\tdrift_us\n")
        for bk, d in kept:
            sc = d["scores"]
            fi.write(f"{d['particle']}\t{bk['root']}\t{bk['event']}\t{bk['block']}\t"
                     f"{bk['status']}\t{bk['npts']}\t{d['L']:.1f}\t{d['contrast']:.2f}\t"
                     f"{bk['chi2']:.2f}\t{sc['muon'][0]:.3f}\t{sc['muon'][1]*100:.1f}\t"
                     f"{sc['proton'][0]:.3f}\t{sc['proton'][1]*100:.1f}\t"
                     f"{np.mean(np.abs(bk['x'])):.1f}\n")
            good = bk["dqdx"] > 0
            for i in np.where(good)[0]:
                dr = ANODE_ABS_X - abs(float(bk["x"][i]))
                fp.write(f"{d['particle']}\t{bk['event']}\t{bk['block']}\t"
                         f"{bk['rr'][i]:.3f}\t{bk['dqdx'][i]:.1f}\t{bk['dx'][i]:.4f}\t"
                         f"{bk['x'][i]:.2f}\t{bk['y'][i]:.2f}\t{bk['z'][i]:.2f}\t"
                         f"{dr:.2f}\t{dr / DRIFT_SPEED:.1f}\n")

    n_by = {}
    for _, d in kept:
        n_by[d["particle"]] = n_by.get(d["particle"], 0) + 1
    print(f"\nscanned {seen} blocks; kept {len(kept)}")
    for p, n in sorted(n_by.items()):
        print(f"  {p:>7s}: {n}")
    print(f"wrote {idx}\nwrote {pts}")

    if args.plot:
        plot_sample(kept, refs, args.plot)
        print(f"wrote {args.plot}")


def plot_sample(kept, refs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"muon": "#2a78d6", "proton": "#e34948"}
    n_of = {p: sum(1 for _, dd in kept if dd["particle"] == p)
            for p in ("muon", "proton")}
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))

    ax = axes[0]
    for n, ls, lab in (("muon", "-", "muon expectation (SBND)"),
                       ("proton", "--", "proton expectation (SBND)")):
        rx, rv = refs[n]
        ax.plot(rx, rv / 1e3, ls, color="#0b0b0b" if n == "muon" else "#52514e",
                lw=1.8, zorder=6, label=lab)
    ax.axhline(56, ls=":", color="#a3a29b", lw=1.4, zorder=1,
               label="flat MIP 56 ke/cm")
    first = dict.fromkeys(col, True)
    for bk, d in kept:
        cen, med, _ = profile(bk["rr"], bk["dqdx"])
        p = d["particle"]
        ax.plot(cen, med / 1e3, "o-", color=col[p], ms=4.5, lw=1.3, alpha=0.8,
                mec="white", mew=0.8, zorder=4,
                label=(f"{p} tracks ({n_of[p]})" if first[p] else None))
        first[p] = False
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 290)
    ax.set_xlabel("residual range from the stopping end  [cm]")
    ax.set_ylabel("dQ/dx  [ke/cm]")
    ax.set_title("binned medians over the reference-table domain", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.2, lw=0.6)

    # right: each track against ITS OWN assigned reference, no free scale removed
    ax = axes[1]
    ax.axhline(1.0, ls=":", color="#a3a29b", lw=1.4)
    first = dict.fromkeys(col, True)
    for bk, d in kept:
        cen, med, _ = profile(bk["rr"], bk["dqdx"])
        p = d["particle"]
        rx, rv = refs[p]
        s = (cen >= rx.min()) & (cen <= rx.max())
        ax.plot(cen[s], med[s] / np.interp(cen[s], rx, rv), "o-", color=col[p],
                ms=4.5, lw=1.3, alpha=0.8, mec="white", mew=0.8,
                label=(f"{p} tracks ({n_of[p]})" if first[p] else None))
        first[p] = False
    ax.set_xlim(0, 60)
    ax.set_ylim(0.6, 1.7)
    ax.set_xlabel("residual range from the stopping end  [cm]")
    ax.set_ylabel("data / its own SBND expectation")
    ax.set_title("ratio to the assigned particle's own curve\n"
                 "(no free scale removed -- this is the raw agreement)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.2, lw=0.6)
    fig.suptitle("The selected dQ/dx-vs-residual-range sample: binned medians of "
                 f"{len(kept)} tracks from 30 MCP2025C events (uncalibrated data)",
                 fontsize=10, color="#52514e")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=140)


if __name__ == "__main__":
    main()
