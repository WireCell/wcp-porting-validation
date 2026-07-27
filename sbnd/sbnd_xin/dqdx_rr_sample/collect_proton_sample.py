#!/usr/bin/env python3
"""Collect the dQ/dx-vs-residual-range points of the protons the OWNER
hand-identified in the doc-62 scan, and merge them with doc 55's muon sample.

Doc 55 sections 6-10 were built on 12 muons and **one** proton, and its section 9
item 3 named the missing thing outright: "A proton population.  One track cannot
separate 'SBND's (A, B) differ' from 'the reconstruction has a dE/dx-dependent
bias'."  Doc 62 provides one -- 13 bundles the owner called protons by eye while
adjudicating the :5012 hand scan.  This script turns that list into the same
per-point TSV the doc-55 machinery already consumes, so
`fit_recombination.py --points ...` and `plot_muon_proton_models.py --points ...`
run on the enlarged sample with no change to either.

Selection is the owner's list, NOT a cut.  That is the point: doc 55 had to
assign particles *by scale* (a track needing k_muon ~ 1.9 was called a proton),
which makes any later statement about the proton scale partly circular.  Here the
identification is independent of the charge, so k_proton is a measurement.  Every
automatic quantity doc 55's selector cuts on (npts, bins, contrast, chi2, shape
residual) is still computed and written out per track, but nothing is dropped on
it -- see `--verbose`.

The one exclusion is model-independent and named in EXCLUDE below: 397920 main 8
is a 278.9 cm / 453-point fitted main, and the owner's own comment places its
proton *at a vertex*, i.e. as one prong of a multi-prong object.  The fitted
block is therefore not the proton, whatever its charge says.

Provenance.  The protons come from `work-mcp1kall-d59k` (the doc-59 1000-event
production run) and the muons from the doc-55 `d55ton` arms.  289343 main 9 is in
BOTH -- it is doc 55's original proton -- and its 77 points are bit-identical
between the two roots (`--control` checks this and refuses to merge if it ever
stops being true).  The proton rows in the merged file are taken from d59k and
the old file's proton rows are dropped, so it is counted once.

Nothing is re-run: every `tracking-stm.root` already exists.

Usage:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
  python3 dqdx_rr_sample/collect_proton_sample.py --verbose
  python3 dqdx_rr_sample/collect_proton_sample.py \
      --merge dqdx_rr_sample/sample_points_p12.tsv \
      --plot  dqdx_rr_sample/proton_sample_p12.png
"""
import argparse
import os

import numpy as np
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(HERE)
SBND_DQDX = ("/nfs/data/1/xqian/toolkit-dev/energy_loss/pion_travel/"
             "stopping_ave_dQ_dx_sbnd.root")
PROTON_LIST = os.path.join(TOP, "scan-d59k", "proton-list.tsv")
ROOT = "work-mcp1kall-d59k"
OLD_POINTS = os.path.join(HERE, "sample_points.tsv")
CONTROL = ("289343", 90, "work-mcp1000b-d55ton")   # in both roots, must agree

HYPS = ["muon", "pion", "kaon", "proton"]
BINS = [(0, 2), (2, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40), (40, 60)]

# SBND geometry / drift, cfg/pgrapher/experiment/sbnd/clus.jsonnet:22,82-83
ANODE_ABS_X = 202.05      # cm, W plane
DRIFT_SPEED = 0.1563      # cm/us

# event -> why this owner-identified proton cannot be used as a proton dQ/dx
# measurement.  Model-independent reasons only; nothing here is a charge cut.
EXCLUDE = {
    "397920": "fitted main is 278.9 cm / 453 pts; the owner puts the proton at a "
              "vertex, so this block is the event, not the proton",
}


def load_refs():
    f = uproot.open(SBND_DQDX)
    return {n: (np.asarray(f[n].values("x"), float),
                np.asarray(f[n].values("y"), float)) for n in HYPS}


def read_list(path):
    """The owner's proton list: (event, main_id, tagger, stm_ok, comment)."""
    out = []
    for line in open(path):
        if line.startswith("#") or line.startswith("event\t"):
            continue
        f = line.rstrip("\n").split("\t")
        out.append(dict(event=f[0], main=int(f[1]), t_us=float(f[3]),
                        len_cm=float(f[4]), tagger=f[5], stm_ok=f[6],
                        comment=f[7]))
    return out


def read_block(root, event, block):
    """One (cluster, pass) block of T_rec_charge, decoded to dQ/dx in e/cm."""
    fp = os.path.join(root, f"nusel_evt{event}", "tracking-stm.root")
    f = uproot.open(fp)
    t = f["T_rec_charge"].arrays(["x", "y", "z", "q", "nq", "rr", "ndf",
                                  "status", "reduced_chi2"], library="np")
    tr = f["Trun"].arrays(["dQdx_scale", "dQdx_offset"], library="np")
    m = t["ndf"] == block
    if not m.any():
        return None
    dQ = (t["q"][m] - tr["dQdx_offset"][0]) / tr["dQdx_scale"][0]
    dx = t["nq"][m]
    return dict(root=root, event=event, block=int(block),
                rr=t["rr"][m], dx=dx, q=t["q"][m],
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


def shape_scores(cen, med, refs, min_bins=3):
    """Free scale k against each hypothesis, and the shape residual left over."""
    out = {}
    for n, (rx, rv) in refs.items():
        s = (cen >= rx.min()) & (cen <= rx.max())
        if s.sum() < min_bins:
            continue
        r = med[s] / np.interp(cen[s], rx, rv)
        k = float(np.exp(np.mean(np.log(r))))
        out[n] = (k, float(np.sqrt(np.mean((r / k - 1) ** 2))), int(s.sum()))
    return out


def diagnostics(bk, refs):
    """Doc 55's selector quantities -- computed for the record, cut on by nobody."""
    d = dict(npts=bk["npts"], chi2=bk["chi2"], L=float(bk["rr"].max()))
    cen, med, cnt = profile(bk["rr"], bk["dqdx"])
    d["nbin"] = len(cen)
    d["rrmin"] = float(cen.min()) if len(cen) else float("nan")
    d["rrmax"] = float(cen.max()) if len(cen) else float("nan")
    near = (bk["rr"] < 2) & (bk["dqdx"] > 0)
    mid = (bk["rr"] >= 20) & (bk["rr"] < 40) & (bk["dqdx"] > 0)
    d["contrast"] = (float(np.median(bk["dqdx"][near]) / np.median(bk["dqdx"][mid]))
                     if near.sum() >= 3 and mid.sum() >= 3 else float("nan"))
    d["scores"] = shape_scores(cen, med, refs) if len(cen) else {}
    d["mean_absx"] = float(np.mean(np.abs(bk["x"])))
    dr = ANODE_ABS_X - np.abs(bk["x"])
    d["drift_us"] = float(np.mean(dr) / DRIFT_SPEED)
    return d


def control_check(refs):
    """289343 blk 90 lives in both roots.  The merged fit is only meaningful if
    d59k's dQ_dx_fit output is the same object d55ton's was."""
    ev, blk, old_root = CONTROL
    a = read_block(old_root, ev, blk)
    b = read_block(ROOT, ev, blk)
    if a is None or b is None:
        return False, "control block missing from one of the roots"
    same = all(a[v].shape == b[v].shape and np.array_equal(a[v], b[v])
               for v in ("x", "y", "z", "q", "dx", "rr"))
    return same, (f"{ev} blk{blk}: {a['npts']} pts in {old_root}, {b['npts']} in "
                  f"{ROOT}, point arrays "
                  + ("IDENTICAL" if same else "DIFFER"))


def write_points(fh, particle, bk, header=True):
    if header:
        fh.write("# per-point sample; dqdx in e/cm, rr/x/y/z/dx in cm\n")
        fh.write("# drift_cm = 202.05 - |x| (anode at |x|=202.05); "
                 "drift_us = drift_cm / 0.1563\n")
        fh.write("particle\tevent\tblock\trr\tdqdx\tdx\tx\ty\tz\tdrift_cm\t"
                 "drift_us\n")
    for i in np.where(bk["dqdx"] > 0)[0]:
        dr = ANODE_ABS_X - abs(float(bk["x"][i]))
        fh.write(f"{particle}\t{bk['event']}\t{bk['block']}\t"
                 f"{bk['rr'][i]:.3f}\t{bk['dqdx'][i]:.1f}\t{bk['dx'][i]:.4f}\t"
                 f"{bk['x'][i]:.2f}\t{bk['y'][i]:.2f}\t{bk['z'][i]:.2f}\t"
                 f"{dr:.2f}\t{dr / DRIFT_SPEED:.1f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true",
                    help="print every track's doc-55 selector quantities")
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--merge", help="also write muons(d55ton) + protons(d59k) "
                                    "into this per-point TSV")
    ap.add_argument("--plot", help="overlay PNG of the proton sample")
    ap.add_argument("--no-control", action="store_true",
                    help="skip the cross-root control (do not use for a merge)")
    args = ap.parse_args()

    refs = load_refs()

    ok, msg = (True, "skipped") if args.no_control else control_check(refs)
    print(f"cross-root control -- {msg}")
    if not ok and args.merge:
        raise SystemExit("REFUSING to merge: d59k and d55ton disagree on the "
                         "control block, so the muons must be re-collected from "
                         "d59k before any joint fit.")

    tracks = []
    for row in read_list(PROTON_LIST):
        bk = read_block(ROOT, row["event"], row["main"] * 10)
        if bk is None:
            print(f"  {row['event']} main {row['main']}: no block -- skipped")
            continue
        d = diagnostics(bk, refs)
        d["excluded"] = EXCLUDE.get(row["event"])
        tracks.append((row, bk, d))

    used = [t for t in tracks if not t[2]["excluded"]]
    print(f"\n{len(tracks)} owner-identified protons, {len(used)} usable "
          f"({len(tracks)-len(used)} excluded by name)")

    if args.verbose:
        print(f"\n  {'event':>7s} {'main':>4s} {'st':>2s} {'npts':>5s} {'L cm':>6s} "
              f"{'nbin':>4s} {'contr':>6s} {'chi2':>5s} {'k_mu':>6s} {'rms%':>5s} "
              f"{'k_p':>5s} {'rms%':>5s} {'drift':>6s} use")
        for row, bk, d in tracks:
            sc = d["scores"]
            km, rm = sc.get("muon", (float("nan"),) * 3)[:2]
            kp, rp = sc.get("proton", (float("nan"),) * 3)[:2]
            print(f"  {row['event']:>7s} {row['main']:4d} {bk['status']:2d} "
                  f"{d['npts']:5d} {d['L']:6.1f} {d['nbin']:4d} "
                  f"{d['contrast']:6.2f} {d['chi2']:5.2f} {km:6.2f} {rm*100:5.1f} "
                  f"{kp:5.2f} {rp*100:5.1f} {d['drift_us']:6.0f} "
                  + ("EXCLUDED: " + d["excluded"] if d["excluded"] else "yes"))

    idx = os.path.join(args.outdir, "proton_index.tsv")
    pts = os.path.join(args.outdir, "proton_points.tsv")
    with open(idx, "w") as fi:
        fi.write("# protons hand-identified by the OWNER in the doc-62 :5012 scan\n")
        fi.write(f"# blocks read from {ROOT}/nusel_evt<ID>/tracking-stm.root, "
                 "block = main_id * 10\n")
        fi.write("# NO cut is applied: the identification is the owner's, and the\n")
        fi.write("# columns below are doc 55's selector quantities kept for the "
                 "record.\n")
        fi.write("# use=no is a named, model-independent exclusion (see the "
                 "script's EXCLUDE).\n")
        fi.write("# k_<h> = free scale needed against hypothesis h; rms_<h> = "
                 "shape residual after taking k out\n")
        fi.write("event\tmain\tblock\tstatus\tuse\tnpts\tL_cm\tnbin\tcontrast\t"
                 "chi2\tk_muon\trms_muon\tk_proton\trms_proton\tmean_absx_cm\t"
                 "drift_us\ttagger\tstm_ok\towner_comment\treason\n")
        for row, bk, d in tracks:
            sc = d["scores"]
            km, rm = sc.get("muon", (float("nan"),) * 3)[:2]
            kp, rp = sc.get("proton", (float("nan"),) * 3)[:2]
            fi.write(f"{row['event']}\t{row['main']}\t{bk['block']}\t"
                     f"{bk['status']}\t{'no' if d['excluded'] else 'yes'}\t"
                     f"{d['npts']}\t{d['L']:.1f}\t{d['nbin']}\t"
                     f"{d['contrast']:.2f}\t{d['chi2']:.2f}\t{km:.3f}\t"
                     f"{rm*100:.1f}\t{kp:.3f}\t{rp*100:.1f}\t"
                     f"{d['mean_absx']:.1f}\t{d['drift_us']:.0f}\t"
                     f"{row['tagger']}\t{row['stm_ok']}\t{row['comment']}\t"
                     f"{d['excluded'] or ''}\n")
    with open(pts, "w") as fp:
        first = True
        for row, bk, d in used:
            write_points(fp, "proton", bk, header=first)
            first = False
    print(f"wrote {idx}\nwrote {pts}")

    if args.merge:
        nmu = 0
        with open(args.merge, "w") as fm:
            fm.write("# doc 55 section 11: the d55ton MUONS + the doc-62 OWNER-"
                     "identified PROTONS (d59k)\n")
            fm.write("# muon rows are copied verbatim from sample_points.tsv; "
                     "its single proton\n")
            fm.write("# (289343 blk 90) is dropped there and re-read from d59k, "
                     "so it is counted once.\n")
            hdr = True
            for line in open(OLD_POINTS):
                if line.startswith("#"):
                    continue
                if hdr:
                    fm.write("# per-point sample; dqdx in e/cm, rr/x/y/z/dx in cm\n")
                    fm.write("# drift_cm = 202.05 - |x|; drift_us = drift_cm "
                             "/ 0.1563\n")
                    fm.write(line)
                    hdr = False
                    continue
                if line.startswith("muon\t"):
                    fm.write(line)
                    nmu += 1
            for row, bk, d in used:
                write_points(fm, "proton", bk, header=False)
        ntrk = len({l.split("\t")[1] + l.split("\t")[2]
                    for l in open(OLD_POINTS) if l.startswith("muon\t")})
        print(f"wrote {args.merge}: {nmu} muon points ({ntrk} tracks) + "
              f"{len(used)} proton tracks")

    if args.plot:
        plot(used, refs, args.plot)
        print(f"wrote {args.plot}")


def plot(used, refs, out):
    """The protons on the two SBND reference curves, and against the free-power
    model of doc 55 section 7g as it stands in the committed json (frozen -- this
    figure asks whether the *existing* model describes the new population, it
    does not refit)."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jf = os.path.join(TOP, "nusel_display", "stm_ref_dqdx.json")
    ref_fit = None
    if os.path.exists(jf):
        j = json.load(open(jf))
        if "ProtonDeDx" in j:
            e = j["ProtonDeDx"]
            ref_fit = (np.array(e["start"] + np.arange(len(e["values"])) * e["step"],
                                dtype=float), np.array(e["values"], float))
        meta = j.get("_meta", {}).get("canonical_keys", {})
    else:
        meta = {}

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4))
    n = len(used)

    ax = axes[0]
    rx, rv = refs["proton"]
    ax.plot(rx, rv / 1e3, "-", color="#0b0b0b", lw=2.0, zorder=6,
            label="proton, Modified Box (the shipped table)")
    if ref_fit is not None:
        ax.plot(ref_fit[0], ref_fit[1] / 1e3, "-", color="#2a78d6", lw=2.0,
                zorder=6, label="proton, free power (doc 55 §7g)")
    mx, mv = refs["muon"]
    ax.plot(mx, mv / 1e3, "--", color="#52514e", lw=1.5, zorder=5,
            label="muon, Modified Box")
    ax.axhline(56, ls=":", color="#a3a29b", lw=1.4, label="flat MIP 56 ke/cm")
    for i, (row, bk, d) in enumerate(used):
        cen, med, _ = profile(bk["rr"], bk["dqdx"])
        ax.plot(cen, med / 1e3, "o-", color="#e34948", ms=4.5, lw=1.2, alpha=0.75,
                mec="white", mew=0.8, zorder=4,
                label=f"owner-identified protons ({n})" if i == 0 else None)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 300)
    ax.set_xlabel("residual range from the stopping end  [cm]")
    ax.set_ylabel("dQ/dx  [ke/cm]")
    ax.set_title("binned medians, one line per track", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.2, lw=0.6)

    ax = axes[1]
    ax.axhline(1.0, ls=":", color="#a3a29b", lw=1.4)
    for i, (row, bk, d) in enumerate(used):
        cen, med, _ = profile(bk["rr"], bk["dqdx"])
        for (gx, gy), col, lab in ((refs["proton"], "#0b0b0b", "Modified Box"),
                                   (ref_fit, "#2a78d6", "free power")):
            if gx is None:
                continue
            s = (cen >= gx.min()) & (cen <= gx.max())
            ax.plot(cen[s], med[s] / np.interp(cen[s], gx, gy), "o-", color=col,
                    ms=4.0, lw=1.1, alpha=0.55, mec="white", mew=0.6,
                    label=f"/ proton {lab}" if i == 0 else None)
    ax.set_xlim(0, 60)
    ax.set_ylim(0.6, 1.6)
    ax.set_xlabel("residual range from the stopping end  [cm]")
    ax.set_ylabel("data / the proton expectation")
    ax.set_title("ratio to the proton curve, no free scale removed", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.2, lw=0.6)

    sub = ("free power: " + ", ".join(f"{k} = {meta[k]}" for k in ("A", "k", "p", "C")
                                      if k in meta)) if meta else ""
    fig.suptitle(f"The {n} owner-identified protons of doc 62, against the SBND "
                 f"proton expectation  (uncalibrated MCP2025C data)\n{sub}",
                 fontsize=9.5, color="#52514e")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=140)


if __name__ == "__main__":
    main()
