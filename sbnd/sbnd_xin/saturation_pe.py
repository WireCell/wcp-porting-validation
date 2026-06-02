#!/usr/bin/env python3
"""SBND saturated-PMT analysis: do bright flashes have zero-PE PMTs (MC vs data)?

Background (chisquare_flags_comparison.md §12.1). QLMatching has an MC-only gate
`mc_saturation_pe` (default 5000 PE, QLMatching.cxx:455-461): for a flash with
total_PE > 5000 in MC, any PMT reading pe==0 is treated as a *simulated saturated*
PMT and masked out of that flash's chi2/LASSO. The premise is that the optical sim
encodes PMT saturation as pe=0, so a zero channel inside a bright flash is saturated
rather than genuinely dark. Data is NOT masked by this gate (it uses ch_mask for
dead channels). This script checks the premise on real samples.

The yes/no question ("is there a saturated-PMT situation?") cannot be answered by the
total-PE-vs-n_zero scatter alone: a saturating PMT (fires when dim, rails to 0 when
bright, sits among the brightest PMTs) and a dead/edge PMT (~0 always) both produce a
flat floor of N zeros. So we add the discriminators that separate them:
  * per-channel zero-fraction, split by total-PE band  -> dead floor vs brightness-on
  * channel-identity of bright-flash zeros, MC vs data  -> common dead set vs MC artifact
  * overlap of bright-flash zeros with ch_mask
  * spatial (y,z) PE map of the brightest flashes        -> central (saturation) vs edge

Data model (verified). Per-TPC archives input_files/input-{10,100}evt-{mc,data}/
opflash_apa{0,1}.tar.gz hold, per event, opflash_tensor_<ev>_0_array.npy of shape
[nflash, 313]: col 0 = flash time (ns), cols 1..312 = per-channel PE. Archive apaN
carries only TPC-N dome-PMT light (asserted below). PMT geometry/type come from the
semi-analytical model JSON (type==1 = dome PMT; x<0 = TPC0, x>=0 = TPC1).

Outputs: saturation_flash_dump.csv, saturation_perchan.csv, pics/saturation_*.png.

Usage:  python3 saturation_pe.py [--threshold 5000]
"""
import argparse
import io
import json
import os
import re
import tarfile

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(HERE, "input_files")
PICS = os.path.join(HERE, "pics")
FLASH_DUMP = os.path.join(HERE, "saturation_flash_dump.csv")
PERCHAN_DUMP = os.path.join(HERE, "saturation_perchan.csv")

SEMIMODEL = "/nfs/data/1/xqian/toolkit-dev/wire-cell-data/sbnd/photodet/semi-analytical-sbnd.json"

# Matching-only dead-channel list, identical for MC and data
# (cfg/pgrapher/experiment/sbnd/qlmatching.jsonnet, applied regardless of reality).
CH_MASK = [39, 64, 66, 71, 85, 86, 87, 115, 138, 141, 197, 217, 221,
           222, 223, 226, 245, 249, 302]

EV_RE = re.compile(r"opflash_tensor_(\d+)_0_array\.npy$")

# (mode, sample) -> input subdirectory holding opflash_apa{0,1}.tar.gz
SAMPLES = {
    ("mc", "10"): os.path.join(INPUT, "input-10evt-mc"),
    ("data", "10"): os.path.join(INPUT, "input-10evt-data"),
    ("mc", "100"): os.path.join(INPUT, "input-100evt-mc"),
    ("data", "100"): os.path.join(INPUT, "input-100evt-data"),
}

MODE_COLOR = {"mc": "C0", "data": "C3"}


# ---------------------------------------------------------------------------
# Geometry: dome-PMT channel sets per TPC
# ---------------------------------------------------------------------------
def load_pmt_geometry():
    """Return (tpc_pmts, pos) where tpc_pmts[apa] is the sorted list of dome-PMT
    channel indices on that TPC and pos[ch] = (x, y, z) cm."""
    od = json.load(open(SEMIMODEL))["OpDets"]
    pos = {i: (o["x"], o["y"], o["z"]) for i, o in enumerate(od)}
    pmts = [i for i, o in enumerate(od) if o["type"] == 1]
    tpc_pmts = {
        0: sorted(i for i in pmts if pos[i][0] < 0.0),   # apa0 = x<0
        1: sorted(i for i in pmts if pos[i][0] >= 0.0),  # apa1 = x>=0
    }
    return tpc_pmts, pos


# ---------------------------------------------------------------------------
# Archive loading (pattern shared with flash_coincidence.py)
# ---------------------------------------------------------------------------
def load_archive(path):
    """Return {event_id: ndarray[nflash, 313]} from one opflash tar.gz."""
    out = {}
    with tarfile.open(path, "r:gz") as t:
        for m in t.getmembers():
            mo = EV_RE.search(m.name)
            if not mo:
                continue
            out[int(mo.group(1))] = np.load(io.BytesIO(t.extractfile(m).read()))
    return out


# ---------------------------------------------------------------------------
# Per-flash records
# ---------------------------------------------------------------------------
def flash_records(tpc_pmts):
    """Yield a dict of per-flash observables over every (mode, sample, apa, event, flash).

    Also asserts the archive carries only same-TPC PMT light.
    """
    chmask = set(CH_MASK)
    for (mode, sample), d in SAMPLES.items():
        for apa in (0, 1):
            store = load_archive(os.path.join(d, "opflash_apa%d.tar.gz" % apa))
            active_all = tpc_pmts[apa]                       # 60 same-TPC PMTs
            active = [c for c in active_all if c not in chmask]  # minus dead channels
            other = np.array([c for c in range(312)
                              if c not in set(active_all)], dtype=int)
            for ev in sorted(store):
                arr = store[ev]
                pe_mat = arr[:, 1:]
                for i in range(arr.shape[0]):
                    pe = pe_mat[i]
                    # archive must carry only same-TPC PMT light
                    assert pe[other].sum() == 0.0, (
                        "non-TPC%d-PMT light in %s/%s apa%d ev%d flash%d"
                        % (apa, mode, sample, apa, ev, i))
                    pe_active = pe[active]
                    pe_active_all = pe[active_all]
                    nz_active = pe_active[pe_active > 0]
                    yield {
                        "mode": mode, "sample": sample, "apa": apa, "event": ev,
                        "flash_index": i, "time_ns": float(arr[i, 0]),
                        "total_pe": float(pe.sum()),
                        "n_active_pmt": len(active),
                        "n_zero_excl_chmask": int((pe_active == 0).sum()),
                        "n_zero_incl_chmask": int((pe_active_all == 0).sum()),
                        "min_pe_active": float(pe_active.min()) if len(active) else float("nan"),
                        "min_nonzero_pe": float(nz_active.min()) if nz_active.size else float("nan"),
                    }


def dump_flash_csv(records):
    cols = ["mode", "sample", "apa", "event", "flash_index", "time_ns", "total_pe",
            "n_active_pmt", "n_zero_excl_chmask", "n_zero_incl_chmask",
            "min_pe_active", "min_nonzero_pe"]
    with open(FLASH_DUMP, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in records:
            f.write(",".join(_fmt(r[c]) for c in cols) + "\n")
    print("wrote", FLASH_DUMP, "(%d flashes)" % len(records))


def _fmt(v):
    if isinstance(v, float):
        return "nan" if np.isnan(v) else "%.6g" % v
    return str(v)


# ---------------------------------------------------------------------------
# Per-channel zero-fraction (the key discriminator)
# ---------------------------------------------------------------------------
def per_channel_zero_fraction(tpc_pmts, sample, threshold):
    """Return {(mode, apa, ch): {'n_all','z_all','n_bright','z_bright'}} for one sample.

    z_* = number of flashes where this channel read 0 PE; n_* = number of flashes.
    'bright' = total_pe >= threshold.
    """
    out = {}
    for (mode, smp), d in SAMPLES.items():
        if smp != sample:
            continue
        for apa in (0, 1):
            store = load_archive(os.path.join(d, "opflash_apa%d.tar.gz" % apa))
            for ch in tpc_pmts[apa]:
                out[(mode, apa, ch)] = {"n_all": 0, "z_all": 0,
                                        "n_bright": 0, "z_bright": 0}
            for ev in sorted(store):
                arr = store[ev]
                for i in range(arr.shape[0]):
                    pe = arr[i, 1:]
                    bright = pe.sum() >= threshold
                    for ch in tpc_pmts[apa]:
                        rec = out[(mode, apa, ch)]
                        rec["n_all"] += 1
                        rec["z_all"] += int(pe[ch] == 0)
                        if bright:
                            rec["n_bright"] += 1
                            rec["z_bright"] += int(pe[ch] == 0)
    return out


def dump_perchan_csv(perchan, sample):
    chmask = set(CH_MASK)
    with open(PERCHAN_DUMP, "w") as f:
        f.write("mode,sample,apa,channel,in_chmask,n_flash,n_zero,"
                "zerofrac_all,n_bright,n_zero_bright,zerofrac_bright\n")
        for (mode, apa, ch), r in sorted(perchan.items()):
            za = r["z_all"] / r["n_all"] if r["n_all"] else 0.0
            zb = r["z_bright"] / r["n_bright"] if r["n_bright"] else float("nan")
            f.write("%s,%s,%d,%d,%d,%d,%d,%.4f,%d,%d,%s\n" % (
                mode, sample, apa, ch, int(ch in chmask), r["n_all"], r["z_all"],
                za, r["n_bright"], r["z_bright"],
                "nan" if np.isnan(zb) else "%.4f" % zb))
    print("wrote", PERCHAN_DUMP)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_totpe_vs_nzero(records, sample, threshold, outfile):
    rows = [r for r in records if r["sample"] == sample]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    rng = np.random.default_rng(0)
    for mode in ("mc", "data"):
        x = np.array([r["total_pe"] for r in rows if r["mode"] == mode])
        y = np.array([r["n_zero_excl_chmask"] for r in rows if r["mode"] == mode], float)
        x = np.clip(x, 1.0, None)
        ax.scatter(x, y + rng.uniform(-0.18, 0.18, y.size), s=12, alpha=0.45,
                   color=MODE_COLOR[mode], label="%s (N=%d)" % (mode, x.size))
        # binned median to expose any upturn with brightness
        edges = np.logspace(np.log10(max(1.0, x.min())), np.log10(x.max()), 12)
        idx = np.digitize(x, edges)
        bx, by = [], []
        for b in range(1, len(edges)):
            m = idx == b
            if m.sum() >= 3:
                bx.append(np.sqrt(edges[b - 1] * edges[b]))
                by.append(np.median(y[m]))
        ax.plot(bx, by, "-o", color=MODE_COLOR[mode], lw=2, ms=4)
    ax.axvline(threshold, color="k", ls="--", lw=1.2,
               label="mc_saturation_pe = %g" % threshold)
    ax.set_xscale("log")
    ax.set_xlabel("flash total PE")
    ax.set_ylabel("# zero-PE PMTs (active set, ch_mask excluded)")
    ax.set_title("Zero-PE PMTs vs flash brightness  (%s-evt; lines = binned median)" % sample)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    _save(fig, outfile)


def plot_totpe_vs_minpe(records, sample, threshold, outfile):
    rows = [r for r in records if r["sample"] == sample]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)
    for ax, key, ylab, title in (
        (axes[0], "min_pe_active", "min PE over active PMTs (incl. zeros)",
         "Minimum PE (literal): 0 whenever any PMT is dark"),
        (axes[1], "min_nonzero_pe", "min PE over PMTs that fired (>0)",
         "Minimum *non-zero* PE (dimmest firing PMT)")):
        for mode in ("mc", "data"):
            x = np.array([r["total_pe"] for r in rows if r["mode"] == mode])
            y = np.array([r[key] for r in rows if r["mode"] == mode], float)
            x = np.clip(x, 1.0, None)
            ax.scatter(x, y, s=12, alpha=0.45, color=MODE_COLOR[mode],
                       label="%s (N=%d)" % (mode, x.size))
        ax.axvline(threshold, color="k", ls="--", lw=1.2)
        ax.set_xscale("log")
        ax.set_xlabel("flash total PE")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which="both")
    axes[1].set_yscale("log")
    fig.suptitle("Flash total PE vs minimum per-PMT PE  (%s-evt)" % sample)
    _save(fig, outfile)


def plot_zerofrac_perchan(perchan, sample, outfile):
    chmask = set(CH_MASK)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for ax, frac_key, n_key, title in (
        (axes[0], "z_all", "n_all", "All flashes"),
        (axes[1], "z_bright", "n_bright",
         "Bright flashes only (total PE >= threshold)")):
        for mode in ("mc", "data"):
            chans, fracs = [], []
            for (m, apa, ch), r in sorted(perchan.items()):
                if m != mode:
                    continue
                n = r[n_key]
                chans.append(ch)
                fracs.append(r[frac_key] / n if n else np.nan)
            ax.plot(chans, fracs, "o", ms=4, color=MODE_COLOR[mode],
                    alpha=0.7, label=mode)
        # mark dead channels
        ymax = ax.get_ylim()[1]
        for ch in chmask:
            ax.axvline(ch, color="0.8", lw=0.8, zorder=0)
        ax.plot([], [], color="0.8", lw=2, label="ch_mask channel")
        ax.set_ylabel("zero-PE fraction")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="center right")
    axes[1].set_xlabel("optical channel index (dome PMTs, both TPCs)")
    fig.suptitle("Per-PMT zero-PE fraction  (%s-evt)  -- dead floor vs brightness-driven" % sample)
    _save(fig, outfile)


def plot_pmt_map(tpc_pmts, pos, sample, threshold, outfile):
    """(y,z) PE map of the brightest apa0 flash in MC and data; zeros marked X."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    chmask = set(CH_MASK)
    for ax, mode in zip(axes, ("mc", "data")):
        store = load_archive(os.path.join(SAMPLES[(mode, sample)],
                                          "opflash_apa0.tar.gz"))
        best = None
        for ev in store:
            arr = store[ev]
            for i in range(arr.shape[0]):
                tot = arr[i, 1:].sum()
                if best is None or tot > best[0]:
                    best = (tot, ev, i, arr[i, 1:])
        tot, ev, i, pe = best
        chans = tpc_pmts[0]
        z = np.array([pos[c][2] for c in chans])
        y = np.array([pos[c][1] for c in chans])
        v = np.array([pe[c] for c in chans])
        fired = v > 0
        sc = ax.scatter(z[fired], y[fired], c=v[fired], s=160, cmap="viridis",
                        norm=matplotlib.colors.LogNorm(), edgecolors="k", linewidths=0.5)
        # zero PMTs: distinguish ch_mask (gray square) from non-mask zeros (red X)
        for c, zz, yy in zip(chans, z, y):
            if pe[c] == 0:
                if c in chmask:
                    ax.scatter([zz], [yy], marker="s", s=120, facecolors="none",
                               edgecolors="0.5", linewidths=1.5)
                else:
                    ax.scatter([zz], [yy], marker="X", s=160, color="red",
                               edgecolors="k", linewidths=0.6)
        fig.colorbar(sc, ax=ax, label="PE")
        nzero = int((v[~np.isin(chans, list(chmask))] == 0).sum()) if len(chans) else 0
        n_nonmask_zero = sum(1 for c in chans if pe[c] == 0 and c not in chmask)
        ax.set_title("%s  brightest apa0 flash  (ev%d, total PE=%.0f)\n"
                     "non-mask zero PMTs: %d" % (mode, ev, tot, n_nonmask_zero),
                     fontsize=10)
        ax.set_xlabel("z [cm]")
        ax.set_ylabel("y [cm]")
        ax.grid(alpha=0.3)
    fig.suptitle("Per-PMT PE map (TPC0 face).  red X = zero PMT (not ch_mask);  "
                 "gray square = ch_mask zero")
    _save(fig, outfile)


def _save(fig, outfile):
    fig.tight_layout()
    fig.savefig(outfile, dpi=120)
    plt.close(fig)
    print("wrote", outfile)


# ---------------------------------------------------------------------------
# Text summary (the yes/no answers)
# ---------------------------------------------------------------------------
def dead_floor(perchan):
    """Per-apa count of non-ch_mask channels that read 0 in 100% of all flashes
    (the permanently-dead detectors missing from ch_mask)."""
    chmask = set(CH_MASK)
    floor = {0: 0, 1: 0}
    seen = set()
    for (mode, apa, ch), r in perchan.items():
        if ch in chmask or (apa, ch) in seen:
            continue
        # dead in this mode if zero in every flash; require both modes agree
        dead_here = r["n_all"] > 0 and r["z_all"] == r["n_all"]
        other = perchan.get(("data" if mode == "mc" else "mc", apa, ch))
        dead_other = other and other["n_all"] > 0 and other["z_all"] == other["n_all"]
        if dead_here and dead_other:
            floor[apa] += 1
            seen.add((apa, ch))
    return floor


def text_summary(records, perchan, sample, threshold):
    chmask = set(CH_MASK)
    print("\n=== saturation summary (%s-evt, threshold=%g) ===" % (sample, threshold))
    rows = [r for r in records if r["sample"] == sample]
    floor = dead_floor(perchan)
    print("  dead-channel floor (non-ch_mask, zero in 100%% of flashes in BOTH modes): "
          "apa0=%d apa1=%d" % (floor[0], floor[1]))
    for mode in ("mc", "data"):
        mr = [r for r in rows if r["mode"] == mode]
        bright = [r for r in mr if r["total_pe"] >= threshold]
        dim = [r for r in mr if r["total_pe"] < threshold]
        nzb = np.array([r["n_zero_excl_chmask"] for r in bright])
        nzd = np.array([r["n_zero_excl_chmask"] for r in dim])
        print("  %-4s: %d flashes  (%d bright >=%g, %d dim)" %
              (mode, len(mr), len(bright), threshold, len(dim)))
        print("        bright n_zero(excl chmask): mean=%.2f median=%d max=%d  "
              "frac with >=1 zero=%.1f%%" %
              (nzb.mean() if nzb.size else 0, int(np.median(nzb)) if nzb.size else 0,
               int(nzb.max()) if nzb.size else 0,
               100 * (nzb > 0).mean() if nzb.size else 0))
        print("        dim    n_zero(excl chmask): mean=%.2f median=%d  frac>=1 zero=%.1f%%" %
              (nzd.mean() if nzd.size else 0, int(np.median(nzd)) if nzd.size else 0,
               100 * (nzd > 0).mean() if nzd.size else 0))
        # LIVE zeros in bright flashes = zeros beyond the dead floor: the only
        # population that matches the user's scenario (bright flash, live PMT at 0).
        lz = np.array([max(0, r["n_zero_excl_chmask"] - floor[r["apa"]]) for r in bright])
        tp = np.array([r["total_pe"] for r in bright])
        print("        bright LIVE-zero (n_zero - dead floor): frac>0=%.1f%% mean=%.2f max=%d"
              % (100 * (lz > 0).mean() if lz.size else 0,
                 lz.mean() if lz.size else 0, int(lz.max()) if lz.size else 0))
        for t in (threshold, 2 * threshold, 4 * threshold, 10 * threshold):
            m = tp >= t
            if m.sum():
                print("          total_pe>=%-7g: N=%4d  live-zero>0 frac=%.1f%%"
                      % (t, m.sum(), 100 * (lz[m] > 0).mean()))
        hi = tp[lz >= 5]
        if hi.size:
            print("          flashes with live-zero>=5: N=%d total_pe median=%.0f "
                  "(25/75=%.0f/%.0f)" % (hi.size, np.median(hi),
                                         np.percentile(hi, 25), np.percentile(hi, 75)))

    # channel identity of bright-flash zeros: a channel counts if it is 0 in >50% of
    # this mode's bright flashes (i.e. systematically dark when the flash is bright)
    def bright_dark_set(mode):
        s = set()
        for (m, apa, ch), r in perchan.items():
            if m == mode and r["n_bright"] >= 3 and r["z_bright"] / r["n_bright"] > 0.5:
                s.add(ch)
        return s
    mc_dark = bright_dark_set("mc")
    da_dark = bright_dark_set("data")
    print("  channels systematically dark in bright flashes (>50%):")
    print("    mc:   %s" % sorted(mc_dark))
    print("    data: %s" % sorted(da_dark))
    inter = mc_dark & da_dark
    union = mc_dark | da_dark
    print("    overlap mc&data = %d / union %d (Jaccard=%.2f)" %
          (len(inter), len(union), len(inter) / len(union) if union else 0))
    print("    of these, in ch_mask: mc %d/%d, data %d/%d" %
          (len(mc_dark & chmask), len(mc_dark), len(da_dark & chmask), len(da_dark)))
    print("    non-ch_mask dark channels: mc %s ; data %s" %
          (sorted(mc_dark - chmask), sorted(da_dark - chmask)))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=5000.0,
                    help="mc_saturation_pe threshold (default 5000)")
    args = ap.parse_args()
    os.makedirs(PICS, exist_ok=True)

    tpc_pmts, pos = load_pmt_geometry()
    print("dome PMTs: TPC0=%d TPC1=%d  (ch_mask=%d channels)"
          % (len(tpc_pmts[0]), len(tpc_pmts[1]), len(CH_MASK)))

    records = list(flash_records(tpc_pmts))
    dump_flash_csv(records)

    # primary statistics sample = 100-evt; 10-evt is a cross-check
    for sample in ("100", "10"):
        if not any(r["sample"] == sample for r in records):
            continue
        perchan = per_channel_zero_fraction(tpc_pmts, sample, args.threshold)
        if sample == "100":
            dump_perchan_csv(perchan, sample)
            plot_totpe_vs_nzero(records, sample, args.threshold,
                                os.path.join(PICS, "saturation_totPE_vs_nzero.png"))
            plot_totpe_vs_minpe(records, sample, args.threshold,
                                os.path.join(PICS, "saturation_totPE_vs_minPE.png"))
            plot_zerofrac_perchan(perchan, sample,
                                  os.path.join(PICS, "saturation_zerofrac_perchan.png"))
            plot_pmt_map(tpc_pmts, pos, sample, args.threshold,
                         os.path.join(PICS, "saturation_pmt_map.png"))
        text_summary(records, perchan, sample, args.threshold)


if __name__ == "__main__":
    main()
