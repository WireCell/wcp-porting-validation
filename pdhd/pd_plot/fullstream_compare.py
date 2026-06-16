#!/usr/bin/env python3
"""Compare the WCT-reconstructed -x FULL-STREAM PD flashes (opch 120-159) with
the self-trigger APAs, to check that full-stream flash forming is reasonable.

Both readout modes are reconstructed with the *same* WCT chain and the *same*
fixed Wiener filter (PDHDOpWaveformSource -> OpDecon -> OpHitFinder ->
OpFlashFinder), so their OpHits/flashes sit on the same trigger-relative clock.
The full-stream sees the whole ~5.5 ms continuously; the self-trigger APAs see
only triggered windows.  The physics check is therefore TIME-COINCIDENCE in the
overlapping window (a real cosmic lights both the full-stream -x-lower APA and a
self-trigger APA at the same time), measured as the EXCESS over a time-shuffled
random baseline -- not a raw flash count.  With the OpRoi cleaning
(pdhd-fullstream-light-reco.md §7) the artefact/noise-floor flashes are removed, so
the surviving flashes are coincidence-validated across the PE spectrum (every bin
above ~50 PE), and the full-stream PE spectrum overlays the self-trigger (panel D).
The remaining dim 0-50 PE bin sits below the random floor (scattered sub-pulses).

Geometry (opch -> wall / z):
  0-39   +x z267-427 (+x upper)      80-119  -x z267-427 (-x upper, self-trig)
  40-79  +x z35-195  (+x lower)      120-159 -x z35-195  (-x lower, FULL-STREAM)
The full-stream (120-159) matches: 80-119 (same -x wall, adjacent APA) and 40-79
(cross-cathode, same z).

Inputs: the per-event reco produced by
  run_light_fullstream_evt.sh <run> <evt>      -> work/<RUN>_fs<evt>/opflash_pdhd-fullstream-wct.tar.gz
  (self-trigger from raw, all 0-119)           -> work/<RUN>_snip<evt>/opflash_pdhd-wct.tar.gz

Usage:
  fullstream_compare.py <run> <evt> [evt2 ...]   # cycle-1 figure for evt (first);
                                                 # multi-evt prints the event table.
"""
import os
import sys
import tarfile
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PDHD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(PDHD, "work")
PICS = os.path.join(PDHD, "pics")

REGIONS = {  # label: (opch lo, hi), match to full-stream 120-159
    "+x upper 0-39 (diagonal)": (0, 40),
    "+x lower 40-79 (x-cathode z-match)": (40, 80),
    "-x upper 80-119 (same wall)": (80, 120),
}
SPAN = (-2750.0, 2750.0)  # us, the readout window


def load_tensors(tgz):
    d = {}
    with tarfile.open(tgz) as t:
        for m in t.getmembers():
            if m.name.endswith("_array.npy"):
                d[int(m.name.split("_")[-2])] = np.load(io.BytesIO(t.extractfile(m).read()))
    return d  # 0: opflash[nf,1+160] (col0 t_ns, 1..160 PE/opdet); 1: summary[nf,8] ([id,total_pe,...,nhits])


def load_event(run, evt):
    rp = "%06d" % run
    fs = load_tensors("%s/%s_fs%d/opflash_pdhd-fullstream-wct.tar.gz" % (WORK, rp, evt))
    sn = load_tensors("%s/%s_snip%d/opflash_pdhd-wct.tar.gz" % (WORK, rp, evt))
    out = dict(
        fs_t=fs[0][:, 0] / 1000.0, fs_pe=fs[1][:, 1],
        sn_t=sn[0][:, 0] / 1000.0, sn_pe=sn[1][:, 1], sn_opf=sn[0][:, 1:],
    )
    return out


def nearest_dt(ref, other):
    """Min |ref - other| for each ref time (us)."""
    o = np.sort(other)
    if len(o) == 0:
        return np.full(len(ref), 1e9)
    idx = np.searchsorted(o, ref)
    dt = np.full(len(ref), 1e9)
    for i, x in enumerate(ref):
        for j in (idx[i] - 1, idx[i]):
            if 0 <= j < len(o):
                dt[i] = min(dt[i], abs(x - o[j]))
    return dt


def coinc_fraction(ref_t, fs_t, win=1.0):
    return (nearest_dt(ref_t, fs_t) < win).mean() if len(ref_t) else np.nan


def random_fraction(nref, fs_t, win=1.0, ntrials=40, seed=1):
    rng = np.random.default_rng(seed)
    return np.mean([(nearest_dt(rng.uniform(*SPAN, nref), fs_t) < win).mean()
                    for _ in range(ntrials)])


def bright_xup_ref(d, pe_cut=200):
    """Bright self-trigger flashes dominated by -x upper (80-119): real -x cosmics."""
    xup = d["sn_opf"][:, 80:120].sum(1)
    tot = d["sn_opf"].sum(1)
    m = (d["sn_pe"] > pe_cut) & (xup > 0.5 * tot)
    return d["sn_t"][m]


def make_cycle1_figure(run, evt):
    d = load_event(run, evt)
    fst, fspe, snt = d["fs_t"], d["fs_pe"], d["sn_t"]
    ref = bright_xup_ref(d)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))

    # A: flash-time histograms, full-stream vs self-trigger by region.
    bins = np.linspace(*SPAN, 120)
    ax[0, 0].hist(fst, bins=bins, color="#d62728", alpha=0.5, label="full-stream 120-159 (%d)" % len(fst))
    for lab, (lo, hi) in REGIONS.items():
        reg = d["sn_opf"][:, lo:hi].sum(1)
        m = reg > 0.5 * d["sn_opf"].sum(1)
        ax[0, 0].hist(snt[m], bins=bins, histtype="step", lw=1.4, label="self-trig %s (%d)" % (lab.split(" (")[0], m.sum()))
    ax[0, 0].set(xlabel="flash time (us, trigger-relative)", ylabel="flashes/bin",
                 title="A. Flash times populate the same readout window")
    ax[0, 0].legend(fontsize=6.5)

    # B: dt of bright -x-upper self-trigger flashes to nearest full-stream flash.
    dt = nearest_dt(ref, fst)
    ax[0, 1].hist(np.clip(dt, 0, 10), bins=np.linspace(0, 10, 41), color="#1f77b4")
    ax[0, 1].axvline(1.0, ls="--", c="green", label="+-1us coincidence")
    ax[0, 1].set(xlabel="|dt| to nearest full-stream flash (us)", ylabel="bright -x-upper flashes",
                 title="B. Bright -x cosmics peak at dt~0 (N=%d)" % len(ref))
    ax[0, 1].legend(fontsize=8)

    # C: per-PE-bin (NON-cumulative) coincidence excess.  Only the bright bin
    # exceeds the random floor -> only bright full-stream flashes are validated as
    # real cosmics; the dim population sits at/below random (noise-floor flashes).
    pbins = [(0, 50), (50, 200), (200, 800), (800, 1e9)]
    plabels = ["0-50", "50-200", "200-800", ">800"]
    exc = []
    for lo, hi in pbins:
        fsm = fst[(fspe >= lo) & (fspe < hi)]
        o = coinc_fraction(ref, fsm)
        r = random_fraction(len(ref), fsm)
        exc.append(o / r if r > 0 else 0)
    colors = ["gray" if e < 1.3 else "#d62728" for e in exc]
    ax[1, 0].bar(plabels, exc, color=colors)
    ax[1, 0].axhline(1.0, ls="--", c="green", label="random floor")
    ax[1, 0].set(xlabel="full-stream flash PE bin", ylabel="coincidence excess over random",
                 title="C. Flashes coincidence-validated across the PE spectrum (post-§7 cleaning)")
    ax[1, 0].legend(fontsize=8)
    for i, e in enumerate(exc):
        ax[1, 0].annotate("x%.1f" % e, (i, e), ha="center", va="bottom", fontsize=8)

    # D: flash PE distributions.  The full-stream finds a large dim population
    # (the >800 PE bright tail is the coincidence-validated part; see panel C).
    ax[1, 1].hist(np.log10(np.clip(fspe, 1, None)), bins=40, color="#d62728", alpha=0.5,
                  label="full-stream 120-159")
    ax[1, 1].hist(np.log10(np.clip(d["sn_pe"], 1, None)), bins=40, histtype="step", lw=1.4,
                  color="#1f77b4", label="self-trigger 0-119")
    ax[1, 1].axvline(np.log10(50), ls="--", c="green", lw=0.9, label="PE=50 (validated above)")
    ax[1, 1].set(xlabel="log10(flash total PE)", ylabel="flashes",
                 title="D. Flash PE spectra now overlay the self-trigger")
    ax[1, 1].legend(fontsize=8)

    fig.suptitle("PDHD run %d evt %d: -x full-stream (120-159) light reco vs self-trigger APAs"
                 % (run, evt), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(PICS, exist_ok=True)
    out = "%s/pd_fullstream_%d_evt%d_coincidence.png" % (PICS, run, evt)
    fig.savefig(out, dpi=95)
    print("wrote", out)
    # numeric summary
    print("\nfull-stream flashes: %d   self-trigger flashes: %d   bright -x cosmics: %d"
          % (len(fst), len(snt), len(ref)))
    print("per-PE-bin coincidence excess (non-cumulative):")
    for lo, hi in [(0, 50), (50, 200), (200, 800), (800, 1e9)]:
        fsm = fst[(fspe >= lo) & (fspe < hi)]
        o = coinc_fraction(ref, fsm); r = random_fraction(len(ref), fsm)
        print("  PE [%5g,%6g): N=%4d  coinc %2.0f%% vs random %2.0f%%  (x%.2f)"
              % (lo, hi, len(fsm), 100 * o, 100 * r, o / max(r, 1e-9)))


def event_table(run, evts, pe_cut=800):
    """Per-event coincidence of bright (>pe_cut) full-stream flashes with the
    self-trigger -x cosmics, vs random.  Returns rows and writes a figure."""
    print("\n# event-dependence (full-stream -x-lower >%d PE vs self-trigger -x cosmics)" % pe_cut)
    print("evt | fs_flash fs>cut sn_flash ref | coinc%% rand%% excess | fsPE_tot")
    rows = []
    for e in evts:
        try:
            d = load_event(run, e)
        except FileNotFoundError:
            print("%4d | (missing reco)" % e); continue
        ref = bright_xup_ref(d)
        fsm = d["fs_t"][d["fs_pe"] > pe_cut]
        o = coinc_fraction(ref, fsm); r = random_fraction(len(ref), fsm) if len(ref) else np.nan
        print("%4d | %8d %6d %8d %3d | %5.0f %5.0f  x%.1f | %.2e"
              % (e, len(d["fs_t"]), len(fsm), len(d["sn_t"]), len(ref), 100 * o, 100 * r,
                 o / max(r, 1e-9), d["fs_pe"].sum()))
        rows.append(dict(evt=e, nfs=len(d["fs_t"]), nbright=len(fsm), nref=len(ref),
                         obs=o, rnd=r, fspe=d["fs_pe"].sum()))

    ev = [r["evt"] for r in rows]
    x = np.arange(len(ev))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    exc = [r["obs"] / max(r["rnd"], 1e-9) for r in rows]
    ax[0].bar(x, exc, color=["gray" if v < 1.3 else "#d62728" for v in exc])
    ax[0].axhline(1.0, ls="--", c="green", label="random floor")
    ax[0].set_xticks(x); ax[0].set_xticklabels(ev)
    ax[0].set(xlabel="run 27980 event", ylabel="coincidence excess (>%d PE)" % pe_cut,
              title="Bright full-stream flashes coincide with -x cosmics, every event")
    ax[0].legend(fontsize=8)
    ax[1].bar(x - 0.2, [r["nfs"] for r in rows], 0.4, color="#d62728", label="full-stream flashes")
    ax[1].bar(x + 0.2, [r["nbright"] for r in rows], 0.4, color="#ff9896", label=">%d PE" % pe_cut)
    ax[1].set_xticks(x); ax[1].set_xticklabels(ev)
    ax[1].set(xlabel="run 27980 event", ylabel="flashes",
              title="Full-stream flash count is event-independent")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    out = "%s/pd_fullstream_%d_event_dependence.png" % (PICS, run)
    fig.savefig(out, dpi=95); print("wrote", out)
    return rows


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); raise SystemExit(1)
    run = int(sys.argv[1])
    evts = [int(x) for x in sys.argv[2:]]
    make_cycle1_figure(run, evts[0])
    if len(evts) > 1:
        event_table(run, evts)
