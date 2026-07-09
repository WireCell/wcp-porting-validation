#!/usr/bin/env python3
"""SPE-template try-and-compare + per-PD deconvolution demonstration.

For every live DAPHNE channel produce docs/pds/pd_ch<CCCC>.png:
  row 0: the data-driven SPE template (raw ADC) | peak-amplitude spectrum
  rows 1-3: raw waveform | deconvolved waveform for a small / medium / large
            isolated example pulse (the user's acceptance criterion is a flat
            decon baseline).

Template try-and-compare per population (docs/pds/spe_template_compare.png):
deconvolve each channel's OUT-OF-SAMPLE average 1-PE pulse (split-half B,
template built from half A) with each candidate template and measure the
post-pulse tail residual beyond the matched-kernel floor
(pdhd/docs/pdhd-spe-template-tuning.md convention).

Candidates: data-driven (own channel, split A), population average,
SPE_NP04_FBK_2024, SPE_NP04_HPK_2024, SPE_NP02_estimate (XA only).

Initial filter settings = PDHD production values (fixed_snr=0.005,
1.5 MHz Gauss post-filter) with the measured per-channel noise RMS; the
dedicated filter scan is a separate step (filter_scan.py).

Usage: spe_compare.py [run]   (default 39252; reads work/light_spe/harvest_*.npz)
"""
import os
import sys

import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pdvd_light as pl
import spe_build as sb

FIXED_SNR = 0.005
CUTOFF = 1.5     # MHz
PRE, POST = sb.PRE, sb.POST

# Stored harvest templates carry PRE pre-peak samples; as a decon kernel the
# template must follow the NP04 "without_pretrigger" convention (peak at
# sample ~10), else every decon spike shifts early by the pre length.
TRIM = PRE - 10


def as_kernel(tmpl):
    return tmpl[TRIM:]

# example-pulse amplitude bands in units of the channel 1-PE mode
BANDS = {"small": (0.75, 1.3), "medium": (3.0, 8.0), "large": (15.0, 1e9)}


def load_harvest(run):
    d = np.load(os.path.join(sb.WORK, f"harvest_r{run:06d}.npz"))
    out = {}
    for k in d.files:
        ch, key = int(k[2:6]), k[7:]
        out.setdefault(ch, {})[key] = d[k]
    return out


def decon(wave, tmpl, rms, n=None, pedestal=0.0):
    """decon with the study filter settings (wave already baseline-subtracted,
    positive-going)."""
    return pl.deconvolve(wave, tmpl, fixed_snr=FIXED_SNR,
                         line_noise_rms=max(rms, 1.0), cutoff_mhz=CUTOFF,
                         pedestal=pedestal, polarity=+1, n=n)


def kernel(tmpl, rms):
    return pl.matched_kernel(tmpl, fixed_snr=FIXED_SNR,
                             line_noise_rms=max(rms, 1.0), cutoff_mhz=CUTOFF,
                             pre=PRE, post=POST)


def tail_metric(dec_shape, kern, peak_idx, rms_floor):
    """post-pulse residual 0.5-4 us after the peak, as % of the decon peak,
    relative to the matched-kernel expectation."""
    s0 = peak_idx + int(0.5 * 1000 / pl.TICK_NS)
    s1 = peak_idx + int(4.0 * 1000 / pl.TICK_NS)
    pk = dec_shape[peak_idx]
    if pk <= 0:
        return np.nan
    d = dec_shape[s0:s1] / pk
    k0 = int(np.argmax(kern))
    k = kern[k0 + (s0 - peak_idx):k0 + (s1 - peak_idx)]
    m = min(len(d), len(k))
    return 100.0 * float(np.mean(d[:m] - k[:m]))


# ------------------------------------------------------------ example pulses
def find_examples(f, harvest):
    """3 example pulses (small/medium/large) per channel.
    Snippets: whole snippet + its decon.  Full-stream: isolated raw window."""
    ex = {ch: {} for ch in harvest}
    for evt in f.events:
        for r in f.records(evt):
            ch = r["opch"]
            if ch not in harvest:
                continue
            h = harvest[ch]
            mode = float(h["mode"])
            adc = r["adc"]
            if r["nsamp"] <= 1024:
                ped = pl.pedestal_head(adc)
                pk = int(np.argmax(adc))
                amp = adc[pk] - ped
                if adc.max() >= pl.ADC_RAIL or pk < PRE or pk + POST > len(adc):
                    continue
                if adc[:40].std() > sb.BASE_STD_MAX * max(float(h["rms"]), 1.0):
                    continue
                w = adc - ped
                cand = ("snip", w, pk)
            else:
                med, _ = pl.pedestal_robust(adc)
                w = adc - med
                hp = sb.hpf(w)
                dsig = float(h["rms"])
                pks, _ = find_peaks(hp, height=sb.FS_SEED_NSIGMA * dsig,
                                    distance=25)
                for p in pks:
                    if p < 2 * PRE or p + 3 * POST > len(w):
                        continue
                    if adc[p - 5:p + 5].max() >= pl.ADC_RAIL:
                        continue
                    raw = w[p - 2 * PRE:p + 3 * POST].copy()
                    base = raw[:PRE].mean()
                    # clean demo pulse: flat pre-baseline, no comparable
                    # earlier pulse in the window (later late-light is physics)
                    if raw[:PRE].std() > sb.BASE_STD_MAX * max(dsig, 1.0):
                        continue
                    hs = np.convolve(hp[p - 2 * PRE:p + 3 * POST],
                                     np.ones(5) / 5, mode="same")
                    amp0 = raw[2 * PRE] - base
                    early, _ = find_peaks(hs[:2 * PRE - 8],
                                          prominence=0.3 * amp0, distance=4)
                    if len(early):
                        continue
                    # small/medium demos additionally require an otherwise
                    # quiet window (the cathode stream is busy); large pulses
                    # keep their genuine late light
                    if amp0 < BANDS["large"][0] * mode:
                        # quiet over the DISPLAYED window (-PRE..+POST) only
                        other, _ = find_peaks(hs[PRE:2 * PRE + POST],
                                              prominence=0.3 * amp0,
                                              distance=4)
                        if [q for q in other if abs(q - PRE) > 10]:
                            continue
                    classify(ex[ch], raw - base, 2 * PRE, amp0, mode)
                continue
            classify(ex[ch], cand[1], cand[2], w[pk], mode)
    return ex


def classify(slot, wave, pk, amp, mode):
    for name, (lo, hi) in BANDS.items():
        if lo * mode <= amp <= hi * mode:
            # keep the largest-amplitude example of each class -> most visible
            if name not in slot or amp > slot[name][2]:
                slot[name] = (wave, pk, amp)
            return


# ------------------------------------------------------------------ per-PD fig
def per_pd_figure(ch, h, ex, run):
    tmpl = h["tmpl"]
    rms = float(h["rms"])
    mode = float(h["mode"])
    fig = plt.figure(figsize=(11, 11))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    tt = (np.arange(len(tmpl)) - PRE) * pl.TICK_NS / 1000.0
    ax.plot(tt, tmpl, lw=1)
    ax.axhline(0, color="k", lw=0.5)
    src = ("population-average FALLBACK" if h.get("fallback")
           else f"data-driven, n={int(h['npass'])}")
    ax.set_title(f"SPE template ({src}, 1PE={mode:.0f} ADC)", fontsize=9)
    ax.set_xlabel("us from peak"); ax.set_ylabel("ADC")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    a = h["amps"]
    ax.hist(a[(a > 0) & (a < 8 * mode)], bins=80)
    ax.axvspan(sb.BAND[0] * mode, sb.BAND[1] * mode, color="r", alpha=0.15)
    ax.set_title("peak-amplitude spectrum (red = 1-PE band)", fontsize=9)
    ax.set_xlabel("ADC"); ax.grid(alpha=0.3)

    for row, name in enumerate(("small", "medium", "large"), start=1):
        axr = fig.add_subplot(gs[row, 0])
        axd = fig.add_subplot(gs[row, 1])
        if name not in ex:
            axr.text(0.5, 0.5, f"no {name} example", ha="center",
                     transform=axr.transAxes)
            axr.axis("off"); axd.axis("off")
            continue
        wave, pk, amp = ex[name]
        dec = decon(wave, as_kernel(tmpl), rms)
        lo, hi = max(0, pk - PRE), min(len(wave), pk + POST)
        tt = (np.arange(lo, hi) - pk) * pl.TICK_NS / 1000.0
        axr.plot(tt, wave[lo:hi], lw=0.8)
        axr.axhline(0, color="k", lw=0.5)
        axr.set_title(f"raw ({name}, {amp / mode:.1f} PE)", fontsize=9)
        axr.set_ylabel("ADC - baseline"); axr.grid(alpha=0.3)
        axd.plot(tt, dec[lo:hi], lw=0.8, color="C1")
        axd.axhline(0, color="k", lw=0.5)
        axd.set_title(f"deconvolved ({name})", fontsize=9)
        axd.set_ylabel("PE / tick"); axd.grid(alpha=0.3)
        if row == 3:
            axr.set_xlabel("us from peak"); axd.set_xlabel("us from peak")

    fig.suptitle(f"PDVD run {run}  DAPHNE ch{ch} "
                 f"(OpDet-type {'cathode XA' if ch < 2000 else 'membrane XA' if ch < 3000 else 'PMT'})",
                 fontsize=11)
    out = os.path.join(pl.PICS, f"pd_ch{ch}.png")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


# --------------------------------------------------------------- try-compare
def compare_templates(harvest, run):
    cv = pl.cvmfs_templates()
    pops = {"cathode": [], "membrane": [], "pmt": []}
    for ch in sorted(harvest):
        if harvest[ch].get("tmpl") is None or "tmpl_a" not in harvest[ch]:
            continue
        pop = ("cathode" if ch < 2000 else
               "membrane" if ch < 3000 else "pmt")
        pops[pop].append(ch)

    # population-average templates (peak-aligned, from split A, mean of
    # peak-normalized shapes scaled back to the channel 1-PE amplitude)
    popavg = {}
    for pop, chs in pops.items():
        if chs:
            popavg[pop] = np.mean(
                [harvest[c]["tmpl_a"] / max(harvest[c]["tmpl_a"].max(), 1e-9)
                 for c in chs], axis=0)

    results = {}
    for pop, chs in pops.items():
        cands = {"data(own)": None, "data(pop-avg)": None}
        if pop in ("cathode", "membrane"):
            cands.update({"NP04_FBK_2024": cv["NP04_FBK_2024"],
                          "NP04_HPK_2024": cv["NP04_HPK_2024"],
                          "NP02_estimate": cv["NP02_estimate"]})
        for ch in chs:
            h = harvest[ch]
            probe = h["tmpl_b"]            # held-out average 1-PE pulse
            rms = float(h["rms"])
            n = 4096
            wave = np.zeros(n)
            wave[100:100 + len(probe)] = probe
            for cname, ct in cands.items():
                if cname == "data(own)":
                    ct = as_kernel(h["tmpl_a"])
                elif cname == "data(pop-avg)":
                    ct = as_kernel(popavg[pop] * h["tmpl_a"].max())
                dec = decon(wave, ct, rms, n=n)
                # decon spike sits near the probe peak minus the kernel's
                # own pre-peak offset; search a generous window
                pk = 100 + PRE
                lo = pk - 40
                pkw = int(np.argmax(dec[lo:pk + 15])) + lo
                kern = kernel(ct, rms)
                m = tail_metric(dec, kern, pkw, rms)
                results.setdefault(pop, {}).setdefault(cname, {})[ch] = m
    plot_compare(results, run)
    return results


def plot_compare(results, run):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, pop in zip(axes, ("cathode", "membrane", "pmt")):
        if pop not in results:
            ax.axis("off")
            continue
        cands = results[pop]
        names = list(cands)
        chs = sorted(next(iter(cands.values())))
        xx = np.arange(len(chs))
        wdt = 0.8 / len(names)
        for i, cname in enumerate(names):
            vals = [cands[cname].get(c, np.nan) for c in chs]
            ax.bar(xx + i * wdt, vals, wdt, label=cname)
        ax.set_xticks(xx + 0.4)
        ax.set_xticklabels([str(c) for c in chs], rotation=90, fontsize=7)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{pop}: held-out 1-PE decon tail residual "
                     f"(0.5-4 us, % of peak)", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"PDVD run {run}: SPE-template candidates, out-of-sample "
                 f"decon-tail residual vs matched kernel (0 = perfect)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(pl.PICS, "spe_template_compare.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out)


def plot_populations(harvest, run):
    """Committed summary: peak-normalized template overlay per population."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, pop, lohi in zip(axes, ("cathode XA 10xx", "membrane XA 20xx",
                                    "PMT 30xx"), (1000, 2000, 3000)):
        for ch in sorted(harvest):
            h = harvest[ch]
            if h.get("fallback") or not (lohi <= ch < lohi + 1000):
                continue
            t = h["tmpl"]
            tt = (np.arange(len(t)) - PRE) * pl.TICK_NS / 1000.0
            ax.plot(tt, t / max(t.max(), 1e-9), lw=0.8, alpha=0.7,
                    label=f"ch{ch}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(pop + " (peak-normalized)", fontsize=10)
        ax.set_xlabel("us from peak"); ax.grid(alpha=0.3)
        ax.legend(fontsize=5, ncol=2)
    fig.suptitle(f"PDVD run {run}: data-driven SPE templates by population")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(pl.PICS, "spe_templates_by_population.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out)


def main():
    run = int(sys.argv[1]) if len(sys.argv) > 1 else 39252
    harvest = load_harvest(run)
    nown = sum(1 for h in harvest.values() if "tmpl" in h)
    print(f"{len(harvest)} live channels, {nown} with own data-driven template")

    # candidate comparison (channels with own template only)
    with_tmpl = {ch: h for ch, h in harvest.items() if "tmpl" in h}
    results = compare_templates(with_tmpl, run)
    for pop, cands in results.items():
        print(f"\n== {pop}: mean |tail residual| %")
        for cname, vals in cands.items():
            v = np.array([x for x in vals.values() if np.isfinite(x)])
            print(f"  {cname:16s} {np.mean(np.abs(v)):6.2f} "
                  f"(median {np.median(v):+6.2f})")

    # population-average fallback for the low-statistics channels
    popavg = {}
    for pop, lohi in (("cathode", 1000), ("membrane", 2000), ("pmt", 3000)):
        shapes = [h["tmpl"] / max(h["tmpl"].max(), 1e-9)
                  for ch, h in with_tmpl.items() if lohi <= ch < lohi + 1000]
        if shapes:
            popavg[pop] = np.mean(shapes, axis=0)
    for ch, h in harvest.items():
        if "tmpl" in h or "mode" not in h:
            continue
        pop = ("cathode" if ch < 2000 else
               "membrane" if ch < 3000 else "pmt")
        h["tmpl"] = popavg[pop] * float(h["mode"])
        h["fallback"] = True
        print(f"  ch{ch}: population-average fallback template "
              f"(1PE mode {float(h['mode']):.0f} ADC, own pulses "
              f"n={int(h.get('npass', 0))})")

    plot_populations(harvest, run)

    # per-PD demonstration figures
    src = [p for p in pl.data_files() if f"run{run:06d}" in p][0]
    f = pl.LightFile(src)
    ex = find_examples(f, harvest)
    for ch in sorted(harvest):
        out = per_pd_figure(ch, harvest[ch], ex.get(ch, {}), run)
        print("wrote", out)

    # machine-readable summary for the doc / future jsonnet generation
    import json
    summary = {}
    for ch in sorted(harvest):
        h = harvest[ch]
        entry = dict(one_pe_adc=float(h["mode"]), noise_rms=float(h["rms"]),
                     n_selected=int(h.get("npass", 0)),
                     fallback=bool(h.get("fallback", False)))
        for pop, cands in results.items():
            for cname, vals in cands.items():
                if ch in vals and np.isfinite(vals[ch]):
                    entry.setdefault("tail_residual_pct", {})[cname] = \
                        round(float(vals[ch]), 3)
        summary[str(ch)] = entry
    outj = os.path.join(pl.PICS, "spe_summary.json")
    json.dump(summary, open(outj, "w"), indent=1)
    print("wrote", outj)


if __name__ == "__main__":
    main()
