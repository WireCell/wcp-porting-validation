#!/usr/bin/env python3
"""Harvest single-PE candidate pulses from the PDVD light data and build
data-driven SPE templates per DAPHNE channel (+ per-population averages).

Self-trigger channels (membrane XA 20xx, PMT 30xx): use the 1024-sample
snippets.  The per-channel amplitude spectrum shows a clean low-amplitude mode
(membrane XA ~32-60 ADC = 1 PE; PMT ~90-160 ADC = SPE peak).  Select isolated,
unsaturated pulses in a band around that mode, flat pre-peak baseline, align
on the peak sample and average -> mean 1-PE response in raw ADC.  That raw-ADC
scale IS the PE normalization for OpDecon (AutoScale + OpHitFinder
area/spe_area convention).

Cathode XA channels (10xx, full-stream): no self-trigger snippets.  Find small
isolated pulses in the 468k-sample stream (local pedestal), same selection,
same averaging.

Outputs (cache, git-ignored):
  work/light_spe/harvest_r<run>.npz   per-channel amplitude arrays + selected
                                      pulse stacks + averaged templates
Plots (committed):
  docs/pds/spe_amplitude_spectra.png  per-channel spectra with the 1-PE band

Usage: spe_build.py [file_index ...]   (default 0 = run 039252)
"""
import os
import sys

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pdvd_light as pl

WORK = os.path.join(pl.PDVD, "work", "light_spe")

# template window around the aligned peak (16 ns ticks)
PRE, POST = 60, 400          # -0.96 us .. +6.4 us (covers slow XA recovery)
BAND = (0.75, 1.30)          # accepted amplitude band around the 1-PE mode
ISO_PROM_FRAC = 0.5          # a second peak above this fraction of mode -> reject
BASE_STD_MAX = 3.0           # head std < this x channel noise rms
MIN_PULSES = 10              # min accepted pulses to keep a channel template

# full-stream small-pulse search
FS_SEED_NSIGMA = 5.0

# cathode template tail repair: start of the fitted region (ticks after peak)
REPAIR_T0 = 20


def one_pe_mode(amps, lo=10, hi=250, bins=60):
    """Mode of the low-amplitude spectrum = 1-PE amplitude estimate."""
    a = amps[(amps > lo) & (amps < hi)]
    if len(a) < 20:
        return None
    h, e = np.histogram(a, bins=bins, range=(lo, hi))
    # smooth lightly to be robust against binning noise
    k = np.convolve(h, [1, 2, 3, 2, 1], mode="same")
    return 0.5 * (e[np.argmax(k)] + e[np.argmax(k) + 1])


def snippet_harvest(f, quiet_rms):
    """-> {ch: {amps, stacks, npass}} from all self-trigger snippets."""
    out = {}
    for evt in f.events:
        for r in f.records(evt, snippets_only=True):
            ch, adc = r["opch"], r["adc"]
            d = out.setdefault(ch, {"amps": [], "wf": []})
            ped = pl.pedestal_head(adc)
            pk = int(np.argmax(adc))
            d["amps"].append(adc[pk] - ped)
            d["wf"].append((adc, ped, pk))
    return out


def select_and_average(wfs, mode, rms):
    """Apply the 1-PE selection to (adc, ped, pk) snippets; average aligned."""
    stack = []
    for adc, ped, pk in wfs:
        amp = adc[pk] - ped
        if not (BAND[0] * mode <= amp <= BAND[1] * mode):
            continue
        if adc.max() >= pl.ADC_RAIL:
            continue
        if pk < PRE or pk + POST > len(adc):
            continue
        if adc[:40].std() > BASE_STD_MAX * max(rms, 1.0):
            continue
        w = adc - ped
        # isolation: no second comparable pulse in the template window.
        # boxcar-smooth first (kills white-noise prominence on the noisy top-
        # membrane channels) and use prominence (not height) so wiggles on the
        # broad XA pulse top do not count as extra peaks.
        ws = np.convolve(w[pk - PRE:pk + POST], np.ones(5) / 5, mode="same")
        pks, props = find_peaks(ws, prominence=ISO_PROM_FRAC * mode,
                                distance=4)
        big = [p for p in pks if abs(p - PRE) > 8]
        if big:
            continue
        stack.append(w[pk - PRE:pk + POST])
    return combine_stack(stack)


def hpf(w, tau_mhz=0.05):
    """OpRoi-style high-pass: H(f) = 1 - exp(-(f/tau)^2).  Removes the slow
    baseline wander of the full-stream records (corner ~20 us) without
    touching the ~1 us XA pulse."""
    n = len(w)
    fr = np.fft.rfftfreq(n, d=TICK_US)  # MHz (tick in us)
    return np.fft.irfft(np.fft.rfft(w) * (1.0 - np.exp(-(fr / tau_mhz) ** 2)),
                        n=n)


TICK_US = pl.TICK_NS / 1000.0


def combine_stack(stack):
    """mean for the peak region, per-sample median beyond +100 ticks (robust
    to residual pileup in the tail); also split-half templates (odd/even
    pulses) for out-of-sample tests.
    -> (tmpl, n, tmpl_a, tmpl_b)"""
    if len(stack) < MIN_PULSES:
        return None, len(stack), None, None

    def comb(st):
        st = np.asarray(st)
        t = st.mean(axis=0)
        t[PRE + 100:] = np.median(st[:, PRE + 100:], axis=0)
        return t

    return (comb(stack), len(stack),
            comb(stack[0::2]), comb(stack[1::2]))


def fullstream_harvest(f, max_events=None):
    """Small isolated pulses from the cathode full-stream channels.

    Pulses are FOUND and isolation-tested on the high-passed trace (the raw
    stream has low-frequency wander that fakes broad bumps), but the template
    window is extracted from the RAW trace with a local pre-peak pedestal, so
    the stored shape is the true SPE response, not the HPF'd one.
    -> {ch: {amps, wf(list of (raw window, hp window)), rms}}"""
    out = {}
    events = f.events if max_events is None else f.events[:max_events]
    for evt in events:
        for r in f.records(evt, fullstream_only=True):
            ch, adc = r["opch"], r["adc"]
            med, _ = pl.pedestal_robust(adc)
            # quiet-noise estimate: adjacent-sample differences are immune to
            # the sparse pulses and the wander that inflate the plain MAD
            dsig = np.median(np.abs(np.diff(adc))) * 1.4826 / np.sqrt(2)
            d = out.setdefault(ch, {"amps": [], "wf": [], "rms": []})
            d["rms"].append(dsig)
            w = adc - med
            hp = hpf(w)
            pks, props = find_peaks(hp, height=FS_SEED_NSIGMA * dsig,
                                    distance=25)
            for p in pks:
                if p < PRE + 50 or p + POST + 50 > len(w):
                    continue
                raw = w[p - PRE - 50:p + POST + 50]
                # amplitude relative to the local raw pre-peak baseline
                base = raw[:50].mean()
                d["amps"].append(raw[PRE + 50] - base)
                d["wf"].append((raw, hp[p - PRE - 50:p + POST + 50]))
    return out


def repair_fs_tail(tmpl):
    """Cathode template tail repair.

    The local pre-peak baselines of the 1-PE harvest windows sit on slowly
    falling tails of earlier activity in the busy stream, biasing the whole
    extracted window low by a near-constant (~-3 ADC).  That spurious
    negative tail (~-14% of the 1-PE peak, per sample, over 6 us) corrupts
    the decon kernel at low frequency: bright pulses deconvolve to a
    non-decaying positive plateau (~12% of the prompt spike) instead of
    returning to baseline like the membrane XA.

    Bright-pulse medians (n~800-1100/ch, amplitude-normalized, 50 us
    windows) show the TRUE response: a positive slow component (tau ~1.2 us)
    and NO measurable undershoot (<5e-4 of prompt beyond 25 us) -- the AC
    coupling is invisible on these timescales.

    Repair: fit a*exp(-t/tau)+c beyond +REPAIR_T0 ticks and keep only the
    exponential (c = the baseline bias).  Validated on ch1010: decon
    tail/peak of >=8 PE pulses drops 0.115 -> 0.020, at/below the membrane
    late-light benchmark (0.026-0.075), flat vs amplitude.
    -> (repaired template, fit params (a, tau_ticks, c) or None)"""
    if tmpl is None:
        return None, None
    x = np.arange(len(tmpl), dtype=float) - PRE
    xi, yi = x[PRE + REPAIR_T0:], tmpl[PRE + REPAIR_T0:]

    def f(t, a, tau, c):
        return a * np.exp(-t / tau) + c

    try:
        popt, _ = curve_fit(f, xi, yi,
                            p0=(max(float(yi[:20].mean()), 1.0), 60.0,
                                float(yi[-60:].mean())),
                            bounds=([0, 5, -np.inf], [np.inf, 500, np.inf]),
                            maxfev=20000)
    except Exception:
        return tmpl, None
    out = tmpl.copy()
    out[PRE + REPAIR_T0:] = popt[0] * np.exp(-xi / popt[1])
    return out, popt


def fs_select_and_average(d, mode, rms):
    stack = []
    pk = PRE + 50
    for raw, hp in d["wf"]:
        base = raw[:50].mean()
        amp = raw[pk] - base
        if not (BAND[0] * mode <= amp <= BAND[1] * mode):
            continue
        # flat local baseline: reject windows on wander slopes or pileup tails
        if raw[:50].std() > BASE_STD_MAX * max(rms, 1.0):
            continue
        # isolation on the smoothed HPF trace
        hs = np.convolve(hp, np.ones(5) / 5, mode="same")
        pks, props = find_peaks(hs, prominence=ISO_PROM_FRAC * mode,
                                distance=4)
        big = [q for q in pks if abs(q - pk) > 8]
        if big:
            continue
        stack.append(raw[pk - PRE:pk + POST] - base)
    return combine_stack(stack)


def main():
    idx = [int(a) for a in sys.argv[1:]] or [0]
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(pl.PICS, exist_ok=True)

    for i in idx:
        f = pl.LightFile(pl.data_files()[i])
        print(f"== {os.path.basename(f.path)} run {f.run} "
              f"({len(f.events)} events)")

        # --- self-trigger channels
        snip = snippet_harvest(f, None)
        results = {}
        for ch in sorted(snip):
            amps = np.array(snip[ch]["amps"])
            # channel noise rms from snippet heads
            rms = np.median([adc[:40].std() for adc, _, _ in snip[ch]["wf"]])
            mode = one_pe_mode(amps)
            if mode is None:
                print(f"  ch{ch}: no usable 1-PE mode (n={len(amps)})")
                continue
            tmpl, npass, ta, tb = select_and_average(snip[ch]["wf"], mode, rms)
            results[ch] = dict(amps=amps, mode=mode, rms=rms,
                               tmpl=tmpl, npass=npass, tmpl_a=ta, tmpl_b=tb)
            status = "OK" if tmpl is not None else "TOO FEW"
            print(f"  ch{ch}: mode={mode:6.1f} rms={rms:5.2f} "
                  f"selected={npass:4d} {status}")

        # --- cathode full-stream channels
        fs = fullstream_harvest(f)
        for ch in sorted(fs):
            amps = np.array(fs[ch]["amps"])
            rms = float(np.median(fs[ch]["rms"]))
            mode = one_pe_mode(amps, lo=max(10, 3 * rms))
            if mode is None:
                print(f"  ch{ch}: no usable 1-PE mode (n={len(amps)})")
                continue
            tmpl, npass, ta, tb = fs_select_and_average(fs[ch], mode, rms)
            tmpl, tailfit = repair_fs_tail(tmpl)
            ta, _ = repair_fs_tail(ta)
            tb, _ = repair_fs_tail(tb)
            results[ch] = dict(amps=amps, mode=mode, rms=rms,
                               tmpl=tmpl, npass=npass, tmpl_a=ta, tmpl_b=tb)
            if tailfit is not None:
                results[ch]["tailfit"] = np.asarray(tailfit)
            status = "OK" if tmpl is not None else "TOO FEW"
            fitmsg = ("" if tailfit is None else
                      f"  tail fit a={tailfit[0]:.2f} "
                      f"tau={tailfit[1] * pl.TICK_NS / 1000:.2f}us "
                      f"bias c={tailfit[2]:+.2f}")
            print(f"  ch{ch}: mode={mode:6.1f} rms={rms:5.2f} "
                  f"selected={npass:4d} {status}{fitmsg}")

        np.savez_compressed(
            os.path.join(WORK, f"harvest_r{f.run:06d}.npz"),
            **{f"ch{ch}_{k}": v for ch, d in results.items()
               for k, v in d.items() if v is not None})

        plot_spectra(f.run, results)


def plot_spectra(run, results):
    chs = sorted(results)
    ncol = 6
    nrow = (len(chs) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow))
    for ax, ch in zip(axes.flat, chs):
        d = results[ch]
        a = d["amps"]
        ax.hist(a[(a > 0) & (a < 300)], bins=75, range=(0, 300),
                histtype="stepfilled", alpha=0.7)
        ax.axvspan(BAND[0] * d["mode"], BAND[1] * d["mode"],
                   color="r", alpha=0.15)
        ax.axvline(d["mode"], color="r", lw=1)
        ax.set_title(f"ch{ch}  1PE~{d['mode']:.0f} ADC  n={d['npass']}",
                     fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(chs):]:
        ax.axis("off")
    fig.suptitle(f"PDVD run {run}: snippet/stream peak-amplitude spectra "
                 f"(red = selected 1-PE band)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(pl.PICS, "spe_amplitude_spectra.png")
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()
