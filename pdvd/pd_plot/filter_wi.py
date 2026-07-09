#!/usr/bin/env python3
"""Wiener-INSPIRED software filter for the PDVD light deconvolution.

The fixed-snr Wiener filter G = conj(H)R/(|H|^2 R + 1) has a DC gain that
depends on R and the template spectrum, so the deconvolved area (= PE scale)
moves with the filter knobs and must be patched up by AutoScale.  Instead,
follow the WCT signal-processing convention: deconvolve with the pure
inverse M/H and apply a *Wiener-inspired* band filter

    F(f) = exp(-0.5 (f/sigma)^power),  F(0) = 1 exactly,

with (sigma, power) fitted to the actual Wiener x Gauss-postfilter band
shape per population.  F(0)=1 keeps the 1-PE decon area at exactly 1
independent of the filter parameters (no AutoScale needed), and sigma is a
single physical high-frequency cutoff that can be trimmed on noisy
populations (membrane-top pickup) without touching the normalization.

Per channel (own-template channels + real quiet-noise records):
  1. reference band = |G H| x postfilter at the milestone-2 settings
     (fixed_snr 0.005; cutoff 1.5 MHz XA / 3.0 MHz PMT);
  2. fit (sigma, power); aggregate per population (median);
  3. metrics for Wiener vs WI at sigma scale factors 0.5..1.2:
     noise floor (real quiet record), held-out 1-PE peak, significance,
     matched-impulse FWHM / sidelobe, raw (un-rescaled) 1-PE decon area
     -- the normalization-stability demonstration.

Outputs: docs/pds/filter_wi.png; tables on stdout.
Usage: filter_wi.py [run]   (default 39252; needs spe_build.py harvest)
"""
import os
import sys

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pdvd_light as pl
import spe_build as sb
import spe_compare as sc
import filter_scan as fsc

N = 4096
FIXED_SNR = 0.005
CUTOFF = {"cathode": 1.5, "membrane_top": 1.5, "membrane_bot": 1.5,
          "pmt": 3.0}
SCALES = (0.5, 0.6, 0.7, 0.8, 1.0, 1.2)
POPS = fsc.POPS

# decision (see pdvd-light-filter.md): power 2, sigma trimmed below the fit
# where the real noise rewards it (membrane top 2.0->3.9, bottom 7.0->9.0;
# cathode/PMT flat).  All in MHz.
CHOSEN = {"cathode": 1.25, "membrane_top": 1.0, "membrane_bot": 1.0,
          "pmt": 3.5}
CHOSEN_P = 2.0


def wi(f, sigma, power):
    return np.exp(-0.5 * (np.abs(f) / sigma) ** power)


def band_reference(ta, rms, cutoff, n=N):
    """|G H| x postfilter band shape (rfft bins) of the fixed-snr Wiener."""
    H = np.fft.fft(np.pad(ta.astype(float), (0, n - len(ta)))[:n])
    N2 = rms ** 2 * n
    S2 = FIXED_SNR * N2
    W = np.abs(H) ** 2 * S2 / (np.abs(H) ** 2 * S2 + N2)
    PF = np.abs(pl.build_postfilter(n, cutoff))
    return (W * PF)[:n // 2 + 1], H


def fit_wi(band, n=N):
    fr = np.fft.rfftfreq(n, d=pl.TICK_NS * 1e-3)      # MHz
    m = (fr > 0) & (band > 1e-4)
    popt, _ = curve_fit(wi, fr[m], band[m], p0=(1.0, 2.0),
                        bounds=([0.05, 0.5], [31.0, 10.0]), maxfev=20000)
    return popt


def wi_operator(H, sigma, power, n=N):
    """Pure-inverse x WI-filter decon spectrum F/H (guarded where H ~ 0)."""
    fr = np.fft.fftfreq(n, d=pl.TICK_NS * 1e-3)
    F = wi(fr, sigma, power)
    aH2 = np.abs(H) ** 2
    eps = (1e-3 * np.abs(H).max()) ** 2
    return F * np.conj(H) / (aH2 + eps)


def impulse(OP, H, n=N):
    """decon of the template with itself -> the filter impulse response."""
    k = np.fft.ifft(OP * H).real
    return np.roll(k, n // 2)


def metrics(OP, H, probe, noise, norm=1.0, n=N):
    wave = np.zeros(n)
    wave[100:100 + len(probe)] = probe
    dec = np.fft.ifft(OP * np.fft.fft(wave)).real
    area_raw = float(dec.sum())              # BEFORE any rescaling
    dec /= norm
    pk = float(dec[100 + sb.PRE - 40:100 + sb.PRE + 15].max())
    nz = np.fft.ifft(OP * np.fft.fft(noise[:n])).real / norm
    floor = float(nz.std())
    k = impulse(OP, H, n) / norm
    k0 = int(np.argmax(k))
    above = np.nonzero(k > 0.5 * k[k0])[0]
    fwhm = float(above[-1] - above[0]) if len(above) else np.nan
    side = 100.0 * float(k[k0:k0 + 300].min() / k[k0])
    return dict(floor=floor, pk=pk, sig=pk / floor if floor > 0 else np.nan,
                fwhm=fwhm, side=side, area_raw=area_raw)


def main():
    run = int(sys.argv[1]) if len(sys.argv) > 1 else 39252
    harvest = {ch: h for ch, h in sc.load_harvest(run).items()
               if "tmpl_a" in h}
    src = [p for p in pl.data_files() if f"run{run:06d}" in p][0]
    f = pl.LightFile(src)
    chpop = {}
    for evt in f.events[:1]:
        for r in f.records(evt):
            if r["opch"] in harvest:
                chpop[r["opch"]] = r["pop"]
    noise = fsc.collect_noise(f, list(harvest))

    # ---- 1+2: per-channel fit, per-population (sigma, power)
    fits, bands, Hs = {}, {}, {}
    for ch, h in harvest.items():
        if ch not in noise:
            continue
        pop = chpop[ch]
        ta = sc.as_kernel(h["tmpl_a"])
        band, H = band_reference(ta, max(float(h["rms"]), 1.0), CUTOFF[pop])
        try:
            fits[ch] = fit_wi(band)
        except Exception as e:
            print(f"ch{ch}: fit failed ({e})")
            continue
        bands[ch], Hs[ch] = band, H
    pop_par = {}
    for pop in POPS:
        chs = [c for c in fits if chpop[c] == pop]
        if not chs:
            continue
        sig = float(np.median([fits[c][0] for c in chs]))
        pw = float(np.median([fits[c][1] for c in chs]))
        pop_par[pop] = (sig, pw)
        print(f"{pop:14s} ({len(chs):2d} ch): sigma={sig:5.2f} MHz  "
              f"power={pw:4.2f}   per-ch sigma "
              f"{min(fits[c][0] for c in chs):.2f}-"
              f"{max(fits[c][0] for c in chs):.2f}")

    # ---- 3: Wiener baseline vs WI at sigma scales (pop-level params)
    rows = {}          # rows[pop][label] = median metric dict
    for pop, (sig0, pw) in pop_par.items():
        chs = [c for c in fits if chpop[c] == pop]
        per = {lbl: [] for lbl in
               ["wiener"] + [f"wi x{s:g}" for s in SCALES]}
        for c in chs:
            h = harvest[c]
            ta = sc.as_kernel(h["tmpl_a"])
            rms = max(float(h["rms"]), 1.0)
            GF, norm = fsc.filter_response(ta, rms, FIXED_SNR,
                                           CUTOFF[pop], N)
            per["wiener"].append(
                metrics(GF, Hs[c], h["tmpl_b"], noise[c], norm))
            for s in SCALES:
                OP = wi_operator(Hs[c], s * sig0, pw)
                per[f"wi x{s:g}"].append(
                    metrics(OP, Hs[c], h["tmpl_b"], noise[c]))
        rows[pop] = {lbl: {k: float(np.median([m[k] for m in ms]))
                           for k in ms[0]}
                     for lbl, ms in per.items() if ms}
        print(f"\n== {pop}  (WI sigma={sig0:.2f} MHz power={pw:.2f}, "
              f"{len(chs)} ch)")
        print("   filter     floor    1PE_pk  signif  FWHM  side%  raw_area")
        for lbl, m in rows[pop].items():
            print(f"  {lbl:9s} {m['floor']:8.4f} {m['pk']:8.4f} "
                  f"{m['sig']:6.1f} {m['fwhm']:5.0f} {m['side']:+6.1f} "
                  f"  {m['area_raw']:6.3f}")

    # ---- chosen per-population parameters
    print(f"\nchosen (power={CHOSEN_P}):")
    for pop, sig in CHOSEN.items():
        chs = [c for c in fits if chpop[c] == pop]
        if not chs:
            continue
        ms = []
        for c in chs:
            OP = wi_operator(Hs[c], sig, CHOSEN_P)
            ms.append(metrics(OP, Hs[c], harvest[c]["tmpl_b"], noise[c]))
        med = {k: float(np.median([m[k] for m in ms])) for k in ms[0]}
        print(f"  {pop:14s} sigma={sig:4.2f} MHz: floor={med['floor']:.4f} "
              f"1PE={med['pk']:.4f} signif={med['sig']:.1f} "
              f"FWHM={med['fwhm']:.0f} side={med['side']:+.2f}%")

    # ---- normalization-stability demo: raw 1-PE area vs fixed_snr (Wiener)
    print("\nraw (pre-AutoScale) 1-PE decon area vs fixed_snr "
          "(one mid channel per population); WI stays at its fitted value:")
    for pop in pop_par:
        chs = sorted(c for c in fits if chpop[c] == pop)
        c = chs[len(chs) // 2]
        h = harvest[c]
        ta = sc.as_kernel(h["tmpl_a"])
        rms = max(float(h["rms"]), 1.0)
        vals = []
        for R in (0.001, 0.002, 0.005, 0.01, 0.02):
            H = Hs[c]
            N2 = rms ** 2 * N
            S2 = R * N2
            G = np.conj(H) * S2 / (np.abs(H) ** 2 * S2 + N2)
            GF = G * pl.build_postfilter(N, CUTOFF[pop])
            vals.append(metrics(GF, H, h["tmpl_b"], noise[c])["area_raw"])
        OP = wi_operator(H, *pop_par[pop])
        awi = metrics(OP, H, h["tmpl_b"], noise[c])["area_raw"]
        print(f"  {pop:14s} ch{c}: wiener {min(vals):.3f}-{max(vals):.3f}"
              f"   WI {awi:.3f} (any sigma/power)")

    plot(run, pop_par, fits, bands, chpop, rows, harvest, Hs)


def plot(run, pop_par, fits, bands, chpop, rows, harvest, Hs):
    fr = np.fft.rfftfreq(N, d=pl.TICK_NS * 1e-3)
    fig, axes = plt.subplots(3, len(pop_par),
                             figsize=(4.5 * len(pop_par), 11), squeeze=False)
    for j, (pop, (sig0, pw)) in enumerate(pop_par.items()):
        chs = [c for c in fits if chpop[c] == pop]
        ax = axes[0][j]
        med = np.median([bands[c] for c in chs], axis=0)
        ax.semilogy(fr, med, "k-", lw=1.2,
                    label=f"Wiener x post (snr {FIXED_SNR}, "
                          f"{CUTOFF[pop]} MHz)")
        ax.semilogy(fr, wi(fr, sig0, pw), "C1-", lw=1.2,
                    label=f"WI fit: sigma={sig0:.2f}, p={pw:.2f}")
        ax.semilogy(fr, wi(fr, CHOSEN[pop], CHOSEN_P), "C2--", lw=1.2,
                    label=f"WI chosen: sigma={CHOSEN[pop]:g}, p={CHOSEN_P:g}")
        ax.set_xlim(0, min(4 * sig0 + 2, 31))
        ax.set_ylim(1e-4, 2)
        ax.set_title(pop)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        ax.set_xlabel("MHz")
        if j == 0:
            ax.set_ylabel("band filter (F(0)=1)")

        ax = axes[1][j]
        sigs = [rows[pop][f"wi x{s:g}"]["sig"] for s in SCALES]
        ax.plot(SCALES, sigs, "o-", ms=4, label="WI")
        ax.axhline(rows[pop]["wiener"]["sig"], color="k", lw=1, ls="--",
                   label="Wiener baseline")
        ax.axvline(CHOSEN[pop] / sig0, color="C2", lw=1, ls="--",
                   label=f"chosen {CHOSEN[pop]:g} MHz")
        ax.set_xlabel("sigma scale factor")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        if j == 0:
            ax.set_ylabel("1-PE significance (median)")

        ax = axes[2][j]
        c = sorted(chs)[len(chs) // 2]
        h = harvest[c]
        rms = max(float(h["rms"]), 1.0)
        ta = sc.as_kernel(h["tmpl_a"])
        GF, norm = fsc.filter_response(ta, rms, FIXED_SNR, CUTOFF[pop], N)
        kw = impulse(GF, Hs[c]) / norm
        kwi = impulse(wi_operator(Hs[c], sig0, pw), Hs[c])
        p0 = int(np.argmax(kw))
        p1 = int(np.argmax(kwi))
        tt = np.arange(-40, 120)
        ax.plot(tt, kw[p0 - 40:p0 + 120], "k-", lw=1, label="Wiener+post")
        ax.plot(tt, kwi[p1 - 40:p1 + 120], "C1-", lw=1, label="WI")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("ticks from impulse peak")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        if j == 0:
            ax.set_ylabel(f"matched impulse (ch{c})")

    fig.suptitle(f"PDVD run {run}: Wiener-inspired filter "
                 f"F=exp(-0.5(f/sigma)^p), F(0)=1 -- fit, significance vs "
                 f"sigma trim, impulse response")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(pl.PICS, "filter_wi.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
