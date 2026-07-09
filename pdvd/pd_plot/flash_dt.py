#!/usr/bin/env python3
"""Flash time-coincidence (Delta_t) study across the four PDVD PD groups:
top-wall membrane XA / bottom-wall membrane XA / cathode XA / PMT.

Goal: measure how tightly the four populations' hit times agree for the same
physical activity, to justify the OpFlashFinder coincidence bin width
(`bin_width`, PDHD = 1000 ns) and expose per-population systematic offsets.

Clock (verified): the `timestamp` branch is in MICROSECONDS on one common
clock, quantized at the 16 ns DAPHNE tick (8 ns substructure); the
full-stream record starts at its timestamp; snippet timestamps are the
trigger times (bright cathode stream pulses line up with snippet-trigger
clusters at the same microsecond values).

Hits:
  - snippets: one hit per snippet; time = timestamp + (decon spike - 64
    presumed trigger sample) x 16 ns; PE = decon area is overkill here, use
    amplitude/mode.  Constant per-population offsets do not matter for the
    window choice and are REPORTED by the pair medians.
  - cathode stream: HPF peaks >= 5 sigma; time = timestamp + sample x 16 ns.

Pairs: cathode<->each other population (the double-sided cathode XAs see
both volumes -> global reference), pmt<->membrane_bot (same volume),
membrane_top<->membrane_bot (cross-volume, via cathode-crossing tracks).

Output: docs/pds/flash_dt.png + quantile table on stdout.
Usage: flash_dt.py [file_index ...]   (default 0 = run 039252)
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
import spe_compare as sc

TRIG_SAMPLE = 64          # nominal self-trigger crossing sample (peak ~76)
PE_MIN_REF = 3.0          # cathode reference hits
PE_MIN = 0.7              # other-population hits
PAIRS = (("cathode", "membrane_top"), ("cathode", "membrane_bot"),
         ("cathode", "pmt"), ("pmt", "membrane_bot"),
         ("membrane_top", "membrane_bot"))


def event_hits(f, evt, harvest):
    """-> {pop: [(t_us, pe), ...]} for one event."""
    hits = {p: [] for p in ("cathode", "membrane_top", "membrane_bot", "pmt")}
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
            pe = (adc[pk] - ped) / mode
            if pe < PE_MIN or adc.max() >= pl.ADC_RAIL:
                continue
            dec = sc.decon(adc - ped, sc.as_kernel(h["tmpl"]),
                           max(float(h["rms"]), 1.0))
            lo = max(0, pk - 40)
            spike = int(np.argmax(dec[lo:pk + 15])) + lo
            t = r["timestamp"] + (spike - TRIG_SAMPLE) * pl.TICK_NS / 1000.0
            hits[r["pop"]].append((t, pe))
        else:
            med, _ = pl.pedestal_robust(adc)
            hp = sb.hpf(adc - med)
            dsig = np.median(np.abs(np.diff(adc))) * 1.4826 / np.sqrt(2)
            pks, _ = find_peaks(hp, height=sb.FS_SEED_NSIGMA * dsig,
                                distance=25)
            for p in pks:
                pe = hp[p] / mode
                if pe < PE_MIN:
                    continue
                t = r["timestamp"] + p * pl.TICK_NS / 1000.0
                hits["cathode"].append((t, pe))
    return hits


def pair_dts(hits, ref_pop, oth_pop, window_us=5.0):
    """Delta_t = t(other) - t(nearest reference hit), within +-window."""
    ref = np.array(sorted(t for t, pe in hits[ref_pop]
                          if pe >= (PE_MIN_REF if ref_pop == "cathode"
                                    else PE_MIN)))
    if not len(ref):
        return []
    out = []
    for t, pe in hits[oth_pop]:
        i = np.searchsorted(ref, t)
        best = min((abs(t - ref[j]), t - ref[j])
                   for j in (max(0, i - 1), min(len(ref) - 1, i)))[1]
        if abs(best) <= window_us:
            out.append(best)
    return out


def main():
    idx = [int(a) for a in sys.argv[1:]] or [0]
    f = pl.LightFile(pl.data_files()[idx[0]])
    run = f.run
    harvest = sc.load_harvest(run if run != 39252 else 39252)
    # ensure every live channel has a usable template (fallbacks like
    # spe_compare) -- only mode/rms/tmpl are needed here
    popavg = {}
    with_tmpl = {ch: h for ch, h in harvest.items() if "tmpl" in h}
    for pop, lohi in (("cathode", 1000), ("membrane", 2000), ("pmt", 3000)):
        shapes = [h["tmpl"] / max(h["tmpl"].max(), 1e-9)
                  for ch, h in with_tmpl.items() if lohi <= ch < lohi + 1000]
        popavg[pop] = np.mean(shapes, axis=0)
    for ch, h in harvest.items():
        if "tmpl" not in h and "mode" in h:
            pop = ("cathode" if ch < 2000 else
                   "membrane" if ch < 3000 else "pmt")
            h["tmpl"] = popavg[pop] * float(h["mode"])

    dts = {pair: [] for pair in PAIRS}
    for evt in f.events:
        hits = event_hits(f, evt, harvest)
        for pair in PAIRS:
            dts[pair].extend(pair_dts(hits, pair[0], pair[1]))

    # table + figure
    fig, axes = plt.subplots(2, len(PAIRS), figsize=(3.6 * len(PAIRS), 7))
    print(f"run {run}: nearest-neighbour Delta_t per population pair (us)")
    print("  pair                                n    median   68% half-width"
          "   95% half-width")
    for j, pair in enumerate(PAIRS):
        d = np.array(dts[pair])
        name = f"{pair[1]} - {pair[0]}"
        if len(d) < 10:
            print(f"  {name:34s} {len(d):5d}   (too few)")
            continue
        med = np.median(d)
        q68 = 0.5 * (np.percentile(d, 84) - np.percentile(d, 16))
        q95 = 0.5 * (np.percentile(d, 97.5) - np.percentile(d, 2.5))
        print(f"  {name:34s} {len(d):5d}  {med:+7.3f}   {q68:8.3f}"
              f"        {q95:8.3f}")
        for i, (rng, bins) in enumerate(((5.0, 100), (1.0, 100))):
            ax = axes[i][j]
            ax.hist(d[np.abs(d) < rng], bins=bins)
            ax.axvline(0, color="k", lw=0.5)
            ax.set_xlabel("Delta_t (us)")
            if i == 0:
                ax.set_title(name, fontsize=9)
        axes[1][j].set_xlim(-1, 1)
    fig.suptitle(f"PDVD run {run}: hit-time coincidence across PD groups "
                 f"(top: +-5 us, bottom: +-1 us)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(pl.PICS, "flash_dt.png")
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()
