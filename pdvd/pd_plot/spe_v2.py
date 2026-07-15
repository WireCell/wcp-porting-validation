#!/usr/bin/env python3
"""Validated SPE templates v2 for the PDVD light chain.

Repro:  cd pdvd/pd_plot && python3 spe_v2.py          # run 39252 harvest
Output: toolkit cfg/pgrapher/experiment/protodunevd/pdvd-spe-templates-v2.json
        (v1 file untouched; flash.jsonnet selects via spe_v2, runner env
        PDVD_SPE_V2 -- see docs/qlmatch/pdvd-lightpattern-sp-investigation.md)

Why: OpDecon's wiener-inspired PE normalization is 1/template-area, so a
corrupted template area is a corrupted measured-PE scale.  The v1 harvest
(spe_build.py + make_light_json.py) shipped four bad templates:
  ch1051  area  8557 (~7x cathode peers)   -> OpDet 7 reads ~x2 low
  ch2020  area -1595 (NEGATIVE: averaged noise self-triggers, AC undershoot)
  ch3010  area 15935 (~25x PMT peers)      -> OpDet 14 reads ~x10 low
  ch3020  area  8479 (~8x PMT peers)       -> OpDet 15 reads ~x5 low

v2 policy, per channel:
  1. VALIDATE the v1 template two ways:
     (a) area > 0 AND area/amplitude within RATIO_BAND x the population
         median (population = cathode / membrane / PMT);
     (b) ganged (XA) channels only: the harvest 1-PE mode ratio to the
         sibling must agree with the coincident bright-pulse amplitude
         ratio (= the true gain ratio) within PAIR_TOL -- catches a
         multi-PE mode latch (ch1051: mode 76 vs sibling-implied ~28)
         that a pure shape test cannot see.
  2. Invalid + ganged (XA pairs): replace with the population-average shape
     scaled to a SIBLING-TRANSFERRED 1-PE amplitude
         A = mode(sibling) x median(coincident bright-pulse amp ratio)
     -- the two sub-channels of one OpDet see identical light, so the
     coincident-pulse amplitude ratio IS the gain ratio.  (ch1051's own
     "1-PE mode" of ~76 ADC is a multi-PE latch: the bright-pulse ratio to
     ch1050 is ~1, so its true 1-PE amplitude is ~mode(1050).)
  3. Invalid + single (PMTs): population-average shape scaled to the
     channel's own 1-PE mode (the standard popavg fallback -- the mode is
     credible even where the averaged template tail is contaminated).
  4. Valid channels are copied bit-for-bit from v1 (including popavg
     fallbacks), so v2 == v1 everywhere except the repaired channels.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pdvd_light as pl          # noqa: E402
import spe_build as sb           # noqa: E402
import spe_compare as sc         # noqa: E402
from make_light_json import final_templates, CFG_DIR  # noqa: E402

RATIO_BAND = 2.5      # area/amp within [1/RATIO_BAND, RATIO_BAND] x pop median
PAIR_TOL = 1.5        # harvest-mode ratio vs bright-pulse gain ratio tolerance
PAIR_MIN_AMP = 500.0  # coincident-pulse selection threshold (ADC over ped)
PAIR_MAX_AMP = 12000.0  # stay below rail on both channels
PAIR_MIN_N = 10       # min coincident pulses for a usable gain ratio
RAWWF = os.path.join(os.path.dirname(HERE), "input_data_light_rawwf")


def pop_of(ch):
    return "cathode" if ch < 2000 else ("membrane" if ch < 3000 else "pmt")


def sibling_of(ch):
    """Ganged partner (XA pairs 10x0/10x1, 20x0/20x1); None for PMTs."""
    if ch >= 3000:
        return None
    return ch + 1 if ch % 10 == 0 else ch - 1


class RawFile:
    """Direct reader of the rawdump/raw_waveform TTree (all events).
    Timestamps are microseconds on a common clock (pd_mapping_audit)."""

    def __init__(self, path):
        import uproot
        self.tree = uproot.open(path)["rawdump/raw_waveform"]
        s = self.tree.arrays(["event", "opchannel", "timestamp"], library="np")
        self.event = s["event"]
        self.opch = s["opchannel"]
        self.ts = s["timestamp"]
        self.events = sorted(set(int(e) for e in self.event))

    def records(self, evt, chans):
        idx = np.where(self.event == evt)[0]
        out = {}
        for i in idx:
            c = int(self.opch[i])
            if c not in chans:
                continue
            adc = np.asarray(self.tree["adc"].array(
                entry_start=i, entry_stop=i + 1, library="np")[0], np.float32)
            out.setdefault(c, []).append((float(self.ts[i]), adc))
        return out


def pair_gain_ratio(f, cha, chb):
    """median amplitude ratio cha/chb over coincident bright pulses.

    Snippet channels: match snippets by timestamp (self-triggers fire
    together on real light).  Full-stream channels: find peaks on one
    stream, read the other at the same sample.
    """
    from scipy.signal import find_peaks
    ratios = []
    for evt in f.events:
        recs = f.records(evt, {cha, chb})
        if cha not in recs or chb not in recs:
            continue
        if len(recs[cha][0][1]) > 4096:        # full streams
            wa = recs[cha][0][1]
            wb = recs[chb][0][1]
            pa = np.median(wa); pb = np.median(wb)
            pks, _ = find_peaks(wa - pa, height=PAIR_MIN_AMP, distance=200)
            for p in pks:
                aa = wa[p] - pa
                bb = (wb[max(0, p - 3):p + 4] - pb).max()
                if PAIR_MIN_AMP < aa < PAIR_MAX_AMP and PAIR_MIN_AMP < bb < PAIR_MAX_AMP:
                    ratios.append(aa / bb)
        else:                                  # self-trigger snippets
            tb = {round(t, 1): w for t, w in recs[chb]}
            for t, a in recs[cha]:
                b = tb.get(round(t, 1))
                if b is None:
                    continue
                aa = a.max() - pl.pedestal_head(a)
                bb = b.max() - pl.pedestal_head(b)
                if (PAIR_MIN_AMP < aa < PAIR_MAX_AMP and
                        PAIR_MIN_AMP < bb < PAIR_MAX_AMP and
                        a.max() < pl.ADC_RAIL and b.max() < pl.ADC_RAIL):
                    ratios.append(aa / bb)
    return (float(np.median(ratios)), len(ratios)) if len(ratios) >= PAIR_MIN_N \
        else (None, len(ratios))


def main():
    run = int(sys.argv[1]) if len(sys.argv) > 1 else 39252
    harvest = sc.load_harvest(run)
    tmpls = final_templates(harvest)   # ch -> (template, is_popavg_fallback)

    # channels with their own harvested template, same construction as
    # make_light_json/final_templates (popavg itself is built AFTER the
    # validations so the bad channels' contaminated tails stay out of it)
    with_tmpl = {ch: h for ch, h in harvest.items() if "tmpl" in h}

    # --- validation: area/amp vs population median
    stats = {ch: (float(t.sum()), float(t.max())) for ch, (t, _) in tmpls.items()}
    med = {}
    for pop in ("cathode", "membrane", "pmt"):
        r = [a / m for ch, (a, m) in stats.items()
             if pop_of(ch) == pop and a > 0 and m > 0]
        r = [x for x in r if x < 10 * np.median(r)]   # loose pre-cut
        med[pop] = float(np.median(r))
    print("population median area/amp:", {k: round(v, 1) for k, v in med.items()})

    bad = []
    for ch, (area, amp) in sorted(stats.items()):
        ratio = area / amp if amp > 0 else -1
        ok = area > 0 and med[pop_of(ch)] / RATIO_BAND <= ratio <= med[pop_of(ch)] * RATIO_BAND
        if not ok:
            bad.append(ch)
            print(f"  ch{ch}: INVALID shape (area {area:.0f}, amp {amp:.1f}, "
                  f"area/amp {ratio:.1f} vs pop median {med[pop_of(ch)]:.1f})")

    # --- validation (b): ganged-pair 1-PE mode consistency vs the
    # coincident bright-pulse gain ratio (catches multi-PE mode latches).
    import glob as _glob
    src = sorted(_glob.glob(os.path.join(
        RAWWF, f"np02vd_raw_run{run:06d}_*_rawwf.root")))[0]
    f = RawFile(src)
    pair_r = {}
    for ch in sorted(harvest):
        sib = sibling_of(ch)
        if sib is None or sib < ch or sib not in harvest:
            continue
        if not ("mode" in harvest[ch] and "mode" in harvest[sib]):
            continue
        r, n = pair_gain_ratio(f, ch, sib)
        pair_r[(ch, sib)] = (r, n)
        if r is None:
            continue
        mode_r = float(harvest[ch]["mode"]) / float(harvest[sib]["mode"])
        agree = 1.0 / PAIR_TOL < (mode_r / r) < PAIR_TOL
        print(f"  pair {ch}/{sib}: bright gain ratio {r:.3f} (n={n}) "
              f"mode ratio {mode_r:.3f} -> {'OK' if agree else 'MISMATCH'}")
        if not agree:
            # A mode latch always OVERestimates the 1-PE amplitude (the
            # finder lands on a multi-PE peak), so only the HIGH-side member
            # is the corrupted one; flagging both would also disqualify the
            # healthy sibling as a repair reference.
            for c, other, rr in ((ch, sib, r), (sib, ch, 1.0 / r)):
                implied = float(harvest[other]["mode"]) * rr
                if float(harvest[c]["mode"]) / implied >= PAIR_TOL:
                    if c not in bad:
                        bad.append(c)
                        print(f"  ch{c}: INVALID amplitude (mode "
                              f"{harvest[c]['mode']:.1f} vs sibling-implied "
                              f"{implied:.1f})")
    print("invalid channels:", sorted(bad))

    # population-average repair shapes (peak-normalized), valid channels only
    popavg = {}
    for pop, lo in (("cathode", 1000), ("membrane", 2000), ("pmt", 3000)):
        shapes = [h["tmpl"] / max(h["tmpl"].max(), 1e-9)
                  for ch, h in with_tmpl.items()
                  if lo <= ch < lo + 1000 and ch not in bad]
        popavg[pop] = np.mean(shapes, axis=0)

    repaired = {}
    for ch in sorted(bad):
        pop = pop_of(ch)
        sib = sibling_of(ch)
        own_mode = float(harvest[ch]["mode"]) if "mode" in harvest[ch] else None
        A, how = None, ""
        if sib is not None and sib in harvest and "mode" in harvest[sib] and sib not in bad:
            ratio, n = pair_r.get((min(ch, sib), max(ch, sib)), (None, 0))
            if ratio is not None and ch > sib:
                ratio = 1.0 / ratio     # stored as amp(lo)/amp(hi); want ch/sib
            if ratio is None:
                ratio, n = pair_gain_ratio(f, ch, sib)
            if ratio is not None:
                A = float(harvest[sib]["mode"]) * ratio
                how = (f"sibling transfer: mode(ch{sib})={harvest[sib]['mode']:.1f}"
                       f" x pair ratio {ratio:.3f} (n={n})")
                if own_mode:
                    how += f"; own mode {own_mode:.1f} " + (
                        "CONSISTENT" if 0.67 < own_mode / A < 1.5 else
                        "REJECTED (multi-PE latch / noise mode)")
        if A is None and own_mode:
            A = own_mode
            how = f"own 1-PE mode {own_mode:.1f} (no usable sibling pair)"
        if A is None:
            peers = [float(harvest[c]["mode"]) for c in harvest
                     if pop_of(c) == pop and "mode" in harvest[c] and c != ch]
            A = float(np.median(peers))
            how = f"population-median mode {A:.1f} (last resort)"
        repaired[ch] = (popavg[pop] * A, how)
        print(f"  ch{ch}: repaired -> popavg({pop}) x {A:.1f} ADC "
              f"[area {repaired[ch][0][sc.TRIM:].sum():.0f}]  ({how})")

    # --- write v2 json: valid channels bit-for-bit from the CURRENT v1 file
    v1 = json.load(open(os.path.join(CFG_DIR, "pdvd-spe-templates.json")))
    v1_of = {ch: v1["templates"][ti]
             for ch, ti in zip(v1["channels"], v1["template_index"])}
    templates, channels, tindex = [], [], []
    for ch in v1["channels"]:
        if ch in repaired:
            kern = repaired[ch][0][sc.TRIM:]
            entry = {"name": f"SPE_PDVD_r{run:06d}_ch{ch}_v2repair",
                     "values": [round(float(v), 4) for v in kern]}
        else:
            entry = v1_of[ch]
        templates.append(entry)
        channels.append(ch)
        tindex.append(len(templates) - 1)
    spe = {
        "comment": (v1["comment"]
                    + "  V2: per-channel validation (area>0, area/amp within "
                    f"{RATIO_BAND}x the population median; ganged-pair 1-PE "
                    "mode consistent with the coincident bright-pulse gain "
                    "ratio); invalid channels "
                    "({}) rebuilt as popavg x a validated 1-PE amplitude "
                    "(ganged channels: sibling mode x coincident bright-pulse "
                    "gain ratio; single PMTs: own mode).  All other channels "
                    "bit-for-bit == pdvd-spe-templates.json.  spe_v2.py; "
                    "docs/qlmatch/pdvd-lightpattern-sp-investigation.md."
                    ).format(",".join(str(c) for c in repaired)),
        "source": v1["source"],
        "templates": templates,
        "channels": channels,
        "template_index": tindex,
    }
    out = os.path.join(CFG_DIR, "pdvd-spe-templates-v2.json")
    with open(out, "w") as fp:
        json.dump(spe, fp, indent=1)
    print(f"wrote {out}")

    print("\nold vs new areas (repaired channels):")
    for ch in repaired:
        old = float(np.sum(v1_of[ch]["values"]))
        new = float(np.sum(spe["templates"][channels.index(ch)]["values"]))
        print(f"  ch{ch}: {old:9.1f} -> {new:8.1f}")


if __name__ == "__main__":
    main()
