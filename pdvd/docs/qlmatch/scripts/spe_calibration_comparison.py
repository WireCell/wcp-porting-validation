#!/usr/bin/env python3
"""Compare PDVD SPE calibrations: our per-channel templates vs LArSoft.

Repro:
    cd pdvd/docs/qlmatch && python3 scripts/spe_calibration_comparison.py
    (writes spe_calibration_comparison.png + prints the tables quoted in
     10_pdvd-spe-calibration-comparison.md)

Sources compared, per DAPHNE offline channel:

  OURS      cfg/pgrapher/experiment/protodunevd/pdvd-spe-templates.json
            (run-039252 1-PE harvest, spe_build.py; ADC @ 16 ns ticks).
            amplitude = template max [ADC], area = template sum [ADC*ticks].
            Because OpDecon runs the F(0)=1 wiener-inspired filter, the
            template IS our PE scale: OpHit pe = decon area, and a real
            pulse of raw area A on a channel whose template area is S_tmpl
            yields pe ~= A / S_tmpl.  So "our SPE area" = S_tmpl.

  KEEPUP    dunesw v10_21_00d00+ standard_reco_protodunevd_keepup.fcl:
            protodunevd_ophit (AlgoSiPM on RAW waveform), flat
            SPEArea = 410 ADC*ticks for every channel, PE = area/410 + 0.43,
            UseCalibrator: false  (per-channel calibrator NOT used).

  PCAL      duneopdet v10_21_02d00 fcl/PhotonCalibratorDUNE.fcl
            (cvmfs, read 2026-07-14): six dated per-channel SPESizes sets
            [ADC*ticks], transcribed verbatim below.  Header: "STILL TO BE
            OBTAINED. DO NOT USE"; v300/v400 membrane values from finger
            plots (Henrique) "most likely underestimated by a factor 2-3";
            v400_2 cathode values from Ajib (indico 69308).  All 24 PMT
            (30xx) entries are placeholder 100 in every set.

  COLLEAGUE independent case-statement SPE *amplitudes* [ADC]
            (cathode 0.6*x, membrane 2010-2041 flat 50 = placeholder,
            2050-2081 0.6*x; no PMTs).

Units: both chains sample at 62.5 MHz (16 ns ticks), so ADC*ticks areas are
directly comparable.  Our chain applies decon before integrating while the
keepup integrates the raw waveform, but an SPE *calibration constant* is a
property of the raw pulse in both cases.
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_templates():
    rel = "pgrapher/experiment/protodunevd/pdvd-spe-templates.json"
    cands = [os.path.join(p, rel)
             for p in os.environ.get("WIRECELL_PATH", "").split(":") if p]
    # sibling checkout layout: <base>/wcp-porting-img/pdvd/... + <base>/toolkit/cfg
    cands.append(os.path.join(HERE, "../../../../../toolkit/cfg", rel))  # scripts/ adds one level
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(rel)


TEMPLATES = _find_templates()

# --- PhotonCalibratorDUNE.fcl (duneopdet v10_21_02d00), verbatim ---------
PCAL_CHANNELS = [
    1010, 1011, 1020, 1021, 1030, 1031, 1040, 1041,
    1050, 1051, 1060, 1061, 1070, 1071, 1080, 1081,
    2010, 2011, 2020, 2021, 2030, 2031, 2040, 2041,
    2050, 2051, 2060, 2061, 2070, 2071, 2080, 2081,
    3010, 3020, 3030, 3040, 3050, 3060, 3070, 3080,
    3090, 3100, 3110, 3120, 3130, 3140, 3150, 3160,
    3170, 3180, 3190, 3200, 3210, 3220, 3230, 3240]

PCAL_SETS = {
    # "Data taken between ???"  waffles/np02_data -> STILL TO BE OBTAINED. DO NOT USE
    "v100": [3482.9, 1893.97, 2148.87, 4332.16, 2063.53, 2000, 2257.92, 2530.51,
             5225.09, 5185.5, 2426.86, 2685.4, 3831.38, 3567.7, 1526.98, 1258.39,
             2000, 2000, 2000, 2000, 3147.79, 3278.53, 1380.77, 2205.66,
             4364.88, 4221.51, 3058.11, 3009.3, 2634.38, 2557.6, 3771.75, 3723.89]
            + [100] * 24,
    # July 10-16.  Same DO-NOT-USE comment.
    "v200": [3450.06, 1950.04, 2138.67, 4251.92, 2063.53, 2000, 2336.25, 2197.08,
             5225.09, 5185.5, 2426.86, 2685.4, 3831.38, 3567.7, 1526.98, 1258.39,
             2000, 2000, 2000, 2000, 2794.99, 2732.12, 2073.26, 1948.84,
             4302.94, 4255.39, 2900.88, 2826.3, 3238.6, 3074.78, 3669.16, 3723.37]
            + [100] * 24,
    # Aug 2-8, np02-config-v3.0.0.csv finger plot (Henrique),
    # "values most likely underestimated by a factor 2-3"
    "v300": [215.63, 121.88, 133.67, 265.75, 128.97, 125.00, 146.02, 137.32,
             326.57, 324.09, 151.68, 167.84, 239.46, 222.98, 95.44, 78.65,
             125.00, 125.00, 125.00, 125.00, 174.69, 170.76, 129.58, 121.80,
             268.93, 265.96, 181.31, 176.64, 202.41, 192.17, 229.32, 232.71]
            + [100] * 24,
    # Aug 10+, np02-config-v4.0.0.csv finger plot, same x2-3 caveat.
    "v400_1": [217.68, 118.37, 134.30, 270.76, 128.97, 125.00, 141.12, 158.16,
               326.57, 324.09, 151.68, 167.84, 239.46, 222.98, 95.44, 78.65,
               125.00, 125.00, 125.00, 125.00, 196.74, 204.91, 86.30, 137.85,
               272.81, 263.84, 191.13, 188.08, 164.65, 159.85, 235.73, 232.74]
              + [100] * 24,
    # Membrane from v4.0.0 finger plot (x2-3 caveat); CATHODE FROM AJIB
    # (indico 69308, Preliminary_LowPE_background_Study.pdf).
    "v400_2": [760.00, 501.00, 665.00, 794.00, 316.00, 290.00, 534.00, 551.00,
               488.00, 584.00, 408.00, 370.00, 660.00, 515.00, 380.00, 313.00,
               125.00, 125.00, 125.00, 125.00, 196.74, 204.91, 86.30, 137.85,
               272.81, 263.84, 191.13, 188.08, 164.65, 159.85, 235.73, 232.74]
              + [100] * 24,
}
KEEPUP_SPE_AREA = 410.0  # flat, opticaldetectormodules_dune.fcl protodunevd_ophit

# Placeholder values per set (entries that are NOT a measurement).
PCAL_PLACEHOLDER = {
    "v100": {2000.0, 100.0}, "v200": {2000.0, 100.0},
    "v300": {125.00, 100.0}, "v400_1": {125.00, 100.0}, "v400_2": {125.00, 100.0},
}

# --- colleague's independent SPE amplitudes [ADC] -------------------------
COLLEAGUE_AMP = {
    1010: 0.6 * 55.0, 1011: 0.6 * 26.5, 1020: 0.6 * 31.1, 1021: 0.6 * 43.0,
    1030: 0.6 * 31.1, 1031: 0.6 * 18.0, 1040: 0.6 * 30.8, 1041: 0.6 * 35.6,
    1050: 0.6 * 48.7, 1051: 0.6 * 47.2, 1060: 0.6 * 29.8, 1061: 0.6 * 29.4,
    1070: 0.6 * 55.0, 1071: 0.6 * 51.0, 1080: 0.6 * 23.9, 1081: 0.6 * 21.4,
    # 2010-2041 flat 50.0 = placeholder (excluded from ratio stats)
    2010: 50.0, 2011: 50.0, 2020: 50.0, 2021: 50.0,
    2030: 50.0, 2031: 50.0, 2040: 50.0, 2041: 50.0,
    2050: 0.6 * 41.2, 2051: 0.6 * 42.0, 2060: 0.6 * 42.8, 2061: 0.6 * 42.4,
    2070: 0.6 * 46.8, 2071: 0.6 * 44.8, 2080: 0.6 * 42.3, 2081: 0.6 * 41.6,
}
COLLEAGUE_PLACEHOLDER = {ch for ch, v in COLLEAGUE_AMP.items() if v == 50.0}

# Known-bad OUR templates (03_pdvd-spe-template.md sec.6): excluded from ratio
# statistics, kept in the table.  1051 = threshold artifact (area 8557),
# 2020 = distorted (negative area).
KNOWN_BAD = {1051, 2020}


def pd_type(ch):
    return {1: "cathode", 2: "membrane", 3: "pmt"}[ch // 1000]


def load_ours():
    d = json.load(open(TEMPLATES))
    out = {}
    for ch, idx in zip(d["channels"], d["template_index"]):
        t = d["templates"][idx]
        v = np.asarray(t["values"], dtype=float)
        out[ch] = dict(amp=v.max(), area=v.sum(),
                       popavg="_popavg" in t["name"], name=t["name"])
    return out


def median_ratio(pairs):
    r = np.array([a / b for a, b in pairs if a > 0 and b > 0])
    if len(r) == 0:
        return None
    return (np.median(r), np.percentile(r, 16), np.percentile(r, 84), len(r))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / den) if den else float("nan")


def main():
    ours = load_ours()
    pcal = {ch: {s: PCAL_SETS[s][i] for s in PCAL_SETS}
            for i, ch in enumerate(PCAL_CHANNELS)}

    # ---------------- per-channel table ----------------
    print("== per-channel table (areas ADC*ticks @16ns, amplitudes ADC) ==")
    hdr = (f"{'ch':>5} {'type':>8} | {'our_amp':>7} {'our_area':>8} {'pop':>3} |"
           f" {'colleag_amp':>11} | {'pcal_v300':>9} {'pcal_v400_2':>11} |"
           f" {'keepup':>6} | {'area/v400_2':>11}")
    print(hdr)
    for ch in PCAL_CHANNELS:
        o = ours.get(ch)
        oa = f"{o['amp']:7.1f}" if o else "   dead"
        oar = f"{o['area']:8.1f}" if o else "        "
        pop = ("pop" if o and o["popavg"] else "   ") if o else "   "
        ca = COLLEAGUE_AMP.get(ch)
        cas = (f"{ca:8.1f}{'(p)' if ch in COLLEAGUE_PLACEHOLDER else '   '}"
               if ca is not None else "        -  ")
        p3, p42 = pcal[ch]["v300"], pcal[ch]["v400_2"]
        p3s = f"{p3:9.1f}" if p3 not in PCAL_PLACEHOLDER["v300"] else f"{p3:7.1f}*p"
        p42s = (f"{p42:11.1f}" if p42 not in PCAL_PLACEHOLDER["v400_2"]
                else f"{p42:9.1f}*p")
        rat = (f"{o['area']/p42:11.2f}"
               if o and p42 not in PCAL_PLACEHOLDER["v400_2"] else "          -")
        print(f"{ch:>5} {pd_type(ch):>8} | {oa} {oar} {pop} | {cas} |"
              f" {p3s} {p42s} | {KEEPUP_SPE_AREA:6.0f} | {rat}")
    print("  (p)/*p = placeholder, excluded from stats below; pop = our template"
          " is the population-average fallback scaled to the channel 1-PE mode")

    # ---------------- block statistics ----------------
    print("\n== ratio statistics (medians [16,84], n) ==")

    def stat_line(label, pairs):
        s = median_ratio(pairs)
        if s is None:
            print(f"  {label}: no overlap")
            return
        med, lo, hi, n = s
        print(f"  {label}: median {med:.2f}  [{lo:.2f}, {hi:.2f}]  n={n}")

    for typ in ("cathode", "membrane"):
        chs = [ch for ch in PCAL_CHANNELS
               if pd_type(ch) == typ and ch in ours and ch not in KNOWN_BAD]

        # ours vs colleague (amplitudes)
        pairs = [(ours[ch]["amp"], COLLEAGUE_AMP[ch]) for ch in chs
                 if ch in COLLEAGUE_AMP and ch not in COLLEAGUE_PLACEHOLDER]
        stat_line(f"{typ:8s} our_amp / colleague_amp", pairs)
        if len(pairs) >= 4:
            print(f"  {'':8s}   spearman(our_amp, colleague_amp) = "
                  f"{spearman([p[0] for p in pairs], [p[1] for p in pairs]):.2f}")
        # same but dropping the colleague's 0.6 prefactor
        pairs06 = [(a, b / 0.6) for a, b in pairs]
        stat_line(f"{typ:8s} our_amp / colleague_raw (no 0.6)", pairs06)

        # ours vs each pcal set (areas)
        for s in ("v100", "v300", "v400_2"):
            pairs = [(ours[ch]["area"], pcal[ch][s]) for ch in chs
                     if pcal[ch][s] not in PCAL_PLACEHOLDER[s]]
            stat_line(f"{typ:8s} our_area / pcal_{s}", pairs)
            if len(pairs) >= 4:
                print(f"  {'':8s}   spearman = "
                      f"{spearman([p[0] for p in pairs], [p[1] for p in pairs]):.2f}")

        # ours vs keepup flat 410
        pairs = [(ours[ch]["area"], KEEPUP_SPE_AREA) for ch in chs]
        stat_line(f"{typ:8s} our_area / keepup_410", pairs)

    # PMTs: only the flat 410 exists on the LArSoft side
    chs = [ch for ch in PCAL_CHANNELS if pd_type(ch) == "pmt" and ch in ours]
    stat_line("pmt      our_area / keepup_410",
              [(ours[ch]["area"], KEEPUP_SPE_AREA) for ch in chs])
    print("  pmt      pcal: all 24 entries placeholder 100 in every set;"
          " colleague: not covered")

    # ---------------- shape factor + implied PE-scale correction ----------
    print("\n== template shape factor (area/amp = effective width, ticks) ==")
    for typ in ("cathode", "membrane", "pmt"):
        w = [ours[ch]["area"] / ours[ch]["amp"] for ch in ours
             if pd_type(ch) == typ and ours[ch]["amp"] > 0 and ours[ch]["area"] > 0]
        print(f"  {typ:8s}: median {np.median(w):.1f}  [{np.percentile(w,16):.1f},"
              f" {np.percentile(w,84):.1f}] ticks (n={len(w)})")

    print("\n== colleague amp -> implied area (amp x our per-channel width) "
          "vs pcal_v400_2, cathode ==")
    for ch in sorted(COLLEAGUE_AMP):
        if pd_type(ch) != "cathode" or ch not in ours:
            continue
        width = ours[ch]["area"] / ours[ch]["amp"]
        implied = COLLEAGUE_AMP[ch] * width
        p42 = pcal[ch]["v400_2"]
        print(f"  {ch}: colleague_amp {COLLEAGUE_AMP[ch]:5.1f} x width {width:5.1f}"
              f" = {implied:7.0f}  vs ajib {p42:6.0f}  (ratio {implied/p42:.2f})"
              f"   our_area {ours[ch]['area']:7.0f}")

    print("\n== implied multiplicative correction to OUR measured PE if "
          "pcal_v400_2 (Ajib) is the true cathode SPE area ==")
    print("   pe_true = pe_ours * (our_template_area / ajib_area)  -- because")
    print("   with F(0)=1 decon, pe_ours = raw_area/our_template_area.")
    corr = []
    for ch in PCAL_CHANNELS:
        if pd_type(ch) != "cathode" or ch not in ours:
            continue
        c = ours[ch]["area"] / pcal[ch]["v400_2"]
        note = "  (popavg template)" if ours[ch]["popavg"] else ""
        if ch in KNOWN_BAD:
            note += "  (KNOWN-BAD template, excluded from stats)"
        else:
            corr.append(c)
        print(f"  {ch}: x{c:.2f}{note}")
    print(f"  cathode overall: median x{np.median(corr):.2f}"
          f"  [{np.percentile(corr,16):.2f}, {np.percentile(corr,84):.2f}]")

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    for typ, m, c in (("cathode", "o", "tab:red"), ("membrane", "s", "tab:blue")):
        xs, ys, lbl = [], [], []
        for ch in PCAL_CHANNELS:
            if pd_type(ch) != typ or ch not in ours:
                continue
            p42 = pcal[ch]["v400_2"]
            if p42 in PCAL_PLACEHOLDER["v400_2"]:
                continue
            xs.append(p42); ys.append(ours[ch]["area"]); lbl.append(ch)
        ax.scatter(xs, ys, marker=m, color=c, label=typ)
        for x, y, l in zip(xs, ys, lbl):
            ax.annotate(str(l), (x, y), fontsize=6, alpha=0.7)
    lim = [0, 1400]
    ax.plot(lim, lim, "k--", lw=0.8, label="1:1")
    ax.plot(lim, [2 * v for v in lim], "k:", lw=0.8, label="x2 / x0.5")
    ax.plot(lim, [0.5 * v for v in lim], "k:", lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(0, 1600)
    ax.set_xlabel("PhotonCalibratorDUNE v400_2 SPE area [ADC*ticks]")
    ax.set_ylabel("our template area [ADC*ticks]")
    ax.set_title("SPE area: ours vs duneopdet v400_2\n(cathode=Ajib, membrane=finger plot /2-3)")
    ax.legend(fontsize=8)

    ax = axes[1]
    for typ, m, c in (("cathode", "o", "tab:red"), ("membrane", "s", "tab:blue")):
        xs, ys, lbl = [], [], []
        for ch, ca in COLLEAGUE_AMP.items():
            if pd_type(ch) != typ or ch not in ours or ch in COLLEAGUE_PLACEHOLDER:
                continue
            xs.append(ca); ys.append(ours[ch]["amp"]); lbl.append(ch)
        ax.scatter(xs, ys, marker=m, color=c, label=typ)
        for x, y, l in zip(xs, ys, lbl):
            ax.annotate(str(l), (x, y), fontsize=6, alpha=0.7)
    lim = [0, 70]
    ax.plot(lim, lim, "k--", lw=0.8, label="1:1")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("colleague SPE amplitude [ADC]")
    ax.set_ylabel("our template amplitude [ADC]")
    ax.set_title("SPE amplitude: ours vs colleague case statement")
    ax.legend(fontsize=8)

    ax = axes[2]
    cath = [ch for ch in PCAL_CHANNELS if pd_type(ch) == "cathode" and ch in ours]
    x = np.arange(len(cath))
    ax.bar(x - 0.2, [ours[ch]["area"] for ch in cath], width=0.4,
           color="tab:red", label="our template area")
    ax.bar(x + 0.2, [pcal[ch]["v400_2"] for ch in cath], width=0.4,
           color="tab:green", label="Ajib v400_2")
    ax.axhline(KEEPUP_SPE_AREA, color="k", ls="--", lw=0.8, label="keepup flat 410")
    ax.set_xticks(x); ax.set_xticklabels([str(c) for c in cath],
                                         rotation=90, fontsize=7)
    ax.set_ylabel("SPE area [ADC*ticks]")
    ax.set_title("cathode SPE areas per channel")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(HERE), "pics", "spe_calibration_comparison.png")
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
