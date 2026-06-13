#!/usr/bin/env python3
"""Can we calibrate the PDHD SPE response from the data itself and beat the
2024 HPK average template?  Short answer (this dataset): no.

This isolates single-photoelectron pulses from the converted raw snippets,
averages them (cross-correlation aligned), compares the result to the 2024
HPK average and the v1 per-channel template, and runs all of them through the
deconvolution to measure the late-tail behaviour.  Produces
spe_calibration_study.png.

Run: python3 make_spe_calibration_study.py
(needs pdhd/work/<run>_<evt>/light-frames-wct.tar.bz2 for the 4 example
events, and the toolkit cfg JSONs / numpy OpDecon in make_perchannel_plots.py)
"""
import io
import json
import os
import sys
import tarfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_perchannel_plots as M  # validated numpy OpDecon

sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/toolkit/flash/test")
import extract_pdhd_spe_templates as E

PDHD = os.path.abspath(os.path.join(HERE, "..", ".."))
WK = os.environ.get("PDHD_WORK_BASE", os.path.join(PDHD, "work"))
CFG = M.CFG
HPK = set(E.HPK_CHANNELS)
N = 1024
PRE = 15
EVTS = [("027305_150", "150"), ("027980_8", "8"),
        ("028084_74408", "74408"), ("029107_983", "983")]


def load_frames(path):
    out = {}
    with tarfile.open(path) as tf:
        for m in tf.getmembers():
            if m.name.endswith(".npy"):
                out[m.name] = np.load(io.BytesIO(tf.extractfile(m).read()))
    return out


def all_snippets():
    """Yield (channel, raw_adc[1024]) for every HPK snippet in the 4 events."""
    for wd, ev in EVTS:
        fr = load_frames(f"{WK}/{wd}/light-frames-wct.tar.bz2")
        raw, ch = fr[f"frame_raw_{ev}.npy"], fr[f"channels_raw_{ev}.npy"]
        for i, c in enumerate(ch):
            c = int(c)
            if c not in HPK:
                continue
            row = raw[i]
            nz = np.nonzero(row)[0]
            if not len(nz):
                continue
            starts = [nz[0]]
            for j in nz[1:]:
                if j >= starts[-1] + N:
                    starts.append(j)
            for s in starts:
                w = row[s:s + N]
                if len(w) == N:
                    yield c, w.astype(float)


def extract_spe(band, W=80, seed=None):
    """Cross-correlation-aligned average of isolated pulses with peak in `band`."""
    cand = []
    for c, w in all_snippets():
        ped = w[:20].mean()
        xv = -1.0 * (w - ped)
        sig = np.std(xv[:20])
        sig = sig if sig > 1 else 4.5
        for k in range(25, N - W - 5):
            v = xv[k]
            if not (band[0] <= v <= band[1]):
                continue
            if not (v >= xv[k - 1] and v > xv[k + 1]):
                continue
            if np.abs(xv[k - 15:k - 7].mean()) > 1.0 * sig or xv[k - 15:k - 7].std() > 1.3 * sig:
                continue
            if xv[k - 12:k + 13].max() > v:          # isolation
                continue
            cand.append(xv[k - PRE:k + W - PRE])
    cand = np.array(cand)
    refn = seed - seed.mean()
    refn /= np.linalg.norm(refn)
    acc = np.zeros(W)
    n = 0
    for c in cand:
        best, bsh = -9, 0
        for sh in range(-6, 7):
            seg = c[PRE - 9 + sh:PRE - 9 + sh + len(seed)]
            if len(seg) < len(seed):
                continue
            sn = seg - seg.mean()
            nn = np.linalg.norm(sn)
            if nn == 0:
                continue
            cc = np.dot(sn / nn, refn)
            if cc > best:
                best, bsh = cc, sh
        if best < 0.6:
            continue
        seg = c[bsh:bsh + W]
        if len(seg) == W:
            acc += seg
            n += 1
    avg = acc / max(n, 1)
    avg -= avg[:8].mean()
    return avg, n, len(cand)


def fwhm(t):
    ip = int(np.argmax(t[:40]))
    half = t[ip] / 2
    l, r = ip, ip
    while l > 0 and t[l] > half:
        l -= 1
    while r < len(t) - 1 and t[r] > half:
        r += 1
    return r - l, ip


def mk(t):
    o = np.zeros(N)
    o[:min(N, len(t))] = t[:N]
    return o


def deconv(adc, tmpl):
    return M.deconvolve(adc, np.fft.fft(tmpl), max(1.0, tmpl.max()), None)


def main():
    spe2024 = json.load(open(f"{HERE}/pdhd-spe-templates-2024.json"))
    hpk2024 = np.array(spe2024["templates"][1]["values"])
    spev1 = json.load(open(f"{CFG}/pdhd-spe-templates.json"))
    c2t = {c: t for c, t in zip(spev1["channels"], spev1["template_index"])}
    seed = hpk2024[:45] - hpk2024[:45].mean()

    # --- SPE average per amplitude band + width-vs-amplitude ---
    bands = [((10, 20), "1 PE (10-20 ADC)"), ((20, 35), "~2 PE (20-35)"),
             ((35, 60), "~3-4 PE (35-60)")]
    spe_avgs = {}
    widths = []
    for band, lbl in bands:
        avg, n, nc = extract_spe(band, seed=seed)
        spe_avgs[lbl] = (avg, n)
        fw, ip = fwhm(avg)
        widths.append((lbl, fw, n))
        print(f"{lbl}: {n} used, FWHM {fw}, peak {avg.max():.1f}@{ip}")
    fw2024, _ = fwhm(hpk2024)
    print(f"2024 HPK template FWHM {fw2024}")

    # --- deconvolution tail over bright HPK snippets ---
    bright = [(c, w) for c, w in all_snippets()
              if (-(w - w[:20].mean())).max() > 200 and c in c2t]
    data_short = mk(spe_avgs["1 PE (10-20 ADC)"][0])
    templates = {
        "2024 HPK avg": mk(hpk2024),
        "v1 per-channel": None,         # per-channel, handled below
        "data 1-PE avg (short)": data_short,
    }
    tails = {k: [] for k in templates}
    for c, w in bright:
        for k, tm in templates.items():
            if k == "v1 per-channel":
                tm = mk(np.array(spev1["templates"][c2t[c]]["values"]))
            d = deconv(w, tm)
            tails[k].append(100 * d[320:960].mean() / max(d.max(), 1e-9))
    print(f"\nbright HPK snippets {len(bright)}; late-tail median % of peak:")
    for k in templates:
        a = np.array(tails[k])
        a = a[np.isfinite(a)]
        print(f"  {k:24s}: {np.median(a):+.2f}%")

    # =================== figure ===================
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

    # panel 1: prompt shapes, unit-peak, peak-aligned
    def unit(t, lo=8, hi=34):
        ip = int(np.argmax(t[:40]))
        return t[ip - lo:ip + hi] / t[ip]
    x = np.arange(-8, 34) * 16
    ax[0].plot(x, unit(hpk2024), "C0-", lw=2, label=f"2024 HPK avg (FWHM {fw2024*16} ns)")
    v1c0 = np.array(spev1["templates"][c2t[0]]["values"])
    ax[0].plot(x, unit(v1c0), "C3-", lw=1.5, label="v1 per-channel (ch0)")
    a1, n1 = spe_avgs["1 PE (10-20 ADC)"]
    fw1, _ = fwhm(a1)
    ax[0].plot(x, unit(a1), "k--", lw=1.8, label=f"data 1-PE avg ({n1} pulses, FWHM {fw1*16} ns)")
    ax[0].axhline(0, color="gray", lw=0.5)
    ax[0].set_xlabel("time from peak [ns]")
    ax[0].set_ylabel("unit-peak amplitude")
    ax[0].set_title("SPE prompt shape\n(data 1-PE avg is narrow — noise/threshold biased)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    # panel 2: width vs amplitude band -> convergence to 2024 at higher SNR
    labels = [w[0].split(" (")[0] for w in widths]
    fws = [w[1] * 16 for w in widths]
    ns = [w[2] for w in widths]
    ax[1].bar(range(len(fws)), fws, color="navy", alpha=0.7)
    ax[1].axhline(fw2024 * 16, color="C0", lw=2, ls="--", label=f"2024 HPK FWHM {fw2024*16} ns")
    for i, (f, nn) in enumerate(zip(fws, ns)):
        ax[1].text(i, f + 4, f"{nn} pul", ha="center", fontsize=8)
    ax[1].set_xticks(range(len(labels)))
    ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylabel("measured SPE FWHM [ns]")
    ax[1].set_title("SPE width vs pulse amplitude\n(higher SNR → converges on the 2024 template)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, axis="y")

    # panel 3: deconvolution late-tail per template
    keys = list(templates.keys())
    data = [np.array(tails[k])[np.isfinite(tails[k])] for k in keys]
    data = [d[(d > -60) & (d < 60)] for d in data]  # clip extreme blowups for display
    bp = ax[2].boxplot(data, vert=True, labels=["2024\nHPK", "v1\nper-ch", "data\n1-PE"],
                       showfliers=False, patch_artist=True)
    for patch, col in zip(bp["boxes"], ["C0", "C3", "gray"]):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_ylabel("decon late-tail [5–15 µs], % of peak")
    ax[2].set_title("Deconvolution quality (bright HPK snippets)\n"
                    "2024 flat ≈ 0; v1 over-subtracts; short data template far worse")
    ax[2].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(f"{HERE}/spe_calibration_study.png", dpi=120)
    plt.close(fig)
    print(f"\nwrote spe_calibration_study.png")


if __name__ == "__main__":
    main()
