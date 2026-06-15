#!/usr/bin/env python3
"""Per-channel PDHD SPE-template / deconvolution-tail study, all 160 OpChannels.

Motivation: on the -x full-stream PDs some deconvolved waveforms (e.g. opch 145)
show a non-zero tail.  This script asks, per channel, whether that is a genuine
SPE-template/data SHAPE mismatch -- and if so tunes the template -- or whether it
is one of the physical effects a template cannot (and should not) remove:

  * real LAr slow scintillation (~1.5 us) -- present on every bright cosmic, NOT
    on scint-free single-PE dark counts.  Flattening it = the v1 over-subtraction.
  * ADC saturation on the brightest pulses (clips both rails) -- information lost,
    not recoverable by any linear template.
  * per-channel data quality (DC-offset / ringing) -- a hardware issue.

Method (faithful NumPy port of flash/src/OpDecon.cxx, validated to ~2e-6 vs the
C++ full-stream decon):
  - probe = small (0.25-1.0 PE/tick), isolated, UNSATURATED pulses -> ~scint-free
    single-PE response.  Average per channel, baseline-subtract the local pre-peak
    level, measure the late tail (1-3 us) as % of peak.
  - reference = the MATCHED-KERNEL tail: deconvolve the template with itself ->
    the intrinsic decon-chain tail (postfilter + Wiener ringing) a PERFECT template
    gives.  Mismatch = data tail - matched-kernel tail (NOT data tail - 0).

Reads the per-event reco produced by run_light_fullstream_evt.sh:
  work/<RUN>_fs<evt>/   light-frames-fullstream-wct.tar.bz2  (opch 120-159, stream)
  work/<RUN>_snip<evt>/ light-frames-wct.tar.bz2             (opch 0-119, snippets)

Outputs (-> pdhd/pics/pd/spe/, git-ignored):
  spe_ch<NNN>.png   one per channel: avg small-pulse decon vs matched kernel
                    (template match) + avg bright-pulse decon (scint tail) + flags
  spe_summary.png   all 160: tail mismatch (data - kernel) and data-quality flags

Usage: spe_template_tune.py <run> [evt evt ...]   (default 27980 8 16 24 104 120 152)
"""
import io, json, os, sys, tarfile
import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PDHD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(PDHD, "work")
PICS = os.path.join(PDHD, "pics", "pd", "spe")
CFG = os.environ.get("PDHD_CFG", "/nfs/data/1/xqian/toolkit-dev/toolkit/cfg/pgrapher/experiment/pdhd")
SPE = os.path.join(CFG, "pdhd-spe-templates.json")

# OpDecon parameters (flash.jsonnet / OpDecon.h, full-stream config).
PRE_TRIGGER, PEDESTAL_BUFFER = 50, 30
LINE_NOISE_RMS, POLARITY = 4.5, -1
POSTFILTER_CUTOFF, FIXED_SNR = 1.5, 0.005
TICK_NS = 16.0

# pulse-averaging windows / selection
PRE, POSTW = 40, 350          # samples before/after peak in the averaging window
SMALL_LO, SMALL_HI = 0.25, 1.0   # small (scint-free) pulse peak band, decon units
BRIGHT_LO, BRIGHT_HI = 5.0, 40.0  # bright cosmic band
ISO = 250                     # nearest other peak (>0.15) must be > this away
RAIL_LO, RAIL_HI = 30, 16353  # ADC rails (+/- margin); pulse must avoid them
MIN_SMALL = 25                # min small pulses to trust a channel average


# --------------------------------------------------------------------------- IO
def load_frames(path, evt):
    out = {}
    with tarfile.open(path) as tf:
        for tag in ("raw", "decon"):
            out[tag] = np.load(io.BytesIO(tf.extractfile(f"frame_{tag}_{evt}.npy").read()))
        out["ch"] = np.load(io.BytesIO(tf.extractfile(f"channels_decon_{evt}.npy").read()))
        out["ti"] = np.load(io.BytesIO(tf.extractfile(f"tickinfo_decon_{evt}.npy").read()))
    return out


def load_templates():
    j = json.load(open(SPE))
    names = [t.get("name", "") for t in j["templates"]]
    tmpl = [np.array(t["values"], float) for t in j["templates"]]
    c2t = {int(c): int(ti) for c, ti in zip(j["channels"], j["template_index"])}
    return names, tmpl, c2t


# ------------------------------------------------------- decon (validated port)
def build_postfilter(n):
    df = 62.5 / n
    cb = POSTFILTER_CUTOFF / df
    sigma = n * np.sqrt(np.log(2.0)) / (2 * np.pi * cb)
    mu = n // 2
    xf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.arange(n) - mu) / sigma) ** 2)
    return np.fft.fft(xf) * np.exp(-1j * 2 * np.pi * np.arange(n) * mu / n)


def matched_kernel(tmpl_vals, n=8192):
    """Decon of a template with itself -> intrinsic tail a PERFECT template gives.
    Length-independent under fixed_snr; n=8192 is ample for a few-us kernel."""
    w = np.zeros(n); v = tmpl_vals[:n]; w[:len(v)] = v
    H = np.fft.fft(w)
    N2 = LINE_NOISE_RMS ** 2 * n; S2 = FIXED_SNR * N2
    K = (np.abs(H) ** 2 * S2) / (np.abs(H) ** 2 * S2 + N2)
    k = np.fft.ifft(K).real
    k = np.fft.ifft(np.fft.fft(k) * build_postfilter(n)).real
    k = np.roll(k, n // 2)
    p = int(np.argmax(k))
    return k[p - PRE:p + POSTW] / k[p]   # peak-normalized, same window as data


def deconvolve(adc, H, post, n):
    """Faithful port of Flash::OpDecon::deconvolve (fixed_snr, flat noise);
    validated to ~2e-6 vs the C++ full-stream decon."""
    nped = PRE_TRIGGER - PEDESTAL_BUFFER
    nz = np.nonzero(adc)[0]
    p0 = nz[0] if len(nz) else 0            # skip the leading-zero DAQ offset
    ped = adc[p0:p0 + nped].mean()
    xv = np.zeros(n); nin = min(n, len(adc))
    xv[:nin] = POLARITY * (adc[:nin] - ped)
    xv[:p0] = 0.0
    N2 = LINE_NOISE_RMS ** 2 * n; S2 = FIXED_SNR * N2
    H2 = np.abs(H) ** 2
    xG = np.conj(H) * S2 / (H2 * S2 + N2)
    xvdec = np.fft.ifft(xG * np.fft.fft(xv)).real
    sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    x = np.fft.ifft(xG * H * sign).real
    im = int(np.argmax(x)); il = im; ir = im
    while il > 0 and x[il] > 0:
        il -= 1
    while ir < n and x[ir] > 0:
        ir += 1
    norm = x[il:ir + 1].sum()
    if norm > 1.0 or norm <= 0.0:
        norm = 1.0
    dped = xvdec[p0:p0 + nped].mean()
    xvdec = np.fft.ifft(np.fft.fft(xvdec) * post).real
    return (xvdec - dped) / norm


def reconstruct(small_shape, tmpl_vals, alpha=0.02, L=PRE + POSTW):
    """Tuned per-channel template = the channel's TRUE SPE response, recovered from
    the measured decon shape: since s=G*p and k=G*T0 (G the fixed filter), DFT(p) =
    DFT(T0)*DFT(s)/DFT(k).  Using p as the template deconvolves p to its own matched
    kernel -> flat tail, no undershoot.  S/K is Tikhonov-regularized (alpha)."""
    ti_k = matched_kernel(tmpl_vals)
    Sf = np.fft.fft(small_shape); Kf = np.fft.fft(ti_k)
    R = Sf * np.conj(Kf) / (np.abs(Kf) ** 2 + alpha * np.mean(np.abs(Kf) ** 2))
    t0 = np.zeros(L); v = tmpl_vals[:L]; t0[:len(v)] = v
    p = np.fft.ifft(np.fft.fft(t0) * R).real
    out = np.zeros(1024); out[:L] = p
    return out


# ---------------------------------------------------------------- measurement
def tail_pct(shape, a_us, b_us, base):
    s0 = PRE + int(a_us * 1000 / TICK_NS); s1 = PRE + int(b_us * 1000 / TICK_NS)
    return 100 * (shape[s0:min(s1, len(shape))].mean() - base)


def undershoot_pct(shape, base):
    """Most-negative excursion 0.25-1.5 us after the peak (the AC-recovery dip a
    template should absorb); a negative value cannot be scintillation."""
    s0 = PRE + int(0.25 * 1000 / TICK_NS); s1 = PRE + int(1.5 * 1000 / TICK_NS)
    return 100 * (shape[s0:s1].min() - base)


def measure(run, events):
    names, tmpl, c2t = load_templates()
    kern = {ti: matched_kernel(tmpl[ti]) for ti in range(len(tmpl))}
    acc_s = {o: np.zeros(PRE + POSTW) for o in range(160)}
    acc_b = {o: np.zeros(PRE + POSTW) for o in range(160)}
    n_s = {o: 0 for o in range(160)}
    n_b = {o: 0 for o in range(160)}
    base_med = {o: [] for o in range(160)}
    base_rms = {o: [] for o in range(160)}
    sat = {o: 0 for o in range(160)}

    def maxrun(mask):
        best = cur = 0
        for v in mask:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best

    for evt in events:
        for dom, tag, pref in ((range(120, 160), "fullstream", "fs"), (range(0, 120), "", "snip")):
            fn = f"{WORK}/{run:06d}_{pref}{evt}/light-frames-{'fullstream-wct' if pref=='fs' else 'wct'}.tar.bz2"
            if not os.path.exists(fn):
                continue
            f = load_frames(fn, evt)
            for o in dom:
                idx = np.where(f["ch"] == o)[0]
                if not len(idx):
                    continue
                w = f["decon"][int(idx[0])].astype(float)
                raw = f["raw"][int(idx[0])].astype(float)
                nzw = w != 0
                if nzw.any():
                    base_med[o].append(float(np.median(w[nzw])))
                    base_rms[o].append(float(np.median(np.abs(w[nzw] - np.median(w[nzw]))) * 1.4826))
                # saturation only meaningful on the continuous full-stream (the
                # self-trigger raw has long zero GAPS between snippets that are not
                # rail clips); snip channels carry sat = -1 (n/a).
                if o >= 120:
                    nz = np.nonzero(raw)[0]
                    if len(nz):
                        r = raw[nz[0]:nz[-1] + 1]
                        if maxrun(r <= RAIL_LO) >= 3 or maxrun(r >= RAIL_HI) >= 3:
                            sat[o] += 1
                else:
                    sat[o] = -1
                pk, prop = find_peaks(w, height=0.15, distance=4)
                h = prop["peak_heights"]
                for i, p in enumerate(pk):
                    if p < PRE or p + POSTW >= len(w):
                        continue
                    lo = np.searchsorted(pk, p - ISO); hi = np.searchsorted(pk, p + ISO)
                    if hi - lo != 1:
                        continue
                    rawmin = raw[p - 15:p + 25].min()
                    if rawmin <= RAIL_LO + 200:        # exclude saturated/near-rail
                        continue
                    prebase = w[p - PRE:p - 12].mean()
                    if abs(prebase) > 0.5:             # exclude baseline drift / pile-up
                        continue
                    if SMALL_LO <= h[i] <= SMALL_HI:
                        acc_s[o] += w[p - PRE:p + POSTW]; n_s[o] += 1
                    elif BRIGHT_LO <= h[i] <= BRIGHT_HI:
                        acc_b[o] += w[p - PRE:p + POSTW]; n_b[o] += 1
    res = {}
    for o in range(160):
        if o not in c2t:
            continue
        ti = c2t[o]
        small = acc_s[o] / n_s[o] if n_s[o] else None
        bright = acc_b[o] / n_b[o] if n_b[o] else None
        mb = float(np.mean(base_med[o])) if base_med[o] else float("nan")
        rr = float(np.mean(base_rms[o])) if base_rms[o] else float("nan")
        cls = "ringing" if (rr >= 0.1 or np.isnan(rr)) else ("offset" if abs(mb) >= 0.05 else "clean")
        smn = small / small[PRE] if small is not None else None
        tail = ktail = exc = under = kunder = uexc = float("nan")
        if smn is not None and n_s[o] >= MIN_SMALL:
            base = smn[:PRE - 12].mean()
            kn = kern[ti]; kbase = kn[:PRE - 12].mean()
            tail = tail_pct(smn, 1.0, 3.0, base)
            ktail = tail_pct(kn, 1.0, 3.0, kbase)
            exc = tail - ktail
            under = undershoot_pct(smn, base)
            kunder = undershoot_pct(kn, kbase)
            uexc = under - kunder   # undershoot beyond the intrinsic kernel (template defect)
        res[o] = dict(ti=ti, ftype="FBK" if ti == 0 else "HPK", small=smn, bright=bright,
                      n_s=n_s[o], n_b=n_b[o], base=mb, rms=rr, cls=cls, sat=sat[o],
                      tail=tail, ktail=ktail, exc=exc, under=under, kunder=kunder, uexc=uexc,
                      dom=("fs" if o >= 120 else "snip"))
    return res, kern, names


def remeasure_tuned(run, events, tuned):
    """Re-deconvolve the tuned full-stream channels with their new templates and
    return the tuned peak-normalized small-pulse average (selection identical to
    measure(): same pulses, found on the DEFAULT decon)."""
    out = {}
    n = 343808
    posts = build_postfilter(n)
    Ht = {o: np.fft.fft(np.pad(tmpl[:n], (0, max(0, n - len(tmpl))))[:n]) for o, tmpl in tuned.items()}
    acc = {o: np.zeros(PRE + POSTW) for o in tuned}
    cnt = {o: 0 for o in tuned}
    for evt in events:
        fn = f"{WORK}/{run:06d}_fs{evt}/light-frames-fullstream-wct.tar.bz2"
        if not os.path.exists(fn):
            continue
        f = load_frames(fn, evt)
        for o in tuned:
            idx = np.where(f["ch"] == o)[0]
            if not len(idx):
                continue
            raw = f["raw"][int(idx[0])].astype(float)
            wd = f["decon"][int(idx[0])].astype(float)
            wt = deconvolve(raw, Ht[o], posts, n)
            pk, prop = find_peaks(wd, height=0.15, distance=4); h = prop["peak_heights"]
            for i, p in enumerate(pk):
                if p < PRE or p + POSTW >= len(wt):
                    continue
                lo = np.searchsorted(pk, p - ISO); hi = np.searchsorted(pk, p + ISO)
                if hi - lo != 1:
                    continue
                if raw[p - 15:p + 25].min() <= RAIL_LO + 200 or abs(wd[p - PRE:p - 12].mean()) > 0.5:
                    continue
                if SMALL_LO <= h[i] <= SMALL_HI:
                    acc[o] += wt[p - PRE:p + POSTW]; cnt[o] += 1
    for o in tuned:
        out[o] = (acc[o] / cnt[o]) / (acc[o][PRE] / cnt[o]) if cnt[o] else None
    return out


def write_type_corrected_json(corrected, names):
    """Write the tuned SPE-templates JSON: same 2-template / 160-channel structure
    as the default, but the named template's values are replaced by the corrected
    (tail-matched) version.  Here only template 0 (FBK) is corrected; template 1
    (HPK, already matched) and the channel map are untouched, so all HPK channels
    stay bit-identical and every FBK channel gets the per-type correction.  No code
    change (OpDecon reads spe_file)."""
    j = json.load(open(SPE))
    for ti, vals in corrected.items():
        j["templates"][ti]["name"] = j["templates"][ti].get("name", f"tmpl{ti}") + "_tuned"
        j["templates"][ti]["values"] = [float(v) for v in vals]
    j["comment"] = ("PDHD SPE templates with the FBK template tail-corrected so its "
                    "deconvolution no longer over-subtracts the post-pulse tail "
                    "(per-type correction; HPK unchanged). See "
                    "pdhd/docs/pdhd-spe-template-tuning.md. Same structure as "
                    "pdhd-spe-templates.json; HPK channels bit-identical.")
    out = os.path.join(CFG, "pdhd-spe-templates-tuned.json")
    json.dump(j, open(out, "w"), indent=1)
    print("wrote", out, "(FBK template corrected; HPK unchanged)")


# ------------------------------------------------------------------- plotting
def per_channel_fig(o, r, kern, tuned_shape=None):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    t = (np.arange(PRE + POSTW) - PRE) * TICK_NS / 1000.0
    kn = kern[r["ti"]]
    # A: small-pulse decon vs matched kernel (template match) + tuned (if any)
    if r["small"] is not None:
        ax[0].plot(t, r["small"], color="C0", lw=1.2,
                   label=f"data, DEFAULT template (N={r['n_s']})")
    ax[0].plot(t, kn, "k--", lw=1.0, label="matched-kernel (perfect template)")
    tuned_txt = ""
    if tuned_shape is not None:
        ax[0].plot(t, tuned_shape, color="C2", lw=1.2, label="data, TUNED template")
        u_t = undershoot_pct(tuned_shape, tuned_shape[:PRE - 12].mean())
        tuned_txt = f"  ->  TUNED {u_t:+.1f}%"
    ax[0].axhline(0, color="0.7", lw=0.5); ax[0].axhspan(-0.08, 0, color="red", alpha=0.04)
    ax[0].set_xlim(-0.3, 5); ax[0].set_ylim(-0.10, 0.35)
    ax[0].set_xlabel("time since peak [us]"); ax[0].set_ylabel("decon, peak-normalized")
    if not np.isnan(r["uexc"]):
        ax[0].set_title(f"A. SPE-template match  (undershoot: data {r['under']:+.1f}% vs "
                        f"kernel {r['kunder']:+.1f}%, defect {r['uexc']:+.1f}%{tuned_txt})", fontsize=9)
    else:
        ax[0].set_title(f"A. SPE-template match  (too few clean small pulses: N={r['n_s']})", fontsize=9)
    ax[0].legend(fontsize=8)
    # B: bright cosmic decon (real scint tail) -- not a template effect
    if r["bright"] is not None:
        bn = r["bright"] / r["bright"][PRE]
        ax[1].plot(t, bn, color="C1", lw=1.2, label=f"avg bright cosmic (N={r['n_b']})")
    if r["small"] is not None:
        ax[1].plot(t, r["small"], color="C0", lw=0.9, alpha=0.7, label="avg small pulse")
    ax[1].plot(t, kern[r["ti"]], "k--", lw=0.8, alpha=0.6, label="matched kernel")
    ax[1].axhline(0, color="0.7", lw=0.5); ax[1].set_xlim(-0.3, 5); ax[1].set_ylim(-0.10, 0.45)
    ax[1].set_xlabel("time since peak [us]"); ax[1].set_ylabel("decon, peak-normalized")
    ax[1].set_title("B. bright cosmic adds real POSITIVE scintillation (not template)", fontsize=9)
    ax[1].legend(fontsize=8)
    sattxt = "n/a" if r["sat"] < 0 else f"{r['sat']}/6"
    flags = (f"class={r['cls']}  baseline={r['base']:+.3f}  rms={r['rms']:.3f}  "
             f"saturating-events={sattxt}")
    fig.suptitle(f"OpChannel {o} ({r['ftype']}, {r['dom']})   |   {flags}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(PICS, exist_ok=True)
    fig.savefig(f"{PICS}/spe_ch{o:03d}.png", dpi=95)
    plt.close(fig)


def summary_fig(res):
    fig, ax = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    chs = sorted(res)
    x = np.array(chs)
    col = {"clean": "C0", "offset": "C1", "ringing": "C3"}
    # top: undershoot defect (data undershoot - matched-kernel undershoot) -- the
    # template-fixable, scint-free mismatch (negative = over-subtraction).
    uexc = np.array([res[o]["uexc"] for o in chs])
    c = [("0.7" if np.isnan(res[o]["uexc"]) else col[res[o]["cls"]]) for o in chs]
    ax[0].bar(x, np.nan_to_num(uexc), color=c)
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].axhline(-2.0, color="green", ls=":", lw=1, label="tuning candidate threshold -2%")
    ax[0].axvline(119.5, color="0.5", ls="--", lw=0.8)
    ax[0].text(60, ax[0].get_ylim()[1] * 0.8, "0-119 self-trigger", ha="center", fontsize=8)
    ax[0].text(140, ax[0].get_ylim()[1] * 0.8, "120-159 full-stream", ha="center", fontsize=8)
    ax[0].set_ylabel("post-pulse UNDERSHOOT excess\nover matched kernel  [% of peak]")
    ax[0].set_title("Per-channel SPE-template defect = scint-free undershoot minus the intrinsic "
                    "(perfect-template) kernel undershoot.  Grey = too few clean pulses; "
                    "blue/orange/red = clean/offset/ringing.")
    ax[0].legend(fontsize=8, ncol=2)
    # bottom: data-quality flags
    base = np.array([res[o]["base"] for o in chs])
    sat = np.array([res[o]["sat"] for o in chs])
    ax[1].bar(x, base, color=c, label="decon baseline (DC offset)")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].axvline(119.5, color="0.5", ls="--", lw=0.8)
    for o in chs:
        if res[o]["sat"] > 0:
            ax[1].plot(o, 0, "v", color="purple", ms=4)
    ax[1].plot([], [], "v", color="purple", ms=5, label="saturates on >=1 event")
    ax[1].set_ylabel("decon baseline median")
    ax[1].set_xlabel("OpChannel")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(PICS, exist_ok=True)
    fig.savefig(f"{PICS}/spe_summary.png", dpi=100)
    plt.close(fig)
    print("wrote", f"{PICS}/spe_summary.png")


def main():
    run = int(sys.argv[1]) if len(sys.argv) > 1 else 27980
    events = [int(a) for a in sys.argv[2:]] or [8, 16, 24, 104, 120, 152]
    res, kern, names = measure(run, events)
    print("templates:", names)
    for ti, nm in ((0, "FBK"), (1, "HPK")):
        kn = kern[ti]
        ue = [res[o]["uexc"] for o in res if res[o]["ti"] == ti and not np.isnan(res[o]["uexc"])]
        print(f"{nm}: matched-kernel undershoot={undershoot_pct(kn, kn[:PRE-12].mean()):+.2f}%   "
              f"per-channel undershoot DEFECT: median {np.median(ue):+.2f}%  std {np.std(ue):.2f}  N={len(ue)}")
    # tuning candidates: full-stream channels (good dark-count stats) with a
    # genuine over-subtraction undershoot, clean class (exclude offset/ringing
    # data-quality channels, which a template cannot fix).
    names_t, tmpl, c2t = load_templates()
    # The FBK 2024-average template systematically over-subtracts the post-pulse
    # tail (median undershoot defect ~-3%); HPK is matched.  Build ONE corrected
    # FBK template (per-TYPE) from the high-statistics average of the FBK
    # full-stream channels with a clear defect.  A single per-type correction
    # GENERALIZES BETTER out-of-sample than 13 per-channel templates and does not
    # shift the absolute PE scale (validated in pd_plot/oos_validate.py; see doc).
    cand = sorted([o for o in res if o >= 120 and res[o]["ti"] == 0 and not np.isnan(res[o]["uexc"])
                   and res[o]["uexc"] <= -2.0 and res[o]["cls"] == "clean" and res[o]["n_s"] >= 50],
                  key=lambda o: res[o]["uexc"])
    print("FBK build channels (uexc<=-2%, clean, N>=50):",
          ", ".join(f"ch{o}({res[o]['uexc']:+.1f}%)" for o in cand))
    fbk_corrected = reconstruct(np.mean([res[o]["small"] for o in cand], axis=0), tmpl[0])
    write_type_corrected_json({0: fbk_corrected}, names_t)
    print("re-deconvolving full-stream FBK channels with the corrected template...")
    fbk_fs = [o for o in res if o >= 120 and res[o]["ti"] == 0]
    tuned_shapes = remeasure_tuned(run, events, {o: fbk_corrected for o in fbk_fs})
    defu, tunu = [], []
    for o in fbk_fs:
        ts = tuned_shapes.get(o)
        if ts is not None and not np.isnan(res[o]["under"]):
            ut = undershoot_pct(ts, ts[:PRE - 12].mean())
            defu.append(abs(res[o]["under"])); tunu.append(abs(ut))
            print(f"  ch{o}: undershoot {res[o]['under']:+.1f}% -> {ut:+.1f}%")
    print(f"mean |undershoot| FBK full-stream: {np.mean(defu):.2f}% -> {np.mean(tunu):.2f}%")
    for o in sorted(res):
        per_channel_fig(o, res[o], kern, tuned_shapes.get(o))
    summary_fig(res)
    print(f"wrote {len(res)} per-channel figures to {PICS}/")
    np.save("/home/xqian/tmp/spe_res.npy", res, allow_pickle=True)


if __name__ == "__main__":
    main()
