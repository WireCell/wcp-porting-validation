#!/usr/bin/env python3
"""PDVD cathode saturation study: what is the best PE estimator for railed pulses?

Repro (order matters -- 1. makes the validation input for 2.):
  1. cd pdvd && export WIRECELL_PATH=$WCT/toolkit/cfg:$WCT/wire-cell-data
     wcsonnet -A input_file=$PWD/input_data_light_rawwf/np02vd_raw_run039252_1176_*_rawwf.root \
              -A output_file=/home/xqian/tmp/satstudy/decon-dump-298567.tar.gz \
              -S run=39252 -S event=298567 \
              -o /home/xqian/tmp/satstudy/decon-dump.json docs/qlmatch/scripts/wct-decon-dump.jsonnet
     wire-cell -l stderr -c /home/xqian/tmp/satstudy/decon-dump.json
  2. cd pdvd/docs/qlmatch && python3 scripts/saturation_recovery_study.py --all
     (or --validate / --study / --realrail / --prevalence)

Methods compared on clipped pulses (see 11_pdvd-saturation-recovery.md):
  raw      -- LArSoft-keepup-style: baseline-subtracted RAW area / template area
              (no deconvolution; answers "OpHit without decon?")
  clip     -- deconvolve straight through the rail (current veto-off behavior)
  tail     -- repair: fill the rail run by back-extrapolating the falling-edge
              exponential (per-channel tau from the SPE template tail), then decon
  twoside  -- repair: fill with min(rising-edge extrapolation, falling-edge
              back-extrapolation), clamped >= the rail, then decon

Ground truth: real bright UNRAILED cathode pulses, clipped at synthetic rails
rail_f = base + f*(peak-base), f in FRACS (clip depth d = 1/f up to 4).
PE_true = wiener-inspired decon area of the UNCLIPPED pulse over the same
window -- i.e. exactly what the production estimator would have reported had
the pulse not railed.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pd_mapping_audit import (RAWWF, PDVD, RAIL, load_cathode_streams,
                              rail_intervals, dump_paths)  # noqa: E402

SATSTUDY = "/home/xqian/tmp/satstudy"
FS = 62.5           # MHz
WI_SIGMA = 1.25     # MHz, cathode branch
WI_POWER = 2.0
WI_EPS_REL = 1e-3
NPED = 50 - 30      # OpDecon pre_trigger - pedestal_buffer
SEG = 16384         # decon segment length for the study
WIN_PRE, WIN_POST = 50, 300   # PE integration window around the peak (OpRoi pads)
FRACS = (0.9, 0.7, 0.5, 0.35, 0.25)
HAND_FLASHES = (37, 41, 57)   # hand-confirmed saturated crossers, evt298567
METHODS = ("raw", "clip", "tail", "twoside", "tmpl")

CATH_ODS = list(range(4, 12))  # cathode X-Arapuca OpDet columns
# offline channel decade x (101x -> x=1) -> OpDet index (pd_mapping_audit OFF2OD)
OFF2OD = {1: 6, 2: 4, 3: 8, 4: 10, 5: 7, 6: 5, 7: 9, 8: 11}


# ------------------------------------------------------------------ templates
def _find_templates():
    rel = "pgrapher/experiment/protodunevd/pdvd-spe-templates.json"
    cands = [os.path.join(p, rel)
             for p in os.environ.get("WIRECELL_PATH", "").split(":") if p]
    cands.append(os.path.join(HERE, "../../../../../toolkit/cfg", rel))  # scripts/ adds one level
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(rel)


class Channel:
    """Per-channel SPE template + derived decon filter pieces + tau fits."""

    def __init__(self, wave):
        self.wave = np.asarray(wave, np.float64)
        self.area = self.wave.sum()
        self.amp = self.wave.max()
        ipk = int(np.argmax(self.wave))
        # falling-edge tau: log-linear fit on template tail
        t = np.arange(ipk + 25, min(ipk + 200, len(self.wave)))
        y = self.wave[t]
        m = y > 0.02 * self.amp
        p = np.polyfit(t[m], np.log(y[m]), 1)
        self.tau_fall = -1.0 / p[0]           # ticks
        # rising-edge tau from the last few pre-peak samples
        tr = np.arange(max(0, ipk - 4), ipk)
        yr = self.wave[tr]
        mr = yr > 0.02 * self.amp
        if mr.sum() >= 2:
            pr = np.polyfit(tr[mr], np.log(yr[mr]), 1)
            self.tau_rise = 1.0 / pr[0]
        else:
            self.tau_rise = 1.0
        self.ipk = ipk
        # template values beyond the stored wave: exp extension with tau_fall
        self._filters = {}

    def tshape(self, k):
        """Template value at offset k from t_start (array of ints >= 0)."""
        k = np.asarray(k)
        out = np.empty(len(k), np.float64)
        inside = k < len(self.wave)
        out[inside] = self.wave[np.clip(k[inside], 0, len(self.wave) - 1)]
        last = self.wave[-1] if self.wave[-1] > 0 else \
            self.amp * np.exp(-(len(self.wave) - 1 - self.ipk) / self.tau_fall)
        out[~inside] = last * np.exp(-(k[~inside] - (len(self.wave) - 1))
                                     / self.tau_fall)
        return out

    def filter(self, n):
        """Decon transfer function G[k] for an n-sample segment (cached)."""
        if n not in self._filters:
            padded = np.zeros(n)
            padded[:min(len(self.wave), n)] = self.wave[:n]
            H = np.fft.fft(padded)
            eps = (WI_EPS_REL * np.abs(H).max()) ** 2
            f = (FS / n) * np.minimum(np.arange(n), n - np.arange(n))
            F = np.exp(-0.5 * (f / WI_SIGMA) ** WI_POWER)
            self._filters[n] = np.conj(H) * F / (np.abs(H) ** 2 + eps)
        return self._filters[n]


def load_channels():
    d = json.load(open(_find_templates()))
    out = {}
    for ch, idx in zip(d["channels"], d["template_index"]):
        if 1000 <= ch < 2000:
            out[ch] = Channel(d["templates"][idx]["values"])
    return out


# ------------------------------------------------------------------ replica
def decon_replica(adc, chan, nfft=None, pedestal=None):
    """NumPy replica of Flash::OpDecon::deconvolve (wiener-inspired mode).

    adc: raw record samples.  The baseline-SUBTRACTED waveform is zero-padded
    to nfft (as in OpDecon.cxx:277-284 -- the pad is 0 in subtracted units,
    NOT raw 0).  pedestal None -> pipeline convention (mean of first NPED).
    """
    n = nfft or len(adc)
    if pedestal is None:
        pedestal = float(np.mean(adc[:NPED]))
    xv = np.zeros(n)
    m = min(n, len(adc))
    xv[:m] = np.asarray(adc[:m], np.float64) - pedestal
    Y = chan.filter(n) * np.fft.fft(xv)
    dec = np.fft.ifft(Y).real
    dec -= dec[:NPED].mean()   # apply_post_blcorr head-pedestal
    return dec


def window_pe(dec, pk, local_baseline=True):
    """Decon integral over [pk-WIN_PRE, pk+WIN_POST) with an optional LOCAL
    decon-baseline correction (median decon level in a quiet band before the
    window x window length) -- the full-stream decon carries low-frequency
    wander that OpRoi's HPF normally removes."""
    lo, hi = max(0, pk - WIN_PRE), min(len(dec), pk + WIN_POST)
    val = float(dec[lo:hi].sum())
    if local_baseline:
        b0, b1 = max(0, pk - 2000), max(0, pk - 500)
        if b1 > b0:
            val -= float(np.median(dec[b0:b1])) * (hi - lo)
    return val


def validate(args):
    """Replica vs the pipeline decon dump (must pass before anything else)."""
    import tarfile
    print("=" * 90)
    print("VALIDATE - NumPy decon replica vs pipeline dump (evt298567 cathode)")
    print("=" * 90)
    tf = tarfile.open(os.path.join(SATSTUDY, "decon-dump-298567.tar.gz"))

    def npy(name):
        import io
        return np.load(io.BytesIO(tf.extractfile(name).read()))

    raw = npy("frame_raw_298567.npy")
    dec = npy("frame_decon_298567.npy")
    chans = npy("channels_raw_298567.npy")
    chd = load_channels()

    def record(i):
        """Extract channel i's record + its offset in the dense frame grid
        (traces sit at their own tbin; the dense array is zero-filled
        elsewhere and real ADC is never 0)."""
        nz = np.flatnonzero(raw[i])
        s = int(nz[0])
        return raw[i][s:int(nz[-1]) + 1], s

    print(f"{'chan':>6} {'max|d|':>10} {'rms(d)':>10} {'rel rms':>9} "
          f"{'pulse-integral rel diff':>24}")
    worst_wf = worst_pe = 0.0
    for i, ch in enumerate(chans):
        ch = int(ch)
        if ch not in chd:
            continue
        r, s = record(i)
        full = decon_replica(r, chd[ch], nfft=468864)
        d = dec[i][s:s + 468864]
        resid = full - d
        rel_rms = resid.std() / max(1e-12, d.std())
        worst_wf = max(worst_wf, rel_rms)
        # pulse-integral check on the brightest unrailed pulse (local decon
        # baseline on BOTH -- the raw pedestal is 20 samples of a 7.5 ms
        # stream, so the decon carries identical slow wander in both)
        pk = int(np.argmax(np.where(r < RAIL, r, -np.inf)))
        pe_pipe = window_pe(d, pk)
        pe_repl = window_pe(full, pk)
        rel = abs(pe_repl - pe_pipe) / max(1e-9, abs(pe_pipe))
        worst_pe = max(worst_pe, rel)
        print(f"{ch:>6} {np.abs(resid).max():>10.2e} {resid.std():>10.2e} "
              f"{rel_rms:>9.2e} {rel:>18.2e} (PE {pe_pipe:9.2f})")
    print(f"worst waveform relative rms: {worst_wf:.2e}; worst pulse-integral "
          f"relative difference: {worst_pe:.2e} "
          f"({'PASS' if worst_pe < 0.01 and worst_wf < 0.01 else 'FAIL'} at <1%)")

    # segment-decon (local baseline, the study's estimator) vs full-stream
    # pipeline decon (local decon-baseline corrected)
    print("\nsegment (SEG=%d, local baseline) vs pipeline full-stream decon:" % SEG)
    worst = 0.0
    for i, ch in enumerate(chans):
        ch = int(ch)
        if ch not in chd:
            continue
        r, s = record(i)
        d = dec[i][s:s + 468864]
        pk = int(np.argmax(np.where(r < RAIL, r, -np.inf)))
        s0 = int(max(0, min(pk - SEG // 2, len(r) - SEG)))
        seg = r[s0:s0 + SEG].astype(np.float64)
        base = seg_baseline(seg)
        dseg = decon_replica(seg, chd[ch], pedestal=base)
        pe_f = window_pe(d, pk)
        pe_s = window_pe(dseg, pk - s0)
        rel = abs(pe_s - pe_f) / max(1e-9, abs(pe_f))
        worst = max(worst, rel)
        print(f"{ch:>6}  pipeline {pe_f:9.2f}  seg {pe_s:9.2f}  rel {rel:.2e}")
    print(f"worst segment-vs-pipeline pulse-integral difference: {worst:.2e} "
          f"({'PASS' if worst < 0.03 else 'FAIL'} at <3%)")


# ------------------------------------------------------------------ study
def seg_baseline(seg):
    """Robust local baseline: median of the two segment ends."""
    return float(np.median(np.concatenate([seg[:1000], seg[-1000:]])))


def repair_runs(w, rail_level, chan, base, mode, fit_m=8):
    """Fill rail runs (samples >= rail_level) on a copy; return the copy.

    mode:
      'tail'    -- A_exit * exp(-(t-exit)/tau_fall) anchored on the first
                   fit_m samples after the run (single slow tau).
      'twoside' -- min(rise-side B*exp((t-entry)/tau_rise), tail) --
                   the exp intersection approximates the peak.
      'tmpl'    -- the channel's own SPE template shape (fast+slow decay),
                   peak assumed at the run start (X-Arapuca rise is ~2-3
                   ticks), amplitude anchored on the fit_m exit samples.
    All fills are clamped >= the measured (railed) samples.
    """
    out = np.array(w, np.float64)
    hit = out >= rail_level
    if not hit.any():
        return out
    d = np.diff(hit.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if hit[0]:
        starts = [0] + starts
    if hit[-1]:
        ends = ends + [len(hit)]
    for i, j in zip(starts, ends):
        y = out[j:j + fit_m] - base       # falling-edge anchor samples
        t = np.arange(len(y))
        m = y > 0
        if m.sum() < 2:
            continue                      # cannot anchor: leave clipped
        tt = np.arange(i, j)
        if mode == "tmpl":
            # template offsets: peak at run start i
            ref = chan.tshape(chan.ipk + (j + t[m] - i))
            A = np.median(y[m] / np.maximum(ref, 1e-9))
            fill = A * chan.tshape(chan.ipk + (tt - i))
        else:
            A = np.median(y[m] * np.exp(t[m] / chan.tau_fall))
            fill = A * np.exp(-(tt - j) / chan.tau_fall)
            if mode == "twoside":
                y2 = out[max(0, i - 4):i] - base
                t2 = np.arange(max(0, i - 4), i)
                m2 = y2 > 0
                if m2.sum() >= 1:
                    B = np.median(y2[m2] * np.exp((i - t2[m2]) / chan.tau_rise))
                    fill = np.minimum(B * np.exp((tt - i) / chan.tau_rise), fill)
        out[i:j] = np.maximum(out[i:j], base + fill)
    return out


def methods_pe(seg, chan, base, pk, rail_level=None):
    """PE by each estimator over [pk-WIN_PRE, pk+WIN_POST) of the segment."""
    lo, hi = max(0, pk - WIN_PRE), min(len(seg), pk + WIN_POST)
    out = {}
    out["raw"] = float(np.clip(seg[lo:hi] - base, 0, None).sum() / chan.area)
    dec = decon_replica(seg, chan, pedestal=base)
    out["clip"] = window_pe(dec, pk)
    if rail_level is not None and (seg >= rail_level).any():
        for name in ("tail", "twoside", "tmpl"):
            rep = repair_runs(seg, rail_level, chan, base, mode=name)
            drep = decon_replica(rep, chan, pedestal=base)
            out[name] = window_pe(drep, pk)
    else:
        out["tail"] = out["twoside"] = out["tmpl"] = out["clip"]
    return out


def harvest_pulses(max_per_chan=3, hmin=3000, hmax=12000, files=None):
    """Bright UNRAILED cathode pulses -> (chan, segment, peak-in-seg, base)."""
    pulses = []
    for rfile in files or sorted(glob.glob(os.path.join(RAWWF, "*_rawwf.root"))):
        import uproot
        t = uproot.open(rfile)["rawdump/raw_waveform"]
        events = sorted(set(t.arrays(["event"], library="np")["event"]))
        for ev in events:
            (waves, _), _ = load_cathode_streams(rfile, ev)
            for c, w in sorted(waves.items()):
                base = float(np.median(w))
                h = w - base
                cand = np.where((h > hmin) & (w < RAIL))[0]
                taken = []
                # greedy isolation: strongest first, >= SEG/2 apart, no rail
                for idx in cand[np.argsort(h[cand])[::-1]]:
                    if len(taken) >= max_per_chan:
                        break
                    if any(abs(idx - tj) < SEG // 2 for tj in taken):
                        continue
                    s0 = int(max(0, min(idx - SEG // 2, len(w) - SEG)))
                    seg = w[s0:s0 + SEG]
                    if seg.max() >= RAIL:      # no real rail anywhere near
                        continue
                    pk = s0 + int(np.argmax(seg))
                    if abs(pk - idx) > WIN_PRE:  # not the local max -> skip
                        continue
                    if h[pk] > hmax:
                        continue
                    taken.append(idx)
                    pulses.append((c, np.array(seg, np.float64), pk - s0,
                                   seg_baseline(seg)))
    return pulses


def study(args):
    print("=" * 90)
    print("STUDY - synthetic clipping of real unrailed pulses (ground truth)")
    print("=" * 90)
    files = sorted(glob.glob(os.path.join(RAWWF, "*_rawwf.root")))[:2]  # 39252+39253
    pulses = harvest_pulses(files=files)
    print(f"harvested {len(pulses)} bright unrailed cathode pulses "
          f"(peak {3000}-{12000} ADC over base) from {len(files)} files")

    chd = load_channels()
    rows = []   # (frac, method, ratio)
    for c, seg, pk, base in pulses:
        chan = chd[c]
        true = methods_pe(seg, chan, base, pk)
        pe_true = true["clip"]           # decon of the UNCLIPPED pulse
        if pe_true <= 0:
            continue
        raw_true = true["raw"]
        rows.append((1.0, "raw", raw_true / pe_true))
        for f in FRACS:
            rail_level = base + f * (seg[pk] - base)
            clipped = np.minimum(seg, rail_level)
            pe = methods_pe(clipped, chan, base, pk, rail_level=rail_level - 0.5)
            for m in METHODS:
                rows.append((f, m, pe[m] / pe_true))

    rows = np.array(rows, dtype=[("f", "f8"), ("m", "U8"), ("r", "f8")])
    print("\nPE_rec / PE_true: median [16,84] per method vs clip depth d = 1/f")
    print(f"{'d':>5} {'frac':>5} |" + "".join(f" {m:>24}" for m in METHODS))
    sel0 = rows[(rows["f"] == 1.0) & (rows["m"] == "raw")]["r"]
    print(f"{'-':>5} {'1.00':>5} | raw-area on UNclipped: median "
          f"{np.median(sel0):.3f} [{np.percentile(sel0,16):.3f},"
          f" {np.percentile(sel0,84):.3f}]  (raw-vs-decon scale check)")
    summary = {}
    for f in FRACS:
        line = f"{1/f:>5.2f} {f:>5.2f} |"
        for m in METHODS:
            r = rows[(rows["f"] == f) & (rows["m"] == m)]["r"]
            med, lo, hi = (np.median(r), np.percentile(r, 16), np.percentile(r, 84))
            summary[(f, m)] = (med, lo, hi)
            line += f"  {med:6.3f} [{lo:5.3f},{hi:5.3f}]"
        print(line)

    # pass/fail at d<=2 (f>=0.5)
    print("\npass bar: |median bias| < 20% and half-band < 30% for d <= 2:")
    for m in METHODS:
        ok = True
        for f in [f for f in FRACS if f >= 0.5]:
            med, lo, hi = summary[(f, m)]
            if abs(med - 1) > 0.20 or (hi - lo) / 2 > 0.30:
                ok = False
        print(f"  {m:8s}: {'PASS' if ok else 'FAIL'}")

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"raw": "tab:gray", "clip": "tab:red", "tail": "tab:blue",
              "twoside": "tab:green", "tmpl": "tab:purple"}
    for m in METHODS:
        d = [1 / f for f in FRACS]
        med = [summary[(f, m)][0] for f in FRACS]
        lo = [summary[(f, m)][1] for f in FRACS]
        hi = [summary[(f, m)][2] for f in FRACS]
        ax.plot(d, med, "o-", color=colors[m], label=m)
        ax.fill_between(d, lo, hi, color=colors[m], alpha=0.15)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.axhspan(0.8, 1.2, color="k", alpha=0.05)
    ax.set_xlabel("clip depth d = true peak / rail")
    ax.set_ylabel("PE_rec / PE_true")
    ax.set_ylim(0, 1.6)
    ax.set_title(f"cathode PE estimators on synthetically clipped real pulses "
                 f"(n={len(pulses)})")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(os.path.dirname(HERE), "pics", "saturation_recovery_bias.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


# ------------------------------------------------------------------ real rails
def realrail(args):
    """Prediction-shape consistency on the hand-confirmed evt298567 crossers."""
    print("=" * 90)
    print("REALRAIL - evt298567 flashes 37/41/57: railed-channel PE per method vs")
    print("           prediction scaled on the UNrailed cathode OpDets (satoff dump)")
    print("=" * 90)
    dump = os.path.join(PDVD, "work/039252_0_satoff/calib-evt298567.json")
    d = json.load(open(dump))
    toff = d["trigger_offsets_us"][0]
    fl = {f["gid"]: f for f in d["flashes"]}
    bundles = {}
    for b in d["bundles"]:
        if b.get("auto_selected"):
            bundles.setdefault(b["flash_gid"], []).append(b)

    rfile = sorted(glob.glob(os.path.join(RAWWF, "np02vd_raw_run039252_*_rawwf.root")))[0]
    (waves, ts), t0 = load_cathode_streams(rfile, 298567)
    chd = load_channels()

    for gid in HAND_FLASHES:
        f = fl.get(gid)
        if f is None or gid not in bundles:
            print(f"flash gid{gid}: not in satoff dump / no auto bundle -- skipped")
            continue
        pred = np.zeros(40)
        for b in bundles[gid]:
            pred += np.asarray(b["pred_pe"], float)
        t_light = f["time"] - toff

        # per-OpDet method PE from raw (sum of the two ganged sub-channels)
        per_od = {}
        railed = {}
        for x in range(1, 9):
            od = OFF2OD[x]
            tot = {m: 0.0 for m in METHODS}
            nrail = 0
            for c in (1000 + 10 * x, 1001 + 10 * x):
                if c not in waves:
                    continue
                w = waves[c]
                s = int((t_light - (ts[c] - t0)) * FS)
                s0 = int(max(0, min(s - SEG // 2, len(w) - SEG)))
                seg = np.array(w[s0:s0 + SEG], np.float64)
                base = seg_baseline(seg)
                lo = s - s0
                pkw = seg[max(0, lo - WIN_PRE):lo + WIN_POST]
                pk = max(0, lo - WIN_PRE) + int(np.argmax(pkw))
                nrail += int((pkw >= RAIL).sum())
                pe = methods_pe(seg, chd[c], base, pk, rail_level=RAIL - 0.5)
                for m in tot:
                    tot[m] += pe[m]
            per_od[od] = tot
            railed[od] = nrail > 0

        un = [od for od in CATH_ODS if not railed[od] and pred[od] > 0]
        ra = [od for od in CATH_ODS if railed[od]]
        meas = np.asarray(f["pe"], float)
        if not un or not ra:
            print(f"flash gid{gid}: unrailed={un} railed={ra} -- nothing to test")
            continue
        s_scale = sum(meas[od] for od in un) / sum(pred[od] for od in un)
        print(f"\nflash gid{gid} (t={f['time']:.1f} us, total {f['total_PE']:.0f} PE, "
              f"scale from {len(un)} unrailed ods = {s_scale:.3f}):")
        print(f"{'od':>4} {'pred*s':>9} {'dump(veto-off)':>14} "
              + "".join(f" {m:>9}" for m in METHODS))
        for od in ra:
            exp = s_scale * pred[od]
            t_ = per_od[od]
            print(f"{od:>4} {exp:>9.0f} {meas[od]:>14.0f}"
                  + "".join(f" {t_[m]:>9.0f}" for m in METHODS))
        for od in un:
            t_ = per_od[od]
            print(f"{od:>4} {s_scale*pred[od]:>9.0f} {meas[od]:>14.0f}"
                  + "".join(f" {t_[m]:>9.0f}" for m in METHODS)
                  + "   (unrailed control)")


# ------------------------------------------------------------------ prevalence
def prevalence(args):
    """Clip-depth d distribution of REAL rails near bright flashes."""
    print("=" * 90)
    print("PREVALENCE - clip depth d = extrapolated-peak/rail for real rails")
    print("             (rails within +-250 samples of a >=1000 PE flash)")
    print("=" * 90)
    import uproot
    chd = load_channels()
    depths = []
    flash_depths = []   # per bright flash: max d over its railed cathode runs
    for rfile in sorted(glob.glob(os.path.join(RAWWF, "*_rawwf.root"))):
        run = os.path.basename(rfile).split("_")[2].replace("run", "").zfill(6)
        t = uproot.open(rfile)["rawdump/raw_waveform"]
        events = sorted(set(t.arrays(["event"], library="np")["event"]))
        for ev in events:
            hits = [p for p in dump_paths()
                    if p.endswith("calib-evt%d.json" % ev) and ("/%s_" % run) in p]
            if not hits:
                continue
            d = json.load(open(hits[0]))
            toff = d["trigger_offsets_us"][0]
            tf = [f["time"] - toff for f in d["flashes"] if f["total_PE"] >= 1000]
            (waves, ts), t0 = load_cathode_streams(rfile, ev)
            per_flash = [[] for _ in tf]
            for c, w in waves.items():
                base = float(np.median(w))
                for a, b in rail_intervals(w):
                    tc = (ts[c] - t0) + 0.5 * (a + b) / FS
                    near = [k for k, x in enumerate(tf) if abs(tc - x) < 250 / FS]
                    if not near:
                        continue
                    y = w[b:b + 8] - base
                    tt = np.arange(len(y))
                    m = y > 0
                    if m.sum() < 2:
                        continue
                    A = np.median(y[m] * np.exp(tt[m] / chd[c].tau_fall))
                    peak_est = A * np.exp((b - a) / chd[c].tau_fall)
                    dep = peak_est / (RAIL - base)
                    depths.append(dep)
                    for k in near:
                        per_flash[k].append(dep)
            flash_depths.extend(max(v) for v in per_flash if v)
    depths = np.array(depths)
    flash_depths = np.array(flash_depths)
    print(f"n rail runs near bright flashes: {len(depths)}")
    for dmax in (1.5, 2, 3, 5):
        print(f"  d <= {dmax:>3}: {100.0*np.mean(depths <= dmax):5.1f}%")
    print(f"  median d = {np.median(depths):.2f}, "
          f"[16,84] = [{np.percentile(depths,16):.2f}, {np.percentile(depths,84):.2f}], "
          f"max = {depths.max():.1f}")
    print("  (the d estimator is itself a tail back-extrapolation: the study "
          "shows it OVERSHOOTS for long runs, so the deep tail is an upper bound)")
    print(f"\nper bright RAILED flash, deepest cathode rail (n={len(flash_depths)}):")
    for dmax in (1.5, 2, 3, 5):
        print(f"  max-d <= {dmax:>3}: {100.0*np.mean(flash_depths <= dmax):5.1f}%"
              f"   <- 'repairable' fraction if the pass bar is d <= {dmax}")
    print(f"  median max-d = {np.median(flash_depths):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--study", action="store_true")
    ap.add_argument("--realrail", action="store_true")
    ap.add_argument("--prevalence", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all or args.validate:
        validate(args)
    if args.all or args.study:
        study(args)
    if args.all or args.realrail:
        realrail(args)
    if args.all or args.prevalence:
        prevalence(args)


if __name__ == "__main__":
    main()
