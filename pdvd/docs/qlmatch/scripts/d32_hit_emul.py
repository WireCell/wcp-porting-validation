#!/usr/bin/env python3
"""doc qlmatch/32 -- Python replica of OpHitFinder (sliding_window + split_pulse + hit_threshold) on the PDVD cathode
decon_roi stream, used to (1) prove closure against the C++ ophits of the frame-dumped arms and (2) prototype the
sat_hit_merge rule before it is written in C++.

Port of toolkit flash/src/OpHitFinder.cxx (sliding_window 104-193, split_pulse 195-298, operator() 329-497) with the
PDVD cathode settings (wct-light-reco.jsonnet cath_hit: fixed_ped_sigma 0.75 -> ped 0, nsigma 3; hit_threshold 3.7;
flash.jsonnet algo split_enable, prominence 0.4 / abs 100, min_peak 3, min_sep 2).

The stream is zero outside the OpRoi ROIs, so pulse finding runs only on the nonzero segments (padded), which is
exact: a pulse cannot start on a zero sample (start threshold 3) and a pulse in progress ends within
num_postsample of the first sub-threshold sample.
"""
import numpy as np

ALGO = dict(adc=3.0, nsig=3.0, tail_adc=1.0, tail_nsig=1.0, end_adc=1.0, end_nsig=1.0, min_width=1, npre=2, npost=2,
            split=True, frac=0.4, absdepth=100.0, minpk=3.0, minsep=2)
PED_SIGMA = 0.75
HIT_THRESHOLD = 3.7


def sliding_window(wf, a=ALGO, ped_sigma=PED_SIGMA):
    st = max(a["adc"], a["nsig"] * ped_sigma)
    tt = max(a["tail_adc"], a["tail_nsig"] * ped_sigma)
    et = max(a["end_adc"], a["end_nsig"] * ped_sigma)
    npre, npost = a["npre"], a["npost"]
    pulses = []
    P = None
    fire = in_tail = in_post = False
    post = 0

    def reg(t_end):
        P["t_end"] = t_end
        if P["t_end"] - P["t_start"] >= a["min_width"]:
            pulses.append(dict(P))

    for i in range(len(wf)):
        v = float(wf[i])
        if (not fire or in_tail or in_post) and v > st:
            if in_tail:
                reg(i - 1)
            if pulses:
                buf = i - pulses[-1]["t_end"] - 1
            else:
                buf = min(npre, i)
            if buf > npre:
                buf = npre
            if in_post:
                te = i - buf
                if te > 0:
                    te -= 1
                reg(te)
            P = dict(t_start=i - buf, t_end=0, t_max=0, peak=0.0, area=0.0)
            for pre in range(P["t_start"], i):
                if wf[pre] > 0:
                    P["area"] += float(wf[pre])
            fire, in_tail, in_post = True, False, False
        if fire and v < tt:
            fire, in_tail, in_post = False, True, False
        if (fire or in_tail) and v < et:
            in_post, fire, in_tail = True, False, False
            post = npost
        if in_post and post < 1:
            reg(i - 1)
            fire = in_tail = in_post = False
        if fire or in_tail or in_post:
            P["area"] += v
            if P["peak"] < v:
                P["peak"], P["t_max"] = v, i
            if in_post:
                post -= 1
    if fire or in_tail or in_post:
        reg(len(wf) - 1)
    return pulses


def split_pulse(wf, p, a=ALGO):
    if not a["split"] or p["t_end"] <= p["t_start"]:
        return [p]
    A, B = p["t_start"], p["t_end"]
    v = lambda i: float(wf[i])
    peaks = []
    i = A
    while i <= B:
        left = v(i - 1) if i > A else -1e300
        if v(i) < left:
            i += 1
            continue
        j = i
        while j < B and v(j + 1) == v(i):
            j += 1
        right = v(j + 1) if j < B else -1e300
        if v(i) >= left and v(i) >= right:
            c = (i + j) // 2
            if v(c) > a["minpk"]:
                peaks.append(c)
        i = j + 1
    if len(peaks) <= 1:
        return [p]

    def valley(pp, qq):
        seg = wf[pp + 1:qq]
        return pp + 1 + int(np.argmin(seg)) if len(seg) else pp + 1

    changed = True
    while changed and len(peaks) > 1:
        changed = False
        for k in range(len(peaks) - 1):
            pp, qq = peaks[k], peaks[k + 1]
            val = valley(pp, qq)
            sm = min(v(pp), v(qq))
            depth = sm - v(val)
            if not (depth >= a["frac"] * sm and depth >= a["absdepth"] and (qq - pp) >= a["minsep"]):
                del peaks[k if v(pp) < v(qq) else k + 1]
                changed = True
                break
    if len(peaks) <= 1:
        return [p]
    subs, start = [], A
    for k in range(len(peaks) - 1):
        cut = valley(peaks[k], peaks[k + 1])
        subs.append([start, cut])
        start = cut + 1
    subs.append([start, B])
    out = []
    for s0, s1 in subs:
        seg = wf[s0:s1 + 1].astype(float)
        k = int(np.argmax(seg))
        out.append(dict(t_start=s0, t_end=s1, t_max=s0 + k, peak=float(seg[k]), area=float(seg.sum())))
    return out


def segments(wf, pad=8):
    nz = np.flatnonzero(wf != 0)
    if not len(nz):
        return []
    br = np.flatnonzero(np.diff(nz) > 2 * pad) + 1
    out = []
    for g in np.split(nz, br):
        out.append((max(0, g[0] - pad), min(len(wf), g[-1] + pad + 1)))
    return out


def to_short(decon_row, scale=100.0):
    """static_cast<short>(scale*x) as gcc/x86-64 executes it: cvttsd2si to int32 (out of range -> INT32_MIN), then
    the low 16 bits.  Out-of-range values are UB in C++; on this build they WRAP (|x| > 327.67 PE/tick)."""
    x = np.trunc(scale * decon_row.astype(np.float64))
    i32 = np.where((x >= -2**31) & (x < 2**31), x, -2**31).astype(np.int64)
    return ((i32 + 32768) % 65536 - 32768).astype(np.int64)


def subpulses(decon_row):
    """All post-split sub-pulses (trace-local ticks), BEFORE the hit_threshold drop."""
    wf = to_short(decon_row)
    out = []
    for lo, hi in segments(wf):
        for p in sliding_window(wf[lo:hi]):
            for s in split_pulse(wf[lo:hi], p):
                s = dict(s)
                for k in ("t_start", "t_end", "t_max"):
                    s[k] += lo
                out.append(s)
    return wf, out


def merge_saturated(wf, subs, cores, pre, post):
    """Candidate sat_hit_merge rule: every sub-pulse (incl. sub-threshold ones) touching [ci - pre, cj + post] of a
    rail core [ci, cj) becomes ONE pulse spanning [min t_start, max t_end]; area = waveform sum over the span."""
    subs = sorted(subs, key=lambda s: s["t_start"])
    used = [False] * len(subs)
    merged = []
    for ci, cj in sorted(cores):
        lo, hi = ci - pre, cj - 1 + post
        idx = [k for k, s in enumerate(subs) if not used[k] and s["t_end"] >= lo and s["t_start"] <= hi]
        if not idx:
            continue
        for k in idx:
            used[k] = True
        a = min(subs[k]["t_start"] for k in idx)
        b = max(subs[k]["t_end"] for k in idx)
        seg = wf[a:b + 1].astype(float)
        k = int(np.argmax(seg))
        merged.append(dict(t_start=a, t_end=b, t_max=a + k, peak=float(seg[k]), area=float(seg.sum()), merged=len(idx)))
    rest = [s for k, s in enumerate(subs) if not used[k]]
    return sorted(merged + rest, key=lambda s: s["t_start"])
