#!/usr/bin/env python3
"""Per-channel (per-PD) light-gain calibration for the PDHD Q/L matcher (run 29107).

The optical SP chain deconvolves all 160 photon detectors with just TWO SPE
templates (FBK / HPK, one shape per *type*); residual per-channel SiPM gain is not
removed and biases the bundle chi2/KS.  This fits that residual into the existing
per-channel ``measured_pe_scale`` knob (Opflash::init multiplies MEASURED PE by it
before pe_err/total/fired are derived -- already plumbed, no C++ change).

Granularity (user-confirmed): grouped **block x type** base + per-channel breakout
only for the few well-sampled, tight-scatter outliers.  This mirrors the prior
SPE-template study (``pdhd-spe-template-tuning.md``): per-channel *shape* tuning
overfit; a per-*type* correction generalised.  Here the dominant signal is indeed
per-TYPE -- FBK reads low everywhere (the documented FBK tail over-subtraction).

Inputs are the dumps the cfg currently ships (work/029107_{0..3}/calib-evt*.json,
vuv_eff 0.01281, APA0 scale 1.14) and the hand-scan labels; both carry ``pred_pe``
at the current model, so the per-channel ratio is read straight from
``op_pes``/``pred_pes`` -- no offline re-prediction needed.

Anchor filter (user spec): low KS, NOT flag_PMT, NOT flag_wtrunc; ``at_x_boundary``
KEPT this time (only close_to_PMT | window_truncated dropped).

  measured_pe_scale[ch] = base(block,type) x outlier_override(ch)

  * base(block,type) = current_block_scale x light-weighted median(pred/meas) over
    the group's clean anchors.  +x (block-0) bases are renormalised by their
    meas-weighted mean so the +x integral -- hence the just-shipped vuv_eff -- is
    held fixed; -x bases use raw ratios (they subsume/refine the old 1.14/1.0).
  * outlier_override: channels significantly AND materially off their group base
    (>3 standard errors and >0.20 absolute), well sampled (N>=12).  Everything else
    shrinks to its group base; static-masked and under-sampled (N<8) channels stay
    at the block default.

Then pe_err_frac is refit by method-of-moments on the gain-corrected high-PE
residuals (the "above the floor" fractional term only; floor + low-PE knee fixed).

Usage:  python3 fit_perchannel_scale.py            # default anchor set
        python3 fit_perchannel_scale.py --drop-xb  # also drop at_x_boundary (robustness)
"""
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PD = os.path.normpath(os.path.join(HERE, ".."))

EVTS = {983: 0, 991: 1, 999: 2, 1007: 3}

# FBK opchannels = template_index 0 in pdhd-spe-templates.json (68 ch); rest HPK.
FBK = {4, 14, 24, 34, 40, 42, 45, 46, 47, 49, 50, 52, 55, 56, 57, 59, 60, 62, 65,
       66, 67, 69, 70, 72, 75, 76, 77, 79, 84, 85, 86, 87, 94, 95, 96, 97, 104,
       105, 106, 107, 114, 115, 116, 117, 120, 121, 124, 125, 127, 129, 130, 131,
       134, 135, 137, 139, 140, 141, 144, 145, 147, 149, 150, 151, 154, 155, 157, 159}
STATIC = {3, 86, 87, 97, 107, 116, 117}     # cfg ch_mask
KS_MAX = 0.2
PMIN = 3.0          # channel "lit in prediction" threshold for ratio pairs
OMIN = 0.5          # channel "lit in measurement" threshold
N_OUT = 12          # min anchors to qualify a channel for an individual override
N_SPARSE = 8        # below this -> never break out (group default)
MAD_CEIL = 0.25     # per-anchor scatter ceiling: above it the deviation is anchor
                    # noise (geometry/model), not a stable per-channel gain -> shrink
HIPE = 8.0          # pred threshold for the pe_err_frac method-of-moments fit


def block(c):
    return 0 if c < 80 else (1 if c < 120 else 2)


def typ(c):
    return "FBK" if c in FBK else "HPK"


def cur_scale(c):
    """measured_pe_scale the dumps were produced at (cfg current)."""
    return 1.14 if c >= 120 else 1.0


def clean(m, drop_xb):
    f = m["flags"]; mt = m["metrics"]
    if f.get("close_to_PMT") or f.get("window_truncated"):
        return False
    if drop_xb and f.get("at_x_boundary"):
        return False
    return mt.get("ks_dis", 9) < KS_MAX


def load_anchors(drop_xb):
    anchors = []
    for evt in EVTS:
        L = json.load(open("%s/work/ql_labels/labels-evt%d.json" % (PD, evt)))["matches"]
        for m in L:
            if clean(m, drop_xb):
                m["_evt"] = evt
                anchors.append(m)
    return anchors


def per_channel(anchors):
    """ch -> list of (meas, pred) pairs over anchors where both lit (mask excluded)."""
    pts = defaultdict(list)
    for m in anchors:
        op = m["op_pes"]; pr = m["pred_pes"]
        for c in range(160):
            if c in STATIC:
                continue
            if pr[c] > PMIN and op[c] > OMIN:
                pts[c].append((op[c], pr[c]))
    return pts


def lw_ratio(pairs):
    """light-weighted pred/meas = the measured_pe_scale EXTRA factor."""
    sm = sum(o for o, p in pairs); sp = sum(p for o, p in pairs)
    return sp / sm if sm > 0 else None


def ch_stats(pairs):
    n = len(pairs)
    rr = [p / o for o, p in pairs]
    med = st.median(rr)
    mad = st.median([abs(x - med) for x in rr]) if n > 1 else 9.0
    se = (mad * 1.4826) / math.sqrt(n) if n > 1 else 9.0     # MAD->sigma, std error of median
    return n, lw_ratio(pairs), med, mad, se


def main():
    drop_xb = "--drop-xb" in sys.argv
    anchors = load_anchors(drop_xb)
    n1 = sum(1 for m in anchors if m["apa"] == 1)
    n0 = sum(1 for m in anchors if m["apa"] == 0)
    tag = " (at_x_boundary DROPPED)" if drop_xb else ""
    print("clean low-ks anchors%s: %d  (+x/apa1=%d  -x/apa0=%d)" % (tag, len(anchors), n1, n0))
    print("  per event:", {e: sum(1 for m in anchors if m["_evt"] == e) for e in EVTS})

    pts = per_channel(anchors)

    # ---- group (block x type) light-weighted extra-scale ----
    grp = defaultdict(list)
    for c, pr in pts.items():
        grp[(block(c), typ(c))] += pr
    gratio = {k: lw_ratio(v) for k, v in grp.items()}
    print("\ngroup (block x type) light-weighted pred/meas (extra factor on current scale):")
    for k in sorted(grp):
        print("  block%d %-3s: nPts=%4d  extScale=%.3f" % (k[0], k[1], len(grp[k]), gratio[k]))

    # ---- per-channel base = current_block_scale x group_extScale ----
    base = {c: cur_scale(c) * gratio.get((block(c), typ(c)), 1.0) for c in range(160)}

    # ---- outlier breakout: significant (>3 SE) AND material (>0.20) off group base ----
    outliers = {}
    sparse = []
    for c in range(160):
        if c in STATIC:
            continue
        n, ext, med, mad, se = ch_stats(pts[c]) if pts[c] else (0, None, None, None, None)
        if n < N_SPARSE:
            if n > 0:
                sparse.append(c)
            continue                                   # -> group default
        if n < N_OUT or ext is None:
            continue
        cand = cur_scale(c) * ext                      # channel's own desired scale
        gbase = base[c]
        if (mad < MAD_CEIL and abs(cand - gbase) > max(0.20, 3.0 * se)
                and (ext > 1.30 or ext < 0.70)):
            outliers[c] = cand

    scale = dict(base)
    for c, v in outliers.items():
        scale[c] = v
    for c in sparse:
        scale[c] = cur_scale(c)                        # under-sampled -> block default, no per-ch
    for c in STATIC:
        scale[c] = cur_scale(c)                        # masked: inert (zeroed in KS/chi2)

    # ---- +x (block-0) degeneracy guard: hold the +x integral (=> vuv_eff) fixed ----
    num = den = 0.0
    for m in anchors:
        if m["apa"] != 1:
            continue
        op = m["op_pes"]
        for c in range(80):
            if c in STATIC:
                continue
            num += scale[c] * op[c]; den += op[c]
    k = den / num if num > 0 else 1.0
    for c in range(80):
        if c not in STATIC:
            scale[c] *= k
    print("\n+x renorm k = %.4f  (holds the +x meas-weighted mean scale at 1.0 => vuv_eff fixed)" % k)

    # ---- report ----
    print("\nfinal block x type bases (block-0 post +x renorm):")
    seen = set()
    rep = {}
    for c in range(160):
        key = (block(c), typ(c))
        if key in seen or c in STATIC:
            continue
        # representative non-outlier channel of this group
        if c in outliers:
            continue
        seen.add(key); rep[key] = scale[c]
        print("  block%d %-3s -> %.3f" % (key[0], key[1], scale[c]))
    print("\nper-channel outlier overrides (N>=%d, >3SE & >0.20 off group; +x post-renorm):" % N_OUT)
    for c in sorted(outliers):
        n, ext, med, mad, se = ch_stats(pts[c])
        print("  ch%-3d %s blk%d  N=%2d extScale=%.2f MAD=%.2f SE=%.3f -> scale=%.2f"
              % (c, typ(c), block(c), n, ext, mad, se, scale[c]))
    print("under-sampled (N<%d) left at block default, NOT scaled:" % N_SPARSE,
          sorted(sparse), [round(lw_ratio(pts[c]), 1) for c in sorted(sparse)])

    # ---- pe_err_frac: method-of-moments on gain-corrected high-PE residuals ----
    Y = SP2 = 0.0
    nfit = 0
    for m in anchors:
        op = m["op_pes"]; pr = m["pred_pes"]
        for c in range(160):
            if c in STATIC:
                continue
            if pr[c] > HIPE and op[c] > OMIN:
                measc = op[c] * (scale[c] / cur_scale(c))    # apply NEW gain to measured
                Y += (pr[c] - measc) ** 2 - measc
                SP2 += pr[c] ** 2
                nfit += 1
    frac = math.sqrt(max(Y / SP2, 0.0))
    print("\npe_err_frac (corrected model, pred>%g, N=%d): %.3f  (current 0.43)" % (HIPE, nfit, frac))

    # ---- emit the full 160-vector + jsonnet artefacts ----
    vec = [round(scale[c], 3) for c in range(160)]
    out = {"measured_pe_scale": vec,
           "group_bases": {"block%d_%s" % k: round(v, 3) for k, v in rep.items()},
           "outliers": {str(c): round(scale[c], 3) for c in sorted(outliers)},
           "plus_x_renorm": round(k, 4),
           "pe_err_frac": round(frac, 3),
           "anchors": len(anchors)}
    dst = os.path.join(HERE, "perchannel_scale.json")
    json.dump(out, open(dst, "w"), indent=1)
    print("\nwrote %s" % dst)

    # jsonnet snippet
    print("\n--- jsonnet (measured_pe_scale) ---")
    ov = ", ".join("%d: %.2f" % (c, scale[c]) for c in sorted(outliers))
    print("    local fbk_ch = %s," % sorted(FBK))
    print("    local perch_override = {%s}," % ", ".join('"%d": %.2f' % (c, scale[c]) for c in sorted(outliers)))
    print("    local pd_scale = function(i)")
    print("      local fbk = std.member(fbk_ch, i);")
    print("      if i >= 120 then (if fbk then %.2f else %.2f)" % (rep[(2, "FBK")], rep[(2, "HPK")]))
    print("      else if i >= 80 then (if fbk then %.2f else %.2f)" % (rep[(1, "FBK")], rep[(1, "HPK")]))
    print("      else (if fbk then %.2f else %.2f);" % (rep[(0, "FBK")], rep[(0, "HPK")]))
    print("    measured_pe_scale: std.makeArray(nchan, function(i)")
    print("      if std.objectHas(perch_override, std.toString(i)) then perch_override[std.toString(i)]")
    print("      else pd_scale(i)),")
    print("    pe_err_frac: %.2f," % round(frac, 2))


if __name__ == "__main__":
    main()
