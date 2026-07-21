#!/usr/bin/env python3
"""PD-family reliability study from the evt298567 hand-scan labels.

Uses the human-confirmed matches (positives) and human-rejected auto-matches
(negatives) of the wfresc scan to
  (1) measure per-channel / per-family meas/pred ratios against the cathode-XA
      ruler -> implied calibration constants vs the adopted f7c66ab8 factors, and
  (2) test whether the membrane/wall XAs and the PMTs (z-wall, bottom) help or
      hurt the KS shape test and the chi2 -> keep / exclude / recalibrate.

Inputs (read-only):
  work/ql_labels/wfresc/labels-evt298567.json   (44 matches, 84 rejected_auto;
      each entry embeds op_pes / op_pe_err / pred_pes, all 40 channels)
  work/039252_0_wfresc/calib-evt298567.json     (flash sat/cov arrays, opdet
      active flags, quality_params of the production fit)

Outputs: PNGs into docs/qlmatch/pics/ + a text report on stdout.

See docs/qlmatch/17_pdvd-pd-family-reliability-evt298567.md for the write-up.
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd"
LABELS = sys.argv[1] if len(sys.argv) > 1 else \
    f"{BASE}/work/ql_labels/wfresc/labels-evt298567.json"
DUMP = sys.argv[2] if len(sys.argv) > 2 else \
    f"{BASE}/work/039252_0_wfresc/calib-evt298567.json"
PICS = sys.argv[3] if len(sys.argv) > 3 else \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pics")

# Channel families (OpDet index = array index), from
# cfg/pgrapher/experiment/protodunevd/qlmatching.jsonnet:120-140.
FAMILIES = {
    "cathode XA": [4, 5, 6, 7, 8, 9, 10, 11],
    "membrane XA": [0, 1, 2, 3, 12, 13, 18, 19],
    "z-wall PMT": [14, 15, 16, 17, 20, 21, 22, 23],
    "bottom PMT": list(range(24, 40)),
}
FAM_COLOR = {
    "cathode XA": "tab:red",
    "membrane XA": "tab:blue",
    "z-wall PMT": "tab:green",
    "bottom PMT": "tab:purple",
}
# Adopted per-family data scale factors (toolkit f7c66ab8, doc 12).
CURRENT_SCALE = {"cathode XA": 10.116, "membrane XA": 1.655,
                 "z-wall PMT": 0.352, "bottom PMT": 0.352}

# Selection thresholds for the ruler analysis (stated in the doc).
MIN_PRED_FAMILY = 20.0   # PE: family sum must predict at least this much
MIN_PRED_CHANNEL = 5.0   # PE: per-channel ratio needs at least this much pred
MIN_GOOD_CATH = 4        # good cathode channels required for the ruler

# vd_surface_flags relax-channel sets (qlmatching.jsonnet:144-146,236-238):
# the close-to-PMT chi2 widening is restricted to the PD channels of the
# surface the activity is actually near. Which sets fired per bundle is not in
# the dump; Part A infers the combination per entry by exact chi2 closure.
RELAX_SETS = {
    "anode(bottom PMTs)": list(range(24, 40)),  # bottom volume only
    "wall ylo": [1, 3, 13, 19],
    "wall yhi": [0, 2, 12, 18],
}


def relax_combos(apa):
    """Candidate relax_channels unions, cheapest first (single sets, then
    unions). The anode set applies only in the bottom volume (apa 0)."""
    keys = ["wall ylo", "wall yhi"]
    if apa == 0:
        keys = ["anode(bottom PMTs)"] + keys
    combos = []
    n = len(keys)
    for bits in range(1, 2 ** n):
        sel = [keys[i] for i in range(n) if bits & (1 << i)]
        chans = sorted(set(sum((RELAX_SETS[k] for k in sel), [])))
        combos.append((tuple(sel), chans))
    combos.sort(key=lambda kc: len(kc[0]))
    return combos


def fam_of(ch):
    for f, chans in FAMILIES.items():
        if ch in chans:
            return f
    raise KeyError(ch)


def load():
    labels = json.load(open(LABELS))
    dump = json.load(open(DUMP))
    flash_by_gid = {f["gid"]: f for f in dump["flashes"]}
    nchan = dump["nchan"]
    active = np.array([1 if od["active"] else 0 for od in dump["opdets"]])
    qp = dict(dump["quality_params"])
    # dump_calib does not round-trip the low-PE error knobs; without them the
    # per_opdet_perr fallback picks the SBND branch and the closure test fails.
    # PDVD production values from cfg/.../protodunevd/qlmatching.jsonnet:356-357.
    qp.setdefault("pe_err_lowpe_frac", 2.0)
    qp.setdefault("pe_err_lowpe_knee", 10.0)
    return labels, dump, flash_by_gid, nchan, active, qp


def entry_arrays(entry, flash_by_gid, nchan):
    """meas, meas_err, pred, sat, cov for one label entry."""
    meas = np.asarray(entry["op_pes"], dtype=float)
    merr = np.asarray(entry["op_pe_err"], dtype=float)
    pred = np.asarray(entry["pred_pes"], dtype=float)
    fl = flash_by_gid[entry["flash_gid"]]
    sat = np.asarray(fl.get("sat", [0] * nchan), dtype=int)
    cov = np.asarray(fl.get("cov", [1.0] * nchan), dtype=float)
    return meas, merr, pred, sat, cov


# ----------------------------------------------------------------------------
# Faithful reimplementation of TimingTPCBundle::examine_bundle()
# (match/src/TimingTPCBundle.cxx:208-296) driven by the dump's quality_params.
# ----------------------------------------------------------------------------

def per_opdet_perr(qp, pred, meas_err):
    if not qp.get("pe_err_on_pred", False):
        return meas_err
    lowpe = qp.get("pe_err_lowpe_frac", -1.0)
    frac = qp.get("pe_err_frac", 0.3)
    floor = qp.get("pe_err_floor", 0.3)
    if lowpe >= 0.0:
        knee = qp.get("pe_err_lowpe_knee", 4.0)
        rel = frac + (lowpe - frac) * np.exp(-pred / knee)
        return np.sqrt((rel * pred) ** 2 + floor ** 2)
    return np.where(pred < qp.get("pe_err_knee", 1.0), floor, frac * pred)


def examine_bundle(meas, merr, pred, sat, mask, qp, flags, relax=None):
    """Return (ks_dis, chi2, ndf) with production semantics.

    mask: per-channel 0/1; masked channels leave chi2/ndf always and leave the
    KS CDFs when mask_ks (production PDVD: mask_ks=true).
    relax: channel list eligible for the close-to-PMT widening
    (TimingTPCBundle relax_channels); None = all channels eligible.
    """
    mask = np.asarray(mask)
    # KS: masked channels -> (0,0) when mask_ks, then self-normalized CDFs in
    # channel-index order, max |diff|.
    if qp.get("mask_ks", False):
        m = np.where(mask == 1, meas, 0.0)
        p = np.where(mask == 1, pred, 0.0)
    else:
        m, p = meas.copy(), pred.copy()
    tm, tp = m.sum(), p.sum()
    ks = 1.0
    if tp > 0:
        cm = np.cumsum(m / tm if tm > 0 else m)
        cp = np.cumsum(p / tp)
        ks = float(np.max(np.abs(cm - cp)))

    ndf_knee = qp.get("pe_ndf_knee", 1.0)
    sat_inflate = qp.get("chi2_sat_inflate", 0.0)
    relax_flag = qp.get("chi2_relax", False)
    pmt_excess = qp.get("chi2_pmt_excess", 350.0)
    pmt_ratio = qp.get("chi2_pmt_ratio", 1.3)
    pmt_inflate = qp.get("chi2_pmt_inflate", 0.5)
    close_pmt = bool(flags.get("close_to_PMT", False))
    relax_ok = (lambda j: True) if relax is None else \
        (lambda j, s=set(relax): j in s)

    chi2, ndf = 0.0, 0
    max_cur, max_bin = -1.0, -1
    for j in range(len(meas)):
        if mask[j] == 0:
            continue
        if not (meas[j] < ndf_knee and pred[j] < ndf_knee):
            ndf += 1
        perr = float(per_opdet_perr(qp, pred[j], merr[j]))
        denom = meas[j] + perr * perr
        if relax_flag and close_pmt and relax_ok(j) and \
           meas[j] - pred[j] > pmt_excess and meas[j] > pmt_ratio * pred[j]:
            denom += (meas[j] * pmt_inflate) ** 2
        if sat_inflate > 0 and sat[j]:
            denom += (meas[j] * sat_inflate) ** 2
        cur = (pred[j] - meas[j]) ** 2 / denom
        chi2 += cur
        if cur > max_cur:
            max_cur, max_bin = cur, j
    if relax_flag and max_bin >= 0 and meas[max_bin] == 0 and pred[max_bin] > 0:
        chi2 -= max_cur - 1
    return ks, chi2, ndf


# ----------------------------------------------------------------------------
# Part A: closure -- reproduce the dumped metrics from the embedded arrays.
# ----------------------------------------------------------------------------

def part_a(entries, flash_by_gid, nchan, active, qp):
    """Closure gate. For close_to_PMT entries the per-bundle relax_channels
    (which PD-surface sets fired) is not in the dump; infer it by exact chi2
    match over the small set of physically possible unions, and stash the
    result on the entry (e["_relax"]) for Part C."""
    print("== Part A: closure (recompute chi2/ndf/ks vs stored metrics) ==")
    dev = []
    relax_tally = {}
    for e in entries:
        meas, merr, pred, sat, cov = entry_arrays(e, flash_by_gid, nchan)
        mt = e["metrics"]
        e["_relax"] = None
        if e["flags"].get("close_to_PMT", False):
            best = None
            for names, chans in relax_combos(e["apa"]):
                ks, chi2, ndf = examine_bundle(meas, merr, pred, sat, active,
                                               qp, e["flags"], relax=chans)
                rel = abs(chi2 - mt["chi2"]) / max(1.0, mt["chi2"])
                if best is None or rel < best[0]:
                    best = (rel, names, chans, ks, chi2, ndf)
                if rel < 1e-9:
                    break
            _, names, chans, ks, chi2, ndf = best
            e["_relax"] = chans
            relax_tally[names] = relax_tally.get(names, 0) + 1
        else:
            ks, chi2, ndf = examine_bundle(meas, merr, pred, sat, active, qp,
                                           e["flags"])
        dev.append((abs(ks - mt["ks_dis"]),
                    abs(chi2 - mt["chi2"]) / max(1.0, mt["chi2"]),
                    abs(ndf - mt["ndf"]),
                    e["flash_gid"], e["cluster_idents"]))
    dev.sort(key=lambda d: d[1], reverse=True)
    worst = dev[0]
    print(f"  entries: {len(entries)}")
    print(f"  max |dks|      = {max(d[0] for d in dev):.3e}")
    print(f"  max |dchi2|/c2 = {max(d[1] for d in dev):.3e}")
    print(f"  max |dndf|     = {max(d[2] for d in dev):.0f}")
    print(f"  worst chi2 entry: flash_gid={worst[3]} clus={worst[4]}")
    print(f"  inferred relax sets (close_to_PMT entries): {relax_tally}")
    ok = max(d[0] for d in dev) < 1e-6 and max(d[1] for d in dev) < 1e-6 \
        and max(d[2] for d in dev) == 0
    print(f"  closure {'PASS' if ok else 'FAIL'}")
    return ok


# ----------------------------------------------------------------------------
# Part B: cathode-ruler ratios.
# ----------------------------------------------------------------------------

def part_b(matches, flash_by_gid, nchan, active, qp):
    print("\n== Part B: cathode-ruler meas/pred ratios (44 matches) ==")
    cath = FAMILIES["cathode XA"]
    per_family = {f: [] for f in FAMILIES}       # normalized R_F/R_cath per match
    per_family_det = {f: [] for f in FAMILIES}   # same, detected (meas>0) chs only
    per_channel = {c: [] for c in range(nchan)}  # normalized r_i per match
    n_used, n_trunc, n_ruler = 0, 0, 0
    scatter = {f: ([], []) for f in FAMILIES}    # (pred, meas) points, good ch
    avail = {f: [0, 0, 0] for f in FAMILIES}     # [covered, saturated, total]
    turnon = {f: ([], []) for f in FAMILIES}     # (pred, detected?) good ch

    for e in matches:
        meas, merr, pred, sat, cov = entry_arrays(e, flash_by_gid, nchan)
        good = (active == 1) & (sat == 0) & (cov >= 1.0) & (pred > 0)
        for f, chans in FAMILIES.items():
            for c in chans:
                if active[c] == 0:
                    continue
                avail[f][2] += 1
                if cov[c] >= 1.0:
                    avail[f][0] += 1
                if sat[c]:
                    avail[f][1] += 1
                if good[c]:
                    scatter[f][0].append(pred[c])
                    scatter[f][1].append(meas[c])
                    turnon[f][0].append(pred[c])
                    turnon[f][1].append(1 if meas[c] > 0 else 0)
        if e["flags"].get("window_truncated", False):
            n_trunc += 1
            continue
        gc = [c for c in cath if good[c]]
        pc = sum(pred[c] for c in gc)
        if len(gc) < MIN_GOOD_CATH or pc < MIN_PRED_FAMILY:
            n_ruler += 1
            continue
        r_cath = sum(meas[c] for c in gc) / pc
        if r_cath <= 0:
            n_ruler += 1
            continue
        n_used += 1
        for f, chans in FAMILIES.items():
            gch = [c for c in chans if good[c]]
            pf = sum(pred[c] for c in gch)
            if pf >= MIN_PRED_FAMILY:
                rf = sum(meas[c] for c in gch) / pf
                per_family[f].append(rf / r_cath)
            # detected-only: given the channel saw ANY light, what scale?
            dch = [c for c in gch if meas[c] > 0]
            pfd = sum(pred[c] for c in dch)
            if pfd >= MIN_PRED_FAMILY:
                per_family_det[f].append(
                    (sum(meas[c] for c in dch) / pfd) / r_cath)
        for c in range(nchan):
            if good[c] and pred[c] >= MIN_PRED_CHANNEL:
                per_channel[c].append((meas[c] / pred[c]) / r_cath)

    print(f"  matches entering calibration: {n_used} "
          f"(excluded: {n_trunc} window_truncated, {n_ruler} weak ruler)")
    print("\n  per-family channel availability over the 44 matches "
          "(unmasked channels):")
    for f in FAMILIES:
        c, s, t = avail[f]
        print(f"    {f:<12} covered {c:4d}/{t:4d} ({100*c/t:5.1f}%)  "
              f"railed {s:3d}/{t:4d} ({100*s/t:4.1f}%)")
    print(f"\n  {'family':<12} {'n':>3} {'median R_F/R_cath':>18} "
          f"{'16-84%':>16} {'current scale':>14} {'implied scale':>14}")
    fam_median = {}
    for f in FAMILIES:
        v = np.array(per_family[f])
        if len(v) == 0:
            print(f"  {f:<12} {0:>3}  (no matches pass the pred threshold)")
            continue
        med = np.median(v)
        lo, hi = np.percentile(v, [16, 84])
        fam_median[f] = med
        print(f"  {f:<12} {len(v):>3} {med:>18.3f} "
              f"[{lo:>6.3f},{hi:>6.3f}] {CURRENT_SCALE[f]:>14.3f} "
              f"{CURRENT_SCALE[f]*med:>14.3f}")
    print("\n  detected-only (channels with meas>0): gain given detection")
    fam_median_det = {}
    for f in FAMILIES:
        v = np.array(per_family_det[f])
        if len(v) == 0:
            print(f"    {f:<12} n=0")
            continue
        med = np.median(v)
        lo, hi = np.percentile(v, [16, 84])
        fam_median_det[f] = med
        print(f"    {f:<12} n={len(v):>2} median {med:6.3f} "
              f"[{lo:6.3f},{hi:6.3f}]  -> scale {CURRENT_SCALE[f]*med:7.3f}")
    print("\n  detected fraction vs predicted PE (good channels, all matches):")
    edges = np.array([0.5, 2, 5, 10, 20, 50, 100, 1e9])
    hdr = "".join(f"{f'[{edges[i]:g},{edges[i+1]:g})':>12}"
                  for i in range(len(edges) - 2)) + f"{'>=100':>12}"
    print(f"    {'family':<12}{hdr}")
    for f in FAMILIES:
        p = np.array(turnon[f][0])
        d = np.array(turnon[f][1])
        cells = []
        for i in range(len(edges) - 1):
            m = (p >= edges[i]) & (p < edges[i + 1])
            cells.append(f"{d[m].mean():5.2f}({m.sum():3d})" if m.sum()
                         else "     --    ")
        print(f"    {f:<12}" + "".join(f"{c:>12}" for c in cells))

    print(f"\n  per-channel normalized ratio (median [16-84%], n) "
          f"for pred >= {MIN_PRED_CHANNEL} PE:")
    chan_stats = {}
    for c in range(nchan):
        v = np.array(per_channel[c])
        if len(v) < 3:
            continue
        med = np.median(v)
        lo, hi = np.percentile(v, [16, 84])
        chan_stats[c] = (med, lo, hi, len(v))
        print(f"    ch {c:2d} [{fam_of(c):<12}] {med:6.3f} "
              f"[{lo:6.3f},{hi:6.3f}]  n={len(v)}")

    # Figure 1: per-channel normalized ratio.
    fig, ax = plt.subplots(figsize=(11, 5))
    for c, (med, lo, hi, n) in chan_stats.items():
        f = fam_of(c)
        ax.errorbar([c], [med], yerr=[[med - lo], [hi - med]], fmt="o",
                    color=FAM_COLOR[f], capsize=3)
    for c in range(nchan):
        if active[c] == 0:
            ax.axvspan(c - 0.4, c + 0.4, color="0.85", zorder=0)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    handles = [plt.Line2D([], [], marker="o", ls="", color=FAM_COLOR[f],
                          label=f) for f in FAMILIES]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="0.85",
                                 label="masked (no prediction)"))
    ax.legend(handles=handles, ncol=3, fontsize=9)
    ax.set_xlabel("OpDet channel")
    ax.set_ylabel("(meas/pred) / R_cath  per match, median")
    ax.set_title("PDVD evt298567 hand-scan matches: per-channel response "
                 "vs the cathode-XA ruler")
    ax.set_yscale("log")
    ax.set_xticks(range(0, nchan, 2))
    fig.tight_layout()
    fig.savefig(os.path.join(PICS, "pd_family_ratio_per_channel_evt298567.png"),
                dpi=130)
    plt.close(fig)

    # Figure 2: per-family distributions. Zeros (family saw NO light while the
    # cathode did) are real and common -- show them as an explicit "R=0" bar
    # instead of letting the log axis swallow them.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    zbar_lo, zbar_hi = 1.5e-3, 3e-3
    for ax, f in zip(axes, ["membrane XA", "z-wall PMT", "bottom PMT"]):
        v = np.array(per_family[f])
        nz = int((v == 0).sum()) if len(v) else 0
        vp = v[v > 0] if len(v) else v
        if len(vp):
            bins = np.logspace(-2, np.log10(vp.max() * 1.5), 15)
            ax.hist(vp, bins=bins, color=FAM_COLOR[f], alpha=0.75)
        if nz:
            ax.bar([np.sqrt(zbar_lo * zbar_hi)], [nz],
                   width=zbar_hi - zbar_lo, color=FAM_COLOR[f], alpha=0.75,
                   hatch="//", edgecolor="k", label=f"R=0 ({nz})")
        if len(v):
            med = np.median(v)
            if med > 0:
                ax.axvline(med, color="k", lw=1.2, label=f"median {med:.2f}")
            else:
                ax.plot([], [], " ", label="median = 0")
        ax.axvline(1.0, color="tab:red", lw=1.0, ls="--",
                   label="cathode ruler (=1)")
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 50)
        ax.set_xlabel("R_F / R_cath per match")
        ax.set_title(f"{f} (n={len(v)})")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("matches")
    fig.suptitle("Family response relative to cathode XAs, evt298567 matches")
    fig.tight_layout()
    fig.savefig(os.path.join(PICS, "pd_family_ratio_dist_evt298567.png"),
                dpi=130)
    plt.close(fig)

    # Figure 3: meas vs pred scatter per family.
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharex=True, sharey=True)
    for ax, f in zip(axes, FAMILIES):
        p, m = np.array(scatter[f][0]), np.array(scatter[f][1])
        ax.plot(p, m, ".", ms=4, color=FAM_COLOR[f], alpha=0.6)
        lim = [1e-1, 1e5]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(f)
        ax.set_xlabel("predicted PE")
    axes[0].set_ylabel("measured PE")
    fig.suptitle("Measured vs predicted PE per good channel, "
                 "evt298567 hand-confirmed matches")
    fig.tight_layout()
    fig.savefig(os.path.join(PICS, "pd_family_meas_vs_pred_evt298567.png"),
                dpi=130)
    plt.close(fig)

    # Figure 5: detection turn-on vs predicted PE.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for f in FAMILIES:
        p = np.array(turnon[f][0])
        d = np.array(turnon[f][1])
        xs, ys, ns = [], [], []
        for i in range(len(edges) - 1):
            m = (p >= edges[i]) & (p < edges[i + 1])
            if m.sum() >= 3:
                xs.append(np.sqrt(edges[i] * min(edges[i + 1], 300.0)))
                ys.append(d[m].mean())
                ns.append(m.sum())
        ax.plot(xs, ys, "o-", color=FAM_COLOR[f], label=f)
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("predicted PE (channel, good = covered/unsaturated)")
    ax.set_ylabel("fraction with measured PE > 0")
    ax.set_title("Detection turn-on per PD family, evt298567 matches")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PICS, "pd_family_detect_turnon_evt298567.png"),
                dpi=130)
    plt.close(fig)

    return fam_median, fam_median_det


# ----------------------------------------------------------------------------
# Part C: include/exclude discrimination test.
# ----------------------------------------------------------------------------

def rank_auc(pos, neg):
    """P(pos < neg) with tie=0.5 -- 'lower metric on true matches' AUC."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    wins = (pos[:, None] < neg[None, :]).sum() + \
        0.5 * (pos[:, None] == neg[None, :]).sum()
    return wins / (len(pos) * len(neg))


def part_c(matches, rejects, flash_by_gid, nchan, active, qp, fam_scale):
    print("\n== Part C: KS/chi2 channel-set variants, "
          "matches (44) vs rejected_auto (84) ==")
    fam_mask = {f: np.zeros(nchan, dtype=int) for f in FAMILIES}
    for f, chans in FAMILIES.items():
        fam_mask[f][chans] = 1

    variants = {
        "V0 production set": active,
        "V1 cathode XA only": active * fam_mask["cathode XA"],
        "V2 drop membrane XA": active * (1 - fam_mask["membrane XA"]),
        "V3 XAs only (no PMTs)": active * (fam_mask["cathode XA"]
                                           | fam_mask["membrane XA"]),
        "V4 drop z-wall PMTs": active * (1 - fam_mask["z-wall PMT"]),
        "V5 drop bottom PMTs": active * (1 - fam_mask["bottom PMT"]),
    }
    # V5: production channel set, per-family rescaled predictions. Uses the
    # detected-only medians: the with-zeros medians are 0 for the membrane XAs
    # (a scale of 0 is family removal, not a calibration -- that case is V2).
    scale_vec = np.ones(nchan)
    for f, chans in FAMILIES.items():
        if f in fam_scale:
            scale_vec[chans] = fam_scale[f]
    print(f"  V6 per-family pred scales (detected-only medians): "
          f"{ {f: round(float(s), 3) for f, s in fam_scale.items()} }")

    def metrics_for(entries, mask, rescale=False):
        out_ks, out_c2n = [], []
        for e in entries:
            meas, merr, pred, sat, cov = entry_arrays(e, flash_by_gid, nchan)
            if rescale:
                pred = pred * scale_vec
            ks, chi2, ndf = examine_bundle(meas, merr, pred, sat, mask, qp,
                                           e["flags"], relax=e.get("_relax"))
            out_ks.append(ks)
            out_c2n.append(chi2 / ndf if ndf > 0 else np.inf)
        return np.array(out_ks), np.array(out_c2n)

    rows = []
    all_variants = list(variants.items()) + \
        [("V6 rescaled families", active)]
    for name, mask in all_variants:
        rescale = name.startswith("V6")
        ks_p, c2_p = metrics_for(matches, mask, rescale)
        ks_n, c2_n = metrics_for(rejects, mask, rescale)
        row = {
            "name": name,
            "auc_ks": rank_auc(ks_p, ks_n),
            "auc_c2": rank_auc(c2_p, c2_n),
            "med_ks_p": np.median(ks_p), "med_ks_n": np.median(ks_n),
            "med_c2_p": np.median(c2_p), "med_c2_n": np.median(c2_n),
            # high-consistency ladder occupancy at the production thresholds
            "p_clean": np.mean((ks_p < qp["hc_clean_ks"])
                               & (c2_p < qp["hc_clean_c2"])),
            "n_clean": np.mean((ks_n < qp["hc_clean_ks"])
                               & (c2_n < qp["hc_clean_c2"])),
            "p_good": np.mean((ks_p < qp["hc_good_ks"])
                              & (c2_p < qp["hc_good_c2"])),
            "n_good": np.mean((ks_n < qp["hc_good_ks"])
                              & (c2_n < qp["hc_good_c2"])),
            "ks_p": ks_p, "ks_n": ks_n, "c2_p": c2_p, "c2_n": c2_n,
        }
        rows.append(row)

    print(f"  {'variant':<24} {'AUC(ks)':>8} {'AUC(c2n)':>9} "
          f"{'med ks m/r':>14} {'med c2n m/r':>16} "
          f"{'clean m/r':>12} {'good m/r':>12}")
    for r in rows:
        print(f"  {r['name']:<24} {r['auc_ks']:>8.3f} {r['auc_c2']:>9.3f} "
              f"{r['med_ks_p']:>6.3f}/{r['med_ks_n']:<6.3f} "
              f"{r['med_c2_p']:>7.1f}/{r['med_c2_n']:<7.1f} "
              f"{r['p_clean']:>5.2f}/{r['n_clean']:<5.2f} "
              f"{r['p_good']:>5.2f}/{r['n_good']:<5.2f}")

    # Per-family chi2 share among matches under production config.
    print("\n  per-family chi2 share (production set, 44 matches):")
    fam_chi2 = {f: 0.0 for f in FAMILIES}
    fam_n = {f: 0 for f in FAMILIES}
    for e in matches:
        meas, merr, pred, sat, cov = entry_arrays(e, flash_by_gid, nchan)
        for f, chans in FAMILIES.items():
            m1 = np.zeros(nchan, dtype=int)
            m1[chans] = 1
            m1 = m1 * active
            _, c2, nd = examine_bundle(meas, merr, pred, sat, m1,
                                       {**qp, "chi2_relax": False}, e["flags"])
            fam_chi2[f] += c2
            fam_n[f] += nd
    for f in FAMILIES:
        avg = fam_chi2[f] / fam_n[f] if fam_n[f] else float("nan")
        print(f"    {f:<12} sum chi2 = {fam_chi2[f]:9.1f}  "
              f"sum ndf = {fam_n[f]:4d}  chi2/ndf = {avg:6.2f}")

    # Figure 4: variant separation.
    fig, axes = plt.subplots(2, len(rows), figsize=(3.0 * len(rows), 6),
                             sharex="row")
    ks_bins = np.linspace(0, 1, 26)
    c2_bins = np.logspace(-1, 3.5, 26)
    for i, r in enumerate(rows):
        ax = axes[0, i]
        ax.hist(r["ks_p"], bins=ks_bins, color="tab:green", alpha=0.6,
                label="matches")
        ax.hist(r["ks_n"], bins=ks_bins, color="tab:orange", alpha=0.6,
                label="rejected")
        ax.set_title(f"{r['name']}\nAUC(ks)={r['auc_ks']:.3f}", fontsize=9)
        ax.set_xlabel("ks_dis")
        if i == 0:
            ax.set_ylabel("entries")
            ax.legend(fontsize=8)
        ax = axes[1, i]
        ax.hist(np.clip(r["c2_p"], c2_bins[0], c2_bins[-1]), bins=c2_bins,
                color="tab:green", alpha=0.6)
        ax.hist(np.clip(r["c2_n"], c2_bins[0], c2_bins[-1]), bins=c2_bins,
                color="tab:orange", alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("chi2/ndf")
        ax.set_title(f"AUC(c2n)={r['auc_c2']:.3f}", fontsize=9)
        if i == 0:
            ax.set_ylabel("entries")
    fig.suptitle("KS / chi2 separation of hand-confirmed vs hand-rejected "
                 "matches under channel-set variants (evt298567)")
    fig.tight_layout()
    fig.savefig(os.path.join(PICS, "pd_family_metric_variants_evt298567.png"),
                dpi=130)
    plt.close(fig)
    return rows


def main():
    labels, dump, flash_by_gid, nchan, active, qp = load()
    matches = labels["matches"]
    rejects = labels["rejected_auto"]
    print(f"labels: {LABELS}")
    print(f"dump:   {DUMP}")
    print(f"matches={len(matches)} rejected_auto={len(rejects)} nchan={nchan}")
    print(f"active channels: {[int(c) for c in np.where(active == 1)[0]]}")
    os.makedirs(PICS, exist_ok=True)

    part_a(matches + rejects, flash_by_gid, nchan, active, qp)
    fam_median, fam_median_det = part_b(matches, flash_by_gid, nchan, active,
                                        qp)
    part_c(matches, rejects, flash_by_gid, nchan, active, qp, fam_median_det)


if __name__ == "__main__":
    main()
