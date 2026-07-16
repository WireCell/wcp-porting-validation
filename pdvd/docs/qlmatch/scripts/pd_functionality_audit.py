#!/usr/bin/env python3
"""Per-PD-channel functionality audit for run 039252 (+ cross-run baseline).

Motivation: a colleague reports that during run 39252 the PMT system (bottom
PMTs and/or z-wall PMTs) may be non-functional, and that among the wall
(membrane) X-Arapucas the *top-sided* ones may be dead while the *bottom-sided*
ones are OK.  This script examines the data to test that, using two independent
evidence streams keyed to the WCT OpDet channel map (index 0..39 = the order of
the calib-dump 'pe' / 'pred_pe' arrays).

Stream A -- raw flash-level LIVENESS (model-independent).  The measured 'pe'
array in the dumps is raw (verified: dead channels 24/27/28/34 read exactly 0;
dim-but-alive channels read small-nonzero).  Per channel we aggregate over all
flashes of all events, per run: fraction of flashes with pe>0, sum PE, and the
median over events of the per-event max PE.  A "peer ratio" compares each
channel's brightness to the median of its same-TYPE, same-POSITION peer group
-- so a dead channel -> 0, a faulty/dim channel -> <<1, a healthy one -> ~1.
This answers "is the channel producing light at all", free of any library.

Stream B -- matching predicted-vs-measured (systematic response).  Per channel,
Sum(meas)/Sum(pred) over auto-selected matched flashes.  CIRCULARITY GUARD:
the dump 'pred_pe' already bakes in the fitted per-type data scale factors
(cathode x10.116, membrane x1.655, PMT x0.352 at QtoL 0.094), and run 39252 is
INSIDE that fit sample -- so a bare meas/pred ~ 1 is NOT evidence of health.
We therefore (i) print the type factor next to every ratio and the recovered
official-efficiency ratio ( = ratio_current x factor ), and (ii) rely on the
top-vs-bottom wall-XA comparison, where both sides share the SAME x1.655 so the
factor cancels and any asymmetry is contamination-free.  The PMT verdict is
framed on liveness + within-PMT consistency, NOT on cross-type meas/pred (the
library's cross-type normalization is explicitly untrusted, see
11_pdvd-questions-dune.md).

Repro:
    cd pdvd/docs/qlmatch
    python3 scripts/pd_functionality_audit.py            # runs 039252/253/349, _satrep dumps
    # writes pdfunc_perchannel.tsv and pdfunc_*.png next to this script,
    # prints all tables quoted in 13_pdvd-pd-functionality-run39252.md.
"""

import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PDVD = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))        # scripts/ -> .../pdvd
WORK = os.path.join(PDVD, "work")
RUNS = ("039252", "039253", "039349")

# --- WCT OpDet channel map (index 0..39) -----------------------------------
# type 0 = X-Arapuca, type 1 = PMT (from the dump 'opdets' entries).
CATH_XA   = list(range(4, 12))              # cathode X-Arapuca (x~0), reference
TOP_XA    = [0, 1, 2, 3]                     # wall/membrane XA, TOP volume (+x)
BOT_XA    = [12, 13, 18, 19]                 # wall/membrane XA, BOTTOM volume (-x)
ZWALL_PMT = [14, 15, 16, 17, 20, 21, 22, 23]  # z-wall PMTs (bottom volume)
BOTTOM_PMT = list(range(24, 40))            # PMTs behind bottom anode (x=-336.5)
AR_BLIND = {13, 29, 32, 39}                 # official eff_Ar = 0 (not "hardware dead")

GROUPS = [
    ("cathode XA (ref)", CATH_XA),
    ("wall XA TOP (+x)", TOP_XA),
    ("wall XA BOTTOM (-x)", BOT_XA),
    ("z-wall PMT", ZWALL_PMT),
    ("bottom PMT", BOTTOM_PMT),
]
# per-type data scale factor already folded into dump pred_pe (qlmatching.jsonnet)
TYPE_FACTOR = {}
for c in CATH_XA:
    TYPE_FACTOR[c] = 10.116
for c in TOP_XA + BOT_XA:
    TYPE_FACTOR[c] = 1.655
for c in ZWALL_PMT + BOTTOM_PMT:
    TYPE_FACTOR[c] = 0.352


def dump_paths(run, tag="satrep"):
    """<run>_<idx>_<tag>/calib-evt*.json for one run (idx = integer only)."""
    paths = []
    for d in sorted(glob.glob(os.path.join(WORK, "%s_*" % run))):
        tail = os.path.basename(d).split("_", 1)[1]
        if not tail.endswith("_" + tag):
            continue
        if not tail[: -len("_" + tag)].isdigit():
            continue
        paths += sorted(glob.glob(os.path.join(d, "calib-evt*.json")))
    return paths


def scan_run(run):
    """Return per-channel Stream-A + Stream-B aggregates for one run."""
    paths = dump_paths(run)
    # Stream A accumulators
    sum_pe = np.zeros(40)
    n_nonzero = np.zeros(40, int)
    n_flash = 0
    evt_max = [[] for _ in range(40)]        # per-event max PE, per channel
    active = np.zeros(40, bool)              # opdet 'active' flag (static)
    auto_masked_any = np.zeros(40, int)      # events where auto_masked
    optype = np.full(40, -1, int)
    opx = np.zeros(40); opy = np.zeros(40); opz = np.zeros(40)
    # Stream B accumulators (per channel, over auto-selected matched flashes)
    b_meas = np.zeros(40)
    b_pred = np.zeros(40)
    n_evt = 0
    for path in paths:
        d = json.load(open(path))
        n_evt += 1
        for od in d["opdets"]:
            k = od["ch"]
            optype[k] = od["type"]
            opx[k], opy[k], opz[k] = od["x"], od["y"], od["z"]
            if od["active"]:
                active[k] = True
            if od["auto_masked"]:
                auto_masked_any[k] += 1
        pe = np.array([f["pe"] for f in d["flashes"]], float)   # (nflash, 40)
        if pe.size:
            sum_pe += pe.sum(0)
            n_nonzero += (pe > 0).sum(0)
            n_flash += pe.shape[0]
            emax = pe.max(0)
            for k in range(40):
                evt_max[k].append(emax[k])
        # Stream B: aggregate pred over auto-selected bundles per flash, pair
        # each unique matched flash's measured pe once (avoid double-count).
        fpe = {f["gid"]: np.array(f["pe"], float) for f in d["flashes"]}
        pred_by_flash = {}
        for bd in d["bundles"]:
            if not bd.get("auto_selected"):
                continue
            g = bd.get("flash_gid")
            if g not in fpe:
                continue
            pred_by_flash.setdefault(g, np.zeros(40))
            pred_by_flash[g] += np.array(bd["pred_pe"], float)
        for g, pr in pred_by_flash.items():
            b_meas += fpe[g]
            b_pred += pr
    med_evt_max = np.array([np.median(v) if v else 0.0 for v in evt_max])
    return dict(run=run, n_evt=n_evt, n_flash=n_flash, sum_pe=sum_pe,
                n_nonzero=n_nonzero, frac_nonzero=n_nonzero / max(n_flash, 1),
                med_evt_max=med_evt_max, active=active,
                auto_masked_any=auto_masked_any, optype=optype,
                opx=opx, opy=opy, opz=opz, b_meas=b_meas, b_pred=b_pred)


def peer_ratio(med_evt_max):
    """Each channel's brightness / median brightness of its same-group peers."""
    pr = np.zeros(40)
    for _, chans in GROUPS:
        for k in chans:
            peers = [c for c in chans if c != k and med_evt_max[c] > 0]
            base = np.median([med_evt_max[c] for c in peers]) if peers else 0.0
            pr[k] = med_evt_max[k] / base if base > 0 else float("nan")
    return pr


def sanity(res252):
    dead = [24, 27, 28, 34]
    ds = res252["sum_pe"][dead]
    assert np.all(ds == 0), f"sanity FAIL: dead set has PE {dict(zip(dead, ds))}"
    cath = res252["med_evt_max"][CATH_XA].mean()
    others = res252["med_evt_max"][TOP_XA + BOT_XA + ZWALL_PMT + BOTTOM_PMT]
    assert cath > others.max(), "sanity FAIL: cathode not the brightest group"
    print("SANITY OK: dead set {24,27,28,34} read 0 PE; cathode XA is brightest "
          "group (map/index verified).\n")


def classify(frac, pr, active):
    if frac == 0:
        return "DEAD (no readout)"
    if not active:
        return "masked (Ar-blind/dead-flagged)"
    if np.isnan(pr):
        return "n/a"
    if pr < 0.03:
        return "NEAR-DEAD"
    if pr < 0.2:
        return "DIM"
    return "ok"


def main():
    res = {r: scan_run(r) for r in RUNS}
    sanity(res["039252"])
    r0 = res["039252"]
    pr0 = peer_ratio(r0["med_evt_max"])

    tname = {0: "XA", 1: "PMT"}
    # --- master per-channel table (run 039252) -----------------------------
    print("=" * 108)
    print("RUN 039252 -- per-channel LIVENESS (Stream A) and matched meas/pred (Stream B)")
    print("=" * 108)
    hdr = ("grp                 ch typ    x       y       z    frac>0  "
           "medMaxPE  peerR  |  B:meas/pred  (xTypeFac -> official)  class")
    for gname, chans in GROUPS:
        print("-" * 108)
        for k in chans:
            frac = r0["frac_nonzero"][k]
            mm = r0["med_evt_max"][k]
            pr = pr0[k]
            bm, bp = r0["b_meas"][k], r0["b_pred"][k]
            ratio = bm / bp if bp > 0 else float("nan")
            fac = TYPE_FACTOR[k]
            offic = ratio * fac if bp > 0 else float("nan")
            cls = classify(frac, pr, r0["active"][k])
            tag = ""
            if k in AR_BLIND:
                tag = " [Ar-blind]"
            print(f"{gname:19s} {k:2d} {tname.get(r0['optype'][k],'?'):>3} "
                  f"{r0['opx'][k]:7.1f} {r0['opy'][k]:7.1f} {r0['opz'][k]:6.1f} "
                  f"{frac:6.2f} {mm:9.1f} {pr:6.2f}  |  {ratio:9.3f}  "
                  f"(x{fac:5.3f} -> {offic:6.2f})  {cls}{tag}")
    # --- group-level meas/pred (Stream B), with circularity note ------------
    print("\n" + "=" * 70)
    print("RUN 039252 -- group meas/pred (Stream B) [circularity-guarded]")
    print("=" * 70)
    print("group                 sumMeas    sumPred   cur    xFac   official")
    for gname, chans in GROUPS:
        ch = [c for c in chans if c not in AR_BLIND]     # exclude Ar-blind
        sm = r0["b_meas"][ch].sum()
        sp = r0["b_pred"][ch].sum()
        cur = sm / sp if sp > 0 else float("nan")
        fac = TYPE_FACTOR[chans[0]]
        print(f"{gname:19s} {sm:10.0f} {sp:10.0f}  {cur:5.2f}  x{fac:5.3f}  "
              f"{cur*fac:6.2f}")
    print("official = cur x type-factor = ratio vs UNSCALED official efficiency; "
          "cur~1 is by-construction (39252 in the fit), so read the top-vs-bottom "
          "asymmetry and liveness, not the absolute.")

    # --- cross-run liveness baseline ---------------------------------------
    print("\n" + "=" * 96)
    print("CROSS-RUN LIVENESS (frac flashes with pe>0 | peerRatio) -- is an anomaly "
          "run-specific?")
    print("=" * 96)
    prr = {r: peer_ratio(res[r]["med_evt_max"]) for r in RUNS}
    print("grp                 ch  " + "  ".join(f"{r}         " for r in RUNS))
    for gname, chans in GROUPS:
        if gname.startswith("cathode"):
            continue
        print("-" * 96)
        for k in chans:
            cells = []
            for r in RUNS:
                cells.append(f"{res[r]['frac_nonzero'][k]:4.2f}|{prr[r][k]:5.2f}")
            print(f"{gname:19s} {k:2d}  " + "   ".join(cells)
                  + (" [Ar-blind]" if k in AR_BLIND else ""))

    write_tsv(res, pr0)
    make_figs(res, pr0)
    print("\nwrote pdfunc_perchannel.tsv, pdfunc_liveness.png, "
          "pdfunc_wallxa_topbottom.png, pdfunc_crossrun.png")


def write_tsv(res, pr0):
    path = os.path.join(HERE, "pdfunc_perchannel.tsv")
    with open(path, "w") as fh:
        fh.write("run\tch\ttype\tgroup\tx\ty\tz\tn_flash\tfrac_nonzero\t"
                 "sum_pe\tmed_evt_max\tpeer_ratio\tb_meas\tb_pred\t"
                 "meas_pred_cur\ttype_factor\tmeas_pred_official\n")
        gof = {}
        for gname, chans in GROUPS:
            for c in chans:
                gof[c] = gname
        for r in RUNS:
            R = res[r]
            pr = peer_ratio(R["med_evt_max"])
            for k in range(40):
                bm, bp = R["b_meas"][k], R["b_pred"][k]
                ratio = bm / bp if bp > 0 else float("nan")
                fac = TYPE_FACTOR.get(k, float("nan"))
                fh.write(f"{r}\t{k}\t{int(R['optype'][k])}\t{gof.get(k,'')}\t"
                         f"{R['opx'][k]:.1f}\t{R['opy'][k]:.1f}\t{R['opz'][k]:.1f}\t"
                         f"{R['n_flash']}\t{R['frac_nonzero'][k]:.4f}\t"
                         f"{R['sum_pe'][k]:.1f}\t{R['med_evt_max'][k]:.2f}\t"
                         f"{pr[k]:.3f}\t{bm:.1f}\t{bp:.1f}\t{ratio:.4f}\t"
                         f"{fac:.3f}\t{ratio*fac:.4f}\n")


def make_figs(res, pr0):
    r0 = res["039252"]
    # Fig 1: per-channel median per-event-max PE, log, colored by group
    colors = {"cathode XA (ref)": "#888", "wall XA TOP (+x)": "#d62728",
              "wall XA BOTTOM (-x)": "#1f77b4", "z-wall PMT": "#2ca02c",
              "bottom PMT": "#9467bd"}
    fig, ax = plt.subplots(figsize=(13, 4.5))
    xs, hgt, cs, labs = [], [], [], []
    xi = 0
    ticks = []
    for gname, chans in GROUPS:
        for k in chans:
            xs.append(xi)
            hgt.append(max(r0["med_evt_max"][k], 1e-2))
            cs.append(colors[gname])
            ticks.append((xi, str(k)))
            xi += 1
        xi += 1
    ax.bar(xs, hgt, color=cs)
    ax.set_yscale("log")
    ax.set_ylabel("median per-event max PE (run 039252)")
    ax.set_xticks([t[0] for t in ticks])
    ax.set_xticklabels([t[1] for t in ticks], fontsize=7)
    ax.set_xlabel("OpDet channel  (grouped: cathode | wall-top | wall-bot | z-PMT | bottom-PMT)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=n) for n, c in colors.items()],
              fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.set_title("PDVD run 039252 per-channel brightness (dead channels 24/27/28/34 -> floor)")
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(HERE), "pics", "pdfunc_liveness.png"), dpi=110)
    plt.close(fig)

    # Fig 2: top-vs-bottom wall XA meas/pred (factor cancels)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for chans, col, lab in ((TOP_XA, "#d62728", "TOP (+x)"),
                            (BOT_XA, "#1f77b4", "BOTTOM (-x)")):
        rs = []
        for k in chans:
            if k in AR_BLIND:
                continue
            bp = r0["b_pred"][k]
            rs.append((k, r0["b_meas"][k] / bp if bp > 0 else np.nan))
        ax.scatter([f"ch{k}" for k, _ in rs], [v for _, v in rs],
                   color=col, s=90, label="wall XA " + lab, zorder=3)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("meas / pred (dump; x1.655 both sides -> cancels)")
    ax.set_title("Wall X-Arapuca top vs bottom -- clean discriminator")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(HERE), "pics", "pdfunc_wallxa_topbottom.png"), dpi=110)
    plt.close(fig)

    # Fig 3: cross-run liveness heatmap (frac nonzero) for candidate channels
    cand = TOP_XA + BOT_XA + ZWALL_PMT + BOTTOM_PMT
    M = np.array([[res[r]["frac_nonzero"][k] for r in RUNS] for k in cand])
    fig, ax = plt.subplots(figsize=(4.6, 11))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(RUNS)))
    ax.set_xticklabels(RUNS, rotation=30)
    ax.set_yticks(range(len(cand)))
    ax.set_yticklabels([f"ch{k}" + (" *" if k in AR_BLIND else "") for k in cand],
                       fontsize=7)
    ax.set_title("frac flashes with PE>0")
    for i in range(len(cand)):
        for j in range(len(RUNS)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="w" if M[i, j] < 0.6 else "k", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(HERE), "pics", "pdfunc_crossrun.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
