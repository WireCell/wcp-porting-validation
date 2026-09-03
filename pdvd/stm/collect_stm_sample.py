#!/usr/bin/env python3
"""Collect the PDVD stopping-muon dQ/dx-vs-residual-range sample (doc pdvd/25 M4/M7).

Port by duplication of sbnd_xin/dqdx_rr_sample/collect_dqdx_rr_sample.py (doc 55)
to PDVD: sweeps every `T_rec_charge` (cluster, pass) block in
work/<RUN6>_<idx>_<tag>/tracking-stm.root (PdvdMagnifyTrackingVisitor, written
by run_pr_evt.sh -stm-fit), keeps only the blocks that are a TAGGED stopping
muon with a clean Bragg profile, and writes the sample as plain text.

Selection, in order (every rejected block is printed with --verbose):

  0. TAGGER verdict: the block's cluster must be STM-tagged and NOT TGM-tagged
     in wct_pr_<RUN6>_<idx>.log ("TaggerCheckSTM: cluster N -> STM=1 TGM=0";
     cluster id = block // 10) and the pass must be the ACCEPTED one
     (T_stm_pass status == 0).  This is the "get rid of TGM etc." step: TGM,
     rejected-pass and untagged blocks never enter.
  1. >= 40 fitted points.
  2. >= 6 populated profile bins (>= 3 points each) reaching rr < 2 cm at the
     stopping end and rr >= 22 cm at the far end.
  3. Bragg contrast = median dQ/dx(rr < 2) / median dQ/dx(20 <= rr < 40) >= 2.0.
  4. median reduced_chi2 <= 2.5.
  5. free-scale shape residual <= 10 % against at least one of the four
     hadron/muon PDVD curves (electron excluded: flat above 15 cm).
  Particle assignment is by the free scale k against the PDVD muon / proton
  tables (0.44 kV/cm, stm/pdvd_ref_dqdx.json); a track needing k_muon ~ 1.9 is
  proton-like.

PDVD geometry: anode (shield plane) |x| = 339.91 cm, drift speed 0.148073 cm/us
(the production Q/L value, doc 25 sec 7a); drift_cm = 339.91 - |x|.

Usage:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 stm/collect_stm_sample.py --tag stm1            # writes stm/sample_index.tsv, stm/sample_points.tsv
  python3 stm/collect_stm_sample.py --tag stm1 --verbose --plot stm/pics/sample_overlay.png
  python3 stm/collect_stm_sample.py --tag stm1 --no-tagger   # ignore the tagger verdict (diagnostic)
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
PDVD = os.path.dirname(HERE)
REF_JSON = os.path.join(HERE, "pdvd_ref_dqdx.json")
HYPS = {"muon": "MuonDeDx", "pion": "PionDeDx", "kaon": "KaonDeDx", "proton": "ProtonDeDx"}
BINS = [(0, 2), (2, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40), (40, 60)]

ANCHOR = "end"           # "end" (default, and the physically right one) = rr from the fit path end.
                         # "peak" re-anchors at the Bragg peak eval_stm_core_impl scores against and
                         # drops the res_length tail.  DIAGNOSTIC ONLY: doc 25 sec 13.6 shows that
                         # tail is collinear muon continuation (median 3.7 deg, 0.88 MIP), so the peak
                         # is a mid-track bump, not a stop -- do not quote field numbers from it.
MIN_NPTS = 40
MIN_BINS = 6
MAX_RRMIN = 2.0
MIN_RRMAX = 22.0
MIN_CONTRAST = 2.0
MAX_CHI2 = 2.5
MAX_SHAPE_RMS = 0.10
MUON_K = (0.85, 1.25)
PROTON_K = (0.85, 1.35)

ANODE_ABS_X = 339.91      # cm, shield plane (protodunevd clus.jsonnet dvm FV_x = +-3399.1 mm)
MAX_ABS_X = 305.0         # cm; points closer than ~35 cm to a CRP plane carry an instrumental dQ/dx rise (doc 25 M3) and are excluded
DRIFT_SPEED = 0.148073    # cm/us

VERDICT_RE = re.compile(r"TaggerCheck(STM|TGM|FC): cluster (\d+) .*?(STM=(\d)\s+TGM=(\d)|TGM=(true|false)|FC=(true|false|\d))")


def load_refs(ref_json=REF_JSON, suffix=""):
    d = json.load(open(ref_json))
    out = {}
    for h, key in HYPS.items():
        t = d[key + suffix]
        x = t["start"] + t["step"] * np.arange(len(t["values"]))
        out[h] = (x, np.asarray(t["values"], float))
    return out


def read_verdicts(log_path):
    """cluster id -> dict(stm=0/1, tgm=0/1, fc=0/1) from the PR log (spdlog may
    split a line; anchor on the tagger token, not the line start)."""
    out = {}
    if not os.path.exists(log_path):
        return out
    for line in open(log_path, errors="replace"):
        if "TaggerCheck" not in line or "cluster" not in line:
            continue
        m = VERDICT_RE.search(line)
        if not m:
            continue
        cid = int(m.group(2)); rec = out.setdefault(cid, {})
        if m.group(1) == "STM":
            rec["stm"] = int(m.group(4)); rec["tgm"] = int(m.group(5))
        elif m.group(1) == "TGM":
            rec.setdefault("tgm", 0)
            rec["tgm"] = 1 if m.group(6) == "true" else rec["tgm"]
        else:
            rec["fc"] = 1 if m.group(7) in ("true", "1") else 0
    return out


def iter_blocks(tag, events=None):
    pat = os.path.join(PDVD, "work", f"*_{tag}", "tracking-stm.root") if tag else os.path.join(PDVD, "work", "*", "tracking-stm.root")
    for fp in sorted(glob.glob(pat)):
        wd = os.path.dirname(fp)
        m = re.match(r"^(\d{6})_(\d+)", os.path.basename(wd))
        if not m:
            continue
        run, idx = m.group(1), m.group(2)
        ev = f"{run}_{idx}"
        if events is not None and ev not in events:
            continue
        f = uproot.open(fp)
        if "T_rec_charge" not in f:
            continue
        t = f["T_rec_charge"].arrays(["x", "y", "z", "q", "nq", "rr", "ndf", "status", "reduced_chi2", "pass"], library="np")
        if len(t["ndf"]) == 0:
            continue
        res_of = {}
        if ANCHOR == "peak":
            if "T_stm_eval" not in f:
                continue
            ee = f["T_stm_eval"].arrays(library="np")
            # the eval chain short-circuits (flag_pass = A || B), so the LAST
            # verdict==1 record of a (cluster, pass) is the accepting call.
            for i in range(len(ee["verdict"])):
                if ee["verdict"][i] == 1:
                    res_of[(int(ee["cluster_id"][i]), int(ee["pass"][i]))] = float(ee["res_length"][i])
        tr = f["Trun"].arrays(["dQdx_scale", "dQdx_offset", "eventNo"], library="np")
        verdicts = read_verdicts(os.path.join(wd, f"wct_pr_{run}_{idx}.log"))
        for b in sorted(set(t["ndf"].tolist())):
            mk = t["ndf"] == b
            dQ = (t["q"][mk] - tr["dQdx_offset"][0]) / tr["dQdx_scale"][0]
            dx = t["nq"][mk]
            dx = np.where(np.abs(t["x"][mk]) > MAX_ABS_X, 0.0, dx)   # near-anode points: dqdx -> 0 (excluded from every profile)
            cid = int(b) // 10
            rr = t["rr"][mk]
            dqdx = np.where(dx > 0, dQ / np.maximum(dx, 1e-9), 0.0)
            x, y, z = t["x"][mk], t["y"][mk], t["z"][mk]
            chi2v = t["reduced_chi2"][mk]
            if ANCHOR == "peak":
                res = res_of.get((cid, int(b) % 10))
                if res is None:
                    continue        # no accepting eval record => no tagger stop to anchor on
                rr = rr - res
                keep = rr >= 0      # drop the residual tail the tagger set aside
                if keep.sum() < 2:
                    continue
                rr, dx, dqdx = rr[keep], dx[keep], dqdx[keep]
                x, y, z, chi2v = x[keep], y[keep], z[keep], chi2v[keep]
            yield dict(workdir=wd, event=ev, evtno=int(tr["eventNo"][0]), block=int(b), cluster=cid,
                       rr=rr, dx=dx, dqdx=dqdx,
                       x=x, y=y, z=z,
                       status=int(t["status"][mk][0]), npts=int(len(rr)),
                       chi2=float(np.median(chi2v)),
                       verdict=verdicts.get(cid, {}))


def profile(rr, dq):
    cen, med, cnt = [], [], []
    for lo, hi in BINS:
        s = (rr >= lo) & (rr < hi) & (dq > 0)
        if s.sum() >= 3:
            cen.append(float(np.median(rr[s]))); med.append(float(np.median(dq[s]))); cnt.append(int(s.sum()))
    return np.array(cen), np.array(med), np.array(cnt)


def shape_scores(cen, med, refs):
    out = {}
    for n, (rx, rv) in refs.items():
        s = (cen >= rx.min()) & (cen <= rx.max())
        if s.sum() < MIN_BINS:
            continue
        r = med[s] / np.interp(cen[s], rx, rv)
        k = float(np.exp(np.mean(np.log(r))))
        out[n] = (k, float(np.sqrt(np.mean((r / k - 1) ** 2))))
    return out


def evaluate(bk, refs, use_tagger=True):
    d = dict(npts=bk["npts"], chi2=bk["chi2"], L=float(bk["rr"].max()) if bk["npts"] else 0.0)
    v = bk["verdict"]
    if use_tagger:
        if v.get("stm") != 1:
            return False, None, dict(d, cut="not STM-tagged")
        if v.get("tgm") == 1:
            return False, None, dict(d, cut="TGM-tagged")
        if bk["status"] != 0:
            return False, None, dict(d, cut="pass not accepted (status %d)" % bk["status"])
    if bk["npts"] < MIN_NPTS:
        return False, None, dict(d, cut="npts")
    cen, med, cnt = profile(bk["rr"], bk["dqdx"])
    d.update(nbin=len(cen))
    if len(cen) < MIN_BINS:
        return False, None, dict(d, cut="nbin")
    d.update(rrmin=float(cen.min()), rrmax=float(cen.max()))
    if cen.min() > MAX_RRMIN or cen.max() < MIN_RRMAX:
        return False, None, dict(d, cut="rr coverage")
    near = (bk["rr"] < 2) & (bk["dqdx"] > 0)
    mid = (bk["rr"] >= 20) & (bk["rr"] < 40) & (bk["dqdx"] > 0)
    if near.sum() < 3 or mid.sum() < 5:
        return False, None, dict(d, cut="contrast stats")
    d["contrast"] = float(np.median(bk["dqdx"][near]) / np.median(bk["dqdx"][mid]))
    if d["contrast"] < MIN_CONTRAST:
        return False, None, dict(d, cut="contrast")
    if bk["chi2"] > MAX_CHI2:
        return False, None, dict(d, cut="chi2")
    sc = shape_scores(cen, med, refs)
    if not sc:
        return False, None, dict(d, cut="no hypothesis in domain")
    d["scores"] = sc
    best = min(sc, key=lambda n: sc[n][1])
    d["best_shape"] = best; d["shape_rms"] = sc[best][1]
    if sc[best][1] > MAX_SHAPE_RMS:
        return False, None, dict(d, cut="shape rms")
    part = None
    if "muon" in sc and MUON_K[0] <= sc["muon"][0] <= MUON_K[1]:
        part = "muon"
    elif "proton" in sc and PROTON_K[0] <= sc["proton"][0] <= PROTON_K[1]:
        part = "proton"
    if part is None:
        return False, None, dict(d, cut="scale in no window")
    d["particle"] = part; d["k"] = sc[part][0]
    return True, part, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="stm1", help="work-dir tag to sweep (work/<RUN6>_<idx>_<tag>)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--plot")
    ap.add_argument("--events", help="comma list or file of <RUN6>_<idx>")
    ap.add_argument("--no-tagger", action="store_true", help="ignore the STM/TGM verdict and pass status (diagnostic)")
    ap.add_argument("--ref-suffix", default="", help="'' = 0.44 kV/cm tables (config); '_E050' = the 0.50 comparison set")
    ap.add_argument("--anchor", choices=["end", "peak"], default="end",
                    help="'end' = residual range from the fit path end (default, and the physically right "
                         "anchor); 'peak' = re-anchor at the peak eval_stm_core_impl scores against -- "
                         "DIAGNOSTIC ONLY, see doc 25 sec 13.6: that tail is muon continuation, so the "
                         "peak is a mid-track bump and its field numbers must not be quoted")
    ap.add_argument("--min-contrast", type=float, help="override MIN_CONTRAST (sensitivity tiers, doc 25 sec 13.6)")
    ap.add_argument("--max-chi2", type=float, help="override MAX_CHI2")
    ap.add_argument("--max-shape-rms", type=float, help="override MAX_SHAPE_RMS")
    args = ap.parse_args()
    global ANCHOR
    ANCHOR = args.anchor
    global MIN_CONTRAST, MAX_CHI2, MAX_SHAPE_RMS
    if args.min_contrast is not None: MIN_CONTRAST = args.min_contrast
    if args.max_chi2 is not None: MAX_CHI2 = args.max_chi2
    if args.max_shape_rms is not None: MAX_SHAPE_RMS = args.max_shape_rms
    events = None
    if args.events:
        events = set(open(args.events).read().split()) if os.path.exists(args.events) else set(args.events.split(","))
    refs = load_refs(suffix=args.ref_suffix)
    kept, seen, cuts = [], 0, {}
    for bk in iter_blocks(args.tag, events):
        seen += 1
        ok, part, d = evaluate(bk, refs, use_tagger=not args.no_tagger)
        if not ok:
            cuts[d["cut"]] = cuts.get(d["cut"], 0) + 1
        if args.verbose:
            print(f"{'KEEP' if ok else 'drop':>4s} {bk['event']} blk{bk['block']:<5d} st{bk['status']} "
                  f"v={bk['verdict']} " + (f"{part:>6s} k={d['k']:.2f} rms={d['shape_rms']*100:.1f}% contrast={d['contrast']:.2f}" if ok
                                           else f"cut={d.get('cut')} " + " ".join(f"{k}={v:.3g}" for k, v in d.items() if k not in ("cut", "scores") and isinstance(v, float))))
        if ok:
            kept.append((bk, d))
    kept.sort(key=lambda kb: (kb[1]["particle"], kb[0]["event"], kb[0]["block"]))
    idx = os.path.join(args.outdir, f"sample_index{args.suffix}.tsv")
    pts = os.path.join(args.outdir, f"sample_points{args.suffix}.tsv")
    with open(idx, "w") as fi, open(pts, "w") as fp:
        fi.write(f"# PDVD stopping-track dQ/dx-vs-rr sample, tag {args.tag}, tagger verdict {'IGNORED' if args.no_tagger else 'required (STM=1, TGM=0, accepted pass)'}\n")
        fi.write(f"# points with |x| > {MAX_ABS_X} cm excluded (instrumental near-CRP dQ/dx rise)\n")
        fi.write(f"# cuts: npts>={MIN_NPTS} nbin>={MIN_BINS} rrmin<={MAX_RRMIN} rrmax>={MIN_RRMAX} contrast>={MIN_CONTRAST} chi2<={MAX_CHI2} shape_rms<={MAX_SHAPE_RMS}; refs {REF_JSON}{args.ref_suffix or ' (0.44 kV/cm)'}\n")
        fi.write("particle\tevent\tevtno\tblock\tcluster\tstatus\tnpts\tL_cm\tcontrast\tchi2\tk_muon\trms_muon\tk_proton\trms_proton\tmean_absx_cm\tfc\n")
        fp.write("# per-point sample; dqdx in e/cm, rr/x/y/z/dx in cm\n")
        fp.write(f"# drift_cm = {ANODE_ABS_X} - |x| (shield plane); drift_us = drift_cm / {DRIFT_SPEED}\n")
        fp.write("particle\tevent\tblock\trr\tdqdx\tdx\tx\ty\tz\tdrift_cm\tdrift_us\n")
        for bk, d in kept:
            sc = d["scores"]; kp = sc.get("proton", (float("nan"), float("nan")))
            fi.write(f"{d['particle']}\t{bk['event']}\t{bk['evtno']}\t{bk['block']}\t{bk['cluster']}\t{bk['status']}\t{bk['npts']}\t{d['L']:.1f}\t{d['contrast']:.2f}\t"
                     f"{bk['chi2']:.2f}\t{sc['muon'][0]:.3f}\t{sc['muon'][1]*100:.1f}\t{kp[0]:.3f}\t{kp[1]*100:.1f}\t{np.mean(np.abs(bk['x'])):.1f}\t{bk['verdict'].get('fc', -1)}\n")
            for i in np.where(bk["dqdx"] > 0)[0]:
                dr = ANODE_ABS_X - abs(float(bk["x"][i]))
                fp.write(f"{d['particle']}\t{bk['event']}\t{bk['block']}\t{bk['rr'][i]:.3f}\t{bk['dqdx'][i]:.1f}\t{bk['dx'][i]:.4f}\t"
                         f"{bk['x'][i]:.2f}\t{bk['y'][i]:.2f}\t{bk['z'][i]:.2f}\t{dr:.2f}\t{dr / DRIFT_SPEED:.1f}\n")
    n_by = {}
    for _, d in kept:
        n_by[d["particle"]] = n_by.get(d["particle"], 0) + 1
    print(f"scanned {seen} blocks; kept {len(kept)}: " + ", ".join(f"{p} {n}" for p, n in sorted(n_by.items())))
    print("rejections: " + ", ".join(f"{k} {v}" for k, v in sorted(cuts.items(), key=lambda kv: -kv[1])))
    print(f"wrote {idx}\nwrote {pts}")
    if args.plot:
        plot_sample(kept, refs, args.plot, args.tag)
        print(f"wrote {args.plot}")


def plot_sample(kept, refs, out, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"muon": "#2a78d6", "proton": "#e34948"}
    n_of = {p: sum(1 for _, dd in kept if dd["particle"] == p) for p in ("muon", "proton")}
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    ax = axes[0]
    for n, ls, lab in (("muon", "-", "muon expectation (PDVD 0.44 kV/cm)"), ("proton", "--", "proton expectation (PDVD 0.44 kV/cm)")):
        rx, rv = refs[n]; ax.plot(rx, rv / 1e3, ls, color="#0b0b0b" if n == "muon" else "#52514e", lw=1.8, zorder=6, label=lab)
    ax.axhline(55, ls=":", color="#a3a29b", lw=1.4, zorder=1, label="flat MIP 55 ke/cm (mip_dqdx)")
    first = dict.fromkeys(col, True)
    for bk, d in kept:
        cen, med, _ = profile(bk["rr"], bk["dqdx"]); p = d["particle"]
        ax.plot(cen, med / 1e3, "o-", color=col[p], ms=4.5, lw=1.3, alpha=0.8, mec="white", mew=0.8, zorder=4, label=(f"{p} tracks ({n_of[p]})" if first[p] else None)); first[p] = False
    ax.set_xlim(0, 60); ax.set_ylim(0, 290); ax.set_xlabel("residual range from the stopping end  [cm]"); ax.set_ylabel("dQ/dx  [e/cm x 1e3]")
    ax.set_title("binned medians over the reference-table domain", fontsize=10); ax.legend(fontsize=8, loc="upper right", framealpha=0.95); ax.grid(alpha=0.2, lw=0.6)
    ax = axes[1]; ax.axhline(1.0, ls=":", color="#a3a29b", lw=1.4); first = dict.fromkeys(col, True)
    for bk, d in kept:
        cen, med, _ = profile(bk["rr"], bk["dqdx"]); p = d["particle"]; rx, rv = refs[p]; s = (cen >= rx.min()) & (cen <= rx.max())
        ax.plot(cen[s], med[s] / np.interp(cen[s], rx, rv), "o-", color=col[p], ms=4.5, lw=1.3, alpha=0.8, mec="white", mew=0.8, label=(f"{p} tracks ({n_of[p]})" if first[p] else None)); first[p] = False
    ax.set_xlim(0, 60); ax.set_ylim(0.6, 1.7); ax.set_xlabel("residual range from the stopping end  [cm]"); ax.set_ylabel("data / its own PDVD expectation")
    ax.set_title("ratio to the assigned particle's own curve\n(no free scale removed -- raw agreement)", fontsize=10); ax.legend(fontsize=8, loc="upper right", framealpha=0.95); ax.grid(alpha=0.2, lw=0.6)
    fig.suptitle(f"PDVD STM-tagged stopping tracks ({tag}): binned medians of {len(kept)} tracks (uncalibrated data)", fontsize=10, color="#52514e")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out, dpi=140)


if __name__ == "__main__":
    main()
