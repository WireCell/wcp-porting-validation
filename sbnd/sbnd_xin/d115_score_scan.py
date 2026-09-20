#!/usr/bin/env python3
"""doc sbnd_xin/115: efficiency, purity and background rate as a function of the BDT score cut.

Why this exists beside d107_selection.py's fixed working points.  The numu and nue scores come
from the uBooNE-trained BDT weights (doc 107 sec 6 item 4; clus.jsonnet points at
uboone/weights/*.xml), so `numu_score > 0.9` and `nue_score > 7.0` are conventions carried over
from MicroBooNE, not an SBND-calibrated operating point.  A baseline quoted at one uncalibrated
cut ages badly.  The scan costs nothing once candidates.tsv exists and lets a later round move
the operating point without re-running the chain.

nue_score == -15 is the nue BDT's "did not evaluate" sentinel (UbooneNueBDTScorer.cxx: the score
is only written when the BDT ran).  It is NOT a low score: sweeping a cut down through it makes
the low end of the curve meaningless.  Those rows are excluded from the nue scan and reported as
a fill fraction instead, exactly as doc 107 sec 5.2 tracks `nue == -15` in its own column.

Definitions are d107_selection.py's, unchanged:
  FV 5<|x|<190, |y|<190, 10<z<450 cm; match = reco vertex within 5 cm of the nearest true vertex;
  selection = score cut AND reco vertex in FV; signal = true CC of that flavour with its true
  vertex in the FV (optionally Edep > E); efficiency and purity per interaction / per candidate.

For a sample with no truth (beam-off) the scan reports the per-gate SELECTED RATE instead, which
is the background side of the same curve.

Usage:
  python3 d115_score_scan.py products/d115/cv    docs/115_scan --label cv    [--edep-min 100]
  python3 d115_score_scan.py products/d115/off   docs/115_scan --label off --gates 1000
"""
import argparse
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NUE_SENTINEL = -15.0
MATCH_CM = 5.0
# The same validated palette d107_selection.py uses, so the two rounds' figures sit together.
COL = {"numu": "#2a78d6", "nue": "#eb6834", "bkg": "#1baf7a"}


def in_fv(x, y, z):
    return 5.0 < abs(x) < 190.0 and abs(y) < 190.0 and 10.0 < z < 450.0


def wilson(k, n, z=1.0):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def tclass(flav, ccnc):
    if ccnc == "NC":
        return "NC"
    return {"numu": "numuCC", "nue": "nueCC"}.get(flav, "CC-" + flav)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("figdir")
    ap.add_argument("--label", required=True)
    ap.add_argument("--edep-min", type=float, default=0.0)
    ap.add_argument("--gates", type=int, default=0,
                    help="beam gates in a truth-less sample (the rate denominator)")
    a = ap.parse_args()

    C = list(csv.DictReader(open(os.path.join(a.src, "candidates.tsv")), delimiter="\t"))
    tpath = os.path.join(a.src, "truth.tsv")
    T = list(csv.DictReader(open(tpath), delimiter="\t")) if os.path.exists(tpath) else []
    f = float

    for r in C:
        r["_fv"] = r["vertex_default"] == "0" and in_fv(f(r["nu_x"]), f(r["nu_y"]), f(r["nu_z"]))
        r["_matched"] = r["vertex_default"] == "0" and 0 <= f(r["t_dist_cm"]) < MATCH_CM
        r["_key"] = (r["run"], r["subrun"], r["event"])
    for t in T:
        t["_fv"] = in_fv(f(t["vx"]), f(t["vy"]), f(t["vz"]))
        t["_cls"] = tclass(t["flav"], t["ccnc"])
        # d107_selection.py:90 -- no threshold means no Edep cut, NOT "Edep > 0".
        t["_edep_ok"] = a.edep_min <= 0 or f(t["Edep"]) > a.edep_min
        t["_key"] = (t["run"], t["subrun"], t["event"])

    os.makedirs(a.figdir, exist_ok=True)
    lines = ["# doc 115 score scan -- sample %s%s" % (a.label,
             ", signal also needs Edep > %g MeV" % a.edep_min if a.edep_min else "")]
    lines.append("# candidates %d, truth interactions %d" % (len(C), len(T)))

    nfill = sum(1 for r in C if f(r["nue_score"]) != NUE_SENTINEL)
    lines.append("# nue BDT evaluated (nue_score != %g): %d/%d = %.1f %%"
                 % (NUE_SENTINEL, nfill, len(C), 100.0 * nfill / len(C) if C else float("nan")))
    lines.append("")

    grids = {"numu": [x / 10.0 for x in range(-30, 71, 1)],
             "nue": [x / 10.0 for x in range(-100, 151, 1)]}
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, flav in zip(axes, ("numu", "nue")):
        col = flav + "_score"
        sig_cls = flav + "CC"
        # signal denominator, per doc 107: true CC of this flavour with its vertex in the FV
        sig = [t for t in T if t["_fv"] and t["_cls"] == sig_cls and t["_edep_ok"]]
        sig_keys = {}
        for t in sig:
            sig_keys.setdefault(t["_key"], []).append(t["idx"])
        eff_x, eff_y, pur_x, pur_y, rate_x, rate_y = [], [], [], [], [], []
        for cut in grids[flav]:
            selected = []
            for r in C:
                v = f(r[col])
                if flav == "nue" and v == NUE_SENTINEL:
                    continue
                if r["_fv"] and v > cut:
                    selected.append(r)
            if T:
                # efficiency: signal interactions with >= 1 MATCHED selected candidate
                hit = set()
                nsig_sel = 0
                for r in selected:
                    if not r["_matched"]:
                        continue
                    k = r["_key"]
                    if k in sig_keys and r["t_idx"] in sig_keys[k] and r["t_ccnc"] == "CC" \
                       and r["t_flav"] == flav:
                        hit.add((k, r["t_idx"])); nsig_sel += 1
                ne, nd = len(hit), len(sig)
                np_, nn = nsig_sel, len(selected)
                if nd:
                    eff_x.append(cut); eff_y.append(ne / nd)
                if nn:
                    pur_x.append(cut); pur_y.append(np_ / nn)
                rows.append([a.label, flav, "%.1f" % cut, ne, nd, np_, nn,
                             "%.4f" % (ne / nd) if nd else "", "%.4f" % (np_ / nn) if nn else ""])
            else:
                # --gates is required for a truth-less sample: defaulting to the candidate
                # events would count only the gates that produced a candidate and drive the
                # rate toward 100 %.
                if not a.gates:
                    raise SystemExit("ERROR: --gates is required for a sample with no truth.tsv")
                ngate = a.gates
                k = len({r["_key"] for r in selected})
                rate_x.append(cut); rate_y.append(k / ngate)
                rows.append([a.label, flav, "%.1f" % cut, "", "", k, ngate, "",
                             "%.5f" % (k / ngate)])
        if T:
            ax.plot(eff_x, [100 * v for v in eff_y], color=COL[flav], lw=2, label="efficiency")
            ax.plot(pur_x, [100 * v for v in pur_y], color=COL["bkg"], lw=2, ls="--",
                    label="purity")
            ax.set_ylabel("per cent")
        else:
            ax.plot(rate_x, [100 * v for v in rate_y], color=COL[flav], lw=2,
                    label="selected gates")
            ax.set_ylabel("per cent of beam gates")
        for wp in ({"numu": [0.9], "nue": [4.0, 7.0]})[flav]:
            ax.axvline(wp, color="#888", lw=1, ls=":")
            ax.annotate("%g" % wp, (wp, ax.get_ylim()[1]), fontsize=8, color="#666",
                        ha="center", va="top")
        ax.set_xlabel("%s cut" % col)
        ax.set_title("%s -- %s" % (a.label, flav))
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.figdir, "d115_scan_%s.png" % a.label), dpi=140)

    with open(os.path.join(a.figdir, "d115_scan_%s.tsv" % a.label), "w") as fh:
        fh.write("sample\tflavour\tcut\tn_eff_num\tn_eff_den\tn_pur_num\tn_pur_den\teff\tpur_or_rate\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")

    # A short readable digest at the conventional points plus the best-purity-at-90%-eff style
    # summary a later round would want.
    for flav in ("numu", "nue"):
        sub = [r for r in rows if r[1] == flav]
        for wp in ({"numu": [0.9], "nue": [4.0, 7.0]})[flav]:
            m = [r for r in sub if abs(float(r[2]) - wp) < 1e-6]
            if not m:
                continue
            # Which arm of the scan a row came from is decided by whether the sample HAS truth,
            # never by an empty denominator -- an MC sample with zero signal of one flavour must
            # still read as "no signal", not silently as a beam-gate rate.
            if T and int(m[0][4] or 0) == 0:
                r = m[0]
                lines.append("%s > %-4g  no true %sCC in the FV: efficiency undefined; "
                             "selected candidates %s" % (flav + "_score", wp, flav, r[6]))
                continue
            if T:
                r = m[0]
                lo, hi = wilson(int(r[3]), int(r[4]))
                lo2, hi2 = wilson(int(r[5]), int(r[6]))
                lines.append("%s > %-4g  efficiency %s/%s = %.1f %% [%.1f, %.1f]   "
                             "purity %s/%s = %.1f %% [%.1f, %.1f]"
                             % (flav + "_score", wp, r[3], r[4],
                                100 * int(r[3]) / int(r[4]), 100 * lo, 100 * hi,
                                r[5], r[6], 100 * int(r[5]) / int(r[6]) if int(r[6]) else float("nan"),
                                100 * lo2, 100 * hi2))
            else:
                r = m[0]
                lines.append("%s > %-4g  selected gates %s/%s = %.2f %%"
                             % (flav + "_score", wp, r[5], r[6],
                                100 * int(r[5]) / int(r[6]) if int(r[6]) else float("nan")))
    txt = "\n".join(lines) + "\n"
    open(os.path.join(a.figdir, "d115_scan_%s.txt" % a.label), "w").write(txt)
    print(txt)
    print("-> %s/d115_scan_%s.{txt,tsv,png}" % (a.figdir, a.label))


if __name__ == "__main__":
    main()
