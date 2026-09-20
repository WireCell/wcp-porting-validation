#!/usr/bin/env python3
"""doc sbnd_xin/116: the pr/150 trajectory-closure instrument on the doc-115 / doc-116 MC arms.

scripts/analysis/pr150/pr150_traj_eval.py is reused as a MODULE (its per-event body one_event(),
FIELDS, sign_p are imported, not copied), because it assumes the flat per-sample layout
work-<sample>-<arm>/pr_evt<ID>/ with unique event numbers and writes under docs/pr/150_figs/traj/.
The MC arms are split per reco1 file (work-r3<s>-d116<cell>/f*/pr_evt<ID>/) with event numbers
that repeat across files, so here an event is keyed by (run, subrun, event) read from the
per-event nusel table, and the output goes to docs/116_figs/traj/<label>.tsv.

The metrics are the pr/150 ones (R2D_<p>, R2Dpm1, med_d, noslice, dead, P1..P4, uncov, qneg,
wig1, len_cm, rows), computed on the main-cluster bundle of each event that has a calib dump, a
tracking-pr.root and the Bee zip; the compare step prints the same row-weighted A/B table with the
paired better/worse count and the exact sign test.

Usage:
  scripts/d116/traj_eval.py extract --arm-dir work-r3cv-d115pr --label cv-baseline [--jobs 16]
  scripts/d116/traj_eval.py compare --a cv-baseline --b cv-tfull
"""
import argparse
import csv
import glob
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(SX, "scripts", "analysis", "pr150"))
import pr150_traj_eval as P  # noqa: E402

OUT = os.path.join(SX, "docs", "116_figs", "traj")


def rse_of(d):
    for p in glob.glob(os.path.join(d, "nusel-evt*.tsv")):
        with open(p) as fh:               # whitespace-aligned columns, not tabs
            fh.readline()
            r = fh.readline().split()
        if len(r) >= 3:
            return r[0], r[1], r[2]
    return None


def one(args):
    d, ev = args
    row = P.one_event((d, ev))
    rse = rse_of(d)
    row["run"], row["subrun"], row["event"] = (rse if rse else ("", "", ev))
    row["dir"] = os.path.relpath(d, SX)
    return row


def extract(a):
    os.makedirs(OUT, exist_ok=True)
    dirs = sorted(glob.glob(os.path.join(a.arm_dir, "f*", "pr_evt*")) + glob.glob(os.path.join(a.arm_dir, "pr_evt*")))
    jobs = [(d, int(os.path.basename(d)[6:])) for d in dirs]
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(one, jobs, chunksize=8))
    p = os.path.join(OUT, a.label + ".tsv")
    cols = ["run", "subrun", "event", "dir"] + [k for k in P.FIELDS if k != "event"]
    with open(p, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(("%.5g" % r[k] if isinstance(r.get(k), float) else str(r.get(k, ""))) for k in cols) + "\n")
    print(f"{a.label}: {len(rows)} events, {sum(r['has'] for r in rows)} with a main-cluster fit -> {p}")


def load(label):
    R = {}
    for r in csv.DictReader(open(os.path.join(OUT, label + ".tsv")), delimiter="\t"):
        if r["has"] == "1":
            R[(r["run"], r["subrun"], r["event"])] = {k: (float(v) if v not in ("", "nan") else float("nan"))
                                                     for k, v in r.items() if k not in ("run", "subrun", "event", "dir")}
    return R


def compare(a):
    A, B = load(a.a), load(a.b)
    keys = sorted(set(A) & set(B))
    print(f"# d116 traj compare A={a.a} B={a.b}  A={len(A)} B={len(B)} common={len(keys)}")
    print("| metric | A | B | delta (row-weighted) | paired better / worse (|d|>0.005) | sign p |")
    print("|---|---|---|---|---|---|")
    for met, better in (("R2D_W", "lower"), ("R2D_U", "lower"), ("R2D_V", "lower"), ("R2Dpm1_W", "lower"), ("P1", "lower"),
                        ("P1_2", "lower"), ("P2", "lower"), ("P3", "lower"), ("P4", "higher"), ("uncov", "lower"),
                        ("qneg", "lower"), ("wig1", "lower"), ("med_d_W", "lower"), ("noslice_W", "lower"),
                        ("len_cm", None), ("rows", None)):
        va = np.array([A[k].get(met, np.nan) for k in keys]); vb = np.array([B[k].get(met, np.nan) for k in keys])
        wa = np.array([A[k]["rows"] for k in keys]); wb = np.array([B[k]["rows"] for k in keys])
        ok = np.isfinite(va) & np.isfinite(vb)
        if met in ("len_cm", "rows"):
            ma, mb = float(np.nansum(va)), float(np.nansum(vb))
            print(f"| {met} (sum) | {ma:.1f} | {mb:.1f} | {mb - ma:+.1f} | | |")
            continue
        if not ok.any():
            continue
        ma = float(np.sum(va[ok] * wa[ok]) / np.sum(wa[ok])); mb = float(np.sum(vb[ok] * wb[ok]) / np.sum(wb[ok]))
        d = vb[ok] - va[ok]
        if better == "lower":
            bet, wor = int((d < -0.005).sum()), int((d > 0.005).sum())
        else:
            bet, wor = int((d > 0.005).sum()), int((d < -0.005).sum())
        print(f"| {met} | {ma:.4f} | {mb:.4f} | {mb - ma:+.4f} | {bet} / {wor} | {P.sign_p(min(bet, wor), bet + wor):.3g} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--arm-dir", required=True); e.add_argument("--label", required=True)
    e.add_argument("--jobs", type=int, default=16)
    c = sub.add_parser("compare"); c.add_argument("--a", required=True); c.add_argument("--b", required=True)
    a = ap.parse_args()
    extract(a) if a.cmd == "extract" else compare(a)
