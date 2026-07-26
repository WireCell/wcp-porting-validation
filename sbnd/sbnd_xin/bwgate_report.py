#!/usr/bin/env python3
"""Doc 55 ON-path check: the beam-window gate must remove work, never change it.

The knob-OFF gate (p54_ab_report.py --base-tag p55opt --opt-tag p56off) proves
the pre-doc-56 path is byte-identical.  This script checks the ON path, which by
construction is NOT byte-identical (out-of-window bundles lose their verdicts):

  1. the bundle LIST per event is unchanged (the gate touches no clustering or
     Q/L product -- only which bundles the taggers look at);
  2. every IN-WINDOW bundle's verdicts (tgm/stm/fc/stmfit/label) are IDENTICAL
     to the ungated arm's -- that is what proves gating did not perturb the
     surviving computation;
  3. the post-PR tree's flag_TGM/STM/FC agree for the in-window clusters;
  4. tracking-stm.root: for every cluster the GATED arm fitted, the FIT trees
     (T_rec_charge trajectory + dQ/dx, T_stm_pass, T_stm_eval) are bitwise equal
     (ROOT files embed timestamps, so the comparison is numeric, per M2).
     T_proj_data is reported SEPARATELY, not as a failure: its per-cell
     charge_err/pred_charge come from TrackFitting's shared 2-D accumulator,
     which is last-writer-wins across clusters (assemble_fitted_charge_2d, and
     it iterates a POINTER-keyed map -- pre-existing).  Ungated, an out-of-window
     cosmic's fit could overwrite the beam bundle's cells (charge_err 8000,
     pred 0); gated, the bundle's own fit values survive.  That is the intended
     effect of running fewer fits, and it improves the beam bundle's 2-D panel;
  5. it reports what the gate removed (tags on out-of-window bundles).

Usage: python3 bwgate_report.py [--base-tag d55ton] [--gated-tag d56bw]
Full write-up: docs/56_beam-window-tagger-gate.md
"""
import argparse
import glob
import io
import json
import os
import sys
import tarfile

import numpy as np

ROOTS = ["mcp10", "mcp1000", "mcp1000b"]
HERE = os.path.dirname(os.path.abspath(__file__))

AP = argparse.ArgumentParser()
AP.add_argument("--base-tag", default="d55ton", help="ungated reference arm")
AP.add_argument("--gated-tag", default="d56bw", help="beam_window_only arm")
AP.add_argument("--window", default="0.2,2.2", help="low,high in us")
A = AP.parse_args()
LO, HI = (float(v) for v in A.window.split(","))


def read_tsv(path):
    """[{col: val}] from a whitespace-aligned nusel TSV."""
    lines = open(path).read().split("\n")
    hdr = lines[0].split()
    out = []
    for ln in lines[1:]:
        p = ln.split()
        if len(p) == len(hdr):
            out.append(dict(zip(hdr, p)))
    return out


def cluster_flags(tarpath, event):
    """{ident: (t0_us, tgm, stm, fc)} from a post-PR pctree; {} if absent."""
    if not os.path.isfile(tarpath):
        return {}
    base = f"pointtrees/{event}/live/pointclouds/namedpcs/cluster_scalar/arrays/"
    arrs = {}
    with tarfile.open(tarpath) as tf:
        for m in tf.getmembers():
            if not m.name.endswith("_metadata.json"):
                continue
            md = json.load(tf.extractfile(m))
            dp = md.get("datapath", "")
            if not dp.startswith(base):
                continue
            npy = m.name[: -len("_metadata.json")] + "_array.npy"
            try:
                arrs[dp[len(base):]] = np.load(io.BytesIO(tf.extractfile(npy).read()))
            except KeyError:
                pass
    if "ident" not in arrs:
        return {}
    n = len(arrs["ident"])

    def col(name):
        # An ABSENT flag array means every cluster is 0, not "unknown":
        # normalize_cluster_flags (MultiAlgBlobClustering.cxx) emits an array
        # for a flag name as soon as ANY cluster carries it and back-fills 0 for
        # the rest, so the name is missing exactly when no cluster was flagged.
        return arrs[name] if name in arrs else np.zeros(n, dtype=int)

    return {int(arrs["ident"][i]): (float(arrs["cluster_t0"][i]) / 1000.0,
                                    int(col("flag_TGM")[i]), int(col("flag_STM")[i]),
                                    int(col("flag_FC")[i]))
            for i in range(n)}


def canon(a):
    """Value-based bytes for an array, INCLUDING jagged (object-dtype) branches.

    np.ndarray.tobytes() on an object array serializes POINTERS, so two files
    holding identical jagged data would always compare unequal -- recurse into
    the elements instead.  Bitwise (never approximate): NaNs compare equal to
    NaNs, which is what "reproduced the fit" has to mean.
    """
    a = np.asarray(a)
    if a.dtype == object:
        return b"||".join(canon(x) for x in a)
    return f"{a.dtype}|{a.shape}|".encode() + a.tobytes()


def stm_fit_by_cluster(path):
    """{cluster_id: {branch: bytes}} of the STM fit tree; None if unavailable."""
    try:
        import uproot
    except ImportError:
        return None
    if not os.path.isfile(path):
        return None
    out = {}
    with uproot.open(path) as f:
        for key in f.keys():
            obj = f[key]
            if not hasattr(obj, "keys") or "cluster_id" not in obj.keys():
                continue
            arrs = obj.arrays(library="np")
            # T_proj_data holds ONE entry of jagged per-cell arrays (object
            # dtype); the flat trees hold one row per point.  Flatten the
            # single-entry jagged case so both compare per cluster_id.
            if np.asarray(arrs["cluster_id"], dtype=object).dtype == object \
               and obj.num_entries == 1:
                arrs = {k: np.asarray(v[0]) for k, v in arrs.items()}
            cid = np.asarray(arrs["cluster_id"]).astype(int)
            for c in np.unique(cid):
                sel = cid == c
                blk = out.setdefault(int(c), {})
                for bn, av in arrs.items():
                    av = np.asarray(av)
                    if av.shape[:1] != cid.shape[:1]:
                        continue
                    blk[f"{key}/{bn}"] = canon(av[sel])
    return out


def stm_pass_clusters(path):
    """cluster_ids that reached an STM pass in this file (T_stm_pass), or set()."""
    try:
        import uproot
    except ImportError:
        return set()
    if not os.path.isfile(path):
        return set()
    with uproot.open(path) as f:
        for key in f.keys():
            if not key.startswith("T_stm_pass"):
                continue
            a = f[key]["cluster_id"].array(library="np")
            return {int(v) for v in np.asarray(a).ravel()}
    return set()


VERDICT_COLS = ["tgm", "stm", "fc", "stmfit", "label"]
n_ev = n_ok = 0
fails = []
inwin_checked = 0
removed = {"tgm": 0, "stm": 0, "fc": 0}
kept = {"tgm": 0, "stm": 0, "fc": 0}
root_pairs = root_same = 0
proj_pairs = [0, 0]   # [compared, differing] T_proj_data blocks of fitted clusters
proj_events = set()

for r in ROOTS:
    for bdir in sorted(glob.glob(f"{HERE}/work-{r}-{A.base_tag}/nusel_evt*")):
        ev = os.path.basename(bdir).replace("nusel_evt", "")
        gdir = f"{HERE}/work-{r}-{A.gated_tag}/nusel_evt{ev}"
        btsv, gtsv = f"{bdir}/nusel-evt{ev}.tsv", f"{gdir}/nusel-evt{ev}.tsv"
        if not (os.path.isfile(btsv) and os.path.isfile(gtsv)):
            fails.append(f"evt{ev}: missing TSV in one arm")
            continue
        n_ev += 1
        b = {row["main_id"]: row for row in read_tsv(btsv)}
        g = {row["main_id"]: row for row in read_tsv(gtsv)}
        ev_fail = []
        if set(b) != set(g):
            ev_fail.append(f"bundle list differs: base-only {sorted(set(b)-set(g))} "
                           f"gated-only {sorted(set(g)-set(b))}")
        for mid in sorted(set(b) & set(g)):
            t0 = float(b[mid]["flash_time_us"])
            inwin = LO <= t0 < HI and mid != "-1"
            if inwin:
                inwin_checked += 1
                diff = [c for c in VERDICT_COLS if b[mid][c] != g[mid][c]]
                if diff:
                    ev_fail.append(f"in-window main {mid}: " + ", ".join(
                        f"{c} {b[mid][c]}->{g[mid][c]}" for c in diff))
                for k in kept:
                    kept[k] += int(b[mid][k] == "1")
            else:
                for k in removed:
                    removed[k] += int(b[mid][k] == "1")
                if any(g[mid][c] not in ("-1", "-") for c in ("tgm", "stm", "fc")):
                    ev_fail.append(f"out-of-window main {mid} still carries a verdict "
                                   f"({g[mid]['tgm']}/{g[mid]['stm']}/{g[mid]['fc']})")

        # post-PR tree flags for the in-window clusters
        bf = cluster_flags(f"{bdir}/pctree-pr-evt{ev}.tar.gz", int(ev))
        gf = cluster_flags(f"{gdir}/pctree-pr-evt{ev}.tar.gz", int(ev))
        if bf and gf:
            for ident, (t0, tgm, stm, fc) in bf.items():
                if not (LO <= t0 < HI):
                    continue
                if ident not in gf:
                    ev_fail.append(f"in-window cluster {ident} absent from gated tree")
                elif gf[ident][1:] != (tgm, stm, fc):
                    ev_fail.append(f"in-window cluster {ident} flags "
                                   f"{(tgm, stm, fc)} -> {gf[ident][1:]}")

        # tracking-stm.root: surviving clusters' fit blocks must be bitwise equal
        # Restrict to the clusters the GATED arm actually fitted.  tracking-stm
        # also carries T_proj_data 2-D charge for clusters with no fit at all
        # (block id = cluster_id*10 + pass), and for an out-of-window cluster the
        # ungated arm's block holds its fit's charge_pred while the gated one
        # cannot -- a difference that IS the intended effect, not a regression.
        bfit = stm_fit_by_cluster(f"{bdir}/tracking-stm.root")
        gfit = stm_fit_by_cluster(f"{gdir}/tracking-stm.root")
        fitted = stm_pass_clusters(f"{gdir}/tracking-stm.root")
        allowed = set(fitted) | {10 * c + p for c in fitted for p in range(10)}
        if bfit is not None and gfit is not None:
            for cid in sorted((set(bfit) & set(gfit)) & allowed):
                fitb = {k: v for k, v in bfit[cid].items() if not k.startswith("T_proj_data")}
                fitg = {k: v for k, v in gfit[cid].items() if not k.startswith("T_proj_data")}
                if fitb:
                    root_pairs += 1
                    if fitb == fitg:
                        root_same += 1
                    else:
                        ev_fail.append(f"tracking-stm.root cluster {cid} FIT block differs")
                projb = {k: v for k, v in bfit[cid].items() if k.startswith("T_proj_data")}
                projg = {k: v for k, v in gfit[cid].items() if k.startswith("T_proj_data")}
                if projb:
                    proj_pairs[0] += 1
                    if projb != projg:
                        proj_pairs[1] += 1
                        proj_events.add(ev)

        if ev_fail:
            fails.append(f"evt{ev}: " + "; ".join(ev_fail))
        else:
            n_ok += 1

print(f"events compared           : {n_ev}  ({n_ok} clean)")
print(f"in-window bundles checked : {inwin_checked}")
print(f"tracking-stm.root FIT blocks (T_rec_charge/T_stm_pass/T_stm_eval) of "
      f"gated-arm fitted clusters: {root_pairs} compared, {root_same} bitwise equal")
print(f"tracking-stm.root T_proj_data blocks of those clusters: {proj_pairs[0]} compared, "
      f"{proj_pairs[1]} differing in {len(proj_events)} event(s) "
      f"(expected: shared-cell last-writer-wins, see the module docstring)")
print(f"tags KEPT     (in-window) : tgm {kept['tgm']}  stm {kept['stm']}  fc {kept['fc']}")
print(f"tags REMOVED  (out-of-win): tgm {removed['tgm']}  stm {removed['stm']}  fc {removed['fc']}")
if fails:
    print(f"\nFAILURES ({len(fails)}):")
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("\nON-PATH CHECK: PASS -- every in-window verdict reproduced, "
      "no out-of-window verdict left behind.")
