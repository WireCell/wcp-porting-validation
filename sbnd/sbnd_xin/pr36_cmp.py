#!/usr/bin/env python3
"""Gate driver for the doc pr/36 tagger-stage round.

The gate is TWO artifacts and neither alone suffices (doc pr/36 sec 10.9):

  1. T_tagger in pr_evt<ID>/tracking-pr.root -- branch-by-branch CONTENTS
     compare (uproot; jagged-aware).  Covers the F3-F6 fields (shw_sp_*,
     brm_*, tro_2_v_*, cme_*, mip_quality_*) and both BDT scores.  All other
     trees in the file are compared the same way (T_kine is the coupling
     channel if clus_geom_helper ever lands).  tagger_tree_ab.py's exit code
     is NOT a gate (rc=0 even when branches move), so the compare is done
     in-driver.
  2. the "tagger" block of calib-pr-evt<ID>.json (PR_EXTRA_STAGES=pr_display)
     -- the ONLY outlet for F1's match_isFC (0 hits in the T_tagger booking
     block).  dump_tagger emits 45 keys; a gate on it alone would PASS
     vacuously on F3-F6.

Plus the never-move channels: mabc-pr.zip members and the pctree rollup
(this stage must not touch Bee or the point-cloud tree), and the nusel TSVs
(score channel; the stmfit column is log-parsed and log-tearing flaky BOTH
directions -- pr/35 sec 11.4 -- so an stmfit-only TSV diff is noise, never
physics).

Usage: pr36_cmp.py <base_arm_dir> <test_arm_dir>

Exit 1 (ESCALATE) on: mabc member movement, pctree movement, or calib
movement outside the tagger/kine blocks.  Tree/TSV movement is reported with
branch names for per-arm attribution but does not set rc -- whether it is
expected depends on which knob the test arm forces.
"""

import sys, os, json, zipfile, tarfile, hashlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import uproot


def sha(data):
    return hashlib.sha256(data).hexdigest()


def zip_members(path):
    out = {}
    with zipfile.ZipFile(path) as z:
        for n in sorted(z.namelist()):
            out[n] = sha(n.encode() + z.read(n))
    return out


def tar_rollup(path):
    h = hashlib.sha256()
    with tarfile.open(path, "r:gz") as t:
        for m in sorted(t.getmembers(), key=lambda m: m.name):
            h.update(m.name.encode())
            f = t.extractfile(m)
            if f:
                h.update(f.read())
    return h.hexdigest()


def _to_py(v):
    """Canonicalize a branch value: ndarray / uproot STLVector / nested
    containers -> plain Python lists+scalars.  Handles doubly-jagged
    (vector<vector<...>>) branches that defeat np.array_equal."""
    if isinstance(v, np.ndarray):
        if v.dtype == object:
            return [_to_py(x) for x in v]
        return v.tolist()
    if isinstance(v, (str, bytes)):
        return v
    if hasattr(v, "tolist"):
        return _to_py(v.tolist())
    if isinstance(v, (list, tuple)) or hasattr(v, "__iter__"):
        return [_to_py(x) for x in v]
    return v


def branch_equal(a, b):
    # Compare via JSON text so NaN == NaN (identity gate, not arithmetic)
    # and -0.0 vs 0.0 still differ.
    return json.dumps(_to_py(a)) == json.dumps(_to_py(b))


def trees_diff(fa_path, fb_path):
    """{tree: {'schema': (only_a, only_b), 'moved': [branch,...]}} for every
    tree in either file; empty dict = identical."""
    out = {}
    with uproot.open(fa_path) as fa, uproot.open(fb_path) as fb:
        ta = {k.split(";")[0] for k in fa.keys(cycle=False)}
        tb = {k.split(";")[0] for k in fb.keys(cycle=False)}
        for tn in sorted(ta | tb):
            if tn not in ta or tn not in tb:
                out[tn] = {"schema": (sorted(ta - tb), sorted(tb - ta)), "moved": []}
                continue
            A, B = fa[tn], fb[tn]
            ka, kb = set(A.keys()), set(B.keys())
            schema = (sorted(ka - kb), sorted(kb - ka))
            aa = A.arrays(sorted(ka & kb), library="np")
            bb = B.arrays(sorted(ka & kb), library="np")
            moved = [k for k in sorted(ka & kb) if not branch_equal(aa[k], bb[k])]
            if schema != ([], []) or moved:
                out[tn] = {"schema": schema, "moved": moved}
    return out


def calib_diff(a_path, b_path):
    """(tagger_subkeys_moved, kine_keys_moved, other_keys_moved)."""
    with open(a_path, "rb") as f:
        a = json.load(f)
    with open(b_path, "rb") as f:
        b = json.load(f)
    keys = sorted(set(a) | set(b))
    tagger, kine, other = [], [], []
    for k in keys:
        if a.get(k) == b.get(k):
            continue
        if k == "tagger":
            ta, tb = a.get(k) or {}, b.get(k) or {}
            tagger = sorted(kk for kk in set(ta) | set(tb) if ta.get(kk) != tb.get(kk))
        elif k.startswith("kine"):
            kine.append(k)
        else:
            other.append(k)
    return tagger, kine, other


def one_event(args):
    base, test, evt = args
    r = {"evt": evt, "calib": None, "tagger_moved": [], "kine_moved": [],
         "other_moved": [], "trees": {}, "mc_diff": [], "pctree": None, "tsv": None}
    bd, td = f"{base}/pr_evt{evt}", f"{test}/pr_evt{evt}"

    cb, ct = f"{bd}/calib-pr-evt{evt}.json", f"{td}/calib-pr-evt{evt}.json"
    eb, et = os.path.exists(cb), os.path.exists(ct)
    if eb and et:
        with open(cb, "rb") as f:
            hb = sha(f.read())
        with open(ct, "rb") as f:
            ht = sha(f.read())
        if hb == ht:
            r["calib"] = "identical"
        else:
            r["calib"] = "DIFF"
            r["tagger_moved"], r["kine_moved"], r["other_moved"] = calib_diff(cb, ct)
    elif not eb and not et:
        r["calib"] = "absent both"
    else:
        r["calib"] = "absent " + ("base" if not eb else "test")

    r["trees"] = trees_diff(f"{bd}/tracking-pr.root", f"{td}/tracking-pr.root")

    mb = zip_members(f"{bd}/mabc-pr.zip")
    mt = zip_members(f"{td}/mabc-pr.zip")
    r["mc_diff"] = sorted(k for k in set(mb) | set(mt) if mb.get(k) != mt.get(k))

    tb = tar_rollup(f"{bd}/pctree-pr-evt{evt}.tar.gz")
    tt = tar_rollup(f"{td}/pctree-pr-evt{evt}.tar.gz")
    r["pctree"] = "identical" if tb == tt else "DIFF"

    with open(f"{bd}/nusel-evt{evt}.tsv", "rb") as f:
        nb = sha(f.read())
    with open(f"{td}/nusel-evt{evt}.tsv", "rb") as f:
        nt = sha(f.read())
    r["tsv"] = "identical" if nb == nt else "DIFF"
    return r


def main():
    base, test = sys.argv[1], sys.argv[2]
    evts = sorted(d[len("pr_evt"):] for d in os.listdir(base) if d.startswith("pr_evt"))
    evts_t = sorted(d[len("pr_evt"):] for d in os.listdir(test) if d.startswith("pr_evt"))
    if evts != evts_t:
        print(f"EVENT SET MISMATCH: base {len(evts)} vs test {len(evts_t)}")
        sys.exit(2)

    with ProcessPoolExecutor(max_workers=24) as ex:
        results = list(ex.map(one_event, [(base, test, e) for e in evts]))

    n = len(results)
    calib_id  = sum(r["calib"] == "identical" for r in results)
    calib_ab  = sum(r["calib"].startswith("absent") for r in results)
    trees_id  = sum(not r["trees"] for r in results)
    mabc_id   = sum(not r["mc_diff"] for r in results)
    pctree_id = sum(r["pctree"] == "identical" for r in results)
    tsv_id    = sum(r["tsv"] == "identical" for r in results)

    print(f"pr36_cmp: {base}  vs  {test}   ({n} events)")
    print(f"  tracking trees: {trees_id}/{n} identical (every tree, every branch)")
    print(f"  calib-pr JSON : {calib_id}/{n - calib_ab} identical"
          + (f"  ({calib_ab} absent-side skips)" if calib_ab else ""))
    print(f"  mabc members  : {mabc_id}/{n} identical")
    print(f"  pctree rollup : {pctree_id}/{n} identical")
    print(f"  nusel TSV     : {tsv_id}/{n} identical")

    rc = 0
    for r in results:
        if r["calib"] == "DIFF":
            line = f"  calib DIFF evt {r['evt']}:"
            if r["tagger_moved"]:
                line += f" tagger{r['tagger_moved']}"
            if r["kine_moved"]:
                line += f" kine{r['kine_moved']}"
            if r["other_moved"]:
                line += f"  *** OTHER keys {r['other_moved']} -- ESCALATE, outside this stage's writers"
                rc = 1
            print(line)
        for tn, d in sorted(r["trees"].items()):
            oa, ob = d["schema"]
            extra = ""
            if oa or ob:
                extra = f" schema(base-only={oa}, test-only={ob})"
            print(f"  tree DIFF evt {r['evt']} {tn}: {len(d['moved'])} branch(es) {d['moved'][:12]}{extra}")
        if r["mc_diff"]:
            print(f"  mabc DIFF evt {r['evt']}: {r['mc_diff']} -- ESCALATE, this stage must not move Bee")
            rc = 1
        if r["pctree"] == "DIFF":
            print(f"  pctree DIFF evt {r['evt']} -- ESCALATE, this stage must not move the pctree")
            rc = 1
        if r["tsv"] == "DIFF":
            print(f"  nusel TSV DIFF evt {r['evt']} (score channel; stmfit-only diffs are log-tearing noise)")
    sys.exit(rc)


if __name__ == "__main__":
    main()
