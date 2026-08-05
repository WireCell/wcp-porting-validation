#!/usr/bin/env python3
"""Gate driver for the doc pr/33 EM-shower-clustering round.

Adapted from pr36_cmp.py (same canonicalization -- see the GOTCHA there: an
ad-hoc np.array_equal over uproot branches produced ~30 phantom T_tagger
moves in pr/37; _to_py + JSON-text compare is the calibrated instrument).

The pr/33 stage (shower_clustering_with_nv inside tagger_check_neutrino) is
upstream of every shower quantity in T_kine, the nue tagger's pi0 block
(NeutrinoTaggerNuE.cxx map_pio_id_showers), and the Bee PR-stage layers:
mc.json (pi0 grouping keyed on pio_id + pi0_ke, MultiAlgBlobClustering.cxx
~:1552) AND shower_track-global.json (point color = PR shower membership,
~:826-866); track_fit/vertices are PR-graph dumps too.  So knob-ON movement
may legitimately hit: any tracking-pr.root tree, the calib tagger/kine
blocks, the four PR-derived mabc members, and the nusel TSVs.

Never-move channels (rc=1 ESCALATE):
  - the pctree rollup (upstream point-cloud tree);
  - mabc members that are imaging-derived: *clustering-global.json and
    *channel-deadarea*;
  - calib movement outside the tagger/kine blocks.
An ssmsp_* branch move in T_tagger is the F3 scoping tripwire -- reported
with a loud marker (not rc, attribution is per-arm) -- see doc pr/33 §10.10
amendment 1.

Tree/TSV/PR-layer movement is reported with names for per-arm attribution
but does not set rc -- whether it is expected depends on which knob the test
arm forces.  The TSV stmfit column is log-parsed and log-tearing flaky both
directions (pr/35 §11.4): an stmfit-only TSV diff is noise, never physics.

Usage: pr33_cmp.py <base_arm_dir> <test_arm_dir>
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


# calib top-level keys the pr/33 stage may legitimately move: the tagger
# block, the kine block, and the five PR-display blocks that render the
# shower clustering itself.  'steiner' and 'proj' are upstream of this
# stage (Steiner graph / projection data) and must never move -> escalate.
PR_DISPLAY_KEYS = {"main_vertex", "segments", "showers", "track_shower", "vertices"}


def calib_diff(a_path, b_path):
    """(tagger_subkeys_moved, kine_keys_moved, display_keys_moved, other_keys_moved)."""
    with open(a_path, "rb") as f:
        a = json.load(f)
    with open(b_path, "rb") as f:
        b = json.load(f)
    keys = sorted(set(a) | set(b))
    tagger, kine, display, other = [], [], [], []
    for k in keys:
        if a.get(k) == b.get(k):
            continue
        if k == "tagger":
            ta, tb = a.get(k) or {}, b.get(k) or {}
            tagger = sorted(kk for kk in set(ta) | set(tb) if ta.get(kk) != tb.get(kk))
        elif k.startswith("kine"):
            kine.append(k)
        elif k in PR_DISPLAY_KEYS:
            display.append(k)
        else:
            other.append(k)
    return tagger, kine, display, other


def mabc_never_move(member):
    """Imaging-derived members this stage must not touch."""
    return ("clustering-global" in member) or ("channel-deadarea" in member)


def one_event(args):
    base, test, evt = args
    r = {"evt": evt, "calib": None, "tagger_moved": [], "kine_moved": [],
         "display_moved": [], "other_moved": [], "trees": {}, "mc_diff": [],
         "pctree": None, "tsv": None}
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
            r["tagger_moved"], r["kine_moved"], r["display_moved"], r["other_moved"] = calib_diff(cb, ct)
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

    print(f"pr33_cmp: {base}  vs  {test}   ({n} events)")
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
            if r["display_moved"]:
                line += f" display{r['display_moved']}"
            if r["other_moved"]:
                line += f"  *** OTHER keys {r['other_moved']} (steiner/proj/...) -- ESCALATE, outside this stage's writers"
                rc = 1
            print(line)
        for tn, d in sorted(r["trees"].items()):
            oa, ob = d["schema"]
            extra = ""
            if oa or ob:
                extra = f" schema(base-only={oa}, test-only={ob})"
            ssmsp = [k for k in d["moved"] if k.startswith("ssmsp_")]
            if ssmsp:
                extra += f"  *** ssmsp moved {ssmsp[:6]} -- F3 SCOPING TRIPWIRE (doc pr/33 §10.10)"
            print(f"  tree DIFF evt {r['evt']} {tn}: {len(d['moved'])} branch(es) {d['moved'][:12]}{extra}")
        if r["mc_diff"]:
            bad = [m for m in r["mc_diff"] if mabc_never_move(m)]
            if bad:
                print(f"  mabc DIFF evt {r['evt']}: {bad} -- ESCALATE, imaging-derived member moved")
                rc = 1
            ok = [m for m in r["mc_diff"] if not mabc_never_move(m)]
            if ok:
                print(f"  mabc DIFF evt {r['evt']} (PR-stage layers): {ok}")
        if r["pctree"] == "DIFF":
            print(f"  pctree DIFF evt {r['evt']} -- ESCALATE, this stage must not move the pctree")
            rc = 1
        if r["tsv"] == "DIFF":
            print(f"  nusel TSV DIFF evt {r['evt']} (score channel; stmfit-only diffs are log-tearing noise)")
    sys.exit(rc)


if __name__ == "__main__":
    main()
