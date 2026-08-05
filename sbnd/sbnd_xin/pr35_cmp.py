#!/usr/bin/env python3
"""Gate driver for the doc pr/35 energy-reconstruction round.

Why not pr32_cmp.py / pr34_cmp.py: pctree-pr does NOT carry KineInfo (doc
pr/35 sec 10.6), so both earlier drivers gate NOTHING for this stage.  The
artifact under test is pr_evt<ID>/calib-pr-evt<ID>.json (PrDisplayDump,
plain JSON -- directly diffable, no archive timestamps), written only when
the arm ran with PR_EXTRA_STAGES=pr_display.  kine_reco_Enu is numu-BDT
var 69 and in the nue reader, so the nusel TSVs are the score channel and
part of the gate.

Per event, base vs test:
  1. calib-pr-evt<ID>.json  -- byte compare; on diff, parse and say whether
     the moved keys are kine_* or elsewhere.  Absent on ONE side only =>
     counted separately ("calib absent"), not a hash mismatch (the
     work-pr31r2-prod48 baseline was run without pr_display).
  2. all mabc-pr.zip member hashes (sha256 over name+payload).
  3. pctree-pr-evt<ID>.tar.gz rollup hash (ProcessPool -- gzip+sha256 holds
     the GIL, a thread pool runs serially).
  4. nusel-evt<ID>.tsv.

Usage: pr35_cmp.py <base_arm_dir> <test_arm_dir>
"""

import sys, os, json, zipfile, tarfile, hashlib, io
from concurrent.futures import ProcessPoolExecutor


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


def json_key_diff(a_path, b_path):
    """Which top-level keys differ between the two calib JSONs."""
    with open(a_path, "rb") as f:
        a = json.load(f)
    with open(b_path, "rb") as f:
        b = json.load(f)
    keys = sorted(set(a) | set(b))
    moved = [k for k in keys if a.get(k) != b.get(k)]
    return moved


def one_event(args):
    base, test, evt = args
    r = {"evt": evt, "calib": None, "calib_moved": [], "mc_diff": [],
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
            r["calib_moved"] = json_key_diff(cb, ct)
    elif not eb and not et:
        r["calib"] = "absent both"
    else:
        r["calib"] = "absent " + ("base" if not eb else "test")

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
    calib_df  = [r for r in results if r["calib"] == "DIFF"]
    mabc_id   = sum(not r["mc_diff"] for r in results)
    pctree_id = sum(r["pctree"] == "identical" for r in results)
    tsv_id    = sum(r["tsv"] == "identical" for r in results)

    print(f"pr35_cmp: {base}  vs  {test}   ({n} events)")
    print(f"  calib-pr JSON : {calib_id}/{n - calib_ab} identical"
          + (f"  ({calib_ab} absent-side skips)" if calib_ab else ""))
    print(f"  mabc members  : {mabc_id}/{n} identical")
    print(f"  pctree rollup : {pctree_id}/{n} identical")
    print(f"  nusel TSV     : {tsv_id}/{n} identical")

    rc = 0
    for r in calib_df:
        kine = [k for k in r["calib_moved"] if k.startswith("kine")]
        other = [k for k in r["calib_moved"] if not k.startswith("kine")]
        line = f"  calib DIFF evt {r['evt']}: kine keys {kine}"
        if other:
            line += f"  *** NON-KINE keys {other} -- ESCALATE, outside this stage's writer"
            rc = 1
        print(line)
    for r in results:
        if r["mc_diff"]:
            print(f"  mabc DIFF evt {r['evt']}: {r['mc_diff']} -- ESCALATE, this stage must not move Bee")
            rc = 1
        if r["pctree"] == "DIFF":
            print(f"  pctree DIFF evt {r['evt']} -- ESCALATE, this stage must not move the pctree")
            rc = 1
        if r["tsv"] == "DIFF":
            print(f"  nusel TSV DIFF evt {r['evt']} (score channel -- expected ONLY when a kine move feeds the BDTs)")
    sys.exit(rc)


if __name__ == "__main__":
    main()
