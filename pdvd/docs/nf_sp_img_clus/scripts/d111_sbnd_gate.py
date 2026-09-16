#!/usr/bin/env python3
"""doc pdvd/111 G3 -- the SBND PR knob-OFF gate: the same config on two pins, per event, every output compared.

Fork by duplication of the comparator in d101_gates.sh (untouched), section 4 only, with the tags as arguments:
  old = work-<s>-<OLD> run on libpin_d102 (clus 091e142b9481, before this round's C++)
  new = work-<s>-<NEW> run on libpin_d111b (this round's build: TRAJASSOC trace + seed_recenter_* at defaults)
Compared per event dir pr_evt*: every TTree of every *.root (awkward, NaN-safe), every member of every *.zip
(sha256), every *.tar.gz via abtest/hash_archive.py --members, every *.json / *.tsv byte for byte.  Symlinked
inputs are skipped.  Zero compared events is reported as a dead net, never as a pass.

Usage: d111_sbnd_gate.py --old d111sold --new d111snew; exit 0 only on BYTE-IDENTICAL
"""
import argparse, glob, hashlib, os, subprocess, sys, zipfile
import awkward as ak
import uproot

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
SB = IMG + "/sbnd/sbnd_xin"
HA = IMG + "/abtest/hash_archive.py"


def real_outputs(d):
    out = {}
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        if os.path.islink(p) or not os.path.isfile(p):
            continue
        if f.endswith((".root", ".zip", ".tar.gz", ".json", ".tsv")) and not f.startswith("."):
            out[f] = p
    return out


def trees(fn):
    f = uproot.open(fn)
    res = {}
    for k in sorted(set(x.split(";")[0] for x in f.keys())):
        o = f[k]
        if not hasattr(o, "arrays"):
            continue
        a = o.arrays(library="ak")
        try:
            res[k] = ak.to_list(ak.nan_to_none(a))
        except Exception:
            res[k] = ak.to_list(a)
    return res


def zipm(fn):
    with zipfile.ZipFile(fn) as z:
        return {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if not n.endswith("/")}


def tarm(fn):
    r = subprocess.run(["python3", HA, "--members", fn], capture_output=True, text=True)
    return r.stdout.replace(fn, "<f>")


def same_file(a, b):
    if a.endswith(".root"):
        ta, tb = trees(a), trees(b)
        if set(ta) != set(tb):
            return False, "tree set %s vs %s" % (sorted(ta), sorted(tb))
        bad = [t for t in ta if ta[t] != tb[t]]
        return (not bad), ("trees differ: %s" % bad if bad else "")
    if a.endswith(".zip"):
        za, zb = zipm(a), zipm(b)
        bad = sorted(n for n in set(za) | set(zb) if za.get(n) != zb.get(n))
        return (not bad), ("members differ: %s" % bad[:5] if bad else "")
    if a.endswith(".tar.gz"):
        return tarm(a) == tarm(b), "hash_archive members differ"
    return open(a, "rb").read() == open(b, "rb").read(), "bytes differ"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    a = ap.parse_args()
    pairs = []
    for old in sorted(glob.glob(f"{SB}/work-*-{a.old}")):
        new = old[: -len(a.old)] + a.new
        for e in sorted(glob.glob(old + "/pr_evt*")):
            pairs.append((new + "/" + os.path.basename(e), e))
    n_ev = n_files = 0
    fails = []
    for new, old in pairs:
        if not (os.path.isdir(new) and os.path.isdir(old)):
            fails.append((os.path.basename(new), "missing dir")); continue
        A, B = real_outputs(new), real_outputs(old)
        if not B:
            fails.append((os.path.basename(new), "baseline has no outputs")); continue
        n_ev += 1
        for f in sorted(set(A) | set(B)):
            if f not in A or f not in B:
                fails.append((os.path.basename(new), "file only on one side: " + f)); continue
            ok, why = same_file(A[f], B[f]); n_files += 1
            if not ok:
                fails.append((os.path.basename(new), f + ": " + why))
    print(f"=== SBND PR ({a.new} vs {a.old}): {n_ev} event dirs, {n_files} files compared")
    for e, w in fails[:20]:
        print(f"    DIFF {e}  {w}")
    if n_ev == 0:
        print("    VERDICT: *** DEAD NET -- nothing compared, NOT a pass ***"); sys.exit(2)
    if fails:
        print(f"    VERDICT: *** NOT IDENTICAL on {len(fails)} item(s) ***"); sys.exit(1)
    print("    VERDICT: BYTE-IDENTICAL")


if __name__ == "__main__":
    main()
