#!/usr/bin/env python3
"""doc 109: output gate between two arms of scripts/d109_arms.sh.

Per event (pr_evt<ID> present in both arms):
  archives   mabc-pr.zip and pctree-pr-evt<ID>.tar.gz by MEMBER CONTENT
             (abtest/hash_archive.py; raw archive bytes embed timestamps, M2)
  records    nusel-evt<ID>.tsv and calib-pr-evt<ID>.json by sha256
  ROOT       tracking-pr.root, every tree / branch / entry, NaN-aware:
             - a tree or branch missing from B            -> FAIL
             - a tree or branch only in B                 -> listed as ADDED
             - a shared branch whose values differ        -> FAIL, unless
               named with --allow TREE.BRANCH (then listed as ALLOWED)
             - --prefix TREE.BRANCH: a jagged branch where B may APPEND
               elements; A's elements must be B's leading elements

Usage:
  scripts/d109_gate.py <labelA> <labelB> [--samples nuecc48 ncpi0 mcp1k]
                       [--allow T_cluster.tgm ...] [--prefix T_tagger.act_cluster_id ...]
                       [--events <file of ids>] [--jobs 24]
Exit 0 = every event PASSes (additions and allowed changes are reported, not failed).
"""
import argparse
import collections
import glob
import hashlib
import os
import sys
from multiprocessing import Pool

import numpy as np
import uproot

SX = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(SX, "..", "..", "abtest"))
import hash_archive  # noqa: E402


def arch_hash(path):
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    for name, payload in hash_archive.members(path):
        h.update(name.encode())
        h.update(payload)
    return h.hexdigest()


def file_hash(path):
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def eq(x, y):
    a = np.asarray(x, dtype=object) if isinstance(x, list) else np.asarray(x)
    b = np.asarray(y, dtype=object) if isinstance(y, list) else np.asarray(y)
    if a.dtype == object or b.dtype == object:
        if len(a) != len(b):
            return False
        return all(eq(p, q) for p, q in zip(a, b))
    if a.shape != b.shape:
        return False
    if a.dtype.kind == "f" and b.dtype.kind == "f":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def prefix_eq(x, y):
    if len(x) != len(y):
        return False
    for p, q in zip(x, y):
        p = list(p)
        q = list(q)
        if len(q) < len(p) or not eq(p, q[:len(p)]):
            return False
    return True


def trees(f):
    return {k.split(";")[0] for k, c in f.classnames().items() if c == "TTree"}


def read_branch(t, b):
    try:
        return t[b].array(library="np")
    except Exception:
        return t[b].array(library="ak").tolist()


def root_cmp(pa, pb, allow, prefix):
    out = collections.Counter()
    notes = []
    if not os.path.isfile(pa) or not os.path.isfile(pb):
        out["root_missing"] += int(os.path.isfile(pa) != os.path.isfile(pb))
        return out, notes
    fa, fb = uproot.open(pa), uproot.open(pb)
    ta, tb = trees(fa), trees(fb)
    out["root files compared"] += 1
    for t in sorted(ta - tb):
        out["FAIL tree_removed"] += 1
        notes.append(f"tree removed {t}")
    for t in sorted(tb - ta):
        out[f"ADDED tree {t}"] += 1
    for t in sorted(ta & tb):
        A, B = fa[t], fb[t]
        ba, bb = set(A.keys()), set(B.keys())
        for b in sorted(ba - bb):
            out["FAIL branch_removed"] += 1
            notes.append(f"branch removed {t}.{b}")
        for b in sorted(bb - ba):
            out[f"ADDED branch {t}.{b}"] += 1
        if A.num_entries != B.num_entries:
            out["FAIL entries"] += 1
            notes.append(f"{t}: entries {A.num_entries} -> {B.num_entries}")
            continue
        out[f"root tree compared {t}"] += 1
        for b in sorted(ba & bb):
            key = f"{t}.{b}"
            out["root branches compared"] += 1
            va, vb = read_branch(A, b), read_branch(B, b)
            if key in prefix:
                same = eq(va, vb)
                if not same and prefix_eq(va, vb):
                    out[f"APPENDED {key}"] += 1
                elif not same:
                    out[f"FAIL prefix {key}"] += 1
                    notes.append(f"prefix differs {key}")
                continue
            if eq(va, vb):
                continue
            if key in allow:
                out[f"ALLOWED {key}"] += 1
            else:
                out["FAIL value"] += 1
                notes.append(f"value differs {key}")
    return out, notes


def one(job):
    ra, rb, evt, allow, prefix = job
    da, db = f"{ra}/pr_evt{evt}", f"{rb}/pr_evt{evt}"
    res = collections.Counter()
    notes = []
    for rel in ("mabc-pr.zip", f"pctree-pr-evt{evt}.tar.gz"):
        ha, hb = arch_hash(f"{da}/{rel}"), arch_hash(f"{db}/{rel}")
        res[f"{rel.split('-evt')[0]} {'same' if ha == hb else 'FAIL differ'}"] += 1
        if ha is None:
            res[f"{rel.split('-evt')[0]} absent"] += 1
        if ha != hb:
            notes.append(f"archive differs {rel}")
    for rel in (f"nusel-evt{evt}.tsv", f"calib-pr-evt{evt}.json"):
        ha, hb = file_hash(f"{da}/{rel}"), file_hash(f"{db}/{rel}")
        res[f"{rel.split('-evt')[0]} {'same' if ha == hb else 'FAIL differ'}"] += 1
        if ha is None:
            res[f"{rel.split('-evt')[0]} absent"] += 1
        if ha != hb:
            notes.append(f"record differs {rel}")
    rc, rn = root_cmp(f"{da}/tracking-pr.root", f"{db}/tracking-pr.root", allow, prefix)
    res.update(rc)
    notes += rn
    return evt, res, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labelA")
    ap.add_argument("labelB")
    ap.add_argument("--samples", nargs="+", default=["nuecc48", "ncpi0", "mcp1k"])
    ap.add_argument("--allow", nargs="*", default=[])
    ap.add_argument("--prefix", nargs="*", default=[])
    ap.add_argument("--jobs", type=int, default=24)
    # sbnd_xin/docs/109 rev 4: restrict the comparison to the events listed in
    # a file (one id per line), so an arm that ran MORE events than the other
    # (e.g. the byte-gate manifest plus a knob's eligible events) can still be
    # gated on the common manifest instead of failing on "event sets differ".
    ap.add_argument("--events", default=None, help="file of event ids to compare (applies to every sample)")
    a = ap.parse_args()
    only = None
    if a.events:
        only = {ln.strip() for ln in open(a.events) if ln.strip()}
    total = collections.Counter()
    failed = []
    for s in a.samples:
        ra, rb = f"{SX}/work-{s}-{a.labelA}", f"{SX}/work-{s}-{a.labelB}"
        ea = {os.path.basename(d)[6:] for d in glob.glob(f"{ra}/pr_evt*")}
        eb = {os.path.basename(d)[6:] for d in glob.glob(f"{rb}/pr_evt*")}
        if only is not None:
            ea &= only
            eb &= only
        if ea != eb:
            print(f"{s}: FAIL event sets differ: only A {sorted(ea - eb)[:5]} only B {sorted(eb - ea)[:5]}")
            failed.append((s, "event set"))
        jobs = [(ra, rb, e, set(a.allow), set(a.prefix)) for e in sorted(ea & eb, key=int)]
        with Pool(a.jobs) as p:
            for evt, res, notes in p.imap(one, jobs):
                total.update(res)
                total[f"events {s}"] += 1
                if any(k.startswith("FAIL") or " FAIL" in k for k in res):
                    failed.append((s, evt, notes[:6]))
    print(f"== d109 gate {a.labelA} vs {a.labelB} ({', '.join(a.samples)})")
    for k in sorted(total):
        print(f"  {k:60s} {total[k]}")
    print(f"== failing events: {len(failed)}")
    for f in failed[:30]:
        print("  ", f)
    print("=== OVERALL:", "PASS" if not failed else "FAIL")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
