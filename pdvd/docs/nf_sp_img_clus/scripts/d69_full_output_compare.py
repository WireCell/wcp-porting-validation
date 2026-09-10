#!/usr/bin/env python3
"""doc pdvd/69 -- compare two PR arms on EVERY saved output, and replay the
comparator doc pdvd/61 used for its legacy-tail "non-determinism" claim.

Read-only: opens files under <det>/work/, writes nothing anywhere.

Default mode (the full-output gate):

    d69_full_output_compare.py --before 'pdvd/work/*_d65vleg' --after 'pdvd/work/*_d66vleg' \
        --before-arm d65vleg --after-arm d66vleg

  pairs event dirs by prefix and compares, per event:
    * every TTree in tracking-pr.root and tracking-stm.root, every shared branch,
      BIT-identically: the flattened values' raw bytes (so NaN payloads, -0.0 and
      dtype all count) plus the per-entry counts at every nesting level (so a
      jagged branch cannot pass by having the same flat content re-partitioned);
    * calib-pr-evt*.json by md5 (plain JSON text, no embedded timestamp);
    * mabc-pr.zip by abtest/hash_archive.py (member content, CLAUDE.md M2).
  Branches present on one side only are listed, not scored.  Exit 1 if anything
  shared differs.

  d51g_branch_census.py, the per-round gate, reads only T_stm_michel and
  T_stm_michel_pts; this reads everything the job saved.

--replay-d61 mode:

    d69_full_output_compare.py --replay-d61 [--pair DIR_A DIR_B ...]

  runs doc 61's inline check (session d9d688aa, transcript L31290 / L31305,
  reproduced verbatim in replay_d61() below) next to the bit comparison above,
  on doc 61's own saved control dirs by default.  For each tree it prints how
  many branches the old check flagged, how many are jagged (vector<>), how many
  REALLY differ, and the longest per-entry vector among any jagged branch the
  old check let through.
"""
import argparse
import collections
import glob
import hashlib
import os
import subprocess
import sys

import awkward as ak
import numpy as np
import uproot

IMG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
HASH_ARCHIVE = os.path.join(IMG, "abtest", "hash_archive.py")
ROOT_FILES = ("tracking-pr.root", "tracking-stm.root")

# doc 61 sec 5.1's control: item1a = pre-fix binary, item1b = post-fix binary,
# item1a2 = the pre-fix binary run a second time (evt 0 only).
D61_PAIRS = [
    ("pdvd/work/039252_0_d61item1a", "pdvd/work/039252_0_d61item1a2"),
    ("pdvd/work/039252_0_d61item1a", "pdvd/work/039252_0_d61item1b"),
    ("pdvd/work/039252_10_d61item1a", "pdvd/work/039252_10_d61item1b"),
    ("pdvd/work/039252_12_d61item1a", "pdvd/work/039252_12_d61item1b"),
]
D61_TREES = ["T_rec_charge", "T_tagger", "T_kine", "T_proj_data"]


def trees(f):
    """{name: TTree} for the TTrees in an open file (cycle numbers dropped)."""
    out = {}
    for k in f.keys():
        obj = f[k]
        if hasattr(obj, "num_entries"):
            out[k.split(";")[0]] = obj
    return out


def branch_differs(ba, bb):
    """True unless the two branches are bit-identical, structure included."""
    a = ba.array(library="ak")
    b = bb.array(library="ak")
    if len(a) != len(b) or a.ndim != b.ndim:
        return True
    for axis in range(1, a.ndim):
        ca = ak.to_numpy(ak.flatten(ak.num(a, axis=axis), axis=None))
        cb = ak.to_numpy(ak.flatten(ak.num(b, axis=axis), axis=None))
        if not np.array_equal(ca, cb):
            return True
    try:
        fa = ak.to_numpy(ak.flatten(a, axis=None))
        fb = ak.to_numpy(ak.flatten(b, axis=None))
    except Exception:                       # strings and other non-numeric leaves
        return ak.to_list(a) != ak.to_list(b)
    return fa.dtype != fb.dtype or fa.tobytes() != fb.tobytes()


def is_jagged(br):
    return br.array(library="ak").ndim > 1


def max_entry_len(br):
    a = br.array(library="ak")
    return int(ak.max(ak.num(a, axis=1))) if len(a) else 0


def md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def archive_hash(path):
    r = subprocess.run([sys.executable, HASH_ARCHIVE, path],
                       capture_output=True, text=True, check=True)
    return r.stdout.split()[0]


def full_compare(args):
    os.chdir(IMG)
    before = {os.path.basename(d)[: -len(args.before_arm) - 1]: d
              for d in sorted(glob.glob(args.before))}
    after = {os.path.basename(d)[: -len(args.after_arm) - 1]: d
             for d in sorted(glob.glob(args.after))}
    common = sorted(set(before) & set(after))
    if not common:                           # a bad glob must not print PASS
        print("REFUSE: no event pairs matched (%d before dir(s), %d after dir(s)) -- "
              "check the globs and arm names" % (len(before), len(after)), file=sys.stderr)
        return 2

    ev_diff = collections.Counter()          # "file/tree" -> events with any difference
    br_diff = collections.defaultdict(set)   # "file/tree" -> branches that ever differed
    n_branch_cmp = collections.Counter()     # "file/tree" -> branch comparisons made
    one_sided = collections.defaultdict(set)
    presence = collections.Counter()
    calib_diff = mabc_diff = calib_n = mabc_n = 0

    for ev in common:
        da, db = before[ev], after[ev]
        for fn in ROOT_FILES:
            pa, pb = os.path.join(da, fn), os.path.join(db, fn)
            if not (os.path.exists(pa) and os.path.exists(pb)):
                if os.path.exists(pa) != os.path.exists(pb):
                    presence[fn] += 1
                continue
            ta, tb = trees(uproot.open(pa)), trees(uproot.open(pb))
            for name in sorted(set(ta) | set(tb)):
                tag = "%s/%s" % (fn[:-5], name)
                if name not in ta or name not in tb:
                    presence[tag] += 1
                    continue
                A, B = ta[name], tb[name]
                sa, sb = set(A.keys()), set(B.keys())
                one_sided[tag] |= {"-" + x for x in sa - sb} | {"+" + x for x in sb - sa}
                if A.num_entries != B.num_entries:
                    ev_diff[tag] += 1
                    br_diff[tag].add("<num_entries>")
                    continue
                bad = [x for x in sorted(sa & sb) if branch_differs(A[x], B[x])]
                n_branch_cmp[tag] += len(sa & sb)
                if bad:
                    ev_diff[tag] += 1
                    br_diff[tag] |= set(bad)
        ca = sorted(glob.glob(os.path.join(da, "calib-pr-evt*.json")))
        cb = sorted(glob.glob(os.path.join(db, "calib-pr-evt*.json")))
        if ca and cb:
            calib_n += 1
            calib_diff += md5(ca[0]) != md5(cb[0])
        za, zb = os.path.join(da, "mabc-pr.zip"), os.path.join(db, "mabc-pr.zip")
        if os.path.exists(za) and os.path.exists(zb):
            mabc_n += 1
            mabc_diff += archive_hash(za) != archive_hash(zb)

    if sum(n_branch_cmp.values()) == 0:
        print("REFUSE: %d event pair(s) matched but no branch was compared -- "
              "no ROOT files found?" % len(common), file=sys.stderr)
        return 2

    print("=" * 78)
    print("%s vs %s" % (args.before_arm, args.after_arm))
    print("  event pairs compared : %d  (before-only %d, after-only %d)"
          % (len(common), len(set(before) - set(after)), len(set(after) - set(before))))
    total_diff = 0
    for tag in sorted(n_branch_cmp):
        print("  %-32s branch comparisons %7d   events differing %3d   branches differing %d"
              % (tag, n_branch_cmp[tag], ev_diff[tag], len(br_diff[tag])))
        total_diff += ev_diff[tag]
        if br_diff[tag]:
            print("      e.g. %s" % sorted(br_diff[tag])[:10])
    for tag, s in sorted(one_sided.items()):
        if s:
            print("  %-32s one-sided branches (not scored): %s" % (tag, sorted(s)))
    for tag, n in sorted(presence.items()):
        print("  %-32s present on one side only in %d event(s)" % (tag, n))
        total_diff += n
    print("  calib-pr json md5 differ   : %d of %d" % (calib_diff, calib_n))
    print("  mabc-pr.zip content differ : %d of %d" % (mabc_diff, mabc_n))
    total_diff += calib_diff + mabc_diff
    print("  VERDICT: %s" % ("BIT-IDENTICAL on every shared output" if total_diff == 0
                             else "DIFFERS (%d event-level differences)" % total_diff))
    return 0 if total_diff == 0 else 1


def replay_d61(ta, tb):
    """doc 61's inline check, verbatim: library='np' + np.array_equal in try/except."""
    bad, excs = [], collections.Counter()
    for k in sorted(set(ta.keys()) & set(tb.keys())):
        va, vb = ta[k], tb[k]
        try:
            ok = (np.array_equal(va, vb, equal_nan=True) if va.dtype.kind in "fc"
                  else np.array_equal(va, vb))
        except Exception as ex:
            ok = False
            excs["%s: %s" % (type(ex).__name__, str(ex)[:60])] += 1
        if not ok:
            bad.append(k)
    return bad, excs


def replay(args):
    os.chdir(IMG)
    pairs = [tuple(p) for p in args.pair] if args.pair else D61_PAIRS
    for da, db in pairs:
        fa = uproot.open(os.path.join(da, "tracking-pr.root"))
        fb = uproot.open(os.path.join(db, "tracking-pr.root"))
        print("=" * 78)
        print("%s  vs  %s" % (os.path.basename(da), os.path.basename(db)))
        for name in D61_TREES:
            A, B = fa[name], fb[name]
            flagged, excs = replay_d61(A.arrays(library="np"), B.arrays(library="np"))
            jag = {k for k in A.keys() if is_jagged(A[k])}
            real = [k for k in A.keys() if branch_differs(A[k], B[k])]
            escaped = sorted(jag - set(flagged))
            esc_len = max((max_entry_len(A[k]) for k in escaped), default=None)
            print("  %-13s branches %4d | doc-61 check flagged %4d | jagged %4d | "
                  "flagged within jagged: %-5s | REAL diffs %d"
                  % (name, len(A.keys()), len(flagged), len(jag),
                     set(flagged) <= jag, len(real)))
            if escaped:
                print("      jagged but not flagged: %d, longest per-entry vector among them %s  %s"
                      % (len(escaped), esc_len, escaped[:9]))
            for e, n in excs.items():
                print("      exception x%d: %s" % (n, e))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--before")
    ap.add_argument("--after")
    ap.add_argument("--before-arm")
    ap.add_argument("--after-arm")
    ap.add_argument("--replay-d61", action="store_true")
    ap.add_argument("--pair", nargs=2, action="append", metavar=("DIR_A", "DIR_B"),
                    help="replay mode: an event-dir pair relative to the repo root")
    args = ap.parse_args()
    if args.replay_d61:
        return replay(args)
    if not all([args.before, args.after, args.before_arm, args.after_arm]):
        ap.error("need --before --after --before-arm --after-arm (or --replay-d61)")
    return full_compare(args)


if __name__ == "__main__":
    sys.exit(main())
