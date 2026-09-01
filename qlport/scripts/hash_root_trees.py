#!/usr/bin/env python3
"""doc 90 sec 7: content hash of a ROOT file's TREES, mtime/UUID-insensitive.

Why this exists.  repeat_check.sh counted distinct Bee-zip content hashes and
called 1 distinct "DETERMINISTIC".  The Bee zip carries the CLUSTERING output.
uBooNE event 5384-136-6805 is bistable in the TAGGER output -- four kine_pio_*
variables swing between two states (kine_pio_angle 14.81 vs 109.51, 95 degrees)
while the Bee zip stays byte-identical every time.  The old gate reported
"deterministic" throughout.  This hashes the trees instead, so that class of
defect is visible.

Raw file bytes are useless here: ROOT embeds a UUID and timestamps, so two
identical-content files never compare equal (the same trap as tarballs, M2).
Hashing branch VALUES is what makes the comparison meaningful.

NaN handling is load-bearing.  T_rec_charge.reduced_chi2 carries NaNs and NaN
!= NaN, so a naive hash of raw bytes makes every run differ -- and, worse, NaN
bit patterns are not guaranteed identical across runs.  Every NaN is
canonicalized to one sentinel before hashing.

ROW ORDER is the other trap, and it is why the first version of this script was
useless.  T_rec_charge emits its per-fitted-point rows in a run-dependent
ORDER: on uBooNE event 6501 all 16 varying branches are identical once sorted,
same 49 rows, permuted.  A sequence-sensitive hash therefore flags every event
on every run and detects nothing.  So the CONTENT hash sorts rows into a
canonical order first.  The emission order is still reported, as a separate
"order" hash, so that instability stays visible instead of being hidden.

Usage:
    hash_root_trees.py <file.root> [...]                  # default physics trees
    hash_root_trees.py --trees T_kine,T_tagger <file>...  # explicit
    hash_root_trees.py --per-tree <file>                  # one line per tree
Output mirrors hash_archive.py: "<sha256>  <path>", so it composes the same way.
"""
import argparse, hashlib, sys
import numpy as np
import uproot

# The physics trees.  Trun/T_proj_data are run metadata; T_bad_ch is channel
# masking.  Defaults are what a determinism gate should care about.
DEFAULT_TREES = ["T_kine", "T_tagger", "T_rec_charge"]
NAN_SENTINEL = np.float64(-1.234567890123456e300)  # one canonical stand-in


def _feed(h, arr, name):
    h.update(name.encode())
    a = np.asarray(arr)
    if a.dtype == object:  # jagged: hash element lengths, then flattened values
        lens = np.array([len(x) if hasattr(x, "__len__") else 1 for x in a], dtype=np.int64)
        h.update(b"|jag|"); h.update(lens.tobytes())
        flat = [np.asarray(x).ravel() for x in a if hasattr(x, "__len__")]
        a = np.concatenate(flat) if flat else np.array([], dtype=np.float64)
    if a.dtype.kind == "f":
        a = np.where(np.isnan(a), NAN_SENTINEL, a.astype(np.float64))
    h.update(str(a.dtype).encode()); h.update(a.tobytes())


def _rowsorted_hash(t, branches):
    """Hash the MULTISET of rows: build whole rows, sort them, then hash.

    Sorting each branch independently would destroy row correspondence and let
    two genuinely different tables collide.  Rows are kept intact.
    """
    cols, names = [], []
    for bn in branches:
        try:
            a = np.asarray(t[bn].array(library="np"))
        except Exception:
            continue
        if a.dtype == object or a.ndim != 1:
            continue  # jagged/array branches: handled by the ordered hash only
        if a.dtype.kind == "f":
            a = np.where(np.isnan(a), NAN_SENTINEL, a.astype(np.float64))
        cols.append(a); names.append(bn)
    if not cols:
        e = hashlib.sha256().hexdigest()
        return e, e
    n = min(len(c) for c in cols)
    keys = [np.asarray(c[:n], dtype=np.float64) for c in cols]
    order = np.lexsort(tuple(reversed(keys)))  # first branch is primary key

    def digest(perm):
        h = hashlib.sha256()
        for nm, c in zip(names, cols):
            h.update(nm.encode()); h.update(str(c.dtype).encode())
            h.update(np.asarray(c[:n])[perm].tobytes())
        return h.hexdigest()

    # Both digests use the SAME algorithm, differing only by the permutation,
    # so comparing them is a real statement about row order.  (Comparing the
    # sorted digest against the sequence-oriented _feed() digest would compare
    # two different algorithms and never match -- an earlier bug here.)
    return digest(order), digest(np.arange(n))


def hash_file(path, trees, per_tree=False, ordered=False):
    r = uproot.open(path)
    present = {k.split(";")[0] for k in r.keys()}
    out = {}
    h_all = hashlib.sha256()
    for tn in trees:  # fixed order: the caller's list, not file order
        h_all.update(tn.encode())
        if tn not in present:
            h_all.update(b"|absent|")
            if per_tree:
                out[tn] = ("(absent)", True)
            continue
        t = r[tn]
        h_t = hashlib.sha256()
        for bn in sorted(t.keys()):
            try:
                _feed(h_t, t[bn].array(library="np"), bn)
            except Exception as e:  # unreadable branch: record that, don't crash
                h_t.update(f"|unreadable:{bn}:{type(e).__name__}|".encode())
        d_ord = h_t.hexdigest()
        d_sorted, d_asstored = _rowsorted_hash(t, sorted(t.keys()))
        d = d_asstored if ordered else d_sorted
        h_all.update(d.encode())
        if per_tree:
            out[tn] = (d, d_sorted == d_asstored)
    return h_all.hexdigest(), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--trees", default=",".join(DEFAULT_TREES),
                    help="comma-separated tree names (default: %(default)s)")
    ap.add_argument("--per-tree", action="store_true",
                    help="also print a per-tree hash, to localize a difference")
    ap.add_argument("--ordered", action="store_true",
                    help="hash rows AS STORED (sequence-sensitive).  Off by default: "
                         "T_rec_charge row order is run-dependent, so this flags "
                         "every event and detects nothing.  Use to study order itself.")
    a = ap.parse_args()
    trees = [t for t in a.trees.split(",") if t]
    rc = 0
    for f in a.files:
        try:
            d, per = hash_file(f, trees, a.per_tree, a.ordered)
        except Exception as e:
            print(f"ERROR  {f}: {type(e).__name__}: {e}", file=sys.stderr); rc = 2; continue
        print(f"{d}  {f}")
        for tn in trees:
            if tn in per:
                content, order_stable = per[tn]
                note = "" if order_stable else "   [rows not in canonical order]"
                print(f"    {content}  {tn}{note}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
