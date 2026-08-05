#!/usr/bin/env python3
"""doc pr/37 sec.5.2 -- same-binary repeat-run identity check for the SBND PR chain.

Compares two arms produced by identical invocations of run_pr_chain_batch.sh on
the same installed binary.  Four artifact classes per event:

  mabc-pr.zip              member-CONTENT hash (abtest/hash_archive.py) -- never
                           the raw archive bytes, which embed mtimes (CLAUDE.md M2)
  pctree-pr-evt<ID>.tar.gz member-CONTENT hash, same reason
  nusel-evt<ID>.tsv        exact bytes
  tracking-pr.root         LEAF-level compare of every tree, via uproot

Trap this script exists to avoid (pr/32 sec.11.1): hash_archive.py prints the
archive PATH in its output line, so hashing its stdout makes every arm "differ".
Only the first field is used.

Usage: pr37_repeat_cmp.py ARM_A ARM_B
"""
import hashlib
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HASHER = os.path.join(HERE, "..", "..", "abtest", "hash_archive.py")


def archive_hash(path):
    """Rollup content hash of an archive's members -- first field only."""
    out = subprocess.run([sys.executable, HASHER, path],
                         capture_output=True, text=True, check=True).stdout
    return out.split()[0]


def _walk(x, h):
    """Recursively feed a possibly-ragged leaf value into the hasher.

    NEVER repr(): a numpy object array holding python objects reprs as
    '<... at 0x7f...>', i.e. a heap address, which makes two identical runs
    hash differently.  That produced a phantom 235-leaf "regression" on the
    five vector<vector<T>> branches of T_proj_data before this was fixed --
    every value was in fact bit-identical and in the same order.  Descend to
    numeric leaves and hash their bytes instead.
    """
    import numpy as np
    a = np.asarray(x, dtype=object) if isinstance(x, (list, tuple)) else np.asarray(x)
    if a.dtype == object:
        h.update(b"[")
        for y in a.ravel():
            _walk(y, h)
            h.update(b"|")
        h.update(b"]")
        return
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    if a.dtype.kind in "US":
        h.update(a.astype("U").tobytes())
    else:
        h.update(np.ascontiguousarray(a).tobytes())


def _digest(a):
    """Canonical sha256 of one leaf's values.

    Exact-bit, deliberately: this is a determinism floor, so two runs must agree
    to the last ULP.  Order-sensitive, so a pure permutation is reported as a
    difference -- classify reorder vs value change before calling it a
    regression (pr/28 sec.11.1 is the precedent).
    """
    h = hashlib.sha256()
    _walk(a, h)
    return h.hexdigest()


def root_leaves(path):
    """{f'{tree}/{branch}': sha256-of-values} for every tree in the file."""
    import uproot
    vals = {}
    with uproot.open(path) as f:
        for key, obj in f.items():
            if not hasattr(obj, "keys"):
                continue
            tname = key.split(";")[0]
            try:
                arrs = obj.arrays(library="np")
            except Exception:
                continue
            for br, a in arrs.items():
                try:
                    vals[f"{tname}/{br}"] = _digest(a)
                except Exception as exc:          # never silently skip a leaf
                    vals[f"{tname}/{br}"] = f"UNHASHABLE:{exc.__class__.__name__}"
    return vals


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a_root, b_root = sys.argv[1], sys.argv[2]

    evts = sorted(int(re.sub(r"\D", "", d)) for d in os.listdir(a_root)
                  if d.startswith("pr_evt"))
    print(f"events: {len(evts)}  arms: {a_root} vs {b_root}")

    n_arch_ok = n_arch_diff = 0
    n_tsv_ok = n_tsv_diff = 0
    leaf_total = leaf_diff = 0
    diff_leaves = {}
    unhashable = set()
    missing = []

    for e in evts:
        ad = os.path.join(a_root, f"pr_evt{e}")
        bd = os.path.join(b_root, f"pr_evt{e}")
        for name in ("mabc-pr.zip", f"pctree-pr-evt{e}.tar.gz"):
            pa, pb = os.path.join(ad, name), os.path.join(bd, name)
            if not (os.path.exists(pa) and os.path.exists(pb)):
                missing.append(f"{e}:{name}")
                continue
            if archive_hash(pa) == archive_hash(pb):
                n_arch_ok += 1
            else:
                n_arch_diff += 1
                print(f"  ARCHIVE DIFF evt {e} {name}")

        pa, pb = os.path.join(ad, f"nusel-evt{e}.tsv"), os.path.join(bd, f"nusel-evt{e}.tsv")
        if os.path.exists(pa) and os.path.exists(pb):
            if open(pa, "rb").read() == open(pb, "rb").read():
                n_tsv_ok += 1
            else:
                n_tsv_diff += 1
                print(f"  TSV DIFF evt {e}")

        pa, pb = os.path.join(ad, "tracking-pr.root"), os.path.join(bd, "tracking-pr.root")
        if os.path.exists(pa) and os.path.exists(pb):
            try:
                la, lb = root_leaves(pa), root_leaves(pb)
            except ImportError:
                print("  (uproot unavailable -- ROOT leaves skipped)")
                break
            for k in sorted(set(la) | set(lb)):
                leaf_total += 1
                if la.get(k) != lb.get(k):
                    leaf_diff += 1
                    diff_leaves[k] = diff_leaves.get(k, 0) + 1
                if str(la.get(k)).startswith("UNHASHABLE") or \
                   str(lb.get(k)).startswith("UNHASHABLE"):
                    unhashable.add(k)

    print(f"\narchive member-content hashes : {n_arch_ok} identical, {n_arch_diff} differ")
    print(f"nusel TSVs                    : {n_tsv_ok} identical, {n_tsv_diff} differ")
    print(f"ROOT leaves                   : {leaf_total} compared, {leaf_diff} differ")
    if diff_leaves:
        print("  per-leaf event counts (top 20):")
        for k, n in sorted(diff_leaves.items(), key=lambda kv: -kv[1])[:20]:
            print(f"    {n:3d}  {k}")
    if unhashable:
        print(f"  leaves that could not be hashed: {len(unhashable)} -> {sorted(unhashable)[:5]}")
    if missing:
        print(f"missing artifacts: {len(missing)} -> {missing[:5]}")

    bad = n_arch_diff + n_tsv_diff + leaf_diff
    print(f"\nFLOOR = {bad}  ({'PASS -- bit-identical' if bad == 0 else 'NON-ZERO'})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
