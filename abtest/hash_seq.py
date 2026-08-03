#!/usr/bin/env python3
"""Sequential member-content hash for a tar[.bz2] -- same content semantics as
abtest/hash_archive.py (sha256 over name+payload, rolled up over members sorted
by name) but reads the stream ONCE in archive order.

hash_archive.py sorts members and then seeks, which on a .bz2 forces a fresh
decompression from the start for every member -- O(n^2) on a 48-member, 1.4 GB
frame archive.  Here we hash while streaming and sort the per-member digests at
the end, which gives the same rollup for a fixed member set.
"""
import hashlib
import sys
import tarfile


def member_digests(path):
    out = {}
    mode = "r|bz2" if path.endswith(".bz2") else ("r|gz" if path.endswith(".gz") else "r|")
    with tarfile.open(path, mode) as tf:
        for ti in tf:
            if not ti.isfile():
                continue
            h = hashlib.sha256()
            h.update(ti.name.encode())
            f = tf.extractfile(ti)
            while True:
                b = f.read(1 << 20)
                if not b:
                    break
                h.update(b)
            out[ti.name] = h.hexdigest()
    return out


def rollup(d):
    h = hashlib.sha256()
    for name in sorted(d):
        h.update(name.encode())
        h.update(bytes.fromhex(d[name]))
    return h.hexdigest(), len(d)


if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]
    da, db = member_digests(a), member_digests(b)
    ra, rb = rollup(da), rollup(db)
    print(f"A {ra[0]}  {ra[1]} members  {a}")
    print(f"B {rb[0]}  {rb[1]} members  {b}")
    if da == db:
        print("VERDICT: IDENTICAL (every member's content hash matches)")
    else:
        only_a = sorted(set(da) - set(db))
        only_b = sorted(set(db) - set(da))
        diff = sorted(n for n in set(da) & set(db) if da[n] != db[n])
        print(f"VERDICT: DIFF  only_in_A={len(only_a)} only_in_B={len(only_b)} "
              f"content_differs={len(diff)}")
        for n in (only_a + only_b + diff)[:10]:
            print("   ", n)
