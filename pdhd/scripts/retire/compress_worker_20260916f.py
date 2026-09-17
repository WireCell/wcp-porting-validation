#!/usr/bin/env python3
"""Round G (doc sbnd_xin/111 sec 17) worker: compress each listed file to <file>.zst, verified.

  compress_worker_20260916f.py LIST MANIFEST      # DRY RUN unless CONFIRM=yes in the environment

Per file, only under CONFIRM=yes:
  1. zstd -19 -T1 into <file>.zst.tmp (never over an existing <file>.zst);
  2. decompress the .tmp as a stream, SHA-256 and size must equal the MANIFEST row (frozen before any
     compression, so a file that changed since the freeze fails here and is left untouched);
  3. only then: set the .tmp's mtime to the manifest mtime, rename it to <file>.zst, remove <file>.
On any mismatch the .tmp is removed and the original stays.

Exit: 0 all done (or dry run clean) | 14 a listed file has no manifest row or its size moved |
      15 a <file>.zst already exists | 20 at least one file failed verification (originals kept).
"""
import os, sys, hashlib, subprocess
from concurrent.futures import ProcessPoolExecutor

JOBS = int(os.environ.get("COMPRESS_JOBS", "24"))

def load(man):
    rows = {}
    for line in open(man):
        p, s, m, h = line.rstrip("\n").split("\t")
        rows[p] = (int(s), int(m), h)
    return rows

def one(args):
    p, size, mtime, want = args
    tmp, dst = p + ".zst.tmp", p + ".zst"
    if os.path.lexists(dst): return ("exists", p, 0, 0)
    r = subprocess.run(["zstd", "-19", "-T1", "-q", "-f", "-o", tmp, p], capture_output=True)
    if r.returncode != 0:
        if os.path.exists(tmp): os.remove(tmp)
        return ("zstd-failed", p, 0, 0)
    h, n = hashlib.sha256(), 0
    d = subprocess.Popen(["zstd", "-d", "-c", "-q", tmp], stdout=subprocess.PIPE)
    for b in iter(lambda: d.stdout.read(1 << 20), b""):
        h.update(b); n += len(b)
    d.wait()
    if d.returncode != 0 or n != size or h.hexdigest() != want:
        os.remove(tmp)
        return ("verify-failed", p, 0, 0)
    zs = os.path.getsize(tmp)
    os.utime(tmp, (mtime, mtime))
    os.rename(tmp, dst)
    os.remove(p)
    return ("done", p, size, zs)

def main():
    lst, man = sys.argv[1], sys.argv[2]
    confirm = os.environ.get("CONFIRM", "no") == "yes"
    rows = load(man)
    paths = [l.rstrip("\n") for l in open(lst) if l.strip()]
    nomatch = [p for p in paths if p not in rows or not os.path.isfile(p) or os.lstat(p).st_size != rows[p][0]]
    if nomatch:
        print(f"   REFUSING: {len(nomatch)} of {len(paths)} listed files have no manifest row or moved in size "
              f"(e.g. {nomatch[0]}) -- the freeze is incomplete.")
        return 14
    print(f"   record gate: {len(paths)}/{len(paths)} files have a manifest row with the current size")
    pre = [p for p in paths if os.path.lexists(p + ".zst")]
    if pre:
        print(f"   REFUSING: {len(pre)} <file>.zst already exist (e.g. {pre[0]})"); return 15
    raw = sum(rows[p][0] for p in paths)
    if not confirm:
        print(f"   DRY RUN: would compress {len(paths)} files, {raw / 2**30:.2f} GiB"); return 0
    print(f"   COMPRESSING {len(paths)} files, {raw / 2**30:.2f} GiB, {JOBS} jobs ...")
    stat, before, after, bad = {}, 0, 0, []
    with ProcessPoolExecutor(JOBS) as ex:
        for st, p, s, z in ex.map(one, [(p,) + rows[p] for p in paths], chunksize=4):
            stat[st] = stat.get(st, 0) + 1
            if st == "done": before += s; after += z
            else: bad.append((st, p))
    print(f"   status: {stat}")
    print(f"   compressed {before / 2**30:.2f} GiB -> {after / 2**30:.2f} GiB, saved {(before - after) / 2**30:.2f} GiB")
    for st, p in bad[:10]: print(f"   NOT DONE ({st}): {p}")
    return 0 if not bad else 20

if __name__ == "__main__":
    sys.exit(main())
