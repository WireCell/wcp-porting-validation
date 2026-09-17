#!/usr/bin/env python3
"""Round G (doc sbnd_xin/111 sec 17): restore files compressed by compress_worker_20260916f.py.

  python3 restore_compress_20260916f.py p85vprod d53v        # dry run: what would be restored
  python3 restore_compress_20260916f.py --confirm p85vprod   # restore every file of those arm families
  python3 restore_compress_20260916f.py --confirm --all      # restore everything in the manifest
  options: --manifest PATH (default: the round-G manifest)

Per file: decompress <file>.zst to <file>.tmp, require size and SHA-256 equal to the manifest row (the
original's hash, frozen before compression), set the manifest mtime, rename to <file>, remove <file>.zst.
The restored file is byte-identical to the original.  Exit 0 all restored | 20 any verification failed.
"""
import os, re, sys, hashlib, subprocess
from concurrent.futures import ProcessPoolExecutor

R   = "/home/xqian/toolkit-dev/wcp-porting-img"
MAN = f"{R}/sbnd/sbnd_xin/archive/records/cleanup-20260916f/pdvd-compress.manifest.tsv"
ARM = re.compile(r"/\d{6}_\d+_([^/]+)/")

def one(args):
    p, size, mtime, want = args
    src, tmp = p + ".zst", p + ".tmp"
    if not os.path.exists(src): return ("already-restored" if os.path.exists(p) else "missing", p)
    if os.path.exists(p): return ("both-exist", p)
    h, n = hashlib.sha256(), 0
    with open(tmp, "wb") as out:
        d = subprocess.Popen(["zstd", "-d", "-c", "-q", src], stdout=subprocess.PIPE)
        for b in iter(lambda: d.stdout.read(1 << 20), b""):
            h.update(b); n += len(b); out.write(b)
        d.wait()
    if d.returncode != 0 or n != size or h.hexdigest() != want:
        os.remove(tmp); return ("verify-failed", p)
    os.utime(tmp, (mtime, mtime))
    os.rename(tmp, p)
    os.remove(src)
    return ("restored", p)

def main():
    a = sys.argv[1:]
    man = MAN
    if "--manifest" in a:
        i = a.index("--manifest"); man = a[i + 1]; del a[i:i + 2]
    confirm = "--confirm" in a; a = [x for x in a if x != "--confirm"]
    everything = "--all" in a; fams = set(x for x in a if x != "--all")
    if not everything and not fams: sys.exit(__doc__)
    rows = []
    for line in open(man):
        p, s, m, h = line.rstrip("\n").split("\t")
        m_ = ARM.search(p)
        if everything or (m_ and m_.group(1) in fams): rows.append((p, int(s), int(m), h))
    pending = [r for r in rows if os.path.exists(r[0] + ".zst")]
    print(f"{len(rows)} manifest rows selected, {len(pending)} still compressed")
    if not confirm:
        print("DRY RUN -- add --confirm to restore"); return 0
    stat, bad = {}, []
    with ProcessPoolExecutor(16) as ex:
        for st, p in ex.map(one, pending, chunksize=4):
            stat[st] = stat.get(st, 0) + 1
            if st != "restored": bad.append((st, p))
    print(f"status: {stat}")
    for st, p in bad[:10]: print(f"NOT RESTORED ({st}): {p}")
    return 0 if not bad else 20

if __name__ == "__main__":
    sys.exit(main())
