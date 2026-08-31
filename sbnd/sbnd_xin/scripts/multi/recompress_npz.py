#!/usr/bin/env python3
"""doc 81 round 4 -- recompress STORED icluster npz IN PLACE, losslessly.

Why this exists.  Until doc 81 round 2 (wcp 3d05bf9) split_group_products.py
wrote every per-event `evt<ID>/icluster-*.npz` with ZIP_STORED, on the premise
that "npz is uncompressed".  The premise is false -- WCT's own ClusterFileSink
goes through custard's miniz_sink, i.e. mz_zip_writer_add_mem(..., MZ_BEST_SPEED)
-- so the legacy per-event path always wrote DEFLATE and only the group path's
splitter produced STORED ones, ~3.8x larger for the same members.  Round 2 fixed
the writer; this fixes the arms already on disk.

WHY THIS IS SAFE, and why it is nonetheless written paranoid.  A zip container's
compression method is not data: every gate in this tree hashes the DECOMPRESSED
member payload (`abtest/hash_archive.py:19-30` and `scripts/multi/stagea_gate.py:22-30`
both `z.read(name)` / `extractfile().read()`), and WCT reads these through miniz,
which inflates transparently.  Measured, not assumed: a Q/L re-run off
hand-compressed imaging reproduced all four products byte-for-byte (doc 81 sec 10.2).

But since doc 81 round 3 retired `work-img-<s>`, the grp0825 copy is the ONLY
copy of that imaging.  So every file is verified member-for-member BEFORE the
original is replaced, and the replace is atomic:

  1. read the original, record namelist ORDER + sha256 of every payload
  2. write a sibling .tmp with ZIP_DEFLATED, same order
  3. re-open the .tmp and require identical order AND identical payload hashes
  4. fsync, then os.replace  (atomic on the same filesystem)

Any mismatch leaves the original untouched and counts as an error.  Already-
DEFLATE files are skipped, so the script is idempotent and safe to re-run after
an interruption.

usage: recompress_npz.py [--dry-run] [--jobs N] [--glob PAT] <root> [<root> ...]
"""
import argparse, hashlib, os, sys, zipfile
from concurrent.futures import ProcessPoolExecutor

DEFAULT_GLOB = "evt*/icluster-*.npz"


def payloads(path):
    """[(name, sha256)] in archive order -- order is load-bearing (doc 76 r2)."""
    with zipfile.ZipFile(path) as z:
        return [(n, hashlib.sha256(z.read(n)).hexdigest()) for n in z.namelist()]


def is_stored(path):
    with zipfile.ZipFile(path) as z:
        il = z.infolist()
        return bool(il) and all(i.compress_type == zipfile.ZIP_STORED for i in il)


def recompress(path):
    """-> (status, before, after).  status in skip|done|error:<why>."""
    try:
        before = os.path.getsize(path)
        if not is_stored(path):
            return ("skip", before, before)
        want = payloads(path)
        tmp = path + ".recompress.tmp"
        with zipfile.ZipFile(path) as zi, \
             zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zo:
            for n, _ in want:
                zo.writestr(n, zi.read(n))
        # verify BEFORE replacing -- this is the whole safety argument
        if payloads(tmp) != want:
            os.unlink(tmp)
            return ("error:payload-mismatch", before, before)
        fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        after = os.path.getsize(tmp)
        os.replace(tmp, path)
        return ("done", before, after)
    except Exception as e:                                  # noqa: BLE001
        tmp = path + ".recompress.tmp"
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return ("error:%s" % type(e).__name__, 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import glob as _g
    files = []
    for r in args.roots:
        files += sorted(_g.glob(os.path.join(r, args.glob)))
    if not files:
        raise SystemExit("no files matched %r under %s" % (args.glob, args.roots))

    if args.dry_run:
        st = sum(os.path.getsize(f) for f in files if is_stored(f))
        n = sum(1 for f in files if is_stored(f))
        print("%d files matched, %d STORED, %.2f GiB; est. -> %.2f GiB at 26.6%%"
              % (len(files), n, st / 2**30, st * 0.266 / 2**30))
        return 0

    tot_b = tot_a = 0
    counts = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for i, (status, b, a) in enumerate(ex.map(recompress, files, chunksize=8), 1):
            counts[status.split(":")[0]] = counts.get(status.split(":")[0], 0) + 1
            tot_b += b
            tot_a += a
            if status.startswith("error"):
                print("  ERROR %s: %s" % (files[i - 1], status), file=sys.stderr)
            if i % 500 == 0:
                print("  %d/%d  %.2f -> %.2f GiB"
                      % (i, len(files), tot_b / 2**30, tot_a / 2**30), flush=True)

    print("files=%d  %s" % (len(files), counts))
    print("%.2f GiB -> %.2f GiB  (saved %.2f GiB, %.1f%%)"
          % (tot_b / 2**30, tot_a / 2**30, (tot_b - tot_a) / 2**30,
             100.0 * tot_a / tot_b if tot_b else 0))
    return 1 if counts.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
