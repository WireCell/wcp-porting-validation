#!/usr/bin/env python3
"""Freeze the ORPHANED DRIVER LOGS of cleanup ROUND I (doc sbnd_xin/121) before removal.

  usage:  python3 archive_orphanlogs_20260921.py files_orphanlogs_pdvd_20260921.txt [...]

Writes one manifest per tree: <tree>-orphanlogs.manifest.tsv with
`relpath <TAB> size <TAB> mtime <TAB> sha256` for every file in the list.  NO TAR -- see
plan_orphanlogs_20260921.py's header: `.log` is not in archive_records' HEAVY list, so a
tar here would carry 27 GiB of log text into the record layer and free a fraction of what
the round claims.  Round D sec 14 set that precedent for heavy content.

Output: $RETIRE_OUT (default sbnd_xin/archive/records/cleanup-20260921/orphanlogs).

EXIT STATUS IS THE CONTRACT, as for archive_tmp: the driver removes nothing unless this
exits 0, and this exits non-zero if ANY file failed to hash or any manifest failed to
write.  It refuses to write over an existing manifest for the same tree (M13).
"""
import os, sys, hashlib
from concurrent.futures import ProcessPoolExecutor

R    = "/home/xqian/toolkit-dev/wcp-porting-img"
OUT  = os.environ.get("RETIRE_OUT",
                      f"{R}/sbnd/sbnd_xin/archive/records/cleanup-20260921/orphanlogs")
JOBS = int(os.environ.get("RETIRE_JOBS", "16"))

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def one(path):
    try:
        st = os.stat(path)
        return path, st.st_size, int(st.st_mtime), sha(path), None
    except OSError as e:
        return path, 0, 0, "", str(e)

def main(lists):
    if not lists:
        sys.exit("usage: archive_orphanlogs_20260921.py <files_orphanlogs_*.txt> ...")
    try:
        os.makedirs(OUT, exist_ok=True)
    except OSError as e:
        sys.exit(f"FAILED creating {OUT}: {e}")
    rc = 0
    for lst in lists:
        tree = os.path.basename(lst).split("_")[2]
        man  = os.path.join(OUT, f"{tree}-orphanlogs.manifest.tsv")
        if os.path.exists(man):
            print(f"REFUSED: {man} already exists (M13)"); rc = 1; continue
        paths = [l.strip() for l in open(lst) if l.strip()]
        if not paths:
            print(f"REFUSED: {lst} is empty -- nothing to freeze"); rc = 1; continue
        bad, tot = 0, 0
        with ProcessPoolExecutor(max_workers=JOBS) as ex, open(man + ".tmp", "w") as fh:
            fh.write("# relpath\tsize\tmtime\tsha256\n")
            for path, size, mt, digest, err in ex.map(one, paths, chunksize=64):
                if err:
                    print(f"  FAILED {path}: {err}"); bad += 1; continue
                fh.write(f"{os.path.relpath(path, R)}\t{size}\t{mt}\t{digest}\n")
                tot += size
        if bad:
            os.unlink(man + ".tmp")
            print(f"{tree}: {bad} file(s) failed to hash -- manifest NOT written"); rc = 1
            continue
        os.replace(man + ".tmp", man)
        print(f"{tree}: {len(paths)} logs, {tot/2**30:.2f} GiB -> {man}")
    if rc: print("FREEZE INCOMPLETE -- the driver must refuse to remove anything.")
    return rc

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
