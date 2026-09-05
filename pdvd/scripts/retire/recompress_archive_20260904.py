#!/usr/bin/env python3
"""doc 89 Phase 4a -- re-encode the archive record layer from gzip to zstd-19.

This is a RE-ENCODING, not a deletion of scientific record (M13).  Every
tarball is rewritten, verified member-for-member, and only then is the .gz
removed.  A tarball that fails verification keeps its .gz and is reported.

Why it is worth doing.  archive/records is 16.5 G of per-arm record tarballs
whose members are per-event logs and compiled configs -- ~100 near-identical
copies of the same text per arm.  gzip's 32 KiB window cannot see across
files; zstd -19 can.  Measured on
prod0825-groupmode-20260825/other/work-pr112i-off-mcp2k.tar.gz:

    128 MB .gz  ->  751 MB raw  ->  8 MB .zst        (16x smaller than the gz)
    round-trip: 14003 members, name/size/sha256 identical

Plain `zstd -19`, deliberately NOT `--long=31`.  A --long=31 frame needs a 2 GB
window to DECODE, so `zstd -d` fails on it without the matching flag -- a
footgun for anyone reading this archive years from now, and the first run of
this script failed its own verification on exactly that.  The measured 16x
does not need it.

EXCLUDED, because they are already-compressed binary and do not shrink:
    clean-slate-20260805/imaging-bases/work-mcp1000.tar.gz  (1.3 G)
    clean-slate-20260805/imaging-bases/work.tar.gz          (0.49 G)
a 382 MB sample of the first went to 381 MB.  They stay .gz.

Also excluded: archive/records/labels/, which holds VERBATIM copies (not
tarballs) of hand-scan label dirs and is what ASSERT 2/6/6b compare against.

Nothing in either repo reads these tarballs programmatically -- the only
reference is pr126_pi0_select.py's error message, which names the directory --
so the extension change is safe.  It is recorded anyway.

Usage:
    scripts/retire/recompress_archive_20260901.py [--jobs N] [--apply]
Default is a DRY RUN that reports the candidate set and the projected saving.
"""
import argparse, hashlib, os, subprocess, sys, tarfile
from concurrent.futures import ProcessPoolExecutor

ROOT = "/home/xqian/toolkit-dev/wcp-porting-img/pdvd"   # doc pdvd/29 fork; ONLY this line differs
REC = os.path.join(ROOT, "archive", "records")
STATE = os.path.join(ROOT, "scripts", "retire", "state-20260904")
SKIP_DIRS = {"labels"}
SKIP_FILES = {
    "clean-slate-20260805/imaging-bases/work-mcp1000.tar.gz",
    "clean-slate-20260805/imaging-bases/work.tar.gz",
}


def digest(cmd, path):
    """(name, size, sha256) per file member, streamed through `cmd`."""
    p = subprocess.Popen(cmd + [path], stdout=subprocess.PIPE)
    out = []
    try:
        with tarfile.open(fileobj=p.stdout, mode="r|") as tf:
            for m in tf:
                h = hashlib.sha256()
                if m.isfile():
                    f = tf.extractfile(m)
                    for c in iter(lambda: f.read(1 << 20), b""):
                        h.update(c)
                out.append((m.name, m.size, h.hexdigest()))
    finally:
        p.stdout.close(); p.wait()
    return out


def one(rel):
    src = os.path.join(REC, rel)
    dst = src[:-3] + ".zst"          # .tar.gz -> .tar.zst
    before = os.path.getsize(src)
    try:
        rc = subprocess.run(
            f'gzip -dc "{src}" | zstd -19 -q -T4 -o "{dst}" -f',
            shell=True, capture_output=True)
        if rc.returncode != 0 or not os.path.exists(dst):
            return (rel, before, 0, "FAIL", f"zstd rc={rc.returncode}")
        a = digest(["gzip", "-dc"], src)
        b = digest(["zstd", "-dc"], dst)
        if a != b:
            os.remove(dst)
            return (rel, before, 0, "FAIL", f"member mismatch ({len(a)} vs {len(b)})")
        after = os.path.getsize(dst)
        os.remove(src)
        return (rel, before, after, "OK", f"{len(a)} members")
    except Exception as e:
        if os.path.exists(dst):
            os.remove(dst)
        return (rel, before, 0, "FAIL", f"{type(e).__name__}: {e}")


def candidates(min_bytes=0):
    out = []
    for dirpath, dirnames, files in os.walk(REC):
        dirnames[:] = [d for d in dirnames
                       if os.path.relpath(os.path.join(dirpath, d), REC) not in SKIP_DIRS]
        for f in files:
            if not f.endswith(".tar.gz"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, f), REC)
            if rel in SKIP_FILES:
                continue
            if os.path.getsize(os.path.join(dirpath, f)) < min_bytes:
                continue
            out.append(rel)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)   # each worker uses -T4
    ap.add_argument("--apply", action="store_true")
    # The size distribution is extreme: 3978 tarballs, median 0.64 MB, but the
    # top 800 hold 85% of the bytes and the top 1600 hold 96%.  Converting the
    # ~2400 sub-MB files costs most of the wall time for <5% of the saving, so
    # there is a floor.  Small tarballs simply stay .gz -- a mixed archive is
    # fine, both extensions decode with no special flags.
    ap.add_argument("--min-mb", type=float, default=1.0)
    a = ap.parse_args()
    cands = candidates(int(a.min_mb * 2**20))
    tot = sum(os.path.getsize(os.path.join(REC, c)) for c in cands)
    print(f"archive/records            {sum(os.path.getsize(os.path.join(dp,f)) for dp,_,fs in os.walk(REC) for f in fs)/2**30:.2f} GiB")
    allgz = candidates(0)
    print(f"candidate .tar.gz          {len(cands)} files >= {a.min_mb} MB, {tot/2**30:.2f} GiB")
    print(f"  (of {len(allgz)} .tar.gz total; the rest are below the floor and stay .gz)")
    done = [os.path.join(dp, f) for dp, _, fs in os.walk(REC) for f in fs if f.endswith('.tar.zst')]
    print(f"already .tar.zst           {len(done)} files, "
          f"{sum(os.path.getsize(x) for x in done)/2**30:.2f} GiB (earlier pass of this round)")
    print(f"excluded (incompressible)  {len(SKIP_FILES)} files")
    print(f"excluded dirs              {sorted(SKIP_DIRS)} (verbatim label copies, M13)")
    if not a.apply:
        print("\nDRY RUN -- re-run with --apply to re-encode.")
        return 0
    os.makedirs(STATE, exist_ok=True)
    led = os.path.join(STATE, "recompress.tsv")
    # Ledger is written INCREMENTALLY.  The first run of this script was killed
    # mid-pass and its record of 156 completed files was lost, because the
    # ledger was only written at the end.  A record layer's ledger must survive
    # the thing it is recording.
    fresh = not os.path.exists(led)
    lf = open(led, "a")
    if fresh:
        lf.write("# doc 89 Phase 4a -- gzip -> zstd-19 re-encode of the record layer\n")
        lf.write("# verified member-for-member (name,size,sha256) before the .gz was removed\n")
        lf.write("rel\tbytes_gz\tbytes_zst\tverdict\tnote\n")
        lf.flush()
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for i, r in enumerate(ex.map(one, cands, chunksize=1), 1):
            rows.append(r)
            lf.write("\t".join(str(x) for x in r) + "\n"); lf.flush()
            if i % 100 == 0:
                d0 = sum(x[1] for x in rows); d1 = sum(x[2] for x in rows)
                print(f"  {i}/{len(cands)}  {d0/2**30:.2f} -> {d1/2**30:.2f} GiB", flush=True)
    lf.close()
    ok = [r for r in rows if r[3] == "OK"]
    bad = [r for r in rows if r[3] != "OK"]
    before = sum(r[1] for r in rows); after = sum(r[2] for r in ok)
    print(f"\n=== RECOMPRESSION ({len(rows)} tarballs) ===")
    print(f"OK   {len(ok)}   {before/2**30:.2f} GiB -> {after/2**30:.2f} GiB "
          f"({before/max(after,1):.1f}x, {(before-after)/2**30:.2f} GiB freed)")
    if bad:
        print(f"FAIL {len(bad)} (kept as .gz):")
        for r in bad[:10]:
            print(f"  {r[0]}: {r[4]}")
    print(f"ledger: {led}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
