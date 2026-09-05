#!/usr/bin/env python3
"""Archive the record layer of the 2026-09-04 pdvd retiring arms (doc pdvd/29).

WHAT THE RECORD LAYER IS HERE.  The heavy classes -- pctree tarballs, mabc
zips, calib dumps, clusters-apa archives, tracking ROOT files, the SP+DNNROI
frame bz2s -- are dropped.  What is archived is the per-event wire-cell log,
the compiled config the arm actually ran (.wct-*.json), the resource/rss
series and img-provenance.txt.  That is what every doc 28/31/32/34/35/36/37/38
claim rests on: the exit reasons, the timings and the config each arm ran
under.  The claims stay re-checkable after the bytes are gone.

SYMLINKS ARE RECORDED, NOT FOLLOWED.  Most retiring arms are ~40% symlink by
count into d27fresh/d28dlfp/d34base, so following them would pull a copy of the
kept substrate into every record tar.  Links are written to <tag>.links.txt.

Never writes into an earlier round's archive tree (CLAUDE.md M13).
"""
import os, re, json, tarfile, sys, hashlib
from concurrent.futures import ProcessPoolExecutor

PDVD = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd"
WORK = os.path.join(PDVD, "work")
STATE = os.environ.get("RETIRE_STATE", os.path.join(PDVD, "scripts", "retire", "state-20260904"))
OUT = os.environ.get("RETIRE_OUT", os.path.join(PDVD, "archive", "records", "pdvd-rounds-20260904"))
JOBS = int(os.environ.get("RETIRE_JOBS", "16"))

HEAVY = [re.compile(p) for p in (
    r'^pctree.*\.(tar\.gz|tlas)$', r'^mabc.*\.zip$', r'^calib(-pr)?-evt.*\.json$',
    r'^clusters-apa.*\.tar\.gz$', r'^tracking-.*\.root$', r'.*\.npz$',
    r'^protodune-sp-dnnroi-frames.*\.tar\.bz2$', r'^opflash.*\.tar\.gz$',
    r'.*\.tar\.bz2$', r'^magnify.*\.root$')]

def is_heavy(n):
    return any(p.match(n) for p in HEAVY)

def archive_one(args):
    tag, group = args
    src = os.path.join(WORK, tag)
    d = os.path.join(OUT, group)
    os.makedirs(d, exist_ok=True)
    tgz, man, lnk = (os.path.join(d, tag + x) for x in
                     (".tar.gz", ".manifest.tsv", ".links.txt"))
    recs, links, dropped = [], [], 0
    for cur, subs, files in os.walk(src):
        subs[:] = [s for s in subs if not os.path.islink(os.path.join(cur, s))]
        for f in sorted(files):
            fp = os.path.join(cur, f)
            rel = os.path.relpath(fp, src)
            if os.path.islink(fp):
                links.append(f"{rel}\t{os.readlink(fp)}")
                continue
            try: sz = os.path.getsize(fp)
            except OSError: continue
            if is_heavy(f):
                dropped += sz
                continue
            recs.append((rel, fp, sz))
    with tarfile.open(tgz, "w:gz") as tf:
        for rel, fp, _ in recs:
            tf.add(fp, arcname=os.path.join(tag, rel))
    with open(man, "w") as fh:
        fh.write("kind\tstate\tcount\tbytes\n")
        fh.write(f"record\tARCHIVED\t{len(recs)}\t{sum(s for _,_,s in recs)}\n")
        fh.write(f"heavy\tDROPPED\t0\t{dropped}\n")
        fh.write(f"link\tRECORDED\t{len(links)}\t0\n")
        for rel, _, sz in recs:
            fh.write(f"file\t{rel}\t{sz}\t\n")
    if links:
        open(lnk, "w").write("\n".join(links) + "\n")
    return tag, len(recs), sum(s for _, _, s in recs), dropped, os.path.getsize(tgz)

def verify(args):
    tag, group = args
    d = os.path.join(OUT, group)
    man = os.path.join(d, tag + ".manifest.tsv")
    # Mixed-codec archive: recompress_archive_20260904.py re-encodes tarballs
    # above its size floor to .tar.zst and leaves the rest .tar.gz.  A record is
    # a record in either codec, so verify whichever is present -- and if BOTH
    # exist that is a half-finished re-encode, which is an error, not a choice.
    gz, zst = os.path.join(d, tag + ".tar.gz"), os.path.join(d, tag + ".tar.zst")
    if os.path.exists(gz) and os.path.exists(zst):
        return (tag, "both .tar.gz and .tar.zst present (interrupted re-encode)")
    tgz = zst if os.path.exists(zst) else gz
    if not (os.path.exists(tgz) and os.path.exists(man)):
        return (tag, "missing")
    try:
        if tgz.endswith(".zst"):
            import subprocess as _sp
            raw = _sp.run(["zstd", "-dc", tgz], capture_output=True).stdout
            import io
            with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
                n = sum(1 for m in tf.getmembers() if m.isfile())
        else:
            with tarfile.open(tgz, "r:gz") as tf:
                n = sum(1 for m in tf.getmembers() if m.isfile())
    except Exception as e:
        return (tag, f"unreadable {e}")
    want = 0
    for i, ln in enumerate(open(man)):
        f = ln.rstrip("\n").split("\t")
        if i and f[0] == "record" and f[1] == "ARCHIVED": want = int(f[2])
    return None if n == want else (tag, f"tar {n} != manifest {want}")

def main():
    plan = json.load(open(os.path.join(STATE, "plan.json")))
    todo = [(t, plan["group"][t]) for t in plan["ARCHIVE"]]
    os.makedirs(OUT, exist_ok=True)
    print(f"archiving {len(todo)} arms -> {OUT}  ({JOBS}-way)")
    raw = gz = drop = 0
    with ProcessPoolExecutor(JOBS) as ex:
        for i, (tag, n, b, dp, g) in enumerate(ex.map(archive_one, todo, chunksize=8)):
            raw += b; gz += g; drop += dp
            if i % 500 == 0:
                print(f"  {i}/{len(todo)} …", flush=True)
    print(f"\nTOTAL archived {raw/2**30:.2f} GiB raw -> {gz/2**30:.2f} GiB gz"
          f"   dropped-if-removed {drop/2**30:.2f} GiB")

    print(f"\n=== INTEGRITY GATE (tar members == manifest record files, {len(todo)} arms) ===")
    bad = []
    with ProcessPoolExecutor(JOBS) as ex:
        for r in ex.map(verify, todo, chunksize=8):
            if r: bad.append(r)
    if bad:
        for t, w in bad[:10]: print(f"  BAD {t}: {w}")
        print(f"FAIL -- {len(bad)}/{len(todo)} arms mismatch"); sys.exit(1)
    print(f"PASS -- all {len(todo)}/{len(todo)} arms match")

if "--verify-only" in sys.argv:
    plan = json.load(open(os.path.join(STATE, "plan.json")))
    todo = [(t, plan["group"][t]) for t in plan["ARCHIVE"]]
    print(f"=== INTEGRITY GATE (tar members == manifest record files, {len(todo)} arms) ===")
    bad = []
    with ProcessPoolExecutor(JOBS) as ex:
        for r in ex.map(verify, todo, chunksize=8):
            if r: bad.append(r)
    for t, w in bad[:10]: print(f"  BAD {t}: {w}")
    print(f"{'FAIL' if bad else 'PASS'} -- {len(todo)-len(bad)}/{len(todo)} arms match")
    sys.exit(1 if bad else 0)

main()
