#!/usr/bin/env python3
"""Freeze the ~/tmp units of cleanup ROUND D (doc sbnd_xin/111) before they are removed.

  usage:  python3 archive_tmp_20260925.py tmp_tier_20260925.txt

Per unit: <tag>.manifest.tsv (rel path, size, SHA-256 -- every regular file),
<tag>.links.txt (symlinks recorded, never followed) and <tag>.tar.zst carrying the
small text layer (scripts, logs, notes <= 256 KiB).  Pins are shared libraries and
preps are per-candidate JSON payloads: hashed, never carried.  Hashes are cached
per inode, so the hardlinked pins of round A's dedup are read once.

Output: $RETIRE_OUT (default sbnd_xin/archive/records/cleanup-20260925/tmp).
EXIT STATUS IS THE CONTRACT.  Round B measured that "a manifest exists" cannot
tell a good freeze from a freeze into the wrong place, so the sweep removes
nothing unless this script exits 0, and this script exits non-zero if any unit
failed to write its manifest, links file or tar.  It refuses to write into an
output dir that already holds a manifest for the same tag (M13).
"""
import os, re, sys, stat, hashlib, tarfile, subprocess
from concurrent.futures import ProcessPoolExecutor

T     = "/home/xqian/tmp"
R     = "/home/xqian/toolkit-dev/wcp-porting-img"
OUT   = os.environ.get("RETIRE_OUT", f"{R}/sbnd/sbnd_xin/archive/records/cleanup-20260925/tmp")
JOBS  = int(os.environ.get("RETIRE_JOBS", "12"))
TEXT  = re.compile(r"\.(sh|py|md|txt|json|jsonnet|log|tsv|csv|rc|out|err|diff|patch|C|cxx|h)$|^[A-Z_]+$")

def tag_of(p):
    return os.path.relpath(p, T).replace("/", "__")

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def freeze(p):
    tag = tag_of(p)
    man, lnk, tzs = (os.path.join(OUT, tag + x) for x in (".manifest.tsv", ".links.txt", ".tar.zst"))
    if os.path.exists(man):
        return tag, f"REFUSED: {man} already exists (M13)", 1
    try:
        os.makedirs(OUT, exist_ok=True)       # fail BEFORE hashing gigabytes, not after
    except OSError as e:
        return tag, f"FAILED creating {OUT}: {e}", 1
    kind = "pin" if os.path.exists(os.path.join(p, "libWireCellClus.so")) else \
           "prep" if os.path.basename(p).startswith("prep_") else "other"
    recs, links, members, cache = [], [], [], {}
    walker = os.walk(p) if os.path.isdir(p) and not os.path.islink(p) else \
             [(os.path.dirname(p), [], [os.path.basename(p)])]
    base = p if os.path.isdir(p) and not os.path.islink(p) else os.path.dirname(p)
    for cur, subs, fs in walker:
        for s in list(subs):
            if os.path.islink(os.path.join(cur, s)):
                links.append(f"{os.path.relpath(os.path.join(cur, s), base)}\t{os.readlink(os.path.join(cur, s))}")
                subs.remove(s)
        for f in fs:
            fp = os.path.join(cur, f); rel = os.path.relpath(fp, base)
            if os.path.islink(fp):
                links.append(f"{rel}\t{os.readlink(fp)}"); continue
            try:
                st = os.lstat(fp)
            except OSError as e:
                return tag, f"FAILED reading {fp}: {e}", 1
            # NEVER OPEN A NON-REGULAR FILE.  Measured 2026-09-13: three stale
            # vgdb-pipe-* FIFOs at the claude-25225 root blocked the stubbed freeze
            # forever at unit 347/362 -- open() on a FIFO waits for a writer.
            if not stat.S_ISREG(st.st_mode):
                links.append(f"{rel}\tSPECIAL mode={oct(st.st_mode)}"); continue
            k = (st.st_dev, st.st_ino)
            try:
                if k not in cache: cache[k] = sha(fp)
            except OSError as e:
                return tag, f"FAILED reading {fp}: {e}", 1
            recs.append(f"{rel}\t{st.st_size}\t{cache[k]}")
            if kind == "other" and st.st_size <= 256 << 10 and TEXT.search(f):
                members.append((fp, rel))
    try:
        os.makedirs(OUT, exist_ok=True)
        open(man, "w").write("\n".join(sorted(recs)) + "\n")
        open(lnk, "w").write("\n".join(sorted(links)) + "\n")
        tmp = tzs[:-4] + ".tar"
        with tarfile.open(tmp, "w") as tf:
            for fp, rel in members: tf.add(fp, arcname=os.path.join(tag, rel))
        rc = subprocess.run(["zstd", "-q", "-10", "--rm", "-f", tmp, "-o", tzs]).returncode
        if rc != 0 or not os.path.getsize(man) and recs:
            return tag, f"FAILED: zstd rc={rc}", 1
    except OSError as e:
        return tag, f"FAILED writing: {e}", 1
    return tag, f"{kind:<5} files {len(recs):>5} carried {len(members):>4} links {len(links):>4}", 0

if __name__ == "__main__":
    tier = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: archive_tmp_20260925.py <tier file>")
    units = [l.strip() for l in open(tier) if l.strip()]
    bad = [u for u in units if not u.startswith(T + "/") or not os.path.lexists(u)]
    if bad: sys.exit(f"REFUSING: {len(bad)} unit(s) outside ~/tmp or missing, e.g. {bad[0]}")
    fails = 0
    with ProcessPoolExecutor(JOBS) as ex:
        for i, (tag, msg, rc) in enumerate(ex.map(freeze, units), 1):
            fails += rc
            print(f"  {i:>4}/{len(units)}  {tag:<60} {msg}")
    print(f"\nfroze {len(units) - fails}/{len(units)} units into {OUT}; failures {fails}")
    sys.exit(1 if fails else 0)
