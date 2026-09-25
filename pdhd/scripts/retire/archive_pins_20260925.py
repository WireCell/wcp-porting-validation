#!/usr/bin/env python3
"""Cleanup ROUND K, second pass (2026-09-25, doc sbnd_xin/125 sec 9): ARCHIVE the cold lib pins.
DRY RUN unless CONFIRM=yes.

  usage:  python3 archive_pins_20260925.py            # plan: cold list + set-relative bytes
          CONFIRM=yes python3 archive_pins_20260925.py  # freeze, archive, verify, then unlink
          restore: ./restore_pins_20260925.sh [pin ...]   (all pins when no argument)

Owner, 2026-09-25 ("Archive cold pins (~26 GiB)").  A pin is kept by the census because an arm that
survives ran on it; rerunning that arm needs the pin, READING the arm does not.  So a pin no production
arm, no live peer and no running process uses can live in one archive instead of on disk, and come back
byte-for-byte when an arm is rerun.  Nothing is lost: every file's sha256 is frozen first, the archive is
extracted in full and compared against that manifest BEFORE any original is unlinked.

COLD = every pin the census KEEPs (tmp_census_20260925.json, kind "pin"), minus
  * HOT: the PERMANENT production pins, d123-libpin (the pin d123base -- the SBND substrate -- was
    produced on, named by 864 files), and every live prefix (d117 d118 d119 wcfm d125flip claude-);
  * any pin a process has mapped (/proc/*/maps) or has its cwd inside;
  * any pin written in the last 2 h.

ONE archive, zstd --long=31: identical libraries across pins compress away (measured 2026-09-25:
35.33 GiB of distinct inodes -> 6.57 GiB).  tar keeps hardlinks, symlinks and modes.  So the archive is
chosen INSTEAD of dedup for these roots, and dedup_pins_20260925.py runs after, on what stays.

FREED BYTES ARE SET-RELATIVE: an inode frees its blocks only if every one of its st_nlink names is
inside the cold set (2.29 GiB are shared with hot pins and stay on disk under the hot name).

Exit codes: 2 no census json | 3 a cold pin moved/vanished since the plan | 4 a guard fires
            5 archive test failed | 6 extracted tree differs from the manifest | 7 unlink failure
"""
import os, sys, json, time, hashlib, subprocess, collections, concurrent.futures as cf

T      = "/home/xqian/tmp"
R      = "/home/xqian/toolkit-dev/wcp-porting-img"
HERE   = os.path.dirname(os.path.abspath(__file__))
STAMP  = "20260925"
REC    = f"{R}/sbnd/sbnd_xin/archive/records/cleanup-{STAMP}/pins-cold"
ADIR   = f"{T}/pin-archive-{STAMP}"
ARCH   = f"{ADIR}/cold-pins-{STAMP}.tar.zst"
CONFIRM = os.environ.get("CONFIRM", "no") == "yes"
NOW    = time.time()
HOT    = {"d102/libpin_d102", "pdhdstm_libpin", "d123-libpin-r6", "p35/libpin_prod", "d123-libpin"}
LIVE   = ("d117", "d118", "d119", "wcfm", "d125flip", "claude-")

cj = f"{HERE}/tmp_census_{STAMP}.json"
if not os.path.exists(cj):
    print(f"REFUSING: no census {cj}"); sys.exit(2)
cold = []
for u in json.load(open(cj)):
    if u["kind"] != "pin" or u["verdict"] != "KEEP": continue
    rel = u["path"][len(T) + 1:]
    if rel in HOT or rel.startswith(LIVE): continue
    cold.append(rel)
cold.sort()

# ---- guards -------------------------------------------------------------------------------
bad = []
mapped, cwds = set(), set()
for pid in os.listdir("/proc"):
    if not pid.isdigit(): continue
    try:
        for line in open(f"/proc/{pid}/maps"):
            if T in line: mapped.add(line.split()[-1])
    except OSError: pass
    try: cwds.add(os.readlink(f"/proc/{pid}/cwd"))
    except OSError: pass
for rel in cold:
    p = f"{T}/{rel}"
    if os.path.islink(p) or not os.path.isdir(p): bad.append((rel, "not a real dir")); continue
    if any(m.startswith(p + "/") for m in mapped): bad.append((rel, "mapped by a process"))
    if any(c == p or c.startswith(p + "/") for c in cwds): bad.append((rel, "a process cwd inside"))
    newest = max((os.lstat(os.path.join(r, f)).st_mtime for r, _, fs in os.walk(p) for f in fs), default=0)
    if NOW - newest < 2 * 3600: bad.append((rel, "written in the last 2 h"))
if bad:
    for b in bad: print("REFUSING:", *b)
    sys.exit(4 if all(os.path.isdir(f"{T}/{b[0]}") for b in bad) else 3)

# ---- inventory (set-relative) -------------------------------------------------------------
rows, inodes = [], {}
for rel in cold:
    for r, ds, fs in os.walk(f"{T}/{rel}"):
        for d in ds:
            q = os.path.join(r, d)
            if os.path.islink(q): rows.append(("l", q, os.readlink(q)))
        for f in fs:
            q = os.path.join(r, f)
            st = os.lstat(q)
            if os.path.islink(q): rows.append(("l", q, os.readlink(q))); continue
            rows.append(("f", q, st))
            e = inodes.setdefault(st.st_ino, [st.st_nlink, st.st_blocks * 512, 0])
            e[2] += 1
freed = sum(b for n, b, c in inodes.values() if c == n)
held  = sum(b for n, b, c in inodes.values() if c < n)
nfile = sum(1 for x in rows if x[0] == "f")
print(f"cold pins {len(cold)} | {nfile} files, {len(rows) - nfile} symlinks, {len(inodes)} inodes")
print(f"FREED (every link inside the cold set) {freed / 2**30:.2f} GiB | shared with a kept name {held / 2**30:.2f} GiB")
for rel in cold: print("   COLD", rel)
open(f"{HERE}/pins_cold_{STAMP}.txt", "w").write("\n".join(cold) + "\n")
if not CONFIRM:
    print("DRY RUN -- nothing archived or removed.  Re-run with CONFIRM=yes."); sys.exit(0)

# ---- 1. freeze the manifest (sha256 per inode) -------------------------------------------
os.makedirs(REC, exist_ok=False)          # a NEW record dir; never write into an existing one (M13)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
first = {}
for k, q, x in rows:
    if k == "f" and x.st_ino not in first: first[x.st_ino] = q
with cf.ThreadPoolExecutor(16) as ex:
    digest = dict(zip(first, ex.map(sha, first.values())))
with open(f"{REC}/cold-pins.manifest.tsv", "w") as m:
    m.write("#relpath\ttype\tsize_or_target\tsha256\tmode\n")
    for k, q, x in rows:
        rel = os.path.relpath(q, T)
        if k == "l": m.write(f"{rel}\tl\t{x}\t-\t-\n")
        else: m.write(f"{rel}\tf\t{x.st_size}\t{digest[x.st_ino]}\t{oct(x.st_mode & 0o7777)}\n")
open(f"{REC}/cold-pins.list", "w").write("\n".join(cold) + "\n")
print(f"== froze {len(rows)} rows -> {REC}/cold-pins.manifest.tsv")

# ---- 2. archive --------------------------------------------------------------------------
os.makedirs(ADIR, exist_ok=True)
if os.path.exists(ARCH): print(f"REFUSING: {ARCH} exists"); sys.exit(5)
lst = f"{ADIR}/cold-pins-{STAMP}.list"
open(lst, "w").write("\n".join(cold) + "\n")
rc = subprocess.call(f"tar -C {T} -cf - -T {lst} | zstd -q --long=31 -T24 -3 -o {ARCH}", shell=True)
if rc != 0 or subprocess.call(["zstd", "-q", "-t", "--long=31", ARCH]) != 0:
    print("REFUSING: archive creation/test failed"); sys.exit(5)
print(f"== archive {ARCH}: {os.path.getsize(ARCH) / 2**30:.2f} GiB, zstd -t OK")

# ---- 3. full extract to a staging tree, compare every manifest row ------------------------
V = f"{ADIR}/verify"
os.makedirs(V)
if subprocess.call(f"zstd -q -d --long=31 -c {ARCH} | tar -C {V} -xpf -", shell=True) != 0:
    print("REFUSING: extract failed"); sys.exit(6)
vfirst, bad = {}, []
for k, q, x in rows:
    v = os.path.join(V, os.path.relpath(q, T))
    if k == "l":
        if not os.path.islink(v) or os.readlink(v) != x: bad.append(v)
    elif not os.path.isfile(v) or os.path.islink(v) or os.path.getsize(v) != x.st_size: bad.append(v)
    else: vfirst.setdefault(os.lstat(v).st_ino, (v, digest[x.st_ino]))
with cf.ThreadPoolExecutor(16) as ex:
    for (v, want), got in zip(vfirst.values(), ex.map(sha, [a for a, _ in vfirst.values()])):
        if got != want: bad.append(v)
if bad:
    print(f"REFUSING: {len(bad)} extracted rows differ, e.g. {bad[0]}; originals untouched"); sys.exit(6)
print(f"== verify: {len(rows)} rows, {len(vfirst)} extracted inodes sha256-identical to the manifest")
for r, ds, fs in os.walk(V, topdown=False):              # remove the staging copy (ours)
    for f in fs: os.unlink(os.path.join(r, f))
    for d in ds:
        q = os.path.join(r, d)
        os.unlink(q) if os.path.islink(q) else os.rmdir(q)
os.rmdir(V)

# ---- 4. release the originals, leave a stub that says where they went ---------------------
fail = 0
for rel in cold:
    p = f"{T}/{rel}"
    for r, ds, fs in os.walk(p, topdown=False):
        for f in fs:
            try: os.unlink(os.path.join(r, f))
            except OSError as e: print("   FAILED", e); fail += 1
        for d in ds:
            q = os.path.join(r, d)
            try: os.unlink(q) if os.path.islink(q) else os.rmdir(q)
            except OSError as e: print("   FAILED", e); fail += 1
    try: os.rmdir(p)
    except OSError as e: print("   FAILED", e); fail += 1
    open(p + ".ARCHIVED", "w").write(
        f"{rel} was archived {STAMP} (doc sbnd_xin/125 sec 9) into {ARCH}\n"
        f"restore: {HERE}/restore_pins_{STAMP}.sh {rel}\n"
        f"manifest: {REC}/cold-pins.manifest.tsv\n")
print(f"== released {len(cold)} cold pins, failures {fail}; stubs <pin>.ARCHIVED written")
sys.exit(7 if fail else 0)
