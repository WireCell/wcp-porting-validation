#!/usr/bin/env python3
"""Cleanup ROUND K, second pass (2026-09-25, doc sbnd_xin/125 sec 9): zstd the per-event LOGS and
COMPILED CONFIGS of the kept sbnd arms.  DRY RUN unless CONFIRM=yes.

  usage:  python3 compress_logs_20260925.py             # plan
          CONFIRM=yes python3 compress_logs_20260925.py   # freeze sha256, compress, verify, unlink
          undo:   ./uncompress_logs_20260925.sh [arm-glob ...]

Owner, 2026-09-25 ("Compress kept-arm logs (~11 GiB)").  Nothing is deleted: each file becomes
<name>.zst next to where it was, its sha256 is frozen first, and the original is unlinked only after
the .zst decompresses to that exact hash.  mtime is kept (zstd copies it).

NAMES (measured 2026-09-25: 11.9 GiB, configs x435, logs x19):
    stdout.log  wct_pr_evt<N>.log  ql.stdout  wct_ql.log  img.stdout  wct_img.log  .wct-cfg-evt<N>.json
NOT calib-pr-evt*.json (M13 calib dumps, and pr150csp3bw's stated reason to be kept).

HELD, never compressed:
  * the live pdvd/119 SBND gates (work-nuecc48-d119*, work-mcp10-m66d119*);
  * the LOGS (not the configs) of the sentinel arms -- pr127_sentinels.py greps pr_evt<N>/*.log in
    them: pr150s0 (reference), d123lgoppr + d123lgflippr (production, the re-baseline's arms), the
    witnesses sent97 and sent150;
  * any file with a second hard link, any file a symlink anywhere in the three trees resolves onto,
    any file a process holds open, and any file written in the last 2 h.

Readers that must decompress first (zstdcat, or the undo script): scripts/d123/{r3_evidence,
r3_ql_compare,r6_rescue_ruling}.py, scripts/d124/pr_null_cmp.py, and the closed pr/8x-14x census
scripts.  archive_records_20260925.py (and its forks) carry *.log.zst / .wct-*.json.zst as record
layer, not heavy (the RECZST rule added with this file).

Exit codes: 3 the plan moved | 5 a .zst failed its hash check (that original is kept) | 16 broken
symlinks rose
"""
import os, re, sys, time, hashlib, subprocess, concurrent.futures as cf

R      = "/home/xqian/toolkit-dev/wcp-porting-img"
SX     = f"{R}/sbnd/sbnd_xin"
HERE   = os.path.dirname(os.path.abspath(__file__))
STAMP  = "20260925"
REC    = f"{SX}/archive/records/cleanup-{STAMP}/logs-zst"
CONFIRM = os.environ.get("CONFIRM", "no") == "yes"
NOW    = time.time()
NAME   = re.compile(r"^(stdout\.log|wct_pr_evt\d+\.log|ql\.stdout|wct_ql\.log|img\.stdout|wct_img\.log|\.wct-cfg-evt\d+\.json)$")
LOGNAME = re.compile(r"\.log$")
LIVE   = ("work-nuecc48-d119", "work-mcp10-m66d119")
SENT   = re.compile(r"^work-.*-(pr150s0|d123lgoppr|d123lgflippr)$|^work-sent(97|150)-")
TREES  = [f"{R}/sbnd/sbnd_xin", f"{R}/pdvd/work", f"{R}/pdhd/work"]

def broken():
    return {t: int(subprocess.run(f"find {t} -xtype l 2>/dev/null | wc -l", shell=True,
                                  capture_output=True, text=True).stdout) for t in TREES}

# ---- plan ---------------------------------------------------------------------------------
arms = sorted(a for a in os.listdir(SX) if a.startswith("work-") and os.path.isdir(f"{SX}/{a}")
              and not os.path.islink(f"{SX}/{a}") and not a.startswith(LIVE))
targets, held = [], {"sentinel log": 0, "hardlinked": 0, "link target": 0, "open": 0, "recent": 0}
for a in arms:
    for cur, subs, files in os.walk(f"{SX}/{a}"):
        subs[:] = [s for s in subs if not os.path.islink(os.path.join(cur, s))]
        for f in files:
            if not NAME.match(f): continue
            p = os.path.join(cur, f)
            if os.path.islink(p): continue
            st = os.lstat(p)
            if SENT.match(a) and LOGNAME.search(f): held["sentinel log"] += 1; continue
            if st.st_nlink > 1: held["hardlinked"] += 1; continue
            if NOW - st.st_mtime < 2 * 3600: held["recent"] += 1; continue
            targets.append((p, st.st_size))
tset = {p for p, _ in targets}
# anything a symlink resolves onto stays (it would break the link)
for t in TREES + ["/home/xqian/tmp"]:
    for cur, subs, files in os.walk(t):
        for n in subs + files:
            q = os.path.join(cur, n)
            if os.path.islink(q):
                rp = os.path.realpath(q)
                if rp in tset: tset.discard(rp); held["link target"] += 1
for pid in os.listdir("/proc"):
    if not pid.isdigit(): continue
    try:
        for fd in os.listdir(f"/proc/{pid}/fd"):
            try:
                rp = os.readlink(f"/proc/{pid}/fd/{fd}")
                if rp in tset: tset.discard(rp); held["open"] += 1
            except OSError: pass
    except OSError: pass
targets = [(p, s) for p, s in targets if p in tset]
tot = sum(s for _, s in targets)
byname = {}
for p, s in targets:
    k = re.sub(r"\d+", "N", os.path.basename(p)); byname[k] = byname.get(k, 0) + s
print(f"arms {len(arms)} | targets {len(targets)} files, {tot / 2**30:.2f} GiB | held {held}")
for k, v in sorted(byname.items(), key=lambda x: -x[1]): print(f"   {v / 2**30:6.2f} GiB  {k}")
lst = f"{HERE}/files_logs_zst_{STAMP}.txt"
if not CONFIRM:
    open(lst, "w").write("\n".join(p for p, _ in targets) + "\n")
    print(f"DRY RUN -- nothing compressed.  list: {lst}"); sys.exit(0)
plan = [l.strip() for l in open(lst) if l.strip()]
if plan != [p for p, _ in targets]:
    print("REFUSING: the target list moved between plan and confirm -- re-plan"); sys.exit(3)

# ---- freeze, compress, verify, unlink -----------------------------------------------------
b0 = broken()
os.makedirs(REC, exist_ok=False)
def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def one(p):
    raw = open(p, "rb").read()
    d = sha_bytes(raw)
    z = p + ".zst"
    if os.path.exists(z): return (p, len(raw), d, "exists")
    rc = subprocess.call(["zstd", "-q", "-6", p, "-o", z])
    if rc != 0: return (p, len(raw), d, "zstd rc")
    back = subprocess.run(["zstd", "-q", "-d", "-c", z], capture_output=True).stdout
    if sha_bytes(back) != d:
        os.unlink(z); return (p, len(raw), d, "hash mismatch")
    st = os.stat(p); os.utime(z, ns=(st.st_atime_ns, st.st_mtime_ns))
    os.unlink(p)
    return (p, len(raw), d, "ok")
with cf.ThreadPoolExecutor(24) as ex:
    res = list(ex.map(one, plan))
with open(f"{REC}/logs-zst.manifest.tsv", "w") as m:
    m.write("#relpath\tsize\tsha256\tstatus\n")
    for p, s, d, st in res: m.write(f"{os.path.relpath(p, R)}\t{s}\t{d}\t{st}\n")
bad = [r for r in res if r[3] != "ok"]
zs = sum(os.path.getsize(p + ".zst") for p, _, _, st in res if st == "ok")
print(f"== compressed {len(res) - len(bad)}/{len(res)} files: {sum(r[1] for r in res) / 2**30:.2f} GiB -> {zs / 2**30:.2f} GiB; "
      f"manifest {REC}/logs-zst.manifest.tsv")
b1 = broken()
for t in TREES: print(f"== {os.path.relpath(t, R)} broken symlinks: {b0[t]} -> {b1[t]}")
if any(b1[t] > b0[t] for t in TREES): print("REFUSING (after the fact): broken symlinks rose"); sys.exit(16)
if bad:
    print(f"   {len(bad)} kept uncompressed, e.g. {bad[0]}"); sys.exit(5)
