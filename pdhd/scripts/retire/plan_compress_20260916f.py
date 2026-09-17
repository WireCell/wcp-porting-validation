#!/usr/bin/env python3
"""Cleanup ROUND G 2026-09-16 (doc sbnd_xin/111 sec 17): COMPRESS, not delete, cold file classes inside
kept pdvd/work arms.  Removes nothing; writes the target list and the SHA-256 manifest.

OWNER, 2026-09-16: "The pdvd/work directory is still sizable, I wonder instead of deleting things, we can
compress some not useful information, and save disk further? Some of the older information may not be
needed immediately?"  Asked per class, the owner chose all four offered:
  csv    gpu_mem_*.csv                      (runtime traces; only the SP runner that writes them names them)
  log    wct_*.log                          (run logs: wct_pr / wct_clus / wct_img / wct_nfsp*)
  calib  calib-pr-evt*.json, calib-evt*.json (the PR and clustering calib dumps)
in the COLD arms only: the pdvd hand-scan sources (scan_arms_20260916e.json) and the named substrate arms.
NOT in scope: production (d103vflip d103vprod1 q29flip q29stm p100flip pvdimg p98von), OPEN p101q, the
doc-113 session's arms and holds (d113vbase d113vnone d111vst d103v0 d103v1 d101vnew), the bare dirs.

Unit: one file -> <file>.zst (zstd -19), original removed only after the decompressed stream's SHA-256
equals the manifest (compress_files_20260916f.sh).  restore_compress_20260916f.py reverses it.

  usage:  python3 plan_compress_20260916f.py          # list + gates
          python3 plan_compress_20260916f.py --hash   # also writes the SHA-256 manifest (the freeze)

Gates (exit 1 on any failure):
  C1  every target is a regular file, not a symlink, st_nlink == 1.
  C2  no symlink under wcp-porting-img/{pdvd,pdhd,sbnd/sbnd_xin,qlport} or ~/tmp (depth 4) resolves, by
      inode, onto a target (a compressed target would leave it dangling).
  C3  no process holds a target open, and no wire-cell job names pdvd.
  C4  no target was written in the last hour, and no <target>.zst exists yet.
  C5  the scope excludes every production / OPEN / held family (asserted, not assumed).
"""
import os, re, sys, json, time, hashlib, subprocess
from concurrent.futures import ProcessPoolExecutor

R     = "/home/xqian/toolkit-dev/wcp-porting-img"
W     = f"{R}/pdvd/work"
HERE  = os.path.dirname(os.path.abspath(__file__))
STAMP = "20260916f"
OUT   = os.environ.get("RETIRE_OUT", f"{R}/sbnd/sbnd_xin/archive/records/cleanup-{STAMP}")
SUF   = ('.' + os.environ["PLAN_SUFFIX"]) if os.environ.get("PLAN_SUFFIX") else ""
ARM   = re.compile(r"^\d{6}_\d+_(.+)$")
HOLD  = {"d103vflip", "d103vprod1", "q29flip", "q29stm", "p100flip", "pvdimg", "p98von", "p101q",
         "d113vbase", "d113vnone", "d111vst", "d103v0", "d103v1", "d101vnew"}
SUBSTRATE = ["d27fresh", "keep", "d51vclus", "d41prov", "d39r2prov"]
CLASSES = [("calib", re.compile(r"^calib(-pr)?-evt\d+\.json$")),
           ("log",   re.compile(r"^wct_.*\.log$")),
           ("csv",   re.compile(r"^gpu_mem_.*\.csv$"))]
fails = []

def check(n, ok, msg):
    print(f"  {'PASS' if ok else 'FAIL'}  {n}: {msg}")
    if not ok: fails.append(n)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    st = os.lstat(p)
    return p, st.st_size, int(st.st_mtime), h.hexdigest()

def main():
    scan = json.load(open(os.path.join(HERE, "scan_arms_20260916e.json")))["pdvd"]
    fams = sorted((set(scan) | set(SUBSTRATE)) - HOLD)
    check("C5", not (set(fams) & HOLD), f"{len(fams)} cold families, none held: {' '.join(fams)}")
    targets, bycls = [], {}
    for d in sorted(os.listdir(W)):
        m = ARM.match(d)
        p = os.path.join(W, d)
        if not m or m.group(1) not in fams or os.path.islink(p) or not os.path.isdir(p): continue
        for cur, subs, fs in os.walk(p):
            subs[:] = [s for s in subs if not os.path.islink(os.path.join(cur, s))]
            for fn in fs:
                for k, pat in CLASSES:
                    if pat.match(fn):
                        q = os.path.join(cur, fn)
                        if os.path.islink(q): continue
                        targets.append(q); bycls.setdefault(k, [0, 0]); bycls[k][0] += 1
                        bycls[k][1] += os.lstat(q).st_size
                        break
    targets.sort()
    for k, (n, b) in sorted(bycls.items()):
        print(f"  {k:6s} {n:6d} files {b / 2**30:7.2f} GiB")
    print(f"  total  {len(targets):6d} files {sum(v[1] for v in bycls.values()) / 2**30:7.2f} GiB")
    # C1
    bad = [q for q in targets if not os.path.isfile(q) or os.lstat(q).st_nlink != 1]
    check("C1", not bad, f"regular single-name files ({len(bad)} not; {bad[:1]})")
    ino = {(os.lstat(q).st_dev, os.lstat(q).st_ino): q for q in targets}
    # C2
    stray = []
    for root in (f"{R}/pdvd", f"{R}/pdhd", f"{R}/sbnd/sbnd_xin", f"{R}/qlport", "/home/xqian/tmp"):
        for cur, subs, fs in os.walk(root):
            if root == "/home/xqian/tmp" and cur[len(root):].count("/") >= 4: subs[:] = []
            for e in fs + subs:
                q = os.path.join(cur, e)
                if not os.path.islink(q): continue
                try: st = os.stat(q)
                except OSError: continue
                if (st.st_dev, st.st_ino) in ino: stray.append((q, ino[(st.st_dev, st.st_ino)]))
            subs[:] = [s for s in subs if not os.path.islink(os.path.join(cur, s)) and s != "archive"]
    check("C2", not stray, f"no symlink resolves onto a target ({len(stray)}; {stray[:2]})")
    # C3
    held = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try: fds = os.listdir(f"/proc/{pid}/fd")
        except OSError: continue
        for fd in fds:
            try: st = os.stat(f"/proc/{pid}/fd/{fd}")
            except OSError: continue
            if (st.st_dev, st.st_ino) in ino: held.append((pid, ino[(st.st_dev, st.st_ino)]))
    ps = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True).stdout.splitlines()
    busy = [l for l in ps if re.search(r"wire-cell|run_pr_evt|run_clus_evt|run_img_evt|wcsonnet", l)
            and "/pdvd" in l and "grep" not in l]
    check("C3", not held and not busy, f"no open target ({held[:1] or 'none'}), {len(busy)} pdvd reco procs")
    # C4
    now = time.time()
    young = [q for q in targets if now - os.lstat(q).st_mtime < 3600]
    exists = [q for q in targets if os.path.lexists(q + ".zst")]
    check("C4", not young and not exists, f"none written in the last hour ({len(young)}), no .zst yet ({len(exists)})")

    lst = os.path.join(HERE, f"files_compress_pdvd_{STAMP}{SUF}.txt")
    open(lst, "w").write("".join(q + "\n" for q in targets))
    print(f"  list: {lst}")
    if "--hash" in sys.argv and not fails:
        os.makedirs(OUT, exist_ok=True)
        man = os.path.join(OUT, "pdvd-compress.manifest.tsv")
        if os.path.exists(man): sys.exit(f"REFUSING: {man} exists (M13)")
        with ProcessPoolExecutor(16) as ex:
            rows = sorted(ex.map(sha, targets, chunksize=32))
        open(man, "w").write("".join(f"{p}\t{s}\t{m}\t{h}\n" for p, s, m, h in rows))
        print(f"  froze {len(rows)} manifest rows (path, size, mtime, sha256) -> {man}")
    print(f"gate failures: {fails or 'NONE'}")
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
