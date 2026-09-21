#!/usr/bin/env python3
"""Cleanup ROUND J 2026-09-21b (doc sbnd_xin/122): COMPRESS, not delete, cold file classes inside
kept pdvd/work and pdhd/work arms.  Removes nothing; writes the target lists and the SHA-256 manifest.

OWNER, 2026-09-21, picking from a measured menu: "Compress cold non-production calib+logs".  This
extends round G (which did pdvd only) to pdhd -- doc 111 sec 12 item 8's recorded open item,
"pdhd/work holds the same cold arm classes as pdvd's; round G's compression would apply, never
measured".  Now measured, at round G's scope: pdhd 3223 files / 10.15 GiB in scope (est save 8.79),
pdvd 5466 files / 4.88 GiB (est save 4.21) -- round G took the pdvd calib mass already, so what is
left there is mostly logs.  Total estimated saving 13.0 GiB.

MEASURED RATIOS, on real files from these trees rather than inherited from round G:
  calib-*.json  6.9x (pdvd) / 6.1x (pdhd)      logs  19.5x / 25.0x
  pctree .tar.gz  1.2-1.3x  -> NOT a class here: already compressed, and 17 GiB of restore burden
  for ~3 GiB would be a bad trade.

sbnd IS DELIBERATELY OUT OF SCOPE, and the reason is the undo path, not the bytes (2.33 GiB):
restore_compress's family filter keys on `/<run6>_<idx>_<arm>/`, which sbnd's `work-<sample>-<arm>`
directories do not match, so an sbnd family could only ever be restored with --all.  A compression
pass whose undo is coarser than its do is not worth 2.33 GiB.

WHY THE SCOPE IS COLD AND NON-PRODUCTION, stated because the number looks small: 243 scripts in
this repo read calib dumps or pctrees BY NAME and only 3 tolerate .zst.  Round G's lesson is that
most of them glob-loop and skip a missing file SILENTLY -- no error, just absent data.  So the
warm/production 60 GiB of the same classes stays uncompressed on purpose; what is left is the
genuinely cold tail.  RESTORE BEFORE ANY NAME-BASED READER TOUCHES A COMPRESSED FAMILY.

THE HOLD SETS ARE IMPORTED FROM plan_20260921b.py, NOT RETYPED.  Round J found two separate bugs
caused by a list existing in two places (sweep_tmp's held prefixes vs tmp_census's LIVE_TOP, and
three stale PROTECTED "LATEST PRODUCTION" lines), so this file reads the round's own TREES config
and its keep_<tree> output instead of keeping a private copy.

  usage:  python3 plan_compress_20260921b.py          # lists + gates
          python3 plan_compress_20260921b.py --hash   # also write the SHA-256 manifest (the freeze)
          python3 plan_compress_20260921b.py --negctl # prove C5 fires: inject a production arm's file

Gates (exit 1 on any failure):
  C1  every target is a regular file, not a symlink, st_nlink == 1.
  C2  no symlink under pdvd/work, pdhd/work, sbnd_xin or ~/tmp (depth 4) resolves, by inode, onto a
      target (compressing it would leave the link dangling).
  C3  no process holds a target open.
  C4  no target was written in the last 7 days, and no <target>.zst exists yet.
  C5  the scope excludes every PRODUCTION and OPEN family -- asserted against plan_20260921b.py's
      own config, not assumed, and proven to fire by --negctl.  It deliberately does NOT exclude
      substrate or hand-scan sources: see the note at the hold set.  PROTECTED is not a hold for
      compression, because it protects an arm's EXISTENCE, not its file encoding.
  C6  every target's arm still appears in this round's keep_<tree> list (we only ever compress
      inside something we are keeping; a releasing arm's files must not be touched).
"""
import os, re, sys, time, hashlib, subprocess
from concurrent.futures import ProcessPoolExecutor

R     = "/home/xqian/toolkit-dev/wcp-porting-img"
HERE  = os.path.dirname(os.path.abspath(__file__))
STAMP = "20260921b"
OUT   = os.environ.get("RETIRE_OUT", f"{R}/sbnd/sbnd_xin/archive/records/cleanup-{STAMP}")
SUF   = ('.' + os.environ["PLAN_SUFFIX"]) if os.environ.get("PLAN_SUFFIX") else ""
COLD  = 7 * 86400
ARM   = re.compile(r"^(\d{6})_(\d+)(?:_(.+))?$")

CLASSES = [("calib", re.compile(r"^calib(-pr)?-evt\d+\.json$")),
           ("log",   re.compile(r"^wct_.*\.log$")),
           ("stdout",re.compile(r"^(img|ql|clus|pr|dump)\.stdout$")),
           ("csv",   re.compile(r"^(gpu_mem|pr_rss|clus_rss)_.*\.csv$"))]
RATIO = {"calib": 6.5, "log": 20.0, "stdout": 20.0, "csv": 25.0}

def load_round_config():
    """Read plan_20260921b.py's TREES dict and PROTECTED union -- the single source of truth."""
    src = open(os.path.join(HERE, "plan_20260921b.py")).read()
    # slice AFTER protected_lines() but BEFORE live_tokens(), which would run git.  Note the
    # order in that file is TREES -> load_citations -> protected_lines -> DOCROOTS -> live_tokens,
    # so cutting at load_citations (the obvious guess) silently loses protected_lines.
    body = src[:src.index("DOCROOTS")]
    ns = {"__name__": "cfgimport", "__file__": os.path.join(HERE, "plan_20260921b.py")}
    exec(compile(body, "plan_cfg", "exec"), ns)
    prot_names, prot_pref = ns["protected_lines"]()
    return ns["TREES"], prot_names, prot_pref

TREES, PROT_N, PROT_P = load_round_config()
fails = []
def check(n, ok, msg):
    print(f"  {'PASS' if ok else 'FAIL'}  {n}: {msg}")
    if not ok: fails.append(n)

def kept_arms(tree):
    p = os.path.join(HERE, f"keep_{tree}_{STAMP}.txt")
    if not os.path.exists(p): sys.exit(f"missing {p} -- run plan_{STAMP}.py first")
    out = set()
    for line in open(p):
        d = os.path.basename(line.strip())
        m = ARM.match(d)
        if m: out.add(m.group(3) or "(bare)")
    return out

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return p, os.lstat(p).st_size, int(os.lstat(p).st_mtime), h.hexdigest()

def main():
    negctl = "--negctl" in sys.argv
    NOW = time.time()
    picks, holdsets = {}, {}
    for tree in ("pdvd", "pdhd"):
        cfg = TREES[tree]
        # ROUND G'S SCOPE, reproduced deliberately after a narrower first attempt yielded
        # 0.28 GiB.  The hold is production + OPEN + the bare substrate dirs, and NOT the
        # substrate or hand-scan-source arms -- round G compressed exactly those, on purpose,
        # "in the COLD arms only: the pdvd hand-scan sources and the named substrate arms".
        # THE DISTINCTION THAT MAKES THAT SAFE: PROTECTED protects an arm's EXISTENCE, not its
        # file encoding.  A compressed arm is still there, still complete, and restorable to
        # byte-identical by restore_compress; nothing in M13 or in PROTECTED's own wording is
        # about encoding.  So PROTECTED is not consulted as a hold here -- which is why C5
        # checks production/OPEN membership and no longer checks PROT_N/PROT_P.
        # What still guards it: C4 (the whole ARM cold for 7+ days, so no live reader), C2 (no
        # symlink resolves onto a target), and the restore script.
        hold = set(cfg["production"]) | {"(bare)"}
        holdsets[tree] = hold
        KEPT = kept_arms(tree)
        sel, byclass = [], {}
        for d in sorted(os.listdir(cfg["work"])):
            p = os.path.join(cfg["work"], d)
            if os.path.islink(p) or not os.path.isdir(p): continue
            m = ARM.match(d)
            if not m: continue
            arm = m.group(3) or "(bare)"
            if arm in hold or arm.startswith(tuple(cfg["open_prefix"]) or ("\0",)): continue
            if arm not in KEPT: continue            # C6: only inside something we keep
            newest = 0; cand = []
            for cur, _, fs in os.walk(p):
                for f in fs:
                    q = os.path.join(cur, f)
                    if os.path.islink(q): continue
                    try: st = os.lstat(q)
                    except OSError: continue
                    newest = max(newest, st.st_mtime)
                    for k, pat in CLASSES:
                        if pat.match(f): cand.append((k, q, st.st_size)); break
            if NOW - newest < COLD: continue        # C4: the ARM is cold, not just the file
            for k, q, s in cand:
                sel.append(q); byclass[k] = byclass.get(k, [0, 0]); byclass[k][0] += 1; byclass[k][1] += s
        picks[tree] = sel
        tot = sum(v[1] for v in byclass.values())
        est = sum(v[1] - v[1] / RATIO[k] for k, v in byclass.items())
        print(f"{tree}: {len(sel)} files, {tot/2**30:.2f} GiB, est save {est/2**30:.2f} GiB")
        for k, v in sorted(byclass.items()):
            print(f"     {k:7s} {v[0]:6d} files {v[1]/2**30:6.2f} GiB  (ratio {RATIO[k]}x)")

    if negctl:
        prod = TREES["pdvd"]["production"][0]
        inj = None
        for cur, _, fs in os.walk(TREES["pdvd"]["work"]):
            if f"_{prod}" not in cur: continue
            for f in fs:
                if CLASSES[0][1].match(f): inj = os.path.join(cur, f); break
            if inj: break
        if inj:
            picks["pdvd"] = picks["pdvd"] + [inj]
            print(f"  NEGATIVE CONTROL: injected a {prod} (production) file -- C5 must FAIL\n     {inj}")

    allt = [p for t in picks for p in picks[t]]
    print(f"TOTAL {len(allt)} files\n")

    bad = [p for p in allt if os.path.islink(p) or not os.path.isfile(p) or os.lstat(p).st_nlink != 1]
    check("C1", not bad, f"regular files, one name each ({len(bad)} not; e.g. {bad[:1]})")

    ino = {(os.lstat(p).st_dev, os.lstat(p).st_ino) for p in allt if os.path.exists(p)}
    stray = []
    for root in (f"{R}/pdvd/work", f"{R}/pdhd/work", f"{R}/sbnd/sbnd_xin", "/home/xqian/tmp"):
        for cur, subs, fs in os.walk(root):
            if root.endswith("/tmp") and cur[len(root):].count("/") >= 4: subs[:] = []
            if "/archive/" in cur: continue
            for e in fs + subs:
                q = os.path.join(cur, e)
                if not os.path.islink(q): continue
                try: st = os.stat(q)
                except OSError: continue
                if (st.st_dev, st.st_ino) in ino: stray.append(q)
    check("C2", not stray, f"no symlink resolves onto a target ({len(stray)}; e.g. {stray[:1]})")

    held = []
    try:
        out = subprocess.run(["lsof", "-Fn", "--"] + allt[:2000], capture_output=True, text=True).stdout
        held = [l[1:] for l in out.splitlines() if l.startswith("n")]
    except FileNotFoundError: pass
    check("C3", not held, f"no process holds a target open ({len(held)})")

    hot = [p for p in allt if os.path.exists(p) and NOW - os.lstat(p).st_mtime < COLD]
    already = [p for p in allt if os.path.exists(p + ".zst")]
    check("C4", not hot and not already,
          f"nothing written in the last 7 d ({len(hot)}), no .zst already present ({len(already)})")

    # ---- C5 asserted against the round's own config, and negctl-proven
    off = []
    for tree in picks:
        for p in picks[tree]:
            m = re.search(r"/(\d{6})_(\d+)(?:_([^/]+))?/", p)
            arm = (m.group(3) if m and m.group(3) else "(bare)")
            if arm in holdsets[tree]:
                off.append(f"{tree}:{arm}:{p}")
    check("C5", not off, f"no production or bare-substrate family in scope "
                         f"({len(off)}; e.g. {off[:1]})")

    for tree in picks:
        out = os.path.join(HERE, f"files_compress_{tree}_{STAMP}{SUF}.txt")
        with open(out, "w") as fh:
            for p in sorted(picks[tree]): fh.write(p + "\n")
        print(f"  list: {os.path.basename(out)}  ({len(picks[tree])} lines)")

    if fails:
        print(f"\nGATES FAILED: {fails}"); return 1
    if "--hash" in sys.argv:
        os.makedirs(OUT, exist_ok=True)
        for tree in picks:
            man = os.path.join(OUT, f"{tree}-compress.manifest.tsv")
            if os.path.exists(man):
                print(f"REFUSED: {man} already exists (M13)"); return 1
            with ProcessPoolExecutor(max_workers=16) as ex, open(man + ".tmp", "w") as fh:
                fh.write("# relpath\tsize\tmtime\tsha256\n")
                for p, s, mt, d in ex.map(sha, picks[tree], chunksize=32):
                    fh.write(f"{os.path.relpath(p, R)}\t{s}\t{mt}\t{d}\n")
            os.replace(man + ".tmp", man)
            print(f"  froze {len(picks[tree])} files -> {os.path.basename(man)}")
    print("\nThis script compressed nothing.  Review the lists, then run the driver (CONFIRM=yes).")
    return 0

if __name__ == "__main__":
    sys.exit(main())
