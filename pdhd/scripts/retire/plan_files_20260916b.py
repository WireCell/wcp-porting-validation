#!/usr/bin/env python3
"""Cleanup ROUND D follow-up 2026-09-16 (doc sbnd_xin/111 sec 14): FILE-level release inside two kept
production arms.  Retires nothing; writes the target lists and the SHA-256 manifest.

OWNER, 2026-09-16 (after round D): "We can remove the 40 G SP frames inside p98von, and remove the
sbnd's final icluster npz files."  Both were doc 111 sec 12 "Open for the owner" items:
  A. pdvd `<evt>_p98von/protodune-sp-dnnroi-frames-anode*.tar.bz2` -- the production SP frames the
     imaging tag `pvdimg` was imaged from (doc pdvd/100 line 422).  pvdimg holds hard links of
     p98von's imaging ARCHIVES only, not of these frames.
  B. sbnd `work-mcp1k-d102m` / `work-mcp2k-d102m` `evt<N>/icluster-apa*-{active,masked}.npz` -- the
     production imaging output stage-A clustering read (doc 106 sec 12, 9.82 + 19.52 GiB), plus the
     `ql_evt<N>/icluster-*.npz` symlinks INSIDE the same arms that point at them (they would dangle).
     work-ncpi0-d102m / work-nuecc48-d102m (0.7 GiB) are NOT in scope: the item the owner approved was
     the mcp1k/mcp2k 29.3 GiB.

  usage:  python3 plan_files_20260916b.py          # lists + gates, no hashing
          python3 plan_files_20260916b.py --hash   # also writes the SHA-256 manifests (the freeze)

Gates (exit 1 on any failure, printed PASS/FAIL):
  F1  every target is a REGULAR file with st_nlink == 1 (a second name would keep the bytes and
      mean some other dir still uses them).
  F2  no symlink anywhere in pdvd/work, pdhd/work, sbnd_xin or ~/tmp (depth 4) resolves -- by inode,
      through any chain -- onto a target, except the listed in-arm ql_evt links of set B.
  F3  every listed link of set B resolves onto a target of set B.
  F4  no process holds a target open.
  F5  pvdimg's archives are not targets (inode-disjoint), and the frames are not linked into pvdimg.
"""
import os, re, sys, glob, hashlib
from concurrent.futures import ProcessPoolExecutor

R     = "/home/xqian/toolkit-dev/wcp-porting-img"
HERE  = os.path.dirname(os.path.abspath(__file__))
STAMP = "20260916b"
OUT   = os.environ.get("RETIRE_OUT", f"{R}/sbnd/sbnd_xin/archive/records/cleanup-{STAMP}")
# the confirm-time re-plan writes files_*.<suffix>.txt, never the plan-time lists (doc-104 defect)
SUF   = ('.' + os.environ["PLAN_SUFFIX"]) if os.environ.get("PLAN_SUFFIX") else ""
SETS  = {
  "pdvd-p98von-frames": sorted(glob.glob(f"{R}/pdvd/work/*_p98von/protodune-sp-dnnroi-frames-anode*.tar.bz2")),
  "sbnd-d102m-icluster": sorted(p for arm in ("work-mcp1k-d102m", "work-mcp2k-d102m")
                                for p in glob.glob(f"{R}/sbnd/sbnd_xin/{arm}/*/icluster-apa*-*.npz")),
}
fails = []

def check(n, ok, msg):
    print(f"  {'PASS' if ok else 'FAIL'}  {n}: {msg}")
    if not ok: fails.append(n)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return p, os.lstat(p).st_size, h.hexdigest()

def main():
    files, links = {}, {}
    for name, paths in SETS.items():
        files[name] = [p for p in paths if not os.path.islink(p)]
        links[name] = [p for p in paths if os.path.islink(p)]
        print(f"{name}: {len(files[name])} regular files, {len(links[name])} in-arm links, "
              f"{sum(os.lstat(p).st_size for p in files[name]) / 2**30:.2f} GiB")
    # F1
    bad = [p for n in files for p in files[n] if not os.path.isfile(p) or os.lstat(p).st_nlink != 1]
    check("F1", not bad, f"targets are regular files with one name ({len(bad)} not; e.g. {bad[:1]})")
    ino = {(os.lstat(p).st_dev, os.lstat(p).st_ino): (n, p) for n in files for p in files[n]}
    # F3
    wrong = [l for n in links for l in links[n]
             if not os.path.exists(l) or ino.get((os.stat(l).st_dev, os.stat(l).st_ino), ("", ""))[0] != n]
    check("F3", not wrong, f"every listed in-arm link resolves onto its own set ({len(wrong)} do not)")
    if any(links["pdvd-p98von-frames"]):
        check("F3b", False, "set A must hold no links")
    listed = {l for n in links for l in links[n]}
    # F2
    stray = []
    roots = [f"{R}/pdvd/work", f"{R}/pdhd/work", f"{R}/sbnd/sbnd_xin", "/home/xqian/tmp"]
    for root in roots:
        for cur, subs, fs in os.walk(root):
            if root == "/home/xqian/tmp" and cur[len(root):].count("/") >= 4:
                subs[:] = []
            for e in fs + subs:
                p = os.path.join(cur, e)
                if not os.path.islink(p) or p in listed: continue
                try: st = os.stat(p)
                except OSError: continue
                if (st.st_dev, st.st_ino) in ino: stray.append(p)
            subs[:] = [s for s in subs if not os.path.islink(os.path.join(cur, s)) and s != "archive"]
    check("F2", not stray, f"no other symlink resolves onto a target ({len(stray)}; e.g. {stray[:2]})")
    # F4
    held = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try: fds = os.listdir(f"/proc/{pid}/fd")
        except OSError: continue
        for fd in fds:
            try: st = os.stat(f"/proc/{pid}/fd/{fd}")
            except OSError: continue
            if (st.st_dev, st.st_ino) in ino: held.append((pid, ino[(st.st_dev, st.st_ino)][1]))
    check("F4", not held, f"no process holds a target open ({held[:1] or 'none'})")
    # F5
    pv = {(os.lstat(p).st_dev, os.lstat(p).st_ino) for p in glob.glob(f"{R}/pdvd/work/*_pvdimg/*") if not os.path.islink(p)}
    check("F5", not (pv & set(ino)), f"pvdimg ({len(pv)} files) shares no inode with the targets")

    for n in files:
        open(os.path.join(HERE, f"files_{n}_{STAMP}{SUF}.txt"), "w").write("\n".join(files[n]) + "\n")
        open(os.path.join(HERE, f"links_{n}_{STAMP}{SUF}.txt"), "w").write(
            "".join(f"{l}\t{os.readlink(l)}\n" for l in links[n]))
    if "--hash" in sys.argv and not fails:
        os.makedirs(OUT, exist_ok=True)
        for n in files:
            man = os.path.join(OUT, f"{n}.manifest.tsv")
            if os.path.exists(man): sys.exit(f"REFUSING: {man} exists (M13)")
            with ProcessPoolExecutor(16) as ex:
                rows = sorted(ex.map(sha, files[n], chunksize=16))
            open(man, "w").write("".join(f"{p}\t{s}\t{h}\n" for p, s, h in rows))
            open(os.path.join(OUT, f"{n}.links.txt"), "w").write(
                "".join(f"{l}\t{os.readlink(l)}\n" for l in links[n]))
            print(f"  froze {n}: {len(rows)} manifest rows -> {man}")
    print(f"gate failures: {fails or 'NONE'}")
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
