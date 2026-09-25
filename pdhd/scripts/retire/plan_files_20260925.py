#!/usr/bin/env python3
"""Cleanup ROUND K 2026-09-25 (doc sbnd_xin/125): FILE-level release inside KEPT sbnd arms.
Retires nothing; writes the target lists and, with --hash, the SHA-256 manifests.
Fork of plan_files_20260921b.py (round J): same gate shape F1-F7, two new gates F8/F9, and the
round J sets replaced by this round's two.

OWNER, 2026-09-25, asked with the sizes and the cost stated ("Both (~90 GB)"):

  A. sbnd-icluster-split  work-<s>-d123base (x7) and work-<s>-d123lgop (x4):
       **/evt<N>/icluster-apa*-{active,masked}.npz      (regular files)
     plus the ql_evt<N>/icluster-*.npz SYMLINKS in the same arms that point at them ("links").
     These are the per-event split copies of the group imaging -- the Q/L job's input handoff
     (run_chain_group.sh:425-432: "The Q/L job above is its ONLY consumer -- the PR chain's
     compiled config contains no icluster/.npz reference at all").  The campaign ran with the
     runner default SBND_QL_KEEP_ICLUSTER=1, so every arm kept them.  They are re-made from the
     arm's g*/ group imaging by the split step (split_group_products.py), which stays.
  B. sbnd-base-qlevt      work-<s>-d123base/**/ql_evt<N>/*   (every file + link in those dirs)
     d123base's OWN Q/L output with reco1 flashes: the pre-flip product (doc 123 secs 17/19).
     Its stage B (d123basepr) is released at directory level in the same round.

  g*/ IS NEVER TOUCHED (gate F8).  It holds the only SP frames (frames-dnn.tar.bz2) and group
  imaging on our disk, every lgop arm links into work-<s>-d123base/g*/ by absolute path, and any
  later arm (the rescue-off gate of doc 123 sec 18.6) will be built on it through hits_arm.sh.

COST, stated for the record: the off-path gates of doc 123 sec 17.3 (flipoff == base) and
sec 19.3 (lgflipoff == hits) can no longer be re-checked from disk without re-running Q/L.

  usage:  python3 plan_files_20260925.py            # lists + gates, no hashing
          python3 plan_files_20260925.py --hash     # also write the SHA-256 manifests (the freeze)
          python3 plan_files_20260925.py --negctl   # prove F3 and F9 fire

Gates (exit 1 on any failure):
  F1  every target is a REGULAR file with st_nlink == 1; every link-target is a symlink.
  F2  no symlink under pdvd/work, pdhd/work, sbnd_xin or ~/tmp (depth 4), OTHER than the set's
      own co-released links, resolves by inode onto a target.
  F3  ALLOW-LIST: every target and link sits under an allowed arm, never under an input/record/
      reference tree.
  F4  no process holds a target open (checked in chunks; an empty list is never passed to lsof).
  F5  PRODUCT GATE: every arm sub-root losing files keeps a non-empty g*/icluster-apa*-active.npz
      that RESOLVES (the re-split source), and for set A in lgop the ql_evt<N>/pctree survives.
  F6  REGENERATION inputs resolve: input_files_reco1, products/d115/*/files.lst, xin-round3-samples.
  F7  every file matching a set's predicate OUTSIDE the allow-list is a declared exclusion.
  F8  (new) no target or link has a g<N>/ path component.
  F9  (new) every co-released link resolves INTO this round's own target set -- a link that
      points anywhere else is not ours to remove.
"""
import os, re, sys, glob, hashlib, subprocess, collections
from concurrent.futures import ProcessPoolExecutor

R     = "/home/xqian/toolkit-dev/wcp-porting-img"
S     = f"{R}/sbnd/sbnd_xin"
HERE  = os.path.dirname(os.path.abspath(__file__))
STAMP = "20260925"
OUT   = os.environ.get("RETIRE_OUT", f"{S}/archive/records/cleanup-{STAMP}")
SUF   = ('.' + os.environ["PLAN_SUFFIX"]) if os.environ.get("PLAN_SUFFIX") else ""

BASE_ARMS = [f"work-{s}-d123base" for s in ("mcp1k","mcp2k","ncpi0","nuecc48","r3cv","r3nue","r3off")]
LGOP_ARMS = [f"work-{s}-d123lgop" for s in ("mcp1k","mcp2k","r3cv","r3off")]
FORBIDDEN = ("input_files_reco1", "xin-round3-samples", "input_data_", "/archive/",
             "/products/", "/ref/", "/docs/", "/scripts/")
ICL = re.compile(r"^icluster-apa\d+-(active|masked)\.npz$")
EVT = re.compile(r"^evt\d+$")
QLE = re.compile(r"^ql_evt\d+$")
GRP = re.compile(r"^g\d+$")

def walk(roots, want_dir):
    """(files, links) whose PARENT dir name satisfies want_dir.  A walk, never a fixed-depth glob
    (round J: the off-beam arms have no f*/ level, and a glob matched ZERO there)."""
    files, links = [], []
    for r in roots:
        for cur, subs, fs in os.walk(r):
            subs[:] = [x for x in subs if not os.path.islink(os.path.join(cur, x))]
            if not want_dir(os.path.basename(cur)): continue
            for f in fs:
                q = os.path.join(cur, f)
                (links if os.path.islink(q) else files).append(q)
    return files, links

def build():
    sets = {}
    # A: regular icluster files under evt<N>/ ; the ql_evt<N>/icluster links pointing at them
    fa, _ = walk([f"{S}/{a}" for a in BASE_ARMS + LGOP_ARMS], lambda b: bool(EVT.match(b)))
    fa = sorted(p for p in fa if ICL.match(os.path.basename(p)))
    _, la = walk([f"{S}/{a}" for a in LGOP_ARMS], lambda b: bool(QLE.match(b)))
    la = sorted(p for p in la if ICL.match(os.path.basename(p)))
    sets["sbnd-icluster-split"] = dict(paths=fa, links=la,
                                       allow=[f"{S}/{a}" for a in BASE_ARMS + LGOP_ARMS])
    # B: everything in d123base's ql_evt<N>/ (files + the icluster links, which point into evt<N>/)
    fb, lb = walk([f"{S}/{a}" for a in BASE_ARMS], lambda b: bool(QLE.match(b)))
    sets["sbnd-base-qlevt"] = dict(paths=sorted(fb), links=sorted(lb),
                                   allow=[f"{S}/{a}" for a in BASE_ARMS])
    return sets

# F7 for set A: every OTHER split icluster copy in sbnd_xin, by where it lives.
DECLARED_OUT = {
 "sbnd-icluster-split": [
   (f"{S}/work-", "the evt<N>/ split copies of every OTHER kept arm: the sec 17/19 flip records "
    "(d123flip*, d123lgflip*) keep theirs so they stay member-comparable to their gates, and no "
    "older arm is in scope this round.  The g<N>/ group copies are excluded by F8, not here."),
   (f"{S}/input_files_reco1", "production input tree -- never a target")],
 "sbnd-base-qlevt": [],
}
REGEN = [f"{S}/input_files_reco1", f"{S}/xin-round3-samples"] + \
        [f"{S}/products/d115/{s}/files.lst" for s in ("cv", "nuecc", "off")]

fails = []
def check(n, ok, msg):
    print(f"  {'PASS' if ok else 'FAIL'}  {n}: {msg}")
    if not ok: fails.append(n)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return p, os.lstat(p).st_size, h.hexdigest()

def subroot(p):
    """the arm sub-root a per-event dir hangs off: <arm> or <arm>/f<NNN>."""
    d = os.path.dirname(os.path.dirname(p))
    return d

def main():
    negctl = "--negctl" in sys.argv
    SETS = build()
    for name, cfg in SETS.items():
        b = sum(os.lstat(p).st_size for p in cfg["paths"])
        print(f"{name}: {len(cfg['paths'])} files, {b/2**30:.2f} GiB; {len(cfg['links'])} co-released links")
    if negctl:
        inj = f"{S}/input_files_reco1/__negctl__/evt1/icluster-apa0-active.npz"
        SETS["sbnd-icluster-split"]["paths"].append(inj)
        injl = f"{S}/work-mcp1k-d123lgop/__negctl__/ql_evt1/icluster-apa0-active.npz"
        SETS["sbnd-icluster-split"]["links"].append(injl)
        print(f"  NEGATIVE CONTROL: injected {inj} (F3 must FAIL) and link {injl} (F9 must FAIL)")
    allt  = [p for c in SETS.values() for p in c["paths"]]
    alll  = [p for c in SETS.values() for p in c["links"]]
    print(f"TOTAL {len(allt)} files, {sum(os.lstat(p).st_size for p in allt if os.path.lexists(p))/2**30:.2f} GiB; "
          f"{len(alll)} links\n")

    # F1
    bad = [p for p in allt if os.path.lexists(p) and (not os.path.isfile(p) or os.path.islink(p)
                                                      or os.lstat(p).st_nlink != 1)]
    badl = [p for p in alll if os.path.lexists(p) and not os.path.islink(p)]
    check("F1", not bad and not badl, f"targets are regular files with one name ({len(bad)} not), "
          f"links are symlinks ({len(badl)} not)")
    # F3
    off = []
    for name, cfg in SETS.items():
        allow = [os.path.realpath(a) for a in cfg["allow"]]
        for p in cfg["paths"] + cfg["links"]:
            ap = os.path.abspath(p)
            rp = os.path.join(os.path.realpath(os.path.dirname(ap)), os.path.basename(ap)) \
                 if os.path.lexists(os.path.dirname(ap)) else ap
            if any(k in rp for k in FORBIDDEN) or not any(rp.startswith(a + "/") for a in allow):
                off.append(p)
    check("F3", not off, f"every target/link under an allowed arm, none under an input/record tree "
                         f"({len(off)} off-list; e.g. {off[:1]})")
    # F8
    gpath = [p for p in allt + alll if any(GRP.match(x) for x in os.path.relpath(p, S).split("/"))]
    check("F8", not gpath, f"nothing under a g<N>/ group dir ({len(gpath)}; e.g. {gpath[:1]})")
    # F9 co-released links resolve into this round's own targets
    tino = {}
    for p in allt:
        if os.path.lexists(p):
            st = os.lstat(p); tino[(st.st_dev, st.st_ino)] = p
    foreign = []
    for p in alll:
        try: st = os.stat(p)
        except OSError: foreign.append(p + " (dangling)"); continue
        if (st.st_dev, st.st_ino) not in tino: foreign.append(p)
    check("F9", not foreign, f"every co-released link resolves into the target set "
                             f"({len(foreign)} do not; e.g. {foreign[:1]})")
    # F2 no OTHER symlink resolves onto a target
    own = set(os.path.abspath(p) for p in alll)
    stray = []
    for root in (f"{R}/pdvd/work", f"{R}/pdhd/work", S, "/home/xqian/tmp"):
        for cur, subs, fs in os.walk(root):
            if root == "/home/xqian/tmp" and cur[len(root):].count("/") >= 4: subs[:] = []
            if "/archive/" in cur + "/": subs[:] = []; continue
            for e in fs + subs:
                p = os.path.join(cur, e)
                if not os.path.islink(p) or os.path.abspath(p) in own: continue
                try: st = os.stat(p)
                except OSError: continue
                if (st.st_dev, st.st_ino) in tino: stray.append(p)
    check("F2", not stray, f"no other symlink resolves onto a target ({len(stray)}; e.g. {stray[:1]})")
    # F4 (chunked; never an empty argument list -- round J's C3 fired on nothing that way)
    held = []
    for i in range(0, len(allt), 2000):
        chunk = [p for p in allt[i:i + 2000] if os.path.lexists(p)]
        if not chunk: continue
        try:
            out = subprocess.run(["lsof", "-Fn", "--"] + chunk, capture_output=True, text=True).stdout
            held += [l[1:] for l in out.splitlines() if l.startswith("n")]
        except FileNotFoundError:
            break
    check("F4", not held, f"no process holds a target open ({len(held)})")
    # F5 product gate
    short = []
    seen = {}
    for name, cfg in SETS.items():
        for p in cfg["paths"]:
            sr = subroot(p)
            if sr not in seen:
                g = [q for q in glob.glob(os.path.join(sr, "g*", "icluster-apa*-active.npz"))
                     if os.path.exists(q) and os.path.getsize(q) > 0]
                seen[sr] = bool(g)
            ok = seen[sr]
            if ok and name == "sbnd-icluster-split" and "-d123lgop" in p:
                e = os.path.basename(os.path.dirname(p))[3:]
                ok = os.path.isfile(os.path.join(sr, f"ql_evt{e}", f"pctree-evt{e}.tar.gz"))
            if not ok: short.append(p)
    check("F5", not short, f"every sub-root keeps its g*/ imaging (resolving) and lgop keeps its "
                           f"Q/L pctree ({len(short)} would not; e.g. {short[:1]}); {len(seen)} sub-roots")
    # F6
    miss = [r for r in REGEN if not os.path.exists(r) or (os.path.isdir(r) and not os.listdir(r))
            or (os.path.isfile(r) and os.path.getsize(r) == 0)]
    check("F6", not miss, f"regeneration inputs resolve and are non-empty ({miss or 'all present'})")
    # F7 set A: undeclared split copies elsewhere
    allow_a = [os.path.realpath(a) for a in SETS["sbnd-icluster-split"]["allow"]]
    declared = [d for d, _ in DECLARED_OUT["sbnd-icluster-split"]]
    undeclared = []
    for cur, subs, fs in os.walk(S):
        if "/archive/" in cur + "/": subs[:] = []; continue
        if not EVT.match(os.path.basename(cur)): continue
        for f in fs:
            q = os.path.join(cur, f)
            if not ICL.match(f) or os.path.islink(q): continue
            rq = os.path.realpath(q)
            if any(rq.startswith(a + "/") for a in allow_a): continue
            if any(q.startswith(d) for d in declared): continue
            undeclared.append(q)
    check("F7", not undeclared, f"every out-of-allow-list split copy is a declared exclusion "
                                f"({len(undeclared)} undeclared; e.g. {undeclared[:1]})")
    for d, why in DECLARED_OUT["sbnd-icluster-split"]:
        print(f"        declared out of sbnd-icluster-split: {d}\n            {why}")

    for name, cfg in SETS.items():
        for kind in ("paths", "links"):
            out = os.path.join(HERE, f"files_{name}{'' if kind == 'paths' else '.links'}_{STAMP}{SUF}.txt")
            with open(out, "w") as fh:
                for p in sorted(cfg[kind]): fh.write(p + "\n")
            print(f"  list: {os.path.basename(out)}  ({len(cfg[kind])} lines)")
    if fails:
        print(f"\nGATES FAILED: {fails}"); return 1
    if "--hash" in sys.argv:
        os.makedirs(OUT, exist_ok=True)
        for name, cfg in SETS.items():
            man = os.path.join(OUT, f"{name}.manifest.tsv")
            lnk = os.path.join(OUT, f"{name}.links.tsv")
            if os.path.exists(man) or os.path.exists(lnk):
                print(f"REFUSED: {man} already exists (M13)"); return 1
            with ProcessPoolExecutor(max_workers=16) as ex, open(man + ".tmp", "w") as fh:
                fh.write("# relpath\tsize\tsha256\n")
                for p, s, d in ex.map(sha, cfg["paths"], chunksize=64):
                    fh.write(f"{os.path.relpath(p, R)}\t{s}\t{d}\n")
            os.replace(man + ".tmp", man)
            with open(lnk, "w") as fh:
                fh.write("# relpath\tlink target\n")
                for p in cfg["links"]: fh.write(f"{os.path.relpath(p, R)}\t{os.readlink(p)}\n")
            print(f"  froze {len(cfg['paths'])} files -> {os.path.basename(man)}, "
                  f"{len(cfg['links'])} links -> {os.path.basename(lnk)}")
    print("\nThis script retired nothing.  Review the lists, then run the driver (CONFIRM=yes).")
    return 0

if __name__ == "__main__":
    sys.exit(main())
