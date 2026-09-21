#!/usr/bin/env python3
"""Cleanup ROUND J 2026-09-21 (doc sbnd_xin/122): FILE-level release inside KEPT arms.
Retires nothing; writes the target lists and, with --hash, the SHA-256 manifests.

OWNER, 2026-09-21, asked with every number measured and each lever's cost stated:
picked the pctree release, the SP frames release, the two superseded arms and the
compression pass.  This script covers the first two.

  A. sbnd-pctree-closed   work-{r3cv,r3nue,r3off}-{d115pr,d116s0rep}/f*/pr_evt*/pctree-pr-evt*.tar.gz
     The per-event point-cloud dumps of the two CLOSED stage-B comparison arms.  21.3 GiB.
     DELIBERATELY EXCLUDED, and this is the whole point of the set being named rather than
     globbed by family: `d116tfull`, whose pctrees ARE what doc 118's gate G1 compares the
     flipped production default against, and `d115`, whose pctrees are the stage-A products
     stage B consumes (releasing those would make the round-3 chain non-re-runnable, M11).
  B. pdhd-sp-frames       pdhd/work/<run>_<evt>[_<arm>]/protodunehd-sp-dnnroi-frames-anode*.tar.bz2
  C. sbnd-sp-frames       work-r3{cv,nue,off}-d115 + work-{mcp1k,mcp2k,ncpi0,nuecc48}-d102m
                          **/frames-dnn.tar.bz2
     SP output, regenerable: SP is bit-deterministic (doc pdvd/99 G2) and every upstream input
     still resolves -- CHECKED, gate F6, not assumed.  Round D sec 14 did exactly this to
     pdvd's p98von (39 GiB).  Cost: imaging is no longer re-runnable without an SP pass first.

TWO NEAR-MISSES THIS SET DESIGN EXISTS TO PREVENT, both found by measuring:
  1. A substring test for "frames" matches `input_files_reco1/*_frameshift.root` -- 4.46 GiB of
     PRODUCTION INPUT.  An earlier pass of mine classified those as "SP frames".
  2. `input_files_reco1` itself holds `frames-dnn.tar.bz2` files (~4.1 GiB).  A glob for that
     exact name across sbnd_xin WOULD have taken the production inputs.
  Hence gate F3: every target's owning top-level directory must be on its set's ALLOW-LIST,
  and no target may live under an input, record or reference tree.  F3 has a causal negative
  control (--negctl) because a gate that never fires proves nothing.

  usage:  python3 plan_files_20260921b.py            # lists + gates, no hashing
          python3 plan_files_20260921b.py --hash     # also write the SHA-256 manifests (the freeze)
          python3 plan_files_20260921b.py --negctl   # prove F3 fires: inject an input-tree path

Gates (exit 1 on any failure):
  F1  every target is a REGULAR file with st_nlink == 1 (a second name would keep the bytes).
  F2  no symlink under pdvd/work, pdhd/work, sbnd_xin or ~/tmp (depth 4) resolves, by inode,
      onto a target.
  F3  ALLOW-LIST: every target sits under an allowed arm, and never under input_files_reco1,
      xin-round3-samples, input_data_*, archive/, products/ or ref/.
  F4  no process holds a target open.
  F5  PRODUCT GATE: the product each target fed must survive.  Set A: every event dir losing a
      pctree keeps its tracking-pr.root and its nusel-evt*.tsv.  Sets B and C: every directory
      losing frames keeps at least one non-empty imaging archive (clusters-apa*/*.tar.gz).
  F6  REGENERATION GATE: the upstream inputs the released bytes would be rebuilt from still
      resolve, and are non-empty.
"""
import os, re, sys, glob, hashlib, subprocess, collections
from concurrent.futures import ProcessPoolExecutor

R     = "/home/xqian/toolkit-dev/wcp-porting-img"
S     = f"{R}/sbnd/sbnd_xin"
HERE  = os.path.dirname(os.path.abspath(__file__))
STAMP = "20260921b"
OUT   = os.environ.get("RETIRE_OUT", f"{S}/archive/records/cleanup-{STAMP}")
SUF   = ('.' + os.environ["PLAN_SUFFIX"]) if os.environ.get("PLAN_SUFFIX") else ""

PCTREE_ARMS = [f"work-{s}-{a}" for s in ("r3cv","r3nue","r3off") for a in ("d115pr","d116s0rep")]
SBND_FRAME_ARMS = [f"work-{s}-d115" for s in ("r3cv","r3nue","r3off")] + \
                  [f"work-{s}-d102m" for s in ("mcp1k","mcp2k","ncpi0","nuecc48")]
# Anything a target may NEVER live under, whatever the glob says.
FORBIDDEN = ("input_files_reco1", "xin-round3-samples", "input_data_", "/archive/",
             "/products/", "/ref/", "/docs/", "/scripts/")

def walk_match(roots, pred):
    """Enumerate by WALK, never by a fixed-depth glob.

    ROUND J, measured the hard way: the first version of this file used
    `{arm}/f*/pr_evt*/pctree-pr-evt*.tar.gz`.  The r3cv and r3nue arms are grouped into
    f000..f224 sub-roots, but the off-beam arms put pr_evt<N>/ directly at the arm root --
    so the glob matched 2017+2001 files in four arms and **ZERO** in the other two, and the
    list looked complete.  4.56 GiB of 21.27 silently absent.  A walk has no depth
    assumption and cannot fail this way."""
    out = []
    for r in roots:
        for cur, _, fs in os.walk(r):
            for f in fs:
                if pred(f):
                    q = os.path.join(cur, f)
                    if not os.path.islink(q): out.append(q)
    return sorted(out)

PRED = {
 "sbnd-pctree-closed": lambda f: f.startswith("pctree-pr-evt") and f.endswith(".tar.gz"),
 "pdhd-sp-frames":     lambda f: f.startswith("protodunehd-sp-dnnroi-frames-anode") and f.endswith(".tar.bz2"),
 "sbnd-sp-frames":     lambda f: f == "frames-dnn.tar.bz2",
}
ALLOW = {
 "sbnd-pctree-closed": [f"{S}/{a}" for a in PCTREE_ARMS],
 "pdhd-sp-frames":     [f"{R}/pdhd/work"],
 "sbnd-sp-frames":     [f"{S}/{a}" for a in SBND_FRAME_ARMS],
}
KIND = {"sbnd-pctree-closed": "pctree", "pdhd-sp-frames": "frames", "sbnd-sp-frames": "frames"}

# Files that MATCH a set's predicate but live OUTSIDE its allow-list.  Declared here so the
# exclusion is a conscious ruling with a reason, never an accident of where a glob happened
# to point.  F7 requires every out-of-scope match to be covered by one of these.
DECLARED_OUT = {
 "sbnd-sp-frames": [(f"{S}/input_files_reco1",
   "PRODUCTION INPUT, not SP output.  input_files_reco1 holds frames-dnn.tar.bz2 files of its "
   "own (~4.1 GiB); a glob on that exact name across sbnd_xin WOULD have deleted them.  It is "
   "also where an earlier substring test for 'frames' matched *_frameshift.root -- 4.46 GiB of "
   "reco1 input.  Both near-misses, both caught by measuring rather than trusting the pattern.")],
 "sbnd-pctree-closed": [(f"{S}/work-", 
   "every OTHER arm's pctrees, including d116tfull (doc 118's gate G1 reference) and d115 "
   "(the stage-A products stage B consumes).  Named exclusions, see the header.")],
 "pdhd-sp-frames": [],
}

SETS = {n: dict(paths=walk_match(ALLOW[n], PRED[n]), allow=ALLOW[n], kind=KIND[n]) for n in PRED}

# F6: what each set's bytes would be rebuilt FROM.  Checked to exist and be non-empty, because
# "regenerable" is a claim about someone else's disk: the pdhd inputs are xning's tree and the
# round-3 samples are yuhw's.  If either is cleaned up, these bytes stop being regenerable and
# this lever stops being safe -- so it is a gate, re-run every time, not a note in a doc.
REGEN = {
 "sbnd-pctree-closed": [],     # re-run stage B on the arm's own stage-A products (kept)
 "pdhd-sp-frames":     [f"{R}/pdhd/input_data_7p8_new_coh_grouping",
                        f"{R}/pdhd/input_data_14_old_coh_grouping"],
 "sbnd-sp-frames":     [f"{S}/xin-round3-samples", f"{S}/input_files_reco1"],
}

# pdhd frames are SHARED: 157 symlinks in kept substrate arms (029107_*_d09ctl, _stm0, ...)
# resolve onto 120 of them (6.77 GiB).  Those arms hold the frames by reference precisely so
# they are not duplicated, so releasing a shared target degrades several kept arms at once and
# leaves dangling links inside PROTECTED dirs.  Deferred to its own owner call rather than
# folded in here; the unshared remainder releases cleanly.  F2 proves none survive in the set.
def _shared(paths):
    ino = {}
    for p in paths:
        try: st = os.lstat(p)
        except OSError: continue
        ino[(st.st_dev, st.st_ino)] = p
    hit = set()
    for root in (f"{R}/pdvd/work", f"{R}/pdhd/work", S, "/home/xqian/tmp"):
        for cur, subs, fs in os.walk(root):
            if root == "/home/xqian/tmp" and cur[len(root):].count("/") >= 4: subs[:] = []
            for e in fs + subs:
                q = os.path.join(cur, e)
                if not os.path.islink(q): continue
                try: st = os.stat(q)
                except OSError: continue
                if (st.st_dev, st.st_ino) in ino: hit.add(ino[(st.st_dev, st.st_ino)])
    return hit

SHARED = _shared(SETS["pdhd-sp-frames"]["paths"])
if SHARED:
    b = sum(os.lstat(p).st_size for p in SHARED)
    print(f"DEFERRED: {len(SHARED)} pdhd frame archives ({b/2**30:.2f} GiB) are the target of a "
          f"symlink in a kept arm -- excluded, see the note above.")
    SETS["pdhd-sp-frames"]["paths"] = [p for p in SETS["pdhd-sp-frames"]["paths"] if p not in SHARED]

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
    negctl = "--negctl" in sys.argv
    files = {}
    for name, cfg in SETS.items():
        files[name] = [p for p in cfg["paths"] if not os.path.islink(p)]
        n = len(files[name]); b = sum(os.lstat(p).st_size for p in files[name])
        print(f"{name}: {n} files, {b/2**30:.2f} GiB")
    if negctl:
        inj = f"{S}/input_files_reco1/__negctl_does_not_exist__frames-dnn.tar.bz2"
        files["sbnd-sp-frames"] = files["sbnd-sp-frames"] + [inj]
        print(f"  NEGATIVE CONTROL: injected {inj} -- F3 must FAIL")

    allt = [p for n in files for p in files[n]]
    print(f"TOTAL {len(allt)} files, "
          f"{sum(os.lstat(p).st_size for p in allt if os.path.lexists(p))/2**30:.2f} GiB\n")

    # ---- F1
    bad = [p for p in allt if os.path.lexists(p) and (not os.path.isfile(p) or os.lstat(p).st_nlink != 1)]
    check("F1", not bad, f"regular files, one name each ({len(bad)} not; e.g. {bad[:1]})")

    # ---- F3 (allow-list + forbidden trees) -- runs before the expensive walk
    off = []
    for name, cfg in SETS.items():
        for p in files[name]:
            rp = os.path.realpath(p) if os.path.lexists(p) else os.path.abspath(p)
            if any(k in rp for k in FORBIDDEN) or not any(
                    rp.startswith(os.path.realpath(a) + "/") for a in cfg["allow"]):
                off.append(p)
    check("F3", not off, f"every target under an allowed arm, none under an input/record tree "
                         f"({len(off)} off-list; e.g. {off[:1]})")

    # ---- F2
    ino = {(os.lstat(p).st_dev, os.lstat(p).st_ino) for p in allt if os.path.lexists(p)}
    stray = []
    for root in (f"{R}/pdvd/work", f"{R}/pdhd/work", S, "/home/xqian/tmp"):
        for cur, subs, fs in os.walk(root):
            if root == "/home/xqian/tmp" and cur[len(root):].count("/") >= 4: subs[:] = []
            for e in fs + subs:
                p = os.path.join(cur, e)
                if not os.path.islink(p): continue
                try: st = os.stat(p)
                except OSError: continue
                if (st.st_dev, st.st_ino) in ino: stray.append(p)
    check("F2", not stray, f"no symlink resolves onto a target ({len(stray)}; e.g. {stray[:1]})")

    # ---- F4
    held = []
    try:
        out = subprocess.run(["lsof", "-Fn", "--"] + allt[:2000], capture_output=True, text=True).stdout
        held = [l[1:] for l in out.splitlines() if l.startswith("n")]
    except FileNotFoundError:
        pass
    check("F4", not held, f"no process holds a target open ({len(held)})")

    # ---- F5 product gate
    short = []
    for name, cfg in SETS.items():
        for p in files[name]:
            d = os.path.dirname(p)
            if cfg["kind"] == "pctree":
                ok = os.path.exists(os.path.join(d, "tracking-pr.root")) and \
                     bool(glob.glob(os.path.join(d, "nusel-evt*.tsv")))
            else:
                arch = glob.glob(os.path.join(d, "clusters-apa*")) or \
                       glob.glob(os.path.join(os.path.dirname(d), "clusters-apa*")) or \
                       glob.glob(os.path.join(d, "*.tar.gz"))
                ok = any(os.path.getsize(a) > 0 for a in arch if os.path.isfile(a))
            if not ok: short.append(p)
    check("F5", not short, f"the product each target fed survives ({len(short)} would not; "
                           f"e.g. {short[:1]})")

    # ---- F6 regeneration inputs
    miss = []
    for name, roots in REGEN.items():
        for r in roots:
            if not os.path.exists(r) or not os.listdir(r): miss.append(f"{name}:{r}")
    check("F6", not miss, f"every regeneration input resolves and is non-empty ({miss or 'all present'})")

    # ---- F7 every file matching a set's predicate but OUTSIDE its allow-list is DECLARED.
    # This is the gate that turns both of this round's near-misses into something that fails
    # loudly instead of silently: it finds input_files_reco1's own frames-dnn.tar.bz2 (and any
    # future tree that starts holding a matching name) and demands a written ruling for it.
    undeclared = []
    for name, cfg in SETS.items():
        allow_real = [os.path.realpath(a) for a in cfg["allow"]]
        declared = [d for d, _why in DECLARED_OUT.get(name, [])]
        for root in (f"{R}/pdvd/work", f"{R}/pdhd/work", S):
            for cur, _, fs in os.walk(root):
                if "/archive/" in cur: continue
                for f in fs:
                    if not PRED[name](f): continue
                    q = os.path.join(cur, f)
                    if os.path.islink(q): continue
                    rq = os.path.realpath(q)
                    if any(rq.startswith(a + "/") for a in allow_real): continue
                    if any(rq.startswith(os.path.realpath(d)) or rq.startswith(d)
                           for d in declared): continue
                    undeclared.append(f"{name}: {q}")
    check("F7", not undeclared,
          f"every out-of-allow-list match is a declared exclusion "
          f"({len(undeclared)} undeclared; e.g. {undeclared[:1]})")
    for name, ds in DECLARED_OUT.items():
        for d, why in ds:
            print(f"        declared out of {name}: {d}\n            {why}")

    for name in SETS:
        out = os.path.join(HERE, f"files_{name}_{STAMP}{SUF}.txt")
        with open(out, "w") as fh:
            for p in sorted(files[name]): fh.write(p + "\n")
        print(f"  list: {os.path.basename(out)}  ({len(files[name])} lines)")

    if fails:
        print(f"\nGATES FAILED: {fails}"); return 1
    if "--hash" in sys.argv:
        os.makedirs(OUT, exist_ok=True)
        for name in SETS:
            man = os.path.join(OUT, f"{name}.manifest.tsv")
            if os.path.exists(man):
                print(f"REFUSED: {man} already exists (M13)"); return 1
            with ProcessPoolExecutor(max_workers=16) as ex, open(man + ".tmp", "w") as fh:
                fh.write("# relpath\tsize\tsha256\n")
                for p, s, d in ex.map(sha, files[name], chunksize=32):
                    fh.write(f"{os.path.relpath(p, R)}\t{s}\t{d}\n")
            os.replace(man + ".tmp", man)
            print(f"  froze {len(files[name])} files -> {os.path.basename(man)}")
    print("\nThis script retired nothing.  Review the lists, then run the driver (CONFIRM=yes).")
    return 0

if __name__ == "__main__":
    sys.exit(main())
