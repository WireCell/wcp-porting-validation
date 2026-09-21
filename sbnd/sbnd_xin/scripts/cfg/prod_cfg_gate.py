#!/usr/bin/env python3
"""prod_cfg_gate.py -- has the compiled production operating point drifted?

doc 77 sec 13.  The knob-defaults doctest
(clus/test/doctest_clus_knob_defaults.cxx) pins the C++ defaults and says so in
its own header: "A green run here does NOT mean production is on the legacy
path."  The compiled-config proof used inside a round compares two freshly
compiled trees, so it catches a change made DURING that round and nothing else.

Neither answers the question a production tree has to answer between rounds:

    is the operating point today still the one that was validated?

At the time this was written, 315 of the SBND PR job's TLAs carried a
production value and no test guarded a single one of them.  doc pr/127 is the
cost of that gap -- a shipped fix died silently for ten days.

This script closes it by comparing a freshly compiled consumer set against a
COMMITTED reference (sbnd_xin/ref/prod-<date>/):

    consumers.sha256   one line per artifact -- detects any drift.  The set is
                       whatever compile_consumers.sh produces; 21 at doc 77,
                       25 after doc 118 added the runtime fit JSONs and the
                       LArSoft 1-step chain, 26 after doc 119 round 5 added the
                       PR runners' allocator block.
    prod_prjob.json    the SBND PR job in full -- so a drift can be NAMED,
                       key by key, not merely detected
    runner_alloc.txt   the runners' allocator block in full, for the same reason

Deliberately NOT wired into any build or runner.  It is a tripwire the owner
fires on purpose; a stale reference failing someone's build is worse than no
reference at all.  When a flip is intended, re-run with --refresh and say in
the round doc which knob moved and on whose word.

Usage:
    scripts/cfg/prod_cfg_gate.py                       # check against newest ref
    scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-01
    scripts/cfg/prod_cfg_gate.py --cfg /path/to/toolkit/cfg
    scripts/cfg/prod_cfg_gate.py --refresh             # adopt the current tree

Exit 0 = the operating point matches the reference; 1 = it drifted.
"""
import argparse, difflib, glob, hashlib, json, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.abspath(os.path.join(HERE, "..", ".."))
CFG = os.environ.get("WCT_CFG", "/nfs/data/1/xqian/toolkit-dev/toolkit/cfg")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def newest_ref():
    refs = sorted(glob.glob(os.path.join(SX, "ref", "prod-*")))
    return refs[-1] if refs else None


def compile_consumers(cfg, out):
    """Compile every live consumer of the SBND/common clustering config."""
    rc = subprocess.run([os.path.join(HERE, "compile_consumers.sh"), cfg, out],
                        capture_output=True, text=True)
    return rc.stdout


def flatten(obj, path=""):
    """JSON -> {dotted path: scalar}, so a drift can be reported by key."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, "%s.%s" % (path, k) if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten(v, "%s[%d]" % (path, i))
    else:
        yield path, obj


# doc sbnd_xin/119 round 5.  The artifacts the CURRENT compile produces, as opposed to the ones
# the reference happens to list.  `prod.standalone` / `prod.wcls` have no extension and
# `runner_alloc.txt` is not JSON, so the rule is by exclusion: everything compile_consumers.sh
# writes except its own logs (`_*.log`, `*.log`, `*.err`).
def produced(outdir):
    return {f for f in os.listdir(outdir)
            if os.path.isfile(os.path.join(outdir, f))
            and not f.startswith("_") and not f.endswith((".log", ".err"))}


# Artifacts kept in FULL inside the reference dir, so a drift can be named and not merely
# detected.  prod_prjob.json since doc 77; runner_alloc.txt since round 5 -- an allocator change
# is a one-line diff and printing it is the whole value of catching it.
KEEP_FULL = ("prod_prjob.json", "runner_alloc.txt")


def name_the_text_drift(ref_txt, new_txt):
    """Line diff for the non-JSON artifacts (the runners' allocator block)."""
    try:
        a = open(ref_txt).read().splitlines()
        b = open(new_txt).read().splitlines()
    except OSError as e:
        return ["  (could not diff: %s)" % e]
    return ["  " + l.rstrip() for l in difflib.unified_diff(a, b, "reference", "current", n=1)]


def name_the_drift(ref_json, new_json, limit=40):
    """Which keys moved.  This is the whole point of keeping the full JSON."""
    try:
        a = dict(flatten(json.load(open(ref_json))))
        b = dict(flatten(json.load(open(new_json))))
    except (OSError, ValueError) as e:
        return ["  (could not diff: %s)" % e]
    out = []
    for k in sorted(set(a) | set(b)):
        if k not in b:
            out.append("  REMOVED %s = %r" % (k, a[k]))
        elif k not in a:
            out.append("  ADDED   %s = %r" % (k, b[k]))
        elif a[k] != b[k]:
            out.append("  CHANGED %s : %r -> %r" % (k, a[k], b[k]))
    if len(out) > limit:
        out = out[:limit] + ["  ... and %d more" % (len(out) - limit)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None, help="reference dir (default: newest ref/prod-*)")
    ap.add_argument("--cfg", default=CFG, help="toolkit cfg root to compile")
    ap.add_argument("--keep", default=None, help="keep the freshly compiled set here")
    ap.add_argument("--refresh", action="store_true",
                    help="OVERWRITE the reference with the current tree (intended flips only)")
    a = ap.parse_args()

    ref = a.ref or newest_ref()
    if not ref or not os.path.isdir(ref):
        sys.exit("no reference dir; expected %s/ref/prod-<date>/" % SX)
    manifest = os.path.join(ref, "consumers.sha256")
    if not os.path.exists(manifest):
        sys.exit("reference has no consumers.sha256: %s" % ref)

    out = a.keep or tempfile.mkdtemp(prefix="prodcfg-")
    print("reference : %s" % ref)
    print("cfg tree  : %s" % a.cfg)
    print("compiling : %s" % out)
    log = compile_consumers(a.cfg, out)
    bad_rc = [l for l in log.splitlines() if l.strip() and not l.strip().endswith("rc=0")]
    if bad_rc:
        print("COMPILE PROBLEM:"); [print("  " + l) for l in bad_rc]

    want = {}
    for line in open(manifest):
        h, _, name = line.strip().partition("  ")
        if name:
            want[name] = h

    drift, missing = [], []
    for name, h in sorted(want.items()):
        p = os.path.join(out, name)
        if not os.path.exists(p):
            missing.append(name); continue
        if sha256(p) != h:
            drift.append(name)

    # doc sbnd_xin/119 round 5 -- artifacts this compile produced that the reference does not
    # list.  Until now the gate only ever looked up the names already in `want`, and --refresh
    # rewrote the manifest over `sorted(want)`, so ADDING a consumer to compile_consumers.sh did
    # nothing at all: the new artifact was compiled, ignored, never adopted, and the gate went on
    # reporting PASS on the old set.  That is the same shape of hole round 5 closes in the runners
    # -- a production change the instrument cannot see -- so it is fixed here, not worked around.
    have = produced(out)
    new = sorted(have - set(want))

    print("checked   : %d artifacts" % len(want))
    if missing:
        print("MISSING   : %s" % ", ".join(missing))
    if new:
        print("NEW       : %s   (in the compiled set, absent from the reference)" % ", ".join(new))
    if not drift and not missing and not new:
        print("PASS -- the compiled operating point matches %s" % os.path.basename(ref))
        if not a.keep:
            shutil.rmtree(out, ignore_errors=True)
        return 0

    if drift:
        print("DRIFT     : %s" % ", ".join(drift))
    if "prod_prjob.json" in drift:
        print("\nSBND PR job, key by key (reference -> current tree):")
        for line in name_the_drift(os.path.join(ref, "prod_prjob.json"),
                                   os.path.join(out, "prod_prjob.json")):
            print(line)
    if "runner_alloc.txt" in drift and os.path.exists(os.path.join(ref, "runner_alloc.txt")):
        print("\nPR runner allocator block (reference -> current tree):")
        for line in name_the_text_drift(os.path.join(ref, "runner_alloc.txt"),
                                        os.path.join(out, "runner_alloc.txt")):
            print(line)
    if a.refresh:
        for name in KEEP_FULL:
            src = os.path.join(out, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(ref, name))
        # Write the manifest from what was PRODUCED, not from what the old reference listed --
        # otherwise a newly added consumer can never be adopted (see the `new` comment above).
        with open(manifest, "w") as fh:
            for name in sorted(have):
                fh.write("%s  %s\n" % (sha256(os.path.join(out, name)), name))
        print("\nREFRESHED %s -- %d artifacts (%d added, %d dropped)."
              % (os.path.basename(ref), len(have), len(new), len(missing)))
        if new:
            print("  added  : %s" % ", ".join(new))
        if missing:
            print("  dropped: %s   <- check this is intended, not a failed compile"
                  % ", ".join(missing))
        print("  Record in the round doc WHICH knob moved and on whose word.")
        return 0
    print("\nIf this drift is an intended flip, re-run with --refresh and record it in the"
          "\nround doc.  If it is not, something changed the production operating point"
          "\nwithout a round behind it -- that is what this gate exists to catch.")
    if not a.keep:
        print("(compiled set left at %s)" % out)
    return 1


if __name__ == "__main__":
    sys.exit(main())
