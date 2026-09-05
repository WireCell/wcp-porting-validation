#!/usr/bin/env python3
"""pdvd work/ retire plan -- doc pdvd/29, 2026-09-04.

OWNER SCOPE, verbatim: "I would like retire some work* file in ./sbnd_xin and
./pdvd directory.  We want to keep the latest production result as well as
their input."  Asked which depth, the owner chose option A: keep the substrate
spine, the gate arms of the three shipped flips, and the live round.

WHY THIS IS NOT A FORK OF sbnd_xin's plan_20260904.py.  `work*` globs to 199
sibling arm dirs in sbnd_xin but to exactly ONE directory here -- pdvd's
retirable unit is the ARM-SUFFIX GROUP over work/<run6>_<idx>_<arm>.  Pointing
the sbnd planner at this tree yields either dirs=0 (the 2026-08-31 catch: a
forked driver silently read the previous round's list and reported zero) or one
catastrophic `rm -rf work/`.  The interlocks are carried; the code is not.

THE SUBSTRATE IS A CHAIN, AND IT IS LOAD-BEARING FIVE LEVELS DEEP:

    keep  ->  d27fresh  ->  d28dlfp  ->  d34base  ->  d37dloff  ->  d39*
     |          |                                                   (LIVE)
     |          `-- 9793 inbound links; stage_pr_tag.sh's documented
     |              default src_tag; the v7-uvwfit baseline of doc 27
     `-- 6208 inbound links, and 960 of them are d27fresh's own
         protodune-sp-dnnroi-frames-anode*.tar.bz2.  `keep` IS the
         imaging input the owner asked to preserve; it is not an old arm.

A FIRST CENSUS OF THAT GRAPH WAS WRONG AND NEARLY INVERTED THE PLAN.  Matching
symlink targets with a pattern anchored on `work/` scored d27fresh at 15 links
and ranked `keep` as the substrate: every `../<evt>_<arm>/...` RELATIVE target
was invisible.  The corrected census (both forms) is 20269 links.  Resolve each
link to the last path component that parses as <run6>_<idx>[_<arm>] -- never
grep the raw target string.

Writes state-20260904/plan.json.  Reads only; deletes nothing.
"""
import json, os, re, subprocess, sys, time

PDVD = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd"
WORK = os.path.join(PDVD, "work")
STATE = os.path.join(PDVD, "scripts", "retire", "state-20260904")
REC = os.path.join(PDVD, "archive", "records", "pdvd-rounds-20260904")
os.makedirs(STATE, exist_ok=True)
os.chdir(PDVD)

EVT = re.compile(r"^(\d{6})_(\d+)(?:_(.*))?$")

# ---------------------------------------------------------------- the sets --
SUBSTRATE = {
    "keep":      "the SP+DNNROI frames -- d27fresh borrows 960 of them; THE imaging input",
    "d27fresh":  "v7-uvwfit baseline (doc 27); stage_pr_tag.sh's default src_tag; 9793 inbound links",
    "d28dlfp":   "symlink spine: d34* resolve their pctree through it",
    "d34base":   "symlink spine: d37* resolve their pctree through it",
    "d37dloff":  "symlink spine: the LIVE d39 round resolves its pctree through it",
    "d31r6e2e":  "substrate for the d32r3* set",
    "perfslide": "substrate, doc 18",
}
GATE = {   # the arms named by the three SHIPPED owner flips
    "d36off":    "doc 36 sec 11 flip gate, OFF side (owner decision 2026-09-04)",
    "d36on":     "doc 36 sec 11 flip gate: the flipped default must reproduce THIS arm member-for-member",
    "d37off0":   "doc 37 flip control (thinning off)",
    "d37off1":   "doc 37 flip control",
    "d37on05":   "doc 37: PDVD production flipped to 0.5 cm -- the shipped point",
    "d38off":    "doc 38 sec 8.1 knob-off control; proves the knob fires (doc 77's ON-but-inert failure mode)",
    "d38h20":    "doc 38 sec 8.1: end_trim_gap_len=200, the shipped 20 cm operating point",
    "d38flip20": "doc 38 sec 8.1: the flipped-default config reproducing d38h20 member-for-member",
}
MISC = {"magnify": "cited by 25 files across docs 23/24/25; the magnify reference set"}
# LIVE PEER round -- protected by PREFIX, re-derived by INTERLOCK 3, never by age.
LIVE_PREFIX = ("d39",)

def arm_of(name):
    m = EVT.match(name)
    return None if not m else (m.group(3) or "(bare)")

alldirs = sorted(d for d in os.listdir(WORK)
                 if os.path.isdir(os.path.join(WORK, d))
                 and not os.path.islink(os.path.join(WORK, d)))
parsed = {d: arm_of(d) for d in alldirs}
UNIVERSE = {d: a for d, a in parsed.items() if a is not None}
# Everything the arm grammar does not parse is OUT OF SCOPE and must stay: the
# 267 `<run6>_light<flash>_<tag>` light-reco dirs are a different naming space.
OUT_OF_SCOPE = sorted(d for d, a in parsed.items() if a is None)

KEEP_ARMS = set(SUBSTRATE) | set(GATE) | set(MISC) | {
    a for a in set(UNIVERSE.values()) if a.startswith(LIVE_PREFIX)}
RETIRE = sorted(d for d, a in UNIVERSE.items() if a not in KEEP_ARMS)
KEEP = sorted(d for d, a in UNIVERSE.items() if a in KEEP_ARMS)

fails = []
def check(n, ok, msg):
    print(f"{'PASS' if ok else 'FAIL'}  INTERLOCK {n}: {msg}")
    if not ok: fails.append(n)

# 1 -- the substrate and gate arms are present at full 120-event coverage
cnt = {}
for d, a in UNIVERSE.items():
    cnt[a] = cnt.get(a, 0) + 1
short = {a: cnt.get(a, 0) for a in list(SUBSTRATE) + list(GATE)
         if a not in ("d31r6e2e", "perfslide", "d38flip20") and cnt.get(a, 0) != 120}
check(1, not short, f"substrate + gate arms at 120/120 events ({short or 'all complete'})")

# 2 -- NO symlink in a KEPT dir may resolve into a RETIRING dir.  This is the
# interlock the corrected census exists to feed; on the first (wrong) census it
# would have passed while d27fresh was in the release pool.
retset = set(RETIRE)
def owner_of(linkpath):
    tgt = os.readlink(linkpath)
    full = os.path.normpath(os.path.join(os.path.dirname(linkpath), tgt))
    own = None
    for part in full.split(os.sep):
        if EVT.match(part): own = part
    return own
dangle = []
for d in KEEP:
    p = os.path.join(WORK, d)
    for e in os.listdir(p):
        fp = os.path.join(p, e)
        if os.path.islink(fp):
            o = owner_of(fp)
            if o in retset: dangle.append(f"{d}/{e} -> {o}")
check(2, not dangle, f"no kept symlink resolves into a retiring dir "
                     f"({len(dangle)} would dangle{': ' + dangle[0] if dangle else ''})")

# 3 -- live-writer guard.  AGE IS NOT LIVENESS: most targets were written by
# this same tree hours ago, so a bare mtime threshold would refuse a correct
# round.  Sample mtimes, wait, re-sample, demand zero change -- plus no
# tree-scoped process and nothing from the live d39 round in the retire set.
def snapshot(dirs):
    out = {}
    for d in dirs:
        acc = []
        for cur, sub, files in os.walk(os.path.join(WORK, d)):
            for f in files:
                try: acc.append(os.path.getmtime(os.path.join(cur, f)))
                except OSError: pass
            if len(acc) > 4000: break
        out[d] = (len(acc), max(acc) if acc else 0)
    return out
sample = RETIRE[::max(1, len(RETIRE) // 400)]
before = snapshot(sample)
ps = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True).stdout
# Scoped to THIS tree on purpose: another user runs wire-cell out of
# /home/jjo/.../pdhd, which cannot touch pdvd/work.  A bare "any wire-cell"
# match would make this interlock unfalsifiable-by-noise.
busy = [l for l in ps.splitlines()
        if re.search(r"wire-cell|run_pr_evt|run_clus_evt|wcsonnet", l)
        and PDVD in l and not re.search(r"grep|plan_20260904", l)]
time.sleep(20)
after = snapshot(sample)
moved = [d for d in sample if before[d] != after[d]]
live_in_retire = [d for d in RETIRE if (UNIVERSE[d] or "").startswith(LIVE_PREFIX)]
check(3, not moved and not busy and not live_in_retire,
      f"no live writer over a 20s window (mtime-moved={moved[:3] or 'none'}, "
      f"procs={len(busy)}, d39-in-retire={live_in_retire or 'none'})")

# 4 -- the 6 broken symlinks are PRE-EXISTING (their target arm `stm1` was
# deleted outside the machinery, as 09-01's DROP dirs were).  Record them so
# the post-condition "no NEW broken links" is checkable rather than tripping on
# damage this round did not cause.
broken = []
for d in UNIVERSE:
    p = os.path.join(WORK, d)
    for e in os.listdir(p):
        fp = os.path.join(p, e)
        if os.path.islink(fp) and not os.path.exists(fp):
            broken.append(f"{d}/{e}")
pre_broken_ok = all(b.split("/")[0] in retset for b in broken)
check(4, pre_broken_ok, f"{len(broken)} pre-existing broken links, all inside "
                        f"retiring arms (target arm 'stm1' is already gone)")

# 5 -- every retiring arm has a verified record tar
have = set()
if os.path.isdir(REC):
    for cur, sub, files in os.walk(REC):
        for f in files:
            # The archive is MIXED after recompress_archive_20260904.py:
            # 2046 tarballs are .tar.zst and 1615 stayed .tar.gz (below the
            # size floor).  Both codecs are a valid record; accept either.
            if f.endswith(".tar.gz"): have.add(f[:-7])
            elif f.endswith(".tar.zst"): have.add(f[:-8])
miss = [d for d in RETIRE if d not in have]
check(5, not miss, f"record layer archived for {len(RETIRE) - len(miss)}/{len(RETIRE)} "
                   f"retiring arms (missing e.g. {miss[:2] or 'none'})")

# 6 -- keep/retire disjoint, and every substrate/gate/live arm is on keep side
must = set(SUBSTRATE) | set(GATE) | set(MISC)
onkeep = {UNIVERSE[d] for d in KEEP}
check(6, not (set(KEEP) & retset) and must <= onkeep,
      f"keep/retire disjoint; all {len(must)} substrate+gate arms on the keep side "
      f"({sorted(must - onkeep) or 'ok'})")

# 7 -- nothing outside the arm grammar is in the retire set
check(7, not (set(OUT_OF_SCOPE) & retset),
      f"{len(OUT_OF_SCOPE)} out-of-scope dirs (the <run6>_light<flash>_<tag> "
      f"light-reco space) untouched")

# ------------------------------------------------------------------ report --
def kb(paths):
    if not paths: return 0
    tot = 0
    for i in range(0, len(paths), 500):
        out = subprocess.run(["du", "-sk"] + paths[i:i + 500],
                             cwd=WORK, capture_output=True, text=True).stdout
        tot += sum(int(l.split("\t")[0]) for l in out.splitlines())
    return tot
rb, kbb = kb(RETIRE), kb(KEEP)
print(f"\nRETIRE {len(RETIRE)} dirs = {rb/1048576:.2f} GiB "
      f"({len({UNIVERSE[d] for d in RETIRE})} arms)")
print(f"KEEP   {len(KEEP)} dirs = {kbb/1048576:.2f} GiB "
      f"({len(KEEP_ARMS)} arms) + {len(OUT_OF_SCOPE)} out-of-scope light dirs")
grp = {}
for d in RETIRE:
    a = UNIVERSE[d]
    grp[d] = ("doc28-perf" if a.startswith(("d28", "r2q", "r2cfg", "prof", "heap")) else
              "doc31-steiner" if a.startswith("d31") else
              "doc32-endcover" if a.startswith("d32") else
              "doc34-35-metric" if a.startswith("d34") else
              "doc36-aniso" if a.startswith("d36") else
              "doc37-terminals" if a.startswith("d37") else
              "doc38-endtrim" if a.startswith("d38") else
              "doc23-27-early")
json.dump(dict(ARCHIVE=RETIRE, group=grp, KEEP=KEEP,
               KEEP_REASONS={**SUBSTRATE, **GATE, **MISC},
               OUT_OF_SCOPE=OUT_OF_SCOPE, bytes_retire_kb=rb, bytes_keep_kb=kbb,
               planned_at=time.strftime("%Y-%m-%dT%H:%M:%S")),
          open(os.path.join(STATE, "plan.json"), "w"), indent=1)
print(f"\nOVERALL: {'PASS' if not fails else 'FAIL ' + str(fails)}")
sys.exit(1 if fails else 0)
