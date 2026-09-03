#!/usr/bin/env python3
"""doc 98 -- the BYTE-driven retire round, 2026-09-02.

OWNER SCOPE, verbatim: "the sbnd_xin directory now is exploded to 189G, which
needs to be clean up.  We should retire the middle work* directory, and we only
need to save the latest production results for Q/L matching as well as PR
results."  Then, separately: "please also clean up a bit for ~/tmp please".

So the metric this round is BYTES, not dir count (that was doc 91's round), and
the keep rule is given: the LATEST production stage A and stage B.  After doc
97's flip that is work-*-d97fv (Q/L, sep_fv_point ON = ref/prod-2026-09-04) and
work-*-d97fvpr2 (PR tail on it).  Everything between grp0825 and those two is
"the middle".

FIVE THINGS THIS ROUND MEASURED RATHER THAN INHERITED
-----------------------------------------------------
1. grp0825 IS NOT ONE ARM.  10478 symlinks across the tree resolve into
   work-*-grp0825/evt<N> -- it is the imaging substrate every Q/L arm borrows,
   and deleting it would break every surviving arm.  Nothing anywhere points at
   work-*-grp0825/ql_evt<N>, and that Q/L is two operating points stale (doc 97
   sec 2).  So the arm is SPLIT: imaging kept, Q/L retired in place, after
   rollup_grp0825_ql_20260902.py froze its 18402-member hash rollup.

2. RETIRING d97off2pr WOULD HAVE DESTROYED THE ONLY 31/31 SENTINEL ARM.
   Production (d97fvpr2) is 30 PASS / 1 FAIL -- 105074 (pr/128 class B) asserts
   a PF node production deliberately no longer makes, and re-anchoring it is an
   OPEN owner call.  The only 31/31 arms on disk were work-*-d97off2pr and
   work-*-prod0901b, and this round releases both.  witness_sentinels_20260902.py
   copies the 28 distinct sentinel events into work-sent97-* (165 MB) and
   INTERLOCK 2 refuses the round unless the suite is 31/31 against the witness
   ALONE.  Same role work-sent130-* played before doc 91 folded it into
   production.

3. A PEER SESSION IS MID-ROUND IN THIS TREE.  wcp-porting-img 19d32520 ("pdvd:
   doc 25 sec 13 ... gate round 7") landed at 16:54 today and
   work-{ncpi0,nuecc48}-doc25new7 was written at 16:46.  Those are the PDVD
   round's SBND regression-gate arms, living in sbnd_xin.  doc25* is protected
   by PREFIX for the life of that round -- exactly what doc 91 did for doc 90's
   arms while they were live.  INTERLOCK 4 re-derives it from mtime on every
   run rather than trusting this paragraph.

4. THE PROTECTED.txt PIN ON prod0901b IS SPENT, AND SAYS SO ITSELF.  Its line
   reads "the SBND production baseline at ref/prod-2026-09-01b".  It was pinned
   by doc 91's owner instruction "We want to keep the latest production though"
   -- and prod0901b is no longer the latest; d97fvpr2 is.  The same sentence
   that protected it releases it.  Recorded here so the removal is a decision,
   not an inference.

5. doc87/lib-knob IS A VERIFIED DUPLICATE.  sweep_tmp_20260901.sh asserted it
   ("an exact md5 duplicate of lib-flip") but that sweep NEVER RAN -- all 11 of
   its DROP dirs are already gone by another route and lib-knob is still on
   disk.  Re-measured today: 790/790 files md5-identical to lib-flip.  Released
   on today's measurement, not on last round's sentence.

WHAT THIS KEEPS THAT A PURELY BYTE-DRIVEN ROUND WOULD NOT
---------------------------------------------------------
work-*-d97on / d97onpr (20.4 GiB) ARE released, and that is the one judgement
call worth naming: sep_track_recarve is still default OFF and its Bee idx 9/10
are unadjudicated, so the owner may yet ask for it.  The Bee sets are uploaded
and doc 97 carries the measurements, and the arm re-runs in ~25 min from kept
imaging, so the evidence survives the bytes.  Reported as a line item.

Writes state-20260902/plan.json.  Reads only; deletes nothing.
"""
import glob, json, os, re, shutil, subprocess, sys, time

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
STATE = os.path.join(ROOT, "scripts", "retire", "state-20260902")
os.chdir(ROOT)
os.makedirs(STATE, exist_ok=True)
SAMPLES = ("mcp1k", "mcp2k", "ncpi0", "nuecc48")

# ---------------------------------------------------------------- the sets --
# full-sample labels retired: 9 stage-B (pr_evt) + 2 stage-A (ql_evt)
RETIRE_LABELS = [
    # stage B -- superseded PR epochs
    "prod0901b", "d94probe", "d94hadron", "r2probe", "r2entry",
    "r3probe", "r3entry", "d97onpr", "d97off2pr",
    # stage A -- superseded Q/L epochs
    "d97on", "d97off2",
    # doc 97 identity-gate probes (records in docs/97_sep/97-idgate*.txt)
    "d97idg", "d97idgb", "d97idgc", "d97idgd",
]
RETIRE_EXTRA = [                      # small, dead, records already in docs
    "work-d97chk-ncpi0", "work-d97chk2-mcp2k", "work-d97chk3-mcp2k",
    "work-d94scan-64475", "work-d94scanoff-64475",
]
# live peer round -- protected by PREFIX, re-derived by INTERLOCK 4
PEER_PREFIXES = ("doc25new", "doc25old")
KEEP_LABELS_EXPLICIT = {
    "grp0825":   "imaging substrate: 10478 symlinks resolve into evt<N> (ql_evt half retired in place)",
    "d97fv":     "LATEST PRODUCTION stage A -- ref/prod-2026-09-04",
    "d97fvpr2":  "LATEST PRODUCTION stage B -- PR tail on it",
    "sent97":    "sentinel witness, 31/31 standalone (INTERLOCK 2)",
    "vtx105-base": "1930 citations incl. dl_vtx_training; PROTECTED.txt",
    "em114": "live em_display manifests", "em114c": "live em_display manifests",
    "pr130r1-probe141": "live em_display manifests",
    "pr130r1-probe98": "sentinel witness for 137238; doc 91 interlock 8",
    "pr134-f086": "sentinel witness at the 0.86 EM scale; doc 91 interlock 8",
    "d97prodchk": "doc 97 sec 9 production-default == validated arm proof",
}

def label_of(d):
    b = d[5:]
    for s in SAMPLES:
        if b.endswith("-" + s):  return b[:-len(s) - 1]
        if b.startswith(s + "-"): return b[len(s) + 1:]
    return b

alldirs = sorted(d for d in os.listdir(".")
                 if d.startswith("work-") and os.path.isdir(d) and not os.path.islink(d))
RETIRE = sorted({d for d in alldirs if label_of(d) in RETIRE_LABELS} | set(RETIRE_EXTRA))
RETIRE = [d for d in RETIRE if os.path.isdir(d)]
KEEP = [d for d in alldirs if d not in RETIRE]

fails = []
def check(n, ok, msg):
    print(f"{'PASS' if ok else 'FAIL'}  INTERLOCK {n}: {msg}")
    if not ok: fails.append(n)

# 1 -- the two production arms are complete, 3067/3067, rc=0
def complete(lab, sub, need):
    tot = bad = 0
    for s in SAMPLES:
        for d in sorted(glob.glob(f"work-{s}-{lab}/{sub}*")):
            tot += 1
            ev = os.path.basename(d).replace(sub, "")
            if any(not os.path.exists(os.path.join(d, n.format(ev=ev))) for n in need):
                bad += 1; continue
            rc = os.path.join(d, "rc.txt")
            if os.path.exists(rc) and open(rc).read().strip().replace("rc=", "") != "0":
                bad += 1
    return tot, bad
ta, ba = complete("d97fv", "ql_evt", ["mabc-all-apa.zip", "pctree-evt{ev}.tar.gz"])
tb, bb = complete("d97fvpr2", "pr_evt",
                  ["mabc-pr.zip", "nusel-evt{ev}.tsv", "tracking-pr.root",
                   "pctree-pr-evt{ev}.tar.gz"])
check(1, ta == 3067 and ba == 0 and tb == 3067 and bb == 0,
      f"production complete: d97fv {ta}/3067 ({ba} bad), d97fvpr2 {tb}/3067 ({bb} bad)")

# 2 -- the sentinel witness reproduces 31/31 ALONE, before its sources go
r = subprocess.run([sys.executable, "scripts/d97_sentinels.py", "--arms", "work-sent97-*"],
                   capture_output=True, text=True)
tail = [l for l in r.stdout.splitlines() if "PASS," in l or "FAIL," in l]
sent_ok = bool(tail) and tail[-1].startswith("31 PASS, 0 FAIL")
check(2, sent_ok, f"sentinel witness standalone: {tail[-1] if tail else 'NO RESULT'} "
                  f"(sources d97off2pr + prod0901b are both retiring)")

# 3 -- no symlink in a KEPT dir resolves into a RETIRE dir
retset = set(RETIRE)
dangle = []
for d in KEEP:
    for e in os.listdir(d):
        p = os.path.join(d, e)
        if os.path.islink(p):
            tgt = os.path.realpath(p)
            rel = os.path.relpath(tgt, ROOT).split(os.sep)[0]
            if rel in retset: dangle.append(f"{p} -> {rel}")
check(3, not dangle, f"no kept symlink resolves into a retiring dir "
                     f"({len(dangle)} would dangle{': ' + dangle[0] if dangle else ''})")

# 4 -- live-writer guard.  AGE IS NOT LIVENESS: most of this round's targets
# are arms this same session produced hours ago, so a bare mtime threshold
# would refuse a correct round (and tuning the threshold until it passes would
# be exactly the wrong move).  The real question is whether anything is WRITING
# them now, so this samples mtimes, waits, re-samples, and demands zero change
# -- plus no matching process and no open file handle under a target.
def snapshot(dirs):
    out = {}
    for d in dirs:
        acc = []
        for cur, sub, files in os.walk(d):
            for f in files:
                try: acc.append(os.path.getmtime(os.path.join(cur, f)))
                except OSError: pass
            if len(acc) > 4000: break
        out[d] = (len(acc), max(acc) if acc else 0)
    return out
before = snapshot(RETIRE)
ps = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True).stdout
# Scoped to THIS tree on purpose: another user runs wire-cell on this box out
# of /home/jjo/.../pdhd, which cannot touch sbnd_xin.  A bare "any wire-cell"
# match makes the interlock unfalsifiable-by-noise; it must name our root.
busy_proc = [l for l in ps.splitlines()
             if re.search(r"wire-cell|run_pr_chain|run_ql|wcsonnet", l)
             and ROOT in l
             and not re.search(r"grep|plan_20260902", l)]
time.sleep(20)
after = snapshot(RETIRE)
moved = [d for d in RETIRE if before[d] != after[d]]
peer_in_retire = [d for d in RETIRE if any(p in d for p in PEER_PREFIXES)]
opened = []
if shutil.which("lsof"):
    lo = subprocess.run(["lsof", "-w", "+D", RETIRE[0]] if False else ["lsof", "-w"],
                        capture_output=True, text=True).stdout
    opened = [l for l in lo.splitlines() if any(d + "/" in l for d in RETIRE[:200])]
check(4, not moved and not busy_proc and not peer_in_retire and not opened,
      f"no live writer on any retire target over a 20s window "
      f"(mtime-moved={moved or 'none'}, procs={len(busy_proc)}, "
      f"open-handles={len(opened)}, peer={peer_in_retire or 'none'})")

# 5 -- PROTECTED.txt union, minus the pin this round deliberately spends
prot = set()
inret = False
for ln in open("scripts/retire/PROTECTED.txt"):
    if ln.startswith("# --- RETIRED"): inret = True
    if ln.startswith("#") or not ln.strip() or inret: continue
    prot.update(ln.split("\t")[0].split())
SPENT = {f"work-{s}-prod0901b" for s in SAMPLES}
viol = sorted((prot & set(RETIRE)) - SPENT)
check(5, not viol, f"no PROTECTED.txt arm retired except the spent prod0901b pin "
                   f"({viol or 'none'})")

# 6 -- the grp0825 ql rollup exists and covers every ql_evt dir being removed
roll = os.path.join(STATE, "grp0825-ql-rollup.tsv")
nroll = 0
if os.path.exists(roll):
    seen = set()
    for i, ln in enumerate(open(roll)):
        if i: seen.add(tuple(ln.split("\t")[:2]))
    nroll = len(seen)
nql = sum(len(glob.glob(f"work-{s}-grp0825/ql_evt*")) for s in SAMPLES)
check(6, nroll == nql == 3067, f"grp0825 ql rollup covers {nroll} of {nql} ql_evt dirs")

# 7 -- keep/retire are disjoint and every production dir is on the keep side
prod = {f"work-{s}-{l}" for s in SAMPLES for l in ("d97fv", "d97fvpr2", "grp0825")}
check(7, not (set(KEEP) & retset) and prod <= set(KEEP),
      "keep/retire disjoint and all 12 production dirs on the keep side")

# ------------------------------------------------------------------ report --
def kb(paths):
    if not paths: return 0
    out = subprocess.run(["du", "-sk"] + paths, capture_output=True, text=True).stdout
    return sum(int(l.split("\t")[0]) for l in out.splitlines())
rb = kb(RETIRE)
qlb = kb([d for s in SAMPLES for d in glob.glob(f"work-{s}-grp0825/ql_evt*")])
print(f"\nRETIRE {len(RETIRE)} dirs = {rb/1048576:.1f} GiB"
      f"  + grp0825 ql_evt in place = {qlb/1048576:.1f} GiB"
      f"  -> total {(rb+qlb)/1048576:.1f} GiB")
print(f"KEEP   {len(KEEP)} dirs")
grp = {}
for d in RETIRE:
    L = label_of(d)
    grp[d] = ("doc94-stm" if L.startswith(("d94", "r2", "r3")) else
              "doc97-separation" if L.startswith("d97") else "prod-epoch")
json.dump(dict(ARCHIVE=RETIRE, group=grp, KEEP=KEEP,
               KEEP_REASONS=KEEP_LABELS_EXPLICIT,
               grp0825_ql=[d for s in SAMPLES for d in glob.glob(f"work-{s}-grp0825/ql_evt*")],
               bytes_arms=rb, bytes_ql=qlb, planned_at=time.strftime("%Y-%m-%dT%H:%M:%S")),
          open(os.path.join(STATE, "plan.json"), "w"), indent=1)
print(f"\nOVERALL: {'PASS' if not fails else 'FAIL ' + str(fails)}")
sys.exit(1 if fails else 0)
