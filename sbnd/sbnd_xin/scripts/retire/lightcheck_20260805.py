#!/usr/bin/env python3
"""Round 2026-08-05 pre-flight: prove that every REAL opflash_apa*.tar.gz and
sp-frames*.tar.bz2 inside the removal set has a byte-identical copy in a
SURVIVING dir, so dropping them (HEAVY class / plan_20260805.py) loses no light
or SP data.  This is the blocking check -- the 08-02 round found 8 differing out
of 25112, so a handful of unique variants is the expected base rate, not a
surprise.  Non-matches are written to light_exceptions.txt and force-ARCHIVED by
archive_records_20260805.py.

Fork of lightcheck_20260803.py.  Differences:
  - reads state-20260805/plan.json (three tiers; R is their union);
  - hub families updated for the pr/23-25 era and the valfast roots;
  - a tier-V arm that is DROPPED without an archive cannot fall back on the
    exceptions file, so any exception inside vf-drop is reported as a hard
    BLOCKER rather than a force-archive candidate.

Read-only.
"""
import os, re, json, filecmp, collections, sys

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
STATE = os.path.join(ROOT, "scripts", "retire", "state-20260805")
os.chdir(ROOT)
plan = json.load(open(os.path.join(STATE, "plan.json")))
R = plan["R"]
ARCHIVE = set(plan["ARCHIVE"])            # arms that CAN take a force-archive

# Sample family -> the surviving hubs that may hold the identical copy.  The
# family split keeps MC event-ID collisions (evt 12 exists in both the r1ql and
# r2patrec samples) from producing a false match.
FAMILY_HUBS = [
    # mcp1k / MC hand-scan families: the valfast mcp1k roots, the mcp1kall
    # subset arms, and the pr/23-25 arms drawn from the 1000-event MC set.
    (re.compile(r'^work-(vfmcp1k-|mcp1kall-|mcp1000b?-|mcp10-|oc19scan-|pr22gap-)'),
     ['work-mcp1kall-d59k', 'work-mcp1000', 'work-mcp10', 'work']),
    # 48-event nueCC data family
    (re.compile(r'^work-(vfnuecc48-|nuecc48-|oe|pr2[6-9]|pr3[0-7]|p1[4-7]|r9|r10|tfix388|vtxfit|dqdx388|prdisp|prdict|spflag)'),
     ['work-nuecc48-nuf', 'work-nuecc48-prod0803', 'work-nuecc48-0804', 'work']),
    (re.compile(r'^work-(vfr1qlmc-|r1qlmc-)'),
     ['work-r1ql-f1', 'work-r1ql-f2', 'work-r1ql-first10']),
    (re.compile(r'^work-(vfr2mc-|r2mc-)'),
     ['work-r2patrec-f1']),
]
# Refreshed 2026-08-05: work-nuecc48-base, work-mrgB-post, the cath01 pair and
# work-r2patrec-f1-nobw were all retired in the 08-02/08-03 rounds.  A hub list
# naming a dir that no longer exists degrades silently to "no candidate found"
# and reports a false MISSING, so it is refreshed against the tree every round.
ALL_HUBS = ['work-mcp1kall-d59k', 'work-mcp1000', 'work-mcp10', 'work',
            'work-nuecc48-nuf', 'work-nuecc48-prod0803', 'work-nuecc48-0804',
            'work-r1ql-f1', 'work-r1ql-f2', 'work-r1ql-first10',
            'work-r2patrec-f1', 'work-oc19scan-old', 'work-nuecc48-prsmoke2']

PAT = re.compile(r'^(opflash_apa.*\.tar\.gz|sp-frames.*\.tar\.bz2|sbnd-sp-frames.*\.tar\.bz2|frames-dnn\.tar\.bz2)$')

def hubs_for(arm):
    for pat, hubs in FAMILY_HUBS:
        if pat.match(arm): return hubs, False
    return ALL_HUBS, True

ok = missing = differs = fallback_used = 0
problems = []
percheck = collections.Counter()
for arm in sorted(R):
    hubs, is_fallback = hubs_for(arm)
    for cur, sub, files in os.walk(arm):
        sub[:] = [s for s in sub if not os.path.islink(os.path.join(cur, s))]
        for f in files:
            p = os.path.join(cur, f)
            if os.path.islink(p) or not PAT.match(f): continue
            percheck[arm] += 1
            evtdir = os.path.basename(cur)
            cands = [evtdir]
            if evtdir.startswith('ql_evt'): cands.append('evt' + evtdir[len('ql_evt'):])
            if evtdir.startswith('evt'):    cands.append('ql_' + evtdir)
            found = None
            for hub in hubs:
                for c in cands:
                    q = os.path.join(hub, c, f)
                    if os.path.isfile(q): found = q; break
                if found: break
            if found is None:
                missing += 1; problems.append(('MISSING', arm, p))
            elif not filecmp.cmp(p, found, shallow=False):
                differs += 1; problems.append(('DIFFERS', arm, f"{p} vs {found}"))
            else:
                ok += 1
                if is_fallback: fallback_used += 1

print("=== light/SP files checked per arm ===")
for arm, n in sorted(percheck.items()):
    print(f"  {arm:38s} {n:5d}")
print(f"\ncovered={ok} missing={missing} differs={differs} "
      f"(fallback-family matches: {fallback_used})  total={sum(percheck.values())}")

for kind, arm, msg in problems[:200]:
    print(f"  !! {kind}: {msg}")
if len(problems) > 200: print(f"  ... and {len(problems)-200} more")

# An exception inside an arm that is going to be ARCHIVED is fine -- the file is
# force-archived and nothing is lost.  An exception inside a tier-V arm that is
# DROPPED whole has nowhere to go: that is a blocker.
blockers = [(k, a, m) for k, a, m in problems if a not in ARCHIVE]
exc = os.path.join(STATE, "light_exceptions.txt")
with open(exc, "w") as fh:
    for kind, arm, msg in problems:
        fh.write(msg.split(" vs ")[0] + "\n")
print(f"\n{len(problems)} exception path(s) -> {exc}")

if blockers:
    print(f"\n!! BLOCKER: {len(blockers)} exception(s) live in DROP-whole tier-V arms, "
          f"which get no archive:")
    for kind, arm, msg in blockers[:50]:
        print(f"     {kind} {msg}")
    print("   move those arms into the ARCHIVE set (tier A in plan_20260805.py) "
          "or promote them to KEEP before proceeding.")
    sys.exit(2)

print("\nALL COVERED" if not problems else
      "\nNOT byte-identical everywhere -- the paths above are force-archived via the "
      "exceptions file (all inside archived arms, so nothing is lost)")
sys.exit(0)
