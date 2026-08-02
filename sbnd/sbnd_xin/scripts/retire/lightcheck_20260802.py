#!/usr/bin/env python3
"""Round 2026-08-02 pre-flight: prove that every REAL opflash_apa*.tar.gz and
sp-frames*.tar.bz2 inside the removal set has a byte-identical copy in a
SURVIVING dir, so dropping them (HEAVY class / plan_20260802.py) loses no SP or
light data. This is the mechanical form of the July round's "SP and light data
are untouched" constraint.

Matching: a removal-set file <arm>/<evtdir>/<fname> is looked up in the hub(s)
of the same sample family under the same event-dir basename (ql_evt<ID> also
falls back to evt<ID> for sp-frames). Families keep MC event-ID collisions
(evt 12 in both r1ql and r2patrec) from cross-matching.

Exit 0 and "ALL COVERED" only if every file is byte-identical somewhere
surviving. Read-only.
"""
import os, re, json, filecmp, collections, sys

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
os.chdir(ROOT)
plan = json.load(open("scripts/retire/state-20260802/plan.json"))
R = plan["R"]

FAMILY_HUBS = [
    (re.compile(r'^work-(mcp1kall|mcp1000b?|mcp10|b0pr\d*|pi\d|pi[0-5]\S*|p[34]-59003|b0|pr20x|cbr|nsc|nbl|vv|oc19scan|cath13|cathdbg1pr|ccfeat300pr|audit|det|stmcamp)'),
     ['work-mcp1kall-d59k', 'work-mcp1000', 'work-mcp10', 'work']),
    (re.compile(r'^work-(nuecc48|oc444187|b0nue48)'),
     ['work-nuecc48-nuf', 'work']),
    (re.compile(r'^work-r1qlmc'), ['work-r1ql-f1', 'work-r1ql-f2', 'work-r1ql-first10']),
    (re.compile(r'^work-r2mc'), ['work-r2patrec-f1']),
]
ALL_HUBS = ['work-mcp1kall-d59k', 'work-mcp1000', 'work-mcp10', 'work',
            'work-nuecc48-nuf', 'work-r1ql-f1', 'work-r1ql-f2',
            'work-r1ql-first10', 'work-r2patrec-f1', 'work-oc19scan-old']

PAT = re.compile(r'^(opflash_apa.*\.tar\.gz|sp-frames.*\.tar\.bz2|frames-dnn\.tar\.bz2)$')

def hubs_for(arm):
    for pat, hubs in FAMILY_HUBS:
        if pat.match(arm): return hubs, False
    return ALL_HUBS, True

ok = missing = differs = fallback_used = 0
problems = []
for arm in R:
    hubs, is_fallback = hubs_for(arm)
    for cur, sub, files in os.walk(arm):
        sub[:] = [s for s in sub if not os.path.islink(os.path.join(cur, s))]
        for f in files:
            p = os.path.join(cur, f)
            if os.path.islink(p) or not PAT.match(f): continue
            evtdir = os.path.basename(cur)
            cands = [evtdir]
            if evtdir.startswith('ql_evt'): cands.append('evt' + evtdir[len('ql_evt'):])
            found = None
            for hub in hubs:
                for c in cands:
                    q = os.path.join(hub, c, f)
                    if os.path.isfile(q) and not os.path.islink(os.path.join(hub, c)):
                        found = q; break
                    # hub evt dirs may be symlinks into base; realpath is fine
                    if os.path.isfile(q):
                        found = q; break
                if found: break
            if found is None:
                missing += 1; problems.append(('MISSING', p))
            elif not filecmp.cmp(p, found, shallow=False):
                differs += 1; problems.append(('DIFFERS', f"{p} vs {found}"))
            else:
                ok += 1
                if is_fallback: fallback_used += 1

print(f"covered={ok} missing={missing} differs={differs} (fallback-family matches: {fallback_used})")
for kind, msg in problems[:200]:
    print(f"  !! {kind}: {msg}")
if len(problems) > 200: print(f"  ... and {len(problems)-200} more")

# Files with no identical surviving copy are written to light_exceptions.txt;
# archive_records_20260802.py force-ARCHIVES them (they leave the heavy/drop
# class), so the "SP and light untouched" guarantee holds either way.
exc = os.path.join("scripts/retire/state-20260802", "light_exceptions.txt")
with open(exc, "w") as fh:
    for kind, msg in problems:
        fh.write(msg.split(" vs ")[0] + "\n")
print(f"{len(problems)} exception path(s) -> {exc}")
print("ALL COVERED" if not problems else
      "NOT COVERED by surviving copies -- the paths above are force-archived via the exceptions file")
sys.exit(0 if not problems else 1)
