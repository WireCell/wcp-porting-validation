#!/usr/bin/env python3
"""Retirement round 2026-08-03: three-tier removal set, archive footprint,
dangling-link dry run, and the four safety asserts.

Fork of plan_20260802.py.  Do NOT re-run that script for this round -- its
survivor list is hardcoded to the 08-02 KEEP set and would treat everything
added since (work-r1ql-*, work-r2patrec-*, the cath01 pair, work-mrgB-post and
the owner's live prod0803 campaign) as a removal candidate.

Differences from the 08-02 fork:
  - THREE tiers, with different dispositions (the 08-02 round had one):
      tier V  valfast arms       -> DROP whole, no archive.  valfast/README.md
                                    ("valfast arms are transient.  Record the
                                    compare summary in the round doc, then
                                    DELETE the roots.")  Exception: the three
                                    arms in VF_UNCITED below have no doc or
                                    script citation at all, so their gate result
                                    exists nowhere else -- they get the tier-1
                                    record-layer treatment instead.
      tier 1  pr/23..pr/25 arms  -> archive record layer, then delete.
      tier 2  merge/test gates   -> archive record layer, then delete.
  - a PROTECTED set that is excluded explicitly rather than by falling outside a
    glob: the owner's live campaign.  os.walk() never descends into it.
  - classification asserts: every work* dir lands in exactly one of
    PROTECTED / KEEP / tierV / tier1 / tier2, and an unclassified dir is a hard
    error (the 08-02 fork silently binned unknowns as 'probes' -- fine when the
    removal set was "everything not KEEP", wrong when tiers have different
    dispositions).
  - three extra safety asserts beyond the 08-02 SP-data scan: no git-tracked
    file, no nusel_labels/ql_labels/decisions* dir, and no real SP frame inside
    any candidate.

Writes state to scripts/retire/state-20260803/plan.json and the three tier lists
to scripts/retire/tier{V,1,2}_20260803.txt.  Read-only w.r.t. work-*.
"""
import os, re, json, subprocess, collections, sys

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
SCR  = os.path.join(ROOT, "scripts", "retire")
STATE = os.environ.get("RETIRE_STATE", os.path.join(SCR, "state-20260803"))
os.makedirs(STATE, exist_ok=True)
os.chdir(ROOT)

dirs = sorted(d for d in os.listdir('.')
              if d.startswith('work') and os.path.isdir(d) and not os.path.islink(d))

# ---------------------------------------------------------------- PROTECTED
# The owner's 2026-08-03 campaign.  Named explicitly so a future glob change
# cannot quietly pull them in; also excluded from every walk below.
PROTECTED = {'work-nuecc48-prod0803', 'work-vfnuecc48-prod0803'}

# The tree is live: work-pr116962-nocosmicveto appeared WHILE this script was
# being written, created by a runner reading work-nuecc48-prod0803.  So an
# unclassified dir is not necessarily a classification bug -- it may be work in
# flight.  Rule: unclassified AND touched within FRESH_HOURS => auto-PROTECT and
# say so loudly; unclassified AND stale => hard error, classify it by hand.
FRESH_HOURS = float(os.environ.get("RETIRE_FRESH_HOURS", "6"))

# ---------------------------------------------------------------- KEEP
SURVIVORS_EXACT = {
    # BASE / HUB -- never delete (regenerating work-mcp1000 alone is a
    # 2000-event imaging run)
    'work', 'work-mcp1000', 'work-mcp10', 'work-mcp1kall-d59k',
    # doc pr/12 cathode-crossing pair
    'work-mcp1kall-cath01', 'work-nuecc48-cath01',
    # the 48-event nueCC campaign roots behind docs pr/1..pr/10
    'work-nuecc48-base', 'work-nuecc48-nuf', 'work-nuecc48-prsmoke',
    'work-nuecc48-prsmoke2',
    # git-tracked hand-scan label stub
    'work-stmcamp-d66new',
    # pr/22 exhibit parent, materialized 2026-08-02
    'work-oc19scan-old',
    # current-production 48-event baseline, held out of tier 2 pending
    # work-nuecc48-prod0803 completing 48/48 (see docs/work-tags.md)
    'work-mrgB-post',
}
SURVIVOR_PREFIXES = ('work-r1ql-', 'work-r2patrec-', 'work-pr22gap-')

def survivor(d):
    return d in SURVIVORS_EXACT or d.startswith(SURVIVOR_PREFIXES)

# ---------------------------------------------------------------- tiers
# tier V: every valfast root, in both spellings (work-vf<sample>-<tag> for the
# PR out-roots, work-<sample>-vf<tag> for the nusel roots the -full arms make).
TIER_V = re.compile(r'^work-(vf(mcp1k|nuecc48|r1qlmc|r2mc)-.+|(mcp1kall|nuecc48|r1qlmc|r2mc)-vf.+)$')

# tier V arms with ZERO citation in docs/, valfast/, *.py, *.sh -- their gate
# result is not recorded anywhere else, so archive rather than drop.
VF_UNCITED = {'work-vfmcp1k-pr24i0', 'work-vfmcp1k-pr24r3a', 'work-vfnuecc48-pr24r3a'}

# tier 2: the master-merge (doc 69) and test-round (doc 70) gate arms.
TIER_2 = {'work-mrgA-pre', 'work-mrgdet-r1', 'work-mrgdet-r2', 'work-d70-eb'}

# tier 1 groups -- archive layout + the work-tags.md table.
GROUPS_1 = [
    ('poc48', r'^work-(poc48c?-\w+|nuecc48-poc0)$'),
    ('pr23',  r'^work-(pr23\S*|mcp1kall-pr23\w+)$'),
    ('pr24',  r'^work-pr24\S*$'),
    ('pr25',  r'^work-pr25\S*$'),
]

def tier(d):
    if d in PROTECTED:   return 'PROTECTED'
    if survivor(d):      return 'KEEP'
    if TIER_V.match(d):  return 'V'
    if d in TIER_2:      return '2'
    for _, pat in GROUPS_1:
        if re.match(pat, d): return '1'
    return 'UNCLASSIFIED'

def group(d):
    t = tier(d)
    if t == 'V':  return 'vf-uncited' if d in VF_UNCITED else 'vf-drop'
    if t == '2':  return 'mergegate'
    for g, pat in GROUPS_1:
        if re.match(pat, d): return g
    return t

import time
now = time.time()
bytier = collections.defaultdict(list)
autoprot = []
for d in dirs:
    t = tier(d)
    if t == 'UNCLASSIFIED' and (now - os.path.getmtime(d)) < FRESH_HOURS * 3600:
        autoprot.append(d)
        t = 'PROTECTED'
    bytier[t].append(d)

if autoprot:
    print(f"!! {len(autoprot)} unclassified dir(s) touched within {FRESH_HOURS:g} h "
          f"-- treating as WORK IN FLIGHT and PROTECTING them:")
    for d in autoprot:
        age = (now - os.path.getmtime(d)) / 3600
        print(f"     {d}   (last touched {age:.1f} h ago)")
    print("   if one of these is actually retirable, classify it explicitly and re-run.\n")

if bytier['UNCLASSIFIED']:
    print("!! UNCLASSIFIED work dirs, none of them recent -- classify them by hand "
          "before running anything else:")
    for d in bytier['UNCLASSIFIED']:
        print(f"     {d}")
    sys.exit(2)

V, T1, T2 = bytier['V'], bytier['1'], bytier['2']
K, P = bytier['KEEP'], bytier['PROTECTED']
R = sorted(V + T1 + T2)                      # the full removal set
assert len(dirs) == len(R) + len(K) + len(P), "tiers do not partition work*"
# ARCHIVE set: tiers 1+2 get a record layer, tier V does not -- except the
# uncited arms, whose result would otherwise be lost outright.
ARCHIVE = sorted(T1 + T2 + [d for d in V if d in VF_UNCITED])

# ---------------------------------------------------------------- footprint
HEAVY = (re.compile(r'^pctree.*\.tar\.gz$'), re.compile(r'^mabc.*\.zip$'),
         re.compile(r'^calib-evt.*\.json(\.gz)?$'), re.compile(r'.*\.npz$'),
         re.compile(r'^clusters-apa.*\.tar\.gz$'),
         re.compile(r'^opflash_apa.*\.tar\.gz$'))
def is_heavy(f): return any(p.match(f) for p in HEAVY)

SPDATA = re.compile(r'^(sp-frames.*\.tar\.bz2|sbnd-sp-frames.*\.tar\.bz2|frames-dnn\.tar\.bz2)$')
RECORD_DIR = re.compile(r'^(nusel_labels|ql_labels|decisions.*)$')

per = {}
spdata_hits, label_hits = [], []
for d in R:
    tot = keep = nk = nh = 0
    for cur, sub, files in os.walk(d):
        sub[:] = [s for s in sub if not os.path.islink(os.path.join(cur, s))]
        for s in sub:
            if RECORD_DIR.match(s): label_hits.append(os.path.join(cur, s))
        for f in files:
            p = os.path.join(cur, f)
            if os.path.islink(p): continue
            try: sz = os.path.getsize(p)
            except OSError: continue
            tot += sz
            if SPDATA.match(f): spdata_hits.append(p)
            if is_heavy(f): nh += 1
            else: keep += sz; nk += 1
    per[d] = dict(tot=tot, keep=keep, nk=nk, nh=nh)

print("=== TIERS (review before archiving) ===")
for t, label in (('V', 'tier V  valfast (DROP whole, no archive)'),
                 ('1', 'tier 1  pr/23-25 campaign (archive, then delete)'),
                 ('2', 'tier 2  merge/test gates (archive, then delete)')):
    tot = sum(per[d]['tot'] for d in bytier[t])
    print(f"\n[{label}]  {len(bytier[t])} dirs, {tot/2**30:.2f} GiB")
    for d in sorted(bytier[t]):
        mark = "  <- ARCHIVED (no citation anywhere)" if d in VF_UNCITED else ""
        print(f"    {d:38s} {per[d]['tot']/2**20:8.0f} MB{mark}")
print(f"\n[KEEP] {len(K)} dirs   [PROTECTED] {len(P)} dirs: {' '.join(sorted(P))}")

print("\n=== ARCHIVE FOOTPRINT (real files except pctree/mabc/calib/npz/clusters/opflash) ===")
stat = collections.defaultdict(lambda: [0, 0, 0])
for d in ARCHIVE:
    s = stat[group(d)]
    s[0] += per[d]['tot']; s[1] += per[d]['keep']; s[2] += per[d]['nk']
for g in sorted(stat):
    t, k, nk = stat[g]
    print(f"{g:14s} total {t/2**30:6.2f} GiB  archive {k/2**20:8.1f} MiB ({nk} files)  reclaim {(t-k)/2**30:6.2f} GiB")
adrop = sum(per[d]['tot'] for d in V if d not in VF_UNCITED)
print(f"{'vf-drop':14s} total {adrop/2**30:6.2f} GiB  archive      0.0 MiB (0 files)  reclaim {adrop/2**30:6.2f} GiB")
T = sum(per[d]['tot'] for d in R)
Kb = sum(stat[g][1] for g in stat)
print(f"{'TOTAL':14s} total {T/2**30:6.2f} GiB  archive {Kb/2**20:8.1f} MiB  reclaim {(T-Kb)/2**30:6.2f} GiB")

# ---------------------------------------------------------------- safety asserts
fail = 0

print("\n=== ASSERT 1: no real SP frame inside the removal set (owner's standing constraint) ===")
if not spdata_hits: print("0 -- PASS")
else:
    fail += 1
    for p in spdata_hits[:50]: print(f"  !! {p}")

print("\n=== ASSERT 2: no hand-scan / decision record dir inside the removal set (M13) ===")
if not label_hits: print("0 -- PASS")
else:
    fail += 1
    for p in label_hits[:50]: print(f"  !! {p}")

print("\n=== ASSERT 3: no git-tracked file inside the removal set ===")
tracked = subprocess.run(['git', 'ls-files', '-z', '--'] + R,
                         capture_output=True, text=True).stdout.split('\0')
tracked = [t for t in tracked if t]
if not tracked: print("0 -- PASS")
else:
    fail += 1
    for t in tracked[:50]: print(f"  !! {t}")

# ---------------------------------------------------------------- dangling links
# Walk EVERYTHING outside the removal set (including PROTECTED, which must not
# depend on a candidate) and resolve every symlink.
Rset = set(R)
bad = collections.Counter()
nlinks = 0
top = [e for e in sorted(os.listdir('.')) if e not in Rset and not e.startswith('.')]
for root in top:
    if os.path.islink(root):
        m = re.search(r'sbnd_xin/(work[^/]*)', os.readlink(root))
        if m and m.group(1) in Rset: bad[(root, m.group(1))] += 1
        continue
    if not os.path.isdir(root): continue
    for cur, sub, files in os.walk(root, followlinks=False):
        for name in sub + files:
            p = os.path.join(cur, name)
            if os.path.islink(p):
                nlinks += 1
                m = re.search(r'sbnd_xin/(work[^/]*)', os.readlink(p))
                if m and m.group(1) in Rset: bad[(root, m.group(1))] += 1

print(f"\n=== ASSERT 4: dangling-link dry run ({nlinks} symlinks outside removal set scanned) ===")
if not bad: print("0 -- PASS, no surviving symlink resolves into any removal candidate")
else:
    fail += 1
    for (s, t), c in bad.most_common(): print(f"  !! {c:6d}  {s} -> {t}")

# ---------------------------------------------------------------- emit
for name, lst in (('V', V), ('1', T1), ('2', T2)):
    with open(os.path.join(SCR, f"tier{name}_20260803.txt"), "w") as fh:
        fh.write("\n".join(sorted(lst)) + "\n")
json.dump({'V': sorted(V), 'T1': sorted(T1), 'T2': sorted(T2), 'R': R,
           'ARCHIVE': ARCHIVE, 'KEEP': sorted(K), 'PROTECTED': sorted(P),
           'per': per, 'group': {d: group(d) for d in dirs}},
          open(os.path.join(STATE, "plan.json"), "w"), indent=1)
print(f"\nremoval set: {len(R)} dirs ({len(V)} V + {len(T1)} 1 + {len(T2)} 2)"
      f" -> scripts/retire/tier{{V,1,2}}_20260803.txt")
print(f"archive set: {len(ARCHIVE)} dirs;  keep: {len(K)};  protected: {len(P)}")
print(f"state: {STATE}/plan.json")
print("\nOVERALL: " + ("PASS -- all four asserts clean" if not fail
                       else f"FAIL -- {fail} assert(s) tripped, do not proceed"))
sys.exit(0 if not fail else 1)
