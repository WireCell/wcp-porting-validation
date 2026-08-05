#!/usr/bin/env python3
"""Retirement round 2026-08-05 (LIGHT): the stale-binary sweep.

Fork of plan_20260803.py.  Do NOT re-run that script for this round -- its KEEP
set predates every pr/28..pr/37 arm and its tier globs do not cover them, so it
would dump ~180 dirs into UNCLASSIFIED.

WHY THIS ROUND EXISTS.  doc pr/33 sec.11.2 established that every binary built
before 2026-08-05 06:32 carried stale objects in build/clus (a waf dependency
miss) and is NOT reproducible from committed source.  So the ~200 pre-cutoff
arms are records of a build that nothing future can be compared against.  The
owner's decision (2026-08-05) is to archive the record layer of every pre-cutoff
NON-HUB arm and delete the arm.  See docs/pr/37 sec.13 and docs/work-tags.md.

DIFFERENCES FROM THE 08-03 FORK -- each one fixes a real defect, do not revert:

  1. ORDERED-PRIORITY CLASSIFICATION.  The 08-03 tier() was already a cascade
     but its classes were disjoint by construction.  Here they are NOT:
     work-nuecc48-prod0803 and work-vfnuecc48-prod0803 are BOTH input hubs and
     PROTECTED.txt entries, so an unordered five-predicate partition sums to 234
     against a universe of 233.  First match wins, and the double-matched set is
     printed rather than silently absorbed.

  2. THE UNIVERSE IS `work*`, INCLUDING THE BARE `work` DIR.  Same glob as the
     08-03 script and as retire_*.sh's survivor census at :120, stated here so
     the survivor arithmetic is checkable.

  3. PROTECTED.txt IS ACTUALLY READ.  The 08-03 driver hardcoded two names and
     no script ever read the file, while its header promised it was the
     carry-forward registry.  Worse, its format is documented as
     `<arm> TAB why TAB who` but three lines put FOUR whitespace-separated arm
     names in field 1 -- a parser taking field 1 as one name protects nothing.
     Here: field 1, word-split.  Ten of its seventeen arms are zero-citation, so
     a citation-driven rule without this would sweep the pr/37 sec.2.5 floor.

  4. THE mtime CUTOFF ONLY EVER KEEPS.  No arm carries a build fingerprint
     (grep -rl libWireCellClus over the arms returns nothing), so directory
     mtime is the only proxy for which binary family produced an arm -- and it
     is a proxy for last-touched, not for run time.  A post-rebuild `touch`
     would promote a stale arm.  Therefore the cutoff class is KEEP-only, and
     every deletion is by explicit tier-list membership reviewed in the dry run.

  5. THE CUTOFF IS A TIMESTAMP, NOT A DATE.  2026-08-05 06:32 is the first clean
     rebuild.  work-vfnuecc48-vf37c ran at 06:24 and is stale-family.  A
     date-granular rule would misclassify all twelve vf37 arms.

  6. HEAVY GAINS calib-pr-evt*.json.  plan_20260803.py:156 and
     archive_records_20260803.py:41 match `^calib-evt.*\\.json$`, but the
     pr/28-37 era writes `calib-pr-evt<N>.json` (pr33_cmp.py:151).  Left alone,
     ~1 GiB of calib dumps would be ARCHIVED contrary to
     archive/records/README.md's stated dropped classes.

  7. THREE NEW ASSERTS.  5: no post-cutoff or PROTECTED dir in any tier list --
     nothing in the 08-03 script catches this, and it is the one that stops the
     clean-source reference family being deleted.  6: brace-expanded citation
     counts into plan.json, so the dry run shows what each deletion costs.
     7: no KEEP/PROTECTED dir matches TIER_V, whose disposition is drop-whole.

  8. RETIRE_FRESH_HOURS DEFAULTS TO 0.  The 08-03 default of 6 h would silently
     auto-PROTECT a large set here (the pr/33 arms are hours old) and mask a
     genuine classification gap.  What it WOULD have absorbed is printed.

Writes scripts/retire/tier{A,D}_20260805.txt and state-20260805/plan.json.
Read-only w.r.t. work-*.
"""
import os, re, json, subprocess, collections, sys, time

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
SCR = os.path.join(ROOT, "scripts", "retire")
STATE = os.environ.get("RETIRE_STATE", os.path.join(SCR, "state-20260805"))
os.makedirs(STATE, exist_ok=True)
os.chdir(ROOT)

# Same glob as retire_*.sh:120's survivor census -- includes the bare `work`.
dirs = sorted(d for d in os.listdir('.')
              if d.startswith('work') and os.path.isdir(d) and not os.path.islink(d))

# --------------------------------------------------------------- the cutoff
# 2026-08-05 06:32 local = the first clean rebuild (doc pr/33 sec.11.2).
CUTOFF = time.mktime(time.strptime("2026-08-05 06:32", "%Y-%m-%d %H:%M"))

# --------------------------------------------------------------- PROTECTED
PROT_TXT = os.path.join(SCR, "PROTECTED.txt")


def read_protected(path):
    """Field 1 of every non-comment line, WORD-SPLIT.

    The file documents `<arm> TAB why TAB who`, but its three vf37 lines pack
    four arm names into field 1.  Splitting on whitespace is what makes the
    registry mean what its header says it means.
    """
    out = set()
    if not os.path.exists(path):
        return out
    for line in open(path):
        line = line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        out.update(line.split("\t")[0].split())
    return out


# Released by the owner 2026-08-05: PROTECTED.txt:27-30 holds the twelve vf37
# arms "until the population campaign has run against the promoted gate --
# owner's call to drop them after".  The campaign has NOT run; the owner
# released them early and explicitly, because the stale-binary finding means no
# future run can be gated against them anyway.  Recorded as an override, not as
# a condition met -- see the RETIRED section of PROTECTED.txt.
RELEASED = {f"work-vf{s}-vf37{a}"
            for s in ("mcp1k", "nuecc48", "r1qlmc", "r2mc") for a in "abc"}

PROTECTED = (read_protected(PROT_TXT) | {'work-nuecc48-prod0803',
                                         'work-vfnuecc48-prod0803'}) - RELEASED

# --------------------------------------------------------------- HUB
# Computed, not hardcoded, wherever possible: a hub is anything a surviving
# symlink resolves into, anything a runner pins by name, or anything git tracks.
PINNED = {
    'work',                      # _runlib.sh:18 default SBND_WORK_ROOT; run_valfast.sh:118
    'work-mcp1000',              # run_full1k_nusel.sh:43 IMGBASE
    'work-mcp1kall-d59k',        # run_valfast.sh:77 (mcp1k ql root)
    'work-nuecc48-nuf',          # run_valfast.sh:78,116; pr37_a2_floor.sh:15
    'work-nuecc48-prod0803',     # pr37_regate.sh:17,67
    'work-nuecc48-0804',         # pr37_regate.sh:17 (pr/31,34,35,36 input hub)
    'work-r1ql-first10',         # run_valfast.sh:79 (pure symlink farm)
    'work-r1ql-f1', 'work-r1ql-f2',   # run_valfast.sh:131; first10's targets
    'work-r2patrec-f1',          # run_valfast.sh:80,146
    'work-oc19scan-old',         # pr/22 exhibit chain parent
    'work-mcp10',                # 10/30-evt hand-scan imaging base
    'work-nuecc48-prsmoke2',     # KEEP since 08-03, 19 doc citations
    'work-stmcamp-d66new',       # git-tracked label stub
    'work-vfnuecc48-prod0803',   # valfast counterpart of the prod0803 root
}


def inbound_targets():
    """Every work* dir that some surviving symlink resolves into."""
    hits = collections.Counter()
    for root in dirs:
        for cur, sub, files in os.walk(root, followlinks=False):
            for name in sub + files:
                p = os.path.join(cur, name)
                if not os.path.islink(p):
                    continue
                m = re.search(r'sbnd_xin/(work[^/]*)', os.readlink(p))
                if m and m.group(1) != root:
                    hits[m.group(1)] += 1
    return hits


INBOUND = inbound_targets()
HUB = (set(INBOUND) | PINNED) & set(dirs)

# --------------------------------------------------------------- tier D
# Delete outright, no archive.  Two populations, both justified in writing:
#
#  (a) the twelve released vf37 arms -- valfast disposal rule
#      (valfast/README.md: "record the compare summary in the round doc, then
#      DELETE the roots").  The summary is in doc pr/37 sec.2.5, and doc pr/37
#      sec.13.5 records that these arms are going.
#  (b) seven arms whose own doc declares them void, or whose name says they are
#      a smoke test / probe.  work-vtxfit388-fix31-VOID-badbuild's sole citation
#      (doc pr/28:726) is the sentence calling it void.
TIER_D_JUNK = {
    'work-vtxfit388-fix31-VOID-badbuild',
    'work-pr30smoke', 'work-pr31-smoke-off', 'work-pr31-smoke-on',
    'work-p14-probe', 'work-p14-probe2', 'work-tfix388-t2probe',
}
TIER_D = RELEASED | TIER_D_JUNK

# tier V pattern, carried from plan_20260803.py:85 for ASSERT 7 only.  Its
# disposition is DROP-WHOLE-NO-ARCHIVE, so a KEEP dir matching it is a naming
# hazard: a future round would delete it without a record layer.
TIER_V = re.compile(
    r'^work-(vf(mcp1k|nuecc48|r1qlmc|r2mc)-.+|(mcp1kall|nuecc48|r1qlmc|r2mc)-vf.+)$')

FRESH_HOURS = float(os.environ.get("RETIRE_FRESH_HOURS", "0"))


def tier(d):
    """ORDERED PRIORITY, FIRST MATCH WINS.  See docstring note 1."""
    if d in HUB:                       return 'HUB'
    if os.path.getmtime(d) >= CUTOFF:  return 'POSTBUILD'
    if d in PROTECTED:                 return 'PROTECTED'
    if d in TIER_D:                    return 'D'
    return 'A'


def group(d):
    """Archive sub-tree name.  One bucket per audit round / campaign family."""
    for pat, g in ((r'^work-pr(2[89]|3[0-7])', 'pr28-37-arms'),
                   (r'^work-(oe5|oebase|oesweep)', 'oe-energy-sweep'),
                   (r'^work-(tfix388|vtxfit|dqdx388|prdisp|prdict)', 'evt388-series'),
                   (r'^work-p1[4-7]', 'p14-17-probes'),
                   (r'^work-r(9|10)', 'r9-r10-arms'),
                   (r'^work-(vf|spflag|oc19|mcp|nuecc48|r1ql|r2)', 'campaign-roots')):
        if re.match(pat, d):
            return g
    return 'other'


now = time.time()
bytier = collections.defaultdict(list)
multi = []
for d in dirs:
    # ASSERT-support: record every dir that matches more than one class rule.
    matched = [n for n, c in (('HUB', d in HUB),
                              ('POSTBUILD', os.path.getmtime(d) >= CUTOFF),
                              ('PROTECTED', d in PROTECTED),
                              ('D', d in TIER_D)) if c]
    if len(matched) > 1:
        multi.append((d, matched))
    bytier[tier(d)].append(d)

# What the 08-03 default (6 h) would have absorbed, printed rather than applied.
would_absorb = [d for d in bytier['A'] if (now - os.path.getmtime(d)) < 6 * 3600]

HUBS, POST, PROT = bytier['HUB'], bytier['POSTBUILD'], bytier['PROTECTED']
TD, TA = bytier['D'], bytier['A']
R = sorted(TD + TA)
ARCHIVE = sorted(TA)                 # tier D gets no record layer, by definition

assert len(dirs) == len(R) + len(HUBS) + len(POST) + len(PROT), \
    "classes do not partition work* -- see the MULTI-MATCH block above"

# --------------------------------------------------------------- footprint
# NOTE 6: calib-pr-evt*.json added.  The 08-03 regex only matched calib-evt*.
HEAVY = (re.compile(r'^pctree.*\.tar\.gz$'), re.compile(r'^mabc.*\.zip$'),
         re.compile(r'^calib(-pr)?-evt.*\.json(\.gz)?$'), re.compile(r'.*\.npz$'),
         re.compile(r'^clusters-apa.*\.tar\.gz$'),
         re.compile(r'^opflash_apa.*\.tar\.gz$'))


def is_heavy(f):
    return any(p.match(f) for p in HEAVY)


SPDATA = re.compile(r'^(sp-frames.*\.tar\.bz2|sbnd-sp-frames.*\.tar\.bz2|frames-dnn\.tar\.bz2)$')
RECORD_DIR = re.compile(r'^(nusel_labels|ql_labels|decisions.*)$')

per = {}
spdata_hits, label_hits = [], []
for d in R:
    tot = keep = nk = nh = 0
    for cur, sub, files in os.walk(d):
        sub[:] = [s for s in sub if not os.path.islink(os.path.join(cur, s))]
        for s in sub:
            if RECORD_DIR.match(s):
                label_hits.append(os.path.join(cur, s))
        for f in files:
            p = os.path.join(cur, f)
            if os.path.islink(p):
                continue
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            tot += sz
            if SPDATA.match(f):
                spdata_hits.append(p)
            if is_heavy(f):
                nh += 1
            else:
                keep += sz
                nk += 1
    per[d] = dict(tot=tot, keep=keep, nk=nk, nh=nh)

# --------------------------------------------------------------- citations
# Brace-expand FIRST.  doc pr/33:1352 writes the clean-binary labels as
# work-pr33-{base48,off48,f1aon48,...} wrapped across a line break inside the
# braces, so a plain scan reports eight of the fourteen reference arms as
# zero-citation.  This is informational (ASSERT 6) -- the cutoff class is what
# actually protects them -- but a future round may want to tier on it.
BRACE = re.compile(r'([A-Za-z0-9_.-]*)\{([^{}]*)\}')


def expand(text):
    """Yield brace-expanded tokens, tolerating newlines inside the braces."""
    for pre, body in BRACE.findall(text):
        for part in body.replace("\n", " ").split(","):
            part = part.strip()
            if part:
                yield pre + part


def citation_counts(names):
    cnt = collections.Counter()
    docroot = os.path.join(ROOT, "docs")
    for cur, sub, files in os.walk(docroot):
        for f in files:
            if not f.endswith((".md", ".txt", ".tsv")):
                continue
            try:
                text = open(os.path.join(cur, f), errors="replace").read()
            except OSError:
                continue
            toks = set(re.findall(r'(?<![\w-])work[\w.-]*', text)) | set(expand(text))
            for n in names & toks:
                cnt[n] += 1
    return cnt


CITES = citation_counts(set(dirs))

# --------------------------------------------------------------- report
print("=== UNIVERSE ===")
print(f"`ls -d work*` (dirs, not symlinks): {len(dirs)}")
print(f"cutoff: {time.strftime('%Y-%m-%d %H:%M', time.localtime(CUTOFF))} "
      f"(doc pr/33 sec.11.2 -- first clean rebuild)")
print(f"PROTECTED.txt parsed: {len(read_protected(PROT_TXT))} arms; "
      f"{len(RELEASED)} released by owner; effective PROTECTED {len(PROTECTED)}")

if multi:
    print(f"\n=== MULTI-MATCH ({len(multi)}) -- resolved by first-match priority "
          f"HUB > POSTBUILD > PROTECTED > D > A ===")
    for d, m in multi:
        print(f"    {d:38s} matches {'+'.join(m)}  -> {tier(d)}")

print("\n=== TIERS (review before archiving) ===")
for t, label in (('D', 'tier D  delete outright, NO archive'),
                 ('A', 'tier A  archive record layer, then delete')):
    tot = sum(per[d]['tot'] for d in bytier[t])
    print(f"\n[{label}]  {len(bytier[t])} dirs, {tot/2**30:.2f} GiB")
    for d in sorted(bytier[t]):
        why = ""
        if d in RELEASED:
            why = "  <- vf37 floor arm, RELEASED by owner 2026-08-05"
        elif d in TIER_D_JUNK:
            why = "  <- void / smoke / probe"
        print(f"    {d:38s} {per[d]['tot']/2**20:8.0f} MB  cites={CITES.get(d,0):3d}{why}")

for t, label in (('HUB', 'HUB'), ('POSTBUILD', 'POSTBUILD (clean-source)'),
                 ('PROTECTED', 'PROTECTED')):
    print(f"\n[{label}] {len(bytier[t])} dirs: {' '.join(sorted(bytier[t]))}")

print("\n=== ARCHIVE FOOTPRINT (all real files except pctree/mabc/calib*/npz/clusters/opflash) ===")
stat = collections.defaultdict(lambda: [0, 0, 0])
for d in ARCHIVE:
    s = stat[group(d)]
    s[0] += per[d]['tot']; s[1] += per[d]['keep']; s[2] += per[d]['nk']
for g in sorted(stat):
    t, k, nk = stat[g]
    print(f"{g:18s} total {t/2**30:6.2f} GiB  archive {k/2**20:8.1f} MiB ({nk} files)  "
          f"reclaim {(t-k)/2**30:6.2f} GiB")
adrop = sum(per[d]['tot'] for d in TD)
print(f"{'tier-D (no archive)':18s} total {adrop/2**30:6.2f} GiB  archive      0.0 MiB (0 files)  "
      f"reclaim {adrop/2**30:6.2f} GiB")
T = sum(per[d]['tot'] for d in R)
Kb = sum(stat[g][1] for g in stat)
print(f"{'TOTAL':18s} total {T/2**30:6.2f} GiB  archive {Kb/2**20:8.1f} MiB  "
      f"reclaim {(T-Kb)/2**30:6.2f} GiB")

# --------------------------------------------------------------- asserts
fail = 0

print("\n=== ASSERT 1: no real SP frame inside the removal set ===")
if not spdata_hits:
    print("0 -- PASS")
else:
    fail += 1
    for p in spdata_hits[:50]:
        print(f"  !! {p}")

print("\n=== ASSERT 2: no hand-scan / decision record dir inside the removal set (M13) ===")
if not label_hits:
    print("0 -- PASS")
else:
    fail += 1
    for p in label_hits[:50]:
        print(f"  !! {p}")

print("\n=== ASSERT 3: no git-tracked file inside the removal set ===")
tracked = subprocess.run(['git', 'ls-files', '-z', '--'] + R,
                         capture_output=True, text=True).stdout.split('\0')
tracked = [t for t in tracked if t]
if not tracked:
    print("0 -- PASS")
else:
    fail += 1
    for t in tracked[:50]:
        print(f"  !! {t}")

# ASSERT 4 -- dangling-link dry run over everything OUTSIDE the removal set.
Rset = set(R)
bad = collections.Counter()
nlinks = 0
top = [e for e in sorted(os.listdir('.')) if e not in Rset and not e.startswith('.')]
for root in top:
    if os.path.islink(root):
        m = re.search(r'sbnd_xin/(work[^/]*)', os.readlink(root))
        if m and m.group(1) in Rset:
            bad[(root, m.group(1))] += 1
        continue
    if not os.path.isdir(root):
        continue
    for cur, sub, files in os.walk(root, followlinks=False):
        for name in sub + files:
            p = os.path.join(cur, name)
            if os.path.islink(p):
                nlinks += 1
                m = re.search(r'sbnd_xin/(work[^/]*)', os.readlink(p))
                if m and m.group(1) in Rset:
                    bad[(root, m.group(1))] += 1

print(f"\n=== ASSERT 4: dangling-link dry run ({nlinks} symlinks outside removal set scanned) ===")
if not bad:
    print("0 -- PASS, no surviving symlink resolves into any removal candidate")
else:
    fail += 1
    for (s, t), c in bad.most_common():
        print(f"  !! {c:6d}  {s} -> {t}")

print("\n=== ASSERT 5 (NEW): no post-cutoff or PROTECTED dir in any tier list ===")
# The one that stops the clean-source reference family being swept.  Nothing in
# the 08-03 script catches it: asserts 1-4 check SP frames, label dirs,
# git-tracked files and symlinks -- none of them ask "is this the baseline the
# next campaign will be gated against?".
leak = [d for d in R if os.path.getmtime(d) >= CUTOFF or d in PROTECTED]
if not leak:
    print(f"0 -- PASS ({len(POST)} post-cutoff + {len(PROT)} protected held out)")
else:
    fail += 1
    for d in leak:
        print(f"  !! {d}")

print("\n=== ASSERT 6 (informational): citation cost of the removal set ===")
cited = [(CITES.get(d, 0), d) for d in R if CITES.get(d, 0)]
print(f"{len(cited)} of {len(R)} removal candidates are cited by at least one doc; "
      f"{len(R)-len(cited)} are uncited")
for n, d in sorted(cited, reverse=True)[:12]:
    print(f"    {n:3d}  {d}")
print("    (brace-expanded -- doc pr/33:1352 writes the pr/33 arm names inside "
      "line-wrapped braces)")

print("\n=== ASSERT 7 (NEW): no KEEP/PROTECTED dir matches TIER_V (drop-whole, no archive) ===")
vleak = [d for d in HUBS + POST + PROT if TIER_V.match(d)]
if not vleak:
    print("0 -- PASS")
else:
    # Not fatal for THIS round -- work-vfnuecc48-prod0803 is a legitimate
    # long-lived valfast root -- but a future round using plan_20260803.py's
    # TIER_V would drop it with no record layer.  Report loudly.
    print(f"!! {len(vleak)} survivor(s) match the tier-V pattern; a future round "
          f"reusing that regex would drop them WITHOUT an archive:")
    for d in vleak:
        print(f"     {d}")
    print("   forbid the '-vf' infix for production roots (docs/work-tags.md).")

if would_absorb:
    print(f"\n=== FRESH_HOURS note ===\nRETIRE_FRESH_HOURS={FRESH_HOURS:g} (08-03 default was 6). "
          f"At 6 h this round would have silently auto-PROTECTED {len(would_absorb)} "
          f"tier-A dir(s):")
    for d in would_absorb[:20]:
        print(f"     {d}   ({(now-os.path.getmtime(d))/3600:.1f} h ago)")

# --------------------------------------------------------------- emit
for name, lst in (('A', TA), ('D', TD)):
    with open(os.path.join(SCR, f"tier{name}_20260805.txt"), "w") as fh:
        fh.write("\n".join(sorted(lst)) + "\n")
json.dump({'A': sorted(TA), 'D': sorted(TD), 'R': R, 'ARCHIVE': ARCHIVE,
           'KEEP': sorted(HUBS + POST), 'HUB': sorted(HUBS),
           'POSTBUILD': sorted(POST), 'PROTECTED': sorted(PROT),
           'RELEASED': sorted(RELEASED), 'cutoff': CUTOFF,
           'per': per, 'cites': dict(CITES),
           'group': {d: group(d) for d in dirs}},
          open(os.path.join(STATE, "plan.json"), "w"), indent=1)

print(f"\nremoval set: {len(R)} dirs ({len(TA)} A + {len(TD)} D)"
      f" -> scripts/retire/tier{{A,D}}_20260805.txt")
print(f"archive set: {len(ARCHIVE)} dirs;  survivors: {len(HUBS)} HUB + "
      f"{len(POST)} POSTBUILD + {len(PROT)} PROTECTED = "
      f"{len(HUBS)+len(POST)+len(PROT)}")
print(f"state: {STATE}/plan.json")
print("\nOVERALL: " + ("PASS -- all gating asserts clean" if not fail
                       else f"FAIL -- {fail} assert(s) tripped, do not proceed"))
sys.exit(0 if not fail else 1)
