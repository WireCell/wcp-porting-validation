#!/usr/bin/env python3
"""Retirement round 2026-08-02: removal set, archive footprint, dangling-link dry run.

Fork of plan.py (2026-07-30 round). Differences:
  - survivor list = the documented KEEP set + the pr/22 exhibit chain
    (work-oc19scan-old was made self-contained by materialize_20260802.sh);
    EVERYTHING else is the removal set (owner decision 2026-08-02: all pr-era
    campaign arms retire, including the pr11v3 census arms).
  - group() knows the pr/11..pr/22-era campaigns; unknown dirs fall into
    'probes' instead of asserting (the July script asserted UNCLASSIFIED).
  - HEAVY additionally drops opflash_apa*.tar.gz (the 8.5 GB full-chain arms
    are dominated by opflash + mabc-apa-face.zip + tracking-stm.root).
  - the dangling-link dry run walks EVERYTHING in sbnd_xin outside the removal
    set (the July version walked a fixed root list, maxdepth-unbounded but
    root-limited).
  - safety scan: reports any REAL sp-frames*/frames-dnn* file inside the
    removal set (irreplaceable SP data must never be in a removal candidate).

Writes state to scripts/retire/state-20260802/ (plan.json) and the removal
list to scripts/retire/tier1_20260802.txt. Read-only w.r.t. work-*.
"""
import os, re, json, collections

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
SCR  = os.path.join(ROOT, "scripts", "retire")
STATE = os.environ.get("RETIRE_STATE", os.path.join(SCR, "state-20260802"))
os.makedirs(STATE, exist_ok=True)
os.chdir(ROOT)

dirs = sorted(d for d in os.listdir('.')
              if d.startswith('work') and os.path.isdir(d) and not os.path.islink(d))

SURVIVORS_EXACT = {
    'work', 'work-mcp1000', 'work-mcp10', 'work-mcp1kall-d59k',
    'work-nuecc48-base', 'work-nuecc48-nuf', 'work-nuecc48-prsmoke',
    'work-nuecc48-prsmoke2',
    'work-mcp1kall-cath01', 'work-nuecc48-cath01',
    'work-stmcamp-d66new',          # git-tracked hand-scan label stub
    'work-oc19scan-old',            # pr/22 exhibit parent, materialized 2026-08-02
}
SURVIVOR_PREFIXES = ('work-r1ql-', 'work-r2patrec-', 'work-pr22gap-')

def survivor(d):
    return d in SURVIVORS_EXACT or d.startswith(SURVIVOR_PREFIXES)

# campaign grouping of the removal set -- archive layout + the work-tags table.
GROUPS = [
    ('pr11-census',   r'^work-(mcp1kall|nuecc48|r1qlmc|r2mc)-(pr11(v2|v3|fix)?|all73(final|fix)|failfix|badallocfix|fix)$'),
    ('pr11-audit',    r'^work-(mcp1kall-ab30\w*|audit\w*-.+|det(base|fix|diag|geo|fixgeo|fixdl|baserep)\w*-.+)$'),
    ('cath13-ccfeat', r'^work-(mcp1kall-(cath13\w*|cathdbg1|ccfeat300b?)|cath13ql|cath13pr|cathdbg1pr|ccfeat300pr|nuecc48-cath13ql)$'),
    ('rescue01',      r'^work-(mcp1kall|nuecc48)-rescue01(pr)?$'),
    ('cbr',           r'^work-(mcp1kall-cbr\w*|nuecc48-cbr\w*|cbr-det\d)$'),
    ('vveto',         r'^work-(mcp1kall-(vveto1k|vv\d+rr)|nuecc48-vveto)$'),
    ('nsc',           r'^work-nsc\w*-.+$'),
    ('nbl',           r'^work-(nbl15|nbloff|nblrep)-.+$'),
    ('isog-u17',      r'^work-(mcp1kall-(isog\w*|u17\w*)|nuecc48-(isog\w*|iso10550\w*|u17\w*))$'),
    ('oc19',          r'^work-(mcp1kall-oc19\w*|nuecc48-oc19\w*|oc444187-\w+|oc19scan-new)$'),
    ('cathA12-b0',    r'^work-(mcp1kall-(cathA12\w*|b0on|b0off)|nuecc48-cathA12\w*|b0\w*(-\w+)?|pi2on-b0off)$'),
    ('pi-partI',      r'^work-(mcp1kall-pi\d\w*|pi\d\S*|p[34]-59003-on|nuecc48-pi5\w*)$'),
    ('pr20x',         r'^work-(pr20x-\w+|mcp1kall-pr20x)$'),
]
def group(d):
    if survivor(d): return 'KEEP'
    for g, pat in GROUPS:
        if re.match(pat, d): return g
    return 'probes'

HEAVY = (re.compile(r'^pctree.*\.tar\.gz$'), re.compile(r'^mabc.*\.zip$'),
         re.compile(r'^calib-evt.*\.json(\.gz)?$'), re.compile(r'.*\.npz$'),
         re.compile(r'^clusters-apa.*\.tar\.gz$'),
         re.compile(r'^opflash_apa.*\.tar\.gz$'))
def is_heavy(f): return any(p.match(f) for p in HEAVY)

SPDATA = re.compile(r'^(sp-frames.*\.tar\.bz2|sbnd-sp-frames.*\.tar\.bz2|frames-dnn\.tar\.bz2)$')

R = [d for d in dirs if not survivor(d)]
K = [d for d in dirs if survivor(d)]

stat = collections.defaultdict(lambda: [0, 0, 0, 0])   # tot, keepbytes, nkeep, nheavy
per = {}
spdata_hits = []
for d in R:
    tot = keep = nk = nh = 0
    for cur, sub, files in os.walk(d):
        sub[:] = [s for s in sub if not os.path.islink(os.path.join(cur, s))]
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
    s = stat[group(d)]; s[0] += tot; s[1] += keep; s[2] += nk; s[3] += nh

print("=== GROUPING (review before archiving) ===")
bygrp = collections.defaultdict(list)
for d in R: bygrp[group(d)].append(d)
for g in sorted(bygrp):
    print(f"[{g}] {len(bygrp[g])} dirs")
    for d in sorted(bygrp[g]): print(f"    {d}  ({per[d]['tot']/2**20:.0f} MB)")

print("\n=== ARCHIVE FOOTPRINT (real files except pctree/mabc/calib/npz/clusters/opflash) ===")
for g in sorted(stat):
    t, k, nk, nh = stat[g]
    print(f"{g:14s} total {t/2**30:6.2f} GiB  archive {k/2**20:8.1f} MiB ({nk} files)  reclaim {(t-k)/2**30:6.2f} GiB")
T = sum(s[0] for s in stat.values()); Kb = sum(s[1] for s in stat.values())
print(f"{'TOTAL':14s} total {T/2**30:6.2f} GiB  archive {Kb/2**20:8.1f} MiB  reclaim {(T-Kb)/2**30:6.2f} GiB")

print("\n=== SP-DATA SAFETY SCAN (real sp-frames/frames-dnn inside removal set; MUST be empty) ===")
if not spdata_hits: print("0 -- no irreplaceable SP data in any removal candidate")
else:
    for p in spdata_hits: print(f"  !! {p}")

# dangling-link dry run: walk EVERYTHING outside the removal set
Rset = set(R)
bad = collections.Counter()
nlinks = 0
top = [e for e in sorted(os.listdir('.')) if e not in Rset and not e.startswith('.')]
for root in top:
    if os.path.islink(root):
        t = os.readlink(root)
        m = re.search(r'sbnd_xin/(work[^/]*)', t)
        if m and m.group(1) in Rset: bad[(root, m.group(1))] += 1
        continue
    if not os.path.isdir(root): continue
    for cur, sub, files in os.walk(root, followlinks=False):
        for name in sub + files:
            p = os.path.join(cur, name)
            if os.path.islink(p):
                nlinks += 1
                t = os.readlink(p)
                m = re.search(r'sbnd_xin/(work[^/]*)', t)
                if m and m.group(1) in Rset: bad[(root, m.group(1))] += 1

print(f"\n=== DANGLING-LINK DRY RUN ({nlinks} symlinks outside removal set scanned) ===")
if not bad: print("0 -- no surviving symlink resolves into any removal candidate")
else:
    for (s, t), c in bad.most_common(): print(f"  {c:6d}  {s} -> {t}")

with open(os.path.join(SCR, "tier1_20260802.txt"), "w") as fh:
    fh.write("\n".join(sorted(R)) + "\n")
json.dump({'R': R, 'K': K, 'per': per, 'group': {d: group(d) for d in dirs}},
          open(os.path.join(STATE, "plan.json"), "w"), indent=1)
print(f"\nremoval set: {len(R)} dirs -> scripts/retire/tier1_20260802.txt; keep: {len(K)} dirs")
print(f"state: {STATE}/plan.json")
