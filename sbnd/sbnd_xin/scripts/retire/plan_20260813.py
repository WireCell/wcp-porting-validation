#!/usr/bin/env python3
"""Retirement round 2026-08-13 -- the pr/66-75 campaign sweep, 74 G -> ~20 G.

Fork of plan_20260811.py.  Same shape and the same six asserts; what changed is
the KEEP set and three defects fixed (below).

Why this round exists: the pr/51, pr/64, pr/66, pr/67, pr/72, pr/73, pr/74 and
pr/75 campaigns regrew the tree from the 08-11 round's 18 survivors / 23 G to
401 work-* dirs / 74 G.  Every one of those arms is a leg of an A/B whose verdict
already lives in its doc, and production has moved 82 clus/cfg commits since the
cb0805 campaign, so none of them is comparable to anything current.

KEEP is 13 explicit names.  It is SMALLER than 08-11's 18 because the three
pr64r4 reference arms and the two oc56 scan-dump arms are superseded by the
prod0813 campaign this round clears the disk for.  It is also load-bearing in a
way 08-11's was not: the owner chose a PR-STAGE-ONLY reprocessing, so the five
work-*-cb0805 Q/L hubs are the prod0813 campaign's INPUT, not a record layer.

  ==> NO PHASE 4 HUB THINNING THIS ROUND.  thin_hubs_20260811.py must NOT be
      re-run.  Nothing inside the five cb0805 hubs may be removed: their
      ql_evt*/pctree-evt*.tar.gz feed the PR chain and their
      ql_evt*/mabc-all-apa.zip feed the Bee builder.  work-img-mcp1k's
      remaining icluster-apa*-masked.npz is likewise a genuine imaging input.

Three defects fixed vs plan_20260811.py, each found during exploration for this
round -- see docs/work-tags.md "RETIREMENT ROUND 2026-08-13":

  1. VOID-pr32-round1/ no longer exists (08-11 removed it); the EXTRA universe
     block is dropped rather than left as dead code that silently matches
     nothing.  Verified: no non-`work`-prefixed removal-candidate dir exists.
  2. ASSERT 4 skipped every hidden top-level entry (`not e.startswith('.')`),
     so the .nutmp/ and .tracetmp/ dirs created since the 08-11 round were
     invisible to the dangling-link dry run.  Both hold 0 symlinks today, so
     this closes a blind spot rather than fixing a live break -- but a blind
     spot in the one assert that protects against stranding links is not worth
     keeping.  Now only .git is excluded.
  3. NEW: a directory-mtime histogram is printed.  It is a HUMAN SANITY REPORT
     ONLY and deliberately does NOT gate anything.  Do not turn it into an
     mtime KEEP rule the way plan_20260805.py had: because the whole tree was
     regenerated after 08-11, the freshness distribution here is degenerate
     (382 of 401 dirs are under 48 h old, 300 under 36 h), so any cutoff coarse
     enough to be safe protects nearly the entire universe.  The gate for this
     round is the explicit KEEP dict, full stop.

KNOWN COST, stated rather than silently absorbed: scripts/analysis/pr57/
oc56_truth.py's DEFAULT_ARMS loses all three of its arms this round.  Two
(work-pr64r4-scan48/scan19) retire here; the third name it lists,
work-pr64-scan1k, is ALREADY stale -- the disk has work-pr64r4-scan1k.  The
08-11 round justified reclassifying oc56scan-evt*.jsonl as HEAVY (dropped, not
archived) on the grounds that "the two arms oc56_truth.py cites are KEPT whole".
That justification does not survive this round, and the oc56 truth table becomes
non-recomputable without a fresh PR_OC56_SCAN_DUMP=1 arm.  Owner-confirmed.

Writes scripts/retire/tierA_20260813.txt and state-20260813/plan.json.
Read-only w.r.t. work-*.
"""
import os, re, json, subprocess, collections, sys, filecmp, time

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
SCR = os.path.join(ROOT, "scripts", "retire")
STATE = os.environ.get("RETIRE_STATE", os.path.join(SCR, "state-20260813"))
os.makedirs(STATE, exist_ok=True)
os.chdir(ROOT)

if os.path.exists(os.path.join(STATE, "removed.tsv")) and not os.environ.get("RETIRE_REPLAN"):
    sys.stderr.write(
        f"REFUSING: {STATE}/removed.tsv exists -- this round has already run (M13).\n"
        f"Fork with a new date/state for a new round; RETIRE_REPLAN=1 to override.\n")
    sys.exit(3)

dirs = sorted(d for d in os.listdir('.')
              if d.startswith('work') and os.path.isdir(d) and not os.path.islink(d))

# ---------------------------------------------------------------- KEEP
KEEP_WHY = {
    # imaging hubs -- runner-pinned (valfast/run_valfast.sh), 4295 inbound symlinks.
    # NOTE for the next round: work-img-mcp1k's inbound count drops 4003 -> 1000
    # when the pr66 arms go, because 3000 of those links are pr66's own.  Do not
    # copy the 4003 figure forward.
    'work-img-mcp1k':        'imaging hub, 4003 inbound symlinks, run_valfast.sh + run_full1k_nusel.sh pin',
    'work-img-nuecc48':      'imaging hub, 193 inbound symlinks, run_valfast.sh pin',
    'work-img-ncpi0':        'imaging hub, 76 inbound symlinks, run_valfast.sh pin',
    'work-img-r1qlmc':       'imaging hub, 10 inbound symlinks, run_valfast.sh pin; only copy of this sim sample',
    'work-img-r2mc':         'imaging hub, 13 inbound symlinks, run_valfast.sh pin; only copy of this sim sample',
    # Q/L hubs -- THE prod0813 CAMPAIGN'S INPUT.  Not thinnable, see docstring.
    'work-mcp1k-cb0805':     'Q/L hub = prod0813 PR input (1000 pctree) + Bee mabc-all-apa source',
    'work-nuecc48-cb0805':   'Q/L hub = prod0813 PR input (48) + Bee source; geom_ab_batch.sh pin',
    'work-ncpi0-cb0805':     'Q/L hub = prod0813 PR input (19) + Bee source',
    'work-r1qlmc-cb0805':    'Q/L hub = prod0813 PR input (10 sim)',
    'work-r2mc-cb0805':      'Q/L hub = prod0813 PR input (13 sim)',
    # git-tracked / M13 / PROTECTED.txt
    'work-stmcamp-d66new':   'git-tracked nusel_labels/ hand-scan state (M13)',
    'work-nuecc48-prsmoke2': '3 git-tracked runner scripts',
    'work-tfix388-r9':       'doc pr/28 sec.15.9 -- NOT reproducible from any surviving input',
}
KEEP = set(KEEP_WHY)

# ---- what the owner released from PROTECTED.txt (superseded this round) -----
def read_protected(path):
    out = set()
    if not os.path.exists(path):
        return out
    for line in open(path):
        line = line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        out.update(line.split("\t")[0].split())
    return out


PROT_LISTED = read_protected(os.path.join(SCR, "PROTECTED.txt"))
RELEASED = sorted(PROT_LISTED - KEEP)


def tier(d):
    return 'KEEP' if d in KEEP else 'A'


def group(d):
    for pat, g in ((r'^work-img-', 'imaging-hubs'),
                   (r'^work-.*-cb0805$', 'ql-hubs'),
                   (r'^work-beeprod0813', 'beeprod0813-probe'),
                   (r'^work-prdisp', 'prdisp-arms'),
                   (r'^work-pr7[0-9]', 'pr70-79-era'),
                   (r'^work-pr6[0-9]', 'pr60-69-era'),
                   (r'^work-pr5[0-9]', 'pr50-59-era'),
                   (r'^work-pr4[0-9]', 'pr40-49-era'),
                   (r'^work-pr3[0-9]', 'pr30-39-era'),
                   (r'^work-oc', 'oc-arms'),
                   (r'^work-evt', 'single-event-traces'),
                   (r'^work-vf', 'valfast-transient'),
                   (r'^work-nuecc48', 'nuecc48-arms'),
                   (r'^work-ncpi0', 'ncpi0-arms'),
                   (r'^work-mcp1k', 'mcp1k-arms'),
                   (r'^work-(r1ql|r2mc|r2patrec)', 'mc-sample-arms')):
        if re.match(pat, d):
            return g
    return 'other'


bytier = collections.defaultdict(list)
for d in dirs:
    bytier[tier(d)].append(d)

K, TA = bytier['KEEP'], bytier['A']
R = sorted(TA)
ARCHIVE = sorted(TA)
assert len(dirs) == len(R) + len(K), "classes do not partition work*"
missing_keep = sorted(KEEP - set(dirs))

# ---------------------------------------------------------------- footprint
# HEAVY is UNCHANGED from 08-11.  Verified sufficient for this round's universe:
# every file >5 MB in the removal set that these eight patterns do not match is
# a log (wct_pr_evt*.log, stdout.log; largest 32 MB), which is record by design.
# All 751 *scan-evt*.jsonl are oc56scan; there is no oc66/oc74 variant.
HEAVY = (re.compile(r'^pctree.*\.tar\.gz$'), re.compile(r'^mabc.*\.zip$'),
         re.compile(r'^calib(-pr)?-evt.*\.json(\.gz)?$'), re.compile(r'.*\.npz$'),
         re.compile(r'^clusters-apa.*\.tar\.gz$'),
         re.compile(r'^opflash_apa.*\.tar\.gz$'),
         re.compile(r'^tracking-pr\.root$'),
         re.compile(r'^oc56scan-evt.*\.jsonl$'))


def is_heavy(f):
    return any(p.match(f) for p in HEAVY)


RECORD_DIR = re.compile(r'^(nusel_labels|ql_labels|decisions.*)$')

# Sized over the WHOLE universe, not just the removal set: 08-11 sized only R,
# so its KEEP table printed 0 MB for every survivor -- and the survivor sizes are
# exactly what the round's disk-target arithmetic rests on.  label_hits stays
# scoped to the removal set (a label dir inside a KEEP arm is not at risk and
# must not be demanded of the archive by ASSERT 2).
Rset_for_labels = set(R)
per = {}
label_hits = []
for d in dirs:
    tot = keep = nk = nh = 0
    for cur, sub, files in os.walk(d):
        sub[:] = [s for s in sub if not os.path.islink(os.path.join(cur, s))]
        for s in sub:
            if RECORD_DIR.match(s) and d in Rset_for_labels:
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
            if is_heavy(f):
                nh += 1
            else:
                keep += sz
                nk += 1
    per[d] = dict(tot=tot, keep=keep, nk=nk, nh=nh)

# ---------------------------------------------------------------- report
print("=== RETIREMENT ROUND 2026-08-13 (pr/66-75 sweep) ===")
print(f"universe {len(dirs)} work* dirs -> KEEP {len(K)}, remove {len(R)}")
if missing_keep:
    print(f"!! KEEP names not on disk: {missing_keep}")
print("\n[KEEP]")
for d in sorted(K):
    print(f"    {d:26s} {per.get(d,{}).get('tot',0)/2**20:8.0f} MB  {KEEP_WHY[d]}")
print(f"    {'--- KEEP TOTAL':26s} {sum(per[d]['tot'] for d in K)/2**30:8.2f} GiB")
print(f"\n[REMOVE] {len(R)} dirs, {sum(per[d]['tot'] for d in R)/2**30:.2f} GiB")
for d in sorted(R, key=lambda x: -per[x]['tot'])[:30]:
    print(f"    {d:26s} {per[d]['tot']/2**20:8.0f} MB  [{group(d)}]")
if len(R) > 30:
    print(f"    ... and {len(R)-30} more (full list: tierA_20260813.txt)")

print(f"\n[RELEASED from PROTECTED.txt, superseded 2026-08-13] {len(RELEASED)}")
print("    " + (" ".join(RELEASED) if RELEASED else "(none)"))

# ---- SANITY REPORT ONLY -- see docstring defect 3.  Gates nothing. ----------
print("\n=== mtime histogram of the REMOVAL set (sanity report, gates NOTHING) ===")
now = time.time()
buckets = [(6, '< 6 h'), (12, '< 12 h'), (24, '< 24 h'), (36, '< 36 h'),
           (48, '< 48 h'), (24 * 7, '< 7 d'), (10 ** 9, 'older')]
seen = set()
for hrs, lab in buckets:
    sel = [d for d in R if d not in seen and (now - os.path.getmtime(d)) < hrs * 3600]
    seen |= set(sel)
    if sel:
        print(f"    {lab:8s} {len(sel):4d} dirs  {sum(per[d]['tot'] for d in sel)/2**30:6.2f} GiB")
print("    NOTE: the tree was fully regenerated after the 08-11 round, so this")
print("    distribution is degenerate and CANNOT be used as a KEEP rule.")

print("\n=== ARCHIVE FOOTPRINT ===")
stat = collections.defaultdict(lambda: [0, 0, 0])
for d in ARCHIVE:
    s = stat[group(d)]
    s[0] += per[d]['tot']; s[1] += per[d]['keep']; s[2] += per[d]['nk']
for g in sorted(stat):
    t, k, nk = stat[g]
    print(f"{g:22s} total {t/2**30:6.2f} GiB  archive {k/2**20:8.1f} MiB ({nk} files)  "
          f"reclaim {(t-k)/2**30:6.2f} GiB")
T = sum(per[d]['tot'] for d in R)
Kb = sum(stat[g][1] for g in stat)
print(f"{'TOTAL':22s} total {T/2**30:6.2f} GiB  archive {Kb/2**20:8.1f} MiB  "
      f"reclaim {(T-Kb)/2**30:6.2f} GiB")

# ---------------------------------------------------------------- asserts
fail = 0

print("\n=== ASSERT 1: no real SP frame is lost -- source dirs survive locally ===")
SP_SOURCES = {
    'mcp1k (1000 data)':  'input_files_reco1/staged-mcp2025c-1000evt',
    'nuecc48 (48 data)':  'input_files_reco1/extracted-2025fall-48evt-fsprod',
    'ncpi0 (19 data)':    'input_files_reco1/extracted-ncpi0',
    'r1qlmc (10 sim)':    'input_files_reco1/extracted-r1ql-f1',
    'r2mc (13 sim)':      'input_files_reco1/extracted-r2patrec-f1',
}
for label, src in sorted(SP_SOURCES.items()):
    ok = os.path.isdir(src) and not os.path.islink(src) and bool(os.listdir(src))
    n = len(os.listdir(src)) if os.path.isdir(src) else 0
    print(f"      {'OK ' if ok else '!! '} {label:20s} {src}  ({n} entries)")
    if not ok:
        fail += 1
print("    No imaging SP layer drops this round -- there is no Phase 4 thinning.")

print("\n=== ASSERT 2: every hand-scan / label record has a verified archive copy (M13) ===")
LABROOT = os.path.join(ROOT, "archive", "records", "labels")


def tree_identical(a, b):
    cmp = filecmp.dircmp(a, b)
    if cmp.left_only or cmp.right_only or cmp.funny_files:
        return False
    _, mismatch, errors = filecmp.cmpfiles(a, b, cmp.common_files, shallow=False)
    if mismatch or errors:
        return False
    return all(tree_identical(os.path.join(a, d), os.path.join(b, d))
               for d in cmp.common_dirs)


if not label_hits:
    print("0 label dirs in the removal set -- PASS (strict form)")
    print("    (the tree's only label dir is work-stmcamp-d66new/nusel_labels, in KEEP;")
    print("     sbnd_xin/vertex_labels/ and overclustering_labels/ are outside work-*)")
else:
    for p in sorted(label_hits):
        rel = os.path.relpath(p, ROOT)
        dst = os.path.join(LABROOT, rel)
        nsrc = sum(len(f) for _, _, f in os.walk(p))
        if not os.path.isdir(dst):
            print(f"  !! NO ARCHIVE COPY: {rel}")
            fail += 1
        elif not tree_identical(p, dst):
            print(f"  !! ARCHIVE COPY DIFFERS: {rel}")
            fail += 1
        else:
            ntag = len(os.listdir(p))
            print(f"  OK  {rel:38s} {ntag:2d} tags, {nsrc:3d} files -> archive copy verified")

print("\n=== ASSERT 3: no git-tracked file inside the removal set ===")
tracked = subprocess.run(['git', '-C', '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img',
                           'ls-files', '-z', '--'] + ['sbnd/sbnd_xin/' + d for d in R],
                          capture_output=True, text=True).stdout.split('\0')
tracked = [t for t in tracked if t]
if not tracked:
    print("0 -- PASS")
else:
    fail += 1
    for t in tracked[:50]:
        print(f"  !! {t}")

print("\n=== ASSERT 4: dangling-link dry run ===")
# Defect 2 fix: only .git is excluded now, so .nutmp/ .tracetmp/ and any future
# hidden top-level dir are covered.
Rset = set(R)
bad = collections.Counter()
nlinks = 0
top = [e for e in sorted(os.listdir('.')) if e not in Rset and e != '.git']
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
print(f"({nlinks} symlinks outside removal set, hidden dirs included)")
if not bad:
    print("0 -- PASS")
else:
    fail += 1
    for (s, t), c in bad.most_common():
        print(f"  !! {c:6d}  {s} -> {t}")

print("\n=== ASSERT 5: every KEEP name exists and is non-empty ===")
bad_keep = [d for d in KEEP if not (os.path.isdir(d) and os.listdir(d))]
print("0 -- PASS" if not bad_keep else f"  !! {bad_keep}")
if bad_keep:
    fail += 1

print("\n=== ASSERT 6: overclustering_labels archived + git-tracked (carried from 08-11) ===")
occ_src = os.path.join(ROOT, "overclustering_labels")
occ_dst = os.path.join(LABROOT, "overclustering_labels")
if not os.path.isdir(occ_dst):
    print("  !! NO ARCHIVE COPY at archive/records/labels/overclustering_labels")
    fail += 1
elif not tree_identical(occ_src, occ_dst):
    print("  !! ARCHIVE COPY DIFFERS from live overclustering_labels/")
    fail += 1
else:
    gitcount = subprocess.run(
        ['git', '-C', '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img', 'ls-files',
         'sbnd/sbnd_xin/overclustering_labels'],
        capture_output=True, text=True).stdout.strip().splitlines()
    if len(gitcount) < 200:
        print(f"  !! only {len(gitcount)} files git-tracked under overclustering_labels (expect >=230)")
        fail += 1
    else:
        print(f"  OK  archive copy verified identical, {len(gitcount)} files committed to git")

# ---------------------------------------------------------------- emit
with open(os.path.join(SCR, "tierA_20260813.txt"), "w") as fh:
    fh.write("\n".join(sorted(TA)) + "\n")
json.dump({'A': sorted(TA), 'D': [], 'R': R, 'ARCHIVE': ARCHIVE,
           'KEEP': sorted(K), 'KEEP_WHY': KEEP_WHY, 'HUB': [], 'POSTBUILD': [],
           'PROTECTED': sorted(KEEP & PROT_LISTED), 'RELEASED': RELEASED,
           'per': per, 'cites': {},
           'group': {d: group(d) for d in dirs}},
          open(os.path.join(STATE, "plan.json"), "w"), indent=1)
print(f"\nremoval set: {len(R)} dirs -> scripts/retire/tierA_20260813.txt")
print(f"survivors: {len(K)}")
print(f"state: {STATE}/plan.json")
print("\nOVERALL: " + ("PASS -- all asserts clean" if not fail
                       else f"FAIL -- {fail} assert(s) tripped, do not proceed"))
sys.exit(0 if not fail else 1)
