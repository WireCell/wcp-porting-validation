#!/usr/bin/env python3
"""Retirement round 2026-08-16 -- the pr/79-86 campaign sweep, 158 G -> ~28 G.

Fork of plan_20260813.py.  Same shape and asserts 1-6 unchanged; KEEP grows
from 13 to 27 names, group() gets new buckets for the families that regrew the
tree, and one new read-only assert (7) is added.  tier() is UNCHANGED --
KEEP-only, exactly as 08-13 left it.  See below for why no refactor happens
this round.

Why this round exists: the pr/79 (dl_vtx_harvest deployment), pr/83
(break_seg_orient), pr/85 (interposed stub) and pr/86 (interposed splice, two
rounds -- sec.14 and sec.15) campaigns regrew the tree from the 08-13 round's
13 survivors / 20 G to 216 work-* dirs / 158 G in three days.  Every one of
those arms is a leg of an A/B whose verdict already lives in its doc.

KEEP is 27 names, not 13, because this round -- unlike 08-13 -- must also
satisfy three owner requirements that pin *reference* arms, not just
*infrastructure*:

  1. all existing hand-scan results on the neutrino vertex
  2. the PR results the interactive display needs (two bokeh servers, ports
     5017 and 5018, are LIVE right now, both pointed at work-r1qlmc-prod0813
     and work-r2mc-prod0813)
  3. the latest production result for nueCC / numu-inclusive (mcp1k) / NCpi0

Full derivation of each name in docs/work-tags.md "RETIREMENT ROUND
2026-08-16" and in the approved plan file.  Short version:
  - vertex_labels/ (483 labels, 6 tags) records `source` inside a
    work-*-prod0813 arm for every label -- KEEP all five -prod0813 arms.
  - baselines.py:36 deployed_dump_path() rewrites -prod0813 -> -ma10 and
    resolves 473 dumps (407+47+19) -- KEEP the three PLAIN -ma10 arms (not
    the 12 -ma10ft2u/-ma10-k20/-ma10k20-harv* variants, which are unreachable
    from the label loader).
  - work-{mcp1k,nuecc48,ncpi0}-pr87ion3 is the doc pr/86 sec.15 knobs-on arm
    at the config that became SBND production (toolkit 771f075b).
  - work-pr87-postflip-{mcp1k,nuecc48,ncpi0} is the ONLY physical evidence
    that the shipped jsonnet reproduces pr87ion3 (doc sec.15.4 asserts
    "42/42 archives == pr87ion3" but never names the arm).

  ==> STILL NO PHASE 4 HUB THINNING THIS ROUND, same reasoning as 08-13: the
      five work-*-cb0805 hubs remain the PR chain's live input (verified this
      round: every existing PR arm, including the newest, reads
      "reading file=.../work-<s>-cb0805/ql_evt<N>/pctree-evt<N>.tar.gz").
      thin_hubs_20260811.py must NOT be re-run.

THE PROTECTED.txt NEAR-MISS, and why this round does NOT refactor tier():
08-13's plan.py parses PROTECTED.txt into PROT_LISTED but only ever prints it
(RELEASED = PROT_LISTED - KEEP, report-only) or intersects it with KEEP for
the driver's interlock 3 -- which therefore adds nothing beyond KEEP itself.
PROTECTED.txt was edited AFTER the 08-13 round ran, adding five prod0813
lines; a straight fork of that plan script would have printed all five as
"RELEASED" and swept them, because tier() never consults PROT_LISTED.

This round's fix is deliberately NOT a refactor: all six PROTECTED.txt names
are already in this round's KEEP_WHY, so the trap cannot fire here.  The
defence is ASSERT 7, a read-only check that PROT_LISTED - KEEP is empty.
Restoring plan_20260805.py's ordered PROTECTED tier is left for a follow-up
hardening commit AFTER this sweep -- the script that deletes ~189 dirs should
not also be exercising a rewritten classifier for the first time, on the same
run as the cd -P fix's first-ever CONFIRM=yes exercise.

KNOWN COST, stated rather than silently absorbed (docs/work-tags.md carries
the full list): ~78 arms across the pr86r2*/pr86b*/pr87off{2..6} families have
no textual record anywhere outside doc pr/86's collective/ledger prose; their
individual verdicts are lost, the doc's summary numbers are not.  The five
work-*-pr85ion2 arms (10 G) retire -- doc pr/85's gate PASS becomes the only
surviving record of that epoch.  work-*-pr86ioff (2 dirs) are aborted partial
arms per doc pr/86 sec.14.7 and retire with no loss.

Writes scripts/retire/tierA_20260816.txt and state-20260816/plan.json.
Read-only w.r.t. work-*.
"""
import os, re, json, subprocess, collections, sys, filecmp, time

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
SCR = os.path.join(ROOT, "scripts", "retire")
STATE = os.environ.get("RETIRE_STATE", os.path.join(SCR, "state-20260816"))
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
    # --- infrastructure, carried from 08-13 (13 names) ------------------
    'work-img-mcp1k':        'imaging hub, run_valfast.sh + run_full1k_nusel.sh pin',
    'work-img-nuecc48':      'imaging hub, run_valfast.sh pin',
    'work-img-ncpi0':        'imaging hub, run_valfast.sh pin',
    'work-img-r1qlmc':       'imaging hub, run_valfast.sh pin; only copy of this sim sample',
    'work-img-r2mc':         'imaging hub, run_valfast.sh pin; only copy of this sim sample',
    'work-mcp1k-cb0805':     'Q/L hub = live PR chain input (verified: every pr87ion3 event reads its pctree-evt*.tar.gz) + Bee mabc-all-apa source',
    'work-nuecc48-cb0805':   'Q/L hub = live PR chain input + Bee source; geom_ab_batch.sh pin',
    'work-ncpi0-cb0805':     'Q/L hub = live PR chain input + Bee source',
    'work-r1qlmc-cb0805':    'Q/L hub = live PR chain input (10 sim)',
    'work-r2mc-cb0805':      'Q/L hub = live PR chain input (13 sim)',
    'work-stmcamp-d66new':   'git-tracked nusel_labels/ hand-scan state (M13)',
    'work-nuecc48-prsmoke2': '3 git-tracked runner scripts',
    'work-tfix388-r9':       'doc pr/28 sec.15.9 -- NOT reproducible from any surviving input',
    # --- requirement 1+2: vertex hand scan + live display (8 names) -----
    'work-nuecc48-prod0813': 'vertex_labels/ source arm (47 labels); PROTECTED.txt',
    'work-ncpi0-prod0813':   'vertex_labels/ source arm (19 labels); PROTECTED.txt',
    'work-mcp1k-prod0813':   'vertex_labels/ source arm (407 labels); PROTECTED.txt',
    'work-r1qlmc-prod0813':  'LIVE on bokeh :5017 and :5018 right now; PROTECTED.txt',
    'work-r2mc-prod0813':    'LIVE on bokeh :5017 right now; PROTECTED.txt',
    'work-mcp1k-ma10':       'baselines.deployed_dump_path() target, 407-event DL-vertex analysis set',
    'work-nuecc48-ma10':     'baselines.deployed_dump_path() target, 47-event DL-vertex analysis set',
    'work-ncpi0-ma10':       'baselines.deployed_dump_path() target, 19-event DL-vertex analysis set',
    # --- requirement 3: latest production (6 names) ----------------------
    'work-mcp1k-pr87ion3':   'doc pr/86 sec.15 knobs-on = current SBND production (toolkit 771f075b)',
    'work-nuecc48-pr87ion3': 'doc pr/86 sec.15 knobs-on = current SBND production (toolkit 771f075b)',
    'work-ncpi0-pr87ion3':   'doc pr/86 sec.15 knobs-on = current SBND production (toolkit 771f075b)',
    'work-pr87-postflip-mcp1k':   'ONLY physical evidence pr87ion3 == bare production (doc sec.15.4)',
    'work-pr87-postflip-nuecc48': 'ONLY physical evidence pr87ion3 == bare production (doc sec.15.4)',
    'work-pr87-postflip-ncpi0':   'ONLY physical evidence pr87ion3 == bare production (doc sec.15.4)',
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
                   (r'^work-.*-prod0813$', 'prod0813-arms'),
                   (r'^work-gate0813', 'gate0813-arms'),
                   (r'^work-.*-ma10', 'ma10-family'),
                   (r'^work-mcp1k-ft2u4$', 'ft2u4-arm'),
                   (r'^work-hvgate0815', 'hvgate0815-arms'),
                   (r'^work-hvdiag0815', 'hvdiag0815-arms'),
                   (r'^work-pr83', 'pr83-family'),
                   (r'^work-.*-pr83(off|on)', 'pr83-family'),
                   (r'^work-pr85', 'pr85-family'),
                   (r'^work-.*-pr85io', 'pr85-family'),
                   (r'^work-pr87-postflip', 'pr87-postflip'),
                   (r'^work-.*-pr87', 'pr87-family'),
                   (r'^work-pr86', 'pr86-family'),
                   (r'^work-.*-pr86', 'pr86-family'),
                   (r'^work-beeprod0813', 'beeprod0813-probe'),
                   (r'^work-prdisp', 'prdisp-arms'),
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
# HEAVY carried unchanged from 08-13/08-11: pctree/mabc/calib/npz/clusters/
# opflash/tracking/oc56scan.  Not re-verified against this round's universe
# beyond a spot check -- if a new HEAVY-sized class appears (e.g. a new
# per-event product introduced by pr/79-86), the archiver's integrity gate
# will still catch a mismatch even if this list is incomplete.
HEAVY = (re.compile(r'^pctree.*\.tar\.gz$'), re.compile(r'^mabc.*\.zip$'),
         re.compile(r'^calib(-pr)?-evt.*\.json(\.gz)?$'), re.compile(r'.*\.npz$'),
         re.compile(r'^clusters-apa.*\.tar\.gz$'),
         re.compile(r'^opflash_apa.*\.tar\.gz$'),
         re.compile(r'^tracking-pr\.root$'),
         re.compile(r'^oc56scan-evt.*\.jsonl$'))


def is_heavy(f):
    return any(p.match(f) for p in HEAVY)


RECORD_DIR = re.compile(r'^(nusel_labels|ql_labels|decisions.*)$')

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
print("=== RETIREMENT ROUND 2026-08-16 (pr/79-86 sweep) ===")
print(f"universe {len(dirs)} work* dirs -> KEEP {len(K)}, remove {len(R)}")
if missing_keep:
    print(f"!! KEEP names not on disk: {missing_keep}")
print("\n[KEEP]")
for d in sorted(K):
    print(f"    {d:30s} {per.get(d,{}).get('tot',0)/2**20:8.0f} MB  {KEEP_WHY[d]}")
print(f"    {'--- KEEP TOTAL':30s} {sum(per[d]['tot'] for d in K)/2**30:8.2f} GiB")
print(f"\n[REMOVE] {len(R)} dirs, {sum(per[d]['tot'] for d in R)/2**30:.2f} GiB")
for d in sorted(R, key=lambda x: -per[x]['tot'])[:30]:
    print(f"    {d:30s} {per[d]['tot']/2**20:8.0f} MB  [{group(d)}]")
if len(R) > 30:
    print(f"    ... and {len(R)-30} more (full list: tierA_20260816.txt)")

print(f"\n[RELEASED from PROTECTED.txt, superseded 2026-08-16] {len(RELEASED)}")
print("    " + (" ".join(RELEASED) if RELEASED else "(none)"))

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

print("\n=== ASSERT 7 (NEW): PROTECTED.txt is honoured, not merely printed ===")
# 08-13's plan.py parsed PROTECTED.txt but only ever printed or intersected the
# result with KEEP -- tier() never consulted it, so a name added to
# PROTECTED.txt AFTER a round's plan script was written would be silently
# swept by the NEXT round's straight fork.  That is exactly what would have
# happened here: PROTECTED.txt was edited at 20:37 on 08-13, after that
# round's 19:57 run, adding the five prod0813 lines.  This assert makes the
# check explicit and blocking rather than trusting KEEP_WHY by inspection.
prot_missing = sorted(PROT_LISTED - KEEP)
if not prot_missing:
    print(f"0 -- PASS  ({len(PROT_LISTED)} PROTECTED.txt names, all in KEEP)")
else:
    fail += 1
    print(f"  !! PROTECTED.txt names NOT in KEEP (would be swept): {prot_missing}")

# ---------------------------------------------------------------- emit
with open(os.path.join(SCR, "tierA_20260816.txt"), "w") as fh:
    fh.write("\n".join(sorted(TA)) + "\n")
json.dump({'A': sorted(TA), 'D': [], 'R': R, 'ARCHIVE': ARCHIVE,
           'KEEP': sorted(K), 'KEEP_WHY': KEEP_WHY, 'HUB': [], 'POSTBUILD': [],
           'PROTECTED': sorted(KEEP & PROT_LISTED), 'RELEASED': RELEASED,
           'per': per, 'cites': {},
           'group': {d: group(d) for d in dirs}},
          open(os.path.join(STATE, "plan.json"), "w"), indent=1)
print(f"\nremoval set: {len(R)} dirs -> scripts/retire/tierA_20260816.txt")
print(f"survivors: {len(K)}")
print(f"state: {STATE}/plan.json")
print("\nOVERALL: " + ("PASS -- all asserts clean" if not fail
                       else f"FAIL -- {fail} assert(s) tripped, do not proceed"))
sys.exit(0 if not fail else 1)
