#!/usr/bin/env python3
"""Retirement round 2026-08-19 -- the pr/40 r7+r9 / pr/83 / pr/84 / pr/91-94
sweep, 149 G -> ~54 G.  PASS 1 of a two-pass round (see Phase 5 of the plan:
a same-day follow-up releases what the new prod0819 arms supersede).

Fork of plan_20260817.py.  tier() UNCHANGED (KEEP-only, two classes).  Asserts
1-8 carried; ONE new assert (9) added.  KEEP grows 30 -> 51: the 08-17 thirty
plus the 21 doc-pr/94 r3 flip-evidence arms.  group() gains ten buckets for the
pr/40..pr/94 generation.

Why this round exists: pr/40 rounds 7+9, pr/83 rounds 2-4, pr/84 rounds 2-3,
pr/91 rounds 1-4, pr/92 rounds 1-2, pr/93 rounds 1-4 and pr/94 (phases 1-6)
regrew the tree from the 08-17 round's 30 survivors / 52 G to 362 work-* dirs
/ 149 G in 54 hours.  Every one of those arms is a leg of an A/B whose verdict
already lives in its doc, and production has moved twice since (pr/93 round-4
flip, then pr/94 Phase 6 at toolkit fd6a116d TODAY, four knobs ON).

THIS ROUND RUNS *BEFORE* A CAMPAIGN, not only after one.  The owner asked to
clean the tree and then re-run Q/L + the full PR chain for nueCC48 / NCpi0 /
mcp1k (1000) / mcp2k (2000) at current production (tag ql0819 -> prod0819).
So KEEP must be closed FORWARD over the campaign's input set, not only
backward over existing evidence -- that is ASSERT 9, new here.  Every prior
round only ever needed the backward direction.

docs/work-tags.md IS AGAIN ONE CAMPAIGN GENERATION STALE -- its newest
retirement section is 2026-08-17 and it has zero mentions of pr91/pr92/pr93/
pr94/pr83r3/pr83r4/pr84r2/pr40r9/latestcheck.  This KEEP dict is derived from
DISK EVIDENCE (ls -1d work-*, git ls-files, and grepping each retained pr/94
arm's own log for its ql_evt provenance), not from the doc.  Closing that gap
in docs/work-tags.md is part of this round's deliverable.

KEEP is 51 names in four groups:

  1. Campaign INPUT (11): the six work-img-* imaging hubs -- which Phase 2
     symlinks per event -- plus the five work-*-cb0805 Q/L hubs, still
     run_valfast.sh's only PR-tail pins.
  2. Git-tracked / not reproducible (3): work-tfix388-r9, work-stmcamp-d66new,
     work-nuecc48-prsmoke2.  Verified: ZERO git-tracked files and ZERO label
     dirs anywhere in the 332 new dirs.
  3. PROTECTED.txt active (17): the four work-cbr3-*, five -vfcbr3on, five
     vf*-cbr3on, work-{r1qlmc,r2mc}-prod0813, work-tfix388-r9.  ALL kept this
     pass even though today's flip supersedes some -- releasing them is pass
     2's job, after the replacement exists.  (NB: PROTECTED.txt's "LIVE on
     bokeh :5017/:5018" justification is STALE -- nothing is listening on
     5017-5019 today.  The prod0813 pair is kept on its OTHER stated ground:
     the only PR product for either MC sample, and neither MC sample is in
     this campaign.)
  4. doc pr/94 Phase 6 flip evidence (21): the work-pr94r3*/r3b* family, the
     only on-disk proof of a production flip that is hours old.

NOT retired here and NOT thinned: dl_vtx_training (already 0 *.pth / 67 M
after 08-17 -- there is no thin_dlruns_20260819.py, nothing to do), and
work-cbr3-census-on must NOT be "thinned to ql_evt* only" -- it acquired six
pr_evt* dirs on 08-18 18:09 and its 3000 ql_evt* are pristine at 08-17 13:57,
so doc 73 sec.12.9's 4/4 hash claim still holds.

Writes scripts/retire/tierA_20260819b.txt and state-20260819b/plan.json.
Read-only w.r.t. work-*.
"""
import os, re, json, subprocess, collections, sys, filecmp, time

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
SCR = os.path.join(ROOT, "scripts", "retire")
STATE = os.environ.get("RETIRE_STATE", os.path.join(SCR, "state-20260819b"))
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
    # --- NOT MINE: the concurrent session's LIVE doc pr/96 round, 5 ------
    # Found by the pass-2 dry run, which listed these for removal.  They were
    # written 23:00-23:27 tonight, alongside that session's commit f0e69780,
    # and docs/pr/96_uncovered-vertex-prongs.md cites them -- an OPEN round
    # with a residual still unresolved.  dbg1 holds exactly evts 70084 and
    # 279955, the two events that doc is about.  Two Claude sessions share
    # this tree; sweeping another session's working arms is not this round's
    # call to make.  Cost of keeping them: 43 MB.
    'work-pr96-dbg1-mcp2k':  "concurrent session's live doc pr/96 debug arm (evts 70084+279955); NOT this round's to retire",
    'work-pr96-dbg2-mcp2k':  "concurrent session's live doc pr/96 debug arm; NOT this round's to retire",
    'work-pr96-dbg3-mcp2k':  "concurrent session's live doc pr/96 debug arm; NOT this round's to retire",
    'work-pr96-fx1-mcp2k':   "concurrent session's live doc pr/96 fix arm; NOT this round's to retire",
    'work-pr96gate-disp':    'AMBIGUOUS OWNERSHIP (evts 47036/47982/49657, 23:27:50, between my two pr96gate arms) -- kept because I cannot prove it is mine; 6.8 MB',
    # --- campaign input, 11 (unchanged from pass 1) -----------------------
    'work-img-mcp1k':        'imaging hub, run_valfast.sh + run_full1k_nusel.sh pin; ql0819 evt<N> links resolve here',
    'work-img-mcp2k':        'imaging hub, mcp2k; ql0819 evt<N> links resolve here',
    'work-img-nuecc48':      'imaging hub, run_valfast.sh pin',
    'work-img-ncpi0':        'imaging hub, run_valfast.sh pin',
    'work-img-r1qlmc':       'imaging hub; only copy of this sim sample',
    'work-img-r2mc':         'imaging hub; only copy of this sim sample',
    'work-mcp1k-cb0805':     'Q/L hub = run_valfast.sh PR-tail pin (pinned_qlroot(); re-pinning to ql0819 is a deliberate later change, not a cleanup side effect)',
    'work-nuecc48-cb0805':   'Q/L hub = run_valfast.sh PR-tail pin',
    'work-ncpi0-cb0805':     'Q/L hub = run_valfast.sh PR-tail pin',
    'work-r1qlmc-cb0805':    'Q/L hub = run_valfast.sh PR-tail pin (10 sim)',
    'work-r2mc-cb0805':      'Q/L hub = run_valfast.sh PR-tail pin (13 sim)',
    # --- git-tracked / not reproducible, 3 --------------------------------
    'work-tfix388-r9':       'doc pr/28 sec.15.9 -- NOT reproducible from any surviving input',
    'work-stmcamp-d66new':   'git-tracked nusel_labels/ hand-scan state (M13)',
    'work-nuecc48-prsmoke2': '3 git-tracked runner scripts',
    # --- the two MC samples: NOT in this campaign, so cbr3on/prod0813 are
    #     still their latest and only products, 6 ------------------------
    'work-r1qlmc-prod0813':  'only PR product for this sim sample; PROTECTED.txt (its ":5017 LIVE" ground is stale -- nothing listens on 5017-5019 today -- but "only PR product" holds)',
    'work-r2mc-prod0813':    'only PR product for this sim sample; PROTECTED.txt',
    'work-r1qlmc-vfcbr3on':  'post-flip Q/L for r1qlmc, not reprocessed by prod0819; input to work-vfr1qlmc-cbr3on',
    'work-r2mc-vfcbr3on':    'post-flip Q/L for r2mc, not reprocessed by prod0819; input to work-vfr2mc-cbr3on',
    'work-vfr1qlmc-cbr3on':  'latest PR out-root for r1qlmc; reads work-r1qlmc-vfcbr3on',
    'work-vfr2mc-cbr3on':    'latest PR out-root for r2mc; reads work-r2mc-vfcbr3on',
    # --- the new single-epoch baseline, 8 (THE point of the round) --------
    'work-nuecc48-ql0819':   'prod0819 Q/L root, 48 evts, bare production at toolkit fd6a116d',
    'work-ncpi0-ql0819':     'prod0819 Q/L root, 19 evts',
    'work-mcp1k-ql0819':     'prod0819 Q/L root, 1000 evts',
    'work-mcp2k-ql0819':     'prod0819 Q/L root, 2000 evts',
    'work-nuecc48-prod0819': 'THE baseline: full 13-stage PR chain, 48 evts; reads work-nuecc48-ql0819',
    'work-ncpi0-prod0819':   'THE baseline, 19 evts; reads work-ncpi0-ql0819',
    'work-mcp1k-prod0819':   'THE baseline, 1000 evts; reads work-mcp1k-ql0819',
    'work-mcp2k-prod0819':   'THE baseline, 2000 evts; reads work-mcp2k-ql0819',
    # --- the mixed-binary equivalence proof, 3 (NEW, doc pr/95 sec 4b) ------
    # A concurrent session committed f0e69780 (doc pr/96, env-gated log-only
    # probe) and relinked libWireCellClus.so at 23:04:06, MID-CAMPAIGN: md5
    # 3fae4385... -> 75652e60..., while mcp2k's PR arm was running.  These arms
    # are the on-disk proof that the two binaries are byte-identical with
    # WCT_PR96_REMSEG_DEBUG unset, which is what makes the baseline valid
    # despite spanning the relink.  Without them the claim is text-only --
    # exactly the work-pr87-postflip-* loss PROTECTED.txt records twice.
    'work-pr96gate-mcp2k':   '12 pre-relink mcp2k events re-run on the POST-relink binary; pr85_hash_gate PASS 24/24 archives + 12/12 nusel TSVs',
    'work-pr96gate-nuedisp': '3 nueCC48 events with PR_EXTRA_STAGES=pr_display; PASS 6/6 archives + 3/3 calib-pr json (1.05-1.17 MB each) + 3/3 nusel TSVs -- closes the pr_display gate hole doc pr/94 flagged',
    'work-probe178410a':     'the ONLY on-disk proof the mcp2k evt 178410 SIGSEGV is non-deterministic: rc=0 / maxrss 683 MB at -j 1 vs rc=139 / 2403 MB at -j 32',
}
KEEP = set(KEEP_WHY)

# Provenance edges checked by ASSERT 8: PR arm -> the Q/L root its own log
# says it read.  Hand-verified once (grep) when this dict was written; the
# assert re-verifies on every run so a stale hardcode cannot silently drift.
PR_PROVENANCE = {
    'work-nuecc48-prod0819': 'work-nuecc48-ql0819',
    'work-ncpi0-prod0819':   'work-ncpi0-ql0819',
    'work-mcp1k-prod0819':   'work-mcp1k-ql0819',
    'work-mcp2k-prod0819':   'work-mcp2k-ql0819',
    'work-vfr1qlmc-cbr3on':  'work-r1qlmc-vfcbr3on',
    'work-vfr2mc-cbr3on':    'work-r2mc-vfcbr3on',
    'work-r1qlmc-prod0813':  'work-r1qlmc-cb0805',
    'work-r2mc-prod0813':    'work-r2mc-cb0805',
}

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
                   (r'^work-mcp2k-cb0816$', 'cb0816-hub'),
                   (r'^work-.*-prod0813$', 'prod0813-arms'),
                   (r'^work-.*-pr87ion3$', 'pr87ion3-family'),
                   (r'^work-.*-harv3$', 'harv3-arms'),
                   (r'^work-mcp1k-kink90', 'kink90-arms'),
                   (r'^work-mcp2k-c2', 'c2-arms'),
                   (r'^work-cbrtrace-', 'cbrtrace-arms'),
                   (r'^work-.*-pr90', 'pr90-family'),
                   (r'^work-.*-pr89', 'pr89-family'),
                   (r'^work-.*-(ql|prod)0819$', 'prod0819-baseline'),
                   (r'^work-pr94r3', 'pr94-final'),
                   (r'^work-pr94', 'pr94-intermediate'),
                   (r'^work-pr93', 'pr93-family'),
                   (r'^work-pr92', 'pr92-family'),
                   (r'^work-pr91', 'pr91-family'),
                   (r'^work-pr84', 'pr84-family'),
                   (r'^work-pr83', 'pr83-family'),
                   (r'^work-pr40', 'pr40-family'),
                   (r'^work-.*-latestcheck$', 'latestcheck-arms'),
                   (r'^work-prod0819-', 'prod0819-spotcheck'),
                   (r'^work-cbr3-(census-on|census-pr-on|bare2evt)', 'cbr3-production'),
                   (r'^work-cbr3-', 'cbr3-family'),
                   (r'^work-cbr2-', 'cbr2-family'),
                   (r'^work-(mcp1kall|ncpi0|nuecc48|r1qlmc|r2mc)-vfcbr3', 'vfcbr3-nusel-roots'),
                   (r'^work-vf(mcp1k|ncpi0|nuecc48|r1qlmc|r2mc)-cbr3', 'valfast-cbr3-prroots'),
                   (r'^work-vf', 'valfast-transient'),
                   (r'^work-nuecc48', 'nuecc48-arms'),
                   (r'^work-ncpi0', 'ncpi0-arms'),
                   (r'^work-mcp2k', 'mcp2k-arms'),
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
# HEAVY carried unchanged from 08-13/08-11/08-16: pctree/mabc/calib/npz/
# clusters/opflash/tracking/oc56scan.  The archiver's integrity gate still
# catches a mismatch even if a new per-event product class was introduced by
# pr/88-90 and is missing from this list.
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
print("=== RETIREMENT ROUND 2026-08-19 PASS 2 (release what prod0819 supersedes) ===")
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
    print(f"    ... and {len(R)-30} more (full list: tierA_20260819b.txt)")

print(f"\n[RELEASED from PROTECTED.txt, superseded by prod0819 (pass 2)] {len(RELEASED)}")
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
    print("    (the tree's only live label dir is work-stmcamp-d66new/nusel_labels, in KEEP;")
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

print("\n=== ASSERT 7: PROTECTED.txt is honoured, not merely printed ===")
prot_missing = sorted(PROT_LISTED - KEEP)
if not prot_missing:
    print(f"0 -- PASS  ({len(PROT_LISTED)} PROTECTED.txt names, all in KEEP)")
else:
    fail += 1
    print(f"  !! PROTECTED.txt names NOT in KEEP (would be swept): {prot_missing}")

print("\n=== ASSERT 8 (NEW): KEEP is closed under PR-arm provenance ===")
# A KEEP PR arm whose Q/L input is being swept is a KEEP name that stops
# being usable the moment the round runs -- no prior round checked this.
# Re-derive each edge from the arm's own log rather than trusting the
# hand-verified PR_PROVENANCE hardcode, so a stale entry cannot pass silently.
prov_fail = []
for pr_arm, expected_src in sorted(PR_PROVENANCE.items()):
    if pr_arm not in KEEP:
        prov_fail.append((pr_arm, expected_src, "arm itself not in KEEP"))
        continue
    evt_dirs = sorted(e for e in os.listdir(pr_arm) if e.startswith('pr_evt')) \
        if os.path.isdir(pr_arm) else []
    found_src = None
    for evt in evt_dirs[:1]:
        for f in os.listdir(os.path.join(pr_arm, evt)):
            if not f.endswith('.log'):
                continue
            with open(os.path.join(pr_arm, evt, f), errors='replace') as fh:
                m = re.search(r'(work-[a-zA-Z0-9-]+)/ql_evt', fh.read())
                if m:
                    found_src = m.group(1)
                    break
        break
    if found_src is None:
        prov_fail.append((pr_arm, expected_src, "no ql_evt provenance line found in log"))
    elif found_src != expected_src:
        prov_fail.append((pr_arm, expected_src, f"log actually reads {found_src}"))
    elif found_src not in KEEP:
        prov_fail.append((pr_arm, expected_src, f"{found_src} not in KEEP"))
    else:
        print(f"  OK  {pr_arm:26s} -> {found_src} (in KEEP)")
if not prov_fail:
    print(f"0 -- PASS  ({len(PR_PROVENANCE)} PR arms, every Q/L input confirmed in KEEP)")
else:
    fail += 1
    for arm, exp, why in prov_fail:
        print(f"  !! {arm} (expected src {exp}): {why}")

print("\n=== ASSERT 9 (NEW): KEEP is closed FORWARD over the campaign input set ===")
# Every prior round only asserted KEEP closed BACKWARD (assert 8: a KEEP PR
# arm's Q/L source is in KEEP).  This round sweeps BEFORE the ql0819/prod0819
# campaign, so KEEP must also contain everything that campaign reads.
#
# The failure this catches is silent, not loud: run_ql_batch.sh:51-53 writes
# "rc=91 ... no-imaging" and EXITS 0 when $IMGBASE/evt<N> is absent, so a
# missing or thinned imaging hub degrades the arm to a short one rather than
# erroring.  A count gate downstream would then compare 900 events to 900.
CAMPAIGN_IMG = {          # imaging hub -> expected evt<N> count
    'work-img-nuecc48':  48,
    'work-img-ncpi0':    19,
    'work-img-mcp1k':  1000,
    'work-img-mcp2k':  2000,
    'work-img-r1qlmc':   10,   # not in the campaign, but the only copy of the sample
    'work-img-r2mc':     13,
}
# Pass 2 runs AFTER the campaign, so assert 9 also requires the products.
CAMPAIGN_OUT = {
    'work-nuecc48-ql0819': ('ql_evt',   48), 'work-nuecc48-prod0819': ('pr_evt',   48),
    'work-ncpi0-ql0819':   ('ql_evt',   19), 'work-ncpi0-prod0819':   ('pr_evt',   19),
    'work-mcp1k-ql0819':   ('ql_evt', 1000), 'work-mcp1k-prod0819':   ('pr_evt', 1000),
    'work-mcp2k-ql0819':   ('ql_evt', 2000), 'work-mcp2k-prod0819':   ('pr_evt', 2000),
}
CAMPAIGN_INPUT = {        # staged/extracted SP input -> expected map line count (None = no map)
    'input_files_reco1/staged-mcp2025c-1000evt':      1001,
    'input_files_reco1/staged-mcp2025c-2nd-2000evt':  2001,
    'input_files_reco1/extracted-2025fall-48evt-fsprod': None,
    'input_files_reco1/extracted-ncpi0':                 None,
}
a9_fail = 0
for hub, want in sorted(CAMPAIGN_IMG.items()):
    if hub not in KEEP:
        print(f"  !! {hub} is NOT in KEEP -- the campaign would silently skip its events")
        a9_fail += 1
        continue
    got = len([e for e in os.listdir(hub) if e.startswith('evt')]) if os.path.isdir(hub) else 0
    if got != want:
        print(f"  !! {hub}: {got} evt* dirs, expected {want}")
        a9_fail += 1
    else:
        print(f"  OK  {hub:20s} {got:5d} evt* dirs, in KEEP")
for src, want in sorted(CAMPAIGN_INPUT.items()):
    ok = os.path.isdir(src) and bool(os.listdir(src))
    if not ok:
        print(f"  !! {src}: missing or empty")
        a9_fail += 1
        continue
    emap = os.path.join(src, 'entry_event_map.tsv')
    if want is None:
        print(f"  OK  {src}  ({len(os.listdir(src))} entries, no entry_event_map.tsv expected)")
    elif not os.path.exists(emap):
        print(f"  !! {src}: entry_event_map.tsv missing (run_ql_batch.sh:90-97 needs it)")
        a9_fail += 1
    else:
        n = sum(1 for _ in open(emap))
        if n != want:
            print(f"  !! {emap}: {n} lines, expected {want}")
            a9_fail += 1
        else:
            print(f"  OK  {src}  entry_event_map.tsv {n} lines")
for arm, (pfx, want) in sorted(CAMPAIGN_OUT.items()):
    if arm not in KEEP:
        print(f"  !! {arm} is NOT in KEEP -- pass 2 would sweep the new baseline")
        a9_fail += 1
        continue
    got = len([e for e in os.listdir(arm) if e.startswith(pfx)]) if os.path.isdir(arm) else 0
    if got != want:
        print(f"  !! {arm}: {got} {pfx}* dirs, expected {want}")
        a9_fail += 1
    else:
        print(f"  OK  {arm:24s} {got:5d} {pfx}* dirs, in KEEP")
# The SP input dirs must not be reachable from this plan's work* universe at all.
reachable = [c for c in CAMPAIGN_INPUT if c.split('/')[0].startswith('work')]
if reachable:
    print(f"  !! campaign input inside the work* universe: {reachable}")
    a9_fail += 1
if not a9_fail:
    print(f"0 -- PASS  ({len(CAMPAIGN_IMG)} imaging hubs + {len(CAMPAIGN_INPUT)} SP inputs + "
          f"{len(CAMPAIGN_OUT)} prod0819 arms, counts verified, all in KEEP)")
else:
    fail += 1

# ---------------------------------------------------------------- emit
with open(os.path.join(SCR, "tierA_20260819b.txt"), "w") as fh:
    fh.write("\n".join(sorted(TA)) + "\n")
json.dump({'A': sorted(TA), 'D': [], 'R': R, 'ARCHIVE': ARCHIVE,
           'KEEP': sorted(K), 'KEEP_WHY': KEEP_WHY, 'HUB': [], 'POSTBUILD': [],
           'PROTECTED': sorted(KEEP & PROT_LISTED), 'RELEASED': RELEASED,
           'PR_PROVENANCE': PR_PROVENANCE,
           'per': per, 'cites': {},
           'group': {d: group(d) for d in dirs}},
          open(os.path.join(STATE, "plan.json"), "w"), indent=1)
print(f"\nremoval set: {len(R)} dirs -> scripts/retire/tierA_20260819b.txt")
print(f"survivors: {len(K)}")
print(f"state: {STATE}/plan.json")
print("\nOVERALL: " + ("PASS -- all asserts clean" if not fail
                       else f"FAIL -- {fail} assert(s) tripped, do not proceed"))
sys.exit(0 if not fail else 1)
