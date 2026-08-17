#!/usr/bin/env python3
"""Retirement round 2026-08-17 -- the pr/88-90 + cathode-rescue sweep, 164 G -> ~52 G.

Fork of plan_20260816.py.  Asserts 1-7 unchanged; KEEP grows from 27 (08-16
Stage-1) / 18 (post-follow-up survivors) to 30 names covering a DIFFERENT
generation of campaigns; group() gets new buckets; ONE new assert (8) is
added.  tier() is UNCHANGED -- KEEP-only.

Why this round exists: doc pr/89 (DL-vertex round 4, topo term NET NEGATIVE
live, swap-guard killed -36), doc pr/90 (unbroken kink, rounds 1-4, D4-only
shipped) and docs 72/73 (cathode rescue rounds 2+3, SEVEN KNOBS flipped into
SBND production TODAY at toolkit 2d8c9e5a) regrew the tree from the 08-16
round's 18 survivors / 23G to 178 work-* dirs / 164G in one day.  Every one
of those arms is a leg of an A/B whose verdict already lives in its doc.

docs/work-tags.md IS ONE CAMPAIGN GENERATION STALE -- it has zero mentions of
cb0816, harv3, mcp2k, pr89, pr90, cbr2, cbr3.  This KEEP dict is derived from
DISK EVIDENCE (grepping wct_pr_evt*.log / wct_ql_evt*.log for provenance,
verifying doc 73 sec.12.9's flip-gate claims), not from the doc.  Closing
that gap in docs/work-tags.md is part of this round's deliverable, done
separately after the sweep lands.

KEEP is 30 names, driven by the owner's two requirements (2026-08-17):

  1. DL-vertex training: retire the unsuccessful arms (handled by the
     separate thin_dlruns_20260817.py, not this script -- dl_vtx_training is
     not a work-* dir).
  2. Keep the LATEST Q/L result for nueCC / NCpi0 / mcp1k (1000-evt data) /
     mcp2k (2000-evt data), and the PR result built on it; retire the rest.
     Owner's explicit choice of "latest": the POST-FLIP arms from today's
     seven-knob production flip (doc 73 sec.12.9, toolkit 2d8c9e5a), not the
     pre-flip cb0805/cb0816 + harv3 family.  work-{mcp1k,nuecc48,ncpi0}
     -pr87ion3 (yesterday's "latest") is explicitly RELEASED from
     PROTECTED.txt this round -- superseded by the flip.

Full derivation of each name in docs/work-tags.md "RETIREMENT ROUND
2026-08-17" (written after this sweep) and in the approved plan file at
/home/xqian/.claude/plans/hi-the-sbnd-xin-directory-eventual-star.md.
Short version:
  - work-img-* (6, incl. NEW work-img-mcp2k) + work-*-cb0805 (5) --
    run_valfast.sh's [ -d "$QL" ] check refuses PR-tail mode without them;
    this is exactly how the 2026-08-05 round mechanically killed it before.
  - work-tfix388-r9 / work-stmcamp-d66new / work-nuecc48-prsmoke2 --
    git-tracked or not reproducible (M13), unchanged since 08-11.
  - work-{r1qlmc,r2mc}-prod0813 -- PROTECTED.txt, LIVE on bokeh :5017/:5018.
    NOT released this round (owner released only the pr87ion3 line).
  - work-cbr3-census-on / -census-pr-on / -bare2evt / -bare2evt-pr -- doc 73
    sec.12.9: a BARE run_ql_batch.sh with no envs reproduces census-on's
    mabc-all-apa.zip + pctree member hashes 4/4 MATCH.  census-on is Q/L
    over 3000 events (mcp1k 1000 + mcp2k 2000); census-pr-on is the PR chain
    on the 40 behavior-changed events (VERIFIED: reads
    work-cbr3-census-on/ql_evt).  bare2evt{,-pr} is the ONLY on-disk
    evidence the shipped jsonnet reproduces census-on -- same role
    work-pr87-postflip-* played at 08-16 (VERIFIED: reads
    work-cbr3-bare2evt/ql_evt, which symlinks its evt<N> into work-img-mcp1k).
  - work-{mcp1kall,ncpi0,nuecc48,r1qlmc,r2mc}-vfcbr3on (post-flip Q/L, valfast
    -full nusel roots with the tagger tail census-on lacks) + work-vf
    {mcp1k,ncpi0,nuecc48,r1qlmc,r2mc}-cbr3on (post-flip PR out-roots).  EACH
    PR arm's log VERIFIED to read its matching -vfcbr3on/ql_evt (grepped
    individually, not inferred) -- see ASSERT 8.

  ==> STILL NO PHASE 4 HUB THINNING THIS ROUND: the five work-*-cb0805 hubs
      remain run_valfast.sh's PR-tail pin.  thin_hubs_20260811.py must NOT
      be re-run.

KNOWN COST, stated rather than silently absorbed (full account in
docs/work-tags.md "RETIREMENT ROUND 2026-08-17"):
  - work-mcp2k-cb0816 (17G) + the four work-*-harv3 arms (9.5G) retire.
    vtx_rules/baselines.py deployed_dump_path() (-prod0813 -> -harv3) stops
    resolving AGAIN; vtx_rules/scankit.py:858's three named harv3 arms are
    gone.  The 1543 vertex_labels/ hand-scan labels (13 tags) SURVIVE
    (self-contained JSON, same precedent as uitest75/vtxscan1 and the 08-16
    -prod0813 drop) but the calib dumps they were scanned against do not.
  - mcp2k has NO complete PR product after this round -- only the 40 events
    in work-cbr3-census-pr-on.
  - work-cbr3-census-offfull (9.6G) + work-cbr3-census-pr-off retire: doc 73
    sec.12.6's both-directions census becomes one-sided ON DISK; the numbers
    survive in the doc text only.
  - doc pr/89 / pr/90 A/B off-arms retire; a pairwise gate is no longer
    re-runnable from these arms (PROTECTED.txt's own "a floor is a PAIR"
    caution applies -- accepted because both rounds are CLOSED and flipped
    or reverted).
  - work-{mcp1k,nuecc48,ncpi0}-pr87ion3 (3.4G) RELEASED from PROTECTED.txt,
    superseded by today's flip; owner-confirmed 2026-08-17.

Writes scripts/retire/tierA_20260817.txt and state-20260817/plan.json.
Read-only w.r.t. work-*.
"""
import os, re, json, subprocess, collections, sys, filecmp, time

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
SCR = os.path.join(ROOT, "scripts", "retire")
STATE = os.environ.get("RETIRE_STATE", os.path.join(SCR, "state-20260817"))
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
    # --- infrastructure / input (11 names) --------------------------------
    'work-img-mcp1k':        'imaging hub, run_valfast.sh + run_full1k_nusel.sh pin',
    'work-img-mcp2k':        'imaging hub for mcp2k (NEW since 08-16); census-on evt<N> symlinks resolve here',
    'work-img-nuecc48':      'imaging hub, run_valfast.sh pin',
    'work-img-ncpi0':        'imaging hub, run_valfast.sh pin',
    'work-img-r1qlmc':       'imaging hub, run_valfast.sh pin; only copy of this sim sample',
    'work-img-r2mc':         'imaging hub, run_valfast.sh pin; only copy of this sim sample',
    'work-mcp1k-cb0805':     'Q/L hub = run_valfast.sh PR-tail pin (mode is mechanically dead without it, per 2026-08-05 precedent)',
    'work-nuecc48-cb0805':   'Q/L hub = run_valfast.sh PR-tail pin',
    'work-ncpi0-cb0805':     'Q/L hub = run_valfast.sh PR-tail pin',
    'work-r1qlmc-cb0805':    'Q/L hub = run_valfast.sh PR-tail pin (10 sim)',
    'work-r2mc-cb0805':      'Q/L hub = run_valfast.sh PR-tail pin (13 sim)',
    # --- git-tracked / not reproducible (3 names) --------------------------
    'work-tfix388-r9':       'doc pr/28 sec.15.9 -- NOT reproducible from any surviving input',
    'work-stmcamp-d66new':   'git-tracked nusel_labels/ hand-scan state (M13)',
    'work-nuecc48-prsmoke2': '3 git-tracked runner scripts',
    # --- PROTECTED, live on the bokeh viewers (2 names) --------------------
    'work-r1qlmc-prod0813':  'LIVE on bokeh :5017 and :5018 right now; PROTECTED.txt; only PR product for this sim sample',
    'work-r2mc-prod0813':    'LIVE on bokeh :5017 right now; PROTECTED.txt; only PR product for this sim sample',
    # --- post-flip production, the owner's "latest" (14 names) ------------
    'work-cbr3-census-on':      'Q/L, 3000 evts (mcp1k+mcp2k), all 7 flip knobs ON; doc 73 sec.12.9 bare-run 4/4 hash MATCH == production',
    'work-cbr3-census-pr-on':   'PR chain on the 40 behavior-changed events; VERIFIED reads work-cbr3-census-on/ql_evt',
    'work-cbr3-bare2evt':       'Q/L, 2 evts, ZERO envs -- the bare-production leg of doc 73 sec.12.9\'s hash-match proof',
    'work-cbr3-bare2evt-pr':    'ONLY on-disk evidence shipped jsonnet == census-on; VERIFIED reads work-cbr3-bare2evt/ql_evt',
    'work-mcp1kall-vfcbr3on':   'post-flip Q/L (valfast -full nusel root, mcp1k); input to work-vfmcp1k-cbr3on',
    'work-ncpi0-vfcbr3on':      'post-flip Q/L (valfast -full nusel root, ncpi0); input to work-vfncpi0-cbr3on',
    'work-nuecc48-vfcbr3on':    'post-flip Q/L (valfast -full nusel root, nuecc48); input to work-vfnuecc48-cbr3on',
    'work-r1qlmc-vfcbr3on':     'post-flip Q/L (valfast -full nusel root, r1qlmc); input to work-vfr1qlmc-cbr3on',
    'work-r2mc-vfcbr3on':       'post-flip Q/L (valfast -full nusel root, r2mc); input to work-vfr2mc-cbr3on',
    'work-vfmcp1k-cbr3on':      'post-flip PR out-root, mcp1k; VERIFIED reads work-mcp1kall-vfcbr3on/ql_evt',
    'work-vfncpi0-cbr3on':      'post-flip PR out-root, ncpi0; VERIFIED reads work-ncpi0-vfcbr3on/ql_evt',
    'work-vfnuecc48-cbr3on':    'post-flip PR out-root, nuecc48; VERIFIED reads work-nuecc48-vfcbr3on/ql_evt',
    'work-vfr1qlmc-cbr3on':     'post-flip PR out-root, r1qlmc; VERIFIED reads work-r1qlmc-vfcbr3on/ql_evt',
    'work-vfr2mc-cbr3on':       'post-flip PR out-root, r2mc; VERIFIED reads work-r2mc-vfcbr3on/ql_evt',
}
KEEP = set(KEEP_WHY)

# Provenance edges checked by ASSERT 8: PR arm -> the Q/L root its own log
# says it read.  Hand-verified once (grep) when this dict was written; the
# assert re-verifies on every run so a stale hardcode cannot silently drift.
PR_PROVENANCE = {
    'work-cbr3-census-pr-on': 'work-cbr3-census-on',
    'work-cbr3-bare2evt-pr':  'work-cbr3-bare2evt',
    'work-vfmcp1k-cbr3on':    'work-mcp1kall-vfcbr3on',
    'work-vfncpi0-cbr3on':    'work-ncpi0-vfcbr3on',
    'work-vfnuecc48-cbr3on':  'work-nuecc48-vfcbr3on',
    'work-vfr1qlmc-cbr3on':   'work-r1qlmc-vfcbr3on',
    'work-vfr2mc-cbr3on':     'work-r2mc-vfcbr3on',
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
print("=== RETIREMENT ROUND 2026-08-17 (pr/88-90 + cathode-rescue sweep) ===")
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
    print(f"    ... and {len(R)-30} more (full list: tierA_20260817.txt)")

print(f"\n[RELEASED from PROTECTED.txt, superseded 2026-08-17] {len(RELEASED)}")
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

# ---------------------------------------------------------------- emit
with open(os.path.join(SCR, "tierA_20260817.txt"), "w") as fh:
    fh.write("\n".join(sorted(TA)) + "\n")
json.dump({'A': sorted(TA), 'D': [], 'R': R, 'ARCHIVE': ARCHIVE,
           'KEEP': sorted(K), 'KEEP_WHY': KEEP_WHY, 'HUB': [], 'POSTBUILD': [],
           'PROTECTED': sorted(KEEP & PROT_LISTED), 'RELEASED': RELEASED,
           'PR_PROVENANCE': PR_PROVENANCE,
           'per': per, 'cites': {},
           'group': {d: group(d) for d in dirs}},
          open(os.path.join(STATE, "plan.json"), "w"), indent=1)
print(f"\nremoval set: {len(R)} dirs -> scripts/retire/tierA_20260817.txt")
print(f"survivors: {len(K)}")
print(f"state: {STATE}/plan.json")
print("\nOVERALL: " + ("PASS -- all asserts clean" if not fail
                       else f"FAIL -- {fail} assert(s) tripped, do not proceed"))
sys.exit(0 if not fail else 1)
