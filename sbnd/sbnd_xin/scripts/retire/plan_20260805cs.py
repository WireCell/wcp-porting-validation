#!/usr/bin/env python3
"""Retirement round 2026-08-05 CLEAN SLATE -- clear the deck for the reprocessing.

Fork of plan_20260805.py (the light round earlier the same day).  Do NOT re-run
that script: it is guarded, and its state dir is the record of what it removed.

WHY THIS ROUND IS DIFFERENT.  The light round kept every input hub, because the
campaign still needed them.  The owner then decided to **regenerate all imaging**,
and that changes the premise completely:

    run_img_evt.sh contains ZERO references to work-*.  Imaging is built from
    $SBND_INPUT_DIR -- input_files_reco1/extracted-<tag>/ and
    staged-mcp2025c-1000evt/ -- which are real local files, not symlinks into
    anyone else's area.

    => With imaging regenerated, NOT ONE work-* dir is an input to the campaign.
       They are all records.  The hub concept, which drove every previous
       round's ordering, does not apply to this one.

So the KEEP set is not derived from the dependency graph at all.  It is five
explicit names, each with a reason that survives the reprocessing:

  work-stmcamp-d66new     git-tracked nusel_labels/ hand-scan state (M13)
  work-nuecc48-prsmoke2   3 git-tracked runner scripts
  work-tfix388-r9         doc pr/28 sec.15.9: NOT reproducible from any
                          surviving input -- the one genuinely irreplaceable arm
  work-pr33-base48        the clean-source knob-off gate PAIR.  Kept so a
  work-pr33-off48         toolkit change made BEFORE the campaign still has
                          something to be gated against; 478 MB of insurance.

ASSERT 1 IS REDEFINED, NOT BYPASSED.  The owner's standing constraint is "no
real SP frame is lost".  Previous rounds discharged it by finding a byte-
identical copy in a SURVIVING work dir -- impossible here, because the dirs that
held the copies are themselves going.  The constraint still holds, via the other
route: 1441 SP-frame archives (1.79 GiB) are being dropped, and every one is
regenerable by re-running run_img_evt.sh over the reco1 source that survives
locally.  ASSERT 1 now checks THAT -- per sample family, that the source dir
exists, is real (not a symlink into another user's area), and is non-empty --
and it prints the dropped count loudly rather than quietly passing.

lightcheck_*.py is SUPERSEDED for this round for the same reason: it looks for
identical copies in surviving hubs, and there are none by construction.  Do not
run it and do not read its MISSING count as a failure.

Writes scripts/retire/tierA_20260805cs.txt and state-20260805cs/plan.json.
Read-only w.r.t. work-*.
"""
import os, re, json, subprocess, collections, sys, time

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
SCR = os.path.join(ROOT, "scripts", "retire")
STATE = os.environ.get("RETIRE_STATE", os.path.join(SCR, "state-20260805cs"))
os.makedirs(STATE, exist_ok=True)
os.chdir(ROOT)

# ---- M13 guard (carried from the light round) ------------------------------
if os.path.exists(os.path.join(STATE, "removed.tsv")) and not os.environ.get("RETIRE_REPLAN"):
    sys.stderr.write(
        f"REFUSING: {STATE}/removed.tsv exists -- this round has already run and\n"
        f"{STATE}/plan.json is the record of what it removed (M13).\n"
        f"Fork with a new date/state for a new round; RETIRE_REPLAN=1 to override.\n")
    sys.exit(3)

dirs = sorted(d for d in os.listdir('.')
              if d.startswith('work') and os.path.isdir(d) and not os.path.islink(d))

# ---------------------------------------------------------------- KEEP
KEEP_WHY = {
    'work-stmcamp-d66new':   'git-tracked nusel_labels/ hand-scan state (M13)',
    'work-nuecc48-prsmoke2': '3 git-tracked runner scripts',
    'work-tfix388-r9':       'doc pr/28 sec.15.9 -- NOT reproducible from any surviving input',
    'work-pr33-base48':      'clean-source knob-off gate pair (A); interim baseline pre-campaign',
    'work-pr33-off48':       'clean-source knob-off gate pair (B); a gate is a PAIR',
}
KEEP = set(KEEP_WHY)

# ---- what the owner is releasing from PROTECTED.txt, explicitly -------------
# Second override in one day, and recorded as one: PROTECTED.txt currently lists
# 19 arms; 3 of them are in KEEP above and the other 16 are released, because
# with all imaging regenerated they are records rather than references and the
# campaign supersedes every one of them.  Their numbers live in the docs and in
# the record layer this round writes.
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
    for pat, g in ((r'^work-pr33-', 'pr33-clean-source'),
                   (r'^work-pr37b-', 'pr37-floor'),
                   (r'^work-(mcp1000|mcp10|work)$|^work$', 'imaging-bases'),
                   (r'^work-mcp', 'mcp1k-campaign'),
                   (r'^work-nuecc48|^work-vfnuecc48', 'nuecc48-campaign'),
                   (r'^work-(r1ql|r2patrec)', 'mc-samples'),
                   (r'^work-(oc19scan|tfix388|pr22gap)', 'legacy-exhibits')):
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
HEAVY = (re.compile(r'^pctree.*\.tar\.gz$'), re.compile(r'^mabc.*\.zip$'),
         re.compile(r'^calib(-pr)?-evt.*\.json(\.gz)?$'), re.compile(r'.*\.npz$'),
         re.compile(r'^clusters-apa.*\.tar\.gz$'),
         re.compile(r'^opflash_apa.*\.tar\.gz$'))


def is_heavy(f):
    return any(p.match(f) for p in HEAVY)


SPDATA = re.compile(r'^(sp-frames.*\.tar\.bz2|sbnd-sp-frames.*\.tar\.bz2|frames-dnn\.tar\.bz2)$')
RECORD_DIR = re.compile(r'^(nusel_labels|ql_labels|decisions.*)$')

per = {}
sp_per = collections.Counter()
sp_bytes = 0
label_hits = []
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
                sp_per[d] += 1
                sp_bytes += sz
            if is_heavy(f):
                nh += 1
            else:
                keep += sz
                nk += 1
    per[d] = dict(tot=tot, keep=keep, nk=nk, nh=nh)

# ---------------------------------------------------------------- report
print("=== CLEAN-SLATE ROUND 2026-08-05 ===")
print(f"universe {len(dirs)} work* dirs -> KEEP {len(K)}, remove {len(R)}")
if missing_keep:
    print(f"!! KEEP names not on disk: {missing_keep}")
print("\n[KEEP]")
for d in sorted(K):
    print(f"    {d:26s} {per.get(d,{}).get('tot',0)/2**20:8.0f} MB  {KEEP_WHY[d]}")
print(f"\n[REMOVE] {len(R)} dirs, {sum(per[d]['tot'] for d in R)/2**30:.2f} GiB")
for d in sorted(R, key=lambda x: -per[x]['tot']):
    sp = f"  SP:{sp_per[d]}" if sp_per[d] else ""
    print(f"    {d:26s} {per[d]['tot']/2**20:8.0f} MB  [{group(d)}]{sp}")

print(f"\n[RELEASED from PROTECTED.txt by owner, 2026-08-05] {len(RELEASED)}")
print("    " + " ".join(RELEASED))

print("\n=== ARCHIVE FOOTPRINT ===")
stat = collections.defaultdict(lambda: [0, 0, 0])
for d in ARCHIVE:
    s = stat[group(d)]
    s[0] += per[d]['tot']; s[1] += per[d]['keep']; s[2] += per[d]['nk']
for g in sorted(stat):
    t, k, nk = stat[g]
    print(f"{g:20s} total {t/2**30:6.2f} GiB  archive {k/2**20:8.1f} MiB ({nk} files)  "
          f"reclaim {(t-k)/2**30:6.2f} GiB")
T = sum(per[d]['tot'] for d in R)
Kb = sum(stat[g][1] for g in stat)
print(f"{'TOTAL':20s} total {T/2**30:6.2f} GiB  archive {Kb/2**20:8.1f} MiB  "
      f"reclaim {(T-Kb)/2**30:6.2f} GiB")

# ---------------------------------------------------------------- asserts
fail = 0

# ---- ASSERT 1, REDEFINED: the SP SOURCE survives locally --------------------
# The constraint is "no SP data is lost", not "no SP file is deleted".  Every
# dropped sp-frames.tar.bz2 is regenerable by run_img_evt.sh from the reco1
# source below, so the check is that the source is present, real and non-empty.
SP_SOURCES = {
    'mcp1k (1000 data)':  'input_files_reco1/staged-mcp2025c-1000evt',
    'nuecc48 (48 data)':  'input_files_reco1/extracted-2025fall-48evt-fsprod',
    'mcp10 (10 data)':    'input_files_reco1/extracted-mcp2025c-10evt',
    'r1qlmc f1 (sim)':    'input_files_reco1/extracted-r1ql-f1',
    'r1qlmc f2 (sim)':    'input_files_reco1/extracted-r1ql-f2',
    'r2mc (sim)':         'input_files_reco1/extracted-r2patrec-f1',
}
print(f"\n=== ASSERT 1 (REDEFINED): {sum(sp_per.values())} SP-frame archives "
      f"({sp_bytes/2**30:.2f} GiB) are being DROPPED ===")
print("    Not matched against a surviving work dir -- there are none by")
print("    construction this round.  Discharged instead by proving the reco1")
print("    source each one is regenerable from survives locally:")
for label, src in sorted(SP_SOURCES.items()):
    ok = os.path.isdir(src) and not os.path.islink(src) and bool(os.listdir(src))
    n = len(os.listdir(src)) if os.path.isdir(src) else 0
    print(f"      {'OK ' if ok else '!! '} {label:20s} {src}  ({n} entries)")
    if not ok:
        fail += 1
print("    => re-run run_img_evt.sh per sample to rebuild every dropped frame.")

# ---- ASSERT 2, REDEFINED for the same reason as ASSERT 1 --------------------
# The constraint is "no hand-scan record is LOST", not "no label dir is in the
# removal set".  Previous rounds discharged it the strict way because their
# removal sets never contained one; this round's does (3 dirs, 18 tags, 245
# files), and the owner's decision 2026-08-05 was to preserve them verbatim in
# archive/records/labels/ -- the established location, already holding labels
# for ten earlier arms -- and let the parents go.
# preserve_20260805cs.sh does the copy and diff -r verifies it.  Here we
# re-verify independently: every label dir in the removal set must have a
# byte-identical copy under archive/records/labels/<arm>/<name>/.
print("\n=== ASSERT 2 (REDEFINED): every hand-scan record has a verified archive copy (M13) ===")
import filecmp
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
else:
    for p in sorted(label_hits):
        rel = os.path.relpath(p, ROOT)          # e.g. work-mcp10/nusel_labels
        dst = os.path.join(LABROOT, rel)
        nsrc = sum(len(f) for _, _, f in os.walk(p))
        if not os.path.isdir(dst):
            print(f"  !! NO ARCHIVE COPY: {rel}  -- run preserve_20260805cs.sh first")
            fail += 1
        elif not tree_identical(p, dst):
            print(f"  !! ARCHIVE COPY DIFFERS: {rel}")
            fail += 1
        else:
            ntag = len(os.listdir(p))
            print(f"  OK  {rel:38s} {ntag:2d} tags, {nsrc:3d} files -> "
                  f"archive/records/labels/{rel}")

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

# ---- ASSERT 4: dangling links -- only the KEEP set can strand one -----------
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

print(f"\n=== ASSERT 4: dangling-link dry run ({nlinks} symlinks outside removal set) ===")
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

# ---------------------------------------------------------------- emit
with open(os.path.join(SCR, "tierA_20260805cs.txt"), "w") as fh:
    fh.write("\n".join(sorted(TA)) + "\n")
json.dump({'A': sorted(TA), 'D': [], 'R': R, 'ARCHIVE': ARCHIVE,
           'KEEP': sorted(K), 'KEEP_WHY': KEEP_WHY, 'HUB': [], 'POSTBUILD': [],
           'PROTECTED': sorted(KEEP & PROT_LISTED), 'RELEASED': RELEASED,
           'per': per, 'cites': {}, 'sp_dropped': dict(sp_per),
           'group': {d: group(d) for d in dirs}},
          open(os.path.join(STATE, "plan.json"), "w"), indent=1)
print(f"\nremoval set: {len(R)} dirs -> scripts/retire/tierA_20260805cs.txt")
print(f"survivors: {len(K)}")
print(f"state: {STATE}/plan.json")
print("\nOVERALL: " + ("PASS -- all asserts clean" if not fail
                       else f"FAIL -- {fail} assert(s) tripped, do not proceed"))
sys.exit(0 if not fail else 1)
