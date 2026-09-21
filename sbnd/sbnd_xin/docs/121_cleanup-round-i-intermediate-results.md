# 121 — cleanup round I (2026-09-21): retire the docs 116/117 ladders, and the log class nine rounds could not see

**Status: PLANNED AND GATED; the destructive steps are the owner's to run** (sec 6). Every
non-destructive step below ran this session: the sentinel baseline, the liveness pin, the
seven-file fork, the census chain, both planners, the pin-dedup measurement and both dry
runs. `/home/xqian` free space going in: **511 G**.

Owner instruction, verbatim (2026-09-21):

> "Now, can you do one round of cleanup for the pdvd pdhd, sbnd_xin and ~/tmp directory? We
> want to keep the latest production as well as the relevant input files. We can retire the
> intermidiate files to save some disks. We have done several round of this in the past.
> Please plan and execute."

and, mid-round: *"after, please update the relevant md file, commit and push"*.

Round H (doc 114, 2026-09-19) closed at **733 G free**. Two days of the docs 115–120
campaigns took 222 G of that back, and **essentially all of it in one tree**: sbnd_xin
76 → 289 G, pdvd 96 → 99, pdhd 82 → 83, `~/tmp` 86 → 91. All four share
`/dev/nvme2n1p6`, so this is one number, not four.

## 0. Repro

```bash
IMG=/home/xqian/toolkit-dev/wcp-porting-img; D=$IMG/pdhd/scripts/retire; cd $D

# the round-start sentinel baseline -- BEFORE anything is touched (doc 91's rule)
python3 $IMG/sbnd/sbnd_xin/scripts/analysis/pr149/sentinels_tolerant.py --arms 'work-*-pr150s0'

# pin liveness to the head that is VERIFIED equal to the remote, not to local main on faith
git -C $IMG -c credential.helper='!gh auth git-credential' \
    ls-remote https://github.com/WireCell/wcp-porting-validation.git refs/heads/main
git -C $IMG rev-parse main            # equal -> remote_head_20260921.txt

# fork the machinery -- the list DERIVED from the drivers' own call sites (sec 2)
for f in toks plan tmp_census archive_records archive_tmp; do
  sed 's/20260919/20260921/g' ${f}_20260919.py > ${f}_20260921.py; done
for f in retire sweep_tmp; do
  sed 's/20260919/20260921/g' ${f}_20260919.sh > ${f}_20260921.sh; chmod +x ${f}_20260921.sh; done

# census -> plan
python3 toks_20260921.py                                       # 8008 dirs -> 8326 tokens
python3 cit_20260916d.py toks_20260921.txt cit_20260921.json   # 8326 -> 765 cited, 213457 hits
python3 scan_arms_20260916d.py --json=scan_arms_20260921.json  # measured, never typed
python3 plan_20260921.py                                       # 39/39 interlocks PASS
python3 tmp_census_20260921.py
python3 plan_orphanlogs_20260921.py                            # 8/8 gates PASS  (NEW, sec 5)
python3 dedup_pins_20260912.py                                 # report only: 2.40 GiB
```

**Round-start sentinel baseline (2026-09-21 11:00, against production `work-*-pr150s0`):
21 PASS, 0 FAIL, 2 OPEN, 7 INERT, 0 SKIP** — the same tally rounds D and H recorded, so
the record is intact going in. Log:
`/home/xqian/tmp/cleanup20260921_sentinels_before.log`. Sec 6 re-runs it after; a
before/after pair is the only form in which the number means anything.

**Operating-point baseline, taken BEFORE anything is deleted** — the same argument as the
sentinels, which sec 8 originally applied only after: `prod_cfg_gate.py --ref
ref/prod-2026-09-21d` → **PASS, 26 artifacts, rc=0**
(`/home/xqian/tmp/cleanup20260921_prodcfg_before.log`). A post-only run cannot tell "this
round broke it" from "it was already broken".

**The sentinel glob still names a live arm, and that was checked rather than assumed.**
Production's *operating point* has moved three times since round H
(`ref/prod-2026-09-20`, `-21c`, `-21d`), but no `d118`/`d119` arm exists on the
`mcp1k`/`mcp2k`/`ncpi0`/`nuecc48` samples, so `pr150s0` is still the newest **full
four-sample stage-B output set** and the glob resolves. A glob pointing at a released arm
either fails to launch or passes vacuously; neither would have been visible in the tally.

## 1. What this round releases

| target | universe | KEEP | RELEASE | freed (set-relative) |
|---|---|---|---|---|
| sbnd_xin | 146 dirs | 124 = 137.01 GiB | **22 dirs** | **130.32 GiB** |
| pdvd | 5705 dirs | 5224 = 67.51 GiB | 481 dirs | **4.90 GiB** |
| pdhd | 2157 dirs | 1913 = 51.83 GiB | 244 dirs | **3.70 GiB** |
| **orphan driver logs** (NEW class, sec 5) | 33 495 logs | 3 262 = 3.31 GiB | **30 233 logs** | **27.39 GiB** |
| `~/tmp` sweep | 759 units | — | 759 units | **29.88 GiB** (30.40 nominal, 0.52 shared) |
| `~/tmp` pin dedup | 52 pin roots, 27.18 GiB | nothing deleted | — | **2.40 GiB** (→ 24.78) |

Directory-level subtotal **138.92 GiB**; with the orphan logs **166.31 GiB**; with the
`~/tmp` sweep and the pin dedup, **≈ 198.6 GiB**, which should take `/home/xqian` from
**511 G to roughly 710 G free**.

The pin dedup is small this round on purpose: round H's dedup already hardlinked the bulk
(53.95 → 27.18 GiB), so only the five pin roots created since — `d115-libsnap`,
`d119-libpin`, `d119r3-libpin` and the two `d109r{3,4}-libsnap` — still have private copies
of libraries another root already holds. 58 files, 2.40 GiB.

**Depth is directory-level, by the owner's choice** (asked with the three prior rounds'
shapes on the table): no file-class release inside kept arms (round D §14), no new
compression pass (round G). The 26 pdvd families compressed on 09-16 stay compressed, and
`restore_compress_20260916f.py --confirm <family>` is still mandatory before any name-based
reader touches them. The orphan logs in sec 5 are a separately-asked exception, and sec 5
says why they are not the thing that was declined.

### 1.1 sbnd_xin — the whole of the directory-level yield

The docs 116/117 ladders on the three round-3 MC samples, 3 dirs per family, 6–10 GiB each:

| family | doc | dirs | verdict |
|---|---|---:|---|
| `d116cs` `d116csp3bw` `d116p3bw` | 116 | 9 | RELEASE — graded rungs; doc 116 recorded **no flip** |
| `d117t1x` `d117t1e` `d117c1x` `d117c1e` | 117 | 12 | RELEASE — doc 117 §11's ADOPT table returned **HOLD** |
| `d117t1xrep` | 117 | 1 | RELEASE — the determinism repeat of a released arm (sec 3.2) |
| `d116tfull` | 116/118 | 3 | **KEEP — the current production output** |
| `d116s0rep` | 116/119 | 3 | **KEEP — the noise-floor control** |
| `d115` | 115 | 3 | **KEEP — stage A, the input layer** |
| `d115pr` | 115 | 3 | **KEEP — the published truth baseline** (owner's call) |
| `d118*` `d119*` | 118/119 | ~35 | KEEP — open, and ~3 GiB in total |

Why the seven released families *can* go: doc 118 §537 names `docs/115_*`, `docs/116_*`,
`docs/117_*` and **`products/d115|d116|d117`** as its untouched M13 record layer. Those
distilled products are 24 MB and already on disk. The record outlives the bytes — the
docs-104+ rule — and under that rule **citation does not keep an arm**, which is exactly
how a gate baseline gets swept if nobody looks. Two did nearly get swept: sec 3.2.

**Said out loud so the next round does not rediscover it:** `d117t1x` is quoted by name in
doc 118 (line 127, M8 −1.17 pt, p 0.016) inside the *current production flip's*
justification. Releasing it is still correct — that number is in the committed doc — but it
is the closest call in the release set.

### 1.2 pdvd / pdhd — housekeeping
Production has not moved since round H (`d116vflip`/`d116vr2`, `d116hflip`/`d116hr2`). The
release is the doc pdvd/116 ladder rungs `d116{v,h}r1`/`d116{v,h}r3` (r2 is production *and*
a hand-scan source, so it is kept twice over), the owner-released `mg18` arms, and one
`heappr_039252_8` heap-profiling dir.

## 2. The fork list, derived rather than remembered

Round H's closing lesson was that its command block forked `retire` and `sweep_tmp` but not
`archive_tmp_*.py`, which `sweep_tmp` calls at line 106 — the sweep stopped mid-round on a
missing file. So this round read the call sites, **including the callees'**, which is where
the missed file actually lived:

```
retire_${STAMP}.sh     -> archive_records_${STAMP}.py, plan_${STAMP}.py
sweep_tmp_${STAMP}.sh  -> archive_tmp_${STAMP}.py, remote_head_${STAMP}.txt,
                          tmp_census_${STAMP}.py, retire_${STAMP}.sh
plan_${STAMP}.py       -> toks_${STAMP}.{py,txt,json}, cit_${STAMP}.json,
                          scan_arms_${STAMP}.json, remote_head_${STAMP}.txt
tmp_census_${STAMP}.py -> tier1_*_${STAMP}.txt, tmp_tier/tmp_worktrees_${STAMP}
```

Seven files forked. `cit_20260916d.py`, `scan_arms_20260916d.py` and
`dedup_pins_20260912.py` take their stamp as an argument and are **not** forked — two
different sets, and conflating them is how round H lost two turns.

## 3. The six defects a plain `sed` fork would have carried

### 3.1 The sample grammar could not see the round-3 samples — the dangerous one

`plan_*.py`'s `SAMP` knew `mcp1k|mcp2k|ncpi0|nuecc48|dbg25a`. **78 of the 146 sbnd arms**
are `work-r3nue-* / work-r3cv-* / work-r3off-*` — doc 115's samples, and every docs 116–120
arm built on them. With `r3*` unknown, `arm_token("work-r3cv-d116tfull")` returns
`"r3cv-d116tfull"`, so:

- `production=["d116tfull"]` cannot match it,
- `open_prefix=("d119",)` cannot match it (the dir starts `work-`, the token starts `r3cv-`),
- `a in LIVE` cannot match it.

**Every docs 115–120 arm would have been unprotectable by name and fallen straight to
tier 1.** This is round A's *"the round grammar grew an `h`"* (plan_20260912.py header
item 2) repeating, and in the same failing-**open** direction that header called the
dangerous half. Fixed by adding `r3cv|r3nue|r3off`; proved by INTERLOCK 11 resolving every
production name to its 3 dirs and by `d116tfull` appearing as the arm token for all three
samples.

### 3.2 Value-first, run against every family rather than the one that looked odd

Under the docs-104+ rule citation does not keep an arm, so the only thing that separates a
gate baseline from a sweep rung is **what other arms use it as**. Run against all nine
`d116`/`d117` families, exactly two have call sites in `scripts/d118 scripts/d119
scripts/d120`, and both are inside a family whose other members are released:

- **`d116tfull` is the current production output.** Doc 118 (`Status: FLIPPED`, toolkit
  `675fd266`) line 5: the flip *is* "doc 116's `tfull`". Its decisive gate G1 compares every
  product of the flipped default **against `work-r3*-d116tfull`**, archive members by
  content — 4 call sites in `scripts/d118/hash_gate.py`, 3 in `stageB_flip.sh`, plus
  `d119/stageB_tail.sh` and `stageB_alloc.sh`.
- **`d116s0rep` is the noise-floor control.** Doc 116's cell table line 114: *"`s0rep` |
  none | production re-run: the noise floor"*, compiled-config **identical** to the no-TLA
  compile (`4ee005c935dac7df`, doc 116 §2). It is the `arm_before` of doc 119 round 0's own
  Repro line, `perf_rank.py nuecc work-r3nue-d116s0rep work-r3nue-d116tfull`.

Releasing `d116s0rep` while keeping `d118fliprep` — the same kind of arm — would have been
incoherent, and that incoherence is what surfaced it. The rule the two cases settle on:
**keep a control when you keep what it controls; release both when you release what it
controls.** That is why `d117t1xrep` *is* released: `d117t1x` goes with it, and the rep has
no call site in `scripts/` or `docs/*.md` at all.

### 3.3 `archive_records`' HEAVY list still did not treat `.zst` as heavy

Doc 111 §12 item 6 carried this forward from round G and it survived round H unapplied.
Every HEAVY pattern anchors on the *original* extension, so
`calib-evt123-group02.json.zst` — one of the 8569 files round G compressed — matched none
of them, read as record layer, and would have been carried bodily into the `.tar.zst`.
Fixed with `r'.*\.zst$'`.

### 3.4 `sweep_tmp` hard-codes its held prefixes

Lines 73 and 133 held `d111 d112 d113 d103 d108` — the *pdvd doc-113 peer's* scratch, held
by round D because that session was live. A plain fork carries a live-peer hold forward
forever. That peer is finished, docs pdvd/113–116 are all committed and pushed, and
`ListAgents` shows five peer sessions all offline or idle, so the list narrows to this
round's own open rounds, `d119 d120`. Each unit still has to clear the census's own rules
(age, pins, citation, PROTECTED) on its own merits.

**A fifth thing the first plan run caught, which was a config slip and not an inherited
defect.** `d119` was held on sbnd and released on pdvd/pdhd. Doc sbnd_xin/119 is **one
cross-detector round** whose rounds 3 and 5 were gated on PDHD and PDVD too and whose round
5 (the allocator flip, `ref/prod-2026-09-21c`) landed *today*. Holding it on one detector
and releasing it on the other two is incoherent; the hold costs 2.38 GiB on pdvd and 1.82 on
pdhd, 2.9 % of the round. Held on all three, and the planner re-run.

### 3.5 The held-prefix list exists TWICE, and the first fix hid the second

Narrowing `sweep_tmp`'s list (3.4) and re-running the census produced a `~/tmp` tier of
**456 units, 4.63 GiB** — against a forecast of ~40 GiB. `tmp_census_*.py` carries its
**own** copy of the same list at line 64, `LIVE_TOP = ("d111","d112","d113","d103","d108")`,
and it is the census that decides what is a unit at all. With only the sweep narrowed, every
`~/tmp/d113`, `d111`, `d103` unit stayed KEEP in the census, so the sweep had nothing to
refuse and the narrowed hold **looked like it had worked**. A fix applied to one of two
copies is indistinguishable from a fix that worked, unless the number is checked against a
forecast — which is the only reason this surfaced.

**And `~/tmp/mg18` was not in the census report at all.** The round-scratch loop maps a
top-level directory to a committed doc through
`RNUM = ^(?:doc|pr|d|p|h|q)(\d{1,3})`, and `mg18` matches none of those prefixes — "mg" is
not a round grammar — so the loop hit `continue` and the directory was **silently skipped**:
never classified, never in the tier, never printed. Its 17.5 GiB (17.4 of it four CMake
build trees, 9 `CMakeCache.txt`) looked exactly like something that had been judged and
kept. This is round D's lesson 2 (the grammar refused `pr149`) in a loop that fails *closed*
rather than open — safe, but silent, and silence is why it took a forecast to notice.

`EXTRA_ROUNDS` is the designed escape hatch for precisely this, and the owner released the
round by name on 2026-09-21, as the `mg10` precedent (doc 105 §13) requires. Both fixed;
census re-run.

### 3.6 Liveness, and why every hold this round is typed by hand

`remote_head_20260921.txt` = `f04d8937`, **verified equal to the remote by `ls-remote` over
https + the `gh` credential helper before it was written** — not taken from local
`origin/main`, which is stale (09-18, `6daaf25d`). Today is the worst case for
`live_tokens()`: docs 115–120 are *all* committed and pushed, so it reports **0 tokens from
0 files** and vouches for nothing. Every hold in sec 1 is therefore a hand-typed judgement,
which is stated here rather than left to look like a derivation.

## 4. The `~/tmp` sweep

**759 units, 30.40 GiB nominal, 29.88 GiB freed** (0.52 GiB is shared with a kept file and
frees nothing). Dry run PASS: no unit written in the last hour, no process with its cwd
inside one, and the three permanent pins intact before and after.

The first census run returned **456 units / 4.63 GiB**; the two defects in sec 3.5 account
for the other 25.25 GiB, and both runs are committed (`tmp_tier_20260921.firstrun.txt`,
`tmp_census_20260921.firstrun.out`) so the claim has its evidence rather than just its
number. The largest single unit is `~/tmp/mg18` at 17.5 GiB — one unit, the whole directory,
visible only after the `EXTRA_ROUNDS` entry.

**One registered git worktree, and this round is the first to exercise that branch.**
`~/tmp/d103/wcp_wt_r3` (554 MB) was invisible to round H because `d103` was a held prefix,
so every previous sweep reported `registered worktrees: 0` — a zero that meant "not looked
at", not "none". It is **not** in the rm tier. It goes through `git worktree remove`, gated
on two conditions checked at confirm time: the tree is clean (`status --porcelain` empty —
verified) and its HEAD is an ancestor of the pinned remote head (`3320311e` ⊂ `f04d8937` —
verified, `merge-base --is-ancestor` rc=0). If either fails the sweep skips it and says so.
It is never `rm -rf`-ed, because that would leave a stale registration behind in the parent
repo.

The order gate is real and fired: a bare `./sweep_tmp_20260921.sh` exits **4** —
*"22 sbnd work dir(s) from tier1_sbnd_20260921.txt are still on disk — run
retire_20260921.sh first"*. The dry run above was taken with `REQUIRE_WORK=no`, which skips
**only** that gate, to get dry-run evidence for the worktree branch before the owner
confirms anything.

## 5. The class nine rounds could not see

`run_pr_evt.sh` writes its per-event driver log as
`<tree>/work/.batch_pr_<run6>_<evt>[_<arm>].log` — **beside** the event directory, not inside
it. So nine rounds of directory-level retirement never touched one: the planner enumerates
directories and `arm_token()` returns `None` for a file. The leading dot also hides them
from `du dir/*`, which is why no round's survey reported them either.

Measured: **33 495 logs, 30.70 GiB**, of which **27.39 GiB name an event directory that no
longer exists** — debris from rounds A–H (`d25r12eager`, `d28r2fb`, `h26q2dprod`, `h28prod`,
`h100a/b`, …). Owner ruling, asked with the measurement in hand: *"Yes, orphans only."*
The 3.31 GiB whose event directory survives stays — that is a kept arm's debug trail.

**Why this is not the file-level release the owner declined.** Round D §14 removed
`frames.tar.bz2` / `icluster.npz` from *inside kept arms*, and the cost was those arms'
re-runnability — PDVD imaging is no longer re-runnable from disk. Nothing of that kind is
possible here: an orphan log's run was removed rounds ago, so there is no product left to
make un-re-runnable. The re-runnability cost is exactly zero, which is why it was worth
asking as a separate question rather than folding into "depth".

New machinery, same shape as the round-D file-level pass:
`plan_orphanlogs_20260921.py` (8 gates), `archive_orphanlogs_20260921.py`,
`retire_orphanlogs_20260921.sh`. Gates O1–O4 filter the candidate set (event dir absent;
arm not in **this round's own `keep_*` list**, read from the directory planner rather than
re-derived; not PROTECTED; not written in the last hour). O5–O8 are the refusals: no
release log's event dir exists, every target resolves inside `work/`, malformed names are
excluded rather than guessed at, and — **O8** — the keep list actually loaded, because a
silent empty keep list would make O2 fire on nothing and look identical to a pass. It
reports 49 kept arm tags on pdvd and 41 on pdhd.

**The record is a manifest, not a tar, and that is the round's one honest compromise.**
`.log` is not in `archive_records`' HEAVY list, so a naive freeze would have compressed
27 GiB of log text into the record layer and freed a fraction of what this claims. Round D
§14 set the precedent for heavy content: path, size, mtime and SHA-256 per file, no tar. So
the record says exactly which logs existed, how big each was and what it hashed to — **not
what they said**. If a released log's contents ever matter, this round cannot give them
back.

## 6. The command block (owner runs this from bash mode: type `!` first, then paste)

The permission classifier refused `CONFIRM=yes` bulk deletes five times in round D and four
times in round H, including after explicit chat consent, so this block is written for bash
mode from the start rather than after a refusal.

```bash
IMG=/home/xqian/toolkit-dev/wcp-porting-img; D=$IMG/pdhd/scripts/retire; cd $D

# 1. the record layer is ALREADY FROZEN (this session: 747/747 arms + 2 orphan-log
#    manifests).  Do NOT re-run the freezes -- they refuse an existing manifest by design
#    (M13) and exit non-zero, which would read as a hard failure on the first paste.
#    Verify instead; the drivers' own record gates (rc=6 / rc=14) are what enforce it.
ls -d $IMG/sbnd/sbnd_xin/archive/records/cleanup-20260921/{sbnd,pdvd,pdhd}-tier1 >/dev/null \
  && echo "record layer present"
wc -l $IMG/sbnd/sbnd_xin/archive/records/cleanup-20260921/orphanlogs/*.manifest.tsv

# 2. the orphan driver logs (sec 5).  THIS MUST PRECEDE THE DIRECTORY RELEASE -- see below.
#    Dry run already PASSED 8/8 this session, and both record-gate negative controls fired.
CONFIRM=yes ./retire_orphanlogs_20260921.sh

# 3. the directory release.  The driver re-runs the planner as its own gate first.
CONFIRM=yes ./retire_20260921.sh 1 sbnd
CONFIRM=yes ./retire_20260921.sh 1 pdvd
CONFIRM=yes ./retire_20260921.sh 1 pdhd

# 4. ~/tmp sweep -- order-gated on step 2, and BEFORE the dedup (doc 114 sec 1)
CONFIRM=yes ./sweep_tmp_20260921.sh
#    if it exits 11 (the unit list moved -- an age-KEEP aged into FREE between census and
#    confirm, round H's first refusal), re-census under CENSUS_SUFFIX so the plan-time
#    record is never overwritten, then re-run:
#      CENSUS_SUFFIX=sweep python3 tmp_census_20260921.py
#      CENSUS_SUFFIX=sweep CONFIRM=yes ./sweep_tmp_20260921.sh

# 5. the non-destructive win: hardlink identical pin libraries, nothing deleted
CONFIRM=yes python3 dedup_pins_20260912.py

# 6. verify -- the sentinel line must still read 21 PASS / 0 FAIL (sec 0's baseline)
df -h /home/xqian | tail -1
python3 $IMG/sbnd/sbnd_xin/scripts/analysis/pr149/sentinels_tolerant.py --arms 'work-*-pr150s0' | tail -1
find $IMG/pdhd/work $IMG/pdvd/work $IMG/sbnd/sbnd_xin -xtype l 2>/dev/null | wc -l   # expect 0
python3 $IMG/sbnd/sbnd_xin/scripts/cfg/prod_cfg_gate.py --ref $IMG/sbnd/sbnd_xin/ref/prod-2026-09-21d; echo rc=$?
```

**Why the orphan logs go first.** Step 3 removes 725 event directories, and every one of
them has a `.batch_pr_*` log beside it that is *currently* held by gate O1 ("the event dir
still exists"). Run the directory release first and those logs become orphans, so
`retire_orphanlogs`' confirm-time re-plan produces a **larger** list than the frozen one,
`cmp` fails, and the driver exits **11** — correctly refusing a list that moved, but for a
reason the round itself caused. Running it first keeps the frozen list and the live list
identical. The logs newly orphaned by step 3 stay for round J, which sec 9 item 2 already
expects.

Every dry run in this block already passed this session, so a disagreement at confirm time
means something moved — the drivers re-plan before they delete precisely so a stale tier
file cannot be executed. If step 3's re-plan disagrees with sec 1's counts, stop and
re-read the plan.

## 7. As executed

*(to be filled once the block has run.)*

## 8. Verification targets

- sentinels **21 PASS / 0 FAIL / 2 OPEN / 7 INERT** at all three checkpoints (before the
  round, after the work trees, after the `~/tmp` sweep).
- broken symlinks **0** in all three trees, against the recorded pre-count of **0**
  (INTERLOCK 4, and gate 16 of the orphan-log driver).
- the three permanent pins intact: `~/tmp/d102/libpin_d102`, `~/tmp/d102m-libsnap`,
  `~/tmp/pdhdstm_libpin`.
- `prod_cfg_gate.py --ref ref/prod-2026-09-21d` still **PASS, 26 artifacts** — against the
  §0 baseline taken before the round, not on its own.
- `/home/xqian` **511 G → ≈ 710 G free** (138.92 directories + 27.39 orphan logs + 29.88
  `~/tmp` + 2.40 dedup ≈ 198.6 GiB, set-relative).

## 9. Carried forward to round J

1. **`work-r3nue-d119flip-torn`** (~1 GiB) — a torn-write arm, held this round only because
   `open_prefix` holds all of `d119`. Releasing it needed an exception to the prefix rule,
   which is not worth 1 GiB. Round J's first release.
2. **The 3.31 GiB of non-orphan driver logs** become orphans as their arms retire. The
   machinery now exists; re-run `plan_orphanlogs_*.py` each round.
3. **All three `PROTECTED.txt` files are stale** — last touched by round F (`70802bd7`).
   They still call `d103vflip`/`d108hflip`/`d102mpr` "THE LATEST PRODUCTION", and rounds H
   and I's releases were never moved to RETIRED. `prot_hit()` only ever *adds* to KEEP, so
   the cost is over-keeping, never mis-deleting — but it is a prerequisite for ever
   releasing `d102mpr` (10.26 GiB) and `d103vprod1` (1.25 GiB), whose cooling-off round is
   long over.
4. **`pdhd/l1sp_wf_v9`, 10.7 GiB** — still no regeneration path recorded.
5. **`pdhd/work/028084_26_d09/gpu_mem_028084_26_aALL.csv`, 384 MB, written 2026-09-21
   10:48** — live profiling output, and the reason that one d09 arm is twice the size of its
   90 siblings. `d09` is substrate, so it is kept; the CSV is not reconstruction product and
   is a file-class question for a later round.
6. `archive_records`' HEAVY list is now `.zst`-aware (sec 3.3). Keep it that way when
   forking.
