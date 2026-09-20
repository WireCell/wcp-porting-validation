# 114 — cleanup round H (2026-09-19): retire the intermediate arms of the closed doc-113 / pr-150 / 114-116 rounds

**Status: FULLY EXECUTED by the owner 2026-09-19.** Free space on `/home/xqian` went **478 G → 733 G (+255 G)**.
Sentinels are **21 PASS / 0 FAIL / 2 OPEN / 7 INERT** at all three checkpoints (before the round, after the work
trees, after the `~/tmp` sweep), and every tree finished with **0 broken symlinks** against a recorded pre-count
of 0. Tree sizes: sbnd_xin 255 → **76 G**, pdvd 108 → **96 G**, pdhd 91 → **82 G**, `~/tmp` 140 → **86 G**.
Numbers as executed are in sec 7; sec 1 is the plan they were measured against. The three permanent pins
(`d102/libpin_d102`, `d102m-libsnap`, `pdhdstm_libpin`) were verified intact after the sweep.

The session planned and gated the round but could not run it: the permission classifier refused four times
(`CONFIRM=yes` dedup, the compound fork+archive, `archive_records` alone, and the driver forks), consistent with
round D's record of five. The owner ran the sec-6 block from bash mode. `archive_records_20260919.py` was
*created* successfully here; only its execution was refused.

Owner instruction, verbatim (2026-09-19): *"Can you do a round of clean up for pdhd, pdvd, sbnd_xin, and ~/tmp
directory? What we want to keep is the latest production, and we can remove the intermediate results and retire
them to save some space. We have done several rounds of this kind of clean up in the past. Can you proceed?"*

## 0. Repro

```bash
D=/home/xqian/toolkit-dev/wcp-porting-img/pdhd/scripts/retire; cd $D
git -C /home/xqian/toolkit-dev/wcp-porting-img rev-parse main > remote_head_20260919.txt   # == remote, verified
python3 toks_20260919.py                                   # 9540 dirs -> 9962 tokens
python3 cit_20260916d.py toks_20260919.txt cit_20260919.json   # 9962 -> 784 cited, 76684 hits
python3 scan_arms_20260916d.py --json=scan_arms_20260919.json  # hand-scan sources, measured not typed
python3 plan_20260919.py                                   # 39/39 interlocks PASS, tier1_<tree>_20260919.txt
python3 tmp_census_20260919.py                             # 449 units -> tmp_tier_20260919.txt
python3 dedup_pins_20260912.py                             # report only: 35.13 GiB recoverable
# the round-start sentinel baseline (doc 91: run it BEFORE touching anything, never only after)
python3 $IMG/sbnd/sbnd_xin/scripts/analysis/pr149/sentinels_tolerant.py --arms 'work-*-pr150s0'
```

**Round-start sentinel baseline (2026-09-19, against production `work-*-pr150s0`): 21 PASS, 0 FAIL, 2 OPEN,
7 INERT, 0 SKIP** — the same tally round D recorded, so the record is intact going in. Log:
`/home/xqian/tmp/cleanup20260919/sentinels_before.log`. The block re-runs it after; a before/after pair is the
only form in which the number means anything (a post-only run cannot tell "this round broke it" from "it was
already broken").

## 1. What this round releases

| tree | universe | KEEP | RELEASE | freed (set-relative) |
|---|---|---|---|---|
| sbnd_xin | 214 dirs | 68 = 54.76 GiB | 146 dirs | **179.72 GiB** |
| pdvd | 6674 dirs | 5464 = 70.01 GiB | 1210 dirs | **12.21 GiB** |
| pdhd | 2652 dirs | 2032 = 53.66 GiB | 620 dirs | **9.37 GiB** |
| `~/tmp` sweep | — | — | 449 units | **28.92 GiB** (1.36 shared, frees nothing) |
| `~/tmp` pin dedup | 51 pin roots, 53.95 GiB | — | nothing deleted | **35.13 GiB** (→ 18.82 GiB after) |

**Total ≈ 265 GiB, minus a small overlap.** `/home/xqian` is the constraining filesystem for all four targets
(they share `/dev/nvme2n1p6`); it stands at 478 G available and should reach roughly 730–740 G.

*Why the last two rows are not simply additive, and why the block runs the sweep before the dedup.* The dedup
figure was measured with all 51 pin roots present, and 8 units in the sweep tier are pin roots —
`~/tmp/d109r4-libsnap` (8.2 G) and `~/tmp/d109r3-libsnap` (3.5 G). None of the dedup's large wins live there
(`d115/libpin_d115`, `d09_libpin/pin`, `d101/libpin*`, `d08_libpin/new2`, `d102/libpin_d102`,
`d109-libsnap/new2` are all KEEP), so running the sweep first frees those 11.7 GiB outright and the dedup then
recovers somewhat less than 35.13 GiB against what survives. Either order is *safe* — hardlinks are
reference-counted, so deleting one link never damages another — but this order avoids double-counting.

The sbnd release is dominated by this session's own doc-113 counterfactual arms — `d113cfqp2v4` 39.11 GiB in a
single dir, `d113cfqpg64` 18.58, `d113cfqpg64dg` 9.74, and five ~4.4 GiB threshold/deghost arms — plus the closed,
**not adopted** doc pr/150 study arms (`pr150cs` 10.57, `pr150tfull` 10.51, `pr150p3bw` 10.44, `pr150r1..r4`
2.74 each) and the doc-109 r3/r4 round arms. pdvd/pdhd release the doc-114/115 sweep arms that the doc-116 flip
superseded. Doc 113 sec 9 already states the imaging census is regenerable from `d102m` in ~30 min; that path is
untouched because `d102m` is production and kept.

## 2. Production, re-derived from primary sources (not inherited)

The 09-16 PROTECTED files predate the 2026-09-18 flip, so every production name was re-read this round.

* **PDHD / PDVD — production MOVED 2026-09-18.** doc pdvd/116 sec 10 is `Status: FLIPPED (owner 2026-09-18)`.
  Shipped **values**: `prefer3` + `tree+path` α 0.5, and the R2 tagger point `stm_proton_muon_guard` true,
  `michel_min_kink_deg` 20, `michel_max_len_cm` 30. The arms carrying those values with no TLA are
  **`d116hflip` / `d116vflip`** (sec 10.0 step (d)); **`d116hr2` / `d116vr2`** are the graded R2 cells they
  reproduce, and `own116h` / `own116v` were scanned on them, so `scan_arms` holds them independently.
  `d108hflip` and `d103vflip` become **substrate** (they are the `SRC` those flip arms were built on, 88 and 138
  citations). This is `feedback_audit_value_first_not_name_adjacency` applied: the flip section names the value,
  the arm table names the arm, and they are different sentences.
* **SBND — stage B production moved to `pr150s0`.** ref/prod-2026-09-17b; the OFF arm of doc pr/150 *is* today's
  production, and 269 `vertex_labels` resolve into it. Stage A stays `work-<s>-d102m` (doc 109 line 512).
* **`d102mpr` is KEPT although superseded** (10.26 GiB). `feedback_gate_source_arm_retired`: a superseded *output*
  arm can still be the *input* a published gate resolves into, and an arm built from different Q/L is never a
  substitute. It carries 175 citations, 56 of them in scripts. Menu item for the owner, not this round's call.
* **`pr150csp3bw` is KEPT although it looks like a sweep point** (10.52 GiB): 225 `vertex_labels` resolve into it,
  so it is a hand-scan source. The citation census is what caught this; a name-adjacency read would have released it.

**Completeness, not just resolution.** INTERLOCK 11 proves a production name *resolves*; it does not prove the arm
is whole. Checked against the inventory's per-arm dir counts, since production now depends on these:

| arm | dirs | expected | |
|---|---|---|---|
| `d116vflip` / `d116vr2` / `d103vflip` | 120 / 120 / 120 | 120 (doc 116 sec 10.0) | OK |
| `d116hflip` / `d116hr2` / `d108hflip` / `d102hcs` | 61 / 61 / 61 / 61 | 61 | OK |
| `d102m` / `pr150s0` / `d102mpr` / `pr150csp3bw` | 4 / 4 / 4 / 4 | 4 samples | OK |
| `d103vprod1` | 1 | **1 by design** | OK |

`d103vprod1` is a single dir and that is correct, not short: doc 111 records it as *"the one event re-run through
the applied tree with no overrides."* Stated here because a future round reading only the dir count would
otherwise flag it.

## 3. What the interlocks held back

39 PASS / 0 FAIL after one correction. The first run FAILed INTERLOCK 14 on `mg18vbase`/`mg18vfix` and
`mg18hbase`/`mg18hfix` — the 2026-09-18 master-merge validation round (toolkit `b93673ef` / `d2646110`). It is
undocumented, so no doc number maps to it. The mg10 precedent (doc 105 sec 13) is that the **owner** releases a
merge-validation round by name, so all four are held (~2.5 GiB). Say the word and they go next round.

Also held: 27 pdvd and 22 pdhd hand-scan source arms (INTERLOCK 13, derived by `scan_arms`, never typed), the
s144 sentinel witness layer, the four `pr134-f086` arms em_display still resolves into, `p101q` (OPEN, doc
pdvd/100 sec 8.7), and `~/tmp/d113` (8.8 GiB) which is held by prefix because the pdvd doc-113 peer session
shares that scratch directory with this round's sbnd work.

## 4. The record gate, and one thing that must happen first

INTERLOCK 15 requires the record to outlive the bytes, and `archive_records_20260919.py 1` freezes a hash
manifest for every target before anything is deleted. That step is **step 2** of the block below and is not optional.

**One gap the machinery cannot see.** `live_tokens()` reads the *content* of uncommitted records, not their
*filenames*. Four untracked files name `d113adopt` only in their names:

```
sbnd/sbnd_xin/docs/pr/150_figs/metrics/d113adopt-{mcp1k,mcp2k,ncpi0,nuecc48}.tsv
```

So INTERLOCK 16 reported 0 live tokens and `d113adopt` (10.30 GiB) sits in the release with its record still
uncommitted. Step 1 of the block commits those four files. Do not skip it — it is the only reason this arm is
safe to release.

## 5. Not done this round

* `~/tmp/d113` (8.8 GiB) — shared scratch with the pdvd peer's doc-113 round; held by prefix.
* `mg18*` in both detector trees — needs the owner to name it (sec 3).
* `d102mpr` (10.26 GiB) and `d103vprod1` (1.25 GiB) — superseded production, kept one round as a cooling-off.
* File-level compression inside kept arms (round G's `.zst` pass) — not re-run; the 26 cold pdvd arms
  compressed on 09-16 are still compressed and still need `restore_compress_20260916f.py` before any
  name-based reader touches them.

## 6. The command block (owner runs this from bash mode: type `!` first, then paste)

```bash
IMG=/home/xqian/toolkit-dev/wcp-porting-img; D=$IMG/pdhd/scripts/retire; cd $D

# 0. sentinel baseline -- ALREADY RUN this session (21 PASS / 0 FAIL / 2 OPEN / 7 INERT).
#    Re-run only if time has passed since; step 7 compares against it.
python3 $IMG/sbnd/sbnd_xin/scripts/analysis/pr149/sentinels_tolerant.py --arms 'work-*-pr150s0' | tail -1

# 1. commit the d113adopt record BEFORE its bytes go (sec 4)
git -C $IMG add -f sbnd/sbnd_xin/docs/pr/150_figs/metrics/d113adopt-*.tsv
git -C $IMG commit -m "sbnd_xin/pr/150 figs: the d113adopt arm's per-sample metrics (record for cleanup round H)"

# 2. create this round's two drivers (the classifier refused these file writes)
sed 's/20260916/20260919/g' retire_20260916.sh    > retire_20260919.sh    && chmod +x retire_20260919.sh
sed 's/20260916/20260919/g' sweep_tmp_20260916.sh > sweep_tmp_20260919.sh && chmod +x sweep_tmp_20260919.sh

# 3. freeze the record layer (M13 / INTERLOCK 15) -- NOT optional, and must precede step 4
python3 archive_records_20260919.py 1

# 4. dry run, then delete.  The driver re-runs the planner as its own gate first.
./retire_20260919.sh 1                       # dry run, all three trees
CONFIRM=yes ./retire_20260919.sh 1 sbnd
CONFIRM=yes ./retire_20260919.sh 1 pdvd
CONFIRM=yes ./retire_20260919.sh 1 pdhd

# 5. ~/tmp sweep (order-gated on the trees above).  BEFORE the dedup -- see sec 1.
./sweep_tmp_20260919.sh                      # dry run
CONFIRM=yes ./sweep_tmp_20260919.sh

# 6. the non-destructive win: hardlink identical pin libraries, nothing deleted.
#    Expect somewhat under 35.13 GiB because step 5 already removed two pin roots.
python3 dedup_pins_20260912.py               # re-measure after the sweep
CONFIRM=yes python3 dedup_pins_20260912.py

# 7. verify -- the sentinel line must still read 21 PASS / 0 FAIL (sec 0's baseline)
df -h /home/xqian | tail -1
python3 $IMG/sbnd/sbnd_xin/scripts/analysis/pr149/sentinels_tolerant.py --arms 'work-*-pr150s0' | tail -1
find $IMG/pdhd/work $IMG/pdvd/work $IMG/sbnd/sbnd_xin -xtype l 2>/dev/null | wc -l   # broken symlinks, expect 0
```

If step 4's dry run disagrees with sec 1's counts, stop and re-read the plan — the driver re-plans before it
deletes precisely so a stale tier file cannot be executed.

## 7. As executed (2026-09-19, owner, bash mode)

| step | result |
|---|---|
| record freeze | manifests for **146/146** sbnd, **1210/1210** pdvd, **620/620** pdhd targets under `sbnd_xin/archive/records/cleanup-20260919/<tree>-tier1` — SHA-256 per file, a `.links.txt`, and a `.tar.zst` of the non-heavy record layer |
| sbnd release | 146 dirs, **179.83 GiB**, rc=0 |
| pdvd release | 1210 dirs, **12.21 GiB**, rc=0 |
| pdhd release | 620 dirs, **9.37 GiB**, rc=0 |
| broken symlinks | pdhd 0, pdvd 0, sbnd_xin 0 (pre-count 0 — the number only means something against that) |
| pin dedup | 51 roots, **53.95 → 27.18 GiB**, 26.77 GiB recovered; 9785 files linked, 653 made read-only; VERIFY 15462 files present, **0 missing** |
| `~/tmp` sweep | refused **twice** before succeeding (see below); finally 453/453 units frozen and removed, rc=0, **26.7 GiB** |
| permanent pins | `d102/libpin_d102`, `d102m-libsnap`, `pdhdstm_libpin` — intact, checked by the sweep itself |
| sentinels | 21 PASS / 0 FAIL / 2 OPEN / 7 INERT at all three checkpoints, identical |

Tree sizes: sbnd_xin 255 → **76 G**, pdvd 108 → **96 G**, pdhd 91 → **82 G**, `~/tmp` 140 → **86 G**.
`/home/xqian` free: **478 G → 733 G (+255 G)**.

**The sweep refused itself, and it was right to.** `sweep_tmp` compares the unit list at confirm time against the
census and stops if it moved:

```
REFUSING: the ~/tmp unit list moved between census and confirm -- a live writer, or a
work release that has not happened yet.  diff:
> /home/xqian/tmp/pr150/scores-d113adopt-{mcp1k,mcp2k,ncpi0,nuecc48}.tsv
```

Those four files were KEEP at census time on the age rule (*"written in the last 2 h"* — I had written them
during the doc-113 eval). By confirm time they had aged past the window and became FREE, so the list grew by
exactly four. This is round D's lesson 6 firing as designed: a unit kept only on AGE is re-judged later, and the
driver refuses a stale list rather than acting on it. Nothing in `~/tmp` was deleted.

**Resolution (re-census done, `CENSUS_SUFFIX` so the plan-time record is never overwritten):**
`tmp_tier_20260919.sweep.txt`, **453 units, 26.73 GiB** (down from 28.92 because the dedup now shares inodes
with some of it, and shared links free less). The diff against the plan-time tier is exactly those four files
and nothing else.

**Then the sweep refused a second time, on a defect in sec 6 of this doc.** The dry run passed (453 units, none
written in the last hour, no process cwd inside, the three permanent pins intact), but `CONFIRM=yes` stopped at:

```
REFUSING: the freeze failed (rc=2; archive_tmp_20260919.log) -- nothing removed
python3: can't open file '.../archive_tmp_20260919.py': [Errno 2] No such file or directory
```

The block's step 2 forked `retire_20260916.sh` and `sweep_tmp_20260916.sh` but **not**
`archive_tmp_20260916.py`, which `sweep_tmp` calls at line 106 to freeze each unit before removing it. The
record gate did exactly its job — it refused rather than remove unfrozen units. Fixed by
`sed 's/20260916/20260919/g' archive_tmp_20260916.py > archive_tmp_20260919.py`; every other
`*_20260919` dependency of the sweep was then verified present (`tmp_tier_20260919.sweep.txt` 453 lines,
`tmp_worktrees_20260919.sweep.txt`, `remote_head_20260919.txt`). Remaining command:

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/pdhd/scripts/retire
CENSUS_SUFFIX=sweep CONFIRM=yes ./sweep_tmp_20260919.sh
```

*Lesson for the next round's command block: derive step 2's fork list from the drivers' own call sites
(`grep -oE '[a-z_]+_\$\{STAMP\}\.(py|sh)'`), never from memory of which files a round "usually" needs.*

## 8. Files

Machinery (all in `pdhd/scripts/retire/`): `toks_20260919.py`, `cit_20260919.json`, `scan_arms_20260919.json`,
`plan_20260919.py`, `archive_records_20260919.py`, `tmp_census_20260919.py`, `remote_head_20260919.txt`,
tier files `tier1_{sbnd,pdvd,pdhd}_20260919.txt` and `tmp_tier_20260919.txt`, logs `plan_20260919.out`,
`tmp_census_20260919.out`, `scan_arms_20260919.out`, `cit_20260919.out`.
Inventory used for the byte table: `/home/xqian/tmp/cleanup20260919/{inventory_raw.tsv,arms.json}`.
