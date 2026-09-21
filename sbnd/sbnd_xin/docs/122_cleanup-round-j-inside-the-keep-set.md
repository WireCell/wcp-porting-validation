# 122 — cleanup round J (2026-09-21b): inside the keep set

**Status: PLANNED AND GATED; the destructive steps are the owner's to run** (sec 7). Round I
spent the directory-level lever — after it, every arm still on disk is kept for a named
reason — so the owner asked what else could go. This round answers that with measurement
rather than instinct, and four of the five things it found were **not** what the first pass
of arithmetic suggested.

Owner, 2026-09-21, after round I closed: *"The SBND_xin directories are still quite big. I
wonder if there are things that can be further retired? Also same question for ./pdhd and
./pdvd work directories?"* Then, from a measured menu with each lever's cost stated: all four
levers, and `pdhd/l1sp_wf_v9` **kept** — *"keep it, it's irreplaceable"*.

Going in: `/home/xqian` **710 G free**; sbnd_xin 159 G, pdvd/work 70.5, pdhd/work 57.4.

## 0. Repro

```bash
IMG=/home/xqian/toolkit-dev/wcp-porting-img; D=$IMG/pdhd/scripts/retire; cd $D

# the class census that framed the round -- where the bytes actually are, by file class
# (sbnd: pctree 76.29 GiB, bee .zip 15.08, frames 14.42, calib 14.14, logs 9.21)

# file-level: pctrees + SP frames, 7 gates, negative control
python3 plan_files_20260921b.py                 # F1-F7 PASS, 10800 files, 38.82 GiB
python3 plan_files_20260921b.py --negctl        # F3 must FAIL -> rc=1
python3 plan_files_20260921b.py --hash          # the freeze

# directory-level: the two superseded arms
python3 toks_20260921b.py && python3 cit_20260916d.py toks_20260921b.txt cit_20260921b.json
python3 scan_arms_20260916d.py --json=scan_arms_20260921b.json
python3 plan_20260921b.py                       # 39/39 interlocks PASS, 5 dirs, 10.27 GiB
python3 archive_records_20260921b.py 1          # 5/5 arms frozen

# compression: round G's scope, extended to pdhd
python3 plan_compress_20260921b.py              # C1-C6 PASS, 8689 files, est save 13.00 GiB
python3 plan_compress_20260921b.py --negctl     # C5 must FAIL -> rc=1
python3 plan_compress_20260921b.py --hash       # the freeze
```

**Baselines, taken before anything is deleted** (round I's rule, and it caught nothing this
time only because it was run): sentinels **21 PASS / 0 FAIL / 2 OPEN / 7 INERT**,
`prod_cfg_gate --ref prod-2026-09-21d` **PASS 26**, broken symlinks **0/0/0**.

## 1. What this round releases

| lever | scope | freed |
|---|---|---|
| sbnd pctrees, `d115pr` + `d116s0rep` | 10 036 files | **21.27 GiB** |
| sbnd SP frames, `d115` + `d102m` | 636 files | **10.30 GiB** |
| pdhd SP frames, unshared only | 128 files | **7.24 GiB** |
| directory: `d102mpr` + `d103vprod1` | 5 dirs | **10.27 GiB** |
| compression, pdvd + pdhd cold arms | 8 689 files, 15.03 GiB in scope | **≈13.00 GiB** (reversible) |
| **total** | | **≈ 62.1 GiB** → `/home/xqian` ≈ **772 G free** |

### 1.1 Four corrections to the menu the owner chose from

The numbers I offered were built from a class census; building the actual delete lists moved
four of them, every one in the safe direction. They are listed because a menu number that
quietly becomes a different number is how a round loses its audit trail.

| lever | offered | actual | why |
|---|--:|--:|---|
| pctrees | 21.3 | **21.27** | — |
| SP frames | 29.6 | **17.54** | three separate reasons, below |
| superseded arms | 11.5 | **10.27** | `d103vprod1` is **13 MB**, not the 1.25 GiB round H's PROTECTED line claimed |
| compression | ~11 | **≈13.00** | my first scope was narrower than round G's; corrected to match the precedent |

The SP frames figure lost 12 GiB to three things, and the first two are near-misses:

1. **4.46 GiB was `input_files_reco1`.** My class census tested `'frames' in filename`, which
   matches `..._frameshift.root` — the **reco1 production inputs**. They were never SP output.
2. **`input_files_reco1` also holds its own `frames-dnn.tar.bz2` files** (~4.1 GiB). A glob for
   that exact name across `sbnd_xin` *would* have deleted them.
3. **1.08 GiB is `protodunehd-sp-frames-raw-anode*.tar.bz2`** — raw ADC, input-like rather than
   SP output, so excluded on the conservative reading.

Both near-misses are now **gates**, not notes: F3 requires every target to sit under an
allow-listed arm, and **F7** requires every file matching a set's pattern but living outside
that allow-list to be a *declared* exclusion with a written reason. F7 is the one that would
have caught both on its own.

### 1.2 What was deliberately NOT released, and why

- **`d116tfull`'s pctrees** — they are what doc 118's gate G1 compares the flipped production
  default against (4 call sites in `scripts/d118/hash_gate.py`). Releasing them would delete a
  published gate's reference. This is why the set is named arm by arm instead of globbed by
  family: `d116tfull` sits in the middle of the `d116` family whose other members went in
  round I.
- **`d115`'s pctrees** — the stage-A products stage B consumes. Releasing them would make the
  whole round-3 chain non-re-runnable (M11). Its *frames* go; its pctrees stay. The chain is
  `reco1 → SP → frames-dnn → imaging → pctree → stage B`, and this round removes exactly one
  link, the one that is bit-deterministically rebuildable.
- **120 pdhd frame archives, 6.77 GiB — DEFERRED, needs its own owner call.** Gate F2 caught
  157 symlinks, in kept substrate arms (`029107_*_d09ctl`, `_stm0`, …), resolving onto them.
  Those arms hold the frames *by reference* precisely so they are not duplicated, so releasing
  a shared target degrades several kept arms at once and leaves dangling links inside
  PROTECTED directories. The unshared 128 release cleanly.
- **sbnd compression, 2.33 GiB** — out of scope for a reason that is about the undo, not the
  bytes: `restore_compress`'s family filter keys on `/<run6>_<idx>_<arm>/`, which sbnd's
  `work-<sample>-<arm>` directories do not match, so an sbnd family could only be restored
  with `--all`. A pass whose undo is coarser than its do is not worth 2.33 GiB.
- **`pctree` compression** — measured at **1.2–1.3×** (already gzipped). 17 GiB of restore
  burden for ~3 GiB is a bad trade, so pctrees are released where they are closed and left
  alone everywhere else, never compressed.
- **`pdhd/l1sp_wf_v9`, 10.23 GiB** — owner ruling, kept. Now carries a PROTECTED line saying
  regeneration is unknown, so no future round re-litigates it.

## 2. "Regenerable" is a claim about someone else's disk — so it is a gate

The frames release rests on SP being bit-deterministic (doc pdvd/99 G2) and on the upstream
inputs still existing. Those inputs are **not ours**:

| set | rebuilt from | owner |
|---|---|---|
| sbnd frames | `xin-round3-samples` → `/nfs/data/1/yuhw/2025-fall-prod-sample/` | yuhw |
| sbnd frames | `input_files_reco1` (4.6 G, 20 files) | ours |
| pdhd frames | `input_data_{7p8_new,14_old}_coh_grouping` → `/nfs/data/1/xning/wirecell-working/data/` | xning |

All four resolve today — checked, not assumed. That is **gate F6**, re-run on every plan and
every confirm, rather than a sentence in this doc: if yuhw or xning cleans up, these bytes stop
being regenerable and the lever stops being safe. A future round gets a FAIL, not a surprise.

## 3. Two bugs the gates found, both "a list that looks complete and isn't"

**3.1 A fixed-depth glob matched zero files in two of six arms.** The first
`plan_files_20260921b.py` used `{arm}/f*/pr_evt*/pctree-pr-evt*.tar.gz`. The `r3cv` and `r3nue`
arms group events into `f000..f224` sub-roots; the **off-beam arms put `pr_evt<N>/` directly at
the arm root**. So the glob found 2017+2001 files in four arms and **0** in the other two —
**4.56 GiB of 21.27 silently absent**, and the list looked finished. Fixed by enumerating with
`os.walk` + a filename predicate, which has no depth assumption. The count now equals `find`
exactly.

**3.2 The compression scope was narrower than the precedent, by 12.7 GiB.** My first version
held out production ∪ substrate ∪ scan-sources ∪ keep_arms ∪ PROTECTED, and returned **10 files
/ 0.28 GiB** — which looked like "the lever is exhausted". Round G's scope was *cold arms
including* the hand-scan sources and substrate. The distinction that makes that correct:
**PROTECTED protects an arm's existence, not its file encoding.** A compressed arm is still
there, still complete, and restorable byte-identical. Corrected to round G's scope: **8 689
files, est. 13.00 GiB**. What still guards it is C4 (the whole arm cold 7+ days, so no live
reader), C2 (no symlink resolves onto a target), and the restore script.

Both are the same failure in different clothes, and both were caught by a number disagreeing
with a forecast rather than by reading the code. Round I found two of these; that makes four in
two rounds, all of the form *a selection that silently under- or over-matches*.

## 4. Why the compression lever stays small on purpose

**243 scripts in this repo read calib dumps or pctrees by name; 3 tolerate `.zst`.** Round G's
lesson is that most of them glob-loop and skip a missing file **silently** — no error, just
absent data. So the ~60 GiB of the same classes sitting in warm and production arms is left
uncompressed deliberately, and only the cold tail is taken.

**RESTORE BEFORE ANY NAME-BASED READER TOUCHES A COMPRESSED FAMILY:**

```bash
python3 pdhd/scripts/retire/restore_compress_20260921b.py --confirm <family ...>   # or --all
```

That script was hardened this round. Round G's fork defaulted to a **single pdvd manifest**;
round J writes two (pdvd + pdhd), so a bare invocation for a pdhd family would have selected
0 rows and printed *"0 manifest rows selected"* — indistinguishable from "that family is not
compressed". It now reads **every** generation's manifest (including round G's), and an unknown
family exits **20** while printing the families that *do* appear. The undo must not have the
same silent-skip failure the pass itself warns about.

## 5. Gate results

| planner | gates | result |
|---|---|---|
| `plan_files_20260921b.py` | F1–F7 | **PASS**, 10 800 files, 38.82 GiB |
| ” `--negctl` | F3 (+F5) | **FAIL as designed**, rc=1 — an injected `input_files_reco1` path |
| `plan_20260921b.py` | 13 × 3 interlocks | **39/39 PASS**, 5 dirs, 10.27 GiB |
| `plan_compress_20260921b.py` | C1–C6 | **PASS**, 8 689 files, est. 13.00 GiB |
| ” `--negctl` | C5 (+C4) | **FAIL as designed**, rc=1 — an injected `d116vflip` production file |
| `retire_files_20260921b.sh` (dry) | rc 3/4/5/6/14 | dry run clean |
| `retire_20260921b.sh 1` (dry) | — | 4+1 present, 0 already gone, symlinks 0/0/0 |

Every gate that matters has a **causal negative control**, because a gate that has never fired
is indistinguishable from one that cannot.

## 6. PROTECTED changes

- `d102mpr`'s line moved to RETIRED (sbnd) — two rounds of cooling-off, owner released it by
  name, INTERLOCK 9 re-checked that no live manifest resolves into it.
- `d103vprod1` **split off** `d103vflip`'s shared line before being retired. Commenting the
  shared line out whole would have silently unprotected `d103vflip`, which is live substrate —
  the same class of accident as everything in sec 3.
- `l1sp_wf_v9` gained a line recording the owner's ruling and that its regeneration path is
  unknown.

## 7. The command block (owner runs this from bash mode: type `!` first, then paste)

```bash
IMG=/home/xqian/toolkit-dev/wcp-porting-img; D=$IMG/pdhd/scripts/retire; cd $D

# 1. records are ALREADY FROZEN this session.  Do NOT re-run the freezes -- they refuse an
#    existing manifest (M13) and exit non-zero.  Verify instead.
ls -d $IMG/sbnd/sbnd_xin/archive/records/cleanup-20260921b >/dev/null && echo "record layer present"
wc -l $IMG/sbnd/sbnd_xin/archive/records/cleanup-20260921b/*.manifest.tsv

# 2. the file-level release (pctrees + SP frames).  38.82 GiB.
CONFIRM=yes ./retire_files_20260921b.sh

# 3. the two superseded arms.  10.27 GiB.
CONFIRM=yes ./retire_20260921b.sh 1 sbnd
CONFIRM=yes ./retire_20260921b.sh 1 pdvd

# 4. compression -- REVERSIBLE, originals removed only after the decompressed SHA-256 matches.
CONFIRM=yes ./compress_files_20260921b.sh

# 5. verify against sec 0's baselines
df -h /home/xqian | tail -1
python3 $IMG/sbnd/sbnd_xin/scripts/analysis/pr149/sentinels_tolerant.py --arms 'work-*-pr150s0' | tail -1
find $IMG/pdhd/work $IMG/pdvd/work $IMG/sbnd/sbnd_xin -xtype l 2>/dev/null | wc -l   # expect 0
python3 $IMG/sbnd/sbnd_xin/scripts/cfg/prod_cfg_gate.py --ref $IMG/sbnd/sbnd_xin/ref/prod-2026-09-21d; echo rc=$?
```

Steps 2–4 are independent: no set's list is affected by another step, and each driver re-plans
and `cmp`s before it acts, so a list that moved refuses (rc=11) instead of executing stale.

## 8. As executed

*(to be filled once the block has run.)*

## 9. Carried forward to round K

1. **The 120 shared pdhd frame archives, 6.77 GiB** — needs a ruling on whether the kept arms
   that reference them may lose them together with their 157 links.
2. **`bee`/`mabc` zips: sbnd 15.08 GiB, pdvd 8.54.** Regenerable from pctrees — so for any arm
   whose pctrees this round released, the zips are the *only* remaining display artefact and
   are no longer cheap to rebuild. That inversion should be decided before the zips are offered.
3. **`.wct-*.json` compiled configs, 6.89 GiB in 28 848 files (sbnd).** ~270 KB each,
   byte-identical within an arm apart from the event id. `archive_records` already keeps one per
   arm in the record layer; on disk every copy is still there. Compressible ~6×, or one-per-arm.
4. **`work-r3nue-d119flip-torn`** — still held by round I's `d119` prefix.
5. `restore_compress` still cannot filter sbnd families by name (sec 1.2). Fix the regex before
   any sbnd compression pass.
