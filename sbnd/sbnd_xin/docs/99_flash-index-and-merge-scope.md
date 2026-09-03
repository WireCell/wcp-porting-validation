# doc 99 — two defects from the doc-92 epoch gate: an out-of-range flash read, and a batch-scoped merge

Both were found by the doc-92 part-2 epoch gate (`f9676556`) and reported there
without a fix. This round fixes both, and — while measuring the first — finds a
third, larger problem underneath it that is **reported, not fixed**, because it
cannot be corrected on the read side.

| | |
|---|---|
| toolkit commit | `dc0cc9af` on `apply-pointcloud` (parent `e88f364d`) |
| wcp-porting-img | this doc + `scripts/d99_*`, `scripts/analysis/d99_*`, `run_pr_chain_batch.sh` |
| gate arms | `work-{ncpi0,nuecc48,mcp1k}-d99fix` / `-d99fixpr` (308 events, both stages) |
| baseline arms | `work-*-d92gate` / `-d92gatepr` — same manifest, same commit, pre-fix |
| binary pin | `~/tmp/d99-libsnap`, `libWireCellClus.so` 2026-09-02 19:32:54 |
| status | **NOT bit-identical, by construction** — see §4. Everything else byte-identical. |

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# the population measurement (both stages; --stage pr is the one that predicts T_cluster)
python3 scripts/analysis/d99_flash_index_census.py --arm 'work-{s}-d97fv' \
    --samples nuecc48,ncpi0,mcp1k,mcp2k --out /home/xqian/tmp/d99-flash-census.tsv --jobs 8
python3 scripts/analysis/d99_flash_index_census.py --arm 'work-{s}-d97fvpr2' --stage pr \
    --samples nuecc48,ncpi0,mcp1k,mcp2k --out /home/xqian/tmp/d99-census-prod-pr.tsv \
    --detail /home/xqian/tmp/d99-detail-prod-pr.tsv --jobs 8

# the arms (14 min at 8/8 jobs) AND all six gate legs, one command.
# Existing arms are skipped (M13), so re-running it just re-gates.
./scripts/d99_flash_gate.sh

# ... and as the post-master-merge gate, against any baseline:
REFARM=d99fix NEWARM=<fresh-arm> ./scripts/d99_flash_gate.sh

# unit tests
cd /nfs/data/1/xqian/toolkit-dev/toolkit && ./build/clus/wcdoctest-clus
```

The per-event summaries those censuses produce are committed, so every number
below is checkable without a 20-minute re-run:

| file | rows |
|---|---|
| `docs/99_flash/d99-census-ql-prod3067.tsv` | 3067 — Q/L stage, production |
| `docs/99_flash/d99-census-pr-prod3067.tsv` | 3067 — PR stage, production (what `T_cluster` holds) |
| `docs/99_flash/d99-census-pr-gate308.tsv` | 308 — PR stage, the gate arm |

The per-cluster `--detail` files are not committed (85k rows); regenerate them
with the commands above.

---

# 1. Defect A — `Grouping::flash_at()` read past the end of the flash point cloud

## Symptom

The doc-92 epoch gate compared two arms that were byte-identical in every
archive and every other ROOT branch, and found 48 disagreeing cells on 15 of 308
events, all of them in `tracking-pr.root:T_cluster`'s `flash_id`,
`flash_time_us`, `flash_pe`. The values on **both** sides were raw memory:
denormal doubles around `1e-310` (pointer bits reinterpreted as `double`) and
integers like 21845 = `0x5555`. 88 of 20 175 rows carried `flash_time == 0` and
`flash_pe == 0` while `flash_id != -1`.

## Root cause

`Facade::Grouping::flash_at()` marked the flash valid on the mere existence of
the PC:

```cpp
if (! this->has_pc("flash")) return flash;
flash.m_valid = true;
flash.m_time = this->get_element<double>("flash", "time", flash_index, 0);
```

`get_element()` (`Facade_Mixins.h:127`) ends in

```cpp
return arr->template element<T>(index);
```

and `Array::element()` (`PointCloudArray.h:270`) is an **unchecked pointer
offset** — `*(reinterpret_cast<const T*>(m_bytes.data()) + index)`. Its `def`
argument guards a missing PC and a missing array, never a missing *row*.

The index it is handed is a *stored* one: `Cluster::get_flash()` passes the
cluster's `"flash"` scalar, written by QLMatching. When that scalar does not
address a row of this grouping's flash PC, all four singular reads run off the
end.

`Facade_Grouping.h` had already documented the correct behaviour —

> Returns an invalid Flash (`operator bool() == false`) if there is no "flash" PC
> **or the index is out of range**.

— so this was an implementation that had drifted from its own declared contract,
not an undecided design question.

## Why it hid

Three reasons, and the third is the one that matters:

1. `operator bool()` returned **true**, so every caller's `if (flash)` test
   passed and the reads looked legitimate.
2. `Facade_Flash.h` carried the caveat *"A 'true' does not guarantee all values
   are valid"*, which reads as a note about the per-OpDet vectors and had the
   effect of normalising the behaviour.
3. **The garbage usually looks plausible.** Of the 158 rows this fix changes on
   the gate manifest, many read `flash_id = 0, flash_time_us = 0, flash_pe = 0`
   — indistinguishable from a real flash 0 at t = 0. Only the denormal doubles
   give it away, and only at full print precision: at 4 decimal places every one
   of them prints as `0.0000`.

## Fix (`dc0cc9af`)

```cpp
const auto ftime = this->get_pcarray<double>("time", "flash");
if ((size_t) flash_index >= ftime.size()) {
    return flash;           // stale/foreign index: addresses no flash
}
flash.m_valid = true;
```

Two deliberate choices:

- **The bound is on the `"time"` array**, which is exactly what `flashes()` uses
  to size its enumeration loop. Every `flash_at()` call made from `flashes()`
  therefore passes by construction, so the enumeration path — and with it
  `QLMatching.cxx:1232`, which iterates `grouping->flashes()` — provably cannot
  move. Bounding on `"ident"`, or on a minimum across the arrays, would not have
  that property. The stage-A gate below confirms it empirically.
- **The singular values are read off their spans, not through `get_element()`**,
  so a PC that omits a companion array yields the documented default instead of
  a garbage read. Validity stays keyed on `"time"` alone, so a missing `"ident"`
  cannot turn a real flash into an invalid one.

A merely *short* companion array turns out to be unreachable —
`Dataset::add()` refuses an array whose major axis disagrees with the ones
already present — which the new test found by failing. That is recorded in the
test so the next reader does not re-derive it.

## Verification

**Unit tests** — `clus/test/doctest_flash_index_bounds.cxx`, 6 cases / 59
assertions, including a causal negative control: build a 4-row flash PC, resolve
index 3 successfully, then rebuild the PC with 3 rows and watch the *same* index
stop resolving with nothing else changed. Disabling the guard fails **3 of the 6
cases and 5 assertions**; the positive controls (in-range reads, enumeration
size, missing-array defaults) still pass, so the tests are specific to the
defect and not to the code shape. Full suite: `wcdoctest-clus` **255/255**.

**Containment, 308-event manifest** (`ref/prod-2026-09-04/gate308-*.txt`), both
stages re-run, baseline `work-*-d92gate{,pr}`:

| leg | tool | result |
|---|---|---|
| stage A, Q/L member content | `d97_ql_gate.py` | **PASS** 308/308 events, 1232 products |
| stage B, archive members | `pr85_hash_gate.py` | **PASS** 616/616 archives (38 + 96 + 482), 0 unpaired |
| stage B, every ROOT branch | `d99_root_branch_census.py` | **PASS** — 1938 tree instances, **236 280 branch instances**, exactly **3** differing pairs |
| per-event `nusel-evt<ID>.tsv` | `cmp` | **PASS** 308/308 |
| per-event `calib-pr-evt<ID>.json` | JSON compare, `*_ms` timers stripped | **PASS** 177/177 |

The three differing pairs are `T_cluster:flash_id` (66 events),
`T_cluster:flash_pe` (55), `T_cluster:flash_time_us` (28) — and nothing else.

`scripts/pr87_root_tree_diff.py` is not sufficient for this claim and was not
used for it: it `break`s at the first differing branch of a tree and truncates
its detail list, which is right for "did anything move?" and useless for
"nothing moved except these three". Hence the exhaustive census script.

All six legs are wired into `d99_flash_gate.sh` itself rather than left in this
prose, because the expected-diff column list is the whole subtlety of the gate
and an instruction that only lives in a doc has to be found and retyped. Wiring
them up immediately earned its keep: it exposed that the driver's new `BASE`
parameter silently kept the `$PWD` the script already had in that name, and —
worse — that two of the legs reported **PASS having compared zero events**. Both
tools now refuse an empty comparison, and a bogus arm name is the negative
control:

```
$ d99_root_branch_census.py bogusarmpr d99fixpr --samples ncpi0 --expect T_cluster:flash_id
REFUSE: 1 missing arm(s), 0 events compared -- nothing was tested
VERDICT: FAIL -- nothing was compared            (rc=1)
```

The wired gate's own output, end to end:

```
--- stage A, Q/L member content            TOTAL events 308, identical 308   rc=0
--- stage B archives, ncpi0/nuecc48/mcp1k  38 / 96 / 482 byte-identical       rc=0
--- every ROOT branch                      PASS, only the expected columns    rc=0
--- predict which rows may move            308 evts, 20156 matched, 158 OOB   rc=0
--- the moved rows are exactly those       158 moved, 0 violations, 0 misses  rc=0
=== D99 FLASH GATE VERDICT: PASS
```

**Causality** — `d99_flash_ab.py`, joining on cluster id per event against a
census computed from the archives' cluster scalars with no ROOT file opened:

```
TOTAL  events=308  T_cluster rows=20175  differing rows=158  events with a diff=66
       sentinel-violations=0   census-mismatches=0
VERDICT: PASS
```

Every row that moved is a cluster the census independently flagged out-of-range,
and every one of them now reads the documented `(-1, 0.0, 0.0)` sentinel.

**A trap worth recording.** The first run of this check failed on 24 events, all
of them "moved but not predicted". The predictor was the *Q/L-stage* census, and
`T_cluster` is written from the **PR-stage** grouping — the PR chain re-clusters,
so on mcp1k evt 59685 the Q/L archive holds 10 clusters and the PR archive holds
22, renumbered 1…22. A one-directional mismatch like that is the signature of a
predictor blind to part of the population, not of a broken fix. Re-run against
`--stage pr`, the match is exact. (The PR pctree does carry the optical PCs; an
earlier grep for `flash` in the *member names* found nothing and I briefly
concluded it did not.)

---

# 2. The residual this does NOT fix — SBND keeps only one APA's flash list

Reported, not fixed. **CLAUDE.md §5.1**: correcting it changes archive content
unconditionally, and the choice of remedy is the owner's.

## What is wrong

`cfg/pgrapher/experiment/sbnd/qlmatching.jsonnet` runs the pipeline **per APA**:
`TensorFileSource → FlashTensorToOpticalPCs → QLMatching`. Each
`FlashTensorToOpticalPCs` ends with

```cpp
lpcs["flash"] = std::move(flash_ds);     // ASSIGN, not append
```

on the same live root node, and its `ident` is the row index *within that APA's*
flash matrix (`fident.push_back((int) r)`). `QLMatching` stores that per-APA row
id on the cluster (`QLMatching.cxx:3722`, `flash->get_flash_id()`).

So the archive ends up with **one** canonical `"flash"` PC — the last APA's —
while cluster scalars written during an earlier APA's matching index a list that
is no longer there. Depending on the number, `flash_at()` then either runs off
the end (defect A, now returning the sentinel) or lands on a **different, real
flash** of the surviving APA's list.

**The matching itself is not affected.** Each APA matched against its own flash
list while that list was live. What is broken is only the *read-back after the
fact* — which is what `T_cluster` does.

## How big it is

`cluster_t0` is the exact discriminator: QLMatching sets it from the flash the
cluster actually matched, and it survives in the archive even when that flash's
PC row does not. So `flash.time() == cluster->get_cluster_t0()` is an
archive-local test for "did I resolve the right row", and it is **bit-exact** —
verified on mcp1k evt 59335, where all 6 correctly-resolved clusters compare
equal and all 5 mis-resolved ones differ.

Production, all 3067 events, one row per matched cluster:

| stage | matched rows | CORRECT | WRONG (a different real flash) | OOB (was undefined) | events touched |
|---|---:|---:|---:|---:|---:|
| Q/L (`work-*-d97fv`) | 50 699 | 25 917 (51.1%) | 24 024 (47.4%) | 758 (1.5%) | 3066 / 3067 |
| PR (`work-*-d97fvpr2`) — this is what `T_cluster` holds | 171 664 | 86 820 (50.6%) | **83 496 (48.6%)** | 1 348 (0.8%) | 3066 / 3067 |

Worked example, mcp1k evt 59335 — 9-row flash PC, 13 matched clusters, scalars
`[5 2 13 6 2 1 11 4 2 0 4 2 5]`; APA-0 clusters resolve correctly, APA-1
clusters do not:

| cid | scalar | matched_flash_gid | cluster_t0 | `flash_at()` resolved | verdict |
|---:|---:|---:|---:|---:|---|
| 1 | 5 | 4 (apa0) | −75257.3320 | −75257.3320 | CORRECT |
| 3 | 13 | 1000012 (apa1) | 1150824.8801 | — | OOB |
| 4 | 6 | 1000006 (apa1) | 816638.9579 | −1161516.3548 | **WRONG** |
| 8 | 4 | 1000004 (apa1) | −75242.9171 | 1513.0167 | **WRONG** |
| 10 | 0 | 0 (apa0) | 212278.8006 | 212278.8006 | CORRECT |

## Options, for the owner to choose between

1. **Fix the archive** — give `FlashTensorToOpticalPCs` a per-APA PC name, or
   append with an APA column, and make the cluster scalar address the merged
   list. Correct at the source and makes `get_flash()` mean what it says.
   Changes Q/L archive bytes, so every downstream hash moves: a full
   revalidation.
2. **Fix the consumer** — have the `T_cluster` writers cross-check
   `flash.time() == cluster_t0` and emit the sentinel when it fails. One
   comparison, no archive change, but it silences the wrong rows rather than
   correcting them, and it changes what a shipped diagnostic column means.
3. **Leave it and document** — the columns are diagnostic only; `cluster_t0`,
   already a `T_cluster` column, carries the right time today.

Nothing in SBND reconstruction reads `Cluster::get_flash()`: the only
reconstruction consumer is `RetileCluster::mutate()`, and SBND's config
instantiates `ImproveCluster_2`, never `RetileCluster` (checked against the
compiled `ref/prod-2026-09-04/prod_prjob.json`). uBooNE is unaffected by
construction — `UbooneClusterSource` writes true row indices into a single
optical tree.

## A note for the next gate

Production `work-*-d97fvpr2` was written by the pre-fix binary, so its
`T_cluster` flash columns still hold the undefined values. **Any future A/B of a
fresh arm against production will show those three columns differing on ~1348
rows across ~3066 events, and that is expected, not a regression.** Pass them to
`d99_root_branch_census.py --expect`, or exclude them.

---

# 3. Defect B — `run_pr_chain_batch.sh` merged the batch, not the arm

## Symptom

`work-nuecc48-d97fvpr2/nusel-events.tsv` held **2 rows instead of 49**, and
`nusel-table.tsv` **10 instead of 547**, in the shipped production reference arm.
All 48 per-event `pr_evt*/nusel-evt*.tsv` were intact.

## Root cause

The merge iterated this invocation's event list:

```bash
for evt in "${EVENT_IDS[@]}"; do
    _t="$OUTROOT/pr_evt${evt}/nusel-evt${evt}.tsv"; ...
done
python3 "$SX/nusel_extract.py" --merge "${_tsvs[@]}" --out "$OUTROOT/nusel-table.tsv" ...
```

but wrote to arm-scoped filenames. A one-event re-run (evt 256587, 2026-09-02
11:32) therefore rewrote the arm-wide merge with that one event.

## Why it hid

Nothing failed and nothing warned: the runner exited 0, the per-event files were
untouched, and **no published number reads the merged tables** — `pr_scores_table.py`
and every doc-92 figure read the per-event files. The only consumer is doc 92's
own T1 check, which is why the doc-92 refresh found it and nothing earlier did.

## Fix

The merge now enumerates `pr_evt*/` under `$OUTROOT`, with the same
`ls | sed | sort -n` idiom the script already uses to discover `EVENT_IDS`, so
row order is unchanged for a full run. It is now idempotent and monotone: a
partial re-run can only refresh rows, never drop them. The log line names both
counts, so it says out loud when the merge is wider than the batch:

```
merged 48 per-event tables in the arm (this batch: 1) -> .../nusel-table.tsv + nusel-events.tsv
```

## Repair of the damaged arm — **owed, not applied**

The damage is in an existing label, so under **CLAUDE.md §5.2** it is opt-in.
`scripts/d99_repair_nuecc48_merge.sh` is written, dry-run clean, and applies only
with `CONFIRM=yes`:

```
interlock 1: 48 per-event tables, all non-empty  OK
interlock 2: re-merge gives 49 event rows (want 49) and 547 table rows (want 547)
interlock 3: every row of the current truncated files is reproduced verbatim  OK
```

Interlock 3 is the one that matters: it proves the regeneration reproduces what
survived, so the repair restores rows rather than replacing them with something
new. The truncated originals are preserved under
`work-nuecc48-d97fvpr2/merge-truncated-20260902/`. To apply:

```bash
CONFIRM=yes ./scripts/d99_repair_nuecc48_merge.sh
```

---

# 4. Status flags

- **Defect A: NOT bit-identical, and cannot be made so.** The cells it changes
  held values read past the end of an array — they were never reproducible, and
  the doc-92 gate caught two otherwise byte-identical arms disagreeing on 48 of
  them. There is no meaningful "byte-identical when off", and a knob whose OFF
  branch preserves an out-of-bounds read would not be worth having. Everything
  outside those three columns **is** byte-identical, proven over 236 280 branch
  instances plus every archive and every stage-A product.
- **Defect B: no output changes.** Only the arm-wide merge of already-written
  per-event tables.
- **The §2 residual: unchanged by this round**, by design. 48.6% of production's
  `T_cluster` flash rows still name a real flash that is not the one the cluster
  matched.
- Config: untouched. No jsonnet changed, so `prod_cfg_gate.py` is unaffected.
- The gate arms were produced by `~/tmp/d99-libsnap` (19:32:54), pinned **before**
  a concurrent session's in-progress `TaggerCheckNeutrino` edits landed in this
  shared tree at 19:39. Those edits are not in either arm and not in `dc0cc9af`.

# 5. Reported, not fixed (pre-existing, out of scope here)

- **`get_element()` is the underlying hazard.** `Facade_Mixins.h:127` backs every
  `get_scalar()` on every facade and performs an unchecked `element<T>(index)`.
  Hardening it is a separate, separately-gated change — it would touch the
  uBooNE path and needs `qlport/scripts/ab_check.sh`. One trap for whoever does
  it: `element<T>(index)` is bounded by `bytes().size() / sizeof(T)`, **not** by
  `size_major()`, and those differ whenever the element type read differs from
  the stored one — so a `size_major()` guard would not be guarding the pointer
  arithmetic it appears to guard.
- **The flashlight join in `flash_at()`** indexes `l_idents[light_index]` with a
  value taken from the data. It is consistent by construction today (the same
  component writes both PCs), so no guard was added rather than adding an
  untested branch.
