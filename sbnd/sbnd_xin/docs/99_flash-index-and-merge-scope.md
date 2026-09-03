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

# the arms AND all six gate legs, one command.
# Existing arms are skipped (M13), so re-running it just re-gates.
./scripts/d99_flash_gate.sh

# ROUND 3 (sec 10): the FLIP gate -- production with both knobs on in the
# jsonnet defaults and NO TLA hatch, byte-gated against round 2's arms.
./scripts/d99r3_flip_gate.sh                 # ncpi0, ~3 min; SAMPLES=... to widen
python3 scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-05   # 21/21, exit 0
# what the flip moves INSIDE an archive, by datapath (member names are tensor
# indices and renumber, so a name diff answers nothing):
python3 scripts/analysis/d99r3_pctree_datapath_diff.py \
    work-ncpi0-d99r2off work-ncpi0-d99r2wr              # stage A
python3 scripts/analysis/d99r3_pctree_datapath_diff.py \
    work-ncpi0-d99r2offpr work-ncpi0-d99r2bothpr --stage pr

# ROUND 2 (sec 6-9): the read fix and the write fix.  Six arms, every leg.
# Same skip-if-exists rule, so a re-run re-gates without re-running anything.
./scripts/d99r2_flash_gate.sh

# ... and the round-2 instrument on its own, against any PR arm.  It needs no
# baseline: T_cluster carries cluster_t0_us and flash_time_us on the same row.
python3 scripts/analysis/d99_tcluster_flash_check.py <armpr> --samples ncpi0,nuecc48,mcp1k

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
| `docs/99_flash/d99r2-tcluster-gate308.tsv` | 308 — **round 2**, per event, all three arms side by side (off / read-on / write-on) |

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

Reported, not fixed **in round 1**. Both sides of it are fixed in round 2 (§6,
§7), each behind a knob that defaults OFF, because correcting the write side
changes archive content unconditionally and flipping it is the owner's call
(CLAUDE.md §5.1).

## What is wrong

> **Correction, 2026-09-02 (round 2).** The first version of this section named
> the wrong site. It said each per-APA `FlashTensorToOpticalPCs` clobbers a
> shared live root with `lpcs["flash"] = std::move(flash_ds)` and that the
> **last** APA's list survives. That is not what happens: every
> `FlashTensorToOpticalPCs` deserializes its **own** pctree from its own input
> tensors (`as_pctree`), so there is nothing shared to clobber — and SBND
> production runs `joint=true`
> (`wct-clus-matching-perevt.jsonnet:91`), so there is no `PointTreeMerging`
> either. The loss is at the **merge**, and the survivor is the **first**
> input's list. The corrected mechanism is below; every measurement in this
> section stands (it was taken from the archives, not from the reading), and the
> worked example — APA0 correct, APA1 wrong — is in fact the evidence for
> *first*, not last.

SBND production runs one `FlashTensorToOpticalPCs` per APA, each on its own
pctree, feeding **one joint `QLMatching`** node (one input port per APA), which
matches each APA in its own isolated run and then merges the per-APA trees into
a single output (`QLMatching.cxx:1109-1116`, reproducing the standalone
`PointTreeMerging` it replaces).

That merge is where the flash list is lost:

```cpp
// QLMatching.cxx merge_pct() -- primary target = runs.front() = input 0
for (const auto& src_pc : src->value.local_pcs()) {
    if (root_pcs_to_merge.find(name) == root_pcs_to_merge.end()
        && !is_per_anode_root_pc(name)) {
        continue;                 // <-- flash / light / flashlight DROPPED
    }
    ...
}
```

`root_pcs_to_merge` is `['opflash']` for SBND, so the merged root keeps **input
0's** canonical `"flash"` PC and drops every other input's. Meanwhile the
per-cluster `"flash"` scalar is the row index *within that input's* flash list
(`QLMatching.cxx:3722`, `flash->get_flash_id()`, which is the flash PC row id —
`fident.push_back((int) r)` in `FlashTensorToOpticalPCs`).

So a cluster matched on any input but the first indexes a list that is no longer
in the archive. Depending on the number, `flash_at()` then either runs off the
end (defect A, now returning the sentinel) or lands on a **different, real
flash** of the surviving input's list.

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

## Options, and what round 2 did with them

1. **Fix the archive** — carry every input's optical PCs through the merge and
   make the cluster scalar address the merged list. Correct at the source and
   makes `get_flash()` mean what it says. Changes Q/L archive bytes, so every
   downstream hash moves. **DONE in §7**, as `QLMatching.merge_flash_pcs`,
   default OFF. (The round-1 sketch here proposed changing
   `FlashTensorToOpticalPCs`; with the mechanism corrected above, that component
   is not the defect site and is untouched.)
2. **Fix the consumer** — round 1 sketched a `flash.time() == cluster_t0`
   cross-check that emits the sentinel on failure. Rejected on the owner's ask
   to fix the read: it silences the wrong rows rather than correcting them.
   Round 2 does the *correcting* version instead — resolve by
   `matched_flash_gid` against the merge-safe `"opflash"` PC. **DONE in §6**, as
   `SbndPrMagnifyTrackingVisitor.flash_by_gid`, default OFF.
3. **Leave it and document** — superseded.

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

## Repair of the damaged arm — **APPLIED 2026-09-02 20:22, on owner authorization**

The damage was in an existing label, so under **CLAUDE.md §5.2** the repair was
opt-in: `scripts/d99_repair_nuecc48_merge.sh` writes nothing without
`CONFIRM=yes`, and the owner ran it.

```
interlock 1: 48 per-event tables, all non-empty  OK
interlock 2: re-merge gives 49 event rows (want 49) and 547 table rows (want 547)
interlock 3: every row of the current truncated files is reproduced verbatim  OK
repaired: 49 event rows, 547 table rows
```

Interlock 3 is the one that matters: it proves the regeneration reproduces what
survived, so the repair restored rows rather than replacing them with something
new. The truncated originals are preserved verbatim under
`work-nuecc48-d97fvpr2/merge-truncated-20260902/` (2 and 10 rows), with a README
recording the repair.

**Verified independently of the script**, and this is the check the defect
originally broke — doc 92's T1, `wc -l nusel-events.tsv == N+1`, now passes on
every production arm:

| arm | events | `nusel-events.tsv` | want | | `nusel-table.tsv` |
|---|---:|---:|---:|---|---:|
| nuecc48 | 48 | 49 | 49 | **PASS** | 547 |
| ncpi0 | 19 | 20 | 20 | **PASS** | 224 |
| mcp1k | 1000 | 1001 | 1001 | **PASS** | 11432 |
| mcp2k | 2000 | 2001 | 2001 | **PASS** | 22641 |

No per-event file was touched, and no published number moves — the score tables
and every doc-92 figure read the per-event files, which were always intact.

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

---

# ROUND 2 — both sides of the §2 residual, each behind a default-OFF knob

Owner, 2026-09-02: *"Can you follow your suggestion, fix the read first, and
then fix the write? For the write, I assume that you can follow your suggest to
fix the archive? Please validate, update the md file, commit and push."*

| | |
|---|---|
| toolkit commits | `<read>` (read fix) and `<write>` (write fix), on `apply-pointcloud` |
| wcp-porting-img | this doc + `scripts/d99r2_flash_gate.sh`, `scripts/analysis/d99_tcluster_flash_check.py`, the `QL_EXTRA_TLA` hatch in `run_ql_evt.sh` |
| knobs | `SbndPrMagnifyTrackingVisitor.flash_by_gid` (read) and `QLMatching.merge_flash_pcs` (write), **both C++ default false** |
| arms | `work-{ncpi0,nuecc48,mcp1k}-d99r2{off,wr}` (stage A) and `-d99r2{off,rd,wr,both}pr` (stage B), 308 events |
| binary pin | `~/tmp/d99r2-libsnap` |
| status | **Nothing is flipped.** Production is unchanged; §9 states what each flip would cost. |

# 6. The READ fix — resolve by gid, not by row index

## Root cause, restated

QLMatching stamps two keys on every matched cluster, and only one of them
survives the merge:

| scalar | meaning | survives the merge? |
|---|---|---|
| `flash` | row index in **that input's** flash PC | **no** — the merge keeps only input 0's flash PC (§2) |
| `matched_flash_gid` | `gid_side * 1000000 + index into that input's flash list` | **yes** — and the `opflash` PC carrying the same gid IS in `root_pcs_to_merge` |

`Cluster::get_flash()` reads the first. The fix adds a reader for the second.

## Fix

- **`Grouping::flash_by_gid(int gid)`** (`clus/src/Facade_Grouping.cxx`) — finds
  the rows of the merge-safe `"opflash"` PC (one row per (flash, channel):
  `gid`/`time`/`ch`/`pe`) carrying that gid and builds a `Facade::Flash`.
- **`Cluster::get_matched_flash()`** (`clus/src/Facade_Cluster.cxx`) — delegates
  with the cluster's own `matched_flash_gid`.
- **Knob `flash_by_gid`** on `SbndPrMagnifyTrackingVisitor`, C++ default
  **false**; when on, `T_cluster`'s `flash_id`/`flash_time_us`/`flash_pe` come
  from `get_matched_flash()`. Present on the PDVD fork too, but **no PDVD config
  wires it** — see the precondition below.

Three details that are decisions, not incidentals:

1. **PE is summed in ascending CHANNEL order, not row order.** The canonical
   flash PC's `value` is `FlashTensorToOpticalPCs`' `sum += pe` over channels
   `0..nchan-1`. Summing the opflash rows in channel order reproduces it
   **bit-exactly**; summing in row order does not. The evidence is §8 leg **D**,
   which is as strong as this claim can get: `d99r2wrpr` reports `value()` read
   straight off the merged flash PC's `value` array while `d99r2bothpr` reports
   the channel-order sum of `opflash` rows, and the two agree by **exact**
   float equality on all 20 156 rows — an independently written array, not a
   tolerance. A unit test pins the ordering itself with values whose two
   summation orders give genuinely different doubles, and asserts they differ
   before testing anything — otherwise the case would prove nothing.
2. **`ident()` becomes the GID.** So `T_cluster`'s `flash_id` changes *meaning*
   under the knob: it is the globally-unique gid, joinable to the `opflash` PC,
   not a per-input row id. §2 option 2 listed "changes what a shipped diagnostic
   column means" as a cost of any consumer-side fix; this one pays it too, and
   in exchange the column becomes unambiguous for the first time.
3. **A gid naming two flashes is refused.** For SBND `gid_side` is the input's
   **anode ident** (`opflash_phys_gid` false, `shared_flash` false), so gids are
   unique across inputs by construction. Under `opflash_phys_gid` the gid side is
   the flash's *physical drift side*, and two inputs holding different flash
   lists can then emit the same gid. `flash_by_gid` detects exactly that (one
   channel appearing twice under one gid) and returns an **invalid** Flash rather
   than a silently doubled PE sum. **That is the precondition for enabling this
   knob anywhere else**, and it is why the PDVD visitor carries the knob but no
   PDVD config sets it: PDVD's gid encoding has not been checked.

# 7. The WRITE fix — carry every input's flash PCs through the merge

`QLMatching.merge_flash_pcs`, C++ default **false**. When true, the multi-input
merge adds `flash`/`light`/`flashlight`/`flashcov` to the concatenated name set,
and each non-primary input is re-based first (`QLMatching::shift_flash_indices()`)
by the row counts already on the merged root:

| shifted | by | why |
|---|---|---|
| `flash.ident` | flash rows | the schema is `ident(=row)`, and the row changes |
| `flashlight.flash` | flash rows | positional join column |
| `flashlight.light` | light rows | positional join column |
| `flashcov.flash` | flash rows | positional (`channel` is a channel id — left alone) |
| each cluster's `flash` scalar | flash rows | this is what `get_flash()` resolves; `-1` stays `-1` |

`write_opflash_pc` runs per input **before** the merge, so `opflash` and its gids
are untouched by the shift. That is deliberate: it leaves the two resolutions
independent, which is what makes §8's cross-check meaningful.

OFF is inert **by construction**, not by measurement: the name set handed to
`merge_pct` is then literally `m_root_pcs_to_merge`, and `shift_flash_indices()`
is never called. Single-input jobs never merge, so this is a no-op for uBooNE.

`FlashTensorToOpticalPCs` is **not** touched. Round 1's §2 named it as the defect
site; with the mechanism corrected (see the correction box in §2) it is innocent.

# 8. Round 2 verification

One command builds all six arms and runs every leg:
`./scripts/d99r2_flash_gate.sh` (arms ~43 min at 8/8 jobs, legs ~10 min;
existing labels are skipped, so a re-run just re-gates).

## The instrument

Not an A/B. `T_cluster` carries **`cluster_t0_us`** — written from the flash the
cluster actually matched — and **`flash_time_us`** — written from whatever the
reader resolved — on the same row, both as the same double over the same
constant. So

```
flash_time_us == cluster_t0_us            (exact, on every matched row)
```

is a **within-file identity**: no baseline arm, no archive census, no
cross-stage join. That last point is why it exists — round 1's causal check
predicted moved rows from a Q/L-stage census and failed on 24 events purely
because `T_cluster` is written from the *PR*-stage grouping
(`feedback_diff_tool_cannot_prove_containment`). `d99_tcluster_flash_check.py`
reads one file and asks it about itself.

**Trap, and it cost a false FAIL on the first run.** "Matched" is *not* `t0 != 0`.
`QLMatching.cxx:1351` pre-stamps **every** cluster with
`(cluster_t0 = -1e12 ns, flash = -1, matched_flash_gid = -1)` and a cluster
nothing matches keeps it — `-1e9` in the µs this tree stores. 19 of 20 175 rows
are that sentinel; both fixes were right to resolve them to nothing. The tool
now counts and prints them instead of putting them in the denominator.

## Results, 308-event manifest, 20 156 matched rows

| leg | claim | result |
|---|---|---|
| **A1** stage A, Q/L member content, `d99r2off` vs `d99fix` | both knobs off ⇒ nothing moves | **PASS** 308/308 events, 1232 products |
| **A2** stage B archives, `d99r2offpr` vs `d99fixpr` | ditto | **PASS** 616/616 (38+96+482), 0 unpaired |
| **A3** every ROOT branch, `--expect ""` | ditto, exhaustively | **PASS** 1938 tree instances, **236 280 branch instances, 0 differing pairs** |
| **B1** `d99r2offpr` | the defect, restated on this manifest | 10 219 / 20 156 = **50.7% CORRECT**, 9 779 WRONG, 158 MISSING |
| **B2** `d99r2rdpr` — READ on | every matched row resolves its own flash | **20 156 / 20 156 = 100.0%**, 0 WRONG, 0 MISSING |
| **B3** containment of the read knob | only the three flash columns move | **PASS** — exactly `T_cluster:flash_id`, `:flash_time_us`, `:flash_pe`, on 308 events; nothing else of 236 280 |
| **B4** stage B archives, read on vs off | the read knob touches no archive | **PASS** 616/616 byte-identical |
| **C1** stage A archives, `d99r2wr` vs `d99r2off` | the write knob DOES change the archive | 308/308 differ — **by design, not a regression** (same member count; the merged flash/light/flashlight PCs grew) |
| **C2** `d99r2wrpr` — WRITE on, read OFF | `get_flash()` itself is now right | **20 156 / 20 156 = 100.0%**, 0 WRONG, 0 MISSING |
| **D** `d99r2wrpr` vs `d99r2bothpr` | the merged row index and the gid are independent resolutions and must agree | **PASS** 20 156 rows joined, 0 only in A, 0 only in B, **0 disagreeing** on `flash_time_us` or `flash_pe` |

Leg **D** is the load-bearing one. `write_opflash_pc` runs before the merge, so
the gid path never sees the shift the write fix applies; the two answers are
arrived at by disjoint code. Agreeing on all 20 156 rows says both fixes are
right, not that they compensate.

**A single-event probe** (`work-ncpi0-d99r2smoke`, evt 18625) shows the write fix
in the archive directly: the merged flash PC goes 13 → 27 rows
(`merge_flash_pcs: input 1 shifted by flash+13 light+591 (10 clusters)`) and the
archive-side census goes 8 CORRECT / 10 WRONG → **18 CORRECT / 0 WRONG**.

## Unit tests

`clus/test/doctest_flash_by_gid.cxx`, 6 cases / 48 assertions: the invalid
inputs (no PC, gid < 0, absent gid); a gid carrying the `gid_side*1000000`
offset of a non-primary input; the channel-order sum, with a `REQUIRE` that the
two summation orders give different doubles *before* testing anything; a causal
negative control that changes only the `gid` column and watches resolution stop
while the flash stays reachable under its new key; the gid-collision refusal;
and the end-to-end case where `get_flash()` returns the WRONG real flash on the
very cluster `get_matched_flash()` gets right.

Mutating exactly the fix — drop the channel-order sort, drop the collision
guard, and make `get_matched_flash()` delegate to `get_flash()` — fails **3 of
the 6 cases and 13 assertions**; the other 3 cases still pass, so the suite is
specific to the fix and not to the code's shape.

Knob defaults are pinned in three more suites:
`root/test/doctest_sbnd_pr_tracking_defaults.cxx` and
`root/test/doctest_pdvd_tracking_defaults.cxx` (`flash_by_gid` false — on PDVD
that assertion IS the tripwire holding §9's "not validated here"), and
`match/test/doctest_qlmatching_config.cxx` (`merge_flash_pcs` false).

**Stated gap.** The write fix's offset arithmetic (`shift_flash_indices`) has
**no unit coverage** — only the default round-trip above. Reaching it from a
doctest needs a multi-input `QLMatching` node, and the alternative, extracting a
testable helper out of a production component, is what M10 forbids. Its evidence
is leg **D**: two disjoint resolutions of the same question agreeing on every
row. Naming the gap rather than papering over it.

Full suites: `wcdoctest-clus` **2863** assertions, `wcdoctest-match` **38**,
`wcdoctest-root` **4053**, all green.

**Also verified in the CANONICAL cmake build, not only waf** (`toolkit/CLAUDE.md`
builds with cmake and CI runs it with `-Werror`; waf is this tree's convention).
Two traps worth writing down:

- `root` is an **opt-in** package — a plain `cmake -S . -B build` prints
  `WCT package 'root' disabled (missing dependency: ROOTSYS)` and silently
  skips it, so none of this round's `root/` changes would have been compiled.
  It needs `-DWITH_ROOTSYS=$(root-config --prefix)`, and adding it to
  `CMAKE_PREFIX_PATH` is not enough.
- tests are **off by default** (`WCT_WITH_TESTS=OFF`), so a green cmake build
  says nothing about a new test file being picked up.

With both on: build **rc=0, zero warnings**; aggregate `wcdoctest` **684 cases /
185 097 assertions**, all passing, and the 6 new `flash_by_gid` cases are in that
count — which is what proves cmake's `file(GLOB doctest*.cxx)` found the new
file, something the waf build cannot tell you.

## Compiled-config proof

- **Off**: `scripts/cfg/prod_cfg_gate.py` — **PASS, 21/21 artifacts** identical to
  `ref/prod-2026-09-04`. Both knobs use the key-suppression idiom, so neither
  key exists in any compiled job.
- **On**: each knob adds **exactly one key to exactly one node** —
  `flash_by_gid: true` on `SbndPrMagnifyTrackingVisitor:pr` (42 nodes compared,
  1 differing) and `merge_flash_pcs: true` on `QLMatching:matching_joint` (101
  nodes compared, 1 differing).

## Freshness / provenance

`local/lib/libWireCell{Clus,Match,Root}.so` at 2026-09-02 20:46–20:47, newer
than the last source edit (20:41); snapshot pinned to `~/tmp/d99r2-libsnap` and
every arm run with `LD_LIBRARY_PATH` pointing at it, because a concurrent
session shares this tree. That session's in-progress PDVD work
(`ClusteringProtectBundle.stm_only_bundles`, `cfg/pgrapher/common/clus.jsonnet`,
`cfg/pgrapher/experiment/protodunevd/pr.jsonnet`) **is** compiled into these
libraries. Leg A1/A2/A3 covers it: with every knob off the arms are byte-identical
to round 1's, built before those edits existed — so that work is inert on SBND,
and it is **not** in either of this round's commits.

# 9. Round 2 status flags

- **Read knob OFF is byte-identical.** Proven, not argued: 236 280 branch
  instances and 616 archives with **zero** differences against the round-1 arms.
- **Read knob ON is NOT byte-identical**, by design: 3 columns × 308 events.
  `flash_id` also changes *meaning* (it becomes the gid).
- **Write knob OFF is byte-identical**, and inert by construction — the merge's
  name set is then literally `root_pcs_to_merge` and no shift runs. Same gate
  legs cover it.
- **Write knob ON is NOT byte-identical and cannot be**: the archive gains the
  non-primary inputs' flash/light/flashlight rows and every non-primary
  cluster's `flash` scalar shifts. **Every downstream hash moves.** Flipping it
  means re-validating the whole chain and re-cutting `ref/prod-<date>/`.
- **~~Nothing is flipped.~~ SUPERSEDED by §10 (2026-09-03): both knobs are now
  ON in SBND production, `ref/prod-2026-09-05`.** The flags in this section
  describe round 2, when production still ran both paths off; they are kept as
  the record of what was true then. §10.6 carries the current flags.
- **uBooNE is untouched** by both: single-input jobs never merge, and no uBooNE
  config carries either key.
- **PDVD carries the read knob in C++ and no config sets it.** The gid-uniqueness
  precondition (§6 detail 3) has not been checked for PDVD's
  `opflash_phys_gid` / `shared_flash` encodings. Check that before wiring it;
  `doctest_pdvd_tracking_defaults.cxx` asserts the default stays false so the
  omission is enforced rather than remembered.
- **Scope of what the gate actually exercised.** The arms carry the `flash`,
  `light`, `flashlight` and `opflash` PCs but **no `flashcov`** — that PC exists
  only when the light chain ran with `emit_coverage`, and none of these inputs
  did. So the write fix's `flashcov.flash` shift, and the merge's
  emplace-vs-append path for that name, are **untested by these 308 events**.
  The logic is uniform with the three PCs that were exercised, but say so rather
  than let a reader assume coverage the manifest did not provide.
- **`Facade_Flash.h`'s `operator bool()` contract was rewritten** to name both
  producers and their separate reasons for returning false, and to state that
  `ident()`/`type()`/`errors()`/`covs()` mean different things on the gid path.
  §1's whole root cause was an implementation drifting from its declared
  contract; adding a second producer of invalid Flashes without amending that
  contract would have re-created the same failure mode.

## If the owner wants to flip

The two knobs are independent and answer different questions.

- **Read only** (`flash_by_gid=true` in `wct-pr-perevt.jsonnet`) — the cheap
  one. Corrects the shipped diagnostic columns with **no archive change at all**
  (leg B4: 616/616 byte-identical), so nothing downstream of the Q/L archive
  needs revalidating. Cost: a fresh `T_cluster` stops being comparable to
  pre-flip arms in three columns, and `flash_id` means the gid.
- **Write too** (`merge_flash=true` in `wct-clus-matching-perevt.jsonnet`) — the
  correct-at-the-source one, which makes `Cluster::get_flash()` mean what its
  name says for every future consumer. Cost: every Q/L archive hash moves, so
  it needs a full chain revalidation and a new `ref/prod-<date>/`. Worth doing
  when a reconstruction consumer actually needs `get_flash()` — today none does
  (`RetileCluster` is the only one and SBND instantiates `ImproveCluster_2`).

Recommendation as of round 2: **flip the read knob, hold the write knob** until
a reconstruction consumer needs it or the next epoch re-cut is happening anyway.

**OVERTAKEN BY MEASUREMENT, 2026-09-03.** The owner flipped both. The "hold the
write knob" half rested on the cost line *"every Q/L archive hash moves, so it
needs a full chain revalidation"* — true as stated, and round 3 measured what
that revalidation actually finds: **nothing outside the optical point clouds**.
Every Bee zip, every `nusel` verdict TSV, every `calib-pr` dump and all 236 280
ROOT branch instances bar the three `T_cluster` flash columns are byte-identical
across 308 events, and the matching scalars do not move. The write knob's cost
was real but far smaller than "full chain revalidation" implies, and the
comparison that would have shown that — today's production against
production-as-flipped — is the one leg round 2 never ran. See §10.3.

# 10. Round 3 — both knobs FLIPPED to SBND production (2026-09-03)

Owner, 2026-09-03: *"Please flip both knobs on and perform the relevant
validations. ... By the way since the write and read was bugs, do we really need
a knob, shouldn't it be default on to avoid confusions?"*

Both knobs are now **ON in SBND production config**. No C++ default changed —
see §10.5 for why that is the answer to the second question rather than a dodge.

## 10.1 What moved — four SBND files, nothing else

| file | knob | was | now |
|---|---|---|---|
| `cfg/pgrapher/experiment/sbnd/qlmatching.jsonnet` | `matching_joint(merge_flash=)` | false | **true** |
| `cfg/pgrapher/experiment/sbnd/wct-clus-matching-perevt.jsonnet` | TLA `merge_flash` | false | **true** |
| `cfg/pgrapher/experiment/sbnd/clus.jsonnet` | `pr(flash_by_gid=)` | false | **true** |
| `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` | TLA `flash_by_gid` | false | **true** |

**Why the module defaults and not only the two TLAs.** The per-event jobs are
not SBND's only callers. `wcp-porting-img/sbnd/wcls-img-clus-matching-xin.jsonnet`
— the **LArSoft** chain — calls `qlm.matching_joint(...)` without passing
`merge_flash`, and calls `clus_maker.pr(...)` without passing `flash_by_gid`
(`pr-operating-point.jsonnet`, which syncs that chain's PR operating point to the
TLA defaults, does not carry either key). A TLA-only flip would have fixed the
standalone per-event chain and left LArSoft on the defect — silently, because
both entry points would still compile and run. This is doc 97's recorded lesson
(`prod.wcls` drifting "is the point, not a side effect") applied before rather
than after the fact.

`pdhd` and `pdvd` import their **own** `qlmatching.jsonnet` and
`protodunevd/pr.jsonnet`, so none of these four lines reaches them.

## 10.2 The compiled-config proof — the flip is exactly two keys

Compiled the full 21-artifact consumer set before and after
(`scripts/cfg/compile_consumers.sh`). The pre-flip set is **21/21 identical to
`ref/prod-2026-09-04`**, so the drift below is attributable to this change and
nothing else.

```
DRIFT : prod_prjob.json   ADDED  [24].data.flash_by_gid    = True   (node "pr")
        sbnd_ql.json      ADDED  [72].data.merge_flash_pcs = True   (node "matching_joint")
same  : the other 19 — every pdhd (6), every pdvd (5), uboone.json,
        prod.wcls, prod.standalone, sbnd_{clus,img,pr,simcheck}.json, bare_prjob.json
```

One key per artifact, one artifact per knob, and **every** instance of each
component carries it (one `SbndPrMagnifyTrackingVisitor` node, one `QLMatching`
node). `prod.standalone` is unmoved because it compiles with
`--ext-code joint=false`: two single-input `matching0`/`matching1` nodes, where
the merge loop never executes, so the write knob is inert there by construction.

**Equivalence, both directions, byte-identical JSON:**

- **The flip *is* the arm that was gated.** Round 2's arms were produced by
  overriding the knobs through the runner's TLA hatch on a tree whose defaults
  were false. Compiling the **pre-flip** tree with `--tla-code flash_by_gid=true`
  / `merge_flash=true` reproduces the **post-flip** tree's default compile
  byte-for-byte, for both jobs. So flipped production is not *similar to* the
  configuration that measured 20156/20156 — it is the same JSON.
- **OFF is still recoverable.** Compiling the **post-flip** tree with explicit
  `false` reproduces the **pre-flip** artifacts byte-for-byte, for both jobs. The
  legacy path did not become unreachable; an A/B against it is still one TLA away.

## 10.3 What the flip actually changes — measured, 308 events, not inferred

Round 2 gated the knobs individually but never compared **today's production**
against **production-as-flipped**. That comparison is what a flip needs, and it
runs entirely on arms already on disk: `d99r2off{,pr}` (production) vs
`d99r2wr` / `d99r2bothpr` (both knobs on).

### Stage A (Q/L), per product, all 308 events

| product | result |
|---|---|
| `mabc-all-apa.zip`, `mabc-apa0-face0.zip`, `mabc-apa1-face0.zip` | **308/308 byte-identical** |
| `opflash_apa0.tar.gz`, `opflash_apa1.tar.gz` | **308/308 byte-identical** |
| `pctree-evt<ID>.tar.gz` | 308/308 differ — see below |

### Inside the archive that does move — by datapath, not by member name

The pctree names its members by **tensor index**, and adding point clouds
renumbers every tensor after the insertion point, so a member-name diff reports
"everything moved" and answers nothing. Keying on each tensor's own `datapath`
(`scripts/analysis/d99r3_pctree_datapath_diff.py`) gives the real answer, and it
is the same on all 308 events:

```
ADDED   0 datapaths
REMOVED 0 datapaths
CHANGED 16 datapaths, every one of them optical:
    flash/arrays/{ident,time,tmax,tmin,type,value}
    light/arrays/{error,ident,time,value}
    flashlight/arrays/{flash,light}
    lpcmaps/arrays/{flash,flashlight,light}
    cluster_scalar/arrays/flash            <- the re-based row index; the fix itself
```

**The `cluster_scalar` arrays that did NOT change are the licence for this
flip:** `cluster_t0`, `matched_flash_gid`, `flag_main_cluster`,
`flag_associated_cluster`, `lm_flag`, `ident`. Those six *are* the matching
result. The merge at `QLMatching.cxx:1109` runs **before** matching at `:1351`,
so if carrying every input's flash PCs had handed the matcher more candidate
flashes, matching itself would have moved and this would be a reconstruction
change needing owner adjudication under CLAUDE.md §5.5. It did not, on any of
the 308 events, and the byte-identical Bee zips say the same thing independently.

### Stage B (PR), all 308 events

| product | result |
|---|---|
| `nusel-evt<ID>.tsv` — the per-bundle tagger verdicts | **308/308 byte-identical** |
| `calib-pr-evt<ID>.json` (timer field masked) | **177/177 identical**; 131 absent in *both* arms |
| `mabc-pr.zip` | **byte-identical** |
| every ROOT branch — 1938 tree instances, **236 280 branch instances** | **3** differing pairs: `T_cluster:flash_id`, `:flash_time_us`, `:flash_pe` |
| `pctree-pr-evt<ID>.tar.gz` | the same 16 optical datapaths, 0 added, 0 removed |

**The Q/L hand-scan calib dump cannot be affected at all**, and that is
structural rather than measured: `dump_calib(runs)` is called at
`QLMatching.cxx:1089`, *before* the merge at `:1109`, and reads the per-APA
`runs` — not the merged tree. `ql_scan/ql_scan_viewer.py` reads `flash_id` from
that dump (and already resolves through `flash_by_gid[b["flash_gid"]]`), so the
owner's Q/L viewer is untouched by either knob.

### And the flip is the *correct* state

| arm | matched rows | CORRECT |
|---|---|---|
| `d99r2offpr` — today's production | 20156 | 10219 = **50.7%** (9779 WRONG, 158 MISSING) |
| `d99r2rdpr` — read on | 20156 | **20156 = 100.0%** |
| `d99r2wrpr` — write on, read off | 20156 | **20156 = 100.0%** |
| `d99r2wrpr` vs `d99r2bothpr` | 20156 joined | **0 disagreeing** |

## 10.4 The flip gate — production reaches the flipped default with no hatch set

The compiled-config proof cannot show that the **runner** reaches the flipped
default when no TLA hatch is set, because the runner, not `wcsonnet`, assembles
the command line. `scripts/d99r3_flip_gate.sh` runs one sample end to end with
`QL_EXTRA_TLA`/`PR_EXTRA_TLA` **empty** and byte-gates the result against the
round-2 arms. Binary pinned at `~/tmp/d99r2-libsnap` on purpose: a concurrent
session landed two `clus` commits (`19830863`, `c27ec4b1`) after round 2's arms
were produced, so an unpinned run would compare a config change against a moving
binary.

Result — `work-ncpi0-d99r3prod{,pr}`, 19 events, no TLA hatch set:

| leg | result |
|---|---|
| stage A: production defaults == `d99r2wr` | **PASS** 19/19 events, 76 products, 0 differing |
| stage B archives == `d99r2bothpr` | **PASS** 38/38 byte-identical |
| every ROOT branch vs `d99r2bothpr`, `--expect ""` | **PASS** 152 tree instances, 24 985 branch instances, **0** differing pairs |
| `T_cluster` self-check, `--require-correct` | **PASS** 1871/1871 = **100.0%** |
| negative control: same 19 events pre-flip (`d99r2offpr`) | 895/1871 = **47.8%** — the instrument can fail; it just does not here |

The `--expect ""` on leg 3 is load-bearing: the claim is that nothing moves at
all, not that only the flash columns move. Reusing round 2's non-empty EXPECT
would have silently permitted the very difference under test.

## 10.5 "Since these were bugs, shouldn't the knob be default ON?"

The instinct is right about the end state and wrong about the mechanism, for one
concrete reason: **a C++ default governs only the configs that do not emit the
key.** Now that SBND emits both keys, flipping the C++ defaults would change
nothing about SBND. Its entire remaining effect is on the *other* binders:

| knob | who else the C++ default governs |
|---|---|
| `merge_flash_pcs` | **pdhd** and **pdvd** — they import their own `qlmatching.jsonnet` and set no key. (uBooNE never reaches it: it does not use `QLMatching` at all.) |
| `flash_by_gid` | **`PdvdPrMagnifyTrackingVisitor`**, the fork-by-duplication sibling, whose gid-uniqueness precondition has never been checked. |

So "default ON to avoid confusion" would, in practice, ship an unvalidated
archive change to PDHD and PDVD and silently enable a read path on a detector
whose gid encoding (`opflash_phys_gid` / `shared_flash`, per-drift-side flash
lists) may not satisfy the precondition the resolver requires. That is CLAUDE.md
§5.1 — a change that cannot be made byte-identical-when-off for those detectors —
and it buys SBND nothing.

**What does remove the confusion is deleting the knob and the legacy path**, so
there is exactly one behaviour and nothing to be confused between. That is this
tree's own recorded precedent: *never hard-code a settled-ON knob — delete the
feature instead.* Leaving a knob permanently pinned true is the worst of the
three options, because the dead branch stays in the code and every reader has to
work out which side runs.

That deletion is a separate round, and it is gated on two things that do not
exist yet:

1. **PDVD's gid encoding checked** against the uniqueness precondition
   (§6 detail 3), then its knob wired and gated — or a decision that PDVD keeps
   the row-index read, in which case the two forks legitimately diverge and only
   the SBND one loses its knob.
2. **PDHD and PDVD gated for `merge_flash_pcs`**, the same way SBND was here:
   their archives move, so each needs its own before/after and epoch re-cut.

Until then the knob is also still doing real work: it is what makes the pre-flip
archive recoverable byte-for-byte (§10.2), which is how every A/B in this tree is
run. Delete it when nothing needs that comparison any more, not before.

**Recommended next step** (owner's call, not started): schedule the PDVD
precondition check as its own short round. It is the single blocker on both
deletion items, it is a read-only investigation of PDVD's gid construction, and
it also settles whether PDVD has been carrying the same wrong-flash defect in its
own `T_cluster` all along — which nobody has measured.

## 10.6 Round 3 status flags

- **SBND production is now ON for both knobs**, `ref/prod-2026-09-05`,
  21/21 artifacts pinned and re-verified.
- **NOT byte-identical, by design and by owner decision.** Stage-A and stage-B
  pctree archives move; every downstream hash of a *pctree* moves with them.
  Everything else does not — see the §10.3 tables.
- **No reconstruction changed.** Bee zips, `nusel` verdict TSVs, `calib-pr`
  dumps and all 236 280 ROOT branch instances outside the three `T_cluster`
  flash columns are byte-identical across 308 events. The matching scalars
  (`cluster_t0`, `matched_flash_gid`, `flag_main_cluster`,
  `flag_associated_cluster`, `lm_flag`) are unchanged, which is what licenses
  that sentence rather than an inference from the knob's intent.
- **`flash_id` has changed MEANING in production**, not just value: it is the
  gid (`gid_side * 1000000 + index`), joinable to the `opflash` PC's `gid`
  column, not a small per-input row id. Any script that assumed a small index
  must be re-read. Checked: the Q/L hand-scan viewer
  (`ql_scan/ql_scan_viewer.py`) reads `flash_id` from the **calib dump**, not
  from `T_cluster`, and already resolves through `flash_gid` — it is unaffected.
- **The C++ defaults stay false**, deliberately; §10.5 says why and what the
  end state is. `doctest_qlmatching_config.cxx` and
  `doctest_sbnd_pr_tracking_defaults.cxx` now say out loud that a green run
  there does **not** mean production is on the legacy path.
- **Pre-flip arms are still comparable.** Any arm recorded before 2026-09-03
  differs from a fresh one in the three `T_cluster` flash columns and in the 16
  optical datapaths. Pass them to `d99_root_branch_census.py --expect` or
  exclude them; do not read the difference as a regression.
- **`flashcov` remains unexercised** (§9): the write fix shifts it, no SBND input
  carries it, and the flip makes that branch reachable rather than dormant.
