# Doc 81 — Re-baseline the four SBND production samples through group (multi-event) mode (2026-08-25)

Owner ask: *"I just worked out the group processing mode, also multiple events
cases in doc 76, now I would like to run these and collect input and output in
the ./sbnd_xin/ directory work*. This is to prepare the upcoming work in MCS and
PR. … we can probably retire some of the existing work* directory … Note, the
validation should ensure byte identical with my existing standalone single
event single stage production."*  Samples, owner-specified: **nueCC48 (48),
NCpi0 (19), mcp1k (1000), mcp2k (2000) = 3067 events.**

**Result.**  The chain now runs reco1 → imaging → clustering+Q/L → PR in GROUPS
of 16 events per `wire-cell` process and writes **exactly the per-event file
layout** a one-event-per-process job writes, so the products are a file-for-file
drop-in for `work-img-<s>` / `work-<s>-ql0819` / the PR arms.  Getting there
needed two C++/config changes and exposed **one silent physics defect** that no
gate in this project was in a position to see:

* **Group-mode IMAGING was not byte-identical** and never had been gated.
  `GridTiling` restarts its blob-ident counter only on EOS, and a group gets no
  EOS between events, so every event after the first was mis-identified — and
  because blob idents key `unordered_map<int,…>` containers downstream, that
  changed blob **counts and charges**, not just labels (sec 2).  Fixed
  (toolkit `30f061ce`); group imaging is now byte-identical.
* **`work-*-prod0823` is a PRE-flip baseline, not current production.**  Two
  production flips landed after it was produced — `fast_xgb_forest` (doc 76
  round 1) and the dual chain / `snapD2` (doc pr/112).  The correct reference is
  `work-pr112i-snapD2-*` (sec 4).

## Repro

```bash
cd wcp-porting-img/sbnd/sbnd_xin
S=<scratch>/doc81

# stage A -- reco1 -> imaging -> clus+Q/L, one process per stage per GROUP,
#            writing the per-event layout
./run_chain_group.sh <reco1.root> work-<s>-grp0825 data --size 16 --layout perevt
#   NCpi0 also needs   --fsproduct 'sbnd::timing::FrameShiftInfo_frameshift__FILTERFRAMESHIFT.'
#   mcp2k is two 1000-entry files: second invocation adds --gbase 63

# stage B -- the 15-stage PR chain, in groups
PR_GROUP_SIZE=16 PR_JOBS=3 PR_EXTRA_STAGES=pr_display \
    ./run_pr_chain_batch.sh work-<s>-grp0825 work-<s>-prod0825 data

# gates
python3 scripts/multi/stagea_gate.py work-<s>-grp0825 \
        --img work-img-<s> --ql work-<s>-ql0819 --jobs 8      # NEW this round
python3 scripts/pr85_hash_gate.py --jobs 6 work-<s>-prod0825 work-pr112i-snapD2-<s>
python3 scripts/pr94_root_gate.py          work-<s>-prod0825 work-pr112i-snapD2-<s>
diff <(sort work-<s>-prod0825/nusel-table.tsv) <(sort work-pr112i-snapD2-<s>/nusel-table.tsv)
```

Binary: toolkit `2aba11dc` (this round's two commits on top of `54cc7649`),
wcp-porting-img `b04a94e`.  M1 freshness proved before every arm.
Unit tests: `wcdoctest-{img,sio,clus,root,util}` all pass (img 15/15 is new).

## 1. Round 0 — the premise checks, and why they were worth running

Nothing entered a work root until all of these passed.

| check | result |
|---|---|
| 0a HEAD per-event PR vs `work-<s>-prod0823`, 4 events | PASS — *and misleading, see sec 4* |
| 0b reco1 dump vs the recorded staged frames, 16 events | **PASS** 16/16 byte-identical |
| 0b all five reco1 art files readable (incl. the two mcp2k parts under `/nfs/data/1/yuhw/`) | PASS |
| 0c imaging ident COLLISION in a group npz | PASS — members are keyed `cluster_<EVENTID>_<kind>.npy`, counts match exactly |
| 0c imaging group-vs-per-event CONTENT | **FAIL — sec 2** |
| Q/L group mode on identical imaging, 16 events | **PASS** 6578/6578 pctree members identical to `ql0819` |

The last row is the one that kept the round alive: it localised the failure to
imaging alone and proved the Q/L half of stage A was already correct.

## 2. The imaging defect (toolkit `30f061ce`)

**Symptom.**  A 16-event nueCC48 group vs the recorded per-event
`work-img-nuecc48`: **126 of 672 npz members differed.**  Only the event at
group position 0 was identical.

**Controls**, all on one binary:

| arm | vs the recorded hub |
|---|---|
| single-event (group of 1), rep A | **42/42 identical** |
| single-event, rep B | **42/42 identical** — run-to-run deterministic |
| group of 16 | 4 of 42 members differ |
| group of 16, rep A vs rep B | **672/672 identical** — deterministic, so carried STATE, not concurrency |

Ruled out: input ordering.  Each event's five frame members are contiguous in
the group archive and `FrameFileSource` reads the 16 idents in `events.txt`
order — this is *not* the doc 76 §10.5 member-order trap.

**Root cause.**  `img/src/GridTiling.cxx:173` assigns blob idents from
`m_blobs_seen++`.  The counter is reset only on EOS (`:56-59`) and **a group
never gets an EOS between events** — measured: 14 `GridTiling` EOS lines for a
16-event group, exactly the same as for one event.  So event N's blobs were
identified from an offset (evt 388 started at 6952 instead of 0).

**Why that is not cosmetic.**  Blob idents key the `unordered_map<int,…>` /
`unordered_set<int>` containers in `InSliceDeghosting`, `ProjectionDeghosting`,
`BlobGrouping` and `LocalGeomClustering`, whose **iteration order depends on the
key values**.  An offset therefore flips order-dependent deghosting decisions.
Measured per event and APA over the group: most differ only in the `bnodes`
ident column, but six (event, APA) pairs differ in blob **count** or blob
**value** (e.g. bnodes 2219→2168, bwedges 44319→43232).

**Fix.**  Restart the sequence at every frame-ident change.  Inert for a
one-event process — the reset fires once, on the first slice, when the counter
is already 0 — so the legacy path is byte-identical by construction and this
ships without a knob, on the doc 76 round 3 precedent (the behaviour it replaces
on event ≥ 2 was simply wrong).

**Gates.**

| gate | result |
|---|---|
| group of 16 vs recorded `work-img-nuecc48` | 126 differing → **672/672 byte-identical** |
| per-event (legacy) on 3 events vs recorded | **126/126 identical** |
| PDHD + PDVD clustering, `ab_compare.sh post_doc76r2 post_doc81img` | **OVERALL PASS** |
| uBooNE MABC, `qlport/ab_check.sh doc81img doc76r3` | zips **35/35**, tagger **35/35** |
| `wcdoctest-img` | **15/15**, revert-proven: with the reset removed, 2 of the 15 fail |

The PDHD/PDVD **imaging** leg still cannot be run — `run_img_evt.sh` needs
`pdhd/work/<run>_<evt>/protodunehd-sp-dnnroi-frames-anode*.tar.bz2`, which the
NF+SP step has to produce first (doc 76 §10.6 recorded the same).  The clustering
leg was run instead; the direct imaging evidence is the SBND gate above.

## 3. The per-event layout (toolkit `2aba11dc`, wcp `b04a94e`)

`wct-pr-perevt.jsonnet` has had `evt_subdir` since doc 76 round 2; the Q/L job
had none.  Added, and the Q/L Bee zip paths routed through the same prefix.
`clus_per_face()` / `clus_all_apa()` are top-level local functions that do not
see the object-scope `evt_out_prefix`, so each takes its own `evt_subdir=''`
parameter — the first attempt used the object local and failed to compile,
which is exactly the M6 shape.

With `evt_subdir` set, MABC's `bee_zip` takes `BeeSink`'s templated path, which
restarts each event's Bee index at 0 — the file a per-event job writes, instead
of a group zip carrying N layer sets at indices 0..N-1, which **cannot** match
by construction.

Two stage-A products have no `%`-template — the imaging npz and the opflash
tars — so `scripts/multi/split_group_products.py` splits them by the event id
already in every member name; the split is verified member-for-member and
refuses on any unkeyed or unaccounted member.  `ClusterFileSink` was
deliberately **not** given `%`-templating: imaging byte-identity is implied
transitively (Q/L reads the npz), and it is a shared `sio` sink every detector
loads.

**Compiled-config proof**, each key probed in both states (doc 77 r2's lesson):

| compile | result |
|---|---|
| Q/L job, `evt_subdir` absent | **byte-identical** to the pre-change compile, zero `ql_evt` occurrences |
| Q/L job, `evt_subdir='ql_evt%1%'` | all three Bee zips under `ql_evt%1%/` |
| PR job | **byte-identical** (it already used the prefix) |

**Stage-A gate** (`scripts/multi/stagea_gate.py`, new — nothing in this tree
compared imaging or Q/L products between two work roots before; `ql_legacy_gate.sh`
re-*runs* Q/L per event and symlinks imaging in, so it gates a binary change,
not a layout):

| sample | archives | result |
|---|---|---|
| nueCC48 48 events | 384 | **PASS** member-content identical to `work-img-nuecc48` + `work-nuecc48-ql0819` |
| NCpi0 19 events | 152 | **PASS** |

## 4. `work-*-prod0823` is a PRE-flip baseline — the reference correction

Stage B at HEAD differed from `prod0823` on 4 of 16 nueCC48 events, in `T_kine`
only (`kine_nu_{x,y,z}_corr`, `kine_energy_*`).  Controls:

| test | result |
|---|---|
| PR group vs PR per-event, same input | **16/16 identical** — PR group mode is clean, doc 76 r3's fix holds |
| PR per-event run twice | **identical** — reproducible, not ASLR noise (M4) |
| opflash, RSE and pctree, new root vs `ql0819` | identical |
| **compiled PR config, HEAD vs prod0823** | **differs in production knobs** |

HEAD carries `dl_vtx_dual_chain: true`, `dual_chain_mode: "snap"`,
`dual_chain_transfer: true`, `dual_chain_transfer_max: 2` and
`fast_xgb_forest: true`; `prod0823` has none of them.  Those are the doc 76
round 1 and doc pr/112 (snapD2) **production flips, both landed after prod0823
ran on 2026-08-23**.  Against the post-flip arms HEAD is byte-identical:

| sample | vs `work-pr112i-snapD2-<s>` | vs `work-pr112i-flipchk-<s>` |
|---|---|---|
| nueCC48 48 | pr85 **96/96**, pr94 **48/48** | pr85 **96/96**, pr94 **48/48** |
| NCpi0 19 | pr85 **38/38**, pr94 **19/19** | pr85 **38/38**, pr94 **19/19** |

**Round 0's 0a spot check passed against `prod0823` only because those four
events are not moved by the dual chain.**  A 4-event spot check cannot see a
production flip — worth recording as the reusable lesson: *size a
"has production moved?" check against the population the flip actually
touches, not against convenience.*

## 5. The nusel-table residual is pre-existing WCT log tearing

6 rows of 771 differ between the new arms and the reference, all in the
log-parsed `fc` / `stmfit` columns, while `pr94` passes on every branch of every
event.  Traced to one event: the line
`check_stm_conditions: cluster 14 no STM fit: fully contained (Mid Point A)`
was **torn** and spliced into a `CreateSteinerGraph` warning, leaving
`… for assoc 67, ter 14 no STM fit: …` — the word `cluster` itself was cut, so
even doc 76 round 3's relaxed `RE_STM_SKIP` cannot match it.

The `evt<ID>` stamps are correct and `slice_group_log.py` recovers the lines it
should; this is not a slicing failure.  **And it is not group-specific.**
Measured tear rate:

| log | lines | torn |
|---|---:|---:|
| group log `wct_pr_g0.log` | 31232 | 10 |
| this round's sliced per-event log, evt 163543 | 1837 | 3 |
| **the reference arm's own per-event log, evt 163543** | 1879 | **5** |

The per-event production log tears *more* than the group slice.  So every nusel
table this project has ever produced carries this noise; the campaign only made
it visible by diffing two of them.  Owner decision: **document and proceed** —
the physics products are byte-identical, and the logging fix is its own round.

## 6. Runner defects fixed on the way

Each would have corrupted a 3067-event campaign quietly:

* `--to img` always reported FAILED (the group prints `stopped after imaging`,
  the success check greps for `ok`).
* **A failed reco1 dump left a 42-byte `frames-dnn.tar.bz2`, and the
  skip-if-present guard uses plain `-s`** — the retry skipped the dump and every
  later stage ran on zero events, writing empty products and reporting a split
  failure three stages downstream.  The guard now requires the archive to list
  frames, a failed dump deletes it, and a zero-event group refuses to continue.
* No way to reach a sample split across several reco1 files: `--gbase` offsets
  the `g<K>` directory names so mcp2k's two parts share one out_root instead of
  the second overwriting the first's frames.
* `--fsproduct`, because the NCpi0 sideband file carries
  `…__FILTERFRAMESHIFT.` (doc 71 §3) and the dump aborts rather than falling
  back.  Default empty ⇒ the TLA is not passed ⇒ the jsonnet omits the key.

Still open, recorded not fixed: **group mode writes `rc=0` unconditionally for
every event of a passing group** (`run_pr_chain_batch.sh:1811`), so the tail
check at `:1874-1892` cannot see a per-event failure inside a group that exited
0.  Coverage in this round is therefore counted from real per-event product
existence, never from `rc.txt`.

## 7. Campaign coverage and stage-A gate

Stage A, all four samples, group of 16, `--layout perevt`:

| sample | events | rc | stage-A archives vs `work-img-<s>` + `work-<s>-ql0819` |
|---|---:|---|---|
| nueCC48 | 48 | 0 | **PASS 384/384** |
| NCpi0 | 19 | 0 | **PASS 152/152** |
| mcp1k | 1000 | 0 | **PASS 8000/8000** |
| mcp2k | 2000 | 0 | **PASS 16000/16000** |
| **total** | **3067** | | **PASS 24536/24536 member-content identical** |

Imaging was clean on the first pass everywhere.  **Seven events needed their
Q/L re-run outside the group** to reach that PASS — see sec 7.1.

### 7.1 Multi-event Q/L is run-to-run NON-DETERMINISTIC on marginal events

The first stage-A gate failed on 7 of 3067 events (mcp1k 286191, 292643;
mcp2k 53793, 99438, 161043, 321101, 350816), always in the Q/L products and
never in imaging.  The differing tensors are the point clouds
(`2dp*_x/_y`, `ucharge_*`, `uwire_index`, `center_*`) and the cluster-identity
arrays (`real_cluster_id`, `assoc_cluster_id`, `isolated`) — but the cluster
COUNTS agree (`normalize_cluster_flags … nclusters=18/38` in both arms), so it
is an ordering/accumulation difference, not different clustering decisions.

**It is not carried state.**  A minimal 2-event reproducer (`285993`, then
`286191`, ~2 min) was built by assembling a group by hand.  Running the *same
pair twice gives different answers*:

| run | events | result vs `ql0819` |
|---|---|---|
| solo | 286191 | MATCH (4 of 5 draws) |
| pair, draw 1 | 285993 286191 | **DIFFER** |
| pair, draw 2 | 285993 286191 | MATCH |
| pair, draw 3 | 285993 286191 | MATCH |
| pair, different predecessors (285443, 285531) | | MATCH |

So this is run-to-run instability, not a value inherited from a specific
predecessor — the doc 76 round 3 hypothesis does **not** apply here.  Prime
suspect is the CLAUDE.md §2 hazard: ~14 pointer-keyed containers in the
clustering chain (`std::map<const Cluster*, int> flash_t0_group`,
`std::unordered_set<const Cluster*> used_clusters`, … in
`clustering_{close,extend,cathode_connect,parallel_prolong,examine_bundles}.cxx`),
some already guarded by `cluster_less_functor` and some not.  Iterating one of
those in address order makes the result depend on heap layout, which differs
between a one-event and a many-event process.

**Per-event Q/L at HEAD is byte-identical to `ql0819` on every one of the seven**
(`scripts/multi/ql_legacy_gate.sh`), so the legacy production path is unaffected
and `ql0819` is not in question.

**Resolution (owner decision).**  Converge the seven and audit separately: each
event's Q/L was re-run outside the group until it matched the reference — six
via a single-event group, and **99438 via the legacy `run_ql_evt.sh` driver**,
because it did not converge in three single-event-group tries (that event
differs *systematically* between the two drivers, not just occasionally, which
is a second thread for the audit round to pull).  The delivered arms are
therefore **3060 events from group Q/L + 7 re-run per-event**, all byte-identical
to `work-<s>-ql0819`.  Recorded here rather than smoothed over.

The determinism audit is **not** done in this round: it touches shared
clustering code that every detector runs and needs its own PDHD/PDVD/uBooNE
gates.  The 2-minute reproducer above is the handle for it.

## 8. Stage B and the nusel tables

Stage B ran `PR_GROUP_SIZE=16 PR_JOBS=8 PR_EXTRA_STAGES=pr_display` (the same
extra stage prod0823 used; without it the outputs are not comparable) into
`work-<s>-prod0825`.

**Coverage is counted from real per-event product existence**, never from
`rc.txt`: group mode writes `rc=0` unconditionally for every event of a group
that exits 0 (`run_pr_chain_batch.sh:1811`), so the tail check at `:1874-1892`
cannot see a per-event failure inside it.  Requiring `mabc-pr.zip` +
`pctree-pr-evt<ID>.tar.gz` + `tracking-pr.root` all non-empty:

| sample | pr_evt | complete | incomplete |
|---|---:|---:|---:|
| nueCC48 | 48 | 48 | 0 |
| NCpi0 | 19 | 19 | 0 |
| mcp1k | 1000 | 1000 | 0 |
| mcp2k | 2000 | 2000 | 0 |
| **total** | **3067** | **3067** | **0** |

### 8.1 Byte-identity gates against the current-operating-point reference

Both tools are required and neither subsumes the other: `pr85_hash_gate.py`
compares **archive member content** (`mabc-pr.zip`, `pctree-pr-evt<ID>.tar.gz`)
and never opens the ROOT file (`pr85_hash_gate.py:34-40`); `pr94_root_gate.py`
compares **every branch of every `tracking-pr.root`**.  Read their counters, not
just the verdict: `pr85` reports an arm with a different archive set as
*missing/unpaired* rather than FAIL (`:66-70`), and `pr94` reports a missing
ROOT file as *skipped* (`:61-63`), so a truncated arm can look clean.  Both
counters are quoted below and both are zero everywhere.

Reference is **`work-pr112i-snapD2-<s>`**, not `work-*-prod0823` — see sec 2.

| sample | reference | pr85 archives | unpaired | pr94 events | differing | skipped |
|---|---|---:|---:|---:|---:|---:|
| nueCC48 | `work-pr112i-snapD2-nuecc48` | 96 | 0 | 48 | 0 | 0 |
| NCpi0 | `work-pr112i-snapD2-ncpi0` | 38 | 0 | 19 | 0 | 0 |
| mcp1k | `work-pr112i-snapD2-mcp1k` | 2000 | 0 | 1000 | 0 | 0 |
| mcp2k | `work-pr112i-snapD2-mcp2k` | 4000 | 0 | 2000 | 0 | 0 |
| **total** | | **6134** | **0** | **3067** | **0** | **0** |

**PASS: every stage-B archive and every ROOT branch of all 3067 events is
byte-identical to the per-event, per-stage production.**

Cross-check against the second recorded arm of the same operating point, which
was produced by an independent invocation (pr/112's flip check):

| sample | reference | pr85 archives | unpaired | pr94 events | differing | skipped |
|---|---|---:|---:|---:|---:|---:|
| nueCC48 | `work-pr112i-flipchk-nuecc48` | 96 | 0 | 48 | 0 | 0 |
| NCpi0 | `work-pr112i-flipchk-ncpi0` | 38 | 0 | 19 | 0 | 0 |

Repro (any row):

```
cd sbnd_xin
python3 scripts/pr85_hash_gate.py --jobs 10 work-<s>-prod0825 work-pr112i-snapD2-<s>
python3 scripts/pr94_root_gate.py            work-<s>-prod0825 work-pr112i-snapD2-<s>
```

Note this gate does **double duty**, and the second job is the one worth
stating: it tests *group == per-event*, and it also re-tests *HEAD == the
recorded baseline* at 3067-event scale.  Docs 78 and 79 flipped new SBND Q/L
defaults ON claiming byte-identity on 186 events, and doc 76 r1 flipped
`fast_xgb_forest` on 308; those claims are now carried on the full production
population.

### 8.2 `nusel-table.tsv`: 72 of 34835 rows, all log-parsed columns

| sample | rows | differing lines |
|---|---:|---:|
| nueCC48 | 547 | 4 |
| NCpi0 | 224 | 2 |
| mcp1k | 11429 | 22 |
| mcp2k | 22635 | 44 |
| **total** | **34835** | **72 (0.21 %)** |

Every one is in `fc` or `stmfit` — the two columns `nusel_extract.py` recovers
by parsing the job log (`eval`/`tgm`/`contained`/`0`/`-1`), while `pr94` compares
every branch of every `tracking-pr.root` and passes.  Cause: sec 5's
pre-existing WCT log tearing, which destroys a parsed verdict line outright.
The rate here (0.21 %) is the same order as the measured tear rate, and the
reference arm's own per-event logs tear *more* than the group slices.

## 9. Retirement

The campaign's own products supersede four families of study arms.  The plan is
built and all ten asserts pass; **the deletion itself is held for the owner** —
this section records what is staged, not what has been removed.

```
python3 scripts/retire/plan_20260825.py        # 10 asserts -> OVERALL: PASS
python3 scripts/retire/archive_records_20260825.py
CONFIRM=yes scripts/retire/retire_20260825.sh A
```

### 9.1 Scope

| family | arms | why released |
|---|---:|---|
| pr/112 + pr/112i option scan | 54 | round shipped (`snapD2` SBND production ON); superseded by the gated `prod0825` arms |
| `work-pr104-on4-*` | 4 | the 08-23 round's own deferred promise, due now that `prod0825` covers all 3067 events |
| `work-pr104-flipchk-*` | 4 | same |
| `work-vtx106-*-nuecc48` | 4 | pr/111 closed |
| **total** | **66** | **72.6 G**, 38 survivors |

The plan's headcount estimate was 60 arms; the removal set is 66.  Every one of
the six is a pr/112 variant (`off`/`snapD1-3`/`uniW*`/`vox`/`trad`/`nofitx`/
`harv`/`probe`/`fid`/`cne`/`noswap`/`flipchk`) — the family was undercounted in
the estimate, not widened here.  Classified against the approved families,
**zero dirs fall outside them**.

### 9.2 The check the ten asserts cannot make

ASSERT 10 greps *scripts* for arm names and is blind to a **doc's** table — the
reason `work-vtx106-cne-*` nearly went in the 08-23 round.  The hand-read this
round found the trap has a second mouth: the pr-numbered docs live in
`docs/pr/`, so a `docs/*.md` sweep misses the newest and most relevant ones
entirely.  Over `docs/**/*.md`, 21 of the 66 arms are cited by some doc:

| doc | arms cited | status |
|---|---:|---|
| `docs/pr/112_dl-vertex-options-short-of-retraining.md` | 11 | closed — "SBND PRODUCTION ON, `snapD2`" |
| `docs/81_group-mode-production.md` (this doc) | 6 | gate references — see 9.3 |
| `docs/work-tags.md` | 4 | the tag registry, updated by the round |
| `docs/pr/111_dl-vertex-vs-exclusion.md` | 2 | closed — keep `fit_exclusion` ON |
| `docs/76_production-perf-round1.md` | 1 | closed |

Every citing round is closed and every cited arm is inside an approved family.

Two further closure checks, both run before anything was staged:

* **label sources** — 3869 label/record files under `vertex_labels/` and
  `dl_vtx_training/` reference 21 distinct `work-*` arms; **none** is in the
  removal set, and the live `work-vtx105-base-*` epoch survives.  All six
  `vtxscan-*` tag dirs that `pr112_dual_eval.py:31-33` names are present and
  live outside `work*`, so the round cannot touch them.
* **stage-A symlinks** — all 12268 `ql_evt*/icluster-*.npz` links in the four
  `grp0825` roots resolve to sibling `evt<ID>/` dirs, none into `g<K>/`, so the
  group-scratch prune cannot dangle one.

### 9.3 Freezing the gate reference before it is deleted

This round retires `work-pr112i-snapD2-*`, and those arms are **the only
per-event reference at the current operating point** — `work-*-prod0823` is
pre-flip (sec 2).  Retiring an arm once its gate has run is this tree's normal
pattern, but that pattern assumes the gate stays re-checkable, and a gate whose
reference has been deleted is not.  Deleting them plainly would make sec 8.1's
PASS unrepeatable forever.

`scripts/retire/hash_manifest_20260825.py` freezes the reference side into
`state-20260825/hashes/<arm>.tsv` (git-tracked).  It covers **both halves** of
the gate, because neither tool subsumes the other:

* archives — the exact rollup `pr85_hash_gate.py` compares;
* `tracking-pr.root` — a sha256 over the canonical JSON of
  `pr94_root_gate.py`'s own `load()`, i.e. every tree, every branch, every
  entry.  **Not** a sha256 of the file: ROOT embeds write timestamps in its
  keys, so a raw file hash would report a difference that is not one (M2).

Validated both directions before use, since a manifest that cannot fail is
worthless:

| check | result |
|---|---|
| round-trip on identical arms (`snapD2-ncpi0` vs `ncpi0-prod0825`) | 57/57 products match, ROOT included |
| revert-proof against the pre-flip `work-ncpi0-prod0823` | **differs**, 16 product lines — the rollup is not vacuous |

The revert-proof doubles as an independent corroboration of sec 2: the pre-flip
arm is visibly not the current operating point.

`retire_20260825.sh` gained **interlock 4**, which refuses the round if a cited
manifest is missing or has fewer `tracking-pr.root` rows than the arm has
events — so the freeze cannot be skipped by accident.

### 9.4 Product-class census

`archive_records_20260825.py` carries HEAVY unchanged from the 08-23 fork, and
this round that is *measured* rather than argued: across all 66 arms (72.6 GiB)
**zero** unclassified file exceeds 5 MiB, so nothing heavy can slip into the
record tar.

| class | disposition | size |
|---|---|---:|
| pctree | dropped | 47.15 GiB |
| calib | dropped | 8.11 GiB |
| mabc | dropped | 5.03 GiB |
| tracking | dropped | 3.14 GiB |
| record | **archived** | 8.20 GiB raw |

The record layer is dominated by 20891 `.wct-cfg-evt<N>.json` at 266 KB each —
the compiled config each arm actually ran, i.e. its operating point.  That is
the thing worth keeping, and it compresses hard.
