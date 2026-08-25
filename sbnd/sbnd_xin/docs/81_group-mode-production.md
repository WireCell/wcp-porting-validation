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

The campaign's own products supersede four families of study arms.  Run
2026-08-25 with the owner's go-ahead, after all ten asserts, the archive
integrity gate and a full dry run had passed.

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


### 9.5 Executed

```
prune_group_scratch.sh   CONFIRM=yes   20 G      rc=0
archive_records_20260825.py            66/66     rc=0
retire_20260825.sh A     CONFIRM=yes   66 dirs   rc=0
```

| | before | after |
|---|---:|---:|
| `sbnd_xin` | 236 G | **144 G** |
| `work*` dirs | 104 | **38** (= `len(KEEP)`) |
| `/nfs/data/1` free | 622 G | **714 G** |
| broken symlinks | 0 | **0** |

Post-deletion checks all clean: `refused=0`, dangling-link repair
`repaired=0 unresolved=0`, **no git-tracked file deleted**, survivor census 38
== `len(KEEP)`, removal manifest 66 rows at
`scripts/retire/state-20260825/removed.tsv`.  The record layer went to
`archive/records/prod0825-groupmode-20260825/` — 8.4 GiB raw compressed to
1.4 G, integrity PASS 66/66 (tar members == manifest record files).

**The freeze was then exercised for real.**  With every `work-pr112i-snapD2-*`
arm deleted, re-running `hash_manifest_20260825.py` on the surviving
`work-<s>-prod0825` arms and diffing against the frozen manifests reproduces
sec 8.1 in full:

| sample | rollups re-verified against a deleted reference |
|---|---:|
| nueCC48 | 144/144 |
| NCpi0 | 57/57 |
| mcp1k | 3000/3000 |
| mcp2k | 6000/6000 |
| **total** | **9201/9201** |

So sec 8.1's PASS is still checkable by anyone, at any later date, from the
git-tracked manifests alone — which is what the tree's "report gates by label
so any PASS can be re-checked later" rule actually requires once the reference
arm is gone.

### 9.6 One layout claim, now tested

Sec 3 justifies the per-event layout partly by "a single event can be re-run in
place".  Stage B exercised the layout at scale (3067 events), but *that* claim
— the legacy per-event driver working out of a group-produced root — was
untested.  It is now:

```
SBND_IMGBASE=$PWD/work-mcp1k-grp0825 \
  scripts/multi/ql_legacy_gate.sh $PWD/work-mcp1k-grp0825 <fresh> 166650
```

`run_ql_evt.sh` re-runs event 166650 out of the `grp0825` root into a fresh
work root and reproduces the group's own Q/L output: `mabc-all-apa.zip`,
`mabc-apa0-face0.zip`, `mabc-apa1-face0.zip` and `pctree-evt166650.tar.gz` all
**SAME**, `rc=0`.  The layout is drop-in for the legacy driver, not merely for
stage B.

---

## 10. Round 2 — two stage-A runner defects, and the re-baseline they precede

Round 1 delivered `work-<s>-{grp,prod}0825` and gated them byte-identical to
the per-event production.  Doc 82 then shipped an **unknobbed** determinism fix
in `match/` (`95c10cd1`, `QLMatching::rescue_empty_flashes()` walking
`std::map<Opflash*,…>` in heap order), which by its own status flag "changes
reconstruction output on the formerly-bistable events".  Those arms are
therefore no longer reproducible at HEAD, and this round prepares their
replacement: fix the two stage-A runner defects first, then re-run.

**Nothing was re-run in this round.**  The campaign below is a plan, costed
from round 1's own `.time.meta`, not a result.

### Repro

```bash
cd wcp-porting-img/sbnd/sbnd_xin

# 1. the staleness, measured rather than inferred (~2 min, scratch only)
SBND_STAGE=$PWD/input_files_reco1/staged-mcp2025c-2nd-2000evt \
SBND_IMGBASE=$PWD/work-mcp2k-grp0825 \
  scripts/multi/ql_legacy_gate.sh $PWD/work-mcp2k-grp0825 <scratch> 53793 100002

# 2. the splitter round-trip + the fail-first coverage test (~10 s, scratch only)
python3 scripts/multi/merge_group_products.py work-mcp2k-grp0825 <s>/g0 100002 100032
python3 scripts/multi/split_group_products.py <s>/g0 <s>/out
printf '100002 100032 999999\n' > <s>/gbad/events.txt   # + the same archives
python3 scripts/multi/split_group_products.py <s>/gbad <s>/outbad     # must rc=1
```

Binary: toolkit `95c10cd1`, unchanged by this round — **no C++ was touched.**

### 10.1 `work-*-{grp,prod}0825` no longer reproduce at HEAD

Chain: round 1 gated `grp0825` byte-identical to `work-<s>-ql0819`
(24536/24536), and doc 82 §2d reports post-fix mcp2k 53793 differing from
`ql0819` on 12/12 draws.  Verified directly, with a control so the test is not
vacuous:

| event | products vs `work-mcp2k-grp0825` |
|---|---|
| 53793 (known bistable) | `mabc-*` 3/3 SAME, **`pctree-evt53793.tar.gz` DIFFER** |
| 100002 (control) | all four **SAME** |

`prod0825` reads that Q/L output, so it inherits the divergence.  Any round
that A/Bs against these arms now gets a mover its own change did not cause
(CLAUDE.md §5 rule 5).

**This supersedes the `WCT_QLRESCUE_CENSUS` recommendation.**  A census was the
right idea when the arms were staying; a re-baseline produces the same mover
list for free, as the diff of the new arms against the old ones, and is
strictly more informative.  Doc 82's own warning applies to reading the doc-81
§7.1 seven as "the unstable events": it is not that list.

### 10.2 The splitter was inflating every imaging npz 3.8x

**Symptom.**  Stage A is 66.6 G for 3067 events and 83 % of it is
`evt<ID>/icluster-apa{0,1}-{active,masked}.npz`.

**Root cause.**  `split_group_products.py` wrote them `ZIP_STORED`, on the
stated premise that "npz is uncompressed".  The premise is false: WCT's own
`ClusterFileSink` writes through custard's `miniz_sink`, which calls
`mz_zip_writer_add_mem(..., MZ_BEST_SPEED)` — **DEFLATE**.  So the legacy
per-event path has always compressed these, and only the group path's splitter
produced STORED ones.  Same event, same members:

| writer | `evt166650/icluster-apa0-active.npz` |
|---|---|
| `work-img-mcp1k` (legacy, WCT `ClusterFileSink`) | 2 886 252 B, DEFLATE |
| `work-mcp1k-grp0825` (group, this splitter) | 10 836 412 B, **STORED** |

Uncompressed member total is 10 834 864 B in both — the payloads were never in
question, only the container.

**Why it hid.**  Every stage-A gate in this tree hashes member *payloads*
(`hash_archive.py:19-30`, `stagea_gate.py:22-30` both `z.read(name)`), so a
STORED-vs-DEFLATE container difference is invisible to all of them — correctly,
since it is not a difference in the data.  The gate that would have caught it
is one nobody runs: comparing file sizes against the legacy hub.

**Fix.**  `ZIP_DEFLATED` (level 6, zipfile's default).  Not a knob: the member
payloads are bit-identical, so this cannot change reconstruction output or any
gate verdict — CLAUDE.md §1's knob rule does not bite.

**Verification.**

| check | result |
|---|---|
| round-trip, mcp2k 100002+100032 rebuilt via `merge_group_products.py` then re-split | **10/10** archives member-content identical to the recorded per-event products |
| size | 17 718 124 → 4 711 389 B (**26.6 %**) and 18 713 712 → 4 762 582 B (25.4 %) |
| WCT reads DEFLATE npz | Q/L re-run of mcp2k 100002 off a hand-compressed imaging dir: `mabc-all-apa.zip`, both per-APA zips and `pctree-evt100002.tar.gz` all **SAME**, rc=0 |
| levels measured | 1 → 38.4 %, **6 → 26.6 %**, 9 → 24.0 % (0.2 s/event at 6) |

Projected: ~55 G of imaging → ~14 G, i.e. stage A **66.6 G → ~26 G**.

### 10.3 Stage A had the coverage hole doc 82 part 4 closed in stage B

**Symptom.**  None yet — latent.  All four `grp0825` arms were audited this
round and are complete (48/19/1000/2000 events with both an imaging npz and a
Q/L pctree), so no delivered product is affected.

**Root cause.**  `split_npz` checked `set(per) - set(evts)` — members for an
event *not* in the group — and never the converse.  `split_opflash` had
neither check.  `link_imaging` skips a missing npz silently, `main` prints
"ok", and `run_chain_group.sh:299` only greps `^\[g$K\] ok`.  So a group that
produced nothing for one of its events split cleanly and reported success at
every level.  This is the stage-A twin of the `rc=0`-unconditional defect doc
82 part 4 fixed in `run_pr_chain_batch.sh:1811`.

**Why it hid.**  Round 1 counted coverage by hand from real per-event product
existence, exactly because of the stage-B defect — so the campaign was safe,
and the runner was never asked the question.

**Fix.**  Both splitters now refuse on `set(evts) - set(per)`, naming the
events.  `split_opflash` also gains the `unknown` check it never had.

**Verification** (revert-proof, on a 3-event `events.txt` over 2 events' worth
of archives):

| splitter | result |
|---|---|
| fixed | `no members for 1 of 3 group events: ['999999']`, **rc=1** |
| pre-fix, `git show HEAD:` | `24 members -> 2 events` … `ok`, **rc=0** |

### 10.4 The re-baseline campaign, planned

Fresh tag (M13 — never write into `grp0825`/`prod0825`): `work-<s>-grp0826`
and `work-<s>-prod0826`, `<s>` ∈ {nuecc48, ncpi0, mcp1k, mcp2k} = 3067 events.

```bash
# stage A -- from reco1, with the fixed splitter
for s in nuecc48 ncpi0 mcp1k mcp2k; do
  SBND_MAX_JOBS=8 ./run_chain_group.sh <reco1-$s.root> work-$s-grp0826 data \
      --size 16 --layout perevt
done      # ncpi0 adds --fsproduct '...__FILTERFRAMESHIFT.'; mcp2k part 2 --gbase 63

# stage B
for s in nuecc48 ncpi0 mcp1k mcp2k; do
  PR_GROUP_SIZE=16 PR_JOBS=16 PR_EXTRA_STAGES=pr_display \
      ./run_pr_chain_batch.sh work-$s-grp0826 work-$s-prod0826 data
done
```

**Full re-run from reco1, not `--from ql`.**  Only `match/` moved
(`git log 2aba11dc..HEAD -- img/ clus/ sio/` is empty), so re-running imaging
is strictly redundant — but round 5 pruned the group scratch, so `--from ql`
would first have to rebuild the group archives with
`merge_group_products.py`, a path never exercised in a campaign, and it would
leave the new arm's imaging inherited rather than produced.  The redundant
imaging costs ~35 min and buys a single-epoch from-source baseline.

**Sizing**, from round 1's own `.time.meta` (sum of per-group wall):

| stage | groups | Σ group-wall | peak RSS/job |
|---|---:|---:|---:|
| A imaging | 194 | 19 225 s | 1.05 G |
| A Q/L | 194 | 13 537 s | 0.98 G |
| B PR | 193 | 14 057 s | 1.49 G |

At `SBND_MAX_JOBS=8` / `PR_JOBS=16` that is ≈ 68 min + ≈ 15 min, plus the
reco1 dump, which has no `.time.meta` and must be measured on the first group.
M5 still governs: check `cut -d' ' -f1 /proc/loadavg` a few minutes in and back
off if it exceeds ncores — the job counts above are a starting point, not a
result.  Disk: ~26 G (A) + ~16 G (B) against 881 G free.

**`SBND_PRECOMPILE_CFG` is now DEFAULT ON** (owner decision), so the two
`run_chain_group.sh` invocations above need no env.  This was planned as OFF
one paragraph earlier in this round, on doc 82 part 3's ground that
precompiling changes the process's allocation history and part 2 had shown the
Q/L answer on a bistable event to be a function of exactly that — part 3's own
ON-vs-OFF test differed on 286191.  Doc 82 round 3 then removed that dependence
at the source (toolkit `95c10cd1`), so the condition part 3 attached — "needs
its own byte-identity gate on the standard manifest" — is **discharged**, not
waived.  Gate, at `95c10cd1`, 32 events over two samples covering **all seven**
known-bistable events plus controls:

| arm | vs `work-<s>-grp0825` |
|---|---|
| mcp1k, 16 evts (incl. 286191, 292643), OFF | **IDENTICAL 64/64** |
| mcp1k, 16 evts, **ON** | **IDENTICAL 64/64** |
| mcp2k, 16 evts (incl. 53793, 99438, 161043, 321101, 350816), OFF | 60/64 — 4 archives of **53793** only |
| mcp2k, 16 evts, **ON** | 60/64 — the **same** 4 archives of 53793, same first differing member |

| direct comparison | result |
|---|---|
| `PRECOMPILE=1` vs `PRECOMPILE=0`, all 32 events | **128/128 archives member-content identical** |

53793 is round 3's own documented mover (doc 82 §2d: it differs from `ql0819`
on 12/12 post-fix draws, its new fixed point), so the only event that moves is
one already on the record as moving, and it moves identically in both arms.
Buys 38 threads → 7 and closes doc pr/97's SIGSEGV hazard, which this was the
only driver still exposed to.  `SBND_PRECOMPILE_CFG=0` restores the old path.

**Scope limit, stated rather than glossed:** `precompile_cfg` is called at all
three stage-A steps (`run_chain_group.sh:194` dump, `:229` imaging, `:274`
Q/L), but `repro_ql_nondet.sh` runs `--from ql`, so the gate above covers
**Q/L only**.  Imaging determinism rests on §2's rep-A/rep-B control
(672/672).  Run **nueCC48 first as a pilot** and check its imaging against
`work-img-nuecc48` (48 events, ~2 min) before the remaining 3019 events start
— that is the imaging leg, and the campaign runs it anyway as a gate row.

**Gates.**  Three, plus one this round adds:

| gate | expectation |
|---|---|
| `verify_frozen_stagea_20260825b.py`-style rollup diff of `work-<s>-grp0826` against `state-20260825b/hashes/stagea-<s>.tsv` | Q/L movers only on formerly-bistable events; **imaging must be 100 % identical**.  NOTE: `stagea_gate.py --img/--ql` no longer applies — §11 retired both reference arms; the frozen manifest is the reference now, and `work-<s>-grp0825` is the on-disk one |
| `pr85_hash_gate.py` + `pr94_root_gate.py` vs `work-<s>-prod0825` | the mover list — the census, for free |
| `nusel-table.tsv` diff vs `prod0825` | the 72 torn rows (§8.2) should clear: doc 82 fixed the tear at `7822a440` |
| imaging as a **control** | no `img/` commit since `2aba11dc`, so any imaging difference means a stale binary, the wrong reco1 file or a wrong `--fsproduct` — stronger than an mtime freshness proof, and free |

**Do not retire `grp0825`/`prod0825` until that mover list is recorded.**
§9.3's own lesson: a gate whose reference has been deleted is not a gate.

---

## 11. Round 3 — retiring the stage-A reference side and the pre-flip PR baseline

Owner, 2026-08-25, naming the three-row candidate table §9 left standing:

> I assume we can safely retire [`work-img-{4 samples}`, `work-*-ql0819`,
> `work-*-prod0823`], and recover the disk

Twelve arms, 39 GiB removed, ~36 G net of the record archive. `sbnd_xin`
**144 G → 108 G**, `work*` dirs **38 → 26**.

### Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin

# 1. freeze BOTH reference sides (two tools -- see 11.2 for why not one)
python3 scripts/retire/hash_manifest_stagea_20260825b.py nuecc48 ncpi0 mcp1k mcp2k
python3 scripts/retire/hash_manifest_pr_20260825b.py \
        work-{nuecc48,ncpi0,mcp1k,mcp2k}-prod0823

# 2. prove the freeze is usable BEFORE deleting anything it describes
python3 scripts/retire/verify_frozen_stagea_20260825b.py nuecc48 ncpi0 mcp1k mcp2k
#   -> PASS 24536/24536

# 3. pre-flights, then the round
python3 scripts/retire/plan_20260825b.py            # 12 asserts, OVERALL: PASS
python3 scripts/retire/archive_records_20260825b.py # integrity PASS 12/12
./scripts/retire/retire_20260825b.sh A              # dry run
CONFIRM=yes ./scripts/retire/retire_20260825b.sh A

# 4. the same check again, with the reference arms now GONE
python3 scripts/retire/verify_frozen_stagea_20260825b.py nuecc48 ncpi0 mcp1k mcp2k
#   -> PASS 24536/24536
```

### 11.1 Why the three rows are not one decision

| row | size | why it may go | what it costs |
|---|---:|---|---|
| `work-img-{nuecc48,ncpi0,mcp1k,mcp2k}` | 19 G | §7 proved them byte-identical to the imaging half of `work-<s>-grp0825`, 24536/24536 | nothing, once the freeze exists |
| `work-<s>-ql0819` | 10.3 G | the Q/L half of that same gate | its `ql_evt*/calib-evt*.json` dumps (nueCC48 146 MB, NCpi0 53 MB) — `grp0825` does not carry them |
| `work-<s>-prod0823` | 10.2 G | explicitly PRE-flip (§4), superseded by `prod0825`; its revert-proof job was already exercised and the result is text in §4 | **docs pr/104–pr/111's A/B references against the pr/104 epoch become text-only** — this was that epoch's last on-disk carrier |

`work-img-{r1qlmc,r2mc}` are **not** in the removal set. No `grp0825` arm exists
for either sim sample, so they are the only copy, not a duplicate. They survive
by being named in `KEEP` — a glob over `work-img-*` would have swept them.

The `calib-evt` loss has a precedent the owner already accepted:
`thin_hubs_20260811.py` dropped mcp1k's and mcp2k's for the same reason in the
08-11 round (both arms held 0 today), and they are regenerated whenever the Q/L
step itself re-runs.

### 11.2 The round's actual hazard: a freeze that preserves nothing

Rows 1 and 2 are the **two halves of one gate reference**, and this round
deletes both. §7's "PASS 24536/24536" stops being re-checkable the moment that
happens — the failure §9.3's interlock 4 was invented to prevent, one layer up.

The trap is that the obvious tool does not work and **fails silently**.
`hash_manifest_20260825.py` selects events with `EVT_RE = r"pr_evt(\d+)$"`;
stage-A arms are laid out `evt<N>/` (imaging) and `ql_evt<N>/` (Q/L). Every
subdir would be skipped, the arm would get a header-only `.tsv`, and 08-25's
`[ -s "$f" ]` interlock would pass it — an M1-shaped vacuous PASS with 29 G
deleted behind it. Hence:

* a **second** freeze tool, `hash_manifest_stagea_20260825b.py`, which imports
  `stagea_gate.py`'s own `NPZ`/`QL` product lists rather than re-listing them,
  so the frozen manifest cannot drift from the definition the gate used;
* interlock 4 and ASSERT 11 check **row counts**, never `[ -s ]`: 8 products ×
  (48 + 19 + 1000 + 2000) events, which must total §7's own **24536**.

Worth recording because it is counter-intuitive: `.npz` **file sizes differ
wildly between the two arms and this is not a defect**. NCpi0 evt105946's
`icluster-apa0-active.npz` is 1 418 630 bytes in `work-img-ncpi0` and
5 760 756 bytes in `work-ncpi0-grp0825` — 4× — and the two are member-for-member
identical; only the container's compression differs. Anyone who checks this
with `ls -l` or `cmp` will conclude the arms diverged (M2).

### 11.3 The freeze, exercised against a deleted reference

Run before the deletion and again after it, against the surviving `grp0825`
arms:

| sample | rollups reproduced from the frozen manifest |
|---|---:|
| nueCC48 | 384/384 |
| NCpi0 | 152/152 |
| mcp1k | 8000/8000 |
| mcp2k | 16000/16000 |
| **total** | **24536/24536** |

The post-deletion run is the one that matters: `work-img-*` and `work-*-ql0819`
no longer exist, and §7's gate still reproduces in full from a git-tracked text
file. `work-<s>-prod0823` is frozen the same way (9201 rollups = 3 products ×
3067 events) — but read that one as **insurance, not gate preservation**:
`prod0825` sits at a different operating point, so there was never a
byte-identity claim between the two to keep checkable. What it buys is a future
revert-reproduction against toolkit `b5c9f43a`.

### 11.4 Two things the asserts caught that a hand-read would not have

**`work-probe178410a` was about to be silently broken.** ASSERT 4 (dangling-link
dry run) found its `evt178410/` was a *symlink* into `work-img-mcp2k`, with the
four `ql_evt178410/*.npz` linking through it. That arm is PROTECTED precisely
because it is the only on-disk proof mcp2k evt 178410's SIGSEGV is
non-deterministic, and a non-deterministic crash cannot be re-captured on
demand. The round would have left five broken links inside it, at delete time,
with no error. Fixed before deleting anything by replacing the link with the
bytes it pointed at (`cp -rL`); the arm grew 6.7 MB → 17 MB and now depends on
nothing outside itself.

**ASSERT 8 would have failed, correctly.** `work-vtx105-base-*` is PROTECTED and
stays, and its own per-event logs name `work-<s>-ql0819` as the Q/L root it
read. Rather than hand-suppress that, the assert gained a **SUCCESSOR** rule:
the substitution to `work-<s>-grp0825` is accepted only when the successor is
itself in `KEEP` *and* that sample's frozen manifest is complete. The
substitution is sound because the products are *proven* identical, not because
they are similar — and the assert now refuses a successor whose proof is
missing.

### 11.5 Repointed, acknowledged, or left alone

Three dispositions, applied deliberately rather than uniformly:

* **REPOINTED** — the two *live* tools, verified by a new ASSERT 12 and
  interlock 5: `scripts/multi/repro_ql_nondet.sh` (doc 82's reproducer, whose
  round closed the same day) and `scripts/multi/ql_legacy_gate.sh`. Doc 82's
  repro command #10 also passed `REF=work-mcp2k-ql0819` *explicitly*, which
  after this round is `exit 1`, not a stale comment; it and the §2/§4 result
  lines now name `grp0825`. **No number in doc 82 changes** — only the path.
* **ACKNOWLEDGED** — the five closed-round arm scripts (`pr107_arms.sh`,
  `pr108_testA.sh`, `pr109_sbnd_arms.sh`, `pr112_arms.sh`,
  `pr112_dual_arms.sh`) and the repro blocks of docs 71/74/76/77/78/79,
  pr/102, pr/108. A script that *records how a finished round was run* keeps
  naming the arm that round actually read; repointing it would claim a
  provenance the round did not have.
* **The standing substitution rule** for anyone re-running one of those:
  `work-<s>-ql0819` → `work-<s>-grp0825`, `work-img-<s>` → the same arm's
  `evt<N>/` layer. `grp0825`'s `ql_evt<N>/` carries every product `ql0819`'s
  did except `calib-evt*.json` and `wct_ql_evt*.log`.

### 11.6 Executed

```
hash_manifest_stagea_20260825b.py      24536 rows   rc=0
hash_manifest_pr_20260825b.py           9201 rows   rc=0
verify_frozen_stagea (pre-delete)   24536/24536     rc=0
plan_20260825b.py                   12 asserts      OVERALL: PASS
archive_records_20260825b.py        12/12           rc=0
retire_20260825b.sh A CONFIRM=yes   12 dirs         rc=0
verify_frozen_stagea (post-delete)  24536/24536     rc=0
```

| | before | after |
|---|---:|---:|
| `sbnd_xin` | 144 G | **108 G** |
| `work*` dirs | 38 | **26** (= `len(KEEP)`) |
| `/nfs/data/1` free | 878 G | **917 G** |
| broken symlinks | 0 | **0** |

Post-deletion checks clean: `refused=0`, dangling-link repair
`repaired=0 unresolved=0`, **no git-tracked file deleted**, survivor census 26
== `len(KEEP)`, removal manifest 12 rows at
`scripts/retire/state-20260825b/removed.tsv`.

The record layer went to `archive/records/stagea-refside-20260825b/` — 5077 MiB
raw compressed to **2.9 G**, integrity PASS 12/12. **`sp-frames.tar.bz2` is
preserved, not lost**: this was first written up as a cost and then checked, and
`archive_records`' `HEAVY` list has no pattern matching it, so all 2067 files
(mcp2k 2000, nueCC48 48, NCpi0 19) are in the imaging-hub tars verbatim. That
is also most of the 2.9 G, which is why the net recovery is ~36 G rather than
the 39 GiB the driver reports removed.

**This round was hygiene, not pressure relief.** `/nfs/data/1` stood at 75 %
with 878 G free before it ran. Nothing was at risk; ~36 G had simply stopped
earning its keep.

### 11.7 What this changes about `grp0825`

`work-<s>-grp0825` is now the **sole** on-disk carrier of stage A — imaging and
Q/L both — for all four data samples. There is no second copy of these products
anywhere in the tree; there is only the frozen manifest. A future round that
releases a `grp0825` arm is deleting the product itself, not one copy of two.
`PROTECTED.txt` now says so at that entry.

---

## 12. Round 4 — recompressing the imaging already on disk

Round 2 (§10.2) fixed the *writer*: `split_group_products.py` had been writing
every per-event `icluster-*.npz` `ZIP_STORED` on a false premise, where WCT's
own `ClusterFileSink` writes DEFLATE.  That fixes future runs.  The arms already
on disk were still 3.8x larger than they needed to be, and the owner asked for
both halves:

> if you can naturally store the [imaging npz] as compressed next time of
> running, and then also compress these and save space for the 0825 directories

### Repro

```bash
cd wcp-porting-img/sbnd/sbnd_xin
python3 scripts/multi/recompress_npz.py --dry-run work-mcp1k-grp0825
python3 scripts/multi/recompress_npz.py --jobs 8 \
        work-{nuecc48,ncpi0,mcp1k,mcp2k}-grp0825
```

### 12.1 Why an in-place rewrite needed to be paranoid

Round 3 (§11) retired `work-img-<s>`.  That was the right call and its freeze
is sound, but it changes the risk on *this* operation: **`grp0825` is now the
only copy of that imaging anywhere**, so a rewrite that corrupts a file has
nothing to fall back on.  `scripts/multi/recompress_npz.py` therefore never
edits in place in the naive sense.  Per file:

1. read the original; record the namelist **order** and `sha256` of every
   payload (order is load-bearing for a group archive, doc 76 round 2);
2. write a sibling `.recompress.tmp` with `ZIP_DEFLATED`, same order;
3. re-open the tmp and require **identical order and identical payload
   hashes** — any mismatch unlinks the tmp, leaves the original untouched and
   counts as an error;
4. `fsync`, then `os.replace` (atomic on one filesystem).

Already-DEFLATE files are skipped, so it is idempotent and safe to re-run after
an interruption.

### 12.2 One correction to §10.2

§10.2 said `hash_archive.py` *and* `stagea_gate.py` hash decompressed payloads,
so neither can see a container change.  Only the second half is right, and it
is the half that matters — `stagea_gate.py:31` dispatches on
`zipfile.is_zipfile(path)`, i.e. by content, and hashes `z.read(name)`.
`hash_archive.py:20` dispatches on the **`.zip` extension**, so an `.npz` falls
through to its tarfile branch and raises; it has never been usable on imaging
npz at all.  The conclusion is unchanged (the gate that compares imaging is
`stagea_gate.py`), but the first verification attempt of this round was vacuous
because of it — it compared two identical *error* lines and reported them equal.
Recorded because a check that cannot fail is worth less than no check.

### 12.3 Verification

| check | result |
|---|---|
| independent member compare, ncpi0 evt105946, all 4 npz | order + payload sha256 **IDENTICAL**; 5 760 756 → 1 288 608 B on the largest |
| `stagea_gate.members()` on the same 4, before vs after | **equal**, 4/4 |
| end-to-end: Q/L re-run off the **recompressed** `work-mcp2k-grp0825`, evt 100002 | `mabc-all-apa.zip`, both per-APA zips, `pctree-evt100002.tar.gz` all **SAME**, rc=0 |
| residual `STORED` after the run | **0 of 12 268** |
| stray `.recompress.tmp` | **0** |
| errors | **0** — `{'done': 12268}` |

### 12.4 Result

| | before | after |
|---|---:|---:|
| imaging npz, 4 arms, 12 268 files | 55.93 GiB | **14.70 GiB** |
| `sbnd_xin` | 108 G | **66 G** |
| `/nfs/data/1` free | 917 G | **958 G** |

41.23 GiB reclaimed with no product deleted and no reconstruction output
touched — only the containers changed, and every gate in this tree reads
through them.  Combined with round 3, `sbnd_xin` is 144 G → 66 G.

Note for the re-baseline (§10.4): `work-<s>-grp0826` will be written compressed
from the start by the round-2 splitter fix, so its stage A should land near
~26 G rather than 66.6 G, and this script is not needed for it.

---

## 13. Round 5 — the disk census, and the three things it found

Owner asked what dominates disk after round 4 and then to act on the answer.
Census of all 104 075 files: tar archives 19.37 GiB (29.6 %), imaging npz 14.81
(22.6 %), PR pctree 9.32, Q/L pctree 6.89, **PR `.groups` scratch 5.08**, the
rest under 3 % each.  Round 4 had knocked imaging off the top spot; what took
its place was `archive/records/` (16 G) plus the staged reco1 inputs (4.0 GiB).

### Repro

```bash
cd wcp-porting-img/sbnd/sbnd_xin
python3 scripts/multi/prune_pr_group_scratch.py \
        work-{nuecc48,ncpi0,mcp1k,mcp2k}-prod0825 \
        work-{nuecc48,ncpi0,mcp1k,mcp2k}-grp0825      # dry run; --apply to act
```
(the real call pairs them `<pr_root> <ql_root>`, four pairs.)

### 13.1 `.groups/` was 5.08 GiB of the group job's own input

`run_pr_chain_batch.sh:1692` builds `<pr_root>/.groups/g<N>.tar.gz` by merging
the group's per-event `<ql_root>/ql_evt<ID>/pctree-evt<ID>.tar.gz`.  It is the
**input** staged for one wire-cell process, not a product; nothing reads it
after the job, since `pr85_hash_gate.py`, `pr94_root_gate.py` and
`nusel_extract.py` all work off `pr_evt<ID>/`.  It is the stage-B twin of the
stage-A group scratch `prune_group_scratch.sh` already reclaims.

`scripts/multi/prune_pr_group_scratch.py` does not delete on that reasoning.
For each `g<N>.tar.gz` it rebuilds the member→sha256 map of the corresponding
`ql_evt<ID>` pctrees and requires **exact equality** — same names, same
payloads, nothing extra either side — before unlinking; anything failing, or
with a missing Q/L side, is kept and named.

| | result |
|---|---|
| verified duplicates | **193 / 193**, 0 kept, 0 mismatches |
| freed | **5.08 GiB** |
| preserved | every `g<N>.rc`, `.time.meta`, `-rse.json`, `-build.log` — the provenance |

### 13.2 A record archive was holding 2.44 GiB of frame data

`archive/records/stagea-refside-20260825b/imaging-hubs/work-img-mcp2k.tar.gz`
was 2.45 GiB where the mcp1k hub of the same round is 21 MB.  The difference is
2000 `sp-frames.tar.bz2`, and the manifests show it plainly: mcp1k archived
2001 record files / 306 MB, mcp2k **4000 / 2.92 GB**.

**Root cause.**  `archive_records_20260825b.py`'s `HEAVY` list (`:49-56`) has
eight patterns — `pctree.*\.tar\.gz`, `mabc.*\.zip`, `calib(-pr)?-evt.*\.json`,
`.*\.npz`, `clusters-apa.*\.tar\.gz`, `opflash_apa.*\.tar\.gz`,
`tracking-pr\.root`, `oc56scan-evt.*\.jsonl`.  **`sp-frames.tar.bz2` matches
none of them**, so frame data was classified `record` and archived.

**Why the guard did not catch it.**  The docstring's assurance is "a census of
all 66 removal arms finds ZERO unclassified file above 5 MiB, so nothing heavy
can slip into the record tar".  That check is **per file**, and each sp-frames
is ~1.25 MB.  2000 of them are 2.44 GiB and every one is under the threshold.
A per-file cap cannot see a many-small-files class.

**Action.**  Verified all 2000 against the staged reco1 input they were copied
from — `input_files_reco1/staged-mcp2025c-2nd-2000evt/e<N>/frames-dnn.tar.bz2`,
still on disk — **2000/2000 member-content identical, 0 mismatches**, then
repacked the tar without them: order and payloads of the 2000 kept log members
verified identical before the atomic replace.  **2.45 GiB → 0.020 GiB.**  The
drop is recorded, not silent: `work-img-mcp2k.dropped-members.tsv` lists all
2000 with their recovery path, and the manifest carries a `record-after-round5`
row plus the explanation.

**Not fixed here, and deliberately so.**  `archive_records_20260825b.py` is
round 3's script and has already produced a committed archive; editing it would
falsify the record of what that round ran.  The next fork should take both
halves of the fix:

```python
# HEAVY: name the class
("frames",  re.compile(r'^(sp-)?frames(-dnn)?.*\.tar\.(bz2|gz)$')),
# and stop relying on a per-file cap: refuse if any single unclassified
# basename PATTERN exceeds, say, 500 MiB in aggregate -- that catches the
# next unnamed heavy class without having to guess its name first.
```

### 13.3 The third candidate was wrong, and is retracted

§13's census listed `input_files/input-3files-lan-reco2` (1.1 G),
`input-10evt-mc` (471 M) and `bee/prod0819` + `bee/prod0813` (855 M) as
"stale-looking", on age alone.  Checked properly before touching anything:

| candidate | references | tracked |
|---|---:|---|
| `input_files/input-3files-lan-reco2` | 5, incl. `scripts/analysis/light/pmt_health_study.py:6` and `flash_t0_lan_reco2.py:20` as a hardcoded `BASE` | — |
| `input_files/input-10evt-mc` | 11, incl. `dump_truth_sed.C`, `flash_coincidence.py`, `check_dead5.py`, `saturation_pe.py` | — |
| `input_files/` + `bee/` overall | — | **238 git-tracked files** |

All of them have live consumers, and deleting them is exactly the ASSERT 10
failure this tree has already hit twice (`vtx_rules/baselines.py`,
`scripts/analysis/pr57/oc56_truth.py`).  **Nothing deleted.**  If that 3.5 G is
wanted back it needs the retirement-round treatment — an explicit
`ACK_BROKEN_REFS` decision per script — not a sweep.  Recorded because the
casual recommendation came first and the check came second, which is the wrong
order.

### 13.4 Result

| | before | after |
|---|---:|---:|
| `.groups` scratch | 5.08 GiB | **0** |
| `work-img-mcp2k.tar.gz` | 2.45 GiB | **0.020 GiB** |
| `archive/` | 16 G | **13 G** |
| `sbnd_xin` | 66 G | **59 G** |
| `/nfs/data/1` free | 958 G | **966 G** |

Across rounds 3–5, `sbnd_xin` is **144 G → 59 G** with no product lost: round 3
retired superseded arms behind frozen manifests, round 4 recompressed, and this
round removed only bytes proven to exist elsewhere.
