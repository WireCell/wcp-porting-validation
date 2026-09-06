# 46 — Six defects found in the docs of 2026-09-04/05 and never closed: which detector still runs each one, and what closing it costs

**Status (2026-09-05). Survey and work order. No code is changed, no config is
changed, no arm is run, and therefore no A/B gate is owed by this document and
none is claimed.** Every claim below is either re-verified against the current
source (tagged **[verified]**, at toolkit `d398ca14` / wcp-porting-img
`c7833b22`) or carried forward from the document that found it and *not*
re-checked (tagged **[doc-only]**). Nothing here is fixed by this round; §5 is
the ranked work order and §6 says what each fix would cost in byte-identity
terms.

**Read §1.1 first: the premise this document was commissioned on is wrong for
two of the six items.** The owner's instruction was *"we should fix this for
SBND and PDHD"* for the three knobbed defects of §2. Re-checking the per-detector
bindings rather than the C++ defaults shows PDHD already carries the fix for two
of them, and the exposed detectors are SBND and uBooNE. Only
`proj_skip_unmapped_face` (§2.2) is open on all three.

---

## 0. Repro

Every number and every per-detector cell below comes from one of these. They are
greps and reads; none of them runs reconstruction.

```bash
cd /home/xqian/toolkit-dev            # toolkit d398ca14, wcp-porting-img c7833b22
T=toolkit; W=wcp-porting-img

# --- sec 1.1, the per-detector matrix -------------------------------------
# A: kine_dqdx_skip_zero_dx -- who binds the TaggerCheckNeutrino key
grep -rn "kine_dqdx_skip_zero_dx" $T/cfg $W/pdvd/wct-pr-perevt.jsonnet \
        $W/pdhd/wct-pr-perevt.jsonnet $W/qlport/uboone-mabc.jsonnet
#   -> only pdvd/wct-pr-perevt.jsonnet:3025 (TLA) and :3787 (key-suppressed bind)
# ... and whether PDHD even runs the consumer:
grep -n "tagger_check_neutrino is not in" $T/cfg/pgrapher/experiment/pdhd/pr.jsonnet   # :509 :542
sed -n '18p' $W/pdhd/wct-pr-perevt.jsonnet                                              # "scope is UP TO THE STM TAGGER"

# B: proj_skip_unmapped_face -- which runtime track-fitting JSON carries it
ls $T/cfg/pgrapher/experiment/*/*track_fitting*.json
grep -l "proj_skip_unmapped_face" $T/cfg/pgrapher/experiment/*/*.json
#   -> protodunevd/pdvd_track_fitting.json ONLY (pdhd_ and sbnd_ do not; uBooNE
#      reads uboone_track_fitting.json, qlport/uboone-mabc.jsonnet:1452)

# C: bad_blob_max_run -- the knob that also carries the cache invalidation
grep -rn "retile_bad_blob_max_run *=" $W/pdvd/wct-pr-perevt.jsonnet $W/pdhd/wct-pr-perevt.jsonnet
grep -rn "bad_blob" $T/cfg/pgrapher/experiment/sbnd/clus.jsonnet $W/qlport/uboone-mabc.jsonnet
grep -n "bad_blob_max_run" $T/cfg/pgrapher/common/clus.jsonnet          # :1421 default null, :1433 key-suppressed
#   -> pdvd 20, pdhd 20, sbnd (no match), uboone (no match)

# --- sec 2-4, the defects themselves ---------------------------------------
sed -n '2533,2555p'  $T/clus/src/PRSegmentFunctions.cxx      # A: cal_kine_dQdx + the guard
sed -n '10000,10040p' $T/clus/src/TrackFitting.cxx           # B: the 2nd-pass loop + the knob
sed -n '629,650p'    $T/clus/src/improvecluster_1.cxx        # C: invalidate_cache gated on the run bound
sed -n '246,325p'    $T/clus/src/Clustering_Util.cxx         # D: round 1 writes back, round 2 does not
sed -n '3450,3460p'  $T/clus/src/TaggerCheckSTM.cxx          # D: the consumer
sed -n '1345,1355p'  $T/match/src/QLMatching.cxx             # E: set_cluster_t0(-1e12)
grep -n "get_cluster_t0" $T/clus/src/MultiAlgBlobClustering.cxx   # E: no matches == no sentinel guard
sed -n '46,54p;107,116p' $T/util/src/PointCloudArray.cxx     # F: move ctor correct, move assign not
grep -rnE "^\s*[A-Za-z_][A-Za-z0-9_.\[\]()*>-]*\s*=\s*std::move\(" \
     --include=*.cxx --include=*.h $T | grep -iE "arr|array"   # F: no live caller (empty)
```

---

## 1. What this document is

Six defects were named in the pdvd/pdhd documents of 2026-09-04 and 2026-09-05,
verified to exist, and left open. They fall into two classes, and the classes
have different costs:

* **§2 — knobbed but not bound.** A default-OFF C++ knob exists and is turned on
  in one or two detectors' configs. The other detectors run the defective path.
  Closing these is a *config* change per detector, plus the arm that grades it.
* **§3 — neither knobbed nor fixed.** Closing these is a *code* change. Five of
  the six change output where they fire and therefore need the default-OFF +
  gate treatment; **exactly one (§3.3) can be fixed outright**, because it has no
  live caller and the fix is byte-identical by construction.

### 1.1 The per-detector matrix — and the correction to the premise

Read down the columns, not across the rows: what matters is the *binding* in
each detector's driver, not the C++ default. **[verified]**

| defect | knob | PDVD | PDHD | SBND | uBooNE |
|---|---|---|---|---|---|
| §2.1 `cal_kine_dQdx` 0/0 NaN | `kine_dqdx_skip_zero_dx` | **ON** (`pdvd/wct-pr-perevt.jsonnet:3025`) | *no exposure* — `TaggerCheckNeutrino` is not in `pipeline_names` | **OFF, exposed** | **OFF, exposed** |
| §2.2 `end()` deref on an unmapped face | `proj_skip_unmapped_face` | **ON** (`protodunevd/pdvd_track_fitting.json`) | **OFF, exposed** | **OFF, exposed** | **OFF, exposed** |
| §2.3 shadow-cluster stale `ClusterCache` | `bad_blob_max_run` | **ON** (20) | **ON** (20, `pdhd/wct-pr-perevt.jsonnet:142`) | **OFF, exposed** | **OFF, exposed** |
| §3.1 `cluster_fc_check` round-2 write-back | *(none)* | exposed | exposed | exposed | exposed |
| §3.2 sentinel-T0 leak into the Bee dump | *(none)* | exposed | exposed | exposed | exposed |
| §3.3 `Array` move-assignment | *(none)* | latent, no live caller on any detector | | | |

Two corrections to the reading that commissioned this document:

1. **PDHD already carries §2.3.** `pdhd/wct-pr-perevt.jsonnet:142` has had
   `retile_bad_blob_max_run = 20` since the chain was built (`e6fa217a`); it is a
   duplication fork of PDVD's driver and inherited PDVD's whole tuned point. The
   exposed detectors are SBND and uBooNE.
2. **PDHD has no exposure to §2.1 today**, because it does not run the consumer:
   `pdhd/wct-pr-perevt.jsonnet:18` scopes the chain "UP TO THE STM TAGGER" and
   `pdhd/pr.jsonnet:509,542` record that `tagger_check_neutrino` is not in
   `pipeline_names`. This is a **condition, not an exemption** — the day PDHD
   turns the PR tail on, the key has to go in with it. The exposed detectors
   today are SBND and uBooNE.

**§2.2 is the one item where "SBND and PDHD" is exactly right**, and it is also
the only one of the three that is undefined behaviour rather than a wrong number.

---

## 2. Knobbed, but only PDVD (and for one, PDHD) is bound

### 2.1 `cal_kine_dQdx` divides by a zero `dx` and the NaN is never caught

*Found in doc 45 §5.4, guarded in doc 45 §8.*

**Symptom.** `kine_reco_Enu` is NaN on candidates whose muon chain contains a
coincident pair of fit points. On PDVD it was 10 of 560 candidates before the
doc-45 exclusion-frame flip and 72 after, because the restored trajectories reach
the condition seven times more often.

**Root cause** **[verified]**, `clus/src/PRSegmentFunctions.cxx:2533`. The
function sums `recomb_model->dE(dQ, dx)` over every fit of the chain. The
prototype carries `dx + 1e-9` (`ProtoSegment.cxx:1316`); the toolkit port dropped
the epsilon, so `PracticalBoxRecombination::dE(dQ, 0)` is 0/0 = NaN and the NaN
survives the sum. A negative `dx` additionally flips the sign of the 50 MeV/cm
clamp. The divergence is recorded in `clus/docs/porting/porting_dictionary.md`.

**Why it hides.** Downstream the value is read through positive-value gates —
K3's `apply_hadronic_dqdx_best` tests `dqdx > 0` — and a NaN fails every one of
them silently, so `kine_best` simply stayed 0 and no error was ever logged.

**The fix that exists.** `skip_zero_dx` on `cal_kine_dQdx`
(`PRSegmentFunctions.cxx:2549`, `if (skip_zero_dx && dx <= 0) continue;`),
threaded through `KineChargeOptions::dqdx_skip_zero_dx` and 24 callers, exposed
as the `TaggerCheckNeutrino` config key `kine_dqdx_skip_zero_dx`. C++ default
`false`.

**What closing it on SBND / uBooNE means.** One key. Both detectors already have
the passthrough: SBND via `tcn_knobs` (`sbnd/clus.jsonnet:1895` →
`knobs=tcn_knobs + {…}` at `:2360`), uBooNE via the same `knobs={}` argument of
`tagger_check_neutrino` (`pgrapher/common/clus.jsonnet:531`). It is **not**
byte-identical where it fires — the point of the knob is to turn a NaN into a
finite number — so each detector owes a knob-on arm and a census of how many
candidates change from NaN to finite, and of what that does to the selection
rows. On PDVD the equivalent census was 73 NaN → 0 with six finite `Enu` values
appearing through K3.

**Note for uBooNE.** uBooNE is a frozen reference chain. A NaN → finite change
moves its published numbers, so the SBND arm should be run first and the uBooNE
decision taken separately with that evidence in hand.

### 2.2 `do_single_tracking` dereferences `wpid_offsets.end()` — the only one open on all three

*Found and fixed behind a knob in doc 45 §13.*

**Symptom.** On PDVD 039253/16 the STM fit dump differed between *every* pair of
arms of that event, on 61 of 6877 rows of `T_rec_charge.pt`, by ≤ 2e-309 —
denormal, run-to-run garbage.

**Root cause** **[verified]**, `clus/src/TrackFitting.cxx:10000-10040`. The
fitter's geometry table (`wpid_offsets` / `wpid_slopes`) is built from the
(apa, face) pairs present in the grouping. `IDetectorVolumes::contained_by()` can
place a path point in a face that is *not* in that table — on 039253/16 the
gap-bridging points of cluster 37 land in apa 2 face 0, a volume with no wire
planes. The second-pass projection loop of `do_single_tracking` then executed
`std::get<0>(offset_it->second)` on `end()`. **This is undefined behaviour, not a
wrong number**: all four projections came out as denormals of order 2⁻¹⁰¹⁸ that
differ run to run, the dQ/dx fit found no cells for those points, and they kept
their 50000 e initial guess. The sibling loop in `trajectory_fit` already guards
the same miss with a WARN and a skip; this one did not.

**The fix that exists.** `TrackFitting::Parameters::proj_skip_unmapped_face`
(`TrackFitting.h:244`, default `0` = legacy), read from the runtime track-fitting
JSON. When > 0 the point is skipped, path and projection vectors stay aligned,
and the call WARNs once with the count, the face and the first point
(`TrackFitting.cxx:10037`).

**What closing it on PDHD / SBND / uBooNE means.** One key in each detector's
runtime parameter file — `pgrapher/experiment/pdhd/pdhd_track_fitting.json`,
`pgrapher/experiment/sbnd/sbnd_track_fitting.json`, and uBooNE's
`uboone_track_fitting.json` (`qlport/uboone-mabc.jsonnet:1452`). **[verified]**
none of the three carries it today.

**How much it should cost: probably nothing, and that is measurable in advance.**
On PDVD the knob fired on 2 of 120 events and left **every production product
identical 120/120** on both chains; only the fit dump of the one wobbling event
changed, and it became reproducible. The same measurement is available on the
other detectors before any flip: run a knob-on arm and read the WARN count. If it
is zero the flip is free and provable; if it is non-zero, those events were
already reading past `end()` and the products they produced were never
trustworthy.

**Note.** One runtime parameter file feeds **both** `TaggerCheckSTM` and
`TaggerCheckNeutrino` (`pgrapher/common/clus.jsonnet:307` and `:536`), so the key
lands on both taggers at once and both need to be in the gate.

### 2.3 The shadow cluster's `ClusterCache` is never invalidated — and the fix is entangled with a second change

*Found in doc 40 §15.2.*

**Symptom.** The retiler's anti-ghost filter removes nothing from the second face
onward on any cluster that spans more than one (apa, face). On PDVD 039252/2, 80
of 493 retiled clusters span more than one face and **not one** had a second-face
`BADBLOB` line in the knob-OFF arm.

**Root cause** **[verified]**, `clus/src/improvecluster_1.cxx:629-650`. The
caller inserts one face's blobs, calls `remove_bad_blobs`, removes some, inserts
the next face's blobs and calls again. `Mixins::Cached` has no child-change hook
and `Cluster::on_insert` / `on_remove` clear only the `sv3d` memo, so
`time_blob_map()` and `npoints()` are filled on first use and never refreshed.
From the second face on, the cached map has no entry for the face being filtered,
`all_new_blobs` is empty, and the function returns nothing.

**Why this one is not "flip a knob."** The `invalidate_cache()` at
`improvecluster_1.cxx:648` is guarded by `if (m_bad_blob_max_run > 0)`, and the
same predicate routes the whole function into `remove_bad_blobs_runs`
(`:706`) — **a different removal rule** (run-bounded, with same-slice adjacency
added). So today, on SBND and uBooNE:

> *"fix the stale cache"* and *"adopt PDVD's run-bounded anti-ghost filter"* are
> the same config change, and they are not the same decision.

Doc 40 §15.2 is explicit that invalidating the cache *unconditionally* changes
legacy output for every multi-face cluster, which is stop-and-ask territory — so
the entanglement was deliberate, not an oversight. Two ways out, both the owner's
to pick, neither taken here:

* **(a) Bind `bad_blob_max_run` on SBND / uBooNE** and grade the run-bounded
  filter on those detectors the way doc 40 §15.9-15.11 graded it on PDVD. This
  closes the cache defect as a side effect of a decision about the filter.
* **(b) Split the invalidation into its own default-OFF knob** so the cache fix
  can be graded alone, on the legacy vote, without adopting the run bound. This
  is the smaller experiment and the cleaner attribution, at the cost of one more
  knob in a file doc 77 is already trying to shrink.

**Exposure.** SBND builds its Steiner retiler with `cm.steiner(retiler=improve2)`
(`sbnd/clus.jsonnet:2071,2088`) and never passes `bad_blob_max_run`; uBooNE the
same (`qlport/uboone-mabc.jsonnet:1263`). The builder key-suppresses when the
argument is null (`pgrapher/common/clus.jsonnet:1421,1433`), so the C++ default 0
applies and both run the legacy path. **[verified]**

---

## 3. Neither knobbed nor fixed

### 3.1 `cluster_fc_check` computes its round-2 boundary points and throws them away

*Found in doc 32 §7 R5 / §16 R5; called "latent" there.*

**Root cause** **[verified]**, `clus/src/Clustering_Util.cxx:246-325`. Round 1
(`flag_cosmic=true`) writes its pair into `result.boundary_first` /
`boundary_second` at `:260-263`. Round 2 runs only when round 1 found no exit
candidate; it computes `bp1_r2` / `bp2_r2` at `:301-319`, uses them for
`do_boundary_check` and `update_boundary_set`, and **never assigns them to the
result**. Whatever round 1 put there survives.

**It is not purely latent.** The consumer is the STM tagger's own gate:
`clus/src/TaggerCheckSTM.cxx:3455-3456` reads
`fc_result.boundary_first/second` in `check_stm_conditions` and carries them into
the exit-endpoint logic that anchors the STM fit. So on exactly the clusters
where round 2 is what produced the decision, the fit is anchored on a pair the
round-2 test never validated. (`TaggerCheckFC` calls the same function but reads
only `is_fc`, so the STM tagger is the sole exposed consumer. **[verified]**)

**One correction to doc 32.** §20 item 3 promoted R5 from "latent" to "on the
critical path for any large pitch fraction". That promotion is **stale**:
`good_point_pitch_frac` was retired in `38245d18` in favour of the anisotropic
ctpc metric (doc 36 §11), so the pitch-fraction argument no longer applies. R5
stands on its own merits — the consumer above — not on that one.

**What closing it needs.** The fix is two assignments. The knob is the awkward
part: `Facade::cluster_fc_check` takes `(cluster, dv, fiducial, tolerance)` and no
config object, so a default-OFF flag has to be threaded in as a parameter from
each caller — `TaggerCheckSTM` and `TaggerCheckFC` have their own configs and
would each need a key. **The number nobody has, and the one that sizes this
whole item: how often round 2 runs at all** (i.e. how often round 1 finds no exit
candidate). That is a log census on an arm already on disk, not a new run, and it
should be the first thing done.

### 3.2 Clusters that never matched a flash leak into the Bee dump 1 480 km off-detector

*Found in doc 40 §8.*

**Root cause** **[verified]**. `match/src/QLMatching.cxx:1351` stamps every
cluster `set_cluster_t0(-1e12)` in an init pass and overwrites it only for
flash-matched clusters. A cluster that never matches keeps the sentinel, and
`x_t0cor = x_raw − dirx·(t0+offset)·v_drift` puts it at |x| ≈ 1.48 × 10⁸ cm. On
PDVD 039252/2 that is **225 live points in 29 clusters** (sizes 1–20 points,
median 6; 161 positive / 64 negative — the two drift directions), and no cluster
is mixed: each affected cluster moves entirely.

The `clustering` Bee layer is written by
`MultiAlgBlobClustering::fill_bee_points_from_cluster`
(`clus/src/MultiAlgBlobClustering.cxx:2884`) in whatever scope the layer's config
names, and **there is no sentinel guard anywhere in that file** — `get_cluster_t0`
does not appear in it at all. **[verified]** The same phenomenon is already
documented inside `clus` for SBND
(`clustering_cathode_bundle_rescue.cxx:981`, *"materialized with the sentinel T0
(56463: x_t0cor off by 1.5e6 m)"*), so this is a known class with no filter at
the display end.

**Severity, stated honestly.** This is **not** a reconstruction error — the
reconstruction never used `x_t0cor` for these clusters. It is a display leak plus
a trap: the affected clusters are invisible in Bee (drawn off-detector), and any
offline consumer that takes the layer's bounding box gets a meaningless one. It
is on every detector.

**What closing it needs.** A default-OFF knob on the Bee point dump that either
drops sentinel-T0 clusters or falls back to raw `x` for them. The gate is the Bee
zip member hashes, which must be identical with the knob off and *will* differ
with it on — so this one is cheap to gate and cheap to grade.

### 3.3 `Array::operator=(Array&&)` — the one item that can be fixed outright

*Found in doc 28 §12 ("left as is, noted for the owner").*

**Root cause** **[verified]**, `util/src/PointCloudArray.cxx:107-116`. The move
assignment does `clear()` and then swaps `m_shape`, `m_ele_size`, `m_dtype`,
`m_bytes` and `m_metadata` — but **not `m_store`**
(`util/inc/WireCellUtil/PointCloudArray.h:391,394`). If the source owned its data,
the destination ends up with `m_bytes` pointing into the *source's* `m_store` and
its own store empty, i.e. it looks like a sharing array and dangles the moment the
source dies; the source is left with the buffer and an empty span. The move
*constructor* at `:46-54` is correct — it moves `m_store`, and a `std::vector`
move preserves the buffer pointer, so `m_bytes` stays valid — and that is what
`Dataset::add` uses (`util/src/PointCloudDataset.cxx:104`,
`std::make_shared<Array>(std::move(arr))`).

**Why this one is different from the other five.** A tree-wide search for a
move-assignment into an `Array` lvalue finds **nothing** (the grep is in §0).
There is no live caller, so adding the missing `m_store` swap **cannot change any
output on any detector — it is byte-identical by construction**. It therefore
needs **no knob and no A/B arm**: a doctest that move-assigns an owning `Array`,
destroys the source and reads the destination is the whole verification, and it
fails today. This is the only one of the six that can be closed as a plain bug
fix.

---

## 4. Explicitly not in this document

Checked while building §1.1 and **dropped, because they are already closed** —
recorded here so nobody re-derives them:

* `detect_proton`'s missing `max_bin == -1` fallback (doc 28 §20 names it as the
  sibling site with no guard). **Fixed**: `clus/src/TaggerCheckSTM.cxx:1991-1996`
  carries the same `min(max_num, L.size()-1)` fallback as the eval site at
  `:2728-2736`. **[verified]**
* The inverted anode/cathode distance in `TaggerCheckSTM::dist_to_anode`. The
  `anode_dist_fix` knob is **on for both PDVD and PDHD**
  (`protodunevd/pr.jsonnet:386`, `pdhd/pr.jsonnet:379`). **[verified]**
* `RetileCluster::remove_bad_blobs` (`clus/src/retile_cluster.cxx:498`) carries
  the same stale-cache hole as §2.3 with no knob at all, and a commented-out
  `// shad_cluster.clear_cache();` at `:707`. **It is dead code**: every detector
  passes `retiler=improve2` explicitly and no config instantiates
  `ClusteringRetile` (both `cm.retile(...)` call sites are commented out,
  `protodunevd/clus.jsonnet:308`, `pdhd/clus.jsonnet:271`). Worth one line only
  because `CreateSteinerGraph`'s `retiler` default is still the string
  `"RetileCluster"` — a trap for a config that omits the key. **[verified]**
* The hard-coded 4000 e in `NeutrinoSteinerGapGraph.cxx:170` is inert at the
  default `sgp_weak_scale = 0`. **[verified]**

Deliberately **out of scope**, because they are deferred design or tuning
questions rather than defects, and each has a home document: the uncapped
component bridge (`connect_graph.cxx:20-26`, doc 39 §14, doc 40 P2); fused-cluster
splitting (doc 39 §6); the `steiner_pc` Bee branch's hard-coded 4000 e shading
(`MultiAlgBlobClustering.cxx:2971`, doc 39 §7, display only); the
`ClusteringUnmergeBundle` per-cluster warning (doc 39 §14, cosmetic); PDHD's
`params.jsonnet` inheriting generic `lar.DL/DT/lifetime` (doc pdhd
`stm-tagger-chain.md` §4, an owner decision because PDHD *sim* consumes them);
and PDVD's `Wire_ind` filter width and stale Wiener-tight set (doc 12 §11.5,
production SP constants).

---

## 5. Work order, ranked

1. **§3.3 `Array` move-assignment.** No knob, no arm, byte-identical by
   construction, one doctest. Do it first because it is the only free one.
2. **§2.2 `proj_skip_unmapped_face` on PDHD, SBND and uBooNE.** Run a knob-on arm
   per detector and read the WARN count *before* proposing any flip. On PDVD the
   answer was 2 events of 120 and zero production change; if the other detectors
   answer zero, the flip is free and provable. This is the item where the owner's
   "SBND and PDHD" reading is exactly right, and it is the only UB of the six.
3. **§3.1 count how often `cluster_fc_check` round 2 runs.** A log census on an
   arm already on disk. That number decides whether R5 is a two-line fix worth a
   knob or a curiosity.
4. **§2.1 `kine_dqdx_skip_zero_dx` on SBND**, with the NaN → finite census and
   the selection rows. Then uBooNE separately, with the SBND evidence in hand.
5. **§2.3** — put option (a) vs (b) of §2.3 to the owner *before* running
   anything, because the two answer different questions and only one of them is
   about the cache.
6. **§3.2 sentinel-T0 Bee filter.** Cheap, gated by Bee member hashes, display
   only. Last because nothing physics-facing depends on it.

## 6. Byte-identity status of each fix, in one table

| item | knob needed? | knob-off byte-identical? | changes output when on |
|---|---|---|---|
| §2.1 | exists | yes (key absent) | yes — NaN becomes finite; selection rows can move |
| §2.2 | exists | yes (key absent) | only where it fires; on PDVD 0 of 120 events' products moved |
| §2.3 | exists, but see (a)/(b) | yes (key absent) | yes — and it also swaps the removal rule |
| §3.1 | **new** (threaded from two taggers) | yes if default-OFF | yes, on clusters where round 2 decides |
| §3.2 | **new** | yes if default-OFF | yes, Bee layer only |
| §3.3 | **none** | n/a — no live caller | nothing, on any detector |

## 7. Files

This document only. No script, no figure, no arm, no config, no C++.

Sources: doc 45 §5.4 / §8 / §13, doc 40 §8 / §15.2, doc 32 §7 R5 / §16 / §20,
doc 28 §12 / §16 / §20, doc 39 §7 / §14, doc 12 §11.5, doc pdhd
`stm-tagger-chain.md` §4 and `01_steiner-wrapped-planes.md` §4.
