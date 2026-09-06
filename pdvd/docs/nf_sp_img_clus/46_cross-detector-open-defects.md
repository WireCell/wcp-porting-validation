# 46 — Six defects found in the docs of 2026-09-04/05 and never closed: which detector still runs each one, and what closing it costs

**Status (2026-09-05). Survey and work order. No code is changed, no config is
changed, no arm is run, and therefore no A/B gate is owed by this document and
none is claimed.** Every claim below is either re-verified against the current
source (tagged **[verified]**, at toolkit `d398ca14` / wcp-porting-img
`c7833b22`) or carried forward from the document that found it and *not*
re-checked (tagged **[doc-only]**). Nothing here is fixed by this round; §5 is
the ranked work order and §6 says what each fix would cost in byte-identity
terms.

**Round 2, same day (§8, §9).** On the owner's instruction — *"Go ahead with
`Array::operator=(Array&&)`, yes, please measure `proj_skip_unmapped_face`
before the flip. We do not need the cache fix for SBND, we should leave it."* —
§3.3 is **FIXED** (toolkit `76f47614`, byte-identical, doctest + negative
control, §9), §2.2 is **MEASURED** on 231 events across the three exposed
detectors and **fires zero times** (§8; no flip taken, that is still the
owner's), and §2.3 is **CLOSED as a decision, not a fix** (§2.3.1). Rounds 1
and 2 change no detector's production output.

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

Round 2 (§8, §9). Full command list with pins and controls in
`pdvd/stm/gates/d46_proj_skip_census.txt`; the short form:

```bash
# --- sec 9: the Array fix ---------------------------------------------------
wcbuild && ./build/util/wcdoctest-util -tc="point cloud array move assignment*"
for pkg in util aux clus; do ./build/$pkg/wcdoctest-$pkg; echo "$pkg rc=$?"; done
# negative control: same assertions, edited header, PRE-fix installed .so
g++ -std=c++17 -DSPDLOG_FMT_EXTERNAL -o negctl negctl.cxx -I $T/util/inc -I local/include \
    -L local/lib -lWireCellUtil -lfmt -ljsoncpp && LD_LIBRARY_PATH=$PWD/local/lib ./negctl

# --- sec 8: the proj_skip_unmapped_face census ------------------------------
PIN=/home/xqian/tmp/d46_libpin
# PDHD (30 evt, STM chain)
(cd $W/pdhd && for d in work/029107_*_stmwc; do e=${d#work/029107_}; e=${e%_stmwc}; \
   mkdir -p work/029107_${e}_d46skip && ln -sfn $PWD/$d/pctree-evt*.tar.gz $PWD/$d/pctree-evt*.tlas work/029107_${e}_d46skip/; done
 LD_LIBRARY_PATH=$PIN PDHD_MAX_JOBS=10 \
   PDHD_PR_TLA="-A trackfitting_config=$PWD/stm/pdhd_track_fitting_d46skip.json" \
   ./run_pr_evt.sh -s d46skip -stm -stm-fit 029107 all)
# SBND cosmic chain (99 evt) and full PR chain (48+19 evt)
(cd $W/sbnd/sbnd_xin/stm_campaign && NJOBS=10 D42_LIBPIN=$PIN \
   SBND_TRACKFIT_JSON=$PWD/sbnd_track_fitting_d46skip.json ./run_d42_stmfit.sh d46skip)
(cd $W/sbnd/sbnd_xin && for set in nuecc48:data ncpi0:sim; do s=${set%%:*}; r=${set#*:}; \
   LD_LIBRARY_PATH=$PIN PR_JOBS=8 \
   SBND_TRACKFIT_JSON=$PWD/stm_campaign/sbnd_track_fitting_d46skip.json \
   ./run_pr_chain_batch.sh work-$s-d97fv work-d46sbnd-on-$s $r; done)
# uBooNE (35 evt); UB_TRACKFIT_JSON is new in run_one.sh
(cd $W/qlport && for i in $(seq 0 34); do LD_LIBRARY_PATH=$PIN \
   UB_TRACKFIT_JSON=$PWD/uboone_track_fitting_d46skip.json ./scripts/run_one.sh $i d46skip_ub; done)

# the count (zero everywhere; the same grep on a PDVD d45skipon arm is the control)
grep -h "proj_skip_unmapped_face" $W/pdhd/work/029107_*_d46skip/wct_pr_*.log \
     $W/sbnd/sbnd_xin/work-stmcamp-d46skip/nusel_evt*/*.log \
     $W/sbnd/sbnd_xin/work-d46sbnd-on-*/pr_evt*/wct_pr_evt*.log \
     $W/qlport/scripts/sweep/d46skip_ub/*_*/wct_5384_*.log | wc -l
# positive control: which loader read my file
grep -rh "Failed to set parameter" $W/pdhd/work/029107_0_d46probe \
     $W/sbnd/sbnd_xin/work-stmcamp-d46probe $W/qlport/scripts/sweep/d46probe_ub | sort -u
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
| §2.2 `end()` deref on an unmapped face | `proj_skip_unmapped_face` | **ON** (`protodunevd/pdvd_track_fitting.json`) | OFF; **0 fires / 30 evt** (§8) | OFF; **0 fires / 166 evt** (§8) | OFF; **0 fires / 35 evt** (§8) |
| §2.3 shadow-cluster stale `ClusterCache` | `bad_blob_max_run` | **ON** (20) | **ON** (20, `pdhd/wct-pr-perevt.jsonnet:142`) | **OFF, exposed** | **OFF, exposed** |
| §3.1 `cluster_fc_check` round-2 write-back | *(none)* | exposed | exposed | exposed | exposed |
| §3.2 sentinel-T0 leak into the Bee dump | *(none)* | exposed | exposed | exposed | exposed |
| §3.3 `Array` move-assignment | *(none needed)* | **FIXED `76f47614`** — no live caller, byte-identical on every detector (§9) | | | |

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

> **Measured, 2026-09-05 — see §8.** 231 knob-ON events across the three
> detectors (PDHD 30, SBND 166 over both chains, uBooNE 35) and **the condition
> never fires**. Because the counter only increments inside the knob-on branch,
> zero fires means those arms are byte-identical to knob-OFF *by construction*,
> so no OFF partner was owed and none was run. No default was flipped.

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
the entanglement was deliberate, not an oversight. Two ways out were put to the
owner: **(a)** bind `bad_blob_max_run` on SBND / uBooNE and grade the
run-bounded filter there the way doc 40 §15.9–15.11 graded it on PDVD, closing
the cache defect as a side effect of a decision about the filter; or **(b)**
split the invalidation into its own default-OFF knob so the cache fix could be
graded alone, on the legacy vote, without adopting the run bound.

### 2.3.1 Owner decision, 2026-09-05: leave it

> *"We do not need the cache fix for SBND, we should leave it."*

**Neither (a) nor (b) is taken. `clus/src/improvecluster_1.cxx` is untouched and
SBND and uBooNE keep the legacy path.** This item is closed as a decision, not
as a fix, and should not be re-opened by a later round without new evidence.

What that decision accepts, stated so it is on the record: on SBND and uBooNE
the retiler's anti-ghost filter is a no-op from the second (apa, face) onward on
any cluster spanning more than one face. Nobody has counted how many SBND
clusters that is — the PDVD number is 80 of 493 on 039252/2, and SBND's two-APA
geometry makes multi-face clusters at least as common. The evidence that would
re-open it is a demonstration that ghost blobs surviving on second faces are
costing SBND STM or PR verdicts; a knob binding alone is not that evidence.

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

### 3.3 `Array::operator=(Array&&)` — FIXED, toolkit `76f47614`

*Found in doc 28 §12 ("left as is, noted for the owner"); fixed 2026-09-05 on
the owner's "go ahead". The verification is §9.*

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
failed before the fix. This is the only one of the six that could be closed as a
plain bug fix, and it is now closed — §9.

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

Found while writing §9's test and **not fixed** (it is an API shape, not a
defect, and changing it would touch every `Array` construction site):
`Array x{Array(v)}` does **not** select the move constructor. The
`template<typename ElementType> Array(std::initializer_list<ElementType>)`
overload (`PointCloudArray.h:104-107`) wins with `ElementType = Array`, so the
result is a silent 1-element array *of Arrays* — `size_major() == 1`, no
warning. `Array x(Array(v))` is worse: it is a function declaration. The only
spelling that means what it looks like is a named temporary plus
`std::move`. **[verified — it cost one build cycle]**

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

> **2026-09-05, doc 47:** PDVD's `Wire_ind` = 5.0 has no bearing on the fit's transverse constant — the filter is already off in wire space (kernel 0.21 mm) and the simulated constant (1.29 mm) is the deconvolution's impact-position residual; the data excess (2.30 mm) is not in the simulation.

---

## 5. Work order, ranked

1. ~~**§3.3 `Array` move-assignment.**~~ **DONE**, toolkit `76f47614` (§9).
2. ~~**§2.2 measure `proj_skip_unmapped_face`.**~~ **DONE**, §8: 0 fires on 231
   events. What is left is the *flip*, which is the owner's and which this
   measurement makes free-but-pointless on these manifests; §8.4 states the case
   both ways.
3. **§3.1 count how often `cluster_fc_check` round 2 runs.** A log census on an
   arm already on disk. That number decides whether R5 is a two-line fix worth a
   knob or a curiosity. **This is now the top open item.**
4. **§2.1 `kine_dqdx_skip_zero_dx` on SBND**, with the NaN → finite census and
   the selection rows. Then uBooNE separately, with the SBND evidence in hand.
   The §8 arms show what this costs: an SBND PR-chain arm is 67 events and ran
   in about ten minutes.
5. ~~**§2.3.**~~ **CLOSED by the owner, 2026-09-05** — leave it (§2.3.1).
6. **§3.2 sentinel-T0 Bee filter.** Cheap, gated by Bee member hashes, display
   only. Last because nothing physics-facing depends on it.

## 6. Byte-identity status of each fix, in one table

| item | knob needed? | knob-off byte-identical? | changes output when on |
|---|---|---|---|
| §2.1 | exists | yes (key absent) | yes — NaN becomes finite; selection rows can move |
| §2.2 | exists | yes (key absent) | only where it fires; 0 fires on PDHD/SBND/uBooNE (§8), and on PDVD 0 of 120 events' products moved |
| §2.3 | exists, but entangled | yes (key absent) | yes — and it also swaps the removal rule; **closed, not taken** (§2.3.1) |
| §3.1 | **new** (threaded from two taggers) | yes if default-OFF | yes, on clusters where round 2 decides |
| §3.2 | **new** | yes if default-OFF | yes, Bee layer only |
| §3.3 | **none** | n/a — no live caller | nothing, on any detector — **shipped `76f47614`** |

## 7. Files

**Round 1:** this document only — no script, no figure, no arm, no config, no C++.

**Round 2:**
* toolkit `76f47614` — `util/src/PointCloudArray.cxx` (the swap),
  `util/inc/WireCellUtil/PointCloudArray.h` (`owns_bytes()`),
  `util/test/doctest_pointcloud.cxx` (the case).
* `pdvd/stm/gates/d46_proj_skip_census.txt` — the §8 record: pin, controls,
  arm commands, per-detector counts.
* A/B copies of the three runtime parameter files, canonical files untouched:
  `pdhd/stm/pdhd_track_fitting_d46skip.json`,
  `sbnd/sbnd_xin/stm_campaign/sbnd_track_fitting_d46skip.json`,
  `qlport/uboone_track_fitting_d46skip.json`, plus the `*_d46probe.json`
  bogus-key copies used for the positive control.
* `qlport/scripts/run_one.sh` — new `UB_TRACKFIT_JSON` override (the uBooNE job
  hard-codes the file's basename, so the override is the symlink target).
* Arms on disk (fresh tags, none committed): `pdhd/work/029107_*_d46skip`,
  `sbnd_xin/work-stmcamp-d46skip`, `sbnd_xin/work-d46sbnd-on-{nuecc48,ncpi0}`,
  `qlport/scripts/sweep/d46skip_ub`, and the three `*d46probe*` probe dirs.

Sources: doc 45 §5.4 / §8 / §13, doc 40 §8 / §15.2, doc 32 §7 R5 / §16 / §20,
doc 28 §12 / §16 / §20, doc 39 §7 / §14, doc 12 §11.5, doc pdhd
`stm-tagger-chain.md` §4 and `01_steiner-wrapped-planes.md` §4.

---

## 8. Round 2a — measuring `proj_skip_unmapped_face` on the three exposed detectors

Owner, 2026-09-05: *"please measure `proj_skip_unmapped_face` before the flip."*
**No default is flipped by this section and no canonical config is touched.**
Full record with every command, pin and control:
`pdvd/stm/gates/d46_proj_skip_census.txt`.

### 8.1 What makes a null result meaningful here

The counter `n_unmapped_skipped` is incremented **only inside** the knob-on
branch (`TrackFitting.cxx:10008`) and the WARN is guarded on it (`:10035`). So a
knob-ON arm with **zero** WARNs took, at every path point, exactly the branch the
legacy path takes: that arm is byte-identical to a knob-OFF arm **by
construction**, and no OFF partner is owed. An OFF partner *would* be owed for
any detector that fired. None did, so none was run.

Three checks, because a count of zero is worthless if the file was not read, the
message could not reach the log, or the glob was empty:

* **Was my file read, and by which loader?** `load_trackfitting_config()` exists
  twice — `TaggerCheckSTM.cxx:1035` and `TaggerCheckNeutrino.cxx:3979` — and each
  catches an unknown key and prints `<ClassName>: Failed to set parameter …`. A
  probe copy carrying a deliberately bogus key was run on one event per chain:
  PDHD and SBND's cosmic chain answered **`TaggerCheckSTM`**, uBooNE answered
  **`TaggerCheckNeutrino`**. Both loaders are therefore covered, and each read
  the copy at my path rather than the canonical file.
* **Can the WARN reach a log at all?** The identical string is present in the
  PDVD arms where the knob fires, at the same runner and log level —
  `pdvd/work/039253_16_d45skipon`: *"skipped 61 of 497 … (apa=2 face=0)"*, and
  `039252_7_d45skipon`: *"skipped 5 of 712 … (apa=3 face=1)"*. A zero is a real
  zero.
* **Did the arms actually produce logs to grep?** An empty glob greps to zero
  exactly like a clean one. Log counts match event counts on all four arms
  (30 / 99 / 67 / 35), none is empty, and every one of SBND's 99 cosmic logs
  mentions `TaggerCheckSTM`, i.e. the tagger ran in each.

Binary pin `/home/xqian/tmp/d46_libpin` (`libWireCellClus.so
e3304cb9b362cd680ee3182452751a96`) = toolkit `d398ca14` plus the §9 `util` fix,
now `76f47614`; the two peer commits that landed during the arms are cfg-only.

### 8.2 The arms and the count

All knob-ON, all fresh tags, all on the pin.

| detector | chain / loader exercised | events | **fires** | points skipped | products |
|---|---|---|---|---|---|
| PDHD | STM (`TaggerCheckSTM`) | 30 | **0** | 0 | 30/30 rc=0, 30 `mabc-pr.zip`, 30 `tracking-stm.root`, 30 non-empty logs, 0 error lines |
| SBND | cosmic STM/TGM/FC (`TaggerCheckSTM`) | 99 | **0** | 0 | 99/99 rc=0, 99 `mabc-pr.zip`, 99 non-empty logs (28 017 lines), all 99 mention `TaggerCheckSTM` |
| SBND | full PR chain (`TaggerCheckNeutrino`) | 48 nueCC48 + 19 NCpi0 | **0** | 0 | 67/67 rc=0, 67 `mabc-pr.zip`, 67 non-empty logs |
| uBooNE | PR chain (`TaggerCheckNeutrino`) | 35 (all of `qlport/filelist`) | **0** | 0 | 35/35 rc=0, 35 `mabc_<idx>.zip`, 35 non-empty logs |
| **total** | | **231** | **0** | **0** | |
| *PDVD, doc 45 §13, for contrast* | both | *120* | *2* | *66* | *products identical 120/120* |

### 8.3 How far the null actually reaches

This is a statement about these manifests, not about these detectors. PDVD fires
on 2 of 120 events = 1.7 %. The 95 % upper limit on the rate from a null
observation is 3/N — and the four arms must be kept **separate**, because they
exercise two different loaders on four different event samples fitting different
objects, so pooling them would assume the very thing that is unknown:

| arm | loader | events | 95 % UL on the per-event rate | excludes PDVD's 1.7 %? |
|---|---|---|---|---|
| SBND cosmic | `TaggerCheckSTM` | 99 | 3.0 % | no |
| SBND full PR | `TaggerCheckNeutrino` | 67 | 4.5 % | no |
| uBooNE PR | `TaggerCheckNeutrino` | 35 | 8.6 % | no |
| PDHD STM | `TaggerCheckSTM` | 30 | 10 % | no |

**So no arm here excludes a PDVD-like rate.** What the measurement establishes is
that the condition is not common on any of these chains — not that it cannot
happen on them. An earlier draft of this section pooled SBND's two chains into
166 events and read the resulting 1.8 % as "SBND is measured"; that pooling is
withdrawn. A later round that wants the stronger claim needs more events, not a
re-reading of these.

### 8.4 Why PDVD is the odd one out — and what to do

**Inference, not measured here.** Both PDVD fires are on **gap-bridging**
rough-path points. Cluster 37 of 039253/16 has two charge components 28 cm apart
and its rough path crosses the gap at y ≈ 1 cm — the CRP boundary — where
`contained_by()` returns apa 2 face 0, a volume with no wire planes. The
condition needs a path point that lies outside *every* face carrying data in that
event, which a two-drift CRP geometry with internal gaps produces routinely and a
single-drift APA geometry largely does not. Note that a face can be missing from
the table for either of two reasons — it is not in the detector's geometry at
all, or it simply carries no data in *this* event (doc 45 §13: PDVD's second
fire, apa 3 face 1 on 039252/7, is a face that exists in other events' tables).

**This is stated as a falsifiable prediction, not as a result.** PDHD has four
anodes × two faces of which only one face per anode images
(`pdhd/pr.jsonnet:14`), so the naive expectation was that PDHD would fire on its
non-imaging faces; it did not, on any of 30 events. The prediction that explains
that is: *the trajectory never leaves the imaged volume, because there is no
internal gap for the rough path to bridge.* A single future PDHD WARN naming a
non-imaging face would refute it, and the WARN prints `apa=` and `face=` exactly
so that check is one grep. Nothing here proves the negative.

**The flip is the owner's and is not taken here.** The case for it: it converts
an undefined read (`std::get<0>` on `end()`) into a logged skip, at a cost these
231 events measure as exactly zero. The case against: it buys nothing observable
on any of the three manifests today, and one more key in three runtime files is
a real cost in a tree doc 77 is trying to shrink. If PDHD or uBooNE is flipped on
this evidence, the honest wording is *"free on 30 / 35 events"*, not *"free"*.

## 9. Round 2b — §3.3 fixed: `Array` move-assignment (toolkit `76f47614`)

**The fix.** `std::swap(m_store, rhs.m_store)` added to
`Array::operator=(Array&&)` (`util/src/PointCloudArray.cxx:107-124`), so the
store travels with the span. `std::vector`'s swap preserves the buffer address,
so the swapped span stays valid; the sharing case (empty store, external span)
is unaffected.

**Byte-identical, and not by gate — by construction.** A tree-wide search finds
no move-assignment into an `Array` lvalue on any code path, so no detector's
output can move. **No knob, no A/B arm, and none is claimed.**

**Verification.**

* `wcbuild` rc=0; freshness proof (`local/lib/libWireCellUtil.so` 19:02 vs the
  source edits at 19:00).
* `wcdoctest-util` 277 cases / 42 587 assertions, `wcdoctest-aux` 22 / 110 738,
  `wcdoctest-clus` 313 / 22 720 — all pass. Only `util` is touched; `aux` and
  `clus` are run because `Array` is the point-tree spine.
* New case *"point cloud array move assignment keeps the store with the span"*
  pins three things: an owning source (destination owns after the move, source
  is left empty, destination still readable after the source is destroyed), a
  sharing source (stays sharing), and the move constructor.
* **Negative control.** The same assertions compiled against the edited header
  but linked to the *pre-fix* installed `libWireCellUtil.so` report
  `dst.owns_bytes() = 0` and `src.owns_bytes() = 1` — the two values the new
  case asserts against. The test discriminates; it is not vacuous.

**One addition to the public API**, kept as small as it can be:
`Array::owns_bytes()` (inline, const, `PointCloudArray.h`) exposes the invariant
the `m_store` member already documents — *"if sharing user data, m_store is
empty"* — so a test can assert ownership without reading freed memory. Without
it the only observable difference is a use-after-free, which is not something to
pin a permanent doctest on.
