# Doc pr/14 — Cathode bundle rescue: reuniting cross-bundle cathode crossers split by the flash-reco absorbing window

**Status: DESIGN (this commit).** No toolkit code or config is changed yet; the
algorithm below is committed before implementation so the design decisions and the
ground-truth validation set are on record. Implementation, gates and validation
results land in follow-up commits and are appended to this doc.

Owner request (2026-08-01): implement an **isolated, removable** post-QLMatching
patch that fixes the doc pr/12 §6 failure mode — a cathode-crossing track whose two
halves sit in *different* Q/L bundles because the flash reconstruction hid the true
flash on one TPC side — and demonstrate it on the hand-scanned event list, with a
no-regression sweep over the full sample.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# The failure-mode census this design targets (doc pr/12 §6):
python3 cathode_nu_census.py --root work-mcp1kall-cath01 --sample mcp1k \
    --scores docs/pr/11_scores-table.tsv --ql-root work-mcp1kall-d59k \
    --merge docs/pr/12_cathode-census.tsv

# The per-event bundle tables the hand scan used:
awk 'NR==1 || 1' work-mcp1kall-d59k/nusel_evt288952/nusel-evt288952.tsv

# (after implementation) QL job with the rescue enabled, one event:
# ./run_clus_matching.sh <evt> data work-mcp1kall-rescue01 --cathode-rescue
```

## 1. Symptom

Doc pr/12 §6: of the 12 neutrino candidates whose *charge* crosses the SBND cathode
while the fitted trajectory does not, **7 have the far half in a different Q/L
bundle**, matched to a different flash 2.3–12 µs away and outside the beam window.
`cathode_connect` (`cfg/pgrapher/experiment/sbnd/clus.jsonnet:376`) correctly refuses
to join them — its flash-time gate exists precisely to keep unrelated bundles apart —
and the beam-window gate (doc 56) then hands the PR chain only the in-beam half. The
join test in pr/12 §6 is decisive: every candidate whose halves ARE one Q/L cluster is
fitted across the cathode (45/45); every unjoined one is not (0/7).

## 2. Root cause — upstream, in the light reconstruction

The SBND flash reconstruction has a known defect: an early flash opens a readout
window of roughly 8 µs which can **absorb a later flash**. A beam interaction whose
track crosses the cathode scintillates in *both* drift volumes and should produce a
beam-coincident flash on each side. When an earlier cosmic flash on one side has
opened its window, that side's beam flash is swallowed: the charge half on that side
then gets Q/L-matched to a *different* flash (the absorbing early one, or another
nearby flash with an acceptable light pattern) — and the light pattern of that wrong
match can look perfectly good, so no matching-quality cut catches it.

The light reconstruction is not part of the toolkit, so the defect cannot be fixed at
its source here. When it is fixed upstream, this patch must be removable by flipping a
single flag — that isolation requirement shapes the whole design.

## 3. Ground truth — the owner's hand scan (8 moves in 7 events)

Sample: mcp1k data (runs 18255/18259) + nueCC48 (437699). "Source" is the flash the
cluster is matched to now; "destination" is where the hand scan says it belongs. The
moved cluster carries no true flash of its own (its real flash was absorbed), so it
**adopts the destination bundle's flash time**. Beam window = [0.2, 2.2] µs.

| event | moved cluster (x,y,z) | source flash (t0 µs, PE, side) | destination flash (t0 µs, PE, side) | direction | Δt0 dest−src (µs) |
|---|---|---|---|---|---|
| 288952 | 22 (0.9, −89.8, 146.6) | #15 (−4.323, 22057, TPC1) | #16 (+2.177, 15869, TPC0) — bundle of cluster 9 | into beam | +6.50 |
| 169824 | 11 (−65.3, −82.4, 100.9) | #12 (−3.141, 39432, TPC0) | #13 (+1.328, 14893, TPC1) — matching 13 | into beam | +4.47 |
| 56463 (a) | 14 (−14.0, 79.5, 240.3) | #17 (+1.185, 19350, TPC0) **beam** | #18 (+5.746, 28958, TPC1) | out of beam | +4.56 |
| 56463 (b) | 20 (27.2, 53.5, 296.7) | #18 (+5.746, 28958, TPC1) | #17 (+1.185, 19350, TPC0) **beam** | into beam | −4.56 |
| 59003 | 26 (3.4, −11.3, 351.1) | #27 (−0.750, 34273, TPC1) | #28 (+1.578, 6352, TPC0) | into beam | +2.33 |
| 437699 | 10 (9.6, 194.2, 350.0) | #11 (+1.593, 26013, TPC1) **beam** | #14 (+13.656, 23626, TPC0) | out of beam | +12.06 |
| 392200 | 14 (−32.8, −167.4, 434.9) | #16 (+0.683, 19656, TPC0) **beam** | #17 (+3.113, 28135, TPC1) | out of beam | +2.43 |
| 398690 | 11 (3.7, 26.1, 214.6) | #13 (+2.057, 43514, TPC1) **beam** | #12 (−5.132, 11091, TPC0) | out of beam | −7.19 |

Three structural facts the algorithm must respect:

1. **Both directions occur, 4/4.** Half the moves take the crosser *out of* the beam
   bundle — the crosser is really an out-of-time cosmic whose half leaked into the
   beam bundle, and the rescue *cleans* the neutrino candidate. The other half
   reunite the far half *into* the beam bundle.
2. **Event 56463 has two moves in opposite directions between the same two flashes.**
   The decision must be per crosser pair, not per bundle pair.
3. **The T0 separation is asymmetric.** Relative to the beam flash, the wrong flash
   is between −7.19 µs (earlier) and +12.06 µs (later). Owner decision: two knobs,
   `rescue_t0_early = 8 µs`, `rescue_t0_late = 13 µs` (the user's "±8 µs" widened on
   the late side because 437699 sits at +12.06).

## 4. Fix — design

### 4.1 Placement: QL all-APA pipeline, after `cathode_connect`, before `examine_bundles`

New ensemble visitor `ClusteringCathodeBundleRescue`
(`clus/src/clustering_cathode_bundle_rescue.cxx`), inserted in the SBND all-APA
pipeline (`sbnd/clus.jsonnet:364-385`) between `cathode_connect` (:376) and
`examine_bundles` (:384). This slot is forced, not chosen:

- **The per-bundle member structure only exists there.** `examine_bundles` collapses
  each flash-time bundle into one cluster before the pctree tarball is written, so the
  PR job never sees bundle members — the a/b/c/d size comparison (§4.3) is impossible
  downstream.
- **Unmerge survival.** The PR job's `unmerge_bundle` splits a flash-collapsed
  cluster back into members and keeps only `real_cluster_main` members together
  (`ClusteringUnmergeBundle.cxx:329-383`). A crosser survives it only because
  `cathode_connect` made the halves ONE member before the collapse. The rescue must
  therefore **merge the pair itself** (as cathode_connect does), not merely retime the
  far half and let `examine_bundles` group them.
- After `cathode_connect` so the generic and cathode merge passes have already
  assembled each half maximally within its own bundle (sizes are meaningful, and the
  rescue only ever sees pairs cathode_connect refused by flash group).
- **Flash-level information is alive there**: the merged root PC `opflash`
  (`gid/time/ch/pe/apa`, written `QLMatching.cxx:3750-3797`, carried by
  `root_pcs_to_merge: ['opflash']`) provides every flash's time and drift side.
  (`Grouping::flashes()` must NOT be used post-APA-merge: the per-side `flash` PC
  keeps only APA0's list and silently mis-resolves TPC1 ids.)

### 4.2 Candidate pairs

For every cluster K_beam with `flash >= 0`, `cluster_t0` in `[beam_window_low,
beam_window_high)` and a point within `cathode_x_cut` of the cathode, search every
K_far with `flash >= 0`, a different `matched_flash_gid`, and
`t0_far − t0_beam ∈ [−rescue_t0_early, +rescue_t0_late]`, whose flash is outside the
beam window (`require_far_out_of_beam`, default true). Geometric acceptance is the
`is_cathode_crossing_pair` test duplicated from `clustering_cathode_connect.cxx:245-518`
at the SBND operating point (`cathode_x_cut=5cm, drift_cut=8cm, min_length_short=2cm,
short_dir_len=25cm, conn_short_cut=30`), with one addition: K_far's coordinates are
evaluated under the **destination-T0 hypothesis** — its x is shifted by
Δt0 × drift_speed × face_dirx before *all* cuts (tip-x, drift, distance and the
close/far regime split), because its stored `x_t0cor` was computed with the wrong T0
(up to ~2 cm off at 12 µs).

### 4.3 Direction: which flash does the merged crosser adopt?

Inputs, per pair (bundle membership = shared `matched_flash_gid`, i.e. the whole
flash group, which may host several mains):

- a = length of K_beam, b = max length of the beam bundle's *other* members;
- c = length of K_far, d = max length of the far bundle's *other* members.

Starting rule (the owner's a/b/c/d sketch completed; the owner stated the hand scan
outranks this sketch, so the rule is validated against §3 and refined until it
reproduces all 8 moves or the disagreements are irreducible and reported):

- beam-dominant (a ≥ b) and far not dominant (c < d) → adopt the **beam** flash
  (far half moves in; the far bundle keeps its own charge d);
- far-dominant (c ≥ d) and beam not dominant (a < b) → adopt the **far** flash
  (beam half moves out; the beam bundle keeps its own charge b);
- both or neither dominant → adopt the **longer half's** flash (small merges to
  large, the owner's base rule).

Every decision is logged with event, gids, t0s, a/b/c/d and the verdict, so the
validation is a log grep.

### 4.4 Execution (order is load-bearing)

1. Snapshot by value: destination t0/flash/gid/lm_flag and both bundles' membership
   (`merge_clusters` destroys both input clusters; reading them afterwards is UB).
2. `merge_clusters` on the pair — same call and defaults as
   `clustering_cathode_connect.cxx:607` (carries `assoc_cluster_id/main` per-blob
   provenance identically).
3. On the fresh merged cluster: `set_cluster_t0(dest_t0)` + `flash` +
   `matched_flash_gid` + `lm_flag` from the snapshot (**after** the merge —
   `merge_clusters` re-derives the flash triple from the longest member,
   `ClusteringFuncs.cxx:370-380`, which would clobber a pre-merge write); then
   `flag_main_cluster=1`, `flag_associated_cluster=0` — the crosser is a bundle main
   by construction, and multi-main flash groups are legal.
4. **Source-bundle main repair**: if the source gid still has members but none
   carries `flag_main_cluster`, promote the longest remaining member. Without this
   the source bundle collapses main-less and becomes invisible to
   unmerge/TGM/STM/Neutrino in the PR job (`ClusteringUnmergeBundle.cxx:155`,
   `TaggerCheckNeutrino.cxx:296`).
5. `add_corrected_points` to re-materialize `x_t0cor/y_cor/z_cor` under the new T0
   (the `improvecluster_2.cxx:265-275` idiom: T0 first, then correct; no geometry
   access on the fresh cluster in between).
6. Re-enumerate everything (children, bundle map, t0 groups) before evaluating the
   next candidate pair — indices and pointers dangle after a merge. Pairs are
   processed in cluster-ident order (deterministic; `sort_clusters`' pointer
   tie-break is not acceptable in a state-mutating loop).

Downstream, `examine_bundles` collapses the destination bundle with the merged
crosser as ONE member, `real_cluster_main` marks it whole, and the PR job's
unmerge passes leave the crosser intact — the same invariant that protects
cathode_connect's crossers today (doc 53).

### 4.5 Isolation and removal

- One new self-contained `.cxx` (all helpers file-static; no shared-helper extraction
  from `clustering_cathode_connect.cxx`, which stays byte-identical — M10).
- One jsonnet factory (`cfg/pgrapher/common/clus.jsonnet`, duplicating the
  `cathode_connect` factory shape) and one gated list insertion in
  `sbnd/clus.jsonnet`, threaded as `cathode_rescue_on` **default false** through
  `clus_all_apa` → `all_apa()` → the `wct-clus-matching-perevt.jsonnet` TLA.
- Off ⇒ compiled config byte-identical (wcsonnet diff in the implementation commit).
  Removal when the light reco is fixed = flip the flag (or delete the three sites).

### 4.6 Accepted, documented staleness

- The op/flash Bee display, the flash `group` array and the lm/nusel calib JSON are
  produced **pre-pipeline** and keep the pre-rescue matching. For rescued events the
  op display's "matching: N" rows disagree with the final bundles by design.
- The switch_scope in/out-of-volume partition is not redone after the retime (≤
  ~2 cm x shift); `add_corrected_points` refreshes the arrays themselves.
- Design-review verdicts (all source-cited, session 2026-08-01): T0-copy grouping is
  exact and bundles can only shrink, never chain; nothing between :376 and :384
  clobbers the post-merge override; provenance carry identical to cathode_connect;
  scope/filter inherited by the merged cluster; opflash gid domain ==
  `matched_flash_gid` domain on SBND.

## 5. Planned verification (results in follow-up commits)

1. **Knob-off gates**: wcsonnet compiled-config diff empty; abtest pdhd+pdvd
   (`events.txt`), qlport uboone `ab_check.sh`, SBND manual `hash_archive.py` A/B on
   standard events; `wcdoctest-clus`; freshness proof before every gate.
2. **Demonstration**: QL + PR chains with the rescue ON for the 7 events into a new
   work tag; log decisions vs the 8 ground-truth moves (target 8/8);
   `cathode_nu_census.py` class flips to `spanned`; Bee sets for owner hand check.
3. **No-regression** (owner decision: full scope): QL stage over all 1000 mcp1k
   events + nueCC48 with the rescue ON; non-firing events byte-identical to the OFF
   arm (member-content hashes); every firing listed and inspected; PR re-run and
   score comparison for firing events only.
4. Determinism: repeat-run identity on at least one firing event.

## 6. Prototype provenance

None — this is a new algorithm, not a WCP port. There is no prototype counterpart to
cite; the failure mode itself is documented in doc pr/12 §6 and the flash-reco
absorbing-window defect is the owner's diagnosis from the SBND light reconstruction.
