# Doc pr/14 — Cathode bundle rescue: reuniting cross-bundle cathode crossers split by the flash-reco absorbing window

**Status: IMPLEMENTED (toolkit `8d6de401`, default OFF) + DEMONSTRATED 8/8 (§7).**
The knob-off byte-identical gates and the full-1k no-regression sweep are the
remaining steps and are appended when done.

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

# --- §7 demonstration: the 7 hand-scan events, rescue ON, fresh tags ---
# toolkit 8d6de401 built + installed (wcbuild); freshness proof per CLAUDE.md.
# mcp1k (6 events; entries from the staged map):
TAG=rescue01 ENTRIES="227 809 599 411 453 490" SBND_CATHODE_RESCUE=1 \
    ./run_full1k_nusel.sh 1000 5           # -> work-mcp1kall-rescue01
# nueCC48 (437699 = idx 29):
mkdir -p work-nuecc48-rescue01 && ln -sfn $PWD/work/evt437699 work-nuecc48-rescue01/evt437699
SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-2025fall-48evt-fsprod \
SBND_WORK_ROOT=$PWD/work-nuecc48-rescue01 SBND_SAVE_ASSOC=1 SBND_CATHODE_RESCUE=1 \
    ./run_nusel_evt.sh data -chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm \
    -main-pair-real -fvx 2.5 -fvy 3 -stm-fit -mip 56000 -unmerge-assoc 29

# the rescue decisions (compare to the §3 table):
grep -h "rescue round" work-mcp1kall-rescue01/ql_evt*/wct_ql_evt*.log \
                       work-nuecc48-rescue01/ql_evt*/wct_ql_evt*.log

# --- full PR chain on the rescued pctrees + census ---
PR_JOBS=6 ./run_pr_chain_batch.sh work-mcp1kall-rescue01 work-mcp1kall-rescue01pr data \
    288952 169824 56463 59003 392200 398690
PR_JOBS=1 ./run_pr_chain_batch.sh work-nuecc48-rescue01 work-nuecc48-rescue01pr data 437699
python3 cathode_nu_census.py --root work-mcp1kall-rescue01pr --ql-root work-mcp1kall-rescue01 \
    --sample mcp1k --out /home/xqian/tmp/cbr_census_mcp.tsv
python3 cathode_nu_census.py --root work-nuecc48-rescue01pr --ql-root work-nuecc48-rescue01 \
    --sample nuecc48 --out /home/xqian/tmp/cbr_census_nuecc.tsv

# --- Bee set (§7.3) ---
python3 make_pr_bee.py -q work-mcp1kall-rescue01 -p work-mcp1kall-rescue01pr \
    -q work-nuecc48-rescue01 -p work-nuecc48-rescue01pr --allow-unevaluated \
    -o /home/xqian/tmp/cbr_rescue_after.zip 288952 169824 56463 59003 437699 392200 398690
./upload-to-bee.sh /home/xqian/tmp/cbr_rescue_after.zip

# --- §7.4 no-regression sweep: full 1000-event mcp1k + all 48 nueCC48, OFF vs ON ---
TAG=cbroff1k ./run_full1k_nusel.sh 1000 6                          # OFF arm
TAG=cbron1k SBND_CATHODE_RESCUE=1 ./run_full1k_nusel.sh 1000 6     # ON arm
# nueCC48 arms: the §7 run_nusel_evt.sh invocation with
# SBND_WORK_ROOT=$PWD/work-nuecc48-cbroff (resp. -cbron, + SBND_CATHODE_RESCUE=1)
# and 'all' instead of 29; imaging symlinked per event from work/evt<ID>.
# Per-event member-content hash OFF vs ON + firing census (verdict per event:
# IDENTICAL / FIRED / MISMATCH):
python3 cbr_sweep_compare.py --off work-mcp1kall-cbroff1k --on work-mcp1kall-cbron1k \
    --out /home/xqian/tmp/cbr_sweep_mcp1k.tsv
python3 cbr_sweep_compare.py --off work-nuecc48-cbroff --on work-nuecc48-cbron \
    --out /home/xqian/tmp/cbr_sweep_nuecc.tsv
# determinism (§5.3): two fresh ON re-runs of 437699 (idx 29) into
# work-cbr-det{1,2}, then 3-way hash_archive vs work-nuecc48-cbron.
# geometry trace of any unexpected firing (raw-stderr [cbrx] lines):
TAG=cbrdbg ENTRIES="472 692" SBND_CATHODE_RESCUE=1 CATHODE_RESCUE_DEBUG=1 \
    ./run_full1k_nusel.sh 1000 2
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

## 5. Verification

### 5.1 Knob-off byte-identical gates — ALL PASS

Both sides built and installed cleanly with freshness proofs
(A = pre-change `3fe65876`, libWireCellClus.so 351120832 B @ 11:25;
B = `8d6de401` knob-off, 355168784 B @ 11:20/11:28); `wcdoctest-clus`
565/565 on B.  Compiled-config proofs: the QL job (`wct-clus-matching-perevt`)
and the PR job (`wct-pr-perevt`) compile **byte-identical** JSON with the knob
off vs the pre-change tree; with `cathode_rescue=true` the compiled pipeline
carries `ClusteringCathodeBundleRescue:all` between `ClusteringCathodeConnect:all`
and `ClusteringExamineBundles:all` with the §4 knob values.

| gate | label pair | result |
|---|---|---|
| abtest pdhd+pdvd, `events.txt`, clus stage | `snap/cbroff_base_clus2` vs `snap/cbroff_new_clus` | `ab_compare.sh` **OVERALL: PASS** (every mabc zip content-identical) |
| qlport uBooNE 35-event manifest | `sweep/cbroff_new_ub` vs `sweep/cbroff_base_ub` | Gate 1 **ZIPS 35/35 content-identical**.  Gate 2 tagger-compare `identical=2 diff=33` = the documented non-discriminating A/A behavior (doc pr/2 §8: an A/A run also gives 33/35 DIFF from T_tagger vector-branch run-to-run noise; ZIPS is the gate) |
| SBND QL+PR chain, 3 events (288952/169824/56463) | `work-mcp1kall-cbroffA` (at `3fe65876`) vs `work-mcp1kall-cbroffB` (at `8d6de401`, knob off) | `hash_archive.py` member-content: `mabc-all-apa.zip`, `pctree-evt*.tar.gz`, `mabc-pr.zip` **IDENTICAL** ×3 events; `nusel-evt*.tsv` identical |

The abtest **img stage could not be re-run** (the PDHD/PDVD SP-frame inputs
were removed in the 2026-07-30 disk-retirement round); the clus stage — the
only stage a `clus/`-only change can affect — ran from the surviving imaging
tarballs on both sides.  A first A-side clus snapshot (`cbroff_base_clus`) was
discarded: the B-side install overlapped its tail, so its later per-event
stages may have loaded the new library (label kept, never compared —
`cbroff_base_clus2` is the clean A side).

### 5.2 No-regression sweep (owner decision: full scope) — §7.4

Full 1000-event mcp1k + all 48 nueCC48, OFF arm vs ON arm at the same
`8d6de401` binary; non-firing events must be byte-identical, every firing
listed.  Results in §7.4.

### 5.3 Determinism

Repeat-run identity on a firing event: §7.4.

## 7. Demonstration — the 7 hand-scan events, 8/8 moves reproduced

Arms `work-mcp1kall-rescue01`(+`pr`) and `work-nuecc48-rescue01`(+`pr`), toolkit
`8d6de401`, rescue ON, fresh tags (Repro above). **The starting §4.3 rule
reproduced all 8 hand-scanned moves on the first pass — no refinement was
needed.** The rescue log lines against the §3 truth table:

| event | rescue log (a/b/c/d in cm) | rule fired | direction vs hand scan |
|---|---|---|---|
| 288952 | c7 (gid 2, +2.177, 312.8) + c19 (gid 1000001, −4.323, 118.2) → gid 2; a=312.8 b=0 c=118.2 d=429.7 | beam-dominant | into beam ✓ |
| 169824 | c15 (gid 1000003, +1.328, 208.0) + c14 (gid 0, −3.141, 119.8) → gid 1000003; a=208.0 b=0 c=119.8 d=261.7 | beam-dominant | into beam ✓ |
| 56463 (a) | c12 (gid 6, +1.185, 169.1) + c21 (gid 1000000, +5.746, 234.3) → gid 1000000; a=169.1 b=199.2 c=234.3 d=332.3 | longer-half | out of beam ✓ |
| 56463 (b) | c14 (gid 6, +1.185, 199.2) + c23 (gid 1000000, +5.746, 332.3) → gid 6; a=199.2 b=1.6 c=332.3 d=403.4 | beam-dominant | into beam ✓ |
| 59003 | c4 (gid 10, +1.578, 144.4) + c29 (gid 1000001, −0.750, 155.2) → gid 10; a=144.4 b=109.4 c=155.2 d=388.3 | beam-dominant | into beam ✓ |
| 437699 | c12 (gid 1000001, +1.593, 25.1) + c7 (gid 2, +13.656, 364.3) → gid 2; a=25.1 b=118.8 c=364.3 d=3.1 | far-dominant | out of beam ✓ |
| 392200 | c16 (gid 0, +0.683, 104.5) + c25 (gid 1000001, +3.113, 227.3) → gid 1000001; a=104.5 b=31.7 c=227.3 d=1.0 | longer-half | out of beam ✓ |
| 398690 | c17 (gid 1000001, +2.057, 255.7) + c8 (gid 5, −5.132, 186.5) → gid 5; a=255.7 b=401.6 c=186.5 d=0.5 | far-dominant | out of beam ✓ |

Notes: 56463 needed both rounds and they went in opposite directions between
the same two flashes, as the hand scan says (the round-1 far bundle's d=403.4
is the round-0 merged crosser, correctly counted as bundle support).  The
longer-half tie-break decided two of the eight (56463a: neither dominant;
392200: both dominant) — both agree with the hand scan.

### 7.2 Downstream: what the PR chain now does (census on the rescued arms)

| event | pr/12 class (before) | after rescue | reading |
|---|---|---|---|
| 169824 | one-sided, far half unjoined | **spanned**, ql_joined=1 | fit crosses the cathode — recovered |
| 59003 | one-sided | **spanned**, ql_joined=1 | fit crosses the cathode — recovered |
| 56463 | one-sided | **eroded**, ql_joined=1; standard chain labels it **TGM** | halves are ONE cluster now; the fit still leaves a >4 cm gap at the cathode (the pr/12 §6 "joined eroded" tracking mode, cf. evt 406796). In the standard nusel chain the joined 546 cm in-beam crosser is TGM-tagged, so 56463 stops yielding a nu candidate — whether that is a correct background rejection or an efficiency loss needs the owner's truth check (§7.4) |
| 288952 | one-sided | joined crosser → **TGM=true** → no nu candidate | the "candidate" was half a cathode-crossing cosmic; whole, the TGM tagger correctly removes it (background rejection win) |
| 392200 | one-sided | crosser moved out; remaining 58 cm beam main is **STM**-tagged | beam bundle cleaned of the misassigned cosmic half; the standard chain then tags the leftover as a stopping muon (no nu candidate) |
| 398690 | one-sided | crosser moved out; candidate = 401.6 cm beam main, **no cathode contact** | ditto |
| 437699 | one-sided | crosser moved out; candidate = remaining beam charge, no cathode contact | ditto (nueCC48) |

All seven downstream outcomes are consistent with the hand-scan intent: the
four into-beam moves give the candidate its whole track (two now fitted
across, one joined-eroded, one revealed as TGM), and the four out-moves clean
the misassigned cosmic half out of the neutrino candidate's bundle.

### 7.3 Bee display (owner hand check)

**After (rescue ON, PR layers + img/clustering/op):**
<https://www.phy.bnl.gov/twister/bee/set/c9d469c3-b473-4283-8f8f-67237fc30722/event/list/>
— index `docs/pr/14_rescue-after.index.txt` (Bee idx 0..6 = 288952, 169824,
56463, 59003, 437699, 392200, 398690), PR→img id map
`docs/pr/14_rescue-after.prid-map.txt`.  288952 is included via
`--allow-unevaluated` (no nu candidate after the TGM tag): its PR layers are
the whole-event dump, so read its `clustering-global` (the joined crosser is
one cluster) rather than the PR layers.

**Before (same events, HEAD pre-rescue):** the doc pr/12 §8 pathological set
<https://www.phy.bnl.gov/twister/bee/set/f9d2ed52-617d-4835-a228-730feeaf84e1/event/list/>.

### 7.4 No-regression sweep — full mcp1k (1000 evts) + nueCC48 (48 evts), OFF vs ON

Four arms at the same `8d6de401` binary (Repro block), all rc=0:
`work-mcp1kall-cbroff1k` / `work-mcp1kall-cbron1k` (1000 events each) and
`work-nuecc48-cbroff` / `work-nuecc48-cbron` (48 events each, full QL + nusel
per event).  Per event, `cbr_sweep_compare.py` compares member-content hashes
(the §5 `hash_archive` definition) of the QL zip, the pctree tarball and the
PR zip, and greps the ON log for rescue firings.  Verdicts:

| sample | IDENTICAL (no firing) | FIRED | MISMATCH |
|---|---|---|---|
| mcp1k (1000) | 991 | 8 | 1 (286191 — pre-existing nondeterminism, exonerated below) |
| nueCC48 (48) | 47 | 1 (437699) | **0** |

The OFF arms contain **zero** rescue log lines (the env-gated TLA is the only
path to the knob).

**The 286191 MISMATCH is NOT the rescue.**  The rescue did not fire there
(component ran in 0.04 ms, no `rescue round` line), yet the arms' archives
differ.  Log diff localizes the first divergence *upstream* of the rescue,
inside QLMatching's final bundle association: a 1.7 cm out-of-beam speck
(cluster ident 1, `lm=1`) is adopted by flash 21 (856.1 µs, KS 0.323) in the
OFF-arm run but flash 17 (931.8 µs, KS 0.319) in the ON-arm run — two
near-degenerate cosmic flashes — which cascades into a different
`examine_bundles` collapse count (`re-stamped 6` vs `5 cluster(s)`) and
different hashes.  Exoneration: 2 fresh OFF re-runs + 2 fresh ON re-runs of
the event (`work-mcp1kall-cbr286191{offr1,offr2,onr1,onr2}`) ALL produce the
identical pctree hash `15900dce…` — equal to the ON arm — so the knob does
not select the outcome; the original OFF-arm run simply caught the rare
branch of the known residual QL-matching run-to-run nondeterminism (doc 28
lineage: pointer nondeterminism FIXED, benign residual remains).  Physically
irrelevant (out-of-beam speck), but this is the first 1-in-1000 sighting in
an A/B context: worth remembering when a single-event QL hash MISMATCH shows
up in future gates — re-run the event before suspecting the change.

**Firings.** All 8 hand-scan moves reproduce move-for-move (§7.1 decisions,
including 56463's two opposite-direction rounds), plus **two new firings**,
both MC events outside the pr/12 list:

| event | move (`rescue round` log) | geometry (`[cbrx]` trace) | PR outcome OFF → ON |
|---|---|---|---|
| 352365 | beam c7 (gid 2, t0 0.722 µs, 220.4 cm) + far c13 (gid 1000002, t0 2.685 µs, 219.1 cm) → **beam** gid 2 (longer-half) | far-regime accept: tips 2.88/0.38 cm from the cathode, 3-D gap 10.5 cm, Hough/PCA angles 4.0°/0.8°, connectivity 3.3 | half-cosmic "nu-candidate" (218.9 cm) → joined 634 cm crosser spanning x −80→+78, y −199→+199 → **TGM=true**, candidate removed (same class as 288952) |
| 395148 | beam c10 (gid 0, t0 1.533 µs, 410.3 cm) + far c24 (gid 1000003, t0 0.078 µs, 102.1 cm) → **beam** gid 0 (longer-half) | close-regime primary accept: tips 0.51/1.13 cm, gap 2.85 cm, angle 2.8° | TGM (424 cm) → joined 533 cm, **STM**; either way cosmic-tagged, no candidate change in kind |

Both are exactly the targeted failure mode (a second flash within ±2 µs of the
beam flash steals one half of a cathode crosser), not collateral: accepted as
correct new rescues, no cut tightening needed.

**Known reporting artifact (352365 only).** The ON-side `nusel-evt352365.tsv`
prints the beam flash as `no-bundle`.  The crosser is present and evaluated —
the PR log tags it TGM and `mabc-pr.zip` holds its 4766-pt main spanning the
cathode — but the QL→PR cluster-ident mapping swapped idents 11↔12 for this
event, and `nusel_extract.py` matches QL idents against PR-Bee idents.  This
is a pre-existing assumption in the analysis script (first event observed to
break it), NOT a chain regression; left as-is per the one-change rule, worth a
follow-up fix in `nusel_extract.py` (match by gid + size, or carry PR idents).

**In-beam label changes (standard nusel chain), all 9 firing events:**

| event | OFF | ON | net |
|---|---|---|---|
| 169824 | nu-candidate (199 cm half) | nu-candidate (286 cm whole, fit spans) | candidate recovered whole |
| 59003 | nu-candidate (144 cm half) | nu-candidate (319 cm whole, fit spans) | candidate recovered whole |
| 437699 | nu-candidate (νe CC) | nu-candidate, bundle −226 pts | candidate cleaned of the 25 cm cosmic stub |
| 288952 | nu-candidate | TGM | fake candidate (half-cosmic) removed |
| 56463 | nu-candidate | TGM (joined 546 cm crosser) | candidate removed — needs owner truth check (§7.2) |
| 352365 | nu-candidate | TGM (see artifact note) | fake candidate removed |
| 392200 | nu-candidate (102 cm) | STM (58 cm leftover) | fake candidate removed |
| 398690 | TGM | TGM (companion moved out) | no change in kind |
| 395148 | TGM | STM | no change in kind |

Net over 1048 events: 2 candidates recovered whole, 1 cleaned, 4 cosmic
fakes removed, 0 candidates altered in any non-firing event (286191's
nondeterministic flip is confined to an out-of-beam speck's flash choice —
no in-beam bundle is affected).

**Determinism (§5.3).**  Two fresh ON re-runs of 437699 (`work-cbr-det1/2`)
vs the sweep arm: all three archives (QL zip `3e959653…`, pctree `9e156311…`,
PR zip `e195f87c…`) 3-way identical.  Independently, the §7 demonstration arm
`work-mcp1kall-rescue01` and the sweep arm `work-mcp1kall-cbron1k` produced
the identical QL zip hash for 56463 (`d4467b1f…`) in separate batch runs.

## 8. Prototype provenance

None — this is a new algorithm, not a WCP port. There is no prototype counterpart to
cite; the failure mode itself is documented in doc pr/12 §6 and the flash-reco
absorbing-window defect is the owner's diagnosis from the SBND light reconstruction.
