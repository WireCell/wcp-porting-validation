# doc pr/107 — `dqdx_fit_keep_all_points`: prototype-parity point retention for the dQ/dx fit (2026-08-21)

**Status:** toolkit knob SHIPPED **DEFAULT OFF** (`dqdx_fit_keep_all_points`, C++ / jsonnet / runner env
`SBND_DQDX_FIT_KEEP_ALL_POINTS`).  OFF gate PASS 94/94 (nueCC48) + 38/38 (NCpi0) byte-identical vs the
previous binary.  ON evaluated on nueCC48 + NCpi0 only (owner: "use the nueCC and NCpi0 to evaluate the
nu vtx again").  **NOT flipped** — owner decides the next step (§7).

Owner (2026-08-21), after pr/106 §9–10 and the code comparison of the exclusion fit in both
implementations: *"What we want is to include the fit exclusion, but correct the dQ/dx fit in the toolkit
… go ahead with implement the option a) to improve the dQ/dx fit in the toolkit to be more consistent with
that in prototype."*

## 0. Repro block

```bash
cd toolkit && ./wcb build --notests -p && ./wcb install --notests -p        # lib 14:26:11 > src 14:24:56
./wcb build -p --alltests --target=wcdoctest-clus && ./build/clus/wcdoctest-clus   # 2356/2356 assertions
# compiled-config proof (full PR pipeline_names TLA, dl_weights=''): key count OFF 0 / ON 1; OFF JSON
# byte-identical to the pre-change tree (git stash round-trip, cmp).
cd sbnd_xin   # /home/xqian/tmp/pr107_arms.sh  (47 + 19 events, lists /home/xqian/tmp/vtx105-evts-<s>.txt)
PR_JOBS=32 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-<s>-ql0819 work-pr107-off-<s> data <evts>
SBND_DQDX_FIT_KEEP_ALL_POINTS=true SBND_DL_VTX_HARVEST=true PR_JOBS=32 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-<s>-ql0819 work-pr107-on-<s> data <evts>
# /home/xqian/tmp/pr107_eval.sh:
python3 scripts/pr85_hash_gate.py work-vtx106-cne-off-<s> work-pr107-off-<s>          # OFF gate
O="--orig-tags vtxscan-harv3-nuecc48 vtxscan-harv3-ncpi0 vtxscan-harv3-mcp1k vtxscan-harv3-delta vtxscan-mcp2k vtxscan-mcp2k-auto vtxscan-mcp2k-ragree"
python3 dl_vtx_training/vtx_target_eval.py $O --harv-base 'work-pr107-on-{sample}' --harv-rows 'work-pr107-on-{sample}' \
  --live-arms base=work-pr107-on-{sample} trad=work-vtx105-trad-{sample} --only-events <list> --closure --table \
  --events-tsv docs/pr/107_on/events-<s>.tsv
python3 scripts/pr90_movers.py   work-pr107-off-<s> work-pr107-on-<s> --tags vtx105 --tsv docs/pr/107_on/movers-<s>.tsv
python3 scripts/pr83r3_scores_ab.py work-pr107-off-<s> work-pr107-on-<s> --tsv docs/pr/107_on/scores-<s>.tsv
grep -h "pre-dQ/dx form_map_graph dropped" work-pr107-{off,on}-<s>/pr_evt*/wct_pr_evt*.log   # drop census
```

## 1. What is corrected, and why option (a)

`TrackFitting::do_multi_tracking` runs: trajectory round 1 (`form_map_graph(flag_exclusion)` +
`multi_trajectory_fit`), round 2 (same), then — **toolkit only** — a third `form_map_graph(flag_exclusion)`
before `dQ_dx_multi_fit` (pr/28 T4, "no prototype counterpart": `PR::Fit::reset()` clears the fit indices
that the prototype's `reset_fit_prop()` resize keeps, so the indices had to be rebuilt).  The prototype
goes `reset_fit_prop()` → `dQ_dx_multi_fit` directly and fits dQ/dx on **every** trajectory point.

`form_map_graph` stores an interior point only if its U+V+W association quantity is > 0
(`TrackFitting.cxx` ~3586, endpoints always kept).  With `fit_exclusion` on (SBND production since
pr/98), `update_association` strips every 2-D cell that is (near-)equidistant from two segments — exactly
the last ~1 cm before a multi-prong junction — so the third pass **deleted** those trajectory points from
the final fit, from the segment "fit" point cloud, and from the DL vertex net's input cloud:
**442 points over 47 nueCC48 events (86 with exclusion off)**, the mechanism behind pr/106 §9's
`fit_exclusion`-OFF gain (DL-alone 34→42/47).

Neither dQ/dx fit consumes the association maps: in the toolkit `m_3d_to_2d`/`m_2d_to_3d` are read only
by `form_map*`, `fit_point`, `trajectory_fit`, `multi_trajectory_fit`, `skip_trajectory_point`,
`examine_segment_trajectory`; `dQ_dx_multi_fit` works from the fit indices and Gaussian response
integrals against the full 2-D charge maps (10 `cal_gaus_integral*` sites, 0 association reads), as does
the prototype's (`map_3D_2DU_set` 0 references in its dQ/dx code).  So the third pass contributes exactly
two things: the sequential fit indices and the drop.  Options considered with the owner:

| | points reaching `dQ_dx_multi_fit` | vs prototype |
|---|---|---|
| production (exclusion pass, drop) | every interior point whose cells were all stripped at a junction is deleted | diverges |
| (b) third pass with `flag_exclusion=false` | still deletes points whose *unfiltered* quantity is 0 (dead/empty regions) | closer, not exact |
| **(a) keep every point, renumber only** | complete trajectory, as the prototype's resize leaves it | **exact point set** |

Option (a) built.  Trajectory rounds 1–2 keep exclusion exactly where the prototype applies it;
`dQ_dx_multi_fit` is untouched; a kept zero-quantity point gets a fresh index, `flag_fix=false` and an
empty association (never read downstream), and receives its dQ from the simultaneous Gaussian fit against
the full charge maps like every other point.

## 2. Implementation (toolkit, DEFAULT OFF, byte-identical when off)

- `TrackFitting::Parameters::dqdx_fit_keep_all_points` (double sentinel, 0 = legacy) with
  `set_parameter`/`get_parameter`; transient `m_keep_zero_quantity_points` set by `do_multi_tracking`
  **only around the pre-dQ/dx `form_map_graph`** (`TrackFitting.cxx` ~8784) and cleared after; the keep
  rule becomes `quantity > 0 || m_keep_zero_quantity_points`.  `inherit_from` copies `m_params`, so every
  local fitter (`NeutrinoVertexFinder.cxx:770`, 4803) follows.  The existing "dropped N of M" DEBUG line
  stays (count 0 when on).
- `TaggerCheckNeutrino`: `m_dqdx_fit_keep_all_points` ← config `dqdx_fit_keep_all_points` (+
  `default_configuration`), pushed to the fitter next to `traj_cover_probe`.
- jsonnet: `common/clus.jsonnet` arg + key suppression → `sbnd/clus.jsonnet` (`clus_pr`/`pr`, both
  signatures + forwards) → `wct-pr-perevt.jsonnet` TLA `dqdx_fit_keep_all_points=false`; runner env
  `SBND_DQDX_FIT_KEEP_ALL_POINTS` (empty = no TLA).
- Tests: `doctest_clus_knob_defaults` pins the config key OFF and a new `TrackFitting` round-trip case
  (default 0, set 1, preset 0).  `wcdoctest-clus` 2356/2356.
- Compiled-config proof: with the runner's `pipeline_names`, `dqdx_fit_keep_all_points` count 0 OFF / 1
  ON; the OFF JSON `cmp`-identical to the pre-change tree.

## 3. Gates — `work-pr107-{off,on}-{nuecc48,ncpi0}` (all rc=0: 47/47, 19/19)

- **OFF gate** (new binary, no env) vs `work-vtx106-cne-off-<s>` (previous binary 14dd031d, same
  config): `pr85_hash_gate` **PASS 94/94** (nueCC48), **PASS 38/38** (NCpi0).
- ON vs OFF archives differ (expected: every dQ/dx fit changes).
- Drop census (DEBUG line, OFF arm): nueCC48 **443 points / 188 fits → 0**, NCpi0 **101 / 46 → 0**.
- Wall (PR tail, log first→last stamp): nueCC48 mean 29.1 → 30.3 s (+4 %), NCpi0 19.6 → 19.6 s.

## 4. Vertex — target metric (targets on the ON arm's own pre-DL cloud, original labels; closure 0 on every arm)

| | nueCC48 M3 (production rule) | nueCC48 DL-alone | NCpi0 M3 | NCpi0 DL-alone |
|---|---|---|---|---|
| production (pr/106 §3, same targets) | 35/47 | 34/47 | 14/19 | 12/19 |
| pr/106 §10 `dl_vtx_cloud_no_exclusion` | 38/47 | 37/47 | 14/19 | 13/19 |
| pr/106 §9 global `fit_exclusion` OFF | 41/47 | 42/47 | 13/19 | 11/19 |
| **pr/107 `dqdx_fit_keep_all_points` ON** | **36/47** | **36/47** | **14/19** | 12/19 |

nueCC48: one fix (46363, the production miss by 9 dropped junction points in §9's ledger), no breaks;
M3 classes hit 36 / not-admitted 5 / wrong-accept 4 / reject-to-wrong 2.  NCpi0 unchanged event-for-event.
Near-target (3 cm) cloud census ON vs the production harvest: point count 541 → 545, sum-dQ within
±5 % on 10 events (46363 +2 %) — much smaller than §9's +4..50 %: most of §9's gain came with the
exclusion-OFF *topology*, not from the retained points alone.

`pr90_movers` (1 cm, vtx105 labels): nueCC48 11 movers > 0.05 cm, **ADVERSE 0** (toward 1, on-click
sub-mm churn 10); NCpi0 1 mover, toward (56982 11.1 → 9.7 cm), ADVERSE 0.

## 5. Selection ledgers (`107_on/scores-<s>.tsv`)

- nue-selected (`nue_score ≥ 4.3`) nueCC48 **35 → 35**; `nue > 0` 40 → 39 (196649 4.24 → −2.32, not
  selected in either arm); `numu > 0` 13 → 13 (54095 0.01 → −0.92, 81597 −0.74 → 0.36).
- NCpi0: `nue ≥ 4.3` 1 → 1, `nue > 0` 2 → 2 (84229 0.69 → −0.06, **56982 −15 → 4.29** — 0.01 below the
  cut), `numu > 0` 10 → 10.

But the kinematics move in 12/47 nueCC48 and 4/19 NCpi0 events (|ΔEnu| > 5 % or shower/segment count):

| evt | Enu OFF → ON | leading e-shower | showers | note |
|---|---|---|---|---|
| 81597 | 1578 → **696** | 1416 → 533 MeV (55 → 35 seg) | 6 → 6 | 18 shower segments become orphans (`shower_id` −1); 19036 pid 11→13, 19103 11→2212; nue stays 4.30, numu −0.74→0.36 |
| 196649 | 1614 → **365** | 1608 → 286 (53 → 15 seg) | 6 → 11 | shower dissolves into 5 pieces (286/36/12/9 …); nue 4.24 → −2.32 |
| 46363 | 1964 → 1850 | 1056 → **1805** (41 → 56) | 19 → 22 | the vertex fix: the e-shower now starts at the target vertex and absorbs the 398 MeV "pion" |
| 489330 | 2254 → 2100 | 1440 → 1481 | 5 → 2 | |
| 168596 | 2402 → 2166 | 1390 → 1564 | 14 → 14 | |
| 246579 / 444187 / 90055 | −3 / −5 / −4 % | ±3 % | 16→14 / 5→4 / 32→33 | |
| 239794 / 42280 / 433451 / 54095 | < 2 % | < 2 % | ±1 | |
| NCpi0 56982 | 776 → 1142 | 146 → **705** (12 → 32 seg) | 32 → 24 | main vertex moves 4.8 cm toward the click; a 705 + 233 MeV shower pair forms at vertex 22030; 22119 pid 211→11, 22120 2212→11; nue −15 → 4.29 |
| NCpi0 84229 / 21073 / 71372 | < 1 % | < 3 % | ±1 | |

The global `fit_exclusion`-OFF arm (§9) keeps 81597 (1421 MeV) and 196649 (1612 MeV) intact and gives
46363 only 399 MeV — so these reshuffles are specific to *this* change, not a re-run of §9.

## 6. Mechanism evidence

- The retained points carry physical charge: 81597 electron start segment 19009 dQ/dx (×10³/cm, from the
  vertex) OFF `183, 62, 49, 37, 46 …` → ON `197, 126, 62, 49, 37 …` — the new second point is the
  junction point, and it takes the proton's overlap charge (proton 19046 second point 171 → 186).  This
  is the prototype's behaviour too (the simultaneous fit splits junction charge between the prongs' points)
  but it raises the first-centimetre dQ/dx of every prong at a junction, which is exactly what the
  track/shower PID and the EM-shower clustering read (`particle_id` flips above).
- No pathological points: nueCC48 `dQ ≤ 0` 652 → 648, `dx ≤ 0.05 cm` 16 → 13, `dQ/dx > 2×10⁵` 870 → 895
  (of 30.7 k); NCpi0 188 → 200 / 6 → 6 / 253 → 256.
- Cloud sizes: nueCC48 29 903 → 29 808 points, NCpi0 11 393 → 11 399 (the retained points are offset by
  trajectory changes in later rounds).

## 7. Verdict / open

The knob does what was asked — the dQ/dx fit now sees the complete trajectory as the prototype's does
(443 + 101 → 0 dropped points), gate byte-identical when off, zero ADVERSE vertex movers, nue/numu
selection counts unchanged on both samples — but the vertex gain is small (+1/47, DL-alone +2, NCpi0 0)
and the retained junction charge changes the early-dQ/dx of prongs at multi-prong vertices, which
reshuffles EM-shower clustering on 12/47 nueCC48 events (two large electron-energy losses, one gain) and
lifts one NCpi0 background to nue 4.29.  NOT flipped; the numu samples were not run.  Owner to direct the
next step (mechanism of the 81597/196649 shower loss — segments leaving the shower vs orphan —, whether
the junction-point charge should be down-weighted in the PID reads, or extend to mcp1k/mcp2k as is).
