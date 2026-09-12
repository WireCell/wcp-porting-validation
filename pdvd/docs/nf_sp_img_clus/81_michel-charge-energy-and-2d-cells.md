# 81 — The Michel's energy as charge: the 2-D residual to the muon fit, and the Michel / STM cells

The owner's request (2026-09-10), in his words: the Michel electron's trajectory wiggles, so neither a range estimate nor a dQ/dx→dE/dx integral along a fitted trajectory is reliable for it; the best estimate is **charge, scaled by a constant** — and the difficulty, that the stopping muon's charge must be removed first, is solved by **subtracting the charge the muon's trajectory fit predicts**. Two deliverables: that estimator, stored in the output; and the **2-D measurements associated with the Michel with the STM part removed** (and the STM's own), for his further work.

**Status (2026-09-10): BUILT and GATED, default OFF, NOT flipped -- the definition needs an owner decision (sec 8).** Toolkit `clus/` + one writer line in `root/`; no `pr.jsonnet` change (the knobs ride the `stm_michel_knobs` bag). The owner's pre-arm decisions: planes combined with the chain's own rule; flip PDVD and PDHD production if the ON arm is sane (sec 4). It is not: on PDHD 45 % of Michels lean on a fitted substitution for cells shared with a non-preloaded cluster (bridged Michels read 0.82 of the chain there against 1.26 on PDVD), and the largest reading is a 19-piece Michel object the chain itself puts at 206 MeV. Both knobs therefore ship OFF in C++ and in both drivers; the flip is one key each and the proofs are written (sec 8).
- **OFF gate PASSES both detectors** (sec 6): every Bee zip member, calib json and `tracking-pr.root` tree identical; PDVD 596/596 candidates bit-identical on all 140 `T_stm_michel` branches, PDHD 325/325 on 130; 0 `is_stm` flips.
- **ON adds output only**: 23 new branches + `T_stm_michel_2d`, every pre-existing branch and point row bit-identical, `census_score.py` unchanged (232/7/46, 138/12/22). `michel_q2d_valid = 1` on 596/596 and 325/325 -- the fit-response guard never fired.
- **Median `michel_ke_q2d_total / michel_ke_total`** 1.21 PDVD / 0.95 PDHD; the muon charge subtracted is 16 % of the Michel cells' measured charge.

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
# build (toolkit clus/ + root/), the tests, the pin (libWireCellClus.so 2f78cea8)
cd /nfs/data/1/xqian/toolkit-dev/toolkit && wcbuild && ./build/clus/wcdoctest-clus && ./build/root/wcdoctest-root
cp -a ../local/lib /home/xqian/tmp/p81/libpin_p81g   # the WHOLE dir -- a partial pin is not a pin (sec 5.3)
# wave 1c: 6 arms (leg on libpin_p75, off + on on libpin_p81g, both detectors; bare production config), detached
WAVE=1c nohup bash $X/d81_arms.sh > /home/xqian/tmp/p81/arms_wave1c.log 2>&1 < /dev/null & disown
bash $X/d81_gates.sh | tee /home/xqian/tmp/p81/gates_1c_v2.log    # OFF gates + what the ON arms write
python3 $X/d81_readout.py --det pdvd --arm p81vq2d3 --off p81voff3 --json /home/xqian/tmp/p81/readout_pdvd.json
python3 $X/d81_readout.py --det pdhd --arm p81hq2d3 --off p81hoff3 --json /home/xqian/tmp/p81/readout_pdhd.json
python3 $X/d81_tail.py /home/xqian/tmp/p81/readout_pdvd.json pdvd    # the conn_type split and every tail candidate by name
python3 $X/d81_tail.py /home/xqian/tmp/p81/readout_pdhd.json pdhd
# the ON arms' TLA, for a one-off:  PDVD_PR_TLA="-S stm_michel_extra={michel_q2d:true,michel_q2d_cells:true}" ./run_pr_evt.sh -nu -stm-fit ...
```

## 1. What the chain computes today, and why it is not this

`michel_ke_best = michel_ke_dqdx + dots_ke_unfit` (`CheckSTM_Michel.cxx`, doc pdvd/52 §3): a dQ/dx→dE/dx inversion summed over every fitted point of the Michel object (`calculate_shower_kinematics`, the calibrated PowerBox inverse of doc pdhd/16), plus a charge→energy conversion at an assumed MIP dE/dx for the admitted companion pieces the fitter produced no segment for (doc pdhd/17 §9). The taken gamma blobs (doc pdvd/71) add `michel_ke_gamma`, their own dQ/dx integrals, into `michel_ke_total`. `michel_ke_range` is kept for comparison only; `michel_ke_charge` is `Shower::get_kine_charge()` with the SHOWER recombination pair and overshoots by ~1.7× (doc pdhd/15 §6).

Every one of those reads charge **through the trajectory**: a point's dQ is the fit's share of the 2-D charge for that trajectory row, and a trajectory that misses part of the electron's deposition (the wiggle, a detached blob the association did not reach, the cells beyond the fit's last row) loses that charge for good. The owner's estimator reads the 2-D cells themselves, and only uses the trajectory fit for what it is good at on the muon — predicting where the muon's charge went.

## 2. The estimator

For the candidate's main cluster the fitter has, at the end of the PR stages, one multi-track dQ/dx fit whose solution is every segment's `fits[i].dQ` (`TrackFitting::dQ_dx_multi_fit`: `pos_3D` indexed by `Fit::index`; the prediction of every 2-D cell is `R · pos_3D` with a whitened response matrix `R`, un-whitened by the row's `total_err`). Nothing refits the main cluster after the chain is known: `find_proto_vertex` ends on the fit, `break_segment` (the anchor / stop split) slices fit rows and keeps their index, companion fits are cluster-scoped.

1. **The muon's prediction.** With `michel_q2d` on, the fitter keeps that fit's response for the main cluster (`TrackFitting::Parameters::keep_dqdx_response`; erased at the next fit's entry so an early return can never leave a stale pairing). The muon rows are the chain's `Fit::index` set (the shared stop-vertex row counts as muon). Guard: every chain row's `dQ` must **be** the stored solution, bit for bit, else `michel_q2d_valid = 0` with a reason code. `pred_mu(cell) = scale · Σ_{k ∈ chain} R(cell, k) · pos_3D(k)`, in electrons.
2. **The Michel's cells.** The chain's own association rule, `kine_charge_from_maps` (`NeutrinoEnergyReco.cxx`): over the union charge maps of every preloaded cluster, per plane, a cell belongs to an object when the nearest point of the object's `associate_points` cloud (fallback `fit`) lies within 0.6 cm (`michel_q2d_dis_cm`). The object is the Michel `Shower`'s member segments (the arm and everything the shower walk gathered, plus the admitted pieces) — role 3; the taken gamma blobs' segments — role 4. A cell in both is the Michel's.
3. **The subtraction, per plane.** `q_p = Σ_cells (charge − pred_mu)`, **signed** (a cell the muon fit over-predicts subtracts; the offline reader has the cells to do otherwise). Companion cells carry no muon prediction (their fits are their own). **Except on a cross-shared cell** (§5.2): a channel-slice that another, *non-preloaded* cluster's blob also covers carries that cluster's charge too — the fitter marks it by setting `charge_err` to its share sentinel (8000) and deweights it. The measured charge there is not the Michel's to sum; the cell contributes the fit's own non-muon prediction `max(pred_all − pred_mu, 0)` instead — *measured where attributable, fitted where not*. The plain sum is persisted beside the headline (`michel_q2d_raw_u/v/w`) with the shared-cell counts (`michel_q2d_nx_u/v/w`). A charge-only Michel (`conn_type 3`, unfitted companion clusters) gets its clusters' cells by the fitter's own blob-coverage predicate (`sel` bit 4).
4. **The planes.** `stm_michel_combine_planes`, the chain's rule forked by duplication: weights U/V/W 0.25/0.25/1.0, and when the two largest planes disagree by more than 4 % the largest is dropped (owner's choice, asked and answered 2026-09-10). Persisted: the three signed sums, the three subtracted muon charges, the three cell counts, the dropped plane.
5. **The constant.** The unfitted-piece conversion the component already holds (`michel_unfit_from_model`, production ON: the bound recombination model inverted at `michel_unfit_dedx` 2.1 MeV/cm, doc pdhd/17 §9): `michel_ke_q2d` (role 3), `michel_ke_q2d_gamma` (role 4), `michel_ke_q2d_total` = both. A non-positive combined charge gives 0 MeV.

Nothing reads any of this back: `michel_ke_best`, `michel_ke_total`, `michel_found`, `is_stm` and every reject bit are unchanged by construction (the block runs after the last verdict and the gamma take, immediately before persist).

## 3. The cells (`michel_q2d_cells` → PC `stm_michel_2d` → `T_stm_michel_2d`)

One row per cell, the main cluster's `cluster_id` as the candidate key: `apa face plane wire time time_slice channel flag` (the readout: `time` is the fit's own key, the slice's first **tick**; `time_slice` = `time / nticks_per_slice`, the key `T_proj_data` uses; `flag` 0 dead / 1 live / 2 bad), `role` (1 the STM footprint, 3 the Michel, 4 a taken gamma), `shared` (1 when the muon fit predicts charge on this cell — the cells the subtraction acts on), `xshared` (1 when the cell is cross-shared with a non-preloaded cluster, §2 item 3 — the cells the headline reads from the fit), `sel` (bit 1: the nearest-point rule above; bit 2: fit support — a nonzero response entry in a Michel-segment column of the main cluster's fit; bit 4: covered by an unfitted admitted cluster's blobs; a role-3 row with `sel = 2` alone is a cell the Michel's fit reaches but its association does not — the headline sums use bits 1 and 4), `charge charge_err` (the measurement; `charge_err` at 8000 is the fitter's share sentinel, which is what `xshared` reads), `pred_mu` (the muon-only prediction), `pred_all` (the whole fit's). Role-1 rows are the cells the chain rows within `michel_q2d_stm_window_cm` (30) of the stop predict charge on and no Michel/gamma claims; −1 = the whole chain (a 3 m muon is 10⁴–10⁵ rows, hence the window). Rows sorted by (role, plane, apa, face, wire, time). The writer is schema-driven (`PdvdPrMagnifyTrackingVisitor::write_pc_tree`), so the tree exists only when a candidate carried the PC; every column depends on the knobs alone (TensorDM's `as_tensors` requires same-named PCs to share their columns).

## 4. Pre-stated criteria (written before wave 1 finished)

- **OFF gate** (both detectors, `p81voff` vs `p81vleg`, `p81hoff` vs `p81hleg`, run side by side on the same production config and on the doc-80 pin vs this pin): every `mabc-pr.zip` member, every calib json, every `tracking-pr.root` tree, every `T_stm_michel` branch and point row byte-identical; no `T_stm_michel_2d`.
- **ON arms** (`p81vq2d`, `p81hq2d`): Bee zip and calib identical to the OFF twin; every production `T_stm_michel` branch identical candidate by candidate; the 17 new scalars and the new tree appear; every point row identical; `census_score.py` reads the OFF arm's numbers.
- **Sane** (the owner's flip condition): `michel_q2d_valid = 1` on every Michel-carrying candidate (the reasons censused otherwise); every energy finite; the median of `michel_ke_q2d_total / michel_ke_total` inside 0.8–1.3 with the tails explained by name; the per-plane spread and the dropped-plane rate not pathological. If met: `michel_q2d: true, michel_q2d_cells: true` in both `wct-pr-perevt.jsonnet` bags, proofs A/B/C, confirmation arms `p81vprod` / `p81hprod` bit-identical to the ON arms.
- Predictions: the estimator should come out **above** `michel_ke_best` on most objects (it collects the charge the trajectory missed) but not by a large factor on clean attached Michels; the subtracted muon charge should be a minority of the Michel cells' charge (the Bragg peak's last cells); gamma terms should track `michel_ke_gamma` loosely (both are that blob's charge, one fitted, one not).

## 5. What was built (toolkit `clus/` + `root/`)

- **`TrackFitting`** (`clus/inc/WireCellClus/TrackFitting.h`, `clus/src/TrackFitting.cxx`): `Parameters::keep_dqdx_response` (0 = off, the shared SBND/uBooNE code path untouched); `DqdxResponse` / `DqdxResponseRow` (the three whitened sparse matrices, `pos_3D`, the per-plane row keys in the fit's own row order with the row's un-whitening `scale`, its flag, and every `(face, wire)` the readout row maps to); the store is **erased at `dQ_dx_multi_fit` entry** for the filtered cluster and re-captured after the prediction, so the three early returns can never leave a response paired with reset fits; `get_dqdx_response(cluster)`, `clear_dqdx_responses()` (also in `release_fit_scratch`); the pure `masked_response_prediction(R, pos, col_mask, row_scale)`.
- **`StmMichelFunctions`**: `stm_michel_combine_planes(sums, weights, asym_switch, dropped_plane*)` — `kine_charge_from_maps`'s plane rule statement for statement, documented for signed input.
- **`CheckSTM_Michel`**: knobs `michel_q2d` (false), `michel_q2d_cells` (false), `michel_q2d_dis_cm` (0.6), `michel_q2d_stm_window_cm` (30); `Record` gains the 17 scalars, a per-role segment list filled by `add_points` (bookkeeping only) and the cell vectors; `michel_q2d_estimate(...)` runs at the publish point after the gamma take; `persist()` writes the scalars under the knob and the `stm_michel_2d` PC under both knobs when non-empty; the fitter gets `keep_dqdx_response = 1` with the first knob.
- **`root/src/PdvdPrMagnifyTrackingVisitor.cxx`**: one `write_pc_tree(... "stm_michel_2d", "T_stm_michel_2d")` line (the class serves PDVD and PDHD; SBND has no STM trees).
- **Tests**: `clus/test/doctest_stm_michel_q2d.cxx` (plane rule: symmetric → weighted mean, asymmetric → the largest plane dropped, negative med+max never trips the switch, all-equal fallback ordering, zero weights; the masked prediction with short masks / short scales / a wrong-length `pos`; the parameter's default 0 and round trip); `doctest_check_stm_michel_defaults.cxx` pins the four knobs. `wcdoctest-clus` 376/376, `wcdoctest-root` pass.
- **No jsonnet**: the knobs ride `stm_michel_knobs` (merged into the component data by `cm.check_stm_michel`); `pdhd/stm_michel_scan/prep_stm_michel_scan.py` `VERDICT_SCALARS` lists the 17 scalars so a payload carries them under the blinded `verdict` key.

### 5.1 Two defects the first ON arm exposed (both fixed before the graded arms)

The first ON arm (`p81vq2d`, pin `2525f189`) wrote `michel_q2d_n_u/v/w = 0/0/0` on every candidate while the fit-support selector found cells — the nearest-point rule associated nothing. A single-event debug run (`039252_15`, the `dmin` diagnostic now on the DEBUG line) read the nearest Michel point **492 cm** from the nearest cell:

1. **The drift frame.** `convert_time_wire_2Dpoint` puts a cell at the *geometric* drift (the raw t0 = 0 frame); a segment's `associate_points` / `fit` cloud stores the cluster's *t0-corrected* x. On PDVD the two differ by the cluster's t0 drift plus the trigger offset — metres (−551, −347, −301, −28 cm on that event's candidates). This is doc pdvd/45's finding on the fitter's exclusion test (`excl_t0_frame`), and it is why the chain's own `michel_ke_charge` — the same `kine_charge_from_maps` rule, uncorrected — is 0 on nearly every PDVD Michel (doc pdhd/16: non-zero on 6 of 160), a frame mismatch rather than "no charge". The estimator measures the shift **per segment and (apa, face)** from the segment's own fit points — `xform->backward(p, t0, face, apa).x() − p.x()`, the median — and subtracts it from the cell's drift before the query (the fitter's round trip without the tick rounding). After it: `dmin` 0.000 cm.
2. **The face.** A readout channel can map to `(face, wire)` pairs on both faces; the rule's "first match" picked the wrong face for the U and V planes on one candidate (`039252_15/91`: 0/0/121 cells). Every `(face, wire)` of the channel is now tested against the cloud (the query is per (apa, face)), and the matching one names the row.

A third consequence of (2) was fixed with it: a plane with **no** cells must not enter the plane rule as a zero-charge plane — it made the one populated plane "the largest" and the 4 % switch dropped it (`039252_15/91` read 0 MeV). A plane with 0 cells now takes weight 0, the rule's own "ignore this plane".

The four candidates of the debug event after the fixes (MeV; chain `best` / `total` in parentheses): `77` **49.9** (76.8 / 76.8 — doc 52 §3.5's over-endpoint object, the V plane dropped), `81` 28.4 + 15.7 gamma = **44.1** (22.0 / 36.3), `82` **11.8** (13.1 / 13.1), `91` 40.5 + 1.4 = **41.9** (38.2 / 38.9). Wave 1b (`p81voff2 / p81hoff2 / p81vq2d2 / p81hq2d2`, pin `libpin_p81f` = `3fb0d262`) is the graded set; wave 1's leg arms stand.

### 5.2 The third defect: cross-shared cells (found on the PDHD wave-1b arm)

The wave-1b PDHD readout (`p81hq2d2`, pin `3fb0d262`) put `michel_ke_q2d` at 33 MeV on a 5.9 cm, 5.5 MeV Michel (`028084_18/17`) and at 0 on 19 of 124 Michels. Cell by cell: the candidate's W plane held **14.2 M electrons on 103 Michel cells the whole fit predicts 0.28 M on**, and 478 of its 532 rows carried `charge_err = 8000` — the fitter's `share_charge_err` sentinel, set by `update_dQ_dx_data` on a channel-slice that a *non-preloaded* cluster's blob also covers and never restored on the multi-fit path. The writer's own `T_proj_data` block for that cluster says the same (64 M measured on its cells, 13 M predicted): a 2-D cell is a **channel-slice**, and on a dense PDHD beam event — induction channels serving both faces, other tracks and showers crossing in projection — most of a Michel's cells also hold somebody else's charge. Summing them raw is not a Michel energy. The rule (§2 item 3): on a cross-shared cell the fit's own non-muon prediction stands in for the measured residual; the plain sum is persisted beside it. After it: `028084_18/17` **5.31 MeV** (chain 5.54), `028084_15/50` 3.80 (2.35). On the PDVD debug event no cell is cross-shared and nothing moves. What the rule cannot see: charge from a *preloaded* companion overlapping the Michel in projection is not flagged (`028084_18/108`: 67.8 MeV against the chain's 23.4 with 0 cross-shared cells — the cells are the preloaded set's alone, so by construction the estimator counts them; the table has them for the owner to judge).

The 13 zero-energy `conn_type 3` Michels of that readout are the charge-only case — the Michel is nothing but unfitted companion clusters, which have no segment, no cloud and no fit; §2's `sel` bit 4 now gives them their clusters' cells by the fitter's blob-coverage predicate.

### 5.3 The leg arms of wave 1 were not a baseline

`p81vleg` / `p81hleg` ran on `libpin_p80`, which holds **only `libWireCellClus.so`**; every other library came from `local/lib`, already rebuilt against this round's changed `TrackFitting` layout, and every event died inside the tracking visitor (0-byte `mabc-pr.zip`, a 723-byte `tracking-pr.root`, `complete 0`, no error line). A partial pin is not a pin. Wave 1c's leg arms `p81vleg2` / `p81hleg2` run on `libpin_p75` (`02557b8d`, toolkit `567a7232`, a full 790-file directory) — the last full pre-change pin; the only C++ between it and this round is doc 80's default-OFF census, whose own OFF gate was byte-identical. The production configs were unchanged through every wave (`cmp` against copies taken before wave 1).

## 6. Gates (wave 1c, `d81_gates.sh` → `/home/xqian/tmp/p81/gates_1c_v2.log`)

All six arms complete (PDVD 120/120, PDHD 61/61), 0 loader deaths, pins unchanged start to end (`02557b8d` leg, `2f78cea8` off/on).

**The OFF gate PASSES on both detectors.** `p81voff3` vs `p81vleg2`: every `mabc-pr.zip` member identical on 120/120 events, every calib json identical on 119/119, **every tree of `tracking-pr.root` identical on every event** (`T_bad_ch`, `T_cluster`, `T_proj`, `T_proj_data`, `T_rec_charge`, `T_stm_michel`, `T_stm_michel_pts`, `Trun`), **596/596 candidates bit-identical on all 140 `T_stm_michel` branches**, 0 `is_stm` flips, 596/596 with identical point geometry, and no `T_stm_michel_2d` anywhere. `p81hoff3` vs `p81hleg2`: the same, 61/61 zips, **325/325 candidates on all 130 branches**. §2b: `p81vleg2` reproduces production (`p79vprod`) exactly, so the baseline is sound and the gate covers this round's C++ *and* doc 80's default-OFF census.

**The ON arms add output and nothing else.** `p81vq2d3` vs `p81voff3`: Bee zip identical 120/120, calib 119/119, every pre-existing tree identical including `T_stm_michel_pts`; `T_stm_michel` gains **23 new branches, 0 dropped, 596/596 bit-identical on all 140 shared ones, 0 `is_stm` flips**; the point-row role histogram is identical (`{1: 170898, 2: 2895, 3: 3718, 4: 411, 5: 280}` both arms); `T_stm_michel_2d` appears on 119 events. PDHD the same on 325/325. `census_score.py` reads the OFF arm's numbers exactly (`is_stm` 232/7/46, `michel_found` 138/12/22). `michel_q2d_valid == 1` on **596/596** PDVD and **325/325** PDHD candidates — the response guard never fired.

Cost: `tracking-pr.root` **49.8 → 60.7 MB** per 120 PDVD events (+22 %), **30.1 → 36.3 MB** per 61 PDHD events (+21 %), all of it `T_stm_michel_2d`.

## 7. What the estimator reads (`d81_readout.py`, `d81_tail.py`)

| | PDVD (`p81vq2d3`, 164 Michels) | PDHD (`p81hq2d3`, 124 Michels) |
|---|---|---|
| `michel_ke_q2d_total / michel_ke_total` p10 / p50 / p90 | 0.87 / **1.21** / 2.10 | 0.38 / **0.95** / 2.07 |
| `michel_ke_q2d_total` MeV p10 / p50 / p90 / max | 6.9 / 32.1 / 54.8 / 500.7 | 0.9 / 8.1 / 44.3 / 80.4 |
| above 60 MeV | 8 of 164 (chain: 3) | 4 of 124 (chain: 2) |
| subtracted muon charge / measured Michel-cell charge p50 | 0.159 | 0.153 |
| Michel cells per plane p50 (u/v/w) | 77 / 78 / 94 | 53 / 50 / 34 |
| plane switch dropped a plane | 94 of 164 | 96 of 124 |
| per-plane spread (max−min)/mean p50 | 0.19 | 0.67 |

The closure check on the cell table holds exactly: `Σ pred_mu` over the role-1 footprint plus the shared Michel cells is **1.0000** of `Σ pred_mu` over every row, on every event of both detectors — no predicted muon charge is outside the table.

## 8. Why this is NOT flipped

The pre-stated sanity bar (§4) was "median inside 0.8–1.3 **with the tails explained by name**". The PDVD median passes and the PDHD median (0.95) sits just below the band's centre; the tails do not, in two distinct ways, and both are definition questions the owner should answer rather than something to tune.

**(a) PDHD leans on the fitted substitution, and it is not neutral.** 56 of 124 candidates (45 %) carry cross-shared cells; on those, `michel_ke_q2d` is **0.689** of the all-measured reading (p10 0.18). On PDVD only 36 of 164 (22 %) are affected and the median is 1.000. So the same branch means "measured charge" on PDVD and "largely the fit's own prediction" on PDHD — exactly the trajectory dependence the method was meant to escape. Split by connection type (the discriminating test):

| | PDVD p50 ratio | PDHD p50 ratio |
|---|---|---|
| attached (`conn_type 1`) | 1.19 (n 131) | 1.23 (n 69) |
| bridged (`conn_type 2`) | 1.26 (n 33) | **0.82** (n 40) |
| charge-only (`conn_type 3`) | — | **0.55** (n 15) |

A bridged Michel's segments live in a *companion* cluster with its own fit, so the main cluster's `pred_all − pred_mu` is ≈ 0 there and the substitution reads the cell as empty. On PDVD, where cross-sharing is rare, bridged Michels are unaffected (1.26); on PDHD they are pulled to 0.82 and the charge-only class to 0.55. **Named mechanism, PDHD-specific.** The three options — keep the substitution, sum the cross-shared cells raw (`michel_q2d_raw_*` is persisted for exactly this), or extend the muon mask to the companions' fits — are the owner's call; the arms and the cell table support all three offline.

**(b) The high tail is the Michel *object*, not the estimator.** The largest reading, `039349_81/54` at 500.7 MeV, is an object of **19 segments / 19 pieces** on a 9.4 cm arm holding 2962 cells; the chain's own `michel_ke_best` there is **206.5 MeV**, four times the Michel endpoint. The estimator is faithful to the object it was handed — the object is mis-assembled. Of the 8 PDVD readings above the endpoint, 5 sit on candidates whose chain energy is *also* above it. The remainder (`039252_3/74` 63.5 vs 33.1, `039349_5/64` 75.8 vs 46.2) are attached/bridged Michels where the estimator collects 1.5–2× the trajectory's charge — the intended effect, unverified against truth.

44 of 164 PDVD and 40 of 124 PDHD candidates sit outside `[0.5, 2]` or above the endpoint; §7's table and `d81_tail.py` list every one by name.

**Therefore both knobs ship default OFF in C++ and are not written into either production driver.** One key each turns them on (`michel_q2d`, `michel_q2d_cells` in `stm_michel_knobs`); §4's proof script `d81_proofs.sh` is written and ready for the day the owner chooses a definition. Nothing in this round can move a verdict — the OFF gate is byte-identical and the ON arms are bit-identical on every pre-existing branch — so flipping later costs one config commit and a confirmation arm, not a re-validation.

## 9. Open points for the owner

1. **The cross-shared definition (a) above** — the one decision that has to be made before this branch means the same thing on both detectors.
2. **The charge-only (`conn_type 3`) class reads 0.55 of the chain.** The chain sums the companion cluster's whole blob charge; the estimator admits cells by the fitter's blob-coverage predicate at zero tolerance. The two are different sets; which is wanted is a definition, not a bug.
3. **No truth comparison.** Everything here is estimator-vs-estimator. The Michel endpoint (52.8 MeV) is the only external anchor used, and 8 PDVD / 5 PDHD readings exceed it.
4. **`michel_q2d_cells` in production costs +22 % on `tracking-pr.root`.** Doc 64's precedent kept bulk diagnostic rows OFF in production and ON in the scan TLA; the owner asked for these measurements, so either is defensible.
5. The 0.6 cm association radius (`michel_q2d_dis_cm`) and the 30 cm STM footprint window (`michel_q2d_stm_window_cm`) are knobs, unswept.
6. *(Added 2026-09-11, owner's question after doc pdvd/88.)* **The cell selection still depends on the Michel's segmentation.** The measured charge replaces the fitted dQ, but a cell counts only if it lies near a Michel segment's associated points. Charge PR never gave a Michel segment is outside the sum: a dropped residual, blob points partitioned to the muon's last segment, or a Michel with no segment at all. That is not the stated intent (the trajectory only for the muon subtraction). A region-based definition and a first measurement are doc 78 action item 9.

   *Answered (doc pdvd/95, 2026-09-11): the region definition is built, gated and FLIPPED in PDVD production.* `michel_q2d_region_cm: 10.0` sums every cell within that radius of the stop whatever role claimed it, so the segmentation no longer selects the cells; the dropped population is real and large (role-0 cells, median **1495** per candidate). This doc's own estimator (`michel_ke_q2d`) ships alongside it — the two are persisted side by side, and on found Michels they read 29.98 and 34.65 MeV against `michel_ke_best`'s 23.23. **Doc 95 does not resolve open points 1–3 above:** the cross-shared substitution is untouched and is still why PDHD stays OFF, the charge-only class is unchanged, and there is still no truth anchor. It adds one: a ~2 MeV phantom is irreducible at a real Bragg peak, with a tail to 48 MeV on 18 % of no-Michel stoppers.
