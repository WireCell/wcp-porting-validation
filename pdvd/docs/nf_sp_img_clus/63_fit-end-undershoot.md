# 63 — T5: the fit-end undershoot

**Status (2026-09-10, overnight). `absorb_bragg_stub: true` is PDVD
PRODUCTION.** The existing default-OFF knob (doc pdhd/03 §6.8), scored for
the first time on the 569-item record against the round-1 candidate
`d62bc`: it fires on 7 items, recovers **3 `is_stm` stoppers**
(`039253_0/110`, `039349_64/65`, `039349_66/78`), adds **0** `is_stm` FPs,
loses **0** TPs, and leaves the `michel_found` census identical. `is_stm`
census 149 / 9 / 119 → **152 / 9 / 116** (efficiency 0.556 → 0.567), class F
5 → 3. The PDHD regression that kept it off does not reproduce on any PDVD
item. Both byte-identical gates PASS (PDVD 578/578 vs `d62bc`, PDHD 325/325
vs `d53h`). Doc 56 §6's three named owners of the undershoot (tip trim,
`end_point_limit`, terminal set) are each shown not to be the mechanism.

Doc pdvd/56 §8's T5 row: five items carry a > 1.67 MIP forward stub past the
fit end (class F, doc 55 §15.3); "which of the tip trim
(`TrackFitting.cxx:2609`), `end_point_limit`, or the terminal set owns
them; `absorb_bragg_stub` regressed on PDHD, so a new mechanism is needed".
This round establishes that none of the three named owners is the
mechanism, that the existing default-OFF `absorb_bragg_stub` knob IS the
mechanism (built for exactly this in doc pdhd/03 §6.8 and never scored on
the record), and scores it.

Companion docs: pdvd/55 §15.3 (class F), pdvd/56 §6, pdhd/03 §6.8
(`absorb_bragg_stub`), pdvd/62 (T3 — two of the five items now carry a found
Michel through T3b).

---

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
wcbuild && ./build/clus/wcdoctest-clus            # 357/357 (no new knob this round)

cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img
W=$HOME/tmp/d63
SURVEY='survey_enable:true,survey_radius_cm:60.0,survey_max_len_cm:25.0'
R=pdvd/docs/nf_sp_img_clus/scripts/d53_run_arms.sh
ARM=d63vleg DET=pdvd SRC=d16vnu JOBS=8 PIN=$W/libpin PR_TLA="-S stm_michel_extra={$SURVEY}" $R
ARM=d63hleg DET=pdhd SRC=d16hnu JOBS=8 PIN=$W/libpin PR_TLA="-S stm_michel_extra={$SURVEY}" $R
ARM=d63a    DET=pdvd SRC=d16vnu JOBS=8 PIN=$W/libpin PR_TLA="-S stm_michel_extra={$SURVEY,absorb_bragg_stub:true}" $R

# byte-identical gates: d63vleg vs d62bc (the round-1 production candidate), d63hleg vs d53h
python3 pdvd/docs/nf_sp_img_clus/scripts/d51g_branch_census.py \
    --before 'pdvd/work/*_d62bc' --after 'pdvd/work/*_d63vleg' --before-arm d62bc --after-arm d63vleg --pts --out $W/g1
python3 pdvd/docs/nf_sp_img_clus/scripts/d51g_branch_census.py \
    --before 'pdhd/work/*_d53h' --after 'pdhd/work/*_d63hleg' --before-arm d53h --after-arm d63hleg --pts --out $W/g2

# the C++ numbers the classifier saw for the five class-F stubs (new DEBUG line, any arm)
python3 pdvd/docs/nf_sp_img_clus/scripts/d63_stop_arms.py d63vleg 039252_5/73 039349_66/78 039253_13/39 039349_18/33 039253_0/110

# the feature arm scored against the frozen record
cd pdhd/stm_michel_scan
./prep_stm_michel_scan.py --det pdvd --arm d63a --outdir $W/prep_d63a --sheetdir $W/sheet_d63a \
    --pin-tranche ../../pdvd/docs/scan/pdvd_stm_michel_scan_sheet.tsv
python3 census_score.py --prep $W/prep_d63a --baseline $HOME/tmp/d62/prep_d62bc --arm d63a --json $W/d63a.json
python3 ../../pdvd/docs/nf_sp_img_clus/scripts/d63_by_name.py $HOME/tmp/d62/prep_d62bc $W/prep_d63a
python3 census_score.py --check                                          # still "0 of 14 differ"
```

---

## 1. Three corrections to doc 56 §6 / T5

Doc 56 named three candidate owners of the undershoot. None is:

1. **The tip trim is geometric, and `:2609` is the wrong side.**
   `TrackFitting.cxx:2609` is the START-side pop loop of `examine_end_ps_vec`;
   the fit-END trimmer is the mirrored loop at `:2693`. Its criterion is
   three-plane `Grouping::is_good_point` at a hard-coded 0.2 cm radius (with
   `good_point_pitch_frac = 0` on PDVD) — support, not charge. A 2.4-MIP stub
   is not what it trims. The doc pdvd/38 island trim is inert
   (`end_trim_gap_len: 0`).
2. **`end_point_limit` cannot move the STM path end.** `do_single_tracking`'s
   final `organize_ps_path(segment, pts, low_dis_limit, 0)` (`:10162`) passes
   0, so the extension block (`:2871`) is dead on the final pass; the knob's
   only surviving role there is `dis_end_point_ext` in `dQ_dx_fit` — the LAST
   point's dx window (0.3 cm after the pass-2 halving). It is a last-point
   dQ/dx knob, not a path-geometry knob.
3. **The Steiner terminal set does not bound the path end.** `do_rough_path`
   anchors on the nearest `steiner_pc` points (terminals AND intermediate
   Steiner points) to `check_stm_conditions`' boundary/exit anchors, and
   `create_steiner_tree` inserts those extreme points into the terminal set
   unconditionally, un-charge-tested, after thinning (`SteinerGrapher.cxx:
   219-221`). Terminal density governs the route, not where it stops.

What the five items actually are (§2): **the stub IS fitted**, as a separate
PR segment hanging off the chain's stop vertex, and the chain (whose profile
the shape tests read) ends one segment short of it because
`stm_michel_classify_stop_arm` sees a hot (> `michel_mip_hi` 2.0), short arm
and returns `kOther` — neither a Michel nor a MIP-like continuation. That is
precisely the case `absorb_bragg_stub` (doc pdhd/03 §6.8, `CheckSTM_Michel.cxx`
`:1428-1431`: `kink < continuation_max_angle_deg 20°`, `len <=
delta_max_len_cm 8`, `mip > continuation_mip_hi 1.3`) was written for. It
was tried once, on PDHD 029107/1 cluster 113, where absorbing a 5.5 cm /
1.64 MIP stub slid the Bragg tail window onto the stub's fading tip and a
clean STM became `no_bragg` — and has been OFF, unscored, since.

## 2. The five class-F items on the current arm

The new `stop-arm` DEBUG line gives the numbers `stm_michel_classify_stop_arm`
actually saw on `d63vleg` (the round-1 production bag, absorb OFF). mip is
÷ `mip_dqdx_median` 47000, the normalisation the thresholds use; the absorb
condition is `kink < 20°`, `len <= 8 cm`, `mip > 1.3`:

| item | scan | stub | len cm | far cm | mip | kink ° | shower | terminal | absorbable |
|---|---|---|---:|---:|---:|---:|---|---|---|
| `039252_5/73` | STM_ONLY | 73013 | 4.20 | 0.0 | 3.01 | **20.6** | 1 | 1 | no — 0.6° over the angle |
| `039349_66/78` | STM_ONLY | 78041 | 1.76 | 0.0 | 2.98 | 11.2 | 1 | 1 | **yes** |
| `039253_13/39` | STM_MICHEL | 39009 | 2.40 | 15.7 | 2.54 | 24.6 | 0 | 0 | no — angle, and not a terminal (15.7 cm continues past it) |
| `039349_18/33` | STM_ONLY | 33010 | 1.28 | 0.0 | 3.30 | 32.9 | 1 | 1 | no — 33° |
| `039253_0/110` | STM_MICHEL | 110007 | 3.00 | 1.8 | 2.15 | 16.6 | 0 | 0 | **yes** |

All five are `kOther` (hot, so not a continuation; > `michel_mip_hi` 2.0, so
not a Michel), all five have `n_stop_arms 1`, `n_ext 0`. Doc 55 §15.3's
"the fit stops short" is therefore precisely: the PR fitted the Bragg stub
as its own segment, and the chain refused it. The offline kinks of the
planning probe (14.8 / 13.6 / 26.2 / 46.5 / 20.7°) were within a few degrees
of the C++ values for four items and 14° off for `039349_18/33`; the C++
numbers are the ones that count.

A sixth item the offline class-F census never flagged also carries an
absorbable stub: `039349_64/65` (STM_MICHEL, one of T3's class-D recoveries
that B did not reach) — see §3.

## 3. `absorb_bragg_stub` scored (`d63a` vs `d62bc`)

`n_stub_absorb` fires on **7 items** (14 `n_ext` in total on the arm; the other
7 are ordinary MIP continuations):

| item | scan | `is_stm` | reject before → after | contrast/expected | KS margin | what happened |
|---|---|---|---|---:|---:|---|
| `039253_0/110` | STM_MICHEL | 0 → **1** | `stop_near_boundary` → STM | 0.71 → 0.96 | 0.013 → 0.003 | stop moves 3 cm onto the stub, off the boundary margin; the Michel, bridged at 2.9 cm before, is now ATTACHED (`conn_type` 2 → 1, same 4.3 MeV) |
| `039349_64/65` | STM_MICHEL | 0 → **1** | `shape_flat` → STM | 0.70 → 0.91 | −0.033 → 0.050 | the absorbed stub supplies the missing rise; its Michel (found by T3b) unchanged |
| `039349_66/78` | STM_ONLY | 0 → **1** | `no_bragg`,`shape_flat` → STM | 0.57 → 0.70 | −0.085 → 0.000 | doc 55's scan 474: the 1.8 cm / 2.98 MIP piece the fit excluded is now the end |
| `039349_13/56` | THRU | 1 → 1 | STM → STM | 0.93 → 1.04 | 0.069 → 0.088 | already an `is_stm` FP; absorbing a 3.6 cm / 1.67 MIP stub makes it look more like a stopper, not less — unchanged verdict, named |
| `039349_47/68` | STM_ONLY | 1 → 1 | STM → STM | 0.93 → 0.99 | 0.058 → 0.005 | unchanged (correct) |
| `039349_55/31` | THRU | 0 → 0 | `no_bragg`,`shape_flat` (both) | 0.57 → 0.56 | −0.020 → −0.041 | unchanged (correct) |
| `039349_62/35` | THRU | 0 → 0 | → `shape_flat` only | 0.56 → 0.64 | −0.042 → −0.072 | `no_bragg` cleared, `shape_flat` holds; unchanged verdict |

Census: `is_stm` TP 149 → **152**, FP 9 → 9 (identical set), FN 119 → 116;
purity 0.943 → 0.944, efficiency 0.556 → 0.567. `michel_found` census
**identical** (TP 132 / FP 22 / FN 20 / TN 373): no Michel was orphaned by the
stop moving onto its stub — the plan's named hazard (`:1438` lets a stub
bypass the Michel guard) did not bite on this record, and on `039253_0/110`
the move improved the Michel's connection. Class F 5 → 3 (by name: the
`039253_0/110` and `039349_66/78` stubs are no longer "past the fit end" —
they ARE the end). Every other class count unchanged (A 9, C 22, D 20, E 5,
G 15, H 20, K 36, L 49/29). Michel attachment 225 role-3 unchanged (one
scan-michel segment moves no-role → swallowed, 47 → 46 / 9 → 10: on
`039349_64/65` the absorbed stub's neighbour). Pin residual unchanged.

**The PDHD failure mode does not reproduce on PDVD.** On the four fires
that did not recover an item, `contrast/expected` moves by −0.01, +0.06,
+0.08, +0.11 — none crosses the 0.6 bar downward. Nothing here needs the
peak-anchored tail window; T7 will still measure that mechanism on its own
merits, but not as a rescue for this knob.

**The three left.** `039252_5/73` misses the 20° angle by 0.6°,
`039253_13/39` by 4.6° (and its stub is not a terminal — the fit continues
15.7 cm past it, so absorbing it would not put the end at the peak anyway),
`039349_18/33` by 13°. Widening `continuation_max_angle_deg` to reach them
would be a threshold moved on the sample that motivated it (CLAUDE.md
§5.7) and is not done; they are the named residual of class F.

## 4. Gates

| gate | result |
|---|---|
| `./build/clus/wcdoctest-clus` | 357 / 357 (no new knob; a new counter branch only) |
| `d63vleg` vs `d62bc` (PDVD, `--pts`) | 578 / 578 identical point geometry, 0 role labels moved, 0 `is_stm` flips; new branch `n_stub_absorb` only — **PASS** (`$W/gate1_pdvd.txt`) |
| `d63hleg` vs `d53h` (PDHD) | 325 / 325 identical, 0 moved, 0 flips — **PASS** (`$W/gate2_pdhd.txt`) |
| binary pin | `libWireCellClus.so` md5 `ec0b4cb976e7` before and after every arm |
| `census_score.py --check` | 0 of 14 differ |
| flip-equivalence (post-round production bag vs round-1 file + `absorb_bragg_stub:true`) | **0 lines** |
| true OFF path (`absorb_bragg_stub:false` forced on both files) | **0 lines** |
| `abtest/compile_all_cfg.sh` + `cmp_cfg.sh`, before vs after the flip | 16 live jobs NORMDIFF 0, **OVERALL PASS** (SBND/uBooNE untouched; `CheckSTM_Michel` is not in their chains) |

## 5. Found on the way, not fixed

1. **Doc 56 §6's `:2609` citation is the START trimmer**; the END trimmer is
   `:2693`, and it is geometric. Doc 56 §6 and §9 get the correction.
2. **`end_point_limit` is a last-point-dQ/dx knob on the STM path**, not a
   path-geometry knob: the final `organize_ps_path` call passes 0. Its name
   suggests otherwise. Left as is; noted for anyone who tunes it expecting
   the path end to move.
3. **`absorb_bragg_stub` on PDHD** is still OFF and still has its one
   recorded regression (029107/1 c113). This round shows the failure mode
   is item-specific, not structural; a PDHD flip needs a PDHD scan record,
   which does not exist.
4. **`039349_13/56`** (THRU, `is_stm` FP on every arm since `d53v`) picks up an
   absorbed 1.67 MIP stub and looks more stopper-like. It is the same FP it
   was; if a future round tackles the 9 `is_stm` FPs, this one now carries
   `n_stub_absorb 1`.
5. The `stop-arm` DEBUG line is unconditional (206 lines on the 120-event
   arm); it costs nothing in the outputs and is the only record of the
   classifier's per-arm numbers.

## 6. Doc 56 update

T5 row marked done (`absorb_bragg_stub` flipped, 3 recovered, class F 5 → 3,
three named leftovers with their angles); §1 row 1's "0.6 cm step / Steiner
terminals" pointer to T5 gets the correction that neither the tip trim nor
the terminal set owns the undershoot; §6's bullet is corrected (`:2693`,
geometric); Order moves to T6; §9 gains the citation correction. Scripts
committed: `scripts/d63_by_name.py`, `scripts/d63_stop_arms.py`.
