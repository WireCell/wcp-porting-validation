# doc pr/109 — Near the vertex, which fit describes the measured 2-D charge better: exclusion ON or OFF? (2026-08-22)

**Status:** §1-§7 measurement only. **§9 (2026-08-22) answers the owner's "why is SBND worse"
question**: `update_association` keeps a 2-D cell only if its segment is *strictly closest of all
siblings in the cluster*, so the strip rate scales with the cluster's segment count — SBND reaches
26-37 segments per exclusion call in the calls that carry 78 % of its associations -- a regime
uBooNE never enters (max 9) -- strips 53.6 % of the associations within 3 cm of the vertex against
21.1 %, and ends with 21 % fewer near-vertex trajectory points against 1 %. Strip rate rises
monotonically with sibling count *within* each detector (Spearman +0.98 SBND, +0.53 uBooNE; calls
with a single segment strip exactly 0 %). The
SBND pattern-recognition knobs are implicated collectively, through the segment population they
produce, not individually; `fit_blob_coverage` is a secondary modulator (flips 2/6 events, removes
all of the bias deepening, leaves the stripping). No code changed in §9. **§8 (2026-08-22) fixes a
real bug** in the toolkit's
`T_proj_data` writer — the tree reported another cluster's prediction — and re-derives every
number below on the fixed arms. **No production/physics change**: the Bee, pctree and score
outputs are byte-identical across the fix (§8.5). The verdicts did not change; the table below is
the pre-fix record, and the post-fix one is §8.4:

| family | entries (event × region) | per-event sign test | median ΔU (post-fix) |
|---|---|---|---|
| **uBooNE toolkit** | 10 / 4 / 4 | 3 better, 2 worse of 5 | **−0.018** |
| **uBooNE prototype** | 9 / 0 / 2 | 4 better, 2 worse of 6 | **−0.036** |
| **SBND toolkit** | 1 / 10 / **22** | **0 better, 6 worse of 6** | **+0.037** |

The answer differs by detector and it is the same in both implementations, so it is not a port
defect. Pre-fix table (the record §4 was written from):

| family | entries (event × region) | per-event sign test | median ΔU = U_ON − U_OFF |
|---|---|---|---|
| **uBooNE toolkit** | 9 ON-better / 4 equivalent / 4 ON-worse | 4/5 ON-better | **−0.024** |
| **uBooNE prototype** | 9 ON-better / 0 equivalent / 2 ON-worse | 4/6 ON-better | **−0.035** |
| **SBND toolkit** | 2 ON-better / 8 equivalent / **23 ON-worse** | **6/6 ON-worse** | **+0.025** |

`U = Σ|y − ŷ| / Σy` is the unexplained fraction of the *measured* 2-D charge inside the 2-D shadow
of a 3 cm sphere around the vertex/junction; lower is better. So:

Against the pre-registered bar (same sign on ≥ 5/6 events) the two answers are **not equally
strong, and the doc says so in both places**: SBND *reaches* it (6/6); uBooNE does *not*
(4/5 toolkit, 4/6 prototype) and rests on the entry counts and the medians instead.

- **On uBooNE the exclusion fit describes the measured 2-D charge better** — in the prototype
  *and* in the toolkit, by the same amount (−2.4 / −3.5 pp of unexplained charge, ~72 % of anchors).
  That is a second, independent confirmation that the dQ/dx fit has no bug: the two codes agree
  on which trajectory the data prefers, not merely on the numbers each produces.
- **On SBND the exclusion fit describes it worse** — +2.5 pp unexplained charge,
  **every one of the 6 events**, and the same sign in χ²/N (+2.4). This is the owner's claim, now
  measured against the data rather than inferred from the vertex efficiency: at the SBND operating
  point the exclusion-OFF trajectory is the better description of the 2-D charge.
- The verdict survives removing coverage entirely (restricting to cells both arms predict > 0):
  9/5/4, 9/0/2, 2/11/**20**, and the per-event sign test is unchanged.

Owner (2026-08-22): "which fit describes the 2D charge measurement better? … we want to check the
case near the vertex … whether we can conclude the exclusion fit can describe the measured 2D
charge better, or at least not worse … another validation of the no bug in dQ/dx fit, and confirm
my claim of trajectory fitting is better with no exclusion fit."

## 0. Repro block

```bash
# SBND arms (6 events, ON = production fit_exclusion=true, OFF, and ON2 = repeat of ON):
sbnd_xin/scripts/pr109_sbnd_arms.sh              # -> work-pr109-{on,off,on2}-nuecc48
# uBooNE arms: NONE run -- the pr/108 arms already carry the readout
#   qlport/scripts/sweep/pr108_wct_{on,off}/<idx>_<ev>/track_com_5384_<ev>.root
#   qlport/scripts/sweep/pr108_wcp_{on,off}/nue_5384_<sr>_<ev>.root
# the whole grid (3 families x 6 events x 2 arms x 3 box sizes x 2 anchor arms):
sbnd_xin/scripts/pr109_run_all.sh /home/xqian/tmp/pr109                    # primary
sbnd_xin/scripts/pr109_run_all.sh /home/xqian/tmp/pr109 _cov --common-pred # coverage removed
python3 sbnd_xin/scripts/pr109_summary.py /home/xqian/tmp/pr109/{ub_wct,ub_wcp,sbnd}.tsv
# one region by hand:
python3 sbnd_xin/scripts/pr109_2d_resid.py --arm ON=A.root:wct --arm OFF=B.root:wct \
        --sigma-from OFF --anchors-from OFF --anchors-from ON --box-cm 3.0 --tag evt
# full output: docs/pr/109_2d-residual-tables.txt
```
Binaries: toolkit `local/lib/libWireCellClus.so` 2026-08-21 16:36 (the pr/108 §10 binary, no
source change this round); prototype `prototype_base/build/pid/*` same era. No knob was added,
no default changed, nothing was flipped.

## 1. The instrument — it already existed

The dQ/dx system minimises |data − R·x|² over the measured 2-D charge, and **both implementations
already persist the per-cell answer**, so this round needed no instrumentation at all:

| | prediction | persisted as |
|---|---|---|
| toolkit | `pred_data_{u,v,w}_2D = R*pos_3D`, `TrackFitting.cxx:7386-7388` → `fill_fitted_charge_2d` (:1131-1170) | tree `T_proj_data` (`root/src/SbndPrMagnifyTrackingVisitor.cxx:233-330` and the uBooNE/SBND analogues) |
| prototype | same algebra, `PR3DCluster_multi_dQ_dx_fit.h:842-845` → `proj_data_{u,v,w}_map` (:847-863) | tree `T_proj_data` (`pid/apps/wire-cell-prod-nue-port.cxx:3249-3320`) |

Both trees carry the identical six branches `cluster_id / channel / time_slice / charge /
charge_err / charge_pred`. Anchors come from `T_rec_charge`, which on both sides carries
`x y z flag_vertex cluster_id sub_cluster_id` **and** the per-point projections `pu pv pw pt`.

**Frames** (verified, uBooNE 5384 6505 both sides): `pu/pv/pw` *are* global channel numbers and
`pt` *is* the `T_proj_data` time_slice, scale 1 — the prototype carries the display `+0.5` on all
four, removed here. `T_bad_ch` start/end are **ticks** (÷4 for uBooNE/SBND slices). The local
affine map (x,y,z) → (pu,pv,pw,pt) fitted per arm on its own points reproduces the projections
with residual **0.0000** wires/slices, so the 3 cm sphere's 2-D shadow is exact.

## 2. Metric and the decision rule — fixed before the numbers

Per (event × region × anchor × arm), over cells of the 2-D shadow of a 3 cm sphere with
`charge > 0`, not on a dead channel (`T_bad_ch`), owned by a single cluster:

| symbol | definition |
|---|---|
| **`U`** | **`Σ|y−ŷ| / Σy` — unexplained charge fraction (headline)** |
| `B` | `(Σŷ−Σy)/Σy` — signed bias |
| `χ²/N` | `Σ((y−ŷ)/σ)²/N`, one canonical σ per cell = the OFF arm's `sqrt(charge_err² + (charge·rel)² + add²)`, the fit's own whitening (rel 0.075 ind / 0.05 col, add 0 / 300 — identical in `qlport/uboone_track_fitting.json` and `TrackFittingPresets.h:31-34`) |
| `Ncol` | fit points inside the sphere — the dof asymmetry, reported, **never divided out** |

`U` and `B` are parameter-count-insensitive; χ²/N is not (exclusion changes how many trajectory
points sit near the vertex — pr/107 counted 443+101 — and the regulariser means `ndf ≠ N − Ncol`),
which is why χ² is reported *beside* `N` and `Ncol` rather than as χ²/ndf.

**Test unit** (fixed before any table): one entry per **(event, region)**, `U` pooled over the
three planes; regions from the two anchor arms merged when their anchors are within 2 cm.
Over the variants v = {2 anchor arms} × {box 2.25, 3.0, 3.75 cm}: `ΔU = median_v (U_ON − U_OFF)`,
`Bsys = (max_v − min_v)/2`. **ON better** if `ΔU < −Bsys`, **ON worse** if `ΔU > +Bsys`, else
**equivalent**; a family verdict needs the same sign on ≥ 5/6 events.

## 3. Controls

1. **Same data — exact.** Over *all* cells, every one of the 18 ON/OFF arm pairs has the identical
   cell set and `max|Δy| = 0` (uBooNE WCT/WCP 6 events each, SBND 6 events). The two arms are
   scored against literally the same measurements.
2. **The arms are really different.** SBND: `pr85_hash_gate.py work-pr109-on work-pr109-off`
   → **FAIL** (as intended; first divergent archive evt 256587 `mabc-pr.zip`), and the TLA is
   value-passed (`run_pr_chain_batch.sh:528` → `fit_exclusion=false`, grep quoted per arm).
   Note its inline comment "SBND config default FALSE" is **stale**: production is
   `fit_exclusion = true` (`wct-pr-perevt.jsonnet:196`).
3. **Rerun band = 0.** `work-pr109-on2` (a repeat of ON) vs `work-pr109-on`: hash gate
   **PASS 12/12 byte-identical**, and `charge_pred` is bit-identical on every cell of all 6 events
   (max |Δŷ| = 0). The known last-writer-wins nondeterminism of the merged `m_fitted_charge_2d`
   (`PrDisplayDump.cxx:1130-1150`, 10.2 % of cells on SBND 18255/388) **did not fire** here under
   `setarch -R`. Shared cells are 0-25 % of the map (SBND 7-18 %, uBooNE 0-25 %) and are excluded from
   the primary numbers.
4. **Coverage removed.** Repeating everything on cells predicted > 0 by *both* arms leaves the
   verdict unchanged (§0 table), so the SBND result is not "ON abandons more charge".
5. **Excluded, with cause: uBooNE 6528 toolkit.** Its saved `T_proj_data` covers 5.0 % of the
   cluster's cells (Σŷ/Σy = 0.027) in **both** arms, so it carries no information about either.
   The fit itself is fine — `T_rec_charge` cluster 19 has 178/186 points with q > 0, Σq 7.64 M vs
   the prototype's 7.25 M — this is an output-persistence defect: `fill_fitted_charge_2d` keeps
   only the *last* snapshot per cluster ident and the log shows it discarding one
   ("cluster ident 1909 is shared by two live clusters"). Reported, not fixed here.

## 4. Results

> **Superseded in part by §8** (2026-08-22). Every number in this section was computed from a
> `T_proj_data` whose `charge_pred` was corrupted by a cross-cluster overwrite in the toolkit's
> writer. The section is kept verbatim as the record; §8.4 carries the same table recomputed on
> the fixed arms. The verdicts did not change — they sharpened.

Verdict counts and the per-event sign test are in the Status table; the full per-entry tables
(with `Bsys`, `ΔB`, `N`, `ΔNcol`) are in `docs/pr/109_2d-residual-tables.txt`. Per plane
(median over all anchors, box 3 cm):

| family | ΔU (U) | ΔU (V) | ΔU (W) | ΔU pooled | Δ(χ²/N) pooled |
|---|---|---|---|---|---|
| uBooNE toolkit | −0.016 | −0.012 | −0.013 | **−0.024** | −1.73 |
| uBooNE prototype | −0.023 | −0.023 | −0.016 | **−0.035** | −2.47 |
| SBND toolkit | +0.023 | +0.001 | +0.036 | **+0.025** | +2.36 |

χ² moves the same way as `U` in all three families, and on SBND the damage is concentrated in
**U and W**, with V neutral.

Against the pre-registered bar: SBND reaches it (**6/6 events ON-worse**). uBooNE does *not* reach
the ≥5/6 bar for "ON better" (4/5 toolkit, 4/6 prototype) — the honest statement there is
**"ON is not worse, and is better on ~72 % of anchors in both implementations"**, with the entry
counts (9 vs 4 and 9 vs 2) and the medians pointing the same way in both codes.

Absolute levels differ and are context, not verdict: median `U` on the ON arm is 0.32 (prototype
uBooNE), 0.52 (toolkit uBooNE), 0.62 (toolkit SBND). The toolkit's saved projection also covers
less of the total event charge on SBND (Σŷ/Σy 0.38-0.51 over the whole map) than the prototype
does on uBooNE (0.46-0.92; toolkit uBooNE 0.50-0.84 with 6528's 0.03 excluded) —
mostly clusters that never got a final fit, which is why the comparison is done *inside* the
vertex box and repeated with coverage removed.

### 4b. Is the SBND result just a coverage statement? No.

`U` is a ratio to the *measured* charge, so an arm that simply predicts nothing over more cells
scores worse without fitting worse. Two discriminators, both on the same tables:

| family | median ΔU | median Δ`uncov` | ΔU − Δ`uncov` | `uncov` ON / OFF |
|---|---|---|---|---|
| uBooNE toolkit | −0.0244 | −0.0006 | −0.0222 | 0.118 / 0.131 |
| uBooNE prototype | −0.0352 | −0.0007 | −0.0081 | 0.003 / 0.005 |
| SBND toolkit | +0.0249 | +0.0052 | **+0.0124** | 0.218 / 0.195 |

So on SBND the uncovered charge accounts for about a fifth of the effect; the rest is genuine
residual on cells both arms do predict, and only 22 % of entries flip sign when it is removed.
Second: ΔU is **flat in coverage** — splitting the SBND entries at the median in-box coverage
(Σŷ/Σy = 1+`B`) gives median ΔU +0.0260 on the low-coverage half (coverage 0.35, n = 22) and
+0.0242 on the high-coverage half (coverage 0.74, n = 23), corr(coverage, ΔU) = −0.085. The
artifact signature (ON-worse entries clustering at low coverage) is absent. The `--common-pred`
run in §0 is the third form of the same check.

What the table *does* show is that SBND leaves ~20 % of the near-vertex measured charge
unpredicted in **both** arms, against 0.3 % for the prototype on uBooNE — a large common
shortfall that this round measures but does not explain (see §5).

## 5. What this settles — and what it does not

**Settles.** (i) The exclusion fit is not a broken dQ/dx: on uBooNE both implementations agree
that the exclusion-ON trajectory describes the measured 2-D charge *better*, by the same amount
and with the same per-plane pattern. Combined with pr/108 Test A (dQ/dx is association-independent,
max|ΔdQ| = 0) and §10 (the fit is stiff where every plane constrains it), that closes the
"is the fit buggy" line for a third time, from the data side. (ii) On SBND the owner's claim is
supported against the 2-D measurement at the registered bar: the exclusion-OFF trajectory
explains more of the charge near the vertex on 6/6 events, in U and W, with χ² agreeing and with
the coverage discriminators of §4b passed. The effect is modest — +2.5 pp of unexplained charge,
of which ~0.5 pp is coverage — so it is a consistent direction, not a large one.

**Does not settle.** Why the two detectors disagree on the sign. The ~20 % of near-vertex charge
SBND leaves unpredicted in both arms is investigated in **§7** (owner request) — it is a static,
induction-plane, per-cell condition that largely cancels in the ON/OFF comparison, and its cause is
only partly identified. The measurement says it is not the port
(the prototype and the toolkit agree on uBooNE, and SBND has no prototype arm to compare with).
Candidates, all consistent with the numbers above: SBND's finer effective sampling and 2-face
geometry putting more prongs inside one coupling window; the `dis_end_point_ext` / close-wire
weights being uBooNE-tuned; and pr/108 §5's observation that the SBND ON trajectory carries 13 %
less charge within 1 cm of the target vertex than OFF — which is the same effect seen here from
the data side.

**Caveats stated up front.** `Ncol` differs between arms (median ΔNcol −1 on SBND, 0 on uBooNE),
so χ²/N is reported beside it and never as χ²/ndf. `T_proj_data` holds each cluster's *last*
dQ/dx fit, not guaranteed to be the same algorithmic stage in both implementations — that weakens
WCP-vs-WCT *absolute* comparison but not the ON-vs-OFF comparison inside each. The toolkit casts
`charge_pred` to `int`, truncating small and negative predictions toward zero.

## 6. Open / owner decision

The SBND result is a measured argument for the exclusion-OFF trajectory at the SBND operating
point, but `fit_exclusion` is SBND **production ON** (`wct-pr-perevt.jsonnet:196`) and pr/98-§10
shipped it that way for pattern-recognition reasons. Nothing is flipped here. Options, none taken:
(a) leave production as is and use this as the physics justification for a DL charge proxy that
does not depend on the split (pr/108 §9 lever (c)); (b) re-tune the near-vertex weights
(`dis_end_point_ext`, close-wire/dead weights) for SBND behind a default-OFF knob and re-run this
same measurement as the acceptance test — it is cheap and now automated; (c) revisit
`fit_exclusion` for SBND with a full nueCC48 + mcp1k arm pair, judged on this metric *and* on the
vertex/selection census, since the two have disagreed before.

> **Root cause superseded by §8** (2026-08-22): the answer is a cross-cluster overwrite in the
> toolkit's `T_proj_data` writer, not a per-cell condition. §8.6 lists what stands and what falls.

## 7. Why does SBND leave ~20 % of the near-vertex charge unpredicted? (owner 2026-08-22: "uBooNE and SBND share the same toolkit code … investigate, I do not need a fix yet")

Repro: `scripts/pr109_uncov_probe.py --arm L=<root>[:wcp] --box-cm 3.0` (per-cell classification and
the distance profile); the `flag` column comes from one extra arm run with `PR_EXTRA_STAGES=pr_display`
(`work-pr109-dbgflag-on-nuecc48`, the proj section of `calib-pr-evt46363.json`) because `T_proj_data`
does not carry it.

**First correction to the framing: the larger split is implementation, not detector.** Median
uncovered charge in the 3 cm box is **0.3 % (prototype, uBooNE)**, **12 % (toolkit, uBooNE)**,
**20 % (toolkit, SBND)**. Same detector, same event, different implementation already costs a
factor 40; the detector change then adds a factor 1.7. Per event the box values run 0.1-28 %
(toolkit uBooNE), 0.1-15 % (prototype uBooNE) and 3-47 % (toolkit SBND).

### 7.1 What the uncovered cells are

| observable | SBND (toolkit) | uBooNE (toolkit) | uBooNE (prototype) |
|---|---|---|---|
| uncovered cells inside the ±10 wire/slice coupling window | 98-100 % | 100 % (except 6805, see below) | 100 % |
| signed offset (nearest fitted point − cell), **covered** cells, U/V | ≈ 0.0 wires, IQR ±0.6 | ≈ 0.0, IQR ±0.75 | ≈ 0.0, IQR ±0.75 |
| same, **uncovered** cells, U/V | **≈ 0.0, IQR ±0.6** | ±2.5 (genuinely off-track) | ±2.5 |
| uncovered fraction at distance < 1 wire *and* < 1 slice | **41-43 %** | **0 %** | **0 %** |
| plane structure of that near-track uncovered set | **U 65 %, V 48 %, W 0 %** | — | — |
| same cells uncovered in the ON and OFF arms | 84-93 % | 99 % | — |

Read together: on SBND the unpredicted cells are **not** a lateral tail the Gaussian response fails
to reach, and not charge the trajectory never visits — they sit at exactly the same offsets from the
trajectory as the predicted ones, **interleaved with them**, on the two induction planes only, and
they are largely the *same cells* whichever arm is run. That last line also means this shortfall
mostly cancels in ΔU, consistent with §4b (Δ`uncov` is a fifth of ΔU).

### 7.2 What it is not

- **Not the dead-channel list.** Only 0.1 % of the uncovered cells sit on a channel named in
  `T_bad_ch` (SBND 46363: 98 rows, none of them in the box).
- **Not the `int` cast.** `charge_pred` is `int32` in *both* implementations' `T_proj_data`, and
  the charges are 10³-10⁵, so truncation cannot make a cell zero.
- **Not a units or frame error.** SBND `pt` is in slices exactly like `time_slice` (point range
  107-716 vs cell range 496-600 for the main cluster), the local affine map has residual 0.0000,
  and the two `(apa, face)` groups SBND writes are (0,0) and (1,0) — both face 0, no wire-index
  collision, 0 duplicate `(channel, time_slice)` keys inside a cluster.
- **Not uBooNE-tuned response widths.** SBND has its own fitted parameter file
  (`cfg/pgrapher/experiment/sbnd/sbnd_track_fitting.json`): `ind_sigma_u_T` 0.484 vs uBooNE's
  0.363, `ind_sigma_v_T` 0.806 vs 0.604, `col_sigma_w_T` 0.094 vs 0.113, `add_sigma_L` 2.49 vs
  1.57, DL/DT 4.0/8.8 vs 6.4/9.8 — 38 of the 44 keys are shared, 6 differ, and the ones that
  differ are exactly the smearing widths, derived from SBND's own SP filters.
- **Not trajectory sampling.** The fitted step is 0.60 cm on both detectors, i.e. a median
  1.0-1.6 wires and 0.5-1.6 slices between consecutive points (p95 ≤ 2.0 wires) — small compared
  with the ±10 wire window, so the window anchoring at `centers_U.front()`
  (`TrackFitting.cxx:6806-6812`) cannot be dropping cells at this spacing.

### 7.3 What it most likely is — and the one measurement that would settle it

Everything above leaves a **per-cell** switch inside the response fill, not a geometric one. The
fill inserts a response entry only when `value > 0 && row.charge > 0 && row.flag != 0`
(`TrackFitting.cxx:6814-6820`, and the V/W twins at :6839 / :6864); `fill_fitted_charge_2d` then
writes `pred_charge = 0` for any cell failing `charge > 0 && flag != 0` (:1133-1140). Two candidates
remain, and they are not exclusive:

1. **Dead-*region* fillers (`flag == 0`), which are not the same thing as dead channels.**
   `prepare_data` spreads dead-channel-blob charge into cells that carry `flag = 0`
   (`TrackFitting.cxx:906-912`). The display dump — the only output that keeps the flag — shows
   **6196 of 16759 cells (37 %) with `flag = 0` on SBND 46363, 70 % of the V-plane cells**, every
   one of them unpredicted by construction. They carry only 4.3 % of the charge, so they explain
   the V-plane *cell* count but not the missing *charge*.
2. **Live induction cells that still get `value == 0` or are otherwise skipped.** On U the
   near-track uncovered cells are `flag = 1` (live) in the display dump, and their neighbours at
   the same offset are predicted. Nothing measurable from the outputs distinguishes them.

Settling (2) needs the per-cell instrument this round deliberately did not build: an env-gated dump
at the fill site of `(apa, face, plane, wire, time, charge, flag, value, row)` for every cell the
loop visits, on both implementations — the Phase-2 dump described in the plan. Until that exists,
the honest statement is: **the near-vertex shortfall is dominated by a static, induction-plane,
per-cell condition that both arms share, of which dead-region fillers are a demonstrated part and
the remainder is unidentified.** It is larger than the ON/OFF effect this doc measures, it mostly
cancels in the ON/OFF comparison, and it deserves its own round.

One uBooNE case belongs with it: event **6805 toolkit**, the only uBooNE entry where 75 % of the
uncovered charge sits on a cluster with **no fitted points at all** (distance = ∞) — a different
mechanism (a cluster that was never fitted) from the interleaved SBND pattern.

---

## 8. The toolkit's unpredicted near-vertex charge was a SAVING bug — found, fixed, and §7 superseded

Owner, 2026-08-22, after §7: *"I think we are very close to figure out what's going on … take the
MicroBooNE case and compare the toolkit and prototype implementation of the track trajectory and
dQ/dx fitting to figure out the issue. I wonder if it is a bug in the actual fitting (e.g. getting
the data integrated) or a bug in presenting or saving the predicted charge. If it is a bug, we
should fix them."*

**Answer: presenting/saving. The fit integrates the same data in both implementations; the tree
that reports the prediction did not report the fitting cluster's own answer.** Fixed. On the
owner's instruction the fix is unconditional (no knob) — see §8.5 for the gate that replaces the
byte-identical bar that choice gives up.

### 8.0 Repro

```bash
# toolkit fc3f16bf + the fix; wcbuild; ./build/clus/wcdoctest-clus  (228/228, incl. 2 new cases)
#
# TRAP, cost one arm: the uBooNE TLA is compared with == "true"
# (uboone-mabc.jsonnet:1513), so QL_FIT_EXCLUSION=1 SILENTLY MEANS false.
# Always read the value back from the job's own log rather than the env:
#   grep -o "fit_exclusion=[a-z]*" sweep/<arm>/<idx>_<ev>/wct_5384_<ev>.log
cd qlport/scripts
QL_FIT_EXCLUSION=true  ./sweep_5384.sh pr109e_wct_on  6   # post-fix, exclusion ON   (verified true)
QL_FIT_EXCLUSION=false ./sweep_5384.sh pr109e_wct_off 6   # post-fix, exclusion OFF  (verified false)
QL_FIT_EXCLUSION=1     ./sweep_5384.sh pr109b_bare_on 6   # PRE-fix binary, exclusion OFF -- gate baseline
QL_FIT_EXCLUSION=1     ./sweep_5384.sh pr109b_wct_on  6   # first post-fix build,  exclusion OFF   } same-config
QL_FIT_EXCLUSION=true  ./sweep_5384.sh pr109b_wct_on2 6   # ALSO landed exclusion OFF (the trap)    } repeat pair
# SBND, same 6 events and the same Q/L baseline as §0:
sbnd_xin/  SBND_FIT_EXCLUSION=true|false PR_JOBS=6 \
    ./run_pr_chain_batch.sh work-nuecc48-ql0819 work-pr109e-{on,off}-nuecc48 data \
        10550 46363 81597 360535 256587 433451
# the grid and the rule, re-pointed at the post-fix arms:
sbnd_xin/scripts/pr109b_run_all.sh /home/xqian/tmp/pr109e
python3 scripts/pr109_summary.py /home/xqian/tmp/pr109e/{ub_wct,ub_wcp,sbnd}.tsv
python3 scripts/pr109_uncov_probe.py --arm A=<root>[:wcp] --box-cm 3.0     # §7's probe
# gates (both sides of each pair at the SAME fit_exclusion):
python3 scripts/pr85_hash_gate.py work-pr109-on-nuecc48 work-pr109e-on-nuecc48
diff qlport/scripts/sweep/pr109b_bare_on/hashes.txt qlport/scripts/sweep/pr109e_wct_off/hashes.txt
```

The `pr109b_*` and `pr109c_*` uBooNE arms are earlier post-fix builds (before, and midway
through, §8.3.1); `pr109e_*` are the arms every number below is quoted from. All are kept.

### 8.1 Stage-by-stage: the fit is not the problem

`dQ_dx_multi_fit`, toolkit (`clus/src/TrackFitting.cxx`) against prototype
(`prototype_base/pid/src/PR3DCluster_multi_dQ_dx_fit.h`):

| stage | prototype | toolkit | verdict |
|---|---|---|---|
| 2-D data map | `prepare_data` (`PR3DCluster_trajectory_fit.h:1804`): cluster projection (flag 0/1/2) **+** the good-channel rectangle (flag 3) | `prepare_data` (`:762`): good-channel rectangle over the cluster bbox ±5 wires / ±20 ticks (flag 1; flag 2 with charge zeroed if q<0) + dead-blob spread (flag 0) | **equivalent in size**: 6528 `n2d` = 836/627/838 (WCP) vs 879/627/825 (WCT) |
| R fill window | `\|wire − centers_U.front()\| ≤ 10 && \|t − centers_T.front()\| ≤ 10` slices (`:371`) | wire-indexed `round(centers_U.front()) ± 10`, `\|row.time − centers_T.front()\| ≤ 10·ntick` (`:6807`) | equivalent |
| fill guard | `value>0 && charge>0 && flag!=0` (`:376`) | identical (`:6815/6840/6865`) | equivalent |
| regulariser | multi-fit constants 0.25/0.75, λ=0.0008 (`:762,793`); the *single*-fit file uses 0.15/0.45, λ=0.0005 | `m_params` carries the single-fit values and the code scales them ×5/3 and λ ×8/5 at `:7266,7315` | **not a bug** (checked before claiming one — M15) |
| **data actually integrated** | — | — | per-position coupling counts from the pr/108 §9 dumps agree: 6528 mean `cu/cv/cw` = **40.9/30.5/37.3** (WCP) vs **39.2/33.2/35.4** (WCT), zero-coupling positions 0/0/0 on both |
| **saving** | `proj_data_{u,v,w}_map` is a **per-cluster** member; the app writes one row per cluster straight out of it (`wire-cell-prod-nue-port.cxx:3272-3320`) | the writer read the **merged** map and tagged cells by **blob ownership** | **the defect** |

### 8.2 The three defects, all in the save path

Nothing in reconstruction reads the merged map: its only consumers are the three Magnify tracking
writers, `PrDisplayDump::dump_proj`, and `TaggerCheckSTM`'s `stm_fit` record — all output/display.

- **S1 — cross-cluster overwrite.** `assemble_fitted_charge_2d()` (`TrackFitting.cxx:1192`)
  flattens every per-cluster snapshot into one cell map, last-writer-wins in **ascending cluster
  ident order**. A satellite cluster holds a main-cluster cell inside its own padded bounding box
  and predicts **0** there; having the higher ident it writes last and wins. That is §7's whole
  signature: damage concentrated near the vertex (where satellites are), cells interleaved with
  covered ones at the same offset, identical in both arms, and induction-heavy — in the collection
  view neighbouring clusters rarely overlap, which is why §7 measured **U 65 % / V 48 % / W 0 %**.
- **S2 — ownership tagging.** `write_proj_data` emitted a row for every cluster owning a *blob* at
  the cell (`fc.clusters`, from `global_rb_map`), carrying whatever prediction survived the merge.
  A row labelled cluster A could therefore be entirely cluster B's answer.
- **S3 — ident-collision discard.** The snapshot store was
  `std::map<Facade::Cluster*, …, PR::ClusterPtrCmp>` — keyed by **ident** — so two live clusters
  sharing an ident compared equal and the earlier snapshot was discarded whole. It fired on uBooNE
  **5384-6528** (twice) and cost the main cluster its entire prediction: cid 19, 2141 cells,
  Σpred = 0. That is the event pr/109 §4 excluded "with cause".

Measured before the fix (`T_proj_data`, main cluster, whole map, ON arm):

| event | arm | cid | cells | cov | Σŷ/Σy |   | event | arm | cid | cells | cov | Σŷ/Σy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6505 | WCT | 14 | 5214 | 0.991 | 0.954 | | 6650 | WCT | 28 | 2178 | 0.817 | 0.773 |
| 6505 | WCP | 14 | 6067 | 0.865 | 0.845 | | 6650 | WCP | 28 | 2494 | 0.848 | 0.850 |
| 6528 | WCT | 19 | 2141 | **0.000** | **0.000** | | 6805 | WCT | 23 | 1525 | 0.786 | 0.643 |
| 6528 | WCP | 19 | 2322 | 0.967 | 1.001 | | 6806 | WCT | 25 | 853 | 0.920 | 0.579 |
| 6532 | WCT | 20 | 4766 | 0.613 | 0.483 | | 6806 | WCP | 26 | 1376 | 0.868 | 0.405 |
| 6532 | WCP | 20 | 8257 | 0.561 | 0.543 | | SBND 46363 | WCT | 19 | 7833 | 0.704 | **0.437** |

### 8.3 The fix

Two commits, separable:

1. `clus/inc/WireCellClus/TrackFitting.h`, `clus/src/TrackFitting.cxx`,
   `root/src/{SbndPr,Sbnd,Uboone}MagnifyTrackingVisitor.cxx` — new
   `TrackFitting::ClusterFitted2D` + `get_cluster_fitted_charge_2d()`; `write_proj_data` emits
   **one row per fitted cluster, out of that cluster's own snapshot**, tagged with its ident (the
   prototype's semantics). The merged-map + blob-ownership path is kept as a documented fallback
   for a fitter that never ran with a cluster filter (the STM holder fed by
   `merge_fitted_charge_2d`), so the tree can never come out empty.
2. Same files — the snapshot store becomes a capture-ordered `std::vector<ClusterFitted2D>` keyed
   on the cluster **pointer** (refits replace in place), and `assemble_fitted_charge_2d()`
   stable-sorts by ident before merging. That keeps the determinism guarantee ident order was
   introduced for (doc pr/28 §4.3) while ending the S3 discard.

**§8.3.1 — one row per 2-D cell, restored.** Emitting each cluster's whole snapshot exposed a
`prepare_data` quirk the old writer had hidden: the dead-region filler writes **one entry per
tick** (`TrackFitting.cxx:896` iterates `slice_index_min .. slice_index_max` one at a time, and
those bounds are in ticks), so `nticks_per_slice` entries divide down to the same `time_slice`.
The old writer dropped those cells entirely — they have no blob owner, so its `fc.clusters` loop
emitted nothing — and the first post-fix build therefore put **duplicate `(channel, time_slice)`
keys in a row: 33 % of uBooNE cells, 6.9 % of SBND cells, against 0 % pre-fix and 0 % in the
prototype**, breaking the one-row-per-cell precondition §0 relies on (an analysis that builds
`cells[(ch, ts)]` silently keeps one of them). Caught by review, not by the gate. The three
writers now accumulate into an ordered per-cell map before filling the branches. Two rules make
that aggregation safe, both added after review found the naive "sum everything" version wrong:

- **Within one snapshot**, the tick entries of a slice add up — the filler split the blob's charge
  across them, so their sum is the slice's dead charge. Live entries and fillers are kept in
  separate accumulators and the fillers are used only when the slice carries no live entry
  (`prepare_data` inserts a filler *only* where no key exists, `TrackFitting.cxx:904`, so a slice
  can hold a real readout at one tick and fillers at the others; adding those onto a measured cell
  would inflate the measured charge the tree reports).
- **Across snapshots** landing in the same row (two live clusters sharing an ident), `charge` and
  `charge_err` are properties of the **readout** and are taken once, not summed; only the
  predictions add.

Verified **0 duplicate keys** over the 35 uBooNE and 6 SBND trees. Verified also, by intersecting
the pre-fix and final trees on `(cluster_id, channel, time_slice)` over all 35 uBooNE events at the
same `fit_exclusion`: of 204290 cells in common, the measured charge changes on **217 (0.11 %)**,
**all of them with `charge_pred == 0` in both trees**, by a median factor of exactly
`nticks_per_slice = 4`. Those are dead-region cells that now carry the slice's whole spread charge
instead of one tick's quarter — the prototype's per-slice convention. **No live cell's measured
charge moved**, so control 1 of §3 still means what it says.

One ordering note for anyone reading the branches by eye: cells within a row are now ordered by
`(channel, time_slice)` rather than by `(apa, face, plane) → (wire, tick)`. On uBooNE the two
coincide; on SBND `ChanScheme::global` is plane-major across APAs, so the interleaving differs.
The cell *set* and the parallel-branch alignment are unchanged.

Tests: `clus/test/doctest_pr109_proj_data_per_cluster.cxx`, two cases — the merge losing the main
cluster's answer on a shared cell (the contract the writers rely on), and two clusters sharing an
ident both keeping their snapshot (revert-proven: the old ident-keyed map holds one).
`./build/clus/wcdoctest-clus` 228/228, 2381 assertions.

Two divergences found and **surfaced, not fixed** (M15 — neither is in `porting_dictionary.md`):

- **Time-bin centring.** The toolkit integrates the time dimension over
  `[tbin − ntick/2, tbin + ntick/2]`, treating the slice's tick as the bin **centre**
  (`TrackFitting.cxx:5684`); the prototype integrates `[tbin, tbin+1]` in slice units with the
  centre at `tbin + 0.5` (`PR3DCluster_dQ_dx_fit.h:169`). If `slice_index` is the slice's *start*
  tick, the toolkit's response is half a slice (≈2 ticks, ≈1 µs) early. Needs the per-cell dump to
  settle; it biases predictions, it does not zero them.
- **6806's W map.** `n2d` W = 1980 (WCT) vs 609 (WCP) while U and V agree to a few percent. The
  bbox-padding hypothesis predicts all three planes inflate, so it does not explain this. The two
  clusters also differ (WCT cid 25 / 853 cells vs WCP cid 26 / 1376), so part of it is genuine
  clustering difference. Unexplained.

### 8.4 What the fix does to the numbers

**§7's headline, recomputed with §7's own probe** (`pr109_uncov_probe.py`, 3 cm box, ON arm,
uncovered *charge* fraction):

| event | pre-fix | post-fix | prototype |   | event | pre-fix | post-fix |
|---|---|---|---|---|---|---|---|
| 6505 | 0.2 % | 0.2 % | 0.1 % | | SBND 10550 | 17.1 % | 14.9 % |
| 6528 | 100 % | *(see below)* | 1.8 % | | SBND 46363 | 42.1 % | **3.3 %** |
| 6532 | 28.0 % | **9.3 %** | 5.6 % | | SBND 81597 | 30.8 % | **12.8 %** |
| 6650 | 7.2 % | **0.2 %** | 0.1 % | | SBND 360535 | 29.9 % | **6.3 %** |
| 6805 | 15.4 % | 14.2 % | 15.2 % | | SBND 256587 | 46.9 % | **3.3 %** |
| 6806 | 3.0 % | 3.0 % | 2.3 % | | SBND 433451 | 2.9 % | 2.6 % |

and the same quantity read off the grid TSVs (median `uncov` over all anchors, box 3 cm):

| family | pre-fix ON | post-fix ON | pre-fix OFF | post-fix OFF |
|---|---|---|---|---|
| uBooNE toolkit | 11.8 % | **0.9 %** | 13.1 % | **1.2 %** |
| uBooNE prototype | 1.7 % | 1.7 % (untouched) | 1.6 % | 1.6 % |
| SBND toolkit | 20.1 % | **4.0 %** | 19.3 % | **3.7 %** |

**The toolkit now sits at the prototype's level (0.9 % against 1.7 %), event by event.** uBooNE 6805
is the one event that does not move — §7 already identified it as a different mechanism (75 % of
its uncovered charge is on a cluster with no fitted points at all), and the prototype shows the
same 15 % there, which is now visible as agreement rather than as toolkit breakage. uBooNE 6528's
main cluster recovers from Σŷ/Σy = 0.000 to **1.013** (prototype 1.001), but the event drops out of
the *probe* for a new and honest reason: with each fitted cluster emitting its whole 2-D map, the
maps of cid 19 and the twin cid-1909 clusters overlap, and every cell in its vertex box is now
multi-owner, which the probe's single-owner filter removes.

**§4's verdicts, recomputed on the post-fix arms with §4's own recipe** (median over all anchors,
box 3 cm; the pre-fix row reproduces the published toolkit-uBooNE row exactly and SBND to ≤0.007,
so the recipe is the same one):

| family | ΔU (U) | ΔU (V) | ΔU (W) | ΔU pooled | Δ(χ²/N) |
|---|---|---|---|---|---|
| uBooNE toolkit, **pre-fix** | −0.016 | −0.012 | −0.013 | −0.024 | −1.73 |
| uBooNE toolkit, **post-fix** | −0.016 | −0.012 | −0.015 | **−0.026** | −1.67 |
| uBooNE prototype (untouched) | −0.023 | −0.023 | −0.014 | −0.026 | −1.36 |
| SBND toolkit, **pre-fix** | +0.030 | +0.003 | +0.039 | +0.026 | +2.37 |
| SBND toolkit, **post-fix** | +0.047 | +0.008 | +0.036 | **+0.036** | +2.53 |

Decision rule, post-fix (`pr109_summary.py`, entries = (event, region), `U` pooled over planes):

| family | entries B/E/W | per-event sign test | median ΔU |
|---|---|---|---|
| uBooNE toolkit | 10 / 4 / 4 | 3 better, 2 worse of 5 | −0.018 |
| uBooNE prototype | 9 / 0 / 2 | 4 better, 2 worse of 6 | −0.036 |
| SBND toolkit | 1 / 10 / 22 | **0 better, 6 worse of 6** | **+0.037** |

**Every conclusion of §4 and §5 survives, and the two that mattered get stronger.** The two
implementations still agree on uBooNE that the exclusion-ON trajectory describes the 2-D data
better — and they now agree *quantitatively* (pooled ΔU −0.026 vs −0.026, where before the toolkit
was measured through a corrupted tree). On SBND the owner's claim still holds at the registered bar
(6/6 events ON-worse) and the effect size grows from **+2.6 pp to +3.6 pp**. The plane pattern is
unchanged: SBND damage in U and W, V neutral.

One caveat the fix introduces on the metric side, stated rather than buried: because each fitted
cluster now emits its whole 2-D map, overlapping maps make more cells multi-owner (median shared
fraction on SBND 7 % → 16 %), so the post-fix sample is not the identical set of cells. It is,
however, the *same* convention the prototype has always used, so this is the first
apples-to-apples comparison of the three families.

### 8.5 Gates — scoped, because the fix is unknobbed

The owner chose an unconditional fix, which gives up the byte-identical bar of CLAUDE.md §1 by
construction (`T_proj_data` is *supposed* to change). What replaces it is a blast-radius proof:

Every pair below has the **same `fit_exclusion`** on both sides, read back from each job's own log.

| check | result |
|---|---|
| SBND Bee/pctree archives, pre-fix vs post-fix ON arm (`pr85_hash_gate.py`) | **PASS, all 12 archives byte-identical** |
| SBND `tracking-pr.root`, per-tree content hash, 6 events | `T_bad_ch` `T_kine` `T_proj` `T_rec_charge` `T_tagger` `Trun` identical **6/6**; `T_proj_data` differs 6/6 |
| SBND `nusel-evt*.tsv` (per-event scores) and merged `nusel-table.tsv` | identical **6/6** and identical |
| uBooNE Bee `mabc_*.zip` member-content rollup, `pr109b_bare_on` vs `pr109e_wct_off` (35 events) | **identical 35/35** (`diff` of the two `hashes.txt`) |
| uBooNE `track_com_*.root` per-tree, same pair, 35 events | `T_bad_ch` `T_kine` `T_proj` `T_tagger` `Trun` identical **35/35**; `T_proj_data` differs 35/35 |
| uBooNE `T_rec_charge`, same pair | identical **as a row set 35/35**; its row *order* differs, and it also differs on **34/35 events between two same-config runs of the same binary** (`pr109b_wct_on` vs `pr109b_wct_on2`, both `fit_exclusion=false`) — a pre-existing write-order non-determinism, mentioned and not touched |
| uBooNE repeat-run band on the metric itself, same pair | `T_proj_data`, `T_kine` and `T_tagger` **identical 35/35 between the two repeat runs**, so ΔU on uBooNE carries **no rerun band** — the same control §3 ran on SBND |
| duplicate `(channel, time_slice)` keys within a row (§8.3.1) | **0** over the 35 uBooNE + 6 SBND post-fix trees (33 % / 6.9 % before the de-duplication) |
| `SbndMagnifyTrackingVisitor` (the third writer changed) | not exercised by these arms *by construction*: `clus.jsonnet:2088` gives it `track_fitting_name: 'stm'`, a holder fed by `merge_fitted_charge_2d` that carries no snapshots, so it takes the unchanged fallback, and it is only instantiated with `-stm-fit` |
| `./build/clus/wcdoctest-clus` | 228 cases / 2381 assertions, SUCCESS |
| freshness (M1) | `local/lib/libWireCellRoot.so` 15:30 vs sources 15:29 |

So: no Bee layer, no pctree tensor, no tagger verdict, no selection score changes. The change is
confined to the tree it was aimed at.

### 8.6 What §7 got right, and what it got wrong

§7 stands as the measurement that located the problem — the uncovered cells really are inside the
coupling window, really are interleaved with covered ones at the same offset, really are
induction-only, and really are the same cells in both arms. Every one of those is what a
**cross-cluster overwrite** looks like, and §7's own framing ("a static per-cell condition, not
geometry") was pointing at it.

What §7 got wrong is the attribution in §7.3: it read the condition as a property of the *cell*
(the fill guard, dead-region `flag = 0` fillers, an unidentified live-U-plane skip) when it was a
property of the *writer*. Concretely, superseded:

- "the near-vertex shortfall is dominated by a static, induction-plane, per-cell condition … the
  remainder is unidentified" — **superseded**: the remainder was S1+S2 and is now measured (§8.4).
- the Phase-2 per-cell `(flag, value, row)` dump named in §7.3 as "the one measurement that would
  settle it" — **not needed** for this question; it would still be the instrument for the
  time-bin-centring divergence of §8.3.
- "the larger split is implementation, not detector" — **still true**, but the split was in the
  saving, not in the fitting, and it is now closed (1.6 % vs 1.7 %).

What survives unchanged from §7: dead-region `flag = 0` fillers are real and are a separate,
smaller contributor (6196/16759 cells on SBND 46363, 70 % of the V-plane cells, but only 4.3 % of
the charge); the ON/OFF stability that made the corruption largely cancel in ΔU is confirmed
directly — the verdicts did not flip, they sharpened; and uBooNE 6805's never-fitted cluster is a
genuinely different mechanism, now shown to be shared with the prototype.

---

## 9. Why the exclusion fit costs SBND and pays on uBooNE (2026-08-22)

Owner: *"zoom in to the toolkit's performance difference regarding the exclusion fit for MicroBooNE
and SBND. SBND has made quite a few improvements in the pattern recognition, which are gated by
different knobs. I suspected that these may lead to the difference that you observed… Basically, we
are after the reason behind why adding exclusion fit makes the charge matching slightly worse."*

**Answer.** `update_association`'s keep rule is a *tournament*: a 2-D cell stays with a segment only
if that segment is **strictly closest of all the sibling segments in the same cluster**. The strip
rate is therefore a rising function of how many siblings the cluster has — measured **within** each
detector below — and SBND's pattern recognition routinely runs that tournament with **26–37**
competing segments, a regime uBooNE never enters (its largest is 9). The same rule consequently
strips far more charge on SBND: **53.6 % of the associated cells within 3 cm of the neutrino vertex
on SBND 46363 against 21.1 % on uBooNE 6505, and 30.5 % against 6.4 % overall.** The
near-vertex trajectory then carries **21 % fewer fitted points** (SBND) versus **1 %** (uBooNE
toolkit) / **4 %** (prototype), so the dQ/dx system has less support exactly where the metric looks,
and its prediction falls further below the measured charge. This is not a knob defect and not a port
defect: it is the arbitration rule meeting a denser segment population. The owner's suspicion is
**confirmed in substance** — the SBND pattern-recognition knobs are implicated, but collectively,
through the segment count they produce, rather than through any single one.

`fit_blob_coverage` (SBND production ON, uBooNE absent) is a genuine **secondary** modulator: it
accounts for essentially all of the exclusion-induced *bias* deepening and flips 2 of the 6 events,
but it does not cause the stripping.

### 9.0 Repro

```bash
# Same binary as §8 (no source edit this round): toolkit 56683366,
#   local/lib/libWireCellClus.so md5 6524549bb4a064fcc5ca83b0474d2352
#   local/lib/libWireCellRoot.so md5 0d85ff1fa144c201132e63b3bd6627f7
# TRAP (§8.0, still live): the uBooNE TLA is compared with == "true"
#   (uboone-mabc.jsonnet:1513), so QL_FIT_EXCLUSION=1 SILENTLY MEANS false.
#   Always read the value back:  grep -o "fit_exclusion=[a-z]*" .../wct_5384_<ev>.log
# TRAP (§8.0): pr109b_run_all.sh:3's header comment names pr109b_wct_{on,off};
#   the BODY correctly uses pr109e_*.  "Fixing" the body to match the comment
#   silently yields an OFF-vs-OFF comparison with plausible-looking dU.

# (a) the factorial: same post-fix binary, fit_blob_coverage forced to the C++ default -1
cd sbnd_xin
SBND_FIT_EXCLUSION=true  SBND_FIT_BLOB_COVERAGE=-1 PR_JOBS=6 ./run_pr_chain_batch.sh \
    work-nuecc48-ql0819 work-pr109f-on-fbcoff-nuecc48  data 10550 46363 81597 360535 256587 433451
SBND_FIT_EXCLUSION=false SBND_FIT_BLOB_COVERAGE=-1 PR_JOBS=6 ./run_pr_chain_batch.sh \
    work-nuecc48-ql0819 work-pr109f-off-fbcoff-nuecc48 data 10550 46363 81597 360535 256587 433451
scripts/pr109f_run_all.sh /home/xqian/tmp/pr109f
python3 scripts/pr109_summary.py /home/xqian/tmp/pr109f/sbnd_fbcoff.tsv   # vs .../pr109e/sbnd.tsv

# (b) the channel decomposition -- read-only on the §8 arms, no run at all
python3 scripts/pr109_chanb_probe.py --tag <evt> \
    --arm ON=<arm>/tracking-pr.root:wct --arm OFF=<arm>/tracking-pr.root:wct

# (c) the strip fraction -- the pr/108 §8 stage dump, one event per file
WCT_TRAJ_DUMP=/home/xqian/tmp/traj_sbnd46363.txt SBND_FIT_EXCLUSION=true PR_JOBS=1 \
    ./run_pr_chain_batch.sh work-nuecc48-ql0819 work-pr109f-dbgtraj-on-nuecc48 data 46363
cd ../../../toolkit/qlport/scripts
WCT_TRAJ_DUMP=/home/xqian/tmp/traj_ub6505.txt QL_FIT_EXCLUSION=true ./run_one.sh 1 pr109f_dbgtraj_on
python3 scripts/pr109_traj_strip.py --dump <dump> --root <tracking root> --label "<name>"
```

Arms: `work-pr109f-{on,off}-fbcoff-nuecc48` (6 events each, all rc=0),
`work-pr109f-dbgtraj-on-nuecc48`, `qlport/scripts/sweep/pr109f_dbgtraj_on`. §8's
`pr109e_*` / `work-pr109e-*` arms are the unchanged reference. Compiled-config proof, evt 46363:
`on-fbcoff` → `fit_exclusion=True fit_blob_coverage=-1`; `off-fbcoff` → `fit_exclusion=<absent>
fit_blob_coverage=-1` (key-suppression idiom ⇒ C++ default false).

### 9.1 First: which cells carry ΔU? Not the ones that lose their prediction

`scripts/pr109_chanb_probe.py` classifies every in-box cell present in **both** arms by what
happened to its prediction — *lost* (`ŷ_OFF > 0, ŷ_ON = 0`), *gained*, *moved* (both non-zero),
*uncovered in both*. Cells uncovered in both arms contribute `|y − 0|` identically on each side and
cancel exactly in ΔU, so §7's uncovered population cannot be the cause; and the *lost* population
turns out to be nearly irrelevant too:

| detector | event | ΔU | share of Δ numerator from **moved** cells | lost − gained (e⁻) |
|---|---|---|---|---|
| SBND | 10550 | +0.037 | **91 %** | +7.7e5 |
| SBND | 46363 | +0.051 | **98 %** | +4.0e4 |
| SBND | 81597 | −0.002 | (Δ numerator ≈ 0) | −1.2e5 |
| SBND | 360535 | +0.061 | **107 %** | −3.5e5 |
| SBND | 256587 | +0.027 | **71 %** | −5.2e4 |
| SBND | 433451 | +0.075 | **97 %** | +3.7e5 |
| uBooNE | 6505 | −0.076 | **100 %** | −1.1e4 |
| uBooNE | 6532 | +0.043 | **99 %** | −5.7e4 |
| uBooNE | 6650 | +0.019 | **100 %** | −7.7e3 |
| uBooNE | 6805 | −0.021 | **62 %** | −1.3e5 |
| uBooNE | 6806 | −0.183 | **100 %** | −1.1e5 |

So ΔU is carried, in both detectors, by cells that **both** arms predict — the prediction moved, it
did not vanish. That also kills the coverage/geometry hypothesis (the §8 time-bin-centring
divergence) as an explanation of ΔU: it would have to act through cells that change coverage class.

### 9.2 The direction: exclusion deepens an existing charge deficit on SBND and relieves it on uBooNE

Median over all (event, region, box) entries, `B = (Σŷ − Σy)/Σy`:

| family | B, exclusion ON | B, exclusion OFF | Δ |
|---|---|---|---|
| SBND, production (`fit_blob_coverage = 0`) | **−0.323** | −0.273 | **−0.049** (worse) |
| SBND, `fit_blob_coverage = -1` | −0.365 | −0.372 | **+0.007** (gone) |
| uBooNE toolkit | −0.178 | −0.225 | **+0.048** (better) |
| uBooNE prototype | −0.061 | −0.074 | +0.013 (better) |

Both detectors *under*-predict near-vertex charge; exclusion makes SBND's deficit deeper and
uBooNE's shallower. Note the third row: with `fit_blob_coverage` off, SBND's ON-vs-OFF bias gap
**disappears entirely** — that knob, not exclusion, is what makes the deficit exclusion-sensitive.

Whole-event (not near-vertex) bias is large on both detectors and agrees between implementations
(SBND −0.33…−0.54; uBooNE WCT −0.10…−0.42 with WCP within a few pp on the same events), so the
deficit itself is a shared property of the 2-D projection, not a toolkit defect. What is
SBND-specific is that uBooNE's near-vertex bias is *much better than its own whole-event average*
(−0.086 vs −0.28 near/all) while SBND's is not (−0.36 vs −0.41): the extra trajectory density at a
vertex buys uBooNE a better local description and buys SBND nothing.

### 9.3 The mechanism: exclusion strips half of SBND's near-vertex associations

`update_association` (`TrackFitting.cxx:2789-2799`) keeps a cell iff
`min_dis_track < min-over-other-segments` or `min_dis_track < 0.3 cm`, arbitrating **only among the
fitted cluster's own segments** (`m_cluster_filter` → `get_segment_edges()`, `:2731-2735`). The
pr/108 §8 stage dump records the association size before (`n0`) and after (`n1`) that call, so the
strip fraction is directly measurable. `scripts/pr109_traj_strip.py`, exclusion ON:

| | SBND 46363 | uBooNE 6505 |
|---|---|---|
| strip %, pass 1 / 2 / 3 | **30.5 / 27.8 / 21.6** | **6.4 / 6.1 / 4.1** |
| strip % within 3 cm of the ν vertex | **53.6** | **21.1** |
| 3–6 cm | 38.9 | 8.8 |
| 6–10 cm | 24.7 | 0.2 |
| > 20 cm | 23.4 | 2.8 |
| **null control** — the 5 hard-`false` call sites | **0.0 %** all passes | **0.0 %** all passes |

Exclusion is ~5× more aggressive overall on SBND and ~2.5× more aggressive in the region the metric
scores. The null control (calls that pass `flag_exclusion = false` by construction —
`NeutrinoPatternBase.cxx:2264/2339/2531`, `NeutrinoVertexFinder.cxx:780/4806`) strips exactly
nothing, confirming the dump is measuring the flag and not something else.

**Why SBND strips more — the arbitration universe, with a dose-response.** The keep test is
"strictly closest of all siblings", so its severity should grow with the number of siblings. Both
fields are on the same dump lines, so this is testable *within* each detector rather than inferred
from a between-detector comparison. Grouping exclusion calls by the number of distinct segments they
arbitrate over, and pooling the associations in each band:

| segments per call | SBND 46363: calls / strip % | uBooNE 6505: calls / strip % |
|---|---|---|
| 1 (no competitor) | 54 / **0.0 %** | 6 / **0.0 %** |
| 2–5 | 6 / 23.5 % | 4 / 2.8 % |
| 6–10 | 3 / 6.4 % | 13 / 5.9 % |
| 11–25 | 9 / 17.5 % | **none** |
| **26–99** | **25 / 30.3 %** | **none** |
| Spearman ρ(nseg, strip) over calls | **+0.98** | +0.53 |

Single-segment calls strip **exactly 0.0 %** in both detectors — there is no competitor to lose to —
which is a second null control on the instrument. Above that the strip rate climbs monotonically
with the sibling count in both detectors, so the tournament interpretation is earned rather than
assumed.

The between-detector difference is then a difference of **regime, not of average**: uBooNE's calls
never exceed 10 segments, while SBND runs 34 of its 97 calls at 11+ and 25 at 26–37, and those 25
carry **78 % of all of SBND's associations**. Their 30.3 % strip is what drives the 30.5 % aggregate;
the rest of SBND's calls strip 15.2 %. Note that the *medians* point the other way (SBND 1.0,
uBooNE 7.0) because SBND's distribution is bimodal — 54 of 97 calls are single-segment and strip
nothing. The claim is about the tail, and the tail is where the charge is.

**And the trajectory loses points as a result.** Counting `T_rec_charge` fitted points within 3 cm
of the *same* anchors (main vertex + 3 junctions, taken from the OFF arm so both arms are scored on
one window), ON vs OFF, summed over the events:

| family | points, ON | points, OFF | **loss** |
|---|---|---|---|
| SBND, production | 416 | 528 | **21 %** |
| SBND, `fit_blob_coverage = -1` | 449 | 548 | 18 % |
| uBooNE toolkit | 341 | 346 | **1 %** |
| uBooNE prototype | 183 | 190 | **4 %** |

Per event the SBND loss reaches 30 % (46363, 360535) with individual junctions far worse
(360535 J2: 36 → 16 points; 81597 J0: 23 → 9). uBooNE is within ±2 points on every region.

Note that this is **not** the logged zero-quantity drop of the final pre-dQ/dx pass: that drop is
0.1 % of points in the dump and 1–9 points per event in the log (`:8909-8918`), exclusion-driven on
5/6 SBND events and 0/6 uBooNE. The 21 % is the *topological* consequence — segments trimmed and
restructured across the 34 exclusion call sites — not a single drop site. It follows that
`dqdx_fit_keep_all_points` (pr/107), which only neutralises the third pass, cannot recover it, which
is consistent with pr/107 measuring so little from it.

### 9.4 The knob: `fit_blob_coverage` is a real but secondary modulator

`fit_blob_coverage` is SBND production ON (`0`, `wct-pr-perevt.jsonnet:1880`, owner flip
2026-08-08) and absent on uBooNE (neither `uboone-mabc.jsonnet` nor `uboone_track_fitting.json`
carries the key ⇒ C++ default `-1` = off, `TrackFittingPresets.h:59`). It deweights, to ×0.1 in the
trajectory fit, live cells outside the fitted cluster's own blob coverage that sit inside an
out-of-scope cluster's. It composes with exclusion on the same cell set — exclusion strips at
`:3608`, coverage deweights the survivors at `:3615`, both keyed off the same `m_cluster_filter` —
and it fires in **every** SBND event, concentrated exactly where the metric looks. Counting its log
line (`:3145`, which carries `vtx_dis` and `vtx_deg`) on the §8 arms:

| evt | arm | positions | cells | pos ≤3 cm | cells ≤3 cm | deg ≥2 |
|---|---|---|---|---|---|---|
| 10550 | ON / OFF | 804 / 891 | 16769 / 17567 | 337 / 483 | 5666 / 9321 | 555 / 724 |
| 46363 | ON / OFF | 1096 / 1365 | 7016 / 8321 | 617 / 802 | 3893 / 5104 | 671 / 805 |
| 81597 | ON / OFF | 289 / 258 | 1545 / 1488 | 177 / 165 | 935 / 902 | 173 / 139 |
| 360535 | ON / OFF | 377 / 494 | 2005 / 2688 | 253 / 381 | 1481 / 2200 | 25 / 164 |
| 256587 | ON / OFF | 1792 / 1290 | 5051 / 3575 | 1112 / 508 | 3266 / 1826 | 1057 / 522 |
| 433451 | ON / OFF | 684 / 851 | 2330 / 3038 | 340 / 442 | 1518 / 2038 | 342 / 438 |

30–60 % of the deweighted charge sits inside the metric's own 3 cm box, and the near-vertex count is
not a fixed offset between arms — it swings +119 % (256587) to −34 % (360535), i.e. the two
mechanisms interact in the measured region.

**Pre-registered rule** (fixed before the numbers were seen, stated in the instrument's own terms —
`pr109_summary.py` calls ON-worse only when `dU > +Bsys`, the entry's own systematic band): flip on
**≥4 of 6** events ⇒ the knob is the cause; **≥5 of 6** still ON-worse ⇒ refuted; between ⇒ partial.

| | entries: better / equiv / worse | per-event sign test |
|---|---|---|
| production, `fit_blob_coverage = 0` | 1 / 10 / **22** | **ON-worse 6/6** |
| `fit_blob_coverage = -1` | **4** / 12 / 15 | ON-**better 2/6**, ON-worse 4/6 |

Per event, with the knob off: 46363 flips (2 better, 3 equivalent, 0 worse), 433451 flips to
mixed, 10550 improves; 360535, 81597, 256587 stay ON-worse. **Verdict: partial** — it removes a
third of the ON-worse events and all of the bias deepening (§9.2), but four events remain ON-worse
and the stripping is essentially unchanged (21 % → 18 % point loss). It is a modulator, not the
cause.

**Validity guard, per event** (turning off a production-ON knob can degrade reconstruction to where
`U` is not comparable, so this is checked before any `fbcoff` number is believed). All 12 jobs rc=0.
Every one of the 24 arm-events retains a space-charge-corrected neutrino vertex, a main cluster
(440–1931 points), all 4 anchor regions, and a non-empty `nusel-evt*.tsv`: **structurally PASS
6/6**. The plan's fourth criterion — in-box single-owner cell count within ±20 % of the production
ON arm — is exceeded on three arm-events (81597 `fbc-ON` +21.2 %, 256587 `fbc-OFF` +20.5 %), **but
it is also exceeded by the production OFF arm itself** (256587 −26.4 %, 10550 +18.7 %, 360535
+11.6 %). That criterion is therefore measuring the anchor/box shift that exclusion produces by
design, not a degenerate reconstruction, and it does not separate the `fbcoff` arms from the §8
baseline. No event was excluded; the per-event numbers are in the sidecar §6e.

### 9.5 A structural difference that is not a knob, and cannot be tuned away

Fitted from each arm's own `T_rec_charge` (`affine()`, residual 0.0000): the two detectors have the
**same wire pitch** — 3.33 wires/cm on U, V and W in both — but **different drift sampling**:

| | wires/cm (U/V/W) | slices/cm | one slice |
|---|---|---|---|
| SBND | 3.33 / 3.33 / 3.33 | **3.20** | **3.13 mm** |
| uBooNE (WCT and WCP identical) | 3.33 / 3.33 / 3.33 | **4.54** | **2.20 mm** |

`update_association` converts each cell to a physical point and applies a **hard-coded 0.3 cm
always-keep floor** (`TrackFitting.cxx:2790/2837/2884`). That floor spans **±0.96 slices on SBND**
against **±1.36 slices on uBooNE**: a cell one slice off the trajectory is auto-kept on uBooNE and
must win the tournament on SBND. This is a plausible additional contributor and it is *reported, not
claimed* — separating it from the segment-count effect would need the floor made configurable, which
is a code change and therefore out of scope for this round (§5 rule 1).

Two further non-knob asymmetries recorded for the record, both affecting where the box is placed
rather than what is inside it: uBooNE runs with the **DL vertex disabled** (`run_one.sh:61` forces
`-A dl_weights=`) while SBND runs it on; and `fit_vertex_min_seg_length` is 1.0 cm on SBND
(`cfg/…/sbnd/clus.jsonnet:3050`) against the C++ default 0 on uBooNE, deciding which segments enter
the vertex fit — it has **no TLA and no runner env**, so it cannot be A/B'd today.

### 9.6 What this does and does not say about production

It does **not** say `fit_exclusion` should be turned off for SBND. This metric scores one thing —
how well the fitted trajectory's dQ/dx explains the near-vertex 2-D charge — and pr/98 §7 shipped
`fit_exclusion` ON for a different and independently validated reason (fits equal-or-better in 11/12
top movers). §9 explains the cost side of that trade honestly; it does not re-price it. The
candidate follow-ups §6 listed are unchanged, and one new one is suggested by §9.3: because the
penalty scales with the sibling-segment count, a **segment-count-aware** relaxation of the
arbitration (for example, exempting cells whose two best competitors are within the fit's own
position resolution, rather than the current strict `<`) would target exactly the population this
round identified. That would be a default-OFF knob and a new round, not a change here.

### 9.7 Scope and limitations

- **n = 6 events per detector**, and the strip-fraction measurement is **n = 1 per detector**
  (46363 and 6505). The segment-count contrast is large (37 vs 9) and the point-loss contrast is
  consistent across all 6+5 events, but a knob-level attribution on 6 events is thin. Widening to
  the full nueCC48 manifest (47 events × 2 arms) plus the 35-event uBooNE sweep would cost roughly
  the same as one pr-round arm pair and is the obvious next step if the conclusion is load-bearing.
- Cross-detector *levels* (U, B) inherit the anchor asymmetry of §9.5; the within-detector ON-vs-OFF
  comparisons, which is what every verdict here rests on, do not.
- **No code changed this round.** `./build/clus/wcdoctest-clus` re-run on the same binary; the
  libraries' md5s are quoted in §9.0 and match §8's post-fix build.

### 9.8 Found, not fixed

`run_pr_chain_batch.sh:1506-1510` maps five env vars — `SBND_MUON_CHAIN_PROTON_VETO`,
`SBND_SHOWER_TYPE_CACHE_REFRESH`, `SBND_SHOWER_TRAJ_DQDX_GUARD`, `SBND_SHOWER_TRAJ_CHAIN_PION`,
`SBND_KINE_SHOWER_VERTEX_BARRIER` — to TLA names that exist nowhere in `cfg/` or `clus/`. Setting
any of them emits `--tla-code` for an undeclared top-level parameter, which is a hard jsonnet
error, not a silent no-op. Not touched in this round.
