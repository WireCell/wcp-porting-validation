# doc pr/109 — Near the vertex, which fit describes the measured 2-D charge better: exclusion ON or OFF? (2026-08-22)

**Status:** measurement only, **no production change, no code change**. The answer differs by
detector and it is the same in both implementations, so it is not a port defect:

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

**Does not settle.** Why SBND leaves ~20 % of the near-vertex charge unpredicted in both arms
(against 0.3 % for the prototype on uBooNE) — that common shortfall is larger than the ON/OFF
effect and is the first thing to look at next. And why the two detectors disagree. The measurement says it is not the port
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
