# 39 — What is left in the STM fit after doc 38, and the second stage nobody measured

**Status.** Analysis only. No code, no config, no production output changes.
Written to answer a direct question from the owner after the doc-36 metric flip
and the doc-38 trim both landed in PDVD production:

> *"Did I understand correctly the biggest issue now is that we have ghost track
> fitted? And in some cases, we lost fitted track completely? Note, here I am
> only talking about the STM tagger fitted track, we will worry about the
> neutrino PR chain later."*

Short answer: **ghost is real but it is a minority of objects, not a property of
the fits**; **nothing was lost in aggregate** — the STM fitted population is
*up* four clusters across the whole campaign; and the residual that remains is
**one defect with two symptoms**, concentrated in a few hundred fused
multi-track objects that are upstream of fitting entirely.

The second half of the question turned out to have a premise worth checking, and
it did not hold: **the fitting config is not scoped to the STM tagger.** The same
`pdvd_track_fitting.json` is handed to `TaggerCheckNeutrino`, so the neutrino PR
chain has already moved — by more than the STM layer did (§4). It is not
"later"; it is unmeasured.

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
SC=docs/nf_sp_img_clus/scripts
T=/home/xqian/tmp/d38_sweep_final.tsv

# the per-cluster grade this doc reads (arms are doc 38's; d38h20 = production)
python3 $SC/d36_fit_twoaxis_scan.py d38off,d38h5,d38h10,d38h20,d38h40 $T

# sec 2 -- who gets a fit at all, and why the rest do not
python3 $SC/d39_fit_population.py /home/xqian/tmp/d38_arms2:d38h20

# sec 3 -- what the residual actually is (median vs mean, concentration,
#          whether low coverage and high ghost are the same clusters)
python3 $SC/d39_residual_profile.py $T d38h20 d38off

# sec 4 -- BOTH fit layers on the same events
python3 $SC/d39_stage_scope.py d36off d36on d38off d38h20

# sec 5.1 -- the metric state of each arm (this is the correction)
grep -m1 'ctpc_aniso_metric ON' /home/xqian/tmp/d38_arms/d38ref_039252_2.log
```

Two output layers are involved and they are **different objects in the same
zip**:

| layer | written by | this doc |
|---|---|---|
| `data/0/0-stm_fit-global.json` | `TaggerCheckSTM::persist_stm_fit` (`TaggerCheckSTM.cxx:614`) | §2, §3 — everything docs 36 and 38 graded |
| `data/0/0-track_fit-global.json` | `TaggerCheckNeutrino`, grouping slot `"nu<idx>"` (`TaggerCheckNeutrino.cxx:3577`) | §4 — never graded before |

## 1. Why the question needed more than the doc-38 tables

Doc 38 graded the trim with `d36_fit_twoaxis_scan.py`, which builds its cluster
list from the **union of the arms' fit outputs**. A cluster that never got a fit
in *any* arm is therefore invisible to it. That script is structurally unable to
answer "did we lose fits entirely?" — its `destroyed` column is a delta against
one baseline arm, not a population. §2 starts from the clustering output instead.

## 2. Nothing was lost

### 2.1 Most "no fit" is the tagger declining, by design

`TaggerCheckSTM` loops over flash-matched **main** clusters only
(`TaggerCheckSTM.cxx:562`), so a census over every cluster in the event answers
the wrong question. Over the 120-event manifest, production arm `d38h20`:

| population | clusters | with no `stm_fit` |
|---|---|---|
| every cluster ≥ 50 charge points | 4845 | 2694 (55.6 %) |
| **clusters the tagger evaluated** (all sizes) | 4948 | — |
| **… of those, ≥ 50 charge points** | **3025** | **874 (28.9 %)** |

The 55.6 % is not a defect and must not be quoted as one: most clusters are never
offered to the fitter. Within the population that *is* offered, the exit reasons
are logged per cluster (counted as distinct ids, never as log lines —
`feedback_log_line_count_is_not_object_count`):

| pre-fit exit reason, clusters ≥ 50 pts | count |
|---|---|
| `fully contained (Mid Point A)` | 778 |
| `N candidate exit points, need exactly N (Mid Point C)` | 46 |
| `single exit point with a N deg mid-track kink` | 40 |
| `round-N forward fit gave only N points` | 10 |
| **total** (`evaluated but no pass recorded` is the companion line, = the total) | **874** |

**778 of the 874 are a stopping-muon tagger correctly declining a cluster that
never enters the detector.** The genuine "could not fit something we wanted to"
bucket is 96 of 3025 = **3.2 %**.

And it did not get worse. Fitted STM clusters ≥ 50 charge points, same 120
events, all four arms present:

| arm | configuration | fitted ≥50 pt |
|---|---|---|
| `d36off` | metric OFF, floor 0.35 — **pre-flip production** | 2147 |
| `d36on` | metric ON, floor 0 | 2157 |
| `d38off` | metric ON, floor 0, no trim | 2159 |
| **`d38h20`** | **metric ON, floor 0, trim 20 cm — production** | **2151** |

Across the entire campaign the STM fitted population is **+4**.

### 2.2 The seven fits the trim removed were 77–91 % empty space

Verified directly from the archives, not only from the sweep TSV. `ghost` is the
fraction of fit points more than 2 cm from *any* 3-D charge in the event;
`extent` is the diagonal of the cluster's bounding box.

| event | cl | fit pts | charge pts | extent | cm/point | ghost |
|---|---|---|---|---|---|---|
| 039349/21 | 61 | 452 | 345 | 131 cm | 0.38 | 91 % |
| 039253/4 | 38 | 153 | 420 | 56 cm | 0.13 | 81 % |
| 039349/55 | 64 | 151 | 53 | 88 cm | 1.67 | 81 % |
| 039349/80 | 67 | 132 | 71 | 80 cm | 1.13 | 77 % |
| 039252/16 | 81 | 126 | 62 | 126 cm | 2.03 | 80 % |
| 039349/25 | 15 | 87 | 115 | 76 cm | 0.66 | 78 % |
| 039349/27 | 51 | 48 | 63 | 66 cm | 1.05 | 81 % |

Median ghost of the removed fits: **81 %**. The trim did not delete tracks; it
deleted ghosts that happened to *be* the whole fit.

**This corrects doc 38 §5.2** — see §5.2 below. Three of the seven are not the
sparse class that doc described.

## 3. The residual is one defect with two symptoms

Production arm `d38h20`: 2152 clusters with a fit, 4 033 846 charge points,
865 953 fit points.

### 3.1 The median cluster is not the weighted mean

| axis | charge/point-weighted | **median cluster** | mean cluster |
|---|---|---|---|
| coverage (own charge within 2 cm of the fit) | 71.5 % | **90.7 %** | 80.5 % |
| ghost (fit points > 2 cm from any charge) | 7.0 % | **0.0 %** | 9.5 % |

**The typical fitted STM track has 90.7 % coverage and zero ghost.** The
weighted aggregates docs 36 and 38 argued over are dominated by a few very large
objects and say almost nothing about the population.

### 3.2 It is concentrated

| worst *k* clusters (of 2152) | share of all uncovered charge | share of all ghost points |
|---|---|---|
| 10 | 28.9 % | 6.7 % |
| 50 | 53.7 % | 24.1 % |
| 100 | 63.0 % | 38.4 % |
| 215 (10 %) | 73.2 % | 61.7 % |

Ten clusters hold 28.9 % of every uncovered charge point in the manifest.

### 3.3 Coverage and ghost are the same clusters

Pearson *r*(coverage, ghost) = **−0.506**, and monotonic across the whole range:

| coverage band | n | mean ghost | median cluster size |
|---|---|---|---|
| < 50 % | 273 | 32.8 % | 912 pts |
| 50–80 % | 371 | 13.1 % | 1188 pts |
| 80–95 % | 855 | 5.1 % | 1506 pts |
| ≥ 95 % | 653 | 3.3 % | 1320 pts |

934 of 2152 clusters (43.4 %) are clean on both axes (≥ 90 % coverage and ≤ 5 %
ghost). 273 are below 50 % coverage, 313 are above 25 % ghost, and **147 are
both** — that overlap is the core pathology population.

This is the finding that reranks the work: **it is one defect, not two.** A fit
that has to span two real tracks fails to cover either and spends its length in
the gap between them. Trim tuning cannot separate those symptoms because they
are the same objects.

### 3.4 The worst objects are not single tracks

`uncov_d` is the median distance of uncovered charge from the fit; `separation`
is the distance between the centroid of the uncovered charge and the centroid of
the covered charge.

| event | cl | charge pts | cov | ghost | extent | uncov_d | separation |
|---|---|---|---|---|---|---|---|
| 039253/2 | 80 | 45 726 | 3 % | 44 % | 349 cm | 16 cm | 21 cm |
| 039252/2 | 84 | 39 286 | 2 % | 33 % | 334 cm | 31 cm | 36 cm |
| 039252/12 | 117 | 38 710 | 3 % | 18 % | 280 cm | 23 cm | 25 cm |
| 039252/5 | 75 | 36 996 | 3 % | 34 % | 477 cm | 22 cm | 12 cm |
| 039252/1 | 79 | 37 431 | 4 % | 25 % | 296 cm | 19 cm | 27 cm |
| 039253/9 | 76 | 35 284 | 4 % | 22 % | 348 cm | 17 cm | 14 cm |
| 039253/11 | 83 | 34 130 | 4 % | 27 % | 273 cm | 15 cm | 26 cm |
| 039252/14 | 73 | 30 200 | 4 % | 4 % | 300 cm | 22 cm | 17 cm |
| 039252/4 | 91 | 24 854 | 9 % | 20 % | 407 cm | 14 cm | 76 cm |
| 039253/5 | 84 | 22 571 | 4 % | 7 % | 393 cm | 48 cm | 43 cm |

22 000–46 000 charge points spanning 270–480 cm at 2–9 % coverage. A single
polyline covers 3 % of such an object because **it is not one track** — it is a
cosmic shower or several tracks fused in imaging (doc 96: the over-clustering is
fused in imaging; doc 32 §2.2). The coverage axis is partly *measuring* that
fusion rather than the fit's quality, which is a limit of the instrument and
should be stated whenever the 71.5 % is quoted.

## 4. The fitting config governs two stages

### 4.1 The binding

`pdvd_track_fitting.json` — the file holding the retired `good_point_pitch_frac`
and the new `end_trim_gap_len` — is passed to **both** taggers:

```
cfg/pgrapher/experiment/protodunevd/pr.jsonnet:1341   tagger_check_stm(      trackfitting_config_file=...)
cfg/pgrapher/experiment/protodunevd/pr.jsonnet:1562   tagger_check_neutrino( trackfitting_config_file=...)
```

Both call `load_trackfitting_config` into their own `TrackFitting` instance
(`TaggerCheckSTM.cxx:385`, `TaggerCheckNeutrino.cxx:778`). Separately,
`ctpc_aniso_metric` is a `Grouping` flag, so it changes every ctpc query in the
job. **Neither change was ever scoped to the STM tagger**, and docs 36 and 38
graded only the STM layer.

### 4.2 What moved in the neutrino layer

120 events present in every arm, so the totals are like for like:

| arm | STM `stm_fit` clusters ≥50pt | STM fit points | nu `track_fit` clusters | nu fit points |
|---|---|---|---|---|
| `d36off` (pre-flip production) | 2147 | 896 379 | 1534 | 109 609 |
| `d36on` (metric ON) | 2157 | 953 441 | 1447 | 94 515 |
| `d38off` (metric ON, no trim) | 2159 | 948 052 | 1447 | 103 715 |
| `d38h20` (**production**) | 2151 | 875 322 | 1621 | 117 149 |

Only **same-epoch** adjacent pairs are attributable to a single change — the
doc-37 Steiner terminal thinning landed between the d36 and d38 rounds
(`feedback_check_the_cfg_epoch_between_arms`):

| pair | single change? | STM | neutrino |
|---|---|---|---|
| `d36off` → `d36on` | **yes — the metric** | +10 cl / +6.4 % pts | **−87 cl / −13.8 % pts** |
| `d36on` → `d38off` | no — doc-37 thinning | +2 cl / −0.6 % pts | +0 cl / +9.7 % pts |
| `d38off` → `d38h20` | **yes — the 20 cm trim** | −8 cl / −7.7 % pts | **+174 cl / +13.0 % pts** |

The neutrino layer moved **more** than the STM layer both times, and in the
opposite direction each time. Nothing here says whether those moves are good —
no coverage or ghost grade has ever been run on `track_fit`. That is the single
largest unmeasured consequence of the two production flips.

## 5. Corrections to earlier statements

### 5.1 `d38ref` is not a pre-metric baseline

An interim report of these numbers described the 28.7 % no-fit rate as measured
"before any of today's changes". Wrong scope. **All three d38 arms already have
the anisotropic metric ON** — the doc-36 flip had landed in
`pdvd/wct-pr-perevt.jsonnet` before the d38 round ran, and every arm's log
carries

```
ctpc_aniso_metric ON: apa 0 face 0 drift_step 2.9615 mm  yscale U/V/W 0.3871/0.3871/0.5807
```

`d38ref` is the **trim** byte-identity reference (old pin, no trim code), not a
pre-metric arm. The pre-metric arm is `d36off`, whose fitted count is 2147
(§2.1). Its per-event logs no longer exist, so its no-fit *denominator* cannot be
recomputed directly; assuming the evaluated population is unchanged — it is
identical (3026) across every d38 arm, and flash matching is upstream of all
three knobs — that puts the pre-metric no-fit rate at 879 of 3026 (29.0 %)
against 875 (28.9 %) now. The conclusion — flat, and dominated by the
`fully contained` exits — is unchanged; the label on it was not.

### 5.2 Doc 38 §5.2's ranges cover four of the seven destroyed fits

Doc 38 §5.2 describes the destroyed class as "51 to 71 charge points strung over
80 to 101 cm, i.e. 1.1–2.0 cm of extent per charge point … fits ran 77–81 % more
than 2 cm from any charge". Measured over all seven (§2.2), the real ranges are
**53–420 charge points**, **56–131 cm extent**, **0.13–2.03 cm per point**, and
**77–91 % ghost**.

**Three of the seven are dense, not sparse**: 039253/4 cl 38 (420 points in
56 cm, 0.13 cm/pt), 039349/21 cl 61 (0.38) and 039349/25 cl 15 (0.66). For those,
the fit left a perfectly well-populated compact object and ran 78–91 % through
empty space — that is *not* the over-clustering class doc 38 attributed all seven
to, and "the trim is refusing to invent a trajectory across a gap" does not
explain it. The four sparse ones do match doc 38's description. The operating-point
decision in doc 38 §8.1 is unaffected: all seven fits were ≥ 77 % ghost either way.

## 6. What this says to do next

1. **Grade `track_fit` the way `stm_fit` was graded** (§4.2). It is a small
   change to `d36_fit_twoaxis_scan.py` — the PC name is the only thing that is
   STM-specific. Until that exists, two production flips have an unmeasured
   effect on the chain that produces the physics output.
2. **Then** decide on the fused-cluster splitting (§3.4, doc 38 §9). It is the
   only lead that moves both axes at once, because §3.3 shows they are one
   defect. Trim retuning is exhausted: doc 38 §5's sweep spans 2–40 cm and the
   coverage axis moves by 0.3 points across the whole range.
3. Do **not** open work justified by "ghost is the biggest issue" or by the
   71.5 % coverage figure alone. §3.1 shows the median fitted track is already
   clean on both axes; the aggregates are ten objects deep.

## 7. Not done

- **No grade on `track_fit`.** §4.2 counts points and clusters only. Whether the
  neutrino layer's extra 174 clusters are real trajectories or ghosts is unknown.
- **The dense three of §5.2** have no mechanism. They are a different failure
  from the sparse four and were not investigated.
- **The 96 genuine no-fit clusters** (§2.1: 46 Mid Point C, 40 Mid Point B,
  10 short forward fit) were counted, not examined.
- **SBND, PDHD, uBooNE.** Everything here is PDVD. SBND remains on the legacy
  isotropic metric and the legacy tip-only trim by the owner's decision.
