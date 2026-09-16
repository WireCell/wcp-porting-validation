# doc pdvd/110 — gaps and zig-zag in the fitted STM trajectories: the viewer skips low-charge rows, and the rows are low where the fit leaves the charge

**Status.** Investigation only. **No C++, no config, no new arm; nothing under `work/` was written.**
The owner's five Bee spots are reproduced as figures on the arm they viewed and on the other side of the
trajectory flip, and each defect is traced to where it is made:

- **The gaps are made by the viewer.** The persisted STM fit is continuous (step p99 0.95 cm). Bee skips every
  point with `q < 0`, and the `stm_fit` layer encodes `q = dQ·0.1 − 1000` without the clamp the
  `track_fit` layer has. So every fitted row with **dQ < 10 ke** disappears. Every named gap is such a run.
- **The rows are low where the fit trajectory has left the track's charge**: a bump off the image (v1–v3,
  pre-flip PDVD), a corner cut across a bend (h1, production PDHD), rows stacked at one time slice across
  side-branch charge (h2, production PDHD). Only 9–12 % of the skipped rows have dQ ≤ 0; the rest are low but
  positive.
- **The PDVD set the owner viewed is pre-flip** (`q29flip`, 2026-09-13; flip `8fc6070e`, 2026-09-15). The
  PDHD set is production. On PDVD production v1 and v3 are fixed, and cl80 is no longer tagged, so it is not drawn
  at all. The two PDHD spots are **flip-made local defects**. Arm-wide, both flips *reduce* holes (PDHD −25 %,
  PDVD −37 % on clusters tagged in both arms).
- Zig-zag is the doc 101/102 lattice mechanism; this doc does not re-derive it.

## 0. Repro

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img
S=pdvd/docs/nf_sp_img_clus/scripts F=pdvd/docs/nf_sp_img_clus/figs

# sec 2-5 -- per-spot figures + TSV (pre-flip vs production arm), read-only
python3 $S/d110_spot_figs.py --det pdvd --out $F/110    # figs/110_v{1,2,3}_{bee,3d,2d,profile}.png, 110_pdvd_spots.tsv
python3 $S/d110_spot_figs.py --det pdhd --out $F/110    # figs/110_h{1,2}_{bee,3d,2d,profile}.png, 110_pdhd_spots.tsv

# sec 1, 3, 6 -- whole-arm census, read-only (about 20 s)
python3 $S/d110_gap_census.py > $F/110_gap_census.txt

# sec 2 -- the client cut in the DEPLOYED Bee bundle (command and excerpt inside)
cat $F/110_bee_client_cut.txt

# sec 1 -- the PDVD Bee set on production (owner-approved upload, 2026-09-16)
python3 $S/d110_bee_sets.py /home/xqian/tmp/d110/bee
(cd /home/xqian/tmp/d110/bee && bash /home/xqian/toolkit-dev/wcp-porting-img/pdvd/upload-to-bee.sh d110_pdvd_stm5.zip)
```

Arms (all existing, read-only):

| det | arm | what | events |
|---|---|---|---|
| PDVD | `q29flip` | the arm in Bee set 51c1e410; pre-flip production (ROOT 2026-09-13) | 120 |
| PDVD | `d103v0` / `d103vflip` | pre-flip / flipped production, same `p100flip` pctrees (doc 103 sec 14) | 120 / 120 |
| PDHD | `d101hnew` / `d108hflip` | pre-flip / flipped production, same `d51hclus` pctrees (doc 108) | 61 / 61 |
| PDHD | `d109hstm` | the arm in Bee set c3165b38; ≡ `d108hflip` on the STM trees (doc 109 G1) | 5 |

At the spots, `q29flip`'s STM rows equal `d103v0`'s (506/166 rows for cl80/cl25, same q<0 counts). The census
agrees to within 0.7 point on every rate.

## 1. The spots, and which generation each Bee set shows

| spot | Bee set, event | run_event [arm] | cluster | click (x, y, z) cm | complaint |
|---|---|---|---|---|---|
| v1 | 51c1e410 evt 1 | PDVD 039349_20 [`q29flip`] | 80 | (313.7, 95.9, 124.9) | zig-zag, off the image |
| v2 | same | same | 80 | (94.5, 30.7, 165.0) | zig-zag, off the image |
| v3 | same | same | 25 | (−270.2, −243.7, 36.0) | gap |
| h1 | c3165b38 evt 0 | PDHD 029107_16 [`d109hstm`] | 108 | (75.3, 438.3, 450.1) | gaps |
| h2 | same | same | 106 | (182.0, 537.4, 401.8) | gap, zig-zag |

Every click lands within 0.07 cm of its cluster's image and 0.9–2.3 cm of its `stm_fit` rows
(`110_gap_census.txt` sec 0). The owner's first h2 click, cl115 at (278.5, 474.6, 299.1), is a through-going
muon (`TaggerCheckSTM: cluster 115 already TGM; skipping`). No fit row of any layer lies within 143 cm of it,
so it was withdrawn and replaced by the cl106 click above.

**The two sets were on different trajectory configs.** PDHD c3165b38 is production (`charge_stepped` + the
two fit knobs, flipped by `b9ce8b4d`). PDVD 51c1e410 was built from `q29flip`, which ran before the PDVD flip
`8fc6070e`. **Erratum to doc 109:** its line "Toolkit at the run: `b9ce8b4d`" holds for the PDHD arm only.

**New PDVD set on production** (arm `d103vflip`, uploaded 2026-09-16, `d110_bee_sets.py`):
<https://www.phy.bnl.gov/twister/bee/set/b6ad4f38-b7fe-408e-bd46-ba5061b48d06/event/list/>
— 0 `039349_2` (3 `stm_fit` clusters), 1 `039349_20` (8), 2 `039349_18` (12), 3 `039252_8` (8), 4 `039253_6` (12).
Served event 1 `stm_fit` member = staged file, byte for byte (68,818 bytes). In production cl80 is no longer
tagged STM (`STM=0`, status 3), so it is **absent** from event 1's `stm_fit` layer. The old set 51c1e410 stays
as it was.

## 2. Gaps: the fit is continuous; the viewer skips rows with q < 0

**Symptom.** Holes of 1–4 cm in the `stm_fit` trajectory, with the image continuous underneath.

**Root cause.** Three facts, each checked:

1. **The client drops the rows.** Bee's `initData` skips every point with `q < QTHRESH`, where QTHRESH = 0
   for every layer but `truth`/`L1` (source `wire-cell-bee3/events/static/js/bee/physics/sst.js:47-52`).
   The **deployed** bundle does the same:
   `s=0;("truth"==this.name||"L1"==this.name)&&(s=500);for(...)null!=e.q&&e.q[r]<s||i.push(r)`
   (`figs/110_bee_client_cut.txt`, fetched from `twister/static/js/bee/dist/bee.js`). `q == 0` is still drawn.
2. **`stm_fit` writes negative q; `track_fit` does not.** Both encode `q = dQ·dQdx_scale + dQdx_offset` =
   `dQ·0.1 − 1000` (pr.jsonnet, both detectors). The PR `track_fit` writer clamps `if (charge < 0) charge = 0;`
   (`clus/src/MultiAlgBlobClustering.cxx:1181`, vertex rows `:1218`). The `stm_fit` writer appends the raw value
   (`:3130`). So `q < 0 ⇔ dQ < 10 ke` on a row of about 0.6 cm, i.e. under about a third of a MIP row (the spot
   window medians are mostly 40–55 ke/cm).
3. **What Bee shows is exactly the ROOT rows.** For every spot cluster present in a layer, the zip `stm_fit`
   rows equal `T_rec_charge` (tracking-stm.root) row for row: same count, |Δq| ≤ 0.005, |Δx| ≤ 0.0005 cm,
   i.e. JSON rounding (`d110_spot_figs.py` asserts it; `zip_eq_root` column in the TSVs).

**The persisted fit has no gaps** (census sec 3): step p50 0.61–0.62 cm, p99 0.94–0.95 cm on all five arms.
Steps > 3 cm are 0.11 % (424 of 384,354 PDHD production; 563 of 559,499 PDVD production). Those are the
doc-102 off-image chords, not the owner's holes.

**Every named hole is a q<0 run** (`figs/110_*_bee.png`: top row = every persisted row, × = q<0; middle row =
what Bee draws; bottom row = `track_fit` as drawn). Window ±12 cm of arc (±24 cm for h2):

| spot | arm | q<0 rows | longest run: rows / cm | max distance fit → image ridge (cm) | dQ/dx median (ke/cm) |
|---|---|---|---|---|---|
| v3 cl25 | `q29flip` (shown) | 10 | **8 / 4.0** | 2.27 | 44.8 |
| v3 cl25 | `d103vflip` | 2 | 2 / 0.6 | 1.19 | 54.1 |
| h1 cl108 | `d101hnew` | 2 | 2 / 0.5 | 1.57 | 53.2 |
| h1 cl108 | `d109hstm` (shown) | 10 | **7 / 3.9** | 2.14 | **25.4** |
| h2 cl106 | `d101hnew` | 0 | – | 1.14 | 52.5 |
| h2 cl106 | `d109hstm` (shown) | 5 | **5 / 2.7** | 2.62 | 48.5 |
| v1 cl80 | `q29flip` (shown) | 5 | 5 / 2.3 | 2.34 | 39.6 |
| v1 cl80 | `d103vflip` | 0 | – | 1.20 | 51.1 |
| v2 cl80 | `q29flip` (shown) | 5 | 3 / 1.1 | 2.32 | 51.3 |
| v2 cl80 | `d103vflip` | 0 | – | 1.55 | 54.2 |

At v3 one row inside the hole carries q = +3616 (row 138). It is drawn alone in the middle of the gap, which is
why the stretch reads as "broken" rather than "missing".

**Why it hid.**
- The PR `track_fit` layer draws the same kind of rows at q = 0: 15.2 % of PDHD production's tagged-cluster
  `track_fit` rows, 10.3 % of PDVD's (census sec 1). They show as zero-charge (blue) points, so PR trajectories
  never show holes.
- Doc 102 sec 1 recorded "Bee q ≤ 0 is a display value" and the clamp asymmetry, but read it as a colour, not
  as rows the client deletes.

**Fix (not built; sec 7).** A clamp in the `stm_fit` writer behind a default-OFF knob. It closes the display
holes, not the low charge.

## 3. Why the fitted charge is below 10 ke on those rows

**The solve.** `TrackFitting::dQ_dx_fit` shares the measured 2-D charge among the fitted rows through a
Gaussian response matrix per plane. It solves a regularised linear least-squares system with BiCGSTAB and no
positivity (`TrackFitting.cxx:9265`, `dQ[i] = pos_3D(i)` at `:9279`), identical to the prototype
(`PR3DCluster_dQ_dx_fit.h`: `lambda = 0.0005` :933, `solveWithGuess` :946, `dQ.push_back(pos_3D(i))` :985).
It is not a port bug.
- Response entries exist only for live cells with charge (`if (value > 0 && prow.charge > 0 && prow.flag != 0)`,
  `:9053/:9081/:9109`).
- `reg_flag_*` fires where a row projects onto a dead cell (`:9051`) or outside the cluster's 2-D cell set
  (`:9122-9146`), and raises that row's smoothness weight (`:9198-9220`).
- The persisted pass is the second-round `adjusted_segment` fit, regularisation on
  (`TaggerCheckSTM.cxx:3780`, recorded at `:3782`).

A row whose projection has left the track's charge, or that shares its cells with its neighbours, gets its dQ
from the regulariser and the neighbours, not from its own data.

**Whole-arm measurement** (census sec 2, every fitted row of every fitted cluster). *Off-charge* on a plane means
no charge of the cluster within ±1 wire on the row's own slice.

| q<0 fraction, by planes off-charge | 0 | 1 | 2 | 3 | share of all q<0 rows at 0 / 1 / 2 / 3 |
|---|---|---|---|---|---|
| PDHD `d108hflip` | 3.8 % | 22.5 % | 28.2 % | 16.9 % | 66 / 15 / 12 / 8 % |
| PDVD `d103vflip` | 2.2 % | 11.7 % | 45.2 % | 33.9 % | 50 / 23 / 17 / 10 % |

| q<0 fraction, by reg_flag × degenerate planes (pixel separation < 0.5) | 0 deg | 1 | 2 | 3 |
|---|---|---|---|---|
| PDHD `d108hflip`, no reg_flag | 2.0 % | 5.2 % | 7.1 % | 13.2 % |
| PDHD `d108hflip`, reg_flag | 15.0 % | 29.9 % | 37.0 % | 40.6 % |
| PDVD `d103vflip`, no reg_flag | 0.9 % | 6.9 % | 8.7 % | 9.9 % |
| PDVD `d103vflip`, reg_flag | 11.2 % | 42.7 % | 48.4 % | 48.6 % |

- Being off the charge, dead cells, and shared cells each raise the rate 5–20×. Together they are the strongest
  predictors.
- Rows on charge in every plane still hold half to two thirds of all q<0 rows, because they are about 90 % of
  all rows. A low row can also sit on charge when its neighbours take the charge.
- Of the q<0 rows of tagged clusters only **9–12 %** have dQ ≤ 0 (PDHD production 698 of 7,333; PDVD 782 of
  6,741; census sec 1). A positivity constraint would therefore leave about 90 % of the holes as they are.

**Caveat on the on-charge test.** d102's exact own-cell test fails 20.0 % / 24.6 % of PDVD production U / V rows
against 4.5 % of W, and PDHD's 7.7–8.7 % on all planes. With ±1 wire PDVD drops to 5.1 / 6.2 / 3.2 % (census
2c). That is a rounding offset on the PDVD induction planes, not charge. The census strata use ±1 wire. The spot
TSVs carry both (`stm_onq_*` exact, `stm_onq1_*` ±1). PDVD U/V own-cell fractions here, and in doc 102's PDVD
examples, read low for that reason.

## 4. The spots, one by one (`figs/110_<spot>_{bee,3d,2d,profile}.png`)

- **v1 (PDVD cl80, pre-flip).** The STM fit bumps 1.5 cm off a thin image (transverse rms ≈ 0.5 cm) over about
  4 cm, and 5 rows fall below 10 ke right after the bump. `q29flip`'s PR fit bumps too. Production (`d103vflip`)
  runs flat on the image: no q<0 row, ridge max 1.20 cm, wiggle max 0.55 cm (was 1.59). But cl80 is untagged in
  production, so it is not drawn.
- **v2 (PDVD cl80, pre-flip).** A 2.4 cm spike 8–10 cm from the click holds a 7-row, 3.4 cm hole (rows 417–423,
  `110_v2_bee.png`; outside the ±12 cm arc window of the TSV), plus a 1.3 cm dip near the click. Production has no q<0 row, but still wiggles up to
  1.31 cm (ridge max 1.55). The track is 15.7° from drift: `pu/pv/pw` barely move while `pt` does, so the
  transverse position rests on the wire lattice. That is the doc-101 snapping regime. Not drawn in production
  (untagged).
- **v3 (PDVD cl25, pre-flip).** The fit kinks about 2 cm toward an off-track charge bump in the image, and the
  8-row, 4.0 cm hole sits on the kink. Production follows the track (ridge max 2.27 → 1.19): 2 q<0 rows, hole
  0.6 cm, still drawn (cl25 tagged). U on-charge is low in both arms (±1 wire 0.83 / 0.74).
- **h1 (PDHD cl108, production).** The image bends by about 2 cm near the click. The pre-flip STM fit and the PR
  fit (both arms) follow the bend. The production STM fit runs straight across it (the doc-106 "levers
  straighten" effect) along the empty lower edge of the W charge band (`110_h1_2d.png`, W own-cell on-charge
  1.00 → 0.81, `reg_flag_w`). Its 7-row, 3.9 cm hole sits there. The local dQ/dx median halves, 53.2 → 25.4 ke/cm.
- **h2 (PDHD cl106, production).** About 18–20 cm of arc before the click, the production fit climbs 4 V wires at a
  **single time slice** (x fixed at 188.0 cm) into side-branch charge and drops back 8 slices later
  (`110_h2_2d.png`). The 5 stacked rows share their W cells (pixel separation 0.03, `110_h2_profile.png`) and
  nearly their U cells, so the solve cannot tell them apart. They fall to about 8 ke, which is the 2.7 cm hole.
  `reg_flag_u` covers about 10 cm around it, and the fit leaves the track by up to 4.5 cm in the wire plane
  (`110_h2_bee.png`). The pre-flip fit runs straight on the track with no q<0 row. PR finds 8–9 short segments
  in this window in both arms, i.e. real branching charge. The fold metric (a backward step) does not fire here
  or at any spot: these excursions go sideways, not back.

## 5. Zig-zag

Not re-derived. Docs 101/102 located it in the trajectory fit: the q² weight and the rounded association window
snap rows to the wire lattice, worst at coarse pitch and when the retile cloud is off-charge. Both levers are
now production on both detectors. What this doc adds:

- **Arm-wide the flips cut rows with wiggle > 1 cm by 57–61 %** on clusters tagged in both arms (PDHD 669 → 288,
  PDVD 1,095 → 422; census 1b).
- **The PDVD zig-zag the owner saw (v1, v2) is the pre-flip one.** In production v1 is flat. v2 keeps 1.3 cm
  wiggles on a near-drift track.
- **The PDHD excursions (h1 corner cut, h2 single-slice climb) are the flip's own local defects.** They are
  smooth or sideways, so a chord-deviation metric does not rank them (h1 wiggle 0.89 → 0.47 while the ridge
  distance grows). The ridge distance and the per-plane on-charge test do.

## 6. Did the flips make the holes better or worse?

Census sec 1b, on the clusters tagged in **both** arms of each pair (same pctrees):

| | common tagged clusters | q<0 rows | holes ≥ 3 rows | hole length (cm) | clusters with a hole | rows wiggle > 1 cm |
|---|---|---|---|---|---|---|
| PDHD `d101hnew` → `d108hflip` (61 evt) | 261 | 7,470 → 6,064 | 774 → **577** | 3,947 → 3,288 | 193 → 162 | 669 → 288 |
| PDVD `d103v0` → `d103vflip` (120 evt) | 422 | 6,264 → 4,561 | 672 → **422** | 3,321 → 2,625 | 253 → 195 | 1,095 → 422 |

**Both flips reduce holes arm-wide** (−25 % PDHD, −37 % PDVD). h1 and h2 are local counter-examples, not the
trend. They are still common in production. On all tagged clusters (census sec 1):
- **PDHD:** 693 holes in 585 m of fit (11.9 per 10 m); 6.6 % of the fitted length lies in them.
- **PDVD:** 588 holes in 970 m (6.1 per 10 m); 4.0 % of the length.

That is why the owner meets them on almost every track.

## 7. What would fix each (none built)

| defect | lever | gate |
|---|---|---|
| display holes | clamp `q < 0` to 0 in the `stm_fit` Bee writer (`MultiAlgBlobClustering.cxx:3130`), as `track_fit` does at `:1181`, behind a default-OFF knob on the `bee_points_sets` entry | off: compiled PR config and every zip member hash identical. On: zero `stm_fit` rows with q<0, the row count unchanged, `T_stm_*`, `T_rec_charge` and calib dumps identical |
| low fitted charge on off-charge / stacked rows (this also feeds the STM tagger's dQ/dx: h1's local median halves) | the trajectory, not the solve: the doc 101/102 levers moved it both ways (−25/−37 % holes arm-wide, new h1/h2-type excursions) | a trajectory change moves tags: doc-56-style STM/Michel hand-scan grade on both detectors |
| dQ ≤ 0 rows | positivity in `dQ_dx_fit` (NNLS or clamp-and-refit) | reaches only 9–12 % of the holes. Tagger-affecting: same hand-scan grade. Not recommended as a first lever |

The clamp is the smallest change that answers "we should have a continuously fitted trajectory" for the
display. It hides nothing: a clamped row is drawn at zero charge, as on the PR layer.

## 8. Not concluded

- **Causality of the sec 3 strata.** Off-charge, reg_flag and degeneracy are measured correlates of a low row.
  No counterfactual arm separates them.
- **How often production makes an h1/h2-type excursion** (a hole present in production but not pre-flip at the
  same place). The net count is lower (sec 6), but the new-vs-removed split was not built.
- **The PDVD U/V own-cell rounding offset** (sec 3 caveat). Not traced to a convention. It may bias doc 102's
  PDVD on-charge numbers.
- **The cl106 side branch.** About 95 image points within 25 cm of h2 lie more than 3 cm from the STM fit in
  both arms. At cluster scale, 450 of 4,196 image points are more than 3 cm away, up to 61 cm (the doc-102 chord
  class). Not studied here.
- **PDVD dead channels are not drawn** in the 2-D figures (no verified raw→rank map for PDVD in the script).
  The per-row `reg_flag_*` carries that information.

## 9. Files

| path | what |
|---|---|
| `scripts/d110_spot_figs.py` | sec 2, 4, 5: per-spot figures + TSV, zip = ROOT assert (fork of `d102_spot_figs.py`, untouched) |
| `scripts/d110_gap_census.py` | sec 1, 2, 3, 6: whole-arm census |
| `scripts/d110_bee_sets.py` | sec 1: PDVD 5-event set on production (fork of `d109_bee_sets.py`, untouched) |
| `figs/110_{v1,v2,v3,h1,h2}_{bee,3d,2d,profile}.png` | the spot figures (top row of `_bee` = every persisted row, middle = what Bee draws) |
| `figs/110_pdvd_spots.tsv`, `figs/110_pdhd_spots.tsv` | per-spot, per-arm metrics |
| `figs/110_gap_census.txt` | census output |
| `figs/110_bee_client_cut.txt` | the q cut in the deployed Bee bundle and in the source |
| `109_stm-fit-layer-scope-pdhd-vs-pdvd.md` | erratum (arm generations) |
