# Run 27409: multi-track over-clusters never separated (FV envelope blindness)

Five run-27409 events (drift group 0/2) kept 2–3 crossing cosmics in one
cluster.  Diagnosis + fix in the toolkit doc `clus/docs/clustering-separate-fv.md`;
this note pins the per-event evidence.  Cluster ids below are from the
pre-fix outputs (the bee set the issues were reported from).

## Symptoms and diagnosis

Instrumented gate tracing (temporary `WCT_SEP_DEBUG` prints, since removed)
showed every cluster in all five events with `num_outside_points = 0`,
`num_outx_points = 0`, `independent_surfaces = 0` while their hull extremes
sat ON the detector walls (e.g. hy = 605.97 cm vs FV_ymax = 606.0; ly = 7.86
vs FV_ymin = 7.61).  The configured FV was the exact active envelope, making
JudgeSeparateDec_2 and the `flag_top` ladder branch unsatisfiable — the
user-visible "four points on surfaces" were seen by the hull but never counted
as outside the FV.

| evt | cluster | topology | binding gate | after fix |
|---|---|---|---|---|
| 40900 | 23 (+26) | iso band + 3 tracks; ONE 5795-blob, 860 cm pre-separate cluster spanning floor-to-top | proceeded via the no-top ladder (11.2° < 25) but Separate_1 had only 6 fallback endpoints; band_recarve then mis-pooled the crossing tracks (seed 16.0/10.0 cm widths at 10.9°) | 3 tracks in 3 clusters; recarve correctly silent (seed angle ≥ 15° + ribbon width gates) |
| 40904 | 31 | two forking near-parallel corner-to-corner tracks (21.9k pts, r1 = 0.019, 46° from ⊥-beam) | Dec_2: nout=2 nsurf=2 but nindep=2, nfar=0 — the endpoint-midpoint sits 47.6 cm from the cluster in the empty space between the prongs (prototype cap 25 cm) | split (`far_point_mid_dis = 60 cm`) |
| 40908 | 27 | 3 tracks, two crossing as an X with interior arm endpoints | ladder 43° > 35; Dec_2 blind; after the FV fix the carve ran (5504 blobs → 15 pieces) but Separate_2's 5 cm relink rejoined the X at the crossing | 3 tracks in 3 clusters (`track_recarve`: arms 2584/8352 npoints, cross fractions 0.93/0.42, residuals 5.3/8.7 cm — a "T": one track ends on the other) |
| 40912 | 16 | 2 tracks crossing near the top, y_max 597.8 = 8.2 cm below the wall | needs `flag_top` (y > FV_ymax = 606.0), unreachable | split (top-branch ladder, 25.5° < 33, via the 15 cm inset) |
| 40924 | 22 | 3 tracks (14.4k pts, r1 = 0.50) | ladder 36.0° > 35 (just); Dec_2 blind | 3 tracks in 3 clusters |

Key structural finding: the FV blindness also **starves Separate_1 of path
endpoints** (it carves between independent points), so even the two events
that did proceed were carved against dense-extreme fallback points only.

## Fixes (toolkit ee054213)

1. 15 cm y/z FV insets in `cfg/pgrapher/experiment/pdhd/clus.jsonnet`
   (y 22.61–591.0, z 15.23–447.30 cm) — the uboone prototype's operating
   point, where all Dec_2 thresholds were tuned.  Same for PDVD (±321.4 y,
   15.05–284.25 z).
2. `drift_side_fv_x=true`: drift-group scopes keep their drift side's FV x
   (group02: [−357.985, −2.54] cm) so out-of-time tracks past the cathode
   (apparent x up to +126 cm) hit the no-T0 x channel.
3. `far_point_x_cut=14*wc.cm`, `far_point_mid_dis=60*wc.cm` (defaults stay
   prototype-exact at 140 cm / 25 cm).
4. `track_recarve=true`: k=2 3D-line self-split of an X/T member with kink
   veto (crossing must be interior to ≥1 arm) and thin-arm residual gate.
5. `band_recarve` seed gates: ribbon width ≥ 6 cm on both seeds, seed angle
   ≥ 15° (39324 genuine band seeds: 25–39°).

## Regression

- New binary + old configs: byte-identical (027409 evt 40908 md5).
- Clustering-only rerun vs archived baselines (`bak-pre-fv/` in each work
  dir): PDHD 027409 evts 0–7,12 + 027380 evts 0–7 and PDVD 39252/39253
  evts 0–4 + 39324 evts 0–10.  Flow analysis: modest separation-family
  reorganizations only (typically +1–2 clusters/event; the five diagnosed
  events and 39324 evt 4 are the largest), zero shredding.  ≤0.3% of a big
  cluster's display points can enter/leave the Bee dump when a small carve
  fragment crosses the isolated-trash threshold (blobs unchanged).
- PDVD 39324 evt 0 refinement invariants (three pairs, 2-band complex,
  whole vertical track) re-verified after every tuning step.

## Round 3: DNN-SP imaging provenance + collinear_interior (toolkit fbc4ad33)

The first post-fix full-chain bee set was built from the WRONG SP frames:
`run_img_evt.sh` defaults to `-d off` and, finding no
`protodunehd-sp-frames-anode*.tar.bz2` in the work dirs, silently fell back
to the OLD `input_data` traditional-SP archives — so the freshly regenerated
DNN-SP frames (with the prolonged-W fix, toolkit 50239595) never reached
imaging.  **PDHD imaging from DNN-SP output needs `-d on` explicitly**
(PDVD is immune: its work dirs carry `protodune-sp-frames-* ->
*-dnnroi-*` symlinks).  Verified the work-dir frames DO carry the W fix
(ch 9543 evt 40920: 21.3 % vs 8.6 % pre-fix coverage; ch 4532 evt 40924:
20.3 % vs 6.2 %) before rerunning imaging + clustering.

On the correct (denser) DNN-SP imaging, evt 40900's two crossing tracks
re-merged through a NEW mechanism: the carve shed a ~24 cm mid-track
fragment of one track (holding the user-flagged point) that a later
proximity merge attached to the other track's cluster.  Fixed by the
`collinear_interior` knob (toolkit fbc4ad33): collinear_recover additionally
absorbs whole short (<50 cm) sibling fragments lying along a track's axis
inside its span.  All five flagged events re-verified PASS; PDVD 39324
evt 0 invariants PASS; OFF-check content-identical.
