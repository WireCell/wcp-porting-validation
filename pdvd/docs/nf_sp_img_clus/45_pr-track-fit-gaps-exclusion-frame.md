# 45 — Why the PR chain's `track_fit` is full of gaps while `stm_fit` is smooth: the exclusion tournament compares each 2-D cell in the wrong drift frame

**Status (2026-09-05, round 2).** Root cause found, proven with the fitter's own exclusion
dump on evt 298595 and on 120 events, fixed behind the `TrackFitting` knob `excl_t0_frame`,
graded on the 120-event STM set (sec 1-5). Round 2 (same day, owner: *"Guard cal_kine_dQdx
first, then flip … then hand-scan the moved vertices"*, *"flip it on for PDVD, commit and
push"*): the NaN-Enu defect is guarded behind `kine_dqdx_skip_zero_dx` (sec 8), **both knobs
are PDVD PRODUCTION since 2026-09-05** (sec 9; the production arm reproduces the graded arm
120/120), the moved vertices are hand-scanned blind (sec 10: at a track end 14 -> 22 of 40),
SBND is measured and NOT flipped (sec 11), and the two leftover `track_fit` questions are
answered (sec 12). Gate record for round 2: `stm/gates/d45_kine_guard_gate.txt`. Doc pdvd/30's
"duplicate segment" attribution is superseded (sec 3; a dated correction is in doc 30).

Owner (2026-09-05), on the round-3 Bee set for evt 298595: *"if you compare the stm_fit
with the track_fit, you can see clearly the track_fit looks quite weird, gaps in the
tracks etc. … the PR is a more sophisticated way to run the track trajectory fit, it
should be similar to the stm_fit, but allow for more segments … the results seem to
suggest there are some major bugs in the PR chain."* And: *"can you also provide an
explanation on why SBND neutrino does not have this issue, is it because the T0 of
neutrino candidate happens to be close to zero time?"* — yes; sec 2.4 gives the number.

## 0. Repro

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/pdvd
# pins (libWireCell*.so + .pcm copied from local/lib): ref = today's production binary
#   /home/xqian/tmp/d45_libpin/ref   libWireCellClus.so md5 fd273dc8f000780f5503af29dca88881
#   /home/xqian/tmp/d45_libpin/new2  the shipped knob (md5 in stm/gates/d45_excl_frame_gate.txt)
# toolkit 4e55b145 -> db166521 (the knob); wcp-porting-img 029a0b50 -> 8fc8b70f (this doc)
S=docs/nf_sp_img_clus/scripts

# A. baseline: the full PR tail (-nu) with today's production knobs, 120 events, stm_fit dump on
ARM=d45nu0 PIN=ref PINROOT=/home/xqian/tmp/d45_libpin MODE=nu JOBS=16 $S/run_d45_arms.sh
python3 $S/d45_trackfit_vs_stmfit.py --tsv /home/xqian/tmp/d45/d45nu0_census.tsv work/*_d45nu0

# B. the proof: every exclusion decision on evt 298595 (doc pr/109 sec 9 instrument)
printf '39252 2 298595 1\n' > /home/xqian/tmp/d45/ev298595.txt
ARM=d45dump PIN=ref PINROOT=/home/xqian/tmp/d45_libpin MODE=nu JOBS=1 EVENTS=/home/xqian/tmp/d45/ev298595.txt \
  WCT_EXCL_DUMP=/home/xqian/tmp/d45/excl_298595.txt WCT_TRAJ_DUMP=/home/xqian/tmp/d45/traj_298595.txt \
  WCT_DQDX_DROP_DEBUG=1 $S/run_d45_arms.sh
python3 $S/d45_excl_frame.py /home/xqian/tmp/d45/excl_298595.txt work/039252_2_d45dump

# C. isolation 2x2 on 298595 (pin ref)
for a in d45exoff:"-S fit_exclusion=false" d45keep:"-S dqdx_fit_keep_all_points=true" \
         d45both:"-S fit_exclusion=false -S dqdx_fit_keep_all_points=true"; do
  ARM=${a%%:*} PIN=ref PINROOT=/home/xqian/tmp/d45_libpin MODE=nu JOBS=1 EVENTS=/home/xqian/tmp/d45/ev298595.txt \
    EXTRA="${a#*:}" $S/run_d45_arms.sh; done
python3 $S/d45_trackfit_vs_stmfit.py work/039252_2_d45nu0 work/039252_2_d45exoff work/039252_2_d45keep work/039252_2_d45both

# D. SBND negative control: the same census on pr/87 zips, and the same dump on 6 nueCC48 events
python3 $S/d45_trackfit_vs_stmfit.py --all-clusters ../sbnd/sbnd_xin/work-87flip-ncpi0/pr_evt{399860,21073,314838}
cd ../sbnd/sbnd_xin && LD_LIBRARY_PATH=/home/xqian/tmp/d45_libpin/ref WCT_EXCL_DUMP=/home/xqian/tmp/d45/excl_sbnd_nuecc48.txt \
  PR_JOBS=6 ./run_pr_chain_batch.sh work-nuecc48-d97fv work-d45sbnd-ref-nuecc48 data 10550 46363 81597 360535 256587 433451

# E. the fix ON (pin new2): smoke + dump on 298595, then the 120 events; knob-off gates in the gate record
ARM=d45on2 PIN=new2 PINROOT=/home/xqian/tmp/d45_libpin MODE=nu JOBS=1 EVENTS=/home/xqian/tmp/d45/ev298595.txt \
  EXTRA="-S excl_t0_frame=true" WCT_EXCL_DUMP=/home/xqian/tmp/d45/excl_298595_on2.txt $S/run_d45_arms.sh
ARM=d45on PIN=new2 PINROOT=/home/xqian/tmp/d45_libpin MODE=nu JOBS=16 EXTRA="-S excl_t0_frame=true" $S/run_d45_arms.sh
python3 $S/d45_trackfit_vs_stmfit.py --tsv /home/xqian/tmp/d45/d45on_census.tsv work/*_d45on

# ---- round 2 (toolkit db166521 -> c1cd59b3 the guard knob; wcp-porting-img 007ae9bb -> a6313bd7 this round) ----
# F. the guard, knob-off gates (pins new2 = the doc-45 knob, new3 = the guard): PDVD -nu/-stm 2 events, SBND bare 3 + exclusion-active 6
for m in nu:d45g1n3 stm:d45g2n3; do ARM=${m#*:} PIN=new3 PINROOT=/home/xqian/tmp/d45_libpin MODE=${m%%:*} JOBS=2 \
  EVENTS=/home/xqian/tmp/d45/events2.txt $S/run_d45_arms.sh; done
python3 $S/d40r3_hash_gate.py d45g1n2 d45g1n3 /home/xqian/tmp/d45/events2.txt; python3 $S/d40r3_hash_gate.py d45g2n2 d45g2n3 /home/xqian/tmp/d45/events2.txt
(cd ../sbnd/sbnd_xin && D42_LIBPIN=/home/xqian/tmp/d45_libpin/new3 D42_NO_STMFIT=1 STM_EVENTS="284349 285999 286065" NJOBS=3 ./stm_campaign/run_d42_stmfit.sh d45g3n3 \
  && LD_LIBRARY_PATH=/home/xqian/tmp/d45_libpin/new3 PR_JOBS=6 ./run_pr_chain_batch.sh work-nuecc48-d97fv work-d45sbnd-off3-nuecc48 data 10550 46363 81597 360535 256587 433451)
# G. both knobs ON via -S (d45on3), then the FLIP in wct-pr-perevt.jsonnet and the canonical production arm (d45prod)
ARM=d45on3 PIN=new3 PINROOT=/home/xqian/tmp/d45_libpin MODE=nu JOBS=12 EXTRA="-S excl_t0_frame=true -S kine_dqdx_skip_zero_dx=true" $S/run_d45_arms.sh
python3 $S/d45_downstream.py d45on d45on3 --tsv /home/xqian/tmp/d45/on_vs_on3.tsv
ARM=d45prod PIN=new3 PINROOT=/home/xqian/tmp/d45_libpin MODE=nu JOBS=14 $S/run_d45_arms.sh
python3 $S/d40r3_hash_gate.py d45on3 d45prod; python3 $S/d45_trackfit_vs_stmfit.py --tsv /home/xqian/tmp/d45/d45prod_census.tsv work/*_d45prod
# H. the blind vertex hand-scan (sheets under /home/xqian/tmp/d45/vscan; labels + key + movers copied to figs/45_vertex_scan/)
python3 $S/d45_vertex_scan.py d45nu0 d45on3 --out /home/xqian/tmp/d45/vscan --n 40 --seed 45
python3 $S/d45_vertex_scan.py --unblind /home/xqian/tmp/d45/vscan/labels.tsv --out /home/xqian/tmp/d45/vscan
# I. SBND decision arms (48 nueCC48 + 19 NCpi0 events), knob OFF vs ON through the runtime-JSON copy
(cd ../sbnd/sbnd_xin && for set in nuecc48:data ncpi0:sim; do s=${set%%:*}; r=${set#*:}; \
  LD_LIBRARY_PATH=/home/xqian/tmp/d45_libpin/new3 PR_JOBS=8 ./run_pr_chain_batch.sh work-$s-d97fv work-d45sbnd-off3all-$s $r; \
  LD_LIBRARY_PATH=/home/xqian/tmp/d45_libpin/new3 PR_JOBS=8 SBND_TRACKFIT_JSON=/home/xqian/tmp/d45/sbnd_track_fitting_d45on.json ./run_pr_chain_batch.sh work-$s-d97fv work-d45sbnd-on3all-$s $r; done)
```

Gate records: `stm/gates/d45_excl_frame_gate.txt` (round 1), `stm/gates/d45_kine_guard_gate.txt` (round 2). Scripts: `d45_trackfit_vs_stmfit.py` (the
census: coverage / ghost / winner / drop per STM cluster), `d45_excl_frame.py` (the dump
reader), `run_d45_arms.sh` (fork of `run_d44_arms.sh` with `MODE=nu|stm` and the debug-env
passthrough).

## 1. Symptom: it is event-wide, and it is the number of segments

`stm_fit` (TaggerCheckSTM's single-track fit) and `track_fit` (TaggerCheckNeutrino's
multi-segment PR fit) fit the same clusters on the same charge with the same
`TrackFitting` code and the same parameter file. The census below grades every STM-tagged
cluster by **coverage** = the fraction of its `stm_fit` points that have a `track_fit`
point within 1 cm, and attributes the loss to the fitter's own counter, the DEBUG line
`do_multi_tracking: pre-dQ/dx form_map_graph dropped N of M trajectory point(s) with zero
plane quantity` (`TrackFitting.cxx:9571-9576`; a trajectory point is dropped when, after
association, it has zero charge on all three planes).

**Evt 298595 (039252/2), baseline arm `d45nu0` = today's production knobs:**

| cluster | raw pts | stm_fit | track_fit | segments | segments with >2 pts | coverage | median dist to stm_fit | worst drop pass |
|---|---|---|---|---|---|---|---|---|
| 39 | 304 | 64 | 34 | 3 | 2 | 0.30 | 4.80 cm | 68/81 |
| 55 | 336 | 69 | 64 | 4 | 1 | 0.81 | 0.23 | 36/97 |
| 79 | 331 | 211 | 260 | 1 | 1 | 0.89 | 0.30 | none |
| 83 | 289 | 49 | 13 | 4 | **0** | **0.10** | 7.16 | 63/71 |
| 86 | 577 | 109 | 32 | 10 | **0** | **0.22** | 2.02 | 169/179 |
| 103 | 940 | 145 | 145 | 1 | 1 | 0.90 | 0.28 | none |
| 109 | 938 | 239 | 196 | 6 | 2 | 0.74 | 0.31 | 165/353 |
| 113 | 1187 | 114 | 86 | 5 | 2 | 0.11 | 5.68 | 107/187 |

Event total: **5142 of 8458 trajectory points (60.8 %) dropped** over 53 fit passes. The
two clusters with a single segment are perfect; every cluster with two or more segments
loses, and on 11 of the 15 PR-fitted clusters of this event **exactly one segment keeps
its interior points** while every other segment is reduced to its two endpoints (79:
258 + 2; 103: 143 + 2; 55: 53 plus four 2-point segments; 83 and 86: no segment survives,
only the `-1` vertex markers). Those endpoint-only segments are the "gaps" in Bee.

**120 events (`d45nu0`, 594 STM-tagged clusters on 119 events):** coverage quantiles
10/25/50/75/90 = 0.06 / 0.17 / **0.44** / 0.87 / 1.00; 54 % of clusters below 0.5.
The drop counter over all passes: **1 468 898 of 2 884 704 points (50.9 %)**, per-event
median 53 %. Coverage against segment multiplicity:

| segments in the cluster | clusters | median coverage | mean |
|---|---|---|---|
| 1 | 86 | 1.00 | 0.94 |
| 2 | 83 | 0.75 | 0.63 |
| 3 | 78 | 0.38 | 0.44 |
| 4-5 | 116 | 0.39 | 0.45 |
| >= 6 | 206 | 0.31 | 0.37 |

**SBND negative control**, the same script on three pr/87 NCpi0 events (`work-87flip-ncpi0`):
drop fraction **0.7 %, 0.7 %, 1.0 %**; 133 PR-fitted clusters, 11 with three or more
segments, **none** of them one-winner. Same code, 60x difference.

## 2. Root cause

### 2.1 Symptom -> mechanism

The multi-segment fit runs `form_map_graph` three times per `do_multi_tracking`
(`TrackFitting.cxx:9105, 9301, 9514`). Each time, for each interior fit point,
`form_point_association` (`:2641`) collects the 2-D cells (time, wire) of the nearby
blobs, and — when `fit_exclusion` is on, PDVD production since pr/98 — `update_association`
(`:3166-3357`) arbitrates every cell between the segment being fit and its sibling
segments in the same cluster: a cell is kept iff its 2-D distance to the segment's own
cloud is < 0.3 cm, or no sibling's cloud is at least as close. A point whose three
plane sets end up empty has zero quantity and is dropped (`:4184`). The STM single-track
path has no exclusion (`form_map`, `:4318-4425`, never calls `update_association`) and
ends on the gap-*filling* resampler `organize_ps_path` (`:9817`) — that is why `stm_fit`
is smooth.

### 2.2 The frame mismatch

`update_association` turns each cell back into a point with the **geometric**
conversion built at `:705-709`:

```cpp
double raw_x = (coord.time - offset_t) / slope_x;      // raw drift frame, t0 = 0
double raw_y = (coord.wire - offset_u) / slope_yu;
WireCell::Point test_point(raw_x, raw_y, 0);
double min_dis_track = exclusion_closest_2d_dis(segment, test_point, apa, face, 0);
```

and `exclusion_closest_2d_dis` (`:3105-3115`) queries the segment's `"fit"` (or `"main"`)
`DynamicPointCloud`, built from `fits[i].point` / `wcpts` (`PRSegmentFunctions.cxx:136-157`,
`TrackFitting.cxx:9653-9657`) — points in the cluster's **t0-corrected** default scope
(`clustering_switch_scope.cxx:109-115`; `SCECorrection::backward` adds
`dirx * cluster_t0 * v_drift` to x, `SCECorrection.h:107-118`). The 2-D query uses `p.x()`
verbatim (`DynamicPointCloud.cxx:419`). So every own and competitor distance is off by
the drift-frame offset of the cluster. On a PDVD cosmic that offset is metres; the
keep rule degenerates into "the segment whose cloud reaches farthest toward the raw
cell wins every cell of the cluster", the others' interior points lose all three planes
and are dropped, and their segments collapse to two endpoints. One winner per cluster is
exactly what sec 1 measured.

The prototype cannot have this defect: its `update_association`
(`PR3DCluster_multi_track_fitting.h:1023-1046`) derives `offset_t` from the cluster's OWN
first point (`first_t_dis = pts[0].mcell->GetTimeSlice()*ts - pts[0].x`), so cell and
cloud share a frame by construction. The toolkit port replaced that with the geometric
offset — an undocumented port divergence, now recorded in
`clus/docs/porting/porting_dictionary.md`.

### 2.3 Proof: the fitter's own exclusion dump

`WCT_EXCL_DUMP` (doc pr/109 sec 9) writes every decision: own distance `min_dis_track`,
nearest competitor `min_other`, keep, floored. On evt 298595 (815 202 decisions, arm
`d45dump`, attributed to Bee clusters by the fit point), against the prediction
`v_drift * |flash_time + trigger_offset|` (the cluster t0 the transform applies is the
flash time relative to the trigger; `trigger_offset_{bot,top}_us` = -2493.3 / -2464.7 in
the pctree sidecar):

| cluster | decisions | own distance median [cm] (IQR) | predicted v*t0 [cm] | median / predicted | floored | kept |
|---|---|---|---|---|---|---|
| 39 | 65 152 | 61.8 (53-66) | 67.9 | 0.91 | 0 | 0.11 |
| 55 | 54 110 | 352.5 (345-356) | 358.4 | 0.98 | 0 | 0.57 |
| 79 | 10 526 | 235.4 (235-236) | 235.6 | 1.00 | 0 | 1.00 (lone segment) |
| 83 | 76 271 | 520.4 (510-527) | 529.4 | 0.98 | 0 | 0.20 |
| 86 | 239 743 | 583.7 (572-592) | 595.2 | 0.98 | 0 | 0.07 |
| 103 | 21 672 | 475.5 (473-478) | 479.1 | 0.99 | 0 | 0.78 |
| 109 | 280 237 | 205.5 (194-215) | 220.7 | 0.93 | 0 | 0.28 |
| 113 | 45 349 | 423.7 (423-424) | 424.0 | 1.00 | 0 | 0.41 |

Over the whole event: **0.0000 of the cells are inside the 0.3 cm floor, the smallest own
distance ever seen is 39.99 cm, and 25 % of cells are kept.** The tournament never once
compared two sub-centimetre distances. The same offset is visible without the dump:
in `tracking-pr.root`, each cluster's corrected `x` is linear in the raw time slice with an
intercept of `-/+ v_drift * (flash_time + 158..187 us)` — cluster 39: +187 us, 86: -196 us,
79/103/109/83: -158 +/- 2 us.

### 2.4 Why SBND does not show it (the owner's question)

Yes — because the beam candidate's t0 is nearly zero. SBND's PR runs only on the
neutrino candidate, whose cluster t0 is the beam flash time: **0.8-2.0 us** on the pr/87
events (log lines `candidate main cluster N (t0 0.791 us ...)`). The frame offset is
therefore `v_drift * t0` = **1.2-3 mm**, against pitch 3 mm and the 0.3 cm floor. The
same dump on six nueCC48 events (arm `work-d45sbnd-ref-nuecc48`, 11.96 M decisions):
own-distance quantiles 10/25/50/75/90 = 0.19 / 0.32 / **0.55** / 0.87 / 1.22 cm, 23 % of
cells inside the floor, 82 % of own distances under 1 cm, **none above 10 cm**, 62 % kept
with a median of 39 competing segments. So the tournament on SBND is arbitrating real
proximity, biased by a millimetre or two — small, but not zero: pr/109 sec 9 measured that
exclusion strips 30 % of SBND's associated cells and half of the near-vertex ones, and part
of that may be this bias (sec 5 measures the knob on SBND). PDVD is the first detector
whose PR fits clusters with t0 in the **milliseconds**: every cosmic candidate of the
per-bundle chain carries a flash time of 2-6.5 ms relative to a trigger offset of -2.5 ms,
i.e. 0.6-6 m of drift. uBooNE (pr/108: "parity-exact through the association stage")
sits in between and its qlport chain folds the beam t0 so the two frames coincide.

### 2.5 What was eliminated on the way

- Plane-angle / slope inversion in `update_association`: PDVD U/V are at +/-60 deg
  (+/-120 deg on the swapped anode 3, doc 27), W at 0 deg; every divisor is finite.
- The Steiner, terminal and channel-mapping fixes: doc 31 sec 8.3 re-ran cluster 86 after
  them and found the symptom bit-identical; the PR's inputs did move event-wide, the gaps
  did not.
- `dqdx_fit_keep_all_points` (pr/107): it suppresses only the THIRD `form_map_graph`'s
  drop. On 298595 it brings points back (sec 1 table below) but at 0.37-0.40 cm median from
  `stm_fit` versus 0.14-0.21 with exclusion off — the trajectory rounds 1-2 still lost their
  charge and the resampler straightened them (doc 30 M1 already called it a mask).
- `traj_degenerate_wcpts_fallback`: retired; doc 30 round 3 measured it inert.

### 2.6 Isolation 2x2 on 298595 (pin ref)

| arm | drop total | cov 39 | 55 | 79 | 83 | 86 | 103 | 109 | 113 |
|---|---|---|---|---|---|---|---|---|---|
| production (`d45nu0`) | 5142/8458 = 60.8 % | 0.30 | 0.81 | 0.89 | 0.10 | 0.22 | 0.90 | 0.74 | 0.11 |
| `fit_exclusion=false` (`d45exoff`) | 22/378 = 5.8 % | 0.97 | 1.00 | 0.89 | 1.00 | 1.00 | 0.90 | 0.97 | 0.04 |
| `dqdx_fit_keep_all_points=true` (`d45keep`) | 0/0 (nothing reaches the third pass's drop) | 0.98 | 0.97 | 0.89 | 1.00 | 0.94 | 0.90 | 0.98 | 0.51 |
| both (`d45both`) | identical to `d45exoff` | | | | | | | | |

With exclusion off, six of the eight clusters reach 0.97-1.00 and the restored legs sit a
median 0.14-0.21 cm from `stm_fit` — the two fitters agree to two millimetres. Clusters 79
and 103 (single segment) are untouched at 0.89/0.90: their 10 % is the STM fit's own end
coverage (doc 32), not a PR defect. **Cluster 113 is different**: 0.04 even with exclusion
off. Both fits lie on real charge (0.6-0.8 cm from the raw cloud) but the PR graph's three
segments run up a different branch of this 1187-point cluster (z to 240 cm, y to -135) than
the STM trajectory (z to 230, y to -127); `mvfit` substituted its leg 0 by a 39 cm chord.
That is a topology disagreement on a branchy cluster, outside this doc's mechanism, and
it stays at 0.04-0.10 in every arm below.

## 3. What doc pdvd/30 got right, and what it got wrong

Right: the STM=1 tag on cluster 86 is well founded; `fit_exclusion=false` restores the leg
to 0.20 cm of `stm_fit`; `dqdx_fit_keep_all_points` restores a *straight* leg 3.45 cm off
and is not the fix; `traj_degenerate_wcpts_fallback` is inert. All four reproduce here.

Wrong: the mechanism. Doc 30 read the stage-1 trace — "81 -> 76 healthy until a
`do_rough_path` graph edit adds a second segment, then 81 -> 2" — as *exclusion contention
with a duplicate segment*, and recommended fixing the duplicate. The collapse coincides
with the second segment because the tournament only runs when there is a competitor —
any competitor. With the cell 584 cm from every cloud, the first sibling to appear wins
or loses everything; duplication is irrelevant, and `PR::add_segment`'s F9 aliasing
already prevents literal duplicates (11 times on this event). Doc 30 carries a dated
correction blockquote; its recommendation 1 is struck through.

## 4. The fix: `excl_t0_frame` (default OFF)

### 4.1 First cut, measured wrong, superseded

The obvious fix — subtract `dirx * cluster_t0 * v_drift` from the test point's raw x
(the same three quantities `SCECorrection::backward` uses) — was built and smoke-tested
first (`d45on1`, pin `new`). It moved every cluster's own distance to a *constant*
340-369 cm = `v_drift * |trigger_offset|` and changed the sign of the error on the bottom
side: the geometric `offset_t` (`:709`, from the grouping's time offset) and
`Grouping::convert_3Dpoint_time_ch` (which produced the cells' ticks from the raw point
in `form_point_association`, `:2729-2735`) do not share a time origin on PDVD. t0
arithmetic cannot know that. Recorded here because the next person will try it first.

### 4.2 The shipped knob: measure the offset with the cells' own conversions

`form_map_graph`, knob on, for the fit point `p` about to be associated: run `p` through
exactly the two conversions the cells came from — `transform->backward(p, cluster_t0,
face, apa)` then `convert_3Dpoint_time_ch(...)` — then back through the geometric
`(tind - offset_t)/slope_t` that `update_association` applies to a cell, and store
`x_geo(p) - p.x` per (apa, face) in `m_excl_x_shift`. `update_association` subtracts it from
`raw_x` before building the test point (guarded by `!m_excl_x_shift.empty()`, so the legacy
path executes the same instructions on the same doubles when the knob is 0; the map is
cleared after each point). A cell is thereby placed at `p.x + (tick_cell - tick_p) /
slope_t` — relative to the point, exact up to the slice quantisation the cells already
carry — whatever the t0 or time-origin conventions of the detector. Plumbing: `Parameters::excl_t0_frame`
(`TrackFitting.h`), `set_parameter`/`get_parameter`, `TaggerCheckNeutrino` config key
`excl_t0_frame` (default false; `set_parameter` is called only when true, so a value in the
runtime TrackFitting JSON survives — the SBND diagnostic arm needs no cfg edit), PDVD TLA
in `pdvd/wct-pr-perevt.jsonnet` with the key-suppression idiom. SBND cfg untouched.
Doctests: `doctest_update_association.cxx` "pdvd45 …" (mixed frame drops the cell, shifted
frame keeps it, a shift for another volume does nothing, zero shift = legacy) and the
knob-default case in `doctest_clus_knob_defaults.cxx`.

### 4.3 Gates (knob OFF, byte-identical; labels in `stm/gates/d45_excl_frame_gate.txt`)

| gate | arms | result |
|---|---|---|
| 1 PDVD `-nu -stm-fit`, 2 events | `d45nu0` (ref) vs `d45g1n2` (new2) | PASS 2/2 (member hashes + calib dump); `tracking-pr.root` identical both events |
| 2 PDVD `-stm -stm-fit`, 2 events | `d44beestm` (ref) vs `d45g2n2` (new2) | PASS 2/2; `tracking-stm.root` identical both events |
| 3 SBND bare production, 3 events | `d45g3ref` vs `d45g3new` / `d45g3n2` | SAME 3/3 (mabc members, pctree members, nusel tsv); plus 6 exclusion-active nueCC48 events ref vs new2: SAME 6/6 on mabc, pctree, nusel, tracking-pr (see 5.3) |
| 4 compiled-config proof | `.wct-pr_d45nu0.json` 0 keys, `.wct-pr_d45on2.json` 1 key | pass |
| doctests | `wcdoctest-clus` | 310 cases passed, 1 skipped, SUCCESS (2 new cases) |


## 5. Grade

### 5.1 Evt 298595, knob ON (`d45on2`, pin new2, exclusion dump on)

| cluster | segments | coverage before -> after | median dist to stm_fit [cm] | own distance median [cm] before -> after |
|---|---|---|---|---|
| 39 | 3 -> 4 | 0.30 -> **0.92** | 0.21 | 61.8 -> 0.89 |
| 55 | 4 -> 3 | 0.81 -> **1.00** | 0.20 | 352.5 -> 0.84 |
| 79 | 1 | 0.89 -> 0.89 | 0.30 | 235.4 -> 0.77 |
| 83 | 4 -> 2 | 0.10 -> **0.94** | 0.18 | 520.4 -> 0.97 |
| 86 | 10 -> 3 | 0.22 -> **0.98** | 0.20 | 583.7 -> 0.85 |
| 103 | 1 -> 2 | 0.90 -> 0.92 | 0.27 | 475.5 -> 0.74 |
| 109 | 6 -> 4 | 0.74 -> **0.98** | 0.19 | 205.5 -> 0.90 |
| 113 | 5 -> 4 | 0.11 -> 0.29 | 1.85 | 423.7 -> 0.86 |

Drop counter **5142/8458 (60.8 %) -> 16/1176 (1.4 %)**, pre-registered < 5 %, SBND's floor
is 1 %. The dump after the fix: own-distance medians 0.74-0.97 cm on every cluster
(pitch 0.765 cm), **13.4 % of cells inside the 0.3 cm floor** (was 0), 90 % kept (was 25 %).
The restored trajectories sit 0.18-0.30 cm from the STM fit of the same charge.
Adding `dqdx_fit_keep_all_points` on top (`d45onkeep`, 2 events) changes one point on
one cluster (109: 265 -> 266) — with the frame right, the third pass's drop has nothing
left to remove (54/5848 = 0.9 % on 039349/23, all at genuine junction ties).

### 5.2 120 events (`d45nu0` -> `d45on`, 594 STM-tagged clusters)

| | baseline | knob ON |
|---|---|---|
| coverage quantiles 10 / 25 / 50 / 75 / 90 | 0.06 / 0.17 / **0.44** / 0.87 / 1.00 | 0.59 / 0.93 / **0.98** / 1.00 / 1.00 |
| mean coverage | 0.50 | 0.88 |
| clusters with coverage < 0.5 / < 0.85 | 54 % / 73 % | 8.6 % / 17 % |
| multi-segment clusters with one winner | 271 of 483 | **2 of 460** |
| median distance stm_fit -> track_fit | 1.27 cm | 0.21 cm |
| ghost fraction (track_fit > 2 cm from charge), median / p90 | 0.00 / 0.09 | 0.00 / 0.09 |
| pre-dQ/dx drop, all passes | 1 468 898 / 2 884 704 = 50.9 % | 100 714 / 1 686 254 = **6.0 %** |
| segments per STM cluster, mean | 5.18 | 4.05 |
| wall, 120 events | 4332 s | 4508 s (+4.1 %) |

Per cluster: coverage improved by more than 0.1 on **395** clusters, worsened by more than
0.1 on **3**. By baseline segment count the median coverage goes 1.00/0.75/0.38/0.39/0.31
(1/2/3/4-5/>=6 segments) -> 1.00/0.99/0.99/0.98/0.97: the dependence on multiplicity is
gone. The 51 clusters still below 0.5 with the knob on are 25 STM clusters that the PR
does not fit at all in either arm (no `track_fit` points; unchanged 25 -> 25) and 26 with
a fit — cluster 113-type topology disagreements. Among clusters the PR does fit, the
median coverage is 0.98 and 26 of 569 are below 0.5. The segment count per cluster falls
from 5.2 to 4.1: with sane charge association the graph passes stop splitting and
re-routing legs that had lost their charge.

Fewer trajectory points reach the pre-dQ/dx pass with the knob on (1.69 M vs 2.88 M) because
the graph converges in fewer refits — `pr_arm_census_diff.py` shows TGM 2549, STM 1782,
STM evaluations 1809 and neutrino candidates 5042 all **unchanged** (the taggers run
before the PR), wall +4.1 %, peak RSS +0.5 %.

### 5.3 SBND (shared component; SBND cfg untouched)

Knob OFF, pin ref vs new2 on the 6 exclusion-active nueCC48 events (10550 46363 81597
360535 256587 433451, source `work-nuecc48-d97fv`): `mabc-pr.zip`, `pctree-pr`, `nusel-*.tsv`,
`tracking-pr.root` all SAME 6/6 (a first comparison made while one arm was still writing
event 256587 said DIFF — the arm was re-run twice on each pin and is deterministic:
`feedback_gate_against_a_running_arm`, again). Knob ON (via a COPY of the runtime JSON
with `excl_t0_frame: 1` passed as `SBND_TRACKFIT_JSON`): selection rows (`nusel`) unchanged
6/6; fits move on 6/6; `kine_reco_Enu` 850 -> 850, 3709 -> 3671, 2091 -> 2219, 2618 -> 2540,
1728 -> 1572, 1479 -> 1473 MeV; PR-fitted clusters 316 -> 317, `track_fit` points 5770 ->
6181, one-winner clusters 2 -> 2, clusters with > 5 % ghost points 48 -> 48. The dump on
46363 (t0 0.836 us, v*t0 = 1.3 mm): own-distance quantiles 0.19/0.32/0.55/0.87/1.23 cm ->
0.19/0.33/0.55/0.86/1.20; floored 23 % -> 21 %. **On SBND this is a millimetre-scale
perturbation of a tournament that already arbitrates real proximity**, exactly as the
owner's question anticipated; a flip there is a separate, small decision.

### 5.4 Downstream on PDVD — the part the owner must weigh

The PR chain's downstream products move a lot, because the fits they are built on were
wrong by metres of charge attribution (`d45_downstream.py d45nu0 d45on`, 560 candidates
present in both arms, 9 lost / 9 gained):

- main vertex moved > 1 cm on 406 candidates, > 5 cm on **321**; median shift 17 cm,
  p90 182 cm, max 582 cm.
- `kine_reco_Enu` (482 finite pairs): median change **-132 MeV**, p10/p90 -695/+122 MeV;
  |dE| > 200 MeV on 210 candidates. On 298595 all five movers go down by 100-160 MeV.
- **`kine_reco_Enu` is NaN on 72 candidates with the knob on, 10 without.** Source:
  `kine_long_muon … dqdx=-nan` — `cal_kine_dQdx` (`PRSegmentFunctions.cxx:2533`) sums
  `recomb_model->dE(dQ, dx)` over every fit of the muon chain with no guard on `dx = 0`;
  a coincident pair of fit points gives 0/0 and the NaN survives the sum. It is a latent
  kinematics defect (11 candidates already hit it in production) that the restored
  trajectories reach seven times more often. **Owed: a `dx <= 0` guard in `cal_kine_dQdx`
  (default-OFF knob, SBND gate), before any flip.**
- Wall +4.1 % (the per-point shift costs eight `backward` + `convert_3Dpoint_time_ch`
  calls per interior fit point per pass).

None of these is a regression of the fit — the trajectories are demonstrably right where
they were wrong before — but every PDVD PR-chain number downstream of the fit changes, and
nothing downstream has been hand-graded on PDVD. That is the trade the flip decision is
about.

### 5.5 Bee

Same event, same pctree, same pin, event 0 = today's production, event 1 = knob ON:
<https://www.phy.bnl.gov/twister/bee/set/0b40b1d3-462e-4a9e-b037-d4fe6643653d/event/list/>
(compare the `track_fit` layer against `stm_fit` on clusters 86, 83, 39, 113). The
production set the owner looked at: `971dc70d-98ca-4f30-88e7-76078ccf64dc`.

## 6. Recommendation

> **Round 2 (2026-09-05):** items 1 and 4 are executed (sec 8-12); item 2 is measured and
> the answer is "not yet" (sec 11).

1. **Flip `excl_t0_frame` on for PDVD production** — *done, sec 9.* The case for:
   the exclusion fit is currently arbitrating charge with a metres-wrong test point on every
   PDVD cosmic candidate, the fix is prototype parity, and it takes coverage of the STM fit
   from 0.44 to 0.98 with 3 of 594 clusters worse. The case for waiting: every downstream
   PR number moves (5.4), and the NaN-Enu count goes 10 -> 72 until `cal_kine_dQdx` is
   guarded. Recommended order: guard first (one small round), then flip, then a hand-scan of
   the moved vertices.
2. **SBND**: measured, small, no flip proposed here. If the owner wants prototype parity
   everywhere, flip both together after the SBND nueCC48 + mcp1k arm pair (pr/98-style).
3. **Do not flip `dqdx_fit_keep_all_points` for this**: with the frame right it changes one
   point in two events.
4. **The 25 STM clusters the PR never fits** and the **113-type topology disagreements**
   (26 clusters below 0.5 with a fit) are the next `track_fit` questions; they are not
   this mechanism.
5. Doc 30's recommendation 1 (fix the duplicate segment) is withdrawn.

## 7. Files

- toolkit: `clus/inc/WireCellClus/TrackFitting.h` (`excl_t0_frame`, `m_excl_x_shift`),
  `clus/src/TrackFitting.cxx` (set/get, `form_map_graph` per-point shift,
  `update_association` three loops, `organize_segments_path_3rd` comment),
  `clus/inc/WireCellClus/TaggerCheckNeutrino.h` + `clus/src/TaggerCheckNeutrino.cxx`
  (config key, echo, conditional `set_parameter`), `clus/test/doctest_update_association.cxx`
  (harness `set_excl_x_shift` + the pdvd45 case), `clus/test/doctest_clus_knob_defaults.cxx`,
  `clus/docs/porting/porting_dictionary.md`.
- wcp-porting-img: this doc; `30_stm-vs-trackfit-discrepancy-evt298595.md` (correction
  blockquotes); `pdvd/wct-pr-perevt.jsonnet` (TLA, default false);
  `scripts/d45_trackfit_vs_stmfit.py`, `scripts/d45_excl_frame.py`, `scripts/d45_downstream.py`,
  `scripts/run_d45_arms.sh`; `stm/gates/d45_excl_frame_gate.txt`.
- Arms (fresh tags, kept): PDVD `d45nu0` (120), `d45on` (120), `d45dump`, `d45on1`
  (superseded first cut), `d45on2`, `d45exoff`, `d45keep`, `d45both`, `d45onkeep`,
  `d45g1new`/`d45g2new`/`d45g1n2`/`d45g2n2`; SBND `work-stmcamp-d45g3ref/new`,
  `work-d45sbnd-{ref,off,off2,on,on2,on2dump,off2dump,off2b,refb}-nuecc48`,
  `work-stmcamp-d45sbnddump`. Dumps and censuses under `/home/xqian/tmp/d45/`.
- Round 2, toolkit: `clus/inc/WireCellClus/PRSegmentFunctions.h` + `clus/src/PRSegmentFunctions.cxx`
  (`cal_kine_dQdx(..., skip_zero_dx)`), `clus/inc/WireCellClus/NeutrinoPatternBase.h`
  (`KineChargeOptions::dqdx_skip_zero_dx`), `clus/inc/WireCellClus/PRShower.h` + `clus/src/PRShower.cxx`,
  `clus/src/NeutrinoEnergyReco.cxx`, `clus/src/NeutrinoShowerClustering.cxx` (24 callers),
  `clus/inc/WireCellClus/TaggerCheckNeutrino.h` + `clus/src/TaggerCheckNeutrino.cxx` (key
  `kine_dqdx_skip_zero_dx`), `clus/test/doctest_cal_kine_dqdx_zero_dx.cxx`,
  `clus/test/doctest_clus_knob_defaults.cxx`, `clus/docs/porting/porting_dictionary.md`.
- Round 2, wcp-porting-img: `pdvd/wct-pr-perevt.jsonnet` (TLA `kine_dqdx_skip_zero_dx`; BOTH
  knobs flipped to true), `scripts/d45_vertex_scan.py`, `stm/gates/d45_kine_guard_gate.txt`,
  `figs/45_vertex_scan/{movers,key,labels,unblinded}.tsv` + six example sheets,
  `figs/45_cl113_prod.png`, `figs/45_d45prod_census.tsv`. Arms: PDVD `d45g1n3`, `d45g2n3`,
  `d45on3`, **`d45prod`** (the post-flip production arm, 120 events); SBND
  `work-stmcamp-d45g3n3`, `work-d45sbnd-off3-nuecc48`, `work-d45sbnd-{off3all,on3all}-{nuecc48,ncpi0}`.
  Pin `new3` = `/home/xqian/tmp/d45_libpin/new3` (libWireCellClus `d604cb27…`).

## 8. Round 2a — the `cal_kine_dQdx` guard (`kine_dqdx_skip_zero_dx`, default OFF)

**Symptom.** `kine_reco_Enu` is NaN on 11 of 569 PDVD candidates in production and on 73 with
`excl_t0_frame` on (sec 5.4). The log line is `kine_long_muon … dqdx=-nan`.

**Root cause.** The vector form `cal_kine_dQdx(vec_dQ, vec_dx, model)` (`PRSegmentFunctions.cxx`),
used by `PRShower::calculate_kinematics` (multi-segment showers) and
`calculate_kinematics_long_muon`, sums `recomb_model->dE(dQ, dx)` over every fit point with no
guard on `dx`. `PracticalBoxRecombination::dE` computes `dQ/dX`; a coincident fit pair (`dx == 0`)
gives `0/0` (or `exp(inf) * 0`) and the NaN passes both clamps (`dE < 0`, `dE > 50 MeV/cm * dx`
are false for NaN) into `kenergy_dQdx`. The prototype (`ProtoSegment.cxx:1316`) divides by
`dx + 1e-9` and multiplies by `dx`, so such a point contributes exactly 0 — the port dropped the
epsilon (porting-dictionary entry added). The segment form `segment_cal_kine_dQdx` already skips
`dx <= 0`; only the vector form lacked it. A negative `dx` is worse than NaN in the legacy code:
the `50 MeV/cm * dx` clamp flips sign and the point *subtracts* energy.

**Why it hid.** Two ways. (i) For a long muon in mode 2 the NaN reaches `kine_reco_Enu` and is
visible — 11 candidates, nobody looked. (ii) For a hadronic shower under K3
(`kine_hadronic_dqdx`, ON in the PDVD operating point) `apply_hadronic_dqdx_best` tests
`dqdx > 0`, which a NaN fails, so the shower silently kept `kine_best = 0` and Enu stayed
finite but low.

**Fix.** `cal_kine_dQdx(vec_dQ, vec_dx, model, bool skip_zero_dx = false)`:
`if (skip_zero_dx && dx <= 0) continue;`. Threaded as `KineChargeOptions::dqdx_skip_zero_dx`
through `PRShower::calculate_kinematics{,_long_muon}` (new trailing defaulted parameters, all
24 callers pass it) from the `TaggerCheckNeutrino` key `kine_dqdx_skip_zero_dx`. Fail-first
doctest `doctest_cal_kine_dqdx_zero_dx.cxx`: the legacy call returned `-nan` on
`{50e3, 0, 50e3} / {1 cm, 0, 1 cm}` (recorded in the gate record); after the fix it pins the
legacy NaN and checks the guarded value equals the two-point sum and that knob-on with no
degenerate point is bit-identical to knob-off.

**Verification (gates PASS, `stm/gates/d45_kine_guard_gate.txt`).** Pins `new2` (the doc-45
knob) vs `new3` (the guard), knob OFF: PDVD `-nu` 2/2 (+ `tracking-pr/stm.root` trees SAME),
PDVD `-stm` 2/2, SBND bare production 3/3, SBND exclusion-active 6/6 (mabc, pctree, nusel,
tracking-pr). Compiled-config proof: knob-off JSON == HEAD's byte for byte; each key once when
on; the `-stm` chain's compiled JSON is identical with both knobs on (it cannot see them).
`wcdoctest-clus` 312 pass.

**Knob on (120 events, `d45on3` vs `d45on`).** NaN rows 73 -> 0. Vertices identical on all 569
candidates. 73 candidates NaN -> finite (recovered Enu median 673 MeV, p10/p90 404/1044). Six
candidates with finite Enu move, all *up* by 19–180 MeV — each is a K3 hadronic shower whose
`kine_best` was silently 0 (calib dumps: `kine_best 0.0 -> 45.5, 78.5, 18.9, 30.1, 129.6, 179.8`).
`T_rec_charge` SAME 120/120; `T_tagger` differs on 14 events in kinematics-derived features
only (`lem_e_dQdx`, `ssm_*angle*`, `*_kine_energy_best`); on one event (039349_37 cluster 39)
a shower energy 0 -> 474 MeV re-evaluates the nue sub-flags gap/pio/br1/br3/tro;
`cosmic_flag` and `numu_cc_flag` unchanged on every row (the BDT scores are not run on PDVD).

## 9. Round 2b — the PDVD production flip

`pdvd/wct-pr-perevt.jsonnet`: `excl_t0_frame = true`, `kine_dqdx_skip_zero_dx = true`
(owner decision 2026-09-05). `d45prod` = the canonical production chain (no `-S`), pin `new3`,
120 events: `d40r3_hash_gate.py d45on3 d45prod` **PASS 120/120** (mabc-pr.zip members + calib
dump), `tracking-pr.root` SAME 120/120, NaN Enu 0/569. Production reproduces the graded arm
exactly; it is NOT bit-identical to the pre-flip chain by design.
`-S excl_t0_frame=false -S kine_dqdx_skip_zero_dx=false` restores the old chain. The cosmic-only
`-stm` production chain is unchanged (sec 8 compiled-config proof).

Production census, `d45nu0` -> `d45prod` (569 PR-fitted STM clusters):

| | pre-flip | post-flip |
|---|---|---|
| coverage of `stm_fit` by `track_fit`, median / mean | 0.48 / 0.52 | **0.98 / 0.92** |
| clusters with coverage < 0.5 | 52.2 % | 4.6 % |
| one-winner multi-segment clusters | 262 / 483 | 2 / 460 |
| segments per cluster | 5.4 | 4.2 |
| trajectory points dropped as zero-charge (all passes) | 64.3 % | 6.7 % |
| clusters better / worse by > 0.1 | | 395 / 3 |

One arm-to-arm wobble surfaced: `tracking-stm.root` of 039253_16 differs between *every* pair of
arms of this event (`d45nu0`, `d45on`, `d45on3`, `d45prod` all distinct) in the STM dump's
`T_rec_charge.pt` branch on 61 of 6877 rows by ≤ 1.9e-309 — denormal garbage from an
uninitialised read in the `stm_magnify` filler. Untouched by both knobs, invisible in every
production product; owed to the owner as a separate one-liner.

## 10. Round 2c — the moved vertices, measured and hand-scanned blind

321 candidates moved their main vertex by more than 5 cm (`d45_vertex_scan.py d45nu0 d45on3`;
the vertices of `d45on3` and `d45prod` are identical). Proxies over all 321:

| | production | knob ON |
|---|---|---|
| distance to own cluster charge, median | 0.49 cm | 0.49 cm |
| distance to the nearest END of the STM trajectory, median | 25.6 cm | **1.6 cm** |
| vertex within 3 cm of an STM end | 14 % | **59 %** |

(at an STM end: production only 26, knob only 171, both 18, neither 106.)

Blind sample of 40 (seed 45, marker A/B randomised per sheet, `key.tsv` opened only after
labelling; labels END = at a track end, KINK = at a kink/branch, MID = mid-track on charge,
OFF = off the charge; `figs/45_vertex_scan/`):

| | END | KINK | MID | OFF | UNCLEAR |
|---|---|---|---|---|---|
| production | 14 | 2 | 17 | 4 | 3 |
| knob ON | **22** | 1 | 13 | **1** | 3 |

Transitions production -> ON: END->END 10, MID->MID 9, **MID->END 7, OFF->END 3, KINK->END 2,
OFF->MID 1** (13 improvements), END->MID 3, END->OFF 1 (4 regressions), MID->KINK 1 (neutral),
UNCLEAR->UNCLEAR 3 (10–30 cm blobs with no track topology). The hand labels agree with the
proxy: the vertex moves *to* a track end, which for a cosmic muon is where it should be. The
four regressions (sheets 03/20-type MID and one END->OFF) are the price; none of them is a
selection flip on PDVD.

## 11. Round 2d — SBND, measured, NOT flipped

Selection rows (`nusel-evt*.tsv`) are identical with the frame knob on: 48/48 nueCC48 and
19/19 NCpi0 (pin `new3`, `work-d45sbnd-{off3all,on3all}-*`). But the kinematics are not a
millimetre perturbation: vertex shift median 0.33 / 1.07 cm with 4 / 3 candidates moving
> 5 cm (max 44 / 78 cm); Enu change median +0.3 / +34.9 MeV, |dE| > 50 MeV on 23 of 48 and
15 of 20 candidates, > 200 MeV on 4 and 7. A 1–3 mm shift near the 0.3 cm floor re-arbitrates
boundary cells, the charge attribution moves, and everything downstream follows. The
`kine_dqdx_skip_zero_dx` knob is inert on SBND (0 NaN-Enu candidates in the 3067-event
production, `work-*-d97fvpr2`). **Recommendation:** an SBND flip needs its own round with truth
Enu on the MC sets, the pi0 66-set census and the doc 91 sentinel suite; nothing here is
evidence for or against it.

## 12. The two leftover `track_fit` questions

**The 25 STM clusters the PR never fits are by design, not a defect.** Reading the
`[nu_per_bundle]` census in the logs: 19 of 25 sit in a flash bundle whose PR candidate is a
*different, longer* STM-tagged cluster (one candidate per bundle; e.g. 039252_17 cluster 88
lost gid 121 to cluster 90, L 194.6 cm), and 6 are below the 15 cm `nu_per_bundle_min_length`
floor (4–13 STM points). Whether the second STM cluster of a bundle deserves its own PR fit is a
design question for the owner, not a bug in the chain.

**Cluster 113 of 298595 is an imaging degeneracy, not a fitter fault.** Its charge sits at
x = 336–338 cm, against the anode, in a 40 × 40 cm triangular *sheet* in y-z
(`figs/45_cl113_prod.png`): a track parallel to the anode plane arrives within a microsecond,
and the three planes' wire crossings tile a fan of ghost blobs. The STM's single path takes one
diagonal through the sheet; the PR's graph builds a V through it (three segments, coverage of
the STM path 0.29 even after the flip, 0 on two of the segments). Neither is "right"; the
information to decide is not in the point cloud. It is the same class as the anode-hugging
clusters of doc pdvd/35.
