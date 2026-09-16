# doc pdvd/111 — why the fitted STM trajectory leaves the image: the Steiner seed, rarely the fit, and a default-OFF seed re-centring knob

**Owner, 2026-09-16** (after doc 110): *"Now, our focus should be on the track trajectory. I understand this would change
tagger, but we can worry about them later. Can you investigate to see what is the reason for the deviated track
trajectory? 1. IS it coming from the seed track trajectory from the graph? (Steiner Graph, Steiner terminal related
issues)? 2. Is it coming from the track trajectory fitting itself (one play did not give good constraint or something
else)? … investigate and understand the situation and then design improvements and implement."*

**How "worry about the tagger later" is read here.** This round does not wait on a doc-56 hand scan. It does not change
a production default either: the lever is built default-OFF, proven byte-identical when off, and measured when on.
Tag changes are reported, not gated. A flip is the owner's call (sec 8).

## Status

- **Question 1, the seed: yes, almost always.** Every STM fit is traced stage by stage (new log-only trace, sec 3):
  - of the fitted rows more than 1 cm from the image ridge, **90 % (PDHD) / 88 % (PDVD) were already that far off in
    the seed at the same place** (the Steiner Dijkstra path, or its 1.2 cm fill; sec 4);
  - the seed defect is **the path, not the terminals and not the cloud**:
    - at the owner's h1 the terminals are the same 18 in every arm;
    - for seed-born deviations 1–5 cm from the ridge, the Steiner cloud has itself moved off the ridge in only 7 %
      of the rows;
    - the path runs a straight Steiner edge longer than 3 cm across the cloud in 40–43 % of them (rows on such edges
      are 9.4 % / 6.3 % of all rows);
    - a round-2 re-route through the tagger's crawl point accounts for another 12–14 %;
  - a second, separate population is **off-image bridges** (seed more than 5 cm from any image: 42 % of the PDHD
    deviated rows, 33 % PDVD). They are straight lines across regions with no charge, the doc-102 chord class, and
    are not addressed here.
- **Question 2, the fit: rarely the origin (8.5 % PDHD, 10.4 % PDVD), but it cannot undo a seed error.**
  - The fit solves each point on its own inside an association window of about one window per pass. Where the seed
    is 1.5–2 cm off, the final row is still more than 1 cm off in 83–85 % of cases (transfer table, sec 4.4).
  - For seed-born deviations the fit did not move the point toward the ridge at all in 59–64 % of rows. It moved it
    only part way in 20–21 %. It reached the ridge and the area smoothing or the skip step undid it in 8–10 %.
  - Where the fit itself leaves a good seed, a single plane pulls the solve off in 43–51 % of those rows on the
    first pass and 64–66 % on the second: "one plane gives a poor constraint" is real, but it is 5 % of all
    deviated rows (sec 4.3).
- **The owner's spots**, on the 2×2 of the two production levers (sampler × fit keys) with the same pctree (sec 1):
  - **h1:** the seed is about 1.9 cm off in all four arms; the dense `charge_stepped` Steiner cloud is what keeps the
    fit glued to it;
  - **h2:** the round-2 crawl re-routed the seed through the side branch (3.7 cm, 8 cm from the click), in production
    only;
  - **v1/v2:** the pre-flip PDVD seed was off; `charge_stepped` put it on the ridge; v2's pre-flip spike is
    fit-born.
- **The lever** is `seed_recenter_sigma` (TrackFitting JSON; C++ default 0 = off). It moves the seed
  transversely onto the local charge ridge before the first fit pass (sec 5–6).
  - **Knob off is byte-identical** on PDHD (61), PDVD (120), SBND (16) and uBooNE (35) (sec 6).
- **Knob on: no setting passes the pre-registered rule, on either detector** (sec 7).
  - **It does what it was built for:** rows more than 1 cm off the ridge fall 8–11 % (PDHD) / 5–6 % (PDVD), and
    off-charge rows fall 3–9 %.
  - **It fixes both PDHD owner spots at σ = 10 mm:** h1 max 1.77 → 0.54 cm, h2 2.55 → 0.93 cm.
  - **But it roughens the seed.** Each point is re-centred on its own, lands on its own local mode, and the seed
    gains transverse jitter at the 0.5 cm scale:
    - seed points with a 3-point wiggle above 0.5 cm go 11.6 → 39.3 % on 029107_16;
    - fitted rows with a chord wiggle above 1 cm go ×4–7;
    - the trajectory gets 3–6 % longer;
    - Bee holes rise 37–91 %, failing P3.
  - The fix is a redesign that moves the seed **as a curve**, not another σ (sec 8).
- **No default changed; nothing flipped.**

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; D=$IMG/pdvd/docs/nf_sp_img_clus; S=$D/scripts; F=$D/figs
T=/home/xqian/toolkit-dev/toolkit

# build (toolkit 06fd9e22), pins: libpin_d111 = the trace only (clus 03d1b506f61c);
# libpin_d111b = trace + knob (clus df541468ff1e; Root rebuilt because its visitors include TrackFitting.h)
(cd $T && ./wcb build --notests -p && ./wcb install --notests -p && ./wcb build --targets=wcdoctest-clus -p \
   && ./build/clus/wcdoctest-clus)                                     # 418 / 418

# sec 1 -- the owner's spots on the stored 2x2 arms (read-only) and on the 2x2 trace arms
python3 $S/d111_spot_frame.py --out $F/111_spot_frame                  # figs/111_spot_frame_{h1,h2,v1,v2,v3}.png, .tsv
C=/home/xqian/tmp/d111/cfg   # pre-flip fit JSONs: git show b9ce8b4d^:.../pdhd_track_fitting.json, 8fc6070e^:.../pdvd_track_fitting.json
TR="WCT_STM_PATH_DEBUG=1 WCT_TRAJ_ASSOC_DEBUG=1"; STEP="-S retile_sampler_strategy='stepped'"; P1=/home/xqian/tmp/d111/libpin_d111
ARM=d111hA0 DET=pdhd SRC=d108hflip JOBS=1 PIN=$P1 EVENTS=029107_16 TRACE_ENV="$TR WCT_TRAJ_ASSOC_CLUSTERS=108,106" \
  PR_TLA="-A trackfitting_config=$C/pdhd_tf_preflip.json $STEP" bash $S/d111_run_arms.sh
#   d111hK: PR_TLA="$STEP";  d111hS: PR_TLA="-A trackfitting_config=$C/pdhd_tf_preflip.json";  d111hA1: no TLA
#   d111v{A0,K,S,A1}: the same on DET=pdvd SRC=d103vflip EVENTS=039349_20, pdvd_tf_preflip.json, CLUSTERS=80,25
python3 $S/d111_spot_stages.py --out $F/111_stages                     # figs/111_stages_*.png, .tsv

# sec 3 -- the trace arms (production config) and their identity gates
ARM=d111htr DET=pdhd SRC=d108hflip JOBS=6 PIN=$P1 TRACE_ENV="$TR" bash $S/d111_run_arms.sh
ARM=d111vtr DET=pdvd SRC=d103vflip JOBS=6 PIN=$P1 TRACE_ENV="$TR" bash $S/d111_run_arms.sh
python3 $S/d111_identity_gate.py --det pdhd --a d108hflip --b d111htr \
    --ignore-members stm_tagged,stm,stm_fit,steiner_graph,steiner_terminals > $F/111_gate_trace_pdhd.txt
python3 $S/d111_identity_gate.py --det pdvd --a d103vflip --b d111vtr > $F/111_gate_trace_pdvd.txt
#   trace off, 5 + 5 events: d111hoff5 (EVENTS of doc 109) vs d109hstm, d111voff5 (doc 110 set) vs d103vflip
python3 $S/d111_identity_gate.py --det pdhd --a d109hstm --b d111hoff5 > $F/111_gate_traceoff5_pdhd.txt
python3 $S/d111_identity_gate.py --det pdvd --a d103vflip --b d111voff5 --allow-subset > $F/111_gate_traceoff5_pdvd.txt

# sec 4 -- the attribution census
python3 $S/d111_stage_attrib.py --det pdhd --arm d111htr --out $F/111_attrib_pdhd     # .txt + _rows.tsv.gz
python3 $S/d111_stage_attrib.py --det pdvd --arm d111vtr --out $F/111_attrib_pdvd

# sec 6-7 -- frozen rule (sha in figs/111_pred.sha256), arms, gates, evaluation
sha256sum -c $F/111_pred.sha256
bash $S/d111_phaseC_arms.sh              # d111hoff, d111hsr{6,10,15} / d111voff, d111vsr{6,10,15}; JOBS=2 each, concurrent;
                                          # knob JSONs figs/111_tf_{pdhd,pdvd}_sr{6,10,15}.json on libpin_d111b
python3 $S/d111_identity_gate.py --det pdhd --a d108hflip --b d111hoff \
    --ignore-members stm_tagged,stm,stm_fit,steiner_graph,steiner_terminals > $F/111_gate_off_pdhd.txt
python3 $S/d111_identity_gate.py --det pdhd --a d109hstm --b d111hoff --events 028084_3,028084_12,029107_5,029107_16,029107_18 > $F/111_gate_off_pdhd_scope5.txt
python3 $S/d111_identity_gate.py --det pdvd --a d103vflip --b d111voff > $F/111_gate_off_pdvd.txt
bash $S/d102_compile_pr.sh pdhd d111post; bash $S/d102_compile_pr.sh pdvd d111post       # a870511c9b22 / 211a49a48229
(cd $S && TAG=d111sold PIN=/home/xqian/tmp/d102/libpin_d102 JOBS=2 bash d101_sbnd_arm.sh; \
          TAG=d111snew PIN=/home/xqian/tmp/d111/libpin_d111b JOBS=2 bash d101_sbnd_arm.sh)
python3 $S/d111_sbnd_gate.py --old d111sold --new d111snew > $F/111_gate_off_sbnd.txt
(cd $IMG/qlport/scripts && LD_LIBRARY_PATH=/home/xqian/tmp/d102/libpin_d102 ./sweep_5384.sh d111ub_old 3 \
   && LD_LIBRARY_PATH=/home/xqian/tmp/d111/libpin_d111b ./sweep_5384.sh d111ub_new 3 \
   && ./ab_check.sh d111ub_new d111ub_old) > $F/111_gate_off_uboone.txt
for s in 6 10 15; do python3 $S/d111_eval.py --det pdhd --base d111hoff --arm d111hsr$s --out $F/111_eval_pdhd_sr$s.txt
                     python3 $S/d111_eval.py --det pdvd --base d111voff --arm d111vsr$s --out $F/111_eval_pdvd_sr$s.txt; done
python3 $S/d111_tag_churn.py --det pdhd --base d111hoff --arms d111hsr6,d111hsr10,d111hsr15 >  $F/111_tag_churn.txt
python3 $S/d111_tag_churn.py --det pdvd --base d111voff --arms d111vsr6,d111vsr10,d111vsr15 >> $F/111_tag_churn.txt
# sec 7 -- why it fails: a TRACED knob-on arm on the owner's PDHD event, the seed jitter, the h2 record check
ARM=d111hsr10tr DET=pdhd SRC=d108hflip JOBS=1 PIN=/home/xqian/tmp/d111/libpin_d111b EVENTS=029107_16 \
  TRACE_ENV="$TR WCT_TRAJ_ASSOC_CLUSTERS=108,106" PR_TLA="-A trackfitting_config=$F/111_tf_pdhd_sr10.json" bash $S/d111_run_arms.sh
python3 $S/d111_seed_jitter.py --det pdhd --arm d111hsr10tr --event 029107_16 > $F/111_seed_jitter.txt
python3 $S/d111_h2_record.py --arms d111hoff,d111hsr6,d111hsr10,d111hsr15 --traced d111hA0,d111hS,d111hA1,d111hsr10tr > $F/111_h2_record.txt
```

Arms (all new tags, M13). The pctrees are symlinks to the production arms' (PDHD `d51hclus` via `d108hflip`, PDVD
`p100flip` via `d103vflip`).

## 1. The owner's spots: the two levers of the flip, taken apart

Doc 110 placed the five spots. Here each is traced on the 2×2 of the two production trajectory levers on one pctree:

- **A0:** `stepped` retile, no fit keys (pre-flip);
- **K:** fit keys only (`fit_weight_pow` 1.5, `assoc_cont_center` 1);
- **S:** `charge_stepped` retile only;
- **A1:** both (production).

The trace arms reproduce the stored arms tree for tree:
- `d111hA0`/`hK`/`hS` equal `d101hnew`/`d101hkf`/`d102hocs` on every STM and PR branch. Their zips differ only in
  the layers doc 109 re-scoped.
- `d111vA0` equals `d103v0`, and `d111vA1` equals `d103vflip`, including the zip.

Ridge offsets are fit-independent. The reference is the charge-weighted axis of the cluster's own image points
within 3 cm (sec 4.1). Window ±12 cm of the click (±24 cm at h2). Columns are the offset p90 / max (cm) of the
**seed** (the round-2 Steiner path), the first-pass fit, and the final trajectory at the final rows' locations,
plus how many off-ridge seed points a solve put back on the ridge and how many of those the area smoothing reverted
(`figs/111_stages.tsv`, `figs/111_stages_<spot>.png`):

| spot | arm | seed p90 / max | fit1 p90 / max | final p90 / max | solve put back / reverted |
|---|---|---|---|---|---|
| h1 cl108 | A0 | 1.50 / 1.92 | 0.56 / 0.73 | **0.46 / 0.57** | 7 / 0 |
| | K | 1.63 / 1.91 | 0.55 / 0.80 | 0.37 / 0.55 | 7 / 0 |
| | S | 1.85 / 1.97 | 1.79 / 1.86 | **1.60 / 1.79** | 9 / 5 |
| | A1 | 1.63 / 1.97 | 1.63 / 1.86 | 1.54 / 1.77 | 6 / 5 |
| h2 cl106 | A0 | 1.24 / 1.72 | 0.51 / 0.78 | 0.40 / 0.66 | 9 / 0 |
| | K | 1.25 / 1.77 | 0.51 / 0.76 | 0.41 / 0.70 | 9 / 0 |
| | S | 0.57 / 0.69 | 0.48 / 0.60 | 0.45 / 0.62 | 0 / 0 |
| | A1 | 1.40 / **2.25** | 1.23 / 2.28 | **1.26 / 2.54** | 7 / 1 |
| v1 cl80 | A0 | 1.71 / 2.13 | 1.85 / 2.13 | 1.57 / 2.11 | 15 / 0 |
| | A1 | 0.57 / 0.72 | 0.39 / 0.53 | 0.35 / 0.53 | 1 / 0 |
| v2 cl80 | A0 | 1.22 / 1.81 | 1.41 / 2.43 | 1.43 / 2.45 | 6 / 1 |
| | K | 1.50 / 1.81 | 1.77 / 2.74 | 1.63 / **3.36** | 5 / 1 |
| | A1 | 0.82 / 1.05 | 0.57 / 1.12 | 0.60 / 1.27 | 0 / 0 |
| v3 cl25 | A0 | 1.26 / 1.55 | 0.83 / 1.30 | 0.77 / 1.64 | 12 / 0 |
| | A1 | 0.96 / 1.48 | 0.46 / 0.63 | 0.37 / 0.91 | 10 / 0 |

(v1–v3 S and K rows are in the TSV. PDVD S behaves like A1, K like A0.)

- **h1 — the seed is off in every arm; what changed is whether the fit could leave it.**
  - The image bends into a V (`figs/111_stages_h1.png`). The Steiner path cuts across it in all four arms, 1.9 cm
    off at the vertex.
  - On the sparse `stepped` cloud (A0, K) the first fit pass pulls every off-ridge point back and nothing reverts
    it: the final trajectory follows the V.
  - On the dense `charge_stepped` cloud (S, A1) the path runs through Steiner points at mid-height. The solve moves
    points toward the ridge, but the area smoothing reverts five of them to the seed position because their
    neighbours did not move with them, and the trajectory stays on the chord.
  - The terminals are the same 18 in all arms. The cloud's per-slab centroid offset from the image is 0.48 → 0.60 cm
    median, and its width relative to the image **narrows** (0.94 → 0.77; `figs/111_spot_frame.tsv`). So the cloud
    did not move off the track; the path through it did.
  - Why a denser cloud holds the fit: part of the association window is centred on the **nearest Steiner point**
    (TrackFitting.cxx `form_point_association`, the `steiner_graph` half). On a dense cloud the nearest Steiner
    point to a seed point is the seed vertex itself, so the window stays on the seed.
- **h2 — a crawl re-route, production only.**
  - In A1 the round-2 seed leaves the round-1 Steiner path by up to 3.69 cm, 7.9 cm from the click, into the side
    branch.
  - `adjust_rough_path` found a break in the round-1 fit, crawled from it, and re-walked Dijkstra through the crawl
    end point (TaggerCheckSTM.cxx:1437-1611).
  - In S the crawl re-routed 233 cm away and h2 is clean. The round-1 fit differs between S and A1 only by the fit
    keys, so the keys moved the break, and the break moved the seed.
  - The fit followed the detour (solve put back 7, reverted 1, 3 skipped).
- **v1, v2 — pre-flip PDVD seeds off the ridge; `charge_stepped` fixed the seed.**
  - v2's 2.4–3.4 cm spike is fit-born (A0/K fit1 max 2.43/2.74 against a 1.81 seed max), the doc-101 lattice
    mechanism.
  - Production v2 still reaches 1.27 cm.
- **v3:** the seed is 1.5 cm off in both arms and the fit recovers it in both.

## 2. The chain behind every persisted row

(toolkit `apply-pointcloud` at `9d8fd892`, plus this doc's trace)

1. **Seed.** `do_rough_path` runs Dijkstra on `steiner_graph` between the two boundary points
   (TaggerCheckSTM.cxx:1162-1269).
   - The edge weight is length × [0.8, 1.2], priced by charge at the two endpoints only (SteinerGrapher.cxx:1382-1396),
     so the path is close to the geometric shortest path through the cloud.
2. **Round 1.** `do_single_tracking(segment, false)` on that path (:3761).
3. **Round 2 is a refit from the Steiner graph, not from the round-1 fit.**
   - `adjust_rough_path` returns an **empty** path unless it finds a break in the round-1 fit (:1612-1616). The
     caller then refits from the **same** Dijkstra path (:3773).
   - When it does find a break, it crawls and re-walks Dijkstra first→crawl-end→last (:1566-1611).
   - Round 2 (:3780) is the persisted pass. The census finds the round-2 seed re-routed in 823 of 1158 PDHD records
     and 1186 of 1847 PDVD records.
4. **Inside `do_single_tracking`** (TrackFitting.cxx):
   - `organize_orig_path` fills the seed at 1.2 cm (straight lines across long edges);
   - `form_map` + `trajectory_fit` run pass 1 (window radius `min(step×0.9, 1.2 cm)`, about 1.1 cm);
   - `organize_ps_path` resamples to 0.6 cm, then pass 2 runs (window about 0.54 cm);
   - the final fill runs (charge-tested since doc pdhd/11).
5. **`trajectory_fit` solves each point on its own.**
   - The point becomes the charge-weighted centroid of its associated cells in each plane.
   - Nothing couples neighbours except two post-solve steps: `skip_trajectory_point`, and the area smoothing, which
     replaces a fitted point by its **input** position when it makes a large triangle with its fitted neighbours.
   - So the fit's reach from its seed is about one window per pass, and an isolated recovery is undone.
6. **The association has two halves** (`form_point_association`):
   - cells around the point's own projection, from blob-graph neighbours;
   - cells around the projection of the **nearest Steiner point**, from `steiner_graph` neighbours.
   The Steiner cloud therefore anchors the fit as well as the seed.

## 3. The trace

Two log-only traces, both printed to stdout and captured per event by `d111_run_arms.sh`:

- **`WCT_STM_PATH_DEBUG`** (doc pdhd/11, unchanged): every stage's points (`seed, org1, map1, fit1, org2, map2, fit2,
  org3, org3f, final`) per `<cluster>/<fwd|bwd>/<r1|r2>` pass.
- **`WCT_TRAJ_ASSOC_DEBUG`** (new, `TrackFitting::trajectory_fit`): one `TRAJASSOC` line per trajectory point. It
  carries:
  - the per-plane association (cell count, charge, the least-squares-weighted wire/time centroid, the projection of
    the solved point, `quantity`);
  - the input and solved positions;
  - **three leave-one-plane-out solves** of the same normal equations;
  - whether the point was skipped or reverted afterwards.
  `WCT_TRAJ_ASSOC_CLUSTERS=<ids>` adds cell-level `TRAJCELL` lines. Every value is computed on the side from matrices
  the fit already built.

**The traces change nothing** (gates by arm label):

| gate | result |
|---|---|
| trace ON, PDHD `d111htr` (61) vs `d108hflip` | **61 / 61 identical**, STM + PR trees and zip (the doc-109 re-scoped layers excluded, `figs/111_gate_trace_pdhd.txt`) |
| trace ON, PDVD `d111vtr` (120) vs `d103vflip` | **120 / 120 identical**, trees and zip (`figs/111_gate_trace_pdvd.txt`) |
| trace OFF (env unset), PDHD `d111hoff5` vs `d109hstm` (5, same scope) | **5 / 5 identical**, including every zip member (`figs/111_gate_traceoff5_pdhd.txt`) |
| trace OFF, PDVD `d111voff5` vs `d103vflip` (5) | **5 / 5 identical** (`figs/111_gate_traceoff5_pdvd.txt`) |
| replay: TRAJASSOC kept positions == the `fit1`/`fit2` stage | PDHD 9,674 calls, PDVD 14,295 calls, max \|Δ\| 0.0005 cm (print precision) |
| `wcdoctest-clus` | 413 / 413 (trace build) |

The replay line matters: it proves the association record describes the fit that ran, not a re-reading of the code.

## 4. Where the deviation is made: the census

`d111_stage_attrib.py` over every persisted STM-fit record of the production-config trace arms:
- PDHD: 61 events, 1158 records, 385,424 rows;
- PDVD: 120 events, 1847 records, 561,346 rows.

Each record is matched to its round-2 trace block by its final positions (all matched).

### 4.1 The metric, and the check that it measures a real departure

- **Ridge offset.** The cluster's image (Bee `clustering-global` points, charge-weighted) is voxelised at 1 cm. Each
  voxel carries the charge-weighted centroid and principal axis of the image within 3 cm. A point's offset is its
  distance from the axis of the nearest voxel. Nothing comes from the fit.
- **Deviated** = more than 1 cm.
- **Checked against the independent 2-D test:**

  | | PDHD | PDVD |
  |---|---|---|
  | rows deviated (> 1 cm) / > 2 cm | 7.8 % / 4.8 % (222 m of 2,502 m) | 5.1 % / 2.9 % (214 m of 3,658 m) |
  | off-charge (±1 wire, ≥ 1 plane) among deviated rows / among the rest | **70.6 % / 2.8 %** | **67.8 % / 7.5 %** |
  | Bee-dropped (q < 0) among deviated / the rest | 24.0 % / 3.7 % | 38.7 % / 2.1 % |
  | share of all q < 0 rows that are deviated | 35.2 % | 49.5 % |

  A deviated row is 9–25× more likely to be off the measured cells.
- **Tie to doc 110:** a third to a half of the Bee holes sit on these rows.

### 4.2 Origin: the earliest stage of the off-run that ends at the final row

| origin | PDHD rows | share | PDVD rows | share |
|---|---|---|---|---|
| **seed** (Steiner path) | 26,462 | **87.9 %** | 24,204 | **84.4 %** |
| org1 (1.2 cm fill of the seed) | 724 | 2.4 % | 1,096 | 3.8 % |
| fit1 | 1,371 | 4.6 % | 1,600 | 5.6 % |
| org2 (0.6 cm resample) | 169 | 0.6 % | 189 | 0.7 % |
| fit2 | 1,173 | 3.9 % | 1,374 | 4.8 % |
| org3 (final fill) | 186 | 0.6 % | 196 | 0.7 % |

The split is the same in every angle-to-drift bin (83–89 % seed). What changes with angle is how many rows deviate:
- PDHD: 21 % at 0–30°, 4–5 % at 50–75°, 20 % at 85–90°;
- PDVD: 4 % at 0–30°, 20–25 % above 75°.

### 4.3 What kind of seed error, and what the fit did with it

Split by how far the seed was:

| | PDHD | PDVD |
|---|---|---|
| **far:** seed > 5 cm from the image (a straight bridge across a region with no charge) | 12,627 (42 %), 205 records | 9,385 (33 %) |
| **near:** seed 1–5 cm off (the owner's kind) | 14,456 (48 %), 807 records | 15,640 (55 %) |
| seed within 1 cm, deviation made later (fit or fill) | 3,007 (10 %) | 3,639 (13 %) |

Near seed-born rows, by sub-class:

| sub-class | PDHD | PDVD |
|---|---|---|
| `long_edge`: the seed segment there is > 3 cm, a straight Steiner edge through or across the cloud | 43 % | 40 % |
| `cloud_na`: the cluster's Steiner cloud is not in the scoped Bee layer (untagged), so it cannot be classified | 33 % | 35 % |
| `crawl_reroute`: the round-2 seed left the round-1 path there (h2's kind) | 12 % | 14 % |
| `cloud_off`: no Steiner point within 0.5 cm of the ridge nearby, i.e. the cloud itself is displaced | 7 % | 7 % |
| `cloud_has_ridge`: on-ridge Steiner points are there, and the path walked past them | 5 % | 4 % |

- Across all rows, not only deviated ones, rows on a seed edge longer than 3 cm are 9.4 % (PDHD) / 6.3 % (PDVD). Among
  all seed-born deviations, near and far, they are 64.7 % / 59.3 %.
- **What the fit did at seed-born deviations** (all distances; from the TRAJASSOC record of each pass at that place):

  | | PDHD | PDVD |
  |---|---|---|
  | no pass moved the point ≥ 0.3 cm toward the ridge | 64 % | 59 % |
  | moved part way | 20 % | 21 % |
  | a solve put it within 1 cm of the ridge; area smoothing or skip undid it | 8 % | 10 % |
  | a solve put it back and kept it; a later stage re-made the offset | 7 % | 10 % |

- **Fit-born deviations** (fit1 + fit2), from the three leave-one-plane-out solves of the TRAJASSOC record:

  | | PDHD fit1 | PDHD fit2 | PDVD fit1 | PDVD fit2 |
  |---|---|---|---|---|
  | dropping ONE plane puts the solve back within 1 cm | 50.6 % | 66.1 % | 42.6 % | 63.8 % |
  | no single-plane drop does (every plane pulls) | 21.4 % | 23.2 % | 21.9 % | 23.5 % |
  | the solve was on the ridge; a later post-solve step moved it | 20.1 % | 5.1 % | 27.7 % | 6.3 % |
  | area-smoothing revert / skip | 4.6 / 3.3 % | 2.6 / 3.1 % | 4.4 / 3.3 % | 1.9 / 4.5 % |

  - When a single plane is the one, it is W as often as U or V combined on PDHD pass 2 (103 against 59 + 58 rows).
    Often more than one plane's removal alone would restore it (`one_plane_any`).
  - "One plane gives a poor constraint" is real. It accounts for 4.9 % (PDHD) / 5.5 % (PDVD) of all deviated rows.

### 4.4 Transfer: how far the fit can pull a displaced seed

| seed offset at the location (cm) | PDHD rows | final offset p50 | final > 1 cm | PDVD final > 1 cm |
|---|---|---|---|---|
| 0–0.5 | 290,069 | 0.18 | 0.2 % | 0.2 % |
| 0.5–1 | 60,799 | 0.46 | 3.7 % | 2.9 % |
| 1–1.5 | 12,958 | 0.99 | 49 % | 48 % |
| 1.5–2 | 3,881 | 1.59 | 83 % | 85 % |
| 2–3 | 2,128 | 2.30 | 94 % | 93 % |
| 3–5 | 2,954 | 3.87 | 99 % | 98 % |

- The final offset tracks the seed offset almost one to one above 1 cm.
- The fit polishes a seed that is already on the charge. It does not bring back one that is not.

## 5. Design: what to change, by the rule written before any lever was built

**The rule (plan, 2026-09-16).** Build a seed-side lever if seed-born deviations are ≥ 30 % on either detector, and a
fit-side lever under the same test. At most one per side.

- **Seed: 90 % (PDHD), 88 % (PDVD) → built.**
- **Fit: 8.5 % / 10.4 % → not built.** The fit-side findings (single-plane pulls, the area-smoothing revert, the
  Steiner-anchored window) are recorded in sec 4 and sec 9.

**Which seed lever.**
- The near class is dominated by a path that runs straight or along an edge **through a cloud that is itself on the
  track**: `long_edge` + `cloud_has_ridge` 45–48 %, against `cloud_off` 7 %. So the lever acts on the path, not the
  sampler.
- A path re-pricing (Dijkstra with stronger charge weights) would change the shared `steiner_graph` every other
  consumer walks.
- The fit's seed can instead be moved where the fit takes it, inside `do_single_tracking`:
  - it touches only trajectories fitted with a JSON that sets the key;
  - it acts on both rounds.

**`seed_recenter_sigma` (mm), with `seed_recenter_iter` (5) and `seed_recenter_max_move` (30 mm).** After
`organize_orig_path`:
- each seed point moves **transversely** to the local path direction (±2 points of the original seed) by a
  Gaussian-kernel mean shift;
- the mean shift runs over the cluster's own 3-D points, weighted by blob charge per point, within a slab of half the
  pass-1 spacing along the path and 3σ across (`TrackFittingUtil::recenter_point_transverse`);
- a mean shift climbs to the nearest density mode, so a brighter branch outside the basin does not capture the
  point (doctest case 4);
- a point whose move would exceed `max_move` keeps its seed position;
- the along-track coordinate never changes, so the fill spacing and the end extension are untouched.

**Offline prototype on 029107_16** (Bee image, the traced `org1` points; not a gate):
- seed points more than 1 cm off: 7.5 % → 5.5 / 3.7 / 3.8 % at σ = 0.5 / 1.0 / 1.5 cm;
- the > 2 cm tail unchanged at 2.2 %, i.e. the far bridges, which have no charge to move toward.

**What it cannot reach, by construction:**
- the far bridges (33–42 % of deviated rows): no image points;
- a crawl re-route into real branch charge (12–14 % of near): the branch is a mode.

Both are named in sec 9.

## 6. The knob and its gates

- **Code** (toolkit `06fd9e22`, pushed to `apply-pointcloud`):
  - `TrackFitting::Parameters::seed_recenter_{sigma,iter,max_move}` + `set_parameter`/`get_parameter`;
  - `TrackFitting::recenter_seed_path`;
  - `TrackFittingUtil::recenter_point_transverse`;
  - the call after `organize_orig_path`, with its own `org1r` trace stage.
- **Knob off:** C++ default `seed_recenter_sigma = 0` ⇒ the call is skipped ⇒ byte-identical. No production JSON
  carries the key; no jsonnet changed.
- **Doctest:** `clus/test/doctest_trackfitting_seed_recenter.cxx`, 5 cases, 20 assertions:
  - the defaults and the round trip;
  - a 12 mm-off seed returns to within 1 mm and keeps its along coordinate;
  - a five-times brighter branch inside the kernel does not capture it;
  - six no-op cases return the input bit for bit.

| gate (knob absent, `libpin_d111b`) | result |
|---|---|
| G1 compiled PR config, PDHD / PDVD | md5 `a870511c9b22` / `211a49a48229`, **equal** to doc 109's values |
| G2 PDHD `d111hoff` (61) vs `d108hflip` | **61 / 61 identical**, STM + PR trees and zip, doc-109 re-scoped layers excluded (`figs/111_gate_off_pdhd.txt`) |
| G2 PDHD `d111hoff` vs `d109hstm` (5 common, every zip member) | **5 / 5 identical** (`figs/111_gate_off_pdhd_scope5.txt`) |
| G2 PDVD `d111voff` (120) vs `d103vflip` | **120 / 120 identical**, trees and zip (`figs/111_gate_off_pdvd.txt`) |
| G3 SBND `work-<s>-d111snew` vs `-d111sold` (`libpin_d102`), pr146 manifest | **BYTE-IDENTICAL**, 16 event dirs, 80 files (`figs/111_gate_off_sbnd.txt`) |
| uBooNE `sweep/d111ub_new` vs `d111ub_old` (`libpin_d102`), `ab_check.sh` | **35 / 35 zips content-identical, tagger logs identical 35** (`figs/111_gate_off_uboone.txt`) |
| the pins are what ran | `wire-cell` carries RUNPATH (not RPATH), so `LD_LIBRARY_PATH` wins; a `dlopen` of `libWireCellClus.so` under each pin maps the pin's file |
| causal control: knob ON `d111hsr10` vs `d111hoff`, first 3 events | **differs on 3 / 3** (the comparator is not blind) |
| `wcdoctest-clus` | **418 / 418** |
| freshness | `libWireCellClus.so` 07:07:52 newer than every edited source (07:06:02); pin md5 `df541468ff1e` |

The clustering job is not gated: `pdhd/clus.jsonnet` and `protodunevd/clus.jsonnet` configure no TaggerCheck or
TrackFitting component, so the changed functions are unreachable there.

## 7. The frozen rule and the knob-on result

**Pre-registration.**
- `figs/111_pred.txt`, sha256 `8f686ca06acb8828…` (`figs/111_pred.sha256`), frozen 2026-09-16T07:16:30. The first
  knob-on arm started at 07:16:47.
- Settings σ = 6, 10, 15 mm and no others.
- Base and knob arms of a detector ran concurrently on `libpin_d111b` (shared load).
- **P6 was replaced before any arm ran**, with the reason written into the file. The simulated muons are contained, so
  no STM fit runs on them, and their PR fit is `do_multi_tracking`, which the knob does not reach. P6′ is a
  clean-record no-regression clause on data.

`d111_eval.py` on the common (event, cluster, pass) records, knob arm vs base (`figs/111_eval_{pdhd,pdvd}_sr{6,10,15}.txt`):

| clause | PDHD σ6 | PDHD σ10 | PDHD σ15 | PDVD σ6 | PDVD σ10 | PDVD σ15 |
|---|---|---|---|---|---|---|
| P1 rows > 1 cm, relative change (need ≤ −20 %) | −8.1 % **FAIL** | −11.1 % **FAIL** | −11.3 % **FAIL** | −5.3 % **FAIL** | −6.3 % **FAIL** | −5.6 % **FAIL** |
| P2 off-charge rows (need ≤ base) | −8.5 % pass | −7.9 % pass | −6.8 % pass | −3.1 % pass | −3.7 % pass | −3.1 % pass |
| P3 Bee holes per 10 m (need ≤ base) | 8.61 → 11.81 **FAIL** | 8.59 → 13.26 **FAIL** | 8.60 → 14.93 **FAIL** | 5.91 → 8.29 **FAIL** | 5.89 → 10.27 **FAIL** | 5.89 → 11.25 **FAIL** |
| P4 image coverage (need ≥ base − 1 pt) | +2.6 pt pass | +4.6 pt pass | +5.3 pt pass | +2.2 pt pass | +4.0 pt pass | +4.6 pt pass |
| P5 h1 / h2 max (need ≤ 1.2 cm; PDVD reported) | 1.26 / 0.71 **FAIL** | 0.54 / 0.93 pass | 0.57 / 2.34 **FAIL** | v2 1.27 → 0.86 | v2 → 0.81 | v2 → 0.60 |
| P6′ clean records, rows > 1 cm (need ≤ 1 %) | 0.37 % pass | 0.33 % pass | 0.42 % pass | 0.18 % pass | 0.18 % pass | 0.19 % pass |
| R median wall ratio (need ≤ 1.20) | 1.028 pass | 1.019 pass | 1.016 pass | 1.016 pass | 1.020 pass | 1.016 pass |
| reported: rows q < 0 | +31 % | +45 % | +57 % | +30 % | +52 % | +62 % |
| reported: rows with chord wiggle > 1 cm | ×5.6 | ×7.3 | ×7.4 | ×3.9 | ×5.1 | ×5.4 |
| reported: fitted length | +4.7 % | +5.8 % | +6.0 % | +2.6 % | +3.9 % | +4.4 % |
| reported: STM tags lost / gained (base 333 PDHD, 563 PDVD) | 74 / 58 | 71 / 68 | 73 / 59 | 103 / 82 | 111 / 94 | 123 / 99 |

**Verdict: no setting passes; the C++ default stays 0 and nothing is flipped.**

**Why it fails, from the traced σ = 10 mm arm on 029107_16** (`d111hsr10tr`, same pin, trace on):
- the re-centring moves 95 % of seed points (median 0.51 cm) and cuts seed points more than 1 cm off the ridge from
  8.1 to 5.7 %;
- it also raises seed points with a 3-point wiggle above 0.5 cm from 11.6 to 39.3 %;
- each point climbs to the charge mean of its own thin slab of a lattice-sampled image, so neighbouring points land
  on different local modes;
- the fit follows the jittered seed, the path gets longer, and `dQ_dx_fit` spreads the same measured charge over
  more, wigglier rows: q < 0 rows +45 %, holes +54 %.
- The missing ingredient is coupling along the path, not the kernel width. That is why no σ passes, and why σ was
  not tuned further (the file allows three values).

**Three things to read with the table:**
- **P4 rewards the defect P3 catches.** A longer, wigglier polyline sweeps more image charge within 1.5 cm, so part of
  the +2–5 point coverage is the jitter. The clause was frozen before this was seen; it is not relaxed or re-read
  here.
- **h2's fix at σ = 10 is indirect: the knob changed which trajectory the crawl produced**
  (`figs/111_h2_record.txt`).
  - Rows of cluster 106 on the side-branch side (e2 < −3 cm) in the window: base 14, every knob arm 0.
  - The round-2 crawl re-route sits 7.9 cm from the click in the base (= production A1). In the traced σ = 10 arm it
    sits 232.6 cm away, where the sampler-only arm S of sec 1 also put it.
  - So the branch detour is gone because the round-1 fit broke elsewhere, not because re-centring pulled the seed off
    the branch. The (cluster, pass) key is the same, the trajectory is not.
  - At σ = 15 mm the window length grows 54.3 → 60.0 cm and h2 fails on jitter.
- **Three events lost every STM candidate under the knob** and ran the Michel stage on nothing. The knob-OFF base arm
  has candidates on all three, so this is a knob effect, not a flaky event:
  - PDHD 028084_28 at σ = 6: base 3 candidates (clusters 106, 110, 131); all three left candidacy (`T_stm_pass`
    status 0 → 2 / 3 / 5);
  - PDVD 039252_13 at σ = 6: base 2 candidates;
  - PDVD 039349_16 at σ = 15: base 4 candidates.
  All three jobs completed; the runner's completeness grep flags them as incomplete. This is the doc-102 g1 class of
  candidate loss.

## 8. What a flip would take, and what the next round should build

**No flip is proposed.** `seed_recenter_sigma` stays in the tree default-OFF, as the measured negative result of this
round and as the instrument for the next.

**The next lever, a redesign rather than a retune:** move the seed as a curve.
- Candidates, in order of simplicity:
  1. re-centre, then smooth the transverse displacement field along the path (a running mean or median of the
     displacement vectors over ±k points) before applying it;
  2. solve all seed points jointly: the kernel pull plus a second-difference penalty on the transverse coordinates;
  3. re-centre on a coarser along-track slab (several fill spacings), so neighbouring points share their image
     points.
- **What must hold** (the same clauses, frozen afresh):
  - P1 gains without the P3 / wiggle cost;
  - the seed's 3-point wiggle does not rise (a clause added to the rule, because this round showed it is the failure
    mode);
  - P4 read together with the fitted length.
- **What it still would not reach:**
  - the far bridges, 33–42 % of deviated rows (no charge to move toward);
  - crawl re-routes into real branch charge (sec 9).

A flip of any trajectory lever is still tagger-affecting: this round's knob arms lose 18–22 % of the base STM tags
and gain back 15–20 % new ones. That grade belongs to the owner's "later".

## 9. Not concluded

- **The far bridges** (33–42 % of deviated rows): the seed crosses regions with no charge on a straight line. This
  is the doc-102 chord class, and bridge pricing in the Steiner graph remains unbuilt. Whether each bridge is a real
  track through a dead region or a glued cluster (doc pdhd/11 R1) was not classified.
- **The crawl re-route** (h2; 12–14 % of near seed-born rows): the break-and-crawl of `adjust_rough_path` is tagger
  logic (it chooses the stopping point). A guard such as "re-route only if the crawl end is on the main track" would
  move tags. Not built.
- **The fit-side mechanisms** stay as measured, without a lever (below the rule):
  - the area smoothing reverting isolated recoveries (8–10 % of seed-born rows);
  - the Steiner-anchored half of the association window (the h1 contrast between `stepped` and `charge_stepped`);
  - single-plane pulls (5 % of all deviated rows).
- **`cloud_na`:** a third of the near seed-born rows are on untagged clusters, whose Steiner cloud is not in the
  scoped Bee layer, so their sub-class is unknown. A calib-dump read of `steiner_pc` would classify them.
- **Simulation.** The plan's truth-based clause could not be run (`figs/111_pred.txt`, P6): the simulated muons are
  contained, so no STM fit runs, and their PR fit uses `do_multi_tracking`. A truth check of the knob needs simulated
  muons that enter the detector.
- **The ridge metric** uses the Bee image, which is blob-sampled. Its 70 % / 3 % agreement with the 2-D cells makes
  it a good detector of departures, not a calibrated distance.

## 10. Files

| path | what |
|---|---|
| toolkit `06fd9e22`: `clus/src/TrackFitting.cxx`, `clus/inc/WireCellClus/TrackFitting.h` | `WCT_TRAJ_ASSOC_DEBUG` trace; `seed_recenter_*` knob |
| toolkit `clus/src/TrackFitting_Util.cxx`, `clus/inc/WireCellClus/TrackFitting_Util.h` | `recenter_point_transverse` |
| toolkit `clus/test/doctest_trackfitting_seed_recenter.cxx` | doctest |
| `scripts/d111_run_arms.sh` | arm runner with trace capture (fork of `d109_arms.sh`) |
| `scripts/d111_identity_gate.py` | trees + zip identity gate |
| `scripts/d111_sbnd_gate.py` | SBND gate (fork of the `d101_gates.sh` comparator) |
| `scripts/d111_spot_frame.py` | sec 1 image-frame decomposition |
| `scripts/d111_spot_stages.py` | sec 1 stage overlays + TSV |
| `scripts/d111_stage_attrib.py` | sec 4 census |
| `scripts/d111_eval.py` | sec 7 metrics |
| `scripts/d111_phaseC_arms.sh` | sec 7 arms (base + 3 settings per detector, concurrent) |
| `scripts/d111_tag_churn.py`, `scripts/d111_seed_jitter.py`, `scripts/d111_h2_record.py` | sec 7 reported numbers |
| `figs/111_*` | figures, TSVs, census, gates, pred file and knob JSONs |
