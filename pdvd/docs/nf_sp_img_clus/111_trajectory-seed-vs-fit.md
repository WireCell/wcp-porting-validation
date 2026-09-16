# doc pdvd/111 — why the fitted STM trajectory leaves the image: the Steiner seed, rarely the fit, and a default-OFF seed re-centring knob; round 2: why the Steiner path leaves the image

**Owner, 2026-09-16** (after doc 110): *"Now, our focus should be on the track trajectory. I understand this would change
tagger, but we can worry about them later. Can you investigate to see what is the reason for the deviated track
trajectory? 1. IS it coming from the seed track trajectory from the graph? (Steiner Graph, Steiner terminal related
issues)? 2. Is it coming from the track trajectory fitting itself (one play did not give good constraint or something
else)? … investigate and understand the situation and then design improvements and implement."*

**How "worry about the tagger later" is read here.** This round does not wait on a doc-56 hand scan. It does not change
a production default either: the lever is built default-OFF, proven byte-identical when off, and measured when on.
Tag changes are reported, not gated. A flip is the owner's call (sec 8).

## Round 2 status (2026-09-16): why the Steiner path leaves the image (sec 11)

**Owner, after round 1:** *"Let's investigate a bit why the Steiner Graph deviate from the 3D image. My understand is
that the origin of the problem must happen inside the retiling step. But I thought that the Steiner Graph Terminal
identificaiton is based on 2D charge, and should bring the terminal back to be consistent with the 3D image. … for
some cases, the Steiner Graph deviates from the 3D images. Can we dig a bit deeper to understand why? What we want is
to figure out some way to reduce these at that level."*

- **Both halves of the premise hold, and the defect sits between them.**
  - **The retile is where off-image material enters.** 54 % (PDHD) / 63 % (PDVD) of retiled points lie more than
    1 cm from the image ridge, against 33 % / 23 % of the image's own points.
    - Of the retiled points 1–5 cm off the ridge, **81 % / 88 % sit on zero-charge cells**, against 17 % / 31 % on
      the ridge.
    - A zero-charge cell is either a dead channel or a cell the retiler painted: `hack_activity_improved` writes
      activity 1e-3 in a ±3-wire × ±3-slice disc around its own two Dijkstra paths.
  - **The terminals do come back onto the image.** They are tighter than the image itself: 23 % / 15 % more than 1 cm
    off, median 0.43 / 0.44 cm.
  - **But terminals are only 13–17 % of the Steiner cloud.**
    - `create_enhanced_steiner_graph` admits every vertex on the Voronoi shortest paths between terminal regions with no
      charge test. These interiors are 36 % / 41 % off the ridge.
    - Of the Steiner vertices 1–5 cm off the ridge, **68 % / 69 % are on painted cells** (a zero-charge plane with no
      dead channel there), and 3 % / 7 % on dead-only cells.
    - The Dijkstra walk prices charge only at an edge's two ends, so a painted end costs at most 1.2 / 0.8 = 1.5×.
- **For most near deviations the graph holds no on-image route** (seed 1–5 cm off).
  - An image oracle (Dijkstra that avoids every off-ridge edge) finds **no on-image route between the same walk ends in
    52 % (PDHD) / 59 % (PDVD)** of these rows.
  - Where one exists (48 % / 41 %), it costs at most 1.5× the chosen route under today's weights, except 1.7 % of the
    PDVD rows.
  - In 70 % / 75 % of these rows the seed edge carrying the deviation has at least one end on a zero-charge cell.
- **Far deviations** (seed more than 5 cm off) are graph bridges:
  - 79 % / 83 % run on edges joining terminal regions, built by `connect_graph_ctpc_with_reference` or the MST of
    `connect_graph_with_reference`;
  - 82 % / 81 % run more than half their length over 1 cm from any retiled point;
  - an on-image route exists in only 7 % / 3 %.
- **Re-pricing today's graph does not reach the pre-registered bar** (offline replay of every round-1 walk, frozen rule
  `figs/111s_ladder_rule.txt`, sec 11.7).
  - The best form, 3-plane 2-D support along each edge, cuts off-ridge seed length by at most 21.5 % / 17.9 %; the
    rule needs 25 %.
  - It also adds lattice staircase (wiggle 4.4 → 6.1 % / 9.7 → 11.7 %).
  - The SBND `steiner_gap_penalty` form moves it by 3 % / 1 %. Pricing painted ends moves it by 16.5 % / 12.1 %.
  - Even a perfect re-pricing could reach at most the rows that have an on-image route: 29 % / 27 % of seed-born
    deviated rows.
  - **No lever was built.**
- **What was built:** `WCT_STEINER_GRAPH_DUMP` (toolkit `724cf205`), log-only.
  - Byte-identical on PDHD (61), PDVD (120), SBND (16 events) and uBooNE (35).
  - Offline it reproduces every rough walk's cost (1151/1151, 1847/1847).
- **Round 1's "cloud displaced in only 7 %" is withdrawn.** It came from the tag-scoped Bee layer. The same rule on
  every cluster gives 23 % / 28 % (sec 11.3).
- **Next round (needs the owner's go):** change what enters the Steiner graph, not how it is priced (sec 11.8).
  - Keep painted cells out of the tree interior, or charge-test the interiors.
  - Or shrink the retiler's painted disc.
  - All are shared-graph changes.

## Status

- **Question 1, the seed: yes, almost always.** Every STM fit is traced stage by stage (new log-only trace, sec 3):
  - of the fitted rows more than 1 cm from the image ridge, **90 % (PDHD) / 88 % (PDVD) were already that far off in
    the seed at the same place** (the Steiner Dijkstra path, or its 1.2 cm fill; sec 4);
  - the seed defect is **the path, not the terminals and not the cloud**:
    - at the owner's h1 the terminals are the same 18 in every arm;
    - for seed-born deviations 1–5 cm from the ridge, the Steiner cloud has itself moved off the ridge in only 7 %
      of the rows (**withdrawn in round 2:** measured on tagged clusters only; on every cluster the same rule gives
      23 % / 28 %, and most of the cloud's off-image material is painted retile cells, sec 11.3 and 11.5);
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

**Round 2 correction:** this split used the tag-scoped Bee cloud. On every cluster (calib `steiner`), the same rule
gives:
- `long_edge` 41 / 38 %;
- `crawl_reroute` 12 / 14 %;
- `cloud_has_ridge` 23 / 21 %;
- `cloud_off` 23 / 28 %.

A 0.5 cm test is also below the cloud's 0.8–0.9 cm spacing. See sec 11.3.

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
  scoped Bee layer, so their sub-class is unknown. A calib-dump read of `steiner_pc` would classify them. (Done in
  round 2, sec 11.3.)
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

## 11. Round 2 — why the Steiner path leaves the image

The owner's round-2 question and the answer in brief are in the "Round 2 status" block at the top. This section holds
the evidence.

### 11.0 Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; D=$IMG/pdvd/docs/nf_sp_img_clus; S=$D/scripts; F=$D/figs
T=/home/xqian/toolkit-dev/toolkit

# build (toolkit 724cf205).  The arms ran on pin libpin_d111s (clus 434468cf1b16): the committed source except three
# whitespace-only lines of TaggerCheckSTM.cxx.  The commit's own build is libpin_d111s2 (clus 362c634a95d6).
(cd $T && ./wcb build --notests -p && ./wcb install --notests -p && ./wcb build --targets=wcdoctest-clus -p \
   && ./build/clus/wcdoctest-clus)                                     # 421 / 421

# sec 11.3 -- round 1's cloud classes on every cluster (no new run: the round-1 trace arms and their calib dumps)
python3 $S/d111s_cloud_census.py pdhd d111htr > $F/111s_cloud_pdhd.txt
python3 $S/d111s_cloud_census.py pdvd d111vtr > $F/111s_cloud_pdvd.txt

# sec 11.4 -- the dump arms (production config) and the gates
P=/home/xqian/tmp/d111/libpin_d111s; TR="WCT_STM_PATH_DEBUG=1 WCT_STEINER_GRAPH_DUMP=1"
ARM=d111hst DET=pdhd SRC=d108hflip JOBS=6 PIN=$P TRACE_ENV="$TR" bash $S/d111_run_arms.sh
ARM=d111vst DET=pdvd SRC=d103vflip JOBS=6 PIN=$P TRACE_ENV="$TR" bash $S/d111_run_arms.sh
ARM=d111shoff5 DET=pdhd SRC=d108hflip JOBS=3 PIN=$P EVENTS="028084_3 028084_12 029107_5 029107_16 029107_18" bash $S/d111_run_arms.sh
ARM=d111svoff5 DET=pdvd SRC=d103vflip JOBS=3 PIN=$P EVENTS="039252_8 039253_6 039349_2 039349_18 039349_20" bash $S/d111_run_arms.sh
python3 $S/d111_identity_gate.py --det pdhd --a d108hflip --b d111hst \
    --ignore-members stm_tagged,stm,stm_fit,steiner_graph,steiner_terminals > $F/111s_gate_dump_pdhd.txt
python3 $S/d111_identity_gate.py --det pdvd --a d103vflip --b d111vst > $F/111s_gate_dump_pdvd.txt
python3 $S/d111_identity_gate.py --det pdhd --a d109hstm --b d111shoff5 > $F/111s_gate_dumpoff5_pdhd.txt
python3 $S/d111_identity_gate.py --det pdvd --a d103vflip --b d111svoff5 --allow-subset > $F/111s_gate_dumpoff5_pdvd.txt
(cd $S && TAG=d111ssnew PIN=$P JOBS=2 bash d101_sbnd_arm.sh)
python3 $S/d111_sbnd_gate.py --old d111sold --new d111ssnew --ignore-branch Trun.toolkit_git > $F/111s_gate_sbnd.txt
(cd $IMG/qlport/scripts && LD_LIBRARY_PATH=$P ./sweep_5384.sh d111ub_s 3 && ./ab_check.sh d111ub_s d111ub_old) > $F/111s_gate_uboone.txt
bash $S/d102_compile_pr.sh pdhd d111spost; bash $S/d102_compile_pr.sh pdvd d111spost        # a870511c9b22 / 211a49a48229
ARM=d111shoff1 DET=pdhd SRC=d108hflip JOBS=1 PIN=/home/xqian/tmp/d111/libpin_d111s2 EVENTS=029107_16 bash $S/d111_run_arms.sh
python3 $S/d111_identity_gate.py --det pdhd --a d109hstm --b d111shoff1 --events 029107_16 --allow-subset \
    > $F/111s_gate_final_pin_pdhd.txt

# sec 11.5 -- the Steiner-level census (the join and replay checks print first) and the painted/dead split
python3 $S/d111s_graph_census.py --det pdhd --arm d111hst --out $F/111s_graph_pdhd     # .txt + _rows.tsv.gz
python3 $S/d111s_graph_census.py --det pdvd --arm d111vst --out $F/111s_graph_pdvd
python3 $S/d111s_painted_split.py --det pdhd --arm d111hst > $F/111s_painted_split_pdhd.txt
python3 $S/d111s_painted_split.py --det pdvd --arm d111vst > $F/111s_painted_split_pdvd.txt

# sec 11.6 -- the owner's spots at the Steiner level
python3 $S/d111s_spot_anatomy.py --out $F/111s_spot                                     # 111s_spot_<spot>.png, .tsv

# sec 11.7 -- the ladder (rule frozen before the full replay) and the exploratory bow/jitter split
(cd $F && sha256sum -c 111s_ladder_rule.sha256)
python3 $S/d111s_replay.py --det pdhd --arm d111hst --out $F/111s_replay_pdhd            # .txt + .json
python3 $S/d111s_replay.py --det pdvd --arm d111vst --out $F/111s_replay_pdvd
python3 $S/d111s_wiggle_split.py --det pdhd --arm d111hst > $F/111s_wiggle_split_pdhd.txt
python3 $S/d111s_wiggle_split.py --det pdvd --arm d111vst > $F/111s_wiggle_split_pdvd.txt
```

All arms are new tags (M13). Their pctrees are symlinks to the production arms' (PDHD `d108hflip`, PDVD `d103vflip`).

### 11.1 The premise against the code

The owner's picture: the retile makes the cloud, the terminals are chosen by 2-D charge and so sit on the image, and a
path between on-image terminals should stay on the image. What the code does (toolkit `724cf205`):

1. **The retiler paints its own paths.** `ImproveCluster_2::mutate` (improvecluster_2.cxx) walks two geometric
   Dijkstra paths between the cluster's boundary points:
   - one on the original cluster's `basic_pid` graph (:128);
   - one on the first retile's `ctpc_ref_pid` graph (:179).
   `hack_activity_improved` (improvecluster_1.cxx:460) then writes activity `1.0e-3` into every empty (wire, slice)
   cell within a ±3-wire × ±3-slice disc around each path point that lacks 3-plane activity (:609-650). The re-tiling
   turns painted cells into blobs, and the sampler turns them into points with a zero charge on the painted plane.
   Doc pdhd/08 capped the bridges of this painting (`hack_max_bridge`, 10 cm in production); the disc around every
   point is untouched.
2. **The Steiner base graph** is `ctpc_ref_pid` on the retiled cluster (CreateSteinerGraph.cxx:288, make_graphs.cxx:73):
   `connect_graph_closely_pid` (in-blob and adjacent-slice edges), then `connect_graph_ctpc_with_reference`, then
   `connect_graph_with_reference` (an MST over the remaining components, uncapped).
3. **Terminals are charge-selected.** `find_steiner_terminals` keeps the per-blob peaks of `calc_charge_wcp` above
   `terminal_charge_threshold` (SteinerGrapher.cxx:367). The extreme points are added without a charge test (:429).
4. **The Steiner cloud is mostly not terminals.** `create_enhanced_steiner_graph` (:1450):
   - builds the Voronoi regions of the terminals on the base graph;
   - keeps the best edge between each pair of adjacent regions;
   - adds every vertex on the `last_edge` chains back to both terminals (:1518). These path interiors get no charge
     test.
   Every terminal is then joined to every Steiner point of its blob (`establish_same_blob_steiner_edges_steiner_graph`,
   :1386, weight = plain length).
5. **The walk.** `TaggerCheckSTM::do_rough_path` runs Dijkstra on `steiner_graph` (TaggerCheckSTM.cxx:1228).
   - The weight is length × (0.8 + 0.4 × the mean of Q0 / (Q + Q0) over the two endpoints) (:1616). A zero-charge end
     costs at most 1.2 / 0.8 = 1.5× a bright one.
   - Nothing looks at the segment between the ends.

So charge selects the terminals, and geometry selects the path interiors, the ends and the route.

### 11.2 The production Steiner stage

**Production Steiner-stage keys** (compiled `d111spost_{pdhd,pdvd}.json`, identical on both detectors):
- `CreateSteinerGraph`:
  - `terminal_charge_threshold` 500;
  - `terminal_min_separation` 5 mm;
  - `terminal_wire_tol` 1;
  - `terminal_adjacent_slice` true;
  - `edge_charge_forward_dead_mix` true.
- `ImproveCluster_2`:
  - `hack_max_bridge` 100 mm;
  - `bad_blob_max_run` 200 mm;
  - `wrapped_channel_activity` true;
  - `terminal_charge_threshold` 500.
- `MultiAlgBlobClustering:clus_pr` `ctpc_aniso_metric` true.
- PDHD `TaggerCheckSTM` `rough_path_require_connected` true.
- `CheckSTM_Michel` runs `steiner_gap_penalty` 2 on its own PR walks; TaggerCheckSTM, which makes `stm_fit`, does not.

### 11.3 Round 1's cloud classes on every cluster

`d111s_cloud_census.py` reads the calib dump's `steiner` block. It holds steiner_pc x/y/z and the terminal flag for every
cluster, not only the tagged ones. The rows are round 1's seed-born deviated rows (origin seed or org1), located at the
persisted seed (`figs/111s_cloud_{pdhd,pdvd}.txt`).

| near rows (seed 1–5 cm off): 14,564 PDHD / 15,912 PDVD | PDHD | PDVD |
|---|---|---|
| round 1's rule on the full cloud: `long_edge` | 41.2 % | 37.8 % |
| `crawl_reroute` | 12.2 % | 13.5 % |
| `cloud_has_ridge` (a Steiner point < 0.5 cm from the ridge within 1.5 cm) | 23.3 % | 20.9 % |
| `cloud_off` | 23.3 % | 27.8 % |
| no Steiner point within 1.5 cm of the seed | 19.4 % | 19.9 % |
| a Steiner point within 1.5 cm lies < 1 cm from the ridge | 57.3 % | 55.4 % |
| nearby Steiner points all ≥ 1 cm off the ridge, one ≤ 1 cm from an image point | 12.0 % | 13.2 % |
| nearby Steiner points all ≥ 1 cm off the ridge, none near the image | 11.4 % | 11.5 % |

- Round 1's 7 % `cloud_off` (sec 4.3) came from the tagged-cluster Bee layer and is withdrawn: on every cluster it is
  23 % / 28 %.
- The 0.5 cm test is finer than the cloud's in-slice spacing (0.80 cm PDHD, 0.88 cm PDVD, doc 102 sec 5.4), so part of
  `cloud_off` is sparse cloud rather than displaced cloud. The 1 cm split below the rule is the fairer one.
- This census sees `steiner_pc` only, not the retiled cloud or the graph's edges. That is why the dump of sec 11.4 was
  built.

### 11.4 The dump and its gates

**`WCT_STEINER_GRAPH_DUMP`** (toolkit `724cf205`) is log-only and prints to stdout.
- From `Steiner::Grapher::create_steiner_tree` (`steiner_graph_dump`):
  - `STGC` header;
  - `STGR`: every retiled point with its per-plane charge;
  - `STGV`: every steiner_pc vertex with its terminal and extreme flags, edge-weight charge, and test_good_point
    live/dead masks at radius 0.2 cm / ch_range 0 and 0.6 cm / ch_range 1;
  - `STGE`: every steiner_graph edge with length, weight and source. The source is `path`, `connect` or `sbt`, from a
    second `voronoi()` and a pre-`sbt` edge snapshot. The base provenance is `closely_same`, `closely_other`, `ctpc`
    or `mst`, from rebuilding the first two stages of `make_graph_ctpc_pid`. It also carries the
    `Steiner::edge_support` sample counts at both radii.
- From TaggerCheckSTM: `STMRP`, the steiner_pc indices of `do_rough_path` and of both crawl walks.
- The pure sampling core is `clus/inc/WireCellClus/SteinerEdgeSupport.h`, with
  `clus/test/doctest_steiner_edge_support.cxx` (3 cases).

**Offline checks** (`figs/111s_graph_{pdhd,pdvd}.txt`, header):

| check | PDHD | PDVD |
|---|---|---|
| every STM seed point is a dumped vertex of its cluster | 687,824 / 687,824 | 1,190,810 / 1,190,810 |
| scipy Dijkstra on the dumped graph = the C++ round-1 rough walk's cost | 1151 / 1151 (1145 on the identical path, 6 equal-cost ties) | 1847 / 1847 (all identical) |
| dump lines rejoined after a stderr log line was spliced into them | 6 | 37 |

**Gates** (dump code on pin `libpin_d111s`, clus `434468cf1b16`, unless noted):

| gate | result |
|---|---|
| dump ON, PDHD `d111hst` (61) vs `d108hflip` | **61 / 61 identical**, STM + PR trees and zip (doc-109 re-scoped layers excluded) (`figs/111s_gate_dump_pdhd.txt`) |
| dump ON, PDVD `d111vst` (120) vs `d103vflip` | **120 / 120 identical**, trees and zip (`figs/111s_gate_dump_pdvd.txt`) |
| dump OFF, PDHD `d111shoff5` vs `d109hstm` (5, every zip member) | **5 / 5 identical** (`figs/111s_gate_dumpoff5_pdhd.txt`) |
| dump OFF, PDVD `d111svoff5` vs `d103vflip` (5) | **5 / 5 identical** (`figs/111s_gate_dumpoff5_pdvd.txt`) |
| SBND `work-<s>-d111ssnew` vs `-d111sold` (16 events, pr146 manifest) | **BYTE-IDENTICAL**, 80 files; `Trun.toolkit_git` ignored: it records the source tree's git HEAD at run time (`f573cdeb-dirty` vs `06fd9e22-dirty`) and was the only differing branch (`figs/111s_gate_sbnd.txt`) |
| uBooNE `sweep/d111ub_s` vs `d111ub_old` | **35 / 35 zips content-identical, tagger logs identical 35** (`figs/111s_gate_uboone.txt`) |
| compiled PR config PDHD / PDVD | md5 `a870511c9b22` / `211a49a48229`, **equal** to doc 109's |
| the commit's own build (`libpin_d111s2`, clus `362c634a95d6`): `d111shoff1` 029107_16 vs `d109hstm` | **identical**, every zip member (`figs/111s_gate_final_pin_pdhd.txt`) |
| `wcdoctest-clus` | **421 / 421** |
| freshness | `libWireCellClus.so` 10:35:11 newer than every edited source (10:34:22); rebuilt 11:10:26 after the whitespace restore (11:09:27) |

`d111_sbnd_gate.py` gained `--ignore-branch` (default none, which is round 1's behaviour; round 1's pair still passes
without it).

**Cost of the dump when on:** 85 s on 029107_16 against about 36 s, and about 16 MB of gzipped log per PDHD event.
Nothing is paid when it is off.

### 11.5 Where the seed-born deviation is made

`d111s_graph_census.py` over every persisted STM record of the dump arms (PDHD 1151 used of 1158, 7 with no image; PDVD
1847). Rows are round 1's seed-born deviated rows, located at the persisted round-2 seed.

**Null floors: how far points of each kind sit from the image ridge**

| points of the fitted clusters | PDHD > 1 cm | PDHD p50 | PDVD > 1 cm | PDVD p50 |
|---|---|---|---|---|
| image (the metric's own scatter) | 33.2 % | 0.84 cm | 22.5 % | 0.71 cm |
| **retiled cloud** | **53.9 %** | 1.08 cm | **62.8 %** | 1.29 cm |
| **terminals** | **23.4 %** | 0.43 cm | **14.7 %** | 0.44 cm |
| path interiors (not terminal, not extreme) | 35.7 % | 0.76 cm | 40.8 % | 0.85 cm |

**Zero-charge cells, by ridge offset** (`figs/111s_graph_*.txt`, `figs/111s_painted_split_*.txt`)

| | PDHD ≤ 1 / 1–5 / > 5 cm | PDVD ≤ 1 / 1–5 / > 5 cm |
|---|---|---|
| retiled points with a zero-charge plane (painted or dead) | 16.9 / **81.4** / 51.2 % | 30.6 / **87.9** / 66.7 % |
| Steiner vertices with a zero-charge plane (painted or dead) | 15.3 / **70.8** / 33.4 % | 26.4 / **76.3** / 52.4 % |
| Steiner vertices **painted** (a zero plane with no dead channel at 0.2 cm) | 12.0 / **67.9** / 31.3 % | 17.3 / **69.0** / 38.0 % |
| Steiner vertices on dead-only cells | 3.3 / 2.9 / 2.2 % | 9.0 / 7.3 / 14.4 % |

- The retiled cloud's off-image part is mostly painted.
- Painted vertices enter the Steiner graph as path interiors, and some as terminals: a painted plane still lets
  `calc_charge_wcp` pass on the other two. 51,481 of the 87,218 PDHD terminals 1–5 cm off the ridge are painted.

**Near rows (seed 1–5 cm off): does the graph hold an on-image route?**

The image oracle re-weights every edge by 1 + 100 × (share of its straight segment more than 1 cm off the ridge). It
then walks the same ends as the production seed: the crawl walks when `adjust_rough_path` crawled, else the rough walk.
The cost ratio compares the oracle route and the chosen route under the production weights.

| near rows at the location | PDHD (14,564) | PDVD (15,912) |
|---|---|---|
| on-image route, cost ≤ 1.05× | 28.3 % | 15.9 % |
| on-image route, 1.05–1.2× | 16.2 % | 16.1 % |
| on-image route, 1.2–1.5× | 3.1 % | 7.7 % |
| on-image route, > 1.5× | 0.0 % | 1.7 % |
| **no on-image route between these ends** | **52.4 %** | **58.6 %** |

**Near rows: the seed edge that carries the deviation**

| | PDHD | PDVD |
|---|---|---|
| at least one end on a zero-charge cell (1 / 2 ends) | **70.1 %** (24.8 / 45.3) | **75.2 %** (26.9 / 48.3) |
| an end > 1 cm off the ridge is a path interior (alone or with a terminal) | 73.1 % | 70.7 % |
| an end > 1 cm off the ridge is a terminal only | 8.9 % | 8.5 % |
| both ends on the ridge (the chord between them leaves it) | 16.5 % | 19.4 % |
| an end > 1 cm from every image point | 39.1 % | 41.4 % |
| edge > 3 cm | 43.7 % | 40.9 % |
| its segment is mostly within 1 cm of image and retiled points | 50.7 % | 44.5 % |
| mostly within 1 cm of retiled points but not of image (a painted or ghost corridor) | 29.9 % | 31.8 % |
| mostly near neither | 19.3 % | 23.3 % |
| source / base: `path` / `closely_other` | 26.4 % | 31.6 % |
| `connect` / `ctpc` | 16.6 % | 20.0 % |
| `sbt` (same-blob terminal edge) | 15.9 % | 11.0 % |
| `path` / `closely_same` | 12.4 % | 9.5 % |
| `connect` / `closely_same` + `closely_other` | 22.6 % | 20.8 % |
| `connect` / `mst` | 5.4 % | 6.8 % |

**Far rows (seed > 5 cm off; 12,622 PDHD / 9,388 PDVD)**
- Base provenance: `connect` / `ctpc` 47.2 / 55.6 %; `connect` / `mst` 32.2 / 27.1 %.
- Their segment is mostly more than 1 cm from any retiled point in 82.0 / 81.1 %; mostly on retiled-only points in
  18.0 / 18.7 %.
- No on-image route between the walk ends in 93.4 / 96.7 %.
- These are the doc-102 chords. They are bridges between image pieces, and re-pricing cannot remove a bridge that is
  the only connection.

**The layer at the ridge foot**, reported as it was frozen in the plan. For each row, the census tests the image,
retiled and Steiner points within 1 cm of the ridge point nearest the seed location. It gives:

| near rows | PDHD | PDVD |
|---|---|---|
| no image at the foot | 1.7 % | 3.3 % |
| image, no retiled point (`retile_gap`) | 0.5 % | 1.6 % |
| retiled point, no Steiner vertex (`tree_omission`) | 8.1 % | 7.2 % |
| a Steiner vertex within 1 cm (`path_choice`) | 89.6 % | 87.9 % |

This split is close to definitional and must not be read as "the graph holds the ridge". A 1 cm ball around any foot
inside the cluster almost always contains a Steiner vertex at 0.8–0.9 cm spacing. The oracle table above is the
measurement: in more than half the rows those vertices are not connected along the ridge. The plan's "retile or tree
layer dominates" stop (`retile_gap` + `tree_omission` ≥ 50 %) did not fire (8.6 % / 8.8 %).

**Which 2-D test can see an off-image edge?** Round-2 seed edges ≥ 0.5 cm, on the ridge (none of the segment more than
1 cm off) against off it (at least half more than 1 cm off):

| test (share of samples failing) | PDHD mean on / off | PDHD ≥ 0.2 on / off | PDVD mean on / off | PDVD ≥ 0.2 on / off |
|---|---|---|---|---|
| 3 planes live or dead, radius 0.2 cm | 0.094 / 0.592 | 21.4 / 74.0 % | 0.095 / 0.603 | 19.0 / 75.4 % |
| 3 planes, radius 0.6 cm / ch_range 1 | 0.002 / 0.161 | 0.3 / 22.6 % | 0.005 / 0.135 | 0.7 / 17.4 % |
| 2 planes, radius 0.2 cm | 0.017 / 0.142 | 4.6 / 25.8 % | 0.010 / 0.237 | 1.9 / 36.9 % |
| SBND `steiner_gap_penalty` (unsupported + 0.25 dead), radius 0.2 cm | 0.001 / 0.044 | 0.4 / 7.2 % | 0.004 / 0.059 | 0.6 / 9.1 % |

The SBND unsupported test fires only where no plane sees charge, so it cannot see a near deviation. At least one
plane's wire always crosses the track nearby.

### 11.6 The owner's spots at the Steiner level

`figs/111s_spot_<spot>.png`. Columns:
1. image, retiled cloud, Steiner vertices and terminals;
2. steiner_graph edges by source, with the seed;
3. the seed against the image oracle and two re-priced replays of the same walks.

The crawl point is kept, so a crawl detour stays in every replay. `figs/111s_spot.tsv`, window as in sec 1:

| spot | retiled / Steiner / terminals > 1 cm off | seed max | seed edge at the worst point | oracle | G_3pl k=2 | G_2pl k=2 |
|---|---|---|---|---|---|---|
| h1 cl108 | 41 / 29 / 22 % | 1.97 cm | `connect` / `closely_other`, 1.71 cm | **0.72** | **0.68** | 1.81 |
| h2 cl106 | 65 / 38 / 17 % | 2.77 | `path` / `closely_other`, 1.31 cm | 2.44 | 2.44 | 2.44 |
| v1 cl80 | 76 / 43 / 8 % | 1.03 | `connect` / `closely_other` | 1.03 | 1.03 | 1.03 |
| v2 cl80 | 77 / 48 / 19 % | 0.99 | `path` / `closely_other` | 0.99 | 0.99 | 0.96 |
| v3 cl25 | 78 / 41 / 10 % | 1.64 | `path` / `closely_other` | 0.93 | 1.15 | 1.54 |

- **h1.**
  - The V's terminals sit on the ridge at its bottom (e2 ≈ −1.5 to −2 cm).
  - `sbt` edges fan across the V: each terminal is wired to every Steiner point of its blob, and the blobs are long
    diagonal strips.
  - The seed walks the upper edge of the cloud.
  - Both the oracle and the 3-plane replay put the seed onto the V. This is a pricing case, and the only spot where
    re-pricing helps.
- **h2.**
  - The worst stretch is round 1's crawl detour into the side branch. Every replay keeps the crawl point, so no
    re-pricing moves it (2.44 cm). It is tagger logic (sec 9), not the Steiner graph.
  - Around it, the retiled cloud sits in a band below the image (65 % of retiled points more than 1 cm off).
- **v1, v2.** The production seed is already within about 1 cm; nothing moves.
- **v3.** A 1.6 cm offset that the oracle removes (0.93) and the 3-plane replay halves (1.15).
- **The h2 maximum appears three ways; they measure different things:**
  - 2.25 cm in sec 1: the persisted seed, measured at the final rows' locations within ±24 cm;
  - 2.77 cm here: seed samples within the window, crawl walks;
  - 0.84 cm in the ladder: the round-1 rough walk, which has no crawl.

### 11.7 The counterfactual ladder and the decision

**Pre-registration.**
- The rule is `figs/111s_ladder_rule.txt`. The original text (sha256 `5464b338…`) was frozen at 10:44:56 and saved with
  the file. Amendment 1 was added at 10:49, before any full-arm replay; the final sha256 `92651f7e…` is in
  `figs/111s_ladder_rule.sha256`.
- The final rule file is from 10:49:56. The PDHD dump arm finished at 10:50:08, so no full-arm replay could exist
  before the freeze. The first full replays failed in the parser (the spliced log lines of sec 11.4) and were rerun on
  identical inputs.
- Two plan changes are written into the rule with their reasons:
  - the 0.2 cm test radius. The plan picked the radius by a vertex null, which cannot fail on lattice points;
  - the added G_2pl family.
- **Amendment 1** adds `P_paint`, reported but not buildable this round. It needs the retiled per-plane charge, which
  only CreateSteinerGraph sees.
- The smoke event 029107_16 was replayed after the original freeze and before amendment 1, which adds a family and
  changes no threshold. No full-arm number existed at either time.

**What is replayed.**
- Every round-1 rough walk (`STMRP rough`) of clusters with a persisted record (1151 PDHD, 1847 PDVD), on its dumped
  graph, with scipy Dijkstra.
- The base weights reproduce the C++ cost (sec 11.4).
- The crawl walks are not replayed, because the crawl point depends on the round-1 fit, which a lever changes. The
  ladder therefore scores the round-1 seed. The census above scored the persisted round-2 seed, which was re-routed by a
  crawl in 823 of 1158 PDHD and 1186 of 1847 PDVD records.

**Metrics.**
- off: share of seed length more than 1 cm off the ridge;
- wig: share of points of the seed resampled at 1.2 cm that sit more than 0.5 cm from their neighbours' chord;
- cov: image charge within 1.5 cm.

**Rule.**
- Qualify on both detectors: off ≤ −25 % relative, wig ≤ 1.10× base, cov ≥ base − 1 point, no new unreachable walk.

**Results** (`figs/111s_replay_{pdhd,pdvd}.txt`, relative change of off, wig):

| candidate | PDHD off (base 9.87 %) | PDHD wig (base 4.43 %) | PDVD off (base 7.82 %) | PDVD wig (base 9.71 %) | qualifies |
|---|---|---|---|---|---|
| G_any k = 2 / 5 / 10 (SBND form) | −2.8 / −3.0 / −3.0 % | 4.51 / 4.52 / 4.52 % | −1.4 / −1.3 / −1.1 % | 9.81 / 9.88 / 9.92 % | no (off) |
| G_3pl k = 2 / 5 / 10 | **−21.5** / −21.1 / −20.3 % | 6.13 / 7.33 / 7.95 % | −17.6 / **−17.9** / −17.2 % | 11.70 / 12.84 / 13.45 % | no (off, wig) |
| G_2pl k = 2 / 5 / 10 | −10.4 / −9.9 / −9.3 % | 4.99 / 5.37 / 5.56 % | −9.0 / −8.1 / −7.5 % | 10.25 / 10.49 / 10.55 % | no (off, PDHD wig) |
| E_len k = 0.5 / 1 / 2 (control) | −2.5 / −2.1 / −1.9 % | 4.55 / 4.60 / 4.63 % | −2.3 / −1.7 / −1.3 % | 10.00 / 10.05 / 10.06 % | control |
| P_paint k = 2 / 5 / 10 (amendment 1) | −16.5 / −15.7 / −15.3 % | 5.25 / 5.51 / 5.62 % | −12.1 / −10.8 / −10.0 % | 11.29 / 11.90 / 12.17 % | no (off, wig); not buildable |

- Coverage rises slightly under every candidate (PDHD 59.2 → 60.7 % at most; PDVD 49.5 → 50.6 %); no cov clause
  fails.
- No walk becomes unreachable.
- Far seed length (more than 5 cm off) barely moves (4.00 → 3.85 % PDHD at most), because bridges have no alternative.
- Spots in the ladder (round-1 rough walk): h1 1.97 → 0.68 cm under G_3pl k=2, G_2pl k≥5 and P_paint; v3 1.64 → 1.15;
  h2, v1 and v2 stay near 1 cm.

**Verdict: no candidate qualifies on either detector, so under the frozen rule no lever is built this round.**

**Why re-pricing cannot get there**
- **The reach is capped.** On-image routes exist in:
  - PDHD: 6,935 of 14,564 near rows and 833 of 12,622 far rows, 28.6 % of seed-born deviated rows;
  - PDVD: 6,586 of 15,912 near rows and 312 of 9,388 far rows, 27.3 %.
  A perfect re-pricing of today's graph could fix only those rows.
- **The wiggle it adds is lattice staircase, not bow.** From the exploratory split (not pre-registered;
  `figs/111s_wiggle_split_{pdhd,pdvd}.txt`, ±2 cm running mean):

  | | PDHD base → G_3pl:2 | PDVD base → G_3pl:2 |
  |---|---|---|
  | raw length / smoothed length | 1.1545 → 1.1794 | 1.1988 → 1.2278 |
  | smoothed length vs the base's | 1.0018 | 1.0013 |
  | jitter rms about the smoothed seed | 0.219 → 0.230 cm | 0.261 → 0.269 cm |
  | off, smoothed seed | 9.92 → 8.13 % (−18 %) | 7.14 → 6.24 % (−13 %) |

  - The production seed is already a staircase (raw / smoothed 1.15–1.20). Steiner points sit on the wire-crossing ×
    slice lattice (doc sbnd_xin/pr/73 sec 4.11).
  - Routing onto the ridge through that lattice adds steps, not detours.
  - Re-route-then-smooth keeps most of the gain but still falls short of 25 %.

### 11.8 What to build next (design; each needs the owner's go)

The census points at what enters the graph, not only at how the walk prices it.

1. **Keep painted cells out of the Steiner tree interior**, or price them in the base graph before the Voronoi step.
   - This acts where 70 % / 75 % of the near carrying edges have a painted or dead end, and where 68 % / 69 % of the
     off-ridge vertices are painted.
   - Pricing painted ends after the tree is built already gives −16.5 % / −12.1 % (P_paint). It is a lower bound: after
     the tree, pricing cannot add the on-ridge connections that the painted corridor replaced.
   - Site: `create_enhanced_steiner_graph` / `CreateSteinerGraph`, a default-OFF key. It changes the shared
     `steiner_graph` (PR walks, the fit association's Steiner half, the taggers).
2. **Charge-test the Voronoi path interiors** with the test the terminals already pass, or run the Voronoi on a
   charge-weighted base graph.
   - The terminals' own null floor shows what charge selection does: 23 % / 15 % of terminals more than 1 cm off,
     against 36 % / 41 % of interiors.
3. **Shrink the retiler's painted disc.** Today it is ±3 wires × ±3 slices around every path point lacking 3-plane
   activity (`hack_activity_improved`, improvecluster_1.cxx:609-650). A default-OFF radius, or no painting within ±1
   wire of real 3-plane activity, would stop the halo at its source.
   - It changes the retiled cloud for every consumer, and the terminals too: some are painted.
   - It needs its own gate, like doc pdhd/08's bridge cap.
4. **Whatever re-routes must also smooth.** The frozen rule of the next round should split bow from jitter, as in doc
   pr/73 sec 4.7, rather than carry one wiggle number.
5. **Far bridges are not a Steiner-pricing problem.** 93 % / 97 % have no on-image alternative. They belong to the
   component bridging (`connect_graph_ctpc_with_reference`, the MST) and to clustering (doc pdhd/11 R1, doc 102).

Recommended order: 1 and 3 measured together on the dump (the dump already carries per-vertex painted flags and per-edge
provenance), then a frozen rule with a bow/jitter clause.

### 11.9 Not concluded

- **Painted against dead is split only for Steiner vertices.** Retiled points carry no dead mask, so the retiled-point
  row of sec 11.5 is painted-or-dead.
- **A zero plane could have other causes**, such as a sampler that failed to assign charge on a live wire. The wrapped-
  wire fix (`wrapped_channel_activity`) is on in production; no other cause was tested.
- **The ladder scores round-1 rough walks**, not the persisted round-2 seed.
- **The oracle uses the ridge metric itself**, so it is an upper bound on what an image-blind lever could find.
- **No tag grade.** Nothing was built, so nothing was graded.

### 11.10 Files (round 2)

| path | what |
|---|---|
| toolkit `724cf205`: `clus/src/SteinerGrapher.cxx` | `steiner_graph_dump` (`WCT_STEINER_GRAPH_DUMP`) |
| toolkit `clus/src/TaggerCheckSTM.cxx` | `STMRP` lines |
| toolkit `clus/inc/WireCellClus/SteinerEdgeSupport.h`, `clus/test/doctest_steiner_edge_support.cxx` | edge-support sampling core + doctest |
| `scripts/d111s_common.py` | dump / trace parser, graph replay helpers |
| `scripts/d111s_cloud_census.py` | sec 11.3 |
| `scripts/d111s_graph_census.py` | sec 11.4 checks, sec 11.5 census |
| `scripts/d111s_painted_split.py` | sec 11.5 painted vs dead vertices |
| `scripts/d111s_spot_anatomy.py` | sec 11.6 figures + TSV |
| `scripts/d111s_replay.py` | sec 11.7 ladder |
| `scripts/d111s_wiggle_split.py` | sec 11.7 exploratory bow/jitter split |
| `scripts/d111_sbnd_gate.py` | `--ignore-branch` added (default: round-1 behaviour) |
| `figs/111s_*` | census, gates, ladder rule + sha, ladder results, spot figures and TSV |
