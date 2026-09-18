# doc pdvd/114 — where the fitted STM trajectory is not supported by the image (detours against gap jumps), why Steiner terminals form off the main path, and a default-OFF terminal admission rule that keeps gap jumping

**Owner, 2026-09-18** (after doc 113): *"In some cases the track trajectory may deviate from the 3D image, and then the
fitted dQ/dx is low because the track trajectory is off. … the deviation of the track trajectory is due to the seed
(related to the Steiner Graph and Steiner Terminal), the seed path are indeed shorter than the real path. … 1. Jump
gaps: we know that our system sometimes has inefficiencies (signal processing, noise filtering, dead channels) … we
want the algorithm to be able to jump gaps, so that the measurements in the active planes can constrain the
measurements. This is the intention. 2. … I can see some of the Steiner terminals are formed outside the main path.
Since the Steiner Graph is formed with these terminals, I can see why the code wants to choose these paths (shorter,
probably) as the shortest path. … id various cases where the best-fit track trajectory is not supported by the 3D
image. Some of them are this kind of detour, and some others are real gap jumping cases. This can be done by looking
at the 3D image as well as each of the 2D projections. … then investigate a bit the Steiner Graph/Terminal formation
rule to see why some of these Steiner Terminals deviate from the main path. This is essentially what we want to
improve. Note, we do not want to remove the capability to jump gaps. We want to refine the track trajectory."*

**Scope.**
- **Catalogue** (sec 2–3): every persisted STM fit record of the production-identical dump arms (PDHD 61 events,
  PDVD 120), each fitted row tested for support in the three wire planes and in 3-D, the unsupported stretches
  classified as gap jumps, detours or fit-born, with figures per class.
- **Mechanism** (sec 4): Phase 1 of the terminal finder replayed offline blob by blob on an extended dump, and every
  off-image terminal attributed.
- **Lever** (sec 4–6): candidate admission rules sized offline against a bar frozen before the arm-wide run; one
  default-OFF knob built (`terminal_blank_plane_mode`, toolkit), gated byte-identical OFF on PDHD, PDVD, SBND and
  uBooNE, and its two levels run and graded under a rule frozen before any knob-on arm. **Knob OFF is byte-identical.
  Knob ON is NOT bit-identical and FAILS the frozen rule. No default is flipped.**

## Status: the answer

**1. Where the trajectory is not supported by the image, and what kind of case each is** (sec 2). A fitted row is
*unsupported* when it is off the cluster's measured charge in at least one live plane (±1 wire, its own 2-D
coordinates) and more than 1 cm from the image in 3-D. Such rows are 5.7 % (PDHD) / 3.6 % (PDVD) of all fitted rows.
Grouped into stretches and classified by whether an image chain joins the stretch's anchors:

| stretches | PDHD | PDVD |
|---|---|---|
| **GAP** — the image is absent between the anchors (a real gap jump) | 970 (61 %), **90 % of the unsupported length** | 1612 (77 %), 93 % of the length |
| … of which dead planes under the stretch (`GAP-dead`) | 26 | 122 |
| … a break shorter than the stretch (`GAP-short`) | 353 | 566 |
| … live planes, empty: SP / imaging inefficiency or a detached piece (`GAP-live`) | 591 | 924 |
| **DETOUR** — the image is continuous and the trajectory left it | 504 (32 %), 7.4 % of the length | 367 (17.5 %), 5.3 % |
| … through a one-blank-plane **terminal** (`D-blank-term`, doc 112's h1 kind) | **255** | 114 |
| … through blank-plane **interiors** only (`D-blank-int`, doc 112's v3 S1/S4 kind) | 110 | 107 |
| … the tagger's crawl re-route (`D-crawl`, doc 111's h2 kind) | 104 | 99 |
| … every seed vertex three-plane (`D-3live`) | 35 | 40 |
| **FIT** — the seed was on the image, the fit left it | 83 (5 %) | 110 (5 %) |
| XID — supported, but by image the Bee layer files under another cluster id (a reference artefact) | 25 | 13 |

- By **length**, the unsupported trajectory is gap jumps: 152 m of 168 m (PDHD) and 142 m of 154 m (PDVD). These are
  the straight bridges of docs 111–113 (`figs/114_case_pdhd_GAP-live_2_frame.png`: the seed leaves the end of one
  image piece and runs 26 cm to the next). The lever below does not touch them, by design.
- By **count**, a third (PDHD) / a fifth (PDVD) of the stretches are detours, and 72 % / 60 % of those carry a
  blank-plane seed vertex (a terminal or an interior). The mean on-image alternative, where one exists in the graph,
  costs 1.5–1.9× the seed's route: the chord through the blank-plane points is the cheaper one, as at h1.
- The doc-112 spots land where expected: h1 in `D-blank-term` (3 off-image one-blank terminals on the seed, on-image
  route 1.58× dearer), v3 as one `D-blank-term` and two `D-blank-int` stretches.

**2. Why terminals form off the main path** (sec 3). Phase 1 of `create_steiner_tree` picks, in each blob, the charge
peaks among the candidates that pass `calc_charge_wcp`. Three facts, measured on 382,126 (PDHD) and 672,331 (PDVD)
dumped terminals with an offline replay that reproduces 99.3 % / 99.7 % of them:

- **The rule forgives an empty plane, and it is the prototype's own rule.** With `disable_dead_mix_cell = false` (what
  this chain passes, in WCP and in the toolkit) a plane at charge exactly 0 passes and is left out of the RMS. The
  prototype's retiler paints its path tube with charge 0 and its `Get_Wire_Charge` returns 0 for an absent wire; the
  toolkit's `1e-3` sentinel becomes the same 0. A two-plane point graded on two bright planes is a legitimate WCP
  terminal candidate. Any refinement is therefore a deliberate divergence, behind a knob.
- **Off-ridge terminals are mostly one-blank points, and three in four of those sit in a blob that ALSO holds an
  on-ridge three-plane candidate** (PDHD 72.5 %, PDVD 48.0 %; "ranking"). They win because the score is the RMS over
  the *nonzero* planes — a point with U 62701, V 1114, W 0 scores 44343 while its three-plane neighbours carry the weak
  plane in their average — and because the per-blob peak finder suppresses only *adjacent* candidates (1-hop on the
  in-blob wire lattice), so a bright ghost a few wires away survives beside the honest peak. Re-scoring alone
  (`rank3`) reaches 18 % of them; the blob-scoped eligibility rule below reaches 67 % / 47 %.
- A further 20 % / 36 % sit in a blob with no three-plane candidate but with one within 1.5 cm ("ghost blob": the
  painted halo tiled into its own blobs), and 8 % / 17 % have none within 1.5 cm ("isolated") — that last population is
  where the graph really is bridging, and it must be kept.
- The 3-live off-ridge terminals (11 % / 9 % of 3-live terminals) are on charge in all three planes by construction,
  and 89 % / 91 % of them lie within 1 cm of an image point: they are the image's width (a wide or doubled track), not
  ghosts. Every off-ridge one-blank terminal's blank plane carries the retiler's `(0, 1e12)` sentinel on PDHD (54,155
  of 54,155); on PDVD 703 of 46,327 carry uncertainty 0 (a channel absent from the activity), the first sighting of
  doc 113's residue.

**3. The lever: refine admission where a three-plane alternative exists, leave the rest alone** (sec 4–6).
- `terminal_blank_plane_mode` on `CreateSteinerGraph` (C++ default `"wcp"` = today): `prefer3` drops a blob's
  zero-plane candidates only when the blob holds a three-plane candidate; `nearby` drops one when a three-plane
  candidate of the cluster lies within `terminal_blank_plane_radius`; `prefer3+nearby` both. **A blob whose
  candidates all have a zero plane is never touched**, so inside a dead or inefficient region every point keeps its
  eligibility and the graph bridges the gap exactly as before.
- Offline sizing against a bar frozen before the arm-wide run (`figs/114_sizing_rule.txt`, sha in the doc): three
  rules qualify on both detectors; `prefer3+nearby 1.0 cm` has the smallest on-ridge cost (1.2 % / 0.7 % of on-ridge
  peaks removed without a replacement within 1.5 cm) while removing 70 % / 51 % of the off-ridge one-blank peaks and
  growing the terminal-free run of 4.1 % / 1.9 % of the catalogue's GAP stretches by more than 2 cm (bar 5 %).
  At h1 it takes the window's off-ridge terminals from 8 to 2; at v3 from 7 to 4 (the rest are interiors, sec 7).
- Knob OFF gates: compiled PR configs unchanged (PDHD `a870511c9b22`, PDVD `211a49a48229`); with the mode set the
  only differences are the two keys in the two `CreateSteinerGraph` nodes; `wcdoctest-clus` 439 / 439 (two new cases);
  the dump-extension build is identical to production on PDHD 61 / 61 and PDVD 120 / 120; the knob build's OFF arm
  against that base, SBND and uBooNE: sec 6.
- **Knob ON** (`d114?p3`, `d114?p3n10`, rule `figs/114_pred.txt` frozen before any arm, sec 6): **both levels FAIL
  the frozen rule; nothing is flipped.** The rule removes 80–92 % of the one-blank terminals, fixes h1 (seed 1.97 →
  0.68 cm, fit 1.77 → 0.47) and improves h2 (fit 2.55 → 2.11), and every gap-jumping clause passes on both detectors
  (truncation ≤ 1.9 %, far bridges flat, dead-touching tags kept, 1–5 clusters losing a fit). But the arm-wide
  off-ridge seed drops 15 % (PDHD) / 6 % (PDVD) against a 25 % bar and fit rows 7.5 % / 2 % against 20 % — the
  catalogue's arithmetic, since detours are 5–7 % of the unsupported length — v3 does not move (its stretches are
  interiors), and the tags churn at a cost on the current records (PDVD Michel purity −0.05, efficiency −0.05;
  PDHD P3 within bounds except an undecided `is_stm` efficiency).

**Recommended next step.** Two things, in this order: (a) the **interiors lever** — a charge-aware base-graph
weight before the Voronoi step (or a charge test on the path interiors), which is what v3 and the `D-blank-int`
residue need and what this round could not size offline (the base graph is not dumped; it needs its own knob and
arm); (b) an **owner look at the PDVD Michel movers** of `d114vp3` (8 new false positives, 13 new misses; none
owner-reviewed), since doc 103 showed such label sets flip on review and the tag clause is the one cost the gap
clauses do not explain. The knob stays in the tree default-OFF as the instrument for both.

**What this does not reach** (sec 7): the two-blank *interiors* (v3's S1/S4; 110 / 107 catalogue stretches) enter
the tree as Voronoi path interiors with no charge test at all — a base-graph pricing lever, not an admission one; the
far bridges (90 % of the unsupported length) are graph-level connections across image gaps; the crawl re-routes are
tagger logic.

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; D=$IMG/pdvd/docs/nf_sp_img_clus; S=$D/scripts; F=$D/figs
T=/home/xqian/toolkit-dev/toolkit; TR="WCT_STM_PATH_DEBUG=1 WCT_STEINER_GRAPH_DUMP=1"

# build 1: the dump extension only (toolkit bde7bc8a carries both halves; the dump arms ran on its first half: pin libpin_d114, clus 7f20bc6b9706; doctest 437/437)
(cd $T && ./wcb build --notests -p && ./wcb install --notests -p && ./wcb build --targets=wcdoctest-clus -p \
   && ./build/clus/wcdoctest-clus)
mkdir -p /home/xqian/tmp/d114/libpin_d114 && cp -a /nfs/data/1/xqian/toolkit-dev/local/lib/libWireCell*.so /home/xqian/tmp/d114/libpin_d114/

# sec 2-5 -- the base dump arms (production config; identical to d113?base) and their gates
P=/home/xqian/tmp/d114/libpin_d114
ARM=d114hbase DET=pdhd SRC=d108hflip JOBS=12 PIN=$P LOGD=/home/xqian/tmp/d114/arm_d114hbase TRACE_ENV="$TR" bash $S/d111_run_arms.sh
ARM=d114vbase DET=pdvd SRC=d103vflip JOBS=12 PIN=$P LOGD=/home/xqian/tmp/d114/arm_d114vbase TRACE_ENV="$TR" bash $S/d111_run_arms.sh
python3 $S/d111_identity_gate.py --det pdhd --a d113hbase --b d114hbase > $F/114_gate_base_pdhd.txt   # 61 / 61
python3 $S/d111_identity_gate.py --det pdvd --a d113vbase --b d114vbase > $F/114_gate_base_pdvd.txt   # 120 / 120
python3 $S/d113_steiner_census.py --det pdhd --arm d114hbase --logd /home/xqian/tmp/d114/arm_d114hbase --jobs 6 --out $F/114_steiner_d114hbase
python3 $S/d113_steiner_census.py --det pdvd --arm d114vbase --logd /home/xqian/tmp/d114/arm_d114vbase --jobs 6 --out $F/114_steiner_d114vbase

# sec 3 -- the catalogue (about 2 min per detector) and its figures (about 10 min)
python3 $S/d114_support_census.py --det pdhd --arm d114hbase --jobs 8 --out $F/114_support_pdhd
python3 $S/d114_support_census.py --det pdvd --arm d114vbase --jobs 8 --out $F/114_support_pdvd
python3 $S/d114_case_figs.py --stretches $F/114_support_pdhd_stretches.tsv $F/114_support_pdvd_stretches.tsv \
    --arm pdhd=d114hbase --arm pdvd=d114vbase --top 3 --out $F/114_case

# sec 4-5 -- the replay, the attribution and the sizing (the bar: figs/114_sizing_rule.txt, sha256 in 114_sizing_rule.sha256)
(cd $F && sha256sum -c 114_sizing_rule.sha256)
python3 $S/d114_terminal_rule.py --det pdhd --arm d114hbase --jobs 8 --stretches $F/114_support_pdhd_stretches.tsv --out $F/114_rule_pdhd
python3 $S/d114_terminal_rule.py --det pdvd --arm d114vbase --jobs 8 --stretches $F/114_support_pdvd_stretches.tsv --out $F/114_rule_pdvd
#   figs/114_sizing_verdict.txt: the Q1-Q4 / Q3' table (the python block in the session log; reproduced in sec 5.3)

# sec 5 -- build 2: the knob (toolkit bde7bc8a); pin libpin_d114k (clus 94d902a2bc8a); doctest 439/439; config proofs
(cd $T && ./wcb build --notests -p && ./wcb install --notests -p && ./wcb build --targets=wcdoctest-clus -p \
   && ./build/clus/wcdoctest-clus && ./build/clus/wcdoctest-clus -tc='steiner blank plane*')
mkdir -p /home/xqian/tmp/d114/libpin_d114k && cp -a /nfs/data/1/xqian/toolkit-dev/local/lib/libWireCell*.so /home/xqian/tmp/d114/libpin_d114k/
for det in pdhd pdvd; do bash $S/d102_compile_pr.sh $det d114knoboff; \
  bash $S/d102_compile_pr.sh $det d114knobp3n -S "steiner_blank_plane_mode='prefer3+nearby'" -S steiner_blank_plane_radius_cm=1.0; \
  bash $S/d102_compile_pr.sh $det d114knobp3 -S "steiner_blank_plane_mode='prefer3'"; done   # figs/114_gate_config.txt

# sec 7 -- the rule, frozen before any knob-on arm; the arms (off + p3 + p3n10 per detector, concurrent); every clause
(cd $F && sha256sum -c 114_pred.sha256)
DET=pdhd JOBS=5 LEVELS="p3 p3n10" bash $S/d114_run_levels.sh; DET=pdvd JOBS=5 LEVELS="p3 p3n10" bash $S/d114_run_levels.sh
DET=pdhd bash $S/d114_post_arms.sh; DET=pdvd bash $S/d114_post_arms.sh
python3 $S/d113_spot_figs.py --out $F/114_spot --logd-root /home/xqian/tmp/d114 \
    --arms pdhd=d114hbase:production,d114hp3:prefer3,d114hp3n10:prefer3+nearby1.0 \
    --arms pdvd=d114vbase:production,d114vp3:prefer3,d114vp3n10:prefer3+nearby1.0
python3 $S/d114_verdict.py > $F/114_verdict.txt
# the other-detector OFF gates of the knob build.  SBND needs a same-config reference: the pre-change clus library
# built from git HEAD d2646110 in a throwaway worktree, dropped into a copy of the knob pin (libpin_d114h0)
(cd $T && git worktree add --detach /home/xqian/tmp/d114/wt_head d2646110 && cd /home/xqian/tmp/d114/wt_head \
   && ./wcb configure <the flags of build/config.log> && ./wcb build --notests -p --targets=WireCellClus)
mkdir -p /home/xqian/tmp/d114/libpin_d114h0 && cp -a /home/xqian/tmp/d114/libpin_d114k/libWireCell*.so /home/xqian/tmp/d114/libpin_d114h0/ \
   && cp -a /home/xqian/tmp/d114/wt_head/build/clus/libWireCellClus.so /home/xqian/tmp/d114/libpin_d114h0/
(cd $S && TAG=d114snew PIN=/home/xqian/tmp/d114/libpin_d114k JOBS=2 bash d101_sbnd_arm.sh)
(cd $S && TAG=d114sold PIN=/home/xqian/tmp/d114/libpin_d114h0 JOBS=3 bash d101_sbnd_arm.sh)
python3 $S/d111_sbnd_gate.py --old d114sold --new d114snew --ignore-branch Trun.toolkit_git > $F/114_gate_sbnd.txt
(cd $IMG/qlport/scripts && LD_LIBRARY_PATH=/home/xqian/tmp/d114/libpin_d114k ./sweep_5384.sh d114ub 3 && ./ab_check.sh d114ub d113ub) > $F/114_gate_uboone.txt
```

All arms are new tags (M13). Their pctrees are symlinks to the production arms' (`d108hflip`, `d103vflip`). Dump logs
are in `/home/xqian/tmp/d114/arm_<arm>/` (scratch, not committed).

## 1. Inputs, definitions, and what was built to read them

**Population.** Every persisted STM fit record (`tracking-stm.root` `T_rec_charge`, one (cluster, pass)) of `d114hbase`
(PDHD 61 events, 1158 records, 385,424 rows, 2502 m) and `d114vbase` (PDVD 120 events, 1847 records, 561,346 rows,
3658 m). Both arms are production-identical (`figs/114_gate_base_*.txt`: 61 / 61 and 120 / 120 against `d113?base`,
themselves identical to `d111?st` and production).

**The dump extension** (toolkit, log-only, `WCT_STEINER_GRAPH_DUMP`). Doc 112 could not replay the per-blob peak
search because the dumped retiled points carried no blob membership. Seven fields are appended LAST to every `STGR`
line — the blob's major index, the wire index per plane, the charge uncertainty per plane — and one `STGB` line per
blob gives its slice, wire ranges and the max/min wire type and interval that `connect_graph_closely_pid` uses for the
in-blob edges. Every doc-111/112/113 parser keeps matching its prefix; `d111s_common.py` reads the new fields when
present. With the variable unset nothing runs.

**Three views of support**, per fitted row (`d114_support_census.py`):
- **2-D, per plane**: a measured charge cell of the record's cluster (`T_proj_data`, q > 0) within ±1 wire of the row's
  own 2-D coordinate (`pu/pv/pw`, `pt`) at its rounded slice — the metric-free test doc 111 sec 4.1 validated the ridge
  metric against. A plane is *dead* at the row when the nearest Steiner vertex within 1.5 cm carries that plane's dead
  bit (`test_good_point`, 0.6 cm / ch 1).
- **3-D**: the distance to the nearest image point (Bee `clustering-global`) of the record's cluster and of any
  cluster, and the canonical ridge offset (`d111_stage_attrib.Ridge`; only where the own-id image is within 3 cm).
- **The seed under the row**: the closest point of the persisted round-2 seed, the two dumped vertices of the carrying
  segment (class 3live / 1blank / 1dead / 2blank+ from the dumped per-plane charge and dead bits, role terminal /
  extreme / interior), the carrying edge's provenance (`path` / `connect` / `sbt` × `closely` / `ctpc` / `mst`), and
  whether the round-2 seed was re-routed there by `adjust_rough_path`'s crawl.

**UNSUPPORTED** = off-charge in ≥ 1 live plane AND (> 1 cm from the own ridge or > 1 cm from every image point). Rows
off a wire but within 1 cm of both are sub-cm wander (**JITTER**: 1.6 % / 4.5 % of rows), counted and not catalogued.
Rows on charge in every plane but > 1 cm off the ridge are **WIDE** (1.9 % / 1.4 %): the image's width, not a ghost.

**Stretches** = maximal runs of ≥ 2 consecutive unsupported rows, closed by the nearest supported rows before and
after (the anchors). **Classes** (thresholds fixed before the arm-wide run, `d114_support_census.py` docstring):

| class | test |
|---|---|
| GAP | no chain of image points (any cluster id, hops ≤ 2 cm — the blob-sampled image's own nearest-neighbour gaps reach 2 cm) inside a ball of radius L/2 + 5 cm joins the anchors' image |
| GAP-dead / GAP-short / GAP-live | ≥ 50 % of the rows have a dead plane / image lies within 1.5 cm of ≥ 50 % of the anchor chord (a break shorter than the stretch) / neither |
| XID | a chain joins the anchors, the chord is covered by image of ANOTHER Bee id and not by the own id (Bee files the fragments of a separated cluster under new ids: a reference artefact, the fit is supported) |
| DETOUR | a chain joins the anchors (the image is continuous) |
| D-crawl / D-bridge / D-blank-term / D-blank-int / D-sbt / D-3live | ≥ 50 % of rows on the crawl re-route / ≥ 50 % carried by a ctpc-or-mst base edge / a 1blank terminal among the seed vertices / else a blank-plane interior / else ≥ 50 % on same-blob terminal edges / else |
| FIT | the seed under ≥ 80 % of the rows is itself supported (its vertices 3live or 1dead, within 1 cm of the image) |

For DETOUR stretches the seed's own sub-route between the anchors' vertices is compared with the cheapest route on the
dumped `steiner_graph` that never leaves the ridge (doc 112's `walk_masked` / `route_stats`).

**Checks.** The census reproduces doc 111 sec 4.1: rows > 1 cm off the ridge 7.8 % (PDHD) / 5.1 % (PDVD); unsupported
among them 68.6 % / 63.2 % against 0.4 % / 0.4 % among the rest (doc 111: off-charge 70.6 % / 67.8 % against 2.8 % /
7.5 %, before the dead-plane exemption). Rows > 1 cm from every image point of any cluster: 5.7 % / 4.8 %, unsupported
87 % / 68 %.

## 2. The catalogue

`figs/114_support_{pdhd,pdvd}.txt`, `_rows.tsv.gz` (every unsupported or wide row), `_stretches.tsv` (one line per
stretch: rows, length, chord, centroid, max distances, class, the chain / coverage / dead / crawl / bridge / sbt / fit
shares, the seed vertices with class and role, and the route comparison).

**Rows.**

| | PDHD | PDVD |
|---|---|---|
| fitted rows | 385,424 (2501.9 m) | 561,346 (3657.9 m) |
| UNSUPPORTED | 21,994 (5.7 %, 174.4 m) | 20,458 (3.6 %, 166.1 m) |
| JITTER (off a wire, within 1 cm) | 6,302 (1.6 %) | 25,331 (4.5 %) |
| WIDE (on charge everywhere, > 1 cm off the ridge) | 7,197 (1.9 %) | 8,105 (1.4 %) |
| off-charge per plane U / V / W | 19,191 / 19,801 / 19,083 | 28,586 / 34,577 / 17,729 |
| dead at the row per plane | 1,189 / 1,844 / 1,341 | 7,833 / 6,860 / 4,032 |

**Stretches** (≥ 2 rows; single unsupported rows, 835 / 1,411, are not counted):

| class | PDHD stretches | rows | length (m) | blank-plane seed vertex | on-image route exists | mean on-image / seed cost | PDVD stretches | rows | length (m) | blank-plane seed vertex | on-image route exists | mean cost ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GAP-dead | 26 | 190 | 1.5 | 65 % | – | – | 122 | 558 | 4.2 | 78 % | – | – |
| GAP-short | 353 | 1,710 | 11.3 | 81 % | – | – | 566 | 2,488 | 16.1 | 82 % | – | – |
| GAP-live | 591 | 16,784 | **139.1** | 71 % | – | – | 924 | 14,288 | **122.0** | 75 % | – | – |
| XID | 25 | 182 | 1.1 | 84 % | – | – | 13 | 55 | 0.3 | 54 % | – | – |
| D-crawl | 104 | 452 | 2.8 | 90 % | 16 of 102 | 1.65 | 99 | 420 | 2.6 | 91 % | 25 of 96 | 2.15 |
| D-bridge | 0 | | | | | | 3 | 14 | 0.1 | 67 % | 1 of 3 | 1.00 |
| **D-blank-term** | **255** | 1,011 | 6.3 | 100 % | 76 of 255 | 1.73 | **114** | 392 | 2.5 | 100 % | 35 of 114 | 1.83 |
| **D-blank-int** | 110 | 412 | 2.6 | 100 % | 36 of 110 | 1.60 | 107 | 348 | 2.1 | 100 % | 26 of 107 | 1.91 |
| D-sbt | 0 | | | | | | 4 | 10 | 0.1 | 0 % | 1 of 4 | 2.51 |
| D-3live | 35 | 128 | 0.8 | 0 % | 13 of 35 | 1.52 | 40 | 141 | 0.9 | 0 % | 12 of 39 | 1.76 |
| FIT | 83 | 290 | 2.8 | 2 % | – | – | 110 | 333 | 3.0 | 1 % | – | – |
| **GAP total** | **970 (61.3 %)** | | **152.0 (90.3 %)** | | | | **1,612 (76.7 %)** | | **142.2 (92.5 %)** | | | |
| **DETOUR total** | **504 (31.9 %)** | | 12.4 (7.4 %) | | | | **367 (17.5 %)** | | 8.2 (5.3 %) | | | |
| FIT total | 83 (5.2 %) | | 2.8 (1.7 %) | | | | 110 (5.2 %) | | 3.0 (1.9 %) | | | |

Reading:
- **Gap jumps carry the length.** GAP-live alone is 139 m / 122 m: the long straight bridges between image pieces (the
  doc-111 "far" class, 93–97 % with no on-image route). They are what the owner's point 1 protects. A gap jump that is
  a straight line across a region with no charge is, by the definitions here, unsupported for its whole length; that
  is the intended behaviour, not a defect, and the catalogue lists them so the lever can be checked against them.
- **Detours carry the count**, and the blank-plane mechanism is in most of them: 72 % of PDHD's and 60 % of PDVD's
  detour stretches have a one-blank or two-blank seed vertex; the D-blank-term class alone is 255 / 114 stretches. An
  on-image route through the graph exists for only 30 % of the detours, and where it exists it costs 1.5–1.9× the
  chord — the doc-112 arithmetic, arm-wide.
- **D-3live** (35 / 40 stretches, every seed vertex three-plane) is small: the ghost is a two-plane object.
- **FIT** (5 % of stretches) reproduces doc 111's "the fit is rarely the origin".
- **PDVD's JITTER is 4.5 % of rows** against 1.6 % on PDHD — its 7.65 mm induction pitch makes ±1 wire a wider
  band, so the ridge-within-1-cm rows that are off a wire are more; the catalogue does not count them.

### 2.1 Cases, one per class (figures `figs/114_case_<name>_{2d,frame,offset}.png`, table `figs/114_case_cases.tsv`)

The figures are doc 112's per-case anatomy: the three wire planes with the cluster's charge cells (grey), the Steiner
vertices by class, the terminals (triangles; red-ringed = one-blank and off the ridge) and the seed (black, magenta
where > 1 cm off the ridge); the ridge frame (s, e2) / (s, e3) with the image, the seed and the cheapest on-ridge
route (dashed blue); the seed's ridge offset along the arc. On PDHD the U and V planes wrap, and the local affine map
that places a vertex on its wire rarely closes away from the doc-112 spots, so most PDHD panels show the W
(collection) plane only; the ridge frame is the 3-D view. Cases are the top three of each class by length × image
distance, one per cluster, not open-ended, within 30 cm of the image; five of 62 could not be drawn (`FAILED` lines in
`figs/114_case_cases.txt`: windows with no vertex to place).

- **D-blank-term — `pdhd_D-blank-term_1`, PDHD 029107_15 cl29** (`figs/114_case_pdhd_D-blank-term_1_{2d,frame}.png`).
  A 14 cm stretch, 3.4 cm off the ridge at most, through a kinked cluster. In the ridge frame the image is a V; the
  seed cuts across its inside through seven one-blank terminals (red rings) and interiors, and **no on-image route
  exists in the graph** between the stretch's anchors. In the W plane the seed rides white cells at W 7297–7300 across
  slices 1050–1062 while the charge band sits two to four wires away. This is h1's mechanism on a busier cluster: 32
  off-ridge terminals in the window, 29 of them one-blank.
- **D-blank-int — `pdhd_D-blank-int_2`, PDHD 029107_10 cl50**. The seed runs a straight 19 cm chord across a bend
  (W 7104 → 7128 over slices 1434–1443) through one-blank interiors; the image bends around it. Within 1 cm of an image
  point everywhere (the bend's two arms are both near the chord) yet 2.9 cm off the ridge and off the W charge along
  the chord: the reading that a 3-D distance alone cannot give and the wire view does.
- **GAP-live — `pdhd_GAP-live_2`, PDHD 029107_5 cl42** (`_frame.png`). The image ends; the seed continues on a
  straight `connect` edge 26 cm to the next piece. Nothing to route on: the on-image route equals the seed. The kind
  the lever must not touch.
- **GAP-short — `pdvd_GAP-short_1`, PDVD 039349_2 cl39**. A 20 cm stretch beside a sparse, broken W band with
  one-dead vertices under it and painted points around: the image is there in pieces, the chain does not close, the
  seed crosses the break 2.5 cm off. Dead planes and painted halo together.
- **D-crawl — `pdvd_D-crawl_1`, PDVD 039252_5 cl83**: four short stretches inside the tagger's re-route
  (`adjust_rough_path`), each the plain shortest path between its anchors, none with an on-image alternative. Tagger
  logic, as doc 111 found at h2.
- **The spots.** h1 (`figs/114_case_h1_*`): as in doc 112, stretch S0 through 3 one-blank terminals and 5 interiors,
  on-image route 1.58× the seed. v3: S0–S3 as in doc 112 sec 4 (S1 and S4 carried by two-blank interiors).

### 2.2 Two reference caveats

- **Bee cluster ids.** The Bee `clustering-global` layer files the fragments of a separated cluster under new ids
  (e.g. PDVD 039349_62 cl28: 92 % of its fit rows are within 1.5 cm of image with ids 28, 268, 269). The 3-D tests use
  any id; the ridge uses the own id and is undefined where that image is farther than 3 cm; XID names the residue
  (25 / 13 stretches).
- **The image is blob-sampled.** Its nearest-neighbour spacing is 0.6 cm (p50) with gaps to 2 cm, hence the 2 cm chain
  hop; a 1.5 cm hop calls sampling texture a gap (first pass of this census: 22 of 33 stretches on 029107_16 were
  "GAP" with the chord fully covered).

## 3. Why terminals form off the main path

### 3.1 The rule, read off the code (and the prototype)

- **Phase 1** (`SteinerGrapher.cxx:368` → `find_steiner_terminals` :995 → `find_peak_point_indices` :803): one blob
  at a time, every point graded by `Cluster::calc_charge_wcp(idx, cut, disable_dead_mix_cell=false)`
  (`Facade_Cluster.cxx:1031-1114`); a candidate when `charge > cut && quality`, with `cut = terminal_charge_threshold`
  (500 e on both detectors in production, 4000 in C++). Under the `false` branch a plane passes when its charge is
  above the cut **or exactly 0**, the charge is the RMS over the **nonzero** planes, and fewer than two nonzero planes
  give 0. Candidates are walked in decreasing (charge, index); the first is a peak; a candidate with a higher-charge
  **adjacent** candidate is not; adjacent peaks are merged by connected components, keeping the one nearest the
  component centroid. Adjacency is the in-blob part of `ctpc_ref_pid` — `connect_graph_closely_pid` phase 1
  (`connect_graph_closely.cxx:530-609`): two points are adjacent when their wire indices on the blob's max-type and
  min-type planes are within `max_wire_interval` / `min_wire_interval`. Out-of-blob neighbours are skipped (doc 31
  sec 12.4).
- **Phases 2–4** remove (wire-range check against the original blobs ±`terminal_wire_tol` = 1 wire, the 6 cm skeleton
  check, the 5 mm thinning) and add the extremes without a charge test.
- **The prototype does the same.** `PR3DCluster_steiner.h:1002-1004` (`if (charge_u==0) flag_charge_u = true;` ×3)
  and `:1033`; both terminal-selecting calls of `create_steiner_graph` pass `false` (`:27`, `:46`). Its retiler paints
  the path tube with charge exactly 0 (`ImprovePR3DCluster.cxx:1731-1732`), and `Get_Wire_Charge` returns 0 for an
  absent wire (`SlimMergeGeomCell.cxx:161-163`). The toolkit's `1.0e-3` sentinel becomes `(0, 1e12)` at
  `improvecluster_1.cxx:1181-1182`, the same 0 on the branch that never reads the uncertainty. `clus/docs/porting/
  porting_dictionary.md` and pr/29 sec 5.9 record `calc_charge_wcp` as a faithful port; no doc names the zero-plane
  forgiveness as a divergence. A refinement is a deliberate divergence (M15) and lives behind a knob.
- **What the retile puts there.** `hack_activity_improved` paints a disc of radius 3 (29 of 7×7 cells) in wire × slice
  around every path point whose three-point window lacks three-plane activity, in all three planes
  (`improvecluster_1.cxx:589-651`); `get_activity_improved` adds dead channels and good charge within 20 cm (`:399-423`).
  Doc 113 measured that painting makes 99 % (PDHD) / 91 % (PDVD) of the one-blank terminals.

### 3.2 The replay, and its check

`d114_terminal_rule.py` rebuilds Phase 1 per blob from the extended dump (candidacy, the wire-lattice adjacency, the
greedy peak walk, the component merge), one dump per cluster with a persisted record:

| | PDHD | PDVD |
|---|---|---|
| clusters with a record and a dump | 1,131 | 1,811 |
| candidate-bearing blobs | 463,816 | 1,059,822 |
| dumped non-extreme terminals | 382,126 | 672,331 |
| **of them replayed Phase-1 peaks** | **379,284 (99.3 %)** | **670,640 (99.7 %)** |
| replayed peaks | 607,323 | 1,173,027 |
| of them dumped terminals | 62.5 % | 57.2 % |
| peaks per candidate-bearing blob | 1.31 | 1.11 |

The replay contains the dumped terminal set; the peaks it holds beyond them are what Phases 2, 3 and the 5 mm
thinning remove. Every rule below is measured against the replay's own baseline, so the later phases do not enter.

### 3.3 Where the dumped terminals are, and why the off-ridge ones are there

Ridge offset ≤ 1 cm = on the ridge. For an off-ridge terminal: *ranking* — its blob also holds an on-ridge 3live
candidate (it lost the ordering); *ghost blob* — none in the blob, one within 1.5 cm in another blob; *isolated* — none
within 1.5 cm.

| PDHD | on ridge | off ridge | off share | ranking | ghost blob | isolated |
|---|---|---|---|---|---|---|
| 3live | 253,744 | 31,590 | 11.1 % | 11,567 (36.6 %) | 14,535 (46.0 %) | 5,488 (17.4 %) |
| **1blank** | 35,380 | **54,155** | **60.5 %** | **39,287 (72.5 %)** | 10,646 (19.7 %) | 4,222 (7.8 %) |
| 1dead | 5,351 | 1,906 | 26.3 % | 248 (13.0 %) | 254 (13.3 %) | 1,404 (73.7 %) |

| PDVD | on ridge | off ridge | off share | ranking | ghost blob | isolated |
|---|---|---|---|---|---|---|
| 3live | 437,997 | 43,928 | 9.1 % | 16,137 (36.7 %) | 20,242 (46.1 %) | 7,549 (17.2 %) |
| **1blank** | 116,242 | **46,327** | **28.5 %** | **22,238 (48.0 %)** | 16,437 (35.5 %) | 7,652 (16.5 %) |
| 1dead | 21,271 | 6,566 | 23.6 % | 588 (9.0 %) | 546 (8.3 %) | 5,432 (82.7 %) |

- **The one-blank terminals are the off-ridge population** (60 % of the class off on PDHD; 29 % on PDVD, where the
  wide induction pitch puts many blank points on the ridge too, doc 112 sec 8).
- **Ranking is the largest cause.** The blob holds an honest on-ridge three-plane candidate and the two-plane point
  outranks it: RMS over the nonzero planes rewards dropping the weak plane. h1's vertex 850 (U 62701, V 1114, W 0)
  scores 44,343; a three-plane point with the same U and V and W 3000 scores 36,240. And a bright ghost that is not
  *adjacent* to the honest peak survives beside it — the peak finder is a local-maximum test on the wire lattice, not
  a per-blob argmax (1.31 / 1.11 peaks per candidate-bearing blob). That is why re-scoring (`rank3`, sec 5) moves
  only 18 %.
- **The dead terminals are mostly isolated** (74 % / 83 %): a dead channel zeroes a whole wire, so nothing three-plane
  is nearby. That is the gap-jumping population, and it is what a refinement must leave alone.
- **The 3live off-ridge terminals are the image's width**: 28,186 of 31,590 (PDHD) and 40,123 of 43,928 (PDVD) lie
  within 1 cm of an image point. On charge in all three planes and on the image, off the charge-weighted axis — a wide
  or doubled track (doc 113 sec 7's v3 bands), not a ghost.
- **The blank plane's nature.** Off-ridge one-blank terminals: the blank plane carries the `(0, 1e12)` sentinel in
  54,155 of 54,155 (PDHD) and 45,624 of 46,327 (PDVD); **703 PDVD terminals (and 1,320 on-ridge ones) carry uncertainty
  0** — a channel absent from the activity altogether, neither painted nor dead. Doc 113 left ~9,700 PDVD one-blank
  terminals unexplained without the retile; these 2,023 are the first of that kind seen with the retile on.

## 4. Lever sizing (offline, before any build)

### 4.1 Candidate rules, replayed per blob

| rule | what changes |
|---|---|
| `wcp` | today |
| `rank3` | eligibility unchanged; the score is the RMS over the non-dead planes with a blank plane counted as 0 |
| `prefer3` | a candidate with a zero plane is eligible only in a blob that holds no three-plane candidate |
| `nearby R` | a candidate with a zero plane is dropped when a three-plane candidate of the cluster lies within R (1.0 / 1.5 / 2.0 cm) |
| `deadonly` | a zero plane passes only when it is dead (doc 112 sec 8's blunt form; the control) |
| `rank3+nearby1.5`, `prefer3+nearby1.0`, `prefer3+nearby1.5` | conjunctions |

`prefer3` and `nearby` test "a plane at charge exactly 0", dead or painted alike — what the C++ can test without a
dead-registry lookup. Inside a dead region every candidate has a zero plane, so neither rule drops anything there; at
its edge a two-plane candidate may yield to a three-plane neighbour, moving a terminal by at most a blob width.

### 4.2 The bar, frozen before the arm-wide sizing, and its three amendments

`figs/114_sizing_rule.txt` (sha256 `414f7b88…`, `figs/114_sizing_rule.sha256`). A rule qualifies on both detectors when:
Q1 it removes ≥ 50 % of the baseline's off-ridge one-blank peaks; Q2 on-ridge peaks removed without a replacement peak
within 1.5 cm are ≤ 10 % of the on-ridge peaks; Q3 the gap-jumping population is kept; Q4 the largest peak-free run per
record (p90) grows ≤ 1 cm. Smallest Q2 cost wins ties.

Three amendments are written into the file with their times and reasons:
1. the run metric counts peaks within 3 cm of the fitted polyline (from 2 cm): where the fit rides an off-image chord,
   an on-image replacement can be > 2 cm from it and would be scored as a loss;
2. `prefer3` / `nearby` test a zero plane, not a non-dead zero plane (the C++ form);
3. **after the first arm-wide sizing had run**: Q3 as written counted the peaks within 2 cm of a GAP stretch's *fitted
   rows* — rows that follow today's two-plane terminals — so an in-blob replacement a few wires away scored as a loss;
   every rule that passed Q1 failed it (PDHD −14 to −49 %, PDVD −5 to −45 %; kept in the table as reported). Q3'
   replaces it: the largest run of the anchor-to-anchor polyline with no peak within 3 cm, base vs rule, failing when
   it grows > 2 cm on more than 5 % of the GAP stretches.

The rule list was fixed after a one-event smoke test (PDHD 028084_0) showed `rank3` moves few peaks; `prefer3` was
added then, before any arm-wide number existed.

### 4.3 The result (`figs/114_rule_{pdhd,pdvd}.txt`, `figs/114_sizing_verdict.txt`)

| rule | det | Q1 off-ridge 1blank peaks | Q2 on-ridge removed, no replacement | Q3 as written | Q3' GAP stretches grown > 2 cm | Q4 p90 run (cm) | |
|---|---|---|---|---|---|---|---|
| rank3 | PDHD | −18.5 % | 0.3 % | −2.4 % | 4 of 970 (0.4 %) | 29.64 → 29.64 | FAIL |
| **prefer3** | PDHD | **−67.0 %** | 1.1 % | −15.8 % | 42 of 970 (4.3 %) | → 29.46 | PASS |
| nearby1.0 | PDHD | −3.6 % | 0.5 % | −2.4 % | 2 (0.2 %) | → 29.46 | FAIL |
| nearby1.5 | PDHD | −38.8 % | 1.1 % | −13.4 % | 8 (0.8 %) | → 28.59 | FAIL |
| nearby2.0 | PDHD | −78.5 % | 1.6 % | −26.0 % | 23 (2.4 %) | → 30.30 (+0.66) | PASS |
| deadonly | PDHD | −100 % | 2.2 % | −50.0 % | 206 (21.2 %) | → 31.49 (+1.85) | FAIL |
| rank3+nearby1.5 | PDHD | −41.0 % | 1.1 % | −13.8 % | 8 (0.8 %) | → 28.59 | FAIL |
| **prefer3+nearby1.0** | PDHD | **−70.5 %** | **1.2 %** | −18.6 % | 40 (4.1 %) | → 29.46 | **PASS** |
| prefer3+nearby1.5 | PDHD | −81.1 % | 1.5 % | −25.6 % | 45 (4.6 %) | → 28.59 | PASS |
| rank3 | PDVD | −18.8 % | 0.2 % | −0.2 % | 2 of 1612 (0.1 %) | 17.23 → 17.23 | FAIL |
| prefer3 | PDVD | −46.7 % | 0.6 % | −5.1 % | 30 (1.9 %) | → 17.24 | FAIL (Q1 by 3 points) |
| nearby1.0 | PDVD | −14.4 % | 0.4 % | −6.5 % | 3 (0.2 %) | → 17.16 | FAIL |
| nearby1.5 | PDVD | −40.9 % | 0.7 % | −14.6 % | 4 (0.2 %) | → 17.23 | FAIL |
| nearby2.0 | PDVD | −66.8 % | 1.3 % | −23.7 % | 11 (0.7 %) | → 17.61 (+0.38) | PASS |
| deadonly | PDVD | −100 % | 1.8 % | −46.2 % | 236 (14.6 %) | → 18.14 (+0.91) | FAIL |
| rank3+nearby1.5 | PDVD | −46.7 % | 0.7 % | −14.6 % | 5 (0.3 %) | → 17.29 | FAIL |
| **prefer3+nearby1.0** | PDVD | **−51.1 %** | **0.7 %** | −11.4 % | 31 (1.9 %) | → 17.29 (+0.06) | **PASS** |
| prefer3+nearby1.5 | PDVD | −65.5 % | 0.9 % | −18.8 % | 32 (2.0 %) | → 17.72 (+0.49) | PASS |

- **Qualifying on both detectors:** `nearby2.0`, `prefer3+nearby1.0`, `prefer3+nearby1.5`. **Built: `prefer3+nearby1.0`**
  (smallest Q2 on both); `prefer3` alone, the simplest form and 3 points short of Q1 on PDVD, runs beside it.
- **`deadonly` is the control that shows why a blanket rule is wrong**: it removes every one-blank peak, and with them
  the coverage of 21 % / 15 % of the GAP stretches (their largest terminal-free run grows by more than 2 cm) — doc 112
  sec 8's cost, now measured on the gap jumps themselves.
- **The owner's spots** (window terminals on / off the ridge, offline replay): h1 15 / 8 → 13 / 2 under `prefer3` and
  `prefer3+nearby1.0` (0 under `nearby2.0` and `deadonly`); v3 52 / 7 → 54 / 4 (`prefer3`), 51 / 4 (`prefer3+nearby1.0`).
  v3's remaining off-ridge points are the two-blank interiors (sec 7).

## 5. The knob

**Code** (toolkit, `clus/`):
- `WireCellClus/SteinerBlankPlane.h`: `BlankPlaneMode`, `parse_blank_plane_mode` (a typo is refused), and the pure
  `apply_blank_plane_policy(candidates, mode, nzero, near3)` on one blob's candidate set.
- `Steiner::Grapher::Config::terminal_blank_plane_mode` (`"wcp"`) and `terminal_blank_plane_radius` (0); applied in
  `find_peak_point_indices` right after the candidates are formed, before the peak search (`SteinerGrapher.cxx`).
  `nzero` counts planes at `charge_value == 0`; `near3` is a `kd_radius` query on the cluster with a per-point cache of
  "is a three-plane candidate" for the `nearby` modes. Under `"wcp"` the parse yields the no-op branch and nothing else
  executes.
- `CreateSteinerGraph`: reads the two keys (`configure`), refuses an unknown mode, round-trips them in
  `default_configuration`. `ImproveCluster_2`'s own Grapher keeps the C++ default (its terminal finder is unchanged).
- jsonnet: `cm.steiner(terminal_blank_plane_mode=null, terminal_blank_plane_radius=null)` with key suppression;
  `pdhd/pr.jsonnet` and `protodunevd/pr.jsonnet` TLAs `steiner_blank_plane_mode` / `steiner_blank_plane_radius`
  (null) threaded into both Steiner passes; the runners `pdhd/wct-pr-perevt.jsonnet`, `pdvd/wct-pr-perevt.jsonnet`
  TLAs `steiner_blank_plane_mode` / `steiner_blank_plane_radius_cm` (null). SBND, uBooNE and ICARUS bind the shared
  builder and pass nothing.
- Doctest `clus/test/doctest_steiner_blank_plane.cxx`: the component defaults, the parser, and the policy on one blob
  (wcp identity; prefer3 keeps only three-plane candidates when the blob has any and leaves an all-two-plane blob
  whole; nearby drops only where the predicate fires and never a three-plane candidate; the conjunction; ordering
  preserved; empty input). `wcdoctest-clus` 439 / 439 (2 new cases, 28 assertions).

**Gates, knob OFF**

| gate | result |
|---|---|
| G1 compiled PR config, knob unset | PDHD `a870511c9b22`, PDVD `211a49a48229`: **unchanged** (doc 109–113 values). With the mode TLA set, the only differences are `terminal_blank_plane_mode` / `terminal_blank_plane_radius` in the two `CreateSteinerGraph` nodes (`pr`, `prrefresh`); `figs/114_gate_config.txt` |
| dump-extension build (`libpin_d114`, clus `7f20bc6b9706`) vs production | `d114hbase` == `d113hbase` **61 / 61**; `d114vbase` == `d113vbase` **120 / 120** (`figs/114_gate_base_*.txt`) |
| knob build (`libpin_d114k`, clus `94d902a2bc8a`), OFF arm vs base (G2) | `d114hoff` == `d114hbase` **61 / 61**; `d114voff` == `d114vbase` **120 / 120** (`figs/114_gate_off_*.txt`), run concurrently with the knob arms |
| SBND 16 events (pr146 manifest, `d101_sbnd_arm.sh`), knob build vs a same-config reference | **BYTE-IDENTICAL, 80 files** (`figs/114_gate_sbnd.txt`). The reference `d114sold` runs the knob pin with `libWireCellClus.so` replaced by the one built from git HEAD `d2646110` in a throwaway worktree (`libpin_d114h0`, clus `e7b4bab1f9a8`): the doc-113 SBND arm is no longer a valid reference because two SBND production changes landed after it (`12798c4f` T_rec_charge provenance, `b93673ef` bundle flash group), and the first comparison against it differed on exactly those trees on every event |
| uBooNE 35 events (`sweep_5384.sh d114ub` vs `d113ub`) | **zips 35 / 35 content-identical**; tagger 34 / 35, the one difference the known bistable `kine_pio_*_2` of event 6805 (doc 90; the same lines differed in doc 113's gate) (`figs/114_gate_uboone.txt`) |
| doctest | 437 / 437 (dump build), **439 / 439** (knob build) |
| freshness | `libWireCellClus.so` 09:32:45 newer than every edited source (last 09:30:02); the test file edited at 09:35:30 changes no library code; pin md5 == installed md5 |
| no peer jobs during either build | `pgrep -c '^wire-cell'` 0 before each `wcb`; the only other wire-cell processes on the box were another user's (`/srv/data/1/jjo`) |

## 6. Knob ON: the frozen rule, the arms and the verdict

**The rule.** `figs/114_pred.txt`, sha256 `1f67b1c0…` (`figs/114_pred.sha256`), frozen 2026-09-18T09:38:19 after the
knob-off gates, the base census and the offline sizing, before any knob-on arm (launched 09:40). Doc 113's clause set
(S1 seed −25 %, P1 fit rows −20 %, P5 spot seed ≤ 1.2 cm; T tags ≥ base − 0.02; P2 / P3 / P4 / P6' / P5f fit quality;
Ga–Gd gap jumping; C, R) plus **W** (seed wiggle ≤ 1.10× base) and **G2** (the knob build's OFF arm identical to the
base). Levels: **P3** = `prefer3`, **P3N** = `prefer3+nearby 1.0 cm`. Arms per detector, concurrent at JOBS 5 on
`libpin_d114k` (clus `94d902a2bc8a`, md5 unchanged through every arm): `d114?off`, `d114?p3`, `d114?p3n10`. All
complete (PDHD 029107_17 under P3 lost its three STM candidates, the job itself finished with every output; counted by
Gd and T, doc 113 amendment-1 reading).

**Verdict: both levels FAIL** (`figs/114_verdict.txt`). Nothing is flipped; the C++ default stays `"wcp"`.

| clause | PDHD P3 | PDHD P3N | PDVD P3 | PDVD P3N |
|---|---|---|---|---|
| S1 seed > 1 cm off (bar −25 %) | 9.87 → 8.37 % (−15.2 %) **FAIL** | → 8.32 % (−15.7 %) **FAIL** | 7.82 → 7.35 % (−6.1 %) **FAIL** | → 7.39 % (−5.5 %) **FAIL** |
| P1 fit rows > 1 cm (bar −20 %) | 7.80 → 7.22 % (−7.4 %) **FAIL** | 7.76 → 7.16 % (−7.7 %) **FAIL** | 5.09 → 4.99 % (−2.0 %) **FAIL** | 5.08 → 4.97 % (−2.2 %) **FAIL** |
| P5 spot seed max (bar 1.2 cm) | h1 1.97 → **0.68** pass | 0.68 pass | v3 1.64 → 1.64 **FAIL** | 1.53 **FAIL** |
| P5f spot fit max (bar base + 0.2) | h1 1.77 → **0.47** pass | 0.47 pass | v3 0.91 → 1.14 **FAIL** | 1.14 **FAIL** |
| T tags (each metric ≥ base − 0.02) | UNDECIDED (is_stm eff −0.027 / +0.009 bounds) | **FAIL** (Michel purity −0.035) | **FAIL** (Michel purity −0.047, eff −0.057) | **FAIL** (Michel purity −0.050, eff −0.044) |
| W seed wiggle (≤ 1.10×) | 4.43 → 4.45 % pass | 4.45 pass | 9.71 → 9.51 pass | 10.05 pass |
| P2 off-charge rows (≤ base) | 8.09 → 7.89 pass | 8.03 → 7.81 pass | 10.56 → 10.55 pass | 10.52 → 10.54 **FAIL** |
| P3 Bee holes / 10 m (≤ base) | 8.57 → 7.79 pass | 8.48 → 7.87 pass | 5.90 → 5.74 pass | 5.89 → 5.76 pass |
| P4 coverage (≥ base − 1 pt) | +0.41 pass | +0.32 pass | +0.31 pass | +0.29 pass |
| P6' clean records > 1 cm (≤ 1 %) | 0.13 % pass | 0.18 % pass | 0.12 % pass | 0.14 % pass |
| Ga truncated fits (≤ 4.48 / 3.37 %) | 1.39 % pass | 1.91 % pass | 0.76 % pass | 0.76 % pass |
| Gb far > 5 cm (≤ 1.10× base) | 4.00 → 4.02 pass | 4.04 pass | 2.31 → 2.24 pass | 2.35 pass |
| Gc dead-touching tags lost vs others | 24.1 % vs 28.3 % pass | 21.7 % vs 28.3 % pass | 15.6 % vs 8.0 % pass | 16.7 % vs 24.0 % pass |
| Gd clusters losing every fit (≤ 19 / 9) | 3 pass | 5 pass | 1 pass | 2 pass |
| C, R (≤ 1.20), G2 | pass, 0.969, 61/61 | pass, 0.936, 61/61 | pass, 1.000, 120/120 | pass, 0.988, 120/120 |

**What the arms say, in order.**

1. **The rule does what it was built to do, to the terminals.** One-blank terminals fall 81,995 → 16,560 (P3) → 9,322
   (P3N) on PDHD and 119,476 → 52,996 → 22,985 on PDVD; off-ridge terminals 87,618 → 49,775 → 44,941 and 96,850 →
   77,550 → 72,823; the Steiner cloud shrinks 13–15 %; the bridges (2,413 / 6,096) are within 2 %
   (`figs/114_steiner_d114*.json`). Every gap clause passes on both detectors and both levels: truncation 0.8–1.9 %,
   far deviations flat, dead-touching tags lost no more than the others, 1–5 clusters losing every fit against
   bars of 19 / 9. **Gap jumping is intact.**
2. **h1 is fixed and h2 improves** (`figs/114_spot_h1.png`, `114_spot.tsv`): the window's one-blank terminals 5 → 0,
   seed max 1.97 → 0.68 cm, fit max 1.77 → 0.47 cm; h2's fit 2.55 → 2.11 cm (its 21 one-blank terminals → 4).
3. **The arm-wide seed and fit move a third of the way to their bars on PDHD and a quarter on PDVD.** Off-ridge seed
   length −15 % / −6 %, fit rows > 1 cm −7.5 % / −2 %. That is the catalogue's arithmetic: the detours are 7.4 % /
   5.3 % of the unsupported length and the one-blank-terminal detours about half of those; the far bridges (90 %+ of
   the length) are untouched by construction. Doc 113's retile removal, which also removed the ghosts, got −14 % / −23 %
   on S1 and −10 % / −8 % on P1 with the same shape. The catalogue re-run on the knob arms
   (`figs/114_support_{pdhd,pdvd}_{p3,p3n10}.txt`) shows where the residue sits: `D-blank-term` stretches 255 → 57 → 32
   (PDHD) and 114 → 66 → 49 (PDVD), while `D-blank-int` rises 110 → 121 → 126 and 107 → 119 → 139 — some chords that
   a terminal used to carry are now carried by the same painted points as Voronoi *interiors* (sec 7 item 1), and the
   GAP classes are unchanged (970 → 992 → 974; 1,612 → 1,584 → 1,570).
4. **v3 does not move, and its fit gets slightly worse.** Its S1 / S4 stretches are two-blank interiors (doc 112 sec 4),
   not terminals; the lever removes the window's one-blank terminals (14 → 6 → 4) and the seed stays at 1.64 / 1.53 cm.
   The fit rises 0.91 → 1.14 cm: with the ghost terminals gone the fit's Steiner-anchored association window (doc 111
   sec 2 item 6) is centred elsewhere, and the record it grades is the pass with the most rows in the window.
5. **Tags churn and, on the current truth records, the churn is a cost** (`figs/114_grade_*.txt`). PDHD P3: `is_stm`
   purity +0.007, efficiency −0.027 (UNDECIDED through the unlabelled bound), Michel purity +0.003 and efficiency
   +0.019 — the cleanest cell. PDHD P3N: Michel purity −0.035 (10 → 14 false positives) with efficiency +0.029. PDVD,
   both levels: Michel purity −0.047 / −0.050 (11 → 19 / 20 false positives) and efficiency −0.057 / −0.044 (58 → 71 /
   68 misses); `is_stm` within bounds. The candidate churn is balanced (PDHD 45–46 only-in-arm against 46–54
   only-in-base; PDVD 54–59 against 78–79), i.e. the same trajectory-change-the-taggers-were-not-tuned-on pattern as
   docs 102 r2 and 113, and none of the new PDVD false positives is owner-reviewed (doc 103 relabelled 29 of 32 such).
6. **Cost:** wall 0.94–1.00× the concurrent OFF arm; the graph is smaller.

**Reading.** The terminal-admission lever is the right instrument for the one-blank-terminal detours, and it is safe for
gap jumping under every clause written for it. It is not, on its own, a trajectory fix at the arm-wide bar, because the
unsupported trajectory is mostly bridges and, among the detours, increasingly interiors once the terminals are cleaned.
The tag cost is real on the current records and is the same kind doc 113 met.

## 7. What this lever does not reach, and what would

1. **Two-blank interiors** (`D-blank-int`, 110 / 107 stretches; v3's S1 / S4). They have charge 0 and can never be
   terminals; they enter as Voronoi shortest-path interiors on a base graph priced by geometry alone. The lever is a
   charge-aware base-graph weight before the Voronoi step, or a charge test on the path interiors (doc 111 sec 11.8
   items 1–2). The base graph is not dumped, so this cannot be sized offline; it needs its own knob and arm.
2. **The far bridges** (`GAP-live`, 90 % of the unsupported length). Graph-level connections (`ctpc` / `mst`) across
   image gaps, 93–97 % with no on-image route (doc 111). Whether each is a real track through a dead or inefficient
   region or a glued cluster is a clustering question (doc pdhd/11 R1, doc 102), not a terminal one.
3. **The crawl re-routes** (`D-crawl`, 104 / 99): `adjust_rough_path` chooses the stopping point; a guard on it moves
   tags (doc 111 sec 9).
4. **The tension with docs 31 / 37 / pdhd-01.** Those rounds fixed terminal *starvation* (wrapped planes, the 500 e
   threshold, the density) and every lever there admitted more terminals. This round removes some. The two are
   reconciled by the blob-scoped form: a blob keeps its terminal whatever the rule does inside it, so the coverage
   those rounds restored is not spent here (Q3', Q4).

## 8. Not concluded

- **The 1,320 + 703 PDVD one-blank terminals whose blank plane has uncertainty 0** (a channel absent from the
  activity): the source is not identified. Doc 113's ~9,700 residue without the retile is likely the same kind.
- **XID and the Bee ids.** The census uses any-id image for the 3-D tests and the own-id ridge; a parent map from the
  calib dump would remove the XID class and sharpen the ridge for separated clusters.
- **The GAP-short class** (353 / 566): a break shorter than the stretch, or the trajectory beside a broken image. Both
  are in it; the wire views of the cases separate them one by one, the census does not.
- **PDHD U / V wire views** of the cases: the local projection rarely closes on the wrapped planes away from the
  doc-112 spots; the W plane and the ridge frame carry those cases.
- **Five case figures** could not be drawn (windows with no placeable vertex); listed in `figs/114_case_cases.txt`.

## 9. Files

| path | what |
|---|---|
| toolkit `clus/src/SteinerGrapher.{h,cxx}` | `STGR` fields + `STGB` line (dump extension); `terminal_blank_plane_*` in `Config`; the policy call in `find_peak_point_indices` |
| toolkit `clus/inc/WireCellClus/SteinerBlankPlane.h` | the pure policy and the mode parser |
| toolkit `clus/src/CreateSteinerGraph.cxx` | the two keys |
| toolkit `clus/test/doctest_steiner_blank_plane.cxx` | 2 cases |
| toolkit `cfg/pgrapher/common/clus.jsonnet`, `cfg/pgrapher/experiment/{pdhd,protodunevd}/pr.jsonnet` | builder args and TLAs, key-suppressed |
| `pdhd/wct-pr-perevt.jsonnet`, `pdvd/wct-pr-perevt.jsonnet` | runner TLAs |
| `scripts/d111s_common.py` | reads the new `STGR` fields and `STGB` when present |
| `scripts/d112_case_anatomy.py` | `logd_for(arm)` (the dump dir by round prefix) |
| `scripts/d114_support_census.py` | sec 1–2 |
| `scripts/d114_case_figs.py` | sec 2.1 |
| `scripts/d114_terminal_rule.py` | sec 3.2–4 |
| `scripts/d114_run_levels.sh`, `d114_post_arms.sh`, `d114_verdict.py` | sec 6 |
| `figs/114_support_*`, `114_case_*`, `114_rule_*`, `114_sizing_rule.txt` (+ `.sha256`), `114_sizing_verdict.txt`, `114_steiner_*`, `114_gate_*`, `114_pred.txt` (+ `.sha256`), `114_eval*`, `114_compare_*`, `114_grade_*`, `114_spot*`, `114_verdict.txt` | results |
