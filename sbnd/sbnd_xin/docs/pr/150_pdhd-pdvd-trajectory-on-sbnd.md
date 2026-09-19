# doc pr/150 — the PDHD/PDVD trajectory on SBND: charge_stepped retile + priced Steiner seed, their impact on the neutrino pattern recognition, the fit against the 2-D measurements and the 3-D image, and a downstream retune

**Status: INVESTIGATION, no SBND production change. Toolkit `ecba69ee` (cfg only) threads the three
PDHD/PDVD Steiner seed knobs into the SBND PR job as default-OFF TLAs (byte-identical when null, sec 1). SBND production
keeps `stepped` and the C++ Steiner defaults.**

Owner request (2026-09-18): PDHD and PDVD production now run (a) the `charge_stepped` PR retile sampler
(doc pdvd/102), (b) the priced Steiner seed — `terminal_blank_plane_mode='prefer3'`,
`base_weight_blank_alpha=0.5`, `base_weight_scope='tree+path'` (docs pdvd/114–116, flipped 2026-09-18), and (c)
the two lattice fit keys `fit_weight_pow 1.5` / `assoc_cont_center 1` (doc pdvd/101). Apply them to SBND, see the
impact on the neutrino pattern recognition (vertex, nueCC, numuCC, NC π⁰), evaluate the track trajectory fit
against the 2-D measurements and the 3-D image, then adjust the other parameters to recover the performance.
**No SBND flip in this doc**: the question is whether the improvement is within reach.

Precedent: doc pr/149 measured (a) and (c) alone on the older `d102mpr` baseline and did not switch; it classed
the vertex choice and the STM verdict as NEEDS-DOWNSTREAM-RETUNE and left an owner scan of the movers undone.
This doc adds (b), re-baselines on the current production operating point (`ref/prod-2026-09-17b`), does the
retune, and does the mover scan blind.

**Answer.** The PDHD/PDVD trajectory is a better trajectory on SBND too — the PR fit closes on the 2-D
measurements and the 3-D image on every plane (W-plane rows > 1 wire off 0.97 → 0.68 %, off-charge rows −21 %,
ridge offsets −12 %, uncovered charge −0.8 points, transverse jitter −7 %), the STM seed loses 75 % of its
detour length and 80 % of its wiggle, and the drift-direction "bow" pr/149 could not move turns out to be in the
image itself (sec 3.3). But the neutrino pattern recognition built on it re-decides: on the 3000 numu events the
vertex within 3 cm of the hand click drops 677 → 620 of 823 (a blind two-arm scan of the 108 movers confirms
the direction), the STM tag reshuffles ±40 candidates symmetrically through rules that have no key, five shipped
fixes no longer hold on their sentinel events, and the π⁰ pairing loses a third of its hand-recovered pairs. A
pre-registered retune ladder on the existing keys (terminal thinning, the charge cut, the DL admission) recovers
none of it — the DL-admission lever makes the vertex worse. **The improvement is not within reach by parameter
retuning; it needs the vertex-choice stage re-tuned against the new trajectory.** Nothing is flipped.

---

## 0. Repro

From `wcp-porting-img/sbnd/sbnd_xin`, toolkit `apply-pointcloud` at `ecba69ee` (the knob threading, sec 1) or
later; the binary is HEAD `f9665bea`'s build, pinned as a whole-`local/lib` copy
(`/home/xqian/tmp/pr150/libpin`, `libWireCellClus.so` md5 `2efa7fa09325`, identical at start and end of every arm).

```bash
P=/home/xqian/tmp/pr150/libpin; D=$PWD/docs/pr/150_figs; A=scripts/analysis/pr150
# --- sec 1: knob threading proofs ---------------------------------------------------------------
bash $A/cfg_proof.sh > $D/150_cfg_proof.txt                                    # BASE=f9665bea
TAG=g16old PIN=$P SAMPLES="mcp1k mcp2k" MANIFEST=$D/../149_figs/manifest_gate16 NO_DL=1 JOBS=4 \
    CFG_TREE=<git archive f9665bea cfg>/cfg bash scripts/pr150_arm.sh          # and g16new (no CFG_TREE), g16stm (STM_FIT=1)
python3 scripts/analysis/pr149/arm_identity.py work-<s>-pr150g16old work-<s>-pr150g16new --allow Trun.cfg_tree
python3 scripts/analysis/pr149/arm_identity.py work-<s>-pr150g16new work-<s>-pr150g16stm --allow Trun.op_config_sha256
python3 $A/members_added.py work-<s>-pr150g16new work-<s>-pr150g16stm
# --- sec 2: pre-registration, cells ----------------------------------------------------------------
sha256sum $D/150_pred.txt                                                       # ee89f4b4... (20:40:55; first physics arm 20:45:40)
TAG=<cell> PIN=$P STM_FIT=1 JOBS=8 [TLA_FILE=$D/tla/<cell>.tla] [TFJSON=$D/../149_figs/149_tf_sbnd_kf.json] \
    [SAMPLES="mcp1k mcp2k"] bash scripts/pr150_arm.sh                          # cells s0 s0rep cs p3bw csp3bw tfull
bash $A/post_cell.sh <1|3> <cell>                                               # metrics, traj, jitter, vertex, movers, STM margin, sentinels
# --- sec 3: trajectory fit vs 2-D / 3-D --------------------------------------------------------------
python3 $A/pr150_traj_eval.py extract --arm pr150<cell> --samples <S...>; ... compare --a pr150s0 --b pr150<cell>
TAG=ts0 PIN=$P STM_FIT=1 JOBS=4 SAMPLES="nuecc48 ncpi0 mcp1k mcp2k" MANIFEST=/home/xqian/tmp/pr150/manifest_trace \
    TRACE_ENV="WCT_STEINER_GRAPH_DUMP=1 WCT_STM_PATH_DEBUG=1" bash scripts/pr150_arm.sh   # and tcsp3bw with TLA_FILE
python3 $A/pr150_trace_eval.py shim --arm pr150ts0 --samples ...; ... support|resid|dqdx|seed
# --- sec 4: hand scans, movers, the blind vertex scan ------------------------------------------------
python3 $A/pr150_vtx_movers.py --a pr150s0 --b pr150csp3bw --samples mcp1k mcp2k --out <movers.tsv>
python3 $A/pr150_vtx_scan.py prepare --a pr150s0 --b pr150csp3bw --movers <movers.tsv> --out <dir> --workers N
#   ... Opus workers, vtx_rules/scan_worker_prompt.md, picks-<w>.json ...
python3 $A/pr150_vtx_scan.py score --out <dir>
python3 $A/pr150_stm_margin.py --a pr150s0 --b pr150csp3bw --samples mcp1k mcp2k
# --- sec 5: the retune ------------------------------------------------------------------------------
sha256sum $D/150_pred_r2.txt
TAG=r<k> PIN=$P STM_FIT=1 JOBS=6 SAMPLES="nuecc48 ncpi0 mcp1k mcp2k" MANIFEST=/home/xqian/tmp/pr150/manifest_ladder \
    TLA_FILE=$D/tla/r<k>.tla bash scripts/pr150_arm.sh                          # k = 1..4
bash $A/post_ladder.sh r1 r2 r3 r4                                              # -> 150_ladder_score.txt
```

Every table below is committed under `150_figs/`: per-event metrics `metrics/<arm>-<sample>.tsv` and
`traj/<arm>-<sample>.tsv`, comparison printouts `150_s{1,3}_*_<cell>.txt`, mover lists `150_s*_movers_*.tsv`,
gates `150_gate_*.txt`, noise floors `150_noise_*.txt`.

---

## 1. The knob wiring (toolkit `ecba69ee`, cfg only) and its gates

SBND already threaded the pr/149 knobs (`retile_sampler_strategy` + `_charge_threshold` / `_wire_product`,
`resample_live_strategy`, `steiner_terminal_min_separation`). The three doc-114/115 seed knobs were not exposed
on SBND at all; `cm.steiner` (`cfg/pgrapher/common/clus.jsonnet:1561-1646`) accepts them key-suppressed, so
`sbnd/clus.jsonnet` `pr()` gains `steiner_blank_plane_mode` / `_radius` / `steiner_base_weight_blank_alpha` /
`_scope` (all null) passed to BOTH `CreateSteinerGraph` instances (`steiner`, `steiner_refresh`), and
`sbnd/wct-pr-perevt.jsonnet` gains the four TLAs. No C++ change; no other detector's file touched.

| proof (`150_cfg_proof.txt`, `150_gate_*.txt`) | result |
|---|---|
| PR job at the production operating point: PRE tree (`f9665bea`) vs POST null | identical, sha `fb889a64b477164f` |
| POST with explicit C++ defaults (`'wcp'` / 0 / `'tree'`), `'prefer3'` + 0.5 + `'tree+path'`, + radius | only `CreateSteinerGraph:pr` and `:prrefresh` change, the expected keys; charge_stepped on top adds only the pr/149 sampler nodes |
| LArSoft one-step chain (`per_apa()`+`pr()` in one config), sync/bare × tracking root | identical 4/4 |
| `prod_cfg_gate.py --ref ref/prod-2026-09-17b` | PASS 21/21 (before and after) |
| per-TLA plumbing (`tla_probe_added.py`): 526 PRE TLAs probed on both trees | 522 identical + 4 identical aborts modulo the moved source positions; the 4 gained TLAs each change the compiled output |
| runtime OFF gate, `manifest_gate16` (16 evt, geometric vertex), PRE cfg tree vs POST, pinned lib | 16/16 identical over 80 files and every ROOT branch (`--allow Trun.cfg_tree`) |
| STM-dump neutrality: POST vs POST + `stm_magnify` + `save_stm_fit=true` | `tracking-pr.root` every branch, calib dump, nusel identical (`--allow Trun.op_config_sha256`); `mabc-pr.zip` gains only `stm_fit-global.json`, the pctree gains only the `stm_*` point clouds (compared by datapath, `members_added.py`: PASS additive-only) |

The last line licenses every campaign arm to carry `tracking-stm.root` (`T_rec_charge` with `pass`/`status`,
`T_stm_pass`, `T_stm_eval` with `ks1`/`ratio1`, `T_proj_data`), which the production batch driver omits.

---

## 2. Cells and the impact census (`150_pred.txt`, sha `ee89f4b4…`, frozen 20:40:55; first physics arm 20:45:40)

| cell | TLAs (`150_figs/tla/`) | = |
|---|---|---|
| `s0` | none | today's production (every knob null, the production fit JSON) |
| `s0rep` | none | the noise floor on Stage 1 |
| `cs` | `retile_sampler_strategy='charge_stepped'` | (a) |
| `p3bw` | `steiner_blank_plane_mode='prefer3'`, `steiner_base_weight_blank_alpha=0.5`, `steiner_base_weight_scope='tree+path'` | (b) |
| `csp3bw` | cs + p3bw | (a)+(b) = the doc-116 trajectory, the trajectory T of the retune |
| `tfull` | csp3bw + `149_tf_sbnd_kf.json` (fit_weight_pow 1.5, assoc_cont_center 1) | (a)+(b)+(c) = PDHD/PDVD production |

Stage A is always `work-<s>-d102m`; every arm runs the production DL vertex, per-event mode, `PR_EXTRA_STAGES=
pr_display,stm_magnify`, the pin above; output `work-<s>-pr150<cell>`. Stage 1 = nuecc48 (48) + ncpi0 (19);
Stage 3 = mcp1k (1000) + mcp2k (2000). Every arm: rc 0 on every event, 0 `DL vertex failed`, pin md5 identical.

**Noise floor: zero.** `s0rep == s0` on all 67 Stage-1 events, 335 files, every ROOT branch, DL vertex included
(`150_noise_s0rep_*.txt`).

### 2.1 Stage 1 — nueCC (48) and NC π⁰ (19), each cell vs s0

Q1 on the ISO stratum (49 events; pr/149 definitions; `150_s1_compare_<cell>.txt`, `150_s1_local_jitter.txt`):

| cell | zzi_rms_dr median Δ (cm) | uncov median Δ | stubs Σ (s0 63) | vertex (40 labelled) ≤1 / ≤3 cm (s0 31 / 34); toward / away | jit_t median Δ (imp/wor, p) |
|---|---|---|---|---|---|
| cs | −0.028 | +0.003 | 79 | 29 / 34; 5 / 6 | −0.0025 (19/14, 0.49) |
| p3bw | −0.014 | −0.003 | 53 | 26 / 33; 3 / 10 | −0.0029 (19/13, 0.38) |
| csp3bw | −0.051 | −0.010 | 58 | 23 / 33; 6 / 8 | −0.0034 (20/14, 0.39) |
| tfull | −0.019 | −0.007 | 67 | 26 / 34; 5 / 6 | −0.0057 (24/11, 0.041) |

The PR fit against the 2-D measurements and the 3-D image, all 67 events, row-weighted (`150_s1_traj_<cell>.txt`;
definitions sec 3):

| cell | R2D W (s0 2.79 %) | R2D U / V (2.79 / 4.52 %) | P1 ridge > 1 cm (11.1 %) | P2 off-charge (5.05 %) | uncov (30.7 %) | q<0 rows (9.1 %) |
|---|---|---|---|---|---|---|
| cs | 2.53 (30/17, p 0.08) | 2.69 / 3.89 | 11.1 | 4.4 | 30.0 | 8.7 |
| p3bw | 2.27 (26/8, p 0.003) | 2.53 / 3.93 | 10.4 | 4.3 | 30.1 | 8.9 |
| csp3bw | **2.03 (42/6, p 1e-7)** | 2.00 / 3.28 | 10.6 | **3.82 (44/7, p 1e-7)** | 29.5 (37/19, p 0.02) | 8.1 |
| tfull | 2.16 (35/9, p 1e-4) | 1.90 / 2.89 | 10.3 (37/18, p 0.015) | 3.6 | 29.8 | 8.4 |

Q2 (`150_s1_compare_<cell>.txt`, `150_s1_sentinels_<cell>.txt`): event_label migrations 0 and nu_evaluated flips
0 in every cell (every Stage-1 event stays a neutrino candidate); TGM flips 0; STM flips 0 (p3bw 1); nue > 7.0
33 → 35 / 30 / 34 / 37, nue > 4.3 37 → 37 / 35 / 41 / 40; |ΔEnu| median 83 / 89 / 80 / 89 MeV (p90 300–420); π⁰
mass window 9 → 12 / 10 / 6 / 13; sentinels (2 applicable on Stage 1, s0 2 PASS): cs FAILs 37112 (pr/125 K3) and
69314 (pr/125 K5), p3bw 69314, csp3bw 37112, tfull 69314. Core time per event (paired medians): cs ×1.18,
p3bw ×1.00, csp3bw ×1.22, tfull ×1.20; RSS ×1.00 (the wall ratios in the printouts are load-confounded: the
cells ran under a peer's jobs at different concurrency).

Vertex movers > 10 cm on the 56 labelled Stage-1 events (`150_s1_vtxmovers_csp3bw.tsv`): 12 (6 away, 6 toward),
scanned blind in sec 4.3.

### 2.2 Stage 3 — the 3000 numu events (mcp1k + mcp2k), each cell vs s0

The impact census (`150_s3_compare_<cell>.txt`, `150_s3_vtx_<cell>.txt`, `150_s3_sentinels_<cell>.txt`; 823
vtx105-labelled events; `s0` sentinels 20 PASS / 0 FAIL / 2 SKIP):

| | cs (a) | p3bw (b) | csp3bw (a+b) | tfull (a+b+c) |
|---|---|---|---|---|
| vertex ≤ 1 cm / ≤ 3 cm of 823 (s0 616 / 677) | 530 / 625 | 547 / 630 | 540 / **620** | 541 / 609 |
| ISO (357; s0 297 at ≤ 3 cm) / control (465; s0 380) | 265 / 360 | 274 / 356 | 269 / 351 | 260 / 349 |
| > 10 cm movers away / toward; labelled events losing their vertex | 69 / 38; 21 | 56 / 32; 16 | 68 / 35; 22 | 79 / 37; 21 |
| event_label migrations (cosmic→nu / nu→cosmic) | 71 (36 / 35) | 60 (29 / 31) | 74 (34 / 40) | 72 (28 / 44) |
| nu_evaluated flips | 68 | 56 | 70 | 67 |
| STM bundle flips / FC flips / TGM | 77 / 145 / 1 | 65 / 0 / 0 | 81 / 146 / 1 | 74 / 146 / 1 |
| STM candidacy flips on common clusters (lost / gained) | 86 (43 / 43) | 78 (36 / 42) | 93 (42 / 51) | 88 (34 / 54) |
| numu > 0.9 (s0 782) | 792 | 776 | 782 | 783 |
| \|ΔEnu\| median / p90 (MeV) | 19.6 / 191 | 13.5 / 192 | 19.8 / 183 | 17.4 / 192 |
| π⁰ mass window (100,170) (s0 52) | 60 | 44 | 56 | 51 |
| sentinels FAIL (s0 0) | 5 | 8 | 5 | 9 |
| ISO stubs Σ (s0 154) | 218 | 153 | 174 | 194 |
| core time per event (paired median) / RSS | ×1.04 / 1.00 | ×1.00 / 1.00 | ×1.03 / 1.00 | ×1.01 / 1.00 |

The one TGM flip is the pr/149 scope flip (mcp2k 411886: the 12 cm bundle gains a Steiner graph under
`charge_stepped`, enters tagger scope, all three taggers −1 → 0; no evaluated TGM verdict moves). The FC 145 / 146
flips are the same Steiner-availability repair pr/149 sec 1 reported (FC 0 → 1 on small clusters that had no
graph under `stepped`); none of them changes an event label. Sentinel failures by name: cs 69314 (pr/125 K5),
171572 (pr/123 r2), 72786 (pr/128 A control), 393505 (pr/129), 292643 (pr/130 B); p3bw the same minus 69314 /
292643 plus 66366, 172794, 67026 (doc 84), 179369 (pr/130 B); csp3bw 171572, 72786, 393505, 313847 (doc 84 r2),
292643; tfull nine (the union plus 315167, 497311). Every failing sentinel is a shipped owner-approved fix whose
named event loses the fixed behaviour on the new trajectory.

**Reading against `150_pred.txt` sec 3.** The trajectory improves on every cell by (t1)–(t3) except cs on (t1)
(jit_t p 9e-6 but the vertex table shows why the trajectory alone is not the deliverable). ISO PR (a)–(d): (a)
zzi_rms_dr median Δ +0.002 / +0.006 / +0.002 / +0.004 (flat: sec 3.3 shows the bow is in the image), ratio ≤ 0
everywhere; (b) uncov ≤ 0 on cs, csp3bw, tfull; (c) vertex toward < away on every cell (**FAIL**, 33/93, 31/73,
30/87, 34/88 on the ISO stratum); (d) stubs +42 % (cs), −1 % (p3bw), +15 % (csp3bw), +26 % (tfull) → cs / csp3bw /
tfull fail (d), p3bw passes. So on SBND the doc-116 trajectory is a better trajectory that the downstream decisions
re-decide: 8.4 % fewer vertices within 3 cm of the click, a symmetric ±40 reshuffle of the neutrino candidates
through the STM tagger, and five shipped fixes that no longer hold on their named events.

---

## 3. The trajectory fit against the 2-D measurements and the 3-D image

The owner asked for the fit to be judged against the 2-D measurements and the 3-D image, not only through the
downstream decisions. Two layers, both on the SAME events in every arm.

### 3.1 The neutrino-PR fit, every event, every cell (`pr150_traj_eval.py`)

Port by duplication of the doc pdvd/111/115 instruments onto SBND's PR fit (`tracking-pr.root` `T_rec_charge`
rows of the main-cluster bundle, `T_proj_data` cells, `T_bad_ch`, the Bee clustering layer as the 3-D image, the
calib dump for the main cluster id). One SBND fact shaped it: the PR "main cluster" is a **bundle**. Its fit rows
carry `real_cluster_id = <stage cluster>·1000 + index` over every stage cluster the bundle merged (evt 175896:
13 of them; the main id alone covers 34 % of the fit's time slices, the union 100 %), so the cells and the image
are taken over the union of the bundle's stage clusters in both `T_proj_data` and the Bee layer.

| metric | definition (row = fit point) |
|---|---|
| R2D (per plane) | share of live rows whose fractional projection (`pu/pv/pw`, global channel rank, plane bases 0 / 3968 / 7936) is > 1 wire from the nearest charged cell of the bundle on the row's rounded slice (doc pdvd/115); dead channels at the row's tick (`T_bad_ch`) excluded; `R2Dpm1` over slices s−1..s+1 |
| P1 | rows > 1 cm from the image ridge (charge-weighted local axis of the 1 cm-voxelised image within 3 cm; `d111_stage_attrib.Ridge` verbatim) |
| P2 | rows off-charge in ≥ 1 live plane (no measured cell within ±1 wire on the row's slice) |
| P3 | Bee-visible holes (runs of ≥ 3 rows with q < 0) per 10 m; qneg = rows with q < 0 |
| P4 | image charge within 1.5 cm of the fit polyline / all image charge of the bundle |
| uncov | pr/149 M3 (cells predicted < 10 % of measured, main id) |

Stage 1 is in sec 2.1. Stage 3 (3000 numu events, row-weighted; `150_s3_traj_<cell>.txt`, strata from
`149_stage2_selection.tsv`):

| | cs | p3bw | csp3bw | tfull |
|---|---|---|---|---|
| R2D W (s0 0.97 %) | 0.79 (238/140, p 5e-7) | 0.84 (165/97, p 3e-5) | **0.68 (246/110, p 4e-13)** | 0.67 (276/102, p 1e-19) |
| R2D U / V (s0 1.18 / 1.57 %) | 1.04 / 1.40 | 1.04 / 1.44 | 0.86 / 1.18 | 0.82 / 1.10 |
| P1 ridge > 1 cm (s0 2.80 %) | 2.78 (p 0.7) | 2.60 (p 2e-4) | 2.47 (p 8e-6) | 2.43 (p 3e-8) |
| P2 off-charge (s0 2.25 %) | 1.97 (p 4e-14) | 2.08 (p 0.01) | 1.76 (p 4e-25) | 1.65 (p 2e-37) |
| P3 holes / 10 m (s0 0.73) | 0.76 | 0.71 | 0.72 | 0.71 |
| P4 image coverage (s0 0.846) | 0.849 | 0.849 | 0.848 (p 0.002) | 0.850 (p 0.009) |
| uncov (s0 22.5 %) | 22.2 (p 3e-7) | 22.7 (worse, p 2e-4) | **21.7 (496/269, p 2e-16)** | 22.0 (p 1e-22) |
| q < 0 rows (s0 3.07 %) | 3.02 | 3.01 | 2.84 | 2.89 |
| jit_t on the 144 long-ISO muons (s0 0.0956 cm) | −0.0048 (68/25, p 9e-6) | −0.0020 (47/33, p 0.15) | **−0.0066 (79/14, p 4e-12)** | −0.0193 (129/4, p 2e-33) |
| bow_x (s0 1.17 cm) | flat | flat | flat (p 0.5) | flat |

(1275–1285 events with a main-cluster fit in both arms; paired better / worse at ±0.005 and the sign-test p.) The
2-D closure of the fit improves on every plane and every cell, most with (a)+(b) together; the fit keys (c) remove
the transverse jitter almost entirely (129 / 4) while the drift-direction bow does not move on any cell (sec 3.3).
The Steiner knobs alone (p3bw) do not improve `uncov` on SBND; the retile does, and the two together do most.

### 3.2 The STM fit and the Steiner seed on the trace subset (arms `ts0` / `tcsp3bw`, `pr150_trace_eval.py`)

The doc pdvd/113–115 instruments need the Steiner-graph and STM-path dumps (`WCT_STEINER_GRAPH_DUMP=1
WCT_STM_PATH_DEBUG=1`, captured from each event's `stdout.log`) and the STM dump, so they ran on a 159-event
subset: the pr/149 Stage-2 `iso` stratum (150 long-ISO muons) + the 3 Stage-2 owner cases + the 5 nueCC/NCpi0
owner cases + mcp1k 57903/56463/284794/59899/58717. The pdvd scripts are untouched; the wrapper builds a shim
layout and injects the SBND plane bases (`d111_stage_attrib.BASE['sbnd'] = (0, 3968, 7936)`, the calib meta's
`base`). 98 / 99 events carry an STM fit (114 / 117 STM records); the muon dQ/dx reference is
`nusel_display/stm_ref_dqdx.json:MuonDeDx` (doc pdvd/42).

| instrument | s0 | csp3bw | reading |
|---|---|---|---|
| S1 seed off the ridge > 1 cm (`d113_steiner_census`, 114 / 117 rough walks, 259 m) | 6.77 % | 5.65 % | −17 % |
| seed > 5 cm off | 3.19 % | 3.66 % | +15 % (bridges: 161 → 176 ctpc/mst edges, 14.2 → 18.0 m) |
| seed wiggle (1.2 cm resample, > 0.5 cm) | 2.57 % | 0.50 % | **−80 %** |
| one-blank terminals / share off-ridge | 5141 / 58 % | 2506 / 75 % | **−51 %** of the class (doc 114's mechanism) |
| retiled points of the STM graphs | 0.70 M | 2.11 M | ×3.0 (the charge_stepped cloud) |
| `CreateSteinerGraph` "no_graph" warnings | 889 | 273 | −69 % (small clusters gain a graph, pr/149 sec 1) |
| unsupported stretches (`d114_support_census`, ≥ 2 rows) | 55 (6.4 m) | 50 (6.5 m) | GAP 39 → 39 (5.6 → 6.3 m, 88 → 97 % of the length) |
| DETOUR stretches / length | 15 / 0.8 m (12.2 %) | 9 / 0.2 m (2.6 %) | **−40 % / −75 %**; D-blank-term 10 → 0, D-blank-int 1 → 3, D-crawl 3 → 5 |
| FIT / XID | 1 / 0 | 1 / 1 | — |
| STM-fit R2D W (`d115_proj_resid`, 38 801 / 38 532 live rows, dead bits from the dump) | 2.14 % | 1.79 % | −16 %; U 1.36 → 1.27, V 1.72 → 1.41 |
| rows OFF in ≥ 1 live plane (±1 slice, > 1.5 wire ≈ P2) | 3.10 % | 2.81 % | −9 % |
| rows with q < 0 | 2.02 % | 1.70 % | −16 % |
| dQ/dx (`d115_dqdx_compare`, accepted STM tracks: 4 paired) | shape rms 0.074, k 0.985 | 0.071, 1.040 | too few accepted stoppers in an ISO-muon subset to read; reported only |

Every seed-side clause of the doc-115 round-2 rule that is arm-relative (U1 DETOUR length −30 %, U2 blank-carried
detours −40 %, S1 ≤ base, W wiggle ≤ 1.10 × base, R2D ≤ base, P2 ≤ base) passes on SBND with the doc-116 knobs;
the far-deviation share (Gb, ≤ 1.10 × base) does not (3.19 → 3.66 %, the added ctpc/mst bridge length on a denser
cloud). The GAP class — 88–97 % of the unsupported length — is untouched by design (gap jumping is kept).

### 3.3 The drift-x bow: physical (`pr150_bow.py`, `150_trace_bow.txt`)

pr/149 sec 13.8 left open whether the ~1.2 cm drift-x bow that dominates its zig-zag metric on long ISO muons is
in the data or made by the fit. On the same 146 ISO events (358 m of ISO segments), per calib fit point matched
to its `T_rec_charge` row, the fit's time slice `pt` is compared with the image's own charge-weighted mean slice on
the row's W wire (bundle cells within ±8 slices):

| | s0 | csp3bw |
|---|---|---|
| fit bow (rms of pt about the chord after a 9-point mean), p50 / p90 (slices; 1 slice ≈ 0.32 cm) | 3.80 / 12.5 | 3.88 / 12.9 |
| image bow (the same on the image's own time), p50 / p90 | 3.78 / 12.5 | 3.85 / 12.4 |
| residual bow (fit − image, smoothed), p50 / p90 | 0.31 / 0.91 | 0.33 / 0.91 |
| residual rms (unsmoothed) p50 | 0.57 | 0.58 |

Median image-bow / fit-bow = **1.00**, residual-bow / fit-bow = **0.08**; the image follows the fit's bow on 136
of 146 events and is straight on 0. **The bow is in the image, not in the fit** (space charge and/or the drift
calibration of the image; pr/149's zig-zag metric measures the detector, and zzi should be read about a smooth
curve). The fit tracks the image's time centroid to 0.57 slices ≈ 0.2 cm rms, identically in both arms.


---

## 4. Hand-scan-anchored metrics, the regression study, the blind vertex scan

### 4.1 What the hand scans can and cannot score on a re-clustered arm (`scripts/analysis/pr150/{pi0_*,pid_*,campaign_ab.sh,geo_rescore.py}`)

Three SBND label sets were repointed at the `pr150<cell>` arms before any number was read:

- **Population census** (`pr_scores_table.py` + `pr142_campaign_ab.py`, wrapper `campaign_ab.sh`): works on any
  arm; it keys on (sample, run, subrun, event) so the cross-sample event-id collision (18255-1-69314) is handled.
- **π⁰ hand scan** (`pr132_pi0_census.py`, 66 hand π⁰ / 132 γ; wrapper `pi0_score.sh`, manifests
  `pi0_manifests.py`): resolves a hand γ through `showers[].id` = the start segment's `pf_node_id`, a per-event
  reconstruction index. `pi0_id_drift.py`: between `pr150s0` and `d102mpr` the shower-id sets are identical on 128
  of 135 events (Jaccard 1.00); between `pr150s0` and `pr150csp3bw` on 1 of 71 (Jaccard **0.00**). Every
  trajectory cell renumbers every id, so all reachable γ read `absent-on-arm` — a property of the scorer, not a
  physics result. Even on `s0` only 23–39 of 132 γ resolve, because the labels are anchored to an arm epoch
  older than `d102mpr` (doc pr/141 §3 documented 9 slots; it is now most of them). Also: the current production
  `kine_shower_fudge_factor` is 0.86 (doc pr/135 quotes 0.84), and `pr132_gamma_ledger.py` has no `--fudge`, so
  its OK band is not comparable to doc pr/135's 90.9 %. No committed doc reports the census on `d102mpr`; the run
  here (7 exact / 14 partial / 25 none / 20 no-group of 66) is a new reference point.
- **pr/148 PID hand scan** (`pr148_score_scan.py` / `pr148_scan_combined.py`; wrapper `pid_scan_score.sh` +
  `pid_agreement.py`): the A5 census regenerates from any arm's logs (311 of 313 rows byte-identical to the
  committed census on `d102mpr`; `scan_combined` byte-identical: precision 0.444, Wilson [0.25, 0.66], doc pr/148
  §12.1 exactly). But the scorers key on (event, shower_id) with no sample and index unguarded: on `csp3bw` 13 of
  72 labelled objects match on (sample, event, shower_id) and **0** of them are the same physical object
  (`start_seg == obj` and length within 2 %). The unforked scorer would have scored 13 wrong objects and printed a
  plausible precision. `s0`: 48 of 72 objects survive, pass-1 precision 3/8.
- **Position-anchored re-scoring** (`geo_rescore.py`, a NEW denominator, not the doc pr/135 / pr/148 numbers):
`geo_rescore.py` matches each hand γ (its stored `reco_start`, 132/132) to the arm's `showers[]` by start
  position (R = 5 cm = the 95th percentile of the known-correct distance, median 0.02 cm) with the axis angle < 90°
  (30° would discard 18 % of known-correct pairs because 90 of 132 label axes are probe axes, not the reco's); where
  the id join still resolves, geometry and id agree 36/36 (d102mpr) and 29/29 (s0). Full arms, the 65 events every
  cell reaches (130 γ):

  | | d102mpr | s0 | cs | p3bw | csp3bw | tfull |
  |---|---|---|---|---|---|---|
  | γ matched by position | 110 | 110 | 84 | 91 | 87 | 90 |
  | π⁰ with both γ matched | 50 | 50 | 34 | 37 | 34 | 39 |
  | π⁰ recovered (both γ in one reco π⁰ group) | 28 | 28 | 21 | 22 | **18** | 20 |
  | pr/148 PID objects matched (36) / re-typed | 36 / 18 | 36 / 8 | 15 / 1 | 21 / 2 | 19 / 1 | 19 / 2 |

  `s0 == d102mpr` on every count (the EM samples did not move between the epochs). Every trajectory cell matches
  fewer hand γ at their labelled start (−20 to −26 of 110) and recovers fewer hand π⁰ pairs (28 → 18–22): the EM
  shower objects the labels sit on are re-segmented (a start that moves > 5 cm reads as unmatched, which the
  matcher cannot tell from a lost shower), and the π⁰ pairing loses a third of its recovered pairs on csp3bw. The
  id-anchored census would have reported all of this as "absent-on-arm" — and has been reporting ~70 % of γ so on
  `d102mpr` itself (76 of the 93 "absent" γ have a reco shower at the labelled position), a defect of the census
  since before this doc (sec 6).

### 4.2 The STM verdict margin (`pr150_stm_margin.py`; pr/149 sec 10 item 1)

`TaggerCheckSTM`'s evaluation (`clus/src/TaggerCheckSTM.cxx:2971-3003`) accepts a candidate's dQ/dx window
iff (g) `ratio2 ≤ 2.0` (accept_guards, SBND true), (a) `ks1 − ks2 < 0`, (b) not (`r < 1.4` and `s > −0.02`) with
`r = √((ks2/0.06)² + ((ratio2−1)/0.06)²)`, `s = ks1 − ks2 + (|ratio1−1| − |ratio2−1|)/1.5·0.3`, then the
residual-length / Michel-residual rules. `margin_shape = min(ks2 − ks1, max((r − 1.4)·0.06, −0.02 − s))`:
positive = the shape test passes by that much (ks units). Since every arm carries `tracking-stm.root`, the margin
is known for EVERY evaluated candidate (`T_stm_eval`, the attempt with verdict 1 else the largest margin), and the
candidacy is `T_stm_pass.status` (0 accepted; 3 = the shape test failed on every attempt; 2 / 4 / 5 / 7 / 8 =
guards, doc pdvd/107). A verdict-1 attempt under a nonzero status is a guard rejection after the shape passed
(evt 389538 cl 11: attempt 2 verdict 1, status 4 mid-point track).

On the 3000 numu events (`150_s3_stm_csp3bw.txt`): 1183 evaluated candidates in s0, 1209 in csp3bw, 1179 common;
deciding-pass status in s0: accepted 450, eval-failed 333, left-charge 221, pre-fit-exit 118, mid-point 34, proton
27. The shape margin is broad (p50 +0.057, p10 −0.025 ks units) and moves by 0.011 (p50) / 0.044 (p90) between the
arms on verdict-stable clusters. **Every one of the 148 verdict flips and 93 candidacy flips passes the shape test
in the losing arm too** (0 flips at the KS / ratio boundary, within or beyond the noise): the flips are the
residual-length / Michel-residual rules (hard-coded constants) and the guards — accepted ↔ eval-failed 19 / 18,
pre-fit-exit → accepted 16 vs 9, proton 11 / 10, left-charge 6 / 2 — and they are symmetric (lost 42 / gained 51).
The hysteresis pr/149 sec 10 item 1 asked about would act on none of them. This is the opposite of doc pdvd/107's
PDHD finding (there the churn was the eval boundary); on SBND the STM churn is a re-decision by the later rules on a
changed trajectory, with no key, and no net bias (numu > 0.9 782 → 782). Class: TRADE, reported.

**Vertex movers** (`pr150_vtx_choice.py`, `pr150_vtx_stub.py`; `150_s3_vtxchoice_csp3bw.txt`, `150_s3_stub_csp3bw.txt`):
of the 70 "away" movers, 30 are CHOICE (s0's vertex is still a PR candidate in csp3bw, another was chosen), 20
STRUCTURE (the candidate set changed; on 14 the click position is no longer a candidate at all), 20 LOST-CANDIDATE
(the bundle newly STM-tagged); of the 38 "toward", 25 CHOICE, 10 STRUCTURE, 2 lost, 1 gained. The pr/149 sec 7.2
stub-branch mechanism is not dominant under (a)+(b): the new main vertex carries a < 5 cm degree-1 stub on 16 of 50
away movers vs 13 of 50 in s0 (gained 7, lost 4) and on 161 of 715 non-movers — prefer3 + base-weight removes most
of the stubs `charge_stepped` alone adds (ISO stubs +42 % → +15 %). The pr/149 adjudication classes on the 350
Q2 movers of the mcp1k half: vertex-choice 105, energy-scale 51, main-cluster 30, pr-structure 13.

### 4.3 The blind two-arm vertex scan (`pr150_vtx_scan.py`; the pr/149 sec 10 item 4 owner scan, done by agents)

Every vtx105-labelled event whose main vertex moves > 10 cm between s0 and csp3bw (both directions) is rendered
TWICE with the doc pr/80 kit (`vtx_rules/scankit.py prepare`: p1 overview, p2 3-D, p4 dQ/dx, p5 cone, p6
evidence table; structurally blind — `sanitize()` strips `main_vertex`, `vertex_scoreboard`, `dirsign`, `rr`,
`showers`), once from each arm's calib dump, under an opaque item id, shuffled (`random.Random(150)`), the
arm hidden; the item → (event, arm) key is written only to `KEY.json`, which the scanners are never told about,
and the dump they zoom with is a copy at a neutral path. Calibration items (10 % of the movers, labelled
non-movers, s0 rendering) are mixed in. Scanners: Opus agents with `vtx_rules/scan_worker_prompt.md`, picks
written once at the end; their transcripts are grepped for `KEY.json`, `manifest.json`, `main_vertex` and
`vertex_scoreboard` reads (`150_s*_vtxscan_audit.txt`: none). A pick is resolved against that ITEM's PR-vertex
pool and measured against the vtx105 click, the arm's own main vertex and the other arm's main vertex:

| class | meaning |
|---|---|
| CONFIRMS-LABEL | the picks on both arms land within 1 cm of the click |
| A0-WAS-WRONG | an "away" mover whose picks on BOTH renderings land on B's main vertex and not on the click |
| REGRESSION | an "away" mover whose s0 pick confirms the click (or s0's vertex) and whose B pick does not land on B's vertex; `REGRESSION-b-agrees` when the B pick does follow B's vertex |
| IMPROVEMENT-CONFIRMED | a "toward" mover whose B pick lands on the click and whose s0 pick does not |
| UNRESOLVED | the rest (abstentions, inconsistent picks) |

**Stage-1 pilot** (12 movers × 2 arms + 1 calibration, 3 workers; `150_s1_vtxscan_score.txt`): calibration
1/1 agrees; classes CONFIRMS-LABEL 1 (46363: csp3bw moved 41 cm away from a click the scanner confirms on both
renderings = a real regression), A0-WAS-WRONG 1 (389538: the 213 cm "away" mover — the scanner picks csp3bw's
vertex on both renderings, 213 cm from the click), IMPROVEMENT-CONFIRMED 2 (180801 77 → 0.4 cm, 163543 33 →
0.1 cm), REGRESSION 1 (521075) + REGRESSION-b-agrees 1 (111412), UNRESOLVED 6. On 9 of the 25 items the pick
coincides with that arm's own main vertex to < 0.05 cm: the rendering hides the answer but shows the arm's
segment structure, so the scanner leans toward each arm's choice; the two-arm design is what makes that visible
(the class asks whether the SAME scanner, shown both, follows the click or the arm).

**Stage 3, both waves** (mcp1k movers 51 → 95 items + 5 calibration, 7 workers; mcp2k movers 57 → 110 items + 6
calibration, 8 workers; `150_s3{a,b}_vtxscan_score.txt`, `150_s3_vtxscan_crosstab.txt`, audits
`150_s3{a,b}_vtxscan_audit.txt`: no hidden-key or answer reads; labels `vertex_labels/vtxscan-pr150-{s3a,s3b}`).
Calibration 9 of 11 agree with the vtx105 click at 1 cm. Of the 108 movers, 20 have no csp3bw vertex at all (the
lost candidates of sec 4.2) and were shown on s0 only. Where the SAME scanner sees the csp3bw rendering:

| direction | picks the click | picks csp3bw's own vertex | neither | abstain |
|---|---|---|---|---|
| away (50 shown on both) | **16** | 9 | 24 | 1 |
| toward (37) | 7 | **10** | 19 | 1 |

On the s0 rendering of the same events the scanner lands on the click 22 / 50 (away) and 14 / 37 (toward). The
scanner is decisive on about half of these (by selection hard) events, and where it is, it sides with the click on
the "away" movers (16 : 9) and with csp3bw's choice on the "toward" movers (10 : 7): the label direction is
supported in both classes, no "away" mover qualifies as A0-WAS-WRONG (a pick on csp3bw's vertex from BOTH
renderings against the click), and 12 "away" movers are CONFIRMS-LABEL (the click found on both renderings —
csp3bw had the right candidate and chose another). The corrected vertex score therefore equals the raw one; the
scan does not soften the −57 at ≤ 3 cm.


---

## 5. The retune

`150_pred_r2.txt`, sha256 `1b63e464d3c327ac…`, frozen 2026-09-18 23:33:50; the first ladder arm started 23:34:16.
Trajectory T = `csp3bw`. From the mechanism table (sec 4.2), the ladder uses EXISTING keys only, one set of ≤ 3
keys, coarse values (compile proof per level in `150_cell_cfg.txt`: each level changes exactly the named keys):

| level | keys on top of csp3bw | acts on |
|---|---|---|
| R1 | `steiner_terminal_min_separation = 0.7` (cm) | the +15 % ISO stubs and the STRUCTURE class (pr/149 amendment 1) |
| R2 | R1 + `retile_sampler_charge_threshold = 6000` | the cloud density (the pr/149 cell that recovered the two Stage-1 EM sentinels) |
| R3 | R1 + `dl_vtx_min_accept_score = 4.0` | the CHOICE class: the pre-2026-08-15 DL admission |
| R4 | R2 + `dl_vtx_min_accept_score = 4.0` | both |

Arms `work-<s>-pr150r{1..4}` on the 626-event ladder manifest (Stage 1 + the pr/149 Stage-2 manifest; 489
vtx105-labelled events; s0 and csp3bw read from the full arms on the same events), JOBS 6, rc 0 everywhere, pin
unchanged. Rule (sec 3 of the file): a) vertex ≤ 3 cm ≥ s0 − 1 % of the labelled AND > 10 cm away ≤ toward + 5;
b) nu_evaluated lost ≤ gained + 5, TGM 0; c) sentinel FAIL ≤ s0 + 1; d) numu > 0.9 within ±3 %, |ΔEnu| median
≤ 25 MeV; e) R2D_W, P2, uncov ≤ s0; f) rc 0, DL failed 0, core ≤ 1.5 ×, RSS ≤ 1.25 ×; split-half guard.
Prediction, written before the arms: no level reaches a). (`150_ladder_score.txt`, `150_lad_sentinels_*.txt`)

| level | vertex ≤ 3 cm of 489 (s0 406; bar 401) | > 10 cm away / toward | nu_evaluated lost / gained | numu > 0.9 (s0 388) | \|ΔEnu\| median | sentinels FAIL (s0 0 on these 10) | R2D_W (s0 1.34 %) | P2 (2.65 %) | uncov (23.0 %) | core | clauses a b c d e f |
|---|---|---|---|---|---|---|---|---|---|---|---|
| csp3bw (T) | 370 | 37 / 24 | 16 / 0 | 385 | 38.6 | 0 | 0.97 | 2.02 | 22.5 | 1.05 | F F P F P P |
| R1 | 364 | 51 / 21 | 18 / 0 | 374 | 39.3 | 2 (393505, 172794) | 1.00 | 2.08 | 23.3 | 1.06 | F F F F F P |
| **R2** | **372** | 45 / 22 | 15 / 0 | 384 | 32.4 | 2 (37112, 393505) | 1.03 | 2.16 | 23.0 | 1.04 | F F F F P P |
| R3 | 334 | 83 / 24 | 18 / 0 | 358 | 40.9 | 2 | 1.00 | 2.09 | 23.7 | 1.06 | F F F F F P |
| R4 | 329 | 85 / 24 | 15 / 0 | 368 | 38.7 | 2 | 1.04 | 2.19 | 23.5 | 1.04 | F F F F F P |

**No level passes; the fewest-failing is R2 (4 of 6 clauses).** Selected: NONE; nothing is recommended for
production, and the confirmation arm on the full 3000 was not run (it is defined for a selected level only).
Readings:
- The vertex is not recovered by any existing key. R1 (terminal thinning, the stub lever) and R2 (thinning +
  a denser charge cut) leave the ≤ 3 cm count where T put it (364–372 vs 406); the DL-admission lever (R3 / R4)
  makes it markedly worse (83–85 away movers vs 37): admitting more DL candidates into the re-rank on the new
  trajectory changes the CHOICE class in the wrong direction. The "away" movers are real (sec 4.3), so the
  bar was the right one.
- Clause b on this manifest is conditioned (pr/149 sec 5.2: the Stage-2 events all have a candidate in the
  epoch reference, so candidates can only be lost); the unconditioned full census (sec 2.2) is what it should be
  read from — symmetric there (40 lost / 34 gained on csp3bw), which misses the +5 bar by 1.
- The two sentinel failures per level sit on the ladder events (393505 pr/129 on every level; 172794 doc 84 r4
  on the R1/R3 side; 37112 pr/125 K3 on the charge-6000 side); csp3bw itself fails none of the 10 sentinels
  present on these events (its five failures are elsewhere in the 3000).
- |ΔEnu| on this ISO-heavy manifest is 32–41 MeV against 20 MeV on the full sample; the 25 MeV bar was set on the
  full-sample number and is missed by every level here, R2 least.
- The trajectory clauses hold on T and R2 (R2D_W −25 %, P2 −20 %, uncov ≤ s0) and are lost by R1 / R3 / R4 through
  `uncov` (the thinned terminals cost image coverage on the long-ISO muons).

**Verdict of the retune: not within reach with the existing keys.** The cost of the doc-116 trajectory on SBND
is a re-decision of the neutrino vertex (and, symmetrically, of the STM tag) by stages tuned on the old fit,
and the keys those stages expose move the wrong things: terminal density (which is not the mechanism once
prefer3 + base-weight are on) and the DL admission (which amplifies the choice churn). Recovering it means
re-tuning the vertex-choice stage itself against the new trajectory (the DL re-rank features, `vertex_z_prior_scale`,
the candidate scoring in `TaggerCheckNeutrino`) with the 108-mover scan set of sec 4.3 as the truth, which is a
round of its own, not a knob.

---

## 6. What this doc did not do; defects found, reported, not fixed

**Not done here.**
- No simulation leg (the owner's scope was data; no truth-level trajectory residual). No Bee upload. No flip.
- The clustering job's 3-D sampler stays `stepped` (pr/149 round 2 showed the whole-cloud resample is not the lever).
- The blind vertex scan was done by agents, not the owner; the items where the scanner's pick lands on neither the
  click nor either arm's vertex (sec 4.3) are exactly the ones an owner scan would settle.
- The ladder used existing keys only. The one mechanism with no key that the study found — the STM later-rule /
  guard re-decisions — is symmetric on SBND and would not be "recovered" by a threshold; no C++ knob was built.
- The dQ/dx clauses (D1–D3) could not be read on SBND: the ISO trace subset holds 4 paired accepted stoppers.

**Defects found, reported, not fixed.**
1. **The id-anchored π⁰ hand-scan census reads renumbering as reconstruction failure.** `pr132_pi0_census.py`
   resolves a hand γ through `showers[].id` (the start segment's `pf_node_id`), a per-event index. On the current
   `d102mpr` reference it resolves 39 of 132 γ and reports the other 93 as `absent-on-arm`; a position match finds a
   reco shower for 76 of those 93 (`geo_rescore.py`). The labels are anchored to the `*-prod0825` / `work-pr131-
   denom*` arm generation, all released. Any census number on an arm newer than that generation is a numbering
   artefact until the labels are re-anchored (by position, as here, or by a re-scan); the doc pr/141 §3 note (9
   slots) understated it.
2. **`pr148_score_scan.py` / `pr148_scan_combined.py` key on (event, shower_id) with no sample and index
   unguarded**: on a re-clustered arm they either `KeyError` or, where the id happens to exist, score a different
   object (csp3bw: 13 id matches, 0 the same physical object). `pid_agreement.py` adds the identity probe.
3. **`pr132_gamma_ledger.py` has no `--fudge`**: its 0.80–1.25 OK band assumes the scan-time EM scale 0.80 while
   production runs 0.86, so its number is not comparable to doc pr/135's 90.9 %.
4. **`SEL.load_manifest()` aborts on a released arm directory** in the manifest's `dump` column; absolute dump paths
   bypass the guard (`pi0_manifests.py`).
5. **`tla_probe_gate.py` refuses a tree that gains TLAs** (its purpose is a same-surface regression gate);
   `tla_probe_added.py` probes the shared surface and the gained TLAs separately, masking the jsonnet source
   positions and log timestamps in identical aborts.
6. **`arm_identity.py` reports "members differ" on an archive that only gained members**; the pctree must be
   compared by datapath (`members_added.py`), never by member name (tensor indices renumber when point clouds are
   added).
7. **`d115_proj_resid.py` / `d114_support_census.py` assume every `tracking-stm.root` holds `T_proj_data`**; SBND
   writes the file for events with no STM candidate (no such tree), so the shim skips them (60 of 159 here).
8. **`vertex_tolerance.py` raises a numpy warning on the empty "kept" set** (harmless; the printout is complete).
9. The `pr149_metrics.py compare` wall-time ratio is load-confounded when cells run at different concurrency; the
   paired `core_s` median (from `pr_scores_table.py`) is the comparable number and is what sec 2 quotes.


---

## 7. Files

- Toolkit `ecba69ee` (cfg only): `cfg/pgrapher/experiment/sbnd/clus.jsonnet` (pr() args + both `cm.steiner` call
  sites), `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` (the four TLAs). No C++ change.
- Launcher `scripts/pr150_arm.sh` (fork of `pr149_arm.sh`: `STM_FIT`, `TRACE_ENV`, placeholder guard).
- Analysis `scripts/analysis/pr150/`: `cfg_proof.sh`, `tla_probe_added.py`, `members_added.py` (sec 1);
  `pr150_metrics.py`, `vertex_tolerance.py`, `q1_verdict.py`, `r2_local_jitter.py`, `r2_zzi_sign.py` (pr/149 forks,
  150_figs paths), `post_cell.sh` (sec 2); `pr150_traj_eval.py`, `pr150_trace_eval.py` (shim + runner for the
  pdvd instruments), `pr150_bow.py` (sec 3); `pr150_stm_margin.py`, `pr150_vtx_movers.py`, `pr150_vtx_choice.py`,
  `pr150_vtx_stub.py`, `pr150_vtx_scan.py`, `pi0_manifests.py`, `pi0_score.sh`, `pi0_id_drift.py`,
  `pid_scan_score.sh`, `pid_agreement.py`, `campaign_ab.sh`, `geo_rescore.py` (sec 4); `pr150_ladder.py`,
  `post_ladder.sh` (sec 5).
- Records `docs/pr/150_figs/`: `150_pred.txt` + `.sha256`, `150_pred_r2.txt` + `.sha256`, `tla/*.tla`,
  `150_cfg_proof.txt`, `150_cell_cfg.txt`, `150_gate_off_*.txt`, `150_gate_stm_*.txt`, `150_noise_s0rep_*.txt`,
  `metrics/*.tsv`, `traj/*.tsv`, `150_s1_*` / `150_s3_*` printouts, mover lists, `150_trace_*.txt`,
  `150_case_*.png`, `150_s1_vtxscan_*`, `150_s3a_vtxscan_*`, `150_s3b_vtxscan_*`, `150_lad_sentinels_*.txt`,
  `150_ladder_score.txt`, `150_preflight.txt`.
- Vertex-scan labels of this round (agent picks, a NEW tag, never merged into vtx105):
  `vertex_labels/vtxscan-pr150-{s1,s3a,s3b}/` (the scored.json of each wave with the item → event/arm key).
- Arms on disk (releasable once this doc is accepted; `docs/work-tags.md`): `work-<s>-pr150{g16old,g16new,g16stm}`
  (16 evt), `work-{nuecc48,ncpi0}-pr150{s0,s0rep,cs,p3bw,csp3bw,tfull}`, `work-{mcp1k,mcp2k}-pr150{s0,cs,p3bw,
  csp3bw,tfull}`, `work-<s>-pr150{ts0,tcsp3bw}` (159 evt), `work-<s>-pr150r{1,2,3,4}` (626 evt).
- Scratch (not committed): `/home/xqian/tmp/pr150/` (pin `libpin`, logs, `tr/` shim + traces, `vtxscan_*/`
  renderings, `scorers*/`, `cfg/`).

