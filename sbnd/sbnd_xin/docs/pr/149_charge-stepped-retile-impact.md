# doc pr/149 — switching the SBND PR retile sampler `stepped` → `charge_stepped`: what it does to isochronous PR and to everything else

**Status: INVESTIGATION, no production change.** Toolkit `f573cdeb` (pushed) adds two config knobs, both default
OFF and byte-identical when off (§3):
- `retile_sampler_strategy` (with `_wire_product` / `_charge_threshold`);
- `steiner_terminal_min_separation`.

SBND production keeps `stepped`.

**Answer to the owner's two questions.**
1. **Isochronous PR does not improve.** Trajectories on long ISO muons do not get straighter, and the
   neutrino vertex on ISO events gets worse.
2. **The degradations are broad, and the dominant one is not recoverable with a sampler or fit parameter.**
   It is refit instability of the downstream decisions (STM verdict, vertex choice) — the same class doc
   pdvd/107 found on PDHD. The fit keys PDHD/PDVD also adopted cause it on their own, on mostly different events.

**Recommendation: do not switch SBND.** §10 lists what would have to come first.

**Round 2 (§13, same day; toolkit `e73850ad`, default OFF): switching the whole cloud does not help either.** The owner asked whether ISO
failed to improve because the track fit still associates pixels through the clustering job's `stepped`
"3d" cloud. A new default-OFF stage (`resample_live_strategy`, toolkit `ClusteringResampleLive`) re-samples
every cluster's own cloud with `charge_stepped` inside the PR job, as the prototype's PR executables do.
- The pre-registered primary endpoint fails: long-ISO muon zig-zag does not fall (whole cloud + retile
  59 improved / 71 worsened; whole cloud alone 55 / 76).
- The premise does not hold. On ~4 500 ISO fit points the main association is the same in every arm
  (15 cells per plane, centroid within 0.19 wire of the solved point).
- Post hoc, the retile *does* reduce the local jitter across the wires (83 / 7 with the resample). But the
  zig-zag metric measures drift x about one chord, which on long ISO muons is a ~1.2 cm smooth bow that no
  sampler or fit key moves.

Owner request (2026-09-16): PDHD and PDVD retiled with `stepped` instead of the prototype's `charge_stepped`;
SBND (`sbnd_xin`) does too. Investigate the switch on SBND, where much of the pattern recognition (neutrino
vertex, taggers) builds on the track fit:
1. do isochronous (ISO) cases improve?
2. what degrades elsewhere, and is it recoverable?

Owner scoping answers:
- the PR retile only, so the clustering 3-D cloud stays `stepped`;
- data only;
- nueCC and NCpi0 first, then numuCC events with large ISO segments;
- ask before any Bee upload.

---

## 0. Repro

From `wcp-porting-img/sbnd/sbnd_xin`, toolkit `apply-pointcloud` at `f573cdeb` or later. Pin: a full copy of
`local/lib` (790 files, `libWireCellClus.so` md5 `091e142b9481`), identical at start and end
(`libpin_md5_{start,end}.txt` compared equal).

```bash
P=/home/xqian/tmp/pr149/libpin; D=$PWD/docs/pr/149_figs
cp -a /nfs/data/1/xqian/toolkit-dev/local/lib $P

# --- Phase 0: is charge_stepped's all-wire branch reachable on SBND? (no arms) ---------------
python3 scripts/analysis/pr149/blob_product_census.py --samples nuecc48 ncpi0 --tsv $D/149_p0_blob_product_nue_ncpi0.tsv
python3 scripts/analysis/pr149/blob_product_census.py --samples mcp1k mcp2k   --tsv $D/149_p0_blob_product_numu.tsv

# --- Phase 1: knob proofs ------------------------------------------------------------------------
bash scripts/analysis/pr149/cfg_proof.sh > $D/149_cfg_proof.txt                     # BASE=9d8fd892
TAG=goffold PIN=$P SAMPLES="mcp1k mcp2k" MANIFEST=$D/manifest_gate16 NO_DL=1 JOBS=4 \
    CFG_TREE=<git archive 9d8fd892 cfg>/cfg bash scripts/pr149_arm.sh
TAG=goffnew PIN=$P SAMPLES="mcp1k mcp2k" MANIFEST=$D/manifest_gate16 NO_DL=1 JOBS=4 bash scripts/pr149_arm.sh
python3 scripts/analysis/pr149/arm_identity.py work-mcp1k-pr149goffold work-mcp1k-pr149goffnew --allow Trun.cfg_tree

# --- pre-registration --------------------------------------------------------------------------
sha256sum $D/149_pred.txt $D/149_pred_amend1.txt      # 47e40845... (05:46:08), 05ce9174... (06:20:29)

# --- arms (stage A always work-<s>-d102m; output work-<s>-pr149<TAG>) ----------------------------
#   TLA files: $D/tla/{cs,csq2000,csq6000,cstsep05,cstsep07}.tla ; fit JSON: $D/149_tf_sbnd_kf.json
# Stage 1 (nuecc48 + ncpi0, 67 events):
TAG=s0 PIN=$P JOBS=8 bash scripts/pr149_arm.sh                        # likewise s0rep
TAG=cs PIN=$P JOBS=8 TLA_FILE=$D/tla/cs.tla bash scripts/pr149_arm.sh  # likewise csq2000, csq6000, cstsep05, cstsep07
TAG=kf PIN=$P JOBS=6 TFJSON=$D/149_tf_sbnd_kf.json bash scripts/pr149_arm.sh
TAG=cskf PIN=$P JOBS=6 TFJSON=$D/149_tf_sbnd_kf.json TLA_FILE=$D/tla/cs.tla bash scripts/pr149_arm.sh
TAG=s0g PIN=$P JOBS=4 NO_DL=1 bash scripts/pr149_arm.sh; TAG=csg PIN=$P JOBS=4 NO_DL=1 TLA_FILE=$D/tla/cs.tla bash scripts/pr149_arm.sh
# Stage 2 (numuCC large-ISO manifest, 559 events):
python3 scripts/analysis/pr149/pr149_metrics.py extract --arm d102mpr --samples nuecc48 ncpi0 mcp1k mcp2k
python3 scripts/analysis/pr149/select_iso_numu.py
TAG=s2<cell> PIN=$P JOBS=6 SAMPLES="mcp1k mcp2k" MANIFEST=$D/manifest_stage2 [TLA_FILE|TFJSON as above] bash scripts/pr149_arm.sh
#   cells: s0 s0rep cs kf cskf csq2000 csq6000 cstsep05 cstsep07
# Stage 3 (all 3000 mcp1k + mcp2k; one driver per sample):
TAG=s3a<cell> PIN=$P JOBS=4 SAMPLES=mcp1k [...] bash scripts/pr149_arm.sh   # and SAMPLES=mcp2k; cells s0 cs kf csq2000

# --- analysis ------------------------------------------------------------------------------------
python3 scripts/analysis/pr149/pr149_metrics.py extract --arm pr149<TAG> --samples <samples>
python3 scripts/analysis/pr149/pr149_metrics.py compare --a pr149s0   --b pr149cs   --samples nuecc48 ncpi0 --movers $D/149_s1_movers_cs.tsv
python3 scripts/analysis/pr149/pr149_metrics.py compare --a pr149s2s0 --b pr149s2cs --samples mcp1k mcp2k \
        --manifest-strata $D/149_stage2_selection.tsv --movers $D/149_s2_movers_s2cs.tsv
python3 scripts/analysis/pr149/pr149_metrics.py compare --a pr149s3as0 --b pr149s3acs --samples mcp1k mcp2k --movers $D/149_s3_movers_s3acs.tsv
python3 scripts/analysis/pr149/vertex_tolerance.py --stage {1,2,3} --base <s0 arm> --arms <arms>
python3 scripts/analysis/pr149/q1_verdict.py --cells cs:s2cs csq2000:s2csq2000 csq6000:s2csq6000 kf:s2kf cskf:s2cskf
python3 scripts/analysis/pr149/tiebreak.py   cs:s2cs csq2000:s2csq2000 csq6000:s2csq6000 kf:s2kf cskf:s2cskf
python3 scripts/analysis/pr149/adjudicate.py --a pr149s0 --b pr149cs --movers $D/149_s1_movers_cs.tsv --tsv $D/149_s1_adjudication_cs.tsv
python3 scripts/analysis/pr149/terminal_spacing.py work-nuecc48-pr149s0 work-mcp1k-pr149s2s0 work-nuecc48-pr149cs work-mcp1k-pr149s2cs
python3 scripts/analysis/pr149/sentinels_tolerant.py --arms work-mcp1k-pr149s3acs work-mcp2k-pr149s3acs work-nuecc48-pr149cs work-ncpi0-pr149cs
python3 scripts/analysis/pr149/arm_identity.py work-mcp2k-pr149s2s0 work-mcp2k-pr149s2s0rep          # noise floor
python3 scripts/bee/make_pr_bee.py -q work-{nuecc48,ncpi0,mcp1k,mcp2k}-d102m -p <s0 arms> --allow-unevaluated \
        -o bee/pr149/pr149-s0-stepped.zip $(cut -f3 bee/pr149/pr149.index.txt | grep -v '^#')  # and -p <cs arms>
```

Round 2 (§13), toolkit with `ClusteringResampleLive` (built in an isolated worktree of `06fd9e22` plus the
round-2 files only, private prefix; pin `libWireCellClus.so` md5 `5f41937c3318`, all 25 pinned files identical
at start and end):

```bash
P=/home/xqian/tmp/pr149r2/libpin; D=$PWD/docs/pr/149_figs
BASE=06fd9e22 bash scripts/analysis/pr149/cfg_proof_r2.sh > $D/149_r2_cfg_proof.txt
./build/clus/wcdoctest-clus                                          # in the round-2 build: 423/423
# gates (manifest_gate16, geometric vertex); CFG_TREE = git archive 06fd9e22 cfg
TAG=r2goffold PIN=$P SAMPLES="mcp1k mcp2k" MANIFEST=$D/manifest_gate16 NO_DL=1 JOBS=4 CFG_TREE=<pre>/cfg bash scripts/pr149_arm.sh
TAG=r2goffnew PIN=$P SAMPLES="mcp1k mcp2k" MANIFEST=$D/manifest_gate16 NO_DL=1 JOBS=4 bash scripts/pr149_arm.sh
TAG=r2gidst   ... TLA_FILE=$D/tla/rsst.tla   # stepped resample = identity gate
TAG=r2gidcs   ... TLA_FILE=$D/tla/rs.tla     # charge_stepped resample = RESAMPLE census
python3 scripts/analysis/pr149/arm_identity.py work-<s>-pr149r2goffold work-<s>-pr149r2goffnew --allow Trun.cfg_tree
python3 scripts/analysis/pr149/arm_identity.py work-<s>-pr149r2goffnew work-<s>-pr149r2gidst --allow Trun.op_config_sha256
sha256sum $D/149_pred_r2.txt                                         # e486f9ad... (10:54:46)
# arms: Stage 1 TAG=r2<cell> (67 evt, JOBS=4); Stage 2 TAG=s2r2<cell> (manifest_stage2, JOBS=6);
#       trace TAG=r2t<cell> (manifest_r2trace, JOBS=2, WCT_TRAJ_ASSOC_DEBUG=1); cells s0 / cs (tla/cs.tla) /
#       rs (tla/rs.tla) / rscs (tla/rscs.tla)
python3 scripts/analysis/pr149/r2_zzi_sign.py --stage 2 --pairs pr149s2r2rscs:pr149s2r2s0 pr149s2r2rs:pr149s2r2s0 \
        pr149s2r2rscs:pr149s2r2cs pr149s2r2cs:pr149s2r2s0 pr149s2r2rscs:pr149s2r2rs      # and --stage 1 with pr149r2*
python3 scripts/analysis/pr149/r2_assoc_trace.py --arms pr149r2ts0 pr149r2tcs pr149r2trs pr149r2trscs --events-manifest $D/manifest_r2trace
python3 scripts/analysis/pr149/r2_local_jitter.py --stage 2 --pairs pr149s2r2cs:pr149s2r2s0 pr149s2r2rs:pr149s2r2s0 pr149s2r2rscs:pr149s2r2s0 pr149s2kf:pr149s2s0
python3 scripts/analysis/pr149/r2_topology.py --base pr149s2r2s0 --arms pr149s2r2cs pr149s2r2rs pr149s2r2rscs --samples mcp1k mcp2k
python3 scripts/analysis/pr149/q1_verdict.py --base1 pr149r2s0 --base2 pr149s2r2s0 --cells r2cs:s2r2cs r2rs:s2r2rs r2rscs:s2r2rscs
python3 scripts/analysis/pr149/vertex_tolerance.py --stage 2 --base pr149s2r2s0 --arms pr149s2r2cs pr149s2r2rs pr149s2r2rscs
python3 scripts/analysis/pr149/sentinels_tolerant.py --arms work-mcp1k-pr149s2r2<cell> work-mcp2k-pr149s2r2<cell> work-nuecc48-pr149r2<cell> work-ncpi0-pr149r2<cell>
```
The analysis commands above run in one pass as `scripts/analysis/pr149/r2_secondary.sh`.

All tables quoted below are committed under `149_figs/`:
- per-event metrics: `metrics/<arm>-<sample>.tsv`;
- comparison printouts: `149_s{1,2,3}_compare_*.txt`;
- mover lists: `149_s*_movers_*.tsv`;
- gates and noise floors: `149_gate_off_*.txt`, `149_noise_*.txt`.

Aborted and superseded: `work-mcp1k-pr149s3{s0,cs,kf}` (≈ 80 events each). They were stopped on purpose,
before any number was read, and relaunched sharded as `s3a*`. They are not used anywhere.

---

## 1. Answers

### Q1 — do isochronous cases improve? **No.**

- **The mechanism is live.**
  - SBND's blobs essentially never exceed the 2500 wire-product cut: ≥ 99.0 % are under it in every
    drift-angle bin, ISO included, on all four samples. So charge_stepped's all-wire branch runs everywhere.
  - The main cluster's Steiner cloud grows ×1.8–2.2 (Stage 3 numu ISO median +563 points).
  - The 4000 e charge cut, not the product cut, bounds the added points (§2).
- **Trajectory smoothness improves only where showers are.**
  - nueCC/NCpi0 ISO: zig-zag rms median Δ −0.071 **cm** (round 2 corrected the unit: the calib dump's
    length unit is cm, and round 1 labelled these numbers mm).
  - Long ISO **muon** tracks, full numu sample: Δ +0.003 cm; path/chord −0.006.
  - ~~The likely reason, read in the code: TrackFitting's main association block reads the clustering job's
    `stepped` cloud, which this switch does not touch.~~ **Tested and refuted in round 2 (§13):** re-sampling
    that cloud too leaves the association unchanged and the metric flat. What does move is the local
    jitter across the wires. The metric is dominated by a smooth drift-direction bow that neither cloud
    reaches (§13.6).
- **The vertex gets worse, and worse on ISO than elsewhere.** Truth is the vtx105 hand clicks. The full numu
  sample (Stage 3, 3000 events):

  | stratum (d102mpr rule) | labelled | ≤ 3 cm s0 → cs | vertex > 10 cm away / toward | candidate lost |
  |---|---|---|---|---|
  | ISO (≥ 10 cm at θ_drift ≥ 75°) | 357 | 297 → 265 (−10.8 %) | 35 / 13 | 13 |
  | control (no such segment) | 465 | 378 → 360 (−4.8 %) | 34 / 27 | 7 |
  | all labelled | 823 | 675 → 625 (−7.4 %) | 69 / 40 | 21 (s0: 3) |

  A >10 cm move is a different vertex choice, not a refit shift, so it is immune to the clicks being picks
  on `stepped`'s own vertices (§5.3). On the large-ISO subset (Stage 2) the ISO-strong stratum lost
  82 → 64 of 89 at ≤ 3 cm, with 12 events moved away vs 1 toward.
- **Pre-registered Q1 rule** (`149_pred.txt` sec 4), pooled over the 455 Stage 1 + 2 ISO events:
  **no cell passes**. cs passes 1 of 4, csq2000 / csq6000 / kf 2 of 4, cskf 1 of 4. The vertex criterion (c)
  fails in every cell at every stage.
- Owner ISO cases:
  - trajectories straighten on the five nueCC/NCpi0 cases;
  - 350186 loses its vertex (0.0 → 40.8 cm) and 21073 too (1.0 → 10.5);
  - mcp1k 57903 (pr/73's zig-zag case) gets *more* zig-zag and a 13 cm vertex shift;
  - 284794 loses its vertex by 164 cm.

### Q2 — what degrades, and is it recoverable?

cs vs s0 on the full samples (3000 numu from Stage 3 + 67 nueCC/NCpi0 from Stage 1):

| degradation | size | mechanism | recoverable? |
|---|---|---|---|
| vertex accuracy | ≤ 3 cm −50 of 675; > 10 cm away 69 vs toward 40 | extra short branches at the vertex (terminals ×1.38) plus refit-unstable vertex choice | **NEEDS-DOWNSTREAM-RETUNE.** No sampler or fit cell recovers it: csq2000 623, kf 613 at ≤ 3 cm. The fit keys alone do the same damage on mostly different events |
| candidate churn | nu_evaluated 34 gained / 34 lost; event_label 71 (36 / 35) | TaggerCheckSTM's verdict sits on its decision boundary: STM 39 0→1, 37 1→0 (doc pdvd/107 class) | **NEEDS-DOWNSTREAM-RETUNE** (the STM evaluation margin); kf alone: 27 / 30 |
| numu > 0.9 working point | 78 gained / 67 lost (net +11), 4.8 % of events move | the same churn | not a bias; not recoverable by a parameter |
| shipped-fix sentinels | 21 PASS / 0 FAIL → **15 / 6** | fixes tuned on `stepped` output: pr/123, pr/125 K3 / K5, pr/128, pr/129, pr/130 | per fix. csq6000 recovers the two EM ones on Stage 1; kf 7 FAIL, csq2000 8 |
| end stubs | ISO 154 → 218 (Stage 3) | nlevel-hop terminal suppression shrinks on a denser cloud (§7.2) | **RECOVERED** by terminal thinning 0.7 cm (amendment 1, exploratory), which does *not* recover the vertex or the churn |
| Enu | \|ΔEnu\| median 20 MeV, p90 192 MeV (numu); no scale shift | PF regrouping | kf alone gives 12 / 184 |
| resources | wall median +0.7 % (numu), +16 % (nueCC); RSS +0.1 % | 2× Steiner cloud | within the doc-102 bar |

**One systematic change is a repair, not a cost.**
- In mcp1k, "no steiner_graph" warnings drop from 5537 to 1903: under `stepped`, many small clusters get too
  few points to build a Steiner graph.
- `cluster_fc_check` then returns its conservative `is_fc=false` (`Clustering_Util.cxx:81-89`), and the
  taggers skip them.
- Under cs, 143 in-beam bundles flip FC 0→1 (on 142 events), and one 12 cm bundle (mcp2k 411886) enters tagger scope at all.
  **None of the 142 events changes its event_label or nu_evaluated** (`149_s3_movers_s3acs.tsv`).
- That one is the only "TGM flip" (−1 → 0: not evaluated → evaluated). No evaluated TGM verdict moved in any
  arm, so the TGM negative control holds.

**Noise floor: zero.**
- s0rep == s0 on all 67 Stage-1 events (335 files) and all 559 Stage-2 events (2795 files), DL vertex included.
- s3as0 == s2s0 on the 214 shared mcp1k events (1070 files), at a different job concurrency.

---

## 2. What the switch changes on SBND (Phase 0, before any arm)

`clus/src/BlobSampler.cxx`:

| | `stepped` (`:782-1031`) | `charge_stepped` (`:1048-1396`) |
|---|---|---|
| wires per view | every `max(3, N/12)` + last | **every wire** when `N_max·N_min ≤ 2500` (`:1200`), else the stepped set |
| charge | not used | a non-stepped wire is dropped if its charge is nonzero and < 4000 (`:1291`, `:1313`); a non-must pair is dropped if any of the three wire charges is nonzero and < 4000 (`:1342-1357`) |
| retile setting | — | `disable_mix_dead_cell: false`, as the prototype retile (`ImprovePR3DCluster.cxx:59`) |

Production stage-A point-cloud trees (`work-<s>-d102m`), with imaging blobs as a proxy for the retile's blobs:

| sample (events) | θ_drift | clusters | blobs | frac N_max·N_min ≤ 2500 | frac stepped points with a nonzero wire charge < 4000 |
|---|---|---|---|---|---|
| nuecc48 + ncpi0 (67) | 0–30° | 70 | 28 494 | 1.000 | 0.728 |
| | 75–85° | 175 | 85 515 | 1.000 | 0.673 |
| | 85–90° | 137 | 77 895 | 0.991 | 0.719 |
| mcp1k + mcp2k (3000) | 0–30° | 3 311 | 1 473 930 | 1.000 | 0.727 |
| | 75–85° | 7 604 | 2 760 295 | 0.999 | 0.696 |
| | 85–90° | 6 096 | 2 875 568 | 0.991 | 0.734 |

(`149_p0_blob_product_{nue_ncpi0,numu}.tsv`)

- **SBND's 3 mm pitch is the prototype's own (uBooNE) pitch**, where the 2500 cut was tuned, and on SBND it
  essentially never binds.
- The planning review had worried that the product cut might make charge_stepped inert on wide ISO blobs.
  That was measured and refuted here, so the wire-product sweep was dropped and only the charge threshold was
  swept (2000 / 6000).
- Caveat: ImproveCluster_2 re-tiles each cluster from its own 2-D activity. The direct measurement is the
  Steiner point count in the arms.

**Who reads the retiled cloud.** Only `steiner` and `steiner_refresh` use `improve2`
(`sbnd/clus.jsonnet:2133,2150`). They build `steiner_pc` / `steiner_graph`, which are read by:
- TaggerCheckSTM;
- TaggerCheckFC (via `cluster_fc_check`);
- the whole neutrino PR: `find_proto_vertex`, `init_first_segment`, `examine_structure_1/2`,
  `find_other_segments`, and the `MyFCN` vertex fit;
- TrackFitting's second association block (`TrackFitting.cxx:3249-3340`).

Not reached directly:
- TGM, `basic_pid`, TrackFitting's main association block and `associate_points` read the clustering job's
  `stepped` "3d" cloud.
- The DL vertex net reads fitted PR points and dQ (`NeutrinoVertexFinder.cxx:4779-4803`), so the sampler
  reaches it only through the refit.

---

## 3. The knobs and their proofs (toolkit `f573cdeb`)

`cfg/pgrapher/experiment/sbnd/{clus,wct-pr-perevt}.jsonnet` gain:
- `retile_sampler_strategy` (null ⇒ `'stepped'`);
- `retile_sampler_wire_product` / `retile_sampler_charge_threshold` (null ⇒ C++ 2500 / 4000, read only under
  `charge_stepped`);
- `steiner_terminal_min_separation` (cm, 0 ⇒ key omitted; both CreateSteinerGraph instances).

`bs_live_face` follows the PDHD pattern, with **its own component name `live-cs-<apa>-<face>` for the
charge_stepped variant**. The LArSoft one-step chain compiles `per_apa()` and `pr()` into one config, and
`ConfigManager::add` lets the last same-named `BlobSampler` win. A same-named variant would have silently
moved stage-A imaging there.

| proof (`149_cfg_proof.txt`, `149_gate_off_*.txt`) | result |
|---|---|
| `prod_cfg_gate.py --ref ref/prod-2026-09-14`, before and after | PASS 21/21, both |
| PR job: pre-knob tree (9d8fd892) vs null / `'stepped'` / separation 0 | identical, sha `8b6ae33855553dd3` |
| `charge_stepped` node diff | only `BlobSampler live-apa{0,1}-0` → `live-cs-…` and `ImproveCluster_2:pr.samplers`; 0 other nodes |
| + separation 0.5 | + `CreateSteinerGraph:{pr,prrefresh}.terminal_min_separation = 5`; 0 other nodes |
| misspelled strategy | compile aborts (assert) |
| one-step LArSoft chain, sync/bare × tracking root on/off | identical 4/4 |
| runtime OFF gate, 16 events (d101 gate list, geometric vertex), knob tree vs pre-knob overlay, same pin | 16/16 identical over 80 files and every ROOT branch; only `Trun.cfg_tree` (the overlay path) differs |

Compiled-config proof inside the arms: each event's `.wct-cfg-evt*.json` of the cs arm carries
`live-cs-apa{0,1}-0` with `charge_stepped`; the s0 arm keeps `['stepped']`; cstsep07 carries
`terminal_min_separation = 7` on both Steiner stages. No C++ change.

---

## 4. Pre-registration

`149_figs/149_pred.txt`, sha256 `47e408459641cfd0…`, frozen 05:46:08; the first physics arm started 05:46:19.
It fixes:
- the arms;
- the ISO strata, drawn from the **epoch-reference** arms `work-<s>-d102mpr`, never from a measured arm;
- the Stage 2 manifest rule;
- Q1 criteria (a) trajectory, (b) coverage, (c) vertex, (d) stubs;
- the Q2 census, the recoverability classes, the Stage-3 cell rule, and the stop-and-report list.

`149_pred_amend1.txt` (sha `05ce9174785c64f2…`, 06:20:29) is **post hoc and exploratory**. It was written after
Stage 1 and part of Stage 2 exposed the terminal mechanism. It adds the terminal-separation lever and cannot
change the verdict.

Three deviations, all recorded:
1. **Two post-hoc attribution arms:** `s0g`/`csg` (geometric vertex).
2. **The Stage-3 cell set is s0, cs, kf and csq2000.** The literal sec-6 rule picks kf (the 2/4 tie broken by
   fewest degradations, `149_tiebreak.txt`: kf 123, csq2000 142, csq6000 143, cs 146, cskf 153). kf is not a
   charge_stepped cell, so the best charge_stepped cell (csq2000) was added.
3. **The tie-break count's term list** was written after the Q1 table was read. It is printed term by term.

Lineage: s0 reproduces `d102mpr` on Enu for all 67 Stage-1 events.

---

## 5. Metrics and their traps

### 5.1 Definitions (`pr149_metrics.py`)

| metric | definition |
|---|---|
| zig-zag `rms_dr`, `ratio`, `fold` | doc pr/73's definitions (in **cm**, the calib dump's length unit; `rms_dr` here is the drift-x component of the residual about the segment chord) on the calib dump's fitted points (the same `Segment::fits()` as the Bee `track_fit` layer), length-weighted over main-cluster segments ≥ 10 cm with ≥ 10 points |
| `uncov` | fraction of the main cluster's measured charge in cells predicted at < 10 % of the measured, from `tracking-pr.root` **`T_proj_data`** (per-cluster `get_cluster_fitted_charge_2d()`) |
| stubs | main-cluster segments < 3 cm with a degree-1 end vertex |
| vertex | 3-D distance of `main_vertex` to the vtx105 rank-1 click; a missing vertex counts as a failure |

On `uncov`: **not** the calib dump's `proj` block. That is the merged, last-writer-wins map, in which a
satellite's fit overwrites the main cluster's prediction (`PrDisplayDump.cxx:1163-1180`).

### 5.2 Strata and conditioning

- **Stage 1 / 3:** ISO = main cluster in d102mpr has ≥ 10 cm of segments at θ_drift ≥ 75° (showers included);
  ISO-strong = at ≥ 85°; control = none.
- **Stage 2** (`149_stage2_selection.tsv`, 559 events):
  - `iso`: top 150 by non-shower length at ≥ 80°, each ≥ 111.6 cm;
  - `vtx_iso`: 256 labelled events with ≥ 10 cm at ≥ 75°;
  - 3 owner cases;
  - `control`: 150 drawn with `default_rng(149)`.
- **Stage 2 is conditioned.** Its pool is "events with a calib dump in d102mpr", i.e. events that already had
  a candidate. Candidates can only be lost there, and STM flips only go 0→1 (Stage 2: 16 of 17). Stage 3
  removes the conditioning and shows the churn is two-sided (§8).

### 5.3 Vertex truth is anchored to `stepped`

The vtx105 clicks are rank-1 picks among PR vertex candidates of the scanned arm. 189 of 823 labelled numu
clicks lie within 0.05 cm of s0's own vertex. Any refit shifts a correct vertex a few mm (median 0.19 cm), so
the ≤ 1 cm count penalises every refit. The doc therefore leads with >10 cm moves, a different vertex choice,
and quotes ≤ 3 cm next to them. The anchoring is common to all strata and all arms.

---

## 6. Stage 1 — nueCC (48) and NCpi0 (19)

Arms `pr149{s0,s0rep,cs,kf,cskf,csq2000,csq6000}`, all rc=0; `149_s1_compare_*.txt`.

### 6.1 Q1 on the ISO stratum (49 events), vs s0

| cell | Steiner pts Δ | terminals Δ | zig-zag rms Δ (cm) | path/chord Δ | uncov Δ | stubs | ≤ 3 cm (40) | > 10 cm away / toward |
|---|---|---|---|---|---|---|---|---|
| s0rep | 0 | 0 | 0 | 0 | 0 | 63 → 63 | 34 → 34 | 0 / 0 |
| cs | +3183 | +173 | −0.071 | +0.005 | +0.003 | 63 → 79 | 34 → 34 | 4 / 4 |
| csq2000 | +3707 | | −0.026 | −0.007 | −0.005 | 63 → 78 | 34 → 29 | 3 / 3 |
| csq6000 | +2397 | | −0.005 | −0.007 | −0.009 | 63 → 61 | 34 → 27 | 7 / 4 |
| kf | 0 | 0 | −0.016 | −0.010 | +0.005 | 63 → 58 | 34 → 34 | 1 / 4 |
| cskf | +3183 | | −0.017 | −0.006 | −0.006 | 63 → 79 | 34 → 31 | 4 / 4 |

All 56 labelled events at ≤ 3 cm (`149_s1_vertex_tolerance.txt`): s0 48, cs 45, csq2000 41, csq6000 42,
kf 48, cskf 42.

### 6.2 Q2

| | s0 | cs | csq2000 | csq6000 | kf | cskf |
|---|---|---|---|---|---|---|
| event_label / nu_evaluated flips | — | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| nuecc48 nue > 7.0 | 33 | 34 | 32 | 34 | 31 | 27 |
| ncpi0 nue > 7.0 (background) | 0 | 1 | 1 | 1 | 0 | 1 |
| nuecc48 numu > 0.9 (mis-ID) | 3 | 5 | 3 | 8 | 5 | 7 |
| ncpi0 π⁰ mass in (100,170) | 6 | 3 | 3 | 5 | 2 | 3 |
| TGM / STM / FC flips | — | 0/0/0 | 0/0/0 | 0/0/0 | 0/1/0 | 0/0/0 |
| \|ΔEnu\| median / p90 (MeV) | — | 83 / 302 | 73 / 381 | 55 / 385 | 57 / 321 | 104 / 406 |
| sentinels PASS / FAIL (Stage-1 events) | 2 / 0 | 0 / 2 | 1 / 1 | 2 / 0 | 1 / 1 | 0 / 2 |
| wall median ratio | — | 1.16 | 1.20 | 1.09 | 0.94 | 1.18 |

The π⁰ window is small-number and pairing-dependent: every arm, kf included, re-pairs most of the 19 NCpi0
events.

Adjudication of cs's 28 movers (`149_s1_adjudication_cs.tsv`):
- **20 vertex-choice and 8 PR-structure; the main cluster changed on none.**
- Vertex-choice movers go through the DL route, whose best score swings (350186: 149 → 19.7).
- nue_score follows the vertex: the nue > 7 gains are vertices that moved **onto** the click
  (30504 31.0 → 0.1 cm, 38856 32.3 → 0.3, 163543 32.9 → 0.1); the losses moved **off** it
  (111412, 235435, 350186, 422851).

### 6.3 The DL vertex is not the cause (post-hoc attribution)

With the geometric vertex on both sides (`s0g`/`csg`):
- cs still loses: ≤ 3 cm 33 → 31, with 12 moved > 10 cm away vs 6 toward;
- stubs 63 → 82;
- nuecc48 nue > 7: 29 → 27.

The DL route even compensates partly: production 48 vs geometric 33 at ≤ 3 cm.

### 6.4 Owner ISO cases (docs pr/24, pr/67)

| event | zig-zag rms s0 → cs (cm) | vertex to click s0 → cs → csq6000 |
|---|---|---|
| nuecc48 42280 | 0.736 → 0.661 | 0.70 → 1.30 → 1.88 cm |
| nuecc48 137238 | 0.526 → 0.194 | 0.22 → 0.29 → 0.17 |
| nuecc48 271851 | 0.664 → 0.540 | (no label) |
| nuecc48 350186 | 0.749 → 0.282 | **0.00 → 40.82 → 0.24** |
| ncpi0 21073 | 0.919 → 0.805 | **0.97 → 10.47 → 2.10** |

---

## 7. Stage 2 — numuCC large-ISO subset (559 events)

`149_s2_compare_s2*.txt`, `149_s2_vertex_tolerance.txt`, `149_q1_verdict.txt`.

### 7.1 Q1, all cells

| cell | ISO ≤ 3 cm (345) | ISO > 10 cm away / toward | ISO lost | ISO-strong ≤ 3 cm (89) | ISO stubs | rms_dr Δ (cm) | ratio Δ | STM flips | lost | control > 10 cm away / toward |
|---|---|---|---|---|---|---|---|---|---|---|
| s0 / s0rep | 288 | 0 / 0 | 0 | 82 | 105 | 0 | 0 | 0 | 0 | 0 / 0 |
| cs | 257 | 31 / 12 | 12 | 64 | 151 | +0.004 | −0.006 | 17 | 15 | 3 / 4 |
| csq2000 | 260 | 34 / 13 | 10 | 67 | 144 | +0.001 | −0.006 | 15 | 11 | 6 / 1 |
| csq6000 | 256 | 35 / 10 | 10 | 71 | 138 | +0.005 | −0.007 | 18 | 14 | 4 / 3 |
| kf | 252 | 31 / 12 | 11 | 64 | 103 | −0.000 | −0.007 | 15 | 13 | 5 / 3 |
| cskf | 249 | 37 / 14 | 11 | 66 | 153 | +0.001 | −0.012 | 17 | 11 | 2 / 4 |
| cstsep05 (amend. 1) | 252 | 43 / 10 | 17 | 67 | 131 | +0.012 | −0.005 | 22 | 19 | 4 / 5 |
| cstsep07 (amend. 1) | 253 | 32 / 15 | 13 | 65 | 120 | +0.010 | −0.005 | 20 | 18 | 4 / 4 |

Owner mcp1k cases:
- 57903: vertex 1.04 → 13.18 cm, ISO zig-zag 0.27 → 1.80 cm (unit corrected in round 2);
- 284794: vertex 0.00 → 163.6 cm;
- 56463, 58717 and 59899 barely move.

### 7.2 The mechanism behind the extra branches

Of the 34 mcp1k ISO labelled events whose vertex moved > 1 cm away, the new main vertex usually carries an
**extra short branch** absent under `stepped`:
- 63551: 2.5 cm electron stub;
- 319611: 9.6 + 3.0 cm;
- 349835: 2.0 cm;
- 400504: 7.6 cm;
- 399856: 2.6 cm.

Main-cluster terminals rise ×1.38 while points rise ×1.83. Terminal nearest-neighbour spacing falls from a
median 0.68 cm (nuecc48) / 0.76 cm (mcp1k) to 0.47 cm (`terminal_spacing.py`).
`find_peak_point_indices` (`SteinerGrapher.cxx:591-640`) suppresses lower-charge candidates within
`nlevel = 3` graph **hops**. On a denser cloud three hops is a shorter distance, so more local maxima survive
as terminals, each a candidate branch end beside the true vertex. This is doc pdvd/102's simulation "c4"
track-start stub.

**Amendment 1** tested the length-based lever (`terminal_min_separation`, doc pdvd/37):
- 0.7 cm returns the terminal count to s0's level and most stubs (ISO 120 vs cs 151, s0 106). **Stubs
  recovered.**
- The vertex and churn are **not recovered**: ≤ 3 cm 253, 32 away / 15 toward, 13 labelled ISO vertices lost.
  Thinning actually *raises* the candidate losses (18 vs cs 15) and the STM flips (20 vs cs 17).
- 0.5 cm is worse, and crashes owner case nuecc48 42280 (§11).
- So the stub mechanism is real, but it is not what carries the dominant cost.

### 7.3 Two different perturbations hit mostly different events

`149_s2_overlap_cs_kf.txt`, `149_s2_lost_candidates.txt`:
- cs loses 15 candidates, 12 of them labelled; s0's vertex was ≤ 3 cm from the click on 7
  (48895, 62613, 281953, 77328, 93776, 174795, 409052). kf loses 13, also with 7 good.
- cs and kf share only **5 of 17 / 15 STM flips** and **10 of 34 / 35 > 10 cm vertex-away movers**.

The retile sampler and the fit weights are unrelated perturbations. Each moves a similar number of events,
mostly different ones, in the same net direction. This is doc pdvd/107's finding measured on SBND: the STM
verdict and (through the refit) the vertex choice sit near their own decision boundaries, and a sub-cm change
of the trajectory crosses them.

---

## 8. Stage 3 — the full numu samples (3000 events)

Arms `pr149s3a{s0,cs,kf,csq2000}` × mcp1k (1000) + mcp2k (2000), all rc=0, pin unchanged.
Files: `149_s3_compare_s3a*.txt`, `149_s3_vertex_tolerance.txt`, `149_s3_flip_directions.txt`,
`149_s3_sentinels_*.txt`.

| vs s3as0 | cs | csq2000 | kf |
|---|---|---|---|
| event_label migrations (cosmic→nu / nu→cosmic) | 71 (36 / 35) | 65 (34 / 31) | 61 (29 / 32) |
| nu_evaluated gained / lost | 34 / 34 | 30 / 29 | 27 / 30 |
| numu > 0.9: gained / lost (net) | 78 / 67 (+11) | 77 / 78 (−1) | 60 / 73 (−13) |
| nue > 7.0 count (s0 3) | 5 | 2 | 3 |
| STM flips 0→1 / 1→0 | 39 / 37 | 34 / 36 | 33 / 30 |
| FC flips 0→1 / 1→0 (Steiner-availability scope, §1) | 143 / 1 | 143 / 1 | 0 / 0 |
| TGM flips | 1 (−1 → 0, scope) | 1 (same event) | 0 |
| vertex ≤ 3 cm, all 823 labelled (s0 675) | 625 | 623 | 613 |
| > 10 cm away / toward | 69 / 40 | 71 / 33 | 71 / 33 |
| ISO (357): ≤ 3 cm (s0 297); away / toward | 265; 35 / 13 | 269; 37 / 14 | 260; 35 / 13 |
| control (465): ≤ 3 cm (s0 378); away / toward | 360; 34 / 27 | 354; 34 / 19 | 353; 36 / 20 |
| ISO stubs (s0 154) | 218 | 211 | 153 |
| \|ΔEnu\| median / p90 (MeV) | 19.6 / 191.6 | 21.2 / 185.1 | 12.2 / 184.4 |
| π⁰ mass window (s0 52) | 60 | 55 | 52 |
| sentinels, full samples (s0 21 PASS / 0 FAIL) | **15 / 6** | 13 / 8 | 14 / 7 |
| wall median / p90 ratio; RSS median | 1.007 / 1.29; 1.001 | 1.012 / 1.32; 1.001 | 1.000 / 1.25; 1.000 |

cs's sentinel failures, each a shipped owner-approved fix whose named event loses the fixed behaviour:
- **37112 pr/125 K3:** e⁻ shower 684.6 < 700 MeV.
- **69314 pr/125 K5:** 23 showers ∉ [14, 20].
- **171572 pr/123 r2:** the guard-freed muon no longer returns as a PF root.
- **72786 pr/128 class A control:** 2 < 4 declines.
- **393505 pr/129:** `mu- 267` gone.
- **292643 pr/130 B:** a π⁰ reappears.

Unconditioned, the candidate churn is two-sided (34 / 34), so cs is not a net efficiency cost. It is a
2.3 %-of-events reshuffle of which candidates survive, plus 4.8 % of the numu working-point decisions. What is
one-sided is the vertex (−50 at ≤ 3 cm, 69 vs 40 at > 10 cm) and the regressions on the named shipped fixes.

---

## 9. Recoverability

### 9.1 Pre-registered Q1 verdict per cell (pooled Stage 1 + 2 ISO, 455 events; `149_q1_verdict.txt`)

| cell | Stage 1 | Stage 2 | pooled | fails |
|---|---|---|---|---|
| cs | 0/4 | 1/4 | **1/4** | a, c, d |
| csq2000 | 2/4 | 2/4 | 2/4 | c, d |
| csq6000 | 3/4 | 2/4 | 2/4 | a, c |
| kf | 2/4 | 2/4 | 2/4 | b, c |
| cskf | 2/4 | 1/4 | 1/4 | a, c, d |

### 9.2 Class per degradation

| degradation | levers tried | outcome | class |
|---|---|---|---|
| vertex choice (ISO > control) | charge threshold 2000 / 6000, fit keys, thinning 0.5 / 0.7 cm | every cell: ISO > 10 cm away ≥ 31, toward ≤ 15; Stage 3 csq2000 623 and kf 613 vs cs 625 at ≤ 3 cm | **NEEDS-DOWNSTREAM-RETUNE:** the vertex-choice stage (DL rerank and PR candidate set) against a changed trajectory |
| candidate churn / STM verdict | same | 10–19 lost in every Stage-2 cell; Stage 3 two-sided in every cell | **NEEDS-DOWNSTREAM-RETUNE:** TaggerCheckSTM's evaluation margin (doc pdvd/107's `flag_pass`) |
| shipped-fix sentinels | csq6000 (Stage 1) | 0 FAIL on the two Stage-1 EM sentinels; Stage 3 csq2000 8, kf 7 | per fix: each was tuned on `stepped` output and needs its own re-check |
| end stubs | thinning 0.7 cm; csq6000 | recovered (Stage 1 63 → 61; Stage 2 120 / 138 vs cs 151) | **RECOVERED** (exploratory lever) |
| extra FC=1 / tagger scope on small clusters | — | a repair, not a cost | n/a |
| Enu reshuffle, π⁰ window | — | the same size as kf alone | not a bias |

**Summary.** The switch can be tuned until its density side-effects (stubs) are gone. It cannot be tuned out
of the thing that costs the most: every trajectory perturbation tested, including the fit keys PDHD/PDVD
production adopted, reshuffles SBND's refit-sensitive decisions and loses more vertices than it gains.

---

## 10. Recommendation and next step

**Do not switch SBND to `charge_stepped`.** Keep the knob OFF, and do not adopt the two fit keys on SBND
either: kf alone costs as much vertex accuracy as cs.

Before revisiting, **the refit instability has to be addressed at its site**, not in the sampler. This is the
same open item PDHD carries (docs pdvd/103–107):
1. **Measure the STM verdict margin.** TaggerCheckSTM's KS / ratio `flag_pass` distance to the threshold, per
   candidate, on s3as0. Then count how many of the 76 STM flips sit within the refit noise
   (doc pdvd/103 measured p90 0.06 on PDHD). That tells whether a hysteresis / margin rule is viable.
2. **Vertex choice.** Test whether the stub-branch vertices (§7.2) are removable by a length or charge floor
   on degree-1 branches *at the candidate main vertex only*, as a default-OFF knob graded on the Stage-2
   manifest. The thinning result says terminal density alone is not enough.
3. **Round 2 changes the trajectory item.** Neither cloud is the lever for ISO trajectories (§13). The next ISO
   round should measure the ~1.2 cm drift-direction bow on long ISO muons at its source (§13.8), not retune a
   sampler.
4. **Owner scan.** 34 A/B events (owner cases, Stage-1 > 10 cm movers, Stage-2 lost candidates and ISO-strong
   movers) are packaged locally at `bee/pr149/` with an annotated `pr149.index.txt`. Not uploaded: say the word
   and they go up. The scan would decide whether any "away" mover is an A0-WAS-WRONG, the one class this doc
   cannot adjudicate from labels drawn on `stepped`.

---

## 11. Defects found, reported, not fixed

1. **Segfault in PR graph surgery**, reproducible:
   - Run: `TAG=cstsep05r PIN=$P SAMPLES=nuecc48 MANIFEST=<42280> TLA_FILE=$D/tla/cstsep05.tla bash scripts/pr149_arm.sh`
     gives rc=139, twice.
   - Stack: `PR::remove_segment` (`PRGraph.cxx:315`) → `boost::remove_edge` → `std::list::_M_erase`, from
     `PatternAlgorithms::break_segment_into_two` ← `break_segments` ← `find_other_segments` ←
     `find_proto_vertex`.
   - Trigger: owner ISO case nuecc48 42280 under `charge_stepped` + `steiner_terminal_min_separation = 0.5`;
     s0, cs and 0.7 cm run it clean.
   - A latent PR-graph defect (an edge removed twice or stale) that this topology reaches. It makes the 0.5 cm
     setting unrecommendable as configured.
2. **`scripts/pr127_sentinels.py` aborts on an event whose `mabc-pr.zip` has no `data/0/0-mc.json`.** An arm
   that loses the candidate has no PF tree (cs: mcp2k 77328), and a KeyError ends the whole run instead of
   failing that sentinel. The production script is untouched; `sentinels_tolerant.py` wraps `pf_texts()` for
   this doc.
3. **`wcls-img-clus-matching-xin.jsonnet` with `pr_operating_point: "preflip"` does not compile**
   (`Persist::load` throws). It fails on the pre-knob tree too, so it is pre-existing and unrelated.
4. **Sentinel event ids collide across samples** (69314 exists in both nuecc48 and mcp2k), and
   `pr127_sentinels.py` takes the first arm glob that has it. Comparisons here use the same arm order for
   every cell, so they are like for like. The registry itself is ambiguous.
5. **Round 1 labelled the zig-zag numbers mm; they are cm** (found in round 2). `pr149_metrics.py` reads the
   calib dump's `x/y/z`, whose `meta.length_unit` is `cm`. §1, §6.1, §6.4 and §7.1 are corrected in place.
   The frozen `149_pred.txt` / `149_pred_r2.txt` keep the old label. Their numeric thresholds were always
   applied in the metric's own unit, so no verdict changes.
6. **`arm_identity.py --allow A --allow B` keeps only B** (argparse `nargs='+'`, last flag wins) and reports a
   false difference on A. The list must be one flag: `--allow A B C`. Round 1's gates each used a single
   `--allow`, so they are unaffected. Round 2 hit it once and re-ran with the list form.

---

## 12. What this doc did not do

- Round 1 did not switch the clustering job's 3-D sampler (owner scope). **Round 2 (§13) switched the PR
  job's copy of it**, the prototype's own step, and found no ISO gain. The clustering job's sampler itself
  stays `stepped`, as in the prototype.
- No simulation leg (owner scope), so there is no truth-level trajectory residual.
- No production flip and no Bee upload.
- No π⁰ fixed-pairing check (doc pr/135 method). The window counts in §6.2 and §8 are pairing-dependent.
- No blinding of the Bee package (house OFF/ON convention); the zip names reveal the arm.

---

## 13. Round 2 — re-sampling the whole cloud, the prototype PR job's step

**Owner question (2026-09-16):** the round-1 hypothesis was that ISO did not improve because the track fit's
association still reads the old `stepped` cloud, not the `charge_stepped` one. Change the cloud altogether on
ISO events and see whether things improve. This round is for understanding only; no production change is
intended.

**Answer: no.** Neither the premise nor the prediction holds:
1. **Pre-registered primary endpoint fails.** On the 150 long-ISO muon events (Stage-2 `iso` stratum):
   - whole cloud + retile vs s0: median Δ`zzi_rms_dr` +0.008 cm, 59 improved / 71 worsened (p = 0.34);
   - whole cloud alone: +0.015 cm, 55 / 76 (p = 0.08).
2. **The association never depended on the cloud.** On ~4 500 ISO fit points per arm, the fit's per-plane
   association is the same in all four arms: 15 cells per plane, and the charge centroid sits 0.08 / 0.19 /
   0.005 wires and 0.24 ticks from the solved point (§13.4).
3. **What the retile does move is the wrong component for this metric (post hoc, §13.6).**
   - The local jitter across the wires falls: whole cloud + retile 83 improved / 7 worsened, p = 1e-17.
   - `zzi_rms_dr` measures drift x about one chord. On long ISO muons that is a ~1.2 cm smooth bow, which no
     cloud and no fit key moves.
4. **Costs are round 1's, not worse.** Same class of vertex loss, the same 17 STM flips, the same
   shipped-fix sentinel failures. Whole cloud alone fails 6 of 12 sentinels.

### 13.1 What the prototype does

- **Clustering** samples with the stepped rule, `WCP2dToy::calc_sampling_points`
  (`2dtoy/src/CalcPoints.cxx:613-700`, every `max(3, N/12)` wires). The toolkit's stepped clustering cloud is
  faithful.
- **The PR executables replace every live cluster's cloud before any PR** ("replace by the new sampling
  points ...", `pid/apps/wire-cell-prod-nue.cxx:1289-1294`, same step `wire-cell-prod-stm.cxx:734`).
  - They use `WCPPID::calc_sampling_points`, the charge_stepped rule with `disable_mix_dead_cell = true`
    (`pid/inc/WCPPID/CalcPoints.h:9-10`).
  - This runs before `Protect_Over_Clustering` and `NeutrinoID`.
  - The retile (`ImprovePR3DCluster.cxx:59`) separately uses `false`.
- **The toolkit's SBND PR job had no such step.** It reads stage A's stepped points from the pctree, which
  makes this a second prototype divergence, undocumented until now.

So "the cloud altogether" was implemented as that step. Switching the clustering job's sampler instead would
also have moved clustering, bundles and the flash match, which the prototype does not do.

### 13.2 Implementation and gates (toolkit, default OFF)

- **New visitor `ClusteringResampleLive`** (`clus/src/clustering_resample_live.cxx`; pure helpers in
  `clus/inc/WireCellClus/ResampleLive.h`; doctest `clus/test/doctest_resample_live.cxx`). No production C++
  file is edited.
  - The PR job holds no IBlob or ISlice, so for each live blob it rebuilds the blob **shape** from its
    `scalar` wire bounds (two dummy layers + U/V/W) and the slice **activity**.
  - Activity precedence: the grouping's ctpc row (live), else the dead-wind registry as `(0, 1e12)`, else
    absent.
  - It then samples, replaces the blob node's `"3d"` and refreshes only `center_x/y/z` and `npoints` in
    `scalar`.
  - Blob nodes are replaced as nodes, in order, so the facade caches and scoped views are notified and the
    per-blob provenance rows stay aligned.
- **Config** (`sbnd/{clus,wct-pr-perevt}.jsonnet`): TLA `resample_live_strategy`.
  - Non-null prepends `resample_live` to the PR pipeline, ahead of `switch_scope`, which rebuilds `x_t0cor`
    and the in-volume split from the new points.
  - The sampler is `live-rs-<cs|st>-<apa>-0`: charge_stepped with `disable_mix_dead_cell: true`, and the
    clustering job's `extra` arrays.
  - null ⇒ stage absent ⇒ byte-identical.

**The gate that licenses the round.** A `stepped` resample must rebuild the clustering job's saved cloud bit
for bit. It does:
- 16/16 events identical over 40 files each: every pctree-pr member, the Bee zip, the calib dump, every
  ROOT branch, nusel;
- the only difference is `Trun.op_config_sha256`, the hash of the compiled config;
- the `RESAMPLE` log line confirms the stage ran (points N → N, `interval_mismatch 0`).

So the shape, x/tick, charge and dead-flag reconstruction is exact wherever a sampled point lands.

| gate (`149_r2_*`) | result |
|---|---|
| `cfg_proof.txt`: prod_cfg_gate before and after | PASS 21/21, both |
| one-step LArSoft chain, sync/bare × tracking root | identical 4/4 |
| ON node diff (charge_stepped / stepped / + retile) | only `ClusteringResampleLive:pr`, 2 × `BlobSampler:live-rs-*`, the `clus_pr` pipeline head (+ round 1's retile nodes); typo aborts |
| `wcdoctest-clus` in the round-2 build | 423/423 (5 new cases, 240 assertions; includes a tiled-vs-rebuilt blob sampling bit for bit under both strategies, and a negative control showing why an absent wire must not be written dead) |
| runtime OFF gate `gate_off_*` (16 evt, geometric vertex, knob tree vs `06fd9e22` cfg overlay, same pin) | 16/16 identical (`Trun.cfg_tree` allowed) |
| identity gate `gate_identity_stepped_*` | 16/16 identical (`Trun.op_config_sha256` allowed) |
| dead-path census `resample_census_gate16.txt` (charge_stepped) | 82 509 blobs; a dead first/last strip wire on U 0.91 % / V 2.72 % / W 2.70 % of blobs; dead only through the hand-declared W gap 0.95 % (those W channels are chndb-bad, dead in imaging too); cloud ×2.57 over all clusters |
| lineage `lineage_s0_*`, `lineage_s2s0_*`: new-pin s0 vs round-1 s0 | Stage 1 67/67 and Stage 2 559/559 events identical over every file and ROOT branch, apart from `Trun.toolkit_git` / `wct_version` / `op_config_sha256`; round-1 arms (kf) are directly comparable |

Build isolation: a peer had uncommitted C++ in `SteinerGrapher.cxx` / `TaggerCheckSTM.cxx` in the shared
toolkit tree. Round 2 therefore built a detached worktree of `06fd9e22` plus only the round-2 files, into a
private prefix. It used the shared build's exact waf flags (package list equal, `root` included), and the pin
prepends that prefix. All 16 running gate jobs loaded the pinned `libWireCellClus.so` (checked in
`/proc/<pid>/maps`).

The peer then pushed that work as `724cf205` (an env-gated, log-only Steiner graph dump). The round-2 commit
**`e73850ad`** sits on top of it; an incremental rebuild of exactly `e73850ad` compiles and passes
`wcdoctest-clus` 426/426.

### 13.3 Pre-registration

`149_figs/149_pred_r2.txt`, sha256 `e486f9ada408…`, frozen 10:54:46, after the non-physics gates and before
the first physics arm (10:55:02). It fixes:
- the four cells (`s0`, `cs` = round 1's retile, `rs` = whole cloud, `rscs` = both, the prototype-faithful
  cell);
- the events: Stage 1 = 67 nueCC/NCpi0, Stage 2 = round 1's 559-event manifest, trace = 8 long-ISO events;
- the primary endpoint and its rule;
- the attribution pairs, with secondaries marked descriptive-only.

Deviations, all recorded:
1. **Unit label.** The pred text says mm; the metric is cm (§11 item 5). The threshold 0.01 is applied in cm.
2. **Trace call selection.** The pred says "last `trajectory_fit` call" of the main cluster. The last call is a
   per-segment refit of 3–5 points, and the first output compared 51 against 783 ISO points across arms. The
   script was changed to the largest call of the highest charge-division method (the whole-cluster fit), after
   seeing that first output.
3. **§13.6 is post hoc:** the jitter/bow split, suggested by doc pr/73 §4.7.
4. **Sentinel denominator.** The round-2 arms cover the Stage-1 + Stage-2 manifests, so 12 sentinels are
   evaluable (10 SKIP), against round 1's 21 on the full samples.

All arms rc=0: Stage 1 4 × 67, Stage 2 4 × 559, trace 4 × 8, gates 4 × 16.

### 13.4 The association is not starved and does not follow the cloud

`WCT_TRAJ_ASSOC_DEBUG` trace (doc pdvd/111) on the 8 long-ISO events (`manifest_r2trace`: top 3 of the `iso`
stratum per numu sample, plus owner cases 284794 and 57903). The table covers ISO fit points (local chord
≥ 75° to drift) of the main cluster's whole-cluster fit (`149_r2_assoc_trace.txt`).

| arm | ISO points | cells per plane U/V/W (median) | \|wcen − wsol\| U/V/W (wires) | \|tcen − tsol\| (ticks) | points with 0 cells on a plane |
|---|---|---|---|---|---|
| s0 | 4522 | 15 / 15 / 14 | 0.081 / 0.186 / 0.005 | 0.238 | 6.7 % |
| cs (retile) | 4121 | 15 / 15 / 13 | 0.085 / 0.190 / 0.005 | 0.252 | 4.1 % |
| rs (whole cloud) | 4531 | 15 / 15 / 14 | 0.088 / 0.193 / 0.005 | 0.244 | 4.0 % |
| rscs (both) | 4516 | 15 / 14 / 13 | 0.082 / 0.187 / 0.005 | 0.245 | 5.8 % |

- The main association (`TrackFitting.cxx:3079-3106`) takes the closest point of the cluster's cloud, walks
  `nlevel` `basic_pid` hops to collect **blobs**, and then takes those blobs' pixels within the distance cut.
- A denser cloud changes which points are hops apart, but on an ISO ribbon the handful of blobs reached, and
  hence the pixel set, is essentially the same.
- Every arm's solved point already sits on its association centroid to 0.2 wire.
- cs, whose Steiner block does change, leaves these numbers as flat as rs does.
- Per event the arms agree too, except where the main cluster itself changed: cs on 284794 moves the main
  vertex to another cluster, as in round 1.

### 13.5 Primary endpoint and attribution

`149_r2_primary_stage2.txt`, `149_r2_primary_stage1.txt`. The comparison is paired over events with a finite
`zzi` in both arms. Improved / worsened means |Δ`zzi_rms_dr`| > 0.01 cm, with an exact sign test.

| comparison | Stage-2 `iso` (150): median Δ`zzi_rms_dr` cm; Δratio; imp / wor; p | Stage-2 `vtx_iso` (256) | Stage-1 ISO (49) |
|---|---|---|---|
| **rscs vs s0 (PRIMARY)** | **+0.0078; −0.0106; 59 / 71; 0.34 → does not improve** | −0.0062; −0.011; 108 / 103; 0.78 | −0.074; +0.011; 27 / 16; 0.13 |
| rs vs s0 (whole cloud alone) | +0.0149; −0.0010; 55 / 76; 0.08 | −0.0033; −0.002; 107 / 103; 0.84 | +0.026; +0.007; 22 / 24; 0.88 |
| cs vs s0 (round 1 reproduced) | −0.0031; −0.0062; 66 / 65; 1.0 | +0.0079; −0.006; 98 / 116; 0.25 | −0.028; +0.001; 27 / 18; 0.23 |
| rscs vs cs (resample on top of retile) | +0.0076; −0.0022; 61 / 69; 0.54 | −0.0144; −0.004; 115 / 87; 0.06 | −0.033; +0.001; 25 / 16; 0.21 |
| rscs vs rs (retile on top of resample) | −0.0146; −0.0071; 77 / 58; 0.12 | +0.0074; −0.008; 96 / 106; 0.53 | −0.054; −0.010; 26 / 17; 0.22 |

- No comparison meets the rule anywhere.
- The whole cloud alone leans the wrong way on long ISO muons.
- The two sub-threshold negatives (Stage 1 rscs −0.074; Stage 2 rscs vs rs) are what the retile contributes.
  They are not a whole-cloud effect.
- The pairing conditions on survival: 11–12 labelled ISO candidates are lost in each ON cell (§13.7), and a
  lost candidate leaves the pairing rather than counting as a failure. That can only flatter an ON cell, and
  the result is null anyway. It would matter for a future small positive (same trap as §5.2).

### 13.6 Why the metric stays flat: jitter vs bow (post hoc)

`r2_local_jitter.py`, `149_r2_local_jitter_stage{1,2}.txt`. The same segments as `zzi` are split into three
terms:
- `jit_x`: point-to-point jitter in drift x, the rms second difference / √6;
- `jit_t`: the same jitter in the in-plane transverse direction (across the wires);
- `bow_x`: the rms drift-x residual about the chord after a 9-point running mean, i.e. the smooth excursion.

The table uses Stage-2 `iso` (150); kf is round 1's fit-key arm vs round-1 s0 (lineage identical).

| vs s0 | `jit_x` (base 0.043 cm) | `jit_t` (base 0.096 cm) | `bow_x` (base ~1.17 cm) |
|---|---|---|---|
| cs | +0.0004; 31 / 43; p 0.20 | **−0.0048; 68 / 25; p 9e-6** | −0.004; 69 / 66; p 0.86 |
| rs | +0.0003; 29 / 38; p 0.33 | −0.0018; 45 / 31; p 0.14 | +0.016; 55 / 78; p 0.06 |
| rscs | −0.0005; 31 / 37; p 0.55 | **−0.0076; 83 / 7; p 1e-17** | +0.009; 63 / 75; p 0.35 |
| kf (round 1) | −0.0010; 38 / 27; p 0.22 | **−0.0106; 113 / 4; p 9e-29** | −0.010; 73 / 61; p 0.34 |

**This is the answer to "why not improved".**
1. **The metric is dominated by a term no lever reaches.** `zzi_rms_dr` (base 1.15 cm) is almost entirely
   `bow_x` (1.17 cm), a smooth drift-direction excursion of the segment about its chord. The drift-x jitter is
   0.04 cm and does not move.
2. **What the levers do fix is invisible to `zzi_rms_dr`.**
   - The charge_stepped retile, which builds the Steiner seed path, removes part of the lattice sawtooth
     *across the wires* (doc pr/73 §2: seed points on the wire-crossing grid). This is −5 % of `jit_t`, and
     −8 % together with the whole-cloud resample.
   - The fit keys remove the most (−11 %).
   - The whole-cloud resample alone removes little (−2 %, not significant), consistent with §13.4.
3. **On the 49 Stage-1 nueCC/NCpi0 ISO events**, `bow_x` is smaller (~0.5 cm) and no term reaches
   significance.

What the bow is has not been adjudicated here.
- Doc pr/73 §2 shows one mechanism on its cases: `multi_trajectory_fit` pins both segment ends to the vertex
  fit points, so an off-ridge vertex forces a bow.
- On data, part of a 1.2 cm excursion over a 100–500 cm muon can also be physical: space-charge distortion of
  drift x, or multiple scattering.

### 13.7 Secondary and costs (descriptive, per pred §5)

Stage 2 (559 numu events, vs `pr149s2r2s0`; `149_r2_s2_compare_*.txt`, `149_r2_s2_vertex_tolerance.txt`,
`149_r2_q1_verdict.txt`, `149_r2_s2_topology.txt`):

| | cs | rs | rscs |
|---|---|---|---|
| ISO (345) ≤ 3 cm (s0 288) | 257 | 249 | 250 |
| ISO > 10 cm away / toward; lost | 31 / 12; 12 | 35 / 10; 12 | 40 / 13; 11 |
| ISO-strong (89) ≤ 3 cm (s0 82) | 64 | 67 | 66 |
| control (87) ≤ 3 cm (s0 70); away / toward | 67; 3 / 4 | 62; 5 / 2 | 64; 2 / 2 |
| ISO stubs (paired) | 105 → 151 | 107 → 108 | 106 → 159 |
| event_label migrations / nu_evaluated flips | 16 / 15 | 15 / 13 | 16 / 13 |
| STM / FC / TGM flips | 17 / 2 / 0 | 17 / 1 / 0 | 17 / 2 / 0 |
| \|ΔEnu\| median / p90 (MeV) | 36 / 201 | 31 / 214 | 28 / 243 |
| wall median / p90 ratio; RSS median | 1.04 / 1.31; 1.00 | 1.12 / 1.45; 1.03 | 1.14 / 1.52; 1.03 |
| events whose cluster count changes (net); main cluster or its length changed | 4 (−5); 0 | 17 (−13); 0 | 19 (−17); 0 |
| round-1 Q1 a–d, pooled 455 ISO events | 1/4 | 1/4 | 1/4 |

Stage 1 (67 nueCC/NCpi0, vs `pr149r2s0`; `149_r2_s1_*`):

| | cs | rs | rscs |
|---|---|---|---|
| all 56 labelled ≤ 3 cm (s0 48) | 45 | 41 | 42 |
| ISO (40) ≤ 3 cm (s0 34); > 10 cm away / toward | 34; 4 / 4 | 31; 6 / 2 | 31; 4 / 4 |
| ISO stubs (s0 63) | 79 | 53 | 90 |
| label / nu_evaluated / STM flips | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| events whose cluster count changes (net) | 0 | 13 (−17) | 13 (−17) |
| wall median / p90 ratio; RSS median | 1.28 / 1.75; 1.01 | 1.28 / 1.64; 1.06 | **1.55 / 2.14**; 1.06 |

Sentinels evaluable on the round-2 manifests (12; `149_r2_sentinels_*.txt`):

| s0 | cs | rs | rscs |
|---|---|---|---|
| 12 PASS / 0 FAIL | 9 / 3 (37112 K3, 69314 K5, 393505 pr/129) | **6 / 6** (+ 77328 pr/125 guard, 315167 pr/93 r4, 497311 doc 84 r1) | 9 / 3 (the same as cs) |

Reading:
- **The vertex and churn cost is round 1's.** Every cell loses ~30–40 ISO vertices > 10 cm away against
  ~10–13 toward, and flips the same 17 STM verdicts. This is refit instability (§7.3), and the extra cloud does
  not change its size.
- **The whole cloud alone is the worst cell on sentinels (6 FAIL)** without a trajectory gain. This is reported,
  not explained.
- **Topology:** the resample changes the cluster count on 17–19 of 559 and 13 of 67 events, always fewer
  clusters. More points let more blobs pass `switch_scope`'s volume filter, so fewer out-of-volume shards are
  split off. The main cluster and its length never change, so the per-event metric join is on the same object.
- **Resources:** the whole cloud costs +12–14 % wall on numu and up to +55 % median (p90 +114 %) on nueCC, with
  RSS +3–6 %. Stage-1 arms ran concurrently at equal job counts, so their ratios are like for like.

### 13.8 Conclusion and next step

- **Switching the cloud altogether does not improve ISO PR on SBND,** and switching the whole cloud alone is
  mildly worse.
- The round-1 explanation was wrong. The association does not depend on which cloud the fit walks, and the
  ISO trajectory deviation that the metric sees is a smooth drift-direction bow outside any sampler's reach.
- The part the samplers can reach, the jitter across wires, does fall with the retile, most with the fit keys.
  It is ~0.1 cm against a ~1.2 cm bow, and it comes with round 1's refit-instability costs.

**Recommended next step: measure the bow at its source before any further ISO lever.**
- **Sample:** the 150 long-ISO muons of the `iso` stratum.
- **Comparison:** the fitted drift-x profile against the image's own time centroid along the track, from the
  ctpc charge per wire and slice (T_proj_data).
- **If the image follows the bow,** it is physical (space charge, scattering), and no PR lever should remove
  it; the zig-zag metric should then be measured about a smooth curve, not a chord.
- **If the image is straight** and the fit bows toward the pinned segment ends, it is doc pr/73's
  seed/vertex pinning in `multi_trajectory_fit`, and that is the lever to test.
