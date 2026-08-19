# doc pr/92 — drop stray satellite showers (overclustered cosmics / second neutrinos) from kine_reco_Enu and the Bee PF tree

Owner request (2026-08-18): three events show clearly-overclustered activity
admitted into the final Particle Flow and neutrino energy; the key
discriminator is "whether the direction is consistent with a known vertex in
the main cluster"; drop such activity (cosmic muons, second major neutrinos)
without dropping legitimate particles.

## Repro

```bash
cd wcp-porting-img/sbnd/sbnd_xin

# V1 pre/post-change byte-identity (knobs off), 4 spot events:
#   pre-change binary = toolkit HEAD 24e6f969 (git stash of the pr/92 edits),
#   post-change binary = pr/92 edits, both knobs-off.
PR_JOBS=2 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-mcp1k-cb0805   work-pr92-prechange-mcp1k   data 350935 321371
PR_JOBS=1 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr92-prechange-nuecc48 data 389538
PR_JOBS=1 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-ncpi0-cb0805   work-pr92-prechange-ncpi0   data 521075
# (same three lines with -postchange- out_roots after git stash pop + wcbuild)

# Probe round (WCT_KINE_SAT_PROBE prints per-candidate metrics, never drops):
WCT_KINE_SAT_PROBE=1 PR_JOBS=32 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-mcp1k-cb0805   work-pr92-probe-mcp1k   data <33-evt list>
WCT_KINE_SAT_PROBE=1 PR_JOBS=32 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr92-probe-nuecc48 data
WCT_KINE_SAT_PROBE=1 PR_JOBS=32 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-ncpi0-cb0805   work-pr92-probe-ncpi0   data

# V2 ON arms:
SBND_KINE_DROP_STRAY_SATELLITES=1 SBND_PF_DROP_STRAY_SATELLITES=1 \
  PR_JOBS=32 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-<sample>-cb0805 work-pr92-on-<sample> data [evts]
```

## Symptom

1. **evt 350935** (mcp1k): a 250-cm overclustered cosmic (cluster 11, not the
   main cluster) becomes shower 11001, ID'd as a **449 MeV electron** with a
   gamma carrier in the PF tree, conn=3, `kine_energy_included=3`, and its
   449 MeV is summed into `kine_reco_Enu` (1203 MeV total).
2. **evt 321371** (mcp1k): a 256-cm cosmic in cluster 18 is mostly dropped,
   but its 29.5-cm tail becomes shower 18004 (conn=2, pdg 13, **97.9 MeV**,
   `included=1`), attached to main-cluster vertex 27013 79.8 cm away and
   collinear (6.4° kink) with the dropped cosmic body.
3. **evt 389538** (nueCC48): a second neutrino is split across satellite
   clusters: shower 19040 (**997 MeV**, conn=2, attached to vertex 11000 —
   the far end of the main muon — 144.5 cm away; rendered "neutron → proton"
   in the pre-pr91 PF tree, gamma after pr91), 54060 (237 MeV, conn=3,
   attached to another satellite's vertex), 65075 (64), 77093 (30), 74086
   (48), 78099 (40), 64073 (20). Enu is biased by ~1.4 GeV.

## Root cause

`fill_kine_tree`'s leftover pass (`clus/src/NeutrinoKinematics.cxx`) admits
every BFS-unreached shower with `start_connection_type <= 3` — no direction,
distance, or cluster check — and `kine_reco_Enu` sums every entry regardless
of the `included` flag. The Bee PF tree (`fill_bee_pf_tree`,
`clus/src/MultiAlgBlobClustering.cxx`) admits the same set (only conn=4 is
skipped), rendering satellites behind gamma-22/neutron-2112 pseudo carriers.

**The prototype has the identical hole** (audited): WCP
`NeutrinoID_kine.h:209-255` sums every conn<=3 shower with no direction
check, and every shower deletion in WCP (`NeutrinoID_em_shower.h`,
`NeutrinoID_shower_clustering.h`) is merge-driven, never a
"points-away-from-the-vertex" veto. The only creation-time angle gates are in
`shower_clustering_with_nv_from_vertices` (reject `angle>50 && dis>6cm` or
`angle>60`), which the three offenders all pass. This fix is therefore NEW
behavior, shipped default-OFF per CLAUDE.md §1.

## Why it hid

Legitimate displaced pi0 gammas ride the same conn-2/3 admission path, so the
population is dominated by correct entries; the offenders only stand out on
events where a cosmic or second neutrino is overclustered NEAR the main
cluster (within the 80 cm association radius or the nv_from_vertices angle
cone).

## Fix (toolkit, all default OFF)

Master knob `kine_drop_stray_satellites` (TaggerCheckNeutrino →
PatternAlgorithms) + PF mirror `pf_drop_stray_satellites` (BeePFConfig).
Decision inside `fill_kine_tree`, on BFS-unreached candidates only (conn 2/3,
start segment in a NON-main cluster). Metrics use a FRESH axis
`shower_cal_dir_3vector(shower, start_point, kine_sat_axis_dis_cut=30cm)` —
the stored `init_dir` for conn-2/3 is exactly the vertex→start chord and
always reads 0°.

Keep-first decision order:
1. pi0-paired shower (`pi0_showers`) → KEEP.
2. `kine_best <= kine_sat_min_energy` (20 MeV) → KEEP.
3. zero axis → KEEP (fail-safe); `d_svtx < kine_sat_prox_max` (8 cm) with a
   main-cluster attachment → KEEP (proximity exemption; ncpi0 71372/19020,
   1975 MeV at 5.2 cm, motivates this).
4. **Arm A**: angle(axis, start − svtx) > `kine_sat_angle_bad` (60°) → DROP.
5. **Arm B**: (d_svtx > `kine_sat_far_dis` (90 cm) OR svtx not in the main
   cluster) AND angle(axis, start − main_vtx) >= `kine_sat_angle_main` (45°)
   → DROP.
6. **Arm C**: `shower_start_is_track_continuation` (NEW,
   `clus/src/PRShowerFunctions.cxx`) → DROP: a same-cluster sibling of the
   start segment that is NOT a member of the shower, collinear
   (< `kine_sat_cont_kink` 25°), and straight-long (or the combined chain
   qualifies). Deliberately NOT
   `segment_is_straight_long_track_or_continuation`: its self-straight arm
   fires on legit EM trunks (ncpi0 506114/19016, 1839 MeV, 11-cm trunk with
   direct > 0.93·len) and its continuation arm accepts in-shower siblings
   (19016's 2.3°-kink pdg-211 sibling 19015 is a shower member).

Dropped showers are skipped wholesale in the kine leftover loop (all four
parallel vectors + the proton binding-energy add stay consistent; Enu
shrinks automatically) and their ids transported via
`TrackFitting::set_dropped_satellite_shower_ids` (replace semantics,
stashed unconditionally per event) to `fill_bee_pf_tree`, which skips them
beside the conn-4 skip and guards the two parent-shower attachment branches.

`kine_reco_Enu` feeds the nue/numu BDT scores
(`root/src/Uboone{Nue,Numu}BDTScorer.cxx`) — score movement on affected
events is the purpose of the fix and is characterized below. T_tagger is
unaffected (all taggers run before `fill_kine_tree`).

Probe mode: env `WCT_KINE_SAT_PROBE` prints one `KINE_SAT_PROBE\t...` TSV
row per candidate (full population, thresholds inert, never drops) so
thresholds are chosen on at-decision-time C++ numbers (pr/83 lesson).

## Per-event evidence (calib start→end axis survey, pre-fix arms)

| evt | shower | E (MeV) | conn | d_svtx | ang_svtx | in_main | d_main | ang_main | killing arm |
|-----|--------|---------|------|--------|----------|---------|--------|----------|-------------|
| 350935 | 11001 | 449.1 | 3 | 31.3 | 93.1° | yes | 127.0 | 48.0° | A (+C) |
| 321371 | 18004 | 97.9 | 2 | 79.8 | 28.1° | yes | 30.6 | 72.9° | C |
| 389538 | 19040 | 997.2 | 2 | 144.5 | 18.5° | yes | 169.4 | 68.8° | B |
| 389538 | 54060 | 236.6 | 3 | 6.6 | 91.6° | no (cluster 74) | 219.4 | 73.4° | A |
| 389538 | 65075 | 64.0 | 3 | 3.2 | 123.8° | no (cluster 64) | 228.7 | 63.6° | A |
| 389538 | 77093 | 29.8 | 3 | 29.6 | 9.5° | no (cluster 78) | 249.6 | 81.3° | B |
| 389538 | 64073 | 20.2 | 2 | 198.8 | 41.3° | yes | 231.2 | 84.1° | B (if > E floor) |
| 389538 | 74086 | 47.6 | 2 | 225.3 | 40.3° | main vtx | 225.3 | 40.3° | none — ACCEPTED residual |
| 389538 | 78099 | 40.5 | 2 | 245.9 | 39.1° | main vtx | 245.9 | 39.1° | none — ACCEPTED residual |

Must-keep sentinels (legit, from the 1196-satellite survey over
nueCC48+ncpi0+mcp1k): ncpi0 506114/19016 (1839 MeV, 4.9°), 84229/69125
(1001 MeV, 24.1°), 521075/18007 (674 MeV, 4.7°), 105946/55043 (508 MeV,
ang_svtx 34.1° at a main-cluster vertex 47 cm out), 71372/19020 (1975 MeV,
proximity-exempt at 5.2 cm).

## Verification (all PASS, 2026-08-18)

**V1 — knobs-off byte-identity** (pre-change binary = HEAD `24e6f969` via
`git stash`, post-change = pr/92 edits, both knobs off, same cfg):
`pr85_hash_gate.py work-pr92-prechange-<s> work-pr92-postchange-<s>` PASS
8/8 archives (350935+321371 mcp1k, 389538 nueCC48, 521075 ncpi0);
nusel tables and `kine_reco_Enu` identical per event.  Compiled-config
proofs: knobs-off `wcsonnet` compile of `wct-pr-perevt.jsonnet` (full PR
pipeline TLA) byte-identical to the HEAD compile; knobs-on compile shows
the new keys.  `./build/clus/wcdoctest-clus` 210/210 PASS.

**Probe round** (`WCT_KINE_SAT_PROBE=1`, drops disabled, C++ at-decision
metrics; arms `work-pr92-probe-{mcp1k,nuecc48,ncpi0}`, 100 events, all
rc=0): 1202 conn-2/3 non-main-cluster candidates, **51 would-drop in 30
events** at the C++ default thresholds (20 MeV / 8 cm / 60° / 45° / 90 cm /
30 cm / 25°).  All 8 target offenders drop — 350935/11001 (A), 321371/18004
(C), 389538: 19040 (B), 54060 (A), 65075 (B), 77093 (B), 64073 (B), and
74086 (B — the fresh 30-cm axis puts it past 45°, better than the crude
survey predicted; only 78099, 40.5 MeV at 39°, remains as the accepted
residual).  Every must-keep sentinel kept: 506114/19016 (1839 MeV),
84229/69125 (1001), 521075/18007 (674), 105946/55043 (508), 71372/19020
(1975, proximity-exempt at 5.2 cm), 259542's real gammas 17138 (999) +
124087 (189).  The thresholds were confirmed as-is — no retune needed.

**V2 — OFF vs ON** (`work-pr92-probe-<s>` vs `work-pr92-on-<s>`, both
SBND_KINE_DROP_STRAY_SATELLITES=1 + SBND_PF_DROP_STRAY_SATELLITES=1 on the
ON arms; all rc=0): `pr85_hash_gate.py` diffs = **exactly the 30
probe-predicted events, mabc-pr.zip ONLY** — every pctree archive
byte-identical (the drop is strictly post-pattern-recognition), and
`nusel-table.tsv` byte-identical 3/3 samples (**no selection changes**).
BDT scores move only on drop events (e.g. 259542 numu 0.19→0.61, 52672
−0.38→−0.99), the intended consequence of removing cosmic/second-nu energy
from `kine_reco_Enu`; no event loses its selected neutrino.  Total energy
removed: **6657 MeV over 30 events** (of ~100).  Owner-event spot checks:

| evt | Enu OFF → ON | PF change |
|-----|--------------|-----------|
| 350935 | 1203.2 → 754.0 (−449.1) | "gamma 449 MeV → e− 449 MeV" gone |
| 321371 | 831.1 → 733.2 (−97.9) | "neutron 97 MeV → mu− 97 MeV" gone |
| 389538 | 2319.8 → 958.0 (−1361.8) | "neutron 955 MeV → proton 955 MeV" + five second-nu gammas gone; kept 9.3 MeV blip re-anchors to root (dropped-parent guard) |
| 349549 / 316025 / 521075 / 506114 / 105946 / 84229 / 269774 | unchanged | untouched (sentinels) |

**V3 — composition gate** (after the cfg flip; bare runs, no env):
`pr85_hash_gate.py work-pr92-on-<s> work-pr92-bare-<s>` **PASS 200/200
archives** (66+96+38) + nusel identical 3/3 — bare production ==
env-forced ON.

**Bee A/B** (30 mover events; per-event notes in `bee/pr92/pr92.index.txt`):
- OFF: https://www.phy.bnl.gov/twister/bee/set/530142d5-e75b-42f0-9346-4f08cfa052bd/event/list/
- ON:  https://www.phy.bnl.gov/twister/bee/set/ab20e00c-251a-4c6a-bc44-38e8fa052b65/event/list/

**SBND production flip**: `wct-pr-perevt.jsonnet`
`kine_drop_stray_satellites = true` + `pf_drop_stray_satellites = true`
(owner flip 2026-08-18); scalars stay at the C++ defaults.

Accepted residuals / notes:
- 389538 shower 78099 (40.5 MeV, 39.1° vs main at 246 cm) survives at
  θ_B=45° — lowering θ_B to catch its 40 MeV would start eating legit
  far gammas (survey: legit keeps up to ~36°).
- The probe prints the internal sequential shower index (get_shower_id()),
  not the display id (cluster*1000+seg) — map by energy/geometry.
- mcp1k drop events 55539/349461/293149/401450/283713/278684/386948 are
  cosmic-satellite classes surveyed in §Per-event evidence; 386948 alone
  sheds 1548.8 MeV of overclustered cosmics.

## Files

- toolkit `clus/src/NeutrinoKinematics.cxx` — decision block + probe + skip
- toolkit `clus/src/PRShowerFunctions.{h,cxx}` — `shower_start_is_track_continuation`
- toolkit `clus/inc/WireCellClus/{NeutrinoPatternBase,TaggerCheckNeutrino,TrackFitting,MultiAlgBlobClustering}.h`,
  `clus/src/{TaggerCheckNeutrino,MultiAlgBlobClustering}.cxx` — knobs + transport + PF skip
- toolkit `clus/test/doctest_clus_knob_defaults.cxx` — default pins
- toolkit `cfg/pgrapher/common/clus.jsonnet`, `cfg/pgrapher/experiment/sbnd/clus.jsonnet`,
  `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` — plumbing (key-suppressed)
- wcp-porting-img `sbnd_xin/run_pr_chain_batch.sh` — SBND_KINE_DROP_STRAY_SATELLITES /
  SBND_KINE_SAT_* / SBND_PF_DROP_STRAY_SATELLITES env block
