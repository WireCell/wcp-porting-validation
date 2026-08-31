# doc pr/123 — pass4_angle over-reach (the over-clustering round)

**Status: VALIDATED — `shower_pass4_prune_detached` (G=40, re-seed) +
`shower_pass4_track_guard_len=50` SBND PRODUCTION ON 2026-08-28 (owner
design decisions §0/§6 + "consistent with previous round" pre-authorization).
141-set marked showers: med qF1 0.887→0.935, Σ q_extra −45 % (the §17.6
metric no prior knob touched); 8 showers to 1.000; zero orphans beyond the
3 correctly-freed muons; 0 vertices moved; nusel identical. Bee pair
uploaded (OFF 24e81f0c / ON 94d3b955). Owner Bee verdict 2026-08-28: ALL
rows good; idx 13/14 OK (nu vertices wrong — vertex-bad class). Round 2
directed: the guard-freed muon must appear in the PF tree (§10).**

Owner directive 2026-08-28: proceed with the recommendation from the pr/121+122
close-out — the next round is `pass4_angle` over-reach, the largest measured
untouched defect (doc [pr/115](115_em-handscan-categorisation.md) §16.5: 48 %
of wrongly-held marks; §17.6: Σ q_extra `4.80e7` on the 20 over-clustered
141-set showers, untouched by every shipped knob; §17.9: first on the order of
work). Requirements consistent with previous rounds (pr/117–121 bar,
dual-manifest validation). The neutrino-vertex campaign is explicitly out of
scope this round (owner: settled for now).

## 0. The over-reach line — owner decision 2026-08-28

Asked before any tuning, per §17.5/§17.9 and the QC finding that both genuine
judgement splits (evt292524, evt58755) and part of a third (evt318769) turn on
this single definition. Options presented: (a) gap OR track-like, (b) body-gap
only, (c) track-like far stubs only, (d) current cone is the line.

**Owner selected (a) — gap OR track-like**: a `pass4_angle`-accepted member
counts as over-reach when EITHER

- **gap** — it is detached from the contiguous shower body: closest approach
  to the rest of the shower exceeds ≈25 cm (the code's own `close_shower_dis`
  scale), regardless of cone angle; OR
- **track-like** — it is track-like (pdg 13/211/2212 or MIP-flat dQ/dx)
  beyond the shower body.

Long contiguous EM showers stay legitimate at any distance. Under this rule
the two QC splits resolve: evt58755 (~26 cm internal gap) → over-clustered;
evt292524 (7 of 8 far members pdg 13/211) → over-clustered.

This definition is (1) the adjudication rule for scoring this round's census
and (2) the design target for any guard knob. The `emscan-*` label dirs are
records and are NOT rewritten (M13); the rule is applied at adjudication time.

## 1. The code site

`clus/src/NeutrinoShowerClustering.cxx`, `shower_clustering_with_nv_from_vertices`,
acceptance at `:2144-2147` (probe tag `pass4_angle` / `pass4_angle_divert`):

```
(angle_v1 < 25   && (pair_dis  < 80 cm  || close_shower_dis < 25 cm)) ||
(angle_v2 < 25   && (front_dis < 40 cm  || close_shower_dis < 25 cm)) ||
(angle_v1 < 12.5 && (pair_dis  < 120 cm || close_shower_dis < 40 cm)) ||
(angle_v2 < 12.5 && (front_dis < 80 cm  || close_shower_dis < 40 cm))
```

- `pair_dis` = segment's closest approach to the shower **start vertex**;
  `front_dis` (`tmp_shower_dis`) = to the start segment's front fit point;
  `close_shower_dis` = to the **shower body** (`shower_get_closest_dis`).
- angle_v1/v2 vs the shower axis from the two anchors; `angle_v2 > 30` early
  skip; associated-vertex veto (`dis1 > 25 cm && dis1 > 0.4·pair_dis`) when
  the segment's cluster is associated to a different vertex.
- **The defect shape**: the disjunction accepts on start-anchored distance
  alone (`pair_dis < 80 cm` at <25°) — nothing requires the segment to be
  near the shower's own body, and nothing looks at the segment's PID. The
  `close_shower_dis` terms are *looseners* (alternatives), not requirements.
- Note: doc 115 §16.5 quoted the (25°/80, 12.5°/130, 5°/200) cone for this
  site; that disjunction is actually the pr/117 **rival** gate at `:2182-2184`
  (and the pass-3 cone). The acceptance above is the authoritative one; this
  round's probe prints it term by term.

## 2. Probe (byte-neutral, this round's round-1 build)

`pr123_probe_pass4_geom` (`NeutrinoShowerClustering.cxx`, under the existing
`WCT_SHOWER_ABSORB_DEBUG` gate): one `SHOWER_ABSORB PASS4_GEOM` line per
ACCEPTED segment —

```
seg pdg len_cm med_dqdx_mip cur cur_nseg cur_len_cm owner divert
pair_dis_cm front_dis_cm body_dis_cm angle_v1 angle_v2 tier
```

`body_dis_cm` is the at-absorb-time gap to the accepting shower's body — the
guard-usable form of the owner rule's gap measure; `med_dqdx_mip` +` pdg` are
the track-likeness features; `tier` = which disjunct accepted (1–4).

## 3. Round plan

1. Probe build (byte-neutral): gates vs production arms on both manifests.
2. `pr123r1-dbgA` (98) + `pr123r1-dbg141` (141) probe arms;
   `scripts/pr123_pass4_census.py` — labeled scatter + guard sweep per the
   owner rule (gap / track-like / OR variants), collateral counted on
   TARGET members and unlabeled absorbs.
3. Knob(s) per census (default OFF), OFF gates, ON adjudication
   (`em117_score.py` both label sets, `owned_census.py` both manifests —
   net owned ≥ 0, zero new orphan events, `pr90_movers.py`), Bee A/B,
   owner review, flip on validation ("consistent with previous round").

Concurrency note: coordinated with the doc-84/MCS session throughout
(message exchanges 2026-08-28): builds interleaved in agreed windows, its
`work-d84r1-flipchk98-*` arms adopted as the post-flip 98-set baseline, and
commits on both sides staged hunk-selectively (its round-2 default-OFF work
shares PRShower/TCN/NPB/doctest/jsonnet with this round's knobs).

## 4. Pre-census — the target population, from existing artifacts (no new arms)

**Exposure** (pr/121 probe arms, `SHOWER_ABSORB DIRECT site=pass4_angle*`
over `stdout.log`): 1037 direct + 18 divert absorbs across the 239 events;
118 events have at least one (13+17 of 98, 38+50 of 141).

**Label-side tally** (raw `marks_by_shower` × `marks_detail.absorbed_by`;
note doc 115 §16.5's "72/48 %" was a member-expanded tally — at raw mark
level the counts are):

| set | total OUT | OUT by pass4_angle | total IN | IN by pass4_angle |
|---|---|---|---|---|
| emscan-0827 (98) | 29 | **13** (3 events: 142421, 269774, 314838) | 246 | 89 |
| emscan-0828-agent5 (141) | 83 | **41** (12 events) | 59 | 7 |

pass4_angle is 45 % / 49 % of all OUT marks — the ~48 % share holds on the
out-of-sample set; `pass3_cone` is second (20 marks, 141-set).

**The labeled-bad geometry already vindicates the two-prong rule** (offline
`dist`/`angle`/`pdg` from the labels):

- *track-like prong needed*: evt171572 seg 10008 (125 cm, pdg 13, 3.2°,
  14.8 cm from start — contiguous, on-axis, a muon) and evt393505 seg 15013
  (108 cm, pdg 13, 10.5°) — no gap or angle cut can touch these.
- *gap prong needed*: evt278420 (7 segs at 100–125 cm, tier 2, mixed pdg) —
  far detached stubs down the narrow cone.
- track-ish pdg (13/211/2212) dominates the OUT marks: 9/13 (98-set),
  ~28/41 (141-set).
- Many OUT marks sit at **offline** angles 84–150° with `tier=None`
  (179369, 283515, 286655) — impossible for the offline cone mirror, so the
  at-absorb-time geometry (axis before later absorbs moved it, the
  `close_shower_dis` looseners) is what actually admitted them. The offline
  scatter cannot be the tuning input; the PASS4_GEOM probe exists for this.

## 5. The measurement — no absorb-time threshold separates; the final body does

**Arms** (probe builds byte-neutral, gates in §7): `work-pr123r1-dbgA2-*`
(98-set) + `work-pr123r1-dbg141v2-*` (141-set), binary with the PASS4_GEOM v2
probe (`snap_dis` = gap to the pass-entry membership snapshot, chain-immune).

**Per-absorb census** (`pr123_pass4_census.py`; 1582 accepted pass4 absorbs
over 239 events; 54 OUT-marked = labeled-bad, 377 TARGET = labeled-good
members of marked showers):

- `body_dis` (gap to the current body at absorb time) does not separate:
  OUT med 8.5 / TARGET med 7.5 — because absorbs CHAIN: the first far stub
  lands at gap ~24-63, every follower sits near the previous stub
  (evt278420: 23.9 then 5-16). The current body launders the gap.
- The chain-immune `snap_dis` un-launders OUT (med 34.0) — **but TARGET
  moves with it** (med 29.5): legitimate EM growth also proceeds by far
  fragment chains from a tiny pass-entry stem. snap25 would kill 208/377
  labeled-good members. Every per-absorb rule fails the same way
  (guard sweep table in the census output; TSV `pr123-pass4-census.tsv`).
- The one clean per-absorb rule is the **track prong**: track-like segments
  longer than 50 cm are never labeled-good (3 hits in 239 events: the two
  labeled muons 171572 125 cm / 393505 108 cm + one 83 cm unlabeled
  muon-like; zero TARGET).

**Final-body prune scan** (`pr123_prune_scan.py`: single-linkage components
of each marked shower's FINAL membership at gap G; component without the
start segment = detached; dump fit points, both manifests):

| G | OUT pruned/66 | scan-target collateral | charge (bad : good) |
|---|---|---|---|
| 25 cm | 35 | 51 segs, 1.34e7 | ~1:1 |
| 30 cm | 31 | 35 segs, 1.14e7 | ~1:1 |
| 40 cm | 25 | 13 segs, 2.09e6 | ~4:1 (8.75e6 : 2.09e6) |

G=40 kills across 9 events (282979 **10/10 marks**, 74326 **6/6**, 318769's
pair, 168432, 179048, 286681, 395597, 499423, 71872); collateral concentrates
in 5 events of which 2 are themselves defective showers (54332's fake seed,
415278 wrong-owner). Restricting components to pass4-absorbed members loses
the walk-followers and halves the kill — the prune is a general
detached-component cleanup, not a pass4-only guard.

**Residual classes this round does not touch**: contiguous far chains into
big showers (269774, 278420 at G≥25's linkage), the root-is-wrong
recognition class (489327, 69232, 54332's 16014 — doc pr/122), the π⁰
collinear split (314838, 142421 — owner-gated), and 179369's 105-155 cm
backward cluster (survives G=40).

## 6. Owner decisions 2026-08-28 (design gate, asked before implementation)

1. **Prune gap G=40 cm** (conservative first round; G can come down later).
2. **Pruned components RE-SEED as their own showers** (rooted at the member
   nearest the kept body, conn 3 within 80 cm of the start vertex else 4 —
   the in_other_clusters typing). No orphans: the zero-new-orphans bar holds.
3. **Track guard at len>50 cm** (track-like = pdg 13/211/2212 or median
   dQ/dx < 1.3 MIP).

## 7. The knobs (both DEFAULT OFF)

- `shower_pass4_prune_detached` (bool, false) + `shower_pass4_prune_gap`
  (40 cm): the post-family prune pass in `shower_clustering_with_nv`
  (after dedup/detach_track_stem/ghost, before the pr/119 census seat, the
  hadronic tag and the π⁰ finders). Members with no usable fit cloud are
  immune. Removal via the new `Shower::detach_member_set` (PRShower.{h,cxx},
  forked BY DUPLICATION from `detach_track_prefix`; production method
  untouched). Probe: `SHOWER_ABSORB PASS4_PRUNE` per component.
- `shower_pass4_track_guard_len` (cm, 0=off): decline at the pass4_angle
  accept branch. Probe: `SHOWER_ABSORB PASS4_TRACK_GUARD`.
- Seats: `NeutrinoPatternBase.h` / `TaggerCheckNeutrino.{h,cxx}` /
  `doctest_clus_knob_defaults.cxx` (2472 assertions) /
  `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` decl+suppression /
  runner env `SBND_SHOWER_PASS4_PRUNE`, `SBND_SHOWER_PASS4_PRUNE_GAP`,
  `SBND_SHOWER_PASS4_TRACK_GUARD_LEN`.
- Compiled-config proofs: off ⇒ cmp rc=0 vs git-HEAD compile (full
  pipeline TLA); on ⇒ keys present, gap suppressed at default.

## 8. Validation — dual manifest, the knobs do the labeled thing

**Firing set**: prune fires on 27/98 + 29/141 events (56 of 239); the track
guard declines exactly 3 absorbs (all 141-set: 171572, 393505, 105074).
`mabc-pr.zip` is the ONLY archive that changes anywhere; `nusel-evt*.tsv`
and `pctree-pr-*.tar.gz` byte-identical on all 239 events; **main vertex
identical on all 239** (dump-level check, 0 moved).

**Ownership census** (`owned_census.py`, OFF→ON):

```
98 set : 27 events changed; owned 5632 -> 5632 (net 0); 6 leading losses, Sum 50.4 MeV
141 set: 29 events changed; owned 3763 -> 3760 (net -3); 15 leading losses, Sum 1149.7 MeV
```

The net −3 is exactly the track guard's three declined muon segments —
171572's two OUT-marked segs and 393505's OUT-marked 108 cm muon — tracks
correctly ceasing to be EM members (the owner's chosen disposition). Every
prune re-seeds with zero ownership loss.

**Scores vs labels** (`em117_score.py --cross-run`, 57 marked rows/141-set +
33/98-set; TSVs `pr123-{141,98}-score-{off,on}.tsv`):

| | 141-set OFF | 141-set ON |
|---|---|---|
| median qF1 (55 rows) | 0.887 | **0.935** |
| Σ q_extra | 4.861e7 | **2.690e7 (−45 %)** |
| Σ q_miss | 2.377e7 | 2.454e7 (+3 %) |

Σ q_extra is the §17.6 headline metric that every previously shipped knob
left untouched. Per-row: **12 gainers** — 171572 0.096→**1.000**, 393505
0.273→**1.000**, 74326 0.396→**1.000**, 282979 0.541→**1.000**, 71872,
286681, 179048, 499423, 168432 all →1.000, 395597 +0.052, 318769 (QC event)
+0.034, **348471 0.895→0.912** (pr/121 follow-on: sheds only the unlabeled
>40 cm companion 31011; the rescued 63050 side stays). **4 collateral rows**
(both manifests): 54332-122091 0.890→0.731 and 281325 0.916→0.766 (each
loses satellites at 56/40+ cm gap — over-reach BY the owner's definition;
the scan targets predate it → Bee adjudication idx 13/14), 293149 −0.015,
415278 −0.005. Both wrong-owner ledger events (76346, 415278) get their
detached components separated.

**Physics scalars**: nusel/BDT byte-identical everywhere. Σ kine_energy_particle
moves on firing events only (net −1.7 GeV over 141 + −0.4 GeV over 98):
freed muon charge leaving EM sums and conn-3-typed re-seeds excluded by the
existing kine_energy_included rule — the designed consequence of the owner's
re-seed disposition, stated here explicitly.

**Bee A/B** (`bee/pr123r1/`, 19 events, annotated index):
- OFF (production) `24e81f0c-4ea4-471b-b2f5-6e724985e920`
- ON (prune G=40 + track guard 50) `94d3b955-6617-4ed4-8263-c8351dc0370e`

**Flip**: `shower_pass4_prune_detached = true` + `shower_pass4_track_guard_len
= 50` in the SBND tcn knobs ("consistent with previous round"
pre-authorization + the owner's three design answers, 2026-08-28); compiled
flipped config byte-identical to the ON-TLA compile (cmp rc=0);
flip-equivalence arms hash-gated in §9.

## 9. Gate ledger

| # | gate | arms | result |
|---|---|---|---|
| 1 | probe build == published post-flip baseline (98) | `d84r1-flipchk98` vs `pr123r1-dbgA` | **PASS 196/196** |
| 2 | 141 post-flip baseline self-consistent | `pr123r1-base141` vs `pr123r1-dbg141` | **PASS 282/282** |
| 3 | probe v2 byte-neutral (98) | `d84r1-flipchk98` vs `pr123r1-dbgA2` | **PASS 196/196** |
| 4 | probe v2 byte-neutral (141) | `base141` vs `dbg141v2` | **PASS 282/282** |
| 5 | knob build, knob off (98) | `d84r1-flipchk98` vs `pr123r1-off1` | **PASS 196/196** |
| 6 | knob build, knob off (141) | `base141` vs `off141` | **PASS 282/282** |
| 7 | concurrent 20:14 lib output-neutral + knob-off (141) | `base141` vs `off141b` | **PASS 282/282** |
| 8 | compiled config off | git-HEAD vs worktree, full-pipeline TLA | **PASS** (cmp rc=0) |
| 9 | compiled config on | prune+guard TLAs | **PASS** (keys present, gap suppressed at default) |
| 10 | ON vs OFF (98) | `off1` vs `on1` | 27 events differ = the firing set, mabc only |
| 11 | ON vs OFF (141) | `off141b` vs `on141` | 30 archives differ = firing set + track guard, mabc only |
| 12 | flipped compile == ON compile | wcsonnet outputs | **PASS** (cmp rc=0) |
| 13 | flip-equivalence arms | post-flip cfg, no env, both manifests vs `on1`/`on141` | **PASS 196/196 + 282/282** (98-set nuecc48 stitched from `flipchk`+`flipchk2`+`flipchk3` — two background-task kills truncated 12 event dirs mid-write; every damaged dir excluded by rc/zip validity check, each event gated exactly once) |

**Binary provenance** (the doc 121 §4 pattern, again not clean and again
carried by gates): the shared `local/lib` was rebuilt at 19:24 (MCS doc-84
flip work, containing this round's probe v1 — my source predates it), 19:59
(this round's knob build), and 20:14 (MCS doc-84 round-2 default-OFF work).
`off141b` (20:14 binary) vs `base141` (19:24) PASS 282/282 and the MCS
session's own OFF gate (100/100 on three samples at report time) attest the
20:14 build; the 141-set ON/OFF comparison (`off141b` vs `on141`) is
entirely within the 20:14 binary. The 98-set `on1` ran the 19:59 binary;
gate 13's flipchk (20:14) vs `on1` spans the rebuild and carries the proof.

## 10. Round 2 — the guard-freed muon must not vanish from the PF tree

**Owner Bee verdict 2026-08-28 (round 1)**: ALL 19 rows good; idx 13/14 OK —
their nu vertices are wrong (vertex-bad class), so the prunes stand. One
follow-up directed: on 171572 the muon is *correctly* excluded from the
shower but is LOST from the PF tree — "it should not be lost".

**Root cause — three independent blockers** (all verified in code + the ON
dump):

1. The pr/93-r4 PF orphan machinery (`pf_orphan_confident_track`, SBND ON)
   and the pr/65 orphan audit are **main-cluster-scoped** (prototype
   `NeutrinoID.cxx:1488` parity). The muon lives in cluster 10; the main
   vertex in cluster 83 → invisible to both (audit prints "0 unclaimed").
2. `segment_orphan_confident_track` requires `particle_score < 1.0`; the
   muon carries the **score-100 sentinel** (rule-assigned PID) → refused.
3. `kine_count_orphan_tracks` (SBND ON) has the same cluster filter and
   predicate → the ~390 MeV also left `kine_reco_Enu` (§8's net kine drop).

Legacy only ever displayed this muon as a member of the fake shower; the
track guard freed it into a class (cross-cluster + sentinel-scored) the
orphan machinery was never scoped for.

**Exposure census** (offline, ON vs OFF dumps, both manifests): exactly
**2** guard-freed-and-lost segments — 171572 seg 10008 (125.1 cm µ) and
393505 seg 15013 (108.5 cm µ). 105074's declined track was re-owned by
another shower downstream (not lost). Critically, **120 pre-existing**
unclaimed ≥50 cm track-pdg segments sit in non-main clusters across 96
events — largely cosmics, which is exactly why the PF orphan scope is
main-cluster-only. A blanket any-cluster/sentinel relaxation would fabricate
~120 cosmic PF roots and pour their KE into kine_reco_Enu. **The fix must be
the guard's own decline set, nothing wider.**

**Design (round 2, both DEFAULT OFF):**

- `SegmentFlags::kPass4GuardFreed` (1<<7): set by the track guard at decline
  time (kMuonStemGuard precedent — the raw flags word is never serialised,
  so the bit alone is inert in every output).
- `pf_orphan_guard_freed` (BeePFConfig): after the orphan audit chain, a
  flagged-and-unclaimed segment gets a root PF node (pr/93 node shape,
  dirsign/fit filters, KeepMC floors). Log `pr123 pf-orphan-guard-freed: EMIT`.
- `kine_count_guard_freed` (TCN → pattern_algos): the kine twin — flagged,
  unclaimed by BFS and showers → `push_segment_kine` (the pr/93 principle:
  PF and kine describe the same particle set). Log
  `kine_count_guard_freed: COUNT`.
- Seats: `PRSegment.h` / `NeutrinoShowerClustering.cxx` (flag set) /
  `MultiAlgBlobClustering.{h,cxx}` / `NeutrinoKinematics.cxx` /
  `NeutrinoPatternBase.h` / `TaggerCheckNeutrino.{h,cxx}` / doctest pin /
  `wct-pr-perevt.jsonnet` + `clus.jsonnet` decl/threading/suppression /
  runner env `SBND_PF_ORPHAN_GUARD_FREED`, `SBND_KINE_COUNT_GUARD_FREED`.

Residual noted: 171572 seg 10005 (10 cm, pdg-11, flag_shower, score-100)
also left the fake shower and stays unowned — below any track threshold, a
small EM stub; not addressed here.

**Round-2 validation (arms `work-pr123r1-r2*`, binary 21:03 on toolkit
`9577592e`):**

| # | gate | arms | result |
|---|---|---|---|
| R2-1 | knob-off == NEW production baseline (98) | `d84r2-prod98` vs `r2off` | **PASS 196/196** |
| R2-2 | knob-off + inert flag, spanning the 20:47 bridge fix AND the 21:03 build, incl. all 3 flag-writing events (141) | `flip141` vs `r2off141pin` (peer knobs env-pinned OFF) | **PASS 282/282** |
| R2-3 | compiled config off / on | keys absent / present | **PASS** |
| R2-4 | ON vs OFF (141) | `r2off141` vs `r2on141` | **exactly 171572 + 393505**, mabc only |
| R2-5 | ON vs OFF (mcp1k / 98-set) | — | byte-identical (no flagged events; 0 expected, 0 seen) |
| R2-6 | flip-equivalence | post-flip cfg no env: `r2flip141` vs `r2on141`, `r2flip98` vs `r2off` | **PASS 282/282 + 196/196** |

Effect on the 2 events: `pr123 pf-orphan-guard-freed: EMIT root seg=10008
... 304.75 MeV` / `seg=15013 ... 268.70 MeV`; `kine_count_guard_freed:
COUNT` both; Σ kine_energy_particle 357.3→662.1 and 460.2→728.9; vertices
identical; nusel TSVs byte-identical; shower membership identical (round-1
fix untouched). Informational: the fresh no-env `r2off141` vs the pinned
arm shows the doc-84 r2 knobs move 3 141-set events (321767, 292524,
499577 — mabc only), recorded for that round's ledger.

**Flip**: `pf_orphan_guard_freed = true` + `kine_count_guard_freed = true`
SBND PRODUCTION ON 2026-08-28 (owner: "it should not be lost in the PF
tree"; upload authorized "once validated"). Bee pair (`bee/pr123r2/`):
OFF `caad3e29-e997-433f-bdd7-37c7e89c8fc9` / ON
`e1da72ac-2d54-4895-8460-083d509c15d9`, 2 events (owner-directed upload).

**Round 2.1 — owner correction on the r2 pair**: the freed track is not
connected to the vertex directly, so it must not hang at root — "contained
in a neutron, and then muon". ON-behavior corrected in place (the pf/F2
precedent, no new knob; toolkit `b08353f4`): the emission now builds a
pseudo-NEUTRON carrier (`append_pseudo_shower` convention, pdg 2112, main
vertex → track near end) with the track as its single leaf — ν → n → µ.
Validation (`work-pr123r1-r21flip141-*`, binary 21:30): mcp1k 104/104
byte-identical, divergence exactly 171572 + 393505 (mabc only), mc.json
verified ν → neutron(304.8/268.7 MeV) → µ⁻. Corrected AFTER set:
`43ba1e1c-38f6-4ece-959c-bf883e0b83e8` (BEFORE unchanged: `caad3e29`).
The 141-set production baseline advances to `work-pr123r1-r21flip141-*`
(mcp1k content-identical to `r2flip141`).
`work-pr123r1-r2off141`+`r2flip141` succeed `flip141` as the 141-set
production baseline; the 98-set baseline is doc-84's `work-d84r2-prod98-*`
(+ this flip = `r2flip98`).

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# arms (wcbuild + freshness proof + wcdoctest-clus each build; launcher
# committed as scripts/pr123_arms.sh)
/home/xqian/tmp/pr123_arms.sh 98  dbgA2    1    # probe v2 (PASS4_GEOM + snap_dis)
/home/xqian/tmp/pr123_arms.sh 141 base141  0    # post-MCS-flip 141 baseline
/home/xqian/tmp/pr123_arms.sh 141 dbg141v2 1
/home/xqian/tmp/pr123_arms.sh 98  off1     0    # knob build, knobs off
/home/xqian/tmp/pr123_arms.sh 141 off141b  0
/home/xqian/tmp/pr123_arms.sh 98  on1  1 SBND_SHOWER_PASS4_PRUNE=1 SBND_SHOWER_PASS4_TRACK_GUARD_LEN=50
/home/xqian/tmp/pr123_arms.sh 141 on141 1 SBND_SHOWER_PASS4_PRUNE=1 SBND_SHOWER_PASS4_TRACK_GUARD_LEN=50
/home/xqian/tmp/pr123_arms.sh 98  flipchk 0    # post-flip cfg, no env
/home/xqian/tmp/pr123_arms.sh 141 flip141 0

# per-absorb census + guard sweep
./scripts/pr123_pass4_census.py --tsv docs/pr/pr123-pass4-census.tsv \
    'work-pr123r1-dbgA2-*' 'work-pr123r1-dbg141v2-*'
# final-body prune scan (G sweep)
./scripts/pr123_prune_scan.py 'work-pr123r1-dbgA2-*' 'work-pr123r1-dbg141v2-*'
# ownership + scores
docs/pr/pr116-bulk/scripts/owned_census.py 'work-pr123r1-off141b-*' 'work-pr123r1-on141-*'
cd em_display && ./prep_pr117.py --tag 123on1 work-pr123r1-on1-{mcp1k,mcp2k,ncpi0,nuecc48} \
  && ./prep_pr121.py --tag 123on141 work-pr123r1-on141-{mcp1k,mcp2k} \
  && ./em117_score.py --tag emscan-0828-agent5 --manifest em114c-123on141-manifest.tsv \
       --prepdir emprep-123on141 --cross-run
```
