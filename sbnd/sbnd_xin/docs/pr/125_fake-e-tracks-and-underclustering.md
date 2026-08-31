# doc pr/125 — owner verdicts: fake electrons → tracks (pass3 guard flip) + under-clustering (37112, 69314)

**Status: IN PROGRESS** (opened 2026-08-29). Follow-on to doc pr/124.

Owner directive (2026-08-29, after scanning the pr/124 Bee pairs):

> "For the following event, evt 94392 (305 MeV electron), evt 52693 (182 MeV
> electron), evt 77328 (180 MeV electron), evt 173819 (301 MeV electron),
> they should not be ided as electron, but track object, check dQ/dx. I am
> not sure why they were treated as electrons. We had similar improvements
> in the past and logged in md file, can you make the improvements, commit
> and push."
>
> "evt 37112, the gamma 549 MeV and the proton 469 MeV should be one EM
> shower, instead of two. They are connected right? In this case, it should
> not be separated as two shower. evt 69314, the [~0 MeV] connected electron
> should be one EM shower, instead of a cascade of electrons?"

Two follow-up decisions taken interactively (AskUserQuestion, on record):

1. **pass3 guard**: measure a dQ/dx qualifier first; if it does NOT separate
   415278 from the four fixed events, **"Flip anyway"** — flip
   `shower_pass3_cone_guard_len=15` accepting 415278's reshuffle as the
   adjudicated cost. Either way the guard flips this round (this resolves
   doc pr/124 §C.3 as option 1, with a dQ/dx refinement attempt first).
2. **37112 scope**: **"One shower, everything"** — γ 549 + the full 469 MeV
   proton content merge into one EM shower, *including* the 12-seg backward
   component that the pr/124 gap-band prune (now ON) detaches; the prune
   should not fire on that component (they are connected).

## Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# Front-1 dQ/dx scan over pass3_cone absorbs (declined-candidate features):
python3 scripts/pr125_p3guard_dqdx.py            # writes docs/pr/pr125-p3guard-dqdx.tsv
# 37112/69314 shower anatomy + manifest-wide merge-candidate scan:
python3 scripts/pr125_merge_anatomy.py           # writes docs/pr/pr125-merge-anatomy.tsv
# Existing-arm inputs (no new arms for phase 1):
#   141-set OFF/probe: work-pr124r1-dbg141v2-mcp2k     ON-guard: work-pr124r1-onC141-mcp2k
#   98-set  OFF/probe: work-pr124r1-dbgv2-{ncpi0,nuecc48,mcp1k}, prod point work-pr124r1-flipA98-*
```

## 1. Where the six events stand (from pr/124 data, verified in calib dumps)

### 1.1 The four fake electrons = the pr/124 front-C fix set

All four are mcp2k (141-set). The implemented-but-OFF
`shower_pass3_cone_guard_len=15` (toolkit `a9545660`,
`NeutrinoShowerClustering.cxx` pass3_cone site) fixes all four
(`work-pr124r1-dbg141v2` OFF vs `work-pr124r1-onC141` ON):

| evt | OFF leading shower | ON | label (emscan-0828-agent5) |
|---|---|---|---|
| 94392 | pdg11 **305.6 MeV**, 5 seg, 103.5 cm | e 45.5 MeV + **µ 98.4 MeV released** (29.8 cm) | `over-clustered` — "77 cm of a DIFFERENT cluster, both pieces PID'd as tracks" |
| 52693 | pdg11 **183.0 MeV**, 14 seg, 91.2 cm | e 153.0 MeV + **µ 108.9 MeV released** (34.3 cm) | `vertex-bad` — "one straight through-going track ~375 cm split into 232 cm muon + 183 MeV shower" |
| 77328 | pdg11 **180.6 MeV**, 2 seg, 28.1 cm | e 38.3 MeV + **p 153.8 MeV released** (16.5 cm) | `correct` but "a proton at the vertex mis-PIDed as a 180 MeV EM shower is a live reading… the display shows no dQ/dx" |
| 173819 | pdg11 **301.9 MeV**, 2 seg, 42.2 cm | e 10.5 MeV + **p 250.2 MeV re-rooted** (37.8 cm) | `not an EM shower` — "7 MeV/cm ≈ 3× MIP, a stopping-proton profile" |

Mechanism ("why they were treated as electrons"): each event's pdg-11 stem
seeded a shower, then **pass3_cone absorbed an adjacent track-PID'd segment**
(µ/µ/p/p) into it; the absorbed track's charge is then reported as electron
energy. The guard declines those absorbs (len>15 cm && |pdg|∈{13,211,2212})
and `kPass4GuardFreed` re-roots the track in the PF tree. Cost row: 415278 —
three declined tracks (π 36.4, µ 56.3, µ 22.1 cm) reshuffle between the
event's two *labeled* showers (qF1 0.959→0.884 AND 0.976→0.910); no length
threshold separates (doc pr/124 §C).

Residual after the guard: the pdg-11 stems themselves (45.5/153.0/38.3/10.5
MeV) stay electrons. Stem med dQ/dx from the seed census: 77328/16016
2.13 MIP, 173819/14028 2.02 MIP (γ-conversion band or proton-like) —
tracked in §2 as a secondary measurement, not a promised fix.

### 1.2 37112 (ncpi0, 98-set) — under-clustering + a prune wrong-fire

OFF (pre-pr/124): shw2 pdg11 549.3 MeV (21 seg, 119.9 cm) + shw3 pdg2212
469.8 MeV (15 seg, 33.4 cm) — the owner's γ/p pair. Production (flipA98):
the gap-band tier-2 prune detaches a 12-seg backward comp (168.7°) from shw3
(469.8→129.9 MeV) — Bee pr124r1 idx 12, backward class (unlabeled exposure,
now owner-adjudicated **wrong**: the component is connected). Merging shw2+shw3
is blocked by two hard-coded rules in `merge_shower_fragments`
(`NeutrinoShowerClustering.cxx:3251/:3274` EM↔EM only; `:3282` main-vertex
γ-pair guard) and `examine_merge_showers`' conn1×conn2 <10° gate.

### 1.3 69314 (nuecc48, 98-set) — electron cascade

Owner: connected electron cascade should be ONE shower. No pr/124 knob
touches it (verified: only a 0.7 cm stub reparents between dbgv2 and
flipA98/onC98). Blocker: `merge_shower_fragments` runs once with no chaining
(`:3252/:3273`, single call site `:5955`), so a cascade A←B←C collapses at
most one link; plus conn-3/4 fragments and the absorber-size gate `!(len2<len1)`.

## 2. Plan of record

- **K1** (only if dQ/dx separates): `shower_pass3_cone_guard_dqdx` — dQ/dx
  term in the pass3 guard's track test (pass4-guard fallback shape,
  `median dQ/dx < 1.3×MIP ⇒ track-like`, `:2296-2323`). Else flip the
  existing `shower_pass3_cone_guard_len=15` plain (owner: "Flip anyway").
- **K2** `shower_pass4_prune2_cont`: tier-2 prune exemption — charge
  continuity sampled along the body↔component connector (pr/118 continuity
  style); continuous ⇒ keep. Must leave 406125-class pr/124 wins pruned.
- **K3** `shower_merge_relax_track_frag`: allow merge_shower_fragments to
  absorb a non-EM fragment under strict continuity (T1 gates + charge
  continuity), incl. the matching relaxation of the `:3282` γ-pair guard.
  Hard bar: no π⁰ γ-pair may merge anywhere on the ncpi0 manifest.
- **K4** `shower_merge_relax_iterate`: bounded fixpoint iteration of
  merge_shower_fragments so cascades collapse.

All DEFAULT OFF; wct-knob 7-seat recipe; dual-manifest byte-identical OFF
gates at final HEAD; probe-armed ON arms scored (em117), movers (vtx105),
owned census, nusel; Bee A/B pair; flips per validation + owner decisions
above. Build/commit windows serialized with the doc-84 round-4 session
(shared seats: TaggerCheckNeutrino.{h,cxx}, doctest, wct-pr-perevt.jsonnet,
run_pr_chain_batch.sh).

## 3. Measurements (all offline, from pr/124 arms — no new arms)

### 3.1 Front 1: dQ/dx does NOT separate 415278 → plain flip at len=15

`scripts/pr125_p3guard_dqdx.py` → `docs/pr/pr125-p3guard-dqdx.tsv`.
982 track-pdg pass3_cone absorbs across both manifests; **15** in the
guard's len>15 decline set. Sorted by median dQ/dx (MIP units, dump muon
tail):

| class | evt/seg | pdg | len | mdqdx |
|---|---|---|---|---|
| COST | 415278/23022 | 211 | 36.4 | 0.475 |
| COST | 415278/23047 | 13 | 56.3 | 0.606 |
| FIX | 94392/45029 | 13 | 29.8 | 0.815 |
| FIX | 52693/29008 | 13 | 34.3 | 0.906 |
| FIX | 94392/45030 | 13 | 46.9 | 1.080 |
| COST | 415278/24072 | 13 | 22.1 | **1.178** |
| FIX | 173819/38038 | 2212 | 37.8 | 2.549 |
| FIX | 77328/36012 | 2212 | 16.5 | 2.776 |

415278's segs (0.475–1.178) bracket the FIX muons (0.815–1.080): 24072 sits
ABOVE every FIX muon. The protons separate trivially; the µ-vs-µ split does
not exist in dQ/dx. **Per the owner's pre-decision ("Flip anyway"), the
guard flips plain at len=15 — no dQ/dx qualifier knob (K1 dead).** Other
decline-set exposure: 396222 (3 segs), 137238 (2), 176502, 175896 (the
pr/124 measured no-op), i.e. 7 events total fire.

### 3.2 37112 anatomy: the three numbers that settle it

OFF (dbgv2-ncpi0): γ shw 67048 (pdg 11, 549.3 MeV, conn 2, sv 84104) and
proton-typed shw 9008 (pdg 2212, 206.3 MeV kine_charge — the owner's "469"
is the Bee display value — 15 seg, conn 2, **same sv 84104**, non-main;
main vertex is 84097). Cloud gaps: 9008-stem↔γ **1.28 cm**; the pr/124
tier-2 pruned 12-seg comp ↔ its own kept stem **28.8 cm** (why the prune
fired, correctly by its own metric) but ↔ γ **3.81 cm**. The whole complex
is one connected EM object; the prune measured isolation only within its
own shower.

**Pass-order dividend**: `merge_shower_fragments` (:5959) runs BEFORE the
tier-2 prune (:6482). Merging 9008 into 67048 first ⇒ the prune's
union-find then sees the comp contiguous with the merged body and keeps it.
One knob (K3) delivers "one shower, everything"; the separate prune
exemption (K2) is unnecessary and was dropped.

### 3.3 K3 gate: shared non-main vertex + gap<6 fires on 37112 alone

`scripts/pr125_merge_anatomy.py pairs` → `docs/pr/pr125-pairs.tsv`
(2460 pairs, 362 with gap<6, 43 track-typed frags at gap<6). Track-typed
frags at gap<6 are dominated by main-vertex conn-1 pairs (a genuine
primary track next to a primary shower — e.g. 396222's 318 cm π at 1.92 cm
from a 2.9 GeV shower, 388's 776 MeV π: must never merge). The
`shared start vertex && vertex != main && gap < 6 cm && track-typed`
conjunction fires **only on 37112** across both manifests; next candidates
sit at 17.4+ cm (287830) and 29.8+ (287654), and 54332's owner-adjudicated
satellites at 56+. Knob: `shower_samevtx_track_absorb` (+gap 6 cm, +frag
len cap 50 cm; 9008 is 33.4).

### 3.4 69314 anatomy + K5 gate: vertex-connected satellite absorb

69314 (νe CC) carries **38 pdg-11 PF entries, 28 below 5 MeV**: the main
68.9 MeV electron (shw 3014) plus ~24 crumb showers (0.06–4.5 MeV, conn
2/3), scattered at 17–35 cm cloud gaps — beyond every merge gate; radius
is the wrong metric (a 30–40 cm ball has π⁰/OUT collateral:
`pr125-satellites.tsv`, 2 OUT-marks + 5 π⁰-paired at R≤40).

The owner's word is "connected", and connectivity is measurable: satellites
whose START VERTEX is a vertex OF the big shower's own member chain
(69314: 40029/44033/45034 attach at 3014's start vertex 3001, etc.).
Manifest-wide (vertex-connected variant, E<10 MeV, host EM E>20):
1474 satellites across 211/239 events — **31 IN-marked (q_miss
recoveries), 1 OUT-marked, 5 π⁰-paired**; the single OUT (168432) and none
of 69314's are conn-4, so restricting to **conn 2/3** leaves
**31 IN / 0 OUT**. The 5 π⁰-paired (junk low-mass pairings, e.g. 7 MeV
partner to a 3.5 GeV shower) cannot be exempted at absorb time (π⁰
pairing runs later); they are watched explicitly in validation (§4).
Knob: `shower_satellite_absorb` (satellite E<10 MeV, |pdg|=11, conn 2/3;
host EM, E>20 MeV, satellite's start vertex in host's vertex set).

## 4. Implementation + validation

Knobs (all DEFAULT OFF, wct-knob 7 seats each): `shower_samevtx_track_absorb`
/ `shower_samevtx_absorb_gap` (6 cm) / `shower_samevtx_absorb_max_len`
(50 cm); `shower_satellite_absorb` / `shower_satellite_absorb_max_mev` (10)
/ `shower_satellite_absorb_host_mev` (20). Two dedicated passes in
`NeutrinoShowerClustering.cxx` after the dedup pass and before
detach/ghost/prunes/π⁰ finders; probe lines `SHOWER_MERGE
tag=samevtx_absorb|satellite_absorb` under pr91_merge_dbg. Compiled-config
proofs: off ⇒ byte-identical to HEAD (a8cbfa4a, incl. the doc-84 r4 flip);
on ⇒ all three keys emitted. wcdoctest-clus 2518/2518.

Peer coordination: doc-84 r4 flip (a8cbfa4a) landed mid-round; my gate arms
pin its three flipped params back to legacy
(`SBND_LONG_MUON_CATHODE_BRIDGE_LEVER=5 _TRACK_PARTNER=0 _SHORT_GAP=0`) so
the pr/124 flipA baselines stay valid; the final production-candidate arm
runs unpinned (both rounds' flips together).

### 4.1 Implementation iterations (three measured corrections)

1. **Vertex identity**: VertexPtr comparison misses conn-3 satellites (same
   graph vertex, different object — 5/7 missed in 69314); a
   find_vertices(start_seg) retry missed ALL (the attach vertex is not an
   endpoint of the satellite's own segment). Final: match by the dump
   display id `cluster_id*1000 + graph_index` (PrDisplayDump convention =
   what the offline scans joined on). Toolkit `67a55f1f`/`ea031faf`.
2. **Pass placement** (measured twice): before the pass4 prunes, the tier-2
   union-find re-detached 5/7 absorbed satellites; before
   `shower_hadronic_tag`, the tagger re-evaluated the ENLARGED showers and
   flipped 22 showers >15 MeV to pdg 211 — **including 69314's own primary
   electron**. Final order: prunes → hadronic tag → samevtx absorb →
   satellite absorb → `recompute_shower_kine_charge_final` → π⁰ finders.
   Post-fix: 0 pdg flips >15 MeV. Toolkit `dc4ab6f8`.
3. **samevtx min_len floor 5 cm** (knobbed): without it the pass absorbed 8
   sub-1-cm track-typed crumbs (satellite-class, and their absorb re-voted
   host PIDs). With it the fire set is exactly the measured gate: 2 fires,
   both 37112.

### 4.2 Round-d validation (final binary, both manifests)

- **OFF gates**: dbg98d/dbg141d vs work-pr124r1-flipA{98,141}: **PASS
  478/478 archives byte-identical** (28+34+38+96+104+178). Same result for
  rounds b and c on their binaries. Peer pins (doc-84 r4 legacy:
  `LEVER=5 TRACK_PARTNER=0 SHORT_GAP=0`) verified equivalent: the unpinned
  dbgP arms are byte-identical to the pinned ones 478/478 — **the doc-84 r4
  flip has zero footprint on these manifests**.
- **Fires**: samevtx 2 (both 37112: proton stem gap 0.00 + prune-re-seeded
  comp 50031 gap 3.81); satellite 1105 across 208/239 events.
- **Movers** (vtx105): 0 on all four sample pairs, 0 ADVERSE. **Owned**:
  net +0 both manifests. **Nusel**: byte-identical both manifests (K5 on!).
- **π⁰-watch** (peer pr/126 coordination): no genuine π⁰ harmed; moved
  pairings were junk (37112 281.2→69.2 — the OFF "π⁰" paired the γ with
  the PROTON shower, the substance of the owner's old "no pi0???" note;
  396222 pio_flag 1→2, 799→260 — guard-driven; 69314 9.1→19.1). 30504 /
  176502 / 256587 / 259542 essentially unchanged.

### 4.3 Owner-event outcomes (round-d ON dumps)

- 37112: shw 67048 = **one shower, 797.3 MeV, 39 segs**, pdg 11 (9008 and
  50031 gone); 84070 keeps pdg 11.
- 69314: 25→**18 showers**; 3014 68.9→74.7 MeV (9 segs) **pdg 11**; all 7
  vertex-connected crumbs absorbed; PF electron entries 38→**30** (sub-5-MeV
  28→21). (Requires K5 — see §5.)
- The four fake electrons: fixed by the guard exactly as measured in doc
  pr/124 §C (see §1.1 table).

## 5. Ship + the K5 decision

**FLIPPED ON** (toolkit `8b371920`, owner-directed):
`shower_pass3_cone_guard_len=15` (resolves pr/124 §C.3 as option 1 — owner:
"Flip anyway" after the dQ/dx non-separator measurement, §3.1) and
`shower_samevtx_track_absorb=true` (37112).

**FLIPPED ON 2026-08-29 (owner verdict, toolkit `d74c5524`)**:
`shower_satellite_absorb` — after scanning the K5 decision pair the owner
answered *"item 3, flip on is fine"*, i.e. option (1) below at the scanned
operating point (`max_mev=10 / host_mev=20`, **not** the cap-3 variant).
Flip validation (compiled-config proof, per-event flip-equivalence
decomposition over 478 archives, knob-off control, 0 movers, nusel 239/239)
is recorded in doc pr/127 §4; production arms are now
`work-pr125r1-flipK5{98,141}-*`. The metric table below is the accepted
cost, unchanged by the flip.

The same message closed the production-pair scan — *"For item 2, things are
good"* — with one exception, 137238, which turned out to be an unrelated
regression of the pr/93 r4 fix for that event: doc pr/127.

**(Superseded) HELD OFF — owner decision**: `shower_satellite_absorb` (the
69314 fix).
It is physics-clean on every hard check (0 movers, owned +0, nusel
identical, 0 pdg flips, no genuine π⁰ touched) but absorbs unlabeled crumb
charge into marked showers everywhere, which the em117 metric books as
impurity:

| metric | OFF | ON (K5) |
|---|---|---|
| 141-set med qF1 | 0.949 | 0.910 |
| 141-set Σ q_extra | 2.41e7 | 3.12e7 |
| 141-set Σ q_miss | 2.45e7 | 2.36e7 |
| 98-set med qF1 | 0.907 | 0.901 |
| 98-set Σ q_extra | 7.49e6 | 1.48e7 |

An E-cap refinement does not remove the cost (cap 3 MeV keeps 74% of fires
and 44% of absorbed energy). The labels are silent on crumbs (31 IN / 0 OUT
among the labeled ones — scan-time marks never adjudicated satellites).
Owner options: (1) flip as-is (E<10) accepting the metric shift as a
metric limitation; (2) flip capped (max_mev=3 — 69314 still fully fixed,
~half the charge cost); (3) keep OFF, 69314 stays a cascade.
Decision Bee pair: §5.1.

### 5.1 Bee record (all UUIDs content-verified against the local zips
### by per-event mc-layer md5 before recording — pr/124 defect closed)

- **Production pair** (10 events, flip content): OFF
  `7f4ffdb1-36b4-4ac5-a761-853e408a2a97` / ON
  `cdf0749a-c74a-4d0d-9083-8f9554679daf` — annotated
  `bee/pr125r1/pr125r1.index.txt`. Hash attribution: exactly idx 0–6
  differ; idx 7–9 verified no-ops; only mabc-pr.zip moves.
- **K5 decision pair** (13 events): OFF
  `b169a068-fab5-4a27-ac4f-6cf1bbbd0eb5` / ON
  `defaa224-b1ba-4fb3-8449-62b85637375f` —
  `bee/pr125r1K5/pr125r1K5.index.txt`. idx 0 = 69314 (the rescue), idx
  1–4 UP movers (IN-recoveries), idx 5–9 DOWN movers (the metric-cost
  class), idx 10–12 π⁰-watch.

### 5.2 Flip validation + new baselines

- flipchk (post-flip cfg, no envs) vs dbg-d: divergent events exactly
  {94392, 52693, 77328, 173819} (141-mcp2k), {396222, 415278} (98-mcp2k),
  {37112} (98-ncpi0); 137238/175896/176502 byte-identical (no-ops); ALL
  other archives identical. 0 movers; nusel identical; owned 98 +0 /
  141 −1 (94392's pseudo-n muon, pr/124 §C).
- 37112 at the production point (K3, no K5): one shower **783.0 MeV /
  35 segs** (797.3 with K5's five crumbs).
- Peer coordination: doc-84 r4 flip footprint on these manifests = ZERO
  (dbgP ≡ pinned baselines 478/478); doc pr/126 π⁰-census refresh expected
  on ≤7/76 groups, peer notified with the exact event list.
- **Production baselines going forward**: `work-pr125r1-flipchk98-*` /
  `work-pr125r1-flipchk141-*` (content = flipA except the 7 events above).
  Toolkit: knobs `9b67d246`+`67a55f1f`+`ea031faf`+`dc4ab6f8`, flip
  `8b371920`.

### 5.3 Post-ship finding: the guard wakes π⁰ Path 2 in 396222 (vertex mutation)

At the flipped production point, `id_pi0_without_vertex` (Path 2) accepts a
group in **396222** — its first acceptance anywhere on SBND (peer doc
pr/126 measured it dormant, 0/76 over 239 events, at the pr/124 point).
Path 2 mutates the neutrino vertex (`NeutrinoShowerClustering.cxx:5675-5676`,
the prototype's own "hack"): 396222's main vertex moves **1.4 cm**
((121.70,−164.44,312.59)→(122.09,−164.42,311.27)) and `kine_reco_Enu`
4613→3797 MeV. The vtx105 movers gate is blind here (396222 unlabeled in
that epoch); nusel selection is unchanged. Cause chain: the guard's three
track declines re-shape the shower set → the Path-1 pairing fails where it
previously succeeded → Path 2 fires. Flagged to the owner (production Bee
pair idx 6 is now a priority row) and to the pr/126 session (Path-2
mutation risk is no longer theoretical). If the owner rules the 396222
outcome adverse, the candidate mitigations are a Path-2 disable knob or a
396222-class qualifier on the guard — neither taken unilaterally.

**§5.3 addendum (pr/126 cross-audit)**: Path-2 dormancy was structural, not
a safety margin — the path is gated on the main-vertex shower population
(`map_vertex_segments[main_vertex].size() > 2 → return` + Path-1 blanket
suppression at :5361), so anything thinning that population can wake it;
the guard's declines were merely first. The vertex-mutating lines are
inherited verbatim from the prototype (NeutrinoID_shower_clustering.h:
696-698) and have zero SBND validation. If mitigation is wanted, a Path-2
disable knob isolates the untested code; a guard qualifier would perturb a
validated knob (peer recommendation, concurred). Also: 396222 has no
scanner π⁰ verdict (pr/126 §4i rescan list, as do 415278/176502/37112),
so its Enu shift is a movement, not yet a regression — π⁰-pairing
questions added to the Bee rows.

**§5.3 π⁰-census cost (pr/126 refresh vs flipchk, wcp cc878ba4)**: the flip
costs exactly one accepted π⁰ group across 239 events (76→75: Path 1 −2 —
the mechanism by which 396222 fell through — Path 2 +1); kine_pio_flag
164/2/73 → 163/3/73. The pr/126 hand-scan identification numbers (52%
exact-pair, 72% shared-γ, blocker breakdown) are EXACTLY unchanged — none
of the 50 hand-paired π⁰ is in the divergence set, as predicted pre-flip.
