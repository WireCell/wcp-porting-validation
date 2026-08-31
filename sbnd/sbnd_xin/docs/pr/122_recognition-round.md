# doc pr/122 — the recognition round: seeds, PID votes, assembly (measured)

**Status: MEASUREMENT COMPLETE — all three guard candidates die or defer;
probes + census ship, no behavior knob selected. (pr/118-P2 / pr/119
precedent.)**

Owner directive 2026-08-28: open the recognition thread — the three cases
(54332 / 166870 / 235435) where the defect is upstream of the admission/merge
geometry of pr/117–120. Same validation bar as previous rounds. Companion:
doc [pr/121](121_ex1-dedup-orphaning.md) (this round's shipped fix). Both
manifests are used throughout: the 98-event owner scan (`emscan-0827`) and
the 141-event out-of-sample scan (`emscan-0828-agent5`, doc 115 §17).

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# arms: see doc pr/121 Repro (shared).  Census:
./scripts/pr122_recog_census.py --tsv docs/pr/pr122-seed-census.tsv \
    --pid-tsv docs/pr/pr122-mu-showers.tsv \
    'work-pr121r1-dbgA-*' 'work-pr121r1-dbg141-*'
grep -h "sfv_straight_guard: decline" work-pr121r1-dbg*/pr_evt*/stdout.log
```

Probes added this round (all env-gated stderr, byte-neutral — gate ledger in
doc pr/121 §4): `SHOWER_SEED` at the in_main_cluster seeder (admission
disjuncts + len/straightness/median dQ/dx per accepted root, ABORT line at the
no-main-vertex return), `SHOWER_PID VOTE/VOTE_MEM/VOTE_SKIP/GUARD/COPY`
(`WCT_SHOWER_PID_DEBUG`, PRShower.cxx: the update_particle_type accumulators,
its guard verdicts, and every `data.particle_type` copy), `EX1_DEDUP` (doc
pr/121).

## 1. Case (a) — 54332: the mis-flagged straight track that seeds a shower

**Mechanism confirmed.** `segment_is_shower_topology` (5-branch spread test,
no dQ/dx term) fires on the straight 32.3 cm track 16014; the legacy demote
guard requires >50 cm; the consumer writes pdg 11 / score 100; the
`in_main_cluster` seeder accepts on the flag disjunction alone:

```
SHOWER_SEED site=in_main_cluster seg=16014 pdg=11 traj=0 topo=1 pdg11=1 long_muon=0
            len_cm=32.28 med_dqdx_mip=1.668 straight=1
```

**The class is real and now has three confirmed events.** Straight
topo-flagged in_main_cluster seeds across both manifests: 13. Labeled:

| event | seg | len cm | med dQ/dx (MIP) | label |
|---|---|---|---|---|
| 54332 | 16014 | 32.3 | 1.67 | **BAD** — OUT-marked (scan: "overclustering a track"); event ceiling 0.686→0.799 |
| 171143 | 8032 | 56.1 | 1.38 | **BAD** — verdict `not an EM shower` ON this shower (262 MeV fake e-) |
| 277298 | 17030 | 49.3 | 1.34 | **BAD** — verdict `not an EM shower` ON this shower (194 MeV fake e-) |
| 444187 | 19079 | 58.6 | 1.20 | **GOOD** — TARGET-marked real electron stem |
| 444187 | 19082 | 52.5 | 1.09 | **GOOD** — same complex |
| 444187 | 19081 | 46.7 | 1.53 | same complex |
| 500083 | 3002 | 13.5 | 1.57 | **GOOD** — event `correct`, 166 MeV real shower |
| 289508 | 17012 | 10.7 | 1.45 | **GOOD** — event `correct`, 397 MeV real shower |
| 46363 / 214469 / 388 / 81597 / 342199 | | 16–38 | 1.26–1.49 | unmarked (nueCC-family stems) |

**Why no guard ships.** Every admission-time feature interleaves:
- **length** — BAD spans 32–56 cm, GOOD spans 10.7–58.6 cm (444187's real
  stems bracket 171143/277298). `shower_topo_demote_len` is dead at any
  useful threshold.
- **median dQ/dx** — BAD {1.34, 1.38, 1.67} vs GOOD {1.09, 1.20, 1.45, 1.53,
  1.57}: fully interleaved. The 98-set alone suggested a clean cut above 1.55
  (54332 highest); the out-of-sample set broke it in BOTH directions —
  500083's good stem at 1.57, 171143/277298's tracks at 1.34/1.38. The
  existing `shower_topo_dqdx_guard` window (spare when ratio >1.75 or <1.2)
  catches none of the three.
- **straightness** — all 13 are straight by construction of the class.

A dqdx>1.6 guard would cover 1 of 3 BAD with a 6% margin to a GOOD — against
the pr/120 precedent (shipped on a 150°-vs-20° margin), that is an overfit,
not a guard. **Killed by measurement.**

**The round's mechanistic finding — inherited flags.** A gidx-carrying
re-probe (`work-pr121r1-dbgB-*`) shows the five longest seeds (both BAD
171143/277298 AND all three GOOD 444187 stems) were **never evaluated by the
topology classifier in their final form at all**: no `shower_topo dbg` line
exists for their graph indices. Their `kShowerTopology` flags were set on
ancestor segments (444187: gidx 22/34, seeds at gidx 79/81/82) and carried
through re-segmentation without re-validation. So for the long-seed class the
classifier's own features are not even defined at seed time, and the
inheritance affects good and bad seeds alike — a seed-time re-validation
would be a new, separately-measured campaign, not a threshold on this
census. The event-level stakes stay on the ledger: 2.89e6 q_extra (54332) +
262 MeV (171143) + 194 MeV (277298) of fake EM object.

## 2. Case (b) — 166870: the γ reported as µ⁻, and who actually did it

**Mechanism confirmed — it is not the vote.** `update_particle_type` only
ever writes pdg 11; the shower's reported type is the verbatim start-segment
pdg copy in `calculate_kinematics`. On 166870 node 85045 (real 4-segment
20.5 cm γ, π⁰ partner at m_γγ≈116 MeV in the labels):

- the shower is created by `from_vertices` with anchor 85045 (15.1 cm,
  pdg 13);
- the **pr/40 r9 `sfv_straight_guard`** (production ON) declines the forced
  e⁻ write because the anchor is a straight long track, and — by design —
  skips the vote entirely (D3: the vote would redo the 13→11);
- `SHOWER_PID COPY shower_id=2 nseg=4 start_seg=85045 pdg=13` — the µ⁻ is
  the guard's declared outcome, not a mis-vote. Had the vote run, even the
  legacy rule flips it (all 4 members land in the shower bucket, 20.5 cm vs
  0).

**Exposure**: the guard fires 17× across 239 events (98-set: 166870, 122660,
259542, 90055; 141-set: 13 fires / 12 events). Judged against labels:
**zero** of the 141-set µ-typed showers overlaps any labeled EM target, and
most of their events are verdict `correct` — including 90055's design case
(a lone straight 29.8 cm track correctly rendered as a hadron carrier).
166870 is the **only** labeled-bad fire, and the evidence of its γ-ness is
its π⁰ pairing — cascade context, not a local anchor feature (122660's
3-seg/3-cluster 30 cm fire is shape-identical and unchallenged).

**Killed as a local guard; routed to the π⁰ thread.** With n=1 positive and
no admission-frame discriminator, any knob is a single-event special case.
The principled fix is pairing-driven: a µ-typed conn-2 object whose
m_γγ with an identified shower lands in the π⁰ window is a γ — exactly the
owner-gated π⁰-hypothesis thread (doc 115 §15/§17.9, deferred by owner).

## 3. Case (c) — 235435: not a recognition hole

The event has a main vertex and assembles **8** showers (no `SHOWER_SEED
ABORT`); the two vertex-attached seeds fire normally. The scan complaint
("the whole event is one EM shower that never assembles") is
**fragmentation** — many 1–2-segment showers (42/79/40/5/5 MeV + a 133 MeV
conn-3 pair) that never merge — i.e. the pr/117/118 merge-thread territory
(and the vertex quality on this event), not the seeding/flag machinery.
Reclassified; no probe follow-up here.

## 4. What ships

- The probes (byte-neutral, gate ledger doc pr/121 §4) and the two census
  scripts + TSVs.
- No behavior knob from this round. The recognition thread's residual is
  routed: (a) needs a better feature than the spread test (or cascade
  context); (b) needs the π⁰ thread; (c) belongs to the merge thread.

## 5. Files

| file | |
|---|---|
| `scripts/pr122_recog_census.py` | seed/PID/assembly census |
| `docs/pr/pr122-seed-census.tsv` | 285 seed rows, both manifests |
| `docs/pr/pr122-mu-showers.tsv` | reported-µ multi-seg showers |
| toolkit `clus/src/NeutrinoShowerClustering.cxx` | SHOWER_SEED probes |
| toolkit `clus/src/PRShower.cxx` | SHOWER_PID probes |
