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

## 3. Measurements

(to be filled — §3.1 dQ/dx scan, §3.2 anatomy, §3.3 merge collateral
pre-scan, §3.4 prune-exemption pre-scan)
