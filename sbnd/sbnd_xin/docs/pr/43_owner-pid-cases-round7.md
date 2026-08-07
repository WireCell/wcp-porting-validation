# doc pr/43 — four owner PID cases (18255: 142421 / 54351 / 56463 / 57661)

**STATUS 2026-08-07 (round 2): the three remaining cases are re-fixed with
three NEW, narrower knobs — see `# Round 2` below. Round 1 remains ROLLED
BACK; nothing from it was re-applied as-is.**

**STATUS 2026-08-07 (round 1): ROLLED BACK, not just defaulted off.** Owner judged the
G3/G4 population/score movement (42/48 nueCC48 events) too broad for what the
four cases needed and asked for the five knobs to be pulled from the code
entirely, to be revisited later rather than left dead-OFF in the tree. All of
F1 `muon_chain_proton_veto`, F2 `shower_type_cache_refresh`, F3/F3b
`shower_traj_dqdx_guard`/`shower_traj_chain_pion`, F4
`kine_shower_vertex_barrier` are reverted out of `toolkit` as of commit
`225d7e7e` (revert of `4aabef3e`); `wcdoctest-clus` 100/100 clean post-revert,
compiled-config grep confirms all five keys are gone. This doc is kept as the
investigation record — the four cases, the root-cause traces, and the fix
shapes below are still believed correct; only the "ship it, even OFF" call is
reversed. Read this before re-attempting: the shapes below are a starting
point, not a green light to reapply as-is.

Also filed as pr/40 round 7 in spirit (same PID-mis-ID family), kept as its own
numbered doc per the owner's request for a dedicated file. Cross-reference:
[doc pr/40](40_track-mis-id-as-electron.md), [doc pr/38](38_pf-missing-tracks.md).

## Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# Phase 0: re-run at HEAD, bare config (no SBND_* overrides), fresh out_roots.
PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-ncpi0-cb0805 <out1> data 142421
PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-mcp1k-cb0805 <out2> data 54351 56463 57661

# G1 (knob-off byte-identical), 48-event nueCC48 manifest:
./run_pr_chain_batch.sh work-nuecc48-cb0805 <off48> data
# reference: git stash the 14 pr/43 files, wcbuild, run the SAME 48-event
# manifest to completion in a directory NOT touched by any concurrent
# rebuild, then git stash pop + wcbuild again before comparing.

# G2/G3/G4 (all five knobs forced on), same 48-event manifest:
SBND_MUON_CHAIN_PROTON_VETO=1 SBND_SHOWER_TYPE_CACHE_REFRESH=1 \
SBND_SHOWER_TRAJ_DQDX_GUARD=1 SBND_SHOWER_TRAJ_CHAIN_PION=1 \
SBND_KINE_SHOWER_VERTEX_BARRIER=1 \
  ./run_pr_chain_batch.sh work-nuecc48-cb0805 <on48> data

# gate: abtest/hash_archive.py member-content hashes on mabc-pr.zip +
# pctree-pr-evt<ID>.tar.gz, never md5sum (M2).
```

HEAD for every number in this doc: `18936f167430b89f73cb4745958cdbf062aabc08`
(clus/cfg: doc pr/40 round 6). M1 freshness proof done before every arm
(`local/lib/libWireCellClus.so` mtime checked against source edit time and
against `git log -1` each time the lib was rebuilt).

## Symptom

Owner hand-scan of four SBND run-18255 events on the port-5017 PR display:

1. **142421** — a track rooted near (111.7, −37.4, 255.7), "cluster 7011", is
   absent from the particle-flow tree **and** from the energy calculation.
   "This is clearly a bug."
2. **54351** — the chain reads `mu- → mu- → proton`; should be
   `pi+ → pi+ → proton`, and the *other* arm (currently `pi+`) should be the
   muon. Two muon candidates exist; the longer won, but the longer one
   terminates in a proton, so it cannot be the muon.
3. **56463** — two muons again; the shorter should be a pion.
4. **57661** — reads `proton → mu- → e-`; should be `proton → pi+ → mu-`,
   because the terminal track's dQ/dx is muon-like.

**These observations were made on an arm that predates this session's own
pr/40 round 6** (round 6 landed at 21:39 the same evening, after the owner's
scan). Re-running at HEAD (Phase 0, below) was step zero: some cases could
already have read differently, and "already fixed upstream" would have been
a legitimate per-case outcome. In the event, round 6 did **not** touch any
of the four — Phase 0 reproduced all four symptoms unchanged at HEAD, with
one exception: 142421's *particle-flow-tree* absence was already resolved
by round 6's F12 (`shower_absorb_track_guard`); only its *energy* absence
survived to HEAD (see F4).

## Root cause: four independent mechanisms

### F1 (case 2) — the muon-candidate proton veto is 1-hop, the proton is 2 hops out

`NeutrinoVertexFinder.cxx` `examine_direction`'s single-muon-per-cluster
selection picks the longest pdg-13 segment at a vertex as "the" muon,
demoting every other pdg-13 sibling to pion — this loop is **always-on**,
prototype-faithful, no existing knob. Its proton-veto (`n_proton` at the
candidate's *immediate* far vertex) only reaches one hop.

Run 18255 evt 54351: candidate 17007 (54.2cm) wins over 17010 (42.6cm) by
length. 17007's own far vertex (v17004) has no proton — but two hops
further, through a 2.7cm muon-pdg continuation stub (17005), sits a
charge-confirmed proton (17011, median dQ/dx > 1.75× MIP). 17007 cannot be
the muon in a chain that terminates in a proton; the veto simply never
looks that far.

**Measured (`WCT_PID_WRITE_DEBUG=2`):** 17007 and 17010 are the only two
segments the single-muon selection loop actually competes at the shared
main vertex (17005); 17005 the *stub segment* is chained to 17007 via a
separate, earlier "long muon" accumulation pass and was never independently
evaluated by the demotion loop — confirming the veto never had a chance to
see 17011.

### F2 (case 3) — a demoted shower's cached particle_type never refreshes

`Shower::update_particle_type` (`PRShower.cxx`) relabels its **start
segment** to electron when `shower_length > track_length`, but leaves the
*Shower object's own* cached `data.particle_type` at whatever a **prior**
call set it to. Run 18255 evt 56463: segment 14005 (309cm) was first
detected as part of a long-muon chain and its wrapping Shower got
`set_particle_type(13)` at that time (`NeutrinoShowerClustering.cxx:132`);
later, once wrapped as a shower, `update_particle_type` relabels the
*segment's own* `particle_info()` to electron (pdg 11) — but the Shower's
cache stays 13.

`MultiAlgBlobClustering::make_shower_leaf` (the Bee PF-tree shower-leaf
renderer) reads `shower->get_particle_type()` — the **cache**, not the
segment's live pdg — so the display shows "mu- 903 MeV" for a segment the
toolkit's own logic had already reclassified to electron. That reads as a
phantom second muon next to the genuine one (14006, 166 MeV), matching the
owner's "two muons, the shorter should be a pion" report — except the
correct fix is not to demote 14006, it is to stop mis-displaying 14005 as a
muon at all.

**Same divergence class as [doc pr/35](35_energy-reconstruction-port-audit.md)
F1 `kine_shower_pdg_live`**, which already fixed the analogous staleness on
the `T_kine` side; this closes the matching gap on the Bee PF-tree side.

### F3 / F3b (case 4) — a terminal shower-trajectory segment is force-classified electron, and its own chain doesn't follow

`segment_determine_shower_direction_trajectory` (`PRSegmentFunctions.cxx`)
has three branches keyed on the segment's two endpoint degrees. Only ONE of
the three ("neither end is degree-1") tries track PID first before
discarding a non-electron conclusion; the other two are fully unconditional
— they never call track PID at all. Run 18255 evt 57661 seg 18007 (8.3cm,
one degree-1 end, one degree-2 end) takes an **unconditional** branch, so
even though nothing in its own dQ/dx says "electron", it is forced to pdg
11 with `particle_score` sentinel 100.

Once rescued (F3) to its track-PID conclusion (pdg 13, muon — matching the
owner's "dQ/dx consistent with muon"), a second gap appears: the short
(5cm) stub 18005 between 18007 and the main-vertex-emanating proton (18003)
is *still* pdg 13 from an earlier, unrevisited per-segment call, giving
`proton → mu- → mu-`. F3b closes this: it walks a main-vertex proton's
short, non-shower, degree-2 continuation chain and relabels every segment
but the deepest (confirmed-muon) one to pion, giving the owner's
`proton → pi+ → mu-`.

**One bug found and fixed en route:** F3's guard corrects the segment's
pdg but, as first written, left the `kShowerTrajectory` flag set — which
made 18007 unreachable by F3b's non-shower chain walk (`flag_shower` still
read true for a segment now pdg 13). Same class of oversight as
[doc pr/40](40_track-mis-id-as-electron.md) round 4 F7 ("clear the shower
flags when a segment stops reading as a shower"); F3 now clears
`kShowerTrajectory` when it fires, mirroring F7's precedent exactly.

### F4 (case 1, energy half) — the kine BFS has the same over-wide barrier pr/38 already fixed on the Bee side

[doc pr/38](38_pf-missing-tracks.md) fixed `fill_bee_pf_tree`'s BFS barrier
(a detached shower's start vertex blocking the track walk) and added an
orphan safety net so a BFS-unreached segment still gets a Bee PF-tree node.
`fill_kine_tree` (`NeutrinoKinematics.cxx`) runs its **own, separate** BFS
with the exact same over-wide barrier — `shower->fill_sets(...,
/*flag_exclude_start_segment=*/false)` takes `fill_sets`'s **default**
`exclude_start_vertex=false`, i.e. the *un*-corrected barrier. pr/38's own
"Residual" section named this gap explicitly and said the vehicle
(`fill_sets`'s `exclude_start_vertex` parameter) exists and is simply never
threaded here.

Run 18255 evt 142421: the PF-tree side is already fixed by round 6's F12
(segments 7011/7012/7018 all appear as root nodes with correct pdg). But
`kine_energy_particle` — the array `kine_reco_Enu` sums — is still missing
all four: 7011 (pi+, 196 MeV), 7012 (proton, 159 MeV), 7018 (mu-, 207 MeV),
and 7013 (proton, 174 MeV, a daughter of a *different* root pi+). ~736 MeV
of real, fitted track energy is silently absent from `kine_reco_Enu`
(1124.34 MeV at HEAD).

## Why it hid

- F1/F3's mechanisms are per-vertex or per-branch decisions inside a large,
  always-on function; a 2-hop or unconditional-branch miss reads as "the PID
  competition picked the wrong winner" rather than "the veto never looked
  far enough" unless the exact vertex topology is walked by hand.
- F2 is a cache-staleness bug with no visible symptom unless the segment's
  *own* pdg (readable from `PrDisplayDump`'s `calib-pr-evt<ID>.json`, doc
  pr/42) is cross-checked against the Bee tree's displayed label for the
  same node — the two sources normally agree, so nothing routinely compares
  them.
- F4 is pr/38's own documented, un-actioned residual; it hid because the
  PF-tree side (the thing anyone actually *looks at* on the Bee display) was
  already fixed, so the display looked correct while the energy sum quietly
  wasn't.

## Fix

Five default-OFF C++ knobs, `wct-pr-perevt.jsonnet` TLA threading, and
tri-state `SBND_*` runner overrides, all C++ default `false` = legacy =
byte-identical:

| # | knob | site |
|---|---|---|
| F1 | `muon_chain_proton_veto` | `NeutrinoVertexFinder.cxx` examine_direction muon-candidate loop |
| F2 | `shower_type_cache_refresh` | `PRShower.cxx` `Shower::update_particle_type` |
| F3 | `shower_traj_dqdx_guard` | `PRSegmentFunctions.cxx` `segment_determine_shower_direction_trajectory` |
| F3b | `shower_traj_chain_pion` | `NeutrinoPatternBase.cxx` `override_shower_traj_chain_pion` (new post-pass) |
| F4 | `kine_shower_vertex_barrier` | `NeutrinoKinematics.cxx` `fill_kine_tree` |

**F1** — new helper `segment_chain_has_proton` (`PRSegmentFunctions.cxx`):
walks a bounded (max 3 hops), non-shower, degree-2 continuation chain from a
muon candidate's far vertex looking for a charge-confirmed proton (same
`median dQ/dx > 1.75× MIP` threshold `segment_has_proton_daughter` already
uses). Threaded into both the pdg-13 and pdg-0 branches of the
muon-candidate loop. A companion helper `segment_chain_continuation`
*collects* the same chain so a disqualified candidate's own stub segments
are relabelled pion alongside it — without this, demoting only the head
segment left an orphaned muon stub between it and the proton.

**F2** — `Shower::update_particle_type` gains a `refresh_type_cache`
parameter; when true and the function relabels its start segment to
electron, it also calls `set_particle_type(11)` on itself, mirroring the
just-written segment pdg into the Shower's own cache immediately rather
than depending on a possibly-absent later `calculate_kinematics()` call to
refresh it.

**F3** — `TrackPidOptions` gains `shower_traj_dqdx_guard`.
`segment_determine_shower_direction_trajectory` is restructured to run
track PID **once, before** branch selection (regardless of which of the
three endpoint-degree branches would otherwise apply) when the guard is on,
and to trust a confident non-electron conclusion
(`segment_dqdx_spares_electron_reclass` confirms non-MIP-like) instead of
falling through to the unconditional electron default. When it fires, it
also clears `kShowerTrajectory` (see "one bug found and fixed en route"
above). Off: the extra track-PID call never runs at all — byte-identical.

**F3b** — new post-pass `override_shower_traj_chain_pion`
(`NeutrinoPatternBase.cxx`), called at the same site as pr/40's F8/F14
(last word before `shower_clustering_with_nv`, after F3 has had its chance
to fire). Walks each main-vertex proton's `segment_chain_continuation`
chain; if every segment in it is pdg 13 and ≤15cm (the length gate exists
specifically to avoid misreading a single long muon track that pattern
recognition merely fragmented into two collinear pieces as a
proton-pion-muon chain), relabels every segment but the deepest to pion.

**F4** — `fill_kine_tree`'s `shower->fill_sets(...)` call now passes
`exclude_start_vertex = m_kine_shower_vertex_barrier` (mirrors pr/38's
`pf_shower_vertex_barrier` exactly). An orphan safety net, gated under the
same knob, then pushes every still-unreached, non-shower, main-cluster
segment with a fitted direction into `kine_energy_particle` via the
existing `push_segment_kine` lambda, emission ordered by encoded id
(cluster_id×1000+graph_index) for reproducibility.

Runner env overrides (`run_pr_chain_batch.sh`): `SBND_MUON_CHAIN_PROTON_VETO`,
`SBND_SHOWER_TYPE_CACHE_REFRESH`, `SBND_SHOWER_TRAJ_DQDX_GUARD`,
`SBND_SHOWER_TRAJ_CHAIN_PION`, `SBND_KINE_SHOWER_VERTEX_BARRIER` (tri-state:
unset = cfg default, 1 = force on, 0 = force off).

## Demonstration — all four cases, final build, all five knobs on together

```
evt 142421:  7011 pi+ 196 MeV, 7012 proton 159 MeV, 7018 mu- 207 MeV -- all
             root-level PF-tree nodes (F12, unchanged by pr/43) AND now in
             kine_energy_particle (F4): kine_reco_Enu 1124.34 -> 2124.16 MeV
             (n_particles 34 -> 38).
evt 54351:   17007 pi+ 163 MeV -> 17005 pi+ 8 MeV -> 17011 proton 135 MeV;
             other arm 17010 mu- 127 MeV.               [owner: exact match]
evt 56463:   14005 e- 891 MeV (was "mu- 903 MeV"); 14006 mu- 166 MeV sole
             muon.                                       [owner: resolved]
evt 57661:   18003 proton 113 MeV -> 18005 pi+ 5 MeV -> 18007 mu- 42 MeV.
                                                          [owner: exact match]
```

Each case verified individually (single knob on) and again with all five
knobs on together — no cross-knob interference observed (142421's
`kine_reco_Enu` is identical, 2124.16 MeV, whether F4 runs alone or with
F1/F2/F3/F3b also on).

## Gates

- **G1 — knob-off byte-identical.** 48-event nueCC48 manifest,
  `abtest/hash_archive.py` member-content hashes (never `md5sum` — M2) on
  `mabc-pr.zip` + `pctree-pr-evt<ID>.tar.gz`. **48/48 PASS.**
  Reference: a genuine `git stash` clean-HEAD arm (labels `pr43-off48` vs
  `pr43_cleanhead_ref48b`) — **not** a pre-existing round-6 artifact
  (`work-pr40r6-off48`), which was tried first and found to disagree with a
  freshly-built clean-HEAD run even though clean-HEAD is perfectly
  run-to-run reproducible (verified: two independent clean-HEAD runs of
  evt 10550 hash-identical to each other, `139918f0…`, but *not* to
  `work-pr40r6-off48`'s own `a65fce7b…`) — i.e. that stored artifact was
  stale/mismatched for reasons unrelated to this change, and reusing it as
  a reference would have manufactured false G1 failures. Also caught and
  discarded: a first clean-HEAD reference attempt was itself invalidated by
  a `wcbuild` race (M1/M3 — "file too short" on 27/48 events) from
  rebuilding this session's own dev tree while that reference batch was
  still running in the background; redone with the rebuild strictly
  sequenced before/after, never during, the reference batch.
- **G2 — the four cases fixed.** All four, individually and with all five
  knobs on together, on the final rebuilt binary. **4/4 PASS** (see
  Demonstration).
- **G5 — `./build/clus/wcdoctest-clus`.** **100/100 test cases, 1042/1042
  assertions, PASS**, both before and after the final rebuild.
- **Compiled-config proof:** each new key threaded through three jsonnet
  levels (`cfg/pgrapher/common/clus.jsonnet`'s `tagger_check_neutrino(...)`
  key-suppression idiom → `cfg/pgrapher/experiment/sbnd/clus.jsonnet`'s
  `clus_pr(...)` and `pr(...)` forwarding → `wct-pr-perevt.jsonnet`'s TLA
  default) and confirmed via the runner's tri-state env override actually
  changing the compiled config (each knob's forced-on run visibly changed
  its target output).

### G3/G4 — population census and score shift (48-event nueCC48, all five knobs forced on)

**This is the load-bearing finding of this round and the reason the flip
below is NOT authorized on the strength of G1/G2/G5 alone.**

- **PF-tree (mc.json) census:** 37/47 events with a `pctree-pr` (evt 116962
  has none, a pre-existing characteristic unrelated to this change) show at
  least one node text change; 566 total node-text diffs across the
  manifest. Many of these are id-renumbering artifacts of the same node
  (identical text reappearing at an adjacent id) rather than pdg changes,
  but a substantial fraction are genuine pdg/energy relabels — this was
  **not measured or bounded per-mechanism** (which of F1/F2/F3/F3b owns
  which fraction) before this doc was written.
- **Score/energy census (`T_tagger`/`T_kine` via `tracking-pr.root`):**
  42/48 events move `numu_score`, `nue_score`, and/or `kine_reco_Enu`.
  Selected large swings: evt 239794 `kine_reco_Enu` +1717.37 MeV; evt 388
  −1026.57 MeV; evt 196649 −542.05 MeV; evt 172230 −475.75 MeV; evt 489330
  +366.17 MeV. `nue_score` swings include evt 342199 (−18.6, near the ±15
  clamp) and evt 38856 (+2.38).
- Both censuses were run against the **final rebuilt binary** (all pr/43
  code + G1's clean-HEAD verification), so they are not an artifact of the
  build-race contamination described under G1.

This reach — most of the 48-event manifest, not the four reported events —
was not anticipated when F1–F4 were scoped against the four owner cases.
F1 and F3/F3b are general topological/dQ/dx rules (not case-specific), so a
broad population footprint is plausible in hindsight; F4 mechanically
mirrors pr/38, whose own census (doc pr/38, `pf_shower_vertex_barrier`) was
also non-trivial (5/48 events). No number here looks obviously *wrong*
(no efficiency collapse, no chi2 explosion), but CLAUDE.md §5 is explicit
that a physics number that "looks wrong" gets reported, not tuned away, and
by extension a shift THIS broad on scores/energy — even one that looks
individually reasonable — is a decision for the owner, not something to
wave through against a pre-authorization scoped to "four events."

## Flip — NOT flipped; deliberately held for owner review

All five knobs stay **C++ default false / SBND TLA default false** in this
commit — none are flipped to the SBND production default, despite this
round's flip having been pre-authorized ("flip if gates pass") during
planning. That authorization was given against an expectation of
four-events-worth of impact; the actual G3/G4 census shows population and
score movement across the large majority of the 48-event manifest. This
doc surfaces the exact numbers (above) instead of exercising the
authorization — the owner should decide the flip with the G3/G4 tables in
hand, not have it decided for them by a scope that turned out to be much
larger than planned.

## Scope and what is NOT claimed

- The per-mechanism attribution of the 566 PF-tree node changes and 42
  score-moved events (which knob owns which event) has not been done. A
  follow-up round should isolate each of F1/F2/F3/F3b/F4 individually
  across the 48-event manifest (5 separate on-arms) before any flip
  decision, the same way pr/40's rounds isolated F9/F10/F11/F12/F14.
- F3b's 15cm-per-segment length gate is a judgment call, not a measured
  threshold — no scan was done to find where "genuine collinear-fragment
  muon" and "real proton→pion→muon chain" separate in length.
- F4 is a mechanical mirror of pr/38's `pf_shower_vertex_barrier`, not
  independently re-derived from the prototype; see
  `porting_dictionary.md`'s new entry for the designed-parity note.
- `numu_score`/`nue_score` are uBooNE-trained BDTs, not SBND-retrained (doc
  pr/41 §5) — the *shifts* reported here are relative-ranking movements on
  an already-uncalibrated score, not claims about absolute physics
  correctness moving closer to or further from truth.

## `porting_dictionary.md` entries

- **F1 `muon_chain_proton_veto`** — designed divergence. The prototype's
  muon-candidate selection has no proton-chain veto at all (1-hop or
  multi-hop); this is a toolkit-only extension of an already-toolkit-only
  (not ported) 1-hop check.
- **F2 `shower_type_cache_refresh`** — cache-consistency bug fix, not a
  port-fidelity question; the prototype has no analogous cached-vs-live
  split (`WCShower` recomputes on read). Same class as pr/35 F1.
- **F3 `shower_traj_dqdx_guard`** — designed divergence. The prototype's
  shower-trajectory PDG assignment is likewise unconditional on two of its
  branches (see `ProtoSegment.cxx`); this is a toolkit-only extension.
- **F3b `shower_traj_chain_pion`** — designed divergence, no prototype
  counterpart (the prototype has no proton→muon chain relabeling rule).
- **F4 `kine_shower_vertex_barrier`** — designed parity with this
  session's own prior fix (pr/38's `pf_shower_vertex_barrier`), not an
  independent prototype port; the prototype's `fill_kine_tree` has no
  barrier/orphan distinction at all (its flat pre-BFS loop gives every
  main-cluster segment a node regardless).

## Verification (how the owner re-checks)

```bash
# Byte-identical-off, one event:
./run_pr_chain_batch.sh <ql_root> <out> data <evt>
python3 abtest/hash_archive.py <out>/pr_evt<ID>/mabc-pr.zip <reference>/pr_evt<ID>/mabc-pr.zip

# Any single fix, forced on:
SBND_MUON_CHAIN_PROTON_VETO=1 ./run_pr_chain_batch.sh work-mcp1k-cb0805 <out> data 54351
SBND_SHOWER_TYPE_CACHE_REFRESH=1 ./run_pr_chain_batch.sh work-mcp1k-cb0805 <out> data 56463
SBND_SHOWER_TRAJ_DQDX_GUARD=1 SBND_SHOWER_TRAJ_CHAIN_PION=1 \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 <out> data 57661
SBND_KINE_SHOWER_VERTEX_BARRIER=1 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-ncpi0-cb0805 <out> data 142421
# then inspect <out>/pr_evt<ID>/mabc-pr.zip's data/0/0-mc.json (or
# calib-pr-evt<ID>.json's "kine" block for F4) directly.

# G3/G4 population/score census reproduction: run the 48-event nueCC48
# manifest with all five SBND_* overrides =1, diff data/0/0-mc.json node
# text against an off-arm, and pull numu_score/nue_score/kine_reco_Enu from
# T_tagger/T_kine in tracking-pr.root (uproot) for both arms.
```


# Round 2 (2026-08-07) — narrow re-attempt, three knobs, per-knob attribution

Owner instruction: retry the three remaining cases; consider (1) simple
in-chain bug fixes and (2) fixes on the Particle information at final PF
formation; either way pdg / shower flags / shower cache / kine / Bee tree /
display must stay CONSISTENT ("a track still displayed as Shower" is the
named anti-pattern). Owner session answers: for 56463 the 411 cm track
14005 IS the muon and 14006 → pi+ (round 1's F2 reading — 14005 as e- — is
dropped); the late pass runs BEFORE the taggers (scores see corrected
labels, full consistency over frozen scores); flip SBND ON same round iff
zero nusel verdict flips and every moved event is attributed to its knob.

## Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# Phase 0 baselines at HEAD 0e098f62 (pr/44 knobs ON):
PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr43r2-head-mcp1k data 54351 56463 57661
# G2 per-knob:
SBND_SINGLE_MUON_PROTON_CHAIN_VETO=1 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr43r2-onK1 data 54351
SBND_SINGLE_MUON_LONG_MUON_CLAIM=1   PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr43r2-onK2 data 56463
SBND_PID_FLAG_RECONCILE=1            PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr43r2-onK3 data 57661
# G3 censuses: 48-evt nueCC48 off/on arms + ncpi0-19 off/on arms; compare
# data/0/0-mc.json trees + nusel-table.tsv per arm (census.py).
```

## Phase 0 — the baseline moved since round 1

pr/44's `shower_long_muon_keep_type` (SBND ON since toolkit 78f77bd8)
already changed case 3: segment 14005's pdg now stays 13 (round 1 measured
11 with a stale shower cache of 13). At HEAD the event displays TWO muons —
`14005 mu- 903` + `14006 mu- 166` — which matches the owner's round-2
reading exactly ("second muon → pion", cluster 14006). Cases 2 and 4
reproduce unchanged.

## Root causes (all three verified by Phase-0 re-runs + code read)

- **56463 — K2.** The single-muon selection (`NeutrinoVertexFinder.cxx`
  `examine_direction`) SKIPS out-edges in `segments_in_long_muon`, so the
  411 cm long-muon chain neither competes for nor claims the vertex muon
  slot; 14006 wins as sole candidate and keeps pdg 13.
- **54351 — K1.** The selection's proton veto is 1-hop; proton 17011 sits
  behind the 2.7 cm stub 17005, so candidate 17007 (54.2 cm) wins by length
  over 17010 (42.6 cm). A muon cannot terminate in a proton.
- **57661 — K3.** `segment_determine_shower_direction_trajectory`
  (`PRSegmentFunctions.cxx`) has two endpoint-degree branches that force
  pdg 11 + sentinel score 100 WITHOUT ever calling track PID; seg 18007
  (8.3 cm, Bragg-rising dQ/dx = stopping track) is forced e-, keeps
  `flag_shower=True` and a single-segment wrapper Shower — the exact
  stale-flag inconsistency class the owner named.

## Fix — three default-OFF knobs (C++ default false = byte-identical)

| # | knob | site | route |
|---|---|---|---|
| K1 | `single_muon_proton_chain_veto` | NVF selection loop | in-chain fix |
| K2 | `single_muon_long_muon_claim` | NVF selection loop | in-chain fix |
| K3 | `pid_flag_reconcile` | new late pass `reconcile_particle_flags` | final-formation fix |

- **K1**: veto walks the bounded (≤3-hop) non-shower degree-2 continuation
  chain (`segment_chain_has_proton`, restored from round 1 F1 verbatim); a
  chain-vetoed candidate demotes to pion together with its stubs
  (`segment_chain_continuation`) and the selection re-picks; a demote-all
  guard falls back to legacy selection if the chain veto would leave no
  muon at all.
- **K2**: a long-muon out-edge claims the muon slot with the chain's summed
  length (deterministic IndexedSegmentSet order); it is itself never
  demoted; other pdg-13 arms demote through the existing conversion loop.
- **K3**: called in `TaggerCheckNeutrino` AFTER `shower_clustering_with_nv`
  and BEFORE the taggers, so tagger features, kine, Bee PF tree and PR
  display all see one labeling. Rule 1: a main-vertex proton's degree-2
  continuation chain ending in a forced-e- (pdg 11 + score sentinel 100)
  terminal gets ordinary track PID re-run; a confident non-electron
  conclusion is adopted, shower flags cleared, the single-segment wrapper
  Shower dissolved (pi0-paired and cached-±13 long-muon showers exempt),
  ≤15 cm pdg-13 stubs → pion. Rule 2 (consistency guard, main cluster):
  confirmed track pdg (13/211/2212) with stale
  kShowerTrajectory/kShowerTopology → flags cleared (pr/40 F7 precedent),
  stale wrappers dissolved.

Runner tri-states: `SBND_SINGLE_MUON_PROTON_CHAIN_VETO`,
`SBND_SINGLE_MUON_LONG_MUON_CLAIM`, `SBND_PID_FLAG_RECONCILE`.

## Demonstration (G2) — each knob alone fixes its case; all-on identical

```
evt 54351 (K1):  17007 pi+ 163 -> 17005 pi+ 8 -> 17011 proton 135; 17010 mu- 127   [owner: exact]
evt 56463 (K2):  14005 mu- 903 (sole muon);  14006 pi+ 177                          [owner: exact]
evt 57661 (K3):  18003 proton 113 -> 18005 pi+ 5 -> 18007 mu- 34;
                 PID_RECONCILE trace: "terminal rescue seg (clus=18 idx=7) 11 -> 13,
                 1 stub(s) -> 211" + "dissolve wrapper shower"; flag_shower cleared  [owner: exact]
```

## Gates

- **G0** freshness: `local/lib/libWireCellClus.so` rebuilt+installed after
  the last source edit, checked before every arm.
- **G2**: 3/3 (above); all-on run byte-reproduces the per-knob trees.
- **G3 per-knob attribution (the round-1 gap, now closed):**
  - nueCC48, four separate on-arms vs `work-pr43r2-off48`: **K1 0/48, K2
    0/48, K3 0/48, all-on 0/48 events moved**; `nusel-table.tsv` 0-diff on
    every arm. The three topologies simply do not occur in the 48 nueCC
    events.
  - ncpi0-19 (`work-pr43r2-off19n` vs `work-pr43r2-onall19n`): **1/19
    moved** — 142421, a single RETEXT `7023 'mu- 4 MeV' -> 'pi+ 4 MeV'`
    (the 1.2 cm shared vertex stub; muon body `7024 mu- 332` and the whole
    tree structure unchanged), nusel 0-diff. Single-knob isolation runs
    prove **K2 is the sole mover** (K1-only and K3-only leave 7023 at mu-).
    Attribution: the long-muon claim now competes at 142421's vertex and
    the ambiguous stub demotes through the existing pion-conversion loop.
    NOTE: this diverges from the pr/44-era display of the same event
    (owner-accepted `7023 mu- 4`); flagged here rather than silently
    shipped — the vertex-region stub label is physically ambiguous between
    mu-/pi+ and the owner should veto the flip if `mu-` is preferred.
- **G4**: `wcdoctest-clus` 1059/1059 assertions PASS (includes the three
  new default-false pins).
- **G5**: runner-TLA compile — knobs-off JSON byte-identical to
  clean-HEAD compile (0-line diff); all three keys present in the
  TaggerCheckNeutrino node when on.
- **G1**: knobs-off vs git-stash clean-HEAD reference
  (`work-pr43r2-off48` vs `work-pr43r2-cleanref48`, rebuilds strictly
  sequenced around the reference batch): **48/48 events, 96/96 archives
  byte-identical** (`hash_archive.py` member-content hashes, fields 1-2).

## Flip — all three SBND PRODUCTION ON (same round)

The owner's round-2 flip policy ("flip if zero nusel verdict flips and
every moved event is attributed to its knob") is met: zero verdict flips on
every arm, and the single moved event (142421, ncpi0-19) is isolated to K2
by single-knob A/B. `wct-pr-perevt.jsonnet` TLA defaults flipped
false→true for all three knobs (cfg-only change, no rebuild). Flip-verify:
bare 3-event run (`work-pr43r2-flipverify`, no env overrides) hash-matches
the gated all-on arm (`work-pr43r2-onall`) exactly on all three events:
`54351 d187a737…`, `56463 3bb36872…`, `57661 2e997765…`.

## Scope and what is NOT claimed

- The 142421 stub relabel (`7023 mu- 4 → pi+ 4 MeV`) diverges from the
  pr/44-era owner-accepted display of that event. It is attributed,
  structure-preserving and nusel-neutral, but the owner may veto: setting
  `single_muon_long_muon_claim=false` in `wct-pr-perevt.jsonnet` (or
  `SBND_SINGLE_MUON_LONG_MUON_CLAIM=0`) restores `7023 mu-` while keeping
  K1/K3.
- K3's rule-1 walk is bounded to 3 hops and only fires behind main-vertex
  protons; forced-electron terminals elsewhere in the graph (not behind a
  proton chain) are out of scope this round.
- The mcp1k manifest beyond the three case events was not censused (only
  nueCC48 + ncpi0-19, matching the pr/44 validation scope).

## Bee evidence

`bee/pr43r2/pr43r2-before.zip` (Phase-0 HEAD baseline) and
`bee/pr43r2/pr43r2-after.zip` (all three knobs on), idx 0/1/2 =
54351/56463/57661 in both.
