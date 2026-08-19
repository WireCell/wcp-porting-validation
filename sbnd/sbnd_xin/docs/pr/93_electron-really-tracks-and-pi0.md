# doc pr/93 — "electrons" that are really tracks, or a hadronic interaction's π⁰ shower (SBND run 18255)

**Status: diagnosis + proposed-design round only.** No C++ change, no jsonnet
knob, no A/B gate. Per owner scope: *"collect some information on the root
cause of these, and think of what can we do to rescue them but not lead to
regression in EM shower."* pr/91 is the template for this shape of round.
**Round 2** (owner: *"why don't you investigate 315167 and add them to the md
file"*) traces Cause D (§1) to rule out three specific mechanisms with direct
evidence and narrows the remaining search to one well-supported, not-yet-
confirmed hypothesis — see the round-2 subsection under Cause D.

Owner's five events, verbatim: *"18255-55595 458 MeV electron: long track in
it. 18255-348471 750 MeV electron: track + shower? 18255-69314 595 electron
track: + multiple gamma ... 18255-292643 289 MeV electron: tracks --> pi0
shower. 18255-315167 1046 MeV electron: multiple tracks in it."* All five are
in run 18255, subrun 1.

All five are members of the **"remaining 42 straight-long e⁻"** residual that
doc pr/40 round 9 explicitly declined to fix and named *"the natural next
trace round"* (`40_track-mis-id-as-electron.md`, §Scope-and-not-claimed). This
is that round.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
ls -la --time-style=full-iso build/clus/libWireCellClus.so   # 2026-08-18 18:45:38
# libs load from build/<pkg>/, not local/lib, in this tree (M1 correction,
# see memory project_wcb_build_race.md) -- this is the freshness proof.

cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
export WCT_PID_WRITE_DEBUG=2 WCT_SHOWER_CONTENT_DEBUG=1
PR_JOBS=4 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh \
    work-mcp1k-cb0805   work-pr93r1-dbg-mcp1k   data 55595 348471 292643 315167
PR_JOBS=1 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh \
    work-nuecc48-cb0805 work-pr93r1-dbg-nuecc48 data 69314

# probes-off control arm, same binary, env the only difference
unset WCT_PID_WRITE_DEBUG WCT_SHOWER_CONTENT_DEBUG
PR_JOBS=4 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh \
    work-mcp1k-cb0805   work-pr93r1-off-mcp1k   data 55595 348471 292643 315167
PR_JOBS=1 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh \
    work-nuecc48-cb0805 work-pr93r1-off-nuecc48 data 69314

python3 scripts/pr85_hash_gate.py work-pr93r1-off-mcp1k   work-pr93r1-dbg-mcp1k
python3 scripts/pr85_hash_gate.py work-pr93r1-off-nuecc48 work-pr93r1-dbg-nuecc48

python3 scripts/pr93_shower_composition.py \
    work-pr93r1-dbg-mcp1k:CASES_mcp1k work-pr93r1-dbg-nuecc48:CASES_69314 \
    work-pr92r2-bare-nuecc48:nueCC48 work-pr92r2-bare-ncpi0:NCpi0 \
    --out docs/pr/pr93-composition.tsv

# round 2 -- 315167 only, add the merge-decision probe to find Cause D's
# absorption site
export WCT_PID_WRITE_DEBUG=2 WCT_SHOWER_CONTENT_DEBUG=1 WCT_SHOWER_MERGE_DEBUG=1
PR_JOBS=1 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh \
    work-mcp1k-cb0805 work-pr93r2-dbg-mcp1k data 315167
unset WCT_PID_WRITE_DEBUG WCT_SHOWER_CONTENT_DEBUG WCT_SHOWER_MERGE_DEBUG
PR_JOBS=1 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh \
    work-mcp1k-cb0805 work-pr93r2-off-mcp1k data 315167
python3 scripts/pr85_hash_gate.py work-pr93r2-off-mcp1k work-pr93r2-dbg-mcp1k
```

**Probe byte-neutrality** (member-content hashes via `pr85_hash_gate.py`, over
`mabc-pr.zip` + `pctree-pr-evt<ID>.tar.gz`, never raw `cmp` — M2): round 1
probes-off vs probes-on — **PASS 8/8 archives** (mcp1k, 4 events) and
**PASS 2/2** (nueCC48, 69314). Round 2 (315167 only, adds
`WCT_SHOWER_MERGE_DEBUG`) — **PASS 2/2**. All arms produced by the same
`build/clus/libWireCellClus.so`, round 1 at mtime 2026-08-18 18:45:38 and
round 2 at 18:58:36 (the peer session's own follow-up build of its round-10
work), both newer than every source file in the tree — no source was edited
this round, in either pass (M1).

Toolkit HEAD `6657e2a5` (a peer session's pr/40 round 10 —
`shower_bragg_protect_start_segment`, confirmed by that session to fire on
**none** of these five events; see coordination note at the end).

## 1. The mechanism, and why it is not one bug

`Shower::calculate_kinematics` (`PRShower.cxx:1189,1307`) copies
`data.particle_type = m_start_segment->particle_info()->pdg()` — a shower
object's displayed identity ("e- 458 MeV") is a snapshot of its **start
segment's** pdg at the moment kinematics runs. Whatever last wrote pdg 11 onto
that segment is the actual root cause; there is no single such writer.

`WCT_PID_WRITE_DEBUG=2` logs every pdg transition site + line number.
Segment ids decode as `cluster_id*1000 + get_graph_index()` (confirmed
memory note), which lets each flagged Bee segment be matched to its own write
tape. Running it on all five gives **four distinct root causes**, matching
what pr/40 round 8 predicted ("not one single place, but multiple places")
and directly falsifying this round's own starting hypothesis (below).

| evt | flagged seg | full pdg tape | root cause |
|---|---|---|---|
| 55595 | 8005 (193.8 cm) | `0→13`(median fallback) `→11` at `NeutrinoTrackShowerSep.cxx:1421` | **A — ungated Case-B reclass** |
| 315167 | 8013 (shower obj, 215.7 cm) | seg 8013 itself: legitimate topology write; but member **8001 (150.7 cm, pdg 2212, proton, score 0.10)** sits inside the shower's membership | **D — proton absorbed into shower's energy/length aggregate, label never re-examined** |
| 292643 | 18009 (84.6 cm shower) | `0→13`(fallback) `→211`(NeutrinoVertexFinder.cxx:2051, pion PID) `→11` at `PRShower.cxx:1008` | **C — the `update_particle_type` vote (H2, as hypothesized)** |
| 348471 | 12007 (53.5 cm) | `2212`(confirmed 3×, `NeutrinoVertexFinder.cxx:2196`) `→11` at `NeutrinoShowerClustering.cxx:merged_shower_start_segment` | **B — unconditional overwrite of a confidently-PID'd proton** |
| 69314 | 3015 (147.1 cm shower) | `0→13`(fallback) `→211`(pion, `:2051`) `→11` at `NeutrinoShowerClustering.cxx:new_shower_accepted` | **B — unconditional overwrite of a confidently-PID'd pion** |

### Starting hypothesis, corrected

Before running the probe, the working hypothesis (from reading
`Shower::update_particle_type`'s accumulator, `PRShower.cxx:964-981` — every
non-shower-flagged member that is not a confirmed proton counts as
`shower_length`, so a muon/pion chain always votes electron) was that this
vote was the common writer for all five. **The tape shows it is the actual
writer for only one of five (292643).** For 55595/348471/69314 the segment
already reads pdg 11 *before* the vote runs — a different, earlier site wrote
it, and the vote (`PRShower.cxx:1008`) merely reconfirms an unchanged value.
This is stated plainly because it looked natural to assume and would have led
to a fix (widening the vote) that misses 4 of the 5 motivating cases.

Also corrected in the course of this: `particle_score==100.0` is **not** a
writer-identifying marker. It is `Segment::m_particle_score`'s constructor
default (`PRSegment.h:185`) *and* the value `segment_determine_dir_track`
leaves in place (`PRSegmentFunctions.cxx:2784`) when the dQ/dx templates
decline and the median fallback writes pdg 13 for MIP-like dQ/dx
(`:2856-2865`). A genuine, correctly-PID'd long MIP muon routinely carries
`score==100.0` — it identifies "no fitted template score", not any particular
call site.

### Cause A — 55595: an ungated branch of `improve_maps_shower_in_track_out`

`NeutrinoTrackShowerSep.cxx:1388-1416`, "Case B" of
`improve_maps_shower_in_track_out`. Condition (paraphrased):
topology counts of shower/track/proton/muon neighbors at both endpoint
vertices satisfy a branching pattern, **and**
`(length < 25cm && pdg != 11) || sg->dirsign() == 0`. Segment 8005 is 193.8 cm
— it can only have reached the write through the **second disjunct**:
`dirsign() == 0`, i.e. the segment's direction was never determined. Once
that fires, length is not consulted at all, and `pdg_code = 11` is written
unconditionally.

Case E, the sibling branch two cases below in the same function
(`:1268-1279`), already carries the pr/40 F2 guard
(`m_shower_reclass_dqdx_guard && segment_dqdx_spares_electron_reclass`).
**Case B has no such guard.** This looks like the F2 fix simply not having
been extended to every case in this function when it shipped.

### Cause B — 348471, 69314: unconditional overwrites at shower-acceptance sites

Two call sites force `set_pdg(11)` on a shower's start segment as a
**side-effect of accepting/merging the shower object**, with **no PID check
at all** on the segment being overwritten:

- `NeutrinoShowerClustering.cxx:3311` (tag `merged_shower_start_segment`) —
  when a shower's own start vertex is found inside another shower's node
  traversal, the two are merged and the start segment is stamped electron.
- `NeutrinoShowerClustering.cxx:2802` (tag `new_shower_accepted`) — a
  candidate shower is admitted purely on aggregate-size gates
  (`total_length`, `n_tracks`, `total_energy` vs fixed cm/MeV thresholds,
  `:2790-2795`), then its start segment is stamped electron.

Both call `set_pdg(11)` directly on the existing `ParticleInfo` object and
**never touch `particle_score`**. The result for 348471's segment 12007 is an
internally self-contradictory record: **`particle_id=11` with
`particle_score=0.23`** — a confident *non*-electron score sitting under an
electron label, because the segment's PID immediately before this write was a
confidently-scored proton (`pdg=2212`, `score=0.23`, confirmed three times at
`NeutrinoVertexFinder.cxx:2196` right before the overwrite). 69314's segment
3015 was a template-PID'd pion (`pdg=211`) at the moment of overwrite. Neither
site asks "does this segment already carry a confident non-electron identity"
before forcing electron — this is the same shape of gap as
`shower_absorb_track_guard` closes for the flood-fill, applied to a different
pair of call sites that currently have no equivalent guard at all.

### Cause C — 292643: the vote, as hypothesized

The one case matching the original hypothesis. Segment 18009 goes
`0→13→211`(pion PID)`→11` and the final write is genuinely
`Shower::update_particle_type` at `PRShower.cxx:1008` — the accumulator
counted the pion-PID'd member as `shower_length` (not `track_length`, which
is reserved for `|pdg|==2212` only) and the vote flipped to electron.

### Cause D — 315167: a structurally different bug — energy/length aggregation, not mislabeling. Traced round 2.

Segment 8013 itself (15.7 cm) is a legitimate small EM stub, correctly
topology-flagged and correctly pdg 11 via the ordinary
`determine_direction`/`kShowerTopology` path
(`NeutrinoTrackShowerSep.cxx:303-326`) — this write is **not** part of the
bug. The bug is that shower object 8013's **membership** also contains
segment **8001: 150.7 cm, `flag_shower=false`, `pdg=2212` (proton),
`particle_score=0.10`** (a confidently-PID'd proton, correctly never
relabeled) — 70% of the shower's declared 215.7 cm length and a large
fraction of its declared 1046.7 MeV. `Shower::calculate_kinematics` computes
the shower's *energy* from `total_length` under the electron mass hypothesis
regardless of what any individual member's own (correct) PID says, so a
150.7 cm proton folded into an "electron" shower inflates its reported energy
under the wrong physics.

**Round 2 trace (`WCT_SHOWER_MERGE_DEBUG=1`, same 315167 event, probe
byte-neutrality PASS 2/2 vs a probes-off control arm).** Three specific
hypotheses were tested and **ruled out** with direct evidence, narrowing the
remaining search space precisely:

1. **Not `examine_shower_1`'s merge pass** (`ex_shower1_merge` tag). Its very
   first log line for this shower already reports `cand_len=215.736
   cand_ke=1046.672` — the shower's *final*, full composition, proton
   included — **before** this pass evaluates a single merge candidate. The
   proton was already a member when this pass started.
2. **Not the flood-fill's own guard failing to apply its predicate.** Read
   directly from the shower's own 252-point fit trajectory in the calib dump
   (not the coarser vertex-to-vertex chord used in §2's post-hoc census):
   `direct_length = 141.14 cm`, `length = 150.73 cm`, ratio **0.936** — this
   satisfies `segment_is_straight_long_track`'s own threshold
   (`direct_length>=34cm` **and** `direct_length>0.93*length`, both true).
   Had `shower_absorb_track_guard`'s `apply_guard` been active when
   `complete_structure_with_start_segment` first considered this segment, its
   `guard_excludes` lambda would have returned true and the proton's far
   vertex would never have been enqueued.
3. **Not the long-muon pre-stamp exemption**
   (`apply_guard = absorb_track_guard && get_particle_type()!=13`, which
   disables the guard entirely for a shower whose *seed segment* carried
   `|pdg|==13` at creation, `NeutrinoShowerClustering.cxx:552-556`). The
   `WCT_PID_WRITE_DEBUG=2` tape for segment 8013 (`clus=8 gidx=13`) shows it
   was born `pdg 0 -> 11` directly at `PRSegmentFunctions.cxx:3046` (the
   ordinary shower-trajectory writer) — it never carried pdg 13, so this
   shower was never exempted from the guard by the long-muon path.
4. **Not `nv_bridge_track` or `shower_stem_backfill`** — neither logs a
   firing line for this event, and `stem_backfill`'s own ceiling
   (`m_stem_backfill_max_len = 30 cm`) would reject a 150.7 cm segment
   outright even if it had run.

So the guarded flood-fill (`shower_clustering_with_nv_in_main_cluster`'s own
call to `complete_structure_with_start_segment`) is **ruled out** as the
absorption site — its guard's own predicate says it should have excluded
this exact segment, and the long-muon exemption that would disable that
guard did not apply. **Leading remaining hypothesis, not yet confirmed**:
`shower_clustering_with_nv_connecting_to_main_vertex` (pass 2 of the four
seeding passes, `NeutrinoShowerClustering.cxx:705+`) has its **own**,
separate per-segment acceptance test —
`!seg_dir_weak(sg1) && (length>3.6cm || (length>2.4cm &&
medium_dQ_dx_norm>2.5))` (`:878-880`) — which is a bare length/dQ-dx
threshold with **no PID check and no straightness check at all**, and is not
gated by `shower_absorb_track_guard` (that guard lives only inside
`complete_structure_with_start_segment`, which this pass's own acceptance
logic does not appear to route every candidate through the same way pass 1
does). A 150.7 cm confidently-PID'd proton trivially clears `length>3.6cm`.
No existing env probe instruments this specific function's segment-accept
decision, so this is reported as the **best-supported open hypothesis, not a
confirmed writer** — confirming it needs one more targeted probe (a new,
env-gated print inside `shower_clustering_with_nv_connecting_to_main_vertex`'s
accept test) in a follow-up round, since adding new instrumentation is a
(byte-neutral, but still new) code change and this round's scope was
diagnosis with the existing probes plus offline analysis.

**Practical implication for a future fix**: if pass 2 is confirmed, the
rescue is structurally identical in shape to K-ACCEPT (§4) — pass 2's own
segment-acceptance test needs the same kind of guard the flood-fill already
has (`shower_absorb_track_guard`'s predicate, or a call to
`segment_is_straight_long_track` directly), since it currently has none.

## 2. Composition census — why a length/segment-count discriminator is not gate-able

Post-hoc, per pdg-11 shower with `kine_best>100 MeV`, `maxUnflag` = longest
non-shower-flagged member carrying pdg 11 itself:

| | evt | E (MeV) | L (cm) | nmem | maxUnflag | maxUnflag/L | seg/cm |
|---|---|---|---|---|---|---|---|
| CASE | 55595 | 458 | 205.1 | 4 | **193.8** | **0.945** | 0.020 |
| CASE | 315167 | 1047 | 215.7 | 4 | 150.7\* | 0.699\* | 0.019 |
| CASE | 348471 | 751 | 128.2 | 16 | 53.5 | 0.417 | 0.125 |
| CASE | 69314 | 596 | 147.1 | 21 | 38.4 | 0.261 | 0.143 |
| CASE | 292643 | 289 | 84.6 | 11 | 22.9 | 0.271 | 0.130 |
| nueCC48 (62) | — | — | — | — | med 16.7/p90 38.5/**max 82.5** | max 0.632 | min 0.054 |
| NCπ⁰ (35) | — | — | — | — | med 10.0/p90 28.6/**max 57.6** | max 1.000 | — |

\* 315167's own `maxUnflag` counts unflagged **pdg-11** members only (its
longest is 7.2 cm); the 150.7 cm value above is its proton member 8001,
reported here for continuity with the "long non-shower object inside" theme
even though it is a different pdg bucket — see Cause D.

**Caveat that governs how this table is used**: both `maxUnflag` and `seg/cm`
are *final-geometry* numbers, measured after every merge and after whichever
writer already fired (pr/83 round-4 lesson: gate on at-decision-time
quantities, not final census). There is no code seat where a shower carries
its final composition and its label has not yet been decided — by the time
these numbers can be computed, causes A/B/C have already run. Keep this table
as evidence the *population is separable*, not as a fix predicate.

**The control tail is itself contaminated with independently-documented
instances of this exact defect** — the raw maximum is not a safe regression
budget:
- nueCC48 **137238** (maxUnflag 82.5, maxUnflag/L 0.632) — pr/74 round-1
  **shape B** roster ("`e-` PF node ≥50 cm, transverse RMS <3 cm — a pencil,
  i.e. a track").
- NCπ⁰ **285567** (57.6) — pr/74 **shape A** roster (`mu- → e- 408 MeV`).
- NCπ⁰ **142421** (41.9, single member, maxUnflag/L 1.000) — pr/44's
  vertex-muon-mis-ID case.
- nueCC48 **235435/2054** (38.6) — the artifact pr/40 round 9 itself flagged
  for owner review.

The true clean-electron ceiling is materially below the raw max in this
table.

### The `trk_frac` replay — the number that actually tests the vote's own logic

`scripts/pr93_shower_composition.py` replays the vote's *own* accumulator
(`trk_frac` = length share of non-shower-flagged members with
`|pdg| ∈ {13,211,2212}`) over the final composition, on both control samples
and the five cases (`docs/pr/pr93-composition.tsv`, 102 rows).

| evt | shower | trk_frac | interpretation |
|---|---|---|---|
| 55595 | 8007 | **0.000** | segment 8005 already reads pdg 11 by dump time — Cause A pre-empted the vote entirely |
| 348471 | 12007 | **0.045** | same — Cause B pre-empted it |
| 69314 | 3015 | **0.050** | same — Cause B pre-empted it |
| 292643 | 18009 | **0.263** | below the vote's own 0.5 flip point *in the final state*, consistent with the vote having already flipped it upstream of a later merge that added more shower-flagged length |
| 315167 | 8013 | **0.699** | above 0.5 — Cause D's proton dominates by length, and yet the label was never re-examined (confirms Cause D is a "vote never re-ran" bug, not a "vote decided wrong" bug) |
| nueCC48 (62) | — | 2/62 > 0.5 | **137238** (already flagged above, pr/74 shape B) and (see below) |
| NCπ⁰ (35) | — | 1/35 > 0.5 | **359980/78100** (0.571) — also on the pr/53 "overclustering-sensitive" roster (a known-touchy event, not independently confirmed as a track-in-shower instance) |

**This result is the round's clearest negative finding: `trk_frac`/vote-
widening (the original hypothesis) reaches only 292643.** For 55595, 348471,
69314 the offending segment is already electron by the time the vote's own
accumulator could see it — a fix at the vote cannot reach them. Fixing the
vote (Cause C) has a very clean margin against both control samples (2/62,
1/35, both independently flagged as touchy events already) but is narrow in
scope for *this* population.

## 3. The π⁰/hadronic class — owner's hint tested in weak and strong form

Owner: *"the hadronic part would stay as a track, not like EM shower topology
for a relatively long region, the stem stays as a long track, not an EM
shower."*

**Weak form (start-segment length alone) fails**: 55% of real nueCC48
electrons (34/62) also have a non-shower-flagged, track-topology start
segment, up to 47.0 cm (evt 116962) — longer than 292643's 22.9 cm stem. An
unbounded graph walk from the shower's attach vertex also fails (it picks up
100–270 cm of the event's *unrelated* prongs even for real electrons, since
real vertices have real tracks nearby).

**Strong form — the directed in-membership track run — matches Cause D and C's
own data directly**, once membership is used instead of a free walk:
292643's shower `18009` has `trk_frac=0.263` from **membership-restricted**
non-shower-flagged track-pdg content (not a free graph walk) — a small but
real track stem inside the object, consistent with "tracks → π⁰ shower."
315167's `trk_frac=0.699` is the sharpest instance in the entire dataset —
its stem-that-stayed-a-track is 150.7 cm, dwarfing every control shower's
value. **The owner's physical picture is directly supported by the
`trk_frac` measurement — it just isn't reachable through the vote for these
particular events, per §2's Cause-D finding, because the vote never re-ran
after the proton arrived.**

`stem_dqdx` (already in the dump, median of the first ~3 samples, in MIP
units): 55595 stem 0.93×MIP, 348471 1.07×MIP, 292643 0.99×MIP, 315167
0.94×MIP, 69314 0.91×MIP — **all pion/muon-like (~1×MIP), none proton-like
(~2×MIP)**. A raw dQ/dx gate would not separate these from real EM showers'
own MIP-like conversion stems, and any such gate has a named adverse
sentinel already in the manifest: NCπ⁰ **506114/19016** is a genuine gamma
with an 11 cm trunk at `direct>0.93·len` — the same geometric shape a naive
stem-straightness gate would flag.

`kine_pio_flag`/`pio_mass`: only 292643 (the owner's explicitly named π⁰ case)
and 55595 (a secondary, unrelated π⁰ artifact from the leftover proton, not
this round's target) carry a nonzero `kine_pio_flag` in the dump; 315167,
348471, 69314 do not — so this event-level flag alone does not identify the
class either.

**Recommendation: defer the π⁰/hadronic class as its own trace, not solve it
this round.** The measurement supports the owner's physical picture
qualitatively (§3 above), but no single available field cleanly separates it
from real EM showers without also flagging a named must-keep sentinel. The
already-ported `track_overclustering` BDT sub-scores `tro_1/2/4/5` (documented
as computing exactly this symptom, "identifies events where a muon track has
been incorrectly clustered with the shower... no bugs found," never mined as
a diagnostic) are the next thing to pull into a composition census before
designing a gate — not attempted this round for time. (`tro_3` is not emitted
in this dump.)

### A physical mechanism for the class, and a discriminator it suggests

The owner separately raised a concrete physical pathway that gives the class
a name: a **charged pion undergoing charge exchange** (e.g.
`π⁺ + n → π⁰ + p`) mid-trajectory, with the resulting π⁰ promptly decaying to
two photons. This is mechanistically distinct from a photon converting near a
primary hadronic vertex — the charged pion travels as an ordinary track
(dE/dx ≈ 1×MIP, indistinguishable from a muon or a charge-exchange-bound
charged pion) up to the interaction point, and the shower begins *at that
point*, mid-flight, with no gap and no separate vertex. This is a direct fit
to 292643's own measurement above: `stem_dqdx=0.99×MIP` (pion/muon-like, not
proton-like) feeding directly into a shower with `trk_frac=0.263` — exactly
the shape a track that later underwent an inelastic hadronic interaction
would leave.

This reframes what the *right* discriminator should look for. §2/§3's
`trk_frac` and `stem_dqdx` are aggregate/endpoint statistics; a charge-
exchange (or any inelastic hadronic scatter that seeds a shower) instead
predicts a **single, localized transition point along the object's own
trajectory** — track-like topology and dQ/dx up to a specific interaction
vertex, shower-like topology and elevated dQ/dx immediately after it, with the
transition itself geometrically sharp (a real inelastic vertex, not a gradual
statistical drift). That is a qualitatively different, and likely more
powerful, discriminator than any of the aggregate quantities measured in this
round — a real EM shower's own conversion stem should not show a comparably
sharp track→shower dQ/dx step at a well-defined 3-D point, since it is not
seeded by an interaction vertex. **Not measured this round** (it needs a
per-point, along-trajectory scan rather than a per-shower aggregate, which the
existing composition census does not attempt) — named here as the most
promising concrete next measurement for the deferred trace, ahead of or
alongside the `tro_1/2/4/5` BDT features above.

## 4. Named rescue designs (documented only — NOT implemented this round)

Each below is scoped to the specific cause it addresses; none is a universal
fix, matching what the tape actually shows.

- **K-CASEB `shower_reclass_case_b_dqdx_guard`** (Cause A, fixes 55595) —
  `NeutrinoTrackShowerSep.cxx:1394-1416`. Extend the same guard Case E already
  carries — `!(m_shower_reclass_dqdx_guard &&
  segment_dqdx_spares_electron_reclass(sg, m_mip_dqdx))` — around Case B's
  `pdg_code=11` write. Default OFF; ON restores parity between the two
  sibling cases in the same function. Population risk: any real EM shower
  whose start segment has `dirsign()==0` at this stage would also be spared
  from Case B — bounded by the existing `segment_dqdx_spares_electron_reclass`
  ratio test (spares when `median/MIP > 1.75` or `< 1.2`), the same population
  pr/40 F2 already validated safe for the sibling case.
- **K-ACCEPT `shower_accept_pid_guard`** (Cause B, fixes 348471, 69314) —
  two call sites, `NeutrinoShowerClustering.cxx:2802` (`new_shower_accepted`)
  and `:3311` (`merged_shower_start_segment`). Before the unconditional
  `set_pdg(11)`, add a check mirroring `shower_absorb_track_guard`'s own
  shape: skip the forced overwrite when the segment already carries
  `has_particle_info() && pdg != 0 && |pdg| != 11 && particle_score() < 1.0`
  (a *confidently* scored non-electron — deliberately narrower than "any
  non-11 pdg", since both motivating segments had scores well under 1.0: 0.23
  and — needs the tape's pre-write score for 69314's pion, not directly
  logged by `WCT_PID_WRITE_DEBUG`, so a follow-up probe should confirm before
  implementation). This is the cleanest fix in the round: it targets a call
  site with **no existing PID check of any kind**, so the population exposed
  is exactly "segments about to be force-relabelled electron while already
  carrying a confident non-electron score" — a population this round's tape
  shows is non-empty and wrong 2/2 times observed.
- **K-VOTE `shower_vote_track_pid_counts`** (Cause C, fixes 292643 only, of
  these five) — `PRShower.cxx:964-976`. Replace
  `is_not_proton = (|pdg| != 2212)` with `is_track_pid = has_particle_info()
  && |pdg| ∈ {13,211,2212}`; bucket test becomes `if (is_shower ||
  !is_track_pid) shower_length += L; else track_length += L;`. No
  score-based conjunct (score 100 is the median-fallback population itself,
  not a confidence signal — gating on it would make this fire on nothing).
  `trk_frac` census (§2): 2/62 nueCC48 and 1/35 NCπ⁰ controls exceed the
  vote's own 0.5 flip point, and both are already independently flagged as
  touchy events (pr/74 shape-B, pr/53 overclustering roster) rather than
  fresh regressions — a clean margin for this knob's narrow scope.
- **Cause D — not designed this round.** Needs
  `WCT_SHOWER_MERGE_DEBUG` to identify the specific absorbing pass before any
  fix can be scoped; named as the immediate next step, not attempted here to
  keep this round to its diagnosis-and-design budget.
- **Not designed this round**: a "retype/split the multi-segment shower"
  knob — `NeutrinoEnergyReco.cxx:327` skips any shower whose kinematics flag
  is already set, and pr/74 documents a dangling-PF-root failure from a prior
  multi-member re-seat attempt at a different site.

None of K-CASEB / K-ACCEPT / K-VOTE touch
`complete_structure_with_start_segment` (the flood-fill itself,
`PRShower.cxx:477`) or `shower_absorb_track_guard` — every cause found this
round is a **labeling** write, not a flood-fill absorption gap, except Cause D
which is a distinct absorption-without-relabeling bug requiring its own
follow-up trace.

## 5. Scope and not-claimed

- Cause D (315167) is narrowed but **still not fully traced**. Round 2 ruled
  out, with direct evidence, the guarded flood-fill, the long-muon exemption,
  and `nv_bridge_track`/`shower_stem_backfill` as the absorption site, and
  named `shower_clustering_with_nv_connecting_to_main_vertex`'s own ungated
  per-segment accept test as the best-supported remaining hypothesis — **not
  confirmed**; confirming it needs a new env-gated probe inside that specific
  function, which is a (small, byte-neutral) code change and was left for a
  follow-up round rather than added under this round's diagnosis-only scope.
- The π⁰/hadronic class (§3) is qualitatively supported by `trk_frac` and,
  now, by a named physical mechanism (charged-pion charge exchange to π⁰),
  but still not reduced to a gate-able predicate this round. Two concrete next
  measurements are now named: `tro_1/2/4/5` BDT features, and — the more
  promising one — a per-point along-trajectory scan for a localized
  track→shower dQ/dx transition, rather than the aggregate statistics this
  round measured.
- K-ACCEPT's exact predicate needs one more probe (pre-write `particle_score`
  for 69314's pion) before implementation; stated as an open item, not
  assumed.
- No knob was implemented, no A/B gate was run, and no source file in the
  toolkit repo was modified this round — `git status` in `toolkit/` is
  unchanged by this round's work.
- `359980`'s NCπ⁰ `trk_frac>0.5` result and `137238`'s nueCC48 result are
  reported, not adjudicated by the owner — flagged for review, not assumed
  contaminated beyond the prior docs' own characterization.

## Cross-links

pr/40 rounds 7–10 (`40_track-mis-id-as-electron.md`, the direct predecessor —
this round is its named "next trace round"), pr/74 (`74_track-shower-
separation-round{1..4}.md`, shape-A/B census and `shower_in_cascade_guard`
family), pr/91 (`91_em-shower-clustering-round1.md`, Route-A/B merge
distinction cited in Cause D), pr/92 round 2 (`92_stray-satellite-shower-
drop.md`, track-vs-EM topology discriminator and the "census both arms"
lesson applied here), pr/46 (`46_evt55595-long-muon-stub-bridge.md`, 55595's
separately-fixed long-muon PF-chain issue — a different symptom on the same
event), pr/80 R4 (`80_vertex-handscan-rules.md`, "no hadronic-shower tag in
the dump" — the standing obstacle for §3), pr/84 round 3
(`84_disconnected-gamma-and-near-vertex-pr.md`, `shower_dedup_start_seg`).

## Coordination note

A peer session (`fix-never-walked-start-vertex`) owns pr/40 round 10 in
`NeutrinoTrackShowerSep.cxx::examine_all_showers`, committed as toolkit
`1a56f86b`+`6657e2a5`; confirmed by that session to fire on none of these five
events. This round makes no source change. K-CASEB's future implementation
site (`improve_maps_shower_in_track_out`, a different function in the same
file) and K-ACCEPT/K-VOTE's sites (`NeutrinoShowerClustering.cxx`,
`PRShower.cxx`) do not overlap that peer's block.
