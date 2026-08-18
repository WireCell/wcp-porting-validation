# doc pr/40 — track (proton/pion/muon) mis-identified as electron: three fixes, SBND PRODUCTION DEFAULT ON

## Repro block

```bash
cd sbnd_xin

# Part 0 attribution probe (temporary instrumentation, since kept as
# permanent WCT_PID_WRITE_DEBUG / WCT_PID_TRACE_DEBUG diagnostics):
PR_JOBS=9 PR_EXTRA_STAGES=pr_display WCT_SHOWER_TOPO_DEBUG=1 WCT_PID_WRITE_DEBUG=1 \
  WCT_PID_TRACE_DEBUG=1 SBND_WCT_LOGLEVEL=trace \
  ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40-probe9b data \
  388 74544 174637 256587 267597 269774 423981 433451 489330

# G0/G1: clean-source reference at the SAME HEAD (stash the pr/40 diff,
# rebuild, run; pop, rebuild) -- see sec 17.2, why the obvious "compare
# against a recent same-day arm" shortcut failed here.
git stash push -m "pr40-g0-clean-check" -- <14 touched toolkit files>
wcbuild
PR_JOBS=6 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40-base48 data
git stash pop
wcbuild   # M1 freshness proof; also cp build/clus/libWireCellClus.so local/lib/
          # if a later build fails at compile (M3-adjacent link-order gotcha, sec 17.2)

# G1/G2/G3/G4: knob-off vs knob-on population, all 48 nueCC48 events
PR_JOBS=6 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40-off48 data
SBND_TRACK_PID_PERSIST_DQDX=1 SBND_SHOWER_RECLASS_DQDX_GUARD=1 SBND_SHOWER_TOPO_DQDX_GUARD=1 \
  PR_JOBS=6 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40-on48 data
python3 ../../abtest/hash_archive.py work-pr40-off48/pr_evt<ID>/{mabc-pr.zip,pctree-pr-evt<ID>.tar.gz}  # vs work-pr40-base48, x48

# flip verification (cfg-only change, no rebuild)
PR_JOBS=1 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40-flip-verify data 388
python3 ../../abtest/hash_archive.py work-pr40-flip-verify/pr_evt388/mabc-pr.zip work-pr40-on48/pr_evt388/mabc-pr.zip
```

## Symptom

Owner report, 9 SBND events where a hadron/muon **track** segment shows up as
an **electron** in the Bee `track_fit` display:

| # | run-event | (x,y,z) cm | cluster.seg | owner note |
|---|---|---|---|---|
| 1 | 18255-388 | (−160.6, 31.7, 425.9) | 23.020 | |
| 2 | 18259-74544 | (131.0, −173.2, 406.6) | 12.018 | |
| 3 | 18255-174637 | (−183.2, −68.5, 307.7) | 9.050 | may be a pion or muon |
| 4 | 18306-256587 | (−91.8, −16.7, 261.7) | 11.079 | |
| 5 | 18279-267597 | (139.2, −154.5, 427.6) | 5.001 | |
| 6 | 18255-269774 | (57.4, −9.2, 117.6) | 13.034 | |
| 7 | 18255-423981 | (53.9, 142.6, 176.2) | 12.013 | |
| 8 | 18255-433451 | (−173.7, 54.0, 235.6) | 4.078 | **long track** mis-ID'd |
| 9 | 18255-489330 | (−40.0, −111.9, 162.9) | 4.018 | |

Owner: *"at some point of the chain, these tracks got mis-identified as the
electron. They may not be one single place, but multiple places. Can you
investigate this thoroughly and provide a fix?"*

All 9 are in the **nueCC48** manifest (`sbnd_xin/valfast/events-nuecc48-cb0805.txt`,
hub `work-nuecc48-cb0805`, reality `data`), so the gate manifest and the study
population are the same 48 events. Segment id decoding
(`clus/src/PrDisplayDump.cxx`, `clus/src/MultiAlgBlobClustering.cxx`):
`id = cluster_id*1000 + segment->get_graph_index()`. `flag_shower` in
`calib-pr-evt<ID>.json` is `kShowerTrajectory || kShowerTopology` **only** —
not the `abs(pdg)==11` disjunct — so `pdg==11 && flag_shower==false` proves a
PID-writing site did it, not the topology/trajectory tests.

## Reproduction (measured directly on the reported segments)

| evt | seg | flag_shower | dirsign | L (cm) | median dQ/dx ÷ 56000 (MIP) |
|---|---|---|---|---|---|
| 388 | 23020 | false | +1 | 2.1 | **2.94** |
| 74544 | 12018 | false | +1 | 11.3 | **2.75** |
| 174637 | 9050 | false | +1 | 24.6 | 1.39 |
| 256587 | 11079 | **true** | +1 | 29.2 | 1.26 |
| 267597 | 5001 | false | −1 | 5.6 | **3.16** |
| 269774 | 13034 | false | +1 | 7.7 | **4.09** |
| 423981 | 12013 | false | +1 | 5.5 | **2.42** |
| 433451 | 4078 | false | +1 | 30.7 | **1.99** |
| 489330 | 4018 | **true** | +1 | 17.6 | **2.73** |

Exactly the split the owner predicted. **Family A** (7/9, no stored shower
flag): a PID/reclassification site wrote pdg 11 directly; 6 of 9 sit above
1.75× MIP, the prototype's own proton threshold. **Family B** (2/9, flag
set): `segment_is_shower_topology` fired.

Population scale on the same 48 events: 1185 main-cluster segments, 1111 pdg
11, of which 249 carry no shower flag; 129 of those (in 42 events) sit above
1.75× MIP with L > 2 cm. The class this bug belongs to is broad, which is why
the fix is gated on the full population (G3/G4), not just the 9.

## Root cause: three distinct mechanisms, one shared root idea

### Root cause 1 — the PID persistence gate discards a computed identity (F1)

`PRSegmentFunctions.cxx`, `segment_determine_dir_track`'s final store:
```cpp
if (pdg_code != 0 && ((dirsign==1 && end_n==1) || (dirsign==-1 && start_n==1))) {
    // ... persist type, mass, AND recompute the 4-momentum ...
}
```
This gates the **type+mass persistence** on the SAME condition that should
only gate the **4-momentum recompute**: whether the segment's direction
points at a topologically free end. Doc pr/7 (2026-07-30) diagnosed this
exactly, and doc pr/31 §10.9 (F8) re-confirmed it at HEAD `6206c46b` — neither
implemented a fix; no knob existed until this round.

Consequence, measured this round: a segment whose dQ/dx genuinely says
"proton" or "muon" — either via the median-dQ/dx fallback
(`medium_dQ_dx > 1.75×MIP ⇒ 2212`, `< 1.2×MIP ⇒ 13`) with `dirsign` left at 0
by that value-agnostic recovery, or via a confident template competition
whose winning direction happens to point at a non-free end — has that
identity **computed and then thrown away**. The segment exits with
`has_particle_info() == false`, indistinguishable from "PID never ran."

### Root cause 2 — wholesale track-to-electron conversion ignores the segment's own charge (F2)

Once a segment has no particle info (root cause 1) or is merely direction-weak
or short, THREE sites in `NeutrinoTrackShowerSep.cxx` convert it to electron
**unconditionally**, with no re-check of its own dQ/dx:

- `examine_all_showers`'s `flag_change_showers` loop — every non-shower
  segment in a shower-dominated cluster (`n_good_tracks==0` and a
  length-ratio family of thresholds) becomes electron.
- `improve_maps_shower_in_track_out` — TWO reclassify loops, one for
  direction-weak/untyped "out_tracks" at a shower-fed vertex, one for
  no-direction segments at the same vertex.
- `improve_maps_no_dir_tracks` Case E — a muon-typed segment (`pdg==13`)
  surrounded by enough showers gets demoted to electron based purely on
  vertex topology (daughter-shower counts, direct-length ratios).

Two of the nine owner cases (423981, 433451) were **already correctly
persisted as proton** by the PID (root cause 1 did not apply — their
direction was confident and the store gate passed) and were converted to
electron anyway by the `out_tracks` loop above, because
`seg_dir_weak(sg)` returned true: both scores (0.1749, 0.1371) sit just above
the proton `is_dir_weak` threshold (0.13 for segments ≥5 cm). This is
independently the mechanism doc pr/9 §6 F2 named and left unimplemented:
*"spare a segment with a stored 2212 and a good score from the wholesale
electron conversion."*

### Root cause 3 — the topology shower test never consults dQ/dx (F3)

`segment_is_shower_topology` (`PRSegmentFunctions.cxx`) builds a per-point
`vec_dQ_dx` array (normalized by `MIP_dQ_dx`) — and, per doc pr/31 GOTCHA 5,
**never reads it for anything but `.size()`**. The whole shower/track call is
decided by a 5-branch geometric spread test alone; the segment's own charge
never gets a vote. Family B (256587, 489330) both fired this test at 1.26×
and 2.73× MIP respectively — a hadron-density profile the test had no way to
see.

## Why it hid

- The persistence gate (root cause 1) makes the failure **silent by
  construction**: a segment with a perfectly good computed identity simply
  has no `particle_info` at all afterward, which every downstream reader
  (correctly, by its own contract) treats the same as "PID never ran."
  Pinning this needed object-level instrumentation, not a `set_pdg(11)`
  grep — the exact gotcha doc pr/9 §7 already paid for with evt 172230:
  *"pinning converters needs OBJECT-level hooks — literal `set_pdg(11)` site
  instrumentation missed it."*
- The wholesale conversion sites (root cause 2) are individually
  **prototype-faithful** (doc pr/9 §4: *"in a shower-dominated cluster, only
  a strong-direction track survives as a track"* is deliberate prototype
  design, not a port bug) — so nothing here looked wrong in isolation; only
  the population census (129 segments across 42 events) showed the class was
  broad.
- The topology test (root cause 3) has carried a dead `vec_dQ_dx` array
  since the port; `pr/31`'s audit flagged it as dead code, not as a missing
  cross-check, because nobody had yet connected it to a track-mislabeled-
  as-shower symptom.
- The un-gated `flag_print` trace that would have shown the PID's own
  conclusion (`segment_determine_dir_track`'s `Seg ... pdg score` line,
  `PRSegmentFunctions.cxx`) was hardcoded `false` with **no config or env
  path at all** — `NeutrinoTrackShowerSep.cxx:173` had
  `// if (seg->cluster() == main_cluster) flag_print = true;` commented out.
  Nobody had ever seen this line fire.

## Fix

Three default-OFF knobs, one per root cause, all threaded through
`TaggerCheckNeutrino` → `PatternAlgorithms` (the same idiom as every other
knob on this branch) and key-suppressed in `cfg/pgrapher/common/clus.jsonnet`
so the knob-off compiled config is byte-identical.

- **F1 `track_pid_persist_dqdx`** (`TrackPidOptions::track_pid_persist_dqdx`,
  `PRSegmentFunctions.cxx segment_determine_dir_track`). When `true`, persists
  type+mass whenever `pdg_code != 0` and gates only the 4-momentum recompute
  on the existing free-end test — the shape doc pr/7 described, though not
  independently re-verified against prototype source this round (the
  `prototype_base` symlink is currently broken on this machine; the toolkit's
  free-end test is used as the stand-in for the prototype's
  `get_particle_4mom(3)>0` guard). When the free-end test fails, the
  4-momentum is a rest-mass-only stub (`E=mass, p=0`), matching the
  prototype's own zero-momentum convention for an uncomputed energy.

- **F2 `shower_reclass_dqdx_guard`** (`PatternAlgorithms`,
  `NeutrinoTrackShowerSep.cxx`, three sites: `examine_all_showers`,
  `improve_maps_shower_in_track_out` ×2, `improve_maps_no_dir_tracks` Case
  E). When `true`, spares a segment from the electron conversion at these
  sites if `segment_dqdx_spares_electron_reclass(seg, m_mip_dqdx)` — the
  segment's own median dQ/dx exceeds 1.75× MIP (proton-like) or falls below
  1.2× MIP (muon-like), the SAME thresholds `segment_determine_dir_track`'s
  own fallback already trusts. A zero/absent median never spares — "no
  evidence" is not "MIP-like evidence." Guards only the *conversion action*,
  not entry to the surrounding `if`/`else if` chain, so a spared segment
  falls through to neither the conversion nor a sibling case in the same
  mutually-exclusive branch (Case E / Case F in `improve_maps_no_dir_tracks`).

- **F3 `shower_topo_dqdx_guard`** (`PatternAlgorithms`,
  `segment_is_shower_topology`, all 4 call sites in
  `NeutrinoTrackShowerSep.cxx` + `NeutrinoVertexFinder.cxx`). When `true`,
  after the existing geometric test and length-based demotions decide
  `flag_shower_topology`, overrides it back to `false` if the segment's
  median dQ/dx is decisively proton- or muon-like (same helper as F2). Runs
  unconditionally — the existing length guards only fire above their own
  length cuts (50 cm legacy, `shower_topo_demote_len`), so this is the only
  guard reachable for Family B's two segments (29.2, 17.6 cm). Does **not**
  touch `flag_dir` or the geometric test itself.

`segment_dqdx_spares_electron_reclass` (new shared helper, `PRSegmentFunctions.h/.cxx`)
is the one piece of logic F2 and F3 both call.

**This is a designed divergence from the prototype, not a port-fidelity
restoration**, for F2 and F3 — the wholesale conversion sites and the
topology test are individually prototype-faithful (see "Why it hid" above).
A new `porting_dictionary.md` section records this explicitly (§ below). F1
is a restoration of prototype behaviour per doc pr/7's reading.

### Runner env escape hatches (`run_pr_chain_batch.sh`)

```
SBND_TRACK_PID_PERSIST_DQDX=1|0
SBND_SHOWER_RECLASS_DQDX_GUARD=1|0
SBND_SHOWER_TOPO_DQDX_GUARD=1|0
```
Same tri-state contract as every other knob on this branch (unset = cfg
default, 1 = force on, 0 = force off).

## Part 0 attribution (which site wrote pdg 11, before any fix)

Object-level instrumentation (`WCT_PID_WRITE_DEBUG`, hooked at the
`particle_info()` setter plus the 6 non-const-ref reseat sites and 12 direct
`set_pdg()` mutations that bypass it) plus an un-gated PID trace
(`WCT_PID_TRACE_DEBUG`) run on the 9 events before any fix landed:

| evt | seg | writer site | PID's own conclusion at that point |
|---|---|---|---|
| 388 | 23020 | `improve_maps_no_dir_tracks` Case E | pdg=13 (muon), score=0.20 — confident, already stored |
| 74544 | 12018 | `examine_all_showers` | pdg=2212, score=100 (abstain-fallback), dirsign=0 — computed, discarded by the persistence gate |
| 174637 | 9050 | `improve_maps_shower_in_track_out` (out_tracks) | pdg=13, score=0.073 — confident, discarded by the persistence gate (free end failed) |
| 256587 | 11079 | `segment_is_shower_topology` direct write | (topology path; PID never ran) |
| 267597 | 5001 | `examine_all_showers` | pdg=2212, score=100, dirsign=0 — same as 74544 |
| 269774 | 13034 | `improve_maps_shower_in_track_out` (no-dir segments) | pdg=2212, score=0.26, dir=−1 — confident, discarded by the persistence gate |
| 423981 | 12013 | `improve_maps_shower_in_track_out` (out_tracks) | pdg=2212, score=0.175 — confident, **already persisted**, overwritten anyway |
| 433451 | 4078 | `improve_maps_shower_in_track_out` (out_tracks) | pdg=2212, score=0.137 — confident, **already persisted**, overwritten anyway |
| 489330 | 4018 | `segment_is_shower_topology` direct write | (topology path; PID never ran) |

This table pins the mechanism that **originally** wrote pdg 11. It predates
F1/F2/F3 and is not the same measurement as the G2 result table below, which
was run with all three knobs together — per-case single-knob attribution
(which of F1/F2/F3 specifically rescues which case) was not isolated this
round; the shape above makes the qualitative split clear (persistence-only:
74544, 174637, 267597, 269774; already-persisted-but-reconverted: 423981,
433451; topology-only: 256587, 489330; a muon-specific demotion: 388) without
claiming exact single-knob credit.

## Gates

**G0/G1 — knob-off byte-identical: PASS, 48/48 events, 96/96 archives.**
First attempt compared against `work-pr39-verify-nuecc48` (built the same
morning, believed to be a valid same-HEAD reference) and got a 46/96 mismatch
— alarming, since the knob-off code path should be provably inert. Two
independent single-event reproductions at current HEAD (`work-pr40-repro-a`,
`work-pr40-repro-b`) both matched `work-pr40-off48` exactly
(`28a1ca60...`), proving the current build is fully deterministic and the
divergence was NOT in this round's code. A `git stash`-based same-HEAD clean
rebuild (`work-pr40-base48`, no pr/40 diff at all) also matched
`28a1ca60...` exactly. **`work-pr39-verify-nuecc48` was simply not a valid
reference for this comparison for reasons out of this doc's scope** (its own
history is not investigated here); the correct reference is
`work-pr40-base48`, and against it: `work-pr40-off48` vs `work-pr40-base48` =
**48/48 events, 96/96 archives (`mabc-pr.zip` + `pctree-pr-evt*.tar.gz`)
byte-identical** via `hash_archive.py`.

**G2 — the 9 cases: 8/9 fixed, 1/9 correctly left alone.**

| evt | seg (on) | off pdg/fs | on pdg/fs | medMIP | verdict |
|---|---|---|---|---|---|
| 388 | 23020 | 11/false | **13**/false | 2.94 | fixed |
| 74544 | 12018 | 11/false | **2212**/false | 2.75 | fixed |
| 174637 | 9050 | 11/false | **13**/false | 1.39 | fixed |
| 256587 | 11079 | 11/true | 11/true | 1.26 | **unchanged** |
| 267597 | 5001 | 11/false | **2212**/false | 3.16 | fixed |
| 269774 | 13034 | 11/false | **2212**/false | 4.09 | fixed |
| 423981 | 12013 | 11/false | **211**/false | 2.42 | fixed |
| 433451 | 4187* | 11/false | **13**/false | 1.99 | fixed |
| 489330 | 4018 | 11/**true** | **2212**/false | 2.73 | fixed |

\* 433451's segment renumbers from `4078` to `4187` (same physical track,
confirmed by position, within 0.04 cm) — the upstream graph's segment
creation order shifts slightly once earlier segments in the same cluster
reclassify, which changes later `get_graph_index()` values. Physically the
same fix, same track.

**256587 is not a bug.** Its own median dQ/dx (1.26× MIP) sits in the
deliberately conservative gap between the muon threshold (< 1.2×) and the
proton threshold (> 1.75×) — F3 correctly declines to override the topology
test's electron call because the evidence is genuinely ambiguous, not because
the guard failed. Loosening the threshold to catch this one case would sweep
a much larger, less-certain population into the guard (see G4) without a
principled reason to move the cut; left as an open residual, not touched this
round.

**G3 — population: 3/48 events move beyond the 9, zero verdict flips.**
`nusel-table.tsv` diff, off vs on, all 48 events: 3 rows differ
(219295, 268067, 52672), **only in the `stmfit` column**
(`contained` ↔ `eval`); the `label` column (nu-candidate / not-tagged
verdict) is unchanged on every one of the 48 events. None of these 3 is among
the 9 owner-reported cases.

**G4 — census: of the 25 segments in the `pdg11 & !flag_shower & med>1.75×MIP
& L>2cm` class (off-arm, all 48 events), 9 move.** Breakdown of what they
move to: 5 → proton (2212), 3 → muon (13), 1 → pion (211). This is the
breadth the flip decision rests on — the fix reaches beyond the 9 named
events by design (it is evidence-based, not event-specific), and the
population effect measured here is narrow (9/1185 main-cluster segments) and
directionally consistent (always toward the pdg the segment's own charge
argues for).

**G5 — evt 388, reported explicitly (target segment is one of the 9, so its
PID is meant to move):**

| quantity | off | on |
|---|---|---|
| main vertex (cm) | (−163.0996, 31.5755, 426.3197) | (−163.1775, 31.4456, 426.1875) |
| kine_reco_Enu (MeV) | 2810 | 2910 |
| numu_score | −0.728 | −1.51 |
| kine_pio_flag | 0 | 0 |

The vertex moves ≈0.19 cm — small, and plausible: segment 23020 sits
essentially at the vertex (0.8 cm away) and the multi-track refit that
positions the vertex can be weighted by nearby segments' particle hypotheses.
This is a genuine, non-zero move, not the "only energy moves" prediction
written into this round's plan — noted here rather than silently revised.
`kine_reco_Enu` and `numu_score` moving is the expected, predicted
consequence of the pdg change (mass and BDT features derived from it);
`kine_pio_flag` is unchanged. (Both off/on values here differ from the
historical doc pr/28 §13.12 snapshot — that snapshot predates ~10 unconditional
commits between pr/28 and this round, each capable of moving these numbers on
its own; the relevant comparison for this gate is pr/40's own off-vs-on
delta, not a re-litigation of that drift.)

**G6 — unit tests: `wcdoctest-clus` 97/97 PASS**, 1003/1003 assertions, both
before and after the SBND flip. New cases: the three F1/F2/F3 defaults added
to "TaggerCheckNeutrino switches are all OFF"; `TrackPidOptions`'s
`track_pid_persist_dqdx` default; a dedicated
`segment_dqdx_spares_electron_reclass` case (no-evidence, proton-like,
muon-like, ambiguous-middle, degenerate-scale), revert-proven (temporarily
widened the proton threshold to 99×, confirmed the test FAILED, restored,
confirmed clean).

**G7 — compiled-config proof: PASS.** Knob-off: all three keys absent from
the compiled `wct-pr-perevt.jsonnet` JSON (checked against a byte-identical
diff to the pre-pr/40 compile). Knob-on (`--tla-code
track_pid_persist_dqdx=true` etc.): all three keys appear with `true`, and
the diff against the knob-off compile is exactly those three lines — nothing
else in the compiled config moves.

## Flip — the SBND production default (owner 2026-08-06)

Owner, in scoping this round: *"Flip to SBND ON if gates pass."* All of
G1/G2/G3 (the stated flip condition) passed:

- G1 48/48 byte-identical.
- G2 fixes 8/9 (the 9th correctly, not incorrectly, untouched).
- G3 shows zero `label` (verdict) regression.

`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`'s three TLA defaults
flipped `false → true`. Verified with a bare single-event run
(`work-pr40-flip-verify`, evt 388, no rebuild needed — cfg-only change):
hash-matches `work-pr40-on48`'s already-gated result exactly
(`33fe37c4f05050902b197d4cdf1156273037c428ff047d3e2cfb2429408948e6`).

## Scope and what is NOT claimed

- **256587 (Family B, 1.26× MIP) is not fixed** — see G2. Open residual, not
  a designed exclusion; a future round could revisit the ambiguous-band
  threshold with a larger sample if the owner wants it narrowed.
- **Per-knob attribution was not isolated.** F1/F2/F3 were gated and flipped
  together (they compound on several of the 9 cases, see Part 0). Isolating
  each knob's individual population effect would need three more 48-event
  runs (F1-only, F2-only, F3-only); not done this round.
- **Not touched, per M15/§5 rule 4** (deliberate prototype-faithful design,
  not a port bug): `examine_all_showers`'s wholesale conversion logic itself
  (doc pr/9 §4), pr/33 P14's sub-pass-1 electron-forcing in
  `shower_clustering_in_other_clusters`, and the five existing
  `porting_dictionary.md` M15 entries (Magnify coords, TrackFitting T frame,
  `calculate_boundary_metric`'s isochronous-endpoint bypass,
  `examine_vertices_3`/`get_local_extension`, `skip_trajectory_point`'s
  unbounded revert). None of these is this round's fix; F1/F2/F3 sit
  alongside them, not in place of them.
- **Round 10's `skip_revert_iso_xext_cut`** (toolkit `a5a824cf`, doc pr/28
  §17) is unrelated: none of its 13 changed events overlaps this round's 9,
  and neither this round's knobs nor round 10's touch the same code paths.
- **The `is_dir_weak` proton/muon thresholds (0.13/0.27, 0.07/0.15)** and the
  main-vertex rescue thresholds (0.09/0.06) that decided several of the 9
  cases (423981, 433451 both cross the proton threshold by 0.02–0.05) remain
  hardcoded and uncalibrated, as recorded in doc pr/8 §6/§11 and doc pr/9 §6
  — this round did not touch them, and their proximity to several of the 9
  cases' scores is worth naming for a future calibration round.
- Two prior-round items this closes: doc pr/7 §5 (F1's persistence fix,
  implemented) and doc pr/9 §6 F2 (the "protect a scored 2212" idea,
  implemented at its sibling `improve_maps_shower_in_track_out` site as well
  as `examine_all_showers`).

## `porting_dictionary.md` entry

New section, "`NeutrinoTrackShowerSep.cxx`'s wholesale track-to-electron
conversion sites and `segment_is_shower_topology` never consult the
segment's own dQ/dx — `shower_reclass_dqdx_guard`/`shower_topo_dqdx_guard`
are designed divergences, not port corrections" — full M15-record with
prototype behaviour description (both mechanisms ARE prototype-faithful),
the measured consequence, and "do NOT make either guard unconditional" (the
ambiguous-band segments, like evt 256587, are intentionally left alone).

## Verification (how the owner re-checks)

```bash
cd sbnd_xin
python3 scripts/analysis/pr40/pr40_seg_pid.py work-pr40-off9-disp work-pr40-on48
python3 ../../abtest/hash_archive.py work-pr40-off48/pr_evt<ID>/mabc-pr.zip work-pr40-base48/pr_evt<ID>/mabc-pr.zip  # x48
./build/clus/wcdoctest-clus
grep -c track_pid_persist_dqdx <(wcsonnet --tla-str input=/dev/null --tla-code 'anode_indices=[0,1]' \
  --tla-str output_dir=/tmp --tla-code run=1 --tla-code subrun=1 --tla-code event=1 --tla-str reality=data \
  cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet)  # 1 (now SBND default true)
```

---

# Round 2 — two follow-on defects from the pr/40 Bee display: F4 zero-energy muon (FIXED, SBND ON), F6 negative-KE stub (FIXED, SBND ON), F5 electron-fathers-proton (NOT fixed, blocked)

## Repro block

```bash
cd sbnd_xin

# G1: knob-off byte-identical, current HEAD, against work-pr40-on48
PR_JOBS=6 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r2-off48 data
python3 ../../abtest/hash_archive.py work-pr40r2-off48/pr_evt<ID>/{mabc-pr.zip,pctree-pr-evt<ID>.tar.gz}  # vs work-pr40-on48, x48

# G2/G4: knob-on population (F5 forced on too, for measurement only -- it is
# NOT the SBND default; see Fix below)
SBND_TRACK_PID_PERSIST_4MOM=1 SBND_SHOWER_PROTON_DAUGHTER_PION=1 SBND_RECLASS_NEVER_COMPUTED_KE_FLOOR=1 \
  PR_JOBS=6 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r2-on48 data
python3 scripts/analysis/pr41/pr41_check.py work-pr40r2-off48 work-pr40r2-on48

# flip verification (cfg-only change for F4/F6, no rebuild)
PR_JOBS=1 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r2-flip-verify data 174637
python3 ../../abtest/hash_archive.py work-pr40r2-flip-verify/pr_evt174637/mabc-pr.zip work-pr40r2-on48/pr_evt174637/mabc-pr.zip

./build/clus/wcdoctest-clus
```

## Symptom

Owner, reviewing the Bee display of the just-shipped pr/40 fix
(`https://www.phy.bnl.gov/twister/bee/set/52dd2243-4e7a-4d55-8831-db12c023d0d5/event/list/`):

- *"for 174637, why the muon has a 0 energy??? this is not correct."*
- *"for 256587, the reason that it is not an electron is that in the end of
  the particle, there is a proton, which is high dQ/dx. These two require
  taking a look to get them completely correct."*
- Clarifying, on 256587 specifically: *"an electron cannot change to proton.
  So the fact that we identified a proton should change it to pion instead
  of electron. Not sure why it was labeled as electron to start with. But
  this is the logic."*

A third defect (F6, negative kinetic energy) was found incidentally while
tracing the first; owner: *"Fix it in this round too."*

## F4 — the rescued muon carries zero energy

**Root cause.** `segment_determine_dir_track`'s final store
(`PRSegmentFunctions.cxx`) gated the 4-momentum computation on the same
free-end test that pr/40's F1 already stopped gating type+mass persistence
on: when the test fails, the stored 4-momentum is a rest-mass-only stub
(`E=mass, p=0`). `D4Vector[0]` is total energy E, and
`Aux::ParticleInfo`'s constructor computes `kinetic_energy = E - mass` — so
the stub reads **exactly 0 MeV**, always, for every segment pr/40's own F1
fix newly rescues into having a stored PID. Evt 174637 seg 9050 (muon,
25.8 cm) is exactly this case.

`segment_cal_4mom` (the function that should have run instead) has **no
actual free-end dependence** — its only direction coupling,
`segment_cal_dir_3vector`, already degrades gracefully to a zero 3-vector
when `dirsign()==0`. The free-end gate on it was external and unnecessary.

**Fix — `track_pid_persist_4mom`** (`TrackPidOptions`,
`PRSegmentFunctions.cxx`). When true, calls `segment_cal_4mom` unconditionally
instead of the rest-mass stub. Owner-approved approach ("Always
segment_cal_4mom") over a narrower `dirsign==0`-only condition.

**Gates.**
- G1 (knob off): **PASS, 48/48 events, 96/96 archives byte-identical**
  (`work-pr40r2-off48` vs `work-pr40-on48`, `hash_archive.py`).
- G2a (evt 174637 seg 9050): off=`(mu-, 0 MeV)`, on=`(mu-, 86 MeV)`. **PASS.**
- G4 census (zero-MeV PF nodes, all 48 events): off=1, on=**0**. **PASS.**

**Flip — SBND production default ON.** Bare single-event run (no env
override, cfg-only change) hash-matches the gated `work-pr40r2-on48` result
for evt 174637 exactly
(`c5c850e34c8b22925acac2ac29ef60f24244eafd5f6918bf396273783dba68f4`).

## F6 — `reclass_pinfo`'s never-computed path reads a negative energy

**Root cause**, found while tracing F4. `reclass_pinfo`
(`NeutrinoTrackShowerSep.cxx`) constructs a `(mass,0,0,0)` `ParticleInfo` and
then, on its non-hadron (`!had`) path, calls
`set_four_momentum(D4Vector(0,0,0,0))` — zeroing E below the mass, so
`kinetic_energy() = 0 - mass`, a **negative** number. Reachable whenever
`reclass_preserve_4mom` (SBND default **true** since pr/40) takes this
branch with a placeholder object that was never actually assigned a computed
4-momentum.

**Fix — `reclass_never_computed_ke_floor`** (`PatternAlgorithms`,
`reclass_pinfo`). When true, the non-hadron/never-computed path leaves the
constructed `(mass,0,0,0)` 4-vector in place instead of zeroing it, so
`kinetic_energy() == 0` rather than `-mass`. All 15 `reclass_pinfo` call
sites thread the new flag.

**Gates.**
- G1: same run as F4 above, same PASS (both knobs off together).
- Population: **not exercised on the 48-event nueCC48 manifest** — a
  negative-energy PF-node census found 0/0 (off/on). This flip rests on
  direct unit-test verification of `reclass_pinfo`'s `!had`/never-computed
  path (`doctest_clus_knob_defaults.cxx`), not on population evidence; stated
  explicitly rather than implying a population gate that wasn't actually
  exercised.

**Flip — SBND production default ON.** Verified by doctest only, per above;
covered by the same G1/flip-verify byte-identical checks as F4 (both knobs
flip together in the same cfg change and the same bare-run verification).

## F5 — an electron cannot father a proton (NOT FIXED — knob stays OFF)

**Root cause.** `set_default_shower_particle_info`
(`NeutrinoPatternBase.cxx`, called from `examine_direction` at stage 4) is
the single choke point where a shower-flagged segment still missing
`particle_info` defaults to electron — mirroring the prototype's
`ProtoSegment::get_particle_type()`, which unconditionally returns 11 for
any shower segment. Neither function ever looks at the graph around the
segment. Evt 256587 seg 11079 (labelled e−, 29.2 cm, median 1.26× MIP — in
the deliberately ambiguous band between pr/40's muon/proton thresholds, so
F2/F3 correctly declined it) **starts exactly at the neutrino vertex**
(d=0.00 cm) and its **far end touches a PID'd, charge-confirmed proton**
(segment 11080, 3.7 cm, median 3.72× MIP, d=0.00 cm) — a daughter an
electron cannot physically produce.

Population census (2209 electron-labelled segments, 48 events) shaped the
rule's width:

| rule | fires |
|---|---|
| any electron segment with a >1.75× MIP *neighbour* (naive) | 348 — rejected |
| + PID'd proton specifically at the far end | 15 |
| + near end **is** the neutrino vertex (graph identity) | 6 |
| **+ proton daughter independently charge-confirmed (>1.75× MIP)** | **5** ← shipped rule |

**Fix (as designed) — `shower_proton_daughter_pion`** (`PatternAlgorithms`,
new helper `segment_has_proton_daughter` in `PRSegmentFunctions.cxx`). When
true, relabels the candidate segment **pion (211)**, not proton — the
owner's explicit correction ("the fact that we identified a proton should
change it to pion instead of electron"), reasoning that the segment itself
cannot BE the proton, only cannot be an electron given what it fathered.

**Why it does not work end-to-end.** Tracing segment 11079's `particle_info`
writes across the whole pipeline (`WCT_PID_WRITE_DEBUG`) found not one
writer but a chain of four:

| # | writer | effect |
|---|---|---|
| 1 | `NeutrinoTrackShowerSep.cxx:234` (`determine_direction`, stage 3) | `pdg 0 -> 11` — unconditional electron default, no `main_vertex` in scope yet |
| 2 | `NeutrinoTrackShowerSep.cxx:929` ×2 | no-op (`11 -> 11`) |
| 3 | `NeutrinoPatternBase.cxx` (`set_default_shower_particle_info`, this fix) | `pdg 11 -> 211` — **the fix fires correctly** |
| 4 | `Shower::update_particle_type` (`PRShower.cxx:788-801`, called from 9 sites in `NeutrinoShowerClustering.cxx`) | `pdg 211 -> 11` — **reverts it** |

Writer #4 unconditionally reasserts electron on a shower's `m_start_segment`
whenever `shower_length > track_length`, with **no** PID or topology
awareness — it runs after #3 in the same pass and silently undoes it. For
256587 specifically, this revert fires: G2b measured off/on both `pdg=11,
flag_shower=true` — **unchanged**. Population-wide, the override survives
end-to-end in **only 1/2209** electron-labelled segments (evt 342199 seg
72098) — the 1-in-2209 case where writer #4 happens not to fire on the same
shower.

**Gate G2b bar, corrected.** The original plan's pass bar required
`flag_shower` to also flip `true -> false`; F5 as designed only ever touches
`particle_info`/pdg, never the shower flags (Bee and `PrDisplayDump` both
read pdg, not `flag_shower`, so this is what the owner's report is actually
about) — the bar was loosened to `pdg == 211` only. Even under the corrected
bar this still fails for 256587, because of the writer-#4 revert above, not
because of an overly strict check.

**Left OFF, not flipped.** `shower_proton_daughter_pion` remains SBND
default false. Turning it on today would not fix the reported case and would
only unpredictably touch the 1-in-2209 segments where writer #4 doesn't
collide with it — not a safe or meaningful flip. `porting_dictionary.md` and
the knob's own docstring (`NeutrinoPatternBase.h`) both record this
end-to-end-broken state explicitly, so a future reader doesn't have to
re-derive the writer chain from scratch.

**Round 3** (not started): guard `Shower::update_particle_type` itself —
most likely threading a `main_vertex`/graph-aware flag through its signature
and its 9 `NeutrinoShowerClustering.cxx` call sites (same mechanical pattern
already used for F6's 15 `reclass_pinfo` sites) so it does not clobber a
start segment already carrying pdg 211.

## Gates summary

| # | gate | bar | verdict |
|---|---|---|---|
| G0 | freshness | `.so` mtime newer than last edit | PASS |
| G1 | knob-off byte-identical | 48/48 events, 96/96 archives vs `work-pr40-on48` | **PASS** |
| G2a | evt 174637 seg 9050 energy | 0 -> nonzero | **PASS** (86 MeV) |
| G2b | evt 256587 seg 11079 pdg | 11 -> 211 | **FAIL** (writer #4 revert, see F5) |
| G4 | census | zero-MeV PF nodes 1 -> 0; proton-daughter rule population effect | PASS (F4); 1/2209 (F5, informational only) |
| G5 | unit tests | `wcdoctest-clus` | **PASS, 98/98 test cases, 1016/1016 assertions** |
| G6 | compiled-config | keys present/absent correctly; flip verified against gated arm | **PASS** |

## Flip — SBND production default (owner 2026-08-06)

Owner: *"for these fixed bugs, we should have their knobs on for SBND as
default."* Two of three are actually fixed:

- **F4 `track_pid_persist_4mom`: SBND ON.** G1/G2a/G4 all pass.
- **F6 `reclass_never_computed_ke_floor`: SBND ON.** Doctest-verified;
  covered by the same G1 byte-identical run as F4.
- **F5 `shower_proton_daughter_pion`: left OFF.** Not actually fixed end to
  end (see above) — flipping it would not deliver the reported behavior and
  CLAUDE.md's stop-and-ask rule for changing a knob's production default
  applies only to changes that are known-correct; this one demonstrably
  isn't yet.

`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`'s F4/F6 TLA defaults
flipped `false -> true`; F5 stays `false`. Verified with a bare single-event
run (`work-pr40r2-flip-verify`, evt 174637, no rebuild needed — cfg-only
change): hash-matches the already-gated `work-pr40r2-on48` result exactly.

## Scope and what is NOT claimed

- **256587 is still not fixed.** This round diagnosed it precisely (four
  sequential writers, the last one previously unknown) but does not close
  it. Round 3 is needed.
- **F6's flip rests on unit-test evidence, not population evidence** — the
  negative-KE precondition was not observed on the nueCC48 manifest in
  either arm. Recorded explicitly rather than implied.
- **F5's helper (`segment_has_proton_daughter`) and its doctest are
  correct and unit-tested** — the defect is entirely in a downstream writer
  this round did not touch, not in the new code.
- `PrDisplayDump.h/.cxx` changes visible in the toolkit working tree during
  this round (`sbnd_xin/docs/pr/42`, the dQ/dx display panel) are **not**
  part of this round and are excluded from this round's commit — different,
  already-documented, concurrent work.

## `porting_dictionary.md` entry

Existing pr/40 section's 256587 note ("not a bug", genuinely ambiguous
median dQ/dx) superseded in part: the *topology* is not ambiguous, only the
intra-segment charge test was the wrong instrument. New section: "
`set_default_shower_particle_info`'s electron default never consults graph
topology — `shower_proton_daughter_pion` is a designed divergence, not a
port correction" — includes the full writer-chain finding and an explicit
"do not flip this knob ON" until the round-3 guard lands.

## Verification (how the owner re-checks)

```bash
cd sbnd_xin
python3 scripts/analysis/pr41/pr41_check.py work-pr40r2-off48 work-pr40r2-on48
python3 ../../abtest/hash_archive.py work-pr40r2-off48/pr_evt<ID>/mabc-pr.zip work-pr40-on48/pr_evt<ID>/mabc-pr.zip  # x48
./build/clus/wcdoctest-clus
grep -E "track_pid_persist_4mom|reclass_never_computed_ke_floor" <(wcsonnet --tla-str input=/dev/null --tla-code 'anode_indices=[0,1]' \
  --tla-str output_dir=/tmp --tla-code run=1 --tla-code subrun=1 --tla-code event=1 --tla-str reality=data \
  cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet)  # both true (now SBND default)
```

---

# Round 3 — closes F5: guard `Shower::update_particle_type` against the proton-daughter-pion override (gate-clean, NOT flipped)

## Repro block

```bash
cd sbnd_xin

# demonstration: evt 256587 seg 11079 survives 211 end-to-end
SBND_TRACK_PID_PERSIST_4MOM=1 SBND_SHOWER_PROTON_DAUGHTER_PION=1 SBND_RECLASS_NEVER_COMPUTED_KE_FLOOR=1 \
  WCT_PID_WRITE_DEBUG=1 PR_JOBS=1 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r3-dbg256587 data 256587
grep "gidx=79 " work-pr40r3-dbg256587/pr_evt256587/stdout.log   # last write: 11 -> 211, no revert

# G1: knob-off byte-identical -- true apples-to-apples needs a git-stash clean
# reference, since F4/F6 are now SBND defaults (round 2) and work-pr40-on48
# predates them
git stash push -m "pr40r3-g1-clean-check" -- clus/inc/WireCellClus/PRShower.h clus/src/NeutrinoShowerClustering.cxx clus/src/PRShower.cxx
wcbuild
PR_JOBS=6 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r3-cleanref48 data
git stash pop
wcbuild   # M1 freshness proof
PR_JOBS=6 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r3-off48 data
python3 ../../abtest/hash_archive.py work-pr40r3-off48/pr_evt<ID>/{mabc-pr.zip,pctree-pr-evt<ID>.tar.gz}  # vs work-pr40r3-cleanref48, x48

# G2/G4: population impact
SBND_TRACK_PID_PERSIST_4MOM=1 SBND_SHOWER_PROTON_DAUGHTER_PION=1 SBND_RECLASS_NEVER_COMPUTED_KE_FLOOR=1 \
  PR_JOBS=6 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r3-on48 data
python3 scripts/analysis/pr41/pr41_check.py work-pr40r3-off48 work-pr40r3-on48
diff <(sort work-pr40r3-off48/nusel-table.tsv) <(sort work-pr40r3-on48/nusel-table.tsv)  # 0 lines

./build/clus/wcdoctest-clus
```

## What this round does

Round 2 traced F5 (`shower_proton_daughter_pion`) end-to-end: the override
fires correctly at `set_default_shower_particle_info`
(`NeutrinoPatternBase.cxx`, `pdg 11 -> 211`) but is silently reverted moments
later by `Shower::update_particle_type` (`PRShower.cxx`, called from 8 sites
in `NeutrinoShowerClustering.cxx`), which unconditionally reasserts electron
on a shower's start segment whenever `shower_length > track_length`. Owner:
*"Yes, I like the first option"* — guard the reassignment itself (skip it
when a start segment is already a proton-daughter-confirmed pion) rather
than broadening the length-majority classification's own criteria (which
would touch every shower in the pipeline with a non-electron, non-proton
PID'd segment, not just this case).

**Why the guard is the narrower fix.** `update_particle_type`'s
classification (`is_shower || is_not_proton`) only exempts a segment from
`shower_length` if it is BOTH not shower-flagged AND specifically PID'd
proton (2212) — every other pdg, including a freshly-relabelled pion (211),
still counts toward `shower_length` regardless of flags. So the broader
option (teach the classification to also recognize pion/muon as track-like)
would change the shower/track majority vote for every shower with such a
segment anywhere in the pipeline; the guard only intercepts the one
reassignment this round's fix specifically produced.

## Fix

`Shower::update_particle_type` (`PRShower.h`/`.cxx`) gains two trailing
parameters, both legacy-default (`nullptr`/`false` = byte-identical):
`VertexPtr main_vertex` and `bool protect_proton_daughter_pion`. Every one
of the 8 `NeutrinoShowerClustering.cxx` call sites already had both `graph`
(reused via the Shower's own `m_full_graph` member, not a new parameter) and
`main_vertex` in scope — no stage-3/stage-4 availability problem here, unlike
F5's own choke point in round 2. Threaded as
`update_particle_type(particle_data, recomb_model, m_mip_dqdx, main_vertex,
m_shower_proton_daughter_pion, m_mip_dqdx_median)`.

Inside the reassignment block, when the guard is active it re-derives
`segment_has_proton_daughter(m_full_graph, m_start_segment, main_vertex,
proton_daughter_mip_dqdx)` and, if it fires, skips the electron reassignment
entirely (if-guarded, not an early `return` — keeps future additions to this
function from being silently skipped for a protected shower).

**MIP-scale finding, not just an implementation detail.** The function's
existing `mip_dqdx` parameter (bound to `m_mip_dqdx` = 50000/units::cm, the
flat-template amplitude used for the reassigned electron's 4-momentum) is a
DIFFERENT scale than what F5's original check used
(`m_mip_dqdx_median` = 43000/units::cm). Reusing `mip_dqdx` for the guard's
re-check would have compared the proton daughter's dQ/dx against a 16%
higher threshold (1.75×50000 vs 1.75×43000) than `set_default_shower_
particle_info` used, meaning the guard could disagree with F5's own verdict
on borderline daughters and silently fail to protect a segment F5 legitimately
relabelled. A sixth parameter, `proton_daughter_mip_dqdx`, carries
`m_mip_dqdx_median` in explicitly so the guard's re-check uses the SAME
scale as the original decision. Do not simplify this back to reusing
`mip_dqdx` — the two scales measure different things and the match is not
coincidental.

**Guard scope is broader than the reported case, by design.**
`segment_has_proton_daughter` requires graph-identity emanation from
`main_vertex`; three of the 8 call sites (`shower_clustering_in_other_
clusters`, `examine_shower_1`, `examine_showers`) process showers in OTHER
clusters, whose own main vertex is not the `main_vertex` argument passed in
— for those, `find_vertices` simply won't match and the guard correctly
returns false. This is why the population measurement (G4 below), not just
the single-event demonstration, is the real check that the guard fires only
where intended.

## Demonstration — evt 256587 seg 11079

`WCT_PID_WRITE_DEBUG` trace, all 3 round-2/3 knobs forced on:

```
gidx=79 pdg 0 -> 11    NeutrinoTrackShowerSep.cxx:234   (stage 3)
gidx=79 pdg 11 -> 11   NeutrinoTrackShowerSep.cxx:929   (no-op) x2
gidx=79 pdg 11 -> 211  NeutrinoPatternBase.cxx:176      (F5 fires)
```

No further write to `gidx=79` for the rest of the run — the
`PRShower.cxx:801` revert that fired every time in round 2 does not fire
here. Confirmed in the final `calib-pr-evt256587.json`: `particle_id: 211`,
`flag_shower: true` (flags intentionally untouched, as designed since
round 2 — see G2b bar below).

## Gates

- **G0 freshness**: `.so` mtime newer than every touched source file,
  verified before each gate below (`wcbuild`, `ls -la`).
- **G1 knob-off byte-identical**: **PASS, 48/48 events, 96/96 archives.**
  `work-pr40r3-off48` (round-3 code, current SBND defaults: F4/F6 on, F5
  off) vs `work-pr40r3-cleanref48` — a git-stash clean-HEAD (`11bbfd75`)
  reference at the SAME defaults, built specifically because `work-pr40-on48`
  (round 1's reference) predates F4/F6 and is no longer a valid "off"
  baseline for this round's diff (confirmed: `work-pr40r3-off48`'s only
  divergence from `work-pr40-on48`, evt 174637's `mabc-pr.zip`, hash-matches
  round 2's `work-pr40r2-on48` exactly — fully explained by F4 being SBND
  default now, not a round-3 regression; see "GOTCHA" in memory).
- **G2a** (evt 174637, unaffected by this round): off=on=86 MeV. **PASS**,
  confirms round 3 didn't disturb F4.
- **G2b** (evt 256587 seg 11079): off `pdg=11`, on `pdg=211`. **PASS** — the
  case this whole investigation started from is now fixed.
- **G4 census**: PF nodes at 0 MeV: 0/0 (F4/F6 unaffected). Segments moved
  `11 -> 211`: **exactly 2** (evt 256587 seg 11079 — newly fixed; evt 342199
  seg 72098 — the one case that already survived in round 2's measurement).
  Matches the round-2 population prediction (5/2209 upper bound on where the
  rule *can* fire; not every one of the 5 is a shower's start segment with
  `shower_length > track_length`, so 2 actually reaching the guarded branch
  is consistent, not a discrepancy).
- **Population regression check**: `nusel-table.tsv` off vs on, all 48
  events, sorted diff: **0 lines** — zero verdict/feature-column impact
  anywhere on the manifest beyond the 2 segments' own pdg.
- **G5 unit tests**: `wcdoctest-clus` **98/98 test cases, 1016/1016
  assertions**, both before and after the guard-structure fix (`return` ->
  `if`-guarded, see Fix above).

## Flip — SBND production default (owner: "flip it default on please")

`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`'s `shower_proton_
daughter_pion` TLA default flipped `false -> true`. Cfg-only change, no
rebuild needed. Verified with a bare single-event run
(`work-pr40r3-flip-verify`, evt 256587, no env override): hash-matches the
already-gated `work-pr40r3-on48` result exactly
(`c2095b1fd3e71a47fa65b9a24a96d1c42837b9747532c6dc49eb5717fb75bc3b`).

All three doc pr/40 round-2/3 knobs (`track_pid_persist_4mom`,
`shower_proton_daughter_pion`, `reclass_never_computed_ke_floor`) are now
SBND production defaults, alongside round 1's original three
(`track_pid_persist_dqdx`, `shower_reclass_dqdx_guard`,
`shower_topo_dqdx_guard`).

## `porting_dictionary.md` / knob docstring updates

Both the `shower_proton_daughter_pion` entry (`porting_dictionary.md`) and
its docstring (`NeutrinoPatternBase.h`) — which round 2 explicitly marked
"KNOWN BROKEN END-TO-END... do not flip" — are updated to record that the
writer-chain is now closed, gates pass, and the knob is flip-ready pending
owner request. The new `PRShower.h` docstring on `update_particle_type`
documents the guard's own default-off, byte-identical contract.

## Scope and what is NOT claimed

- **Not flipped** — see above.
- **The guard's `main_vertex` scoping to the immediate cluster is correct
  behavior, not a limitation** — a shower in a different cluster has a
  different main vertex, and the F5 relabeling rule was never meant to
  apply there. Population measurement (G4) confirms no unintended firing.
- **This closes F5 as a mechanism**, not as an open-ended guarantee that
  every possible proton-daughter-pion case in the full detector will be
  found — the underlying rule (5/2209 in this round's population framing)
  was scoped and owner-approved in round 2; this round only makes that rule
  reach the output reliably.

---

# Round 4 — F7/F8: a pion stops being a Shower, a muon cannot father two protons

## Repro block

```bash
cd sbnd_xin

# G1: knob-off byte-identical -- git-stash clean-HEAD reference
git stash push -m "pr40r4-base48-gate" -- cfg/pgrapher/common/clus.jsonnet \
  cfg/pgrapher/experiment/sbnd/clus.jsonnet cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet \
  clus/docs/porting/porting_dictionary.md clus/inc/WireCellClus/NeutrinoPatternBase.h \
  clus/inc/WireCellClus/PRSegmentFunctions.h clus/inc/WireCellClus/TaggerCheckNeutrino.h \
  clus/src/NeutrinoPatternBase.cxx clus/src/NeutrinoVertexFinder.cxx clus/src/PRSegmentFunctions.cxx \
  clus/src/TaggerCheckNeutrino.cxx clus/test/doctest_clus_knob_defaults.cxx
wcbuild
PR_JOBS=6 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r4-base48 data
git stash pop
wcbuild   # M1 freshness proof; cp build/clus/libWireCellClus.so local/lib/ once (link-order trap, see G0)

# G1/G2/G3/G4: knob-off vs knob-on, all 48 nueCC48 events
PR_JOBS=6 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r4-off48 data
SBND_SHOWER_PROTON_DAUGHTER_PION_DISSOLVE=1 SBND_MUON_MULTI_PROTON_PION=1 \
  PR_JOBS=6 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r4-on48 data
python3 scripts/analysis/pr40r4/pr40r4_check.py work-pr40r4-off48 work-pr40r4-on48
diff <(sort work-pr40r4-off48/nusel-table.tsv) <(sort work-pr40r4-on48/nusel-table.tsv)  # 0 lines

./build/clus/wcdoctest-clus
```

## Symptom

Owner, reviewing the round 2/3 Bee display of the `shower_proton_daughter_pion` (F5) fix:

1. **evt 18306-256587**: *"the particle flow do show the pion+, but we do not see the
   proton after it in the particle flow. Also the end point of this pion+ is at an
   isolated piece. It seems that the EM shower were modified as pion, but not on the
   individual tracks, this is a problem."*
2. **evt 18255-489330**: *"in the particle flow, there is one muon --> two protons.
   This is not physical, in this case, the muon should be changed to pion."*

## Root cause 1 — F5 changes the pdg, not the shower membership (F7)

`shower_proton_daughter_pion` (round 2/3) relabels a shower-flagged segment's `particle_id`
11 -> 211 in `set_default_shower_particle_info` (`NeutrinoPatternBase.cxx`), but never
touches `SegmentFlags::kShowerTrajectory`/`kShowerTopology`. Those flags are exactly what
`shower_clustering_with_nv_in_main_cluster` (`NeutrinoShowerClustering.cxx:116-119`) tests:
`is_shower_seg = flags_any(kShowerTrajectory) || flags_any(kShowerTopology) ||
|pdg|==11`. A relabelled-but-still-flagged segment still satisfies the first two disjuncts,
so a `Shower` object is still rooted there.

Measured on evt 256587 seg 11079 (pi+, 29.1 cm): it owns a 3-segment shower whose OTHER
members are seg 11080 (the charge-confirmed proton daughter, 3.7 cm, PID'd 2212 -- the very
evidence F5 used to relabel 11079 in the first place) and seg 81153 (0.35 cm, cluster 81, a
non-main-cluster fragment). Two consequences, exactly what the owner saw:

- `fill_bee_pf_tree` (`MultiAlgBlobClustering.cxx:1173`) pre-claims every shower-owned
  segment (`used_segs = shower_segs`) before the track BFS runs, so proton 11080 never gets
  its own particle-flow node -- it is silently swallowed into the shower.
- the pi+ Bee node's displayed `end` point is the SHOWER's end (`(-95.0, -10.1, 266.6)`, a
  0.35 cm fragment absorbed from cluster 81) rather than segment 11079's own end
  (`(-90.2, -17.2, 264.3)`) -- the "isolated piece" the owner flagged.

## Root cause 2 — no proton-multiplicity veto on a muon (F8)

Segment 4019 (mu-, 65.2 cm, evt 489330) has its far (non-neutrino-vertex) endpoint at a
vertex where TWO charge-confirmed protons attach (seg 4018, 17.6 cm; seg 4044, 10.0 cm,
both PID'd 2212 with median dQ/dx > 1.75x MIP). The prototype has no PID rule that
consults a track's neighbor multiplicity at all -- there is nothing to "correct", this is
new physics-motivated logic the owner requested directly.

Segment 4019 sits behind a degree-2 kink vertex (4007) from a second muon segment (4043,
28.4 cm, running to the neutrino vertex). **Owner decision:** relabel only 4019; 4043 stays
mu-. (Asked directly during planning: propagating pion across the kink was rejected in
favor of the narrower change.)

## Fix

**F7 `shower_proton_daughter_pion_dissolve`** (`NeutrinoPatternBase.cxx`,
`set_default_shower_particle_info`): when the F5 override fires (or, on re-entry, when a
segment is already 211 from a prior pass -- `examine_direction` runs more than once, so the
OVERRIDE test widens to `pdg()==11 || (dissolve-knob-on && pdg()==211)` to make the clear
idempotent), also `unset_flags(kShowerTrajectory)` / `unset_flags(kShowerTopology)`. With
the flags gone, `is_shower_seg` at the shower-clustering seed no longer fires for that
segment, so no `Shower` is rooted there and its neighbours stay ordinary tracks.

**F8 `muon_multi_proton_pion`** (`NeutrinoPatternBase.cxx`, new
`override_muon_multi_proton_pion`, called immediately after
`set_default_shower_particle_info` in `examine_direction` -- same per-cluster `main_vertex`,
same last-word-before-shower-clustering position): for every non-shower-flagged, PID'd-muon
segment, test the new `segment_at_multi_proton_vertex` (`PRSegmentFunctions.h/.cxx`, sibling
of F5's `segment_has_proton_daughter`) at EITHER endpoint other than `main_vertex`, with
`min_protons=2`. On a fire, relabel to pion (211) via `segment_cal_4mom`. Both knobs are
config keys threaded the same way as every other pr/40 knob (default `false`, key-suppressed
when off).

## Demonstration

```
=== F7 (evt 256587) ===
  off: seg 11079 pdg=211 flag_shower=True  shower_id=11079
  off: seg 11080 (proton) pdg=2212 flag_shower=False shower_id=11079
  on:  seg 11079 pdg=211 flag_shower=False shower_id=-1
  on:  seg 11080 (proton) pdg=2212 flag_shower=False shower_id=-1
  on mc.json: node 11079 text='pi+  105 MeV' end=[-90.17, -17.17, 264.35]
  on mc.json: children of 11079: [11080]        <- proton is now a PF child
  VERDICT: proton 11080 is a direct PF child of pion 11079: True
  VERDICT: node end is segment's OWN end (not shower fragment's): True

=== F8 (evt 489330) ===
  off: seg 4019 pdg=13
  off: seg 4043 (sibling, should stay mu-) pdg=13
  on:  seg 4019 pdg=211
  on:  seg 4043 (sibling, should stay mu-) pdg=13   <- unchanged, as decided
  on mc.json: node 4019 text='pi+  188 MeV'
  on mc.json: node 4043 text='mu-  92 MeV'
```

Both owner-reported cases fixed exactly as specified.

## Population census (48-event nueCC48, off vs on)

Exactly 5 segments move, all attributable:

| evt | seg | off pdg/flag_shower | on pdg/flag_shower | why |
|---|---|---|---|---|
| 256587 | 11079 | 211 / shower | 211 / **track** | F7 fires (owner case 1) |
| 256587 | 81153 | 13 / track | **11** / track | released from the dissolved 11079-shower; no longer inherits the shower's member-pdg vote, defaults to electron on its own (non-main cluster, no `mc.json` visibility) |
| 342199 | 72098 | 211 / shower | 211 / **track** | F7 fires (same class as case 1, not owner-reported but predicted by the round-2/3 population census of 2/2209) |
| 342199 | 72096 | 13 / track | **11** / track | same release-from-dissolved-shower effect as 81153 (non-main cluster) |
| 489330 | 4019 | 13 (mu-) | **211 (pi+)** | F8 fires (owner case 2) |

The 81153/72096 secondary moves are a direct, expected consequence of dissolving a shower:
a member segment that is itself independently shower-flagged, previously absorbed and
majority-voted to the dissolved shower's pdg, is no longer absorbed and gets its own
DEFAULT-fill pdg (11, ordinary shower default) instead. Both are on non-main clusters, so
neither produces an `mc.json`/Bee visibility change -- but G3's `nusel-table.tsv` diff
confirms they cause zero downstream verdict/feature impact either.

## Gates

- **G0 freshness**: `local/lib/libWireCellClus.so` newer than every touched source file,
  verified before each gate. Hit the documented link-order stale-.so trap once (`local/lib`
  preceded `build/clus` in the linker's `-L` search order after a prior session's
  install left a stale copy there): `cp build/clus/libWireCellClus.so local/lib/` once,
  then `wcbuild` succeeded.
- **G1 knob-off byte-identical**: **PASS, 48/48 events, 96/96 archives.** `work-pr40r4-off48`
  (current code, both round-4 knobs at their false default) vs `work-pr40r4-base48`
  (git-stash clean-HEAD `9d5c6a9a` reference, same 48 events) -- `abtest/hash_archive.py`
  member-content hashes of `mabc-pr.zip` + `pctree-pr-evt<ID>.tar.gz`: **0 mismatches**.
- **G2a** (evt 256587 seg 11079): proton 11080 becomes a direct PF child, node end point is
  the segment's own. **PASS.**
- **G2b** (evt 489330 seg 4019/4043): 4019 -> pi+, 4043 stays mu-. **PASS.**
- **G3 population**: `nusel-table.tsv` off vs on, 48 events, sorted diff: **0 lines** -- zero
  verdict/feature-column impact anywhere on the manifest.
- **G4 census**: exactly 5 segments move (table above), all attributed; zero unexplained
  moves.
- **G5 unit tests**: `wcdoctest-clus` **99/99 test cases, 1025/1025 assertions**.
- **G6 compiled-config**: both keys absent from the compiled JSON with the knobs at their
  function defaults (`false`); both present, `true`, with `--tla-code
  shower_proton_daughter_pion_dissolve=true --tla-code muon_multi_proton_pion=true`.
  Verified directly with `wcsonnet`.

## Flip — SBND production default

`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`'s `shower_proton_daughter_pion_dissolve`
and `muon_multi_proton_pion` TLA defaults flipped `false -> true`. Cfg-only change, no
rebuild needed. Verified with a bare 2-event run (`work-pr40r4-flip-verify`, evts 256587 +
489330, no env override): both `mabc-pr.zip` hashes match the already-gated
`work-pr40r4-on48` result exactly --
`b6cfa864...c04c58b` (256587) and `a0980d7f...9e88b4` (489330).

Both doc pr/40 round-4 knobs are now SBND production defaults, alongside all six of rounds
1-3 (`track_pid_persist_dqdx`, `shower_reclass_dqdx_guard`, `shower_topo_dqdx_guard`,
`track_pid_persist_4mom`, `shower_proton_daughter_pion`, `reclass_never_computed_ke_floor`).

## `porting_dictionary.md` entries

Two new sections (both designed divergences, no prototype anchor -- the prototype has
neither a shower-dissolution-on-relabel mechanism nor a proton-multiplicity veto on track
PID): "Relabelling a shower segment's PDG does not make it stop being a Shower" (F7) and
"A muon segment cannot terminate in a multi-proton hadronic vertex" (F8).

## Scope and what is NOT claimed

- **No propagation across the degree-2 kink** (evt 489330 seg 4043 stays mu-) -- owner's
  explicit choice, asked and answered during planning; not derived from any population
  measurement.
- **The two secondary moves (81153, 72096) are not independently owner-reviewed** -- they
  are a mechanical, fully-explained consequence of F7 dissolving a shower, verified to have
  zero downstream verdict impact (G3), but not a case the owner looked at directly.
- **This is not a general "clean up multi-membership shower artifacts" pass** -- only the
  specific mc.json/particle-flow consequences of F5's earlier relabelling are addressed.

---

# Round 5 -- three owner cases: three segment-level fixes implemented, gate-clean when
off, but **G2 FAILS on all three and one causes a new regression** (NOT flipped, negative
result)

## Repro block

```bash
cd sbnd_xin

# G1: knob-off byte-identical -- git-stash clean-HEAD reference
git stash push -m "pr40r5-base48-gate" -- cfg/pgrapher/common/clus.jsonnet \
  cfg/pgrapher/experiment/sbnd/clus.jsonnet cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet \
  clus/inc/WireCellClus/NeutrinoPatternBase.h clus/inc/WireCellClus/PRSegmentFunctions.h \
  clus/inc/WireCellClus/TaggerCheckNeutrino.h clus/src/NeutrinoShowerClustering.cxx \
  clus/src/NeutrinoTrackShowerSep.cxx clus/src/PRSegmentFunctions.cxx clus/src/TaggerCheckNeutrino.cxx \
  clus/test/doctest_clus_knob_defaults.cxx
wcbuild   # hit the M1 link-order stale-.so trap here; cp build/clus/libWireCellClus.so local/lib/, re-run
PR_JOBS=6 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r5-base48 data
git stash pop
wcbuild   # M1 freshness proof again (hit the trap a second time on restore)
PR_JOBS=6 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r5-off48 data
# 48/48 events, 96/96 archives (mabc-pr.zip + pctree-pr-evt<ID>.tar.gz), 0 mismatches

# the three owner cases, all three knobs forced on
SBND_TRACK_PID_PERSIST_DQDX_ELECTRON_GUARD=1 SBND_SHOWER_CONNECT_MAIN_VERTEX_STRAIGHT_GUARD=1 \
  SBND_SHOWER_TRAJ_STRAIGHT_GUARD=1 PR_JOBS=1 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-ncpi0-cb0805 work-pr40r5-cases data 84229
SBND_TRACK_PID_PERSIST_DQDX_ELECTRON_GUARD=1 SBND_SHOWER_CONNECT_MAIN_VERTEX_STRAIGHT_GUARD=1 \
  SBND_SHOWER_TRAJ_STRAIGHT_GUARD=1 PR_JOBS=1 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr40r5-cases data 54341 55715

# isolation check for the 15005 regression (F11 alone)
SBND_TRACK_PID_PERSIST_DQDX_ELECTRON_GUARD=0 SBND_SHOWER_CONNECT_MAIN_VERTEX_STRAIGHT_GUARD=0 \
  SBND_SHOWER_TRAJ_STRAIGHT_GUARD=1 PR_JOBS=1 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 /home/xqian/tmp/pr40r5/verify-f11only data 55715
# seg 15005: pdg 211 (all-off) -> pdg 11 (F11 alone). F11 owns the regression.

./build/clus/wcdoctest-clus   # 100/100 test cases, 1035/1035 assertions
```

## Owner's three cases and the attribution (Phase 1)

| run-evt | owner's point | owner's id | reading | topology |
|---|---|---|---|---|
| 18364-84229 | (73.0, 129.2, 380.0) | seg 19038 | electron -> muon? (stopping mu + Michel) | main vtx `19042`(d2) -> seg `19039` (e-, 4.9cm) -> vtx `19043`(d3) -> seg `19038` (e-, 21.2cm, owner's point) + 3.6cm shower stub `19040` |
| 18255-54341 | (134.7, 168.6, 155.5) | seg 18007 | electron -> muon? (stopping mu + Michel) | main vtx `18002`(d2) -> seg `18005` (e-, 21.3cm) -> vtx `18004`(d3) -> `18006` (1.7cm, shower) + `18007` (1.7cm, mu-, owner's point) |
| 18255-55715 | (-38.6, -37.5, 492.2) | seg 15007 | not-electron -> muon (exiting mu, pi+ parent is wrong) | main vtx `15003` -> seg `15005` (pi+, 6.1cm) -> vtx `15004`(d4) -> seg `15007` (e-, 14.7cm, `flag_shower` 1, owner's point) + `15006` (proton, 15.4cm) + `15035` (1.1cm) |

Phase 1b's per-segment `WCT_PID_WRITE_DEBUG`/`WCT_PID_TRACE_DEBUG`/`WCT_SHOWER_TOPO_DEBUG`
trace found **three independent mechanisms**, not one shared bug (owner: "Fix all three"):

- **F9** (`84229`): F1's persist-on-dQ/dx rescue (`track_pid_persist_dqdx`,
  `PRSegmentFunctions.cxx` `segment_determine_dir_track`) fires unconditionally once
  `pdg_code != 0`, including on a segment whose own free-end direction test failed --
  this is the round-1-introduced regression that turned 19038 from muon (pre-pr/40) into
  electron. Bisected and confirmed necessary+sufficient with a single-knob-off arm.
- **F10** (`54341`): `shower_clustering_connecting_to_main_vertex`
  (`NeutrinoShowerClustering.cxx`) has three skip branches (`pdg==11`,
  `pdg==2212 && dqdx`, `pdg==211 && dqdx`) but none for a long, straight track with no
  confident PID yet -- so seg 18005 gets absorbed into the shower seeded downstream of it
  before track PID ever gets a chance to run on it.
- **F11** (`55715`): `segment_is_shower_trajectory` (`PRSegmentFunctions.cxx`) has no
  straightness exemption at all (only `segment_is_shower_topology` got one, pr/40 F3) --
  seg 15007, straight and MIP-like but under the 34cm absolute-length floor, gets the
  trajectory-door shower flag regardless.

None of the three measured segments reach the existing 34cm absolute-length threshold at
`NeutrinoVertexFinder.cxx:1432-1447`; all three fixes therefore rely on the ratio branch
(`direct_length > 0.93*length`), added as a new shared helper
`segment_is_straight_long_track` (`min_length=10cm`, `min_direct=34cm`,
`straight_ratio=0.93`). Prototype cross-check (`prototype_base/pid/`): none of the three
sites have an analogous straightness/muon guard, and there is no Michel/stopping-muon rule
anywhere in the PR chain (only in the downstream STM/cosmic taggers) -- all three are
**designed divergences**, not port-fidelity fixes (M15).

## Fix (implemented, gate-clean off, three new knobs)

- **F9** `track_pid_persist_dqdx_electron_guard` -- narrows F1's rescue condition in
  `segment_determine_dir_track`: the unconditional-persist branch is skipped specifically
  when the would-be persisted pdg is 11 and the free-end direction test itself failed
  (`pdg_code == 11 && !free_end_dir`). Everything else about F1 (`track_pid_persist_dqdx`)
  is untouched, including the separate F4 `track_pid_persist_4mom` gate.
- **F10** `shower_connect_main_vertex_straight_guard` -- adds a fourth skip branch to
  `shower_clustering_connecting_to_main_vertex`'s existing three-branch guard: `if
  (segment_is_straight_long_track(sg)) continue;`.
- **F11** `shower_traj_straight_guard` -- `segment_is_shower_trajectory` gains a fourth,
  default-`false` parameter; when true, a shower-trajectory verdict is overridden to
  `false` if `segment_is_straight_long_track(seg)` also holds. Threaded only at
  `NeutrinoTrackShowerSep.cxx`'s `separate_track_shower` call site (the one Phase 1
  identified for 55715) -- the other three call sites
  (`NeutrinoVertexFinder.cxx:93,2547-2548`, `PRSegmentFunctions.cxx:2705`) keep the 3-arg
  form and default to the legacy `false`, out of this round's scope.

Plumbing follows every prior pr/40 round exactly: `NeutrinoPatternBase.h` members ->
`TaggerCheckNeutrino.{h,cxx}` (config read, `default_configuration()` round-trip,
`pattern_algos.m_... =`) -> `cfg/pgrapher/common/clus.jsonnet` (key-suppression idiom) ->
`cfg/pgrapher/experiment/sbnd/{clus,wct-pr-perevt}.jsonnet` threading (4 sites each, same
shape as F7/F8) -> `run_pr_chain_batch.sh` tri-state env overrides -> doctest default-false
assertions plus a 5-check hand-built-graph test case for `segment_is_straight_long_track`.

## Demonstration -- the segment-level fix works, the Bee/mc.json outcome does not (G2)

All three arms below use `work-pr40r5-cases` (all three knobs forced on):

| case | segment pdg, intended | segment pdg, measured | mc.json PF tree, measured | verdict |
|---|---|---|---|---|
| 84229 | seg 19038 -> mu- | **mu- (13), correct** | still ONE node: `id=19039 'e- 89 MeV' end=[73.0,129.2,380.0]` -- unchanged from before the fix | **G2a FAIL** |
| 54341 | seg 18005 -> mu- (stem), 18006/18007 -> e- child(ren) | seg 18005 = **proton (2212)**, 18007 = mu- (13) | split shape achieved: `id=18005 'proton 171 MeV'` -> children `18006 'e- 19 MeV'`, `18007 'mu- 11 MeV'` | **G2b FAIL** (shape right, stem label wrong) |
| 55715 | seg 15007 -> mu-, seg 15005 unchanged (pi+) | seg 15007 = **mu- (13), correct**; seg 15005 = **e- (11), was pi+ (211)** | ONE node: `id=15005 'e- 105 MeV' start=[-48.0,-38.2,482.1] end=[-34.5,-38.2,496.8]` -- covers all of 15005+15006+15007, endpoint lands at the owner's clicked point | **G2c FAIL + regression** |

## Root cause of the G2 failures

The displayed Bee/mc.json outcome is decided at the **shower seeding/absorption
boundary**, not at the segment's own pdg -- fixing a segment's pdg upstream of that
boundary does not change what a `Shower` rooted at a neighbor renders:

- `Shower::complete_structure_with_start_segment` (`PRShower.cxx:337-408`) flood-fills the
  downstream sub-tree from a shower-seeded segment with **no per-segment shower test** --
  once any neighbor is legitimately shower-flagged (84229's true 4.9cm stem `19039`;
  55715's now-declassified-from-a-different-door `15007`... see below), the whole
  downstream chain is swallowed regardless of the other members' own pdg.
- `Shower::update_particle_type` (`PRShower.cxx:788`) then sets the **shower's start
  segment** to pdg 11 whenever non-proton member length exceeds proton member length -- a
  confident `mu-` member counts as shower length, same class of bug round 3's
  `protect_proton_daughter_pion` guard fixed for a *different* trigger (a relabelled
  pion daughter), but with no equivalent guard for a muon member.
- A second, independent seeding path, `shower_clustering_with_nv_in_main_cluster`'s
  `is_shower_seg` test (`NeutrinoShowerClustering.cxx:116-119`), is what actually reaches
  84229 and 55715 -- **F10 only gates
  `shower_clustering_connecting_to_main_vertex`**, a sibling function, so it does not
  intercept this path at all. This is why 84229's fix (a different mechanism, F9) shows no
  display change, and why 55715's absorption still happens even with F11 active.

**84229** (G2a): 19038's own pdg is fixed by F9, but it is absorbed into the shower seeded
by its neighbor 19039 (a genuine 4.9cm shower-flagged stub, correctly flagged, out of
scope for any of the three fixes) via the flood-fill above -- the display cannot change
without also touching the seeding/absorption boundary itself.

**54341** (G2b): the split shape IS achieved (F10 correctly stops the 21.3cm stem from
being absorbed into the downstream shower), but once un-shielded from the shower path, seg
18005 goes through **ordinary track PID** for the first time -- not through any of the
three round-5 guards -- and that PID concludes proton (2212) from the segment's own dQ/dx,
which runs elevated (~1.55x MIP) plausibly from Bragg-peak rise near the stopping point.
Confirmed by `WCT_PID_WRITE_DEBUG`: no logged transition through pdg 11 for this segment in
the post-fix arm at all (unlike every other cluster-18 segment, which is logged going
`0->11->11` via the normal shower-default path) -- 18005 is classified 0->2212 directly by
track PID, bypassing the shower-labeling machinery the guards target. **This exposes a
pre-existing gap in ordinary track PID** (no muon-vs-proton Bragg-peak discrimination for a
short, high-dQ/dx track end) that the shower flag was incidentally masking, not a defect
introduced by F9/F10/F11.

**55715** (G2c, regression): isolated with a single-knob arm (`F11=1`, `F9=F10=0`) against
an all-off baseline -- seg 15005 reads `pdg=211` with all three knobs off, and `pdg=11`
with **F11 alone** on. F11 clearing 15007's shower-trajectory flag removes it as a shower
seed at its own door, but `is_shower_seg` (`NeutrinoShowerClustering.cxx:116-119`) then
re-seeds the shower one segment further up, at 15005 -- which the owner's Round-5 planning
answer explicitly said must stay untouched ("Only 15007 becomes mu-"). **This is a genuine
new regression against an explicit owner decision**, not merely an unmet display goal.

## Gates

- **G0 freshness**: `local/lib/libWireCellClus.so` newer than every touched source file.
  Hit the M1 link-order stale-.so trap twice (once after the initial edits, once after the
  `git stash pop` restore) -- both times fixed with `cp build/clus/libWireCellClus.so
  local/lib/`, re-run `wcbuild`.
- **G1 knob-off byte-identical**: **PASS, 48/48 events, 96/96 archives, 0 mismatches.**
  `work-pr40r5-off48` (round-5 code, all three new knobs at their `false` default) vs
  `work-pr40r5-base48` (git-stash clean-HEAD `3fa0aeb3` reference, same 48 events),
  `abtest/hash_archive.py` member-content hashes.
- **G2a/b/c**: **FAIL, all three** -- see Demonstration table above.
- **G3/G4**: not run -- gated on G2 passing first (CLAUDE.md M13/§5 discipline: don't scale
  up measurement on a fix that doesn't yet do what it's meant to).
- **G5 unit tests**: `wcdoctest-clus` **PASS, 100/100 test cases, 1035/1035 assertions**
  (round-5's new `segment_is_straight_long_track` case included).
- **G6 compiled-config**: **PASS.** All three keys absent from the compiled JSON with the
  knobs at their function defaults (`false`); all three present and `true` with
  `wcsonnet --tla-code track_pid_persist_dqdx_electron_guard=true --tla-code
  shower_connect_main_vertex_straight_guard=true --tla-code
  shower_traj_straight_guard=true`.

## Flip -- NOT flipped

All three `wct-pr-perevt.jsonnet` TLA defaults stay `false`. G1/G5/G6 are clean, so the
code is safe to land (byte-identical off, no build/test regression), but **G2 fails on
every one of the three owner-reported cases**, and F11 introduces a **new regression
against an explicit owner decision** (seg 15005 in 55715). CLAUDE.md §5 rule 5 ("an A/B
gate FAILs and the diff is not explained by your intended change: report... and stop
iterating") and rule 7 ("a physics number looks wrong: report, don't tune parameters to
make it look right") both apply -- 54341's proton-vs-muon call is exactly that kind of
number. This round is landed as infrastructure (the three knobs, the shared
`segment_is_straight_long_track` helper, the attribution) but not as a fix.

## `porting_dictionary.md` entries

Three new sections recording F9/F10/F11 as designed divergences (no prototype anchor --
confirmed against `prototype_base/pid/`), each explicitly marked **"segment-level fix
only; does not reach the Bee/mc.json display outcome without also changing the shower
seeding/absorption boundary (`PRShower.cxx:337-408`, `PRShower.cxx:788`,
`NeutrinoShowerClustering.cxx:116-119`); not flipped; G2 open."**

## Scope and what is NOT claimed

- **This round does not fix any of the three owner-reported display cases.** The
  segment-level pdg is corrected in all three (19038, 15007 confidently; 18005 changes but
  not to the expected label), but the visible Bee/mc.json outcome only changed for 54341,
  and even there not to the intended label.
- **The 15005 regression is a real, confirmed defect** against the owner's explicit
  Round-5 planning answer, not a hypothetical risk -- isolated to F11 alone via a clean
  single-knob A/B, reported here rather than patched around.
- **The 54341 "proton" outcome is not diagnosed as a bug in F9/F10/F11** -- it is ordinary
  track PID running for the first time on a previously shower-shielded segment and calling
  a plausible-but-unverified proton, likely from Bragg-peak dQ/dx rise. Whether that is the
  right call physically is an open question for the owner, not something this round tunes.
- **No F12/redesign attempted in this round.** The structural fix implied by the root-cause
  analysis (teaching the seeding/absorption boundary itself about a confident non-electron
  member, in the shape of round 4's F7 shower-dissolve) is a materially bigger change than
  three surgical guards and needs its own scoping decision from the owner before
  implementation.

# Round 6 -- the boundary-level fixes: F12 absorption guard + F14 Michel-stem rescue
(F13 measured dead), all three owner cases fixed, owner-accepted residuals on the
parent stubs

## Repro block

```bash
cd sbnd_xin

# Phase-1 traces (round-5 F11-only regression writer + 54341 proton writer)
WCT_PID_WRITE_DEBUG=1 SBND_SHOWER_TRAJ_STRAIGHT_GUARD=1 PR_JOBS=1 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 /home/xqian/tmp/pr40r6/trace-55715-f11only data 55715
# -> "PID_WRITE_DEBUG set_pdg ... clus=15 gidx=5 pdg -> 11 at NeutrinoShowerClustering.cxx:connecting_to_main_vertex"

# full-transition writer histories (WCT_PID_WRITE_DEBUG=2, new this round)
WCT_PID_WRITE_DEBUG=2 PR_JOBS=1 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 /home/xqian/tmp/pr40r6/trace2-55715-off data 55715
WCT_PID_WRITE_DEBUG=2 SBND_SHOWER_TRAJ_STRAIGHT_GUARD=1 SBND_SHOWER_ABSORB_TRACK_GUARD=1 \
  SBND_SHOWER_CONNECT_PROTECTED_PION_GUARD=1 PR_JOBS=1 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 /home/xqian/tmp/pr40r6/trace2-55715-on data 55715

# the three owner cases, all six pr40 r5+r6 knobs forced on (G2 smoke arm)
SBND_TRACK_PID_PERSIST_DQDX_ELECTRON_GUARD=1 SBND_SHOWER_CONNECT_MAIN_VERTEX_STRAIGHT_GUARD=1 \
  SBND_SHOWER_TRAJ_STRAIGHT_GUARD=1 SBND_SHOWER_ABSORB_TRACK_GUARD=1 \
  SBND_SHOWER_CONNECT_PROTECTED_PION_GUARD=1 SBND_MICHEL_STEM_MUON_RESCUE=1 \
  PR_JOBS=1 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-ncpi0-cb0805 work-pr40r6-cases data 84229
# (same env) ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr40r6-cases data 54341 55715

# post-flip bare-config verification arm (NO env overrides = the production
# defaults after this round's flip; also includes nueCC48 evt 10550 for the
# hash-match against the gated on-arm)
PR_JOBS=2 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-ncpi0-cb0805 work-pr40r6-flipverify data 84229
PR_JOBS=2 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr40r6-flipverify data 54341 55715
PR_JOBS=1 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r6-flipverify data 10550

# G1 (knobs off byte-identical) -- see the G1 note below for why pr40r5-off48
# is a valid clean-HEAD reference this round
PR_JOBS=6 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r6-off48 data
# hash_archive.py member hashes, off48 vs work-pr40r5-off48, mabc-pr.zip + pctree

# G3 (population, flip set forced on across the 48-event manifest)
SBND_TRACK_PID_PERSIST_DQDX_ELECTRON_GUARD=1 SBND_SHOWER_CONNECT_MAIN_VERTEX_STRAIGHT_GUARD=1 \
  SBND_SHOWER_TRAJ_STRAIGHT_GUARD=1 SBND_SHOWER_ABSORB_TRACK_GUARD=1 SBND_MICHEL_STEM_MUON_RESCUE=1 \
  PR_JOBS=6 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r6-on48 data

./build/clus/wcdoctest-clus
```

## Phase 1 -- the round-5 mechanism map was two-thirds wrong, traces first

Round 5 hypothesized the 55715 regression came from `shower_clustering_with_nv_in_
main_cluster`'s `is_shower_seg` re-seed and round 6's plan aimed a guard at
`examine_showers`' re-root.  Both wrong -- the Phase-1 trace (existing round-1 probe)
showed the writer is `shower_clustering_connecting_to_main_vertex`'s accept-time
force-set (probe tag `connecting_to_main_vertex`):

- **55715**: with F11 on, the straightness demotion (`NeutrinoVertexFinder.cxx:1445`)
  correctly makes 15007 `mu-`; `connecting_to_main_vertex` then selects the unshielded
  6.1 cm parent 15005 as its EM candidate (UNDER `segment_is_straight_long_track`'s
  10 cm floor, so F10's straight branch cannot save it) and force-sets it to 11.
- **54341**: 18005's 2212 is a direct `0->2212` (invisible to the +-11-filtered probe;
  neighbors 18004/18006 show the examine_direction recomputes at `NeutrinoVertexFinder
  .cxx:1807`, timing the write inside examine_direction, before the F8/F14 call site).
- **84229**: `pf_shower_vertex_barrier=true` in SBND production; a segment excluded
  from every shower view either joins the track BFS or lands in the doc pr/38 orphan
  safety net (`MultiAlgBlobClustering.cxx`) -- guaranteed its own PF node either way
  (19038 dirsign=-1, so the orphan `dirsign==0` skip does not bite).

## Fix

- **F12 `shower_absorb_track_guard`** -- per-segment exclusion inside
  `Shower::complete_structure_with_start_segment` (`PRShower.cxx`), threaded to all 7
  call sites: skip (and terminate the walk at) a confidently PID'd non-electron
  (`pdg != 0 && |pdg| != 11`) that is `segment_is_straight_long_track`; do NOT claim it
  in `used_segments`; exempt long-muon pseudo-showers (`get_particle_type()==13`).
- **F13 `shower_connect_protected_pion_guard`** -- fifth skip branch in
  `connecting_to_main_vertex` (`pdg==211 && segment_has_proton_daughter`).  **Measured
  DEAD, never to be flipped** -- see below.
- **F14 `michel_stem_muon_rescue`** -- new pass `override_michel_stem_muon`
  (`NeutrinoPatternBase.cxx`, F8's call site): a `pdg==2212`, straight-long,
  main-vertex-emanating stem with >=1 shower-like sibling at its stopping vertex
  relabels `mu-` -- the toolkit's own Michel rescue rule (`NeutrinoVertexFinder.cxx`
  "a stopped proton cannot produce a Michel electron") minus its two reach limits
  (weak-direction only; stopping vertex degree exactly 2 -- 18005 has confident
  direction and degree 4).

Debug infra: `WCT_PID_WRITE_DEBUG=2` now logs EVERY pdg transition (the round-1 probe
only logged +-11 ones, blind to 211->2212).  Physics-inert, kept.

## Demonstration (work-pr40r6-cases, all six knobs on; the post-flip bare-config
arm work-pr40r6-flipverify is identical on all three -- F13 confirmed inert)

| case | owner ask | measured mc.json | verdict |
|---|---|---|---|
| 84229 | seg 19038 -> mu-, Michel child | `19039 'pi+ 38'` -> { `19038 'mu- 77'`, `19040 'e- 5'` } | **G2a PASS** (owner accepted 19039 pi+) |
| 54341 | stem mu-, Michel child(ren) | `18005 'mu- 74'` -> { `18006 'e- 19'`, `18007 'mu- 11'` } | **G2b PASS** (exact) |
| 55715 | 15007 mu-, 15005 untouched, 15006 proton kept | `15005 'proton 84'` -> { `15006 'proton 147'`, `15007 'mu- 60'`, `15035 'mu- 3'` } | **G2c PASS** (owner accepted 15005 proton) |

**The two owner-accepted residuals, fully attributed** (WCT_PID_WRITE_DEBUG=2
histories):

- 19039 (84229): OFF `0->11` (F1's poisoned undirected electron persist,
  `PRSegmentFunctions.cxx:2019`) then `11->11` reassert (`PRShower.cxx`).  ON (F9
  removes the poison): `0->13` -- its own honest muon call -- then `13->211` at the
  single-muon selection (`NeutrinoVertexFinder.cxx:1679`; 19038 wins the muon slot).
  The 4.9 cm stub and 19038 are physically the same stopping muon split at a degree-3
  vertex; a collinear-muon-fragments rule would be a new round.
- 15005 (55715): OFF `0->2212` (own charge, `NeutrinoVertexFinder.cxx:1476`) ->
  `2212->13` (Michel rescue at :1807, TRIGGERED BY 15007's WRONG e- LABEL) ->
  `13->211` (single-muon demotion, :1679).  ON: `0->2212` and nothing else -- the
  baseline pi+ was derivative of the bug being fixed; proton is the segment's own
  call.  Matches the owner's original reading ("mislabeled pion due to an attached
  proton").

**F13 negative result**: 15005 is already 2212 at candidate-selection time, so a
`pdg==211` guard cannot fire (confirmed: F11+F13 arm still shows the merged
`e- 105 MeV` node; F11+F13+F12 fixes it).  The legacy `pdg==2212` skip band
(`>1.45x MIP, nd<=3`) does not catch it either (its ratio sits in the 1.3-1.45
window); F12's exclusion makes the candidate fail EM acceptance instead, which is
sufficient.  Kept in-tree as a documented dead knob (doc pr/36 F2 precedent),
default false, excluded from the flip.

## Gates

- **G0 freshness**: `wcbuild` x2 (round-6 code; then the WCT_PID_WRITE_DEBUG=2
  widening); `local/lib/libWireCellClus.so` == `build/clus/` both times.  The M1
  link-order trap did NOT recur this round.
- **G1 knobs-off byte-identical**: **PASS, 48/48 events, 96/96 archives, 0
  mismatches** (`work-pr40r6-off48` vs `work-pr40r5-off48`,
  `abtest/hash_archive.py` member hashes, `mabc-pr.zip` + `pctree-pr-evt*.tar.gz`).
  The r5 off arm is a valid clean-HEAD reference this round because round 5
  flipped nothing: it was itself gated 48/48 vs the git-stash clean base at
  `3fa0aeb3`, and HEAD `aab6ccce` is byte-wise that same source with the same
  false defaults -- byte-identity is transitive.  (The round-3 lesson "a prior
  arm stops being a valid reference once defaults change" does not bite here
  precisely because no default changed in round 5.)
- **G2a/b/c**: **PASS, all three** -- table above.
- **G3 population**: **PASS -- nusel-table.tsv diff is 0 lines** across the 48-event
  manifest (`work-pr40r6-off48` vs `work-pr40r6-on48`, flip set F9/F10/F11/F12/F14
  forced on).  Zero verdict flips, zero score movement at table precision.
- **G4 census** (off48 vs on48 calib JSON, 48 events): 26/48 events restructure,
  fully decomposed -- 95 shower-membership-only reassignments + 126 renumbered-pair
  + 86 length/combination knock-ons (F12's absorption changes), 19 flag-only clears
  (F11's straightness population, 11-30 cm straight segments plus 2 short-segment
  knock-ons on evt 256587), **42 pdg transitions dominated by the intended
  11->13 direction (19 recoveries)**; counter-direction 13->11 is 7, of which 6 are
  sub-cm score-100 never-PID'd stubs released to their own default-fill shower
  (round 4's documented "secondary release" class) and 1 (evt 46363 seg 86123,
  8.4 cm) is a genuine re-PID with confident score 0.30.  Long-muon pseudo-shower
  count unchanged (2 off == 2 on) -- broken-muon reassembly intact, the F12
  exemption works.
- **G5 unit tests**: `wcdoctest-clus` **PASS 100/100 cases, 1042/1042 assertions**.
- **G6 compiled-config**: **PASS** -- keys absent off (compiled JSON byte-identical
  to HEAD; note the TaggerCheckNeutrino node only exists with the runner's
  `pipeline_names` TLA -- grep the pipeline-enabled compile, not the bare one);
  present exactly once each with the TLAs on.

## Flip -- FLIPPED, all five knobs SBND PRODUCTION DEFAULT ON

Owner pre-authorized during round-6 planning ("flip all if gates pass") and
separately accepted the two parent-stub residuals (19039 pi+, 15005 proton) when
asked.  G1 96/96 + G2 3/3 + G3 0-line diff met the pre-authorization bar ->
`wct-pr-perevt.jsonnet` TLA defaults flipped: `track_pid_persist_dqdx_electron_
guard`, `shower_connect_main_vertex_straight_guard`, `shower_traj_straight_guard`
(the round-5 trio, whose flip round 5 explicitly deferred), `shower_absorb_track_
guard`, `michel_stem_muon_rescue` all `false -> true`.
`shower_connect_protected_pion_guard` (F13) stays `false` -- measured dead, never
flip.

Cfg-only flip verified with bare runs (no env overrides, `work-pr40r6-flipverify`):
nueCC48 evt 10550 hash-matches the gated on-arm exactly (`mabc-pr.zip`
`139918f0...`, `pctree` `3bf518d2...`), and all three owner-case `mc.json` trees
are byte-wise the Demonstration table's.

## Scope and not-claimed

- The single-muon-per-cluster selection (`NeutrinoVertexFinder.cxx:1679` pion
  demotion) is untouched; the two owner-accepted parent-stub labels (19039 pi+,
  15005 proton) are its output on newly-honest inputs.  A collinear-muon-fragments
  rule (19039+19038 are physically one muon) is a possible future round, not
  attempted.
- `shower_clustering_connecting_to_main_vertex`'s accept-time force-set and its
  1.3-1.45x MIP proton-skip blind spot are documented (F13's negative result), not
  changed.
- The template competition's lack of an absolute quality gate (54341's original
  proton call) is compensated by F14's topology rescue, not fixed at the source.

---

# Round 7 -- census + owner Bee scan, mislabeling confirmed in both cases, TWO
proposed fixes written up but NOT implemented this round (investigate-and-
document only, per owner scope)

## Repro block

```bash
cd sbnd_xin

# Phase 0 item zero -- the pr/90 self-check on 320865 (does the pr/90
# teb_turn_min_arm_frac break, shipped hours earlier the same day, PRODUCE
# the electron?)
SBND_TEB_TURN_MIN_ARM_FRAC=0 WCT_SHOWER_TOPO_DEBUG=1 WCT_PID_WRITE_DEBUG=2 \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr40r7-teboff data 320865
SBND_TWO_END_BREAK=0 WCT_SHOWER_TOPO_DEBUG=1 WCT_PID_WRITE_DEBUG=2 \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr40r7-nobreak data 320865

# Phase 0 -- writer-site traces (WCT_PID_WRITE_DEBUG=2, WCT_SHOWER_TOPO_DEBUG=1)
./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr40r7-trace data 320865
./run_pr_chain_batch.sh work-cbr3-census-on work-pr40r7-trace2k data 54629

# Phase 1 -- census reruns, full manifest (no event-id list = every ql_evt<ID>
# under the QL root; NOTE: for mcp1k this is the full 1000, not the
# nu_evaluated=1 subset some retired arms used -- see the G0 note below on
# why those retired arms could not be reused as-is)
PR_JOBS=16 ./run_pr_chain_batch.sh work-mcp1k-cb0805   work-pr40r7cen-mcp1k   data
PR_JOBS=8  ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r7cen-nuecc48 data
PR_JOBS=6  ./run_pr_chain_batch.sh work-ncpi0-cb0805   work-pr40r7cen-ncpi0   data

python3 scripts/analysis/pr40/pr40r7_census.py \
  work-pr40r7cen-mcp1k work-pr40r7cen-nuecc48 work-pr40r7cen-ncpi0 \
  --min-len 20 --out /home/xqian/tmp/pr40r7/census_full.tsv

# Phase 2 -- Bee scan set, top-50 rows / 45 unique events
python3 scripts/bee/make_pr_bee.py \
  -q work-mcp1k-cb0805 -q work-nuecc48-cb0805 -q work-ncpi0-cb0805 \
  -p work-pr40r7cen-mcp1k -p work-pr40r7cen-nuecc48 -p work-pr40r7cen-ncpi0 \
  -o bee/pr40r7/pr40r7-scan.zip <45 event ids, ranked>
./upload-to-bee.sh bee/pr40r7/pr40r7-scan.zip
```

## Symptom

Owner (2026-08-17): SBND events **54629** (mcp2k) and **320865** (mcp1k) each
show a **long, track-like object reconstructed and displayed as an electron**,
evidenced by the object's own track signature (dQ/dx).  Owner asked to
investigate, find more cases across the sample, generate Bee links for a hand
scan, add the round onto this doc, and propose solutions -- explicitly
**investigate-and-document only this round**: no C++/jsonnet change, no gate,
no flip.  (Superseding an earlier framing in the planning transcript that had
assumed standing "flip ON if gates pass" authorization from the pr/90 round
carried over automatically -- it does not; each round is scoped separately.)

The `nu-candidate` row of each event's per-bundle nusel TSV (one row per flash
bundle; do not read the last row) confirms both are genuinely long selected
neutrino main clusters:

| evt | sample | main cluster | npts | length |
|---|---|---|---|---|
| 54629 | mcp2k | 15 | 2495 | 156.0 cm |
| 320865 | mcp1k | 13 | 3109 | 207.1 cm |

## Phase 0 item zero -- 320865 is a same-day pr/90 side-effect, not an
independent pr/40 bug

320865's mislabeled segment (13001) exists only because `teb_turn_min_arm_frac`
(pr/90 round 2, shipped hours earlier the same day) splits the cluster's
198.6 cm segment.  Three configurations of the same event, reading per-segment
pdg/flag_shower/length/dQ/dx straight from `tracking-pr.root:T_rec_charge`
(validated exact against `pr_display`'s calib JSON, 71/71 segments, 0
mismatch -- no `PR_EXTRA_STAGES=pr_display` rerun needed anywhere this round):

| configuration | main-cluster long segments (>5cm) | electron present? |
|---|---|---|
| **current production** (`teb_turn_min_arm_frac=0.4`) | 13001: 48.05cm, `pdg=11`, 1.76xMIP; 13002: 153.27cm, `pdg=13` (muon) | **YES** (13001) |
| `SBND_TEB_TURN_MIN_ARM_FRAC=0` (pre-round-2 legacy break location) | 13001: 7.35cm, `pdg=2212` (proton); 13002: 193.55cm, `pdg=13` (muon) | no -- proton stub + correctly-labelled muon |
| `SBND_TWO_END_BREAK=0` (break stage fully disabled) | 13000: 200.48cm, `pdg=13` (muon), one segment | no -- one clean muon |

**Fully unbroken, the whole 200 cm object is one correctly-labelled muon.**
The electron only appears once the break lands at the *correct* kink location
(pr/90's own round-2 fix) and produces a 48.05 cm arm that sits narrowly
outside two existing pr/40 guards (below).  This is reported plainly rather
than folded silently into "a pr/40 bug": **the pr/90 round-2 fix, itself
correct and already SBND production, has a same-day side effect on 320865's
PID.**  54629 (a different sample, unrelated to any pr/90 knob) stands as the
round's independent case.

## Phase 0 -- writer-site attribution (`WCT_PID_WRITE_DEBUG=2` +
`WCT_SHOWER_TOPO_DEBUG=1`)

**320865 / seg 13001 -- exactly the hypothesized site, confirmed by trace.**
Three sites in this codebase set `kShowerTopology`/`kShowerTrajectory`; two are
already guarded (F3 `shower_topo_dqdx_guard`, `PRSegmentFunctions.cxx:4303`;
F11 `shower_traj_straight_guard`, `:2103`).  The trace shows the third:

```
TOPO_REEXAM id=-1 clus=13 gidx=1 enter pdg=211 score=0.164
PID_WRITE_DEBUG setter id=-1 clus=13 gidx=1 pdg 211 -> 13  at PRSegmentFunctions.cxx:2927
TOPO_REEXAM id=-1 clus=13 gidx=1 after-pid pdg=13 score=0.160
PID_WRITE_DEBUG set_pdg id=-1 clus=13 gidx=1 pdg -> 11  at NeutrinoVertexFinder.cxx:topo-escape(M5)
```

`NeutrinoVertexFinder.cxx:3288-3327` (`improve_vertex`'s topology re-exam)
re-derives PID, gets `pdg=13` (muon) at `score=0.160`, then its own escape
branch only *declines* the electron override when `pdg==13 && score<0.06` --
0.160 fails that bar, so it force-sets `pdg=11` anyway.  **No straightness or
dQ/dx test at all on this path.**  13001 misses F3's `demote_len=50cm` by
1.95 cm and `1.75xMIP` by 0.6% -- a near-miss, but this third site was simply
never gated by either existing knob.

**54629 -- hypothesis (an unconditional `segment_determine_dir_track` write)
was WRONG; trace found three DIFFERENT, previously unnamed sites**, one per
segment:

```
PID_WRITE_DEBUG setter id=-1 clus=15 gidx=7  pdg 0  -> 11  at NeutrinoVertexFinder.cxx:1659   (seg 15007, 31.0cm)
PID_WRITE_DEBUG setter id=-1 clus=15 gidx=11 pdg 13 -> 11  at NeutrinoVertexFinder.cxx:1714   (seg 15011, 94.6cm)
PID_WRITE_DEBUG setter id=-1 clus=20 gidx=13 pdg 13 -> 11  at NeutrinoShowerClustering.cxx:1401 (seg 20013, 113.0cm)
```

- **`NeutrinoVertexFinder.cxx:1659`** (inside `examine_direction`): the branch
  `if (flag_shower_in && current_sg->dirsign()==0 && !is_shower)` sets `pdg=11`
  unconditionally.  This is a **coverage gap in the EXISTING pr/74 P1 guard**
  (`shower_in_cascade_guard`, already SBND ON): that guard's
  `segment_shower_in_cascade_vetoed` call is wired **three lines down**, into
  the sibling `else if (flag_shower_in)` branch that tests
  `abs(cur_pdg)==13 || cur_pdg==0` -- it was never wired into this
  `dirsign()==0` branch, which is reached first and returns before the guarded
  branch is ever considered.
- **`NeutrinoVertexFinder.cxx:1714`** (same function, `examine_direction`'s
  "many/long daughter showers" wholesale-reclass block): when a segment sits
  at a vertex with `num_daughter_showers>=4` (or long daughter showers) and an
  **angle mismatch** with a neighbor exceeds 135-155 deg (three sub-conditions),
  it force-sets `pdg=11` -- overwriting an already-correct `pdg=13`.  No charge
  or straightness test.
- **`NeutrinoShowerClustering.cxx:1401`** (inside
  `shower_clustering_connecting_to_main_vertex`, setting a new shower's
  `start_seg` direction/type): `if (pdg==0 || abs(pdg)==13)` unconditionally
  writes `pdg=11`.  No charge or straightness test.  (Distinct from the
  `total_length<70cm`-gated accept-time force-set named in earlier pr/40
  rounds -- this fires earlier, on `start_seg` specifically.)

All three of 54629's long pdg-11 segments have **decisively nonzero,
MIP-scale median dQ/dx** (1.15x, 1.19x, 1.42x MIP) and
**`segment_is_straight_long_track` returns TRUE on every one** -- the single
geometry-only lever fires universally across both events regardless of the
charge value:

| evt | seg | L (cm) | D/L | med dQ/dx xMIP(43000) | flag_shower | writer site |
|---|---|---|---|---|---|---|
| 54629 | 15007 | 31.00 | 0.974 | 1.42 | 0 | `NeutrinoVertexFinder.cxx:1659` |
| 54629 | 15011 | 94.59 | 0.980 | 1.15 | 0 | `NeutrinoVertexFinder.cxx:1714` |
| 54629 | 20013 | 113.02 | 0.956 | 1.19 | 0 | `NeutrinoShowerClustering.cxx:1401` |
| 320865 | 13001 | 48.05 | 0.942 | 1.76 | 1 | `NeutrinoVertexFinder.cxx:3320` (topo-escape M5) |

**Note, not chased this round**: 54629's neutrino candidate is `fc=0` /
`stmfit=eval` (not fully contained) -- the PID mislabel is real regardless,
but the owner may separately want to ask whether this candidate is even a
contained neutrino.

## MIP-scale correction (affects reading any pr/40 threshold table)

`pr40_seg_pid.py` and this doc's own round-1 table use `MIP=56000`.  Every
C++ knob actually binds to **`m_mip_dqdx_median = 43000`** e/cm (e.g. F3's
`segment_dqdx_spares_electron_reclass` call passes the function's own
`mip_dqdx_median` parameter, and the calib dump's `meta.mip_dqdx_median` is
43000).  This round uses 43000 throughout.  `pr40_seg_pid.py`'s 56000 is a
latent scale bug -- flagging here rather than fixing the script, since round 3
already spent time on exactly this class of mismatch (`m_mip_dqdx` 50000 vs
`m_mip_dqdx_median` 43000) and a wrong MIP silently shifts every 1.2x/1.75x
verdict by 30%.

## Census -- new tool, no rerun of the pr_display stage needed

`tracking-pr.root:T_rec_charge` (written unconditionally by every PR arm)
carries `real_cluster_id`, `particle_id`, `flag_shower`, `q`, `nq`, `x/y/z` --
validated exact against the `pr_display` calib JSON (71/71 segments, 0
mismatch, `work-r2mc-prod0813`).  `dQ/dx = (q+1000)*10/nq` e/cm; length = sum
of consecutive fit-point distances.  New script
`scripts/analysis/pr40/pr40r7_census.py` reads this directly across an arbitrary
number of arms -- no `PR_EXTRA_STAGES=pr_display` rerun required at all.  It
differs from the existing `pr40_seg_pid.py` in three ways: reads
`T_rec_charge` instead of calib JSON; reports `flag_shower` as a **column**,
not a `!flag_shower` cut (`pr40_seg_pid.py`'s cut would have EXCLUDED 320865's
own case -- it is Family B, `flag_shower=1`); uses the correct 43000 MIP scale.

**Fresh PR reruns, current production, no env overrides**, over the full
authorized manifest (ended up covering more than planned -- see the G0 note
below on why the previously-existing `work-vf*-cbr3on` arms could not be
reused):

| sample | events run | rc=0 | with a nu-candidate |
|---|---|---|---|
| mcp1k | 1000 (the full QL sample, not just a 445-event subset) | 1000/1000 | 522 |
| nueCC48 | 48 | 48/48 | 48 |
| NCpi0-19 | 19 | 19/19 | 19 |

**mcp2k (54629's own sample) is explicitly NOT covered** -- owner's scoping
decision; state this gap plainly rather than imply full coverage.

Selection: `particle_id==11 AND is_main_cluster AND length>20cm`.  **Result:
212 candidate segments** across the manifest (78 mcp1k, 105 nueCC48, 29
ncpi0) -- muon-like (<1.2xMIP): 90; ambiguous (1.2-1.75xMIP, the
deliberately-uncut dead band per the evt-256587 precedent): 78; proton-like
(>=1.75xMIP): 43; no dQ/dx evidence: 1.  Ranked by
`length x max(0, 1-|xMIP-1|) x (1.0 if geometrically straight-long else 0.3)`
so a long, ~1xMIP, straight segment sorts first.  **320865/13001 IS in this
census** (rank 49/212, xMIP=1.76 near the proton-like edge) -- direct
cross-validation that the predicate catches the round's own motivating case.
Interestingly, **evt 138009** (nueCC48) also appears at rank 16/212
(seg 12094, 41.6cm, flag_shower=1, xMIP=0.84) -- this is one of the events the
owner separately flagged as "multiple tracks" in a prior review, an
independent hint the two symptom reports may share a mechanism.

Full ranked TSV: `scripts/analysis/pr40/pr40r7_census.py`'s output at
`/home/xqian/tmp/pr40r7/census_full.tsv` (not committed -- scratch; regenerate
with the Repro block's command against `work-pr40r7cen-{mcp1k,nuecc48,ncpi0}`,
named above and not yet registered in `docs/work-tags.md` -- that file is a
dated retirement/campaign log, not a running index, and this round doesn't
own a retirement round to add a proper entry in; the arm names and their
provenance live in this Repro block instead).

**G0 note -- why the retired-and-rebuilt `work-vf*-cbr3on` arms could not be
reused as census input**: `work-vfmcp1k-cbr3on`, `work-vfnuecc48-cbr3on`, and
`work-vfncpi0-cbr3on` all predate the 2026-08-17 14:50 cathode-rescue-round-3
flip (`2d8c9e5a`) -- built 13:25-13:38.  A single-event spot check
(nueCC48 evt 10550) showed a hash MISMATCH against a fresh current-production
rerun (mcp1k evt 320865 happened to match by coincidence -- that event's
topology never touches the cathode-rescue path, which is why an earlier,
narrower check missed the staleness).  Fresh reruns were built instead (the
table above); this is the reason the mcp1k pass covers the full 1000 rather
than reusing the retired 445-event subset.

A concurrent session separately flagged (then, after direct verification,
retracted) a suspicion that the *compiled binary* (not just the arm) was
stale relative to `812c7add`'s three cathode-rescue C++ fixes.  Checked
directly this round: `strings build/clus/libWireCellClus.so` contains the
round-3 knob keys and their full runtime log-format strings (not just a
static literal), and the source files' actual filesystem mtimes (12:39:08)
predate the `.so`'s build mtime (12:39:42) by 34 seconds -- the .so is
current.  The false alarm traced to comparing against the commit's timestamp
(12:41:28) rather than the source files' actual mtime -- the same class of
trap CLAUDE.md's M1 correction warns about, in the opposite direction.  Noted
here since it is a recurring failure mode worth a general callout, not because
it changed anything about this round's arms.

## Bee scan set for the owner

Top 50 ranked segments -> **45 unique events** (several events carry more than
one flagged segment), most-flagrant-first, **5 rows below the cut dropped**
(the 212-candidate full list has 167 more beyond what's linked here -- available
in the census TSV named above if the owner wants a deeper pass after this one).

**Link**: `https://www.phy.bnl.gov/twister/bee/set/5c5018d6-6db6-45b9-81f7-0338eda9741d/event/list/`

Bee-index order (event -> flagged segment(s), length, xMIP; `*` = one of the
round's two motivating events):

| idx | evt | seg(s) | L (cm) | xMIP |
|---|---|---|---|---|
| 0 | 350935 | 11001 | 251.4 | 1.09 |
| 1 | 283713 | 17006 | 252.8 | 1.17 |
| 2 | 55595 | 8005 | 193.8 | 1.28 |
| 3 | 407280 | 16010 | 128.8 | 1.14 |
| 4 | 281837 | 13002 | 124.3 | 1.16 |
| 5 | 55539 | 23005 | 108.9 | 1.13 |
| 6 | 314507 | 51007, 17002 | 61.7, 32.3 | 1.04, 1.57 |
| 7 | 64921 | 11002 | 84.9 | 1.35 |
| 8 | 71222 | 22007 | 73.2 | 1.28 |
| 9 | 316025 | 16009 | 81.7 | 1.36 |
| 10 | 395610 | 28002 | 53.3 | 1.05 |
| 11 | 285567 | 8035 | 47.5 | 1.15 |
| 12 | 280972 | 7159 | 46.0 | 1.18 |
| 13 | 401450 | 24076, 24074 | 38.7, 27.9 | 1.06, 1.07 |
| 14 | 290729 | 12007 | 50.7 | 1.30 |
| 15 | 138009 | 12094, 12095 | 41.6, 43.1 | 0.84, 1.75 |
| 16 | 395060 | 24012 | 39.7 | 1.12 |
| 17 | 286191 | 63011 | 35.3 | 1.04 |
| 18 | 348471 | 12007 | 53.5 | 1.40 |
| 19 | 69314 | 3015 | 38.4 | 1.19 |
| 20 | 30504 | 11080, 11020 | 42.9, 41.6 | 1.29, 1.68 |
| 21 | 286681 | 72038 | 36.7 | 1.19 |
| 22 | 90055 | 13048 | 29.8 | 1.18 |
| 23 | 293149 | 4001 | 26.4 | 1.10 |
| 24 | 56982 | 22111 | 24.2 | 0.93 |
| 25 | 321371 | 18004 | 25.8 | 1.15 |
| 26 | 349461 | 71014 | 40.4 | 1.47 |
| 27 | 352233 | 51012 | 38.6 | 1.45 |
| 28 | 234638 | 10030 | 28.9 | 1.28 |
| 29 | 214469 | 16057 | 27.4 | 1.27 |
| 30 | 389538 | 19041 | 39.5 | 1.53 |
| 31 | 277298 | 17003 | 45.0 | 1.60 |
| 32 | 349549 | 12012 | 35.2 | 1.50 |
| 33 | 433451 | 4031 | 27.3 | 1.36 |
| 34 | 278684 | 10003 | 20.5 | 1.19 |
| 35 | 292643 | 18009 | 22.9 | 1.29 |
| 36 | 315167 | 8006 | 42.4 | 1.63 |
| 37 | 268067 | 15084 | 21.9 | 1.29 |
| 38 | 239794 | 2080 | 21.5 | 1.32 |
| 39 | 386948 | 16005 | 23.2 | 1.41 |
| 40 | 64409 | 8113 | 24.8 | 1.47 |
| 41 | 437699 | 11024 | 24.9 | 1.47 |
| 42 | 348691 | 51079 | 20.3 | 1.36 |
| 43 | 54095 | 17044 | 20.7 | 1.38 |
| 44 | **320865*** | 13001 | 48.1 | 1.76 |

(54629 itself is not in this Bee set -- see the mcp2k scope note above; its
Bee links are recorded separately in the pr90r4-{before,after} sets in
`docs/pr/90_unbroken-kink-mcp1k.md` sec 10.10, and its three writer sites are
fully attributed above with no Bee scan needed to confirm them.)

Owner verdicts should be recorded per event/segment under a **fresh** tag
(`overclustering_labels/pr40r7-scan/`, auto-created on first save by
`overclustering_display/serve_overclustering_scan.sh`) -- never into an
existing label dir (M13).

## Proposed fixes -- NAMED, NOT IMPLEMENTED this round

Per owner scope, these are documented candidates for a follow-up round, after
the Bee scan confirms which census hits are real instances of the mechanism.

- **Candidate 1 (320865 class)** -- `shower_topo_reexam_straight_guard`
  (proposed name): a third skip branch in
  `NeutrinoVertexFinder.cxx:3288-3327`'s topology re-exam, same shape as F10
  (`shower_connect_main_vertex_straight_guard`) / F11
  (`shower_traj_straight_guard`) -- decline the flag-set/`pdg=11` write when
  `segment_is_straight_long_track(sg)` is true.  `false` = legacy =
  byte-identical when off.  Given the Phase 0 item-zero finding, this should
  be framed as a **defensible safety net**, not "the fix for 320865" -- the
  segment only exists because of a same-day pr/90 side effect, and the
  cleanest long-term fix may instead be on the pr/90 side (e.g. widening the
  break's own quality check so a sub-`shower_topo_demote_len` arm is treated
  more conservatively).  Flag both options to the owner rather than picking.
- **Candidate 2 (54629 class)** -- THREE separate proposed guards, since
  Phase 0 found three distinct, previously-unnamed writer sites rather than
  the single hypothesized one:
  - `examine_direction_dirsign_shower_in_guard` (proposed name):
    extend pr/74's existing `shower_in_cascade_guard` predicate
    (`segment_shower_in_cascade_vetoed`) to also cover the
    `dirsign()==0` branch at `NeutrinoVertexFinder.cxx:1659` -- today the
    guard is wired only into the sibling `abs(pdg)==13||pdg==0` branch three
    lines down.  This is the smallest, most surgical of the three: reusing an
    ALREADY-SHIPPED, ALREADY-ON knob's own predicate at a second call site,
    not a new mechanism.
  - `daughter_shower_angle_reclass_straight_guard` (proposed name): guard the
    `num_daughter_showers>=4` / angle-mismatch wholesale reclass at
    `NeutrinoVertexFinder.cxx:1714` with `segment_is_straight_long_track`,
    same shape as F10/F11.
  - `shower_connect_start_seg_straight_guard` (proposed name): guard the
    `start_seg` direction/type write at
    `NeutrinoShowerClustering.cxx:1401` (`pdg==0 || abs(pdg)==13` ->
    unconditional electron) the same way.
  All three reuse the same, already-proven-universal geometry lever
  (`segment_is_straight_long_track`, fires on every long pdg-11 segment found
  this round, both events) rather than a new charge threshold -- consistent
  with round 3's own lesson about not inventing a fourth MIP scale.

None of these five names are implemented, gated, or wired into config this
round.  A follow-up round should re-run the same trace method on a
representative subset of the owner's confirmed Bee-scan verdicts before
committing to any one shape, exactly as this round's own Phase 0 found the
original single-mechanism hypothesis for 54629 wrong.

## Stale statements in this doc, annotated (found this session, not previously
flagged)

- The H1 (line 1) says "three fixes" -- the doc now documents 14+ knobs
  across seven rounds.
- Round 2's `## Flip` (line ~631) says F5 `shower_proton_daughter_pion` was
  "left OFF" -- superseded by round 3 (line ~838), which flips it ON; current
  config confirms `true`.
- Round 5's `## Flip -- NOT flipped` (line ~1253) says all three round-5 TLA
  defaults "stay false" -- superseded by round 6 (line ~1450), which flips
  all three ON; current config confirms `true`.

## Scope and not-claimed

- No C++, jsonnet, or config change shipped this round -- owner explicitly
  scoped this round to investigation + documentation + a Bee scan for
  confirmation first.
- The five proposed-fix names above are NOT gated, NOT tested, NOT wired into
  any component -- they are documented candidates only.
- 212 census candidates found; only the top 50 (45 unique events) went into
  the Bee set -- the remaining 167 are in the (uncommitted, scratch) full TSV
  and available for a deeper pass if the first 45 don't exhaust the owner's
  scanning budget.
- mcp2k is not covered by the census (54629's own sample) -- a future round
  extending the census there is a candidate follow-up, not attempted here.
- 54629's containment status (`fc=0`, `stmfit=eval`) is noted but not
  investigated -- a separate question from the PID mislabel.

---

# Round 8 (2026-08-18) -- owner verdicts on the round-7 predictions; cross-cluster
writer traced (static, not yet runtime-confirmed); gap-jump + PF-particle fix
design, NOT IMPLEMENTED

## Repro block

```bash
cd sbnd_xin

# All analysis this round reads ALREADY-EXISTING arm output -- zero PR chain
# reruns (see "Why no reruns this round" below).  Arms used:
#   work-pr91r2-prod-mc, work-pr91r2-prod-ncpi0   (toolkit cca9f167)
#   work-pr40r7cen-{mcp1k,nuecc48,ncpi0}          (round 7, 2026-08-17)
# Per-event calib JSON: work-pr91r2-prod-{mc,ncpi0}/pr_evt<ID>/calib-pr-evt<ID>.json
# Round-7 census (uncommitted scratch): /home/xqian/tmp/pr40r7/census_full.tsv

# gap metrics + segment dQ/dx/straightness for 286906/409546/521075 --
# python3 reading calib-pr-evt<ID>.json's steiner/track_shower/segments/
# vertices blocks directly (no new script committed this round; the
# one-off snippets are quoted inline below so the owner can re-run them).

# 45-Bee-event straight_long cross-check against round 7's own TSV column:
python3 - <<'PY'
import csv
bee45 = [350935,283713,55595,407280,281837,55539,314507,64921,71222,316025,
         395610,285567,280972,401450,290729,138009,395060,286191,348471,
         69314,30504,286681,90055,293149,56982,321371,349461,352233,234638,
         214469,389538,277298,349549,433451,278684,292643,315167,268067,
         239794,386948,64409,437699,348691,54095,320865]
rows = list(csv.DictReader(open('/home/xqian/tmp/pr40r7/census_full.tsv'), delimiter='\t'))
byevt = {}
for r in rows: byevt.setdefault(int(r['evt']), []).append(r)
n_straight = sum(1 for e in bee45 if byevt.get(e) and
                  max(byevt[e], key=lambda r: float(r['length_cm']))['straight_long'] == 'True')
print(f"{n_straight}/{len(bee45)} Bee events have straight_long=True on their dominant segment")
PY
```

## Symptom (owner scan, 2026-08-18)

Owner scanned three events from doc pr/91's Bee set (idx 9/10/21 of
`pr91r2.index.txt`, the "rung-2-class predictions, still unscanned" flagged
by doc pr/84 round 2 §11's cross-cluster F1-rung-2 census) and gave verdicts:

> 286906, a gap between the long muon and the vertex, this should be one
> single muon. Note, if we have a long track pointing at the vertex, they
> could be 1. EM shower, then gamma 2. hadron track, then neutron hadron
> track 3. muon, then likely real gap in SP etc. If the long muon is not
> pointing or relevant to the nu vertex, this long muon should not be
> counted in the PF or energy reconstruction, since it is very likely to be
> a different event
> 409546, seems to be OK, since the track is short, so it could be an
> electron.
> 521075, this is good, the EM shower is clearly a gamma, the PF is good.

Owner directive for this round: update the doc's understanding and proposed
fix; **investigation and design only**, implementation deferred to the next
round (matching every prior pr/40 round's scoping discipline). Owner also
set the general principle bounding the fix's location: the fix belongs
**after** the neutrino vertex is determined -- direction-match the
disconnected object against the *determined* vertex, jump the gap if it
matches, then fix the PF particle. Explicitly **not** in scope: the earlier
imaging-stage severing (`clustering_separate`/`ClusteringProtectOverclustering`)
-- 286906's underlying gap is a signal-processing inefficiency and the owner
states plainly that stage is not fixable.

## Why no reruns this round

A peer Claude session is concurrently editing `NeutrinoShowerClustering.cxx`
/ `PRShower.cxx` (doc pr/91 round 3, the `complete_structure_with_start_
segment` frontier-walk bug) and has an uncommitted edit in
`run_pr_chain_batch.sh` in this shared `wcp-porting-img` repo. To avoid any
collision this round runs **zero PR chain reruns**: every measurement below
reads already-existing arm output (`work-pr91r2-prod-{mc,ncpi0}`, toolkit
`cca9f167`) plus static code reading. The mechanism attribution below is
therefore **high-confidence but not yet runtime-trace-confirmed** --
`WCT_PID_WRITE_DEBUG=2 WCT_SHOWER_TOPO_DEBUG=1` single-event reruns on
286906/409546 are the first item for the *next* round, once the peer's
runner-script edit has landed.

## Mechanism -- one write site explains all three events

All three flagged objects (286906 shower on segs 9002+9003, 409546 shower on
seg 9000, 521075 shower 18007) carry `start_connection_type == 2` --
cross-cluster directional association, one shower object anchored at a
vertex that belongs to the *main* cluster while its own segment(s) live in a
different `Facade::Cluster`. There is exactly one place in the codebase that
**creates** a conn-2 shower from a fresh (not-yet-classified) track segment:
`PatternAlgorithms::shower_clustering_with_nv_from_vertices`
(`clus/src/NeutrinoShowerClustering.cxx:1302`). (The other four conn-2 write
sites the grep turns up -- `:3497/:3503/:3951/:3954` -- are pi0 vertex
re-anchoring on shower objects that already exist; ruled out, not relevant
to a fresh track-to-electron mislabel.)

`shower_clustering_with_nv_from_vertices` already does almost exactly what
the owner is asking for. It runs a Hough-angle search from each
other-cluster's steiner point cloud to the nearest main-cluster vertex
(`:1440-1524`): computes an angle between the cluster's local direction and
the line to the vertex, refined with a `2 cm`-radius local-center vector when
the raw angle is small; accepts the association at `angle < 60`, or
`angle` in `[50,60]` only if `dis < 6 cm` (`:1523-1524`). This **is** the
direction-match-against-the-determined-vertex test the owner is describing,
already running after `main_vertex` is fixed, already spanning the gap as an
association (`shower->set_start_vertex(vertex, 2)`, `:1533`).

The bug is narrower than "no such mechanism exists." Once the directional
match accepts, `:1601-1620` unconditionally overwrites the anchoring
segment's PID with no straightness or dQ/dx test at all:

```cpp
// NeutrinoShowerClustering.cxx:1601-1620 (shower_clustering_with_nv_from_vertices)
int pdg = 0;
if (start_seg->has_particle_info() && start_seg->particle_info()) {
    pdg = start_seg->particle_info()->pdg();
}
if (pdg == 0 || std::abs(pdg) == 13) {         // <- no length/direct/dQdx test
    auto four_momentum = segment_cal_4mom(start_seg, 11, particle_data, recomb_model, m_mip_dqdx);
    auto pinfo = std::make_shared<Aux::ParticleInfo>(11, particle_data->get_particle_mass(11),
                                                       particle_data->pdg_to_name(11), four_momentum);
    start_seg->particle_info(pinfo);
}
```

This is the same shape of hole that pr/40 rounds 5-6 already closed at three
sibling sites with a `segment_is_straight_long_track` guard: F10
`shower_connect_main_vertex_straight_guard` (`NeutrinoShowerClustering.cxx
:813`, a *same*-cluster connecting-to-main-vertex site), F11
`shower_traj_straight_guard` (`PRSegmentFunctions.cxx:2103`), F13's
proton-daughter-pion guard (`NeutrinoShowerClustering.cxx:831-834`). This
cross-cluster site was simply never given the same treatment.

## Per-event evidence

**286906** (main vertex cluster 41; the object lives in cluster 9,
`is_main_cluster=0`):

| seg | L (cm) | direct/L | med dQ/dx (xMIP) | pdg | flag_shower | role |
|---|---|---|---|---|---|---|
| 9001 | 126.89 | 0.987 | 1.19 | 13 | 0 | the muon body -- NOT in the shower, NOT in PF, NOT in kine |
| 9002 | 8.68 | 0.994 | 1.20 | 11 | 0 | shower's `start_segment` -- the mislabel site |
| 9003 | 5.22 | 0.933 | 0.73 | 11 | 1 | absorbed into the same shower |

`segment_is_straight_long_track` (`length>10cm`, `direct>=34cm` or
`direct/length>0.93`) evaluates **true** on seg 9001 alone (126.89 cm,
direct 125.26 cm, ratio 0.987) and seg 9002 is too short to qualify in
isolation, but it is collinear with 9001 across their shared vertex 9002
(the vertex and segment IDs coincide, unrelated): **4.9 deg kink**, **0.28
cm RMS transverse residual** over the combined 133.9 cm arc, dQ/dx 1.19x vs
1.20x MIP on the two halves -- one straight, MIP-scale object split by the
graph into "the muon" and "the shower's anchor". (seg 9002's own 1.2026x
MIP sits 0.26% above the F2 guard's 1.2 "muon-like" threshold -- a
coincidence, not load-bearing; the collinearity is the robust evidence.)

**409546** (main vertex cluster 41; object in cluster 9, `is_main_cluster=0`):

| seg | L (cm) | direct/L | med dQ/dx (xMIP) | pdg | flag_shower |
|---|---|---|---|---|---|
| 9000 | 15.78 | 0.977 | 2.73 | 11 | 0 |

Same writer, no leftover track piece (cluster 9 here is only this one
segment) -- a cleaner instance of the identical bug, without 286906's
"orphaned muon body" symptom. Owner's "seems OK, it's short, could be an
electron" is a read of PID plausibility; it does not settle the separate
graph/PF-accounting question this round investigates. Recorded here rather
than resolved by fiat -- flagged as an open question below.

**521075** (main vertex cluster 94; shower 18007 in cluster 18): 18
segments, 15 in the main cluster, `kine_best=674.5 MeV` -- a genuine
multi-pronged cascade. Its one long/straight member (seg 18026, 27.7 cm,
ratio 0.866, 2.31x MIP) is a normal bremsstrahlung/conversion sub-track
inside a real shower, not a lone dominant object. Correctly a shower;
untouched by anything proposed below.

## Gap metric -- point-cloud closest approach, not fitted-vertex distance

Fitted-vertex-to-vertex distance gives **1.89 cm** (286906) / **2.80 cm**
(409546) / **3.94 cm** (521075) -- a thin, unsafe margin sitting right
against doc pr/84 round 2's own proven-**ADVERSE** 2.5-2.9 cm bridges
(nueCC 38856, the `conn3_stitch_max` 3 cm sweep that fragmented a 1244 MeV
electron and flipped `nue` 3.25 -> -3.45).

The **steiner / track_shower point-cloud** closest approach between the two
clusters is a cleaner, less fit-biased metric:

| evt | steiner cl-to-cl (cm) | track_shower cloud (cm) | verdict |
|---|---|---|---|
| 286906 | 1.387 | 1.297 | must bridge |
| 409546 | 1.277 | 1.109 | must bridge |
| 521075 | 2.916 | 2.837 | must NOT bridge |

A full **1.5 cm margin** separates the two must-fix cases (1.11-1.39 cm)
from the must-not-touch case (2.84-2.92 cm) on this metric, vs. ~0.9 cm and
sitting on top of the adverse band on the fitted-vertex metric. **Proposal:
gate any next-round gap-jump on point-cloud closest approach at
~1.5-2.0 cm**, not vertex-fit distance -- satisfies "the cut should be as
small as possible" while clearing both required cases with margin.

**`conn3_stitch_max` (the existing graph-bridge knob, SBND production =
1 cm) cannot reach either case regardless of radius.**
`stitch_disconnected_main_cluster` (`NeutrinoGraphAudit.cxx:1658`) only
iterates segments with `is_main_cluster==true`; cluster 9 in both events has
`is_main_cluster=0`. This is a structurally different code path
(cross-cluster directional association vs. same-cluster stitch) -- raising
`conn3_stitch_max` is not the fix and should not be proposed next round.

## PF / energy omission -- a second, separable bug, already present today

Independent of the pdg mislabel: 286906's 127 cm muon (seg 9001) sits in a
`Facade::Cluster` that is never the nusel main cluster. Under SBND
production knobs (`pf_track_main_cluster_only=true`,
`pf_shower_vertex_barrier=true`, `pf_orphan_track_parentage=true`) it is
invisible to `fill_bee_pf_tree`'s main-vertex BFS
(`MultiAlgBlobClustering.cxx`) and to `fill_kine_tree`'s BFS+shower walk
(`NeutrinoKinematics.cxx`) -- both gate on `same_cluster()` /
`used_vertices` unconditionally in every orphan-rescue path that exists
today (`pf_orphan_track_parentage`'s rescue included). Today the muon
contributes **zero** PF nodes and **zero MeV** to `kine_reco_Enu`, silently
-- exactly the failure mode the owner names ("should not be counted in the
PF or energy reconstruction... if not relevant", except here it *is*
relevant and is being dropped anyway).

Worth stating plainly: `kine_energy_included` (values 1 vs 3) is
**advisory only** -- `NeutrinoKinematics.cxx:330-334` sums every element of
`kine_energy_particle` into `kine_reco_Enu` regardless of the `included`
flag. "Should not be counted in energy reconstruction" therefore cannot be
implemented by writing `included=3`; only actual omission from
`kine_energy_particle` (today's default, silent, and in 286906's case
wrong) achieves that.

## The "hadron -> neutron" carrier the owner asked for already exists

`MultiAlgBlobClustering.cxx:1805-1833` (`append_pseudo_shower`) already
assigns a conn-2/3 pseudo-parent PDG **2112 (neutron)** whenever the wrapped
shower's own PDG is not 11/22 (gamma pseudo-parent otherwise):

```cpp
const int pdg = (std::abs(sh->get_particle_type()) == 11 ||
                 std::abs(sh->get_particle_type()) == 22) ? 22 : 2112;
```

Nothing new is needed for the owner's "hadron track -> neutron hadron
track" case -- it is already implemented display-side. The actual gap is
entirely upstream: getting a genuinely hadronic disconnected object correctly
PID'd (not force-set to 11) so this existing carrier logic ever sees it.

## 45-event Bee census cross-check (existing round-7 data, zero rerun)

Round 7's own census TSV already carries a `straight_long` column. Re-read
against the 45-event Bee scan set (idx list in round 7's Bee table above):
**40 of 45** have `straight_long=True` on their dominant flagged segment --
the same-shape pathology (a straight, MIP-scale object mislabeled electron)
is systemic across the sample, not confined to this one writer site.

**Caveat, stated plainly rather than merged into one number**: round 7's
census predicate required `is_main_cluster` (`particle_id==11 AND
is_main_cluster AND length>20cm`), which structurally **excludes**
286906/409546 (cluster 9, `is_main_cluster=0` in both). The 45-event
population and the {286906, 409546} pair are two overlapping-in-mechanism
but non-identical populations: the 45-event set is dominated by
same-cluster reclassification writers (the unguarded branches in
`NeutrinoVertexFinder.cxx examine_direction`, `:1634` and `:1667-1715`, and
similar sites), while 286906/409546 are this round's cross-cluster
`shower_clustering_with_nv_from_vertices` site. Both need the same shape of
fix (a straightness/dQ-dx guard before the pdg=11 write, reusing
`segment_is_straight_long_track` as F10/F11/F13 already do) but at different
call sites -- this round only designs the cross-cluster one; the 45-event
population is flagged as the same-cluster analogue for a follow-up round,
not attempted here.

## Comprehensive fix design (named, NOT implemented this round)

Two parts, matching the owner's own two-clause ask ("jump the gap" + "fix
the PF particle").

**Part A -- PID guard (small, mirrors existing precedent).** Add a
`segment_is_straight_long_track` test (reusing the existing helper, no new
logic) into `shower_clustering_with_nv_from_vertices`'s pdg-write block
(`NeutrinoShowerClustering.cxx:~1601-1620`), same shape as F10/F11/F13:
decline the `pdg==0||pdg==13 -> 11` overwrite when the candidate (or its
collinear continuation across the shared vertex, per 286906's 9001+9002
pattern) tests straight and long. Proposed knob name
`shower_connect_from_vertices_straight_guard`, C++ default `false`, key
omitted when off => byte-identical. **This alone fixes the mislabel for
both 286906 and 409546** -- it does not by itself connect anything or fix
the PF/energy omission.

**Part B -- the gap-jump + PF fix. Two candidate shapes; this round
presents both rather than choosing (owner input wanted before next round
commits):**

- **B1 -- display/energy patch, no graph edit.** Leave the PR graph as two
  separate `Facade::Cluster`s. When Part A declines the electron
  conversion, additionally (a) extend the PF-tree orphan rescue --
  `pf_orphan_track_parentage`'s `same_cluster()` gate is unconditional
  today (`MultiAlgBlobClustering.cxx`, both orphan pools) -- to also walk a
  declined-electron association's own cluster, and (b) extend
  `fill_kine_tree`'s BFS (`NeutrinoKinematics.cxx`) the same way so the
  muon's KE is actually summed. Same shape/risk class as doc pr/84's F1/F2
  (display+kinematics only, no fit/PID side effects). Does **not**
  literally "connect the track to the main cluster" -- it is a bookkeeping
  fix layered on top of a graph that stays split, so anything downstream
  that queries graph connectivity (rather than walking the PF/kine trees)
  still sees two objects.
- **B2 -- the graph bridge the owner actually asked for.** Reuse the
  *already-computed* directional match from
  `shower_clustering_with_nv_from_vertices` (the same angle/distance test
  gating Part A) to trigger a real cross-cluster bridge: extend
  `connect_direct`/`create_segment_for_cluster`
  (`NeutrinoPatternBase.cxx`) -- today scoped to a single `Facade::Cluster`
  because `do_rough_path`'s Dijkstra runs on that cluster's own
  `"steiner_graph"` -- to build a direct bridge segment across the
  point-cloud gap when it is small (this round's proposed ~1.5-2.0 cm
  point-cloud threshold) and the candidate is straight/MIP-like. Because
  the gap is a signal-processing hole with no charge to route through (per
  owner), a straight-line bridge segment is the natural shape here, not
  `do_rough_path` routing -- consistent with how `connect_direct` already
  falls back safely when routing fails. This is the structurally correct
  fix: once merged, every downstream PID/PF/kine consumer sees one
  continuous track and needs no special-casing on either end. **Higher
  risk/effort**: no code path in this codebase today merges two
  `Facade::Cluster`s or creates a segment that crosses that boundary --
  this is new machinery, not an existing-knob extension, and needs its own
  design review (does the bridge segment get its own `Facade::Cluster`
  reassignment, or do the two clusters merge; how does this interact with
  the `Facade::Cluster`-scoped assumptions baked into `do_rough_path`,
  `mvga`, and `conn3_stitch_max` elsewhere in the same pipeline).

**Recommendation for next round: B2**, on the strength of the owner's own
framing ("modify the Graph") and because B1 leaves exactly the kind of
graph/display divergence doc pr/84 rounds 1-3 spent three rounds cleaning up
(mc.json moving independently of the underlying graph). But B2's cost and
architectural novelty should be confirmed with the owner before committing,
rather than assumed.

## Open questions for the owner (next round should get an answer before
implementing)

1. **B1 vs B2** -- accept the graph-bridge scope and cost of B2, or start
   with the lower-risk B1 display/energy patch and revisit B2 later?
2. **If B2**: does the bridge segment simply move into the main
   `Facade::Cluster` (folding the smaller cluster's charge into the main
   one), or should the two clusters be formally merged? This affects every
   other `Facade::Cluster`-scoped mechanism in the pipeline (mvga,
   `conn3_stitch_max`, `do_rough_path`) and is worth deciding once rather
   than per-site.
3. **Threshold**: confirm the proposed ~1.5-2.0 cm point-cloud
   closest-approach cut (vs. 1.11-1.39 cm for the two must-fix cases,
   2.84-2.92 cm for the must-not-touch case) -- or does the owner want a
   different margin / a different metric entirely?
4. **409546**: bridge it under the same rule as 286906 (this round's
   evidence says it is the identical writer bug), or treat "seems OK, could
   be an electron" as an owner override that should exempt it from Part A
   too? The round-8 default assumption is: fix the writer bug uniformly
   (Part A applies to both), and let B1/B2 decide independently whether
   409546's now-correctly-PID'd object is worth graph-connecting (it may
   end up staying displayed as a short, disconnected, correctly-labelled
   object either way, since it has no leftover track body like 286906's
   9001) -- flagging this reading for owner confirmation rather than
   assuming it.

## Scope and not-claimed

- No C++, jsonnet, or config change this round -- owner explicitly scoped
  this round to investigation, understanding, and fix design; implementation
  deferred to the next round.
- Mechanism attribution (`shower_clustering_with_nv_from_vertices`,
  `:1601-1620`) is **static-code-read, not yet runtime-trace-confirmed** --
  no `WCT_PID_WRITE_DEBUG`/`WCT_SHOWER_TOPO_DEBUG` rerun was done this round
  (see "Why no reruns this round" above). First item for the next round.
- Zero PR chain reruns this round; all evidence comes from arms that already
  existed before this round started (`work-pr91r2-prod-{mc,ncpi0}` at
  toolkit `cca9f167`, `work-pr40r7cen-*` from round 7).
- Neither Part A's guard shape nor B1/B2 is gated, tested, or wired into any
  component -- both are documented candidates only, with open questions
  above that the owner should answer before the next round picks one.
- The 45-event Bee census cross-check is a re-read of round 7's own
  (uncommitted, scratch) TSV -- no new census script, no new manifest run.
  The same-cluster analogue of this round's cross-cluster fix (the writers
  behind that 40/45 population) is named but not designed this round.
- 521075 confirmed unaffected by anything proposed here; no action needed
  on it.

---

# Round 9 (2026-08-18) -- rounds-7+8 fixes IMPLEMENTED: five straight-track
PID guards + the B2 cross-cluster bridge; small-scale validation (45-Bee +
nueCC48 + NCpi0-19); SBND production flip

## Repro block

```bash
# Build (toolkit @ 02b26421 + this round), freshness proof, unit tests
cd /nfs/data/1/xqian/toolkit-dev/toolkit
./wcb build --notests -p && ./wcb install --notests -p    # rc=0 both
ls -la local/lib/libWireCellClus.so                       # 2026-08-18 12:59 -- newer than every source edit
./build/clus/wcdoctest-clus                               # 210/210, 0 failed

# Compiled-config proofs (M6), full PR pipeline TLA:
#   bare compile byte-identical to pre-change snapshot (cmp rc=0);
#   full-pipeline knobs-off compile byte-identical to a compile of the
#   HEAD-jsonnet overlay (cmp rc=0);
#   all 9 new keys present when forced on via --tla-code (grep, 1 each).

cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
BEE45="350935 283713 55595 407280 281837 55539 314507 64921 71222 316025 395610 285567 280972 401450 290729 138009 395060 286191 348471 69314 30504 286681 90055 293149 56982 321371 349461 352233 234638 214469 389538 277298 349549 433451 278684 292643 315167 268067 239794 386948 64409 437699 348691 54095 320865"
MCP47="$BEE45 286906 409546"       # 45-Bee set + the two round-8 must-fix events

# V2 OFF arms (bare run == production, doc 68)
PR_JOBS=32 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-mcp1k-cb0805   work-pr40r9-off-mcp1k   data $MCP47
PR_JOBS=32 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr40r9-off-nuecc48 data
PR_JOBS=32 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-ncpi0-cb0805   work-pr40r9-off-ncpi0   data
# V2 ON arms: same three commands with
#   SBND_SHOWER_CONNECT_FROM_VERTICES_STRAIGHT_GUARD=1
#   SBND_SHOWER_CONNECT_START_SEG_STRAIGHT_GUARD=1
#   SBND_EXAMINE_DIRECTION_DIRSIGN_SHOWER_IN_GUARD=1
#   SBND_DAUGHTER_SHOWER_ANGLE_RECLASS_STRAIGHT_GUARD=1
#   SBND_SHOWER_TOPO_REEXAM_STRAIGHT_GUARD=1
#   SBND_SHOWER_NV_BRIDGE_TRACK=1
#   SBND_PF_TRACK_BRIDGED_CLUSTERS=1
# into work-pr40r9-on-{mcp1k,nuecc48,ncpi0}  (scalars stay C++ 25 deg / 1.8 cm)

# V1 (pre-change ref, 3 spot events): git stash -> rebuild -> run
# 286906+409546 (mcp1k) + 521075 (ncpi0) into work-pr40r9-prechange-ref-* ->
# stash pop -> rebuild; compare vs the V2 OFF arms:
python3 scripts/pr85_hash_gate.py work-pr40r9-prechange-ref-mcp1k work-pr40r9-off-mcp1k
python3 scripts/pr85_hash_gate.py work-pr40r9-prechange-ref-ncpi0 work-pr40r9-off-ncpi0

# V2 gates + movers
python3 scripts/pr85_hash_gate.py work-pr40r9-off-mcp1k   work-pr40r9-on-mcp1k
python3 scripts/pr85_hash_gate.py work-pr40r9-off-nuecc48 work-pr40r9-on-nuecc48
python3 scripts/pr85_hash_gate.py work-pr40r9-off-ncpi0   work-pr40r9-on-ncpi0
diff work-pr40r9-off-nuecc48/nusel-table.tsv work-pr40r9-on-nuecc48/nusel-table.tsv
diff work-pr40r9-off-ncpi0/nusel-table.tsv   work-pr40r9-on-ncpi0/nusel-table.tsv
python3 scripts/analysis/pr40/pr40r7_census.py work-pr40r9-off-mcp1k work-pr40r9-off-nuecc48 work-pr40r9-off-ncpi0 --min-len 20 --out /home/xqian/tmp/pr40r9/census_off.tsv
python3 scripts/analysis/pr40/pr40r7_census.py work-pr40r9-on-mcp1k  work-pr40r9-on-nuecc48  work-pr40r9-on-ncpi0  --min-len 20 --out /home/xqian/tmp/pr40r9/census_on.tsv

# Bee A/B (48 events: 45-Bee order + 286906 409546 521075)
python3 scripts/bee/make_pr_bee.py -q work-mcp1k-cb0805 -q work-ncpi0-cb0805 \
  -p work-pr40r9-off-mcp1k -p work-pr40r9-off-ncpi0 \
  -o bee/pr40r9/pr40r9-off.zip $BEE45 286906 409546 521075
python3 scripts/bee/make_pr_bee.py -q work-mcp1k-cb0805 -q work-ncpi0-cb0805 \
  -p work-pr40r9-on-mcp1k -p work-pr40r9-on-ncpi0 \
  -o bee/pr40r9/pr40r9-on.zip $BEE45 286906 409546 521075
./upload-to-bee.sh bee/pr40r9/pr40r9-off.zip   # -> <OFF-URL>
./upload-to-bee.sh bee/pr40r9/pr40r9-on.zip    # -> <ON-URL>

# V3 (after the cfg flip): bare reruns vs the ON arms
# PR_JOBS=32 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh ... work-pr40r9-bare-... data ...
# python3 scripts/pr85_hash_gate.py work-pr40r9-on-<s> work-pr40r9-bare-<s>
```

## Owner decisions opening this round (2026-08-18)

1. Scope = round 8 Part A + **B2 graph bridge** (round 8's recommendation).
2. **Include the round-7 five named in-cluster candidates** -- they are the
   fixes for the 45-Bee-event population, implemented this round.
3. Gap cut = **1.8 cm** point-cloud closest approach (midpoint of the
   proposed 1.5-2.0; must-fix 1.11-1.39, must-not 2.84-2.92).
4. **409546 uniform**: same guard rule for all; Part A/B2 decide by geometry,
   no per-event override.

## Site corrections found implementing (supersede round-7/8 prose)

- **D1 -- round 7 candidate 2c's ":1401" IS the round-8 Part A site.** The
  only `pdg==0||abs(pdg)==13 -> 11` write in NeutrinoShowerClustering.cxx is
  the one in `shower_clustering_with_nv_from_vertices` (line drift since the
  trace-era commits; verified via `git show ba21b2da:...` -- the write sat at
  :1385-1401 then). Corroboration: 54629's seg 20013 lives in cluster 20 with
  main cluster 15 -- cross-cluster, which only `from_vertices` produces.
  `shower_connect_start_seg_straight_guard` is therefore re-targeted at the
  genuinely-unguarded accept-time `set_pdg(11)` in
  `shower_clustering_connecting_to_main_vertex` (the real same-cluster
  analogue: F10 vetoes only at seed time on the seed's OWN geometry, so a
  short anchor collinear with a long straight sibling passes F10 and reaches
  the accept-time write).
- **D2 -- round 7 candidate 2a's "dirsign()==0 branch" is dead code** (the
  branch sits after dirsign is assigned +-1 six lines above), and the ":1659"
  trace line is the already-guarded pr/74 P1 branch. The real coverage gap is
  P1's thresholds: `segment_shower_in_cascade_vetoed` requires length > 40 cm
  AND median dQ/dx < 1.3xMIP; 54629's seg 15007 (31.0 cm, 1.42xMIP) fails
  both conjuncts, but its 0.97 direct/arc straightness is decisive.
  `examine_direction_dirsign_shower_in_guard` therefore adds a GEOMETRY arm
  (`segment_is_straight_long_track`) beside the P1 veto in all three
  flag_shower_in branches -- it widens pr/74 P1 by geometry, it does not
  "cover a missed branch".
- **D3 -- Part A alone would have been a NO-OP for 286906**: after the
  declined write, `shower->update_particle_type` runs for the same shower and
  redoes 13->11 via the majority vote (PRShower.cxx:981-1009 -- a pdg-13
  start segment counts toward shower_length, and a pure 9002+9003 shower
  always trips `shower_length > track_length`). The guard co-guards that call
  (skipped only when the guard fired; single-segment 409546 is immune anyway
  via the `edges().size()<=1` early return).
- **D4** -- knob-5 site: escape condition NeutrinoVertexFinder.cxx (topology
  re-exam in `improve_vertex`), third arm added to the re-demote condition;
  write sites confirmed at current line numbers.

## What was implemented (all C++ defaults false; keys suppressed when off)

| knob | site | effect when ON |
|---|---|---|
| `shower_connect_from_vertices_straight_guard` | `shower_clustering_with_nv_from_vertices` pdg block + `update_particle_type` co-guard (D3) | decline the cross-cluster track->e- overwrite when the anchor or its collinear continuation (new shared helper, below) is straight-long; shower kept as conn-2 with track PID => `append_pseudo_shower` renders the 2112 neutron carrier |
| `shower_connect_start_seg_straight_guard` | `shower_clustering_connecting_to_main_vertex` accept-time `set_pdg(11)` (D1) | decline the pdg write only; kAvoidMuonCheck/structure untouched |
| `examine_direction_dirsign_shower_in_guard` | `examine_direction` flag_shower_in branches (D2) | geometry arm beside pr/74 P1 cascade veto |
| `daughter_shower_angle_reclass_straight_guard` | `examine_direction` daughter-shower angle reclass | guard the write only (guarding the outer condition would re-route control into the pdg==11 else-if chain) |
| `shower_topo_reexam_straight_guard` | `improve_vertex` topology re-exam | third re-demote arm: straight-long keeps the track PID instead of the set_flags+pdg-11+score-100 escape.  SAFETY NET framing per round 7 -- the pr/90-side fix for 320865 remains the open alternative |
| `sfv_kink_max` (25 deg) | continuation-arm tunable | max kink for the collinear-continuation test |
| `shower_nv_bridge_track` + `shower_nv_bridge_max_gap` (1.8 cm) | Step 5 of `from_vertices`, before shower creation | B2: straight-long cross-cluster candidate with exact steiner-cloud gap < cut => NO conn-2 shower, NO pdg-11; a straight 2-point zero-charge bridge segment (main-cluster-stamped, two synthetic fits) joins the main-cluster vertex to the track (break_segment at an interior point, the legacy break reused); bridge + rescued-cluster segments shielded from every shower flood-fill/absorber (8 complete_structure call sites pre-seeded, 6 absorber guards, in_other_clusters skip); bridged ids transported on TrackFitting |
| `pf_track_bridged_clusters` | `fill_bee_pf_tree` BFS gates | lets the PF track BFS traverse nv-bridged clusters despite `pf_track_main_cluster_only` (orphan pools untouched) |

New shared helper `segment_is_straight_long_track_or_continuation(graph, seg,
max_kink_deg=25)` (PRSegmentFunctions): `segment_is_straight_long_track(seg)`
OR a same-cluster sibling across either endpoint vertex collinear within
`max_kink_deg` (`segment_pair_kink_deg`, -1=unmeasurable never collinear) that
is itself straight-long or forms a qualifying combined chain (same 10 cm /
34 cm / 0.93 constants).  Needed because PATH C hands the guard a broken
sub-10 cm HALF of the track (286906: 8.68 cm anchor at 4.9 deg to the
126.89 cm body).

Kinematics side needs NO edit: `fill_kine_tree` gates on graph reachability
only, so the BFS traverses the bridge (one extra 0 MeV / pdg-0 element in
`kine_energy_particle` for the bridge itself) and the far track's KE is
summed; the pdg-13/13 continuation rule de-duplicates the muon rest mass.

## RESULTS

### V1 -- knobs-off byte-identical to a genuine pre-change build: PASS

`git stash` -> rebuild -> run 286906+409546 (mcp1k) and 521075 (ncpi0) as
`work-pr40r9-prechange-ref-{mcp1k,ncpi0}` -> `git stash pop` -> rebuild.
`pr85_hash_gate.py` vs the V2 OFF arms: **PASS 4/4 + 2/2 archives
byte-identical.**  (Compiled-config side: bare compile and full-PR-pipeline
knobs-off compile both byte-identical to HEAD-jsonnet compiles, cmp rc=0.)

### V2 -- OFF vs ON, 100 events (33 mcp1k + 48 nueCC + 19 NCpi0): all clean

Note: the 45-event Bee set spans all three samples (31 mcp1k / 12 nueCC48 /
2 ncpi0 -- round 7's census covered all three), so the mcp1k arm carries 33
events (31 Bee + 286906 + 409546) and the nueCC48/NCpi0 full arms cover the
rest.  All 200 per-event runs rc=0.

- **Selection: zero movers.**  `nusel-table.tsv` byte-identical OFF vs ON in
  all three samples.
- **Archives: 30 events differ, mabc-pr.zip ONLY; 0 pctree movers** (the
  pctree archive carries the `clustering_` imaging tree, not PR segments, so
  this is expected).  Movers: 20/33 mcp1k, 9/48 nueCC48, 1/19 ncpi0.
- **Bridges fired on exactly three events** (`pr40r9 nv_bridge` log lines):
  | evt | gap | outcome |
  |---|---|---|
  | 286906 | 1.39 cm | seg 9002 anchor declined e- (continuation arm, 4.9 deg to 9001); bridge cluster 9 -> main 41; PF tree gains **mu- 308.6 MeV (seg 9001)**; `kine_energy_particle` [56.7, 63.2(fake e-), 4.3, 1.4] -> [56.7, 0.0(bridge), 48.4(pi+ 9002), **308.6(mu- 9001)**, 10.2(e- 9003), 4.3, 1.4] -- kine_reco_Enu 125.6 -> 429.6 MeV.  Owner shape achieved; 9002's own label lands pi+ from ordinary track PID (the F10 G2 class -- geometry fixed, per-segment PID owner-reviewable in Bee). |
  | 409546 | 1.28 cm | uniform rule: seg 9000 e- 90.9 MeV -> **mu- 63.9 MeV**, bridged.  Owner's "seems OK, could be an electron" recorded; if the owner overrides, the geometry gates cannot distinguish it -- a dQ/dx cap knob would be the follow-up. |
  | 407280 | 0.68 cm | bonus: Bee idx 3's flagrant 128.8 cm, 1.14xMIP seg 16010 is this same cross-cluster class -- bridged. |
  **521075 (must-not-touch, gap 2.92 cm): byte-identical** -- no bridge, no
  guard, not among the movers (the single ncpi0 mover is 259542).
- **320865** (knob 5): seg 13001 e-/kShowerTopology -> **pi+ track**, flag
  cleared.
- **Census** (`pr40r7_census.py`, same predicate as round 7,
  `/home/xqian/tmp/pr40r9/census_{off,on}.tsv`): candidate rows 172 -> 147;
  **straight_long=True 63 -> 42**.  22 straight-long e- rows fixed
  (277298/17003, 281837/13002, 286191/63011, 286681/72038, 314507/51007,
  320865/13001, 321371/18004, 348691/51079, 349461/71014, 352233/51012,
  386948/16005, 389538/19040+19041, 395060/24012, 401450/24073+24074+24076,
  407280/16010, 55539/23005, 64921/11002, 71222/22007, 90055/13048).
- **Displacement check (the F11 lesson): one new straight-long e- row**,
  235435/2054 (38.6 cm, 1.57xMIP, flag_shower=0), from a re-segmentation
  inside that nueCC event's shower region (segs 2041/2043 replaced by
  2054/2055 + a new proton 2057); two more re-segmentation movers without a
  new straight-long row: 163543 (segs 14118/14121/14123 -> 14120/14122/14124,
  one piece now mu-) and 174637 (new segs 9096-9102, one particle_id=4).
  All three are nueCC48 signal events, none change nusel; all three are
  appended to the Bee A/B set (idx 48-50) for owner review.
- **Guard firing counts** (events with >=1 firing, ON arms):
  `shower_connect_from_vertices_straight_guard` 14 (15 declines, every
  anchor pdg=13), cascade `straight_veto` 10, `daughter_shower_angle_
  reclass_straight_guard` 7, `shower_topo_reexam_straight_guard` (no log
  line of its own; visible via 320865-class flips),
  `shower_connect_start_seg_straight_guard` **0 firings on this manifest**
  (narrow by design; kept for the F13 55715-class shape).
- The remaining 42 straight-long e- (350935 idx 0, 55595 idx 2, ...) are
  written by sites NOT named in rounds 7-8 -- same population round 7
  predicted would need its own trace round.  NOT claimed fixed.

### Flip -- SBND PRODUCTION ON (owner-authorized scope, 2026-08-18)

All 7 bools flipped `true` in `cfg/pgrapher/experiment/sbnd/
wct-pr-perevt.jsonnet` (scalars stay null => C++ 25 deg / 1.8 cm).
Compiled-config proof: full-pipeline compile shows all 7 keys `: true`,
scalar keys absent.  **V3 bare-run composition gate: bare reruns
`work-pr40r9-bare-{mcp1k,nuecc48,ncpi0}` vs the ON arms -- PASS 66/66 +
96/96 + 38/38 = 200/200 archives byte-identical** (doc 68: bare run ==
production, single-sourced in cfg).

## Bee links (51-event A/B pair, index `bee/pr40r9/pr40r9.index.txt`)

- BEFORE (previous production):
  https://www.phy.bnl.gov/twister/bee/set/ae0253c4-ce14-4bea-a387-14a5fffaa59e/event/list/
- AFTER (round 9 ON = new production):
  https://www.phy.bnl.gov/twister/bee/set/7b48de57-9bb2-4ff1-a3c8-abb1bea4b094/event/list/

Set = the 45 round-7 Bee events (same order) + 286906/409546/521075 (idx
45-47) + the three re-segmentation movers 163543/174637/235435 (idx 48-50).
25 of 51 differ between the sets; the index file annotates each.

## Scope and not-claimed

- The remaining 42 straight-long pdg-11 census rows (incl. the two largest
  Bee cases 350935 and 55595) are OTHER writer sites -- not designed, not
  fixed, the natural next trace round.
- `shower_connect_start_seg_straight_guard` shipped ON but never fired on
  this manifest -- its census bite is unmeasured (flipped as part of the
  family for uniformity; trivially revertable).
- 409546 is bridged+muon under the uniform rule; the owner's "could be an
  electron" comment is the standing counter-signal to re-examine in Bee.
- 235435's new straight-long e- seg 2054 is the one displacement-class
  artifact found; owner review requested (Bee idx 50).
- mcp2k and the full mcp1k are NOT covered (owner scoped this round to
  45-Bee + nueCC48 + NCpi0-19); the 1000-event evaluation is deferred.
- Runtime-trace confirmation of the round-8 static attribution is now DONE
  (the `pr40r9 nv_bridge`/guard log lines above are the runtime evidence).

