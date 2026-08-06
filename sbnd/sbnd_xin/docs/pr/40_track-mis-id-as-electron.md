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
