# doc pr/85 — near-vertex PR quality: the interposed stub

**Status: investigation only. Nothing is implemented — no C++, no jsonnet, no
knob.** The one experiment run here uses `-A` overrides on knobs that already
exist.

Owner report (2026-08-15), from the neutrino-vertex hand scan:

> "1. near the vertex, one track aiming at the vertex got merged to a nearby
> track first, and then missing the last part of connecting to neutrino vertex.
> 2. near the vertex, there are multiple tracks go zig and zag, and multiple
> small segments are showing up. Note, for a neutrino vertex, in general the
> connection should be simple, it would be one vertex with multiple tracks (or
> shower stem) coming out of it. Note in my existing PR code, I have many
> `examine*` are to improve this situation, but we still have some failures."

The answer to the last sentence is §3: **every `examine_*` pass runs before this
clutter exists, and the one pass that runs after it is disabled in SBND
production.**

This doc continues doc pr/84 §5 ("the improvement family"), which measured the
same defect from the particle-tree side and left it at 29 hierarchy inversions
in 21 events. pr/84's measurements are not restated or edited here.

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# the census: every number in secs 2, 3.3 and 5.2 (read-only, ~40 s)
python3 pr85_near_vertex_census.py

# the three experiment arms on the six events (three ql roots each; shown for
# mcp1k, repeat with work-nuecc48-cb0805 / work-ncpi0-cb0805 for the rest)
export PR_JOBS=3 SBND_DL_VTX_MIN_ACCEPT=10 PR_EXTRA_STAGES=pr_display
export SBND_WCT_LOGLEVEL=debug WCT_ES3_MERGE_CENSUS=1
./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr85-b2    data 59685 280972 278785
SBND_MAIN_VERTEX_GRAPH_AUDIT=true \
./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr85-mvga2 data 59685 280972 278785
SBND_MAIN_VERTEX_GRAPH_AUDIT=true SBND_MVGA_SATELLITE=3.0 \
./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr85-sat3  data 59685 280972 278785

# the sentinels.  -a is REQUIRED: WCT logs carry invalid UTF-8, so plain grep
# treats them as binary and prints nothing at all.
grep -a 'snap_main_vertex_to_kink: SNAP' work-pr85-b2/pr_evt*/wct_pr_evt*.log
grep -a 'ES3CENSUS'   work-pr85-b2/pr_evt*/wct_pr_evt*.log | grep -ac 'predmerge=1'
grep -a 'mvga: op'    work-pr85-mvga2/pr_evt*/wct_pr_evt*.log | grep -av eval
```

Binary pinned for every arm above: `local/lib/libWireCellClus.so` mtime
**2026-08-15 15:41**, unchanged across all runs (checked before and after).
Note this is *later* than the `-ma10` dumps the population census reads — see
the caveat in §2.4.

---

## 1. Symptom — six events, plus two the six do not cover

Picked by the rule in §2.3 from the 473 hand-scan labels: three of each mode,
non-isochronous, the owner's click agreeing with the reconstruction, and not
already owned by doc pr/83 or pr/84. Panels rendered with
`vtx_rules/scankit.py:panel_zoom` at the clicked vertex.

Which of the owner's two sentences each answers is stated per group, because
they are not interchangeable.

**Mode 1a-VIA — the prong reaches the vertex only through a stub.** This is
sentence (2)'s "small segments" and sentence (1)'s "missing the last part of
connecting to the vertex" at the same time: the last part is not missing, it is
a separate 0.3–1.3 cm object.

| event | clicked vertex | its degree | the prong | interposed stub |
|---|---|---|---|---|
| **evt59685** | 6007 | 2 | seg 6002, **103.3 cm**, charge 1.18 cm from the click | seg 6006, **1.18 cm** |
| **evt280972** | 7174 | 3 | seg 7030, **71.2 cm**, charge 0.29 cm away | seg 7158, **0.29 cm** |
| **evt360535** | 7022 | **1** | seg 7089 **43.1 cm** *and* seg 7097 **30.6 cm**, both 1.27 cm away | seg 7024, **1.27 cm** |

evt360535 is the sharpest: the reconstruction put the neutrino vertex at the
**free end of a 1.27 cm stub** (degree 1), while the actual three-prong junction
7001 — carrying 43 cm and 31 cm of track — sits 1.27 cm away and is not the
vertex.

**Mode 2 — a chain of stubs at the vertex.**

| event | clicked vertex | its degree | stubs within 3 cm |
|---|---|---|---|
| **evt278785** | 5003 | 3 | 2.25 cm (seg 50015) + 2.21 cm (seg 5016) |
| **evt437699** | 11005 | 4 | 2.12 cm (seg 11011) + 1.48 cm (seg 11099) |
| **evt314838** | 18018 | 2 | 0.30 cm (seg 18021) + 0.72 cm (seg 18093) |

The two groups above are one phenomenon seen twice: in mode 1a-VIA the stub
stands *between* the vertex and a real prong; in mode 2 it hangs *off* the vertex
with a free end. evt280972 and evt314838 are both at once.

**Two more, because the six above do not contain the shape sentence (1)
describes most literally.** "Merged to a nearby track" reads most plainly as
*one segment where there should be two*, or *a prong that never joins at all*.
Neither is in the six, so one of each is named here. They are symptom examples
only — the §5 experiment was run on the six, not on these — and §7 Q5 records
that nothing in this doc addresses them.

| event | shape | what is there |
|---|---|---|
| **evt388224** | **1b STRADDLE** | clicked vertex 3004, degree 4. Segment 3008, **18.0 cm**, passes within 2.66 cm of the click **2.4 cm along its own arc** — its interior, not an end. Two prongs reconstructed as one object, never broken at the vertex. |
| **evt407280** | **1a-CUT** | clicked vertex 51013 in cluster 51, degree 2. Segment 16010, **128.8 cm**, comes within 2.98 cm — and is in **cluster 16**. No PR-graph path exists because the two objects were never clustered together. |

evt407280 is the honest limit of this doc: 6 of the 8 CUT prongs are
cross-cluster, so the missing connection is an upstream clustering decision, not
a PR-graph one — the same boundary doc pr/84 §9 drew for evt284794.

---

## 2. The census

`pr85_near_vertex_census.py`, over the 473 hand-scan labels that have a deployed
`-ma10` dump.

### 2.1 Scored at the click, not at the reconstructed vertex

Every metric is measured at the owner's rank-1 pick (`vtx_io.load_labels()`
→ `truth`), never at the dump's `main_vertex`. Scoring at the reconstructed
vertex would mix "the topology here is ugly" with "the wrong vertex was picked",
and the second is doc pr/80's subject. 462 of 473 labels have a dump vertex
within 1 cm of the click and are scorable; the other 11 are their own class.

### 2.2 What is there

```
mode 1  a >=10 cm prong whose charge comes within 3 cm of the click but
        which carries NO edge to the clicked vertex ............ 32  (6.9%)
          1a-VIA   connected, but only through stubs ........... 21
          1a-CUT   no path to the vertex at all ................  8   (6 of 8 are
                                                                      in a different
                                                                      cluster)
          1b STRADDLE its interior passes the vertex unbroken ..  4
        (excluded: a prong reaching the click through a REAL segment,
         an ordinary grand-daughter topology) .................   6
mode 2  >=2 segments under 3 cm with charge within 3 cm ........ 35  (7.6%)
both modes in one event ........................................ 10
```

**The three-way split of mode 1 is the load-bearing result of this section.**
Reported as one number, "37 detached prongs" invites the fix "add an edge". Two
thirds of them are **already connected** — through a sub-3 cm segment. The fix
is to collapse the interposed stub, not to add anything.

Two population facts worth recording separately:

- **149 of 462 clicked vertices have degree 1.** A third of the hand-scanned
  neutrino vertices are, in the PR graph, the end of a line rather than a
  junction. That is a much larger number than this doc's 32 + 35 and it is not
  explained here.
- **177 of 462 events have ≥2 PR-graph vertices within 3 cm of the click**
  (128 with 2, 39 with 3, 10 with ≥4). The vertex region is routinely
  represented by a cluster of vertices, not one.

### 2.3 The defect is not an artefact of a mis-picked vertex

The doc pr/80 §10.8 discipline: a flag measured only on the events it was
designed for says nothing. Against the 318 events where the click and the
reconstructed `main_vertex` agree within 1 cm:

| | in agreeing events | overall | ratio |
|---|---|---|---|
| mode 1 | 22 / 318 = 6.9 % | 6.9 % | **×1.00** |
| mode 2 | 26 / 318 = 8.2 % | 7.6 % | **×1.08** |

Both ≈1. The clutter is there just as often when the vertex is right, so this is
a topology defect and not a symptom of vertex selection.

Isochronous events are counted but held out of the candidate list, per the
owner's "things can be complicated when isochronous": **18 of the 32 mode-1 and
21 of the 35 mode-2 events** have a near-vertex segment within 10° of the
drift-perpendicular plane (the doc pr/73 §4.5 measure, same bin edge). They are
not claimed to be correct — only out of scope.

### 2.4 Caveat on the population numbers

The census reads the `-ma10` arm, produced 2026-08-15 06:35. The concurrent doc
pr/83 flipped `break_seg_orient = true` in SBND production later the same day,
and the binary was rebuilt at 15:41. `break_seg_orient` changes `break_segment`,
which is exactly the call that creates the interposed stub (§3.2) — so these
population numbers are pre-flip.

All six named events were re-measured at HEAD (`work-pr85-b2`) and are unchanged
in degree, stub count and interposed-prong count. All **57** affected events were
then re-run at HEAD (`work-pr85-head57`, `PR_JOBS=6`, rc=0 on every one) and
re-scored at the same clicks:

| | on `-ma10` (pre-flip) | at HEAD (post-flip) |
|---|---|---|
| mode 1 | 31 | 32 |
| mode 2 | 35 | 34 |
| events whose `(stubs, via, cut, straddle)` changed at all | — | **1 of 56** |

The single mover is **evt400474**, where one stub became an interposed prong
(`2,0,0,0` → `1,1,0,0`) — a reclassification within the family, not a repair.
evt59899 is excluded from the 56: at HEAD its click no longer lands within 1 cm
of any PR-graph vertex, and it is one of doc pr/83's own three events, so its
change is theirs. **The population is stable across the flip** and the §2.2
counts stand.

---

## 3. Root cause

### 3.1 `examine_structure_3` is not involved — 0 proposed merges in 24 junctions

The first reading of complaint 1 — "got merged to a nearby track" — points
straight at `examine_structure_3` (`NeutrinoStructureExaminer.cxx:480`), which
merges any degree-2 junction whose arms are collinear within 18° at 10 cm and
27° at 3 cm, cluster-wide and with no distance-to-vertex awareness.

It is not the culprit. The pass carries its own env-gated census
(`WCT_ES3_MERGE_CENSUS`, `:491`, log-only), which reports every degree-2
junction it evaluates together with `predmerge`. Across the six events:

| event | junctions ES3 evaluated | `predmerge=1` |
|---|---|---|
| 59685 | 1 | 0 |
| 280972 | 5 | 0 |
| 278785 | 1 | 0 |
| 360535 | 13 | 0 |
| 437699 | 2 | 0 |
| 314838 | 2 | 0 |
| **total** | **24** | **0** |

**ES3 never proposed a merge on any of these events.** It also exonerates
`es3_stub_guard` (SBND production **on**, `wct-pr-perevt.jsonnet:1675`): the
guard never had to decline anything, because the angle test never passed.

evt59685's single junction is the reason why. At the time ES3 runs, cluster 6
holds **three vertices and two edges** — two arms of 103.5 and 104.4 cm meeting
at 44.8° (at 2 cm; 52.9° at 3 cm, 60.8° at 5 cm). That is a clean, correct V.
The 1.18 cm stub that the final dump shows **does not exist yet**.

### 3.2 The clutter is created after every `examine_*` pass has run

`examine_*` lives inside `find_proto_vertex` (`NeutrinoPatternBase.cxx:2847-2935`),
`improve_vertex` and `determine_main_vertex`. The two operations that create the
stubs run *after* the overall main vertex is final, in `TaggerCheckNeutrino`:

```
  overall_main_vertex                       TaggerCheckNeutrino.cxx:1379
  snap_main_vertex_to_kink                                        :1394
  improve_vertex   (the final one)                                :1400
  main_vertex_graph_audit                                         :1411   <-- OFF
  clustering_points / examine_direction
```

**`snap_main_vertex_to_kink`** (doc pr/50 `vertex_kink_snap`, SBND production
**on**) is the mechanism for mode 1. When the image shows a turn the fitted
trajectory rounds, it calls `break_segment` at the corner K
(`NeutrinoVertexFinder.cxx:2772`), makes the resulting new vertex the main
vertex, and flags it `kProtectedBreak` — explicitly, in its own comment, to
"shield against `examine_vertices` / ES2 / ES3 re-merging the deliberate break".
The old vertex stays where it was, still carrying the long prong. The piece
between them is the interposed stub. It fires on two of the six:

```
evt59685   SNAP cluster 6 old=(143.70,-142.52,194.78) new=(143.07,-143.22,194.78)
           turn=48.3 deg arc=0.93 cm bendV=28.4 deg
evt280972  SNAP cluster 7 old=( 46.74,-171.71, 84.53) new=( 47.37,-171.62, 84.08)
           turn=108.2 deg arc=1.07 cm bendV=26.1 deg
```

`arc` is the distance from the old vertex to the corner: **0.93 cm** and
**1.07 cm**, against measured interposed stubs of 1.18 cm and 0.29 cm. In
evt59685 the panel shows it directly — 6007 is the snapped-to corner, 6002 is
the old position, and the 103 cm prong never moved.

The snap is doing what pr/50 designed it to do; the corner it moves to is the
better vertex. What it does not do is take the prong with it.

**The final `improve_vertex`** is the other source, and doc pr/51 already said
so — the comment at `TaggerCheckNeutrino.cxx:1405` reads "the micro-stubs it
must absorb are **created there**: 142421's 7081/7082, 285567's 81/82/83". That
is the whole of mode 2's origin, written down a round ago.

### 3.3 `examine_vertices_4` cannot absorb a stub at the main vertex

`examine_vertices_4` (`NeutrinoStructureExaminer.cxx:2024`) is the pass whose job
is exactly this: absorb a segment whose `direct_length < 2.0 cm`, or whose
direction magnitude is `< 3.5 cm` and lies within 10° of drift-perpendicular.

Its length predicate **passes on 8 of the 11 stubs across the six events**. It
absorbs none of them, and the reason is structural rather than numerical. The
pass needs a *dying* vertex satisfying all of (`:2096`):

```cpp
boost::degree(vd1, graph) >= 2 && !v1->flags_any(VertexFlags::kProtectedBreak)
    && examine_vertices_4p(...) && v1 != main_vertex
```

A terminal stub at the neutrino vertex has exactly two endpoints: the main
vertex, barred by `v1 != main_vertex`, and a free end of degree 1, barred by
`degree >= 2`. **There is no third candidate.** The census prints
`ev4_eligible_endpoint: NONE` for every such stub — evt278785's 5016, evt437699's
11011 and 11099, evt314838's 18093, evt280972's 7161.

The guard is right on its own terms — pr/48 added `kProtectedBreak` precisely
because this floor once erased evt59335's correctly-found kink break, and moving
the main vertex from underneath the pass that just chose it would be worse. The
consequence is simply that clutter *at* the main vertex is the one place this
cleanup cannot reach.

---

## 4. Why it hid

A sub-2 cm segment is invisible at any scale a scanner normally works at. In the
20 cm panels of §1 the six events look like clean multi-prong vertices; the
defect only appears at 6 cm half-width. And it does not change the vertex
*position* — in all six the reconstructed `main_vertex` is within 0.01 cm
of the owner's click. Every metric the vertex work has tracked so far (doc pr/78,
pr/79, pr/80: distance to truth at 1 cm) scores these events **correct**, because
by that metric they are. The defect is in the graph the vertex sits in, which
nothing was measuring.

---

## 5. The experiment: turn on what already exists

`main_vertex_graph_audit` is **false** in SBND production
(`wct-pr-perevt.jsonnet:1394`, all `mvga_*` numerics `null` = C++ defaults). It
was written for this defect (doc pr/51) and it is the only pass that runs after
the operations of §3.2. No code is needed to test it.

### 5.1 What it does to the six events

| event | baseline | `mvga=true` | `mvga=true, satellite=3 cm` |
|---|---|---|---|
| evt59685 | deg 2, 1 stub, 1 interposed | — | — |
| evt280972 | deg 3, 3 stubs, 1 interposed | deg 2, **2** stubs | deg 2, **1** stub |
| evt360535 | deg 1, 1 stub, 2 interposed | — | — |
| evt278785 | deg 3, 2 stubs | — | — |
| evt437699 | deg 4, 2 stubs | deg 3, **1** stub | deg 3, 1 stub |
| evt314838 | deg 2, 2 stubs, 1 interposed | deg 1, **1** stub, vertex re-seated 0.60 cm | same |
| **total** | **11 stubs, 5 interposed prongs** | **8 stubs, 5 interposed** | **7 stubs, 5 interposed** |

op3 fires three times (`stub-absorb` on evt280972's 1.40 cm seg 7161 and
evt437699's 1.48 cm seg 11099, `stub-reseat` on evt314838's 0.72 cm seg 18093);
op1 and op2 never fire. Adding `mvga_satellite=3.0` buys one more stub.

**The verdict is mixed and should be read as such.** Existing machinery removes
**4 of 11 stubs and 0 of 5 interposed prongs**. It is a partial answer to mode 2
and no answer at all to mode 1.

### 5.2 Why op3 declines — the ledger

op3's accept conditions (`NeutrinoGraphAudit.cxx:391-416`) are: the anchor has
≥2 incident segments; `len < mvga_stub` (2 cm); the far vertex is not the main
vertex, is **degree 1**, and is not `kProtectedBreak`; and either corridor
overlap ≥0.7 with a sibling or ≤4 valid fits. The census evaluates these offline
and reproduces mvga's actual behaviour on all six events exactly. Over all 78
stubs in the 35 mode-2 events:

| count | why op3 does or does not reach this stub |
|---|---|
| 27 | eligible (subject to the overlap / point-degeneracy gate) |
| 19 | `len ≥ mvga_stub` 2.0 cm |
| 18 | not incident on the main vertex — needs `mvga_satellite > 0` |
| **11** | **far vertex degree ≠ 1 — op3 is TERMINAL-only; this is an INTERPOSED stub** |
| 3 | the main vertex has only 1 incident segment |

The 11 are the whole of mode 1a-VIA, and they are out of reach **by
construction**: an interposed stub's far end carries the long prong, so its
degree is ≥2 and the `degree(far) == 1` line rejects it. No setting of any
`mvga_*` number changes that. doc pr/84 §7 P5 guessed this ("mvga's existing ops
do not cover the *interior* case"); this is the line.

---

## 6. Answering the owner's question

> "I have many `examine*` are to improve this situation, but we still have some
> failures."

Three distinct reasons, in the order they bite:

1. **Timing.** The `examine_*` family runs inside `find_proto_vertex` and
   `improve_vertex`. The stubs are created *after* it, by
   `snap_main_vertex_to_kink` and the final `improve_vertex`. ES3 evaluated 24
   junctions across the six events and proposed zero merges, because at the time
   it looked the clutter was not there (§3.1).
2. **Scope.** `examine_vertices_4`'s length predicate passes on 8 of 11 stubs but
   it can never fire, because both candidate dying vertices are barred — one is
   the main vertex, the other has degree 1 (§3.3).
3. **A disabled pass.** The one audit positioned after the creating operations,
   `main_vertex_graph_audit`, is off in SBND production; and even on, its op3 is
   terminal-only and reaches 4 of 11 stubs and none of the interposed ones
   (§5).

---

## 7. Proposals

All default OFF per CLAUDE.md §1. **None is implemented.** Each is listed with
the gate it would need.

| id | proposal | scope | gate |
|---|---|---|---|
| **Q1** | `mvga_interposed` — drop op3's `degree(far) == 1` requirement for the specific shape *anchor = main vertex, stub < `mvga_stub`, far vertex degree ≥ 2, and the stub collinear with a prong at the far end*, absorbing the stub by merging the far vertex INTO the main vertex (`merge_vertex_into_another`, `NeutrinoPatternBase.cxx:2546`). This is the direct fix for mode 1a-VIA — 21 events, 11 stubs in the mode-2 census. Sharpens doc pr/84 P5, which named the interior case but not the blocking line. | changes the graph | full A/B + mover census; requires `main_vertex_graph_audit` to be on, so it cannot be validated alone |
| **Q2** | flip `main_vertex_graph_audit = true` for SBND, with `mvga_satellite` set. Zero new code. On this six-event sample it removes 4 of 11 stubs and moves one vertex 0.60 cm. **The 0.60 cm move is the whole risk**: it is a real vertex relocation on an event whose vertex was already correct to 0.01 cm. | changes the graph | the standard manifest A/B, plus a per-event verdict on every vertex that moves — doc pr/51 R3 left exactly this un-adjudicated |
| **Q3** | `vks_carry_prong` — in `snap_main_vertex_to_kink`, after `break_segment` succeeds and the new corner becomes the main vertex, if the residual piece to the old vertex is under a threshold, merge the old vertex into the new one so the prong follows the vertex. Attacks mode 1 at its source instead of cleaning up after it, and needs no new pass. **No cut is proposed here**: the observed scale is SNAP `arc` = 0.93 and 1.07 cm on the two firing events and interposed stubs of 0.29–2.69 cm across the 21 mode-1a-VIA events, but those are all **fit-space** distances while `break_segment` works in **wcpt** space. Per doc pr/84 §8.2, a radius must be measured in the space it will be applied in, from inside the code — this doc supplies the range, not the number. | changes the graph | the pr/50 manifest must not regress (172230, 131357, 342199, 360535, 469665, 57441), plus the 21 mode-1a-VIA events |
| **Q4** | raise `mvga_stub` above 2.0 cm. 19 of 78 stubs are declined on this line alone, several by centimetres (evt278785's 2.21, evt437699's 2.12). Cheapest possible change and the least principled — 2.0 cm was not fitted against this population. | one number | a sweep over the 35 mode-2 events with a mover census at each value |
| **Q5** | mode 1a-CUT (8 events) and 1b STRADDLE (4 events) are **not** addressed by any of the above. 6 of the 8 CUT prongs are in a different cluster, which is an upstream clustering question (the same one doc pr/84 §9 left open for evt284794). STRADDLE needs a break at the vertex, which is `segment_search_kink`'s decision, upstream of every pass in this doc. | — | out of scope here; recorded so the 32 is accounted for |

**Recommended order.** Q2 first, because it costs no code and its result is a
fact rather than a prediction — but note it buys 4 stubs, not the family. Q1 is
the fix that addresses the largest population and it depends on Q2. Q3 is the
more principled version of Q1 (prevent rather than repair) and is the one worth
building if only one is built. Q4 should not go first: raising a ceiling that was
never fitted to this population, on a pass that is currently off, changes an
untested thing twice.

---

## 8. Surfaced, deliberately not proposed (CLAUDE.md §5 rule 4)

**8.1 149 of 462 clicked vertices have degree 1 in the PR graph.** A third of
hand-scanned neutrino vertices are the end of a segment rather than a junction.
evt360535 is the extreme case in this doc — a degree-1 vertex at the tip of a
1.27 cm stub, with the real 3-prong junction 1.27 cm away. Whether this is a
representation convention (a shower stem is one segment; a single-prong ν
interaction is legitimate) or a defect is **not decided here**, and no proposal
in §7 depends on the answer. It is a much bigger number than anything else in
this doc and deserves its own round.

**8.2 The interaction with doc pr/83 is measured but not resolved.** pr/83
shipped `break_seg_orient` (SBND production **on** as of today) which changes
`break_segment` — the call `snap_main_vertex_to_kink` uses to create the
interposed stub. The 462-event census predates the flip; all 57 affected events
were re-run at HEAD and 1 of 56 changed (§2.4), so the two knobs address
different things and neither supersedes the other. pr/83, pr/84 §5 and this doc
are three views of
one owner complaint: pr/83 measures duplicate trajectories *inside* a segment,
pr/84 measures what stubs do to the particle hierarchy, and this doc measures
the graph topology at the vertex.

---

## 9. Open

- The HEAD re-measure covers the 57 events already known to be affected, so it
  confirms the population does not *shrink* under the `break_seg_orient` flip
  but cannot detect events the flip newly *creates*. That would need a full
  512-event arm at HEAD, which this round did not run.
- mvga's op3 `eval` probe lines are `SPDLOG_LOGGER_TRACE`, so at `debug` the
  decline reasons are not in the log. §5.2's ledger is computed offline from the
  dump geometry instead and reproduces the observed firings exactly, but it
  cannot see `kProtectedBreak`, which the dump does not carry. A trace-level
  rerun would close that gap.
- The 11 labels whose click is further than 1 cm from any PR-graph vertex are
  not analysed here. They are the events where the vertex is not a candidate at
  all — doc pr/78 §3's admission gap, measured there and not re-opened.
