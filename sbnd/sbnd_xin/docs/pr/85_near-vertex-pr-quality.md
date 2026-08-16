# doc pr/85 — near-vertex PR quality: the interposed stub

**Status: §§1–9 investigation (2026-08-15 morning); §10 implementation round
(same day) — Q1+Q2+Q4 SBND PRODUCTION ON (toolkit `bc0227cb` + `412b4e93`),
Q3 ships as code default OFF.** The §5 experiment uses `-A` overrides on
knobs that already existed.

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

---

## 10. Implementation round (2026-08-15) — Q1 + Q2 + Q3, Q4 as sweep

Owner instruction: implement §7's improvements, validate on nueCC48 + NCpi0 +
mcp1k, flip SBND production on gate pass (pre-authorized), before/after Bee
links for the six events and the top movers.  Scope chosen by the owner:
**Q1 + Q2 + Q3 together, Q4 as a sweep adopted only if cleanly positive.**

### 10.0 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# Stage A -- the six events, per-knob arms (baseline = work-pr85-b2)
export SBND_WCT_LOGLEVEL=debug PR_EXTRA_STAGES=pr_display PR_JOBS=3
SBND_VKS_CARRY_PRONG=1.5 \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr85i-q3 data 59685 280972 278785
SBND_MAIN_VERTEX_GRAPH_AUDIT=true SBND_MVGA_SATELLITE=3.0 SBND_MVGA_INTERPOSED=true \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr85i-q12 data 59685 280972 278785
# (repeat each with work-nuecc48-cb0805: 360535 437699; work-ncpi0-cb0805: 314838;
#  work-pr85i-all = both env sets together)

# the census, scored on any arm (PR85_DUMP_ROOT unset = the original 462-event census)
PR85_DUMP_ROOT=work-pr85i-all python3 pr85_near_vertex_census.py

# byte-identity gate between two arms (hash_archive member rollup, never paths)
python3 scripts/pr85_hash_gate.py work-ncpi0-pr83on work-ncpi0-pr85ioff
```

### 10.1 What was built

Three default-OFF knobs (C++ defaults; jsonnet key-suppression at every layer;
`doctest_clus_knob_defaults.cxx` pins; runner envs `SBND_MVGA_INTERPOSED`,
`SBND_MVGA_INTERPOSED_ANGLE`, `SBND_VKS_CARRY_PRONG`):

| knob | default | what it does |
|---|---|---|
| `mvga_interposed` (+ `mvga_interposed_angle`, 150°) | false | **Q1.** op3's `degree(far)==1` line (`NeutrinoGraphAudit.cxx`) becomes a classifier: an interposed stub at the **main-vertex anchor only**, under `mvga_stub`, far vertex not `kProtectedBreak`, and collinear within `mvga_interposed_angle` of a far prong, has ALL far prongs spliced through the stub's wcpts onto the main vertex; the stub is removed and the far vertex (degree 0) dropped.  Sentinel `mvga: op3 stub-interposed`, TRACE probe `mvga: op3 eval-interposed`. |
| `vks_carry_prong` (cm) | 0 = off | **Q3.** In `snap_main_vertex_to_kink` (production ON), after the break, when `best.arc` < threshold every arm at the OLD vertex is spliced through the residual's wcpts onto the new main vertex, the residual removed, the old vertex dropped — the interposed stub is never created.  `best.arc` **is wcpt-space** (arms are built from `sg->wcpts()`, `cum[]` arclength — §7 Q3's "fit-space" caveat was wrong and is corrected here), so the measured 0.93/1.07 cm arcs are directly the threshold scale.  Sentinel `snap_main_vertex_to_kink: CARRY`. |
| (no new knob) | — | **Q2.** flip `main_vertex_graph_audit=true` + `mvga_satellite=3.0`, gated on the §10.4 mover adjudication. |

Both edits share one splice implementation, `carry_prong_verify` /
`carry_prong_execute` (`PRSegmentFunctions.{h,cxx}`): SegmentPtr identity
preserved (fits/flags/particle_info survive; wcpts respliced; refit by the
pass's existing trailing `do_multi_tracking`), all-or-nothing pre-verification
(0.01 cm endpoint wcpt matches on every chain, far vertex valid and distinct,
`find_segment(far, anchor) == nullptr` — the examine_structure B.7 setS-alias
landmine), decline leaves the graph exactly as production.
`merge_vertex_into_another` was rejected for this job: it deletes and re-routes
prongs via an **unchecked** `create_segment_from_vertices` (silent prong drop)
and loses fits/flags/particle_info.

### 10.2 Stage A — the six events

Sentinels: CARRY fired on exactly the two SNAP events (59685 arc 0.93 cm,
1 arm; 280972 arc 1.07 cm, 2 arms); `stub-interposed` fired on 59685
(len 1.18 cm, vf_deg 2, far_angle 175.2°) in the Q1+Q2 arm.  Census tuples per
arm (stubs / interposed prongs at the click):

| event | baseline | Q3 only | Q1+Q2 | combined |
|---|---|---|---|---|
| evt59685 | 1 / 1 | **0 / 0** | **0 / 0** | **0 / 0** |
| evt280972 | 3 / 1 | **0 / 0** | 1 / 1 | **0 / 0** |
| evt278785 | 2 / 0 | 2 / 0 | 2 / 0 | 2 / 0 |
| evt314838 | 2 / 1 | 2 / 1 | 1 / 1 | 1 / 1 |
| evt360535 | 1 / 2 | 1 / 2 | 1 / 2 | 1 / 2 |
| evt437699 | 2 / 0 | 2 / 0 | 1 / 0 | 1 / 0 |
| **total** | **11 / 5** | 6 / 3 | 6 / 4 | **5 / 3** |

Three observations worth the table:

- **Q3 at source beats repair-after.** On 280972 the carry does not just
  remove the residual — the two sibling stubs (0.78, 1.40 cm) never form
  either: 3 stubs → 0.  The Q1+Q2 arm on the same event absorbs two stubs and
  still leaves the interposed one (7158, re-fit to 0.23 cm).
- **Every survivor in the combined arm is a named class**: 278785's pair
  (2.25/2.21 cm) sits above the `mvga_stub` 2.0 cm ceiling (§10.5's sweep);
  314838's stub meets its far prong at **82.5°** (trace probe,
  `work-pr85i-tr314838`) — a genuine corner the 150° collinearity gate is
  *supposed* to decline, absorbing it would fabricate a straight-through
  connection; 360535 is the degree-1 main-vertex anchor documented at §10.1
  as out of scope (`incident.size() < 2`).
- The reseat on 314838 (0.72 cm move, the §7 Q2 risk case) reproduces
  exactly as in §5.1.

### 10.3 Gates (labels; every PASS re-checkable)

- **Gate 0**: `wcbuild` rc=0; freshness `libWireCellClus.so` 16:44 > last
  edit 16:38; `wcdoctest-clus` 2069/2069; compiled production config with
  knobs off **byte-identical** to the pre-change HEAD render
  (scratch `pr85-cfg-base.json` == `pr85-cfg-off.json`); knobs on, all five
  keys appear with their values (`pr85-cfg-on.json`).
- **Gate 1** (knob-off byte identity vs the pr/83 production reference arms,
  `scripts/pr85_hash_gate.py`, mabc-pr.zip + pctree per event):
  - NCpi0: `work-ncpi0-pr83on` vs `work-ncpi0-pr85ioff` — **PASS 38/38**.
  - nueCC48: `work-nuecc48-pr83on` vs `work-nuecc48-pr85ioff` — **PASS 94/94**.
  - mcp1k: (§10.3-TBD)
  - mcp1k: `work-mcp1k-pr83on` vs `work-mcp1k-pr85ioff` — **PASS 890/890**.
  - **Total 1022/1022 archives byte-identical with every knob off**, proving
    `vks_carry_prong=0` inert inside the production-ON snap pass and the op3
    accept-block restructure control-flow-identical.

### 10.4 Stage B and the mover adjudication — the round's pivot

The first full-knob 57-event arm (`work-pr85i-all57`: Q1+Q2+Q3, reseat at its
150° default) cleaned the topology (stubs 95→45 summed over matched events)
but the **mover adjudication against the owner's hand labels** — the exact
check doc pr/51 R3 left undone — found two >1 cm regressions, and per-knob
attribution arms (`work-pr85i-q3x` / `-q12x`) split them cleanly:

| event | base→arm click-to-main | attributed to |
|---|---|---|
| evt280017 | 0.00 → 1.64 cm | **op3 stub-reseat** (`reseat_dis=1.69`) — pre-existing Q2 machinery |
| evt174114 | 0.00 → 1.51 cm | **Q3 CARRY** (arc 0.55; post-carry refit + improve_vertex drift off the corner, main lands 1.56 cm from the SNAP corner it sat 0.06 cm from in baseline) |

The full CARRY ledger (10 firings on the 57): 4 better / 6 worse by
click-to-main, net **zero** at the 1 cm correctness bar (gains 316025
2.53→0.40, loses 174114), while the same snap residuals are cleaned by Q1's
interposed absorb with ≤0.25 cm vertex motion (mvga runs AFTER the final
improve_vertex, so nothing re-polishes the vertex off its point; CARRY runs
before it, and the local optimizer wanders on the carried graph).  And both
label-set reseat firings moved the vertex OFF the owner's click (280017
1.64 cm, 314838 0.60 cm — §5.1's "0.60 cm move is the whole risk" is hereby
adjudicated: **adverse**).

**Operating point chosen (the owner's authorized iteration):**
`main_vertex_graph_audit=true, mvga_satellite=3.0, mvga_interposed=true,
mvga_reseat_angle=0` (absorb-only op3) — **Q3 ships as code, default OFF**
(its ledger above is the reason; a future round could revisit with a turn
floor).  New runner envs `SBND_MVGA_RESEAT_ANGLE`, `SBND_MVGA_STUB` added for
this and the §10.5 sweep.

**Candidate-config Stage B** (`work-pr85i-cand57`, 57/57 rc=0), scored at the
owner's clicks vs the HEAD baseline `work-pr85-head57`:

- topology, summed over matched events: **stubs 95→45, interposed prongs
  26→17, CUT 9→9, STRADDLE 3→3**; 36 events improve, zero worsen except
  the one flagged below.
- movers: 12 events with main-vertex motion >0.1 cm.  **Every mover ≥0.5 cm
  moves TOWARD the click** (48367 0.85→0.00, 400504 1.30→0.00, 394642
  1.49→0.66, 411460 3.05→1.88, 316025 2.53→2.09); the worst adverse move is
  **0.27 cm** (click-resolution scale).  Both §10.4-table regressions are
  gone (174114: 0.25 cm; 280017: 0.00 — its 1.69 cm stub is now absorbed
  in place, strictly better than the reseat).
- three events leave the census's 1 cm click-match (284637, 409546, 411460)
  — in all three the main vertex itself is unmoved (0.00/0.04 cm) or moves
  toward the click (411460, 1.20 cm closer); what vanished is a *clutter
  vertex near the click*, which is the fix working.
- the one adverse tuple: evt66272 gains a 1a-CUT because **op1** (pre-existing
  dup-corridor merge) removed a genuine 100%-overlap duplicate (3.49 cm,
  1.5e5 dQ vs the 11.1 cm / 9.1e5 survivor) and the census now sees the
  survivor prong with no path to the clicked vertex.  Correct removal,
  connectivity-presentation cost; main vertex untouched; recorded, not fixed
  here.

### 10.5 Q4 sweep — `mvga_stub` fitted against this population, 2.5 adopted

§7 Q4's objection was that 2.0 cm "was not fitted against this population";
this sweep is the fit.  On top of the §10.4 candidate config, over the same
57 events (`work-pr85i-s23`, `work-pr85i-s25`, 57/57 rc=0 each):

| | cand (2.0) | 2.3 | 2.5 |
|---|---|---|---|
| stubs (matched sum) | 45 | 38 | **35** |
| interposed / cut / straddle | 17 / 9 / 3 | same | same |
| adverse movers | — | none | none |

The only mover in either arm is evt138009, 0.43 cm **toward** the click
(0.75→0.54).  2.5's three extra removals over 2.3 (59247 ×2, 61461 ×1) cost
nothing; evt268067 — doc pr/51's own charge-less-bridge event — gets its
2.29 cm interposed stub absorbed (carried 2 prongs, 168.0°) with the main
vertex byte-identical in position across all four arms.  278785's 2.21/2.25 cm
pair is *still* declined at 2.5 — by the overlap/collinearity gates, not the
ceiling — so raising further only adds risk: **2.5 adopted, 3.0 not pursued.**

### 10.6 Stage C round 1 — the marginal-overlap class, two more knob values

Full-sample knob-on arms at the §10.5 config (`work-*-pr85ion`, 1000+48+19
events, rc=0 everywhere; footprint 139/1000 + 12/47 + 7/19).  nueCC48 nue
ledger: **two signal events recovered, none lost** — evt268067 (pr/51's own
poster event) goes nue −15.0 → **+1.28** with its cosmict flag cleared 1→0,
evt38856 goes −2.8 → **+4.3**; the three other nue changes stay on their side
of zero.  Zero >1 cm vertex movers on nueCC48/NCpi0.  Wall/RSS medians
unchanged (22–23 s, 1.47 GB).

mcp1k, however, produced four adverse >1 cm movers — and they share one
signature, visible in the sentinel line itself:

```
mvga: op3 stub-absorb ... nfit=4 overlap=0.75    (172090 0.00->1.68, 168614 0.00->1.53,
                                                  286353 2.00->3.06, 285971 0.00->0.96)
```

Across ALL 143 Stage-C absorbs: 114 fired at overlap=1.00 (every adjudicated
one clean), 21 at exactly (nfit=4, overlap=0.75) — the class containing every
adverse mover.  A 4-point stub at 3/4 overlap is not a pure duplicate; it
carries real charge, and deleting it shifts the op4 refit under the vertex.
These enter through the **degeneracy gate** (`nfit <= mvga_stub_pts` = 4), so
no overlap threshold alone can stop them.  Two existing-knob values close
exactly this class and nothing else observed:

- `mvga_dup_frac = 0.8` — the overlap gate now rejects 0.75;
- `mvga_stub_pts = 3` — the degeneracy bypass no longer admits 4-point stubs.

Verification arm (`work-pr85i-v2chk`, the 4 adverse + 4 clean-firing events):
all three pure-adverse movers now 0.00 (no firing); 286353 drops to a 0.46 cm
move (its 0.75-absorb is declined, a legitimate 151.0° interposed absorb
remains); every clean firing survives — 280017's overlap=1.00 absorb, 48367's
win (0.85→**0.00** on the click), 66272 and 174114 unchanged (174114's stub
now removed by op1 at overlap=1.00: with `stub_pts=3` op1 no longer skips
4-point segments — same removal, different op).

Final operating point for the flip:
```
main_vertex_graph_audit = true
mvga_satellite = 3.0
mvga_interposed = true
mvga_reseat_angle = 0     // absorb-only; both label-set re-seats were adverse (§10.4)
mvga_stub = 2.5           // fitted by the §10.5 sweep
mvga_dup_frac = 0.8       // §10.6: marginal-overlap absorbs are the adverse-mover class
mvga_stub_pts = 3         // §10.6: closes the degeneracy bypass for 4-point stubs
```
(`vks_carry_prong` stays at its C++ default 0 = OFF.)

### 10.7 Stage C round 2 (flip config) — PASS

Fresh full-sample arms `work-*-pr85ion2` (1000 + 48 + 19 events, rc=0
everywhere) at the §10.6 final operating point, vs the same `work-*-pr85ioff`
baselines:

- **Movers**: the four §10.6 adverse movers are gone (172090 / 168614 /
  285971 move 0.00; 286353 down to 0.46 cm).  The only remaining >1 cm
  movers with calib: 400504 (1.30 → **0.00** on the click), 348515 (an event
  whose vertex is 55 cm off in both arms — noise on a lost event), 180801
  (77 cm off in both arms, moves 1.0 cm *toward* the click).
- **57-label census** (merged over the three on2 arms): stubs **95→42**,
  interposed prongs **26→17**, CUT 9→9, STRADDLE 3→3; the single worse
  tuple is the adjudicated §10.4 evt66272 op1 case; the three click-unmatch
  events are the adjudicated benign clutter-removals.
- **nueCC48**: the §10.6 recoveries persist (268067 nue −15.0 → +1.28 with
  cosmict cleared; 38856 −2.8 → +4.3); no nue-positive event lost.
- **Footprint**: mcp1k 126/1000, nueCC48 21/48, NCpi0 6/19 events differ
  (mabc member hash).
- **Tagger-flip ledger** (the §4-plan hand-check items, all in the Bee set):
  cosmict 1→0 recoveries on 405234 + 268067; cosmict 0→1 on 169774 (vertex
  already 11.8 cm off in both arms), 59753 / 62561 (no hand label, clean
  overlap=1.00 absorbs), and **319611 — the one flagged concern: its vertex
  sits exactly on the owner's click in both arms, an op1 dup-merge
  (overlap=1.00, reconnects=1) removed a genuine duplicate, and the cosmic
  tagger then flipped.  Bee idx 13; owner hand-check requested.**
- **Runtime/RSS**: mcp1k medians 15 s / 1.15 GB (off) vs 16 s / 1.15 GB
  (on2), p90 identical; small samples unchanged.
- **Post-flip smoke**: after the production flip commit, a bare six-event
  rerun (`work-pr85i-postflip`, no env overrides) is byte-identical to the
  flip-config arms — 12/12 archives.

### 10.8 Ship record

- toolkit `apply-pointcloud`: `bc0227cb` (knobs, DEFAULT OFF) +
  `412b4e93` (SBND production flip: `main_vertex_graph_audit=true,
  mvga_satellite=3.0, mvga_interposed=true, mvga_reseat_angle=0,
  mvga_stub=2.5, mvga_dup_frac=0.8, mvga_stub_pts=3`;
  `vks_carry_prong` stays OFF).
- wcp-porting-img `main`: this doc §10, `pr85_near_vertex_census.py`
  `PR85_DUMP_ROOT` extension, `scripts/pr85_hash_gate.py`, runner envs
  (`SBND_MVGA_INTERPOSED`, `SBND_MVGA_INTERPOSED_ANGLE`,
  `SBND_VKS_CARRY_PRONG`, `SBND_MVGA_RESEAT_ANGLE`, `SBND_MVGA_STUB`,
  `SBND_MVGA_DUP_FRAC`, `SBND_MVGA_STUB_PTS`), `docs/pr/85_bee.index.txt`.
- Bee sets (18 events, index in `docs/pr/85_bee.index.txt`):
  - before (production baseline):
    https://www.phy.bnl.gov/twister/bee/set/bae7f8e2-5a56-439c-8d62-ea3c3b7145ba/event/list/
  - after (flip config):
    https://www.phy.bnl.gov/twister/bee/set/e1424225-d475-4e2b-a63d-ffa5e6e128b7/event/list/
- Arms kept for re-checking: `work-pr85i-{q3,q12,all,q3x,q12x,v2chk,tr314838}`
  (Stage A / attribution), `work-pr85i-{all57,cand57,s23,s25}` (Stage B +
  sweep), `work-*-pr85ioff` / `work-*-pr85ion` / `work-*-pr85ion2`
  (Gate 1 + Stage C), `work-pr85i-postflip` (smoke).
  `work-pr85i-stub23` is an aborted partial launch — ignore it.

### 10.9 Residuals and follow-ups

- **Q3 `vks_carry_prong` ships OFF** with its §10.4 ledger; a future round
  could revisit with a turn-strength floor (the one >1 cm adverse CARRY,
  174114, had the weakest turn of the ten firings: 33.6°).
- 278785's 2.21/2.25 cm stub chain survives every configuration — declined
  by the overlap/collinearity gates, not the ceiling.
- 314838 keeps one 82.5° stub: a genuine corner the collinearity gate is
  designed to decline.
- evt360535's degree-1 anchor class (§8.1's 149 events) remains its own
  round.
- evt66272 / evt319611: op1 side-effects flagged for owner hand-check
  (Bee idx 16 / 13).
- Mode 1a-CUT and 1b STRADDLE remain out of scope (§7 Q5).

---

## 11. Addendum (2026-08-15) — the 2-D overlay, and two blind spots it found

The owner asked for the near-vertex pictures to be drawn **over the 2-D
measurement** rather than on the reconstruction alone. Doing so exposed two
places where the §2 census under-reports, both of them properties of the
census's own constants rather than of the reconstruction.

### 11.0 Repro

```bash
cd sbnd_xin
# one PNG per event: U/V/W, wire index vs slice, for the APA holding the vertex
python3 pr85_panels2d.py /home/xqian/tmp/pr85-2d 268067 349945 38856 38856:12106
```

`pr85_panels2d.py` needs no wire geometry: `proj[]` in the calib dump *is* the
measurement (`wire`, `slice`, `charge` per apa/plane) and every fitted point
already carries `pu/pv/pw` and `pt`. Coordinates and the growing-window recipe
are taken from `pr_display_viewer.py:2370-2540` verbatim so a picture here and
port 5017 can be discussed in the same words. Three conversions are load-bearing
and each one, if wrong, produces the *same* misleading picture — a polyline
offset from the charge, read as "the PR does not follow real charge":

- `points[].pt` is in **ticks**, `proj[].slice` is in **slices**; divide by
  `meta.nticks_per_slice` keyed on `(apa, face)` (SBND: 4).
- points with `apa < 0` are dropped, never defaulted to APA 0 (doc pr/3).
- dead bands are in slice units — `s0/s1`, not `t0/t1`.

**Alignment gate, run before reading anything off a 2 cm stub:** render one long
unambiguous track at full extent (`HALF=400`) and confirm the polyline rides a
charge ridge along its whole length in all three planes. A tick/slice or APA
error is glaring on a 43 cm track and invisible-but-fatal on a 2.3 cm stub.
`ALIGNCHECK-evt38856` (segment 12030, 43.5 cm) passes in U, V and W.

`proj[]` is deliberately **not** filtered by `cluster_id`: a mode-1 CUT prong
lives in a different cluster from the click (evt407280: click in cluster 51,
the 128.8 cm prong in cluster 16), so filtering would erase exactly the evidence
the overlay exists to show.

### 11.1 Blind spot A — the 3–10 cm band between `STUB` and `LONG`

§2 counts a segment as clutter when it is shorter than `STUB = 3.0` cm, and as a
prong when it is longer than `LONG = 10.0` cm. A non-incident segment **between**
those is counted as neither. The overlay made this visible immediately on
evt349945, whose clicked vertex 18011 has degree 2 but **four** segments with
charge within 3 cm:

| seg | length | charge→click | incident |
|---|---|---|---|
| 18017 | 11.78 cm | 0.00 | yes |
| 18016 | 4.44 cm | 0.00 | yes |
| **18013** | **8.27 cm** | 0.38 | **no** |
| **18003** | **3.11 cm** | 1.10 | **no** |

Measured over the same 462 events: **20 have at least one non-incident segment
in the 3–10 cm band touching the click, 3 have two or more.** Against §2.2's 32 +
35 that is a ~4 % under-count, not a reinterpretation — the two modes stand — but
the §2 numbers should be read as a **lower bound**, and any future re-tuning of
`STUB`/`LONG` must re-derive them rather than reuse them.

Distribution of segments whose charge comes within 3 cm of the click, over the
462 (this is the raw crowding number, independent of any length cut):

| segments touching the click | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| events | 131 | 170 | 103 | 32 | 15 | 10 | 1 |

### 11.2 Blind spot B — scoring at the click cannot see clutter that mis-picked the vertex

§2.1 scores at the owner's click on purpose, so that ugly topology is not
confounded with a wrongly chosen vertex. The cost of that choice is now
measured. **evt38856** is clean at the click — vertex 12004, degree 2, two long
prongs (12013 at 25.6 cm and 12030 at 43.5 cm), no stubs, nothing detached — and
the census therefore records it as a non-event. But the reconstruction put
`main_vertex` at **12106, 30.1 cm away**, and *that* vertex carries seven
segments within 3 cm, four of them not incident:

| seg | length | charge→vertex | incident |
|---|---|---|---|
| 12082 | 3.48 cm | 0.00 | yes |
| 12083 | 1.94 cm | 0.00 | yes |
| 12084 | 7.33 cm | 0.00 | yes |
| 12008 | 15.37 cm | 1.79 | no |
| 12018 | 12.79 cm | 1.79 | no |
| 12017 | 4.31 cm | 2.55 | no |
| 12036 | 2.98 cm | 2.99 | no |

This is the §1 symptom in full, sitting at the vertex the pipeline actually
chose. The census is structurally blind to it: `b1 = 30.1` cm puts the event in
pr/80's mis-picked-vertex class, and pr/85 never looks there.

So §2.3's "the defect is not an artefact of a mis-picked vertex" remains true as
stated — the base rates x1.00 / x1.08 are measured on the 318 click==reco events
and are unaffected — but the converse was never established and is false: **there
is near-vertex clutter that only appears at mis-picked vertices.** Whether it is
a cause of the mis-pick or a consequence of it is not answered here.

### 11.3 Follow-up

A round that scores at **both** the click and the reco `main_vertex`, over the
149 degree-1 events of §8.1 and the `b1 > 1 cm` events excluded here, would close
both blind spots at once. Not started; no knob is proposed on this evidence.
