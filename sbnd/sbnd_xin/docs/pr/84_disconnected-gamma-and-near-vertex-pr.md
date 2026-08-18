# doc pr/84 — the "disconnected gamma" bug, and near-vertex over-segmentation

**Status: investigation only. Nothing is implemented, no knob is added, no config
is touched.** The owner asked for understanding and proposals; §7 lists the
proposals and the gate each would need. All numbers are from the deployed
(`min_accept = 10`, doc pr/79) arms of the prod0813 sample, 493 events.

## 0. Repro

```
cd sbnd_xin
python3 pr84_pseudo_parent_census.py > /home/xqian/tmp/pr84.txt 2>&1; echo rc=$?
```

Every table below is a section of that output. The per-event evidence:

```
cd sbnd_xin
# the particle-flow tree the owner is looking at
unzip -p work-mcp1k-ma10/pr_evt283713/mabc-pr.zip data/0/0-mc.json | python3 -m json.tool

# the sentinels.  -a is REQUIRED: WCT logs carry invalid UTF-8, so plain grep
# treats them as binary and prints nothing at all.
grep -a "pr54 keep-isolated\|pr54 isolated-residual" work-mcp1k-ma10/pr_evt65289/wct_pr_evt65289.log
grep -a "conn3_unreachable"                          work-mcp1k-ma10/pr_evt65289/wct_pr_evt65289.log
grep -a "snap_main_vertex_to_kink\|endpoint mismatch" work-mcp1k-ma10/pr_evt283713/wct_pr_evt283713.log
```

Deployed dumps are `work-{mcp1k,ncpi0,nuecc48}-ma10/pr_evt<N>/calib-pr-evt<N>.json`;
Bee archives are `mabc-pr.zip` in the same directory. Nothing under `work-*/` was
written, regenerated or deleted for this study.

---

## 1. Symptom

The owner's four events, in their words:

> "283713, weird PR, where there is a 209 MeV gamma inside the connected main
> cluster??? 287517, Vertex not very good. 284794, why not connected? 65289 …
> disconnected gamma @ 209 MeV"

and their own split of the two families:

> "1. near the neutrino vertex, the pattern recognition are not great, small track
> back and force, not simple enough. 2. some of these events have some
> disconnected gamma, which are weird. These gamma have electron which are
> connected to the main cluster where the neutrino vertex is, but why they are
> identified as gamma, which should be disconnected. This seems to be a bug to
> fix, while the earlier problem can be an improvement."

That reading is correct, and the mechanism is sharper than the phrasing suggests.
The four Bee trees:

**evt65289** — two gammas at the vertex, one of them 202 MeV, flying 0.3 cm:

```
gamma  202 MeV   start=(-81.6,66.3,309.8)  end=(-81.3,66.4,309.8)      <-- 0.31 cm
  └── e-  202 MeV   start=(-81.3,66.4,309.8)  end=(-74.0,71.9,301.3)
gamma  109 MeV   start=(-81.6,66.3,309.8)  end=(-83.7,65.3,310.9)      <-- 2.60 cm
  └── e-  109 MeV   start=(-83.7,65.3,310.9)  end=(-95.0,58.8,316.1)
proton 188 MeV / proton 250 MeV / mu- 200 MeV   [directly off the vertex]
```

**evt283713** — the 209 MeV gamma, flying 0.47 cm, and a *neutron* carrying the
567 MeV muon:

```
gamma  209 MeV   start=(-83.98,67.35,35.62)  end=(-84.15,67.62,35.27)  <-- 0.47 cm
  └── e-  209 MeV   start=(-84.15,67.62,35.27)  end=(-91.27,89.53,25.76)
neutron 567 MeV  start=(-83.98,67.35,35.62)  end=(-83.85,67.48,35.90)  <-- 0.34 cm
  └── mu- 567 MeV  start=(-83.85,67.48,35.90)  end=(-98.16,79.69,265.25)
```

**evt284794** — "why not connected?": the two protons at the vertex are carried by
invisible neutrons that flew 3 cm:

```
neutron  4 MeV   start=(-47.2,-20.5,280.5)  end=(-44.2,-21.8,280.1)    <-- 3.25 cm
  └── proton  4 MeV
neutron 83 MeV   start=(-47.2,-20.5,280.5)  end=(-46.5,-24.0,281.4)    <-- 3.60 cm
  └── proton 83 MeV
mu- 639 MeV      [directly off the vertex]
```

**evt287517** — "vertex not very good": the 343 MeV muon is a **daughter of a
6 MeV proton** that is 1.05 cm long:

```
proton  6 MeV    start=(71.4,140.0,356.3)  end=(70.6,140.5,355.9)      <-- 1.05 cm
  └── mu- 343 MeV  start=(70.6,140.5,355.9)  end=(37.8,103.1,484.6)    <-- 138 cm
proton 86 MeV  (6.05 cm)  /  mu- 18 MeV  (3.25 cm)
```

---

## 2. Root cause

### 2.1 "gamma" is not a particle-ID statement

No shower is ever assigned PDG 22. A shower's PDG is its start segment's
(`PRShower.cxx:1010`, `:1122`); the only `set_particle_type` call in the package
writes 13, for long muons. PDG 22 appears exclusively on a **synthetic parent node**
that the Bee particle-flow writer splices in above a shower:

```cpp
// MultiAlgBlobClustering.cxx:1433
bool direct = (conn_type == 1);
```
```cpp
// MultiAlgBlobClustering.cxx:1779-1782, append_pseudo_shower
const int pdg = (std::abs(sh->get_particle_type()) == 11 ||
                 std::abs(sh->get_particle_type()) == 22) ? 22 : 2112;
```

So a `gamma` node in the tree means exactly one thing: **the reconstruction could
not walk from the main vertex to this shower through the PR graph.** EM showers get
a gamma, everything else gets a neutron — which is why evt283713's muon and
evt284794's protons are carried by neutrons. It is the same rule, not a second bug.

`start_connection_type` semantics (`PRShower.h:59`, `PrDisplayDump.cxx:528-530`):
1 = direct, 2 = indirect, 3 = gap/association, 4 = not clearly connected (dropped
from the tree entirely). Only type 1 renders as a real daughter.

### 2.2 The main cluster's own graph is disconnected

A cluster is one contiguous lump of charge, so its segment graph ought to be
connected. In **29 of 493 events (5.9%)** it is not, and both of the owner's gamma
events are that case.

**evt65289**, cluster 19 — two components:

| component | vertices | segments | contains the main vertex |
|---|---|---|---|
| A | 19007 (deg 3), 19006, 19008, 19012 | 19010 (23 cm p), 19011 (38 cm p), 19015 (76 cm µ) | **yes** |
| B | 19009 (deg 2), 19010, 19011 | 19012 (14 cm e, 109 MeV), 19013 (16 cm e, 203 MeV) | no |

The two components' nearest vertices, 19007 and 19009, are **2.60 cm apart**.

**evt283713**, cluster 17 — two components, and this time the main vertex is on the
*small* one:

| component | vertices | segments | contains the main vertex |
|---|---|---|---|
| A | 17000, 17001, 17003 (deg 2), 17005 (deg 3), 17006 | 17006 (253 cm), 17007 (246 cm µ), 17008, 17009 (26 cm e, 209 MeV) | no |
| B | **17089 (deg 1)**, 17004 | 17063 (25 cm proton) | **yes** |

Main vertex 17089 is at (-83.98, 67.35, 35.62); the true three-prong junction is
17003 at (-84.15, 67.62, 35.27) — **0.472 cm away**. Everything except the proton is
therefore unreachable, and gets a synthetic parent.

### 2.3 Four upstream causes, one rendering rule

The four events reach the same output through four *different* upstream failures.
This is the central finding: fixing any one of them fixes one event.

| # | cause | event | evidence |
|---|---|---|---|
| **D1** | `find_other_segments` keeps a residual as — the source comment's own words at `NeutrinoOtherSegments.cxx:838` — "a disconnected piece of this cluster's graph". Knob `other_seg_keep_isolated`, **SBND PRODUCTION ON** (doc pr/54, `wct-pr-perevt.jsonnet:1553`). | 65289 | `pr54 keep-isolated: cluster 19 n_points=32 length=15.74 cm v1=(-83.4,65.1,310.4) v2=(-95.3,58.4,315.7)` — segment 19012 exactly |
| **D2** | `snap_main_vertex_to_kink` moves the vertex 4 cm; the rebuilt proton branch is then attached to a **new** vertex at the *old* position instead of to the existing junction. The pr/30 P8 endpoint-mismatch sentinel fires twice at `tol=0.300 strict=false` and the code proceeds. | 283713 | `SNAP cluster 17 old=(-83.98,67.35,35.62) new=(-82.73,63.89,37.12) turn=30.4 deg`, then two `add_segment: vertex/segment endpoint mismatch (doc pr/30 P8)` lines whose `front=(-83.982,67.353,35.620)` is where 17089 ends up |
| **D3** | the prongs at the vertex are in **separate clusters** (44 and 45, at 3.25 and 3.60 cm) and were never fused, so they can only ever be an association | 284794 | dump: `seg 44008` cluster 44, `seg 45009` cluster 45, main cluster 12 |
| **D4** | near-vertex over-segmentation: a 1.05 cm segment sits between the muon's real endpoint (13002) and the chosen main vertex (13015) | 287517 | dump: `seg 13007 len=1.05 pid=2212 13002->13015` |

Downstream of all four, `shower_conn3_unreachable` (**SBND PRODUCTION ON**,
`wct-pr-perevt.jsonnet:1228`, doc pr/74 round 2 K5) promotes graph-unreachable
main-cluster segments to visible showers at connection type 3, logging the anchor
distance as it goes:

```
evt283713  pr74 conn3_unreachable: promote gidx=7  len 245.6cm conn=3 anchor_dis 0.3cm
evt283713  pr74 conn3_unreachable: promote gidx=6  len 252.8cm conn=4 anchor_dis 230.4cm
evt65289   pr74 conn3_unreachable: promote gidx=13 len 15.6cm  conn=3 anchor_dis 0.3cm
evt65289   pr74 conn3_unreachable: promote gidx=12 len 14.4cm  conn=3 anchor_dis 2.6cm
```

**An anchor distance of 0.3 cm is the whole bug in one number.** The pass computes
exactly the quantity that shows the piece is touching the vertex, and then labels it
an association anyway. The knob is doing its job — it exists to make these segments
visible at all, and before it they were dropped — but it has no rung between
"invisible" and "carried by a neutral".

The SBND config already knows the two knobs are coupled;
`wct-pr-perevt.jsonnet:1578` describes this pass as serving "the disconnected
components `other_seg_keep_isolated` above creates".

---

## 3. Why it hid

**A pseudo-gamma is indistinguishable, in the tree, from a correct one.** A π⁰
renders as `pi0 → gamma → e-` *by design*, and its gammas legitimately sit at the
decay vertex with zero rendered length. Of the 39 zero-length pseudo-parents in the
sample, **23 are π⁰ daughters and correct**. A reader scanning trees sees gammas
everywhere and has no way to tell which are structural.

evt283713 makes this concrete: the event has a genuine reconstructed π⁰
(`kine_pio_flag=1`) whose two daughters render as `gamma 301 MeV` and
`gamma 27 MeV` under a `pi0 124 MeV` node — correct — *next to* the spurious
`gamma 209 MeV`. Same node type, same shape, different meaning.

---

## 4. The measurement that kills the obvious fix

The obvious remedy is to suppress pseudo-parents whose rendered length is tiny. **It
is wrong**, and this is the load-bearing result of the study.

Two populations hide under "short rendered length". The rendered length is
*derived from the anchor* — conn-2/3 sets the shower's start point from its start
vertex (`PRShower.cxx:1140`) — so when the PF writer passes the shower's own start
vertex as the connection vertex (the `start_vtx not in BFS tree` fallback,
`MultiAlgBlobClustering.cxx:1552-1563`), the node is drawn from a point to itself
and collapses to zero **regardless of how far away the shower really is**.

Measuring instead the shower's nearest actual **charge** to the neutrino vertex
(KE ≥ 20 MeV):

| population | n | min | median | max | in main cluster |
|---|---|---|---|---|---|
| rendered length **== 0.00** | 15 | 4.91 | **38.77** | 172.64 | **0 / 15** |
| rendered length 0–3 cm | 24 | 0.00 | 4.46 | 163.71 | 5 / 24 |
| rendered length ≥ 3 cm (control) | 331 | 0.00 | 43.84 | 365.81 | 5 / 331 |

*(distances in cm, from the shower's start segment's closest fitted point to the
reconstructed neutrino vertex)*

This table counts **dump shower rows**; §3's 39/23 counts **Bee tree nodes**. They
are different objects — one shower can render as more than one node, and π⁰
daughters are grouped differently — so the two are not meant to add up.

The zero-length class is the **opposite** defect: 15 genuinely remote associations,
median 38.8 cm away, **none of them in the main cluster**, whose synthetic parent
fails to draw the gap it exists to represent. Suppressing short parents would hide
15 real associations while fixing the owner's events — a net loss.

### The selector that does work

**Main-cluster membership + charge within 3 cm of the neutrino vertex** isolates the
class exactly: **7 showers in 5 events of 493**.

| event | shower | KE (MeV) | conn | charge → ν (cm) | length (cm) |
|---|---|---|---|---|---|
| evt283713 | 17007 | 567.8 | 3 | 0.34 | 245.6 |
| evt283713 | 17009 | **209.9** | 3 | 0.47 | 25.7 |
| evt65289 | 19013 | **202.9** | 3 | 0.31 | 15.6 |
| evt347129 | 53021 | 156.7 | 3 | 0.00 | 41.4 |
| evt65289 | 19012 | 109.5 | 3 | 2.60 | 14.4 |
| evt169626 | 22024 | 107.9 | 3 | 0.00 | 30.5 |
| evt174752 | 48010 | 22.0 | 3 | 0.00 | 7.7 |

The owner found 2 of these 5 events unaided. **evt347129, evt169626 and evt174752
are predictions of this study that nobody has looked at** — they are the hand-check
that would confirm or refute the selector before any code is written. That check
costs three Bee loads and is the cheapest next step in this document.

Note the class is *small*: 5 events in 493 (1.0%). It is worth fixing because each
instance corrupts the event's particle hierarchy and its energy accounting
(conn ≥ 3 changes `mc_included`), not because it is common.

---

## 5. The improvement family, sized

The owner's first complaint — "small track back and forth, not simple enough" —
shows up in the particle tree as a **hierarchy inversion**: a sub-2 cm segment
wedged near the vertex does not merely add a node, it becomes the *parent* of the
long prong behind it.

Counting real-particle nodes under 2 cm that parent a child over 10 cm and more
than 5× longer than themselves: **29 inversions in 21 of 491 events (4.3%)**.
evt287517 is one of them (`proton 6 MeV`, 1.05 cm → `mu- 343 MeV`, 138 cm), ranked
18th by stub length. The worst are shorter still:

```
evt283040   pi+  0 MeV     0.00 cm  ->  mu-  188 MeV     67.0 cm
evt394796   pi+  174 MeV   0.04 cm  ->  mu-  350 MeV    139.7 cm
evt314838   proton  40 MeV 0.30 cm  ->  proton  250 MeV  37.2 cm
evt353223   e-  143 MeV    0.35 cm  ->  mu-  130 MeV     40.0 cm
evt277276   pi+  11 MeV    0.57 cm  ->  pi+  339 MeV    128.6 cm
```

Three cross-links worth recording:

- **evt59335** appears here (`pi+ 6 MeV`, 1.10 cm → `mu- 182 MeV`) and is already on
  record in doc pr/80 §13.5 as a wrong-end-polarity hand-scan miss at 64.8 cm — the
  same near-vertex confusion seen from the scanner's side.
- **evt285567** also appears, and is one of the events doc pr/51 built
  `main_vertex_graph_audit` for.
- **evt283040 is the top entry here** (`pi+ 0 MeV`, 0.00 cm → `mu- 188 MeV`, 67 cm)
  and is also one of the three events (283040 / 59899 / 72586) the owner gave to the
  concurrent **doc pr/83** duplicate-trajectory investigation
  (`scripts/pr83_dup_metric.py`). **pr/83 and this §5 are looking at the same
  complaint from two angles** — pr/83 measures duplicate and self-retracing
  trajectories inside a segment; this section measures what the resulting stubs do to
  the particle hierarchy. Neither supersedes the other, and P5 below should be
  reconciled with pr/83's findings before anyone builds it.

*(Numbering note: this doc took 84 because pr/82 is reserved by doc pr/81 §"NEXT
ROUND" for the ~2000-event DL-vertex sample, and pr/83 was claimed the same minute by
the concurrent duplicate-trajectory session.)*

**Continued in doc pr/85** (`85_near-vertex-pr-quality.md`), which takes this
section over from the PR-graph side rather than the particle-tree side: it
measures the same stubs as *interposed segments* between the neutrino vertex and
its prongs (21 events), names the two operations that create them
(`snap_main_vertex_to_kink` and the final `improve_vertex`, both running **after**
every `examine_*` pass), and shows that `main_vertex_graph_audit` op3 cannot reach
them because of its `degree(far_vertex) == 1` terminal-only line — which is the
missing detail in P5 above.

### The dropped-residual sub-family

`other_seg_keep_isolated_ok` (`NeutrinoOtherSegments.cxx:33-38`) requires
`points >= 25 **AND** length >= 3 cm`. Because it is an `AND`, a long sparse track
dies on point count alone. Of **331** distinct `pr54 isolated-residual drop`
records:

| | count | blocked *only* by the 25-point floor |
|---|---|---|
| length ≥ 15 cm | 10 | **10** |
| length ≥ 30 cm | 6 | **6** |

evt284794's is the extreme case: **71.35 cm at 21 points**, straight
(`dir_mag=68.29`), running from (-48.3,14.0,346.1) to (-47.6,-22.2,288.2) — whose
near end is **7.9 cm from the neutrino vertex**. A 71 cm track was discarded for
having four points too few. evt287517 loses a 14.29 cm residual at the vertex the
same way.

---

## 6. Summary against the owner's two categories

| owner's category | this doc | events | footprint |
|---|---|---|---|
| **"a bug to fix"** — disconnected gammas whose electron touches the main cluster | D1, D2, D3 → one rendering rule (§2.1) | 65289, 283713, 284794 | 7 showers / 5 events of 493 |
| **"can be an improvement"** — near-vertex PR not simple enough | D4 (§5) | 287517 | 29 inversions / 21 events of 491; plus 10 long residuals dropped |

The unification worth stating outright: **evt284794's `neutron 83 MeV` is the same
rendering rule as evt283713's `gamma 209 MeV`, firing on a different upstream
cause.** Nothing about it is specific to photons or to EM showers.

---

## 7. Proposals

All default OFF per CLAUDE.md §1; none is implemented. Listed with the gate each
would need before it could be considered.

| id | proposal | scope | gate |
|---|---|---|---|
| **P1** | `pf_direct_when_touching` — in `append_pseudo_shower`, render the shower leaf directly (no synthetic parent) when its start segment is in the **main cluster** *and* its nearest charge is within `pf_touch_max` (3 cm) of the main vertex. `start_connection_type` is **not** modified, so kinematics, `mc_included` and every tagger see exactly what they see today. | display / PF tree only | the 7 showers in §4 flip and nothing else does: `hash_archive.py` on `mabc-*.zip` across the standard manifest, plus the 3 predicted events hand-scanned first |
| **P2** | `conn3_stitch` — the root fix. Where `shower_conn3_unreachable` finds a small `min_dis` to a main-vertex-reachable anchor, **reconnect the component** (`merge_vertex_into_another`, `NeutrinoPatternBase.cxx:2546`) instead of promoting it at connection type 3. The BFS then reaches it and it becomes conn 1 naturally. Natural home is a new op in `main_vertex_graph_audit`, which is verified to run **after** `find_other_segments` (`TaggerCheckNeutrino.cxx:1411` vs `NeutrinoPatternBase.cxx:2885`) and so can repair what that pass creates. | changes the graph | full A/B plus a mover census; this moves vertex fits and therefore everything downstream. The most valuable and by far the most expensive of the five. |
| **P3** | zero-length pseudo-parents: in the `start_vtx not in BFS tree` fallback (`MultiAlgBlobClustering.cxx:1552-1563`), anchor `gstart` at the **main vertex** rather than at the shower's own start vertex, so the 15 remote associations of §4 draw their real gap instead of collapsing. Deliberately the **opposite** direction to P1, in the same function. | display / PF tree only | the 15 nodes gain nonzero length; no other node moves |
| **P4** | relax the `other_seg_keep_isolated` floor from `points ≥ N AND length ≥ L` to `(points ≥ N AND length ≥ L) OR length ≥ L_long`, recovering the 10 long residuals of §5. **Dependency, not a footnote: on its own this manufactures more disconnected components, hence more spurious gammas.** It should not ship before P2. | changes the graph | 10 recovered segments accounted for individually; requires P2 |
| **P5** | a `main_vertex_graph_audit` op for D4: collapse an interior segment under ~2 cm whose two endpoints are vertices, one of them the main vertex, and which is collinear with a neighbouring prong. 29 candidates in 21 events. mvga's existing ops (op1 duplicate prong, op2 charge-less bridge, op3 terminal stub) do not cover the *interior* case. Note mvga is currently `false` in SBND production. | changes the graph | mover census over the 21 events |

**Recommended order.** P1 first: it is display-only, its target set is enumerated to
seven rows, and it makes the owner's four events read correctly for scanning
without touching a physics number. P3 next, same cost, and it fixes the population
P1 must not touch. P2 is the real fix and should be its own round with its own A/B.
P5 stands alone. P4 must not go first.

---

## 8. Surfaced, deliberately not proposed (CLAUDE.md §5 rule 4)

**8.1 `PRShower::set_start_vertex` diverges from the prototype.** The toolkit
early-returns on an invalid vertex, leaving the *previous* connection type in place:

```cpp
// clus/src/PRShower.cxx:88-99
Shower& Shower::set_start_vertex(VertexPtr vtx, int type)
{
    if (!vtx || !vtx->descriptor_valid()) {
        m_start_vertex = nullptr;
        return *this;                     // conn type NOT updated
    }
    ...
    data.start_connection_type = type;
```

WCP assigns unconditionally (`prototype_base/pid/src/WCShower.cxx:529-532`). Since
the connection type is precisely what decides pseudo-parent insertion, a failed
re-seat can silently flip an electron to a gamma. Two readings: (a) the toolkit
guard is a deliberate null-safety improvement and the stale type is an accepted
consequence; (b) it is an unintended divergence. **Not picked here** — it is not
listed in `clus/docs/porting/porting_dictionary.md`, and no event in this study is
attributed to it. Recorded so it is findable if a future case points at it.

**8.2 `merge_nearby_vertices` compares `wcpt()`, not `fit()`.** The co-location pass
that could in principle stitch a split component uses a **0.1 cm** radius in
Steiner-point space:

```cpp
// clus/src/NeutrinoPatternBase.cxx:2374
if (ray_length(Ray{vtx1->wcpt().point, vtx2->wcpt().point}) >= 0.1 * units::cm) {
```

Every distance in this document is a **fit** distance, and the calib dumps carry no
`wcpt`. **This doc therefore proposes no radius number.** Any future radius knob
must first measure the wcpt-space separation of the split-component vertex pairs
from inside the code. Doc pr/80 §13's result that 0.8 cm collapses 3901 co-located
groups over 473 labels while breaking none is a **display-side** measurement of a
different vertex set and does not transfer.

---

## 9. Open

- The three predicted events (**evt347129, evt169626, evt174752**) are unscanned.
  They are the cheap validation of §4's selector and should be looked at before any
  of P1–P5 is built.
- evt284794's D3 — prongs at the vertex living in separate clusters — is an
  *upstream clustering* question, not a PR-graph one. P1 would make its tree read
  correctly (protons directly off the vertex) without fusing the clusters. Whether
  clusters 44 and 45 should have been merged into 12 in the first place is a
  separate round.
- 29 of 493 events have a split main cluster but only 3 have a stranded vertex
  within 5 cm. The other 26 are wide splits, a different phenomenon, and this study
  says nothing about whether they are correct.

---

# ROUND 2 (2026-08-17) — the fix round: F1/F2 display knobs + F3 conn3 stitch

**Status: implemented.**  Owner request (2026-08-17): "for the main cluster, I
can have a track --> gamma --> electron.  Inside one cluster, everything
should be connected ... one such example is evt 283713, proton --> gamma
(209 MeV) + neutron (507 MeV), all in the same cluster, another case is evt
407280.  another case 54921, also evt 316025.  Can you investigate and fix
these first?"

**Event-number note.**  "54921" exists in no sample; **64921** does — it is
bee_idx 7 of the pr40r7 track-misid scan the owner was reading, and its PF
tree shows exactly the reported pattern (`gamma 283 MeV → e-` beside proton +
muon prongs).  This round proceeds with 64921; if the owner meant a different
event, §12's selector will find it.

## 10. Round 2 repro

```
cd sbnd_xin
# arms (base = clean HEAD 75ff3e29; off/disp/stitch = branch with the three
# knobs, PR_JOBS=32, owner-authorized; manifest = prod0813 lists + the 15
# extra pr40r7-scan events in their HOME samples -- see the sample-collision
# note in sec 11):
PR_EXTRA_STAGES=pr_display PR_JOBS=32 ./run_pr_chain_batch.sh \
    work-mcp1k-cb0805 work-pr84r2-base-mc data $(cat /home/xqian/tmp/pr84r2-events-mc446.txt)
#   ... same for {off,disp,stitch} x {mc,nue,ncpi0}; disp adds
#   SBND_PF_DIRECT_WHEN_TOUCHING=1 SBND_PF_PSEUDO_GAP_FROM_MAIN=1, stitch adds
#   SBND_CONN3_STITCH_MAX=3.

# gates
python3 scripts/pr83r3_hash_gate.py work-pr84r2-off-mc work-pr84r2-base-mc \
    work-pr84r2-off-nue work-pr84r2-base-nue work-pr84r2-off-ncpi0 work-pr84r2-base-ncpi0
python3 scripts/pr84r2_disp_gate.py work-pr84r2-disp-mc work-pr84r2-off-mc \
    work-pr84r2-disp-nue work-pr84r2-off-nue work-pr84r2-disp-ncpi0 work-pr84r2-off-ncpi0
python3 scripts/pr84r2_pf_census.py work-pr84r2-off-mc work-pr84r2-disp-mc
python3 scripts/pr84r2_pf_census.py work-pr84r2-off-mc work-pr84r2-stitch-mc

# compiled-config proof (M6): knob-off byte-identical vs HEAD cfg, all six
# keys appear knob-on (needs the runner's pipeline_names TLA -- the bare
# default pipeline has no tagger_check_neutrino and compiles the keys away):
#   /home/xqian/tmp/pr84r2_cfgproof/{pipe-off,pipe-head,pipe-on}.json
```

## 11. The four owner events, diagnosed

Fresh-baseline (HEAD 75ff3e29, arms `work-pr84r2-base-*`) evidence; all four
pathologies survive the four post-doc mvga production flips (pr/83 r3/r4/r4b,
pr/86 r2).  Two flavors, both "main vertex stranded on a small piece":

| evt | node | conn | flavor | key numbers |
|---|---|---|---|---|
| 283713 | gamma 209 MeV (0.47 cm) **+ neutron 567 MeV (0.34 cm)** | 3 | same-cluster split graph (D2: snap + pr/30 P8 mismatch strand vertex 17089) | `conn3_unreachable` anchor_dis 0.3/0.5 cm; shower charge 0.00 cm from vertex |
| 316025 | gamma 477 MeV (0.33 cm) | 3 | same-cluster split graph; SNAP fired here too | anchor_dis 2.1 cm; charge 0.00 cm; 3/3 main-cluster segments |
| 407280 | gamma 414 MeV (2.98 cm) | 2 | **cross-cluster**: nusel main cluster = 16 (the shower's own), vertex in 25-point fragment cluster 51 | fitted charge min 2.98 cm; NO sentinel fires |
| 64921 | gamma 283 MeV (6.75 cm) | 2 | **cross-cluster**: nusel main = 11 (shower's), vertex in cluster 62 | charge min 6.75 cm |

(The owner's "507 MeV neutron" reads 567 MeV in the dump.)  The conn=2
cross-cluster flavor is NEW relative to round 1's D1–D4: the vertex was
*determined* into a tiny fragment of the bundle while the event body —
including the big shower — lives in the cluster nusel itself calls main.
Root cause there is vertex determination (doc pr/52 territory), out of scope
here; the display fix covers it, the graph fix deliberately does not.

**Sample-collision trap (recorded for the next round):** 14 of the 45
pr40r7-scan events are nueCC48/NCpi0 events, not mcp1k — event numbers
collide across samples, and feeding them to the mcp1k hub fails with "no
pctree".  Every one is already in its home sample's prod0813 manifest, so the
three arms together cover all 45.

## 12. The fixes (all default OFF)

Toolkit commits: `a58ca54f` (knobs, ALL DEFAULT OFF) + `9f1cbd9e`
(cfg/sbnd production flip).  Knob threading follows the pr/34
(BeePFConfig) and pr/74-K5 (Tagger) precedents exactly; compiled-config proof
in §10.

**F1 `pf_direct_when_touching` + `pf_touch_max` (display-only, = P1
extended).**  In `fill_bee_pf_tree`'s non-pi0 indirect loop
(`MultiAlgBlobClustering.cxx` `append_showers`), a conn-2/3 shower whose fit
cloud comes within `pf_touch_max` (C++ default 3 cm) of the main vertex
(`shower_get_closest_point`, render time, no state change) is appended as a
direct leaf in the same children array the pseudo carrier would have landed
in.  `start_connection_type` is NOT modified — kinematics, `mc_included`,
every tagger untouched; only mc.json can move.  pi0 daughters are exempt by
construction (the branch is upstream of the pi0 grouping — round 1's C1
placement trap avoided).  Distance-primary selector, NOT rendered-length (§4's
trap): the 15 genuinely-remote zero-length carriers (charge floor 4.91 cm)
stay pseudo-parented.

**F1 rung 2 `pf_touch_cross_main` + `pf_touch_cross_max` (display-only,
separately flippable).**  Extends F1 to a conn-2 shower in a DIFFERENT
cluster than the vertex when the shower's cluster carries
`Flags::main_cluster` and the distance is within 8 cm.  Offline census over
the 512 baseline events (start-point distance, upper bound of the fit-cloud
metric):

| evt | shower | KE (MeV) | shower cl / vtx cl | d (cm) | pio_id |
|---|---|---|---|---|---|
| mc 407280 | 16010 | 414.2 | 16 / 51 | 2.98 | -1 (also rung 1) |
| mc 64921 | 11002 | 283.7 | 11 / 62 | 6.75 | -1 |
| mc 286906 | 9002 | 63.2 | 9 / 41 | 4.46 | -1 |
| mc 409546 | 9000 | 90.9 | 9 / 41 | 3.96 | -1 |
| ncpi0 506114 | 19016 | 1839.1 | 19 / 28 | 6.20 | 0 (pi0-grouped => F1-exempt) |
| ncpi0 521075 | 18007 | 674.5 | 18 / 94 | 6.81 | -1 |
| ncpi0 71372 | 19020 | 1974.6 | 19 / 92 | 5.20 | 0 (pi0-grouped => F1-exempt) |

Seven candidates in 512, all high-KE, all in the cluster nusel calls main —
the pathology class exactly.  **286906, 409546, 521075 are new predictions
nobody has scanned.**

**F2 `pf_pseudo_gap_from_main` (display-only, = P3).**  In the
`start_vtx not in BFS tree` fallback the pseudo carrier is anchored at the
MAIN vertex instead of the shower's own start vertex, so §4's 15 remote
associations draw their real median-38.8-cm gap instead of collapsing to a
zero-length node.  Non-interacting with F1 by construction (F1 fires ≤ 3 cm,
this population ≥ 4.91 cm).

**F3 `conn3_stitch_max` (graph fix, = P2, the root fix).**
`PatternAlgorithms::stitch_disconnected_main_cluster` (NeutrinoGraphAudit.cxx,
modeled on `swap_orphan_dup_audit`), called in TaggerCheckNeutrino right
after the mvga block and BEFORE `clustering_points`.  Deterministic
fixed-point loop (cap 8): global argmin of `segment_get_closest_point`
(the exact metric `conn3_unreachable` logs as anchor_dis) over (unreachable
main-cluster segment, reachable vertex); if within the radius (deploy 3 cm),
bridge with the `connect_direct` recipe (`do_rough_path` charge-following —
declines with a sentinel if the charge really is disconnected;
`create_segment_for_cluster`; `add_segment`), then ONE op4-style
whole-cluster refit.  Every downstream pass sees a connected graph, so the
piece classifies conn-1 naturally; `shower_conn3_unreachable` stays ON as the
backstop for wider gaps.  Rejected alternatives (recorded): in-block stitch
inside `shower_clustering_in_other_clusters` (unfitted segment mid-pass,
manual conn bookkeeping) and `merge_vertex_into_another` (destroys the fits
of every incident segment for non-co-located vertices and moves the
junction).  Known limitation: the bridge lands vertex-to-vertex even when the
closest approach is mid-segment — accepted at ≤3 cm gaps.  F3 moves fits and
conn types (`mc_included`, kinematics) — full A/B below, its own flip.

## 13. Verification

Arms: `work-pr84r2-{base,off,disp,stitch,stitch1,prod}-{mc,nue,ncpi0}` (446 /
47 / 19 events; base = clean HEAD `75ff3e29`, off/disp/stitch* = fix build,
prod = fix build under the flipped production cfg, no env overrides).  All
gates below re-checkable from those arms.

**V0 (baseline).**  All four pathologies reproduce at HEAD under the four
post-round-1 mvga flips (sentinels + trees quoted in §11).

**V1 knob-off byte gate: PASS=1024 FAIL=0** (`pr83r3_hash_gate.py`, 512
events x {mabc-pr.zip, pctree}, off vs base).  Freshness proof +
`./build/clus/wcdoctest-clus` 2129/2129 before the A/B.  Compiled-config
proof: knob-off JSON byte-identical to HEAD; all six keys appear knob-on
(M6 trap hit and recorded: the proof MUST pass the runner's
`pipeline_names` — the bare default pipeline contains no
tagger_check_neutrino and silently compiles the keys away).

**V2 display gate (disp vs off): PASS=512 FAIL=0, mc.json-moved=39**
(`pr84r2_disp_gate.py`) — every diff confined to `data/0/0-mc.json`.
Decomposition (`pr84r2_pf_census.py`): **32 F1 suppressions** (all four §11
events' carriers, round-1's 65289 pair, and all three round-1 unscanned
predictions 347129/169626/174752 — the §4 selector confirmed in vivo) +
**106 F2 re-anchors** (every zero-length remote carrier now draws its real
gap, e.g. a 567 MeV gamma 0 → 60.8 cm; none suppressed).  The 39th moved
event (286906) is an id-renumber only: the off arm allocates a pseudo id
even for a carrier the KE floor then drops, the suppress branch does not.
nusel scores: **0 movers**; nusel labels identical.

**V3 graph gate.**  At the swept 3 cm: 42 bridges / 28 events, and nueCC
**38856 is ADVERSE** — two speculative 2.5/2.9 cm bridges fragment the
1244 MeV electron into three pieces and flip nue 3.25 → −3.45.  At **1 cm**
(shipped): **10 bridges / 10 events, every one sub-cm-family**; 9 score
movers, all accounted and favorable-or-neutral (283713 enu 1513→2034 — the
stranded 567 MeV muon rejoins the vertex tree; 66272 rescues an invisible
pi+ → proton branch, enu 162→442; nueCC 168596's 1929 MeV primary
consolidates 30→17 nodes; five small numu wiggles ≤0.5); **ncpi0
untouched; zero nu-candidate label flips**; 407280/64921 unchanged
(cross-cluster, out of F3's reach — asserted).  `do_rough_path` declined 0
times at either radius.

**Production composition (prod vs off).**  Scores == the stitch1 mover set
exactly (F1/F2 add zero); labels identical; PF census 42 moved events =
the F1/F2/F3 union with the expected overlaps.

**F1.0 rung-2 probe: FAILED, rung 2 stays dark.**  With
`pf_touch_cross_main=1` on all 7 census candidates: **zero movers** —
`Flags::main_cluster` is NOT set on the event-body cluster at PF-writer
time (confirmed with WCT_BEE_PF_PRINT on 64921: pseudo-gamma still added).
64921 + the 4 unscanned rung-2 predictions are deferred; the root cause
(vertex determined into a tiny fragment of the bundle) is doc pr/52
territory.

## 14. Flip records (SBND production, 2026-08-17)

| knob | value | evidence |
|---|---|---|
| `pf_direct_when_touching` | **true** | V2 |
| `pf_touch_max` | null (C++ 3 cm) | V2 |
| `pf_pseudo_gap_from_main` | **true** | V2 |
| `conn3_stitch_max` | **1** (cm) | V3 at 1 cm; the 3 cm sweep is the counter-evidence |
| `pf_touch_cross_main` / `pf_touch_cross_max` | **false / null (dark)** | F1.0 probe failure |

## 15. Bee sets (before/after, identical 24-event order)

- BEFORE (all knobs off):
  <https://www.phy.bnl.gov/twister/bee/set/46300d3c-70b6-4e3d-b755-81bcc0d1ae8d/event/list/>
- AFTER (flipped production):
  <https://www.phy.bnl.gov/twister/bee/set/d595c5d2-5633-4f23-ac76-352ee1027ec3/event/list/>
- Index: `bee/pr84r2/pr84r2.index.txt` — idx 0-3 the owner's four; 4-7
  round-1's event + predictions; 8/14/16-20 the F3 movers; 9/10/21 the THREE
  NEW unscanned rung-2-class predictions (286906 / 409546 / 521075); 11-13/15
  multi-suppression events; 22-23 ncpi0 display movers.

**Scan requests to the owner:** (a) the four fixed events read correctly
now; (b) the three rung-2-class predictions — if they are the same
pathology, that plus 64921 sizes the deferred vertex-in-fragment round;
(c) 168596 and 316025 — the two biggest F3 reorganizations — look
physically right.

## 16. Records

- `scripts/pr84r2_disp_gate.py` — member-level display gate (only mc.json
  may move).
- `scripts/pr84r2_pf_census.py` — PF-tree movers census between two arms.
- Movers TSVs: `/home/xqian/tmp/pr84r2_movers_stitch_{mc,nue,ncpi0}.tsv`
  (scratch; the numbers are quoted in §13).
- Sample-collision note (§11): 14 of the 45 pr40r7-scan events live in
  nueCC48/NCpi0, not mcp1k.


---

# ROUND 3 (2026-08-18) — two showers on one start segment

Owner scan of the round-2 Bee sets reported six PF problems. Four of them
("duplicated electron", "clicking is weird", "the entire PF disappeared") are
one defect, and it is **not cosmetic**: the same shower is also counted twice
in the reconstructed neutrino energy. Owner directive: *"this is a clear bug
... Isn't that coming from the same underlying issue. If we fix the underlying
issue, not just display, we are better, right? Aim to fix the underlying issue
please."* — so this round fixes the shower set, not the renderer.

## 17. Round 3 repro

```
cd sbnd_xin
# attribution of the twin (which pass built it) -- log-only, byte-neutral:
WCT_SHOWER_CREATE_DEBUG=1 PR_EXTRA_STAGES=pr_display PR_JOBS=4 \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr84r3-dbg-mc data \
      169626 174752 347129 394532
grep SHOWER_CREATE_DEBUG work-pr84r3-dbg-mc/pr_evt394532/stdout.log

# arms: manifest = the 24 round-2 Bee events (the 4 twin events are inside it)
PR_EXTRA_STAGES=pr_display PR_JOBS=32 ./run_pr_chain_batch.sh \
    work-mcp1k-cb0805 work-pr84r3-off-mc data $(cat /home/xqian/tmp/pr84r3-mc.txt)
#   ... same for {off,dedup,prod} x {mc,nue,ncpi0}; dedup adds
#   SBND_SHOWER_DEDUP_START_SEG=1 SBND_PF_UNIQUE_NODE_IDS=1; prod is a BARE
#   run against the flipped cfg (production-composition proof).

# gates
python3 scripts/pr83r3_hash_gate.py work-pr84r3-off-mc work-pr84r2-prod-mc ...
python3 scripts/pr84r2_disp_gate.py work-pr84r3-off-mc work-pr84r3-dedup-mc ...
python3 scripts/pr84r3_dedup_census.py work-pr84r3-off-mc work-pr84r3-dedup-mc
```

## 18. Root cause: the shower set contains twins

`fill_bee_pf_tree` ids a shower node by the prototype convention
`cluster_id*1000 + seg_id` (`MultiAlgBlobClustering.cxx:1588`,
`NeutrinoID.cxx:1268`) — the START SEGMENT's id. Two `PR::Shower` objects
built on the same start segment therefore emit two nodes carrying one jsTree
id, and jsTree keys its model by id (`_model.data[...]`,
`wire-cell-bee3/events/static/js/lib/jstree.min.js`). In 394532 the repeated
id sat on a node **and its own descendant** (`e- 66 MeV` [8033] inside
`e- 66 MeV` [8033]) — selecting it blanked the whole panel, exactly as the
owner saw. The id addresses nothing else: `events/static/js/bee/physics/mc.js`
draws each node as one straight line from `data.start` to `data.end`.

`WCT_SHOWER_CREATE_DEBUG` attributes every twin to one of two passes:

| evt | twin | built by | why it was allowed |
|---|---|---|---|
| 169626 | seg 22024: sid0 conn1 88.1 MeV 1 seg · sid1 conn3 107.9 MeV 6 segs | `nv_in_main_cluster` + `in_other_clusters_A` | the start-segment pick in `shower_clustering_in_other_clusters` (`NeutrinoShowerClustering.cxx:1752-1766`) consults **no claim at all**, and the segment it takes off the cluster's own vertex can belong to another cluster entirely |
| 174752 | seg 48010: sid0 conn1 9.6 · sid2 conn3 22.0 | same | same |
| 347129 | seg 53021: sid0 conn1 65.0 2 segs · sid1 conn3 156.7 8 segs | same | same |
| 347129 | seg 11000: sid14 · sid15, both conn3 63.9 MeV 1 seg | `in_other_clusters_B` + `conn3_unreachable` (pr/74 K5) | K5 *does* guard on `map_segment_in_shower` (`:2030`), but `update_shower_maps` only refreshes that map at the END of the function, so K5 reads a map that predates loop B |
| 394532 | seg 39023 (conn1 30.6 / conn3 38.0) and seg 8033 (sid8/sid9, both 66.0) | both patterns | both |

**The prototype has the identical hole** (`NeutrinoID_shower_clustering.h`
:1481-1495 picks `sg` off `map_vertex_segments[vertex]` with no
`map_segment_in_shower` test, and its `mc_id` is the same
`cluster_id*1000 + seg_id`), so this is not a porting divergence (M15) — it is
a latent defect both codebases share, and the fix is a deliberate improvement
over the prototype, shipped behind a knob.

**The twins predate round 2.** They are present in the round-2 `off` arm
(`169626 off: dup_ids={22024: 2}`); round-2's F1 removed the pseudo-gamma
wrapper that had been hiding them.

## 19. The same twins inflate the neutrino energy

`fill_kine_tree` is called with the same `showers` set
(`TaggerCheckNeutrino.cxx:1937`), so a twin is counted twice:

| evt | `kine_reco_Enu` off | on | delta | what was removed |
|---|---|---|---|---|
| 169626 | 825.4 | 737.3 | −88.1 | the 1-segment 88.1 MeV view of the same electron the 107.9 MeV view describes |
| 174752 | 188.7 | 176.0 | −12.6 | 9.6 + 22.0 replaced by one 18.9 MeV entry |
| 347129 | 700.8 | 571.9 | −128.9 | 65.0 (small view of the 156.7 electron) + a verbatim 63.9 duplicate |
| 394532 | 352.2 | 248.2 | −104.1 | 38.0 (same segment as the kept 30.6) + a verbatim 66.0 duplicate |

Note the census metric `kine_dup` (identical `(pdg, energy)` entries) only sees
the *verbatim* duplicates: 169626 and 174752 double-count with two different
energies for one electron and are invisible to it. The authoritative metric is
`twins` (showers sharing a start segment). One false positive is worth
recording: nueCC **256587** has two genuinely different 1.961 MeV showers
(segs 50150 / 95195) — coincidence, not a twin, and it is unchanged by the fix.

## 20. The fixes

**S1 `shower_dedup_start_seg`** (bool, C++ default **false**) —
`PatternAlgorithms::merge_showers_sharing_start_segment`
(`NeutrinoShowerClustering.cxx`), called from `shower_clustering_with_nv`
**after `examine_showers` and before the pi0 finders**, i.e. after every pass
that can create or retarget a shower and before every consumer (pi0 pairing,
`fill_kine_tree`, the Bee PF writer) reads the set. Each group of showers
sharing a start segment collapses onto its most directly connected member
(order: lowest `start_connection_type`, then most segments, then highest
`kine_best`, then lowest `shower_id`), which **absorbs** the others via
`Shower::add_shower` — whose membership gate makes the overlap idempotent — so
no charge is lost; the survivor's `flag_kinematics` is cleared and the
production `calculate_shower_kinematics` recomputes **only** it. Grouping is
keyed on the start segment's graph index (true identity, deterministic map
order), never a pointer (pr/34 §6). `conn_type == 4` members never participate:
they are skipped by both the PF tree and the kine tree, so folding their charge
in would be a change with no reported defect behind it.

Measured effect — the survivor keeps the fuller view AND the direct anchor:

```
169626  seg 22024  sid0 (88.1, 1 seg, conn1)  <- absorbs sid1 (107.9, 6 segs, conn3)
                => (107.9, 6 segs, conn1), one root node instead of two
347129  seg 53021  sid0 (65.0, 2 segs, conn1) <- absorbs sid1 (156.7, 8 segs, conn3)
                => (156.7, 8 segs, conn1); the `e- 156` that used to sit inside
                   its own descendant `e- 63` is now the root
```

**G1 `pf_unique_node_ids`** (bool, C++ default **false**) — in
`fill_bee_pf_tree`, a node whose id is already taken is re-issued from a range
above the natural id space and the collision is logged
(`pr84 pf_id_collision`). This is the standing guarantee that `mc.json` is
valid jsTree input; with S1 on it fired **0 times** on the gate manifest, which
is the intended state (a firing is itself a finding: a shower leaf can still
collide with the track node of the same segment, and the pseudo/pi0 counter
starts at 1 inside the same number space).

## 21. Verification (24-event gate manifest = the round-2 Bee set)

- **V0** `wcbuild` rc=0; `build/clus/libWireCellClus.so` 06:49 newer than every
  edit (libs load from `build/<pkg>/`); `./build/clus/wcdoctest-clus`
  **210 cases / 2131 assertions, 0 failed**, including the two new default pins.
- **V1 knob-off byte gate**: `work-pr84r3-off-*` vs the round-2 production arms
  `work-pr84r2-prod-*` — `pr83r3_hash_gate.py` **PASS=48 FAIL=0** (24 events ×
  {`mabc-pr.zip`, `pctree-pr-evt*.tar.gz`}, member-content hashes, never raw
  `cmp`). This doubles as the proof that the round-2 arms correspond to this
  source, so no separate `base` arm was needed.
- **V2 knob-on blast radius**: `dedup` vs `off` — `pr83r3_hash_gate.py`
  **PASS=44 FAIL=4**, and the 4 failures are the 4 twin events' `mabc-pr.zip`
  only; **every pctree tarball is byte-identical**, so no fit or graph moved.
  `pr84r2_disp_gate.py`: the only moved member is `data/0/0-mc.json`
  (`mc.json-moved=4 / 0 / 0`). `pr84r3_dedup_census.py`: `changed=4`,
  `dup_ids` 4 events → **0**, `twins` 4 events → **0**.
- **V2b physics movers**: calib `tagger`/`kine` blocks move in exactly those 4
  events and nowhere else (0/7 nueCC, 0/3 NCpi0, 10/14 mcp1k untouched). Enu as
  in §19. Score movers are all BDT-input shifts with no selection change; the
  largest is **347129**, where `cosmict_flag`/`cosmict_flag_7` 1.0 → 0.0 lifts
  the cosmic sentinel and `nue_score` moves −15.0 (sentinel) → −4.30 (real BDT
  output) — still negative, still not a candidate. **Zero nusel label flips**
  across all 24 events (`nusel-evt*.tsv` identical everywhere).
- **V3 production composition**: bare-config `work-pr84r3-prod-*` (no env
  overrides, flipped cfg) vs `work-pr84r3-dedup-*` — `pr83r3_hash_gate.py`
  **PASS=48 FAIL=0**, i.e. the shipped operating point reproduces the gate arm
  exactly.
- **Sentinels**: 6 `pr84 shower_dedup` absorptions in 4 events, 0
  `pr84 pf_id_collision`, and both are silent in the knob-off arm.
- **Compiled-config proof (M6)**: with the runner's `pipeline_names` TLA, the
  knob-off compile of `wct-pr-perevt.jsonnet` is byte-identical to the
  pre-change compile and contains neither key; knob-on shows
  `"shower_dedup_start_seg" : true` and `"pf_unique_node_ids" : true`.

## 22. Flip records (SBND production, 2026-08-18)

| knob | value | evidence |
|---|---|---|
| `shower_dedup_start_seg` | **true** | V1/V2/V2b/V3 above; owner directive to fix the underlying issue |
| `pf_unique_node_ids` | **true** | standing invariant; fires 0 times with S1 on |

This one **changes physics output** (Enu on 4 events of the 492-event round-2
census) — the §1 bar is met by the knob-off byte gate, and the population is
bounded by that census rather than by a fresh full A/B, per the owner's
instruction to keep this round's validation small.

## 23. Answers to the owner's other three reports (no code this round)

- **168596 — "pi0 does not make sense: 21 + 33 MeV"**: correct, and the
  suspicion that the pi0 should hold the major electron is confirmed by the
  code's own numbers. The display grouping pairs shower sid3 (start seg 14072)
  with sid6 (128147) at mass **114.1 MeV** — computed from `get_kine_charge()`,
  where sid3 was **179.9 MeV** at pairing time. After pairing,
  `set_start_vertex` + a re-`calculate_kinematics` moved sid3's `kine_best`
  onto its range estimate, **21.5 MeV**, which is what the leaf prints. So the
  node text and the mass come from two different energy estimates of the same
  shower. Separately, the tagger's own `pio_kine` block pairs **33.5 + 2039.1
  MeV, mass 96.4** — the kinematic pi0 *does* contain the major electron.
  Also note the pi0 node prints a reconstructed **invariant mass** in a slot
  labelled MeV of KE; the toolkit matches the prototype here
  (`fill_pi0_reco_tree` uses `map_pio_id_mass[...].first` as the KE), so this
  is M15 territory: both readings recorded, neither picked.
- **285567 — "which PF contains (59.7, −165.5, 294.2)?"**: segment **8040**
  (cluster 8 = the main cluster, pdg 11, 34.5 cm, part of shower 8105 = the
  1193 MeV node). It looks missing for two reasons. (i) The point sits in a
  **disconnected component of the main cluster**: only 8 vertices are reachable
  from main vertex 8007, and segments 8035/8040/8041 (vertices 8022/8023/8024/
  8027) plus 8001/8015/8016 are not — the round-2 conn-3 family again, with
  gaps 2.06/2.30 cm, above the shipped `conn3_stitch_max = 1 cm` (3 cm was
  measured ADVERSE on nueCC 38856, §13). (ii) Bee draws every PF node as ONE
  straight `start → end` line (`mc.js`), so a 14-segment shower's interior is
  invisible by construction. The 1193 MeV "electron" spans that detached piece
  and absorbs the muon's continuation, which is also why `mu- 111 MeV` shows
  too few daughters, and it is pi0-paired with a 5.3 MeV blip at mass 119.7.
  Graph round, as the owner said.
- **Proposal (not this round)**: `mc.js` already supports optional
  `traj_x/traj_y/traj_z` polylines. Emitting them for PF nodes would make
  "which PF contains this point" answerable by clicking. Not done here: it
  would change every event's `mc.json`, and the deployed viewer may not match
  the local `wire-cell-bee3` checkout.

## 24. Round 3 Bee sets (before/after, same 24 events, same order as round 2)

- BEFORE (round-3 knobs off = byte-identical to the round-2 production arms,
  i.e. exactly what the owner scanned):
  <https://www.phy.bnl.gov/twister/bee/set/3ac3c39e-eac5-446d-a741-7e4133cd3032/event/list/>
- AFTER (post-flip production):
  <https://www.phy.bnl.gov/twister/bee/set/9d145c32-eebe-4470-aa9b-4495322a5368/event/list/>
- Index: `bee/pr84r3/pr84r3.index.txt`.  **Only idx 5 / 6 / 7 / 11**
  (347129, 169626, 174752, 394532) differ between the two sets; the other 20
  events are byte-identical, which is the visual form of the V2 gate.

**Scan requests to the owner:** (a) the four fixed events -- one node per
electron, no self-nesting, and the panel survives clicking; (b) 168596 and
285567 read as the §23 answers describe (no code changed for them this round);
(c) still open from round 2: the three rung-2-class predictions
286906 / 409546 / 521075 and the deferred 64921.

## 25. Round 3 records

- `scripts/pr84r3_dedup_census.py` — twins / duplicate-id / kine-duplicate
  census, one or two arms.
- Probes (permanent, env-gated, byte-neutral): `WCT_SHOWER_CREATE_DEBUG`
  (creation site + retarget of every shower) and `shower_id` in the
  `WCT_BEE_PF_PRINT` shower-leaf line.
- Arms: `work-pr84r3-{dbg,off,dedup,prod}-{mc,nue,ncpi0}`; manifests
  `/home/xqian/tmp/pr84r3-{mc,nue,ncpi0}.txt`.
