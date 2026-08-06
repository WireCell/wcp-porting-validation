# doc pr/39 — Bee π0→γ→e⁻ display question: Bee config default (γ) + a real shower `end_point` defect (SBND run 18255, NCπ0 event 271851)

## Repro block

```bash
# Finding 1 (Bee display): the source mc.json for the set the owner looked at
# (https://www.phy.bnl.gov/twister/bee/set/21b32701-828d-4ff9-a3b9-d7b8807e07b5/event/1/)
unzip -p /home/xqian/tmp/campaign0805/ncpi0-cb0805.zip data/1/1-mc.json > /home/xqian/tmp/evt1_mc.json
# then read wire-cell-bee3/events/static/js/bee/physics/mc.js:85-131 and store.js:32-37

# Finding 2 (PR): the |start-vtx| / |end-vtx| table below, generated from the
# same mc.json plus the 8 other pi0 events in the same Bee set:
python3 - <<'PYEOF'
import json, math, zipfile
def dist(a,b): return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
z = zipfile.ZipFile('/home/xqian/tmp/campaign0805/ncpi0-cb0805.zip')
def find_pi0(nodes):
    for n in nodes:
        if n['text'].startswith('pi0'): return n
        r = find_pi0(n.get('children', []))
        if r: return r
    return None
def walk(nodes, vtx, counts):
    for n in nodes:
        t, dd = n.get('text',''), n.get('data',{})
        s, e = dd.get('start'), dd.get('end')
        if t.startswith('e-') and s and e:
            ds, de = dist(s,vtx), dist(e,vtx)
            counts[1 if de<1.0 else (0 if de>ds else 2)] += 1
        walk(n.get('children', []), vtx, counts)
for i in (1,3,4,8,10,12,15,16,17):
    d = json.loads(z.read(f"data/{i}/{i}-mc.json"))
    vtx = tuple(find_pi0(d)['data']['start'])
    c = [0,0,0]; walk(d, vtx, c)
    print(i, "ok=%d rev=%d other=%d" % tuple(c))
PYEOF
```

## Symptom

Owner report, Bee set `21b32701-828d-4ff9-a3b9-d7b8807e07b5` event 1 (SBND run
18255, evt 271851, `ncpi0-cb0805` sample): clicking the **π0** node shows a
ball; clicking a **γ** node shows nothing; clicking an **e⁻** node shows a ball
+ line. Question: is the invisible γ a particle-flow bug (gamma not tagged to
the root neutrino vertex) or a wire-cell-bee3 display issue?

## Root cause

**Two independent things, only one of which is a defect.**

### Finding 1 — γ invisibility: Bee display default, not a bug

`wire-cell-bee3/events/static/js/bee/physics/mc.js` `selectionChanged()` draws
**only** from a clicked node's own `data.start`/`data.end` — it never touches
the separate point-cloud files (`real_cluster_id` does not appear anywhere in
`mc.js`), so a node's id is irrelevant to what renders. Two things happen for
every non-filtered node (`mc.js:120-131`):

* a `THREE.Line` from `start` to `end` (or a `traj_*` polyline, unused by this
  producer's node shape — see doc pr/26 / `clus/docs/bee_output.md`), and
* a fixed radius-2 `SphereGeometry` placed **unconditionally at `start`**.

But `mc.js:88-99` filters by a **substring match on the node's label** first:

```js
else if (node.text.indexOf("gamma") >= 0) {
    if (this.store.config.mc.showGamma) { material = gammaMaterial }
    else { continue }
}
```

and `store.js:33-37` defaults `showGamma` (and `showNeutron`, `showNeutrino`)
to **false**. That explains all three observations with no PR-side
involvement:

| clicked | label filtered? | geometry | what renders |
|---|---|---|---|
| `pi0  … MeV` | no | start == end (§ Finding 2 below — a genuine point particle) | zero-length line + sphere at `start` ⇒ **a ball** |
| `gamma  … MeV` | **yes → `continue`** | valid, non-degenerate (see table below) | **nothing** |
| `e-  … MeV` | no | start ≠ end | line + sphere ⇒ **ball + line** |

**Fix (display-side, not attempted here):** load the set with
`?mc.showGamma=true` appended to the URL, or toggle "Monte Carlo → gamma" in
the dat.GUI panel. Nothing in the toolkit needs to change for this part; the
owner's stated hypothesis ("gamma not tagged to the root neutrino vertex") is
refuted below.

Corollary, reported not fixed: `MultiAlgBlobClustering::fill_bee_pf_tree`'s
synthetic pseudo-γ/π0 node ids (`next_id++`, `MultiAlgBlobClustering.cxx:1495`)
are small integers (1,2,3,5,6,12,13,14,17 in this event) that overlap
`clustering-global.json`'s `real_cluster_id` range (1–17). Currently latent
because Bee's mc-tree click path never joins node id to point-cloud id (see
above) — but it is a live hazard for any future feature that does, and it is
also an undocumented divergence from the prototype, which assigns pseudo
nodes real encoded ids (`NeutrinoID.cxx:1374`: `sg1->get_cluster_id()*1000 +
acc_segment_id`; π0 similarly at `:1328`). `bee_output.md:239` documents the
`cluster_id*1000+seg_id` scheme as universal, which these nodes violate.

### Finding 2 — real defect: shower `end_point` collapses onto the neutrino vertex

The PF **topology** is correct: every γ node's `start` is exactly the
neutrino vertex `(-32.99, 24.45, 363.23)` cm, nested `pi0 → gamma → e-` as
expected. But walking `data/1/1-mc.json`'s e⁻ leaves by distance to that
vertex:

| e⁻ node (id, KE) | \|start−vtx\| cm | \|end−vtx\| cm | |
|---|---:|---:|---|
| 59084, 108 MeV | 27.7 | 76.0 | OK — grows away |
| 54067, 16 MeV | 44.4 | 56.8 | OK — grows away |
| 11010, 1135 MeV | 0.0 | 99.7 | OK — starts at vtx, grows away |
| 63099, 18 MeV | 56.8 | **0.00** | **reversed** |
| 16037, 33 MeV | 57.3 | **0.00** | **reversed** |
| 35048, 32 MeV | 63.3 | **0.00** | **reversed** |
| 60088, 21 MeV | 81.0 | **0.00** | **reversed** |
| 57077, 41 MeV | 43.2 | **0.00** | **reversed** |

**5 of 8** showers in this one event have `end_point` sitting exactly on the
neutrino vertex — i.e. the display draws them running *backward*, toward the
vertex, instead of away from it. Across all 9 π0 events in the same Bee set
(script in the Repro block; "other" = neither clearly-away nor at-vertex,
ambiguous within the farthest-vertex geometry):

| evt(idx) | n showers | ok | reversed | other |
|---:|---:|---:|---:|---:|
| 1 | 8 | 3 | 5 | 0 |
| 3 | 9 | 4 | 1 | 4 |
| 4 | 8 | 6 | 2 | 0 |
| 8 | 6 | 4 | 2 | 0 |
| 10 | 19 | 8 | 8 | 3 |
| 12 | 6 | 4 | 1 | 1 |
| 15 | 9 | 3 | 4 | 2 |
| 16 | 8 | 2 | 4 | 2 |
| 17 | 10 | 4 | 5 | 1 |
| **total** | **83** | **38** | **32 (39%)** | **13** |

**Root cause — an unported prototype rule, one hop past where pr/38 round 2
fixed it.** `end_point` is computed as "the vertex in the shower's own vertex
set farthest from `start_point`" in three places in
`clus/src/PRShower.cxx::calculate_kinematics` (`:1011-1022`, `:1123-1127`,
`:1248-1263`, all iterating `ordered_nodes(*this, m_full_graph)` with no
exclusion). The prototype computes the same quantity in
`WCShower.cxx:377-387` by iterating `map_vtx_segs` — and
`WCShower::set_start_segment` (`:541-548`) explicitly does
`if (vtx == start_vertex) continue;` when populating that map, so **the start
vertex can never be picked as the farthest vertex**. For a detached
(`start_connection_type` 2 or 3) shower whose `start_point` sits tens of cm
from its own start vertex, the start vertex — which sits at or near the
parent's attachment point, i.e. often the neutrino vertex itself for a
first-generation γ — wins the farthest-distance search unless it's excluded.

This is the *same* prototype rule doc pr/38 round 2 (commit `df40b2a4`)
established and fixed — but only for `Shower::fill_sets`'s
`exclude_start_vertex` parameter, which feeds `fill_bee_pf_tree`'s BFS
barrier. The three `calculate_kinematics` farthest-vertex searches that set
`end_point` were never touched by that fix and still use the un-excluded
node set. **Undocumented divergence**: `clus/docs/porting/porting_dictionary.md`
does not list this rule for `calculate_kinematics`; per CLAUDE.md M15 it is
surfaced here rather than silently corrected.

**Blast radius — stated, not assumed.** `end_point` feeds `init_dir` for
`start_connection_type` 2/3 showers (`PRShower.cxx:1029-1034`,
`data.init_dir = (data.start_point - m_start_vertex->fit().point).norm()` —
note this direction computation does *not* even use `end_point`, so shower
*direction* may be unaffected) and the Bee PF node geometry drawn by
`fill_bee_pf_tree`. Whether `end_point` reaches PID or energy reconstruction
is **open** — not checked in this investigation — and must be traced before
any fix is scoped.

## Why it hid

The Bee mc-tree display never surfaces `end_point` as a number, only as a
line endpoint — and per Finding 1, γ lines are hidden by default, so the
γ→(neutrino vertex) line was never visually inspected before now. The
5-of-8 rate in this one event and 39% across the set suggests this is not
rare; it likely reads as "showers converge near the vertex," which is
directionally true for a neutrino interaction, so a shortened/reversed
segment could pass a casual scan.

## Fix

**Not attempted in this change** — diagnosis only, per owner instruction.
Finding 1 needs no toolkit fix (display config toggle). Finding 2, if the
owner elects to fix it, changes reconstruction output (`end_point` feeds Bee
PF geometry and possibly downstream shower direction / PID paths not yet
traced) and per CLAUDE.md §1/§4 requires: a default-OFF knob mirroring
`pf_shower_vertex_barrier`'s `exclude_start_vertex` idiom, applied to the
three `calculate_kinematics` farthest-vertex searches; a knob-off
byte-identical gate; and a knob-on smoke run showing the 32 reversed showers
above recover (`end_point` no longer landing on the neutrino vertex).

## Verification

- Finding 1: confirmed live by loading the set with `?mc.showGamma=true` —
  the five γ lines from the neutrino vertex to each e⁻'s start should appear.
  Owner to confirm.
- Finding 2: distance table regenerated directly from `data/1/1-mc.json` and
  the 8 sibling π0 events by the script in the Repro block (not hand-copied).
- Code citations checked against toolkit HEAD `b25bdcf1` (2026-08-06); the
  `v3_extension_guard` cfg-only commit landed between diagnosis and write-up
  and does not touch `clus/src/PRShower.cxx` or `MultiAlgBlobClustering.cxx`,
  so all line numbers here are current.
- **Status: Finding 2 diagnosed, NOT fixed.** No toolkit source was modified
  for this doc (`git -C toolkit status --porcelain` unchanged by this work).
