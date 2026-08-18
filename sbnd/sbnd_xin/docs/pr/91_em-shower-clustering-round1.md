# doc pr/91 round 1 — EM shower clustering: why one shower comes out as many, and why end points land on the neighbour

**Status.** Sections 1 and 3–9 are **diagnosis and report only** — three
`std::getenv`-gated debug probes, proven byte-neutral below, and no behaviour
change. Section **2b** adds the one fix the owner asked for on top:
`shower_endpoint_skip_orphan_vtx` — C++ default OFF, **SBND PRODUCTION ON since
2026-08-18** (owner flip), validated on a 24-event sample at the owner's
instruction (*"no need to do full blown validation yet"*). A standard-manifest
gate is still owed; see the footprint caveat in sec 2b.

Owner scope, verbatim: *"for these events, it seems that they did not go through
the EM shower clustering like what we usually do. And treated as tracks in the
PF. What happened is that when there is a EM shower, the internal structure is
very complicated, but not important, we should just cluster things together, by
directions etc. Can you examine what is the issue here? Note, the EM shower
clustering part is something we have not fully validated. This can be the first
attempt."* And on the display: *"I am not asking about the bee display, they are
OK."* — so nothing in `MultiAlgBlobClustering.cxx` / `mc.js` is touched or
proposed here.

Cross-links: pr/33 (EM shower clustering **port** audit — fidelity, 8 knobs, all
SBND ON), pr/74 (track/shower separation, K4/K5, `shower_absorb_track_guard`),
pr/84 round 3 (`shower_dedup_start_seg`, SBND ON, which §2 shows is the source
of the end-point problem).

---

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
./wcb build --notests -p                    # HEAD 7143162a + the three probes
./build/clus/wcdoctest-clus                 # 210 cases / 2132 assertions, 0 failed

cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# control arm, probes OFF
PR_JOBS=4 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr91r1-off-mc data \
  169626 174752 347129 394532
# probe arm, same binary, env vars the ONLY difference
export WCT_SHOWER_CONTENT_DEBUG=1 WCT_SHOWER_MERGE_DEBUG=1 WCT_SHOWER_ENDPOINT_DEBUG=1
PR_JOBS=4 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr91r1-dbg-mc data \
  169626 174752 347129 394532

python3 scripts/pr91_shower_content.py work-pr91r1-dbg-mc            # all tables below
python3 scripts/pr91_point_owner.py \
  work-pr91r1-dbg-mc/pr_evt169626/calib-pr-evt169626.json -4.5 157.5 442.9

# F1 attribution control: same binary + probes, dedup knob forced OFF.
# The runner's tri-state is 0/1 -- `=false` leaves the cfg default (ON).
export SBND_SHOWER_DEDUP_START_SEG=0
PR_JOBS=4 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr91r1-dedupoff-mc data \
  169626 174752 347129 394532
```

Probe byte-neutrality (member-content hashes via `abtest/hash_archive.py`, never
raw `cmp` — M2): `off` vs `dbg` over `mabc-pr.zip`, `pctree-pr-evt<ID>.tar.gz`,
`calib-pr-evt<ID>.json`, `nusel-evt<ID>.tsv` for all 4 events —
**PASS=16 FAIL=0**. Both arms were produced by the same freshly built
`build/clus/libWireCellClus.so` (mtime 2026-08-18 07:50:46, newer than the last
source edit 07:50:12 — M1). As a bonus check, `work-pr91r1-dbg-mc`'s
`mabc-pr.zip` rollups equal `work-pr84r3-dedup-mc`'s for 169626 and 394532, so
the shipped round-3 production output also reproduces from committed source at
this HEAD.

Probes added (all env-gated, all stderr, which the runner captures into each
event's `stdout.log`):

| env var | file | what it prints |
|---|---|---|
| `WCT_SHOWER_CONTENT_DEBUG` | `clus/src/NeutrinoShowerClustering.cxx` | one `SHOWER_CONTENT` block per shower: header, every member segment with its own length / charge / energy share, and every **orphan vertex** (a vertex in the shower's view that no member segment touches) |
| `WCT_SHOWER_MERGE_DEBUG` | same | `SHOWER_MERGE` at all four merge decision sites: the candidate pair, every quantity in the condition, the verdict, and a `SKIP_PASS` line when a whole pass never runs |
| `WCT_SHOWER_ENDPOINT_DEBUG` | `clus/src/PRShower.cxx` | `SHOWER_ENDPOINT`: every candidate vertex of the farthest-vertex end-point search with `touched_by_member`, the winner, and a `tag=add_shower` line per shower absorb with the node-count delta |

---

## 1. Answers to the five questions, up front

1. **174752 — "why not a single electron, instead of e⁻→γ→e⁻?"**
   The two electrons **touch**: the 34.7 MeV shower's start is **1.704 cm** from
   the 18.9 MeV shower's charge. The one pass that could merge them measures the
   distance to the parent's **start segment only**, and that number is
   **4.914 cm**, against a hard **3 cm** gate. Off by 1.9 cm on a distance that
   is measured to the wrong object. §4.
2. **169626 — "what is contained in the 567 MeV gamma?"**
   28 segments across 11 PR clusters, 144.25 cm, `kine_charge = 567.6 MeV`.
   Dominated by cluster 13 (10 segments, 76.8 cm, ≈304 MeV) and cluster 53
   (6 segments, 50.6 cm, ≈171 MeV), plus cluster 52 (4 seg, 9.7 cm, ≈57 MeV) and
   nine sub-cm fragments in seven other clusters. Full table in §5.
3. **169626 — "why is the 107 MeV electron's end point on a different EM shower?"**
   Because it is not a point on that shower at all. `get_end_point()` is a
   farthest-**vertex** search over the shower's view, and
   `shower_dedup_start_seg` (pr/84 r3, **SBND ON today**) imported a foreign
   vertex into that view. §2 — this is a production finding, and the leading
   result of the round.
4. **394532 — "the end points of the 30 MeV electron and the 66 MeV shower are
   at each other's EM shower."** Exactly the same mechanism, and the reason they
   swap: the 30 MeV shower imported vertex 8000 (cluster 8 = the 66 MeV
   shower's start), the 66 MeV shower imported vertex 39029 (cluster 39 = the
   30 MeV shower's charge). §2.
5. **347129 / 394532 — "two EM showers, 4 gammas inside the electron", "one big
   EM shower".** The nesting is display parenting of showers that are genuinely
   separate objects: the four candidates sit **23.3 / 23.3 / 41.0 / 63.1 cm**
   from the electron's start segment, so the 3 cm gate is not marginal there.
   394532's 66 MeV shower is the one exception — it passes the distance gate at
   1.5 cm and then fails on **direction**, 51.3° and 81.7° against 15°. §4.

One structural fact underpins all of it: **`examine_merge_showers`, the only
pass in the chain whose job is merging showers, does nothing in any of the four
events.** It emits `SKIP_PASS reason=no_conn2_at_main_vertex n_type1=1 n_type2=0`
in 169626, 174752, 347129 and 394532 alike. §6.

---

## 2. F1 — a shower's end point can be a vertex it does not own

### The mechanism, in three steps

1. **`Shower::set_start_vertex` calls `this->add_vertex(vtx)`**
   (`clus/src/PRShower.cxx:174`). The start vertex therefore lives in the
   shower's `TrajectoryView` node set even though no member segment reaches it.
   This is a *known, documented* toolkit/prototype divergence: the prototype's
   `WCShower::fill_sets` reads `map_vtx_segs`, which never holds the start
   vertex (`WCShower.cxx:547`), and the toolkit records that in
   `clus/inc/WireCellClus/PRShower.h:178-186` under doc pr/38. For a conn-3
   shower built by `shower_clustering_in_other_clusters`, that start vertex is
   the *nearest main-cluster or in-shower vertex* — i.e. routinely a vertex in
   somebody else's cluster.
2. **While the shower owns it, it is invisible.** `get_end_point()` is a
   farthest-vertex scan (`PRShower.cxx:1184-1197` and `:1296-1307`), and SBND
   runs with `shower_endpoint_exclude_start_vertex = true`
   (`wct-pr-perevt.jsonnet:1120`), which skips exactly that vertex.
3. **`Shower::add_shower` hands it to a shower that does not own it.** Its node
   loop is unconditional (`PRShower.cxx:355-370`): *every* vertex of the
   absorbed shower joins the absorber, orphans included. In the absorber the
   imported vertex is no longer the start vertex, so the exclusion in step 2 no
   longer protects it, and it competes in — and often wins — the farthest-vertex
   search.

`shower_dedup_start_seg` (pr/84 round 3, SBND ON since toolkit `7143162a`) is
the caller that made step 3 routine: it exists to absorb a twin shower onto one
start segment, and the twin is exactly the conn-3 object carrying a foreign
start vertex.

### Measurement — six absorbs, five wrong end points, four events

`SHOWER_ENDPOINT tag=add_shower` and `ORPHAN_VTX` / `WINNER` lines,
`work-pr91r1-dbg-mc`:

| evt | shower (node) | absorbed | imported orphan vertex | its cluster | dis from start | became the end point? |
|---|---|---|---|---|---|---|
| 169626 | 0 (`e- 107`, node 22024, cl 22) | sid 1 | **13012** | 13 — the 567 MeV shower's charge | 63.682 cm | **YES** (best member vertex was 58.5 cm) |
| 174752 | 0 (`e- 18`, node 48010, cl 48) | sid 2 | **14001** | 14 — the 34.7 MeV shower's start vertex | 4.914 cm | **YES** |
| 347129 | 0 (`e- 156`, node 53021, cl 53) | sid 1 | **11000** | 11 — the 63.9 MeV shower's start | 63.645 cm | no (a member vertex was 82 cm out) |
| 347129 | 14 (`e- 63`, node 11000, cl 11) | sid 15 | **53031** | 53 — the main cluster | 67.806 cm | **YES** (this shower has ONE 13.6 cm segment) |
| 394532 | 0 (`e- 30`, node 39023, cl 39) | sid 1 | **8000** | 8 — the 66 MeV shower's start | 9.354 cm | **YES** |
| 394532 | 8 (`e- 66`, node 8033, cl 8) | sid 9 | **39029** | 39 — the 30 MeV shower's charge | 14.776 cm | **YES** |

Node-count deltas from the same probe: 169626 `2 -> 13`, 174752 `2 -> 4`,
347129 `3 -> 14` and `2 -> 3`, 394532 `3 -> 4` and `2 -> 3`.

### The knob-off control — F1 does not exist without the dedup

Third arm, `work-pr91r1-dedupoff-mc`: same binary, same probes, only
`SBND_SHOWER_DEDUP_START_SEG=0` added (the runner's tri-state is `0`/`1`, not
`false` — an earlier `=false` attempt silently ran with the knob still ON, and
that mislabelled arm `work-pr91r1-nodedup-mc` is kept but must not be read as a
knob-off record). Sentinel proof it took: `pr84 shower_dedup` fires **0** times
per event, against 2/2/3/3 in the probe arm. With the knob off:

- orphan vertices that are **not** the shower's own start vertex: **0**
- end points with `touched_by_member=0`: **0**
- `Shower::add_shower` calls of any kind: **0**

So in these four events `shower_dedup_start_seg` is the *only* caller of
`add_shower` — `examine_merge_showers`, the `in_other_clusters` absorb and the
`examine_showers` absorb all decline (§4/§6) — and F1 appears **only** with the
knob on. Scope that honestly: this establishes the attribution *for these
events*, not that no other caller could ever import an orphan. In particular
`examine_showers`' retarget calls `set_start_vertex(main_vertex, 1)`, which
would demote a shower's previous start vertex to a plain orphan by the same
step-2 logic; that path did not fire here and is untested.

The 347129 shower-14 row is the starkest: a **single 13.6 cm segment** in
cluster 11 reports an end point **67.8 cm** away in cluster 53, because that is
where the vertex it inherited happens to be.

The control is in the same dumps: every *other* shower's orphan vertex carries
`is_start_vtx=1` and is correctly skipped. Only the six dedup survivors have an
orphan that is not their own start vertex, and exactly those five report an end
point with `touched_by_member=0`.

### What this means, and what is NOT claimed

- The reported end point is **display and geometry only** — `get_end_point()`
  feeds the PF node's `data.end`, `examine_showers`' angle tests and
  `cal_dir_3vector` callers. It does **not** enter `kine_charge`,
  `kine_energy_particle` or `kine_reco_Enu`, all of which sum over member
  segments. So the round-3 energy fix stands: no energy number here is wrong
  because of F1.
- It is nonetheless wrong, it is **in SBND production today**, and it is what
  the owner saw in three separate events.
- Whether the pr/84 r3 **keeper rule** should also change is a second, separate
  question this raises. It sorts connection type ascending *first*, so in 169626
  the survivor is the **1-segment conn-1 88 MeV stub** and it absorbs the
  6-segment conn-3 view. Where "most direct" and "fullest" disagree, round 3
  chose direct. **Both readings are on the table; no pick is made here.** Note
  the two questions are independent — fixing the orphan import does not require
  touching the keeper rule, and vice versa.

Three candidate remedies, all default-OFF knobs, **none implemented**:

- **R1 (narrowest)** — in the farthest-vertex search, skip any vertex no member
  segment touches. Purely local to `calculate_kinematics`; restores "the end
  point is on the shower".
- **R2** — in `add_shower`, import only vertices incident to a segment being
  imported (i.e. drop the unconditional node loop's orphans). Closer to the
  prototype's `map_vtx_segs` semantics; wider blast radius, since the view's
  node set is read by `update_shower_maps` and by `examine_shower_1`'s
  `shower_vertices` test.
- **R3 (root)** — stop `set_start_vertex` adding the start vertex to the view at
  all, matching the prototype. This is the documented divergence at
  `PRShower.h:178-186`; the existing `exclude_start_vertex` plumbing shows the
  size of the surface it touches.

R1 is the one whose footprint is plausibly small enough to gate on the standard
manifest; R2 and R3 are not census-boundable and would need a full A/B.

---

## 2b. F1 FIXED behind `shower_endpoint_skip_orphan_vtx` (DEFAULT OFF)

Owner's call after reading §2: *"This is a easy fix that you can work out
first… no need to do full blown validation yet. Just validate with smaller
sample."* Remedy **R1** implemented — the narrow one, acting at the point of
use rather than on the view.

**The change.** `Shower::calculate_kinematics` gains a trailing
`bool endpoint_skip_orphan_vertices = false`. When set, both farthest-vertex
searches (`PRShower.cxx:1197`, `:1309`) skip any node no member segment
touches; the touched set is built once per call from `ordered_edges`, and is
not even built when the knob is off. `calculate_kinematics_long_muon` needs
nothing — it already restricts candidates to `muon_vertices_by_index`.

Threaded the standard way: `m_shower_endpoint_skip_orphan_vtx{false}` in
`NeutrinoPatternBase.h` and `TaggerCheckNeutrino.h`, `configure` read,
`default_configuration` round-trip, pass-through, all 11 `calculate_kinematics`
call sites, `cfg/pgrapher/common/clus.jsonnet` signature + key suppression,
sbnd `clus.jsonnet` (2 param blocks + 2 pass-throughs), and
`wct-pr-perevt.jsonnet` **at `false`**. Runner env
`SBND_SHOWER_ENDPOINT_SKIP_ORPHAN_VTX` (tri-state `0`/`1`). Doctest pin added.

**Proofs.**

- Doctests: 210 cases / 2134 assertions, 0 failed.
- Freshness (M1): `build/clus/libWireCellClus.so` 08:12:23 vs last source edit
  08:10:49.
- Compiled config (M6): knob-off `wcsonnet` output **byte-identical** to the
  pre-change compile with the runner's own `pipeline_names`; knob-on shows
  `"shower_endpoint_skip_orphan_vtx" : true` at line 617.

**V1 — knob off is byte-identical to shipped production.** 24-event pr/84 r3
manifest (`work-pr91r2-off-{mc,nue,ncpi0}`) vs `work-pr84r3-dedup-*`,
member-content hashes over `mabc-pr.zip` + the pctree tarball:
**PASS=48 FAIL=0.**

**V2 — knob on moves exactly the end points, nothing else.**
`work-pr91r2-on-*` vs `work-pr91r2-off-*`, same 24 events:
**PASS=43 FAIL=5** — and the five are `mabc-pr.zip` only. **Every pctree
tarball is byte-identical**, and within each moving zip the only member that
changed is `data/0/0-mc.json`. `nusel-evt<ID>.tsv` is byte-identical for all
five, i.e. **zero score or label movers**.

Six showers changed, in five events, and `end` is the only field that moved on
any of them. `kine_reco_Enu` and the whole `kine_energy_particle` list are
byte-identical in all five — the prediction in §2 that this is geometry-only
holds exactly:

| evt | shower (node) | conn | nseg | end before | end after | moved |
|---|---|---|---|---|---|---|
| 169626 | 0 (22024) | 1 | 6 | (−15.78, 118.73, 461.16) — cl 13 | (−32.35, 96.75, 410.27) = vtx 57074, its own member | 57.86 cm |
| 174752 | 0 (48010) | 1 | 2 | (−57.06, −191.10, 411.12) — cl 14 | (−62.05, −195.73, 408.10) = vtx 48010, its own | 7.45 cm |
| 347129 | 14 (11000) | 3 | 1 | (−192.29, −38.06, 184.47) — cl 53 | (−196.62, −107.29, 179.89) = vtx 11001, the far end of its ONE segment | 69.51 cm |
| 394532 | 0 (39023) | 1 | 2 | (192.79, −182.39, 26.71) — cl 8 | (201.18, −191.20, 17.05) = vtx 39031, its own | 15.54 cm |
| 394532 | 8 (8033) | 3 | 1 | (199.93, −190.25, 16.44) — cl 39 | (198.24, −187.84, 21.53) = vtx 8001, its own | 5.88 cm |
| 168596 | 6 (128147) | 2 | 3 | (−177.36, 40.71, 175.58) | (−190.83, 57.07, 295.12) | 121.41 cm |

Independent cross-check: 394532's two repaired end points are **exactly** the
values the pre-dedup arm `work-pr91r1-dedupoff-mc` produces —
`(201.18, −191.20, 17.05)` and `(198.24, −187.84, 21.53)`. The fix restores the
geometry the dedup had perturbed, rather than inventing a third answer.

### 168596 — F1 has a second source, and it is not the dedup

nueCC **168596** was not one of the four reported events and is the round's
most useful surprise. `pr84 shower_dedup` fires **0** times there, and the two
`add_shower` calls in the event both target shower 2, not shower 6. Yet
shower 6 carried orphan vertex **14051** (same cluster, 85.5 cm from its start)
and reported an end point 121 cm away.

Provenance: shower 6 has `pio_id = 0`, `pio_mass = 114.06`, and
`start_vertex_id = 14052` — it was **re-seated onto the π⁰ vertex by
`id_pi0_with_vertex`**, whose `set_start_vertex(pi0_vertex, 2)` demotes the
shower's previous start vertex to a plain orphan by exactly the step-2 logic in
§2. So the general statement is: **any pass that re-seats a shower's start
vertex leaves the old one behind as an orphan**, and `shower_dedup_start_seg`
is the loudest but not the only such pass. R1 catches all of them because it
acts where the vertex is *used*, not where it is introduced. (This is precisely
the untested path flagged in §2's control-arm note; it is now measured.)

It also means the knob's footprint is **not** bounded by the dedup's: 1 of the
20 events we did not hand-pick moved. Twenty events is far too small to
estimate a rate — that is the gate this round did not run, and it is what a
flip decision needs.

**FLIPPED — SBND PRODUCTION ON, 2026-08-18** (toolkit `4ff9870f`), owner's
call after scanning the Bee pair below: *"This is good, you can flip is on this
for SBND production for now."* The C++ default stays `false`; only the SBND
operating point turns it on (the `v3_extension_guard` idiom).

**V3 — bare-run production composition.** Re-ran the same 24 events with **no
`-A` override at all**, so the cfg flip is the only thing in play, and gated
against the knob-on arm: **PASS=48 FAIL=0**. The SBND operating point remains
the single source of truth (doc 68: a bare run *is* production). Compiled-config
proof: the bare compile now carries `"shower_endpoint_skip_orphan_vtx" : true`
and is byte-identical to the earlier explicit knob-on compile.

**Footprint caveat, on the record in the cfg comment and here.** The 24 events
were chosen for this defect, so V2 is a targeted sample, not a population gate.
168596 — the one mover nobody hand-picked — shows the population is not the
dedup's. **1 of the 20 non-hand-picked events moved**: a footprint signal, not
a rate. A standard-manifest gate is still owed and is the obvious next round.

### Bee sets for the owner's scan

Same 24 events and the same order as the pr/84 round-3 sets, so `idx N` is the
same event in both rounds.

- **BEFORE** (knob off = shipped production today):
  `https://www.phy.bnl.gov/twister/bee/set/e81dfbf9-3801-46a4-ad61-31b5511127f1/event/list/`
- **AFTER** (`shower_endpoint_skip_orphan_vtx = true`; **this is production as
  of the 2026-08-18 flip**):
  `https://www.phy.bnl.gov/twister/bee/set/04107fda-6306-44e0-aa8c-12a352e32235/event/list/`

Five differ — **idx 5 (347129), 6 (169626), 7 (174752), 11 (394532), 14
(168596)** — and in each the only visible change is where a PF node's drawn
line stops. Per-event notes in `bee/pr91r2/pr91r2.index.txt`.

---

## 3. F2 — one physical EM shower split across several `PR::Shower` objects

PF node ids are `cluster_id*1000 + seg_id`, so the split is readable straight
off the Bee tree. In all four events the fragments live in **different PR
clusters** from the electron:

| evt | showers at / near the main vertex (cluster: energy) |
|---|---|
| 169626 | 22: e⁻ 107 · 42: 13 · 52: 34 · 53: **567** · 39: 10 |
| 174752 | 48: e⁻ 18 · 14: 34 |
| 347129 | 53: e⁻ 156 · 51: 89 · 52: 23 · 54: 75 · 11: 63 · 39: 6 |
| 394532 | 39: e⁻ 30 · 8: 66 · 31: 18 · 38: 82 · 40: 46 |

The `gamma NN MeV` nodes between them are **synthetic carriers** — one per
shower the PF writer cannot reach directly, energy copied from the child. They
are a symptom of this split, not objects; per the owner's instruction the
display is left exactly as it is.

**Not every split is wrong.** 169626 is, as the owner says, a classic NCπ⁰: the
107 MeV object and the 567 MeV object are the two converted gammas, 38.8 cm
apart, and they *should* be two showers. The code even pairs them —
`kine_pio_flag = 2`, `mass = 212.98 MeV`, `E1 = 107.87`, `E2 = 567.61`,
`angle = 49.44°`. Likewise 347129's four candidates at 23–63 cm. The split that
is genuinely wrong is 174752's, where the two touch at 1.7 cm.

---

## 4. Why the merges did not happen — the gate table

From `SHOWER_MERGE tag=ex_shower1_merge`, the second half of `examine_shower_1`
(`clus/src/NeutrinoShowerClustering.cxx:2684-2790`), which is the only pass that
can absorb a **conn-3** shower into the main conn-1 shower:

| evt | candidate (node) | conn | len | kine | `min_dis` to parent's START SEGMENT | angle | angle1 | rejected by |
|---|---|---|---|---|---|---|---|---|
| **174752** | 14000 | 3 | 8.88 | 34.74 | **4.914** | — | — | `conn>2 && min_dis > 3 cm` |
| 169626 | 42032 | 3 | 3.67 | 13.21 | 59.980 | — | — | same |
| 169626 | 52042 | 3 | 9.67 | 34.35 | 73.909 | — | — | same |
| 169626 | 53056 | 3 | 144.25 | 567.61 | 38.774 | — | — | same |
| 347129 | 51011 | 3 | 23.27 | 89.83 | 23.318 | — | — | same |
| 347129 | 52020 | 3 | 7.33 | 23.17 | 23.318 | — | — | same |
| 347129 | 54027 | 3 | 16.33 | 75.30 | 41.033 | — | — | same |
| 347129 | 11000 | 3 | 13.60 | 63.90 | 63.104 | — | — | same |
| 394532 | 38013 | 3 | 19.58 | 82.21 | 8.842 | — | — | same |
| 394532 | 40032 | 3 | 10.93 | 46.37 | 31.244 | — | — | same |
| **394532** | 8033 | 3 | 10.49 | 66.04 | **1.503** (passes) | **51.325** | **81.724** | `angle < 15 && angle1 < 15` |

Two distinct failure modes, and the distinction matters:

- **174752 is a measurement-target bug, not a threshold that is merely tight.**
  `min_dis = shower_get_closest_dis(shower1, shower->start_segment())` measures
  to the parent's **start segment** (48010, a 3.98 cm e⁻ piece), not to the
  parent's charge. `scripts/pr91_point_owner.py` on the 34.7 MeV shower's start
  point gives the full ranking: **1.704 cm** to segment 48009 — a member of the
  same parent shower — and **4.914 cm** to the start segment 48010. The two
  showers are touching; the gate is looking at the wrong piece of the parent.
- **394532's 66 MeV shower fails on direction, not distance.** It is 1.5 cm from
  the parent's start segment and 9.4 cm from the parent's own charge, but the
  30 MeV electron runs along `(0.06, −0.53, −0.84)` while the 66 MeV shower runs
  along `(0.56, −0.61, −0.56)` — they open a wide V at the vertex. Merging them
  would be a physics decision (are these two legs of one conversion, or two
  objects?), not a bookkeeping fix. Reported, not proposed.

The other two merge sites, from the same dumps:

- **`examine_merge_showers` never runs** —
  `SKIP_PASS reason=no_conn2_at_main_vertex n_type1=1 n_type2=0` in all four
  events. It only classifies conn-1 and conn-2 showers at the main vertex, and
  in these events there are **zero conn-2 showers**, so the pass returns before
  looking at any geometry.
- **`shower_clustering_in_other_clusters`' absorb points downstream only.** Its
  `dir_shower` is the *new* (downstream) shower's axis and `angle1` is measured
  to the vector from that shower's vertex toward the candidate, so an existing
  shower sitting upstream fails by construction. Measured: 174752 shower 2 vs
  shower 0 at `angle = 56.5°`; 169626 shower 4 (the 567 MeV) vs shower 0 at
  `angle = 50.3°, angle1 = 133.5°`. Its per-segment cone does fire — it is what
  pulled 54059/55060/56061/57062/59064 into the 107 MeV shower at 7–13° — but
  only for loose segments, never to reunify two showers.

---

## 5. What is inside the 567 MeV EM shower (169626, shower_id 4, node 53056)

conn-3, start vertex 13001, start `(−11.896, 129.219, 463.269)`, 28 segments,
11 clusters, 144.25 cm, `kine_best = kine_charge = 567.612 MeV`. `E_est` is the
member's charge share of that total.

| cluster | segs | length | ≈E |
|---|---|---|---|
| 13 | 10 | 76.8 cm | 303.9 MeV |
| 53 | 6 | 50.6 cm | 171.5 MeV |
| 52 | 4 | 9.7 cm | 57.1 MeV |
| 42 | 1 | 3.3 cm | 15.7 MeV |
| 47 | 1 | 0.9 cm | 5.8 MeV |
| 51 | 1 | 0.7 cm | 5.3 MeV |
| 37 | 1 | 0.9 cm | 2.2 MeV |
| 46 | 1 | 0.3 cm | 2.1 MeV |
| 44 | 1 | 0.3 cm | 1.8 MeV |
| 41 | 1 | 0.3 cm | 1.2 MeV |
| 43 | 1 | 0.3 cm | 0.9 MeV |

The four largest members are `13018` (10.3 cm, 54.4 MeV), `53054` (18.0 cm,
54.8 MeV), `13013` (14.4 cm, 38.9 MeV) and `53052` (10.0 cm, 35.6 MeV). Seven of
the eleven clusters contribute a single sub-cm fragment — together ≈19 MeV, 3 %
of the shower. The per-segment table is in
`scripts/pr91_shower_content.py work-pr91r1-dbg-mc --only 169626`.

For comparison, the `e- 107 MeV` object is one 28.7 cm segment in cluster 22
(96.9 MeV) plus five sub-cm fragments in clusters 54/55/56/57/59 (≈11 MeV) — so
of the two π⁰ gammas, one is reconstructed as a single stem and the other as a
144 cm, 11-cluster cascade.

### The point (−4.5, 157.5, 442.9)

`scripts/pr91_point_owner.py`, scanning every **fitted trajectory point** (not
endpoints — endpoints missed this the same way they missed 285567's point):

```
  d_traj     seg  clus      len    pdg  shower flag_shower  pt_idx  npts  owner
   1.564   53057    53   14.222   2212      -1       False      13    25  IN NO SHOWER
   7.978   53056    53    5.877     11   53056        True      10    11  shower 53056 conn=3 567.6MeV
```

The point is **1.56 cm from segment 53057**: a 14.22 cm, proton-PID
(`particle_score = 100`), `flag_shower = false` segment in cluster 53 whose
`shower_id` is **−1** — it is in **no shower at all**, and it is the only such
segment in the event. It is not remote: its vertex 53061 is segment 53056's own
end vertex, i.e. it hangs directly off the 567 MeV shower's stem.

Why it is in no shower, from the same dumps:

1. The flood-fill that built shower 4 stops there.
   `shower_absorb_track_guard` is **SBND ON**
   (`wct-pr-perevt.jsonnet:1217`, pr/74 F12) and refuses a segment with a
   confident non-electron PDG that is a straight long track
   (`PRShower.cxx:458-470`). A 14 cm proton at score 100 is precisely its target.
2. The cone absorber never even offers it to shower 4:
   `shower_clustering_in_other_clusters` skips candidates in the *same cluster*
   as the shower's start segment (`NeutrinoShowerClustering.cxx:2027`), and
   53057 is cluster 53 like 53056. The other showers did see it and rejected it
   on angle — `new_sid=1 angle=116.0 dis=34.6`, `new_sid=2 angle=107.6
   dis=37.4`, `new_sid=3 angle=80.9 dis=64.1`.
3. Having no shower, it gets no PF node: the writer emits shower nodes plus
   main-cluster BFS-reachable track segments, and cluster 53 is not the main
   cluster (22 is).

So this is a **designed** exclusion doing its job (keep a proton out of an EM
shower) with an undesigned consequence (14 cm of charge invisible in the PF
tree). Whether a proton hanging off a shower stem should appear as a PF track
node is a display question the owner has ruled out of scope for now; it is
recorded here so it is not rediscovered.

---

## 6. Prototype parity — none of §3/§4 is a porting divergence (M15)

Every gate in §4 is line-faithful to `prototype_base`:

| gate | toolkit | prototype |
|---|---|---|
| `examine_merge_showers` runs **before** `shower_clustering_in_other_clusters`, so conn-3/4 showers do not exist yet when it runs | `NeutrinoShowerClustering.cxx:4088` vs `:4096` | `NeutrinoID_shower_clustering.h:291` vs `:295` |
| conn-1 ← conn-2 only, at the main vertex, single 10° test on 100 cm directions | `:1820-1848` | `:380-426` |
| conn-3 admitted only at `min_dis ≤ 3 cm` **to the parent's start segment**, only when the first half added nothing, only the top-energy group | `:2746`, `:2684` | `NeutrinoID_em_shower.h:561-592` |
| `in_other_clusters` absorb is downstream-only (25°/80 cm, 12.5°/120 cm) | `:2089-2093` | `:1533-1553` |

That is the resolution of the apparent contradiction with pr/33: pr/33 audited
this stage for **fidelity** and passed it, correctly. The behaviour is a shared
latent gap in the prototype and the port together, so any fix here is a
deliberate improvement over the prototype and belongs behind a knob — the same
posture pr/84 round 3 took for its own shared-hole fix.

F1 (§2) is the exception: `set_start_vertex` → `add_vertex` is a **toolkit-only**
divergence, already documented at `PRShower.h:178-186` (doc pr/38), and
`add_shower`'s unconditional node union has no prototype counterpart because
the prototype's `WCShower::add_shower` merges `map_vtx_segs`, which never held
the start vertex in the first place.

---

## 7. Fix options (proposals only — nothing implemented, no footprint claimed)

Ordered by how well the evidence supports them.

- **P1 — end point on the shower (F1).** R1/R2/R3 in §2. R1 is the narrow one.
  This is a real defect with a measured mechanism and it is live in production;
  it is the one item here that has an unambiguous "correct" answer.
- **P2 — measure the conn-3 admission distance to the parent's charge, not its
  start segment.** Would turn 174752's 4.914 cm into 1.704 cm and admit the pair
  to the geometry test it currently never reaches. A knob like
  `ex_shower1_conn3_dis_to_shower` (bool) is the minimal form; raising the 3 cm
  number instead is the cruder alternative and would sweep in 394532's 8.8 cm
  and 347129's 23 cm candidates too. **Blast radius: not census-boundable** —
  every event with a conn-3 shower near the main shower is a candidate, so this
  needs a full A/B plus a hand scan, not the round-3 style bounded census.
- **P3 — a late direction-based consolidation pass**, i.e. what the owner
  described ("just cluster things together, by directions"). Structurally this
  means running a merge *after* `shower_clustering_in_other_clusters` instead of
  before it, so conn-3 showers are eligible at all (G-A). The evidence says such
  a pass must be conservative: in these four events it would have to merge
  174752's pair and leave 169626's π⁰ gammas, 347129's four candidates and
  394532's wide-V pair alone. Angle alone does not separate those cases —
  169626's 567 MeV shower is 50.3° from the electron and 347129's are 23–63 cm
  away, but 394532's 66 MeV is 1.5 cm away at 51°. **Recommend not attempting
  P3 until P2 is measured**, because P2 is a strictly smaller change that
  addresses the only unambiguously-wrong split in the sample.

---

## 8. What this round does NOT claim

- No determinism check was run on this stage (pr/33 left that open and it stays
  open).
- No A/B gate on a physics change — there is no physics change.
- **No population footprint.** Four events, hand-picked by the owner from a Bee
  scan. Nothing here supports a statement about how often F1 or the 3 cm gate
  fires across a sample, and in particular §2's table must not be read as
  "5 in 4 events" — the four events were selected *because* they looked wrong.
- No claim about `kine_reco_Enu`: F1 does not touch it, and the round-3 energy
  numbers are unaffected.
- The 394532 wide-V pair and 347129's four candidates are **reported, not
  judged** — whether they should be one object is a physics call for the owner.
- 285567's disconnected main-cluster component (deferred graph round) and
  168596's π⁰ energy-source inconsistency (M15-open) are untouched.

---

## 9. Records

- Arms: `work-pr91r1-off-mc` (probes off), `work-pr91r1-dbg-mc` (probes on),
  `work-pr91r1-dedupoff-mc` (probes on + `SBND_SHOWER_DEDUP_START_SEG=0`); the
  earlier `work-pr91r1-nodedup-mc` used `=false` and did NOT disable the knob —
  169626 / 174752 / 347129 / 394532, hub `work-mcp1k-cb0805`, `data` reality.
- Scripts: `scripts/pr91_shower_content.py`, `scripts/pr91_point_owner.py`.
- Toolkit commits: `77353cf4` (the three env-gated probes, no behaviour
  change) and the §2b knob commit.
- Gate arms: `work-pr91r2-{off,on}-{mc,nue,ncpi0}` (24 events); V1 PASS=48/48
  vs `work-pr84r3-dedup-*`, V2 PASS=43/48 with the 5 movers enumerated in §2b.
  `work-pr91r2-dbg168596-nue` carries the probe trace for the pi0-re-seat path.
- Permanent probes now in the tree: `WCT_SHOWER_CREATE_DEBUG` (pr/84 r3),
  `WCT_SHOWER_CONTENT_DEBUG`, `WCT_SHOWER_MERGE_DEBUG`,
  `WCT_SHOWER_ENDPOINT_DEBUG` (this round).
- Round 2 (§10): `work-pr91r3-probe168596-nue`, hub `work-nuecc48-cb0805`,
  toolkit `4ff9870f`, probe-neutrality 4/4 vs `work-pr91r2-on-nue`.
- Round 2 (§10): `work-pr91r3-probe168596-nue`, hub `work-nuecc48-cb0805`,
  toolkit `4ff9870f`, probe-neutrality 4/4 vs `work-pr91r2-on-nue`.


---

## 10. Round 2 — 168596: why the pieces inside the 2039 MeV electron were never clustered into it

Owner question (2026-08-18), after §2b shipped: *"this electron which is inside
the big EM shower should be clustered by the big EM shower; in this event there
are 2 pieces not clustered, can you explain why?"*

**Repro.** Fresh label, probes on, one event:

```
cd wcp-porting-img/sbnd/sbnd_xin
WCT_SHOWER_CREATE_DEBUG=1 WCT_SHOWER_CONTENT_DEBUG=1 \
WCT_SHOWER_MERGE_DEBUG=1 WCT_SHOWER_ENDPOINT_DEBUG=1 \
PR_JOBS=1 PR_EXTRA_STAGES=pr_display \
./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr91r3-probe168596-nue data 168596
```

Toolkit `4ff9870f` (= SBND production). Right-binary proof: the run emitted 20
`SHOWER_CREATE_DEBUG`, 112 `SHOWER_CONTENT`, 36 `SHOWER_MERGE` and 291
`SHOWER_ENDPOINT` lines (the installed lib had to carry the probe format
strings). Probe neutrality vs the production arm `work-pr91r2-on-nue`:
`calib-pr-evt168596.json` and `nusel-evt168596.tsv` byte-identical, `mabc-pr.zip`
and `pctree-pr-evt168596.tar.gz` PASS under `hash_archive.py` member hashes —
**4/4, no output moved**.

### 10.1 Every shower that was NOT absorbed into the 2039 MeV electron

`examine_shower_1`'s second half is the only pass that can pull an existing
shower into the main conn-1 shower. All five non-trivial candidates and their
verdicts, from one probe run (`NeutrinoShowerClustering.cxx:2734-2772`).
Distances are per associate point against the big shower's 5436-point cloud.

| candidate | displayed | conn | own len | median dist | frac < 3 cm | **verdict** | margin |
|---|---|---|---|---|---|---|---|
| `14072` | `e- 21 MeV` (the fake π⁰ γ) | 1 | 7.70 cm | **1.88 cm** | **0.96** | `conn1_vtx_outside_parent` | graph, not geometric |
| `123140` | `e- 14 MeV` | 2 | **2.85 cm** | 16.82 cm | **0.50** | `len_lt_3cm` | **1.5 mm** short |
| `130160` | `e- 23 MeV` | 2 | 4.58 cm | 6.69 cm | 0.00 | `geometry_fail` | angle1 88.8° vs 15° |
| `128147` | `e- 33 MeV` (the real π⁰ γ) | 2 | 9.80 cm | 12.00 cm | 0.00 | `geometry_fail` | min_dis 85.8 cm vs 28 |
| `69107` | `e- 6 MeV` | 3 | 2.47 cm | 7.16 cm | 0.00 | `len_lt_3cm` | — |

Only two of the five are actually *inside* the big shower's charge: `14072`
(96 % of its points within 3 cm) and, partially, `123140` (50 %). **Those are
the two this section reads as the owner's "2 pieces"** — but note `123140`'s
cloud is bimodal (median 16.8 cm), so if the intended pair was instead
`14072` + `128147` (the π⁰'s two daughters), `128147`'s verdict is the
`geometry_fail` row above and its 85.8 cm stand-off is genuine, not a bug.

`123140` misses a hard length cut by **1.5 mm** (2.852 vs 3.000 cm) even though
its start vertex 14051 *is* a node of the big shower — a conn-2 shower hanging
directly off the parent, killed before any geometry is looked at.

The rest of §10 is about `14072`, whose rejection is not geometric at all.

### 10.2 The mechanism: a vertex that is in the view but was never walked

`shower_clustering_with_nv_in_main_cluster` (`:512-576`) walks out from the main
vertex; on each branch the first shower-flagged segment roots its own shower and
the walk stops there. Four showers were seeded in the main cluster:

```
SHOWER_CREATE_DEBUG site=nv_in_main_cluster shower_id=0 start_seg=14060 conn=1 start_vtx_gidx=17
SHOWER_CREATE_DEBUG site=nv_in_main_cluster shower_id=1 start_seg=14042 conn=1 start_vtx_gidx=27
SHOWER_CREATE_DEBUG site=nv_in_main_cluster shower_id=2 start_seg=14094 conn=1 start_vtx_gidx=27
SHOWER_CREATE_DEBUG site=nv_in_main_cluster shower_id=3 start_seg=14072 conn=1 start_vtx_gidx=52
```

Shower 2 — the future 2039 MeV electron — was born at **vertex 14027**.
`Shower::set_start_vertex` calls `this->add_vertex(vtx)` (`PRShower.cxx:177`),
so 14027 entered the view at that instant. The initial
`complete_structure_with_start_segment` then explicitly refuses to enqueue
`m_start_vertex` (`PRShower.cxx:479-486`, `:526`, `:532`), and the other three
segments at 14027 (`14042`, `14093`, `14070`) were all already in the shared
`used_segments`. **14027 is therefore a node of the view that was never
expanded.**

Later, `examine_showers` merges the 31.8 cm stem `14168` into shower 2, retargets
it (`:3114-3119`) and re-floods with a **fresh** used-set:

```cpp
shower->add_segment(sg);
shower->set_start_vertex(main_vertex, 1);
shower->set_start_segment(sg);
IndexedSegmentSet tmp_used_segments;                       // fresh — not the shared set
shower->complete_structure_with_start_segment(tmp_used_segments, ...);
```

The fresh set is why `14168`, `14093` and `14174` — all consumed as track
segments by the seeding BFS — *are* members of shower 2 in the final output.
`used_segments` does **not** permanently bar a segment. What bars it is the
**vertex frontier**: the walk enqueues a vertex only when
`!this->has_node(v)` (`PRShower.cxx:527`, `:533`). Processing segment `14093`
the walk sees its two vertices 14017 (already a node) and **14027 (already a
node — put there by `set_start_vertex` at creation, never expanded)** and
enqueues neither. The 4.74 cm proton-PID segment `14070` that hangs off 14027
is never examined, so vertex **14052** — and with it `14072` and `14095` —
never enter the view.

Measured, from the content probe (shower 2: 60 segments, 96 distinct vertices
touched, **0 orphan nodes**):

- `14070` member? **no**.  `14095` member? **no**.
- vertex 14051 in the view? **yes**.  vertex **14052 in the view? no**.
- nearest view vertex to 14052: **14027, at 4.46 cm**.

Everything downstream follows. Two separate gates ask the same pure
graph-membership question and both reject `14072`:

- `examine_showers`' absorb block (`:3138`) — `conn_type1 == 1 && start_vtx1 != main_vertex && shower_vertices.count(start_vtx1)`; this is how showers 0 and 1 (start vertices 14017 and 14027, both nodes) *were* absorbed.
- `examine_shower_1` (`:2739`) — `conn_type1 == 1 && shower_vertices.find(start_vtx1) == end()` → the `conn1_vtx_outside_parent` line above.

The stub's charge is 96 % within 3 cm of the parent's charge and its start
vertex is 4.46 cm from the parent's nearest vertex. **Neither gate consults any
of that.** `shower_absorb_track_guard` is *not* involved: it needs
`min_length = 10 cm` (`segment_is_straight_long_track`) and `14070` is 4.74 cm —
a useful negative, since §5 blamed that guard for the analogous 53057 case at
14.22 cm.

### 10.3 This is a porting divergence, and it is the pr/38 one

The frontier prune itself is faithful — the prototype does the same thing
(`WCShower.cxx:735-742`: already in `map_vtx_segs` ⇒ insert but do not enqueue).
The divergence is **what seeds that visited-set**:

| | start vertex enters the view? |
|---|---|
| prototype `WCShower::set_start_vertex` (`WCShower.cxx`) | **no** — assigns `start_vertex` and `start_connection_type` only; `map_vtx_segs` untouched |
| toolkit `Shower::set_start_vertex` (`PRShower.cxx:177`) | **yes** — calls `this->add_vertex(vtx)` |

So in the prototype 14027 is absent from `map_vtx_segs`, the re-flood *enqueues*
it, the walk continues through `14070` to 14052, and the conn-1 membership gate
at `NeutrinoID_em_shower.h:579` then **passes** — `14072` is absorbed into the
big shower and no π⁰ is ever built from it.

This is the same toolkit-only divergence already recorded in doc pr/38 and in
`PRShower.h:178-186`, where its known consequence was confined to what
`fill_sets` returns. The new finding is that it also **prunes the flood-fill
frontier**, permanently walling off a branch — a much larger blast radius than
pr/38 recorded. Per M15 this is surfaced, not fixed: it is a real divergence
whose repair changes production output and needs its own default-OFF knob.

Two candidate repairs, neither implemented, neither costed:

1. **Frontier-only fix** — let `complete_structure_with_start_segment` enqueue a
   node that has never been *expanded*, rather than one that is merely present
   (track expanded-vertices separately from view membership). Narrow, and
   restores prototype reachability without touching `set_start_vertex`.
2. **Divergence fix** — stop `set_start_vertex` adding the vertex. Closer to the
   prototype but changes `fill_sets`, every BFS barrier built from a shower's
   node set, and §2b's own orphan bookkeeping. Much wider.

Option 1 is the one to cost first.

### 10.4 What this section does not claim

- No knob, no threshold change, no flip — diagnosis only.
- Only 168596 was measured. No population estimate for either gate, and no
  claim about how often a shower's creation vertex ends up unexpanded.
- `examine_merge_showers` emitted **zero** `tag=merge_showers` lines on this
  event, so §10 says nothing about that pass here (consistent with §4: it only
  merges conn-2 into conn-1, and `14072` is conn-1).
- `examine_shower_1` entered with `flag_skip=1`, i.e. its first half was
  suppressed and only the second half — the block traced above — ran.
- Why `123140` is 2.85 cm rather than longer (a fragmentation question) was not
  investigated.
- The claim that the seeding BFS reached 14052 via `14070` is read off the
  BFS layer order in the creation trace; the probe does not print
  `used_segments` directly. It does not affect §10.2's conclusion, which rests
  on the measured view membership of 14027/14052.

### 10.5 The 3 cm cut gates shower OBJECTS, not segments

A natural misreading of §10.1 is that EM clustering refuses short pieces. It
does not. Shower 2 (the 2039 MeV electron) has **30 of its 60 member segments
shorter than 3 cm**, down to **0.09 cm**, totalling 29.48 cm — including
cross-cluster fragments (`90128` 0.09 cm, `80118` 0.16 cm, `62100` 0.30 cm …).
There are two different routes into a shower and only one of them is gated:

- **Route A — graph-connected.** `complete_structure_with_start_segment`
  (`PRShower.cxx:442`) floods the PR graph and swallows every connected segment.
  It has **no length test and no geometry test** — its only exclusions are
  `used_segments` and `shower_absorb_track_guard` (≥10 cm, non-electron). This is
  how those 30 sub-3 cm segments got in.
- **Route B — already promoted to its own `PR::Shower`.** Then only the
  shower-to-shower merge passes can attach it, and `get_total_length() < 3 cm`
  (`:2740`) applies. `get_total_length()` sums the *whole candidate shower*.

So the operative question for `123140` is not "why was it cut" but **"why was it
a separate shower object at all?"** — and the answer is that it is in **cluster
123**, never graph-connected to cluster 14, so Route A could never reach it.
`shower_clustering_with_nv_from_vertices` promoted it to a conn-2 shower seated
on main-cluster vertex 14051 (`:1548-1588`, probe site `nv_from_vertices_break`).

That promotion also **closed the other door**: `update_shower_maps` (`:239`)
rebuilds `used_shower_clusters` from every segment in every shower, and
`shower_clustering_in_other_clusters` — which runs later (`:4096`) and absorbs
free other-cluster material into showers — opens with

```cpp
if (used_shower_clusters.find(cluster) != used_shower_clusters.end()) continue;   // :1936
if (map_cluster_length[cluster] < 4 * units::cm) continue;                        // :1937
```

Cluster 123 is already claimed, so the cone absorber never considers it; and at
2.85 cm it would have failed the 4 cm cluster-length gate anyway. `123140`
therefore had exactly **one** door in the whole chain — `examine_shower_1`'s
3 cm total-length gate — and missed it by 1.5 mm.

The design logic is visible once the two routes are separated: **inside a
cluster, connectivity is the evidence, so no size gate is needed; across
clusters there is no connectivity evidence, so size and geometry stand in for
it.** A small EM fragment sitting inside a big shower but imaged into its own
cluster has neither, which is the structural gap. Nothing here is a divergence —
the prototype carries the same three `get_total_length() < 3*units::cm` tests
(`NeutrinoID_em_shower.h:581`, `NeutrinoID_shower_clustering.h:498/512`, the
latter two with its own `// too short ...` comment), and `id_pi0_with_vertex`
has no length gate in either tree.

---

## 11. Round 3 — the 168596 fix: `shower_walk_visited_parity`, SBND PRODUCTION ON

### 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
# check `pgrep -f wire-cell` first (shared tree); wcbuild + install
wcbuild
ls -la local/lib/libWireCellClus.so                    # M1 freshness: 2026-08-18 12:04:08
./build/clus/wcdoctest-clus                             # 210/210, 0 failed

cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
for s in nuecc48 ncpi0; do
  SBND_SHOWER_WALK_VISITED_PARITY=0 PR_JOBS=32 PR_EXTRA_STAGES=pr_display \
    ./run_pr_chain_batch.sh work-$s-cb0805 work-pr91r4-off-$s data
  SBND_SHOWER_WALK_VISITED_PARITY=1 WCT_SHOWER_WALK_DEBUG=1 \
  PR_JOBS=32 PR_EXTRA_STAGES=pr_display \
    ./run_pr_chain_batch.sh work-$s-cb0805 work-pr91r4-on-$s data
done
python3 scripts/pr85_hash_gate.py work-pr91r4-off-nuecc48 work-pr91r4-on-nuecc48
python3 scripts/pr85_hash_gate.py work-pr91r4-off-ncpi0   work-pr91r4-on-ncpi0
diff work-pr91r4-off-nuecc48/nusel-table.tsv work-pr91r4-on-nuecc48/nusel-table.tsv
diff work-pr91r4-off-ncpi0/nusel-table.tsv   work-pr91r4-on-ncpi0/nusel-table.tsv

# after the flip: bare-run composition gate
for s in nuecc48 ncpi0; do
  PR_JOBS=32 PR_EXTRA_STAGES=pr_display \
    ./run_pr_chain_batch.sh work-$s-cb0805 work-pr91r4-bare-$s data
done
python3 scripts/pr85_hash_gate.py work-pr91r4-on-nuecc48 work-pr91r4-bare-nuecc48
python3 scripts/pr85_hash_gate.py work-pr91r4-on-ncpi0   work-pr91r4-bare-ncpi0
```

### 1. Why the divergence exists — sourced, not open

Round 2 (sec 10.3) left the answer as "surfaced, not fixed" and flagged the
frontier prune as a much larger blast radius than doc pr/38 recorded. There is
**no recorded reason** for the divergence -- it is an unflagged authorship
artifact, not an intentional toolkit choice:

- `git log -L 172,184:clus/src/PRShower.cxx` shows `this->add_vertex(vtx)` in
  the **very first commit of the file** (`ce3f1ebd`, "Latest chapter on PR
  infrastructure"), when the method was still `Shower::start_vertex(VertexPtr)`
  with no `type` argument. It has never been touched since except to add a
  null guard (`f639218b`).
- The port audit table records the method as **"Equivalent; see §2.7
  null-check"**
  (`clus/docs/patternrecognition/prvertex_prsegment_prshower_review.md:100`);
  §2.8 raised only the missing `nullptr` guard. `porting_dictionary.md` does
  not list `set_start_vertex` at all.
- Root cause: `PR::Shower` derives from `TrajectoryView`, whose node set does
  double duty as both view MEMBERSHIP and flood-fill VISITED. The prototype
  keeps the two separate: `start_vertex` is a bare pointer,
  `map_vtx_segs` is the visited set, and
  `WCPPID::WCShower::set_start_vertex` (`WCShower.cxx:529-532`) assigns the
  pointer and connection type only.

Per M15 this makes it a *fixable* divergence, not a deliberate one.

**Prototype invariant.** Every key of `map_vtx_segs` is a vertex incident to a
member segment, inserted only by `set_start_segment(seg,map)`
(`WCShower.cxx:542-550`, explicitly skips `start_vertex`),
`add_segment(seg,map)` (`:694-701`), `add_shower` (`:681-692`, merges the
absorbed shower's own segment→vertex map) and the walk itself (`:703-755`).
The start vertex is never a key while it is the start vertex.

### 2. The fix: `shower_walk_visited_parity` (C++ default OFF)

One semantic change: *the flood-fill's frontier test becomes "has this vertex
been walked", not "is this vertex present in the view."*

`clus/inc/WireCellClus/PRShower.h` gains a private
`std::set<node_descriptor> m_walked_nodes` — the toolkit analogue of
`map_vtx_segs`' key set, membership-tested only, never iterated, so it
introduces no pointer-order dependence — and
`complete_structure_with_start_segment` gains a trailing
`bool walk_visited_parity = false`.

`clus/src/PRShower.cxx` — bookkeeping is unconditional (pure state, no
behaviour change on its own); only the frontier predicate is knob-gated:

| site | change |
|---|---|
| `set_start_vertex` | `add_vertex(vtx)` unchanged — deliberately **not** recorded in `m_walked_nodes`. This is the whole fix. |
| `set_start_segment` | records the two endpoints it adds (already skips `m_start_vertex`) |
| `add_segment` | records both endpoints when `flag_include_vertices`; this branch has NO `!= m_start_vertex` guard on `add_vertex` itself, so the bookkeeping adds its own guard to avoid marking the CURRENT start vertex walked (which would wrongly survive a later re-seat) |
| `add_shower` | merges the absorbed shower's own `m_walked_nodes` — mirrors the prototype's `add_shower` merging the absorbed shower's segment→vertex map, so an imported orphan node stays unwalked |
| `complete_structure`, initial seed | records the start segment's two endpoints as it pushes them (unconditional, as the prototype does) |
| `complete_structure`, worklist frontier | `const bool unseen = walk_visited_parity ? !m_walked_nodes.count(vd) : !this->has_node(vd);` — the `!= m_start_vertex` guards are unchanged, so the CURRENT start vertex is still never walked, exactly as in the prototype |

A vertex is recorded in `m_walked_nodes` the moment it is enqueued, so it can
be pushed at most once per walk — termination is unchanged.

Env-gated probe `WCT_SHOWER_WALK_DEBUG` (stderr, byte-neutral): one
`SHOWER_WALK rewalk` line per node the parity predicate re-enqueues that
`has_node` would have pruned.

**Threading** mirrors `shower_endpoint_skip_orphan_vtx` exactly:
`NeutrinoPatternBase.h` + `TaggerCheckNeutrino.h` members, `configure` read,
`default_configuration` round-trip, pass-through to `pattern_algos`, all
8 `complete_structure_with_start_segment` call sites in
`NeutrinoShowerClustering.cxx`, doctest pin, `cfg/pgrapher/common/clus.jsonnet`
(signature + key-suppression), sbnd `clus.jsonnet` (2 defaults + 2
pass-throughs), `wct-pr-perevt.jsonnet`, and the runner's tri-state loop
(`SBND_SHOWER_WALK_VISITED_PARITY`, contract `0`/`1`).

**Why this design, not the alternatives named in sec 10.3.** Not a
start-vertex-only special case: the visited-set semantic covers both a former
start vertex and an `add_shower`-imported foreign node with one rule, and the
probe confirms the extra reach over the special case never fires in this
sample (exactly one re-expansion total, a former start vertex — see below).
Not "stop `set_start_vertex` calling `add_vertex`" (sec 10.3 option 2): that
would touch `fill_sets`, every BFS barrier built from a shower's node set, and
the two existing endpoint knobs' own point-of-use workarounds — much wider
than needed. **Robustness / physics**: the fix makes the walk strictly more
absorptive, so containment matters. `shower_absorb_track_guard` (SBND ON,
pr/40 F12) already refuses a confidently-PID'd long straight non-electron and
terminates the walk there — tracks cannot be swallowed by this fix; only short
stubs sitting inside the shower cone can, and in 168596 the newly-reachable
piece is a 4.74 cm proton stub, well under the 10 cm guard threshold and
physically inside the EM shower.

### 3. Validation — 67-event sample (nueCC48 + NCpi0 19), fresh binary

**V1 — knob off is byte-identical to a genuine pre-change build.** Rather than
trust a possibly stale pre-existing arm (the round-1 cross-binary trap), the
pre-fix source was `git stash`ed, rebuilt, and run on 3 spot events (168596 +
2 others) as `work-pr91r4-prechange-ref`; the post-fix binary (knob off) was
then run on the same 3 events as `work-pr91r4-postchange-ref`. **PASS — all
6 archives byte-identical.**

**V2 — knob on, all 67 events.** `work-pr91r4-on-{nuecc48,ncpi0}` vs
`work-pr91r4-off-{nuecc48,ncpi0}`, member-content hashes over `mabc-pr.zip` +
`pctree-pr-evt<ID>.tar.gz`:

- **nueCC48 (48 events, 96 archives): PASS=95/96 — exactly ONE mover, evt
  168596's `mabc-pr.zip`.** Its pctree tarball is unchanged.
- **NCpi0 (19 events, 38 archives) — the pi0-veto sample: PASS=38/38, zero
  movers.** No genuine two-gamma pi0 was disturbed by making the walk more
  absorptive.
- Both samples' merged `nusel-table.tsv` are **byte-identical** (`diff` rc=0)
  — zero score/label movers, zero `nu-candidate` status changes anywhere in
  67 events.

**168596 mechanism, confirmed by probe.** `WCT_SHOWER_WALK_DEBUG` fires
**exactly once** across all 67 events:

```
SHOWER_WALK rewalk shower_id=2 via_seg=14093 vtx=14027
```

— precisely the round-2 diagnosis: shower 2 (the 2039 MeV electron) re-walks
its own former start vertex 14027 via member segment 14093. Downstream:

| | off (legacy) | on (fixed) |
|---|---|---|
| shower count | 17 | 16 |
| `pio_mass` | 114.05835... (spurious pi0) | all `-1.0` (sentinel, no pair) |
| `kine_reco_Enu` | 2331.291 MeV | 2324.257 MeV (−0.3%) |
| `nusel-evt168596.tsv` | — | byte-identical to off |

Mechanism (a) predicted in sec 10.3 fired: `examine_showers`' re-flood
(`:3119`) reaches 14052 via the newly-walked 14027→14070 edge, absorbs
segment 14072 directly, and the `:3138` absorb block removes the now-subsumed
shower — no separate shower survives to reach `id_pi0_with_vertex`, so the
pairing never has a second candidate to pair the 2039 MeV electron's charge
against. The small `kine_reco_Enu` drop is the borrowed-charge correction:
14072's points, previously double-counted through the ownership-free
`dis_cut = 0.6 cm` 2-D sum (sec 1/round-2 finding), are now credited once, as
genuine shower members.

**Acceptance criteria (set before the run) — met:**

| sample | criterion | result |
|---|---|---|
| NCpi0 (19) | no genuine two-gamma pi0 destroyed | zero movers — met trivially |
| nueCC48 (48) | `kine_reco_Enu` stable, no `nu-candidate` status lost, movers enumerated | 1 event moves, `nusel` unchanged, Enu −0.3% |
| 168596 | spurious pi0 gone | confirmed, mechanism traced |

"pi0 count unchanged" was explicitly **not** the bar — 168596's pi0 is
*supposed* to disappear, and it is the only one that does.

**V3 — bare-run composition gate**, after flipping `wct-pr-perevt.jsonnet`'s
`shower_walk_visited_parity` to `true`: a bare run (no `-A`/env override) vs
`work-pr91r4-on-*`, all 67 events. **PASS on 134/134 archives**, both
`nusel-table.tsv` byte-identical. The SBND operating point stays
single-sourced in cfg (doc 68).

### 4. FIX SHIPPED: SBND PRODUCTION ON (toolkit, owner-authorized 2026-08-18)

C++ default stays `false` (`m_shower_walk_visited_parity{false}` in
`NeutrinoPatternBase.h`); `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`
flips it `true`. Compiled-config proof (M6): bare compile of
`wct-pr-perevt.jsonnet` shows `"shower_walk_visited_parity" : true` in the
output; the pre-flip compile showed the key absent (suppressed, default
`false`).

### 5. What this round does not claim

- No population estimate beyond the 67-event sample — the O(1000) numu gate
  and the `shower_endpoint_skip_orphan_vtx` population gate (sec 2b "STILL
  OWED") are both still owed, unaffected by this round.
- `123140` (sec 10, missed the 3 cm total-length gate by 1.5 mm) is untouched
  — a different gate, in a different cluster.
- No change to `set_start_vertex`, `fill_sets`, or the two existing endpoint
  knobs (`shower_endpoint_exclude_start_vertex`,
  `shower_endpoint_skip_orphan_vtx`).
- The `add_shower`-imported-node reach of the fix (vs a start-vertex-only
  special case) is real in the code but never fired in this 67-event sample —
  exactly one re-expansion total, and it was a former start vertex.
