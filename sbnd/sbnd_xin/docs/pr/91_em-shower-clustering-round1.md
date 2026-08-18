# doc pr/91 round 1 — EM shower clustering: why one shower comes out as many, and why end points land on the neighbour

**Status: DIAGNOSIS AND REPORT ONLY.** No behaviour knob is shipped by this
round, no default is changed, no SBND flip is proposed. The only code added is
three `std::getenv`-gated debug probes, proven byte-neutral below.

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

- Arms: `work-pr91r1-off-mc` (probes off), `work-pr91r1-dbg-mc` (probes on) —
  169626 / 174752 / 347129 / 394532, hub `work-mcp1k-cb0805`, `data` reality.
- Scripts: `scripts/pr91_shower_content.py`, `scripts/pr91_point_owner.py`.
- Toolkit commit: the three env-gated probes only, no behaviour change.
- Permanent probes now in the tree: `WCT_SHOWER_CREATE_DEBUG` (pr/84 r3),
  `WCT_SHOWER_CONTENT_DEBUG`, `WCT_SHOWER_MERGE_DEBUG`,
  `WCT_SHOWER_ENDPOINT_DEBUG` (this round).
