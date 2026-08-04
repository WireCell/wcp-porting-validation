# doc pr/32 — neutrino vertex identification: prototype ↔ toolkit fidelity audit

**Why.** Fifth in the porting-audit series, after pr/28 (vertex fit + trajectory
dQ/dx), pr/29 (Steiner graph build), pr/30 (proto-vertex + track-segment
finding) and pr/31 (topology / PID / direction). This one covers **step 4 of
the eight** in doc pr/27 §0 — the stage that decides *which vertex is the
neutrino interaction point*.

**Status.** §1-§9 were written **AUDIT ONLY** -- no code, no config, no event
run -- and every "this changes the output" in them is an argument from source.
**§10 filters, §11 implements**: four fixes shipped and turned on for SBND.

> **§11 (implementation, 2026-08-04) is the current state: all four kept
> findings are FIXED and are SBND PRODUCTION DEFAULTS ON** (toolkit
> `407c5ba9`), each gated 48/48 byte-identical on the nueCC48 manifest.
> §11.7 corrects §10.2's one substantive error.
>
> **§10 (owner filter, 2026-08-04) supersedes §3 and §8 where they disagree.**
> §1–§9 were written against toolkit `4f2e7303`. Re-verified against
> **`f8f2150a`**, the twelve divergences filter to **four** (F1–F4) with
> proposed solutions; **four more (P4, P8, P10, P11) were already fixed or
> resolved between the two revisions** and are closed. §10.8 lists this doc's
> own errors. Still no code changed this round.

**Headline.** The scoring machinery is a faithful port — every term of
`compare_main_vertices`, every rung of the `calc_conflict_maps` angle ladder,
every term of `compare_main_vertices_global`, and the whole
`examine_main_vertices_local` back-to-back filter reproduce the prototype
term-for-term, and the C++ defaults for all four retuned constants are
bit-identical to the prototype literals. Twelve behavioural divergences follow
(§3). One dominates: **P2**, the DL vertex re-ranker that actually picks SBND's
neutrino vertex has no prototype counterpart at all. **P1** — two functions read
the skeleton-snapped vertex where the prototype reads the continuous fit, and
where four other sites in the same file read the fit — is real but bounded by
`fit_distance()`, and does not bite where no fit has run.

Separately, and arguably more consequential than any single divergence: **SBND
does not run at the prototype operating point** (§4). `vertex_z_prior_scale` is
**100 cm where the prototype hard-codes 200 cm** — the upstream-z term in the
score ladder is twice as steep — and the MIP dQ/dx reference is 48000 against
the prototype's 43e3. Both are deliberate SBND config, not port defects.

**Which toolkit was read.** A concurrent session is editing
`NeutrinoVertexFinder.cxx` in this tree right now. Every toolkit file was
therefore snapshotted with `git show HEAD:<path>` **before** reading, at
**`4f2e7303`**, and every toolkit anchor below is relative to that revision —
not to the working tree. Re-derive anchors against `git show 4f2e7303:` if they
do not match what you see.

---

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit && git rev-parse --short HEAD   # 4f2e7303

# The exact snapshots this audit was written against:
for f in clus/src/NeutrinoVertexFinder.cxx clus/src/NeutrinoDeghoster.cxx \
         clus/src/NeutrinoPatternBase.cxx ; do
  git show 4f2e7303:$f > /home/xqian/tmp/pr32/$(basename $f)
done

# Prototype side (read-only, line numbers stable):
ls prototype_base/pid/src/{NeutrinoID_track_shower.h,NeutrinoID_improve_vertex.h,\
NeutrinoID_deghost.h,NeutrinoID_DL.h,NeutrinoID.cxx}
```

No build, no run, no A/B. Nothing in this doc requires one.

---

## §0 Scope

Doc pr/27 §6 defines this stage as five sub-steps in `TaggerCheckNeutrino::visit()`
order: 6a `determine_main_vertex` per cluster, 6b deghosting, 6c the global
choice (DL first, traditional fallback), 6d `improve_vertex`, 6e the final
`examine_direction`. All five are audited.

### The function map, re-derived at `4f2e7303`

Every toolkit anchor in pr/27 §6 has drifted. This table is rebuilt from the
snapshot, not inherited.

| # | function | prototype | toolkit (`4f2e7303`) |
|---|---|---|---|
| 1 | `determine_main_vertex` | `NeutrinoID_track_shower.h:1249` | `NeutrinoVertexFinder.cxx:2498` |
| 2 | `examine_main_vertex_candidate` | `…track_shower.h:1400` | `…VertexFinder.cxx:274` |
| 3 | `compare_main_vertices_all_showers` | `…track_shower.h:1451` | `…VertexFinder.cxx:358` |
| 4 | `compare_main_vertices` | `…track_shower.h:1589` | `…VertexFinder.cxx:760` |
| 5 | `calc_conflict_maps` | `…track_shower.h:1725` | `…VertexFinder.cxx:543` |
| 6 | `examine_direction` | `…track_shower.h:1876` | `…VertexFinder.cxx:1059` |
| 7 | `find_cont_muon_segment` | `…track_shower.h:2304` | `…VertexFinder.cxx:961` |
| 8 | `change_daughter_type` | `…track_shower.h:654` | `…VertexFinder.cxx:2719` |
| 9 | `examine_main_vertices_local` | `NeutrinoID.cxx:529` (2nd overload) | `…VertexFinder.cxx:2815` |
| 10 | `examine_main_vertices` (global) | `NeutrinoID.cxx:416` (1st overload) | `NeutrinoPatternBase.cxx:2418` |
| 11 | `compare_main_vertices_global` | `NeutrinoID.cxx:801` | `…VertexFinder.cxx:3007` |
| 12 | `check_switch_main_cluster` | `NeutrinoID.cxx:743` | `…VertexFinder.cxx:3171` |
| 13 | `check_switch_main_cluster_2` | `NeutrinoID.cxx:960` | `…VertexFinder.cxx:3265` |
| 14 | `determine_overall_main_vertex` | `NeutrinoID.cxx:333` | `…VertexFinder.cxx:3745` |
| 15 | `determine_overall_main_vertex_DL` | `NeutrinoID_DL.h:4` | `…VertexFinder.cxx:3304` |
| 16 | `swap_main_cluster` | `NeutrinoID.cxx:735` | `NeutrinoPatternBase.cxx:2395` |
| 17 | `calc_dir_cluster` | `NeutrinoID.cxx:691` | `NeutrinoPatternBase.cxx:2335` |
| 18 | `get_dir` / `vertex_get_dir` | `NeutrinoID.cxx:917`, `…improve_vertex.h:1020` | `NeutrinoPatternBase.cxx:1812` |
| 19 | `improve_vertex` | `NeutrinoID_improve_vertex.h:44` | `…VertexFinder.cxx:2019` |
| 20 | `fit_vertex` | `…improve_vertex.h:11` | `…VertexFinder.cxx:1888` |
| 21 | `search_for_vertex_activities` | `…improve_vertex.h:1039` | `…VertexFinder.cxx:56` |
| 22 | `eliminate_short_vertex_activities` | `…improve_vertex.h:365` | `…VertexFinder.cxx:1691` |
| 23 | `deghosting` | `NeutrinoID_deghost.h:13` | `NeutrinoDeghoster.cxx:588` |
| 24 | `deghost_segments` | `NeutrinoID_deghost.h:45` | `NeutrinoDeghoster.cxx:353` |
| 25 | `deghost_clusters` | `NeutrinoID_deghost.h:233` | `NeutrinoDeghoster.cxx:74` |

### Explicitly NOT audited

* **`MyFCN` and the internals of `fit_vertex`** — doc **pr/28**. The
  *orchestration* around it (the `improve_vertex` fit loop, the 0.5 cm re-track
  threshold) **is** in scope and is audited below.
* **The `examine_structure_*` family** (`examine_structure_final`,
  `examine_structure_4`, `examine_vertices`, `NeutrinoID_final_structure.h` ↔
  `NeutrinoStructureExaminer.cxx`). It is called from `determine_main_vertex`
  but it is its own ~700-line-per-side machinery shared with other stages, and
  it deserves its own audit.
* **`do_multi_tracking`** — the fitting engine, doc pr/27 §2.
* **Stage 5 onward** (`NeutrinoID_shower_clustering.h`, energy, taggers).

---

## §1 How far to trust each claim

Same two tiers as pr/28 §3b, carried through pr/29–pr/31.

* **Tier A — read in full, both sides, line by line.** Trust these.
  `compare_main_vertices`, `calc_conflict_maps`, `examine_main_vertices_local`,
  `find_cont_muon_segment`, `determine_main_vertex`, the DL legacy branch and
  its veto, the `compare_main_vertices_global` term ladder, the
  `improve_vertex` fit loop and its PID-recheck block, and the
  `compare_main_vertices_all_showers` PCA-projection block.
* **Tier B — verified by targeted grep and constant sweep, not read end to
  end.** `examine_direction` (428 prototype vs 632 toolkit lines — its
  `flag_final` clause and constant histogram were checked, its PDG-reassignment
  ladder was not read in full), `search_for_vertex_activities`,
  `eliminate_short_vertex_activities`, the deghosting pair, and
  `check_switch_main_cluster{,_2}`. Findings drawn only from Tier B are marked.

---

## §2 What matches

This section is the load-bearing one. Twelve divergences read very differently
against a broken port than against a faithful one, and this port is faithful.

### 2.1 `determine_main_vertex` — the control flow is exact

Prototype `:1249-1399` against toolkit `:2498-2717`, step for step:

| step | prototype | toolkit |
|---|---|---|
| scan for `flag_save_only_showers`, break on first `!flag_in && ntracks>0` | `:1268-1281` | `:2508-2523` |
| if mixed **and** main cluster: `improve_vertex(…, false)` then `fix_maps_shower_in_track_out` | `:1284-1291` | `:2529-2534` |
| rebuild `map_vertex_track_shower` from a second full scan | `:1296-1309` | `:2541-2553` |
| all-showers: degree-1 vertices first, then the rest | `:1314-1322` | `:2559-2585` |
| mixed: only candidates with `ntracks > 0` | `:1325-1327` | `:2588-2595` |
| all-showers ⇒ `compare_main_vertices_all_showers`, else early return | `:1336-1344` | `:2602-2620` |
| mixed ⇒ `examine_main_vertices_local`, then size 1 / >1 / 0 | `:1346-1370` | `:2624-2659` |
| `examine_structure_final` only when **not** all-showers | `:1373-1377` | `:2664-2666` |
| `examine_direction(main_vertex, flag_final=false)` | `:1381` | `:2670` |

The **second full scan** at step 3 is not redundant — `improve_vertex` may have
moved or merged vertices between the two — and the toolkit reproduces it rather
than caching the first result. The degree-1-first ordering in the all-showers
branch is preserved. Both early-return paths are preserved.

### 2.2 `compare_main_vertices` — all ten terms, in order

Read side by side (prototype `:1589-1722`, toolkit `:760-958`):

| # | term | prototype | toolkit |
|---|---|---|---|
| 0 | `max_length_muon` = longest non-shower non-proton segment | `:1600-1610` | `:773-792` |
| 1 | `n_in > n_out` ⇒ `−(n_in−n_out)/4` | `:1649` | `:861` |
| 1′ | else ⇒ `−(n_in−n_out)/4 + (n_in+n_out)/8` | `:1651` | `:863` |
| 2 | upstream-z prior | `:1665` | `:881` |
| 3 | per shower segment `+1/8` | `:1672` | `:897` |
| 4 | shower with daughter length >45 cm `+1/8` | `:1675-1676` | `:900-901` |
| 5 | per track segment `+1/4` | `:1678` | `:904` |
| 6 | clear proton `+1/4` | `:1680-1681` | `:911-912` |
| 7 | else directed non-shower `+1/8` | `:1682-1683` | `:913-914` |
| 8 | is `max_length_muon` and `max_length > 35 cm` `+1/8` | `:1685-1686` | `:917-918` |
| 9 | inside fiducial volume `+0.5` | `:1697-1698` | `:927-928` |
| 10 | `− calc_conflict_maps(vtx)/4` | `:1705-1706` | `:935-936` |
| — | argmax, strict `>`, **no tie-break** | `:1713` | `:944` |

The prototype's `1/4./2.` idiom (integer `1` divided by double `4.`, then by
`2.`) is preserved as `1.0/4.0/2.0` — same 0.125, no integer-division trap
introduced or removed. The `45 cm`, `35 cm` and `0.5` literals are identical.

A `/4.` count mismatch in the mechanical sweep (10 prototype, 9 toolkit)
resolves entirely to prototype `:1691`, a `std::cout` inside `if (flag_print)`.
**No scoring term is missing.**

### 2.3 `calc_conflict_maps` — the BFS and both penalty ladders

Prototype `:1725-1873`, toolkit `:543-758`. The frontier BFS seeded at the
assumed vertex, the first-writer-wins `map_seg_dir` guard, the per-segment
direction conflict (`+1.0` firm / `+0.5` weak, showers only above 5 cm), the
`flag_check` suppression when the outgoing partner is a shower trajectory or
both are showers, the hard-coded beam direction `(0,0,1)`, and both ladders:

| max in-out angle | penalty | prototype | toolkit |
|---|---|---|---|
| `< 35°` | `+5` | `:1837-1838` | `:721-722` |
| `< 70°` | `+3` | `:1839-1840` | `:723-724` |
| `< 85°` | `+1` | `:1841-1842` | `:725-726` |
| `< 110°` | `+0.25` | `:1843-1844` | `:727-728` |
| `angle_beam < 60 && max_angle < 110` | `+1` | `:1848-1849` | `:732-733` |
| `else angle_beam < 45 && max_angle < 70` | `+3` | `:1850-1851` | `:734-735` |

plus `n_in > 1` ⇒ `+(n_in−1)`, or `+(n_in−1)/2` when all incoming are showers
(`:1857-1862` ↔ `:742-747`), and shower-in/track-out ⇒
`+min(n_in_shower, n_out_tracks)` (`:1865-1866` ↔ `:751-752`).

Note the second beam clause is `else if` in **both** trees, so
`angle_beam < 45 && max_angle < 70` is unreachable — the first clause already
caught it. That is a prototype quirk faithfully preserved, not a toolkit bug.

### 2.4 `examine_main_vertices_local` — the back-to-back filter

Prototype `NeutrinoID.cxx:529-688`, toolkit `:2815-3004`. Identical: the
`vertices.size()==1` early return; degree-1 vertices kept unconditionally; the
pair loop over segments longer than 10 cm with direction 3-vectors at **both**
15 cm and 30 cm; the muon rule (`>165°` at either scale, one segment pdg 13,
one longer than 30 cm); the proton rule (`>170°`, proton/unknown pair, **both**
longer than 20 cm); the rescue scan (`>35 cm` daughter shower, or
`!dir_weak && len > 6 cm` track) with its `break`; the relabel-to-muon block
with its shower-trajectory skip and its `>40 cm && dirsign==0` shower-topology
gate; and the `find_cont_muon_segment` walk pushing the chain-end vertex.

### 2.5 `compare_main_vertices_global` — term for term

Prototype `NeutrinoID.cxx:801-916`, toolkit `:3007-3170`. Every term present
and in order: z-prior (`:819` ↔ `:3034`), `+1/8` shower / `+1/4` track
(`:826`/`:828` ↔ `:3050`/`:3052`), `+1/4` clear proton / `+1/8` directed track
(`:831`/`:833` ↔ `:3061`/`:3063`), `+0.25` in main cluster (`:836` ↔ `:3070`),
`+0.5` for `in_fv || in main cluster` (`:844` ↔ `:3085-3088`), the
`vertex_get_dir(…, 5 cm)` pointing terms `+0.25` under 15° and `+0.125` under
30° (`:874`/`:877` ↔ `:3123`/`:3126`), and the isolation penalty
`−0.25 × num_tracks` when `delta==0`, not main cluster, host track length
under 6 cm (`:894` ↔ `:3147`).

The prototype's conflict term here is **commented out** (`:852`); the toolkit
correctly has none.

### 2.6 `find_cont_muon_segment`

Prototype `:2304-2369`, toolkit `:961-1057`. Same 15 cm direction pair, same
`(π − angle)` convention, same 50 cm re-measurement for segments over 50 cm,
same acceptance `(angle<10 || angle1<10 || (sg_length<6 cm && (angle<15 ||
angle1<15))) && (ratio<1.3 || flag_ignore_dQ_dx)`, same
`length·cos(angle)` argmax, same `flag_cont` semantics.

### 2.7 The constant sweep

The per-function constant histogram (pr/30's method) was run over all
twenty-five pairs. **Every** apparent difference resolved to one of: a debug
`std::cout`; an inlined predicate (`get_flag_shower()` expanding to the
three-clause `kShowerTrajectory || kShowerTopology || |pdg|==11`); the
`boost::degree(...) <= 1` idiom replacing `map_vertex_segments[v].size()==1`;
a `43e3` literal replaced by the `m_mip_dqdx_median` member whose **C++ default
is 43000** (§4); a `200*units::cm` replaced by `m_vertex_z_prior_scale` whose
**C++ default is 200 cm**; or a block that is commented out in the prototype.

**Zero genuine constant divergences in the ten pairs read in full**; the
remaining fifteen were cleared by the sweep alone, whose known blind spot is
stated in §9. That is what makes §3 a list about behaviour rather than about
tuning — for the pairs where it is a verified result.

### 2.8 The DL legacy branch is a faithful port

Prototype `NeutrinoID_DL.h:4-130` ↔ toolkit `:3304-3440`, `flag_rerank==false`.
Same `dQdx_scale = 0.1` / `dQdx_offset = -1000` input scaling (the prototype
calls it "hack for now"; the toolkit exposes it as config with those
defaults); same point cloud — every graph vertex plus every **segment interior**
fit point, endpoints excluded; same nearest-vertex snap; same "reject if every
connected long track points away and is high-dQ/dx" veto with the same
`length > 15 cm && !dir_weak && !flag_start` gate; same `min_dis > dl_vtx_cut`
rejection.

---

## §3 Divergences

Ranked by how far the effect propagates. All twelve are **unconditional** —
none sits behind a knob — so all twelve are CLAUDE.md §5 rule 1, the owner's
call. Nothing is proposed here.

### P1 — the conflict-penalty angles are measured from the snapped vertex, not the continuous fit

**Tier A.** Prototype `calc_conflict_maps` takes both direction 3-vectors from
the **fitted** vertex position:

```cpp
// prototype NeutrinoID_track_shower.h:1804, :1808
map_in_segment_dirs[sg]  = sg->cal_dir_3vector(vtx->get_fit_pt(), 10*units::cm);
map_out_segment_dirs[sg] = sg->cal_dir_3vector(vtx->get_fit_pt(), 10*units::cm);
```

The toolkit takes them from `wcpt()`:

```cpp
// toolkit NeutrinoVertexFinder.cxx:670, :678
map_in_segment_dirs[sg]  = segment_cal_dir_3vector(sg, vtx->wcpt().point, 10*units::cm);
map_out_segment_dirs[sg] = segment_cal_dir_3vector(sg, vtx->wcpt().point, 10*units::cm);
```

`compare_main_vertices_all_showers` has the same substitution in its PCA
projection — prototype `:1480` uses `get_fit_pt()`, toolkit `:401-403` uses
`vtx->wcpt().point` — and again at the min/max endpoint z-comparisons
(`:443` and `:456`).

**What `wcpt()` actually holds — and why this is smaller than it first looks.**
`fit_vertex` → `MyFCN::UpdateInfo` writes **both** members: `vtx->fit().point =
fit_pos`, the continuous fit, and `vtx->wcpt().point = vtx_new_pt`, that fit
**snapped to the nearest skeleton node** (`MyFCN.cxx:485-496`). So `wcpt()` is
not an unfitted position — it is the fitted position, quantised. The gap between
the two is exactly what `Vertex::fit_distance()` returns
(`PRVertex.h:85`: `m_fit.distance(m_wcpt.point)`), the quantity doc pr/27 §14
lists as undumped-and-directly-useful.

The divergence is therefore **a quantisation to the skeleton lattice, not a
fitted-vs-unfitted swap**. It is bounded by `fit_distance()`, not by the fit
displacement.

**Where the two trees agree exactly.** Prototype `ProtoVertex`'s constructor
initialises `fit_pt = wcpt` (`ProtoVertex.cxx:17-19`), so before any fit
`get_fit_pt()` *is* the wcpt position — the same value the toolkit's fallback
lambda returns when `fit().valid()` is false. Wherever no fit has run, there is
no divergence at all.

**Which narrows the reach substantially.** `improve_vertex` is called from
`determine_main_vertex` only under `!flag_save_only_showers &&
cluster.get_flag(main_cluster)` (`:2529`). So:

* `calc_conflict_maps` (reached via `compare_main_vertices`, mixed branch only)
  is exposed **for the main cluster**, where a fit has run — by `fit_distance()`
  per vertex.
* `compare_main_vertices_all_showers` is the **all-showers** branch, which is
  exactly the branch where `improve_vertex` did *not* run. Whether `fit()` is
  nonetheless valid there from an earlier `do_multi_tracking` was **not
  verified** — see §7. If it is not, P1 does not bite at that site at all.

**Why it still matters where it bites.** These angles feed the
`<35 / <70 / <85 / <110` ladder, i.e. the `+5 / +3 / +1 / +0.25` conflict
penalties and `angle_beam`. The conflict total is divided by 4, so one rung
crossed is a 1.25 or 0.75 swing against a ladder whose topology terms are 0.125
apart. A sub-centimetre shift in the point the 10 cm direction vector is
measured *from* can cross a rung on a short segment.

**What makes this a defect rather than a choice**: the toolkit is inconsistent
with itself. The same file uses the fit-or-fallback idiom
`fit().valid() ? fit().point : wcpt().point` at `compare_main_vertices:870`,
`examine_main_vertices_local:2839`, `find_cont_muon_segment:977` and
`compare_main_vertices_global:3032`. Only these two functions read `wcpt()`
raw. Whatever the intent, the file does not apply it uniformly.

### P2 — the DL vertex re-ranker has no prototype counterpart, and it is what SBND runs

**Tier A.** The prototype's entire DL path is 177 lines: run SCN, take the
nearest `ProtoVertex` to the regressed point, apply the backward-track veto,
apply `min_dis > dl_vtx_cut`, accept or decline. There is no re-rank, no top-K,
no composite score, no acceptance threshold.

The toolkit ports that faithfully (§2.8) as the `m_dl_vtx_rerank == false`
branch — but that branch is **not the default**. The default requests
`dl_vtx_top_k` (5) voxels, snaps each to a vertex, and scores them with a
seven-term composite declared at `:3557-3566`:

| term | formula | weight |
|---|---|---|
| `s_dl` | `dl_score × m_dl_vtx_score_scale` | 1000 |
| `s_snap` | `− min(2.0, snap_dis / 5 cm)` | `W_SNAP_L` 5 cm |
| `s_fwd_z` | `− 0.25 × clamp((z − z_min)/400 cm, 0, 1)` | `W_FWD_Z` 0.25 |
| `s_clen` | `+ 2.0 × min(1, L_host / 60 cm)` | `W_CLEN` 2.0 |
| `s_isol` | `− 2.0` if `L_host < 6 cm` and host ≠ main cluster | `W_ISOL` 2.0 |
| `s_main` | `+ 2.0` if host == main cluster | `W_MAIN` 2.0 |
| `s_fv` | `+ 0.5` inside fiducial volume | `W_FV` 0.5 |

accepted only if the argmax scores `≥ m_dl_vtx_min_accept_score` (4.0, `:3642`).
The in-code comment at `:3580` says the soft snap penalty "replaces the old hard
`min_dis > 2*dl_vtx_cut` gate" — so this is a considered redesign, not drift.

**Reach.** SBND sets `dl_weights = 'uboone/scn_vtx/t48k-m16-l5-lr5d-res0.5-CP24.pth'`
(`wct-pr-perevt.jsonnet:389`) and leaves `dl_vtx_rerank` at its `true` default.
Per doc **pr/4** the SCN DL vertex is the SBND default. **SBND's neutrino vertex
is therefore chosen by code with no prototype counterpart**, and when DL
accepts, the entire traditional global ladder is skipped (§4).

**§5 rule 4 — both readings, neither picked.** *(a)* This is a deliberate
toolkit design, documented in pr/27 §6, adopted in pr/4, and the legacy path is
retained and faithful — so "prototype fidelity" is simply not the right frame.
*(b)* The re-ranker's weights are uncalibrated for SBND in the same way the
BDTs are (doc pr/2 gap G1); `W_MAIN`, `W_CLEN` and the 4.0 acceptance floor
were not derived on SBND, and the term that dominates a 5-candidate field
(`s_dl × 1000`) is scaled by a constant chosen for uBooNE score magnitudes.

### P3 — the shower-trajectory recheck in `improve_vertex` changed gate, step size and scale

**Tier A.** Prototype `NeutrinoID_improve_vertex.h:288-289`:

```cpp
if ((pair_result.first <= 2 || sg->get_medium_dQ_dx()/(43e3/units::cm) > 1.6 && pair_result.first <= 3)
    && sg->get_flag_shower_trajectory()) {          // <-- reads the STORED flag
  if (!sg->is_shower_trajectory()) {                // <-- recompute, default step 10 cm
```

Toolkit `:2393-2394`:

```cpp
if ((pair_result.first <= 2 || (medium_dQdx/m_mip_dqdx_median > 1.6 && pair_result.first <= 3))
    && segment_is_shower_trajectory(sg, 10*units::cm, m_mip_dqdx)) {        // <-- RECOMPUTES
    if (!segment_is_shower_trajectory(sg, 1.0*units::cm, m_mip_dqdx_median)) {  // <-- 1 cm, other scale
```

Three changes stacked in two lines:

1. **The outer gate is a recomputation, not a flag read.** The prototype asks
   "is this segment *currently labelled* a trajectory shower?" — a label set
   upstream in stage 3. The toolkit re-derives it. A segment whose stored flag
   and freshly-computed answer disagree takes the opposite branch.
2. **The inner step size is 1.0 cm, not the prototype's 10 cm default**
   (`ProtoSegment.h:85`: `is_shower_trajectory(double step_size = 10.*units::cm)`).
   pr/31 audited `is_shower_trajectory`; the step size sets the sampling of the
   trajectory-vs-topology test, and a tenfold change is not cosmetic.
3. **The two calls use different dQ/dx scales** — `m_mip_dqdx` outside,
   `m_mip_dqdx_median` inside. In SBND those are 56000 and 48000. The prototype
   uses one convention throughout.

Consequence: whether a main-vertex leg gets its direction re-determined by
`determine_dir_track` immediately after the vertex fit.

### P4 — `used_segments` is iterated by pointer while the loop mutates PID

**Tier A.** `examine_main_vertices_local` collects the back-to-back segments
into a pointer-keyed set and then iterates it:

```cpp
std::set<SegmentPtr> used_segments;      // toolkit :2838
...
for (auto sg1 : used_segments) {         // toolkit :2933  <-- pointer-address order
    ... sg1->particle_info()->set_pdg(13);
    change_daughter_type(graph, vtx, sg1, 13, muon_mass, ...);
    auto results = find_cont_muon_segment(graph, sg1, vtx);
    while (results.first != nullptr) { ... }
```

The body **mutates** PID via `change_daughter_type`, and `find_cont_muon_segment`
reads PID and median dQ/dx of neighbouring segments. So the terminal vertex a
later iteration walks to depends on relabelling a earlier iteration already did.
Iteration order therefore changes which vertices enter `tmp_vertices`, which is
the candidate list handed to the no-tie-break argmax.

**What makes this pointed rather than inherited**: the toolkit fixed the sibling
container eleven lines earlier and said so —
`IndexedVertexSet tmp_vertices;  // order by stable graph index, not pointer address`
(`:2820`) — but left `used_segments` a `std::set<SegmentPtr>`. `IndexedSegmentSet`
exists and is used elsewhere in the same file. The prototype has the identical
shape at `NeutrinoID.cxx:603`; this is a half-completed fix, not a fresh
regression.

### P5 — the candidate-order determinism inverts, and the argmax has no tie-break

**Tier A.** Every container that decides candidate *order* differs in kind:

| producer | prototype | toolkit |
|---|---|---|
| `flag_save_only_showers` scan | `map_vertex_segments` — `std::map<ProtoVertex*,…>` | `ordered_nodes(graph)` `:2508` |
| `map_vertex_track_shower` build | pointer-keyed map `:1296` | `ordered_nodes` `:2541` |
| all-showers candidate push | pointer-keyed iteration `:1314-1322` | `ordered_nodes` `:2559`, `:2578` |
| mixed candidate push | pointer-keyed iteration `:1325-1327` | `ordered_nodes` `:2588` |
| `examine_main_vertices_local` output | `std::set<ProtoVertex*>` → `std::copy` `NeutrinoID.cxx:647` | `IndexedVertexSet` → `std::copy` `:3002` |

**Why the order is not cosmetic here.** `compare_main_vertices` selects with a
strict `>` over the candidate vector and **no tie-break** (prototype `:1713`,
toolkit `:944`) — first in vector order wins. Every scoring term except the
z-prior and the conflict penalty is a dyadic rational: `1/8`, `1/4`, `1/2`,
`(n_in−n_out)/4`, so two candidates tie exactly iff their dyadic parts agree
**and** their z values are bit-equal. That is rare, not ordinary — the z-prior
is unconditional for every descriptor-valid candidate (`:881`).

It is generic in exactly one regime, and that regime is P7's: an
invalid-descriptor candidate **skips the z-prior** (`:878`) while remaining in
the argmax (`:943`), so its score is purely dyadic. Two such candidates tie by
construction. **P5's reach is therefore contingent on P7's reachability** —
loose end 2 — and the two findings are coupled rather than independent.

So: the toolkit is deterministic and the prototype is not — the same inversion
pr/30 recorded, now with a demonstrated consumer. But the two orders are also
*different*, so on any tied event the two trees pick different vertices, and
prototype parity is not recoverable by any ordering choice because the
prototype's own order is an allocator artifact.

The one place order does **not** matter is the `flag_save_only_showers` pre-scan:
it is an OR with an early `break`, so the answer is order-independent even
though the traversal is not.

### P6 — `flag_start` changed from an exact index match to a nearest-endpoint test

**Tier A.** The prototype decides which end of a segment touches a vertex by
exact Steiner-index equality, and falls through **uninitialised** when neither
matches:

```cpp
// prototype :1628-1631 (compare_main_vertices), :1764-1767 (calc_conflict_maps),
//           NeutrinoID_DL.h:94-98 (the DL veto)
bool flag_start;
if      (sg->get_wcpt_vec().front().index == vtx->get_wcpt().index) flag_start = true;
else if (sg->get_wcpt_vec().back().index  == vtx->get_wcpt().index) flag_start = false;
// no else — flag_start is read uninitialised
```

The toolkit substitutes a distance comparison, always defined:

```cpp
// toolkit :834-835 and :608-609
bool flag_start = (ray_length(Ray{wcps.front().point, start_vtx->wcpt().point}) <
                   ray_length(Ray{wcps.back().point,  start_vtx->wcpt().point}));
```

and in the DL veto (`:3427-3428`) uses a third formulation,
`find_vertices(graph, sg)` followed by `v1 == min_vertex` — which inherits
pr/30's `find_vertices` ordering divergence, giving that finding a concrete
consumer in this stage.

`flag_start` decides the sign of every direction conflict and the in/out
classification, so it is upstream of both penalty ladders.

**§5 rule 4 — both readings.** *(a)* The toolkit removes genuine undefined
behaviour and is strictly better where the prototype's indices do not match.
*(b)* Where they *do* match the two agree, so the change only bites in the
cases the prototype never defined — which means the toolkit is now making a
definite decision the prototype left to chance, and three different
formulations of "which end" now coexist in one file.

### P7 — invalid-descriptor candidates keep a score of 0 and stay in the argmax

**Tier A.** `compare_main_vertices` initialises every candidate to 0 (`:766`),
then guards the proton-topology loop (`:796`) and the per-segment / z-prior loop
(`:878`) with `if (!vtx->descriptor_valid()) continue;` — but the fiducial loop
(`:926`), the conflict loop (`:934`) and the argmax (`:943`) are unguarded.

A candidate with an invalid descriptor therefore reaches the argmax carrying
`0 + (0.5 if in FV) − conflicts/4`. Since real candidates routinely score
negative (the z-prior and the conflict penalty are both subtractive), a
skipped-over candidate can win. The prototype has no descriptor concept and no
such path.

Whether an invalid descriptor is reachable at this point was **not** verified —
see §7.

### P8 — the in/out direction maps are pointer-keyed and drive an argmax

**Tier A, shared with the prototype.** `calc_conflict_maps` builds
`std::map<SegmentPtr, geo_vector_t> map_in_segment_dirs` / `map_out_segment_dirs`
(`:652-653`) and scans them for the maximum in-out angle with a strict `>`
(`:688-698`). On an angle tie the winning `sg1`/`sg2` is decided by pointer
address — and `sg1`/`sg2` determine `flag_check` (whether the whole ladder is
suppressed) and `angle_beam` (whether the backward-going penalty fires).

The prototype is identical (`:1793-1794`, `:1816-1825`), so this is inherited.
It is listed because the toolkit has indexed containers and used them for the
sibling case in P4.

### P9 — degenerate-direction guards added

**Tier A.** The toolkit skips or zeroes where the prototype would compute with a
zero-length direction vector: `if (dir1.magnitude() == 0 || dir2.magnitude() == 0) continue;`
(`find_cont_muon_segment:999`, and the 50 cm re-measurement at `:1016`), and the
`angle1 = angle2 = 0` initialisation in `examine_main_vertices_local:2866-2874`.

The prototype's `TVector3::Angle()` on a zero vector yields NaN, and every
subsequent `>` comparison is false — so the prototype *implicitly* rejects.
The toolkit's `angle1 = 0` initialisation is **not** equivalent to NaN in the
`examine_main_vertices_local` case: `0 > 165` is false, same outcome. In
`find_cont_muon_segment` the `continue` skips the `max_ratio1` bookkeeping the
prototype would still have performed. The behaviours coincide on the acceptance
tests and differ on the bookkeeping.

### P10 — three raw `boost::edges(graph)` remain in this file

**Tier A.** pr/28 §10 swept `boost::out_edges` → `sorted_out_edges` across 117
sites; `boost::edges` (whole-graph iteration) was not in that sweep. Three
remain in the audited file:

| site | use | order-sensitive? |
|---|---|---|
| `:773` | `max_length_muon` scan in `compare_main_vertices` | **yes** — strict `>` argmax over length, ties by edge order |
| `:2028` | `existing_segments` build in `improve_vertex` | no — set insertion |
| `:3137` | host-cluster length/count in the global isolation term | no — sum and count |

Only `:773` can change an answer, and only on an exact length tie between two
non-shower non-proton segments.

### P11 — `existing_segments` is a pointer-keyed set, iterated

**Tier B.** Built as `std::set<SegmentPtr>` at `improve_vertex:2023` and
iterated at `eliminate_short_vertex_activities:1841`. The surrounding block
(`:1830-1845`) is a proximity test against existing segments; whether it
accumulates a minimum (order-independent) or breaks early (order-dependent) was
**not** read. Listed as a determinism candidate, not a confirmed divergence.

### P12 — `map_cluster_main_candidate_vertices` is not ported

**Tier A.** The prototype records the per-cluster candidate list before
filtering (`NeutrinoID_track_shower.h:1332`) and exposes it via
`get_map_cluster_candidate_vertices()` (`NeutrinoID.h:1720`). The toolkit has
no equivalent.

**Reach is diagnostic only.** Every consumer is an app-level output-tree filler
— `wire-cell-prod-nue.cxx:1522`, `-nnbar:1570`, `-nue-mt:1539`, `-pi0:1523`,
`-nue-port:1534`. No algorithm reads it. So this changes no decision — but it
is the concrete reason doc pr/27 §6 records "candidate list, per-cluster and
global → **nothing**" in its display-payload table, and it is what a vertex-tuning
display would need first.

---

## §4 The SBND operating point

**This stage's tunables are all live in SBND, and two of them move the score
ladder itself.** The C++ defaults are bit-faithful to the prototype; the SBND
config is not. Both facts matter.

| knob | C++ default | prototype literal | **SBND value** | where |
|---|---|---|---|---|
| `vertex_z_prior_scale` | 200 cm (`NeutrinoPatternBase.h:239`) | `200*units::cm` | **100.0** | `wct-pr-perevt.jsonnet:295` |
| `mip_dqdx_median` | 43000 (`NeutrinoPatternBase.h:114`) | `43e3/units::cm` | **48000** | `sbnd/clus.jsonnet:889` |
| `mip_dqdx` | 50e3 | — (stage-3 scale) | **56000** | `wct-pr-perevt.jsonnet:115` |
| `muon_dqdx_curve` | `{0.8866, 0.9533, 18 cm, 0.4234}` | `0.8866+0.9533·(18/L)^0.4234` | **`[0.8826, 1.0587, 18, 0.4745]`** | `wct-pr-perevt.jsonnet:352` |
| `dl_weights` | `""` (DL off) | hard-coded `.pth` path | **uboone SCN `.pth`** | `wct-pr-perevt.jsonnet:389` |
| `dl_vtx_rerank` | `true` | — (no counterpart) | `true` (default) | `common/clus.jsonnet:474` |
| `dl_vtx_top_k` | 5 | — | 5 (default) | — |
| `dl_vtx_min_accept_score` | 4.0 | — | 4.0 (default) | — |
| `dl_vtx_cut` | 25 mm | `dl_vtx_cut` | `null` ⇒ 25 mm | `wct-pr-perevt.jsonnet:371` |

Three consequences worth stating plainly:

1. **The upstream-z prior is twice as steep in SBND as in the prototype.** The
   term is `−(z − z_min)/scale`; halving the scale doubles the penalty. Against
   a ladder whose topology terms are 0.125 apart, a 100 cm z-spread now costs
   1.0 instead of 0.5. This is the single largest numerical difference between
   what SBND runs and what the prototype ran, and it is deliberate — the config
   comment at `:289` says it "reproduces 100/102/80/50 exactly", i.e. it was
   fitted, not inherited.
2. **Every dQ/dx ratio in this stage is ~10% smaller in SBND.** The prototype's
   `43e3` appears at ten sites in this stage — four in `examine_direction`, two
   in `improve_vertex`, one in `find_cont_muon_segment`, two in
   `deghost_segments`, one in the DL veto — and all ten route through
   `m_mip_dqdx_median`. With 48000, every `ratio > X` cut is harder to trip and
   the `ratio < 1.3` muon-continuation test is more permissive. The SBND
   `muon_dqdx_curve` retune partly compensates for the same shift; the two were
   derived together (`wct-pr-perevt.jsonnet:346-348`).
3. **The traditional global ladder may not execute at all.**
   `determine_overall_main_vertex` runs only under
   `if (!flag_dl_changed)` (`TaggerCheckNeutrino.cxx`, pr/27 §6c). With DL on
   and the re-ranker accepting, `compare_main_vertices_global`,
   `check_switch_main_cluster`, `check_switch_main_cluster_2` and the short
   high-dQ/dx proton-tagging block are all skipped. **How often DL accepts on
   SBND was not measured** — see §7. If it is most events, then §2.5's careful
   term-for-term verification describes a fallback, not production.

**Reading `clus.jsonnet` alone gives the wrong operating point** — the pr/25
flip and the z-prior retune live in `wct-pr-perevt.jsonnet`. This is the same
trap recorded in pr/31 §4.

---

## §5 Looks like a divergence, and is not

Thirteen candidates checked and cleared. Recorded so the next audit does not
re-open them.

**5.1 `examine_main_vertices_local` is not toolkit-only.** A name grep across
the whole prototype tree returns nothing, which reads as a 192-line invented
function. It is not: it is the **second overload** of `examine_main_vertices`,
`NeutrinoID.cxx:529`, taking `ProtoVertexSelection&` where the first
(`:416`) takes no argument. The 165/170° angles, the 35/20/6 cm rescue trio and
the `find_cont_muon_segment` walk are all there. The toolkit renamed it to
disambiguate the overload; the global one kept the name
(`NeutrinoPatternBase.cxx:2418`). *Verified behaviourally, by the constants,
not by the symbol.*

**5.2 The second candidate filter is commented out in the prototype.**
`NeutrinoID.cxx:651-687` — the `max_length > 40 cm` block with the `<1 cm` /
`<5 cm` length ladder and the `>155°` test — sits inside `/* … */`. The toolkit
correctly has no counterpart. Same shape as pr/29's commented-out kruskal and
pr/30 §5.1's never-incremented counter. **Do not "restore" it.**

**5.3 The global scorer's conflict term is commented out.** Prototype `:852`
`//  map_vertex_num[vtx] -= num_conflicts/4.;`. Absent in the toolkit. Correct.

**5.4 `muon_dqdx_cut` is not a changed formula.** `NeutrinoPatternBase.h:294`
evaluates `m_muon_dqdx_curve[0] + [1]·pow([2]/L, [3])`, and the member defaults
`{0.8866, 0.9533, 18 cm, 0.4234}` reproduce the prototype literal
`0.8866+0.9533*pow(18*units::cm/length, 0.4234)` bit-identically — the in-code
comment at `:292-293` says so and the arithmetic confirms it. SBND overrides the
four numbers (§4); the *port* is exact.

**5.5 `n_out_showers` is computed and never used** — in `calc_conflict_maps`,
in **both** trees (prototype `:1791`/`:1807`, toolkit `:650`/`:676` with an
explicit `(void)n_out_showers` at `:754`). Shared dead code.

**5.6 `max_length` in `examine_main_vertices_local` is dead in both.** Its only
reader was the commented-out block of §5.2. Toolkit `:2819`/`:2856` keeps
computing it. Harmless.

**5.7 The pair-loop idiom differs and is equivalent.** Prototype
`for (it2 = it1; …) { if (sg1 == sg2) continue; … }` over a set; toolkit
`for (j = i+1; …)` over a vector. Both enumerate unordered pairs exactly once.

**5.8 Three locals in `find_cont_muon_segment` are dead in both.**
`max_angle`, `max_ratio`, `max_ratio1_length` are assigned and never read —
prototype `:2344-2352`, toolkit `:1034-1043` with explicit `(void)` casts at
`:1048-1050`. The prototype's only reader is the commented-out
`if (max_ratio1 > 1.2 && …) flag_cont = false;` at `:2356-2357`.

**5.9 The `if (fiducial_utils)` guard cannot change the pick.** Toolkit `:925`
wraps the entire `+0.5` fiducial loop. If the utility is null, *every* candidate
loses the same 0.5, so the argmax is unchanged. Not a divergence.

**5.10 `segment_is_shower_trajectory(sg, 10*units::cm, …)` matches the
prototype default.** `ProtoSegment.h:85` declares
`is_shower_trajectory(double step_size = 10.*units::cm)`. The toolkit passing
10 cm explicitly is faithful. (The **1.0 cm** call is P3 — a different site.)

**5.11 `examine_direction`'s final-pass forcing clause is exact.** Prototype
`:1983` `if (get_flag_dir()==0 || is_dir_weak() || get_flag_shower() ||
flag_final)`; toolkit `:1208` `if (current_sg->dirsign() == 0 ||
seg_dir_weak(current_sg) || is_shower || flag_final)`. Same four clauses, same
order, same disjunction. The "final call is authoritative" property of pr/27 §6e
holds in both trees.

**5.12 The `1/4./2.` integer-division idiom is preserved, not fixed.**
Prototype `1/4./2.` is `(1 / 4.0) / 2.0` = 0.125, not integer division; the
toolkit's `1.0/4.0/2.0` is the same value. No trap introduced or removed.

**5.13 The DL input scaling is a faithful port of a prototype "hack".**
`dQdx_scale = 0.1`, `dQdx_offset = -1000` — the prototype declares them inline
with the comment "hack for now … (need to be synced with main code)"
(`NeutrinoID_DL.h:8-9`). The toolkit exposes both as config with exactly those
defaults (`common/clus.jsonnet:474`). Faithful, and the prototype's own caveat travels
with them.

---

## §6 Determinism

Four separate issues, and they do not point the same way.

**Fixed in the toolkit, still broken in the prototype** — the candidate ordering
of P5. Every producer that feeds the no-tie-break argmax was converted to
`ordered_nodes` / `IndexedVertexSet`. The prototype's remain pointer-keyed. The
toolkit is reproducible run to run; the prototype is not.

**Still broken in the toolkit** — P4 (`used_segments`, and it mutates), P8
(`map_in/out_segment_dirs`, argmax on ties), P11 (`existing_segments`,
unverified), P10 (`boost::edges` at `:773`, argmax on ties).

**Inherited and benign** — `map_seg_dir` (`:601`) and `used_vertices` (`:634`)
in `calc_conflict_maps` are both pointer-keyed and both iterated, but each loop
accumulates a commutative sum, so the total is order-independent. Listed so a
future sweep does not treat them as unfinished work.

**The verdict is not proven.** No repeat-run identity check was performed as
part of this audit. The claim "the toolkit is deterministic here" is a reading
of the containers, not a measurement — and P4, P8 and P10 are counter-examples
to it within this very stage.

---

## §7 Loose ends

Seven, all needing an owner decision or a measurement this audit did not make.

1. **How often does the DL re-ranker accept on SBND?** This decides whether
   §2.5's global-ladder verification describes production or a fallback. One
   census over the valfast manifest, counting `flag_dl_changed`, would settle
   it. Nothing else in §4.3 can be judged without it.
2. **Is an invalid-descriptor vertex reachable at `compare_main_vertices`?**
   P7's severity is entirely contingent on this — and so, now, is P5's, since
   the two couple through the skipped z-prior. If unreachable, P7 is dead code
   and P5 reduces to a rare-tie effect; if reachable, a candidate scoring 0 can
   beat scored candidates *and* ties become generic.
3. **Is `fit().valid()` true in the all-showers branch?** `improve_vertex` does
   not run there (`:2529`), so if no earlier `do_multi_tracking` wrote the
   vertex fits, P1 does not bite at `compare_main_vertices_all_showers` at all
   and reduces to a single-function finding.
4. **`eliminate_short_vertex_activities:1841`** — read the block and decide
   whether P11 is a real ordering dependence or a min-accumulation.
5. **Three formulations of "which end of the segment touches this vertex"** now
   coexist in one file: exact index match, nearest-endpoint distance (`:608`,
   `:834`), and `find_vertices` order (`:3427`). Whether they agree in all cases
   is a self-consistency question independent of the prototype.
6. **The `else if` on the backward-going beam penalty** makes
   `angle_beam < 45 && max_angle < 70` unreachable in both trees. Recorded in
   §2.3 as faithfully ported. Whether the prototype *intended* `if` is a
   question for the prototype's author, not a porting decision — M15 applies.
7. **The porting dictionary still has no section for this stage** — fourth
   audit in a row (pr/29, pr/30, pr/31, pr/32). Every divergence above is
   undocumented by construction, so §5 rule 4 applies to all of them and none
   is picked here.

Carried forward, unresolved: pr/30's P1–P7 and the `update_association`
coordinate question; pr/29's D1–D3; pr/31's P1–P15; doc pr/7's PID-persistence
fix, still proposed and not implemented.

---

## §8 Summary

| # | divergence | tier | reach |
|---|---|---|---|
| P1 | conflict + PCA angles from the skeleton-**snapped** vertex, not the continuous fit; 4 other sites in the same file read the fit | A | bounded by `fit_distance()`; main-cluster only; no effect where no fit ran |
| P2 | DL re-ranker (7 terms, accept ≥ 4.0) has no prototype counterpart — and picks SBND's vertex | A | the neutrino vertex, on every SBND event where DL accepts |
| P3 | shower-traj recheck: flag-read → recomputation; step 10 cm → 1 cm; two different dQ/dx scales | A | direction re-determination of main-vertex legs |
| P4 | `used_segments` pointer-set iterated while the body mutates PID | A | which vertices enter the candidate list |
| P5 | candidate order inverts (toolkit deterministic, prototype not); argmax has no tie-break | A | the pick on exactly-tied candidates — generic only in P7's regime |
| P6 | `flag_start`: exact index match (prototype UB) → nearest-endpoint; 3 formulations | A | sign of every direction conflict |
| P7 | invalid-descriptor candidates keep score 0 and stay in the argmax | A | unknown — reachability unverified |
| P8 | `map_in/out_segment_dirs` pointer-keyed argmax; ties pick `sg1`/`sg2` | A | `flag_check` and `angle_beam`, on ties |
| P9 | degenerate-direction guards added; NaN-rejection → explicit skip | A | bookkeeping only, on zero-length directions |
| P10 | three raw `boost::edges` remain; `:773` is an argmax | A | `max_length_muon`, on length ties |
| P11 | `existing_segments` pointer-set iterated at `:1841` | B | unverified |
| P12 | `map_cluster_main_candidate_vertices` not ported | A | diagnostic only — no algorithm reads it |

**Not divergences, but the biggest numbers in this doc** (§4): SBND runs
`vertex_z_prior_scale = 100` against the prototype's 200, and
`mip_dqdx_median = 48000` against 43e3. Both deliberate; both change this
stage's decisions more than most of P1–P12.

---

## §9 What is NOT claimed

* **No event was run.** No build, no A/B, no gate. Every "this changes the
  output" is an argument from source. None is measured.
* **§4's three consequences are readings of config, not of a run.** In
  particular, "the traditional global ladder may not execute" is a control-flow
  argument; the acceptance rate is loose end 1.
* **P7 and P11 are unverified in their central premise** — reachability and
  order-sensitivity respectively. They are listed as candidates, and marked.
* **P1 was initially written on a wrong premise and is corrected here.** The
  first draft called `wcpt()` "the raw skeleton point"; `MyFCN::UpdateInfo`
  (`MyFCN.cxx:485-496`) writes both members, so `wcpt()` after a fit is the
  fit *snapped to a skeleton node*. The finding survives, bounded by
  `fit_distance()` rather than by the fit displacement. Anyone re-reading this
  stage should check what writes a member before characterising a read of it.
* **The constant sweep has a known blind spot.** The extraction regex treats a
  comparison operator greedily, so a `units::cm` literal immediately preceded by
  `=` (as in `val -= 0.5*units::cm`) and any literal written with spaces
  (`W_SNAP_L * units::cm`) can be missed. Every constant claim in §2.2, §2.3,
  §2.4, §2.5 and §2.6 was therefore confirmed by **direct reading of both
  sides**, not by the sweep. The sweep's role was triage only.
* **`examine_direction` was not read end to end** — 428 prototype against 632
  toolkit lines. Its `flag_final` clause (§5.11) and its constant histogram were
  checked; its PDG-reassignment ladder (`:1232-1348`, `:1529-1548`) and its
  long-muon set population were not. A divergence could be hiding there.
* **Deghosting is Tier B throughout.** The constant sweep found only the `43e3`
  substitution of §4; the two functions were not read line by line.
* **§6's determinism verdict is judged, not proven.** No repeat-run check.
* **Stage 4's boundary functions are unaudited** — the `examine_structure_*`
  family in particular, which `determine_main_vertex` calls at `:2665` and which
  can move vertices before the final `examine_direction` sees them.
* **Nothing here is a recommendation.** All twelve divergences alter production
  output unconditionally, which is CLAUDE.md §5 rule 1: the owner decides, and
  §5 rule 4 forbids picking a reading on an undocumented divergence. Both
  readings are given for P2 and P6; the rest are stated as facts about the two
  sources.

  *§10 lifts this restriction for the four findings the owner's filter keeps:
  the owner asked for solutions, so §10 gives one per finding. Each is still a
  default-OFF knob, not a change of the production path.*

---

## §10 Owner filter — twelve divergences to four, with solutions

**The request.** *"Skip the ones that are improvements over the previous
prototype implementation, and only focus on the ones that are bugs or missing
from the port. Give a filtered list as well as the solutions."* Plus one
standing pre-clearance, in the owner's words:

> *"In the toolkit, we are missing id information for some data, so we have to
> use positions to do it, this is not a problem."*

That sentence resolves an entire class by itself — every place the toolkit
replaces a prototype Steiner-**index** equality with a **position** comparison
is cleared in advance and is not a divergence to report. It disposes of P6
outright.

**Which revision.** Everything below is re-derived against **`f8f2150a`**, the
committed HEAD, via `git show f8f2150a:<path>` — *not* the working tree. A
concurrent session is holding uncommitted edits to
`clus/inc/WireCellClus/NeutrinoPatternBase.h` and
`clus/src/NeutrinoTrackShowerSep.cxx` in this checkout, so working-tree line
numbers for those two files are not reproducible. `NeutrinoVertexFinder.cxx`,
`PRSegmentFunctions.cxx` and `TrackFitting.cxx` are clean at HEAD.

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit && git rev-parse --short HEAD   # f8f2150a
mkdir -p /home/xqian/tmp/pr32
for f in clus/src/NeutrinoVertexFinder.cxx clus/src/PRSegmentFunctions.cxx \
         clus/src/TrackFitting.cxx clus/src/MyFCN.cxx \
         clus/inc/WireCellClus/NeutrinoPatternBase.h \
         clus/inc/WireCellClus/PRVertex.h clus/inc/WireCellClus/PRCommon.h ; do
  git show f8f2150a:$f > /home/xqian/tmp/pr32/$(basename $f)
done
```

### §10.1 The filter

| # | verdict | why |
|---|---|---|
| **P1** | **KEEP → F1** | genuine port bug; **widened** — loose end 3 is now resolved and it bites at both sites |
| P2 | drop | deliberate toolkit design (doc pr/4), *per the rule*; the tuning concern survives — §10.6 |
| **P3** | **KEEP → F2** | genuine port bug; **three stacked changes at two sites**, with a root cause the doc did not reach |
| P4 | **closed** | **already fixed** at HEAD — `22249ff4` |
| P5 | drop | toolkit is deterministic, prototype is not — an improvement |
| P6 | drop | **pre-cleared by the owner** (index → position); also removes prototype UB |
| **P7** | **KEEP → F3** | latent bug, toolkit-only path, guard asymmetry |
| P8 | **closed** | **already fixed** at HEAD — `026a7501` |
| P9 | drop | the difference is **unobservable** — proven below |
| P10 | **closed** | **already fixed** at HEAD — `c05bc5f7` |
| P11 | **closed** | **resolved deliberately** and documented in the header — doc pr/28 §14.8 |
| **P12** | **KEEP → F4** | genuinely missing from the port — but display-only, do it on demand |

Twelve → **four**. Three of those four are code defects (F1, F2, F3); the
fourth (F4) is a missing diagnostic with no algorithmic reach.

---

### §10.2 F1 (= P1) — the conflict and PCA angles are measured from the snapped vertex

**Confirmed at HEAD**, and **wider than §3 states**.

The two raw-`wcpt()` sites survive the revision drift:

| site | HEAD anchor | prototype |
|---|---|---|
| `calc_conflict_maps` in-dirs | `NeutrinoVertexFinder.cxx:686` | `NeutrinoID_track_shower.h:1804` `get_fit_pt()` |
| `calc_conflict_maps` out-dirs | `:694` | `:1808` `get_fit_pt()` |
| `…_all_showers` PCA point cloud | `:388` | `:1468` `get_fit_pt()` |
| `…_all_showers` axis projection | `:405-407` | `:1480` `get_fit_pt()` |
| `…_all_showers` z tie-breaks | `:446`, `:459`, `:482`, `:526`, `:535-536` | `:1542`, `:1551`, `:1567`, `:1574` `get_fit_pt()` |
| `…_all_showers` path endpoints | `:455` | `:1504-1505` `get_closest_wcpoint(get_fit_pt())` |

The prototype uses `get_fit_pt()` at **every** one of those. So the finding is
not "two functions"; it is **two functions and nine expressions**, of which the
last row is arguably equivalent (snapping the fit point to the nearest Steiner
node ≈ `wcpt` after `UpdateInfo`) and the other eight are not.

**Loose end 3 is RESOLVED: `fit().valid()` IS true at this point.** §3 left this
open and scoped P1 conditionally. Traced at HEAD:

1. `Fit::valid()` is `index >= 0` (`PRCommon.h:145-148`).
2. `TrackFitting::multi_trajectory_fit` fits the vertex and then explicitly
   sets `vertex_fit.index = -1` (`TrackFitting.cxx:3896`) — a faithful port of
   the prototype's `vtx->set_fit(init_p, 0, -1, …)`
   (`PR3DCluster_multi_track_fitting.h:342`, third argument). Read alone, this
   says every fitted vertex ends up *invalid*.
3. But `do_multi_tracking` is not finished: under `if (flag_dQ_dx)` it calls
   `reset_fit_prop()` on every vertex (preserving `flag_fix`) and then runs a
   **third** `form_map_graph` (`:8350`), which re-assigns
   `start_v->fit_index(count)` / `end_v->fit_index(count)` to every vertex
   still carrying `-1` (`:3247-3265`).
4. `Fit::reset()` clears `index`, `flag_fix`, `range` — **not `point`**
   (`PRCommon.h:134-138`). The fitted position survives the reset.

So after any `do_multi_tracking(…, flag_dQ_dx = true, …)` — which is what
`improve_vertex` and stage 3 both run — every vertex attached to a fitted
segment has `fit_index >= 0` and a fitted `fit().point`. **P1 therefore bites
in the all-showers branch too**, not only in the main cluster, and the §3
hedge ("if it is not, P1 does not bite at that site at all") can be struck.

**Why it is a bug and not a choice.** `grep -n "fit().valid() ? "` on the
`f8f2150a` snapshot returns **23 vertex-position reads** through the idiom
(`:84`, `:886`, `:993`, `:1633-1634`, `:1789-1790`, `:2644`, `:2709`, `:2741`,
`:2855`, `:3072`, `:3077`, `:3128`, `:3148`, `:3154`, `:3269`, `:3388`,
`:3446`, `:3540`, `:3589`, `:3603`, `:3630`) against the nine expressions
tabulated above that read `wcpt()` raw. The file does not apply its own
convention uniformly, and the prototype has exactly one convention.

**Magnitude.** The gap is `Vertex::fit_distance()` (`PRVertex.h:85`) — the
continuous fit minus its snap to the nearest Steiner node — because
`MyFCN::UpdateInfo` writes both members (`MyFCN.cxx:487` `vtx_fit.point =
fit_pos`, `:496` `vtx->wcpt().point = vtx_new_pt`). It is a lattice quantisation, not
a fitted-vs-unfitted swap. It matters because these angles feed the
`<35 / <70 / <85 / <110` ladder whose rungs are worth `+5 / +3 / +1 / +0.25`
before the `/4`, against topology terms spaced `0.125` apart.

#### Solution F1

Add one knob, `vertex_dir_use_fit_point` (C++ default **false** = today's
behaviour), on `TaggerCheckNeutrino` → `PatternAlgorithms`. When true, replace
the nine raw reads with the file's own existing idiom, hoisted to one helper so
there is a single convention:

```cpp
// Faithful port of ProtoVertex::get_fit_pt().  The prototype's ctor sets
// fit_pt = wcpt (ProtoVertex.cxx:17-19), so get_fit_pt() is never
// uninitialised; PR::Vertex has no such ctor, so the wcpt fallback stands in
// for the un-fitted case.  doc pr/32 §10.2.
static inline WireCell::Point vtx_fit_pt(const VertexPtr& v)
{
    return v->fit().valid() ? v->fit().point : v->wcpt().point;
}
```

Leave `:455` (`do_rough_path`) alone: the prototype's own counterpart snaps to
a Steiner node there, so `wcpt()` is the faithful read.

**Watch item, not part of the fix.** `PR::Vertex` has no constructor
initialising `m_fit.point` from `m_wcpt.point`, so a vertex created after the
last fit and never fitted carries `fit().point == (0,0,0)` — the fallback is
load-bearing, and dropping it (e.g. "just read `fit().point`, the prototype
does") would put the origin into a score ladder. Initialising it in
`make_vertex()` would be the closer port, but that reaches outside this stage.

**Gate.** Not expected byte-identical when on — `fit_distance()` is nonzero on
127/127 vertices of evt 388 (doc pr/28 A4). Standard nueCC48 arm.

---

### §10.3 F2 (= P3) — the shower-trajectory recheck, and why a naive parity patch would kill the block

**Confirmed at HEAD**, `NeutrinoVertexFinder.cxx:2409-2410`, and the doc
**understates it**: the same substitution appears at a **second site the doc
does not list**, `:2337`.

| | prototype | toolkit HEAD |
|---|---|---|
| outer gate, `improve_vertex` direction pass | `sg1->get_flag_shower_trajectory()` — **stored flag** (`NeutrinoID_improve_vertex.h:248`) | `segment_is_shower_trajectory(sg1, 10*units::cm, m_mip_dqdx)` — **recomputed** (`:2337`) |
| outer gate, shower-traj recheck | `sg->get_flag_shower_trajectory()` — **stored flag** (`:287`) | `segment_is_shower_trajectory(sg, 10*units::cm, m_mip_dqdx)` — **recomputed** (`:2409`) |
| inner test | `sg->is_shower_trajectory()` — recompute, **step 10 cm** default (`ProtoSegment.h:85`), internal scale **`50000/units::cm`** (`ProtoSegment.cxx:566`) | `segment_is_shower_trajectory(sg, **1.0 cm**, **m_mip_dqdx_median**)` (`:2410`) |

Scale dictionary, so the third row is unambiguous: the prototype's *internal*
`50000/units::cm` in `is_shower_trajectory` is the toolkit's `mip_dQ_dx`
parameter (default `50000/units::cm`, SBND `m_mip_dqdx = 56000`). The
prototype's `43e3/units::cm` is `m_mip_dqdx_median` (SBND 48000). So the
**outer** toolkit call carries the faithful scale and the **inner** one does
not — it passes the 43e3-analog where the function wants the 50000-analog.

**The load-bearing observation.** The obvious parity patch — "make the inner
call `(10*units::cm, m_mip_dqdx)` like the prototype" — **makes the whole block
dead code**. With the outer gate also recomputing at `(10 cm, m_mip_dqdx)`, the
inner `!segment_is_shower_trajectory(sg, 10 cm, m_mip_dqdx)` is the negation of
a condition established one line earlier, so it is always false and the body
never runs.

The premise that argument rests on, stated so it can be attacked: the two calls
are **adjacent lines** with nothing between them, and
`segment_is_shower_trajectory` reads only `seg->fits()` and derived lengths /
median dQ/dx — its one side effect is `seg->set_flags(kShowerTrajectory)`
(`PRSegmentFunctions.cxx:1074`), which the function never reads back. So for
fixed arguments it returns the same value twice.

The 1.0 cm step is therefore not a typo: it is what keeps the block alive after
the outer gate was changed from a flag read to a recomputation. **The two
changes are one change**, and it must be undone as one.

**Root cause — the toolkit's `kShowerTrajectory` flag cannot be cleared by the
test that sets it.** This is why reading the stored flag was not an option for
whoever wrote `:2409`:

* prototype `is_shower_trajectory()` **resets then sets**: line 544
  `flag_shower_trajectory = false;`, line 608 `… = true` if it fires. Every
  call re-caches, including *demoting* a segment that no longer qualifies.
* toolkit `segment_is_shower_trajectory()` **only sets** — `set_flags` at
  `:1074`, no `unset_flags` anywhere in it.

Explicit demotions *are* ported and are complete — prototype
`set_flag_shower_trajectory(false)` at `NeutrinoID_track_shower.h:2058`,
`:2102`, `:2109`, `:2178` map exactly onto toolkit `unset_flags` at
`NeutrinoVertexFinder.cxx:1334`, `:1395`, `:1472` (the prototype's `:2102` and
`:2109` are the two arms of one if/else that the toolkit hoists). Repo-wide at
HEAD there are exactly one set and three unset sites. **The only missing clear
is the one inside the test itself.**

Consequence, in the prototype's own terms: the recheck block exists to say
*"this leg was labelled a trajectory shower, but a fresh look says it is not —
re-determine its direction as a track."* The prototype **also demotes the
label**, so `get_flag_shower()` goes false downstream. The toolkit re-determines
the direction and **leaves the segment labelled a shower**.

#### Solution F2

One knob, `shower_traj_recheck_parity` (C++ default **false**), doing three
things together — they cannot be separated without killing the block:

1. In `segment_is_shower_trajectory`, mirror the prototype's re-cache:
   `if (flag) seg->set_flags(kShowerTrajectory); else seg->unset_flags(kShowerTrajectory);`
2. At `:2337` and `:2409`, read the stored flag
   (`sg->flags_any(SegmentFlags::kShowerTrajectory)`) instead of recomputing.
3. At `:2410`, recompute at the prototype's parameters —
   `segment_is_shower_trajectory(sg, 10*units::cm, m_mip_dqdx)`.

With (1) and (2) in place the block is alive again for the prototype's reason:
the stored flag is stale (it predates this pass's refit) and the fresh 10 cm
test can disagree with it.

**Checked, because step (2) at `:2337` is the weaker half.** `:2337` sits inside
`if (!sg1->particle_info())` — segments that have never been typed — so "read
the stored flag" is only faithful if the flag means the same thing in both
trees for such a segment. It does:

* `segment_is_shower_topology` writes **only** `kShowerTopology`
  (`PRSegmentFunctions.cxx:389`); the prototype's `is_shower_topology()` writes
  only `flag_shower_topology` (`ProtoSegment.cxx:320`, `:454`, `:531`). Neither
  touches the trajectory flag, so `:2316` does not contaminate `:2337`.
* Stage 3 is a faithful port of the gating: toolkit
  `NeutrinoTrackShowerSep.cxx:54-62` runs `segment_is_shower_topology`, then
  `segment_is_shower_trajectory(seg, 10 cm, m_mip_dqdx)` **only if not
  topology** — identical to prototype `separate_track_shower`
  (`NeutrinoID_track_shower.h:9-22`).
* A segment that was shower-**topology** in stage 3, or that was created after
  stage 3, therefore never had the trajectory test run and carries **false** —
  and it carries false on the prototype side too, because
  `ProtoSegment`'s ctor initialises `flag_shower_trajectory(false)`
  (`ProtoSegment.cxx:24`) exactly as the toolkit's flags zero-initialise.

So the stored flag at `:2337` is "what stage 3 or a later explicit demotion
left", identically on both sides, and step (2) is the faithful read rather than
a silent reroute of untyped segments into `determine_dir_track`.

**Blast radius — the reason this must be default-OFF and gated hard.** Step (1)
changes a flag with **31 `kShowerTrajectory` occurrences in
`NeutrinoTrackShowerSep.cxx` alone, every one of them a read**, plus
`get_flag_shower()`-equivalent sites in `PrDisplayDump.cxx` and
`MultiAlgBlobClustering.cxx`. It is not a local change. **`NeutrinoTrackShowerSep.cxx`
is one of the two files the concurrent session is editing** — coordinate before
touching it.

**Gate.** nueCC48, knob-off must be byte-identical (steps 1–3 are all inside
`if (m_shower_traj_recheck_parity)` / the flag-write branch), knob-on measured
with a counter on how often the stored flag and the fresh test disagree.

---

### §10.4 F3 (= P7) — invalid-descriptor candidates keep score 0 and stay in the argmax

**Confirmed at HEAD**, `compare_main_vertices`. The guard is applied to two
loops and omitted from four places:

| block | HEAD anchor | guarded? |
|---|---|---|
| proton-topology pre-pass | `:812` | **yes** |
| z-prior + per-segment ladder | `:894` | **yes** |
| `min_z` scan | `:888-890` | **no** |
| fiducial `+0.5` | `:940-947` | **no** |
| conflict `− conflicts/4` | `:950-953` | **no** |
| argmax | `:956-963` | **no** |

A candidate with an invalid descriptor reaches the argmax carrying
`0 + (0.5 if in FV) − conflicts/4`, and real candidates routinely score
negative because the z-prior and the conflict penalty are both subtractive. The
unguarded `min_z` scan is the same defect one step earlier: such a candidate's
z still sets the origin of everyone else's prior.

**Reachability — argued, not measured.** Candidates are collected from
`ordered_nodes(graph)` (`:2588`, `:2559`, `:2578`), so every one is
descriptor-valid at collection; between collection and use only
`examine_main_vertices_local` runs, and it retypes segments and pushes
chain-end graph vertices — it does not call `remove_vertex`. On that reading
the path is dead code. That is a control-flow argument, not a measurement, and
this doc's §9 discipline says so out loud: **do not call the fix "free" or
"byte-identical" — call it *expected* byte-identical and gate it like anything
else.**

#### Solution F3

Filter once at function entry instead of sprinkling four more guards:

```cpp
// doc pr/32 §10.4: two loops guarded on descriptor_valid(), four not.
// Filtering here makes the whole scorer see one candidate set.
std::vector<VertexPtr> scored;
for (auto vtx : vertex_candidates) if (vtx->descriptor_valid()) scored.push_back(vtx);
```

and score/argmax over `scored`. Knob `main_vertex_require_descriptor`, C++
default **false**. Add a one-line DEBUG counter for how many candidates the
filter drops — if it is 0 everywhere, the finding is confirmed dead and the
knob can be retired rather than carried forever.

---

### §10.5 F4 (= P12) — `map_cluster_main_candidate_vertices` is not ported

**Confirmed.** Prototype records the per-cluster candidate list before
filtering (`NeutrinoID_track_shower.h:1332`) and exposes it
(`NeutrinoID.h:1720`). Re-checked at HEAD: the toolkit has no equivalent, and
**every** prototype consumer is an app-level output-tree filler —
`wire-cell-prod-{nue,nnbar,nue-mt,pi0,nue-port}.cxx`. No algorithm reads it.

It is the one item that literally matches *"missing from the port"* while
changing no decision. Kept in the list because the owner's rule names it, and
because it is exactly what a vertex-tuning display would need first — doc
pr/27 §6 records this row as "candidate list, per-cluster and global →
**nothing**".

#### Solution F4

Do **not** add a member to the algorithm classes. Emit it where the display
already goes: one array in `PrDisplayDump` (default-OFF dumper, so this is a
diagnostic-schema change, not a physics change), written from
`determine_main_vertex` just before `examine_main_vertices_local` — the same
point the prototype records it. **Implement on demand only**; it costs a
schema bump and buys nothing until someone is tuning the pick.

---

### §10.6 Dropped, and why

**P2 — the DL re-ranker.** Dropped **per the rule**, not because the question
went away. It is a deliberate toolkit design (doc pr/27 §6, adopted in pr/4),
the prototype's 177-line legacy path is retained and faithful (§2.8), and
"prototype fidelity" is not the right frame for a component with no
counterpart. **The reading (b) concern survives as a tuning item and must not
be lost here:** `W_MAIN`, `W_CLEN` and the 4.0 acceptance floor were not
derived on SBND, and the dominant term `s_dl × 1000` is scaled for uBooNE score
magnitudes. That belongs with doc pr/2 gap G1, not with a port-fidelity fix.

**P5 — candidate ordering.** The toolkit is deterministic (`ordered_nodes`,
`IndexedVertexSet`) and the prototype is not (pointer-keyed `std::map`). That
is an improvement. Its coupling to P7 is addressed by F3.

**P6 — `flag_start`.** **Pre-cleared by the owner**: the toolkit lacks the
Steiner-index information and must use positions. Independently, the prototype
reads `flag_start` uninitialised when neither index matches
(`NeutrinoID_track_shower.h:1627-1631`, and three more at `:119`, `:180`,
`:343`/`:354`), so the toolkit also removes real UB.
The third formulation (`find_vertices` order in the DL veto, `:3427`) is the
same question asked a third way and agrees wherever the vertex is an endpoint.

**P9 — degenerate-direction guards.** Dropped, and the doc's own hedge can be
retired: the difference is **not observable**. §3 says the `continue` at
`find_cont_muon_segment:1014` skips `max_ratio1` bookkeeping the prototype
would still do. Re-read at HEAD: `max_ratio1` is written at `:1058` and its
only reader is its own `>` comparison at `:1057`; after the loop it is never
read (the prototype's only consumer, `if (max_ratio1 > 1.2 …) flag_cont =
false;`, is commented out at `NeutrinoID_track_shower.h:2356-2357`). Both
`max_ratio1` and `max_ratio1_length` are dead — the toolkit even `(void)`-casts
the latter at `:1066`. Skipping dead bookkeeping changes nothing.

---

### §10.7 Already fixed or resolved since `4f2e7303` — closed

All four determinism findings are gone at HEAD. `4f2e7303` and `f8f2150a` are
the same day; the audit simply predates these commits.

| # | what §3 reported | state at `f8f2150a` | fixed by |
|---|---|---|---|
| P4 | `used_segments` pointer-set iterated while the body mutates PID | sorted into `ordered_used_segments` by graph index before the loop, `:2967-2978`, with the reason in-code at `:2949` | `22249ff4` (doc pr/28 §14) |
| P8 | `map_in/out_segment_dirs` pointer-keyed, argmax on ties | both declared `std::map<SegmentPtr, geo_vector_t, PR::SegmentIndexCmp>`, `:668-669` | `026a7501` (doc pr/28 §15) |
| P10 | raw `boost::edges` driving `max_length_muon` | **zero** `boost::edges` remain in the file; `:790` is `ordered_edges(graph)` with the reason in-code at `:789` | `c05bc5f7` (doc pr/28 §11) |
| P11 | `existing_segments` pointer-set, order-sensitivity unverified | **resolved deliberately**: the block at `:1841-1870` takes a running min over three distances — order-insensitive — and the container must stay pointer-keyed because `Segment::get_graph_index()` is not guaranteed unique. Rationale is in `NeutrinoPatternBase.h` above the `eliminate_short_vertex_activities` declaration | doc pr/28 §14.8 |

P11 also answers **loose end 4** in full: it is a min-accumulation, not an
ordering dependence, and the pointer keying is a considered decision with a
written justification — leave it alone.

---

### §10.8 Corrections to this doc

Found while re-verifying. Listed so the next reader does not inherit them.

1. **§3 P10 says "three raw `boost::edges` remain".** There were **eight** at
   `4f2e7303` (`:773`, `:1105`, `:1552`, `:1844`, `:2029`, `:2295`, `:3137`,
   `:3515`). The three listed were the ones tabulated, not the ones present.
   Moot at HEAD — all eight are `ordered_edges` — but the count was wrong when
   written.
2. **§3 P3 lists one recomputation site; there are two.** `:2337` in the
   `improve_vertex` direction pass makes the same substitution as `:2409`
   against prototype `NeutrinoID_improve_vertex.h:247`.
3. **§3 P3 does not reach the root cause.** The recomputation is not
   gratuitous: the toolkit's `kShowerTrajectory` cannot be cleared by the test
   that sets it, so a flag read would return a monotone OR of every past call.
   §10.3.
4. **§7 loose end 3 is resolved, and the answer widens P1.** `fit().valid()` is
   true after `do_multi_tracking` because the pre-dQ/dx `reset_fit_prop()` +
   third `form_map_graph` re-assign every vertex a fit index. §10.2.
5. **§7 loose end 4 is resolved** — P11, §10.7.
6. **§3 P9's "bookkeeping differs" is unobservable** — `max_ratio1` has no
   consumer after the loop. §10.6.
7. **§3 P1's site list is incomplete** — nine expressions, not two functions'
   worth of two. §10.2.
8. **The `4f2e7303` banner at the top of this doc is stale** for §3's anchors
   in the four closed findings. §1–§9 are still correct *as of that revision*;
   §10 is the current-HEAD view.

---

### §10.9 What §10 still does not claim

* **No event was run this round either.** No build, no A/B, no gate. F1's
  "not byte-identical when on" is a prediction from `fit_distance() > 0`
  measured in doc pr/28 A4, not a run of this knob.
* **F3's unreachability is an argument from control flow**, not a measurement.
  That is precisely why the counter is part of the proposed solution.
* **F2's blast radius is counted, not measured** — 31 `kShowerTrajectory`
  occurrences in `NeutrinoTrackShowerSep.cxx` is a grep, and how many change
  answers is unknown until the knob runs.
* **F2's "a naive inner-at-10 cm patch kills the block" is an argument from
  purity**, not from a run. `segment_is_shower_trajectory` has one side effect
  (`set_flags`) which it never reads back; if some future edit makes it read
  its own flag, the argument stops holding.
* **The four closed findings are closed on a code read**, not on a determinism
  re-run. Doc pr/28 §11/§14/§15 carry those measurements; this doc only
  verifies that the code at `f8f2150a` matches what they claim.
* **Nothing outside stage 4 was re-audited.** `examine_direction`'s
  PDG-reassignment ladder, the deghosting pair and the `examine_structure_*`
  family remain as §9 left them.
* **§4's SBND operating point is untouched by this filter.**
  `vertex_z_prior_scale = 100` against the prototype's 200, and
  `mip_dqdx_median = 48000` against 43e3, are still the biggest numbers in this
  doc and are still deliberate config, not port defects.

---

## §11 Implementation — all four fixed, **all four SBND production defaults ON**

**Status: SHIPPED.** Toolkit **`407c5ba9`** (parent `1e169602`). This section
supersedes §10 where they disagree; §10's one substantive error is corrected in
§11.7.

Each finding is a knob on `TaggerCheckNeutrino` → `PatternAlgorithms`, **C++
default `false` = today's path**, and each is set **`true`** in
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`. The flip is free on this
manifest: every arm is byte-identical (§11.2), and every knob is nonetheless
*engaged* rather than inert (§11.3).

| knob | was | what it restores |
|---|---|---|
| `vertex_dir_use_fit_point` | P1 | eleven expressions measure from the continuous fit, not the Steiner snap |
| `shower_traj_recheck_parity` | P3 | stored-flag gates, 10 cm inner test, and a **clearable** `kShowerTrajectory` |
| `main_vertex_require_descriptor` | P7 | one candidate set across all six scoring blocks |
| `main_vertex_candidate_flag` | P12 | `kMainCandidate`, the missing `map_cluster_main_candidate_vertices` |

### §11.1 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit && git rev-parse --short HEAD   # 407c5ba9
wcbuild                                   # BOTH build/ and local/lib -- see §11.6
./build/clus/wcdoctest-clus               # 95 cases / 984 assertions

cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# Arms.  The four env vars are TRI-STATE: unset = cfg default, 1 = force on,
# 0 = force off -- which is what lets the gate arm turn them back off now that
# the SBND operating point ships them on.
SBND_MAX_JOBS=5 \
SBND_VERTEX_DIR_USE_FIT_POINT=0 SBND_SHOWER_TRAJ_RECHECK_PARITY=0 \
SBND_MAIN_VERTEX_REQUIRE_DESC=0  SBND_MAIN_VERTEX_CANDIDATE_FLAG=0 \
  ./run_pr_chain_batch.sh work-nuecc48-prod0803 work-pr32r2-off48 data
# ... and 1/1/1/1 for work-pr32r2-allon48, plus the three single-knob arms.

PR32_JOBS=16 python3 pr32_cmp.py work-pr32r2-off48 work-pr32r2-allon48
```

`pr32_cmp.py` (this round, committed here) compares pctree-pr **member-content**
hashes via `abtest/hash_archive.py`, both nusel TSVs, and the PR30/PR32AUDIT
counters. Two traps it now handles, both of which produced a wrong answer first:
`hash_archive.py` prints the **file path** in its output, so hashing its stdout
makes every arm "differ"; and `f1_mean_cm` is a per-event mean, so summing it
over 48 events reports 28.9 cm for what is really 0.61 cm.

### §11.2 Gate — every arm 48/48

Baseline **`work-pr31-f2off48`**, the doc pr/31 §11 knob-off arm at the parent
`1e169602`. (Not `work-pr30-f2on`, and emphatically not `work-vfnuecc48-0804`,
which predates the `oov_prototype_parity` flip.)

| comparison | pctree-pr member hashes | nusel-table / nusel-events |
|---|---|---|
| baseline vs `work-pr32r2-off48` | **48/48 identical** | identical |
| `off48` vs `work-pr32r2-f1on48` | **48/48 identical** | identical |
| `off48` vs `work-pr32r2-f2on48` | **48/48 identical** | identical |
| `off48` vs `work-pr32r2-f34on48` | **48/48 identical** | identical |
| `off48` vs `work-pr32r2-allon48` | **48/48 identical** | identical |

Every `PR30AUDIT` counter is unchanged between baseline and knobs-off, which is
the independent check that the knob-off path really is the old path and not a
coincidence of hashes. 48/48 `rc=0` in every arm.

Config:

* knobs off vs `1e169602` `cfg/` — **byte-identical**, md5
  `0c24101b93143a06e8e326298af821de` both sides;
* knobs on — all four keys present, +162 bytes;
* **after the SBND flip, the compiled config is IDENTICAL to the TLA-forced
  all-on config**, so a bare production run *is* `work-pr32r2-allon48`. Confirmed
  end to end on three events (`work-pr32r2-flip3`): `PR32AUDIT … knobs[use_fit_point=true
  traj_parity=true require_desc=true cand_flag=true]` with no TLA, and the
  pctree hashes match `allon48` on all three.

**Scope of the claim.** "Byte-identical" here means the pctree-pr archive member
contents and the two nusel tables, on 48 nueCC data events. It is not a
statement about the 1000-event population — the valfast baseline is still stale
(§11.8).

### §11.3 Engaged, not inert — what the counters say

Summed over the 48 events of `work-pr32r2-off48` (legacy) and
`work-pr32r2-allon48`:

| counter | legacy | all-on | reading |
|---|---|---|---|
| `f1_reads` / `f1_fit_valid` | 5047 / **5047** | 5047 / 5047 | **every** read has a valid fit |
| `f1_mean_cm` | 0.613 | 0.612 | mean \|fit − wcpt\| |
| `f1_max_cm` | 11.411 | 11.411 | worst single vertex |
| `f2_gate` | 117 | 117 | outer-gate evaluations |
| `f2_disagree` | **22** | 0 | stored flag vs a fresh 10 cm test |
| `f2_body` | 7 | 4 | recheck body actually runs |
| `f2_demoted` | 0 | **4** | segments the refresh takes out of shower-trajectory |
| `f3_cand` / `f3_dropped` | 2219 / **0** | 2219 / 0 | invalid-descriptor candidates |
| `f4_flagged` | 0 | **3774** | vertices marked `kMainCandidate` |

Three of these are results, not bookkeeping:

1. **`f1_fit_valid == f1_reads == 5047` settles §7 loose end 3 by measurement.**
   §10.2 argued from the code that the third `form_map_graph` re-validates every
   vertex's fit index; 5047/5047 confirms it. P1 was live at **both** sites,
   including the all-showers branch, exactly as §10.2 predicted.
2. **F1's magnitude is 0.613 cm mean, 11.41 cm max — over all eleven
   expressions, not over the ladder alone.** The counter fires on every call to
   the helper, which includes the `pts.push_back` loop over *every* graph vertex
   of the cluster in `compare_main_vertices_all_showers`; those reads feed a PCA
   point cloud, not a direction vector or an argmax. So 0.613 cm is the mean
   Steiner quantisation across the whole substituted population, and the subset
   that actually feeds the `calc_conflict_maps` ladder was **not** separated.
   Read it as "the substitution moves points by ~6 mm", not as "the conflict
   ladder sees a 6 mm shift". Separating the two would need a per-site counter
   and is not done here.
3. **`f3_dropped == 0` over 2219 candidates makes P7 measured dead code.** §10.4
   could only offer a control-flow argument and explicitly refused to call the
   fix free; the counter is the measurement that argument was missing. The knob
   can be retired once a wider census agrees — recorded rather than done.

`f2_disagree = 22` is the direct evidence for F2: on 48 events the stored label
and a fresh 10 cm test disagree 22 times, which is precisely the population the
prototype's flag read and the toolkit's recomputation split on.

**But read `f2_gate = 117` against `f2_body = 7` before weighting F2 heavily.**
The outer gate is evaluated 117 times in 48 events and the recheck body runs 7 —
so ~110 evaluations pass the gate and are then vetoed by the inner 1.0 cm test,
and under parity the body runs 4 times instead of 7. This block is *rare*: it
touches single digits of segments per 48 events on either setting. That matters
because F2 is the one knob whose fix reaches outside this stage — the clearable
`kShowerTrajectory` has 31 downstream readers in `NeutrinoTrackShowerSep.cxx` —
so "byte-identical on 48 events" is thinner evidence here than for F1, F3 or F4:
the block simply does not fire often enough on this manifest to exercise the
flag's downstream consumers. F2 is the knob most in need of the 1000-event
census (§11.8).

### §11.4 What each fix actually changed

**F1.** One helper, `pr32_vtx_pt(v, use_fit)`, at eleven expressions:
`calc_conflict_maps` in- and out-direction vectors (`NeutrinoVertexFinder.cxx`,
prototype `:1804`/`:1808`); and in `compare_main_vertices_all_showers` the PCA
point cloud (`:1468`), the axis projection (`:1480`), the four forward-z
tie-breaks (`:1542`, `:1551`, `:1567`, `:1574`) and the Steiner path endpoints
(`:1504-1505`). The fallback to `wcpt()` stays and is load-bearing.

**F2.** Three coupled edits, plus one in a different file:

* `pr32_shower_traj_gate()` at both outer gates — reads the stored flag under
  parity, recomputes otherwise, and counts the disagreement on the legacy path
  (asking for the fresh answer under parity would re-introduce the mutation the
  knob removes);
* the inner test at `(10 cm, m_mip_dqdx)` under parity, `(1.0 cm,
  m_mip_dqdx_median)` otherwise;
* `PR::g_shower_traj_refresh_flag` — a process-wide flag written once per event
  by `TaggerCheckNeutrino`, the same transport as `g_graph_endpoint_policy` —
  makes `segment_is_shower_trajectory` clear `kShowerTrajectory` on entry,
  mirroring `ProtoSegment.cxx:544` *before* its own `length > 50 cm` early
  return, so even the early-out path demotes.

**F3.** A local filtered copy at function entry; the caller's
`std::vector<VertexPtr>&` is never mutated.

**F4.** `VertexFlags::kMainCandidate = 1<<2`, set in `determine_main_vertex` at
the prototype's recording point, emitted by `PrDisplayDump` as
`"main_candidate"`. A dump taken without the knob reads *no candidate
information*, not *no candidates*.

### §11.5 Tests

`build/clus/wcdoctest-clus`: **95 cases / 984 assertions, 0 failed** (was
91/963). Four new cases, and the F2 pair is **revert-proven**: replacing
`if (g_shower_traj_refresh_flag)` with `if (false)` fails exactly
*"clus pr32 F2 refresh clears the flag on a negative answer"* and nothing else.

* the two `segment_is_shower_trajectory` flag polarities — monotone by default,
  demoting under refresh, with `kShowerTopology` proven to survive (the clear is
  bit-masked, not `clear_flags()`);
* `kMainCandidate` is independent of `kNeutrinoVertex` in both directions —
  `MultiAlgBlobClustering` and `PrDisplayDump` both branch on the latter alone
  and would render every candidate as the neutrino vertex if the bits collided;
* `fit_distance()` is `|fit − wcpt|`, an unfitted vertex reports
  `valid() == false` with `fit().point == (0,0,0)` (which is why the fallback
  exists), and `Fit::reset()` clears the index but **not** the point — the
  property that lets the third `form_map_graph` re-validate an already-fitted
  vertex.

### §11.6 Two mistakes, recorded rather than hidden

**A first set of five arms was voided.** In this tree
`toolkit-dev/.envrc:26` puts every `toolkit/build/<pkg>` on `LD_LIBRARY_PATH`
**before** `local/lib`, so wire-cell loads `build/clus/libWireCellClus.so`, not
the installed copy. A `wcb build --targets=wcdoctest-clus` run to revert-prove a
doctest — with the source deliberately reverted — relinked that library **under
a running batch**: 7 events died with `E [sys] Failed to load …: file too
short`, and events starting after the relink ran the reverted code. The dead
arms are quarantined under `sbnd_xin/VOID-pr32-round1/` and were re-run as
`work-pr32r2-*`. **Rule: builds and arms are mutually exclusive; check
`pgrep -fc wire-cell` first.** This also corrects the long-standing note that
`build/` is not on `LD_LIBRARY_PATH` — it is, and it wins.

**F1's gap was first reported as 28.9 cm.** `f1_mean_cm` is a *per-event* mean;
the comparison script summed it over 48 events. The true figure is
**0.613 cm** mean, 11.41 cm max. Caught because 28.9 cm was physically
implausible for a Steiner-lattice quantisation — the number looking wrong was
the only signal.

### §11.7 Correction to §10

**§10.2 said to leave the `do_rough_path` call alone**, on the grounds that the
prototype snaps to a Steiner node there so `wcpt()` is the faithful read. That
is wrong: `do_rough_path` performs the snap **itself**, via
`cluster.kd_steiner_knn(1, point, "steiner_pc")`
(`NeutrinoPatternBase.cxx:97-104`). It is therefore the exact counterpart of the
prototype's `get_closest_wcpoint(get_fit_pt())`, and the faithful argument is
the fit point. F1 includes that site; the count is **eleven** expressions, not
the nine §10.2 tabulated plus a spared tenth.

Also: §10.3 listed one recomputation site in the doc body and found the second
at `:2337` in the same section — both are fixed, and the fix depends on the flag
becoming clearable, which is a change in `PRSegmentFunctions.cxx`, not in
`NeutrinoVertexFinder.cxx`. The audit's blast-radius note stands: 31
`kShowerTrajectory` reads in `NeutrinoTrackShowerSep.cxx` are downstream of that
clear, and the 48-event evidence that they do not move is `f2_demoted = 4` with
48/48 byte-identical output.

### §11.8 What is still NOT claimed

* **48 events, not 1000.** The valfast/1000 population gate has been stale since
  round 6, and the count of knobs riding on it is now **ten** — five from doc
  pr/30 §12, one from pr/31 §11, four from here. That is past the point where
  "stale baseline" is a footnote: regenerating it should be scheduled, not
  mentioned. "Byte-identical" in §11.2 is a statement about this manifest only.
* **`f3_dropped = 0` is 48 events of evidence, not a proof.** P7 stays behind a
  knob until a wider census agrees; retiring it is a decision, not a
  consequence.
* **F1 is byte-identical on this manifest but it is not a no-op.** It moves a
  0.6 cm measurement point on 5047 reads. On a larger sample it should be
  expected to change vertices; the right reading of §11.2 is "no event in this
  48 crossed a ladder rung", not "this cannot change anything".
* **`f2_disagree = 22` is not 22 changed events.** It is 22 gate evaluations
  where the two answers differ; only 4 of them reached a demotion and none
  changed an output here.
* **No run-to-run repeat check** was done in this round. The arms are single
  runs; the determinism evidence is doc pr/28's, not this doc's.
* **`main_candidate` has no viewer yet.** `PrDisplayDump` emits the field; the
  Bokeh PR display does not draw it.
