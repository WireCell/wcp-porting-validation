# doc pr/138 — the shower splitter: MASTER PLAN (scan → implement → optimise)

**Status: PLAN, 2026-08-31. No code, no arm, no knob. This file is the spine for
the whole splitter campaign; the measurements it rests on are doc pr/137 §10–§15.
Owner: *"we will devote some session time to do the scan... we can then move back
to the implementation and optimization"* and *"we need to improve the scanning
tool a bit."***

## 0. Where we are, in five lines

- The **architecture** is the owner's: cluster generously, then split (pr/137 §1.1).
- The **kernel** works: segment-level ray 2-means, 0.825 median purity recovering a
  known partition (pr/137 §13a).
- The **trigger** is the hard part, and the borrow that made it work is ATLAS's
  *local-maxima-with-a-valley* (pr/137 §10, §13).
- Current best estimate: **~48 % efficiency at ~80 % purity**, and a measured
  **ceiling of roughly half the over-clustering** (pr/137 §15).
- Everything rests on **22 objects labelled by one scanner (the agent)**. The
  owner scan is the calibration that turns those into measurements.

## 1. Phase A — the scan (this is the next session)

### A0. Prerequisites — DONE 2026-08-31

- [x] `TRIM` added to the verdict vocabulary (pr/137 §15.6)
- [x] `docs/pr/pr137-curated-set.tsv` regenerated with `valley_best` (§15.2's fix)
- [x] `bee/pr137r2/pr137r2.index.txt` regenerated: 50 objects, 41 events
- [x] `bee/pr137r2/pr137r2-{off,on}.zip` rebuilt for the corrected event list
- [ ] **Upload — owner authorises** (`./upload-to-bee.sh`, CLAUDE.md §5.6)

### A1. Tooling — the owner asked for this, and Scan A says what to fix

Ranked by what actually slowed the agent scan down, most valuable first:

1. **The verdict vocabulary** (done, A0). Without `TRIM`, ~45 % of the S2 stratum
   gets forced into `KEEP` or `SPLIT` and both are wrong.
2. **A θ-φ ray-map view in `em_display`.** This is the view the call is made on:
   project every member point onto the unit sphere about the reference vertex and
   plot (θcosφ, θsinφ). Two objects are two blobs; one object is one blob. The
   existing 3-D view makes this judgement slow because the separation is angular,
   not spatial. `pr137_curate.py`'s panel 1 is the reference implementation.
3. **The width-vs-depth overlay.** `w_single(r) = 3.575 + 0.0283·r` (pr/137 §12)
   drawn as a dashed line under the object's own transverse RMS. Empirically every
   agent `SPLIT` sat 2–10× above it and every `KEEP` at or below — the fastest
   single discriminator to *read*, even though it did not win the AUC ranking.
4. **Per-object navigation.** The scan is per-shower, not per-event: 50 objects
   across 41 events, and 9 events carry two. The viewer must jump to a named
   `node_id` and highlight only that object.
5. **Boundary recording.** For `SPLIT` the label needs segment → part; for `TRIM`
   the junk segment list. Today `em_display` records per-shower marks
   (`marks_by_shower`), which is the right granularity (pr/137 §10.1) but the wrong
   semantics — extend the writer, do not repurpose the EM marks.
6. **Blind mode.** The proxy class must not be displayed
   ([[feedback_blind_the_scan_sheet]] — a leaked header makes the agreement number
   circular). `pr137_curate.py --sheets` is blind by default; the viewer needs the
   same discipline.

**Fork, do not edit** (M10): `em_display/` is a production scan tool with existing
records behind it. A `split_display` fork or an additive mode, never an in-place
rewrite of the EM scan path.

### A1.1 `split_display` — BUILT 2026-08-31

`split_display/` (fork, not an edit — em_display keeps its records and its port).

```bash
./split_display/serve_split_display.sh 5022 \
    --scan-tag splitscan-0901-owner --owner-only
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -L 5022:localhost:5022 wcgpu1
#   http://localhost:5022/split_viewer
```

| file | what it is |
|---|---|
| `split_display/split_model.py` | payload + the pre-filled grouping proposal |
| `split_display/split_tree_js.py` | the drag-and-drop tree, browser side |
| `split_display/split_viewer.py` | the Bokeh app |
| `split_display/serve_split_display.sh` | launcher, port 5022 |
| `split_display/selftest_split_display.py` | 16 checks, Python side |

**What it does**, against the owner's five requirements:

1. **Drag between groups.** Four drop columns — Group 0 / 1 / 2 / JUNK — and the
   3-D cloud recolours the moment something lands, with no server round trip for
   the repaint.
2. **Finer than the EM cluster.** Every segment is its own draggable row, so a
   boundary that runs through a bundle can be drawn segment by segment.
3. **Click to highlight.** Clicking a card or a row boosts it in the 3-D view and
   dims everything else; tapping a point in the 3-D view highlights and scrolls to
   the matching card. Both directions, one selection channel.
4. **Two levels.** BUNDLE is the directory — a spatially connected set of segments
   (single-linkage at 4 cm, the production idiom) — and SEGMENT is the file.
   Dragging a bundle moves its whole segment list.
5. **The em_display 3-D view**, imported rather than duplicated: `em3d`'s
   orthographic trackball, drag to rotate, shift-drag to pan.

**The verdict is read off the columns, never typed**: 1 non-empty group = KEEP,
2 = SPLIT2, 3 = SPLIT3, anything in JUNK = TRIM. So a saved label cannot disagree
with the grouping that produced it.

**The proposal that pre-fills the groups** is the round-2 result: seeded angular
maxima decide how many parts (`valley_best`, doc pr/137 §15.2) and each *bundle*
is assigned to the nearer winning seed direction by its own charge-weighted ray.
A bundle is never cut by the proposal — a machine cut through a connected bundle
is harder to correct by hand than one that is too coarse. **JUNK is never
pre-filled**, and that is a measurement, not caution: doc pr/137 §15.3 found half
of all *healthy* showers are already fragmented at 2–4 cm, and a pre-flag on
disconnection alone marked 53 of 256587's 128 segments as junk — 256587 being a
textbook single shower.

**M13 guard.** The tool writes a `.split_display_tag` marker into its label dir on
first save, and **refuses** to write into any directory that holds labels without
one. A mis-typed `--scan-tag` cannot overwrite `emscan-0827`.

**Verified** (`/home/xqian/tmp/pr138-shot2.png`): headless-chromium render of
evt396222 node9059 — 42 bundles, 165 draggables, 4 drop zones, 0 page errors; a
synthesised HTML5 drag of a 22-segment bundle from Group 0 to Group 2 moved
Group 0 from 21 bundles/36 % q to 20/15 %, Group 2 to 1 bundle/21 %, recoloured
those points green in the 3-D view, and flipped the derived verdict SPLIT2 →
SPLIT3.

**Four browser traps this cost, recorded so the next viewer does not pay them
again.** All four fail *silently* — no console error, just a tree that is not
draggable:

1. **Bokeh 3 renders every view inside an open shadow root**, so
   `document.getElementById` finds nothing. Walk the roots
   (em_display's `selftest_em3d_browser.py:136-150` documents the same trap).
2. **`js_on_change` on a `visible=False` widget never reaches the client.** Two
   builds armed the tree from a hidden `TextInput` and bound zero handlers.
3. **A property set while the document is being BUILT is serialised as initial
   state, not emitted as a change** — so a callback registered on it never fires
   on first paint. `curdoc().js_on_event(DocumentReady, ...)` is the one channel
   that is neither, and it is what arms the tree now.
4. **`pts` means a list of sources to `em3d.JS_REDRAW` and a single source here.**
   Concatenating the two read `pts[0].data` off a ColumnDataSource. Ours is
   `cloud` now.

Per-node listeners were replaced by **one delegated listener set on the document**
that finds its card with `composedPath()[0]`, because drag events are
`composed: true` and cross the shadow boundary while `event.target` is retargeted
to the host. `draggable="true"` is emitted by Python into the HTML, so arming a
card needs no DOM write at all.

### A1.2 Bee links in the viewer — ADDED 2026-08-31

Owner: *"we also need the bee link in them, so that we can understand the event
more before doing the divide."* The split tool shows **one object**; Bee shows
everything around it, which is what a boundary call often needs.

`split_display/bee_links.py` resolves them from records that already exist. Every
uploaded round leaves a `<name>.url` (the set URL) beside a `<name>.index.txt`
(bee_idx → event), and **Bee addresses an event by its index in the set**:

```
https://www.phy.bnl.gov/twister/bee/set/<uuid>/event/<bee_idx>/
```

So the viewer scans `bee/*/` once at startup, joins each `.url` to its index, and
prints every set that holds the current event as a deep link. **37 of the owner's
41 events already resolve** to at least one uploaded set — no new upload was
needed to make the feature useful.

Two honest limits:

- **The 4 remaining events have no uploaded set.** `bee/pr137r2/` covers all 41
  and would also give a matched OFF/ON pair for every one, but it is built and
  **held** — a set with no `.url` contributes nothing here by construction.
- **The links are not fetch-verified.** This node has no outbound HTTPS (curl
  exits 35), so the URL *scheme* is taken from the 329 `/event/list/` and ~50
  `/event/<n>/` occurrences in the committed docs, and the indices from the very
  index files the uploads were made with. Verified as far as it can be here, and
  no further.

### A1.3 The two panels, event context, and growable groups — ADDED 2026-08-31

**θ-φ ray map** (A1 item 2) and **width-vs-depth against `w_single(r)`**
(item 3) are now panels under the 3-D view. The ray map shares the *same*
ColumnDataSource as the 3-D cloud, so the drop recolour and the click highlight
reach it for free — there is no second colour channel to keep in step. The width
panel draws one curve per non-empty group plus the dashed null; doc pr/137 §15.4
found every hand-labelled SPLIT sat 2–10× above that line and every KEEP at or
below it.

**Event context.** The display holds ONE object, and the owner asked the right
question of it: *"why is the black star not on the displayed segments?"* The
answer for evt396222 node9059 is physics, not a bug — the object's nearest point
is **14.5 cm** from the ν vertex and its `start_connection_type = 2` ("gap"),
i.e. a photon that converted 14.5 cm downstream, against a ~18 cm mean conversion
length in LAr. Nothing else in that event is closer either: the nearest point of
*any* of its 180 segments is the same 14.5 cm, which is what an NC-like topology
with no visible activity at the vertex looks like. **The rest of the event is now
drawn faint grey**, so the gap reads as a gap rather than as a missing object.
(This shower is 123 of the event's 180 segments; the second-biggest, node 9084 at
2.14e7, is a separate scan object of its own.)

**The black star is the ν main vertex** — `main_vertex` from the calib dump, and
the reference point every angle in the tool is measured from. The grouping
proposal is *angular from the star*, so "two objects" means two directions as
seen from it. One gap to record: doc pr/137 §1.2a wanted the reference to be a
*parameter* (the π⁰ decay vertex when the shower has one, else the ν vertex), and
the viewer always uses the ν vertex. It matters for the NC chain, where K24
re-seats a decay point away from the ν vertex.

### A1.4 evt396222's vertex — checked, and it is a real outlier

Owner: *"why the black star is not at one of the red segments? I thought for this
event, the nu vertex is at the one shower? Or it was somehow changed by NCpi0
chain?"* Checked all three readings:

- **Not the π⁰ chain.** `dump['main_vertex']` and the `is_main` entry in
  `vertices[]` agree to **0.00 cm**, so nothing re-seated it after it was written.
- **The vertex IS attached to that shower** — the owner's expectation is right.
  Main vertex `id 9038`, cluster 9, **degree 1**, and the one segment attached to
  it is **seg 9059, the shower itself**. It is the shower's own start vertex.
- **But it is fitted 14.5 cm upstream of any charge.** Its fit point carries
  `dQ = 0.0`, its `fit_distance = 15.04`, and the nearest point of *any* of the
  event's 180 segments is that same 14.5 cm. The shower's
  `start_connection_type = 2` ("gap").

**And it is the only one of the 41.** Census over the owner's events, main-vertex
distance to the nearest charge in the event:

| | |
|---|---|
| median | **0.00 cm** |
| p90 | 0.00 cm |
| > 5 cm | **1 / 41** |
| max | **14.50 cm (evt396222)** |

Every other main vertex sits *on* charge, with `fit_distance` 0.28–1.05 against
396222's 15.04. So this is a genuine reconstruction anomaly, not a display
artefact and not the NC chain — a degree-1 main vertex extrapolated into empty
space. **Reported, not tuned** (CLAUDE.md §5.7).

**Why it matters for the scan, and not only for this event:** the reference point
is what every angle in the tool is measured from, so on 396222 the grouping
proposal is built from an extrapolated point. The viewer now prints a red warning
whenever the vertex-to-charge gap exceeds 5 cm, and the "rotate about" control
lets the scanner orbit the object centroid instead.

**Groups grow on demand.** Owner: *"for busy events, there may be many groups…
3 can be the default though."* `+ group` / `- group`; three columns plus JUNK by
default, up to the palette. `- group` only ever removes an **empty trailing**
column, so a click cannot silently reassign segments already placed. The verdict
gains `SPLIT4+`, and the saved label records `n_parts` and `n_groups` so the exact
count survives the bucket.

### A2. The owner scan itself

**50 objects, `owner_scan=1` in `docs/pr/pr137-curated-set.tsv`**, spread
25 S1 / 15 S2 / 10 S3. Only **11 of the 50 fire** the trigger — deliberately.
**This is a calibration set, not a purity sample:** its job is (a) the unbiased
false-split rate on the 25 random controls and (b) agent-vs-owner agreement.

Per object: `KEEP` / `SPLIT2` / `SPLIT3` / `TRIM` / `UNSURE`, plus the boundary for
a SPLIT and the junk list for a TRIM. Labels into a **fresh** tag
`em_labels/splitscan-0901-owner/` (M13 — never into `emscan-*` or
`splitscan-0901-agent`).

### A3. The agent scan

The remaining ~120 blind sheets in `work/pr137_sheets/`, same vocabulary, into
`em_labels/splitscan-0901-agent/` (26 objects already labelled: 15 from pr/137
§14.2, 11 from Scan A §15.4).

### A4. The three numbers Phase A must produce

1. **Agent-vs-owner agreement** on the 50 overlap. This is the noise floor: if it
   is *A*, every agent-derived rate carries a floor of (1 − *A*) and **no trigger
   may be claimed to beat it**. Report this BEFORE any trigger claim.
2. **The false-split rate** on the 25 random controls — the efficiency-independent
   safety number.
3. **The real merge count**: how many of the 44 proxy-MERGED are genuine two-object
   cases rather than `TRIM`. Scan A's 11-object sample says ~55 %; this fixes the
   efficiency denominator, which is currently an estimate.

**Gate on Phase A:** if agreement is below ~0.8, the agent labels cannot carry the
statistics — say so and ask for more owner scan rather than propagating them.

## 2. Phase B — implementation

Only after A4. Ordered so nothing is fitted before it is calibrated.

### B1. Stage 1 — `WCT_SHOWER_SPLIT_DEBUG`, byte-neutral probe

**Insertion point: the LAST pass in the chain**, after every merging pass — not
`:8213` as pr/137 §6 originally said. pr/137 §15.7 corrects this; the passes in
between are `shower_dedup_start_seg` (`:8234`), `shower_pass4_prune_detached`
(`:8591`), `shower_pass4_prune_gap2` (`:8745`), `samevtx_absorb` (`:9270`) and
`satellite_absorb` (`:9386`) — **all SBND PRODUCTION ON**. Splitting last is not a
compromise; it is the owner's architecture stated correctly: *merge them together,
then separate cleanly*. It also means the end-of-chain dump — which every number
in pr/137 §12–§15 is computed from — **is** the population the splitter sees.

Emits per candidate: seed list, `valley_best`, `d2_best`, `frac_best`,
`angle_best`, the part membership, and the `w_pull` against the §12 null. `getenv`
idiom exactly as `WCT_SHOWER_XCLUS_DEBUG` (toolkit `deca3467`). Proven with the
standard **478/478** hash gate.

### B2. Stage 2 — the accept test, fitted to Phase A labels

Pre-registered feature set: `valley_best` plus **one** of `d2_best` /
`q_ratio` / `gap_scaled`. A 50/50 event-hash holdout, opened **once**. With ~170
labels the holdout half holds only ~15–25 positives, so a 2-feature cut is the
honest ceiling — **not** a fitted classifier, and no ML dependency is permitted
anyway (toolkit CLAUDE.md).

### B3. Stage 3 — the kernel, with k from the seeds

pr/137 §14.2: on the two largest fired objects the fixed-k=2 kernel returns a
degenerate partition (balance 0.003) while θ-φ shows three lobes. **k comes from
the seed count**, k = 3 reachable only on a residual test (§1.2b).

Knobs, all DEFAULT OFF, key-suppressed in jsonnet so the compiled config is
byte-identical when off:
`shower_split_rays`, `shower_split_min_mev`, `shower_split_min_valley`,
`shower_split_min_d2`, `shower_split_min_frac`, `shower_split_max_parts`,
`shower_split_ref`.

**Write recipe** (fork `pass4_prune_detached`, `:8591-8726`): `detach_member_set`
→ `make_shared<Shower>(graph)` → `set_start_vertex` / `set_start_segment` /
`add_segment` → `showers.insert` → `update_shower_maps` → **own the kinematics
refresh** (there is no free recompute at the end of the chain).
`detach_member_set` refuses a set containing the start segment, so the daughter
keeping it is structurally the "kept" one and a 3-way split is two peel calls.

**Energy does not conserve across a split** (`NeutrinoEnergyReco.cxx:48-145`, no
cross-shower 2D dedup): E(A)+E(B) ≥ E(parent) in the overlap. Any π⁰-mass or
`q_extra` claim must name its regime.

### B4. The one-γ veto

pr/137 §14.2's named false-positive class: 389538 is ONE photon whose e⁺e⁻ pair is
resolved — two arms meeting at a **shared origin** at 3–4 MIP. So one 2-MIP stub at
a shared origin is a **veto**, not a trigger. Add the shared-origin dE/dx and
common-point test to the stage-1 tape so it is measured before it is trusted.

## 3. Phase C — optimisation and composition

### C1. Compose with pr/136

The splitter only earns its place if it lets the `onV1c90` escape ship. Arm:
`onV1c90 + splitter`, judged on pr/136 §11.2's instruments:

- `q_extra` back to ≈ **7.0 %** (the OFF value) — the whole point;
- `q_miss` stays near `onV1c90`'s **11.3 %** — the completeness gain survives;
- π⁰ census exact **≥ 33**;
- the owner finds no new merged γs in a rescan of the pr/137 §2 population.

### C2. Byte-identity and the standard bar

Knob-off gate PASS on the standard 239-event manifest (478 archives), freshness
proof before the A/B (M1), `./build/clus/wcdoctest-clus` green, compiled-config
proof for every new jsonnet key.

### C3. Deferred, with reasons

- **The TRIM front.** ~45 % of proxy "merges" are shower + detached junk, and it
  **survives** `prune_detached` and `prune_gap2` (both ON). Real, separate, and not
  the splitter's job — the splitter has no trim operation.
- **The named residual** (pr/137 §15.5): three cases where the minor part raises no
  density peak — small in charge (181050), tiny in angular extent (122660: 5°×11°
  at 90–115 cm), or connected by charge (142421). No instrument tested reaches
  them. This is the ceiling, not a to-do.
- **Spatial/Arbor topology as a trigger: measured dead** (pr/137 §15.3, 1.2–1.9×
  enrichment). May still serve as a kernel.
- **181050** (doc pr/136 §11.9) — the last open item on `onV1c90d25`, independent
  of the splitter.

## 4. Sequencing, and why this order

The owner's proposed order — **scan session (with tool work) → implementation →
optimisation** — is the right one, for a reason worth stating: *every number in
pr/137 §12–§15 rests on one scanner.* Implementing first would mean fitting a
threshold to labels whose noise floor is unmeasured, which is exactly the failure
pr/130 and pr/136 §11.3 already paid for twice.

The one thing that could have inverted the order — *"is the offline population the
population the splitter sees?"* — is now answered without an arm: pr/137 §15.7
places the splitter last in the chain, and the dump is written after every merging
pass, so **the objects in the Bee package already are the objects the splitter
would see.** Nothing in Phase B needs to run before Phase A.

**Phase A is not blocked by anything except the upload authorisation.**

## Repro

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
scripts/pr137_null_model.py         # the w_single(r) null
scripts/pr137_seed_split.py         # kernel + multiplicity trigger
scripts/pr137_trigger_bakeoff.py    # all features, two positive classes
scripts/pr137_curate.py --sheets    # the curated set + BLIND contact sheets
```
