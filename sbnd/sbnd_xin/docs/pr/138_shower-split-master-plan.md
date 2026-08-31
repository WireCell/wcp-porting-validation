# doc pr/138 — the shower splitter: MASTER PLAN (scan → implement → optimise)

**Status: PHASE A CLOSED 2026-08-31 — the owner hand-scanned all 172 curated
objects. Still no code, no arm, no knob. §1b folds the scan in; §2 is the revised
Phase B, and it is the next session's work.** The measurements this file rests on
are doc pr/137 §10–§15 plus §1b below.

## 0. Where we are, in five lines

- The **architecture** is the owner's: cluster generously, then split (pr/137 §1.1).
- The **trigger is solved**: `valley_best` (ATLAS's local-maxima-**with-a-valley**,
  pr/137 §10) reaches **0.79 efficiency at 0.77 purity** against 172 owner labels,
  against pr/137 §4's 27–36 % ceiling. The threshold is not retuned (§A5.4).
- The **kernel is solved for two-way splits and broken for three** — SPLIT2 median
  boundary agreement **1.000**, SPLIT3 mean 0.620, SPLIT4+ 0.377 (§A5.6). That is
  Phase B's real work.
- **Over-clustering is 4.7 % of EM objects** (S1 random control), and the shipped
  rule would wrongly cut **1.2 %** of them (§A5.2).
- pr/137's arm-difference proxy is **retired**: it was wrong in both directions
  (§A5.2), which is what produced the old 27–36 % null.

## 1. Phase A — the scan  *(CLOSED 2026-08-31; results in §1b)*

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

### A1.4 evt396222's vertex — CORRECTED 2026-08-31: it IS the π⁰ chain

> **This section replaces an earlier, wrong answer** (committed in `421d5228` and
> `8b790e60`). Those said "not the π⁰ chain" and "by elimination the 3-plane
> trajectory fit `fit_point()` at `TrackFitting.cxx:4476`". Both are false. The
> corrected trace is below; the retraction is spelled out under *Why it hid*.

Owner: *"why the black star is not at one of the red segments? I thought for this
event, the nu vertex is at the one shower? Or it was somehow changed by NCpi0
chain?"* — and, on the second pass, *"How can the fit vertex be so much off?"*

**Symptom.** evt396222's main vertex (`id 9038`, cluster 9, **degree 1**, the one
attached segment being `seg 9059`, the shower itself) sits **14.50 cm** from the
nearest point of any of the event's 180 segments, with `dQ = 0.0` and
`fit_distance = 15.04`. It is the only one of the 41 owner events like that:
median main-vertex-to-charge distance 0.00 cm, p90 0.00 cm, >5 cm in **1/41**,
max 14.50 cm = evt396222; every other main vertex has `fit_distance` 0.28–1.05.

**Root cause — the π⁰ decay-point re-seat, and it is deliberate.**
`NeutrinoShowerClustering.cxx:7886`, in `id_pi0_without_vertex` (path 2):

```cpp
// Update main vertex position (hack) - set to reconstructed pi0 decay point
main_vertex->fit().point = vtx_point;
main_vertex->fit().dQ    = 0;
```

**Which of the two sites.** `id_pi0_backproject_vertex` (the pr/133 K21
back-projection proposer) carries the identical two lines at `:6241`, labelled
*"The P2 acceptance, verbatim mechanics (vertex hack …)"*, so the attribution
needs an argument rather than an assumption. Two independent ones, both pointing
at `:7886`:

- **396222 is not on the bp chain's firing list.** `pi0_bp_vertex_miss_cm = 8`
  went SBND production ON in `30d4263e` with the measured note *"fires on **2 of
  239** events (76346 60.2 cm, 116962 83.5 cm upstream)"* — 396222 is neither.
- **396222 was already a path-2 acceptance in pr/132 round 1**, i.e. before the
  `:6241` site existed at all (`id_pi0_backproject_vertex` arrives in `8432b7af`,
  round-9/K19 groundwork in `42d728e9`, both 2026-08-30). The round-2 comment
  quoted below is that round-1 record.

*Not claimed:* the knob's state inside `work-pr136-onV1c90-*` specifically was
not read back from that arm's config. Discriminating the two sites inside this
arm would need `WCT_PI0_PAIR_DEBUG=1` on a rerun, which the two arguments above
make unnecessary rather than impossible.

The dump agrees, line for line:

| | |
|---|---|
| `showers[9059].pio_id` | **0** |
| `showers[130313].pio_id` | **0** — the same π⁰ |
| `pio_mass` | **135.75 MeV** |
| shower 9059 | 2879.4 MeV, 123 seg, 723.7 cm, start 14.50 cm from the vertex, conn 2 |
| shower 130313 | **35.0 MeV**, 4 seg, 6.9 cm, start 104.3 cm from the vertex, conn 2 |
| implied opening angle | **24.7°** (θ_min for a 2914 MeV π⁰ is 5.3°, so allowed) |

And the source comment 26 lines above the write names this very event:
*"the round-1 ADVERSE path-2 acceptances are low-mass (122660 m=85.2, 171143
m=75.5) while the good ones sit near 135 (**396222 m=133.5**, 169626 m=138.9); a
decay-point shift cap does NOT separate them (shifts 23.0/5.8 vs **14.5**/59.6
cm)."* **14.5 cm is our number.** The vertex was moved to the reconstructed γγ
back-projection point — which, for a π⁰, is the *right* estimate of the ν vertex,
since a π⁰ decays essentially at rest-frame zero range. The shower's own first
charge is 14.5 cm downstream because that is where the photon converted
(`start_connection_type = 2`, "gap"). **So the owner's instinct was right twice
over: the vertex does belong to that shower, and the NC π⁰ chain did move it.**

**Why it hid — and why the earlier answer was wrong.** Two independent traps.

1. *The test had no power.* `PrDisplayDump.cxx:445-460` writes **both**
   `main_vertex` and the `is_main` entry of `vertices[]` from the same
   `vertex->fit().point`. They agree by construction, whoever moved the point;
   "they agree to 0.00 cm ⇒ nothing re-seated it" was never a valid inference.
2. *The `Fit` struct is internally inconsistent after the hack* — and this is a
   **real defect**, separate from the re-seat. The hack writes `point` and `dQ`
   and nothing else, so the vertex's `pu / pv / pw / pt` and `reduced_chi2` still
   describe the **old, on-charge** location. Measured on this vertex:

   | | `pt` | `pu` | `pv` | `pw` |
   |---|---|---|---|---|
   | recorded in `vertices[].fit` | 1442.05 | 621.09 | 1571.11 | 1043.70 |
   | what seg 9059's **point 0** projects to | 1442.1 | 621.1 | 1571.1 | 1043.7 |
   | what the recorded **3-D point** projects to | 1388.5 | 598.6 | 1547.4 | 997.4 |

   The stored projections are the shower's first point to every printed digit;
   the stored 3-D point is 14.5 cm away. `reduced_chi2 = 6.54` is likewise seg
   9059 point 0's chi2 — a **pre-move** number. Reading it as evidence of a bad
   fit is exactly the mistake the earlier section made.

   *Bounding the consequence, not speculating about it:* the only readers of a
   **vertex's** fit projections found in `clus/` are
   `NeutrinoVertexFinder.cxx:1310` (`row.hv_reduced_chi2`, which
   `PrDisplayDump.cxx:811` only dumps) and the dump's own `fit_json`. No
   production *decision* was found reading them. `fit().index` and `fit().paf`
   are read by `TrackFitting.cxx:6543/7107`, but those passes run before the π⁰
   acceptance. So the demonstrated damage is: it lies to the dump, it lies to
   this scan tool, and it misled this investigation for a day.

**Fix — none applied, and the reason.** Both halves would change production
output: re-projecting the vertex after the re-seat changes `vertices[].fit`, and
re-weighing the pairing changes reconstruction. Either needs a default-OFF knob
and a gate (CLAUDE.md §1). What ships this round is the **display**: the viewer
now names the cause instead of guessing at it (`split_model.pio_partner`,
`split_viewer._vertex_note_html`).

**The pairing itself — reported, not tuned (CLAUDE.md §5.7).** 2879.4 MeV against
a 35.0 MeV, 4-segment, 6.9 cm crumb 104 cm away, at 24.7°, giving 135.75 MeV. An
82:1 energy asymmetry is the classic combinatoric-fake signature, and shower 9059
is 123 segments / 723.7 cm — plausibly the very over-clustering this round exists
to split, which would mean the π⁰ mass rests on an inflated energy. Not acted on
here; flagged for the owner.

**Left alone**: `PrDisplayDump.cxx:445` still comments `fit_distance` as *"How
far improve_vertex/MyFCN moved this vertex off its seed point."* On this vertex
that attribution is wrong twice — MyFCN cannot have run (degree 1 fails
`FitVertex`'s `ntracks` gate at `MyFCN.cxx:500`), and what actually moved it was
the π⁰ hack. It is an unrelated production file and this round changes no C++, so
it is recorded here rather than edited.

**Why it matters for the scan.** The reference point is what every angle in the
tool is measured from, so on 396222 the proposal is built from the π⁰ decay point
rather than from the shower's own start. The viewer prints the explanation
whenever the vertex-to-charge gap exceeds 5 cm, and "rotate about" lets the
scanner orbit the object centroid instead.

### A1.5 The colour modes — ADDED 2026-08-31

Owner: *"There is a problem, the color of the group is gone now. No red vs.
blue."* **Measured first.** It is not a rendering fault and not a regression from
the wheel-zoom commit: a headless probe reads red 894 / blue 437 out of the live
cloud on object 1, and a webgl-vs-canvas A/B shows per-point CSS colours repaint
identically under both backends. The census is the answer —

```
groups proposed over the 50 owner objects:  {1: 39,  2: 11}
```

**39 of 50 objects get a single-group proposal**, so group colouring has nothing
to say and the cloud is uniformly group-0 blue. The trigger is honest; the
display had only one thing to show.

So the display now has three: a `colour by` selector, `group` (default — the
verdict is read off groups, so red-vs-blue stays the primary signal) /
**`bundle`** (the unit the owner actually drags; the tree lists 42 of them on
evt396222 and they were visually identical) / **`charge`** (a 9-stop viridis log
ramp — the split criterion *is* a charge dip between two maxima, so the scanner
should be able to see the quantity the trigger reads). The mode rides as a
reserved `_mode` key inside the **existing** `cmap` payload rather than as a new
widget callback: four earlier builds lost handlers to Bokeh 3 binding traps, so a
proven channel gets reused, not duplicated. Both figure titles print the active
mode, because the bundle palette necessarily reuses the group hues.

One honest defect fixed alongside: `propose()` could report *"2 groups
proposed: valley_best=0.091"* and still hand back one group (evt389538 node19021)
— the per-bundle ray vote sends every bundle to one seed when no connected bundle
carries the minority lobe. The reason string now says what actually happened.

### A1.6 Eight of the 50 are track-typed — CURATION FINDING

Owner, on evt99838: *"I only see the major track in it, not any of the EM shower.
I want to confirm that you essentially want me to look at the part relevant for
our algorithm, not judging the entire event, right?"* — **Yes**, and the object
is exactly what they saw. Census of `particle_id` over the 50 owner objects:

| pid | n |
|---|---|
| 11 (e) | 42 |
| 13 (μ) | **5** |
| 2212 (p) | **2** |
| 211 (π) | **1** |

| event | node | pid | length | kine_best |
|---|---|---|---|---|
| 99838 | 14004 | 13 | 473 cm | 1047 MeV |
| 389538 | 19021 | 2212 | 182 cm | 837 MeV |
| 292524 | 9018 | 13 | 202 cm | 436 MeV |
| 176502 | 109141 | 2212 | 183 cm | 533 MeV |
| 286681 | 72040 | 13 | 109 cm | 420 MeV |
| 122660 | 54071 | 13 | 54 cm | 280 MeV |
| 415278 | 23047 | 13 | 97 cm | 245 MeV |
| 278420 | 18002 | 211 | 47 cm | 137 MeV |

Cause: the pr/137 §14 population is *"`onV1c90` objects with q > 1e6 and ≥3
segments"*, and the dump's `showers[]` container holds every reco particle, not
only EM ones. **The curated set is NOT regenerated** — the owner is already
scanning it and it is a record (M13). Instead the viewer now prints the reco
identity on every object and badges a non-electron **TRACK-TYPED, not an EM
shower**, so the scanner can dispose of one in a second (as the owner already
did: evt99838 → KEEP, *"This is one good track, no need to split"*). The eight
are excluded from the efficiency denominator in §A4 and named there.

### A1.7 Navigation and saved state — FIXED 2026-08-31

Owner: *"when I switch the event using the prev/next button, the object should
change as well as the note should be updated. So far I see that those are not
changing"* and *"please write on the screen, if I have saved my scan, so I know
which event I scanned, which event I have not."* Both were real:

- `load()` never wrote back to the `object` Select, so the dropdown kept showing
  the previous object.
- `load()` never re-seated `note (optional)` or the confidence buttons. **This
  was a data-integrity bug, not only cosmetic**: a note typed on object *k* stayed
  in the box and would have been written onto object *k+1*'s label, silently, in
  a scientific record.
- there was no saved/unsaved indicator anywhere.

Now: `load()` is a re-entrancy-guarded wrapper (writing `jump.value` from inside
the loader echoes through `jump.on_change`, so the flag makes the echo a no-op
instead of a second full load); every per-object widget is re-seated on
navigation; the `object` dropdown carries **✓ *verdict*** on everything already
saved and **· --** on everything not; and a banner reads either
**✓ SAVED *timestamp*, verdict, confidence** or **NOT YET SAVED**, with
*scanned N of 50 in this tag*. Saving refreshes all three without a reload.

**Groups grow on demand.** Owner: *"for busy events, there may be many groups…
3 can be the default though."* `+ group` / `- group`; three columns plus JUNK by
default, up to the palette. `- group` only ever removes an **empty trailing**
column, so a click cannot silently reassign segments already placed. The verdict
gains `SPLIT4+`, and the saved label records `n_parts` and `n_groups` so the exact
count survives the bucket.

### A1.8 Layout and point style at many groups — FIXED 2026-08-31

Owner: *"The group can be a lot, can you move the 3D display to the right side of
the window, so that they do not overlap"* and *"the transparency of the point can
be less, and thicker in the 3D view, so that things can be viewed clearer."*

**The overlap was measured, not guessed.** At 6 groups the tree Div's *content*
grew to **838 px** while Bokeh had reserved 780 and had already placed the 3-D
canvas at **x = 790** — so the columns painted over the 3-D view. Two causes,
both fixed:

- **no `box-sizing: border-box`.** Each `.col` carries 1 px border + 4 px padding
  a side; under the default content-box those 10 px are added *on top of* the
  flex basis, so seven columns could not fit the box they were given.
- **`flex: 1 1 0` divided the width** instead of scrolling: at 6 groups the
  columns were 115 px, too narrow to read a segment row.

Now the columns are a fixed readable **186 px** and the **row scrolls**, with the
tree clipped at `TREE_W = 800` — a single constant that `split_tree_js` writes
into the CSS and `split_viewer` reserves in the layout, so the two can no longer
drift. Measured after: tree right edge **805**, first canvas **810**, at 3 groups
and at 10 (`scrollWidth` 2096 clipped to 800). The scrollbar is forced visible
(it is the only cue that more columns exist), and `dragover` auto-scrolls the row
within 60 px of either edge — verified by dropping a bundle into **Group 7**, a
column that starts off-screen.

**More groups, since that was the premise.** `MAX_NGROUPS` is `len(GROUP_COLORS)`,
so the palette *was* the cap: extended **6 → 10**, first six unchanged so a label
already saved keeps the colour it was scanned under.

**Point style**, now single-sourced in `split_tree_js` (it had been written twice,
in Python and in JS, with nothing keeping them in step): alpha **0.85 → 1.00**,
size **4.0 → 6.0**, highlight 7.0 → 10.0, dimmed 0.12/2.5 → 0.18/3.0.

**A trap caught by a new test, not by the browser.** The auto-scroll calls
`_cols()`, which lives in `JS_FIND` — and `JS_SETUP` did not include `JS_FIND`,
so the first `dragover` would have thrown `ReferenceError` with no symptom beyond
"drag stopped working". `selftest_split_display.py` now lints every blob actually
passed to `CustomJS` for a helper used but not declared, **and** for the
mirror-image trap (the same `const` declared twice by double-concatenation). Both
arms mutation-tested: the pre-fix blob reports `missing: ['_cols']`, a
double-concatenated blob reports `dups: ['_tree','_cols','_walkAll']`. 39 checks.

### A1.9 The scan opens to the full 172 — 2026-08-31

Owner, after finishing the calibration 50 in one sitting: *"Actually the scan is
not too bad, I guess that I can scan the rest of events as well."* The server now
runs **without `--owner-only`**, so all **172** curated objects load.

- **Order: the owner-50 first, then the remaining 122 by descending charge.** The
  50 are what §A4's agent-vs-owner agreement is computed on, so they must stay
  identifiable and must not be shuffled in among the rest. Labels are keyed by
  `(event, node)` and never by index, so the re-order cannot disturb anything
  already saved — verified: all 50 still read ✓ after the change.
- Every calibration object carries a **`*`** in the dropdown.
- **`next unsaved ▸`** added. At 50 objects `next >` was enough; at 172 hunting
  the ticks by eye is exactly the busywork a scan tool should absorb. It wraps
  once and reports when the set is complete.

**This changes what Phase A can claim, for the better.** §A4 was written for
~50 owner labels carrying a noise floor and ~190 agent labels carrying the
statistics, with the gate *"if agreement is below ~0.8 the agent labels cannot
carry the statistics."* If the owner labels all 172, the **owner's own labels
carry the statistics** and the agent scan (26 labels in
`em_labels/splitscan-0901-agent/` so far) becomes a cross-check rather than the
denominator. The agreement number is still computed — it is now a check on the
*agent*, not a licence for it — and the (1 − *A*) floor stops gating the trigger
claim. The three numbers of §A4 are then measured on **164** objects (172 less
the 8 track-typed of §A1.6), not on 42.

### A2. The owner scan itself

> **Superseded in scope by §A1.9**: the owner finished these 50 and opened the
> scan to all 172. The 50 remain the calibration overlap described here.

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

**Denominator correction (§A1.6).** Eight of the 50 owner objects are
track-typed — evt99838/14004, 389538/19021, 292524/9018, 176502/109141,
286681/72040, 122660/54071, 415278/23047, 278420/18002 — so all three numbers are
reported over **42**, with the 8 listed separately as `KEEP (not an EM object)`.
They are still scanned (a KEEP is one click) and still count towards the
agent-vs-owner agreement, since agreeing that a muon is one object is a real, if
easy, agreement; they are excluded only from the false-split and merge-count
denominators.

## 1b. PHASE A RESULT — the owner's 172-object scan, folded in (2026-08-31)

**Status: Phase A is CLOSED.** All 172 curated objects carry an owner hand label,
170 of them at high confidence. Reproduce every number below with

```bash
python3 scripts/pr138_scan_analysis.py --kernel     # -> docs/pr/pr138-scan-analysis.tsv
```

### A5.1 What the labels say

| | KEEP | SPLIT2 | SPLIT3 | SPLIT4+ | TRIM |
|---|---|---|---|---|---|
| all 172 | 121 | 34 | 8 | 2 | 7 |
| the 164 EM objects (§A1.6) | 115 | 33 | 8 | 2 | 6 |
| **S1** random control, n=85 | 77 | 3 | 1 | 0 | 4 |
| **S2** known merges, n=29 | 15 | 9 | 3 | 2 | 0 |
| **S3** feature-enriched, n=39 | 19 | 15 | 3 | 0 | 2 |

**43 of 164 EM objects (26%) are splittable**, but that is a *stratified* set. The
population number is S1's, below.

### A5.2 The three §A4 numbers

**1 — agent-vs-owner agreement**, the noise floor, on the 26-object overlap:
**exact verdict 0.692, SPLIT-vs-not 0.846**. Four disagreements, three of them
agent-splits-owner-keeps. The gate said "if agreement is below ~0.8 the agent
labels cannot carry the statistics" — 0.846 clears it, and it no longer matters,
because §A1.9 means the **owner's own labels carry the statistics** and the agent
scan is now a check on the agent.

**2 — the false-split budget**, from **S1 only** (the feature-independent stratum
drawn before any feature was consulted; the pooled `S1 + S1+S2` row in the script
is stratum-contaminated by construction and must never be quoted as a prevalence):

| | |
|---|---|
| population prevalence of over-clustering | **4 / 85 = 4.7 %** |
| the shipped rule fires on | 3 / 85 = 3.5 % of all objects |
| of those, really splittable | 2 / 3 |
| **wrongly cut, per EM object** | **1 / 85 = 1.2 %** (≲ 4 % at 95 % CL — n=85 with 4 positives is thin, and this is the number to re-measure first on any larger sample) |

**3 — the arm-difference proxy against truth.** This is the retrospective that
explains pr/137 §4:

| proxy | KEEP | SPLIT | TRIM | n |
|---|---|---|---|---|
| MERGED | 19 | **21** | 0 | 40 |
| SINGLE | 96 | **22** | 6 | 124 |

**Only 52.5 % of proxy-MERGED is really splittable, and proxy-SINGLE hides 22 real
splits (17.7 %).** So the proxy was wrong in *both* directions, and pr/137 §4's
27–36 % purity ceiling was a property of the labels, not of the features. That
null is now retired.

### A5.3 The bake-off, against real labels

Fit = even event id (n=87, 24 splits); holdout = odd (n=77, 19 splits).
Ranked on **fit only**:

| feature | AUC | purity @ 50 % eff | cut |
|---|---|---|---|
| **`valley_best`** | **0.930** (printed 0.070; low = split) | **0.857** | `valley_best ≤ 0.52` |
| `angle_best` | 0.841 | 0.571 | `≥ 24.1°` |
| `w_pull` | 0.825 | 0.600 | `≥ 3.42` |
| `sep_scaled` | 0.771 | 0.480 | `≥ 6.53` |
| `m_pi0` | 0.753 | 0.545 | `≥ 48.4 MeV` |
| `valley` (pr/137 §4's version) | 0.748 | 0.857 | `≤ 0.81` |
| `n_seed` | 0.712 | 0.419 | `≥ 4` |
| … | | | |
| `vgap_min`, `r_ratio`, `n_2mip`, `q_ratio` | 0.50–0.55 | — | **dead** |

**The ATLAS borrow is the whole result.** `valley_best` — minimise the charge
valley over *all* seed pairs carrying ≥ 3 % of the charge — is the single feature,
and the owner's three stated factors come in behind it in his own order:
direction (`angle_best`), size (`w_pull`), distance (`sep_scaled`).

**A caveat that must travel with this table.** The fit-half 2-feature scan returns
**six rules at purity 1.000** on 24 positives; that is overfitting, and the second
cut of the best one (`w_pull ≤ 4.85`) is an *upper* bound on a feature the single-
feature ranking says discriminates *upward* — i.e. noise. **None of those pairs is
carried into Phase B.**

### A5.4 The shipped proposal, scored as a trigger — the bar

The rule already in `split_model.propose()` — a second angular maximum **and**
`valley_best ≤ 0.95` — against the owner labels:

| | efficiency | purity |
|---|---|---|
| all 164 | **0.791** | **0.773** |
| fit half | 0.917 | 0.917 |
| holdout half | 0.632 | 0.600 |

The fit/holdout gap is **not** overfitting — this rule was never fitted; it is
sampling variance across 24 vs 19 positives, and it is the honest width of the
uncertainty on 0.79/0.77.

A fit-only threshold scan says **0.95 is already the knee** (0.90 → 0.875/0.913,
0.95 → 0.917/0.917, 0.99 → 0.917/0.815). **So the threshold is not retuned.**
`valley_best ≤ 0.95` with `n_seed ≥ 2` is pre-registered as-is for Phase B, and
the holdout is spent.

**Against pr/137 §4's 27–36 %, this is 0.773.** The trigger is no longer the
blocker.

### A5.5 Were the labels anchored by the pre-fill?

They could have been: the tool pre-fills the columns and prints the proposal in
words, which is exactly the trap [`blind the scan sheet`] warns about. Measured
rather than assumed:

- the owner **overrode the pre-fill on 34 of 164** objects, **in both directions**
  — 19 overrides ended SPLIT, 15 ended KEEP/TRIM;
- the proposal fired and the owner said KEEP/TRIM anyway on **9 of 42** fires;
- the proposal was silent and the owner split anyway **10** times;
- on the 26-object overlap the **blind** agent agrees with the proposal **0.826**
  of the time and the pre-filled owner **0.783** — the anchored scanner agrees
  *less*.

There is no anchoring signature. The labels are usable.

### A5.6 THE KERNEL — solved for k=2, unsolved for k≥3

Charge-weighted agreement between the proposal's boundary and the owner's, best
label matching, over the 43 owner-SPLIT objects:

| verdict | n | median | mean | ≥ 0.90 | < 0.60 |
|---|---|---|---|---|---|
| **SPLIT2** | 33 | **1.000** | **0.927** | 25 | 2 |
| SPLIT3 | 8 | 0.671 | 0.620 | 2 | 4 |
| SPLIT4+ | 2 | 0.467 | 0.377 | 0 | 2 |

**Every one of the six worst boundaries is a ≥3-way split.** The two-way kernel is
done — median agreement is *exactly* 1.000. The k≥3 case fails because `propose()`
hard-wires k=2, which is precisely the "the seed count **is** the multiplicity
decision" shape §10 borrowed from ATLAS/CMS/GARLIC and §15 did not implement.
**10 of 43 splits (23 %) are out of reach for that one reason.**

### A5.7 The two error lists, and what the owner wrote on them

**The 9 real splits the shipped rule misses.** Seven of nine have
`valley_best = 1.0` — *no charge dip at all* — with `n_seed` 3–4 and small
`angle_best` (6.6–35.5°). That is the owner's factor 4 exactly: *"the two gammas
may be connected directly, so a split would be nice but more difficult."* There is
no valley to find because the lobes overlap. **This is the trigger's honest blind
spot, and it is a different problem from the one `valley_best` solves.**

**The 10 false fires.** Two carry the owner's own diagnosis, and they name a
systematic:

> evt318769 node31026 — *"incorrect neutrino vertex, actually both groups should be one"*
> evt278420 node61027 — *"this is a single EM shower pointing not to the nu vertex, but the end point of a track."*
> (and evt281781 node89069, a SPLIT3: *"incorrect vertex"*)

**Every feature in the trigger is measured from the reference vertex**, so a
mis-placed vertex manufactures a fake angular bimodality. §A1.4's evt396222 — where
the π⁰ chain re-seats the main vertex 14.5 cm off the charge — is the same
mechanism from a different cause. At least 2 of 10 false fires have a named,
fixable origin.

**A third class the splitter cannot fix at all: UNDER-clustering.** Five comments
say the piece belongs *somewhere else*:

> evt122660 node9110 — *"The few clusters at the end of this major cluster should be part of the main cluster. Right now, the main cluster did not include them."*
> evt122660 node53070 — *"This should be part of the EM shower"*
> evt122660 node54071 — *"keep, but this should be part of the earlier EM shower cluster"*
> evt292524 node9018 (TRIM) — *"These Trimmed part should belong to another EM shower."*
> evt98844 node6013 (SPLIT2) — *"the small part would belong to another cluster."*

Read together these are a **requirement, not a complaint**: a cut that leaves the
detached piece as an orphan has not finished the job. **A split must re-home its
daughter**, and the owner says where, twice, for TRIM as well as SPLIT.

**Two objects the owner excused**, both worth honouring:
evt389538 node19021 — *"This entire group is a separate neutrino event, so it is OK
to keep"* (pile-up, not over-clustering); and evt396222 node9059 — the only **low**
confidence label in 172 — *"a very busy event with many different EM showers…
multiple tracks misclustered as EM shower. I am not sure if this event is really
useful for our purpose."* That is the same object §A1.4 found carrying a 2879 MeV
π⁰ leg, reached independently.

## 2. Phase B — implementation (REVISED by the Phase A result, 2026-08-31)

**The staging is inverted from the original plan, because the data inverted it.**
pr/137 assumed the kernel was fine and the trigger was the blocker. §A5 says the
opposite is now true: the trigger reaches **0.79 efficiency at 0.77 purity** and
needs no retuning, while the kernel is **exact for two-way splits and broken for
three-way**. Phase B is ordered accordingly, and each stage names the number it
must not make worse.

**This is the next session's work. Nothing below is started.**

### B0. What Phase A already settled — do not re-open

| | settled | evidence |
|---|---|---|
| the trigger feature | `valley_best` | AUC 0.930, purity@50 % 0.857 (§A5.3) |
| the threshold | `valley_best ≤ 0.95`, `n_seed ≥ 2` | fit-only scan: 0.95 is the knee (§A5.4); **holdout spent, do not re-tune** |
| the 2-way boundary | done | median agreement **1.000**, mean 0.927 (§A5.6) |
| the label set | 172 owner labels, unanchored | §A5.5 |
| the old proxy | retired | 52.5 % / 17.7 % wrong both ways (§A5.2) |

### B1. Stage 1 — `WCT_SHOWER_SPLIT_DEBUG`, byte-neutral probe  *(unchanged)*

**Insertion point: the LAST pass in the chain**, after every merging pass — not
`:8213` as pr/137 §6 originally said (§15.7 corrects it). The passes in between —
`shower_dedup_start_seg` (`:8234`), `shower_pass4_prune_detached` (`:8591`),
`shower_pass4_prune_gap2` (`:8745`), `samevtx_absorb` (`:9270`),
`satellite_absorb` (`:9386`) — are all SBND PRODUCTION ON. Splitting last is the
owner's architecture stated correctly: *merge them together, then separate
cleanly*. It also means the end-of-chain dump every §A5 number is computed from
**is** the population the splitter will see.

Emits per candidate: the seed list, `valley_best`, `angle_best`, `w_pull`,
`sep_scaled`, the part membership, and the reference vertex used. `getenv` idiom
exactly as `WCT_SHOWER_XCLUS_DEBUG` (toolkit `deca3467`). **Gate: 478/478
byte-identical**, and the probe's own fire list must reproduce §A5.4's 42 fires on
the 164 scanned objects — a probe that does not is wired to the wrong population.

### B2. Stage 2 — the accept test, PRE-REGISTERED, not fitted

`valley_best ≤ 0.95 AND n_seed ≥ 2`, transcribed from §A5.4. **No second feature**
— the fit-half 2-feature scan returned six rules at purity 1.000 on 24 positives,
which is noise, and its best pair cuts `w_pull` in the direction opposite to its
own single-feature ranking (§A5.3). Knobs `shower_split_min_valley` (default 0.95)
and `shower_split_min_seeds` (default 2) exist so the operating point is *movable*,
not so it is searched again.

Success criterion, stated before the work: on the 164 scanned objects the C++
trigger must reproduce the offline fire list **object for object**. A disagreement
is a porting bug, not a new measurement.

### B3. Stage 3 — the kernel, with **k from the seed count**  ← THE OPEN PROBLEM

This is where the round's remaining value is. §A5.6: SPLIT2 median agreement
1.000, SPLIT3 mean 0.620, SPLIT4+ mean 0.377, and **all six worst boundaries are
≥3-way**. `propose()` hard-wires k=2; 10 of 43 splits (23 %) are unreachable for
that one reason.

- take **k = the number of surviving angular maxima**, not 2 — the ATLAS/CMS/
  GARLIC shape §10 borrowed and §15 did not implement;
- assign each **segment** to its nearest surviving seed (the action space is
  segments — `Shower::detach_member_set`, `PRShower.cxx:640-700`);
- **re-check on the 10 objects the owner called SPLIT3/SPLIT4+**: the target is
  their mean agreement, 0.620 → ≥ 0.85, with SPLIT2's 0.927 **not regressing**.
  Both halves of that are a gate, not an aspiration.
- `shower_split_max_parts` (default 2) is what turns it on, so k≥3 ships behind
  its own knob and the 2-way result can be validated first.

**Write recipe** (fork `pass4_prune_detached`, `:8591-8726`): `detach_member_set`
→ `make_shared<Shower>(graph)` → `set_start_vertex` / `set_start_segment` /
`add_segment` → `showers.insert` → `update_shower_maps` → **own the kinematics
refresh** (there is no free recompute at the end of the chain).
`detach_member_set` refuses a set containing the start segment, so the daughter
keeping it is structurally the "kept" one and a 3-way split is two peel calls.

**Energy does not conserve across a split** (`NeutrinoEnergyReco.cxx:48-145`, no
cross-shower 2D dedup): E(A)+E(B) ≥ E(parent) in the overlap. Any π⁰-mass or
`q_extra` claim must name its regime.

### B4. Stage 4 — RE-HOME the daughter  ← NEW, and it is the owner's requirement

Five of the owner's 24 comments say the detached piece belongs somewhere
specific — *"should be part of the earlier EM shower cluster"*, *"the small part
would belong to another cluster"*, *"These Trimmed part should belong to another
EM shower"* (§A5.7). **A cut that leaves an orphan has not finished the job**, and
this is also the difference between SPLIT and TRIM as the owner uses them: TRIM
means *cut it off and give it away*, SPLIT means *cut it and both halves stand*.

- after a peel, offer the daughter to the nearest EM shower under the existing
  absorb predicates (`samevtx_absorb` `:9270`, `satellite_absorb` `:9386` are the
  house idioms — **fork, do not extract**, M10);
- the metric is the owner's own: `q_extra` must fall without `q_miss` rising,
  measured on the pr/136 arms;
- knob `shower_split_rehome`, DEFAULT OFF, so B3 can be validated without it.

### B5. Stage 5 — the vertex-quality veto  ← NEW, from the false-fire list

Every trigger feature is measured from the reference vertex, so a mis-placed
vertex manufactures a fake bimodality. Named by the owner on 2 of the 10 false
fires — *"incorrect neutrino vertex, actually both groups should be one"* — and by
§A1.4's evt396222, where the π⁰ chain re-seats the main vertex 14.5 cm off every
piece of charge.

**Measure before trusting**: add the vertex-to-nearest-charge distance, the π⁰
`pio_id`, and `fit_distance` to the stage-1 tape, then ask whether any of them
separates the 10 false fires from the 34 true fires. `vgap_min` as computed
offline does **not** (AUC 0.499, §A5.3), so this is a hypothesis with a named test
and a real chance of being measured dead — say so if it is.

### B6. The one-γ veto  *(kept, still unmeasured)*

pr/137 §14.2's named false-positive class: 389538 is ONE photon whose e⁺e⁻ pair is
resolved — two arms meeting at a **shared origin** at 3–4 MIP. One 2-MIP stub at a
shared origin is a **veto**, not a trigger. Add the shared-origin dE/dx and
common-point test to the stage-1 tape so it is measured before it is trusted.
(The owner's *"this entire group is a separate neutrino event"* on 389538
node19021 is a *different* object in the same event — pile-up, not pair
resolution. Do not conflate them.)

### B7. Explicitly NOT in Phase B

- **The small-angle / no-valley class.** Seven of the nine misses have
  `valley_best = 1.0` — no charge dip exists, because the two γs overlap (§A5.7,
  the owner's factor 4). `valley_best` cannot find them and no threshold move
  will; this needs a different observable (a transverse-profile fit, or the
  two-stub dE/dx of §B6 used as a *trigger* rather than a veto). **Scoped out, with
  the reason, rather than papered over by loosening the cut** — loosening to 0.99
  costs 10 points of purity for zero efficiency (§A5.4).
- **Anything fitted to the holdout.** It has been opened once (§A5.4).

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

**Epilogue, 2026-08-31.** It went the way the order predicted, and one thing it
could not have predicted: the noise floor came back at 0.846 SPLIT-vs-not, but the
owner then scanned all 172 himself, so the floor stopped mattering (§A1.9). What
did matter is the part that could only be learned by measuring — **the proxy every
pr/137 number rested on is wrong in both directions** (§A5.2). Implementing first
would have fitted a threshold to it. The order paid for itself.

## Repro

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
scripts/pr137_null_model.py         # the w_single(r) null
scripts/pr137_seed_split.py         # kernel + multiplicity trigger
scripts/pr137_trigger_bakeoff.py    # all features, two positive classes
scripts/pr137_curate.py --sheets    # the curated set + BLIND contact sheets

# PHASE A RESULT -- every number in section 1b
python3 scripts/pr138_scan_analysis.py --kernel   # -> docs/pr/pr138-scan-analysis.tsv

# the scan tool, and its selftest (28 checks)
split_display/serve_split_display.sh 5022 --scan-tag splitscan-0901-owner --owner-only
split_display/selftest_split_display.py

# section A1.4 -- the pi0 re-seat that moved evt396222's vertex 14.5 cm
python3 -c "import sys;sys.path[:0]=['split_display','scripts'];import split_model as SM;\
print(SM.pio_partner(396222, 9059))"
grep -n 'reconstructed pi0 decay point' ../../../toolkit/clus/src/NeutrinoShowerClustering.cxx

# section A1.5 / A1.6 -- the two censuses quoted above
python3 scripts/pr138_scanset_census.py
```
