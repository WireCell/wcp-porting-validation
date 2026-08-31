# doc pr/138 — the shower splitter: MASTER PLAN (scan → implement → optimise)

> **SUPERSEDED IN PART, 2026-08-31.** §3 recommended *not* flipping the splitter
> alone and identified `onV1c90 + splitter` as the pairing worth scanning. The
> owner then flipped **that pairing to SBND production** — *"lets use 'onV1c90 +
> splitter' as the new baseline for SBND production"* — so
> `shower_pass4_prefilter_v1_escape`, `..._max_v2 = 90` and `shower_split` are
> now **ON** in `wct-pr-perevt.jsonnet` (flip-equivalence gate PASS 478/478,
> `work-pr138r3-flipchk-*`). **Read §3's recommendation as executed, not
> pending.** The next round is `139_pi0-after-the-splitter.md`.


**Status: PHASE A CLOSED, PHASE B SHIPPED, and the FLIP QUESTION ANSWERED — all
2026-08-31.** The owner hand-scanned all 172 curated objects (§1b); the splitter
is in the toolkit behind `shower_split`, **DEFAULT OFF**, gate-clean (§2); and
§3 measures what it does to physics at four operating points. **`onV1c90` +
splitter gives the best π⁰ census the campaign has measured (35 of 66) with
`q_extra` back at the production floor** — the splitter alone at production does
not. §4 answers the owner's "when the situation is clean" question; §5 ranks
what to do next. The knob is still `false`; a flip needs his word.

## 0. Where we are, in five lines

- The **architecture** is the owner's: cluster generously, then split (pr/137 §1.1).
- The **trigger is solved**: `valley_best` (ATLAS's local-maxima-**with-a-valley**,
  pr/137 §10) reaches **0.79 efficiency at 0.77 purity** against 172 owner labels,
  against pr/137 §4's 27–36 % ceiling. The threshold is not retuned (§A5.4).
- The **kernel is exact for two-way splits and short for three.** The shipped C++,
  scored on its own arm against the hand labels: SPLIT2 median **1.000**, mean
  **0.974**, 21 of 27 boundaries *exactly* right; k≥3 reaches **0.756** (from
  0.635 at the default cap) against a pre-registered **≥ 0.85 — MISSED**, so k≥3
  ships behind a non-default knob value and is not production-eligible (§B3).
- **Over-clustering is 4.7 % of EM objects** (S1 random control), and the shipped
  rule would wrongly cut **1.2 %** of them (§A5.2).
- pr/137's arm-difference proxy is **retired**: it was wrong in both directions
  (§A5.2), which is what produced the old 27–36 % null.
- The splitter is **in the toolkit, DEFAULT OFF**, byte-identical when off, and
  runs last among the shower passes and *before* the π⁰ finders (§B1).
- **It pays off only in composition.** `onV1c90 + splitter`: `q_extra` 12.0 % →
  **6.7 %** (production is 6.9 %), π⁰ exact **32 → 35** of 66, 0 ADVERSE movers.
  The splitter alone at production moves no π⁰ and costs 3 impossible pairs (§3).
- **On a vertex that sits on its charge the splitter makes zero mistakes**
  (13 fires, 13 right); **all 8 false fires are bad-vertex objects**, and 2 of
  them broke a good π⁰ (§4). That is the front worth the next round (§5.1).

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

## 2. Phase B — implementation  *(EXECUTED 2026-08-31; this section is now the record)*

**The staging was inverted from the original plan, because the data inverted it.**
pr/137 assumed the kernel was fine and the trigger was the blocker. §A5 said the
opposite. Phase B was ordered accordingly, and this is what it produced.

| stage | state | the number it is judged on |
|---|---|---|
| **B1** `WCT_SHOWER_SPLIT_DEBUG` probe | **DONE** | accept decision agrees with the offline kernel on **172 / 172** scanned objects |
| **B2** the accept test | **DONE, DEFAULT OFF** | C++ end-to-end **eff 0.767 / pur 0.805** on the 164 EM objects |
| **B3** the kernel | **SHIPPED at `max_parts=2`; k≥3 measured SHORT** | SPLIT2 median **1.000**, mean **0.974**, 21 of 27 exact · k≥3 0.635 → **0.756**, **target 0.85 MISSED** |
| **B4** re-home the daughter | **NOT STARTED**, by decision | — |
| **B5** vertex-quality veto | **instrumented, not tested** | the tape now carries the four fields §B5 asks for |
| **B6** the one-γ veto | **NOT STARTED** | — |
| **B7** small-angle / no-valley | **scoped out, unchanged** | — |

Toolkit: `clus/src/NeutrinoShowerClustering.cxx` (+576), `NeutrinoPatternBase.h`,
`TaggerCheckNeutrino.{h,cxx}`, `clus/test/doctest_clus_knob_defaults.cxx`,
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`. wcp-porting-img:
`run_pr_chain_batch.sh` env block, three scripts, three TSVs.

### B0. What Phase A settled — and one thing it got wrong

| | settled | evidence |
|---|---|---|
| the trigger feature | `valley_best` | AUC 0.930, purity@50 % 0.857 (§A5.3) |
| the threshold | `valley_best ≤ 0.95`, `n_seed ≥ 2` | fit-only scan: 0.95 is the knee (§A5.4); **holdout spent, not re-tuned** |
| the 2-way boundary | done | §A5.6, and **better than it said** — see the amendment |
| the label set | 172 owner labels, unanchored | §A5.5 |
| the old proxy | retired | 52.5 % / 17.7 % wrong both ways (§A5.2) |

**Amendment to §A5.6 — that table conflated two different failures.** It scored
the proposal's boundary on *every* owner-SPLIT object, including objects where
the accept test never fired: an object the trigger declined has no boundary, and
scoring it as a boundary failure charges the kernel for the trigger's miss.
Conditioned on the trigger having fired (`pr138_kernel_k.py`, n = 34 of 43):

| | §A5.6 as printed | conditioned on firing |
|---|---|---|
| SPLIT2 | n=33, median 1.000, mean 0.927 | **n=27, median 1.000, mean 0.982, 25 of 27 ≥ 0.90** |
| k≥3 | n=10, mean 0.571 | n=7, mean 0.573 |

So the two-way kernel is *better* than Phase A claimed, and the k≥3 deficit is
unchanged — 3 of the 10 k≥3 objects were never triggered at all, and those belong
to §B7's no-valley class, not to B3.

**And one thing Phase A could not see: 16 % of the owner's SPLIT boundaries cut
through a spatially connected bundle.** 7 of 43 objects, carrying **24.7 % of the
scanned split charge**, and 5 of the 7 are k≥3. A bundle-level assignment can
never reproduce those, which is the real ceiling on B3 and the reason the k≥3
kernel needs a finer unit than the k=2 one.

### B1. `WCT_SHOWER_SPLIT_DEBUG` — the byte-neutral probe  *(DONE)*

**Insertion point: after `em_start_backext`, before
`recompute_shower_kine_charge_final`** — i.e. after every merging pass
(`shower_dedup_start_seg` `:8234`, `shower_pass4_prune_detached` `:8591`,
`shower_pass4_prune_gap2` `:8745`, `samevtx_absorb` `:9270`, `satellite_absorb`
`:9386`, `em_collinear_merge`, `em_start_backext`, all SBND production ON) and
**before the π⁰ finders**. Splitting last among the merges is the owner's
architecture stated correctly — *merge them together, then separate cleanly*;
splitting before the finders is the physics: a γ pair over-clustered into one
shower can only be PAIRED into a π⁰ after it has been cut apart.

The probe ports `pr137_lib.angular_maxima` + `split_model.propose` to C++ in
full — the in-situ bandwidth `σ(r) = (3.575 + 0.0283·r)/r` clipped to [2°, 60°],
the charge-weighted angular density, greedy seeding at `sep_scale 1.6` with
`max_seeds 4`, the 25-sample great-circle valley, the charge fractions, the
4 cm single-linkage bundles and the assignment. Every constant is commented with
its Phase A provenance and the fact that moving it invalidates §A5.4's fire list.

**Fidelity, measured not asserted** (`pr138_probe_compare.py`, arm
`work-pr138r1-dbg-{mcp1k,mcp2k,ncpi0,nuecc48}`, the 125 events holding the
scanned objects):

| | |
|---|---|
| scanned objects present on the tape | **172 / 172** |
| member count identical to the dump | 170 / 172 |
| point count identical to the dump | 170 / 172 |
| `n_seed` identical to the offline kernel | 170 / 172 |
| **accept decision identical** | **172 / 172** |
| \|Δ`valley_best`\| | median **0**, max 0.101 |

(All four rows compare against the offline kernel run on the **calib dump's own
main vertex**, which is the decisive frame. `pr138_probe_compare.py` also prints
a second row computed at the vertex the *tape* reports; that one scores lower
only because the tape prints the vertex to 0.01 cm, and it is a control, not the
comparison.)

**The two exceptions are one event and one named mechanism, and the comparison
was built to catch exactly this.** The π⁰ back-projection re-seats the main
vertex *after* the splitter runs (`:6241`, and path 2 at `:7886` — §A1.4), so on
a π⁰ event the calib dump's `main_vertex` is not the point the probe measured
from. The tape therefore prints the vertex it used. It moved on **5 of 172
objects in 2 events**: evt76346 by **60.16 cm** (the pr/133 K21 back-projection —
which also changed the membership there, 15 segments → 4 and 5 → 3) and
evt396222 by **14.50 cm** (§A1.4's path-2 re-seat). Neither changes a verdict:
both evt76346 objects are owner-KEEP and neither fires either way.

Everything else — 167 objects — is bit-for-bit the same population, the same
features and the same decision. **B1's stated criterion is met.**

### B2. The accept test  *(DONE, DEFAULT OFF)*

Shipped as `shower_split` (false), `shower_split_max_valley` (0.95),
`shower_split_min_frac` (0.03), plus the population floors
`shower_split_min_charge` (1e6 raw `Fit::dQ`) and `shower_split_min_nseg` (3),
which are Phase A's `q_floor`/`nseg_floor` transcribed.

**One deliberate change from §B2's pre-registration.** §B2 wrote the rule as the
*best pair* by minimum valley. The C++ instead accepts seeds greedily in density
order — the brightest is always a seed, a later one joins iff it carries ≥ 3 % of
the charge and a valley ≤ 0.95 separates it from one already accepted — so that
**the trigger and the multiplicity decision are literally the same computation**,
which is the ATLAS/CMS/GARLIC shape §10 borrowed.

**Neither threshold moved**; the holdout stays spent. What changed is the
quantifier, and the two rules were compared **offline on the 164 labelled
objects** (`pr138_kernel_k.py`, `onV1c90` dumps — no C++ arm was ever run with
the pair rule, so this comparison is offline and is not a measurement of two
binaries):

| rule | fires | efficiency | purity |
|---|---|---|---|
| §A5.4's best-pair | 44 | 0.791 | 0.773 |
| shipped greedy | 43 | 0.791 | **0.791** |

They differ on **one object of 164** — evt281567 node99193, owner KEEP, which the
greedy rule correctly declines. The number that certifies the shipped rule is not
that comparison but the C++'s own end-to-end score below.

**End-to-end, from the arm's own tape** (`pr138_smoke_split.py`), the C++ fires
on **41 of the 164** scanned EM objects: **efficiency 0.767, purity 0.805**. The
two fires it loses relative to the 43 above are the honesty check — the seeds
separate but every bundle falls to one of them, so there is no partition to make
and firing with one part is not a split. Against pr/137 §4's 27–36 % ceiling this
is 0.805.

Over the whole population the pass sees, **45 of 258 candidates fire (17.4 %)** —
the scanned set is stratified and enriched, so the unbiased prevalence remains
§A5.2's S1 number (4.7 %), not this one.

### B3. The kernel, with k from the seed count  *(cap=2 SHIPPED; k≥3 MISSED its target)*

`shower_split_max_parts` (default **2**) caps how many parts one candidate is cut
into; `shower_split_snap` (0.80) governs only the k≥3 path.

**The C++ was run at both settings and scored from its own tape** — this is the
partition the shipped binary actually drew, not the offline kernel's prediction.
Arms `work-pr138r1-on2-*` (`max_parts=2`, the default) and `work-pr138r1-on4-*`
(`max_parts=4`), same 125 events, same 164 labelled EM objects.

| C++ arm | trigger eff / pur | SPLIT2 (n=27) median / mean / exact | k≥3 (n) mean | all |
|---|---|---|---|---|
| **`max_parts=2`** *(default)* | 0.767 / 0.805 | **1.000** / **0.974** / **21** | 0.635 (6) | 0.913 |
| `max_parts=4` | 0.791 / 0.810 | **1.000** / 0.953 / 16 | **0.756** (7) | 0.912 |

And the same two configurations as the offline kernel predicted them
(`pr138_kernel_k.py`, thirteen acceptance × assignment variants, conditioned on
the trigger firing) — **a separate frame, quoted so the port can be checked**:

| offline variant | SPLIT2 (n=27) | k≥3 (n=7) | all (34) |
|---|---|---|---|
| the Phase A kernel | 0.982 | 0.573 | 0.897 |
| **`max_parts=2`** | **0.974** | 0.574 | 0.892 |
| §B3's pre-registered rule (greedy k, bundle/centroid) | 0.966 | 0.701 | 0.911 |
| **`max_parts=4`** | 0.953 | 0.772 | 0.916 |

**The two frames agree**: 0.974 vs 0.974 on SPLIT2 at cap 2, 0.953 vs 0.953 at
cap 4, and 0.756 vs 0.772 on the k≥3 class (7 objects; the C++ additionally
requires the partition to be non-trivial before it counts as fired). That
agreement — reached from a numpy prototype and a hand C++ port that share no
code — is the strongest evidence the port is faithful.

**Verdict on §B3's gate, stated as it was pre-registered: MISSED, on both arms.**
The target was k≥3 mean ≥ 0.85 with SPLIT2's boundary not regressing.

Read **within one frame** — never across the two, since the C++ and the offline
tables above count slightly different objects:

| frame | k≥3 at cap 2 | k≥3 at cap 4 | SPLIT2 cost |
|---|---|---|---|
| the C++ arms | 0.635 (n=6) | **0.756** (n=7) | mean 0.974 → 0.953, **exact 21 → 16 of 27** |
| offline, same 7 objects both columns | 0.574 | 0.772 | mean 0.974 → 0.953 |

(The C++ cap-4 column carries one *more* k≥3 object than cap 2, because a seventh
one only produces a non-trivial partition once more than two seeds are available;
the offline frame holds the object set fixed, which is why its baseline is lower.)

Either way the movement is real and neither reaches 0.85, so k≥3 stays behind its
own knob at a non-default value and is **not production-eligible**.

Note also what `max_parts=4` buys on the *trigger* (efficiency 0.767 → 0.791 at
purity 0.810): more seeds means more objects whose partition is non-trivial,
which is a separate effect from boundary quality and is not a reason to ship it.

Three things are now known about *why*, and none of them is a tuning problem:

1. **The unit, not the count.** Taking k from the seed count alone (the
   pre-registered rule) buys 0.573 → 0.701. The rest comes from letting a
   straddling bundle be cut at the segment level, which is exactly the 16 % /
   24.7 %-of-charge finding in §B0. The k=2 path deliberately does *not* do this:
   at two parts the bundle/centroid rule is essentially exact and a finer unit can
   only add speckle, so the unit is chosen by k and that is not a free parameter.
2. **`max_seeds = 4` is a hard ceiling.** Two of the owner's objects are 5-part
   (evt396222 node9059, evt415278 node23037) and both sit at the bottom of the
   table. Raising it would move `n_seed`, hence the trigger, hence §A5.4's fire
   list — so it was not raised.
3. **Three of the ten k≥3 objects never fire at all** and belong to §B7.

**The write path** forks `pass4_prune_detached` (`:8591-8726`):
`detach_member_set` → `make_shared<Shower>(graph)` → `set_start_vertex` /
`set_start_segment` / `add_segment` → `showers.insert` → `update_shower_maps` →
its own `calculate_shower_kinematics` (the only later recompute,
`recompute_shower_kine_charge_final`, is knob-gated and no-ops in production).

**One deliberate divergence from the fork source, and it is load-bearing.**
`pass4_prune_detached` seeds a re-homed component at the member nearest the
*kept body*. A split seeds the daughter at the member nearest the **reference
vertex** instead, because the π⁰ finders run after this pass and read
`get_start_point()` and `get_init_dir()` — a daughter seeded at its downstream
end would point back at its sibling and poison exactly the π⁰ mass this round
exists for. The tape carries the check: over all **116 peels across both ON
arms**, cos(start ray, body ray) has **median 0.997, minimum 0.219, and zero
backwards**.

**Knob-ON smoke**: `on2` — 45 fires → **45 peels, 0 refusals**, conn 3 : 36 /
4 : 9, daughter-to-start-vertex distance median 38.2 cm. `on4` — 47 fires →
**71 peels, 0 refusals**, conn 3 : 56 / 4 : 15, median 39.6 cm. The forward check
holds on all 116 peels across both arms.

**Energy still does not conserve across a split**
(`NeutrinoEnergyReco.cxx:48-145`, no cross-shower 2D dedup): E(A)+E(B) ≥
E(parent) in the overlap. Any π⁰-mass or `q_extra` claim must name its regime.
Nothing in this round makes such a claim.

### B4. Re-home the daughter — NOT STARTED, and that is a decision

§A5.7 is unchanged and still right: five of the owner's comments say the detached
piece belongs somewhere specific, so a cut that leaves an orphan has not finished
the job. It is left out of this commit on purpose — it is a *second* behaviour
change with its own knob, its own `q_extra`/`q_miss` measurement on the pr/136
arms, and its own owner review, and folding it in would make the gate above
answer two questions at once.

### B5. The vertex-quality veto — instrumented, not yet tested

The tape now carries, per candidate, the four things §B5 asked for: the
**reference vertex actually used**, the **vertex-to-nearest-member-charge
distance** (`vgap_cm`), and the main vertex's own **`reduced_chi2`** and **`dQ`**
(`vchi2`, `vdQ`), plus the `vfit` validity flag. `pio_id` is *not* on the tape and
cannot be — the π⁰ finders run after this pass — so that join is made offline
against the dump.

Two things are already visible and both are honest warnings rather than results:
offline `vgap_min` does **not** separate the false fires (AUC 0.499, §A5.3), and
the vertex the splitter sees differs from the one the scan saw on 5 of 172
objects, by up to 60 cm (§B1). The second is a *new* fact this round produced and
it is the more interesting one: the owner's two "incorrect neutrino vertex" false
fires may be measuring a vertex that a later pass then moves.

### B6. The one-γ veto  *(kept, still unmeasured)*

Unchanged from the plan. The shared-origin dE/dx test is not on the tape yet.

### B7. Explicitly NOT in Phase B  *(unchanged, and now with one more reason)*

- **The small-angle / no-valley class.** Seven of the nine misses have
  `valley_best = 1.0` — no charge dip exists, because the two γs overlap (§A5.7,
  the owner's factor 4). Loosening to 0.99 costs 10 points of purity for zero
  efficiency (§A5.4). §B0's amendment adds the second reason: three of these are
  *also* the k≥3 objects that drag §A5.6's k≥3 mean down, so the class is a
  bigger share of the residual than it looked.
- **Anything fitted to the holdout.** It has been opened once (§A5.4).

### B-gate. The bar, met

- **Byte-identity — PASS, rc=0 on all four samples: 132 + 212 + 38 + 96 =
  478 / 478 archives byte-identical, `missing/unpaired events: 0` on every one**
  (that last line is what rules out a vacuous PASS: `pr85_hash_gate.py` compares
  only events common to both arms, so a sample that silently lost an event would
  still say PASS). Labels `work-pr138r1-bare-<s>` vs `work-pr138r1-off-<s>`,
  `<s>` ∈ {mcp1k 66, mcp2k 106, ncpi0 19, nuecc48 48} = the standard 239-event
  manifest, production config. **The OFF arm ran with
  `WCT_SHOWER_SPLIT_DEBUG=1`** — the probe *executing* the whole kernel, seeding,
  valley, bundles, assignment, and writing its tape — which is a strictly
  stronger statement than the shipping configuration, where the env is unset and
  the pass returns on its first line.
- **Freshness (M1)**: `local/lib/libWireCellClus.so` 09:36:45 >
  `NeutrinoShowerClustering.cxx` 09:28:04. Both arms ran against *pinned* library
  snapshots (`/home/xqian/tmp/pin-pr138{bare,off}` via `LD_LIBRARY_PATH`), so a
  peer's `wcbuild` mid-campaign cannot void the comparison.
- **Unit tests**: `./build/clus/wcdoctest-clus` **235/235**, assertions
  2601 → **2617** — the eight new knobs are pinned in
  `doctest_clus_knob_defaults.cxx`, which is the file that makes "default OFF"
  a test rather than a claim.
- **Compiled-config proof**: the arms' own `.wct-cfg-evt*.json` — no
  `shower_split*` key on the OFF arm, `shower_split: true` on the ON arm.
- **Determinism**: candidates are walked in a `(cluster_id, segment id)` sorted
  order; the seed sort breaks density ties by point index; no pointer-keyed
  container is iterated.

## 3. Should `shower_split` be turned ON for SBND?  *(MEASURED 2026-08-31)*

**Answer: not on its own — but `onV1c90 + splitter` is, and it is the best π⁰
census the campaign has measured.** The owner asked *"does it improve the shower
clustering and π⁰ reconstruction?"* Measured on his own instruments at **four**
operating points, 239 events each:

| arm | `q_miss` | `q_extra` | **π⁰ exact** | impossible pairs |
|---|---|---|---|---|
| production **today** (the baseline) | 15.1 % | 6.9 % | **32** / 66 | 19 / 56 |
| production **+ splitter** | 19.4 % | 5.4 % | 32 / 66 | 22 / 56 |
| `onV1c90` — the pr/136 escape, no splitter | 11.6 % | **12.0 %** | 33 / 66 | 16 / 47 |
| **`onV1c90` + SPLITTER** | 16.7 % | **6.7 %** | **35** / 66 | 19 / 49 |

Read the last row against the first — that is the real decision:

- **`q_extra` is back at the production floor**: 12.0 % → **6.7 %**, against
  production's 6.9 %. **The splitter hands back every point of `q_extra` the
  escape paid.** That is precisely what §C1 was written to test, and it passes.
- **π⁰ exact 32 → 35 of 66 (48.5 % → 53.0 %)** — three more hand π⁰s
  reconstructed than production, and two more than the escape alone.
- `q_miss` reads 15.1 % → 16.7 %, but **94 % of that rise is a measurement
  artefact** (§3.3): it is the completeness instrument penalising the splitter
  for cutting objects the *later* hand scan says must be cut. The escape's own
  completeness gain (15.1 % → 11.6 %) survives underneath it.
- **0 vertex movers and 0 ADVERSE** at both operating points (159 compared
  labels, `--tags vtx105`).

**So: turning `shower_split` on ALONE is not worth it** — at production `q_extra`
is already at its floor, the π⁰ census does not move, and three pairs cross into
kinematically impossible. **Turning it on TOGETHER with the pr/136 escape is a
real gain on both of the owner's stated goals**, and it is the pairing that
should go to a Bee scan.

**One thing must be fixed first either way** (§4): every false fire — including
the two that broke good π⁰s — sits in the class the owner himself named, an
object whose ν vertex is tens of cm off its charge.

`shower_split` stays `false` in `wct-pr-perevt.jsonnet`. A flip needs an explicit
owner request on record, and this one wants a scan first.

### 3.1 One number cannot answer this, so there are two arms pairs

| pair | config | what it asks |
|---|---|---|
| `poff` / `pon` | production | the **safety** gate. `q_extra` is already at its floor here (6.9 %), so a splitter can only take charge *out* of showers — this measures whether turning it on breaks anything |
| `c90off` / `c90on` | + the pass-4 `angle_v1` escape (pr/136's `onV1c90`) | the **efficacy** test. The escape buys `q_miss` and *pays* `q_extra`; handing that `q_extra` back is the job §C1 designed the splitter for |

Both pairs: 239 events, the standard manifest, dumps on, differing **only** by
`SBND_SHOWER_SPLIT`, run against the pinned `pin-pr138off` library.

### 3.2 The instruments, at the production point  *(the SAFETY gate)*

| instrument | OFF | ON | Δ |
|---|---|---|---|
| `q_miss` (hand-scan attribution, 90 marked showers) | 15.1 % | 19.4 % | **+4.3 pt** |
| `q_extra` | 6.9 % | 5.4 % | **−1.5 pt** |
| median `q_f1` | 0.918 | 0.918 | 0 |
| **π⁰ census exact** (of 66) | **32** | **32** | 0 — *but not the same 32* |
| pairs sharing a γ | 74 % | 76 % | +2 |
| **π⁰ pairs kinematically impossible** | **19 / 56** | **22 / 56** | **+3, worse** |
| over-clustering class (m > 160) | 8 | 7 | −1, better |
| vertex movers (`pr90_movers.py --tags vtx105`) | — | — | **0 movers, 0 ADVERSE** over 159 compared labels |

### 3.3 The raw `+4.3 pt` of `q_miss` is mostly a measurement artefact, and I can name it

**`pr136_completeness.py` cannot grade a splitter on its own.** Its target is
`(members ∪ marked-in) − marked-out` from the **2026-08-27/28 attribution scan**,
which called several of these objects *one shower*; the **2026-09-01 split scan**
says they are three to five. So a **correct** cut is scored as a miss. Decomposed
per object:

| class | n | Δ`q_miss` | Δ`q_extra` |
|---|---|---|---|
| the split scan says **SPLIT** | 6 | **+4.01 pt** | −0.90 pt |
| the split scan says **KEEP** | 4 | +0.29 pt | −0.41 pt |
| fired, no split label | 4 | 0 | −0.15 pt |
| no fire on this object | 1 | 0 | −0.04 pt |
| **total** | | **+4.30** | **−1.50** |

**93 % of the `q_miss` rise is the instrument penalising the splitter for obeying
the later scan** — 91 % of it on just three objects (evt415278 nodes 23012 and
23037, evt84229 node 69134). And the mirror image: **28 % of the `q_extra` gain
sits on objects the split scan calls KEEP**, where the attribution scan agrees
the charge did not belong (evt278420 node 61027: −1.69e6 `q_extra` for +1.5e5
`q_miss`, on an object labelled KEEP).

**The two hand scans disagree in both directions. Neither alone adjudicates a
split.** Where they do agree — the KEEP class — the trade is `q_miss` **+0.29 pt**
for `q_extra` **−0.41 pt**, which on the owner's "balance them" instruction is
mildly favourable, not the 3:1 loss the raw row suggests.

**A measurement caveat, proven not assumed.** The absolute `q_miss` here is 15.1 %
against doc pr/136 §11.2's published **14.0 %**. That is the *sidecar*, not the
arm: scoring pr/136's **own** `f086probe` dumps with **this** round's (probe-less)
prepdir returns 15.1 % as well. `em117_score.py` prefers a sidecar built by
`prep_em_scan.py --parse-probes`, which needs `WCT_SHOWER_CONTENT_DEBUG` on the
arm; this round's arms did not run it. Reported lossiness of the fallback join is
**0 members on all four scores**, and every arm here is scored the same way, so
the OFF→ON deltas are like-for-like. **The absolutes are not comparable across
arms built with different probe sets** — that is an operational rule worth
keeping, not a one-off.

### 3.4 The π⁰ answer — the census count hides two gains and two losses

"32 → 32" is not "nothing happened". The **sets differ**:

| event | OFF | ON | |
|---|---|---|---|
| **56243** | no-group | **exact** | **BETTER** — the splitter created the pairable γ. This is the mechanism working. |
| **415278** | no-group | partial | BETTER |
| **314838** | **exact** | partial | WORSE |
| **165157** | partial | no-group | WORSE |

and eight hand π⁰ pairs moved, three crossing from possible into
**kinematically impossible**:

| event | m OFF | m ON | R OFF | R ON | |
|---|---|---|---|---|---|
| 165157 | 152.1 | 107.8 | 1.132 | 0.824 | **crossed into impossible** |
| 281165 | 140.6 | 121.7 | 1.107 | 0.928 | **crossed into impossible** |
| 314838 | 121.4 | 76.7 | 1.274 | 0.570 | **crossed into impossible** |
| 54332 | 109.0 | 91.1 | 0.938 | 0.725 | further from 135 |
| 169356 | 131.2 | 128.5 | 0.974 | 0.955 | further from 135 |
| 396222 | 412.9 | 614.4 | 6.775 | 5.751 | further from 135 (the busy event) |
| 56243 | 160.6 | 156.7 | 1.193 | 1.163 | closer to 135 |
| 91917 | 124.2 | 127.3 | 1.108 | 1.152 | closer to 135 |

**Two of the three crossings are false fires** — evt165157 node 9000 and evt54332
node 122091 are both owner-**KEEP** objects the trigger cut anyway, and cutting
them halved a γ (165157: γ₂ 187.9 → 94.4 MeV, mass 152 → 108). That is the whole
π⁰ cost, and it is not intrinsic to splitting: it is the false-fire rate.

### 3.4b The efficacy point — `onV1c90` OFF → ON, where the splitter earns its place

| instrument | OFF | ON | Δ |
|---|---|---|---|
| `q_miss` | 11.6 % | 16.7 % | +5.1 pt (**+4.85 of it the §3.3 artefact**) |
| `q_extra` | **12.0 %** | **6.7 %** | **−5.3 pt** |
| **π⁰ census exact** | 33 | **35** | **+2** |
| pairs sharing a γ | 77 % | 77 % | 0 |
| π⁰ pairs impossible | 16 / 47 | 19 / 49 | +3 |
| over-clustering class | 4 | 3 | −1 |
| vertex movers | — | — | **0 movers, 0 ADVERSE** |

Genuine cost where both hand scans agree: `q_miss` **+0.30 pt** for `q_extra`
**−0.42 pt**. The census set moves on eight events — **four better** (280972 and
56243 no-group → **exact**, 314838 partial → **exact**, 269774 no-group →
partial) against **four worse** (281485 and 396222 partial → none, 165157 partial
→ no-group, 54332 exact → partial). Note **314838 goes the OTHER way here**: with
the escape on it becomes exact *after* the split — the same object §3.5 shows
three instruments disagreeing about.

The three pairs that cross into impossible are **165157**, **281165** and — at
production — **54332**, and 165157 and 54332 are false fires on owner-KEEP
objects. Same two events at both operating points. **The π⁰ cost is the
false-fire rate, not the splitting.**

### 3.5 evt314838 — three owner instruments, three different answers

The sharpest case in the round, and it is an **owner adjudication item**, not
something to resolve by picking a side:

| instrument | verdict on cutting evt314838 node110088 |
|---|---|
| the **split scan** (2026-09-01) | **SPLIT2, high confidence** — cut it |
| the **attribution scan** (2026-08-27/28) | **cut it** — `q_f1` 0.721 → 0.863, purity 0.715 → **1.000**, `q_extra` 3.17e6 → **0** |
| the **hand π⁰ label** | **do not** — the pair needs the 645 MeV γ that includes exactly that charge; after the cut R 1.274 → 0.570 and the census loses it |

pr/136 already flagged 314838 as an owner-scan-vs-overlay contradiction; this is
a third instrument joining the disagreement. It states the round's central
tension in one event: **EM-clustering purity and π⁰ energy pull in opposite
directions on an over-clustered shower**, and no amount of splitter tuning
resolves that — it is a question about which label is right.

## 4. "When the situation is reasonably clean" — the owner's gate, measured

> *"There are cases where it is difficult to get it right. incorrect neutrino
> vertex, very busy events etc. What we want is when the situation is reasonably
> clean, we get decent results against the hand scan results."* — owner, 2026-08-31

This reframes the residual: the hard classes are a **scope boundary to name**, not
a purity number to tune against. So the splitter was stratified by conditions
measured **from the arm, never from the label or the outcome**, with physical
rather than scanned thresholds (`scripts/pr138_clean_strata.py`):

| axis | bound | why |
|---|---|---|
| vertex sits **on** the charge | `vgap ≤ 5 cm` | every trigger feature is a ray *from* the ν vertex; a vertex in empty space manufactures a fake bimodality |
| vertex not re-seated later | \|Δv\| < 1 cm | the π⁰ chain moves `main_vertex` **after** the splitter (§B1: 60.16 cm on evt76346) |
| event not busy | `n_cand ≤ 3` | the owner's "very busy events", counted rather than eyeballed |

### 4.1 The answer is yes, and it is sharp

| stratum | n | real splits | fires | efficiency | **purity** |
|---|---|---|---|---|---|
| everything | 164 | 43 | 41 | 0.767 | 0.805 |
| **vertex on charge** (`vgap ≤ 5`) | **65** | **17** | **13** | **0.765** | **1.000** |
| vertex off charge (`vgap > 5`) | 99 | 26 | 28 | 0.769 | 0.714 |
| event not busy | 131 | 31 | 30 | 0.774 | 0.800 |
| busy | 33 | 12 | 11 | 0.750 | 0.818 |

**All eight false fires are in the bad-vertex class. Zero in the good-vertex
class.** Where the ν vertex sits on the charge the splitter catches 13 of 17 real
splits and makes **no mistakes** — that is the owner's criterion met.
(§4.2b is the caveat that must travel with this: `vgap ≤ 5 cm` selects
*vertex-attached* objects, which is not the same as *"the vertex is right"*.)

**Two honest bounds on that.** Thirteen fires with zero failures bounds the true
failure rate at about **23 % at 95 % confidence** (rule of three), so "perfect" is
thin, not established. And the clean stratum genuinely has fewer splits to find
(11 of 51 vs 32 of 113 for the full three-axis CLEAN definition) — over-clustering
concentrates in messy events, which is physically sensible and is why any gate
costs efficiency.

**The vertex axis carries the whole result.** The busy-event axis moves purity by
+0.018 and the re-seat axis by nothing; adding them to the definition only costs
efficiency. Do not over-build the gate.

### 4.2 A free test the definition could have failed

The owner flagged specific objects in his own scan comments — *"incorrect
neutrino vertex"*, *"a very busy event … I am not sure if this event is really
useful for our purpose"*. Those comments were **never consulted** to build the
definition above:

| object | his words | a-priori stratum |
|---|---|---|
| evt318769 node31026 | "incorrect neutrino vertex, actually both groups should be one" | HARD (`vgap` 38.2 cm) |
| evt281781 node89069 | "incorrect vertex" | HARD (`vgap` 14.6 cm) |
| evt396222 node9059 | "very busy event … not sure if useful" (the only low-confidence label in 172) | HARD (`\|Δv\|` 14.50 cm) |

**3 of 3.** The instrument agrees with the scanner on every object he called out.

### 4.2b But `vgap` does not mean "the vertex is wrong", and I could not fix that

**A photon converts a mean 9/7·X₀ ≈ 18 cm from its origin, so a large `vgap` is
exactly what a real γ looks like.** The population tape says the median `vgap`
over **all 400** production EM candidates is **9.6 cm**, and **44.5 %** exceed
13 cm (charge-weighted 25 %). That number must *not* be read as "44.5 % of ν
vertices are wrong" — it is a mixture of wrong vertices and honest conversion
gaps, and that mixture is why `vgap` scored AUC 0.499 as a *trigger* feature
(§A5.3) even though it stratifies the *fires* so cleanly.

**The obvious separator was tested and is dead** (`scripts/pr138_vertex_gap.py`).
The hypothesis: a real conversion gap is *empty*, a wrong vertex sits inside
other activity, so the charge-free fraction of the vertex→object ray should be
high for γs and low for bad vertices. Measured over the 41 fires:

| feature | AUC (correct cut > false fire) | |
|---|---|---|
| `vgap` | 0.265 | false fires have the *larger* gap (§4.1) |
| **`void_frac`** (ray empty of any charge) | **0.146** | **backwards** — false fires median 0.883, correct cuts 0.400 |
| `occ_other` (ray empty of *other* objects' charge) | 0.536 | no separation |

**The false fires have the CLEANEST gaps of all.** So the failure mode is not
"the vertex is buried in junk" — it is the opposite, and it is worth stating
plainly because it explains why the trigger fires there at all:

> **Seen from the wrong origin, one shower genuinely looks like two
> well-separated ones.** The object has two angularly distinct lobes with a real
> charge dip between them; the dip is an artefact of the viewpoint, not of the
> charge. No feature measured *from that same vertex* can tell the difference.

That is a real limit, not a tuning gap, and it is why §5.1 proposes fixing the
vertex rather than adding a feature.

### 4.3 A veto is a dial, not a free filter — priced

Every false fire has `vgap ≥ 13.4 cm — but so do 20 of the 33 correct cuts, which
run out to 231.7 cm. `vgap` does **not** separate the two classes; it trades:

| `vgap ≤` | fires | right | wrong | efficiency | purity |
|---|---|---|---|---|---|
| 5 cm | 13 | 13 | 0 | 0.302 | 1.000 |
| 10 cm | 14 | 14 | 0 | 0.326 | 1.000 |
| 13 cm | 16 | 16 | 0 | 0.372 | 1.000 |
| 15 cm | 19 | 18 | 1 | 0.419 | 0.947 |
| 30 cm | 27 | 25 | 2 | 0.581 | 0.926 |
| **no veto** | 41 | 33 | 8 | **0.767** | **0.805** |

A veto at 13 cm removes **all eight** false fires — including both that broke a
π⁰ (165157 at 13.4 cm, 54332 at 39.6 cm) — and costs 17 of 33 correct cuts.
**On the owner's "balance them" instruction that is his call, not the script's**,
and §5.1 proposes a cheaper alternative than paying it.

## 5. Where to improve next — ranked by what this round measured

Each item names the number it must move and the evidence it rests on. Nothing
here is a plan to tune a threshold; the thresholds are all measured out.

### 5.1 **The ν vertex, not the splitter** ← the highest-value front

**What the data says.** All 8 false fires have a ν vertex 13–96 cm from the
candidate's own charge (§4.1). Two of them destroyed hand π⁰s (165157: γ₂ 187.9 →
94.4 MeV, mass 152 → 108; 54332: γ₁ 185.9 → 129.8, mass 109 → 91) and those are
2 of the 3 pairs that crossed into kinematically impossible at **both** operating
points. On the vertex-attached stratum the splitter makes **zero** mistakes.

**The population scale, and its caveat.** Over all 400 production EM candidates
the median `vgap` is **9.6 cm** and **44.5 % exceed 13 cm**. But §4.2b: that
mixes wrong vertices with honest photon conversion gaps (≈18 cm mean free path),
**and this round could not separate the two.** `void_frac` was tested and came
back at AUC 0.146 — *backwards*: the false fires have the emptiest gaps of all.

**So the failure mode is sharper than "a bad vertex" and harder:** seen from the
wrong origin a single shower really does present two angularly separated lobes
with a real charge dip. Nothing measured *from that vertex* distinguishes it from
a genuine two-γ object. The fix has to come from outside the splitter.

**The proposal, in order of cost:**

1. **Ask whether the π⁰ re-seat should run BEFORE the splitter.** §B1 measured
   the main vertex moving **60.16 cm** (evt76346) and **14.50 cm** (evt396222)
   *after* the splitter has already measured every feature from the old point —
   5 of 172 objects. The π⁰ back-projection exists precisely because the ν vertex
   was wrong; running it first would give the splitter the corrected point. One
   ordering change, one gate, and it is testable on the arms that now exist.
2. **A `shower_split_max_vgap` knob (DEFAULT 0 = off) as a SCOPE declaration** —
   "the splitter does not act where its own reference point is untrustworthy" —
   priced at §4.3's table for the owner to pick a point. It is a blunt dial, not
   a discriminator, and §4.2b is why. Cheap, honest, and it buys the π⁰s back:
   a bound at 13 cm removes both pairs that broke.
3. **A vertex-quality instrument that does not read from the vertex.** The one
   thing not tried: does the *candidate's own* direction point back at the ν
   vertex within the pointing resolution? pr/129 built exactly that discriminator
   for a different question and it worked there. This is the one lead left, and
   it is a measurement, not a plan.

### 5.2 **A joint label set — the instruments currently contradict each other**

**What the data says.** §3.3: 93–94 % of the `q_miss` rise at both operating
points is the completeness instrument scoring a *correct* cut as a miss, because
its target comes from the 2026-08-27/28 attribution scan which called those
objects one shower. And the mirror: 28 % of the `q_extra` gain sits on objects
the split scan calls KEEP, where the attribution scan agrees the charge does not
belong. §3.5's evt314838 has **three** owner instruments giving three answers.

**This is now the binding constraint on measuring any further EM-clustering
work**, because every candidate improvement will be graded by an instrument whose
ground truth predates the question. The proposal:

1. **Re-mark the ~15 hand-marked showers the splitter touches** in the split
   tool, so `target` is defined per *part* rather than per merged object. That is
   a small scan — 15 objects, not 90 — and it makes `pr136_completeness.py`
   able to grade a splitter at all.
2. **Adjudicate evt314838 explicitly** (§3.5). It is the cleanest statement of
   the tension and one owner call settles which instrument leads.
3. Record in the tooling that **`pr136_completeness.py`'s absolute `q_miss`
   depends on which probe tapes the arm ran** (§3.3, proven by crossing pr/136's
   own dumps with this round's prepdir: 14.0 % → 15.1 %). Deltas within a matched
   pair are safe; absolutes across arms are not.

### 5.3 **B4 — re-home the daughter**, now with a measured mechanism

Still the owner's own written requirement from §A5.7 (*"should be part of the
earlier EM shower cluster"*, five separate comments). This round adds the
mechanism that makes it valuable rather than merely tidy: **charge peeled into an
orphan is charge the π⁰ finder cannot use.** At the production point the splitter
detaches 9.4e7 of charge and the π⁰ census does not move; at the escape point it
detaches the same and the census gains 2. A daughter offered to the nearest EM
shower under the existing absorb predicates keeps that charge available to
pairing. Metric: `q_extra` must not rise and π⁰ exact must not fall; knob
`shower_split_rehome`, DEFAULT OFF.

### 5.4 **Split-aware π⁰ pairing** — the tension §3.5 names

Splitting reduces a γ's energy, and the finder then pairs on the reduced value:
6 of 8 moved pairs went further from 135 MeV. Two candidate directions, both
measurable on the arms that now exist:

- run the pairing over the **pre-split** shower set as well and keep whichever
  pairing scores better — expensive but decisive;
- let a split daughter and its sibling be considered as a **single candidate γ**
  by the finder, which is what the π⁰ hand label for 314838 is effectively asking
  for.

### 5.5 The k≥3 kernel and the no-valley class — unchanged, and still the ceiling

- **k≥3** missed its pre-registered target (0.635 → 0.756 against ≥0.85, §B3) and
  is capped by `max_seeds = 4`, which cannot be raised without moving the trigger.
  Two owner objects are 5-part.
- **The no-valley class** (§B7): 7 of 9 missed splits have `valley_best = 1.0` —
  no charge dip exists because the γs overlap. Needs a different observable
  (transverse-profile fit, or the two-stub dE/dx of §B6 used as a *trigger*).

Both are real and both are below §5.1–§5.3 in value, because a better kernel on a
bad vertex is still a wrong cut, and an unmeasurable improvement is not one.

### 5.6 What NOT to do

- **Do not add another feature measured from the ν vertex** to fix the false
  fires. §4.2b tested the best candidate and it came back backwards; the
  information is not there.
- **Do not retune `valley_best ≤ 0.95`.** Its holdout is spent (§A5.4), the fires
  it produces are 80 % right overall and 100 % right on a good vertex, and the
  misses are a different problem (§5.5).
- **Do not chase the raw `+4.3 pt` of `q_miss`.** 93 % of it is §3.3's artefact.
- **Do not flip `shower_split` alone at production.** §3: no π⁰ gain, +3
  impossible pairs, and `q_extra` is already at its floor there.

## 6. Phase C leftovers

### C2. Byte-identity and the standard bar

Knob-off gate PASS on the standard 239-event manifest (478 archives), freshness
proof before the A/B (M1), `./build/clus/wcdoctest-clus` green, compiled-config
proof for every new jsonnet key.

### C1. Compose with pr/136 — ANSWERED in §3

The composition §C1 asked for is measured: `onV1c90 + splitter` returns `q_extra`
to 6.7 % (the OFF value is 6.9 %), keeps the escape's completeness gain, and
lifts the π⁰ census to 35 of 66. The remaining §C1 item — *"the owner finds no
new merged γs in a rescan"* — is the Bee scan §3 recommends, not yet done.

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

## 7. Sequencing, and why this order

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

# ============ THE FLIP QUESTION (section 3) and WHAT NEXT (sections 4-5) ============
# four arms, 239 events each, dumps on, pinned binary; only SBND_SHOWER_SPLIT differs
#   work-pr138r2-poff-<s>   / -pon-<s>     production config
#   work-pr138r2-c90off-<s> / -c90on-<s>   + SBND_PASS4_V1_ESCAPE=1 SBND_PASS4_V1_MAXV2=90
./scripts/pr138_flipcheck.sh                 # manifests -> prep -> em117_score x2 ->
                                             # completeness, pi0 census, mass closure, movers
python3 scripts/pr138_predict_delta.py --tape 'work-pr138r2-pon-*'   # PRE-REGISTERED signs
python3 scripts/pr138_flip_analyze.py --png  # -> pr138-flip-decision.{tsv,png}
python3 scripts/pr138_clean_strata.py --png  # -> pr138-clean-strata.{tsv,png}   section 4
python3 scripts/pr138_vertex_gap.py          # -> pr138-vertex-gap.tsv           section 4.2b

# section 3.3's baseline proof: the 15.1 % vs pr/136's 14.0 % is the PREPDIR, not the arm
cd em_display && ./em117_score.py --tag emscan-0827 \
    --manifest em117-136f086probe98-manifest.tsv --prepdir emprep-138poff \
    --tsv /home/xqian/tmp/xtest-oldarm-newprep-98.tsv   # returns 15.1 %, not 14.0 %

# ===================== PHASE B -- every number in section 2 =====================
# B0 amendment + B3: thirteen acceptance x assignment variants, offline
python3 scripts/pr138_kernel_k.py            # -> docs/pr/pr138-kernel-k.tsv

# the three arms.  Binaries are PINNED so a peer's wcbuild cannot void them:
#   /home/xqian/tmp/pin-pr138bare  = HEAD before this change
#   /home/xqian/tmp/pin-pr138off   = HEAD with it
# GATE (production config, the standard 239-event manifest, four samples)
LD_LIBRARY_PATH=/home/xqian/tmp/pin-pr138bare:$LD_LIBRARY_PATH \
  PR_JOBS=32 ./run_pr_chain_batch.sh work-<s>-grp0825 work-pr138r1-bare-<s> data <239 evts>
LD_LIBRARY_PATH=/home/xqian/tmp/pin-pr138off:$LD_LIBRARY_PATH WCT_SHOWER_SPLIT_DEBUG=1 \
  PR_JOBS=32 ./run_pr_chain_batch.sh work-<s>-grp0825 work-pr138r1-off-<s>  data <239 evts>
python3 scripts/pr85_hash_gate.py work-pr138r1-bare-<s> work-pr138r1-off-<s>; echo rc=$?

# FIDELITY + ON (onV1c90 config, so the population is the scan's; 125 evts)
export SBND_PASS4_V1_ESCAPE=1 SBND_PASS4_V1_MAXV2=90 WCT_SHOWER_SPLIT_DEBUG=1
LD_LIBRARY_PATH=/home/xqian/tmp/pin-pr138off:$LD_LIBRARY_PATH \
  PR_JOBS=32 ./run_pr_chain_batch.sh work-<s>-grp0825 work-pr138r1-dbg-<s> data <125 evts>
SBND_SHOWER_SPLIT=1 ...                      work-pr138r1-on2-<s>   # max_parts 2 (default)
SBND_SHOWER_SPLIT=1 SBND_SHOWER_SPLIT_PARTS=4 ... work-pr138r1-on4-<s>   # the k>=3 experiment

# B1: does the C++ reproduce the offline kernel?  (the vertex-controlled join)
python3 scripts/pr138_probe_compare.py       # -> docs/pr/pr138-probe-compare.tsv
# B2 + B3: the boundary the C++ ACTUALLY drew, and the peel it performed
python3 scripts/pr138_smoke_split.py --tape 'work-pr138r1-dbg-*' --on 'work-pr138r1-on2-*'
python3 scripts/pr138_smoke_split.py --tape 'work-pr138r1-on4-*' \
        --tsv docs/pr/pr138-smoke-split-on4.tsv
# tests + freshness
./build/clus/wcdoctest-clus; echo rc=$?
ls -la ../../../local/lib/libWireCellClus.so ../../../toolkit/clus/src/NeutrinoShowerClustering.cxx
```
