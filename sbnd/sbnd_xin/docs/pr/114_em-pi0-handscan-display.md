# doc pr/114 — `em_display`: a hand-scan display for EM shower clustering and π⁰

**Status: SHIPPED, scan-ready.** 94-event sample, probes parsed, **66/66 static
self-test checks and 21/21 in a real headless browser**. **No C++ and no jsonnet
changed — the toolkit repo is untouched, so no A/B gate is owed and none is
claimed.**

> **Round 3 (§11) added a real 3-D view** — rotate/zoom/pan with Bee's own charge
> cloud under the skeleton, inside the display, so the labels come along. Row 1
> is now a tab set, 3-D by default.

**Result.** A Bokeh display at `sbnd_xin/em_display/` for the owner's next
validation step. EM mode marks segments in/out of a selected shower against the
clustering's own acceptance geometry and, crucially, against **the pass that
actually absorbed each one** — read from probes pr/91 and pr/93 already built
into the code, not inferred. π⁰ mode builds a π⁰ from scratch (assign two γ,
set start points, choose or back-project the vertex) and shows the mass under
**both** direction conventions, because the code's own stored mass and angle use
different recipes and do not close. Four measurements came out of building it:
the stage-2 re-run **reproduces prod0825 on every physics field for all 94
events** (only the per-process `shower_id` counter moves); the lossy
`segments[].shower_id` join is **15 of 1567 showers (1.0 %), all 15 repaired** by
the probe; the gate the display draws (`pass3_cone`) is the **largest single
absorber at 41 % of 4030 absorptions**, with `pass4_angle` second at 21 %; and
**doc pr/113's census had a falsy-zero bug** that undercounted π⁰-paired events
by 10× — found because this display re-derives the pairing independently.

> **Read doc pr/113 §10 first if you have the round-1 lists.** They changed:
> nueCC 98 → **90**, NCπ⁰ 23 → **46**, numuCC-EM 79 → **79** (byte-identical).

## Repro

```sh
cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin

# stage 2 -- re-run the 94-event sample with the shower probes on (~10 min, 4 fresh arms).
# ALREADY DONE: the four work-em114-* arms exist, so this now prints "NOTHING WAS RUN"
# and exits 0.  That is the expected steady state, not a success -- move an arm aside
# to genuinely redo it.
./em_display/run_em114_probe.sh

# parse the probes, rebuild the manifest, resolve Bee links offline (no network)
python em_display/prep_em_scan.py --parse-probes

# the three self-tests behind every number in this doc
python em_display/selftest_repro.py          # reproduction + membership repair
python em_display/selftest_em_display.py     # drives the viewer's callbacks, 66 checks
python em_display/selftest_em3d_browser.py   # drives the 3-D view in headless chromium, 21

# serve
./em_display/serve_em_display.sh 5021 --scan-tag <your-tag>
#   ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -L 5021:localhost:5021 wcgpu1
#   http://localhost:5021/em_display_viewer
```

Anchors: toolkit `8d93260d` (**unchanged**), wcp-porting-img `6373efa` (the
pr/113 round-2 correction, which this round depends on).

---

## 0. Two corrections to the brief, made once and then built around

### 0.1 The π⁰ vertex is already reconstructed — the display judges it

The ask was *"for NCpi0 … it would also need to reconstruct the pi0 vertex, and
then display."* The code already does. `id_pi0_without_vertex` back-projects the
two γ rays with `WireCell::ray_closest_points`
(`clus/src/NeutrinoShowerClustering.cxx:4177`; helper `util/src/Point.cxx:125-171`)
and then **overwrites the main vertex** with the answer:

```cpp
// NeutrinoShowerClustering.cxx:4338-4340
main_vertex->fit().point = vtx_point;   // "(hack) - set to reconstructed pi0 decay point"
main_vertex->fit().dQ = 0;              // <- the tell-tale
```

Also, *"without vertex"* does **not** mean the event has no neutrino vertex: the
function returns early on `!main_vertex` (`:3961`) and only considers showers
with `start_connection_type == 3`, i.e. **detached** γ. A genuinely vertex-less
event gets no π⁰ at all.

So the deliverable is smaller than asked and more useful: show the vertex the
code computed, mirror the same back-projection so its inputs can be varied, mark
whether the code's own 25° gates would have accepted it, and let the owner
override. All three are built.

**A by-product worth knowing: `id_pi0_without_vertex` never fires on this
sample.** Across all 1433 prod0825 dumps there are **70 events carrying 78
accepted π⁰ groups**, and every one is flag-1 (`id_pi0_with_vertex`). Two
independent lines agree:

- **0 of 70** events have the `fit.dQ == 0` tell-tale on the main vertex, so no
  main vertex was ever overwritten by the back-projection;
- **0 of 78** groups contain a `start_connection_type == 3` member — the only
  kind `id_pi0_without_vertex` considers (`:4126`). The observed pairs are
  (2,2) × 46 and (1,2) × 32.

So the detached-γ path is live code that has produced nothing here. That is a
finding for a later round, not something this display fixes — but it is also what
makes §8's inference safe, and it would stop being safe the moment the path
fires.

### 0.2 Direction and π⁰ pairing needed no new dump field

`showers[]` already carries `start`, `end`, `start_connection_type`,
`start_vertex_id`, `num_segments`, `kine_charge`, `pio_id`, `pio_mass`,
`stem_dqdx`; `segments[]` carries `shower_id`, `dirsign`, `length`, `points[]`.
The shower axis is not a dump key, but the stage-2 probe prints it (`dir15`), and
where the probe is absent `em_geom.shower_init_dir` mirrors the C++ branch logic
(`PRShower.cxx:1552-1562` single-segment, `:1618-1640` multi). Two one-line
getter adds to `PrDisplayDump.cxx` were considered and **declined** (§6).

---

## 1. What was built

```
em_display/em_display_viewer.py    the app: one program, two modes
em_display/em_geom.py              the C++ geometry, mirrored with citations
em_display/em3d.py                 round 3: the 3-D camera, its CustomJS, the Bee cloud
em_display/prep_em_scan.py         manifest + probe parser + Bee index/build
em_display/run_em114_probe.sh      stage-2 launcher
em_display/serve_em_display.sh     bokeh serve wrapper, port 5021
em_display/selftest_em_display.py  66 headless checks over the real sample
em_display/selftest_em3d_browser.py  21 checks driving a real headless chromium
em_display/selftest_repro.py       reproduction + membership repair
em_display/em114-manifest.tsv      the 94-event sample, links, per-event stats
em_display/emprep/emprep-evt*.json 94 probe sidecars
em_display/README.md               operating manual
```

Fork by duplication (CLAUDE.md §2 Code / M10): `pr_display/pr_display_viewer.py`
is **byte-untouched**. Its projection block, detector box, reentrancy guards,
Bokeh-3.9 DataTable repaint workaround and label-writer shape were copied and
then changed freely. No shared helper was extracted.

---

## 2. The sample

94 events = the whole corrected NCπ⁰ list (46) + the curated nueCC48 arm (48).
By arm: ncpi0 19, nuecc48 48, mcp1k 10, mcp2k 17.

| | n |
|---|---|
| events | 94 |
| …with a reconstructed π⁰ group | **42** |
| …with a lossy shower join | 12 |
| …with a probe sidecar | 94 |
| …with an offline Bee link, no upload needed | **78** |

42 of 94 carrying an actual reco π⁰ pair is what makes the π⁰ mode worth
building as an *audit* as well as a construction tool — and that number only
exists because of the pr/113 §10 correction. Under the round-1 census it would
have been 4.

---

## 3. Stage 2 reproduces prod0825 — measured, not assumed

The concern was real: prod0825 ran in **group mode** (`PR_GROUP_SIZE=16`) against
`libWireCellClus.so` installed 2026-08-25 **02:04**, and the re-run is per-event
against the lib installed **16:34** the same day, with the doc-80 MCS commits in
between.

`selftest_repro.py` compares every one of the 94 events, field by field:

> **all 94 events identical on `main_vertex`, `kine`, `tagger`, `showers` and
> `segments`.**

The **only** field that moves is `showers[].shower_id`, and it moves everywhere:
prod0825 numbers evt21073's showers 20…33 where the re-run numbers them 0…13,
because `Shower::get_shower_id()` is a **per-process** counter and prod0825 ran
16 events per process. It is not a physics quantity. Everything in this round
therefore joins on `showers[].id` / the probe's `node_id`
(`cluster_id*1000 + segment index`), which is stable.

*(Consequence worth recording: any consumer joining on `shower_id` across arms or
across group boundaries is wrong. `scripts/pr93_shower_composition.py:85` labels
its two id columns the wrong way round — it emits the start-segment id as
`shower_id` and the sequential id as `start_seg`. Mentioned, not fixed,
CLAUDE.md §5.)*

Because the two agree, **the display reads the prod0825 dumps** — the canonical
products — and takes only probe text from the em114 arms.

---

## 4. Membership: the dump is lossy in 1 %, and the display says so

`segments[].shower_id` stores **one** shower per segment
(`PrDisplayDump.cxx:490`), so when two showers overlap the loser's members
disappear from the join and it comes back looking *empty rather than nested* —
the condition `NeutrinoShowerClustering.cxx:116-126` describes.

Measured over the 94-event sample (1567 showers):

| | n |
|---|---|
| showers | 1567 |
| lossy join (`join != num_segments`) | **15 (1.0 %)** |
| repaired by the probe | **15 (100 %)** |
| probe member count == `num_segments` | **1567 / 1567** |

Worst cases, both in the sample and both used as the display's regression tests:
ncpi0 **evt84229 shower 69134** (43 of 50 joined, a 958 MeV shower) and ncpi0
**evt463565 shower 109073** (**0 of 5** — renders as an empty shower).

**The key point is that the dump is lossy but not silent.** `num_segments` is the
shower's own count, so the mismatch is computable with no re-run at all. The
display shows a `joined` column and a banner on affected events, and with a probe
sidecar it uses the non-lossy membership and the banner reads `REPAIRED`. Without
a sidecar it still refuses to pretend: the banner says the join is incomplete and
by how much.

**Axis provenance:** 1565 of 1567 shower axes come from the probe's `dir15`, i.e.
the C++'s own `shower_cal_dir_3vector(shower, start, 15 cm)`. Only 2 fall back to
the Python mirror. The probe prints components at `%.3f`, so vectors arrive with
|v| in 0.9994–1.0004 and are re-normalised on read — ≈0.06° on any angle, far
below anything a scan turns on, but they must *be* unit vectors for the
arithmetic. (Caught by the self-test, which asserted unit length.)

---

## 5. Where segments actually come from — a census

The `site=` tag names the pass that absorbed each segment. Over 4030 absorptions
in the 94 events:

| absorbing pass | n | share |
|---|---|---|
| `pass3_cone` (`:1310-1350`) | 1665 | **41.3 %** |
| `pass4_angle` (`:1964-1977`) | 838 | 20.8 % |
| `in_main_cluster` | 714 | 17.7 % |
| `from_vertices` | 408 | 10.1 % |
| `pass4_proximity` | 114 | 2.8 % |
| `pass3_cluster_map` | 67 | 1.7 % |
| 9 others (`in_other_clusters_*`, `examine_shower_1_*`, `conn3_unreachable`, `stem_backfill`, …) | 224 | 5.6 % |

Mechanism split: 2754 `direct` adds, 1259 flood-fill `walk_add`, 17
`walk_exclude`. Shower–shower splices are rare (22 total, led by
`examine_shower_1_assoc` 8 and `examine_showers_vtxcontain` 6).

**This is what justifies the acceptance plot and simultaneously bounds it.** The
gate it draws is the largest single absorber by 2×, so the panel is looking at
the dominant mechanism — but 59 % of absorptions happen elsewhere, so the panel
is a *geometry aid*, never a verdict. The UI says so in two directions:

- **above every step ≠ rejected** — `pass4_angle` uses different constants
  (`:1964-1967`);
- **below a step ≠ absorbed** — `shower_cone_absorb_guard` is **SBND-ON**
  (`wct-pr-perevt.jsonnet:1671`) and declines a confidently-PID'd non-electron
  straight track longer than 50 cm (`:1336-1351`, pr/93 Cause D).

The `absorbed by` column is the authority.

---

## 6. What the display shows, and the two traps it keeps apart

### 6.1 π⁰ mass, both conventions, side by side

`mass = √(4·E₁·E₂·sin²(θ/2))` with `E = kine_charge` — the code's own formula and
the code's own energy (`:3771`, `:4199`, `:4250`). The display computes θ two
ways: from the showers' **own axes**, and from the two **vertex→start chords**.

Both are shown because the code itself is inconsistent here: the mass it stores
uses `local_dirs` (`get_init_dir()` or the vertex chord, `:3760-3771`) while the
angle it stores is recomputed from `shower_cal_dir_3vector(…,15 cm)`
(`:3812-3831`). Over 282 events carrying both, `2√(E₁E₂)·sin(angle/2)` reproduces
the stored mass within 1 % in only **171**, and misses by >5 % in **78**.

Two mechanisms are located and **neither is picked** (CLAUDE.md §5.4): the
within-block recipe mismatch above, *and* the fact that the stored pair may be
one no reconstruction accepted (§6.2). The 78 is a mix of both and is not
attributed to either.

*Unit convention verified on the way:* `kine.kine_pio_angle` is in **degrees**
in the dump although `:3830` is an `acos()` — converted at
`NeutrinoKinematics.cxx:597-598`. Treating it as radians yields a mass wrong by a
factor you would not notice. (House precedent: the dQ/dx unit trap, doc pr/42.)

### 6.2 `pio_id` is the pairing; `kine_pio_*` is a BDT feature

They can name **different pairs**. On ncpi0 evt21073 the accepted groups are
(60081 + 31023, 127.2 MeV) and (11008 + 63100, 111.2 MeV) — while `kine_pio_*`
reports `energy_1` 680.2 (from 60081) and `energy_2` 104.7 (from 63100): a third
pairing, mass 207.25, that no reconstruction accepted. Cause: `pio_kine` comes
from a separate highest-energy scan over *all* candidate pairs (`:3777-3832`,
`:4260-4297`), while `map_pio_id_mass` records only what the winner loop accepted
(`:3836-3891`, `:4319-4334`).

The display draws them in **separate blocks** and never merges them. The
self-test asserts this (it would fail if a future edit merged the tables), and
`docs/pr/pr113-ncpi0.index.txt`'s `pio_mass` column is the `kine` variety — which
is why that column runs 10 → 1166 MeV and is almost never 135.

---

## 6.3 The back-projection has three branches, and the middle one is the common case

Caught by review *after* the first commit, and worth recording because a
plausible-looking mirror was wrong on the majority of real pairs.

`id_pi0_without_vertex` does not compute one vertex; it computes one of three:

| condition | vertex | line |
|---|---|---|
| both γ longer than 15 cm | midpoint of closest approach | `:4182-4200` |
| **exactly one** longer than 15 cm | **re-ray the short γ from that midpoint, re-intersect, keep the closest point on the LONG γ's ray** (not the midpoint; and no 3 cm fallback on this branch) | `:4203-4247` |
| both ≤ 15 cm | no π⁰ | `:4254-4255` |

The first mirror implemented only the first branch. Over the 78 accepted pairs on
disk the split is **29 both-long / 49 one-short** — so the majority took a branch
that did not exist in the code. Measured displacement between the midpoint the
first version would have shown and the vertex the code actually computes, over
those 49 pairs:

| | cm |
|---|---|
| median | **2.68** |
| p90 | **24.89** |
| max | **43.40** |
| pairs off by >1 cm / >5 cm / >20 cm | 36 / 20 / 6 |

All three branches are now mirrored, `pi0_backproject` returns which one ran, the
UI names it, and `selftest_em_display.py` asserts both live branches are
exercised and that one-short is 49 of 78 — so a regression that silently drops a
branch fails the test rather than quietly moving a vertex 25 cm.

## 7. Verification

`selftest_em_display.py`, 30 checks, all passing, over the real sample:

- **evt463565 shower 109073** renders as `5 / 5` with the note *"join lossy 0/5,
  REPAIRED by probe"* — not as an empty shower. The §4 regression case.
- **evt84229 shower 69134** has all 50 members, a unit-length probe axis, arrows
  in ≥2 panels, a populated candidate table with `absorbed by` filled, and at
  least one segment inside a pass-1 tier.
- **evt21073** yields exactly 2 accepted groups at 111.2 / 127.2 MeV, the kine
  block is labelled a BDT feature and its 207.25 shown separately.
- A hand-built pair (60081 + 31023) gives θ = 32.1°, m = **125.7 MeV**; the
  back-projection returns verdict `ok` with a 2.44 cm closest-approach gap.
  (The reco's own accepted group for those same two showers reads **127.2 MeV** —
  not a discrepancy: 125.7 is the *axis* convention, the reco's is `local_dirs`
  per §6.1. A 1.5 MeV spread between the two conventions on a well-behaved pair
  is the scale to have in mind when reading the >5 % misses.)
- Both back-projection branches fire on the real sample, 29 both-long and
  **49 one-short**, and the test pins that split (§6.3).
- Snap moves a γ start onto a real fitted point; a label round-trips through
  save → reload with both γ slots and both verdicts restored; the record carries
  the reco's groups **and** the `kine_pio_*` block **and** full provenance.
- The M13 tag guard refuses an implicit tag that already holds labels and accepts
  an explicit one.

`selftest_repro.py` produces §3 and §4's numbers.

**No A/B gate is owed.** No C++, no jsonnet, no config: there is no
byte-identicality claim to make and none is made. The stage-2 probes are
env-gated and stderr-only, and §3 measures rather than assumes that.

---

## 8. Declined, and what travels instead

Two one-line plain-getter adds to `PrDisplayDump.cxx` were offered and the owner
chose the re-run without them:

- `showers[].init_dir` from `Shower::get_init_dir()` (`PRShower.h:158`);
- `showers[].pio_method` from `map_pio_id_mass.at(id).second`
  (1 = `id_pi0_with_vertex`, 2 = `..._without_vertex`, currently dropped at
  `PrDisplayDump.cxx:618-620`).

So the toolkit stays unchanged and two caveats travel in the display instead. The
first turned out to cost almost nothing — the probe's `dir15` covers 1565/1567
axes (§4). The second means the display cannot *state* which π⁰ finder produced a
vertex; it infers it from `start_connection_type == 3` plus the `fit.dQ == 0`
tell-tale and labels the inference as such. §0.1's finding — that the flag-2 path
never fires on this sample — makes that inference safe for now, and would stop
being safe the moment it does fire.

## 9. What is NOT claimed

- **No physics claim about any event.** This round ships an instrument and the
  measurements needed to trust it. No shower is judged, no π⁰ is confirmed, and
  no knob is proposed.
- **No purity or efficiency.** The sample is reconstruction-defined on data;
  there is no truth to measure against (doc pr/113 §6.1).
- **The acceptance plot is not a verdict** (§5). 59 % of absorptions happen
  outside the gate it draws.
- **The 78/282 mass–angle non-closure is not attributed** to either of its two
  located mechanisms (§6.1).
- **`em_geom.shower_cal_dir_3vector` is not bit-exact.** The C++ walks the
  shower's view graph; the Python walks the dump's `fill_sets` membership. It
  affects only the 2 showers not covered by the probe's `dir15`.
- **`ray_closest_points` is mirrored, including its quirks.** The C++ signals the
  parallel case by returning the two ray origins, indistinguishable from an
  answer; and `scale1` divides by a quantity that vanishes when the rays already
  intersect (`Point.cxx:153`), yielding inf/nan silently. The Python reports
  `parallel` / `degenerate` instead of propagating either. That is a deliberate
  divergence in the *display*, not a proposed change to the code.

## 10. Two operational notes

**One prod0825 log is truncated, and a shipped tool greps it.**
`scripts/bee/make_pr_bee.py:78` decides whether an event was evaluated by
grepping the per-event log for `TaggerCheckNeutrino: selected main cluster`.
ncpi0 **evt399860**'s prod0825 log is **22 KB against a 207 KB median** and is
missing that line, so the builder refuses an event whose dump is perfectly good
(main vertex, 9 showers, 47 segments). It is isolated — 18 of 19 ncpi0 and 48 of
48 nuecc48 prod0825 logs carry the line, and the em114 arms carry it 94 of 94 —
so this is one truncated file, not a group-mode pattern. `prep_em_scan.py`
therefore points `make_pr_bee.py` at the **em114** arms, which is safe precisely
because §3 shows the two are equal on every physics field. Mentioned, not fixed
(CLAUDE.md §5): the general lesson is that a **log-derived** predicate is weaker
than a **product-derived** one, and `has_main` from the dump disagrees with it
here.

**The Bee set is built and waiting.** `bee/em114/` holds four zips (94 events,
51 MB) with their `.index.txt` and `.prid-map.txt` sidecars; every zip's first
member is `data/0/...`, which is what the server validates (`views.py:241`).
Upload is the owner's step (§5.6) — after it, dropping each returned URL beside
its zip as `.url` and re-running `prep_em_scan.py` fills the remaining 16 links
with no other change.

---

## 11. Round 3 — a real 3-D view inside the display

**Status: SHIPPED.** 66/66 static self-test checks **and 21/21 in a real
headless browser**. Still **no C++ and no jsonnet — the
toolkit repo is untouched, so no A/B gate is owed and none is claimed.**

The owner's ask, verbatim: *"the em_display is good, but for the hand scan, it is
not as good as bee, where one can easily zoom in rotate etc. and see the result.
One major advantage of the em_display is that it has all the lable information,
which is great to record the result. I wonder if we can have this 3D display
feature (like what's in bee) integrated into the em_display?"*

Row 1 is now a tab set: **3-D** (default) and **2-D projections**. The 3-D panel
rotates, zooms and pans, draws the same charge cloud Bee draws, and every
existing label control works off it — including a box select that resolves to
whole segments.

### 11.1 Repro

```sh
cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin
python em_display/selftest_em_display.py     # 66 static checks, 0 failures
python em_display/selftest_em3d_browser.py  # 21 checks in headless chromium, 0 failures
python em_display/selftest_repro.py         # unchanged: 1567/1567
./em_display/serve_em_display.sh 5022 --scan-tag <your-tag>
```

### 11.2 Not three.js, and the reason is on disk

Bee loads **three.js r145 from a CDN**
(`wire-cell-bee3/events/templates/events/event.html:299-306`). The only copy in
the tree is **r71** (`events/static/js/lib/three.min.js`, 420 KB, 2015), which is
*not* dead weight — `physics/deadarea.js:20-25` fetches it at runtime and
concatenates it into a Blob Web Worker that uses `THREE.Geometry`,
`SplineCurve3` and `BufferGeometry().fromGeometry()`, all removed by r125+. So
the on-disk copy is the wrong version for the main scene and is pinned to the
worker. On top of that, Bee's own bundle `js/bee/dist/bee.js` is gitignored and
**not built** here, there is no `node`/`npm` on this box, and `em_display` is a
single-script Bokeh app with no `static/` dir (all 19 Bokeh apps in the tree
are). Vendoring Bee is a project, not a step.

### 11.3 What was built instead, and the fact it rests on

An **orthographic trackball inside an ordinary Bokeh figure**. Every glyph
carries 3-D columns plus the projected pair it draws; a `CustomJS` recomputes the
projection in the browser each drag frame and calls `source.change.emit()`. No
new dependency, no JS asset, no build step — and because the glyphs live in
normal data space, Bokeh's tap, box-select and hover keep working. That is the
whole reason for putting the 3-D view *inside* em_display rather than beside it.

It works because of one non-obvious thing, **read in the shipped bokehjs rather
than recalled** (`bokeh/server/static/js/bokeh.js`):

- `UIEventBus.__trigger` calls `this._trigger_bokeh_event(plot_view, e)` at its
  tail, **after** the active-tool switch and unconditionally. So `Pan`,
  `PanStart`, `PanEnd` and `MouseWheel` reach `js_on_event` **with no pan or
  scroll tool active** — which is what lets a bare drag mean "rotate" with
  nothing fighting it for the gesture. `PointEvent` carries `modifiers`
  (shift/ctrl) and cumulative `delta_x`/`delta_y`.
- `GlyphRendererView.connect_signals` does `this.connect(this.model.data_source
  .change, update)`. So mutating `source.data.<col>` **in place** and calling
  `change.emit()` repaints locally without assigning `.data` — the difference
  between a local repaint and shipping 25 000 points back to the server on every
  frame of a drag.

| gesture | effect | mechanism |
|---|---|---|
| drag | rotate | `js_on_event(Pan)`, guarded by the live gesture state |
| shift+drag | pan | same handler, `cb_obj.modifiers.shift` |
| wheel | zoom | a real `WheelZoomTool` as `active_scroll` (it is also what calls `preventDefault`, so the page does not scroll) |
| Box Select | segment selection | the guard suspends rotation while it is active |

**Framing is set from the 3-D bounding sphere of the reconstruction, never from
the projected extent.** `right/up/fwd` is orthonormal, so `u² + v² ≤ R²` for
every camera: rotating an elongated track from broadside to end-on can neither
balloon it out of frame nor shrink it to a dot, and all zoom stays the user's.
Framing off the projected extent would re-fit on every drag frame — the same
failure the `Range1d`-not-`DataRange1d` comment in the viewer already warns
about, one level up. The **cloud does not set the frame** by default: a
cosmic-laden cloud spans the whole TPC and would leave the neutrino a speck.

**Depth cueing, not depth sorting.** Bokeh draws in row order, so the only
occlusion cue available without permuting every column every frame is alpha and
size falling off with depth. Sorting 34 000 points per frame to get true
occlusion is not worth it; fading is the cue that carries depth in a still frame
and motion parallax covers the rest on a drag.

### 11.4 The blocker, cleared before anything was designed

doc pr/13 warns that **`img-global` is the only raw-frame layer** in a Bee zip —
dumped pre-pipeline, before `ClusteringSwitchScope` creates the corrected arrays
— while `clustering-global` and the PR layers are in `(x_t0cor, y_cor, z_cor)`,
with a per-cluster T0 offset running to **±121 cm**. Drawing the skeleton over
the wrong cloud would be worse than having no 3-D at all, so it was measured
first. Calib dump vs the zip's **own PR layers**, first 12 manifest events:

```
dump segments[].points[]  ->  track_fit-global : NN median 0.00043 cm (max 0.00085)
dump main_vertex          ->  vertices-global  : 0.00007 .. 0.00059 cm
```

The same numbers, to JSON rounding. The dump is in the PR-layer frame, and pr/13
pins the PR layers to `clustering-global` (NN median 0.0010 cm). Hence:
**`clustering-global` is the base layer; `img-global` is offered only behind a
red warning naming what it is.**

Worth recording *why* the obvious check was not the one that decided it. Fit
points sit ~0.34 cm from **both** clouds — which reads as "no offset" but is just
the point spacing, and is that small for both only because fit points live on the
in-beam cluster, whose T0 shift is ~0. A near-miss like that is exactly how a
misaligned overlay ships. The `track_fit-global` comparison is the one with
discriminating power, and `selftest_em_display.py` pins it on three events so it
cannot rot.

### 11.5 The cloud, and what it costs

`bee/em114/*.zip` (built in round 2) already hold what Bee draws, so the panel
and the Bee link beside it show the same reconstruction. Over the 94:
**median 33 868 points, p90 56 255, max 81 814** (ncpi0 evt256587).

- Decimation walks a **fractional index**, not a `[::k]` stride: deterministic,
  proportional per cluster, and it hits the budget exactly. A stride cannot —
  at 25 586 points with a 25 000 budget it takes k=2 and throws away half the
  event to save 586 points.
- Numeric columns go over the wire as **float32 numpy arrays**, which Bokeh
  serialises as binary buffers rather than JSON numbers. The initial document
  went from **3.68 MB to 1.37 MB** (0.42 MB JSON + 0.95 MB buffers), 2.7×
  smaller and far faster to parse — and this display is always used through an
  ssh tunnel, so that is felt.
- Event load, all 94: **mean 0.22 s, worst 0.28 s**. Server-side projection of
  the largest cloud (81 814 points): **14.6 ms**.
- The zips are gitignored, so a fresh clone gets the display but not the cloud.
  The panel then draws the skeleton and says so in a banner — the same pattern
  as the optional probe sidecar.

### 11.6 Two browser-only bugs, caught by reading bokehjs

Neither would have produced a single server-side error. Both were found by
reading the shipped `bokeh.js` rather than by testing, because there is nothing
here to test them with.

**(a) `toolbar.active_drag` is the configuration, not the live state.** The first
version guarded rotation with `if (p.toolbar.active_drag != null) return;`. But
`Toolbar._active_change` writes the live gesture to **`this.gestures[et].active`**
and never touches `active_drag`, which stays at whatever it was configured to
(here `None`). The guard would have been permanently false and rotation would
have fought box-select on every drag. The correct read is
`toolbar.gestures.pan.active` — exactly what `UIEventBus.__trigger` itself
consults. `BoxSelect`, `BoxZoom`, `Lasso` and `Pan` all declare `event_type`
`"pan"`, so the one check covers every drag tool.

**(b) A dict in `CustomJS.args` — a trap that turned out narrower than feared,
and the correction is the point.** Bokeh serialises a Python dict as
`{"type":"map", entries:[...]}`, which looked like it would arrive as a JS `Map`
and make `cfg[i].alpha` undefined. Reading `_decode_map` in `bokeh.js` shows it
returns a **plain object whenever every key is a string**, and a real `Map` only
when one is not. So string-keyed config dicts were always fine. The code now
passes three parallel arrays anyway — not to dodge a bug that does not exist, but
because it lets one table (`_PT_CFG`) be the single source of truth that both the
Python fill and the JS frames read. The selftest guards the trap **as it actually
is**: no *non-string-keyed* dict in `args`.

### 11.7 What is verified

`selftest_em_display.py` grew from 30 to **66** checks. The new ones:

- `camera_basis` orthonormal over an (az, el) grid, and `u² + v² + d² = |p − c|²`
  exactly; `u² + v² ≤ R²` for every camera over a 168-camera sweep.
- Presets `x-z` and `z-y` reproduce the 2-D panels exactly (`right`/`up` to 1e-9).
- The §11.4 frame assertion on three named events.
- Cloud loader: exact budget, equal-length columns, every cluster surviving
  decimation, and a missing zip returning `None` rather than raising.
- The layer contract: `set(RENDER) == set(LAYER_KEYS)`, every 3-D renderer
  layer-controlled except the pick surface, and the cloud checkbox hiding *both*
  colour modes (they share a CDS).
- Column-length invariant on all 94 events.
- A 3-D box over many points resolving to a handful of **segments**, marking
  working off it, tap-in-fill-mode landing on a real fitted point, and a stale
  selection being cleared on event switch (replacing `.data` does not clear
  `.selected`, and a stale index would make the next "mark IN" hit the previous
  event's segment).
- The JS lint: every free name supplied through `args`, brackets balanced, the
  live-gesture guard present and `active_drag` absent, no divergent copy of the
  projection, no non-string-keyed dict in `args` — **plus a test of the linter
  itself**, since a linter that never fires proves nothing (its first version
  reported 150 false positives because it only took the first declarator of
  `const rx = …, ry = …, rz = …`).

**And the browser-side code IS machine-tested after all** —
`selftest_em3d_browser.py`, **21 checks, 0 failures**. The first pass of this
round concluded it could not be: there is no `node`, `deno`, `esprima`, `js2py`,
`dukpy` or `quickjs` in the tree, so the JS could not even be parsed here, and
§11.7 was written around a manual check-list. That was the wrong conclusion from
the right evidence — a *JS engine* is absent, but **playwright's bundled chromium
is installed** (`~/.cache/ms-playwright/chromium-1228`), and a headless browser
executes the code the same way the owner's will. The lesson is narrow and worth
keeping: *"no interpreter for this language"* is not the same question as
*"nothing here can run this code"*.

The script starts its own bokeh server, drives real mouse gestures, and reads the
result back out of the live models. Bokeh compiles a `CustomJS` body **lazily**,
on first execution, so loading the page proves nothing — every check triggers a
handler. Two of them could not have been obtained any other way:

- **The two mirrors of the projection agree exactly.** After a camera change the
  browser's own `u`/`v` are read back out of the `ColumnDataSource` and compared
  point by point against `em3d.project` in Python: over 200 sampled points the
  **worst |Δu|, |Δv| is 0.00e+00 cm**. That is the drift risk the whole
  two-mirror design carries, closed by measurement rather than by discipline.
- **The gestures do what §11.3 claims.** A synthetic drag really produces `Pan`
  events with no drag tool active and really rotates (az −1.571 → −0.371);
  shift+drag pans (`x_range.start` −300 → −434) and leaves the camera untouched;
  the wheel zooms (span 600 → 246 cm); and **activating Box Select really
  suspends rotation** — the §11.6(a) bug, confirmed fixed rather than merely
  reasoned about — while selecting **493 points that resolve to 27 segments**,
  which is §11.3's selection-unit claim demonstrated.

It also closes a hole the tab change opened. Moving the three projections into a
`TabPanel` changed their render path: they now initialise *lazily*, and Bokeh has
a long-standing wart where a plot in an inactive tab comes up mis-sized. So "the
2-D path is untouched" was not quite true, and the test now asserts it directly —
switch to the tab, **three distinct panels at full size**, and a tap in two of
them still pins a full 3-D point (`['6.1', '10.0', '258.1']`). (Writing that check
needed one correction of its own: every Bokeh figure owns *two* stacked canvases
at the same rect, so the first version tapped the same panel twice and filled x
and y but never z.) Bokeh 3 also renders every view inside an **open shadow root**
— five deep here — so `document.querySelectorAll('canvas')` finds nothing at all
and the walker has to descend `shadowRoot` explicitly.

What still needs a human, because it is a judgement and not an assertion:

| on | check |
|---|---|
| ncpi0 **evt256587** (82 k cloud) | is a drag still smooth at `max points` 100 000? If not, lower the default — do not cap silently |
| any event | does the depth fading actually read as depth, or does the cloud read as fog? |
| ncpi0 **evt84229** | is the cloud+skeleton overlay legible enough to judge membership from? |
| any event | is `frame the reco` the right default, or is `frame the cloud` wanted more often? |

### 11.8 Noticed, not touched

A stray untracked file named `angle` at the root of `wcp-porting-img`, almost
certainly a mis-redirect. Reported rather than deleted (CLAUDE.md §5).

---

## 12. Round 4 — the scan-ergonomics round

**Status: SHIPPED.** 105/105 static self-test checks and **29/29 in a real
headless browser**. Still **no C++ and no jsonnet — the toolkit repo is
untouched, so no A/B gate is owed and none is claimed.**

The owner's ask, verbatim, four items:

> *"1. currently, everything is shown on the left side, and it would be good to
> spread things to the right in the browser as well, since I have a bigger
> screen. 2. In the 3D plot, I only need to see the results related to the
> neutrino candidate, not the cosmic muons, which makes the display noisier.
> 3. I wonder if there is a way for me to a) click a point in the 3D display, so
> that I can rotate around that point b) be able to click the 3D display so that
> I can use that to select things in and out? 4. when I click an EM shower, it
> would be bettet to improve the highlight in the display, so I know exactly
> which part is included, and which part is not, so that I can further click to
> include them or not. Note, it would be great if there is a way to diferentiate
> what initially have vs. what I clicked for the hand scan. Similar comment to
> the pi0, since I want a way to click a vertex to say if this is pi0's vertex
> point."*

### 12.1 Repro

```sh
cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin
python em_display/selftest_em_display.py    # 105 static checks, 0 failures (~60 s)
python em_display/selftest_em3d_browser.py  # 29 checks in headless chromium, 0 failures
python em_display/selftest_repro.py         # unchanged: 1567/1567
./em_display/serve_em_display.sh 5022 --scan-tag <your-tag>
```

### 12.2 Item 2 first, because it needed a measurement before a design

"Only the neutrino candidate, not the cosmic muons" splits into two questions
with different answers, and the split is the finding.

**The skeleton is already the candidate.** `WCPPID::NeutrinoID` is constructed
from the main cluster plus the other clusters of the same flash bundle and never
sees anything else, so every segment in a `calib-pr-evt*.json` belongs to the
candidate by construction. What makes it *look* spread out is real: measured over
all 94 events, the bounding radius of all reco is a median **162 cm** while the
main cluster alone is **38 cm**, and the furthest reco point sits a median
**239 cm** from the main vertex. In evt 64591 the main cluster is a 186 cm track
and the 298 MeV shower hangs off its *far* end, 190 cm from the vertex. Hiding
any of that would hide segments the scan exists to judge, so nothing is hidden.

**The charge cloud is where the cosmics are.** `clustering-global` is every
cluster in the readout — 22 of them in evt 64591, of which 3 carry the
reconstruction. So the filter belongs there, and the reduction is large:

| | median | p90 | max |
|---|---|---|---|
| cloud points, all clusters | 33 868 | 56 255 | 81 814 |
| cloud points, candidate only | **6 135** | 12 584 | 30 323 |
| fraction of the cloud kept | **0.186** | 0.396 | 0.552 |
| clusters kept | 2 | — | 7 |

**How the two are matched.** The numberings do not meet — the dump's
`cluster_id` is WCP's PR sub-cluster index (17, 24, 25, 57…85 in evt 64591) and
the cloud's `real_cluster_id` is WCT's clustering id (3, 13…20, 30) — and the Bee
PR layers do not bridge them either: `track_fit-global` carries dump *segment*
ids (17002, 81037), not cluster ids in either namespace. So the link is made in
space, and the direction that works is **reco → cloud**: every reco point sits on
real charge, so whichever cloud clusters the reconstruction lives on *are* the
candidate's clusters. A cloud cluster is kept when ≥ 5 reco points have their
nearest cloud point (within 2 cm) in it.

The check that licenses this is coverage, not the reduction. Over all 94 events:

```
largest shower's fitted points on a KEPT cluster : min 0.9966  median 1.0000
every reco point on a KEPT cluster               : min 0.9989  median 1.0000
```

i.e. the filter essentially never eats charge the scan is about. A relative
threshold (≥ 0.5 % of matched points) was measured as well: it moves the kept
fraction by 0.001 and drops the worst coverage to 0.971, so the flat, explainable
number wins.

**Two implementation notes that are not cosmetic.**

*Filter before decimating.* The candidate cut runs on the full arrays and the
`max points` budget is then spent on what survives. The other order is silently
wrong: the 54 477-point event would go to 25 000, then the candidate's ~2 100
points would be scaled to about a thousand, while the readout said "showing
25 000". The readout now names three numbers, not two — drawn, of candidate, of
total, plus the cluster count — so which reduction bit is always visible.

*scipy is an accelerator, never a requirement.* The nearest-neighbour match is
`cKDTree` when scipy imports (median 7 ms) and a hand-rolled uniform grid hash
when it does not (median 164 ms, worst 951 ms — correct, but a second per click
is not a scan). The fallback is not assumed equivalent: the selftest runs **both
over all 94 events and asserts the kept-cluster sets are identical**, 94/94. A
per-event cache of the parsed cloud and the match result makes changing `max
points` or the colour mode cost 2 ms instead of a reload.

Net effect on the scan loop: event load is **mean 0.184 s, median 0.178,
p90 0.225, max 0.297** — slightly *faster* than round 3's 0.22 s mean, because
the smaller payload and the JSON cache more than pay for the match.

**Reading older labels.** The saved `camera` block now carries `cloud_scope`
(`"neutrino-candidate"` / `"all-clusters"`), `cloud_candidate` and
`cloud_cluster_ids`, because a verdict of "under-clustered" means something
different if four fifths of the charge was filtered out of the view. Records
written before this round have **no `cloud_scope` key at all**, and a reader must
treat that absence as **`"all-clusters"`** — round 3 had no filter and drew the
whole readout. Reading a missing key as the new default would silently relabel
every event scanned before today. At the time of writing that is one record,
`em_labels/emscan-0827/labels-evt64591.json`.

### 12.3 Item 1 — two columns, and the tab that was silently setting the width

The app was one 880-wide strip down the left of the screen. It is now a header
band over `row(3-D view + its controls, everything else)`, default total width
**1980 px**, with a **3-D panel size** selector (620 / 760 / 900 / 1100) for a
bigger screen.

One thing had to be found by measuring the DOM rather than by reading the layout
code: the page came out **2233 px** wide, ~200 px more than the 3-D tab needs.
The cause is that a `Tabs` is as wide as its *widest* panel including the
inactive ones, and the 2-D tab was three 420-wide projections = 1260 px — wider
than the 3-D tab's 1082, so an invisible tab was setting the page width and
adding a horizontal scrollbar. The projections are now stacked 2-over-1 and are
**bigger** as a result (520 × 400 each, was 420 × 330), and the page is 2065 px.

The browser test asserts the two-column claim geometrically — the acceptance
plot's canvas must start to the right of where the 3-D canvas ends — because
"they are in a `row`" is a statement about source code and not about pixels.

### 12.4 Item 3 — one gesture, seven jobs

`a tap in 3-D does` (a `Select`, because seven actions do not fit as radio
buttons and a scanner sets it once and then clicks):

| action | what a tap / box does |
|---|---|
| `select segment(s)` | as round 3 — leaves a selection for the mark buttons |
| `mark IN` / `mark OUT` | marks on the click itself, no trip to a button |
| `toggle IN / OUT / clear` | cycles the same segment on repeated clicks |
| `orbit around it` | re-centres the camera on the clicked point — item 3(a) |
| `fill x / y / z` | as round 3 |
| `make it the pi0 vertex` | item 4's last sentence, in one click |

**Orbiting** sets `cam_c` to the clicked point, which puts it at the projection's
origin, and re-centres the ranges on zero *keeping their current span* so the
zoom survives. `cam_R` is deliberately **not** rewritten: it is the
zoom-independent scale the depth cue normalises by. The known consequence,
documented rather than fixed: orbit far from the bounding-sphere centre and the
depth fading flattens, because the event no longer spans ±R about the new centre.

**Vertices are tappable, and can never be marked.** They are reachable from
`_tap3.renderers` but deliberately *not* from `_box3.renderers`, and they have
their own handler on their own source. The reason is concrete: `mark` keys
`state["marks"]` by segment id, and a vertex swept up by a bulk box carries none
— a `kind`-column-in-`pick_src` design would have put `"-1": "in"` into a saved
label file. Different renderer lists make that impossible rather than unlikely.

Setting the π⁰ vertex from a tap is a reentrancy edge and is ordered for it: the
x/y/z boxes are written first under `_suspend`, *then* the mode radio moves to
`manual`, then the redraws are called explicitly. Flipping the radio first runs
`on_vtx_mode`, which re-reads the boxes — and at that instant they still hold the
previous point.

### 12.5 Item 4 — the halo stack, and the bug the order was hiding

Six halo layers now draw in a fixed order, and **the order is the message**:

```
selection (cyan, 17 px)   what the next mark button will hit
your mark (13 px)         green IN / red OUT
gamma members (11 px)     pi0 slot colours, blue / red
reco members (9 px)       what the CLUSTERING said
the segment itself (2 px)
your mark again (4 px, dashed, ON TOP)
```

Before this round the reco halo was drawn *first* and the mark halo over it, so
**marking a member erased the evidence that it was a member** — precisely the
thing the owner asked to be able to see. Widest underneath keeps the bands
concentric and every state readable at once: yellow inside green = a member you
confirmed, yellow inside red = a member you are removing, green with no yellow =
a non-member you are adding. The dashed repeat on top is a second, redundant
channel for the same distinction, because a thin yellow band inside a thick green
one is easy to miss on a laptop panel. A legend under the controls spells out all
five states with swatches.

Three more things in the same direction:

- **The selection is now drawn** (cyan). Round 3's pick surface is invisible by
  design — Bokeh hit-tests geometry, not paint — so a box-select gave no feedback
  at all until something was already marked. It reads the *same* resolver the
  mark buttons read, so what is drawn and what would be marked cannot disagree.
- **`frame the shower`**, a third framing mode: with it on, picking a shower in
  the table also frames it. Measured on evt 64591 that is **R 300 cm → 32 cm**, a
  9× zoom onto the thing being judged. Not the default — a table click must not
  move the camera under the scanner unless asked — and it falls back to framing
  the reco when nothing is selected.
- **π⁰ mode draws both gammas' members** in their slot colours, because in π⁰
  mode the shower table's selection is usually pointed at the gamma you are
  *about* to assign, and nothing on screen said what slot 1 already held.

`dim what is not in this shower` exists and is **default OFF**. The
`show_all_toggle` comment three screens up is the record of exactly this default
going the wrong way on the owner once already: fading what is not in the shower
fades the segments the scan is deciding about.

### 12.6 Two bugs found on the way, one of them by the browser

**(a) A re-opened label drew none of its own marks.** `load_label` restores
`sel_shower` and `marks` from disk, and it ran *last* in `load()` — after every
draw. So re-opening a labelled event put the marks back into state and then drew
nothing from them: the halos were blank on exactly the events that already had an
answer. `load_label` now runs first, and the shower's table row is re-selected
too. Present since round 1; it took item 4's "I know exactly which part is
included" to make it visible.

**(b) A standing selection made the next gesture a no-op** — found by the real
browser, and *not* findable by the static test. Bokeh fires `selected.indices`
only when the list **changes**, so: box a region in `select`, switch the action to
`mark IN`, box the same region again → nothing happens, because the index list is
identical. The static test could not see it because it assigns indices directly
and had cleared them in between. Changing the action now drops any standing
selection. (The related re-arm inside the marking path — clearing after a mark so
`toggle` can fire on the same segment twice — was designed in from the start and
is asserted both ways.)

### 12.7 What is verified

`selftest_em_display.py`: **30 → 66 → 105** checks. New in this round: the toggle
cycle and its re-arm; mark-on-tap for IN and OUT; orbit re-centring with the zoom
and `cam_R` held and the clicked point landing on the origin; a vertex tap
setting the π⁰ vertex, agreeing with the x/y/z boxes, and *never* reaching
`state["marks"]`; box-select's renderer list being `[pick_src]` alone; the halo
draw order and the dashed-on-top repeat; dim defaulting off and working when on;
a re-opened label drawing its marks; the cloud filter's three-number readout,
its `all clusters` escape, and the budget being spent on the candidate; the
scipy/numpy agreement and the coverage numbers over all 94 events; and
`frame the shower` zooming and falling back.

`selftest_em3d_browser.py`: **21 → 29** checks. It was also **re-anchored**: it
used to find canvases by literal pixel widths (`w > 410 && w < 425`) from the
round-3 layout, and this round both resized the 3-D panel and added a size
selector. A filter that matches nothing does not fail — it passes vacuously —
so every rect filter now reads the width off the model by `name=`. New checks:
the right-hand column really being to the right of the 3-D canvas; the selection
halo existing and matching the selected-segment count; a box marking IN and then
OUT and re-arming in between; `orbit around it` moving `cam_c` *and* reprojecting
the whole cloud; and the readout confirming the drawn cloud is the candidate.

Still open, still a human judgement:

| on | check |
|---|---|
| ncpi0 **evt256587** | with the candidate filter on, its cloud is 30 323 points, the sample's worst. Is a drag still smooth? |
| any event | does the depth fading read as depth, or as fog? |
| any event | is `frame the reco` still the right default now that `frame the shower` exists? |
| any event | is `dim what is not in this shower` useful enough to be worth its default being OFF? |

## 13. Round 5 — correlating the tables, the plot and the display

### 13.1 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin
python em_display/selftest_em_display.py      # 137 checks, 0 failures
python em_display/selftest_em3d_browser.py    # 37 checks, 0 failures
python em_display/selftest_repro.py           # 1567/1567 probe joins
em_display/serve_em_display.sh 5017 emscan-0827
```

No C++, no jsonnet — **no A/B gate owed**, as in rounds 1–4.

### 13.2 What prompted it

The first real label, `em_labels/emscan-0827/labels-evt64591.json`, recorded two
judgements — "the π⁰ reconstruction is right" and "this piece belongs in one of
the EM showers". The first came through cleanly. The second did not, and reading
it back exposed four more gaps the owner then named directly.

**The mark was filed against the wrong shower, and the format could not say so.**
The file reads `em.shower: 78025`, `em.marks: {"60008": "in"}`. Segment 60008 is
a 0.75 cm, 4.10 MeV stub that the reconstruction left as its own shower
(`shower_id 10`, `pio_id -1`). Measured against the two gammas:

| | vs 83044 (298.06 MeV) | vs 78025 (50.74 MeV) |
|---|---|---|
| angle off that shower's fitted axis | **10.8°** | 89.2° |
| angle from the π⁰ vertex | **12.1°** | 68.5° |
| perpendicular offset from the axis | 15.9 cm | 101.9 cm |
| along-axis position | 83.2 cm | 1.4 cm |
| 3-D gap to nearest fitted point | 41.1 cm | 99.0 cm |

83044's own segments span 0 → 46.8 cm along its axis with a maximum transverse
spread of 21.7 cm, so the stub sits **past the reconstructed end, on the axis,
inside the transverse envelope the shower already occupies** — an unabsorbed
tail. `on_gamma` assigns whichever shower the table has selected, so assigning
slot 1 = 83044 then slot 2 = 78025 leaves `sel_shower` on 78025; a mark made
before that step is saved against the wrong one with nothing in the record to
distinguish the two histories.

Two cross-checks in the same file went the other way and are worth recording as
validation of round 4:

- `cloud_candidate: 3774` of 28 658 equals `npts_bundle` for `main_id 17` in
  `nusel-evt64591.tsv` **exactly**. The candidate filter is a purely spatial
  match that knows nothing about flashes, and it reproduced the point set the
  neutrino selection itself used.
- The hand-placed π⁰ vertex is reco vertex **17002** to 0.005 cm, and reproduces
  `kine_pio_dis_1` (4.167 vs 4.169) and `dis_2` (19.190 vs 19.188). Tap-to-set-
  vertex is exact to the click's own `%.1f` rounding.

### 13.3 Marks belong to a shower

`state["marks"]` is now `{shower node: {segment: kind}}`. A mark with no shower
selected is **refused**, not filed; `on_pick` and the mark buttons only report
success when `apply_marks` returns true, because the first cut of this let the
refusal be overwritten by a success-looking line (caught by the browser harness,
which reported 27 segments "marked" with every value reading `-`).

The record writes `em.marks_by_shower` and **no flat map** — a derived copy
beside the authoritative one can disagree with it, and that ambiguity is the bug.
Alongside it, `em.marks_detail` carries per marked segment its `dist`, `angle`,
`tier`, `ellip`, `length`, `pdg`, `cluster_id`, `absorbed_by` and `owner`,
measured against the shower it was marked for, plus that shower's `member_span`.
Those are the quantities `pass3_cone` is cut on; measuring them at save time
means a later fit joins labels without re-deriving axes from the dump.

Round-4 files still load: a flat `marks` is attributed to the `em.shower` the
file named, `state["legacy_marks"]` is set, and the banner says so in red.
**Nothing rewrites `labels-evt64591.json`** — the owner has not said which shower
they meant, and the geometry above is an inference, not their answer.

### 13.4 The acceptance plot was scaled to the gate, not to the comparison

"Why do I not see the points belonging to the existing EM shower cluster" was a
literal and accurate report. Members *were* plotted, orange, with
`show_all_toggle` already default ON — but `acc` is 430 px wide with
`x_range=(0, 220)`, `y_range=(0, 90)`, because tier 3 reaches 200 cm / 5°. For
shower 78025 the members that plotted sat at 3.9 and 17.8 cm: **8 % of the axis
width**, two size-9 dots among 29 others. A distinct glyph would have made two
dots in the corner into slightly clearer dots in the corner; the range was the
problem.

Three changes:

- **Autoscale** to the members plus any marked segment, ×1.3, floors 40 cm / 15°,
  caps at the gate box. For shower 83044 that is 0–110 cm × 0–32° instead of
  0–220 × 0–90. `zoom to this shower` turns it off; segments outside the zoomed
  range are counted in the line underneath.
- **The seed segment was missing entirely** — a real bug. The shower's own start
  segment contains the start point, so `start → closest` is the zero vector,
  `angle_deg` returns `None`, and the `angle is not None` guard dropped it. It is
  the segment every other member is measured against. Now plotted at angle 0;
  shower 83044 went from 16 of 17 members to 17 of 17.
- **A comparison line in words**, which is the part that aggregates over events:
  *"already in shower 83044: 17 segments — distance 0.0–51.6 cm, angle 0.0–24.9°;
  60008 marked IN — 84.4 cm, 10.8°, pass-1 tier 2, absorbed by nothing; angle
  inside the member spread, distance 1.6× the furthest member."*

That last readout is the tuning signal in one line: **pass-1 tier 2 accepts this
segment on distance and angle, and nothing absorbed it.**

### 13.5 The other three asks

**A table click drives the 3-D view.** `on_shower_select` sets
`view_tabs.active = 0`, and `fit_mode` now defaults to `frame the shower` — which
is §12.7's open question answered by the owner asking for exactly this. The
invariant in the round-4 comment survives: the other two modes still do not
re-frame on a table click, only the default moved.

`focus_points` now includes the selected shower's **marks**, so `refit` reaches
what you marked; and marking something outside the current frame says so rather
than putting the halo off-screen. The camera is not moved on a mark — that would
throw away the scanner's zoom mid-judgement.

**Dim whole showers away.** A `MultiChoice` of the event's showers drives the
same alpha column the panels already read (0.05 for excluded, against 0.16 for
"not in this shower" and 0.95 normal) and drops those segments from the candidate
table. Naming a shower is the stronger statement, so an exclusion beats
`dim what is not in this shower`.

**Linked brushing.** The candidate table, the acceptance plot and the 3-D pick
cloud are three views of one segment list; `sync_selection(origin)` rewrites the
other two from the origin rather than unioning with them, so a stale selection in
a panel nobody is looking at cannot leak into what the mark buttons act on.
`pick_src` is written under `_suspend` because `on_pick` is the marking path —
without it, mirroring a table click into the 3-D cloud while `a tap in 3-D does`
is set to `mark IN` would apply a mark nobody asked for. `selected_cand_ids` is
now deduped across all three: with the views synced, the same segment is in all
of them, and the cyan halo was being pushed twice per selection.

**One colour per shower.** `seg_color(i)` keyed on the enumeration index, so two
segments of one shower came out two unrelated hues and the display never said
which pieces were already considered one object. Colour now comes from the
shower's rank in the event's energy-sorted table, with a swatch column in the
shower table as the key and neutral grey for segments no shower claims.
Category20 is ordered as hue *pairs*, so taken raw it gave evt64591's two gammas
`#1f77b4` and `#aec7e8` — two shades of one blue, on the one comparison that must
not be ambiguous. The palette walks the ten dark entries first.

### 13.6 What is verified

| | |
|---|---|
| static | **137/137** (was 105) — per-shower marks on two showers at once, the refusal, the round-4 read path and its banner, the record's `marks_by_shower`/`marks_detail`, exclusion alpha and table filtering, brushing both ways with no doubled halo, one colour per shower and distinct gamma hues, the seed segment at angle 0, the autoscale and its hidden count, the off-frame warning |
| browser | **37/37** (was 29) — the refusal through a real box gesture, a table click switching tabs, brushing table → plot → 3-D through the live socket, dimming fading exactly the excluded segments in the running view, one colour per shower in `seg3_src`, no JS errors |
| repro | **1567/1567** probe joins, unchanged |

Still open, still human judgement:

| on | check |
|---|---|
| ncpi0 **evt256587** | 30 323-point candidate cloud, the sample's worst. Is a drag still smooth? |
| any event | does the depth fading read as depth, or as fog? |
| evt64591 | was the 60008 mark meant for shower 83044 (10.8° off its axis) rather than 78025 (89.2°)? |
| any event | with the zoom anchored on members *and* marks, is the plot still the thing you read when a mark is far out — or is the comparison line enough? |

### 13.7 Round 5b — the two things the first real re-scan exposed

Both came out of the owner re-marking evt64591 on the round-5 build within the
hour, which is the fastest possible feedback and worth recording as such.

**A migrated mark plus a new one is a contradiction, and nothing said so.** The
round-4 file's flat mark was attributed to 78025 on load, exactly as designed and
as the red banner announced. The owner then marked 60008 against 83044 — the
shower the geometry pointed at — but nothing removed the migrated one, so the
saved record claimed the segment for **both**:

| marked against | dist | angle | pass-1 tier | ellip |
|---|---|---|---|---|
| **83044** (298 MeV, 17 seg) | 84.4 cm | **10.8°** | **2** | **14.25** |
| 78025 (50.7 MeV, 3 seg) | 101.6 cm | 89.1° | none | 412.84 |

The migration is still right — the alternative is dropping a real mark on load —
but "belongs to both" is not a judgement anyone can hold. `mark_conflicts()` now
finds segments marked IN against more than one shower and the mark list prints
the competing rows **with the numbers that settle them**, including `ellip`,
which is the code's own tie-break at `:1314-1315`. The same warning fires at the
save. The record is still written either way: it is the scanner's, not the
tool's to veto.

Resolved at 20:25 UTC to `{"83044": {"60008": "in"}}` — one shower, and the one
78025 does not even accept at pass-1. **This is the answer to §13.6's open
question**, from the owner rather than from the geometry: the extra piece belongs
to the 298 MeV gamma.

**A marked segment kept the reconstruction's colour.** Marking a piece into a
shower and leaving it painted as whatever it was before means the display shows
the reco's answer while the record holds a different one. `effective_owner`
now resolves the colour through the marks — IN repaints into the new shower, OUT
drops back to neutral — patched on the `c` column alone, for the same reason
`refresh_dim` patches only `a`: this runs on every tap, and assigning `.data`
re-serialises every polyline in four sources.

One thing worth stating because it surprised the test first: a segment the reco
left as a **one-segment shower of its own** (which is what most orphan stubs are,
60008 included) starts in *its own* colour, not grey. Grey means no shower claims
it at all — like the 186 cm track 17002, `shower_id -1`.

Counts after 5b: static **150**, browser 37, repro 1567/1567.

### 13.8 Round 5c — a PID correction does not move the energy

The owner scanned two more events and asked a question the display could not
answer: *"for the pi0 reconstruction, how is the energy of this muon shower
calculated for the pi0 mass?"*

**The chain, all read in the source.** The panel's mass uses
`shower_energy()` = `showers[].kine_charge`, and

```
E = Σ_p(w_p Q_p)/Σw / recom / fudge × w_value × 1e-6      NeutrinoEnergyReco.cxx:188
```

so E scales as `1/(recom·fudge)`. **Which pair is used is chosen by
`Shower::get_flag_shower()`, not by the PDG** (`NeutrinoEnergyReco.cxx:204`):

```
kShowerTrajectory || kShowerTopology || |pdg| == 11     — on the START SEGMENT
                                    PRShower.cxx:1460-1464 and :1578-1582
```

| object | recom | fudge | 1/(recom·fudge) |
|---|---|---|---|
| track-flagged | 0.70 | 0.95 | 1.504 |
| shower-flagged | 0.50 | 0.80 | 2.500 |
| \|pdg\|==2212 | 0.35 | 0.95 | 3.008 |

Defaults at `NeutrinoPatternBase.h:41-52`; SBND overrides none of them —
`wct-pr-perevt.jsonnet:674-689` documents the formula and leaves the values
alone.

**On evt166870 this is not academic.** Shower 85045 is the object the owner wants
promoted to a gamma: start segment 85045, **pdg 13**, 15.1 cm, and neither shower
flag — so all three disjuncts are false, `kine_charge` was converted with the
**track** pair, and its 38.59 MeV is **1.66× smaller** than the identical
collected charge would give in a shower-flagged object (64.16 MeV). Gamma 1
(87058) *is* shower-flagged (`kShowerTopology` on its start segment), so it
already uses the shower pair.

Consequence for the mass, quoted in the panel:

| | E₁ | E₂ | axis-convention mass |
|---|---|---|---|
| as reconstructed | 173.8 | 38.6 | **116.1 MeV** |
| 85045 promoted to a shower | 173.8 | 64.2 | **149.7 MeV** |

Neither is 135. **Only the track-flagged gamma is promoted** in that line, and
that is a real correction to the first cut of this feature: the mass goes as
√(E₁E₂), so flipping *both* gammas is an identity — one rises by 1.66 and the
other falls by 1.66 and they cancel exactly. Flipping both was what the code did
until the readout was checked against the numbers.

**Two record gaps the same two labels exposed.**

1. `EM_VERDICTS` had `not an EM shower` and no inverse, so "this muon should be a
   gamma" could only live in the free-text note — evt166870's does. Added
   **`is an EM shower (reco PID wrong)`**, **appended**: a label stores the
   verdict string and is read back with `.index()`, so re-ordering would silently
   re-label every record written before the change. `em.reco` and each
   `pio.gammas[]` slot now also carry `particle_id`, `flag_shower`,
   `kine_hypothesis` and the other-hypothesis energy, so a PID verdict is
   checkable later against what the reco actually thought at the time.
2. evt166870 was saved with `vertex_how: "manual"` and `vertex: null` — the mode
   was selected but the x/y/z boxes were empty, so the whole vertex convention
   silently dropped out of the record. `pio_vertex()` now returns a reason and
   the panel prints it in red next to the empty mass.

**What the two labels say** (both read, neither edited):

- **evt169356** — π⁰ vertex is the **main vertex** (`vertex_how: main_vertex`,
  coordinates identical to `main_vertex`). Vertex convention θ 96.93°, mass
  **138.09**, which reproduces the accepted group's own 138.0899; axis convention
  155.91. `confidence: certain`, **but both verdicts are null** — the judgement
  the record exists to hold was not entered.
- **evt166870** — three different pairings are in play and the label picks a
  fourth. `pio_id` groups {10013, 87058} (128.75); `kine_pio_*` names
  87058 + **10074, a proton** (167.35 MeV, mass 329.66) — §5.4's trap live, and
  the reason the manifest's `kine_pio_mass` column reads 329.7 for this event;
  the owner assigned 87058 + 85045. Marks: `{87058: {54042: "in"}}` — segment
  54042 is a 0.38 cm, 2.40 MeV stub, which is a *different* statement from the
  note about 85045.

Counts after 5c: static **160**, browser 37, repro 1567/1567.

### 13.9 Round 5d — the pi0 verdict is retired

The owner asked what a π⁰ verdict is *about*, and the honest answer was that
nothing said: not the UI (`verdict for this pi0`), not the README, not the doc,
not the record. The only evidence was the option list itself — `wrong pairing`,
`wrong start point`, `wrong vertex`, `shower mis-grouped`, `not a pi0` are all
**fault descriptions of a reconstruction**, so the list only cohered as a
judgement on the code, with the scanner's gamma slots as the instrument. But the
verdict named none of the *three* pairings the panel displays (`pio_id`,
`kine_pio_*`, the scanner's own), so on an event where they differ it is
undefined which one it grades.

evt166870 is that event: the accepted group is {10013, 87058} and the owner
assigned {87058, 85045}. A verdict of "pi0 correct" there would have contradicted
the gamma slots in the same record, with nothing to notice it.

The owner's resolution: *"what I started is the reconstruction of the code, and
then I do correction ... you naturally have both information and this is
sufficient."* That is right, and checkable — the record already carries both
sides independently:

| the code | the scanner |
|---|---|
| `pio.reco_groups` — accepted `pio_id` pairings + masses | `pio.gammas` — pair, starts, energies, members, axes |
| `pio.reco_kine` — the whole `kine_pio_*` block | `pio.vertex`, `pio.vertex_how` |
| | `mass_axis_convention`, `mass_vertex_convention` |

The difference between the columns is the judgement, and unlike a verdict string
it is quantitative and aggregates.

**Known loss, stated rather than papered over.** *"There is no π⁰ in this
event"* is not expressible as a correction: empty gamma slots are also what "not
scanned" looks like, and with no gammas assigned no `pio` block is written at
all. No replacement was invented — the note carries it until the owner asks for
one.

**A pre-5d verdict is read and preserved.** `PIO_VERDICTS_LEGACY` still parses
old files, `state["pio_verdict_legacy"]` holds the value, and `on_save` writes it
straight back. Re-saving an event scanned before this round cannot silently
destroy a past judgement (M13 in spirit: a saved verdict is a record of a scan).
Verified by a selftest that plants a `wrong pairing` in a written label, reloads
and re-saves.

**The EM verdict is untouched** and keeps its anchor (`em.shower` names the
shower it grades). It also carries statements corrections cannot make —
`not an EM shower` and, from §13.8, `is an EM shower (reco PID wrong)`.

**A related ambiguity found in the same exchange, NOT fixed.** `vertex_how`
distinguishes whose vertex it is — `manual` is the scanner's, `main_vertex` and
`backproject` are the code's — and a 3-D click always lands as `manual` even when
it snaps to a reconstructed vertex (evt64591: `manual`, coordinates = reco vertex
17002 to 0.005 cm). But `main vertex` is *option 0, the default*, so
`vertex_how: "main_vertex"` cannot be told apart from never touching the control.
evt169356 is in exactly that state. Surfaced to the owner; no field added.

Counts after 5d: static **165**, browser 37, repro 1567/1567.

## 14. Round 6 — twelve events the owner named, and one that does not exist

> **Superseded in part by §15.** The thirteenth event, `18255-259774`, was a
> typo for `18255-269774` — which was in the display all along. The
> investigation below is still correct on its own terms (259774 as written
> really was never reconstructed here) and is what made the typo findable,
> so it is kept rather than rewritten.

### 14.1 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# stage 2 probes for the four additions (fresh arm, M13)
WCT_SHOWER_CONTENT_DEBUG=1 WCT_SHOWER_ABSORB_DEBUG=1 WCT_SHOWER_MERGE_DEBUG=1 \
PR_EXTRA_STAGES=pr_display PR_JOBS=4 \
  ./run_pr_chain_batch.sh work-mcp1k-grp0825 work-em114b-mcp1k data \
      169626 174752 347129 394532
python em_display/prep_em_scan.py --parse-probes work-em114b-mcp1k --no-bee-index \
      --out /home/xqian/tmp/scratch.tsv
python em_display/prep_em_scan.py --bee-build bee/em114b \
      --bee-events 169626,174752,347129,394532
python em_display/selftest_repro.py            # 98/98
python em_display/selftest_em_display.py       # 177
python em_display/selftest_em3d_browser.py     # 40
```

No C++ and no jsonnet are touched, so **no A/B gate is owed** — as in rounds 1-5.

### 14.2 The question that opened the round

> *"I wonder why the display does not have 18255-259774 event?"*

Because the display shows exactly the rows of `em114-manifest.tsv`, and that
file is generated from `pr113-ncpi0.index.txt` (46) plus the `nuecc48` rows of
`pr113-nuecc.index.txt` (48). 259774 is in neither.

The deeper answer is that **it was never reconstructed here at all**:

| probe | result |
|---|---|
| `work*/pr_evt259774` | none, in any arm |
| `$3 == 259774` in every arm's `nusel-events.tsv` | no row |
| `bee/*/*.index.txt` | no entry in any set |

mcp1k is the **first 1000 events** of
`input_files_reco1/data_MCP2025C_reco1_frameshift_first1000ev.root` (staged as
`e0..e999`); mcp2k is a separate 2000-event staging. Event numbers inside a run
are not contiguous, and 259774 falls in a gap in both pools — run 18255 jumps
66615 → 276198 in mcp1k and 105690 → 273559 in mcp2k. Reaching it means staging
that event from upstream MCP2025C: a data step, not a display step, so it is an
ask rather than something done here.

It is nevertheless carried as a **real row** in the adds file. `prep_em_scan.py`
now *names* every event whose dump is missing rather than counting it, so
`18255-259774` prints at every regeneration and becomes a scannable row by itself
on the day it is reconstructed.

### 14.3 What the twelve reachable events needed

Of the thirteen events named, eight were **already in the display** (389538,
235435, 506114, 84229, 37112, 142421, 90055, 256587) and four were reconstructed
but had never been in the pr/113 lists (169626, 174752, 347129, 394532).

Two of the owner's run prefixes differ from what is on disk, and the disk values
are used: **256587 is run 18306**, not 18255, and 84229 is run 18364. Also
recorded as read: `"90055: shower stem got ided as proton256587, track arm got
ided as electron"` is two entries with a missing newline — 90055 (shower stem →
proton) and 256587 (track arm → electron).

The four additions needed all three of dump / probe / cloud:

| | source | result |
|---|---|---|
| dump | `work-mcp1k-prod0825` | already there |
| probe | `work-em114b-mcp1k` (new arm, 4 events, rc=0) | 4 sidecars, 98 total |
| cloud | `bee/em114b/em114b-mcp1k.zip` (built, 1.4 MB) | 3634 points render for 169626 |

`selftest_repro.py` extends to them: **98/98 events identical** to prod0825 on
main_vertex, kine, tagger, showers and segments, and probe membership exact on
1595/1595 showers. So the round-6 arm reproduces production the same way the
round-2 arm did.

### 14.4 The trap: `bee_round` is not `bee_url`

`bee_round` names the local zip the 3-D cloud is read from
(`em3d.bee_zip_path`); `bee_url` needs a server-minted UUID and therefore an
upload. The pre-round-6 `bee_index()` built its map by iterating `*.url` files,
so **a locally-built set was invisible to it** — the four new events would have
kept whatever older *uploaded* set happened to contain them.

They are in `prod0813` (uploaded) and `prod0819`. `'em114b'` sorts before
`'prod0813'`, so the single-string `prefer` would have handed exactly those four
rows a two-epoch-old reconstruction — the same failure §-earlier describes for 78
of 94 events, re-armed by adding a round whose name also begins with `e`.

Two changes, and a check that pins them:

- `prefer` is a **sequence**, `("em114", "em114b")`, last wins.
- a second pass gives *preferred* rounds that have an index and a zip but no
  `.url` the round with an **empty url** — correct cloud, honest blank link.

The check is spatial, because there is no event id inside the zip to compare:
for **every** manifest row, the distance from the dump's fit points to that zip
member's own `track_fit-global` layer.

| binding for evt347129 | zip idx | NN median |
|---|---|---|
| `em114b/em114b-mcp1k` | 2 | **0.0005 cm** |
| `prod0813/mcp1k-prod0813` | 321 | 0.0314 cm |
| `prod0819/mcp1k-prod0819` | 331 | 0.0314 cm |

Threshold 0.01 cm, so the margin is 60×. Note what the older rows are: the *same
event*, correctly indexed, but a different reconstruction epoch — the fit points
moved by ~0.03 cm. A genuinely wrong event would be tens of cm. The check catches
both. Worst median over all 98 rows: **0.00052 cm**.

### 14.5 The owner's note is a question, not an answer

Each added event carries the hint that came with it. It is a **new manifest
column** `scan_note`, appended (never inserted, so a diff of the .tsv does not
show 94 unchanged rows as changed), rendered as a read-only banner, and
deliberately **not** loaded into `note_in`.

`note_in` is the scanner's editable text and is what becomes `label["note"]`.
Had the hint been loaded there, the first save would either have overwritten it
or recorded it as though the scanner had typed it — and a later reader could not
tell the question from the answer. Pinned by three checks: the hint is on screen,
it is not in `note_in`, and a save does not put it in the record.

### 14.6 Byte-identity of the scan already in progress

The manifest is regenerated, and a scan is live against it, so the 94
pre-existing rows were diffed field by field against the committed file:

```
old rows=94  new rows=98
column delta: ['scan_note']   columns removed: []
ADDED: ['169626', '174752', '347129', '394532']   DROPPED: []
pre-existing rows changed on an OLD column: 0
```

Zero. The labels already saved under `emscan-0827` are unaffected; nothing under
`em_labels/` was read for anything but display, and nothing was written to it.

### 14.7 Left open

- **The upload.** `bee/em114b/em114b-mcp1k.zip` is built and not uploaded —
  outward-facing, CLAUDE.md §5.6. Until then those four rows have no external
  Bee link (the 3-D view works regardless).
- ~~**259774** needs staging from upstream MCP2025C.~~ **Closed in §15: it was a
  typo for 269774, already in the display. No staging needed.** Original note:
  Worth knowing *where the
  owner saw it*: a truth list or someone else's Bee set names which pool to
  stage from.
- The `vertex_how == "main_vertex"` default ambiguity from §13.9 is still open.

Two latent items, noticed and deliberately not fixed in this round:

- **`--parse-probes` with no argument globs `work-em114-*`**, which does *not*
  match `work-em114b-mcp1k`. Nothing is broken today — the README documents the
  explicit form, and that is what was run — but the no-arg form silently covers
  only the original arm. `work-em114*` would cover both.
- **`bee_build`'s arm resolution is now stricter than it was**: it requires the
  candidate root to contain `pr_evt<ID>` for *every* event in the arm, where it
  used to accept any directory that existed. That is what routes round 6's four
  events to `work-em114b-mcp1k` instead of the em114 arm that lacks them. The
  consequence to know about: a future *full* `--bee-build bee/em114` over the
  98-event sample would find mcp1k's 14 events in neither `work-em114-mcp1k`
  (10) nor `work-em114b-mcp1k` (4), and fall through to prod0825 — the arm whose
  truncated log the §-earlier gotcha is about. Not exercised by anything run
  here; build per-round sets, or merge the arms first.

Counts after round 6: static **177**, browser **40**, repro **98/98**
(1595/1595 showers). The selftests write into `em_labels/selftest114/`; the
owner's `emscan-0827` tag was read and never written (M13).


---

## 15. Round 7 — the typo, and "have I already scanned this one?"

Two small requests, one of which closed round 6's open item.

### 15.1 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
python em_display/prep_em_scan.py               # 98 rows, 13 notes
python em_display/selftest_em_display.py        # 191
python em_display/selftest_em3d_browser.py      # 42
python em_display/selftest_repro.py             # 98/98
```

No C++ and no jsonnet are touched, so **no A/B gate is owed** — as in rounds 1-6.

### 15.2 `259774` was `269774`

> *"for this event 259774, is there a similar event number? I must have a typo"*

Yes, and it is unambiguous. Over all **3089** distinct `(run, event)` pairs
reconstructed anywhere in this tree, exactly **one** string is within one edit or
one adjacent transposition of `259774`:

```
run 18255  evt 269774   1 edit   ALREADY IN THE DISPLAY (sample nuecc48)
```

It corroborates three independent ways:

| signal | value |
|---|---|
| run the owner wrote | 18255 — matches, and the sample is nuecc48 |
| the note, *"multiple pi0"* | `n_pio_groups = 2`, `n_pio_showers = 4` — and only **7 of 98** scan events have ≥ 2 π⁰ groups at all |
| the event named beside it | 389538 is also nuecc48 run 18255; in that index 269774 sits between 268784 and 271851 |

The third is the one that makes it a *typo* rather than a coincidence: the owner
was plausibly reading down a nuecc48 index and slipped one digit.

269774 was already a scannable row — it simply had no note. So the fix is one
cell. `pr114-owner-adds.index.txt` gains a note-only row (the existing 8-row
pattern) and the phantom `259774` row is **commented out, not deleted**: it
contributed no manifest row, so demoting it changes nothing in the `.tsv`, while
leaving it live would make `prep_em_scan.py` report a missing event forever. The
provenance paragraph stays, reframed — the fact that 259774 could not have been
scanned by anyone here is itself evidence for the typo reading.

**Regeneration gate** — field-by-field over all 98 rows, both directions:

```
columns identical: True     rows 98 / 98     added: none     dropped: none
CELLS CHANGED: 1
  evt 269774  col scan_note   '' -> 'multiple pi0'
```

### 15.3 The scan-status chip

> *"On the event display, on the top, can you also add which one that I already
> scanned?"* … *"I do not need the event list there, but at the top just say
> whether I have saved result or not."*

The display already counted `3/98 events labelled` — but that counter lives in
`info`, which is **in the right-hand column**, and a count cannot answer "is
*this* one done". New `scan_status` Div, placed directly under the header row
and above the Bee banner:

- green — **✔ you have already scanned this event** — a saved result exists in
  tag `<tag>`, saved `<utc>`
- grey — **not scanned yet** — no saved result for this event in tag `<tag>`

Two decisions worth recording, because both are failure modes rather than
preferences:

1. **Read from the filesystem, not from `state["saved"]`, and wake it on a
   timer.** `state["saved"]` is a load-time snapshot, so with two tabs open on
   the same tag — likely, since a restart tells the owner to reload — a
   snapshot-driven chip keeps saying "not scanned yet" after the *other* tab
   saved. `state["saved"]` supplies only the timestamp, dropped when it is not
   ours to quote.

   The disk read alone was **not enough**, and this is worth recording because
   the first version shipped with the gap: `refresh_scan_status` fires only from
   `refresh_info`, i.e. on load, save and touch. Nothing wakes it while the
   scanner *sits* on one event, so the other tab's save still would not appear
   until they navigated away and back — the very case the disk read exists for.
   A `curdoc().add_periodic_callback(refresh_scan_status, 5000)` closes it: one
   `stat` every 5 s, and re-assigning an unchanged `Div.text` syncs nothing. The
   test that covers it creates the label file from outside the session and
   asserts the chip flips, in a throwaway tag so the suite stays re-runnable.
2. **Disk state only.** Unsaved-edit state is already rendered by `refresh_info`
   as `[unsaved]`. Duplicating it in the chip would give two indicators that can
   disagree. A check pins the separation.

The event dropdown was **not** ticked with ✓ marks: the owner explicitly narrowed
away from a per-event list. It remains an easy add.

### 15.4 Verification

| suite | result |
|---|---|
| `selftest_em_display.py` | **191/191**, was 177 |
| `selftest_em3d_browser.py` | **42/42**, was 40 |
| `selftest_repro.py` | **98/98** identical, 1595/1595 showers |
| manifest gate | 1 cell changed (§15.2) |

The owner-note count check was **de-magic-numbered** — it read `== 12` and failed
on 13. It now parses `pr114-owner-adds.index.txt` and asserts every note in the
index reaches the manifest *and* that no manifest row invents a note the index
does not have, so adding an event is a data change while a note that silently
fails to arrive is still a failure. That is the same class of bug as round 6's
hardcoded `len(V.LABELS) == 94`.

Live proof on the owner's own instance (port 5017, tag `emscan-0827`), not just
on a throwaway server:

```
first event shown : evt64591        # the owner saved this one at 13:25
chip text         : ...#2e7d32...&#10004; you have already scanned this event

== evt269774 ==                     # the typo fix, on the owner's instance
   chip : not scanned yet -- no saved result for this event in tag emscan-0827.
   note : what you asked to look at here: multiple pi0 -- your note from the ...
JS errors: none
```

The browser check walks **shadow roots** — Bokeh 3 renders every widget in its
own shadow root, so a plain `document.querySelectorAll('div')` cannot see the
text and the first version of the check failed with `None`. Measured position:
`top: 73 px, height: 14` — genuinely at the top of the page.

`em_labels/emscan-0827/` was read throughout and never written: all three of the
owner's labels still carry their original mtimes (13:25, 13:37, 13:49). The
selftests write to `em_labels/selftest114/` and `em3dbrowsertest`.

### 15.5 Left open

- The Bee **upload** for round 6's four events is still not done —
  `bee/em114b/em114b-mcp1k.zip` is built but uploading is outward-facing
  (CLAUDE.md §5.6). Those four rows show a working 3-D cloud and a blank
  external link; the banner says so.
- Pre-existing, not fixed here (noticed while working): `load_label` does a bare
  `json.load` at :2313 while the save path guards `ValueError`. A truncated label
  file would raise into the session callback rather than being reported.

---

## 16. Round 8 — the scanner's own start vertex, direction, and event topology

Three requests from the live scan on evt169626.

### 16.1 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
python em_display/selftest_em_display.py        # 217
python em_display/selftest_em3d_browser.py      # 53
python em_display/selftest_repro.py             # 98/98
```

No C++ and no jsonnet are touched, so **no A/B gate is owed** — as in rounds 1-7b.

### 16.2 What was asked

> *"one of the EM shower, I need to change the start vertex, I do not have this
> capability in the display"* … *"with vertex changed, very likely I also need to
> be able to define the direction, by clicking another end point"* …
> *"it is an no-vertex pi0 NCpi0, can we also add a label for that?"*

### 16.3 The seam, and the invariant that governs it

`shower_start(node)` and `shower_axis(node)` are the *only* two inputs the
pass-1 gate has. The candidate table (`:1881`), `seg_vs_shower` (`:2403`) and
`mark_metrics` (`:2444`) each call exactly those two and nothing else, so an
override placed in them propagates to the table, the acceptance plot and the
saved metrics with no other code changed.

The invariant, and it is the thing that would have gone wrong quietly:

> **The start and the axis must move together.** The probe's `dir15` is
> `shower_cal_dir_3vector(shower, start, 15 cm)` — anchored at the
> *reconstruction's* start. Had `shower_start` honoured an override while
> `shower_axis` kept returning `dir15`, `seg_vs_shower` would have taken the
> angle between a direction anchored at the old start and a displacement
> measured from the new one. That is not a physical quantity, and it looks
> entirely plausible in the acceptance plot and in the saved record.

So an overridden start invalidates the probe value. Two ways to replace it:

| the scanner… | axis | `axis_source` |
|---|---|---|
| moves the start only | `shower_cal_dir_3vector(members, new_start, 15)` — the same formula, at the new point | `python@start_override` |
| also clicks a second point | `norm(p2 − start)`, exact by construction | `manual@override` |

`axis_source` stops saying `"probe"` in both cases: em_geom:161's own docstring
records that the Python mirror is not bit-exact (`shower_ordered_edges` vs
`fill_sets` membership). The recompute is memoised on `(node, start)` — without
it, `mark_metrics` would walk every member point once per segment.

The recompute uses the **reco's** member set, not marks-included: moving two
inputs at once would make the before/after uninterpretable.

### 16.4 Round 8b — the correction has to reach the π⁰ mass too

Reported from the live scan within the hour:

> *"when I clicked the pi0 tab, and then click the EM shower again, it seems that
> it goes back to the original start vertex … I am also confused which one was
> used to do the calculation of the pi0 mass."*

Both halves were real, and the second was the serious one.

Round 8 put the override in the `slot is None` branch only, on the reasoning
that every π⁰ caller passes a slot. That kept the gamma slots off the EM path —
but `shower_axis` takes **no slot at all**, so it had *already* been using the
corrected axis. The result inside one saved record:

| | geometry used |
|---|---|
| `mass_axis_convention` | the scanner's corrected axis |
| `mass_vertex_convention` | the reconstruction's start |

Two masses, two different geometries, nothing on screen saying so. An EM start
correction is a correction to the **shower**, not to EM mode, so the precedence
is now, most specific first:

```
gstart[slot]    a start set for THIS gamma slot, in pi0 mode
em_start[node]  the shower's corrected start          <- was missing
reco start      the dump's own
```

`start_source(node, slot)` names which one was returned, and the π⁰ panel prints
it per gamma — *"γ1: your corrected start from EM mode (…), axis
python@start_override"* — so the question "which one was used" is answered on
screen rather than inferred from a number that moved.

The apparent "revert" was a second, smaller bug: `on_mode` did not call
`refresh_emstart`, so the readout went stale across a tab switch. The state
itself always survived.

### 16.5 The record

`em.reco` keeps the **reconstruction's** answer, unchanged — `shower_axis` gained
`use_override=False` for exactly this, because filing a hand-aimed axis inside
`reco` would let a later reader attribute the scanner's judgement to the
reconstruction. What the gate actually used is named separately:

```
em.reco.axis / axis_branch / axis_source     the reco's, always
em.axis_used / _branch / _source             what the gate used
em.reco_start, em.start_used                 both points, so the move is checkable
em.reco_start_vertex_id
em.start_override_by_shower                  keyed BY SHOWER, not flat
em.start_override_vertex_id_by_shower        which reconstructed vertex, when it was one
em.dir_point_by_shower
pio.gammas.<slot>.start_source               per gamma: which of the three
event_flags                                  event-level, beside em and pio
```

Keyed by shower because `mark_metrics` runs for *every* marked shower, not only
the selected one — a flat pair would be written from the selected shower and read
back against another. Same reasoning as the round-5 note on `marks_by_shower`.

### 16.6 Gestures

`a tap in 3-D does` gained **"make it this shower's START"** and **"aim this
shower's AXIS through it"**. Both work on the fit-point surface and on the
**vertex** surface — the latter is the gesture the request actually named
("change the start vertex"), and a tap that lands on a reconstructed vertex
records *which one*. Buttons cover snap-to-nearest-vertex, snap-to-nearest-fit-
point, typed x/y/z, and reset. The start in use, the reco start it replaced, and
the aim point are all drawn, under a new `emstart` layer key — a renderer
registered under a key `apply_layers` cannot name is invisible forever.

Marks made *before* a start moves are warned about out loud: `mark_metrics`
recomputes tier, angle and distance at save time, so those marks would otherwise
get a record whose geometry the scanner never saw.

### 16.7 The event-level flag

`event_flags` is a list at the **root** of the record, beside `em` and `pio`, so a
later pass selecting "the no-vertex NCπ⁰ events" reads one key and never opens a
shower block. Vocabulary today is one entry, `no_vertex_ncpi0`; adding the next
class is a one-line change to `EVENT_FLAGS`, with no schema change to labels
already on disk. `CheckboxGroup` labels are plain text — the first version
shipped HTML entities and they rendered literally.

### 16.8 Verification

| suite | |
|---|---|
| `selftest_em_display.py` | **217/217**, was 191 |
| `selftest_em3d_browser.py` | **53/53**, was 42 |
| `selftest_repro.py` | 98/98 identical, 1595/1595 showers |

The round-8 check *"the pi0 gamma starts are NOT touched by an EM override"* was
**replaced, not deleted**: it asserted the behaviour the owner reported as a bug
the same day. Live on 5017, clicking vertex 13000:

```
start & axis - reco start (-14.1, 118.7, 465.1) | yours (-19.2, 95.7, 475.4)
             = vertex 13000, 25.7 cm away
```

### 16.9 Left open

- The two-point aim uses the start **in use**; if the start later moves, the
  aim point is kept and the direction re-derives from the new start. That is the
  intended reading of "aim through this point", but it is a choice.
- `load_label` still does a bare `json.load` while the save path guards
  `ValueError` (carried from §15.5, not fixed here).

---

## 17. Round 9 — which recombination pair a gamma's charge is converted with

> *"for event 166870, when we calculate the pi0 mass, the energy of the EM shower
> should use the charge inferred one instead of the kinetic energy."* … *"This is
> about the display."*

### 17.1 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
python em_display/selftest_em_display.py        # 224
python em_display/selftest_em3d_browser.py      # 53
python em_display/selftest_repro.py             # 98/98
```

No C++ and no jsonnet are touched, so **no A/B gate is owed** — as in rounds 1-8.

### 17.2 What was actually wrong

`shower_energy()` already returned `kine_charge`, so the first reading — "the
display uses a range/kinetic energy" — is not what was happening. The owner's own
saved record named the real case:

> note: *"85045 should be an EM shower, part of pi0"*, with 85045 in gamma slot 2.

`kine_charge` is `charge / (recom × fudge)`, and **which pair** was used is
decided by `Shower::get_flag_shower()` — a property of the reconstruction, not of
the slot a scanner later drops the object into:

| shower | pdg | `flag_shower` | converted as | `kine_charge` |
|---|---|---|---|---|
| 87058 | 11 | True | shower (0.50, 0.80) | 173.8 MeV |
| **85045** | **13** | **False** | **track (0.70, 0.95)** | **38.6 MeV** |

So slot 2 was carrying *a track's* energy. Re-converted under the shower pair the
same collected charge is **64.2 MeV**, and the π⁰ mass moves:

```
axis convention    116.1 -> 149.7 MeV
vertex convention  116.3 -> 149.9 MeV      (pi0 rest mass 134.98)
```

Both conventions share `E1·E2`, so both move together.

### 17.3 The control, and why it defaults OFF

A `Select` per gamma slot: **as reconstructed** (default) / **as EM shower
(charge-inferred)**.

The default is the reconstruction's number, and that is the important decision.
Defaulting to the EM hypothesis would have made the saved evt166870 record
*display* 149.7 where 116.1 was written — a scan record reading differently than
when it was saved, with no diff and no flag. That is the §1 failure mode exactly.
A record with no `energy_hypothesis` key (everything saved before this round)
restores to `as reconstructed`, and a check pins it.

The switch reaches the π⁰ arithmetic and nothing else: all four `shower_energy`
call sites are on the π⁰ path, so `em.reco.kine_charge`, the shower table and the
manifest's `em_max` are untouched by design.

The re-conversion **reuses `kine_hypothesis(node)[2]`** rather than computing the
ratio again — for anything the reco did not flag as a shower, "the other pair" IS
the shower pair, so that value is already the charge-inferred EM energy. One
implementation, nothing to drift.

Three cases are handled distinctly, because they are different statements:

- `flag_shower` **False** → the warning names the reco's label and offers the
  number the switch would give;
- `flag_shower` **True** → already charge-inferred; the switch is a no-op and
  says so, rather than silently double-converting;
- `shower_is_em()` returns **None** (start segment not in the dump) → *unknown*,
  not "track". Nothing is re-converted and the panel says why.

### 17.4 The record

```
pio.gammas.<slot>.energy                    what the mass used
pio.gammas.<slot>.energy_hypothesis         as_reconstructed | as_em_shower
pio.gammas.<slot>.energy_as_reconstructed   the reco's own number
pio.gammas.<slot>.energy_other_hypothesis   (pre-existing)
```

Both numbers on every record, so a later reader can recompute either mass without
going back to the dump.

### 17.5 Noticed

The panel already *said* the answer before this round — *"if every track-flagged
gamma here is really an EM shower, the axis-convention mass becomes 149.7 MeV"* —
it just had no way to apply it. That line now drops out when the switch is on,
where it would be restating the number in use.

### 17.6 Left open

- The switch is per gamma slot and per event, and is not remembered across
  events; an object the scanner repeatedly judges mis-flagged must be switched
  each time.
- Nothing ties the switch to the EM-mode verdict. That coupling was considered
  and rejected: verdicts are per-shower in EM mode while gammas are assigned in
  π⁰ mode, so an EM-mode click would silently move a π⁰ mass.

## 18. Round 10 — a whole shower into another one, in one gesture

> *"For evt 179242, I want the EM shower in 71022 be completely included in
> shower 4002, how to achieve that in the display? This is better than click one
> segment after another?"* … *"This above is for EM clustering"*

### 18.1 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
python em_display/selftest_em_display.py        # 248
python em_display/selftest_em3d_browser.py      # 58
python em_display/selftest_repro.py             # 98/98
```

The case: **evt172942**, shower **71022** (10 segments over 8 clusters, 96.9 MeV)
into shower **4002** (6 segments, 368.8 MeV).

No C++ and no jsonnet are touched, so **no A/B gate is owed** — as in rounds 1-9.

### 18.2 The event number was 172942, and it was resolved by fingerprint

179242 is not in the scan set, and it is not reconstructed anywhere in this tree.
It was not guessed at from the digits: the two shower ids in the request are a
fingerprint, and **shower 4002 and shower 71022 occur together in exactly one of
the 98 scan events** — 172942. 179242 is that number with one adjacent
transposition (`…2942` → `…9242`). The uniqueness is pinned by a test, so a
future manifest change that breaks it fails rather than quietly invalidating this
section.

This is the second typo in the scan (§15 resolved 259774 → 269774). The
difference is worth noting: §15 needed edit distance over 3089 pairs plus three
corroborating facts, because the request carried nothing but the number. Here the
request carried *data* — two shower ids — and data beats string distance.

### 18.3 The answer to the question as asked: yes, and here it is

Marking segment by segment was the only path. It is ten clicks for this case,
each of which can land on the wrong row, and the set is never visible before it
is committed. The new control:

```
shower table      -> 4002                          (the shower being scanned)
whole shower      -> 71022  (96.9 MeV, 10 seg)
                     [select all its segments]     -> 10 lit in cyan
                     [mark IN]                     -> 10 filed against 4002
```

Three clicks, and — the part that matters more than the click count — the
membership comes from the **probe**, not from the scanner's aim. A fragment that
is hard to hit, off-screen, or hidden behind another segment cannot be the one
that gets missed.

### 18.4 It selects; it does not mark

A direct *mark all IN* button would have been one click shorter. It was rejected:

- the selection is what the cyan halo draws and what `selected_cand_ids()` hands
  the mark buttons, so selecting puts the exact set about to be marked on screen
  **before** it is committed;
- all four mark buttons then work on it unchanged — **`mark OUT` on a whole
  shower** says "this is not part of the one I am scanning" in one gesture, which
  was free;
- no new marking path exists, so nothing new can misfile a mark against the
  wrong shower (the round-5 bug).

`add to selection` accumulates, so several fragments of one EM shower go in
together.

The shower being scanned is excluded from its own menu: every one of its segments
is already a member, so a bulk `mark IN` there changes nothing and still writes an
entry per segment into the record.

### 18.5 The failure this design exists to avoid

`fill_cand_table` drops a segment that is `show members too`-hidden or that
belongs to a **dimmed-away** shower. Dimming is how a scanner reduces clutter
before judging, so "dim 71022 away, then merge it in" is a realistic order of
operations — and a selection built by scanning the candidate table's rows would
have returned **zero** segments and marked nothing, with no complaint.

`select_segments` therefore takes its ids from `members_of(node)` and writes them
into all three views, including `pick_src` — which `draw_segments` fills from
every segment, unfiltered. The status line reports the count that will actually
be marked (`selected 10 of 10`), resolved through the same
`selected_cand_ids()` the mark buttons call, never the list that was asked for.
Two honest cases come out of that:

| case | what happens | what is said |
|---|---|---|
| shower dimmed away | all 10 selected, mark lands | amber: *"not listed in the candidate table … they ARE selected"* |
| segment with < 2 fitted points | drawn in no view, unreachable | red, named |

The dimmed case is a test, not a claim.

### 18.6 A real bug the browser found, that Python could not

The first browser run failed all five new checks: the menu still listed the
scanned shower and the app reported *"pick the shower you are scanning first"*
after a row had been clicked.

Cause, and it is not in the new code: **`shower_src.selected` survives an event
switch.** `load()` restored the highlight when the event had a saved label but
never *cleared* it otherwise, so an unlabelled event opened with row *k* of the
previous event still highlighted while `state["sel_shower"]` was `None` behind
it — and because Bokeh syncs a property only when it **changes**, clicking that
very row did nothing at all. The scanner had to click some other row first and
come back.

Fixed by clearing unconditionally before restoring. The Python test cannot see
this class of bug — it assigns `indices` directly and had cleared them — which is
the same reason round 4 needed the browser for the tap-action re-arm. The
invariant is now pinned on both sides: Python asserts an event switch leaves the
highlight empty, the browser asserts the click works.

The *restore* half was checked separately, because the fix touches a path every
one of the 98 events takes: all five labels in the owner's live `emscan-0827` tag
re-open with the right shower row highlighted (`87058`, `67030`, `22034`,
`97197`, `83044`), read-only, and a labelled re-open is pinned in the test tag.

### 18.7 What is not claimed

Whether the candidate table's own rows support shift-click range selection was
**not** established — a draft README asserted that they do not, the probe meant
to settle it was inconclusive, and the claim was removed rather than shipped. The
button reads membership directly, which makes the question moot for this task.

### 18.8 Left open

- The bulk control operates on **showers**. Merging by cluster id — "everything
  in cluster 54" — is not offered; no case has asked for it.
- Merging a shower in creates **no** conflict by itself: membership is not a
  mark, and `mark_conflicts()` fires only on a segment marked IN against two
  showers. But a scanner who had already marked those segments against 71022
  itself can now produce ten conflicts in one gesture instead of one at a time.
  They are still caught in the marks list and at save; nothing warns earlier.

## 19. Round 11 — more than one π⁰ per event

> *"In evt 281485, I have multiple pi0, so I need to select multiple pi0 mass to
> store, different combination of the gamma, how to achieve that in the
> display?"*

### 19.1 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
python em_display/selftest_em_display.py        # 270
python em_display/selftest_em3d_browser.py      # 63
python em_display/selftest_repro.py             # 98/98
```

The case: **evt281485** — 48 segments, 20 showers, **19 EM**, and the
reconstruction groups exactly **one** π⁰, from showers 15036 + 90104. The
scanner reads more than one. Seven showers carry more than 25 MeV, so the
pairing is a judgement, not a lookup.

No C++ and no jsonnet are touched, so **no A/B gate is owed** — as in rounds 1-10.

### 19.2 What was missing

Two gamma slots hold one pairing, and the record had one `pio.gammas` /
`mass_axis_convention` / `mass_vertex_convention`. A second π⁰ could only be
recorded by overwriting the first, and an *alternative* pairing of the same π⁰
could not be recorded at all — which is the more common need, since the question
being scanned is usually "which two of these seven".

### 19.3 A stored pairing is frozen numbers

`store this pairing` snapshots the live pairing: both gammas with their resolved
start, axis, `start_source`, energy, `energy_hypothesis` and
`energy_as_reconstructed`, plus the vertex, both θ and both masses.

Frozen **numbers**, not a reference to the slots, and this is the load-bearing
decision. Every input a mass is built from stays editable afterwards —
`state["em_start"]` in particular is keyed by *shower* and is edited in **EM**
mode — so a candidate that merely named its showers would be silently re-priced
by a start correction made later, in a different panel. That is the round-8b
failure class exactly. Same discipline as `marks_detail`, measured at save time.

On evt281485:

| # | γ1 | γ2 | E1 | E2 | θ axis | m axis | m vertex |
|---|---|---|---|---|---|---|---|
| 1 | 15036 | 87078 | 164.7 | 82.0 | 141.9° | 219.7 | 208.4 |
| 2 | 84070 | 91112 | 73.4 | 68.7 | 69.7° | **81.2** | 93.7 |
| 3 | 15036 | 88090 | 164.7 | 65.9 | 84.7° | **140.4** | 162.0 |

(π⁰ rest mass 134.98. The `note` column flags which masses fall in the code's own
accept window.)

### 19.4 One builder, not two

`gamma_record(slot)` and `pio_pairing()` were lifted out of `on_save` **unchanged**
so that a stored candidate and the record's top-level block are built by the same
code. Two builders would drift, and the fields that would drift first are the
ones that must not: round 9's `energy_hypothesis` and `energy_as_reconstructed`
are per **pairing**, not per event, so a candidate that re-converted a
track-flagged gamma has to be distinguishable from one that did not.

The extraction was **proved** behaviour-preserving rather than asserted: the
record produced by the pre-change viewer (`git show HEAD:…`) and by the new one
was compared on three different pairing shapes — two gammas with an EM-mode start
correction, an energy-hypothesis switch and a manual vertex; a single filled slot
with the main vertex; two gammas with a back-projected vertex. The only
difference across all three is the intended new key:

```
37a38
>    "candidates": [],
215a217
>    "candidates": [],
412a415
>    "candidates": [],
```

### 19.5 Load-back, and the one honest asymmetry

`load into the slots` restores the gammas, the energy hypotheses and the vertex
mode, and pins the start through **`gstart`** — the slot-scoped override, which
beats `em_start` in `shower_start`'s precedence — so loading candidate 2 cannot
move candidate 1's geometry.

It pins the start only where the live chain would not already produce it.
Pinning unconditionally would make every loaded candidate report
`start_source = gamma_slot_override` for a start that came from the
reconstruction, and the record would carry that false provenance forever.

`shower_axis` reads the per-shower `em_start` and there is no per-slot equivalent,
so the **axis**-convention angle cannot be pinned. An EM-mode start correction
made after storing therefore moves the axis mass and not the vertex mass. The
load reports it instead of absorbing it:

> ⚠ what is on screen is **not** what was stored — axis-convention mass
> 81.2 → 47.6 MeV. The stored candidate is unchanged…

### 19.6 What is deliberately not automatic

- **Nothing is auto-added at save.** If the slots hold a pairing that is not in
  the list, the panel and the save note say so; Save still records it as the
  record's top-level `pio.gammas`, so nothing is lost. Appending to a curated
  list on the scanner's behalf, using a dedup key the tool invented, is the worse
  failure — it either drops a real candidate or duplicates one.
- **A shower in two candidates is not flagged as an error.** It means they are
  alternative pairings of the same gamma. The panel says which reading applies —
  two real π⁰ need four distinct showers — rather than leaving it to be inferred
  from the ids.
- **`clear gammas` clears the slots only.** The stored pairings are the record.
- The dedup key is not the shower pair: the same two gammas under a different
  vertex convention, energy hypothesis or start are a different mass, and storing
  both is what the list is for. An exactly identical re-store is refused **by
  candidate number**.

### 19.7 The record

```
pio.candidates            [ … ]  ALWAYS written, empty list included
pio.candidates[i]         gammas{1,2}, vertex, vertex_how, backproject,
                          mass_axis_convention, theta_axis_convention,
                          mass_vertex_convention, theta_vertex_convention,
                          stored_utc
pio.gammas / mass_*       unchanged: the pairing in the slots at save time
```

Always written so a reader can tell *"this scanner stored no alternatives"* from
*"this record predates the list"* — an absent key says neither. A record saved
before round 11 has no key, and re-opens showing exactly the one pairing it
always did; a test pins that, as in round 9.

The pi0 block is now also built when the slots are **empty** but candidates
exist. Without that clause, storing two pairings and pressing `clear gammas`
before Save would have dropped both.

### 19.8 Left open

- Candidates are per event and there is no way to mark one as *the* answer. The
  list is a set of measurements, not a ranking; if a later pass needs "the
  scanner's chosen pairing", that is a new field, not a re-reading of this one.
- `gstart3_src` gained a `name=` so the browser test could read the gamma markers
  back. Inert at runtime; noted because it is the only line in this round that
  touches a round-3 model.

## 20. Round 12 — the marks reach the π⁰ energy

> *"for event 1409534, I have added the cluster 27015 to cluster 69032, but when
> I calculate the pi0 mass, it still did not account the energy of the 27015"*

### 20.1 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
python em_display/selftest_em_display.py        # 287
python em_display/selftest_em3d_browser.py      # 66
python em_display/selftest_repro.py             # 98/98
```

The case: **evt409634**, shower **27015** (10 segments, 105.05 MeV) merged into
shower **69032** (2 segments, 39.06 MeV).

No C++ and no jsonnet are touched, so **no A/B gate is owed** — as in rounds 1-11.

### 20.2 The event, and what "cluster 27015" is

Fingerprint again, and this one leans entirely on the objects rather than the
digits: evt409634 is the only event of the 98 that holds **both** shower 69032
and segment 27015. The number as written, 1409534, is seven digits where every
event id here is six.

27015 is a **shower** — 10 segments over 8 clusters, 105.05 MeV — not a lone
segment. So the action was round 10's whole-shower merge, and:

> **round 10 created this gap.** It made "move this shower into that one" one
> gesture, and the energy did not follow.

### 20.3 What was wrong

`kine_charge` is the reconstruction's energy over the segments the
*reconstruction* put in the shower (`NeutrinoShowerClustering.cxx:3771`). A mark
changes the membership and nothing else, so the mass stayed built on 39.06 MeV.

| | E1 | m axis | m vertex |
|---|---|---|---|
| reco membership (before, and still the default) | 39.1 | 41.0 | 83.4 |
| with the marks counted | **144.1** | **78.8** | **160.3** |

The angles do not move. A mark changes which charge belongs to the shower, not
where it points — and the panel says so, because a reader seeing both masses move
would reasonably assume the geometry changed too.

### 20.4 Why this is arithmetic and not an estimate

The probe sidecar carries a per-segment `E_est`, and it is an **exact
decomposition** of `kine_charge`: shower 69032's two members are
29.498 + 9.560 = **39.058**, which is its `kine_charge` to the last digit. So
adding a marked segment's share operates on the same number the C++ mass formula
reads.

There is deliberately **no fallback** for the 3 % of segments (177 of 5830) that
no shower owns and that therefore have no `E_est`. `E_est/dQ` is not constant
between showers — 6.05e-5 against 4.99e-5 MeV per electron on evt409634 alone —
so a dQ-derived number would be a different quantity in the same units. Those
segments are named in red and counted **neither way**.

### 20.5 The default was decided by the data, not by taste

**Six of the twelve records already on disk carry marks on a gamma's shower.** A
default of "include them" would have silently re-priced six masses the scanner
had already judged, with no diff and no flag. So the control defaults to the
reconstruction's membership, the panel states out loud what is being left out and
what the mass would become, and both numbers go into the record. Same shape as
round 9, for the same reason.

What those six become, computed read-only against the live `emscan-0827` tag —
the `m saved` column is reproduced exactly by the default, which is itself the
check that the OFF path is untouched:

| event | marks on a gamma's shower | m axis | → | m vertex | → |
|---|---|---|---|---|---|
| 166870 | γ1 87058 +2.4 | 116.1 | 116.9 | – | – |
| 169626 | γ1 53069 +23.6 | 144.6 | 147.7 | 145.4 | 148.6 |
| 269774 | γ1 13237 **−77.9**, γ2 97197 +90.3 | 162.6 | 181.4 | 228.8 | 255.3 |
| 284235 | γ1 6004 +1.6, γ2 74027 +16.7 | 161.1 | 176.8 | 161.4 | 177.2 |
| 347129 | γ2 11000 +19.8 | 130.7 | 149.4 | 131.1 | 149.9 |
| 64591 | γ1 83044 +4.1 | 174.0 | 175.2 | 149.6 | 150.6 |

evt269774 is the one that exercises the sign convention in both directions.

**Nothing was re-saved.** Those records are the scanner's; the numbers are
reported so the decision to revisit them is theirs (M13).

### 20.6 The four cases the sum keeps apart

| mark | effect | why |
|---|---|---|
| **member** marked IN | **nothing** | already inside `kine_charge` |
| non-member marked OUT | **nothing** | never was in it |
| member marked OUT | −`E_est` | |
| non-member marked IN | +`E_est` | the case that was asked for |

The first is not hypothetical: round 10's whole-shower select can sweep up
members, since `show members too` is on by default. It is pinned by a test that
bulk-marks a member IN and asserts the energy does not move.

A marked segment's `E_est` was converted with **its own** shower's recombination
pair, so under the EM hypothesis the round-9 ratio is applied per segment,
reusing `kine_hypothesis` rather than keeping a second copy of the constants — a
segment from a track-flagged shower scales by 0.665/0.4 = 1.6625.

### 20.7 The record

```
pio.gammas.<slot>.energy_includes_marks    false on every pre-round-12 record
pio.gammas.<slot>.energy_marks_delta       what the marks moved it by
pio.gammas.<slot>.energy_without_marks     the energy before that delta
pio.gammas.<slot>.energy_marks_detail      [{seg, kind, E_est, energy, owner,
                                             owner_kine}]
pio.gammas.<slot>.energy_marks_unknown     marked, but owned by no shower
```

Proved additive rather than asserted, the same way round 11's extraction was:
records from the pre-change viewer and the new one, over three pairing shapes,
differ **only** by these five keys — every existing value, energy and mass
included, is byte-for-byte the same.

### 20.8 Noticed, not fixed

`g1_ehyp` / `g2_ehyp` are **not** reset when an event without a π⁰ block is
opened, so a round-9 hypothesis switch can carry over from the previous event
until `load_label` overwrites it. `emark_mode` is reset explicitly (round 12), so
the two now behave differently. Reported rather than fixed in the same change;
it is a pre-existing leak in a different round's control.

`kine_div` and `gstart3_src` gained `name=` so the browser test could read them.
Inert at runtime.

## 21. Round 13 — the π⁰ vertex point, and one keystroke from losing it

> *"when I save the pi0 with a manual vertex, did you actually save the manual
> vertex point? same for the NCpi0 vertex point reconstructed by back-tracking.
> These need to be inside the scan result."*

### 21.1 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
python em_display/selftest_em_display.py        # 291
python em_display/selftest_em3d_browser.py      # 66
python em_display/selftest_repro.py             # 98/98
```

No C++ and no jsonnet are touched, so **no A/B gate is owed** — as in rounds 1-12.

### 21.2 The answer: yes, both, as points

Checked against the twelve records on disk rather than against the code.

**Manual** — `labels-evt64591.json`:

```
pio.vertex_how  "manual"
pio.vertex      [-96.59, -27.60, 175.72]
```

**Back-projected** (both `no_vertex_ncpi0` events, evt169626 and evt347129) —
`labels-evt347129.json`:

```
pio.vertex_how      "backproject"
pio.vertex          [-203.008, -100.171, 208.143]
pio.backproject     verdict "ok"      branch "one_short"   gap    7.530
                    theta   86.033    mass   131.128
                    angle1  0.000     angle2 23.620
                    dis1    62.178    dis2   18.795
                    len1    34.599    len2   13.595
event_flags         ["no_vertex_ncpi0"]
main_vertex         {cluster_id 53, x -193.099, y -41.811, z 189.115}
```

So the back-projected case keeps not just the point but **which of the three C++
branches produced it** (`one_short` here — the branch that re-rays the short
gamma), the closest-approach gap, both back-projection angles against the 25°
gates, and `verdict`, which records whether the code itself would have kept it.
Round 11's stored candidates each carry their own copy of all of this, so a
per-pairing vertex is saved too.

The event's own reconstructed vertex is at the record's top level as
`main_vertex`, independently of which convention the π⁰ used.

### 21.3 What the question turned up

The vertex was saved. It was **one keystroke from not being**.

`load_label` restored a manual vertex into `state["vtx_manual"]` and moved the
mode radio, but never filled the **x/y/z boxes**. So a re-opened manual-vertex
event showed the right vertex everywhere while the three boxes read empty — and
`_manual_point()` reads all three:

```python
def _manual_point():
    try:
        return (float(man_x.value), float(man_y.value), float(man_z.value))
    except ValueError:
        return None
```

Editing any *one* box fires `on_manual`, which re-reads all three, gets `None`
from the two still-empty ones, and assigns it: `state["vtx_manual"] = None`.
Reproduced on evt64591 before the fix:

```
boxes on re-open : '' '' ''
state vtx_manual : (-96.594, -27.597, 175.720)
--- nudge x to -96.0 ---
state vtx_manual : None
would save vertex: None   how: manual
```

`vertex: null` under `vertex_how: "manual"` — a record that says a manual vertex
was chosen and does not say where. Silent, and it survives a save.

`set_pio_vertex` already had the correct order (boxes first, then state, under
`_suspend`); only the restore path was missing it. Fixed the same way, and the
round-11 candidate load-back already wrote the boxes.

After:

```
boxes on re-open : -96.6 -27.6 175.7
--- nudge x to -96.0 ---
state vtx_manual : (-96.0, -27.6, 175.7)
```

Note the restore quantises to 0.1 cm, because that is what `set_pio_vertex`
writes and the boxes are the display's own convention. A vertex set by tapping a
fit point keeps full precision in the record but reads back through the boxes at
0.1 cm if it is then edited — 0.05 cm at worst, against ~0.3 cm fit-point
spacing, so ~0.06° on a 50 cm baseline.

### 21.4 Why no record on disk hit it

Of the twelve, exactly one (evt64591) uses `manual`, and its vertex is intact:
the boxes only matter once a re-opened event is edited, and that had not
happened yet. Found by the question, not by a symptom.

### 21.5 Left open

Nothing new. The `g1_ehyp` / `g2_ehyp` cross-event leak noted in §20.8 is still
open and is the same class of bug — a control whose state and its widget can
disagree across a `load`.

## 22. Round 14 — the corrected start and axis reach the back-projected vertex

> *"for event 76346, I made some corrections to the EM shower start point and
> direction, but when I tries to calculate pi0 mass again, and use the
> back-project the two gamma, it does not seem to use the latest hand scan
> information, so the pi0 vertex is not at the correct position."*

Correct, and it was a real gap. The report is confirmed and fixed.

### 22.1 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
python em_display/selftest_em_display.py        # 311
python em_display/selftest_em3d_browser.py      # 71
python em_display/selftest_repro.py             # 98/98
```

No C++ and no jsonnet are touched, so **no A/B gate is owed** — as in rounds 1-13.

Event 76346 is in the manifest as written; no typo this round. It is pinned by
fingerprint rather than by number: the three showers the owner corrected —
14058, 14059, 50052 — co-occur in exactly one of the 98 events.

### 22.2 Root cause

`pio_vertex()` called

```python
bp = G.pi0_backproject(sh1, sh2, cur_segments(), anchor)
```

and `pi0_backproject` builds each gamma's ray itself, from the reconstruction
(`em_geom.gamma_ray`, mirroring `NeutrinoShowerClustering.cxx:4139-4142`):

```
test_p = the shower's fitted point closest to the main vertex
dir    = shower_cal_dir_3vector(shower, test_p, 15 cm)
```

Neither input consults `state["em_start"]`, `state["gstart"]` or
`state["em_dir"]`. So the panel printed **"your corrected start from EM mode"**
for the two mass conventions — round 8b made `shower_start` honour the
correction on the π⁰ path for exactly that reason — and then back-projected from
the reconstruction's geometry anyway, in the same readout, with nothing saying
so. Round 8b fixed this disease for the masses and did not reach this function.

### 22.3 How badly it showed on evt76346

From the owner's own label (`emscan-0827/labels-evt76346.json`, `em` block):

| shower | reco `test_p` | their start | moved | reco axis vs theirs |
|---|---|---|---|---|
| 14059 | (-106.52, -10.22, 362.29) | (-109.94, -35.38, 345.82) | 30.3 cm | **155.5°** |
| 14058 | (-106.52, -10.22, 362.29) | (-92.01, -58.29, 338.36) | 55.6 cm | 47.4° |
| 50052 | (54.25, -145.12, 466.60) | (-109.94, -35.38, 345.82) | **231.5 cm** | 84.0° |

The 155.5° on 14059 is the headline: the reconstruction had that shower
effectively pointing the other way, and the owner turned it round. Its reco
`test_p` is also the main vertex itself, to 12 decimal places — and so is
14058's, because both showers' closest fitted point to the anchor *is* the
anchor.

Which is why, for the pair 14059 + 14058, the back-projection did not merely
land in the wrong place:

```
today  (reco rays)  branch -           verdict degenerate   vertex (-106.52, -10.22, 362.29)
                    theta -   gap -    -- both rays start at the same point
with the hand scan  branch one_short   verdict ok           vertex (-114.72, -58.77, 328.04)
                    theta 65.9  gap 2.03  angles 0.0 / 4.7
```

`degenerate` means the two rays share an origin, so the returned "vertex" is
just that origin — the main vertex. The scanner's geometry turns that into a
real back-projection with a 2.0 cm closest approach that passes both 25° gates.
The vertex moves **60.0 cm**.

The π⁰ slots were never recorded for this event (`pio` is `null` — it was saved
from EM mode), so which pair the owner had in the slots is not knowable from the
record. All three plausible pairs are therefore reported rather than one being
picked:

| pair | today | with the hand scan | moves |
|---|---|---|---|
| 14059 + 14058 | `degenerate`, at the main vertex | `ok`, gap 2.03 cm | 60.0 cm |
| 14059 + 50052 | `angle_gate`, gap 140.7 cm, angle1 180.0° | `degenerate` (they pinned both starts to vertex 48038) | 47.3 cm |
| 14059 + 41031 (the reco's own `pio_id=0` pair) | `angle_gate`, gap 7.02 cm | `angle_gate`, gap 1.09 cm | 9.9 cm |

Shower 50052 is the clearest case for the fix and for one of its details: it is
**0.4 cm long**, and its reco `test_p` sits 231 cm away, in a different part of
the detector. `shower_cal_dir_3vector` over a 0.4 cm stub is not a direction
anybody should use.

### 22.4 The fix

One injection point, in `em_geom.pi0_backproject`:

```python
def pi0_backproject(sh1, sh2, segments, anchor, ray1=None, ray2=None):
```

`rayN` is `(point, direction)` for a gamma whose start or axis the scanner has
corrected, or `None` to build that gamma's ray the way the code does. The
viewer's `gamma_scan_ray(node, slot)` produces it, with the same precedence
`shower_start` and `shower_axis` use, so the ray, the two mass conventions and
the EM gate cannot disagree about what the scanner said:

| what the scanner set | ray origin | ray direction | provenance string |
|---|---|---|---|
| nothing | — | — | `None` — the mirror builds its own |
| start only | that start | `shower_cal_dir_3vector` at it, 15 cm | `em_start_correction + dir15@probe_members` |
| axis only | the mirror's own `test_p` | through the aimed point | `reco_ray_point + manual_axis` |
| both | that start | through the aimed point | `em_start_correction + manual_axis` |
| a π⁰-slot start | `gstart[slot]` | as above | `gamma_slot_override + …` |

Three details are load-bearing.

**`None` is the guarantee, not the arithmetic.** A gamma the scanner never
touched injects no ray at all, so `pi0_backproject` falls through to its own
`gamma_ray` and is byte-for-byte the pre-round-14 function. Rebuilding the
un-overridden ray in the viewer would look equivalent and is not:
`gamma_scan_ray` resolves membership through `members_of` (the probe sidecar)
while `gamma_ray` uses the dump join, and the two differ on exactly the lossy
showers `em_geom.join_completeness` names — evt347129 among them, which is one
of the two records on disk that already use back-projection. Measured across all
2490 shower pairs of the 98 manifest events: `pi0_backproject(a, b, segs,
anchor)` and `pi0_backproject(..., ray1=None, ray2=None)` differ **0 times**.

**Probe membership when only the start moved.** The dir15 recompute uses
`members_of`, which is `shower_axis`'s own `python@start_override` branch — the
axis the panel is displaying. Building the ray from the mirror's dump-join
membership instead would have recreated round 8b's disease inside one record: a
vertex from one member set, an axis on screen from another. It is a deliberate
divergence from the mirror and it is named in the provenance string.

**The short gamma is not re-rayed when the scanner has stated it.** The
one-short branch (`:4203-4247`) re-derives the short shower's direction because
the code does not trust the stub's own. A scanner who has said where the stub
starts and where it points has answered exactly that question. The branch's own
arithmetic — keep the closest point on the *long* gamma's ray, not the midpoint
— is untouched, because that is what the branch is. `short_rerayed` in the
result says which happened.

Degradation is honest: on evt76346, shower 14058 is 6.9 cm long and its start
was moved 55.6 cm, so no member point falls inside the 15 cm window and there is
no direction to be had. That gamma injects no ray, the mirror builds its own,
and the provenance reads `em_start_correction + dir15 undefined (no member point
within 15 cm)` rather than quietly substituting some other vector.

### 22.5 Both readings, and which one is the default

They answer different questions — *"what vertex would the code compute here"*
versus *"what vertex does my geometry imply"* — so both are offered under
**back-projection geometry**, and the panel always shows the one it is not
using, with the distance between them:

```
rays: your corrected start / axis — γ1 em_start_correction + manual_axis,
                                    γ2 em_start_correction + manual_axis
the other reading (the reconstruction's own rays) puts it at
(-106.5, -10.2, 362.3), verdict degenerate — 60.0 cm away
```

On the legacy setting the same line appears in amber with *"you have corrected
this pair's geometry and the vertex above is **not using it**"*, so the state
round 14 fixes can no longer be silent.

The default is the scanner's geometry. A start override or an aimed axis exists
for exactly one reason, and unlike round 12's marks it carries no other reading.
**Nothing on disk is re-priced by that default**, by two separate mechanisms:

* a gamma the scanner never touched injects no ray (§22.4), so the 14 records
  with no correction cannot move at all; and
* `pio.backproject_geometry` is written into every record, and `load_label`
  restores it. Absent — every record written before this round — means the
  reconstruction's own rays. The rule is applied by **what the record says**,
  never by whether the number happens to differ.

Both back-projected records on disk therefore re-open exactly as saved. Measured
read-only, for the record, what the new default *would* give if the owner ever
chose to re-save them:

| record | as saved | under the new default | moves | m(vertex) |
|---|---|---|---|---|
| evt169626 (53069 + 22034) | (3.75, 208.54, 414.68), gap 11.36 cm, angles 4.9°/5.5° | (9.05, 207.00, 411.87), gap **1.05 cm**, angles 0.6°/0.5° | 6.19 cm | 145.4 → 144.7 MeV |
| evt347129 (53021 + 11000) | (-203.01, -100.17, 208.14) | identical | **0.00 cm** | 131.1 MeV |

evt347129 does not move because the owner's start override for shower 11000 is
the reco's `test_p`, to the last bit. evt169626's correction *improves* the
back-projection sharply — the two rays go from missing each other by 11.4 cm to
1.1 cm — while the mass barely moves, which is the expected signature of a
correction that fixes a vertex without changing the opening angle.

**Neither record was re-saved** (M13). Re-pricing a past judgement is the
owner's call, made per event.

### 22.6 What the record now carries

`pio.backproject_geometry` at the top level (`"handscan"` / `"reco"`), and
inside `pio.backproject`:

```
geometry      "handscan"
ray1_source   "em_start_correction + manual_axis"
ray2_source   "em_start_correction + manual_axis"
ray1_given    true          ray2_given  true
short_rerayed false
alt           { geometry "reco", verdict "degenerate", vertex […], … }
```

`alt` is the full result under the other geometry, present whenever either gamma
was corrected. So a record carries both vertices and a later reader can see how
far the hand scan moved it without re-running anything. Round 11's stored
pairings each carry their own copy, and `on_pio_load` restores a candidate's
geometry along with its start and energy hypothesis — a stored pairing is frozen
numbers, and the geometry its vertex was built from is one of them.

### 22.7 One thing that deliberately does not move

The 15 cm branch test reads `showers[].total_length`, a reconstruction number.
Round 12's IN / OUT marks move a gamma's **energy**, not its length, so a stub
built up by hand stays on the one-short branch — 50052 stays a 0.4 cm shower for
branch selection however much charge is marked into it. That is mirror fidelity,
not an oversight, but it looks like a bug from the outside, so the panel now says
it in the back-projection block.

`backproject.mass` also stays the mirror's own: raw `kine_charge`, ignoring
rounds 9 and 12. It is what the C++ would compute. The masses that honour the
hypothesis switch and the marks are `mass_axis_convention` and
`mass_vertex_convention`, as before.

### 22.8 Left open

Unchanged from §20.8 and §21.5: `g1_ehyp` / `g2_ehyp` are still not reset on
event load, so a round-9 hypothesis switch can leak across events until
`load_label` overwrites it. `bp_geom` and `emark_mode` are both reset in `load()`
for exactly that reason; the two energy Selects are the remaining pair.

## 23. Round 15 — the π⁰ vertex mode is per-event, on the way back as well

> *"For the pi0 panel, I wonder if you actually save my choice of 'main vertex'
> 'back-project ...', 'manual'? When I go back to the previous events, I feel
> that it was just set on the previous event, instead of what I actually chose
> for this current event before."*

Half right, and the half that is right is the half that matters. **The choice
was always saved.** Every record on disk carries it, correctly, and nothing has
to be re-scanned. What was broken is the **restore**: re-opening an event put
back only two of the three choices, so for the third the radio kept showing
whatever the previous event had left in it.

### 23.1 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
python em_display/selftest_em_display.py        # 323
python em_display/selftest_em3d_browser.py      # 75
python em_display/selftest_repro.py             # 98/98
```

No C++ and no jsonnet are touched, so **no A/B gate is owed** — as in rounds 1-14.

The measurement that opened the round, read-only against the owner's own tag
(`/home/xqian/tmp/em114r15/repro.py`): re-open each of the 18 labelled events of
`emscan-0827` three times, once with the radio parked on each mode first, and
compare what comes up against what the record says.

```
before   27 of 54 re-opens show the WRONG vertex mode
after     0 of 54
```

### 23.2 Root cause

`load_label` restored the mode with two positive branches and no default:

```python
if pio.get("vertex_how") == "manual" and pio.get("vertex"):
    ...
    vtx_mode_group.active = 2
elif pio.get("vertex_how") == "backproject":
    vtx_mode_group.active = 1
```

There is no branch for `main_vertex`, and nothing anywhere reset the radio — it
is not in `load_label`'s reset block (which has handled `em_verdict`,
`conf_group`, `event_flag_group`, `g1_ehyp`, `g2_ehyp` and `note_in` since round
9) and not in `load()` (which resets `emark_mode` and, since round 14,
`bp_geom`). A `RadioButtonGroup` keeps its value across an event switch like any
other widget, so `main vertex` — the most common answer, and the default — was
the one choice the display could not put back.

Which records that hits, in `emscan-0827`:

| `vertex_how` | records | restored before round 15 |
|---|---|---|
| `main_vertex` | 9 | **no** — showed the previous event's mode |
| `backproject` | 3 | yes |
| `manual` | 3 | yes |
| `pio` absent (saved from EM mode) | 3 | **no** — nothing to restore from, so also inherited |

`on_pio_load` — the button that loads a *stored pairing* back into the slots —
has had the correct shape since round 11:

```python
vtx_mode_group.active = {"main_vertex": 0, "backproject": 1, "manual": 2}.get(how, 0)
```

a total mapping with a fallback. `load_label` never grew one. So loading a
stored candidate showed the right mode while re-opening the event did not, which
is the inconsistency behind "I feel that it was just set on the previous event".

### 23.3 A second leak, found while checking the first, and worse

The manual x/y/z boxes leak with the radio, and so do the **EM-mode start
boxes**. `state["vtx_manual"]` *is* cleared in `load()`, so a leaked `manual`
mode reads honestly as *no point set* — but the boxes still held the previous
event's numbers, and one keystroke in one of them would have built a vertex out
of two coordinates belonging to another event. That is the round-13 evt64591
failure mode with the sign flipped.

`em_sx / em_sy / em_sz` are worse, because nothing reports them. `_anchor_for_snap`
reads the boxes **first** and falls back to the selected shower's own start only
when they do not parse:

```python
def _anchor_for_snap():
    try:
        return (float(em_sx.value), float(em_sy.value), float(em_sz.value))
    except (TypeError, ValueError):
        pass
    node = state["sel_shower"]
    return shower_start(node) if node is not None else None
```

So *snap start to nearest reconstructed vertex* and *aim axis at nearest fit
point* anchored on a point typed into a previous event, with nothing on screen
to say so. Measured (`/home/xqian/tmp/em114r15/anchor.py`, run against the
pre-round-15 file and the fixed one):

```
type a start on evt463565, switch to evt84229, select shower 69134
  before   _anchor_for_snap() -> (-96.6, -27.6, 175.7)   335.1 cm from the shower's own start
  after    _anchor_for_snap() -> (103.5, 124.1, 397.5)   the shower's own start
```

### 23.4 The fix

Three resets into `load_label`'s existing unconditional block — the one that
already runs before the "does a record exist" test, so it covers an unlabelled
event, a record with no π⁰ block, and a saved record alike:

```python
vtx_mode_group.active = 0
man_x.value = man_y.value = man_z.value = ""
em_sx.value = em_sy.value = em_sz.value = ""
```

and the restore made total, the way `on_pio_load` always was:

```python
_how = pio.get("vertex_how")
if _how is not None:
    vtx_mode_group.active = {"main_vertex": 0, "backproject": 1,
                             "manual": 2}.get(_how, 0)
```

Two details are load-bearing:

- **`manual` with a null vertex stays on `manual`, boxes empty.** That is what
  such a record was saved as, and `refresh_kine` says *no point set* for it.
  Falling back to `main vertex` there would put a vertex on screen that the
  record does not carry.
- **An unrecognised `vertex_how` falls back to `main vertex`, not to what was on
  screen.** Such a value can only come from a future build; inheriting would
  reintroduce exactly this bug for it.

`bp_geom` and `emark_mode` stay where they are, reset in `load()`. They work;
moving them would only enlarge the diff.

`man_x / man_y / man_z` gained `name=` so the browser test can read them out of
the live page. Fourth instance of that class (round 11 `gstart3_src`, round 13
`kine_div`, round 14 `vtx_mode_group`).

### 23.5 What is on disk

Nothing was rewritten this round, and nothing needs to be. All 15 π⁰-bearing
records of `emscan-0827` were re-priced read-only and every one is
self-consistent — the stored `vertex` matches its own stated `vertex_how`:

```
main_vertex (9)   stored vertex == the dump's main_vertex to 1e-6 cm
backproject (3)   59.56 / 62.18 / 58.49 cm from the main vertex
manual      (3)   177.04 / 3.48 / 67.91 cm from the main vertex
```

**What that check can and cannot say.** It rules out a *torn* record — one whose
label and point disagree. It cannot prove the convention is the one that was
meant: had the leaked radio ever been re-saved over an event, the point and the
label would have been recomputed together and would agree with each other while
both being wrong. What argues against that having happened is provenance, not
arithmetic — the nine `main_vertex` records were saved 17:27–19:03 on 08-27, and
the three `backproject` ones are all accounted for (evt169626 and evt347129 from
the round-14 survey, evt76346 saved 08-28 03:57 under `backproject_geometry:
handscan`, i.e. deliberately, with the round-14 control). If any event's mode
still looks wrong on re-opening now, it is worth re-saving that one event; the
display can no longer be the reason.

### 23.6 Tests

`selftest_em_display.py` 311 → **323**, `selftest_em3d_browser.py` 71 → **75**.
Eight of the twelve new static checks **fail against the pre-round-15 file** and
pass after (`/home/xqian/tmp/em114r15/st_pre.log`) — the four that pass in both
are the save-side check, which was never broken, and the live-data guards.

The save side is checked through the real save path, once per mode, and the
restore side re-opens with the radio parked on each of the three modes in turn:
a same-mode test passes on the broken code. One guard is deliberately live-data
rather than a fixed list — every `vertex_how` present anywhere under `em_labels`
must map to a button, and `main_vertex` must still be represented, so the
regression case cannot quietly leave the corpus.

The browser block pins it where the owner saw it: save a manual vertex from the
real page, switch events (radio back to `main vertex`, all boxes empty), switch
back (radio on `manual`, boxes refilled).

### 23.7 Correction to §20.8 / §21.5 / §22.8

Those sections carried "`g1_ehyp` / `g2_ehyp` are still not reset on event load".
That has been **wrong since round 9**: `git log -S 'g1_ehyp.value = EHYP_RECO'`
returns `8efe41a` (round 9), and the reset sits in `load_label`'s block at
`em_display_viewer.py:3204`. The note was carried forward across three rounds
without being re-checked against the file.

After this round every widget that carries a **per-event answer** is reset on
load: `em_verdict`, `conf_group`, `event_flag_group`, `g1_ehyp`, `g2_ehyp`,
`note_in`, `vtx_mode_group`, `man_x/y/z`, `em_sx/y/z` in `load_label`, and
`emark_mode`, `bp_geom`, `excl_choice` in `load()`. The view and cloud controls
— `fit_mode`, `view_size`, `cloud_layer`, `cloud_color`, `cloud_max`,
`cloud_scope`, `tap_action`, `tap_toggle`, `dim_toggle`, `seg_color_mode`,
`gstart_slot`, `mode_group` — persist deliberately: they are how the scanner
wants to look at events, not answers about one. That split was made by reading
the widget list, not by test.

## 24. Round 16 — a stored pairing can be told from today's answer, and replaced

> *"When you store the pairing into the stored π⁰ pairing table, I believe the
> information saved is not the best information (e.g. energy after EM clustering
> fix, update of the π⁰ vertex choice etc), right? I wonder if it can be fixed?"*

Right, and it is two different things wearing one coat:

- a row can go **stale** — an input under it (a start, an axis, a mark) moved
  after it was stored, so the frozen numbers are no longer what the chain
  computes;
- a row can simply **not count a correction that exists** — it was stored with
  *gamma energy membership* on `reco membership only`, so the EM clustering fix
  made by hand sits outside it. Re-priced under its own settings that row has
  not moved at all; it was never the best number;
- and the same for the **vertex choice**: a row stored under one convention
  while the scan has since settled on another is not stale either, it is an
  alternative that may no longer be wanted.

Both were invisible from the table, and one of them could not even be corrected:
storing the fixed pairing was **refused as a duplicate**.

### 24.1 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
python em_display/selftest_em_display.py        # 358
python em_display/selftest_em3d_browser.py      # 79
python em_display/selftest_repro.py             # 98/98
```

No C++ and no jsonnet are touched, so **no A/B gate is owed** — as in rounds 1-15.

The two regressions, run against the pre-round-16 file and the fixed one
(`/home/xqian/tmp/em114r16/ab.py`, `ab2.py`; evt409634, shower 27015 merged into
69032 exactly as round 12 did it):

```
1. store the SAME pairing again with the marks counted
   before   1 row  ['41.0']   "that pairing is already stored as candidate 1 --
                               same gammas, vertex, starts and energy hypotheses"
   after    2 rows ['41.0', '78.8']

2. load a row that was stored WITH the marks counted (E1 144.11 MeV)
   before   switch reads 'reco membership only', screen shows E1 39.06 MeV
   after    switch reads 'include my IN / OUT marks', screen shows E1 144.11 MeV
```

The second one is the worse of the two: the panel displayed 39.06 MeV for a row
whose own stored energy is 144.11 MeV, immediately after being asked to load
that row.

### 24.2 Root cause

**`_pair_sig` did not include the membership switch.** It is the key that
decides "is this pairing already stored", and it carried the gammas, the vertex,
the starts and the energy *hypothesis* — but not the round-12 switch and not the
round-14 geometry. Two pairings with different energies therefore collapsed to
the same signature, so `on_pio_add` declined the corrected one and the "the
pairing now in the slots is not one of them" nudge stayed silent as well.

**`on_pio_load` restored the hypothesis and the geometry but not the switch.**
Rounds 9 and 14 both established "the candidate's own setting comes back, absent
means off"; round 12's switch was left out, so a stored row was put on screen
under whichever setting the *event* had been restored with.

**The table had no way to say a row was behind.** Deliberately — a stored
candidate is frozen because it is the scanner's curated judgement, and
`on_pio_load` has always *reported* a disagreement rather than absorbing it. The
defect is not the freeze. It is that the freeze had **no exit**: nothing said a
row had gone stale, and nothing replaced one on purpose.

### 24.3 What that costs on the owner's own tag

Measured read-only over all 15 stored pairings in `emscan-0827`
(`/home/xqian/tmp/em114r16/survey2.py`, 2026-08-28 04:55 — the scan is live, so
the counts move), each row re-priced under the settings it was stored with:

| event | row | gammas | flag |
|---|---|---|---|
| evt76346 | 1 | 14059+14058 | **stale** — g1 axis turned 6.0°, g2 start moved 55.6 cm, g2 axis turned 47.4°; and g1 ignores marks (−9.4 MeV) |
| evt409634 | 1 | 21002+69032 | g2 ignores your marks (**+105.1 MeV**) |
| evt281485 | 1 | 15036+88090 | g2 ignores your marks (+75.7 MeV) |
| evt54332 | 2 | 122091+27025 | g1 ignores your marks (+52.5 MeV) |
| evt47212 | 1 | 2103+59027 | g1 ignores your marks (−45.6 MeV) |
| evt54332 | 1 | 16014+128111 | g1 ignores your marks (−35.8 MeV) |
| evt347129 | 1 | 53021+11000 | g2 ignores your marks (+19.8 MeV) |
| evt284235 | 1 | 6004+74027 | g1 +1.6 MeV, g2 +16.7 MeV |
| evt64591 | 1 | 83044+78025 | g1 ignores your marks (+4.1 MeV) |
| evt166870 | 1 | 87058+85045 | g1 ignores your marks (+2.4 MeV) |

**1 of 15 rows is stale; 10 of 15 do not count marks that exist on their gammas.**
The signs matter: three of them are *negative*, so counting the marks would lower
those energies. Nothing here says which reading is right — that is the scan's
judgement, and it is exactly why the row is flagged rather than rewritten.

### 24.4 The fix

1. **`_pair_sig` gains `energy_includes_marks` per gamma and
   `backproject_geometry`.** Both normalised so an absent key reads as the
   pre-round-12 / pre-round-14 default — otherwise every old row would look
   different from an identical new one.
2. **`on_pio_load` restores the candidate's membership switch**, absent meaning
   off, symmetric with rounds 9 and 14.
3. **`cand_drift(c)`** compares a candidate's own inputs against today's —
   energies re-priced under *its* hypothesis and *its* switch, starts and axes
   against today's `shower_start` / `shower_axis`, and the reconstruction's own
   vertex for a `main_vertex` row. It touches no widget, so the table can carry
   the flag without the controls flickering.
4. **`cand_unused_marks(c)`** is the separate question: marks that exist on this
   row's gammas and are not in its energies, with what they are worth. Same
   arithmetic the live panel has warned with since round 12.
4b. **`cand_convention_note(c)`** is the third: the row's `vertex_how` (and, for
   a back-projection, its round-14 geometry) against the one this event was last
   **saved** with — the saved record rather than the live radio, so the note does
   not flicker while rows are being looked at. No row on disk differs today; it
   is the owner's own second example and it costs one comparison.
5. **`update to today's numbers`** — the exit. It replaces the selected row in
   place, keeps its `stored_utc`, stamps `updated_utc`, and appends what the row
   said before to a `supersedes` list. Two guards make it a judgement rather than
   an accident: a row must be selected, and the slots must be holding *that
   row's* two gammas, which is what `load into the slots` does. Delete-and-
   re-store would have done the same arithmetic while losing the row's place in
   the list and its first-stored time, and renumbering every note that named it.
6. **`energy_for(node, as_em, with_marks)`** — `gamma_energy`'s arithmetic with
   the switches passed in. `gamma_energy` now delegates to it, so re-pricing a
   stored row and pricing the live one can never drift apart. Same reason round
   11 gave for lifting `gamma_record` out of `on_save`.

`refresh_kine` calls `refresh_pio_cands` at its end, so the flag is live from
every handler that already refreshes the π⁰ panel rather than from fifteen call
sites a new handler could forget. Measured rather than assumed
(`/home/xqian/tmp/em114r16/cost.py`): `refresh_kine` including the table is
**0.24 ms** on evt281485 (3 stored rows, the most on disk) and **0.48 ms** on
evt76346, whose back-projection runs twice for round 14's `alt`.

### 24.5 What deliberately does not change

**Nothing re-prices itself.** A stored row is still frozen; the update is a
button the scanner presses, on a row they selected, after loading it. No record
on disk was rewritten this round.

**The `gamma energy membership` default was the owner's call, and they took it
— see §24.7.** The finding that prompted it: **every record on disk that carries
marks had `energy_includes_marks: false` on both gammas** — 14 segments marked
into 14059 on evt76346, 10 into 69032 on evt409634, 10 and 8 on evt54332 — so no
hand-made clustering fix reached any saved mass.

**`em_start` / `em_dir` are still not restored by `load into the slots`.** They
are per *shower*, shared by every candidate that names it, so restoring one row's
would move another's. The update button absorbs that drift when pressed; the
load path still will not reverse it. evt76346 row 1 is the live example.

### 24.6 Tests

`selftest_em_display.py` 323 → **351**, `selftest_em3d_browser.py` 75 → **78**.

The static block rebuilds round 12's merge on evt409634 and then pins: the
delegation (`energy_for` == `gamma_energy` for both switch positions), the
signature telling the two rows apart while an absent key still reads as off,
the store that used to be refused, stale-versus-does-not-count as two separate
flags (including a row that becomes stale when the marks are taken away again,
and one that becomes stale from an EM start correction), the switch coming back
with the row it belongs to, the update button's in-place replacement with
`stored_utc` kept and `supersedes` appended, its refusal when the slots hold a
different pairing, the vertex-convention note in all three of its states, and
the plain-text table cell. One guard is live-data: at least one stored row on
disk must have a gamma with marks and the switch off, so the case the owner
reported cannot quietly leave the corpus. It is a floor, not an equality — the
scan is live and the count moves.

The browser block presses the new button in the real page on evt281485 —
including the honest "nothing had moved" result and the no-row-selected refusal.

### 24.7 Round 16b — the membership default is now ON

Round 12 made counting the scanner's own IN / OUT marks **opt-in**, and §24.5
measured what that cost: of the 18 records in `emscan-0827`, *every* one that
carries marks was saved with the switch off, so no hand-made clustering fix
reached any saved mass — including deltas of +105.1, +75.7 and −45.6 MeV. Put to
the owner as a question with the numbers, they chose to flip it. The marks exist
because the reconstruction's membership is wrong; counting them is what the scan
is for.

```python
emark_mode = Select(..., value=EMARK_MARKS, ...)   # was EMARK_RECO
```

and `load()`'s per-event reset returns to `EMARK_MARKS` rather than
`EMARK_RECO`.

**Nothing already on disk is re-priced, and that is not an assertion.** The flip
is only safe because `load_label` sets the switch **from the record**, and until
this round that restore was one-way — it could turn the switch on, never off.
With the default on, a record saved with it off would have been silently
re-priced the moment it was re-opened. So the restore was made total, guarded on
the record actually carrying gammas:

```python
if pio.get("gammas"):
    emark_mode.value = (EMARK_MARKS
                        if any(g["energy_includes_marks"] for g in both slots)
                        else EMARK_RECO)
```

Measured over all 18 labelled events, re-opening each and comparing the switch,
both gamma energies, both masses and every stored candidate's energies
(`/home/xqian/tmp/em114r16/emark_snap.py`, run against the pre-flip file and the
flipped one):

```
events compared      : 18
events that re-price : 0
switch restored OFF  : 15   (every record that carries a pi0 block)
switch takes the new default : 3   (evt172942, evt174752, evt76350 -- pio null,
                                    no gammas, no mass: nothing to price)
```

What this **does** change is what a *new* save records, and what a *re-save*
records: an event re-opened, re-marked and saved again now writes the mass with
the marks counted unless the switch is put back. That is the intended effect —
and the round-16 row flags are what make it visible per stored pairing.

The round-12 selftest check that asserted the old default is **inverted, not
deleted**: it is the one that would catch this default moving again by accident.
`selftest_em_display.py` 351 → **358**, `selftest_em3d_browser.py` 78 → **79**.

---

## 25. Round 17 — a marked segment no shower owns is explained, not just named

> *"For evt 169626 … for the segment 53070, in the EM clustering, I added it to
> the gamma1 (53069), but in the pi0 mode, I see this warning `γ1: 53070 marked
> but owned by no shower, so the probe has no per-segment energy for it — not
> counted either way, and no dQ estimate is offered because E_est/dQ is not
> constant between showers.` This is somewhat strange. Can you check and improve
> the presentation?"*

The message was **true and useless**. Every fact needed to make it useful was
already on disk — in the dump, in the probe sidecar, and in the scanner's own
saved label — and none of it was on screen. Nothing about the energy arithmetic
changes in this round.

### 25.1 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
python em_display/selftest_em_display.py        # 397
python em_display/selftest_em3d_browser.py      # 86
python em_display/selftest_repro.py             # 98/98

# the reported message, before and after, off the owner's own scan tag:
python /home/xqian/tmp/em114r17/probe.py

# nothing a record can hold moves (pre-change copy vs live):
python /home/xqian/tmp/em114r17/ab.py /home/xqian/tmp/em114r17/pre.tsv  \
       /home/xqian/tmp/em114r17/pre
python /home/xqian/tmp/em114r17/ab.py /home/xqian/tmp/em114r17/post.tsv \
       /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin/em_display
cmp /home/xqian/tmp/em114r17/pre.tsv /home/xqian/tmp/em114r17/post.tsv
```

No C++ and no jsonnet are touched, so **no A/B gate is owed** — as in rounds 1-16.

### 25.2 What is actually true about evt169626 / segment 53070

Read out of `work-mcp1k-prod0825/pr_evt169626/calib-pr-evt169626.json` and
`em_display/emprep/emprep-evt169626.json`:

| fact | value | source |
|---|---|---|
| `shower_id` | **-1** — in no shower | the dump, `segments[]`, on its own |
| `particle_id` / `particle_score` | 2212 (proton) / 100 | the dump |
| `flag_shower` | false | the dump |
| length | 14.09 cm, 24 fitted points | the dump |
| where it attaches | `start_vertex_id 53063` = γ1 start segment 53069's `end_vertex_id` | the dump |
| the reco's decision | `SHOWER_ABSORB EXCLUDE shower_start_seg=53069 seg=53070 pdg=2212` at `in_other_clusters_A` | the probe log |
| a second, independent one | `SHOWER_MERGE … cand_seg=53070 angle=148.173 dis=28.207 verdict=outside_cone` | the probe log |
| its charge | ΣdQ 1 690 300 e = **20.1 %** of γ1's own 8 421 799 e | the dump's per-point `dQ` |

So the reconstruction did not overlook this segment. The exclusion is F12
(`PRShower.cxx:722-727`, doc pr/40 round 6, knob `absorb_track_guard`): a
straight long track with a confident non-electron PID is not absorbed into a
shower. It considered *that* segment for *that* shower and refused. The
scanner's mark overrules one named, sourced decision — which is exactly what a
hand scan is for, and is a very different act from working around a hole in the
display. The old message read like the latter.

The scan record was never the problem. `labels-evt169626.json` already carries

```json
"53070": {"kind": "in", "owner": null, "pdg": 2212, "length": 14.09,
          "absorbed_by": "in_other_clusters_A (walk_exclude)",
          "angle": 48.56, "dist": 1.2e-05, "tier": null}
```

**Arm check.** The `absorb` records come from `work-em114b-mcp1k` (probes on)
while the dump on screen is `work-mcp1k-prod0825` (probes off — zero
`SHOWER_ABSORB` lines, which is the debug print being off, not a different
configuration). Since `apply_guard` is a knob state, that had to be settled
before quoting the guard as the reason: shower membership is **identical across
the two arms for all 8 showers** (6/1/1/1/2/1/30/1 segments, same ids), and the
prod0825 dump independently lists 53070 as the event's only `shower_id == -1`
segment. The two arms agree on this event's clustering.

### 25.3 Why the old message was wrong to blame dQ

> *"no dQ estimate is offered because E_est/dQ is not constant between showers"*

The between-shower spread is real — on this event alone the eight showers convert
at 1.22e-5 to 7.20e-5 MeV per collected electron, a factor of **5.9** — but the
sentence points at the wrong thing. The charge is not missing; it is sitting in
the dump. What is missing is a defensible *conversion*, and saying "no dQ
estimate" made it sound like a data gap.

It is also worth recording why the obvious fix was **rejected**: convert the
segment with the target shower's own factor. Within a shower `E_est/dQ` is
constant to 0.04 %, which looks like licence — but it is constant **by
construction**: `E_est` is `kine_charge` split across members in proportion to
`dQ`, so the ratio is identically `kine_charge / ΣdQ` and says nothing about a
segment the reconstruction never converted. For 53070 it would give +107.8 MeV
(537.1 → 644.9, and the π⁰ mass with it), while the same charge under a
neighbouring shower's factor gives anything from 20.7 to 121.7 MeV. And the
reco's PID here is proton at score 100: the code's own converter would use
`KINE_PROTON` (0.35 × 0.95), not the shower pair (0.50 × 0.80) — the same
tens-of-percent axis round 9 exists to expose. The no-fallback rule
(`marks_energy` docstring) stands; it was a decision, not an oversight.

What *can* be said with no conversion at all is the **charge fraction**: the same
`dQ` field on both sides of the ratio. That is the number the panel now gives.

### 25.4 The fix

Six pieces, all presentation.

1. **`seg_dq` / `shower_dq`** — a segment's collected charge from the dump's
   fitted points, and the sum over a shower's members. The claim these rest on is
   pinned in the selftest: the probe's per-member `dQ` equals this sum to
   **5.9e-6 relative over 5 700 member segments across all 98 events**, which is
   the sidecar printing one decimal. That identity is what puts an orphan segment
   on the same scale as a member without inventing a conversion. Deliberately
   *not* `refresh_impact`'s filtered sum (it drops `dQ < 0` points; the two differ
   by ~1 % over a shower).
2. **`dump_shower_id`** — the reco's own assignment, read from the dump so the
   "in no shower" half of the message needs no probe.
3. **`absorb_reason(sid, node)`** — the probe record that names *this* shower.
   `absorb_site` takes `recs[-1]`, which answers "where did it end up"; asked why
   it is *not* in a given shower, last-wins names the wrong shower. 13023 on this
   very event carries a `walk_add` into 13012 then a `direct` into 53069, and 9 of
   the sample's 23 `walk_exclude` segments were absorbed elsewhere afterwards.
4. **`unknown_mark_row` / `unknown_mark_why` / `unknown_mark_note` /
   `unknown_mark_lines`** — one measurement and one explanation, rendered long for
   the π⁰ panel and short for the EM mark list.
5. **`marks_energy` returns `unknown_rows`** beside the existing `unknown`.
   `delta` is untouched — that is the round's safety guarantee, and it is checked
   structurally (`energy_for` and `_pair_sig` must not contain the string
   `unknown_rows`) as well as numerically (§25.6).
6. **The EM mark list says it too**, where the mark is made. The report's shape
   was "I did this in EM clustering and met the objection in π⁰ mode"; both
   panels now read from `marks_energy`, so they cannot drift.

What the π⁰ panel says on evt169626 now:

```
⚠ γ1 · 53070 — a 14.1 cm segment the reconstruction PID'd proton (score 100)
  and put in no shower (the dump gives it shower_id -1). There is no per-segment
  energy for it, so it is not added and γ1 stays at 537.1 MeV — though its
  charge, ΣdQ 1.69e+06 e, is 20% of γ1's own 8.42e+06 e.
  why: the code considered it for this very shower and refused —
  SHOWER_ABSORB EXCLUDE shower_start_seg=53069 seg=53070 pdg=2212 at
  in_other_clusters_A, the straight-long-track guard, which declines a
  confidently PID'd non-electron (PRShower.cxx:722-727, F12 / doc pr/40 round 6).
  Your mark overrules that one decision.
  your mark is kept either way — marks_by_shower and marks_detail record it with
  that reason, and the membership you scanned is what the record is for. What is
  missing is only the MeV: no estimate is offered because charge→energy is a
  per-object choice — this event's own showers convert at 1.22e-05 to 7.20e-05
  MeV per electron, a factor of 5.9. And for a proton-PID'd object the code
  would not use the shower recombination pair either (0.35×0.95 against
  0.50×0.80).
```

and the EM mark list, where the mark is made:

```
53069: 39039 IN, 42042 IN, 43043 IN, 53070 IN  ← scanning
  ↳ 53070 — a 14.1 cm segment the reconstruction PID'd proton (score 100) and
    put in no shower (the dump gives it shower_id -1). Your mark IS recorded,
    with the reason, but there is no per-segment energy for it: ΣdQ 1.69e+06 e,
    20% of shower 53069's own charge, will not reach the π⁰ mass. Why: the code
    considered it for this very shower and refused — the straight-long-track
    guard (PRShower.cxx:722-727).
```

### 25.5 How often this is met

Over the 98 manifest events:

```
segments with no per-segment E_est : 196 of 5830 (3.4%), on 84 of 98 events
walk_exclude records               : 23 segments over 21 events
   of those, still owned by nothing : 14
   pdg of the excluded segment      : muon 12, proton 10, pion 1
orphan marks on the owner's own scan: 1 of 87 marks (evt169626 / 53070)
```

So this is not a one-off; it is a message the scan meets on most events as soon
as a mark lands outside the reconstruction's own membership. (The README's
earlier "177 of 5830" is corrected to 196 — the count of dump segments absent
from `_est_map`, which is what the code actually tests.)

### 25.6 Nothing a record can hold moves

`/home/xqian/tmp/em114r17/ab.py` walks all 98 events and dumps, per shower,
every `marks_energy` delta under both hypotheses, every `energy_for` under all
four switch combinations, and per stored candidate its `_pair_sig` and its full
JSON, plus the live pairing JSON:

```
pre.tsv  : 9698 rows over 98 events   f080d03bfc1821163940d054a52a71de
post.tsv : 9698 rows over 98 events   f080d03bfc1821163940d054a52a71de
cmp -> IDENTICAL
```

Cost of the extra `marks_energy` in `refresh_mark_list`, measured rather than
assumed (`/home/xqian/tmp/em114r17/cost.py`): 0.02-0.09 ms per refresh, of which
`marks_energy` is 0.00-0.02 ms.

### 25.7 Tests

`selftest_em_display.py` 358 → **397**, `selftest_em3d_browser.py` 79 → **86**,
`selftest_repro.py` 98/98 unchanged.

The round-12 check that asserted the old wording is **upgraded, not deleted**:
it now requires the panel to say the segment is in no shower *and* what is at
stake *and* that the mark survives — asserting the old string would have passed
on a message that told the scanner nothing. Also pinned: the probe-`dQ`
identity over the whole sample; that `absorb_reason` does not behave like
`absorb_site`; that with `state["prep"]` blanked the message degrades to what
the dump alone supports and invents no reason; that an OUT mark is not described
as an addition; and a live-data guard that reads every scan tag on disk
read-only and renders each orphan mark found (87 marks over 11 events, 1 orphan,
0 bad).

The browser test makes the mark through the real buttons on evt169626 —
`cand_src` row for 53070, `mark IN` — and reads both panels back off the live
document. Nothing is saved: that event's real record lives in the owner's tag
(M13).

### 25.8 Not done, and left to the owner

- **The MeV is not offered.** §25.3 gives the two candidate conversions and the
  spread that blocks choosing between them. If the owner wants an orphan
  segment's charge counted, it needs its own control and its own decision about
  which factor — and it must **not** be folded into `gamma energy membership`,
  which means "marks with a known `E_est`"; giving that switch a second meaning
  would break the round-12/16b restore logic.
- **`refresh_impact`'s filtered ΣdQ** drops `dQ < 0` points where the probe keeps
  them, so its number is ~1 % from the probe's for the same segments. Reported,
  not changed — it is a different, older readout and not this round's business.
- Two stray files at the repo root remain from earlier mis-redirects and are not
  touched: `angle` (empty, Aug 4) and `sbnd_xin/--quiet` (74 KB of TSV, Aug 20).

---

## 26. Round 18 — an unowned segment marked into a shower is counted *as* that shower

> *"I feel when we added a particle into an EM shower, it should then be treated
> as an EM shower to do the calculation. Of course, it is possible this
> information is not available, in this case, have a reasonable guess, it is fine
> for now. We have the information in the toolkit after the algorithm
> improvement, which will be based on these hand scan."*

The owner's decision after §25 showed them what was being dropped. Round 17
refused to name a MeV; round 18 names one, under its own switch, labelled an
estimate everywhere it appears.

### 26.1 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
python em_display/selftest_em_display.py        # 427
python em_display/selftest_em3d_browser.py      # 90
python em_display/selftest_repro.py             # 98/98

# both readings of evt169626, off the owner's own scan tag:
python /home/xqian/tmp/em114r17/probe18.py

# nothing already saved moves (round-17 HEAD vs this):
python /home/xqian/tmp/em114r17/ab18.py /home/xqian/tmp/em114r17/pre18.tsv  \
       /home/xqian/tmp/em114r17/pre18
python /home/xqian/tmp/em114r17/ab18.py /home/xqian/tmp/em114r17/post18.tsv \
       /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin/em_display
cmp /home/xqian/tmp/em114r17/pre18.tsv /home/xqian/tmp/em114r17/post18.tsv
```

No C++ and no jsonnet are touched, so **no A/B gate is owed** — as in rounds 1-17.

### 26.2 The number, and why this one

`orphan_energy(node, sid)` = `seg_dq(sid) × kine_charge(node) / ΣdQ(members of node)`.

That factor is **the same ratio the probe's own `E_est` decomposition uses for a
member of that shower**, which is the property that makes the result comparable
with the rest of E1 rather than a number in the same units. Checked in the
selftest rather than asserted: applying γ1's factor to a *member's* `dQ`
reproduces the probe's `E_est` for it to **0.0005 MeV** over the members tested.
So a counted orphan is treated identically to a segment the reconstruction had
put in that shower — which is what *"treated as an EM shower"* has to mean here.

It also follows the round-9 hypothesis switch, by construction: the estimate is
handed to the same rescale line an owned segment takes, with the target shower
standing in for the owner the segment has not got. Pinned in the selftest on
γ2 = 22034, which the reco called a track (pdg 211): the counted orphan's ratio
between `as_em` and `as_reconstructed` is exactly
`(0.70 × 0.95) / (0.50 × 0.80) = 1.6625`.

On evt169626, segment 53070:

| | |
|---|---|
| factor (γ1's own) | 6.37725e-5 MeV per electron |
| ΣdQ(53070) | 1 690 300 e |
| estimate | **107.795 MeV** |
| γ1 | 537.1 → **644.9** MeV (→ 668.5 with the other three marks) |
| π⁰ mass, axis convention | 147.7 → **161.3** MeV (both in the code's accept window) |
| π⁰ mass, vertex convention | 148.6 → **162.3** MeV |

### 26.3 What it is not

Said on screen every time the number is shown, not only here. The per-shower
factor is an **average** over that shower's members. It is flat within a shower
only because `E_est` is `kine_charge` split in proportion to `dQ` — constant *by
construction*, not because charge-to-energy is genuinely flat. Recombination runs
with dQ/dx, so a dense track-like segment does not convert like a sparse shower;
and 53070 is exactly that case, PID'd proton at score 100, whose own pair
(0.35 × 0.95) is not the shower pair (0.50 × 0.80). Between showers the factor
spans 1.22e-5 to 7.20e-5 MeV/e on this event alone, a factor of 5.9.

This is the "reasonable guess" the owner asked for, and it is recorded as such:
`energy_orphan_detail` keeps the segment, the kind, the energy, the `est_factor`
and the shower it was borrowed from, and the estimated rows inside
`energy_marks_detail` carry `estimated: true`. Absent means exact, which is every
row written before this round. A later reader — including the toolkit-side
algorithm this scan is meant to inform — can subtract it exactly.

### 26.4 Its own switch, deliberately

`marked segments no shower owns` — `leave them out` / `count as part of the
shower`, default **count**.

It is **not** a third state of `gamma energy membership`. That switch means
"marks with a known `E_est`", and rounds 12/16b key its restore on
`energy_includes_marks`; giving it a second meaning would silently re-price every
record already saved with marks on, the moment it was re-opened. As its own key,
`energy_includes_orphans` follows the idiom rounds 9, 12, 14 and 15 all used:
**absent means the pre-round default** (off — what every record written before
this round was saved with), while a fresh event takes the new default.

The restore is **total**, for round 16b's reason: with the default on, a
one-way restore would re-price every older record. `load_label` sets the switch
from the record including *off*, and `on_pio_load` does the same per stored row.

`_pair_sig` gains both gammas' flags through `bool()`, so a pairing re-priced by
counting an unowned segment can be stored beside the one that did not count it,
and no pre-round-18 row looks different from an identical new one. The stored-row
table gains a fourth flag, `gN leaves out an unowned segment (est +107.8 MeV)`,
kept separate from *stale* and *ignores your marks* for the same reason those are
separate from each other, and reported by `update to today's numbers`.

### 26.5 Nothing already saved moves

`/home/xqian/tmp/em114r17/ab18.py` dumps, per labelled event, what the scanner
sees the moment it re-opens — the restored switches, both gamma energies, both
masses, every stored candidate's energies and full JSON — plus the round-12
off-path arithmetic over every shower (`marks_energy` under both hypotheses and
`energy_for` under all four switch combinations, with the new switch off):

```
pre18.tsv  (round-17 HEAD) : 9796 rows over 98 events   4d3b3c069e8ac52b5dac44d6d3f9b70f
post18.tsv (round 18)      : 9796 rows over 98 events   4d3b3c069e8ac52b5dac44d6d3f9b70f
cmp -> IDENTICAL
```

And on the owner's own scan: **44 gamma records read, 0 carry
`energy_includes_orphans`**, so every one of them restores to `leave them out`
and re-opens at the mass it was saved with. Exactly 1 of the 87 marks on disk is
on an unowned segment (evt169626 / 53070), so this changes what a **new** save or
a **re-save** records, and nothing else.

**The consequence to state plainly, because it is the first thing the scanner
meets:** opening evt169626 — the event this was reported on — shows the switch
**off** and γ1 still at 537.1 MeV, because its record predates the switch. That
is the safety working, not the feature failing. It applies to all 18 events
already scanned. Taking the estimate on one of them is deliberate: flip *marked
segments no shower owns* to `count as part of the shower` and save. A fresh,
unlabelled event opens with it on. Confirmed in the live app: `evt169626 ->
leave them out`, `evt84229 -> count as part of the shower`.

### 26.6 Tests

`selftest_em_display.py` 397 → **427**, `selftest_em3d_browser.py` 86 → **90**,
`selftest_repro.py` 98/98 unchanged.

One bug the headless test could not have caught, found by the live smoke on
5017: the shared `on_change` loop for the energy switches calls `refresh_kine()`
only, and this is the one switch the EM-mode **mark list** also reads — so
flipping it left the note under a mark saying "will not reach the π⁰ mass" about
charge that was by then being counted. `orph_mode` now has its own handler that
rebuilds both. The headless test had been calling the two refreshes by hand,
which is exactly what hid it; the new check drives the real callback instead.

The round-12 and round-17 checks that describe the *uncounted* behaviour are
**pinned to the switch off, not deleted** — that is still exactly what off has to
mean, and it is the check that would catch the two paths merging by accident.
New: the factor identity against the probe's `E_est`; the `as_em` rescale
reaching a counted orphan; the panel in both states; the EM mark list in both
states; the record's three new keys and the `estimated` marker; that the two
readings are different `_pair_sig`s so both can be stored while an absent key
still reads as off; the fourth table flag; and the restore — a record saved
counting it comes back counting it, one written before this round turns it back
off and shows the mass it was saved with, and nothing on the real scan re-prices.

The browser test flips the switch on the live page after marking 53070 through
the real buttons and reads `E1 537.1` / `E1 644.9` back out of the document.

### 26.7 Left open

- The estimate borrows the *target shower's* factor. A per-segment conversion
  (the reco's own recombination at that segment's dQ/dx) would be better and is
  not available from the dump; the owner's note says the toolkit will have it
  after the algorithm improvement this scan is feeding. The record keeps enough
  to redo the arithmetic when it does.
- `refresh_impact`'s ΣdQ still uses its own filter (it drops `dQ < 0` points
  where the probe keeps them, ~1 % over a shower). Reported in §25.8, and its
  **numbers are unchanged**. Its closing sentence was rewritten, though: it sits
  directly above the mark-list note and is about the same segment, and until this
  round *"ΣdQ is … not its MeV value"* was true of everything on screen. It now
  points at the π⁰ panel and distinguishes the exact `E_est` from the estimate,
  instead of reading as a denial that the MeV two lines below exists.
- **With no probe sidecar `_est_map()` is empty, so every mark would land in
  `unknown` and — with the switch on — become an estimate.** It cannot happen on
  this sample (`has_probe = 1` for all 98 manifest events), so the protection is
  the manifest, not the code. No guard was added; it is stated here instead.

### 26.8 Refresh cost

Measured on the same two events round 16 used, since `cand_unused_orphans` calls
`marks_energy` twice per gamma per stored row and each call now reaches `seg_dq`
and `shower_dq` (`/home/xqian/tmp/em114r17/cost18.py`):

```
evt281485  3 stored rows   refresh_kine 0.25 ms   refresh_pio_cands 0.21 ms
evt76346   1 stored row    refresh_kine 0.49 ms   refresh_pio_cands 0.29 ms
evt169626  0 stored rows   refresh_kine 0.39 ms   refresh_pio_cands 0.04 ms
```

Round 16 measured 0.24 ms and 0.48 ms on the first two, so the round-18 work is
inside the noise.

---

## 27. Round 19 — the coverage gap: 141 beam events the sample never contained

**Owner's question.** *"Have I included all the EM shower events (E > 100 MeV)
in the numuCC samples (1000 + 2000)?"* — **No.** Of the **171** PR-evaluated
mcp1k+mcp2k events whose leading EM shower clears 100 MeV, **30** are in the
98-event sample and **141** are not, including **all 79** of the numuCC-EM list
pr/113 delivered. (171 out of a 1366-event pool, not out of 3000 — §27.2.)
Those 141 are now prepared and served on **port 5018** under tag
`emscan-0828-beam141`. The 5017 scan is byte-identically untouched.

### 27.1 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# 1. name the gap (read-only: reads the pr/113 census and the live manifest)
python scripts/pr114c_unscanned_em.py       # -> docs/pr/pr114c-beam-em100-unscanned.index.txt

# 2. stage 2 probes for the 141, fresh arms (~9 min, PR_JOBS=6)
./em_display/run_em114c_probe.sh            # -> work-em114c-{mcp1k,mcp2k}

# 3. epoch-matched Bee clouds, LOCAL ONLY -- not uploaded (owner-gated)
python em_display/prep_em_scan.py --sample-index docs/pr/pr114c-beam-em100-unscanned.index.txt \
    --prepdir em_display/emprep-c --no-bee-index --bee-build bee/em114c \
    --out /home/xqian/tmp/em114c-beebuild.tsv

# 4. probes + manifest.  --bee-prefer MUST name em114c (see 27.4)
python em_display/prep_em_scan.py --sample-index docs/pr/pr114c-beam-em100-unscanned.index.txt \
    --prepdir em_display/emprep-c --bee-prefer em114,em114b,em114c \
    --parse-probes work-em114c-mcp1k work-em114c-mcp2k \
    --out em_display/em114c-manifest.tsv

# 5. gates
python em_display/prep_em_scan.py --out /home/xqian/tmp/em114-regen.tsv   # knob-off
cmp /home/xqian/tmp/em114-regen.tsv em_display/em114-manifest.tsv         # -> identical
python em_display/selftest_repro.py         # 98/98, the pr/114 record
python em_display/selftest_repro_c.py       # 141/141, this round

# 6. serve
./em_display/serve_em_display.sh 5018 --scan-tag emscan-0828-beam141 \
    --manifest $PWD/em_display/em114c-manifest.tsv --prepdir $PWD/em_display/emprep-c
```

Toolkit anchor `8d93260d` (**unchanged — no C++ was touched this round**),
2026-08-28.

### 27.2 The audit

**The pool first, because "all" needs a denominator.** The two arms hold
**1000 + 2000 = 3000** processed events (`work-mcp{1,2}k-prod0825/pr_evt*`,
and 3000 rows of `nusel-events.tsv`). Only **1366** of them — mcp1k 461,
mcp2k 905 — wrote a `calib-pr-evt<ID>.json`, which is what it means for
`TaggerCheckNeutrino` to have selected and evaluated a main cluster. That 1366
is the pr/113 census denominator (`pr113_topology_census.py:18`, `n_calib`), and
it is the pool every number below is out of.

The other **1634** are out of scope *and cannot be brought in by this method*:
with no main cluster evaluated there is no `showers[]` array, so there is no
reconstructed EM shower to threshold at 100 MeV and nothing for the display to
draw. They are not a set of missed EM events — they are events this
reconstruction pass declined to reconstruct. Finding EM showers among them is a
different question (a selection-efficiency one) and would need truth or a
re-run, neither of which exists here: all four prod0825 arms are data (pr/113
§0). Worth noting the pre-PR bundle verdict disagrees — 1501 of the 3000 carry
`event_label == nu-candidate` — but the two pools are not joinable (pr/113 §6),
so it does not move the 1366.

Within that pool: `docs/pr/pr113-emshower-sample.tsv`, rows with
`sample ∈ {mcp1k, mcp2k}` and `em_max ≥ 100 MeV`, joined against
`em_display/em114-manifest.tsv` on `(sample, event)`:

| pr/113 bucket | definition | n | in the 5017 scan | unscanned |
|---|---|---:|---:|---:|
| `is_numucc_em` | muon primary ≥ 30 cm, `em_max ≥ 100` | 79 | **0** | **79** |
| `is_ncpi0_reco` | no muon, γγ pair | 27 | 27 | 0 |
| `is_nuecc_reco` | no muon, no pair, vertex-rooted e | 42 | 2 | 40 |
| `other_em` | *none of the three* | 23 | 1 | 22 |
| **total** | | **171** | **30** | **141** |

(Pool 1366 PR-evaluated of 3000 processed; 1195 of the pool have no EM shower
reaching 100 MeV.)

**The gap is by construction, not by accident.** §2 built the sample as the
NCπ⁰ list (46) + the curated nueCC48 arm (48), and `scan_sample()` narrows the
pr/113 nueCC list with `if r[0] == "nuecc48"`. So the numuCC-EM list — the one
`pr113_topology_census.py:35` annotates `<-- the owner's ask` — was never
wired in, and neither were the beam rows of the nueCC list. The 30 beam events
on screen are the 27 NCπ⁰ ones plus 3 of the 4 round-6 owner adds.

**`other_em` has no pr/113 name and appears on no delivered list.** 23 beam
events carry a ≥ 100 MeV EM shower with no muon primary ≥ 30 cm, no γ pair and
no vertex-rooted electron — they fall through all three verdicts of the
priority ladder. They are a hole in pr/113's own selection, found only by
asking the coverage question this way round, and 22 of them are in this scan.

**One caveat on the 79.** `is_numucc_em` inherits pr/113's 30 cm `mu_len` floor
(raw flag 1025/1366 beam events, floored 791/1366). A short-muon event with a
large shower is not in the 79 — it lands in `nuecc` or `other_em`, both of
which this scan also covers, so at 141 events the floor no longer selects
anything away.

### 27.3 The sample as served

141 events, `em_display/em114c-manifest.tsv`, `origin` carrying the bucket:

| | n |
|---|---:|
| events | 141 |
| `numucc_em` / `nuecc` / `other_em` | 79 / 40 / 22 |
| mcp1k / mcp2k | 52 / 89 |
| …with a probe sidecar | **141** |
| …with an epoch-matched 3-D charge cloud | **141** |
| …with a reconstructed π⁰ group | 25 |
| …with a lossy shower join | 10 |
| …with an external Bee link | 0 — the set is built, **not uploaded** |

`em_max` over the 79 numuCC rows: min 103.5, median 195.2, max 2498.2 MeV;
41 in tier 100, 38 in tier 200.

### 27.4 Two traps this round had to step around

**The manifest is the sample definition of a live scan.** `prep_em_scan.py`'s
`--out` *defaults* to `em_display/em114-manifest.tsv` — the file the 5017
display reads and the file 97 saved labels in `em_labels/emscan-0827/` join
against. Every invocation here passes `--out` explicitly. The gate is step 5 of
the repro: with no flags the script still reproduces the 98-event manifest
**byte-identically** (`cmp`, not a diff of counts), which is what makes
`--sample-index` and `--bee-prefer` default-OFF in the CLAUDE.md §1 sense.

**A new Bee round that is not in `prefer` gets captured by an older epoch.**
This is the failure `bee_index.__doc__` describes, and it was live here: before
`bee/em114c` existed, 50 of the 141 resolved to an already-uploaded set and
**46 of those were `prod0813`** — two reconstruction epochs behind the prod0825
skeleton the display draws over the cloud. Both the 3-D charge cloud and the
click-through link would have shown a different answer to the one being
scanned, and both render fine. Hence `--bee-prefer em114,em114b,em114c` in step
4, and hence all 141 rows now read `em114c/em114c-<arm>`.

### 27.5 What was verified

- **`selftest_repro_c.py`** (a fork of `selftest_repro.py` pointed at the new
  arm; the pr/114 original is untouched): **all 141 events identical** to
  prod0825 on `main_vertex`, `kine`, `tagger`, `showers`, `segments`. The §3
  guarantee now holds for this sample too, so the display keeps reading the
  prod0825 dumps and takes only probe text from `work-em114c-*`.
  Membership: 1987 showers, 12 lossy joins, **12 of 12 repaired** by the probe;
  axis from the probe's `dir15` for 1982 of 1987.
- **The 5017 scan is untouched**: `em114-manifest.tsv` byte-identical (mtime
  still 2026-08-27 16:20), `em_display/emprep/` still 98 sidecars,
  `em_labels/emscan-0827/` still 97 labels, `selftest_repro.py` still 98/98,
  and the server still answers. Nothing was written into an existing label dir,
  arm or Bee round (M13).
- **The 5018 app drives**, checked headlessly through the Bokeh model API: no
  console errors, `event_select` holds 141 options (`evt54341` … `evt500083`),
  the charge cloud reaches the browser, the banner reads *"Bee set built but
  not uploaded (em114c/em114c-mcp1k) — the 3-D cloud below IS this set"* and
  *"probe sidecar loaded"*, and the scan-status band reads *"not scanned yet"*
  under tag `emscan-0828-beam141`.
- **Both arms, not just the first.** 89 of the 141 are mcp2k and the landing
  event is mcp1k, so a working first screen proves only half the sample and
  none of the navigation. Driving `event_select` to `evt47696` switches the
  event, redraws 4211 cloud points, and flips the banner to
  `em114c/em114c-mcp2k` with the probe still loaded — so the second zip yields
  points and the event-switch path works.
- **The new `origin` values are inert.** This manifest's `origin` carries
  `numucc_em` / `nuecc` / `other_em` where pr/114's carried `reco` / `ncpi0` /
  `curated`. The viewer never branches on it: `:3884` prints it in the header
  and `:4230` stores it in the saved label record. So the bucket travels into
  every label as provenance and changes no behaviour.

### 27.6 Not done

- **The Bee upload.** `bee/em114c/em114c-mcp1k.zip` (25 MB) and
  `em114c-mcp2k.zip` (44 MB) are built and **not uploaded** — outward-facing
  and owner-gated (CLAUDE.md §5.6). The 3-D cloud inside em_display does not
  need them; only the external click-through link does. To publish:
  `./upload-to-bee.sh bee/em114c/em114c-<arm>.zip`, save the printed URL as
  `bee/em114c/em114c-<arm>.url`, then re-run step 4.
- **No filtering of the 141 by bucket.** The `origin` column carries
  `numucc_em` / `nuecc` / `other_em`, so a narrower scan is a one-line
  re-filter of the index, not a rebuild.
- **The Bee upload is the only thing still outstanding.** Everything else in
  this round is committed: `scripts/pr114c_unscanned_em.py`,
  `docs/pr/pr114c-beam-em100-unscanned.index.txt`,
  `em_display/{em114c-manifest.tsv,selftest_repro_c.py,run_em114c_probe.sh}`,
  and edits to `em_display/prep_em_scan.py`, `em_display/README.md` and this
  doc. `run_em114c_probe.sh` went in with **`git add -f`** — `*.sh` is
  gitignored there (M9), so a plain `git add -A` silently drops step 2 of the
  repro.
- **`em_display/emprep-c/` (141 sidecars) is NOT committed, and that differs
  from its 98-event counterpart.** `em_display/emprep/` *is* tracked (98 files,
  force-added past the `*.json` ignore), so the precedent says the scan's probe
  input gets frozen with the scan. These 141 are left out as a build product —
  regenerable by step 2 — but the inconsistency is deliberate-by-omission
  rather than decided, and is the owner's call: 2.0 MB to make the round
  reproducible without re-running the probe. `bee/em114c/` (the zips) stays out
  either way.
