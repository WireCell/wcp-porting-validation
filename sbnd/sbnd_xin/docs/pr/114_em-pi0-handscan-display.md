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
