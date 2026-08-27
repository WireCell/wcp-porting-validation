# EM shower clustering & π⁰ hand-scan display

A Bokeh display for **validating and improving** the post-vertex EM shower
clustering and π⁰ reconstruction: mark segments in or out of a shower, build a
π⁰ from scratch and watch the mass come out, and click through to Bee.

Full write-up: [`../docs/pr/114_em-pi0-handscan-display.md`](../docs/pr/114_em-pi0-handscan-display.md).
Sample and coverage audit it grew out of: [`../docs/pr/113_em-shower-pi0-long-muon-coverage-audit.md`](../docs/pr/113_em-shower-pi0-long-muon-coverage-audit.md).

## Quick start

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin
./em_display/serve_em_display.sh 5021 --scan-tag myscan1
```

then from a laptop

```bash
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
    -L 5021:localhost:5021 <user>@wcgpu1.phy.bnl.gov
# open http://localhost:5021/em_display_viewer
```

The keepalive flags matter: a bare `ssh -L` gets reaped during the long pauses a
hand scan is made of, and Bokeh's JS does not auto-reconnect (doc pr/88).

## What is on the screen

**Two columns**, under a header band. The **left** is the view and its controls;
the **right** is everything you read or click while looking at it. Total width
1980 px by default, and the **3-D panel size** selector (620 / 760 / 900 / 1100)
takes the left column up on a bigger screen.

**Left — two tabs.** **3-D** (the default) and **2-D projections** (X-Y, Y-Z,
X-Z stacked 2-over-1, active volume and cathode in red). See
[The 3-D view](#the-3-d-view) below. The projections stay because the free-space
manual x/y/z pin needs them: two panels each give two of the three coordinates,
and a rotatable canvas gives a *ray*, not a point.

**Right — the acceptance plot.** Angle to the shower axis vs distance from
the shower start, with the three pass-1 tiers drawn as dashed steps. Every
segment is a dot; a dot **under** a step is inside that tier. **Squares are the
segments already in the selected shower**, circles everything else, green/red the
ones you marked.

By default the axes are **scaled to what is being compared** — the members plus
anything you marked, with 30 % headroom — not to the 220 cm × 90° box the gate
needs. Members occupy a corner of the full box (on evt64591's shower 78025 they
sit inside the first 8 % of the axis), which made "how does this piece compare
with the ones already in" unanswerable at a glance. **zoom to this shower** turns
it off and restores the full gate box; anything cropped out is counted in the
line underneath, never silently dropped.

The **comparison line** below the plot is the same thing in words, and it is what
aggregates over events: the member spread in distance and angle, then each marked
segment with its distance, angle, pass-1 tier, `absorbed by`, and whether it sits
inside the member spread.

This is deliberately *not* a cone drawn over the projections. A 3-D cone does not
project to a cone, so a wedge on X-Y would be decorative and would invite exactly
the wrong reading. Distance and angle are the two quantities
`NeutrinoShowerClustering.cxx:1310-1312` actually tests, so this panel is the
gate itself with nothing approximated.

**Right — the shower table.** One row per `showers[]`, sorted by `kine_charge`.
Click a row to select that shower: the view **switches to the 3-D tab and frames
it** (with `frame the shower`, the default). The leftmost column is the shower's
**colour swatch** — the same colour its segments are drawn in.

**Right — dim these showers away.** Pick any number of showers here and their
segments fade almost out in 3-D and in the projections, and drop out of the
candidate table. The scan question is always "does this piece belong to *that*
shower", and on a busy event the other showers are the noise in that judgement.

### One colour per shower

Segments are coloured by the shower that owns them, so two pieces of one shower
look alike and two pieces of different showers do not. Segments no shower claims
stay neutral grey — they are what the scan is deciding about, and giving them a
hue of their own would read as membership. `colour by segment` restores the old
per-segment palette when you need to tell two adjacent segments apart.

**The colour follows your marks.** Mark a segment `IN` and it is repainted in
that shower's colour immediately; mark a member `OUT` and it drops back to grey.
So the colours show the clustering *as you are redefining it*, while the
`in shower` column keeps saying what the reconstruction did. Note that a segment
the reco left as a one-segment shower of its own — which is what most orphan
stubs are — starts in **its own** colour, not grey; grey means no shower claims
it at all.

The palette walks ten distinct hues before it uses their light twins: Bokeh's
Category20 is ordered as hue *pairs*, and taken raw it gave a π⁰'s two gammas two
shades of the same blue — the one comparison that must not be ambiguous.

### The views are brushed together

The candidate table, the acceptance plot and the 3-D view are three views of one
list of segments. Clicking in any of them selects the same segments in the other
two and draws the cyan halo. The view you clicked is authoritative — the other
two are rewritten from it, so a stale selection in a panel you are not looking at
cannot leak into what the mark buttons act on.

**Then one of two panels**, chosen by the mode switch.

## The 3-D view

Rotatable and zoomable like Bee, with the same charge cloud under it — but
inside em_display, so every label control works off it.

| gesture | effect |
|---|---|
| **drag** | rotate |
| **shift + drag** | pan the view |
| **wheel** | zoom |
| **tap** | pick (see below) |
| toolbar **Box Select** | select segments — and it *suspends rotation* while it is on |

There is no separate "rotate mode / select mode" switch: picking Box Select (or
Pan) in the toolbar is what steps rotation aside, and un-picking it brings
rotation back.

**Depth is shown by fading, not perspective.** The projection is orthographic —
nearer points are drawn more opaque and slightly larger. That is a deliberate
choice, not a shortcut: an orthographic map composes exactly with Bokeh's own
zoom, keeps every glyph in ordinary data space (which is what keeps tap, box
select and hover working), and does not distort the angles this scan is about.

**Framing is rotation-proof.** The view is set from the 3-D bounding sphere of
the *reconstruction* — fit points, vertices, shower points — never from the
projected extent. So an elongated track swinging from broadside to end-on can
neither balloon out of frame nor shrink to a dot, and all zoom is yours. The
**charge cloud does not set the frame** by default: a cosmic-laden cloud spans
the whole TPC and would leave the neutrino a speck. `refit` re-frames the
current choice, and there are three:

- `frame the shower` (**default since round 5**) — just the selected shower (or,
  in π⁰ mode, the two assigned gammas) **plus anything you have marked into it**,
  since that is what you are judging. With it on, picking a row in the table also
  frames it: on evt 64591 that is R 300 cm → 32 cm. With nothing selected it
  falls back to `frame the reco`.
- `frame the reco` — everything the reconstruction put in the event;
- `frame the cloud` — the charge cloud as currently filtered.

The other two modes deliberately do **not** re-frame on a table click — only the
default moved. If you mark a segment that falls outside the current view the
status line says so and `refit` will reach it; the camera is never moved under
you on a mark, because that would throw away your zoom mid-judgement.

**Presets.** `x-y`, `x-z` and `z-y` reproduce the 2-D panels exactly, so if you
lose your bearings you can step back to a view you already trust and rotate out
of it again. (`z-y` is named honestly: there is no roll, so a *z*-horizontal Y-Z
view is not reachable — that preset shows the same plane with the axes swapped.)

**A tap does one of seven jobs**, set by `a tap in 3-D does`. Every one of them
works from a box select too, so they all bulk-apply.

| action | what a tap / box does |
|---|---|
| `select segment(s)` | leaves a selection for the mark buttons; the cyan halo shows it |
| `mark IN` / `mark OUT` | marks on the click itself, no trip to a button |
| `toggle IN / OUT / clear` | cycles the same segment on repeated clicks |
| `orbit around it` | re-centres the camera on the clicked point, keeping your zoom |
| `fill x / y / z` | writes the point's real coordinates into the manual boxes |
| `make it the pi0 vertex` | sets the π⁰ decay vertex there and switches to `manual` |

A single click in 3-D is a ray; every action anchors it on a real **fitted
point**, which is what makes it a position. Changing the action drops any
standing selection — Bokeh only fires a selection change when the index list
actually changes, so without that the first gesture after a switch would do
nothing.

**Reconstructed vertices are tappable too** (for `make it the pi0 vertex`, and
for orbiting), but only by a *tap*, never by a box. A vertex has no segment id,
so a box that swept one up would have nothing to mark — the two tools carry
different renderer lists so that cannot happen rather than merely not happening.

**How to read a segment** — the legend under the controls, and the layer order
that makes it work:

| band | meaning |
|---|---|
| soft yellow, 9 px | in this shower, per the **reconstruction** |
| green / red, 13 px + a dashed repeat on top | **you** marked it IN / OUT |
| cyan, 17 px | selected — the next *mark* button hits this |
| blue / red, 11 px | gamma 1 / gamma 2 members (π⁰ mode) |

The widest band is underneath, so they stay concentric and every combination
reads at once: yellow inside green = a member you confirmed, yellow inside red =
a member you are taking out, green with no yellow = something you are adding.
`dim what is not in this shower` fades the rest; it is **off by default**,
because fading what is not in the shower fades the segments you are deciding
about.

### The charge cloud, and which frame it is in

The cloud comes from `../bee/<round>.zip` — the same file the Bee link opens,
so the panel and the Bee page show the same reconstruction.

**It is filtered to the neutrino candidate by default.** `clustering-global`
holds every cluster in the readout and most of them are cosmic muon: over this
sample the candidate is a median **18.6 %** of the cloud (median 6 135 points
of 33 868; worst event 30 323 of 81 814). A cloud cluster is kept when at least
5 reco points have their nearest cloud point, within 2 cm, in it — the dump is
the candidate by construction, so whichever clusters the reconstruction lives on
*are* the candidate's. The two cluster numberings do not meet, so this is matched
in space, not by id.

It is checked by coverage rather than by the reduction: over all 94 events the
largest shower's fitted points stay on a kept cluster with a minimum of
**0.9966** (median 1.0000). `all clusters` puts the cosmics back whenever you
want them.

The filter runs **before** `max points` decimates, so the budget is spent on the
candidate and not on the whole readout, and the readout names three numbers —
drawn, of candidate, of total — plus how many clusters of how many were kept.

**`clustering-global` is the default and `img-global` is not.** They are not the
same frame. `img-global` is dumped pre-pipeline, before the corrections the
reconstruction works in, and per cluster it can sit up to ~121 cm from the
skeleton drawn over it (doc pr/13). It is offered, with a red warning, because
it is occasionally what you want to look at — but never as the default. That the
dump and the corrected layers really are one frame is measured, not assumed:
the dump's fit points land on the zip's own `track_fit-global` layer to a
**median 0.0004 cm**, and the selftest pins it.

**The zips are gitignored**, so a fresh clone has the display but not the cloud.
The panel then draws the skeleton and says so in a banner.

## EM mode

Select a shower, then mark any segment `IN` / `OUT` / `?` from the candidate
table, by selecting dots in the acceptance plot, or by tapping / boxing straight
in the 3-D view. The shower's axis is drawn as an arrow, its members are haloed,
and your marks are drawn green/red.

### A mark belongs to a shower

**Every mark is recorded against the shower selected when you made it**, and with
no shower selected the mark is **refused** rather than filed somewhere. One event
can hold marks for several showers at once; the halos show only the shower you
are scanning, and **marks in this event** underneath the candidate table lists
all of them with the shower each belongs to.

A segment marked `IN` against **two** showers is a contradiction, not an
opinion — it is called out in that list with the pass-1 numbers that decide it
(distance, angle, tier, and the `ellip` tie-break the code itself uses at
`NeutrinoShowerClustering.cxx:1314-1315`), and again at the save. The record is
still written: it is yours, not the tool's to veto. Unmark it on the losing
shower and save again.

This is not a nicety. Until round 5 marks were one flat `{segment: in/out}` per
event and the record named a single `em.shower` — whichever row happened to be
selected when *Save* was pressed. A mark made while shower A was up and saved
after the table moved to B was written against B with nothing to say otherwise,
and **assigning a π⁰ gamma slot moves the table selection**, so the π⁰ workflow
reaches that state on its own. **Opening a round-4 label still works** — its
marks are attributed to the shower the file named — and the banner says so, in
red, so the attribution is a prompt rather than a silent assumption. Doc pr/114
§13.2 works one such case through.

If a segment you mark falls outside the current frame, the message says so and
`refit` will reach it: the frame is the shower **plus what you marked**, since
that is what you are judging. The camera is not moved under you on a mark.

The saved record carries `marks_by_shower` and, alongside it, `marks_detail`:
per marked segment its distance, angle, pass-1 tier and ellipsoidal rank against
the shower it was marked for, plus that shower's own member spread. Those are the
numbers a gate is cut on, measured at save time so each label is self-contained
and a later fit needs no re-derivation from the dump.

The candidate table's columns:

| column | meaning |
|---|---|
| dist / angle | to the selected shower's **start point** and **axis** |
| pass-1 | which of the three tiers accepts it — or `-` |
| ellip | the ellipsoidal ranking metric (40 cm long, 5 cm across) that decides which shower claims a segment several accept |
| in shower | which shower currently owns it |
| **absorbed by** | **which pass put it there**, from the code's own probe |

`absorbed by` is the column this display exists for. It is not inferred — it is
the `site=` tag the clustering itself printed at the moment it absorbed the
segment. Fifteen distinct values occur over the sample; the census is below.

**The pass-1 tiers are pass 1 only — read the plot in both directions.** The
gate the steps draw (`pass3_cone`) is the single largest absorber — **41 % of
4030 absorptions** over the 94-event sample — but not the only one:

| absorbing pass | share |
|---|---|
| `pass3_cone` (the steps) | 41.3 % |
| `pass4_angle` (`:1964-1967`, different constants) | 20.8 % |
| `in_main_cluster` | 17.7 % |
| `from_vertices` | 10.1 % |
| 11 others | 9.9 % |

So **above every step ≠ rejected** — it may have come in through another pass.
And **below a step ≠ absorbed**: `shower_cone_absorb_guard` is SBND-ON and
declines a confidently-PID'd non-electron straight track longer than 50 cm
(`:1336-1351`, pr/93 Cause D). The `absorbed by` column is the authority; the
plot is the geometry.

The **verdict** includes `vertex-bad (undecidable)`. Use it. If the neutrino
vertex is wrong the in/out question has no answer, and recording that is worth
more than a guess. It also includes **`is an EM shower (reco PID wrong)`** — the
inverse of `not an EM shower`, for a track- or muon-PID'd object that should have
been a gamma. New verdicts are **appended**, never re-ordered: a label stores the
verdict string and is read back by index.

### Energy does NOT follow a PID correction

The π⁰ panel shows, per gamma, **which recombination its energy was converted
with** and what the same charge gives under the other hypothesis. This matters
because re-labelling an object in the scan does not move its energy:
`kine_charge` was fixed upstream by

```
E = Σ_p(w_p Q_p)/Σw / recom / fudge × w_value × 1e-6     NeutrinoEnergyReco.cxx:188
```

and which `(recom, fudge)` pair is used comes from `Shower::get_flag_shower()` —
`kShowerTrajectory || kShowerTopology || |pdg| == 11`, evaluated on the **start
segment only** (`PRShower.cxx:1460-1464`). So:

| object | recom | fudge | 1/(recom·fudge) |
|---|---|---|---|
| track-flagged | 0.70 | 0.95 | 1.504 |
| shower-flagged | 0.50 | 0.80 | 2.500 |
| \|pdg\|==2212 | 0.35 | 0.95 | 3.008 |

A gamma the reco flagged track-like therefore carries **1.66× less** energy than
the identical collected charge in a shower-flagged one. The panel quotes the
π⁰ mass with the track-flagged gammas promoted — **only those**, because the mass
goes as √(E₁E₂) and flipping both cancels exactly. These are the C++ defaults;
SBND overrides none of them (`wct-pr-perevt.jsonnet:674-689`).

The record carries `particle_id`, `flag_shower`, `kine_hypothesis` and
`kine_charge_other_hypothesis` on `em.reco` and on each `pio.gammas[]` slot, so
a "PID wrong" verdict is checkable later against what the reco actually thought.

## π⁰ mode

1. Select a shower, hit **selected shower → gamma 1**; repeat for gamma 2.
2. Each gamma's start point defaults to the reco's `showers[].start`. To override:
   turn on **tap fills x/y/z**, tap the same point in **two** projections (each
   panel gives two of the three coordinates), then **snap start to nearest fit
   point** — which puts the start on a real fitted point rather than a free
   position in space.
3. Choose the decay vertex: the main vertex, a **back-projection** of the two
   gammas, or a manual point.
4. Read the mass.

Two masses are shown side by side, and that is the point:

- **axis convention** — the angle between the two showers' own axes;
- **vertex convention** — the angle between the two vertex→start chords.

The code itself uses different direction recipes for the mass it stores
(`:3771`) and the angle it stores (`:3830`), and they do not close. Seeing both
is how you tell a genuine π⁰ from a bookkeeping artefact.

### There is no π⁰ verdict — the correction *is* the judgement

Retired in round 5d. The workflow is "start from the code's reconstruction, then
correct it", and the record already holds both sides independently:

| the code's answer | yours |
|---|---|
| `pio.reco_groups` (the accepted `pio_id` pairings and their masses) | `pio.gammas` (the pair you assigned, with starts, energies, members, axes) |
| `pio.reco_kine` (the whole `kine_pio_*` block) | `pio.vertex` + `pio.vertex_how` |
| | `mass_axis_convention` / `mass_vertex_convention` |

The difference between the two columns is the judgement, and unlike a verdict it
is quantitative. The verdict also had no anchor: the panel shows *three* pairings
(`pio_id`, `kine_pio_*`, yours) and the verdict named none of them, so on an
event where your pair differs from the reco's — evt166870 — "pi0 correct" and the
gamma slots could contradict each other with nothing to notice.

**Known loss, not papered over:** *"there is no π⁰ in this event"* cannot be said
as a correction, because empty gamma slots are also what "not scanned" looks
like. No replacement has been invented; put it in the note until there is one.

A π⁰ verdict written by an older build is **read and preserved** — re-saving such
an event will not delete a past judgement — but no new one is written.

**`vertex_how` is what says whose vertex it is.** `main_vertex` and `backproject`
are the reconstruction's (the first is also the *default*, so it cannot be
distinguished from never touching the control); `manual` is yours. Clicking a
point in 3-D with *make it the pi0 vertex* always lands as `manual`, even when
the point you click is a reconstructed vertex.

### Two things the panel keeps apart on purpose

**`showers[].pio_id` is the pairing. `kine_pio_*` is a BDT feature.** They can
name *different pairs*. On ncpi0 evt21073 the accepted groups are
(60081+31023, 127.2 MeV) and (11008+63100, 111.2 MeV) — while `kine_pio_*`
reports E1 = 680.2 (from 60081) and E2 = 104.7 (from 63100), a third pairing no
reconstruction accepted, mass 207.25. `kine_pio_*` is filled by a separate
highest-energy scan over *all* candidate pairs, accepted or not. They are drawn
in separate blocks and never merged.

**The π⁰ vertex is usually already reconstructed.** When `id_pi0_without_vertex`
accepts a pair it *overwrites* the main vertex with the back-projected decay
point (`NeutrinoShowerClustering.cxx:4338-4340`, tell-tale `fit.dQ == 0`). So the
back-projection here mirrors code that already ran; the job is to judge it, not
to invent it. ("without vertex" means the gammas are **detached**, conn type 3 —
not that the event lacks a neutrino vertex; that case returns early at `:3961`.)

## Membership is 99 % the dump and 100 % the probe

`segments[].shower_id` stores **one** shower per segment, so when two showers
overlap the loser's members vanish from the join and it looks empty rather than
nested. Measured over the 94-event sample: **15 of 1567 showers (1.0 %)**, all of
them EM. Worst cases are ncpi0 evt84229 shower 69134 (43 of 50 joined, a 958 MeV
shower) and ncpi0 evt463565 shower 109073 (**0 of 5** — renders empty).

The display never guesses. It compares the join against the shower's own
`num_segments` and says so in the `joined` column and a banner. With a probe
sidecar loaded it uses the non-lossy membership instead and the banner says
`REPAIRED`. Over the 94-event sample: **15 lossy showers, 15 repaired,
1567/1567 exact.**

## Data it reads

| input | what for |
|---|---|
| `work-<arm>-prod0825/pr_evt<ID>/calib-pr-evt<ID>.json` | everything drawn |
| `em_display/emprep/emprep-evt<ID>.json` | non-lossy membership, `dir15` axis, `absorbed by` |
| `em_display/em114-manifest.tsv` | the event list, the Bee link, per-event stats |
| `bee/em114/em114-<arm>.zip` | the 3-D charge cloud (optional; absent ⇒ skeleton only) |

Nothing here is ever written. Exactly one code path writes anything — `on_save` —
and only into `../em_labels/<tag>/labels-evt<ID>.json`.

### Regenerating

```bash
# stage 2: re-run the sample with the shower probes on (fresh arms, ~10 min)
./em_display/run_em114_probe.sh

# parse the probes + rebuild the manifest + resolve Bee links offline
python em_display/prep_em_scan.py --parse-probes

# self-tests
python em_display/selftest_repro.py         # reproduction + membership repair
python em_display/selftest_em_display.py    # drives the viewer's callbacks, 105 checks
python em_display/selftest_em3d_browser.py  # drives the 3-D view in headless chromium, 29
```

The probes land in `pr_evt<ID>/stdout.log`, **not** in `wct_pr_evt<ID>.log`:
wire-cell's `-l` flags route only the spdlog logger, and a raw `fprintf(stderr)`
follows the subshell redirect in `run_pr_chain_batch.sh:1642`.

## Bee links

**All 94 events have a live Bee link** (uploaded 2026-08-27, `bee/em114/*.url`).
Two things about them worth knowing:

- **They point at the `em114` sets on purpose.** The same event exists in many
  older sets — `prod0813` among them — and those are *different reconstructions*.
  `bee_index(prefer="em114")` makes the matching epoch win every collision;
  without it, sorted-glob order silently sent 78 of 94 events to a Bee page whose
  clustering disagrees with the panels beside it.
- **A freshly-uploaded set can return 500 on the very first hit** while the
  server finishes unpacking it. Reload once. Verified: 94/94 return 200.

The mechanics, for the next round:  the
set UUID comes from `../bee/<round>/<tag>.url` and the per-event index from the
sibling `.index.txt`, and `/event/<n>/` is used directly as the on-disk directory
name server-side. For the other 16:

```bash
python em_display/prep_em_scan.py --bee-build bee/em114   # ALREADY DONE, see below
./upload-to-bee.sh bee/em114/em114-<arm>.zip     # <- YOURS to run, not mine
# save the printed URL as bee/em114/em114-<arm>.url, then re-run prep
```

`bee/em114/` is **already built**: four zips (mcp1k 10, mcp2k 17, ncpi0 19,
nuecc48 48 events; 51 MB total) with their `.index.txt` and `.prid-map.txt`
sidecars, each starting at `data/0/` as the server requires. Only the upload is
left.

*Gotcha the build exposed:* `make_pr_bee.py` decides "was this event evaluated"
by grepping the per-event **log**, and ncpi0 evt399860's prod0825 log is
truncated (22 KB vs a 207 KB median) so it refuses an event whose dump is fine.
`prep_em_scan.py` therefore points it at the **em114** arms, which carry the line
for 94 of 94 and are equal to prod0825 on every physics field.

The upload is outward-facing and owner-gated (CLAUDE.md §5.6). The build step is
local and touches no network.

## Labels

`../em_labels/<tag>/labels-evt<ID>.json`, one file per event per tag, written
tmp + `os.replace` so a record is never half-written. The `em` and `pio` blocks
upsert independently — scanning EM now and π⁰ later does not drop the first half.

Each record stores **the reconstruction's answer next to yours**: the shower's
membership and where it came from (`probe` or `dump-join`), its axis and which
branch produced it, the `pio_id` groups, the `kine_pio_*` block, the vertex with
how it was obtained, and a `camera` block — the az/el, centre, radius and cloud
layer you were looking at — so a later re-read can put the event back on screen
the way it was seen. A later tuning fit joins one file per event and never has to
re-read a dump.

Marks live in `em.marks_by_shower` — `{shower: {segment: in/out/?}}` — and in
`em.marks_detail`, which carries for each marked segment its `dist`, `angle`,
`tier`, `ellip`, `length`, `pdg`, `cluster_id`, `absorbed_by` and current
`owner`, measured against the shower it was marked for, next to that shower's
`member_span` (n, and the distance and angle range of the segments the
reconstruction already put in). No flat `marks` map is written: a derived copy
alongside the authoritative one could disagree with it, and that ambiguity is the
bug per-shower keying exists to remove.

**Reading older records.** A round-4 file has a flat `em.marks` and one
`em.shower`; it still loads, its marks are attributed to that shower, and the
banner says so in red. A round-3 file has no `camera.cloud_scope` key, and that
absence means `all-clusters` — the whole cloud was on screen. Nothing rewrites an
old file; re-mark and save if the attribution is not what you meant.

> **A scan tag is a scientific record (CLAUDE.md M13).** Passing `--scan-tag`
> explicitly is consent to write into that set. Without it the viewer uses
> `emscan1` and **refuses to write** if that directory already holds labels.

## Known limits

- **The shower axis is the probe's `dir15`**, i.e. the C++'s own
  `shower_cal_dir_3vector(shower, start, 15 cm)`, for 1565 of 1567 showers. The
  probe prints components at `%.3f`, so it arrives with |v| in 0.9994–1.0004 and
  is re-normalised on read — about 0.06° on any angle, far below anything a scan
  turns on. The other 2 showers fall back to a Python mirror of `init_dir`
  (`PRShower.cxx:1552-1562` / `:1618-1640`); that mirror is exact for conn 1 and
  conn 2/3 and only approximate on the `shower_cal_dir_3vector` fallback branch.
- **`Σ dQ` in the impact line is fitted-point charge, not a calibrated energy.**
  It sizes the change; it does not price it in MeV.
- **No lasso over a 2-D projection**, and there still isn't one: a lasso there
  selects everything along the third axis, and that ambiguity is not worth the
  mis-marks. **Box select in the 3-D view is a different thing and is offered**,
  because the selection unit is the *segment*, not the point — a segment either
  is or is not in the shower, so hitting it through any of its projected points
  is unambiguous however the view is rotated. The cyan halo draws the segments a
  box resolved to, and the status line names them, before you mark them.
- **The 3-D view's browser code IS tested**, by `selftest_em3d_browser.py` — it
  starts a server, drives headless chromium with real mouse gestures, and checks
  that the browser's projection matches Python's exactly, that a bare drag
  rotates, that shift+drag pans, that the wheel zooms, that Box Select suspends
  rotation, that a box can mark and re-arm, and that the right-hand column really
  lands to the right of the 3-D canvas. What no test can judge is whether a drag
  *feels* smooth on the worst cloud, or whether the depth fading reads as depth;
  doc pr/114 §12.7 keeps a short human check-list for exactly those.
- The two `dir15`-less showers, and any event scanned without a probe sidecar,
  show an empty `absorbed by` column — the banner says so rather than leaving you
  to infer it.
- **The candidate cloud filter is a spatial match, not an identity.** It keeps
  whole clusters, so a cosmic that genuinely overlaps the candidate's charge is
  kept with it, and a piece of the candidate that no reco point reaches is not.
  Measured worst case over the sample: 0.34 % of the largest shower's fitted
  points on a dropped cluster. `all clusters` is one click away and the readout
  always says how many clusters of how many are being drawn.
- **The acceptance plot's zoom is anchored on the members *and* your marks**, so
  a mark far outside the shower pulls the range back out and squashes the member
  spread again. That is the right trade — hiding what you just marked would be
  worse — but it means the comparison line, not the plot, is the thing to read
  when a mark is a long way out.
- **`dim these showers away` fades, it does not delete.** Excluded segments are
  still tappable in 3-D at alpha 0.05 and still selectable; they are removed from
  the candidate table but they remain part of the reconstruction, and nothing
  about a mark on one is blocked.
- **The colour swatch is per event, not global.** A shower's colour comes from
  its rank in *this* event's energy-sorted table, so the same shower id in a
  different event can be a different colour. It is a key for reading one screen,
  not an identity across the sample.
- **A label with no `camera.cloud_scope` key predates the candidate filter**, and
  means `all-clusters` — that was the only behaviour before it existed. Do not
  read a missing key as the current default.
- **Orbiting far from the centre flattens the depth cue.** `cam_R` is the
  zoom-independent scale the fading normalises by and `orbit around it`
  deliberately does not rewrite it; `refit` restores both.
