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

**Directly under the header: have you already scanned this event?** A green
**✔ you have already scanned this event** (with the tag and the save time) or a
grey **not scanned yet**. It answers for *the event on screen* — the
`n/98 events labelled` counter in the right column cannot. It reports **disk
state only**: the separate `[unsaved]` marker next to the counter is the one that
tracks edits you have not saved yet. Read from the filesystem, and re-checked
every 5 s, so a second tab open on the same tag picks up a save made in the
first without you having to navigate away and back.

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

### Changing a shower's start and direction

The reconstruction's start for a shower is not always right, and everything the
pass-1 gate says depends on it. Set your own:

- **by clicking** — put `a tap in 3-D does` on *make it this shower's START* and
  click a reconstructed vertex (or any fit point). Then *aim this shower's AXIS
  through it* and click a second point to set the direction by eye.
- **by button** — `start = nearest vertex`, `start = nearest fit point`,
  `aim axis at nearest fit point`, or type x / y / z and press `use these`.
- `reset start` / `reset axis` put the reconstruction's back.

The **start and the axis always move together**. The probe's `dir15` is anchored
at the reconstruction's start, so once you move the start it no longer applies:
the axis is recomputed with the same formula at your new point
(`axis_source: python@start_override`), or, if you clicked a second point, it is
exactly the direction through it (`manual@override`). The readout under the
buttons always says which.

Your start is a correction to the **shower**, so the π⁰ tab uses it too — both
mass conventions are built on the same geometry, and the π⁰ panel names the
start each gamma used. A start set for a gamma slot in π⁰ mode is more specific
and still wins.

The record keeps `em.reco.*` as the reconstruction's own answer and
`em.axis_used` / `em.start_used` as what the gate actually used, plus both start
points, so the move is checkable later.

If you already marked segments on that shower, the readout says so: the saved
tier / angle / distance are recomputed from the start and axis in force at save
time, not from the ones you marked against.

### This event's topology

A checkbox above the note box, saved as `event_flags` at the root of the record
beside `em` and `pio`. One entry today — **no-vertex π⁰ (NCπ⁰)** — for events
that need separate treatment downstream, so a later pass can select them without
opening a shower block.


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

### A whole shower at once

"Shower B is really part of shower A" is a *merge*, and clicking its segments one
after another is the wrong tool for it — ten clicks, each of which can land on
the wrong row, and no way to see the set before committing it.

Pick the shower you are scanning, choose the other one in **whole shower**, press
**select all its segments**, then **mark IN**:

```
shower table      -> 4002                          (the shower you are scanning)
whole shower      -> 71022  (96.9 MeV, 10 seg)
                     [select all its segments]     -> 10 lit in cyan
                     [mark IN]                     -> 10 filed against 4002
```

Three clicks instead of ten, and the membership comes **from the probe**, not
from your aim — so a fragment that is hard to hit, off-screen, or hidden behind
another segment cannot be the one you miss.

It **selects; it does not mark.** The cyan halo shows exactly what the next
button will hit, so you see the set before you commit it, and all four mark
buttons then work on it unchanged — **mark OUT** on a whole shower says "this is
not part of the one I am scanning" in one gesture too. **add to selection** keeps
what is already selected, so several fragments go in together.

The shower being scanned is **not** in its own menu: every one of its segments is
already a member, so a bulk `mark IN` there would change nothing and still write
an entry per segment into the record.

The status line always reports **the count that will actually be marked**, e.g.
`selected 10 of 10`. Two cases make that more than a formality:

- a shower **dimmed away** is dropped from the candidate table but keeps its
  fitted points in the 3-D pick cloud, so all of it is still selected and the
  mark still lands — the line says so rather than letting an empty table read as
  a lost mark;
- a segment with **fewer than two fitted points** is drawn in no view and cannot
  be reached by any gesture. It is named in red.

The button reads membership from the probe rather than from the candidate
table's rows, and that is deliberate: the table is a *view*, filtered by
`show members too` and by whatever you have dimmed away, so a selection built out
of its rows is only as complete as what happens to be listed. The dimmed case
above is exactly that failure, and it is pinned by a test.

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

### Which energy a gamma contributes

`kine_charge` is `charge / (recom × fudge)`, and **which pair** was used was
decided by `Shower::get_flag_shower()` — a property of the reconstruction, not of
the slot you drop the object into. A shower the reco called a track or a proton
therefore carries a track's or a proton's energy, which is the wrong number for a
photon.

The **energy hypothesis** selector, one per gamma slot:

- **as reconstructed** (default) — the reco's own `kine_charge`, unchanged.
- **as EM shower (charge-inferred)** — the same collected charge re-converted
  with the shower pair (0.50, 0.80).

evt166870 is the worked case: shower 85045 is pdg 13 with `flag_shower` false, so
its 38.6 MeV is a track's energy; as an EM shower the same charge gives 64.2 MeV
and the mass moves 116.1 → 149.7 (π⁰ rest mass 134.98). The panel warns when a
slot holds a non-shower-flagged object and names the number the switch would give.

The default is the reconstruction's on purpose: a record saved before this
control existed has no `energy_hypothesis` key and re-opens on the reco's energy,
so it still reads the mass it was saved with. The record keeps `energy`,
`energy_hypothesis` and `energy_as_reconstructed` together.

A gamma the reco already flagged as a shower is left alone — the switch says so
rather than double-converting.


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

### Which segments a gamma's energy is summed over

`kine_charge` is the **reconstruction's** energy, over the segments the
**reconstruction** put in the shower. Marking a segment IN changes the
membership and nothing else — so merging shower 27015 (10 segments, 105.05 MeV)
into shower 69032 (39.06 MeV) on evt409634 left the π⁰ mass built on 39.06.

**gamma energy membership** decides it:

| | E1 | m axis | m vertex |
|---|---|---|---|
| `reco membership only` | 39.1 | 41.0 | 83.4 |
| `include my IN / OUT marks` (default since round 16b) | **144.1** | **78.8** | **160.3** |

The angles do not move — a mark changes *which charge belongs to the shower*,
not where it points.

**It defaults to counting your marks (round 16b).** It did not until then, and
the reason it changed is a measurement: of the 18 records on disk, *every* one
carrying marks had been saved with the switch off, so no hand-made clustering fix
reached any saved mass — including +105.1 MeV on evt409634 and −45.6 on evt47212.
The marks exist because the reconstruction's membership is wrong.

**Re-opening a saved record still shows the mass it was saved with.** The switch
is set from the record, in both directions: a record saved with it off turns it
back off, and only a record with no π⁰ block at all takes the new default.
Measured over all 18 labelled events at the flip: 0 re-priced, 15 restored off,
3 (`pio: null`, no gammas, no mass) took the default. With the switch off the
panel still says what is being left out and what the mass would become.

The arithmetic is exact, not an estimate. The probe gives a per-segment `E_est`
that **sums to `kine_charge`**: shower 69032's two members are 29.498 + 9.560 =
39.058, which is its `kine_charge` to the last digit. So adding a marked
segment's share is arithmetic on the same number the C++ mass formula reads.

Four cases the panel keeps apart:

- **a member marked IN adds nothing** — it is already inside `kine_charge`.
  Reachable in one gesture since the whole-shower select, with `show members too`
  on by default;
- **a non-member marked OUT takes nothing off** — it was never in it;
- a **member** marked OUT does take its `E_est` off;
- a segment **no shower owns** has no `E_est`. It is named in red and **not
  counted**, and no dQ-derived estimate is offered: `E_est/dQ` is not constant
  between showers (6.05e-5 against 4.99e-5 MeV per electron on evt409634 alone),
  so that number would be a different quantity in the same units. 3.4 % of
  segments (196 of 5830) are in this position, on 84 of the 98 events, so it is
  a case the scan meets often — see below for what the panel now says about it.

A marked segment's `E_est` was converted with **its own** shower's recombination
pair, so under `as EM shower (charge-inferred)` the same round-9 ratio is applied
per segment — a segment coming from a track-flagged shower is scaled by
0.665/0.4 = 1.6625, exactly as the shower itself would be.

#### A mark on a segment no shower owns is a decision being overruled

Marking such a segment in is not working around a gap in the display. The
reconstruction has an opinion about that segment, and it is on disk. Both panels
now say it — the EM mark list where the mark is *made*, the π⁰ panel where its
energy goes missing — in three parts:

1. **what it is and where the code put it** — its length, the PID the
   reconstruction gave it, and `shower_id -1` straight out of the dump. That
   half needs no probe sidecar and survives on an event without one.
2. **why** — the probe's own absorb record for *this* shower. On evt169626,
   `SHOWER_ABSORB EXCLUDE shower_start_seg=53069 seg=53070 pdg=2212` at
   `in_other_clusters_A`: the straight-long-track guard
   (`PRShower.cxx:722-727`, F12 / doc pr/40 round 6) declined a confidently
   PID'd non-electron. The reconstruction considered that exact segment for
   that exact shower and refused. The mark overrules one named decision.
3. **how much is at stake, and why it is still not a MeV** — `ΣdQ` for the
   segment against `ΣdQ` over the shower's members, as a percentage. On
   evt169626 segment 53070 carries 1.69e6 e, **20 % of γ1's own 8.42e6 e** — so
   the scanner can see the size of what is left out without being handed a
   number that would need a conversion nobody can pin. The conversion is the
   blocker, not the charge: this event's own eight showers convert at 1.22e-5 to
   7.20e-5 MeV per electron, a factor of 5.9, and for a proton-PID'd object the
   code would not use the shower recombination pair (0.35×0.95) either.

`ΣdQ` here is the sum of the segment's fitted-point `dQ` from the dump. That is
the same quantity the probe reports per member segment — checked over the 5 700
member segments of all 98 events, worst relative difference 5.9e-6 — which is
what puts a segment the probe never saw on the same scale as one it did. It is
*not* the filtered sum the EM-mode **impact** line uses (that one drops points
with `dQ < 0`); the two differ by about 1 % over a shower and are deliberately
kept apart.

The energy behaviour is unchanged: the marked segment is still not summed, and
no stored mass moves. The reason it is looked up per shower rather than
last-record-wins is that a segment can carry several absorb records (13023 on
evt169626 has a `walk_add` then a `direct`), and 9 of the 23 `walk_exclude`
segments in the sample were absorbed somewhere else afterwards.

The record keeps both numbers and the working: `energy_includes_marks`,
`energy_marks_delta`, `energy_without_marks`, and `energy_marks_detail` with each
segment's `E_est`, its owning shower and that shower's kine label — so a later
reader can recompute either energy without the probe sidecar.

### More than one π⁰, and more than one way to pair the gammas

Two slots hold **one** pairing. An event with two π⁰ needs two masses in the
record, and an event where the pairing is uncertain needs the alternatives side
by side — so pairings are **stored** rather than overwritten.

Assign both gamma slots, press **store this pairing**, repeat. Each stored entry
freezes its own gammas, their starts, their energy hypotheses, the vertex and
**both** mass conventions:

| # | γ1 | γ2 | E1 | E2 | θ | m axis | m vertex | vertex |
|---|---|---|---|---|---|---|---|---|
| 1 | 15036 | 87078 | 164.7 | 82.0 | 141.9° | 219.7 | 208.4 | main_vertex |
| 2 | 84070 | 91112 | 73.4 | 68.7 | 69.7° | 81.2 | 93.7 | main_vertex |

Pick a row and **load into the slots** to put one back on screen — the arrows and
the start markers follow, so pairing 2 can be looked at and then pairing 1 again.
**remove** drops a row, and the rest are renumbered (a note that names one by
number goes stale; say the shower ids instead).

**A stored pairing is frozen numbers, not a reference.** Everything a mass is
built from stays editable afterwards — `em_start` in particular is keyed by
*shower* and lives in EM mode — so a candidate that merely named its showers
would be silently re-priced by a later start correction. Loading one back
recomputes it live and **says so if the two disagree**:

> ⚠ what is on screen is **not** what was stored — axis-convention mass
> 81.2 → 47.6 MeV. The stored candidate is unchanged…

That is the one real asymmetry: `load into the slots` pins the start
(slot-scoped, so loading candidate 2 cannot move candidate 1's), but the
*axis*-convention angle comes from `shower_axis`, which reads the per-shower
`em_start` and cannot be pinned per slot. So a start correction made after
storing moves the axis mass and not the vertex mass, and the message names which
one moved.

**The freeze has an exit (round 16).** Frozen is right — a stored row is your
curated judgement — but the table now says when a row has fallen behind, and one
button replaces it. Two different things are flagged, in the row's *note* column
and summarised under the table:

| flag | what it means |
|---|---|
| `NOT today's numbers: …` | an input under the row moved after it was stored — a start, an axis, a mark, or the reconstruction's own vertex. Each row is re-priced under the hypothesis and membership switch **it** was stored with, so what is listed is what actually moved, not which switch happens to be up now. |
| `gN ignores your marks (+105.1 MeV)` | the row is not stale — re-priced under its own settings it has not moved — but it was stored with *gamma energy membership* on `reco membership only`, so marks you have since made on that gamma are not in its energy. The number is what they are worth. |
| `vertex is main_vertex; this event was last saved as backproject` | the row uses a different vertex convention (or, for a back-projection, a different one of round 14's two ray sets) from the one this event was **last saved** with. Alternatives are what the list is for, so this is a note and not a fault. |

**update to today's numbers** replaces the selected row in place: it keeps the
row's `stored_utc`, stamps `updated_utc`, and appends what the row said before
to a `supersedes` list, so no reading is ever destroyed. It requires the row to
be selected *and* the slots to be holding that row's own two gammas — press
**load into the slots** first. (Delete-and-re-store would do the same arithmetic
while losing the row's place in the list and its first-stored time.)

**load into the slots** also brings back the row's own membership switch, the way
it already brought back its energy hypothesis and its back-projection geometry.
Before round 16 it did not, so a row stored at 144.1 MeV could be displayed at
39.1 MeV the instant it was loaded.

**A shower in two candidates is not an error.** It means they are *alternative
pairings of the same gamma*, and the panel says so — two real π⁰ in one event
need four distinct showers, and when they are distinct it says that instead.

**Nothing is auto-added.** If the slots hold a pairing that is not in the list,
the panel says so and Save still records it, as the record's single top-level
`pio.gammas`. Silently appending to a list you curated would be worse than
saying it.

The record grows one key. `pio.candidates` is **always written, empty list
included**, so a reader can tell "this scanner stored no alternatives" from
"this record predates the list". A record saved before round 11 has no key at
all, and re-opens showing exactly the one pairing it always did.

A row that has been updated (round 16) carries two more keys: `updated_utc`, and
`supersedes` — a list, oldest first, of what the row said each time it was
replaced (`stored_utc`, both masses, θ, the vertex and its convention, the two
energies and whether each counted marks). A row that was never updated has
neither key.

### The vertex point is in the record, whichever way it was chosen

All three conventions save the **point**, not just the mode:

| `vertex_how` | what is stored |
|---|---|
| `main_vertex` | `pio.vertex` — the reconstruction's own vertex |
| `manual` | `pio.vertex` — the point you typed or tapped |
| `backproject` | `pio.vertex`, plus `pio.backproject` with the **branch** that produced it (`both_long` / `one_short`), the closest-approach `gap`, `angle1` / `angle2` against the code's 25° gates, `dis1` / `dis2`, `len1` / `len2`, `theta`, `mass` and `verdict` — the last recording whether the code itself would have kept this vertex , plus `geometry` / `ray1_source` / `ray2_source` / `short_rerayed` naming the rays it was built from and `alt`, the whole result under the other geometry |

Every stored π⁰ pairing carries its own copy, and the event's reconstructed
vertex is at the record's top level as `main_vertex` regardless of which
convention the π⁰ used.

Re-opening an event with a manual vertex fills the x/y/z boxes from the record.
That matters: `_manual_point()` reads all three boxes, so before round 13 —
when the restore set only the internal state and left the boxes empty — editing
any one of them made the other two read as blank and **wiped the vertex**,
saving `vertex: null` under `vertex_how: "manual"`.

**The mode is per-event, on the way back as well.** Re-opening an event puts the
radio on whatever that event's record says, and an event with no record — or a
record saved from EM mode, with no π⁰ block — opens on `main vertex`. Before
round 15 only `manual` and `backproject` were restored, so a record saved as
`main vertex` came back showing whichever mode the *previous* event had left:
27 of 54 re-opens of the live tag were wrong. The manual x/y/z boxes and the
EM-mode start boxes are cleared with it — the start boxes are what
`_anchor_for_snap` reads first, so before round 15 *snap to nearest vertex* and
*aim axis at nearest fit point* could anchor on a point typed into the previous
event, 335 cm away, with nothing on screen to say so.

### Which rays the back-projection uses

Back-projecting the two gammas needs a ray per gamma, and there are two honest
ways to build one. **back-projection geometry** picks between them:

| setting | ray origin | ray direction |
|---|---|---|
| **my corrected start / axis** (default) | the start you set for that gamma | the axis you aimed, or the code's 15 cm recipe re-evaluated at your start |
| the reconstruction's own rays (mirror) | the shower's fitted point closest to the main vertex | `shower_cal_dir_3vector` there, 15 cm |

The second is a faithful mirror of `id_pi0_without_vertex` and answers *"what
vertex would the code compute here"*. The first answers *"what vertex does my
geometry imply"*. Both are useful, so **the panel always shows the one it is not
using, and how far apart the two are** — you never have to flip the control to
find out whether it matters.

Before round 14 only the second existed, silently: the panel printed *"your
corrected start from EM mode"* for the two masses and back-projected from the
reconstruction anyway. On evt76346 that was the difference between a `degenerate`
result sitting on the main vertex and a clean back-projection 60 cm away.

Three things worth knowing:

* **A gamma you never corrected changes nothing.** It injects no ray and the
  mirror builds its own, so the default cannot move a vertex you did not touch.
  Records written before round 14 re-open on the mirror, whatever the setting
  would default to, because `pio.backproject_geometry` says so — an absent key
  means the reconstruction's rays.
* **A short gamma is not re-rayed if you stated its direction.** The code
  re-derives a sub-15 cm shower's direction because it does not trust the stub's
  own; if you have aimed it, that question is answered. `short_rerayed` in the
  record says which happened.
* **Move a start clear of its shower and there is no direction to be had.** With
  no member point inside the 15 cm window the gamma injects no ray, and the
  provenance string in the panel says exactly that instead of substituting some
  other vector.

Your IN / OUT marks do **not** change which branch runs. The 15 cm test reads the
reco's `total_length`; marks move a gamma's energy, not its length.

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
| `bee/em114/em114-<arm>.zip`, `bee/em114b/em114b-mcp1k.zip` | the 3-D charge cloud (optional; absent ⇒ skeleton only) |
| `../docs/pr/pr114-owner-adds.index.txt` | events added by hand + the owner's note per event |

Nothing here is ever written. Exactly one code path writes anything — `on_save` —
and only into `../em_labels/<tag>/labels-evt<ID>.json`.

## Adding an event to the scan

`em114-manifest.tsv` is **generated**. Editing it by hand works until the next
`prep_em_scan.py` run and then silently loses the edit. The durable input is
`../docs/pr/pr114-owner-adds.index.txt`:

```
sample <TAB> run <TAB> subrun <TAB> event <TAB> origin <TAB> note
```

`scan_sample()` merges it over the pr/113 lists and it does two jobs at once:

- an event **already** in the pr/113 sample keeps the row it had — same origin,
  same Bee round, same dump — and gains only the note;
- an event **new** to the sample is appended.

An added event needs three things before it is scannable, and the display tells
you which one is missing rather than degrading quietly:

1. **a prod0825 dump** — `work-<sample>-prod0825/pr_evt<ID>/calib-pr-evt<ID>.json`.
   Without it there is no row at all, and `prep_em_scan.py` prints the run-event
   pair under *NOT IN THE DISPLAY* so it stays visible at every regeneration.
2. **a probe sidecar** — run the PR chain with the three `WCT_SHOWER_*_DEBUG`
   env vars into a **fresh** arm (`work-em114b-<sample>`, round 6's), then
   `prep_em_scan.py --parse-probes work-em114b-<sample>`. Without it the
   *absorbed by* column is empty and a lossy shower join cannot be repaired.
3. **a Bee zip** — `--bee-build bee/em114b --bee-events <ids>`. The set is named
   after the output directory, so this adds a set instead of rebuilding the
   94-event one the live display is reading.

### `bee_round` and `bee_url` are not the same thing

This is the one trap worth stating twice. `bee_round` names the **local zip** the
3-D charge cloud is read out of; `bee_url` is the **external link**, and it needs
a UUID that only the server mints on upload. So a set built here and not yet
uploaded has a working 3-D view and no link, and the banner says exactly that
("Bee set built but not uploaded … the 3-D cloud below IS this set").

The danger is what fills the gap if you let it. The four events added in round 6
are absent from `em114` but present in `prod0813` (uploaded) and `prod0819` —
two older reconstruction epochs — and `em114b` sorts *before* `prod0813`, so a
single-string `prefer` would have bound them to a two-epoch-old cloud drawn under
a prod0825 skeleton, silently. `prefer` is a sequence now, and
`selftest_em_display.py` checks **every** manifest row by measuring the distance
from the dump's fit points to the zip's own `track_fit-global` layer: 0.0005 cm
when the binding is right, 0.0314 cm against prod0813/prod0819 — a 60× margin,
and a genuinely wrong *event* would be tens of cm.

### Regenerating

```bash
# stage 2: re-run the sample with the shower probes on (fresh arms, ~10 min)
./em_display/run_em114_probe.sh

# parse the probes + rebuild the manifest + resolve Bee links offline
python em_display/prep_em_scan.py --parse-probes

# self-tests
python em_display/selftest_repro.py         # reproduction + membership repair, 98/98
python em_display/selftest_em_display.py    # drives the viewer's callbacks, 224 checks
python em_display/selftest_em3d_browser.py  # drives the 3-D view in headless chromium, 53
```

The probes land in `pr_evt<ID>/stdout.log`, **not** in `wct_pr_evt<ID>.log`:
wire-cell's `-l` flags route only the spdlog logger, and a raw `fprintf(stderr)`
follows the subshell redirect in `run_pr_chain_batch.sh:1642`.

## Bee links

**94 of the 98 events have a live Bee link** (uploaded 2026-08-27,
`bee/em114/*.url`). The four added in round 6 have a **local set only** —
`bee/em114b/em114b-mcp1k.zip`, built and not uploaded — so their 3-D view works
and their external link is blank until someone runs `upload-to-bee.sh`.

Two things about the links worth knowing:

- **They point at the `em114` sets on purpose.** The same event exists in many
  older sets — `prod0813` among them — and those are *different reconstructions*.
  `bee_index(prefer=("em114", "em114b"))` makes the matching epoch win every
  collision; without it, sorted-glob order silently sent 78 of 94 events to a Bee
  page whose clustering disagrees with the panels beside it. `prefer` is a
  **sequence**, last wins — round 6 re-armed this exact trap with a round whose
  name also starts with `e`.
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
tmp + `os.replace` so a record is never half-written. The presence of that file is
exactly what the scanned/not-scanned chip under the header reports. The `em` and `pio` blocks
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
