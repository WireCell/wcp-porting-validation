# The PDVD STM + Michel hand-scan rubric, verdict-blind — apply this exactly

> Doc pdvd/99 §6.4, the blind swap scan. **PDVD port of the PDHD rubric v5**
> (`pdhd/docs/scan/pdhd_stm_michel_scan_rubric.md`, docs pdhd/18–19), which was
> itself ported from the frozen PDVD rubric of doc pdvd/55 and folds in the
> owner's PDVD re-judges (docs pdvd/68, pdvd/70 §9) and the owner's PDHD rulings
> (doc pdhd/19 §8). The rules are v5's; the detector section is PDVD's; the
> PDHD-only traps (APA seam, APA0/APA2 deficits, wrapped U/V) are removed. The
> last section lists every change.
>
> **Version.** The sha of this file is recorded before the first wave and
> stamped on every record. It does not change while scanners work.

You are hand-scanning reconstructed cosmic-ray events from **ProtoDUNE vertical
drift (PDVD)**. Each item is one reconstructed object: a fitted track, plus
every other charge cluster the display draws near it.

The question, in the detector owner's own words, has three parts:

1. identify the **STM** — the stopping muon, and where it stopped,
2. identify the **Michel electron** clustering — the electron from the muon's
   decay at rest,
3. leave the clusters associated with **neither**.

## Hard rules — breaking any of these invalidates the round

1. **Read only**: this rubric, your task file, your item list, and the files
   under `<SHOTS>/<event>_<cluster>/` for **your own** items. Nothing else.
   Specifically, never open:
   * any file with `key`, `labels`, `verdicts`, `carried`, `queue`, `sheet` or
     `adj` in its name. They hold the chain's verdicts and other people's
     answers;
   * the prep payloads (`smprep-*.json`), any `tracking-*.root`,
     `calib-*.json`, `mabc-*.zip`, or anything under `pdvd/work/`,
     `pdvd/docs/`, `pdhd/docs/` or `pdhd/work/`;
   * anything under `/home/xqian/tmp/` other than, inside
     `/home/xqian/tmp/p99scan/`: `RUBRIC.md`, `AGENT_TASK.md`, `shots/`, your
     own item list, your own out dir and your own scratch dir;
   * another scanner's out dir. Do not `ls` the `v_parts` tree.
2. **Never run `scan_harness.py`** in any mode. Your only write is one record
   per item, through `mkv.py`, into your own out dir.
3. **Write each item's record as soon as you have judged it**, before moving
   to the next. If you run out of room mid-chunk, everything you have written
   is kept and someone else finishes the rest, but only if you wrote as you
   went.

## This scan is verdict-blind

`context.json` has been stripped of the chain's own verdict: `is_stm`,
`reject_names`, `in_fv`, and the chain's particle-flow summary (`flow`). Do not
try to reconstruct it. You are the check on it. The items were not chosen to be
mostly stoppers or mostly through-goers; judge each on its own.

What is **not** hidden, because the display draws it in every frame, is the
chain's *typing* of each object: the `group` and `chain` columns of the object
table, and `seg_head`'s group counts. So "the chain calls S12004 the Michel
object" is visible. **Treat it as the reconstruction's opinion that you are
grading, not as evidence.** A `michel` group on a piece does not make it a
Michel, and a missing one does not mean there is none. The `status` line gives
the chain's range energy; that is a length, not a verdict.

## The detector

* **Drift is vertical, and it is along x.** **Up is +x.** There are two drift
  volumes with a **horizontal cathode at x ≈ 0** (its faces at |x| ≈ 3 cm): the
  **top volume** x > 0 with the top anode (CRP) plane at **x = +339.9 cm**, the
  **bottom volume** x < 0 with the bottom anode plane at **x = −339.9 cm**.
  Active y runs −336.4 → +336.4 cm and active z 0.8 → 298.4 cm; y and z are
  both horizontal. The red dashed box is the active boundary, and every number
  in `ends` is measured to it (checked on this sample: of 2468 fit ends, the
  99th percentile of |x| is 339.8 cm).
* **The display's axis labels do not show "up".** The side view draws Y
  vertically and the end view draws X horizontally, but **physically up is
  +x**: in the top view (Z vs X) up is the vertical axis, in the end view
  (X vs Y) up is to the **right**, and in the side view (Z vs Y) up points out
  of the page. Never take a panel's vertical axis to be up.
* **Most cosmics enter through the top anode plane (x ≈ +340).** An entry with
  `face.x` ≈ 0 at x > 0 is the normal way in. An entry on a side wall (y or z
  face) is also common.
* **Anode faces.** `seams_at_stop.anode_face` is the distance from the stop to
  the nearer anode plane (|x| = 339.9). **An end with `anode_face` ≲ 2 cm is at
  an anode.** At the **top** anode that is where a cosmic comes *in* (rule 7);
  at the **bottom** anode (x ≈ −340) with a flat profile it is an exit
  (`ANODE:`). Only a clear rise or clear decay topology makes an end at an
  anode a stop, and then say in the evidence that it sits there.
* **Cathode, the crossing trap.** A track whose fit ends within a few cm of the
  cathode may simply **cross into the other drift volume**, where the
  clustering can put the continuation in a different cluster, and the display
  places every cluster at its own t0-corrected position, so the continuation
  can sit visibly offset in x. An end at the cathode with **no Bragg rise** is
  `THRU` (or `FRAG_THRU` if you can see the continuation); it is a stop only
  when the profile rises there. `ends.seams_at_stop.cathode` is the distance in
  cm from the stop to the cathode's face.
* **The two volumes have different readout electronics, and their charge
  scales differ by up to ~10 %.** A track that crosses the cathode can step in
  dQ/dx at the crossing. That step is not a Bragg rise and not a collapse:
  judge a rise **relative to the plateau in the volume where the track ends**.
* **CRU seams.** Each anode plane is tiled by charge-readout units, with seams
  at **y = −168.5, 0, +168.5** and **z = 149.65** (drawn as dotted lines).
  `ends.seams_at_stop.seam_y` / `seam_z` give the distance from the stop. An
  end within a few cm of a seam with a flat profile is a **gap crossing**, not
  a stop (`SEAM:`). A seam is a vertical sheet through both volumes.
* **Wires.** U and V are induction planes and W is collection; none is wrapped
  on PDVD. In `f_meas` the W row is the most direct charge picture. Grey
  hatched bands are dead channels, and inside one the "measured" charge is the
  imaging's filler.
  A thin **horizontal** line across many channels at one time slice is charge
  arriving at a single drift time, i.e. an isochronous track (this one or
  another object). It is not a continuation along the drift direction.

## What you look at, per item

Each item has a directory `<SHOTS>/<event>_<cluster>/` holding **eight PNGs**
and a `context.json`.

| file | what it is for |
|---|---|
| `a_proj_full.png` | the three 2-D projections at full detector extent — overall topology, which faces it touches |
| `b_3d_wide.png` | the 3-D point cloud, the whole object |
| `c_3d_stop.png`, `d_3d_stop.png`, `e_3d_stop.png` | the 3-D cloud zoomed to ±45 cm about the stop at three azimuths 90° apart — *along the Michel direction* vs *close and backward* is a 3-D judgement, and one view can fake either |
| `f_meas.png` | U/V/W measured, predicted and difference, ±150 channels/slices around the stop, with dead channels. **Coloured cells** are the cells this cluster's own fit touched. **Grey squares, in the `measured` column only, are every OTHER live charge cell of the event** — other clusters, other flash bundles, unclustered activity — from the imaging. Read here: does this track's charge end, run into a dead band, or **continue in grey** |
| `g_dqdx.png` | **dQ/dx vs signed arc length** through the stop, with the muon (solid) and electron (dashed) reference curves |
| `h_dqdx_zoom.png` | the same panel as `g_dqdx.png`, upscaled 3×. **Read this one for the Bragg judgement** — there is no need to crop anything yourself |

Read `a`–`f` and `h`. You may skip `g` (it is `h` at a third of the size).

### Display traps

1. **Never read charge off the 3-D frames.** Points are coloured by dQ/dx
   (turbo), but the particle-flow overlay draws each PR segment in its own
   **categorical** colour and every vertex as an opaque brown square; turbo
   near 1–1.5e5 is also orange. Charge comes only from the dQ/dx panel and the
   `dqdx` column. The 3-D frames are for geometry. The amber "picked object"
   band has been removed from these frames.
2. **The orange entry circle is not always at the far end** — it can sit
   mid-track. Do not read it as the track's start.
3. **The 3-D frames draw the detector box in perspective**, so a far box edge
   can project right next to the stop. **Never read containment off a 3-D
   frame** — use `ends.stop.d_face`, the per-axis `face` numbers and
   `ends.seams_at_stop`.
4. **`cos_fwd` is measured against the local tangent at the stop**, not the
   track's overall chord. On a curved track it lies: a piece sitting *on* the
   muon line can read `cos_fwd −0.94` as though it were off-axis. On any
   visibly curved track, judge the off-axis distance from the 3-D azimuths.
5. **The dQ/dx panel is not a census of the drawn objects.** Objects of dozens
   of points can be absent from it while small specks appear. The object table
   is the census.
6. **The projections are locked at full detector extent**, where 10 cm is a
   handful of pixels. Containment is a numbers call.

`context.json` carries:
* the numbers the display shows: `ends.entry` / `ends.stop` with `xyz`,
  per-axis `face` distances and `d_face` (distance to the nearest active face,
  cm), and `ends.seams_at_stop` (`seam_y`, `seam_z` to the CRU seams, `cathode`
  to the cathode face, `anode_face` to the nearer anode plane, `wall_*` to the
  box; `seam_x` is the distance to x = 0 and duplicates `cathode`);
* one row per drawn object with its `key`, `group`, `npts`, `size` (cm),
  `dqdx` (e/cm), `dstop`, `chain`, and geometry: `d_min` / `d_max` (cm from the
  stop) and `cos_fwd` (+1 = straight ahead of the muon, −1 = back along the
  muon's own body);
* `status` — the event, cluster and the chain's muon length / energy. That is
  range, not a verdict.

## Verdict — pick exactly one

* **`STM_MICHEL`** — the track ends inside the volume **and** there is decay
  activity past the end that reads as an electron.
* **`STM_ONLY`** — ends inside the volume with a Bragg rise, and nothing past
  the end that reads as an electron. Isolated compact pieces judged to be
  muon-capture gammas do **not** promote this to `STM_MICHEL`.
* **`THRU`** — it does not stop: it crosses or exits a face (an anode plane, a
  side wall), crosses the cathode or a CRU seam without a rise, runs into a
  dead region with a flat profile, or charge continues past the fit end.
  **An end in the middle of the volume is NOT a through-goer just because its
  profile is flat** — see rule 3 of the precedence order.
* **`FRAG_STM_MICHEL` / `FRAG_STM_ONLY` / `FRAG_THRU`** — the drawn cluster is
  visibly only a fragment of a larger object; the verdict still describes the
  **full** object.
* **`MESSY`** — not one track at all: an EM blob broken into pieces, fused or
  over-clustered tracks, no coherent spine.
* **`UNCLEAR`** — you genuinely cannot tell. Say why in `--notes`.

### Which evidence decides — the precedence order

The two kinds of stop evidence are the **Bragg rise** (from the dQ/dx panel)
and the **decay topology** (an electron leaving the end). Apply them in this
order:

1. **Topology is sufficient.** A clear electron-like arm or object at the fit
   end means the muon stopped: `STM_MICHEL`, *even when the profile looks flat
   or ragged*. A muon that keeps going does not throw off a ~5–50 MeV electron
   at the point where its fit happens to end. On the PDVD re-judges the owner
   moved 8 of 13 agent `THRU` calls that carried a reconstructed Michel to
   stoppers (doc pdvd/70 §9.2), and 4 of the 6 verdicts changed in doc pdvd/68
   went the same way. An earlier agent record had read "flat profile" as
   overriding the arm; it does not.

   **What counts as topology.** An arm or object that is clearly a second
   particle: it leaves the stop at an angle (a kink), or starts a separate
   piece. A short stub running **straight on** from the fit end (cos_fwd ≳ 0.9,
   no kink, a few cm) is weak: it can be the muon's own continuation. It does
   not decide the verdict on its own; it needs a rise or rule 3.
   An arm attached **at the stop vertex** counts as decay activity **even when
   it runs back along the muon body**, unless the track's own line continues
   in grey past that end (then it is `THRU`, `CONTINUES:`).

   This holds at an upper end too — see rule 7's second bullet.
2. **Otherwise the Bragg rise decides.** With no electron-like activity, a
   rise at the end means `STM_ONLY`.
3. **A flat profile means `THRU` only where the track could have left.** The
   track stopped if all of these hold:
   * **its own charge ends** in all three planes of `f_meas` at the fit end;
   * no dead band swallows the end;
   * the end is not at a face (an anode: `anode_face` ≲ 2 cm; a side wall), the
     cathode or a CRU seam (use the numbers);
   * **nothing continues past it** — not in the 3-D frames and the
     projections, **and not in `f_meas`'s grey cells**; no large object in the
     table starts at the stop and carries on;
   * **it is not the track's upper end** (rule 7).

   **Reading a continuation in the grey cells.** `f_meas` draws the rest of
   the event in grey, **including other flash bundles** (which the 3-D frames
   hide under `bundle only`). A continuation is grey charge that **picks up the
   track's own line at the fit end and carries on in the same direction, in
   all three planes at matching time slices**. Grey that merely crosses the
   region, or sits in one plane only, is another object. A grey continuation
   makes the end `THRU` (or `FRAG_THRU` if you can tell the pieces are one
   track); say `CONTINUES:` in `--notes`. Grey is per-channel charge on its own
   scale; use it for *where*, not *how much*.

   When all of them hold, the missing rise is a measurement failure, not
   evidence of a through-goer (the Bragg peak can be unresolved, or carried by
   a short piece at the end). Call it `STM_ONLY`, or `STM_MICHEL` if there is
   decay activity, at most `medium`, with `--notes` starting `FLAT_STOP:`. This
   rule rests on the argument above, not on an owner example, and it applies
   only at a lower end (or where rule 7 does not apply).

   **Exception — near-isochronous tracks** (x nearly constant along the track,
   i.e. a nearly **horizontal** track on PDVD, so the whole track sits in a few
   time slices): their imaging and their ends are unreliable. Rule 3 does not
   apply; judge them `medium` / `low`.
4. **No measurement is not a flat profile.** If the last ~20 cm has too few
   live points to judge (a dead region, a short track), a flat reading means
   nothing. Decide from topology or rule 3 if you can. Otherwise answer
   `UNCLEAR` and start `--notes` with `NO_MEASUREMENT:`.
5. **A face is not a stop test by itself.** An end 3 cm inside a face with a
   clear rise, or with a Michel, is a stopper; the fiducial inset is a rule
   for the chain's energy, not for this question. An end *at* a face with a
   flat profile is an exit (`--notes` `ANODE:` for the anode planes, `FACE:`
   for the side walls).
6. **A busy stop is `MESSY`.** If another large object runs along or across
   the muon near its end (a long cluster lying on the body, a shower), so that
   you cannot isolate where this track ends and what leaves it, answer
   `MESSY`.
7. **Direction: cosmic muons travel down, and down is −x.** A stopping cosmic
   muon stops at the **lower** end of its track, and its upper end reaches a
   face (usually the top anode plane, x ≈ +340, or a side wall) or a gap it
   crossed. Compute from `ends`:
   `dx = stop.xyz[0] − entry.xyz[0]` and `chord` = the distance between
   `entry.xyz` and `stop.xyz`. The fit end is the **upper end** when
   `dx ≥ 0.3 × chord` (the track rises at least ~17° towards the fit end).
   For a flatter track (`|dx| < 0.3 × chord`) geometry cannot tell direction,
   and this rule does not apply. **Use x, not y.**

   At an **upper** fit end, a stop needs an upward-going particle. Among
   cosmics that is rare: a particle made in an interaction below, not a
   cosmic muon. So:
   * **Rule 3 does not apply at an upper end.** A flat profile there is where
     the track's charge **starts**: its upper part was not imaged, went into
     another cluster or bundle (look in `f_meas` for grey carrying its line on
     past that end), or the particle was produced in the volume. That is not a
     stop: `THRU`, with `--notes` starting `DIRECTION:`.
   * **At an upper end, topology or a clear rise makes it a stop** (owner
     ruling). Apply rule 1 at an upper end as anywhere else. A clear
     electron-like arm or separate piece leaving the fit end (a kink, or a
     piece that starts there) makes it `STM_MICHEL`. As everywhere, a
     straight-on stub is not topology, and neither is the track's own line
     continuing in grey (`CONTINUES:` → `THRU`). With no topology, a clear
     Bragg rise makes it `STM_ONLY`. Either way, at most `medium`, `--notes`
     starting `DIRECTION:`, and say what the arm or the rise looked like.
   * **Not ruled: an upper end with no topology and no readable profile**
     (dead band, too few points). Rule 7 says `THRU`, rule 4 says `UNCLEAR`,
     and the owner has not chosen between them. Call it `THRU` with
     `DIRECTION:`, and add `NO_MEASUREMENT:` to the notes so the item can be
     re-graded when the owner rules.
   * **A rise at the lower end** (the chain's "entry" end) is `REVERSED:`. The
     verdict for the fit end is then `THRU` if that end is flat, or `UNCLEAR`.
   * The same test explains an entry at the **bottom** anode plane
     (x ≈ −340): a track that enters through the floor is not a cosmic muon
     going down.

## Bragg — how to read `h_dqdx_zoom.png`

**The sign on this plot does NOT tell you which side of the stop something is
on.**
* Points **on the muon chain** are plotted at a real signed arc length:
  positive going back up the muon.
* **Every other object** (Michel, delta, dots, gamma) is plotted at **minus
  its 3-D distance from the stop**, whatever direction it lies in.

So a delta ray 21 cm back along the muon body appears at s ≈ −21, exactly
where a Michel dot 21 cm past the stop would. On the negative axis only the
**magnitude** carries information. To decide which side a piece is on, use its
`d_min` / `d_max` with `cos_fwd` and the three 3-D azimuths.

The test is a **rise above the track's own plateau, following the shape of the
muon reference curve** over the last 10–20 cm. Read it **relative to that
track's plateau, never as an absolute number**. There are no absolute charge
anchors in this rubric: the frozen PDVD rubric's were wrong (doc pdvd/55
§17.3), and on PDVD the two drift volumes read on different scales. Some
points poking above the reference curve near the end is common and fine;
staying under it is also fine.

* **A ragged rise still counts.** In the owner's words: *"the Bragg peak is not as
  consistent, but I feel the scan is OK."* Point-to-point scatter of a factor
  of two is normal. The question is whether the profile climbs towards the end,
  not whether it climbs monotonically.
* Ignore zero-charge points. They are fit points with no measured charge, and
  they fake contrasts. Check `f_meas` before reading a run of low points as a
  collapse.
* A muon at its own Bragg peak cannot continue at MIP. An arm leaving the stop
  at plateau charge is a *second particle*.

## The stopping point — when to move the pin

The pin defaults to the fit's last point, and that is right most of the time.
Move it only when the picture says the fit is wrong.

### The fit OVERSHOOTS: peak, then collapse

**The signature is a profile that peaks several cm before the fit's last point
and then collapses.** The mechanism, in the owner's words: *"there might be a
small gap between the stopping STM and the Michel electron leading to low
dQ/dx fit."* The fit bridges the muon's true stop and the Michel, and the
interpolated points across that gap carry little charge. **The diagnostic is
the collapse, not the height of the peak.**

When it is there, the segment past the peak is the **Michel**, even when the
chain types it muon. **The collapse is the diagnostic; its depth is not a
threshold.** What matters is that the charge falls well below the muon's own
plateau exactly where a muon at its Bragg peak would be well above it. Set
`--pin-rr <cm>` to the arc length of the junction between the last real muon
segment and that one, tag that segment `michel`, and start `--notes` with
`OVERSHOOT:`.

Not every terminal dip is this. A last point 20–30 % low, or a single
partial-charge point at the very end, is ordinary scatter: leave the pin alone
and say so in the evidence. The signature is a *sustained* collapse over
several cm into something the fit then bridges.

**When the collapse falls inside a single segment**, there is no separate row
to retag: the tag alphabet is per-object and cannot split one. Move the pin
anyway, since that is the measurement that matters. Tag the segment for what
*most* of it is, and say in the evidence that the row is split and which part
is which.

**What the bridge is.** The segment past the peak is the Michel when it
**carries charge**, i.e. it is the start of the electron, collapsed below the
plateau. When the fit instead bridges a **charge-free gap** to a detached piece
further out, the empty bridge is a fit artifact: tag it `delta / other`, and
tag the far piece by the attachment-distance rule (`gamma` beyond ~10 cm).

**Moving the stop moves the attribution with it.** A small isolated piece
that reads as a lone capture gamma beside an `STM_ONLY` can become part of the
Michel once the stop moves back to where it belongs.

### The fit stops SHORT: the hot tip

The other direction manufactures **false `STM_MICHEL`**, so check for it on
every item. *The Bragg peak sits 1–3 cm **past** the fit's last point*, in a
short piece lying straight ahead (`cos_fwd` ≳ +0.85, a couple of cm) whose
charge is **several times the track's own plateau**, and the chain offers that
piece as the Michel.

**The charge is the load-bearing clause** — not the angle and not the chain's
typing. An electron's first centimetres are at or below the plateau. A short
piece at several times the plateau is what the muon's own last centimetres look
like. So that tip is the **muon**, the true stop is at its far end, and the
item is `STM_ONLY` unless there is *other* decay activity further out.

A short forward piece **at or below the plateau**, especially one whose charge
*falls* as it goes out, is a real Michel: that is the ordinary `STM_MICHEL`
case, not this one.

`--pin-rr` cannot move the pin past the fit's last point, so:
* tag the tip `muon`, whatever the chain says;
* judge the verdict from what is left past the tip;
* start `--notes` with `UNDERSHOOT:` and say where the real stop is.

## Attribution — every drawn object gets exactly one tag

| tag | rule |
|---|---|
| `muon` | on the muon side of the stop, including a stub carrying straight on past a fit end that is not a stop |
| `michel` | attached at the stop (**including an arm from the stop vertex that runs back along the body**), or a piece past it consistent with the electron. **Within ~10 cm of the stop, direction does not discriminate**: a Michel is emitted isotropically, so a piece there is part of the decay whichever way it points |
| `gamma` | isolated, past the stop, clear of the muon body line, with `d_min` ≤ 60.0 cm (a hard edge). **Charge is not a criterion**: a faint speck 40 cm out is an uncertain gamma, not a confident delta |
| `delta / other` | lies **on** the muon body line and does **not** start at the stop vertex, at any distance; or belongs to a different object entirely; or is a degenerate row (no `dqdx`, size 0, or a centroid far from where `d_min` says it is) |

**`straddles the stop` is not used in this round.** The per-object alphabet
cannot split a row, and an object that genuinely spans the stop is handled
exactly as the split-row case above: tag it for what most of it is and say so.
`mkv.py` refuses it.

**When two rules collide, off-axis separation and attachment decide.** They
collide in both directions:

* *michel vs delta*: a piece 3 cm behind the stop but 6 cm out to the side
  reads `cos_fwd −0.39`. The michel rule claims it (within 10 cm, direction does
  not discriminate) and the delta rule also claims it (close and backward);
* *michel vs gamma*: a piece 9 cm out is inside the michel radius **and**
  isolated past the stop. Here **attachment** breaks the tie. If there is a
  clean charge-free gap between it and the stop, it is a detached dot
  (`gamma`). **Within ~5 cm of the stop a compact piece is part of the Michel
  even across a gap.** The ±45 cm frames cannot resolve a 3 cm gap. Beyond
  ~10 cm, a detached piece is `gamma`.

Resolve these on **how far the piece sits off the muon's line**, and let that
outrank both the `cos_fwd` number and the chain's own label:
* **hugging the muon body** → `delta / other`, at any distance;
* **well clear of the body line** → `michel` if attached or bridged to the
  stop, `gamma` if detached and compact.

Use the three 3-D azimuths to judge the transverse offset, and say in the
evidence which way you went.

**Faint far specks, 20–60 cm out.** How to read them depends on the verdict:
* on an **`STM_MICHEL`**, a speck is `gamma` when it lies on the Michel's side
  (roughly along the electron's direction, or on the same side of the stop),
  and `delta / other` when it lies off in an unrelated direction;
* on an **`STM_ONLY`**, the gammas are **capture** gammas, and a nuclear
  de-excitation cascade is **isotropic**. So an isolated compact speck within
  ~60 cm, clear of the body line, is `gamma` **in any direction**.

**`MESSY`: tag every row `delta / other`**, kind `none`. **`UNCLEAR`:** tag
rows by what you can see (the chain `muon`, the rest `delta / other` if you
cannot place them); the kind follows the tags as always.

**`THRU` / `FRAG_THRU`: tag fitted chain segments `muon` and everything else
`delta / other`**, kind `none`. `gamma` and `michel` are defined relative to a
stop. If you find yourself wanting `gamma` or `michel` on a `THRU`, the real
question is whether the verdict is right (see the precedence order). A fitted
piece with real charge continuing straight on past a `THRU` end is `muon`; a
row with no charge value or zero size is `delta / other`.

**Key shapes vary, and `mkv.py` checks them.** A fitted segment's key is its
bare id (`"107010"`); an unfitted cluster's key is `C` plus its id (`"C218"`).
Use the `key` field from `context.json` verbatim.

## `michel_kind` — mechanical, derived from your own tags

| you tagged | `michel_kind` |
|---|---|
| a `michel` piece and a `gamma` piece | `both` |
| a `michel` piece, no `gamma` | `attached` |
| a `gamma` piece, no `michel` | `detached dots` |
| neither | `none` |

`muon` and `delta / other` never affect it. `mkv.py` enforces this rule. It
also enforces the verdict against the tags:
* `STM_MICHEL` / `FRAG_STM_MICHEL` need a `michel` tag;
* `STM_ONLY` / `FRAG_STM_ONLY` may not have one;
* `THRU` / `FRAG_THRU` / `MESSY` carry kind `none`.

## Confidence

* `high` — the picture settles it.
* `medium` — you had to weigh two readings.
* `low` — you are guessing; say why in `--notes`.

Confidence describes the **verdict**. Put doubt about a tag or the kind (for
example, a far speck that decides `both` vs `attached`) in `--notes`, not in the
confidence.

On earlier rounds a double scan showed this field is calibrated: `high` calls
reproduced every time, and `low` calls did not reproduce at all. Mark a
genuinely uncertain call `high` and you corrupt that calibration.

## Evidence — required, one real paragraph per item

Write what you actually saw, so the call can be audited later without
re-opening the pictures. Cover:
* the length, and where the track entered and stopped relative to the faces,
  the anode planes, the cathode and the CRU seams (numbers from `ends`);
* which volume (top x > 0 / bottom x < 0) the stop is in, and the direction
  (rule 7: `dx` against the chord);
* what the dQ/dx profile did in the last 10–20 cm relative to its own plateau;
* what sits past the stop, where (distance, direction) and why it reads as
  Michel / gamma / delta;
* anything that made the call hard.

An example of the *shape* (invented numbers, no real item):

> 140 cm, enters through the top anode plane (face.x 0.4 cm at x = +339),
> stops at x = +212, 63 cm from the nearest side wall and 40 cm from the
> y = 168.5 seam, in the top volume; dx = −127 of a 140 cm chord, so the fit
> end is the lower end. The profile sits on its plateau and climbs to about
> twice it over the last 14 cm, following the muon curve; ragged, one point back
> at plateau, but a rise. A 9 cm two-piece arm leaves the stop at about 50 deg at
> plateau charge — a muon at its Bragg peak cannot continue at MIP, so the arm
> is a second particle, and 9 cm is the right range for a ~20 MeV electron. One
> speck 31 cm out, forward and clear of the body line: gamma. Three pieces
> 25–60 cm back along the body at cos_fwd −0.9 or below: delta / other. No dead
> region at the stop in any plane, and no grey continuation.

Do not write "looks good" or "clear STM". `mkv.py` refuses anything under 12
words, but the bar is the paragraph above, not the word count.

## How to write the record

One command per item, after you have looked at the frames:

```
python3 <MKV> '<event>/<cluster>' <VERDICT> '<michel_kind>' <confidence> \
  --shots-dir '<SHOTS>' --out-dir '<OUT>' \
  --tags 'muon:107002 michel:107004 gamma:C218' \
  --default 'delta / other' \
  --evidence '...one real paragraph...' \
  [--notes 'OVERSHOOT: ...'] [--pin-rr 4.8]
```

* **`--tags` splits on whitespace, so write `other:` for `delta / other`.**
  Writing `'delta / other:75021'` is rejected with a confusing "bad tag group
  'delta'".
* `--default` tags every row you did not name, so a forgotten object cannot
  ship as a silent hole. `mkv.py` **refuses** a record with any untagged row.
  Use it only when `delta / other` really is right for the unnamed rows.
* `--notes` prefixes that are counted across the round: `OVERSHOOT:`,
  `UNDERSHOOT:`, `NO_MEASUREMENT:`, `FLAT_STOP:` (rule 3), `CATHODE:` (the end
  is at the cathode crossing), `SEAM:` (the end is at a CRU seam), `ANODE:`
  (the end is at an anode plane, `anode_face` ≲ 2 cm), `FACE:` (the end is at a
  side wall), `DEAD:` (the end runs into a dead band), `ISOCHRONOUS:` (the
  track is near-isochronous, i.e. nearly horizontal), `REVERSED:` (the chain's
  "entry" end looks like the real stop candidate; the verdict is still for the
  fit end), `DIRECTION:` (rule 7: the fit end is the track's upper end),
  `CONTINUES:` (grey charge in `f_meas` continues the track past the fit end).
  An item may carry several, separated by `; `.
* The near-isochronous exception limits rule 3 only. A near-isochronous track
  that plainly crosses two faces can still be a `high` `THRU`.

If `mkv.py` refuses, fix the record and re-run it. Never hand-write the JSON.

## What changed from PDHD v5, and why

| # | change | source |
|---|---|---|
| P1 | detector section rewritten for PDVD: vertical drift along x with up = +x, horizontal cathode at x ≈ 0, anode planes at x = ±339.9, CRU seams at y = −168.5 / 0 / +168.5 and z = 149.65, volume-dependent charge scale | `smgeom.ENVELOPE` / `SEAMS["pdvd"]`; the 2468 fit ends of this sample (|x| p99 339.8, max 341.3; cathode-side min 2.96); doc pdhd/29 §8 and doc pdvd/99 (top/bottom scale) |
| P2 | the display's vertical axes are not "up" on PDVD, named as a trap | the frames' axis labels (side view Z vs Y, end view X vs Y) |
| P3 | rule 7 uses `dx` (x is vertical), and the entry-at-the-floor test is the bottom anode plane | P1 |
| P4 | an end at the top anode plane is an entry, at the bottom one an exit | P1, rule 7 |
| P5 | removed: APA seam at z ≈ 231, APA0 / APA2 charge deficits and coverage holes, wrapped-U/V cell factor | PDHD hardware, absent on PDVD |
| P6 | "at the anode" read from `anode_face` ≲ 2 cm (v4's ≲ 2 cm, the display's box already the job's sensvol on PDVD) | P1 |
| P7 | rule 1: a backward arm at the vertex is decay activity unless the track's line continues in grey (the v5 note on its withdrawn example, stated as the rule) | doc pdhd/19 §8 |
| P8 | hard rule 1 lists this round's files; item keys and owner-example keys removed | doc pdhd/19 §4 |
