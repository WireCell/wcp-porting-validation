# The PDHD STM + Michel hand-scan rubric — apply this exactly

> Doc pdhd/18, tag `smx18`; **v4 for the review round, doc pdhd/19, tag
> `smx19`**. Ported from the frozen PDVD rubric
> (`pdvd/docs/scan/pdvd_stm_michel_scan_rubric.md`, doc pdvd/55), not copied:
> the detector is different, the scan is **verdict-blind**, and the corrections
> doc pdvd/55 §17.3–17.4 and the owner's re-judges (doc pdvd/68, doc pdvd/70 §9)
> recorded are folded in. The last section lists every change. **v4 changes the
> display (the anode, the cathode and the grey cells in `f_meas`) and adds
> rule 7, direction** — read those sections even if you know v3.
>
> **Version.** The sha of this file is recorded before each wave and stamped on
> every record. It changes only between waves, never while scanners work.

You are hand-scanning reconstructed cosmic-ray events from **ProtoDUNE
horizontal drift (PDHD)**. Each item is one reconstructed object: a fitted
track, plus every other charge cluster the display draws near it.

The question, in the detector owner's own words, has three parts:

1. identify the **STM** — the stopping muon, and where it stopped,
2. identify the **Michel electron** clustering — the electron from the muon's
   decay at rest,
3. leave the clusters associated with **neither**.

## Hard rules — breaking any of these invalidates the round

1. **Read only**: this rubric, your task file, your item list, and the files
   under `<SHOTS>/<event>_<cluster>/` for **your own** items. Nothing else.
   Specifically, never open:
   * any file with `key` or `labels` in its name (`*_key*.tsv`, `labels.json`,
     `*labels*.json`). They hold the chain's verdicts and other people's
     answers;
   * the prep payloads (`prep-*/smprep-*.json`), any `tracking-*.root`,
     `calib-*.json`, `mabc-*.zip`, or anything under `pdhd/work/`. They carry
     the chain's verdict;
   * **(v4) anything under `/home/xqian/tmp/h18/`** (the previous round: its
     frames, records and waves are the answers you are checking), anything
     under `/home/xqian/tmp/h19/` other than `RUBRIC.md`, `AGENT_TASK.md`,
     `shots/`, your own item list and your own out dir, and
     **nothing under `pdhd/docs/` except this rubric** (the previous round's
     record, its queue and its doc name these items and their verdicts), nor
     `pdvd/docs/scan/`;
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
try to reconstruct it. You are the check on it.

What is **not** hidden, because the display draws it in every frame, is the
chain's *typing* of each object: the `group` and `chain` columns of the object
table, and `seg_head`'s group counts. So "the chain calls S12004 the Michel
object" is visible. **Treat it as the reconstruction's opinion that you are
grading, not as evidence.** A `michel` group on a piece does not make it a
Michel, and a missing one does not mean there is none.

## The detector

* **Drift is along x.** There are two drift volumes with a **central cathode
  at x ≈ 0**; it is drawn as the dotted line at x = 0 in the top and end views.
  The anode wire planes are at **|x| = 352.1 cm**. Active y runs 7.6 → 606 cm
  and active z runs 0.2 → 462 cm. The red dashed box is the active boundary,
  and **(v4) it and every number in `ends` are now the true active volume**
  (below).
* **Four APAs.** APA0 is x<0, z<231; APA1 is x>0, z<231; APA2 is x<0, z>232;
  APA3 is x>0, z>232. The APA seam at **z ≈ 231** is drawn as a grey dotted
  line in the side and top views.
* **Most cosmics enter through the top (y ≈ 606).** An entry at `d_face` ≈ 0 on
  y is normal.
* **Anode — (v4) the numbers are now right.** Through v3 the box and every x
  distance in `ends` put the anode faces at |x| = 358 cm, ~6 cm outside the
  wire planes, and v3 told you to read `face.x` ≲ 8 cm as "at the anode". The
  display now uses the job's own active volume, |x| = 352.1 cm (no reconstructed
  end in this sample goes past 352.2). **So `face.x` and
  `seams_at_stop.anode_face` are the real distance: an end with `face.x` ≲ 2 cm
  is at the anode face.** With a flat profile it is an exit (`ANODE:`). Only a
  clear rise or clear decay topology makes it a stop, and then say in the
  evidence that it sits at the anode. **Do not subtract 6 cm any more.**
* **Cathode, the PDHD-specific trap.** A track whose fit ends within a few cm
  of x = 0 may simply **cross into the other drift volume**, where the
  clustering can put the continuation into a different cluster. The display
  places every cluster at its own t0-corrected position, so the continuation
  can sit visibly offset. An end at the cathode with **no Bragg rise** is
  `THRU` (or `FRAG_THRU` if you can see the continuation); it is a stop only
  when the profile rises there. `ends.seams_at_stop.cathode` is the distance in
  cm from the stop to the cathode plane — **(v4) now to the cathode's own face
  at |x| = 0.16 cm** (through v3 it measured to |x| = 2.54, a fiducial cut, so
  v3 numbers read 2.4 cm short).
* **APA seam.** An end within a few cm of z ≈ 231 with a flat profile is a gap
  crossing, not a stop. `ends.seams_at_stop.seam_z` gives that distance.
* **Two known charge deficits.** Neither is a physics feature of the track:
  * **APA0** (x<0, z<231): collection-plane hardware fault. Charge there reads
    about 0.65 of the other APAs, and more of its points are partial. A Bragg
    rise in APA0 is a rise **relative to that track's own lower plateau**;
    judge the shape, never the height.
  * **APA2** (x<0, z>232), especially its bottom / far-z corner (y ≲ 60 cm): up
    to half the fit points can sit where the image has no charge, so dQ/dx
    shows **zeros and near-zeros that are coverage holes, not a collapse**.
    Check `f_meas`: a real collapse has measured charge falling off. A coverage
    hole has the predicted band continuing over empty measured cells.
  * **(v3) APA0 makes holes too.** Its weak collection plane can leave measured
    W charge only in patches. The dQ/dx then turns into a smooth periodic wave
    with nothing to read, which is `NO_MEASUREMENT` territory, not a flat
    profile (wave-1 scanners).
* **Wires.** U and V are wrapped induction planes, and W is collection. In
  `f_meas` the W row is the most direct charge picture. Grey hatched bands are
  dead channels, and inside one the "measured" charge is the imaging's filler.
  **(v4)** On U and V the coloured cells read 2–3× the channel's charge (the
  fit's 2-D tree counts a wrapped channel once per wire segment the fit
  touched). Compare U/V cells with each other, never with W or the grey cells.
  **(v3)** A thin **horizontal** line across many channels at one time slice
  is charge arriving at a single drift time, i.e. an isochronous track (this
  one or another object). It is not a continuation along the drift direction.

## What you look at, per item

Each item has a directory `<SHOTS>/<event>_<cluster>/` holding **eight PNGs**
and a `context.json`.

| file | what it is for |
|---|---|
| `a_proj_full.png` | the three 2-D projections at full detector extent — overall topology, which faces it touches |
| `b_3d_wide.png` | the 3-D point cloud, the whole object |
| `c_3d_stop.png`, `d_3d_stop.png`, `e_3d_stop.png` | the 3-D cloud zoomed to ±45 cm about the stop at three azimuths 90° apart — *along the Michel direction* vs *close and backward* is a 3-D judgement, and one view can fake either |
| `f_meas.png` | U/V/W measured, predicted and difference, ±150 channels/slices around the stop, with dead channels. **Coloured cells** are the cells this cluster's own fit touched. **(v4) Grey squares, in the `measured` column only, are every OTHER live charge cell of the event** — other clusters, other flash bundles, unclustered activity — from the imaging, out to ±200 channels/slices of the fit end (so the whole ±150 frame is covered). Read here: does this track's charge end, run into a dead band or coverage hole, or **continue in grey** |
| `g_dqdx.png` | **dQ/dx vs signed arc length** through the stop, with the muon (solid) and electron (dashed) reference curves |
| `h_dqdx_zoom.png` | the same panel as `g_dqdx.png`, upscaled 3×. **Read this one for the Bragg judgement** — there is no need to crop anything yourself |

Read `a`–`f` and `h`. You may skip `g` (it is `h` at a third of the size).

### Display traps

1. **Never read charge off the 3-D frames.** Points are coloured by dQ/dx
   (turbo), but the particle-flow overlay draws each PR segment in its own
   **categorical** colour and every vertex as an opaque brown square; turbo
   near 1–1.5e5 is also orange. Charge comes only from the dQ/dx panel and the
   `dqdx` column. The 3-D frames are for geometry. (The amber "picked object"
   band of the PDVD round has been removed from these frames.)
2. **The orange entry circle is not always at the far end** — it can sit
   mid-track. Do not read it as the track's start.
3. **The 3-D frames draw the detector box in perspective**, so a far box edge
   can project right next to the stop. **Never read containment off a 3-D
   frame** — use `ends.stop.d_face` and `ends.seams_at_stop`.
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
  cm), and `ends.seams_at_stop` (distances to the APA seam, the cathode and the
  anode face);
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
* **`THRU`** — it does not stop: it crosses or exits a face (anode, top,
  bottom, front, back), crosses the cathode or the APA seam without a rise,
  runs into a dead region or a coverage hole with a flat profile, or charge
  continues past the fit end in `f_meas`. **An end in the middle of the volume
  is NOT a through-goer just because its profile is flat** — see rule 3 of the
  precedence order (v2).
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
   went the same way. The agent record had read "flat profile" as overriding the
   arm; it does not.

   **(v2) What counts as topology.** An arm or object that is clearly a second
   particle: it leaves the stop at an angle (a kink), or starts a separate
   piece. A short stub running **straight on** from the fit end (cos_fwd ≳ 0.9,
   no kink, a few cm) is weak: it can be the muon's own continuation. It does
   not decide the verdict on its own; it needs a rise or rule 3. The owner
   called two such items `THRU` and `MESSY` where the blind calibration had
   called them `STM_MICHEL` on the stub alone (`028084_27/57`, `029107_10/50`).
   An arm attached **at the stop vertex** counts as decay activity **even when
   it runs back along the muon body**. The owner took a backward arm doubling
   along the body as the Michel on `028084_1/39` and `028084_17/55`, where the
   blind calibration had called it body activity.
2. **Otherwise the Bragg rise decides.** With no electron-like activity, a
   rise at the end means `STM_ONLY`.
3. **(v2, corrected v3) A flat profile means `THRU` only where the track could
   have left.** The track stopped if all four hold:
   * **its own charge ends** in all three planes of `f_meas` at the fit end;
   * no dead band or coverage hole swallows the end;
   * the end is not at a face (anode: **(v4)** `face.x` ≲ 2 cm), the cathode or the APA
     seam (use the numbers);
   * **nothing continues past it** — not in the 3-D frames and the
     projections, **and (v4) not in `f_meas`'s grey cells**; no large object in
     the table starts at the stop and carries on;
   * **(v4) it is not the track's upper end** (rule 7).

   **(v4) Reading a continuation in the grey cells.** Through v3, `f_meas`
   showed only this cluster's own cells, so it could not show a continuation
   the clustering had put in another cluster. v4 draws the rest of the event in
   grey, **including other flash bundles** (which the 3-D frames hide under
   `bundle only`). A continuation is grey charge that **picks up the track's
   own line at the fit end and carries on in the same direction, in all three
   planes at matching time slices**. Grey that merely crosses the region, or
   sits in one plane only, is another object. A grey continuation makes the end
   `THRU` (or `FRAG_THRU` if you can tell the pieces are one track); say
   `CONTINUES:` in `--notes`. Grey is per-channel charge on its own scale; use
   it for *where*, not *how much*.

   In that case the missing rise is a measurement failure, not evidence of a
   through-goer (the Bragg peak can be unresolved, or carried by a short piece
   at the end). Call it `STM_ONLY`, or `STM_MICHEL` if there is decay activity,
   at most `medium`, with `--notes` starting `FLAT_STOP:`. The owner called
   both such calibration items stoppers where the blind scan had said
   `THRU` / `UNCLEAR` (`028084_15/102`, `028084_1/39`). The owner's four `THRU`
   items all end at a face, the APA seam, or on a near-isochronous track.

   **Exception — near-isochronous tracks** (x nearly constant along the track,
   so the whole track sits in a few time slices): their imaging and their ends
   are unreliable. Rule 3 does not apply; judge them `medium` / `low`.
4. **No measurement is not a flat profile.** If the last ~20 cm has too few
   live points to judge (a dead region, an APA2 coverage hole, a short
   track), a flat reading means nothing. Decide from topology or rule 3 if you
   can. Otherwise answer `UNCLEAR` and start `--notes` with `NO_MEASUREMENT:`.
5. **A face is not a stop test by itself.** An end 3 cm inside a face with a
   clear rise, or with a Michel, is a stopper; the fiducial inset is a rule
   for the chain's energy, not for this question. An end *at* a face with a
   flat profile is an exit (`--notes` `ANODE:` for the anode faces).
6. **(v2) A busy stop is `MESSY`.** If another large object runs along or
   across the muon near its end (a long cluster lying on the body, a shower),
   so that you cannot isolate where this track ends and what leaves it, answer
   `MESSY` (owner, `029107_10/50`).
7. **(v4) Direction: cosmic muons travel down.** A stopping cosmic muon stops at
   the **lower** end of its track, and its upper end reaches a face (usually
   the top, y ≈ 606) or a gap it crossed. Compute from `ends`:
   `dy = stop.xyz[1] − entry.xyz[1]` and `chord` = the distance between
   `entry.xyz` and `stop.xyz`. The fit end is the **upper end** when
   `dy ≥ 0.3 × chord` (the track rises at least ~17° towards the fit end).
   For a flatter track (`|dy| < 0.3 × chord`) geometry cannot tell direction,
   and this rule does not apply.

   At an **upper** fit end, a stop needs an upward-going particle. Among
   cosmics that is rare: a particle made in an interaction below, not a
   cosmic muon. So:
   * **Rule 3 does not apply at an upper end.** A flat profile there is where
     the track's charge **starts**: its upper part was not imaged, went into
     another cluster or bundle (look in `f_meas` for grey carrying its line on
     past that end), or the
     particle was produced in the volume. That is not a stop: `THRU`, with
     `--notes` starting `DIRECTION:`.
   * **Only a clear Bragg rise at the upper end makes it a stop.** Activity at
     that end is not enough on its own, because a production vertex looks like
     decay activity. With a clear rise, call `STM_ONLY` / `STM_MICHEL` at most
     `medium`, starting `DIRECTION:`, and say what the rise looked like.
   * **A rise at the lower end** (the chain's "entry" end) is `REVERSED:`. The
     verdict for the fit end is then `THRU` if that end is flat, or `UNCLEAR`.
   * The same test explains an entry at the **bottom** face (y ≈ 7.6): a track
     that enters through the floor is not a cosmic muon going down.

   This rule is new in v4. The previous round's rubric had no direction test,
   and 22 of its 183 hand stoppers stopped at the upper end, 14 of them on
   steep tracks (`|dy| ≥ 0.6 × chord`). Several had their other end within a
   few cm of the floor.

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
anchors in this rubric: the PDVD rubric's were wrong (doc pdvd/55 §17.3), and
on PDHD the scale differs between APAs (APA0 reads ~0.65). Some points poking
above the reference curve near the end is common and fine; staying under it is
also fine.

* **A ragged rise still counts.** In the owner's words: *"the Bragg peak is not as
  consistent, but I feel the scan is OK."* Point-to-point scatter of a factor
  of two is normal. The question is whether the profile climbs towards the end,
  not whether it climbs monotonically.
* Ignore zero-charge points. They are fit points with no measured charge, and
  they fake contrasts. On APA2 in particular, check `f_meas` before reading a
  run of low points as a collapse.
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
several cm into something the fit then bridges. An APA2 coverage hole (above)
is not a collapse either.

**When the collapse falls inside a single segment**, there is no separate row
to retag: the tag alphabet is per-object and cannot split one. Move the pin
anyway, since that is the measurement that matters. Tag the segment for what
*most* of it is, and say in the evidence that the row is split and which part
is which.

**(v3) What the bridge is.** The segment past the peak is the Michel when it
**carries charge**, i.e. it is the start of the electron, collapsed below the
plateau. When the fit instead bridges a **charge-free gap** to a detached piece
further out, the empty bridge is a fit artifact: tag it `delta / other`, and
tag the far piece by the attachment-distance rule (`gamma` beyond ~10 cm). The
overshoot and attribution rules then agree (wave-1 scanner, `029107_19/111`).

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
| `michel` | attached at the stop (**including an arm from the stop vertex that runs back along the body**, v2), or a piece past it consistent with the electron. **Within ~10 cm of the stop, direction does not discriminate**: a Michel is emitted isotropically, so a piece there is part of the decay whichever way it points |
| `gamma` | isolated, past the stop, clear of the muon body line, with `d_min` ≤ 60.0 cm (a hard edge, v2). **Charge is not a criterion**: a faint speck 40 cm out is an uncertain gamma, not a confident delta |
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
  (`gamma`). **(v2) Within ~5 cm of the stop a compact piece is part of the
  Michel even across a gap.** The ±45 cm frames cannot resolve a 3 cm gap, and
  the owner called two 3.2 cm clumps the Michel where the blind calibration had
  called them detached (`028084_26/109`). Beyond ~10 cm, a detached piece is
  `gamma`; the owner agreed on `028084_4/41` and `028084_23/114`.

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
  ~60 cm, clear of the body line, is `gamma` **in any direction**. This
  reverses the PDVD rule for this case (doc pdvd/55 §17.4 item 4).

**`MESSY`: tag every row `delta / other`**, kind `none`. **(v3) `UNCLEAR`:**
tag rows by what you can see (the chain `muon`, the rest `delta / other` if you
cannot place them); the kind follows the tags as always. The tag vocabulary
presupposes an identified stop.

**`THRU` / `FRAG_THRU`: tag fitted chain segments `muon` and everything else
`delta / other`**, kind `none`. `gamma` and `michel` are defined relative to a
stop. If you find yourself wanting `gamma` or `michel` on a `THRU`, the real
question is whether the verdict is right (see the precedence order).

**Key shapes vary, and `mkv.py` checks them.** A fitted segment's key is its
bare id (`"146004"`); an unfitted cluster's key is `C` plus its id (`"C218"`).
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

Doc pdvd/55 §17.4 found 3 PDVD records that broke this by hand.

## Confidence

* `high` — the picture settles it.
* `medium` — you had to weigh two readings.
* `low` — you are guessing; say why in `--notes`.

Confidence describes the **verdict**. Put doubt about a tag or the kind (for
example, a far speck that decides `both` vs `attached`) in `--notes`, not in the
confidence.

On PDVD a double scan showed this field is calibrated: `high` calls reproduced
24 of 24 times, and `low` calls did not reproduce at all. Mark a genuinely
uncertain call `high` and you corrupt that calibration.

## Evidence — required, one real paragraph per item

Write what you actually saw, so the call can be audited later without
re-opening the pictures. Cover:
* the length, and where the track entered and stopped relative to the faces,
  the cathode and the APA seam (numbers from `ends`);
* which APA the stop is in;
* what the dQ/dx profile did in the last 10–20 cm relative to its own plateau;
* what sits past the stop, where (distance, direction) and why it reads as
  Michel / gamma / delta;
* anything that made the call hard.

An example of the *shape* (invented numbers, no real item):

> 140 cm, enters through the top 0.4 cm from y = 606, stops 63 cm from the
> nearest face and 88 cm from the cathode, in APA1. The profile sits on its
> plateau and climbs to about twice it over the last 14 cm, following the muon
> curve; ragged, one point back at plateau, but a rise. A 9 cm two-piece arm
> leaves the stop at about 50 deg at plateau charge — a muon at its Bragg peak
> cannot continue at MIP, so the arm is a second particle, and 9 cm is the
> right range for a ~20 MeV electron. One speck 31 cm out, forward and clear of
> the body line: gamma. Three pieces 25–60 cm back along the body at cos_fwd
> −0.9 or below: delta / other. No dead region at the stop in any plane.

Do not write "looks good" or "clear STM". `mkv.py` refuses anything under 12
words, but the bar is the paragraph above, not the word count.

## How to write the record

One command per item, after you have looked at the frames:

```
python3 <MKV> '<event>/<cluster>' <VERDICT> '<michel_kind>' <confidence> \
  --shots-dir '<SHOTS>' --out-dir '<OUT>' \
  --tags 'muon:146002 michel:146004 gamma:C218' \
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
  is at the cathode crossing), `SEAM:` (the end is at the APA seam), `ANODE:`
  (the end is at an anode face, i.e. **(v4)** `face.x` ≲ 2 cm — see The detector),
  `FACE:` (the end is at the top / bottom / front / back face), `DEAD:` (the
  end runs into a dead band or a coverage hole), `ISOCHRONOUS:` (the track is
  near-isochronous), `REVERSED:` (the chain's "entry" end looks like the real
  stop candidate; the verdict is still for the fit end, and the far end cannot
  be judged from these frames), **(v4)** `DIRECTION:` (rule 7: the fit end is
  the track's upper end), `CONTINUES:` (grey charge in `f_meas` continues the
  track past the fit end). An item may carry several, separated by `; `.
* **(v3) Stubs on a `THRU`.** A fitted piece with real charge continuing
  straight on past a `THRU` end is `muon`. A row with no charge value or zero
  size is `delta / other`.
* **(v3) The near-isochronous exception limits rule 3 only.** A
  near-isochronous track that plainly crosses two faces can still be a `high`
  `THRU`.

If `mkv.py` refuses, fix the record and re-run it. Never hand-write the JSON.

## What changed from the PDVD rubric, and why

| # | change | source |
|---|---|---|
| 1 | detector section: horizontal drift, central cathode, APA seam at z ≈ 231, APA0 / APA2 charge deficits, wrapped U/V | doc pdvd/50 r2, doc pdhd/10 |
| 2 | verdict-blind: `is_stm`, `reject_names`, `in_fv`, `flow` removed; chain typing named as residual leak | owner 2026-09-11; doc pdvd/70 §9 |
| 3 | precedence order: topology outranks a flat-looking profile; no-measurement ≠ flat; a face is not a stop test | docs pdvd/68, pdvd/70 §9.2–9.4 |
| 4 | no absolute charge anchors; plateau-relative only | doc pdvd/55 §17.3 |
| 5 | `straddles the stop` withdrawn (was used 0 / 569 times and contradicted the no-split rule) | doc pdvd/55 §17.4 item 2 |
| 6 | far specks on `STM_ONLY` are capture gammas in any direction | doc pdvd/55 §17.4 item 4 |
| 7 | verdict-vs-kind consistency enforced by `mkv.py` | doc pdvd/55 §17.4 item 1 |
| 8 | `h_dqdx_zoom.png` supplied; amber selection band removed from the frames | doc pdhd/18 §3 |
| 9 | counted `--notes` prefixes | doc pdhd/18 |

**v2 (after the blind calibration on the owner's 30, doc pdhd/18 §5).** The
v1 blind scan agreed with the owner on 18 of 18 `high` verdicts and 3 of 11
`medium`. Every change below is one the owner's verdicts on those 30 items
decide:

| # | change | owner evidence |
|---|---|---|
| 10 | an end mid-volume where the charge ends in all planes is a stop, even with a flat profile (`FLAT_STOP:`); near-isochronous tracks excepted | `028084_15/102`, `028084_1/39` (stoppers); four owner `THRU` all at a face / seam / isochronous |
| 11 | a collinear straight-on stub is weak topology; it does not decide the verdict alone | `028084_27/57` (`THRU`), `029107_10/50` (`MESSY`) |
| 12 | an arm from the stop vertex running back along the body is the Michel | `028084_1/39`, `028084_17/55` |
| 13 | within ~5 cm, a compact piece is the Michel even across a gap | `028084_26/109` |
| 14 | a busy stop is `MESSY` | `029107_10/50` |
| 15 | gamma radius a hard 60.0 cm; degenerate rows are `delta / other`; confidence scopes the verdict; `ANODE:` / `REVERSED:` prefixes | scanner objections, calibration wave |

**v3 (after main wave 1).** One factual correction, found by a wave-1 scanner.
Six of its fifteen fit ends sat 5.9–7.1 cm "inside" the anode face. Measured
on all 317 items: no end reaches |x| > 352.2 cm, while the box and the `ends`
numbers put the face at 358 cm.

| # | change | evidence |
|---|---|---|
| 16 | `face.x` ≲ 8 cm is AT the anode: an exit with a flat profile | the |x| distribution of all 634 ends (doc pdhd/18 §6.2) |
| 17 | `FACE:` / `DEAD:` / `ISOCHRONOUS:` prefixes; THRU stubs; the isochronous exception limits rule 3 only | wave-1 scanner objections |
| 18 | an overshoot bridge through a charge-free gap is a fit artifact, not the Michel; `UNCLEAR` tagging | wave-1 scanner, `029107_19/111` |
| 19 | APA0 patchy collection charge is a hole, not a flat profile; the horizontal line in `f_meas` named | wave-1 scanners |
| 20 | rule 3's "nothing continues" is read off the 3-D frames, not `f_meas`: `f_meas` draws only this cluster's own cells | a wave-1 scanner; `prep_stm_michel_scan.proj_cells` reads one `T_proj_data` row, the muon's cluster |

The wave-1 items whose stop sits in that band, and the wave-1 `FLAT_STOP`
calls (rule 3 as written in v2 leaned on `f_meas` for "nothing continues"),
were re-scanned under v3. The re-scan supersedes their v2 record
(`resolve.py --supersede`); the v2 records are kept and listed.

Left unchanged: the overshoot rule. On `028084_23/114` the owner placed no
pin where the blind scan moved it 2.7 cm, but on PDVD the owner moved pins by
1.2–9.5 cm (doc pdvd/70 §9.4). One item does not set a length threshold.

**v4 (the review round, doc pdhd/19).** Two display defects the smx18 scanners
found were fixed, and the direction question it left open is now a rule.

| # | change | evidence |
|---|---|---|
| 21 | the box and every `ends` x number use the job's active volume: anode \|x\| = 352.1, cathode face \|x\| = 0.16. `face.x` ≲ 2 cm is at the anode (was ≲ 8); do not subtract 6 cm | job-log `AnodePlane` sensvol; `smgeom.ENVELOPE["pdhd"]` (doc pdhd/19) |
| 22 | `f_meas` draws the event's other live charge in grey (±200 channels / slices of the fit end, all bundles). Rule 3's "nothing continues" now includes it; `CONTINUES:` prefix | prep `--ctx-cells`, gated on the fitted cells (doc pdhd/19) |
| 23 | U/V coloured cells read 2–3× the channel charge on wrapped planes | ctpc vs `T_proj_data`, doc pdhd/19 |
| 24 | rule 7, direction: an upper fit end (`dy ≥ 0.3 × chord`) is not a flat stop; only a clear rise makes it a stop, at most `medium`; `DIRECTION:` prefix | 22 of 183 smx18 hand stoppers stopped at the upper end (doc pdhd/18 §7, queue tier C) |
| 25 | the previous round's files are off limits | the review re-scans items the previous round judged |
