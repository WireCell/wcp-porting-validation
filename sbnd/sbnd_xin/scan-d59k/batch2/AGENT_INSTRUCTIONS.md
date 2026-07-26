# Batch-2 hand scan: STM vs neutrino candidate (SBND d59k) — sub-agent instructions

You are scanning one slice of a 383-event sample from an SBND neutrino-selection
study. For **every in-beam bundle** in your slice you decide one thing: **is this
a stopping muon (STM) or is it still a neutrino candidate (nu)?**

Full key: `/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin/docs/61_nusel-handscan-key.md`
(read §1–§3 — but this file overrides it where they differ).

## Hard rules

1. **Write exactly one file: your output TSV.** Nothing else, anywhere. Do not
   run the viewer, do not start a bokeh server, do not run a browser, do not
   write into any `nusel_labels/` directory, do not edit any file under
   `sbnd_xin/`. All the evidence you need is already on disk (below).
2. **The verdict field has exactly two legal values: `STM` or `nu`.** Nothing
   else — not `cosmic`, not `nu-weak`, not `unclear`. A bundle that is not a
   stopping muon *stays a neutrino candidate*, because the TGM- and LM-tagged
   cosmics are already removed from this sample. Everything about *why* goes in
   the separate `quality` field.
3. **Do not decide from the tagger's own columns** (`auto_label`, `stm`, `tgm`,
   `fc`, `stmfit`, and the "rej:"/"pass" text in the dQ/dx panel). Those are what
   this scan is testing. You may read them, and you should say when you disagree,
   but the verdict must rest on the images and the geometry.
4. Judge every bundle in your skeleton file. If evidence is missing (no dQ/dx
   panel, no drawn points), still give a verdict and say so in the reason.

## quality — one of exactly these five

| quality | meaning |
|---|---|
| `clean` | verdict well supported by both the picture and the dQ/dx |
| `weak` | plausible but featureless (single track, no vertex activity, no fit) |
| `cosmic-like` | not a stopping muon, but the topology reads cosmic (through-going, or truncated with flat MIP dQ/dx) |
| `junk` | the cluster is not one physical object (sparse disconnected blobs / fragments) |
| `unclear` | a reconstruction pathology dominates; the owner should look |

`conf` is `high` / `med` / `low`.

## The physics

1. **STM = stopping muon.** One end must **cross a boundary** (the entry), the
   other must **stop inside** with a Bragg rise. A Michel electron may follow —
   a short low-dQ/dx spray or hook at the stopping end.
2. **Neutrino candidates have vertex structure**: kinks, ≥2 prongs from one
   point, a short high-dQ/dx stub (proton) plus a long MIP prong.
3. **Fully contained ⇒ not STM.** A muon that stops inside had to enter from
   outside. Contained + vertex structure is the neutrino signature.
4. **Grazing ≠ entering.** A track running *parallel* to a wall and touching it
   has not entered through it (reference: evt50787, 6 cm of z travel over a
   49 cm track at z≈497). A muon entering through a wall travels *across* it.
5. **Beam direction is evidence.** SBND's beam is +z. A long prong going +z is
   ν-like; a near-vertical track (|dy| dominant, tiny dz) is cosmic-shaped.

### The dQ/dx panel is the decisive evidence, not the topology

`<evt>-r<row>-dqdx.png` plots fitted dQ/dx vs residual range, with the **muon
stopping expectation** (green) and the **flat MIP line** (grey, 56 ke/cm):

* data **follows the green curve up to ~165 ke/cm at rr→0 ⇒ it really stopped**
  (reference: evt48301, ks1=0.010; evt50765, ks1=0.018);
* data **stays flat on the grey MIP line right down to rr=0 ⇒ nothing stopped
  there** — the track was cut off (broken cluster, out of readout, exiting).
  Reference evt48895: the tagger called it STM and the owner says that is wrong,
  purely on this;
* data **spikes well ABOVE the green curve (250–300 ke/cm) in the last few cm ⇒
  proton-like, not a muon** (reference: evt51051, a 10 cm stub at 3.24 MIP).

A Bragg peak **alone is not STM** — the muon of a contained νμ interaction stops
too. STM needs Bragg **and** a boundary-crossing entry.

### Traps that have already cost time

* **Containment: use the numbers, not your eye.** The FV box is inset from the
  detector box by only 2.0/2.5/3.0 cm, so at full-detector zoom the two lines
  coincide. Use the `bbox` in `info.json` (`lo`/`hi` = [x,y,z] cm of the drawn
  bundle) against:
  * `DET_BOX  x=(-201.05, 201.05)  y=(-199.312, 199.312)  z=(0.85, 500.15)`
  * `FV_BOX   x=(-199.05, 199.05)  y=(-196.812, 196.812)  z=(3.85, 497.15)`
  A bundle end within ~3 cm of a DET_BOX face has crossed that boundary; an end
  ≥10 cm inside every face is contained. x=±201 is the anode plane, x≈0 is the
  cathode.
* **Orange squares are merge fragments of the main cluster**, not a second track
  and not a kink.
* **A slope break exactly at x=0 is the cathode**, where two clusters get joined
  and a ~1.5 cm offset lives. Not a physics kink.
* **Constant-z tracks are badly measured** (e.g. dz=7 cm over 165 cm): that is
  the prolonged-W geometry where collection ROIs break, so truncation is the
  default explanation, not containment.
* **Junk has a density signature.** Real tracks run 5–10 points/cm; junk is
  ≲3 points/cm in disconnected blobs spread over 100+ cm. The skeleton gives you
  `npts_main` and `len_cm` — use the ratio to pre-flag, then confirm in the image.
* **Grey points are context**, not part of the bundle: the display is in
  `--charge-src pr` mode, so pieces the un-merge removed from this cluster are
  drawn grey. If the drawn point count (`bbox.n` in `info.json`) is far below the
  skeleton's `npts_main`, the bundle is a small post-un-merge main inside a
  bigger blob — that alone does not make it junk.

## Your evidence, per bundle

In your shots directory:

* `<evt>-r<row>-dqdx.png` — the dQ/dx panel + the fit's own text. **Always read
  this one first.**
* `<evt>-r<row>-zoom.png` — the three projections (X-Y, Y-Z, X-Z) zoomed to the
  bundle: topology, kinks, branches, ends. **Always read this.**
* `<evt>-r<row>-ctx.png` — the same projections at full-detector range: where in
  the TPC it sits. Read only when the `bbox` numbers leave the entry/containment
  question open (that is the common case for an STM call, so expect to read it
  for candidates you are inclined to call STM).
* `info.json` — `bundles[]`, one entry per bundle in grab order, each with
  `event`, `row`, `table_row`, `bbox` (`lo`/`hi`/`n` = drawn points) and `divs`
  (every text panel the display drew, including the STM fit text). Read this
  first, in full — it is small and it gives you numbers for free.

`<row>` in the file names is the display row index, and `info.json`'s
`bundles[].row` matches it. Map a row to a skeleton line by `event` +
`table_row[5]`, which is `main_id` (0-indexed; the columns are
`# grp t(us) PE(grp) beam main clusters npts len(cm) tgm/stm/fc/lm stm_fit prev auto_label ...`).
In `divs`, the entry containing `STM fit` is the fit text (kink/exit_L/ks1 …, the
same text the dQ/dx png shows) and the one containing `flash time` is the
bundle's own info table.

## Output — exactly this, tab-separated

Write **one row per bundle in your skeleton file**, in the same order:

```
event	main_id	verdict	quality	conf	reason
54175	18	nu	cosmic-like	high	flat 50-60 ke/cm to rr=0 with no Bragg, so the "stopping" end is truncated; enters the anode x=-201 and the far end stops mid-detector
```

* Header line exactly `event	main_id	verdict	quality	conf	reason`.
* `event` and `main_id` copied verbatim from the skeleton (they are the join key
  — a typo silently drops the row).
* **Do not** add `flash_gid`, `t_us` or any other column; they are joined in
  later from the skeleton.
* `reason` is one line naming the **evidence** — "flat MIP to rr=0", "vertex +
  3 cm stub at 3x MIP", "1.4 pts/cm in 8 blobs", "enters top wall y=199, both
  arms reach the anode". No tabs and no newlines inside it. Mention explicitly
  when you disagree with the tagger's label and why.
* Report anything odd (tagger bugs, clustering pathologies) in the reason text.
  **Do not fix anything.**

When done, reply with: how many bundles you judged, the STM/nu split, the quality
tally, and the events you found genuinely hard (with row keys) so a human can
re-check those first.
