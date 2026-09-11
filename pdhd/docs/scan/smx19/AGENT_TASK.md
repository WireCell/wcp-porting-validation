# Your task: verdict-blind review scan of assigned PDHD items (doc pdhd/19)

You are doing a physics hand scan of reconstructed ProtoDUNE horizontal-drift
(PDHD) cosmic-ray events. Work carefully and independently. These items were
scanned once before; you are the independent second look, on a corrected
display, and you must not see the first answer.

## Read this first — in full, before your first item

`/home/xqian/tmp/h19/RUBRIC.md` (rubric **v4**)

It defines:
* the verdict alphabet and the **precedence order**, rules 1–7: topology
  before a flat profile; a flat profile means `THRU` only where the track could
  have left (`FLAT_STOP:` otherwise); a busy stop is `MESSY`; **rule 7
  (new): cosmic muons travel down, so a flat upper end is where a track
  starts, not where it stops**;
* how to read the Bragg profile;
* the two ways the fit gets the stopping point wrong;
* the PDHD detector traps (cathode, APA seam, the APA0 and APA2 charge deficits);
* **what v4 changed in the display**: the box and the `ends` numbers are now
  the true active volume (anode |x| = 352.1, do not subtract 6 cm), and
  `f_meas` now draws the event's other charge as **grey squares**;
* the per-object tag rules, and how `michel_kind` is derived.

Follow it exactly.

## Hard prohibitions — absolute

1. **Read only**: the rubric, this file, your item list, and
   `<SHOTS>/<event>_<cluster>/` for **your own** items.
   * Never open any file with `key`, `labels`, `verdicts` or `queue` in its
     name.
   * Never open any prep payload (`smprep-*.json`), ROOT file, calib json or
     zip, or anything under `pdhd/work/`, `pdhd/docs/` or `pdvd/docs/`.
   * **Never open anything under `/home/xqian/tmp/h18/`** — that is the
     previous scan of these same items.
   * Under `/home/xqian/tmp/h19/`, open only `RUBRIC.md`, this file, `shots/`,
     your item list and your own out dir.

   These hold the chain's verdict or the previous scanner's answers. Reading
   one destroys the independence this round exists to measure, with no way to
   recover it.
2. **Never run `scan_harness.py`**, and never write anywhere except your own
   out dir, through `mkv.py`.
3. **Do not list or read any other scanner's out dir** (`v_parts/*`).

## Paths

* shots dir `<SHOTS>` = `/home/xqian/tmp/h19/shots`
* `<MKV>` = `/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan/campaign/mkv.py`
* `<ITEMS>` (your item list, one `<event>/<cluster>` per line) and `<OUT>`
  (your private out dir) are given in the message that assigns your work.

An item's frames live at `<SHOTS>/<event>_<cluster>/`: the `/` in the key
becomes `_`, so `029107_18/146` → `<SHOTS>/029107_18_146/`.

## Per item, in this order

1. `cat` the item's `context.json`:
   * containment: `ends.stop.d_face`, the per-axis `face` distances and
     `ends.seams_at_stop` (cathode, seam_z, anode_face) — **true distances now**;
   * **direction (rule 7)**: `dy = stop.xyz[1] − entry.xyz[1]` and the chord
     between the two ends; the fit end is the upper end when `dy ≥ 0.3 × chord`;
   * which APA the stop is in, from `ends.stop.xyz`;
   * the object rows: `key`, `group`, `size`, `dqdx`, `d_min`, `d_max`,
     `cos_fwd`, `chain`.
2. **Read these seven PNGs** with the Read tool: `a_proj_full`, `b_3d_wide`,
   `c_3d_stop`, `d_3d_stop`, `e_3d_stop`, `f_meas`, `h_dqdx_zoom`.
   * `h_dqdx_zoom.png` is the Bragg evidence, already upscaled 3×. **Do not
     write crop or zoom helpers.**
   * The three `*_3d_stop` azimuths give the transverse offset that decides
     `michel` vs `delta / other`.
   * `f_meas`: coloured cells are this track's own; **grey squares** in the
     `measured` column are everything else the wires saw. Does the track's
     charge end, run into a dead band, or continue in grey along its own line
     in all three planes?
3. Decide, in the rubric's precedence order:
   * the verdict and confidence;
   * a tag for **every** object row;
   * the pin, if the fit overshoots.

   `michel_kind` follows mechanically from your tags.
4. **Write the record immediately with `<MKV>`**, before the next item. If
   `mkv.py` refuses, read its message, fix the record and re-run it.

If you run low on context, **stop and report what you completed**. Everything
written is banked and the rest is reassigned.

## Judgement

Judge each item on its own evidence. These items were picked for a second
look, so expect them to be harder than average. That is not a reason to lean
in either direction. Do not aim for any particular mix of verdicts. Mark
confidence honestly: a genuinely uncertain call marked `high` is worse than the
uncertainty. Say in the evidence paragraph what made a call hard.

## Report back (your final message)

* one line per item: `key — VERDICT / michel_kind / confidence [pin_rr] [notes prefix]`
* the keys where you moved the pin (overshoot), saw an undershoot, applied
  rule 7 (`DIRECTION:`), or saw a grey continuation (`CONTINUES:`)
* the items you found hardest, and the competing reading for each
* anything in the rubric that was ambiguous, or that you had to stretch, with
  an item key. Do not suggest the rubric be changed mid-round; just report.
