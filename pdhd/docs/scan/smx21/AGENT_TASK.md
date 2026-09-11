# Your task: a verdict-blind scan of assigned PDHD items (doc pdhd/19 §9, rubric v5)

You are doing a physics hand scan of reconstructed ProtoDUNE horizontal-drift
(PDHD) cosmic-ray events. Work carefully and on your own. These items were
scanned before. You are an independent look under a corrected rubric, and you
must not see any earlier answer. A second scanner judges the same items
separately.

## Read this first, in full, before your first item

`/home/xqian/tmp/h21/RUBRIC.md` (rubric **v5**)

It defines:
* the verdict alphabet and the **precedence order**, rules 1–7. **Read
  rules 1, 3 and 7 closely: v5 changed them.** Rule 7 (cosmic muons travel
  down) now lets topology decide at an upper end, as rule 1 does everywhere;
* how to read the Bragg profile, and the two ways the fit gets the stopping
  point wrong;
* the PDHD detector traps (cathode, APA seam, the APA0 and APA2 charge
  deficits);
* the display: the box and the `ends` numbers are the true active volume, and
  `f_meas` draws the event's other charge as **grey squares**;
* the per-object tag rules, and how `michel_kind` is derived.

Follow it exactly.

## Hard prohibitions (absolute)

1. **Read only** the rubric, this file, your item list, and
   `<SHOTS>/<event>_<cluster>/` for **your own** items.
   * Never open any file with `key`, `labels`, `verdicts`, `queue`, `rulings`,
     `sheet` or `adj` in its name, and never open `ops/` or `lbl_*`.
   * Never open any prep payload (`smprep-*.json`), ROOT file, calib json or
     zip, or anything under `pdhd/work/`, `pdhd/docs/` or `pdvd/docs/`.
   * **Never open anything under `/home/xqian/tmp/h18/`, `/home/xqian/tmp/h19/`
     or `/home/xqian/tmp/h20/`.** Those are earlier scans. Read frames only
     through `<SHOTS>`.
   * Under `/home/xqian/tmp/h21/`, open only `RUBRIC.md`, this file, `shots/`,
     your item list, your own out dir and your own scratch dir.

   These hold the chain's verdict or the earlier answers. Reading one destroys
   the independence this pass exists to provide, and it cannot be recovered.
2. **Never run `scan_harness.py`**. Write only through `mkv.py`, into your own
   out dir. Temporary files go in your own scratch dir, never anywhere shared.
3. **Do not list or read the other scanner's out dir or scratch dir.**

## Paths

* `<SHOTS>` = `/home/xqian/tmp/h21/shots`
* `<MKV>` = `/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan/campaign/mkv.py`
* `<ITEMS>` (your item list, one `<event>/<cluster>` per line), `<OUT>` (your
  private out dir) and `<SCRATCH>` (your private scratch dir) are given in the
  message that assigns your work.

An item's frames are in `<SHOTS>/<event>_<cluster>/`. The `/` in the key
becomes `_`, so `029107_18/146` → `<SHOTS>/029107_18_146/`.

## Per item, in this order

1. `cat` the item's `context.json`:
   * containment: `ends.stop.d_face`, the per-axis `face` distances and
     `ends.seams_at_stop` (cathode, seam_z, anode_face);
   * **direction (rule 7)**: `dy = stop.xyz[1] − entry.xyz[1]` and the chord
     between the two ends. The fit end is the upper end when `dy ≥ 0.3 × chord`;
   * which APA the stop is in, from `ends.stop.xyz`;
   * the object rows: `key`, `group`, `size`, `dqdx`, `d_min`, `d_max`,
     `cos_fwd`, `chain`.
2. **Read these seven PNGs** with the Read tool: `a_proj_full`, `b_3d_wide`,
   `c_3d_stop`, `d_3d_stop`, `e_3d_stop`, `f_meas`, `h_dqdx_zoom`.
   * `h_dqdx_zoom.png` is the Bragg evidence, already upscaled 3×. Do not
     write crop or zoom helpers. If you must make any file, it goes in
     `<SCRATCH>`.
   * The three `*_3d_stop` azimuths give the angle and transverse offset of
     anything leaving the end. That decides topology (rule 1) and `michel` vs
     `delta / other`.
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

## Judgement

Judge each item on its own evidence. These items were picked for another
look, so expect them to be hard. That is not a reason to lean either way. Do
not aim for any particular mix of verdicts. Mark confidence honestly. Say in
the evidence paragraph what made a call hard, and name the competing reading.

## Report back (your final message)

* one line per item: `key — VERDICT / michel_kind / confidence [pin_rr] [notes prefix]`
* for each item, one sentence: what, if anything, leaves the fit end (angle,
  length, attached or detached), and whether you counted it as topology
* the items you found hardest, with the competing reading for each
* anything in the rubric that was ambiguous or that you had to stretch, with
  an item key. Do not propose rubric changes mid-round; just report.
