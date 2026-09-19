# Your task: a verdict-blind hand scan of PDVD items (doc pdvd/116)

You are doing a physics hand scan of reconstructed ProtoDUNE vertical-drift (PDVD) cosmic-ray events. Work carefully
and on your own. Other scanners work in parallel on other items. You must not see their answers, nor anybody's
earlier answer on any item.

## Read this first, in full, before your first item

`/home/xqian/tmp/d116/round_pdvd/RUBRIC.md`

It defines:
* the verdict alphabet and the **precedence order**, rules 1–7 (topology first; a flat profile means `THRU` only
  where the track could have left; cosmic muons travel down, and **on PDVD down is −x**);
* the PDVD detector: vertical drift along x, the horizontal cathode at x ≈ 0, the anode planes at x = ±339.9, the
  CRU seams, and **the display trap that the panels' vertical axes are not "up"**;
* how to read the Bragg profile, and the two ways the fit gets the stopping point wrong;
* the display: `f_meas` draws the event's other charge as **grey squares**;
* the per-object tag rules, and how `michel_kind` is derived.

Follow it exactly.

## Hard prohibitions (absolute)

1. **Read only**: the rubric, this file, your item list, and `<SHOTS>/<event>_<cluster>/` for **your own** items.
   * Never open any file with `key`, `labels`, `verdicts`, `record`, `carried`, `sheet`, `queue`, `pred`, `prereg`,
     `moves`, `bounds`, `items` (other than your own item list) or `adj` in its name.
   * Never open any prep payload (`smprep-*.json`), ROOT file, calib json, zip or log.
   * Never open anything under `/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/` or
     `/home/xqian/toolkit-dev/wcp-porting-img/`, except the one `mkv.py` you run.
   * Under `/home/xqian/tmp/`, open nothing except, inside `/home/xqian/tmp/d116/round_pdvd/`: `RUBRIC.md`, this
     file, `shots/`, your item list, your own out dir and your own scratch dir. Do not list
     `/home/xqian/tmp/d116/` or `/home/xqian/tmp/d116/round_pdvd/` themselves.

   These places hold the chain's verdict or earlier answers. Reading one destroys the blindness of the round, and
   it cannot be recovered.
2. **Never run `scan_harness.py`**. Write only through `mkv.py`, into your own out dir. Temporary files go in your
   own scratch dir, never anywhere shared.
3. **Do not list or read any other scanner's out dir or scratch dir.** Keep your own list of what you have done.

## Paths

* `<SHOTS>` = `/home/xqian/tmp/d116/round_pdvd/shots`
* `<MKV>` = `/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan/campaign/mkv.py`
* `<ITEMS>`, `<OUT>` and `<SCRATCH>` are given in the message that assigns your work:
  * `<ITEMS>` is your item list, one `<event>/<cluster>` per line;
  * `<OUT>` is your private out dir;
  * `<SCRATCH>` is your private scratch dir.

An item's frames are in `<SHOTS>/<event>_<cluster>/`. The `/` in the key becomes `_`, so `039252_17/88` →
`<SHOTS>/039252_17_88/`.

Run mkv.py as `python3 <MKV> KEY VERDICT KIND CONF --shots-dir <SHOTS> --out-dir <OUT> --tags '...' --evidence '...'`.
Run `python3 <MKV> --help` once for the exact options.

## Per item, in this order

1. `cat` the item's `context.json`:
   * containment: `ends.stop.d_face`, the per-axis `face` distances and `ends.seams_at_stop` (`anode_face`,
     `cathode`, `seam_y`, `seam_z`);
   * **direction (rule 7)**: `dx = stop.xyz[0] − entry.xyz[0]` and the chord between the two ends. The fit end is the
     upper end when `dx ≥ 0.3 × chord`. **x, not y**;
   * which volume the stop is in (top x > 0, bottom x < 0);
   * the object rows: `key`, `group`, `size`, `dqdx`, `d_min`, `d_max`, `cos_fwd`, `chain`. The `group` and `chain`
     columns are the reconstruction's own typing: treat them as opinion.
2. **Read these seven PNGs** with the Read tool: `a_proj_full`, `b_3d_wide`, `c_3d_stop`, `d_3d_stop`, `e_3d_stop`,
   `f_meas`, `h_dqdx_zoom`.
   * `h_dqdx_zoom.png` is the Bragg evidence, already upscaled 3×. Do not write crop or zoom helpers.
   * The three `*_3d_stop` azimuths give the angle and transverse offset of anything leaving the end. That decides
     topology (rule 1) and `michel` vs `delta / other`.
   * `f_meas`: coloured cells are this track's own; **grey squares** in the `measured` column are everything else
     the wires saw. Does the track's charge end, run into a dead band, or continue in grey along its own line in all
     three planes?
3. Decide, in the rubric's precedence order:
   * the verdict and confidence;
   * a tag for **every** object row;
   * the pin, if the fit overshoots.

   `michel_kind` follows mechanically from your tags.
4. **Write the record immediately with `<MKV>`**, before the next item. If `mkv.py` refuses, read its message, fix
   the record and re-run it.

If you run low on context, **stop and report what you completed**. Everything written is banked, and the rest is
reassigned.

## Judgement

Judge each item on its own evidence. Do not aim for any particular mix of verdicts, and do not let one item's call
influence the next. Mark confidence honestly: `medium` and `low` rows are the ones the owner will look at, and a
genuinely uncertain call marked `high` is worse than the uncertainty. Say in the evidence paragraph what made a call
hard.

## Report back (your final message)

* one line per item: `key — VERDICT / michel_kind / confidence [pin_rr] [notes prefix]`
* the keys where you moved the pin (overshoot) and where you saw an undershoot
* the items you found hardest, and the competing reading for each
* anything in the rubric that was ambiguous, or that you had to stretch. Be concrete, with an item key.
