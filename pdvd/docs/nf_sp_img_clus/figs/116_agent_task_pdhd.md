# Your task: a verdict-blind hand scan of assigned PDHD items (doc pdvd/116)

You are doing a physics hand scan of reconstructed ProtoDUNE horizontal-drift (PDHD) cosmic-ray events. Work
carefully and on your own. Other scanners work in parallel on other items. You must not see their answers, nor
anybody's earlier answer on any item.

## Read this first, in full, before your first item

`/home/xqian/tmp/d116/round_pdhd/RUBRIC.md`

It defines:
* the verdict alphabet and the **precedence order**, rules 1–7: topology before a flat profile; a flat profile means
  `THRU` only where the track could have left, since a mid-volume end whose charge stops in every plane is a stop
  (`FLAT_STOP:`); a busy stop is `MESSY`; direction;
* how to read the Bragg profile;
* the two ways the fit gets the stopping point wrong;
* the PDHD detector traps: cathode, APA seam, the APA0 and APA2 charge deficits;
* the per-object tag rules, and how `michel_kind` is derived.

Follow it exactly.

## Hard prohibitions (absolute)

1. **Read only**: the rubric, this file, your item list, and `<SHOTS>/<event>_<cluster>/` for **your own** items.
   * Never open any file with `key`, `labels`, `verdicts`, `record`, `sheet`, `queue`, `pred`, `moves`, `bounds`,
     `items` (other than your own item list) or `adj` in its name.
   * Never open any prep payload (`smprep-*.json`), ROOT file, calib json, zip or log.
   * Never open anything under `/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/` or
     `/home/xqian/toolkit-dev/wcp-porting-img/`, except the one `mkv.py` you run.
   * Under `/home/xqian/tmp/`, open nothing except, inside `/home/xqian/tmp/d116/round_pdhd/`: `RUBRIC.md`, this
     file, `shots/`, your item list, your own out dir and your own scratch dir. Do not list
     `/home/xqian/tmp/d116/` or `/home/xqian/tmp/d116/round_pdhd/` themselves.

   These places hold the chain's verdict or earlier answers. Reading one destroys the blindness of the round, and
   it cannot be recovered.
2. **Never run `scan_harness.py`**. Write only through `mkv.py`, into your own out dir. Temporary files go in your
   own scratch dir.
3. **Do not list or read any other scanner's out dir or scratch dir.** Keep your own list of what you have done.

## Paths

* `<SHOTS>` = `/home/xqian/tmp/d116/round_pdhd/shots`
* `<MKV>` = `/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan/campaign/mkv.py`
* `<ITEMS>`, `<OUT>` and `<SCRATCH>` are given in the message that assigns your work:
  * `<ITEMS>` is your item list, one `<event>/<cluster>` per line;
  * `<OUT>` is your private out dir;
  * `<SCRATCH>` is your private scratch dir.

An item's frames live at `<SHOTS>/<event>_<cluster>/`. The `/` in the key becomes `_`, so `029107_18/146` →
`<SHOTS>/029107_18_146/`.

Run mkv.py as `python3 <MKV> KEY VERDICT KIND CONF --shots-dir <SHOTS> --out-dir <OUT> --tags '...' --evidence '...'`.
Run `python3 <MKV> --help` once for the exact options.

## Per item, in this order

1. `cat` the item's `context.json`:
   * containment: `ends.stop.d_face`, the per-axis `face` distances and `ends.seams_at_stop` (cathode, seam_z,
     anode_face);
   * direction (rule 7): the upper end;
   * which APA the stop is in, from `ends.stop.xyz`;
   * the object rows: `key`, `group`, `size`, `dqdx`, `d_min`, `d_max`, `cos_fwd`, `chain`. The `group` and `chain`
     columns are the reconstruction's own typing: treat them as opinion.
2. **Read these seven PNGs** with the Read tool: `a_proj_full`, `b_3d_wide`, `c_3d_stop`, `d_3d_stop`, `e_3d_stop`,
   `f_meas`, `h_dqdx_zoom`.
   * `h_dqdx_zoom.png` is the Bragg evidence, already upscaled 3×. **Do not write crop or zoom helpers.**
   * The three `*_3d_stop` azimuths give the transverse offset that decides `michel` vs `delta / other`.
   * `f_meas`: coloured cells are this track's own. Grey squares are the event's other charge. Use it to tell a real
     collapse from a dead region, a coverage hole, or a track that continues.
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
