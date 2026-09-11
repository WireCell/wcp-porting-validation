# Your task: verdict-blind hand scan of assigned PDHD items (doc pdhd/18)

You are doing a physics hand scan of reconstructed ProtoDUNE horizontal-drift
(PDHD) cosmic-ray events. Work carefully and independently.

## Read this first — in full, before your first item

`/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/docs/scan/pdhd_stm_michel_scan_rubric.md`

It defines:
* the verdict alphabet and the **precedence order**, rules 1–6: topology
  before a flat profile; a flat profile means `THRU` only where the track could
  have left, since a mid-volume end whose charge stops in every plane is a stop
  (`FLAT_STOP:`); a busy stop is `MESSY`;
* how to read the Bragg profile;
* the two ways the fit gets the stopping point wrong;
* the PDHD detector traps (cathode, APA seam, the APA0 and APA2 charge deficits);
* the per-object tag rules, and how `michel_kind` is derived.

Follow it exactly.

## Hard prohibitions — absolute

1. **Read only**: the rubric, this file, your item list, and
   `<SHOTS>/<event>_<cluster>/` for **your own** items.
   * Never open any file with `key` or `labels` in its name.
   * Never open any prep payload (`smprep-*.json`), ROOT file, calib json or
     zip, or anything under `pdhd/work/` or `pdvd/docs/scan/`.
   * Never open anything under `/home/xqian/tmp/h18/` except `shots/` and your
     own out dir.

   These hold the chain's verdict or other people's answers. The scan is
   verdict-blind, and reading one destroys that blindness for the whole round,
   with no way to recover it.
2. **Never run `scan_harness.py`**, and never write anywhere except your own
   out dir, through `mkv.py`.
3. **Do not list or read any other scanner's out dir** (`v_parts/*`). Other
   scanners are working this round in parallel, and their verdicts are exactly
   the anchor that stops an independent scan being independent. Keep your own
   list of what you have done; do not `ls` the tree.

## Paths

* shots dir `<SHOTS>` = `/home/xqian/tmp/h18/shots`
* `<MKV>` = `/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan/campaign/mkv.py`
* `<ITEMS>` (your item list, one `<event>/<cluster>` per line) and `<OUT>`
  (your private out dir) are given in the message that assigns your work.

An item's frames live at `<SHOTS>/<event>_<cluster>/`: the `/` in the key
becomes `_`, so `029107_18/146` → `<SHOTS>/029107_18_146/`.

## Per item, in this order

1. `cat` the item's `context.json`:
   * containment: `ends.stop.d_face`, the per-axis `face` distances and
     `ends.seams_at_stop` (cathode, seam_z, anode_face);
   * which APA the stop is in, from `ends.stop.xyz`;
   * the object rows: `key`, `group`, `size`, `dqdx`, `d_min`, `d_max`,
     `cos_fwd`, `chain`.
2. **Read these seven PNGs** with the Read tool: `a_proj_full`, `b_3d_wide`,
   `c_3d_stop`, `d_3d_stop`, `e_3d_stop`, `f_meas`, `h_dqdx_zoom`.
   * `h_dqdx_zoom.png` is the Bragg evidence, already upscaled 3× for you.
     **Do not write crop or zoom helpers**; they cost time and add nothing.
   * The three `*_3d_stop` azimuths give the transverse offset that decides
     `michel` vs `delta / other`.
   * `f_meas` tells a real collapse from a dead region or a coverage hole.
3. Decide, in the rubric's precedence order:
   * the verdict and confidence;
   * a tag for **every** object row;
   * the pin, if the fit overshoots.

   `michel_kind` follows mechanically from your tags.
4. **Write the record immediately with `<MKV>`**, before the next item. If
   `mkv.py` refuses, read its message, fix the record and re-run it.

If you run low on context, **stop and report what you completed**. Everything
written is banked and the rest is reassigned automatically. An unreported,
half-finished chunk is the only way work gets lost here.

## Judgement

Judge each item on its own evidence. Do not aim for any particular mix of
verdicts, and do not let one item's call influence the next. Mark confidence
honestly: `medium` and `low` rows are the ones the owner will look at, and a
genuinely uncertain call marked `high` is worse than the uncertainty. Say in
the evidence paragraph what made a call hard.

## Report back (your final message)

* one line per item: `key — VERDICT / michel_kind / confidence [pin_rr] [notes prefix]`
* the keys where you saw an **overshoot** (moved the pin) and an **undershoot**
* the items you found hardest, and the competing reading for each
* anything in the rubric that was ambiguous, or that you had to stretch. Be
  concrete: a specific objection with an item key is worth far more than "it
  went fine". Do not suggest the rubric be changed mid-round; just report.
