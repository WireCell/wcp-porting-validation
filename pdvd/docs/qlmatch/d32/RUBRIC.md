# doc qlmatch/32 blind Q/L scan — scanner rubric (round 1)

You are judging, for one charge cluster of a PDVD (ProtoDUNE vertical drift) event, **which candidate light flash
it produced**, i.e. which flash gives its T0.

## Each sheet
- **Rows.** One row per candidate flash, lettered A, B, ... in flash-time order.
- **In each row:**
  - **X-Z and X-Y projections** of the whole event (grey), with this cluster in **red** and any co-clusters of the
    hypothesis in **orange**. Every cluster is drift-corrected to this candidate's T0; the red boxes are the two
    drift volumes.
  - **Bars vs line.** Measured PE per channel (grey bars) against the PE this cluster is predicted to produce (blue
    line), on the cluster's side. A red **R** above a channel means that channel's waveform hit the ADC rail. Its
    measured PE is a *reconstruction* of a clipped pulse, not a direct measurement.
  - **Light maps.** Measured and predicted 2-D maps of the same side.
  - **Numbers.**
    - measured / predicted PE on the unrailed channels and on all channels;
    - the cluster's x-extent at this T0;
    - its distance outside the drift box, and its distance to the anode and to the cathode.

## How to judge (identical for every sheet)
1. **Geometry at the T0 comes first.**
   - At the right T0, the cluster lies inside its drift volume. "Outside drift box" should be about 0; a few cm is
     tolerable, but ≳ 10 cm is implausible.
   - A track that enters or exits through the anode or cathode plane should touch that plane at the right T0.
   - A cluster that ends exactly on the cathode or anode at one candidate, and floats inside at another, is evidence
     for the one where it touches, if its topology says it crosses (e.g. a long straight through-going track).
2. **Light pattern on the unrailed channels.**
   - The measured bars should be high where the blue line is high.
   - The measured amplitude should be at least comparable to the predicted amplitude.
   - Measured may exceed predicted: other clusters also make light in the same flash.
   - Predicted far above measured on the unrailed channels (pred/meas unrailed ≳ 3–5) disfavours the candidate.
   - Predicted far below measured is weak evidence against: the flash may belong mainly to other clusters.
   - The absolute PE scale is uncertain by about ×3. Judge shape first, then gross amplitude.
3. **Railed (R) channels are a sanity check only.** Do not accept or reject a candidate on the height of an R bar.
   An R channel should simply be bright where the prediction is bright.
4. **Cathode channels 4–11 see both volumes.** A cluster near the cathode is weakly constrained in T0, so lean on
   the geometry.
5. **Choosing between candidates.**
   - Several candidates can look acceptable. Pick the one where geometry and the unrailed pattern agree best.
   - Flash time itself is not evidence.
   - If none is plausible, answer `none`.
   - If you cannot decide, answer `unsure`. Do not guess: an honest `unsure` is better than a coin flip.
6. **Confidence.**
   - `high`: clear.
   - `med`: probable.
   - `low`: a lean only. It is counted like `unsure`.

## Recording (the only allowed write)
```
python3 /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/qlmatch/scripts/d32_record.py \
    --wave <your wave number> --id <sheet id> --pick <letter|none|unsure> --conf <high|med|low> \
    --reason "<one sentence citing only what the sheet shows>"
```
Record one verdict per sheet listed in your wave's INDEX.md. The recorder refuses unknown ids, letters not on the
sheet, and a second verdict for the same id.

## Blindness rules (violations void your whole wave)
- Read only these:
  - your wave directory (`INDEX.md` and the sheet PNGs);
  - this RUBRIC;
  - the recorder script.
- Do **not** read or list any other file or directory. That includes:
  - `/home/xqian/tmp/p32/scan_key/` and other waves;
  - anything under `pdvd/work/`, `ql_labels`, `ql_scores` or `decisions*`;
  - calib json files and docs.
- Do not try to work out which light reconstruction or which matching arm a sheet comes from.
