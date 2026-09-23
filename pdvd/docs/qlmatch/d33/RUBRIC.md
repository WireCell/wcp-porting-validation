# doc qlmatch/33 blind Q/L scan: scanner rubric (round 2)

You are judging, for one charge cluster of a PDVD (ProtoDUNE vertical drift) event, **which candidate light flash
it produced**, i.e. which flash gives its T0. The owner's experience is that **geometry plus the light pattern** decide
this. The absolute light scale and the railed channels do not.

## Each sheet
One row per candidate flash. The letters A, B, ... are in **random order**: they do not follow flash time. In each row:
- **X-Z and X-Y projections** of the whole event (grey), shifted to this candidate's T0.
  - This cluster is **red**; orange points are the cluster's own associated pieces.
  - The red boxes are the two drift volumes.
- **ZOOM**: this cluster alone at this T0, with the anode (green dashed) and cathode (purple dashed) planes when they
  are within reach.
- **Channel bars**, this side of the detector:
  - grey bars are the measured PE on **unrailed** channels;
  - **hatched red bars marked R are railed channels**: the channel saturated, so it is *bright*, but its value is not
    shown;
  - the **cyan band** is the light predicted for the *other* clusters that are securely matched to this same flash;
  - the **solid blue line** is cyan + **this** cluster, and the dotted line is this cluster alone.
- **Maps**: measured (railed channels drawn at the unrailed maximum) and predicted (co-matched + this cluster).
- **Numbers**:
  - unrailed meas vs pred (this cluster, co-matched, and (co+this)/meas);
  - the cluster's x-extent, its distance outside the drift box, and its distances to the anode and cathode.

## How to judge (identical for every sheet)
1. **Geometry at the T0 first.**
   - At the right T0 the cluster lies inside its drift volume. "Outside drift box" should be about 0; a few cm is
     tolerable, but ≳ 10 cm is implausible.
   - A track that visibly crosses the anode or cathode plane, or ends on it (long straight through-going tracks,
     tracks leaving the volume), should touch that plane at the right T0 (use the ZOOM).
   - A contained blob or short track floats inside at any in-box T0, so geometry only excludes candidates.
2. **Light pattern: does this flash have room for this cluster, in the right place?**
   - A flash is usually made by several clusters together. Compare the **solid blue line** (co-matched + this) with
     the grey bars on the **unrailed** channels.
   - At the right candidate the peaks of the solid line sit on the bright unrailed channels, and adding this cluster
     does not push the line far above the data.
   - `(co+this)/meas` near 0.3–3 on unrailed channels is healthy; the absolute scale is uncertain by ×3.
   - Well above ~3–5 (the prediction greatly exceeds the data) disfavours the candidate.
   - Well below 1 is only weak evidence against: other, unmatched clusters may also be lit in that flash.
   - For short or tiny clusters, the **pattern** is the evidence. Ask whether the channels this cluster predicts to be
     brightest are lit in the data.
3. **Railed (R) channels only say "bright".** A railed channel where the prediction is large is consistent. A railed
   channel where nothing is predicted means other light is present. Never accept or reject on an R bar's height:
   none is shown.
4. **Cathode channels (4–11) see both volumes.** Near-cathode clusters are weakly constrained in T0; lean on geometry.
5. **Nearby candidates.** Two candidates less than ~10 µs apart place the cluster within ~1.5 cm of each other in x.
   Geometry cannot separate them, so decide on the light pattern. If the patterns are also equivalent, answer
   **`tie:X,Y`**. A tie is an answer ("these are equally good, and better than the rest"), not a dodge. Use it only
   when the listed candidates really cannot be separated.
6. **Answers.**
   - a single letter: the candidate where geometry and the unrailed pattern agree best;
   - `tie:X,Y[,Z]`: equally good candidates, all clearly better than the others;
   - `none`: no candidate is plausible (every one is geometrically impossible or clearly wrong in pattern);
   - `unsure`: you cannot decide. An honest `unsure` beats a coin flip.
7. **Confidence.**
   - `high`: clear.
   - `med`: probable; a reasonable scanner would agree.
   - `low`: a lean only, counted like `unsure`.
   - Commit (`med`) when geometry excludes all but one candidate, or when the pattern clearly singles one out. Don't
     downgrade just because the cluster is small.

## Recording (the only allowed write)
```
python3 /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/qlmatch/scripts/d33_record.py \
    --wave <your wave number> --id <sheet id> --pick <letter|tie:X,Y|none|unsure> --conf <high|med|low> \
    --reason "<one sentence citing only what the sheet shows>"
```
- Record one verdict per sheet listed in your wave's INDEX.md.
- The recorder refuses unknown ids, letters not on the sheet, and a second verdict for the same id.

## Blindness rules (violations void your whole wave)
- Read only these files:
  - your wave directory (`INDEX.md` and the sheet PNGs);
  - this RUBRIC;
  - the recorder script.
- Do **not** read, list or search any other file or directory. That includes:
  - `/home/xqian/tmp/p33/scan_key_r2d/` and other waves;
  - anything under `pdvd/work/`, `ql_labels`, `ql_scores` or `decisions*`;
  - calib json files, docs and scripts other than the recorder.
- Do not try to work out which reconstruction or matching arm a sheet relates to.
