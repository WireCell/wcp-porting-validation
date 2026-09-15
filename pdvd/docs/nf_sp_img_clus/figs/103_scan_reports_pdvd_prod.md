# doc pdvd/103 sec 10.5: PDVD production-lineage blind scan (smx11), the 11 scanners' reports (summaries of their hand-backs; round /home/xqian/tmp/d103/round_pdvd2)

# PDVD round_pdvd2 w1_a0
The agent ended on an API 500 before its hand-back, but had already written all 22 of 22 records (none missing vs its wave). Audit: 202 tool calls, 0 flagged. No report text exists; the records carry the evidence and notes.
Tally from records: STM_MICHEL 7, STM_ONLY 3, THRU 7, FRAG_THRU 4, UNCLEAR 1.

Audit: 202 tool calls; flagged 0

# PDVD round_pdvd2 w1_a1 scanner report (summary of its hand-back)
22 / 22 written.

**Tally**
- STM_MICHEL 6; STM_ONLY 3.
- THRU 7.
- MESSY 3; UNCLEAR 2.

**Pins moved (overshoot)**
- 039253_6/102: rr 9.5, on a segment boundary.
- 039349_65/41: rr 11, collapse inside row 41012.
- 039349_71/67: rr 9, collapse inside row 67011.
- No undershoot.

**Hardest items (competing reading)**
- 039349_65/41: STM_MICHEL vs MESSY / dead-band loss; "both" rests on specks 36-43 cm out.
- 039349_71/67: STM_MICHEL vs STM_ONLY (charge loss by a V dead channel).
- 039252_10/95: STM_ONLY DEAD vs an overshoot Michel tail.
- 039253_1/93: STM_ONLY vs a THRU seam crossing.
- 039349_4/76: THRU vs UNCLEAR (grey reaches slice 0, the start of the window).
- MESSY items 039252_16/30, 039253_2/117, 039349_16/50.
- 039349_36/63: STM_MICHEL on a 0.4 cm speck 3.5 cm out, vs STM_ONLY.

**Rubric points it named**
1. **Upper-end THRU with nothing continuing, on direction alone:** 039349_79/58, 039253_12/98, 039252_14/39, 039252_9/97. They move together if the owner disagrees.
2. **Same-slice grey on isochronous tracks** is weak continuation evidence (039252_16/117, 039253_8/59).
3. **Tests that cannot be run:** a y=0 CRU seam continuation outside the f_meas window (039253_1/93); the readout-window edge at slice 0 (039349_4/76).
4. **C rows with no dqdx** treated as degenerate, which blocks a capture-gamma call (039252_10/95), the same as round 1's gap 2.
5. **Split-row overshoot hides the Michel start in the tags** (039349_65/41, 039349_71/67).

Audit: 184 tool calls; flagged 0

# PDVD round_pdvd2 w1_a2 scanner report (summary of its hand-back)
22 / 22 written. 039349_23/55 was rewritten once (it corrected which points are hot); the verdict stays THRU.

**Tally**
- STM_MICHEL 2; STM_ONLY 7.
- THRU 12.
- UNCLEAR 1.

**Pins:** none moved; no undershoot.

**Hardest items (competing reading)**
- 039253_14/121: UNCLEAR vs MESSY / STM_MICHEL. Imaging failure, branching end near a seam.
- 039349_23/55: THRU vs STM_MICHEL, with an overshoot pin at 5.15 and the Michel lost past the frame edge.
- 039349_5/43: THRU vs STM_MICHEL (an 11 cm back-running arm, rule 1).
- 039349_55/75: STM_ONLY at the upper end vs THRU (upper part not imaged).
- 039349_28/50: THRU vs STM_MICHEL (Michel-typed loop at the y wall).
- 039349_76/77: THRU vs STM_ONLY FLAT_STOP (about 10 slices short of the frame edge).
- 039253_12/39: THRU vs UNCLEAR (along the y=0 seam, charge lost).
- 039349_29/40: THRU vs STM_MICHEL (a side piece 6 cm back).

**Rubric points it named**
1. **End of the recorded frame** (slice ~1585-1597: 039349_38/25, 039349_76/72, 039349_23/55, 039349_76/77), treated as "runs into a dead region".
   - It alone turns 039349_76/72 from FLAT_STOP into THRU.
   - Upper ends at slice ~20 (039349_71/64, 039349_29/40).
   - No prefix exists. **The owner should rule.**
2. **Channels jumping across CRU seams** in f_meas, so the three-plane grey test is incomplete (039253_12/39, 039349_14/35, 039349_24/62).
3. **Confidence cap at a flat upper end** is not stated.
4. **Isochronous smear vs topology arm** (039253_14/121, 039349_28/50).
5. **Two-point specks tagged gamma** decide detached dots / both.

Audit: 204 tool calls; flagged 0

# PDVD round_pdvd2 w1_a3 scanner report (summary of its hand-back)
22 / 22 written. It rewrote 039349_69/20 (high → medium, + NO_MEASUREMENT:) and 039349_45/17 (THRU → FRAG_THRU).

**Tally (its own list)**
- STM_MICHEL 6; STM_ONLY 3.
- THRU 9; FRAG_THRU 1.
- UNCLEAR 3.

**Pins moved (overshoot)**
- 039253_12/77: rr 5.0.
- 039252_8/43: rr 5.5, collapse inside single row 43021, so STM_ONLY is forced.
- 039253_0/43: rr 72, the fit bridges a 60 cm gap.
- No undershoot.

**Hardest items (competing reading)**
- 039252_4/107: UNCLEAR vs STM_ONLY; a grey track crosses the end piece.
- 039349_62/57: UNCLEAR; an 11 cm piece at the top anode by the z seam.
- 039253_0/43: UNCLEAR vs FLAT_STOP; the re-pinned end is outside the f_meas window.
- 039349_1/62: STM_MICHEL vs a seam-gap crossing.
- 039252_17/40: a piece at 9.8 cm; tagged michel instead of gamma it would be STM_MICHEL.
- 039349_38/68: a straight grey crossing track judged a separate object.

**Rubric points it named**
1. **Time-window edge** (run 039349 slice ~1585-1595) decides 039349_36/50, 039349_69/20 and 039349_60/61, called THRU under DEAD:.
   - Check on the same run: 039349_56/48 ends 65 slices clear of the edge, called FLAT_STOP.
   - The alternative, that the display pads past the data, would make all three FLAT_STOP.
2. **Split-row overshoot vs mkv.py** (039252_8/43).
3. **FRAG candidates left as THRU / STM_ONLY:** 039349_57/64, 039252_17/40.
4. **A rule-7 upper end cut by the window** (039349_69/20).
5. **A gamma-geometry speck on a crossing grey track** tagged other (039349_48/55).
6. **Seam distance "a few cm"** is undefined (7 / 10.5 cm).

Audit: 203 tool calls; flagged 0

# PDVD round_pdvd2 w1_a4 scanner report (summary of its hand-back)
22 / 22 written.

**Tally**
- STM_MICHEL 3; STM_ONLY 9.
- THRU 7; FRAG_THRU 3.

**Pins:** 039252_8/120 moved to rr 6.5 (overshoot collapse inside a single muon row). No undershoot.

**Hardest items (competing reading)**
- 039252_8/120: STM_ONLY forced by mkv.py. The collapse is inside one row, so no michel tag is possible. Competing: STM_MICHEL, or seam charge loss at the z=149.65 seam 0.8 cm away.
- 039253_13/74: specks at 10.2-10.4 cm (gamma) decide STM_ONLY vs STM_MICHEL.
- 039349_50/44, 039253_8/58: isochronous tracks where the chain's Michel pieces were overruled by grey continuation.
- 039349_16/54: FRAG_THRU vs MESSY.
- 039252_17/87: THRU vs UNCLEAR along the z seam.
- 039252_7/71: THRU vs STM_MICHEL (DIRECTION, a hook past a seam dip).
- 039349_21/54: THRU vs STM_ONLY (DIRECTION, ragged scatter).
- 039349_38/50: STM_ONLY vs THRU (short rise near the y seam).
- FLAT_STOP items 039349_64/19, 039349_47/63, 039349_11/53: vs UNCLEAR / NO_MEASUREMENT (dead W / V bands).

**Rubric points it named**
1. **Readout-window edge inferred** (run 039349: 039349_5/25, 039349_79/36, 039349_68/24, 039349_52/59, 039349_38/50). 039349_52/59 flips to FLAT_STOP if the inference is rejected.
2. **Seam charge loss** along a seam or at a crossing before the end.
3. **The split-row rule vs mkv.py** (039252_8/120), the same as round 1's rubric-gap 4.
4. **A vertex arm on an isochronous track** (039349_50/44).
5. **A C-row with no charge value** counts as degenerate (039252_8/120 C476), the same as round 1's gap 2.

Audit: 203 tool calls; flagged 0

# PDVD round_pdvd2 w1_a5 scanner report (summary of its hand-back)
22 / 22 written.

**Tally**
- STM_MICHEL 5; STM_ONLY 7.
- THRU 7.
- MESSY 1; UNCLEAR 2.

**Pins moved (overshoot)**
- 039252_11/31: rr 4.7.
- 039253_9/36: rr 8.5.
- 039252_6/97: rr 3.7.

**Undershoot:** 039252_14/38. A 17 cm kinked piece past the fit end at 1.6-3x plateau, tagged muon.

**Hardest items (competing reading)**
- 039252_14/38: UNCLEAR vs STM_ONLY upward / THRU cathode crossing / STM_MICHEL.
- 039252_16/89: UNCLEAR (isochronous) vs THRU / STM_MICHEL.
- 039252_11/31, 039253_9/36: OVERSHOOT STM_MICHEL vs STM_ONLY.
- 039252_16/80: MESSY vs SEAM THRU.
- 039349_28/19: THRU at an upper end near the cathode vs DIRECTION STM_ONLY.
- 039349_23/52: SEAM THRU vs a flat stop at the stub's far end.

**Rubric points it named**
1. **Main gap: a fit end just inside the edge of the imaged slices** (run 039349 slice ~1595: 039349_49/73, 039349_47/51, 039349_41/65, 039349_52/31). **Resolved the OPPOSITE way to w1_a2 / a3 / a4 / a9 / a10.**
   - It first wrote them THRU / DEAD:, then reverted to FLAT_STOP STM_ONLY (medium).
   - Its reason: rule 3 holds because the charge ends inside the imaged range.
   - **The owner ruling is needed; the scanners are inconsistent on this class.**
2. **One dead plane at the end:** it read the two live planes as satisfying rule 3 (039349_51/14, 039349_60/17).
3. **A delayed-Michel-like grey with no object row** cannot be tagged (039349_55/74).
4. **Far C-rows with no dqdx** tagged delta / other; gamma would change kind (039349_16/42), the same as round 1's gap 2.
5. **Hot-tip rule for longer pieces** (039252_14/38).

Audit: 212 tool calls; flagged 0

# PDVD round_pdvd2 w1_a6 scanner report (summary of its hand-back)
22 / 22 written. One correction: 039253_6/122 rewritten from high to medium (isochronous cap).

**Tally**
- STM_MICHEL 6; STM_ONLY 3.
- THRU 8.
- MESSY 2; UNCLEAR 3.

**Pins moved (overshoot)**
- 039349_60/59: rr 6.2.
- 039349_68/63: rr 10.0.
- No undershoot.

**Hardest items (competing reading)**
- 039252_8/105: UNCLEAR, flat stop vs z-seam crossing.
- 039253_0/88: MESSY vs THRU CONTINUES.
- 039349_43/54: UNCLEAR vs STM_MICHEL flat stop / MESSY.
- 039349_25/49: UNCLEAR, 74 points.
- 039349_18/53: MESSY vs UNCLEAR.
- 039349_83/81: FLAT_STOP vs THRU (charge fades).
- 039349_72/18: THRU vs STM_ONLY on the direction rule.
- 039253_13/69: THRU, with arms at a horizontal track's end read as artefacts.
- 039253_16/116: STM_MICHEL vs a corner exit.
- Kind choice: both vs attached turns on one far speck (039252_3/89, 039253_16/116).

**Rubric points it named**
1. **The readout window ends the track** (run 039349 slice ~1590-1600: 039349_57/21, 039349_56/20, 039349_30/28). No rule; recorded as DIRECTION / CONTINUES.
2. **Isochronous confidence cap** when the call rests on rule 1 or 2.
3. **A seam near a clear rise.**
4. **A horizontal grey line with a horizontal track:** continuation vs another object.
5. **A Michel arm attached a few cm behind the stop.**
6. **A faint branch on a crossing isochronous track:** at the vertex, but tagged other.

Audit: 203 tool calls; flagged 1
> ADJUDICATED (2026-09-15): the one flag is a false positive. The flagged command is an mkv.py write followed by
> `grep -l '"medium"' .../v_parts/w1_a6/*039253_6*122* || grep -rl 'medium' .../v_parts/w1_a6 | grep -i 122`.
> Both greps are confined to the scanner's OWN out dir (allowed by AGENT_TASK.md). The first grep succeeded (its
> output is the file path), so the recursive fallback did not run. No other scanner's dir, and no chain answer,
> was read. w1_a6 passes.

# PDVD round_pdvd2 w1_a7 scanner report (summary of its hand-back)
22 / 22 written.

**Tally:** STM_MICHEL 5, STM_ONLY 6, THRU 7, UNCLEAR 4.

**Pins:** 039252_8/51 moved to rr 4.8 (arc length; the table's straight line is 4.17). No undershoot.

**Hardest items (competing reading)**
- 039349_38/63, 039349_34/54: UNCLEAR. A grey continuation runs straight on, then ends mid-volume. Competing: FRAG_STM_ONLY, or STM_MICHEL on a grey-only Michel, which mkv.py cannot record.
- 039252_3/103: UNCLEAR (near-isochronous) vs THRU CONTINUES / STM_MICHEL.
- 039253_16/104, 039349_72/47: THRU on a collapse at the end of the recorded data vs STM_MICHEL OVERSHOOT.
- 039252_8/51: stopper vs a z-seam crossing 2.9 cm away.
- 039349_69/64: STM_ONLY vs STM_MICHEL (unclustered sideways charge, no row).
- 039349_56/46: FLAT_STOP vs overshoot (equal humps).

**Rubric points it named**
1. **Readout frame edge** (039253_16/104 at slice ~2500, 039349_72/47 at ~1595), filed under DEAD:. Negative control: 039252_14/80. It asks that the rubric name the case and give it a prefix.
2. **A Michel that exists only in grey or unclustered charge** cannot be recorded (039349_38/63, 039349_34/54, 039349_69/64).
3. **A grey continuation that itself ends mid-volume** has no rule.
4. **A crossing into another CRU's channel range** leaves no grey in the f_meas window (039349_33/54, 039252_5/79).
5. **The near-isochronous threshold is not numeric** (039349_0/58 at 6 deg).

Audit: 205 tool calls; flagged 0

# PDVD round_pdvd2 w1_a8 scanner report (summary of its hand-back)
22 / 22 written. 039253_17/107 was rewritten (FLAT_STOP → OVERSHOOT, pin 8.5). 039252_6/96's first write failed on an apostrophe and was re-run.

**Tally**
- STM_MICHEL 6; STM_ONLY 5.
- THRU 9.
- MESSY 1; UNCLEAR 1.

**Pins moved (overshoot)**
- 039253_17/107: rr 8.5 (inside row 107001, so STM_ONLY).
- 039349_66/85: rr 26 (charge-empty bridge).
- No undershoot.

**Hardest items (competing reading)**
- 039253_14/101: STM_MICHEL low vs STM_ONLY (charge loss along dead W channel).
- 039349_53/44: STM_MICHEL at the upper end vs THRU DIRECTION.
- 039349_66/85: which piece is the Michel after a 26 cm pin move.
- 039253_17/107: overshoot vs dead-U charge loss.
- 039349_3/50: gradual decline vs overshoot.
- 039253_13/32: a 2.9 cm piece as Michel vs the muon's last cm.
- 039252_15/77: isochronous, grey in U/V but not W.

**Rubric points it named**
1. **Seam ends:** a continuation reads out on another CRU outside the f_meas window, so "no grey" is not evidence (039349_40/62, 039252_6/96, 039349_27/21, 039252_8/102).
2. **One segment holding both muon and Michel** (039253_17/107): the alphabet cannot split it; same as round 1's gap 4.
3. **Distances on overshoot items** are relative to the fit end, not the moved pin (039349_66/85).
4. **Bridge rules disagree:** "bridged to the stop = michel" vs the overshoot clause (039349_66/85).
5. **Upper end with no topology and no profile** (039349_57/66).
6. **Dead channel on an induction vs the collection plane.**
7. **Cold straight continuation** has no rule (039253_14/101).
8. **Gradual decline vs overshoot** (039349_3/50).
9. **Isochronous grey in some planes only.**
10. **Grey specks with no object row** (039349_50/70).
11. **"Near" a seam or wall** is undefined (039349_28/13).
12. **An equal hump earlier on the profile** (039349_81/52).
13. **Busy f_meas with a clean 3-D view** (039252_5/78).

Audit: 205 tool calls; flagged 0

# PDVD round_pdvd2 w1_a9 scanner report (summary of its hand-back)
22 / 22 written.

**Tally**
- STM_MICHEL 1; STM_ONLY 5.
- THRU 8; FRAG_THRU 3.
- MESSY 3; UNCLEAR 2.

**Pins:** 039253_17/21 moved to rr 4.5. No undershoot. The bridged fit ends on 039349_42/49 and 039253_2/86 were left unpinned (not stops).

**Hardest items (competing reading)**
- 039253_17/21: STM_MICHEL (0.8 cm vertex piece) vs STM_ONLY FLAT_STOP.
- 039349_21/56: MESSY vs STM_ONLY.
- 039252_9/77: MESSY vs THRU / STM_MICHEL.
- 039349_44/60: MESSY vs STM_MICHEL; the chain's arm lies on a crossing grey track.
- 039253_6/114: UNCLEAR vs STM_MICHEL / THRU / MESSY.
- 039349_77/20: a 2.4 cm straight-on stub tagged muon gives STM_ONLY; tagged michel it gives STM_MICHEL.

**Rubric points it named**
1. **Readout-frame edge** (slice ~1590: 039349_11/27, 039349_77/34, 039349_73/70, 039349_3/68). It called the three lower ends THRU under DEAD:. Read literally, rule 3 would make them FLAT_STOP. **Needs an owner ruling.**
2. **Bridged fit ends on non-stops:** muon vs delta for the bridge.
3. **Straight-on stub at or below plateau at a rule-3 stop:** rule 1 (weak) vs the undershoot section (real Michel), the same as round 1's gap 5.
4. **MESSY vs UNCLEAR** when the crossing track shows only in f_meas.
5. **FRAG_THRU whose grey continuation ends inside the window** (039253_4/44).
6. **Rule 7 cut edge:** 039252_9/77, dx/chord −0.297.
7. **Missing coloured cells in f_meas** (039349_14/49, 039253_0/90).
8. **RUBRIC.md hard rule 1 still names /home/xqian/tmp/p99scan/.** It followed AGENT_TASK.md.

Audit: 203 tool calls; flagged 0

# PDVD round_pdvd2 w1_a10 scanner report (summary of its hand-back)
19 / 19 written. After its own advisor review it rewrote 039349_74/22, 039349_53/28, 039349_1/72 and 039349_44/59.

**Tally**
- STM_MICHEL 2; STM_ONLY 6; FRAG_STM_ONLY 1.
- THRU 7; FRAG_THRU 1.
- MESSY 1; UNCLEAR 1.

**Pins:** none moved; no undershoot.

**Hardest items (competing reading)**
- 039349_72/43: STM_ONLY FLAT_STOP vs UNCLEAR / NO_MEASUREMENT. The track runs along a dead W line.
- 039349_48/16: THRU through the corner vs STM_MICHEL, with forked 2-5 cm stubs as the electron.
- 039349_53/28: THRU (rule 7 plus the end of the window) vs STM_ONLY on a 1.8x rise at the upper end.
- 039349_44/59: FLAT_STOP vs THRU, if the window closes at the fit end.
- 039253_17/98: UNCLEAR vs MESSY / THRU. Near-isochronous, 90-degree kink.

**Rubric points it named**
1. **The readout window ends at the fit end** (slice ~1575-1597 on run 039349; 039349_53/28, 039349_44/59, 039349_1/72).
   - No clause covers it; it treated the edge as a caveat, not a rule-3 disqualifier.
   - This is the same gap as round 1's rubric-gap 1, now on the production 6400-tick window.
2. **Side-wall distance** has no number (anode ≲ 2 cm only). It treated 7-14 cm as not at a face.
3. **A seam running along the track** rather than crossed (039349_1/72).
4. **Upstream same-track pieces not in the chain:** muon vs delta (039349_50/16, 039252_14/67).

Audit: 178 tool calls; flagged 0

