# doc pdvd/103: summaries of the blind scanners' reports (pdhd) -- hardest items and rubric gaps, for the owner (verbatim reports are in the session; these are condensed)

# PDHD w1_a0 scanner report (summary of its hand-back)
21 / 21 written. Tally: 9 THRU, 1 FRAG_THRU, 6 STM_ONLY, 3 STM_MICHEL, 2 UNCLEAR; 11 high, 10 medium.

**Pins and undershoots**
- Pin moved: 029107_15/118 (rr 10.0, cathode 2.1 cm).
- Undershoot: 028084_26/41.
- 028084_17/34: the chain's 0.4 cm "Michel" speck was retagged muon (it is an anode exit).

**Hardest items**
- 029107_15/118: STM_MICHEL + pin vs THRU (a cathode crossing).
- 029107_19/29: THRU (grey continues in UVW) vs STM_MICHEL with a pin at 4.7 cm.
- 029107_4/124: UNCLEAR (W is dead) vs STM_MICHEL at a Y-fork.
- 028084_9/42: STM_ONLY on a 3 cm climb at the upper end vs THRU DIRECTION.
- 028084_24/67: FLAT_STOP vs an overshoot vs THRU.
- 028084_28/41: FLAT_STOP vs a readout-edge loss.

**Rubric points it named**
1. Rule 1 (an angled piece) vs a grey continuation at a lower end. (029107_19/29)
2. How big a "clear rise" must be. (028084_9/42)
3. A pin and a tag contradict on a split row. (029107_15/118)
4. Frame-edge loss is not a listed trap. (028084_28/41)
5. Isochronous: is a terminal rise alone enough? Split three ways. (028084_26/94, 029107_15/133, 028084_4/53)
6. Every unfitted C-row counts as degenerate (none has a dQ/dx), which forces delta / other. (C191 at 52 cm could be a capture gamma)
7. Notes prefix order: count tokens, not position 0.

# PDHD w1_a1 scanner report (summary of its hand-back)
21 / 21 written through mkv.py. Tally: 6 STM_MICHEL, 6 STM_ONLY, 6 THRU, 1 FRAG_THRU, 2 UNCLEAR; 10 high, 9 medium, 2 low.

**Pins and undershoots**
- Pins moved: 028084_3/90 (rr 4.6), 028084_2/116 (rr 7.4, near-isochronous).
- Undershoot: 028084_20/28 (clean, 1.2 cm tip at 1.96e5).
- Borderline undershoot, no prefix: 028084_12/33.

**Hardest items**
- 028084_18/85: THRU at the anode face. The competing reading is STM_MICHEL on a 7.3 cm, 40° faint piece lying in the anode plane.
- 028084_12/33: STM_ONLY FLAT_STOP vs STM_MICHEL (straight-on stub).
- 029107_18/30: UNCLEAR, on the APA seam with the U end in a dead band.
- 029107_3/22: STM_MICHEL vs STM_ONLY; a faint arm in a ghost sheet.
- 028084_2/116: near-isochronous; STM_MICHEL vs STM_ONLY vs THRU.
- 028084_30/61: STM_MICHEL vs MESSY.
- 028084_22/3: STM_MICHEL vs STM_ONLY; two low-charge straight-on stubs.

**Rubric ambiguities it named**
1. At the anode, a faint angled piece lying in the anode plane: exit smear or topology? (028084_18/85)
2. Detached pieces at 9.6–12.3 cm, michel vs gamma. The scanner called gamma, giving STM_ONLY / detached dots; the chain calls them Michel. (029107_11/86, 029107_24/33)
3. Undershoot "several times plateau" vs rule 1's weak-stub-plus-rule-3. (028084_12/33)
4. Ragged rise vs scatter at a face. (028084_25/113)
5. A michel tag at d_min 10.9 cm, continuing an attached arm. (028084_30/61)
6. A thin horizontal line in all three f_meas columns on wrapped planes, probably the fit path. (029107_3/22, 028084_18/85, 029107_24/33, 028084_1/60)
7. Compact rows with no dQ/dx at 50–59 cm forced to delta / other. (029107_12/129, 028084_23/51)

# PDHD w1_a2 scanner report (summary of its hand-back)
21 / 21 written. Three were rewritten only to add NO_MEASUREMENT: (029107_6/45, 028084_12/37, 029107_24/21).
Tally: STM_MICHEL 8, STM_ONLY 4, THRU 8, MESSY 1.

**Pins and undershoots**
- No pins moved, no undershoots.
- An overshoot was considered and rejected on 029107_22/108.

**Hardest items**
- 029107_29/31: STM_MICHEL on an angled arm vs THRU (CONTINUES, since grey runs on in V).
- 028084_2/40: STM_MICHEL on a low-charge straight-on stub vs STM_ONLY.
- 029107_24/21: THRU vs a Michel at the upper end.
- 029107_3/23: THRU vs a stop at the upper end.
- 029107_23/41: STM_ONLY at medium, because the end is at the cathode.
- 029107_12/29: MESSY vs STM_MICHEL.
- 029107_14/89: FLAT_STOP vs an unseen exit along a V dead band.

**Notes on its own records**
- 029107_27/38: REVERSED. The floor end rises to about 150k.
- 028084_28/96: its CONTINUES claim is weak (the grey line sits at another t0).
- 029107_27/99: really decided by rule 2.

**Rubric points it named**
1. The readout-window edge has no prefix; DEAD: was used. (029107_13/26, 029107_13/97)
2. A continuation past the cathode is invisible in f_meas. (029107_6/45, 029107_23/41)
3. A straight-on stub vs a forward Michel under a clear rise. (028084_2/40)
4. Does row 28 NO_MEASUREMENT: apply when another arm already decides?
5. Anode distance 5.1 cm, just over the ANODE rule. (028084_28/96)
6. The ~10 cm tie-break for a speck past a Michel arm's tip. (029107_24/111, 028084_0/36)

# PDHD w1_a3 scanner report (summary of its hand-back)
21 / 21 written. 029107_28/30 was rewritten once, retagging 30007 delta / other under the UNCLEAR rule.
Tally: 7 STM_MICHEL, 3 STM_ONLY, 7 THRU, 1 FRAG_THRU, 3 UNCLEAR.

**Pins and undershoots**
- Pins moved:
  - 029107_26/111 (rr 7.0, a split row);
  - 028084_2/20 (rr 7.0);
  - 028084_16/47 (rr 6.5, FACE).
- Undershoot: 029107_14/23.
- Ambiguous hot tip, not invoked: 029107_28/30 (isochronous).

**Hardest items**
- 029107_11/26: STM_MICHEL on a 3.2 cm piece, 4.9 cm from the APA seam, vs THRU.
- 028084_18/22: STM_ONLY low vs THRU/CONTINUES (V-only grey, crossings).
- 028084_16/47: STM_MICHEL + pin at the back face vs THRU through the face.
- 028084_9/52: THRU at the anode vs STM_ONLY on a ragged 1.5x lift.
- 028084_23/49: UNCLEAR vs FLAT_STOP vs MESSY.
- 029107_28/27: UNCLEAR vs FLAT_STOP vs THRU (isochronous).
- 029107_26/111: is a 6 cm tail with a hook a Michel?
- 028084_8/29: rule 7 THRU, medium (the start inside the volume is unexplained).

**Rubric points it named**
1. The seam/cathode prefix threshold ("a few cm"); ≤ 3 cm was used.
2. Rule 7 at dy/chord 0.28, just under 0.3. (029107_28/93)
3. Lower ends whose rule-3 conditions can't be checked: UNCLEAR for two, STM_ONLY low for one. NO_MEASUREMENT: was also put on two STM_ONLY records.
4. One-plane grey when the other planes are covered by crossings. (028084_18/22)
5. Straight-on stub conflict: rule 1 v2 vs the undershoot section. (028084_18/22, 029107_17/49)
6. cos_fwd on a long arm that kinks in f_meas. (028084_16/47)
7. A chain row typed pdg 13 that is not the muon. (029107_4/58)
8. The isochronous exception has no verdict path. (029107_28/30, 029107_28/27)

# PDHD w1_a4 scanner report (summary of its hand-back)
21 / 21 written. Scratch helper ctx.py is in its own scratch dir. After an advisor review of its own transcript, three records were rewritten:
- 029107_7/47: STM_ONLY → STM_MICHEL both;
- 028084_11/46: STM_ONLY FLAT_STOP → UNCLEAR;
- 029107_13/111.

Tally: 11 THRU, 7 STM_MICHEL, 2 STM_ONLY, 1 UNCLEAR.

**Pins and undershoots**
- Pins moved:
  - 029107_21/43 (rr 6.0);
  - 028084_16/110 (rr 5.5, ANODE);
  - 029107_13/111 (rr 13.0; recorded as STM_ONLY because of the split row).
- Undershoot-like signals it did not act on: 029107_5/34, 029107_8/107, 029107_13/111.

**Hardest items**
- 029107_5/34: STM_MICHEL low at the upper end on a hook arm vs the arm being the same particle (THRU/UNCLEAR).
- 029107_13/111: STM_ONLY + pin vs STM_MICHEL (overshoot inside one segment).
- 029107_7/47: michel vs a detached capture gamma, 8–15 cm out after an empty stretch.
- 028084_16/110: STM_MICHEL at the anode vs THRU (an anode exit with an edge drop).
- 029107_3/51: STM_MICHEL vs MESSY (a four-prong vertex with an isochronous crossing).
- 028084_11/46: UNCLEAR (a coverage hole before the end).
- 029107_21/43: overshoot vs the muon carrying on straight.

**Rubric points it named**
1. An overshoot inside one segment can't be recorded as STM_MICHEL, because mkv needs a michel row. (029107_13/111)
2. A detached, non-compact piece 5–10 cm out. (029107_7/47)
3. Straight-on vs kink when cos_fwd is taken against a hook; the charge rises along the piece. (029107_5/34)
4. A coverage hole just before the end, with charge reappearing. (028084_11/46)
5. The anode "clear rise" bar, given low end charge on anode exits. (028084_16/110)
6. Display:
   - 029107_1/50 has an empty f_meas;
   - at a cathode end the continuation is not visible (028084_19/46);
   - f_meas vs 3-D disagree on attachment (029107_7/47).

# PDHD w1_a5 scanner report (summary of its hand-back)
18 / 18 written. Two records were rewritten through mkv.py to fix their notes: 028084_2/66 (CONTINUES dropped) and 028084_15/50.
Tally: STM_MICHEL 6, STM_ONLY 1, THRU 10, MESSY 1.

**Pins and undershoots**
- Pin moved: 028084_5/30 (rr 5.9).
- Undershoot: 029107_5/26. A 7 cm straight-on piece at 2x plateau, whose hottest charge sits at the kink.

**Hardest items**
- 029107_14/106: THRU (low, DIRECTION) vs STM_ONLY with a displaced Bragg peak.
- 028084_18/17: MESSY vs THRU (DIRECTION + NO_MEASUREMENT, APA2 holes).
- 029107_21/71: kind both vs attached.
- 028084_5/30: overshoot vs no overshoot; the kind stays attached either way.
- 028084_9/27: an APA0 wave could imitate the rise.

**Rubric points it named**
1. Rule 7 vs rule 4 at an upper end with no readable profile → THRU DIRECTION + NO_MEASUREMENT. (028084_3/28, 028084_2/66)
2. CONTINUES needs all three planes: what to do when one plane is dead? (028084_15/50 kept it; 028084_2/66 dropped it)
3. The undershoot wording has no numbers. (029107_5/26)
4. Overshoot vs a smeared end with nothing to bridge to. (029107_14/106)
5. Degenerate means size 0: 2-point 0.2 cm rows were tagged gamma. (028084_19/59)
6. Attachment distance alone decided the verdict. (028084_9/27, gamma at 16–18 cm)
7. A faint parallel image trail not in the object table, read as ghost charge. (028084_27/33)
