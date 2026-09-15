# doc pdvd/103 owner review queue (pdhd)
Cells column: letters of the cells that tag the item (A0 production, K fit knobs, S sampler, A1 both on); '.' = not tagged.  Label source: new_agent = this round's verdict-blind scan; smx22/record/owner* = existing.

## A. False positives of A1 (both on) on the union record

### is_stm: 9 (also an FP in A0: 1)

| key | hand verdict | michel_kind | label source | confidence | tagged in | notes (new labels) |
|---|---|---|---|---|---|---|
| 028084_12/46 | THRU | none | smx22 | medium | A0K.A1 |  |
| 028084_17/35 | THRU | — not set — | owner | medium | ...A1 |  |
| 028084_18/122 | THRU | none | smx22 | medium | .K.A1 |  |
| 029107_14/106 | THRU | none | new_agent | low | ...A1 | DIRECTION: fit end is the upper end (dy/chord 0.42); competing reading STM_ONLY  |
| 029107_16/46 | THRU | none | smx22 | high | ..SA1 |  |
| 029107_17/49 | THRU | none | new_agent | high | ...A1 | DIRECTION: upper fit end, flat profile; ANODE: the straight-on stub reaches x=-3 |
| 029107_2/64 | THRU | none | smx22 | medium | ...A1 |  |
| 029107_22/22 | THRU | none | new_agent | high | ...A1 | CONTINUES: grey in U, V and W carries the fragment line on past the fit end at m |
| 029107_3/86 | THRU | none | smx22 | medium | ...A1 |  |

### michel_found: 14 (also an FP in A0: 2)

| key | hand verdict | michel_kind | label source | confidence | tagged in | notes (new labels) |
|---|---|---|---|---|---|---|
| 028084_10/109 | STM_ONLY | detached dots | smx22 | medium | ...A1 |  |
| 028084_23/114 | STM_ONLY | detached dots | owner | medium | A0K.A1 |  |
| 028084_8/120 | STM_ONLY | none | smx22 | medium | ...A1 |  |
| 028084_9/27 | STM_ONLY | detached dots | new_agent | medium | ..SA1 | Rise read relative to a very low APA0 plateau; far piece 159024 at 16-18 cm is t |
| 029107_11/86 | STM_ONLY | detached dots | new_agent | medium | ..SA1 | detached pieces at 9.6-12 cm (chain Michel objects) tagged gamma by the gap rule |
| 029107_12/33 | STM_ONLY | detached dots | smx22 | medium | ..SA1 |  |
| 029107_19/111 | STM_ONLY | detached dots | smx22 | high | ...A1 |  |
| 029107_23/116 | STM_ONLY | none | smx22 | medium | ...A1 |  |
| 029107_23/41 | STM_ONLY | none | new_agent | medium | .KSA1 | CATHODE: stop 0.2 cm from the cathode, accepted as a stop on a clear rise; mediu |
| 029107_26/111 | STM_ONLY | none | smx22 | high | ..SA1 | OVERSHOOT: peak at s~8-12 cm, sustained collapse over the last ~6 cm inside segm |
| 029107_27/39 | STM_ONLY | detached dots | smx22 | medium | .KSA1 |  |
| 029107_4/58 | STM_ONLY | detached dots | smx22 | medium | A0.SA1 |  |
| 029107_6/122 | STM_ONLY | detached dots | smx22 | high | .K.A1 |  |
| 029107_7/34 | STM_ONLY | none | smx22 | medium | ...A1 |  |

## B. New blind labels on items tagged is_stm or michel_found in some cell: 69

| key | verdict | michel_kind | confidence | is_stm in | michel_found in | display arm | notes |
|---|---|---|---|---|---|---|---|
| 028084_0/121 | THRU | none | high | .... | .KS. | d102hocs | DIRECTION: fit end is the upper end (dy=+0.78 chord), chain entry near the floor |
| 028084_0/130 | THRU | none | high | .... | ..S. | d102hocs | ANODE: face.x 0.0 cm; long face-to-face track, no rise |
| 028084_0/36 | STM_MICHEL | both | high | ...A1 | ...A1 | d102hcs | Rise is weak (~1.4x plateau over ~5 cm); STM from the kinked arm. 194022 tagged  |
| 028084_0/91 | THRU | none | high | ..S. | ..S. | d102hcs | DIRECTION: fit end is the upper end (dy +35 of 54 cm chord), flat end; CONTINUES |
| 028084_1/60 | THRU | none | high | .... | .K.A1 | d102hcs | ANODE: end at face.x 0.1 cm, flat profile; chain Michel object is a 0.8 cm strai |
| 028084_11/46 | UNCLEAR | none | medium | .... | ..S. | d102hocs | NO_MEASUREMENT: APA0 wave over the last 20 cm and no U/W cells over the last ~25 |
| 028084_12/132 | STM_MICHEL | attached | high | ..SA1 | ..SA1 | d102hcs | Chain-typed pdg-13 branch 132009 retagged delta / other (branch 13 cm back on th |
| 028084_12/37 | THRU | none | high | .... | ...A1 | d102hcs | DIRECTION: upper end of a floor-to-mid-volume fragment, no topology; CONTINUES:  |
| 028084_13/39 | THRU | none | high | .... | .K.. | d101hkf | ANODE: face.x 0.3 cm; last 15 cm at wave-crest height (APA0), not read as a rise |
| 028084_15/24 | THRU | none | high | .... | ..S. | d102hcs | DIRECTION: upper end (dy/chord 0.87); SEAM: 1.5 cm from z=231; ISOCHRONOUS: dx 2 |
| 028084_16/47 | STM_MICHEL | both | medium | ..S. | ..SA1 | d102hcs | OVERSHOOT: peak at s~12-15 cm, collapse over the last ~6 cm inside 47007; pin mo |
| 028084_17/34 | THRU | none | high | .... | .K.A1 | d102hcs | ANODE: fit end 0.4 cm from the anode face with flat profile; the chain Michel 33 |
| 028084_18/22 | STM_ONLY | none | low | ..S. | .... | d102hocs | FLAT_STOP: lower mid-volume end, charge ends in U/W, straight-on stub only; NO_M |
| 028084_18/85 | THRU | none | medium | .... | ..S. | d102hocs | ANODE: end at face.x 0.3 cm, flat profile; faint 7 cm angled piece 85016 on the  |
| 028084_19/46 | THRU | none | high | ..S. | ..S. | d102hocs | CATHODE: 1.6 cm from the cathode face, no rise; stubs 45045 (other cluster) and  |
| 028084_2/20 | STM_MICHEL | both | high | .KS. | .KSA1 | d102hcs | OVERSHOOT: peak at s~8-13 cm, collapse over the last ~7 cm (end of 20024 plus al |
| 028084_2/40 | STM_MICHEL | attached | medium | ..SA1 | ..S. | d102hcs | STM_MICHEL vs STM_ONLY rests on 40025, a straight-on 5.4 cm stub at ~0.4x platea |
| 028084_20/28 | STM_ONLY | none | high | .K.A1 | .... | d102hcs | UNDERSHOOT: 1.2 cm hot tip 28002 (1.96e5, straight ahead) is the muon peak; real |
| 028084_21/27 | STM_MICHEL | attached | high | ..SA1 | ..SA1 | d102hcs |  |
| 028084_22/3 | STM_MICHEL | attached | medium | ...A1 | .K.A1 | d102hcs | straight-on 3-4 cm low-charge stub is the only Michel candidate; STM_ONLY is the |
| 028084_24/111 | STM_MICHEL | both | high | .KSA1 | .KSA1 | d102hcs | gamma vs delta on the three far specks sets both vs attached; they are tiny (0.1 |
| 028084_24/136 | FRAG_THRU | none | high | .K.. | .K.. | d102hcs | CONTINUES: grey track carries the line on past both ends in U, V and W; DIRECTIO |
| 028084_24/62 | THRU | none | high | .... | ..S. | d102hocs | ANODE: face.x 0.9 cm; DIRECTION: upper end (0.43), other end 1 cm from the floor |
| 028084_24/65 | THRU | none | high | .... | .KSA1 | d102hcs | ANODE: fit end on the anode face (0.4 cm), flat profile, no topology |
| 028084_24/71 | THRU | none | high | .... | .K.A1 | d102hcs | ANODE: fit end at face.x=0.0 with flat profile; DIRECTION: fit end marginally up |
| 028084_25/39 | THRU | none | high | .... | .KSA1 | d102hcs | DIRECTION: upper fit end, flat profile; ANODE: fit end face.x 0.0 |
| 028084_26/41 | STM_MICHEL | attached | medium | ..SA1 | ..SA1 | d102hcs | UNDERSHOOT: 41033 is a 1.2 cm forward tip at ~2.4x plateau, tagged muon; real st |
| 028084_26/94 | STM_ONLY | detached dots | medium | ...A1 | .... | d102hcs | ISOCHRONOUS: x within 10 cm over 261 cm, charge ragged with dips; the rise rests |
| 028084_27/33 | STM_MICHEL | both | high | .KSA1 | .KSA1 | d102hcs | Parallel faint image-point trail alongside the body (not a table object) noted;  |
| 028084_3/90 | STM_MICHEL | attached | high | ..SA1 | .KSA1 | d102hcs | OVERSHOOT: peak at s=5-17 then 4.6 cm collapse (90017) before the 90 deg arm 900 |
| 028084_30/61 | STM_MICHEL | attached | medium | ..S. | ..S. | d102hocs | busy stop: faint parallel shower-typed pieces along the last 35 cm, read as imag |
| 028084_5/28 | THRU | none | high | .... | ...A1 | d102hcs | ANODE: fit end face.x 0.4 cm, flat; DIRECTION: fit end is the upper end (dy/chor |
| 028084_5/30 | STM_MICHEL | attached | high | .K.. | .K.. | d101hkf | OVERSHOOT: pin moved 5.9 cm back to the 30038/30021 junction; 30021 (27k, half p |
| 028084_6/35 | THRU | none | high | .... | .KS. | d102hocs | ANODE: fit end face.x 0.1 cm, flat profile; single hot last point is scatter. |
| 028084_9/27 | STM_ONLY | detached dots | medium | .... | ..SA1 | d102hcs | Rise read relative to a very low APA0 plateau; far piece 159024 at 16-18 cm is t |
| 028084_9/42 | STM_ONLY | none | medium | ..S. | .... | d102hocs | DIRECTION: fit end is the upper end (dy +38 of 48 cm chord), track enters throug |
| 028084_9/52 | THRU | none | medium | .... | ...A1 | d102hcs | ANODE: fit end face.x 0.2 cm, modest ragged lift not a clear rise, no topology |
| 029107_0/54 | THRU | none | high | .... | ..S. | d102hocs | ANODE: fit end face.x 0.0 cm, flat profile. |
| 029107_1/50 | THRU | none | high | .... | ..SA1 | d102hcs | ANODE: face.x 0.0 cm; DIRECTION: upper end (0.95), other end 0.4 cm above the fl |
| 029107_11/26 | STM_MICHEL | attached | medium | ...A1 | ...A1 | d102hcs |  |
| 029107_11/86 | STM_ONLY | detached dots | medium | .KSA1 | ..SA1 | d102hcs | detached pieces at 9.6-12 cm (chain Michel objects) tagged gamma by the gap rule |
| 029107_12/112 | THRU | none | high | .K.. | .K.A1 | d102hcs | DIRECTION: fit end is the upper end (dy=+0.58 chord), chain entry at the floor;  |
| 029107_12/29 | MESSY | none | medium | .... | ..SA1 | d102hcs | Rule 6 busy stop. Competing reading: STM_MICHEL on the chain Michel pieces near  |
| 029107_13/111 | STM_ONLY | detached dots | medium | ..S. | ..S. | d102hcs | OVERSHOOT: peak s~13-17, collapse s~3-9 to 0.6x the plateau, pin 13.0 cm; SPLIT  |
| 029107_14/106 | THRU | none | low | ...A1 | .... | d102hcs | DIRECTION: fit end is the upper end (dy/chord 0.42); competing reading STM_ONLY  |
| 029107_14/23 | STM_MICHEL | attached | high | ...A1 | ...A1 | d102hcs | UNDERSHOOT: 23056 (2 cm, 203k, cos_fwd 0.92) is the muon hot tip; real stop ~2 c |
| 029107_15/118 | STM_MICHEL | both | medium | ...A1 | ...A1 | d102hcs | OVERSHOOT: peak ~8e4 at 12 cm then collapse to 0.6-3e4 at 4-8 cm; pin moved to 1 |
| 029107_16/122 | STM_ONLY | detached dots | high | .KS. | .... | d102hocs |  |
| 029107_17/116 | STM_MICHEL | both | high | ..SA1 | ..S. | d102hcs | Gamma tag on 377004 (same side as the Michel) decides both vs attached. |
| 029107_17/49 | THRU | none | high | ...A1 | ...A1 | d102hcs | DIRECTION: upper fit end, flat profile; ANODE: the straight-on stub reaches x=-3 |
| 029107_19/29 | THRU | none | medium | ..S. | .... | d102hocs | CONTINUES: grey carries the line past the fit end in U, V and W with only a mild |
| 029107_20/148 | STM_ONLY | none | high | .KSA1 | .... | d102hcs |  |
| 029107_20/31 | THRU | none | high | .... | .K.. | d101hkf | ANODE: fit end face.x 0.1 cm with a flat profile; single hot last point is scatt |
| 029107_21/47 | STM_ONLY | none | high | ..SA1 | .... | d102hcs |  |
| 029107_21/71 | STM_MICHEL | both | medium | .K.A1 | .KS. | d102hcs | Tag doubt: 71013 may be a faint Michel continuation rather than an empty bridge, |
| 029107_22/108 | STM_MICHEL | both | high | ...A1 | ..SA1 | d102hcs | Borderline overshoot considered: peak at s~4-5 then the last ~1 cm at 20-40k; le |
| 029107_22/22 | FRAG_THRU | none | high | ...A1 | ..S. | d102hcs | CONTINUES: grey in U, V and W carries the fragment line on past the fit end at m |
| 029107_23/41 | STM_ONLY | none | medium | .KSA1 | .KSA1 | d102hcs | CATHODE: stop 0.2 cm from the cathode, accepted as a stop on a clear rise; mediu |
| 029107_24/111 | STM_MICHEL | attached | high | ...A1 | ...A1 | d102hcs | 432068 tagged michel as the contiguous tip of the arm (d 28.4 vs the arm d_max 2 |
| 029107_24/21 | THRU | none | medium | .... | ...A1 | d102hcs | CONTINUES: grey carries the line on past the fit end in U, V and W; DIRECTION: u |
| 029107_27/38 | THRU | none | high | .... | ...A1 | d102hcs | ANODE: fit end 0.1 cm from the anode face with a flat profile; DIRECTION: upper  |
| 029107_29/31 | STM_MICHEL | attached | medium | .... | .KS. | d102hocs | Profile unreadable (APA0 smooth wave with U/W coverage holes); verdict from topo |
| 029107_3/22 | STM_MICHEL | attached | medium | .K.. | .... | d101hkf | arm 22012 at an angle but very faint (5.8e3) inside a broad ghost-point sheet on |
| 029107_3/51 | STM_MICHEL | attached | medium | .... | ...A1 | d102hcs | Busy vertex: MESSY competes (crossing near-isochronous track at the stop time in |
| 029107_3/92 | STM_MICHEL | attached | high | .KSA1 | .KSA1 | d102hcs |  |
| 029107_5/26 | STM_MICHEL | attached | high | ..S. | ..S. | d102hocs | UNDERSHOOT: real stop about 7 cm past the fit end, at the far end of the straigh |
| 029107_5/34 | STM_MICHEL | attached | low | .... | ...A1 | d102hcs | Low: DIRECTION: upper end (0.90), stop called only on topology per rule 7 bullet |
| 029107_6/45 | THRU | none | high | .... | ..S. | d102hocs | CATHODE: fit end 0.0 cm from the cathode with a straight-on piece of another clu |
| 029107_7/47 | STM_MICHEL | both | medium | ..SA1 | ..SA1 | d102hcs | Stop is solid (rise, lower end, mid-volume); STM_MICHEL rests on reading the non |

## C. Other new blind labels at medium / low confidence: 19

| key | verdict | michel_kind | confidence | notes |
|---|---|---|---|---|
| 028084_12/33 | STM_ONLY | none | medium | FLAT_STOP: APA0 lower end, charge ends in all planes, only a weak ragged rise; 33004 strai |
| 028084_15/47 | THRU | none | medium | ISOCHRONOUS: dx 1.8 cm over 50 cm, ghost sheets; CONTINUES: grey horizontal line continues |
| 028084_16/102 | THRU | none | medium | DIRECTION: fit end is the upper end (dy/chord 0.91), flat, other end on the floor; DEAD: U |
| 028084_2/66 | THRU | none | medium | DIRECTION: fit end is the upper end (dy/chord 0.51); NO_MEASUREMENT: APA2 bottom-corner wa |
| 028084_23/49 | UNCLEAR | none | low | NO_MEASUREMENT: body charge an order below MIP over 50 cm, last-10 cm bump has no plateau, |
| 028084_24/44 | STM_ONLY | none | medium | FLAT_STOP: lower end 65 cm inside the anode, APA0, charge ends in U/V/W at the same slice, |
| 028084_24/67 | STM_ONLY | none | medium | FLAT_STOP: lower end, charge ends in all planes, no rise. Competing readings: OVERSHOOT wi |
| 028084_25/113 | THRU | none | medium | ANODE: end at face.x 1.5 cm; ragged last 5 cm read as scatter, not a clear rise |
| 028084_28/41 | STM_ONLY | none | medium | FLAT_STOP: lower end mid-volume, charge ends in all three planes, no rise (APA0 ragged). C |
| 028084_3/28 | THRU | none | medium | DIRECTION: fit end is the upper end (dy/chord 0.88), other end on the floor; NO_MEASUREMEN |
| 028084_4/53 | THRU | none | medium | DIRECTION: fit end is the upper end (dy +131 of 131 cm chord), track enters through the fl |
| 028084_8/29 | THRU | none | medium | DIRECTION: upper fit end of a track that reaches the floor, flat profile, no topology |
| 029107_0/59 | THRU | none | medium | ANODE: fit end face.x 0.0 cm; modest 1.5x elevation 3-12 cm before the end is not read as  |
| 029107_15/133 | UNCLEAR | none | medium | NO_MEASUREMENT: last ~40 cm of the fit carry no measured charge (133014 dQ/dx 6.8e3); ISOC |
| 029107_18/30 | UNCLEAR | none | low | NO_MEASUREMENT: wavy profile, terminal 2 cm spike unreadable; SEAM: end 3.8 cm from seam,  |
| 029107_18/65 | UNCLEAR | none | low | NO_MEASUREMENT: 23 cm track, no plateau or rise; ISOCHRONOUS: x varies 1.4 cm, one time sl |
| 029107_22/34 | STM_ONLY | none | medium | FLAT_STOP: mid-volume lower end, own charge ends in all planes, no continuation; NO_MEASUR |
| 029107_28/27 | UNCLEAR | none | low | NO_MEASUREMENT: near-zero charge over the last 15 cm, APA0 isochronous wave, no topology;  |
| 029107_4/124 | UNCLEAR | none | medium | NO_MEASUREMENT: whole W trace in a dead band, dQ/dx is filler wave; DEAD: end inside W dea |

## D. Calibration disagreements (blind vs existing record): 3

- 028084_16/110: existing THRU (smx22, medium) vs blind STM_MICHEL (medium)
- 029107_28/30: existing STM_MICHEL (smx22, low) vs blind UNCLEAR (low)
- 029107_3/23: existing STM_MICHEL (smx22, medium) vs blind THRU (medium)

## E. New labels that name the readout-window edge: 4

- 028084_24/62: THRU (high); is_stm in ....; notes: ANODE: face.x 0.9 cm; DIRECTION: upper end (0.43), other end 1 cm from the floor
- 028084_28/41: STM_ONLY (medium); is_stm in ....; notes: FLAT_STOP: lower end mid-volume, charge ends in all three planes, no rise (APA0 ragged). C
- 029107_13/26: THRU (high); is_stm in ....; notes: DIRECTION: upper end with a flat APA0 profile and only a straight-on stub; DEAD: the end s
- 029107_13/97: THRU (high); is_stm in ....; notes: DIRECTION: upper end, flat profile, no topology; DEAD: the end sits at time slice ~1480, a
