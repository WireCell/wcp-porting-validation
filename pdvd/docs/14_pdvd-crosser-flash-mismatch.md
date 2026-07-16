# PDVD cathode-crossing matches are paired to the wrong (dim) flash — run 039252

**Status: analysis only. No toolkit code, config, matcher, or drift velocity was
changed.** All inputs are our own `_vcal` pipeline products. A drift-velocity
change would be a stop-and-ask (CLAUDE.md §5.1); a wrong-looking physics number
is reported, not tuned (§5.7).

## Question

Hand-scanning run 039252 evt 298567 (ql_scan port-5017 display), several
cathode-crossing matches (one bottom apa-0 cluster + one top apa-4 cluster
sharing a flash) carry **implausibly low total PE** — e.g. gid24 = 41 PE,
gid36 = 110 PE, gid42 = 50 PE. A cathode crosser must produce *strong* light:
the X-ARAPUCAs sit on the cathode. User hypothesis: the **wrong flash is
paired**, and the true flash is a nearby brighter one — possibly requiring a
**different drift velocity** (later flash ⇒ higher v, earlier ⇒ lower v) to keep
the two halves meeting at the cathode. Test it, and check whether one common
velocity + flash re-assignment works across the whole run.

## Repro

```bash
cd pdvd
# per-event detail (evt298567 flash-by-flash, all candidate flashes in +/-250us):
python3 ql_light_calib/crosser_flash_velocity_evt298567.py
# common-velocity solver + run-wide sweep (writes crosser_flash_records.csv):
python3 ql_light_calib/crosser_common_velocity.py work/039252_*_vcal/calib-evt*.json
# figures:
python3 ql_light_calib/crosser_flash_figures.py
# inputs: work/039252_*_vcal/calib-evt*.json  (18 events, the candles reco)
#         photlib/pdvd-photlib-vis-v5-175nm.json (10 cm trilinear ANN library)
```

## Method

- **Crosser geometry is degenerate in `pi = t_flash * v`.** The T0 correction is
  a rigid x-shift `sign_offset * t_flash * v` per half, so the two halves'
  relative shift is `-2 * t_flash * v` and the cathode meeting depends only on
  `pi`. A different flash `t_k` reproduces the same meeting at
  `v_k = pi*/t_k` (`v_k/v_nom = t_pick/t_k`). Geometry alone cannot choose the
  flash — the **light** breaks the degeneracy. (`meeting()`,
  `best_pi()` in `crosser_common_velocity.py`.)
- **Predicted cathode light is ~invariant to a ~50 us T0 shift.** The PDVD
  library is a 10 cm trilinear grid (`step_cm=[10,10,10]`), so a 50 us
  (~7.8 cm) shift moves points < one cell; the predicted cathode pattern barely
  changes across neighbor flashes (verified: `predCath@k` column). So the
  **measured** cathode PE is the discriminator.
- **Cathode-only discrimination.** The prior gid61 study
  (`13_pdvd-qlmatching-debug-evt298567.md`) showed wall-XA/PMT predictions over-read
  up to 135x while the cathode (opdets 4–11) is trustworthy. KS is computed on
  the cathode channels only, **dropping channels the per-flash saturation veto
  zeroed** (measured ~0 where the template predicts large — e.g. gid61's railed
  opdet 10), which otherwise inflates the correct flash's KS (`cath_ks()`).
- **A valid match needs all three**: bright (cathode measured PE ≥ 300),
  cathode-pattern consistent (KS ≤ 0.40, the model's flatness residual is ~0.35),
  and meeting (d ≤ 25 cm, |x_mid| ≤ 15 cm). The flash field is dense, so no
  single gate is unique. Among valid flashes the smallest meeting distance wins.
- **Self-check:** the 3 hand-validated bright crossers gid47/61/83 must recover
  their own flash. They do (gid47 OK, gid83 OK, gid61 recovered once the
  saturation veto is handled). gid24 is the one hand-validated crosser the light
  puts elsewhere — flagged as a conflict below, not silently overruled.

## Result 1 — cathode crossers are systematically on a dim flash while a brighter one is available

Across the **18 events of run 039252, 74 cathode-crossing pairs** (the display's
`find_crossers` picks). Search window ±200 us = the meeting tolerance at nominal
velocity (so it is physically bounded, not arbitrary):

| | crossers | note |
|---|---|---|
| pick already on a bright cathode-consistent flash | **14** | correct |
| **pick is dim; ≥1 brighter cathode-consistent flash also meets (WRONG-FLASH)** | **60 (81%)** | defect |

The brighter flash is found at **nominal velocity** — no velocity change needed.
It is systematically **earlier** than the mis-picked dim flash: wrong-flash
`dt = true - pick` has **median −67 us, mean −73 us** (range −196..+127).

**Two important qualifications (do not over-read the 81%):**
- **The brighter flash is often not unique.** Only **14/60** wrong-flash crossers
  have *exactly one* bright+consistent+meeting flash; the median is **2 rivals**
  (max 10). So "a brighter equally-valid flash exists" is solid; *which* flash is
  the true T0 is ambiguous for ~3/4 of them and needs a hand-scan to pin. The
  automated single pick (min meeting distance among the valid set) is a
  best-guess, not a proof.
- **Diagnostic vs production.** These are `find_crossers`' picks (the crossers/
  candles hand-scan display), *not* the production QLMatching auto-selector. Of
  the 60, production QLMatching itself auto_selected the **dim** flash in **27**
  and the **bright** match in only 3 — so ~27/74 are genuine *production*
  mis-assignments; the remainder are diagnostic-tool picks from the best-KS pool.

evt298567 exemplars (nominal v = 0.1586):

| pick | measCath | → brighter flash | Δt | measCath | cathKS | meet d | n_rivals | verdict |
|---|---|---|---|---|---|---|---|---|
| gid36 (110 PE) | 109 | **gid35** | −103 us | 2792 | 0.23 | 5.9 cm | 1 | wrong-flash, **unique** |
| gid42 (50 PE) | 50 | gid41 | −29 us | 5230 | 0.14 | 4.1 cm | ≥2 | wrong-flash (rivals present) |
| gid24 (41 PE) | 41 | gid21 | −106 us | 625 | 0.32 | 9.8 cm | ≥2 | **conflicts hand-scan** (below) |
| gid47/61/83 | bright | (self) | 0 | — | 0.09–0.35 | — | — | pick correct |

**Root cause (candidate-pool defect).** `find_crossers` (and, for the 27,
production QLMatching) only consider flashes both clusters already **share a
bundle** with, then take the smallest meeting distance in that restricted pool —
where a *dim* flash often wins by a few cm. The brighter flash (e.g. gid35, which
meets at d=5.9 cm at nominal velocity) is never in the pool because the clusters
have no bundle at it. Confirming the *brighter* flash is the true T0 (vs a
coincident bright cathode flash from other activity) requires the hand-scan.

## Result 2 — velocity is nominal, not 0.153 (and this is not a velocity measurement)

This method is **leverage-poor** for velocity: `v_k = pi*/t_true` has sensitivity
∝ |t_flash|, so near-trigger crossers carry almost no information (evt298567
gid61 at t=−127 us reads +15% — noise). It should not be read as a velocity
measurement. The defensible statement:

- On the crossers whose flash is **unambiguous** (the 14 pick-already-bright,
  including the hand-validated gid47/61/83), implied **v = 0.158 — nominal**.
- The **−4% (≈ field-response 1.53 mm/us)** appears only on **two** evt298567
  crossers (gid24, gid36) whose flash identity is the very thing in question.
- Run-wide, the |t|-weighted implied v is **0.157–0.158** with a broad spread
  (0.135–0.182); nominal 0.1586, FR 0.153 and 0.150 all "fix" ~the same count.

So the FR=1.53 clue was a good lead but is **not** reopened by this analysis. The
drift velocity for this data is already pinned to 1.586 (cathode-pinned two-point
closure on `_vcal`; physical 1.566±0.006). A genuine FR-vs-reco velocity
consistency test is the dedicated full-drift-closure study, not this leverage-poor
side-analysis.

**Velocity vs time-offset (dt-vs-t_pick regression, 64 non-rail crossers).**
Fit `dt = slope*t_pick + b`: **slope = +0.0040** (⇒ v_true = 0.1580,
−0.4% of nominal — i.e. *nominal*), **intercept = −57 us**. RMS residuals:
constant-offset-only **59 us** ≈ full model 59 us ≪ velocity-only 82 us. The
data **favors a flat ~−60 us offset over a velocity error**: the mis-picked dim
flash sits a roughly *constant* ~60 us *after* the true bright flash, independent
of flash time (a velocity error would tilt `dt` proportionally to `t_pick` — it
does not). The scatter is large (±100 us), so this is a flash-*selection* defect
centered ~60 us late, not a clean single-valued physical time offset. See
`docs/pics/pdvd-crosser-dt-vs-t.png` and `pdvd-crosser-velocity-hist.png`.

**Anode cross-check — it is NOT a real time-base offset (would break the anode).**
A genuine constant −60 us charge/light time offset would displace *every*
reconstructed x by `v*dt ≈ 0.1586 * 60 ≈ 9.5 cm`, and anode-anchored tracks
would then reconstruct ~9.5 cm off the anode. Two facts exclude that:
1. **14/74 crossers are correctly picked (dt=0).** A global time (or velocity)
   offset would shift *all* of them; a subset being exactly right means the
   effect is per-crosser flash mis-selection, not a global miscalibration.
2. **The anode is independently calibrated to sub-cm** (the `_ctoff` ctoffset
   exam pinned the anode edge u=0 from anode-touching tracks; touchers ~+0.08 cm).
   A 9.5 cm offset would have been caught there. These crossers cannot self-test
   it — they end mid-drift (`|anode_u|` median 137 cm, i.e. they never reach the
   anode; the pick→match anode-end shift is the trivial `v*dt = +10.6 cm`).

Conclusion: the −60 us is the gap to a *mis-selected* dim flash, not a physical
time shift. The correct (brighter, earlier) flash is consistent with the existing
nominal velocity and sub-cm anode calibration.

## Result 3 — the gid24 conflict (present both readings, do not overrule)

gid24 is one of the four hand-scanned crossers (`find_crossers.KNOWN_298567`,
validated on the geometry: two halves meet at the cathode). The light prefers
the brighter gid21 (625 vs 41 cathode PE) 106 us earlier. Two readings:

1. **Hand-scan right (gid24):** the pair genuinely crosses at gid24; its light is
   low for a detector/efficiency reason (as gid61's cathode pattern was distorted
   by saturation + model residual in the prior study), and gid21 is a coincident
   bright cathode flash from other activity that happens to meet.
2. **Light right (gid21):** gid24 (41 PE) is unphysically dim for a cathode
   crosser and gid21 is the true flash; the hand-scan validated geometry, not
   magnitude.

The hand-scan `crossers` tag predates the `_vcal` reco (cluster ids were
renumbered), so the two are not strictly the same object. Not resolved here.

## Findings

1. **Dominant defect: flash mis-assignment, at nominal velocity.** For 60/74
   (81%) run-039252 cathode crossers a *brighter*, cathode-pattern-consistent
   flash that also satisfies cathode meeting exists ~67 us *earlier* than the
   dim pick. Root cause is the shared-bundle candidate pool + min-distance pick.
   **Caveats:** the brighter flash is *unique* for only 14/60 (median 2 rivals),
   so the exact reassignment is often ambiguous; and production QLMatching
   auto_selected the dim flash for 27/60 (the rest are `find_crossers` diagnostic
   picks). "≥1 brighter equally-valid flash exists in 81%" is the safe claim.
2. **Not a velocity error — velocity is nominal.** On the unambiguous crossers
   implied v = 0.158 (nominal); the dt-vs-t regression prefers a flat offset over
   a velocity term (slope ≈ 0). The −4%/FR-1.53 appears only on two disputed
   evt298567 crossers and does not generalize. This is leverage-poor and is not a
   velocity measurement (§ Result 2).
3. **Not a real time-base offset either.** A −60 us offset would move anode
   reconstruction ~9.5 cm; the sub-cm anode calibration and the 14 correctly-
   picked crossers exclude a global shift. It is per-crosser flash mis-selection.
4. **gid24 hand-scan vs light conflict** surfaced, both readings given (§ Result 3).
5. Candidate fix (NOT implemented, would need a default-OFF knob + revalidation):
   widen the crosser flash candidate pool beyond shared-bundle flashes and rank by
   cathode light (bright + KS) as well as meeting distance.

## Open items

- Whether the brighter light-preferred flashes are the true T0s (vs coincident
  bright cathode flashes from other activity) needs a **hand re-scan** — cheapest
  is an eyeball on port 5017: does reassigning gid36→gid35 line up c107+c4000131
  as one track across the cathode? Use a **fresh tag**, not the existing
  `crossers`/`candles` labels (M13).
- For the ambiguous cases (n_rivals ≥ 2), the true flash is not decidable from
  cathode light alone; wall/PMT timing or the full waveform is needed.
