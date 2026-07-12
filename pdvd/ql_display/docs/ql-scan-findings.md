# AI Q/L hand scan of run 039252 events 0-9 — findings

Scan performed 2026-07-10 by Claude (event 0 directly, events 1-9 by one
scan agent each following the same protocol), rubric = `ql-scan-criteria.md`,
method = `ql-scan-method.md`. Everything below is reconstructable from
`../decisions/decisions-evt*.jsonl` (each verdict carries its evidence in
`reason`).

**To review interactively** (selections auto-restore per event):

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
./ql_scan/serve_ql_scan.sh 5017 --tag claude work/039252_{0,1,2,3,4,5,6,7,8,9}/calib-evt*.json
```

Start with the `low`-confidence decisions (83 of 662) — grep them per event:
`grep '"confidence": *"low"' ql_display/decisions/decisions-evt<ID>.jsonl | python -m json.tool` —
and the per-event "most interesting cases" notes in §4.

## 1. Headline numbers

662 explicit verdicts over every auto-selected bundle with a >5 cm main
cluster; sub-5 cm autos (~25-50/event, hidden by the viewer's length filter)
were implicit keeps, as in a human scan.

| event | judged | keep | reject | add | agree w/ auto |
|---|---|---|---|---|---|
| evt298567 (idx 0) | 54 | 30 | 24 | 0 | 56% |
| evt298581 (1) | 63 | 28 | 35 | 1 | 44% |
| evt298595 (2) | 71 | 37 | 34 | 2 | 52% |
| evt298609 (3) | 64 | 32 | 32 | 1 | 50% |
| evt298623 (4) | 78 | 41 | 37 | 0 | 53% |
| evt298637 (5) | 72 | 44 | 28 | 0 | 61% |
| evt298651 (6) | 66 | 38 | 28 | 1 | 58% |
| evt298665 (7) | 76 | 40 | 36 | 0 | 53% |
| evt298679 (8) | 57 | 26 | 31 | 1 | 46% |
| evt298693 (9) | 61 | 35 | 26 | 2 | 57% |
| **total** | **662** | **351** | **311** | **8** | **53%** |

Confidence: 228 high / 359 med / 83 low. The 53% keep rate lands between
SBND (human confirmed nearly everything — matcher well calibrated) and PDHD
(human kept only ~15% of auto picks): the PDVD round-1 matcher is doing real
work but roughly half its selections do not survive scrutiny.

The scan's metric profile matches the human precedent (PDHD kept
pred/meas median 0.76):

| | ks q25/50/75 | chi2/ndf | pred/meas |
|---|---|---|---|
| keep (351) | 0.15/0.27/0.39 | 2.4/6.7/17 | 0.56/**0.79**/1.11 |
| reject (311) | 0.29/0.41/0.58 | 5.6/20/76 | 0.04/**0.33**/1.04 |

3/351 keeps vs 111/311 rejects fall outside pred/meas [0.1, 10] — amplitude
remains the leading discriminator; KS separates only relatively (keep median
0.27 confirms the gold-pair finding that PDHD-style KS ceilings are inert
here).

## 2. Systematic findings (matcher-relevant)

1. **One-cluster-many-flashes is the dominant auto-matcher failure.**
   ~90 clusters per the 10 events were auto-kept on 2-5 flashes
   simultaneously (174 verdict lines participate in resolving them; e.g. one
   cluster claimed 5 flashes in evt298623). Physically a cluster has one T0.
   The LASSO fits each flash independently and gladly reuses a long track as
   background for every flash it vaguely fits. **Knob implication: a
   cross-flash exclusivity/veto pass (keep the best-amplitude+shape home,
   PDHD xtpc_joint_pin-style) would fix the largest error class without any
   PE-scale progress.**
2. **Bright orphan flashes.** Every event has 5-10 flashes of 3-15 kPE whose
   only candidates are amplitude-starved (r < 0.1) accidentals — all
   rejected. Estimated tens of kPE per event with no in-volume charge
   partner: out-of-image/under-threshold activity, not matcher failure. A
   `reject_overpred`-mirror (reject *under*-pred: flash needs > x5-10 the
   candidate's light) would auto-kill most of this accidental class.
3. **nearPD asymmetry works as a criterion.** meas >> pred concentrated on
   the PD nearest the track = proximity effect, keepable (kept at r as low
   as 0.2); pred >> meas on a channel the flash doesn't light = never
   excusable, rejected regardless of chi2 (seen at r up to 18 with
   chi2/ndf ~1 — the pred-based errors mask amplitude failure; a raw
   amplitude-ratio cut is complementary to chi2, not redundant).
4. **at_cathode T0 degeneracy is real and severe.** Cathode-hugging clusters
   fit 10+ flashes at indistinguishable KS (e.g. c110/c23 in evt298609,
   c18/c27/c36/c48 in evt298651 appearing as strength-0 candidates in 20+
   groups each). Their keeps are low-information for calibration; several
   are flagged `low`. Matches the planned post-scan treatment of
   `flag_at_cathode`.
5. **spec_end rubric held.** Every spec_end auto encountered (~8) was either
   independently rejectable or its cluster had a clean spec_end-free home.
6. **Repeated flash signatures.** Two events show near-identical measured
   patterns repeating 5-200 µs apart (evt298609 grp150/158/163: same ~2 kPE
   bottom ch4-5 spike x3; evt298651 grp191/193 twins 20 µs apart) — only one
   instance can have the charge partner. Afterpulsing or a recurring local
   source near those PDs; worth a light-side look.
7. **Single-channel giants.** Several flashes carry 2-7 kPE on ONE bottom
   channel (evt298693 grp152: 7.2 kPE on ch17 of an 8.2 kPE flash;
   evt298623 grp66/78/118 ~2.2 kPE single-channel) that no prediction
   approaches — saturation-like / point-source-like, and consistent with
   the uncalibrated per-channel scale (x13 spread within cathode XAs).
8. **LASSO artifacts**: `badmatch,nearPD` candidates with pred ~0 PE recur
   in nearly every group and are never credible; and the ident=-1 pseudo
   cluster (uid 3999999) appears as an auto in two events — dump-side
   artifact worth a look in the calib writer.

## 3. The 8 adds (all conservative re-homings, none pure rescues)

Every add moves a cluster whose auto flash was rejected to the flash the
compare-table evidence says it belongs to (r ~ 1 at the new home):
evt298581 c95→grp50 (r1.01); evt298595 c47→grp119 (ks0.11 r1.02, LASSO had
st0.09 at its auto home), c192→grp87 (r1.12, atCATH, low); evt298609
c110→grp117 (degenerate cathode cluster, low); evt298651 c36→grp116 (ks0.10
c2n0.3 r1.05 — textbook override of a st0.88 auto pick); evt298679
c10→grp156 (r0.56, both auto slots bad); evt298693 c299→grp114 (twin-flash
disentangle), c6→grp117 (rescues the flash with the event's worst
overpredict, low).

## 4. Per-event review pointers (agents' "most interesting cases")

- **evt298567**: c15 (695 cm) three-way flash ambiguity (grp63/71/75, kept
  75 low); c96 trio → grp84; note grp84's apparent-but-rejected cathode
  alignment with c76 (c76 kept at grp100 on ks0.09+consist — re-check).
- **evt298581**: grp70 c369 rejected at strength 0.92 (pred spikes x3 on a
  dark channel); grp140 c1 kept low at r0.29; 11 kPE grp139 orphan.
- **evt298595**: grp119/121 c47 swap; grp110 has all-bad autos (kept one
  amplitude anchor, low); two ~8-9 kPE orphan flashes.
- **evt298609**: c171 4-flash resolution; grp188 kept at r11.6 (wtrunc
  excuse — the scan's most aggressive keep, re-check); repeated ch4-5 spike
  trio.
- **evt298623**: c120 (489 cm crosser) rejected everywhere — a long track
  with no T0; grp95 pruned 5→2; grp186 c242 kept at r2.27 on shape.
- **evt298637**: c132 (663 cm) triple → the 11.4 kPE giant; grp174 r=11
  overprediction at c2n1.5 rejected — chi2 blind to scale.
- **evt298651**: c36 migration (the flagship add); grp187 collinear fragment
  trio pruned; grp191/193 twin orphans.
- **evt298665**: grp77 c70 rejected ONLY for spec_end despite good metrics —
  most likely false negative of the rubric; c325/c1 cosmics share 5 large
  flashes, two left orphaned.
- **evt298679**: grp156/157/168 triangle untangle; grp59 c15 kept at r3.31
  (wtrunc); biggest flashes of the event (8-15 kPE) all unmatched; c91
  (620 cm) fits nothing sanely.
- **evt298693**: grp117 c189 rejected at r18 despite ks0.20 (atCATH warning
  case); grp152 single-channel 7.2 kPE; grp98 c188 kept at auto against
  compare-table preference (low, re-check).

## 5. Implications for the §2-item-7 tuning list (pdvd-ql-pending.md)

- **measured_pe_scale refit**: 351 scan-vetted matches (228 high-conf) now
  exist — enough to refit per-channel scale on scan GT instead of the
  single-topology beam-gold sample.
- **reject_overpred**: turn ON with a ceiling near pred/meas ~ 5-10 (only 3
  keeps sit outside [0.1,10] and each carries a wtrunc excuse); add the
  under-pred mirror for the accidental class.
- **KS/chi2 ladder ceilings**: still premature — keep median ks 0.27 means
  any PDHD-style ceiling (0.06-0.10) rejects most true matches; revisit
  after the per-channel refit.
- **New knob candidate**: cross-flash cluster exclusivity (one T0 per
  cluster) — the largest single improvement available (fixes ~90 duplicate
  keeps across 10 events); doable without any calibration input.

## 6. Caveats

This is an AI scan intended as a test against the human scan (tag `first10`
when it exists). Verdict quality is uneven in the known-hard classes:
at_cathode clusters (T0-degenerate), r 0.1-0.3 nearPD keeps, and the
three-way flash ambiguities — all flagged low/med with the alternatives
named in `reason`. Disagreements found during the human review are exactly
the interesting sample: they localize where the rubric (or the round-1
calibration) misleads.

## 7. Xe/175 nm re-scan round (2026-07-11)

After the library switch (toolkit `0adb15fa`: 175 nm grid, `eff_Xe`,
ch 13/29/39 unmasked, QtoL 0.082) the 10 events were reprocessed and the scan
REDONE from scratch (independent agents, same rubric; old `claude` tag =
128 nm round, valid only against the archived `128nm-calib-evt*.json` dumps).
Decisions: `../decisions-xe/`; labels tag **`claude-xe`**.

| event | judged | keep | reject | add | agree w/ auto |
|---|---|---|---|---|---|
| evt298567 (idx 0) | 60 | 25 | 35 | 6 | 42% |
| evt298581 (1) | 61 | 27 | 34 | 3 | 44% |
| evt298595 (2) | 76 | 37 | 39 | 6 | 49% |
| evt298609 (3) | 68 | 38 | 30 | 0 | 56% |
| evt298623 (4) | 82 | 39 | 43 | 2 | 48% |
| evt298637 (5) | 79 | 43 | 36 | 2 | 54% |
| evt298651 (6) | 75 | 34 | 41 | 3 | 45% |
| evt298665 (7) | 83 | 35 | 48 | 2 | 42% |
| evt298679 (8) | 55 | 27 | 28 | 3 | 49% |
| evt298693 (9) | 71 | 35 | 36 | 2 | 49% |
| **total** | **710** | **340** | **370** | **29** | **48%** |

Confidence: 174 high / 429 med / 136 low. Metric profile:

| | ks q25/50/75 | chi2/ndf | pred/meas |
|---|---|---|---|
| keep (340) | 0.20/0.28/0.38 | 1.8/4.2/9.4 | 0.64/**0.90**/1.19 |
| reject (370) | 0.27/0.39/0.53 | 4.3/14/48 | 0.07/**0.34**/0.93 |

3/340 keeps vs 109/370 rejects outside pred/meas [0.1, 10].

Deltas vs the 128 nm round (§1-§5):

1. **The calibrated QtoL turned amplitude into a sharp knife.** Keep
   pred/meas tightened from 0.56/0.79/1.11 to 0.64/0.90/1.19 (median ~0.9);
   every scan agent independently reported the r ∈ [0.3, 3] band as the
   decisive criterion, with rejects bimodal: amplitude-starved
   strength≈0.95 stubs (r < 0.1 — the LASSO's high-strength accidental
   riders on bright flashes) and a few x5-10 over-predictors whose true
   flash is window-truncated.
2. **The auto set grew (662 → 710 judged) and the additions hurt purity.**
   Agreement dropped 53% → 48%; the growth is mostly (a) small marginal
   bundles admitted at the new QtoL and (b) more one-cluster-many-flashes
   duplicates (~85 arbitration rejects across 10 events, several clusters
   auto-kept on 3-5 flashes). Cross-flash exclusivity remains the single
   largest available improvement, unchanged from §5.
3. **More adds (8 → 29), now including genuine rescues.** With amplitude
   discriminating, agents confidently re-homed greedy wrong-flash autos
   (e.g. evt298567 c18/c73/c139/c141) and rescued LASSO-stranded big
   tracks onto bright unowned flashes (evt298679: 620 cm track → 15.3 kPE
   flash at r 1.09; evt298637: 448 cm track → 9.8 kPE flash at r 1.10;
   evt298651: 318 cm never-auto cluster at r 0.87).
4. **Newly-live ch 13/29/39 are unremarkable** — populated in bottom-volume
   maps/panels, consistent with predictions in keeps, and NO verdict hinged
   on them; ch32 correctly absent everywhere. The unmasking added ndf
   without adding pathology.
5. **Cathode in-group gain spread is still the visible residual**: multi-
   bundle bright-flash sums of ~2x were tolerated (known x2.5-3 spread),
   single-cathode-XA channels occasionally carry x8-12 per-channel
   mispredictions with a clean total, and KS stays relative-only (keep
   median ks 0.28 ≈ round 1's 0.27 — the library switch did not move KS,
   consistent with the gold-pair A/B where 175 nm won only marginally).
6. **Bright-flash orphans persist** (7-15 kPE flashes with no amplitude-
   sane candidate in several events) — charge partner truncated or outside
   the readout; deliberately left unmatched rather than force-fitted.

Tooling note: evt298595 exposed a uid collision (two distinct ident=-1
clusters share uid 3999999 under one flash). `make_labels.py` now accepts an
optional `bundle_idx` in a decision line to disambiguate, and the round-trip
check downgrades the inherent scan-state ambiguity (the viewer keys ticks by
(flash, uid)) to a WARNING. The labels JSON remains the exact record.

Review:

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
./ql_scan/serve_ql_scan.sh 5017 --tag claude-xe work/039252_{0,1,2,3,4,5,6,7,8,9}/calib-evt*.json
```

## 8. Geometry-fix round (2026-07-11, tag `claude-geomfix`)

**Root cause found while reviewing the display**: PDVD's active-volume box
in `QLMatching::compute_geometry` was built from only ONE of an anode's two
wire "faces". On PDHD/SBND the two faces of an anode cover the full Y/Z and
differ only slightly in X, so a single face is a safe proxy for the whole
box. PDVD instead splits each anode's Y range across its two faces (adjacent,
disjoint halves), so using one face left the drawn box (and the light-
prediction inclusion gate at `QLMatching.cxx:~1340`) covering only
Y=[-168.5,+336.4] cm instead of the true +-336.4 cm — any charge below
y~-168 cm was silently excluded from predicted light. Fixed in toolkit
`565ccd62` (new `tpc_extra_faces` knob, PDVD-only, empty/no-op for
PDHD/SBND — byte-identical there) by unioning face 1's box in. Impact check
on evt0: 990/3795 bundles with non-trivial predicted light changed by >5%,
several clusters going from 0 predicted PE (unmatchable) to thousands.

All 120 PDVD `-calib` dumps (039252/039253/039349, all idx) were
reprocessed under the fix; QtoL was refit on the same 80 beam-flash gold
pairs: **0.082 -> 0.070** (fixing the Y truncation raised predicted light
overall, so QtoL moved down to compensate). The 10 scan events were
reprocessed again at the new QtoL and rescanned FROM SCRATCH (independent of
both the 128nm round and the pre-geometry-fix Xe round) — evidence under
`png-geomfix/` + `context-geomfix/`, decisions under
`decisions-geomfix/decisions-evt*.jsonl`, labels tag `claude-geomfix`.

| event | groups w/auto | auto bundles | keep | reject | add |
|---|---|---|---|---|---|
| 298567 | 56 | 75 | 39 | 36 | 1 |
| 298581 | 62 | 73 | 40 | 33 | 6 |
| 298595 | 64 | 87 | 55 | 32 | 1 |
| 298609 | 64 | 82 | 54 | 28 | 0 |
| 298623 | 75 | 98 | 58 | 40 | 0 |
| 298637 | 61 | 89 | 57 | 32 | 0 |
| 298651 | 78 | 104 | 55 | 49 | 0 |
| 298665 | 65 | 92 | 48 | 44 | 1 |
| 298679 | 48 | 64 | 39 | 25 | 0 |
| 298693 | 64 | 81 | 51 | 30 | 2 |
| **total** | | **845** | **496** | **349** | **11** |

Agreement with auto = 496/845 = **58.7%** (up from 48% in the pre-fix Xe
round) — consistent with the matcher now actually seeing the charge it
predicts light for, rather than padding the auto set with amplitude-starved
accidentals from the truncated box. Confidence-weighted evidence and
per-bundle reasons are in the `decisions-geomfix/*.jsonl` files.

Deltas vs the pre-fix Xe round:
1. The dominant auto-matcher error is still **one-cluster-many-flashes**
   duplicates (every event has 1-15 clusters auto-selected on 2-5 flashes
   simultaneously); resolved via the compare-table (`competing_flashes`)
   exactly as before — amplitude/chi2 first, KS relative-only.
2. **evt298651 is notably messier post-fix**: only 47/78 auto-having groups
   retain any kept cluster, and the scanning agent explicitly flagged the
   matcher as unreliable there (many auto picks a few x10^-4 to 10^-2 of
   the measured light) — a genuine event, not a scan artifact; worth a
   closer look if this event matters downstream.
3. The evt298595 uid=3999999 collision (two distinct ident=-1 clusters
   sharing a uid under one flash) is confirmed to persist post-fix; handled
   the same way (`bundle_idx` disambiguation, WARNING not error).
4. amplitude-starved rejects (pred/meas << 0.1, flat/near-zero predicted
   line against real measured bars) remain the majority failure mode among
   rejects, same as every prior round — the truncation bug affected WHICH
   clusters got nonzero predicted light, not the shape-matching logic
   itself.

Review:

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
./ql_scan/serve_ql_scan.sh 5017 --tag claude-geomfix work/039252_{0,1,2,3,4,5,6,7,8,9}/calib-evt*.json
```

## 9. Cathode-crosser standard-candle scan (2026-07-11, tag `crossers`)

The owner hand-picked 4 cathode-crossing pairs in evt298567 (trimmed
`claude-geomfix` scan state, gids 24/47/61/83). Those picks were quantified
and generalized into a five-criterion recipe + finder
(`find_crossers.py`, method and all numbers in
`docs/ql-cathode-crosser-recipe.md`): same-flash top+bottom pair, spans
>=50 cm, yz-continuity <=25 cm, min(global-PCA, local-end) axis angle
<=10 deg, full-cluster closest approach <=25 cm with meeting midpoint
|x_mid| <=10 cm (this last cut kills the overlapping-halves fake), flash =
min-d with brighter-flash tie-break.

- Self-check: recovers the 4 owner picks at exactly the picked flashes.
- New finds (all render-verified): evt298567 +2 (gid42, gid96), evt298581
  6 (incl. gid173: a 4 m + 3.6 m pair collinear to 1.3 deg on a 59 PE
  flash), evt298595 5. Total 17 pairs / 3 events.
- Key negative lesson: distance+collinearity alone is NOT enough — the
  dominant fake is a same-flash pair whose x-extents overlap across the
  cathode at that T0 (d ~ 8 cm, aG ~ 1 deg, but meets 48 cm inside a
  volume). The |x_mid| cut removes it.
- Dim-flash candles are real: 3 of 17 picks sit on 41-59 PE flashes with
  ~10x overprediction (the owner's own gid24 pick among them) — candidates
  for saturation-veto / remove_late_light investigations, not scan errors.
- Labels: `work/ql_labels/crossers/` (crossers-only: all other auto bundles
  rejected). Decisions record: `ql_display/decisions-crossers/`.

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
./ql_scan/serve_ql_scan.sh 5018 --tag crossers work/039252_{0,1,2}/calib-evt*.json
```
