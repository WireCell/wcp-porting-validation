# 24 — Information and figure requests for the PDVD Q/L matching technote

Status: ANSWERED 2026-07-18 (analysis side) — every item below now carries an
**Answer** block with numbers and doc citations. Items (or sub-items) that
require the owner's decision or effort are explicitly marked **[NEEDS OWNER]**;
everything else is ready for the drafting session to consume.
Originally written by the paper-drafting session (2026-07-18). Each item lists
what the technote needs, why, and the concrete deliverable.

The technote lives in `~/DUNE/Papers/PDVD_Charge_Light_Matching/` and follows the
PDHD companion paper's structure, with two new sections: "Charge-readout geometry
and drift calibration" and "Post-matching refinements".

---

## 1. Space-charge correction **[BLOCKING — sec:postmatch-sce]**

The paper outline includes space-charge correction (SCE) among the post-matching
improvements, but no document in `docs/` describes a PDVD SCE study (only an
incidental mention of a t0/velocity/SCE degeneracy band in the PDHD
anode-time-consistency exam, `qlmatch/03`).

Needed (pick one):
- **If an SCE correction is applied or being developed:** method (simulated field
  map vs data-driven from crossing tracks?), where it enters the chain
  (clustering coordinates? post-matching drift-coordinate assignment?), typical
  displacement magnitude at the cathode, and 1–2 validation plots (e.g. spatial
  residuals of anode–cathode crossers before/after correction).
- **If not applied:** a short statement of the intended approach and expected
  magnitude for a surface detector, so the section can be written as outlook and
  the limitation stated honestly.

**Answer (2026-07-18): NOT applied — write the section as outlook/limitation.**
No explicit SCE correction (field map or data-driven) is applied anywhere in the
PDVD chain; the only SCE hooks in the toolkit are uBooNE-specific and inert for
PDVD. Facts available to write the outlook honestly:

- *Longitudinal common mode is absorbed by the velocity convention, not
  corrected.* PDVD data reco uses the cathode-pinned **convention velocity
  v = 1.586 mm/µs** (pooled gauss direct-pin, ±0.003) while the loss-corrected
  **physical velocity is 1.566 ± 0.006 mm/µs**; the 0.020 mm/µs gap is the
  anode+cathode detection shortfall (δa+δc ≈ 3.5 cm over 338.5 cm of drift)
  deliberately folded into an effective velocity so track extents span the full
  anode→cathode geometry (`qlmatch/02` §8.10–8.12, `qlmatch/14`). Any static
  longitudinal SCE distortion is degenerate with this pinning.
- *Magnitude scale from the PDHD companion analysis:* the anode-time-consistency
  exam bounds the degenerate t0/velocity/SCE band at **±2 cm**, with measured
  residuals at or below it (anode-edge pile-up −0.35 cm; cathode-edge
  −0.46/+0.52 cm; 41 crosser-pair midpoints at +0.69 cm, MAD 0.63)
  (`qlmatch/03` §5.3–5.4). A comparable sub-cm-to-few-cm scale is the honest
  expectation for PDVD.
- *Data-driven transverse evidence at the PDVD cathode:* on QL-confirmed
  cathode crossers, the tip-to-tip connection-vector direction is SCE-noisy
  (**34–69°** off the track axis) while the cluster PCA stays tight (<10°) —
  the two truncated tips are transversely displaced at the 6 cm cathode. The
  clustering response was to *widen the acceptance cuts* (cc3a operating point,
  item 2) rather than correct positions
  (toolkit `clus/docs/cathode-crossing-clustering.md`, PDVD campaign section).
- *Intended approach (outlook):* a data-driven map from T0-tagged cathode
  crossers, as prototyped on SBND (`sbnd_xin/cathode_distortion.py`): there the
  data transverse offset at the cathode is ~1.4 cm (rigid part
  T_yz = (−0.22, +1.34) cm) vs ~0.35–0.48 cm in MC
  (`match/docs/cathode-offset-correction.md`). The same procedure applies to
  PDVD once enough matched crossers are accumulated.
- **Cannot provide:** before/after validation plots of an applied correction —
  none exists.

## 2. Post-QLMatch cathode-crossing clustering **[BLOCKING — sec:postmatch-cathode]**

The outline calls for "cathode-plane crossing clustering after QLMatch". The
existing docs cover the *pre-match* `cathode_connect` clustering stage
(`nf_sp_img_clus/23`, run with `use_flash_t0=false`) and the *in-match* xTPC
crosser machinery (`09`, `qlmatch/16`), but not a distinct post-matching
re-clustering step that consumes the matched flash T0.

Needed:
- Algorithm description: how matched-flash T0 repositions clusters in the drift
  direction, the merge criteria across the cathode plane (position tolerance,
  angle?), and its relation to `cathode_connect`.
- Statistics on run 039252: merges per event, examples of pairs joined only
  after T0 is known.
- One before/after figure (projections or a Bee screenshot pair) of a
  cathode-crossing track re-unified after matching.

**Answer (2026-07-18): fully documented — this is the all-TPC
`ClusteringCathodeConnect` stage with the cc3a operating point (adopted
2026-07-17).** Authoritative doc: toolkit
`clus/docs/cathode-crossing-clustering.md` §6 + "PDVD 6 cm-cathode crosser
recall campaign (cc1a → cc3a)"; runner defaults `pdvd/run_clus_evt.sh` (cc3a
block); toolkit commit 4bc8ff04, wcp c255930.

- **Algorithm.** After QLMatching assigns each cluster its flash T0, every
  point is repositioned to `x_t0cor = x_apparent − v·t0` (per drift volume,
  with the per-crate trigger offsets of item 5). The final all-TPC clustering
  stage is the pipeline `[switch_scope, cathode_connect]` (no
  `examine_bundles`), so `cathode_connect` is the *sole* merge path there. It
  is the same `ClusteringCathodeConnect` component as the pre-match stage, but
  run with `use_flash_t0=premerged` (true in production): candidate pairs must
  share a flash-T0 group — i.e. the two halves' *matched flashes* gate the
  merge. Geometry gates (cc3a values): both tips within
  **cathode_x_cut = 8 cm** of the cathode plane; same drift depth
  **|ΔX| < drift_cut = 14 cm** (spans the 6 cm cathode + ~3 cm truncation per
  side); opposite drift volumes; collinearity within **angle_cut = 10°** (local
  Hough *or* cluster PCA). Pairs with 3D gap < **dis_cut = 16 cm** (close
  regime) accept on collinearity alone — the tip-to-tip connection vector is a
  cathode artifact (drift gap + SCE transverse), not track direction. Two
  PDVD-specific relaxations: **crosser_conn_relax = 75°** (relaxes, not drops,
  the connection-alignment bound in the both-long PCA branch; the ~90°
  connection of parallel coincident cosmics remains the purity guard) and
  **crosser_pca_angle = 15°** (admits bent/SCE-curved crossers; genuine
  crossers have inter-half PCA p90 ≈ 19° vs coincidences p50 ≈ 35° on a
  748-real/52-coincidence labeled table). Toolkit/jsonnet defaults stay OFF
  (byte-identical); the operating point is a runner default.
- **Census (28-event manifest, 039252×18 + 039349×10):** recall of
  QL-confirmed crossers **161 → 179 → 188 of 212 (76% → 84% → 89%)** for
  cc1a → cc2a → cc3a, unexplained merges flat at 9 (≤5% of spanning),
  engulfing 0, spurious ~0. The residual ~11% = QL over-pins (halves not
  reaching the cathode) + crossers with perpendicular tip connections
  indistinguishable from coincidences.
- **Run 039252 per-event statistics (NEW, computed 2026-07-18 from
  `clustering-global`, spanning = ≥20 points on each side of x=0):**
  cathode-spanning clusters go **57 → 176** (18 events) from the pre-merge
  nm4b baseline to the current production defaults — **+119 merges, mean +6.6
  per event, every event gains** (range +1 to +11):

  | idx | evtNo | before | after | | idx | evtNo | before | after |
  |---|---|---|---|---|---|---|---|---|
  | 0 | 298567 | 3 | 9 | | 9 | 298693 | 3 | 4 |
  | 1 | 298581 | 3 | 13 | | 10 | 298707 | 3 | 12 |
  | 2 | 298595 | 2 | 10 | | 11 | 298721 | 1 | 9 |
  | 3 | 298609 | 2 | 10 | | 12 | 298735 | 0 | 6 |
  | 4 | 298623 | 4 | 10 | | 13 | 298749 | 3 | 7 |
  | 5 | 298637 | 7 | 10 | | 14 | 298763 | 4 | 9 |
  | 6 | 298651 | 0 | 4 | | 15 | 298777 | 7 | 18 |
  | 7 | 298665 | 3 | 9 | | 16 | 298791 | 2 | 10 |
  | 8 | 298679 | 5 | 12 | | 17 | 298805 | 5 | 14 |

  (Caveat for the text: "after" also includes the doc-23 QL accuracy gains,
  which add matched clusters; the dominant delta is the cc3a merge.)
- **Before/after figure: DONE** —
  `pics/24_cathode_merge_before_after_evt298567.png` (this docs dir):
  evt298567, nm4b clusters 46+89 (2402+2408 pts, bottom half stops at
  x ≈ −4 cm, top half starts at x ≈ +2 cm) re-unified into one 4810-pt cluster
  spanning x ∈ [−301, +305] cm. Bee screenshot alternative: same event at
  index 0 of the run-039252 set (item 8), before-view = `img-*` instances
  (pre-T0), after = `clustering-global`.

## 3. Cathode-side non-matched track handling

`qlmatch/20–22` document the non-matched long-track campaigns (nm3/nm4b) and the
wrong-flash-vs-truly-unmatched decomposition. The outline additionally lists
"non-matched tracks at the cathode plane side" as its own improvement.

Needed: is this (a) the cathode containment/demotion work in
`qlport/pdvd-cathode-containment-flash-demotion.md`, (b) part of the xtpc
crosser rescue (`qlmatch/16` §10.6), or (c) a separate rescue not yet
documented? If (c): definition, procedure, and numbers; one example figure.

**Answer (2026-07-18): it is (a) AND (b) — there is no separate (c).** The
nm3/nm4b campaigns (`20–22`) are cathode-agnostic; the repo's cathode-side
treatments are exactly these two, with different adoption status:

- **(a) Cathode containment / flash demotion**
  (`docs/qlport/pdvd-cathode-containment-flash-demotion.md`) addresses
  *wrong-flash* (matched-but-dim) cathode crossers: under the correct bright
  flash one half is pushed past the cathode and fails the hard
  `require_containment` prefilter (tolerance `cathode_ext1`, C++ default
  1.2 cm), deleting that flash as a candidate; the joint match falls back to a
  dim flash. **Adopted as PDVD runner production defaults 2026-07-14** (toolkit
  jsonnet stays null/byte-identical): `cathode_ext1` 1.2 → **2.0 cm**, plus the
  **+13.507 µs** ("2 cm anode pull") extra trigger offset, and the anode
  counterpart `anode_ext1_margin` 1.0 → 2.0 cm (floor −3 → −4 cm). 18-event
  census: ~1152/~2100 (~55%) auto-selected clusters change matched flash under
  the combined trial values (532 toward brighter); anode knob alone: +363
  contained bundles, +28 auto-selected, 107 clusters (5.2%) change flash.
- **(b) xTPC cathode rescue** (`qlmatch/16` §10.6) addresses *truly-unmatched*
  crosser halves that overshoot the cathode: knobs
  `xtpc_cathode_tol`/`xtpc_cathode_qfrac` (C++ default 0 = OFF) grant a
  junk-tolerant provisional containment admission + widened xTPC candidacy,
  purged before the fit unless the pair acquires the crosser scenario flag.
  **Implemented, default-OFF, NOT adopted** (knob-off byte-identical; demo on
  evt298567 pins top-97 + bot-139 onto flash 96; blast radius 8 lost / 6
  gained of 113 selection pairs; census pending).
- Suggested paper treatment: describe (a) as the adopted improvement and (b)
  as an implemented refinement pending validation. Example figure: the
  evt298567 crosser of docs 16/17 (Bee index 0, item 8).

## 4. Cathode X-ARAPUCA absolute PE scale (limitations text)

Docs 03/11 state the cathode PE scale is provisional (no resolved 1-PE feature;
no official DAPHNE gain calibration). For the limitations section: the latest
provisional scale factor in use, the ±25% colleague-amplitude cross-check status
(`qlmatch/10`), and whether an official calibration is expected on any timeline.

**Answer (2026-07-18):**

- **Scale in use.** There is no explicit ADC-per-PE multiplier anywhere in the
  chain: the stored per-channel 1-PE template in raw ADC *is* the calibration
  (pe ≈ raw_area/template_area via OpDecon AutoScale + `spe_area=100`). For the
  cathode channels no 1-PE feature is resolved (threshold-limited continuum;
  the apparent 16–28 ADC mode is a 5σ turn-on artifact), so the **membrane-XA
  1-PE *area* is transferred** to the cathode channels — shape validated
  (held-out residual 2.8% → 1.0% after tail repair), absolute scale
  "±(SoF gain unknown)" (`docs/03_pdvd-spe-template.md` §4–5). A data-driven
  static per-channel scale was fit (657 vetted pairs) but is **not deployed**
  (held-out KS unchanged; in-group relative gains span only ×3.2/×2.5)
  (`docs/11` §2). What *is* deployed (owner-adopted 2026-07-14, from 192
  cathode-crosser anchors, `qlmatch/12`, toolkit
  `protodunevd/qlmatching.jsonnet:197–217`): per-PD-type **effective
  VUV-efficiency scale factors** at unchanged QtoL = 0.094 —
  `eff_scale_cathode = 10.116`, `eff_scale_membrane = 1.655`,
  `eff_scale_pmt = 0.352`. State plainly in the paper that these are effective
  weights (efficiency × gain × model bias), not physical gain measurements.
- **±25% cross-check: PASSED.** Our per-channel 1-PE amplitudes vs a
  colleague's independent SPE set: cathode ratio median **1.07 [0.80, 1.25]**
  (Spearman 0.69); membrane 0.95 once the colleague's 0.6× cathode-only
  prefactor is undone — "unity to ~±25%", the strongest external validation of
  the provisional cathode scale (`qlmatch/10` §2). The alternative cathode
  *area* set implying ×1.88 is disfavored (rank correlation only 0.20).
- **Official calibration: none exists, no timeline.** dunesw v10_21_00d00 has
  no per-channel DAPHNE gain/SPE calibration (flat `SPEArea` only; channel map
  has efficiency fields but no gain fields); the official
  `PhotonCalibratorDUNE.fcl` sets are headed "STILL TO BE OBTAINED. DO NOT
  USE" and span ×16. Sharpened asks were sent to the PDS/DAPHNE consortium;
  no timeline has been given (`docs/11` §2, `qlmatch/10` §1/§3).

## 5. Per-crate trigger offsets (appendix table)

Docs 10 records per-event per-crate charge-window offsets (TDE/BDE up to 32 µs)
resolved via the `trigoff` tree, with beam-flash closure −0.9 µs in 99/120
events. For the appendix: the final constant offsets per crate (or confirmation
they are per-event quantities with no run constant), and any update to the
99/120 closure figure after later fixes (saturation keep-and-mark, ac3).

**Answer (2026-07-18): they are per-event quantities — no per-crate run
constants exist; the closure figure is unchanged.**

- **Per-event, per-crate.** "The charge windows float per event (±15 µs,
  σ 9–12 µs) and per CRATE: the TDE (top CRPs) and BDE (bottom CRPs) windows
  each open on their own 64-sample frame boundary, up to 32 µs apart (≈5 cm of
  drift). So the offset is per event AND per drift volume — a per-run constant
  can never beat ±15 µs" (`docs/10_pdvd-ql-pending.md` §1). Per event the
  chain stamps `offset_bot_us`/`offset_top_us` (≈ −2.1…−2.5 ms) from the
  `trigoff` tree into the opflash metadata (`run_light_evt.sh`), and
  `run_clus_evt.sh` passes them as QLMatching `trigger_offsets=[bot, top]`.
  The per-run *residual* table (`data/ql_trigger_offset.txt`) is **empty**; the
  only run-level constant applied is the production
  `PDVD_QL_EXTRA_OFFSET_US = +13.507 µs` (the 2 cm anode pull, item 3a), common
  to both crates. The *light* window, by contrast, IS a per-run constant
  (trigger-locked ±0.3 µs: 5000.4 / 5000.6 / 2500.5 µs for
  039252/039253/039349). Suggested appendix table: exactly these facts (float
  ranges, 32 µs crate skew, light-window constants, +13.507 µs residual).
- **Closure: 99/120 within ±5 µs, residual median −0.9 µs — final.** Neither
  the saturation keep-and-mark work (`qlmatch/11` §6) nor the ac3 accuracy
  campaign (`qlmatch/23`) re-measured beam-flash closure; no later doc updates
  it. Quote it as is.

## 6. Out-of-sample validation events

All accuracy numbers (85.4% agreement, 78 missed, 118 phantom — `qlmatch/23`)
are in-sample on the 18 hand-scanned events of run 039252. Are any hand-scanned
events available (or plannable) from runs 039253/039349 — even a handful — to
quote an out-of-sample check? If not, the paper states the in-sample caveat
plainly.

**Answer (2026-07-18): none exist — state the in-sample caveat.** Inventory of
all hand-scan records (`work/ql_labels/`: 11 tags; `ql_display/decisions-*`):
every populated tag/decisions set covers run 039252 evt298567–298805 only; the
`cc39349`/`cc39252` label dirs are empty placeholders. **No scan of 039253 or
039349 exists.** Two softeners available to the text:
- An in-sample *blind* consistency check exists: evt298581 was scanned twice
  independently (owner tag `cathxa` vs AI tag `claude-cathxa`); the
  session-recorded bundle agreement was ~94%, but that number is not persisted
  in a doc — cite the label files, or re-run `ql_agree_score.py` on the two
  tags if a quotable number is wanted.
- Out-of-sample scanning is *plannable*: reconstructed outputs for 039349 (10
  events, same instances) exist and the ql_scan viewer supports them.
  **[NEEDS OWNER]** — scanning effort/decision.

## 7. Detector schematic inputs

The paper needs TPC and PDS schematic figures (PDHD analog:
`figs/_make_detector_schematics.py`). Options:
- OK to generate matplotlib/TikZ schematics from the numbers in `docs/07`
  (geometry/FV) and `docs/12` (PD positions/channel map)? (default)
- Or is there an official ProtoDUNE-VD drawing (TDR figure, DocDB) you prefer,
  with permission to reproduce?

**Answer (2026-07-18): the default is fully supported — docs 07 + 12 contain
every number needed.** Key values (WCT frame, cathode at x=0):
- **TPC (`docs/07`):** 8 CRP anode quadrants (idents 0–7, 4 per drift side; 16
  faces, 48 planes, 13,840 wires), W collection plane |x| = 341.55 cm, plane
  stack W/V/U/shield = 341.55/341.23/340.23/339.91 cm (wires v6); cathode
  thickness **6.0 cm** (drift-facing surfaces ±3.0 cm); effective drift
  338.55 cm; wire extent y ±336.4 cm, z ∈ [0, 304] cm rough box; CRM modules
  y=168.5 × z=149.65 cm, 4 rows × 2 columns; FV = cathode-edge ±3.0 cm to
  ±335.835 cm with 15 cm y/z insets. Caveats to respect in the drawing:
  `apa_plane`/`res_plane` are ProtoDUNE-SP template constructs, not physical
  CRP planes (§3/§6).
- **PDS (`docs/12`):** full 40-channel x/y/z table — 8 cathode XAs (ch 4–11,
  x=0), 8 membrane XAs (top x=305.6/229.0; bottom x=−201.1/−277.7, |y|=417.61),
  8 z-wall PMTs (x≈−205.9/−281.7), 16 bottom PMTs (x=−336.47); dead ch
  24/27/28/34, masked ch 32. Caveat: the four dead-channel positions are
  cosmetic estimates mirrored from live neighbours, not surveyed.
- **[NEEDS OWNER]** only if an official TDR/DocDB drawing is preferred instead
  (choice + reproduction permission). No such drawing is referenced in our
  docs tree.

## 8. Bee screenshot event list

Screenshots will be captured from
`http://localhost:8000/set/eec93799-7a6f-4474-8196-688ddbdd91bb/event/0/`.
Planned shots: (a) a fully clustered event colored by cluster, (b) a
cathode-crossing track showing the two half-tracks, (c) a matched result with
per-flash PE overlay / ql_scan view. Please confirm:
- which event indices in this set best show (a)–(c) (e.g. is evt298567 in it?),
- that the set stays live during drafting,
- any preferred camera orientation convention.

**Answer (2026-07-18):**
- **Set identity/mapping.** The UUID cannot be bound from our repo
  (`upload-to-bee.sh` never persists returned UUIDs), but it is a
  `localhost:8000` set — i.e. an upload of `pdvd/upload-combined-run039252.zip`
  (18 events; per-event instances: img-global, img-crp0..7,
  clustering-global, op, channel-deadarea-group0123/4567). Bee index =
  ascending event order: **index i ↔ eventNo 298567 + 14·i**, so
  **evt298567 = index 0 (yes, it is in the set)**, evt298581 = index 1, …,
  evt298805 = index 17. (The 039349 companion zip maps index 0–9 ↔ evt
  19409…19589, step 20.)
- **Recommended indices.** (a) fully clustered, colored by cluster: **index 15
  (evt298777)** — richest event, 18 cathode-spanning clusters — or index 1
  (evt298581). (b) two half-tracks: **index 0 (evt298567)**, the documented
  hero crosser (docs 15/16/17): before-view = `img-*` instances (pre-T0,
  halves separate), after = `clustering-global` (merged); the matplotlib
  counterpart is `pics/24_cathode_merge_before_after_evt298567.png` (item 2).
  (c) matched result with PE overlay: **index 0 (evt298567)** with the `op`
  instance (flashes + Q/L predicted PE), which has full ql_scan documentation
  behind it.
- **Stays live:** cannot be guaranteed from this side — the set lives on the
  drafting machine's local Bee. The source zips
  (`pdvd/upload-combined-run039252.zip`, 54.8 MB;
  `upload-combined-run039349.zip`, 12 MB) remain on disk, so the set can be
  re-uploaded at any time (index↔event mapping is reproducible: ascending
  event order).
- **Camera orientation:** no convention is documented in our docs.
  **[NEEDS OWNER]** if the paper wants a fixed one.

## 9. Citations and front matter

- Author list: currently the four PDHD-paper authors (Qian, Jo, Ning, Zhang) as
  a placeholder — confirm names/order for PDVD.
- Preferred ProtoDUNE-VD detector reference: is there a detector/installation
  paper beyond the VD TDR (arXiv:2312.03130)? A cold-box/CRP performance paper
  to cite for the CRPs?
- PDVD DNN-ROI: is the training documented anywhere citable beyond
  `nf_sp_img_clus/15` (e.g. the DNN-ROI technical paper in preparation covers
  PDVD)?
- The colleague light-reco chain (jjo's waveform files): how to acknowledge —
  co-author, acknowledgment, or internal-note citation?

**Answer (2026-07-18): mostly [NEEDS OWNER]; factual sub-answers below.**
- Author list/order: **[NEEDS OWNER]** — a scientific-authorship decision.
- Detector reference: the **only** ProtoDUNE-VD reference present anywhere in
  our docs tree is the VD TDR (arXiv:2312.03130); no cold-box/CRP performance
  paper is cited. (DocDB 203, cited in docs 07/others, is ProtoDUNE-*SP*
  wire-stack constants — do not use it as a VD reference.) X-ARAPUCA PDE
  papers arXiv:2502.05042 and arXiv:2511.12328 are cited in
  `docs/02_light-reco-plan.md` and are appropriate for the PDS efficiency
  discussion. Whether a better VD installation paper exists is
  **[NEEDS OWNER / literature check]** — nothing further is citable from here.
- DNN-ROI: **nothing citable beyond `nf_sp_img_clus/15`.** That doc records a
  MobileNetV3-UNet (6-channel input: loose_lf, mp2_roi, mp3_roi, tight_lf,
  decon_charge, gauss; TorchScript in `wire-cell-data/dnnroi/pdvd/*.ts`),
  trained on the **bottom drift volume only (anodes 0–3; anodes 4–7 are
  out-of-domain)**; training labels/dataset size are not documented there
  (design notes live in the external `DNN_ROI_SP/docs/`). Whether the DNN-ROI
  technical paper in preparation covers PDVD cannot be verified locally —
  **[NEEDS OWNER]** to ask its authors.
- jjo acknowledgment: **[NEEDS OWNER]** — authorship/acknowledgment decision.

---

When an item is fulfilled, please either extend the relevant numbered doc (and
note it here) or drop files/plots into a location referenced from this file;
the drafting session will pick them up and clear the corresponding markers.

**Fulfillment log (2026-07-18):** all nine items answered inline above by the
analysis session. New artifacts: `pics/24_cathode_merge_before_after_evt298567.png`
(item 2 figure) and the item-2 per-event merge table (computed from
`work/039252_*/mabc-all-apa.zip` vs `_nm4b`, spanning = ≥20 pts/side of x=0).
Owner decisions outstanding: item 6 out-of-sample scan (whether to scan
039349), item 7 official-drawing preference, item 8 camera convention,
item 9 authors/references/acknowledgment.

---

## Drafting-session consumption log (2026-07-18, paper side)

All nine answers were incorporated into the technote (sections rewritten:
post-matching refinements in full; drift-velocity story; limitations;
appendix tables + trigger-offset note; SPE-scale cross-check text; per-type
factors labeled as effective weights). Findings and corrections from the
drafting side:

1. **Item 1 velocity aside is stale — production is v = 1.48073 mm/µs, not
   1.586.** The item-1 answer states "PDVD data reco uses the cathode-pinned
   convention velocity v = 1.586", but `run_clus_evt.sh` has defaulted
   `PDVD_DRIFT_SPEED_*_MMUS` to **1.48073** (the doc-06 golden-crosser W-decon
   measurement) since commit `2aebbfa` (2026-07-14, "superseded 1.53 FR-Argon
   which superseded the 1.586 cathode-pinned convention"), so every production
   result consumed by the paper (cc3a, doc-23 accuracy campaign) ran at
   1.48073. The paper now presents the three-generation story (ensemble 1.568
   → anchored physical 1.566 ± 0.006 → adopted golden-crosser 1.481 ± 0.02)
   and lists the 1.48–1.59 endpoint-definition spread as a limitation.
   ~~[NEEDS OWNER — confirm]~~ **CONFIRMED by owner (2026-07-18):** 1.481 is
   the production value — single-track but "very reliable", geometrically
   consistent with the as-built geometry and between the two TPCs. The paper's
   velocity paragraph now carries the doc-06 internal-consistency checks, and
   a new Appendix B discusses the related empirical 2 cm containment offset
   (pull +13.507 µs + widened cushions; origin unclear, census pending). The
   doc-06 population follow-up remains an A-item.
2. **Item 6:** the `work/ql_labels/` hand-scan records are not on the drafting
   machine, so the evt298581 blind double-scan agreement (~94 %, session
   memory) could not be recomputed locally. The paper states the in-sample
   caveat plainly, plus the "039349 outputs exist, scanning is effort-only"
   softener. Re-running `ql_agree_score.py` on cathxa vs claude-cathxa needs
   the data machine → still open if a quotable number is wanted.
3. **Item 8:** the live local set `eec93799…` is NOT the combined 18-event
   upload — it holds a SINGLE event (bee index 0 = **evt298581**) with four
   instances (`clustering-global` [x uncalibrated, ~1.5e8], `img-global`,
   `img-side-bot/top`) and no `op` instance. A headless Playwright capture
   (bee-video pattern) of the cluster-colored `img-global` view is now the
   paper's clustered-event figure (`figs/bee_clustered_evt298581.png`).
   **[NEEDS OWNER]:** upload `upload-combined-run039252.zip` (with the `op`
   instance) to the drafting machine's Bee and share the set UUID so the
   matched-PE overlay screenshot — the paper's ONE remaining \todofig — can
   be captured.
4. **Item 9:** a literature search (2026-07-18) found no ProtoDUNE-VD
   detector/installation paper beyond the VD TDR (arXiv:2312.03130), matching
   the answer above; the X-ARAPUCA PDE papers (arXiv:2502.05042 →
   `MantheyCorchado:2025ist`, arXiv:2511.12328 → `Botogoske:2025sxz`) are now
   fetched and cited. Author list still **[NEEDS OWNER]**.
5. Camera-orientation (item 8 sub-question) is moot for now: the single
   default reset-camera view was used.
