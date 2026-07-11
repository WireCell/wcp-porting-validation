# Q/L hand-scan judgment criteria — inferred from the SBND/PDHD human scans

Prepared for the AI-performed hand scan of the first 10 PDVD events (run
039252, idx 0-9). Sources: the saved human labels
(`pdhd/work/ql_labels/labels-evt{983,991,999,1007}.json`,
`sbnd/sbnd_xin/work/ql_labels/{mc,data}/labels-evt*.json`), the viewer docs
(`sbnd_xin/docs/ql-scan-display.md`, `pdhd/docs/ql-scan-display.md`), and the
written cuts in `sbnd_xin/docs/pe-error-study.md` §Selection. Numbers below
come from `../analyze_labels.py` (run it to reproduce).

## 1. What the human labels contain

The scan verdict is binary per flash↔cluster bundle. Three populations:

| set | kept (auto confirmed) | human-added | rejected auto |
|---|---|---|---|
| PDHD data (4 evts) | 59 | 66 | 328 |
| SBND mc (10 evts)  | 94 | 19 | 1 |
| SBND data (10 evts)| 87 | 13 | 0 |

Two very different regimes:

- **SBND** = "load auto-match, confirm, occasionally add": the matcher was
  already well calibrated there, so the scan mostly ratifies it and adds a
  few matches the LASSO dropped.
- **PDHD** = heavy pruning: 328 of 387 auto picks rejected, and *more than
  half* of the final trusted set (66/125) are human-ADDED bundles with
  `strength = 0` — i.e. pairs the LASSO fit had zeroed out but the human
  judged physical. The scan is genuinely corrective, not a rubber stamp.

## 2. What separates kept from rejected (PDHD, the pruning signal)

Quantiles (q05/q50/q95) and pass fractions:

| metric | kept | rejected | comment |
|---|---|---|---|
| pred/meas (`total_pred_light/total_PE`) | 0.06 / **0.76** / 1.53 | 0.00 / **0.05** / 1.07 | **strongest discriminator**: rejects are amplitude-starved — the cluster predicts a few % of the measured light (accidental pairings with a brighter flash) |
| ks_dis | 0.035 / **0.10** / 0.30 | 0.14 / **0.56** / 1.0 | shape agreement; ks<0.2 keeps 78% of kept, only 11% of rejected |
| chi2/ndf | 0.5 / **1.6** / 31 | 1.5 / **8.1** / 148 | separates but overlaps; kept tail to ~30 exists |
| ndf | 39 / **72** / 79 | 0 / **16** / 74 | rejects are few-channel matches — a pattern lit on a handful of PMTs is not trustworthy |
| strength | 0.13 / 0.60 / 0.94 | 0.18 / 0.78 / 0.93 | does NOT separate (rejected even higher) — LASSO amplitude alone is not the human criterion |
| total_PE | 550 / 3274 / 14419 | 52 / **175** / 12822 | rejects concentrate at dim flashes |

2-D occupancy (kept | rejected) shows the human's operating region is
roughly `ks < 0.2` **and** `chi2/ndf < 5`, but neither cut is applied as a
hard ceiling — e.g. 3 kept live at ks 0.2-0.5 with chi2/ndf > 35, and 12
rejected sit inside ks < 0.2. The boundary cases are settled by the *visual*
evidence (geometry + pattern), not the scalars.

Flag behavior — important negative result: `window_truncated` incidence is
HIGHER in kept (0.39) than in rejected (0.12), and `close_to_PMT` is the same
in both (~0.2). So the human did **not** reject matches for carrying these
flags; the pe-error-study "drop window_truncated / close_to_PMT" cut is a
*downstream fit-sample* cut, not a scan-verdict criterion. `potential_bad_match`
is essentially absent from kept sets (0.00) — when the matcher itself flags a
bad match the human agrees.

## 3. The inferred human decision procedure

Reconstructed from the label structure, the viewer's evidence panels, and the
docs' selection rules:

1. **Navigate per flash (coincidence group), seeded by auto-match.** For each
   flash with candidate bundles, judge whether each auto pick is right, and
   whether any non-auto candidate should be added.
2. **Amplitude sanity** — can this cluster's charge plausibly produce this
   flash's light? pred/meas within roughly a factor of a few of 1 is healthy;
   pred/meas ≲ 0.1 with no missing-cluster explanation = accidental ⇒ reject.
3. **Pattern shape** — the per-channel predicted pattern (bars-vs-line panel,
   2-D PE map) must peak on the right detectors: light concentrated where the
   drift-corrected cluster is. KS is the numeric proxy; the eye decides in
   the 0.06-0.4 grey zone.
4. **Geometry consistency at the flash T0** — after the T0 x-shift the
   cluster must sit sensibly in its drift box: endpoints touching
   anode/cathode should be consistent with `at_x_boundary`/track topology
   (a through-going muon must reach the boundaries; a contained blob must
   not poke out). This is the same physics as the matcher's containment
   flag, re-checked by eye.
5. **One flash per cluster; many clusters per flash.** When one flash's light
   needs several clusters, tick several bundles (their predictions sum).
   When one cluster fits several flashes, use the compare table (all
   candidate flashes for that cluster) and keep only the best — time
   proximity of competing flashes and relative ks/amplitude decide.
6. **Add what the LASSO dropped.** A bundle with strength 0 but good
   shape/amplitude/geometry (typically a piece of a multi-cluster flash, or
   the second half of a cathode/anode crosser) is added by hand. This was the
   *majority* of the PDHD trusted set — do not trust the matcher's zero.
7. **Few-channel matches are suspect.** ndf ≲ 20 (light on a handful of
   channels) is rejected unless the geometry is unambiguous.

## 4. PDVD round-1 adaptations

PDVD's calibration state differs from PDHD/SBND (see
`pdvd/docs/pdvd-qlmatching.md` §4a), so the numeric anchors shift:

- **No KS ceiling.** The beam-flash gold pairs (known-true matches) have KS
  median 0.45 — the per-channel PE scale is uncalibrated (x13 spread within
  the cathode-XA group alone). KS is only useful *relatively*: among
  candidates for the same flash/cluster, prefer the lower KS; never reject on
  KS alone.
- **pred/meas is centered near 1 by construction** (QtoL = 0.11 fitted on the
  gold pairs) with ~x3 scatter. Treat pred/meas within [0.1, 10] as
  inconclusive, outside as a strong signal (same logic as PDHD but wider).
- **chi2/ndf inflated** (gold median 9.7 with the pred-based pe_err). Use
  ≳100 as "bad", not 5.
- **at_cathode bundles are weakly constrained in T0** (cathode XAs see both
  volumes, ~89% of the light): a cathode-hugging cluster fits many flashes.
  Judge these mainly through the compare-flash evidence and the y-z overlap
  of pattern vs cluster; keep conservative and mark low confidence.
- **Cathode crossers**: PDVD's joint fit handles them natively (one flash,
  bundles in both volumes). If one volume's half is auto-selected and the
  other isn't, consider adding the partner (people criterion #6).
- **40 channels only** (vs SBND 312 / PDHD 80): ndf is naturally smaller;
  scale criterion #7 down (suspect below ~5-8 informative channels).
- `window_truncated` / `close_to_PMT` / `consistent=false` are recorded but
  are NOT rejection grounds, matching the human's usage.

## 5. Verdict vocabulary used by the AI scan

Per bundle: `keep` (auto pick confirmed), `reject` (auto pick removed →
`rejected_auto`), `add` (non-auto bundle promoted into `matches`), each with
`confidence` high/med/low and a one-line reason. Low-confidence and
at_cathode decisions are the ones the human reviewer should re-check first.
