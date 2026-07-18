# 24 — Information and figure requests for the PDVD Q/L matching technote

Status: OPEN — written by the paper-drafting session (2026-07-18).
Audience: analysis author. Each item lists what the technote needs, why, and the
concrete deliverable (a paragraph of facts, a number, or a plot). Items marked
**[BLOCKING]** gate a section that currently carries `\todofig` /
`[TO BE COMPLETED]` markers; the rest refine limitations text or the appendix.

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

## 3. Cathode-side non-matched track handling

`qlmatch/20–22` document the non-matched long-track campaigns (nm3/nm4b) and the
wrong-flash-vs-truly-unmatched decomposition. The outline additionally lists
"non-matched tracks at the cathode plane side" as its own improvement.

Needed: is this (a) the cathode containment/demotion work in
`qlport/pdvd-cathode-containment-flash-demotion.md`, (b) part of the xtpc
crosser rescue (`qlmatch/16` §10.6), or (c) a separate rescue not yet
documented? If (c): definition, procedure, and numbers; one example figure.

## 4. Cathode X-ARAPUCA absolute PE scale (limitations text)

Docs 03/11 state the cathode PE scale is provisional (no resolved 1-PE feature;
no official DAPHNE gain calibration). For the limitations section: the latest
provisional scale factor in use, the ±25% colleague-amplitude cross-check status
(`qlmatch/10`), and whether an official calibration is expected on any timeline.

## 5. Per-crate trigger offsets (appendix table)

Docs 10 records per-event per-crate charge-window offsets (TDE/BDE up to 32 µs)
resolved via the `trigoff` tree, with beam-flash closure −0.9 µs in 99/120
events. For the appendix: the final constant offsets per crate (or confirmation
they are per-event quantities with no run constant), and any update to the
99/120 closure figure after later fixes (saturation keep-and-mark, ac3).

## 6. Out-of-sample validation events

All accuracy numbers (85.4% agreement, 78 missed, 118 phantom — `qlmatch/23`)
are in-sample on the 18 hand-scanned events of run 039252. Are any hand-scanned
events available (or plannable) from runs 039253/039349 — even a handful — to
quote an out-of-sample check? If not, the paper states the in-sample caveat
plainly.

## 7. Detector schematic inputs

The paper needs TPC and PDS schematic figures (PDHD analog:
`figs/_make_detector_schematics.py`). Options:
- OK to generate matplotlib/TikZ schematics from the numbers in `docs/07`
  (geometry/FV) and `docs/12` (PD positions/channel map)? (default)
- Or is there an official ProtoDUNE-VD drawing (TDR figure, DocDB) you prefer,
  with permission to reproduce?

## 8. Bee screenshot event list

Screenshots will be captured from
`http://localhost:8000/set/eec93799-7a6f-4474-8196-688ddbdd91bb/event/0/`.
Planned shots: (a) a fully clustered event colored by cluster, (b) a
cathode-crossing track showing the two half-tracks, (c) a matched result with
per-flash PE overlay / ql_scan view. Please confirm:
- which event indices in this set best show (a)–(c) (e.g. is evt298567 in it?),
- that the set stays live during drafting,
- any preferred camera orientation convention.

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

---

When an item is fulfilled, please either extend the relevant numbered doc (and
note it here) or drop files/plots into a location referenced from this file;
the drafting session will pick them up and clear the corresponding markers.
