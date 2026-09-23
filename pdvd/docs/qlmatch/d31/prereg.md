# doc 31 pre-registration (written 2026-09-23 before any Q/L arm ran)

Arms (fresh tags, M13): `q31ctl` = production Q/L on light `_g31off` (knob off, required hash-identical to `_keep`);
`q31tot` = identical except light `_q31tot` (`PDVD_SAT_REPAIR_MODE=tot`, cathode only).  Clustering pin
`/home/xqian/tmp/p100/libpin_p100b`, the pin of `q29flip`.  120 events of `pdvd/stm/events.txt`.

Preconditions (stop and report if either fails):
1. gate g31off: `_g31old` (pre-change libWireCellFlash) vs `_g31off` hash-identical on 120/120 light archives, with the
   null pair `_g31off` vs `_g31offb` identical first;
2. `_g31off` hash-identical to `_keep` on 120/120 (otherwise the pair is also a July-vs-today light delta).
Closure (reported, not a stop): `q31ctl` calib dumps equal to `q29flip`'s.

Eligible population (only flashes with a sat flag on a cathode OpDet 4..11 can change), counted on `q29flip`
(`scripts/d31_scan_subset.py`): 830 of 4062 flashes; judged autos on them 513 (482 agree / 31 phantom) of 775 judged;
scan positives on them 576 (482 covered / 94 missed).  The scan is therefore NOT underpowered for this lever.

Hand-scan rule (doc 23, as used in docs 26-29): `ql_agree_score.py --truth-uid-map-tag keep`, objective tiers, long
tracks.  ToT PASSES the scan if one of agree / phantom / missed improves and neither other worsens, on the whole
18-event sample; the cathode-railed subset is reported alongside (not a separate gate).  Paired movers with a sign
test (`d100r3_scan_pairs.py`) are reported; a PASS carried by movers with sign-test p > 0.2 is stated as "not
separable from churn".

Doc 11 sec 6 numbers (analyze_sat_terms.py / analyze_sat_maskfit.py on the 18 run-039252 events): reported for both
arms.  An extra arm `q31toti` (PDVD_QL_CHI2_SAT_INFLATE=x) runs only if, on q31tot, the selected railed-term inflate
scan's p90 at 0.5 is below 2.0 (i.e. the cap at 4.0 no longer binds); x = the smallest of {0.25, 0.35} whose p90 is
>= 3.0.  Otherwise 0.5 stays.

Doc 12 numbers (fit_qtol_crossers.py): reported for both arms; expected unchanged (anchors exclude sat-flagged
channels).  Any shift > 2 % in a per-type median is reported as a finding, nothing is re-tuned.

No flip is made by this doc; the verdict is a recommendation to the owner.
