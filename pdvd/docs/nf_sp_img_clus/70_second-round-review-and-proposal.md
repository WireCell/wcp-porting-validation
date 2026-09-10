# 70 — Second-round review of `CheckSTM_Michel`: the owner's four questions, and the next proposal set

**Update 4 (2026-09-10): P3b and P2 are built (toolkit `299d8bc4`) and gated, each in its own doc: pdvd/72 (P3b) and pdvd/73 (P2).**

**P3b is in PDVD production** (`moved_stop_michel_kink_min: 60.0`, doc 72):
- the moved-stop veto spares the owner-confirmed Michel `039349_48/21` (132.6°);
- `michel_found` 133 / 12 / 25 → **134 / 12 / 24**;
- nothing else moves, `is_stm` included (225 / 7 / 51); 90° is bit-identical.

**P2 stays OFF** (doc 73):
- none of the three sub-knobs adds a judged `michel_found` TP, as predicted;
- each attaches an arm the owner tagged delta / other or muon;
- (b) also loses the stopper `039253_13/73`: its new 2.4 cm seed fails P1's 3 cm test.

**Corrections to §4.2 and §6** (marked in place):
- the 60° cut spares one Michel, not two;
- the through-going kinks were survey-arm numbers;
- the kink window is 15 cm, so (c) is a *shorter* window;
- on production, doc 70's four named P2 items are out of reach at the argued points.

**Update 3 (2026-09-10): P4 is built (toolkit `d227d5b8`), gated and in PDVD
production. It has its own doc, pdvd/71; from P4 on each proposal carries its own md file, and P1
stays here in §9–§10.** P4 (`michel_gamma_collect`) adds the Michel's isolated
gamma blobs to the Michel object as role-4 members with their own energy
(`michel_ke_gamma`, `michel_ke_total`). It follows the owner's three criteria:
along the Michel direction, a dot near the stop, and an energy guard against
over-clustering. At 35 cm:
- the owner's gamma tags inside the Michel object go from 9 to 75 of 159;
- 0.958 of the tagged blobs it takes are gamma or Michel;
- no pre-existing branch, zip or calib moves.

Then, on the owner's call (X0 = 14 cm, a gamma's conversion length ~18 cm, 3 ×
18 ≈ 54 cm), the radius is **50 cm** (doc 71 §11). That widens admission.
Production now reads `is_stm` **225 / 7 / 51**; the three items below move
the way the record says, and `michel_found` is unchanged. Gamma tags in the
object reach 104 of 159 at purity 0.933.

**Correction to §10.4:** `d71vsp` ran with the survey, i.e. 60 cm companion
admission, and production does not. The first bare-production census on the
record reads `is_stm` **223 / 8 / 53**, not 225 / 7 / 51. Three named items
move with the admission radius (doc 71 §7).

**Update 2 (2026-09-10, §10): P1 is built and gated, knob default OFF
(toolkit `f66a8b9f`).** Knob off: byte-identical on PDVD (120 events) and
PDHD (61) against the merged binary. Knob on: the C++ matches the offline
rule item for item, `is_stm` 197 / 7 / 79 → **221 / 7 / 55**, `michel_found`
identical. **On the owner's go ("flip with sparse please"), both
`topology_stop_evidence` and `topology_clears_sparse` are now PDVD
production** (§10.4): `is_stm` 197 / 7 / 79 → **225 / 7 / 51** on the updated
record (purity 0.970, efficiency 0.815, F1 0.886), 0 new false positives.
PDHD stays OFF.

**Update (2026-09-10, after the owner's `smx4` scan, §9): P1 is cleared for
implementation.** The owner judged, blind to P1, all 54 Michel-carrying items
production rejects. At the argued point (10 MeV / 3 cm) P1 fires on 25 of them:
**24 are stoppers, 0 through-going, 1 MESSY**. On the grading record with the
owner's verdicts folded in (`pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json`),
`is_stm` goes 197 / 7 / 79 → **221 / 7 / 55** (purity 0.966 → 0.969, efficiency
0.714 → 0.801, F1 0.821 → 0.877). All three record-THRU items P1 admits are
stoppers; 8 of the 13 record-THRU items the owner judged are. **Correction to
§3.3:** its first table counted 5 items that keep a reject bit P1 does not clear
(`profile_sparse` ×4, `continuation` ×1). The exact first sizing was +22 TP /
+3 FP, not +26 / +4, and its "F1 0.807 → 0.88" compared a union baseline with a
payload result. §3.3 now carries the exact numbers; §9.5 is the implementation
plan.

**Status (2026-09-10). Review and proposal — no C++, no config touched; the
owner's `smx4` labels and a new merged record were added in §9.** After the doc 56–68 campaign (11 T-series flips in the 25-key
`stm_michel_knobs` bag; `is_stm` 144 / 9 / 125 → 197 / 7 / 87, `michel_found`
F1 0.735 → 0.863) and the owner's `smx3` scan, the owner asked four questions:
(1) is the 0.6 cm trajectory step right for PDVD's strip pitch; (2) the stop
rests on two kinds of evidence — topology (a track turn, a distinct arm) and
the dQ/dx rise — and the rise should only count when it genuinely matches a
Bragg peak, whereas topology alone is very strong; (3) clear Michel
topologies the chain did not identify; (4) isolated gamma blobs that belong to
the Michel and are not clustered with it. This doc answers each against the
code as it stands and the merged scan record, and proposes seven items (§6),
each a default-OFF knob or a config route, each with its sizing on today's
production arm and the gate it would have to clear.

**The one number that reorganises the next round (§3):** of the 75 stoppers
production still misses, **40 already carry a Michel the chain found and
attached at the stop** — the stopper was then rejected on the dQ/dx shape
tests alone. The verdict has 18 reject sites and not one topology test, while
the scan rubric (doc 55) says "a Michel topology is sufficient on its own". A
topology-first verdict recovers 22 of the 40 for 3 possible false positives at
the argued operating point (F1 0.828 → 0.874 on the payload population;
corrected from "26 for 4, F1 0.807 → ~0.88", §3.3), and those 3 plus 11 more
were a blind re-judge for the owner, not a tuning question. The owner found no
false positive among them (§9).

Companion docs: pdvd/56 (the task set), 62–66 (T3–T8), 67–68 (the owner's
operating points and scan), 54 (the two stop defects), 65 §3–§4 (sampling),
pdhd/13 §5 (D3: no arm at the stop).

---

## 0. Repro

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img
export STM_SCAN_RECORD=$PWD/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_verdicts.json   # merged record, doc 68 sec 2

# the census this doc starts from (production = the anchor arm d68a3, today's 25-key bag)
python3 pdhd/stm_michel_scan/census_score.py --prep /home/xqian/tmp/d68/prep_d68a3 --arm d68a3 \
        --baseline /home/xqian/tmp/d67/prep_d67v                    # -> /home/xqian/tmp/d68/score_merged_d68a3.txt

# every table in sec 3-5: the four is_stm cells split by michel_found, the P1 grid, the
# 12 found-stopper Michel misses with the classifier's own per-arm line, the gamma-tag roles
python3 pdvd/docs/nf_sp_img_clus/scripts/d70_sizing.py --prep /home/xqian/tmp/d68/prep_d68a3 \
        --arm d68a3 --out /home/xqian/tmp/d70r2                     # -> /home/xqian/tmp/d70r2/sizing.txt
# (first version, before the P1 gate's residual-bit test: /home/xqian/tmp/d70/sizing.txt)

# the record is untouched
python3 pdhd/stm_michel_scan/census_score.py --check                # "0 of 14 differ"

# sec 9: P1's blind re-judge smx4 -- built, served on :5017, scored
python3 pdvd/docs/nf_sp_img_clus/scripts/d70_build_smx4.py --cells /home/xqian/tmp/d70/d70_cells.json \
        --prep /home/xqian/tmp/d68/prep_d68a3 --outprep pdhd/stm_michel_scan/prep-pdvd-smx4 \
        --sheet pdvd/docs/scan/pdvd_stm_michel_smx4_sheet.tsv \
        --questions pdvd/docs/scan/pdvd_stm_michel_smx4_questions.json \
        --key pdvd/docs/scan/pdvd_stm_michel_smx4_key.tsv          # refuses to rebuild an existing set
(cd pdhd/stm_michel_scan && ./serve_stm_michel_scan.sh 5017 --det pdvd --scan-tag smx4 \
        --manifest $PWD/../../pdvd/docs/scan/pdvd_stm_michel_smx4_sheet.tsv --prepdir $PWD/prep-pdvd-smx4 \
        --questions $PWD/../../pdvd/docs/scan/pdvd_stm_michel_smx4_questions.json)
cp -p pdvd/work/stm_michel_labels/smx4/labels.json pdvd/docs/scan/pdvd_stm_michel_smx4_labels.json
python3 pdvd/docs/nf_sp_img_clus/scripts/d70_score_smx4.py --labels pdvd/docs/scan/pdvd_stm_michel_smx4_labels.json \
        --key pdvd/docs/scan/pdvd_stm_michel_smx4_key.tsv --prep /home/xqian/tmp/d68/prep_d68a3 \
        --write-merged pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json   # -> /home/xqian/tmp/d70/score_smx4.txt
STM_SCAN_RECORD=$PWD/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json \
    python3 pdhd/stm_michel_scan/census_score.py --prep /home/xqian/tmp/d68/prep_d68a3 --arm d68a3 \
        --baseline /home/xqian/tmp/d67/prep_d67v                    # -> /home/xqian/tmp/d70/score_merged4_d68a3.txt

# sec 10: P1 built (toolkit f66a8b9f) and gated.  Two pins, both full copies of local/lib:
#   /home/xqian/tmp/d71/libpin_base = the merged 98140fee build (libWireCellClus e5cd0364be81)
#   /home/xqian/tmp/d71/libpin_p1   = f66a8b9f                  (libWireCellClus 0d7027b23351)
(cd /nfs/data/1/xqian/toolkit-dev/toolkit && wcbuild; wcbuild; ./build/clus/wcdoctest-clus)   # 360/360
#   (the first wcbuild fails to link the doctest against the OLD installed lib -- the
#    new-symbol trap; the library itself installs, and the second pass links)
pdvd/docs/nf_sp_img_clus/scripts/d70_p1_arms.sh    # d71vleg d71hleg (base) | d71voff d71hoff (P1, off) | d71vp1 d71vsp (P1 on)
pdvd/docs/nf_sp_img_clus/scripts/d70_p1_gates.sh   # -> /home/xqian/tmp/d71/gates.log: OFF gates, prep, rule check, census
```

The arm's per-event logs (`pdvd/work/<evt>_d68a3/wct_pr_<evt>.log`) carry the
`CheckSTM_Michel stop-arm:` DEBUG line doc 63 added; `d70_sizing.py` reads it
for §4. All line numbers below are from `clus/src/CheckSTM_Michel.cxx`,
`clus/src/StmMichelFunctions.cxx` and `clus/src/TrackFitting.cxx` as of toolkit
`HEAD` today (docs 51/54/62 and pdhd/13/15 cite older revisions).

---

## 1. Where we stand

**What shipped (PDVD production, `pdvd/wct-pr-perevt.jsonnet` `stm_michel_knobs`).**
The bag has 25 keys; eleven are the T-series flips of docs 57–68:

| knob | doc | what it bought (on the record it was graded on) |
|---|---|---|
| `stop_retreat_max: 2` | 57 (T1a) | +2 stoppers, 0 cost; pin residual 4.60 → 3.53 cm |
| `stop_split_max: 1` | 58 (T1c) | +1, 0 cost; first knob that mutates the PR graph |
| `stm_kink_asym_enable` (tagger) | 59 (T1b) | +2, 0 cost; a hot-into-cold kink is now a kink |
| `moved_stop_michel_guard` | 61 (T2c) | michel −5 FP / −1 TP |
| `stop_local_michel_pieces` | 62 (T3b) | michel +17 TP / +4 FP — the disconnected same-cluster pieces |
| `michel_range_energy_guard` (+ `dis_cm 3.0`, doc 67) | 62 (T3c) | michel −21 FP / −3 TP, then −5 FP more at 3 cm |
| `absorb_bragg_stub` | 63 (T5) | +3 stoppers, 0 cost |
| `ks_margin −0.02`, `compare_range_cm 45` | 67 | +26 / −1 stoppers, −3 FP, 0 new |
| `bragg_peak_anchor` + `bragg_peak_search_cm 3.0` | 68 | +28 / −7 stoppers, 3 FPs swapped for 3 |

Left OFF with a measured reason: `stop_local_residual_cm` (doc 62, dropped by
the owner), `publish_other_arms` (doc 64, rows only), `profile_geometry_guard`
(doc 66, costs 5 TPs), `dx_norm_length` 4 mm (doc 68, worse on top of the
anchor and a shared file).

**The census today** (`census_score.py` on the merged record, 548 of 580
judged items have a payload on `d68a3`):

| | TP | FP | FN | TN | purity | efficiency |
|---|---:|---:|---:|---:|---:|---:|
| `is_stm` | 197 | 7 | 75 | 269 | 0.966 | 0.724 |
| `michel_found` | 132 | 17 | 25 | 374 | 0.886 | 0.841 |

On the 580-item union (an item with no candidate counts as 0) the FN columns
read 87 and 32 — same arm, larger denominator (doc 68 §3). (§9.3 updates
these on the record with the owner's `smx4` verdicts: `is_stm` 197 / 7 / 79,
`michel_found` 133 / 12 / 25.) Failure classes:
A 7, B 75, C 17, D 25, F 3, G 15, H 20, K 34, L_plateau 46, L_sparse 26. The
C1 stop table says where the 75 missed stoppers sit: collapse 24 / rise 35 /
flat 16 by shape; sentinel 35 / kink 40 by anchor.

**What the `smx3` scan taught** (doc 68 §2): on the 17 items the owner
re-judged, smx1a was overridden on 6, and four of those were stoppers smx1a had
called THRU — two at high confidence. The three "new false positives" of the
anchor were all stoppers when the owner looked. smx1a was scanned with the
reconstruction visible; its one-directional under-call of stoppers is now
measured, not suspected. That fact drives §3's recommendation: where a
proposal's cost is a set of THRU items that carry a reconstructed Michel, the
cost is a scan question first.

---

## 2. Question 1 — the trajectory fit and the 0.6 cm step

### 2.1 What the step is, and what it is not

* The production step is **`low_dis_limit / 2` = 0.6 cm**, derived, not a key:
  pass 1 (`organize_orig_path`, `TrackFitting.cxx:10049`) runs at 1.2 cm,
  passes 2 and 3 (`organize_ps_path`, `:10134`, `:10162`) at the halved value
  (`:10103`); `do_multiple_tracking` mirrors it (`:9343`, `:9536`, `:9729`).
  The resampling rule (`:2841-2868`) drops a gap under 0.8× the step, keeps one
  under 1.6×, subdivides above — 0.48 / 0.96 cm bands at 0.6 cm.
  (`clus/docs/patternrecognition/do_multi_tracking_review.md:311-321` still
  calls the third pass "hardcoded 0.6 cm"; that has been knob-derived since doc
  61 §9.)
* The four sampling keys (`low_dis_limit` 12, `end_point_limit` 6,
  `dx_norm_length` 6, `div_sigma` 6 mm) are **bit-equal** on PDVD, PDHD, SBND
  and the C++ uBooNE preset (`TrackFittingPresets.h:71-88`). The sampling was
  inherited from uBooNE and never re-tuned for any detector.
* Pitch, computed from the shipped wire files: PDVD U/V **7.650 mm**, W
  **5.100 mm**, uniform over all 48 planes; PDHD 4.669 / 4.669 / 4.792; SBND
  3.000. Drift step (4 ticks × 0.5 µs × drift speed): PDVD 2.96 mm, PDHD 3.15,
  SBND 3.13.

| detector | step / pitch (U, V, W) | step / drift step |
|---|---|---:|
| PDVD | **0.78 / 0.78 / 1.18** | 2.03 |
| PDHD | 1.29 / 1.29 / 1.25 | 1.90 |
| SBND | 2.00 / 2.00 / 2.00 | 1.92 |

The step is two drift steps everywhere and anything from 0.78 to 2.0 pitches —
the same inversion doc 34 documented for the ctpc radius. PDVD U/V is the only
case where the step is **shorter** than a pitch: two adjacent fit points can
share the same U and V strip and differ only in W and drift.

* `dx_norm_length` normalises the local point spacing `dx[i]` (which *is* the
  step, `:8611-8619`) inside the regularisation matrix (`:9101-9112`; multi
  `:8235`). With both at 6 mm the smoothness penalty is scale-neutral **by
  coincidence of defaults**; doc 65 §4 moved them independently, which is why
  `dx_norm_length` 4 mm behaves as "more smoothing", not "finer sampling".
* The pitch-sensitive constant in the fit is not the step but the **end
  trim**: `examine_end_ps_vec` (`:2562`; back loop `:2699`) pops points that
  fail `is_good_point` at a fixed **0.2 cm** on three planes, with
  `good_point_pitch_frac` 0 on PDVD. On the exact drift × pitch lattice a
  0.2 cm test cannot be met for a band of phases when the half-pitch is
  0.38 cm (doc 34 :90-93); doc 32 §11.1 measured the three-plane pass rate at
  0.2 cm as 0.175 on PDVD against 0.624 on SBND, reaching 0.715 only at 0.4 cm.
  Production compensates with `ctpc_aniso_metric` (ON), not with the radius.

### 2.2 What the measurements say

Doc 65 §3–§4 already measured this on the record:

* The plateau dQ/dx autocorrelation is positive at lag 1 only (0.42 PDVD,
  0.60 PDHD) and below 1/e from 1.2 cm — the fit's dQ/dx is smoothed over
  about two 0.6 cm samples on PDVD. The 0.5–3 cm tail median therefore
  averages ~4 points of which ~2 are independent.
* Moving `low_dis_limit` by ±4 mm changed **which clusters the tagger flags at
  all** (167 / 136 of 569 items unmatched) and flipped `is_stm` on 68 / 63 of
  the rest in both directions — the step is a parameter of the whole STM
  tagging, and this record cannot grade it.
* `dx_norm_length` 4 mm, the one sampling change the record mildly preferred
  (michel F1 0.863 → 0.875), was scanned (doc 68 §4) and is **worse on top of
  the anchor** (`is_stm` F1 0.807 → 0.792): the 4 mm smoothing moves the peak
  the 3 cm anchor reads.

### 2.3 Assessment and proposal (P5, low priority)

The step is not what limits the stopper verdict. At 0.78 pitch the PDVD
profile is already sampled finer than its strips resolve; the profile is
smoother than its sampling; and the shape tests were shown (doc 65 §2) to be
limited by the KS margin, not by window widths. The pitch-relevant defect is
the 0.2 cm end-trim radius on the lattice, which the anisotropic metric
already addresses. **Recommendation: keep 0.6 cm.** A pitch-scaled step
(`low_dis_limit = 1.5 × pitch` ≈ 1.15 cm on PDVD, i.e. a 0.57 cm step) is
today's value to within 5 %.

What *is* worth building, because every later sampling question runs into it:
**P5 — a `pr.jsonnet` parameter `stm_trackfitting_config_file`** (default = the
shared `trackfitting_config_file`, so the compiled config is byte-identical
when unset) that lets `TaggerCheckSTM` and `CheckSTM_Michel` carry their own
fit file. Today one file reaches three components (`protodunevd/pr.jsonnet:175`,
`:1550`, `:1695`, `:1853`), and doc 68 §4 could not flip `dx_norm_length`
alone because `TaggerCheckNeutrino` reads the same file. Any change that moves
the candidate set still needs a new scan; P5 only makes such a change scopeable.

---

## 3. Question 2 — two kinds of evidence for the stop

### 3.1 Where topology enters today, and where it does not

The stop point and the stopper verdict are decided separately:

* **The stop** is the tagger's kink row: `find_first_kink`
  (`TaggerCheckSTM.cxx:1770-1870`, with T1b's asymmetric clause ON on PDVD)
  → `read_stm_anchor` (`CheckSTM_Michel.cxx:1009`, `:1022-1026`), which clamps
  both no-kink sentinels to the fit's last row. Then three dQ/dx-gated
  corrections: extend (`:1564-1599`, a *small* angle means "keep walking"),
  T1a retreat (`:1601-1640`, no angle term), T1c split (`:1642-1699`, a bend of
  ≥ 15° is necessary but never sufficient). All three are skipped once
  `bragg_confirmed(chain)` holds (`:1554-1563`, pure dQ/dx).
* **The verdict** (`:1719-2790`) has **18 `reject_bits |=` sites and zero
  turn tests**: anchor `:1764-1803`, contrast `:1806-1817`, KS `:1836-1850`,
  template PID `:1861-1868`, geometry `:1883-1914`, arms `:1917-1992` (the only
  angle-bearing bit, `R_CONTINUATION` `:1977`, is a *collinearity* veto),
  coverage `:2760-2775`, fiducial `:2779-2781`; `is_stm = (reject_bits == 0)`
  at `:2790`. Michel presence is **deliberately excluded** from `is_stm`
  (`StmMichelFunctions.h:389-391`, doc pdhd/03: "the flag must not qualify
  through the daughter").

The scan rubric the census grades against says the opposite (doc 55 :129): *a
Michel topology is sufficient on its own*; THRU means *no Bragg rise at the fit
end* and *a ragged rise still counts*. So a stopper whose Michel dilutes its
last centimetres — doc 55 §15.2's mechanism, doc 68 §3's "anchor-lost" family —
is a stopper to the scanner and a `shape_flat` to the chain.

On the current arm, kink-anchored stops are 95 of 197 found stoppers (48 %)
against 26 of 269 correctly rejected through-goers (10 %): topology at the
tagger is a strong discriminator, but it fixes the stop on fewer than half the
stoppers, and the anchor flip did not change that ratio. The anchor moved the
collapse-shaped cell only (stopper/found collapse 14 → 38) and left the 34-pin
stop residual byte-identical: it moved the *window*, not the stop.

### 3.2 The four cells, split by whether a Michel was found

`d70_sizing.py` §A, arm `d68a3`, 548 items:

| cell | n | scan says Michel | **`michel_found`** | stop arm > 0 | `bragg_valid` |
|---|---:|---:|---:|---:|---:|
| TP | 197 | 105 | 95 | 96 | 197 |
| FP | 7 | 0 | **0** | 0 | 7 |
| FN | 75 | 52 | **40** | 44 | 69 |
| TN | 269 | 0 | **14** | 36 | 257 |

**Forty of the 75 missed stoppers already have a Michel the chain found and
attached.** Their reject bits are the two shape tests and nothing else in 30
of 40 (`shape_flat` 16, `no_bragg,shape_flat` 14), `profile_sparse` in 5,
boundary in 2, `continuation` in 1. None of the 7 false positives has a Michel
or a stop arm. Fourteen THRU items also carry a `michel_found = 1` — the
population a topology-first rule would admit.

The 40 and the 14 differ in Michel quality (§B of the script):

| quantile 10 / 50 / 90 | TP with Michel (95) | FN with Michel (40) | TN with Michel (14) |
|---|---|---|---|
| `michel_ke_best` (MeV) | 9.5 / 22.5 / 42.5 | 5.9 / 18.6 / 37.5 | 1.8 / 6.5 / 26.5 |
| `michel_len` (cm) | 2.4 / 7.5 / 13.6 | 2.2 / 5.4 / 13.5 | 0.9 / 2.4 / 8.1 |
| `conn_type` 1 : 2 | 79 : 16 | 29 : 11 | 10 : 4 |

The Michel-carrying missed stoppers look like the Michel-carrying found ones;
the Michel-carrying through-goers are mostly sub-3 cm, sub-7 MeV objects.

### 3.3 Proposal P1 — `topology_stop_evidence`

**Rule.** After `michel_found` is derived (`:2728`) and before `is_stm`
(`:2790`): if the candidate carries a Michel of sufficient quality —
`michel_conn_type ∈ {1, 2}`, `michel_ke_best ≥ topology_michel_ke_min`,
`michel_len ≥ topology_michel_len_min_cm`, and the stop is inside the fiducial
inset — clear **`R_NO_BRAGG` and `R_SHAPE_FLAT` only**. Every other bit stays:
`plateau_off_mip`, `profile_sparse`, `continuation`, `vertex_hadron`,
`stop_near_boundary`, `cluster_not_track`, `stop_into_dead`, `not_muon_pid`,
`profile_geometry`. This is the owner's rule written into the verdict:
topology (a Michel at the end *is* the track turning into a distinct, cooler
arm) is sufficient; the rise is required only when topology is absent. Knob
default OFF; `michel_found` is bit-identical by construction (the rule reads
it, never writes it); `is_stm` moves only 0 → 1.

**Sizing** (exact offline re-verdict — both bits are single-consumer, doc 67
confirmed that route item for item on real arms — and an item that keeps any
other reject bit stays rejected). *Corrected 2026-09-10:* the first version of
this table left out that last test and counted five items P1 cannot move:
`039252_3/45`, `039349_41/51`, `039349_61/21`, `039349_82/54` (`profile_sparse`)
and `039349_5/65` (`continuation`). At the argued point it read +26 / +4. Exact,
on the doc 68 record (`d70_sizing.py` §B, payload population, 548 items):

| KE ≥ (MeV) | len ≥ (cm) | FN → TP | TN → FP | `is_stm` TP / FP | purity | efficiency |
|---:|---:|---:|---:|---|---:|---:|
| 5 | 0 | 29 | 5 | 226 / 12 | 0.950 | 0.831 |
| 5 | 3 | 24 | 3 | 221 / 10 | 0.957 | 0.812 |
| 8 | 3 | 22 | 3 | 219 / 10 | 0.956 | 0.805 |
| **10** | **3** | **22** | **3** | **219 / 10** | **0.956** | **0.805** |
| 12 | 3 | 20 | 3 | 217 / 10 | 0.956 | 0.798 |
| 15 | 3 | 17 | 2 | 214 / 9 | 0.960 | 0.787 |

Today: 197 / 7, purity 0.966, efficiency 0.724, F1 0.828 on this payload
population (0.807 on the 580-item union, where an item with no candidate counts
as 0; the first version of this paragraph compared the two). At the argued
point (10 MeV = the T2c/T3c floor already in the bag, 3 cm = the range-energy
distance) F1 is 0.874 (union 0.854). Purity falls from 0.966 to 0.956 **if the
three admitted THRU items are really THRU** — and that was the question, not a
tuning knob:

* admitted: `039349_39/56` (smx1a medium, 29.0 MeV, 7.9 cm), `039349_45/25`
  (FRAG_THRU, medium, 27.2 MeV, 8.4 cm), `039349_81/51` (medium, 12.9 MeV,
  4.8 cm, `conn_type` 2). (`039252_3/45`, low confidence, was on the first list;
  it keeps `profile_sparse`.) None was in the owner's `smx3` re-judge; three
  Michel-carrying "THRU" items in that re-judge turned out to be stoppers.
* not recovered at 10 MeV / 3 cm (18): two on `stop_near_boundary`
  (`039252_2/103`, `039349_27/41` — correctly left to the fiducial rule), six
  that keep `profile_sparse` or `continuation` (the five above plus
  `039253_6/90`, `039349_54/56`; `039252_3/45` is a TN), and ten short or faint
  Michels (1.3–4.2 cm, 3–14 MeV) including the owner's `039253_12/93` ("not
  fully identified", 4.6 MeV, 2.6 cm) and `039349_36/46` (4.6 MeV, 1.8 cm).

**Scan need — `smx4` group A: DONE (§9).** The 14 TN-with-Michel and the 40
FN-with-Michel (54 items), served blind (P1's answer and the record's verdict
not shown). The owner called all three admitted items stoppers (and
`039252_3/45` too). On the updated record P1 at 10 MeV / 3 cm costs 0 false
positives for 24 stoppers.

### 3.4 P1b — the Bragg match on the other side

For items *without* topology the owner's rule asks the rise to genuinely
match a Bragg peak. Today that is `contrast ≥ 0.6 × expected` plus the KS
margin on the anchored profile. Doc 65 §5.1's unbuilt refinement — the anchor
re-origins the profile only when the anchored peak row exceeds the plateau by a
factor (a "rise precondition"), otherwise the geometric origin stands — targets
the 7 stoppers the 3 cm anchor lost (doc 68 §3). Three of those seven
(`039253_13/73`, `039253_3/66`, `039349_36/46`) carry a Michel and are inside
P1's 40; the two at ≥ 10 MeV are recovered by P1 (§9). The other four
(`039253_0/44` — `michel_found` 0, the owner's "Michel not identified" —,
`039253_6/85`, `039349_15/23`, `039349_76/75`) are P1b's real target.
(Corrected: the first version put `039253_0/44` among the Michel-carrying
items.) Size it offline from the payloads' `profile` before any C++ (the anchor
is exactly reproducible offline, doc 65 §2.2); build it only if it recovers
≥ 2 of the 4 without a new FP.

---

## 4. Question 3 — Michels the chain did not identify

### 4.1 The population

`michel_found` misses 32 scan-Michel items on the union. Seven have **no
candidate on the arm** at all (`039252_1/109`, `039253_3/60`, `039253_7/30`,
`039253_8/65`, `039349_33/60`, `039349_7/20`, `039349_81/62` — the tagger's
domain, doc 68 §4's leads, including the owner's "the track did not go to the
end … seems to have a Michel"). Thirteen are missed stoppers as well — their
Michel search never got a correct stop to search from (§3 recovers the
stopper; the Michel needs the stop). **Twelve are found stoppers whose Michel
was missed**; `d70_sizing.py` §C traces each with the scan's michel-tagged
segments matched into the arm and the classifier's own line:

| mechanism | items | what the arm shows |
|---|---|---|
| **The Michel is inside the muon chain** (role 1, the fit runs straight through: offline kink 175–178°) | `039252_16/32` (32007, 3.4 cm, 0.17 MIP), `039253_3/61` (61008, 2.4 cm, 0.27 MIP), `039349_60/40` (40002, 6.8 cm, 1.35 MIP), `039349_64/65` (65003, 3.8 cm, 1.47 MIP) | `n_stop_arms = 0`; because `bragg_confirmed(chain)` is already true on a found stopper, T1a/T1c never run (`:1617`, `:1654`) — the collinear tail cannot be split off |
| **An arm exists at the stop and fails a Michel gate** (`kOther`) | `039349_11/19` (19004: `len 8.74 far_len 54.37 mip 0.22 kink 81.1 shower 1` — fails `mip > 0.3` AND `len + far_len ≤ 25`), `039349_69/56` (56006, 13.6 cm, 0.29 MIP, kink 75°), `039253_3/61` (61007: `len 7.80 far_len 7.08 mip 0.62 kink 18.0` — the C++ 5 cm direction window reads 18° where the offline 5 cm chord reads 40°; below 20° yet too cool for a continuation, so `kOther`) | the diluted PDVD Michel sits under `michel_mip_lo` 0.3; a Michel with its own brems subtree exceeds `michel_max_len_cm` through `far_len`; the kink window is shorter than the turn |
| **A Michel was found, then vetoed** | `039252_2/79` (79002: `kind 1 len 9.26 mip 0.43 kink 59.6`, role 3, `n_michel_veto = 1`: T2c demoted it at 8.9 MeV after a stop move — owner: "the current identified end point is OK"), `039253_3/61` (T3c vetoed the one dot at 9.3 MeV / 7.3 cm while the attached arm 61007 was `kOther`; the missed stopper `039349_32/63` is the same shape: arm 63006 `kind 0 len 8.63 far_len 206.63 mip 0.78 kink 91.6 shower 1` — a textbook Michel arm rejected on `far_len` alone, then its dots vetoed at 4.0 MeV / 5.5 cm) | T2c reads KE alone; the arm's turn is not consulted. T3c is right about the dots — the loss is upstream, in the attached gate |
| **An interior arm the scanner calls Michel** | `039349_64/24` (24003, 14.4 cm, 1.11 MIP, kink 128°, `n_body_other = 1`) | interior arms can only be delta / hadron / other (`StmMichelFunctions.cxx:520-535`) — doc 64's 33 |
| **No fitted segment within 8 cm of the stop** | `039349_30/45`, `039349_43/66` (owner: "both"), `039349_58/69` (`n_stop_gammas = 7` — the dots are reconstructed, as capture gamma), `039349_72/11` (owner) | pdhd/13 D3: on PDVD 77 of 84 Michel losses have no arm at the stop; the fix is upstream of the classifier (Steiner terminal fragmentation, doc 62 §5) or in §5's collection |

The owner's smx3 notes land on these mechanisms exactly: `039253_0/44` and
`039349_48/21` ("Michel not identified" / "did not get accessed") are among
§3's 75 missed stoppers but not its 40 — their Michel was not found either, so
they are in the thirteen above (corrected from "in §3's 40") — `039349_48/21` carries a `kind 1` arm at
8.7 MeV that T2c demoted; `039253_13/73` ("clear Michel, the muon did not
reach the end, thus less clear Bragg peak") is the dilution mechanism in the
owner's words.

### 4.2 Proposals P2, P3, P3b

**P2 — PDVD operating points for the attached gate** (`StmMichelFunctions.cxx:482-518`,
`measure_arm :459-479`), three sub-knobs, each default = today:
(a) `michel_mip_lo_turned` — a lower charge floor (0.15) that applies only when
the arm turns hard (`kink ≥ 60°`), so a diluted Michel is admitted by its
topology, never by low dQ/dx alone (the comment at `:500-504` keeps its
meaning); (b) `michel_far_len_shower_exempt` — when the arm is shower-flagged,
`far_len` does not count against `michel_max_len_cm` (the 54 cm subtree of
19004 is the Michel's own brems; 63006's 206 cm subtree says the exemption
needs its own cap, e.g. `far_len ≤ 60 cm`, or the arm inherits an
over-clustered structure); (c) `michel_kink_window_cm` 5 → 10 as an option
(61007). Sizing: 4 items (`039349_11/19`, `039349_69/56`, `039253_3/61`,
`039349_32/63`) plus every other stop arm the change admits — listed by name
from the DEBUG line; `is_stm` can move through `michel_guards_stop`
(`:1972-1975`), so the census reports both flags.

*Corrected 2026-09-10 (doc 73 §2), re-measured on bare production (`p4v50`):*
- **(c) is misframed.** The classifier's kink window is 15 cm (`StmMichelArmThresholds::dir_window`), not 5 cm, so the helpful change is a *shorter* window for the Michel turn test.
- **The four named items do not all reach on production:**
  - `039349_11/19` needs (a) and (b), and 19004's whole subtree is ~66 cm, so it is out under a 60 cm cap.
  - `039349_69/56` no longer has a stop arm.
  - `039253_3/61` is (c)'s only.
  - `039349_32/63`'s 206 cm subtree is out under any cap.
- At the argued points (a) + (b) mostly turn bridged Michels into attached ones on items that already have `michel_found`; the predicted `michel_found` gain is about 0.

**P3 — `michel_collinear_split`.** On a *found* stopper (Bragg confirmed) whose
last chain segment continues past the Bragg peak for ≥ 3 cm at a dQ/dx that has
fallen back below 0.5 × plateau, split the segment at the fall (reuse
`stm_michel_stop_split`'s row test and `PR::break_segment`, `:1671-1688`) and
re-run the stop-arm classification on the remainder. This is T1c's mechanism
with the *opposite* precondition — today both retreat and split are skipped
when Bragg is confirmed, which is exactly when the swallowed Michel is
invisible. Sizing: 4 items; the two hot cases (`039349_60/40`, `039349_64/65`
at 1.35–1.47 MIP over 4–7 cm) may be Bragg tail rather than Michel and are the
named risk. Gate: `is_stm` 0 new FPs (a split can only move a found stopper's
stop backwards; `michel_guards_stop` then decides).

**P3b — a turn exemption for the moved-stop veto.** T2c (`:2701-2706`) skips
when the attached arm turns hard: `moved_stop_michel_kink_min` 60°. Checked
on the arm's own DEBUG lines: the two owner-confirmed Michels it demotes turn
59.6° (`039252_2/79`, 8.9 MeV) and 132.6° (`039349_48/21`, 8.7 MeV); the three
THRU items it rightly demotes turn 17.2° (`039252_4/55`), 43.5°
(`039349_20/41`) and 48.2° (`039349_61/62`) — length does not separate them
(3.9–9.3 vs 4.9–5.8 cm), KE does not (doc 61), the turn does, at 2 of 2
against 0 of 3. The range-energy veto (T3c) needs no exemption: on
`039253_3/61` and `039349_32/63` it is right about the dots, and the loss is
P2's (the attached arm was `kOther`). Exact offline; a 60° cut on 5 items is a
small-sample threshold and is stated as such.

*Corrected 2026-09-10 (doc 72 §2):*
- **The 60° cut spares one item, not two.** 59.6 < 60, so it spares `039349_48/21` only.
- **The THRU kinks above (17 / 44 / 48°) are survey-arm numbers.** On bare production (35 or 50 cm admission), `039349_61/62` turns 58.7°; the 60 cm survey admission changes its fit.
- A threshold in (58.7°, 59.6°] would spare both Michels. A threshold in (59.6°, 132.6°] spares `039349_48/21` alone.

---

## 5. Question 4 — the Michel shower and its isolated gammas

### 5.1 What the object holds today

* **Admission** (`:1381-1425`): every cluster of the event with the same
  `matched_flash_gid` (a hard `rec.gid ≥ 0` prerequisite, `:1394`), ≤ 25 cm
  long, within `admit_radius` of **the tagger's stop before
  extend/retreat/split** (`:1413`, `:1420`). With production knobs that radius
  is max(`michel_dot_radius_cm` 15, `stop_gamma_radius_cm` 35) = **35 cm**; the
  scan arms' `survey_enable` raises it to 60 cm and draws what it sees as role 6
  rows — which is why the owner could tag gammas out to 58 cm at all.
* **The Michel object**: the attached arm(s) plus everything
  `complete_structure_with_start_segment` reaches by graph connectivity
  (`:2026-2030`, no distance test), plus companion **pieces** whose cluster
  comes within 15 cm of the final stop and whose segments are ≤ 25 cm and
  farther from the muon body than from the stop (`:2164-2225`), plus T3b's
  same-cluster disconnected pieces. Energy `michel_ke_best = michel_ke_dqdx +
  dots_ke_unfit` (`:2686`).
* **The capture gamma** (role 5, `:2315-2524`) is a *separate* object: the
  ring 15 < d ≤ 35 cm from the final stop, ≤ 10 cm, 0.2–20 MeV, cluster-level
  body exclusion, one Shower per cluster, its energy in `stop_gamma_ke_*` and
  never in `michel_ke_*`. It does not ask whether a Michel exists.
* Two display facts: role 4 is **never emitted** (`:2240-2242`, a documented
  free slot), and shower members found by the graph walk get PDG 11 but **no
  point rows unless `survey_enable`** (`:2041-2059`) — production under-draws
  its own Michel object.

### 5.2 The scan's gamma tags against the arm

The record carries 334 `gamma`-tagged segments on 161 items, 94 of them
scan-Michel items ("both" = attached Michel plus detached dots is the majority
Michel kind: 88 of 164). `d70_sizing.py` §D matches them into `d68a3` by
geometry, the C2 rule:

| gamma-tagged segments in the arm | role 3 (in the Michel) | role 5 (capture gamma) | role 6 (survey only) |
|---|---:|---:|---:|
| all, 307 on 152 items | 29 | 42 | 236 |
| on scan-Michel items, 192 on 90 items | 12 | 16 | 164 |
| on scan-Michel items with `michel_found = 1`, 165 on 78 items | 11 | 8 | 146 |

Distance to the stop: quartiles 20 / 30 / 42 cm, p90 51, none beyond 60 (the
survey radius bounds what the scanner saw); length median 0.7 cm, p90 3 cm —
dots. By ring, on scan-Michel items: 10–15 cm mostly role 3 (9 of 12);
15–20 cm split (5 gamma, 18 survey); 20–35 cm 10 gamma, 72 survey; **35–60 cm
70 of 71 survey-only**. So on the 78 Michel events the chain found, **154 of
165 tagged gamma fragments sit outside the Michel object** (146 never
attached to anything, 8 booked as capture gamma). The object holds about 7 %
of the gamma charge the owner associates with it, and `michel_ke_best` is a
core-only energy. Conversely, 22 STM_ONLY items the scanner marked "detached
dots" have `n_stop_gammas ≥ 1`: their dots *are* reconstructed, as the µ⁻
capture signature — and on µ⁺ events (16 role-5 segments on scan-Michel items)
the same code books Michel brems as capture gammas.

### 5.3 Proposal P4 — `michel_gamma_collect`

After the Michel core is fixed (conn_type 1 or 2, `:2272`) and before the
capture-gamma stage (`:2315`), collect **unclaimed** companion segments and
unfitted companion clusters that are short (≤ `michel_gamma_max_len_cm` 10),
within `michel_gamma_radius_cm` (60) of the **final** stop, farther from the
muon body than `dot_body_exclusion_cm` (the cluster-level test the gamma stage
already has, `:2393-2418`), and — optionally, `michel_gamma_forward_only` —
in the forward hemisphere of the Michel's start direction. Publish them as
**role 4 "michel gamma"** (the free slot; claiming, since they carry energy),
count `n_michel_gammas`, and add `michel_ke_gamma` and `michel_ke_total` =
`michel_ke_best + michel_ke_gamma`. `michel_ke_best` itself is **not**
changed, so `michel_found`, the range-energy veto, T2c and `is_stm` are
bit-identical by construction; only rows and new branches appear when the knob
is on. When a Michel exists, the capture-gamma stage yields to role 4 (a µ⁺
has no capture gamma); when none does, role 5 is unchanged.

Two mechanics come with it: (i) admission must be computed from the final
stop, or re-run once after the stop moves — today a blob 10–40 cm from the
final stop may never have been admitted because the ring was drawn around the
tagger's stop (`:2164-2174` names a 266 cm case); (ii) the closest existing
code is `PatternAlgorithms::shower_clustering_in_other_clusters`
(`clus/src/NeutrinoShowerClustering.cxx:3684`: nearest-vertex attach,
connection type 3 / 4 by distance, force PDG 11, then
`complete_structure_with_start_segment`) and the ring helper
`stm_michel_stop_gamma_ring` (`StmMichelFunctions.cxx:537-546`); fork, do not
share (CLAUDE.md M10).

**Sizing and truth.** Membership: the 334 gamma tags are the truth — the gate
is "gamma-tagged segments in role 3/4" from 29 to most of 307 with the
non-Michel items' tags (115 segments on 62 items, many of them the µ⁻ capture
gammas) *not* absorbed on STM_ONLY events; the census reports both. Energy:
there is no truth in the record; report the `michel_ke_total − michel_ke_best`
shift and the fraction of Michel events above the 52.8 MeV endpoint before and
after (doc 51 §3.1 already flags `039252_15/77` at 76.8 MeV). Scan need: none
for membership; the display should draw role 4 so the owner can judge the
association on the next scan.

---

## 6. The proposal set, ranked

| id | knob(s), default OFF | mechanism site | sizing on `d68a3` (merged record) | gate | scan need |
|---|---|---|---|---|---|
| **P1** | `topology_stop_evidence`, `topology_michel_ke_min` 10, `topology_michel_len_min_cm` 3 — a Michel of sufficient quality clears `R_NO_BRAGG` + `R_SHAPE_FLAT` only | verdict, between `:2728` and `:2790` | exact: +22 TP / +3 FP on the doc 68 record (first version: +26 / +4, §3.3); **after `smx4`: +24 TP / 0 FP**, 197/7/79 → 221/7/55, F1 0.821 → 0.877; `michel_found` bit-identical; sub-knob `topology_clears_sparse` +4 TP / 0 FP more | exact offline re-verdict, then one arm; byte-identical OFF | **`smx4` DONE (§9)**: 0 of the admitted items are THRU. **Built and gated (§10); PDVD PRODUCTION with `topology_clears_sparse` (§10.4)**: 197/7/79 → 225/7/51 |
| **P4** | `michel_gamma_collect` (+ radius 60, max_len 10, forward_only), role 4, `michel_ke_gamma/_total`, `n_michel_gammas`; admission from the final stop | new block after `:2272`; `:1381-1425` | 154 of 165 gamma fragments on 78 Michel events outside the object today | rows + new branches only; `michel_ke_best`, `michel_found`, `is_stm` bit-identical | **Built, gated and PDVD PRODUCTION (doc 71)**: at 35 cm gamma tags in the object 9 → 75 of 159, purity 0.958, every pre-existing output identical; **radius 50 cm on the owner's call (§11)**: 104 of 159, purity 0.933, `is_stm` 223/8/53 → 225/7/51, three named movers all as the record says |
| **P3b** | `moved_stop_michel_kink_min` 60° — T2c skips a hard-turning attached Michel | `:2701-2706` | ~~+2 owner-confirmed Michels (59.6°, 132.6°)~~ **+1 at 60°, `039349_48/21` (59.6 < 60)**, 0 of doc 61's 3 THRU re-admitted (17°, 44°, ~~48°~~ **58.7° on bare production**; doc 72 §2) | exact offline | none. **Built and PDVD PRODUCTION (doc 72)**: +1 Michel (`039349_48/21`), nothing else moves; 90° identical |
| **P2** | `michel_mip_lo_turned` 0.15 @ kink ≥ 60°; `michel_far_len_shower_exempt` (capped; built as `michel_far_len_shower_max_cm`); `michel_kink_window_cm` ~~10~~ **shorter than today's 15 cm: 5 (doc 73 §2)** | `StmMichelFunctions.cxx:482-518`, `:459-479` | 4 named items (2 of them then lose their T3c veto for free) + whatever else the DEBUG line admits. **On production (doc 73 §2): one needs a cap above ~66 cm, one is (c)'s, one has no stop arm, one is out under any cap; ~0 `michel_found` TP predicted** | arm; both flags reported | none. **Built and gated, NOT flipped (doc 73)**: 0 judged `michel_found` TP (as predicted). Each sub-knob attaches owner-tagged delta / muon arms; (b) loses `039253_13/73` |
| **P3** | `michel_collinear_split` — split a confirmed chain's last segment where dQ/dx falls after the peak, re-classify the remainder | after `:1699`, reuse `stm_michel_stop_split` + `break_segment` | 4 items (2 cool, 2 hot = named risk) | arm; 0 new `is_stm` FPs | none |
| **P1b** | anchor rise precondition (doc 65 §5.1) | `:1764-1803` | ≤ 3 stoppers not already inside P1 | offline from `profile` first | — |
| **P5** | `pr.jsonnet` `stm_trackfitting_config_file` (default = shared file) | `protodunevd/pr.jsonnet:175, :1550, :1695` | none; enables a scoped `dx_norm_length` / step study | compiled-config diff 0 when unset | any step change → new scan |

**Recommended order:** P1 — `smx4` found no cost (§9); built, gated and
flipped to PDVD production with `topology_clears_sparse` (§10) → P4 (the Michel object's
completeness; no verdict moves; **done, doc 71, PDVD production**) → P3b + P2 together (the attached gate and
its vetoes, one arm; **done, docs 72–73: P3b PDVD production, P2 left OFF**) → P3 → P1b → P5 at the owner's discretion. Each is a
default-OFF knob under doc 56's bar: byte-identical OFF path on both
detectors, every gain and loss by item name, flip only on the owner's rule.

**Observations, no proposal:** the production display omits the shower-walk
members' rows (`:2041-2059`); the "15-key default" comment at
`wct-pr-perevt.jsonnet:224` is stale (25 keys); `do_multi_tracking_review.md`'s
"hardcoded 0.6 cm" is stale since doc 61 §9; doc line citations in docs 51,
54, 62 and pdhd/13, 15 predate the current `CheckSTM_Michel.cxx`.

---

## 7. Gates

| gate | result |
|---|---|
| code / config / labels touched | no C++, no config. Scripts `d70_sizing.py` (P1 gate corrected, §3.3), `d70_build_smx4.py`, `d70_score_smx4.py`. New record files only: the owner's labels committed byte-identical to `pdvd/work/stm_michel_labels/smx4/labels.json` (md5 `2a026c1e`) as `pdvd_stm_michel_smx4_labels.json`; the new merged record `pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json` (601 records, 54 from smx4; the 547 others byte-equal to the doc 68 record's; the owner's PF tags laid **on top of** the record's, since `pf_segments` holds overrides only: 0 earlier tags dropped, gamma tags 334 → 345). The doc 68 record is untouched (md5 `c8db1220`, git clean) |
| `census_score.py --check` | 0 of 14 differ (record untouched), before and after `smx4` |
| `smx4` display | all 54 items rendered in process with payload and question panel (scratch labeldir); the Michel radio resets to "not set" on an unlabelled item; the live label dir was empty before the scan; the process on :5017 was checked by its command line |
| §9 numbers | `d70_score_smx4.py` stdout (`/home/xqian/tmp/d70/score_smx4.txt`); its production row reproduced by `census_score.py` on the new record (`/home/xqian/tmp/d70/score_merged4_d68a3.txt`: `is_stm` 197 / 7 / 79, `michel_found` 133 / 12 / 25) |
| P1 C++ (§10) | `wcdoctest-clus` 360/360; knob-OFF gate PDVD 120/120 zips, 578/578 candidates × 133 `T_stm_michel` branches; PDHD 61/61, 325/325; knob ON = the offline rule on 568/568 candidates (§10.2) |
| PDVD flip (§10.4) | compiled JSON of the committed `pdvd/wct-pr-perevt.jsonnet`: flip-equivalence (pre-flip file + `{topology_stop_evidence:true, topology_clears_sparse:true}` vs the flipped file) 0 lines; OFF path (both files + both keys false) 0 lines; pre-flip vs flipped exactly the two keys (`/home/xqian/tmp/d71/flip/F_*.json`) |
| §3.3 corrected grid | `d70_sizing.py` with the residual-bit test (`/home/xqian/tmp/d70r2/sizing.txt`); the doc 68-record row of §9.3 is the same number from the other script (219 / 10) |
| every number in §3–§5 | `d70_sizing.py` stdout (`/home/xqian/tmp/d70/sizing.txt`), read-only over `prep_d68a3`, the two baseline preps and the merged record |
| §2 facts | `TrackFitting.cxx` / `TrackFittingPresets.h` / wire files, cited by line; doc 65 §3–§4 numbers quoted, not recomputed |
| §3.1 / §5.1 code facts | `CheckSTM_Michel.cxx`, `StmMichelFunctions.cxx` at toolkit `HEAD`, cited by line |

## 8. Doc 56 update

Doc 56's Order paragraph gains one line: the campaign's close-out list is
superseded by this doc's §6 as the next task set; the grading record stays the
merged one.

---

## 9. The owner's `smx4` scan — P1 graded (2026-09-10)

The owner: "I would like to proceed to P1, please serve the display, and make
it clear what I need to select", then, after scanning, "review it and update
the 70*.md and next step".

### 9.1 What was scanned, and how it was blinded

`scripts/d70_build_smx4.py` took all 54 judged items on which production
(`d68a3`) reads `is_stm 0` and `michel_found 1`: the 40 the record called
stoppers and the 14 it called THRU (§3.2). Production reads the same on every
one (the script asserts it), so the display, which shows production's answer,
could not tell the two groups apart. The order was shuffled (seed 70), every
sheet row was tranche 1, and the group, the record verdict and P1's reading
lived only in `pdvd/docs/scan/pdvd_stm_michel_smx4_key.tsv`. The blue panel
stated production's reading and its reject bits in words, and the choices:
verdict button, Michel radio, pin only if the fit end is wrong, and **a note
whenever the muon stops but the chain's Michel is the wrong object**. The
residual leak was stated on the panel: the chain-answer box prints the
Michel's KE and length, which are P1's gate variables ("judge from the
picture"). Four items were already owner verdicts from `smx3`; they stayed in as
a repeat look.

### 9.2 The owner against the record

| record said | n | owner: stopper | THRU | MESSY |
|---|---:|---:|---:|---:|
| stopper (the 40 FN-with-Michel) | 40 | 36 | 1 | 3 |
| THRU (the 14 TN-with-Michel) | 14 | **8** | 5 | 1 |

Stopper/not agreement is 41 of the 50 items the owner judged. Changed:
THRU → stopper `039252_12/90`, `039252_3/45`, `039349_35/30`, `039349_39/56`,
`039349_45/25`, `039349_76/23`, `039349_78/22`, `039349_81/51`; stopper → THRU
`039252_8/82`; → MESSY `039253_6/90`, `039349_54/56`, `039349_63/41`
("overclustering"), `039349_12/45`. The direction is the one doc 68 found (4 of
its 6 changes were THRU → stopper): smx1a under-calls stoppers that carry a
reconstructed Michel, here on 8 of 13 judged record-THRU items. The set is
selected (every item has a production Michel), so this is not the record's
error rate.

The four repeat looks agree with `smx3` on stopper/not 4 of 4. Three are
identical; `039253_12/93` moved from Michel "both" to "attached", with the same
comment worded twice ("not fully identified" / "Michel did not cover the entire
thing").

### 9.3 P1 against the owner

At the argued point (exact rule), P1 fires on 25 of the 54: **24 owner
stoppers, 0 THRU, 1 MESSY** (`039349_63/41`, which the census does not score).
All three record-THRU items it admits (`039349_39/56`, `039349_45/25`,
`039349_81/51`) are stoppers. On `039349_39/56` the owner wrote down the
mechanism P1 rests on: *"The reason there is no Bragg peak is that it overlapped
with the Michel electron, so the dQ/dx was shared."*

Census on the `d68a3` payload population (`d70_score_smx4.py` §4; the
production rows are reproduced by `census_score.py`):

| record | arm | TP | FP | FN | purity | efficiency | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| doc 68 (smx1a + smx3) | production | 197 | 7 | 75 | 0.966 | 0.724 | 0.828 |
| doc 68 | P1 10 MeV / 3 cm | 219 | 10 | 53 | 0.956 | 0.805 | 0.874 |
| **+ smx4** | production | 197 | 7 | 79 | 0.966 | 0.714 | 0.821 |
| **+ smx4** | **P1 10 MeV / 3 cm** | **221** | **7** | **55** | **0.969** | **0.801** | **0.877** |
| + smx4 | P1 10 / 3 + `topology_clears_sparse` | 225 | 7 | 51 | 0.970 | 0.815 | 0.886 |
| + smx4 | `michel_found` (P1 does not touch it) | 133 | 12 | 25 | 0.917 | 0.842 | 0.878 |

On the 576-item union of the updated record (four items became MESSY), where
an item with no candidate counts as 0, `is_stm` F1 goes 0.801 → 0.857. `michel_found` moves only because the truth moved: the
owner's verdicts remove 5 of its 17 false positives (production is unchanged).

The P1 grid on the updated record (new TP / new FP over production, exact rule):

| KE ≥ \ len ≥ | 0 | 2 | 2.5 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 0 MeV | +34 / +6 | +31 / +2 | +28 / 0 | +27 / 0 | +26 / 0 | +21 / 0 |
| 5 | +30 / +3 | +29 / +2 | +26 / 0 | +26 / 0 | +26 / 0 | +21 / 0 |
| 8 | +26 / 0 | +26 / 0 | +24 / 0 | +24 / 0 | +24 / 0 | +21 / 0 |
| **10** | +26 / 0 | +26 / 0 | +24 / 0 | **+24 / 0** | +24 / 0 | +21 / 0 |
| 15 | +18 / 0 | +18 / 0 | +18 / 0 | +18 / 0 | +18 / 0 | +17 / 0 |

The six owner-THRU items in the set carry "Michels" of 0.8–2.4 cm and
1.7–7.0 MeV, so either cut alone separates them from every stopper P1 touches.
The argued point sits outside both. The looser rows buy +2 to +4 stoppers, but
they were picked on this very scan: they are offered, not recommended (doc 62:
never ship the in-sample best row as a default).

### 9.4 What P1 does not reach, and what the scan says about the rest

P1 leaves 20 owner stoppers in the set at `is_stm 0`:

* **Four keep only `profile_sparse` besides the shape bits:** `039252_3/45`,
  `039349_41/51`, `039349_61/21`, `039349_82/54`. All four are owner
  STM_MICHEL with 6–22 cm, 12–27 MeV Michels. `profile_sparse` means the dQ/dx
  has too few live points to judge, i.e. the rise evidence is missing, which is
  exactly where the owner's rule lets topology decide. Clearing it too gives
  **+4 TP, 0 FP** on the updated record (on the doc 68 record its one "FP" was
  `039252_3/45`, now an owner stopper). Proposed as sub-knob
  `topology_clears_sparse`, default off, measured in the same arm; the owner
  decides.
* **Four are on `stop_near_boundary`:** `039252_2/103`, `039349_27/41`,
  `039349_35/30`, `039349_78/22` (the owner: "Again the STM did not read the
  end?"). All are owner stoppers, but the fiducial inset is a containment rule
  for the energy, not a stop test; P1 does not touch it.
* **Two keep an arm bit:** `039349_5/65` keeps `continuation` (owner: stopper
  with detached dots) and `039349_76/23` keeps `vertex_hadron` (a 0.7 MeV
  Michel, pin moved 2.3 cm). Both belong to P2 / P3.
* **Ten have short or faint Michels** (0.9–4.2 cm, 3–14 MeV): `039252_12/90`,
  `039253_12/93`, `039253_8/64`, `039349_19/52`, `039349_22/56`,
  `039349_36/46`, `039349_38/57`, `039349_63/55`, `039349_68/63`,
  `039349_70/61`. These are what the grid's looser rows would reach.

**Two P1 recoveries are right for the wrong reason.** `039252_16/98` (owner
STM_ONLY, "hadronic shower?") and `039349_81/51` (STM_ONLY, detached dots, the
µ⁻ capture signature of §5.2) are stoppers, but the object that fired P1 (both
`conn_type` 2) is not a Michel to the owner. `is_stm` is right on both, and
`michel_found` stays a false positive, as it is today.

**Stop-point evidence (for P3 / T1a, not P1).** Pins moved on six stoppers by
1.2–9.5 cm: `039349_68/63` 1.2, `039349_76/23` 2.3, `039349_63/55` 4.3,
`039349_19/52` 4.7, `039349_70/61` 6.5, `039349_28/60` 9.5. Notes: "the muon
end point was not done properly" (`039253_0/102`) and "PR not exactly right
near vertex" (`039349_61/21`). `039253_8/31` (THRU) carries a pin moved
29.1 cm, which the census ignores on a THRU item. `039253_12/93`'s "Michel did
not cover the entire thing" is a P4 case. So are the owner's PF tags: on 8
items the owner tagged segments, 11 more of them as gamma, so the updated
record carries 345 gamma tags (§5.2's 334 are on the doc 68 record). P4 is
sized on the updated record when it is built.

**The last unreviewed production FPs.** Three of production's seven FPs are
smx1a-only verdicts: `039349_13/56`, `039349_38/60`, `039349_59/14` (all
medium confidence, none with a Michel). The other four are owner THRU. Given
8 of 13 here and 4 of 6 in doc 68, a three-item look is cheap.

### 9.5 Decision and next step (executed in §10; flipped with `topology_clears_sparse`)

P1 clears the owner's rule at the argued point with 0 new false positives on
the owner's own verdicts. Implementation, under doc 56's bar:

1. **C++**, `CheckSTM_Michel.cxx`, between `michel_found` (`:2728`) and
   `is_stm` (`:2790`). Knobs: `topology_stop_evidence` (bool, default false),
   `topology_michel_ke_min` (10.0 MeV), `topology_michel_len_min_cm` (3.0 cm),
   `topology_clears_sparse` (false). The rule is exactly §3.3's: `michel_found`
   1, `conn_type` ∈ {1, 2}, both minima; clear `R_NO_BRAGG` and
   `R_SHAPE_FLAT` (plus `R_PROFILE_SPARSE` with the sub-knob). Persist
   `topology_cleared_bits` (the bits it cleared; written only when the knob is
   on) so the census can name every fire. Round-trip the
   keys in `default_configuration()` (`:251`), extend
   `doctest_check_stm_michel_defaults.cxx`, and add a rule test on synthetic
   reject bits.
2. **OFF gate**, byte-identical, PDVD and PDHD, against a fresh legacy arm on
   the **merged binary** (toolkit `98140fee`, pinned). `local/lib` moved today,
   so the doc 68 arms are not a valid baseline.
3. **ON arm, PDVD**, `is_stm` and `michel_found`, graded on the updated record.
   It must reproduce this section item for item: +24 named stoppers and 0 new
   FP at 10 / 3, +4 more with `topology_clears_sparse`, and `michel_found`
   bit-identical.
4. **Flip** in `pdvd/wct-pr-perevt.jsonnet` on the owner's go. PDHD stays OFF
   (it has no hand-scan record). From here on, the grading record is
   `pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json`.

Then P4 → P3b + P2 → P3 → P1b → P5, as §6.

---

## 10. P1 built and gated (2026-09-10)

The owner: "please stop 5017, and proceed to the steps before P4, please
update the md file, commit and push." The `smx4` display on :5017 is stopped.
This section is §9.5's steps 1–3. Step 4, the production flip, was first held
back (it changes PDVD production output, the owner's call) and then applied on
the owner's reply, "flip with sparse please" (§10.4).

### 10.1 What was built (toolkit `f66a8b9f`)

* **The rule is one pure function**, `stm_michel_topology_clear`
  (`StmMichelFunctions.{h,cxx}`). From reject_bits, michel_found, conn_type, KE,
  length, the two minima and `clears_sparse`, it returns the bits to clear.
  This is the same expression `d70_score_smx4.py` applied offline. Three new
  doctest cases pin it: the shape bits clear, and every other bit survives;
  `R_PROFILE_SPARSE` clears only on request; no Michel, a charge-only object,
  a short or faint one (including `039349_77/52`'s 9.966 MeV) and a NaN clear
  nothing. Both minima are inclusive.
* **`CheckSTM_Michel.cxx`** has four knobs: `topology_stop_evidence` (false),
  `topology_michel_ke_min` (10 MeV), `topology_michel_len_min_cm` (3 cm) and
  `topology_clears_sparse` (false). They are round-tripped in
  `default_configuration()` and pinned in
  `doctest_check_stm_michel_defaults.cxx`. The call sits **just before
  `persist`**, after `R_CLUSTER_NOT_TRACK` and `R_STOP_NEAR_BOUNDARY` are
  set, so every bit is final when it runs. A DEBUG line names each fire.
  `topology_cleared_bits` is persisted **only when the knob is on** (the
  survey's pattern), so the knob-off tree keeps its branch list.
  (The length knob carries its unit: `_cm`, renamed from this doc's first
  `topology_michel_len_min`.)
* **`prep_stm_michel_scan.py`** carries `topology_cleared_bits` into the
  payloads.

### 10.2 Gates

Six arms on the same input (`d16vnu` / `d16hnu`), each on a private pin:
`d71vleg` and `d71hleg` on the merged binary before the change, and the other
four on `f66a8b9f`: `d71voff` and `d71hoff` (knob off), `d71vp1`
(`topology_stop_evidence`) and `d71vsp` (plus `topology_clears_sparse`). All
arms use the scan TLA doc 68 used (the survey), and none touches production.

| gate | result |
|---|---|
| unit tests | `wcdoctest-clus` 360/360, 3 new |
| freshness / pins | installed lib 09:42:14, after the last source edit (09:41:11); pins md5 `e5cd0364be81` (base) and `0d7027b23351` (P1), unchanged before and after every arm; 0 loader deaths |
| compiled config | leg and off arms: no `topology_*` key in `CheckSTM_Michel`; `d71vp1`: `topology_stop_evidence` only; `d71vsp`: both keys |
| **OFF gate PDVD** (`d71vleg` vs `d71voff`, 120 events) | `mabc-pr.zip` member content 120/120; `calib-pr` JSON 119/119 (`039252_11` writes none on either binary, as on `d68a3`); `T_stm_michel` **578 / 578 candidates bit-identical on all 133 branches**, 0 new / 0 dropped; `T_stm_michel_pts` geometry and roles 578 / 578 |
| **OFF gate PDHD** (`d71hleg` vs `d71hoff`, 61 events) | zip 61/61; calib 61/61; **325 / 325 on 133 branches**; points 325 / 325 |
| the prediction transfers | `d68a3` (doc 66's binary, the arm §9 was computed on) vs `d71vleg` (merged binary): 568 / 568 payload verdicts identical — the merge moved nothing here, so §9's numbers apply unchanged |
| **ON = the rule** (`d70_p1_check.py`) | `d71vp1` vs `d71vleg`: 0 mismatches on 568 candidates in `is_stm`, `reject_bits` and `topology_cleared_bits`; no other verdict field moved. The same for `d71vsp` (`--sparse`). On `d71voff` the rule never fires and the branch is absent from all 568 |
| `census_score.py --check` | 0 of 14 differ |

Labels: gate outputs in `/home/xqian/tmp/d71/` (`gates.log`, `g_pdvd/`,
`g_pdhd/`, `p1_check.txt`, `sp_check.txt`, `off_check.txt`,
`score_d71v*.txt`); preps `/home/xqian/tmp/d71/prep_d71v{leg,off,p1,sp}`.

### 10.3 What P1 does on PDVD

Graded on the smx1a + smx3 + smx4 record, payload population
(`census_score.py`):

| arm | `is_stm` TP / FP / FN | purity | efficiency | F1 | `michel_found` TP / FP / FN |
|---|---|---:|---:|---:|---|
| `d71vleg` (P1 off) | 197 / 7 / 79 | 0.966 | 0.714 | 0.821 | 133 / 12 / 25 |
| **`d71vp1` (P1, 10 MeV / 3 cm)** | **221 / 7 / 55** | **0.969** | **0.801** | **0.877** | 133 / 12 / 25 |
| `d71vsp` (+ `topology_clears_sparse`) | 225 / 7 / 51 | 0.970 | 0.815 | 0.886 | 133 / 12 / 25 |

This is §9.3's prediction exactly. The rule fires on 32 candidates and flips 26
of them to `is_stm 1`: the 24 owner stoppers of §9.3, and two items the census
does not score, `039349_63/41` (owner MESSY, "overclustering") and
`039253_0/99` (smx1a MESSY; not in the smx4 set, since the record never called
it a stopper or a THRU). Six fire without flipping because another bit remains:
`039252_3/45`, `039349_41/51`, `039349_61/21` (`profile_sparse`),
`039349_5/65` (`continuation`), `039349_46/58` (`profile_sparse`, MESSY) and
`039349_81/54` (`plateau_off_mip`, MESSY). With `topology_clears_sparse` it
flips 31: add `039252_3/45`, `039349_41/51`, `039349_61/21`, `039349_82/54`
(owner stoppers) and `039349_46/58` (MESSY).

**One observation, no change made.** P1 has no upper energy bound. Two MESSY
items carry P1-qualifying objects far above the 52.8 MeV Michel endpoint:
`039349_81/54` (206 MeV) keeps `plateau_off_mip` and does not flip, but
`039349_46/58` (76 MeV) **does** flip once `topology_clears_sparse` is on. The
census does not score either, but nothing in P1 stops such an object from
clearing the shape bits on a candidate with no other objection.
Every stopper P1 flips carries ≤ 50.7 MeV. A `topology_michel_ke_max` of about
60 MeV would cost nothing on this record and is physically argued. It is
offered as an option, not built.

### 10.4 The production flip — applied with `topology_clears_sparse` (owner, 2026-09-10)

The flip was first held back and shown as a one-line diff. The owner
answered "flip with sparse please". The whole change to
`pdvd/wct-pr-perevt.jsonnet` is two keys after `bragg_peak_search_cm: 3.0,`,
with a comment carrying the C++ defaults, the guarantee and the graded result:

```jsonnet
        topology_stop_evidence: true,
        topology_clears_sparse: true,
```

The two minima stay **unset** at their C++ defaults (10 MeV, 3 cm, what the
scored arm ran), for the inert-key reason docs 58 and 61 found. PDVD
production is therefore exactly what arm `d71vsp` ran: `is_stm` 197 / 7 / 79 →
**225 / 7 / 51** (purity 0.966 → 0.970, efficiency 0.714 → 0.815, F1 0.821 →
0.886), 0 new false positives, `michel_found` identical. The new baseline prep
for later rounds is `/home/xqian/tmp/d71/prep_d71vsp`.

**Correction (doc 71 §7, 2026-09-10): "exactly what arm `d71vsp` ran" is not
right.** Every §10 arm carried the survey TLA, which widens companion admission
from 35 to 60 cm, and doc 53 measured that widening as a fit perturbation on
20 % of PDVD candidates. Production runs without the survey. The first
bare-production arm (`p4vleg`, the same f66a8b9f binary, the flipped bag) reads
`is_stm` **223 / 8 / 53** on the record (purity 0.965, efficiency 0.808), with
`michel_found` 133 / 12 / 25 unchanged. Three items differ from `d71vsp`:
- `039253_10/93`: record THRU, `is_stm` 1 in production;
- `039349_64/24`: record STM_MICHEL, 0 in production;
- `039349_64/61`: record STM_ONLY, 0 in production. This is doc 53's
  "scan before trusting 60 cm" flip, and the record says 60 cm had it right.

The P1 gain above (+28 / 0 FP) was measured with both arms at 60 cm and stands
as measured there. Its bare-production size was not measured separately. The
baseline prep for later rounds is the bare-production `/home/xqian/tmp/p4/prep_p4v50`
(doc 71 §11: production with the owner's 50 cm P4 radius, which also brings production
back to 225 / 7 / 51).

Compiled-config proofs, on the committed file (`/home/xqian/tmp/d71/flip/F_*.json`):

* **flip-equivalence:** the pre-flip file plus `-S
  stm_michel_extra={topology_stop_evidence:true,topology_clears_sparse:true}`
  against the flipped file: 0 lines.
* **OFF path:** both files with both keys forced false: 0 lines.
* **what moved:** the pre-flip file against the flipped file differs by exactly
  the two keys.

The file was edited with no PDVD run in flight. PDHD stays OFF, since there is
no PDHD hand-scan record.

### 10.5 Next

**P4** (`michel_gamma_collect`), sized on the updated record's 345 gamma tags,
graded against the new production (`d71vsp`'s verdicts are what PDVD now
produces). The `topology_michel_ke_max` cap of §10.3 stays an open option.

*Done:* P4 is doc 71 (built, gated, PDVD production at 35 cm). The sentence
above about `d71vsp` is corrected in §10.4. Next is P3b + P2, in their own doc.
*(Update 4: P3b + P2 are done in docs 72 and 73: P3b in PDVD production, P2
built, gated and left OFF. Next is P3.)*
