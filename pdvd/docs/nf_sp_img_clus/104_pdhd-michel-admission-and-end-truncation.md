# 104 — PDHD's Michel purity cost is not an admission problem (and what it is)

**Status (2026-09-15, round 7 of the doc 101/102/103 campaign).**
* **Question.** Turn the two trajectory levers on for PDHD — the doc 101 fit knobs (`fit_weight_pow` 1.5 +
  `assoc_cont_center` 1) and the doc 102 retile `charge_stepped`. PDVD runs both in production since toolkit
  `8fc6070e` (doc 103 sec 14). PDHD is blocked on **one** metric: Michel purity 0.946 → 0.895 (−0.051) against a
  −0.020 bar, while both `is_stm` metrics pass (−0.017, −0.015) and Michel efficiency **gains** +0.060.
* **What doc 103 sec 14.4b set this round.** "Round 7 builds an admission rule on sec 10.4's class — default OFF
  with a byte-identical gate, graded on both detectors so PDVD keeps its D1."
* **Answer: that premise is wrong, and this round's job was to establish it before any C++ was written.**
  1. **No cut on the persisted admission geometry separates PDHD's 6 A1-only Michel false positives from its 77
     true Michels** (sec 2). Four levers refuted with numbers: energy floor, length floor, muon-continuation veto,
     and requiring the muon's own Bragg peak at the stop.
  2. **The false positives are not one class** (sec 3). Measured at point level: 2 of the 5 measurable ones are
     *the muon's own charge*, re-labelled after the trajectory stopped short; one is a separate object 14 cm away
     on a muon that got *longer*; one has no fitted points at all. They do not share a region of feature space
     because they do not share a cause.
  3. **The mechanism is trajectory-end truncation, not Michel admission** (sec 3). On the flagship case the muon
     loses 51 cm and 19.4 cm of it comes back as a 64.1 MeV "Michel" at 1.26 MIP. The owner's own `own103h` notes
     say the same thing: *"end track was not found correctly, so the dQ/dx is off near the end"*, *"michel pin
     moved 12.1 cm"*.
  4. **One lever does clear the arithmetic bar on both detectors** — a 3 cm length floor restricted to attached
     Michels (sec 2.4). It is reported and **not adopted**: it leaves 4 of the 6 targeted false positives standing,
     discards 1.5 (PDHD) and 4.3 (PDVD) true Michels per false positive removed, half its PDHD gain comes from
     removing errors production *also* makes, its joint passing window is 0.2 cm wide, and it would hand back
     PDVD's banked Michel efficiency. It was found by scanning the same data it would be graded on.
* **What can still move the verdict without code** (sec 5). The grade compares a **reviewed A1 side against an
  unreviewed A0 side**: of the Michel movers, `own103h` judged 8 of 33 A1-only tags and **0 of 21** A0-only tags.
  Amendment 6 set the remedy for exactly this shape on PDVD. **Amendment 7** (frozen, sha `321360da`,
  2026-09-15T18:47:32) applies it to PDHD; the 24-item set `own103h2` is drawn and served.
* **Not flipped. No production change, no toolkit change in this round.** PDHD keeps production's trajectory until
  the symmetric check is read.
* **Scope.** No C++, no config, no new reconstruction arm. Every number comes from the doc 101/102 arms
  `d101hnew` / `d102hcs` (PDHD, 61 events) and `d103v0` / `d103v1` (PDVD, 120 events), pin
  `/home/xqian/tmp/d102/libpin_d102`, `libWireCellClus` md5 `091e142b9481`.

---

## 0. Repro

```bash
cd pdvd/docs/nf_sp_img_clus/scripts
IMG=/home/xqian/toolkit-dev/wcp-porting-img

# sec 2 -- the feature table and the four levers (V1: both reproduce the committed grade before any lever is read)
python3 d104_michel_features.py     > ../figs/104_michel_features.txt
python3 d104_lever_sizing.py        > ../figs/104_lever_sizing.txt
# sec 3 -- where A1's Michel charge comes from, with the both-arms true Michels as the control
python3 d104_michel_provenance.py   > ../figs/104_michel_provenance.txt

# sec 5 -- the symmetric check.  The amendment is frozen BEFORE the set is drawn.
(cd ../figs && head -1 104_pred_amend7.sha256 | sha256sum -c)
D103_PDHD_RECORD=$IMG/pdhd/docs/scan/pdhd_stm_michel_smx27_verdicts.json \
  python3 d104_tpmover_set.py --new-record $IMG/pdhd/docs/scan/pdhd_stm_michel_smx28_verdicts.json \
    --owner-record $IMG/pdhd/docs/scan/pdhd_stm_michel_own103h_verdicts.json \
    --out /home/xqian/tmp/d104/own/set_pdhd_tpmover
D103_PDHD_RECORD=$IMG/pdhd/docs/scan/pdhd_stm_michel_smx27_verdicts.json \
  python3 d104_tpmover_score.py --new-record $IMG/pdhd/docs/scan/pdhd_stm_michel_smx28_verdicts.json \
    --owner-record $IMG/pdhd/docs/scan/pdhd_stm_michel_own103h_verdicts.json \
    --set /home/xqian/tmp/d104/own/set_pdhd_tpmover \
    --labels $IMG/pdhd/work/stm_michel_labels/own103h2/labels.json \
    --record-out $IMG/pdhd/docs/scan/pdhd_stm_michel_own103h2_verdicts.json
```

The starting grade this round argues about is doc 103's
`figs/103_union_grade_pdhd_own103h_stmonlyneg.txt` (truth `own103h > smx27 > smx28`, amendment 5's
`--stm-only-unset-negative`). Both new sizing scripts assert they reproduce it before printing anything (V1).

---

## 1. The question doc 103 left

| PDHD, A0 `d101hnew` (production) → A1 `d102hcs` (both levers) | A1 − A0 | bar −0.020 |
|---|---|---|
| `is_stm` purity 0.984 → 0.967 | −0.017 | pass |
| `is_stm` efficiency 0.614 → 0.599 | −0.015 | pass |
| **`michel_found` purity 0.946 → 0.895** | **−0.051** | **FAIL** |
| `michel_found` efficiency 0.603 → 0.664 | +0.060 | pass |

Michel false positives go 4 → 9, and true Michels 70 → 77. A1 is **better** at finding Michels and also admits
more junk. Doc 103 sec 10.4 named the junk: six clusters the owner reads as stoppers with no visible Michel, to
which A1 attaches a piece. Doc 103 sec 12.4 had already refuted an energy floor. This round asks whether *any*
admission cut works.

**The arithmetic target.** With TP′ = 77 − t and FP′ = 9 − f, the bar needs f ≥ 2.85 + 0.08·t: remove 3 false
positives while losing at most 1 true Michel, or 4 while losing at most 9.

---

## 2. Four levers, all refuted (`figs/104_lever_sizing.txt`)

Every row is **A1 + lever vs production A0 unchanged** — the only framing that describes a shippable unit, since
the knob would ship with the flip and production has no knob. Doc 103 sec 12.4's "floor on both arms" framing is
not repeated: it measures a knob nobody would ship, and it always fails for the wrong reason, because the floor
also cleans up A0's false positives and so raises the bar it is judged against.

### 2.1 The structural trap: `michel_len` means three different things

`michel_found` is `michel_conn_type > 0` (`CheckSTM_Michel.cxx:4482`), and `conn_type` has three writers:

| conn | how the Michel was found | `michel_len` | `michel_mip`, `michel_kink_deg` |
|---|---|---|---|
| 1 attached | a `kMichel` arm at the stop vertex | that arm's fitted length | **measured** |
| 2 bridged | nearest admitted companion piece | ~0 **by construction** | never measured (0.00, −1) |
| 3 charge only | an unfitted companion cluster | ~0 **by construction** | never measured (0.00, −1) |

PDHD's Michels split TP {1: 56, 2: 20, 3: 1} against FP {1: 6, 2: 2, 3: 1}. So a cut on `michel_len` pooled over
all three removes conn 2/3 objects wholesale for a reason that has nothing to do with their size, and a cut on
`mip` or `kink` silently exempts them — 3 of PDHD's 9 false positives and 21 of its 77 true Michels.

### 2.2 The four levers

| lever | best PDHD setting | Michel purity Δ | efficiency Δ | reading |
|---|---|---|---|---|
| **L1** energy floor `michel_ke_best ≥ E` | 8 MeV | −0.004 | −0.043 | fails (doc 103 sec 12.4) |
| **L2a** length floor, all conn | 1.5 cm | −0.014 | −0.009 | passes **only** at 1.5 (2.0 fails on efficiency, 1.0 on purity) |
| **L2b** length floor, conn 1 only | 3.0 cm | −0.012 | +0.009 | passes 3.0–3.5 |
| **L3** continuation veto (conn 1, long, MIP-like, barely turned) | any | −0.040 | +0.060 | fails — removes 1 of 9 |
| **L4** require the muon's Bragg peak at the stop | 0.6 × expected | −0.035 | −0.078 | fails — 6 of 9 FPs pass the test, 16 TPs fail it |

**L1** is refuted as doc 103 found: the false positives span 1.1–64.1 MeV while 13 true Michels sit below 10 MeV.

**L3** is the physically appealing one — a MIP-like arm that barely turns is muon continuation, which is exactly
what the gate's own continuation clause says (`StmMichelFunctions.cxx:620-625`: kink < 20°, len > 3 cm,
0.7 ≤ mip ≤ 1.3). Widening it catches **one** false positive (`029107_28/109`, the 19.4 cm / 1.26 MIP / 31.3° one,
which escapes the real clause by 11° and clears the Michel turn test by 1.3°). The rest sit inside the true-Michel
bulk: TP `mip` quartiles 0.33 / 0.74 / 1.07 / 1.20 / 1.93 and `kink` 15 / 47 / 73 / 107 / 175 against false
positives at `mip` 0.44–1.26 and `kink` 31–156.

**L4** is the other physically appealing one: if the muon did not stop here, whatever is attached here is not a
Michel. But `bragg_here` is true for 6 of the 9 false positives and false for 16 true Michels, so it costs four
times what it buys.

### 2.3 Why no cut works

Doc 103 sec 12.4 read this as "9 items are too few to fit a cut on". Sec 3 gives the stronger reason: **they are
not a class**. Three different failure modes land in the same output branch, so no region of the admission feature
space contains them and not the true Michels.

### 2.4 The one lever that clears the bar, and why it is not adopted

**L2b at 3.0 cm** — reject an *attached* Michel whose arm is shorter than 3 cm — passes on both detectors:

| | PDHD | PDVD |
|---|---|---|
| purity | 0.934 (−0.012) | 0.952 (+0.028) |
| efficiency | 0.612 (+0.009) | 0.661 (+0.000) |

3 cm is not an arbitrary value: it is `continuation_min_len`, the length scale already in the gate for "is this a
real arm". That is the strongest thing that can be said for it. Against it, from `figs/104_lever_sizing.txt`:

* **It does not remove the class it was aimed at.** Of PDHD's 4 removed false positives, **2** are A1-only and
  **2 are shared with production A0, which keeps them** — those improve the A1 − A0 *delta* without touching the
  regression. **4 of the 6 A1-only false positives survive**, including the 19.4 cm flagship.
* **It does not discriminate, it shrinks.** 6 true Michels discarded per 4 false positives removed on PDHD
  (1.5 each); 13 per 3 on PDVD (4.3 each). Purity rises because the short-arm region is dense in *both*.
* **Its joint passing window is 0.2 cm.** PDHD fails at 2.8 and at 4.0; PDVD fails at 3.5. On 61 + 120 events.
* **It costs the detector that already passes.** PDVD banked Michel efficiency +0.054 in doc 103 sec 12; L2b hands
  all of it back (+0.000) to fix a regression PDVD does not have.
* **It was found by scanning the same data it would be graded on.** A pre-registered threshold does not protect
  against that. Adopting it would need its own pre-registration and an out-of-sample grade (amendment 7 sec 7).

---

## 3. Where A1's "Michel" charge comes from (`figs/104_michel_provenance.txt`)

A0 and A1 read byte-identical pctrees — `d102_run_arms.sh` stages both with `readlink -f` onto the same
`*_d51hclus` tarballs — so their 3-D points are in the same frame and directly comparable. For each candidate, take
A1's Michel-member points (`T_stm_michel_pts`, `role == 3`) and measure the distance to **A0's muon chain points**
(`role == 1`) of the same cluster. If A1's "Michel" sits on A0's muon, it is charge A0 fitted as muon.

### 3.1 The six

| key | Michel pts | median dist to A0 chain | within 0.6 cm | median dist to A1's *own* chain | muon_len A0 → A1 |
|---|---|---|---|---|---|
| `028084_10/109` | 3 | **0.09 cm** | **100 %** | 0.14 | 589.5 → 578.5 |
| `029107_28/109` | 33 | **0.27 cm** | **76 %** | **9.09** | **595.9 → 544.8** |
| `029107_12/95` | 30 | 1.92 cm | 17 % | 3.67 | 158.5 → 142.8 |
| `029107_27/39` | 25 | 3.88 cm | 16 % | 5.84 | 62.1 → 40.2 |
| `029107_19/111` | 2 | 13.93 cm | 0 % | 0.12 | 108.6 → **121.8** |
| `029107_23/41` | — | conn 3, no fitted points | — | — | not a candidate in A0 |

`029107_28/109` is the clearest case in the set: the muon loses **51 cm**, and 33 points that lie a median 0.27 cm
from where A0 fitted muon — but 9.09 cm from A1's own chain — come back as a 19.4 cm, 64.1 MeV, 1.26 MIP "Michel".
That is not a Michel electron; it is the muon, and the trajectory stopped short of it.

### 3.2 The control, which is what makes this readable

A true Michel is attached at the stop, so being *near* the muon proves nothing. The control is the 50 true Michels
**both** arms tag: median containment **5 %**, with 2 of 42 measurable items at 50 % or more, against **2 of 5** in
the false-positive class. Elevated, clearly — and equally clearly **not a veto**: one true Michel (`028084_17/55`,
4 points) is 100 % contained. Containment is a diagnostic, and in any case A0's chain does not exist when A1 runs.

### 3.3 What this means, and what is deliberately not done about it

The defect is upstream of the Michel builder: **the fitted trajectory ends short of the muon's own charge, and the
orphaned tail is offered to the Michel logic, which correctly finds charge there.** This matters beyond the Michel
grade — it is the same end region the stopping-muon dQ/dx of docs pdhd/16, 17, 29 and 50 is measured in, and the
owner's `own103h` note *"end track was not found correctly, so the dQ/dx is off near the end"* is that defect seen
from the other side.

**An end-reach fix is not attempted in this round.** Moving the stop point moves `is_stm`, moves the stopping-muon
dQ/dx those four docs rest on, and moves PDVD production, which shipped on 2026-09-15. It is its own campaign with
its own gate, and scoping it inside a round whose stated goal was an admission knob would be dishonest about the
blast radius. Sec 7 names the cheap first step.

---

## 4. Why the admission layer cannot fix this

The Michel builder is doing its job. Given a trajectory that ends 51 cm early, there *is* charge attached at the
claimed stop, it *is* about a MIP, and it *does* turn by 31°. Every quantity the gate sees is telling it the truth;
the input is wrong. A cut tightened enough to reject this case rejects real Michels with the same measurements —
which is exactly what sec 2's four levers show numerically.

This is the general shape: **a false positive whose provenance is wrong, rather than whose features are wrong,
cannot be cut away in the layer that reads the features.**

---

## 5. The one-sided review, and amendment 7 (`figs/104_pred_amend7.txt`)

The PDHD grade compares a reviewed side against an unreviewed one. From doc 103's amendment-2 split (c):

| Michel movers | n | owner-judged | false positives |
|---|---|---|---|
| tagged in **A1** only | 33 | **8** | 6 |
| tagged in **A0** only | 21 | **0** | 1 |

`own103h` was *built* as a false-positive adjudication queue (amendment 2 sec 2), so the one-sidedness is by
construction and the cost landing on the reviewed subset is expected — the untouched-population split reads Michel
purity −0.011 on 179 of 196 items, which is a **sensitivity signal, not a rescue**. What is genuinely not
established is what the same reviewer would do to the A0 side, and `figs/103_d1_margin.txt` puts PDHD Michel purity
**3 labels** from passing.

**Amendment 7** (sha256 `321360daea6e18f721f2e7b206b202b5f56fac7304a6f798f3f75a8dfe1415ad`, frozen
2026-09-15T18:47:32−07:00, before the set was drawn) applies amendment 6's remedy to PDHD, Michel only —
both `is_stm` metrics already pass, so no `is_stm` mover is drawn. It is symmetric in effect: a relabel-negative in
stratum A1 **adds an A1 false positive** (the delta gets worse), in stratum A0 it **adds an A0 false positive** (the
delta gets better). It can move the grade either way, and it may simply confirm D2.

**The set `own103h2`** (`figs/104_own103h2_set.txt`): 39 Michel TP-movers, strata A1 23 / A0 16; 10 drawn from each
with `random.Random(107)`, plus 4 controls from a pool of 45; 24 items, shuffled, shown on the A1 payload where the
item is an A1 candidate (20) else A0 (4), behind the `own103h` question panel with no chain answer, prior label or
stratum on screen. Drawn keys live only in `items.tsv` and were not read before serving.

**The scorer was written before any label existed**, as amendment 6 required of its PDVD twin, and fed every prior
label back unchanged it reproduces the committed headline exactly — A0 121/2/76 and 70/4/46, A1 118/4/79 and
77/9/39, deltas −0.017 / −0.015 / −0.051 / +0.060, reading D2, every rate zero. So the scorer is neutral under
identity, and anything it reports later is the owner's labels and not the fold.

`d104_tpmover_score.py` folds first and then asks the grader what the fold did: an item is relabel-**negative** if
it is still in the post-fold Michel population with `m_truth` false, and **removed** if it left that population at
all. One definition — the grader's — so the rates and the grade can never disagree. This matters more on PDHD than
it did on PDVD, where a Michel is simply the verdict `STM_MICHEL`; on PDHD it is a stopper whose `michel_kind` is
attached or both, with amendment 5's rule that an owner `STM_ONLY` carrying no kind is Michel-negative.

**Served** on port 5017 (`stm_michel_viewer.py --det pdhd --tag own103h2`), label shas recorded beforehand in
`/home/xqian/tmp/d104/label_shas_before_own103h2.txt`.

---

## 6. The decision

**PDHD is not flipped in this round.** D2 stands on Michel purity until the symmetric check is read. Amendment 7
sec 6: D1 only if every one of the four metrics is ≥ A0 − 0.020 on **both** the folded grade and the projection,
and a pass is reported with its numbers — it is not itself authority to change a production default.

The three options doc 103 sec 12.5 put to the owner are unchanged, now with sec 2 and 3 under them:

1. **Hold PDHD at production** until the check is read — where this round leaves it.
2. **Accept the trade** — Michel purity 0.946 → 0.895 for Michel efficiency 0.603 → 0.664, with both `is_stm`
   metrics passing. Sec 3 sharpens what is being accepted: some of those false positives are truncated muons, and
   the same truncation corrupts the stopping-muon dQ/dx whether or not a Michel is tagged.
3. **A geometric admission rule** — refuted by sec 2 for the persisted geometry. What is left is sec 7.

If the check turns the grade, the flip unit is `cfg/pgrapher/experiment/pdhd/pr.jsonnet:1408` `'stepped'` →
`'charge_stepped'` plus `fit_weight_pow` 1.5 / `assoc_cont_center` 1 into `pdhd_track_fitting.json`, with the PDVD
treatment: a proof arm, `d103_flip_gate.py --det pdhd`, and the `'stepped'` escape hatch verified byte-identical.
**Do not reuse doc 102's `102r2_flip.patch`** — it flips both detectors.

---

## 7. Round 8: the one discriminator not yet refuted

Sec 2's levers all ask *what does the attached piece look like*. The question sec 3 says to ask instead is **where
is the muon's Bragg rise — at the claimed stop, or beyond the far end of the candidate arm?** For a truncated muon
it is beyond; for a real stopper with a Michel it is at the stop. L4 is the degenerate version of this (contrast at
the stop only), and its failure does not refute the comparison.

That comparison is not persisted today, and the cheap first step needs no C++: **`survey_enable` and
`publish_other_arms` are pure-writer knobs** — they add `rej` / `d_stop` / `d_body` columns to
`T_stm_michel_pts` and role-7 rows for the rejected arms without changing a verdict. One PDHD arm with both on
yields the rejected-arm population and the per-piece distances, and the `stop-arm:` DEBUG line
(`CheckSTM_Michel.cxx:3492-3501`) additionally carries `shower`, `terminal`, `kink5` and `far_full`, which are the
four gate inputs that exist nowhere in the output. Only after that is there a basis for a discriminator.

The end-reach defect of sec 3.3 is the larger fish and is deliberately left whole: it needs its own doc, its own
gate on `is_stm` and the stopping-muon dQ/dx, and its own PDVD re-grade.

---

## 8. Not concluded

* Whether the symmetric check turns PDHD's grade — the set is served, unlabelled at the time of writing.
* Whether the sec 3 truncation is driven by the fit knobs or the sampler. Doc 103's per-lever Michel purity (K
  0.880, S 0.906, A1 0.895) points at the **fit knobs** as the larger contributor, but the arms were never read at
  point level per lever.
* How much of the stopping-muon dQ/dx of docs pdhd/16, 17, 29, 50 moves under the truncation. Sec 3 measures the
  trajectory, not the dQ/dx.
* Whether `029107_19/111` (muon *grew* 13 cm, Michel 14 cm away) and `029107_23/41` (charge-only, not an A0
  candidate) share any mechanism at all with the truncation cases, or are two more separate stories.

---

## 9. Files

| path | what |
|---|---|
| `scripts/d104_michel_features.py` | the admission feature table by `conn_type`, both detectors; shared loader for the other two |
| `scripts/d104_lever_sizing.py` | the four levers, A1-only framing, with the V1 assertion and the L2b anatomy |
| `scripts/d104_michel_provenance.py` | the point-level containment test and its both-arms control |
| `scripts/d104_tpmover_set.py` | the `own103h2` draw (fork of `d103_tpmover_set.py`; Michel only, seed 107) |
| `scripts/d104_tpmover_score.py` | the fold, rates, projection and reading (written before any label) |
| `figs/104_michel_features.txt` | sec 2.1 |
| `figs/104_lever_sizing.txt` | sec 2.2, 2.4 |
| `figs/104_michel_provenance.txt` | sec 3 |
| `figs/104_pred_amend7.txt` + `.sha256` | amendment 7, frozen before the draw |
| `figs/104_own103h2_set.txt` | the set counts and the identity smoke test |
| `pdhd/docs/scan/pdhd_stm_michel_own103h2_verdicts.json` | the owner record (written when the labels land) |
