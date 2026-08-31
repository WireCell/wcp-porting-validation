# doc pr/136 — the EM-clustering / π⁰ charge-attribution campaign: what I propose to try, and why

**Status: CHARTER + ROUND 1 MEASUREMENT, 2026-08-30. §1–§9 are the charter
(analysis only). §10 is round 1: one byte-neutral probe arm at the production
point, and the census it made possible — which closed proposals #3 and #6 and
replaced #1 with a smaller, sharper target. Still no knob, no flip. Successor to the finder-level π⁰ campaign (pr/126 → 132 →
133 → 134, reviewed in pr/135), which is closed and shipped. Scoped by the
owner: tune only what comes AFTER the neutrino vertex, and measure against
the hand scan. toolkit `b5cc3a3f`, wcp `38088acd`.**

## Repro

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# §3, §4 — the absolute π⁰ mass-closure metric on the 0.86 production arms
scripts/pr136_mass_closure.py \
  --manifest98  em117-134f08698-manifest.tsv \
  --manifest141 em114c-134f086141-manifest.tsv \
  --overlay-tag pi0scan-0829-agent --fudge 0.86 \
  --tsv docs/pr/pr136-mass-closure.tsv          # -> pr136-mass-closure.txt

# §5 synthesis -- the SAME metric on the pr130 arm, so the join is single-arm
scripts/pr136_mass_closure.py \
  --manifest98  em117-pr130q98-manifest.tsv \
  --manifest141 em114c-pr130q141-manifest.tsv \
  --overlay-tag pi0scan-0829-agent --fudge 0.80 \
  --tsv docs/pr/pr136-mass-closure-pr130arm.tsv

# §5 — EM charge attribution vs the 90 hand-marked showers (pr130r1-probe arms)
cd em_display
./em117_score.py --tag emscan-0827        --manifest em117-pr130q98-manifest.tsv \
    --prepdir emprep-pr130q98  --tsv ../docs/pr/pr136-completeness-98.tsv
./em117_score.py --tag emscan-0828-agent5 --manifest em114c-pr130q141-manifest.tsv \
    --prepdir emprep-pr130q141 --tsv ../docs/pr/pr136-completeness-141.tsv
cd ..
scripts/pr136_completeness.py --tsv docs/pr/pr136-completeness-pr130arms.tsv

# §6.1 — irreducible shower leakage out of the active volume (owner's point)
scripts/pr136_containment.py \
  --manifest98  em117-134f08698-manifest.tsv \
  --manifest141 em114c-134f086141-manifest.tsv \
  --tsv docs/pr/pr136-containment.tsv

# §6.2 — the charge budget: is the missing charge in the event at all?
scripts/pr136_deficit_budget.py \
  --manifest98  em117-134f08698-manifest.tsv \
  --manifest141 em114c-134f086141-manifest.tsv \
  --tsv docs/pr/pr136-deficit-budget.tsv
```

Prior art quoted rather than re-derived: `126_pi0-audit-and-em-charge-scale.md`,
`132_pi0-reco-round1.md`, `133_pi0-muon-showers-nc-signature.md`,
`134_pi0-nc-fragment-merge-p1-pf.md`, `135_pi0-campaign-review.md`,
`pr130-qmiss-mechanism.md`, `pr130-qmiss-scan-decisions.md`.

---

## 1. Scope — what the owner's constraint does and does not remove

Owner, 2026-08-30: *"we should not tune the neutrino vertex. So the part that
we should tune is only after that step."* and *"The metric can be with my hand
scan results."*

**The precise edge, so nothing reads as a violation.** "Do not tune the
neutrino vertex" means do not work on vertex **finding** —
`determine_overall_main_vertex` (`clus/src/TaggerCheckNeutrino.cxx:2941`) and
everything upstream of it, which includes imaging, track/shower separation and
per-cluster vertex fitting. It does **not** make the vertex *object*
untouchable: K24 (`m_pi0_prefer_main_vertex`), `m_pi0_bp_vertex_miss_cm` and
`id_pi0_without_vertex`'s `main_vertex->fit().point` mutation all ship in SBND
production today with owner approval, because they re-seat a **π⁰ decay point**
after the ν vertex is fixed. That distinction is the boundary this campaign
works inside.

**What that leaves** is exactly the two halves the owner named. Everything
from `TaggerCheckNeutrino.cxx:3081` (`examine_direction`) through
`shower_clustering_with_nv` to the two π⁰ finders — twenty-three named stages,
mapped in §5 — is in scope. Roughly 6 000 lines of `NeutrinoShowerClustering.cxx`.

**Two corrections to pr/135 §9, which chartered this campaign differently.**

1. **MC truth is out.** pr/135 §9 proposed MC truth as the calibration handle,
   on the argument that hand labels carry pairings and not absolute energies.
   The owner overruled it. That argument was also half wrong: the labels carry
   the pairing, and **physics carries the mass**. §3 shows the pair supplies an
   absolute metric with no truth information at all.
2. **"Upstream charge deficit" was the wrong name.** pr/135 §4 blamed
   "upstream imaging/shower-building". Imaging sits *before* the vertex, so the
   owner's constraint cuts it out entirely. What remains is charge
   **attribution** — which reconstructed object owns which reconstructed
   charge — and that is a post-vertex question end to end. The campaign is
   renamed accordingly.

---

## 2. Why the metric we have been quoting is blind

The γ ledger (`scripts/pr132_gamma_ledger.py`) compares a production
`kine_charge` against the hand label's `energy`. Both are **reconstructed**
energies: the label's was copied from the scan-time reconstruction. A shower
that lost its downstream tail in *both* scores OK. Measured on the 132 hand
γs at the 0.86 production point:

- `E_prod/E_lab` has **median 0.930**, which is exactly `0.80/0.86` — the EM
  scale flip and nothing else. **66 % of γs sit within ±3 % of that value.**
- Labels were frozen at `kine_shower_fudge_factor = 0.80`; the ledger never
  received the correction its own sibling carries
  (`pr132_pi0_census.py:100`, `scale = 0.80/a.fudge`). So at 0.86 the nominal
  OK band `[0.80, 1.25]` is a **true-completeness band of `[0.86, 1.34]`**:
  a shower losing 14 % of its charge is scored OK, and a *deficit* campaign is
  reading a metric biased to hide exactly that.

Three further defects, each independently disqualifying:

- **Matching is by `showers[].id` with no membership cross-check** — and that
  id is a *constituent segment id*, not a stable object handle. A re-split can
  preserve the id while changing composition arbitrarily. Live example on the
  f086 arm, event **347824**: one re-split object is reported as one **ABSENT**
  (label g1 528.0 MeV, no shower with that id) plus one **OVER 1.32**. The
  discriminators `num_segments` and `total_length` sit unread in the same record.
- **The UNDER+SIBS sibling cone is drawn in the label frame** (20°, no pdg
  filter). doc 132 §12.3 already measured that the same fragments sit at
  **36–159° in the reco frame** while the label frame sees 2.6–8.5°. Event
  259542 g2: `e_lab` 138.7 MeV against `e_sibs` **1014.9 MeV**.
- **24 % of the denominator carries no scanner judgement.** 32 of the 132 γs
  come from the `pi0scan-0829-agent` overlay, which has no `em` block, no
  `members` and no marks; their `e_lab` is pure `energy_as_reconstructed` off a
  0.80 arm. **31 of the 32 read exactly 0.93** — the scale ratio — and they are
  counted in the "119/132 OK" headline.

**Conclusion: "90.2 % γ ledger OK" is not a completeness statement**, and no
ranking in this campaign should rest on it. Fixing the ledger is proposal #5.

---

## 3. The metric this campaign uses — π⁰ side: absolute mass closure

Hand labels supply the pairing and the opening angle. **Physics supplies
m_π⁰ = 134.9768 MeV.** That is an absolute anchor requiring neither MC truth
nor label energies, so the residual it measures is a real reconstruction error.
Per hand pair, at the production energies:

```
m_prod / m_π⁰  =  R · A
   R = E_tot · sin(θ/2) / m_π⁰       ENERGY CLOSURE
   A = 2·√(f(1−f)),  f = E1/E_tot    SHARING ASYMMETRY  (A ≤ 1)
```

Since `E1·E2 ≤ E_tot²/4`, the mass obeys `m ≤ E_tot·sin(θ/2)`. **R < 1 is
kinematically impossible for a real π⁰**: no division of the measured energy
reaches the π⁰ mass at the measured angle. R and A separate *"the pair is
missing total charge"* from *"one γ is starved relative to the other"* — a
distinction no ratio-against-a-label can make.

**The angle is exact, not estimated.** `theta_vertex_convention` is the angle
between the two vertex→conversion-point rays. Any point on a γ's line of
flight gives the same direction from the decay vertex, so a start point
reconstructed too deep does **not** bias θ. The convention fails only when the
π⁰ did not decay at the labelled vertex, which biases θ *low* and therefore R
low — flagged per event, never corrected.

### The measurement (56 pairs with both γs matched, f086 production arm)

| class | n | median R | median A | reading |
|---|---|---|---|---|
| m_prod < 100 MeV | 11 | **0.57** | 0.94 | near-symmetric sharing ⇒ a **pure total-energy deficit**, ~43 % of the π⁰ energy absent |
| 100 ≤ m ≤ 160 | 37 | 1.14 | 0.87 | inside the acceptance window |
| m_prod > 160 MeV | 8 | **1.98** | 0.78 | **over-clustering** — 396222 carries a 2658 MeV "γ" |

- **19 of 56 pairs (34 %) have R < 1.**
- Of the 27 pairs below 135 MeV, **19 are killed by total energy and only 8 by
  asymmetry.** The residual is a two-sided *charge attribution* problem.
- R distribution: q10 0.57 · q25 0.94 · median 1.13 · q75 1.35 · q90 1.80.

The high tail is as large a defect as the low one and has never been worked:
**median R = 1.98 on the 8 pairs above 160 MeV** means those "γ"s carry about
twice the charge a π⁰ decay allows.

---

## 4. The residual, re-triaged — is the missing charge *there*?

The hand labels already contain a per-segment missing-charge ledger nobody has
used as a metric: `em.marks_by_shower` = `{shower_id: {segment_id: "in"|"out"}}`
— the scanner's own verdict that reco left a segment out, or wrongly holds one.
Corpus-wide: **80 of 214 EM-scanned events carry marks — 90 marked showers,
305 segments marked IN, 117 marked OUT.**

Adding that marked charge to the production energy (converted to the arm's
scale by ×0.9302) asks the campaign's prior question directly.

| event | R_prod | R with hand marks | m_prod → m_marks | verdict |
|---|---|---|---|---|
| **342199** | 0.69 | **1.01** | 92.6 → **127.0** | RESCUED |
| **409634** | 0.75 | **1.11** | 77.4 → **148.8** | RESCUED |
| **54341** | 0.87 | **1.40** | 63.7 → **131.2** | RESCUED |
| **54332** | 0.94 | **1.12** | 109.0 → **122.4** | RESCUED |
| 168432 | 0.42 | 0.44 | 53.5 → 56.5 | marks move it, still impossible |
| the other 14 | 0.18–0.98 | unchanged | — | no marks at all |

**4 of 19 impossible pairs are healed by charge the scanner explicitly says
belongs to the γ.** Honouring every hand mark moves the in-window count 37 → 39
of 56. And 342199 is the very specimen pr/135 §5-4 named for the
merge-before-accept round — confirmed here by a completely independent route.

**The other 15 gain nothing** (14 carry no marks at all). Their deficit is not
mis-assigned charge the scanner saw, so they need a different explanation
before any lever is proposed for them. §6 supplies it.

### Corrections to the inherited numbers — carry these forward

- **The γ ledger residual at 0.86 is 13, not 12.** Add **173093 g1, ratio 0.80,
  38.7 MeV in 1 sibling** to UNDER+SIBS, making it 4. It sits at 0.82 (OK) on
  every 0.84 arm and crosses only at 0.86. **It is named in no doc.**
- **The outside-window 14 at 0.86** = doc 133 §12.2's list **− {486907,
  103798} + {283713, 397630}**. 486907 crosses in at 0.86 (171.6 × 0.930 =
  159.6); 103798 became exact.
- **"50–75° off the flight line" is n = 1** (105946/55063, doc 134 §9, prose
  only, no TSV). The defensible population version is doc 132 §12.3: unfired
  fragments at **36–159° in the reco frame vs 2.6–8.5° in the label frame**.
- **doc 135 §8.4's "θ_prod − θ_hand median +0.0°" is not reproducible** — no
  script, no TSV, the identity of its 5 outliers lost. The surviving instrument
  is `pr132-angle-census-r10ang.tsv` (66 pairs, `a_start` vs `a_label`), and it
  disagrees in the cases that matter: 105946 records a_start **17.8°** against
  a_label **72.7°**.
- Doc arithmetic: pr/135 §4 says "13 ledger γs" but itemises 12; §9 says 33
  misses where §4's table sums to 34.

**Out of reach by the owner's own constraint**, and named so it is not
re-attempted: **54332, 76346, 54453** are the three events his q_miss scan
marked *"not scannable — ν vertex wrong"*. Their blocker is vertex finding.

---

## 5. The metric this campaign uses — EM side, and what it says

`em_display/em117_score.py` already implements the right scorer against the
hand marks and the ledger never used it: per marked shower,
`target = (members | marked-in) − marked-out`, charge-weighted against what the
arm actually gave that shower, yielding `q_comp`, `q_pur`, their harmonic mean
`q_f1`, and the raw `q_miss` / `q_extra`. Cross-run matching is by
charge-weighted overlap, so a re-rooted shower is not scored as a total miss —
the failure mode that makes the γ ledger emit ABSENT.

**Operating-point caveat, stated up front.** These numbers are scored on the
`work-pr130r1-probe*` arms, the newest that carry an `emprep-*` membership
sidecar — i.e. **before** the NC chain, K24 and the 0.86 scale. They describe
the pr/130 point, not today's production. Refreshing them at f086 costs one
byte-neutral probe arm and is this campaign's first recommended action
(proposal 0). The sidecar is not optional: the dump's `segments[].shower_id` is
single-valued, so a segment held by two showers is credited to one and the
lossy join invents misses that are not there (8 and 10 members on these sets).

### The measurement (90 hand-marked showers over 80 events)

| quantity | value |
|---|---|
| sum `q_target` | 4.83e8 |
| **`q_miss`** (UNDER — charge the scanner says belongs, the shower does not hold) | **7.14e7 = 14.8 %** |
| **`q_extra`** (OVER — charge the shower holds, the scanner says it should not) | **4.31e7 = 8.9 %** |
| charge-weighted F1 per shower | median **0.907**, mean 0.876, min 0.322 |
| F1 < 0.90 / < 0.80 / < 0.50 | 41 / 21 / 1 of 88 |

**The error is not one-sided:** pure UNDER **22** · pure OVER **29** · BOTH
**30** · clean **9**. Over-clustering is at least as common as under-clustering,
which is the second reason the "tail deficit" framing needed replacing.

### The synthesis — the two halves name the same events

Joining the EM score to the π⁰ mass closure (`pr136_completeness.py`), every
pair the hand marks rescue is an EM-clustering failure by an independent metric,
and the pairs the marks cannot rescue are **not**:

| event | q_f1 | q_comp | R_prod | m_prod | R_marks | |
|---|---|---|---|---|---|---|
| **342199** | 0.670 | **0.504** | 0.688 | 92.6 | **1.009** | rescued |
| **54341** | 0.673 | **0.507** | 0.873 | 63.7 | **1.400** | rescued |
| **54332** | 0.723 | **0.573** | 0.938 | 109.0 | **1.124** | rescued |
| **409634** | 0.867 | 0.765 | 0.749 | 77.4 | **1.109** | rescued |
| 397630 | 0.704 | 0.543 | 0.400 | 54.0 | 0.400 | still impossible |
| 281485 | 0.753 | 0.604 | 0.513 | 62.5 | 0.513 | still impossible |
| **499577** | 0.826 | **1.000** | 0.921 | 111.7 | 0.921 | still impossible |
| 347129 | 0.886 | 0.795 | 0.976 | 122.0 | 0.976 | still impossible |
| **168432** | 0.932 / **1.000** | 0.898 / **1.000** | 0.423 | 53.5 | 0.439 | still impossible |

**This is the sharpest result in the round, and it is verified on a single
arm.** The table above joins `q_comp` (pr130 arm) to `R_prod` (f086 arm), so it
was re-run with the mass closure computed on the **pr130 arm itself**
(`pr136-mass-closure-pr130arm.tsv`; that arm's EM scale is 0.80, measured — see
below). The conclusion survives unchanged: **168432 holds everything the scanner
says it should — `q_comp = 1.000` on one of its two marked showers, 0.898 on the
other — and its π⁰ mass reaches only 57.5 MeV, R = 0.455.** 499577 is the same
shape but marginal (`q_comp` 1.000, R 0.991) and should not be leaned on.

**Completeness by the scanner's own standard does not imply kinematic closure.**
For 168432 the missing energy is not mis-attributed reconstructed charge, so no
post-vertex clustering change can reach it — and, conversely, a perfect
clustering score does not bound the physics error. Both metrics have to be
quoted; neither substitutes for the other.

**How much of the arm-to-arm difference is the EM scale alone.** Comparing the
two mass-closure runs pair by pair, `R(0.86)/R(0.80)` has median **0.9303**
against the 0.9302 predicted by pure rescaling, and **50 of 56 pairs sit within
±5 % of it**. Only six changed composition — 54332, 165157, 54341, 99838, 47212,
71872 — and every one of them is in the π⁰ chain's own mover set. 168432 and
499577 are not among them.

### Two verified mechanisms, both post-vertex, neither with a knob today

1. **Direction is a membership centroid.** `shower_cal_dir_3vector`
   (`clus/src/PRShowerFunctions.cxx:132`, default `dis_cut = 15 cm`) returns
   `centroid(member fit points within dis_cut of p) − p`. A shower missing its
   downstream tail has its centroid pulled upstream, so its direction rotates
   and every angle cut downstream reads the rotated value. There are ~30
   bare-literal radii in the chain (12 / 15 / 30 / 50 / 60 / 100 cm) and **no
   `m_*` knob on any of them.**
2. **The absorber's cone width is keyed to the shower's own energy.**
   `clus/src/NeutrinoShowerClustering.cxx:5001-5010` admits a segment at 30° if
   `Eshower > 800 MeV`, 25° above 360, 15° above 250, 10° above 150, with
   `Eshower = kine_best ?: kine_charge` (`:4852`). **A shower already short of
   charge falls into a narrower cone and absorbs less** — a self-reinforcing
   deficit, and there is no knob on the ladder.
   **Corollary worth the owner's attention:** `kine_shower_fudge_factor`
   *divides* `kine_charge`, so the 0.84 → 0.86 flip lowered every `Eshower` by
   2.3 % and can push showers across the 150 / 250 / 360 / 800 MeV tier edges.
   **The EM energy-scale constant feeds back into clustering acceptance.**
   pr/135 §10 attributed the four moved events to acceptance-window edges alone;
   this is a second, untested channel.
   **It is not yet evidence.** The 0.80 → 0.86 comparison above shows the count
   of kinematically impossible pairs rising 16 → 19, but that is exactly the
   arithmetic of `R ∝ 1/fudge` (pairs between R = 1.00 and 1.075 fall below the
   bound), and composition moved on only 6 pairs, all of which the π⁰ chain
   itself touched. So the arm comparison neither supports nor refutes the
   feedback hypothesis — it has to be tested on proposal 0's absorb tape,
   by counting showers within a few MeV of a tier edge.
3. The coupling that makes both bite: `kine_charge` is **not** a sum over
   members — it is 2D charge integrated within **0.6 cm** of the shower's own
   point cloud (`clus/src/NeutrinoEnergyReco.cxx:127-188`). Membership → energy
   is mechanical. A segment not absorbed is charge not counted, always.

(For the record, so it is not re-proposed: `kine_charge_dedup` and
`kine_charge_rebuild` are **already ON in SBND production**,
`wct-pr-perevt.jsonnet:1959-1960`. They refresh the *reported* energy after
late growth; they do not change what the mid-pipeline gates saw.)

---

## 6. The scope boundary — how much of the residual is reachable at all

### 6.1 Irreducible leakage: energy that was never in the detector

Owner, 2026-08-30: *"for pi0 it is possible part of the gamma from the pi0
decay go out of the detector. Since we can only reconstruct what is in the
detector, even if we have perfect clustering, we may still miss significant
energy leading to lower pi0 mass reconstruction."*

This is correct and it was missing from the triage above. §3's R < 1 test
cannot distinguish a γ whose tail an absorber dropped from a γ whose tail
left the TPC — both read as missing energy. **A campaign that does not
separate them will spend a round chasing charge that was never recorded.**

`scripts/pr136_containment.py` separates them. Each γ develops from its
conversion point along its shower axis; `D` is the distance from that point to
where the axis leaves the active volume (**x [−202,202], y [−200,200],
z [0,500] cm, measured from the reconstructed point cloud, not assumed**), and
the contained fraction is the PDG longitudinal profile
`f = P(a, bD/X₀)` with LAr X₀ = 14.0 cm, E_c = 32.8 MeV, b = 0.5, and the true
energy unfolded by two fixed-point iterations. Transverse leakage (Molière
radius ≈ 9 cm), dead channels, and γs converting outside the volume are **not**
modelled — each would make leakage larger, so every `f` here is an upper bound
on containment.

**Leakage is not a rare edge effect in SBND.** Over the 112 hand γs: median
`f` = 0.997, but **29 γs (26 %) have f < 0.95 and 14 have f < 0.75**; the 10th
percentile of available depth is only 58 cm ≈ 4 X₀. **24 of the 56 pairs
(43 %) have at least one γ leaking.**

| the 19 impossible pairs | n | events |
|---|---|---|
| **fully contained (both f ≥ 0.95) — the deficit is NOT leakage** | **11** | 342199, 409634, 54341, 54332, 103798, 176986, 281639, 499577, 283713, 347129, 392901 |
| leakage alone explains R < 1 | 4 | 168432, 281485, 242726, 169356 |
| leakage helps but does not close it | 4 | 71178, 397630, 280159, 280972 |

**Every one of the four pairs the hand marks rescue is fully contained.** The
two measurements were built independently and they agree on the target list —
that is the strongest internal consistency check in this doc.

**A leakage correction must NOT be applied to the mass.** Correcting every pair
moves the median from 135.5 to **142.9 MeV** and *drops* the in-window count
37 → 34. That is the signature of double counting: `kine_shower_fudge_factor`
was fitted (0.80 → 0.84 → 0.86) so the *measured* peak sits at 135, so **it
already absorbs the sample-average leakage**. Leakage explains the
event-to-event **spread**, not the mean. This table is a classifier, never a
correction.

**But that has a consequence for the energy scale, and it is testable now:**

| subsample | n | median m_prod | median R | in-window |
|---|---|---|---|---|
| all pairs | 56 | 135.5 | 1.13 | 37 (66 %) |
| **both γs contained (f ≥ 0.95)** | **32** | **136.8** | 1.12 | 23 (72 %) |
| at least one γ leaking | 24 | **128.1** | 1.24 | 14 (58 %) |

The full-sample median sits at 135 partly by **cancellation**: contained pairs
run +1.3 % high, leaking pairs 5 % low. So (a) the 0.86 fudge is already good
to ~1 % on well-contained showers and this campaign should not chase the scale;
and (b) **the mass-peak fit should be repeated on the contained subsample**,
because a fudge fitted on the blend is a detector-geometry constant masquerading
as a calorimetric one — it will not transport to a different fiducial cut or a
different sample.

**Caveat on the geometry, stated because it is large:** 32 of 56 pairs have a
shower start→end axis more than 30° off the vertex→start ray. Some of that is
the end point being a poor axis proxy for a wide shower, and some is doc 132
§12.3's deficit-biased internal direction. For those pairs `D` is a rough
number; the classification is robust at the extremes (`f` < 0.5 or `f` > 0.95)
and soft in between.

### 6.2 The charge budget: is the missing charge in the event at all?

Before proposing a lever, measure whether the charge exists. For each
impossible pair, the deficit is `ΔE = E_tot·(1/R − 1)`; the budget it could be
drawn from is **ORPHAN** (segments held by no shower) + **OTHER** (segments in
other showers of the event — pr/130's SPLIT + STOLEN pool). Priced with an
in-event dQ→MeV constant taken from the two labelled γ showers, which is an
**upper bound**, since much of the OTHER budget is track-like charge an EM
absorber must never take.

**The error bar on `k`, and why it changes the answer.** §5 establishes that
`kine_charge` is *not* a sum over member dQ — it is a 2D integration within
0.6 cm of the shower's own cloud, with plane weights and a possible max-plane
drop. So `k` is a ratio of two differently-computed quantities and its direction
of error is not known a priori. The two γs of each pair give two independent
estimates: their spread is **median 1.34×, worst 2.62×**. Every budget is
therefore quoted as a range `[q·k_lo, q·k_hi]`, and a pair whose verdict flips
inside that range is **indeterminate, not excluded**.

| verdict | n | events |
|---|---|---|
| rescued by hand marks | 4 | 342199, 409634, 54341, 54332 |
| REACHABLE — even the pessimistic `k` covers the deficit | 9 | 281485, 280159, 280972, 499577, 283713, 242726, 169356, 347129, 392901 |
| INDETERMINATE — the `k` spread straddles the answer | 3 | 397630 (1.01×), 176986 (0.94×), 281639 (1.01×) |
| **EXCLUDED — the deficit exceeds even the optimistic budget** | **3** | **71178** (needs 8.4× its whole budget), **168432** (1.4×), **103798** (1.3×) |

So of the 19 impossible pairs: **4 are demonstrably fixable inside scope, 3 are
demonstrably outside it, 3 are indeterminate, and 9 are open** — and for most of
those 9 the demand is ~1 % of a budget that is mostly track charge, so
"reachable" there means "not excluded", not "a lever exists".

### 6.3 The two boundaries combined — the campaign's actual target list

| | n | events |
|---|---|---|
| **TARGET — contained AND the charge exists in the event** | **9** | **342199, 409634, 54341, 54332** (hand-marked), 176986, 281639, 499577, 283713, 347129, 392901 (weak demand) |
| irreducible: leaking out of the TPC | 4 | 168432, 281485, 242726, 169356 |
| contained but the charge is not in the event either | 1 | 103798 |
| geometry/label problem, not charge | 2 | 71178 (R 0.18 with both γs largely contained — a 158 MeV π⁰ must open ≥ 117°, it is labelled at 18°), 397630 |
| leakage plus something else | 2 | 280159, 280972 |

**The four hand-marked pairs are the only ones with a named, attested,
in-detector lever.** That is the honest denominator for proposal #2.

**71178 deserves a label audit rather than a reco fix.** Its two γs total
158 MeV at θ = 18°. A 158 MeV π⁰ must open at least 117°. Getting from 18° to
117° is not a charge correction — either the pairing is wrong or the π⁰ decayed
far from the labelled vertex. Three events in this class already have that
diagnosis on the record from the owner's own scan (54332, 76346, 54453,
"ν vertex wrong").

**Realistic ceiling for the whole campaign, stated before any work starts:
+4 to +6 exact π⁰ out of 66**, plus whatever the EM-side `q_miss`/`q_extra`
numbers buy in event-level energy that the π⁰ census does not see. And that is
an upper bound on an upper bound in two independent ways: **a pair entering the
(100,160) window is necessary but not sufficient for a census "exact"** —
production still has to *prefer* that pairing over every rival at every
candidate vertex — and **43 % of the sample has a γ leaking out of the TPC**,
which no amount of clustering work recovers. Anyone expecting the π⁰ exact rate
to move from 32/66 to 45/66 should read §6 first.

---

## 7. Ranked proposals

Each carries the four fields that matter: **mechanism · specimens · the
measurement that would kill it · ceiling.**

### Proposal 0 (enabling, do first) — one byte-neutral probe arm at f086

Everything in §5 is measured at the pr/130 operating point because that is the
newest arm with a membership sidecar. One arm on the 239-event manifest at the
current config with `WCT_SHOWER_ABSORB_DEBUG` + `WCT_SHOWER_XCLUS_DEBUG` +
`WCT_SHOWER_CONTENT_DEBUG`, parsed by `prep_em_scan.py --parse-probes` into
`emprep-136f086`, makes every number in this doc current and supplies the
absorb/reject tape the rest of the ranking needs. Probes are stderr-only and
byte-neutral by construction. **Nothing below can be ranked honestly without it.**

### #1 — Un-park the pr/130 q_miss front, now gated on a π⁰ metric

**Mechanism.** 74.1 % of the missing charge is in **other reconstructed
objects** — SPLIT 32.4 % + STOLEN 38.9 % + UNTOUCHED 2.9 % — present in the
event and mis-attributed, entirely post-vertex. Only 17.0 % is REROOT (the reco
never built the object) and 6.7 % DECLINED by a shipped guard. This is the same
pool §5 measures as `q_miss` = 14.8 % of target charge.

**The owner already adjudicated it.** `em_labels/emscan-0829-pr130qmiss/` is
his own scan, and the answer was **MERGE on 17 of 17** scannable questions.

**The live lead is precise, and it is a read, not a knob.** Four of the seven
target showers — 122660/9110, 181050/15006, 463565/13001, 469665/15003 —
emit **no `SHOWER_XCLUS` lines at all** and carry **79 % of the charge**. They
are never enumerated as cross-cluster absorbers by
`shower_clustering_with_nv_from_vertices`, so no predicate, ordering rule or
tie-break at any existing seat can reach them. The next step is to instrument
the enumeration itself and find out why.

**What changed since the park** (2026-08-29, *"we will move on for now, may
come back to this later"*). The 463565 warning was that a merge front must be
gated on a π⁰ metric and not on `q_extra` alone — because the owner said there
*should* be two showers there, and fixing the energy can cost the two-γ
separation. **That gate now exists**: the π⁰ census plus §3's mass closure,
which is exactly the "did we keep two γs, and does their mass close" test that
was missing. This is why the front is ranked #1 now and was not then.

**Specimens.** 342199, 105946, 21073 are simultaneously MERGE-approved by the
owner and π⁰ blockers. On the EM side the worst-F1 rows are 175896 (0.322),
284206 (0.524), 52044 (0.533), 318769 (0.560), 142421 (0.606), 342199 (0.670).

**Killed by**: instrumenting the enumeration and finding the four are excluded
for a reason that cannot be changed post-vertex.

**Ceiling**: 58.7 % of kept `q_miss` is recoverable under the pr/128 precedent;
on the π⁰ side the 4 rescuable pairs of §4. Note this will **not** be a local
default-OFF guard like everything in pr/123 → pr/130 — expect a full 239-event
gate.

### #2 — 342199-class merge-BEFORE-accept, with the K22-v2 merged-cloud machinery

**Mechanism.** The label pair is window-rejected at *fragment* charges
(56.7 × 64.2; healed 80.3 × 121.8 gives m ≈ 131), so the merge has to happen
*before* acceptance. K12's domain, retried with machinery that did not exist
when K12 died. §4 confirms it from the other side: hand marks alone take
342199's mass 92.6 → 127.0 and its R 0.69 → 1.01, and §5 gives it `q_comp`
0.504 — half the charge the scanner assigns is not held.

**Specimens.** 342199 (both γs UNDER+SIBS with exactly one sibling each), plus
409634, 54341, 54332 — the other three §4 rescues, all with low `q_comp`.

**Killed by**: the fragments not being graph-reachable, or the merge costing an
accepted pair elsewhere.

**Ceiling: +1 to +4 exact.** Every specimen has hand-attested charge behind it,
which no previous π⁰ proposal could say.

### #3 — Break the energy-ladder feedback

**Mechanism.** §5 item 2. A knob so `examine_showers`' tier is not evaluated on
the shower's own possibly-deficient `kine_charge`, or so the ladder re-runs
after late growth. The first measurement is free and comes out of proposal 0's
tape: **how many showers sit within a few MeV of the 150 / 250 / 360 / 800 MeV
tier edges, and how many segments were rejected only because of the tier they
landed in.** The same tape answers whether the 0.84 → 0.86 flip moved anything
across those edges — a question pr/135 §10 left open.

**Killed by**: a census showing few showers sit near a tier edge, or that the
segments rejected at the narrow tiers would not have been accepted anyway.

**Ceiling**: unknown until the census runs — which is why it is #3 and not #1.

### #4 — The over-clustered side, which has never been worked

**Mechanism.** `q_extra`, not `q_miss`. §5 finds **29 pure-OVER and 30 both**
against 22 pure-UNDER, i.e. the majority of hand-marked showers hold charge the
scanner says they should not; §3 finds 8 pairs at median R = 1.98.

**Specimens.** 396222 (a 2658 MeV "γ"), 506114, 21073, 259542, 348691, 142421,
286655, 56243; the ledger's OVER 4 (99838, 165157, 347824, 489327); the
over-extended-start mirror class (489327 at −34.1 cm, 347824 at −11.1 cm,
doc 132 §13.1); and 175896, whose `q_pur` is **0.192**.

**Killed by**: the over-charge turning out to be a labelling artefact rather
than a reco merge — checkable directly, since `q_extra` is keyed to segments
the scanner marked "out".

**Note**: pr/130's `q_extra` scan collapsed 4.661e6 → 0, but it only asked what
to merge *in*. This side is genuinely unmeasured.

### #5 — Fix the metric itself, as the campaign's first engineering act

New `scripts/pr136_gamma_ledger.py` (fork, do not edit the shipped one — M10):
apply `scale = 0.80/fudge`; match on membership overlap rather than
`showers[].id`; cross-check `num_segments` / `total_length`; draw the sibling
cone in the reco frame; and report the 100 base γs that carry scanner
judgement separately from the 32 overlay γs that do not. Also repair
`pr126_pi0_select.py --selftest`, which is currently unrunnable because
`build_rows` loads the released scan-time manifests at `:200` — so the guard
that would catch a corrupted label set does not run.

### #6 — Refit the EM energy scale on the CONTAINED subsample (cheap, analysis only)

**Mechanism.** §6.1 shows the full-sample peak sits at 135 partly by
cancellation: contained pairs run +1.3 % high, leaking pairs 5 % low. A fudge
fitted on that blend is a *detector-geometry* constant wearing the clothes of a
calorimetric one, and it will not transport to a different fiducial cut, a
different sample, or ICARUS. Re-running `pr135_pi0_peak_prod.py` on the 32 pairs
with both γs at `f ≥ 0.95` gives the calorimetric number.

**Specimens**: none needed — this is a whole-sample fit.
**Killed by**: the contained-subsample peak agreeing with the blend inside its
CI, which would say the cancellation is coincidental and the blend is safe.
**Ceiling**: no census change; it de-risks every mass window in the chain and
answers whether 0.86 is right *for the reason we think it is*. Do it before any
future scale flip.

### #7 — Close the residue, one way or the other

§6 has now accounted for most of the 19: 4 rescuable, 4 irreducibly leaking, 3
excluded on the charge budget, 2 geometry/label. For what is left the question
is whether the deficit is sub-segment charge (invisible to the labels by
construction — see §8) or a label-audit problem. The dump's `proj[]` carries
per-plane measured `charge` vs `charge_pred` per (wire, slice) with
`cluster_id`, which is where charge that exists in the image but landed in no
segment would show up. Read-only, offline, no re-run. **If the charge is not
there, say so and stop** — that is a result, and it bounds the campaign.

**71178 is a label-audit item, not a reco item.** Its two γs total 158 MeV at a
labelled opening angle of 18°, and both are largely contained. A 158 MeV π⁰ must
open at least 117°. Either the pairing is wrong or the π⁰ decayed far from the
labelled vertex; no clustering change reaches it.

### Park with a reason

- **The deep-start class.** K17 (`shower_em_backext_perp_cm`) died twice: v1
  took the census 31 → **21** exact by swallowing true γ2s wholesale; v2 with a
  30° continuation guard reached 26, net **−5**. Do not retry without a new
  mechanism.
- **A direction-radius sweep.** doc 132 §17.2 measured that the charge-centroid
  ray carries the *same* bias as the start ray on 105946 (19.1° vs 17.8°
  against a label 72.7°), so widening the radius toward the whole-shower
  centroid is predicted dead **for the pairing angle**. It remains untested for
  the *absorption* cone axis, which is a different consumer — but that is a
  proposal-0 census question, not a knob.
- **52044 is reclassified.** pr/135 §9-2 queued it as a "wrong partner" pairing
  fix. It is pr/130's **REROOT** class — the reco never built the object
  (`q_comp` 0.364, `q_pur` 1.000, and the opening angle is right to 1.7°:
  a_start 112.2° vs a_label 110.5°). It belongs in #1, not in a pairing round.

---

## 8. Out of scope, and why

- **Imaging and charge recovery** — upstream of the vertex; the owner's
  constraint removes it.
- **Vertex finding** — explicitly excluded. This also removes 54332, 76346 and
  54453, which the owner's own q_miss scan called "not scannable — ν vertex
  wrong".
- **MC truth calibration** — overruled by the owner. §3 shows it is not needed:
  m_π⁰ is the anchor.
- **Sub-segment charge.** The binding limit of the hand-scan metric: every hand
  judgement is keyed to an **existing reco segment id** — `members`,
  `marks_by_shower`, `energy_marks_detail[].seg`, even
  `energy_orphan_detail[].seg` (orphans are `shower_id < 0` segments, still
  segments). There is no hand-drawn extent, hull or region anywhere in the
  schema. **Charge that wire-cell never turned into a segment is invisible to
  the labels by construction**, and proposal #6 is the only way to see it.
- **Shower leakage out of the active volume** — the owner's point, quantified
  in §6.1. 26 % of hand γs have a contained fraction below 0.95 and 43 % of
  pairs have at least one leaking γ. This is a **physics floor, not a defect**:
  we can only reconstruct charge that was deposited in the TPC. It is out of
  scope in the strongest sense — no reconstruction change of any kind recovers
  it — and the only responses available are (a) classify it, so the campaign
  does not chase it, which §6.1 now does; (b) fit the energy scale on contained
  showers only (proposal #6); and (c) if the analysis ever needs an unbiased π⁰
  mass, apply a containment weight or a fiducial cut at *analysis* level, which
  is a different decision from anything in this chain.
- **The EM energy scale.** Closed self-consistently at 0.86 (pr/135 §10), and
  §6.1 shows it is good to ~1 % on contained showers. If this campaign lands
  charge, the scale moves — that is a *consequence* to re-measure at the end,
  not a task.

---

## 9. Open owner decisions carried forward

1. **397630** — still the one adjudicated cost of the ν-vertex preference rule
   (pr/135 §9-4). §6 now adds that it is **not reachable** by re-attributing
   reconstructed charge (needs 1.0× its entire budget), which weakens the
   "rescue it with a track-length bound" option further.
2. **The 393505 sentinel Enu window** is still calibrated at 0.84 and fails at
   0.86 by 0.1 MeV (559.9 against [560, 572], pr/135 §10). It needs a
   deliberate rebase, not a silent retune.
3. **173093 g1** newly crosses the ledger's UNDER line at 0.86 (§4). It is
   named in no doc and has never been scanned.
5. **Should the π⁰ mass window (100,160) be containment-aware?** §6.1 finds
   43 % of pairs have a leaking γ, and a fixed window applied to a population
   whose mass is systematically pulled down by leakage rejects real π⁰ for a
   detector reason. Widening it costs purity. This is an owner-level physics
   decision, not a knob to try — raised here, not taken.
4. **`pr126_pi0_select.py --selftest` is broken** and has been since the
   2026-08-31 retire round. Nothing is validating the label corpus.

---

## 10. Round 1 — proposal 0 executed, and what its tape says

**Status: MEASUREMENT COMPLETE, 2026-08-30. Proposals #3 and #6 are closed by
measurement; #1 is answered and replaced by a sharper, smaller target.**

### Repro (round 1)

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# proposal 0 -- the byte-neutral probe arm at the f086 production point
PR_JOBS=32 ./scripts/pr136_arms.sh 98  f086probe 1
PR_JOBS=32 ./scripts/pr136_arms.sh 141 f086probe 1
./scripts/pr136_manifests.sh f086probe
# NOTE the explicit --out: prep_em_scan.py ALWAYS rewrites its --out manifest,
# and the default is the tracked em114-manifest.tsv.  Without --out a
# probe-parsing run silently truncates that scan record to its header (M13).
python3 em_display/prep_em_scan.py --prepdir em_display/emprep-136f086 \
    --out /home/xqian/tmp/pr136-parse-manifest.tsv --no-bee-index \
    --parse-probes work-pr136-f086probe-{mcp1k,mcp2k,ncpi0,nuecc48}

# #3 refresh -- completeness at the PRODUCTION point (no longer a cross-arm join)
cd em_display
./em117_score.py --tag emscan-0827        --manifest em117-136f086probe98-manifest.tsv \
    --prepdir emprep-136f086 --tsv ../docs/pr/pr136-completeness-f086-98.tsv
./em117_score.py --tag emscan-0828-agent5 --manifest em114c-136f086probe141-manifest.tsv \
    --prepdir emprep-136f086 --tsv ../docs/pr/pr136-completeness-f086-141.tsv
cd ..
scripts/pr136_completeness.py --src98 pr136-completeness-f086-98.tsv \
    --src141 pr136-completeness-f086-141.tsv --tsv docs/pr/pr136-completeness-f086arms.tsv

# #1 -- WHY the missing charge is unreachable (the enumeration instrument)
scripts/pr136_xclus_enum.py                     # -> docs/pr/pr136-xclus-enum.tsv
# #3 -- the energy-ladder tier census
scripts/pr136_tier_census.py                    # -> docs/pr/pr136-tier-census.tsv
# #6 -- the mass peak on the contained subsample
scripts/pr136_peak_contained.py                 # -> docs/pr/pr136-peak-contained.tsv
```

The arm is `work-pr136-f086probe-{mcp1k,mcp2k,ncpi0,nuecc48}`, **239 / 239
events, every `rc=0`**, shipped production config (fudge 0.86, the pr/133+134
chain), four getenv-gated stderr tapes and no `SBND_*` env.

**Byte-neutrality is gated, not asserted.** Four getenv tapes at once is more
probe surface than any earlier round validated in one go, and every number in
§10 rests on the arm being the production point. `work-pr134-f086-*` is the
same config with the probes off, so it is the reference:

```bash
for s in mcp1k mcp2k ncpi0 nuecc48; do
  python3 scripts/pr85_hash_gate.py work-pr134-f086-$s work-pr136-f086probe-$s; echo rc=$?
done
```

**PASS, all four samples, rc=0: 132 + 212 + 38 + 96 = 478 / 478 archives
byte-identical** (66 + 106 + 19 + 48 = 239 events, 0 unpaired). The probe arm
*is* the f086 production point.

### 10.1 §5's completeness numbers, now at the production point

| | pr130 arms (§5) | **f086 arms (this round)** |
|---|---|---|
| `q_miss` / q_target | 14.8 % | **14.0 %** |
| `q_extra` / q_target | 8.9 % | **7.0 %** |
| median per-shower `q_f1` | 0.907 | **0.918** |
| worst `q_f1` | 0.322 | **0.524** |
| pure UNDER / pure OVER / both / clean | 22 / 29 / 30 / 9 | **16 / 29 / 31 / 14** |

The pr/133 + pr/134 chain and the 0.86 flip bought ~0.8 pt of completeness and
~1.9 pt of purity on the hand-marked population, and moved six showers out of
pure-UNDER into clean. **The §5 ↔ §3 synthesis is now single-arm** — the
cross-arm caveat that §5 carried is retired.

Worst rows at f086 (was 175896 0.322 on the pr130 arms; that shower is now
1.000/0.513 = an over-clustering row, not an under-clustering one):
284206 0.524, 52044 0.533, 318769 0.560, 142421 0.606, 278420 0.669,
342199 0.670.

### 10.2 The enumeration census — the load-bearing result of this round

`pr136_xclus_enum.py` takes every segment the scanner says a hand-marked shower
should hold and it does not (192 segments, 46 showers, 6.55e7 charge = the whole
of `q_miss`), and asks **which mechanism made it unreachable**. Evidence first:
if the `SHOWER_XCLUS` tape carries a line for the (shower, segment) pair, that
line is the verdict; only when the tape is silent is the reason inferred, and
each inference cites the source line that produced it.

**The join has two defensible readings, so every number below is a bracket.**
A tape line's `shower=` is the start segment *at pass time*; a shower that
re-roots later (`examine_showers_retarget`) or absorbs another shower changes
id. The **strict** join credits only the pair `(final shower, segment)`; the
**expanded** join also credits a line emitted under a predecessor id whose start
segment the final shower now owns. 26 of the 192 rows differ between them.
Strict is the conservative reading of "this shower refused it"; expanded is the
conservative reading of "some component of this object looked at it". Both are
reported; neither is hidden.

| verdict | n_seg (strict → expanded) | share of `q_miss` | what it means |
|---|---|---|---|
| `REJECT` | 47 → 59 | **27.5 % → 32.0 %** | the shower evaluated it and the cone refused — a threshold |
| `OWNED` | 25 → 39 | **8.0 % → 16.0 %** | another shower already held it; dropped before any geometry |
| `SAME_CLUSTER` | 15 → 14 | 22.4 % → 19.9 % | in the shower's own cluster — the graph walk's job, no cone applies |
| `MAIN_CLUSTER_SKIP` | 1 | 0.7 % | main-cluster segment the main-vertex walk claims |
| **`NO_SEAT`** | **91 → 70** | **37.6 % → 29.3 %** | **the shower never ran a cross-cluster candidate loop at all** |
| `ABSENT` | 13 → 9 | 3.8 % → 2.1 % | had a seat, no tape line either way — residual |

**Between 35.5 % and 48.0 % of the missing charge is reachable by changing a
predicate at an existing cross-cluster seat — so at least half is not.** That
bracket re-scopes the whole front: pr/123 → pr/130 spent eleven rounds tuning
admission predicates, and even on the optimistic reading the population they
could ever have moved is under half the deficit.

**`NO_SEAT` is structural, and the reason is now named.** Exactly two seats in
the code enumerate cross-cluster candidates: pass 4 of
`shower_clustering_with_nv_from_vertices` (`NeutrinoShowerClustering.cxx:2396`)
and sub-pass A of `shower_clustering_in_other_clusters` (`:3868`). Sub-pass B
(`:4093`) has **no candidate loop at all** — it creates a shower from a leftover
cluster, completes the graph walk, and inserts. So a shower built anywhere else
never evaluates one cross-cluster candidate:

| the seat the shower was actually built at | n_seg (strict → exp.) | share of `q_miss` |
|---|---|---|
| `in_other_clusters_B` | 33 → 33 | 12.3 % (unchanged: sub-pass B emits no tape at all) |
| `in_main_cluster` | 26 → 23 | 16.3 % → 12.2 % |
| `examine_showers_retarget` | 13 → 3 | 3.5 % → 1.0 % |
| (no walk site recorded) | 9 → 9 | 3.2 % |
| `connecting_to_main_vertex` + retarget | 7 → 2 | 1.8 % → 0.6 % |
| `examine_shower_1_tmp` | 2 → 0 | 0.3 % → 0 |
| `conn3_unreachable` | 1 → 0 | 0.4 % → 0 |

This confirms pr/130 item 4's live lead and generalises it from 4 showers to 13,
with the mechanism named: **it is not that those showers lost an arbitration —
they were never in one.** The four pr/130 specimens now have individual
diagnoses: 122660/9110 (conn 1, `in_main_cluster`), 181050/15006 (conn 1,
`examine_showers_retarget`), 469665/15003 (conn 1,
`connecting_to_main_vertex`) and 463565/13001 (conn 3, `in_other_clusters_B`).
All four already span 14–22 clusters, so the missing charge is not "this shower
cannot cross a cluster boundary" — it is "this shower is never offered the
direct-cone route that the owner's MERGE verdicts describe".

**Split by the matched shower's connection type, the two halves separate cleanly:**

| conn | share of `q_miss` | composition, strict | composition, expanded |
|---|---|---|---|
| 1 (at the ν vertex) | 36.2 % | `NO_SEAT` 60 %, `SAME_CLUSTER` 38 %, **`REJECT` 0 %** | `NO_SEAT` 38 %, `SAME_CLUSTER` 38 %, `OWNED` 18 %, `REJECT` 4 % |
| 2 | 42.4 % | `REJECT` 65 %, `OWNED` 16 %, `SAME_CLUSTER` 11 %, `ABSENT` 8 % | `REJECT` 72 %, `OWNED` 19 %, `SAME_CLUSTER` 5 % |
| 3 | 21.4 % | `NO_SEAT` 74 %, `SAME_CLUSTER` 20 %, `OWNED` 5 % | `NO_SEAT` 72 %, `SAME_CLUSTER` 20 %, `OWNED` 7 % |

**On the strict join, not one MeV of missing charge on a conn-1 shower was ever
refused by a predicate; on the expanded join 4 % was, and a further 18 % lost an
ownership race.** Conn-1 showers are the ones attached to the neutrino vertex —
the π⁰ γs. Either way the conn-1 half is dominated by charge no cone ever
weighed (`NO_SEAT` 38–60 %) plus charge inside the shower's own cluster
(`SAME_CLUSTER` 38 %, a graph-walk question), and the threshold rounds this
campaign can run reach conn-2 almost exclusively.

### 10.3 The sharpest single finding: a pre-filter that is stricter than the test it guards

Pass 4 computes both angles and then applies a cheap early filter before the two
expensive KD-tree calls:

```c++
// NeutrinoShowerClustering.cxx:2400-2404, :2434
double angle_v1 = angle(dir_shower, pair_point - start_pt);   // from the shower start
double angle_v2 = angle(dir_shower, pair_point - point);      // from the cluster anchor
if (angle_v2 > 30) { ...continue; }                            // the PRE-FILTER
...
// :2450 -- the acceptance disjunction it guards
if ((angle_v1 < 25   && (pair_dis < 80cm  || close_shower_dis < 25cm)) ||
    (angle_v2 < 25   && (tmp_shower_dis < 40cm || close_shower_dis < 25cm)) ||
    (angle_v1 < 12.5 && (pair_dis < 120cm || close_shower_dis < 40cm)) ||
    (angle_v2 < 12.5 && (tmp_shower_dis < 80cm || close_shower_dis < 40cm)))
```

**Two of the four acceptance clauses do not mention `angle_v2` at all**, yet the
pre-filter discards the candidate on `angle_v2` alone. Reading the tape's
`angle_v1` column for the 29 early-rejected segments in the census:

| event | shower | seg | q | share of `q_miss` | `angle_v1` | `angle_v2` | `pair_dis` |
|---|---|---|---|---|---|---|---|
| 142421 | 108104 | 7010 | 5.91e6 | **9.0 %** | **21.3°** | 39.6° | 14.6 cm |
| 314838 | 110088 | 13010 | 2.04e6 | 3.1 % | **21.6°** | 60.6° | 17.4 cm |
| 52044 | 58029 | 24009 | 8.33e5 | 1.3 % | 14.4° | 162.5° | 25.6 cm |
| 84229 | 69134 | 9058 | 7.38e5 | 1.1 % | 9.9° | 50.5° | 20.2 cm |
| 105946 | 55063 | 53029 | 7.22e5 | 1.1 % | 18.4° | 51.6° | 71.8 cm |
| 409634 | 27015 | 69032 | 4.87e5 | 0.7 % | 9.3° | 169.7° | 7.4 cm |
| 105946 | 55063 | 53030 | 2.57e5 | 0.4 % | 24.4° | 64.3° | 69.1 cm |
| 409634 | 27015 | 69033 | 1.58e5 | 0.2 % | 9.3° | 169.7° | 7.4 cm |
| 105946 | 55063 | 54032 | 1.52e5 | 0.2 % | 14.7° | 52.4° | 64.6 cm |
| 54341 | 96031 | 77019 | 1.43e4 | 0.0 % | 8.9° | 42.8° | 36.3 cm |

**10 segments carrying 17.3 % of all `q_miss` satisfy an acceptance clause on
`angle_v1` and were killed by the `angle_v2` pre-filter before that clause was
ever evaluated** — including the single largest missed segment in the census,
which alone is 9 % of the deficit. The count is a **lower bound**: the early
tape does not carry `close_shower_dis`, and clauses 1 and 3 also admit on
`close_shower_dis < 25 / 40 cm`.

**Prototype check (M15, mandatory before calling this a defect).**
`prototype_base/wire-cell/pid/src/NeutrinoID_shower_clustering.h:1299` carries
`if (angle1/3.1415926*180. > 30) continue;` with the identical acceptance
disjunction below it (prototype `angle1` ≡ toolkit `angle_v2`, prototype `angle`
≡ toolkit `angle_v1`). **The pre-filter is faithful to the prototype.** So this
is *not* a porting defect and must not be "fixed" as one — it is a candidate
*improvement* to an algorithm both codebases share, and it ships the same way
everything else here ships: a default-OFF knob with a byte-identical gate.

Two sub-classes, and they are not the same proposal:

- **forward escapes** (`angle_v2` 30–90°): 7 segments, **15.0 % of `q_miss`**,
  including both top segments. 142421, 314838, 84229, 105946 (×3), 54341.
- **backward escapes** (`angle_v2` > 90°): 3 segments, 2.3 %. 52044, 409634 (×2).
  These sit *behind* the shower relative to the cluster anchor — the same
  geometry class as K17 back-extension, **which died twice** (pr/124: v1 31→21
  exact; v2 26 exact, net −5). Keep them separate or the forward result inherits
  a known death.

**Which π⁰ specimens each class actually reaches — stated precisely, because
the two are easy to conflate.** Both of 409634's escape segments sit at
`angle_v2` = 169.7°, i.e. **entirely in the backward class**, so a forward-only
knob does *not* reach 409634. Forward-only reaches 54341 (a §4 hand-marked
rescue), 142421 and 105946 (both pr/130 MERGE-approved specimens), 314838 and
84229. 52044 and 409634 are backward-only.

### 10.4 Proposal #3 (the energy-ladder feedback) is measured DEAD

`examine_showers`' cone width really is keyed to the absorbing shower's own
`kine_charge` (`:4999-5013`), and the fudge really does divide it — the
mechanism is exactly as §5 described. It does not matter, because the population
is nowhere near the edges:

- 2051 EM showers over 239 events; **median distance to the nearest tier edge
  97.8 MeV**; 84.4 % are more than 50 MeV from one; only **4.9 % are within
  10 MeV** and 1.0 % within 2 MeV.
- The direct test: the 0.84 → 0.86 flip is a 2.4 % energy step, and it moved
  **20 of 2051 showers (0.98 %)** across a tier edge — all 20 downward, as
  arithmetic requires. The largest is 137238/143056 at 354 MeV crossing 360.
- 85.9 % of EM showers sit in tier 0 (below 100 MeV), where no ladder clause
  fires at all.

**Closed.** The self-reinforcement is real in the source and irrelevant in the
data. It also retires the last live version of the "energy-ladder feedback"
hypothesis that §5 flagged as untested.

### 10.5 Proposal #6 (refit the scale on contained showers) — done, and 0.86 survives

`pr136_peak_contained.py`, on the same 56 pairs, splitting by §6.1's containment:

| cell | n | n_in | peak (MeV) [CI68] | implied fudge [CI68] |
|---|---|---|---|---|
| A all pairs (the blend the 0.86 fit saw) | 56 | 40 | 134.7 [130.5, 138.2] | 0.858 [0.831, 0.880] |
| **B contained pairs (f₁,f₂ ≥ 0.95)** | **32** | **25** | **136.5 [132.8, 139.7]** | **0.869 [0.846, 0.890]** |
| C ≥ 1 leaking γ | 24 | 15 | 127.7 [100.0, 138.3] | 0.814 |
| D contained ∧ geometry-clean | 14 | 10 | 135.3 [128.2, 141.4] | 0.862 |

Cell A reproduces `pr135-peak-f086-cells.tsv` cell A to 0.04 MeV — an
independent check that the containment TSV's `m_prod` is the same estimator.

**Reading: the leakage blend biases the scale by +1.8 MeV (+1.3 % in the fudge),
which is inside its own CI68, and the fudge in force (0.86) sits inside the
contained-only CI68 [0.846, 0.890].** So the 0.86 flip is calorimetrically
correct, not a leakage artefact, and **the scale is not the front** — stop
chasing it. What survives is the transportability caveat: a fudge fitted on the
blend carries this sample's average leakage, so it is partly a
detector-geometry constant and does not transport to another fiducial cut or
another detector. If SBND ever changes the fiducial volume, refit on cell B.

### 10.6 The ranking after round 1

| | proposal | status after this round |
|---|---|---|
| **#1** | **pass-4 `angle_v1` escape from the `angle_v2` pre-filter (forward class)** | **NEW, and now the top of the list: 15.0 % of `q_miss`, 7 segments, prototype-checked, one bounded predicate, default-OFF-able.** Was invisible until the tape existed. |
| #2 | 342199-class merge-before-accept | unchanged: +1 to +4 exact, hand-attested charge |
| #3 | energy-ladder feedback | **CLOSED — measured dead (§10.4)** |
| #4 | the over-clustered side | unchanged; `q_extra` is now 7.0 %, and it is the *cost* side of #1, so it gets measured either way |
| #5 | fix the γ ledger | unchanged |
| #6 | contained-subsample refit | **DONE (§10.5); 0.86 confirmed** |
| — | **`NO_SEAT`: give conn-1 / `in_other_clusters_B` showers a cross-cluster route** | **NEW, and the largest single pool (37.6 %), but it is a new seat, not a predicate — expect a full 239-event gate and a large diff. Rank it after #1 because #1 is bounded and this is not.** |
| — | old #1 (un-park pr/130's q_miss front as a *predicate* round) | **superseded**: the tape says predicates reach 35.5 % of the deficit and 0 % of the conn-1 half |

**The measurement that would kill #1, stated before it is written.** The
hand-scan metric alone cannot see this knob's cost: `pr136_completeness.py`
scores 46 marked showers, while the escape fires on every conn-2 shower in all
239 events, so charge wrongly added to an *unmarked* shower is invisible to it.
Three instruments, and any one of them failing kills the round:

1. **`q_extra` must not rise by more than `q_miss` falls** on the 90 marked
   showers (`pr136_completeness.py`, both sets).
2. **The π⁰ census exact count must not drop from 32 / 66** — the population
   instrument for "did we destroy a two-γ separation", which is exactly the
   463565 warning that parked the pr/130 front.
3. **The mass closure's R > 1 over-clustering class must not grow** beyond its
   8 pairs at median R = 1.98 (`pr136_mass_closure.py`).

And expect the diff to exceed the 10 measured segments: admitting a segment to
shower A changes what B sees later in the same pass — the chain effect pr/123
found. The 10 are the *seed*, not the bound.

---

## 11. Round 2 — `shower_pass4_prefilter_v1_escape`, the knob §10.3 asked for

**Status: MEASURED, rounds 2 and 3. The proximity-braked variant
(`onV1c90d25`) recovers 1.5 pt of `q_miss` at zero measured cost on the
hand-scan purity metric and costs exactly one π⁰ census row — on an event where
the owner's own EM marks contradict the agent-produced π⁰ pairing. NOTHING IS
FLIPPED; §11.6 states the owner decision.** Three knobs, all DEFAULT OFF, one
behaviour change between them:

| knob | type | default | meaning |
|---|---|---|---|
| `shower_pass4_prefilter_v1_escape` | bool | `false` | let a pass-4 cross-cluster candidate past the `angle_v2 > 30` pre-filter when it still satisfies an `angle_v1` clause of the acceptance disjunction |
| `shower_pass4_prefilter_v1_max_v2` | double (deg) | `0` | ceiling on `angle_v2` for that escape; `0` = none. Inert while the escape is off |
| `shower_pass4_prefilter_v1_max_dis` | double (cm) | `0` | proximity bound on `pair_dis` for that escape; `0` = none. Added in round 3 (§11.4). Inert while the escape is off |

The predicate, at `NeutrinoShowerClustering.cxx:2433`:

```c++
if (angle_v2 > 30) {
    const bool p4_v1_escape =
        m_shower_pass4_prefilter_v1_escape &&
        (m_shower_pass4_prefilter_v1_max_v2 <= 0 ||
         angle_v2 < m_shower_pass4_prefilter_v1_max_v2) &&
        (m_shower_pass4_prefilter_v1_max_dis <= 0 ||
         pair_dis < m_shower_pass4_prefilter_v1_max_dis) &&
        ((angle_v1 < 25   && pair_dis < 80  * units::cm) ||
         (angle_v1 < 12.5 && pair_dis < 120 * units::cm));
    if (!p4_v1_escape) { ...legacy tape...; continue; }
    ...ESCAPE tape...
}
```

Only `angle_v1` and `pair_dis` appear, because they are the only quantities in
hand at the filter — the `close_shower_dis` halves of clauses 1 and 3 need the
KD-tree call the filter exists to avoid. So the escape is a *subset* of the
disjunction it restores, never a widening of it, and the §10.3 ceiling of 10
segments is a floor rather than a bound.

### 11.1 The population, priced from the OFF tape before any arm was spent

The f086 probe arm's tape carries **13 927** `pass4_angle_early` rejections over
the 239 events. Applying the escape predicate offline:

| escape variant | fires | events touched | of the 10 §10.3 seeds |
|---|---|---|---|
| unbounded | **810** | 123 of 239 | 10 |
| `max_v2` 140 | 610 | 114 | 7 |
| **`max_v2` 90 (forward only)** | **350** | 92 | **7** |
| `max_v2` 70 | 275 | 82 | 7 |
| `max_v2` 50 | 159 | 71 | 2 |

**This is why the ceiling knob exists.** The escape's `angle_v2` distribution is
nearly flat from 30° to 180°, and **57 % of its population sits above 90°** —
behind the shower relative to the cluster anchor, the K17 back-extension
geometry that died twice (pr/124: v1 31 → 21 exact, v2 26 exact / net −5). Those
460 candidates carry only 3 of the 10 target segments. `max_v2 = 90` halves the
blast radius and keeps 7 of 10.

One more reason to expect a large diff: the escape's first sub-condition
(`angle_v1 < 25 && pair_dis < 80 cm`) **implies** clause 1 of the disjunction, so
a firing is an admission unless the associated-vertex guard (`:2455`) or the
pr/123 track guard declines it. Firing ≈ admitting; 810 firings is 810 candidate
admissions, against 10 measured targets. **80× collateral is the risk this round
has to price, and it is priced by measurement, not by argument.**

### 11.2 The three kill instruments and their OFF values

All measured on `work-pr136-f086probe-*`, the gated production point:

| instrument | OFF value | fails if |
|---|---|---|
| hand-scan attribution (`pr136_completeness.py`, 90 marked showers) | `q_miss` 14.0 %, `q_extra` 7.0 %, median `q_f1` 0.918 | `q_extra` rises by more than `q_miss` falls |
| π⁰ census exact (`pr132_pi0_census.py --fudge 0.86`) | **32 / 66 = 48.5 %** | drops |
| mass closure over-clustering class (`pr136_mass_closure.py`) | 8 pairs at median R = 1.98; 19 of 56 impossible; classes 11 / 37 / 8 | the R > 1 class grows |

### 11.3 Round 2 result — the mechanism works, the knob as written does not

**The default-OFF requirement is met.** `work-pr136-off1-*` (new binary, probes,
no knob env) against `work-pr136-f086probe-*`:

**PASS, rc=0 on all four samples: 132 + 212 + 38 + 96 = 478 / 478 archives
byte-identical.** Every downstream metric reproduces the OFF point exactly
(`q_miss` 14.0 %, `q_extra` 7.0 %, median `q_f1` 0.918, census 32 / 66, closure
classes 11 / 37 / 8). **Compiled-config proof**, on the runner's own saved
`.wct-cfg-evt<ID>.json`: the OFF arm's compiled JSON contains **zero** knob keys
and, after normalising the arm name inside output paths, differs from the
pre-change arm's by **zero lines**. `onV1`'s carries `..._escape: true` with
`..._max_v2` correctly suppressed at 0; `onV1c90`'s carries both.

**Two ON arms, 239 events each, every `rc=0`:**

| | OFF | `onV1` (unbounded) | `onV1c90` (`max_v2` 90) |
|---|---|---|---|
| escape fired | — | 582 in 123 events | 302 in 92 events |
| … admitted | — | 320 | 195 |
| of the 10 §10.3 seeds | — | 10 fired, 8 admitted | 7 fired, **7 admitted** |
| segments re-owned vs OFF | — | 775 over 102 events | 391 over 71 events |
| `q_miss` | 14.0 % | **10.4 %** | **11.3 %** |
| `q_extra` | 7.0 % | **12.1 %** | **12.2 %** |
| median `q_f1` | 0.918 | 0.917 | 0.918 |
| **π⁰ census exact** | **32 / 66** | **30 / 66** ✗ | **33 / 66** ✓ |
| mass-closure R > 1 class | 8 of 56 | 4 of 44 | 4 of 47 |

**Verdict against the three pre-registered criteria (§11.2):**

1. **`q_extra` must not rise by more than `q_miss` falls — FAILED by both.**
   `onV1c90` recovers 2.7 pt of `q_miss` and pays 5.2 pt of `q_extra`.
2. π⁰ census exact — `onV1` **fails** (32 → 30); `onV1c90` **passes and improves**
   (32 → **33**, and γ-sharing 74 % → 77 %).
3. mass-closure over-clustering class — **this criterion turned out to be the
   weakest of the three and should not be leaned on.** The closure matches γs by
   `showers[].id`, which is the constituent-segment id §2 already flagged as
   unstable; a membership change renames showers, and the matched-pair count
   falls 56 → 47 (absent/zero 5 → 11). The R > 1 count dropping 8 → 4 is
   therefore not evidence of improvement. Criteria 1 and 2 use charge-overlap
   matching and acceptance respectively, and both are sound.

**Rule 1 is a criterion I wrote down before the arm ran, and it failed. The
knob as written is killed.** What survives is more interesting than a pass would
have been.

**The mechanism works exactly where §10.3 predicted, cleanly.** Per marked
shower, `onV1c90` against OFF:

| event / shower | `q_miss` recovered | `q_extra` paid |
|---|---|---|
| **142421 / 7010** (the largest missed segment in the census) | **+5.91e6** | **−6.2e4** |
| 314838 / 110088 | +2.49e6 | +3.8e4 |
| 84229 / 69134 | +7.38e5 | **0** |
| 122660 / 9110 | +2.02e5 | **0** |

Four clean recoveries, 9.3e6 of charge, essentially no collateral — and 142421
seg 7010 alone was 9 % of the whole deficit.

**The failure is concentrated, not diffuse, and it is mostly not the escape.**
Of the 2.48e7 total `q_extra` rise: **269774 / 97197 alone is +1.24e7 = 50 %**,
the top three showers are 79 %, the top five are 97 %. And 269774's escape only
admitted **8 segments worth 3.3e6 to a different shower (13237)** — the +1.24e7
landed on 97197, which the escape never touched. **That charge arrived through
the chain**, i.e. through the downstream passes reacting to a changed shower
graph, exactly the pr/123 effect §10.6 warned about.

**A prediction that was wrong in the useful direction**, worth recording because
it changes how the next offline estimate should be read: the OFF tape predicted
465 firings on the 98 events finished in all three arms and the ON arm produced
**317**. The offline estimate double-counts contested segments — in the OFF arm
every shower that looks at a segment emits its own `REJECT` line, but once one
shower absorbs it, `update_shower_maps` marks it owned and every later shower
skips it at the OWNED gate before reaching the pre-filter. **The direct effect is
self-limiting; only the chain effect multiplies.**

### 11.4 The separator, and round 3

The clean recoveries and the runaways separate on **proximity**, not on angle:

| | `pair_dis` | `angle_v1` |
|---|---|---|
| clean recoveries (142421, 314838, 84229) | **14.6, 17.4, 20.2 cm** | 21.3°, 21.6°, 9.9° |
| runaway events (269774, 406125, 174771, 318769) | **26 – 61 cm** | 12 – 24° |

`angle_v1` does not separate them; `pair_dis` does, with a gap between 21 and
26 cm. Swept offline on the OFF tape at `max_v2` = 90:

| `max_dis` | fires | events | clean winners kept | fires inside the 4 runaway events |
|---|---|---|---|---|
| none | 350 | 92 | 3 | 21 |
| 30 cm | 51 | 29 | 3 | 2 |
| **25 cm** | **33** | **22** | **3** | **1** |
| 20 cm | 22 | 16 | 2 | 0 |

**25 cm keeps every clean winner, cuts the blast radius 10-fold (350 → 33
fires, 92 → 22 events) and leaves one firing in the four runaway events.**
20 cm loses 84229. Hence `shower_pass4_prefilter_v1_max_dis`, default 0, and
round 3's arm `onV1c90d25`.

### 11.5 Round 3 result — the proximity brake

**Gate first.** The round-2 gate ran on the two-knob binary; `max_dis` is
default 0 and is only read inside the escape branch, which is unreachable when
the escape is off — but that is an argument, and the house rule is a gate.
`work-pr136-off2-*` (three-knob binary, probes, no knob env) against
`work-pr136-f086probe-*`: **PASS, rc=0 on all four samples, 478 / 478 archives
byte-identical**, and the mass closure reproduces 11 / 37 / 8 with 19 of 56
impossible. `./build/clus/wcdoctest-clus` 235 / 235 cases, 2601 / 2601
assertions.

**The four arms side by side (239 events each, every `rc=0`):**

| | OFF | `onV1` | `onV1c90` | **`onV1c90d25`** |
|---|---|---|---|---|
| escape fired / events | — | 582 / 123 | 302 / 92 | **32 / 22** |
| segments re-owned vs OFF | — | 775 in 102 evts | 391 in 71 evts | **80 in 15 evts** |
| `q_miss` | 14.0 % | 10.4 % | 11.3 % | **12.5 %** |
| `q_extra` | 7.0 % | 12.1 % | 12.2 % | **7.0 %** |
| … of which VIOLATION | 3.2 % | — | 3.8 % | **3.2 %** |
| … of which UNJUDGED | 3.8 % | — | 8.4 % | **3.8 %** |
| median `q_f1` | 0.918 | 0.917 | 0.918 | 0.918 |
| showers with `q_f1` < 0.80 | 18 | 18 | 20 | **17** |
| π⁰ census exact | 32 / 66 | 30 ✗ | 33 ✓ | **31 ✗** |

**Correction to an earlier draft of this section:** 122660 is a `onV1c90`
recovery (+2.02e5 at zero `q_extra`), **not** a `onV1c90d25` one — the 25 cm
brake does not fire on that event. `onV1c90d25`'s recoveries among the marked
showers are 142421 (+5.91e6), 314838 (+2.49e6) and 84229 (+7.38e5).

**And one adverse row this section had not surfaced:** on **181050** the
recovery is **negative in both ON arms** — `q_miss` gets *worse* by 1.83e6, with
29 segments changing owner. It is the only row in the reviewed set where the
knob moves charge the wrong way against the hand marks, and it is included in
the Bee package below.

**Criterion 1: `onV1c90d25` PASSES, cleanly and in the original pre-registered
form.** `q_miss` falls 1.51 pt; `q_extra` moves **−0.01 pt**, with the violation
and unjudged terms each flat to two decimals. The 25 cm brake did exactly what
§11.4 predicted: it kept the recoveries and removed the collateral.

**Criterion 2: it FAILS, by one row, and the row deserves reading.**

### 11.6 The one π⁰ the brake costs — and why it is an owner question

`onV1c90d25` changes the class of **exactly one** census row against OFF:
**314838, `exact` → `partial`, `g2:absent-on-arm`.** Every other one of the 66
is unchanged. What happened, from the dumps:

| arm | accepted π⁰ group | mass |
|---|---|---|
| OFF | shower **110088** (645.0 MeV) + shower **13010** (110.3 MeV) | 121.4 MeV |
| `onV1c90d25` | shower **110088** (**779.6** MeV) + 114095 (3.8 MeV crumb) | 106.6 MeV |

**This is a real physics loss, not an id-matching artefact** — I checked, having
first suspected the `showers[].id` instability of §2. The escape absorbed
segment 13010 into 110088 (645 → 779.6 MeV) and the second γ ceased to exist.
This is precisely the **463565 warning that parked the pr/130 front**: fixing
the energy can cost the two-γ separation.

**But the two hand-scan products disagree about this event, and the knob is
obeying the stronger one.**

- `em_labels/emscan-0827` — the **owner's own EM hand scan** — marks, for shower
  110088: **IN** = {109070, 109071, **13010**, 13011, 83027, 85029, 87031, 89033,
  91035, 93037, 94038, 95039}, **OUT** = {110083, 110084, 110085, 110088, 110089,
  111091, 111092}. It says segment 13010 belongs *inside* 110088, and it
  re-roots the shower off its own start segments.
- `em_labels/pi0scan-0829-agent` — an **agent-produced pairing overlay**, not an
  owner scan — makes 13010 the second γ (γ1 = 110088 at 693.4 MeV, γ2 = 13010 at
  118.6 MeV, θ = 26.3°, mass 130.6 MeV).

**These cannot both be right**, and the census scores the agent overlay. Sizing
it: the 66-row denominator is **50 owner-paired + 16 agent-overlay**; of the 32
exact, **30 are owner-paired and 2 are overlay**. 314838 is one of those 2, and
one of only 5 events where an overlay pairing coexists with owner EM marks.

**So `onV1c90d25`'s only census cost is an agent-produced pairing being
overruled by the owner's own marks.** I am not rescoring the criterion on that
basis — it was registered before the arm ran and it failed as written. The
decision is the owner's:

> **314838: does the EM hand scan (absorb 13010 into 110088) or the π⁰ overlay
> pairing (13010 is the second γ) govern?** If the marks govern, `onV1c90d25`
> costs nothing measurable and gains 1.5 pt of `q_miss`. If the pairing governs,
> the escape needs a π⁰-aware guard that declines an absorb when the candidate
> is itself an accepted γ.

**Also unresolved, and not mine to take:** `onV1c90` buys **+1 census exact**
(32 → 33, γ-sharing 74 % → 77 %) by paying 5.2 pt of `q_extra` — of which only
0.55 pt is a scanner violation and 4.61 pt is charge the scan never ruled on.
The pr/125 K5 flip accepted exactly that kind of trade with the reason recorded
in `wct-pr-perevt.jsonnet:1951`. Two variants, two different bets; **neither is
flipped and both are on the table.**

### 11.7 What round 3 does not claim

- **No knob is flipped.** All three default OFF; SBND production config is
  unchanged apart from the three inert keys being *available*.
- The `q_extra` decomposition of §11.3 is a **new measurement, not a rescoring**.
  It was motivated by the defect §2 raised against the γ ledger, applied to the
  metric this campaign uses; it does not retroactively pass round 2.
- **UNJUDGED charge is not proven benign.** It is charge the scan never saw in
  that shower, and it can be genuine over-clustering nobody has ruled on. The
  only way to settle it is a hand scan of the 15 events `onV1c90d25` changes —
  a small enough diff to package as a Bee A/B pair, which is the natural next
  owner step.

### 11.8 Owner review package — three-way Bee A/B (round 3)

Built and uploaded 2026-08-31 at the owner's request ("bee links for me to judge
onV1c90 overclustering as well as the lost pi0 in onV1c90d25").

| set | URL | arm |
|---|---|---|
| **OFF** | `https://www.phy.bnl.gov/twister/bee/set/8207f8a2-b889-452f-933f-571dccbcf8d9/event/list/` | `work-pr136-off2-*` (gated 478/478) |
| **c90** | `https://www.phy.bnl.gov/twister/bee/set/e5330080-e818-4684-aa2e-21a89e4f5a99/event/list/` | `work-pr136-onV1c90-*` |
| **d25** | `https://www.phy.bnl.gov/twister/bee/set/a99a25b1-f7be-4e3c-9302-a1e5da332fec/event/list/` | `work-pr136-onV1c90d25-*` |

Same 13 events in the same order in all three sets, so the owner flips between
them at one bee index. Annotated index: `bee/pr136r3/pr136r3.index.txt`.

**Content-verified before the links were reported** (the pr/133 rule): all three
zips carry the 13 events in the intended order with all seven layers, and the
per-event `shower_track-global` fingerprints show exactly the expected pattern —
c90 differs from OFF on 12 of 13, d25 only on idx 6 / 7 / 8 / 10, and idx 12
(463565, the negative control) is byte-identical across all three.

| idx | event | why it is in the package |
|---|---|---|
| 0 | 269774 | c90 `q_extra` **+1.57e7 = 63 % of the entire rise**, +1.32e6 recovered; essentially all UNJUDGED, and it arrived through the chain — the escape admitted 8 segments (3.3e6) to shower 13237 while the +1.24e7 landed on 97197 |
| 1 | 406125 | c90 +3.81e6 extra, **zero** recovered, and **mostly VIOLATION (2.60e6)** — the sharpest single "is this a genuine over-cluster?" test |
| 2 | 318769 | c90 +2.96e6 extra **and** +2.73e6 recovered — the trade in one event |
| 3 | 174771 | c90 +1.49e6 extra, +4.58e5 recovered |
| 4 | 105946 | c90 net favourable: +7.02e5 extra against +1.28e6 recovered; pr/130 MERGE-approved specimen |
| 5 | 98844 | c90 +4.89e5 extra, zero recovered — small pure cost |
| 6 | **314838** | **the lost π⁰**: OFF 110088 (645) + 13010 (110) at 121.4 MeV → both ON arms 110088 (780) + a 4 MeV crumb at 106.6 MeV. The scan contradiction of §11.6 |
| 7 | 142421 | the clean win, both arms: +5.91e6 (9 % of the whole deficit) at −6.2e4 |
| 8 | 84229 | clean win, both arms: +7.38e5, zero extra |
| 9 | 122660 | c90 +2.02e5 at zero extra; **d25 does not fire** — shows what the brake gives up |
| 10 | 181050 | **ADVERSE in both arms**: `q_miss` worsens by 1.83e6, 29 segments re-owned |
| 11 | 259542 | c90 loses a π⁰ (census partial → none); no marked EM shower, so invisible to `q_extra` |
| 12 | 463565 | **negative control**: the event that parked the pr/130 front; byte-identical in all three sets |
