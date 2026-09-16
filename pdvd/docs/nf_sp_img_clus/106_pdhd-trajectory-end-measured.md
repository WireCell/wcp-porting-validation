# 106 — the PDHD trajectory end, measured: the levers straighten the track, they do not truncate it

**Status: measurement only. No code, no config, no new arm, no flip. PDHD stays at production.**

Round 9 was commissioned to measure a defect that docs 104 and 105 had inferred from five hand-picked cases:
*trajectory-end truncation* under the two trajectory levers. The measurement was run over every STM candidate in
all 61 PDHD events, on four arms already on disk, using no hand label of any kind.

**It does not find that defect.** The end is unmoved on 72 % of muons; truncation (6 %) is very nearly balanced by
advance (7 %); the code's own retreat counter does not move at all; and the 1.8 % net length loss is fully
accounted for by the tracks getting **straighter**, which is what the levers are for. What the measurement does
find is two things nobody had named: a dominant **candidate-set churn** four to five times larger than the end
effect, and **five direction reversals** in which the same muon is read backwards.

---

## 0. Repro

Everything below comes from one command against arms that already exist. Nothing is regenerated.

```bash
IMG=/home/xqian/toolkit-dev/wcp-porting-img
cd $IMG/pdvd/docs/nf_sp_img_clus/scripts
python3 d106_end_geometry.py > ../figs/106_end_geometry.txt
```

Runtime is about a minute. The script carries three self-checks that abort it rather than print a wrong table
(sec 1.2), and it reads `T_stm_michel` and `T_stm_michel_pts` only — no scan record is opened.

---

## 1. The four arms, and the three traps between them and a number

### 1.1 The arms

All four are on disk with 61/61 events, built on the same pctrees, so `(event, cluster_id)` is a stable join
(doc 101 sec 6.5 `idset_compare.txt`; doc 102 built its data arms on the doc 101 pctrees).

| cell | arm | what it carries | provenance |
|---|---|---|---|
| **A0** | `d101hnew` | production: `'stepped'` sampler, no fit knobs | doc 101 sec 6.2, no `PR_TLA` |
| **K** | `d101hkf` | fit knobs only (`fit_weight_pow` 1.5 + `assoc_cont_center` 1) | doc 101 sec 6.3 |
| **S** | `d102hocs` | retile sampler only (`charge_stepped`) | doc 102 line 574, `PR_TLA="$CS"` |
| **A1** | `d102hcs` | **both** levers — the cell doc 103's grade calls A1 | doc 102 line 112, kf json + `$CS` |

with `$CS = -S retile_sampler_strategy='charge_stepped'` (doc 102 line 84). The arm names were **not** trusted:
the work dirs carry no `pr.json` and the run logs do not echo the keys, so the levers were read off the committed
repro blocks of docs 101 and 102, which are the record of how each arm was built.

Doc 103's per-lever Michel purity is **K 0.880 / S 0.906 / A1 0.895** against A0's 0.946 — not additive, "both"
beats "fit knobs alone". Every table here therefore reports all three levers against A0 separately, and this doc
assumes no additivity anywhere.

### 1.2 The three traps, and the checks that catch them

**Population trap 1 — never intersect all four arms.** The candidate set moves (sec 2). A four-way intersection
keeps exactly the clusters that all four configurations agreed on, i.e. the ones the levers did not disturb; it is
selected *against* the effect and would have reported "no truncation" for the wrong reason. Each lever is compared
on its own pairwise intersection with A0, and lost/gained are reported as strata rather than dropped.

**Population trap 2 — `muon_len` is 0 for an early reject.** It is set at `CheckSTM_Michel.cxx:3085`
(`prof.total_length`) with struct default 0 at `:1152`. Differencing a 0 against a real length fabricates a 100 %
truncation. Pairs require `muon_len > 0` in **both** arms. On these arms nothing is dropped — all 341/320/328/333
candidates have a length — so the trap does not bite here, but the guard stays.

**Classifier trap — a forward projection cannot see an advance, and mis-scores a reversal.** Both were found by
checking a case instead of trusting a bin, and both are documented in sec 4.1.

| check | what it asserts |
|---|---|
| **V1** | doc 104 sec 3.1's published lengths reproduce from these arms: `029107_28/109` 595.9 → 544.8, `029107_27/39` 62.1 → 40.2, `029107_19/111` 108.6 → 121.8 |
| **V2** | outside the reversal class, no pair fires both the truncation and the advance test, so the cascade order is not a hidden choice |
| **V3** | every `muon_len > 0` pair has a role-1 chain in both arms, so sec 2 and sec 4 quote **one** denominator, not two |

A fourth guard is not an assert but matters as much: in sec 4.3 a track enters the wiggle comparison only if
**both** arms give a usable chain. An earlier version skipped a whole class when the two arms disagreed on
length, and silently printed the table without its most important row.

---

## 2. The dominant effect is candidate churn, not the end

| lever | \|A0\| | \|L\| | shared | lost | (of which `is_stm`) | (of which Michel) | gained | (of which `is_stm`) | lost `muon_len` p50/p90 |
|---|---|---|---|---|---|---|---|---|---|
| K | 341 | 320 | 271 | 70 | 22 | 28 | 49 | 14 | 94.3 / 401.6 |
| S | 341 | 328 | 252 | 89 | 22 | 34 | 76 | 25 | 157.6 / 477.3 |
| **A1** | 341 | 333 | **261** | **80** | **20** | **34** | **72** | **27** | 145.5 / 494.9 |

**A1 loses 80 of A0's 341 candidates (23 %) and gains 72.** Of the lost, 20 carried `is_stm` and 34 carried a
Michel tag. Their median length is 145.5 cm — these are substantial tracks, not junk. Doc 101 sec 6.5 established
that no lost id is fitted anywhere in the knob-on file, so they are **gone, not renumbered**.

Set against this, the entire end effect of sec 4 is 16 truncated pairs. **The churn is five times larger than the
thing round 9 was sent to measure**, and it is the channel through which the levers actually move the tagger.

---

## 3. Both ends barely move

Paired displacement over each lever's own intersection, cm, p50 / p90:

| lever | \|Δstop\| | \|Δ tagger_stop\| | \|Δentry\| | `stop_dis` A0 | `stop_dis` L |
|---|---|---|---|---|---|
| K | 0.62 / 5.20 | 0.42 / 3.32 | 0.19 / 1.70 | 0.62 | 0.58 |
| S | 0.96 / 10.29 | 0.76 / 5.71 | 0.37 / 3.00 | 0.60 | 0.59 |
| **A1** | **0.98 / 8.89** | 0.69 / 4.79 | 0.39 / 2.31 | 0.63 | 0.54 |

The median stop moves **0.98 cm** — about 1.6 profile steps, under two wire pitches. `tagger_stop` (the tagger's
own fit end, `:1558`) moves less than `stop` (the graph vertex it is refined to, `:2908`/`:3069`), and `stop_dis`,
the gap between them, is unchanged at ~0.6 cm in both arms: the refinement is not what is moving.

---

## 4. The end classes: truncation is real, rare, and balanced by advance

Perpendicular offset of the lever's stop from A0's own chain (cm):

| lever | p25 | p50 | p75 | p90 | < 0.6 cm |
|---|---|---|---|---|---|
| K | 0.14 | 0.34 | 1.06 | 2.88 | 179 (0.66) |
| S | 0.24 | 0.56 | 1.46 | 3.17 | 133 (0.53) |
| A1 | 0.24 | 0.52 | 1.53 | 3.62 | 141 (0.54) |

Classes, one per pair, priority **reversal > truncated > advanced > same end > off-trajectory**:

| lever | pairs | reversal | truncated | advanced | same end | off-traj | trunc depth p50/p90/max | adv p50/p90 |
|---|---|---|---|---|---|---|---|---|
| K | 271 | 1 (0.00) | 18 (0.07) | 20 (0.07) | 216 (0.80) | 16 (0.06) | 6.28 / 32.34 / 36.67 | 6.54 / 83.62 |
| S | 252 | 7 (0.03) | 9 (0.04) | 22 (0.09) | 186 (0.74) | 28 (0.11) | 11.25 / 16.21 / 18.66 | 5.35 / 17.45 |
| **A1** | 261 | **5 (0.02)** | **16 (0.06)** | **19 (0.07)** | **187 (0.72)** | 34 (0.13) | 8.75 / 22.12 / 23.99 | 5.96 / 22.07 |

**The headline.** 72 % of A1's muons keep their stop within 3 cm. Truncation is 6 % and **advance is 7 %** — the
end moves both ways in nearly equal numbers. That is scatter, not a truncation bias.

"Off-trajectory" (13 %) is not read as a defect: these levers move the fit everywhere, so a stop more than one
wire pitch off A0's polyline may simply be the same end on a slightly different curve.

### 4.1 Two traps in the classifier itself

Both were caught by inspecting a case rather than trusting a bin, and both would have inverted a headline.

**A reversal scores as a giant truncation.** `029107_16/46` has A1's stop **1.78 cm from A0's *entry*** and A1's
entry 1.16 cm from A0's stop: the same track, read in the opposite direction. The first version of the classifier
called it "truncated by 264.76 cm". Reversals are now detected first and reported as their own class (sec 5).

**An advance is invisible to a forward projection.** Projecting A1's stop onto A0's chain bounds the arc by
`muon_len(A0)`, so the offset can never be positive and "advanced" read **0.00 for every lever** — an artifact, not
a measurement. `029107_19/111`, which doc 104 sec 3.1 reported as a muon that *grew* (108.6 → 121.8), projected to
arc = 108.60 = exactly `muon_len(A0)` and was being binned "off-trajectory". Advance is therefore measured by the
**reverse** projection, A0's stop onto the lever's chain. Both directions are computed for every pair.

A third, quieter one: the reversal test must use the record's own `entry_pt`/`stop_pt`, not the profile polyline,
because **the two do not share an origin**. Over 104 A0 candidates, `|chain[-1] − stop_pt|` is 0.000 at both p50
and p90 — the profile ends exactly at the stop — but `|chain[0] − entry_pt|` is **0.67 at p50, 2.01 at p90 and
8.38 at most**: the profile starts *after* the entry point. A perpendicular test near arc 0 measures against the
wrong reference, and an interim version lost 3 of 4 reversals that way. This asymmetry is a property of the
branches and is worth knowing for anything else that joins these two trees.

### 4.2 Where the net shortening actually lives

Sec 2's paired lengths look asymmetric where sec 4 looks balanced: A1 has 107 pairs shortening by more than 5 cm
against 21 lengthening by more than 5, a 5:1 ratio, and a net `sum ratio` of 0.982. Both readings are true only if
the net loss sits **outside** the two end classes. Signed length change per class (no projection enters this):

| lever | class | n | Δ`muon_len` p50 | sum Δ (cm) | share of net |
|---|---|---|---|---|---|
| A1 | reversal | 5 | −11.97 | −62.9 | 0.07 |
| A1 | truncated | 16 | −14.57 | −227.4 | 0.26 |
| A1 | advanced | 19 | +3.27 | +80.1 | −0.09 |
| **A1** | **same end** | **187** | **−2.82** | **−757.6** | **0.88** |
| A1 | off-traj | 34 | −5.27 | +102.4 | −0.12 |
| A1 | **ALL** | 261 | | **−865.4** | 1.00 |

**88 % of the net shortening comes from tracks whose ends did not move.** Truncation contributes 26 %. The
shortening is a small, broad loss — a median 2.82 cm — spread across almost every muon.

### 4.3 That shortening is the levers working, not failing

A chain that loses length while both of its ends stay put has to be getting straighter. Local arc/chord over a
6 cm window (10 × 0.6 cm profile steps) is the wiggle measure; 1.000 is a straight segment, larger is zig-zag.
Whole-track tortuosity is reported beside it because it is dominated by real multiple scattering.

| lever | class | n | local arc/chord A0 → L | **paired Δ p50** | tortuosity A0 → L | paired Δ |
|---|---|---|---|---|---|---|
| K | same end | 204 | 1.0549 → 1.0434 | **−0.0064** | 1.0996 → 1.0855 | −0.0122 |
| S | same end | 175 | 1.0541 → 1.0435 | **−0.0050** | 1.0972 → 1.0852 | −0.0095 |
| **A1** | **same end** | **174** | **1.0547 → 1.0398** | **−0.0108** | 1.1007 → 1.0737 | **−0.0231** |

The per-track difference is the one to read; a difference of medians can hide a shift every track shares.

**The closure.** If straightening is the whole story, then per track Δ`muon_len` should equal
Δtortuosity × (end-to-end distance):

| lever | n | predicted | observed | unexplained residual p50 (p10, p90) |
|---|---|---|---|---|
| K | 204 | −1.65 cm | −1.53 cm | +0.09 (−0.99, +2.00) |
| S | 175 | −1.26 cm | −1.28 cm | +0.16 (−1.19, +1.89) |
| **A1** | 174 | **−3.18 cm** | **−3.03 cm** | **+0.19 (−1.84, +2.31)** |

It closes. The 1.8 % net length loss is zig-zag removal, which is precisely what doc 101 built these knobs to do
(it measured the zig-zag median falling 4–7 % on PDHD data, sec 6.4). **A shorter muon here is a better-measured
muon, not a truncated one.**

---

## 5. What the measurement did find: five muons read backwards

| lever | reversals | keys |
|---|---|---|
| K | 1 | `029107_10/80` |
| **S** | **7** | `028084_22/103`, `029107_10/80`, `029107_11/74`, `029107_16/46`, `029107_2/39`, `029107_21/51`, `029107_6/53` |
| A1 | 5 | `029107_11/74`, `029107_16/46`, `029107_2/39`, `029107_20/17`, `029107_20/24` |

A1's five, with the record's own end branches:

| key | len A0 | len A1 | \|s₁−e₀\| | \|e₁−s₀\| | Michel A0 → A1 | `is_stm` A0 → A1 |
|---|---|---|---|---|---|---|
| `029107_11/74` | 26.1 | 16.7 | 4.65 | 0.18 | 0 → 0 | 0 → 0 |
| **`029107_16/46`** | 266.2 | 235.3 | **1.78** | **1.16** | 1 → 1 | **0 → 1** |
| `029107_2/39` | 41.2 | 29.2 | 1.17 | 4.10 | 0 → 0 | 0 → 0 |
| `029107_20/17` | 32.1 | 18.6 | 2.54 | 5.63 | 0 → 0 | 0 → 0 |
| `029107_20/24` | 58.3 | 61.1 | 1.55 | 2.10 | 0 → 0 | 0 → 0 |

This is a worse failure mode than truncation: a reversed muon puts the Bragg peak at the wrong end, so every
stopping-muon quantity built on it is measured from the entry. `029107_16/46` **becomes a stopping-muon candidate
in A1 having not been one in A0**, on a track whose direction A1 reads backwards.

**Attribution: the sampler, not the fit knobs** — S 7, K 1. The class is defined by a constant (`SWAP_CM`, set
to 10 cm because `|chain[0] − entry_pt|` reaches 8.38 cm, sec 4.1), so the attribution carries its own stability
check rather than resting on that choice:

| `SWAP_CM` | 3 | 5 | **10** | 15 | 25 |
|---|---|---|---|---|---|
| K | 0 | 1 | **1** | 2 | 2 |
| **S** | **3** | **5** | **7** | **7** | **7** |
| A1 | 2 | 4 | **5** | 5 | 5 |

S exceeds K at every threshold tried, and the counts are flat from 10 cm up, so the working value sits on a
plateau rather than an edge. The attribution is not an artifact of the constant. As with Michel purity (K 0.880 / S 0.906 / A1 0.895),
the combination is not the sum: A1 has 5 where S alone has 7. That non-monotonicity is recorded as observed and is
**not** explained here.

Five of 261 is 2 %, and this round does not diagnose the mechanism — that is sec 8.

---

## 6. The dQ/dx: the Bragg shape is intact, the absolute scale is not

| lever | both valid | bragg lost / gained | `short_track` flips | Δcontrast p50 | Δ(contrast−expected) p50/p90 | **`plateau_med` relative p50/p90** |
|---|---|---|---|---|---|---|
| K | 221 | 13 / 8 | 9 | +0.0139 | +0.0111 / +0.5879 | +0.0255 / +0.2939 |
| S | 207 | 9 / 7 | 7 | +0.0383 | +0.0353 / +0.7385 | +0.0345 / +0.3784 |
| **A1** | 213 | 10 / 10 | 6 | **−0.0008** | **+0.0085** / +0.5910 | **+0.0377 / +0.4715** |

`contrast = tail_med/plateau_med` in fixed residual-range windows (`StmMichelFunctions.cxx:337`); `expected` is the
muon dQ/dx model at the same `rr` (`:339`), so it barely moves and the difference is a damage measure.
`short_track` (`:320`) halves the plateau window, so its flips are tracked separately — 6 of 261 for A1.

**Two different claims, and only one survives.**

* **Shape: unchanged.** A1's median Δcontrast is −0.0008 and Δ(contrast − expected) is **+0.0085**, i.e. very
  slightly *better*. `bragg_valid` is lost by 10 and gained by 10 — a wash.
* **Scale: moved.** `plateau_med` rises by a median **3.8 %** and by **47 % at p90**. Docs pdhd/16, 17, 29 and 50
  calibrate on the absolute dQ/dx scale, so this is the real blast radius of a PDHD flip, and it is not small.

### 6.1 Correction to doc 104 sec 3.3 (and doc 105, which carried it)

Doc 104 sec 3.3 stated that the truncation *"corrupts the stopping-muon dQ/dx near the end independently of the
Michel grade"*, and doc 105 repeated it. At aggregate level **that is now measured and not supported**: the Bragg
shape is unchanged. The claim was a reasonable inference from five cases and it did not survive the population.

What replaces it is narrower and still real: the absolute dQ/dx **scale** moves ~4 % in the median, and 16 of 261
muons are genuinely truncated with a median depth of 14.6 cm. A pointer to this section has been added to doc 104.

---

## 7. What this means for the PDHD Michel question

Sec 7 of the figure splits by whether the lever tagged a Michel — a reco flag, so no hand label is involved:

| lever | stratum | n | truncated | trunc depth p50/p90 | Δ`muon_len` p50 / p10 |
|---|---|---|---|---|---|
| A1 | **michel in A1** | 95 | **9 (0.09)** | 12.68 / 23.62 | **−4.28 / −21.76** |
| A1 | no michel in A1 | 166 | 7 (0.04) | 4.73 / 9.72 | −2.95 / −14.22 |
| A1 | `is_stm` in A0 | 109 | 6 (0.06) | 6.47 / 16.70 | −3.65 / −15.36 |

The association doc 104 found is **real but modest**: clusters where A1 tags a Michel are about twice as likely to
be truncated and shorten roughly half again as much. It holds on 9 clusters.

So doc 104's mechanism is not withdrawn — it is resized. Truncation explains a handful of the Michel false
positives, not the class. Combined with docs 104 and 105, which refuted five admission levers, and with sec 2 here,
**the channel through which the levers cost PDHD its Michel purity is most likely the candidate churn, not the
end geometry** — 80 clusters lost and 72 gained, against 16 truncated. That is a hypothesis this round does not
test; it is named as round 10's question, not as a finding.

**PDHD stays at production.** Nothing in this round changes the doc 103 / doc 104 grade (Michel purity −0.035
against a −0.020 bar), and this round proposes no flip.

---

## 8. Round 10: the two candidates, in order

1. **The candidate churn (recommended).** It is five times the size of every end effect measured here and it is
   the only channel large enough to carry a −0.035 purity cost. The question: *which tagger decision drops the 80
   and admits the 72* — doc 101 sec 6.5 explicitly left this unchased ("Which tagger decision drops a cluster is
   not chased"). It needs no new arm; `reject_bits`, `has_pass`, `topology_cleared_bits` and the per-check
   counters are already persisted in `T_stm_michel` on all four arms.

   > **Correction (round 10, doc 107 sec 4.1).** This was a *size* argument and size was the wrong test. Round 10
   > ran the direct one: **8 of A1's 9 Michel false positives are clusters production also scored**, only 1 is a
   > gained candidate, and **20 of its 77 true positives are gained**. The churn is large, benign, and on the
   > efficiency side — it is where the +0.085 efficiency gain comes from, not the −0.035 purity cost. Chasing it
   > was still productive (it identified the mechanism: the STM evaluation's `flag_pass` flipping, 60–67 % of the
   > churn in both directions), but the ranking below should not be reused.
   >
   > The pointer in the sentence above is also wrong: `reject_bits` / `has_pass` / `topology_cleared_bits` live in
   > `T_stm_michel`, and a **lost cluster has no `T_stm_michel` row at all**, so those columns cannot describe it.
   > The churn is decided upstream, and the record that answers it is `T_stm_pass` / `T_stm_eval` in
   > `tracking-stm.root` (doc 107 sec 1.2).
2. **The direction reversals.** Smaller (5 of 261) but a worse failure, sampler-attributed, and cheap to localise
   because the cases are named in sec 5. Worth doing whether or not PDHD ever flips, since S is already **live in
   PDVD production** (toolkit `8fc6070e`) — the same reversal mechanism should be counted there.

What is explicitly **not** recommended: a fix aimed at the end. Sec 4 says the end is stable, sec 4.3 says the
shortening is the levers working, and sec 5 of the figure shows `retreat_len` moving by **0.00 at both p50 and
p90** with A0 retreating on 5 pairs and A1 on 2 — the doc-57 retreat step at `CheckSTM_Michel.cxx:2975` is not the
site, and a fix aimed there would miss.

---

## 9. Not concluded

* **Why the churn happens.** Sec 2 counts it and does not explain it. Round 10.
* **Why the sampler reverses direction.** Sec 5 names five cases and stops.
* **Whether the +3.8 % `plateau_med` shift is right or wrong.** It is a change, measured against A0, with no
  truth anchor — PDHD's scan records carry only **3 pins** across all of `smx27`/`smx28`/`own103h`/`own103h2`,
  against PDVD's 25/21 (doc 101 sec 6.5). Nothing here can say which arm's scale is closer to the truth. Any
  future PDHD scan should request pins.
* **Non-additivity.** K/S/A1 is non-monotonic in Michel purity (0.880 / 0.906 / 0.895) and in reversals (1 / 7 / 5).
  Observed twice, explained neither time.
* **PDVD.** Every number here is PDHD. S and the fit knobs are live in PDVD production; the reversal count on
  PDVD is unmeasured.

---

## 10. Files

* **Script (new):** `scripts/d106_end_geometry.py` — the whole round, one command, three asserts.
* **Figure (new):** `figs/106_end_geometry.txt`.
* **Amended:** `104_pdhd-michel-admission-and-end-truncation.md` sec 3.3 — a pointer to sec 6.1 here.
* **Arms (existing, unmodified):** `pdhd/work/*_{d101hnew,d101hkf,d102hocs,d102hcs}`, 61 events each.
* **Unchanged:** no toolkit code, no jsonnet, no scan record, no new arm, no production default.
