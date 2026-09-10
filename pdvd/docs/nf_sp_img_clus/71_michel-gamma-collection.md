# 71 — P4: the Michel's isolated gamma blobs (`michel_gamma_collect`)

**Update (2026-09-10, §11): on the owner's call the radius is 50 cm in PDVD
production.** The argument: X0 = 14 cm, a gamma's conversion length ~18 cm, and
three of them ~54 cm. The bare-production arm at 50 cm reads `is_stm`
**225 / 7 / 51** (purity 0.970, efficiency 0.815) against 223 / 8 / 53 at 35 cm.
Exactly three verdicts move, all the way the owner's record says
(`039253_10/93` THRU 1 → 0; `039349_64/24`, `039349_64/61` stoppers 0 → 1), and
`michel_found` is unchanged. The owner's gamma tags inside the Michel object
reach **104 of 159** (65.4 %) at purity 0.933. The same three over-clustered
cores are the only Michels above 60 MeV, and they gain nothing. The C++ matches
its offline twins with 0 mismatches. The widening moves 107 of 578 candidates'
fits (doc 53's preload perturbation) and `mabc-pr.zip` on 91 of 120 events.

**Status (2026-09-10): built behind a default-OFF knob, gated, and switched on
in PDVD production at 35 cm (§9); the radius then became 50 cm (§11). PDHD
stays OFF.**

All seven pre-stated criteria (§4) hold.
- **Knob off** is byte-identical on both detectors.
- **Knob on at 35 cm** changes no pre-existing output: `mabc-pr.zip` 120/120 and
  calib 119/119 are identical; so are 578/578 candidates on all 131 shared
  `T_stm_michel` branches, and every point row of role ≠ 4. It only adds six
  branches and 286 role-4 rows on 54 candidates.
- **Recall:** the owner's gamma tags inside the Michel object go from 9 to
  **75 of 159** (5.7 % → 47.2 %).
- **Purity:** of the tagged blobs it takes, **0.958** (68 of 71) are gamma or
  Michel. The three delta / other blobs are named.
- **Energy:** the three over-clustered cores (76.8, 76.2 and 205.7 MeV) gain
  nothing, and no total goes above 60 MeV.
- **Rule check:** 0 mismatches against the C++ log.

At 60 cm the recall is 74.8 % at purity 0.926. That radius widens admission and
so moves 112 candidates' fits. It is the owner's decision (§8).

One correction to doc 70 §10.4 comes out of this round (§7). PDVD production
runs at 35 cm admission, and its census on the record is `is_stm` 223 / 8 / 53,
not the 225 / 7 / 51 of the survey arms.

The owner, after P1 (doc 70 §10): "can you proceed to P4? please update the
70*.md, for the rest of the requirements are similar to the previous round. If
the improvements are confirmed, please turn on the knobs for production for
PDVD … for each one please have its own md file." And, while this round was
being designed: "I have hand scan results, but may not cover everything, the
basic idea is 1. direction along the Michel electron 2. dots near by 3.
Be careful of energy, overclustering may lead to a huge energy."

This is P4 of doc 70 §5.3 / §6. From P4 on, each proposal gets its own doc.
P1 stays in doc 70 §9–§10.

Companion docs: pdvd/70 §5 (the question and the first sizing), 51 (the
capture-gamma stage, role 5), 53 (the survey and its preload perturbation), 15
(the Michel as one object).

---

## 0. Repro

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img
export STM_SCAN_RECORD=$PWD/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json

# sec 2: the sizing that chose the rule, read-only on the survey arm of doc 70 sec 10
python3 pdvd/docs/nf_sp_img_clus/scripts/d71_p4_sizing.py --prep /home/xqian/tmp/d71/prep_d71vsp \
        --out /home/xqian/tmp/p4/sizing                     # -> /home/xqian/tmp/p4/sizing.txt

# sec 4: build (toolkit clus/), twice for the new-symbol doctest link, then the tests
(cd /nfs/data/1/xqian/toolkit-dev/toolkit && wcbuild; wcbuild; ./build/clus/wcdoctest-clus)   # 363/363
cp -a /nfs/data/1/xqian/toolkit-dev/local/lib /home/xqian/tmp/p4/libpin_p4      # libWireCellClus 1af886223e13

# sec 5-6: six arms on two pins, bare production config, then every gate and the grading
pdvd/docs/nf_sp_img_clus/scripts/d71_p4_arms.sh     # p4vleg p4hleg (P1 pin) | p4voff p4hoff (P4, off) | p4v35 p4v60 (P4 on)
pdvd/docs/nf_sp_img_clus/scripts/d71_p4_gates.sh | tee /home/xqian/tmp/p4/gates.log

# sec 11: the owner's 50 cm radius -- AFTER the jsonnet carries michel_gamma_radius_cm: 50.0,
# one bare-production arm (p4v50) on the P4 pin, then prep, census, movers and grading
JOBS=24 pdvd/docs/nf_sp_img_clus/scripts/d71_p4_r50.sh > /home/xqian/tmp/p4/r50.log 2>&1
```

---

## 1. The question, and what the object holds today

Doc 70 §5 has the mechanics. The Michel object is:
- the attached stop arm(s);
- whatever the shower walk reaches by graph connectivity;
- the companion pieces whose cluster comes within `michel_dot_radius_cm`
  (15 cm) of the final stop.

Its energy `michel_ke_best` is that object's dQ/dx energy plus the charge of its
unfitted pieces. Beyond 15 cm, a same-bundle blob of ≤ 10 cm in the ring
15–35 cm can become a **capture gamma** (role 5, doc 51), a separate object whose
energy never enters `michel_ke_*`. Everything else the admission brought (35 cm
in production) is fitted and then left with no role at all.

On the scan arms (the survey, 60 cm), the owner tagged 345 segments `gamma`. On
found-Michel events, about 150 of 165 of them sit outside the Michel object
(doc 70 §5.2).

## 2. Sizing — the owner's three criteria against the tags

`d71_p4_sizing.py` works on `prep_d71vsp`, doc 70's P1 arm with the survey on,
so every unclaimed fragment to 60 cm has a point row. On the 95 candidates with
a fitted Michel (conn 1 or 2), it measures each unclaimed companion **cluster**
the way the C++ does:
- `d_stop`, the closest fit point to the final stop;
- the cosine between (stop → blob centroid) and (stop → centroid of the Michel
  object's rows), which is criterion 1;
- the distance to the Michel object and to the muon body (`rr ≥ 5 cm`, plus
  the deltas).

It then joins each cluster to the owner's tags. Each tag is resolved in its
record's own baseline prep (smx4 → smx3 → smx1a): 0 were ambiguous and 12 were
unresolved.

| tag | clusters | < 15 cm | 15–35 cm | 35–60 cm | cos > 0.5 (60°) | closer to the Michel than to the body | ≤ 10 cm long |
|---|---:|---:|---:|---:|---:|---:|---:|
| gamma | 151 | 4 | 83 | 64 | **125** | **140** | 149 |
| delta / other | 69 | 7 | 20 | 42 | 21 | 24 | 66 |
| untagged | 5 | 2 | 3 | 0 | 1 | 2 | 5 |
| michel | 2 | 2 | 0 | 0 | 2 | 2 | 2 |

The owner's direction criterion and the body test each separate the gamma
tags from the "delta / other" tags by about a factor of three. Only 5 of the
227 clusters are untagged, so on these items the tags are nearly complete.

The grid below uses a length cap of 10 cm and a per-blob energy cap of 20 MeV.
The energy is approximate here: Σ dQ/dx/MIP × 2.1 MeV/cm × step, rescaled ×1.84
to the C++ `stop_gamma_ke_tot` on the 8 items where the two can be compared.

| radius | cone | body test | gamma | delta / other | untagged | purity |
|---:|---|---|---:|---:|---:|---:|
| 35 | none | no | 84 | 24 | 5 | 0.778 |
| 35 | none | yes | 78 | 4 | 2 | 0.951 |
| **35** | **60°** | **yes** | **71** | **3** | 1 | **0.959** |
| 60 | none | yes | 136 | 21 | 2 | 0.866 |
| 60 | 60° | yes | 115 | 9 | 1 | 0.927 |
| 60 | 45° | yes | 108 | 5 | 1 | 0.956 |

Energy (criterion 3) at 60 cm with no total cap: the Michel total's maximum goes
51.1 → 67 MeV, and 3 candidates end above the 52.8 MeV endpoint (at 35 cm: 2,
max 58.6). Those are approximate energies, but they are enough to say a guard
is needed.

## 3. The design, and the radius fork

**What P4 does.** It runs after the capture-gamma stage, so role 5 and its body
snapshot are untouched, and before the survey. It offers every **unclaimed**
companion cluster with a fitted segment to one pure function,
`stm_michel_gamma_gate`. The gates, first failure reported:

| code | test | default |
|---:|---|---|
| 1 | `d_stop` > `michel_gamma_radius_cm` | 35 |
| 2 | cluster length > `michel_gamma_max_len_cm` | 10 |
| 3 | cos < `michel_gamma_cos_min` | 0.5 = 60° |
| 4 | `d_body` ≤ `d_mich` | |
| 5 | blob energy > `michel_gamma_max_ke_mev` | 20 |
| 6 | a non-finite input | |

Once `michel_found` and `michel_ke_best` are final (after the T2c / T3c
vetoes and after the coverage test, which reads every point row),
`stm_michel_gamma_take` takes the survivors nearest first. It keeps the Michel
plus its blobs at ≤ `michel_gamma_total_ke_max_mev` (60 = the 52.8 MeV endpoint
plus resolution). A blob that would cross that line is skipped. A core already
above it collects nothing, so an over-clustered Michel is never made worse.

**What P4 writes, and what it does not.**
- The taken blobs get **role-4 rows**, the documented free slot, which the
  display already draws as "dots".
- Six branches, written only when the knob is on: `n_michel_gammas`,
  `n_michel_gamma_cand`, `n_michel_gamma_capped`, `michel_ke_gamma`,
  `michel_ke_total` = `michel_ke_best + michel_ke_gamma`, and
  `michel_gamma_dis_max`.
- It does not `set_pdg`, builds no Shower and adds no particle-flow node.
- `michel_ke_best`, `michel_found`, every reject bit and `is_stm` are never
  written.
- So at equal admission, `mabc-pr.zip`, the calib JSON and every pre-existing
  branch must be identical ON vs OFF. That is a stronger gate than P1's.
- Showing the blobs in the Bee PF tree is a named follow-up, not part of this
  change.

**The radius fork.**
- **35 cm (the default):** production admission is 35 cm, measured from the
  tagger's stop. At the 35 cm default the knob admits nothing new, so it is
  verdict-neutral.
- **60 cm:** P4 widens admission to 60 cm, which is what the survey does. Doc
  53 §6.2 measured that admitting those companions moves **20 %** of PDVD
  candidates' fits through `preload_clusters`. Its one large flip,
  `039349_64/61` (contrast 0.53 → 1.50), was flagged "scan before trusting
  60 cm" and never scanned.
- The flip candidate is therefore 35 cm. 60 cm is run and graded as evidence,
  and it is the owner's decision.

**A known limitation.** P4 measures from the final stop, but admission is from
the tagger's stop. At 35 cm, a blob within 35 cm of the final stop that lies
beyond 35 cm of the tagger's stop was never admitted. §7 counts these.

## 4. Pre-stated confirmation criteria for the PDVD flip

Written before any P4 output was looked at. The arms were launched at 10:41:25
(the runner's stamp files), and this doc was created at 10:45:25 with this
section in it. The first look at a P4 DEBUG line came after that.

1. **OFF gate**: `p4vleg` (P1 pin) vs `p4voff` (P4 pin, knob off), PDVD and PDHD.
   - `mabc-pr.zip` member content identical;
   - calib JSON identical;
   - every `T_stm_michel` branch identical;
   - every `T_stm_michel_pts` row identical.
2. **Neutrality**: `p4voff` vs `p4v35`.
   - `mabc-pr.zip` and calib identical;
   - every pre-existing `T_stm_michel` branch identical on every candidate;
   - every `T_stm_michel_pts` row of role ≠ 4 identical; role-4 rows only added.
3. **Recall**: at 35 cm, gamma tags in the Michel object ≥ 3× the knob-off count
   (role 3 only).
4. **Purity**: tagged-good / tagged role-4 segments ≥ 0.90.
5. **Energy**: no `michel_ke_total` above 60 MeV (by construction); the count
   above 52.8 MeV is reported.
6. **Rule check**: 0 decision mismatches between the C++ DEBUG lines and the
   Python twins; the taken blobs' geometry re-derives from the payload.
7. `wcdoctest-clus` passes, the pins are unchanged, and there are 0 loader
   deaths.

All hold → `michel_gamma_collect: true` in `pdvd/wct-pr-perevt.jsonnet`, with
the five sub-knobs unset at their C++ defaults. Any failure → no flip, and the
failing items are named.

## 5. What was built (toolkit `d227d5b8`)

* `StmMichelFunctions.{h,cxx}`: `stm_michel_gamma_gate` and `stm_michel_gamma_take`,
  pure, with three new doctest cases:
  - every gate's boundary and the order they fire in;
  - a non-finite input rejected first;
  - the cap: skip-and-continue, exactly-at-the-cap taken, a core already over
    the cap takes nothing, NaN.
* `CheckSTM_Michel.cxx`:
  - Six keys in `configure()` and `default_configuration()`, pinned in
    `doctest_check_stm_michel_defaults.cxx`.
  - The admission radius is `max(old, michel_gamma_radius_cm)` only when the
    knob is on; off, it is the old expression, untouched.
  - Phase 1 (after the capture-gamma stage) measures and gates, and reserves the
    survivors in `rec.claimed` so the survey leaves them alone.
  - Phase 2 (after the P1 rule, just before publish) takes them and writes the
    rows. With the survey on, a reserved blob that was not taken gets its role-6
    row back with rej 11 (energy guard) or 12 (Michel vetoed).
  - Three DEBUG lines (`michel-gamma:`, `michel-gamma-cl:`,
    `michel-gamma-take:`) carry every number the gates read.
  - Branches are persisted only when the knob is on.
* `prep_stm_michel_scan.py` carries the six branches into the payloads.

---

## 6. Gates

Six arms on the same input (`d16vnu` / `d16hnu`), all bare production config
(no survey TLA), launched concurrently at 10:41:25, 30 jobs, and done in about
15 minutes. Labels: `p4vleg` / `p4hleg` on the P1 pin; `p4voff` / `p4hoff`,
`p4v35` and `p4v60` on the P4 pin. Gate outputs are in `/home/xqian/tmp/p4/`
(`gates.log`, `g_pdvd.txt`, `g_pdhd.txt`, `g_v35.txt`, `g_v60.txt`,
`score_p4.txt` / `.json`); preps are `/home/xqian/tmp/p4/prep_p4v{leg,off,35,60}`.

| gate | result |
|---|---|
| unit tests | `wcdoctest-clus` 363 / 363 (3 new cases); the first `wcbuild` fails only at the doctest link against the old installed lib (the new-symbol trap), and the second passes |
| freshness / pins | installed lib 10:39:04, after the last source edit (10:38:15). Pins md5 `0d7027b23351` (P1) and `1af886223e13` (P4), unchanged before and after every arm; 0 loader deaths. The runners' one "incomplete" event, `039252_11`, has no STM candidate ("nothing to reconstruct") on every arm and binary, as in doc 70 §10 |
| compiled config | `p4v35`'s `CheckSTM_Michel` block differs from production by exactly one line, `michel_gamma_collect: true`; `p4v60` adds `michel_gamma_radius_cm: 60` |
| **OFF gate PDVD** (`p4vleg` vs `p4voff`, 120 events) | `mabc-pr.zip` member content 120 / 120; calib 119 / 119 (`039252_11` writes none on either); **578 / 578 candidates bit-identical on all 131 `T_stm_michel` branches**; `T_stm_michel_pts` geometry and roles 578 / 578 |
| **OFF gate PDHD** (`p4hleg` vs `p4hoff`, 61 events) | zip 61 / 61; calib 61 / 61; **325 / 325 on 130 branches**; points 325 / 325 |
| **neutrality** (`p4voff` vs `p4v35`) | zip **120 / 120**; calib **119 / 119**; **578 / 578 on all 131 shared branches**, plus 6 new (`n_michel_gammas`, `n_michel_gamma_cand`, `n_michel_gamma_capped`, `michel_ke_gamma`, `michel_ke_total`, `michel_gamma_dis_max`); every `T_stm_michel_pts` row of role ≠ 4 identical on 578 / 578; role-4 rows 0 → 286 on 54 candidates |
| **rule check** (`d71_p4_score.py` §D) | `p4v35`: 194 blob lines on 89 candidates, gate codes {0: 100, 1: 6, 2: 2, 3: 71, 4: 14, 5: 1}, **0 gate and 0 take mismatches**. For the 81 taken blobs, `d_stop` / `cos` / `d_mich` / `d_body` re-derived from the payload's rows agree with the C++ log to ≤ 0.0103 cm / 0.0009 (the rows are rounded to 0.01 cm); the Michel direction agrees to 0.0032. `p4v60`: 383 lines, 0 / 0 mismatches |
| `census_score.py --check` | 0 of 14 differ |

The first run of the point-row neutrality check crashed on `039252_11`, which
has no `T_stm_michel_pts` tree at all. The committed script now skips such an
event, and the re-run is appended to `gates.log`.

## 7. What P4 does, graded on the owner's tags

**Recall** (`d71_p4_score.py` §A). The denominator is every gamma tag on a
scan-Michel item whose Michel the arm found: 159 tags on 76 items.

| arm | role 3 (the object) | role 4 (P4) | role 5 (capture γ) | no row | in the Michel object |
|---|---:|---:|---:|---:|---:|
| `p4voff` (production before) | 9 | 0 | 8 | 142 | **5.7 %** |
| **`p4v35`** | 9 | **66** | 8 | 76 | **47.2 %** |
| `p4v60` | 9 | 110 | 8 | 32 | 74.8 % |

- At 35 cm, 45 items gain gamma tags in the object and none loses one.
- Every one of the 44 tags that 60 cm adds lies beyond 35 cm of the final stop.
- **None** is lost to the tagger-stop / final-stop offset of §3's limitation.
- The offline sizing predicted 71 gamma / 3 delta at 35 cm and 115 / 9 at 60 cm,
  counting clusters on the survey arm. The C++ takes 67 / 3 and 112 / 9, counting
  segments on the production arms.

**Purity** (§B), over role-4 segments on judged items:

| arm | role-4 segments | gamma | michel | delta / other | untagged | on unjudged items | purity |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`p4v35`** | 82 | 67 | 1 | 3 | 0 | 11 | **0.958** |
| `p4v60` | 143 | 112 | 1 | 9 | 0 | 21 | 0.926 |

- **The three at 35 cm:** `039252_15/81` seg 281008, `039252_3/74` 225041 and
  `039349_81/51` 200011, all "delta / other".
- **Items with no Michel on the record where P4 fires** (both STM_ONLY): the
  Michel the chain found there is itself a `michel_found` false positive.
  - `039252_16/98` (kind not set, 2 segments);
  - `039349_81/51` ("detached dots", 1 segment, the delta above).
- **Untagged:** on judged items every role-4 segment carries a tag. The 11
  (35 cm) and 21 (60 cm) role-4 segments on items the record does not judge are
  unmeasured. The owner's tags do not cover everything, and this is where the
  gap sits.
- **The two tables count differently.** The recall table counts gamma tags on
  scan-Michel items (66 land in role 4, each on its own segment; no segment
  absorbs two). The purity table counts role-4 segments on every judged item.
  Its 67th gamma segment, and its one Michel-tagged segment, are the two blobs
  on `039252_16/98`, an item the record says has no Michel.

**Energy** (§C). The Michel's own `michel_ke_best` is unchanged;
`michel_ke_total` adds the blobs.

| arm | Michels with ≥ 1 blob | blobs | refused by the 60 MeV guard | total q50 / q90 | > 52.8 MeV (best → total) | > 60 MeV |
|---|---:|---:|---:|---|---|---|
| **`p4v35`** | 53 of 156 | 81 | 1 | 23.1 / 44.8 (best 20.9 / 42.5) | 3 → 6 | 3 → 3 |
| `p4v60` | 75 of 157 | 141 | 4 | 24.8 / 47.5 | 3 → 10 | 3 → 3 |

- **The three Michels above 60 MeV** are the over-clustered cores already
  there: `039252_15/77` 76.8, `039349_46/58` 76.2 and `039349_81/54` 205.7 MeV.
  The guard gives each of them **nothing**, at both radii.
- **The three new totals above the 52.8 MeV endpoint at 35 cm:**
  `039253_9/111` 42.7 → 58.1, `039349_31/51` 51.1 → 57.1 and `039349_63/41`
  46.3 → 56.7. The last is MESSY (owner: "overclustering"), and its second blob
  was refused by the guard.
- **The largest additions:** `039253_3/29` +18.8, `039253_2/117` +17.6,
  `039253_9/111` +15.5 and `039349_54/56` +14.9 MeV.

**The production census, reconciled.** `census_score.py` on the record, payload
population 544:

| arm | admission | `is_stm` TP / FP / FN | purity | efficiency | F1 | `michel_found` |
|---|---|---|---:|---:|---:|---|
| `p4vleg` = `p4voff` = `p4v35` (production) | 35 cm | **223 / 8 / 53** | 0.965 | 0.808 | 0.880 | 133 / 12 / 25 |
| `p4v60`; = doc 70's `d71vsp` (survey) | 60 cm | 225 / 7 / 51 | 0.970 | 0.815 | 0.886 | 133 / 12 / 25 |

- **The correction.** Doc 70 §10.4 called 225 / 7 / 51 "what PDVD now
  produces". It is what the 60 cm-admission arms produce. Production runs
  without the survey, so at 35 cm, and reads 223 / 8 / 53.
- **The three items that move with the admission radius:**
  - `039253_10/93` (THRU, medium) is `is_stm` 1 at 35 cm and 0 at 60;
  - `039349_64/24` (STM_MICHEL, medium) is 0 at 35 cm and 1 at 60;
  - `039349_64/61` (STM_ONLY, medium) is 0 at 35 cm and 1 at 60. This is doc
    53 §6.2's "scan before trusting 60 cm" flip (contrast 0.53 → 1.50); the
    smx1a record calls it a stopper, so 60 cm has it right.
- `039349_42/41` (MESSY) enters the sheet only at 60 cm.
- `michel_found` is identical at both radii.

## 8. The 60 cm evidence — the owner's decision

At `michel_gamma_radius_cm` 60, P4 widens companion admission to 60 cm, the
same pool the survey fits.

**What it does on this record:**
- Gamma tags in the object reach 74.8 % (vs 47.2 %), at purity 0.926 (vs 0.958).
- `is_stm` goes 223 / 8 / 53 → **225 / 7 / 51**: +2 stoppers, −1 false
  positive, the three items of §7.
- `michel_found` is unchanged.
- 10 Michel totals end above 52.8 MeV, none above 60.
- The six extra delta / other blobs are named in `score_p4.txt`.

**What it costs:** it is no longer rows-only.
- 112 of 578 candidates move on some branch: `ks_*`, `contrast`, `muon_ke_*`,
  `plateau_med`, and the stop point on 18.
- `mabc-pr.zip` differs on 107 of 120 events.

That is doc 53's preload perturbation, and on this record it happens to move
the verdict the right way on all three items it touches. Making production
equal to the arms every census since doc 68 was graded on is an argument for
it. It is not flipped here: a verdict-moving change goes to the owner. The
one-line change is `michel_gamma_radius_cm: 60.0` in the same bag.

## 9. The production flip — 35 cm, on the owner's standing go

The owner's instruction ("If the improvements are confirmed, please turn on the
knobs for production for PDVD") was given before this round, conditional on
§4's criteria. All seven hold, so the flip is one key in the PDVD
`stm_michel_knobs` bag, after `topology_clears_sparse`. The comment carries the
C++ default, the guarantee and the graded result:

```jsonnet
        michel_gamma_collect: true,
```

The five sub-knobs stay **unset** at their C++ defaults (35 cm, 10 cm, cos 0.5,
20 MeV, 60 MeV, what `p4v35` ran), for the inert-key reason docs 58, 61 and 70
give. PDVD production is therefore exactly arm `p4v35`: every pre-existing
output as before, plus the role-4 blobs and the six branches. The baseline prep
for later rounds is `/home/xqian/tmp/p4/prep_p4v35`.

Compiled-config proofs on the committed file (`/home/xqian/tmp/p4/flip/`):
* **flip-equivalence:** the pre-flip file plus `-S stm_michel_extra={michel_gamma_collect:true}`
  (`pre_B.json`) against the flipped file (`post_A.json`): **0 lines**.
* **OFF path:** both files with `michel_gamma_collect:false` (`pre_C`, `post_C`): **0 lines**.
* **what moved:** pre-flip (`pre_A`) against flipped: exactly `+ "michel_gamma_collect": true`.

The file was edited with no PDVD job of this tree in flight: the wire-cell
processes running were another user's PDHD jobs. PDHD stays OFF, since it has
no hand-scan record.

## 10. Next, and observations

* **The 60 cm radius** (§8) is a one-line decision for the owner. On this record
  it is +2 stoppers / −1 FP / +27 % gamma recall. Its cost is a
  verdict-moving, zip-moving admission change on 112 candidates.
* **The particle flow.** The blobs are members of the STM module's Michel
  object (rows, energy), but not of the Bee PF tree: no PDG, no Shower. Putting
  them there, as the Michel's e⁻ children or as γ nodes, would move
  `mabc-pr.zip`. It is a separate, named follow-up, best after the owner has
  looked at role 4 on the display.
* **`michel_found` false positives** carry blobs too (`039252_16/98`,
  `039349_81/51`). P4 inherits the Michel verdict; it does not judge it.
* **Next proposal: P3b + P2** (doc 70 §6), in its own doc: the attached-gate
  operating points and the T2c exemption for a hard-turning Michel. The
  `topology_michel_ke_max` cap of doc 70 §10.3 stays open; §7's energy table
  shows the same three over-clustered cores are the only Michels above 60 MeV.

---

## 11. The owner's 50 cm radius (2026-09-10)

The owner, after reading §8: "The radiation length is 14 cm, for gamma, it
would be 18 cm, so 3-sigma stel would be 54 cm. I feel we should have a radius
cut at 50 cm, can you do that? commit and push. Turn it on".

The argument, in numbers:
- X0 = 14 cm in liquid argon.
- A gamma's mean free path to pair conversion is 9/7 X0 ≈ 18 cm.
- By three conversion lengths (≈ 54 cm), 1 − e⁻³ = 95 % of the gammas have converted.

Past that, a blob is more likely unrelated than the Michel's. The cut is
50 cm. This is the owner's decision, so §4's criteria do not gate it. The arm
below grades it by name.

**The change.** It is config only; the C++ is `d227d5b8`, unchanged. One key
goes in the PDVD `stm_michel_knobs` bag, after `michel_gamma_collect`, with the
argument in its comment:

```jsonnet
        michel_gamma_radius_cm: 50.0,
```

Past the 35 cm capture-gamma radius, this key **widens companion admission** to
50 cm. The added companions enter the candidate's fit through `preload_clusters`
(doc 53 §6.2), so, unlike the 35 cm flip, pre-existing outputs and verdicts can
move. The capture-gamma ring (35 cm) and the Michel piece radius (15 cm) keep
their own tests.

Compiled-config proofs on the committed file (`/home/xqian/tmp/p4/r50/`):
* **flip-equivalence:** the pre-change file plus `-S stm_michel_extra={michel_gamma_radius_cm:50.0}`
  (`pre_B`) against the new file (`post_A`): **0 lines**.
* **what moved:** pre-change (`pre_A`) against new: exactly `+ "michel_gamma_radius_cm": 50`.
* **P4 forced off:** both files with `michel_gamma_collect:false` differ only by
  that key. The key is inert there: the C++ reads it only inside the
  `michel_gamma_collect` ternary of the admission radius and in the
  collect-gated phase 1.

The file was edited with no PDVD job of this tree in flight. The arm below
(`p4v50`, P4 pin `1af886223e13`, **no TLA**) was launched after the edit and
therefore is production.

### 11.1 What 50 cm does — arm `p4v50`, which is production

Outputs: `/home/xqian/tmp/p4/r50.log`, `g_v50.txt`, `score_p4v50.txt`,
`score_p4_r50.txt` / `.json`; prep `/home/xqian/tmp/p4/prep_p4v50`.

* **The arm:** 120 / 120 events, 0 loader deaths, pin `1af886223e13` before and
  after. As on every arm, the runner's one "incomplete" event is the
  no-candidate `039252_11`.
* **What the widening moves** (against `p4v35`, production at 35 cm):
  - `mabc-pr.zip` differs on 91 of 120 events.
  - 471 of 578 candidates are bit-identical on all 137 branches. 107 move, on
    the muon profile quantities (`ks_*`, `contrast`, `plateau_med`,
    `muon_ke_*`) as well as the P4 branches. This is doc 53's preload
    perturbation, about 18 % of candidates.
* **Verdicts, by name, against the record:**

| item | record | `is_stm` 35 cm → 50 cm |
|---|---|---|
| `039253_10/93` | THRU (medium) | 1 → **0** (a false positive removed) |
| `039349_64/24` | STM_MICHEL (medium) | 0 → **1** |
| `039349_64/61` | STM_ONLY (medium) | 0 → **1** (doc 53's "scan before trusting 60 cm" flip) |

  These are exactly the three items §7's correction named, and all three move
  the way the record says. `michel_found` moves on none. `039349_42/41` (MESSY)
  joins the payload population. From 50 to 60 cm no verdict moves at all.

| arm | radius | `is_stm` TP / FP / FN | purity | efficiency | F1 | `michel_found` |
|---|---|---|---:|---:|---:|---|
| `p4v35` | 35 cm | 223 / 8 / 53 | 0.965 | 0.808 | 0.880 | 133 / 12 / 25 |
| **`p4v50` (production)** | **50 cm** | **225 / 7 / 51** | **0.970** | **0.815** | **0.886** | 133 / 12 / 25 |
| `p4v60` | 60 cm | 225 / 7 / 51 | 0.970 | 0.815 | 0.886 | 133 / 12 / 25 |

Production and the survey arms every census since doc 68 was graded on
(60 cm admission) now agree on every verdict.

* **Recall:** the owner's gamma tags inside the Michel object.

| radius | role 3 | role 4 | in the object |
|---|---:|---:|---:|
| off | 9 | 0 | 5.7 % |
| 35 cm | 9 | 66 | 47.2 % |
| **50 cm** | 9 | **95** | **65.4 %** (104 / 159) |
| 60 cm | 9 | 110 | 74.8 % |

  - 59 items gain gamma tags in the object and none loses one.
  - All 29 tags that 50 cm adds over 35 cm lie 35–50 cm from the final stop.
* **Purity:** 124 role-4 segments: 97 gamma, 1 michel, 7 delta / other, and 19
  on items the record does not judge (unmeasured). That is 98 / 105 = **0.933**.
  - The four delta / other blobs beyond 35 cm: `039253_15/45` 148030,
    `039349_48/54` 211011, and `039349_61/51` 97008 and 178009.
  - Items without a Michel where P4 fires: `039252_12/90` (STM_ONLY, detached
    dots, 1 segment) joins `039252_16/98` and `039349_81/51`.
* **Energy:** 69 Michels carry ≥ 1 blob (122 blobs); 3 blobs are refused by
  the 60 MeV guard.
  - Totals above 52.8 MeV: 8, against 3 cores.
  - Above 60 MeV: the same 3 over-clustered cores, which gain nothing.
  - Largest additions: `039349_51/24` +24.1 → 44.3, `039349_42/41` (MESSY)
    +19.1 → 59.6, `039253_3/29` +18.8 → 39.4 MeV.
* **Rule check:** 304 blob lines on 119 candidates, gate codes {0: 152, 1: 6,
  2: 4, 3: 121, 4: 20, 5: 1}. **0 gate and 0 take mismatches.** The taken blobs
  re-derive from the payload to ≤ 0.0106 cm.
* `census_score.py --check`: 0 of 14 differ.

**Baseline prep for later rounds:** `/home/xqian/tmp/p4/prep_p4v50`, which is
PDVD production.
