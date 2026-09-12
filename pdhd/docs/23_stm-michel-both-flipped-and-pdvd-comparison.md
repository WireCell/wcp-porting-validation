# doc pdhd/23 — the owner turns on both held sets, and how PDHD now compares to PDVD

**Status:** the **wide Bragg anchor and P1 are flipped** in PDHD production on the owner's ruling
of 2026-09-12, graded as **one unit** (arm `h23a`). `is_stm` 82/1/65/109 → **95/3/52/107**,
purity 0.988 → 0.969, efficiency 0.558 → **0.646**; on owner+high-confidence truth
60/1/10/58 → **65/1/5/58**, purity 0.984 → **0.985**, efficiency 0.857 → **0.929**.
`michel_found` unchanged at 68/2/18. The confirmation arm `h23conf` (the flipped file, no TLA) is
**341/341 bit-identical** to `h23a`, so production runs what was measured. §4 compares the
finished PDHD chain to PDVD's.

**§6, added after the owner scanned the three false positives (2026-09-12):** two of the three
were **not false**. On the corrected record `smx23` production reads **96/1/52/107, purity
0.990** — and *neither* flipped set costs any purity at all.

## 0. Repro

```sh
I=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img ; H=/home/xqian/tmp/h22
PIN=/home/xqian/tmp/p65/libpin_p65        # toolkit d65f8165, libWireCellClus md5 3e23bf8a
S=$I/pdvd/docs/nf_sp_img_clus/scripts/d53_run_arms.sh

# the combined arm: the doc-22 production file PLUS the four keys, as one unit
ARM=h23a DET=pdhd SRC=d16hnu JOBS=8 PIN=$PIN LOGD=$H/arm_h23a \
  PR_TLA="-S stm_michel_extra={bragg_wide_anchor_cm:8.0,bragg_wide_anchor_rise_min:1.3,bragg_wide_anchor_tail_max:0.8,topology_stop_evidence:true}" \
  bash $S

python3 $I/pdhd/docs/scan/h21/d21_grade.py h22g h22c h22p1 h23a
python3 $I/pdhd/docs/scan/h22/d22_truth2.py h22g h23a      # BOTH truth readings
python3 $I/pdhd/docs/scan/h22/d22_item_trace.py            # why the costs overlap
```

## 1. The ruling

Doc pdhd/22 left two sets measured and held, each because it spends purity. The owner ruled to
**turn on both**. P1 is taken in its recommended variant — `topology_stop_evidence` alone at the
C++ default floors (10 MeV / 3 cm). `topology_clears_sparse` and `topology_michel_ke_min:3` are
**not** taken: the first buys nothing on adjudicated truth while costing 2 more false positives,
the second costs under both readings and additionally flips 5 UNCLEAR and 1 MESSY item.

## 2. Graded as one unit — and the twin held

| arm | on top of `h22g` | `is_stm` | purity | eff | reading B | purity B | eff B |
|---|---|---|---|---|---|---|---|
| `h22g` | doc-22 production | 82/1/65/109 | 0.988 | 0.558 | 60/1/10/58 | 0.984 | 0.857 |
| `h22c` | wide anchor | 83/2/64/108 | 0.976 | 0.565 | 61/1/9/58 | 0.984 | 0.871 |
| `h22p1` | P1 | 94/3/53/107 | 0.969 | 0.639 | 64/1/6/58 | 0.985 | 0.914 |
| **`h23a`** | **both** | **95/3/52/107** | **0.969** | **0.646** | **65/1/5/58** | **0.985** | **0.929** |

> **These are on `smx22`, the record as it stood when the arm ran. §6 supersedes them:** the owner
> later ruled `028084_3/72` a genuine stopper, so on `smx23` `h22c` is 84/1/64/107 and `h23a`/
> `h23conf` is **96/1/52/107, purity 0.990**. The *measurements* are unchanged — the truth moved.

**Pre-registered prediction: 95/3/52/107, purity 0.969, efficiency 0.646, the wide anchor firing
on exactly `028084_21/132` and `028084_3/72`. HELD EXACTLY.**

The prediction that mattered is that **the costs overlap rather than add**. `h22c` costs +1 TP/+1
FP and `h22p1` costs +12 TP/+2 FP, but the combined arm's false positives are **the same two P1
already had** — because `028084_3/72` is tipped by *both* knobs through different paths (doc
pdhd/22 §5.1). The wide anchor's only unique contribution is the true stopper `028084_21/132`.
Naive arithmetic would have predicted 4 false positives; the truth is 3.

This is the campaign's **fourth combined arm and the first whose twin held** — `h21f` beat its
twin, `h21z` and `h22c` failed theirs. It held because the prediction was built from the
per-item trace rather than from arm-level arithmetic.

Branch gate `h22g → h23a`: 314/341 bit-identical, 17 `is_stm` flips, **point geometry identical
341/341, 0 role moves** — both knobs move verdict bits only, no fit moves. The wide anchor fires
on **2 of 341** candidates; P1 clears bits on **25 of 341**.

## 3. The flip and its confirmation

| gate | result |
|---|---|
| compiled-config, **no TLA**, vs the doc-22 production config | **4 added, 0 removed, 0 changed** |
| the flipped **file** vs `h23a`'s measured TLA config | **0 added, 0 removed, 0 changed** — the same config |
| negative control (a file against itself) | 0 / 0 / 0 |
| **confirmation arm `h23conf`** — flipped file, **no TLA**, vs `h23a` | **341/341 bit-identical on all 149 shared branches; 0 `is_stm` flips; point geometry identical 341/341; 0 role moves** |
| `h23conf` census / Michel | identical: **95/3/52/107** and **68/2/18** |
| `h23conf`, owner+high-confidence reading | identical: **65/1/5/58**, purity 0.985, efficiency 0.929 |
| binary pin across all 11 arms of rounds h22–h23 | md5 `3e23bf8a` before and after, every arm |
| records untouched | 7 label files + the `smx22` verdicts (`a36de425`) byte-unchanged |

**Production runs exactly what was measured.**

### The campaign end to end (docs pdhd/21 + 22 + 23)

| | before | after | |
|---|---|---|---|
| `is_stm` | 61/0/86/110 | **95/3/52/107** | purity 1.000 → 0.969, efficiency 0.415 → **0.646** |
| `michel_found` | 56/11/30 | **68/2/18** | purity 0.836 → **0.971**, efficiency 0.651 → **0.791** |
| owner+high-confidence truth | 1.000 / 0.843 | **0.985 / 0.929** | purity essentially held, efficiency +0.086 |

Efficiency **+55 % relative** on the all-truth reading, for three false positives — none of which
was owner-adjudicated at the time.

**On the corrected record `smx23` (§6) the same campaign reads:**

| | before | after | |
|---|---|---|---|
| `is_stm` | 61/0/87/108 | **96/1/52/107** | purity 1.000 → **0.990**, efficiency 0.412 → **0.649** |
| owner+high-confidence truth | 1.000 / 0.662 | **0.985 / 0.930** | efficiency +0.268 |

— i.e. **+35 stoppers for exactly one false positive**, and that one came with the Michel bag,
which is the trade the owner approved in doc pdhd/22.

## 4. How PDHD now compares to PDVD

The short answer: **PDHD is now the purer chain; PDVD is still the more efficient one**, and part
of that efficiency gap is a difference in how the two records were scanned rather than in the two
chains themselves.

### Stopping-muon identification

| | PDVD (`p93vprod`, doc pdvd/93) | PDHD (`h23conf`, on `smx23`) |
|---|---|---|
| all judged | 242/8/44 — purity **0.968**, eff **0.846** | 101/1/57/110 — purity **0.990**, eff **0.639** |
| with a candidate | 242/8/34 — purity 0.968, eff **0.877** | 96/1/52/107 — purity **0.990**, eff **0.649** |
| judged items | 576 (601 scanned, 120 events) | 256 of 303 (317 scanned, 61 events) |
| hand-stopper fraction | 286/576 = **49.7 %** | 148/256 = **57.8 %** |

**PDHD's purity now EXCEEDS PDVD's — 0.990 against 0.968** — on a record where PDHD carries one
false positive and PDVD carries eight. Efficiency is **~0.21–0.24 lower on PDHD** under either
denominator convention, and the conventions must be stated, because PDVD's own doc warns that
"quoting an efficiency without saying which population it is on is how 0.846 and 0.877 get
confused for movement."

### Michel identification

| | PDVD | PDHD |
|---|---|---|
| all judged | 144/12/25 — purity 0.923, eff **0.852** | 68/2/18 — purity **0.971**, eff 0.791 |

**PDHD's Michel purity is materially better** (0.971 vs 0.923) at somewhat lower efficiency.

### Why the comparison is not clean, stated rather than buried

1. **PDVD's base scan was not blind.** Doc pdvd/55 §16.3: scanners read `context.json`, which
   carries `is_stm` and `reject_names`, *before* opening a frame — "every scanner saw the chain's
   answer before looking at the picture … the bias pulls toward agreement, so the purity and
   efficiency are if anything optimistic." PDHD's `smx18`/`smx22` scan was **verdict-blind**. Some
   unknown part of the 0.21 gap is this, not the chains.
2. **Different truth machinery.** PDVD folds the owner's verdict into a flat field at merge time;
   PDHD needs `d18_census.py`'s precedence (`owner_review` > `owner_smx1` > agent). `census_lib.py`
   says so explicitly: reading PDHD's flat field gives the *agent* call and the wrong number.
3. **PDVD's record includes 147 owner re-judges drawn from the hard cases and the decision
   boundary** — a deliberately adversarial slice that PDHD's record does not have in the same form.
4. **Neither number is an absolute efficiency.** Both populations are *the chain's own STM
   candidate pool*, not all stopping muons in the data. A muon never proposed as a candidate is
   invisible to both metrics. PDHD's own `max_candidates` 8→64 result — 16 extra candidates on 7
   of 61 events, carrying +5 hand stoppers — shows the pool itself moves with configuration.
5. **Sample sizes differ by ~2×** (576 vs 257 judged).

**So: PDHD is at PDVD's purity and roughly three-quarters of its efficiency**, with the caveat
that PDVD's efficiency is measured on a non-blind scan and is, by its own documentation,
optimistic. A like-for-like statement would need a blind PDVD re-scan, which does not exist.

## 5. What is NOT concluded

* **Not** that the gap is physics. Points 1–5 above are unquantified; the blindness difference
  alone could account for a meaningful part of it.
* **Not** an absolute efficiency for either detector — both denominators are candidate pools.
* **Not** that PDHD is finished. The chain now carries 25 of PDVD's knobs; the remainder are
  measured dead on PDHD, held with a number, or (for `absorb_bragg_stub`) known to break an event.
* **Not** a re-judged record: `smx22` as it stands.
* ~~**Not** settled that the 3 false positives are false.~~ **SETTLED — see §6.** The owner
  scanned all three on 2026-09-12: one is a genuine stopping muon (the chain was right), one is
  MESSY and leaves the population, one is a real false positive. Production purity is **0.990**,
  not 0.969, and the campaign's total cost is **one** false positive.

## 6. The owner scanned the three false positives — and two were not false

Served on **:5017** under label tag **`own23`** — its own tag, empty at the start, so taps and
pins could not reach a record — with `prep-pdhd-h23conf`, so the chain's answer on screen was
*today's*, not the pre-flip one. Record **`smx23` = `smx22` + these three rulings**, built by
`mkowner_record.py --skip-v5`; `smx22` is never written (M13).

| item | agent (smx22) | owner (own23) | effect |
|---|---|---|---|
| `028084_3/72` | THRU, *medium* | **STM_MICHEL**, Michel *attached*, pin at rr 6.02 cm | **the chain was right** — FP → TP |
| `028084_12/46` | FRAG_THRU, *medium* | **MESSY** — *"messy, does not see to be an STM"* | unscored; leaves the population |
| `029107_10/44` | THRU, **high** | **THRU**, confirmed | a real false positive — the only one |

### Re-scored on `smx23`

| arm | | reading A | purity | eff | reading B | purity | eff |
|---|---|---|---|---|---|---|---|
| `p82bhoff` | pre-campaign | 61/0/87/108 | 1.000 | 0.412 | 47/0/24/59 | 1.000 | 0.662 |
| `h22base` | doc 21 | 80/0/68/108 | 1.000 | 0.541 | 59/0/12/59 | 1.000 | 0.831 |
| `h22g` | doc 22 | 82/1/66/107 | 0.988 | 0.554 | 60/1/11/58 | 0.984 | 0.845 |
| `h22c` | + wide anchor | 84/1/64/107 | 0.988 | 0.568 | 62/1/9/58 | 0.984 | 0.873 |
| `h22p1` | + P1 | 95/1/53/107 | 0.990 | 0.642 | 65/1/6/58 | 0.985 | 0.915 |
| **`h23conf`** | **production** | **96/1/52/107** | **0.990** | **0.649** | **66/1/5/58** | **0.985** | **0.930** |

**Two claims in this doc and in doc pdhd/22 are retracted by this.**

1. **The wide anchor never cost purity.** §5 held it because, combined with the Michel bag, it
   produced a "second false positive" `028084_3/72`. That item is a genuine stopping muon. On
   `smx23` the anchor's marginal effect is **+2 TP / +0 FP** (`h22g` → `h22c`). It was free all
   along, and the owner's decision to turn it on is vindicated by the very item that argued
   against it.
2. **Neither flipped set costs purity.** `h22g` → `h23conf` is **+14 TP / +0 FP**; purity *rises*
   0.988 → 0.990. The campaign's entire purity cost is the **one** false positive
   `029107_10/44`, which came with the Michel bag — the trade the owner approved.

**The ruling is not free scoring for the new chain.** `028084_3/72` is a genuine stopper that
*every* arm before doc 23 missed, so it moves TN → FN for them: the pre-campaign baseline gets
*worse* (efficiency 0.415 → 0.412) and so does doc 21's (0.544 → 0.541). It costs the old chains
a miss and pays the new one a true positive.

**Caveat, stated rather than buried.** The blind is gone (owner decision 2026-09-08): the chain's
verdict was on screen while these three were judged, and `revealed_before_label` is true on all
three. This is a physicist's review of the reconstruction, not an independent verdict, and the
bias runs toward agreement with the chain — two of the three rulings did agree with it.

## 7. APA0 excluded — the owner's cut, and what it does to the PDVD comparison

APA0 has a known hardware problem on PDHD, so the owner asked for the numbers on the three
healthy APAs. **The convention is doc pdhd/04 §6.2's, not invented here:** face 0 = x<0 (APA 0,2),
face 1 = x>0 (APA 1,3), z<231 cm = APA 0,1, z>231 cm = APA 2,3 — so **APA0 = x<0 ∧ z<231 cm**,
matching `smgeom.py`'s `apa0` sensvol (x −352…0 cm, z 0…230 cm).

A candidate is assigned to the APA holding **most of its role-1 muon points** — doc 04's own
rule. That reads the *fit*, not the verdict: bucketing by the chain's `stop_x/y/z` would be
circular for the 52 misses, where no stop was declared. 36 of 256 candidates span more than one
APA, so a **strict** variant (drop any candidate with *any* point in APA0) is reported alongside.

### `is_stm`, production (`h23conf`) on `smx23`

| population | TP/FP/FN/TN | n | purity | efficiency |
|---|---|---|---|---|
| all four APAs | 96/1/52/107 | 256 | 0.990 | 0.649 |
| **APA0 excluded (majority)** | 79/1/30/70 | 180 | **0.988** | **0.725** |
| **APA0 excluded (strict)** | 77/0/28/69 | 174 | **1.000** | **0.733** |
| APA0 only | 17/0/22/37 | 76 | 1.000 | **0.436** |
| APA1 only | 18/0/11/19 | 48 | 1.000 | 0.621 |
| APA2 only | 26/1/9/29 | 65 | 0.963 | 0.743 |
| APA3 only | 35/0/10/22 | 67 | 1.000 | 0.778 |

**APA0 is the weak APA, by a wide margin: 0.436 against 0.621 / 0.743 / 0.778.** Excluding it
lifts production from 0.649 to **0.725** (majority) or **0.733** (strict). Under the strict cut
purity is **1.000** — the single false positive `029107_10/44` has points in APA0.

**The deficit predates this campaign** and the flips did not close it: on `p82bhoff` APA0 was
0.282 against APA2/APA3's 0.486/0.533, and APA0 gained **+0.154** across the campaign where APA3
gained **+0.245**.

**The Michel side is APA-flat**: `michel_found` efficiency 0.739 in APA0 against 0.727 in APA1,
and excluding APA0 moves it only 0.793 → 0.812 (purity 0.972 → 0.963). Whatever APA0 costs, it
costs it in **stopper identification**, not in finding the Michel once a stop is called.

### The comparison to PDVD, redone

| | PDVD (`p93vprod`) | PDHD, all 4 APAs | PDHD, **APA0 excluded** |
|---|---|---|---|
| `is_stm` purity | 0.968 | 0.990 | **0.988** (strict **1.000**) |
| `is_stm` efficiency | **0.877** *(with a candidate)* | 0.649 | **0.725** (strict **0.733**) |
| `michel_found` purity | 0.923 | 0.972 | 0.963 |
| `michel_found` efficiency | **0.878** *(with a candidate)* | 0.793 | 0.812 |

PDVD's *with-a-candidate* column is the like-for-like one, because PDHD's population is by
construction the chain's candidate pool. On that basis:

* **PDHD is the purer chain on both sides** — 0.988 vs 0.968 for stoppers, 0.963 vs 0.923 for
  Michels — and that was already true before the cut.
* **The efficiency gap narrows from 0.228 to 0.144** (0.877 − 0.733), i.e. excluding APA0
  accounts for **about 37 %** of it. The remaining ~0.14 is not APA0.

### What this does NOT establish

1. **The comparison is now asymmetric**: PDHD-healthy against PDVD-*all*. No equivalent bad
   region was excluded on PDVD, and none is named in the PDVD docs. If PDVD has one, its 0.877
   is correspondingly pessimistic.
2. **The mechanism behind APA0's STM deficit is not identified here.** Doc pdhd/04 §7.5 and §8.5
   concluded the induction deficit behind the TGM rejection excess is **detector-wide, not
   APA0** — all four APAs in a 20–26 % band. That was a point-level charge measurement of one
   specific mechanism (APA0's field response). This is a different quantity, the STM verdict
   efficiency, and on it APA0 *is* clearly worse. The two are not the same claim, but the tension
   is real and this table does not resolve it.
3. **PDVD's scan was not blind** (§4), which inflates its numbers by an unknown amount.
4. Neither number is an absolute efficiency — both denominators are candidate pools.

## Files

| what | where |
|---|---|
| pre-registered twin (written before the arm) | `docs/scan/h23/preregistered_twin.txt` |
| the combined arm and its gate | `docs/scan/h23/census_h23a.txt`, `g_h23a.txt` |
| both truth readings, and the per-item trace | `docs/scan/h22/d22_truth2.py`, `d22_item_trace.py` |
| the flip | `pdhd/wct-pr-perevt.jsonnet` `stm_michel_knobs` |
| the owner's scan (§6) | `docs/scan/smx23/owner_rulings_own23.json`, `provenance.json`, labels tag `own23` |
| the corrected record | `docs/scan/pdhd_stm_michel_smx23_verdicts.json` |
| re-scored, and its grader | `docs/scan/h23/census_smx23.txt`, `d23_grade.py` |
| per-APA, and APA0 excluded (§7) | `docs/scan/h23/census_apa.txt`, `d23_apa.py` |
