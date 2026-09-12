# doc pdhd/23 — the owner turns on both held sets, and how PDHD now compares to PDVD

**Status:** the **wide Bragg anchor and P1 are flipped** in PDHD production on the owner's ruling
of 2026-09-12, graded as **one unit** (arm `h23a`). `is_stm` 82/1/65/109 → **95/3/52/107**,
purity 0.988 → 0.969, efficiency 0.558 → **0.646**; on owner+high-confidence truth
60/1/10/58 → **65/1/5/58**, purity 0.984 → **0.985**, efficiency 0.857 → **0.929**.
`michel_found` unchanged at 68/2/18. The confirmation arm `h23conf` (the flipped file, no TLA) is
**341/341 bit-identical** to `h23a`, so production runs what was measured. §4 compares the
finished PDHD chain to PDVD's.

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
is owner-adjudicated, and none of which counts against the chain on adjudicated truth.

## 4. How PDHD now compares to PDVD

The short answer: **purity has converged; efficiency has not**, and part of the remaining gap is
a difference in how the two records were scanned rather than in the two chains.

### Stopping-muon identification

| | PDVD (`p93vprod`, doc pdvd/93) | PDHD (`h23a`, this doc) |
|---|---|---|
| all judged | 242/8/44 — purity **0.968**, eff **0.846** | 100/3/57/110 — purity **0.971**, eff **0.637** |
| with a candidate | 242/8/34 — purity 0.968, eff **0.877** | 95/3/52/107 — purity 0.969, eff **0.646** |
| judged items | 576 (601 scanned, 120 events) | 257 of 303 (317 scanned, 61 events) |
| hand-stopper fraction | 286/576 = **49.7 %** | 147/257 = **57.2 %** |

**Purity is matched: 0.969–0.971 against 0.968.** Efficiency is **~0.21 lower on PDHD** under
either denominator convention — and the conventions must be stated, because PDVD's own doc warns
that "quoting an efficiency without saying which population it is on is how 0.846 and 0.877 get
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
* **Not** settled that the 3 false positives are false — all three are agent-only items, none
  owner-adjudicated, and on owner+high-confidence truth the flip costs **zero** new FP. A hand
  look at `028084_3/72`, `028084_12/46` and `029107_10/44` would settle it: three items.

## Files

| what | where |
|---|---|
| pre-registered twin (written before the arm) | `docs/scan/h23/preregistered_twin.txt` |
| the combined arm and its gate | `docs/scan/h23/census_h23a.txt`, `g_h23a.txt` |
| both truth readings, and the per-item trace | `docs/scan/h22/d22_truth2.py`, `d22_item_trace.py` |
| the flip | `pdhd/wct-pr-perevt.jsonnet` `stm_michel_knobs` |
