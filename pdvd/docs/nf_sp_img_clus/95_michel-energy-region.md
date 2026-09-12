# 95 — The Michel's energy from a region around the stop (doc 78 action item 9)

**Status: BUILT, GATED and FLIPPED in PDVD production** (toolkit `d65f8165`; four keys in
`pdvd/wct-pr-perevt.jsonnet`). PDHD stays OFF. The knobs are default-OFF in C++, the OFF path is
byte-identical on both detectors, and the ON path is purely additive — `michel_ke_best`,
`michel_found`, `is_stm` and every reject bit are untouched by construction, so nothing that
already existed can move. What the flip publishes is a **new** energy branch beside the old one,
not a replacement for it: switching the headline would feed P1's 3 MeV floor and the T2c/T3c
vetoes and is a separate decision, with the per-item table and the `is_stm` delta in §4.4.

The owner's intent, unchanged since doc 81 and restated on 2026-09-11: **the Michel's energy must
not depend on the Michel's own trajectory or segmentation.** The fit enters only to predict, and
subtract, the *muon's* charge. Doc 81 built that subtraction correctly but selected the Michel's
cells by association to a Michel *segment*, so charge PR never gave a Michel segment — a dropped
residual, blob points partitioned to the muon's last segment, or a Michel with no segment at all —
was outside the sum. This round replaces the selection with a region around the stop.

**The one-line result:** it works, and by a wide margin on the class that matters — **34.6 MeV on
found Michels against 4.9 MeV where the owner says a muon stopped with no Michel** (7.1:1), where
today's headline `michel_ke_best` reads **0.00** on every one of those no-Michel classes and
23.2 MeV on the found ones. It also recovers energy on items the chain misses entirely. What it
does **not** yet have is a defensible radius: my pre-registered rule for choosing R selected none,
and that is reported as a conflict, not papered over.

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
export STM_SCAN_RECORD=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_smx7_smx8_smx9_verdicts.json

cd /home/xqian/toolkit-dev/toolkit && wcbuild && ./build/clus/wcdoctest-clus   # 396/396
cp -a ../local/lib /home/xqian/tmp/p95/libpin_p95b                            # Clus 3e23bf8a444d

SUF=b WAVE=off PIN=/home/xqian/tmp/p95/libpin_p95b bash $X/d95_arms.sh   # p95voffb + p95hoffb
SUF=b WAVE=on  PIN=/home/xqian/tmp/p95/libpin_p95b bash $X/d95_arms.sh   # p95vq2db

bash $X/d95_gates.sh > /home/xqian/tmp/p95/gates.log 2>&1; echo rc=$?
python3 $X/d95_region.py  --arm p95vq2db --twin      # every table in §4, §7
python3 $X/d95_size_p4.py --arm p95vq2db --own       # §6
```

Pre-registration: `/home/xqian/tmp/p95/pred.txt`, mtime **before any arm ran**. Graded in §7.

## 1. What was built (toolkit `clus/`, all default OFF)

Two knobs beside doc 81's four, in `CheckSTM_Michel`:

- `michel_q2d_region_cm` (**0 = off**) — the 2-D radius about the stop. Every cell inside it
  contributes `measured − muon prediction`, **whatever role, if any, claimed it**. That is the
  whole definition change: the Michel's segmentation no longer selects the cells.
- `michel_q2d_region_ctl_cm` (**−1 = off**) — the same radius on a centre that far back up the
  fit, where no Michel can be. The body control.

Three per-cell columns on `stm_michel_2d`: `d_stop_cm`, `d_ctl_cm`, `own_blob`. The first two make
**every radius a filter over one arm** rather than an arm per radius (docs 91/92's twin rule).
`own_blob` records whose charge a cell is: bit 1 the main cluster's blobs, bit 2 an admitted
unfitted companion's, 0 neither.

Cells no role claims, inside a region, are emitted as **role 0** — the population the
association-based selection drops, which doc 78 item 9 asked to measure. Nothing reads any of it
back: `michel_ke_best`, `michel_found`, `is_stm` and every reject bit are unchanged by
construction, and the branches appear only when the knob is on.

## 2. The frame, and the twin

A cell's 2-D point is in the **raw t0 = 0 drift frame**; a 3-D fit point is in the cluster's
**t0-corrected** frame. On PDVD those differ by *metres* — it is why the chain's own
`michel_ke_charge` reads 0 on 94–96 % of found Michels (doc 81 §5.1). A region needs a distance
from the stop to each cell, so the trap sits squarely in its path.

The implementation carries no offset anyone must maintain: a centre goes through `backward(t0)` +
`convert_3Dpoint_time_ch` and then through **the same `convert_time_wire_2Dpoint` the cells use**,
so both sides of every distance are one identical conversion apart, in one frame, by construction.
(`convert_3Dpoint_time_ch` returns *(tick, local wire)* despite its name — `TrackFitting.cxx:3881`.)

**Twin: the offline reader reproduces the C++ `michel_q2d_region_{u,v,w}` on 596 / 596 candidates,
0 differing.** That is what licenses every offline radius below: the same arithmetic the component
ran, not an approximation.

## 3. Gates (`d95_gates.sh` → `/home/xqian/tmp/p95/gates2.log`)

Pin `libpin_p95b` (Clus `3e23bf8a444d`), 572 libraries md5-identical before and after every arm;
each arm also logged that md5 at its own entry and exit.

| gate | result |
|---|---|
| **1. OFF identity, PDVD** — `p95voffb` vs `p93vprod` | **596 / 596 candidates bit-identical** on every branch and point row; branches 149 v 149, none either side; 0 `is_stm` flips, 0 `michel_found` flips; 0 zips differing |
| **2. OFF identity, PDHD** — `p95hoffb` vs `p85hoff` | **325 / 325 bit-identical**; 130 v 130 branches; 0 flips; 0 zips differing |
| **3. ADDITIVE** — `p95vq2db` vs `p95voffb` | **596 / 596 bit-identical on every PRE-EXISTING branch and point row**; 149 → 198 branches, **exactly the 49 pre-registered new ones, none unexpected, none missing**; 0 `is_stm` flips, 0 `michel_found` flips, 0 zips differing |
| 4. trees + calib | all 8 trees identical on every one of 120 events; calib 119 same, **0 differing** |
| 5. census (546 with a candidate) | `is_stm` 242 / 8 / 34, michel 144 / 12 / 20 — **identical on the OFF and ON arms** |

`michel_q2d_valid == 1` on 596 / 596 with reason codes `{0: 596}` (prediction X2). Role-0 cells —
the population the association-based selection drops — median **1495** per candidate, zero on none.

*On the census denominator:* 546 is the with-candidate population; doc 93's headline 242/8/**44**
(eff 0.846) is the 576-all-judged one, where the 10 the tagger never hands on count as misses
(doc 89 §1). Both are correct, and an earlier version of the gate script quoted the 576 figure
beside the 546 output, which read as a mismatch.

*Pre-existing and symmetric:* both PDVD arms exit `rc=1` with 120/120 trees and zips but **119
calib dumps** — event `039252_11`. Today's production `p93vprod` has the identical 119 and the
identical missing event, so it cancels in the A/B.

**A gate-script defect, found and fixed (kept, not argued away).** The first run printed
`*** PIN CHANGED between launch and gates -- every comparison below is VOID ***` over three gates
that had in fact passed. The `md5sum` line still hard-coded `libpin_p95` (binary A) while the
before-manifest came from `libpin_p95b` (binary B, what the arms ran), so it compared two
different directories. The tell was that exactly **one of 572** hashes differed — the Clus library
— which is two pins of the same tree, not a library swapped mid-run. Fixed, and both logs are
kept: `gates.log` (the defect) and `gates2.log` (corrected). Same species as doc 93's mover
classifier: it failed safe, and it still had to be fixed.

## 3.1 Proofs (`d95_proofs.sh` → `/home/xqian/tmp/p95/proofs.txt`)

| proof | result |
|---|---|
| A: PRE + production's key set vs POST unset | **0 lines** |
| B: POST with every key forced back vs PRE | the four keys present at their C++ initializers (`false`, `false`, `0`, `-1`), matching the grepped `m_michel_q2d*` values |
| C: PRE vs POST | exactly the four keys |
| D: PDHD | unchanged against git HEAD, **0** `michel_q2d` lines |

**Proof A here carries less than doc 93's, and is labelled accordingly.** There, the TLA was the
measured arm's, so 0 lines meant "production runs exactly what the arm ran". Here the measurement
arm ran the wide diagnostic radius (40 cm) while production carries 10 cm, so 0 lines means only
that the file compiles to production's four keys. The other half — that those keys are the reading
this doc measured — is closed by `d95_confirm.py` (§3.2), which predicts the production radius's
scalars from the wide arm's own cell table (exact: 596/596) and checks the confirmation arm
reproduces them item by item.

## 3.2 Confirmation arm (`d95_confirm.py` → `/home/xqian/tmp/p95/confirm.txt`)

`p95vprodb` is the flipped file with **no TLA at all**, on the same pin: 120/120 trees and zips.

**596 / 596 candidates reproduce the twin exactly — "production runs the reading this doc
measured".** The check is not a config diff but arithmetic on both sides: the measurement arm's
cell table, filtered to `d_stop_cm <= 10`, is precisely the cell set a production run at 10 cm
sums (role 1/3/4 cells are emitted regardless of radius; role-0 cells outside R are excluded by
the same predicate the C++ applies), and the offline sum reproduces the C++ on 596/596. So this
closes the half of proof A that the wide measurement radius left open, and it does so more
strongly than a config comparison could.

**The other 16 live jobs are untouched.** `compile_all_cfg.sh` before and after (PRE swapped in
under an `EXIT` trap, with no arm compiling from the file), then `cmp_cfg.sh`: every SBND / PDHD /
PDVD clustering, imaging, NF+SP and simulation job comes back with the same element count, same
order, same edges and **NORMDIFF 0 — OVERALL PASS** (`/home/xqian/tmp/p95/cfg_{before,after}`).
The flipped file was verified back in place afterwards by md5 (`45729e29`, against the pre-flip
`1032d9be`), independently of the trap's own report — leaving PRE in place would have silently
reverted production, which is the one failure of this procedure that looks like success.

## 4. The measurement (`d95_region.py`)

**Read this split by class or not at all.** A median over all candidates is meaningless here:
through-going muons outnumber real stoppers 2:1 and have **no stop**, so their "stop region" is
just more track and reads like the control by construction. Pooled, the control *exceeded* the
signal (29.5 vs 37.2 MeV at 40 cm) and I had written that up as a blocking bias before splitting
by the record's verdict. It was an artifact.

| class (judged, with cells) | n | owner-judged |
|---|---:|---:|
| TP_found — owner says Michel, chain found it | 135 | 39 |
| TARGET — owner says Michel, chain found none | 10 | 5 |
| ZERO_CTL — owner says stopper, **no** Michel | 97 | 6 |
| THRU | 270 | 36 |

### 4.1 Region energy at the stop, MeV — median [p25, p75]

| class | R=5 | R=10 | R=15 | R=20 |
|---|---|---|---|---|
| **TP_found** | 22.7 [17.0, 29.7] | **34.6** [25.6, 44.4] | 40.4 [28.1, 50.8] | 41.7 [29.7, 55.3] |
| TARGET | 6.8 [2.6, 14.5] | 8.6 [6.2, 20.6] | 12.7 [6.1, 24.3] | 14.7 [5.9, 35.4] |
| **ZERO_CTL** | 3.2 [1.4, 8.5] | **4.9** [2.1, 10.9] | 6.7 [3.1, 14.4] | 8.8 [5.0, 18.7] |
| THRU | 2.7 [0.7, 5.9] | 5.2 [2.2, 10.0] | 7.4 [3.3, 13.5] | 9.8 [4.4, 17.4] |

Body control (same estimator, up the fit): TP 3.1, TARGET 3.8, ZERO_CTL 2.9, THRU 6.0 at R=10.

| | R=2.5 | R=5 | R=7.5 | R=10 | R=12.5 | R=15 | R=20 |
|---|---|---|---|---|---|---|---|
| TP excess over own control | 11.7 | 21.5 | 27.7 | 31.5 | 34.7 | 35.1 | 34.3 |
| ZERO_CTL less own control | 1.5 | 2.3 | 2.3 | 2.0 | 2.1 | 1.9 | 1.8 |
| **ratio TP / phantom** | 6.3 | 7.0 | 7.0 | **7.1** | 6.6 | 6.1 | 4.7 |

The phantom's control-subtracted excess is **flat at ~2 MeV at every radius** — that is the honest
floor of the method at a real Bragg peak.

### 4.2 The radius: the pre-registered rule selected none

`pred.txt` asked for the smallest R where **(a)** the TP curve has plateaued (growth < 5 % over the
next 5 cm) **and (b)** the zero-control is still consistent with 0. On the data, (a) first holds at
**R=15**; (b) is satisfied at **no** radius — ZERO_CTL is 4.9 MeV at R=10 and never approaches 0.

`pred.txt` also says what to do about it: *"If (a) and (b) disagree … that is a REPORTED CONFLICT
and the owner picks. I do not split the difference silently."* So **no R is claimed as
pre-registered.** Any R quoted here is **post-hoc**, and the case for R≈10 is that the signal-to-
phantom ratio peaks there (7.1) while the owner's own constraint — the Michel sits near the stop —
argues against the wider radii where the ratio decays to 4.7.

### 4.3 Where the charge comes from, and the scope question

Per-cell at R=10: role 3 (the Michel's own cells) **0.354 MeV/cell**, role 4 0.227, role 0 0.089,
role 1 0.084. Michel cells carry ~4× the rest; the rest is a nearly uniform residual.

Doc 78 item 9 scopes the region to *"the main cluster and the admitted companions"*, but the charge
maps are the union over **every preloaded cluster** — my implementation was role-blind over all of
them, which is a deviation from the specification. `own_blob` measures the cost:

| median MeV | R=5 | R=10 | R=15 | R=20 |
|---|---|---|---|---|
| TP_found, all cells | 22.7 | 34.6 | 40.4 | 41.7 |
| TP_found, own only | 21.5 | 31.3 | 34.2 | 35.1 |
| ZERO_CTL, all cells | 3.2 | 4.9 | 6.7 | 8.8 |
| ZERO_CTL, own only | 2.6 | 4.1 | 5.1 | 6.2 |

The restriction costs ~10 % of signal and removes ~16 % of the phantom (and 44 % of THRU:
5.20 → 2.91), moving the ratio 7.1 → 7.6. **Real but modest**, which is why it is reported and
recommended rather than rebuilt into the C++ this round: `own_blob` ships in the cell table, so any
consumer can take the restricted sum, and making it the headline is a one-key follow-up.

### 4.4 Against the existing estimators (median MeV)

| class | `michel_ke_best` | `michel_ke_q2d` (doc 81) | **region R=10** | region R=10, own |
|---|---:|---:|---:|---:|
| TP_found | 23.23 | 29.98 | **34.65** | 31.25 |
| TARGET | **0.00** | **0.00** | **8.56** | 6.82 |
| ZERO_CTL | 0.00 | 0.00 | 4.91 | 4.06 |
| THRU | 0.00 | 0.00 | 5.20 | 2.91 |

The ordering region > association > fit is the intended effect: each step collects charge the
previous one's trajectory missed. 1.49× `michel_ke_best` on found Michels is consistent with doc 86
§10's finding that the energy is under-counted on about a third of the found population.

### 4.5 The ten TARGET items — where both existing estimators read exactly 0

| item | conf | region R=10 | own | body ctl |
|---|---|---:|---:|---:|
| `039349_51/29` | owner | **46.8** | 46.8 | 5.0 |
| `039253_3/61` | medium | **45.3** | 44.5 | 1.9 |
| `039349_58/69` | medium | 21.8 | 28.6 | 4.7 |
| `039252_9/101` | owner | 16.8 | 16.7 | 2.9 |
| `039349_64/65` | low | 10.4 | 5.5 | **39.9** |
| `039349_43/66` | owner | 6.7 | **0.0** | 1.5 |
| `039349_44/28` | owner | 6.6 | 7.6 | 1.8 |
| `039349_60/40` | medium | 6.1 | 6.1 | 5.2 |
| `039349_72/11` | owner | 2.6 | 3.0 | 1.1 |
| `039349_69/56` | medium | 2.2 | 1.8 | **19.7** |

Four carry substantial, localized energy (16–47 MeV against controls of 1.9–5.0). Two are
**unreliable** — and they are exactly doc 94's known cases: `039349_64/65`, whose own body control
(39.9) dwarfs its stop reading, is doc 94's messy cluster (`ctl_off` 23), and `039349_69/56` is its
one confirmed fit-through. `039349_43/66` (doc 94's `off == 0`) reads **0.0** restricted to its own
blobs. So this **does not overturn doc 94**: it agrees wherever doc 94 had a measurement, and adds
an energy where doc 94 had only point counts.

### 4.6 Reliability: a per-item flag that comes free

A cluster whose **body control** is large is one the muon subtraction fits poorly *everywhere*, so
its stop reading is inflated too. At a control above 10 MeV:

| class | inflated | median all | median clean |
|---|---:|---:|---:|
| TP_found | 18 (13 %) | 34.6 | 34.2 |
| TARGET | 2 (20 %) | 8.6 | 11.7 |
| ZERO_CTL | 17 (18 %) | 4.9 | **4.0** |
| THRU | 95 (35 %) | 5.2 | 4.3 |

Excluding them barely moves the signal and lowers the phantom, taking the ratio to **8.6**. The
gradient (13 % → 35 % from real stoppers to through-goers) is itself a sanity check.
`michel_ke_q2d_ctl` is emitted per candidate, so a consumer applies this without recomputing.

*Correction to my own first version:* I first flagged on `ctl ≥ 0.5 × region`, which fires whenever
the **region is small** and so flagged weak-signal items rather than inflated clusters — it made
ZERO_CTL's "reliable" median (8.1) come out *above* its overall median (4.9), which reads
backwards. The absolute test above is the corrected one.

## 5. The owner's four points

1. **Ideal case stays as-is.** `michel_ke_best` is untouched; the region is additional branches.
2. **Isolated gammas are counted by construction** — the region is role-blind, so a gamma blob near
   the stop contributes whether or not the Michel object claimed it, and whether or not a Michel
   object exists at all. **And the capture-gamma worry was unfounded** (§7, Z2): STM_ONLY items
   *with* a μ⁻ capture gamma read **lower** (3.3) than those without (5.3), so no exclusion is
   needed at this radius.
3. **Energy at the end of the STM minus the STM prediction — delivered.** §4.1 and §4.4.
4. **Adding a Michel from the energy — sized, and it does not clear the bar.** §6.

## 6. Point 4, sized (`d95_size_p4.py`) — NEGATIVE

The rule: `is_stm` 1, `michel_found` 0, Bragg-confirmed (doc 82's gate = the owner's "dQ/dx highly
consistent with expectation"), region energy ≥ E_min. 111 candidates qualify on clause 1
(10 targets, 93 STM_ONLY, 8 THRU).

| E_min | targets | STM_ONLY | THRU | purity |
|---:|---:|---:|---:|---:|
| 4 | 4 | 46 | 5 | 0.07 |
| 10 | 2 | 20 | 1 | 0.09 |
| 15 | 2 | 11 | 0 | 0.15 |
| 25 | 2 | 6 | 0 | 0.25 |

Against a **pre-registered bar of purity ≥ 0.8 with 0 THRU fires**, nothing clears — the best point
is 0.25. The blocker is the phantom's tail, not its median: STM_ONLY items reach 48.3, 35.9, 35.1,
32.9 MeV, several at `high` confidence. **Prediction P1 held** — I registered in advance that this
would fail, and it failed for the reason registered (doc 86 §8.2's neighbouring rule sized at 0.5;
doc 94 closed the population).

## 7. The pre-registration, graded by name

| | prediction | result |
|---|---|---|
| R1 | selected R lands 10–20 cm | rule selected none; post-hoc R≈10 (§4.2) |
| R2 | control rises with R | **HELD** (ZERO_CTL 1.9→8.8; THRU control 0.8→14.9) |
| Z1 | STM_ONLY under 5 MeV at the selected R | **HELD** — 4.9 at R=10 |
| Z2 | capture-gamma items read higher | **MISSED** — 3.3 vs 5.3, the opposite |
| Z3 | body control below the stop region on the same items | **HELD** per class; failed pooled (§4) |
| F1 | region ≥ association on most TP | **HELD** — 105/135 (78 %) |
| F2 | median region / `ke_best` in 1.2–2.5 | **HELD** — 1.49 |
| F3 | no 500 MeV object | **HELD** — max 74.0 MeV (doc 81's was 500.7) |
| F4 | under 15 of 135 TP above 52.8 MeV | **MISSED** — exactly 15 |
| M1 | TARGET reads low; ~0 on the `off == 0` items | **HELD** — median 8.6; `43/66` reads 0.0 |
| P1 | point 4 does not clear | **HELD** (§6) |
| X2 | `michel_q2d_valid` 1 everywhere | **HELD** — 512/512 |
| X3 | closure | **HELD** as restated — 512/512 scalar vs table. *As written it was trivially true* |
| X4 | plane-drop rate rises above doc 81's 57 % | **HELD** — 84 %, and it is now load-bearing |
| X5 | dead cells contribute ~0, not a negative bias | median 0, max 3620 per candidate |

*Two denominators, stated so they cannot be confused.* This table grades on the **512 judged**
candidates that carry a record verdict (135 + 10 + 97 + 270); §3's gate reports **596**, which is
every candidate including the unjudged. So "X2 held, 512/512" and the gate's "`michel_q2d_valid`
== 1 on 596/596" are the same fact on two populations. The same care applies to the census: 546 is
the with-candidate population, 576 the all-judged one (doc 89 §1). This round has already had to
correct one denominator conflation in the gate script, and doc 93 §5 needed the same correction.

## 8. What this does not settle

- **There is no energy truth anchor, and nothing here supplies one.** The record carries verdicts,
  Michel kind, confidence, tags and a stop pin — **no energy field**. So it can say whether a Michel
  exists, never what it weighs. Nothing above shows the region energy is *correct*; it shows it is
  stable, reads ~0 where the owner says there is no Michel, and exceeds the two existing estimators
  in the direction intended.
- **The radius is post-hoc** (§4.2), by the pre-registration's own conflict clause.
- **The plane-drop rule is now load-bearing** — 84 % of candidates drop a plane, against doc 81's
  57 %. A wider region means more cells and more dead exposure. It was predicted (X4) and is not
  re-tuned here, but it is the obvious next systematic.
- **A ~2 MeV phantom is irreducible at any radius** at a real Bragg peak, and its tail reaches
  48 MeV on 18 % of STM_ONLY items. That is what kills point 4.
- **The scope deviation is documented, not fixed in C++** (§4.3).
- **PDHD is not flipped and cannot be judged here.** Doc 81 §8a measured 45 % of its Michels leaning
  on the cross-shared fitted substitution (bridged 0.82, charge-only 0.55) against 22 % / 1.000 on
  PDVD, and there is no owner hand-scan of this chain on PDHD. Its exposure is the OFF gate alone.

## 9. Next, ranked

1. **The scope follow-up, and it is one key.** §4.3 shows restricting the region to
   `own_blob > 0` — doc 78 item 9's own wording, "the main cluster and the admitted companions" —
   costs ~10 % of signal, removes ~16 % of the phantom and 44 % of the through-going
   contamination, moving the ratio 7.1 → 7.6. It was not rebuilt into the C++ this round because
   the gain is modest and the cell table already carries `own_blob`, so the restricted sum is
   computable today. Making it the headline is a knob plus one arm.

   > **CORRECTED by doc pdvd/96 — this item is wrong twice over, and the numbers did not
   > reproduce.** `own_blob > 0` is **not** doc 78 item 9's scope: bit 2 covers only the
   > *segment-less* companions and `n_dot_clusters_unfit` is **0 on all 596** candidates, so it
   > never fires, while a companion that *did* produce segments set **no bit at all**. The column
   > was main-cluster-only. Nor is it "one key" — no scope knob existed; doc 96 had to add bit 4
   > (a preloaded *fitted* companion, which fires on 2.5 % of cells) and sweep every
   > `(face, wire)` rather than the first. With the specification-faithful filter the measured
   > result is **ratio 7.06 → 7.22, not 7.1 → 7.6**, and a 20 000-sample bootstrap puts the
   > difference at +0.299 with 95 % CI **[−0.612, +2.156]** — *not significant*. The flip that
   > shipped rests on specification fidelity, a 27.1 % cut in through-going contamination and a
   > negative-prediction clamp — **not** on this ratio.
2. **The ~2 MeV pedestal is the accuracy limit, and its mechanism is unexamined.** Cells the muon
   fit models carry a uniform **+0.084 MeV/cell** residual (role 1) — the fit slightly
   under-predicts everywhere, and a region sums that over hundreds of cells. This sets the floor
   under every number in §4 and is the one thing that would improve the energy itself rather than
   its selection. It is a `TrackFitting` question, not a `CheckSTM_Michel` one.

   > **CORRECTED by doc pdvd/96 — neither "uniform" nor "unexamined" survives measurement.**
   > 0.084 MeV/cell is a *mean* over a strongly prediction-dependent quantity. Binned by
   > `pred_mu`, the **median** residual/prediction runs **−0.113, −0.080, −0.022, +0.032, +0.063,
   > +0.081, +0.092, +0.105** — the fit slightly *over*-predicts in the middle and under-predicts
   > ~10 % at the top, where the Bragg peak is, and that shape holds inside every record class.
   > And the mechanism was already on the record: doc 42 measured the signed bias per plane
   > (U −0.221, V −0.217, W −0.101) and **refuted the clipped-window explanation empirically**,
   > while doc 44 §7 named charge-dependent whitening as the candidate. The `TrackFitting`
   > attribution stands; "unexamined" does not.
3. **The plane rule is now load-bearing and was not designed to be.** 84 % of candidates drop a
   plane (doc 81: 57 %). More cells and more dead exposure push the two largest planes apart more
   often, so a rule written for the association's ~80 cells per plane is now arbitrating a
   region's many hundreds. Predicted (X4) and not re-tuned here.
4. **Doc 88 §9.5 item 2 — grade the PDVD flips of docs 83–95 on PDHD's `smx18` record.** Still the
   standing cross-detector item, and this round adds to the pile: PDHD carries none of it, and
   doc 81 §8a's cross-shared substitution is the known reason it cannot simply be turned on there.
5. **No truth anchor still.** Everything here is estimator-against-estimator plus the 52.8 MeV
   endpoint. Simulation is the obvious route and the obvious trap — a generator and an estimator
   sharing a recombination model will agree with themselves.

**Recommendation: 1, then 2.** Item 1 is nearly free and improves the number that just shipped;
item 2 is the only lead that would make the energy more *accurate* rather than better *selected*.
Point 4 (§6) is closed by measurement at purity 0.25 against a bar of 0.8 and should not be
retried without a different discriminator — the blocker is the phantom's tail, and item 2 is what
would shrink it.
