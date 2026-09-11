# doc pdvd/92 — the wide Bragg-peak read, built OFF, and the re-judge that changed its headline (doc 91 §10 item 1)

**Status: built behind three default-OFF knobs, gated, NOT flipped.** No jsonnet file changed; PDVD
and PDHD production are untouched.

**The one-line result:** the rule works exactly as the offline twin predicted, but doc 91's headline —
*+5 stoppers at 0 judged THRU* — **did not survive the blind re-judge**. The owner reversed their own
smx7 call on `039349_71/37` (STM_MICHEL → THRU), which is one of the five gains. On the corrected
record the selected point gains **4 stoppers and 1 false positive**: efficiency 0.832 → 0.846,
purity 0.971 → 0.968. It is now a real trade, not a free gain — and a third blind look (§5.1, `smx9`) confirmed that item
THRU, 2 of 3, so the cost is settled rather than suspected. §6 has the numbers, §9 the pick.

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; X=$IMG/pdvd/docs/nf_sp_img_clus/scripts; S=$IMG/pdvd/docs/scan
export STM_SCAN_RECORD=$S/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_smx7_verdicts.json
# 1. the twin, BEFORE any C++ existed; then the pre-registration, before the build and the arms
python3 $X/d92_twin.py --prep /home/xqian/tmp/p90/prep_p90vprod --cfg /home/xqian/tmp/p90/proofs/post.json \
    --root-arm p90vprod --json /home/xqian/tmp/p92/twin.json > /home/xqian/tmp/p92/twin.txt 2>&1; echo rc=$?
# 2. build, freshness proof, unit tests   (run from the toolkit dir: wcbuild calls ./wcb)
cd /nfs/data/1/xqian/toolkit-dev/toolkit && wcbuild > /home/xqian/tmp/p92/build.log 2>&1; echo rc=$?
./build/clus/wcdoctest-clus > /home/xqian/tmp/p92/doctest.log 2>&1; echo rc=$?
# 3. pin local/lib (a peer's wcbuild would otherwise swap the binary under the live arms)
cp -L /nfs/data/1/xqian/toolkit-dev/local/lib/*.so* /home/xqian/tmp/p92/libpin_p92/
# 4. proofs, 5. arms, 6. gates
bash $X/d92_proofs.sh      > /home/xqian/tmp/p92/proofs.txt     2>&1; echo rc=$?
WAVE=1 bash $X/d92_arms.sh > /home/xqian/tmp/p92/arms_wave1.log 2>&1; echo rc=$?
bash $X/d92_gates.sh       > /home/xqian/tmp/p92/gates.log      2>&1; echo rc=$?
# 7. the smx8 blind tranche, its display, and the scoring + fold into a NEW record
cd $IMG/pdhd/stm_michel_scan && python3 $X/d92_build_smx8.py --twin /home/xqian/tmp/p92/twin.json \
    --prep /home/xqian/tmp/p90/prep_p90vprod --outprep $PWD/prep-pdvd-smx8 \
    --sheet $S/pdvd_stm_michel_smx8_sheet.tsv --questions $S/pdvd_stm_michel_smx8_questions.json \
    --key $S/pdvd_stm_michel_smx8_key.tsv
./serve_stm_michel_scan.sh 5018 --det pdvd --scan-tag smx8 --manifest $S/pdvd_stm_michel_smx8_sheet.tsv \
    --prepdir $PWD/prep-pdvd-smx8 --questions $S/pdvd_stm_michel_smx8_questions.json
# the viewer writes pdvd/work/stm_michel_labels/smx8/labels.json (a sibling of the per-event dirs)
python3 $X/d92_score_smx8.py --labels $S/pdvd_stm_michel_smx8_labels.json --key $S/pdvd_stm_michel_smx8_key.tsv \
    --prep /home/xqian/tmp/p90/prep_p90vprod --twin /home/xqian/tmp/p92/twin.json \
    --write-merged $S/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_smx7_smx8_verdicts.json
# 8. the headroom, on BOTH records
python3 $X/d92_headroom.py --prep /home/xqian/tmp/p90/prep_p90vprod --twin /home/xqian/tmp/p92/twin.json --arm p92v13
```

Outputs in `/home/xqian/tmp/p92/`.

## 1. What was built

Three knobs, all default OFF, in one new block placed **after doc 75's geometric fallback and before
P1** (`stm_michel_topology_clear`):

| knob | C++ default | meaning |
|---|---|---|
| `bragg_wide_anchor_cm` | `0.0` (off) | W — the wider peak search, cm back from the fit end |
| `bragg_wide_anchor_rise_min` | `1.5` | R — the wide peak's 5-point mean, × the wide plateau median (0 = no guard) |
| `bragg_wide_anchor_tail_max` | `0.8` | D — the median of the rows past the peak, × the same plateau (0 = no guard) |

It runs only where the reading that stands still carries a shape bit (`no_bragg | shape_flat |
plateau_off_mip | profile_sparse`), re-anchors with T7's own recipe (`end_L = L[max 5-point mean] +
0.2 cm`) on the **same live geometric rows** within W, reads the same four tests there, and clears
all four only if that reading sets none of them **and** R **and** D hold. It never sets a bit, so it
cannot lose a stopper, and it makes no Michel object. `bragg_wide_fired` and `bragg_wide_shift_cm`
are written only when the knob is on.

**One deliberate divergence from doc 75's fallback.** The fallback *replaces* `rec.bragg` / `ks_*` /
`ratio_*`. This block does **not**. `rec.bragg` is read again downstream at `:2950` — `bragg_here`,
which under `michel_guards_stop` demotes a continuation arm to `kOther` — and by the T8 charge
support. Replacing it would let the wide read move `R_CONTINUATION`, a coupling no offline twin
models. Left alone, what the block publishes is exactly *"the same bits with the four shape bits
cleared, then P1"* — which is what §2's twin predicts candidate by candidate — and every shape field
stays identical on every candidate, movers included.

**A convention checked, not assumed.** The D guard takes a median of the rows past the peak. The twin
uses `np.median`; the C++ uses `stm_michel_median`, which averages the two middle values for an even
count (`StmMichelFunctions.cxx:208–215`) — numpy's convention. They agree. It mattered:
`039349_44/28` has 10 rows past the peak and reads d 0.77 against a D of 0.8.

## 2. The twin, written before the C++ existed

`d92_twin.py` reproduces production on **585 / 585** candidates — `is_stm`, `reject_bits`,
`topology_cleared_bits`, the anchor shift, the fallback flag — reading the unrounded fit rows from
`T_stm_michel_pts` (doc 91's payload-rounding lesson). The arms are therefore a confirmation, not a
measurement.

The order is checkable by mtime, and it matters: `twin.json` **15:36:28**, the pre-registration
`pred.txt` **15:37:33**, the first edit to `CheckSTM_Michel.cxx` **15:39:38**, the build **15:42:59**,
the arms **15:45:36**.

It predicted, at the two pre-registered points (both W 8, D 0.8):

| arm | R | fires on | `is_stm` 0→1 | on the smx7 record |
|---|---:|---:|---|---|
| `p92v13` | 1.3 | 7 | 5 | 243 / 7 / 42 |
| `p92v15` | 1.5 | 5 | 3 | 241 / 7 / 44 |

**The case that would have looked like a gate failure.** The twin also predicts the block fires on
two candidates whose `is_stm` does **not** move — `039253_8/65` and `039349_48/54`. P1 was going to
clear those bits anyway, so the verdict is unchanged, but `topology_cleared_bits` goes to 0 because
the wide read cleared them first. Doc 91's sizing never modelled these, because it only looked at
items production rejects.

## 3. Compiled-config proofs (`proofs.txt`)

- **A** — bare production compiles with **no** `bragg_wide_*` key: the OFF path is production's own
  compiled config, and the C++ default is 0 = off.
- **B** — each arm's TLA puts all three keys in the compiled `CheckSTM_Michel` config at the right
  values. Without this an ON arm could be running the knob off and "passing" vacuously.
- **C** — `pdvd/wct-pr-perevt.jsonnet` is byte-identical to git HEAD. No jsonnet changed.
- **D** — PDHD's file has no `bragg_wide` line and is unchanged. PDHD stays OFF.

`wcbuild` rc 0; `local/lib/libWireCellClus.so` 15:42:47, newer than the 15:39:38 source edit
(freshness proof, M1); `wcdoctest-clus` 396/396 cases, 23885 assertions.

## 4. Gates (`gates.log`)

| gate | result |
|---|---|
| **OFF, PDVD** — `p92voff` vs `p90vprod` | **identical**: 120 / 120 zips identical member-for-member, all 8 trees identical on every event, 596 / 596 candidates bit-identical on all 147 shared branches, no shared branch moved, 0 `is_stm` flips, point geometry identical on 596 / 596 |
| **OFF, PDHD** — `p92hoff` vs `p88hoff` | **identical**: 61 / 61 zips, all 8 trees, 325 / 325 candidates bit-identical on all 130 shared branches, 0 flips |
| **ON** `p92v13` | **PASS** — fired on 7 (twin 7), `is_stm` movers 5 (twin 5), bits-only 2 (twin 2); every firing item matches the twin exactly on `reject_bits`, `topology_cleared_bits`, `is_stm` and `bragg_wide_shift_cm`; census 243 / 7 / 42 = the twin's prediction |
| **ON** `p92v15` | **PASS** — fired on 5 (twin 5), movers 3 (twin 3), bits-only 2 (twin 2); census 241 / 7 / 44 = the twin's prediction |
| frozen branches | `michel_*`, `ks_*`, `ratio_*`, `tail_med`, `plateau_med`, `comp_fwd*`, `comp_bwd*`, `bragg_anchor_*`, `n_live_pts`, `dead_frac_cmp` unmoved on all 596 candidates in **both** arms |
| pin | `libpin_p92`, 572 libraries, md5 identical before and after every arm |

The OFF gate is the bar this round had to clear, and it clears it exactly: with the knob absent the
new binary reproduces production byte for byte on both detectors.

**The census in this table is on the smx7 record**, deliberately: the twin's prediction was computed
on that record, so this is a like-for-like check that the arm does what was predicted. §6 gives the
same arms on the corrected smx8 record, where the numbers are different (242 / 8 / 44) because the
*record* moved, not because the code did.

In `p92v13` the only branches that move at all are `reject_bits` (5), `is_stm` (5),
`topology_cleared_bits` (2) and the capture-gamma family (2 candidates). The last was pre-registered
in `pred.txt`: `stop_gamma_require_stm` is ON in PDVD production (doc 85), so a candidate whose
verdict stops rejecting may now publish a capture gamma. It is a consequence of the verdict moving,
not a second effect of the rule.

**One defect, in the gate script rather than the code.** The first gate run labelled `p92v13` FAIL
because that gamma family was missing from the script's allow-list, although `pred.txt` had
explicitly predicted it. The script was corrected to allow the pre-registered family (and to print
it), and re-run; the log quoted here is the re-run. `p92v15` passed on both runs. `michel_found`
moves on no item in either arm, as §1 requires.

## 5. The `smx8` blind re-judge (:5018) — 19 items, all labelled

The tranche was the **11-item decision set** — every candidate any of the four candidate operating
points (W/R/D = 8/1.3/0.8, 8/1.5/0.8, 8/1.3/1.0, 10/1.3/0.8) would turn into a stopper — plus **8
controls** no grid point moves. Every item read the same in production (`is_stm` 0, `michel_found` 0,
shape bits only); order shuffled; the group and record verdict only in the key file.
`039253_12/93` was dropped by the builder for reading `michel_found` 1, which would have marked it out.

**The controls are the null, and on the stopper call they held 8 of 8** — so the stopper-vs-THRU
re-judge is stable, and what moved on that axis moved at the boundary. **The null is narrower than
it looks, though:** one control, `039349_29/40`, went STM_MICHEL → STM_ONLY. That keeps it a stopper,
so it does not dent the 8/8, but it is a Michel-axis move on a control — and that item is one of doc
90's five pinned fit-through Michels. Read the 8/8 as a null for the stopper call only, not for the
Michel call.

**The decision set: 8 of 11 held, 3 overturned.**

| item | record | smx8 | effect |
|---|---|---|---|
| `039349_71/37` | **STM_MICHEL, owner (smx7)** | **THRU** (pin 6.5 → 1.8 cm) | a gain becomes a false positive at **both** R points |
| `039349_50/46` | THRU, medium | STM_MICHEL (pin 7.2) | only reached at D 1.0 |
| `039349_51/44` | THRU, medium | STM_MICHEL (pin 5.4) | only reached at D 1.0 |

- **Q1 answered: `039349_81/25` is THRU**, confirming the owner's smx7 call. R 1.3 excludes it by
  only 0.02 — but it excludes the right item.
- **Q2 answered, and my pre-registration missed it.** I predicted at least 3 of the 4
  medium-confidence d 0.93–0.97 items would be confirmed THRU. Only **2** were (`039349_17/62`,
  `039349_24/58`); the other two are stoppers.
- **The item that matters is `039349_71/37`.** The owner judged it STM_MICHEL in smx7 at 14:28 and
  THRU in smx8 at 15:50, moving the pin from 6.5 cm to 1.8 cm. Doc 91's "0 judged THRU" headline
  rested on that verdict. It reads p 1.81 — above both R 1.3 and R 1.5 — so **R 1.5 does not protect
  against it**.

### 5.1 `smx9`, the blind tie-break — the item is THRU, 2 of 3

Two blind owner verdicts disagreeing is not a result, so the item was served a **third** time, blind,
with 6 fresh controls drawn from items not used in smx8 (`d92_build_smx9.py`, tag `smx9`). The two
prior calls were deliberately **not** shown: an anchored third call settles nothing.

| | smx7 | smx8 | **smx9** | majority |
|---|---|---|---|---|
| `039349_71/37` | STM_MICHEL (pin 6.5) | THRU (pin 1.8) | **THRU** (pin 3.0) | **THRU, 2 of 3** |

The controls held **6 of 6**, and the two most recent independent readings agree. So the false
positive is real and settled: the rule's one cost at either operating point is this track, and it is
through-going. The census in §6 is unchanged — the smx8 record already carried it as THRU, so the
fold of smx9 (`d92_fold_smx9.py`) moves no stopper / Michel / judged class and is provenance only.

What this does **not** say is that the item is easy: one of three blind owner readings called it a
Michel, and the pin has landed at 6.5, 1.8 and 3.0 cm on three viewings. It sits inside the record's
own resolution. That is the honest characterisation of the rule's single false positive.

## 6. The corrected census (`score_smx8.txt`)

The fold changed the stopper/THRU class on 4 of 601 records; judged stoppers 285 → 286.

| | smx7 record | **smx8 record (corrected)** |
|---|---|---|
| production | 238 / 7 / 47 — eff 0.835, pur 0.971 | 238 / 7 / 48 — eff 0.832, pur 0.971 |
| `p92v13` (R 1.3) | 243 / 7 / 42 — eff 0.853, pur 0.972 | **242 / 8 / 44 — eff 0.846, pur 0.968** |
| `p92v15` (R 1.5) | 241 / 7 / 44 — eff 0.846, pur 0.972 | 240 / 8 / 46 — eff 0.839, pur 0.968 |

On the corrected record, item by item:
- **R 1.3: +4 stoppers** (`039252_9/101`, `039253_12/93`, `039349_44/28`, `039349_51/29`) **+1 THRU**
  (`039349_71/37`).
- **R 1.5: +2 stoppers** (`039252_9/101`, `039349_44/28`) **+1 THRU** (the same one).

**R 1.3 dominates R 1.5:** same single false positive, twice the gain. The R guard is no longer what
the choice turns on.

**What must not be done with this.** The tranche was drawn *from the boundary* — by the very quantity
being measured — so re-selecting W/R/D on the corrected record would be circular. In particular
D 1.0 now looks better only because two of its four cost items were re-judged into stoppers; that is
exactly the circularity, not evidence. The operating point stays doc 91's pre-registered
W 8 / R 1.3 / D 0.8. Settling D 1.0 needs a fresh tranche that was not used to choose it.

## 7. Where things stand: is there still room in PDVD STM + Michel?

Measured, crediting the rule at W 8 / R 1.3 / D 0.8, on the corrected record (`headroom_smx8.txt`;
the smx7-record version is in `headroom.txt` and differs only in the totals).

**The 44 remaining missed stoppers, by who can still act:**

| bucket | n | what it is |
|---|---:|---|
| (a) the owner's own preference | 8 | fiducial / continuation / hadron guard (doc 70 §3.3, doc 77 §7.5) |
| (b) the tagger's | 10 | no candidate ever reaches `CheckSTM_Michel` (doc 89 §3) |
| **(c) chain-side, with a named lever** | **5** | a Michel object exists and P1 rejects it |
| (d) chain-side, no lever | 21 | shape-only, no Michel object; charge says nothing that separates them from a through-going track |

- **The headroom is (c): 5 items, 1.7% of the 286 judged stoppers.** Recovering every one at zero
  cost moves efficiency 0.846 → 0.864.
- **All five miss P1 on the same condition — `michel_len` below the 3 cm floor** (0.9, 1.3, 2.0, 2.3,
  2.5 cm). The entire remaining stopper-side lever is one threshold,
  `topology_michel_len_min_cm` — and doc 89 §5.1 recorded that this floor *is* what protects purity
  now that the energy floor is 3 MeV. Lowering it is precisely the trade the owner was protected from.
- **41 of the 44 are owner-judged**, so this is no longer a re-judging question.

**The Michel side holds the remaining lever.** Of 25 missed Michels: 5 are the tagger's, 10 are
missed because the stopper call is missed first, and **10 are candidates where the chain calls the
stop and finds no Michel** — doc 90 §8 item 2's off-fit-charge population. Three of those 10
(`039252_9/101`, `039349_44/28`, `039349_51/29`) are this round's own gains: the rule makes the stop
call, and the Michel is the very next step.

**The honest read: the stopper side is measured out.** One knob, five items, at most +1.8 points of
efficiency, and that knob trades directly against purity. The Michel side still has a population of
10 with a named cause. And this round is itself evidence of the deeper limit — the ground truth moved
under a re-judge of items the owner had already judged, by enough to turn a "+5 at 0 FP" into
"+4 at 1 FP". At this level the record's own resolution, not the algorithm, is what bounds the answer.

## 8. What this does not settle

- **Out-of-record purity.** W, R and D were chosen on the smx7 record. No build can prove purity away
  from it.
- **How reproducible `039349_71/37` is.** §5.1 settles its verdict (THRU, 2 of 3) but not its
  difficulty: three blind owner readings put the pin at 6.5, 1.8 and 3.0 cm and split 1–2 on the
  call. The rule's single false positive is a track the record itself resolves only barely.
- **The Michel for the gains.** They read `is_stm` 1 / `michel_found` 0 — doc 78 item 8's population.
- **PDHD is not graded.** The rule is OFF there; doc 88 §9.5 item 2 is still open.

## 9. Next, ranked

1. **The owner's call on whether to flip at all.** R 1.3 dominates R 1.5, and after §5.1 the trade is
   fully measured and no longer contingent: **+4 owner-judged stoppers against 1 confirmed false
   positive** (eff 0.832 → 0.846, purity 0.971 → 0.968). Both arms are built, gated and predicted
   exactly; flipping is a one-line jsonnet change plus the usual proofs.
2. **Doc 90 §8 item 2 / doc 78 item 8 — the 10 stop-called-but-no-Michel candidates.** §7 makes this
   the one population left with a named cause and a real size.
3. **A fresh, non-boundary tranche** if D 1.0 is ever to be settled honestly.
4. **Doc 88 §9.5 item 2** — grade the PDVD flips on PDHD's smx18 record.

**Recommendation: decide 1, do 2, then pause the stopper side for good.** §7 measures the stopper-side
ceiling at five items behind a purity-trading threshold; item 2 is the only place where another round
of this kind still has something to work on.
