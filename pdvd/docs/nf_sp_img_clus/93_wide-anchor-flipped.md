# doc pdvd/93 — the wide Bragg-peak read, FLIPPED in PDVD production (doc 92 §9 item 1)

**Status (2026-09-11): FLIPPED in PDVD production.** The keys are `bragg_wide_anchor_cm: 8.0`,
`bragg_wide_anchor_rise_min: 1.3` and `bragg_wide_anchor_tail_max: 0.8` in
`pdvd/wct-pr-perevt.jsonnet`. PDHD stays OFF. **No C++ changed this round** — the code shipped in doc
92 (`ec6990e0`), default OFF; this round only routes the knobs through the production config.

**The one-line result:** production now buys **+4 owner-judged stoppers for 1 confirmed false
positive** — `is_stm` 238/7/48 → 242/8/44, efficiency 0.832 → 0.846, purity 0.971 → 0.968. Unlike doc
92, nothing here is contingent: the false positive was settled by a blind third look (doc 92 §5.1),
and the confirmation arm proves the flipped file runs exactly what measured that trade.

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
S=$IMG/pdvd/docs/scan
export STM_SCAN_RECORD=$S/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_smx7_smx8_smx9_verdicts.json
# 0. snapshot the file BEFORE the edit -- every proof below is against this copy
cp -L $IMG/pdvd/wct-pr-perevt.jsonnet /home/xqian/tmp/p93/pre_evt.jsonnet
# 1. the compiled-config proofs (A-D)
PRE_EVT=/home/xqian/tmp/p93/pre_evt.jsonnet bash $X/d93_proofs.sh \
    > /home/xqian/tmp/p93/proofs.txt 2>&1; echo rc=$?
# 2. the other 16 live jobs, before and after (the file is swapped back by an EXIT trap)
cd $IMG/abtest && ./compile_all_cfg.sh /home/xqian/tmp/p93/cfg_before   # with PRE in place
                  ./compile_all_cfg.sh /home/xqian/tmp/p93/cfg_after    # with POST in place
                  ./cmp_cfg.sh /home/xqian/tmp/p93/cfg_{before,after}
# 3. the confirmation arm: the FLIPPED file, no TLA, on the pin p92v13 ran on
JOBS=8 bash $X/d93_arms.sh > /home/xqian/tmp/p93/arms.log 2>&1
# 4. the gates: identity vs p92v13, effect vs p90vprod, census vs the smx9 record
bash $X/d93_gates.sh > /home/xqian/tmp/p93/gates.log 2>&1; echo rc=$?
```

Outputs in `/home/xqian/tmp/p93/`.

## 1. What was flipped, and why it was the owner's call

Doc 92 built the rule behind three default-OFF knobs and gated it, but deliberately did **not** flip
it: the measurement had turned from doc 91's "+5 stoppers at 0 judged THRU" into a genuine trade, and
trading purity for efficiency is the owner's decision, not the agent's. The owner reviewed the
corrected trade and chose to flip at R 1.3.

**What the rule does.** Doc 68's Bragg anchor searches for the dQ/dx peak within 3 cm of the fit end.
That is right for a peak at the end, but blind to a stop whose rise begins further back — the
anchored profile then reads flat and the candidate is rejected on shape. Where one of the four shape
bits is still set, this rule searches the peak again within a **wider** window `W` and clears those
bits only if the wider reading both **rises** (peak ≥ `R` × the wide plateau) and has a **quiet tail**
past the peak (median of the rows beyond it ≤ `D` × the wide plateau, over ≥ 3 such rows).

**It clears bits only.** `rec.bragg`, `ks_*` and `ratio_*` are deliberately left untouched — unlike
doc 75's geometric fallback, which replaces them. The reason is `bragg_here`
(`CheckSTM_Michel.cxx:2950`): it reads `rec.bragg` to demote a `kContinuation` stop arm to `kOther`
under `michel_guards_stop`, so writing `rec.bragg` here would silently couple this rule to the
continuation logic. Keeping it to bits is also why every shape field stays byte-identical on all 596
candidates in doc 92's OFF gate.

**The operating point is doc 91's pre-registered W 8 / R 1.3 / D 0.8.** It was chosen *before* the
arms ran and is not re-selected here. Doc 92 §6's circularity warning stands and is repeated because
it is easy to lose: the smx8 tranche was drawn **from the decision boundary**, by the very quantity
being measured, so re-choosing W/R/D on the corrected record would be circular. In particular D 1.0
looks better on that record only because two of its four cost items were re-judged into stoppers —
that is the circularity, not evidence for it. Settling D 1.0 honestly needs a fresh tranche that was
not used to choose it.

**R 1.3 dominates R 1.5** — the same single false positive, twice the gain — so the choice of R is
not what the flip turns on; only whether to flip at all.

## 2. The proofs (`scripts/d93_proofs.sh`, `/home/xqian/tmp/p93/proofs.txt`)

All three keys are **new**: none appeared in the compiled config before this round.

| key | before | after | note |
|---|---|---|---|
| `bragg_wide_anchor_cm` | absent (C++ `0.0` = off) | `8.0` | the master switch |
| `bragg_wide_anchor_rise_min` | absent (C++ `1.5`) | `1.3` | a real change from the default |
| `bragg_wide_anchor_tail_max` | absent (C++ `0.8`) | `0.8` | **inert**: equals the C++ default |

| proof | result |
|---|---|
| **A: PRE + the arm's TLA vs POST unset** | **0 lines** |
| B: POST with all three forced back (`cm:0.0, rise_min:1.5, tail_max:0.8`) vs PRE | exactly three lines, each key present at its C++ initializer vs absent. The initializers are grepped and printed by the script — `m_bragg_wide_anchor_cm{0.0}`, `m_bragg_wide_anchor_rise_min{1.5}`, `m_bragg_wide_anchor_tail_max{0.8}` — so the component reads the same numbers either way (the docs 58 / 61 / 82 inert-key form). |
| C: PRE vs POST | exactly the three keys, nothing else |
| D: PDHD | `pdhd/wct-pr-perevt.jsonnet` unchanged against git HEAD; **0** `bragg_wide_anchor` lines in it |
| the other 16 live jobs (SBND / PDHD / PDVD clustering, imaging, NF+SP, simulation) | `compile_all_cfg.sh` before and after, then `cmp_cfg.sh`: every job NORMDIFF 0, same element count, order and edges — **OVERALL PASS** (`/home/xqian/tmp/p93/cfg_{before,after}`) |

**Why `tail_max` is written even though it is inert.** Docs 70 and 90 left such keys unset, and that
precedent was considered. It does not apply here: those keys were not in their arm's TLA, and this
one is. The arm that **measured** the +4/−1 trade, `p92v13`, ran the TLA
`{bragg_wide_anchor_cm:8.0, bragg_wide_anchor_rise_min:1.3, bragg_wide_anchor_tail_max:0.8}`, and
proof A's claim — *the flipped file compiles to exactly what the measured arm ran* — is a 0-line
claim only if the file carries the same three keys. Dropping `tail_max` would have traded that
0 for a 1-line diff plus an inertness argument. The inert lines belong in proof B instead, which is
where they now are.

*(Footnote: the compiled JSON prints `tail_max` as `0.80000000000000004`, the nearest double to 0.8.
Proof A being 0 lines shows the TLA route produced the identical double, so nothing turns on it.)*

## 3. The confirmation arm: production runs what was measured

`p93vprod` is the flipped file with **no TLA at all**, run on `libpin_p92` — the same pinned binary
`p92v13` ran on, md5-identical across all 572 libraries before and after (so a peer's `wcbuild`
cannot be an explanation). 120/120 events, 0 loader deaths, rc 0 (16:30:02 → 16:36:34).

| `p93vprod` vs `p92v13` | result |
|---|---|
| candidates bit-identical on every branch **and every point row** | **596 / 596** |
| candidates moved / `is_stm` flips | **0 / 0** |
| Bee zips differing | **0 events** |
| trees, 120 events | all 8 identical on every event: `T_bad_ch`, `T_cluster`, `T_proj`, `T_proj_data`, `T_rec_charge`, `T_stm_michel`, `T_stm_michel_pts`, `Trun` |
| calib json | 119 same, **0 differ** |

**This is the load-bearing result of the round.** Routing the three keys through the production file
produces bit-for-bit what routing them through the arm's TLA produced, so the +4/−1 trade measured in
doc 92 is the trade production now runs — not something adjacent to it.

## 4. The effect on production (`p93vprod` vs pre-flip `p90vprod`)

589 / 596 candidates are untouched. **Seven** move, and all seven were predicted:

| candidate | record | `is_stm` | what moved |
|---|---|---|---|
| `039252_9/101` | STM_MICHEL | 0 → 1 | `is_stm`, `reject_bits` 12 → 0 |
| `039253_12/93` | STM_MICHEL | 0 → 1 | + the `stop_gamma_*` family |
| `039349_44/28` | STM_MICHEL | 0 → 1 | `is_stm`, `reject_bits` 12 → 0 |
| `039349_51/29` | STM_MICHEL | 0 → 1 | + the `stop_gamma_*` family |
| **`039349_71/37`** | **THRU** | **0 → 1** | **the one false positive** |
| `039253_8/65` | STM_MICHEL | 1 → 1 | `topology_cleared_bits` only |
| `039349_48/54` | STM_MICHEL | 1 → 1 | `topology_cleared_bits` only |

- **`is_stm` fell on nothing.** No stopper was lost to the flip.
- **The last two are the pre-registered "bits-only" class** doc 92's twin caught and doc 91's sizing
  could not see: the rule clears their shape bits, but P1 (`stm_michel_topology_clear`) would have
  cleared them anyway, so the verdict does not move. They are worth stating because they show the
  rule fires *more often* than the `is_stm` count suggests.
- **The `stop_gamma_*` movement on two candidates is downstream and expected**: `stop_gamma_require_stm`
  (doc 85) withholds the capture gamma from a rejected candidate, so a candidate newly accepted as a
  stopper stops having its gamma withheld. The zips differ on exactly those two events
  (`039253_12`, `039349_51`) and on no others.

**The flip also changes the `T_stm_michel` schema: 147 → 149 branches.** Doc 92's writer is
conditional — `bragg_wide_fired` and `bragg_wide_shift_cm` are written only when
`bragg_wide_anchor_cm > 0` — so they were absent from pre-flip production and are present now. No
branch was removed. This is an intended doc 92 output and nothing downstream reads by position, so it
does not affect the trade; it is recorded because **the first version of this gate could not have
seen it.** The per-candidate comparison iterates the *baseline's* branches, so a branch present only
in the new arm is structurally invisible to it, and the fork had dropped `d90_gates.sh`'s explicit
"branches only in" line. That line is restored in `d93_gates.sh`, which matters because the next
round's script forks from it. The "589/596 untouched, seven move" result above is therefore a
statement about the 147 shared branches, and is unchanged by the two added ones.

## 5. The census (`/home/xqian/tmp/p93/score_p9*.json`)

Scored against the **smx1a..smx9** record (601 records, 286 judged stoppers).

| population | pre-flip `p90vprod` | **flipped `p93vprod`** |
|---|---|---|
| `is_stm`, all 576 judged | 238 / 7 / 48 — eff 0.832, purity 0.971 | **242 / 8 / 44 — eff 0.846, purity 0.968** |
| `is_stm`, with a candidate (546) | 238 / 7 / 38 — eff 0.862, purity 0.971 | 242 / 8 / 34 — eff 0.877, purity 0.968 |
| `michel_found`, all 576 judged | 144 / 12 / 25 — eff 0.852, purity 0.923 | **144 / 12 / 25 — identical** |
| `michel_found`, with a candidate (546) | 144 / 12 / 20 — eff 0.878, purity 0.923 | **144 / 12 / 20 — identical** |

**Both denominators are given deliberately, because they are not interchangeable** (doc 89 §1). The
all-judged rows count the judged items the STM tagger never hands on as misses; the with-a-candidate
rows exclude them. For `is_stm` the gap is the 10 judged stoppers with no candidate (48 = 38 + 10);
the 576-judged denominator makes the 286 judged stoppers reconstructible (242 + 44). Quoting an
efficiency without saying which population it is on is how 0.846 and 0.877 get confused for movement.

`census_score --check`: 0 of 14 differ.

**This reproduces doc 92 §6 exactly, from a different record.** Doc 92 computed 238/7/48 → 242/8/44
against the **smx8** record; this is the **smx9** record. Getting the same numbers is an independent
confirmation of `d92_fold_smx9.py`'s own assertion that the tie-break fold moved no stopper, Michel
or judged class — the fold's claim is no longer only self-reported.

**`michel_found` is untouched.** The rule makes the stop call and nothing else; it neither finds nor
loses a Michel. That is exactly why §8's next item exists.

## 6. A gate-script defect, found and fixed

The first gate run printed `*** MOVERS DIFFER FROM THE PREDICTION ***`. **The arm was correct; the
script was wrong**, and the distinction matters enough to record.

`d93_gates.sh` classified movers by the *direction* of the `is_stm` flip, expecting the false
positive to fall 1 → 0. But a false positive is an `is_stm` **0 → 1 on a THRU-judged item** — it
rises, exactly like a gain. All five rises therefore landed in "gains", the cost list came back
empty, and the comparison failed against its own expectation on a run that was in fact precisely as
predicted. The fix classifies by the **record's verdict** (`census_lib.is_stopper`), not by
direction, and additionally pre-registers the two bits-only movers so they cannot read as
unexplained.

Both logs are kept: `/home/xqian/tmp/p93/gates.log` (the defect) and `gates2.log` (corrected). The
seven movers and every number are identical in the two runs — only the summary line changed. This is
the same species as doc 92 §4's `stop_gamma_*` allow-list omission: a gate that is stricter than its
own allow-list produces a false alarm, which is the safe direction to fail, but it still has to be
fixed rather than argued away.

## 7. What this does not settle

- **The false positive is settled as a verdict, not as an easy call.** `039349_71/37` was judged
  STM_MICHEL, then THRU, then THRU again blind — but the pin landed at 6.5 / 1.8 / 3.0 cm across the
  three viewings. It sits inside the record's own resolution (doc 92 §5.1).
- **D 1.0 is still unsettled and must not be re-selected on this record.** The smx8 tranche was drawn
  from the decision boundary, so re-choosing on it is circular; D 1.0 looks better there only because
  two of its four cost items were re-judged into stoppers. A fresh, non-boundary tranche is required.
- **PDHD is OFF and ungraded for this rule.** Doc 92 gated the knob-off path byte-identical on PDHD;
  nothing here says the rule would help or hurt there.
- **Purity away from the record is not measured.** 470 of the record's 601 rows are single-scanner
  agent calls (smx1a); the purity figures describe the judged record only.
- **The two bits-only movers** mean the rule's firing rate exceeds its `is_stm` effect. On this
  record that surplus is harmless (P1 would have cleared them), but it is not a general guarantee.

## 8. Next, ranked

1. **Doc 92 §9 item 2 — the 10 stop-called-but-no-Michel candidates**, recomputed on the **flipped**
   prep (`prep_p93vprod`). This is now the only population left with a named cause and a real size.
   **Scope correction:** doc 90 §8 item 2's "8 unfitted Michels" is a *different, pre-flip*
   population and is stale. Three of the ten — `039252_9/101`, `039349_44/28`, `039349_51/29` — are
   stop-called only because this round's rule fires, so scoping from a pre-flip prep would silently
   drop them.
2. **Then pause the stopper side.** Doc 92 §7 measured its ceiling: of the remaining missed stoppers
   only 5 have a chain-side lever, all five blocked by the same `topology_michel_len_min_cm` 3 cm
   floor, which is what protects purity. At most +1.8 efficiency points, traded directly against
   purity — and this round showed the ground truth itself moves by more than that under a re-judge.
3. **A fresh, non-boundary tranche** if D 1.0 is ever to be settled honestly.
4. **Doc 88 §9.5 item 2** — grade the PDVD flips on PDHD's smx18 record.

**Recommendation: do 1, then 2.** The stopper side is measured out; the Michel side is where the last
named population lives.

