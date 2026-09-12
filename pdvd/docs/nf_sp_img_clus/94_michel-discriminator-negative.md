# doc pdvd/94 — the 10 stop-called-but-no-Michel candidates: no discriminator (doc 92 §9 item 2)

**Status: READ-ONLY. No C++, no jsonnet, no knob, no arm, no scan. The result is NEGATIVE and that
is the deliverable.** Doc 90 §8 item 2 asked to *find a discriminator before any rule*; this doc
reports that on the current record and production none separates the ten from the null, names the two
mechanisms that actually produce them, and recommends closing the item.

**The one-line result:** the ten do not look like the 135 candidates where the chain finds a Michel —
they look like the 93 where the chain and the owner **agree there is no Michel**. Half of them have no
unfitted charge at the stop at all, and the charge-past-the-stop signal that marks every success is
absent (median 1.5 points, against 35 for the successes and 2.0 for the null).

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
S=$IMG/pdvd/docs/scan
export STM_SCAN_RECORD=$S/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_smx7_smx8_smx9_verdicts.json
python3 $X/d94_offfit.py --prep /home/xqian/tmp/p93/prep_p93vprod \
    --json /home/xqian/tmp/p94/offfit.json > /home/xqian/tmp/p94/offfit.txt 2>&1; echo rc=$?
```

`prep_p93vprod` is doc 93's flipped production. `d94_offfit.py` is a fork by duplication
(CLAUDE.md M10) of `d90_offfit.py`, which is untouched.

## 1. The population, and why it is measured on the flipped prep

Bucket (c) of doc 92 §7, recomputed on doc 93's flipped production: **owner-judged Michel, the chain
calls the stop, and the chain finds no Michel.**

| group | n | owner-judged | what it is |
|---|---:|---:|---|
| **TARGET** | 10 | 5 | the item: owner says Michel, chain finds none |
| **N1 (the null)** | 93 | 4 | owner says stopper but **not** Michel; chain finds none — they *agree* |
| TP (context) | 135 | 39 | owner says Michel and the chain finds it |

**Doc 90 §8 item 2's "8 unfitted Michels" is a different, now-stale population.** Three of the ten —
`039252_9/101`, `039349_44/28`, `039349_51/29` — are stop-called *only because doc 93's wide Bragg
read fires*. Scoping from a pre-flip prep would silently drop them. Because those three are the
agent's own output, every table below splits **3 NEW vs 7 pre-existing**: a discriminator that works
mainly on the new three is an artifact of doc 93, not a Michel-finding rule.

**Two caveats on the null, stated rather than hidden.** Only 4 of N1's 93 are owner-judged, and only
5 of the 10 targets are; the rest are single-scanner `smx1a` calls (4 medium, 1 low among the
targets). Doc 90 §6 warns against treating un-pinned items as clean negatives. N1 is a **weak null** —
which matters mainly if the result were positive. It is not.

## 2. Three levers ruled out before any charge was measured

Measured on this prep and record (doc 93's round-2 orientation):

- **"No arm at the stop" is not a discriminator.** `n_stop_arms == 0` on 80% of TARGET — but on
  **97% of the null** and only 8% of TP. A missing stop arm is simply what "no Michel found" looks
  like; it is the *mechanism*, not a signal.
- **PR never fitted-and-dropped anything.** `pf.seg_rej` is empty on all ten, so no discarded residual
  is holding the Michel. (Doc 88's floored residual keep therefore cannot reach them either, as doc
  90 §6 already found for its sample.)
- **There is no companion cluster at the stop.** The only cluster within 10 cm is the candidate's own
  (its id equals the candidate's), at 0.2–2.1 cm.

So if the Michel's charge exists, it is **inside the muon's own cluster** — either swallowed by the
fit, or lying off it unfitted. That is what §3 measures.

## 3. The ten, item by item

`off` = own-cluster imaged points more than 2 cm off the nearest fitted point, within 10 cm of the fit
end. `ctl_off` = the same reading on a body control 30–40 cm upstream, where no Michel can be.

| item | new | conf | off | q_off (k e) | past | past>2cm | ctl_off | arms | dead | pin_rr |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `039252_9/101` | NEW | owner | 12 | 127 | 2 | 0 | 0 | 1 | 1 | 5.0 |
| `039253_3/61` | | medium | 36 | 383 | 91 | 33 | 0 | 1 | 0 | 2.1 |
| `039349_43/66` | | owner | **0** | 0 | 0 | 0 | 0 | 0 | 1 | 4.8 |
| `039349_44/28` | NEW | owner | **0** | 0 | 4 | 0 | — | 0 | 2 | 3.7 |
| `039349_51/29` | NEW | owner | 32 | 833 | 8 | 0 | 0 | 0 | 0 | 9.0 |
| `039349_58/69` | | medium | 15 | 109 | 1 | 0 | 0 | 0 | 1 | 3.0 |
| `039349_60/40` | | medium | **0** | 0 | 0 | 0 | 0 | 0 | 1 | 6.1 |
| `039349_64/65` | | low | 31 | 146 | 8 | 6 | **23** | 0 | 1 | — |
| `039349_69/56` | | medium | **0** | 0 | 0 | 0 | 0 | 0 | 3 | — |
| `039349_72/11` | | owner | **0** | 0 | 0 | 0 | 0 | 0 | 2 | 3.0 |

- **Five of the ten have `off == 0`:** no unfitted charge at the stop whatsoever. No off-fit rule can
  ever reach them. Four of those five carry an owner pin **3.0–6.1 cm** back from the fit end — the
  fit-through signature of doc 86 and doc 90 §6.2: the Michel *is* the fit's own last few cm.
- **The body control does real work.** `039349_64/65` reads `off` 31 at the stop but `ctl_off` 23
  thirty cm upstream: that cluster is simply messy, so its off-fit charge is not a stop feature. It is
  also the one `low`-confidence item. Every other item's control is 0, so the measurement itself is
  sound.

## 4. The null has the same signal

| group | n | `off ≥ 1` | med `off` | med `q_off` | **med `past`** | med `ctl_off` |
|---|---:|---:|---:|---:|---:|---:|
| TARGET | 10 | 50% | 6.0 | 54 | **1.5** | 0 |
| **N1 (null)** | 93 | **30%** | 0.0 | 0 | **2.0** | 0 |
| TP (chain succeeds) | 135 | 78% | 7.0 | 41 | **35.0** | 0 |
| TARGET: the 3 NEW | 3 | 67% | 12.0 | 127 | 4.0 | 0 |
| TARGET: the 7 pre-existing | 7 | 43% | **0.0** | 0 | **0.0** | 0 |

**`past` is the column that decides this — but read the right comparison.** On the ten the median
charge past the fit end is **1.5 points**, against the null's **2.0**: indistinguishable. *That*
equality is the result. The TP column (median **35**) is shown for scale, not as an independent test:
those are candidates where a Michel object was actually built, and charge past the stop is close to a
precondition for building one, so the contrast is partly definitional. The honest statement is the
weaker-sounding one, and it is enough: **the ten do not merely fail a threshold — on the measurement
that matters they are the null.**

**Threshold sweep** — a usable rule needs high TARGET and low N1:

| rule | TARGET /10 | N1 null /93 | TP /135 | of the 3 NEW |
|---|---:|---:|---:|---:|
| `off ≥ 1` | 5 | 28 | 105 | 2 |
| `off ≥ 5` | 5 | 21 | 83 | 2 |
| `off ≥ 10` | 5 | 13 | 52 | 2 |
| `off ≥ 30` | 3 | 2 | 23 | 1 |

The strict end recovers **3 items at 2 false fires** — but one of those three is `039349_64/65`, whose
control says its charge is a messy cluster, and another is `039253_3/61`, a single-scanner `medium`
call. What survives is roughly **one owner-judged item** (`039349_51/29`, itself one of doc 93's
three) against a null that is mostly un-owner-judged. That is not a discriminator; it is noise with a
favourable cut.

**The artifact check fails in the predicted direction.** The 3 items doc 93 created have median `off`
12; the 7 pre-existing have median **0**. Whatever off-fit signal exists is concentrated in the
agent's own recent output — exactly the confound the split was built to expose.

## 5. What the ten actually are

Doc 90 §6's two mechanisms, now measured on the current population:

The obvious reading of the five `off == 0` items is **fit-through**: no charge beside the fit, owner
pins 3.0–6.1 cm back, so the Michel must be *inside* the fit. That reading is testable, and it mostly
**fails**. If the Michel were inside the fit, the fit's own rows from the pin to the end would read
**above** the plateau. Restoring doc 90 §D's measurement (which the first version of `d94_offfit.py`
had dropped):

| item | pin_rr | plateau (k e/cm) | med tail | **tail / plateau** | reading |
|---|---:|---:|---:|---:|---|
| `039349_69/56` | — | 46 | 94 | **2.05** | a real excess — the Michel plausibly *is* in the fit |
| `039349_43/66` | 4.8 | 54 | 64 | 1.19 | at plateau |
| `039349_60/40` | 6.1 | 55 | 59 | 1.08 | at plateau |
| `039349_44/28` | 3.7 | 83 | 56 | **0.67** | *below* plateau |
| `039349_72/11` | 3.0 | 60 | 40 | **0.68** | *below* plateau |

**So "fit-through" is confirmed on one of the five, not five.** Two sit at plateau and two read *below*
it — the opposite of the excess a swallowed Michel would deposit. On four of these five there is no
Michel-like charge **anywhere**: none beside the fit (`off == 0`), and none in the fit's own tail.

This is a correction to the first draft of this section, which asserted fit-through for all five from
`off == 0` plus the pins alone. The absence of off-fit charge is not evidence of charge inside the
fit; it had to be measured, and when measured it mostly is not there.

**It strengthens the negative result rather than weakening it.** The ten are not a population whose
charge the chain is looking for in the wrong place — for most of them there is no excess charge to
find at all. That is why no threshold on any of these quantities separates them from the null, and it
is why the remaining lead is narrow: at most one or two items where splitting the fit could recover a
Michel object, which is a PR-level change, not a knob.

## 6. What this does not settle

- **It does not say the owner is wrong.** The 5 owner-judged targets are Michels; the record is the
  ground truth. It says the chain has **no charge-geometry evidence** to find them with.
- **The null is weak** (4 of 93 owner-judged). A positive result on it would need re-judging first; a
  negative one is the safer direction to be wrong in, and is what we have.
- **Splitting the fit is untested.** The fit-through five might be reachable by re-fitting the last
  few cm rather than by looking beside the fit. That is a substantially larger change than a knob, it
  would touch PR rather than `CheckSTM_Michel`, and nothing here sizes it.
- **Purity away from the record is unmeasured** — 470 of the record's 601 rows are single-scanner.

## 7. Next, ranked

1. **Pause the stopper *and* Michel sides.** Doc 92 §7 measured the stopper ceiling at 5 items behind
   one purity-trading floor; this doc measures the Michel side's last named population and finds no
   lever. Both halves of the PR are now closed by measurement rather than by fatigue.
2. **Doc 88 §9.5 item 2** — grade the PDVD flips of docs 83–93 on PDHD's `smx18` record. This is the
   highest-value item left: it tests everything already shipped against a detector that has not voted.
3. **A fresh, non-boundary tranche** if D 1.0 (doc 92 §6) is ever to be settled honestly.
4. **Fit-splitting for the fit-through five** — only if the owner wants a PR-level change; it is not a
   knob and this doc does not size it.

**Recommendation: 1, then 2.** The measured headroom on this chain is exhausted; the unexamined risk
is cross-detector, not further tuning on PDVD.
