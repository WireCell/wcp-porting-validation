# 96 — The region's scope, and the prediction bias behind the phantom (doc 95 §9 items 1–2)

Doc 95 shipped the region-based Michel energy into PDVD production and closed with a ranked next
list. The owner asked for items 1 and 2. **Both were misdescribed in doc 95 §9, by me**, and this
doc's first job is to correct my own record rather than build on it.

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
export STM_SCAN_RECORD=$IMG/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_smx7_smx8_smx9_verdicts.json

cd /home/xqian/toolkit-dev/toolkit && wcbuild && ./build/clus/wcdoctest-clus   # 396/396
cp -a ../local/lib /home/xqian/tmp/p96/libpin_p96
md5sum /home/xqian/tmp/p96/libpin_p96/*.so* > /home/xqian/tmp/p96/libpin_md5_before.txt

PIN=/home/xqian/tmp/p96/libpin_p96 LOGD=/home/xqian/tmp/p96 WAVE=off   bash $X/d96_arms.sh
PIN=... WAVE=base  bash $X/d96_arms.sh     # scope 0 at R=40, the same-binary partner
PIN=... WAVE=scope bash $X/d96_arms.sh     # scope 1 at R=40, the measurement arm
bash $X/d96_gates.sh > /home/xqian/tmp/p96/gates.log 2>&1
python3 $X/d96_scope.py    --arm p96vscope --twin --json /home/xqian/tmp/p96/scope.json
python3 $X/d96_pedestal.py --arm p96vscope
PRE_EVT=/home/xqian/tmp/p96/pre_wct-pr-perevt.jsonnet SCOPE=1 R=10.0 CTL=35.0 bash $X/d96_proofs.sh
bash $IMG/abtest/compile_all_cfg.sh /home/xqian/tmp/p96/cfg_after96
bash $IMG/abtest/cmp_cfg.sh /home/xqian/tmp/p95/cfg_after /home/xqian/tmp/p96/cfg_after96
PIN=... WAVE=confirm bash $X/d96_arms.sh   # p96vprod, the flipped file with NO TLA
python3 $X/d96_confirm.py --arm p96vscope --confirm p96vprod --radius 10.0 --scope 1
```

Pre-registration: `/home/xqian/tmp/p96/pred.txt`, **mtime before any doc-96 code was edited,
built, launched or measured**. Graded by name in §6, misses included.

## 1. The two corrections to doc 95 §9

**§9 item 1 said restricting the region to `own_blob > 0` implements doc 78 item 9's "the main
cluster and the admitted companions", and that it was "one key". Both halves were wrong.**

- `own` bit 2 covers only `rec.unfit_dot_clusters` — the *segment-less* subset of companions —
  and `n_dot_clusters_unfit` is **0 on all 596 PDVD candidates**, so bit 2 never fires. Value
  counts confirm it: doc 95's arm carries only 0 and 1. The column was **main-cluster-only**.
- A companion that **did** produce segments is preloaded, its charge **is** in the union charge
  maps, and it set **no bit at all**. The flag could not express the specification's scope.
- And it is not a config key: no scope knob existed. It is new C++.

**§9 item 2 said the pedestal is "a uniform +0.084 MeV/cell residual" whose "mechanism is
unexamined". Both halves were wrong.** 0.084 is a *mean* over a strongly prediction-dependent
quantity (§5), and docs 42/44 had already measured the bias per plane and named its leading
mechanism.

Two further corrections this round publishes:

| where | the claim | the correction |
|---|---|---|
| doc 95 jsonnet block | "any other radius or scope is recomputable offline without re-running" | true only out to the **running** radius. `own_blob` and role-0 rows are computed inside `if (in_region \|\| in_ctl)`, so at production's R=10 a cell beyond 10 cm has no row and `own_blob == 0` means "not computed". Corrected in the config comment. |
| doc 95 §8, §9.3 | plane-drop "57 % → 84 %" | crosses populations (164 Michels vs 512 all-judged) **and** branches (`michel_q2d_dropped_plane` vs `_region_dropped_plane`). Not restated here as a delta. |
| doc 95 §9.4 | grade on PDHD `smx18` | the current PDHD record is **smx22**, and `census_lib.py:65-69` says to use `d18_census.py` for headline PDHD numbers. |
| doc 95 §4 tables | class medians | doc 95's offline MeV is **signed**; production's `michel_ke_q2d_region` **floors at 0** (`dQ <= 0 → 0.0`). This doc grades on the floored quantity. |

## 2. What was built (`clus/`, default OFF)

One knob, `michel_q2d_region_scope` (int, default **0**). Every behaviour change rides on
`scope > 0`, so scope 0 is byte-identical to doc 95's `d65f8165`:

| value | `own_blob` column | region/ctl sum | clamp |
|---|---|---|---|
| **0** (default) | bits {1,2}, single `(face,wire)` — doc 95 verbatim | every cell in radius | no |
| **1** | bits {1,2,**4**}, `fw`-swept | `own != 0` — main **+ companions** | yes |
| **2** | as 1 | `own & 1` — main cluster only | yes |

Three edits in `michel_q2d_estimate`, plus the companion list passed to it:

- **bit 4** marks a preloaded *fitted* companion — the population that set no bit before. It
  **fires on 2.5 % of region cells**, so the specification's scope is real and was uncredited.
- the own test now sweeps **every `(face, wire)`** the channel maps to, as the role tests already
  did. Testing `fw.front()` alone was cosmetic for a column and a systematic loss on wrapped
  channels once it *filters*. Own coverage **69.9 % → 78.2 %**.
- a **negative muon prediction is clamped to 0** before subtracting. `pred_mu < 0` occurs on
  0.167 % of cells (min −58 016 e) and can only inflate.

**A trap, stated because it is easy to misuse.** Under `scope > 0`, `michel_q2d_region_nd_*`
(dead cells) and `michel_q2d_n_role0` stay **region-wide** by design, while
`michel_q2d_region_n_*` counts only what was summed. So **`nd / n` is not a fraction and can
exceed 1.** The gate treats both as frozen and checks they do not move.

## 3. Gates, proofs, configs

| gate | arms | verdict |
|---|---|---|
| 1 OFF identity (PDVD) | `p96voff` vs `p95vprodb` | **PASS** — 596/596 bit-identical on every branch **and point row**; 0 movers; 0 `is_stm` flips; 0 zips; branch sets 198 = 198; all **9 trees identical on 120 events**; census unmoved (`is_stm` 242/8/34, michel 144/12/20) |
| 2 OFF identity (PDHD) | `p96hoff` vs `p95hoffb` | **VOID — see §3.1.** Not a pass, not a failure of this change |
| 3 cross-binary inertness | `p96vbase` vs `p95vq2db` | **PASS** — 596/596 across two binaries; the new key at its default 0 is inert |
| 4 the scope effect | `p96vscope` vs `p96vbase` | **PASS** — 21 of 21 pre-registered movers, **0 unexpected**, **0 frozen moved**, 0 `is_stm` flips, 0 `michel_found` flips, 0 point rows, 0 zips |

**4b, the cell-table positive control** (invisible to a branch gate):

```
p96vbase   own_blob {0: 411225 (30.1 %), 1: 954333 (69.9 %)}          <- doc 95's numbers, exactly
p96vscope  own_blob {0: 297763 (21.8 %), 1: 1033665 (75.7 %), 4: 34130 (2.5 %)}
```

At scope 0 the column is byte-for-byte doc 95's; at scope 1 bit 4 appears. This is the cleanest
evidence that the gating sits exactly where intended.

**Proofs** (`d96_proofs.sh`): **A = 0 lines** (the flipped file compiles to exactly production's
key set); **B** forces every key back to its C++ initializer and reproduces PRE, with
`michel_q2d_region_scope: 0` as the inert-key line and `m_michel_q2d_region_scope{0}` grepped
from source; **C = exactly one added line**, `"michel_q2d_region_scope": 1` (the script states
more than one is a defect); **D** PDHD unchanged vs git HEAD, `michel_q2d` lines in PDHD **0**.

**Configs**: `compile_all_cfg.sh` **16/16 jobs, 0 failures**; `cmp_cfg.sh` against doc 95's set —
**NORMDIFF 0 on all 16**, elements/order/edges same, **OVERALL PASS**. Note what this does *not*
say: none of the 16 jobs is `wct-pr-perevt.jsonnet`, and none imports it, so it is evidence my
edit leaked nowhere else — not a second check on the flip.

### 3.1 Why gate 2 is void, and what carries PDHD instead

The gate reports NOT IDENTICAL, and the cause is a **stale baseline**, not this change:

```
my baseline p95hoffb ran     2026-09-11 ~18:14
peer 51a25063 landed         2026-09-11 20:10:40   +33 lines in pdhd/wct-pr-perevt.jsonnet
                             (ks_margin -0.02, max_candidates 64, bragg_peak_anchor,
                              bragg_anchor_geo_fallback; efficiency 0.415 -> 0.544)
my arm p96hoff ran           2026-09-12 05:36-05:44        <- AFTER that flip
peer b5b9b9fc landed         2026-09-12 06:05:44   (doc pdhd/22)
```

The diff is exactly that config delta: +1 branch `bragg_anchor_fallback`, 16 NEW candidates (the
candidate cap), 20 `is_stm` flips (the efficiency rise). The moved branches are
`bragg_anchor_shift_cm`, `chain_support_min`, `comp_fwd*`, `comp_bwd*`, `contrast`,
`contrast_expected`, `ks_flat`, `ks_mu`, `n_tail` — **not one `michel_q2d*` branch**.

What carries PDHD instead: (i) `grep -c michel_q2d pdhd/wct-pr-perevt.jsonnet` = **0**, so
`michel_q2d_estimate` — where every substantive edit lives — is never called there; (ii) gate 1
is the *stronger* test, since PDVD production has `michel_q2d: true` and therefore exercises that
function fully at scope 0, returning 596/596; (iii) gate 3 confirms it across two binaries.

A clean re-baseline was **not** attempted: it needs the pre-change binary reinstalled into the
shared `local/lib` while a peer is actively running PDHD arms and committing — the hazard the
pinning discipline exists to prevent. **Recorded as void; pred.txt X1's PDHD half is UNGRADED,
not held.**

## 4. The measurement

Twin: the offline reader reproduces the C++ scope-1 sums on **596/596 candidates, 0 differing**.
`K` = 4.27058e-05 MeV/electron, spread **5.7e-15** over 576 candidates — doc 95's constant.

### 4.1 Region energy at R=10 by scope, median MeV, floored

| class | scope 0 (all) | scope 1 (own) | scope 2 (main) |
|---|---:|---:|---:|
| TP_found (135) | 34.65 | **34.28** | 31.81 |
| TARGET (10) | 8.56 | 6.35 | 6.82 |
| ZERO_CTL (97) | 4.91 | **4.75** | 4.51 |
| THRU (270) | 5.20 | **3.79** | 3.59 |
| **ratio TP / phantom** | **7.06** | **7.22** | 7.05 |

The zero-control **does not rise** under a narrowed scope, which was the flip precondition.

### 4.2 The headline gain is not significant, and the flip does not rest on it

Bootstrap, 20 000 paired resamples of TP_found and ZERO_CTL, same draw per scope:

| | ratio | 95 % CI |
|---|---:|---|
| scope 0 | 7.06 | [4.59, 9.44] |
| scope 1 | 7.22 | [5.05, 11.25] |
| **difference** | **+0.299** | **[−0.612, +2.156]** — spans zero |

Scope 1 beats scope 0 in only **71.8 %** of resamples. **The pre-registered rule selected scope 1
on a point estimate the data cannot support.** Per pred.txt's escalation clause the conflict was
reported rather than split, and the owner ruled (2026-09-12): **flip on fidelity, not the ratio.**

What *is* well measured, and is the actual case for scope 1: the estimator now sums the cells its
own specification names (bit 4 fires on 2.5 % of cells); through-going contamination falls
**27.1 %** on a **270**-candidate sample; the clamp removes an unphysical 0.091 %; and found-Michel
energy is untouched at **−1.1 %**.

### 4.3 The TARGET drop is not signal loss

The median moves 8.56 → 6.35, but the four largest items barely move:

| item | scope 0 | scope 1 | body ctl |
|---|---:|---:|---:|
| `039349_51/29` | 46.8 | **46.8** | 5.0 |
| `039253_3/61` | 45.3 | **44.9** | 1.9 |
| `039349_58/69` | 21.8 | **20.6** | 4.5 |
| `039252_9/101` | 16.8 | **16.7** | 2.9 |
| `039349_64/65` | 10.4 | 5.5 | **39.8** ← doc 94 unreliable |
| `039349_43/66` | 6.7 | 3.3 | 1.4 ← doc 94's `off == 0` |
| `039349_69/56` | 2.2 | 1.8 | **18.4** ← doc 94 unreliable |

The median is dragged by two items doc 94 already called unreliable by their own body control,
plus doc 94's `off == 0` case. The recovery that motivated doc 95 is preserved.

## 5. Item 2 — the prediction bias, diagnosed not fixed

The owner ruled this **diagnosis-only**: any real fix lives in `masked_response_prediction` /
`TrackFitting`, which PDHD, SBND and uBooNE all share (CLAUDE.md §5 item 3, M10).

**It is not a uniform pedestal.** Median `(charge − pred_mu)/pred_mu` by `pred_mu` decile:

```
ALL        80.542  0.795 -0.113 -0.080 -0.022  0.032  0.063  0.081  0.092  0.105
TP_found   46.217  0.239 -0.175 -0.109 -0.031  0.018  0.062  0.072  0.092  0.106
ZERO_CTL   52.164  0.287 -0.127 -0.082 -0.016  0.037  0.061  0.074  0.083  0.094
THRU       96.089  0.977 -0.084 -0.088 -0.025  0.025  0.064  0.083  0.096  0.121
```

The fit slightly **over**-predicts in the middle deciles and under-predicts ~10 % at the top,
where the Bragg peak lives — and the shape is the same inside every record class, so it is a
property of the **fit**, not of a class (**P1 HELD**).

**Per plane, and this is where my planning-stage explanation was wrong.** I claimed the pooled top
decile was "dominated by U and V". Measured, the pooled top decile is **35.0 % U / 32.9 % V /
32.2 % W** — nearly even. The real mechanism is that the per-plane *medians* differ
(U +0.151, V +0.171, W +0.061 over all role-1 cells), and a pooled median over three populations
with different centres lands between them and describes none. The apparent agreement of pooled
+0.105 with doc 42's W `B_foot` (−0.101) was a coincidence of pooling.

| plane | own top-decile median res/pred | doc 42 \|B_foot\| | within 1.5× |
|---|---:|---:|---|
| U | +0.148 | 0.221 | yes |
| V | +0.157 | 0.217 | yes |
| **W (collection)** | **+0.026** | **0.101** | **no — ~4× smaller** |

**P2 MISSED**, by its own wording (it was registered on the collection plane). The *ordering*
matches doc 42 (induction bias > collection bias); the collection magnitude does not.

**P3 UNRUNNABLE, as pre-declared.** The closure test `Σ pred_mu / Σ dQ over mask_mu` needs the
per-point fitted `dQ`. `T_stm_michel_pts` persists `cluster_id / role / seg_id / x / y / z` only.
pred.txt committed in advance to reporting this as unrunnable rather than substituting a weaker
proxy. Emitting the fit's `dQ` per point is a one-branch change in `TrackFitting`'s writer.

**P4 MISSED**: 36.0 % of the ZERO_CTL phantom within R=10 comes from cells with `pred_mu == 0`,
against a registered bar of ≥40 %. Role 0 carries no muon prediction at all yet reads
0.089 MeV/cell against role 1's 0.084 — so the additive floor dominates the mean, but not by the
margin I predicted.

## 6. The pre-registration, graded by name

| | prediction | result |
|---|---|---|
| S1 | scopes 1 and 2 within 5 % on TP_found | **MISSED** — 7.2 % (34.28 vs 31.81). Fitted companions carry real charge; that is what bit 4 bought |
| S2 | ratio lands in 7.3–8.5 | **MISSED** — 7.22 |
| S3 | TP_found falls 5–15 % | **MISSED** — 1.1 % |
| S4 | the `fw` sweep recovers cells | **HELD** — 69.9 % → 78.2 % |
| S5 | the clamp lowers sums by < 1 % | **HELD** — 0.091 % |
| P1 | the shape is a property of the fit | **HELD** |
| P2 | top decile consistent with doc 42 on W | **MISSED** — +0.026 vs 0.101 |
| P3 | closure ratio below 1 on all planes | **UNRUNNABLE**, pre-declared |
| P4 | ≥40 % of the phantom from unmodelled cells | **MISSED** — 36.0 % |
| X1 | OFF byte-identical both detectors | PDVD **HELD**; PDHD **UNGRADED** (§3.1) |
| X2 | cross-binary inertness | **HELD** — 596/596 |
| X3 | only the registered branches move | **HELD** — 21/21, 0 unexpected, 0 frozen |
| X4 | `michel_q2d_valid == 1` on every candidate | **HELD** |
| X5 | the twin holds to 1e-6 | **HELD** — 596/596, 0 differing |

Five of fourteen missed, one unrunnable, one ungraded. **Doc 95 §4.3's "7.1 → 7.6" did not
reproduce** — its "own" was bit-1-only without the `fw` sweep, a narrower filter than the
specification-faithful one. pred.txt registered those numbers as *predictions* for exactly this
reason.

## 7. Confirmation

`p96vprod` runs the **flipped file with no TLA at all**, on the same pin — production's own
route. 120/120 events, no dead-net warning. `d96_confirm.py` predicts what that arm must write,
from the measurement arm's *own* cell table filtered to R=10 and the scope-1 predicate (clamp
included), and compares item by item:

```
twin prediction from p96vscope at R=10.0 scope=1: 596 candidates
confirmation arm p96vprod                       : 596 candidates
  candidates where the arm reproduces the twin EXACTLY: 596 / 596
  VERDICT: PRODUCTION RUNS THE READING THIS DOC MEASURED -- every candidate matches the twin
```

**Why this is needed and what it closes.** Proof A (§3) shows only that the flipped file compiles
to production's key set — a claim about the *config*. It cannot show those keys are what was
measured, because the measurement arm ran the wide diagnostic radius (R=40) while production
carries R=10. This closes the other half by exact arithmetic on both sides, with no tolerance to
tune: a match on every candidate, or the flip does not stand. There is no third outcome.

Independently visible in production output: a `p96vprod` file carries
`own_blob {0: 2496, 1: 5223, 4: 94}` — **bit 4 is present**, so the specification-faithful scope
is live through the config rather than only through the measurement arm's TLA. Its cell count is
smaller than the arm's because production runs R=10 against the arm's R=40, which is exactly the
"not recomputable beyond the running radius" correction of §1.

## 8. What this does not settle

- **There is still no energy truth anchor.** The record carries verdicts, Michel kind, confidence,
  tags and a stop pin — no energy field. A better ratio is better *selection*, never evidence that
  either number is *correct*.
- **The headline metric did not move significantly** (§4.2). The flip is justified by
  specification fidelity, the 270-candidate THRU reduction and the clamp — and by nothing else.
- **PDHD is ungraded this round** (§3.1), and a valid PDHD OFF gate is owed once its config is
  quiet.
- **The ~2 MeV phantom is unchanged in kind.** Scope 1 moves it 4.91 → 4.75. §5 is why: most of it
  is an additive floor from cells the muon fit does not model, not a prediction defect a scope can
  remove.
- **Bit 2 is still dead** on PDVD (`n_dot_clusters_unfit` 0/596). Scope 1 and scope 2 differ only
  through bit 4.

## 9. Next, ranked

1. **The prediction bias is now a specific, testable claim and it belongs in `TrackFitting`.**
   doc 44 §7's named test — refit with σ driven by the *predicted* rather than the measured charge,
   or with the relative term off — against the per-plane profile in §5. It is a cross-detector
   change needing PDHD/SBND/uBooNE gates, so it is its own round. This is the only lead that would
   make the energy more *accurate* rather than better *selected*.
2. **Emit the fit's per-point `dQ`** so the closure test of §5/P3 can actually run. One branch in
   `TrackFitting`'s writer, and it converts an unrunnable check into a measurement.
3. **A valid PDHD OFF gate**, once the PDHD config is not being actively flipped.
4. **The plane rule** remains load-bearing and un-retuned; doc 95 §9.3's framing needs restating on
   the region branch alone before it can be acted on.
5. **Doc 88 §9.5 item 2** — grade the PDVD flips on PDHD's record, now **smx22** with
   `d18_census.py`, not smx18 with `census_lib`.
