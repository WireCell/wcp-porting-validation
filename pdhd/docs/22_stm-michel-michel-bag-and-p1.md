# doc pdhd/22 — the Michel bag, P1, and six knobs that were never actually on

**Status:** the **Michel bag is flipped** in PDHD production (16 keys), and doc pdhd/21's
already-committed five are now gated against the binary production actually runs.
`michel_found` **56/11/30 → 68/2/18** (purity 0.836 → 0.971 *and* efficiency 0.651 → 0.791) for
**one** stopper false positive; `is_stm` 80/0/67/110 → **82/1/65/109**. Two sets are measured and
**held** for the owner, each because it spends purity: the **wide Bragg anchor** (free alone,
+1 TP/+1 FP in combination) and **P1** (+12 stoppers, free under one truth reading and 2 FPs
under the other). No C++ change; the flip is config-only.

**This round also retracts a claim from doc pdhd/21.** Six keys in round h21's Michel bag — and
the wide anchor that `h21w` "measured" — did not exist in that round's binary and were silently
ignored. See §1; doc 21 §5 is corrected in place.

## 0. Repro

```sh
I=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img ; H=/home/xqian/tmp/h22
PIN=/home/xqian/tmp/p65/libpin_p65        # toolkit d65f8165, libWireCellClus md5 3e23bf8a
S=$I/pdvd/docs/nf_sp_img_clus/scripts/d53_run_arms.sh

# the finding of sec 1 -- which knobs existed in the OLD pin (082376c5) and which did not
git -C /nfs/data/1/xqian/toolkit-dev/toolkit grep -l '"stop_snap_reachable"' 082376c5 -- clus/   # nothing
git -C /nfs/data/1/xqian/toolkit-dev/toolkit grep -l '"ks_margin"'           082376c5 -- clus/   # present (control)

# the Michel bag, as one arm.  NEVER -S stm_michel_knobs= : that REPLACES the bag.
# No spaces inside the TLA.  These same 16 keys are now IN the file, so h22conf below
# reruns this with no TLA at all and must come back bit-identical.
BAG16='stop_retreat_max:2,stop_split_max:1,split_kink_min_deg:10,michel_gamma_collect:true,michel_gamma_radius_cm:50.0,moved_stop_michel_kink_min:60.0,retreat_tail_strict:true,stop_local_michel_pieces:true,michel_range_energy_guard:true,stop_local_residual_cm:5.0,stop_gamma_require_stm:true,moved_stop_michel_reach_min_cm:6.5,michel_near_stop_arm_cm:5.0,stop_local_residual_min_points:5,stop_local_residual_min_len_cm:5.0,stop_snap_reachable:true'
ARM=h22g DET=pdhd SRC=d16hnu JOBS=6 PIN=$PIN LOGD=$H/arm_h22g \
  PR_TLA="-S stm_michel_extra={$BAG16}" bash $S

# the confirmation arm: the FLIPPED FILE, no TLA.  Must be bit-identical to h22g.
ARM=h22conf DET=pdhd SRC=d16hnu JOBS=8 PIN=$PIN LOGD=$H/arm_h22conf bash $S

# grade (the grader REFUSES to print anything unless p82bhoff first reproduces 61/0/86/110)
python3 $I/pdhd/docs/scan/h21/d21_grade.py h22base h22g h22w h22c h22p1 h22conf
python3 $I/pdhd/docs/scan/h21/d21_michel_census.py p82bhoff h22g h22conf
python3 $I/pdhd/docs/scan/h22/d22_truth2.py h22g h22p1        # BOTH truth readings

# the compiled-config proof, before/after, structurally (NOT by grep -- see sec 1)
python3 $I/pdhd/docs/scan/h22/d22_cfg_keys.py $H/compiled_before.json $H/compiled_h22g.json

# P1's offline twin, on P1's exact admission test; and what is left on the baseline
python3 $I/pdhd/docs/scan/h22/d22_p1_twin.py h22g
python3 $I/pdhd/docs/scan/h22/d22_bits.py    h22g

# the bit-identity gate that makes the flip quotable
python3 $I/pdvd/docs/nf_sp_img_clus/scripts/d51g_branch_census.py \
  --before "$I/pdhd/work/*_h22g" --after "$I/pdhd/work/*_h22conf" \
  --before-arm h22g --after-arm h22conf --pts --out $H/gates/g_h22conf.txt
```

## 1. The finding: six knobs in round h21's Michel bag never executed

Round h21 graded a 21-key "full production candidate" (`h21z`) on the pin `libpin_p82` =
toolkit `082376c5`. **Six of those keys did not exist in that binary.** They were added to
`CheckSTM_Michel.cxx` *after* the pin:

| key | introduced by |
|---|---|
| `michel_near_stop_arm_cm` | `5904f1d2` (doc pdvd/83) |
| `moved_stop_michel_reach_min_cm` | `77f4d2d6` (doc pdvd/84) |
| `stop_gamma_require_stm` | `51adc923` (doc pdvd/85) |
| `stop_local_residual_min_points`, `stop_local_residual_min_len_cm` | `314186f8` (doc pdvd/87) |
| `stop_snap_reachable` | `a4bba314` (doc pdvd/88) |
| `bragg_wide_anchor_cm` *(tested by arm `h21w`)* | `ec6990e0` (doc pdvd/92) |

`get(config, key, default)` drops an unknown key **silently**, so these were passed and ignored.
The compiled config carried them; the component never read them. A compiled-config proof (M6)
cannot catch this — the keys *are* in the JSON. Only the binary knows.

**Two consequences.**

1. `h21z`'s Michel gain came from **10 live keys, not 16**. Flipping all 16 on that evidence
   would have put six never-executed knobs into production.
2. **Arm `h21w` was vacuous.** Doc pdhd/21 §5 and the `stm_michel_knobs` comment block both
   record "the wide anchor recovers NOTHING on PDHD". That measured a key the binary ignored:
   `h21w` being identical to `h21g` is the *signature of an inert key*, not a negative result.
   Corrected in §5 below and in both places in the tree.

### The instrument, and the one I threw away

Authoritative test — does the pinned **source** contain the key:

```sh
git grep '"<knob>"' 082376c5 -- clus/
```

**Positive control: `ks_margin`**, which is present at `082376c5` and demonstrably works (arm
`h21k`, +11 stoppers). Any instrument that reports `ks_margin` absent is broken.

A `strings -x` test on the binary was tried **first and rejected** by exactly that control: it
reported `ks_margin` missing from a binary that uses it, because the literal is stored
overlapping as `m_ks_margin`/`vks_margin` and a whole-line match fails.

This round produced three more vacuous checks before they were caught — a `grep '"key":'` against
a compiled JSON that writes `"key" : value` (space before the colon), a patch-id loop that
printed "not upstream" because it never ran, and a file-count that listed 23 while claiming 24.
The countermeasure is in the tree: `docs/scan/h22/d22_cfg_keys.py` parses the JSON structure
instead of pattern-matching it, and ships with a self-diff negative control that must report
`0 added, 0 removed, 0 changed`.

## 2. The push

Commit `51a25063` (doc pdhd/21) was committed but unpushed. Pushed this round as **`a60905a3`**.

SSH is dead in this tree (`Permission denied (publickey)`), and local `main` is **152 commits
diverged** from a stale `origin/main`; the true remote head was `84058b3a`. Local `main` was
therefore **never** pushed. Method: fetch over HTTPS with the `gh` credential helper, cherry-pick
`51a25063` onto `FETCH_HEAD` in a throwaway worktree, gate on patch-id, push, and verify by
**reading the remote SHA** rather than trusting `rc=0`.

* patch-id before `2444a97c…` == after `2444a97c…` — the peer's history rewrite preserved the patch
* `84058b3a..a60905a3`, and `ls-remote` returns `a60905a3` == the local SHA

## 3. The pin, and the gate nobody had run

Production loads `local/lib`, **not** the pin. Every arm in round h21 ran `libpin_p82`
(`082376c5`), while `local/lib` had moved to `3e23bf8a`. So the flip committed in `51a25063` had
never been shown to describe what production does — a gap in *already-committed* work.

Pin for this round: **`/home/xqian/tmp/p65/libpin_p65`**, toolkit `d65f8165`, `libWireCellClus.so`
md5 `3e23bf8a`, identical before and after every arm. It is my own build of a clean tree, and
`build/clus/libWireCellClus.so` == `local/lib/libWireCellClus.so` == the pin (all `3e23bf8a`),
which is the M1 freshness proof. `./build/clus/wcdoctest-clus`: **396/396 cases, 23889
assertions, 0 failed.**

**Arm `h22base`** — the flipped production file, **no TLA**, on the new pin. Pre-registered
prediction: identical to `h21p` in every scored respect.

| gate | result |
|---|---|
| census vs `smx22` | **80/0/67/110**, purity **1.000**, efficiency **0.544** — identical to `h21p` |
| the 19 recovered items | the same 19, by name |
| new false positives | **0** |
| `michel_found` | **56/11/30**, moved on 0 items |
| branch census vs `h21p` (same config, other binary) | **341/341 bit-identical on all 131 shared branches; 0 `is_stm` flips; point geometry identical 341/341; 0 role moves; no new branches** |
| pin md5 before/after | `3e23bf8a` / `3e23bf8a` |

**The twin HELD EXACTLY.** The peer's seven commits between `082376c5` and `d65f8165` are
inert-when-off for the PDHD config, so commit `51a25063` does describe production. Had this
failed, the round would have stopped and reported rather than tuned (CLAUDE.md §5.7).

**Standing requirement this establishes:** production's `local/lib` must be at or after
`d65f8165` for the flipped bag to behave as measured.

## 4. The Michel bag

All arms on `libpin_p65`, against `h22base`, population fixed at the committed 303 items.
Compiled-config proofs taken **before** each arm: `h22t` **10 added / 0 removed / 0 changed**,
`h22g` **16 / 0 / 0**, `h22w` **3 / 0 / 0**, `h22c` **19 / 0 / 0**, with a self-diff negative
control at `0 / 0 / 0`.

| arm | keys | `is_stm` | purity | eff | `michel_found` | purity / eff |
|---|---|---|---|---|---|---|
| `h22base` | production (the doc-21 five) | 80/0/67/110 | **1.000** | 0.544 | 56/11/30 | 0.836 / 0.651 |
| `h22t` | the **10** live in `h21z` | 82/1/65/109 | 0.988 | 0.558 | 67/2/19 | 0.971 / 0.779 |
| **`h22g`** | the **full 16** | 82/1/65/109 | 0.988 | 0.558 | **68/2/18** | **0.971 / 0.791** |
| `h22w` | wide anchor only | 81/0/66/110 | **1.000** | 0.551 | 56/11/30 | unchanged |
| `h22c` | `h22g` + wide anchor | 83/**2**/64/108 | 0.976 | 0.565 | 68/2/18 | 0.971 / 0.791 |

**`h22t` reproduced `h21z` exactly** — 82/1/65/109, Michel 67/2/19, the same new FP
`029107_10/44`, the same 21 recovered items — **on a different binary**. That is the strongest
available confirmation of §1: ten keys alone reproduce what sixteen appeared to do, because six
were never read. The twin **held**.

**`h22g` vs `h22t` is the first PDHD measurement of the six late knobs**: the stopper side is
*identical*, and the Michel side gains exactly **+1 true Michel** (67→68) at no cost. Small, but
real, and free.

**Flipped: the full 16.** `michel_found` **56/11/30 → 68/2/18** — purity 0.836 → 0.971 *and*
efficiency 0.651 → 0.791, +12 true Michels and −9 false ones — for **one** stopper false
positive (`029107_10/44`), purity 1.000 → 0.988. That is exactly the trade the owner approved,
so the pre-registered flip rule is satisfied.

### The flip, and its gates

| gate | result |
|---|---|
| compiled-config proof, **no TLA**, vs the pre-edit file | **16 added, 0 removed, 0 changed** |
| flipped file's config vs `h22g`'s TLA config | **0 added, 0 removed, 0 changed** — the same config |
| other consumers of `wct-pr-perevt.jsonnet` | re-checked, not inherited: 5 other references are **comments citing line numbers**; no jsonnet `import`s it |
| **confirmation arm `h22conf`** — flipped file, **no TLA**, vs `h22g` | **341/341 bit-identical on all 146 shared branches; 0 `is_stm` flips; point geometry identical 341/341; 0 role moves** |
| `h22conf` census / Michel | identical: **82/1/65/109** and **68/2/18** |
| binary pin across all 9 arms | md5 `3e23bf8a` before and after, every arm |
| `wcdoctest-clus` | 396/396 cases, 23889 assertions, 0 failed |

**Production runs exactly what was measured** — that is what the confirmation arm establishes,
and it is why this flip is quotable rather than merely plausible.

## 5. The wide anchor: free alone, not free in combination

**`h22w` overturns doc pdhd/21 §5.** On a binary that implements the knob it recovers **+1
stopper (`028084_21/132`) at ZERO false positives**, purity 1.000 held. Less than PDVD's
+4-for-1-FP, but it costs nothing — and it is a *measurement*, where `h21w` was a key the binary
ignored. Its branch gate is the cleanest of the round: **340/341 bit-identical, point geometry
identical 341/341** — it moves a verdict bit and nothing else.

**But the combined arm `h22c` FAILED its pre-registered twin, and that decides it.** Predicted
83/**1**/64/109; actual **83/2/64/108, purity 0.976**. The TP side held *exactly* (83, the 22
named items) and the Michel side held *exactly* (68/2/18) — but a **second false positive,
`028084_3/72`, appeared that neither parent produces**: `h22g`'s only new FP is `029107_10/44`,
and `h22w` has none. It exists only in combination.

The falsifier named the mechanism in advance: `bragg_wide_anchor_cm` re-reads the Bragg peak in
a wider window, moving the rr origin and therefore the dQ/dx profile the stop-local and Michel
tests are built on. The branch gates show it: `h22c` flips 5 `is_stm` and moves geometry on
125/341, against 3 and 1 for its parents.

**So the wide anchor is HELD, not flipped.** Alone it is +1/0; added to the Michel bag its
*marginal* cost is **+1 TP / +1 FP**, taking purity to 0.976 — more than the owner approved, and
a purity trade nobody has ruled on. Same rule as P1: measure, report, recommend, leave unflipped.

This is the third arm in two rounds to show that **a knob set must be graded as a unit** — `h21f`
beat its twin, `h21z` failed its, `h22c` failed its. Assembling a combined result by adding
single-knob arms has now been wrong three times in a row.

## 6. P1: closed, measured, and **not** flipped — but read both columns

Every P1 number from round h21 was taken against the **pre-flip** baseline (86 missed stoppers).
The flip has since recovered 19 of those, so both the eligible pool and the null floor moved:
sizing on `h22base` gives 0.462 recoverable against a **0.174** null floor, against 0.448/0.136
before (`p1_sizing_postflip.txt`). Re-measured here on the real production baseline, `h22g`.

The twin used P1's **exact** admission test (`CheckSTM_Michel.cxx:4444`): `michel_found` ∧
`michel_conn_type ∈ {1,2}` ∧ `michel_ke_best ≥ topology_michel_ke_min` ∧ `michel_len ≥ 3 cm`,
clearing `no_bragg`/`shape_flat` after every other bit, so `is_stm` moves only 0→1. A
`michel_found`-only proxy predicts 23 recoveries where the truth is 12 — it is not used.

**The twin HELD EXACTLY on all three arms — 0 missing, 0 unpredicted, every new FP named.**

| arm | knobs | reading A `is_stm` | purity A | eff A | reading B | purity B | eff B |
|---|---|---|---|---|---|---|---|
| `h22g` | production | 82/1/65/109 | 0.988 | 0.558 | 60/1/10/58 | 0.984 | 0.857 |
| `h22p1` | `topology_stop_evidence` | 94/**3**/53/107 | 0.969 | **0.639** | 64/**1**/6/58 | **0.985** | **0.914** |
| `h22p2` | + `topology_clears_sparse` | 95/**5**/52/105 | 0.950 | 0.646 | 64/1/6/58 | 0.985 | 0.914 |
| `h22p3` | + `topology_michel_ke_min:3` | 99/**6**/48/104 | 0.943 | 0.673 | 64/**2**/6/57 | 0.970 | 0.914 |

Reading A is `d21_grade.py`'s (`owner_review` > `owner_smx1` > agent). Reading B keeps an agent
verdict only when its own `confidence` field is `high`, excluding the rest rather than guessing —
44 of 317 records are owner-adjudicated; B scores 129 items, A scores 257.

**The two readings disagree, and that disagreement is the finding.** Under A, P1 costs two extra
false positives. Under B it costs **nothing** — purity 0.984 → 0.985, unchanged within one item —
while efficiency rises 0.857 → 0.914. Both of A's extra FPs (`028084_12/46`, `028084_3/72`) are
agent-only items the scanner itself did **not** mark high confidence. So "does P1 cost purity"
reduces to "do you trust a medium-confidence agent verdict against the chain" — the owner's call.

**Not flipped**, by the rule fixed before the numbers were known: nonzero cost under reading A →
measure, report both readings, recommend, leave unflipped. `michel_found` is unchanged at 68/2/18
on all three (P1 reads the Michel object, never writes one), and point geometry is identical
341/341 for `h22p1`/`h22p2` — P1 moves verdict bits only.

**Recommendation, if the owner takes it: `h22p1` — P1 alone at the C++ default floors.**
`topology_clears_sparse` buys *nothing* under reading B (identical to `h22p1`) while costing two
more FPs under A; `topology_michel_ke_min:3` costs under **both**, and additionally flips 5
UNCLEAR and 1 MESSY item — pushing the chain into a population the scanners could not judge.

## 7. What is NOT concluded

* **Not** that the flipped chain is optimal. It is the zero-cost set (doc 21) plus the one
  purity-costing trade the owner explicitly approved.
* **Not** attributed *within* the Michel bag. Sixteen keys ran as a unit. `h22g` vs `h22t`
  attributes only the **six late knobs as a block** (+1 true Michel, free); which *individual*
  Michel knob does the work is unknown, and round h21's `h21s` showed ~2 of the gain belongs to
  the stop-topology three. Attribution would cost ~13 arms and was not asked for.
* **Not** the wide anchor. Free alone (+1/0), **+1 TP / +1 FP marginal** in combination — held
  for the owner. Its retraction of doc 21 §5 stands regardless of that ruling.
* **Not** P1. Held, with both truth readings reported because they disagree.
* **Not** a claim that either truth reading is the correct one. Reading B is not "the truth" —
  it is the subset where the owner adjudicated or the scanner was confident. Both are reported
  precisely because the P1 decision turns on which one the owner trusts.
* **Not** a re-judged record. `smx22` as it stands, per the owner's 2026-09-11 ruling.
* **Not** a reopening of the plateau window. Doc pdhd/21 §5's negative stands — `plateau_mip_hi`
  **did** exist in the `082376c5` pin, so unlike the wide anchor that measurement was real.
* **Not** free of chain movement. `h22g` is only **219/341** bit-identical with 3 `is_stm` flips
  and moves point geometry on **125/341** candidates — expected for gamma collection, which adds
  points to the Michel object, but it is not a pure relabelling.
* **Not** a statement about any binary before `d65f8165`. Six of the flipped keys do not exist
  in older builds and would be silently ignored (§1).

## 8. What is left, ranked

1. **The owner's ruling on P1** (`h22p1`): +12 stoppers, efficiency 0.558 → 0.639, for 2 false
   positives under reading A and **none** under reading B. The largest gain still on the table,
   and the decision reduces to whether a medium-confidence agent verdict outranks the chain.
2. **The owner's ruling on the wide anchor**: +1 stopper for +1 FP *in combination* — small, and
   the least attractive of the open trades, but it is measured and costed.
3. **Attribute the Michel bag** knob by knob, if it is ever worth ~13 arms. The six late knobs
   are worth +1 Michel *as a block*; nobody knows which one.
4. **Re-check `028084_3/72` and `029107_10/44`** — the two false positives this round buys. Both
   are agent-only, neither owner-adjudicated. A hand look would convert an operating-point
   argument into a factual one, and `028084_3/72` is implicated in both `h22c` and P1.
5. The 31-knob PDVD inventory is now graded out: flipped, measured-dead, or held with a number.

## Files

All under `pdhd/docs/scan/h22/` unless stated.

| what | where |
|---|---|
| pre-registered twins, each written before its arm | `preregistered_twin.txt` |
| the binary gate (§3) | `census_h22base.txt`, `g_h22base.txt` |
| the Michel bag + wide anchor (§4, §5) | `census_h22tgw.txt`, `g_h22t.txt`, `g_h22g.txt`, `g_h22w.txt` |
| the combined arm whose twin failed (§5) | `census_h22c.txt`, `g_h22c.txt` |
| compiled-config proofs, all arms + negative control | `cfg_proofs.txt` |
| compiled-config checker — parses structure, self-tested | `d22_cfg_keys.py` |
| what is left on the post-flip baseline, and P1's pools | `d22_bits.py`, `p1_sizing_postflip.txt` |
| P1's offline twin, using P1's exact admission test | `d22_p1_twin.py` |
| P1 measured (§6) | `census_h22p.txt`, `g_h22p1.txt`, `g_h22p2.txt`, `g_h22p3.txt` |
| both truth readings, and its instrument (§6) | `census_truth_readings.txt`, `d22_truth2.py` |
| the confirmation arm (§4) | `census_h22conf.txt`, `g_h22conf.txt` |
| graders, reused unchanged from round h21 | `../h21/d21_grade.py`, `../h21/d21_michel_census.py` |
| the flip | `pdhd/wct-pr-perevt.jsonnet` `stm_michel_knobs` |

**Binary provenance.** Every arm in this round ran `/home/xqian/tmp/p65/libpin_p65` —
toolkit `d65f8165`, `libWireCellClus.so` md5 `3e23bf8a`, verified identical before and after
every arm. Round h21's arms ran `libpin_p82` (`082376c5`, md5 `5f2c3ede`), which is why §1
exists. `wcdoctest-clus`: 396/396 cases, 23889 assertions, 0 failed.

**Records untouched**, re-verified end to end: `smx1 2d9c2351` · `smx18 32aa55c2` ·
`smx19 be4527e9` · `smx20 59a4de40` · `smx21 8953b683` · `own19 6759c7dd` · `own20 f725e5ba`,
and the `smx22` verdicts file `a36de425`.
