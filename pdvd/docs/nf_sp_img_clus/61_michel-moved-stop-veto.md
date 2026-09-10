# 61 — T2: Michel admission at a moved stop, plus doc 56 §9 fixes

**Status (2026-09-09). One knob shipped and flipped to PDVD production:
`moved_stop_michel_guard`. T2a and T2b are measured negatives — no knob for
either. §9 items 1, 4, 5 fixed; item 2 reaffirmed as explain-don't-fix.**

---

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
wcbuild && ./build/clus/wcdoctest-clus            # 357/357

cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img
python3 pdhd/stm_michel_scan/census_score.py --check     # "0 of 14 differ"

# a new arm: prep its payloads into scratch first (never into prep-pdvd),
# then score.  NOTE (doc 56 sec 9 item 5): prep_stm_michel_scan.py REFUSES to
# re-draw tranche 1 once a work/stm_michel_labels/*/labels.json exists for the
# detector -- --pin-tranche is required in that case, not optional:
W=$HOME/tmp/d61
./pdhd/stm_michel_scan/prep_stm_michel_scan.py --det pdvd --arm <tag> \
    --outdir $W/prep_<tag> --sheetdir $W/sheet_<tag> \
    --pin-tranche pdvd/docs/scan/pdvd_stm_michel_scan_sheet.tsv
python3 pdhd/stm_michel_scan/census_score.py --prep $W/prep_<tag> --baseline $W/prep_d59v --arm <tag>
```

Producing a new arm is the doc-53 recipe with the binary pinned:

```bash
ARM=<tag> DET=pdvd SRC=d16vnu JOBS=8 PIN=<snapshot> \
    PR_TLA='-S stm_michel_extra={moved_stop_michel_guard:true}' \
    pdvd/docs/nf_sp_img_clus/scripts/d53_run_arms.sh
```

---

## 1. Where T2 landed: one knob, not three

Doc 56's T2 row asked three questions: (a) does Michel admission need a
charge/KE test rather than median dQ/dx (the 8 `michel_mip_lo`-failing arms);
(b) do the 7 "passes everything, still not admitted" arms hang off a vertex
other than `stop_v`; (c) does a mechanism that MOVES the stop (T1a's retreat,
T1c's split) need a stronger Michel-admission gate than a stop the tagger
itself found, given that docs 57/58/59 named 6 spurious `michel_found`
attachments this exact way. **(c) is confirmed and shipped as
`moved_stop_michel_guard`. (a) and (b) are measured this session and come
back negative — no C++, no knob, for either.**

---

## 2. T2a: charge/KE on the "mip_lo fails" and "passes everything" arms — measured negative

`segment_integrated_dQ` (`PRSegmentFunctions.h:263`) is the right primitive —
graph-free, sums `fit.dQ` over valid points — but nothing converts it to KE
without an `IRecombinationModel`, which `StmMichelArmThresholds` does not
carry. Rather than build that plumbing speculatively, this session measured
what it would find on the 9 arms doc 56 §3 named
(`039349_32/63`, `039349_7/4`, `039349_82/54`, `039253_3/61`, `039252_16/98`,
`039253_17/77`, `039349_22/63`, `039252_0/75`, `039349_11/19`):

| item | seg len (cm) | mip (÷47000) | any nearby unfitted charge (`n_dots`/`dots_ke_*`) |
|---|---:|---:|---|
| every one of the 9 | 4.0–9.0 | 0.10–0.96 | **0 on all 9** |

None of the 9 has any unfitted companion charge cluster nearby — `n_dots`,
`dots_ke_dqdx`, `dots_ke_unfit` are all zero. A rough total-charge estimate
(length × median dQ/dx, before recombination) puts these arms at 1.3–9.9 MeV
— the same order the existing `michel_mip_lo` gate already reads on its low
end, not clearly above it. **On the arms the fit actually reconstructs, the
charge these arms carry is genuinely small.** The scan attributed several of
them on the 3-D picture (doc 56 §3), which is consistent with doc 55 §15.2's
own mechanism: a Michel dilutes the very Bragg rise the chain measures, so
its own charge measurement at the fit is diminished too. This is a T3-shaped
problem (residual/charge outside the terminal floor) more than a T2
admission-threshold one — T3's own scope already names this population, and
no code changes here.

---

## 3. T2b: the "passes everything" arms — measured negative, population already moved

Traced the same 9 arms against the **current** (`d59v`) arm rather than doc
56's stale `d53v` list, since T1a/T1b/T1c have since recovered some of the
items this population was named against:

* **`039253_17/77` no longer applies**: T1b's asymmetric kink already
  recovers `michel_found=1` for it (doc 59 §5) — it is not in the "still not
  admitted" set on `d59v`.
* **mip readings on `d59v`** (using the driver's real `mip_dqdx_median =
  47000`, not `census_lib`'s stale `MIP_MEDIAN` constant of 54000): 0.10–0.96
  — most fail `michel_mip_lo` (0.3) or `michel_mip_hi` (2.0) cleanly, not
  narrowly.
* **`n_stop_arms` is 1 on 8 of the remaining 8** — there is no ALTERNATE
  vertex at the stop for these arms to hang off. The "hanging off a vertex
  other than `stop_v`" hypothesis doc 56 §3 raised does not hold on this
  population, checked directly against `pf.chain_role`/`pf.vtx`, not assumed.

The overall `n_stop_arms` distribution over the 566 `d59v` items: 378 with 0,
175 with 1, 11 with 2, 2 with 3 — and of the 33 census `michel_found` FNs, 19
have at least one stop arm (i.e. an arm existed and still failed a gate),
consistent with most rejections being a clean threshold failure rather than
a wrong-vertex problem. **This is a one-paragraph negative finding, not a
task**: the "passes everything" framing in doc 56 was itself measured against
a population that has since moved out from under it as T1a/T1b/T1c recovered
items; on the current arm there is no clean residual population left to
design a T2b knob against. No C++ for T2b this round.

---

## 4. T2c: veto a moved-stop attachment when its charge cannot support it

### 4.1 The mechanism

`CheckSTM_Michel.cxx`, immediately after `rec.michel_ke_best` is finalized
(`michel_ke_dqdx + dots_ke_unfit`) and before `rec.michel_found` is derived:

```cpp
if (m_moved_stop_michel_guard && rec.michel_conn_type == 1 &&
    (rec.n_retreat > 0 || rec.n_split > 0) &&
    rec.michel_ke_best < m_moved_stop_michel_ke_min) {
    rec.michel_conn_type = 0;
    ++rec.n_michel_veto;
}
```

`michel_conn_type == 1` means the arm is attached directly to `stop_v`
(`michel_dis_cm` is a literal `0` assignment in that same code path, not a
measurement, so it is not part of the condition). `n_retreat`/`n_split` are
already set upstream in the same `Record` when a stop-move mechanism fired
this event — both are non-zero only when T1a's retreat or T1c's split found
a stop the tagger's own fit missed, and the two are mutually exclusive (the
split only runs when the retreat did not fire). The demotion is via
`michel_conn_type`, which `michel_found = (michel_conn_type > 0) ? 1 : 0`
reads next — **deliberately not via `reject_bits`**, so `is_stm` is
untouched. A new `n_michel_veto` counter records that the mechanism fired
even on items where nothing else visibly changes.

Default OFF: `m_moved_stop_michel_guard{false}`,
`m_moved_stop_michel_ke_min{10.0}` (doc 55 §15.1's own KE floor, re-used here
for a narrower population — that cut targets `conn_type==2, dis_cm>3`; this
targets `conn_type==1, dis_cm==0`, no overlap).

### 4.2 Config threading

No jsonnet edit was needed to run the feature arm: `stm_michel_extra`
already merges on top of the driver's `stm_michel_knobs` bag (doc pdvd/51),
so `-S stm_michel_extra={moved_stop_michel_guard:true}` was the whole arm
override. The production flip (§4.5) adds one key to the bag itself.

### 4.3 Tests

`clus/test/doctest_check_stm_michel_defaults.cxx`:
`moved_stop_michel_guard == false`, `moved_stop_michel_ke_min == 10.0`. No
logic-level doctest for the veto itself, consistent with every other guard
in this file (`vertex_kink_guard`, `stop_retreat_max`, …) — none has one
either; the byte-identical OFF-path gate plus the real-arm census decides
this task, same as T1a/T1b/T1c.

### 4.4 Byte-identical gates

`d61vleg` (knob off) vs **`d59v`** (current PDVD production candidate —
T1a+T1b+T1c all on, NOT `d53v`); `d61hleg` vs `d53h` (PDHD, knob stays off
there permanently — no PDHD hand-scan record):

```
d61_pdvd_off  (T_stm_michel)
  matched candidates      : 578  (before 578, after 578)
  branches shared / new / dropped : 116 / 1 / 0
  NEW branches   : n_michel_veto
  candidates BIT-IDENTICAL on all 116 shared branches : 578 / 578
  is_stm FLIPS : 0

d61_pdhd_off  (T_stm_michel)
  matched candidates      : 325  (before 325, after 325)
  branches shared / new / dropped : 111 / 6 / 0
  NEW branches   : n_michel_veto, n_retreat, n_split, retreat_len,
                    split_kink_deg, split_len   (the 5 PDHD-side "new"
                    branches are T1a/T1c's own fields, expected since
                    d53h predates both -- not this round's change)
  candidates BIT-IDENTICAL on all 111 shared branches : 325 / 325
  is_stm FLIPS : 0
```

Both PASS. SBND has zero exposure: it does not bind `CheckSTM_Michel` at all
(confirmed, no `check_stm_michel` call anywhere under `cfg/.../sbnd/`).

### 4.5 The real arm: exactly the predicted 6 items, nothing else

`d61v` = `d59v`'s config + `stm_michel_extra={moved_stop_michel_guard:true}`,
scored against `d59v`:

|  | `is_stm` TP | FP | FN | TN | purity | efficiency |
|---|---:|---:|---:|---:|---:|---:|
| `d59v` (before) | 149 | 9 | 119 | 270 | 0.943 | 0.556 |
| `d61v` (after)  | **149** | **9** | **119** | **270** | **0.943** | **0.556** |

`is_stm` is **bit-identical on all 566 common items** — checked by name, not
assumed (the veto never touches `reject_bits`, so this is the expected, and
now confirmed, outcome).

|  | `michel_found` TP | FP | FN | TN | purity | efficiency | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `d59v` (before) | 119 | 44 | 33 | 351 | 0.730 | 0.783 | 0.755 |
| `d61v` (after)  | **118** | **39** | **34** | **356** | **0.752** | **0.776** | **0.764** |

**The veto fires on exactly 6 items — the same 6 the offline census-payload
estimate predicted before this arm ran, by name:**

| item | scan verdict | `n_retreat`/`n_split` | `michel_ke_best` (MeV) | `michel_found` before → after |
|---|---|---|---:|---|
| `039252_2/79` | THRU | split=1 | 8.92 | 1 → 0 |
| `039252_4/55` | THRU | split=1 | 4.62 | 1 → 0 |
| `039349_20/41` | THRU | retreat=1 | 5.73 | 1 → 0 |
| `039349_48/21` | THRU | retreat=2 | 8.67 | 1 → 0 |
| `039349_61/62` | THRU | split=1 | 7.97 | 1 → 0 |
| `039349_36/63` | **STM_MICHEL** | retreat=1 | **6.61** | 1 → 0 |

**5 named spurious attachments removed, exactly the 5 docs 57/58/59 flagged
as feeding T2 — 0 new false positives, 0 new true positives.** One real cost,
also named: `039349_36/63`, a genuine scan-confirmed Michel, loses its
correct attachment. **No threshold cleanly separates the two populations**:
two of the recovered FPs (`039252_2/79` at 8.92 MeV, `039349_48/21` at 8.67
MeV) sit ABOVE the lost TP's 6.61 MeV, so a lower floor that spares
`039349_36/63` would also spare the two highest-KE false positives — worse,
not better. 10.0 MeV is not an arbitrary choice between two clean clusters;
it is the operating point doc 55 §15.1 already established for the general
population, applied here to a narrower one.

**Net: F1 0.755 → 0.764.** Purity gains from 5 fewer false positives outweigh
the loss of 1 true positive, on a base of 119 TP / 44 FP. Confirmed on the
real arm, not offline-predicted.

### 4.6 Production flip

Same bar as T1a/T1b/T1c: `is_stm` bit-identical (confirmed, 0 of 566 flips);
every gain and every loss named (5 removed, 1 lost, 0 new either way);
net-positive F1. **Flipped**: `moved_stop_michel_guard: true` added to
`pdvd/wct-pr-perevt.jsonnet`'s `stm_michel_knobs` bag.
`moved_stop_michel_ke_min` stays **unset** at its C++ default (10.0, the
value the scored arm ran) — pinning it explicitly, even at the same numeric
value, makes it an inert-but-present key that survives an override forcing
`moved_stop_michel_guard` back to `false`, breaking the OFF-path
byte-identity check. This is the exact trap doc pdvd/58 §6 already
documented and fixed the same way for `split_kink_min_deg`; caught and
corrected before landing, not after.

Verified, not assumed:
* **Flip-equivalence** (production bag + no override == default bag + the
  arm's `stm_michel_extra` override): compiled-JSON diff, **0 lines**.
* **True OFF-path byte-identity**: the SAME override
  (`stm_michel_extra={moved_stop_michel_guard:false}`) applied to the
  PRE-round file (`git show HEAD:pdvd/wct-pr-perevt.jsonnet`, before this
  session's edit) and to the POST-round file (after the flip), compiled
  JSON diff, **0 lines**.

PDHD stays OFF: no PDHD STM/Michel hand-scan record exists to confirm it
there.

---

## 5. §9 fixes

### 5.1 Item 1 — `TrackFitting.cxx`'s pass-3 literal

Doc 56 §9 item 1 itself had two factual errors (confirmed this session):
line numbers were ~12 off, and the cited `:10076` named the wrong function —
`do_single_tracking` has **no** hard-coded step at all; both its passes
already derive from `m_params.low_dis_limit`. The real site is
`do_multi_tracking`, pass 3, `TrackFitting.cxx:9714`:

```cpp
low_dis_limit = 0.6*units::cm;                       // before
low_dis_limit = m_params.low_dis_limit/2.;           // after (reuses pass 2's own expression, :9536)
```

**Proven byte-identical in IEEE double, not just matched in value**:
`0.6*units::cm == 6.0` and `12.0/2. == 6.0` bit-for-bit, and `low_dis_limit`
is `12.0` mm on PDVD, PDHD, **and** SBND (every `*_track_fitting.json` in the
tree) and equals the C++ default (so uBooNE, which sets none of these keys,
inherits it too) — there is no configuration in the tree today where this
edit changes the computed value.

**Touched line 9714 only** — `end_point_limit` nearby is halved at `:9537`
and never reset before `dQ_dx_multi_fit` consumes it at `:9890`; a
"symmetric" edit there would be a real behavior change, not a no-op.

Also fixed the stale comment this drift produced: `TrackFitting.h:246-248`
(the FINAL `organize_ps_path`'s gap threshold reads "1.6 × low_dis_limit"
but that `low_dis_limit` is already the pass-2-halved local, so at the
shipped default the threshold is 1.6 × 0.6 cm = 0.96 cm, not 1.92 cm — the
comment now says so explicitly) and the matching
`_comment_traj_final_fill_charge_test` strings in
`pdvd_track_fitting.json` and `pdhd_track_fitting.json` (both routinely
co-edited throughout this campaign, not "another experiment").
`sbnd_track_fitting.json`'s copy of the same stale comment is **explicitly
left untouched** — SBND is a different experiment this doc never otherwise
touches, out of scope per CLAUDE.md §5.3 even for a comment-only fix; noted
in §6 below, not fixed.

**Gate — the general PatternAlgorithms path, not the STM arms.**
`do_multi_tracking` has 40 call sites across `NeutrinoStructureExaminer`,
`NeutrinoVertexFinder`, `NeutrinoPatternBase`, `NeutrinoGraphAudit`,
`NeutrinoOtherSegments` — none of the STM/Michel arms (`d61v`/`d61vleg`/
`d61hleg`) exercise it, since `check_stm_michel`'s own fit path is
`do_single_tracking`, which has no literal to fix. The exercising pipeline
is the **legacy** `-nu-legacy` chain (`tagger_check_neutrino`,
pre-doc-48), run directly for this gate on 3 PDVD events (`039252_0`,
`039252_10`, `039252_12`), pre- and post-fix, same pinned-binary discipline:

| check | result |
|---|---|
| `T_rec_charge` (the direct fitted-track product), all 19 branches, 3 events | **bit-identical**, before vs after |
| `mabc-pr.zip` content hash (`hash_archive.py`, timestamp-insensitive) | **identical**, before vs after |
| `T_tagger`/`T_kine`/`T_proj_data` (the legacy neutrino-tail trees) | **bit-identical**, before vs after — *corrected by doc 69; this row first read "differ, before vs after"* |

> **Correction (doc 69, 2026-09-10).** The third row and the paragraph
> below it are wrong. There is no non-determinism in the legacy tail. The
> saved control dirs (`pdvd/work/039252_{0,10,12}_d61item1{a,b}`,
> `039252_0_d61item1a2`) are bit-identical on every branch of all 8 trees
> in `tracking-pr.root`, in all 4 pairs: same binary twice, and before vs
> after on each event. The "differences" came from the inline check used
> here: `arrays(library='np')` then `np.array_equal` under
> `except: ok = False`. With `library='np'`, a `vector<>` branch comes back
> as an object array of arrays, and that check marks every such branch
> "differ" whatever it contains. That is why the same-binary control showed
> the same kind and scope of "differences", and why the flat-only
> `T_rec_charge` "passed". The item-1 fix is therefore byte-identical on
> **all** legacy trees, not only on `T_rec_charge`. M4 does not describe
> this tail. Evidence and replay: doc 69 §2
> (`scripts/d69_full_output_compare.py --replay-d61`). The original
> paragraph is kept below for the record.
>
> ~~The third row looked like a failure until checked against a control:
> running the **same, unmodified** (pre-fix) binary **twice** on the same
> event produces the **same kind and scope** of
> `T_tagger`/`T_kine`/`T_proj_data` differences — this is **pre-existing
> non-determinism in the deprecated legacy neutrino tail**, not something
> this fix introduces (CLAUDE.md's M4 already documents this class of tail
> as not bit-stable).~~

`T_rec_charge` is the direct product of `TrackFitting`/`PR::Fit` and is the
meaningful gate for this specific edit; it is bit-identical on all 3 events,
carrying real non-trivial data (2252 fitted points on `039252_0` alone, not a
vacuous match).

### 5.2 Item 4 — `census_score.py`'s F/H classes tracked by name

Doc 56's own diagnosis was imprecise: `census_score.py`'s F/H classes were
already id-free (re-derived from segment geometry, no `detail` string
dependency at all — that dependency lives in the separate `mkfailures.py`/
`pdvd_stm_michel_failures.tsv` register, which this fix does not touch). The
real gap: F/H were a bare per-arm **recount**, so a genuine behavior change
and a re-segmentation artifact were indistinguishable in the printed number.

Fixed by factoring the F/H thresholds into a shared `flagged_ids(pay, cls)`
helper (used by both the main class loop and the new tracking section, so
the two definitions cannot drift apart) and adding a new **§B2** that
matches each `--baseline`-flagged segment into the new arm by geometry
(reusing `match_segment`, the same primitive the Michel-attachment metric
already uses) and reports **kept / lost / new by name**:

```
=== B2. F/H tracked by NAME (baseline-flagged segment matched by geometry into this arm) ===
  F_fit_stops_short          baseline   5  kept   5  lost   0  new   0
  H_fit_unsupported_by_charge baseline  20  kept  20  lost   0  new   0
```

Sequenced **before** any T2c arm was scored (the same instrument grades
both this fix and T2c's FP recovery — it had to be settled first, not
in flux while grading). Gate: `--check` still **0 of 14 differ**, and the
identity case (`--baseline` defaulting to `--prep`) shows every baseline
item correctly "kept" — the fix changes what is reported, not the counts.
Re-verified across two genuinely different real arms (`d59v` as baseline,
`d61v` as prep): same result, F/H both fully kept, 0 lost, 0 new — as
expected, since T2c's veto never touches segment geometry.

`prep_stm_michel_scan.py`'s `VERDICT_SCALARS` also needed `n_michel_veto`
added (the same absent-on-older-arm guard `n_retreat`/`n_split` already use)
— without it the new field existed in the ROOT tree but was silently
dropped at prep time, caught when the veto trace initially showed
`n_michel_veto=None` on items the byte-identical gate had already proven
fired.

### 5.3 Item 5 — the `--pin-tranche` repro note

One-line fix: doc 56 §0's repro block, and this doc's own §0 above, now
state that `prep_stm_michel_scan.py` refuses to re-draw tranche 1 once any
`work/stm_michel_labels/*/labels.json` exists, and `--pin-tranche <sheet>`
is mandatory in that case — matching what every round's actual repro
command already does, just not previously documented as a requirement.

### 5.4 Item 2 — reaffirmed, not fixed

`vertex_kink_reject` (`TaggerCheckSTM.cxx:2474`) is inert on PDVD
(`m_vertex_kink_guard` is `false` there) and live only on SBND, which has no
STM/Michel hand-scan record to confirm a fix against. T1b's real kink
already corrects the common case wherever its new clause fires
(`m_vertex_kink_guard` untouched). No SBND-blind change proposed this round
either.

---

## 6. Found on the way, not fixed

1. `sbnd_track_fitting.json`'s copy of the stale
   `_comment_traj_final_fill_charge_test` gap-threshold description (item
   1's fix) is left untouched — SBND is a different experiment this doc
   never otherwise touches; a comment-only fix there is still out of scope
   per CLAUDE.md §5.3 without the owner's say-so.
2. **Withdrawn (doc 69, 2026-09-10).** ~~The legacy `-nu-legacy` neutrino
   tail (`T_tagger`/`T_kine`/`T_proj_data`) is confirmed non-bit-stable
   run-to-run on the SAME unmodified binary — a pre-existing condition
   (CLAUDE.md M4), not introduced here, and out of scope for this doc's own
   task set.~~ The saved same-binary pair is bit-identical on every tree. The
   "non-determinism" was the §5.1 comparator marking every `vector<>` branch
   as different (doc 69 §2).
3. T2a's measurement (§2) hands T3 a sharper starting point than "T2a
   failed": the charge these 9 arms carry is genuinely small at the fit,
   consistent with doc 55 §15.2's dilution mechanism — the natural next
   question is whether a residual/charge test outside the terminal floor
   (T3's own scope) can recover any of them, not whether the admission
   gate's threshold is wrong.

---

## 7. Gates

| gate | result |
|---|---|
| `./build/clus/wcdoctest-clus` | 357/357 |
| `d61vleg` vs `d59v` (PDVD), `d61hleg` vs `d53h` (PDHD) | both PASS, 0 `is_stm` flips |
| compiled-config grep / diff | OFF-path 0 occurrences; ON-path exactly 1 key diff |
| flip-equivalence (prod bag == arm override) | 0-line diff |
| true OFF-path (pre-round file == post-round file, same forced-off override) | 0-line diff |
| `census_score.py --prep <scratch> --arm d61v` vs `d59v`, by name | 5 FP removed, 1 TP lost, 0 new either way, `is_stm` bit-identical (0 of 566) |
| §9 item 1: `T_rec_charge` before/after, 3 events | bit-identical. Re-compared by doc 69: **all 8 legacy trees** bit-identical, before/after and same-binary-twice. ~~legacy-tail noise shown to be pre-existing via same-binary-twice control~~ was a comparator artifact |
| §9 item 4: `census_score.py --check` | still 0 of 14 differ |
| `census_score.py --check` | 0 of 14 differ |

---

## 8. Update to doc 56

Doc 56's T2 row is marked done with the three-way split: T2c shipped and
flipped to PDVD production; T2a and T2b are measured negatives, each with
their evidence. §9 items 1, 4, 5 marked fixed; item 2 reaffirmed as
explain-don't-fix. The "Order" paragraph now points at T3 next — the
natural continuation, since T2a's own measurement (dilution at the fit,
not admission-threshold) is exactly T3's residual/charge question.
