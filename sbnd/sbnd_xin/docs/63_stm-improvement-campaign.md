# 63 — STM tagger improvement campaign (against the doc-62 owner baseline)

**Status.** Round 1 SHIPPED (record in §4): **6 of the 15 false STMs fixed,
zero regressions** on the 72-bundle owner baseline, knob-off byte-identical.
Round 2 is next.  Each round is a default-OFF knob in `TaggerCheckSTM.cxx`,
evaluated on the full baseline before it is committed.  Only the STM tagger
is touched — TGM, LM and FC are out of scope by the owner's instruction.

**The target.** `scan-d59k/stm-baseline.tsv` (owner verdicts = truth):

| | n | goal |
|---|---|---|
| code-FALSE-STM (tagger STM, owner not-STM) | 15 | fix as many as possible |
| code-MISSED-STM (62613:17) | 1 | recover |
| code-STM-correct | 36 | **keep every one** |
| code-not-STM-correct | 20 | **keep every one** |

A round is an improvement only if it fixes ≥ 1 error and regresses **0** of the
56 correct verdicts.  The owner's acceptance standard applies throughout: a
Michel electron or a proton at the stopping end does NOT disqualify an STM tag
(doc 62 §4), so no round may reject on "the residual is not empty" alone.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# diagnostics extracted from the STORED d59k production (read-only, M13):
python3 stm_campaign/extract_stm_diag.py --out /home/xqian/tmp/stm-campaign/diag
# one evaluation round (see stm_campaign/run_round.sh -h):
./stm_campaign/run_round.sh r0                  # reference arm, knobs off
./stm_campaign/run_round.sh r1 -stm-guards     # round-1 arm
python3 stm_campaign/score_round.py --round work-stmcamp-r1 --ref work-stmcamp-r0
```

The production flag set is `run_full1k_nusel.sh`'s `NUF` verbatim
(`-chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm -main-pair-real -fvx 2.5
-fvy 3 -stm-fit -mip 56000 -unmerge-assoc`); rounds append their knob flag.
Work roots `work-stmcamp-<round>/` symlink imaging (`evt<ID>`) and the Q/L
pctree (`ql_evt<ID>`) from `work-mcp1kall-d59k` — the Q/L step is never re-run,
so a round re-executes exactly the PR tagger tail (seconds per event).  Nothing
under `work-mcp1kall-d59k/` is written.

## 1. Where every error comes from (diagnosis, from stored d59k fits)

Sources: the `stm_pass`/`stm_eval`/`stm_fit` PCs that the `-stm-fit` production
already persisted in every `pctree-pr-evt<ID>.tar.gz`, plus the tagger logs.
Extractor: `stm_campaign/extract_stm_diag.py`.  All 72 baseline bundles reached
the round-2 fit (no pre-fit exits).  Statuses: the 15 false STMs were all
**accepted (status 0)**; the miss 62613:17 was killed by **detect_proton
(status 5)**; the 20 correct non-tags exited with status 2 (long leftover),
3 (dQ/dx eval), 4 (extra tracks) or 5 (proton).

Key measurements (all offline, on the stored per-pass fits; MIP = 56000 e/cm):

1. **Charge desert under the fit** — longest contiguous stretch of the fitted
   muon segment with per-point dQ/dx < 0.02 MIP:
   278794:7 = **23.4 cm**, 285443:7 = **6.7 cm**, every other accepted bundle
   (false or correct) = **0.0 cm**.  The fit bridged detached objects through
   empty space — the owner's one-objectness ruling, measurable.  A ≥ 3 cm veto
   separates with infinite margin on this set.

2. **Normalization hole in the eval acceptance** — the first acceptance branch
   (`ks1-ks2 < -0.02 && ...`) never checks that the measured charge resembles
   a muon in NORMALIZATION.  285443:7 was accepted with ratio2 = **3.96**
   (measured charge 4× BELOW the flat-MIP reference — a ghost-mess, not a
   muon).  The highest CORRECT accepting call is ratio2 = **1.48** (283463:14,
   a sub-MIP track the owner accepts — a first-try cap at 1.1 killed it; see
   the round-1 record).  The shipped cap is **2.0**: no under-collected muon
   loses half its charge, a ghost does.

3. **Spike-not-ramp at the stop** — a real Bragg is a RAMP: the muon table
   gives ~1.7 MIP at rr = 3 cm rising to ~3 at rr = 0.5 cm, so the 1.5–4 cm
   "shoulder" before the stop is elevated.  A ν vertex is a SPIKE: MIP-flat
   approach, then one or two very high points.  peak(< 1.5 cm)/median(2–6 cm)
   and the shoulder median separate:

   | bundle (owner) | spike | shoulder (MIP) |
   |---|---|---|
   | 290294:12 (vertex) | 7.0 | 0.66 |
   | 409546:8 (vertex) | 7.4 | 0.81 |
   | 285366:13 (vertex) | 5.3 | 1.69 |
   | 289832:10 (vertex) | 3.6 | 0.84 |
   | worst correct STM: 401824:4 | 3.1 | 1.87 |
   | next: 409458:8 | 2.7 | 1.38 |

   Veto `spike > 3.2 && shoulder < 1.8` catches the four vertex cases and
   protects every correct STM on BOTH conditions (401824 fails both).
   278662:1 ("could be a vertex activity", spike 2.7/shoulder 0.97) is inside
   the correct population's spike range — left for a later round.

4. **The miss** — detect_proton killed 62613:17 through branch
   `dQ_dx[max_bin] > 4.3 MIP && ks3 > 0.03` by margins of 0.0007 (4.3007) and
   0.003 (ks3 = 0.033), while the muon hypothesis fits the end region almost
   perfectly: **ks1 = 0.030, ratio1 = 0.94, ratio3 = 1.06**.  The two proton
   vetoes the owner CONFIRMS (288859:9, 319809:20): 288859 fired through the
   delta-ray path (untouched), 319809 through branch B1 with ks1 = 0.063 and
   ratio3 = 1.22.  A muon-consistency guard `ks1 < 0.045 && ratio3 < 1.1 ⇒
   not a proton` recovers the miss and keeps both correct rejections, with
   ~2× margins on both axes.

5. **What does NOT separate** (measured, so nobody retries it):
   - The flat-track false STMs (48895, 392200, 321107, 321371, 72586) are
     indistinguishable in dQ/dx from owner-ACCEPTED flat STMs (68956, 282715,
     315409, 175428, 172596…): both run ~1 MIP into the stop, same KS values,
     same residual lengths (the accepting comb values interleave: false
     −0.012..−0.053 vs correct −0.001..−0.105).  The owner separated them on
     the DISPLAY (topology/context), not the dQ/dx panel.
   - Off-track steiner charge within 8 cm of the stop: vertex cases often show
     **0** (the prongs live in other clusters of the bundle), while correct
     STMs show up to 33 off-track points (Michels, deltas).  Prong-counting on
     the main cluster's own steiner cloud does not work.

## 2. The round plan

- **Round 1 — `stm_accept_guards` (C++ `accept_guards`, default false).**
  Three pass-level acceptance guards from §1.1–1.3: charge-desert ≥ 3 cm,
  eval-acceptance ratio2 > 2.0, spike > 3.2 with shoulder < 1.8 MIP.
  Expected: fixes 6 of 15 false STMs (278794, 285443, 289832, 285366, 290294,
  409546) — 4 of the 5 vertex-as-Bragg, 2 of the 6 multi-object — with zero
  expected regressions.  The desert and spike guards are computed once per
  pass from the final fit (call-independent); the ratio2 cap is inside
  eval_stm_core (the offline kill-list is conservative for it, since a killed
  call lets the || chain try later calls the stored records never executed —
  the re-run decides).

- **Round 2 — `stm_proton_muon_guard` (C++ `proton_muon_guard`, default
  false).**  §1.4: in detect_proton's end-proton branches, a candidate whose
  end region matches the muon hypothesis (ks1 < 0.045 && ratio3 < 1.1) is not
  a proton.  Expected: recovers 62613:17; 288859/319809 unchanged.

- **Round 3+ — exploratory (design against round-1/2 re-runs).**  Remaining
  false STMs: 278662 (vertex, inside correct spike range), 349241, 353223,
  402330, 72586 (multi-object needing bundle-level topology), 48895, 392200,
  321107, 321371 (flat tracks — §1.5 says dQ/dx cannot do it; candidate
  signals: dead-region proximity at the stop, readout-window truncation
  geometry, leftover-track straightness), 321107/321371 (unstated by owner).
  Each will only ship with a §1-style measured separation.

Rounds are cumulative: round N's arm runs with rounds 1..N's knobs ON.
Per-round record in §4; every round commits (both repos) and pushes.

## 3. Evaluation protocol (per round)

1. `wcbuild` then freshness proof (`local/lib/libWireCellClus.so` mtime).
2. `./build/clus/wcdoctest-clus` passes.
3. **Knob-off gate**: 10-event subset re-run with the new binary, knobs off;
   `nusel-evt*.tsv` diff-identical and `hash_archive.py` member-identical
   (mabc-pr.zip, pctree-pr tarball) vs the round-0 reference.  Compiled-config
   proof: knob key absent when off, present when on (`wcsonnet | grep`).
4. **Knob-on arm**: all 72 baseline events, `run_round.sh r<N> <flags>`;
   `score_round.py` prints the confusion vs the owner baseline, the per-bundle
   fixes/regressions vs round 0, and any verdict flip on non-adjudicated
   bundles that share those events (reported as collateral, judged case by
   case — they have no truth label).
5. The round ships only on: fixes ≥ 1, regressions = 0, gate PASS.

## 4. Round records

### Round 0 — reference (work root `work-stmcamp-r0`)

All 72 baseline events re-run at the pre-campaign HEAD (2a821fd2) with the
production flag set.  Score: **56 correct / 16 wrong**, exactly the d59k
stored picture (the 15 false STMs + the miss).  This root is the A-arm for
every later byte-identical gate and the `--ref` for scoring.

### Round 1 — `accept_guards` (work roots `r1off`, `r1`, `r1b`)

- Knob-off gate: 10 mixed events with the round-1 binary, knobs off, vs r0:
  **GATE PASS** (nusel TSVs diff-identical; mabc-pr.zip and pctree-pr member
  content hashes identical).  Compiled-config proof: `accept_guards` key
  absent off / present on.  `wcdoctest-clus` 565/565.
- First knob-on arm (`r1`, cap ratio2 > 1.1): 6 fixed, **1 regression** —
  283463:14 (owner "STM is fine"), a sub-MIP track whose accepting call has
  ratio2 = 1.48.  The diagnosis table had been mis-read (its 1.48 overlooked
  when claiming "correct max 0.93").  The evaluation loop caught it.
- Cap raised to 2.0 (285443 stays dead through its 6.7 cm desert, 289832
  through its spike — both call-independent).  Re-run (`r1b`):
  **6 FIXED / 0 REGRESSED / no collateral flips** —
  fixes 278794:7, 285366:13, 285443:7, 289832:10, 290294:12, 409546:8.
  Score 62/72 correct (was 56).  SHIPPED.

## 5. Files

| path | what |
|---|---|
| `stm_campaign/extract_stm_diag.py` | offline diagnostics from stored pctree-pr tarballs (read-only) |
| `stm_campaign/run_round.sh` | one evaluation arm: fresh work root, 72 baseline events, production flags + round knobs |
| `stm_campaign/score_round.py` | confusion vs owner baseline + fixes/regressions vs reference arm |
| `scan-d59k/stm-baseline.tsv` | the doc-62 truth set |
| `/home/xqian/tmp/stm-campaign/` | scratch: extracted diagnostics TSVs |
