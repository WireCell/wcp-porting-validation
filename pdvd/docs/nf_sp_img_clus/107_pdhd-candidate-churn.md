# 107 — the PDHD candidate churn is the STM evaluation flipping, and it is not the purity cost

**Status: measurement only. No code, no config, no new arm, no flip. PDHD stays at production.**

Doc 106 sec 2 found that the two trajectory levers' dominant effect on PDHD is **candidate churn** — A1 loses 80 of
A0's 341 STM candidates and gains 72, five times the end effect — and sec 8 recommended chasing it on the grounds
that it was *"the only channel large enough to carry the −0.035 purity cost."* Doc 101 sec 6.5 had left the
question open in one line: *"Which tagger decision drops a cluster is not chased."*

This round chases it, and answers two questions. The mechanism question has a clean answer: **the churn is the
upstream STM tagger's own evaluation changing its mind, 60–67 % of it, symmetrically in both directions.** The
size argument does not survive its own test: **8 of A1's 9 Michel false positives are clusters production also
scored**, while **20 of its 77 true positives are gained ones**. The churn is large, it is real, and it is
*net beneficial* — it explains the efficiency gain, not the purity cost.

---

## 0. Repro

```bash
IMG=/home/xqian/toolkit-dev/wcp-porting-img
cd $IMG/pdvd/docs/nf_sp_img_clus/scripts
python3 d107_churn.py > ../figs/107_churn.txt
```

Read-only, on the four arms of doc 106 sec 1.1, plus `tracking-stm.root` which was already written beside every
`tracking-pr.root`. Sec 5 is the only part that touches truth, and it uses the doc 103 grader's own population.

---

## 1. Where candidacy is actually decided

`CheckSTM_Michel.cxx:2617-2623`:

```cpp
if (!cluster->get_flag(Flags::main_cluster)) continue;
if (cluster->get_flag(Flags::TGM)) continue;
if (m_require_stm_flag && !cluster->get_flag(Flags::STM)) continue;   // PDHD's branch
if (!m_require_stm_flag && !cluster->has_pc("stm_pass")) continue;
```
then sorted by `ident()` and capped at `m_max_candidates` = 8 (`:2627`). `m_require_stm_flag` defaults **true**
(`:963`) and no PDHD config sets it, so the operative gate is `Flags::STM`, set at `TaggerCheckSTM.cxx:641` when
`check_stm_conditions()` returns true and the cluster is not TGM.

### 1.1 Two things that look like the answer and are not

**`T_cluster`'s tagger columns are dead.** `stm`, `tgm`, `fc` and `lm` are **always 0** on PDHD — 0 nonzero in
7272 rows over 20 events — the same unfilled-column defect already recorded for SBND. "They are identical between
the arms" is therefore vacuous, not a result, and was very nearly reported as one here.

What *is* filled is `is_main` (2530) and `in_scope` (6650), and those **are** identical on all 23266 shared
clusters. That is a real result with a consequence: **every one of the 80 lost ids is still present in A1, still
`is_main`, still `in_scope`.** The churn is a pure candidacy decision, not a clustering or main-cluster-choice
effect — so doc 102 sec 13.5's open hypothesis, *"most likely the main-cluster choice inside a bundle… Bundle
membership was not checked"*, is **refuted**.

A related trap avoided: **`flash_id` is not a bundle key.** Among the `is_main` clusters of a single event it takes
the value 72 six times and 96 four times. It is not used anywhere in this round.

**A geometric "relabel" test is meaningless here and was run before that was realised.** Matching each lost
candidate's chain against every A1 candidate's chain gives 0.00 overlap for all 80 — not because the tracks
vanished, but because clusters partition the charge and the pctrees are identical, so a cluster id denotes the
same charge in both arms. An id reassignment was never possible. The zero is a consistency check, not a finding.

### 1.2 The two trees that do answer it

`tracking-stm.root`, the upstream tagger's own output:

* **`T_stm_pass`** — one row per *(cluster, pass)*. `pass` is a pass **index** (0 forward, 1 backward,
  `TaggerCheckSTM.cxx:981`), **not** a verdict; 24 of 1140 clusters have two rows. Reading it as a verdict gives
  "24 clusters pass" against 341 candidates, which is how this was first misread.
* **`T_stm_eval`** — one row per `eval_stm_core` attempt, up to 8 (the `||` chains at `:3922-3953`), each carrying
  its own pass index.

`status` is the exit code:

| status | meaning | site |
|---|---|---|
| **0** | **accepted as STM** — `flag_pass`, no mid-point tracks, no proton | `:4051` |
| **3** | **`flag_pass` false — the STM evaluation itself did not pass** | `:4056` |
| 5 | `flag_pass` true but `detect_proton` fired | `:4056` |
| 4 | mid-point tracks (`check_other_tracks`) | `:4047` |
| 2 | charge left beyond the kink (`left_L` > 40 cm, or > 7.5 cm at > 2 MIP) | `:3884` |
| 1 / 7 / 8 | TGM / pre-fit geometric exits / no fit | `:3862`, `:3970+`, `:3803` |

### 1.3 The checks

**V1 — candidacy *is* status 0, exactly.** On every arm the candidate set and the status-0 set are **equal**, with
zero residue both ways (A0 341/341, K 320/320, S 328/328, A1 333/333). So `main_cluster` and not-TGM never bind
beyond it here, and the `max_candidates` cap never fires — no PDHD event reaches 8 candidates. **Every candidate
gained or lost below is an STM tagger verdict change and nothing else.**

**V2 — `flag_pass == any(verdict)` holds within one pass, never across passes.** Keyed per *(cluster, pass)* it
passes on all four arms, 0 violations of 427/436/426/443 status-3 passes. Pooling a cluster's two passes instead
reports one false violation each on A0 and S: `028084_19/63` has a forward pass that scored verdict 1 and still
exited status 7 — the pre-fit exits at `:3970+` run *after* the eval at `:3916-3953` — beside a backward pass with
status 3 and 8 failed attempts. That is an aggregation artifact, not a code inconsistency, and the first version
of this script published it as a violation.

---

## 2. The answer: the STM evaluation is what churns

Status transition of every lost and gained candidate:

| lever | direction | dominant transition | n | share | second |
|---|---|---|---|---|---|
| K | lost 70 | 0 accepted → **3 eval failed** | 39 | 0.56 | 5 proton (15, 0.21) |
| K | gained 49 | **3 eval failed** → 0 accepted | 29 | 0.59 | 2 left-charge (10, 0.20) |
| S | lost 89 | 0 accepted → **3 eval failed** | 54 | 0.61 | 4 mid-point / 2 left-charge (9, 9) |
| S | gained 76 | **3 eval failed** → 0 accepted | 48 | 0.63 | 2 left-charge (15, 0.20) |
| **A1** | **lost 80** | 0 accepted → **3 eval failed** | **48** | **0.60** | −1 no row / 4 mid-point (8, 8) |
| **A1** | **gained 72** | **3 eval failed** → 0 accepted | **48** | **0.67** | 2 left-charge (16, 0.22) |

**Status 3 is `flag_pass == false`** — `eval_stm_core` never returned true on any of its attempts. It is the
largest single transition in every lever and in both directions, and for A1 it is the *same number both ways*:
48 lost to it, 48 gained from it.

The second contributor is status 2, the left-charge cut (`left_L > 40 cm`, or `> 7.5 cm` above 2 MIP), which is
strongly asymmetric — A1 gains 16 from it and loses 7 to it.

The whole population by status shows how small the moves are against the pool:

| status | A0 | K | S | A1 |
|---|---|---|---|---|
| 0 accepted | 341 | 320 | 328 | 333 |
| 3 eval failed | 409 | 418 | 411 | 427 |
| 7 pre-fit exit | 255 | 252 | 258 | 240 |
| 2 left-charge | 98 | 99 | 92 | 91 |
| 4 mid-point | 24 | 25 | 31 | 33 |
| 5 proton | 13 | 22 | 14 | 11 |

### 2.1 The evaluation is sitting on its own decision boundary

For the 0↔3 flips, the eval quantities barely move:

| lever | direction | n | `n_eval` A0 → L | `ks1` A0 → L | `ratio1` A0 → L |
|---|---|---|---|---|---|
| A1 | 0 → 3 (lost) | 48 | 3 → 8 | 0.0809 → 0.0832 | 1.3117 → 1.2688 |
| A1 | 3 → 0 (gained) | 48 | 8 → 2 | 0.0811 → **0.0607** | 1.3408 → 1.2035 |

`n_eval` going 3 → 8 and 8 → 2 is not an independent finding — the `||` chain stops at the first success, so a
failing cluster necessarily exhausts all 8 attempts. It is a consistency check that the flip really is the eval.

What matters is that **`ks1` and `ratio1` move by a few percent and the verdict flips.** The same physical charge,
refitted with a sub-centimetre trajectory change (doc 106 sec 3: median stop displacement 0.98 cm), crosses the
evaluation's threshold. This is the *refit instability* doc 103 was named for, located at a specific gate.

---

## 3. What the churn carries

| lever | stratum | n | `is_stm` | Michel | `muon_len` p50 |
|---|---|---|---|---|---|
| A1 | lost | 80 | 20 | 34 | 145.5 |
| A1 | gained | 72 | 27 | 36 | 123.4 |

These are reco flags on objects the *other* arm never scored, so they are a size, not a grade. Note they are
nearly **balanced** — 20 vs 27 `is_stm`, 34 vs 36 Michel. A churn that is large and balanced does not by itself
cost anything, which is the first sign that doc 106's size argument needed a direct test.

---

## 4. The direct test: the churn is not the purity cost

Using the doc 103 grader's own population and Michel truth (amendments 3 and 5, `--stm-only-unset-negative`), never
a hard-coded key list:

| | count |
|---|---|
| graded Michel population | 196 |
| A1 tags | 77 true / **9 false** |
| **A1 false positives that are *gained* candidates** | **1 of 9** |
| A1 true positives that are *gained* candidates | **20 of 77** |
| A0's 4 false positives lost under A1 | 0 of 4 |

**Eight of A1's nine Michel false positives are clusters production also scored.** The churn brings in almost no
false positives — and it brings in **20 real Michels**, which is where A1's efficiency gain comes from
(`michel_found` efficiency 0.604 → 0.689, +0.085 in the folded grade of doc 104 sec 5.3).

So the two halves of the grade have different causes, and they can now both be named:

* **the efficiency gain (+0.085) is the churn** — 20 true Michels on clusters production never considered;
* **the purity cost (−0.035) is not** — it lives on the 8 shared clusters, the same objects both arms scored and
  read differently, which is the population docs 104 and 105 spent two rounds on and doc 106 sec 7 showed are
  about twice as likely to be truncated.

### 4.1 Correction to doc 106 sec 8

Doc 106 recommended the churn as round 10's first target because it was *"the only channel large enough to carry
a −0.035 purity cost."* That was a size argument, and size was the wrong test: this round's sec 4 tests the claim
directly and it fails. The churn is large, benign and on the efficiency side. A pointer has been added to doc 106.

The recommendation was still worth following — it is what produced sec 2's mechanism and sec 4's split of the
grade — but it was justified by the wrong reasoning, and the ranking in doc 106 sec 8 should not be reused.

---

## 5. What this leaves

The PDHD Michel purity cost has now survived four rounds of explanation attempts:

| round | doc | proposed cause | outcome |
|---|---|---|---|
| 7 | 104 | Michel admission geometry (4 levers) | refuted |
| 7 | 104 | one-sided hand labels | real, worth +0.016, leaves −0.035 |
| 8 | 105 | Bragg position / truncation reach | refuted (`far_full` = 0, AUC ≈ 0.5) |
| 9 | 106 | trajectory-end truncation | resized to 6 %, balanced by advance |
| **10** | **107** | **candidate churn** | **refuted — it is the efficiency gain** |

What has *not* been refuted, and is now the whole of the remaining explanation, is the 8 shared-cluster false
positives: clusters both arms carry, both arms fit, and read differently. Doc 104 sec 3 measured five of them at
point level and found they are not one class. **That is a population of 8, and it is not obvious any general rule
exists over it.**

## 6. Round 11: the honest options

1. **Stop explaining and decide.** The cost is 8 clusters. Four rounds have each removed a candidate explanation
   without moving the number. The owner's live options are unchanged and are now much better characterised than
   when doc 103 posed them: hold PDHD at production, or accept the trade (−0.035 purity for +0.085 efficiency,
   with both `is_stm` metrics passing).
2. **The STM evaluation's stability, as a defect in its own right.** Sec 2.1 shows a sub-centimetre refit flipping
   `flag_pass` on 96 clusters in A1 alone. That is worth knowing whatever PDHD does, it is **live in PDVD
   production** since `8fc6070e`, and it is unmeasured there.
3. **The direction reversals of doc 106 sec 5** — 5 in A1, sampler-attributed, also live in PDVD.

Options 2 and 3 are not PDHD-flip questions; they are toolkit questions that the PDHD campaign happened to
surface, and both apply to a detector already running these levers in production.

---

## 7. Not concluded

* **Why `eval_stm_core` is threshold-marginal on ~96 clusters.** Sec 2.1 locates it and does not explain it.
* **Whether the churn's 20 gained true Michels are a real efficiency gain or truth-record artefacts.** They are
  graded positive by the doc 103 record, but that record was built on `h25base` (doc 106's note on scan
  provenance), and gained clusters are by construction ones production never showed a scanner.
* **PDVD.** Every number here is PDHD; the same decomposition on PDVD is one command and has not been run.
* **Status 2 (left-charge), the asymmetric second contributor** — A1 gains 16 from it, loses 7 to it. Unexamined.

---

## 8. Files

* **Script (new):** `scripts/d107_churn.py`.
* **Figure (new):** `figs/107_churn.txt`.
* **Amended:** `106_pdhd-trajectory-end-measured.md` sec 8 — a pointer to sec 4.1 here.
* **Arms (existing, unmodified):** `pdhd/work/*_{d101hnew,d101hkf,d102hocs,d102hcs}`, `tracking-stm.root` and
  `tracking-pr.root` of each.
* **Unchanged:** no toolkit code, no jsonnet, no scan record, no new arm, no production default.
