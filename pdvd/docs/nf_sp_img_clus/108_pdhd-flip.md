# 108 — PDHD flipped to `charge_stepped` + the two fit-lattice keys, on an explicit owner override

**Status: PDHD production default CHANGED. Toolkit `cfg/` only — no C++.**
**This flip is a D2, not a D1.** It was applied on the owner's explicit decision of 2026-09-15 to accept the
trade, overriding frozen amendment 7. That is recorded here as an override, not as a passed grade.

---

## 0. Repro

```bash
IMG=/home/xqian/toolkit-dev/wcp-porting-img; S=$IMG/pdvd/docs/nf_sp_img_clus/scripts

# sec 2 -- compiled-config proof and the escape hatch (pre-flip tree from `git archive HEAD cfg`)
CFG_ROOT=/home/xqian/tmp/d108/preflip/cfg bash $S/d102_compile_pr.sh pdhd d108preflip
bash $S/d102_compile_pr.sh pdhd d108flip
bash $S/d102_compile_pr.sh pdhd d108escape -S "retile_sampler_strategy='stepped'"

# sec 3 -- the proof arm: the FLIPPED CONFIG with no TLA overrides, on the graded arm's own pctrees
ARM=d108hflip DET=pdhd SRC=d102hcs JOBS=6 PIN=/home/xqian/tmp/d102/libpin_d102 PR_TLA="" \
  bash $S/d102_run_arms.sh
python3 $S/d103_flip_gate.py --det pdhd --a d102hcs --b d108hflip
python3 $S/d103_flip_gate.py --det pdhd --a d102hcs --b d108hflip --tree T_stm_michel_pts
```

`-S` parses its value as **jsonnet code**, so the escape hatch needs the inner quotes:
`-S "retile_sampler_strategy='stepped'"`. Written bare, `wcsonnet` aborts with
`Unknown variable: stepped` (rc 134).

---

## 1. What changed, and on whose authority

| file | change |
|---|---|
| `cfg/pgrapher/experiment/pdhd/pr.jsonnet` | `strategy_name=if retile_sampler_strategy == null then 'stepped'` → `'charge_stepped'` (one call site), plus the provenance comment and a correction to the now-misleading "same 'stepped' samplers" comment above it |
| `cfg/pgrapher/experiment/pdhd/pdhd_track_fitting.json` | `+ "fit_weight_pow": 1.5`, `+ "assoc_cont_center": 1`, with `_comment_d108_fit_lattice_knobs` |
| `cfg/pgrapher/experiment/protodunevd/pdvd_track_fitting.json` | **comment only** — its `_comment_d103_fit_lattice_knobs` asserted PDHD does *not* carry these keys, which is now false |

`clus.jsonnet`'s `bs_live_face` default stays `'stepped'` on both detectors — the PDVD flip (`8fc6070e`) left it
alone too, and only the `pr.jsonnet` retile call site moves. Clustering therefore still builds the 3-D point cloud
with `'stepped'`; only the **retile** for the steiner stage runs `charge_stepped`.

### 1.1 The grade this flip was applied on

Folded grade, A0 `d101hnew` → A1 `d102hcs`, truth `own103h > smx27 > smx28` under amendment 5 plus the symmetric
owner check `own103h2` (doc 104 sec 5.3):

| metric | A0 | A1 | Δ | bar −0.020 |
|---|---|---|---|---|
| `is_stm` purity | 0.975 | 0.959 | −0.016 | pass |
| `is_stm` efficiency | 0.626 | 0.616 | −0.011 | pass |
| **`michel_found` purity** | **0.914** | **0.880** | **−0.035** | **FAIL** |
| `michel_found` efficiency | 0.604 | 0.689 | **+0.085** | pass |

Frozen amendment 7 (`figs/104_pred_amend7.txt`, sha256 `321360da…`, frozen 2026-09-15T18:47:32) states: *"PDHD
D1 only if every metric ≥ A0 − 0.020 on both the folded grade and the projection. Otherwise D2 stands and PDHD
stays at production."* Michel purity clears neither the folded (−0.035) nor the projected (−0.036) form.

**The owner overrode that rule and accepted the trade** — purity for efficiency, with both `is_stm` metrics
passing — after four rounds each refuted one candidate explanation for the cost:

| round | doc | proposed cause | outcome |
|---|---|---|---|
| 7 | 104 | Michel admission geometry (4 levers) | refuted |
| 7 | 104 | one-sided hand labels | real, worth +0.016 |
| 8 | 105 | Bragg position / truncation reach | refuted |
| 9 | 106 | trajectory-end truncation | resized to 6 %, balanced by a 7 % advance |
| 10 | 107 | candidate churn | refuted — it is the efficiency gain |

What remains unexplained is **8 clusters both arms score and read differently** (doc 107 sec 4), which doc 104
sec 3 showed are not one class.

### 1.2 Costs this flip carries, measured and named

* **The absolute stopping-muon dQ/dx scale moves**: `plateau_med` **+3.8 % at p50, +47 % at p90** (doc 106 sec 6).
  Docs pdhd/16, 17, 29 and 50 calibrate on that scale. The Bragg **shape** is unchanged (median Δcontrast
  −0.0008), so the Michel/Bragg logic is not disturbed — but any absolute dQ/dx number in those four docs is
  stated against the pre-flip fit and has not been restated.
* **Five of 261 paired muons are read backwards** — entry and stop exchanged, sampler-attributed (doc 106 sec 5).
  One, `029107_16/46`, becomes an `is_stm` candidate in A1 having not been one in A0.
* **The STM evaluation is threshold-marginal**: a sub-centimetre refit flips `flag_pass` on 96 clusters (doc 107
  sec 2.1). This is the churn's mechanism and it is **also live in PDVD production** since `8fc6070e`, unmeasured
  there.

---

## 2. Compiled-config proof and the escape hatch

| label | what | md5 |
|---|---|---|
| `d108preflip` | `cfg` at HEAD (pre-flip), via `git archive` | `434208b1d840` |
| `d108flip` | the applied flip | `87a86589c767` |
| **`d108escape`** | **flip + `-S retile_sampler_strategy='stepped'`** | **`434208b1d840`** |

**The escape hatch is byte-identical to the pre-flip job.** Setting `retile_sampler_strategy='stepped'` restores
production exactly; removing the two keys from `pdhd_track_fitting.json` restores the C++ defaults 2.0 / 0.

Key-level diff, pre-flip → flip, over the 51-node compiled PR job:

* **8 differing nodes, all `BlobSampler`, all the `strategy` key and nothing else** — 4 PDHD anodes × 2 faces;
* `['stepped']` → `[{'disable_mix_dead_cell': False, 'name': 'charge_stepped'}]`.

**The two fit keys do not appear in the compiled config at all**, and that is expected: `pdhd_track_fitting.json`
is read at **runtime** by `TaggerCheckSTM` and `CheckSTM_Michel` (both confirmed to reference
`pgrapher/experiment/pdhd/pdhd_track_fitting.json` in the compiled job). A byte-identical compiled jsonnet does
**not** mean the fit is unchanged. In a full `-nu` job `TaggerCheckNeutrino` is a third runtime consumer; it is
not in this proof's pipeline list.

---

## 3. The flip gate: the applied config reproduces the graded arm exactly

Arm `d108hflip`: 61/61 events, `SRC=d102hcs` so the pctrees are byte-identical inputs, **`PR_TLA` empty** so the
flipped production config is what ran, pin `libpin_d102` clus md5 `091e142b9481` **unchanged before and after**.

| tree | events | rows / clusters | branches | result |
|---|---|---|---|---|
| `T_stm_michel` | 61 | 333 clusters | 199 | **IDENTICAL** |
| `T_stm_michel_pts` | 61 | 100107 clusters | 10 | **IDENTICAL** |
| `T_stm_pass` | 61 | 1158 rows | 9 | **IDENTICAL** |
| `T_stm_eval` | 61 | 4219 rows | 13 | **IDENTICAL** |

`T_stm_pass` and `T_stm_eval` live in `tracking-stm.root`, which `d103_flip_gate.py` does not read — it exits 1
with `KeyInFileError` on them. They were compared directly instead; the gate script was **not** modified.

**Why this can be exact.** The graded arm took its fit knobs through
`-A trackfitting_config=figs/101_tf_prod_pdhd_kf.json`, while the flip puts them in the production file. Those two
JSONs differ in exactly one key, `_d101_knobs`, which is a documentation string no C++ reads (it says the file is
*"a verbatim copy of pdhd_track_fitting.json plus the two fit lattice knobs"*). Every other key and value matches.

**Resources.** Peak RSS p50 1.66 → 1.63 GB, unchanged. Wall p50 reads 49 → 33 s, but that is **not** like-for-like
— the graded arm ran `JOBS=3` and this one `JOBS=6`, under different machine load — so no timing claim is made.

---

## 4. What is not concluded

* **The −0.035 is not explained**, only bounded to 8 clusters. It is accepted, not fixed.
* **Docs pdhd/16, 17, 29, 50 are not restated** against the flipped scale (sec 1.2).
* **PDVD is unmeasured** for the two defects this campaign found in the levers it already runs: the direction
  reversals and the STM evaluation's threshold-marginality.
* **`clus.jsonnet` still images with `'stepped'`** on both detectors; whether the retile and the clustering
  *should* share a strategy was never asked.

## 5. Files

* **Toolkit:** `cfg/pgrapher/experiment/pdhd/pr.jsonnet`, `cfg/pgrapher/experiment/pdhd/pdhd_track_fitting.json`,
  `cfg/pgrapher/experiment/protodunevd/pdvd_track_fitting.json` (comment only).
* **Arm (new):** `pdhd/work/*_d108hflip`, 61 events.
* **Unchanged:** no C++, no scan record, no other detector's config, `d103_flip_gate.py` and
  `d102_compile_pr.sh` reused as-is.
