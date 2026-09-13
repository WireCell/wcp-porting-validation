# doc pdhd/26 — PDHD vs PDVD, refreshed: STM and STM+Michel efficiency and purity, the Michel energy spectra on the region estimator (now PDHD production too), and dQ/dx vs residual range against the expectation

**The owner's request (2026-09-12):** a comprehensive PDHD vs PDVD comparison of the STM and STM+Michel
results; Michel energy spectra for the Michels the chain identifies, PDHD vs PDVD; for the stoppers with a
good dQ/dx vs RR, the mean dQ/dx against the expectation. Mid-way: *"PDHD should update to region based
energy estimation for this"* and *"Please flip it as default for PDHD"*.

**Answers, PDHD APA0 excluded (strict) as in docs pdhd/23–25, both detectors on their production chains:**

1. **The flip (§1).** PDHD production now publishes PDVD's region-based Michel energy with PDVD's own five
   keys, no PDHD tuning. It is purely additive: on the measured arm every pre-existing branch, point row, tree, zip,
   calib file and verdict is bit-identical to production on all 341 candidates / 61 events, and the confirmation
   arm on the flipped file is bit-identical to the measured arm, including the new per-cell tree.

*Answers 2–5 — STM and STM+Michel efficiency and purity, the Michel energy spectra, and dQ/dx vs residual range — are written in the next commit, on the flipped chain.*

**Production.** §1 flips five keys into `pdhd/wct-pr-perevt.jsonnet` on the owner's go. No C++ change
(toolkit `81ff37d7`, pin `libpin_p96`, `libWireCellClus` md5 `4e1db810`), no other detector's file, no record
or label written, new tags only. Everything after §1 is read-only.

## 0. Repro

```sh
I=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img ; D=$I/pdhd/docs/scan ; X=$D/d26
export STM_SCAN_RECORD=$D/pdhd_stm_michel_smx27_verdicts.json     # PDHD truth (doc pdhd/25 sec 8); PDVD = smx1a..smx9

# 1. part A, the flip.  Pre-registration and TLA written first: $X/preregistered.txt, $X/tla_q2d.txt
md5sum /home/xqian/tmp/p96/libpin_p96/*.so* > /home/xqian/tmp/h27/libpin_md5_before.txt
(cd $D && bash h25/run_arm.sh h26q2d d26/tla_q2d.txt)                  # measurement: production file + 5 keys, 61 events
MODE=additive bash $X/d26_gates.sh > $X/gate_h26q2d.txt                # vs h26conf (production): PASS
#    (compile-only h27cfg0 / h27cfgT from the unedited file, then) edit pdhd/wct-pr-perevt.jsonnet, then
bash $X/d26_proofs.sh > $X/cfg_proofs.txt                              # A 0/0/0 + whole config, B inert, C 5 keys, D PDVD unchanged
(cd $D && bash h25/run_arm.sh h26q2dprod)                              # confirmation: the flipped FILE, no TLA
MODE=confirm bash $X/d26_gates.sh > $X/gate_h26q2dprod.txt             # vs h26q2d: bit-identical
python3 $I/pdvd/docs/nf_sp_img_clus/scripts/d81_readout.py --det pdhd --arm h26q2dprod --json <scratch>/readout.json  # cross-shared cells
python3 $I/pdvd/docs/nf_sp_img_clus/scripts/d81_tail.py <scratch>/readout.json pdhd      # (same two for --det pdvd --arm p96vprod)
```

**Conventions (doc pdhd/25 §0, unchanged).** PDHD truth = record `smx27`, owner precedence
(`owner_review` > `owner_smx1` > agent), population = the committed 303-item key; **strict** drops any candidate
with a role-1 point in APA0 (x < 0, z < 231 cm), **majority** drops the APA0-majority ones. PDVD truth = the
merged smx1a…smx9 record's flat verdict, population = items with a candidate (546); the all-judged figure
(576, the 10 stoppers the tagger never hands on counted as misses) is printed beside it. MESSY/UNCLEAR are
unscored on both. **Both efficiencies are on the chain's own candidate pool** — neither is an absolute
efficiency. Every number below is printed by a committed script from the arms named.

## 1. Part A — the region-based Michel energy flipped into PDHD production

### 1.1 What was flipped, and why these values

| key | C++ default | PDVD production | **PDHD production now** |
|---|---|---|---|
| `michel_q2d` | false | true | **true** |
| `michel_q2d_cells` | false | true | **true** |
| `michel_q2d_region_cm` | 0.0 (off) | 10.0 | **10.0** |
| `michel_q2d_region_ctl_cm` | −1.0 (off) | 35.0 | **35.0** |
| `michel_q2d_region_scope` | 0 | 1 | **1** |

The estimator (docs pdvd/95, 96): all 2-D charge in a 10 cm region around the stop on the candidate's own
cells (main cluster + admitted companions), minus the charge the muon's trajectory fit predicts there,
through the bound recombination model — so the energy does not depend on how PR segmented the Michel. The
same sum on a region 35 cm back up the muon is the **body control**. **The key set is PDVD's, inherited with
no PDHD tuning**: R = 10 was post-hoc on PDVD (doc 95) and scope 1 was flipped there on fidelity (doc 96). A
PDHD-selected radius would make the two spectra different estimators. Doc pdvd/81 §8a held PDHD OFF because
its Michels lean more on cross-shared cells; the owner took the flip with that known, and §3.3 re-measures it.

### 1.2 Gates

| gate | label / file | result |
|---|---|---|
| pre-registration | `scan/d26/preregistered.txt` (written before `h26q2d` launched) | predictions 1a–f, 2 |
| measurement arm | `h26q2d` = unedited production file + `tla_q2d.txt`, 61/61, rc 0, pin md5 before = after | — |
| **additive gate** | `scan/d26/gate_h26q2d.txt` (`MODE=additive`) vs `h26conf` | **PASS**: 341/341 candidates bit-identical on all 149 pre-existing branches and every `T_stm_michel_pts` column; 0 `is_stm` / `michel_found` / `reject_bits` changes; exactly the 49 registered new branches and one new tree `T_stm_michel_2d`; every other tree of `tracking-pr.root` and `tracking-stm.root`, `mabc-pr.zip` members and `calib-pr-evt*.json` identical on 61/61 events; census on `smx27` unmoved (strict 89/1/15/67, all 114/2/33/105, Michel 47/3/10/38); `michel_q2d_valid` 341/341 |
| compiled config | `scan/d26/cfg_proofs.txt` | A `h27cfgT → h27cfg` **0/0/0**, whole compiled config identical after the tag rename; B keys forced back = exactly the 5 keys at the C++ initializers; C `h27cfg0 → h27cfg` **exactly 5 added**; D PDVD file unchanged |
| **confirmation arm** | `h26q2dprod` = the flipped file, no TLA; `scan/d26/gate_h26q2dprod.txt` (`MODE=confirm`) vs `h26q2d` | **PASS, bit-identical**: 61/61, rc 0, pin md5 unchanged against the manifest taken before `h26q2d`; 341/341 candidates on all 198 branches and every point column; every tree including all rows of `T_stm_michel_2d`, zip members and calib json identical on 61/61 events; census unmoved |

Two of my own defects, caught before they could matter:
* the gate's first run (`gate_h26q2d_run1.txt`, kept) took its census on `smx23`, because
  `d25_bragg_michel` reads `STM_SCAN_RECORD` at import and the script set a variable but not the environment.
  Both arms were on the same record, so "same" still held. The script now exports it, and the re-run reads
  `smx27`'s tuples.
* the pre-registered expectation E2 ("PDHD region medians lower than PDVD's") missed; see §3.

**Stale comment, not edited:** `pdvd/wct-pr-perevt.jsonnet` still says "PDHD stays OFF and carries no
michel_q2d key at all". It is PDVD's production file, and this doc does not touch it.

## Files

| file | what |
|---|---|
| `pdhd/wct-pr-perevt.jsonnet` | §1: the five keys at the end of `stm_michel_knobs` |
| `scan/d26/preregistered.txt`, `tla_q2d.txt` | §1: predictions and the measured TLA, written before the arm |
| `scan/d26/d26_gates.sh` | §1: additive and confirmation gates (fork of `pdvd/.../d95_gates.sh`) |
| `scan/d26/gate_h26q2d.txt`, `gate_h26q2d_run1.txt`, `gate_h26q2dprod.txt` | §1: gate outputs (run1 = census on the wrong record, kept) |
| `scan/d26/d26_proofs.sh`, `cfg_proofs.txt` | §1: compiled-config proofs A–D |
| `scan/d26/q2d_readout_pdhd.txt`, `q2d_readout_pdvd.txt` | §3.3 of the full doc: `d81_readout` + `d81_tail` on both production arms |
