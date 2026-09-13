# 99 — One gain constant for the PDVD top electronics in SP, a full rerun, and the hand-scan record carried onto it

**Status (2026-09-13):**
- **Knob:** shipped **default OFF**. The compiled production config is byte-identical (G1).
- **Rerun and carry:** the full SP → imaging → clustering → PR rerun is done for OFF and ON, and the hand-scan record is
  carried onto both.
- **Charge:** the constant does exactly what it was built to do. The same top track gains ×1.1228 [1.1176, 1.1275]
  (1/s = 1.1249), and the bottom is unchanged at 1.0000 on 30/30 tracks.
- **Pre-registered closure: MISSED on the full sample.** Top/bottom is **0.967 [0.953, 0.987]**, a CI that does not cover
  1. On the 111 tracks present in both arms it reads 0.972 [0.952, 1.000] (§7). s was not retuned.
- **Rerunning SP itself costs STM efficiency** against production, independent of the constant: is_stm eff 0.877 → 0.801
  (§4.3).
- **Disk:** the July SP-frame retire is **HELD** for the owner (§9). The OFF trim is done.

Doc pdhd/29 §8–9 found PDVD top-volume stopper dQ/dx at **0.889 [0.875, 0.903]** of bottom (M2 fit). There:
- bottom agreed with PDHD inside the budget;
- SP's gain was applied correctly per run;
- the offset followed the TDE (top) / BDE (bottom) readout boundary.

This doc does three things:
- applies that number as one SP constant for the top electronics;
- reruns the chain from the raw frames;
- carries the hand-scan record onto the rerun.

Owner, 2026-09-13:

> "we want to have one calibration constant included, I feel that we can fold it as a separate gain for SP first,
> (note only top electroncis PDVD). We need to make sure the hand scan results can still be used."

> "First, I think we should only have one scaling factor, instead of plane-dependent constants. Second, I do not want
> to copy the raw frame to my own disk, since it takes over too much disks."

> "Note, we probably should keep the latest SP results, and we can retire the previous round's SP result"
> (with "trim after imaging" for the control arm).

## Repro

```bash
# knob gate (G1): compiled config before/after, production entry + every importer
#   -> d99/g1_compiled_config.txt
# one-event checks (G2, sign, L1SP share), run from pdvd/ with the production runner defaults
setarch x86_64 -R ./run_nf_sp_dnnroi_evt.sh -O _p98voff 039252 0
setarch x86_64 -R ./run_nf_sp_dnnroi_evt.sh -O _p98g2rep 039252 0
setarch x86_64 -R ./run_nf_sp_dnnroi_evt.sh --top-gain-scale 0.889 -O _p98von 039252 0
setarch x86_64 -R ./run_nf_sp_dnnroi_evt.sh -L off -O _p98g2nol1 039252 0
python3 abtest/hash_archive.py work/039252_0_<arm>/protodune-sp-dnnroi-frames-anode*.tar.bz2      # d99/g2_determinism_hashes.txt
python3 docs/nf_sp_img_clus/scripts/d99_frames.py --events 039252_0 --arms p98voff p98von --nol1 p98g2nol1

cd pdvd/docs/nf_sp_img_clus/scripts
# matcher controls (G4a identity, G4b shuffled) and fork controls (G5a grade, G5b dQ/dx + space charge)
python3 d99_match.py --arm p96vprod
python3 d99_match.py --arm p96vprod --shuffle
python3 d99_grade.py
python3 d99_dqdx_drift.py --arm p96vprod --out /home/xqian/tmp/p98/dq/dqdx_p96vprod.json
python3 ../../../../pdhd/docs/scan/d29/d29_space_charge.py --json /home/xqian/tmp/p98/dq/dqdx_p96vprod.json --fig /home/xqian/tmp/p98/dq/sc.png

# G0: PR on d51vclus with the production pin reproduces p96vprod (3 events, arm p98g0; d99/g0_pr_reproduction.txt)

# the arms (raw frames read in place from pdvd/input_data; nothing copied)
ARM=p98voff JOBS=6 ./d99_sp_arm.sh
ARM=p98von EXTRA="--top-gain-scale 0.889" JOBS=6 ./d99_sp_arm.sh
ARM=p98voff ./d99_chain.sh img ; ARM=p98von ./d99_chain.sh img
python3 d99_manifest.py --out-dir ../d99                                   # frame manifests + G3
SRC=d27fresh DST=p98kq EVENTS="039349_81 039349_4 039349_15 039349_68 039349_2 039252_13 039252_0 039253_0" ./d99_stage_q.sh
ARM=p98kq ./d99_chain.sh clus ; ARM=p98kq ./d99_chain.sh pr                 # clustering+PR drift control
SRC=p98voff DST=p98voffq ./d99_stage_q.sh ; ARM=p98voffq ./d99_chain.sh clus ; ARM=p98voffq ./d99_chain.sh pr
SRC=p98von  DST=p98vonq  ./d99_stage_q.sh ; ARM=p98vonq  ./d99_chain.sh clus ; ARM=p98vonq  ./d99_chain.sh pr
python3 d99_frames.py --all --arms p98voff p98von --json <sums>            # d99/frame_sums_120evt.txt
(cd ../../.. && python3 docs/nf_sp_img_clus/scripts/d99_readout_window.py)  # sec 4.1 -> d99/readout_window_effect_p98voff.txt

# carry the record (sec 5), grade (sec 6), measure (sec 7-8); per ARM in p98voffq p98vonq
python3 d99_match.py --arm $ARM --out <match.json> --carried ../../scan/pdvd_stm_michel_smx9_carried_$ARM.json   # d99/match_$ARM.txt
python3 d99_rescan.py --match <match.json> --arm $ARM --tsv ../d99/rescan_$ARM.tsv --baseline ../d99/rescan_identity_p96vprod.tsv
python3 d99_dqdx_drift.py --arm $ARM --record ../../scan/pdvd_stm_michel_smx9_carried_$ARM.json --out <dq>/dqdx_$ARM.json
python3 ../../../../pdhd/docs/scan/d29/d29_space_charge.py --json <dq>/dqdx_$ARM.json --fig <png>   # d99/space_charge_$ARM.txt
(cd ../../../.. && python3 pdhd/docs/scripts/d16_stm_energy_scales.py --det pdvd --arm "pdvd/work/*_$ARM" --chain-C 0.7941 --out <prefix>)  # d99/c_refit_$ARM.txt
python3 d99_grade.py            > ../d99/grade_p96vprod_p98voffq_p98vonq.txt
python3 d99_grade.py --ok-only  > ../d99/grade_all_okonly.txt
python3 d99_michel_ql.py        > ../d99/michel_ql_p96vprod_p98voffq_p98vonq.txt
python3 d99_closure.py --write-common <dq> > ../d99/closure_pertrack.txt   # then d29_space_charge.py on dqdx_<arm>_common.json
                                                                           #   -> d99/space_charge_<arm>_common.txt
# disk (sec 9)
CONFIRM=1 ./d99_trim_off.sh
./d99_retire_keep_sp.sh ; NEGCTL=1 ./d99_retire_keep_sp.sh                 # dry run + negative control only (HELD)
```

## 1. The constant and where it acts

**s = 0.889**, pre-registered in doc pdhd/29 before any rerun (M2 A_top/A_bottom 0.890, bootstrap 0.889 [0.875, 0.903]).
It is one number for all three planes of the four top anodes (idents 4–7), per the owner.

SP has **two** top-electronics charge normalisations, and the constant goes into both:

| consumer | legacy top value | with s | how charge scales |
|---|---|---|---|
| OmnibusSigProc `postgain` (`sp.jsonnet` make_sigproc; used at `OmnibusSigProc.cxx:916` as `m_inter_gain * m_ADC_mV`) | 1.0 | s | ∝ 1/s |
| L1SPFilterPD `gain_scale` → `kernels_scale` (kernels are ADC/e) and the three ADC thresholds `l1_raw_asym_eps` 20, `raw_ROI_th_adclimit` 10, `adc_sum_threshold` 160 (`sp.jsonnet`, `l1sp_after_dnnroi.jsonnet`) | 1.0 | s | ∝ 1/s |

Without the second row, every top U/V region that L1SP rewrites would stay at the old scale inside a corrected volume.

Not touched:
- `params.elecs[1]` (JsonElecResponse, postgain 1.36): it is shared with simulation.
- NF: the top channel-noise DB has hard-coded responses and `gain_scale` 1.0, so NF output is identical by construction.
- The bottom anodes, which never read the knob.

The knob is threaded through four places:
- `make_sigproc(..., top_gain_scale=1.0)` in `sp.jsonnet`;
- the same argument on `l1sp_after_dnnroi.jsonnet`;
- a top-level argument `top_gain_scale=1.0` of `pdvd/wct-nf-sp-dnnroi.jsonnet`, passed to both;
- `run_nf_sp_dnnroi_evt.sh --top-gain-scale <s>`, which adds the TLA only when given.

**Default OFF; production unchanged.** Flipping it is the owner's next-round decision, and so are the downstream constants
that were fitted on the old top scale (§8).

## 2. Gates

| gate | what | result | record |
|---|---|---|---|
| **G1** compiled config | Production entry with the runner's TLAs, L1SP on and off: md5 before = after = after with `top_gain_scale=1.0`. 19 entry files import either jsonnet (default TLAs, external variables supplied where needed). 15 compile identically. 4 fail identically before and after: 3 `wcls-sim-drift-*` need real ext vars, and `pdvd/wcls-nf-sp-out.jsonnet` cannot find `params.jsonnet` (pre-existing, M12-type rot, not fixed). ON (s = 0.889): **exactly 20 values change**, the 5 keys of §1 on each of anodes 4–7 | **PASS** | `d99/g1_compiled_config.txt` |
| knob-on smoke | top L1SP log: `kernels_scale=0.8890` (ON) vs `1.0000` (OFF) | seen | SP logs of 039252_0 |
| **G2** determinism | 039252_0 OFF run twice under `setarch -R`: 8/8 frame archives have identical member-content hashes. ON vs OFF: anodes 0–3 identical, 4–7 differ | **PASS** | `d99/g2_determinism_hashes.txt` |
| **G0** PR reproduction | PR on the `d51vclus` pctree with `libpin_p96` and today's `wct-pr-perevt.jsonnet` (arm `p98g0`; events 039252_0, 039253_0, 039349_10) vs `p96vprod`: every branch of 9+7 trees identical. The one apparent difference, `T_rec_charge.reduced_chi2`, is NaN≠NaN in a list compare: the NaN pattern is equal and the finite values are bit-equal. `mabc-pr.zip` 25/25 members identical, calib json identical | **PASS** | `d99/g0_pr_reproduction.txt` |
| **G3** bottom identity, all events | Member-content hash of every SP frame archive of both arms (1920): anodes 0–3 identical OFF = ON on 120/120 events; anodes 4–7 differ on 120/120. The manifests list every archive (OFF 39.05 GiB, ON 39.07 GiB, 0 missing) | **PASS** | `d99/g3_bottom_identity.txt`, `d99/frames_manifest_p98vo{ff,n}.txt` |
| frame sums, all events | ON/OFF `gauss` sums over 120 events: bottom 1.0000 on all 12 anode-planes; top per-channel median 1.1249 on all 12 anode-planes (§3) | as predicted | `d99/frame_sums_120evt.txt` |
| **G4a** matcher identity | record → `p96vprod` itself: 601/601 `ok`, same key, f_fwd = f_rev = 1.000, no x offset | **PASS** | `d99/g4a_match_identity_p96vprod.txt` |
| **G4b** matcher negative | each event's record against another event's arm: 0 `ok`, 600 `lost`, 1 `merged` | **PASS** | `d99/g4b_match_shuffled_p96vprod.txt` |
| **G5a** grading fork | the committed record on `p96vprod` through `d99_grade.py`: is_stm (242, 8, 34, 262), michel (144, 12, 20) | **PASS** | `d99/g5a_grade_gate.txt` |
| **G5b** dQ/dx fork | `d99_dqdx_drift.py` on `p96vprod`: 183/183 per-track rows identical to the committed `d27/dqdx_drift.json`. Doc pdhd/29's space-charge fit on it reproduces `space_charge.txt` line for line except the figure path (M2 0.890; bootstrap 0.889 [0.875, 0.903]) | **PASS** | `d99/g5b_*.txt` |
| bottom, physics level | per-track dQ/dx plateau ON/OFF on the 30 bottom tracks present in both arms: 1.0000 on 30/30 | **PASS** | `d99/closure_pertrack.txt` |

**Libraries.** `local/lib` was compared against the production pin `libpin_p96`: of 572 libraries, only
`libWireCellClus.so` differs (a PDHD-only change, doc pdhd/28).
- `run_nf_sp_dnnroi_evt.sh:26` prepends `local/lib` itself, so an outer pin does not reach SP.
- SP maps no `libWireCellClus` (checked in `/proc/<pid>/maps`), so SP runs the pinned bytes.
- The SP driver records every `local/lib` md5 at start and end.
- Imaging, clustering and PR run with the pin prepended.

No C++ was changed, so no package's `wcdoctest` applies; the toolkit change is two jsonnet files (G1).

## 3. Sign, size, and what L1SP carries

Per-plane sums of the saved `gauss` frames (DNN-ROI + L1SP output), ON / OFF:

| | U | V | W | record |
|---|---|---|---|---|
| bottom (anodes 0–3), plane sum, 039252_0 | 1.0000 | 1.0000 | 1.0000 | `d99/g2_frames_039252_0.txt` |
| top (anodes 4–7), plane sum, 039252_0 | 1.1303 | 1.1331 | 1.1258 | ″ |
| top, per-channel median (channels > 3e4 e in both), 039252_0 | 1.1249 | 1.1249 | 1.1249 | ″ |
| bottom, plane sum, **120 events** | 1.0000 | 1.0000 | 1.0000 | `d99/frame_sums_120evt.txt` |
| top, plane sum, **120 events** | 1.1383 | 1.1331 | 1.1257 | ″ |
| top, per-event median-channel ratio, p50 over 120 events, every top anode | 1.1249 | 1.1249 | 1.1249 | ″ |

1/0.889 = 1.1249: the sign is right, and the size is exact channel by channel.

Over the 120 events the U and V plane sums are 0.7–1.2 % above 1/s (0.5–0.7 % on 039252_0). On W they are within 0.1 %. SP's absolute thresholds (ROI, DNN-ROI mask,
L1SP triggers on U/V) now admit a little more top charge. That is part of putting the constant in SP rather than scaling
afterwards. It is also why the Michel *control* region, which is mostly small charges, scales by more than 1/s (§6.2).

L1SP share: OFF was compared against the same OFF configuration with `-L off`. On top U/V, L1SP rewrote channels carrying
**0.2–5.9 %** of each anode-plane's charge (0 on W, which it does not process). Small, but not negligible for a constant
meant to be uniform, hence the second row of §1.

**A runner artefact, not a failure.** Some SP jobs exit rc = 2 even though all 8 frame sinks closed normally.
- **Cause:** the runner runs under `set -e` (`run_nf_sp_dnnroi_evt.sh:19`). When the background `nvidia-smi` monitor has
  exited with its own error code, the runner dies in `wait $NVSMI_PID`, before printing `DNN-ROI done`.
- **Handling:** the driver treats those events as incomplete and reruns them. The first-pass frames were hashed beforehand,
  so the rerun doubles as a determinism check.
- **Result:** on the OFF arm 4 events exited rc = 2 (039252_1, 039253_8, 039253_10, 039253_12). After the rerun, **8/8
  archives are identical on all 4** (`d99/sp_rc2_rerun_identity.txt`).

## 4. The arms

| arm | what | status |
|---|---|---|
| `p98voff` | NF + SP + DNN-ROI + L1SP with the knob unset, all 120 events, plus imaging. Its first clustering + PR pass ran with the SP frames in the input dir (§4.1) and is kept only as the measurement of that difference | SP 120/120, imaging 120/120; frames trimmed to 6 gate events (§9) |
| `p98von` | same with `--top-gain-scale 0.889`, plus imaging. **All frames kept** (the latest SP) | SP 120/120, imaging 120/120 |
| `p98voffq`, `p98vonq` | clustering + PR, staged like production (below); **the arms every grade in this doc uses** | clustering 120/120 and PR 120/120 on both |
| `p98kchk` | imaging control: today's imaging on the July `_keep` frames, 8 events | 8/8 identical to `d27fresh` |
| `p98kq` | clustering + PR control: today's clustering + PR on `d27fresh` imaging, 8 events | 8/8 identical to `d51vclus` / `p96vprod` |
| `p98g0` | PR control on the `d51vclus` pctree, 3 events | identical to `p96vprod` (G0) |

Every SP job ran under `setarch -R`. `local/lib` md5s were unchanged from start to end of both SP arms.

The chain driver's completion check for PR requires the tagger's `CheckSTM_Michel: N candidate(s)` log line. It marks one
PR "FAIL": `039252_13` on `p98voffq`. That run completed normally (`tracking-pr.root` written, all sinks closed, no error)
with **no** STM candidate, so the tagger prints no summary line. This is the same signature as production's `039252_11`,
which has no candidate either. Both OFF and ON measure 120 processed events: OFF has candidates in 119, ON in 120.

### 4.1 A staging difference that looked like physics

The first clustering + PR pass on `p98voff` tagged **503** STM candidates against `p96vprod`'s **596**. 205 record items
that match cleanly by geometry had lost their candidate, none of them to another cluster. The tagger log names the reason
on the lost stoppers: `readout_edge_guard: … stop at tick 6356.0 within 60 ticks of the readout window edge`.

The cause is in the clustering runner:
- `run_clus_evt.sh:270-282` sets `readout_window_ticks` from an SP frame archive **in the clustering input directory** when
  one is there, and falls back to 10000 otherwise.
- Production's `d51vclus` was staged by `scripts/stage_ql_tag.sh` with the imaging archives only, so **all 120 production
  events ran with 10000**.
- The `p98voff` directories hold the SP frames themselves, so 84 events (run 039349) ran with **6400**, the frames' real
  length (both the July and the new frames are 6400 ticks × 500 ns).
- With the true window, the PR's `readout_edge_guard` fires 363 times instead of 264 and removes stoppers whose stop lies
  in the last 60 ticks.
- The runner's own comment still says "10000 ticks × 0.5 us".
- Record: `d99/readout_window_effect_p98voff.txt` (`scripts/d99_readout_window.py`). It holds the candidates (596 / 503 /
  597 on `p96vprod` / `p98voff` / `p98voffq`), the guard firings (264 / 363 / 256), the window per event, and the 205
  cleanly matched items that lost their candidate.

This is neither the gain constant nor the SP rerun. For this round the arms reproduce production:
- `d99_stage_q.sh` stages clustering into `p98voffq` / `p98vonq` exactly as `stage_ql_tag.sh` does: imaging archives +
  `img-provenance.txt`, no frames.
- The `p98voff` clustering + PR pass is kept as the measurement of the difference.

**Open for the owner** (not changed here): production's edge guard runs on a 10000-tick window over 6400-tick frames, so it
cannot fire at the real readout edge. Enabling the real window would move STM tagging by about a hundred candidates and
needs its own graded round.

### 4.2 Reproduction controls

* **Imaging** (`p98kchk`): today's imaging of the July frames equals `d27fresh` on 8/8 events, 16/16 archives each
  (`d99/imaging_drift_control.txt`).
* **Clustering + PR** (`p98kq`): today's clustering + PR, staged like production from `d27fresh` imaging, equals `d51vclus`
  and `p96vprod` on 8/8 events (`d99/clus_pr_drift_control.txt`):
  * pctree (member content);
  * `.tlas`;
  * all 280 + 63 branches of `tracking-pr/stm.root` (NaN-aware);
  * all 25 `mabc-pr.zip` members.
* **PR alone** (G0): identical.
* **Staging of the graded arms:** every `p98voffq` clustering sidecar (`pctree-evt*.tlas`) equals `d51vclus`'s token for
  token on 120/120 events, apart from the opflash path spelling. `readout_window_ticks=10000` on all 120, against 84 × 6400
  on the frames-in-dir pass (`d99/clus_tlas_vs_production.txt`).

So with production staging, the whole chain downstream of SP reproduces production byte for byte. It follows that:
- **every difference between `p98voffq` and `p96vprod` comes from rerunning SP**: July frames used v6 wires and the SP
  code/config of 07-13; today's use v7-uvwfit wires and today's SP;
- **every difference between `p98vonq` and `p98voffq` comes from the one constant.**

### 4.3 What rerunning SP alone does (p98voffq against production)

This is not the constant, but anyone flipping the constant pays it, because a flip means rerunning SP.

| | `p96vprod` (July SP) | `p98voffq` (today's SP, knob off) | record |
|---|---|---|---|
| events with an STM candidate / candidates | 119 / 596 | 119 / 597 | `d99/c_refit_*.txt` |
| is_stm with a usable chain | 265 | 255 | ″ |
| is_stm on the (carried) record: TP/FP/FN/TN | 242/8/34/262 | 177/19/44/154 | `d99/grade_p96vprod_p98voffq_p98vonq.txt` |
| is_stm purity / efficiency | 0.968 / 0.877 | 0.903 / 0.801 | ″ |
| over all hand stoppers: Bragg-path accept | 213/276 = 0.772 | 154/221 = 0.697 | ″ |
| over all hand stoppers: topology-rescued | 29/276 = 0.105 | 23/221 = 0.104 | ″ |
| Michel (all judged): purity / efficiency | 0.923 / 0.878 | 0.831 / 0.805 | ″ |
| top/bottom dQ/dx plateau, M2 bootstrap | 0.889 [0.875, 0.903] | 0.878 [0.858, 0.907] | `d99/g5b_space_charge_p96vprod.txt`, `d99/space_charge_p98voffq.txt` |
| recombination C refit (reported only) | 0.7919 ± 0.0047 | 0.7994 ± 0.0059 | `d99/c_refit_*.txt` |

The loss sits entirely in the **Bragg-path acceptance** (0.772 → 0.697). Topology rescue is flat (0.105 → 0.104). The tagger
reads the new SP's charge profiles with its dQ/dx shape tests and rejects stoppers it accepted on the July SP. The owner's
STM tagging knobs were tuned on the July SP.

The top/bottom offset itself barely moved: 0.889 → 0.878, inside doc pdhd/29's CI. Which of the SP differences (v6 → v7-uvwfit
wires, or the SP code/config since 07-13) moves the shape tests is **not separated here** (§10).

## 5. Carrying the hand-scan record

**Why it cannot be used as it is.** The 601-item record (`pdvd_stm_michel_smx1a_…_smx9_verdicts.json`) is keyed
`run_evt/cluster_id`, and its 577 tag sets are keyed by segment id `cluster*1000 + graph index`. Every arm it was scanned on
or graded against reads the one `d51vclus` clustering, so the ids never moved. Those arms are d53v, d67v, d68a3, d68d4,
p85v*, p88vprod, p90vprod and p96vprod. A rerun of SP + imaging + clustering renumbers clusters. The grader's `A.get(key)`
would then silently attach a verdict to whatever object now carries that number.

**What carries.** A verdict is a judgement about a particle (stopping muon, Michel, through-going), not about the numbering.
It carries wherever the same object is found again. `d99_match.py` finds it by geometry:

* **Baseline:** the key's cluster's image points in `p96vprod`'s Bee dump (`mabc-pr.zip` clustering-global), which is the
  clustering the scanner looked at.
* **Coverage fractions**, for every cluster of the new arm in the same event:
  * **f_fwd** = baseline points with a new-cluster point within 1.0 cm;
  * **f_rev** = new-cluster points with a baseline point within 1.0 cm.
* **x offset:** points sit at their bundle's t0-corrected x, so a changed flash match moves a whole cluster in x. If the best
  f_fwd < 0.7, an x offset is taken from (y,z)-nearest pairs and the match is rescored. The offset is reported, and
  |dx| > 1 cm is flagged.
* **Status**, with thresholds pre-registered before any new arm existed:
  * `ok`: f_fwd ≥ 0.7, second < 0.2, f_rev ≥ 0.5, best ≥ 2× second;
  * `split`, `merged`, `ambiguous`;
  * `collision`: two keys → one cluster;
  * `lost`.
* **Tags:** carried at **cluster** level (the companion cluster is matched the same way). Segment level is attempted and
  recorded but is not a carry criterion. The scan arms carried the survey TLA (companion segments fitted) and production
  does not, so even on `p96vprod` itself a Michel/gamma *segment* often has no fitted counterpart.
* **Pins:** the placed x/y/z are carried (x shifted by dx). `pin_rr` is turned into a point on the baseline profile, then
  into a new rr on the new fit.

**The carried record is a new file per arm** (`../scan/pdvd_stm_michel_smx9_carried_<arm>.json`), keyed by the new keys.
- Each item carries `carried_from` (its original key), `source_record`, `base_arm`, `target_arm`, `match{…}` and the tag
  map.
- Status `ok`, `merged` and `split` are carried. `lost` and `collision` are not.
- The smx record is untouched (M13).

**Re-scan list.** Every item a verdict should not be carried onto blindly goes to the re-scan list (`d99_rescan.py`). An
item is flagged for any of:
- a non-`ok` match;
- no STM candidate on the new arm;
- a stop moved > 5 cm;
- a moved t0;
- a MESSY/UNCLEAR/FRAG verdict;
- a Michel/gamma tag whose cluster did not remap;
- a pin > 3 cm from the new fit.

**Controls, before any new arm existed.**
- **Identity** (record → `p96vprod`): 601/601 `ok`, same key, f = 1.000.
- **Shuffled events:** 0 `ok`.
- **Identity re-scan baseline** (`d99/rescan_identity_p96vprod.tsv`): 62 items are flagged even with nothing changed:
  - 31 with no STM candidate on `p96vprod` (the same 31 `d25_bragg_michel` counts as "no candidate");
  - 29 MESSY/UNCLEAR/FRAG;
  - 4 Michel/gamma tags on an ambiguous companion cluster;
  - 1 pin.

  The arm lists report what is **new** against it.

**Result on the two arms:**

| | `p98voffq` | `p98vonq` | record |
|---|---|---|---|
| match status (601) | ok 565, merged 8, split 1, lost 23, collision 4 | ok 554, merged 13, split 2, lost 28, collision 4 | `d99/match_<arm>.txt` |
| top / bottom `ok` | 330 / 205 | 323 / 201 | ″ |
| `ok` f_fwd p5 / p50; f_rev p5 / p50 | 0.932 / 0.991; 0.922 / 0.990 | 0.931 / 0.991; 0.889 / 0.987 | ″ |
| items in the carried record | 574 | 569 | `../scan/pdvd_stm_michel_smx9_carried_<arm>.json` |
| flagged for re-scan / new vs identity | 254 / 229 | 271 / 249 | `d99/rescan_<arm>.txt`, `.tsv` |
| **cannot be carried without a look** (non-ok match or no candidate) | **193** (top 115, bottom 61) | **210** (top 124, bottom 68) | ″ |
| … by verdict | THRU 116, STM_MICHEL 41, STM_ONLY 24, MESSY 11, FRAG 1 | THRU 128, STM_MICHEL 39, STM_ONLY 25, MESSY 17, FRAG 1 | ″ |

**Answer to "can the hand scan still be used":** yes, for most of it.
- The geometric match finds the scanned object again for 92–94 % of the record (554–565 of 601 `ok`, median coverage 0.99
  both ways).
- The verdict travels with the object and is written into a new record keyed for the new arm.
- What does **not** carry is mostly a lost *candidate*, not a lost *object*. The object is there, but the new chain did not
  put it in front of the tagger. About three fifths of those are THRU verdicts (128 of 210 on `p98vonq`), which cost
  nothing if left unscanned.
- The stopper items among the 210 on `p98vonq` (STM_MICHEL 39, STM_ONLY 25, MESSY 17) are the ones a re-scan would need.

## 6. Grades: the constant (OFF → ON), with production for context

**Read OFF → ON for the constant.** `p96vprod` differs from both by the SP rerun (§4.3) and is context only.

| is_stm (TP/FP/FN/TN), purity, eff | `p96vprod` | `p98voffq` | `p98vonq` |
|---|---|---|---|
| all carried items | 242/8/34/262, 0.968, 0.877 | 177/19/44/154, 0.903, 0.801 | 180/15/42/147, 0.923, 0.811 |
| `ok` matches only | (same) | 177/19/44/154, 0.903, 0.801 | 180/15/42/146, 0.923, 0.811 |
| top: TP/FP/FN, eff | 168/3/23, 0.880 | 121/12/33, 0.786 | 128/8/32, 0.800 |
| bottom: TP/FP/FN, eff | 74/5/11, 0.871 | 56/7/11, 0.836 | 52/7/10, 0.839 |
| Bragg-path accept / topology-rescued (all hand stoppers) | 0.772 / 0.105 | 0.697 / 0.104 | 0.703 / 0.108 |
| Michel, all judged (TP/FP/FN), purity, eff | 144/12/20, 0.923, 0.878 | 103/21/25, 0.831, 0.805 | 102/28/28, 0.785, 0.785 |
| GOLDEN (Bragg-path + Michel) over hand Michel | 108/161 = 0.671 | 73/126 = 0.579 | 74/128 = 0.578 |

Records: `d99/grade_p96vprod_p98voffq_p98vonq.txt`, `d99/grade_all_okonly.txt`.

### 6.1 Movers OFF → ON, by original record key

- **Scope:** 337 items are graded on both arms; 57 are graded only on OFF and 47 only on ON (candidate present on one arm
  only).
- **is_stm, 32 moved:**
  - top hand stoppers: 12 gained and 12 lost;
  - top THRU: 3 newly accepted (FP) and 5 no longer accepted.
- **michel_found, 26 moved:**
  - top Michel: 5 gained and 10 lost;
  - top STM_ONLY: 4 newly found (FP);
  - top THRU: 5 newly found (FP) and 1 no longer found;
  - bottom: 1 (STM_ONLY 0 → 1).
- Every mover is listed by key in the grade record.

**Reading.**
- **is_stm:** OFF → ON is churn inside the statistical error (eff 0.801 → 0.811 ± 0.027, purity 0.903 → 0.923, 12 up / 12
  down). No census shift in either direction.
- **Michel:** a little worse (purity 0.831 → 0.785, 7 more FP), again within about 1.5σ. The Michel admission thresholds are
  charge thresholds fitted on the old top scale (§8).
- **Bottom:** essentially no movers. Bottom SP is identical (G3); its few changes come through cross-volume clustering and
  light matching.

### 6.2 The Michel energy closes cleanly

Median MeV [p16, p84], hand Michel items (`d99/michel_ql_p96vprod_p98voffq_p98vonq.txt`):

| `michel_ke_q2d_region` | top | bottom | top/bottom |
|---|---|---|---|
| `p96vprod` | 34.22 (n 96) | 36.85 (n 38) | 0.929 |
| `p98voffq` | 33.15 (n 66) | 41.10 (n 29) | 0.807 |
| `p98vonq` | 40.38 (n 65) | 41.10 (n 29) | **0.983** |

Per item, on the items carried onto both arms with a Michel found on both:
- **region energy:** ON/OFF **1.14 [1.10, 1.16]** on top (n 64) and **1.00** on bottom (n 33);
- **control-region energy:** 1.43 on top (small charges crossing absolute thresholds, §3) and 1.00 on bottom.

Light matching barely moved:
- top: 188 of 192 matched clusters keep their flash, and 4 have |Δt0| > 1 µs;
- bottom: 144 of 145 keep it, and 1 has |Δt0| > 1 µs.

## 7. dQ/dx closure against the pre-registration

**Pre-registered** (doc pdhd/29 plan, before any rerun):
- the top/bottom plateau on ON → 1.00, with a CI covering 1;
- ON/OFF top Σgauss ≈ 1.125 on every top plane, and bottom exactly 1.

This moves top onto **bottom**, not onto the absolute expectation. Bottom's ≈ 5 % excess over PDHD (doc pdhd/29 §8.2) is
untouched.

| | top plateau (n) | bottom plateau (n) | median ratio | **M2 A_top/A_bottom, bootstrap** | record |
|---|---|---|---|---|---|
| `p96vprod` | | | | 0.889 [0.875, 0.903] | `d99/g5b_space_charge_p96vprod.txt` |
| `p98voffq`, full sample | 0.9317 (102) | 1.0577 (32) | 0.8809 | 0.878 [0.858, 0.907] | `d99/space_charge_p98voffq.txt` |
| `p98vonq`, full sample | 1.0170 (103) | 1.0577 (30) | 0.9615 | **0.967 [0.953, 0.987]** | `d99/space_charge_p98vonq.txt` |
| `p98voffq`, tracks in both arms | 0.9219 (81) | 1.0577 (30) | 0.8716 | 0.874 [0.860, 0.890] | `d99/space_charge_p98voffq_common.txt` |
| `p98vonq`, tracks in both arms | 1.0330 (81) | 1.0577 (30) | 0.9767 | **0.972 [0.952, 1.000]** | `d99/space_charge_p98vonq_common.txt` |

- **Per track** (`d99/closure_pertrack.txt`): top ON/OFF **1.1228 [1.1176, 1.1275]** (81 tracks), bottom **1.0000** (30 of
  30).
- **Frame sums** (§3): bottom 1.0000 on every plane; top 1.1383 / 1.1331 / 1.1257. The second pre-registered item holds (U
  and V are about 1 % above, §3).
- **M3 − M2 loss:** the position-only model (M3) against the offset model (M2). OFF: +1.08 [+0.62, +1.65], so the offset is
  needed. ON: +0.06 [−0.09, +0.23], so an offset is no longer distinguishable from pure position dependence.

**Verdict: the pre-registered closure MISSES on the full sample.** 0.967 [0.953, 0.987] does not cover 1. s is not retuned.
The miss decomposes into three measured parts:

1. **The constant itself is exact.** The same top tracks gain ×1.1228 against 1/s = 1.1249, and bottom tracks do not move.
2. **Today's SP starts lower than the SP s was measured on.** OFF reads 0.878, not 0.889 (§4.3). So even a perfect constant
   predicts 0.878 × 1.125 = 0.988 on the full sample and 0.874 × 1.123 = 0.981 on the common tracks, not 1.00.
3. **The track sample changes.**
   - ON gains 22 top tracks and loses 21. The gained tracks sit lower (median plateau 0.989) than the common ones on ON
     (1.033).
   - That pulls the full-sample ratio from 0.977 (common) to 0.962 (full).
   - The M2 fit runs on the full sample.

On the tracks present in both arms, ON reads **0.972 [0.952, 1.000]**. That interval covers both 1.00 and the mechanical
expectation 0.981. The residual top-below-bottom there is 2–3 %, and it is not significant at this sample size.

## 8. Downstream constants: reported, not applied

Every constant fitted on reconstructed top charge was fitted on the old top scale, and none is changed here:

* **Recombination C** (`d16_stm_energy_scales.py`, record-free, range vs dQ/dx energy on is_stm chains):
  * fit C = 0.7919 ± 0.0047 (`p96vprod`), 0.7994 ± 0.0059 (`p98voffq`), **0.8630 ± 0.0077** (`p98vonq`); the chain runs
    with 0.7941;
  * at the chain's C, the dQ/dx-energy / range-energy median is 1.0104 on OFF and **1.1173** on ON;
  * so a flip without a C refit makes the chain's dQ/dx energies read about 11 % high overall (top-weighted);
  * ON/OFF C is 1.080. That is the top ×1.125 diluted by the bottom tracks, which do not move.
* **Michel region energy:** top rises 33.15 → 40.38 MeV (§6.2). Any energy window on the Michel was set on the old scale.
* **Michel / capture-gamma admission and the tagger's absolute charge thresholds:** the Michel purity drop (§6) is the
  visible symptom.
* **Charge-light (QtoL):** flash assignments are nearly unchanged (§6.2), but the charge-to-light ratio used in matching was
  set on the old top scale and is not refit.
* **Simulation:** `params.elecs` is untouched, so simulated top charge is still produced and reconstructed at the old
  (self-consistent) scale. The constant is a data-only correction as built.

Refitting any of these is the owner's decision, after the flip decision.

## 9. Disk: OFF trim done; the previous round's SP frames HELD

Owner: trim the control arm's frames after imaging; keep the latest SP results; retire the previous round's.

**OFF trim — DONE** (`d99_trim_off.sh`).
- **Scope:** only `work/*_p98voff/protodune-sp-dnnroi-frames-anode*.tar.bz2`. The six gate events are kept: 039252_0,
  039252_2, 039253_0, 039253_1, 039349_0, 039349_10.
- **Guard:** the whole trim is refused unless every archive has its manifest line, its event's G3 line says PASS, and the
  event's imaging is complete.
- **Before deleting:**
  - dry run: 912 archives, 36.75 GiB;
  - negative control (one manifest line withheld): refused, rc 14;
  - run only after the 120-event frame sums, which read both arms.
- **Executed:** 912 of 912 deleted, 36.75 GiB freed. 48 OFF archives remain (6 events × 8), and all 960 ON archives are
  intact.
- **Recoverability:** SP is bit-deterministic (G2, and the rc=2 reruns), and the manifests hold every deleted archive's
  member hashes. Any OFF event can be regenerated and checked.

**The July SP frames — HELD, not executed** (`d99_retire_keep_sp.sh`). The frames are
`/home/xqian/pdvd-frame-store/*_keep/protodune-sp-dnnroi-frames-anode*` (v6-wire SP of 07-13).

Dry run with the final keep list (`d99/retire_keep_sp_dryrun.txt`):
- 768 archives, 29.37 GiB, and 1536 links pointing at them, closure reached in 2 passes;
- 24 events kept:
  - 039252_0..17 and 039253_0, which a committed script or doc Repro reads (doc qlmatch/18, doc nf_sp_img_clus/28,
    `check_clus97_tail_waveforms.py`, doc pdhd/29's `d29_gain_recomb.py`);
  - 039349_2/4/15/68/81, this doc's own imaging and clustering controls (`p98kchk`, `p98kq`);
- the census of cited events passes;
- negative control (an ON-arm frame injected): refused, rc 13;
- liveness guard: refused while PR was running, rc 11.

**Why it is held.** The owner's instruction came before §4.3 existed. The July frames are the only surviving copy of the SP
that production (`p96vprod`) and every hand scan were built on. §4.3 shows that rerunning SP alone costs 0.07 in is_stm
efficiency. Separating v6 wires from the SP code/config needs exactly these frames as the reference (§10 item 2). Deleting
them is irreversible, and disk is not forcing it: `/home/xqian` has 385 G free.

The script and dry run are ready. `pdvd/scripts/retire/PROTECTED.txt`'s `keep` line ("it holds the SP+DNNROI frames d27fresh
borrows") is untouched. **The owner decides.**

## 10. Not concluded / next

1. **Flip decision (owner).** For the constant:
   - charge closes per track (1.1228 of 1.1249);
   - the Michel energy top/bottom goes 0.807 → 0.983;
   - is_stm is unchanged within churn against OFF.

   Against flipping now: a flip requires rerunning SP, and today's SP alone costs is_stm efficiency 0.877 → 0.801 against
   production (§4.3). **Recommended: do not flip until item 2 is understood.**
2. **Split the SP-rerun loss** (§4.3):
   - rerun SP on a handful of events with the v6 wires and today's SP config, then the reverse;
   - compare Bragg-path acceptance against the July frames (held, §9) and `p98voffq`.
3. **Re-scan** the stopper items that cannot be carried without a look on the arm that is eventually adopted: on `p98vonq`,
   STM_MICHEL 39 + STM_ONLY 25 + MESSY 17 (`d99/rescan_p98vonq.tsv`).
4. **Refit the downstream constants** for a flipped arm (§8): C (0.8630 on `p98vonq`), Michel thresholds, QtoL.
5. **Readout window** (§4.1): production's edge guard sees 10000 ticks over 6400-tick frames.
6. **Candidates rose on ON:** 597 → 656 (events with a candidate 119 → 120). Not traced here. The per-key grades above use
   only the record's items.
7. **The 2–3 % residual** top-below-bottom on the common tracks (0.972 [0.952, 1.000]) is not significant here; a larger
   stopper sample would tell.

## Files

| file | what |
|---|---|
| toolkit `cfg/pgrapher/experiment/protodunevd/sp.jsonnet`, `l1sp_after_dnnroi.jsonnet` | `top_gain_scale` knob, default 1.0 |
| `pdvd/wct-nf-sp-dnnroi.jsonnet`, `pdvd/run_nf_sp_dnnroi_evt.sh` | TLA + `--top-gain-scale` |
| `scripts/d99_sp_arm.sh`, `d99_chain.sh`, `d99_stage_q.sh` | SP arm driver; imaging → clustering → PR driver; production-style clustering staging |
| `scripts/d99_frames.py`, `d99_manifest.py` | per-plane frame sums and L1SP share; frame manifests + G3 |
| `scripts/d99_control_compare.py` | imaging / clustering / PR drift controls (pctree, tlas, NaN-aware trees, mabc) |
| `scripts/d99_match.py`, `d99_rescan.py` | geometric carry-over of the record; re-scan list |
| `scripts/d99_grade.py`, `d99_dqdx_drift.py` | grading and dQ/dx forks with their controls |
| `scripts/d99_closure.py` | per-track OFF → ON plateau ratio, common-track samples |
| `scripts/d99_readout_window.py` | §4.1: candidates, edge-guard firings and clustering window per arm |
| `scripts/d99_michel_ql.py` | Michel energy by volume, per-item ON/OFF, flash changes |
| `scripts/d99_trim_off.sh`, `d99_retire_keep_sp.sh` | guarded OFF-frame trim; guarded retire of the July SP frames (held) |
| `../scan/pdvd_stm_michel_smx9_carried_p98voffq.json`, `_p98vonq.json` | the carried hand-scan records (new keys; the smx record untouched) |
| `d99/` | gate records and every number in this doc |
