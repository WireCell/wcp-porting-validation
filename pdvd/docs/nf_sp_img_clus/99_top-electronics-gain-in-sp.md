# 99 — One gain constant for the PDVD top electronics in SP, a full rerun, and the hand-scan record carried onto it

**Status (2026-09-13):**
- **Knob:** shipped **default OFF**. The compiled production config is byte-identical (G1).
- **Rerun and carry:** the full SP → imaging → clustering → PR rerun is done for OFF and ON, and the hand-scan record is
  carried onto both.
- **Charge:** the constant does exactly what it was built to do. The same top track gains ×1.1228 [1.1176, 1.1275]
  (1/s = 1.1249), and the bottom is unchanged at 1.0000 on 30/30 tracks.
- **Pre-registered closure: MISSED on the full sample.** Top/bottom is **0.967 [0.953, 0.987]**, a CI that does not cover
  1. On the 111 tracks present in both arms it reads 0.972 [0.952, 1.000] (§7). s was not retuned.
- **Round 2 correction (§4.3): nothing in SP drifted since July except the wire file.**
  - Today's SP code, run with the July wire file, reproduces the July frames sample for sample.
  - Production ran SP with the v5 wire order and everything after SP with v7-uvwfit. The rerun arms use v7-uvwfit
    throughout.
  - The v7 channel order redistributes a few percent of U/V charge between neighbouring channels; plane sums stay within
    1 %.
  - Round 1's "rerunning SP costs is_stm efficiency 0.877 → 0.801" overstated it. The tagged population keeps its size
    (265 / 255 / 273 is_stm), but about 30 % of the tagged clusters swap in both directions, and the record sees only the
    losses (§0, §6.3).
- **Disk:** the OFF trim is done. The July SP frames are **retired** (owner yes, 2026-09-13: 768 archives, 29.4 GiB, a
  sha256 manifest first); they regenerate bit for bit with `--wires` (§9).
- **Round 3 (§6.4): the newly tagged side was scanned blind.** Three blind agent scanners, one frozen rubric, 91 clusters
  `p98vonq` tags that production does not, and 45 seeded controls production tags that `p98vonq` does not:
  - purity **0.846 [0.801, 0.883]** on the newly tagged side against **0.907 [0.853, 0.942]** on the dropped side;
  - difference −0.061, 95 % [−0.174, +0.060]: **purity-neutral within this sample** by the pre-registered reading, with
    the point estimate on the low side;
  - the newly tagged side holds three times as many objects no one can call (MESSY/UNCLEAR 14 % against 4 %);
  - half of its non-stoppers end at the frame edge. On run 039349, clustering and PR assume a 10000-tick window over
    6400-tick frames on both arms (§4.1).
- **Round 4 (§4.4): each event's real readout window, on both arms.**
  - Only run 039349's frames are 6400 ticks; runs 039252 and 039253 are 10000.
  - With the real window, the PR edge guard untags 9 of the 136 scanned objects, all in run 039349: THRU 5, UNCLEAR 2,
    STM_ONLY 2.
  - Purity on the scanned objects that stay reads **0.889 against 0.929** (−0.040, 95 % [−0.143, +0.073]). The window
    explains about a third of the gap.
  - Where the window does not change, all four new arms reproduce their sources byte for byte.
  - The new runner override `PDVD_READOUT_NTICKS` is unset by default; production is unchanged.

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

## 0. Round 2: the owner's three points

Owner, 2026-09-13 (round 2):

> "What we want to demonstate that with latest configuration, 1. everything are good 2. the charge asymmetry between top
> and bottom are fixed. 3. I assume the STM+Michel are largely staying the same as before. Please confirm."

**The latest configuration** is arm `p98vonq`: v7-uvwfit wires in SP *and* in imaging, clustering and PR, plus
`top_gain_scale=0.889`. Production (`p96vprod`) differs in two ways. Its SP frames predate v7 (§4.3), and it has no
constant.

**1. Everything is good: confirmed.**
- With the knob off, the compiled production config is md5-identical (G1). It is still identical after round 2's
  `wires_file` TLA (`d99/g1_wires_file_compiled_config.txt`).
- With the knob on:
  - exactly 20 config values change;
  - bottom SP frames are identical on 120/120 events;
  - top frames scale ×1.1249 per channel.
- Every step after SP reproduces production byte for byte (§4.2).
- Production's SP frames are reproduced sample for sample by today's code with the July wire file (§4.3). Nothing in SP,
  NF or DNN-ROI changed since July except the wire file, which imaging had already adopted on 09-03.
- **Two open issues predate this change and are not caused by it:**
  - production's SP frames carry the v5 wire order while its imaging, clustering and PR use v7-uvwfit (§4.3 item 4);
  - on run 039349, production's PR edge guard runs a 10000-tick window over 6400-tick frames (§4.1, §4.4).

  The latest-configuration arms fix the first; the second is unchanged in every arm.

**2. The top/bottom charge asymmetry is fixed: confirmed for the charge, with one stated miss.**
- **Per track:** top ×1.1228 [1.1176, 1.1275] against 1/s = 1.1249; bottom 1.0000 on 30/30.
- **Michel region energy**, top/bottom on hand Michels: 0.807 → **0.983**.
- **Stopper dQ/dx plateau**, top/bottom:
  - full sample: 0.878 → **0.967 [0.953, 0.987]**. The pre-registered test (CI covers 1) misses.
  - the 111 tracks in both arms: **0.972 [0.952, 1.000]**. The remaining 2–3 % is not significant there.
- s was not retuned (§7).

**3. STM+Michel largely stay the same: confirmed for the population, not for individual clusters.**

Record-free census (`d99/population_is_stm_transitions.txt`, `d99/michel_ql_p96vprod_p98voffq_p98vonq.txt`):

| | production `p96vprod` | latest, knob off `p98voffq` | **latest `p98vonq`** |
|---|---|---|---|
| STM candidates | 596 | 597 | 656 |
| is_stm (top / bottom) | 265 (184 / 81) | 255 (171 / 84) | **273 (189 / 84)** |
| is_stm with a Michel | 147 | 142 | **163** |
| Michel region energy, all chain STM+Michel, top / bottom median MeV | 34.22 / 36.00 | 32.24 / 40.76 | 36.35 / 39.96 |

- **The population keeps its size.** The latest configuration tags 3 % more stoppers than production and 11 % more with a
  Michel.
- **Individual clusters swap, symmetrically.**
  - Of production's 265 is_stm clusters, 183 (0.691) are is_stm on `p98vonq`.
  - Of `p98vonq`'s 273, 182 (0.667) were is_stm in production.
  - The same swap appears between production and knob-off (0.694 / 0.722) and between knob-off and knob-on
    (0.784 / 0.725).
- **Floor and control.**
  - Bottom clusters whose SP input is identical still swap 6–7 % (OFF → ON bottom 0.940). That is the floor from
    cross-volume clustering and light matching.
  - Production → production gives 1.000 (null control).
  - A quarter to a third of tagged clusters sit close enough to a threshold that a few-percent change in the input flips
    them, in both directions.
  - For the wire-order change the threshold is mostly upstream of the tagger, in clustering or candidate selection (§6.3).
- **Why the hand-scan record reads a drop.** The record holds production's candidates, so it sees clusters that stop being
  tagged but not clusters that start. That is why record-based is_stm efficiency reads 0.877 (production) → 0.801 (off) →
  0.811 (on) while the tagged population does not shrink.
- **What the record does measure on the scanned items:** purity 0.968 / 0.903 / 0.923.
- **Measured in round 3 (§6.4):** the purity of the clusters tagged only on the latest configuration, by a blind scan of
  both sides of the swap. It is 0.846 against 0.907 on the side production alone tags (difference −0.061, 95 %
  [−0.174, +0.060], purity-neutral within the sample). The newly tagged side carries more unjudgeable objects (14 % against
  4 %), and half of its non-stoppers sit at the frame edge of §4.1. With those items set aside (post hoc) the two sides
  read 0.912 and 0.943.

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

# sec 4.4: each event's real readout window (d99/frame_nticks_120evt.txt, from the p98von frames)
WAVE=guard ARM=p99rgon   SRC=p98vonq  ./d99rw_arms.sh     # PR only: source pctree linked, .tlas window set per event
WAVE=guard ARM=p99rgprod SRC=p96vprod ./d99rw_arms.sh
WAVE=full  ARM=p99rwnul  SRC=p98von EVENTS="039349_7" NOWIN=1 ./d99rw_arms.sh   # PDVD_READOUT_NTICKS unset control
WAVE=full  ARM=p99rwon   SRC=p98von   ./d99rw_arms.sh     # clustering + PR, PDVD_READOUT_NTICKS per event
WAVE=full  ARM=p99rwprod SRC=d27fresh ./d99rw_arms.sh
python3 d99rw_identity.py --arm p99rwnul --base p98vonq --nt all --events 039349_7   # d99/rw_identity_p99rwnul.txt
python3 d99rw_identity.py --arm <arm> --base <source> --nt 10000       # d99/rw_identity_<arm>_10000.txt (arm = p99rgon p99rgprod p99rwon p99rwprod)
python3 d99rw_identity.py --arm p99rwon --base p98vonq --nt 6400       # d99/rw_identity_p99rwon_6400.txt; likewise p99rwprod vs p96vprod
python3 d99rw_identity.py --arm p99rwon --base p99rgon --nt 6400       # d99/rw_identity_p99rwon_vs_p99rgon_6400.txt; likewise prod
python3 d99rw_census.py --arms p96vprod p99rgprod p99rwprod p98vonq p99rgon p99rwon   # d99/rw_census.txt
python3 d99rw_regrade.py --on p99rgon --prod p99rgprod                 # d99/rw_regrade_guard.txt
python3 d99rw_regrade.py --on p99rwon --prod p99rwprod                 # d99/rw_regrade_full.txt

# round 2 (sec 0, 4.3, 6.3): July vs today's frames, the wire-file split, the record-free population
python3 d99_frames.py --events 039252_0 039252_2 039253_0 039253_1 039349_0 039349_10 --arms keep p98voff   # d99/frames_july_vs_p98voff_6evt.txt
python3 d99_frames.py --all --arms keep p98von                                                             # d99/frames_july_vs_p98von_120evt.txt
python3 d99_frame_identity.py --base keep --arm p98voff --events 039252_0 039252_2 039253_0 039253_1 039349_0 039349_10
python3 d99_frame_identity.py --base keep --arm p98von --all --anodes 0 1 2 3
(cd ../../.. && setarch x86_64 -R ./run_nf_sp_dnnroi_evt.sh --wires protodunevd-wires-larsoft-v5.json.bz2 -O _p99w5 039252 0)
(cd ../../.. && setarch x86_64 -R ./run_nf_sp_dnnroi_evt.sh --wires protodunevd-wires-larsoft-v6.json.bz2 -O _p99w6 039252 0)
(cd ../../.. && setarch x86_64 -R ./run_nf_sp_dnnroi_evt.sh --wires protodunevd-wires-larsoft-v6.json.bz2 -O _p99w6 039349 83)
python3 d99_frame_identity.py --base keep --arm p99w5 --events 039252_0          # likewise keep/p99w6, p99w5/p99w6, p98voff/p99w5; keep/p99w6 on 039349_83
python3 d99_population.py --pairs p96vprod:p96vprod p96vprod:p98voffq p98voffq:p96vprod p98voffq:p98vonq p98vonq:p98voffq p96vprod:p98vonq p98vonq:p96vprod
# wire order v5/v6/v7: d99/wire_channel_order_v5_v6_v7.txt, d99/wire_neighbour_change_v6_v7.txt (inline python over the three wire files)

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
./d99_retire_keep_sp.sh ; NEGCTL=1 ./d99_retire_keep_sp.sh                 # dry run + negative control
CONFIRM=1 ./d99_retire_keep_sp.sh    # owner yes 2026-09-13 -> d99/retire_keep_sp_confirm.txt, d99/retire_keep_sp_manifest.txt

# round 3 (sec 6.4): the blind swap scan; R=/home/xqian/tmp/p99scan (scratch: preps, frames, scanner out dirs)
C=../../../../pdhd/stm_michel_scan
for A in p98vonq p96vprod; do (cd $C && ./prep_stm_michel_scan.py --det pdvd --arm $A --redraw --ctx-cells \
    --outdir $R/prep_$A --sheetdir $R/sheet_$A); done                          # scratch only; no label, no repo sheet
python3 d99_swap_scan_set.py --on p98vonq --prod p96vprod --n-controls 45 --seed 20260913 --out $R/set \
    --prep-sheet p98vonq $R/sheet_p98vonq/pdvd_stm_michel_scan_sheet.tsv \
    --prep-sheet p96vprod $R/sheet_p96vprod/pdvd_stm_michel_scan_sheet.tsv   # -> d99/swap_scan_key.tsv
bash $C/campaign/shoot.sh $R/shoot_p98vonq pdvd $R/set/sheet_p98vonq.tsv $R/prep_p98vonq 3   # then p96vprod with 2
python3 $C/campaign/mkzoom.py $R/shots; python3 $C/check_shots.py $R/shots        # -> d99/swap_scan_check_shots.txt
python3 $C/campaign/nextwave.py $R $R/set/items_all.txt w1 --agents 3 --per 23   # w2 after every w1 report; rubric
                                  # d99/swap_scan_rubric.md, task d99/swap_scan_agent_task.md, reports d99/swap_scan_reports/
python3 d99_swap_scan_score.py --round $R --key ../d99/swap_scan_key.tsv \
    --record-out ../../scan/pdvd_stm_michel_sw99_verdicts.json > ../d99/swap_scan_score.txt
python3 d99_swap_scan_score.py --round $R --key ../d99/swap_scan_key.tsv \
    --exclude-keys ../d99/swap_scan_window_flagged.tsv > ../d99/swap_scan_score_window_excluded.txt   # post hoc
python3 d99_latest_record.py --carried ../../scan/pdvd_stm_michel_smx9_carried_p98vonq.json \
    --sw99 ../../scan/pdvd_stm_michel_sw99_verdicts.json \
    --out ../../scan/pdvd_stm_michel_p98vonq_carried_sw99_verdicts.json > ../d99/latest_record_merge.txt
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
| **G1w** `wires_file` TLA (round 2) | production entry with the runner's TLAs: md5 with the new TLA absent or empty = G1's (L1SP on 412e5252…, L1SP off d944ec94…); with `wires_file` = v5 or v6: exactly 8 lines change, the `WireSchemaFile` filename | **PASS** | `d99/g1_wires_file_compiled_config.txt` |
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
- The `p98voff` directories hold the SP frames themselves, so every event ran with its frames' real length: **6400** ticks
  on the 84 events of run 039349 and 10000 on the 36 events of runs 039252 and 039253 (`d99/frame_nticks_120evt.txt`; the
  July frames regenerate sample for sample, §4.3). *Corrected in §4.4: an earlier version said all frames are 6400 ticks.*
  Production's 10000 fallback is wrong on run 039349 only.
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

**Open for the owner** (not changed here; measured in §4.4): on run 039349 production's edge guard runs on a 10000-tick
window over 6400-tick frames, so it cannot fire at the real readout edge. Enabling the real window would move STM tagging by about a hundred candidates and
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
- **every difference between `p98voffq` and `p96vprod` comes from rerunning SP**. §4.3 shows this means the wire file
  alone: the July frames carry the v5 wire order, today's the v7-uvwfit order;
- **every difference between `p98vonq` and `p98voffq` comes from the one constant.**

### 4.3 What differs between production's SP and today's (p98voffq against production)

**Correction to round 1.** Round 1 said here that "rerunning SP alone costs STM efficiency", and it left open whether the
wires or the SP code/config were the cause. Round 2 measured it.

**1. The July frames carry the v5 wire order.**
- The July SP arm (`_ct4`, its frames later linked as `<evt>_keep`) started 07-13 at 14:21.
- `protodunevd-wires-larsoft-v6.json.bz2` entered wire-cell-data at 14:35, and `params.jsonnet` switched to it at 14:43
  (toolkit `e4eda3c2`).
- 25 of the 120 July jobs started before the v6 file existed; 95 started after (`d99/july_sp_job_start_times.txt`).
- The per-plane wire order of v5 and v6 is the same (item 4), and SP gives identical frames with either (item 3). So all
  July frames carry the v5 order. Round 1's "v6 wires" was wrong in name only.

**2. The frames barely move.** Comparing July and today's knob-off frames, content member by member:
- **Records:** `d99/frame_identity_july_vs_p98voff_6evt.txt`; bottom anodes on all 120 events
  `d99/frame_identity_july_vs_p98von_bottom_120evt.txt`.
- **W:** sample-identical on anodes 0, 1, 4, 5; on anodes 2, 3, 6, 7 it differs by at most 6.2e-3 sum|Δ|/sum (median ≤ 2.1e-4).
- **U/V:** on anodes 1–7 they differ by a median 0.9–12 % sum|Δ|/sum; anode 0 is identical or nearly so.
- **Plane sums:** still agree to within about 1 % (0.997–1.010), with a per-channel median ratio of 1.0000 (`d99/frames_july_vs_p98voff_6evt.txt`,
  `d99/frames_july_vs_p98von_120evt.txt`).

Charge is moved between neighbouring channels, not gained or lost.

**3. The wire file is the whole difference.**
- `run_nf_sp_dnnroi_evt.sh --wires <file>` (new TLA `wires_file`, default off; the compiled production config is unchanged,
  `d99/g1_wires_file_compiled_config.txt`) runs today's SP with any wire file.
- **Match:** on 039252_0, today's SP with the v5 wires, and again with the v6 wires, reproduces the July frames **sample
  for sample on all 8 anodes and every frame tag** (raw, wiener, gauss). Records: `d99/frame_identity_keep_vs_p99w5_039252_0.txt`
  and `…_keep_vs_p99w6_039252_0.txt`; v5 vs v6 is identical too.
- **After the switch:** 039349_83, a July job that started at 14:58 (after v6 existed), is also reproduced sample for sample
  by today's SP with the v6 wires (`d99/frame_identity_keep_vs_p99w6_039349_83.txt`).
- **Contrast:** the same run differs from today's v7 frames in exactly the pattern of item 2 (`…_p98voff_vs_p99w5_039252_0.txt`).
- **Conclusion:** no change to the SP code, the SP config, the DNN-ROI model or NF since July reaches these frames.

**4. What v7-uvwfit changes for SP:** the per-plane wire order that OmnibusSigProc's 2-D deconvolution and NF's channel
groupings run over (`d99/wire_neighbour_change_v6_v7.txt`, `d99/wire_channel_order_v5_v6_v7.txt`).
- **Anodes 2, 3, 6, 7:** face 0 and face 1 exchange their channel sets (the face assignment doc pdvd/27 corrected). 189–190
  of 286 U/V neighbour pairs in a face are new. W, deconvolved channel by channel, moves by at most 6e-3.
- **Anodes 0, 1, 4, 5:** some planes are listed in reverse, and one wire is added per plane, with one new neighbour pair.
  U/V output still changes by a few percent on anodes 1, 4 and 5.
- **v5 → v6** moved U/V wire positions only, not the order.
- **The mixed geometry:** production's imaging, clustering and PR already use v7-uvwfit (adopted 09-03, `228f1c39`), on SP
  frames made with the v5 order. The rerun arms are the first with one geometry throughout.

**5. What that does downstream.**

| | `p96vprod` (SP with the v5 wire order) | `p98voffq` (SP with v7-uvwfit, knob off) | record |
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

**On the record:** the record-based loss sits in the **Bragg-path acceptance** (0.772 → 0.697), and topology rescue is flat
(0.105 → 0.104).

**Record-free** (§6.3):
- the tagged population keeps its size: candidates 596 → 597, is_stm 265 → 255, with a Michel 147 → 142;
- 31 % of production's tagged clusters are not tagged on `p98voffq`, and 28 % of `p98voffq`'s were not tagged in production.

**Reading:** a few-percent redistribution of U/V charge changes *which* clusters are tagged, in both directions; it does
not remove stoppers wholesale.
- **Most of this churn is upstream of the tagger.** Of the 81 production is_stm clusters not tagged on `p98voffq`, 44
  are not even STM candidates there, meaning clustering or candidate selection changed. Only 33 reach the tagger and fail
  its tests.
- **The record counts one side.** It was built from production's candidates, so it counts only the clusters that leave.
- **Tuning:** the owner's STM tagging knobs were tuned on the v5-order SP.

The top/bottom offset itself barely moved: 0.889 → 0.878, inside doc pdhd/29's CI.

### 4.4 Round 4: each event's real readout window, on both arms

**Frame length depends on the run** (`d99/frame_nticks_120evt.txt`, read from the `p98von` frames; the July frames
regenerate sample for sample, §4.3):

| runs | events | SP frame length | production staging's window |
|---|---|---|---|
| 039252, 039253 | 36 | 10000 ticks | 10000, correct |
| 039349 | 84 | 6400 ticks | 10000, wrong |

An earlier version of §4.1 and §6.4 said every frame is 6400 ticks; only run 039349's are.

**Arms** (`scripts/d99rw_arms.sh`; per event, `setarch -R`, pin `libpin_p96` Clus `4e1db810` before and after):

| arm | built from | what can move |
|---|---|---|
| `p99rgprod`, `p99rgon` | the `p96vprod` / `p98vonq` pctree linked, its `.tlas` copied with the event's window; PR only | PR's `readout_edge_guard`; clustering and cluster ids are the source's |
| `p99rwprod`, `p99rwon` | the `d27fresh` / `p98von` imaging, staged like production (`d99_stage_q.sh`), clustering with `PDVD_READOUT_NTICKS` set per event, then PR | also clustering, through QLMatching's window-truncation flag |
| `p99rwnul` | `p98von`, `039349_7`, `PDVD_READOUT_NTICKS` unset | nothing (control) |

`PDVD_READOUT_NTICKS` is a new, default-unset override in `run_clus_evt.sh`. Production and every existing arm are
unchanged.

**Gates.**
- **Knob unset:** `p99rwnul` equals `p98vonq` on `039349_7` in pctree, `.tlas`, all 343 PR branches and 25/25 `mabc-pr`
  members (`d99/rw_identity_p99rwnul.txt`).
- **Where the window does not change** (the 36 events with 10000-tick frames), all four arms reproduce their source byte
  for byte: 36/36 each (`d99/rw_identity_{p99rgprod,p99rgon,p99rwprod,p99rwon}_10000.txt`).
- **Completion:** 119/120 on every arm, judged by output, and the two missing events are different cases.
  - `039252_11` on the production arms has no STM candidate, as on `p96vprod` (10000-tick frames, nothing changed).
  - `039349_30` on the latest arms is a result: it had 2 candidates on `p98vonq`, and the real window rejects both (stops
    at ticks 6342.6 and 6352.8). It is the only event the window empties, so PR writes no candidate summary line.

**On run 039349, the window acts almost only through the PR guard.**
- **Clustering hardly moves.** The pctree changes on 2 of 84 events on the latest arm (`039349_32`, `039349_62`) and on
  1 on production (`039349_62`), each through a different flash match (`flash_id`, `cluster_t0`). Everywhere else the
  re-clustered arm equals the PR-only arm (82/84 and 83/84, `d99/rw_identity_p99rw*_vs_p99rg*_6400.txt`).
- **The guard does the rest** (`d99/rw_census.txt`, re-clustered arms; runs 039252 and 039253 identical):

  | run 039349, 84 events | production: 10000 → real window | latest: 10000 → real window |
  |---|---|---|
  | `readout_edge_guard` firings | 138 → 243 | 174 → 305 |
  | STM candidates | 409 → 314 | 450 → 334 |
  | is_stm | 159 → 148 | 167 → 153 |
  | is_stm with a Michel | 83 → 81 | 89 → 86 |

**The §6.4 blind scan, re-graded on the real-window pair** (`d99/rw_regrade_full.txt`). All 136 scanned objects carry
by geometry. The PR-only pair gives the same fates and the same purity numbers (`d99/rw_regrade_guard.txt`); the guard
line differs on one item (`039349_32/48`, below).
- **Swap sets:** on_only 91 → 84, prod_only 82 → 78. All of the change is in run 039349.
- **9 scanned objects leave their set, all untagged on their own arm; none leaves because the other side now tags it.**
  - Newly tagged side, 8 of 91: THRU 4 (`039349_22/51`, `_26/23`, `_32/48`, `_54/20`), UNCLEAR 2 (`039349_40/48`,
    `_69/44`), STM_ONLY 2 (`039349_7/67`, `_75/63`, both `medium`).
  - Control side, 1 of 45: THRU (`039349_59/14`).
  - Each is rejected by the guard with its stop at ticks 6344–6388, inside the last 60 ticks. The exception is
    `039349_32/48` on the re-clustered arm, where its event's flash match moved; on the PR-only arm the guard rejects it at
    tick 6387.7.
- **The 19 frame-edge items:**
  - 7 leave, all in run 039349. The 5 in runs 039252 and 039253 stay, as they must.
  - Run 039349's other 7 stay tagged. For the two checked, the tagger's stop lies 143 (`039349_69/23`) and 256
    (`039349_82/41`) ticks before the edge.
  - The regex had missed two edge items, which also leave (`039349_54/20`, `039349_69/44`).
- **One unscanned object enters on_only:** `039349_47/76`. The same object stops at tick 6321 on the latest arm and at
  6388 on production, 67 ticks apart across the 60-tick guard.

**Purity on the scanned objects that stay** (not pre-registered: the sets changed):

| | newly tagged side | control side | difference [95 %] |
|---|---|---|---|
| §6.4, pre-registered (10000-tick window everywhere) | 0.846 [0.801, 0.883] (66/78) | 0.907 [0.853, 0.942] (39/43) | −0.061 [−0.174, +0.060] |
| real window, scanned objects that stay | **0.889** [0.846, 0.921] (64/72) | **0.929** [0.878, 0.959] (39/42) | **−0.040** [−0.143, +0.073] |

Not judged: the 1 object new to on_only and the 34 unsampled members of prod_only (as in §6.4).

**Reading.**
- **The real window removes mostly what the scanners called non-stoppers or could not call:** 7 of the 9 leavers.
- **It closes about a third of the §6.4 gap** (−0.061 → −0.040). The rest is not the window.
- **The cost is 2 `medium` STM_ONLY calls.** Their stop lies in the last 60 ticks, where a stop cannot be told from a
  truncation by construction.
- **The guard's 60 ticks is a knife edge:** an object whose stop moves by tens of ticks between arms crosses it
  (`039349_47/76`).
- **Not a production change.** Production staging still runs 10000 on run 039349. Adopting the real window (by setting
  `PDVD_READOUT_NTICKS` per run, or by staging the frames) is the owner's call.
  - It moves run 039349's is_stm by −11 on production and −14 on the latest configuration, and its STM candidates by
    about a hundred.
  - **Most of those removals are unjudged.** The swap scan judged 1 of production's 11 and 8 of the latest
    configuration's 14. The rest were tagged on both arms, so the scan never drew them, and whether removing them is a
    gain or a loss is not measured here. Some carry a verdict in the carried record; that was not examined.
  - Adopting the window therefore wants a look at run 039349's removed tags, not just the knob.

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

**Read OFF → ON for the constant.** `p96vprod` differs from both by the SP wire order (§4.3). The record-based numbers below
count only production-scanned clusters, so read them together with the record-free census in §6.3.

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

### 6.3 The STM population without the record

`scripts/d99_population.py` → `d99/population_is_stm_transitions.txt`.

**Method.** Every is_stm cluster of arm A is found in arm B by geometry, with the record carry's own matcher and
thresholds. It is then classified by what B's tagger says about the matched cluster.

| A is_stm → B | n | stays is_stm | candidate, not is_stm | not a candidate | object not matched |
|---|---|---|---|---|---|
| `p96vprod` → `p96vprod` (null) | 265 | 265 (1.000) | 0 | 0 | 0 |
| `p96vprod` → `p98voffq` | 265 | 184 (0.694) | 33 | 44 | 4 |
| `p98voffq` → `p96vprod` | 255 | 184 (0.722) | 19 | 47 | 5 |
| `p98voffq` → `p98vonq` | 255 | 200 (0.784) | 22 | 24 | 9 |
| `p98vonq` → `p98voffq` | 273 | 198 (0.725) | 22 | 45 | 8 |
| `p96vprod` → `p98vonq` | 265 | 183 (0.691) | 27 | 43 | 12 |
| `p98vonq` → `p96vprod` | 273 | 182 (0.667) | 19 | 62 | 10 |

**By volume.**
- Knob-off → knob-on: top 0.708, bottom **0.940**. The bottom SP input is identical, so its 6 % is the floor set by
  cross-volume clustering and light matching.
- Knob-on → knob-off: top 0.635, bottom 0.929.
- Production → knob-off: top 0.685, bottom 0.716. Both volumes see the U/V redistribution.

**Michel, among clusters that stay is_stm.**
- production → `p98voffq`: kept 95, lost 5, gained 12;
- `p98voffq` → `p98vonq`: kept 111, lost 5, gained 5;
- production → `p98vonq`: kept 92, lost 9, gained 15.

**Reading.**
- The swap is symmetric, and the tagged population and its Michel fraction are conserved (§0 point 3).
- Between a quarter and a third of tagged clusters flip under a few-percent change in their input (the U/V
  redistribution, or the ×1.125 top scale).
- **Where they flip depends on the change.**
  - Wire order: most losses are upstream of the tagger. 44 of 81 (production → `p98voffq`) and 43 of 82
    (production → `p98vonq`) are no longer STM candidates, from clustering or candidate selection.
  - The constant: losses split evenly between candidacy (24) and the tagger's own tests (22).
- The hand-scan record measures only the clusters that leave.

### 6.4 Round 3: a blind scan of both sides of the swap

Owner, 2026-09-13: *"can you use up to 3 sub-agent to scan the clusters the latest configuration tags but production
doesn't?"*

**Design, fixed before any verdict** (`d99/swap_scan_prereg.md`, `scripts/d99_swap_scan_set.py` → `d99/swap_scan_key.tsv`).
- **Scanned side (`on_only`):** all 91 clusters `p98vonq` tags (is_stm 1) that `p96vprod` does not. In production 62
  were not candidates, 19 were candidates the tagger rejected, and for 10 the object itself did not match (§6.3's rows).
- **Control side (`prod_only`):** a seeded 45 of the 80 clusters `p96vprod` tags that `p98vonq` does not. That is 82
  minus the two event/cluster keys that also occur on the scanned side. The control is sized to the 3-agent budget, so
  the difference's interval is control-limited.
- **Why a control.** The record's own purity (§6) comes from other scanners under another rubric, partly with the chain's
  answer visible. Comparing a fresh blind number against it would compare two instruments. Both sides here are judged by
  the same scanners, under the same rubric, in one shuffled list.
- **Display.** Each item is shown on the arm that tags it, so every item is "tagged" on its own display. Each arm has its
  own prep (`prep_stm_michel_scan.py --ctx-cells`, scratch only).
  - `scan_harness.py --blind --hide-selection` strips `is_stm`, the reject names, `in_fv` and the flow summary.
  - 0 of 136 `context.json` files name an arm.
  - `check_shots.py`: 0 blank, 0 incomplete, `c_3d_stop` unique colours min 1530 (`d99/swap_scan_check_shots.txt`).
- **Rubric** (`d99/swap_scan_rubric.md`, sha `d760e223…`, stamped on all 136 records). It is the PDHD v5 rubric ported to
  PDVD:
  - vertical drift along x, with up = +x, checked on this sample's fit ends: 59 of 200 reach x > +320, 11 reach x < −320;
  - rule 7 on `dx`;
  - the anode planes at x = ±339.9 (99th percentile of |fit end x| 339.8);
  - the CRU seams;
  - the PDHD-only traps removed.

  It was frozen for the whole round; the scanners' objections are recorded, not folded in.
- **Scanners.** 3 general-purpose subagents per wave, 2 waves (23/23/23, then 23/23/21), each about 21–25 min. Private
  out dirs; no scanner saw a key, a record or another scanner's calls.
  - One scanner consulted its own advisor tool on one item and lowered that call's confidence (high → medium).
  - Reports: `d99/swap_scan_reports/`.
- **Readout, pre-registered.**
  - stopper = `STM_MICHEL`, `STM_ONLY` and their `FRAG_`; non-stopper = `THRU`, `FRAG_THRU`; `MESSY`/`UNCLEAR` excluded
    and counted;
  - purity = stoppers / judged;
  - the statistic is the difference on_only − prod_only, bootstrapped over items.

**Result** (`scripts/d99_swap_scan_score.py` → `d99/swap_scan_score.txt`; record `../scan/pdvd_stm_michel_sw99_verdicts.json`):

| | **newly tagged (`on_only`)** | **dropped (`prod_only`, control)** |
|---|---|---|
| items / judged / stoppers | 91 / 78 / 66 | 45 / 43 / 39 |
| verdicts | STM_MICHEL 45, FRAG_STM_MICHEL 1, STM_ONLY 20, THRU 11, FRAG_THRU 1, MESSY 4, UNCLEAR 9 | STM_MICHEL 24, STM_ONLY 15, THRU 3, FRAG_THRU 1, MESSY 2 |
| **purity** (Wilson 68 %) | **0.846 [0.801, 0.883]** | **0.907 [0.853, 0.942]** |
| MESSY + UNCLEAR | **13 (14.3 %)** | 2 (4.4 %) |
| purity with MESSY/UNCLEAR counted as not stoppers | 0.725 | 0.867 |
| `high`-confidence calls only | 0.907 (39/43) | 0.920 (23/25) |
| top / bottom | 0.875 (42/48) / 0.800 (24/30) | 0.968 (30/31) / 0.750 (9/12) |
| production-side status: not a candidate | 0.868 (46/53) | 0.900 (18/20) |
| production-side status: candidate, tagger rejected | **0.765 (13/17)** | **0.944 (17/18)** |
| object not matched | 0.875 (7/8) | 0.800 (4/5) |
| hand STM_MICHEL among hand stoppers | 0.697 (46/66) | 0.615 (24/39) |
| chain `michel_found` against the hand Michel: purity / efficiency | 0.894 / 0.913 | 0.913 / 0.875 |

**The statistic:** purity(on_only) − purity(prod_only) = **−0.061**, 68 % [−0.120, +0.001], 95 % [−0.174, +0.060]
(10 000 bootstraps). By the pre-registered reading this is **purity-neutral within this sample**. The point estimate is on
the low side, and the 68 % interval just reaches zero.

**Where the difference sits.**
- **Not in candidacy.** Clusters that became candidates only on the latest configuration are as pure as those that
  stopped being candidates (0.868 against 0.900).
- **In the tagger's own flips.** Candidates production's tagger rejected and `p98vonq` accepts read 0.765 (13/17). The
  mirror set reads 0.944 (17/18).
  - The gap is largest on top (0.875 against 0.968), where the ×1.125 charge moves the tagger's absolute thresholds, which
    were fitted on the old scale (§8).
  - Suggestive, not significant at these counts.
- **In what cannot be judged.** The newly tagged side carries three times the MESSY/UNCLEAR rate. Most are near-isochronous
  tracks and busy stops.

**The frame edge — a blind, independent sighting of §4.1.**
- All six scanner reports, unprompted, name the same thing: in run 039349, the dead-channel hatching and every grey track
  in `f_meas` stop at slice ≈ 1588–1600.
- Run 039349's SP frames are 6400 ticks = 1600 slices × 4 ticks (checked on `039349_7_p98von`, anode 4: `frame_gauss`
  1536 × 6400). Runs 039252 and 039253 have 10000-tick frames (`d99/frame_nticks_120evt.txt`). Clustering ran with `readout_window_ticks=10000` on **both** arms (`pctree-evt*.tlas` of `039349_7_p98vonq` and
  `_p96vprod`). That is §4.1's staging.
- A track the frame cuts reaches the tagger as though it ended inside the window.
- 19 items name the edge (`d99/swap_scan_window_flagged.tsv`, a regex over notes and evidence, **post hoc**):
  - **11 on the newly tagged side, 10 of them in run 039349.** Verdicts THRU 6, STM_ONLY 3, UNCLEAR 1, STM_MICHEL 1, so
    **half of that side's 12 non-stoppers**.
  - 8 on the control side.
  - 5 of the 19 (1 newly tagged, 4 controls) are in runs 039252 and 039253, whose frames are 10000 ticks long. What those
    scanners saw near slice 1500–1630 is not the readout window's end (§4.4).
- **Sensitivity, post hoc, with those 19 set aside** (`d99/swap_scan_score_window_excluded.txt`): 0.912 (62/68) against
  0.943 (33/35), difference −0.031, 95 % [−0.132, +0.070]; `high` calls 0.950 against 0.952. The pre-registered number
  above is the result; this only shows where most of its gap comes from.
- **Re-graded with each event's real window in §4.4:** 7 of the 19 leave their set (all in run 039349). Purity on the
  scanned objects that stay reads 0.889 against 0.929 (−0.040).

**Calibration against the existing record** (stopper-or-not on the items that have one).
- **Newly tagged side:** 15/21 agree (`high` 11/15). **All six disagreements go one way:** the record says THRU and the
  blind scan says STM_MICHEL, four of them at `high`.
  - These are objects production did not tag. The record's scanners saw "is_stm 0" on screen when they judged them.
  - That is the under-call doc pdvd/68 found in the agent record taken with the answer visible
    (`feedback_blind_the_scan_sheet`).
  - It also means the carried record's "12 THRU among the 23 newly tagged items it covers" overstates this side's false
    positives.
- **Control side:** 36/40 (`high` 24/25), disagreements in both directions.
- Neither number adjusts the purity statistic.

**Derived, mixing instruments** (the shared, tagged-on-both clusters from the record, which carries that bias):
- whole-arm is_stm purity `p98vonq` 0.937 against `p96vprod` 0.959;
- with the frame-edge items set aside, 0.959 against 0.970.

**The latest configuration's hand record.** `scripts/d99_latest_record.py` builds
`../scan/pdvd_stm_michel_p98vonq_carried_sw99_verdicts.json`:
- the 569 carried items (§5);
- plus the 68 newly tagged clusters the carry did not reach (STM_MICHEL 32, STM_ONLY 17, THRU 8, MESSY 4, UNCLEAR 7);
- the 23 items both cover keep their carried verdict (`d99/latest_record_merge.txt`).

Doc pdvd/98 §11 grades on it. The smx record and the carried records are untouched (M13).

**What the scanners could not settle, for the owner** (their reports list the keys):
- whether the frame edge counts as a no-measurement end;
- michel against gamma for pieces 5–10 cm out, which decides the verdict and not just the kind;
- decay-like charge with no object row, which forces STM_ONLY;
- the grey-continuation test cannot see across a CRU seam, because `f_meas` does not reach the neighbouring CRU.

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

## 9. Disk: OFF trim done; the July SP frames retired (owner yes, 2026-09-13)

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

**The July SP frames — RETIRED** (`CONFIRM=1 d99_retire_keep_sp.sh`, 2026-09-13, after the owner's yes: "Please retire the
July-frame to save some disk"). The frames were `/home/xqian/pdvd-frame-store/*_keep/protodune-sp-dnnroi-frames-anode*`
(SP of 07-13 with the v5 wire order).

- **Executed** (`d99/retire_keep_sp_confirm.txt`): 768 archives and the 1536 links onto them (`work/<evt>_keep/<frame>` and
  `work/<evt>_d27fresh/<frame>`) deleted; dangling links under `pdvd/work` 0 before and 0 after; `wire-cell` not running.
- **Manifest first** (`d99/retire_keep_sp_manifest.txt`): file sha256 and bytes of all 768 archives, and every removed link
  with its target, written and counted (768 hash lines) before the first `rm`.
- **Disk:** `/home/xqian` available 450.50 → 482.04 GB (+31.5 GB = 29.4 GiB); the store 40 G → 11 G.
- **Kept:** 192 archives = the 24 events below, unchanged (`d99/retire_keep_sp_dryrun_final.txt`, the same census).
- **What it costs.** Re-imaging from the July frames is no longer possible for the other 96 events without regenerating them:
  `run_nf_sp_dnnroi_evt.sh --wires protodunevd-wires-larsoft-v5.json.bz2 -O _<tag> <run> <evt>` reproduces them sample for
  sample (§4.3), and the manifest's sha256 checks the result. The imaging, clustering and PR products built on them
  (`d27fresh`, `d51vclus`, `p96vprod`) are real files and stay readable. One analysis read the frames through that path:
  doc pdvd/98's PDVD crops (production arm). Doc 98 now reads the latest configuration (`p98vonq` on the `p98von` frames,
  all 960 archives kept), and its round-1 production numbers stay as the committed `scan/d98/` tables.

Dry run with the final keep list, before the owner's yes (`d99/retire_keep_sp_dryrun.txt`):
- 768 archives, 29.37 GiB, and 1536 links pointing at them, closure reached in 2 passes;
- 24 events kept:
  - 039252_0..17 and 039253_0, which a committed script or doc Repro reads (doc qlmatch/18, doc nf_sp_img_clus/28,
    `check_clus97_tail_waveforms.py`, doc pdhd/29's `d29_gain_recomb.py`);
  - 039349_2/4/15/68/81, this doc's own imaging and clustering controls (`p98kchk`, `p98kq`);
- the census of cited events passes;
- negative control (an ON-arm frame injected): refused, rc 13;
- liveness guard: refused while PR was running, rc 11.

**Why it was held, and why it could go.**
- **Round 1:** the frames were held as the only copy of the SP that production and every hand scan were built on, and as
  the reference for splitting the SP difference.
- **Round 2:** that split is done (§4.3). Today's SP with `--wires protodunevd-wires-larsoft-v5.json.bz2` regenerates the
  July frames sample for sample from the raw frames. They are no longer unique evidence.
- **Round 3:** the owner said retire, and it was run as above.

`pdvd/scripts/retire/PROTECTED.txt`'s `keep` line ("it holds the SP+DNNROI frames d27fresh borrows") is not edited here: that
file carries a peer's uncommitted changes. The line now describes only the 24 kept events.

## 10. Not concluded / next

1. **Flip decision (owner).** For the constant:
   - charge closes per track (1.1228 of 1.1249);
   - the Michel energy top/bottom goes 0.807 → 0.983;
   - is_stm is unchanged within churn against OFF.

   - the tagged STM population is the same size or larger (§0, §6.3).

   The SP rerun that a flip implies also moves SP onto the v7-uvwfit order that imaging already uses. That is not a loss
   to avoid; it is a consistency production does not have yet. What a flip changes cluster by cluster is a symmetric
   swap of about 30 % of tagged clusters, and the purity of the newly tagged side is not yet measured.

   **Scan of the newly tagged side — DONE in round 3 (§6.4).** Purity 0.846 against 0.907 on the dropped side,
   purity-neutral within the sample (95 % [−0.174, +0.060]). The gap sits in the tagger's own flips and in the frame-edge
   items, and the newly tagged side carries more unjudgeable objects.
2. **Split the SP difference — DONE in round 2** (§4.3). The wire file is the whole difference; SP code and config did
   not drift.
3. **Re-scan** the stopper items that cannot be carried without a look on the arm that is eventually adopted: on `p98vonq`,
   STM_MICHEL 39 + STM_ONLY 25 + MESSY 17 (`d99/rescan_p98vonq.tsv`).
4. **Refit the downstream constants** for a flipped arm (§8): C (0.8630 on `p98vonq`), Michel thresholds, QtoL.
5. **Readout window** (§4.1): on run 039349, production's edge guard sees 10000 ticks over 6400-tick frames. **Round 3 saw its cost
   blind** (§6.4): half of the newly tagged side's non-stoppers end at the frame edge, and both arms carry the same
   staging. **Real-window arms and re-grade: DONE in round 4 (§4.4).**
   - Only run 039349 is affected (6400-tick frames).
   - The guard untags 9 scanned objects; 7 of them are THRU or UNCLEAR.
   - The gap narrows from −0.061 to −0.040 but does not close.
   - **Open for the owner:** whether production staging should use the real window on run 039349 (`PDVD_READOUT_NTICKS`
     or the frames in the input dir). It removes 11 is_stm clusters on production and 14 on the latest configuration.
   - After that comes the owner's look at the medium/low calls, including the 2 STM_ONLY calls the guard removes
     (`039349_7/67`, `039349_75/63`), and at the frame-edge items (`d99/swap_scan_reports/`).
6. **Candidates rose on ON:** 597 → 656 (events with a candidate 119 → 120). Not traced here. The per-key grades above use
   only the record's items.
7. **The 2–3 % residual** top-below-bottom on the common tracks (0.972 [0.952, 1.000]) is not significant here; a larger
   stopper sample would tell.

## Files

| file | what |
|---|---|
| toolkit `cfg/pgrapher/experiment/protodunevd/sp.jsonnet`, `l1sp_after_dnnroi.jsonnet` | `top_gain_scale` knob, default 1.0 |
| `pdvd/wct-nf-sp-dnnroi.jsonnet`, `pdvd/run_nf_sp_dnnroi_evt.sh` | TLAs + `--top-gain-scale`, `--wires` (both default off) |
| `scripts/d99_sp_arm.sh`, `d99_chain.sh`, `d99_stage_q.sh` | SP arm driver; imaging → clustering → PR driver; production-style clustering staging |
| `scripts/d99_frames.py`, `d99_manifest.py` | per-plane frame sums and L1SP share; frame manifests + G3 |
| `scripts/d99_control_compare.py` | imaging / clustering / PR drift controls (pctree, tlas, NaN-aware trees, mabc) |
| `scripts/d99_match.py`, `d99_rescan.py` | geometric carry-over of the record; re-scan list |
| `scripts/d99_grade.py`, `d99_dqdx_drift.py` | grading and dQ/dx forks with their controls |
| `scripts/d99_closure.py` | per-track OFF → ON plateau ratio, common-track samples |
| `scripts/d99_readout_window.py` | §4.1: candidates, edge-guard firings and clustering window per arm |
| `pdvd/run_clus_evt.sh` | `PDVD_READOUT_NTICKS` (§4.4): the clustering window given explicitly; unset = the frame-or-10000 rule (proven on `p99rwnul`) |
| `scripts/d99rw_arms.sh`, `d99rw_identity.py`, `d99rw_census.py`, `d99rw_regrade.py` | §4.4: the real-window arms (PR-only guard pair, re-clustered pair, unset control); identity against the sources; STM census by frame length; the §6.4 scan re-graded on the new pairs |
| `d99/frame_nticks_120evt.txt`, `d99/rw_*` | §4.4: SP frame length per event; the identity gates, census and re-grades |
| `scripts/d99_frame_identity.py` | §4.3: sample-level identity of two SP frame arms per anode, plane and frame tag |
| `scripts/d99_population.py` | §6.3: record-free STM census and is_stm transitions between arms (geometric match) |
| `scripts/d99_michel_ql.py` | Michel energy by volume, per-item ON/OFF, flash changes |
| `scripts/d99_trim_off.sh`, `d99_retire_keep_sp.sh` | guarded OFF-frame trim; guarded retire of the July SP frames (executed 2026-09-13) |
| `../scan/pdvd_stm_michel_smx9_carried_p98voffq.json`, `_p98vonq.json` | the carried hand-scan records (new keys; the smx record untouched) |
| `scripts/d99_swap_scan_set.py`, `d99_swap_scan_score.py`, `d99_latest_record.py` | §6.4: blind swap-scan item set and key; the pre-registered scorer (and its post-hoc `--exclude-keys`); the latest configuration's hand record |
| `d99/swap_scan_*` | §6.4: pre-registration, key, rubric, agent task, frame check, six scanner reports, scores, frame-edge list |
| `../scan/pdvd_stm_michel_sw99_verdicts.json`, `../scan/pdvd_stm_michel_p98vonq_carried_sw99_verdicts.json` | the blind swap-scan record (tag `sw99`, 136 items); the carried record plus sw99 on `p98vonq` (637 items) |
| `d99/` | gate records and every number in this doc |
