# 29 — PDVD Q/L matching since July: what changed, where the hand-scan agreement went, and the lever scan

**Status (2026-09-13): COMPLETE.** The drop from July `tm0k` 764 / 118 / 78 to production `p100flip` 678 / 97 / 163 is
fully attributed: code since July −22 agree, v7 wires in imaging + clustering −41, today's SP on the v7 wires −20, top
gain −3 (§4–5). Drift velocity and every Q/L parameter were unchanged. Of the pre-registered levers, the LASSO boundary
weight 0.1 passes the hand-scan rule (683 / 97 / 158) and the 120-event STM gate. **FLIPPED into production** as a
runner default (`PDVD_QL_LASSO_BWEIGHT=0.1`; exported empty recovers 0.2) (§7). Most of the loss stays open: it is top-volume
clustering, not a Q/L parameter.

Owner, 2026-09-13: *"Can you investigate a bit more on the history of the Q/L matching degradation? I recall that we have
updated the geometry, and the gain, nothing else, right? It is possible that we would need to make some adjustments on the
other parameters for the PDVD Q/L matching? Can you investigate and improve the situation? Please review the old md files,
as well as git history."* Mid-way: *"I do not recall changing drift velocity, right? Lasso parameter change could be
relevant."* Scope answers the same day: flip a lever into production **if all gates pass**; split the July step with
**config-level arms only**.

Pre-registration: `d29/prereg.md` (written after the Step A forensics, before any arm ran).

## Repro
```bash
# run from pdvd/ unless noted; S=docs/qlmatch/scripts; records docs/qlmatch/d29; scratch /home/xqian/tmp/p29
# sec 3-4: existing dumps only (read-only scripts import ql_display/ql_agree_score.py, they do not fork it)
python3 ql_display/ql_agree_score.py --tag p98voffq --truth-uid-map-tag keep          # the one unscored September arm
(cd $S && python3 d29_uidmap_audit.py --truth-pairs keep:p99wflip keep:p98voffq keep:p100flip \
    --cluster-pairs p99wflip:p98voffq p98voffq:p100flip) > docs/qlmatch/d29/uidmap_audit.txt
(cd $S && python3 d29_attribution.py --chain p99wflip p98voffq p100flip p101q --july tm0k) > docs/qlmatch/d29/attribution.txt
(cd $S && python3 d29_forensics.py --base p99wflip --arm p98voffq) > docs/qlmatch/d29/forensics_step2.txt
(cd $S && python3 d29_forensics.py --base p98voffq --arm p100flip) > docs/qlmatch/d29/forensics_step3.txt
# sec 5-6: knob wct-clustering.jsonnet wires_file / run_clus_evt.sh PDVD_CLUS_WIRES (G1 d29/g1_wires_file.txt), then the arms
# (these ran BEFORE the sec-7 flip: to reproduce them now, add PDVD_QL_LASSO_BWEIGHT= (empty) to every arm but q29l6/q29l7)
(cd $S && setsid nohup bash -c 'bash d29_chain.sh step0; bash d29_chain.sh levers' > /home/xqian/tmp/p29/chain.log 2>&1 < /dev/null &)
(cd $S && python3 d29_lever_eval.py --base p100flip --control q29base --arms <arms> --cfg-dir /home/xqian/tmp/p29/cfg) > docs/qlmatch/d29/lever_eval.txt
(cd $S && bash d29_chain.sh combo)                                                     # q29c1, then lever_eval_combo.txt
# sec 7: the STM gate on the candidate (120 events, clustering + PR), then the flip and F1
(cd $S && ARM=q29stm ENVS="PDVD_QL_LASSO_BWEIGHT=0.1" bash d29_stm_arms.sh && bash d29_stm_gate.sh q29stm)
# flip: run_clus_evt.sh ': "${PDVD_QL_LASSO_BWEIGHT=0.1}"'; compiled-config proof d29/flip_compiled_config.txt
(cd $S && env -u PDVD_QL_LASSO_BWEIGHT ARM=q29flip bash d29_stm_arms.sh)               # F1: production, no overrides
(cd $S && bash d29_f1_check.sh)                     # q29flip == q29stm: d99rw_identity.py --nt all + calib dumps cmp
```

## 1. The owner's questions, answered
- **Drift velocity: not changed.** The runner passes 1.48073 mm/µs on both sides as in July (`run_clus_evt.sh`
  `PDVD_DRIFT_SPEED_{BOT,TOP}_MMUS` default); every September sidecar and calib dump reads 1.48073 / 0.148073 cm/µs.
  The peer config commit `228f1c39` (2026-09-03, "E=450 V/cm production defaults") changed the wires file and added
  DL/DT/lifetime to `params.jsonnet`, which nothing before Q/L matching reads.
- **LASSO and every other Q/L parameter: not changed.** Git shows no default change to any Q/L cut since July
  (`qlmatching.jsonnet` gained only the byte-identical `qtol`; `run_clus_evt.sh` offset 13.507 µs, velocity, ladder, LASSO
  λ 0.2, rescue and cull defaults unchanged; `lasso_boundary_weight` never set, so the jsonnet literal 0.2), and all 51
  `quality_params` keys of the calib dumps (ladder, flag and amplitude settings; the LASSO settings are not among them) are
  identical across every September arm except QtoL in `p101q`.
- **But more than geometry and gain changed what the matcher is given** (§2), and the LASSO is where half of the recent
  losses happen (§4): intact truth bundles whose strength goes to 0 on the new clustering.

## 2. Everything since the July tuning that reaches Q/L matching on run 039252

July reference: `tm0k`/`tm0` (2026-07-22, docs qlmatch/26 and 28). Git (wcp at `b8eb56a8`, toolkit at `d3b398fc`):

| date | commit | change | reaches Q/L via | knob |
|---|---|---|---|---|
| 07-22 | wcp `56eb6f3f` | `run_light_evt.sh` tail merge ON (`_tmerge` light; `tm0` scored on it) | light | default |
| 07-23 | tk `315f9653f` | `merge_pct` carries the per-anode ctpc / dead-map clouds across APA merges ("NOT bit-identical" on PDVD in its own gate) | pre-Q/L group clustering + the Q/L merge | none |
| 07-25 | tk `882ad107`, `11bec7f4` | `realign_perblob` default true; per-blob row order | stated structural no-op on PDVD | C++ default |
| 08-25 | tk `95c10cd16` | `rescue_empty_flashes` / shared-flash walk in flash order, not pointer order | Q/L rescue on clusters with bundles on two flashes | none |
| 09-02 | wcp `a61fa097` | `stm/run_campaign.sh` exports `PDVD_LIGHT_SUFFIX=_keep` (production reads the pre-merge light, as `tm0k` did) | light | runner |
| 09-03 | tk `228f1c39` | `params.jsonnet` wires v6 → v7-uvwfit | SP (July SP used v5 order) + imaging (July v6) + clustering | default |
| 09-03 | tk `4e2bd2f1`/`16c7728a`, wcp `3d1257f7`/`cee3f54d` | `wrapped_channel_charge = true` (doc pdvd/31 round 6, owner flip) | charge of wrapped U/V points in every pre-Q/L stage | driver TLA |
| 09-03 | wcp `c1aa2525`, `2a8f91b9` | staging source `_keep` → `d27fresh` (v7 re-imaging of the July frames); provenance guard | imaging | runner |
| 09-04 | wcp `c747b6b4` | `SAVE_ASSOC` 0 → 1 (three additive per-blob arrays; `clustering-global` byte-identical in doc pdvd/39) | none expected | runner |
| 09-13 | wcp `f52a374e` | real readout window | no-op on 039252 (10000 ticks) | runner |
| 09-13 | tk `b22ff59a`, wcp `cabbdd0b` | SP top gain 0.889; staging → `pvdimg` (today's SP, v7 order) | SP → imaging → clustering | driver |
| 09-13 | tk `d3b398fc`, wcp `b8eb56a8` | `qtol` knob, default 0.094 | none (byte-identical) | — |

Ruled out: SP/NF C++ (doc pdvd/99 round 2 regenerates the July frames sample for sample with the July wires), imaging
config (`img.jsonnet`, `wct-img-all.jsonnet` unchanged), the light archives (`039252_light*_keep` untouched since
07-16; the September dumps' flash lists are identical), every PR/STM-only flip of docs pdvd/28–100 (downstream of Q/L),
the master merges (no `match/`, `protodunevd` or pre-Q/L clustering changes beyond the table), doc 97's `sep_fv_point`
(SBND only).

## 3. The reference and the scorer: what can and cannot be compared
- **The July numbers were scored differently.** Doc 26 scored `tm0` with `--truth-time-map` → 763/118/79; doc 28 scored
  the same tag with the map + `--truth-time-shift -13.507 --truth-uid-map-tag tm0k` → 752/116/91 (11 agree from
  arguments). Production reads `_keep` light at the 13.507 µs pull, so the like-for-like July reference is **`tm0k`
  764/118/78** (`_keep` light, no map, no shift).
- **The July calib dumps are gone** (`tm0k`, `tm0`, `rc14`, `cathxa`, … dropped in the 2026-09-04 cleanup; their
  `scores.json` survive). Today's arms are scored with the truth's cluster uids mapped by geometry through `keep` (the
  July clustering, all 1529 truth uids present), no time map (decided on `p100flip`: it joins more), no shift.
- **The map is sound** (`d29/uidmap_audit.txt`): keep → `p99wflip` / `p98voffq` / `p100flip` maps 1200 / 1186 / 1191 of the
  1208 objective long truth entries, (y,z) centroid median 0.8–1.2 cm; entries landing > 100 cm away in y 17 / 19 / 23,
  on par with the v7 → v7 cluster controls (20, 31). There is no v6/v7 face-swap artefact in the join.

## 4. Attribution with the dumps that exist (`d29/attribution.txt`, `forensics_step2.txt`, `forensics_step3.txt`)

`d29_attribution.py` first recomputes the four recorded scores from the scorer's functions (all REPRODUCED), then scores
every arm on full truth and on the **common truth** (1475 of 1529 entries mapping in every arm):

| step | from → to | what changed | agree / phantom / missed, full truth | common truth |
|---|---|---|---|---|
| — | `tm0k` | July reference | 764 / 118 / 78 | — |
| 0 | `tm0k` → `p99wflip` | wires v6 → v7 (imaging + clustering), code 07-22 → 09-03 (`315f9653f`, `95c10cd16`, …), wrapped charge ON | → 701 / 100 / 140 | not in common space |
| 1 | `d27fresh` ≡ `d51vclus` ≡ `p99rwprod` ≡ `p99wflip` | 09-03 → 09-13 binaries, real window | calib dumps identical | — |
| 2 | `p99wflip` → `p98voffq` | SP rerun with today's code on v7 wires (U/V wire order), gain off | → 681 / 90 / 157 | 699/97/135 → 677/89/156 |
| 3 | `p98voffq` → `p100flip` | SP top gain 0.889 | → 678 / 97 / 163 | → 674/95/161 |
| (4) | `p100flip` → `p101q` | QtoL 0.0783 (doc pdvd/100 §8, not adopted) | → 672 / 95 / 169 | → 668/93/167 |

- **Step 0** (list diff by event, volume and time; approximate, the uid spaces differ): 67 pairs newly missed, **52 of
  them top**, 7 recovered; 24 phantoms resolved, 8 new.
- **Step 2, 35 newly missed** (24 top), 13 recovered:
  - 21 are **cluster-identity changes** (17 top): clustering hands the matcher a different object. The bounds, fixed in
    advance, are an npoints ratio outside [0.5, 2] or an x end moving more than 10 cm.
  - 10 are **mis-picks.** The truth bundle's ks (0.098) and χ²/ndf (2.8) are unchanged, but its LASSO strength goes
    0.96 → 0, and xtpc_pin is lost 3×.
  - 3 have no truth bundle; 1 is unmatched.
- **Step 3, 20 newly missed** (17 top), 16 recovered:
  - 11 are **mis-picks** (8 won by a plain-LASSO bundle). The truth bundle's χ²/ndf goes 1.14 → 0.86 with the gain, and
    its strength 0.92 → 0.
  - 8 are identity changes, all top; 1 has no truth bundle.
- **Correction to doc pdvd/100 §8.5.** "The gain flip costs −23 agree" is two steps: the SP wire-order rerun costs −20,
  the gain −3.
- **Reading.** The recoverable-by-Q/L class is the LASSO zeroing an intact truth bundle (the owner's hint). The rest are
  upstream clustering changes, concentrated in the top volume.

## 5. Splitting step 0 (config-level arms; `d29/step0_arms.txt`, `attribution_from_q29v6.txt`, `forensics_wires.txt`)

New default-OFF knob: `wct-clustering.jsonnet` `wires_file` / `run_clus_evt.sh` `PDVD_CLUS_WIRES` (G1 `d29/g1_wires_file.txt`:
compiled config identical at the default on 039252_0, 039253_15, 039349_7; with v6 exactly one leaf, the
`WireSchemaFile` filename). Arms by `scripts/d29_arms.sh`, clustering only, 18 events, `_keep` light, pin `libpin_p100b`;
control `q29base` (production config on `pvdimg`) reproduces `p100flip`'s calib dumps byte for byte on 18/18
(`d29/control_q29base_p100flip.txt`).

| arm | input | change | agree / phantom / missed | unmapped truth |
|---|---|---|---|---|
| `tm0k` (July record) | July v6 imaging | July code | 764 / 118 / 78 (no map; short 13, cluster-missing 0) | — |
| `q29v6` | `keep` = the same July v6 imaging | today's code and config, wires v6 | **742 / 108 / 98** (keep map; short 13, cluster-missing 0) | 7 |
| `q29v6nw` | `keep` | + `wrapped_channel_charge=false` | 742 / 108 / 98 — calib dumps byte-identical to `q29v6` 18/18 | 7 |
| `q29v7nw` | `pvdimg` | production + `wrapped_channel_charge=false` | 678 / 97 / 163 — calib dumps byte-identical to `q29base` 18/18 | 35 |

- **The wrapped-charge flip does not touch Q/L matching.** It changes 16 config leaves and every pctree (0/18 identical),
  but the matcher's calib dumps are byte-identical on both geometries.
- **Today's code on July's inputs: −22 agree / −10 phantom / +20 missed.** tm0k and `q29v6` share imaging, light, the
  Q/L config (git: no default change since 07-22) and the same excluded positives; what differs is the code since 07-22
  (`315f9653f` dead/ctpc maps through the merge, `95c10cd16` flash-ordered rescue, `11bec7f4` row order) and `SAVE_ASSOC`.
  Caveats: tm0k was scored without a map and `q29v6` through the keep map (v6 → v6 clustering, 7 truth entries unmapped);
  scored without a map `q29v6` reads 708 / 106 / 83 with 47 positives cluster-missing, i.e. its clusters are renumbered, so
  the keep-map score is the like-for-like one. Split among the commits: not done (owner: config-level arms only).
- **The v6 → v7 wires in imaging and clustering: −41 / −8 / +42, the largest single step.** Common truth (1469 entries):
  `q29v6` 735 / 101 / 96 → `p99wflip` 699 / 97 / 134. 50 truth pairs newly missed (37 top), 10 recovered.
  Forensics (`d29/forensics_wires.txt`): 26 cluster-identity changes (22 top), 16 mis-picks (12 won by a plain-LASSO
  bundle; the truth bundle's c2n 3.17 unchanged, ks 0.088 → 0.102, strength 0.96 → 0; flags lost at_x_boundary 3,
  consistent 3, xtpc_consistent 4), 7 without a truth bundle, 1 unmatched — the same two classes as steps 2 and 3.

**The whole drop adds up:** 764 / 118 / 78 → 678 / 97 / 163 = code since July (−22 / −10 / +20) + v7 wires in imaging and
clustering (−41 / −8 / +42) + today's SP on the v7 wires (−20 / −10 / +17) + top gain (−3 / +7 / +6).

## 6. Lever scan on production (`d29/prereg.md` §2–3, `d29/lever_eval.txt`, `lever_eval_combo.txt`)

Production config on `pvdimg` plus ONE lever, clustering only, 18 events; the control `q29base` = `p100flip` byte for
byte. Every lever arm's compile-only config differs from the control only in its lever's leaves (`lever_eval.txt`).
Rule: against `p100flip` (678 / 97 / 163) a metric improves and none regresses by more than 2 pairs — on full truth,
on the even-idx half (319 / 50 / 101), the odd-idx half (359 / 47 / 62) and common truth. Merit = Δagree − Δphantom − Δmissed.

| arm | lever (production value) | full truth | Δ | even | odd | rule | merit |
|---|---|---|---|---|---|---|---|
| `q29l1` | LASSO λ 0.1 (0.2) | 682 / 101 / 159 | +4 / +4 / −4 | ok | x | fail | +4 |
| `q29l2` | LASSO λ 0.15 | 680 / 101 / 161 | +2 / +4 / −2 | x | x | fail | 0 |
| `q29l3` | LASSO λ 0.3 | 676 / 94 / 165 | −2 / −3 / +2 | ok | ok | **pass** | −1 |
| `q29l4` | strength cutoff 0.02 (0.05) | 679 / 99 / 162 | +1 / +2 / −1 | x | ok | fail | 0 |
| `q29l5` | background weight 0.3 (0.5) | 678 / 98 / 163 | 0 / +1 / 0 | x | x | fail | −1 |
| `q29l6` | **LASSO boundary weight 0.1** (0.2) | **683 / 97 / 158** | **+5 / 0 / −5** | +4/−1/−4 ok | +1/+1/−1 ok | **pass** | **+10** |
| `q29l7` | LASSO boundary weight 0.4 | 677 / 86 / 164 | −1 / −11 / +1 | −2/−7/+2 ok | +1/−4/−1 ok | **pass** | +9 |
| `q29l8` | ladder χ²/ndf ceilings 35/35/35/60 (12/12/12/30) | 673 / 101 / 168 | −5 / +4 / +5 | x | x | fail | −14 |
| `q29l9` | pin keeps the strength-cutoff exemption (min strength 0.02) | 680 / 104 / 161 | +2 / +7 / −2 | x | x | fail | −3 |
| `q29v7nw` | wrapped charge off (on) | 678 / 97 / 163 | 0 / 0 / 0 (dumps identical) | x | x | fail | 0 |
| `p101q` | QtoL 0.0783 (0.094) | 672 / 95 / 169 | −6 / −2 / +6 | x | x | fail | −10 |
| `q29c1` | boundary weight 0.1 + λ 0.3 (prereg combination) | 679 / 98 / 162 | +1 / +1 / −1 | +2/+2/−2 ok | −1/−1/+1 ok | pass | +1 |

- **The owner's LASSO hint is where the recoverable matches are, and the parameter is the boundary weight, not λ.**
  `lasso_boundary_weight` shrinks the L1 penalty of boundary / near-PMT bundles; it is the C++ default 0.2
  (`QLMatching.h:374`), never tuned for PDVD (doc 19's phase-4 sweep did not include it). Doc 28's rl5 tried 0.1 at the
  July **offset-0** frame and lost (724 / 98 / 119 vs 728 / 94 / 115); at production's pulled frame on today's clustering
  it gains.
- **What 0.1 moves** (`d29/pairs_p100flip_q29l6.txt`, `forensics_q29l6_{losses,recoveries}.txt`): 6 truth matches
  recovered (5 top), all LASSO mis-picks on `p100flip` (4 plain-LASSO winners, 2 rescue picks); 1 lost (a bottom
  mis-pick); phantoms 3 new / 3 resolved. 6 vs 1 is small (sign test p ≈ 0.13); the rule, fixed in advance, is what
  adopts it.
- **The same parameter passes in both directions.** 0.4 kills 11 phantoms at −1 agree / +1 missed. The pre-registered
  merit ranks 0.1 (+10) over 0.4 (+9) by one unit: the boundary weight trades missed against phantoms, and on these 18
  events neither side dominates. This is reported, not resolved.
- **The combination is worse than 0.1 alone** (+1 merit): λ 0.3 undoes what 0.1 recovers.
- λ alone, the strength cutoff, the background weight, the old ladder ceilings and the pin exemption do not pass; neither
  wrapped charge (no effect) nor QtoL 0.0783.
- **Re-running this table after the flip (§7).** The `p100flip` / `q29base` baseline and every lever arm except `q29l6` /
  `q29l7` (which set the variable themselves) ran at boundary weight 0.2. Re-running them now gives 0.1 unless
  `PDVD_QL_LASSO_BWEIGHT=` (empty) is added to the arm's `ENVS`. The same holds for any older PDVD Q/L or STM gate arm
  (`d99*`, `d100*`) re-run against its archived outputs.

## 7. Decision: LASSO boundary weight 0.1 into production; what stays open

**The STM gate** (`scripts/d29_stm_gate.sh q29stm`, records `d29/stm_gate_q29stm_*.txt`). `q29stm` = production
staging on all 120 events with `PDVD_QL_LASSO_BWEIGHT=0.1`, clustering + PR (`scripts/d29_stm_arms.sh`; 120 clustered,
119 PR + 039349_30 with no STM candidate, as on `p100flip`), against `p100flip`:

| check | `p100flip` | `q29stm` | bar | |
|---|---|---|---|---|
| is_stm purity (owner-corrected record, carried 635/635) | 0.963 ± 0.012, eff 0.872 | 0.963 ± 0.012, eff 0.875 | within 1.5σ | pass (+0.00σ) |
| Michel purity | 0.924 ± 0.021, eff 0.848 | 0.924 ± 0.021, eff 0.853 | within 1.5σ | pass (+0.00σ) |
| census: candidates / is_stm / Michel | 540 / 259 / 160 | 542 / 261 / 160 | reported | |
| record-free transitions (null 1.000) | — | 257 of 259 stay is_stm; 257 of 261 back; no Michel lost or gained among those staying | reported | |
| record movers | — | 1 (a top stopper gains its tag, 039349_34/49) | reported | |
| crosser closure | 0.833 | 0.833 (same 192 anchors) | reported | |
| events identical (pctree, tlas, PR trees, mabc-pr) | — | 43 / 120 | reported | |

The boundary weight changes the Q/L output of 77 of 120 events: the flash association and matching-driven merges that go into
the pctree, and from there the PR trees. On the corrected record the STM chain still comes out exactly as pure, and both
efficiencies go up slightly.

**Adopted by the pre-registered rule, not by significance.** The hand-scan gain is 6 truth matches recovered against 1
lost (sign test p ≈ 0.13) and holds on both halves; the rule was fixed before the scan and this is the same standing the
doc 23–28 adoptions have. It should not be quoted as a measured 5-pair gain.

**The flip** (owner, 2026-09-13: flip if all gates pass): `run_clus_evt.sh` gets `: "${PDVD_QL_LASSO_BWEIGHT=0.1}"` next to
the λ default, where the other tuned Q/L defaults (ladder, λ) already live. The toolkit literal (0.2 in `qlmatching.jsonnet`)
and the C++ default are untouched. **Scope:** in this repository `run_clus_evt.sh` is the only job that compiles
`wct-clustering.jsonnet` / `qlmatching.jsonnet` (`stm/run_campaign.sh`, `profile_ql.sh`, the staging and arm scripts all
call it), so the whole wcp PDVD chain gets 0.1. A job that compiles the toolkit `qlmatching.jsonnet` directly (e.g. an
external or LArSoft configuration) still gets 0.2, like the λ and ladder defaults before it. `PDVD_QL_LASSO_BWEIGHT=`
(exported empty) recovers 0.2; runs that set the variable explicitly are unaffected.
Proofs: compiled config after the flip == before the flip + `PDVD_QL_LASSO_BWEIGHT=0.1`, and after the flip with the
variable empty == before the flip (md5, 3 events, `d29/flip_compiled_config.txt`: the only leaf that moves is the
QLMatching `lasso_boundary_weight`). **F1 PASS** (`scripts/d29_f1_check.sh`): `q29flip` = production after the flip with no
overrides, all 120 events, is identical to `q29stm` on 120 / 120 (pctree, tlas, PR trees, mabc-pr; `d29/f1_identity.txt`)
and its calib dumps are byte-identical on 120 / 120, while they differ from pre-flip `p100flip` on 120 / 120
(`d29/f1_calib.txt`). The bundle strengths move in every event, even where the final matches do not. PR: 119 + 039349_30
with no STM candidate, as on `p100flip`.

**Open, in order of size:**
1. **The v6 → v7 wires in imaging and clustering (−41 agree / +42 missed), plus today's SP on those wires (−20 / +17).**
   Most of the lost pairs are clustering handing the matcher a different object in the top volume (26 of 50 and 21 of 35),
   not a Q/L parameter. This is the physics question left: which top-volume clustering decisions changed with the wire
   order, and whether the hand scan's truth (July clusters mapped by geometry) still describes the right object.
2. **Code since July (−22 / +20)**, not split among `315f9653f`, `95c10cd16`, `11bec7f4` and `SAVE_ASSOC` (owner:
   config-level arms only).
3. **The boundary weight is a trade, not an optimum.** 0.4 passes the same rule in the other direction (−11 phantoms at
   −1 agree / +1 missed); the merit ranked 0.1 one unit ahead. A larger hand-scanned sample would settle it.
4. **The scan truth is from July** (an AI scan plus one owner event on the v6 clustering). A re-scan of run 039252 on
   today's production would replace the geometric map with judged truth.

## Files
| file | what |
|---|---|
| `pdvd/wct-clustering.jsonnet`, `pdvd/run_clus_evt.sh` | `wires_file` TLA / `PDVD_CLUS_WIRES` (default '' = params.jsonnet, byte-identical) |
| `pdvd/run_clus_evt.sh` | **production default `PDVD_QL_LASSO_BWEIGHT=0.1`** (§7; empty recovers 0.2) |
| `docs/qlmatch/scripts/d29_common.py`, `d29_uidmap_audit.py`, `d29_attribution.py`, `d29_forensics.py` | §3–4 (import the scorer) |
| `docs/qlmatch/scripts/d29_arms.sh`, `d29_chain.sh`, `d29_lever_eval.py` | §5–6 arms and the pre-registered rule |
| `docs/qlmatch/scripts/d29_stm_arms.sh`, `d29_stm_gate.sh`, `d29_f1_check.sh` | §7 120-event STM gate arm, F1 arm, gate analysis (reuse the doc pdvd/99–100 scripts), F1 identity + calib cmp |
| `docs/qlmatch/d29/prereg.md`, `g1_wires_file.txt`, `control_q29base_p100flip.txt` | pre-registration, knob G1, control |
| `docs/qlmatch/d29/uidmap_audit.txt`, `attribution*.txt`, `forensics_*.txt`, `scores_p98voffq.md` | §3–4 |
| `docs/qlmatch/d29/step0_arms.txt`, `lever_eval.txt`, `lever_eval_combo.txt`, `pairs_p100flip_q29l6.txt` | §5–6 |
| `docs/qlmatch/d29/stm_gate_q29stm_*.txt`, `flip_compiled_config.txt`, `f1_*.txt` | §7 gate, flip proof, F1 |
