# 68 — The owner's option scan (`smx3`): the 3 cm Bragg-peak anchor flipped, `dx_norm_length` 4 mm measured and left off

**Status (2026-09-10). The owner hand-scanned 49 items chosen to decide the two
options doc 67 §5 left open. `bragg_peak_anchor` (3 cm search) is now PDVD
PRODUCTION. `dx_norm_length` 4 mm beats the old production on the scan but NOT
the new one: on top of the anchor it lowers `is_stm` F1 0.807 → 0.792 — and its
file is also read by the neutrino chain. It stays OFF.** On
the merged record (smx1a + the owner's 49 verdicts, 580 judged items),
production → anchor: `is_stm` TP 176 / FP 7 / FN 108 → **197 / 7 / 87**
(purity 0.962 → 0.966, efficiency 0.620 → 0.694, F1 0.754 → 0.807),
`michel_found` unchanged. The scan also found that the agent-made smx1a labels
were wrong on 6 of the 17 contested record items, 4 of them stoppers labelled
THRU — which is why doc 65 §5.1 and doc 67 §5 read the anchor as a purity
loss. PDHD stays OFF. SBND/uBooNE untouched.

The owner's instructions: "I am happy to do the scan … include the 8 + 35
candidates … you can drop the stop_local_residual_cm", then, after scanning,
"analyze them and fold them into the decisions and updates". The decision rule
is the owner's from doc 67: "if the results lead to better reconstructed STM +
Michel w.r.t. our scan results, they would be good to be updated".

Companion docs: pdvd/65 §4–§5 (the two options first measured), pdvd/67 §5 (the
list this answers), pdhd/12 (the display), pdvd/55 (smx1a, the record this
corrects).

---

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img
W=$HOME/tmp/d68; PIN=$HOME/tmp/d66/libpin          # libWireCellClus md5 c83d6227aca6
S='survey_enable:true,survey_radius_cm:60.0,survey_max_len_cm:25.0'
R=pdvd/docs/nf_sp_img_clus/scripts/d53_run_arms.sh; X=pdvd/docs/nf_sp_img_clus/scripts
# the two options on today's (doc 67) production bag, and both together
ARM=d68a3 DET=pdvd SRC=d16vnu JOBS=14 PIN=$PIN PR_TLA="-S stm_michel_extra={$S,bragg_peak_anchor:true,bragg_peak_search_cm:3.0}" $R
ARM=d68d4 DET=pdvd SRC=d16vnu JOBS=14 PIN=$PIN PR_TLA="-S stm_michel_extra={$S} -A trackfitting_config=$HOME/tmp/d65/tf_dx04.json" $R
ARM=d68ad DET=pdvd SRC=d16vnu JOBS=16 PIN=$PIN PR_TLA="-S stm_michel_extra={$S,bragg_peak_anchor:true,bragg_peak_search_cm:3.0} -A trackfitting_config=$HOME/tmp/d65/tf_dx04.json" $R
for a in d68a3 d68d4 d68ad; do
  (cd pdhd/stm_michel_scan && ./prep_stm_michel_scan.py --det pdvd --arm $a --outdir $W/prep_$a --sheetdir $W/sheet_$a \
      --pin-tranche ../../pdvd/docs/scan/pdvd_stm_michel_scan_sheet.tsv)
done
# the scan set (sec 1), served on :5017
python3 $X/d68_build_scan_set.py --prod $HOME/tmp/d67/prep_d67v --anchor $W/prep_d68a3 --dx $W/prep_d68d4 \
    --outprep pdhd/stm_michel_scan/prep-pdvd-smx3 --sheet pdvd/docs/scan/pdvd_stm_michel_smx3_sheet.tsv \
    --questions pdvd/docs/scan/pdvd_stm_michel_smx3_questions.json
(cd pdhd/stm_michel_scan && ./serve_stm_michel_scan.sh 5017 --det pdvd --scan-tag smx3 \
    --manifest $PWD/../../pdvd/docs/scan/pdvd_stm_michel_smx3_sheet.tsv --prepdir $PWD/prep-pdvd-smx3 \
    --questions $PWD/../../pdvd/docs/scan/pdvd_stm_michel_smx3_questions.json)
# the owner's verdicts folded in (sec 2-4)
python3 $X/d68_score_scan.py --labels pdvd/docs/scan/pdvd_stm_michel_smx3_labels.json \
    --questions pdvd/docs/scan/pdvd_stm_michel_smx3_questions.json \
    --prod $HOME/tmp/d67/prep_d67v --anchor $W/prep_d68a3 --dx $W/prep_d68d4 \
    --write-merged pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_verdicts.json
cd pdhd/stm_michel_scan
STM_SCAN_RECORD=$PWD/../../pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_verdicts.json \
    python3 census_score.py --prep $W/prep_d68a3 --baseline $HOME/tmp/d67/prep_d67v --arm d68a3
python3 census_score.py --check                    # default record (smx1a): 0 of 14 differ
```

---

## 1. What was scanned

Built by `scripts/d68_build_scan_set.py` against TODAY's production (`d67v`,
doc 67's bag), not against the bag doc 65 measured on — the anchor's contested
set moved when doc 67 flipped `ks_margin` and `compare_range_cm`:

| group | items | what | payload shown |
|---|---:|---|---|
| 1 | 13 | anchor 3 cm (`d68a3`) flips `is_stm` against production AND disagrees with smx1a: 6 "new FPs", 7 "lost stoppers". All 8 of doc 67 §5's list are inside | production (the anchor is verdict-only; geometry asserted identical) |
| 2 | 32 | candidates that exist with `dx_norm_length` 4 mm (`d68d4`) and are not in the 569-item record: 30 only with the option, 2 (`039253_12/93`, `039349_81/56`) also production candidates that postdate the 569 draw | the option's reconstruction |
| 3 | 4 | record items where dx 4 mm flips `is_stm` AND disagrees with smx1a | the option's reconstruction |

Not scanned, and why: the 35 record items dx 4 mm makes DISAPPEAR from the
candidate set (their verdicts are in the record; 6 are stoppers, a known
efficiency cost) — none of them is the same track as any of the 32 new
candidates (median-distance test, 0 pairs; 30 of the 32 overlap no production
candidate at all).

The display (`stm_michel_viewer.py --questions`, doc pdhd/12's app) gained a
blue per-item panel naming the option and what production and the option each
read, and, for the anchor, a green dot-dash line on the dQ/dx panel where the
option puts the Bragg peak, drawn through the pin's origin. The panel never
shows smx1a's verdict. Without `--questions` the app is unchanged (headless
self-test 51 148 / 0 PDVD, 30 167 / 0 PDHD; pin persistence 34 / 0 PDVD).

## 2. The owner against smx1a, and the merged record

On the 17 re-judged record items (groups 1 and 3) the owner and smx1a agree on
stopper/not for **11 of 17**. Six changed:

| item | smx1a (confidence) | owner |
|---|---|---|
| `039252_2/79` | THRU (high) | STM_MICHEL, attached — "the current identified end point is OK" |
| `039349_43/66` | THRU (medium) | STM_MICHEL, both — pin moved 4.6 cm |
| `039349_72/11` | THRU (high) | STM_MICHEL, attached — pin moved 2.8 cm |
| `039349_48/21` | THRU (medium) | STM_MICHEL, attached — "Michel electrons did not get accessed" |
| `039349_61/58` | STM_MICHEL (medium) | MESSY |
| `039349_76/25` | STM_ONLY (medium) | THRU |

Four of the six are stoppers smx1a called THRU, two of them at high
confidence. smx1a was scanned with the reconstruction on screen (the blind was
removed on 2026-09-08, doc pdhd/12 §13), and on exactly these items production
says `is_stm = 0` — the plausible source of a one-directional under-call. The
contested set is selected (it is where smx1a and an option disagree), so 6/17
is not the record's error rate; it is stated as a finding, and the record is
not re-scanned here.

**The merged record** `pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_verdicts.json`
(601 records = smx1a's 569 with the owner's verdict on the 17, plus the 32 new
items in tranche 3) is the grading record from this doc on. smx1a itself is
untouched (M13); the owner's raw labels are committed as
`pdvd_stm_michel_smx3_labels.json`. `census_lib.py` reads the merged record when
`STM_SCAN_RECORD` names it; unset, it reads smx1a, so `census_score.py --check`
(doc 55's literals) is unchanged. **Every number in docs 56–67 is on smx1a;
every number here is on the merged record** — do not compare across the two
without saying so.

## 3. The anchor: flipped

Group 1, per item, against the owner:

| | items |
|---|---|
| option right | `039252_2/79`, `039349_43/66`, `039349_72/11` — all three "new FPs" that the owner calls stoppers |
| production right | the 3 true new FPs `039349_14/22`, `039349_24/23`, `039349_30/21` (THRU), and all 7 "lost stoppers" `039253_0/44`, `039253_13/73`, `039253_3/66`, `039253_6/85`, `039349_15/23`, `039349_36/46`, `039349_76/75` (owner: every one a stopper) |

Census on the merged record (580 judged; `d68_score_scan.py`):

| | `is_stm` TP / FP / FN | purity | eff | F1 | `michel_found` TP / FP / FN |
|---|---|---:|---:|---:|---|
| production (`d67v`) | 176 / 7 / 108 | 0.962 | 0.620 | 0.754 | 132 / 17 / 32 |
| **anchor 3 cm (`d68a3`)** | **197 / 7 / 87** | **0.966** | **0.694** | **0.807** | 132 / 17 / 32 |

The standard scorer (`census_score.py` on the 548 items with a payload) gives
the same move: 176 / 7 / 96 → 197 / 7 / 75, F1 0.774 → 0.828. By name: 28
stoppers gained, 7 lost, 3 FPs new, 3 removed (`039252_8/93`, `039349_21/26`,
`039349_26/51`).

**Which bar it clears.** Not the 0-new-FP bar docs 57–63 and 67 used — it
swaps three false positives for three. It clears the owner's rule, on both axes
at once: purity up (0.962 → 0.966) and efficiency up (0.620 → 0.694), with
`michel_found` untouched. The cost is named: 7 true stoppers lost, four of them
carrying a Michel (`039253_0/44`, `039253_13/73`, `039253_3/66`,
`039349_36/46`) — the 3 cm window landing on the Michel's dilution, the
mechanism doc 65 §5.1 described, now confirmed against the owner's eye. The
gain is outside the enriched tranche: tranche 1 goes 40 → 39 TP, tranche 2
135 → 157.

**Production edit**: `bragg_peak_anchor: true`, `bragg_peak_search_cm: 3.0` in
`stm_michel_knobs` (the C++ defaults are `false` / 10.0, so neither key is
inert). Doc 65 §5.1 and doc 67 §5's anchor rows are superseded by this section.

## 4. `dx_norm_length` 4 mm: better on the scan, held

Group 2 (32 never-scanned candidates): the owner calls 13 of them stoppers
(8 STM_MICHEL, 5 STM_ONLY) and 19 THRU. The option finds 5 of the stoppers
(`039253_3/60`, `039253_8/65`, `039349_50/55`, `039349_75/67`, `039349_81/62`);
7 are candidates it flags but rejects (`039252_1/109`, `039252_8/72`,
`039253_12/93`, `039253_7/30`, `039349_33/60`, `039349_57/18`, `039349_7/20` —
efficiency leads the option exposes); `039349_81/56` is found by both; all 19
THRU are correctly rejected. Group
3: option right on 2 (`039349_48/21`, `039349_76/25`), production right on 1
(`039349_31/51`), 1 MESSY.

| | `is_stm` TP / FP / FN | purity | eff | F1 | `michel_found` TP / FP / FN | F1 |
|---|---|---:|---:|---:|---|---:|
| production (`d67v`) | 176 / 7 / 108 | 0.962 | 0.620 | 0.754 | 132 / 17 / 32 | 0.843 |
| dx 4 mm (`d68d4`) | 179 / 4 / 105 | 0.978 | 0.630 | 0.767 | 134 / 18 / 30 | 0.848 |

`is_stm`: 11 gained, 8 lost, 0 new FPs, 3 removed. `michel_found`: 7 gained, 5
lost, 4 new FPs, 3 removed. Better on the scan, modestly.

**On top of the anchor it is no longer better.** After §3's flip the
decision-relevant comparison is anchor vs anchor + dx 4 mm (arm `d68ad`, both on
today's bag): `is_stm` 197 / 7 / 87 → 192 / 9 / 92 (purity 0.966 → 0.955, F1
0.807 → 0.792) — 12 stoppers gained, 17 lost, 3 FPs new (`039349_57/26`,
`039349_68/55`, `039349_81/51`), 1 removed (`039349_30/21`). Among the 17 lost
are `039252_2/79` and `039349_43/66`, two of the three items the owner
confirmed the anchor gets right: the two options act on overlapping profiles,
and the 4 mm smoothing moves the peak the anchor reads. `michel_found` still
gains by the same amount as alone (F1 0.843 → 0.848, +7 / −5 TP). So, scope
aside, dx 4 mm is not an improvement on today's production for the stopper
verdict, and it is left OFF. A Michel-only benefit would need the STM-only
route below AND a way to keep the tagger's fit at 6 mm.

**And it could not be flipped alone anyway.** `dx_norm_length` is not a `CheckSTM_Michel` knob. It lives in
`toolkit/cfg/pgrapher/experiment/protodunevd/pdvd_track_fitting.json`, which
`pr.jsonnet` passes as ONE `trackfitting_config_file` to three components —
`TaggerCheckSTM`, `CheckSTM_Michel` and `TaggerCheckNeutrino`. Changing it
changes an existing physics constant for the whole PDVD PR job, including the
neutrino chain this scan never graded (CLAUDE.md §5.1). Two routes for the
owner: (a) change the shared file and validate the neutrino chain separately;
(b) a new default-legacy `pr.jsonnet` parameter that gives the two STM
components their own track-fitting file (byte-identical when unset), leaving
the neutrino chain on the shipped one. Neither is built here; given the
combined arm, neither is recommended now.

## 5. The owner's notes, as leads

- `039253_0/44`, `039253_12/93`, `039349_48/21`: "the Michel electron is not
  (fully) identified / did not get accessed" — `michel_found` FNs the owner
  sees directly.
- `039253_13/73`: "a clear Michel, the muon did not reach the end, thus less
  clear Bragg peak"; `039349_33/60`: "the track did not go to the end … which
  seems to have a Michel" — fits that stop short of the real end (doc 63's
  undershoot family).
- `039252_2/79`: "the current identified end point is OK" — the anchor's
  reading of this THRU-labelled item was right.

## 6. Gates

| gate | result |
|---|---|
| C++ | untouched — every arm ran doc 66's binary, `libWireCellClus.so` md5 `c83d6227aca6` before and after each |
| viewer, without `--questions` | `selftest_stm_michel_scan.py` 51 148 / 0 PDVD, 30 167 / 0 PDHD; `selftest_pin_persistence.py` 34 / 0 PDVD. PDHD 27 / 1: check 10 is pre-existing — it runs only once a detector has `smx1` labels and looks for a PDVD item, and PDHD has had `smx1` labels since the owner's 2026-09-09 scan. Not fixed here |
| viewer, with `--questions` | headless: 49 items and questions, the panel in the layout, the anchor line at the item's shift with no pin (2.80 cm on `039252_2/79`) and at shift − pin rr with one (−6.80 at rr 9.6), no line on dx items, no question text carrying a verdict word; a browser load of :5017 with no page errors |
| scan set | 49 payloads + the reference curve; production and anchor geometry asserted identical on every group-1 item |
| records | smx1a untouched; the owner's labels committed byte-identical to `work/stm_michel_labels/smx3/labels.json` (md5 `52edf4e0`); `census_score.py --check` on smx1a 0 of 14; `census_score.py` on the merged record runs (548 scored) |
| flip-equivalence (edited production vs the pre-flip file + `d68a3`'s override) | **0 lines** |
| true OFF path (`bragg_peak_anchor: false`, `bragg_peak_search_cm: 10.0` on both files) | **0 lines** |
| `abtest/compile_all_cfg.sh` + `cmp_cfg.sh`, before vs after (`$W/cfg_all_pre`, `$W/cfg_all_post`) | 16 live jobs NORMDIFF 0, **OVERALL PASS** |
| other scan tags | PDHD `smx1`, PDVD `smx1` and `smx1a` label files unchanged (backup `$W/label_backup_20260910_055509`; :5017's PDHD `smx1` server was stopped to serve `smx3`) |
| incomplete events | `039252_11` has no STM candidate on any arm, production included; `039252_4` loses its one candidate under dx 4 mm (as in doc 65's `d65d4`) |

## 7. Doc 56 / 67 updates

Doc 56's status and Order paragraph: eleven knobs PDVD production; the anchor
row answered; `dx_norm_length` left OFF (worse on top of the anchor, and a
shared file); the grading record is
now the merged one. Doc 67 §5: anchor → flipped here; `dx_norm_length` →
measured on the owner's scan, left OFF; `stop_local_residual_cm` dropped by the
owner. Scripts committed: `scripts/d68_build_scan_set.py`,
`scripts/d68_score_scan.py`.
