# 65 — T7: the sampling step and the two shape tests

**Status (2026-09-10, overnight). Three measurements delivered, one knob
built and left OFF, no threshold moved.** (1) The KS margin is the shape-test
lever: `ks_margin` 0 → −0.05 buys 34 TPs for 5 FPs on the shape tests alone;
the contrast threshold and the tail/plateau windows are flat;
`compare_range_cm` 60 gains 10 TPs / −5 FPs. (2) The sampling step is a
parameter of the whole STM tagging, not of the Michel stage: ±4 mm on
`low_dis_limit` removes 136–167 of the 569 items from the candidate set;
`dx_norm_length` 4 mm is the one sampling setting the record mildly prefers
(michel F1 0.863 → 0.875). (3) The plateau autocorrelation drops below 1/e
from 1.2 cm on both detectors (lag-1 0.42 PDVD / 0.60 PDHD). (4)
`bragg_peak_anchor` — the prototype's peak-anchored origin, ported behind a
knob — at a 10 cm window trades purity for efficiency (TP 152 → 182, FP 9 →
22); at 3 cm it is purity-neutral (0.944) with +16 net TPs (F1 0.709 →
0.753) and a swapped FP set (4 new, 3 removed). Neither meets the 0-new-FP
bar; both are the owner's candidates, item lists in §5. Both byte-identical
gates PASS.

Doc pdvd/56 §8's T7 row (analysis; the owner decides): sweep
`low_dis_limit` / `dx_norm_length`; measure the profile autocorrelation
length; re-derive `bragg_tail_*`, `compare_range_cm`, `bragg_contrast_min`,
`ks_margin` jointly on §4's table; test the peak-anchored `rr` origin; note
`stm_recomb_calibrated`. This round ships ONE default-OFF knob (the
peak-anchored origin, `bragg_peak_anchor`) and scores it, and delivers the
three measurements as tables for the owner. **No threshold is moved** here
(CLAUDE.md §5.1 / §5.7, doc 56 §4): §3 names the operating point the tables
point at, and leaves the move to the owner.

Companion docs: pdvd/56 §4 and §6, pdvd/55 §15.2, pdhd/16 §6.2 and pdhd/17
§2–§3 (the charge scales), pdvd/61 §9 item 1 (the pass-3 step), pdvd/63 (T5,
whose `absorb_bragg_stub` this round re-scores together with the anchor).

---

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
wcbuild && ./build/clus/wcdoctest-clus            # 357/357 (two new default pins)

cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img
W=$HOME/tmp/d65
# measurement 1: the two shape tests re-scored offline on the current candidate's payloads
python3 pdvd/docs/nf_sp_img_clus/scripts/d65_shape_thresholds.py $HOME/tmp/d64/prep_d63a_fix
# measurement 3: the profile autocorrelation length per detector
python3 pdvd/docs/nf_sp_img_clus/scripts/d65_autocorr.py pdhd/stm_michel_scan/prep-pdvd $HOME/tmp/d64/prep_d63a_fix pdhd/stm_michel_scan/prep-pdhd

# measurement 2: the sampling sweep -- four scratch copies of the shipped track-fitting JSON
python3 - <<'EOF'
import json
b=json.load(open('/nfs/data/1/xqian/toolkit-dev/toolkit/cfg/pgrapher/experiment/protodunevd/pdvd_track_fitting.json'))
for tag,k,v in (("ld08","low_dis_limit",8.0),("ld16","low_dis_limit",16.0),("dx04","dx_norm_length",4.0),("dx09","dx_norm_length",9.0)):
    d=dict(b); d[k]=v; json.dump(d,open('/home/xqian/tmp/d65/tf_%s.json'%tag,'w'),indent=1)
EOF
SURVEY='survey_enable:true,survey_radius_cm:60.0,survey_max_len_cm:25.0'
R=pdvd/docs/nf_sp_img_clus/scripts/d53_run_arms.sh
# (run alongside round 3, on round 3's pinned binary $HOME/tmp/d64/libpin; baseline d64vleg)
for t in s08:ld08 s16:ld16 d4:dx04 d9:dx09; do a=${t%%:*}; j=${t##*:}
  ARM=d65$a DET=pdvd SRC=d16vnu JOBS=3 PIN=$HOME/tmp/d64/libpin PR_TLA="-S stm_michel_extra={$SURVEY} -A trackfitting_config=$W/tf_$j.json" $R
done

# the knob: peak-anchored rr origin, on this round's binary
ARM=d65vleg DET=pdvd SRC=d16vnu JOBS=8 PIN=$W/libpin PR_TLA="-S stm_michel_extra={$SURVEY}" $R
ARM=d65hleg DET=pdhd SRC=d16hnu JOBS=8 PIN=$W/libpin PR_TLA="-S stm_michel_extra={$SURVEY}" $R
ARM=d65v    DET=pdvd SRC=d16vnu JOBS=8 PIN=$W/libpin PR_TLA="-S stm_michel_extra={$SURVEY,bragg_peak_anchor:true}" $R
# the narrower window (sec 5.1), run on the round-5 binary $HOME/tmp/d66/libpin against its own leg d66vleg
ARM=d65v3   DET=pdvd SRC=d16vnu JOBS=7 PIN=$HOME/tmp/d66/libpin PR_TLA="-S stm_michel_extra={$SURVEY,bragg_peak_anchor:true,bragg_peak_search_cm:3.0}" $R
python3 pdvd/docs/nf_sp_img_clus/scripts/d51g_branch_census.py \
    --before 'pdvd/work/*_d64vleg' --after 'pdvd/work/*_d65vleg' --before-arm d64vleg --after-arm d65vleg --pts --out $W/g1
python3 pdvd/docs/nf_sp_img_clus/scripts/d51g_branch_census.py \
    --before 'pdhd/work/*_d53h' --after 'pdhd/work/*_d65hleg' --before-arm d53h --after-arm d65hleg --pts --out $W/g2
cd pdhd/stm_michel_scan
for a in d65v d65s08 d65s16 d65d4 d65d9; do
  ./prep_stm_michel_scan.py --det pdvd --arm $a --outdir $W/prep_$a --sheetdir $W/sheet_$a \
      --pin-tranche ../../pdvd/docs/scan/pdvd_stm_michel_scan_sheet.tsv
  python3 census_score.py --prep $W/prep_$a --baseline $HOME/tmp/d64/prep_d64vleg --arm $a --json $W/$a.json
  python3 ../../pdvd/docs/nf_sp_img_clus/scripts/d63_by_name.py $HOME/tmp/d64/prep_d64vleg $W/prep_$a
done
python3 census_score.py --check
```

---

## 1. Corrections to doc 56's T7 row

1. **`stm_recomb_calibrated` is `true` in both production drivers**
   (`pdvd/wct-pr-perevt.jsonnet:1198`, `pdhd:1132`); `false` is only the
   `pr.jsonnet` default. It selects the recombination model bound to
   `check_stm_michel` — an ENERGY knob. Neither shape test reads a
   recombination model: `expected` is a ratio of two muon-TABLE medians at the
   profile's own `rr` (`StmMichelFunctions.cxx:240-242`), and `kslike_compare`
   normalises both vectors to unit sum (`util/src/KSTest.cxx:228-235`). Doc
   pdhd/16 §6.2 measured it: every verdict input bit-identical across the flip
   on 579 PDVD / 325 PDHD candidates. The real scale mismatch — the table is
   ideal charge, the profile is reconstructed charge, ~7 % apart (doc pdhd/17
   §3) — cancels in both shape tests and survives only in `do_track_comp`
   (`R_NOT_MUON_PID`, a difference of two scale-contaminated distances) and in
   `R_PLATEAU_OFF_MIP` (absolute, against `mip_dqdx` 55000). So "the Bragg
   reference and the data are on different charge scales" is true and
   irrelevant to the two tests T7 names.
2. **The peak-anchored origin already exists in-tree**, ported and hardened:
   `TaggerCheckSTM.cxx:2811-2889` (`eval_stm_core_impl`, `end_L = L[max_bin]
   + 0.2 cm` after a 5-point running-mean peak search; the prototype's
   `ToyFiducial.cxx:1551`). `CheckSTM_Michel` cites that function as the source
   of its KS window but its `rr` origin is the chain's geometric far end
   (`StmMichelFunctions.cxx:163-165`). §4 ports the anchoring.
3. **Three knobs are shared with production mechanisms**:
   `bragg_plateau_lo/hi_cm` and `profile_min_dqdx_frac` are also the retreat's
   and split's plateau / live definitions (`stop_retreat_max: 2`,
   `stop_split_max: 1`, PDVD production); `bragg_contrast_min` also guards the
   chain walk (`bragg_confirmed`). Any window move re-scores T1a and T1c; the
   tables below are the VERDICT's view of each window and do not include that
   coupling.
4. **The sampling sweep needs a copied JSON**, not a TLA key: the track-fitting
   file is read at runtime, whole, by three components per detector
   (`TaggerCheckSTM`, `CheckSTM_Michel`, `TaggerCheckNeutrino`), and only three
   of its keys are reachable through the knob bag. A byte-identical compiled
   jsonnet therefore does NOT mean the fit is unchanged — each sweep arm's JSON
   md5 is recorded beside the binary's.

## 2. Measurement 1 — the two shape tests, re-scored (owner decides)

Judged items with a valid profile on the round-2 production candidate
(`d63a`, 525 items, 261 scan stoppers). The verdict here is the two SHAPE
tests alone (contrast ≥ `bragg_contrast_min` × expected AND
`ks_flat − ks_mu > ks_margin`); the full chain adds the other reject bits
and reads TP 152 / FP 9 / F1 0.709 on the same items (shipped shape tests
alone: TP 162 / FP 16 / F1 0.726).

### 2.1 `bragg_contrast_min` × `ks_margin`, from the published fields

Cells are TP / FP and F1; shipped = 0.60 / 0.00:

| `ks_margin` | cmin 0.50 | 0.55 | **0.60** | 0.65 | 0.70 | 0.75 | 0.80 | 0.90 |
|---|---|---|---|---|---|---|---|---|
| −0.05 | 206/33 0.813 | 203/27 **0.815** | 196/21 0.808 | 191/16 0.804 | 177/11 0.776 | 172/8 0.768 | 155/7 0.721 | 110/4 0.576 |
| −0.02 | 186/20 0.785 | 185/18 0.786 | 181/16 0.778 | 177/12 0.775 | 167/9 0.752 | 163/7 0.744 | 148/6 0.701 | 108/4 0.568 |
| **0.00** | 165/19 0.730 | 165/18 0.732 | **162/16 0.726** | 160/12 0.727 | 153/9 0.712 | 150/7 0.706 | 138/6 0.670 | 103/4 0.549 |
| +0.02 | 135/14 0.647 | 135/13 0.649 | 133/12 0.644 | 132/12 0.641 | 130/9 0.639 | 129/7 0.639 | 121/6 0.613 | 96/4 0.522 |
| +0.05 | 95/7 0.514 | 95/7 0.514 | 95/7 0.514 | 94/7 0.509 | 93/6 0.507 | 93/5 0.508 | 90/4 0.497 | 73/4 0.423 |

**The KS margin is the lever, not the contrast.** Along the shipped
`ks_margin 0` row the contrast threshold trades 4 TP per 3 FP between 0.50
and 0.65 (F1 flat at 0.73); down the shipped `cmin 0.60` column, relaxing
`ks_margin` to −0.05 buys **34 TPs for 5 FPs** (F1 0.726 → 0.808). Doc 55
§15.2 explains why: an item WITH a Michel has a systematically weaker rise
in its fitted profile, and the KS test against the muon template is the
test that rise fails first (`shape_flat` is the dominant reject, doc 56 §4).
The best cell on this grid is `cmin 0.55, ks_margin −0.05` (F1 0.815); the
argued move is the margin alone. Both are operating points chosen on the
sample that motivates them and are **not moved here**.

### 2.2 The window knobs, recomputed from the profile and the reference curve

Agreement gate at the shipped windows (n 525): recomputed `ks_flat − ks_mu`
matches the published value to 1e-4 on every item; recomputed `contrast`
matches to a median 0.0000 / p90 0.033 / max 1.17 (the tail: the payload's
profile is rounded to 0.1 e/cm and 0.01 cm and the C++ live profile also
drops `dx ≤ 0` rows — the recomputation is a faithful instrument for a
sweep, not a bit-identical one). Shape tests alone, shipped otherwise:

| knob | setting | TP | FP | FN | purity | eff | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| `bragg_tail_hi_cm` | 2.0 | 121 | 11 | 147 | 0.917 | 0.451 | 0.605 |
| | **3.0 (shipped)** | 162 | 16 | 106 | 0.910 | 0.604 | 0.726 |
| | 4.0 | 164 | 18 | 104 | 0.901 | 0.612 | 0.729 |
| | 5.0 | 164 | 20 | 104 | 0.891 | 0.612 | 0.726 |
| `bragg_plateau` | 15–30 cm | 164 | 15 | 104 | 0.916 | 0.612 | 0.734 |
| | **20–40 (shipped)** | 162 | 16 | 106 | 0.910 | 0.604 | 0.726 |
| | 25–50 | 162 | 12 | 106 | 0.931 | 0.604 | 0.733 |
| | 20–60 | 163 | 12 | 105 | 0.931 | 0.608 | 0.736 |
| `compare_range_cm` | 25 | 147 | 10 | 121 | 0.936 | 0.549 | 0.692 |
| | **35 (shipped)** | 162 | 16 | 106 | 0.910 | 0.604 | 0.726 |
| | 45 | 168 | 14 | 100 | 0.923 | 0.627 | 0.747 |
| | 60 | 172 | 11 | 96 | 0.940 | 0.642 | **0.763** |

The tail and plateau windows are flat to ±0.01 in F1 — the shipped values
are not what limits the tests. `compare_range_cm` is not flat: a longer KS
window (60 cm) gains 10 TPs and drops 5 FPs (F1 0.726 → 0.763), consistent
with §2.1 — the KS test on 35 cm of a rising profile is too easily failed,
and more plateau in the comparison helps it. `compare_range_cm` also drives
`do_track_comp` and the live-fraction diagnostics (§1.3), so a move there is
a three-consumer change. Recommendation for the owner, in order:
`ks_margin` −0.02 to −0.05 (one knob, one consumer, +16 to +34 TP for +0 to
+5 FP on the shape tests alone), then `compare_range_cm` 45–60. Neither is
moved in this round.

## 3. Measurement 3 — the profile's autocorrelation length

Plateau rows (rr 20–80 cm, q > 0), residual of log q about a 9-point running
median, lag-k autocorrelation pooled over items:

| lag (points) | lag (cm) | PDVD `d53v` (516 items) | PDVD `d63a` (513) | PDHD `d53h` (274) |
|---:|---:|---:|---:|---:|
| 1 | 0.6 | **0.418** | 0.414 | **0.601** |
| 2 | 1.2 | −0.119 | −0.121 | 0.056 |
| 3 | 1.8 | −0.263 | −0.261 | −0.267 |
| 4 | 2.4 | −0.200 | −0.198 | −0.317 |
| 5 | 3.0 | −0.065 | −0.062 | −0.187 |
| 6–8 | 3.6–4.8 | ≈ 0 | ≈ 0 | −0.09 → −0.01 |

Point-to-point |Δ log q| 0.113 (PDVD) / 0.087 (PDHD), doc 56 §6's numbers
reproduced. The autocorrelation is positive at lag 1 only — **0.42 on PDVD,
0.60 on PDHD** — and below 1/e from lag 2 (1.2 cm) on both detectors; the
negative lobe at lags 2–5 is the running-median detrending. So the fit's
dQ/dx is smoothed over about two 0.6 cm samples on PDVD and closer to three
on PDHD — the smoother (`dx_norm_length` 6 mm, identical on both) acts on a
PDHD profile that is intrinsically smoother (0.479 vs 0.765 cm pitch, more
wires per cm). The tail median over 0.5–3 cm averages ~4 points of which ~2
are independent on PDVD; that is the effective sample size behind
`n_tail >= 3`. A step change (§4) is the direct test of whether this matters.

## 4. Measurement 2 — the sampling sweep (owner decides; nothing moved)

Four arms on round 3's pinned binary (`b9e44ec66996`), each a scratch copy of
the shipped `pdvd_track_fitting.json` with ONE key changed (md5s: shipped
`eaa1dd67a8b6`; ld 8 `05beb2f50f8c`; ld 16 `cbd0ba17482a`; dx 4 `5fddfd9b0c18`;
dx 9 `0738e9908725`), scored against `d64vleg` (the same binary at the shipped
file). The JSON is read by the tagger AND the Michel stage, so both fits
move — and with them the candidate set itself:

| arm | `low_dis_limit` / `dx_norm_length` (mm) | payloads | **unmatched** | `is_stm` TP / FP / FN | purity / eff / F1 | `michel` TP / FP / FN | F1 |
|---|---|---:|---:|---|---|---|---:|
| `d64vleg` | **12 / 6 (shipped)** | 566 | 3 | 152 / 9 / 116 | 0.944 / 0.567 / 0.709 | 132 / 22 / 20 | 0.863 |
| `d65s08` | 8 / 6 | 402 | **167** | 121 / 6 / 92 | 0.953 / 0.568 / 0.712 | 101 / 31 / 23 | 0.789 |
| `d65s16` | 16 / 6 | 433 | **136** | 120 / 8 / 105 | 0.938 / 0.533 / 0.680 | 105 / 27 / 21 | 0.814 |
| `d65d4` | 12 / 4 | 534 | 35 | 146 / 7 / 110 | 0.954 / 0.570 / 0.714 | 129 / 19 / 18 | **0.875** |
| `d65d9` | 12 / 9 | 539 | 30 | 147 / 8 / 113 | 0.948 / 0.565 / 0.708 | 127 / 25 / 21 | 0.847 |

**The step is not a parameter of the Michel stage; it is a parameter of the
whole STM tagging.** Moving `low_dis_limit` by ±4 mm changes which clusters
`TaggerCheckSTM` flags `STM` at all — 167 / 136 of the 569 scan items no
longer exist as candidates (they are UNMATCHED, not mis-scored, the same
reading doc 59 established for T1b's 3), and among the ~400 that remain
`is_stm` flips on 68 / 63 items in both directions (ld 8: 28 TPs gained, 59
lost, 5 FPs new, 8 removed). The scan record cannot grade a candidate set it
never saw, so the per-item purity/efficiency on the matched subset (0.953 /
0.568 at 8 mm) is not comparable to the shipped row and is quoted only to
show it is not obviously better. What the record does say: the pin residual
(median 5.27 → 3.71 cm, within 2 cm 3 → 8 at ld 8 — on 30 not 36 pins),
class G (coiled ends) 15 → 21 at 8 mm and 15 → 6 at 16 mm, class L
`profile_sparse` 29 → 50 at 16 mm, and the shape census (collapse-shaped
found stoppers 12 → 20 at 8 mm, 5 at 16 mm) all move the way a finer /
coarser resampling of the same profile should. Michel attachment: role 3
225 → 148 / 155 with 42 / 36 scan-michel segments LOST (no fitted segment
there at all) at 8 / 16 mm — the PR's own segmentation changes under the
step.

`dx_norm_length` (the smoother's length scale, the knob doc 56 §6 named as
"the most direct Bragg-relevant") is the gentler lever: 35 / 30 unmatched,
11 / 9 `is_stm` flips. At **4 mm** the shape census barely moves, F/H by
name are kept (3/3, 15/16), `is_stm` gains 4 TPs and loses 10 (net −6, FP
9 → 7), and `michel_found` reaches the best F1 of the campaign (0.875: FP
22 → 19, FN 20 → 18); at 9 mm it goes the other way (michel F1 0.847). The
autocorrelation (§3) says the smoother is already averaging ~2 samples on
PDVD; 4 mm is the less-smoothed direction, and the census prefers it mildly
for the Michel and not for the stopper. **No JSON value is moved**: the
change reaches three components and a candidate set the record only
partially covers. If the owner wants to pursue `dx_norm_length` 4 mm, the
gate is a re-scan of the 35 unmatched items plus the 10 lost `is_stm` TPs
by name (`$W/byname_d65d4.txt`), not this table.

## 5. The knob: `bragg_peak_anchor` — measured, NOT flipped

`d65v` (anchor on, 10 cm search window) vs `d65vleg` (same binary, off):

| | `is_stm` TP | FP | FN | purity | eff | F1 | `michel_found` |
|---|---:|---:|---:|---:|---:|---:|---|
| leg | 152 | 9 | 116 | 0.944 | 0.567 | 0.709 | 132 / 22 / 20 |
| anchor 10 cm | **182** | **22** | 86 | 0.892 | 0.679 | 0.771 | identical |

71 `is_stm` flips on 566 items: **38 TPs gained, 8 lost, 19 FPs new, 6
removed**. The anchor fires (`bragg_anchor_shift_cm > 0`) on **404 of 566
items**, median shift 5.3 cm, p90 9.4 cm — i.e. within a 10 cm window the
5-point running-mean maximum is almost never the last row, so on most items
the origin moves several cm back and the tail / KS windows read a different
stretch of profile. That is what buys the 38 (rise-to-end and
collapse-shaped stoppers whose peak sits a few cm before the fit end — the
stop overshoot doc 54 §1 describes, read correctly for the first time on
`039252_15/81`, `039252_17/88`, `039253_15/45`, `039253_8/28`, `039349_1/52`,
`039349_19/52`, … all `STM_MICHEL` with an attached Michel at KE 4–39 MeV),
and it is also what costs the 19: through-going tracks with a hot spot in
their last 10 cm now read as a Bragg peak (`039252_2/79` and `039252_4/55`
are two of T2c's moved-stop THRU items; their anchored contrast/expected is
0.81 and 0.63; the 19 new FPs range 0.63–1.19 against the 0.60 bar). The 8
lost TPs are Michel-carrying stoppers whose anchored window lands on the
Michel's own dilution (`039253_2/89`, `039253_3/66` at 18–37 MeV) or on a
sparse stretch.

Against every prior flip's bar — 0 new `is_stm` FPs, purity ≥ 0.93 — this
fails clearly (purity 0.944 → 0.892), even though F1 rises 0.709 → 0.771.
It is the same trade §2.1's `ks_margin` row offers by a different route:
efficiency for purity, on a record that tags by eye. **Not flipped**; the
owner has both levers with their item lists (`$W/byname_d65v.txt`). A
narrower search window (`bragg_peak_search_cm: 3`, arm `d65v3`, run on the
round-5 binary against its leg) is measured in §5.1 to show whether the
FP cost is the window or the idea.

### 5.1 The 3 cm window (`d65v3`, round-5 binary, vs its leg `d66vleg`)

| | `is_stm` TP | FP | FN | purity | eff | F1 |
|---|---:|---:|---:|---:|---:|---:|
| leg | 152 | 9 | 116 | 0.944 | 0.567 | 0.709 |
| anchor 10 cm | 182 | 22 | 86 | 0.892 | 0.679 | 0.771 |
| **anchor 3 cm** | **168** | **10** | 100 | **0.944** | 0.627 | **0.753** |

The narrow window fires on 342 of 566 items (median shift 2.3 cm) and flips
37: **23 TPs gained, 7 lost, 4 FPs new (`039252_2/79`, `039349_14/22`,
`039349_43/66`, `039349_72/11`), 3 removed (`039252_8/93`, `039349_21/26`,
`039349_22/45`)**. `michel_found` identical. Purity is unchanged at 0.944 and
efficiency rises 0.567 → 0.627 — the same efficiency the `ks_margin −0.02`
row buys, by reading the profile from its peak instead of loosening the KS
bar. The 7 lost TPs (`039252_14/81`, `039253_3/66`, `039253_6/85`,
`039349_43/54`, `039349_72/65`, `039349_76/75`, `039349_9/55`) are
Michel-carrying or short stoppers whose 3 cm-anchored windows land on the
Michel's dilution or too few live rows; `039252_14/81` is the same fragile
item C alone also lost in doc 62.

**Not flipped.** It is the best `is_stm` operating point this campaign has
seen (+16 net TP at unchanged purity, F1 0.709 → 0.753), but it is the
second window tried after seeing the first (a choice made on this record),
and it moves `is_stm` on 37 items with a swapped FP set — not the 0-new-FP
bar every `is_stm` flip in docs 57–63 met. It is the owner's candidate:
`-S stm_michel_extra={bragg_peak_anchor:true,bragg_peak_search_cm:3.0}`, with
the 4 new FPs and 7 lost TPs above as the items to look at first. A "rise"
precondition on the peak (the anchored contrast must exceed the geometric
one, or the peak row must sit above the plateau by some factor) is the
obvious next refinement and is not built here.

## 6. Gates

| gate | result |
|---|---|
| `./build/clus/wcdoctest-clus` | 357 / 357 (two new default pins) |
| `d65vleg` vs `d64vleg` (PDVD, `--pts`) | 578 / 578 identical point geometry, 578 / 578 bit-identical on all 124 shared branches, 0 role labels moved, 0 `is_stm` flips; new branch `bragg_anchor_shift_cm` only — **PASS** (`$W/gate1_pdvd.txt`) |
| `d65hleg` vs `d53h` (PDHD) | 325 / 325 identical, 0 moved, 0 flips — **PASS** (`$W/gate2_pdhd.txt`) |
| binary pin | `libWireCellClus.so` md5 `0af3b1e04a88` before and after every round-4 arm; the four sweep arms ran on round 3's `b9e44ec66996` with their JSON md5s recorded in §4 |
| `census_score.py --check` | 0 of 14 differ |
| window recomputation agreement (§2.2) | KS to 1e-4 on all 525; contrast median 0 / p90 0.033 |
| production config | untouched this round (no flip); no flip-check needed |

## 7. Found on the way, not fixed

1. **`ks_margin` is the dominant shape-test lever** (§2.1): −0.05 buys 34 TPs
   for 5 FPs on the shape tests alone. Unchanged: a threshold chosen on the
   sample that motivates it; the owner's call.
2. **`compare_range_cm` 60** gains 10 TPs / −5 FPs on the KS test (§2.2) but
   feeds three consumers (`do_track_comp`, the live-fraction diagnostics).
   Unchanged.
3. **The step moves the candidate set** (§4): `low_dis_limit` ±4 mm changes
   which clusters the tagger flags at all (136–167 of 569 items unmatched).
   Any sampling change needs a re-scan, not this record.
4. **`dx_norm_length` 4 mm** is the one sampling setting the record mildly
   prefers for the Michel (F1 0.863 → 0.875) and mildly disfavours for the
   stopper (net −6 TP); 35 items unmatched. Unchanged.
5. **The peak anchor at 10 cm is too eager** (§5): it moves the origin on 71 %
   of items. Whether a narrower window keeps the 38 recoveries without the
   19 FPs is §5.1's question; if it does not, the idea needs a "is this a
   rise" test before the anchor (the prototype anchors on the tagger's own
   kink-bounded profile, which this port does not have).
6. **`stm_recomb_calibrated`'s note in doc 56 §8 T7 is corrected** (§1.1): the
   knob is on in production and does not reach the shape tests.

## 8. Doc 56 update

T7 row marked done (one knob measured and left OFF; three measurement tables
delivered; the recommended operating points named for the owner — `ks_margin`
first, `compare_range_cm` second, `dx_norm_length` 4 mm as the one sampling
lever worth a re-scan); §6's `stm_recomb_calibrated` and step bullets
corrected; Order moves to T8. Scripts committed:
`scripts/d65_shape_thresholds.py`, `scripts/d65_autocorr.py`.
