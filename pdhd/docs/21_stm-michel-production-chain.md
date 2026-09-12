# doc pdhd/21 — grading PDVD's STM/Michel knobs on the PDHD hand-scan record, and the first PDHD flip

**Status:** four knobs flipped in PDHD production (`ks_margin`, `max_candidates`,
`bragg_peak_anchor` + `bragg_peak_search_cm`, `bragg_anchor_geo_fallback`), graded on the
`smx22` record. `is_stm` **61/0/86/110 → 80/0/67/110, purity 1.000 HELD, efficiency 0.415 →
0.544, zero new false positives.** No C++ change; the flip is a config-only change to
`pdhd/wct-pr-perevt.jsonnet`'s `stm_michel_knobs` bag. Two further sets are measured and
**not** flipped, each because it costs purity — the owner's call, not the agent's.

## 0. Repro

```sh
I=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img ; H=/home/xqian/tmp/h21
PIN=/home/xqian/tmp/p82/libpin_p82          # toolkit 082376c5, libWireCellClus md5 5f2c3ede
S=$I/pdvd/docs/nf_sp_img_clus/scripts/d53_run_arms.sh

# 1. an arm (5-10 min for the 61 events).  NEVER -S stm_michel_knobs= : that REPLACES the bag.
ARM=h21f DET=pdhd SRC=d16hnu JOBS=6 PIN=$PIN LOGD=$H/arm_h21f \
  PR_TLA="-S stm_michel_extra={ks_margin:-0.02,max_candidates:64,bragg_peak_anchor:true,bragg_peak_search_cm:3.0,bragg_anchor_geo_fallback:true}" \
  bash $S

# 2. grade it against smx22 straight from tracking-pr.root (no prep needed).  The script
#    REFUSES to print any arm number unless p82bhoff first reproduces the committed 61/0/86/110.
python3 $I/pdhd/docs/scan/h21/d21_grade.py h21f
python3 $I/pdhd/docs/scan/h21/d21_michel_census.py p82bhoff h21f     # the Michel side

# 3. the movers gate: every branch, not a hand list
python3 $I/pdvd/docs/nf_sp_img_clus/scripts/d51g_branch_census.py \
  --before "$I/pdhd/work/*_p82bhoff" --after "$I/pdhd/work/*_h21f" \
  --before-arm p82bhoff --after-arm h21f --pts --out $H/gates/g_h21f.txt
```

## 1. Where PDHD stood, and why nothing had moved

Doc pdhd/18 produced the PDHD record (`smx18` → `smx22`, 317 items over 61 events). Against
it, PDHD production read **`is_stm` purity 1.000 / efficiency 0.415** — 86 missed stoppers and
**zero** false ones. The chain was not making mistakes; it was refusing to commit. 76 of the 86
misses carry `shape_flat`, 45 carry `no_bragg`, only 12 carry `plateau_off_mip`.

The cause was structural. PDVD's `stm_michel_knobs` bag holds **45 keys**; PDHD's held **14**.
Thirteen were identical and `plateau_mip_hi` differed. The other **31 PDVD knobs were simply
absent**, so the C++ default applied — and the reason is recorded 19 times in the PDVD jsonnet:
*"no PDHD hand-scan record"*. That record now exists.

## 2. The instrument

* **`census_lib.py` ported to `--det pdhd`** (`STM_DET`, defaulting to `pdvd`). The 30 PDVD
  scripts that import it are byte-unchanged; a bad detector fails loudly. **`MIP_MEDIAN` = 48000
  was read from the COMPILED config, not the jsonnet source** — the standing rule, and the same
  shortcut `census_lib`'s own PDVD comment had taken. `STM_BITS` bit order was validated by
  decoding `reject_bits` → `reject_names` on **303/303** key rows, not assumed to be shared.
  **`census_score.py` was deliberately NOT ported**: the knob graders import `census_lib` only,
  and `census_score --check` wants PDVD doc-55 literals with no PDHD counterpart.
* **`d21_grade.py`** grades an arm against `smx22` directly from `tracking-pr.root`, applying
  `d18_census.py`'s owner precedence (`owner_review` > `owner_smx1` > agent) — which
  `census_lib`/`census_score` do **not** implement. It refuses to print anything unless
  `p82bhoff` first reproduces the committed census.
* **`d21_michel_census.py`** grades `michel_found` against the hand `michel_kind` on hand
  stoppers, excluding (never scoring as "no Michel") an owner stopper carrying no kind.
* Every arm: compiled-config proof before launch, `complete 61, incomplete 0, rc=0`, and the
  pin md5 `5f2c3ede7076` identical before and after — no arm is unattributable.

## 3. What was graded

All arms on `libpin_p82`, against the existing `p82bhoff` baseline, population fixed at the
committed 303 judged items.

| arm | knobs | `is_stm` | purity | eff | `michel_found` |
|---|---|---|---|---|---|
| `p82bhoff` | production | 61/0/86/110 | 1.000 | 0.415 | 56/11/30 (0.836/0.651) |
| `h21k` | `ks_margin:-0.02` | 72/0/75/110 | **1.000** | 0.490 | unchanged |
| `h21m` | `max_candidates:64` | (see below) | 1.000 | — | unchanged |
| `h21g` | anchor + search 3.0 + geo fallback | 71/0/76/110 | **1.000** | 0.483 | unchanged |
| `h21w` | + wide anchor 8.0/1.3/0.8 | **identical to h21g** | 1.000 | 0.483 | unchanged |
| `h21s` | retreat 2 / split 1 / kink 10 | 61/0/86/110 (**+0**) | 1.000 | 0.415 | 58/10/28 |
| `h21j` | `h21s` + the Michel bag | 63/0/84/110 | 1.000 | 0.429 | **67/2/19** (0.971/0.779) |
| `h21x` | `ks_margin` + cap | 72/0/75/110 | 1.000 | 0.490 | unchanged |
| **`h21f`** | **the flipped set** | **80/0/67/110** | **1.000** | **0.544** | unchanged |
| `h21z` | `h21f` + `h21s` + Michel bag | 82/**1**/65/109 | 0.988 | 0.558 | **67/2/19** |
| `h21a` | `topology_stop_evidence` | 74/**1**/73/109 | 0.987 | 0.503 | unchanged |
| `h21b` | + `topology_clears_sparse` | 76/**3**/71/107 | 0.962 | 0.517 | unchanged |
| `h21c` | + `topology_michel_ke_min:3` | 78/**4**/69/106 | 0.951 | 0.531 | unchanged |

**`max_candidates`** cannot show in a census whose population is fixed at the 303: its gain is
in the 14 candidates production never reaches. Arm `h21m` showed all **325 existing candidates
BIT-IDENTICAL with 0 `is_stm` flips**, set 325 → 341 — PDVD doc 79's precondition — and the
committed `h18b` key scores the extras at **+5 hand stoppers / 0 FP**.

## 4. The pre-registered twins: two held exactly, two failed

Written before each arm existed (`docs/scan/h21/preregistered_twin.txt`), naming every predicted
mover **by item**, not by count.

* **HELD EXACTLY (0 missing, 0 unpredicted):** `h21k`, `h21a`, `h21b`, `h21c` — all four wave-1
  arms, including the single new FP `028084_12/46`, and `michel_found` moving on 0 items.
* **HELD EXACTLY:** `h21x`. `ks_margin` and `max_candidates` act on disjoint populations and do
  not interact, as predicted.
* **FAILED:** `h21f` gained **19**, not the predicted 17 (the naive union of `h21k`'s 11 and
  `h21g`'s 10, overlapping on 4). Two items move only in combination: `028084_12/101` and
  `029107_19/30`. The mechanism was named in the falsifier and is confirmed in the data — the
  anchor shifts the peak 2.80 cm, which clears `no_bragg` *and* pulls the KS margin from −0.056
  to −0.001, inside the −0.02 window. **Neither knob alone suffices.**
* **FAILED (stopper side) / HELD (Michel side):** `h21z`'s Michel census landed on the predicted
  67/2/19 exactly, but its stopper side gave 82/1/65/109 against a predicted 80/0/67/110 — +2
  stoppers and **one new FP**. Again the falsifier named the mechanism: the Michel bag moves
  point geometry on 108 of 325 candidates, and the stopper tests read the profile built from
  those points.

**Consequence, and it governs the flip:** a knob set must be graded as a unit. Assembling a
predicted gain by adding single-knob arms is wrong in both directions here.

## 5. What does NOT transfer from PDVD

This is the campaign's most useful negative result: PDVD's knobs are not portable on their own
authority.

* ~~**The wide Bragg anchor** — PDVD's doc 93 flip, worth +4 stoppers for 1 FP there — recovers
  **nothing** on PDHD. `h21w` is identical to `h21g` on every count and the same 10 items.~~
  **RETRACTED 2026-09-12 by doc pdhd/22 §1. This claim was VOID, not merely wrong.**
  `bragg_wide_anchor_cm` was introduced in `ec6990e0` (doc pdvd/92), *after* the `082376c5`
  pin every arm in this round ran on. The binary did not implement the key and dropped it
  silently, so `h21w` being identical to `h21g` is the signature of an **inert key**, not a
  measurement. Re-measured on a binary that implements it (arm `h22w`, pin `libpin_p65` =
  `d65f8165`): the wide anchor recovers **+1 stopper (`028084_21/132`) at ZERO false
  positives**, purity 1.000 held — less than PDVD's +4-for-1-FP, but it costs nothing here.
* **`plateau_mip_hi` 1.6 → 2.0** — PDVD's doc 90 flip, +1 at 0 FP there — recovers **zero**.
* **No plateau window move earns its keep, length-aware or not.** This answers the open item of
  doc pdhd/20 §9.1, negatively:
  * `short_track` (the window already halves below 40 cm) fires on **5 of 86** misses;
  * the 12 `plateau_off_mip` misses are **long** tracks (138–446 cm, median 34 plateau points)
    whose whole plateau reads **0.28–0.56 MIP** — a charge-scale problem, not a length one;
  * the hand-THRU items rejected on the same bit read **0.22–0.54 MIP** — *interleaved* with the
    misses, so **no window position separates them**. A sweep confirms it: lo 0.5 → +2/+1 FP,
    0.4 → +3/+1, 0.3 → **+4/+5**.
* **Stop topology** (`stop_retreat_max` 2, `stop_split_max` 1, `split_kink_min_deg` 10) — PDVD
  gained +2 and +1 — gains **0 stoppers** on PDHD.

## 6. The flip

Four knobs into `pdhd/wct-pr-perevt.jsonnet`'s `stm_michel_knobs` bag (five keys — the anchor
carries its search window). Gates, all PASS:

| gate | result |
|---|---|
| compiled-config proof, **no TLA**, vs the pre-edit file | **5 keys added, 0 removed, 0 changed** |
| other live jobs | `run_pr_evt.sh` is the file's only consumer, so the diff above is that check |
| **confirmation arm `h21p`** — the flipped file, **no TLA**, vs the TLA arm `h21f` | **341/341 candidates BIT-IDENTICAL on all 131 branches; 0 `is_stm` flips; point geometry identical 341/341; 0 role moves** |
| `h21p` census vs `h21f` | identical: **80/0/67/110**, purity 1.000, efficiency 0.544 |
| `h21p` Michel side | unchanged, **56/11/30** |
| binary pin across every arm | md5 `5f2c3ede7076` before and after, all 12 arms |

**Production runs exactly what was measured** — that is what the confirmation arm establishes,
and it is why the flip is quotable rather than merely plausible.

They were flipped **without** an owner ruling because there was no trade to rule on: purity
1.000 is held and there are zero new false positives.

## 7. What is NOT concluded

* **Not** that the flipped set is optimal. It is the zero-cost subset.
* **Not** the Michel bag. `h21z` lifts `michel_found` from 56/11/30 to **67/2/19** — purity
  0.836 → 0.971 *and* efficiency 0.651 → 0.779, +11 true and −9 false Michels — but spends **1
  stopper false positive**, breaking purity 1.000. Per the owner's 2026-09-11 ruling ("decide
  per knob as sized"), that is their call. **Recommended.**
* **Not** attributed within the Michel bag. It ran as one 13-knob bag coupled to stop topology
  (`retreat_tail_strict` needs `stop_retreat_max > 0`). Which Michel knob does the work is
  unknown, and `h21s` shows ~2 of the +11 belongs to stop topology.
* **Not** P1. Every variant costs purity (0.987 / 0.962 / 0.951). `h21c` additionally flips 5
  UNCLEAR and 1 MESSY item — pushing the chain into the population scanners could not judge.
* **Not** free of chain movement. `h21f` is only **167/325** bit-identical with 20 `is_stm`
  flips, though point geometry is identical **325/325**, so no fit moved. `h21z` moves geometry
  on 108 candidates — a different class of change.
* **Not** a re-judged record. Per the owner's ruling this round grades on `smx22` as it stands;
  both truth readings were reported where they differ.

## 8. What is left, ranked

1. **The owner's ruling on the Michel bag** (`h21z`): +11/−9 Michels and +2 stoppers for 1
   stopper FP. The largest measured gain still on the table.
2. **Attribute the Michel bag** knob by knob, if the ruling is favourable.
3. **The owner's ruling on P1**, which is a straight purity-for-efficiency trade.
4. The 31-knob inventory is now graded down to these; the remainder are measured dead on PDHD.

## Files

| what | where |
|---|---|
| pre-registered twins (all arms, written before each) | `docs/scan/h21/preregistered_twin.txt` |
| censuses | `docs/scan/h21/census_{wave1,wave2a,wave2b,h21f,h21z,michel}.txt` |
| movers gates | `docs/scan/h21/g_h21*.txt`, `gates_wave1.txt` |
| graders | `docs/scan/h21/d21_grade.py`, `d21_michel_census.py` |
| the flip | `pdhd/wct-pr-perevt.jsonnet` `stm_michel_knobs` |
| ported library | `pdhd/stm_michel_scan/census_lib.py` (`STM_DET`) |
