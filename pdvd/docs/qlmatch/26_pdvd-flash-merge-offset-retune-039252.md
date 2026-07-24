# 26 — Flash tail-merge + QL offset removal / cut+velocity retune campaign (run 039252)

Status: **COMPLETE** (2026-07-22). **ADOPTION (owner, 2026-07-22): the flash
tail merge is the PDVD production default** (`run_light_evt.sh`
`PDVD_FLASH_TAIL_MERGE` 0→1; option (a) of the doc-28 comparison —
agreement-neutral, fixes the track-C split-match defect, keeps the pulled
frame/velocity/cuts). The offset-0 retune (steps 5-6) was NOT adopted; the
rc14 point stays opt-in (docs 27-28).

Adoption Bee sets (run 039252 scan set, 18 evts; bee idx = charge idx 0-17,
crossers at idx 3 = evt 298609 tracks A/B, idx 6 = evt 298651 track C):

- **adopted production (tail merge ON, tag tm0)**:
  <https://www.phy.bnl.gov/twister/bee/set/a6c3d73c-ddda-4945-8dfe-3bee459b3f6f/event/list/>
- pre-merge baseline (tag tm0k):
  <https://www.phy.bnl.gov/twister/bee/set/8e180ef8-be2a-43ac-86bf-aaec657fcce2/event/list/> Owner-directed follow-up to doc 23
(`pdvd/docs/23_pdvd-light-timing-check.md`, §4/§6/§7): fix the split-flash
defect in the light reconstruction, remove the 13.507 µs production QL pull,
retune the anode/cathode containment cuts and the drift velocity, and carry
the existing hand/AI-scan verdicts across the reprocess by mapping instead of
re-scanning.

## Repro

```
# split-pair anatomy + raw-peak evidence (doc 23 §7d)
cd pdvd/docs/qlmatch && python3 scripts/aca_flash_split.py --raw
# channel-vs-time raw/decon maps of the pair (doc 23 §7d figures)
python3 scripts/aca_flash_wf.py
# position ladder for the retune (doc 23 §7c)
python3 scripts/aca_positions.py
```

Baseline production op point (all runner-env defaults as of the doc-23 state):
`PDVD_QL_EXTRA_OFFSET_US=13.507`, `PDVD_QL_ANODE_MARGIN_CM=2.0`,
`PDVD_QL_CATHODE_EXT1_CM=2.0`, `PDVD_DRIFT_SPEED_{BOT,TOP}_MMUS=1.48073`,
light tags `work/039252_light<evt>_keep/`.

## 1. Problem recap (from doc 23)

1. **Flash split (§7d).** OpFlashFinder cuts ONE physical flash into two
   members 1.17–1.31 µs apart (tracks B/C). The late member is 97–99.9%
   cathode-XA PE from LAr slow-tail ophits 1.6–1.7 µs wide on OpDets *already
   lit* in the fast member; narrow wall-XA/PMT hits attach to it only because
   their self-trigger records are +0.7–0.9 µs late (§4), and they carry <3% of
   its PE. Track C's cluster matched the LATE member (raw 2801.95) while the
   brighter fast member (2800.64, 13.9k PE) went unmatched. The channel-vs-time
   maps (`pics/pdvd_flash_split_wf_{B,C}.png`) show one connected cathode pulse
   crossing both flash times with no second onset at the late member.
2. **The 13.507 µs pull (§7).** `PDVD_QL_EXTRA_OFFSET_US=13.507` (≡ 2.0 cm at
   v=0.148073 cm/µs) is a charge-placement compensation, not a flash-time
   error; light↔charge absolute closure holds (§7 A-C-A crossers). It is to be
   removed and compensated by the position cuts and a slight velocity
   reduction, tuned against the crossers and the mapped scan verdicts.

## 2. Fix-candidate evaluation: hit-level vs flash-assembly

Two candidate fixes for the split (doc 23 §7d closing paragraphs):

**(a) Hit-level** — stop `split_pulse` from cleaving the slow-tail sub-peak,
or book wide tail hits at their start time (`wide_hit_mode='start'`).
Rejected:

- The cathode ophit branch has **no wide-hit plumbing at all**
  (`wct-light-reco.jsonnet:127` — only the mem/pmt branches take
  `wide_hit_mode`, lines 137–151), and the tail hits (1.6–1.7 µs) sit *below*
  the 2.0 µs `wide_hit_min_width_us` default anyway; enabling
  start-mode-with-lower-threshold on the cathode would rebook **every** ≳1 µs
  hit detector-wide and pull flash times toward hit onsets globally.
- Suppressing the split needs `split_min_prominence_abs` above the tail
  sub-peak amplitude (C: 4.3k-PE hit) — that would also stop separating
  genuine pile-up, which shares the same hit-level shape. The hit-level view
  cannot distinguish "my own slow tail" from "a second flash on the same
  channels"; only the flash level knows which channels the seed already lit.

**(b) Flash-assembly merge** — absorb a later dim flash whose PE is dominated
by wide late-tail hits on the seed's own lit channels. **Chosen.** The
criterion is exactly the observed signature, is inert for genuine pile-up
(narrow second onset fails the width test; a different track fails the
lit-channel test; a brighter flash fails the PE-ratio cap), and tolerates the
self-trigger passengers naturally through a PE-dominance fraction. The
existing `flash_refine` satellite merge cannot catch these pairs (PE ratio
0.53–0.82 > 0.5, 4–23 fired PDs, not subsets), so a new gate is required
either way.

**Merged-flash time (owner decision 2026-07-22): keep the seed's fast-peak
time/time_width.** The PE-weighted recompute used by the legacy satellite
merge would drag B's merged time ~+0.5 µs late; physically the flash t0 is the
scintillation onset, and keeping the seed time makes the old→new scan mapping
exact for fast-member verdicts. (x-placement difference between the two
choices is only ~0.07 cm at v=0.148 cm/µs — the choice is about semantics,
not matching power.)

## 3. Design: `flash_tail_merge` knob in OpFlashFinder (default OFF)

New config keys (C++ defaults; all round-tripped in
`default_configuration()`):

| key | default | meaning |
|---|---|---|
| `flash_tail_merge` | false | master knob; absent ⇒ byte-identical legacy |
| `tail_window_us` | 3.0 | absorb flash j with `t_j − t_i ≤ window` |
| `tail_min_width_us` | 1.0 | hit counts as slow-tail if width ≥ this |
| `tail_pe_frac` | 0.7 | fraction of j's PE that must be wide + seed-lit |
| `tail_pe_ratio` | 1.0 | sanity cap `PE_j ≤ ratio · PE_i` |

Gate (inside `refine_flashes`, which runs when `enabled || tail_merge`):
`ok_tail` iff `dt ≤ tail_window` AND `PE_j ≤ tail_pe_ratio·PE_i` AND
`Σ pe(hits of j with width ≥ tail_min_width on OpDets with seed.pes ≥
fired_pe) ≥ tail_pe_frac · PE_j`. Legacy criteria (`ok_legacy`) apply
verbatim only when `flash_refine` is on; the two are OR-ed. On a tail-only
merge the seed's `time`/`time_width` are restored after `construct_flash`.
The row/col `adjacent()` grid is NOT consulted (it is built with
horizontal-drift assumptions, meaningless for PDVD).

Observed pairs vs the gate: B late member 19.9k/24.4k PE (ratio 0.82, wide-lit
frac ≳0.97), C 7.4k/13.9k (0.53, ≳0.999) — both pass; track A (already single)
unaffected. Wall-XA/PMT passengers (<3% PE) ride along inside the 0.7
dominance margin.

PDHD safety: PDHD runs `flash_refine:true, refine_subset_merge:true`; the tail
knob defaults OFF and the legacy path is restructured provably-identically
(the one entangled line is the loop's window break — widened only when
tail_merge is on). Knob-off byte-identicality is gated on both PDVD and PDHD
light outputs (§5 checklist).

## 4. Retune parameter map (all runner envs — no code change needed)

| parameter | runner env (default) | compiled key | C++ consumer |
|---|---|---|---|
| extra offset "pull" | `PDVD_QL_EXTRA_OFFSET_US` (13.507) | folded into `trigger_offset(s)` | `QLMatching.cxx:1621` (charge-x via flash time) |
| cathode cut | `PDVD_QL_CATHODE_EXT1_CM` (2.0; C++ 1.2) | `cathode_ext1` | `QLMatching.cxx:1777/4450` |
| anode margin | `PDVD_QL_ANODE_MARGIN_CM` (2.0; C++ 1.0; floor = −2−margin) | `anode_ext1_margin` | `QLMatching.cxx:4578` |
| drift velocity | `PDVD_DRIFT_SPEED_{BOT,TOP}_MMUS` (1.48073) | `drift_speed(s)` + clus `drift_speed_b/t` | `QLMatching.cxx:1621` + BlobSampler (clus.jsonnet) |

Removing the pull shifts charge 2.0 cm shallower (toward the anode); the
compensation budget is split between the anode margin, the cathode ceiling,
and a slight velocity reduction, re-derived from the A-C-A crossers at
offset 0 and validated on the mapped scan set.

## 5. Campaign steps and acceptance bar

1. **This doc** (plan + fix evaluation). *(this commit)*
2. **Toolkit `flash_tail_merge` knob**: OpFlashFinder + new
   `doctest_opflashfinder.cxx` + PDVD `flash.jsonnet` key-suppressed args +
   runner threading (`wct-light-reco.jsonnet`, `run_light_evt.sh` env
   `PDVD_FLASH_TAIL_MERGE`, default 0). Gates: wcdoctest-flash; compiled-config
   proof both ways; knob-off `hash_archive.py` PASS on PDVD and PDHD light
   archives (labels reported here).
3. **Merge-ON light rerun** under NEW tag suffix `_tmerge` (never overwriting
   `_keep`): 18-evt scan set (298567 + step 14) + crossers 298609/298651.
   Checks: B/C pairs merged at the fast-member times, A unchanged; per-event
   merge census (Δt, PE ratio, wide-lit fraction of every merge) to confirm no
   genuine pile-up absorbed.
4. **Merge-aware scan remap**: extend `ql_display/remap_scan_state.py` /
   `ql_agree_score.py` joins with an old→new flash map (an old flash maps to
   the merged flash that absorbed it, not just nearest-time within 0.5 µs) and
   a geometric cluster fallback (y/z point overlap) for when velocity changes
   shift cluster composition. QL rerun at the *current* op point with
   `PDVD_LIGHT_SUFFIX=_tmerge` isolates the flash change: agreement vs mapped
   truth expected ≈ baseline or better.
5. **Offset-removal retune sweep** at `PDVD_QL_EXTRA_OFFSET_US=0`: re-derive v
   from the crossers, then a small grid (v × anode margin {1,2,3 cm} × cathode
   ext1 {1.2, 2.0 cm}); objectives = crosser closure at both planes
   (`aca_positions.py`), mapped-scan agreement, containment-flag sanity.
6. **Validation + conditional adoption.** Acceptance bar (owner-set):
   mapped-scan agreement ≥ current production baseline AND crosser closure at
   anode and cathode. Met ⇒ flip runner defaults
   (`PDVD_FLASH_TAIL_MERGE`→1, `PDVD_QL_EXTRA_OFFSET_US`→0, new
   margin/ext1/velocity) in one commit; C++/cfg defaults stay OFF/legacy.
   Not met ⇒ stop and report; defaults untouched.

Each step lands as its own commit (+push). New work/label tags only; existing
`_keep` light dirs, `ql_labels/`, `decisions-*`, `ql_scores/` are never
touched.

## Step log

### Step 2 (2026-07-22): `flash_tail_merge` knob landed — gates PASS

Toolkit: `flash/{src,inc}/OpFlashFinder.*` new keys `flash_tail_merge`
(false) / `tail_window_us` (3.0) / `tail_min_width_us` (1.0) / `tail_pe_frac`
(0.7) / `tail_pe_ratio` (1.0); `refine_flashes` runs when
`flash_refine || flash_tail_merge`, criteria OR-ed, tail-only merges restore
the seed time; new `flash/test/doctest_opflashfinder.cxx` (6 cases / 40+
assertions incl. PDHD-legacy-unchanged and not-merged pile-up variants);
PDVD `flash.jsonnet` `opflash_finder()` key-suppressed args. Runner:
`wct-light-reco.jsonnet` args + `run_light_evt.sh` env `PDVD_FLASH_TAIL_MERGE`
(default 0) with `PDVD_TAIL_{WINDOW_US,MIN_WIDTH_US,PE_FRAC}` overrides.

Verification (all after `wcbuild` + freshness proof on
`local/lib/libWireCellFlash.so`):

- `./build/flash/wcdoctest-flash`: 12/12 cases, 82 assertions PASS.
- Compiled-config proof: HEAD-vs-new `wct-light-reco.jsonnet` knob-off
  compile byte-identical (diff empty); knob-on adds exactly the five tail
  keys.
- Knob-off A/B, PDVD (`flash_refine:false` path): tag
  `work/039252_light298609_abtailoff` vs production
  `039252_light298609_keep` — `hash_archive.py` member hash
  `36b544b6…` (11 members) IDENTICAL; compiled configs identical modulo tag.
- Knob-off A/B, PDHD (`flash_refine:true, subset_merge:true` legacy path):
  tag `pdhd/work/029107_allpd1007_abtailoff` vs `029107_allpd1007` —
  member hash `6cb7bd09…` (7 members) IDENTICAL.

Knob-on smoke (`work/039252_light298609_tmerge`, evt 298609): 431 → 433
flashes.

- 3 tail merges: track B pair 5399.991 (24384) + 5401.156 (19942) → ONE
  flash 5399.991 / 44326 PE (seed time kept); 2602.180 absorbed
  2603.426/2422; 6659.292 absorbed 6662.108/1918. Track A (3519.753 /
  22117 PE) untouched.
- 5 "new" flashes (10–14 PE) are RESCUED dim flashes: each was split by the
  same defect into two sub-10 PE fragments that both failed the
  `min_total_pe=10` quality cut in the legacy run (their hits all carry
  `flash_id=-1` there); the merge runs before the quality cut, reunites
  fragment + wide (1.0–1.2 µs) tail hit on the seed's own lit channel, and
  the combined flash passes. Physically correct recovery of dropped light;
  per-hit provenance verified for all five.

### Step 3 (2026-07-22): merge-ON light tags `_tmerge` + census — CLEAN

All 18 scan-set events (298567..298805 step 14, includes crossers
298609/298651) rerun with `PDVD_FLASH_TAIL_MERGE=1` into
`work/039252_light<evt>_tmerge/` (production `_keep` untouched). Census
(`scripts/tmerge_census.py`, compares `_keep` vs `_tmerge` per event):

- **86 merges, 72 rescues, ~1% of ~400 flashes/event touched; zero
  gate-inconsistent merges** (every merge has 0 < Δt ≤ 3 µs, PE ratio ≤ 1,
  wide+seed-lit PE fraction ≥ 0.7 — min observed 0.735).
- Track B: 5401.156/19942 → seed 5399.991, merged 44326 PE. Track C:
  2801.947/7382 → seed 2800.636 (Δt 1.311 µs, ratio 0.53, wide-lit 0.802) —
  the doc-23 §7d pairs are gone. Track A untouched.
- The 72 rescues are the same defect at small scale: fragment pairs that
  BOTH fell below `min_total_pe=10` in `_keep` (hits all `flash_id=-1`
  there) reunite pre-quality-cut and survive as one 10–15 PE flash —
  previously dropped light recovered.
- 2 "anomalies" inspected by hand: benign merge+rescue cascades (a seed
  absorbing both a surviving small flash and sub-cut fragments, mixed
  `[-1, id]` hit provenance). No genuine pile-up absorbed anywhere: narrow
  second onsets and unlit-channel flashes all left alone.

### Step 4 (2026-07-22): merge-aware scan remap — flash change is
### agreement-neutral

Repro:
```
./scripts/stage_ql_tag.sh 39252 <idx> tm0k   # + tm0, idx 0..17
PDVD_MAX_JOBS=6 PDVD_LIGHT_SUFFIX=_keep   ./run_clus_evt.sh -calib -s tm0k 39252 all
PDVD_MAX_JOBS=6 PDVD_LIGHT_SUFFIX=_tmerge ./run_clus_evt.sh -calib -s tm0  39252 all
python3 ql_display/tmerge_time_map.py --old-tag tm0k --new-tag tm0 --out work/ql_scores/tm0/time_map.json
python3 ql_display/ql_agree_score.py --tag tm0k
python3 ql_display/ql_agree_score.py --tag tm0 --truth-time-map work/ql_scores/tm0/time_map.json
```

Tooling: new `ql_display/tmerge_time_map.py` (per-event old→new flash-time
pairs: a gone flash maps to the nearest earlier flash surviving in BOTH
dumps within 3.5 µs — 59 pairs across the 18 events, 0 unmapped; 86
light-level merges → 59 because QL flash admission never admitted the small
ones); `ql_agree_score.py --truth-time-map` and `remap_scan_state.py
--time-map` (both default-off, byte-identical behavior without the flag)
translate truth/pick times recorded at merged-away members to the seed time,
with reported positive-wins collision resolution.

Scores (objective tiers, long tracks, tol 0.5 µs; both arms at TODAY's
production runner defaults, only the light tag differs):

| arm | agree | phantom | agree% | missed | missed% | unknown |
|---|---|---|---|---|---|---|
| `tm0k` (_keep light) | 764 | 118 | 86.6% | 78 | 9.3% | 180 |
| `tm0` (_tmerge light) | 763 | 118 | 86.6% | 79 | 9.4% | 182 |

`tm0k` reproduces the documented ac3 op point exactly (764/118/78) —
baseline confirmed. The merge arm moves 16 truth times (2 negative verdicts
dropped by positive-wins at evt298749 t=60.65 — a split pair whose two
scanner verdicts reunite as agrees). Redistribution detail: gains
298595 (+1 agree/−1 missed), 298749 (+2/−2), phantoms −1 at
298567/298609/298665; losses are single marginal pairs at
298735/298763/298777/298609/298805 whose LASSO pick shifted when the merged
flashes changed the candidate set. Net −1 agree / +1 missed / +0 phantom out
of 842 positives: **the tail merge is agreement-neutral at the current op
point** — the doc-23 pathology fix costs nothing on the scan set.

### Step 5 (2026-07-22): offset-0 retune sweep — best point rtp1

Repro:
```
python3 docs/qlmatch/scripts/retune_positions.py            # crosser closure vs v
# per point: stage_ql_tag.sh ... ; then e.g. (best point rtp1)
PDVD_MAX_JOBS=6 PDVD_LIGHT_SUFFIX=_tmerge PDVD_QL_EXTRA_OFFSET_US=0 \
  PDVD_DRIFT_SPEED_BOT_MMUS=1.4794 PDVD_DRIFT_SPEED_TOP_MMUS=1.4794 \
  PDVD_QL_ANODE_MARGIN_CM=1.0 ./run_clus_evt.sh -calib -s rtp1 39252 all
python3 ql_display/ql_agree_score.py --tag rtp1 \
  --truth-time-map work/ql_scores/tm0/time_map.json \
  --truth-time-shift -13.507 --truth-uid-map-tag tm0k
```

Scoring machinery grew two more default-off options (both needed to score an
offset-0 / velocity-changed run against the frozen truth): `ql_agree_score.py
--truth-time-shift` (uniform frame shift; the truth was recorded at
`PDVD_QL_EXTRA_OFFSET_US=13.507`, the sweep dumps sit at 0 ⇒ −13.507) and
`--truth-uid-map-tag` (geometric old→new cluster uid map by 3 cm y/z cell
overlap; velocity changes renumber cluster idents wholesale — without it the
sweep loses 164–349 of 842 truth positives to `cluster-missing`; with it, 0
lost at every point). Reference re-scored under the identical machinery:
`tm0` (offset 13.507, v 1.48073, cuts 2.0/2.0) = **752 agree (86.6%) / 116
phantom / 91 missed (10.8%)**.

**Crosser closure at offset 0** (`retune_positions.py`, W-erf endpoints of
the doc-23 A-C-A tracks, no pull): the anode side is v-insensitive — all six
ends land at u −0.39..−1.36, i.e. *between the shield and the W plane*
(physical; with the pull they sat 1.5 cm beyond the W plane) — so the anode
floor can tighten from −4 to −3 (margin 2.0→1.0) with ≥1.6 cm slack. The
cathode side sets the v band: v=1.48073 needs cathode_ext1 ≥ 2.29 (track A
bot escapes the current 2.0 ceiling), v=1.4794 needs ≥ 1.98 (just fits 2.0),
v=1.47 pulls track A 3.1 cm short of the cathode it physically touches
(vetoed — its good-looking score below is a containment artifact of
over-shrinking, not physics).

Sweep (all offset 0, `_tmerge` light, scored vs mapped truth):

| tag | v (mm/µs) | margin | ext1 | agree | agree% | phantom | missed | missed% |
|---|---|---|---|---|---|---|---|---|
| rtv0 | 1.48073 | 2.0 | 2.0 | 713 | 88.0% | 97 | 130 | 15.4% |
| rtv1 | 1.4794 | 2.0 | 2.0 | 717 | 88.2% | 96 | 126 | 14.9% |
| rtv2 | 1.4764 | 2.0 | 2.0 | 708 | 87.4% | 102 | 134 | 15.9% |
| rtv3 | 1.47 (vetoed) | 2.0 | 2.0 | 731 | 88.2% | 98 | 111 | 13.2% |
| **rtp1** | **1.4794** | **1.0** | **2.0** | **718** | **88.2%** | **96** | **125** | **14.8%** |
| rtp2 | 1.4794 | 2.0 | 1.5 | 710 | 88.1% | 96 | 133 | 15.8% |
| rtp3 | 1.4794 | 1.0 | 1.5 | 711 | 88.2% | 95 | 132 | 15.7% |
| rtp4 | 1.4764 | 2.0 | 1.5 | 709 | 87.3% | 103 | 133 | 15.8% |

Reading: anode margin 1.0 is free; REDUCING cathode_ext1 below 2.0 costs 7
agrees (it cuts real cathode-touching matches — matches the ladder's
"A bot needs ≥ 1.98"); the in-band velocity optimum is 1.4794 (the second
crosser's own W-decon measurement). **Best physical point: rtp1 = offset 0,
v 1.4794, anode margin 1.0, cathode ext1 2.0.**

### Step 6 (2026-07-22): acceptance bar NOT met — no defaults flipped

rtp1 vs the identically-scored baseline: agree% 88.2 vs 86.6 (better),
phantom 96 vs 116 (better), but **agree 718 vs 752 and missed 125 vs 91
(10.8→14.8%) — coverage regresses**, failing the owner's historical
adoption rule (flip only if a metric improves and none regresses). Per the
campaign plan, production runner defaults are left UNTOUCHED
(`PDVD_QL_EXTRA_OFFSET_US=13.507`, v 1.48073, margin 2.0, ext1 2.0,
`PDVD_FLASH_TAIL_MERGE=0`).

Where the 34 net losses live (rtv0 shows ~all of the +39 missed appear from
the offset removal ALONE, before any retune): of 60 newly-missed pairs at
rtv1, 44 still have a CONTAINED candidate bundle at the truth flash that the
matcher no longer auto-selects, 16 lose the bundle entirely — i.e. the
blocker is not containment but the flag windows and quality gates
(`at_cathode`/boundary flags, prefilter, high-consistent ladder, rescue
gates) that docs 18/19/23 tuned AT the pulled position; moving every track
2 cm re-lands them on the wrong side of those tuned thresholds.

**Owner decision points (nothing changed pending your call):**
1. *Adopt the tail merge alone* (`PDVD_FLASH_TAIL_MERGE=1`, offset kept):
   step 4 showed it agreement-neutral (763/118/79 vs 764/118/78) while
   fixing the doc-23 split-flash pathology and recovering dropped light —
   separable, low-risk adoption.
2. *Full offset-0 adoption* (physics-correct charge placement, doc 23 §7)
   needs a flag-window recalibration campaign at the new frame (re-derive
   the doc-18/19 operating points with offset 0) — rtp1 is the starting
   point; expected to recover most of the 44 contained-but-unselected
   losses since their bundles survive all containment gates.
3. *Accept rtp1 as-is*, trading 34 agrees / +34 missed for physical
   positions, −20 phantoms and +1.6% agreement rate.

## Files

- `scripts/aca_flash_split.py`, `scripts/aca_flash_wf.py`,
  `scripts/aca_positions.py` — doc 23 evidence tools this campaign builds on.
- (step 2+) toolkit `flash/src/OpFlashFinder.cxx`,
  `flash/inc/WireCellFlash/OpFlashFinder.h`,
  `flash/test/doctest_opflashfinder.cxx`,
  `cfg/pgrapher/experiment/protodunevd/flash.jsonnet`;
  `pdvd/wct-light-reco.jsonnet`, `pdvd/run_light_evt.sh`.
- (step 4+) `ql_display/tmerge_time_map.py` (new), `ql_display/
  ql_agree_score.py` (`--truth-time-map`, `--truth-time-shift`,
  `--truth-uid-map-tag`, all default-off), `ql_display/remap_scan_state.py`
  (`--time-map`); scores under `work/ql_scores/{tm0,tm0k,rtv0..3,rtp1..4}/`.
- (step 5) `scripts/retune_positions.py` (crosser closure vs v at offset 0);
  QL sweep tags `work/039252_<idx>_{tm0k,tm0,rtv0-3,rtp1-4}/` (staged from
  `_keep` clustering, matching-only).
