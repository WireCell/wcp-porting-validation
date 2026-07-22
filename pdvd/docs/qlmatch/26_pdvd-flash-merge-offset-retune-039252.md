# 26 — Flash tail-merge + QL offset removal / cut+velocity retune campaign (run 039252)

Status: **IN PROGRESS** (2026-07-22). Owner-directed follow-up to doc 23
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

## Files

- `scripts/aca_flash_split.py`, `scripts/aca_flash_wf.py`,
  `scripts/aca_positions.py` — doc 23 evidence tools this campaign builds on.
- (step 2+) toolkit `flash/src/OpFlashFinder.cxx`,
  `flash/inc/WireCellFlash/OpFlashFinder.h`,
  `flash/test/doctest_opflashfinder.cxx`,
  `cfg/pgrapher/experiment/protodunevd/flash.jsonnet`;
  `pdvd/wct-light-reco.jsonnet`, `pdvd/run_light_evt.sh`.
- (step 4+) `ql_display/remap_scan_state.py` extension, scoring outputs under
  fresh `work/ql_scores/<tag>/`.
