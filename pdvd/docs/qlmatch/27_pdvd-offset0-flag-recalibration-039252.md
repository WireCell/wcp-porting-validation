# 27 — Offset-0 flag-window recalibration campaign (run 039252)

Status: **IN PROGRESS** (2026-07-22). Owner-directed follow-up to doc 26:
recover the ~34 net scan-agreed pairs lost when the 13.507 µs pull is removed
(doc 26 step 6, decision point 2), by recalibrating the position-dependent
flag windows and quality gates at the physical (offset-0) frame.

Frame under recalibration (= doc 26 rtp1, all sweeps below inherit it):
`PDVD_QL_EXTRA_OFFSET_US=0`, `PDVD_DRIFT_SPEED_{BOT,TOP}_MMUS=1.4794`,
`PDVD_QL_ANODE_MARGIN_CM=1.0`, `PDVD_QL_CATHODE_EXT1_CM=2.0`,
`PDVD_LIGHT_SUFFIX=_tmerge` (tail-merged light). Truth scoring:
`ql_agree_score.py --truth-time-map work/ql_scores/tm0/time_map.json
--truth-time-shift -13.507 --truth-uid-map-tag tm0k`.

Baselines (identical scoring machinery):
| point | agree | agree% | phantom | missed | missed% |
|---|---|---|---|---|---|
| tm0 (offset 13.507, production) | 752 | 86.6% | 116 | 91 | 10.8% |
| rtp1 (offset 0, doc 26 best) | 718 | 88.2% | 96 | 125 | 14.8% |

Target: close the missed gap (125 → ~91) without phantom inflation; adoption
per the doc-23 rule (a metric improves, none regresses) vs the tm0 row.

## Phase 0 — forensics: where the 59 newly-missed pairs go

Of the 59 rtp1-vs-tm0 newly-missed: **41 mis-picks** (cluster matched a
DIFFERENT flash), **16 no-bundle-at-truth-flash**, **2 unmatched with a good
bundle present**. Truth-bundle frame comparison (29 present in both frames):
ks/χ² essentially UNCHANGED (e.g. 0.33→0.33, 174/10→177/10) while LASSO
`strength` collapses 0.9x→0.00 — the bundle's quality is intact; the PICK
machinery flipped. Dominant discrete flips on the truth bundle:
`-xtpc_pin` ×7, `-consistent` ×6 (several at ks ≈ 0.10 = exactly the
hc_good tier edge), `±at_x_boundary` ×6; 9 crosser-flagged bundles
(at_cathode + xtpc_* in the old frame) are GONE from candidacy entirely.

Mechanism map (from `QLMatching.cxx` / `TimingTPCBundle.cxx` reading):
- `cathode_in = u_cathode + cathode_ext1` is simultaneously the containment
  ceiling, the `at_cathode`/`at_x_boundary` flag ceiling (`:4669`), and the
  xtpc cathode-rescue reference (`:4702-4730`). Real late-charge in-cathode
  tails (doc 16) sit 2 cm deeper at offset 0, crossing it.
- Losing `at_x_boundary` breaks the xtpc candidate admission (`:4123`), so
  the greedy pin (`:4287`) never forms; a pinned cluster otherwise has all
  rivals culled (`cull_inconsistent:2137`) and is strength-cutoff-exempt
  (`:1981`) — pin loss re-opens the LASSO competition and the truth bundle
  gets zeroed with unchanged metrics. The `xtpc_cathode_tol` rescue
  (PDVD_QL_XTPC_CATHODE_TOL_CM=10, ks ≤ 0.32) is the designed re-admission
  path, tuned in the pulled frame.
- `consistent` comes from the highconsist ladder (B1-B4,
  `TimingTPCBundle.cxx:292-311`); B2 `hc_good_ks=0.10` and the B4 miss
  branch (requires a boundary flag) both sit exactly where the frame shift
  lands marginal bundles.

## Phase 1 — single-mechanism diagnostics (this commit)

Four one-knob variants on top of the rtp1 frame, each isolating one
mechanism (18 evts each, tags rc1..rc4):

| tag | change vs rtp1 | probes |
|---|---|---|
| rc1 | `PDVD_QL_XTPC_CATHODE_TOL_CM=14` | xtpc cathode re-admission depth |
| rc2 | `PDVD_QL_CATHODE_EXT1_CM=2.5` | ceiling itself (diagnostic — quantifies the recoverable pool; adoption would prefer mechanism-specific windows per the owner's maintain-or-reduce guidance) |
| rc3 | `PDVD_QL_HC_GOOD_KS=0.12 PDVD_QL_HC_MISS_KS=0.10` | highconsist tier edges |
| rc4 | `PDVD_QL_POSTCULL=0` | post-fit cull share (diagnostic only) |

Results (recovery counter `scripts/rc_recovery.py`, target = the 58
uid-mapped pairs missed at rtp1 but not at tm0):

| tag | agree | agree% | phantom | missed | recovered | new-missed | new-phantom |
|---|---|---|---|---|---|---|---|
| rc1 xtpc tol 14 | 716 | 88.3% | 95 | 127 | 0 | 2 | 0 |
| rc2 ext1 2.5 | 716 | 88.0% | 98 | 127 | 0 | 2 | 2 |
| rc3 hc tiers loose | 686 | 88.7% | 87 | 157 | 6 | 38 | 9 |
| rc4 postcull off | 718 | 82.5% | 152 | 125 | 1 | 7 | 56 |

**All four mechanisms are nulls** (rc3/rc4 net-destructive). The blockers sit
in bundle FORMATION, not in the windows swept.

## Phase 2 — log forensics: the sc1 light gate is the root cause

Case study evt298581 uid 182 (GONE class), traced through the QL debug logs
of both frames (`work/039252_1_{tm0,rtp1}/wct_clus_039252_1.log`):

1. The truth flash (dump gid 15, internal flash 37, raw t=797.037 µs)
   survives with identical PE; cluster 182 is an xtpc side-0 candidate at it
   in BOTH frames.
2. The pair test vs its top-half partner is IDENTICAL in both frames:
   `d=3.13 cm, sc1=true, pin=true` (`QLXTPC pair 0/182 1/361`), and the
   greedy pin binds the pair to the truth flash in both
   (`QLXTPCPIN pair 0/182 (flash 37) ... d=3.13cm`).
3. tm0: the bundle is genuinely contained (cathode end inside the
   `at_cathode` window) → never provisional → `purge_unconfirmed_
   cathode_rescue` doesn't apply → JOINT-PIN survives, `cull_inconsistent`
   drops all rivals, truth wins (strength 0.96).
4. rtp1: the same cathode end sits ~1.7 cm deeper → past `cathode_in` → the
   bundle forms via the provisional rescue. The purge keeps a provisional
   only if it carries `flag_xtpc_scenario1` (`QLMatching.cxx:2088`) — and
   that flag is denied by the **scenario-1 light gate**
   (`xtpc_sc1_light_gate`: ks ≤ 0.3, c2n ≤ 50; production ON since doc 19
   phase 3). The genuine crosser half has ks = 0.39 (truncated half-pattern,
   unchanged between frames) → flag denied → `cathode-rescue DROP ...
   flash 37 (no cross-volume confirmation)` **despite the pin** → the pin
   evaporates with the bundle → rivals return → LASSO picks a ks-0.16
   wrong flash.

Confirmed by elimination: rc5/rc6 (purge's own `cathode_ks_max` 0.6/off)
change nothing — the DROP persists with the gate value visibly active in the
log — because the keep-branch requires the scenario-1 flag *first*; the ks
ceiling is downstream of it.

Why it hid at the pulled frame: the sc1 light gate was tuned (doc 19) as a
junk-steering quality control for bundles that were ALREADY contained — the
purge only ever saw genuinely-overshooting junk. At offset 0 the whole
formerly-contained cathode-toucher population routes through the provisional
path and meets a gate never meant for it.

## Phase 3 — sc1 light-gate recalibration

| tag | change vs rtp1 | agree | phantom | missed | recovered | new-phantom |
|---|---|---|---|---|---|---|
| rc7 `SC1_KS=0.45` | ks ceiling up | 719 | 109 | 124 | 0 | 14 |
| rc8 gate OFF | diagnostic | 718 | 130 | 125 | 1 | 34 |

**Both null AND phantom-inflating** — the gate does real protective work, and
relaxing it also grants sc1 to rival pairings (uid 182: with ks 0.45 the
d=6.08 rival at another coincident flash gets sc1 + is contained, while the
pinned truth bundle is STILL purged — the partner half's ks 0.58 blocks its
sc1 flag; `DROP … flash 37` persists with the gate visibly at 0.45).

The pin is the discriminator the purge ignores: `QLXTPCPIN` binds the pair to
the truth flash by best combined light in BOTH frames, but
`purge_unconfirmed_cathode_rescue` (:2085-2102) consults only
`flag_xtpc_scenario1`. The joint-pin post-dates the purge's doc-19 design.

## Phase 4 — `xtpc_pin_confirms_rescue` knob (C++, default OFF)

A greedy-pinned bundle (direction-confirmed collinear pair, d < dmax, chosen
by min ks-sum over coincident flashes) counts as cross-volume confirmation in
the purge, exempt from the sc1 light gate and the cathode ks ceiling. New key
`xtpc_pin_confirms_rescue` (C++ default false) in `QLMatching`, key-suppressed
jsonnet arg, runner env `PDVD_QL_PIN_CONFIRMS_RESCUE` (default 0).

Gates: `wcdoctest-match` 4/4; knob-off byte-identical vs the pre-change
binary at the rtp1 frame (tag `039252_1_rcoff` vs `039252_1_rtp1`:
calib-evt298581.json diff EMPTY, mabc-all-apa.zip member hashes IDENTICAL);
compiled config carries the key only when on (tag rc9 work dir).

**rc9 = rtp1 frame + pin-confirms-rescue ON:**

| tag | agree | agree% | phantom | missed | recovered | new-missed | new-phantom |
|---|---|---|---|---|---|---|---|
| rc9 | 723 | 87.8% | 100 | 120 | **5** | 0 | 4 |

First real recovery; the traced case lands exactly (uid 182 auto → truth
flash −1710.7, ks 0.39, pinned). The 4 new phantoms are all pin-confirmed
rescued bundles at ks 0.38–0.55 (scanner-rejected) — the price of the class,
possibly trimmable with a pin-specific ks ceiling later. Remaining 54 lost:
42 mis-picks / 10 no-bundle / 2 unmatched — dominated by SINGLE-VOLUME
cathode-touchers, which have no cross-volume partner and hence NO rescue
path at all: pushed past the ceiling they simply drop from candidacy.

## Phase 5 — solo (single-volume) light-quality cathode rescue

rc10 (= rc9 + ceiling 3.5, diagnostic) recovered 10 but traded ~1:1 (4
new-missed, 7 new-phantom). Its genuine recoveries separate cleanly from its
junk by light quality: recovered ks 0.03–0.29 (c2n 0.3–38) vs new phantoms
c2n 48.7/376.5 and ks 0.41. Hence the principled knob instead of the blunt
ceiling: **`cathode_rescue_solo_ks/_c2n`** (C++ defaults 0/0 = off) — a
third keep-branch in the purge for provisional bundles with NO cross-volume
partner, kept on their own light (ks ≤ solo_ks AND chi2/ndf ≤ solo_c2n).
Runner envs `PDVD_QL_CATHODE_SOLO_KS/_C2N`. Gates: `wcdoctest-match` 4/4;
knob-off byte-identical vs the rc9-era binary (tag `039252_1_rcoff2` vs
`039252_1_rc9`: calib diff empty, mabc hashes identical).

## Phase 6 — sc1 rival hijack and the eviction ks margin

Remaining-class forensics (3 pure mis-picks, truth bundle metrics unchanged,
strength 0.9→0.00): the WRONG pick **gained `xtpc_scenario1`** at the new
frame (spurious cathode pairing within dmax 25 cm) and `cull_inconsistent`'s
sc1-priority branch then evicted the scan-endorsed high-consistent truth
(truth ks 0.03/0.09 evicted by sc1 rivals ks 0.12/0.24; third case = a
ladder-edge `consistent` flip at ks 0.10). The sc1 priority was the designed
"steering fix" (a tight crosser overrides an accidental high-consistent
match) — its assumption that sc1 = genuine crosser breaks at the offset-0
frame. With the pin now honored in the purge, two remedies tested:
sc1-gate tightening (rc12/rc13: recovers 15/12 but collateral 13/6 — the
0.10–0.3 sc1 truths lose protection) and the surgical
**`sc1_evict_ks_margin`** knob (C++ default −1 = off): a high-consistent
rival whose ks beats the cluster's best sc1 ks by more than the margin is
spared the eviction and competes in the LASSO (scenario-2 precedent).
Gates: `wcdoctest-match` 4/4; knob-off byte-identical vs the rc11-era binary
(tag `039252_1_rcoff3` vs `039252_1_rc11`: calib diff empty, mabc hashes
identical).

## Phase 7 — combined results and the campaign op point

All at the doc-26 rtp1 frame, scored vs mapped truth (58 target pairs):

| tag | knobs | agree | agree% | phantom | missed | recov | new-miss | new-phm |
|---|---|---|---|---|---|---|---|---|
| rtp1 | (doc 26 best) | 718 | 88.2% | 96 | 125 | — | — | — |
| rc9 | pin | 723 | 87.8% | 100 | 120 | 5 | 0 | 4 |
| rc10 | pin + ceiling 3.5 | 724 | 87.8% | 101 | 119 | 10 | 4 | 7 |
| rc11 | pin + solo .30/40 | 724 | 88.2% | 97 | 119 | 7 | 1 | 4 |
| rc12 | rc11 + sc1 ks .10 | 722 | 88.5% | 94 | 121 | 15 | 13 | 4 |
| rc13 | rc11 + sc1 ks .20 | 726 | 88.6% | 93 | 117 | 12 | 6 | 4 |
| **rc14** | **rc11 + margin .05** | **728** | **88.6%** | **94** | **115** | **11** | **3** | **4** |
| rc15 | rc14 + sc1 ks .20 | 725 | 88.6% | 93 | 118 | 13 | 8 | 4 |

**Campaign op point: rc14** = rtp1 frame + `PDVD_QL_PIN_CONFIRMS_RESCUE=1`
`PDVD_QL_CATHODE_SOLO_KS=0.30` `PDVD_QL_CATHODE_SOLO_C2N=40`
`PDVD_QL_SC1_EVICT_KS_MARGIN=0.05`.

| metric | tm0 (production) | rtp1 | rc14 | rc14 vs production |
|---|---|---|---|---|
| agree | 752 (86.6%) | 718 | 728 (88.6%) | −24 (rate +2.0) |
| phantom | 116 | 96 | 94 | **−22** |
| missed | 91 (10.8%) | 125 | 115 (13.6%) | +24 |

The recalibration recovers 10 of the 34 net agrees the offset removal cost
(and 11/58 of the target-pair list, with the rc12 diagnostic showing ~15
reachable at higher collateral). The remaining ~24 net losses are
heterogeneous single-pair LASSO re-equilibrations (candidate-set ripples
with no flag or metric change on the truth bundle) — recovering them means
re-deriving the doc-19 LASSO/ladder economy (strength cutoff, boundary
weights, hc tiers) at the new frame, a full retuning campaign of its own;
the rc3/rc4/rc7/rc8/rc12 diagnostics all show those knobs re-trade along a
phantom↔missed frontier rather than dominating.

**Adoption status: owner-gated, nothing flipped.** Against the no-regression
rule rc14 still fails on agree/missed (−24/+24) while winning phantoms (−22)
and rate (+2.0). Decision points: (1) adopt the tail merge alone (doc 26,
agreement-neutral) and stay at offset 13.507; (2) adopt rc14 as the new
production frame, taking the coverage cost for physical positions, −22
phantoms and +2.0% rate; (3) commission the LASSO-economy retune on top of
rc14. All knobs are default-OFF in the toolkit; the rc14 op point lives
purely in runner envs.

## Files

- toolkit `match/{src,inc}/QLMatching.*`: `xtpc_pin_confirms_rescue`,
  `cathode_rescue_solo_ks/_c2n`, `sc1_evict_ks_margin` (all default OFF);
  `cfg/pgrapher/experiment/protodunevd/qlmatching.jsonnet` key-suppressed
  args.
- `pdvd/wct-clustering.jsonnet` + `pdvd/run_clus_evt.sh`: env threading
  (`PDVD_QL_PIN_CONFIRMS_RESCUE`, `PDVD_QL_CATHODE_SOLO_KS/_C2N`,
  `PDVD_QL_SC1_EVICT_KS_MARGIN`).
- `scripts/rc_recovery.py` — target-pair recovery counter; sweep tags
  `work/039252_<idx>_{rc1..rc15,rcoff,rcoff2,rcoff3}` (records, keep).
