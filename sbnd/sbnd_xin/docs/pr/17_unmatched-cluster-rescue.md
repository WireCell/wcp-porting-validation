# Doc pr/17 — Unmatched-cluster rescue: adopt a flashless cathode-crossing continuation into the beam bundle

**Status: SHIPPED, SBND DEFAULT ON (`rescue_unmatched` pass of
`ClusteringCathodeBundleRescue`; C++ default stays OFF = byte-identical
everywhere else).  Demonstrated on the founding event (§6); validated on
1000 mcp1k + 48 nueCC48 events (§7: fires 1/1000 — the founding event — and
0/48); default flip proofs §10.  Escape: `SBND_RESCUE_UNMATCHED=0` /
`cathode_rescue_unmatched=false` restores the pre-pr/17 path byte-identically.**

Owner request (2026-08-01, follow-up to docs pr/14 + pr/15): "can it also
consider the non-matched clusters in addition to the nearby bundles? For
example run 18255 evt 56463 — can recover a neutrino out of it."

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# --- the founding case: run 18255 evt 56463 (mcp1k entry 599), vertex veto ON ---
# candidate census over the production (veto-ON) arms:
python3 scripts/analysis/ql/unmatched_census.py work-mcp1kall-vveto1k /home/xqian/tmp/u17_census.tsv 6

# knob-off identity check across the binary change (3 events, fresh tag):
TAG=u17offchk ENTRIES="42 472 599" ./run_full1k_nusel.sh 1000 3
python3 scripts/analysis/ql/ql_arm_compare.py work-mcp1kall-vveto1k work-mcp1kall-u17offchk 292533 395148 56463

# demo, knob ON (56463):
TAG=u17on56463 ENTRIES="599" SBND_RESCUE_UNMATCHED=1 ./run_full1k_nusel.sh 1000 1
PR_JOBS=1 ./run_pr_chain_batch.sh work-mcp1kall-u17on56463 work-mcp1kall-u17on56463pr data 56463
# baseline PR for the same event (veto-ON arm, no unmatched rescue):
PR_JOBS=1 ./run_pr_chain_batch.sh work-mcp1kall-vveto1k work-mcp1kall-u17basepr56463 data 56463

# --- no-regression sweep: full 1000-event mcp1k + all 48 nueCC48, ON vs veto-ON baseline ---
# (first attempt, tag u17on1k, is VOID: a concurrent-session `wcb install`
#  swapped libWireCellClus.so mid-sweep, 55ed8412 -> 0c7951fa, failing 16
#  entries with the M3 load race.  Retained but never compared.  The u17on1kb
#  re-sweep below printed the lib md5 before and after: 0c7951fa both ends.)
TAG=u17on1kb SBND_RESCUE_UNMATCHED=1 ./run_full1k_nusel.sh 1000 6
SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-2025fall-48evt-fsprod \
SBND_WORK_ROOT=$PWD/work-nuecc48-u17on SBND_SAVE_ASSOC=1 SBND_RESCUE_UNMATCHED=1 \
    ./run_nusel_evt.sh data -chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm \
    -main-pair-real -fvx 2.5 -fvy 3 -stm-fit -mip 56000 -unmerge-assoc all
python3 scripts/analysis/cathode/cbr_sweep_compare.py --off work-mcp1kall-vveto1k --on work-mcp1kall-u17on1kb \
    --out /home/xqian/tmp/cbr17/u17_sweep_mcp1k.tsv
# determinism: two more fresh ON runs of entry 599 (56463), 3-way QL-product hash
TAG=u17det1 SBND_RESCUE_UNMATCHED=1 ENTRIES=599 ./run_full1k_nusel.sh 1000 1
TAG=u17det2 SBND_RESCUE_UNMATCHED=1 ENTRIES=599 ./run_full1k_nusel.sh 1000 1
# attribution gate for the 50 stale-baseline mismatches (sec 7.1): same 50
# entries, knob-ON and knob-OFF at ONE binary, then pairwise compare
ENT=$(cut -d' ' -f1 /home/xqian/tmp/cbr17/attr_entries.txt | tr '\n' ' ')
TAG=u17attroff SBND_RESCUE_UNMATCHED=0 ENTRIES="$ENT" ./run_full1k_nusel.sh 1000 6
TAG=u17attron  SBND_RESCUE_UNMATCHED=1 ENTRIES="$ENT" ./run_full1k_nusel.sh 1000 6
python3 scripts/analysis/cathode/cbr_sweep_compare.py --off work-mcp1kall-u17attroff --on work-mcp1kall-u17attron \
    --out /home/xqian/tmp/cbr17/u17_sweep_attr.tsv
python3 scripts/analysis/cathode/cbr_sweep_compare.py --off work-nuecc48-vveto --on work-nuecc48-u17on \
    --out /home/xqian/tmp/cbr17/u17_sweep_nuecc.tsv
# ^ stale-baseline artifacts (sec 7.2); the attribution baseline is a fresh
#   knob-off run at the same binary family:
SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-2025fall-48evt-fsprod \
SBND_WORK_ROOT=$PWD/work-nuecc48-u17off SBND_SAVE_ASSOC=1 SBND_RESCUE_UNMATCHED=0 \
    ./run_nusel_evt.sh data -chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm \
    -main-pair-real -fvx 2.5 -fvy 3 -stm-fit -mip 56000 -unmerge-assoc all
python3 scripts/analysis/cathode/cbr_sweep_compare.py --off work-nuecc48-u17off --on work-nuecc48-u17on \
    --out /home/xqian/tmp/cbr17/u17_sweep_nuecc_cleanbase.tsv
# per-pair cut tracer (sec 7.4), env-gated, prints [cbrx] lines to stderr:
#   CATHODE_RESCUE_DEBUG=1 + any of the runs above (work-nuecc48-u17dbg,
#   work-mcp1kall-u17dbg56463)
```

## 1. Symptom

With the doc pr/15 vertex veto ON (SBND production default), run 18255 evt
56463's neutrino interaction is correctly rejoined into ONE 3529-pt, 259.5 cm
cluster — and the Q/L matching then leaves it with **no flash at all**
(`matched_flash_gid = −1`, sentinel `cluster_t0`).  Pre-veto, its two split
pieces were absorbed into two different cosmic bundles (pr/15 §2); rejoined as
one whole cluster, no wrong flash's light pattern claims it.  A flashless
cluster is invisible downstream: it belongs to no bundle, is never in the beam
window, is skipped by every tagger and by PR — and it even vanishes from the
post-QL Bee `clustering` layer, because its corrected coordinates were
materialized with the sentinel T0 (x_t0cor off by 1.5e6 m; only the raw
`img-global` layer shows it).  The pr/14 bundle rescue cannot touch it either:
its candidate loop requires `flash >= 0` on both sides
(`clustering_cathode_bundle_rescue.cxx:402`), and its direction rule needs a
far bundle that does not exist.

The observable beam-window "candidate" in the veto-ON baseline is the muon's
TPC0 continuation stub — PR puts the vertex on the stub's far end at
(−121.4, −59.0, 395.0) cm, 300+ cm from the true vertex, with
`nue_score = −15` (the failure sentinel) and `kine_reco_Enu = 512` MeV.

## 2. Root cause

Same upstream defect as pr/14 (SBND flash-reco absorbing window, pr/12 §6) in
a different failure shape: instead of the far half being matched to a *wrong*
flash (pr/14's case), the far half is matched to *no* flash.  Which shape an
event takes depends on whether a wrong-window flash's light pattern happens to
fit the charge — after the pr/15 veto changed the cluster composition of
56463, none did.

## 3. Fix — the `rescue_unmatched` pass

A second candidate pass inside `ClusteringCathodeBundleRescue` (same file,
same slot: QL all-APA pipeline after `cathode_connect`, before
`examine_bundles`), gated by a new `rescue_unmatched` knob (C++ default
false ⇒ pass absent ⇒ byte-identical):

1. K_beam as in the bundle pass: flash-matched, `cluster_t0` in the beam
   window, in analysis scope, length ≥ `min_length_short`.
2. K_far ranges over **flashless clusters** (`flash < 0` or `gid < 0`) with
   ≥ `unmatched_min_npts` points (default 200) and length ≥
   `unmatched_min_length` (default 30 cm) — floors that keep specks and
   debris out of the beam bundle.
3. The same duplicated `is_cathode_crossing_pair` geometry test, with the
   orphan evaluated in its **raw scope** (its x_t0cor is sentinel-corrupted,
   §1) under the destination-T0 hypothesis: raw x is the t0=0 frame, so the
   x-shift is `−dirx · t0_beam · v_drift` (sub-mm at beam times — but kept
   for uniformity with the bundle pass).
4. Direction is **forced into the beam bundle** — an unmatched cluster has no
   flash to offer.  The merge, flash-identity re-stamp, main-flag stamp and
   `add_corrected_points` under the beam T0 (which repairs the orphan points'
   sentinel x_t0cor) mirror the bundle pass; there is no source bundle to
   repair.  The pass runs to exhaustion AFTER the bundle pass, on the
   post-move state, one merge per round with full re-enumeration
   (ident-ordered, deterministic).
5. No scope-filter gate on the orphan (the in/out-of-volume partition needs a
   T0 it does not have); the tip-at-the-cathode geometry is the containment
   gate.  Unaccepted orphans are restored to the analysis scope at pass end.

Risk shape: this pass can only ADD charge to the beam bundle.  A wrongly
adopted cosmic half produces a joined cathode crosser that the TGM/STM taggers
evaluate downstream — the self-correcting pattern observed on pr/14's
288952/352365.

Config: `rescue_unmatched` arg on the common `cathode_bundle_rescue` factory
(`cfg/pgrapher/common/clus.jsonnet`, key-suppressed default false) →
`cathode_rescue_unmatched` on SBND `clus_all_apa`/`all_apa`
(`cfg/pgrapher/experiment/sbnd/clus.jsonnet`) → TLA `cathode_rescue_unmatched`
in `wct-clus-matching-perevt.jsonnet` (default false pending this validation).
Runner: `SBND_RESCUE_UNMATCHED=1` enables, `=0` forces off, unset inherits the
config default.  Needs `cathode_rescue` (the pass lives inside that
component).

## 4. Candidate population (why this is worth having)

Census over the production veto-ON arms (`scripts/analysis/ql/unmatched_census.py`: flashless
clusters ≥ 200 pts and ≥ 30 cm whose cathode-nearest tip is within 5 cm of
the cathode, in an event that has an in-beam bundle):

- mcp1k (1000 data events): **14 events** qualify — including 56463 (3529
  pts, 259.5 cm) and 169824 (2520 pts, 234.2 cm — a pr/14 hand-scan event).
- nueCC48: **1 event** — 444187, a 24422-pt, 515.7 cm flashless cluster.

This is the coarse pool: an actual firing additionally needs a beam-window
partner whose cathode tip matches, so the firing rate is lower (§7).

## 5. Verification — knob-off byte-identity

Binary lineage: pre-change `c4586ae4…` → post-change `55ed8412…`
(`local/lib/libWireCellClus.so`, install 15:42 > source 15:37, freshness
proof).  `./build/clus/wcdoctest-clus` 565/565.

- Compiled-config proofs (wcsonnet, HEAD overlay vs tree):
  - SBND QL job (`wct-clus-matching-perevt`, full TLA set): knob-off compile
    **byte-identical** to HEAD; knob-on compile carries
    `"rescue_unmatched" : true` exactly once.
  - SBND PR job (`wct-pr-perevt`, production pipeline TLAs): byte-identical.
  - uBooNE job (`qlport/uboone-mabc.jsonnet` via `compile_ub_cfg.sh`):
    byte-identical.
- Runtime knob-off identity across the binary change (fresh tag
  `work-mcp1kall-u17offchk`, production defaults incl. both prior rescues, vs
  the `work-mcp1kall-vveto1k` baseline): events 292533 (vveto firing),
  395148 (bundle-rescue firing), 56463 (both) — `mabc-all-apa.zip` + pctree
  member-content hashes **3/3 identical**, nusel TSVs identical, and the
  pre-existing `rescue round` log lines fire identically.

## 6. Demonstration — 56463 recovered end-to-end

Tag `work-mcp1kall-u17on56463` (+`pr`), `SBND_RESCUE_UNMATCHED=1`:

```
rescue round 0:            c14 (gid 6, 1.185 us, 199.2 cm) + c21 (gid 1000000, 5.746 us, 332.3 cm) -> gid 1000000   (pre-existing pass, unchanged)
unmatched rescue round 0:  c12 (gid 6, 1.185 us, 169.1 cm) + c19 (unmatched, 267.7 cm, 3529 pts) -> gid 6
```

The pairing is the one the geometry predicted: the orphan nu's muon-arm
cathode tip (1.83, 28.8, 319.2) cm sits 2.44 cm from the beam-bundle muon
continuation's tip (−0.27, 27.5, 318.9) cm, collinear.  Post-QL pctree ground
truth: the beam bundle main is now **5076 pts, 414.1 cm** (TPC0 stub + whole
rejoined nu).  Taggers evaluate it: TGM=false, STM=0, FC=false ⇒ beam-window
nu-candidate.  Full PR chain (`work-mcp1kall-u17on56463pr`):
`selected main cluster (t0 1.185 us, L 429.3 cm, 17 associated)`.

| | veto-ON baseline | + unmatched rescue |
|---|---|---|
| beam main | 1547 pts, 173 cm (muon TPC0 stub) | 5076 pts, 414 cm (whole interaction) |
| reco vertex (cm) | (−121.4, −59.0, 395.0) — stub far end | **(131.4, 166.3, 185.7)** — the pr/15 vertex V ≈ (132.7, 166.8, 188.6) |
| numu_score | 0.94 | **4.06** |
| nue_score | −15 (failure sentinel) | −4.3 (real evaluation) |
| kine_reco_Enu | 512 MeV | **1363 MeV** |
| cosmict_flag | 0 | 0 |

Known reporting artifact: the `nusel-evt56463.tsv` row for the beam bundle
prints `no-bundle` — the pr/14 §7.4 QL→PR ident-mapping artifact in
`nusel_extract.py` (first seen on 352365), NOT a chain failure; the pctree and
the tagger log above are the ground truth.  Fixing the extractor is still the
follow-up it was in pr/14.

## 7. No-regression sweep

Binary lineage during the sweeps (the concurrent session installed twice
mid-campaign): ON arms `0c7951fa…`, nueCC48 knob-off baseline `83747b9f…`.
Every arm below carries its own before/after `md5sum
local/lib/libWireCellClus.so` bracket in its launch log — no arm straddles an
install.

### 7.1 mcp1k, 1000 data events (`work-mcp1kall-u17on1kb`)

- 1000/1000 completed (entry 631/evt 280884 failed once with zero-byte
  outputs and no error line — the known transient install/load class; its
  single re-run matched the baseline's wall/RSS profile exactly).
- **The pass fired in exactly 1 of 1000 events: 56463, the founding case**
  (`unmatched rescue round 0: c12 (gid 6, t0 1.185 us, 169.1 cm) + c19
  (unmatched, 267.7 cm, 3529 pts) -> gid 6`).  A knob-ON event with zero
  adoptions logs nothing (the summary line is guarded by `nadopt > 0`), so
  the single summary line IS the census.
- The §4 census said 14 candidate events; 1 firing.  The census only
  screened the orphan (size, cathode-tip distance, a beam bundle existing
  somewhere); the pass additionally demands the full cathode-crossing pair
  geometry against a specific in-beam partner.  The selectivity is the
  point: nothing is adopted on tip proximity alone.
- Arm comparison vs `work-mcp1kall-vveto1k` (`u17_sweep_mcp1k.tsv`): 943
  IDENTICAL, 8 FIRED, 49 MISMATCH.  Of the FIRED rows, 6 are pr/14 pass-1
  bundle moves fired identically in both arms (hashes `= = =`: 59003,
  169824, 352365, 392200, 395148, 398690); 56463 is the pass-2 adoption
  (all archives differ, as intended); 288952 fired pass-1 but with DIFF
  hashes — it joins the mismatch class below.
- **Attribution gate for the 49+1 mismatches** (all with zero pass-2
  firings): the historical `vveto1k` baseline (14:30) predates the
  concurrent session's two same-day installs and their evolving
  uncommitted work, so ON-vs-vveto diffs are not attributable.  All 50
  events were re-run knob-ON and knob-OFF at ONE binary
  (`work-mcp1kall-u17attron` / `work-mcp1kall-u17attroff`, 50/50 ok each,
  `83747b9f…` md5 bracket unbroken across both arms):
  **50/50 hash-identical** (`u17_sweep_attr.tsv`; 288952's pass-1 move
  fires identically in both arms, `= = =`).  Every baseline mismatch is
  the concurrent session's tree drift; the knob's only effect in 1000
  events is 56463.

### 7.2 nueCC48, 48 nue MC events

First comparison, `work-nuecc48-u17on` vs the historical
`work-nuecc48-vveto` baseline, showed 6 mismatching events (10550, 30504,
42280, 234638, 271851, 444187) with **zero** pass-2 firings — and 10550 /
271851 are exactly the events reshaped by the concurrent session's pr/16
`min_length` round.  The historical baseline predates several of that
session's landed changes: stale baseline, not a knob effect.

Attribution re-run at one binary family, fresh knob-off baseline
`work-nuecc48-u17off` (`SBND_RESCUE_UNMATCHED=0`, bracket `83747b9f…` both
ends) vs `work-nuecc48-u17on`:

- **48/48 hash-identical** (`u17_sweep_nuecc_cleanbase.tsv`).  The one
  "FIRED" row, 437699, is the pr/14 **pass-1** bundle move — default-ON in
  both arms, fired identically, hashes `= = =`.
- Zero pass-2 firings; `rescue_unmatched=true` is a strict no-op on this
  sample.  Note this includes evt 444187, whose 24422-pt flashless orphan
  WAS scope-flipped to raw and restored by the pass every round — its
  byte-identical output is direct evidence the flip/restore path has no
  side effects on non-fired events.

### 7.3 Determinism (evt 56463, knob ON)

Four independent ON runs — `u17on56463`, the `u17on1kb` sweep, and two
fresh single-event tags `u17det1`/`u17det2` — give member-content-identical
`mabc-all-apa.zip` and pctree (3 pairwise `scripts/analysis/ql/ql_arm_compare.py` checks, all
`zip== pctree==`).

### 7.4 The nueCC48 near-miss (444187) — observation for the owner

The census's one nueCC candidate did not fire, and the final pctree makes
it look temptingly close: the 24422-pt / 515.7 cm flashless cluster (ident
5, TPC0, cathode tip x=−0.27 cm) has a raw closest approach of **1.88 cm**
to beam-window cluster ident 19 (3546 pts, 210 cm, t0=1.573 µs, TPC1) —
inside every distance cut.  Under `CATHODE_RESCUE_DEBUG=1`
(`work-nuecc48-u17dbg`) zero pair tests within 25 cm were reached in this
event, i.e. at rescue time (before examine_bundles / unmerge reshape the
clusters) no cross-TPC pair existed within `max_dis`; the final-pctree
pairing is a post-rescue configuration.  Also of note: ident 19 is matched
to pseudo-flash gid 1000000 (the empty-flash rescue class).  Whether the
pass should run later in the chain or admit pseudo-flash partners is an
owner call, out of scope here (the tracer itself validates on 56463: both
its pass-1 move and pass-2 adoption print ACCEPT).

## 8. Prototype provenance

None — new algorithm, no WCP counterpart (same status as pr/14).  Retire with
the parent component when the upstream flash reconstruction is fixed.

## 9. Related corrections surfaced (not fixed here)

- Doc pr/15 §6's table line "+5.746 µs bundle … holds the rejoined 3529-pt
  nu" is contradicted by the pctree: the rejoined nu is **flashless** (gid
  −1); the 5526-pt gid-1000000 main is a different crosser assembled by the
  pr/14 bundle pass.  §1 here is the corrected reading.
- Flashless clusters are invisible in every post-QL Bee layer (sentinel-T0
  corrected coordinates) — worth remembering when hand-scanning: check
  `img-global` for charge that no bundle color accounts for.

## 10. SBND default flip (pr/14 §9 pattern)

After the §7 validation, `cathode_rescue_unmatched` defaults flipped to
TRUE at all three SBND threading levels (`clus_all_apa`, the `all_apa`
wrapper, and the `wct-clus-matching-perevt.jsonnet` TLA); the common
factory arg and the C++ default stay FALSE, so every non-SBND consumer is
untouched.  Runner: unset `SBND_RESCUE_UNMATCHED` now inherits ON;
`=0` escapes to the legacy path, `=1` forces on explicitly.

Compile proofs (wcsonnet, production QL TLA set, `/home/xqian/tmp/cbr17/`):

- `flip_default.json` (no TLA) == `flip_on.json` (explicit
  `cathode_rescue_unmatched=true`): **byte-identical** — the new default IS
  the validated ON configuration.
- `flip_esc.json` (explicit `=false`) vs `flip_default.json`: the diff is
  exactly one line, `"rescue_unmatched" : true` — the escape suppresses the
  key and nothing else.
- `flip_esc.json` vs the pre-flip knob-off compile (`tree_off.json`)
  differs only by the concurrent pr/18 `protect_iso_band` /
  `protect_iso_band_xext` keys (commit c7d7fbcd, landed mid-campaign) —
  i.e. the escape path is byte-identical modulo that unrelated, already
  documented change.

Bare-default smoke (`work-mcp1kall-u17defsmoke`, entry 599, NO
`SBND_RESCUE_UNMATCHED` in the environment): the adoption fires from pure
defaults (`unmatched rescue round 0: c12 … + c19 … -> gid 6`) and the QL
products are member-hash identical to the validated ON arm
(`scripts/analysis/ql/ql_arm_compare.py` vs `u17on1kb`: `zip== pctree==`).
