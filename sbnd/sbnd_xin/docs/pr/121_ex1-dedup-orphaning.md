# doc pr/121 — the examine_shower_1 dedup orphaning fix (evt348471, doc 115 §17.7)

**Status: COMPLETE — `shower_ex1_dedup_rehome` SBND PRODUCTION ON 2026-08-28
(owner: validation "consistent with previous round"). Fires on exactly 1 event
in 239; 348471 qF1 0.205→0.895; flip-equivalence PASS 196/196 + 282/282.
Bee pair uploaded (below) for owner review.**

Owner directive 2026-08-28: investigate and improve the regression reported in
doc [pr/115 §17](115_em-handscan-categorisation.md) — the 141-event
out-of-sample scan found `shower_pass4_best_owner` (SBND production ON since
pr/117) orphans 12 segments on evt 17394-348471. Validation bar "consistent
with previous round" (pr/117–120). This doc is the regression front; the
recognition front is doc [pr/122](122_recognition-round.md). The doc-number
continuity: §17's before/after Bee pair was committed under `bee/pr121r1/`
(`e16624d`), and `em_display/prep_pr121.py` is §17's sidecar builder — this
round continues that number.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# arms (toolkit <HASH>, wcbuild + freshness proof + wcdoctest-clus 2452/2452)
/home/xqian/tmp/pr121_arms.sh 98  off0   0     # probe build, no env
/home/xqian/tmp/pr121_arms.sh 98  dbgA   1     # + WCT_SHOWER_{ABSORB,CONTENT,PID,TOPO}_DEBUG
/home/xqian/tmp/pr121_arms.sh 141 base141 0    # post-pr/120-flip production baseline, 141 evts
/home/xqian/tmp/pr121_arms.sh 141 dbg141 1
/home/xqian/tmp/pr121_arms.sh 98  off1   0     # knob build, knob off
/home/xqian/tmp/pr121_arms.sh 141 off141 0
/home/xqian/tmp/pr121_arms.sh 98  on1   0 SBND_SHOWER_EX1_DEDUP_REHOME=1
/home/xqian/tmp/pr121_arms.sh 141 on141 0 SBND_SHOWER_EX1_DEDUP_REHOME=1
# + work-pr121r1-dbgB-* (11 seed events, probes; doc pr/122 gidx join),
#   work-pr121r1-dbgon-mcp1k (348471 ON+probes, hash-identical to on141),
#   work-pr121r1-{flipchk,flip141}-* (post-flip config, no env)

# census
./scripts/pr121_dedup_census.py --tsv docs/pr/pr121-dedup-census.tsv \
    'work-pr121r1-dbgA-*' 'work-pr121r1-dbg141-*'
```

## 1. Root cause — three mechanisms chained, one defect

From the probe stream (`work-em114c-prodnowdbg-mcp1k/pr_evt348471`), the final
dumps of both arms, and the code:

1. **Both arms**: `examine_showers_retarget_seed` makes shower 60026 (15
   members, 352.6 MeV) absorb proton seg 12052 (54.1 cm, pdg 2212) and re-seat
   its start segment to 12052.
2. **Knob ON only**: `pass4_angle_divert` re-owns 3 segs to shower 63050; the
   changed composition flips `examine_shower_1`'s association gates; the trial
   shower1 (start seg 12052) is **accepted** and splices 63050's 7 segments
   (`SHOWER_ABSORB SPLICE site=examine_shower_1_assoc into_start_seg=12052
   from_start_seg=63050 from_nseg=7`).
3. **The defect** (`clus/src/NeutrinoShowerClustering.cxx`, accept-branch
   dedup): the loop erases ANY pre-existing (main_vertex, conn-1) shower whose
   start segment equals shower1's. Its own comment says "stale single-segment
   wrapper", but the predicate never checks the segment count — here it erases
   the retargeted ex-60026 shower with its 12 EM members, **re-homing
   nothing**. shower1's own walk holds only 12052 + the spliced 7;
   `detach_track_stem` then peels the proton head, leaving the shower rooted
   at 37017 with 92.0 MeV; the 12 ex-60026 members dump with `shower_id = -1`.

Segments 12052/12054 are unowned in BOTH arms — that is the benign
proton-stem detach, not part of the defect.

## 2. Exposure census — the defect is exactly one event in 239

`pr121_dedup_census.py` over the two probe arms (98-event `emscan-0827`
manifest + 141-event `emscan-0828-agent5` manifest, 239 events; both arms
byte-neutral, §4):

```
EX1_DEDUP fires: 1  (erase single-seg wrapper: 0, ERASE MULTI-SEG: 1, kept: 0)
  MULTI-SEG ERASE evt348471 into=12052 old_shower=2 nseg=13 kine=709.9 MeV final_orphans=14/29
```

The dedup **never fires on its design case** (a stale single-segment wrapper)
anywhere in either manifest; its only fire in 239 events is the 13-member
erase. Context for the splice site itself: 21 `examine_shower_1_assoc`
splices across the 239 events (12 events); in all 20 benign splices the
receiving root segment stays owned by its own shower — 348471 is the unique
case whose receiver was re-rooted onto a retarget-seeded proton stem.

## 3. The knob

`shower_ex1_dedup_rehome` (C++ default **false** = legacy drop =
byte-identical). When on, a multi-segment dedup victim is absorbed into
shower1 (`add_shower`, the same re-home shape the pr/84 start-seg dedup uses)
before the erase, and shower1's kinematics are recomputed once (the existing
kinematics call predates the dedup loop). Single-segment wrappers keep the
legacy drop either way.

Seats: `NeutrinoPatternBase.h` / `TaggerCheckNeutrino.{h,cxx}` (configure /
default_configuration / pattern_algos push) / `doctest_clus_knob_defaults.cxx`
pin / `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` decl +
key-suppression / runner env `SBND_SHOWER_EX1_DEDUP_REHOME`.

Expected knob-on outcome on 348471: the ex-60026 members re-home into the
accepted shower, the proton head is peeled as before, and the event returns
to (at least) its pre-pass4 grouping — with the 63050 side now merged, which
is the direction the scan asked for: the verdict on this event is
`under-clustered`, with 63050 explicitly marked IN.

**Recommendation carried by this round**: keep `shower_pass4_best_owner` ON
with this fix (the §17.6 census showed the two merge knobs clean and the two
non-orphaning pass4 hand-offs, 181050/292524, are re-arbitration whose
verdict is the owner's — they are idx 1/2 of the uploaded `bee/pr121r1/`
pair).

## 4. Gate ledger

| # | gate | arms | result |
|---|---|---|---|
| 1 | probe build == production (98 set) | `pr120r1-flipchk-*` vs `pr121r1-off0-*` | **PASS 196/196** |
| 2 | probes byte-neutral (98 set) | `off0` vs `dbgA` | **PASS 196/196** |
| 3 | post-pr/120 production baseline (141 set) | `em114c-prodnow-*` vs `pr121r1-base141-*` | 6 events differ, ALL attributed: 7 `pr120 stem_backfill_back_guard` fires (67394 283515 286655 292643 179369 347824) — the pr/120 flip acting on this sample, expected |
| 4 | probes byte-neutral (141 set) | `base141` vs `dbg141` | **PASS 282/282** |
| 5 | knob build, knob off (98 set) | `off0` vs `off1` | **PASS 196/196** |
| 6 | knob build, knob off (141 set) | `base141` vs `off141` | **PASS 282/282** |
| 7 | compiled config, knob off | git-HEAD vs worktree, full-pipeline TLA | **PASS** (cmp rc=0) |
| 8 | compiled config, knob on | `shower_ex1_dedup_rehome=true` TLA | **PASS** (key present in tagger node) |
| 9 | ON vs OFF (98 set) | `off1` vs `on1` | **PASS 196/196** — zero fires |
| 10 | ON vs OFF (141 set) | `off141` vs `on141` | **281/282** — evt348471 the ONLY divergence, = the census prediction |
| 11 | ON probe arm == clean ON arm | `on141` vs `dbgon` (348471) | **PASS 2/2** |
| 12 | flip-equivalence | post-flip cfg, no env, both manifests vs `on1`/`on141` | **PASS 196/196 + 282/282** |

**Binary provenance, recorded because it is not clean** (the doc 115 §17.6
situation in reverse): a concurrent session rebuilt the shared
`local/lib/libWireCellClus.so` at 18:41 — after this round's knob build
(18:19, which produced the off/on/dbg arms) and at the start of the
flip-equivalence arms, which therefore ran that binary (this round's
committed sources plus the concurrent session's uncommitted default-OFF MCS
work). The gates carry the proof anyway: `flipchk`/`flip141` (18:41 binary,
flipped config, no env) are byte-identical to `on1`/`on141` (18:19 binary,
env knob) on all 478 archives — so the concurrent work is output-neutral
AND the config flip equals the validated operating point.

## 5. Validation — the knob does exactly one thing, and it is the labeled thing

**Firing set = 1 event in 239.** `owned_census.py` OFF→ON:

```
98 set : events changed 0 of 98,  owned 5632 -> 5632
141 set: events changed 1 of 141, owned 3751 -> 3763 (net +12), 0 events lose energy
  348471   showers 8->8   owned 15->27/29   leading 92.0 -> 427.8 MeV (+335.7)
```

**evt348471 ON**: one shower `60026` with 19 members — the 12 formerly
orphaned ex-60026 EM segments AND the 63050 side; only the benign proton stem
12052/12054 stays unowned (`detach_track_stem`, identical to the pre-pass4
reconstruction). nusel TSV byte-identical, main vertex identical. The probe
stream shows the exact designed sequence: the assoc splice, `EX1_DEDUP
... old_nseg=13 erase=1`, `pr121 ex1_dedup_rehome: shower1 seg=12052 absorbs
dedup victim shower_id=2 nseg=13`.

**Score vs the `emscan-0828-agent5` marks** (charge-weighted, member dQ from
the hash-identical `dbgon` probe arm): qF1 **0.205 → 0.895** (precision
0.859, recall 0.934). The scan-epoch shower was 0.932 — the fix lands
*above* a plain restore in the labeled direction (the verdict was
"under-clustered" with 63050 marked IN; ON merges the 63050 side). The
residual: "extra" = the 63050-side companion segs 31011/39019/63051 the
marks never enumerated; "miss" = 38018, an IN mark that was already unmerged
at the scan epoch — pre-existing under-clustering untouched by this knob.

**Bee A/B** (`bee/pr121r2/`):
- BEFORE (current production) `5e51037e-72e9-44ca-b42e-f3049f2e21c0`
- AFTER (dedup re-home) `cfe993ce-3da4-4a57-8f3b-790af0a9536d`
- 1 event, annotated index `bee/pr121r2/pr121r2.index.txt`.

**Flip**: `shower_ex1_dedup_rehome = true` in the SBND tcn knobs
("consistent with previous round" pre-authorization, 2026-08-28);
flip-equivalence arms (post-flip config, no env) hash-gated against the ON
arms on both manifests: **PASS 196/196 + 282/282** (gate 12).

## 6. Files

| file | |
|---|---|
| `scripts/pr121_dedup_census.py` | EX1_DEDUP census (fork header inside) |
| `docs/pr/pr121-dedup-census.tsv` | the one census row |
| `/home/xqian/tmp/pr121_arms.sh` | arm launcher (both manifests; committed copy: `scripts/pr121_arms.sh`) |
| toolkit `clus/src/NeutrinoShowerClustering.cxx` | probes + knob |
| toolkit `clus/src/PRShower.cxx` | SHOWER_PID probes (doc pr/122) |
