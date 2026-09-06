# Steiner terminals on wrapped induction planes — the PDHD charge ceiling on three detectors, the fix that was never applied, and what "nearby wires" can and cannot buy

Round of 2026-09-05. Follows `stm-tagger-chain.md` §12 (the charge census that
found `ncharge = 3` is exactly zero on PDHD) and §13 (the hand scan that rejected
the retiler knob). Owner's three questions, in order:

1. Do PDVD and SBND have the same problem? **No** (§2).
2. If not, is it PDHD's wire geometry — "more chaotic" wire crossings? **It is
   the wrap topology, which is regular, meeting a reverse (wire → channel)
   lookup that only exists on one code path** (§3, §4).
3. Is it a problem, and how should PDHD be fixed robustly, considering nearby
   wires? **It is a problem; the fix already exists and was never applied where
   it matters** (§4.4, §5). "Nearby wires" answers a *different*, detector-
   independent residual, measured here for the first time (§6), and a design
   with a measured upper bound is in §7.

Status flags are stated per claim: **byte-identical** gates in §8; everything
in §5 is a **knob-on smoke measurement on 30 events, NOT hand-scanned**.

## 0. Repro

```bash
# Two binary pins.  Sections 2-5 use the sec 8 pin of stm-tagger-chain.md (unchanged
# code); section 6 needs the probe extension of this round and uses a NEW pin.
#   /home/xqian/tmp/pdhdstm_libpin   libWireCellClus.so f143ab82fa999663...  (toolkit 869a554c)
#   /home/xqian/tmp/d46_libpin       libWireCellClus.so c6ecaf2e95bfd51c...  (this round's toolkit commit)
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd
export LD_LIBRARY_PATH=/home/xqian/tmp/pdhdstm_libpin
TLA="-S retile_wrapped_channel_activity=true -S wrapped_channel_charge=true"

# --- sec 2-4: the census on EXISTING dumps (no reconstruction), then the arm sec 12.4 never ran
W=../../wire-cell-data/protodunehd-wires-larsoft-v1.json.bz2
python3 docs/scripts/d46_steiner_plane_multiplicity.py --wires $W --det pdhd --geometry     # sec 3
for e in 0 10 12 14; do mkdir -p work/029107_${e}_phdumpwc; ln -sfn $PWD/work/029107_${e}_stm0/pctree-evt*.{tar.gz,tlas} work/029107_${e}_phdumpwc/; done
for e in 0 10 12 14; do WCT_STEINER_PHASE_DUMP=1 PDHD_KEEP_CFG=1 PDHD_PR_TLA="$TLA" ./run_pr_evt.sh -s phdumpwc -stm -stm-fit 029107 $e; done
python3 docs/scripts/d46_steiner_plane_multiplicity.py \
    "pdhd_prod_4evt:work/029107_*_phdump/wct_pr_029107_*.log" \
    "pdhd_prod_evt0:work/029107_0_phdump/wct_pr_029107_0.log" \
    "pdhd_retiler_evt0:work/029107_0_phdumpw/wct_pr_029107_0.log" \
    "pdhd_clussampler_evt0:work/029107_0_wccdump/wct_pr_029107_0.log" \
    "pdhd_clussampler_retiler_evt0:work/029107_0_wccdumpw/wct_pr_029107_0.log" \
    "pdhd_both_4evt:work/029107_*_phdumpwc/wct_pr_029107_*.log" \
    "pdhd_both_evt0:work/029107_0_phdumpwc/wct_pr_029107_0.log" \
    "pdhd_both_probe_evt0:work/029107_0_phdumpx/wct_pr_029107_0.log" \
    --thr 500 --validate --wires $W --det pdhd \
    --tsv docs/figs/d46_plane_multiplicity_pdhd.tsv --seg-tsv docs/figs/d46_segment_attribution_pdhd.tsv
python3 docs/scripts/d46_steiner_plane_multiplicity.py \
    "pdvd_039252_2_d31r6e2e:../pdvd/work/039252_2_d31r6e2e/wct_pr_039252_2.log" \
    "pdvd_039252_2_probe:../pdvd/work/039252_2_d46dump/wct_pr_039252_2.log" \
    --thr 500 --validate --det pdvd --tsv docs/figs/d46_plane_multiplicity_pdvd.tsv
python3 docs/scripts/d46_steiner_plane_multiplicity.py \
    "sbnd_5evt:../sbnd/sbnd_xin/work-d31r7probe2/pr_evt*/wct_pr_evt*.log" \
    "sbnd_2evt_probe:../sbnd/sbnd_xin/work-d46probe/pr_evt*/wct_pr_evt*.log" \
    --thr 4000 --validate --det sbnd --tsv docs/figs/d46_plane_multiplicity_sbnd.tsv

# --- sec 5: the 30-event arm with both knobs, and the sec 12.5 metrics
for d in work/029107_*_stm0; do n=${d%_stm0}_stmwc; mkdir -p $n; ln -sfn $PWD/$d/pctree-evt*.{tar.gz,tlas} $n/; done
PDHD_MAX_JOBS=8 PDHD_KEEP_CFG=1 PDHD_PR_TLA="$TLA" ./run_pr_evt.sh -s stmwc -stm -stm-fit 029107 all
python3 docs/scripts/d46_harvest_counters.py stm0 stmw stmwc --churn
for arm in stm0 stmw stmwc; do
  python3 docs/scripts/d42_proj2d_resid.py --det pdhd --out /home/xqian/tmp/d46/ana/resid_$arm work/029107_*_$arm/tracking-stm.root
  python3 docs/scripts/d42_dqdx_rr.py --det pdhd --ref stm/pdhd_ref_dqdx.json --ref-key MuonDeDx --out /home/xqian/tmp/d46/ana/dqdx_$arm work/029107_*_$arm/tracking-stm.root
done   # -> docs/figs/d46_stm_arms.tsv (columns documented in the file header)

# --- sec 6: the probe extension (toolkit commit of this round), NEW pin, env set vs unset
export LD_LIBRARY_PATH=/home/xqian/tmp/d46_libpin
WCT_STEINER_PHASE_DUMP=1 PDHD_KEEP_CFG=1 PDHD_PR_TLA="$TLA" ./run_pr_evt.sh -s phdumpx -stm -stm-fit 029107 0
                         PDHD_KEEP_CFG=1 PDHD_PR_TLA="$TLA" ./run_pr_evt.sh -s d46base -stm -stm-fit 029107 0
cd ../pdvd   # pctree of 039252_2_d31r6e2e predates -save-assoc, hence the guard override; both arms share it
WCT_STEINER_PHASE_DUMP=1 PDVD_KEEP_CFG=1 PDVD_ALLOW_NO_ASSOC=1 ./run_pr_evt.sh -s d46dump -stm-fit 39252 2
                         PDVD_KEEP_CFG=1 PDVD_ALLOW_NO_ASSOC=1 ./run_pr_evt.sh -s d46base -stm-fit 39252 2
cd ../sbnd/sbnd_xin; R=$PWD/work-d46probe; mkdir -p $R
for e in 2 14; do ln -sfn $PWD/work-dbg25a-d97prodchk/ql_evt$e $R/ql_evt$e; done
for i in 1 5; do SBND_WORK_ROOT=$R WCT_STEINER_PHASE_DUMP=1 ./run_pr_evt.sh mc -nu $i; done   # idx 1 = evt 2, idx 5 = evt 14
# gates (sec 8)
cd ..; python3 ../abtest/hash_archive.py pdhd/work/029107_0_{phdumpx,d46base}/mabc-pr.zip pdvd/work/039252_2_{d46dump,d46base}/mabc-pr.zip
```

Committed products: this doc; `docs/scripts/d46_steiner_plane_multiplicity.py`,
`docs/scripts/d46_harvest_counters.py`; `docs/figs/d46_plane_multiplicity_{pdhd,pdvd,sbnd}.tsv`,
`docs/figs/d46_segment_attribution_pdhd.tsv`, `docs/figs/d46_stm_arms.tsv`; the
correction pointer under `stm-tagger-chain.md` §12.4. Toolkit: the env-gated
probe extension in `clus/src/SteinerGrapher.cxx` (§6.1). Not committed: the arms
`work/029107_*_{phdumpwc,stmwc,phdumpx,d46base}`, `pdvd/work/039252_2_{d46dump,d46base}`,
`sbnd_xin/work-d46probe`.

## 1. What the question is

`Cluster::calc_charge_wcp` (`Facade_Cluster.cxx:1031-1112`) is the Steiner
candidacy gate. The only Steiner call site passes `disable_dead_mix_cell = false`
(`CreateSteinerGraph.cxx:323`), so a plane with charge exactly 0 is *exempt* and
the RMS is taken over the non-zero planes — **but is 0 unless more than one plane
is non-zero**. `ncharge > 1` is therefore the terminal ceiling at any threshold.
§12.3 measured that on PDHD no point has three-plane charge and 78 % have fewer
than two. The retiled cluster the gate reads is never persisted, so every number
below comes from the `WCT_STEINER_PHASE_DUMP` probe's `phase=P0_cluster` lines,
parsed by one tracked script for all three detectors.

**Positive control, re-run.** The script reproduces the candidate predicate
offline and compares its count with the run's own `steiner_p1_blobs: ncand_pt`
counter, summed over the calls whose `npt > 1000` (the dump's own cut) of the
component that produced the dump. Two traps the scratch parser of §12 avoided by
hand and this script handles explicitly: `CreateSteinerGraph` runs
`find_steiner_terminals` **twice** per cluster with the same flags
(`establish_same_blob_steiner_edges` at `CreateSteinerGraph.cxx:299`, then
`create_steiner_tree` at `:323`), so the counter line repeats verbatim and must be
counted once; and `ImproveCluster_1` emits the same counter under its own
component tag for its own calls. With both handled the control is **exact** on
the production arms (126 910 = 126 910 over 4 events) and within the dump's 0.1 e
print precision elsewhere (largest deviation +40 of 344 720).

## 2. Question 1: the three detectors on one instrument

Same predicate, same probe, same script. PDVD and SBND numbers are from dumps that
already existed (`pdvd/work/039252_2_d31r6e2e`, the round-6 end-to-end arm with
both wrapped-plane knobs ON; `sbnd_xin/work-d31r7probe2`, five ν-MC events, no
knobs — SBND has nothing to fix).

| retiled-cloud points | PDHD **production** (4 evt) | PDHD retiler ON (evt 0) | PDHD **both knobs** (4 evt) | PDVD 039252/2 | SBND 5 ν-MC |
|---|---|---|---|---|---|
| n | 922 469 | 169 471 | 823 571 | 455 972 | 39 251 |
| `ncharge = 0` | 0.206 | 0.123 | 0.058 | 0.173 | 0.178 |
| `ncharge = 1` | 0.577 | 0.393 | 0.163 | 0.146 | 0.215 |
| `ncharge = 2` | 0.217 | 0.435 | 0.247 | 0.219 | 0.259 |
| **`ncharge = 3`** | **0.000** | 0.048 | **0.532** | 0.462 | 0.348 |
| **eligible (`> 1`)** | **0.217** | 0.484 | **0.779** | 0.680 | 0.608 |
| combinations seen | W, UW, VW, U, V | + UVW, UV (0.008) | all 7 | all 7 | all 7 |
| q = 0 on U / V / W | 0.799 / 0.938 / 0.252 | 0.703 / 0.659 / 0.228 | 0.248 / 0.259 / 0.241 | 0.346 / 0.340 / 0.345 | 0.375 / 0.411 / 0.436 |
| candidates at the arm's floor | 0.138 @ 500 e | 0.302 | 0.419 | 0.383 @ 500 e | 0.176 @ 4000 e |

**PDVD and SBND do not have the PDHD defect.** Both show every plane
combination, symmetric zero fractions across the three planes, and a three-plane
share of 0.35–0.46. PDHD production shows an asymmetry that no charge effect
produces — U and V zero on 80 % and 94 % of points while W is zero on 25 % — and
with the fix (§4.4) it lands at 0.53, *above* both others.

The symmetric ~0.35 zero fraction that survives on all three detectors is a second
phenomenon, taken up in §6.

## 3. Question 2: the wrap topology, measured

From `protodunehd-wires-larsoft-v1.json.bz2` (`--geometry`):

| PDHD, every (anode, face) | U | V | W |
|---|---|---|---|
| wires / distinct channels | 1148 / 800 | 1148 / 800 | 480 / 480 |
| segments in **pitch order** | seg-0 ×400, seg-1 ×400, seg-2 ×348 | seg-2 ×348, seg-1 ×400, seg-0 ×400 | seg-0 ×480 |
| channels by (segments in total, segments on the **sensitive** face) | (2,1): 104 · (3,1): 348 · **(3,2): 348** | same | (1,1): 480 |

Three things follow, and none of them is "chaotic":

- **The stripes are regular and mirror-symmetric.** Along the pitch axis each
  induction plane is three contiguous diagonal stripes; V's order is U's reversed
  because the two planes wrap in opposite senses. A channel's segments alternate
  faces (seg 0 and seg 2 on one face, seg 1 on the other) and never change plane
  or APA. W does not wrap; its two faces read disjoint channel ranges.
- **On the one sensitive face per APA** (`params.jsonnet` nulls the wall face:
  even anode → face 0, odd → face 1) the 400 **segment-1** wires belong to
  channels whose segment-0 wire is on the *dead* face. `Gen::AnodePlane` builds
  `IWirePlane::channels()` by skipping every `segment > 0` wire
  (`gen/src/AnodePlane.cxx:242-256`), so the sensitive plane's channel list does
  not contain those 400 channels at all. That is the fact every defect below rests
  on.
- **348 of 800 channels have two segments on the sensitive face** (seg 0 and
  seg 2, ≥ 400 pitches = 187 cm apart). That is the irreducible physical ambiguity
  of a wrapped APA: one channel lights two wires in one RayGrid layer
  (`img/src/GridTiling.cxx:121-142` fans a channel out to all its wires on the
  face). The other 452 channels are unambiguous on the sensitive face.

PDVD for contrast: 1568 continuations (11.3 % of wires), always one per channel,
always on the *sibling* face, and both faces are live — so a continuation's
channel is missing from *its own* face's list but its charge is physical on both.
SBND: zero continuations; every wire→channel idiom is exact there by construction,
which is why SBND cannot exhibit either defect.

## 4. The mechanism, proven by segment attribution

### 4.1 Two reverse lookups on one path

Every *forward* fan-out in the toolkit (channel → its wires) is correct: the
tiler (`GridTiling.cxx:122`) and the ctpc builder
(`PointTreeBuilding.cxx:305-326`, which writes one channel's charge to every
segment, in the face each segment belongs to). The Steiner stage's input is built
by two *reverse* lookups, both broken on a wrapped face:

1. **The retiler's activity** (`improvecluster_1.cxx:1028-1064`): activity is
   collected per wire index (1148 entries), and the ISlice the sampler will read is
   keyed by `channels[wire_idx]` — the 800-entry channel list indexed by a wire
   index. U survives only at indices < 400 and happens to be addressed correctly
   there; V's surviving 400 entries are mis-addressed by 748 wires; everything at
   index ≥ 400 is dropped. `retile_wrapped_channel_activity` resolves by ident
   through `IAnodePlane::channel()` instead.
2. **The sampler's per-point charge** (`BlobSampler.cxx:383-426`): the point's
   nearest wire → its channel ident → `p_chi2i` (ident → position in *this*
   plane's list) → the slice activity. A segment-1 wire's channel is not in the
   list; the legacy path inserts 0 and reads `channels[0]`, normally silent, so
   `charge_val = charge_unc = 0`. `wrapped_channel_charge` resolves by ident
   against the activity instead. Segment-2 wires are *not* affected: their
   channel's segment 0 is on the same face, so it is listed.

### 4.2 The discriminating test

For every dumped point of event 029107/0, the nearest U and V wire and its
segment were recomputed offline from the wires file (same pitch-argmin rule as
`pimpos->closest`, sensitive face per APA):

| P(q ≠ 0 \| segment of the nearest wire) | seg 0 | **seg 1** | seg 2 | share of points on seg 1 |
|---|---|---|---|---|
| U, production | 0.426 | **0.000** | 0.010 | 0.59 |
| V, production | 0.310 | **0.000** | 0.009 | 0.52 |
| U, retiler ON | 0.755 | **0.006** | 0.689 | 0.60 |
| V, retiler ON | 0.691 | **0.026** | 0.747 | 0.54 |
| **U, both knobs** | 0.741 | **0.728** | 0.688 | 0.61 |
| **V, both knobs** | 0.688 | **0.725** | 0.746 | 0.55 |

The zero is deterministic by segment, not statistical. With the retiler fixed,
seg 0 and seg 2 read at the ~0.7 level a healthy plane shows on every detector,
and **seg 1 — 55–60 % of the points — reads zero**: that is defect 2 alone. In
production (defect 1 as well) seg 2 dies too and even seg 0 drops to 0.43 / 0.31,
because the tiling itself was built from mis-addressed activity.

Joint distribution on the retiler-ON arm: both U and V non-zero only where the
nearest wires are (seg 0, seg 0) or (seg 2, seg 2) — 0.50 of those points — and
never in any combination containing a seg-1 wire. Those two stripe intersections
are 13 % + 7 % of the face by area (uniform-face prior computed by the script),
which is the 0.048 of §12.4. In production the retiler's scrambled V activity
kills the (0,0) region as well (0.000), which is why §12.3 read "never UV". The
"chaotic crossings" reading is wrong: the pattern is three stripes and their
mirror images.

### 4.3 Why §12.4's 2 × 2 could not see it

§12.4 concluded `wrapped_channel_charge` "does nothing for terminals" from an arm
that set it in the **clustering** job (`PDHD_CLUS_TLA`) and re-ran the PR job
without it. The retiler's samplers are the PR job's own eight `BlobSampler`
instances (`cfg/pgrapher/experiment/pdhd/pr.jsonnet:1263`), and they take the
knob from the `wrapped_channel_charge` argument of `pdhd/wct-pr-perevt.jsonnet:87`.
The compiled config proves the consumer set: those eight samplers are referenced
by exactly one component, `ImproveCluster_2:pr`, and nothing else. So the
clustering-job knob rewrote the persisted pctree's charges, which the Steiner
stage never reads (`feedback_retiled_cloud_not_the_sampled_one`), and the arm
that repairs the segment-1 stripe was never run — `wccdumpw` ≡ `phdumpw` to three
decimals is the signature. §9's worry that the PR-job knob must match the Q/L job
does not apply either: the retiled cloud is discarded after the stage.

### 4.4 The arm that fixes it

`PDHD_PR_TLA="-S retile_wrapped_channel_activity=true -S wrapped_channel_charge=true"`
— compiled-config proof before any number was read: 8 × `wrapped_channel_charge:
true` on the samplers, `wrapped_channel_activity: true` on the retiler,
`terminal_charge_threshold` 500. The 2 × 2 becomes a 2 × 2 × 2 on event 029107/0:

| PR job: retiler / sampler | `ncharge = 3` | eligible | cand @ 500 e | q = 0 on U |
|---|---|---|---|---|
| off / off (production) | 0.000 | 0.134 | 0.082 | 0.912 |
| on / off (§12.4's "retiler on") | 0.048 | 0.484 | 0.302 | 0.703 |
| off / on | 0.000 | 0.134 | 0.082 | 0.912 |
| **on / on** | **0.495** | **0.755** | **0.393** | **0.276** |
| (clustering-job sampler on, PR off / on — the §12.4 rows) | 0.000 / 0.048 | 0.134 / 0.475 | 0.082 / 0.295 | 0.911 / 0.705 |

Over the four §12 events: `ncharge = 3` **0.000 → 0.532**, eligible **0.217 →
0.779**, candidates at 500 e **0.138 → 0.419**, zero fraction on U/V **0.80/0.94
→ 0.25/0.26** — the same 0.24 the W plane has. This is PDVD's production
configuration (`protodunevd/pr.jsonnet:52,95` pass both), which is why PDVD reads
0.46.

## 5. Question 3a: is it a problem — the 30-event arm

The §12.5 table gains its missing row. Same 30 pctrees, same pin, same scripts
(`docs/figs/d46_stm_arms.tsv`; the `stm0`/`stmw` rows re-derive the published
ones exactly):

| arm (PR job knobs) | accepted passes | few-terminal warns | no-steiner | terminals in | `f_off_far` U | `f_off_near` U | `k_pop` | contrast ≥ 2 (n) | doc-55 stopping μ | STM tags |
|---|---|---|---|---|---|---|---|---|---|---|
| `stm0` production | 175 | 3719 | 1317 | 163 639 | 0.721 | 0.047 | 0.888 | 17 | 1 | 175 |
| `stmw` retiler only (§13 scanned this) | **207** | 1560 | 574 | 385 623 | 0.586 | 0.056 | 0.927 | 24 | 1 | 211 |
| **`stmwc` both knobs** | 178 | **960** | **319** | **489 043** | **0.565** | 0.059 | **0.971** | 19 | **3** | 182 |

Read it carefully, because the columns disagree:

- **The Steiner stage itself is healthiest with both knobs** — few-terminal
  bail-outs fall by 74 % (3719 → 960), clusters with no Steiner cloud by 76 %,
  terminals triple, the coverage number `f_off_far` reaches its best value, the
  dQ/dx population scale `k_pop` moves from 0.888 to 0.971 (a 3 % residual against
  the reference instead of 11 %), and the doc-55 stopping-muon selection finds 3
  tracks where every other arm finds 0–1.
- **Accepted STM fit passes do NOT follow**: 178, essentially production's 175,
  against the half-fix's 207. And the STM tag count returns to 182 from 211.
- **The tag *set* moves as much as ever**: 30/30 events differ from `stm0`
  (60 kept by both, 122 gained, 115 lost — 237 contested tags against §13's 224),
  and 30/30 differ from `stmw` too (139 kept, 43 gained, 72 lost).

So the half-fixed arm §13 scanned was a **different object**: its candidates
lived only in the two corner stripes (§4.2), i.e. a spatially biased subset of
every track, and it produced 29 more accepted passes than either the broken or
the fully fixed chain. §13.3's verdict — that the added tags were 73 %
through-goers — was a verdict on that biased population. The full fix changes
STM verdicts on 237 tags relative to production, and **nothing here says in which
direction**. That is a hand scan, on a fresh label tag (M13), and it is the
owner's; the `stm_scan` app and its blind (§13.1) apply unchanged. The
Steiner-side counters say the input to the taggers is now what PDVD's is; whether
the taggers do better with it is the open question §13 left, now on the right
arm.

**Not done, deliberately:** no default was flipped. `pdhd/wct-pr-perevt.jsonnet`
still carries both knobs `false`; production PDHD PR output is untouched by this
round.

## 6. Question 3b: the residual, and what "nearby wires" would reach

### 6.1 The probe extension

Every zero fraction in §2 that survives the fix — ~0.25 on PDHD, ~0.35 on PDVD and
SBND — was unexplained, and the owner's "consider nearby wires" is a hypothesis
about exactly that population: a sampled point whose nearest wire reads nothing
while a wire one or two pitches away does. The probe could not answer it (it
carried no wire index), so `create_steiner_tree`'s dump lines gained ten fields,
appended last so every existing parser keeps matching its prefix:

`wu wv ww` (the point's wire index per plane), `uu uv uw` (charge uncertainty
per plane; the retiler's dead/forced sentinel is `(0, 1e12)`,
`improvecluster_1.cxx:1058-1059`), `ts` (the point's blob slice index), and
`nu nv nw` — the largest ctpc charge on a wire within ±2 of the point's wire in
the same slice, face and plane, the point's own wire excluded. The ctpc 2-D clouds
are what the retiler built its activity from (`get_activity_improved` →
`Grouping::get_overlap_good_ch_charge`), so "a neighbour had charge" is read from
the retiler's own source; one window query per (apa, face, plane) per call, then
map lookups. All of it sits under the existing `WCT_STEINER_PHASE_DUMP` gate —
byte-identical when unset (§8).

### 6.2 What the zero planes are

| zero plane, split | PDHD both knobs, evt 0 | PDVD 039252/2 | SBND 2 ν-MC |
|---|---|---|---|
| zero fraction U / V / W | 0.276 / 0.281 / 0.246 | 0.278 / 0.263 / 0.260 | 0.401 / 0.393 / 0.502 |
| of which **sentinel** (`unc = 1e12`) | **1.000 / 1.000 / 1.000** | 0.999 / 1.000 / 0.977 | 0.935 / 0.961 / 0.918 |
| sentinel **with** a charged wire within ±2 | 0.454 / 0.478 / 0.685 | 0.673 / 0.679 / 0.489 | 0.243 / 0.276 / 0.213 |
| live wire, a charged neighbour within ±2 | 0 / 0 / 0 | 0.001 / 0 / 0.017 | 0.031 / 0.017 / 0.043 |
| live wire, nothing within ±2 ("empty") | 0 / 0 / 0 | 0 / 0 / 0.006 | 0.034 / 0.021 / 0.038 |

**The residual is not a lookup failure and not a tie.** On all three detectors
the zero planes are, to within a few percent, wires the *retiler itself* marked
with the `(0, 1e12)` sentinel. Two writers share that sentinel: dead channels
(`get_activity_improved` step 2) and the wires `hack_activity_improved`
**forces** around every path point — a disc of radius 3 in (wire, tick) around
each trajectory sample (`improvecluster_1.cxx:586-593`), written wherever the
plane had no activity so that the tiling connects along the path. The probe does
not separate the two, but SBND ν-MC has few dead channels and still reads
0.40–0.50, so the forced band is the bulk of it there. Those wires have no
measured charge by construction; the sampler reports exactly what the retiler
wrote. That the W plane of SBND is zero on half its points is the width of that
forced band, not a signal-processing property.

Hence §2's symmetric ~0.35 on PDVD/SBND: it is the fraction of retiled points that
sit on forced or dead wires of a given plane, and the healthy PDHD arm now shows
the same ~0.25.

### 6.3 What a neighbour fallback could buy

Whether a forced wire has a *charged* neighbour within ±2 is what decides the
owner's idea, and the split above answers it: 45–69 % of the forced/dead points
on PDHD, 49–68 % on PDVD, 21–28 % on SBND. Taking every zero plane to its ±2
neighbour maximum raises the eligible fraction

| eligible (`ncharge > 1`) | now | with the ±2 fallback | gain |
|---|---|---|---|
| PDHD both knobs | 0.755 | **0.919** | +16 pts |
| PDVD | 0.749 | **0.943** | +19 pts |
| SBND | 0.596 | **0.720** | +12 pts |

That is an **upper bound** (every zero plane recovered, no quality condition),
and it is a real lever on all three detectors — larger, on PDVD and SBND, than
anything the threshold scans of docs pdvd/25 and pdvd/31 moved. It is also, by
construction, a change to *which* point in a blob wins the local-maximum test,
because a borrowed charge enters the RMS. §7 says what to do with it.

### 6.4 The ghost pairs are a small effect

The 348 channels with two sensitive segments (§3) are the one place where the
ident-resolved lookup is *over*-generous: a point on segment 0 is credited with
the channel's whole activity even when the deposit was on segment 2. On the
fully fixed PDHD dump, 33 % (U) and 38 % (V) of the points sit on such a channel,
but the partner segment is under another dumped point in the same slice for only
**2.4 % (U) and 1.1 % (V)** of all points — a lower bound (only clusters above
1000 points are dumped), but an order of magnitude below every other effect in
this document. Design note only; see §7 Tier 2.

## 7. Design: what to do, in order

**Tier 1 — the fix exists; apply it in the job that matters (owner decision).**
Both knobs in `pdhd/wct-pr-perevt.jsonnet`, i.e. PDVD's production pair. The
only thing standing between measurement and flip is the hand scan of §5, on a
fresh tag: the population is the 237 contested (event, cluster) tags of
`stm0 → stmwc`, stratified as §13.1 did, in the same blind viewer. The
Steiner-side evidence for the flip is unambiguous (§5's first bullet); the
tagger-side evidence does not exist yet. The Q/L clustering job's own
`wrapped_channel_charge` — the persisted pctree, which feeds dQ/dx, track fitting
and `MultiAlgBlobClustering.cxx:2971` — stays parked with the owner as §9 left
it; it is independent of this round's mechanism and of its measurements.

**Tier 3 (before Tier 2, because it is larger) — a candidacy-side neighbour
fallback, `terminal_charge_neighbor_wires` (C++ default 0 = off).** The
measured upper bound is +12 to +19 points of eligibility on every detector
(§6.3). The robust form, which follows from what the zeros *are* (§6.2):

- Apply it in `find_peak_point_indices` (the Steiner caller), **not** in
  `BlobSampler` — the persisted pctree and every other consumer of
  `charge_value` stay byte-identical, and the fallback is scoped to terminal
  candidacy where the retiler's forced wires are the population.
- Fire only when the plane reads **the sentinel** (`charge_unc > 1e10`), never on
  a live wire that measured zero: on SBND the "live, nothing nearby" points are
  real empties (3 %) and must stay exempt as they are today.
- Take the ±k neighbour **maximum from the ctpc in the same slice**, k = 2 (the
  probe's measurement; k = 3 would exceed the forced band's own radius and start
  reading the next track).
- Use the borrowed value for **eligibility only** (it makes `ncharge > 1`
  reachable), and keep the local-maximum ranking on the RMS of *measured* planes
  — otherwise the point that wins a blob is the one with the best neighbour, which
  is the wrong object. This is the part that makes it robust rather than merely
  bigger: it can add terminals in the forced band without moving any terminal
  that exists today.
- Grade it the way doc pdvd/37 graded `terminal_min_separation`: terminal
  density and largest terminal-free run per track (`steiner_terminal_attribution.py`),
  on PDVD and SBND as well as PDHD, since the lever is detector-independent.

**Tier 2 — collection-anchored de-ghosting, PDHD only, design note.** For the
348 two-segment channels, apportion an induction channel's activity between its
two sensitive segments by the W charge of the retiled blobs covering each
segment in the same slice; W never wraps, so it is the one unambiguous anchor,
and a segment with no covering blob gets weight 0, leaving 452 of 800 channels
untouched. Measured need: ≤ 2.4 % of points (§6.4). Not worth a knob until Tier 1
is scanned and Tier 3 is graded; recorded so that the next person who sees an
over-charged induction point on PDHD knows the mechanism and its size.

**What "nearby wires" is not:** it is not the PDHD fix. Nothing in the
neighbour fallback touches the segment-1 stripe — a point on a seg-1 wire has no
charge on *any* of its neighbours in the legacy lookup, because the whole stripe
is unlisted. Tier 1 first.

## 8. Gates and controls

| gate | result |
|---|---|
| Compiled-config proof, `phdumpwc` / `stmwc` / `phdumpx` | 8 × `wrapped_channel_charge: true` (samplers), `wrapped_channel_activity: true` (retiler), threshold 500 — checked before any number was read |
| Offline predicate vs `ncand_pt` (§1) | EXACT on `phdump` 4 evt, `phdump` evt 0, `wccdump`, SBND; +1 / +1 / +5 / +40 / +61 of 51 140 / 50 572 / 68 861 / 344 680 / 124 409 elsewhere (0.1 e print precision at the floor); `npt` sums match the dumped point count on every arm |
| Probe extension inert, PDHD 029107/0, same binary, env unset vs set | `mabc-pr.zip` member hashes identical (`5d1d46eb…`, 9 members) |
| Probe extension inert, PDVD 039252/2, same binary, env unset vs set | `mabc-pr.zip` member hashes identical (`b1a020be…`, 21 members); the `-stm-fit` pipeline writes no calib dump (`pr_display` is inert without `TrackFitting`), so there is nothing else to compare |
| `./build/clus/wcdoctest-clus` | 22 720 / 22 720, rc = 0; zero compiler warnings on the touched file |
| Freshness | `local/lib/libWireCellClus.so` 18:21:23 > `SteinerGrapher.cxx` 18:20:09; `d46_libpin` copy hashes `c6ecaf2e…` = `local/lib` |
| Arms | `phdumpwc` 4/4 rc 0; `stmwc` 30/30 `pr_resource` markers, runner rc 0; `phdumpx`/`d46base`/`d46dump`/`d46base`/SBND 2 all rc 0 |
| Binary pins | §2–§5 on `pdhdstm_libpin` (`f143ab82…`, stm-tagger-chain §0); §6 on `d46_libpin`. No number in this document compares across the two pins: the probe arm `phdumpx` reproduces `phdumpwc`'s census on the same event to the last digit (68 866 candidates both) |

Not claimed: an img/clus A/B (vacuous for `SteinerGrapher`, doc pdvd/37); a
cross-pin identity (the new build contains a peer's uncommitted `TrackFitting`
work, which the Steiner stage does not call — the same-event agreement above is
the evidence, not a gate).

## 9. Recommendation and next step

1. **Hand-scan `stm0 → stmwc`** (237 contested tags, fresh label tag) with the
   §13 apparatus; the PR-job pair flips on that verdict alone. This is the one
   blocking item and it is the owner's.
2. **Build Tier 3 as a default-OFF knob** in the shape of §7 (sentinel-only,
   ±2 ctpc neighbour, eligibility-only) and grade it on all three detectors with
   `steiner_terminal_attribution.py`. It is the first lever in this campaign that
   moves PDVD and SBND by more than a threshold retune does.
3. Do not spend more rounds on the floor (`steiner_terminal_charge`): §12.5 and
   §5 agree it is not binding on any arm.

## 10. Files

Tracked: `docs/steiner-wrapped-planes.md` (this), `docs/scripts/d46_steiner_plane_multiplicity.py`,
`docs/scripts/d46_harvest_counters.py`, `docs/figs/d46_plane_multiplicity_{pdhd,pdvd,sbnd}.tsv`,
`docs/figs/d46_segment_attribution_pdhd.tsv`, `docs/figs/d46_stm_arms.tsv`, the
§12.4 pointer in `docs/stm-tagger-chain.md`. Toolkit: `clus/src/SteinerGrapher.cxx`
(probe fields, env-gated). Inputs reused: `work/029107_*_{stm0,stmw,phdump,phdumpw,wccdump,wccdumpw}`,
`pdvd/work/039252_2_d31r6e2e`, `sbnd_xin/work-d31r7probe2`, `sbnd_xin/work-dbg25a-d97prodchk/ql_evt{2,14}`.
