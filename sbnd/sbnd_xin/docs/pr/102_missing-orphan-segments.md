# doc pr/102 — missing orphan segments on the 1000-event numuCC sample: near-vertex + hadronic shower audit

Status: **AUDIT, diagnosis only.** No C++ or jsonnet changed; the only
behavioural inputs are production defaults at toolkit HEAD `6bf0aafb` plus the
byte-neutral `traj_cover_probe` (gated log-only, pr/96 §9). Owner question:
*how serious is the "missing orphan segment" problem on the 1000 PR numuCC
events — near the vertex and in hadronic showers (EM-shower interiors are
tolerated) — and is it worth fixing?*

**Answer in one line: 40 / 445 evaluated events (9.0 %) carry at least one
flagged uncovered prong (median 6.2 % of the neutrino cluster's charge, max
23.8 %); the near-vertex core of it is 24 events (5.4 %) of which 20 are
hadronic-region misses; the pr/98–101 flips did not reduce it (42 → 40); two
mechanisms explain the top exhibits, one with a fix already specified
(pr/67 S1 / pr/96 F1 + a new length disjunct) and one newly measured here
(Steiner-fragmentation / nnf=0 shadowing) that no existing proposal reaches.
Moderately serious: worth one targeted admission round, not an emergency.**

---

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# 1. the fresh 1000-event mcp1k arm at HEAD (the "after" epoch).  ~50 min at
#    PR_JOBS=16 on a box shared with a 19-job PDVD campaign (loadavg <= 51/64).
PR_JOBS=16 PR_EXTRA_STAGES=pr_display SBND_TRAJ_COVER_PROBE=1 \
  ./run_pr_chain_batch.sh work-mcp1k-ql0819 work-pr102-head-mcp1k data
# 1000/1000 rc=0; 461 calib dumps; toolkit 6bf0aafb, local/lib/libWireCellClus.so
# mtime 2026-08-20 17:50 (newer than every clus/cfg source; git clean there).

# 2. the census (fork of pr96_uncover_census.py; that file is untouched)
python3 scripts/pr102_region_census.py work-pr102-head-mcp1k  --quiet \
  --tsv docs/pr/102_groups-head.tsv     --summary-tsv docs/pr/102_events-head.tsv
python3 scripts/pr102_region_census.py work-mcp1k-prod0819    --quiet \
  --tsv docs/pr/102_groups-prod0819.tsv --summary-tsv docs/pr/102_events-prod0819.tsv

# 3. before/after
python3 scripts/pr102_ab_compare.py docs/pr/102_events-prod0819.tsv \
  docs/pr/102_events-head.tsv --label-a prod0819 --label-b head

# 4. gates (sec 2.3): fork integrity + relocated pr/96 calibration
python3 scripts/pr96_uncover_census.py  work-mcp2k-prod0819/pr_evt{279955,70084,405707} \
  work-mcp1k-prod0819/pr_evt{283713,316025,395148} --tsv /tmp/a.tsv
python3 scripts/pr102_region_census.py --pr96-compat <same six dirs> --tsv /tmp/b.tsv
# stdout and TSV byte-identical

# 5. exhibit panels (before = prod0819, after = head)
python3 scripts/pr99_point_panels.py work-mcp1k-prod0819 work-pr102-head-mcp1k \
  <outdir> 490361 -176.8 -119.9 454.1 --box 35 --tag B2-hadstub-490361   # etc.

# 6. secondary graph-level check
PR86_DUMP_ARMS=work-pr102-head-mcp1k python3 pr86_orphan_census.py
```

Arms: `work-pr102-head-mcp1k` (new, registered in `docs/work-tags.md` and
`scripts/retire/PROTECTED.txt` before launch), reading `work-mcp1k-ql0819`;
before-epoch and calibration events read from `work-{mcp1k,mcp2k}-prod0819`
(nothing written into them). `work-pr102-dbg-mcp1k` reserved, **unused** — every
mechanism call below came from evidence already in the arm's own debug logs.

---

## 1. Symptom and scope

The owner asked for a general audit of *missing orphan segments* — imaged
charge with no fitted trajectory over it, doc pr/96's metric — on the
1000-event numuCC sample, with the scope stated as: near-vertex activity
always matters; hadronic showers matter ("somewhere between the EM shower and
primary track"); EM-shower interiors need not be fully reconstructed and must
not flood the metric.

pr/96 measured exactly this quantity but kept only groups within 15 cm of the
main vertex (`--dvtx 15`), *because* EM rinds flood the metric otherwise
(evt283713: 30 % "uncovered", all rind). The hadronic-shower region is
precisely the population that cut discards. This round replaces the blanket
cut with an ownership/region classifier.

## 2. The measurement

### 2.1 `scripts/pr102_region_census.py`

Fork of `scripts/pr96_uncover_census.py` (frozen; never edited). Unchanged:
the zip loader (charge = `clustering-global`, never `img-global`), the >3 cm
uncovered test against `track_fit-global`, 2 cm single-link grouping, the
shape features and cuts (`npts>=40, len>=5, rms<=0.8, qfrac>=0.03`), the
`pr54 isolated-residual drop` log join, the q=15000 main-vertex lookup.

New: a typing join against the calib dump (`PR_EXTRA_STAGES=pr_display`), on
two facts verified event-by-event this round:

- `track_shower.particle_id` is the **owning segment id** (values join
  `segments[].id` 100 %), NOT a pdg. Per-point pdg is the two-hop join
  point → segment → `segments[].particle_id` / `flag_shower` / `shower_id`.
- `showers[].id` is the **stem segment id**; `segments[].shower_id` joins it;
  `showers[].particle_id` is the shower pdg (211 = A5 hadronic re-type,
  pr/99 r3). EM membership = pdg-11 *shower membership*, never pdg 11 alone
  (`(pdg 11, flag_shower=false)` fit stems exist).

Each shape-passing group is matched to calib points at 1 cm (`f_m` = matched
fraction; median 1.00, 7/51 classified groups below 0.9) and classified,
first match wins:

| class | rule | flag |
|---|---|---|
| NEARVTX | dvtx ≤ 15 cm (`--dvtx-near`, pr/96 value) | yes |
| HAD-A5 | owned by / within 10 cm of a pdg-211 (A5) shower | yes |
| HAD-ADJ | within 10 cm (`--dhad`) of a hadron TRACK segment (pdg 2212/211/321, `flag_shower` false; **pdg 13 excluded** — that is corridor) or of a secondary ≥2-track vertex; EM fraction ≥ 0.8 overrides to EM-INT | yes |
| EM-INT | ≥ 60 % (`--ftype`) of matched points owned by pdg-11-shower members | **never** (owner scope) |
| TRK-COR | majority track-owned (the muon-wing / fit-quality family) | yes, severity 2 |
| OTHER / UNTYPED | none of the above / `f_m` < 0.5 | yes, hand-look |

Evidence columns per group: owner-pdg histogram, `d_had`+segment,
`d_a5`, `d_sec`, cross-cluster fit-less charge (`xcd`/`xcn`), the pr54 join,
and (post-pr/99 arms) the nearest `A5 hadronic census:` log record — the A5
line's `shower id` is the **ordinal** `shower_id`, mapped to the stem via the
dump.

### 2.2 Statuses

`NO_ZIP` (prod0819 only — its non-evaluated events were pruned), and per-zip
`MISSING_LAYER` (no `track_fit` layer = no fitted neutrino; **553 of 1000**
at HEAD), `NO_MAIN_VERTEX` (2), `FRAME_MISMATCH` (3; see §6.4). Scored
denominator = 445 events in **both** epochs, and it is the **same 445
events** — rates below are quoted over evaluated events, the 555
no-fitted-neutrino events reported as their own class exactly as pr/86 §2.3.

### 2.3 Gates, all run before any headline number

- **Fork integrity**: `--pr96-compat` vs `pr96_uncover_census.py` on the six
  pr/96 calibration events — stdout and TSV **byte-identical**.
- **Relocated pr/96 calibration.** `work-cbr3-census-on` (pr/96's calibration
  arm) was retired 2026-08-19; all six labelled events survive at the same
  prod0819 epoch (279955/70084/405707 in mcp2k, 283713/316025/395148 in
  mcp1k; frozen record `/home/xqian/tmp/pr96/calib6.tsv`). Verdicts: 279955 +
  70084 flagged, 283713 (the rind canary) + 316025 + 405707 clean —
  **5/6 as pr/96 §5**; 395148 fires at prod0819 where the cbr3 epoch had it
  below the `qfrac` floor (54 pts / 0.025 → 127 pts / 0.064, same structure,
  same coordinates) — and pr/96's **own** §6 prod0819 census lists 395148 as
  flagged, so this reproduces the published record exactly; the §5 table was
  cbr3-epoch-specific.
- **Classifier on the six**: 279955/70084/395148 → NEARVTX; 283713's 42 rind
  groups → none classified (fail the unchanged shape cuts), 0 flagged.
- **Typing join**: `main_vertex` ≡ zip q=15000 vertex within 0.1 cm on
  442/445; `particle_id ↔ segments[].id` join 100 %.
- **Hand-check (8 exhibits, 2 per class, prod0819)**: every one lands
  sensibly (§5 panels); thresholds frozen at the defaults with **zero**
  adjustment.

### 2.4 What the classifier changed structurally

pr/96's flagged count was *gated* by the unconstrained `--dvtx` cut. Here the
cut only moves groups between classes: total flagged events are 37 / 40 / 40
at `--dvtx-near` 10 / 15 / 25 cm (prod0819: 41 / 42 / 42). The one number the
owner cares about is no longer hostage to the one unpinned constant.

---

## 3. Population — mcp1k at HEAD (`work-pr102-head-mcp1k`, 445 evaluated)

| class | groups | events | rate | note |
|---|---|---|---|---|
| NEARVTX | 27 | **24** | 5.4 % | **22 groups (20 events) are hadronic-region**: owner segment pdg 2212/211, `d_had` ≤ 10, or A5-adjacent |
| HAD-A5 | 0 | 0 | — | no uncovered group *owned by* an A5 shower |
| HAD-ADJ | 2 | 2 | 0.4 % | beyond 15 cm of the vertex, still hadron-adjacent |
| EM-INT | 2 | 2 | — | reported, never flagged (owner scope) |
| TRK-COR | 18 | 17 | 3.8 % | 15 groups are parallel muon wings (ang ≤ 20°, no pr54 hit — the pr/94 §9.10 fit-quality family); **3 carry a pr54-dropped real candidate** |
| OTHER / UNTYPED | 0 | 0 | — | |
| **any flag** | 47 | **40** | **9.0 %** | flagged-group charge: median 6.2 %, p90 11.5 %, max 23.8 % of the cluster |

Charge fractions are of the neutrino cluster's imaged charge, i.e. directly
Enu-relevant.

**A5 crossover.** 16 showers re-typed hadronic (A5) in 15 / 1000 events; 4 of
those 15 events also carry a flagged uncovered group (315167, 395148, 410008,
486687). Two exhibits sit within 10 cm of an A5-**verdict-1** shower stem
(410008, 395148) — uncovered activity *inside the hadronic shower*, the exact
owner concern. Hadronic showers are strongly enriched in this defect
(4/15 = 27 % vs 9 % base rate).

### 3.1 Before/after — did pr/98–101 fix it?

`pr102_ab_compare.py`, identical 445-event denominator:

| | prod0819 (pre pr/98–101) | HEAD | fixed | persist | new |
|---|---|---|---|---|---|
| NEARVTX events | 24 | 24 | 7 | 17 | 7 |
| any-flag events | 42 | 40 | 9 | 33 | 7 |

**No.** The flips moved individual events (67394 — pr/86's adverse case — is
among the fixed; 174576/402880/287654 are class migrations, not losses) but
the population is stable. This is consistent with pr/96 §2.1's sharpest
finding: `fit_exclusion` acts on the *fits*, and the admission decisions were
bit-identical across it — the missing prongs here are admission losses, which
no shipped knob touches.

### 3.2 Secondary graph-level corroboration

`pr86_orphan_census.py` on the same arm (461 dumps): 33 events with ≥1 orphan
segment at the worse anchor (20×1, 8×2, 5×3). Its exemplar gate reads REVIEW
— expected: the pinned exemplar triples are from the pr/85-epoch arms, 15+
production flips ago. Quoted only as corroboration that the independent
graph-level defect rate (7.2 %) sits in the same few-percent band as the
charge-level rate (9.0 %).

---

## 4. Root causes — from the arm's own logs, no rerun needed

`SBND_TRAJ_COVER_PROBE=1` was on for the whole arm, so every
`find_other_segments` component's life (`pr67 fos step8/step9` KEEP / DROP /
SELECTED with bbox) is in the logs, as are the unconditional `pr54
isolated-residual drop` and mvga op1 lines. Bucketing the top 14 exhibits by
matching those lines to each group's own cluster and coordinates:

### B1 — found, fitted, then killed by the 25-Steiner-terminal floor (6 events)

The pr/96 §2 / pr/67 §9.2 family, alive at HEAD. The candidate is KEPT at
step8, SELECTED at step9, routed and fitted — then
`other_seg_keep_isolated_ok` (`clus/src/NeutrinoOtherSegments.cxx:32`) deletes
it because `min_points = 25` counts **Steiner terminals**, not size:

| event | dropped candidate | terminals | class |
|---|---|---|---|
| **292384** | **145.5 cm** | 23 | TRK-COR (a second, whole *track*) |
| **284794** | **71.3 cm** | 21 | NEARVTX (141 cm group, dvtx 7.4) |
| **387850** | **67.1 cm** | 16 | TRK-COR, 23.8 % of the cluster's charge |
| 277298 | 17.1 cm | 8 | NEARVTX (dvtx 4.3) |
| 386442 | 5.0 cm | 16 (nnf 9) | NEARVTX + A5-adjacent |
| 401450 | 4.3 cm | 3 | NEARVTX |

A 145 cm track deleted by a 25-*terminal* floor is the loudest single fact in
this audit. The floor was sized against 3–10-point noise (pr/54 §13) and is
doing that job — but it has no length term and no not-faked term.

### B2 — Steiner fragmentation / nnf=0 shadowing (7 events; **newly measured**)

The dominant *near-vertex hadronic* mechanism, previously invisible. The
finder looks exactly at the group's location — and sees only 1–3-terminal
fragments, each with `nnf=0` (every point reads as "faked": fewer than 2
planes see it clear of existing segments), dropped as `single_point` /
`nnf0_short` / `nnf0_2d_shadowed`. A group of 50–450 imaged points reduces to
almost nothing in residual-terminal space because it sits, in 2-D projection,
on top of the already-fitted prongs:

| event | group | fos components at the group |
|---|---|---|
| 490361 | 14.9 cm, 10.6 %, own 2212, dvtx 4.6 | single-point nnf=0 fragments at d = 0.4–5.8 cm |
| 278420 | 20.3 cm, 11.5 %, own 211, dvtx 4.0 | single-point nnf=0 at 9–10 cm |
| 282204 | 45.8 cm, 8.6 %, dvtx 4.7 | `nnf0_2d_shadowed` (7 pts, 10.9 cm!) + fragments |
| 284791 | 13.9 cm, 8.2 %, dvtx 4.6 | single-point nnf=0 |
| 486687 | 45.0 cm, 6.9 %, dvtx 3.7 | single-point nnf=0 |
| 410008 | 6.7 cm, 7.3 %, own 211, **A5 v=1 shower 4.2 cm away** | mixed: tiny KEEPs + nnf=0 drops |
| 487303 | 12.2 %, FRAME_MISMATCH event | nnf0_short at 8.6 cm |

pr/96 F1 (`min_nnf`) **cannot reach this family by construction** — the
discriminator it would admit on is exactly zero here. The 2-D shadow test and
the near-vertex hadronic geometry are structurally in conflict: short hadron
stubs at a busy vertex are always projection-shadowed by their neighbours.

### B3 — no residual component at all (2 events: 54341, 283771)

The finder produced nothing anywhere near the group (283771: nearest
component 21 cm away). Either the terminals were tagged (claimed by existing
segments) before grouping, or Steiner sampling left nothing. One step
upstream of B2; same conflict, earlier seat.

### B4 — fit-collapse + op1 dup-merge (279955 class): **zero hits**

No `mvga: op1 dup-merge` line accompanies any top exhibit on this sample.
pr/96's open residual is real but rare on mcp1k; the admission families above
dominate.

### The muon wings (15 TRK-COR groups)

Parallel to the owning muon at ≤ 20°, no dropped candidate at the charge —
the fit ran along the track but 3–10 cm of transverse charge is beyond the
3 cm coverage radius. The pr/94 §9.10 fit-quality family; visible, bounded,
low priority; kept flagged at severity 2 so it stays measurable.

---

## 5. Exhibits

Panels (before = prod0819 top row / after = HEAD bottom row; charge grey,
fits coloured, star = main vertex, X = group centroid):

- `102_panel-evt490361_B2-hadstub-490361.png` — three-prong vertex; a dense
  hadronic fan at the vertex with no trajectory through it, both epochs.
- `102_panel-evt278420_B2-isoblob-278420.png` — 20 cm isochronous-compressed
  near-vertex prong, fit only along its edge.
- `102_panel-evt292384_B1-longdrop-292384.png` — the 145.5 cm dropped track:
  a full-length unfitted line, identical in both epochs.
- `102_panel-evt277298_B1-neardrop-277298.png` — 17.1 cm candidate dropped at
  8 terminals, 4.3 cm from the vertex.
- `102_panel-evt410008_A5adj-410008.png` — uncovered charge inside an
  A5-tagged (verdict 1) hadronic shower.

Full rows: `102_groups-{head,prod0819}.tsv`, per-event
`102_events-{head,prod0819}.tsv`.

---

## 6. Verdict and proposals

### 6.1 Serious or not, per class

- **NEARVTX (5.4 % of evaluated events, 20/24 hadronic-region): serious
  enough to act on.** These are vertex prongs carrying a median 6 % (max
  14 %) of the cluster's charge — directly the owner's stated priority, and
  the pr/98–101 flips measurably did not touch the family.
- **Hadronic-shower interior (HAD-A5/ADJ + the A5-crossover NEARVTX subset):
  real but small** (2 events beyond 15 cm; 4 of 15 A5 events affected). Fixing
  B1+B2 fixes most of it — same mechanisms, same seats.
- **TRK-COR: two populations.** The 3 pr54-carrying events are B1 (serious —
  one is a whole missing track at 23.8 % of the cluster charge, one a 145 cm
  second track). The 15 muon wings are fit-quality cosmetics: **tolerate**.
- **EM-INT: tolerated by scope** (2 events, reported only).
- **The 555 no-fitted-neutrino events** are out of this audit's reach by
  construction (no fit to compare against) and remain the standing pr/86 §8.1
  question.

### 6.2 Proposals (none implemented here; all default-OFF knobs when built)

- **P1 — build pr/67 S1 / pr/96 F1, `other_seg_keep_isolated_min_nnf`**, and
  add the second disjunct this round's evidence demands:
  **`other_seg_keep_isolated_min_length_admit`** (admit any candidate whose
  fitted `track_length` ≥ L cm regardless of terminal count; L ~ 30–50 from
  §4 B1 — the dropped tail is 67–145 cm against noise components that are
  ≤ 10 cm by pr/54 §13's own sizing). Targets with measured discriminators:
  386442 (nnf 9), 277298 (nnf 8), and the three long drops. Gate per pr/96
  §9: knob-off byte-identical on the standard manifests; knob-on census with
  `pr102_region_census.py` as the before/after metric (a segment that exists
  but does not cover the charge is not a fix — pr/96 F1's own caveat).
- **P2 — the B2 family needs a design decision, not a threshold.** The
  nnf/2-D-shadow test is structurally blind to near-vertex hadronic stubs.
  The honest shape is pr/96 F2 (an uncovered-charge disjunct *inside* the
  existing predicate) but seated **earlier** — at the terminal
  tagging/grouping step where B2's components fragment, not at the accept
  test where B1 dies. Concretely: a terminal whose imaged charge is > 3 cm
  in 3-D from every existing *fit* trajectory should not be droppable as
  2-D-"faked" without a second look. This round supplies the measured
  population (7 of the top 14) and the seat
  (`step8 nnf0_2d_shadowed / single_point`); it deliberately does not pick
  the mechanism — surfaced for the owner per house rule.
- **P3 — nothing for the muon wings** (pr/94 §9.10 family, cosmetic) and
  **nothing for B4** here (rare on this sample; 279955 remains pr/96's open
  item).

### 6.3 What this round explicitly does not do

No knob flips, no floor changes (`min_points` 25 stays — pr/54 §13 sized it
against real noise and pr/96 §8 already declined lowering it), no
re-litigation of `fit_exclusion` (ON since pr/98 and measured here as not the
lever for this family).

### 6.4 Surfaced, not chased

1. **3 FRAME_MISMATCH events** (286681, 400636, 487303): the calib dump's
   `main_vertex` is 269–365 cm from the zip's q=15000 vertex — a
   different-vertex pick between the dump and the Bee zip writers, pr/86 §8.2
   family. 487303 is also a 12.2 % NEARVTX exhibit.
2. **395148** moved from below the qfrac floor (cbr3 epoch) to flagged
   (prod0819/HEAD): its near-vertex structure grew between epochs. Owner
   hand-look candidate; also carries an A5 v=1 shower.
3. The A5 census line's `shower id` is the ordinal `shower_id`, not the stem
   segment id — recorded here because the join is easy to get wrong.
4. mcp2k (2000 events, same epoch roots) was not re-produced this round;
   §3's rates are mcp1k. The census runs on any arm as-is.

---

## 7. Verification ledger

- Fresh arm: 1000/1000 `rc=0`; 461 calib dumps (identical evaluated
  population to prod0819); loadavg kept ≤ 51/64 alongside a colleague's
  19-job PDVD campaign (PR_JOBS=16, not 32, for that reason).
- Fork integrity byte-identical (stdout + TSV) on the six calibration events;
  relocated pr/96 calibration 5/6 with the sixth reproducing pr/96's own
  published prod0819 row; canary 283713: 0 flagged.
- Same-445-event denominator in both epochs (comparator prints only-A/only-B
  as 0/0).
- M13: nothing written into any existing arm, label dir, or snapshot; new
  outputs under `work-pr102-head-mcp1k` and `docs/pr/102_*`.
- No toolkit edit: `git -C toolkit status` clean in `clus/ cfg/`;
  the two named probes are the pr/96-gated ones, both proven byte-neutral in
  that round.

---

# ROUND 2 (2026-08-20/21) — P1+P2 implemented; len_admit=30 SBND PRODUCTION ON; min_nnf + uncover_3d built, validated, STAY OFF

Owner request: *implement P1 and P2 with default-OFF knobs, validate, and if
validation passes turn them on by default for SBND production; 32 CPUs.*
Outcome in one line: **three knobs shipped default-OFF; the length disjunct
(`other_seg_keep_isolated_len_admit = 30`) passed every gate and is
PRODUCTION ON; the nnf disjunct and the P2 3-D uncovered radius FAILED
validation (measured below) and stay OFF**; one latent shower-view
use-after-free crash found and fixed along the way.

## 7. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# knobs (all default OFF in C++): SBND_OSEG_MIN_NNF / SBND_OSEG_LEN_ADMIT (cm)
# / SBND_OSEG_UNCOVER_3D (cm) -> other_seg_keep_isolated_{min_nnf,len_admit} /
# other_seg_uncover_3d on tagger_check_neutrino.

# Gate 1 (knob-off byte identity, SHIPPING binary), 234/234 PASS:
for s in nuecc48 ncpi0 mcp1k mcp2k; do
  python3 scripts/pr85_hash_gate.py work-pr102r2-base-$s work-pr102r2-off2-$s; done

# the operating-point arms (before/after for every number below):
#   before work-pr102r2-offmerged-mcp1k   after work-pr102r2-l30full-mcp1k
SBND_OSEG_LEN_ADMIT=30 PR_JOBS=32 PR_EXTRA_STAGES=pr_display SBND_TRAJ_COVER_PROBE=1 \
  ./run_pr_chain_batch.sh work-mcp1k-ql0819 work-pr102r2-l30full-mcp1k data

# adjudication:
python3 scripts/pr90_movers.py work-pr102r2-offmerged-mcp1k work-pr102r2-l30full-mcp1k --tags harv3
python3 scripts/pr102_region_census.py work-pr102r2-l30full-mcp1k --quiet
python3 scripts/pr102_ab_compare.py docs/pr/102_events-r2off.tsv docs/pr/102_events-l30.tsv

# flip-equivalence (bare config, no env, vs the validated arms): 18/18 PASS
python3 scripts/pr85_hash_gate.py work-pr102r2-l30full-mcp1k work-pr102r2-flip-mcp1k
python3 scripts/pr85_hash_gate.py work-pr102r2-l30-ncpi0    work-pr102r2-flip-ncpi0
```

Binary: toolkit at this round's two commits on top of `2979bd26` (doc 74);
`wcdoctest-clus` 2305/2305; freshness proofs before every gate. The round rode
through two concurrent-session hazards, both proven harmless: a mid-arm
library relink (binary equivalence 6/6 archives, `work-pr102r2-beq2-mcp1k`)
and a transient cfg-signature collision (167 affected events patch-rerun and
key-audited: 1000/1000 compiled configs carry the knob keys).

## 8. What was built (all C++ defaults OFF ⇒ byte-identical)

- **P1a `other_seg_keep_isolated_min_nnf`** (int, 0=off): admit an isolated
  residual below the 25-Steiner-terminal floor when its number_not_faked ≥ N
  (pr/67 §10.3 S1; nnf was computed at the seat and thrown away).
- **P1b `other_seg_keep_isolated_len_admit`** (cm, 0=off): admit any isolated
  candidate whose fitted track length ≥ L, at any terminal count — the §4 B1
  answer (the floor counts terminals, not size; the dropped tail was
  67–145 cm against noise ≤ 10 cm).
- **P2 `other_seg_uncover_3d`** (cm, 0=off): imaged charge farther than R in
  3-D from every existing fitted trajectory (i) cannot be 2-D-tagged in
  find_other_segments step 1, and (ii)+(iii) counts toward nnf at step 8 and
  the re-eval — the §4 B2 (nnf=0 fragmentation) answer.
- Threading: `NeutrinoPatternBase.h` → `TaggerCheckNeutrino.{h,cxx}` →
  `cfg/pgrapher/common/clus.jsonnet` → `sbnd/clus.jsonnet` (4 sites) →
  `sbnd/wct-pr-perevt.jsonnet` → `run_pr_chain_batch.sh` envs; doctest pins +
  6 new predicate cases (`doctest_other_seg_keep_isolated.cxx`).
- **Crash fix (unknobbed, UB, gated)**: the pr/99 ghost-member drop removes
  a segment + degree-0 vertices from the FULL graph after purging only the
  dropping shower's view; any other shower still holding those descriptors
  iterates freed storage (`Shower::fill_sets` UAF — reproducible SIGSEGV on
  18255-399998 with P2 on, rc=135 twice). Fixed by purging every shower's
  view first (`NeutrinoShowerClustering.cxx`, pure filter-set erase).
  Knobs-off byte-identity re-proven after the fix (Gate 1 below is on the
  post-fix binary).

## 9. Gates

| gate | result |
|---|---|
| Gate 0: build + doctest | rc=0; **2305/2305** |
| compiled-config OFF (pristine cfg tree vs edited, same TLAs) | byte-identical |
| compiled-config ON | exactly the 3 keys appear |
| Gate 1 knob-off, shipping binary, 4 samples (nuecc48 96, ncpi0 38, mcp1k numu50 70, mcp2k numu50 30 archives) | **PASS 234/234** |
| peer-binary equivalence (mid-round relink) | PASS 6/6 |
| flip compiled-diff | exactly `other_seg_keep_isolated_len_admit: 30` added |
| flip-equivalence bare-vs-env (mcp1k 16 + ncpi0 2 archives) | **PASS 18/18** |

## 10. Validation statistics (this is §"statistics" the owner asked for)

### 10.1 The failed operating point: P1(min_nnf=4, len_admit=30) + P2(3.0 cm)

Full mcp1k before/after (`offmerged` vs `onmerged`, identical 445-event
evaluated set):

- Census: flagged events 40 → 33; flagged uncovered charge Σqfrac
  3.28 → 2.09 (−36%); on the 12 §5 exhibit events, flagged groups 15 → 4.
- **Vertex movers: 106, of which 28 ADVERSE** (moves up to 237.6 cm off a
  correct click) — the stop-the-line class.
- **nueCC48 nue-score sign flips: 12 (≈8 losses incl. six 4.30 → −15)**;
  14 numu flips; 45/48 events changed. The pr/30 fit_exclusion failure
  shape, reproduced.

Single-knob attribution on the 28 ADVERSE events:

| knob alone | movers | ADVERSE |
|---|---|---|
| min_nnf=4 | 3 | 1 (170792, 2.16 cm) |
| len_admit=30 | **0** | **0** |
| uncover_3d=3.0 | 24 | **23** |

**P2 is the destroyer** — its step-1 re-tagging changes component formation
in every cluster with any 3-D-far charge, and the resulting segments re-anchor
vertex selection globally. Powerful (it produced most of the −36% charge
recovery) but far too blunt at 3.0 cm. **STAYS OFF.**

### 10.2 The nue ledger sweep that set the operating point (nuecc48, 48 evts)

| corner | events changed | nue lost | nue gained |
|---|---|---|---|
| min_nnf=4 | 36/48 | **4** (42280, 350186, 389538, 444187) | 1 (30504) |
| min_nnf=8 | 24/48 | 1 (389538 4.3 → −15) | 1 (30504) |
| **len_admit=30** | **2/48** | **0** | **0** |
| min_nnf=8 + len_admit=30 | 24/48 | 1 | 1 |

min_nnf's admissions of small (4–8-terminal) fragments inside shower-rich
nue events perturb the nue BDT at scale; even at 8 it carries the named
389538 loss. **min_nnf STAYS OFF** pending an owner hand-scan of
389538-vs-30504 (the machinery is built, gated, and one env var away).

### 10.3 The shipped point: len_admit = 30 cm, full-sample numbers

- **Footprint: 5 admissions / 1000 events**, every one `reason=len`:
  292384 (145.5 cm, nnf 18), 284794 (71.4 cm, nnf 0), 387850 (67.1 cm,
  nnf 16), 279256 (45.3 cm, nnf 19), 288859 (43.9 cm, nnf 5).
- mcp1k: 6/1000 events changed (the 5 admits + none); movers **2, zero
  ADVERSE**; cosmict flips 0; numu sign flips **1** — 18255-284794
  (2.12 → −1.28, vertex unmoved 0.14 cm) — the round's named hand-check
  item: its recovered 71.4 cm second track is real charge, and whether the
  event should still select numu is the owner's call (Bee idx 2).
  (349549/401450 diffs are the M4 DL-vertex bit-instability: identical PR
  structure — 50 seg/86 vtx/13 shower and identical keep/drop lines — with
  a jittered score; not the knob.)
- nuecc48: 2/48 changed, zero nue/numu sign flips. ncpi0: 1/19 changed
  (37112 nue −2.50 → −2.47).
- Census: flagged events 40 → 38; the two rescues are exactly the §4 B1
  giants (292384, 387850). 284794's track is admitted but its event region
  stays flagged (partial coverage).
- The B1 tail of doc pr/102 §4 is closed: **no candidate ≥ 30 cm can die on
  the terminal floor in SBND production any more.**

### 10.4 Bee review (before/after, 10 events, index `bee/pr102r2/pr102r2.index.txt`)

- before (knobs off): https://www.phy.bnl.gov/twister/bee/set/c94d6274-7628-4f82-b0a5-ce1021bc9c60/event/list/
- after (len_admit=30): https://www.phy.bnl.gov/twister/bee/set/a5710cfa-5f69-43f0-85b3-1d64f269bbdd/event/list/
- Priority: idx 0/1 (the rescued 145.5/67.1 cm tracks), **idx 2 (284794
  numu flip — the one open adjudication)**, idx 5–8 (what min_nnf / P2
  would additionally recover, for a future round's scope decision).

## 11. Residuals and the P2 path forward (recorded, not chased)

1. The B2 family (nnf=0 fragmentation, 7 of the audit's top 14) remains the
   dominant unfixed near-vertex-hadronic mechanism. P2 as built fixes it
   (Stage A: exhibit flagged groups 15 → 4) but at 23 ADVERSE movers. The
   measured narrowing for a next round: apply the 3-D escape ONLY at the
   step-8/re-eval nnf seats (no step-1 re-tagging — the re-eval is the seat
   pr/67 P5 already named as able to kill real charge with no 3-D
   evidence), and/or a larger radius; the knob split is trivial on top of
   this round's plumbing.
2. min_nnf=8: −1/+1 on nueCC48; owner scan of 389538 (lost) vs 30504
   (gained) decides.
3. 170792 (min_nnf's one small ADVERSE, 2.16 cm) — Bee-able on request.
4. 18255-399998's crash was the exposed instance of the ghost-drop UAF; the
   fix is unconditional (UB), gated by this round's 234/234.
