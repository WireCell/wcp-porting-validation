# PDVD doc 27 — the 039349/53 "isolated piece" was a v6 point tree re-tiled with v7 faces (stale-geometry inputs, not over-clustering)

**Status:** ROOT-CAUSED; process fix + guards shipped. The over-clustering
investigation the owner ordered in doc 26 §8 is closed as *not a clustering
defect*: the input cluster is one continuous 200 cm track. What produced the
"isolated piece 75 cm away" and the chords across a "charge void" was a
geometry mismatch between the point tree the PR job read (sampled with the
`protodunevd-wires-larsoft-v6` wires file on 2026-09-02) and the anodes the PR
job compiled after the 2026-09-03 07:51 config commit `228f1c39`
(`protodunevd-wires-larsoft-v7-uvwfit`). The two wires files order the two
faces of anodes 2, 3, 6 and 7 oppositely, so every re-tiled cluster on those
anodes landed one face height (168.4 cm) off in y.

Three deliverables: (1) the PDVD runners now record the wires file in the
imaging and point-tree provenance and **refuse** to run a job whose compiled
wires file differs from its input's (`PDVD_ALLOW_STALE_GEOMETRY=1` downgrades
to a warning); (2) a log-only WARN in `ImproveCluster_1` when a cluster's
points lie outside the face volume it is being re-tiled into (toolkit; outputs
byte-identical, §6.3); (3) a fresh, self-consistent 120-event arm `d27fresh`
(imaging + clustering + PR all on today's geometry) replaces the mixed arms as
the baseline for the doc 25 campaign (§6.4).

**Corrections to earlier documents.** Doc 26 §7.5 said of 039349/53 "here the
void is real in the charge too" — wrong; the Bee `clustering` layer I compared
against was in the input frame while the fits were one face off, so the
"charge" I found near the fits was a coincidental cluster. The owner's doc 26
§8 decision 2 (039349/53 is over-clustering, separate the pieces) rested on
that statement and is withdrawn here. Doc 26 §3/§7.1 attributed the verdict
churn between the pre- and post-07:51 arms to the E = 450 V/cm operating
point; at least the PR-stage part of it is this face swap. Doc 26 §7.5's other
case, 039349/14 (anodes 4/5, faces unaffected), keeps a genuine steiner
coverage defect (§4.3) and stays with the Steiner-terminals campaign.

## 0. Repro

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/pdvd
# the stale mix, as run on 2026-09-03 09:40 (pctree from 2026-09-02 11:50, v6):
ls -la --time-style=full-iso work/039349_53_d25r13fix/pctree-evt23277.tar.gz   # 09:24 copy of the 11:50 file
python3 /home/xqian/tmp/doc27/tag_check.py d25r13fix 53 4639.6 53               # 0/469 steiner pts on charge at dy=0, 469/469 at dy=+168.4
# the regenerated chain (today's geometry, tag d27v7): imaging -> clustering -> PR
./run_img_evt.sh  -s d27v7 039349 53           # frames symlinked from work/039349_53_keep/, input/ dir present
PDVD_LIGHT_SUFFIX=_keep ./run_clus_evt.sh -s d27v7 -save-pctree -calib 039349 53
./run_pr_evt.sh   -s d27v7 -stm-fit 039349 53  # 49 s; no "degenerate split skipped" line
python3 /home/xqian/tmp/doc27/tag_check.py d27v7 53 4639.6                      # every steiner cloud on its charge at dy=0
python3 stm/gates/r27_face_swap_png.py                                            # docs/pics/doc27_039349_53_face_swap.png
# the guard (refuses before wire-cell starts):
./run_pr_evt.sh -s d27guard -stm-fit 039349 53   # sidecar says v6, job compiles v7 -> rc=3
# the 120-event self-consistent arm:
/home/xqian/tmp/doc27/fresh_pipeline.sh          # img -> clus -> pr for stm/events.txt under tag d27fresh
```

`tag_check.py` and the probe scripts live in `/home/xqian/tmp/doc27/`
(scratch); the committed versions are `stm/gates/r27_face_swap_png.py` and
`stm/gates/r27_tag_check.py`.

## 1. Symptom

Doc 26 §7.5, 039349/53 PR cluster 53: the STM fit and the steiner cloud ran
from A (x 343, y 140, z 40) along a track to z ≈ 87, then nothing for 75 cm,
then a 17-point piece at V (x 210, y 112, z 161); three PR segments were
straight chords across the gap, and the near-duplicate pair among them drove
the `examine_partial_identical_segments` fixed point that doc 26 fixed. Read
against the Bee `clustering` layer, the gap looked like a gap in the imaged
charge as well.

## 2. Root cause

### 2.1 The input cluster is continuous; the steiner cloud is one face away

The point tree the PR job read (`work/039349_53_d25r13fix/pctree-evt23277.tar.gz`,
a copy of the 2026-09-02 11:50 Q/L output) has 61 live clusters. The one with
`matched_flash_gid` 127 and `cluster_t0` 4639.6 us — PR cluster 53 — is input
cluster 60: 1556 points, `x_t0cor` 206–344 cm, **y 224–335 cm**, z 27–166 cm,
wire-plane ids 103 and 119 (anodes 6 and 7, face 0), with 200–330 points in
every 20 cm of z from 40 to 160. No gap.

PR cluster 53's steiner cloud (469 points) has **zero** points within 3 cm of
any point of the input tree, and **all 469** within 3 cm of input cluster 60
once shifted by +168.4 cm in y. Its STM fit (y 112–139) is shifted the same
way. The x–z projection matches the input cluster (bottom-left panel of the
picture); only y is off, by exactly one face height.

### 2.2 The two wires files order the faces of anodes 2, 3, 6, 7 oppositely

Parsing the wire endpoints of both files (`Store.anodes[].faces[]`):

| anode | v6 face 0 | v6 face 1 | v7-uvwfit face 0 | v7-uvwfit face 1 |
|---|---|---|---|---|
| 0, 1, 4, 5 | y [−168, −1] | y [−336, −168] | y [−168, −1] | y [−336, −168] |
| 2, 3, 6, 7 | **y [168, 336]** | **y [1, 168]** | **y [1, 168]** | **y [168, 336]** |

The point tree carries each blob's face index as sampled by the Q/L job (v6:
face 0 of anode 6 is the outer half). `ImproveCluster_1`/`RetileCluster`
build `m_face[apa][face->which()]` from the anodes of the job that is running
(v7: face 0 of anode 6 is the inner half) and re-tile the blob's 2D activity
in that face's ray grid. Same wire indices, other face → the retiled blobs
come out 168.4 cm lower in y. The face indexing in the toolkit is internally
consistent (`WirePlaneId::face()` = `IAnodeFace::which()` = position in
`anode->faces()`, both in the Facade cache and in the retiler); the two jobs
simply did not share a wires file.

### 2.3 Probe run

With a temporary WARN per (cluster, anode, face) in `ImproveCluster_1::mutate`
and after the retile in `CreateSteinerGraph` (tag `d27probe`, same stale
point tree):

```
doc27 probe: cluster 53 apa 6 face 0 (m_face which=0 ident=0 sens y=[0.6,168.5]) orig n=... y=[223.6,335.1] -> sampled y=[56.7,168.1]
doc27 probe: [main 53] src n=1556 y=[223.6,335.1] wpids=103,119 -> retiled n=1641 y=[56.7,168.1]
doc27 probe: cluster 14 apa 2 face 0 (sens y=[0.6,168.5]) orig y=[169.0,172.8] -> sampled y=[1.1,5.5]
doc27 probe: [main 60] src n=2794 y=[-324.9,-199.0] wpids=31,95 -> retiled n=11099 y=[-327.6,-196.8]   (anodes 1/5: no shift)
```

Every cluster touching anodes 2, 3, 6 or 7 moved by one face height; none on
anodes 0, 1, 4, 5 did. On this event 43 of the 61 live clusters and 46 % of
the points are on the swapped anodes.

### 2.4 Why the shifted cloud is also *incomplete*

The re-tile keeps a new blob only if it is supported by the original cluster
(`remove_bad_blobs`); with the originals one face away, only the blobs that
happen to project onto some original activity survive — hence the z 88–160
hole and the surviving fragments at z ≈ 105 and 161–164. That hole is what
turned a straight track into "two pieces plus chords", and the chords' shared
trunk is where the doc 26 fixed point lived.

## 3. Why it hid

- **The Bee `clustering` layer and the PR fits were in different frames**, and
  I compared them anyway. The `clustering` layer is the input tree (v6
  sampling); `stm_fit`, `track_fit`, `vertices` and the calib dump's `steiner`
  come from the retiled clusters (v7 faces). Points near the fits were a
  coincidental cluster; the input cluster's own points were 168 cm away in y
  and never entered the doc 26 picture. Lesson: before comparing two layers,
  match a known object across them (here: the flash-group id and t0 identify
  the cluster in both).
- **The arms straddled the config change.** Point trees: 2026-09-02 11:50
  (v6). PR arms `d25r13base/fix`, `d25r14*`: 2026-09-03 09:40–10:49 (v7).
  Nothing in the runner recorded which wires file a product was made with, so
  the PR job could not know its input was stale. Doc 26 §7.1 did list the
  commits between the arms and named `228f1c39` — but read it as a physics
  operating-point change, not as a geometry change that invalidates every
  earlier product.
- **A global y offset of one face height is invisible in x–z**, which is the
  projection the doc 26 pictures led with, and in-detector everywhere (both
  halves are sensitive volume). The regenerated run's steiner cloud is 3–4×
  denser (1457 vs 469 points) — the stale run also silently lost most of the
  cluster.
- `imaging` archives from July (v6) were never re-made. The `_keep` archives
  are still v6; every clustering job compiled after 07:51 that reads them has
  the same problem one stage earlier (blob faces from v6, sampling with v7).
  No such Q/L arm exists in `work/` at the time of writing (checked: no
  `clusters-apa-*` newer than 07:51 outside `d27*`).

## 4. What this changes in docs 25 and 26

### 4.1 039349/53 is not over-clustering

The input cluster 60 is one track from (343, 305, 40) to (207, 279, 166) with
no gap; the regenerated PR (tag `d27v7`) re-tiles it into 1457 steiner points
that cover every 20 cm z bin the input does (§6.1). There are no chords, no
duplicate pair, no `degenerate split skipped` line, and the event runs in 49 s.
The owner's doc 26 §8 decision 2 — "the gap is real, do not connect the
pieces, look at separation after merges" — was made on my wrong §7.5 reading
and is withdrawn; there is nothing to separate.

### 4.2 The doc 26 verdict census and the "802 flips"

Doc 26 §7.1 compared the pre-crash-fix and post-crash-fix binaries **on the
same (stale) inputs and the same (v7) config** and found zero flips — that
result stands: the crash fix moves nothing. The 802 STM flips between the
pre-07:51 and post-07:51 arms, attributed to the E = 450 V/cm operating point,
are now known to contain the face swap on half the detector at the PR stage
(every retile, STM fit and steiner cloud on anodes 2, 3, 6, 7). How much is
physics and how much is geometry is answered by the `d27fresh` arm (§6.4),
which is the first PDVD arm whose imaging, clustering and PR share one wires
file since the change.

### 4.3 039349/14 keeps its steiner defect

Cluster 36 there is on anodes 4/5 (faces unchanged between v6 and v7), and its
steiner cloud was never shifted (426 of 632 points on charge at dy = 0). In the
regenerated run the cluster is longer (Bee 34, z 5–233) and its steiner cloud
(1726 points) covers z 0–90 and 140–240 but has **no points in z 100–140**
where the input has 660, plus 246 points at z < 40 where the input has 6. That
is a real Steiner-terminal/graph coverage question and stays with the
campaign the owner opened for it (doc 26 §8 item 1). Note for that campaign:
any earlier picture of /14 made from a mixed arm should be redone from
`d27fresh`.

### 4.4 The doc 26 loop fix stands

The `examine_partial_identical_segments` fixed point (doc 26 §2) is a real
non-termination for any degenerate split point and the guard is still right.
What changes is the story of how PDVD reached it: through the artificial
two-pieces-plus-chords geometry of a stale input, not through a charge gap.

## 5. Fix

### 5.1 Process rule

Imaging, clustering (point tree) and PR must be produced under one wires
file. After any change to `params.files.wires` (or anything that changes face
ordering or wire positions), the chain is regenerated from imaging; a point
tree or imaging archive from before the change is not an input for a job
compiled after it.

### 5.2 Runner guards (wcp-porting-img, `pdvd/run_*_evt.sh`)

- `run_img_evt.sh` writes `work/<dir>/img-provenance.txt` with
  `wires=<file>` (read from the compiled config's `WireSchemaFile`) and the
  time.
- `run_clus_evt.sh` compares its compiled wires file with the imaging
  archives' `img-provenance.txt`: **ERROR (rc 3) on mismatch**, WARNING when
  the archives predate the provenance file; with `-save-pctree` it appends
  `wires=` and `img_wires=` to the `pctree-evt<N>.tlas` sidecar.
- `run_pr_evt.sh` compares its compiled wires file with the sidecar's
  `wires=`: **ERROR (rc 3) on mismatch** before wire-cell starts, WARNING for
  a pre-doc-27 sidecar. `PDVD_ALLOW_STALE_GEOMETRY=1` downgrades either ERROR
  to a WARNING (for controlled reproductions such as §6.2).

### 5.3 Toolkit diagnostic (log-only)

`clus/src/improvecluster_1.cxx`, in `ImproveCluster_1::mutate` per (anode,
face): if the original cluster's points on that face fall outside the face's
`sensitive()` bounding box in y or z (1 cm tolerance), a WARN names the
cluster, the anode/face, both ranges and the likely cause. No data path
changes (§6.3). It fires 67 times on the stale 039349/53 input, all on anodes
2, 3, 6, 7, and never on the regenerated inputs.

## 6. Verification

### 6.1 The two events, regenerated (tag `d27v7`, 2026-09-03 12:14)

| | stale `d25r13fix` | regenerated `d27v7` |
|---|---|---|
| 039349/53 track cluster: steiner pts on own charge at dy = 0 | 0 / 469 (469 / 469 at dy = +168.4) | 1457 / 1457 |
| steiner z coverage (20 cm bins, input : steiner) | 88–160 empty | every bin populated: 22:15, 184:234, 216:193, 240:250, 346:303, 244:274, 236:188 |
| PR segments crossing the gap | 3 chords (112–181 cm) | none (the cluster is not a ν candidate) |
| `degenerate split skipped` (doc 26 guard) | fires | 0 |
| PR wall time | 23 s (post doc 26) / 40+ min (pre) | 49 s |
| 039349/14 cluster 36: steiner on charge at dy = 0 | 426 / 632 | 1498 / 1726 (Bee 34, now z 5–233) |
| retile face mismatches in the PR log (probe lines) | 43 clusters shifted | 0 of 93 (evt 53), 0 of 77 (evt 14) face lines |

Every PR steiner cloud of both regenerated events sits on its own input
cluster at dy = 0 (`tag_check.py d27v7 53 -` / `14 -`).

### 6.2 Guards

- `run_pr_evt.sh -s d27guard` on the stale pctree with a `wires=…v6…` sidecar
  line: `ERROR … pctree was sampled with wires=protodunevd-wires-larsoft-v6.json.bz2
  but this PR job compiles wires=protodunevd-wires-larsoft-v7-uvwfit.json.bz2`,
  rc 3, no wire-cell process started.
- Same with `PDVD_ALLOW_STALE_GEOMETRY=1`: runs; the §5.3 WARN fires 67× (apa
  2: 16, apa 3: 15, apa 6: 21, apa 7: 15; both faces; none elsewhere).
- Regenerated runs: 0 WARN lines.

### 6.3 Byte identity of the toolkit change

Same stale input, same config: round-2 library (`/home/xqian/tmp/doc25r13/lib_post2`,
arm `d25r13fix`) vs WARN library (`/home/xqian/tmp/doc27/lib_warn`, arm
`d27guard` with the override): `mabc-pr.zip` member hashes identical
(`abtest/hash_archive.py`). `./build/clus/wcdoctest-clus`: SUCCESS.
uBooNE 35-event sweep `qlport/scripts/sweep/doc27base` (lib_post2) vs
`doc27warn` (lib_warn), `ab_check.sh doc27warn doc27base`: ZIPS 35/35
content-identical; TAGGER identical 34, diff 1 = event 5384-136-6805, the
documented layout-bistable event (doc 26 §7.4 showed both libraries produce
both of its states). SBND `work-{nuecc48,ncpi0}-doc25r14fix` (lib_post2) vs
`-doc27warn` (lib_warn), `pr85_hash_gate.py`: PASS 96/96 and 38/38 archives
byte-identical; the WARN never fires on SBND (0 lines in 67 logs).

### 6.4 The self-consistent 120-event arm `d27fresh`

`/home/xqian/tmp/doc27/fresh_pipeline.sh`: imaging (v7-uvwfit) → clustering
with `-save-pctree -calib` (`PDVD_LIGHT_SUFFIX=_keep`) → PR `-stm-fit`, for
every event of `stm/events.txt`, into `work/<run6>_<idx>_d27fresh/`. Each dir
carries `img-provenance.txt` and a sidecar with `wires=` and `img_wires=`.
Results (`stm/gates/r27_fresh_census.py d27fresh d25r13fix` →
`stm/gates/r27_census_d27fresh_vs_d25r13fix.tsv`; imaging 12:22–12:30,
clustering 12:30–12:34, PR 12:34–12:47 at 8+4+4 / 8+5+5 / 8+5+5 parallel jobs):

| | stale `d25r13fix` (v6 tree, v7 PR) | fresh `d27fresh` (v7 throughout) |
|---|---|---|
| events with PR outputs | 120 | 120 (sidecar `wires=` v7-uvwfit on all 120) |
| mains evaluated (STM verdict lines) | 5865 | 5543 |
| STM = 1 | 676 | 659 |
| TGM = true | 1164 | **1592** |
| ν candidates (`nu_per_bundle`) | 624 | 590 |
| retile face-mismatch WARN lines | (binary predates the WARN) | **0** |
| `degenerate split skipped` (doc 26 guard) | 3 (039349/14, /53) | **0** |
| total PR wall | 5571 s | 8488 s (max 744 s, 039252/5; no pass-budget WARN) |

The through-going-muon count rises by 37 %: on the swapped anodes the stale
retile had broken tracks into pieces that `check_tgm` could not follow to two
boundaries, and the same breakage fed the STM and ν-candidate lists. The fresh
arm is slower because its steiner clouds are complete (3–4× the points). The
doc 26 guard never fires on consistent inputs — the fixed point needed the
artificial two-pieces-plus-chords geometry.

## 7. Next steps

1. **Use `d27fresh` as the baseline** for everything in the doc 25 campaign
   that was measured on `d25r13*`/`d25r14*` arms (STM census, dQ/dx samples,
   Michel candidates): those arms are geometry-mixed on half the detector.
   Re-derive the numbers doc 25 §13 quotes from `d27fresh` before quoting them
   again.
2. **Steiner-terminals campaign (owner's item 1):** start from 039349/14 in
   `d27fresh`, where the mid-track hole (§4.3) is now visible on a consistent
   input.
3. **Over-clustering round (owner's item 2): closed** — no case. If a real
   two-piece cluster turns up in `d27fresh`, reopen with doc 26 §8.1's plan.
4. Consider recording the wires file inside the point-tree tarball itself
   (a `cluster_scalar`/metadata entry) so the guard does not depend on the
   sidecar; the runner guard is enough for the PDVD scripts but a LArSoft
   integration will not have the sidecar.

## 8. Retire round (2026-09-03, owner's instruction)

The owner asked to retire the products associated with the earlier geometry,
to remove the confusion and free disk. Census first (`pdvd/work`: 56.1 GB in
~1930 dirs), then one manifest-driven deletion:

- **Deleted (1257 dirs, 30.87 GB), listed with sizes and last-modified times in
  `stm/gates/r27_retire_v6_manifest.tsv`:** the eight 120-event PR arms run on
  v6 point trees (`d25r12eager`, `d25r12fast`, `d25r13base`, `d25r13fix`,
  `d25r14pre`, `d25r14cf`, `d25r14base2`, `d25r14fix2`) and their single-event
  probes (`d25r13chk*`, `d25r13dbg`, `d27probe`, `d27guard`); the Sep 2 Q/L +
  point-tree arms on v6 (`stm1` 120 events with its PR outputs, `stm2`, `stm3`,
  `stm4`, `stm4off`, `m1on`, `m1gate`, `thr*`, `e4`, `g1`, `dbg1`, `cens8`,
  `d25r11on/off`). No process had any of them open. Consequence for the
  record: the doc 25 §13 and doc 26 §5/§7.4 PDVD gates are no longer
  re-runnable from disk (their hash tables and census TSVs remain committed);
  their replacement is `d27fresh`.
- **Kept:** `d27fresh` (the baseline, 9.0 GB), `d27v7` (this document's
  two-event record), the light arms (`*_light*`: the light chain does not use
  the wires file), and the raw SP frames inside every `_keep` dir (the inputs
  `d27fresh` was imaged from, by symlink).
- **Stamped, not deleted:** the 120 `_keep` dirs' July imaging archives and
  Q/L outputs (v6; 7.3 GB including the frames). Each now carries an
  `img-provenance.txt` saying `wires=protodunevd-wires-larsoft-v6.json.bz2`
  (retroactive: `params.jsonnet` held v6 from 07-13 14:43 to 09-03 07:51), so
  `run_clus_evt.sh` **refuses** them under today's anodes (tested: rc 3 on a
  `_keep`-seeded tag). `scripts/stage_ql_tag.sh` now seeds from `d27fresh` by
  default and carries the provenance file along. These, and the July Q/L
  calibration arms (`nm4b`, `ccprod`, `cctt`, `tm0*`, `ac*`, `cathxa`, `rc14`,
  `cc3a`, `magnify`, the untagged dirs, ~6 GB), are the hand-scan records of
  the Q/L rounds (docs 19–24; `work/ql_labels`, `ql_scores` key on their calib
  dumps), so deleting them is the owner's call, not mine — see the report.
- **Not touched:** the peer session's same-day probes (`v7img`, `v7chk`,
  `v7test`, `v450`, `d31phase`, `stm1pf`, `r10*`, `b12*`, `dqdxdbg*`,
  `pr67probe*`, `currptsprobe`, `stmrcidfix`, `drainprobe`, `stageprobe`,
  `keepall`, `fixon*`, `fixoff*`, `noexcl*`; < 0.6 GB together).

Disk after the round: 543 GB free on the volume (was ~512 GB).

## Milestone log

- **2026-09-03** — investigation opened per doc 26 §8 (over-clustering on
  039349/53). Census: the PR cluster's steiner cloud had 0 points on any input
  point and all 469 on input cluster 60 after a +168.4 cm y shift; the input
  cluster is one continuous track. Probe run (`d27probe`, temporary WARNs in
  `ImproveCluster_1`/`CreateSteinerGraph`) named the mechanism: v6 vs
  v7-uvwfit face ordering on anodes 2, 3, 6, 7; point tree from 09-02 11:50,
  PR arms after the 09-03 07:51 wires change. Regenerated both events
  (`d27v7`): clouds on charge, no hole on /53, guard silent, 49 s. Shipped
  runner provenance guards + the log-only WARN; `d27fresh` 120-event
  regeneration and the uBooNE/SBND gates of the WARN launched.
- **2026-09-03 (later)** — gates of the WARN-only toolkit change: uBooNE 35/35
  zips, tagger 34/35 (6805, the bistable one); SBND 96/96 + 38/38; PDVD
  `mabc-pr.zip` identical on the stale input. `d27fresh` complete: 120/120,
  0 WARN, 0 guard fires, TGM 1164 → 1592, STM 676 → 659, ν candidates
  624 → 590 versus the stale arm (§6.4). Peer-session files
  (`pdvd/docs/nf_sp_img_clus/scripts/steiner_*.py`) noticed untouched.
- **2026-09-03 (retire round)** — 1257 v6-era dirs / 30.9 GB deleted per
  `stm/gates/r27_retire_v6_manifest.tsv`; `_keep` imaging stamped v6 so the
  guard refuses it; `stage_ql_tag.sh` seeds from `d27fresh`. Toolkit WARN
  committed as `9cbed05f`; doc 27 round as `2a8f91b9`.
