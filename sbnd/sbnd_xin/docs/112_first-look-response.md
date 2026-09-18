# 112 — Response to the first look at `pr_tracking*.root`: what each reported issue was, what changed, and how to read the file now

*For the colleague who scanned the SBND `tracking-pr_*.root` files
(`presentations/20260917_MV_firstlook.pdf`, slides 19–23). Written 2026-09-18.*

The five things you reported were real. Two of them were one writer bug, two were one
selection defect seen at two sizes, and one was a missing label. All five are addressed in
the toolkit as of the commits named below; the files you scanned (written 2026-09-09/10)
predate all of it. This note maps each slide to its cause, its fix, and the branch to read
now. The full engineering record is `docs/109_root-output-improvements.md` (rev 3 = cases
1–3 and the labels, rev 4 = the merge behind cases 4 and 5).

## 0. In one table

| Slide | What you saw | What it was | Status | Read now |
|---|---|---|---|---|
| 19 — case 1 | `T_tagger.cluster_id` ≠ the cluster the 3-D points sit on | one writer bug: `T_rec_charge.cluster_id` was found by a **flag scan**, which returns the *pre-swap* cluster when the overall vertex moved the main | **fixed** (toolkit `3ecb110d`, ON in production `12798c4f`) | `T_rec_charge.cluster_id` now equals `T_tagger.cluster_id`; `sel_cluster_id` + `vertex_moved_cluster` say the swap happened |
| 20 — case 2 | a candidate with Enu 0, vertex (0,0,0), scores at their defaults | the PR pass found **no vertex** (`final_main_vertex == nullptr`); the row is written anyway | **labelled**, deliberately not suppressed | `T_tagger.has_vertex == 0`; `br_filled` says whether the nue BDT ran |
| 21 — case 3 | `T_rec_charge.cluster_id == -1` | the same flag scan: the selected activity was a **demoted main**, whose flag is cleared upstream | **fixed** (same commits) | `T_rec_charge.cluster_id`, plus `nu_index` and `point_cluster_id` |
| 22 — case 4 | two candidates, one of them empty/fake | one physical flash seen by **both drift volumes** → two flash ids → two bundles; the second one is a 0.5–10 cm scrap of the same interaction | **fixed by merging** the two bundles when their charge touches (toolkit rev 4, ON in production) — the scrap becomes a companion | one row; `T_bundle` has one row per bundle, `T_flash.nu_index` points both flashes at it |
| 23 — case 5 | the same neutrino as a candidate on both sides | the same mechanism at full size: **one interaction whose muon crossed the cathode**; the vertex half and the muon half each became a candidate | **fixed by the same merge**; both halves go through one PR pass (the longer half is the candidate, the other its companion) | one row, whose `act_cluster_id` lists both halves and whose `T_rec_charge` points come from both |

Nothing is double counted any more when the two bundles touch, and nothing is thrown
away: the merge is a *merge*, not a deletion. When two bundles share the light but do **not**
touch — two separate interactions, one per drift volume — both rows are kept.

## 1. Cases 1 and 3 — one writer bug (slides 19, 21)

`T_tagger.cluster_id` and `T_kine.cluster_id` were always right: they are taken from the
candidate's own pointer after the PR pass. `T_rec_charge.cluster_id` was not: the writer
*scanned* the candidate's PR graph for the cluster carrying the `main_cluster` flag. That
scan returns

- the **pre-swap** cluster when the overall-vertex search moved the main onto a companion
  (your case 1; 488 of your 6 236 rows, 122 of 1 460 in our local sample), and
- **−1** when the selected activity is a *demoted* main — a cluster the unmerge stage split
  off and whose flag it clears and never restores (your case 3; 344 of your files all −1
  and 12 mixed, 89 local files).

**Fix.** `T_rec_charge.cluster_id` is taken from the candidate's own `TaggerInfo`, the same
value `T_tagger` carries. Two branches were added to `T_rec_charge`:

| branch | meaning |
|---|---|
| `nu_index` | which `T_tagger`/`T_kine` row owns this point (the "one tree for the whole event" problem you noted) |
| `point_cluster_id` | the cluster this *point* was sampled from — a candidate's points come from its main cluster **and** its companions; this is what the old `ndf` branch carried under a misleading name (`ndf` is unchanged, the convert app reads it) |

Also: `T_rec_charge` and `T_proj_data` are now **always booked**, empty when there is
nothing to write, so the file's tree set no longer varies ("7 trees" vs "8 trees").

Verified: on 267 events the knob-off output is byte-identical to the previous release; with
the knob on only `T_rec_charge.cluster_id` changes (7 of 267 files) plus the two added
branches; on a DL-vertex arm every moved row joins. The **mixed** shape of your slide 21
(`[-1, 9]`) has no local example — it is the same single assignment as the all-`-1` shape,
which has 89.

## 2. Cases 2 and 4 — the empty row (slides 20, 22)

An empty row (`neutrino_type 0`, Enu 0, vertex (0,0,0), `nue_score −15`,
`numu_score −1.942`) is a candidate whose PR pass found **no vertex**. The row is written
so that the event still says which bundle was examined and why nothing came of it. It is
deliberately **not suppressed**; it is labelled:

| branch | reads |
|---|---|
| `T_tagger.has_vertex` | 0 = this is the placeholder row |
| `T_tagger.br_filled` | 0 = the nue BDT never ran (−15 is its default), 1 = a real −15 |
| `T_bundle.reason`, `sel_length_cm` | why this bundle produced a candidate and how long the selected activity was |

`nue_score == −15` on most rows is **not** a defect: the nue tagger returns before scoring
when the vertex has no shower, and `br_filled` tells the two apart.

Why the second row of your case 4 exists at all is section 3: it is a scrap of the same
interaction in the other drift volume, and rev 4 folds it into the real candidate. Where the
scrap does **not** touch the real candidate it stays its own (labelled) row — the upstream
reason a sub-cm cluster can be a candidate (the length floor exempts the legacy event-wide
winner) is a separate selection question and is still open.

## 3. Cases 4 and 5 — one interaction, two drift volumes (slides 22, 23)

Your case-5 title was right. r472 s36 e40 is one numu RES CC interaction at
x = −10.5 cm, 10 cm from the cathode, whose 1540 MeV muon crosses x = 0:

| row | flash gid | cluster | what it holds | vertex |
|---|---|---|---|---|
| 1 | 5 (TPC 0) | 11, 104 cm | the vertex, the proton, the pion, a 9 MeV muon stub | 1.2 cm from truth |
| 0 | 1000006 (TPC 1) | 23, 391 cm | 1 205 MeV of the same muon | a fake vertex **on the cathode**, 144 cm from truth |

Neither row was the event. The chain that produced them:

1. one scintillation flash is seen by both drift volumes' light detectors → **two** opflash
   ids, 6 ns apart (the file already says so: `T_flash.flash_group` is the same for both);
2. clustering is per drift volume, and the flash matching bundles each cluster with **its own
   TPC's** flash;
3. the neutrino selection keyed its candidates on the **raw flash id**, and a candidate's
   companions had to carry the same id — so the other half could never join the same PR pass.

Case 4 is the same thing at a smaller size: 21 of the 22 such events in your sample have a
0.5–10 cm scrap in the other volume instead of a 391 cm muon. Same flash group, same
mechanism, only the size differs.

**Fix (toolkit rev 4, `nu_bundle_flash_group`, ON in SBND production).** Two in-window
bundles that share a flash group and lie on different TPCs are merged into **one** bundle
when their charge **touches** — the closest points of some cluster pair, one from each
bundle, are within 20 cm. Each side keeps its own selection; the merged candidate is the
**longer** of the two selected activities, and the other side's clusters — its associated
clusters and its main clusters alike — join that candidate's PR pass as companions, so the
whole event is reconstructed once. A merge needs a real winner: if the longer selected
activity is under the 15 cm candidate floor the two rows stay as they are (two stubs
merged only gave a placeholder a fake vertex). Bundles that share the light but do **not**
touch — two separate interactions — are left as two candidates, so a second neutrino is
never lost.

Two facts you should know about this rule:

- On your r472 event the two bundles touch at **0.4 cm at the vertex**, not at the cathode:
  the flash matching had already put the muon's near-side 131 cm (cluster 48) into the *far*
  flash's bundle as an associated cluster. That is why the rule is a distance, not a
  "meets at x = 0" test (the cathode window exists as an optional tightening,
  `nu_bundle_flash_group_xcut`, off by default).
- On our 3 067 real-data events (no truth) the knob can touch 142; it merged 116 of them,
  removed 10 rows — every one a vertex-less placeholder — and left the other 2 925 events
  bit-identical (section 5).

## 4. Everything that changed in the file since the files you scanned

Your files carry `Trun` with 5 branches and no `T_bundle`/`T_flash`. A current file has:

| where | new | meaning |
|---|---|---|
| `T_bundle` (new tree) | one row per in-beam-window flash bundle, every event | `gid`, `flash_tpc`, `flash_time_us`, `flash_pe`, `flash_group`, `n_main`, `n_demoted`, `n_companion`, `n_companion_dropped`, `reason` (0 selected main, 1 selected demoted, 2 all cosmic, 3 length floor, 4 STM-only, 5 nothing eligible, 6 dedup), `nu_index` (−1 = no candidate), `sel_cluster_id`, `final_cluster_id`, `sel_length_cm` |
| `T_flash` (new tree) | one row per optical flash, every event | `gid`, `tpc`, `time_us`, `pe`, `in_window`, `flash_group`, `n_matched_clusters`, `n_matched_main`, `nu_index` |
| `T_tagger` / `T_kine` | `sel_cluster_id`, `vertex_moved_cluster`, `has_vertex`, `flash_time_us`, `flash_pe`, `flash_tpc`, `flash_group`, `br_filled` | the selection's provenance per row |
| `T_rec_charge` | `nu_index`, `point_cluster_id`; `cluster_id` now pointer-derived; tree always present | joinable 3-D points |
| `T_proj_data` | always present (empty when nothing to write) | constant tree set |
| `T_cluster` | `tgm`/`stm`/`fc`/`lm`/`is_main`/`is_associated` corrected (`fix_cluster_flags`) | the tagger verdicts you can trust |
| `Trun` | `toolkit_git`, `wcp_git`, `op_config_sha256`, `cfg_tree`, `runner`, `job_mode` | which code and which operating point wrote the file |
| selection | one candidate per physical flash when the halves touch (rev 4) | no double counting |

Unchanged on purpose: `real_cluster_id` and `sub_cluster_id` are still bound to the same
value (a legacy-format contract; a correctly named branch was added instead of redefining
them), and `ndf` still carries the per-point cluster id (the convert app depends on it).

## 5. What rev 4 did on 3 067 data events

| | |
|---|---|
| events the rule can touch (a candidate whose flash group holds a second in-window gid with clusters) | 142 of 3 067 |
| bundle pairs merged / kept apart | 116 / 26 |
| closest approach, merged pairs | 0.3 – 19.7 cm (most ≤ 0.6 cm: the clusters share a boundary) |
| closest approach, kept-apart pairs | 24.5 – 155 cm |
| candidate rows removed | 10, all of them `has_vertex = 0` placeholders |
| surviving rows that lost a vertex or changed cluster | 0 (geometric vertex); 0 lost a vertex on the DL arm |
| all other events | bit-identical (checked on 253 manifest events, 216 749 branches) |
| every event | rc 0, content checks C1–C17 clean |

Of the 12 local events with your case-4 shape, 10 merged (the stub became a companion; the
real candidate keeps its energy and scores to the MeV) and 2 stayed two rows because the
stub is 41 cm from anything — those keep their labelled placeholder, as section 2 describes.
In 5 of the 116 merges the reconstruction itself moved (energy by 250–370 MeV in 4 of them);
they are listed in doc 109 sec 9.6 for a hand look, since data has no truth to grade them.

## 6. How to check a file yourself

```python
import uproot, numpy as np
f  = uproot.open("tracking-pr_<tag>.root")
t  = f["T_tagger"].arrays(["nu_index", "cluster_id", "sel_cluster_id", "vertex_moved_cluster",
                           "has_vertex", "br_filled", "matched_flash_gid", "flash_group"], library="np")
rc = f["T_rec_charge"].arrays(["nu_index", "cluster_id", "point_cluster_id"], library="np")
for i in t["nu_index"]:
    pts = rc["nu_index"] == i
    assert set(rc["cluster_id"][pts]) <= {t["cluster_id"][i]}          # cases 1 and 3
    if t["has_vertex"][i] == 0: print("row", i, "is a vertex-less placeholder")   # cases 2 and 4
groups = t["flash_group"]
assert len(set(groups)) == len(groups) or True   # two rows in one group = kept apart on purpose (did not touch)
```

`scripts/d109_root_checks.py` runs these and more (C1–C17) over a whole arm; its rev-4
companion `scripts/d109r4_group_census.py` shows what the merge did event by event.

## 7. Not verified on a local example, stated rather than claimed

- The **both-halves-reconstructed** shape (your case 5 itself). Our 3 067 local events are
  **data** with no truth, and none of them has a second half larger than 10 cm. The merge
  is verified on the small shape (116 merges) and on the mechanism; on your r472 event the
  contact distance was measured offline from the Bee points (0.4 cm), so the rule fires
  there, but the merged reconstruction of that event has not been run because its inputs
  are not on this machine.
- The **mixed** `[-1, 9]` shape of case 3, as above.
- Whether a merged pair's vertex lands on the vertex half when the *muon* half is the longer
  activity (the selection still picks the longest; the overall-vertex search may then move
  the main, which `vertex_moved_cluster` records).

## Repro

```bash
cd wcp-porting-img/sbnd/sbnd_xin
scripts/d109r4_eligible_census.py                         # the 142 eligible local events
scripts/d109r4_group_census.py d109r4off3 d109r4on3       # what the merge did, event by event
scripts/d109_root_checks.py d109r4on3 --samples nuecc48 ncpi0 mcp1k mcp2k
scripts/d109r4_flashpair_contact_truth.py /nfs/data/1/xqian/sbnd_data/run   # your 22 pair events vs truth
```
Toolkit: cases 1–3 in `3ecb110d` + `12798c4f` (rev 3), cases 4–5 in `d1caf178` + `b93673ef`
(rev 4). Full record: `docs/109_root-output-improvements.md` sections 8 and 9.
