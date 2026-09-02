# ref/prod-2026-09-04 — SBND production operating point

**What moved from `prod-2026-09-03`, and on whose word.**

Owner, 2026-09-02, after scanning the doc-97 Bee A/B and adjudicating the three
flips that had looked adverse:
*"These are good idx 5-7 are all improvements. We can set this new running as
default on for SBND production."*

`sep_fv_point` now defaults **true** in
`cfg/pgrapher/experiment/sbnd/{clus,wct-clus-matching-perevt}.jsonnet`. The C++
defaults in `clus/src/clustering_separate.cxx` stay
`fv_inset_yz = 0 / far_point_x_cut = 140 cm / far_point_mid_dis = 25 cm /
dec1_guard_main_angle = -1`, so no other detector moves.

## The drift — four keys on two components, in the four SBND clustering jobs

```
DRIFT : prod.standalone, prod.wcls, sbnd_clus.json, sbnd_ql.json
  ADDED  ClusteringSeparate:apa0-0 .fv_inset_yz           = 150     (15 cm)
  ADDED  ClusteringSeparate:apa0-0 .far_point_x_cut       = 140     (14 cm)
  ADDED  ClusteringSeparate:apa0-0 .far_point_mid_dis     = 600     (60 cm)
  ADDED  ClusteringSeparate:apa0-0 .dec1_guard_main_angle = 45      (deg)
  ... and the same four on the second anode
      (apa1-0 in the three standalone jobs, apa1-1 in prod.wcls, whose
       per_volume() path clusters anode 1 on face 1)
```

The other **17 of 21** artifacts — uboone, the six pdhd jobs, the five pdvd
jobs, `sbnd_pr.json`, `sbnd_img.json`, `prod_prjob.json`, `bare_prjob.json` and
the sim checks — are byte-identical to `prod-2026-09-03`
(`sbnd_xin/docs/97_sep/97-flip-drift.txt` names every key). `prod-2026-09-03` is
left byte-untouched (M13); this is a new generation because the change is a
deliberate physics change and is **not** byte-identical.

`prod.wcls` drifting is the point, not a side effect: it is the LArSoft entry
point, and it inherits the flip through `clus_per_face`'s default the same way
`sep_vertex_veto` does.

## What the knob does

Puts the **per-APA `separate()` pass, and only that pass**, at the PDHD/PDVD
separation operating point. `JudgeSeparateDec_2` decides whether a cluster
reaches a detector surface by testing its convex-hull extremes against
`FV_* ± margin`; SBND's fiducial volume is inset from the physical wall by only
0.65–2.05 cm, so that test effectively asks for a point **on** the wall and a
cosmic stopping a few cm short contributes no surface contact. doc 96 §6.1
measured the consequence: `Dec_2` accepts **0 of 74** in-time clusters,
including 0 of the 33 longer than 250 cm.

PDHD and PDVD fix this by insetting `FV_ymin/ymax/zmin/zmax` in the
`DetectorVolumes` metadata (`clus/docs/clustering-separate-fv.md`). That block
is shared, via `select_scope_fv`, with `clustering_neutrino` and the containment
taggers. The new C++ knob `fv_inset_yz` applies the inset to the single
`ScopeFV` that `clustering_separate()` builds and to nothing else, which is what
makes it adoptable here.

## Validation behind the flip (doc 97 §5, §9)

| | |
|---|---|
| rescues | **both** doc-95 separation cases — 272-2-30 and 105-23-21, each TGM → nu-candidate |
| all 3067 SBND data events + 48 nueCC + 19 NCπ0 | 34799 bundles, **6 in-beam verdict flips**, 3 in-beam bundles appear / 3 disappear |
| the six flips | 290654, 318703, 95911 gain a candidate; 105074, 162363, 392901 lose or re-tag one — the owner adjudicated **all three of the latter as improvements** |
| in-beam populations | nu-candidate 1599 → 1597, TGM 509 → 508, STM 414 → 417, LM 54 → 54 |
| what the 41.6% blast radius is | of 1277 changed events only **10** show any in-beam movement; **38 of 40** in-beam mains that change keep their length to within 0.05 cm — re-partition at unchanged extent, not splitting or merging |
| baseline | a **fresh** knob-off arm `work-<s>-d97off2pr`, not `r3entry`: the 2026-08-25 stage-A root no longer reproduces on 3 of 3067 events (doc 97 §2.1) |
| production default == validated arm | 139 events (119 data + the 20-event MC debug group) re-run with **no flag**, byte-identical to `work-<s>-d97fv` on all 556 products — the manifest includes every one of the six flip events |
| shipped-fix sentinels | **30 of 31 PASS**; the one failure is mcp2k 105074, which is one of the three flips the owner adjudicated as an improvement |
| `wcdoctest-clus` | the knob's own 3 cases / 59 assertions pass (`-tc="clus sep fv inset*"`) |

## Reproducing the previous behaviour

```
run_ql_evt.sh -no-sep-fv-point          # or SBND_SEP_FV_POINT=0
```

That returns the clustering to `prod-2026-09-03`.

## Still absent, and must stay absent for now

`sep_track_recarve` — the other knob doc 97 shipped, still default **false**.
It rescues 272-2-30 only, it reaches **50 events this knob does not** (so the
two together are a configuration nobody has run), and its own two sentinel
failures (mcp2k 94392 pr/129, mcp2k 53793 doc-84-r2) are unadjudicated.
