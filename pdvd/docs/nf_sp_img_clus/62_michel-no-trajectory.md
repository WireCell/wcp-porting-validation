# 62 — T3: the Michel with no trajectory

**Status (2026-09-09 night). Two knobs shipped and flipped to PDVD
production: `stop_local_michel_pieces: true` (T3b) and
`michel_range_energy_guard: true` (T3c, 5 cm default). Together (`d62bc`)
they take the `michel_found` census from TP 118 / FP 39 / FN 34 to
TP 132 / FP 22 / FN 20 — F1 0.764 → 0.863, purity 0.752 → 0.857 — with
`is_stm` bit-identical on all 566 common items and scan-tagged michel
segments carrying role 3 up from 189 to 225. The third knob,
`stop_local_residual_cm` (T3a, the stop-anchored `pr54` keep), is measured
and left OFF: it buys 2 Michels for 5 spurious ones and flips `is_stm` on 3
items by changing the PR graph. Both byte-identical gates PASS (PDVD
578/578, PDHD 325/325). Doc 54 §2.5's 27 no-drop lumps are answered: 18 are
single-terminal fragmentation, at most 8 are the "already covered" case.**

Doc pdvd/56 §8's T3 row: the `pr54` isolated-residual drop discards every
residual in the Michel-size band, so add a default-OFF stop-local admission;
plus the detached scan-michel segments and doc 55 §15.1's range-energy gate
on `conn_type 2`. This round ships three default-OFF knobs and one log-only
forward, scores each on the frozen 569-item record against the current PDVD
production candidate `d61v`, and flips the confirmed subset.

Companion docs: pdvd/54 (the two mechanisms, §2 is this doc's starting
point), pdvd/55 (the scan; §15.1 is knob C), pdvd/56 (the task set), pdvd/61
(T2c, whose veto route knob C reuses), pdhd/13 (the detached Michel).

---

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
wcbuild && ./build/clus/wcdoctest-clus            # 357/357 (five new default pins)

cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img
W=$HOME/tmp/d62
# the probe (sec 1): pr54 drops vs the candidate stops, and sec 15.1 recomputed, on the baseline
python3 pdvd/docs/nf_sp_img_clus/scripts/d62_probe.py d61v $HOME/tmp/d61/prep_d61v

# the arms (binary pinned under $W/libpin; SURVEY on every arm, as on every scan arm)
SURVEY='survey_enable:true,survey_radius_cm:60.0,survey_max_len_cm:25.0'
R=pdvd/docs/nf_sp_img_clus/scripts/d53_run_arms.sh
ARM=d62vleg DET=pdvd SRC=d16vnu JOBS=4 PIN=$W/libpin PR_TLA="-S stm_michel_extra={$SURVEY}" $R
ARM=d62hleg DET=pdhd SRC=d16hnu JOBS=4 PIN=$W/libpin PR_TLA="-S stm_michel_extra={$SURVEY}" $R
ARM=d62ab   DET=pdvd SRC=d16vnu JOBS=4 PIN=$W/libpin PR_TLA="-S stm_michel_extra={$SURVEY,stop_local_residual_cm:20.0,stop_local_michel_pieces:true}" $R
ARM=d62c    DET=pdvd SRC=d16vnu JOBS=4 PIN=$W/libpin PR_TLA="-S stm_michel_extra={$SURVEY,michel_range_energy_guard:true}" $R
ARM=d62c3   DET=pdvd SRC=d16vnu JOBS=4 PIN=$W/libpin PR_TLA="-S stm_michel_extra={$SURVEY,michel_range_energy_guard:true,michel_range_energy_dis_cm:3.0}" $R
ARM=d62v    DET=pdvd SRC=d16vnu JOBS=4 PIN=$W/libpin PR_TLA="-S stm_michel_extra={$SURVEY,stop_local_residual_cm:20.0,stop_local_michel_pieces:true,michel_range_energy_guard:true}" $R
ARM=d62fos  DET=pdvd SRC=d16vnu JOBS=4 PIN=$W/libpin PR_TLA="-S stm_michel_extra={$SURVEY,stop_local_residual_cm:20.0,traj_cover_probe:true}" $R
# two attribution arms run once the first seven were read (sec 4.2 / 4.5):
ARM=d62b    DET=pdvd SRC=d16vnu JOBS=10 PIN=$W/libpin PR_TLA="-S stm_michel_extra={$SURVEY,stop_local_michel_pieces:true}" $R
ARM=d62bc   DET=pdvd SRC=d16vnu JOBS=10 PIN=$W/libpin PR_TLA="-S stm_michel_extra={$SURVEY,stop_local_michel_pieces:true,michel_range_energy_guard:true}" $R   # = the production bag

# byte-identical gates: d62vleg vs d61v (PDVD), d62hleg vs d53h (PDHD)
python3 pdvd/docs/nf_sp_img_clus/scripts/d51g_branch_census.py \
    --before 'pdvd/work/*_d61v' --after 'pdvd/work/*_d62vleg' --before-arm d61v --after-arm d62vleg --pts --out $W/g1
python3 pdvd/docs/nf_sp_img_clus/scripts/d51g_branch_census.py \
    --before 'pdhd/work/*_d53h' --after 'pdhd/work/*_d62hleg' --before-arm d53h --after-arm d62hleg --pts --out $W/g2

# the feature arms, scored against the frozen record (--pin-tranche is REQUIRED once labels exist)
cd pdhd/stm_michel_scan
for a in d62ab d62b d62c d62c3 d62v d62bc; do
  ./prep_stm_michel_scan.py --det pdvd --arm $a --outdir $W/prep_$a --sheetdir $W/sheet_$a \
      --pin-tranche ../../pdvd/docs/scan/pdvd_stm_michel_scan_sheet.tsv
  python3 census_score.py --prep $W/prep_$a --baseline $HOME/tmp/d61/prep_d61v --arm $a --json $W/$a.json
  python3 ../../pdvd/docs/nf_sp_img_clus/scripts/d62_by_name.py $HOME/tmp/d61/prep_d61v $W/prep_$a          # sets by NAME, counters, C2 by name
done
python3 census_score.py --check                                          # still "0 of 14 differ"
# doc 54 sec 2.5's 27 no-drop lumps, read off the fos arm
python3 ../../pdvd/docs/nf_sp_img_clus/scripts/d62_fos.py d62fos $HOME/tmp/d61/prep_d61v d61v
```

---

## 1. What T3 is, and four corrections to how doc 56 framed it

1. **The `pr54` drop is not a separate stage.** `find_other_segments`
   (`NeutrinoOtherSegments.cxx:48`) has one caller, `find_proto_vertex`
   (`NeutrinoPatternBase.cxx:2919`), and `CheckSTM_Michel` calls
   `find_proto_vertex` itself on the main cluster and on every admitted
   companion (`CheckSTM_Michel.cxx:1309/1321/1324`). The four floors reach it
   through `stm_michel_partition_keys` (`pr.jsonnet`) and
   `apply_pattern_knobs` (`:776-779`): PDVD runs `keep_isolated=true`,
   `min_points 25`, `min_length 3 cm`, `len_admit 30 cm`. The drop happens
   BEFORE the stop vertex exists — but the TAGGER's stop is already known
   (`read_stm_anchor`, `:905-906`, runs at `:1234`; the partition at
   `:1305`). That is the hook knob A uses.
2. **There is no coded "2–24-point band"** — it is the observed maximum of
   the dropped population under those floors (doc 54 §2.3). `d53v`, `d59v`
   and `d61v` each log 363 drops and 37–38 keeps (the drop line is
   DEBUG-level; the runner's file sink is `debug`, which is the only reason
   the arms carry it — nothing persists it).
3. **Keeping the residual alone cannot set `michel_found`.** A kept residual
   becomes a DISCONNECTED piece of the main cluster's graph
   (`add_segment(graph, new_seg, v1, v2)` + a joint `do_multi_tracking`
   refit). The attached Michel path needs an arm at `stop_v`; the "dots"
   loop iterates companion CLUSTERS only (`oc != main`, `:1869-1998`). So a
   second, same-cluster admission (knob B) is needed for A to do anything
   for the Michel.
4. **Doc pdhd/13 §8's "held fix" is partly discharged**: `companion_max_len_cm`
   shipped at 25 and `dot_max_len_cm` went 10→25 (doc pdhd/15);
   `michel_dot_radius_drift_cm`, `michel_dot_delay_only`, the `michel_any`
   branch and the admission-rejection log line were never implemented and
   stay out of this round (§7).

Also relevant, measured on `d61v`: `michel_conn_type` histogram {0: 400,
1: 121, 2: 45} — no `conn_type 3` (charge-only) item exists on the census.

## 2. The probe (before any C++)

`d62_probe.py` over the `d61v` logs and payloads (`$W/probe_d61v.txt`):

| quantity | value |
|---|---|
| `pr54 isolated-residual drop` lines, `d61v` | 363 (76 of 120 events) |
| of those, in a scan-candidate cluster | 345 |
| within 20 cm of the candidate's OWN stop | **74** |
| in a non-candidate cluster, within 20 cm of some candidate's stop | 2 (of 18) |
| candidates with ≥ 1 such drop | **39** |

By scan verdict × `michel_found` (the 39): `STM_MICHEL` found 8 / missed
**5**; `STM_ONLY` missed 4; `THRU` found 1 / missed **13**; `MESSY` 8. So the
population knob A+B can act on is **5 recoverable class-D items**
(`039349_30/45`, `039349_31/51`, `039349_36/63`, `039349_64/52`,
`039349_64/65` — the third is the T2c casualty of doc 61) against **14
through-going risks**, before the radius / length / body gates of the piece
admission are applied. Class D total on `d61v` is 34, so the recoverable
ceiling for this mechanism is 5 of 34.

Doc 55 §15.1 recomputed on `d61v` (judged items, `michel_found`):

| cut | TP | FP | FN | purity | eff | F1 |
|---|---:|---:|---:|---:|---:|---:|
| as shipped (`d61v`) | 118 | 39 | 34 | 0.752 | 0.776 | 0.764 |
| drop `conn 2/3`, `dis > 5`, `KE < 10` | 115 | 18 | 37 | 0.865 | 0.757 | **0.807** |
| drop `conn 2/3`, `dis > 3`, `KE < 10` | 115 | 13 | 37 | 0.898 | 0.757 | **0.821** |

Both rows lose the SAME three scan-michel items (`039252_14/81` dis 13.5 KE
1.5; `039253_13/73` dis 14.4 KE 5.8; `039349_44/28` dis 14.9 KE 1.0 — all
far beyond either distance, all under 6 MeV); the 3 cm row removes five more
non-michel items (`039253_4/91`, `039349_0/28`, `039349_3/26`,
`039349_34/48`, `039349_7/57`, at 3.2–4.8 cm) at no additional TP cost on
this record. **The C++ default is 5 cm**, the physically argued distance
("an electron born at the stop cannot travel 5 cm through liquid argon and
deposit under 10 MeV", doc 55 §15.1); 3 cm was that section's best-F1 row
chosen on this same record, so it is measured here (`d62c3`) and offered
as the owner's option rather than shipped as the default.

## 3. Design

Four changes, all default-OFF, all through the opaque `stm_michel_knobs` /
`stm_michel_extra` bag (no `clus.jsonnet` / `pr.jsonnet` change; SBND does
not bind `CheckSTM_Michel`).

**A — `stop_local_residual_cm`** (double, 0 = off; arm: 20, doc 54 §2.6's own
number). `PatternAlgorithms` gains `m_other_seg_keep_anchor_cm`,
`m_other_seg_keep_anchors` and a fire counter (`NeutrinoPatternBase.h`,
next to the `other_seg_keep_isolated*` floors). At the drop site
(`NeutrinoOtherSegments.cxx`) the floors are evaluated first, unchanged; only
when they refuse and the anchor radius is set is the nearer fitted endpoint's
distance to the anchors computed, and `near_anchor` keeps the residual
through the existing keep path (same `add_segment` + joint refit), logging
`pr54 keep-isolated near-anchor: …`. `CheckSTM_Michel` sets the tagger's stop
as the single anchor after `apply_pattern_knobs` and reads the fire count
back after the main partition (`n_kept_near_stop_main`) and after the
companion loop (`n_kept_near_stop_comp`) — the anchors reach the companions'
partitions too, which is desirable (a companion whose only fit was dropped
reports `segs.empty()` → charge-only today). `TaggerCheckNeutrino` never sets
the new members, so SBND/uBooNE stay on the unchanged code path; the free
function `other_seg_keep_isolated_ok` is untouched.

**B — `stop_local_michel_pieces`** (bool). In the pieces pass, before the sort,
every fitted segment of the main cluster that is not in the chain, not
claimed, and DISCONNECTED from the chain — neither endpoint vertex is a
chain vertex and neither has an out-edge to a chain segment, which is
exactly what a `pr54`-kept piece looks like and which an attached interior
arm (T6's population) fails by construction — gets the three gates a
companion piece gets (`d_stop <= michel_dot_radius_cm`, length `<=
dot_max_len_cm`, the `dot_body_exclusion_cm` body test) and joins `pieces`
with the main cluster's id. Everything downstream (sort, `n_dots`, role 3,
`conn_type 2`, shower assembly, `michel_ke_best`) is unchanged code.
Counter `n_local_pieces`.

**C — `michel_range_energy_guard`** (bool) with `michel_range_energy_dis_cm`
5.0 / `michel_range_energy_ke_min` 10.0. Right after the T2c veto and before
`michel_found` is derived: a `conn_type 2/3` object with `michel_dis_cm >
dis` and `michel_ke_best < ke` is demoted via `michel_conn_type = 0` (never
`reject_bits`, so `is_stm` is untouched by construction; role-3 point rows
stay, as for T2c). Counter `n_michel_range_veto`.

**D — `traj_cover_probe` forwarded** into the partition (`pattern_knob_keys`
+ `apply_pattern_knobs`): log-only, so doc 54 §2.5's "27 lumps with no drop
line" can be read on an arm (§5).

All four counters are persisted to `T_stm_michel` and added to
`prep_stm_michel_scan.py`'s `VERDICT_SCALARS`; the five defaults are pinned in
`doctest_check_stm_michel_defaults.cxx`.

## 4. Results

Baseline `d61v` (547 scored, 3 unmatched). Every number below is from the
real arm's payloads, matched by name; `is_stm` is checked on every common
item, not assumed.

### 4.1 The `michel_found` census, arm by arm

| arm | knobs | TP | FP | FN | purity | eff | **F1** | `is_stm` flips |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `d61v` | baseline | 118 | 39 | 34 | 0.752 | 0.776 | 0.764 | — |
| `d62b` | B | **135** | 43 | 17 | 0.758 | 0.888 | 0.818 | **0 of 566** |
| `d62ab` | A+B | 136 | 48 | 16 | 0.739 | 0.895 | 0.810 | 3 (see 4.3) |
| `d62c` | C (5 cm) | 115 | **18** | 37 | 0.865 | 0.757 | 0.807 | **0 of 566** |
| `d62c3` | C (3 cm) | 115 | 13 | 37 | 0.898 | 0.757 | 0.821 | 0 of 566 |
| `d62v` | A+B+C | 133 | 26 | 19 | 0.836 | 0.875 | 0.855 | 3 (see 4.3) |
| `d62bc` | **B+C — the production candidate** | (§4.5) | | | | | | |

### 4.2 B — the same-cluster pieces: 17 Michels found, 0 lost, 4 spurious

`stop_local_michel_pieces` alone admits **70 pieces on 27 items**. Every
one of the 17 new `michel_found` TPs is a scan `STM_MICHEL` item whose
Michel was fitted by the PR but disconnected from the chain — 14 of them
without any `pr54` keep involved at all (`n_kept_near_stop_* = 0`, i.e. the
piece was already in the graph and the dots loop simply never looked at
the main cluster):

`039252_0/75`, `039252_16/98`, `039253_0/102`, `039253_0/110`,
`039253_13/39`, `039253_3/29`, `039253_3/61`, `039253_3/79`, `039349_22/63`,
`039349_23/54`, `039349_31/51`, `039349_32/63`, `039349_36/63`, `039349_5/64`,
`039349_5/65`, `039349_7/4`, `039349_82/54`

— seven of which are doc 56 §3's own "passes everything / `mip_lo` fails"
arms (`039349_32/63`, `039349_7/4`, `039349_82/54`, `039253_3/61`,
`039252_16/98`, `039349_22/63`, `039252_0/75`): doc 61's T2b concluded they
"hang off no alternate vertex" — correct, they hang off NO vertex of the
chain at all, they are disconnected pieces, and this is the mechanism T2b
could not see. `039253_0/110` and `039253_13/39` are two of T5's five class-F
items (their Bragg stub is also fitted separately; here their Michel is
found). `039349_36/63` is T2c's named casualty, back as a TP by a different
route (bridged, 17.5 MeV).

Cost: 4 new FPs, all bridged at 0.2–1.9 cm from the stop — `039349_20/32`
(STM_ONLY, 0.97 MeV), `039349_62/63` (STM_ONLY, 15.0 MeV), `039349_76/23`
(THRU, 0.67 MeV), `039349_78/22` (THRU, 21.6 MeV). Two of the four are under
1 MeV, but a KE floor on bridged pieces is doc 55 §15.1's territory (knob
C), which at 5 cm does not reach them; a floor at `dis` this small would be
a new operating point chosen on this sample and is not taken.

`is_stm` is bit-identical on all 566 common items — B touches nothing
upstream of the verdict. Michel attachment (C2, scan-tagged michel segments
matched by geometry): role 3 **189 → 225**, no role **83 → 47**, swallowed
9 → 9, survey 8 → 8. Class D (michel FN) 34 → 17; class C (michel FP) 39 → 43.

### 4.3 A — the stop-anchored residual keep: a measured mixed result, NOT flipped

`stop_local_residual_cm: 20` fires **46 times on 40 items** (main cluster;
1 companion fire on 1 item). On top of B it adds exactly two Michels
(`039349_30/45`, `039349_64/65`, both class-D items the probe named) — and
costs: one Michel TP lost (`039349_61/21`: the kept residual near the stop
changes the graph, `stop_v` no longer snaps (`R_STOP_UNMATCHED`, `stop_dis`
1.1 → 6.5 cm), the chain is walked greedily and its Michel arm 21014 is
swallowed into role 1), **five extra michel FPs** (`039253_0/44`,
`039253_15/36`, `039349_16/56`, `039349_20/73`, `039349_23/43`) and **three
`is_stm` flips**, two of them on items where only A fired:

| item | scan | `is_stm` | what changed |
|---|---|---|---|
| `039252_12/114` | THRU | 0 → 1 (**new FP**) | chain 2 → 1 segments, plateau 14.3k → 38.6k e/cm (`plateau_off_mip` gone), contrast 0.94 → 1.17 |
| `039252_8/93` | THRU | 1 → 0 (FP removed) | chain 3 → 1 segments, tail 46.7k → 36.1k, `no_bragg`+`shape_flat` |
| `039253_0/44` | STM_ONLY | 0 → 1 (new TP) | a local piece admitted, KS margin moved 0.062/0.050 → 0.047/0.090 |

So A is not "charge sharing in the joint refit" only — a kept residual
changes the PR graph the later stages build (segment count, vertex
positions, the stop snap), and on the ~40 items where it fires the verdict
can move either way. Net: `is_stm` TP +1, FP count unchanged with one THRU
swapped for another, `michel_found` +2 / −1 TP and +5 FP. That is not the
bar every prior flip met (0 new `is_stm` FP), and the two Michels it adds
are bought at a worse purity than B's own. *Update (doc pdvd/86, 2026-09-11): `039253_0/44`, one of the five extra FPs above, is an owner-confirmed Michel (smx3, and again smx5: a blob at the stop PR drops as an isolated residual). Re-counted on today's production, the 20 cm keep touches 31 items (14 THRU); with a size floor (≥ 5 points, ≥ 5 cm) at 5 cm it touches 5 (2 owner Michels, 0 THRU). The floor is doc 78 action item 7.* **A stays OFF**; it is measured,
named, and left for the owner (the 5 recoverable class-D items the probe
named split 3 to B alone, 2 to A+B).

### 4.4 C — doc 55 §15.1 as a knob: purity 0.752 → 0.865 at the argued 5 cm

`michel_range_energy_guard` fires **27 times** at 5 cm (`d62c`), on exactly
the items the probe predicted: 21 FPs removed (13 THRU, 8 STM_ONLY, all
5.8–14.6 cm from the stop, all under 7.2 MeV) and **3 TPs lost** —
`039252_14/81` (13.5 cm, 1.5 MeV), `039253_13/73` (14.4 cm, 5.8 MeV),
`039349_44/28` (14.9 cm, 1.0 MeV). All three sit 13–15 cm from the stop
carrying 1–6 MeV: a Michel electron born at the stop cannot do that, so on
these three the scan's `STM_MICHEL` and the chain's bridged object disagree
about which piece of charge is the Michel; they are named here, not tuned
around. `is_stm` bit-identical (0 of 566). At 3 cm (`d62c3`) five more
non-michel items go (`039253_4/91`, `039349_0/28`, `039349_3/26`,
`039349_34/48`, `039349_7/57`, 3.2–4.8 cm) at no further TP cost, F1
0.821 — that is doc 55's row and it stays the owner's option: the C++
default is the physically argued 5 cm, and production leaves both
thresholds unset.

### 4.5 B+C — the production candidate (`d62bc`)

| | TP | FP | FN | TN | purity | eff | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `d61v` | 118 | 39 | 34 | 356 | 0.752 | 0.776 | 0.764 |
| **`d62bc`** | **132** | **22** | **20** | 373 | **0.857** | **0.868** | **0.863** |

`is_stm` bit-identical (0 of 566; census 149 / 9 / 119 / 270 unchanged).
15 Michels found (`039252_0/75`, `039252_16/98`, `039253_0/102`,
`039253_0/110`, `039253_13/39`, `039253_3/29`, `039253_3/79`, `039349_22/63`,
`039349_23/54`, `039349_31/51`, `039349_36/63`, `039349_5/64`, `039349_5/65`,
`039349_7/4`, `039349_82/54`), 1 lost (`039349_44/28`, C's 14.9 cm / 1.0 MeV
casualty; C's other two casualties, `039252_14/81` and `039253_13/73`, are
rescued by B, which finds their Michel as a near piece at 1.8 / 2.2 cm
before C looks at the far one), 21 FPs removed by C, 4 added by B (§4.2),
and two of B's own recoveries vetoed by C (`039253_3/61` at 7.3 cm / 9.3
MeV, `039349_32/63` at 5.5 cm / 4.0 MeV — both scan `STM_MICHEL`, both
inside C's kinematic exclusion; named, not tuned around). Michel attachment
role 3 189 → 225, no role 83 → 47. Class D 34 → 20, class C 39 → 22, class E
(range-energy-impossible objects still carrying `michel_found`) 32 → 5.
Classes A/B/F–L unchanged (A 9, B 119, F 5, G 15, H 20, K 36, L 49/29).
Pin residual unchanged (5.27 cm median, 3 within 2 cm — no stop moves).

This is the bag now in `pdvd/wct-pr-perevt.jsonnet` and the baseline for
the next round. The T5-shaped residue: class D's remaining 20 are mostly
items whose Michel the PR never fitted at all (no piece to admit) or whose
stop is wrong; the two C casualties above are the price of a physical cut
on a record that tags by eye.

## 5. The 27 lumps with no drop line (the fos arm)

Doc 54 §2.5 found 44 unfitted in-bundle lumps within 20 cm of a stop
(≥ 20 points, ≥ 2×10⁵ e⁻) on `d53v`, only 17 containing a `pr54` drop, and
hypothesised — unverified — that the other 27 were consumed by step 1 of
`find_other_segments` ("already covered": 3-D within `search_range`, or all
three 2-D projections within 0.8 × it, or dead channels). With
`traj_cover_probe` now forwarded into `CheckSTM_Michel`'s partition, the
`d62fos` arm answers it directly. Lumps re-derived on the `d61v` payloads
(same recipe, `$W/d62_fos.py`): **44 lumps** —

| class | n |
|---|---:|
| a `pr54` drop inside the lump (baseline) | 18 |
| no drop, but a residual COMPONENT was proposed there and killed by a named filter (`step8 DROP`, `step9 …`) | **18** |
| no drop and no proposed component at all | **8** |

So doc 54's step-1 hypothesis can hold for at most 8 of the 27, not all of
them. For 18, `find_other_segments` DID build components over the lump and
rejected them downstream of the tagging: the verdicts on the 115 components
overlapping those 18 lumps are `step8 DROP single_point` **64** (a component
that is one Steiner terminal — the lump's terminals are fragmented into
singletons the Steiner MST never joins), `DROP nnf0_2d_shadowed` 13 (the
2-D-shadow case doc 54 named, but acting on a proposed component, not in
step 1), `DROP nnf0_short` 3, and 35 `KEEP`/`SELECTED` that fall inside the
5 cm-padded box without covering the lump (it stays unfitted in the
payload). So the second mechanism is **single-terminal fragmentation**, not
coverage tagging. The 8 with nothing are the genuine "already covered /
no terminal" class. `039349_81/54` (the 423-Steiner-point lump doc 54
singled out) has 95 proposed components, 34 kept and 27 `single_point`
drops: fragmented, not invisible. Counts only; no fix proposed here — the
lane is the Steiner terminal density near a stop (doc pdvd/31/37 machinery),
which is neither T3's nor any remaining doc-56 task's.

## 6. Gates

| gate | result |
|---|---|
| `./build/clus/wcdoctest-clus` | 357 / 357 (five new default pins in `doctest_check_stm_michel_defaults.cxx`) |
| `d62vleg` vs `d61v` (PDVD, `d51g_branch_census.py --pts`) | 578 / 578 candidates identical point geometry, 0 role labels moved, 0 `is_stm` flips; new branches only (`n_kept_near_stop_main/comp`, `n_local_pieces`, `n_michel_range_veto`) — **PASS** (`$W/gate1_pdvd.txt`) |
| `d62hleg` vs `d53h` (PDHD) | 325 / 325 identical, 0 moved, 0 flips — **PASS** (`$W/gate2_pdhd.txt`) |
| compiled-config proof | the five new keys appear in the compiled `CheckSTM_Michel` data only with the arm TLA; absent (C++ defaults) without it (`$W/cfg_on.json`, `cfg_off.json`) |
| binary pin | `libWireCellClus.so` md5 `3a7bd50cc01a` before and after every arm (`$W/libpin`) |
| `census_score.py --check` | still 0 of 14 differ |
| `is_stm` on `d62b`, `d62c`, `d62c3` | bit-identical, 0 of 566 common items |
| `is_stm` on `d62ab`, `d62v` | 3 flips, all named (§4.3) |
| flip-equivalence (compiled JSON: post-round production bag vs pre-round file + the `d62bc` override) | **0 lines** (`$W/flip_check.sh`, `fc_*.json`) |
| true OFF path (the same force-off override on the pre-round and the post-round file) | **0 lines** — no inert-but-present key; both thresholds left unset |
| SBND / uBooNE | no jsonnet of theirs touched; the two new `PatternAlgorithms` members are never set by `TaggerCheckNeutrino`, so their code path is unchanged by construction; `abtest/compile_all_cfg.sh` + `cmp_cfg.sh` before/after the flip: all 16 live jobs (incl. `sbnd_pr`, `sbnd_clus`) NORMDIFF 0, **OVERALL PASS**. No SBND/uBooNE production claim is made. |
| eight arms, one binary | `d62vleg d62hleg d62ab d62b d62c d62c3 d62v d62bc d62fos`, md5 `3a7bd50cc01a` before and after each |

## 8. Doc 56 update, and what this round leaves

Doc 56's T3 row is marked done with the three-way outcome (B and C
shipped and flipped, A measured and left OFF), §1 row 3 gets the
`michel_found` census 0.764 → 0.863, the Order paragraph moves to T5, and
§9 gains three items: doc 61's T2b conclusion re-read (§7.1), the
kept-residual graph change (§7.2), and the fos result on the 27 lumps
(§5). The production bag comment names the owner's two options: `A`
(`stop_local_residual_cm:20.0`) and the 3 cm variant of C.

Scripts committed with this doc: `scripts/d62_probe.py` (§2),
`scripts/d62_by_name.py` (the set-by-name diff with the T3 counters and
the C2-by-name census), `scripts/d62_fos.py` (§5).

## 6. Gates

(pending)

## 7. Found on the way, not fixed

1. **Doc 61's T2b conclusion was right for the wrong reason.** The seven
   "passes every gate, still not admitted" arms of doc 56 §3 hang off no
   alternate vertex because they hang off no chain vertex at all — they are
   disconnected pieces of the main cluster, which nothing in the Michel
   search looked at. B admits them. Doc 61 §3 should be read with this.
2. **A kept residual is a graph change, not a refit.** `stop_local_residual_cm`
   changes segment counts, vertex positions and the stop snap on the items
   it fires on (§4.3). Any future keep-more-residuals knob in this chain
   needs the same `is_stm`-by-name check, not a "disconnected piece, cannot
   matter" argument.
3. **B's four spurious pieces** are all within 2 cm of the stop and two are
   under 1 MeV; doc 55 §15.1's floor does not reach them by construction
   (`dis > 5`). A near-stop KE floor on bridged pieces would be a new
   operating point and is not proposed here.
4. **Single-terminal fragmentation** is the dominant reason 18 of the 27
   no-drop lumps never reach a fit (§5) — a Steiner-terminal-density
   question near the stop (doc pdvd/31/37), outside doc 56's task set.
5. **Doc pdhd/13 §8's leftovers** (`michel_dot_radius_drift_cm`,
   `michel_dot_delay_only`, the `michel_any` branch, the admission-rejection
   log line) remain unimplemented; B makes `michel_any` less pressing
   (bridged objects now set `michel_found` through the same path).
6. **Pin residual under a `d61v` baseline** reads 5.27 cm median / 3 within
   2 cm on every arm here (the pins are placed on the BASELINE chain, so this
   number is relative to `d61v`'s geometry and not comparable to docs
   57–59's 4.60→2.78 cm sequence, which used `d53v`/`d58v`). A+B moves one
   pin item out of the 2 cm band (`039349_61/21`, §4.3); B and C move none.
7. The compile-only staging dir `pdvd/work/039252_0_d62cfg` (symlinks only)
   was created for the compiled-config proof and is left in place.

## 8. Doc 56 update

(pending)
