# 66 — T8: the coiled and the unsupported fit, as fields the chain writes

**Status (2026-09-10, overnight). Writer fields shipped (eight
`T_stm_michel` branches, one `T_stm_michel_pts` column, unconditional);
the guard `profile_geometry_guard` built, scored and left OFF.** The chain's
`end_arc_span` reproduces the offline class G item for item (15/15 flagged,
551/551 not, |Δ| median 2e-4); `n_unsupported_segs > 0` agrees with class H
on 552 of 558 items, the 6 disagreements all from the two plateau
definitions (the chain's 20–40 cm live median vs the offline median of every
q > 0 row). The guard would veto 29 judged items, 6 of them `is_stm = 1`: **5
true stoppers and 1 false positive** — it fails the strict bar and stays OFF.
Both byte-identical gates PASS (PDVD 578/578 bit-identical on all 125 shared
branches, PDHD 325/325). This closes doc pdvd/56's task set: T1a, T1b, T1c,
T2, T3, T4, T5, T6, T7, T8 all executed.

Doc pdvd/56 §8's T8 row: publish `arc/span` over the last 20 cm and
`charge_supported` per segment as reject-class inputs (classes G 18 and
H 20 of doc 55 §17.1). This round writes them from the chain (unconditional
fields on `T_stm_michel`, a per-point `q_sup` column on `T_stm_michel_pts`),
checks them against the offline classes item by item, and scores the one
default-OFF guard that turns them into a reject bit.

Companion docs: pdvd/55 §17.1 items 4–5 (the classes), pdvd/56 §6 and §7,
pdvd/61 §9 item 4 (F/H tracked by name), pdvd/40 §12 (the unsupported-run
machinery upstream).

---

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
wcbuild && ./build/clus/wcdoctest-clus            # 357/357 (five new default pins)

cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img
W=$HOME/tmp/d66
SURVEY='survey_enable:true,survey_radius_cm:60.0,survey_max_len_cm:25.0'
R=pdvd/docs/nf_sp_img_clus/scripts/d53_run_arms.sh
ARM=d66vleg DET=pdvd SRC=d16vnu JOBS=7 PIN=$W/libpin PR_TLA="-S stm_michel_extra={$SURVEY}" $R
ARM=d66hleg DET=pdhd SRC=d16hnu JOBS=7 PIN=$W/libpin PR_TLA="-S stm_michel_extra={$SURVEY}" $R
ARM=d66v    DET=pdvd SRC=d16vnu JOBS=7 PIN=$W/libpin PR_TLA="-S stm_michel_extra={$SURVEY,profile_geometry_guard:true}" $R
python3 pdvd/docs/nf_sp_img_clus/scripts/d51g_branch_census.py \
    --before 'pdvd/work/*_d65vleg' --after 'pdvd/work/*_d66vleg' --before-arm d65vleg --after-arm d66vleg --pts --out $W/g1
python3 pdvd/docs/nf_sp_img_clus/scripts/d51g_branch_census.py \
    --before 'pdhd/work/*_d53h' --after 'pdhd/work/*_d66hleg' --before-arm d53h --after-arm d66hleg --pts --out $W/g2
cd pdhd/stm_michel_scan
for a in d66vleg d66v; do
  ./prep_stm_michel_scan.py --det pdvd --arm $a --outdir $W/prep_$a --sheetdir $W/sheet_$a \
      --pin-tranche ../../pdvd/docs/scan/pdvd_stm_michel_scan_sheet.tsv
done
python3 census_score.py --prep $W/prep_d66v --baseline $W/prep_d66vleg --arm d66v --json $W/d66v.json
python3 ../../pdvd/docs/nf_sp_img_clus/scripts/d63_by_name.py $W/prep_d66vleg $W/prep_d66v
python3 ../../pdvd/docs/nf_sp_img_clus/scripts/d66_geometry_fields.py $W/prep_d66vleg     # sec 2: C++ fields vs offline classes G/H
python3 census_score.py --check
```

---

## 1. What ships

**Fields on `T_stm_michel`** (unconditional, −1 = not computed): over the last
`end_window_cm` (20) of the geometric profile, `end_arc_cm`, `end_span_cm`
(all-pairs 3-D diameter), `end_arc_span`, `n_end_pts`; over every fitted
segment of the main cluster against the chain's plateau median
(`rec.bragg.plateau_med`): `n_unsupported_segs` (segments ≥
`unsupported_min_len_cm` 20 with median dQ/dx < `unsupported_frac` 0.25 of the
plateau), `unsupported_len_cm`, `unsupported_frac_min`, and
`chain_support_min` (the least-supported CHAIN segment's ratio).

**Column on `T_stm_michel_pts`**: `q_sup`, the owning segment's median dQ/dx
over the plateau median, per point, every role, unconditional (−1 without a
plateau). The offline `charge_supported = dqdx_med / plateau_med` of doc
55 §17.1 item 4, written by the chain.

**Knob `profile_geometry_guard`** (bool, default false): `end_arc_span >=
profile_arc_span_max` (1.5) or `n_unsupported_segs > 0` sets a new reject
bit `R_PROFILE_GEOMETRY = 1u << 13` ("profile_geometry" in the name table,
`census_lib.STM_BITS` and the prep's `BITS`). Every reject bit is a hard veto
of `is_stm`; the literals are doc 55 §17.1's own class definitions, chosen
on this record, so the flip bar is the STRICT one — 0 `is_stm` TPs lost, not
net F1.

## 2. Agreement with the offline classes (`d66vleg`, fields only)

| C++ field vs offline class | agree | C++ only | offline only |
|---|---:|---:|---:|
| `end_arc_span >= 1.5` vs class G | **566 / 566** (15 flagged, 551 not) | 0 | 0 |
| `n_unsupported_segs > 0` vs class H | 552 / 558 (17 flagged, 535 not) | 3 | 3 |

`end_arc_span` matches the offline arc/span to |Δ| median 0.0002, p90 0.0005,
max 0.018 over 564 items — the C++ uses the all-pairs diameter on every item
while the offline rule falls back to the endpoint chord above 60 points, and
no item in the census has more than 60 rows in its last 20 cm, so the two are
the same number. Class G is now a chain product.

The six H disagreements are all one thing: the chain's plateau is
`rec.bragg.plateau_med` (the live median over rr 20–40 cm), the offline
rule's is the median of every `q > 0` row of the whole profile; their ratio
is 1.05 median but 0.88 → 1.24 at p10 → p90, and on the six items it is far
outside that (`039252_13/74`: 40015 vs 5931 e/cm — the whole profile is
unsupported, so the offline "plateau" is itself the unsupported charge and the
offline class cannot fire; `039349_30/41`: 15535 vs 35442, the reverse). The
chain's definition is the one the verdict already uses (`R_PLATEAU_OFF_MIP`),
so `n_unsupported_segs` is "unsupported relative to the plateau the verdict
reads", which is the useful meaning; class H's 20 stays as the offline
descriptive count.

`q_sup` is present on every `stm_michel_pts` row of every carrier (13
columns; 0.4 % of rows carry −1, the items with no plateau), median 0.93 on
the muon chain.

## 3. The guard scored (`d66v` vs `d66vleg`)

`profile_geometry_guard: true` sets `R_PROFILE_GEOMETRY` on **29 of 547 judged
items**, of which 6 were `is_stm = 1`:

| item | scan | why |
|---|---|---|
| `039252_3/74` | STM_MICHEL | unsupported segment(s) — the same item that tops doc 56's "> 8 cm no-role segments" list; its chain is fine, its cluster carries long faint pieces |
| `039253_8/62` | STM_MICHEL | doc 54's own item: the unfitted lump's neighbourhood fits as an unsupported piece |
| `039349_37/39` | STM_MICHEL | bridged Michel at 0.4 cm, 29 MeV; an unsupported companion-side piece |
| `039253_17/127` | STM_ONLY | coiled end / unsupported |
| `039349_12/44` | STM_ONLY | idem |
| `039349_22/45` | THRU | **the one FP removed** (one of the 9 `is_stm` FPs since `d53v`) |

`is_stm` census 152 / 9 / 116 → **147 / 8 / 121** (purity 0.944 → 0.948,
efficiency 0.567 → 0.549, F1 0.709 → 0.695). `michel_found` identical; every
other class count identical. **The strict bar (0 `is_stm` TPs lost) fails
by five; the guard stays OFF.** The finding behind it is real and is what the
fields are for: a stopping muon's cluster can carry a long unsupported fitted
segment (a delta or a piece of nearby activity the fit spanned) without its
own chain being wrong — so "the cluster has an unsupported fit" is not "the
chain's profile is not a measurement". `chain_support_min` (the least
supported CHAIN segment) is the field a future guard should read; on the 5
lost TPs it is ≥ 0.6, on `039349_22/45` it is the chain itself that is faint.
Not tuned here.

## 4. Gates

| gate | result |
|---|---|
| `./build/clus/wcdoctest-clus` | 357 / 357 (five new default pins) |
| `d66vleg` vs `d65vleg` (PDVD, `--pts`) | 578 / 578 identical point geometry, **578 / 578 bit-identical on all 125 shared branches**, 0 role labels moved, 0 `is_stm` flips; 8 new branches (`end_arc_cm`, `end_span_cm`, `end_arc_span`, `n_end_pts`, `n_unsupported_segs`, `unsupported_len_cm`, `unsupported_frac_min`, `chain_support_min`) — **PASS** (`$W/gate1_pdvd.txt`) |
| `d66hleg` vs `d53h` (PDHD) | 325 / 325 identical, 0 moved, 0 flips — **PASS** (`$W/gate2_pdhd.txt`) |
| `q_sup` column | present on every `stm_michel_pts` carrier (unconditional emplacement; the branch census's pts compare reads `role/seg_id/x/y/z/q` and is unaffected) |
| `d66v` vs `d66vleg` | 6 `is_stm` flips, all `R_PROFILE_GEOMETRY`, named above; `michel_found` identical |
| binary pin | `libWireCellClus.so` md5 `c83d6227aca6` before and after every arm (incl. `d65v3`, doc 65 §5.1) |
| `census_score.py --check` | 0 of 14 differ (`profile_geometry` appended to `census_lib.STM_BITS` and the prep's `BITS` as bit 13; the committed payloads carry no such bit) |
| production config | untouched this round; no flip |

## 5. Found on the way, not fixed

1. **Two plateau definitions.** The verdict's `plateau_med` (live, 20–40 cm)
   and the offline census's `median(q > 0)` (whole profile) differ by 5 %
   typically and by a factor on items whose whole profile is faint (§2). The
   offline class H should be read with that in mind; not changed.
2. **The guard is the wrong shape** (§3): it vetoes on ANY unsupported fitted
   segment of the cluster; the chain's own support (`chain_support_min`) is
   the field that would separate "the profile is not a measurement" from "the
   cluster has junk". A `chain_support_min`-based guard is not built or
   scored here.
3. **Class G's 15 items** all have `end_arc_span` written now; only one of
   them is `is_stm = 1` and it is a true stopper — the coiled end is not what
   is limiting purity on this record.
4. **`n_end_pts` and the 20 cm window** are on the geometric profile (dead
   rows included), matching the offline rule; a live-only variant would need
   its own offline twin.

## 6. Doc 56 update, and the campaign close-out

T8 row marked done (fields shipped, guard OFF with the five named costs).
Doc 56's Order paragraph now closes the task set: **T1a, T1b, T1c, T2, T3, T4,
T5, T6, T7, T8 all executed** across docs 57–66. What the campaign leaves for
the owner, in order of evidence: (1) `ks_margin` −0.02 to −0.05 (doc 65 §2.1);
(2) `bragg_peak_anchor` at a 3 cm window (doc 65 §5.1, +16 net `is_stm` TP at
unchanged purity, a swapped 4/3 FP set); (3) `compare_range_cm` 45–60 (doc 65
§2.2); (4) `michel_range_energy_dis_cm` 3 (doc 62 §4.4); (5) a re-scan for
`dx_norm_length` 4 mm (doc 65 §4); (6) `stop_local_residual_cm` (doc 62 §4.3,
mixed); (7) `publish_other_arms` in production (doc 64 §5, rows only). Named
leads with no code yet: the 33 scan-michel `kOther` arms (doc 64 §3.3), the 3
class-F stubs over the 20° angle (doc 63 §3), single-terminal fragmentation
near the stop (doc 62 §5), a `chain_support_min` guard (§5.2 here).
