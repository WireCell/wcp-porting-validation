# 64 — T6: the interior arms, published as role 7

**Status (2026-09-10, overnight). Knob `publish_other_arms` shipped
(default OFF), rows-only: on the ON arm every verdict branch is
bit-identical to the leg (124/124 shared `T_stm_michel` branches, 0 `is_stm`
flips, role-1/2/3/5/6 point histograms identical) and 278 role-7 segments
(4960 points) appear on 166 items. Production stays OFF, following the
survey's precedent (diagnostic rows: off in production, on in the scan
arms' TLA) — the one judgment call of this campaign, flagged for the owner
in §5. A prep-script gap is fixed on the way: role-2 deltas had never
reached `chain_role`, so 206 items' published deltas read as "no role" in
every offline metric; C3 on the same arm drops 605 → 520 segments from that
fix alone. The interior-arm table (§3) resolves doc 56's 172 `delta / other`
tags into 172 companion survey segments, 80 deltas, 45 role-7 arms and 42
still-unnamed pieces, and finds 33 scan-tagged MICHEL segments among the
chain's `kOther` arms — the lead for a future admission round.**

Doc pdvd/56 §8's T6 row: classify the 172 no-role same-cluster segments the
scan tagged `delta / other` on stoppers; publish `role 7 = other` so display
and scan can see them; the over-clustering half goes upstream. This round
ships the role-7 knob (rows only, no verdict path), fixes a prep-script gap
that had hidden every published delta from the offline metrics, and
re-derives the interior-arm table against the scan's tags on the current
production candidate.

Companion docs: pdvd/56 §7–§8, pdvd/53 (the survey, role 6), pdhd/12 (the
display), pdvd/62 (T3b — the disconnected pieces that were part of the "no
role" set until this week).

---

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
wcbuild && ./build/clus/wcdoctest-clus            # 357/357 (one new default pin)

cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img
W=$HOME/tmp/d64
SURVEY='survey_enable:true,survey_radius_cm:60.0,survey_max_len_cm:25.0'
R=pdvd/docs/nf_sp_img_clus/scripts/d53_run_arms.sh
ARM=d64vleg DET=pdvd SRC=d16vnu JOBS=6 PIN=$W/libpin PR_TLA="-S stm_michel_extra={$SURVEY}" $R
ARM=d64hleg DET=pdhd SRC=d16hnu JOBS=6 PIN=$W/libpin PR_TLA="-S stm_michel_extra={$SURVEY}" $R
ARM=d64v    DET=pdvd SRC=d16vnu JOBS=6 PIN=$W/libpin PR_TLA="-S stm_michel_extra={$SURVEY,publish_other_arms:true}" $R

# byte-identical gates: d64vleg vs d63a (the round-2 production candidate), d64hleg vs d53h
python3 pdvd/docs/nf_sp_img_clus/scripts/d51g_branch_census.py \
    --before 'pdvd/work/*_d63a' --after 'pdvd/work/*_d64vleg' --before-arm d63a --after-arm d64vleg --pts --out $W/g1
python3 pdvd/docs/nf_sp_img_clus/scripts/d51g_branch_census.py \
    --before 'pdhd/work/*_d53h' --after 'pdhd/work/*_d64hleg' --before-arm d53h --after-arm d64hleg --pts --out $W/g2

# the role-7 arm: verdicts must be identical to the leg, role-3 rows identical, role-7 rows new
cd pdhd/stm_michel_scan
for a in d64vleg d64v; do
  ./prep_stm_michel_scan.py --det pdvd --arm $a --outdir $W/prep_$a --sheetdir $W/sheet_$a \
      --pin-tranche ../../pdvd/docs/scan/pdvd_stm_michel_scan_sheet.tsv
done
python3 census_score.py --prep $W/prep_d64vleg --baseline $HOME/tmp/d63/prep_d63a --arm d64vleg --json $W/d64vleg.json
python3 census_score.py --prep $W/prep_d64v    --baseline $HOME/tmp/d63/prep_d63a --arm d64v    --json $W/d64v.json
python3 ../../pdvd/docs/nf_sp_img_clus/scripts/d63_by_name.py $W/prep_d64vleg $W/prep_d64v      # 0 verdict differences expected
python3 ../../pdvd/docs/nf_sp_img_clus/scripts/d64_interior_arms.py $W/prep_d64v                # the T6 table (sec 3)
python3 census_score.py --check                                                                # still "0 of 14 differ"
# the prep correction (sec 1): the same baseline arm re-prepped with roles 2 and 7 in the role set
./prep_stm_michel_scan.py --det pdvd --arm d63a --outdir $W/prep_d63a_fix --sheetdir $W/sheet_d63a_fix \
    --pin-tranche ../../pdvd/docs/scan/pdvd_stm_michel_scan_sheet.tsv
python3 census_score.py --prep $W/prep_d63a_fix --baseline $HOME/tmp/d63/prep_d63a --arm d63a
```

---

## 1. Two corrections to the framing, one of them a prep-script fix

1. **Interior `kOther` arms were never dropped by C++ — only uncounted
   offline.** `stm_michel_classify_chain_arm` returns `kOther` for an
   interior arm longer than `delta_max_len_cm` (8) and cooler than
   `vertex_hadron_mip` (1.4), or a short one whose far vertex continues; the
   caller (`CheckSTM_Michel.cxx`, the interior loop) does `++rec.n_body_other`
   and that count has been persisted to `T_stm_michel` since doc pdvd/48. It
   never reached the scan payload: `n_body_other` was not in
   `prep_stm_michel_scan.py`'s `VERDICT_SCALARS`. A `kOther` arm at the STOP
   vertex, by contrast, really had nothing — no counter, no rows (the
   `for (auto a : stop_arms)` loop had branches for continuation and Michel
   only, including the continuation `michel_guards_stop` demotes to `kOther`).
2. **Role 2 never reached `chain_role`.** `prep_stm_michel_scan.py` built its
   `extra` segment set from roles `(3, 4, 5, 6)`, so every published delta
   (role 2, `add_points(rec, arm, 2)`) read as "no role" in every offline
   metric — `census_score.py`'s C3 "interior arms" and
   `d56_failure_mechanisms.py` §8's 172 alike. Measured on the committed
   `d53v` payloads: role 2 appears in 0 of 569 `chain_role` dicts while 211
   items carry `n_delta > 0`. So part of doc 56 §8's "152 attached to the
   chain interior" was `kDelta`, not `kOther`, and the viewer's
   `chain_group` branch for role 2 (`"delta / other"`) had never executed.
   Fixed this round: the role set is `(2, 3, 4, 5, 6, 7)`; §2 quotes what
   moves.

Also: 172 (T6's number, scan-tag-gated `delta / other`) and 664 (doc 56 §7's
C3, every same-cluster no-role segment on a stopper) are different
populations; the table in §3 is the tag-gated one.

## 2. What ships

**`publish_other_arms`** (bool, default false, `CheckSTM_Michel`). During
classification the `kOther` arms are only COLLECTED — interior (beside
`++n_body_other`) and stop (a new `++n_stop_other`) — and their role-7 rows
are emitted LAST, after the Michel object, the capture gamma and the survey
have claimed what they claim: an interior arm the Michel shower walk pulled
into the object keeps its role-3 row (the survey-gated shower extras skip
claimed segments, and `add_points` has no claimed check of its own), and
role 7 itself claims nothing (`add_points` exempts 6 and 7). Rows only —
no verdict reads them; `n_other_published` counts what was emitted.
Persisted: `n_stop_other`, `n_other_published` (new), `n_body_other`
(existing, now in `VERDICT_SCALARS`). Doctest pins the default.

**Prep / scorer / viewer.** Roles 2 and 7 in the prep's role set and the
per-role point bags (`v["delta"]` was already emitted; `v["other"]` is new);
`census_score.py`'s C2 names role 7; the viewer gains an `other` layer
(triangle, `#9467bd`) in every role-keyed list — `LAYERS`, `QSCAT`/`SRCQ`,
`MEAS_TRACKS`, the three `for nm in (...)` loops, `chain_group` (7 →
`"delta / other"`, the scanner's own group for it), `chain_note`, the
measurement-panel outline list. `PF_TAGS` (the scanner's frozen tag
palette) is untouched.

## 3. Results

### 3.1 The prep correction, on the same arm

Re-prepping the round-2 candidate `d63a` with roles 2 and 7 in the role set
(`prep_d63a_fix` vs `prep_d63a`): every class count identical; **206 items**
now carry a role-2 entry in `chain_role`; the C2 Michel-attachment row moves
"no role" 46 → 41 with the 5 appearing as `role 2 delta` (five scan-tagged
michel segments the chain had classified as deltas — visible for the first
time); C3 "same-cluster no-role segments on scan stoppers" **605 → 520
segments** (478 → 443 over 5 cm), 167 items unchanged. Doc 56 §7's 664 was
inflated by every published delta.

### 3.2 The role-7 arm (`d64v` vs `d64vleg`)

| | `d64vleg` | `d64v` |
|---|---:|---:|
| `T_stm_michel` branches shared / new / dropped | — | 124 / 0 / 0 |
| candidates bit-identical on all 124 | — | 410 / 578 (the other 168 differ in `n_other_published` only) |
| `is_stm` flips / `michel_found` flips | — | **0 / 0** (566 common items, by name) |
| role-1 / 2 / 3 / 5 / 6 points | 165272 / 2847 / 3568 / 253 / 3380 | **identical** |
| role-7 points | 0 | **4960** |
| role-7 segments | 0 | **278** on 166 items |
| `n_body_other + n_stop_other` (all items) | 282 | 282 |
| `n_other_published` | — | 278 |

The 4 counted-but-unpublished arms are interior `kOther` arms that the Michel
object walk had already claimed (role 3) by the time role 7 was emitted —
the case the plan's reviewer named; emitting last and not claiming is what
keeps the role-3 count at 3568 on both arms.

The 278 role-7 segments: median length 8.7 cm, 173 over 8 cm; median 0.33
MIP (÷ 47000), 7 over 1.4 MIP; **210 at an interior vertex, 68 at the stop**
(the class that had no counter and no rows before this round). Scan tags on
them: `delta / other` 206, `muon` 36, **`michel` 33**, untagged 3.

### 3.3 The interior-arm table against the scan's tags

Scan-tagged segments on the 268 judged scan stoppers, by the chain's role
(`d64v`; the leg's numbers in parentheses where they differ):

| scan tag | no role | role 1 muon | role 2 delta | role 3 michel | role 3 michel (other cl.) | role 5 gamma (other cl.) | role 6 survey (other cl.) | **role 7 other** | unfitted cluster |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `delta / other` | **42** (87) | 0 | 80 | 1 | 0 | 0 | 172 | **45** | 15 |
| `gamma` | 0 | 0 | 0 | 0 | 30 | 42 | 239 | 0 | 1 |
| `michel` | **8** (41) | 10 | 5 | 195 | 30 | 0 | 8 | **33** | 0 |
| `muon` | 376 (391) | 254 | 0 | 3 | 0 | 0 | 0 | 15 | 1 |

Doc 56 §8's 172 "no-role same-cluster `delta / other` segments" therefore
resolve, on the current arm, into: **172 companion-cluster survey segments**
(the scanner's "other" is mostly the survey pile, role 6, not the main
cluster at all), **80 role-2 deltas** (attached, ≤ 8 cm, interior — invisible
offline until this round), **45 role-7 arms** (33 attached > 8 cm interior at
median 0.28 MIP — doc 56's "46 of them > 8 cm at median 0.30 MIP, none
hadron-hot" population, now named by the chain itself; 11 attached ≤ 8 cm
interior; 1 at the stop), and **42 still with no role**: 13 attached ≤ 8 cm,
11 attached > 8 cm, 11 far (> 6 cm from the chain), 7 near or in between.
Those 42 are segments the chain never made ARMS of — hanging off an arm's
far vertex rather than a chain vertex, or disconnected pieces beyond T3b's
radius — and are the residual for any future round; `039252_3/74` (8),
`039253_2/77` (4), `039253_12/41` (3) carry most of the long ones.

**The 33 scan-michel `kOther` arms** are the finding this round did not go
looking for: segments the scanner called Michel, attached to the chain,
classified by `stm_michel_classify_chain_arm` / `_stop_arm` as `kOther`. The
33 split: interior arms (the Michel hangs off a vertex BEFORE the stop —
i.e. the stop overshot the Michel's root, doc 54 §1's mechanism, and T1a/T1b/
T1c did not reach it) and stop arms that failed a Michel gate (`mip`, length
or kink). They are visible now; doc 56's "8 `michel_found` FN whose Michel
the fit reached" (§3 there) should be re-read against this list.

## 4. Gates

| gate | result |
|---|---|
| `./build/clus/wcdoctest-clus` | 357 / 357 (one new default pin) |
| `d64vleg` vs `d63a` (PDVD, `--pts`) | 578 / 578 identical point geometry, 0 role labels moved, 0 `is_stm` flips; new branches `n_stop_other`, `n_other_published` only — **PASS** (`$W/gate1_pdvd.txt`) |
| `d64hleg` vs `d53h` (PDHD) | 325 / 325 identical, 0 moved, 0 flips — **PASS** (`$W/gate2_pdhd.txt`) |
| `d64v` vs `d64vleg` (the ON arm, `$W/gate3_on.txt`) | 124 / 124 shared branches; the only mover is `n_other_published` (168 items); 0 `is_stm` flips; role-1/2/3/5/6 point histograms identical; role 7 new — **rows only, as designed** |
| `d63_by_name.py prep_d64vleg prep_d64v` | 0 verdict differences on 566 items; C2 "no role" 41 → 8 with 33 becoming role 7 |
| binary pin | `libWireCellClus.so` md5 `b9e44ec66996` before and after every arm |
| `census_score.py --check` | 0 of 14 differ (the committed `d53v` payloads predate the prep fix and still reproduce doc 55) |
| prep fix | `prep_d63a_fix` vs `prep_d63a`: 0 class-count differences; C3 605 → 520; 206 items gain role 2 |
| production config | untouched this round — no flip-check needed; `pdvd/wct-pr-perevt.jsonnet` carries no `publish_other_arms` key |

## 5. The flip decision — production stays OFF, flagged for the owner

`publish_other_arms` is diagnostic rows, not a reconstruction change: there
is no `is_stm` or `michel_found` improvement to confirm, so the owner's
"if confirmed, turn it on" rule has nothing to act on. The closest precedent
is the survey (role 6): also rows-only, ruled OFF in production by the owner
on 2026-09-08 and ON on every scan arm through the `SURVEY` TLA. This round
follows that precedent: production is untouched; `d53_run_arms.sh`'s
documented `SURVEY` bag now reads
`{survey_enable:true,survey_radius_cm:60.0,survey_max_len_cm:25.0,publish_other_arms:true}`
so every future scan arm carries role 7. Unlike the survey, role 7 widens no
admission and fits nothing extra (it names arms the partition already
fitted), so flipping it in production would cost only `stm_michel_pts` rows
(4960 on 578 candidates, +3 %) — a one-line, reversible edit if the owner
prefers the display to see them in production too.

## 6. Found on the way, not fixed

1. **33 scan-michel segments are chain `kOther` arms** (§3.3). Some are stop
   arms that fail a Michel gate, some are interior arms rooted before the
   stop. A future admission round has a named list; nothing is changed here.
2. **42 `delta / other` segments still carry no role** — not arms of any chain
   vertex. Second-generation branches (off an arm's far vertex) are the
   likely majority; not traced this round.
3. **The over-clustering half** (doc 56's phrase): the 172 companion survey
   segments the scanner tagged `delta / other` are same-bundle clusters
   within 60 cm of the stop, mostly not over-clustering at all but nearby
   activity; the upstream `unmerge_assoc` / `protect_bundle` question is not
   this round's and was not opened.
4. `census_score.py`'s C3 metric is now "same-cluster segments with no role
   INCLUDING role 7" on an arm that publishes role 7 (426 on `d64v` vs 520
   on the leg) — the two numbers are not comparable across arms with the
   knob on and off; doc 56 §7's table note says which.
5. The viewer's `chain_group` role-2 branch became live for the first time
   with the prep fix (deltas now group under "delta / other" in the table);
   not exercised interactively this round beyond a syntax check and the
   module-level role lists.

## 7. Doc 56 update

T6 row marked done (rows-only knob, production OFF per the survey
precedent, the 172 resolved, the 33 michel `kOther` arms as the lead); §7's
664 corrected with the prep-fix footnote; Order moves to T7; §9 gains the
role-2 gap and the role-7 lead. Scripts committed:
`scripts/d64_interior_arms.py`.
