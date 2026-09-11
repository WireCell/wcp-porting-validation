# 80 — The segment census: a row for every PR segment of the muon's cluster the chain left unnamed

Doc 78 action item 1. Output only, no physics: it makes items 2, 4 and 5 of doc 78 gradeable and gives the hand-scan display something to click on the segments the scanner has been tagging blind.

**Status (2026-09-10): DONE.** Knob `segment_census` in `CheckSTM_Michel` (C++ default false; toolkit `b1412f3c`). OFF byte-identical on both detectors; ON leaves every production branch and every existing point row identical and adds 10 781 role-8 rows with the gate that left each segment. **Left OFF in PDVD production by the owner's diagnostic-rows rule; rides with the scan TLA** (§6). One correction to doc 78 fell out (§5.1): the four "backward" Michels are attached arms the classification did not take, not body-exclusion victims.

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
mkdir -p /home/xqian/tmp/p80/libpin_p80 && cp /nfs/data/1/xqian/toolkit-dev/local/lib/libWireCellClus.so /home/xqian/tmp/p80/libpin_p80/   # the census build, md5 efb3430c
cd /home/xqian/toolkit-dev/toolkit && ./build/clus/wcdoctest-clus          # 371 cases pass (the defaults test pins the new key)
nohup bash $X/d80_arms.sh > /home/xqian/tmp/p80/arms_wave1.log 2>&1 < /dev/null & disown     # p80voff, p80hoff, p80vcen
bash $X/d80_gates.sh > /home/xqian/tmp/p80/gates.log 2>&1
```

Baselines: PDVD `p79vprod` (production after the doc 79 flip, P75 pin `02557b8d`), PDHD `p75hoff` (P75 pin). Record: smx1a+smx3+smx4.

## 1. What is missing from the output

Doc 78 §3.2: **42 of the record's michel tags on 32 judged stoppers point at PR segments of the muon's own cluster that have no row in `T_stm_michel_pts`** — the chain writes the muon (role 1), its deltas (2), the Michel object (3), gamma-collect blobs (4), the capture gamma (5), surveyed companions (6, survey off in production) and kOther arms (7, off in production); a PR segment of the main cluster that none of those took simply does not exist in the chain's output. 21 of the 42 touch the stop; on the five michel_found-0 items the segments run backward along the muon body (`039349_64/24` s24003, `039349_9/19` s19003, `039349_64/52` s52018, `039349_69/56` s56006) or are attached at the stop and not taken (`039253_3/61` s61007, `039349_32/63` s63006). Nothing in the output says which gate left them, so doc 78's offline reading of the T3b gates (dot radius 15 cm, piece cap 25 cm, `d_body < d_stop`) was a reconstruction of the chain's logic, not the chain's own word.

## 2. The change (toolkit, `clus/src/CheckSTM_Michel.cxx`)

- Knob `segment_census` (bool, default false), in `configure`, `default_configuration` (with the comment), the member, and `doctest_check_stm_michel_defaults.cxx`.
- **The census block**, placed last of all — after the Michel object, the capture gamma, the survey, the kOther arms and the gamma collect, immediately before `persist`. For every segment `pa.find_cluster_segments(g, *main)` returns (deterministic: `ordered_edges`) that is not in the chain, not in `rec.claimed` and not already written (`rec.pseg`), it writes **role-8 rows** via `add_points(..., 8, nullptr, rej, d_stop, d_body, keep_dead=true)` with the doc 62 T3b piece gates read on it, in T3b's own order and with T3b's own expressions (duplicated, not shared — M10):

  | `rej` | meaning |
  |---:|---|
  | 15 | no endpoint vertices |
  | 13 | attached to the chain (an endpoint vertex on the chain, or an out-edge to a chain segment) — T3b never looks at these |
  | 2 | farther than `michel_dot_radius_cm` (15) from the stop |
  | 3 | longer than `dot_max_len_cm` (25) |
  | 4 | the body test: closer to the muon body (rr > `dot_body_exclusion_cm` 5) than to the stop |
  | 14 | passes every T3b gate and is still unclaimed |
  | 9 | T3b (`stop_local_michel_pieces`) off, gates not applied |
  | 10 | no stop vertex |

  Rows sorted by graph index. Two scalars, `n_census_segs` and `n_census_admissible` (the rej-14 count), persisted only when the knob is on.
- Role 8 **claims nothing** (`add_points` exempts it as it does 6 and 7) and **no verdict reads it**; the `chain_coverage` test counts `rec.px` before these rows exist (the same placement argument as the gamma collect, doc 71).
- The three per-row columns `rej` / `d_stop` / `d_body` are emitted when the survey **or** the census is on (`add_survey_cols` and the `T_stm_michel_pts` writer); with both off the schema is the production one.
- Not the survey's path: `preload_clusters` is untouched, so the muon's own profile branches cannot move (doc 53's 20–25 % mover rate was the reason doc 78 asked for a separate pass).

**The binary under test carries another round's uncommitted code.** A concurrent session is building doc pdvd/81 (`michel_q2d`, the charge-based Michel energy) in the same working tree: uncommitted edits in `CheckSTM_Michel.cxx`, `StmMichelFunctions.{h,cxx}`, `TrackFitting.{h,cxx}` and `root/src/PdvdPrMagnifyTrackingVisitor.cxx`, all behind knobs the defaults doctest pins false. The census hunks are separable (10 hunks; the toolkit commit of this round is those alone), but no binary of the census alone exists: every pin here contains both rounds' code, and the OFF gate therefore covers both rounds' OFF paths at once. Two consequences are recorded in §4: the first wave of arms is void (a Clus-only pin against a `libWireCellRoot` the other session reinstalled mid-arm crashed every job at the ROOT-writing stage), and the arms the doc reports ran on a full snapshot of `local/lib` (`libpin_p80full`, 229 files, Clus `2525f189`, Root `46bf5105`).

Consumers: `prep_stm_michel_scan.py` adds role 8 to the role set and reads `rej`/`d_stop`/`d_body` for it (regression: 585/585 payloads of a knob-off arm byte-identical before/after the edit); `stm_michel_viewer.py` names the three new codes and puts role 8 in the "unassigned" group, where the scanner's michel/gamma/delta buttons apply.

## 3. Criteria and predictions (written before any arm was read: `/home/xqian/tmp/p80/pred.txt`)

- **OFF gate** byte-identical on both detectors: PDVD 596/596 candidates on every branch and point row, every Bee member, calib json, every `tracking-pr.root` tree; PDHD 325/325 likewise. The only OFF-path edits are two `|| m_segment_census` conditions and one `&& role != 8`, all inert when off.
- **ON arm** `p80vcen`: every production `T_stm_michel` branch bit-identical on all 596 candidates, two new scalars; every role 1–7 point row identical in value and order, role-8 rows appended, the three columns present on every row; the 36 PR-fitted orphan tags of doc 78 §3.2 gain a row (the 6 never-fitted do not); the four backward Michels read rej 4, s61007 / s63006 read rej 13, `039349_64/52`'s four free pieces at 11–13 cm read 4 or 14 (reported); rej 14 rare, every fire named; the Bee `stm` layer may gain points, every other layer identical; census on the record identical to `p79vprod` (232 / 7 / 46, 138 / 12 / 22).
- **Flip**: if the OFF gate passes and the ON predictions on branches, rows and census hold, `segment_census: true` goes into PDVD production (the display needs it on production output, and rows cost nothing a verdict reads); PDHD stays off, as with every knob without a PDHD record — though the owner may want this one on there for the display alone.

## 4. Gates (`d80_gates.sh` → `/home/xqian/tmp/p80/gates.log`)

**Wave 1 is void.** The first arms (`p80voff`, `p80vcen`, `p80hoff`) ran on a Clus-only pin (`libpin_p80`, `efb3430c`). At 18:58 a concurrent session reinstalled seven libraries in `local/lib` (Apps, Clus, Iface, Match, Mcs, Root, Util); every PDVD job in flight or started in the next minute died with rc 139 at the ROOT-writing stage (two with "failed to load plugin WireCellRoot"), 12 events on `p80voff` and 11 on `p80vcen`, the last events of run 039349. A per-event re-run on the same pin (`d80_rerun.sh`) crashed again with no build running, and the same event ran clean unpinned: the pinned `libWireCellClus.so` and the new `libWireCellRoot.so` no longer agree on what the tracking visitor reads. PDHD's wave-1 arm completed before the reinstall and passed its gate (61/61 zips, 8/8 trees, 325/325 × 130 branches), but is superseded. The crashed dirs stay under `work/` as the record of the failure (`039349_74_p80nopin` is the single unpinned test event).

**Wave 3, the arms the doc reports**, ran on a full snapshot of `local/lib` taken at 19:05 (`libpin_p80full`, 229 files, Clus `2525f189`, Root `46bf5105`; it carries the census and the other round's uncommitted default-OFF code, §2). 120/120 and 61/61 complete, 0 loader deaths, pin unchanged before and after.

| gate | result |
|---|---|
| **OFF, PDVD** `p80boff` vs `p79vprod` | Bee zip identical on 120/120 events (every member), calib json 119/119, **all eight `tracking-pr.root` trees identical on every event**, **596/596 candidates bit-identical on all 140 branches and every point row**, 0 flips |
| **OFF, PDHD** `p80bhoff` vs `p75hoff` | 61/61 zips, 61/61 calib, all eight trees, **325/325 × 130 branches and every point row** |
| **ON** `p80bcen` vs `p79vprod`, production output | Bee zip identical on 120/120 (the `stm` layer does not draw `stm_michel_pts`), calib 119/119, `T_bad_ch` / `T_cluster` / `T_proj` / `T_proj_data` / `T_rec_charge` / `Trun` identical on every event; **596/596 candidates bit-identical on all 140 production branches**, 0 flips; two new branches `n_census_segs`, `n_census_admissible` |
| **ON**, the point rows | **role 1–7 rows identical in value and order on 596/596 candidates**; three new columns `rej` / `d_stop` / `d_body`; **10 781 role-8 rows on 227 candidates** |
| **ON**, census on the record | `is_stm` 232 / 7 / 46 and `michel_found` 138 / 12 / 22, identical to `p79vprod`; `--check` 0 of 14 |

Every prediction of §3 held except one mechanism reading, §5.1.

## 5. What the census says

### 5.1 The 42 orphan michel tags of doc 78

| chain reading on `p80bcen` | n | items |
|---|---:|---|
| role 8, **rej 13 — attached to the chain, not taken** | **29** | the 21 that touch the stop (`039252_0/75` s75019 s75020, `039252_14/37`, `039252_14/81`, `039252_16/32`, `039252_16/98`, `039252_2/39`, `039253_0/102`, `039253_2/89` s89015, `039253_3/29`, **`039253_3/61` s61007**, `039253_3/79`, `039349_11/19`, `039349_22/22` s22010, `039349_22/63`, `039349_23/54`, **`039349_32/63` s63006**, `039349_5/64`, `039349_7/4` s4009 s4011, `039349_82/54`) **and the four "backward" segments of doc 78 §3.2 — `039349_64/24` s24003 (d_stop 2.5), `039349_9/19` s19003 (3.3), `039349_64/52` s52018 (5.5), `039349_69/56` s56006 (4.7)** — plus `039253_2/89` s89009, `039349_2/56` s56008 s56017, `039349_22/22` s22008 |
| role 8, rej 4 — the body test | 7 | `039349_64/52` s52016 s52017 s52019 s52020 (11–13 cm from the stop, 6–8 cm from the body), `039253_8/65` s65005, `039253_17/77` s77007, `039349_22/22` s22007 |
| role 8, rej 2 — beyond the 15 cm dot radius | 1 | `039349_64/71` s71001 (17.2 cm) |
| still no row | 7 | never fitted by the PR: `039252_12/95` s95021, `039252_3/68` s68021, `039252_9/101` s101006, `039253_8/65` s65015, `039349_34/46` s46012, `039349_5/65` s65019, `039349_9/47` s47018 |

**The correction to doc 78.** Doc 78 §3.2 read the four backward-running segments on the michel_found-0 items as victims of the body exclusion (`d_body < d_stop` offline). The chain's own word is **rej 13: they are connected to the muon chain** — an endpoint vertex on the chain or an out-edge to a chain segment — and T3b never looks at them, because T3b admits only *disconnected* pieces. They are attached arms near the stop that the Michel classification did not take (the same family as doc 64's kOther arms, but not at the stop vertex itself: d_stop 2.5–5.5 cm). Doc 78 action item 4 is therefore not a body-exclusion exception but **an attached-arm rule**: an arm hanging off the chain within a few cm of the stop, pointing away from the muon's end, is a Michel candidate. The body test does bite, on the four free pieces of `039349_64/52`'s shower at 11–13 cm and on three members of found Michels.

### 5.2 Across the sample

| rej | rows | meaning |
|---:|---:|---|
| 13 | 6 193 | attached to the chain, not taken |
| 2 | 4 243 | beyond the dot radius |
| 4 | 180 | the body test |
| 3 | 165 | longer than the piece cap |
| 14 | **0** | passes every T3b gate yet unclaimed — **never**: T3b takes everything it admits |
| 9, 10, 15 | 0 | |

538 role-8 segments over the 585 payloads; `n_census_admissible` is 0 on every candidate. The output now names every PR segment of the muon's cluster.

## 6. Flip: not into production — into the scan TLA

The pre-stated flip criteria hold (§3). The knob is nonetheless **left OFF in PDVD production**, for the owner's standing rule on diagnostic rows (2026-09-08, applied to the survey in doc 53 and to `publish_other_arms` in doc 64): rows the verdict does not read stay off in production and ride with the scan TLA. `segment_census:true` is added to the `SURVEY` TLA example in `d53_run_arms.sh` beside `publish_other_arms:true`; the scan arms that feed the display carry it from here. Turning it on in production is a one-key change (`segment_census: true` in `stm_michel_knobs`) whose cost is 10 781 rows and three columns in `T_stm_michel_pts` per 120 events and nothing else — the owner's option, as with the survey.

Toolkit commit `b1412f3c` (the census hunks alone; the binary that ran also carried the other round's code, §2 — the committed hunks were compiled and tested only together with it).

## 7. Next

Doc 78 items 2 (the peak-then-drop stop mover), 4 (backward-Michel admission) and 5 (the P4 rejection census), each now gradeable segment by segment.
