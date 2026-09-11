# 79 — The candidate cap: `max_candidates` 8 → 64 in PDVD production

Doc 77 §7.1 / doc 78 §6: the first action item, because it recovers two owner-confirmed Michel stoppers with a config value, no threshold and no scan.

**Status (2026-09-10): CONFIRMED and FLIPPED in PDVD production** (`max_candidates: 64` in `pdvd/wct-pr-perevt.jsonnet`). No C++. The 578 production candidates are bit-identical; the two named items come back as stoppers with a found Michel; 16 further, never-scanned candidates appear on the same six events. Confirmation arm: §6.

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
mkdir -p /home/xqian/tmp/p79
# the arm (bare production, P75 pin 02557b8d = today's production binary, detached)
nohup bash $X/d79_arms.sh > /home/xqian/tmp/p79/arms_wave1.log 2>&1 < /dev/null & disown
# the gates against the production baseline p75vprod
ARM=p79vcap bash $X/d79_gates.sh > /home/xqian/tmp/p79/gates_p79vcap.log 2>&1
# the flip proofs (PRE_EVT = the perevt file before the flip)
PRE_EVT=/home/xqian/tmp/p79/pre_wct-pr-perevt.jsonnet bash $X/d79_proofs.sh
# the confirmation arm (flipped file, no TLA)
WAVE=2 ARMS2="p79vprod|" nohup bash $X/d79_arms.sh > /home/xqian/tmp/p79/arms_wave2.log 2>&1 < /dev/null & disown
ARM=p79vprod bash $X/d79_gates.sh > /home/xqian/tmp/p79/gates_p79vprod.log 2>&1
```

## 1. The mechanism

`CheckSTM_Michel` collects the STM-flagged main clusters of the event, sorts them by cluster id, and keeps the first `max_candidates` (`CheckSTM_Michel.cxx:1527-1533`; C++ default 8, not set in any config). The cap predates the hand scan and exists for wall time. On the 120-event production sample it fires on 6 events (9–13 candidates; `039252_2`, `039253_6`, `039253_8`, `039349_18`, `039349_64`, `039349_81`) and drops 19 tagger-accepted clusters (`T_stm_pass` status 0). Two of the 19 are on the record as **STM_MICHEL, owner-judged**: `039253_8/65` and `039349_81/62`. They were never reconstructed, so doc 77 §2 counted them under "no candidate at all" and §4 under "missed Michel (a)". PDHD's cap fires too (7 of 61 events on `p75hoff`, 9–12 candidates).

## 2. Criteria and predictions (written before the arm was read: `/home/xqian/tmp/p79/pred.txt`)

- The 578 baseline candidates present and bit-identical on every `T_stm_michel` branch and point row (the dropped clusters carry the highest ids on each event, so the existing candidates are processed, and claim, first).
- New candidates: 19, on exactly the six events; `039253_8/65` and `039349_81/62` `is_stm` 1 (the tagger accepted them); `michel_found` not predicted; the other 17 unjudged, reported by name, not graded.
- Every other output identical except on the six events, where the STM layers gain the new candidates' objects.
- Census: `is_stm` 230/7/58 → 232/7/56 if the two hold; `michel_found` 136/12/29 → at most 138/12/27; no item leaves.
- **Flip criteria**: the first point exact; both named items recovered with 0 new `is_stm` FP on judged items and no lost TP; then flip PDVD only, proofs A/B/C, confirmation arm identical to the graded arm.

## 3. The arm: `p79vcap` (`max_candidates` 64) vs `p75vprod` (`d79_gates.sh` → `/home/xqian/tmp/p79/gates_p79vcap.log`)

120/120 events complete, 0 loader deaths, pin `02557b8d` before and after, no "keeping the first" line in any log.

| check | result |
|---|---|
| shared candidates | **578 / 578 bit-identical on all 140 branches**; 0 `is_stm` flips; 578 / 578 identical point geometry, 0 role labels moved |
| candidates | 578 → **596** (18 new; one of the 19 dropped clusters writes no candidate) |
| Bee zip (sha256 per member) | identical on 114 events; on the six cap events the `track_fit`, `shower_track`, `vertices` and `mc` layers differ (the new candidates' objects); calib json identical on 113, differs on the same 6 |
| `tracking-pr.root` | `T_bad_ch`, `T_cluster`, `T_proj`, `Trun` identical on every event; `T_proj_data`, `T_rec_charge`, `T_stm_michel`, `T_stm_michel_pts` differ on exactly the six cap events |
| census on the record | `is_stm` 230 / 7 / 46 → **232 / 7 / 46**; `michel_found` 136 / 12 / 22 → **138 / 12 / 22** (judged items with a candidate 544 → 546); `--check` 0 of 14 |

Predictions held, with one count off by one (18 new candidates, not 19).

### 3.1 The 18 new candidates

| item | `is_stm` | bits | `michel_found` | Michel KE / length | muon length | record |
|---|---:|---:|---:|---|---:|---|
| **`039253_8/65`** | **1** | 0 | **1** | 31.7 MeV / 10.8 cm | 93 cm | **STM_MICHEL / both / owner** |
| **`039349_81/62`** | **1** | 0 | **1** | 40.6 MeV / 15.0 cm | 47 cm | **STM_MICHEL / attached / owner** |
| `039253_6/114` | 1 | 0 | 1 | 25.5 / 13.2 | 54 | unjudged |
| `039253_6/117` | 1 | 0 | 0 | (5.4 / 7.1, under P1's floor) | 507 | unjudged |
| `039253_8/93` | 1 | 0 | 0 | | 106 | unjudged |
| `039349_18/42`, `/46`, `/47` | 1 | 0 | 0 | | 266, 386, 111 | unjudged |
| `039349_64/80` | 1 | 0 | 0 | | 184 | unjudged |
| `039349_81/63` | 1 | 0 | 0 | | 249 | unjudged |
| `039252_2/113`, `039253_6/99` | 0 | 1024 | 0 | | 69, 151 | unjudged |
| `039253_6/109` | 0 | 1096 | 0 | | 125 | unjudged |
| `039253_6/94`, `039253_8/77`, `039253_8/81` | 0 | 4 | 0 | | 203, 521, 350 | unjudged |
| `039253_8/98` | 0 | 520 | 0 | | 197 | unjudged |
| `039349_64/73` | 0 | 512 | 0 | | 8 | unjudged |

Ten of the 18 read `is_stm` 1, three with a Michel object; the eight rejected read the usual bits. None of the 16 unjudged items was ever a candidate, so the record cannot grade them — they are the scan need doc 77 §7.6 named, and they are production output from now on.

## 4. Gates

There is no C++ and no OFF path: the knob is a config value the component has always read. The gate is the arm itself — 578 / 578 bit-identical shared candidates (§3), which is the "order could matter" risk doc 77 §7.1 raised, closed. PDHD is untouched (its cap stays at 8; §7).

## 5. The flip (`d79_proofs.sh` → `/home/xqian/tmp/p79/proofs.txt`)

`max_candidates: 64,` at the head of `stm_michel_knobs` in `pdvd/wct-pr-perevt.jsonnet`, with the comment naming the six events, the two items and the census. Compiled-config proofs on the production pipeline:

| proof | result |
|---|---|
| A: PRE + `-S stm_michel_extra={max_candidates:64}` (the arm's config) vs POST unset | **0 lines** |
| B: POST + `-S stm_michel_extra={max_candidates:8}` vs PRE | exactly one line: `"max_candidates": 8` present vs absent (a value knob has no key-suppression form; 8 is the C++ default) |
| C: PRE vs POST | exactly one line: `"max_candidates": 64` |

## 6. Confirmation arm `p79vprod` (flipped file, no TLA; `/home/xqian/tmp/p79/gates_p79vprod.log`)

120/120 complete, 0 loader deaths, pin `02557b8d`, no cap line in any log. Against `p79vcap`: **596 / 596 candidates bit-identical on all 140 branches and every point row** (`d51g_branch_census --pts`, `/home/xqian/tmp/p79/g_cap_vs_prod.txt`). Against `p75vprod` it reproduces §3 exactly: 578 / 578 shared candidates bit-identical, 18 new, both named items `is_stm` 1 with a found Michel, census 232 / 7 / 46 and 138 / 12 / 22, `--check` 0 of 14. The production baseline prep for the next round is `/home/xqian/tmp/p79/prep_p79vprod` (585 payloads).

## 7. What stays open

- **The 16 unjudged new candidates** need a scan before their `is_stm` 1 / `michel_found` 1 readings count for anything. Ten `is_stm` 1 on never-scanned clusters is a purity question the record cannot answer today.
- **PDHD**: the cap fires on 7 of 61 events; the knob stays at 8 there for want of a record. When PDHD is scanned, the same one-line change applies.
- **64 is "no cap" on this sample**, not a studied value. An event with more than 64 STM-flagged mains would still be capped.

## 8. Next

Doc 80: the segment census (doc 78 action item 1).
