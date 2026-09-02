# 93 — the colleague's 8 MC events through our production chain: taggers before the PR round

> **Result: our chain agrees with the colleague's on all 8 events, event for
> event.** 5 STM, 2 TGM, and **36-77-17 is NOT STM-tagged by us either** — it
> is the only one of the 8 that survives to `TaggerCheckNeutrino` and comes
> out `nu-candidate`. So the disagreement to debug is *our chain vs Prabhjot's
> older scan*, not our chain vs the colleague's latest.
>
> Bee (same event order as the colleague's set, indices line up 1:1):
> **https://www.phy.bnl.gov/twister/bee/set/16e3d89d-cf91-4639-9342-2413458ea8fb/event/list/**
> colleague's: https://www.phy.bnl.gov/twister/bee/set/9797078d-763a-4202-9af0-2d53127f1bd2/event/list/

## Repro block

```bash
cd wcp-porting-img/sbnd/sbnd_xin
# stage A0 -- MC reco1 extraction.  -mc: products are simtpc2d/DetSim, not
# sptpc2d/Reco1.  -caf none: the file carries no FrameShiftInfo/PTB/TDC product.
./run_reco1_dump.sh -mc -caf none -t stmfb8 \
    input_files_reco1/stm_tagger_feedback/type2-8evt-reco1.root

# stages A1/A2/B -- imaging, Q/L, then the 15-stage PR chain, reality=sim
./scripts/stmfb8_run.sh all            # == img, then ql, then pr

# the tables
python3 pr_scores_table.py --root work-stmfb8-pr --sample stmfb8 --out scores.tsv
cat work-stmfb8-pr/nusel-events.tsv work-stmfb8-pr/nusel-table.tsv

# the Bee set (fork of make_pr_bee.py; see sec 6)
python3 scripts/bee/make_stmfb_bee.py -q work-stmfb8-ql -p work-stmfb8-pr \
    -o bee/stmfb8/stmfb8.zip 4 17 22 28 12 31 10 25
(cd bee/stmfb8 && BROWSER=echo bash ../../upload-to-bee.sh stmfb8.zip)
```

Nothing was tuned, no default moved, no production runner was edited. The only
new files are `scripts/stmfb8_run.sh` and the Bee fork `scripts/bee/make_stmfb_bee.py`.

## 1. The sample, and why it is *not* a data sample

`input_files_reco1/stm_tagger_feedback/type2-8evt-reco1.root` → 8 entries.
The branch list settles the lineage without asking:

| product | data branch (C++ default) | this file |
|---|---|---|
| DNN-SP wires | `recob::Wires_sptpc2d_dnnsp_Reco1.` | `recob::Wires_simtpc2d_dnnsp_DetSim.` |
| bad-channel masks | `ints_sptpc2d_badmasks_Reco1.` | `ints_simtpc2d_badmasks_DetSim.` |
| Wiener summary | `doubles_sptpc2d_wienersummary_Reco1.` | `doubles_simtpc2d_wienersummary_DetSim.` |
| FrameShiftInfo | present | **absent** |

plus `GenieGen`/`G4`/`corsika`/`MCTruth`/`MCShower` products. So: **MC**, run
with `-mc -caf none` and `reality=sim` throughout (doc 67 §2 is the precedent).

**`reality` is the position-correction switch, and it is not cosmetic.** It
gates `pos_offset` in *both* stage-A Q/L and stage-B PR
(`cfg/.../sbnd/wct-clus-matching-perevt.jsonnet`, default `sim`):

```
$ diff <(wcsonnet -A reality=data … wct-clus-matching-perevt.jsonnet) \
       <(wcsonnet -A reality=sim  … wct-clus-matching-perevt.jsonnet)
< "pos_offset": [0, -1.1,  6.7]      # apa0, DATA only
< "pos_offset": [0,  1.1, -6.7]      # apa1, DATA only
```

Running these MC events as `data` would have shifted every reconstructed point
by ~6.8 cm in y–z and flipped scope / in-window / PID decisions wholesale
(doc pr/38 round 3). The runner stamps `.lineage_reality` and stage B checks it.

RSE, in art-file order (`EventAuxiliary` scan) — confirmed identical to the
`opflash_tensorset_<ID>_metadata.json` the runner derives RSE from:

| entry | RSE | event id |
|---:|---|---:|
| 0 | 827 / 27 / 4 | 4 |
| 1 | 304 / 6 / 28 | 28 |
| 2 | 707 / 18 / 12 | 12 |
| 3 | 146 / 60 / 31 | 31 |
| 4 | 36 / 77 / 17 | 17 |
| 5 | 966 / 2 / 22 | 22 |
| 6 | 921 / 29 / 10 | 10 |
| 7 | 658 / 38 / 25 | 25 |

Eight different runs — which is exactly why the per-event path was used rather
than `run_chain_group.sh` (§6).

## 2. "Event No. 6, 7" — read off the colleague's set, not guessed

The colleague's Bee set orders the events differently from the art file. Its
`event/list/` page resolves the reference with no interpretation:

| his Event No. | RSE | his note |
|---:|---|---|
| 0 | 827-27-4 | |
| **1** | **36-77-17** | **not STM tagged in this run** |
| 2 | 966-2-22 | |
| 3 | 304-6-28 | |
| 4 | 707-18-12 | |
| 5 | 146-60-31 | |
| **6** | **921-29-10** | **TGM tagged** |
| **7** | **658-38-25** | **TGM tagged** |

Our Bee set is built in **that** order, so index *i* is the same event in both.

## 3. The tagger table (the deliverable)

One row per **in-beam** bundle; every event had exactly one in-beam flash and
one in-beam bundle. `stmfit=eval` means the STM tagger ran on that main.

| bee_idx | RSE | our verdict | t0 (µs) | flash PE | main len (cm) | main pts | TGM | STM | FC | LM | n_bundles | colleague |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 827-27-4 | **STM** | 0.366 | 6384 | 114.4 | 1293 | 0 | 1 | 0 | 0 | 15 | same |
| 1 | 36-77-17 | **nu-candidate** | 1.216 | 5490 | 69.8 | 1360 | 0 | **0** | 0 | 0 | 16 | **not STM — same** |
| 2 | 966-2-22 | **STM** | 0.292 | 6947 | 89.7 | 1824 | 0 | 1 | 0 | 0 | 13 | same |
| 3 | 304-6-28 | **STM** | 1.430 | 27834 | 109.0 | 1884 | 0 | 1 | 0 | 0 | 13 | same |
| 4 | 707-18-12 | **STM** | 0.738 | 15168 | 123.6 | 3523 | 0 | 1 | 0 | 0 | 15 | same |
| 5 | 146-60-31 | **STM** | 1.513 | 7393 | 100.9 | 1212 | 0 | 1 | 0 | 0 | 16 | same |
| 6 | 921-29-10 | **TGM** | 1.483 | 2320 | 67.9 | 170 | 1 | 0 | 0 | 0 | 14 | **TGM — same** |
| 7 | 658-38-25 | **TGM** | 0.449 | 27868 | 128.1 | 2140 | 1 | 0 | 0 | 0 | 11 | **TGM — same** |

Committed as `bee/stmfb8/stmfb8-tagger-summary.tsv`; the per-bundle source is
`work-stmfb8-pr/nusel-table.tsv` (all 113 bundles, in-beam and out).

**Evaluated-and-said-no, not never-evaluated.** With `beam_window_only=true`
the taggers only see in-window mains, and out-of-window rows carry `-1` rather
than `0` — so a table that collapsed the two would reproduce the "not tagged"
claim without explaining it. Counted over every bundle row, not eyeballed:

```
total bundle rows : 113
in_beam=1         :   8   of which tgm/stm/fc in {0,1} : 8
in_beam=0         : 105   of which tgm/stm/fc all -1   : 105
rows breaking the pattern : 0
```

So all 8 verdicts above are real `0`/`1`: **no event of the 8 lost its
candidate to the beam-window gate.** That was the leading alternative
explanation — it is exactly what doc 67 §6 turned out to be — and it is ruled
out here.

## 4. 36-77-17 — where the STM verdict was actually made

> **SUPERSEDED by doc 94 §2 (2026-09-01).**  This section concluded that
> 36-77-17's `STM=0` came from the core fit-based condition.  The
> `save_stm_fit` dump says otherwise: pass status is **5**, i.e.
> `flag_pass` was TRUE -- the eval accepted it as a stopping muon -- and
> `detect_proton` is what saved it, calling a proton end at ks1 = 0.047.
> That is load-bearing: doc 94 §5 rests on the control being saved by
> `detect_proton`, not by the eval.


Only this event reaches `TaggerCheckNeutrino` (the other 7 are cosmic-tagged,
so the neutrino tagger never runs — which is why their `tracking-pr.root` is
13 KB and carries no calib dump).

From `work-stmfb8-pr/pr_evt17/wct_pr_evt17.log`:

```
TaggerCheckSTM: evaluate_demoted_mains: 1 demoted main(s) added
TaggerCheckSTM: skipped 6 out-of-scope main cluster(s)
TaggerCheckSTM: beam_window_only [0.200, 2.200) us: 2 main(s) evaluated, 19 out of window
cathode_guard:     cluster 20 stop x=-54.37cm apa=0 face=0 bb.x=[-201.45,-0.45]cm
                   cathode_x=-0.45cm dist=53.92cm
check_other_tracks: cluster 20 seg 1/1: len=2.9cm medQ=1.17MIP lenThr=1.1cm straight=0.918
TaggerCheckSTM: cluster 20 → STM=0 TGM=0
check_stm_conditions: cluster 24 no STM fit: fully contained (Mid Point A)
TaggerCheckSTM: cluster 24 → STM=0 TGM=0
```

So the two mains in the window were both evaluated and both returned STM=0:

- **cluster 24** exits early with *"no STM fit: fully contained (Mid Point A)"*.
- **cluster 20** — the 69.8 cm in-beam main — went the full distance. Neither
  veto guard fired: the cathode guard measured the stop point **53.9 cm** from
  the cathode (it only acts within 5 cm), and the second-track guard found a
  single leftover segment of **2.9 cm** (thresholds are 25 cm / 15 cm). The
  `STM=0` therefore came from the core fit-based condition, not from a guard.

The BDT-side cosmic block, which runs on the *same* main, is the sharpest
handle on why (`T_tagger`, event 36-77-17):

| `cosmict_2_*` feature | value | reading |
|---|---:|---|
| `particle_type` | 13 | reconstructed as µ⁻ |
| `flag_inside` | 1 | the far end **is inside the FV** — geometrically a stopping muon |
| `dQ_dx_end` | **1.089 MIP** | **no Bragg rise at the far end** (the flag's own bar is > 1.4 MIP) |
| `dQ_dx_front` | 1.143 MIP | flat, same as the end |
| `flag_dir_weak` | 1 | direction weakly determined |
| `angle_beam` / `theta` / `phi` | 115.1° / 115.1° / 177.9° | |
| `n_muon_tracks` / `total_shower_length` / `valid_tracks` | 2 / 0 cm / 0 | |

→ `cosmict_flag_2 = 1` ("single segment muon going in wrong direction or
exiting FV", `clus/src/NeutrinoTaggerCosmic.cxx:701`). Note this is a **numu-BDT
cosmic input, not a stopping-muon verdict** — it does not contradict
`TaggerCheckSTM`; it independently records the same physical fact: *the track
ends inside the FV but shows a flat ~1.1 MIP dQ/dx profile at that end.*

Event-level scores for 36-77-17: `numu_score = -0.177`,
`nue_score = -15.0` (the "not filled" sentinel — needs an equality test, not a
threshold), `cosmic_flag = 1`, `cosmict_flag = 1`, `cosmict_10_score = 0.70`,
`numu_cc_flag = 1`, `kine_reco_Enu = 346.9 MeV`,
`kine_reco_add_energy = 105.7 MeV`, `T_cluster` 62 in-scope clusters.

**Recommended next step for the debug round:** the missing Bragg peak is the
thing to confirm directly. That means the STM fit dump (`save_stm_fit` →
`tracking-stm.root`, dQ/dx vs residual range; doc 41), which doc pr/3 confirms
is diagnostic-only and never changes the STM verdict. `run_pr_chain_batch.sh`
omits it and carries no env knob for it, so wiring one means a fork of that
runner (M10) — a small, contained piece of work, deliberately not done here.

## 5. Validation of the run itself

| check | result |
|---|---|
| config tripwire before the run | `scripts/cfg/prod_cfg_gate.py` **PASS 21/21** vs `ref/prod-2026-09-01c` (2026-09-01 19:26) |
| binary | pinned `~/tmp/prod0901b-libsnap` (the prod0901b arm's snapshot), md5s in `bee/stmfb8/`; its `libWireCellGen.so` predates `feaac7b6`, which is a **comment-only** header change (`git show --stat`) |
| compiled PR config vs `ref/prod-2026-09-01c/prod_prjob.json` | **22 keys differ, all accounted for** (§5.1) — zero physics knobs |
| compiled Q/L config vs the canonical compile | **20 keys differ**, all paths / RSE / `save_tensors` — zero physics knobs |
| `DL vertex failed` | **0** across all 8 logs (the SCN vertex really ran) |
| per-event `rc.txt` | 0 × 8; 8 `evt*`, 8 `ql_evt*`, 8 `pr_evt*` dirs (counted, not taken from the runner's own "failed: 0") |
| output completeness | full debug output — `mabc-pr.zip`, `pctree-pr-evt<ID>.tar.gz`, `PR_EXTRA_STAGES=pr_display`; nothing suppressed |
| Bee set | UUID content-verified after upload: 8 rows, DAQ IDs and order identical to the colleague's set |

### 5.1 The PR compiled-config diff, key by key

Diffed by component identity, not array index — adding `pr_display` shifts every
later node and inflates a positional diff to 11 526 spurious keys.

| difference | why |
|---|---|
| `DetectorVolumes:dv-apa0-1` `pos_offset` **removed** (6 keys) | `reality=sim`; the reference is `data`. §1. |
| `MultiAlgBlobClustering:clus_pr` `bee_points_sets[0].coords` `y_cor,z_cor → y,z` | same `reality` gate — with no `pos_offset` the Bee layer uses uncorrected coords |
| `+ PrDisplayDump:pr` node, `+ pipeline[15]` | `PR_EXTRA_STAGES=pr_display`, the calib dump; read-only, no physics effect (doc 92 §2.3) |
| `TaggerCheckNeutrino:pr` `dl_weights '' → uboone/scn_vtx/…CP24.pth` | the **reference** is the deviant: `compile_prjob_cfg.sh` deliberately empties it to keep the non-bit-stable DL vertex out of gates (CLAUDE.md M4). Our run has the production DL vertex ON, as `run_pr_chain_batch.sh` intends. |
| `TaggerCheckNeutrino:pr` `+ vertex_scoreboard=true` | auto-set alongside `pr_display` (doc pr/75) |
| run/subrun/event, 6 file paths | per-event |

One further deviation from every `prod0901b` arm lives **upstream of this
config**, in stage A0 and so not visible in the PR diff: production data runs
the reco1 dump with `caf_offset_mode=product` + `caf_offset_override=0`, which
reads the authoritative `FrameShiftInfo::fFrameApplyAtCaf`. This file carries no
such product, so `-caf none` omits the key entirely. Forced by the input, not a
choice. The quantity it feeds — the opflash tensor-set metadata — was checked
directly instead: the run/subrun/event of all 8
`opflash_tensorset_<ID>_metadata.json` match the art file's `EventAuxiliary`
exactly (§1).

## 6. Why the per-event path, and what was forked

`run_chain_group.sh` (the production stage-A driver) **cannot run this sample**:
it hardcodes `caf_offset_mode=product` — which aborts on a file with no
`FrameShiftInfo` — and passes the *data* product names. It would also need an
`rse_map` for 8 events spanning 8 runs. The per-event drivers get RSE from each
event's own opflash metadata for free, and doc 92 §3.2 records group and
per-event mode as byte-equivalent on the PR outputs (hash gate 38/38, ROOT gate
19/19). No production runner was edited.

`scripts/bee/make_pr_bee.py` also does not fit and was **forked, not edited**
(M10) into `scripts/bee/make_stmfb_bee.py`, for two reasons that are both
consequences of 7 of the 8 events being cosmic-tagged:

1. It *skips* the PR job's own `clustering-global`. On a cosmic-tagged event
   that layer is the only thing the PR job contributes at all, so skipping it
   would leave 7 of 8 events Q/L-only. The fork carries it as
   `clustering-pr-global` — the same name the colleague's set uses.
2. It *refuses* an event with no selected nu candidate, because the three PR
   point layers degenerate to a whole-event dump. That is a statement about a
   zip that **has** those layers; here they are simply absent (verified on
   evt 4 and evt 25: `mabc-pr.zip` holds only deadarea + clustering-global), so
   there is no degenerate layer to ship and the refusal is replaced by a note.

## 7. Where things live

```
input_files_reco1/extracted-stmfb8/       frames-dnn.tar.bz2, opflash_apa{0,1}.tar.gz
work-stmfb8-ql/evt<ID>/                   icluster-apa{0,1}-{active,masked}.npz
work-stmfb8-ql/ql_evt<ID>/                pctree-evt<ID>.tar.gz, mabc-all-apa.zip, opflash
work-stmfb8-pr/pr_evt<ID>/                tracking-pr.root, mabc-pr.zip, pctree-pr-*, nusel-evt<ID>.tsv
work-stmfb8-pr/nusel-{table,events}.tsv   the tagger tables of sec 3
bee/stmfb8/                               zip + url + annotated index + summary/score TSVs
```
