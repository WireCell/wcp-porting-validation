# 76 — P5: an STM-only TrackFitting configuration

This is P5 of doc 70 §2.3 / §6, the last item of its order. The owner asked for each proposal in its own file; this is the P5 file.

**Status (2026-09-10): DONE.** Jsonnet only, no C++ (toolkit `50af7a70`). Nothing is flipped: the parameter's default *is* today's behaviour, the compiled production config is byte-identical (§4), and the smoke arm that points the STM side at a copy of the shipped file reproduces production on every output of all 120 events (§5.1). The second smoke arm shows what the handle is for: `dx_norm_length` 4 mm on the STM side alone moves the STM chain and everything it writes, and nothing else (§5.2) — reported, not flipped.

## 0. Repro

```bash
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img; X=$IMG/pdvd/docs/nf_sp_img_clus/scripts
# the fit-file copies the smoke arms read (tf_same.json = the shipped file byte for byte; tf_dx04.json = dx_norm_length 4 mm)
cp /nfs/data/1/xqian/toolkit-dev/toolkit/cfg/pgrapher/experiment/protodunevd/pdvd_track_fitting.json /home/xqian/tmp/p76/tf_same.json
python3 -c "import json;d=json.load(open('/home/xqian/tmp/p76/tf_same.json'));d['dx_norm_length']=4.0;json.dump(d,open('/home/xqian/tmp/p76/tf_dx04.json','w'),indent=2)"
# the compiled-config proofs (PRE_* = copies of the two files before the change) -> /home/xqian/tmp/p76/proofs.txt
PRE_PR=/home/xqian/tmp/p76/pre_pr.jsonnet PRE_EVT=/home/xqian/tmp/p76/pre_wct-pr-perevt.jsonnet $X/d76_proofs.sh
# the two smoke arms (bare production, P75 pin, detached)
nohup bash $X/d76_arms.sh > /home/xqian/tmp/p76/arms.log 2>&1 < /dev/null & disown
```

## 1. The question

Doc 70 §2 asked whether the fit's 0.6 cm step (`low_dis_limit` 12 mm, `dx_norm_length` 6 mm, on a 0.78-pitch detector) limits the stopper verdict, and answered no (§2.2–2.3: keep 0.6 cm). What it did find is that the sampling question cannot even be *studied* on the STM chain alone: one `trackfitting_config_file` reaches three components — `TaggerCheckSTM`, `CheckSTM_Michel` and `TaggerCheckNeutrino` (`protodunevd/pr.jsonnet:175, :1550, :1695, :1853`) — so doc 68 §4's `dx_norm_length` 4 mm arm moved the neutrino path with it and could not be flipped alone even had it won. **P5**: a `pr.jsonnet` parameter that lets the two STM components carry their own fit file, defaulting to the shared one.

## 2. The three binding sites

| component | role | reads | after P5 |
|---|---|---|---|
| `TaggerCheckSTM` (`pr.jsonnet:1550`) | the stopping-muon tagger: the fit whose kink row is the stop, whose `stm_pass` products the Michel stage anchors on | `trackfitting_config_file` | `stm_trackfitting_config` |
| `CheckSTM_Michel` (`:1695`) | the STM/Michel chain: re-fits the main and its companions, the verdict profile | `trackfitting_config_file` | `stm_trackfitting_config` |
| `TaggerCheckNeutrino` (`:1853`) | the neutrino pattern recognition on the beam bundle | `trackfitting_config_file` | unchanged |

The PDVD production pipeline (`pdvd/wct-pr-perevt.jsonnet`'s `pipeline_names`) has no `tagger_check_neutrino` stage today; the third site matters for the neutrino jobs that reuse `pr.jsonnet`, and for the day it is added.

## 3. The change (jsonnet only)

- **`cfg/pgrapher/experiment/protodunevd/pr.jsonnet`**: `pr()` gains `stm_trackfitting_config_file=null`, documented next to `trackfitting_config_file`. An object-level `local stm_trackfitting_config = if stm_trackfitting_config_file == null then trackfitting_config_file else stm_trackfitting_config_file` feeds `tagger_check_stm` and `check_stm_michel`; `tagger_check_neutrino` keeps `trackfitting_config_file`. The `null` default is the key-suppression idiom in another form: the emitted string is the very same string, so the compiled JSON does not change.
- **`pdvd/wct-pr-perevt.jsonnet`**: TLA `stm_trackfitting_config = null` beside `trackfitting_config`, threaded into `clus_maker.pr(...)`. `-A stm_trackfitting_config=/abs/path.json` scopes a fit-parameter study to the STM chain (the runner's TLA hook, `PDVD_PR_TLA`, already passes `-A` arguments through: doc 68 §0 used it for `trackfitting_config`).
- PDHD's `pr.jsonnet` has the same three sites (`pdhd/pr.jsonnet:190, :1537, :1693, :1851`) and is **not touched**: doc 70 named `protodunevd/pr.jsonnet`; the PDHD counterpart is a one-line mirror when a PDHD study needs it.
- No other experiment's config is touched; `common/clus.jsonnet`'s builders already take the file per component.

## 4. Compiled-config proofs (`d76_proofs.sh` → `/home/xqian/tmp/p76/proofs.txt`)

The proofs compile the production perevt job (`pipeline_names` = the 13 production stages) with `wcsonnet`, pre-change and post-change, and diff the sorted JSON. A shadow search path (`-P`) holding only the pre-change `pr.jsonnet` stands in for "yesterday's tree"; proof 0 shows the shadow actually wins over `WIRECELL_PATH`.

| # | what | result |
|---|---|---|
| 0 | the post-change perevt (which passes the new parameter) compiled against the **pre**-change `pr.jsonnet` must fail | fails: `has no parameter stm_trackfitting_config_file` |
| 1 | TLA unset: pre perevt + pre `pr.jsonnet` vs post perevt + post `pr.jsonnet` | **0 lines** |
| 1b | TLA unset: pre perevt against pre vs post `pr.jsonnet` (the `pr()` default alone) | **0 lines** |
| 2 | `-A stm_trackfitting_config=/home/xqian/tmp/p76/tf_same.json` vs unset | exactly the two `trackfitting_config_file` values (`tagger_check_stm`, `check_stm_michel`) |
| 2b | the same with `tagger_check_neutrino` added to the pipeline | 4 lines = the same two values; `TaggerCheckNeutrino:pr` keeps `pgrapher/experiment/protodunevd/pdvd_track_fitting.json` |

## 5. Smoke arms (bare production, P75 pin `02557b8d`, `d76_arms.sh`)

The predictions below were written before the arms finished; §5.1–5.2 are the results (`d76_gates.sh` → `/home/xqian/tmp/p76/gates.log`; baseline `p75vprod`, doc 75's confirmation arm on the same pin).

- **`p76vsame`**: `-A stm_trackfitting_config=/home/xqian/tmp/p76/tf_same.json`, a byte-identical copy of the shipped file at an absolute path. **Prediction: every output identical to the production baseline `p75vprod`** (the components resolve the absolute path and read the same 61 keys). This is the plumbing proof at the output level, on all 120 events.
- **`p76vdx4`**: the same file with `dx_norm_length` 4 mm (the doc 65 / doc 68 option, every other key verbatim). **Prediction: the STM-side outputs move** (the fit's smoothing changes `T_stm_michel` and the `stm_fit` products on many candidates; doc 68 §4 saw `is_stm` flip on 29 items and 32 new candidates when the whole chain moved) **and the non-STM outputs do not**: in the production pipeline the only other fit consumer is absent, so the check is that the moved branches are the STM chain's and the Bee zip's non-STM layers are unchanged. It is graded on the record **as an observation only**: doc 70 §2.3 — any step or smoothing change that moves the candidate set needs a new scan, and doc 68 §4 already measured this one as worse on top of the anchor.

### 5.1 `p76vsame`: identical to production on every output

120/120 events complete (the usual `039252_11` writes no candidate on every arm), 0 loader deaths, pin unchanged. Against `p75vprod`: **every `mabc-pr.zip` member identical on all 120 events** (sha256 per member: the 16 dead-area layers, `clustering`, `mc`, `shower_track`, `steiner_graph`, `steiner_terminals`, `stm`, `stm_fit`, `track_fit`, `vertices`); the calib json identical on 119/119; `T_stm_michel` **578/578 candidates bit-identical on all 140 branches**, every point row identical; **every tree of `tracking-pr.root` identical on every event** (`T_bad_ch`, `T_cluster`, `T_proj`, `T_proj_data`, `T_rec_charge`, `T_stm_michel`, `T_stm_michel_pts`, `Trun`); `census_score.py` the same 230 / 7 / 46 and 136 / 12 / 22. The TLA reaches both components (proof 2) and changes nothing (this arm).

Two comparator traps surfaced and are fixed in `d76_gates.sh`: `numpy.array_equal` on a jagged (STL-vector) branch compares object identities and reports a false difference on `T_proj_data`; a list comparison of `T_rec_charge` fails on its NaN `reduced_chi2` entries (NaN ≠ NaN). The tree check maps NaN to None and compares lists.

### 5.2 `p76vdx4`: the STM chain moves, the rest does not

`dx_norm_length` 4 mm for `TaggerCheckSTM` and `CheckSTM_Michel` only. 120/120 events complete; two write no candidate (`039252_11` as always, and `039252_4`, whose only candidate disappears under the changed fit). Against `p75vprod`:

| output | moved? |
|---|---|
| `T_stm_michel`: candidates 578 → 576, 545 matched, **0 of 545 bit-identical**, `is_stm` flips 13 | the STM chain, as expected |
| `T_stm_michel_pts`, `T_rec_charge` (the fits), `T_proj_data` | 119 events |
| `T_cluster` and the Bee `clustering` layer | 13 events — the chain's own cluster edits (unmerge, no-trajectory pieces) |
| Bee layers `track_fit` / `stm_fit` (119 events), `mc` (117), `shower_track` (78), `vertices` (71), `stm` / `steiner_graph` / `steiner_terminals` (53) | everything `CheckSTM_Michel` writes into the PR graph |
| `T_bad_ch`, `T_proj`, `Trun`, the 16 dead-area layers | **identical on every event** |

In the production pipeline `CheckSTM_Michel` is the last stage that touches the PR graph, so "the rest" is small here; the point of the handle — `TaggerCheckNeutrino` untouched — is proved at the config level (§4, proof 2b) because that stage is not in the PDVD production pipeline. On the record, as an observation only: `is_stm` 230 / 7 / 46 → **222 / 8 / 52**, `michel_found` 136 / 12 / 22 → 137 / 13 / 23 — the same direction doc 68 §4 measured with the whole chain moved ("worse on top of the anchor"). Not a candidate for a flip; the reason P5 exists is that this number can now be measured for the STM side alone.

## 6. No flip

There is nothing to flip: `stm_trackfitting_config_file = null` is today's behaviour, proved byte-identical in §4. What P5 delivers is the handle doc 70 §2.3 asked for — the next sampling study (`dx_norm_length`, `low_dis_limit`, the end-trim radius) can run on the STM chain alone, with `TaggerCheckNeutrino` held fixed, and its candidate-set movement graded on its own.

## 7. Next

The owner's review of the PDVD STM chain against the hand-scan record: doc 77.
