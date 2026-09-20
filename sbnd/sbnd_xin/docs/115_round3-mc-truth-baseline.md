# 115 — SBND round-3 samples: a truth baseline for the current production chain

**Status: BASELINE.** No toolkit C++ and no jsonnet is changed by this round. The one code
change is a runner flag, `run_chain_group.sh --mc`, additive and default OFF, with its
byte-identity gate in §3. Nothing here is a recommendation to flip anything.

This round exists because doc 107 asked for it. Doc 107 measured νμCC efficiency 69.0 % and
purity 86.3 % on a large BNB MC sample, but its νeCC numbers rested on **31** true νeCC in the
fiducial volume, and it closes with:

> A proper nueCC efficiency/purity needs the intrinsic-nue sample (doc 102's nueCC set) for
> signal and this sample for the numu/NC/cosmic background.

Haiwang's round-3 samples are that pair, plus an off-beam data arm for the cosmic background
rate. The intrinsic-νe sample carries **1511** true νeCC in the same FV — 49× doc 107's 31.

## Repro

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# the samples are reached through a symlink that is NOT tracked (same convention as
# input_files), so a fresh checkout has to make it before anything below resolves:
ln -sfn /nfs/data/1/yuhw/2025-fall-prod-sample/xin-round3-samples xin-round3-samples

# 0. pre-flight: operating point, pinned binary, event-ID census
python3 scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-17b; echo rc=$?
mkdir -p ~/tmp/d115-libsnap && cp -a ../../../local/lib/*.so ~/tmp/d115-libsnap/   # M1
git -C ../../../toolkit rev-parse HEAD > ~/tmp/d115-libsnap/TOOLKIT_HEAD
for s in cv nuecc off; do ./scripts/d115/rse_census.sh $s 16; done

# 1. stage A: reco1 -> imaging -> clustering + Q/L, then the products gate
for s in cv nuecc off; do
    JOBS=16 ./scripts/d115/stageA.sh $s
    ./scripts/d115/stageA_complete.sh $s || echo "REFUSE: $s stage A incomplete"
done

# 2. stage B: the 15-stage PR + tagger + BDT chain
for s in cv nuecc off; do JOBS=16 ./scripts/d115/stageB.sh $s; done

# 3. truth, the join, and every table below
for s in cv nuecc off; do JOBS=24 ./scripts/d115/analyze.sh $s; done
```

Logs and gate evidence: `docs/115_logs/`. Tables: `products/d115/<sample>/`. Figures:
`docs/115_sel/`, `docs/115_sel_edep100/`, `docs/115_vtx/`, `docs/115_scan/`, `docs/115_off/`,
`docs/115_time/`.

## 1. The samples

Staged by Haiwang 2026-09-19 under `sbnd_xin/xin-round3-samples` →
`/nfs/data/1/yuhw/2025-fall-prod-sample/xin-round3-samples/` (ai-helper #30).

| key | dir | reco1 files | events | what it is |
|---|---|---:|---:|---|
| `cv` | `mc-cv` | 154 | 2 017 | inclusive BNB CV, `aurora_SBND2026A_gen2_BNBLight_prodgenie_corsika_proton_rockbox0p1_sbnd_CV_v10_14_02_03_reco1_sbnd` — GENIE BNB ν + CORSIKA cosmics, rockbox (dirt) geometry. There is no dedicated νμCC sample in SBND2026A; νμCC is selected out of this one by truth. |
| `nuecc` | `mc-nuecc` | 225 | 2 001 | exclusive intrinsic νe CC, `…_EX_nuecc_v10_14_02_05_reco1_sbnd`, same production family |
| `off` | `beam-off` | 1 (merged) | 1 000 | Run-1 off-beam data, `data_SBND2026A_gen2_InTime-Run1_v10_14_02_02_reco1_sbnd`, frameshifted, 21 runs. No truth — this is the cosmic background arm. |

Products, confirmed by a branch listing on the staged files rather than assumed:

| | MC (`cv`, `nuecc`) | data (`off`) |
|---|---|---|
| DNN-SP wires | `recob::Wires_simtpc2d_dnnsp_DetSim.` | `recob::Wires_sptpc2d_dnnsp_Reco1.` |
| bad-channel masks | `ints_simtpc2d_badmasks_DetSim.` | `ints_sptpc2d_badmasks_Reco1.` |
| Wiener summary | `doubles_simtpc2d_wienersummary_DetSim.` | `doubles_sptpc2d_wienersummary_Reco1.` |
| optical | `opflashtpc{0,1}`, process `Reco1` | same |
| FrameShift | **absent** | `sbnd::timing::FrameShiftInfo_frameshift__FRAMESHIFT.` |

### 1.1 One `out_root` per reco1 file, and why

`run_chain_group.sh --layout perevt` keys its products by the art **event number alone**
(`ql_evt<ID>/`), and so does `run_pr_chain_batch.sh` (`pr_evt<ID>/`). SBND MC event numbers are
only unique within a `(run, subrun)`. Measured by `scripts/d115/rse_census.sh`:

| sample | events | unique (run, subrun, event) | unique **event number alone** |
|---|---:|---:|---:|
| `cv` | 2 017 | 2 017 | **50** |
| `nuecc` | 2 001 | 2 001 | 1 856 |
| `off` | 1 000 | 1 000 | 1 000 |

A single `out_root` per MC sample would therefore have silently overwritten most of the sample.
The campaign runs **one `out_root` per reco1 file** (`work-r3cv-d115/f000`, `f001`, …), which is
sufficient because the census also proved every file internally unique: 154/154 and 225/225 files
carry exactly **one subrun** each and no repeated event number inside a file. `--gbase` is not
needed — each root has its own `g0` — and nothing downstream pays for the split, because the
analysis joins on `(run, subrun, event)` read from `T_tagger`'s own branches (doc 109), not on a
directory name.

## 2. The operating point

| | |
|---|---|
| toolkit HEAD | `0a2807f4` (branch `apply-pointcloud`), recorded before and after every arm |
| binary | pinned snapshot `~/tmp/d115-libsnap` (230 `.so`), taken after a `wcbuild` |
| config tripwire | `prod_cfg_gate.py --ref ref/prod-2026-09-17b` → **PASS, 21/21 artifacts**, before and after the campaign |
| chain | `wct-reco1-dump.jsonnet` → `wct-img-all.jsonnet` → `wct-clus-matching-perevt.jsonnet` → `wct-pr-perevt.jsonnet`, 15-stage PR pipeline |
| flags | none beyond the per-event arguments. The SBND production operating point **is** the jsonnet TLA defaults (doc 92 §0); `PR_EXTRA_STAGES=pr_display` is the calib dump and has no physics effect (doc pr/3). `SBND_QL_KEEP_ICLUSTER=0` drops the imaging→Q/L handoff npz, which nothing downstream reads (doc 87). |
| `reality` | `sim` for both MC samples, `data` for beam-off. The runner's own lineage check enforces the match against stage A's `.lineage_reality`. |

**A stale-library trap was caught here (M1).** The installed `local/lib` was dated 12:59 while
HEAD `0a2807f4` was committed at 13:53, and that commit touches `img/` and `clus/` C++. The libs
were therefore not the ones HEAD describes. `wcbuild` was re-run, the freshness proof redone (no
source newer than any `.so`), and only then was the snapshot pinned. Without this every number
below would have been produced by an unidentifiable binary.

## 3. The one code change: `run_chain_group.sh --mc`

`run_chain_group.sh` is the group-mode stage-A driver. Before this round it could read SBND
**data** reco1 files only: its dump stage hardcoded `caf_offset_mode=product` and left the three
TPC product TLAs empty, i.e. at the C++ defaults, which are the data names. MC carries the same
products under the `DetSim` process and the `simtpc2d` label, and carries no `FrameShiftInfo` at
all — so on MC the old path aborts in the dump rather than falling back.

`--mc` appends exactly the four TLAs `run_reco1_dump.sh`'s `-mc` branch has always used (doc 67
§2) and changes nothing else:

```
wire_product=recob::Wires_simtpc2d_dnnsp_DetSim.
badmask_product=ints_simtpc2d_badmasks_DetSim.
summary_product=doubles_simtpc2d_wienersummary_DetSim.
caf_offset_mode=none                     # replaces 'product'
```

Additive and default OFF, the same shape as the `IMG_EXTRA_TLA` / `QL_EXTRA_TLA` channels doc 113
added to this same file. `caf_offset_mode` is carried in a variable rather than appended a second
time, so exactly one such TLA is ever emitted and no silent last-one-wins is possible. `--mc` says
nothing about `reality` (still the positional `data|sim`), and it is **refused** together with
`--fsproduct`, which only means anything under `caf_offset_mode=product`.

### Gates — `docs/115_logs/mc_flag_gates.txt`

**V1 — byte-identical when off.** Pristine vs modified script, same data reco1 file (nuecc48),
`--size 16 --groups 0 --to img`, `--mc` not given:

| artifact | verdict |
|---|---|
| `.wct-cfg-dump.json` | IDENTICAL after out_root normalisation, 2797 B, sha `b9ab25b27661a52c` |
| `.wct-cfg-img.json` | IDENTICAL, 161883 B, sha `4758f4898d0b5c83` |
| `icluster-apa{0,1}-{active,masked}.npz` | IDENTICAL by member content (192/144/192/144 members) |

The `.npz` had to be compared by member content, not raw bytes: they are zip containers with
embedded timestamps (M2).

**V2a — the MC dump equals the established MC path.** One mc-cv file (18 entries, run 719
subrun 45) dumped by `run_reco1_dump.sh -mc -caf none` and by `run_chain_group.sh … sim --mc`,
compared with `hash_archive.py` on member content:

| archive | members | verdict |
|---|---:|---|
| `frames-dnn.tar.bz2` | 90 | IDENTICAL `413c1385473f8d37…` |
| `opflash_apa0.tar.gz` | 54 | IDENTICAL `3f61e8344e1d7882…` |
| `opflash_apa1.tar.gz` | 54 | IDENTICAL `df10ad4a6736ddcb…` |

**V2b — MC group mode ≡ per-event through imaging *and* Q/L.** V2a proves only the dump. No MC
had ever gone through `run_chain_group.sh`, and the Q/L stage takes `reality` as its own TLA, so
`reality=sim` threading correctly through group-mode imaging and Q/L was an untested claim that
the whole campaign rests on. Three events of the same file were also run down the per-event
doc-93 path (`run_img_evt.sh mc`, `run_ql_evt.sh mc -save-pctree`):

| event | pctree members | verdict |
|---|---:|---|
| evt3 | 406 | IDENTICAL `0085cffc7fafd143…` |
| evt10 | 406 | IDENTICAL `2380820858baee4d…` |
| evt12 | 408 | IDENTICAL `2fa8c7344b7089a1…` |

Doc 92 §3.2 records group ≡ per-event for **data**; this extends the claim to MC.

## 4. Truth: two independent sources, cross-checked

The standalone chain produces **no truth layer**. Doc 107 could read truth out of Bee's
`0-mc.json` because its sample came from the LArSoft-embedded 1-step chain, which runs
`wclsTensorSetLabeler`; ours has no such stage. Truth here is built from two sources that share
no code, no library and no machine:

1. **the sample's GENIE TSV** (`truth/<sample>-truth.tsv`), made at FNAL with PyROOT and the
   nusimdata dictionaries: run, subrun, event, `n_nu`, `inu`, `nu_pdg`, `ccnc`, `mode`,
   `interaction_type`, `E_nu_MeV`, `vtx_x/y/z_cm`, `t_ns`, …;
2. **`Edep` per interaction**, read here out of the reco1 files by the existing
   `d108_reco1_truth_vectors.C` — bare ROOT, no LArSoft and no PyROOT (this environment has
   neither) — walking `SimEnergyDeposits_ionandscint_priorSCE_G4` → `MCParticle.ftrackId` → the
   largeant `Assns` → the generator `MCTruths` key. This is the same quantity Bee's truth nodes
   carry; doc 108 §3.3 validated the two agree to ±0.11 MeV.

`Edep` is not in the TSV, and doc 107 §5.6's signal definition and its Edep-binned vertex table
both need it. The macro also re-derives pdg, CC/NC, mode, vertex and time, so
`scripts/d115/build_truth.py` does not merely join the two — it asserts they agree row by row and
fails loudly if they do not.

**Result: all 7 433 interactions agree** (3 647 in `cv`, 3 786 in `nuecc`), on every shared
field, within ±0.11 MeV / ±0.02 cm / ±0.001 µs. Two independent extraction paths, no disagreement.

### 4.1 Conventions

Taken from doc 107 §5.5 so the two records stay comparable.

- **FV** `5 < |x| < 190`, `|y| < 190`, `10 < z < 450` cm — the owner's analysis FV, cathode
  excluded. Note this is *not* the sample TSV's `in_active_tpc` column, which is the whole active
  volume (`|x|<200, |y|<200, 0<z<500`), and *not* the toolkit's tagger FV either.
- **Match** reco vertex (`T_tagger.nu_x/y/z`) within **5 cm** of the nearest true vertex in the
  event. Vertex association, not a charge-based match.
- **Selection** score cut **and** reco vertex in the FV. Cuts `numu_score > 0.9`,
  `nue_score > 7.0`, and `nue_score > 4.0`.
- **Signal** true CC of that flavour with its true vertex in the FV, optionally `Edep > 100 MeV`.
- **Flavour is sign-blind** (±14 → numu, ±12 → nue), matching `TensorSetLabeler`. This is why the
  in-active-TPC νμCC count here is **706** where the sample README says 687 — the 19 ν̄μ.
- **Multi-neutrino events** keep **every** in-FV interaction as its own denominator row, as doc 107
  does. No truth-side picking rule is applied; picking by proximity to the reco vertex would bias
  the efficiency.
- Intervals are 68 % Wilson.

### 4.2 The truth census

| sample | interactions | **in FV** | νμCC | νeCC | NC | in FV **and** Edep > 100 MeV |
|---|---:|---:|---:|---:|---:|---:|
| `cv` | 3 647 | 783 | 557 | 5 | 221 | 709 (557 / 5 / 147) |
| `nuecc` | 3 786 | 1 626 | 90 | **1 511** | 25 | 1 618 (90 / 1 511 / 17) |

The `Edep > 100 MeV` cut removes **no CC interaction at all** in either sample — only NC
(74 in `cv`, 8 in `nuecc`). That matters for §11: it means the νμCC and νeCC efficiencies are
identical under both definitions, and the whole difference between the two variants sits in the
NC background.

## 5. What ran

Every arm rc=0, every stage-A gate on products, no event lost anywhere.

| sample | stage A wall | stage-A gate | stage B | rc≠0 | disk A + B |
|---|---:|---|---:|---:|---:|
| `cv` | 1 839 s | `expected=2017 ql_evt_dirs=2017 short=0 missing=0 empty=0 surplus=0` → **COMPLETE** | 2 017 events | **0** | 8.6 + 6.3 GB |
| `nuecc` | 2 262 s | `expected=2001 ql_evt_dirs=2001 … 0/0/0/0` → **COMPLETE** | 2 001 events | **0** | 9.3 + 9.5 GB |
| `off` | 1 075 s | `expected=1000 ql_evt_dirs=1000 … 0/0/0/0` → **COMPLETE** | 1 000 events | **0** | 4.6 + 3.0 GB |

Cost, as process wall summed over all jobs (the arms ran 16-way; §0's wall times are elapsed):

| sample | imaging | clustering + Q/L | PR (per event) | peak RSS/process |
|---|---|---|---|---|
| `cv` | 154 groups, 13 732 s, 89 s mean | 9 628 s, 63 s mean | 2 017 ev, 19 488 s, **9.7 s** mean | 0.91 / 0.49 / 0.44 GB med |
| `nuecc` | 225 groups, 15 368 s, 68 s mean | 13 887 s, 62 s mean | 2 001 ev, 37 316 s, **18.6 s** mean | 0.91 / 0.49 / **1.15** GB med |
| `off` | 63 groups, 7 680 s, 122 s mean | 6 111 s, 97 s mean | 1 000 ev, 10 873 s, **10.9 s** mean | 0.91 / 0.51 / 0.37 GB med |

**How the nuecc stage-B arm was actually produced.** The Repro block's single-driver form
(`stageB.sh nuecc`) produces this arm, but slowly: nuecc reco1 files hold ~9 events each and
the driver walks sub-roots one at a time, so the effective parallelism is ~9 against an
18.6 s/event PR stage — over four hours. The arm here was produced by that primary driver plus
three `scripts/d115/stageB_range.sh` workers on **disjoint** descending index ranges
(224→113, 112→80, 126→113) running concurrently. Each per-file sub-root is its own out_root
with no shared state, a sub-root either worker finishes is skipped by the other's `npr==nql`
guard, and every range boundary was checked against the live position of the other workers
before launch. The primary used `PR_JOBS=12`, the range workers `PR_JOBS=8`. Nothing
physics-relevant depends on the split — the products are per-event and identical either way —
but the doc states what ran, not what the shortest recipe would have run. `cv` and `off` were
each produced by a single `stageB.sh`.

The νeCC sample is ~2× the PR cost per event of the other two and carries ~2.6× the median
resident set — the nue BDT runs on far more events there (fill fraction 69.5 % against 9.8 %).
Doc 92 §7's 15.3 core-s median for νeCC events is reproduced (18.6 s mean, 15.0 s median).

**Provenance (V8).** Every one of the 5 018 `tracking-pr.root` carries exactly one value for
each doc-109 provenance string, and it is the pinned HEAD:

```
toolkit_git      = 0a2807f4ed43d4d6c7f539dbe0167bfe52892e47
wct_version      = apply-pointcloud-0.35.0-1367-g772c753d
op_config_sha256 = 399badee163e0f4d24c1e7927650690a4eab1ac9b04c7722002aaae12077a0b6
numu_xgboost_xml = uboone/weights/numu_scalars_scores_0923.xml
nue_xgboost_xml  = uboone/weights/XGB_nue_seed2_0923.xml
dl_weights       = uboone/scn_vtx/t48k-m16-l5-lr5d-res0.5-CP24.pth
```

**An independent check that the join is right (V11).** Doc 108 §3.5 measured, on the
*colleague's LArSoft-chain* sample with *Bee* truth, that `T_tagger.flash_time_us − true ν time`
= **+0.136 µs**, 98.9 % of rows within 0.1 µs. Reproduced here on a different chain, different
samples and a different truth source:

| sample | matched rows | median | within 0.1 µs | vs reference |
|---|---:|---:|---:|---:|
| `cv` | 660 | **+0.1360 µs** | 98.8 % | +0.0000 µs |
| `nuecc` | 1 406 | **+0.1355 µs** | 99.0 % | −0.0005 µs |

A shuffled join would smear this to the beam-gate width. It also says MC timing threads
correctly through the standalone chain and that `reality=sim` did not move the flash clock.

## 6. νμCC selection — `mc-cv`

Signal: true νμCC (sign-blind) with its vertex in the FV — **557** interactions.

| step | count | fraction |
|---|---:|---|
| candidate vertex within 5 cm | 463 | 83.1 % [81.5, 84.7] |
| … and the reco vertex in the FV | 460 | 82.6 % [80.9, 84.1] |
| … and `numu_score > 0.9` — **efficiency** | **387** | **69.5 %** [67.5, 71.4] |
| score-cut efficiency given a matched FV candidate | 387/460 | 84.1 % |

| selected candidates (`numu_score > 0.9`, reco vertex in FV) | count | fraction |
|---|---:|---|
| all selected (in 445 events) | 448 | |
| **signal — purity** | **387** | **86.4 %** [84.7, 87.9] |
| no true vertex within 5 cm | 45 | 10.0 % |
| … of which a real νμCC 5–50 cm away (misplaced vertex) | 23 | 5.1 % |
| … nearest true vertex > 50 cm (cosmic / other activity) | 17 | 3.8 % |
| true NC in FV | 12 | 2.7 % |
| true vertex outside the FV | 3 | 0.7 % |
| true νeCC in FV | 1 | 0.2 % |

Dropping the true-vs-reco vertex requirement (doc 107 §5.7): event-level efficiency
**426/557 = 76.5 %**, purity **418/448 = 93.3 %**.

**This reproduces doc 107 on a different chain, a different sample and an independent truth
source** — which is the strongest statement this round makes about the standalone chain:

| | doc 107 (13 216 evt, LArSoft chain, Bee truth) | d115 `cv` (2 017 evt, standalone chain, GENIE+SED truth) |
|---|---|---|
| vertex < 5 cm, all true ν in FV | 67.7 % | **68.8 %** |
| … νμCC | 80.0 % | **83.1 %** |
| … NC | 35.8 % | **32.1 %** |
| **νμCC efficiency** | **69.0 %** | **69.5 %** |
| **νμCC purity** | **86.3 %** | **86.4 %** |
| §5.7 event-level efficiency / purity | 77.0 % / 94.7 % | **76.5 % / 93.3 %** |

Requiring `Edep > 100 MeV` in the signal changes **nothing** for νμCC — the denominator is
identical (557), because no true CC interaction in either sample deposits under 100 MeV.

## 7. νeCC selection — `mc-nuecc`, the number this round exists for

Signal: true νeCC with its vertex in the FV — **1 511** interactions, against doc 107's 31.

| step | `nue_score > 7.0` | `nue_score > 4.0` |
|---|---|---|
| true νeCC in FV | 1 511 | 1 511 |
| candidate vertex within 5 cm | 1 067 = 70.6 % [69.4, 71.8] | same |
| … and the reco vertex in FV | 1 066 = 70.5 % | same |
| … and the score cut — **efficiency** | **618 = 40.9 %** [39.6, 42.2] | **734 = 48.6 %** [47.3, 49.9] |
| score-cut efficiency given a matched FV candidate | 618/1 066 = 58.0 % | 734/1 066 = 68.9 % |
| selected candidates | 632 | 781 |
| **purity** | **618/632 = 97.8 %** [97.1, 98.3] | **734/781 = 94.0 %** [93.1, 94.8] |

Doc 107's 31-event estimate was 35.5 % [27.5, 44.4] efficiency and 11/12 purity. The measured
40.9 % sits inside that interval, with ~1/7 the width — doc 107's νeCC numbers were consistent,
they were simply not a measurement. **They can now be quoted.**

Where the other 59 % goes, which is what a later round has to attack:

| loss | interactions | share of the 1 511 |
|---|---:|---:|
| no candidate vertex within 5 cm | 444 | **29.4 %** |
| — of those: no candidate in the event at all | 40 | 2.6 % |
| — candidate vertex 5–20 cm away | 148 | 9.8 % |
| — 20–50 cm | 140 | 9.3 % |
| — > 50 cm | 116 | 7.7 % |
| matched, but the reco vertex is outside the FV | 1 | 0.1 % |
| good vertex in the FV but fails `nue_score > 7` | 448 | **29.7 %** |
| — recovered by loosening the cut to 4 | 116 | 7.7 % |
| **selected** | **618** | **40.9 %** |

The split is almost exactly even between *vertex/pattern-recognition* loss (29.4 %) and *EM
identification* loss (29.7 %). Doc 107 §5.8 hand-scanned this on 31 events and found 7 vertex
failures against 13 score failures — the same shape, now measured.

The νμCC purity in this sample is **47/296 = 15.9 %** and is not a defect: it is an intrinsic-νe
sample, so almost everything a νμ cut selects is a true νeCC. It is quoted only to show the
νμ and νe selections are not disjoint at these working points.

## 8. The neutrino vertex

Distance from each in-FV true vertex to the **nearest** candidate vertex in its event — a
partition of the denominator, not overlapping cuts. `docs/115_vtx/`.

**`mc-cv`** (783 true ν in FV):

| population | N | matched < 5 cm | 5–20 | 20–50 | > 50 | no candidate |
|---|---:|---|---:|---:|---:|---:|
| all true ν in FV | 783 | **539 = 68.8 %** [67.2, 70.5] | 50 | 35 | 62 | 97 |
| νμCC | 557 | 463 = 83.1 % | 29 | 12 | 30 | 23 |
| νeCC | 5 | 5 = 100 % | 0 | 0 | 0 | 0 |
| NC | 221 | 71 = 32.1 % | 21 | 23 | 32 | 74 |
| Edep < 100 MeV | 74 | 7 = 9.5 % | 1 | 4 | 9 | 53 |
| 100 ≤ Edep < 300 | 126 | 66 = 52.4 % | 13 | 11 | 13 | 23 |
| Edep ≥ 300 MeV | 583 | 466 = 79.9 % | 36 | 20 | 40 | 21 |

**`mc-nuecc`** (1 626 true ν in FV):

| population | N | matched < 5 cm | 5–20 | 20–50 | > 50 | no candidate |
|---|---:|---|---:|---:|---:|---:|
| all true ν in FV | 1 626 | **1 122 = 69.0 %** [67.8, 70.1] | 155 | 146 | 158 | 45 |
| νeCC | 1 511 | 1 067 = 70.6 % | 148 | 140 | 116 | 40 |
| νμCC | 90 | 51 = 56.7 % | 6 | 5 | 25 | 3 |
| NC | 25 | 4 = 16.0 % | 1 | 1 | 17 | 2 |

Three readings worth carrying forward:

1. **The vertex, not the flavour tagger, is the leading νeCC loss.** 29.4 % of true νeCC never
   get a candidate within 5 cm, and only 2.6 % of those are "no candidate at all" — the chain
   almost always produces *a* candidate, it puts the vertex in the wrong place.
2. **NC is where the chain loses most.** 74 of 221 in-FV NC interactions in `cv` produce no
   candidate at all, and only 32.1 % get a vertex. That is mostly low visible energy:
   53 of the 74 have `Edep < 100 MeV`.
3. **Deposited energy predicts the vertex almost completely.** 9.5 % at `Edep < 100 MeV`,
   52.4 % at 100–300 MeV, 79.9 % above 300 MeV. Requiring `Edep > 100 MeV` lifts the `cv`
   all-ν number from 68.8 % to **75.0 %** and changes no CC number at all.

## 9. Beam-off: the cosmic background rate

1 000 Run-1 off-beam gates, no truth. The denominator is every gate the chain was **given**,
read from the census — not the gates that happened to produce a candidate. `docs/115_off/`.

| | rate per beam gate (68 % Wilson) |
|---|---|
| gate has ≥ 1 neutrino candidate | 87/1000 = **8.70 %** [7.85, 9.63] |
| … with the reco vertex in the FV | 25/1000 = **2.50 %** [2.05, 3.04] |
| … and `numu_score > 0.9` — **νμ selection** | 5/1000 = **0.50 %** [0.32, 0.78] |
| … and `nue_score > 7.0` — **νe selection** | 0/1000 = **0.00 %** [0.00, 0.10] |
| … and `nue_score > 4.0` | 0/1000 = **0.00 %** [0.00, 0.10] |

88 candidate rows in total, of which only 2 get a νe BDT score evaluated at all (2.3 %).
The five νμ-selected gates have `kine_reco_Enu` 517–1559 MeV, median 1112 MeV.

**Zero of 1 000 cosmic-only gates pass the νeCC selection at either working point.** With no POT
or trigger normalisation available in the staged sample this is a rate per gate and nothing
more — but as a rate it is the number a later round is scored against.

## 10. The working points are uncalibrated — score scans

The numu and nue scores come from the **uBooNE-trained** BDT weights (§5 provenance;
doc 107 §6 item 4). `numu_score > 0.9` and `nue_score > 7.0` are MicroBooNE conventions, not an
SBND-calibrated operating point, so this round also ships efficiency- and purity-versus-cut
curves per flavour: `docs/115_scan/d115_scan_{cv,nuecc,off}.{txt,tsv,png}`. A later round can
move the operating point off these TSVs without re-running the chain.

`nue_score == -15` is the nue BDT's "did not evaluate" sentinel, not a low score; it is excluded
from the nue scan and reported as a fill fraction instead. Fill fractions: `cv` 95/967 = 9.8 %,
`nuecc` 1 337/1 924 = 69.5 %, `off` 2/88 = 2.3 %.

Moving the νe cut from 7.0 to 4.0 buys 7.7 points of efficiency (40.9 → 48.6 %) for 3.8 points
of purity (97.8 → 94.0 %), and still admits 0/1000 beam-off gates. On these three samples alone
that trade looks favourable — but it is a physics decision on a sample with no POT weighting,
so nothing is recommended here.

## 11. What is and is not comparable to doc 107

**Comparable, and it agrees:** the νμCC ladder and the vertex fractions (§6). Same definitions,
same script (`d107_selection.py`, reused unmodified), agreement to ~1 point on every entry.

**One denominator difference, stated as counts rather than waved away.** Doc 107's truth came
from Bee, which lists an interaction only if it has a particle above 10 MeV KE — doc 107 §2.2
says so and warns against using it as an efficiency denominator. Ours is the complete GENIE
interaction list with no such drop. The effect is visible and small:

| | `cv` in FV | `cv` in FV, Edep > 100 MeV | removed |
|---|---:|---:|---:|
| all | 783 | 709 | 74 (all NC) |
| νμCC | 557 | 557 | **0** |
| νeCC | 5 | 5 | **0** |
| NC | 221 | 147 | 74 |

Because **no CC interaction in either sample deposits under 100 MeV**, every CC efficiency and
purity in §6 and §7 is *identical* under both definitions. The whole denominator question lives
in the NC background and in the "all true ν in FV" vertex number (68.8 % → 75.0 %). So the CC
numbers here are comparable to doc 107's, and the all-ν vertex number needs the qualifier.

**Not comparable / not done here:**

- **The chain is different.** Doc 107 ran the LArSoft-embedded 1-step chain; this is the
  standalone chain. That they agree to ~1 point on νμCC is a result, not an assumption.
- **The association is by vertex, not by charge.** Doc 107 §7 recommends a charge-based truth
  match using Bee's `truth_trackid_labeled`; the standalone chain does not produce that point
  set, so it is out of scope for this round and remains open.
- **The DL/SCN vertex is ON and is not bit-stable** (M4). These numbers are reproducible to
  physics precision, not byte for byte. No byte gate is claimed on any of them; the byte gates
  in §3 cover only the `--mc` runner change, whose arms were run without the PR stage.
- **No POT or trigger normalisation.** The beam-off rate is per gate; MC and beam-off are not
  combined into a prediction.
- **Flavour is sign-blind** (§4.1), so "νμCC" includes ν̄μCC throughout.

## 12. Open items

1. **The νeCC vertex is the biggest single lever**: 29.4 % of true νeCC lose the 5 cm match, and
   only 2.6 points of that is "no candidate at all". 148 sit 5–20 cm out — near misses that a
   vertex-refinement round could plausibly recover, worth ~10 points of efficiency.
2. **EM identification costs an almost equal 29.7 %.** 116 of those are recovered by moving the
   cut to 4.0; the remaining 332 fail below 4 and are the doc-107 §5.8 "EM shower
   misreconstructed" category, now with a 332-event population to scan instead of 10.
3. **The BDT weights are uBooNE's.** An SBND-trained pair is the obvious next step, and this
   round provides the labelled sample to train on.
4. **NC with `Edep < 100 MeV` is effectively invisible** to the chain (9.5 % vertex efficiency,
   72 % with no candidate). Whether that matters depends on the analysis; it is not a defect.
5. **`d107_truth_tagger_consistency.py` mis-counts on doc-109+ files** — its `act_*` roster
   predates companions (`act_role` 2/3). `d115_truth_join.py` applies `act_role <= 1`; on this
   round's arms that filter removes 55 655 companion entries against 14 064 real roster entries,
   so a roster read with the doc-107 script would be ~5× larger than the one doc 107 measured.
   Reported, not fixed:
   the doc-107 script is that round's record and stays byte-untouched (M10).

