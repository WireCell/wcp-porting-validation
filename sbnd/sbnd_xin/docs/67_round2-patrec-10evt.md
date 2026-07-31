# 67 — yuhw's round2-patrec 10 MC events through our production chain: the "missed STM" claim

> **Update (§9): his Bee set and ours are different events.** His ten are
> run 32/subrun 10 and run 31/subrun 88; every file in the wcgpu1 list is
> run 713, and none of his 15 truth neutrino vertices matches any of the 4 896
> in the list's 104 events. So the STM claim has **not** yet been tested on the
> same events — §§4–8 below describe the ten events of the list as written.

**Short answer: our chain did not miss a stopping muon in those events.** The
three events flagged as "true STM but not tagged" (ordinals 1, 5, 6 = evts 3,
11, 12) contain **no stopping muon in the beam window** — MC truth says so:
evt 11 has no muon that stops inside the active volume at *any* time, evt 12's
two stopping muons sit at t₀ = −1.03 ms and −1.69 ms, and evt 3's two sit at
t₀ = −631 µs and −313 µs. Our chain reconstructs those out-of-time cosmics,
matches them to their own flashes at the right times, and — with the
beam-window tagger gate off — **tags them STM** (§6). Production simply does
not evaluate out-of-time bundles (doc 56 gate), so they never become an event
label. The one event of the ten that *does* have a true in-time stopping muon,
evt 23 (ordinal 7), our chain **tags STM** (§4).

The chain did find **one apparent miss** in these ten events, and it is not an
STM one: **evt 7** (ordinal 3) has a fully contained 442-MeV ν<sub>µ</sub> CC
muon at t₀ = 900 ns, entirely inside the AV, and **no in-beam bundle reaches
the label table** — its 11 164-PE beam flash ends up matched to nothing, so no
tagger runs on it (§7). The loss is inside the Q/L step, after the cull;
whether it is a clustering merge or a solver drop is left open. Reported, not
fixed here.

Bee (10 events, ordinal order):
**https://www.phy.bnl.gov/twister/bee/set/d4f3a348-5d7d-4c8c-8bd0-494c4e309489/event/list/**

## Repro block

```bash
cd wcp-porting-img/sbnd/sbnd_xin
L=/nfs/data/1/yuhw/2025-fall-prod-sample/round2-patrec/mc_paths-v10_14_02_03-100files-wcgpu1.lst

# 1. extract the first file of the list (13 events) -- MC product names (sec 2)
./run_reco1_dump.sh -mc -caf none -t r2patrec-f1 $(head -1 $L)

export SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-r2patrec-f1
export SBND_WORK_ROOT=$PWD/work-r2patrec-f1
export SBND_MAX_JOBS=6 SBND_SAVE_ASSOC=1
NUF="-chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm -main-pair-real \
     -fvx 2.5 -fvy 3 -stm-fit -mip 56000 -unmerge-assoc"     # doc 59 production set

# 2. imaging, then the production nusel chain (Q/L -> TGM/STM/FC -> label table)
./run_img_evt.sh   mc all
./run_nusel_evt.sh mc $NUF all
#   -> work-r2patrec-f1/nusel-table.tsv (per bundle), nusel-events.tsv (per event)

# 3. the "what would the taggers say without the beam-window gate" arm (sec 6)
mkdir -p work-r2patrec-f1-nobw
for e in 3 4 7 9 11 12 23 24 31 34; do
  ln -sfn $PWD/work-r2patrec-f1/evt$e    work-r2patrec-f1-nobw/evt$e
  ln -sfn $PWD/work-r2patrec-f1/ql_evt$e work-r2patrec-f1-nobw/ql_evt$e
done
SBND_WORK_ROOT=$PWD/work-r2patrec-f1-nobw \
  ./run_nusel_evt.sh mc $NUF -no-bwonly <idx>          # idx 1..10

# 4. MC truth census (bare ROOT, no LArSoft) -- sec 5
ROOT_INCLUDE_PATH=/nfs/data/1/xqian/toolkit-dev/toolkit/root/src \
  root -l -b -q 'mc_truth_muons.C("'$(head -1 $L)'",0,9,-2000000,2000000,200)' \
  | grep -E '^EVENT|pdg=   13|pdg=  -13'

# 5. Bee (10 events, ordinal order)
printf '3\n4\n7\n9\n11\n12\n23\n24\n31\n34\n' > scan-r2patrec/first10.txt
./make_scan_bee.sh scan-r2patrec/bee work-r2patrec-f1 scan-r2patrec/first10.txt
```

Nothing was tuned and no default moved. The only code/config change is the
extraction plumbing of §2 (new jsonnet TLAs, keys omitted when unset).

## 1. Which ten events — and why this reading

The list `mc_paths-v10_14_02_03-100files-wcgpu1.lst` holds **10 MC reco1 art
files** (104 events total: 13, 6, 11, 5, 10, 11, 13, 12, 13, 10), all run 713,
one subrun each. "The 1st 10 events using this list" is read here as **the
first 10 art entries in list order**, i.e. entries 0–9 of the first file
(`…-7b1f-18f4-1bfb-4f0c.root`, 13 entries):

| ordinal | entry | run/subrun/event |
|---:|---:|---|
| 1 | 0 | 713 / 74 / **3** |
| 2 | 1 | 713 / 74 / **4** |
| 3 | 2 | 713 / 74 / **7** |
| 4 | 3 | 713 / 74 / **9** |
| 5 | 4 | 713 / 74 / **11** |
| 6 | 5 | 713 / 74 / **12** |
| 7 | 6 | 713 / 74 / **23** |
| 8 | 7 | 713 / 74 / **24** |
| 9 | 8 | 713 / 74 / **31** |
| 10 | 9 | 713 / 74 / **34** |

(Entries 10–12 of the same file — evts 36, 37, 45 — were processed too and are
in the tables, but are outside the requested ten.)

The competing reading, "the first event of each of the 10 files", is ruled out
by the ids themselves: those ten entry-0 events are 3, 3, 11, 2, 9, 4, 3, 3, 2,
1 — **six duplicate ids**, which no Bee set can carry unambiguously. It is also
checked directly in §5: none of those ten events has an in-beam stopping muon
either, so the conclusion of this note does not depend on which reading is
right.

## 2. Plumbing: reading MC reco1 files (the only change made)

The reco1 art sources were pinned against SBND **data** reco1 (`sptpc2d` module
label, process `Reco1`). These MC files carry the same three TPC products under
the **DetSim** process and the `simtpc2d` label:

| product | data branch (C++ default) | MC branch (this sample) |
|---|---|---|
| DNN-SP wires | `recob::Wires_sptpc2d_dnnsp_Reco1.` | `recob::Wires_simtpc2d_dnnsp_DetSim.` |
| bad-channel masks | `ints_sptpc2d_badmasks_Reco1.` | `ints_simtpc2d_badmasks_DetSim.` |
| Wiener summary | `doubles_sptpc2d_wienersummary_Reco1.` | `doubles_simtpc2d_wienersummary_DetSim.` |
| OpFlash per TPC | `recob::OpFlashs_opflashtpc{0,1}__Reco1.` | identical |

All three are already `SBNDReco1FrameSource` config keys, so **no C++ change and
no rebuild** was needed. `cfg/pgrapher/experiment/sbnd/wct-reco1-dump.jsonnet`
gains four TLAs (`wire_product`, `badmask_product`, `summary_product`,
`flash_process`) that are **omitted from the compiled config when empty**, and
`run_reco1_dump.sh` gains `-mc` which sets the three MC names.

Gate (§4 of CLAUDE.md, jsonnet change):

```
wcsonnet --tla-str input=/x.root --tla-str output_dir=/out <before|after>.jsonnet
  -> byte-identical, 2261 B          (knobs unset = data extraction unchanged)
with the three MC TLAs set:
  "wire_product" : "recob::Wires_simtpc2d_dnnsp_DetSim."
  "badmask_product" : "ints_simtpc2d_badmasks_DetSim."
  "summary_product" : "doubles_simtpc2d_wienersummary_DetSim."
```

MC files carry no PTB/TDC/DAQ-header products, so `-caf none` (the
`frame_apply_at_caf` re-reference is a data-only correction; with the key
absent `FlashTensorToOpticalPCs` is a no-op).

**Extraction is sound**, checked against yuhw's own LArSoft MC dump
(`input_files/input-10evt-mc/frames-dnn.tar.bz2`), which is an independent
route from MC to the same interchange format:

| | ours (evt 3) | yuhw MC dump (evt 2) |
|---|---|---|
| frame shape / dtype | (11276, 3427) f4 | (11276, 3427) f4 |
| tickinfo `[t, tick, tbin]` | `[0, 500, 0]` | `[0, 500, 0]` |
| chanmask_bad triplets | 93 | 93 |
| non-zero \|q\| mean / p99 | 1149 / 7159 | 1107 / 7595 |

Same shapes, same tick, same charge scale — the ×50 frame scale and 3427 ticks
carry over from data to MC unchanged.

## 3. What ran

13/13 events imaged (`ok: 13, failed: 0`), 13/13 through the production nusel
chain, rc=0. Flags are the **doc-59 production set verbatim** (`NUF` above),
beam window `[0.2, 2.2) µs` on `cluster_t0`, beam-window gate ON (doc 56
default), mode `mc` ⇒ `reality=sim` (per-TPC position offsets are a data-only
calibration and stay off). Outputs: `work-r2patrec-f1/` (169 MB).

The beam window is right for this MC: **every one of the 13 events has exactly
one flash in [0.2, 2.2) µs** (0.36–1.57 µs), and the truth beam-neutrino times
are 264–1475 ns — the window admits the beam bundle, so no verdict below is an
artifact of the gate cutting the beam bundle out.

## 4. Our tagger verdicts (production chain)

One row per event, the in-beam bundle only. Full per-bundle tables:
`work-r2patrec-f1/nusel-table.tsv`.

| # | evt | in-beam flash (µs / PE) | main len / pts | TGM | STM | FC | LM | label |
|---:|---:|---|---|:--:|:--:|:--:|:--:|---|
| 1 | 3 | 0.557 / 2 277 | 19.1 cm / 18 | 0 | 0 | 0 | 0 | nu-candidate |
| 2 | 4 | 0.364 / 623 | — | — | — | — | — | **no in-beam bundle** |
| 3 | 7 | 1.038 / 11 164 | — | — | — | — | — | **no in-beam bundle** |
| 4 | 9 | 0.986 / 648 | 15.5 cm / 317 | **1** | 0 | 0 | 1 | TGM |
| 5 | 11 | 0.531 / 338 | — | — | — | — | — | **no in-beam bundle** |
| 6 | 12 | 0.553 / 21 391 | — | — | — | — | — | **no in-beam bundle** |
| 7 | 23 | 0.653 / 9 849 | 120.3 cm / 2 488 | 0 | **1** | 0 | 0 | **STM** |
| 8 | 24 | 1.060 / 603 | — | — | — | — | — | **no in-beam bundle** |
| 9 | 31 | 1.303 / 4 068 | 108.4 cm / 280 | 0 | 0 | **1** | 0 | nu-candidate |
| 10 | 34 | 1.097 / 29 059 | 274.1 cm / 5 026 | 0 | 0 | 0 | 0 | nu-candidate |

(Beyond the requested ten: evt 36 nu-candidate 121.6 cm, evt 37 **STM**
84.6 cm, evt 45 nu-candidate with two in-beam bundles.)

## 5. MC truth: where the neutrino was, and where the stopping muons are

From `sim::MCTrack` (`mc_truth_muons.C`, §Repro). "AV" = |x| < 200, |y| < 200,
0 < z < 500 cm. Origin 1 = beam neutrino, 2 = cosmic.

**Scope of the census.** It is `sim::MCTrack` only, above an energy threshold:
**200 MeV** for the table below and for the reading-B scan, re-run at **20 MeV**
for evts 11 and 12 (the two flagged events with no in-beam bundle) — that lower
pass found nothing new. So "no stopping muon" throughout this note means "no
`sim::MCTrack` muon above that threshold whose end point is inside the AV". A
muon that entered below threshold, or one `mcreco` produced no MCTrack for,
would not appear.

| # | evt | beam-ν charged activity | in-beam-window stopping muon? | stopping cosmic muons (t₀, end point) |
|---:|---:|---|---|---|
| 1 | 3 | ν vertex (−134, 47, **507**) — downstream of the TPC; 1348-MeV µ leaves downstream | **no** | −631 µs → (153, 65, 276); −313 µs → (105, −11, 292) |
| 2 | 4 | ν vertex (370, −495, 630) — outside the cryostat | no | −74 µs, +14 µs, −800 µs, −994 µs |
| 3 | 7 | 442-MeV µ **(114, 7, 207) → (79, 52, 332), both in AV**, t₀ = 900 ns | contained ν µ (see §7) | +235 µs, −50 µs, −238 µs |
| 4 | 9 | 323-MeV µ entering, ends (−8, −199, 288) in AV, t₀ = 845 ns | (enters from below; tagged TGM) | +247 µs |
| 5 | 11 | ν vertex (116, **−356**, −113) — below/upstream of the TPC, only neutrons reach it | **no — and no stopping muon at any time** | none |
| 6 | 12 | 912-MeV µ (220, 13, −55) → (240, 148, 273): **x > 200**, outside the AV in the LAr between TPC and wall (light, no charge) | **no** | −1029 µs, −1687 µs |
| 7 | 23 | 751-MeV µ enters at z<0, **stops at (172, −74, 94) in AV**, t₀ = 505 ns | **YES** | +1001 µs |
| 8 | 24 | none in AV; the in-time 17-GeV cosmic (t₀ = 2.0 µs) misses the TPC | no | −393 µs, −1133 µs, −1433 µs, −608 µs, −1440 µs |
| 9 | 31 | 170-MeV µ, 12-cm stub in AV, t₀ = 1164 ns | no | −1631 µs, +78 µs |
| 10 | 34 | 871-MeV µ (37, −164, 229) → exits in x, t₀ = 960 ns | no | −1159 µs, +356 µs, +1393 µs, −1398 µs |

So of the ten, exactly **one** event has a true stopping muon in the beam
window — evt 23 — and our chain tags it **STM**. This sample is a
CORSIKA-overlay MC: every event carries 2–5 cosmic muons that stop inside the
TPC, but they are spread over ±1.5 ms, i.e. **hundreds of microseconds away
from the beam window**. That is what a hand scan sees as "obvious stopping
muons in the event"; it is not what the STM tagger is asked about.

The same census was run over the entry-0 event of all ten files (the
alternative reading of §1): none of those has an in-beam-window stopping muon
either.

## 6. Would our STM tagger have caught them? Yes — with the gate off it does

Second arm, identical binary and flags plus `-no-bwonly`, so the taggers
evaluate **every** matched bundle rather than only the beam-window one
(`work-r2patrec-f1-nobw/`). The in-beam rows are **identical to production in
all 10 events** (the gate only adds bundles, it does not change verdicts), and
out of time the STM tagger fires exactly where truth puts the stopping muons:

| # | evt | bundles | TGM | STM | STM bundles (flash time / main length) |
|---:|---:|---:|---:|---:|---|
| 1 | 3 | 9 | 4 | **2** | −436 µs / 68 cm; **−312.5 µs / 244 cm** |
| 2 | 4 | 10 | 4 | 3 | −821 µs / 236 cm; −74 µs / 231 cm; +14 µs / 68 cm |
| 3 | 7 | 14 | 4 | 4 | −582 µs / 5 cm; −238 µs / 398 cm; −49 µs / 26 cm; +236 µs / 299 cm |
| 4 | 9 | 9 | 2 | 2 | −647 µs / 116 cm; +247 µs / 272 cm |
| 5 | 11 | 14 | 4 | **0** | — |
| 6 | 12 | 13 | 6 | 1 | +490 µs / 231 cm |
| 7 | 23 | 11 | 5 | 2 | **+0.7 µs / 120 cm** (the production one); +1001 µs / 138 cm |
| 8 | 24 | 9 | 4 | 1 | −607 µs / 85 cm |
| 9 | 31 | 10 | 7 | 1 | +78 µs / 202 cm |
| 10 | 34 | 12 | 5 | 2 | −1142 µs / 17 cm; +356 µs / 52 cm |

The alignment with truth is direct: evt 3's STM bundle at **−312.5 µs** is the
cosmic muon truth puts at **t₀ = −312.7 µs** ending at (105, −11, 292); evt 11,
the one event with no stopping muon in truth at any time, is the one event with
**zero** STM bundles. So the STM tagger is working on this sample — what
production does is *scope* it to the beam-window bundle (doc 56), by design.

**This is the most likely source of the disagreement.** A chain that does not
apply the beam-window gate — or that does not do Q/L flash matching at all, so
every cosmic sits at its raw drift position and the "main" cluster is just the
biggest one — will present a stopping cosmic as *the* candidate and expect an
STM tag. Ours refuses to label an event by an object 300 µs out of the beam
gate. Worth asking yuhw: (a) did his run apply the beam-window gate and the
`-lm`/`-stm-fit`/`-unmerge-assoc` production flags, (b) is the object he is
looking at the *matched in-beam* bundle or just a cluster in the display, and
(c) what flash time / t₀ does his chain assign it.

## 7. The one event that looks like a real miss: evt 7

Evt 7 (ordinal 3) is the only event of the ten where truth puts a **fully
contained, in-time** neutrino muon in the AV — 442 MeV, (114, 7, 207) →
(79, 52, 332), t₀ = 900 ns — and our chain reports **no in-beam bundle**: the
11 164-PE flash at 1.038 µs reaches the label table matched to nothing.

What the Q/L log actually shows, in order (`ql_evt7/wct_ql_evt7.log`):

1. the beam flash is evaluated against three APA1 clusters —
   `cluster 0` (24 778 pts, pred 30 906 PE vs meas 11 164), `cluster 3`
   (5 567 pts, pred 21 224 PE) and `cluster 34` (7 pts, pred 18 PE): the two
   viable partners are large clusters that **over-predict** the measured light
   by 2–3×;
2. the cull stage keeps a beam-window bundle explicitly —
   `QLCULLINC apa1 cluster 5 KEEP beam-window bundle flash 9 (beam_pref exempts
   it from the rival-consistent drop)`;
3. the final `flash_bundles_map` of that APA1 job carries bundles at
   −1377, −917, −582, −238, −137, +172, +236, +567 µs and **nothing at
   ~1.0 µs** — matching by *time* here, not by flash id, so the local/global
   id convention cannot confuse the reading.

So the loss happens **inside the Q/L step, after the cull** — the label table
downstream only reports what it is given. Which of two mechanisms it is has
**not** been established here and both are open:

- the in-time charge was clustered into one of the large out-of-time clusters
  (a clustering/merge problem — note the over-prediction in step 1 is what a
  cluster carrying extra, out-of-time charge looks like), or
- a valid pairing existed and the solver dropped it in the final round (a
  matching problem).

Reported, not fixed here (CLAUDE.md §5 rule 7 / "mention, don't fix in the same
change"). Follow-up: locate the reconstructed points nearest the truth muon in
`ql_evt7/pctree-evt7.tar.gz`, see which cluster owns them, and only then decide
which of the two mechanisms to chase.

Two further notes on the "no in-beam bundle" events, which are **not** defects:
evts 4, 11, 12, 24 have their beam-ν interaction outside the active volume
(dirt / cryostat LAr / outside), so light without TPC charge is the physically
correct outcome — evt 12's 21 kPE in-beam flash comes from a 912-MeV muon
travelling at x ≈ 220–240 cm, in the LAr *between* the TPC and the wall.

## 8. Bee

One set, 10 events, **ordinal order** — Bee index *i* is line *i*+1 of
`scan-r2patrec/first10.txt`, i.e. Bee 1 = evt 3 … Bee 10 = evt 34:

**https://www.phy.bnl.gov/twister/bee/set/d4f3a348-5d7d-4c8c-8bd0-494c4e309489/event/list/**

Layers per event (doc 59 `make_stmfit_bee.py` merge): `img-global`,
`clustering-global`, `op` (the Q/L flash↔cluster match — use it to read each
cluster's t₀), the dead-area layers, and `stm_fit-global` (the STM
track fits). The `op` layer is the one to open first when comparing with a
display that has no flash matching: it is where the −300 µs cosmics separate
from the beam bundle.

## 9. Update — the two Bee sets are NOT the same events

yuhw's Bee set is
`https://www.phy.bnl.gov/twister/bee/set/169bdddf-0f71-44ab-aab1-47a896316040/event/list/`.
It holds 10 events, and **none of them is in the wcgpu1 list**. Three
independent checks, all agreeing:

**(a) The run/subrun/event labels.** Bee's own event list:

| # | his set 169bdddf | our set d4f3a348 |
|---:|---|---|
| 1 | 32-10-6 | 713-74-3 |
| 2 | 32-10-10 | 713-74-4 |
| 3 | 32-10-13 | 713-74-7 |
| 4 | 32-10-14 | 713-74-9 |
| 5 | **0-0-16** | 713-74-11 |
| 6 | 32-10-21 | 713-74-12 |
| 7 | 32-10-39 | 713-74-23 |
| 8 | 32-10-43 | 713-74-24 |
| 9 | 31-88-5 | 713-74-31 |
| 10 | 31-88-12 | 713-74-34 |

Every one of the ten files in `mc_paths-v10_14_02_03-100files-wcgpu1.lst` is
**run 713**, subruns **{0, 5, 35, 40, 51, 52, 59, 68, 70, 74}** — read twice by
independent code paths (`EventAuxiliary` via `TTree::Draw`, and the toolkit's
own `SBNDReco1Reader` through the opflash tensor-set metadata, which is what
puts `713-74-3` on our Bee set). **Runs 31 and 32 do not occur in this list at
all**, and neither does subrun 10 or 88. His 8 + 2 split across two subruns also
does not fit the list, whose first file alone has 13 events.

**(b) MC truth vertices — the label-independent check.** His `mc` layer (a
neutrino-only particle-flow tree) gives 15 neutrino vertices across the 10
events. Compared against **every** beam-origin truth vertex in **all 104 events
of the 10 files** (4 896 points, `sim::MCTrack` origin 1, E > 50 MeV): the
closest match to any of his 15 is **13.5 cm** away, typical nearest 40–110 cm,
worst 360 cm. Identical events would match to < 0.01 cm. The residual vectors
point in random directions, so it is not a coordinate-frame offset either.

```
his evt 0 vtx (17.08, -139.91, 16.25)  -> nearest ours file6 evt22 (74.0,-194.9,-3.7)   dist 81.7 cm
his evt 3 vtx (-39.83, 72.08, 450.64)  -> nearest ours file3 evt34 (-23.1, 67.7, 454.5) dist 17.7 cm
his evt 5 vtx (-51.19, -68.95,-365.31) -> nearest ours file2 evt46 (-54.7,-56.0,-366.4) dist 13.5 cm
... (all 15 in the session log; none below 13.5 cm)
```

**(c) The layer inventory** (useful for what his chain did run):

| layer | his set | our set |
|---|:--:|:--:|
| `img-global`, `clustering-global`, `deadarea`, `op` | yes | yes |
| `mc` (particle-flow truth tree) | **yes** | no |
| `stm_fit-global`, `track_fit`, `vertices`, `shower_track` | **no** | `stm_fit-global` yes |

So he is running the same MABC/Q-L Bee dumper (same layer names, same
`op` flash↔cluster layer) with the pf truth tree enabled, but **without the
STM-fit / PR layers** — i.e. the tagger tail of `run_nusel_evt.sh` did not
produce output in that set.

### 9.1 Full inventory of the staged files — his events are in none of them

Every entry of every file in the wcgpu1 list, read with the new
`mc_find_rse.C` (bare ROOT, `EventAuxiliary` only, works on `/pnfs` and
`xroot://` too):

```
$ root -l -b -q 'mc_find_rse.C("<list>", -1, -1)'
FILE   1 run 713 subrun 74 nevt 13 events: 3 4 7 9 11 12 23 24 31 34 36 37 45   ...-7b1f-18f4-1bfb-4f0c.root
FILE   2 run 713 subrun 35 nevt  6 events: 3 9 12 17 18 46                      ...-1233-281b-350b-480e.root
FILE   3 run 713 subrun  0 nevt 11 events: 11 17 25 27 30 34 36 37 38 46 47     ...-68b7-0715-d2b4-a1fa.root
FILE   4 run 713 subrun 68 nevt  5 events: 2 6 11 23 43                         ...-1fb7-fb06-5b32-db2e.root
FILE   5 run 713 subrun 59 nevt 10 events: 9 15 21 22 23 30 39 40 41 50         ...-a6a0-2395-0263-6d62.root
FILE   6 run 713 subrun  5 nevt 11 events: 4 5 15 18 22 28 34 36 39 43 46       ...-dad6-2031-00a1-30ae.root
FILE   7 run 713 subrun 51 nevt 13 events: 3 6 7 8 13 19 20 21 25 34 35 41 44   ...-978b-40cc-538d-4e7b.root
FILE   8 run 713 subrun 70 nevt 12 events: 3 5 9 15 17 18 19 25 29 36 37 48     ...-6be3-50c1-cb2e-ab24.root
FILE   9 run 713 subrun 40 nevt 13 events: 2 9 11 17 21 23 27 28 34 35 37 41 49 ...-b9e8-35c5-ce0d-e6ab.root
FILE  10 run 713 subrun 52 nevt 10 events: 1 6 7 8 19 22 23 30 42 48            ...-f62c-e7fb-8427-11ac.root

$ root -l -b -q 'mc_find_rse.C("<list>", 32, 10)'    ->  no MATCH
$ root -l -b -q 'mc_find_rse.C("<list>", 31, 88)'    ->  no MATCH
```

104 events, **one run (713)**, subruns {0, 5, 35, 40, 51, 52, 59, 68, 70, 74}.
Note also that if a job read this list in order, its first ten events would be
exactly file 1's — i.e. ours. His ten instead split 8 + 2 across two files.

### 9.2 Where his events must be, and how to find them

The wcgpu1 list is the **first 10 files of the 100-file gpvm list**, in the same
order (basenames verified one by one). The other 90 live only on
`/pnfs/sbn/data_add/sbn_nd/aurora/mc/v10_14_02_03/prodgenie_corsika_proton_rockbox0p1_sbnd/Gen2_2026/CV/reco1/000041/0004{11,12,13,14,16,18,19}/`,
which is **not mounted on wcgpu1** (`/pnfs` does not exist here) and not
copied anywhere under `/nfs/data`. Only these 10 files are local, so his two
files cannot be identified from this machine.

On a node that can see `/pnfs` (or with `xroot://` paths) one command finds
them:

```sh
root -l -b -q 'mc_find_rse.C("mc_paths-v10_14_02_03-100files-gpvm.lst", 32, 10)'
root -l -b -q 'mc_find_rse.C("mc_paths-v10_14_02_03-100files-gpvm.lst", 31, 88)'
# ~2 s per file; prints  MATCH <file-index> <entry> run .. subrun .. event ..  <path>
```

Then stage those two files here and rerun the Repro block verbatim — the
`-mc` extraction and the whole chain are already in place.

Independent confirmation once staged (in case the run/subrun labels in his set
are not the art ones): his `mc` layer's neutrino vertices, which must match to
< 0.01 cm. Kept in `scan-r2patrec/bee_169bdddf_vertex_match.txt`:

| his Bee event | ν vertex (cm) |
|---:|---|
| 0 | (17.08, −139.91, 16.25) |
| 1 | (−125.29, −528.35, −624.13) |
| 2 | (48.10, 56.52, 471.06) and (−25.73, 131.67, −421.13) |
| 3 | (−39.83, 72.08, 450.64) |
| 4 | (13.25, −161.67, 495.69) and (31.44, 185.09, 451.40) |
| 5 | (−51.19, −68.95, −365.31) and (383.70, 333.11, −323.15) |
| 6 | (−147.75, 114.00, 495.88) |
| 7 | (616.11, 172.48, −1224.09), (−403.89, 416.17, −449.53), (225.43, −136.96, 175.82) |
| 8 | (9.21, −35.67, 333.07) |
| 9 | (−246.89, 343.40, −377.08) |

**Consequence for §§4–7.** Everything in this note is about the ten events of
the wcgpu1 list. His "1st / 5th / 6th" refer to *his* ten, which are different
events, so the STM claim has not actually been tested against our chain yet.
What is needed to close it is either (i) the file list / sample his set came
from, or (ii) his set's run-subrun-event triplets resolved to files — then the
same procedure in the Repro block reruns those exact events. The findings that
do stand independently of which ten events are meant: our chain tags the one
true in-beam stopping muon it was given (evt 23), the STM tagger fires on
out-of-time stopping cosmics when the beam-window gate is lifted, and evt 7's
contained in-time muon gets no in-beam bundle (§7).

## Status

- Chain reproduced on the colleague's sample: **10/10 events processed**, 13/13
  including the rest of file 1.
- The STM claim on ordinals 1/5/6 **of this list** is not confirmed; truth says
  there is no in-beam stopping muon in those events, and the STM tagger does
  fire on the out-of-time ones when the gate is lifted. **But his Bee set is a
  different sample entirely (§9)**, so the claim is still open on *his* events —
  it needs his file list to be closed.
- One apparent miss found and reported: **evt 7's contained in-time
  ν<sub>µ</sub> CC muon produces no in-beam bundle** — loss inside the Q/L
  step, mechanism open (§7).
- Toolkit change: `wct-reco1-dump.jsonnet` MC product TLAs (compiled config
  byte-identical when unset). No C++ change, no rebuild, no default moved.
