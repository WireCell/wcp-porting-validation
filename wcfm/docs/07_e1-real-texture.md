# 07 — E1 on real events: the WC_FM_Sim pilot neutrino depos through the wcfm chain (Route 1)

*2026-09-25. Analysis + workspace inputs only: a `lar` depo-extraction job, a depo-file sim driver forked by
duplication, 80 new workspace events (101–140, 201–220, 301–320), the E1 probe. No toolkit code changed;
events 1–100 untouched.*

## 0. Repro

```
# 1. depo extraction from the WC_FM_Sim pilot G4 files (apptainer + cvmfs stock dunesw v10_20_08d00)
wcfm/e1/run_extract.sh numu 20 ; wcfm/e1/run_extract.sh nue 20        # -> wcfm/work/e1/depos-{numu,nue}.tar.bz2
# 2. workspace events 101+ (jsons, per-event input depo files, docs/07_tables/events.md)
python3 scripts/e1_make_events.py
# 3. sim (same detector sim as events 1-100, depo file as the source), imaging + truth tiers, FM sidecar, graphs
for e in $(seq 101 140) $(seq 201 220) $(seq 301 320); do ./run_sim_depo_evt.sh 1 $e; done          # <= 6 parallel
for e in ...; do ./run_img_evt.sh -C -L 4 -O _sub4f 1 $e; ./run_fm_evt.sh -D gpu -O _fm 1 $e; done   # <= 4 parallel
python3 scripts/gnn_dataset.py --sub _sub4f --fm _fm --out /home/xqian/tmp/wcfm-gnn/graphs_e1 --jobs 8 101-140,201-220,301-320
# 4. the E1 probe (reference rows + null control from the doc 06 gun graphs in /home/xqian/tmp/wcfm-gnn/graphs_f)
python3 scripts/e1_texture_probe.py --graphs /home/xqian/tmp/wcfm-gnn/graphs_e1 --out docs/07_tables
```

Freshness: `local/lib/libWireCellImg.so` 2026-09-25 08:30:20 (built after the BlobDepoFill fix's source edit
06:21, toolkit `1e6b2905`), recorded per event in `sim-provenance.txt` (`libimg=`); no build was run.

## 1. The question and the inputs

Doc 06 §6 defined E1: a sample with charge texture and landmarks, and a model-free go/no-go — (a) the three-view
consistency statistic of doc 06 §3.1 rises from ≈ prior to > 2× prior on 0° slabs; (b) a 1-D alignment of q_U(u),
q_V(v), q_W(w) recovers the u↔v↔w correspondence to within one 4-wire cell over ≥ 80 % of the length. The owner
asked to prototype it on real physics rather than a synthetic textured generator: the 40 WC_FM_Sim pilot events
(GENIE + G4 in the FM's own detector, FD-HD 1x2x6), with Route 2 (CORSIKA cosmic depo sets, no LArSoft) as the
fallback if the `lar` environment was broken. It was not broken; Route 2 was not needed (§7).

**What the pilot files hold.** `/home/xqian/work/WC_FM_Sim/runs/{numu,nue}/g4_<flavor>.root` (20 events each,
2026-07-02, dunesw v10_20_08d00, `dune10kt_v6_refactored_1x2x6.gdml`) carry
`sim::SimEnergyDeposits_IonAndScint__G4` — post-recombination electron counts per G4 step with TrackID, PdgCode,
Time and MidPoint — plus MCParticles and MCTruth. The `sp_*.root` files drop the SEDs. The FM training packs
(doc 06 follow-up survey) carry only 2-D per-pixel truth and cannot label 3-D cells; the `*_smeared` runs reuse
the same G4 with smeared truth (a tru0-style leak) and are not used. uproot cannot decode the SED branch (its
strided and generic object readers both fail on `sim::SimEnergyDeposit`), so the extraction is a `lar` job.

**The extraction job** (`wcfm/e1/`): a two-node Wire-Cell graph inside the art module `WireCellToolkit`,
`wclsSimDepoSetSource{art_tag: IonAndScint, scale: -1, id_is_track: true}` → `DepoFileSink`
(`wcls-extract-depos-fdhd.jsonnet`), fcl `depo_extract_fdhd.fcl` with the refactored
`dunefd_1x2x6_simulation_services`, runner `run_extract.sh` (SL7 apptainer image
`fnal-dev-sl7:latest` = sha256 `2191aae0…`, stock `setup dunesw v10_20_08d00 -q e26:prof`; WC_FM_Sim's own
`setup_env.sh` is not sourced — it aborts without the custom larwirecell build, which a two-node stock graph does
not need). 20 events per flavour in 23 s (numu) and 2 min (nue), rc = 0 both. Traps recorded for the next reader:
`id_is_track` must be set explicitly (the pilot's own sim jsonnet used `false` = SED vector index); `scale: -1`
(electrons are negative in WCT); `depo_data_N` is a 0-based per-job counter (art event N+1); the file keeps 7 data
columns (t ns, q, x y z mm, dL, dT) and 4 info columns (G4 id, pdg, gen, child) — no energy; negative ids are the
dropped EM-shower daughters (`KeepEMShowerDaughters: false`) charged to the negated parent id.

**Verification of the tar** (`docs/07_tables/events.md`, column "ndepos"): the per-event depo counts equal the
20 "SimDepoSetSource got N depos" lines of the pilot's own `sp.log` (5 913 … 123 241 numu, up to 486 k nue);
t ∈ [0, 11] µs (GENIE at t = 0, no beam spread; the late tail is Michel / neutron-capture time); xyz inside
the 1x2x6 active box (x ± 363 cm, y −600…+274 cm, z 0…1393 cm); pdg ∈ {±13, ±11, ±211, 2212, nuclei};
ids ∈ [−12 456, 12 696].

## 2. Event construction (`scripts/e1_make_events.py`)

Real numu events have random muon angles (3°–38° to the anode plane in this sample, table below), but E1 needs
≥ 20 single-track 0° events. G4 physics is rotation invariant (LArSoft's recombination depends on dE/dx only), so
each numu event's depos are rigidly rotated about the primary vertex, about the axis x̂ × d̂_µ, so that the primary
muon's direction has zero drift-direction component (exactly 0°: events 201–220) and a second copy at 0.5°
(301–320). The natural orientations are kept (101–120 numu, 121–140 nue). The texture (Landau straggling, delta
rays), the landmarks (vertex, hadrons, Michels, showers) and the wrapped-wire geometry are therefore real; only the
orientation is chosen. Depos leaving the active box after rotation are dropped (|x| ≥ 3 cm and ≤ 362.5 cm,
|y| ≤ 600 cm, 0 ≤ z ≤ 1393 cm); the retained fraction of the primary's charge is ≥ 0.98 on every event.

Primary = the smallest positive G4 id among |pdg| = 13 (11 for nue); vertex = its earliest depo; direction =
end-to-end of its time-ordered depos; APAs to simulate = those holding ≥ 5×10⁵ electrons or touched by the
primary (ident = 2·⌊z / 232.39 cm⌋ + [y > 0], verified on the gun events). Each event gets
`events/000001_<evt>.json` with the keys every downstream script reads (`kind` = `nu` / `nurot`, `tracks[0]` =
the primary with `tail`, `head`, `angle_deg`, `length_cm`, `anodes`, `seed` = 1000 + evt) and
`work/000001_<evt>/e1-depos-in.tar.bz2` (one `depo_data_0` / `depo_info_0`, gen 0, electrons negative, t as
extracted). `iso_slab_probe.py` and `gnn_dataset.py` run unchanged on the new jsons; the doc 06 probe ignores
them (its populations are `kind == 'iso'`).

Event table: `docs/07_tables/events.md` (80 rows). Summary: primary muon path length 83 cm … 13.6 m (median
8.6 m); natural angle to the anode 2.9° … 38.2° (median 10.8°); 1–7 APAs per event; 5.9 k … 485 k depos.
The muon of every rotated event lies at a fixed drift coordinate, |x| ≥ 57 cm from the APA.

**The sim** (`wct-sim-depofile-nf-sp.jsonnet`, fork by duplication of `wct-sim-iso-track-nf-sp.jsonnet`; runner
`run_sim_depo_evt.sh`, single event only): `DepoFileSource → DepoSetDrifter{drifter: 'Drifter'} → DepoSetFanout
→ per anode [DepoTransform → Reframer → AddNoise → Digitizer → NF → SP → FrameFileSink] + DepoFileSink` (drifted
depos + priors). The per-depo `Drifter` cannot take an `IDepoSet`; `DepoSetDrifter` wraps the identical Drifter
component, so drift speed 1.6 mm/µs, DL 7.2, DT 12 cm²/s, lifetime 8 ms and the binomial fluctuation are those of
events 1–100 (the pilot's own LArSoft sim used 1.60563 / 4 / 8.8 / 10.4; we keep the wcfm values so that E1 is
comparable to events 1–100). The bagger's readout gate is not applied: depos drifting in after the 3 ms readout
fall outside every frame and every slice. Everything below the fan, and every output name, is byte-identical to
the gun sim, so `run_img_evt.sh`, `run_fm_evt.sh` and `gnn_dataset.py` ran unchanged. Compiled-config proof:
the compiled JSON of event 201 holds `DepoFileSource:e1deposrc → DepoSetDrifter:e1setdrifter →
DepoSetFanout:simfan` with the Drifter's `xregions` at ±39.5 / ±130.0 / ±3629.2 mm.

## 3. The chain run and what it cost

All 80 events: sim `run_sim_depo_evt.sh` rc = 0 (10–81 s, ≤ 0.8 GB each; 6 in parallel); imaging
`run_img_evt.sh -C -L 4 -O _sub4f` (4 in parallel) and the FM sidecar `run_fm_evt.sh -D gpu -O _fm` (3 s per
anode); graphs `gnn_dataset.py --sub _sub4f`. Imaging per anode-event: median 12 s and 0.5 GB, but a heavy tail
entirely made of the nue shower APAs — event 122 anode 2: 1188 s and 15.7 GB (628 k sub-blobs, 7.4 M blob-blob
edges); event 126 anode 2: 664 s and 16.2 GB; the rotated-numu anodes that E1 uses take 8–162 s and ≤ 10 GB.
The owner asked why. From the per-node `Timer:` lines of the 122/anode-2 log:

| component | wall |
|---|---|
| ChargeSolving, 6 rounds (3 × uniform + uboone) | 545 s |
| BlobDepoFill truth catchers (tru0 291 s on 485 k depos, tru 19 s) | 310 s |
| ProjectionDeghosting, 2 passes | 171 s |
| InSliceDeghosting, BlobClustering, LocalGeomClustering, BlobGrouping, sinks | ≈ 150 s |
| BlobCutting itself | 2 s |

BlobCutting at 4 wires turns each shower blob into 20–130 pieces ("in=4 → sub=78", "in=6 → sub=107" in the log),
so the whole legacy chain runs on a 10–30× larger graph whose measure count stays small (14 k): a few huge
per-slice components in which the cut pieces of one parent share wires, which is where the LASSO Gram matrix
approaches N² non-zeros and every coordinate-descent sweep is O(N²). Peak RSS is flat at 15–19 GB across a 10×
range of blob counts (53 k → 570 k; doc 05's 147 k-cell anode: 19.4 GB) while wall scales with blobs, which
points at a transient inside the largest per-slice solve rather than the graph size — inferred from the code
(`util/src/LassoModel.cxx` Gram build and sweep, `img/src/CSGraph.cxx` dense Eigen `A`, `R`), not measured:
per-component sizes are only logged at trace level. The truth-catcher time is the known unfixed quadratic of
`img/src/BlobDepoFill.cxx` (per slice, per depo, per wire, over every blob in the slice, ray crossings recomputed
per blob; img `efficiency-concerns.md` item 4), running twice on the full pre-deghost sub-blob set.

What the three production detectors teach (their imaging configs and perf docs, surveyed read-only): PDHD, PDVD
and SBND run the same node chain and the same solver settings as `wcfm/img.jsonnet` (BlobClustering, 2 ×
ProjectionDeghosting, 3 × [BlobGrouping, ChargeSolving uniform + uboone, LocalGeomClustering], 3 ×
InSliceDeghosting, GlobalGeomClustering); none uses BlobCutting, none has truth catchers; their measured cost is
6–22 s and < 1 GB per APA (PDHD), 5–11 s per CRP (PDVD), 8 s median per event (SBND). Their perf campaigns
already removed the imaging quadratics that show up first (BlobGrouping, once 78 % of the job; ProjectionDeghosting
projection eviction, −51 % RSS; several LASSO restructures, after which `clus/docs/imgclus-optimization-log.md`
declares the LASSO Fit at the byte-identity floor); on SBND the LASSO is 10–15 % of CPU, on the sub-blob tier
46 %. The tier changes which term dominates, not the algorithm. Two known, unfixed items are the ones this tier
hits: InSliceDeghosting's 2-view × 3-view loops (concern 1) and BlobDepoFill's depo × blob loop (concern 4). Doc
05 §9 had already recorded the `_sub4` tier as a research tier, not a production knob, for this reason.

Options, none taken here: (i) keep nue events off the cut tier (they serve E3, not E1) or cut them at 8 wires;
(ii) a default-OFF `truth_only` knob in `img.jsonnet` ending the active chain after the tru0 sink, with the legacy
"kept" context taken from the uncut apa tier by containment (−70 % time, most of the memory); (iii) index blobs
by wire in BlobDepoFill (truth-only component, gated byte-identical on the tru0 archives); (iv) a jemalloc sampling
profile of one nue anode to settle Gram-transient vs graph copies (`abtest/profile_img.sh` does not take the wcfm
TLAs; precompile with `wcsonnet`, profile the `wire-cell -c` step, M17).

## 4. Criterion (a): the three-view consistency statistic on real 0° slabs

Same statistic and code as doc 06 §3.1 (`e1_texture_probe.consistency`, copied from `iso_slab_probe.main`), on
the busiest slice of the primary muon's slab; the muon slab = cells > 30 cm from the vertex in the contiguous
run of slices around the busiest one holding ≥ 10 % of its real cells (an isochronous track spans 2–4 slices; a
real muon rotated to 0° end-to-end still wanders in x by centimetres through multiple scattering, so the exactly
isochronous population comes as pieces of 40–250 cm, each in its own 2–9 slices — table `consistency.md`,
column "slab slices"). The doc 06 gun rows re-computed by the same function reproduce doc 06 exactly (event 1
anode 10: 989 candidates, 69 real, prior 0.070, AP 0.087, CV 0.12 / 0.09 / 0.06).

| stratum | anode-events | AP / prior: min / median / max | share > 2× prior | events with ≥ 1 anode > 2× | median AP (prior) | wire-charge CV along the muon U / V / W (median) |
|---|---|---|---|---|---|---|
| gun 0° (doc 06 reference = null) | 13 | 1.04 / 1.36 / 2.46 | 1 / 13 | 1 / 7 | 0.092 (0.070) | 0.14 / 0.22 / 0.17 |
| gun 0.5° (doc 06 reference) | 5 | 1.17 / 2.97 / 6.77 | 4 / 5 | 3 / 3 | 0.180 (0.059) | 0.45 / 0.45 / 0.60 |
| **numu rotated 0°** | 87 | 0.85 / **3.70** / 19.7 | **68 / 87** | **19 / 20** | 0.125 (0.033) | 0.72 / 0.71 / 0.74 |
| numu rotated 0.5° | 87 | 0.58 / 3.52 / 23.6 | 73 / 87 | 20 / 20 | 0.132 (0.032) | 0.71 / 0.73 / 0.74 |
| numu natural (2.9°–38°) | 84 | 0.98 / 2.19 / 19.9 | 48 / 84 | 15 / 20 | 0.422 (0.203) | 0.86 / 0.88 / 0.86 |

Full rows: `docs/07_tables/consistency.md`; per event: `rot0_by_event.md`.

**Criterion (a) is met.** On real 0° slabs the statistic is at 3.7× the prior (median), above 2× on 78 % of the
anode-events and on 19 of 20 events, where the gun's constant-dQ/dx slabs sit at 1.36× (one of 13 above 2×).
The along-track wire-charge CV is 0.72 against 0.14–0.22: real Landau straggling and delta rays give every wire a
different charge, and the three views agree on it (the figures of §5 show the same spikes in U, V and W along the
truth line). The value is the same at 0.5° (3.5×) and lower for the natural tilted muons (2.2×, on a prior three
times higher because tilted tracks have far fewer ghosts per slice), so this is texture, not slice splitting.

## 5. Criterion (b): does the texture pin the u↔v↔w correspondence?

`e1_texture_probe` builds q_U(u), q_V(v), q_W(w) from the packed wire charge of the slab's slices indexed by
wire-in-plane on the muon's face (tru0 `wnodes` map wip ↔ channel), takes the truth correspondence as the
straight line fitted to the real muon cells' (u, v, w) centres (`bnodes` wire ranges), and slides 32-wire windows
along u in steps of 8 (interior texture only, no endpoints): for each window the offset of v = a + b·u is searched
over the slab's own V range (the deghosting ambiguity, 160–930 wires wide, so chance = 9 / range ≈ 0.03) with the
slope b from the truth (an oracle for the direction; the direction is what the hexagon geometry constrains, the
offset is what the texture must supply), scored by the Pearson correlation of the window with q_V resampled along
the hypothesis; a window is recovered if the predicted v at its centre is within one 4-wire cell of the truth.
A second, global search leaves direction and offset free over the whole muon piece (secondary; it is dominated
by the ends of the piece, which are landmarks, and its r-maximum is fragile — reported for completeness).

| stratum | anode-events | windows | recovered, pooled: U–V / U–W | chance | per anode-event median U–V / U–W | anode-events ≥ 0.5: U–V / U–W | ≥ 0.8 on both |
|---|---|---|---|---|---|---|---|
| gun 0° (null control) | 13 | 353 | 0.07 / 0.11 | 0.034 / 0.036 | 0.06 / 0.09 | 1 / 0 of 13 | 0 |
| **numu rotated 0°** | 84 | 2415 | **0.54 / 0.77** | 0.028 / 0.031 | **0.60 / 0.88** | 59 / 73 of 84 | 18 |
| numu rotated 0.5° | 84 | 2461 | 0.55 / 0.78 | 0.026 / 0.031 | 0.60 / 0.86 | 56 / 71 of 84 | 18 |

By muon-piece length (rotated 0°): < 150 u wires 0.74 / 0.90 (25 anode-events), 150–300 wires 0.47 / 0.69 (27),
> 300 wires 0.55 / 0.78 (32). Per event (`rot0_by_event.md`, pooled over the event's anodes): U–W ≥ 0.80 on 11 of
20 events and ≥ 0.69 on 16; U–V ≥ 0.60 on 9. The three weakest events are the shortest muon (211: 191 cm, 19
windows, 0.00 / 0.26) and the two 13-m muons whose vertex APAs carry hadrons in the same slices (201, 217: 0.33 /
0.51). Full rows: `alignment.md`; one figure per event (the anode with the longest piece): `e1-<evt>-anode<N>.png`
— top panel q_U(u) with q_V, q_W resampled along the truth line, bottom the busiest slice's cells in y–z.

**Criterion (b) is met on the collection pair and not, at the 80 % bar, on the induction pair.** The texture
alone recovers the U↔W correspondence to within one cell over 77 % of the windows (median anode-event 88 %) and
the U↔V one over 54 % (median 60 %), against 7–11 % on the gun slabs and 3 % chance. Both are far above the null:
source 2 works on real data. The U–V shortfall is a per-view fidelity effect rather than missing texture: U and
V are the two wrapped induction planes, whose channel charge sums the segments of both faces (the other face's
activity is added to q_U and q_V but not to q_W) and whose SP is the bipolar deconvolution; the U–W pair carries
one such plane, the U–V pair two. That is a statement about the sequences the association model will see, and it
argues for anchoring the association on W and using per-face wire nodes in E2, not against the approach.

## 6. Verdict and what it means for E2 / E3

**E1: GO.** The two go conditions of doc 06 §6 hold on real physics in the FM's own detector: the consistency
statistic rises from ≈ prior to 3.7× (a), and interior texture alone pins the cross-view correspondence far
above chance, meeting the one-cell / 80 % bar on U–W and reaching 60 % on U–V (b). The information that the doc
06 chain could not use *is in real data*; the gun's NO-GO was a verdict on the sample.

Context that changes the E2 controls (`context.md`): on the real 0° slabs the legacy chain keeps a median 5 % of
the true charge (gun: 6 %), and the doc 06 geometric rule — 0.60–1.00 charge recall on 10 of 13 gun slabs — drops
to a median 0.16 (r < 1.5 cm; 0.25 at r < 3 cm): real slabs are MCS pieces with delta rays and other particles in
the same slices, so the "chord of the hexagon" is no longer the answer, and control (i) of E2 is weak on real data.
The texture is what transfers. Consequences for E2: keep the projection loss (the predicted support must reproduce
the three wire sequences) as the central term, with the W sequence as the anchor and U/V sequences built per face;
the slab unit is the contiguous 2–9-slice run of §4, not the whole track; the continuity term must tolerate
delta-ray branches. For E3 the landmark sample now exists: 20 nue events (showers) and the 20 natural numu
events (vertices, hadrons, Michels) are imaged on the same tiers.

Caveats. The primary-muon direction is an oracle in the window test (the offset is the texture test, the
direction is left to geometry); the global free search is not a reliable estimator as coded (its r-maximum
picks wrong slopes on half the pieces). The rotation preserves G4 physics but not the detector's dependence on
the field direction beyond recombination (none in this chain). The drift parameters are the wcfm ones, not the
pilot's LArSoft values. Statistics: 20 rotated events, 84 anode-events, 2 415 windows for the 0° stratum.

## 7. Route 2 — not needed, recorded

The `lar` environment worked at the first attempt, so the fallback was not run. For the record: the PDHD CORSIKA
depo sets `DNN_ROI_SP/data/pdhd/generated/depos-evt{0..49}.tar.bz2` are raw un-drifted WCT depo files with G4
track id and pdg (≈ 0.9 M depos/event, q ≤ 0, t ± 3.2 ms) that the PDHD-native stageB driver
(`stageB_pdhd/wct-sim-nf-sp-dnnroi-pdhd.jsonnet`) runs unchanged, but no truth tier was ever built on them and the
FD-HD wcfm configs cannot take them (PDHD APAs at x = ±3.57 m, cathode at 0); the PDVD set
(`/home/xqian/work/data/pdvd/generated/`, 250 events) is unusable as is (78 % of depos carry q = +1).

## 8. Files

- `wcfm/e1/wcls-extract-depos-fdhd.jsonnet`, `e1/depo_extract_fdhd.fcl`, `e1/run_extract.sh` — the depo-extraction `lar` job.
- `wcfm/scripts/e1_make_events.py` — events 101–140, 201–220, 301–320 (jsons committed; `work/000001_<evt>/e1-depos-in.tar.bz2` and `work/e1/depos-*.tar.bz2` not).
- `wcfm/wct-sim-depofile-nf-sp.jsonnet`, `wcfm/run_sim_depo_evt.sh` — the depo-file sim driver and runner (fork by duplication; `wct-sim-iso-track-nf-sp.jsonnet` and `run_sim_evt.sh` untouched).
- `wcfm/scripts/e1_texture_probe.py` — criteria (a) and (b), context, figures; `--pops` / `--events` slice the run.
- `wcfm/docs/07_tables/`: `events.md`, `consistency.md`, `alignment.md`, `context.md`, `rot0_by_event.md`, 20 figures.
- Outputs (not committed): `wcfm/work/000001_{101..140,201..220,301..320}{,_sub4f,_fm}/`, graphs `/home/xqian/tmp/wcfm-gnn/graphs_e1/` (308), batch logs `/home/xqian/tmp/wcfm-e1/`.
- Code cited: `img/src/BlobCutting.cxx`, `img/src/BlobDepoFill.cxx:81-162,309-363`, `img/src/CSGraph.cxx:33-219,144-148`, `util/src/LassoModel.cxx:112-251`, `gen/src/DepoSetDrifter.cxx`, `sio/src/DepoFileSource.cxx`, larwirecell `SimDepoSetSource.cxx:220-229,272-283`.
