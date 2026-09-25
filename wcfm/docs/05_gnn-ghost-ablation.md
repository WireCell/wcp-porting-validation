# 05 — The GNN ghost ablation: charge-only vs charge + FM on the solver's factor graph

*2026-09-25. Follows doc 04 §5 (F5 NO-GO) and its consequence: re-pose the question at the ambiguity
scale, against the right baseline, on ≥ 100 events. This doc builds the sub-blob graph dataset (owner's
step 1) and trains one GNN twice, with and without the FM inputs (step 2). No toolkit code changes; no
production output changes; everything here is scripts, runs and a doc in `wcfm/`. **Round 1 (§8, same
day): the label, a second fold split and the doc 02 physics metrics — the charge-GNN result holds and gets
stronger on a clean label; the FM verdict stays NO-GO; the `tru0` "real" cells are 94 % diffusion-tail /
wrapped-wire artefacts holding 4.5 % of the charge.***

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/wcfm
# events 11..100 (random iso / overlay / cosmic, seeds 1000+evt; events 1-10 untouched, byte-identical)
python3 gen_iso_tracks.py --extend 100                       # -> events/000001_{11..100}.json, gnn_events.txt
for e in $(seq 11 100); do ./run_sim_evt.sh 1 $e; done      # NEVER `1 all`: it re-simulates 1-10 (rm -f the frames)
for e in $(seq 1 100);  do ./run_img_evt.sh -C -L 4 -O _sub4 1 $e; done   # 4-wire sub-blob tier + truth tiers
for e in $(seq 11 100); do ./run_fm_evt.sh -O _fm 1 $e; done              # FM sidecars (1-10 exist from doc 04)
python3 scripts/gnn_dataset.py 1-100 --sub _sub4 --fm _fm --out /home/xqian/tmp/wcfm-gnn/graphs --jobs 8
PY=/home/xqian/toolkit-dev/WC_FM_DINO/.venv-dino/bin/python   # torch 2.10 + sklearn; no torch_geometric needed
$PY scripts/gnn_train.py --graphs /home/xqian/tmp/wcfm-gnn/graphs --out /home/xqian/tmp/wcfm-gnn/abl_charge --arms charge --seeds 0,1,2 --device cuda:0
$PY scripts/gnn_train.py --graphs /home/xqian/tmp/wcfm-gnn/graphs --out /home/xqian/tmp/wcfm-gnn/abl_fm     --arms fm     --seeds 0,1,2 --device cuda:1
$PY scripts/gnn_train.py --graphs x --out /home/xqian/tmp/wcfm-gnn/ablation --merge /home/xqian/tmp/wcfm-gnn/abl_charge /home/xqian/tmp/wcfm-gnn/abl_fm
# round 1 (sec 8): track-cell label, second fold split, physics metrics + views
G=/home/xqian/tmp/wcfm-gnn
$PY scripts/gnn_train.py --graphs $G/graphs --out $G/abl_charge_q1k --arms charge --seeds 0,1,2 --label qmin:1000:ig --device cuda:0
$PY scripts/gnn_train.py --graphs $G/graphs --out $G/abl_fm_q1k     --arms fm     --seeds 0,1,2 --label qmin:1000:ig --device cuda:1
$PY scripts/gnn_train.py --graphs x --out $G/ablation_q1k --merge $G/abl_charge_q1k $G/abl_fm_q1k
$PY scripts/gnn_train.py --graphs $G/graphs --out $G/abl_charge_f1  --arms charge --seeds 0,1,2 --fold-seed 1 --device cuda:0
$PY scripts/gnn_train.py --graphs $G/graphs --out $G/abl_fm_f1      --arms fm     --seeds 0,1,2 --fold-seed 1 --device cuda:1
$PY scripts/gnn_train.py --graphs x --out $G/ablation_f1 --merge $G/abl_charge_f1 $G/abl_fm_f1
python3 scripts/gnn_physics.py --graphs $G/graphs --out $G/physics     --run charge=$G/abl_charge     --run fm=$G/abl_fm     --plot 1:10,6:all,8:all,9:9
python3 scripts/gnn_physics.py --graphs $G/graphs --out $G/physics_q1k --run charge=$G/abl_charge_q1k --run fm=$G/abl_fm_q1k --plot 1:10,6:2,8:11
python3 scripts/gnn_physics.py --graphs $G/graphs --out $G/physics_f1  --run charge=$G/abl_charge_f1  --run fm=$G/abl_fm_f1
```

The batch caps were 6 (sim), 5 (imaging, `-P` over events, one process per anode inside) and 4 (FM).
Runs were launched with the load already at 8–12 from a colleague's SBND campaign and stayed below 20.
Toolkit `c58501b8` / `local/lib` of 2026-09-24 21:24 (doc 04 §3.2), wcp `a58170b3` + this commit.

## 1. Why the F5 probe could not answer the owner's question

The owner's question (2026-09-25): *3-D blobs are built on merged wires and are therefore ambiguous;
at the sub-blob level (each wire/strip crossing combination) we want to judge which is true and which
is ghost.* Doc 04 §5 posed exactly that label (sub-blob, `tru0`) but

1. at a **20-wire cut**, where per-view charge consistency alone decides the label (AP 0.987), not at
   the crossing scale where the three strips of a ghost carry the same charge as its real neighbours;
2. against **fifteen charge summaries**, not against the chain that makes the decision in production
   (charge solving + deghosting), so the cells the chain gets wrong were never isolated;
3. with a **linear readout on isolated cells**. The ghost decision is a graph property — a ghost's
   strips are the strips of real cells in the same slice, and the evidence against it is that their
   charge is already explained — which no per-cell probe can see, and which is what a GNN with
   strip-sharing edges can learn.

So the gate is re-posed inside the GNN: the same network with and without the FM inputs, on the
solver's own factor graph, scored on the cells the legacy chain decides wrongly.

## 2. The dataset (`scripts/gnn_dataset.py`, owner's step 1)

### 2.1 Sample

`gen_iso_tracks.py --extend 100` adds events 11–100 to the 10 hand-listed ones (§0): per event the
seeded RNG draws the kind — 30 % single iso track (θ ∈ {0, 0.5, 1, 2, 3, 5, 10}°), 40 % iso overlay of
2–4 tracks at the same drift distance (the ghost factory of doc 02 §2), 30 % cosmic (1–2 ordinary
tracks) — then the geometry exactly as for events 1–10. `gnn_events.txt` lists the kinds
(iso1 34, iso2 17, iso3 12, iso4 13, cosmic1 10, cosmic2 14). Events 1–10 are byte-identical to doc 02
(md5 checked before/after the extend). Simulation: 19 s and 0.66 GB per event.

### 2.2 Tiers

The sub-blob tier is `run_img_evt.sh -C -L 4 -O _sub4`: `BlobCutting` with `length_threshold` 4
(`max_depth` 10, `min_length` 2), so a cell is at most a few wires wide per view — the crossing scale
(the 20-wire `_sub` tier of doc 03 is kept, read-only). Same chain otherwise, same truth tiers:
`tru0` = every sub-blob after `BlobClustering` with its `BlobDepoFill` true charge (ghost ⇔ 0),
`apa` = the legacy chain's survivors after charge solving and deghosting. On event 1 (θ = 0, the doc 02
slab) the 4-wire cut gives 2.4 × the sub-blobs of the 20-wire cut (anode 10: 42 625 vs 17 566), 47 s
and 1.3 GB per anode. The legacy chain's decision per cell = present in `apa` with `val > 0`
(joined by the geometric key face/slice/strips; idents are shared too).

### 2.3 Graph

One graph per (event, anode), the charge solver's factor structure:

| element | content |
|---|---|
| blob node | one sub-blob: the 15 charge summaries of doc 04 §5 (per view n_ch, n_pix, log Σq, log mean q; pairwise log ratios), the 3 × 128 mean-pooled FM descriptor, label (`tru0`), legacy decision, face/slice/ident |
| wire node | one (plane, channel, slice) **pixel** covered by ≥ 1 strip: packed charge (0.25 × Σ₄ ticks, doc 04 §1), plane, and the 128-d FM row at that pixel (zero when the FM saw no active pixel) |
| bw edge | blob covers wire, one per strip wire (Σ over a wire's blobs is the solver's measurement constraint) |
| bb edge | `BlobClustering`'s blob–blob edge (adjacent slices, all-plane overlap), symmetrised |

The wire node *is* the FM's unit (a pixel), so the FM enters where it was computed and the network
pools it itself; the blob-level pooled descriptor is kept for continuity with doc 04. `bbedges` of the
cluster file are cross-slice node indices (checked: 0 % same-slice). Files: `/home/xqian/tmp/wcfm-gnn/graphs/graph-000001_<evt>-anode<N>.npz`
(not committed; 3.6 s per event to build).

## 3. The GNN and the protocol (`scripts/gnn_train.py`, owner's step 2)

Plain torch (2.10, `.venv-dino`), no torch_geometric — a new dependency is a team decision and none
is needed: `index_add_` / `scatter_reduce_` do the message passing.

* Inputs. **charge arm**: blob = 15 charge numbers; wire = log1p q + plane one-hot. **fm arm**: the
  charge arm inputs + the 128-d FM row on every wire node + the 384-d pooled FM on every blob node.
  Standardised per feature on the training folds.
* Network (`FactorGNN`, h = 64, L = 3 rounds): wire ← {mean, max} over the blobs covering it (+ log
  degree); blob ← mean over its wires per plane (U, V, W kept separate) and mean over its bb
  neighbours; residual two-layer MLPs with LayerNorm; readout one logit per blob = P(ghost). The
  charge arm is a learned charge solver on the solver's own graph; the fm arm differs only by inputs.
  ≈ 100 k parameters (charge) / 130 k (fm).
* Training: Adam 1e-3, 25 epochs, one graph per step, BCE. 5 folds over **events**, stratified by
  kind × n_tracks, one fixed assignment (`FOLD_SEED 0`) for both arms and all seeds; for outer fold k
  the next fold is the inner validation fold that picks the epoch (best validation AP on the hard
  population); the 5 held-out folds are pooled. 3 seeds per arm.
* Metrics (ghost = positive, as in doc 04): pooled held-out AP and AUC on all cells, on the **hard
  population** (cells the legacy chain decides wrongly: kept ghosts + dropped real cells), and per
  kind; the legacy chain's own precision/recall as the reference operating point.

**Pre-registered (script header and here, before the first run):** the fm arm beats the charge arm
when, over the 3 seeds, its mean pooled held-out AP on the hard population is higher by **≥ +0.02**
and the gap exceeds **twice the larger seed standard deviation**. Anything else is NO-GO for F4 on
these features.

## 4. Results

### 4.1 The sample and the legacy chain at the crossing scale (`05_tables/dataset.md`)

| sample | events | anode-events | sub-blobs | ghost fraction | legacy kept | legacy precision | legacy recall | hard cells (FP + FN) | ghost fraction in hard |
|---|---|---|---|---|---|---|---|---|---|
| all | 100 | 261 | 5 196 639 | 0.405 | 276 034 | 0.684 | **0.061** | 2 991 746 | 0.029 |
| iso (1–4 tracks) | 76 | 202 | 4 701 895 | 0.408 | 198 978 | 0.630 | 0.045 | 2 732 055 | 0.027 |
| cosmic (1–2 tracks) | 24 | 59 | 494 744 | 0.374 | 77 056 | 0.825 | 0.205 | 259 691 | 0.052 |

Wire nodes 1.70 M, blob–wire edges 48.7 M, blob–blob edges 49.1 M; blobs per graph median 6 805, max
166 151 (event 88 anode 0). 7.9 % of the sub-blobs have a view without an FM pixel (a strip over
channels the packing left inactive); 6.3 % of the wire nodes have no FM row.

The first finding is the legacy chain itself: at the 4-wire cut it keeps 5 % of the cells with precision
0.68 and **recall 0.06** — it drops 94 % of the real cells on iso events and 80 % on cosmics (doc 03 §6
saw the same at 20 wires as "captured charge 0.27–0.72"). So the *hard population* — the cells the chain
decides wrongly — is 3.0 M cells, 97 % of them dropped real cells; at the crossing scale the production
question is "which real cell to keep", not "which ghost to remove". Costs of the tiers on this box:
sim 26 s / 0.64 GB median per event; imaging with the 4-wire cut median 8 s / 0.48 GB per anode but
up to 460 s / **19.4 GB** (event 46 anode 3, 147 k sub-blobs: the legacy solver on a 4-wire cell set
is the memory hog, not the FM); FM sidecar 3 s / 1.2 GB per anode-event.

### 4.2 The ablation (`05_tables/ablation_summary.md`, `ablation_result.json`)

Pooled held-out scores over the 5 folds, mean ± sd over 3 seeds; ghost = positive class:

| population | n | ghost frac | charge arm AP | charge AUC | fm arm AP | fm AUC |
|---|---|---|---|---|---|---|
| all cells | 5 196 639 | 0.405 | 0.9746 ± 0.0015 | 0.9777 | **0.9847 ± 0.0008** | 0.9887 |
| **all, hard** (pre-registered) | 2 991 746 | 0.029 | **0.8412 ± 0.0112** | 0.9832 | **0.8412 ± 0.0138** | 0.9873 |
| iso | 4 701 895 | 0.408 | 0.9756 ± 0.0013 | 0.9783 | 0.9848 ± 0.0010 | 0.9887 |
| iso, hard | 2 732 055 | 0.027 | 0.8381 ± 0.0125 | 0.9830 | 0.8425 ± 0.0153 | 0.9881 |
| cosmic | 494 744 | 0.374 | 0.9659 ± 0.0030 | 0.9727 | 0.9842 ± 0.0005 | 0.9894 |
| cosmic, hard | 259 691 | 0.052 | **0.8906 ± 0.0118** | 0.9870 | 0.8634 ± 0.0082 | 0.9839 |

Per-event AP median 0.987–0.989 for both arms, 10th percentile 0.94 (charge) / 0.96 (fm). Wall 6 min
(charge) / 7.3 min (fm) per seed on one RTX 4090, 7 GB peak; the picked epochs spread over 5–24, so 25
epochs is enough but not generous.

**Verdict on the pre-registered gate: gap = +0.0001 on the hard population (0.8412 vs 0.8412),
margin +0.02 → NO-GO.** The FM inputs do not help on the cells the legacy chain gets wrong.

### 4.3 Operating points against the legacy chain (`05_tables/operating_points.md`)

The GNN scores turned into a keep/drop decision (3-seed mean logit), all 5.2 M held-out cells:

| decision rule | precision (kept cells real) | recall (real cells kept) | iso recall | cosmic recall |
|---|---|---|---|---|
| legacy chain (charge solving + deghosting) | 0.684 | 0.061 | 0.045 | 0.205 |
| charge GNN, P(real) > 0.5 | 0.920 | 0.971 | 0.970 | 0.983 |
| charge GNN at recall 0.95 | 0.940 | 0.950 | 0.948 | 0.969 |
| charge GNN at recall 0.90 | 0.962 | 0.900 | 0.896 | 0.933 |
| fm GNN, P(real) > 0.5 | 0.954 | 0.963 | 0.963 | 0.968 |
| fm GNN at recall 0.95 | 0.964 | 0.950 | 0.950 | 0.954 |
| fm GNN at recall 0.90 | 0.983 | 0.900 | 0.899 | 0.908 |

(At the legacy precision 0.684 both GNNs reach recall 1.000 — every real cell scores above the 55 % most
ghost-like ghosts — so that row is uninformative and is omitted.)

## 5. Readings

1. **The GNN on the solver's factor graph is the result.** With nothing but the 15 charge numbers per
   cell and the wire charges, message passing over shared strips keeps 97 % of the real 4-wire cells at
   precision 0.92, where the legacy chain keeps 6 % at 0.68. On the hard population (97 % dropped real
   cells) its AUC is 0.98. This is the charge-conservation reasoning the solver does, learned on the same
   graph, without the whole-track deletions of doc 02 §5 / doc 03 §6. It does not involve the FM.
2. **The FM inputs are a small, real gain on the easy cells and nothing on the hard ones.** All-cell AP
   0.985 vs 0.975 is outside the seed spread (sd 0.001), and at fixed recall 0.95 the fm arm has 40 %
   fewer ghosts among the kept cells (precision 0.964 vs 0.940). On the hard population the gap is
   0.000 ± 0.02, and on cosmics the fm arm is *worse* (0.863 vs 0.891) — the doc 04 §5 pattern again
   (the linear probe lost 0.15–0.2 AP on cosmics). Reading: the per-plane descriptors encode texture that
   separates obvious ghosts (isolated, low-charge crossings) a little better than four charge numbers
   per view do, but not the cross-view consistency that decides a dropped real cell against its ghost
   neighbours sharing the same strips. The doc 04 zero-shot result (tri-view disagreement AUC 0.51) says
   the same from the other side.
3. **What the gate does and does not say.** It is one architecture (h 64, L 3), one training budget and
   one label definition (`tru0`, any true charge in the cell); the fm arm has 26 × (blob) / 33 × (wire) the input width and was
   given no extra capacity or epochs, and its validation peaks were higher on every fold — it fits the
   easy cells faster. A bigger network or a cross-view attention block could change the all-cell number;
   nothing here suggests it would change the hard-population one, because the information the hard
   cells need (which of two strip-sharing cells owns the charge) is in the graph, not in the pixels.

**Consequence for the campaign.** F4 (the FM sidecar join into the imaging chain) stays on hold: the
FM is not what the ghost decision needs. The GNN is worth building on charge alone, and the FM can be
re-tested as an optional input the day a cross-plane objective (doc 01 §2.1, doc 35 #8) moves the
zero-shot tri-view AUC off 0.5. The build order that follows from §4.3: (a) the charge GNN as a
toolkit stage after `BlobCutting` (the graph is exactly `IBlobSet` + the slice's wire charges, so the
input is available in-process; TorchScript export of `FactorGNN` is plain `index_add_`); (b) a
sub-blob keep/drop knob, default OFF, whose knob-on output replaces `ProjectionDeghosting` +
`InSliceDeghosting` on the sub-blob tier; (c) the doc 02 iso baseline (captured charge, ghost fraction
among survivors) re-measured with it.

## 6. Open items

- **Label definition** — resolved in §8.1: `tru0` is not a usable target for the toolkit stage; train on
  `qmin:<Q>:ig` (track cells vs pure ghosts, tail band ignored) or on a charge regression.
- **The tail cells and `BlobDepoFill`** (§8.1): 2.9 M cells with 0 < q_true < 1000 e, most of them > 10 cm
  from any track cell, in W-wire columns and wrapped-wire images of the track. Whether the fill rule leaks
  through the U/V wrap (same wire index, other segment) or the tiled blob geometry does is not settled here;
  it decides what "true charge in a cell" means on FD-HD and needs a look before `tru0`/`tru` are used as a
  per-cell regression target. Doc 02's "relative threshold" caveat was this.
- **Fold sensitivity** — checked in §8.3: the `tru0` verdict moves with the split (one charge seed
  collapses on fold seed 1), the `qmin` verdict does not.
- **Legacy solver memory at the 4-wire cut.** 19.4 GB and 460 s on a 147 k-cell anode; the `_sub4`
  tier is a research tier, not a production knob, until that is addressed (the GNN stage would replace
  the solver on those cells, not run after it).
- **Cross-view input.** The natural next model-side test is not a bigger FM but a *pair* feature: for
  two cells sharing a strip, the FM rows of the shared pixels are identical by construction, so any
  cross-view signal must come from the other two strips — a contrastive objective over `bw` edges
  (same deposit vs ghost) is the right pre-training target, and this dataset is its training set.
- The `--extend` events reuse `make_event`'s overlay rule (same drift distance) for cosmics too, so
  "cosmic2" events are overlapping cosmic pairs — intended as harder controls, but not the doc 02
  single-cosmic control; `cosmic1` (10 events) is that.

## 7. Files

wcp (`4df84a67`): `wcfm/gen_iso_tracks.py` (`--extend N`, `random_spec`), `wcfm/events/000001_{11..100}.json`,
`wcfm/gnn_events.txt`, `wcfm/scripts/gnn_dataset.py`, `wcfm/scripts/gnn_train.py`, `docs/05_*.md`,
`docs/05_tables/{dataset.md, dataset_summary.txt, ablation_summary.md, ablation_result.json, operating_points.md}`,
`docs/README.md`. Round 1 (this commit): `scripts/gnn_train.py` (`--label`, `--fold-seed`, ignore mask),
`scripts/gnn_physics.py`, `docs/05_tables/{physics_tru0.md, physics_q1k.md, physics_f1.md, ablation_summary_q1k.md,
ablation_result_q1k.json, ablation_summary_f1.md, ablation_result_f1.json, ap_real_q1k.md, view-*.png}`. Work products (not committed): `work/000001_{11..100}/` (sim), `work/000001_*_sub4/`
(imaging + truth tiers, 4-wire cut), `work/000001_{11..100}_fm/` (sidecars); scratch
`/home/xqian/tmp/wcfm-gnn/` (graphs 2.1 GB, `abl_charge/`, `abl_fm/`, `ablation/`, logs). Toolkit: no
change (`c58501b8`).

## 8. Round 1 — the label, a second split, and the physics metrics

Owner's step 1 of 2026-09-25: harden the charge-GNN result before any C++. Three checks; all runs 3 seeds,
same folds and epochs as §3; 6–7.5 min per seed.

### 8.1 What `tru0` calls "real"

The true charge of the 3.09 M `tru0`-real cells (`BlobDepoFill` `val`, electrons):

| q_true | share of "real" cells | share of true charge | legacy chain keeps |
|---|---|---|---|
| (0, 1) | 20.3 % | 0.0 % | 5.5 % |
| [1, 10) | 20.1 % | 0.1 % | 4.6 % |
| [10, 100) | 42.1 % | 1.7 % | 4.4 % |
| [100, 1000) | 11.2 % | 2.7 % | 7.1 % |
| [1000, 10 000) | 3.3 % | 13.0 % | 24 % |
| [10 000, 100 000) | 3.0 % | 73.4 % | 22 % |
| ≥ 100 000 | 0.1 % | 9.0 % | 3.6 % |

**93.7 % of the `tru0`-real cells hold < 1000 e and together 4.5 % of the charge.** They are not the
cells of a track: in the views (`05_tables/view-000001_{1-anode10,6-anode2,8-anode11}.png`, orange) they
form W-wire columns spanning the whole slab and straight-line / triangular *images* of the track hundreds
of cm away — event 6 anode 2 has them at y ≈ −600 cm while both tracks of the event lie at y ∈ [−127, 41]
(the second track is on anode 3, y > 0): wrapped-wire images. Quantified with the cell centres: on event 1
anode 10, 87 % of the tail cells (88 % of their charge) are > 10 cm from any track cell; event 6 anode 2
89 %; event 3 anode 4 66 %; event 8 anode 11 42 %; the cosmic event 9: 0 %. Median tail charge 15–85 e
against ~10⁵ e for a full 4-wire crossing.

Consequences: (i) the §4 hard population (97 % "dropped real cells") was 90 % these cells, so the
pre-registered §4.2 number measured how well each arm recovers diffusion-tail / wrap-image cells, which is
not the owner's question; (ii) the sensible label is **track cell (q_true ≥ 1000 e) vs pure ghost
(q_true = 0), tail band ignored** — `gnn_train.py --label qmin:1000:ig` (2.30 M labelled cells, 194 353
track cells, 8.5 % real); (iii) how `BlobDepoFill` puts 15–85 e into a cell 500 cm from the track is an
open item (§6) — doc 02's "the ghost label must be a relative threshold" was this effect seen from the
other side.

### 8.2 The doc 02 physics metrics of the keep sets (`scripts/gnn_physics.py`, `05_tables/physics_*.md`)

Charge recall = Σ q_true(kept) / Σ q_true; track-cell recall = fraction of q ≥ 1000 e cells kept; ghost
fraction = kept cells with q_true = 0; keep ⇔ P(real) ≥ 0.5 (3-seed mean logit), all 100 events held out.

| rule (training label) | cells kept | charge recall | track-cell recall | ghost fraction of kept | ghost share of kept measured charge | tail cells kept |
|---|---|---|---|---|---|---|
| legacy chain | 5.3 % | **0.175** (iso 0.110, cosmic 0.429) | 0.229 | 0.316 | 0.029 | 0.05 |
| charge GNN (`tru0`) | 62.8 % | 0.9992 | 0.999 | 0.080 | 0.028 | 0.97 |
| fm GNN (`tru0`) | 60.1 % | 0.9987 | 0.999 | 0.046 | 0.022 | 0.96 |
| **charge GNN (`qmin:1000:ig`)** | **25.2 %** | **0.982** | **0.989** | **0.008** | 0.008 | 0.38 |
| fm GNN (`qmin:1000:ig`) | 40.2 % | 0.993 | 0.990 | 0.003 | 0.005 | 0.65 |

Per event (doc 02 events; `physics_q1k.md`): the legacy chain's charge recall is 0.02–0.34 on the eight
iso events and 0.74–0.89 on the two cosmics; the `qmin` charge GNN is ≥ 0.977 on every event except the
three-track overlay event 7 (0.953; fm 0.985), with ghost fraction 0.000–0.015 except the two-track
overlay event 6 (0.146; fm 0.058). Views: on the doc 02 slab (event 1 anode 10, 42 625 cells, 181 track
cells) the legacy chain keeps 4 track cells and 586 ghosts; both `qmin` GNNs keep all 181 and **0**
ghosts. On the four-track event 8 anode 11 the charge GNN keeps 1 619 / 1 627 track cells and 0 of the
38 746 ghosts (fm: 1 620 and 36) — and 13 044 of the 30 425 tail cells, versus 22 354 for the fm arm: the
tail band, which neither label constrains, is where the two arms differ most.

### 8.3 The gate on the clean label, and the second split

| run | label | fold seed | charge arm hard AP | fm arm hard AP | gap | verdict |
|---|---|---|---|---|---|---|
| §4 | `tru0` | 0 | 0.8412 ± 0.0112 | 0.8412 ± 0.0138 | +0.0001 | NO-GO |
| round 1 | `tru0` | 1 | 0.7956 ± 0.0582 (seeds 0.837 / **0.713** / 0.836) | 0.8542 ± 0.0106 | +0.059 | NO-GO (gap < 2 × sd 0.058) |
| round 1 | **`qmin:1000:ig`** | 0 | **0.9980 ± 0.0007** | **0.9991 ± 0.0001** | +0.0011 | NO-GO |

On the clean label the hard population is 237 k cells (87 k pure ghosts the chain kept + 150 k track cells
it dropped; the chain's own precision/recall there 0.338 / 0.229) and both arms separate it almost
perfectly. With the track cell as the positive class (`ap_real_q1k.md`): AP 0.9935 ± 0.0012 (charge) vs
0.9956 ± 0.0008 (fm), AUC 0.9992 vs 0.9996; on the hard cells 0.9994 vs 0.9996. The fm gain is outside the
seed spread and one-tenth of the margin.

The `tru0` verdict is split-sensitive (one charge seed collapses on fold seed 1 while the fm arm is stable,
sd 0.011 vs 0.058) — on a target that is mostly tail cells, the FM's per-pixel texture is a steadier cue
than four charge numbers per view. That is consistent with §5 reading 2 and does not change the verdict.

### 8.4 What round 1 changes in §5

- Reading 1 is **confirmed and stronger**: on track cells vs pure ghosts the charge-only GNN reaches
  AUC 0.999, charge recall 0.98 with 0.8 % ghosts among the kept cells, against the legacy chain's 0.175
  with 32 % ghosts. This is the number to build the toolkit stage on, trained with `qmin:<Q>:ig` (or a
  charge regression), not `tru0`.
- Reading 2 stands: the FM adds a small, consistent, sub-margin gain (fewer ghosts at fixed threshold,
  +0.01 charge recall on the overlay events 6/7, AP(real) +0.002) and keeps more of the tail band. F4 stays
  on hold; the FM remains an optional input for the stage.
- New: the tail band (2.9 M cells, 4.5 % of q_true, spatially artefacts) needs a decision before training
  the production stage — the charge solver's answer is "drop them" (they get ~0 solved charge), and the
  `qmin` charge GNN drops 62 % of them unasked. Deciding it in the truth tier (`BlobDepoFill` fill rule,
  §6) is cleaner than deciding it in the loss.
