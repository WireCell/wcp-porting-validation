# doc pr/52 — How the SBND neutrino vertex is determined, and what can be improved without retraining the DL model

Status: **reference + proposal** (no code changed in this round).
Written 2026-08-08 in answer to three owner questions, given that hand scans
show a sizable set of events with correct vertices and some with wrong ones,
and that no DL training pipeline is currently in hand:

1. Is there room to fine-tune the DL vertex model directly?
2. What is the post-DL vertex-ranking code, and can it be improved for SBND?
3. When does the DL vertex fall back to the traditional vertex — does the
   traditional path also need tuning?

Short answers: (1) not by touching the weights today, but the model *class*
ships in-tree and the weights are a plain state_dict, so a light fine-tune
is a labels-and-compute problem, not a missing-code problem — and every
near-term lever is on the inference/acceptance side; (2) yes — SBND runs a
toolkit-only "rerank" scorer that was explicitly designed to be tuned
per-detector, and it was last tuned on 36 annotated events; (3) the
traditional path is never dormant — it generates *all* candidates, always
runs per-cluster, and takes over globally whenever the DL declines — so it
is worth tuning too, and some hand-scan failures are candidate-set failures
that no vertex-selection tuning can fix (doc pr/51 class).

## Repro block

No runs are needed; every claim is a code citation, verified at
toolkit `apply-pointcloud` HEAD `ba5bbe59` on 2026-08-08:

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
sed -n '1030,1052p' clus/src/TaggerCheckNeutrino.cxx        # DL-first / fallback wiring
sed -n '3847p;4104,4169p;4194,4199p' clus/src/NeutrinoVertexFinder.cxx   # rerank scorer
sed -n '684p;1724,1738p' cfg/pgrapher/experiment/sbnd/clus.jsonnet       # SBND operating point
sed -n '1,30p;55,90p' pyutil/python/SCN_Vertex.py           # inference glue
```

Prior docs this builds on: pr/4 (DL adoption), pr/2 §2e/§7.4/G3 (gap
analysis), pr/32 (vertex-ID port audit), pr/48 (kProtectedBreak veto),
pr/50 (vertex_kink_snap), pr/51 (near-vertex graph pathologies).

---

## 1. The pipeline map — where the vertex comes from (answers Q3)

The vertex decision has **two stages**, and the DL only ever touches the
second one.

### Stage 1 — per-cluster (always runs, DL or not)

`determine_main_vertex` (`clus/src/NeutrinoVertexFinder.cxx:2992`) runs for
the main cluster and each surviving other cluster inside
`TaggerCheckNeutrino`:

- Candidate generation: every PR vertex without an inward-pointing strong
  segment (`examine_main_vertex_candidate`, :352) and, in the mixed
  track/shower branch, with ≥1 attached track (:3086).
- Pruning: `examine_main_vertices_local` (:3330) drops back-to-back
  two-track topologies (>165°/muon, >170°/protons) and substitutes the far
  end of a continuation-muon chain.
- Ranking: `compare_main_vertices` (:873) — an **additive scorer** ported
  term-for-term from the uBooNE prototype (`NeutrinoID_track_shower.h:1589`):
  proton in/out topology (−(n_in−n_out)/4), upstream-z prior
  (−Δz / `vertex_z_prior_scale`; **SBND=100 cm** vs uBooNE 200,
  `sbnd/clus.jsonnet:1094`), +1/8 per shower (+1/8 long-daughter bonus),
  +1/4 per track, +1/4 clear proton (else +1/8 directed track), +1/8
  longest muon >35 cm, **+0.5 fiducial volume**, −conflicts/4. Strict
  argmax.
- All-showers clusters use a different, non-additive procedure
  (`compare_main_vertices_all_showers`, :436): PCA extremes → local Steiner
  fit → shower direction, with "smaller z wins" as the fallback at every
  failure point.

### Stage 2 — global (DL-first, traditional as fallback)

`TaggerCheckNeutrino.cxx:1030-1052` is the whole story:

```cpp
if (!m_dl_weights.empty()) {
    flag_dl_changed = pattern_algos.determine_overall_main_vertex_DL(...);
}
if (!flag_dl_changed) {
    final_main_vertex = pattern_algos.determine_overall_main_vertex(...);
}
```

The DL (`NeutrinoVertexFinder.cxx:3847`) is **not a score term inside the
traditional ranking — it is an alternative selector for the global stage**.
Two properties matter for interpreting scan failures:

- **The DL never creates a vertex.** Its top-K voxels are snapped to the
  nearest *existing* ProtoVertex; the candidate set is every vertex in the
  PR graph (:3880-3893, wider than stage 1's filtered list). If the true
  vertex is not a PR-graph vertex, no amount of DL confidence can select
  it — that failure class belongs to doc pr/51 (graph robustness), not to
  vertex selection.
- **When the DL accepts, the traditional global ladder is skipped
  entirely** (`examine_main_vertices`, `check_switch_main_cluster`,
  `compare_main_vertices_global` never run). The DL path then does its own
  post-processing: main-cluster swap if needed, `examine_direction`,
  short-stub proton re-tag, long-muon cleanup (:4246-4299).

### The fallback routes to the traditional vertex

The traditional `determine_overall_main_vertex` (:4322) runs whenever ANY
of these hold — this is the complete list:

| # | route | evidence in the log |
|---|-------|---------------------|
| 1 | `dl_weights` empty (geometric arm, all identity gates) | none — silent by design |
| 2 | toolkit built without Python (`HAVE_PYTHON_INC` unset — not the case in this build) | **none at all** |
| 3 | weights path fails `Persist::resolve` | WARN `dl_weights path not found` (`TaggerCheckNeutrino.cxx:245`) |
| 4 | SCN import/inference throws (the libpython LD_PRELOAD trap, doc pr/4 §3) | WARN `DL vertex failed: ...` (:4313) |
| 5 | rerank winner scores below `dl_vtx_min_accept_score` | TRACE `rerank rejected ... staying with traditional vertex` (:4206) |
| 6 | pr/48 veto: traditional main vertex carries `kProtectedBreak` (`two_end_break` on) | TRACE `keeping protected two-end-break vertex over the DL choice` (:4237) |

All six produce rc=0. Route 3 is **not** caught by the standard
`grep -c "DL vertex failed"` check — worth remembering when auditing a
batch. Routes 5 and 6 are *by design* and are expected to fire on a healthy
fraction of events.

**Answer to Q3 in one sentence:** the traditional path always builds the
candidates and always ranks them per-cluster, and it decides the global
winner on every event where the DL declines or is vetoed — so yes, the
traditional ranking is a live tuning target, not a legacy remnant.

Finally, regardless of which branch won, the pr/50
`snap_main_vertex_to_kink` and a final `improve_vertex` run afterwards
(`TaggerCheckNeutrino.cxx:1075-1086`) — the selected vertex is still
locally refined downstream.

---

## 2. The DL model itself — what "fine-tuning directly" would take (answers Q1)

### What it is

- Architecture: `DeepVtx` (`pyutil/python/SCN/DeepVtx.py`) — a
  SparseConvNet 3-D submanifold UNet, planes [16,32,64,128,256], per-voxel
  2-class sigmoid output, collapsed to `score = sigmoid(vtx) − sigmoid(bg)`
  ∈ [−1,+1] (`SCN_Vertex.py:82`).
- Weights: a **plain PyTorch state_dict**,
  `wire-cell-data/uboone/scn_vtx/t48k-m16-l5-lr5d-res0.5-CP24.pth` (27 MB,
  uBooNE-trained). Not TorchScript; loaded and cached per-process by
  `SCN_Vertex.py:_load_model`.
- Runtime: embedded CPython inside wire-cell (`pyutil/src/SCN_Vertex.cxx`),
  CPU-only, `torch.set_num_threads(1)`, ~1 s/event.
- Input: the PR-graph vertices + segment interior fit points in **cm**,
  charge feature `dQ × dQdx_scale + dQdx_offset` (defaults 0.1 / −1000 — a
  ported prototype "hack for now", see §4 Tier B), voxelized at **0.5 cm**
  after subtracting the cloud minimum. The min-subtraction makes the input
  translation-invariant — the stated justification (doc pr/4 §5) for
  running uBooNE weights on SBND at all.
- Output: top-K `[x,y,z,score]` voxels (K=5 in production).

### Can the model be tuned directly, today?

**Not by editing weights, and there is no knob that changes what the net
computes.** But the situation is better than "no training code, no
options":

1. **Everything needed for a light fine-tune is already in-tree** except
   the training loop: the model class, the exact voxelization
   (`SCN_Vertex.py:voxelize`), and the weights as an initializable
   state_dict. A fine-tune script is ~100 lines of torch (load state_dict,
   per-voxel cross-entropy with the true-vertex voxel positive, low
   learning rate, freeze early planes if the sample is small). What is
   genuinely missing is **labels in the right form**: the true vertex 3-D
   position per event, matched to the same PR point cloud the net sees.
   This is a *medium-term* option — see §5 item 4 for what the current scan
   campaign should record to enable it.
2. **The near-term levers are all on the inference/acceptance side** — and
   that is not a consolation prize. The raw net score is only one of seven
   terms in the selector that actually picks the vertex (§3), and that
   selector — not the net — was the thing last calibrated, on 36 annotated
   uBooNE-era events. Re-calibrating it on the owner's SBND scan labels is
   the same *kind* of operation as fine-tuning, applied at the decision
   layer instead of the representation layer, and it needs no GPU and no
   new training code.
3. Full SBND retraining stays **gap G3** (doc pr/2) — open, acknowledged,
   and out of scope here.

---

## 3. The post-DL ranking — the part that is designed to be tuned (answers Q2)

SBND pins `dl_vtx_rerank=true` (`sbnd/clus.jsonnet:1729-1738`), so
production runs the **rerank branch** (`NeutrinoVertexFinder.cxx:3986-4211`)
— a toolkit-only addition with **no prototype counterpart** (doc pr/32 §P2).
Mechanics:

1. Ask the net for the top-K=5 voxels with scores.
2. Snap each voxel to its nearest ProtoVertex; dedup, keeping the higher
   score; iterate in graph-index order (deterministic argmax).
3. Score every snapped candidate with a 7-term composite (:4132-4169):

| term | formula | weight source |
|------|---------|---------------|
| `s_dl` | `dl_score × dl_vtx_score_scale` (=1000) | **config knob** |
| `s_snap` | `−min(2.0, snap_dis / 5 cm)` | constexpr `W_SNAP_MAX/W_SNAP_L` |
| `s_fwd_z` | `−0.25 × clamp((z−min_z)/400 cm, 0, 1)` | constexpr `W_FWD_Z` |
| `s_clen` | `+2.0 × min(1, L_host/60 cm)` | constexpr `W_CLEN/W_CLEN_L` |
| `s_isol` | `−2.0` if host <6 cm and not main | constexpr `W_ISOL` |
| `s_main` | `+2.0` if host == main cluster | constexpr `W_MAIN` |
| `s_fv` | `+0.5` if inside fiducial volume | constexpr `W_FV` |

4. Accept the argmax iff `best_score ≥ dl_vtx_min_accept_score` (=4.0)
   (:4198). **There is no distance gate in this mode** — `dl_vtx_cut` is
   used only by the legacy (`rerank=false`) branch, i.e. it is **dead in
   SBND production** even though it is threaded through the jsonnet.

Design intent, straight from the code comments (:4104-4110, :469):

> *Empirical weights tuned on 36 annotated events (2026-04-15). … The three
> active geometric signals (main, clen, isol) dominate when DL is uncertain
> (scores ~0.005); DL dominates when confident (scores >0.1).*
> *min_accept_score: … correct uncertain-regime picks score 8-12, failure
> cases 3-5.*

So the composite is calibrated such that with a typical uncertain net score
~0.005, `s_dl ≈ 5` is comparable to `s_main + s_clen = 4` — geometry can
carry an uncertain DL, and a confident DL (score > 0.1 ⇒ `s_dl` > 100) can
overrule geometry. **This scorer was already tuned once on annotated
events; the owner's SBND scan labels are exactly the input needed to tune
it again, this time on SBND.**

The currently reachable knobs (all `TaggerCheckNeutrino` config, per-value
defaults at `TaggerCheckNeutrino.cxx:462-469`):

- `dl_vtx_min_accept_score` (4.0) — the DL-vs-traditional decision
  boundary. The single highest-leverage number in this doc.
- `dl_vtx_score_scale` (1000) — how much net confidence counts vs geometry.
- `dl_vtx_top_k` (5) — how many candidates get a second look.
- `dQdx_scale` / `dQdx_offset` (0.1 / −1000) — the net's charge input
  calibration (§4 Tier B).
- The seven `W_*` weights are **constexpr** — retuning them requires a
  small default-OFF knob round first (§5 item 2c).

Caveat: only `dl_weights` and `dl_vtx_cut` are TLAs of
`wct-pr-perevt.jsonnet`; the four `dl_vtx_*` rerank knobs reach production
through the pinned values in `sbnd/clus.jsonnet:1729-1738`, so A/B-ing them
today means editing that pin (or a follow-up round adds TLAs).

---

## 4. The tunable surface, tiered by leverage and cost

**Tier A — DL acceptance operating point (config-only, highest leverage).**
`dl_vtx_min_accept_score`, `dl_vtx_score_scale`, `dl_vtx_top_k`. The 4.0
threshold and 1000 scale encode a calibration done on 36 uBooNE-era
annotated events with uBooNE weights; nothing about them is SBND-derived.
The owner's scan labels support a proper operating-point choice (§5).

**Tier B — the net's charge input (config-only, cheap sanity check).**
`dQdx_scale=0.1, dQdx_offset=-1000` maps dQ into the feature range the
uBooNE net saw in training. SBND's dQ scale differs from uBooNE's; if the
mapped values sit outside the training distribution, the net's confidence
is systematically distorted *before* any threshold matters. Two actions:
(i) histogram the actual `dQ×0.1−1000` values fed to the net on a few SBND
events and compare against the uBooNE-era expectation; (ii) pin the two
keys explicitly at the SBND `tagger_check_neutrino` call site — today they
silently inherit common defaults, the exact hazard doc pr/4 §1 pinned the
`dl_vtx_*` four against (already flagged in doc pr/2:924).

**Tier C — the traditional scorer (fallback + per-cluster stage).**
Reachable now: `vertex_z_prior_scale` (SBND already 100 vs uBooNE 200) and
`muon_dqdx_curve`. Not yet reachable: the `compare_main_vertices` term
weights (1/4, 1/8, 0.5, the 35/45 cm lengths) are hardcoded — a faithful
prototype port, never revisited for SBND's shorter detector and different
track-length distributions. A knob round exposing them (default = current
constants, byte-identical off) would let the DL-declined subset be tuned on
the same scan labels.

**Tier D — candidate-set quality (not a vertex-selection problem).**
Both selectors can only choose among PR-graph vertices. The pr/51
pathologies (duplicated corridors, charge-less bridges, micro-stubs, and
the true-corner candidate losing by a hair) corrupt or displace the
candidate set itself. For any scanned failure, the first question is
whether the true vertex *existed* as a candidate — if not, the fix is the
pr/51 `main_vertex_graph_audit`, not tuning.

---

## 5. Proposed program: turning the hand-scan labels into improvements

The unifying idea: **classify every mislabeled event by which route (§1
table) chose its vertex and whether the true vertex was in the candidate
set.** That classification alone tells us which tier fixes it.

1. **Vertex scoreboard (first code round, small, default-OFF).** A
   diagnostic knob on `TaggerCheckNeutrino` that logs, per event: the DL
   top-K voxels with raw scores; each snapped candidate's seven composite
   terms and total; the accept/reject decision and route taken (§1 table
   row); the traditional winner and its `compare_main_vertices` score when
   that path runs; and the final vertex position. One parseable line block
   per event. (Some of this is extractable from existing TRACE logging, but
   a structured dump makes the joins with scan labels mechanical.)

2. **Join with the scan labels and split the failures:**
   - (a) *DL had it right but was rejected* (true vertex = a top-K snap,
     score 3–4 range) → lower/retune `dl_vtx_min_accept_score` /
     `dl_vtx_score_scale`. The defaults comment says correct uncertain
     picks score 8–12 and failures 3–5 *on uBooNE-era annotations* — the
     SBND version of that ROC is exactly what the scoreboard + labels give.
   - (b) *DL wrong and accepted* → raise the threshold, or re-fit the
     composite weights (needs the Tier-A/§5.2c knob round to expose `W_*`).
     With ~equal counts of (a) and (b) events, a simple grid over
     (threshold, scale) maximizing corrected-minus-broken is enough; no ML
     machinery needed.
   - (c) *Both selectors wrong, true candidate existed* → Tier C: retune
     the traditional terms on the DL-declined subset.
   - (d) *True vertex never a candidate* → pr/51 graph audit territory;
     exclude from tuning fits so they don't drag the operating point.

3. **Tier B calibration check** (no labels needed, an afternoon): compare
   the SBND net-input charge distribution against uBooNE's expectation;
   decide whether `dQdx_scale/offset` need SBND values; pin them at the
   SBND call site either way.

4. **Start recording true vertex positions in the scans.** Today's scan
   verdicts are per-event correct/incorrect. A clicked 3-D true-vertex
   position (Bee supports this) per event upgrades the same effort into:
   (i) route classification without ambiguity, (ii) distance-resolved
   acceptance curves, and (iii) **the training labels a future in-tree
   DeepVtx fine-tune needs** (§2.2.1) — making the medium-term option real
   without waiting for a full G3 retraining campaign.

5. **Ops hardening (bookkeeping, near-zero cost):** add a batch check for
   the un-grepped `dl_weights path not found` route; note that
   `dl_vtx_cut` is dead under rerank (either retire it from the SBND
   threading or leave a comment); consider promoting the four `dl_vtx_*`
   knobs to `wct-pr-perevt.jsonnet` TLAs so operating-point A/Bs don't
   require editing the pin.

Suggested order: 3 (cheap, independent) → 1 → 2a/2b (config-only wins) →
2c/Tier C knob round → 4 feeding the medium-term fine-tune.

---

## 6. Open items and stale-doc notes

- **Stale in-tree docs** (flagged, deliberately not fixed in this round):
  `clus/docs/vertex_determination.md:230` documents the legacy
  `min_dis < dl_vtx_cut` acceptance — code that does not execute under the
  SBND production config — and none of the rerank machinery;
  `clus/docs/porting/neutrino_id_function_map.md:178` lists the DL
  signature without the four rerank parameters. Doc pr/4's cited line
  numbers (`:3217`, `:3659`) have drifted to `:3847` / `:4322`.
- `wct-pr-perevt.jsonnet:10` header still claims `dl_weights` "stays off" —
  contradicted by the doc-pr/4 flip at `:636`.
- Adoption evidence for DL-on remains one event (pr/4 §5); the scoreboard
  program above is also the missing systematic validation.
- Gap G3 (SBND retraining) stays open; §5.4 defines what to collect now so
  it is actionable later.
