# pr/42 — dQ/dx panel for the PR event display

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 <out> data 388 256587
./pr_display/serve_pr_display.sh 5017 <out>/pr_evt*/calib-pr-evt*.json
#   ssh -L 5017:localhost:5017 wcgpu1.phy.bnl.gov
#   open http://localhost:5017/pr_display_viewer
```

Gate labels for the byte-identical claim below: `work-dqdx-off2` (no
`pr_display`) vs `work-dqdx-on3` (`PR_EXTRA_STAGES=pr_display`, this change).

## What this adds

Click a particle-flow row and its measured dQ/dx appears in a new panel,
plotted against the muon/proton/pion/kaon reference curves (tracks) or the
1×/2× MIP lines (showers), so a track called "electron" can be checked by
eye rather than trusted from the score alone.

**Direction depends on particle kind, per owner instruction**: for a track
the interesting end is the stopping end (Bragg rise vs. residual range); for
an EM shower the stopping end means nothing — the interesting part is the
**beginning** (the stem near the vertex), where 1 vs 2 MIP separates e⁻ from
a converted γ. The panel auto-picks **End** for a track node, **Start** for a
shower node, with a manual override toggle and a segment dropdown for
stepping through a multi-segment shower.

**The six 2-D wire-plane panels are hidden by default** (not deleted —
`--wire-planes` on `serve_pr_display.sh`'s underlying `bokeh serve` command
line restores them; construction and data-filling are unchanged either way,
only the layout row is conditional).

## Where the data comes from

Everything the panel needs was **already** in `calib-pr-evt<ID>.json`:
`segments[].points[]` carries raw `dQ` (electrons), `dx` (cm) and `rr`
(residual range, cm) — `PrDisplayDump.cxx` `fit_json()` /
`dump_graph()`. So the C++ change is small additive fields, not a new data
path:

| field | where | what |
|---|---|---|
| `dqdx_ref` (top-level) | `PrDisplayDump::dump_dqdx_ref()` | five dQ/dx-vs-residual-range reference curves (muon/proton/pion/kaon/electron), sampled from the same `ParticleDataSet::get_dEdx_function()` instances the PID uses, on a fixed 0–100 cm grid (step 0.25 cm). Units `e/cm`. Dumped per event even though the tables are static, so an old arm's JSON keeps its own templates. |
| `meta.mip_dqdx_median` / `meta.mip_dqdx_flat` | `dump_meta()` | 43000 / 50000 e/cm, the two MIP reference scales (shower-stem normalization and `do_track_comp`'s flat template). Display-only constants — deliberately **not** read from the taggers' own members (`NeutrinoPatternBase.h` stores `43000/units::cm`, `TaggerCheckNeutrino.h` stores `43000.0` — not the same internal representation, so mirroring either would be arbitrary). |
| `segments[].particle_score` / `.dir_weak` / `.length` / `.start_vertex_id` / `.end_vertex_id` | `dump_graph()` | plain getters, already computed by `segment_determine_dir_track`. `length` also fixes the PF table's `L (cm)` column, previously blank for track rows (only `showers[].total_length` populated it). |
| `showers[].stem_dqdx` | `dump_showers()` | the literal ≤20-sample `Shower::get_stem_dQ_dx()` output (MIP units) the nue/single-photon taggers cut on. |

All read-only: no PID/direction function is invoked from the dump, only
plain getters and the already read-only `get_stem_dQ_dx` (verified by reading
its full body including the ≤3-hop multi-segment walk — it only reads graph
state and appends to a local vector).

**`particle_dataset` resolution — the one non-obvious wiring detail.**
`PrDisplayDump` resolves the `ParticleDataSet` component itself
(`NeedParticleData`-style, but with `Factory::find_maybe_tn` instead of the
throwing mixin, so a pipeline that runs `pr_display` without
`tagger_check_neutrino`/`tagger_check_stm` ahead of it just gets `dqdx_ref`
omitted, not a hard failure). **The resolution default must be the full
`"Type:Name"` string `"ParticleDataSet:ParticleDataSet"`, not the bare type**
— `Factory::find_maybe_tn` parses the config string with
`String::parse_pair`, and a colon-free string resolves to an empty instance
name, which does not match the registered instance's actual name. This was
caught empirically during verification (the WARN fired even though the
component's own construction log line showed `ParticleDataSet` present in
the same compiled job) — the identical bare-type default sits unnoticed in
`ClusteringFuncsMixins.h`'s `NeedParticleData` mixin too, masked there
because every existing caller (`TaggerCheckNeutrino`, `TaggerCheckSTM`)
overrides it with an explicit `wc.tn(particle_dataset)` jsonnet string. No
jsonnet change was needed to fix this — the C++ default was wrong, not the
config.

## Unit convention — read this before trusting a number

Both the measured points and the reference curves are **e/cm**, so they sit
on the same axis directly — `dQ/dx = points[].dQ / points[].dx` (dx is
already in cm; `PrDisplayDump.cxx` `fit_json()`).

**The template dump does *not* divide by `units::cm` a second time.** This
was the one real bug caught during verification: the muon plateau
`particle_dataset.jsonnet`'s header documents as 54657.7 e/cm at rr=59.5cm
came out as 5465.8 (exactly 10× low) with an extra division mirrored from
`do_track_comp`. `do_track_comp` needs that division for a reason that does
not apply here — its own data side (`fits[i].dQ / fits[i].dx`) uses `dx` in
**internal** length units (not pre-divided by cm), so its comparison scale is
e/cm scaled down by `units::cm`=10 to match. The dump's per-point dQ/dx
already uses `dx` pre-divided by cm, so it is already plain e/cm; copying
`do_track_comp`'s division divides the template but not the data it's
compared against. Fixed by calling `fn->scalar_function(rr_cm)` with no
further scaling. Verified by reproducing the documented plateau value exactly
after the fix, and by cross-checking a stopping proton segment (evt 256587,
seg 11080): measured dQ/dx near rr=0 was ~230k e/cm, between the proton
template (~262k) and well above the muon template (~168k) — the right order
of magnitude and the right ranking, which the 10×-low version would not have
shown (templates would have sat near 5–26k, absurdly below the data).

## Start-mode orientation

Distance-from-start is **not** the dumped `rr`, which is defined toward the
stopping end. It's recomputed client-side from `points[].x/y/z`: integrate
arc length in `fits()` order, then orient by whichever end (first or last
point) sits nearer `showers[].start`, tie-broken toward the segment's own
`fits()[0]` when no shower row exists (e.g. a track node viewed in Start mode
via the manual override).

## End-mode `rr` — a plan-stage mistake worth recording

The original plan draft proposed "correcting" `rr` for `dirsign == -1`
segments by recomputing `L.back()-L[i]`. **That would have been wrong** and
was caught before implementation: `PrDisplayDump.cxx` already orients `rr`
correctly for both `+1` and `-1` — a `-1` direction means the stopping end
sits at `fits[0]`, so `rr = L` (increasing from `fits[0]`) *is* the residual
range there. Verified on evt 256587: a `dirsign +1` segment's `rr` runs
49.17→0.0 and a `dirsign -1` segment's runs 0.6→24.57 — both correctly put
the stopping end at `rr = 0`. Only `dirsign == 0` is genuinely ambiguous
(never observed in the events sampled here); the panel falls back to raw arc
length from `fits()[0]` and flags it in the caption.

The dump's `rr` and the PID's own internal `end_L` differ by `+0.15cm` minus
a 0 or 1cm offset (`PRSegmentFunctions.cxx segment_do_track_pid`'s
`offset_length`) — negligible except right at the Bragg peak; noted in the
panel caption rather than "fixed" (fixing it would mean re-deriving the
PID's exact offset per call site, not a display concern).

## `stem_dqdx` markers — a known imprecision

The Start-mode panel overlays the shower's `stem_dqdx` samples (converted
back to e/cm) as diamonds at the **local segment's own point positions**,
index-matched and capped to whichever list is shorter. This is exact when
`get_stem_dQ_dx`'s walk stayed within the plotted (start) segment — the
common case for a short stem. If the walk had to continue into a downstream
segment (only when the start segment has fewer than 20 points), the tail
diamonds drift off the local x-axis. That's a faithful picture, not a
rendering bug: those samples really did come from a different segment's
points; reproducing the exact multi-hop walk client-side was judged not
worth the complexity for a diagnostic overlay.

## Gate

Byte-identical (`abtest/hash_archive.py`, member-content hashes, not raw
`md5sum` — CLAUDE.md M2), `work-dqdx-off2` (no `pr_display`) vs
`work-dqdx-on3` (`pr_display` on, this change), two shower-rich events (388:
14 showers, 256587: 27 showers) — the load-bearing check, since `stem_dqdx`
calls into `PR::Shower`:

| output | evt 388 | evt 256587 |
|---|---|---|
| `mabc-pr.zip` | `33fe37c4…` 7 members, **PASS** | `f924b361…` 7 members, **PASS** |
| `pctree-pr-evt<ID>.tar.gz` | `8e946950…` 425 members, **PASS** | `deb4115c…` 425 members, **PASS** |
| `nusel-evt<ID>.tsv` | identical | identical |

`./build/clus/wcdoctest-clus`: 98/98 test cases, 1016/1016 assertions,
unaffected (`pr_display` has no dedicated test file today — coverage is this
gate plus the JSON-content checks above).

## Status

Shipped in the `pr_display` diagnostic stage only — not a reconstruction
output. `calib-pr-evt<ID>.json` exists solely when `PR_EXTRA_STAGES=pr_display`
names the stage, so no default-OFF knob is needed under CLAUDE.md §1.

## Where to go next

- Per-field JSON schema, panel-by-panel: doc pr/26 (predates this addition;
  this doc's "Where the data comes from" table is the delta).
- PR chain overview + this stage's place in it: doc pr/41 §4.
