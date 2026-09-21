# doc sbnd_xin/118 — the PDHD/PDVD trajectory becomes SBND production, and the cross-detector PR profiling campaign opens

**Status: FLIPPED** (toolkit `675fd266`). SBND production now runs the `charge_stepped` retile sampler, the Steiner
`prefer3` admission with base-graph pricing `alpha 0.5 / tree+path` at both passes, and the two
lattice fit keys `fit_weight_pow 1.5` / `assoc_cont_center 1` — i.e. doc 116's `tfull`, the
configuration PDHD and PDVD production already run (except `charge_stepped`, which PDVD adopted and
PDHD did not; see sec 9.2). Toolkit `cfg/pgrapher/experiment/sbnd/{clus.jsonnet,
sbnd_track_fitting.json}`; production reference `sbnd_xin/ref/prod-2026-09-20/`.

**This flip overrides a frozen rule, and this doc says so rather than rewriting it.** Doc 116
sec 15.4 recommended against it; doc 117 sec 11's ADOPT table returned HOLD. Sec 1 states what
changed, what did not, and on whose word.

**Doc 115 is now the PRE-flip baseline.** `docs/115_*` and `products/d115/` describe the
configuration this round replaced. They are records, not current production. Their repro blocks —
and docs 116's and 117's — pin `prod_cfg_gate.py --ref ref/prod-2026-09-17b`, which from today
reports the 3-artifact drift of sec 4. That is **correct behaviour, not a failure**: those rounds
ran before the flip. Point the gate at `ref/prod-2026-09-20` to check the tree as it is now.

---

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# pin: ~/tmp/d115-libsnap, libWireCellClus.so md5 71ebd5aeb3868cbc0803f2fa2654b46b, toolkit 0a2807f4
#      == local/lib -- the flip is cfg only, read through WIRECELL_PATH, so no rebuild and the
#      binary is bit-for-bit the one doc 116 ran.

# V0, BEFORE any edit: no inherited drift
python3 scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-17b            # PASS 21/21

# G2  the production fit JSON is the measured arm's, key for key
python3 scripts/d118/tf_key_gate.py            > docs/118_figs/118_gate_tfkeys.txt

# G3 + G5  what the flip changes in the compiled config, and that the LArSoft 1-step chain sees it
bash     scripts/d118/cfg_diff.sh              > docs/118_figs/118_cfg_diff.txt

# G1  the decisive one: re-run stage B at the flipped DEFAULT and compare to the doc-116 tfull arm
JOBS=10               bash scripts/d118/stageB_flip.sh cv    f000
JOBS=10               bash scripts/d118/stageB_flip.sh nuecc f000 f001 f002
JOBS=10 OFF_N=20      bash scripts/d118/stageB_flip.sh off
JOBS=8 ARM_SUFFIX=rep bash scripts/d118/stageB_flip.sh cv    f000      # the determinism control
python3 scripts/d118/hash_gate.py cv nuecc off > docs/118_figs/118_gate_hash.txt
python3 scripts/d118/root_gate.py --arms work-r3cv-d118flip work-r3cv-d118fliprep "control"

# G4 + G7 + G8  the tripwire: name the drift, then cut the new reference
python3 scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-17b  > docs/118_figs/118_drift.txt
cp -a ref/prod-2026-09-17b ref/prod-2026-09-20
for n in sbnd pdhd pdvd; do echo "0000...0  ${n}_track_fitting.json" >> ref/prod-2026-09-20/consumers.sha256; done
python3 scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-20 --refresh
python3 scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-20            # PASS 25/25
```

---

## 1. The decision, stated as what it is

### 1.1 What the frozen rules said

| record | verdict | why |
|---|---|---|
| doc 116 sec 15.4 | *"Do I recommend changing the SBND baseline? No — not on this evidence."* | eight primary metrics all "not separable"; CPU and memory paid; 10–13 % churn |
| doc 117 sec 11, ADOPT table (`117_pred.sha256` `1a7570c4…`, frozen before the first arm) | **HOLD** | adopt only if a *primary* metric improves or the package is cheaper; neither held |

Both stand as written. Neither is amended by this round.

### 1.2 What changed between them, and what did not

Doc 116 sec 15.4 was written before sec 16.3 existed. Its own item 4(a) named what would change the
answer — *"the vertex-choice re-tune against the new trajectory … the chooser is where a gain would
come from"* — and doc 117 ran exactly that. The evidence moved:

| measurement (νe, 1 626 true interactions in the FV unless noted) | baseline `c2` | `tfull` `t2` | p |
|---|---:|---:|---:|
| true vertex within 1 cm | 599 (36.8 %) | **669 (41.1 %)** | **0.00027** |
| within 2 cm | 913 (56.2 %) | 958 (58.9 %) | 0.023 |
| median nearest-candidate distance | 0.960 cm | **0.840 cm** | 0.00026 |
| **selected** νe signal's own vertex, > 4 cut (n 562) | 0.850 cm | **0.780 cm** | **0.00026** |
| **selected** νμ signal's own vertex, `cv` (n 348) | 0.830 cm | **0.760 cm** | **0.017** |
| trajectory closure R2D_W, `nuecc` / `cv` | 2.13 / 0.81 % | 1.44 / 0.54 % | −32 % / −33 % |

with **no primary metric degraded**, the 1 cm gain **dose-responsive** in the knobs (single knobs
≈ 0: `cs` +0.8 p 0.52, `p3bw` −0.2 p 0.87; both +2.5 p 0.027; both plus the fit keys +4.3), and the
effect present on two independent constructions of the vertex and, on the selected-signal
construction, in **both** samples.

What did **not** change is the rule. Doc 117 pre-registered the vertex-distance metric as
**secondary**, and a secondary metric cannot promote a package under that round's own ADOPT table.
Doc 117 sec 10.2 said the conversion is a replication with the metric declared primary.

### 1.3 The owner's decision

> *"Let's flip as the SBND production run … the decision comes from better track trajectory as well
> as better vertex accuracy."* … *"Here flip means to go against your earlier recommendation, but I
> feel we see enough benefits."* — 2026-09-20

This is an explicit override of a frozen rule, of the same kind as the PDHD flip recorded in
`pdhd_track_fitting.json`'s `_comment_d108_fit_lattice_knobs` (*"THIS IS D2, NOT D1 … The owner
overrode that rule explicitly"*). The override is recorded in the config comments the next reader
will actually meet, not only here.

### 1.4 What remains unconfirmed

Stated so the flip is not read as a replication that happened:

1. the 1 cm gain is **one** sample at p 0.00027 and one (`cv`) at p 0.078 — same sign, not
   separable;
2. on `cv` the same column is slightly negative at 3–5 cm;
3. the multiplicity of doc 116 sec 16.3's table is 50 tests; p 0.00027 and 0.00026 survive a
   Bonferroni 0.001, the `cv` values do not;
4. doc 117 sec 10.2's independent-νe replication is **still the open item** (sec 10). Post-flip it
   is a validation rather than a decision, because `t2` is now what production runs.

---

## 2. What was flipped — exactly `tfull`, and why no subset

| # | component | before | after | where |
|---|---|---|---|---|
| (a) | retile sampler for the Steiner cloud | `'stepped'` | `'charge_stepped'` | `clus.jsonnet:2162` |
| (b) | Steiner blank-plane admission + base-graph pricing, **both** passes | keys omitted (C++ `'wcp'` / 0 / `'tree'`) | `'prefer3'` / `0.5` / `'tree+path'` | `clus.jsonnet:2276,2278,2279` and `:2303,2305,2306` |
| (c) | lattice fit keys | C++ defaults 2.0 / 0 | `fit_weight_pow 1.5`, `assoc_cont_center 1` | `sbnd_track_fitting.json` |

Unchanged and load-bearing: `steiner_blank_plane_radius` stays null; `dl_vtx_dual_chain=true`,
`dual_chain_mode='snap'`, `dual_chain_transfer_max=2.0`, `fit_exclusion=true` are untouched. Doc 117
sec 10.2 item 3 — flip the trajectory **with** the 2-step chain, never without: `t1x` (this
trajectory with `fit_exclusion=false`) is the worst cell of that grid, M8 −1.17 pt, p 0.016.

**No subset was graded, so none is available.** The gain is super-additive and the single-knob cells
are ≈ 0. In particular there is no `p3bw + fit keys without cs` cell anywhere in docs 116/117, so
the cost table below must not be read as an invitation to drop `charge_stepped` to recover the
CPU — that would be a configuration nobody has measured. The way to pay for `charge_stepped` is
Part B, not a partial adoption.

### 2.1 The escape hatch

Each component reverts independently, one line each:

```
-S retile_sampler_strategy='stepped'
-S steiner_blank_plane_mode='wcp'
-S steiner_base_weight_blank_alpha=0 -S steiner_base_weight_scope='tree'
# (c): delete the two keys from cfg/pgrapher/experiment/sbnd/sbnd_track_fitting.json
#      -- C++ defaults are 2.0 and 0, so removing them restores the pre-flip fit exactly.
```

### 2.2 Where the flip was placed, and why it matters

At the **call sites in `clus.jsonnet`**, in the `f9665bea` form (`if X == null then <new> else X`),
not at the `wct-pr-perevt.jsonnet` TLA defaults.

The LArSoft **1-step chain** (`sbnd/wcls-img-clus-matching-xin.jsonnet:282,290`) calls
`clus_maker.pr()` through the generated `sbnd/pr-operating-point.jsonnet`, which carries 22 knobs
and **none of these four**. A TLA-only flip would have left LArSoft production on the pre-flip
trajectory, silently — `clus.jsonnet:1005` records precisely that trap for `flash_by_gid`:
*"a TLA-only flip would have left LArSoft on the defect."* Sec 3.4 is the measurement that the
placement worked. No regeneration of the wrapper was needed.

Read that narrowly: G5 shows the two chains agree **on these four knobs**, not that the generated
wrapper is otherwise current. Its header pins its sync to toolkit `0ad642235` and names a
`scripts/compile-both.sh` acceptance test that does not exist in this tree. That is why the chain
itself is now a gate artifact — see sec 3.5.

---

## 3. Gates

### 3.1 G1 — output equivalence against the measured arm (the decisive one)

Two of the three components never enter the compiled config, so **no config-hash argument can prove
this flip**: `load_trackfitting_config` reads the fit JSON at runtime with a plain `ifstream`, and
`Trun.op_config_sha256` is computed from the wcsonnet output. The file says so itself
(`_comment_diffusion`: *"a byte-identical compiled config does NOT mean the fit is unchanged"*).

So the gate is at the outputs. `scripts/d118/stageB_flip.sh` re-runs stage B **at the flipped
production default** — the doc-116 `tfull` stage-B invocation minus exactly `PR_EXTRA_TLA` and
`SBND_TRACKFIT_JSON`, and nothing else — and `hash_gate.py` compares every product against
`work-r3*-d116tfull`, archive members by content (M2).

| sample | events | files compared | differ: provenance/timing | differ: **anything else** |
|---|---:|---:|---:|---:|
| `cv` (f000) | 18 | 82 | 28 | **0** |
| `nuecc` (f000–f002) | 24 | 120 | 48 | **0** |
| `off` (20 gates) | 20 | 82 | 21 | **0** |

**How much of the flip each sample exercises.** Component (c) is read by `TaggerCheckSTM` /
`TaggerCheckNeutrino`, so an event that never reaches a fit does not test it. Events whose
`TaggerCheckNeutrino:pr` ran above 50 ms (doc 116 sec 14's criterion): `cv` **10 of 18**,
`nuecc` **24 of 24**, `off` **1 of 20** — 35 of the 62. The `nuecc` arm is what carries the fit
keys; `cv` and `off` mostly test (a) and (b) and the plumbing. Configuration identity for (c) does
not depend on runtime at all — it is G2 plus the compile-sha match below — but "62 events" should
not be read as 62 exercises of the fit.

`pctree-pr-evt*.tar.gz`, `mabc-pr.zip` and `nusel-evt*.tsv` are **identical member for member on
every event of all three samples.** Every difference that exists is one of exactly two things:

- `Trun`'s four provenance branches — `op_config_sha256`, `toolkit_git`, `trackfitting_config`,
  `wcp_git` — i.e. the record of *where the configuration came from*, which is the one thing this
  flip is designed to change; every other tree of `tracking-pr.root` (`T_tagger` 1 242 branches,
  `T_kine`, `T_cluster`, `T_bundle`, `T_flash`, `T_proj`, `T_proj_data`, `T_rec_charge`,
  `T_bad_ch`) is identical;
- one key of the calib dump, `vertex_scoreboard.dual_chain.off_ms` (top level and per candidate) —
  a wall-clock stopwatch in milliseconds. On `cv` event 10 that is **1 key of 160 640**.

### 3.2 The control, which is what makes 3.1 a gate rather than an assertion

`ARM_SUFFIX=rep` re-runs cv f000 **at the same configuration** and the gate is pointed at the pair.
No byte-level instrument in this arc had ever been run on two runs of one configuration — doc 116
sec 4's "the noise floor is zero" is a statement about *metrics*, not archive members or ROOT
containers.

| | same config, two runs | flip vs `tfull` |
|---|---|---|
| files differing | 28 of 82 | 28 of 82 |
| ROOT trees differing | `Trun.wcp_git` only | `Trun` × 4 provenance branches only |
| calib dump | `dual_chain.off_ms` | `dual_chain.off_ms` (+ provenance) |

The two classes the gate forgives are exactly the classes that move **when nothing moves**.

### 3.3 Three defects in my own instruments, found by that control

Recorded because each of them would have produced a confident wrong answer:

1. **Events keyed by basename across sub-roots.** Event ids restart in every stage-A input file, so
   `f000/pr_evt25` was being compared against some other file's event 25. The first gate run
   reported `pctree`, `mabc-pr.zip` and `nusel` as divergent on nearly every `cv` event — an entire
   fabricated physics difference. Fixed to key on `(sub-root, event)`; the flat beam-off arm needed
   the same fix a second time, where the wrong key silently matched **0** events and printed
   "every tree identical".
2. **`np.array_equal` raises on jagged arrays** rather than returning False. The except-branch
   marked ~150 healthy `T_tagger` branches as differing, and I nearly reported a determinism defect
   in the tagger's per-candidate features. A comparator that throws is a comparator that lies.
3. **NaN ≠ NaN.** `T_rec_charge.reduced_chi2` carries NaN on rows with no fit; it was flagged on
   every arm pair *including the control*. It is bit-identical: 10 of 2 587 rows on `cv`, 22 of
   14 582 on `nuecc`, 2 of 755 on `off`, all NaN in both arms, **0 genuinely different values**.

Also: a bad `${#EVTS[@]:-$nql}` substitution aborted the nuecc loop after one of three sub-roots
while still printing a clean summary — caught by counting `pr_evt` dirs against stage A, not by
reading the runner's verdict.

### 3.4 G2–G8

| gate | what it checks | result |
|---|---|---|
| **V0** | `prod_cfg_gate.py --ref prod-2026-09-17b` **before** the first edit | **PASS 21/21** — the drift below has no inherited component |
| **V1** | pin freshness | `71ebd5aeb3868cbc0803f2fa2654b46b` == `local/lib`, toolkit `0a2807f4` |
| **G2** | production fit JSON vs the arm's `149_tf_sbnd_kf.json`, `_`-keys stripped as `load_trackfitting_config` does | **47 live keys, identical** |
| **G3** | compiled node diff, no-TLA, pre vs post | exactly **3 changed nodes** (`CreateSteinerGraph:pr`, `:prrefresh`, `ImproveCluster_2:pr`) + the two `BlobSampler` renames; `trackfitting_config_file` **unchanged** |
| **G3b** | post-flip no-TLA compile sha vs the measured cell | **`3db5df01e33ff853` = the doc-116 `csp3bw` cell's sha, exactly** — this is what covers every emitted key nobody read individually, including `[11]/[12].data.strategy[0].disable_mix_dead_cell = False` in sec 4, a key this round never chose: it is the object form the `charge_stepped` sampler emits, and it is right because the whole document is byte-for-byte the cell that was graded |
| **G4** | blast radius across the 21 consumers | **3 drift, all SBND** (`bare_prjob.json`, `prod_prjob.json`, `sbnd_pr.json`); uBooNE, PDHD, PDVD **byte-identical** |
| **G5** | the LArSoft 1-step chain | same 3 changed nodes, same keys — the two chains move together |
| **G7** | the three `*_track_fitting.json` added to the consumer set | done, **before** the refresh |
| **G8** | new reference | `ref/prod-2026-09-20/` **PASS 25/25**; `prod-2026-09-17b` now drifts on the 3 SBND artifacts, as it should |

`./build/clus/wcdoctest-clus` is run for hygiene; nothing in C++ changed.

**The two halves of the proof, together:** the flipped default compiles byte-for-byte to doc 116's
`csp3bw` config (G3), and the fit JSON it now resolves is key-for-key the arm's (G2). `tfull` =
`csp3bw` + those keys. So the production operating point after this flip **is** doc 116's `tfull`,
and G1 shows it on the outputs of 62 events as well.

### 3.5 The gap this flip exposed, and closed

Adding `fit_weight_pow`/`assoc_cont_center` to `sbnd_track_fitting.json` moved **zero of the 21**
consumer hashes. The tripwire would have passed while the fit changed. `compile_consumers.sh` now
hashes the SBND, PDHD and PDVD `*_track_fitting.json` as artifacts 22–24, so a future flip of this
family cannot pass silently.

**The same-shaped hole one level over, also closed.** Step (c) of the consumer set compiles
`wcls-img-clus.jsonnet` and the standalone Q/L job, and **neither calls `pr()`** — so the chain that
actually runs the PR taggers under LArSoft, `sbnd/wcls-img-clus-matching-xin.jsonnet`, was in none
of the artifacts either. This round had to verify by hand (G5) that it tracked the flip; a future
divergence — someone regenerating `pr-operating-point.jsonnet` from a stale TLA set — would not have
been caught. Step (g) now compiles it at `pr_operating_point=sync` as artifact **25**.

`ref/prod-2026-09-20` therefore carries **25 artifacts, not 21**: the 21 compiled consumers, the
three runtime fit JSONs, and the LArSoft 1-step chain.

---

## 4. The named drift, key by key

`docs/118_figs/118_drift.txt`, `prod-2026-09-17b` → this tree:

```
DRIFT     : bare_prjob.json, prod_prjob.json, sbnd_pr.json      (3 of 21; the other 18 identical)

  REMOVED [11].data.strategy[0] = 'stepped'
  ADDED   [11].data.strategy[0].disable_mix_dead_cell = False
  ADDED   [11].data.strategy[0].name = 'charge_stepped'
  CHANGED [11].name : 'live-apa0-0' -> 'live-cs-apa0-0'
  REMOVED [12].data.strategy[0] = 'stepped'
  ADDED   [12].data.strategy[0].disable_mix_dead_cell = False
  ADDED   [12].data.strategy[0].name = 'charge_stepped'
  CHANGED [12].name : 'live-apa1-0' -> 'live-cs-apa1-0'
  CHANGED [13].data.samplers[0].name : 'BlobSampler:live-apa0-0' -> 'BlobSampler:live-cs-apa0-0'
  CHANGED [13].data.samplers[1].name : 'BlobSampler:live-apa1-0' -> 'BlobSampler:live-cs-apa1-0'
  ADDED   [14].data.base_weight_blank_alpha = 0.5
  ADDED   [14].data.base_weight_scope = 'tree+path'
  ADDED   [14].data.terminal_blank_plane_mode = 'prefer3'
  ADDED   [20].data.base_weight_blank_alpha = 0.5
  ADDED   [20].data.base_weight_scope = 'tree+path'
  ADDED   [20].data.terminal_blank_plane_mode = 'prefer3'
```

`[14]` is `CreateSteinerGraph:pr`, `[20]` is `CreateSteinerGraph:prrefresh`, `[13]` is
`ImproveCluster_2:pr`. Component (c) appears nowhere in this list — sec 3.5.

---

## 5. What production now costs, and what it buys

Doc 116 sec 14, quoted against the same-conditions `s0rep` control (the ±10 % environment term is
that instrument's resolution; `wall_s` is I/O-contention dominated and is not read on its own):

| | `cv` | `nuecc` | `off` |
|---|---:|---:|---:|
| PR-stage compute, vs pre-flip | **+10 %** | **+19 %** | +25 % (tiny absolute) |
| peak RSS p50 / max | 0.47 / 1.37 GiB (unchanged) | 1.17 / **2.21** GiB (was 1.15 / 1.51) | 0.38 / 1.37 (unchanged) |
| events > 2 GiB | 0 | **5 of 2 001** | 0 |

**A per-process memory cap of 2 GiB no longer holds on νe-like events; 2.5 GiB does.** `cv` is
unchanged. These are one-event-per-process jobs — a batching driver must apply the cap to the batch.

All of the CPU is `charge_stepped` (it triples both Steiner builds, ×2.9 and ×3.4); `p3bw` and the
fit keys are free. The memory tail needs **both** knobs: `charge_stepped` alone tops out at
1.91 GiB with 0 events above 2 GiB.

Also carried, and not to be forgotten: **10–13 % of the selected signal changes identity**, so the
uBooNE BDT operating points and doc 107's cut package are now tuned against a different trajectory
(sec 10 item 2); and `med_d_W` 0.2381 → 0.2459 wires is the one closure clause the fit keys move
the wrong way.

---

# Part B — the PR profiling campaign

> **EXECUTED — see `docs/119_pr-profiling-rounds.md` (2026-09-21).** Doc 119 supersedes the sizing
> below wherever the two disagree, and it corrects three premises of this part:
> (1) `want_2d` is a PDHD/PDVD lever only — `check_stm_conditions` belongs to `TaggerCheckSTM`,
> and SBND's whole `TaggerCheckSTM:pr` stage is 0.3 % of the job (sec 8 item 1 is wrong);
> (2) "the PDHD `-nu` max event (174 s / 5.54 GB)" is two different events (sec 7 item 4);
> (3) `CheckSTM_Michel` is no longer 28 % of the PDHD `-nu` arm — it is 17.4 %, and Steiner is
> 50.3 % (sec 6 table).
> Outcomes: round 1 (tcmalloc on the SBND PR chain) measured, gated on 62 events and **flipped**;
> round 2 (`proj_pad` for SBND) measured, gated — and **NOT flipped** despite the pre-authorisation,
> because SBND already keeps 87.9 % of its proj cells where PDHD kept 2.4 %, so the doc-30 trade
> this was authorised on does not exist here (doc 119 sec 6).

The flip's cost is what makes this urgent, and the owner has authorised multiple rounds. Scope:
PDHD and PDVD (STM+Michel tagger) and SBND (Neutrino tagger), CPU and memory.

## 6. What is already measured — and three things not to re-propose

| detector / chain | the ladder | source |
|---|---|---|
| SBND Neutrino, **post-flip** `nuecc` | `TaggerCheckNeutrino:pr` **11.51 s of a 17.13 s job (67 %)**, of which the dual second pass is **4.71 s (27 % of the whole job)**; `CreateSteinerGraph:pr` + `:prrefresh` **2.73 s** (0.77 s pre-flip); BDT scorers 2.1 s | doc 116 §14, doc 117 §7 |
| PDHD/PDVD STM (`-stm`) | `TaggerCheckSTM::visit` **58.2 %** → `do_single_tracking` 48.5 → `dQ_dx_fit` 42.8 → **`fill_fitted_charge_2d` 19.9 %**; `CreateSteinerGraph::visit` **26.4 %**; ~40 % container machinery | doc pdvd/30 §12.1 |
| PDHD/PDVD Michel (`-nu`) | `CheckSTM_Michel` p50/p90/max **6.6 / 29.3 / 174.3 s** (PDHD), **3.5 / 10.1 / 25.6 s** (PDVD) — **28 % of the PDHD arm's CPU, 20 % of PDVD's**, for almost no memory | doc pdvd/30 §8 |

**Instruments** (all exist): `abtest/timecmd.py` (`.time.meta`: `maxrss_kb` = getrusage CHILDREN
high-water, not the 2 s VmHWM sampler that under-reports on 19 % of PDHD and 55 % of PDVD jobs);
the in-job ladder from `MultiAlgBlobClustering.cxx:3698-3771` `Perf` — **two** instruments, the
inline `MABC timing:` line per stage and the `TICK:`/`MEM:` block emitted *once at destruction*, so
a killed job has the first and not the second; enabled by config `perf: true` (already on in all
three productions) **plus** a debug-level spdlog build **plus** `-L debug`. Parsers:
`scripts/d116/perf_cost.py`, `pdhd/stm/perf/d30_pr_census.py`, `scripts/pr142_perf.py`.

**Two findings that change how round 0 must be run:**

- **SBND is not missing a PR profiler**, but the one it has is stale:
  `scripts/perf/profile_pr11.sh` hardcodes a 13-stage `pipeline_names` where production runs 15
  (no `protect_bundle`, no `steiner_refresh`). Do not profile with it as-is. The fix is not to
  re-type the list — point the profiler at the arm's own precompiled `.wct-cfg-evt<ID>.json`, which
  `run_pr_chain_batch.sh:2003` already leaves in every PR dir and which *is* production by
  construction.
- **SBND's PR runners do not preload tcmalloc.** `run_pr_evt.sh:317-321` and
  `run_pr_chain_batch.sh:1950,2134` **assign** `LD_PRELOAD="$PYLIB"`, so libpython replaces rather
  than joins, and `grep tcmalloc` on both returns nothing. PDHD and PDVD preload
  `libtcmalloc_minimal` at every stage, and SBND's own `run_clus_evt.sh` does, verified
  byte-identical on 5 events × 3 archives.

  **Read the runner's own comment before treating this as an oversight** (`run_pr_evt.sh:307-310`):
  *"The -stm / -tgm / bare -p arms must keep the exact process environment they had before the
  doc-pr/4 default flip -- they are A/B comparison arms."* The **conditional** structure is
  deliberate and load-bearing. What the comment does not address is tcmalloc, which is simply
  absent. So the accurate statement is: the *shape* is by design, the *absence* is unexplained, and
  a round-1 lever must preserve the shape (sec 8).

  Two consequences either way: a profile of SBND PR under `libtcmalloc_and_profiler` is **not** the
  production allocator, unlike PDHD/PDVD — carry the caveat or run a glibc control; and any
  comparison against a pre-existing SBND PR arm must account for the allocator, because those arms
  ran without it.

**Heap profiling uses jemalloc sampling, not `HEAPPROFILE`** (doc pdvd/28 §0 measured the
gperftools heap path ~25× slower): `MALLOC_CONF=prof:true,…,lg_prof_sample:19` plus
`pdhd/stm/perf/je2pprof.py` → `google-pprof --inuse_space`. With tcmalloc, RSS is not a heap metric
(1.65 GB live vs 3.26 GB RSS on one event).

**Manifests**: SBND 16-event runtime gate `docs/pr/149_figs/manifest_gate16/{mcp1k,mcp2k}.txt`;
PDVD 120 `pdvd/scripts/perf_manifest.tsv`; PDHD 61 enumerated in
`pdhd/docs/scripts/d30_run_pr_arm.sh`; doc-30 busy sets of 6 and 7.

**Do NOT re-propose.** Doc 30 §12.2 measured three byte-identical levers and reverted all three
(−0.2 % core-s total): the `APAFacePlane` pointer cache (hit rate 40 %, wrapped wires), the row-set
`std::move` (only 30–33 % of rows are single-`Coord2D`), and the "third copy"
`merge_fitted_charge_2d` move — which §11 had called the recommended next step and §12.2(c) then
measured at peak **5.00 → 5.00 GB**. Also parked with reasons: `release_post_nu`, the
`GraphAlgorithms` LRU, BiCGSTAB tolerance (moves results), `nu_max_main_length_cm` (an owner
physics decision, not a perf lever).

## 7. Round 0 — census and profile, on the post-flip tree, no code changes

Sizing `CreateSteinerGraph` at its pre-flip 0.77 s instead of its real 2.73 s would mis-rank every
target; and doc 30's profiles predate both `f9665bea` and the doc-108 fit-key flip, so PDHD/PDVD
need re-profiling for the same reason.

1. Fork `profile_pr11.sh` → `scripts/perf/profile_pr118.sh` (M10; the pr/11 script is a record),
   pointed at an arm's `.wct-cfg-evt<ID>.json`. Precompiled cfg (M17), tcmalloc+profiler, never
   `setarch -R`, `OMP/MKL_NUM_THREADS=1`.
2. **CPU**: SBND — the worst post-flip `nuecc` events by TICK, a median one, and a `cv` one;
   PDHD/PDVD — the doc-30 busy sets, so the numbers sit beside §12.1.
3. **Close the named gap**: every gperftools profile in the tree is `-stm`. `CheckSTM_Michel` —
   28 % / 20 % of the PDHD / PDVD `-nu` arms — has **never been CPU-attributed**.
   `pdhd/profile_pr.sh MODE=-nu` is already built for it. This is the deliverable the owner's
   "STM+Michel tagger for PDHD and PDVD" asks for.
4. **Memory**: jemalloc sampling on the five SBND `nuecc` events above 2 GiB (named in the doc-116
   per-event records) and on the PDHD `-nu` max event (174 s / 5.54 GB).
5. **Census**: `perf_cost.py` (SBND) + `d30_pr_census.py` (PDHD/PDVD) into one table, plus a
   glibc-vs-tcmalloc control on SBND PR so the profile's allocator is accounted for.

Deliverable: a ranked, sized target list per detector, each with the gate it would need.

## 8. Round 1 — the two byte-identical levers, no display trade

1. **`want_2d`** (doc 30 §13.8, never taken). `check_stm_conditions` fits each pass twice —
   round 1 on the rough path (`TaggerCheckSTM.cxx:3606`), round 2 on the adjusted path (`:3622`) —
   but `begin_pass_record` captures the 2-D map only after round 2 (`:3624`). **About half of the
   ~103 fills per event are built in full and read by nobody.** A `want_2d` flag through
   `do_single_tracking` drops them byte-identically. Worth less on PDHD/PDVD since the pad knob made
   each fill ~5× cheaper — but SBND has no pad knob yet, so on SBND it is at full value.
2. **tcmalloc in the SBND PR runners** (sec 6). Runner-level, reco-neutral, with SBND's own
   `run_clus_evt.sh` byte-identity precedent. Two constraints, both from the runner itself:
   it must **join** `LD_PRELOAD`, not assign, or it silently drops libpython and the job runs the
   geometric vertex fallback; and it must **not** change the process environment of the `-stm` /
   `-tgm` / bare `-p` arms, which `run_pr_evt.sh:307-310` keeps frozen on purpose because they are
   A/B comparison arms. So the lever is scoped to the full PR pipeline and ships behind its own
   switch, and any core-s claim against an older SBND arm states which allocator each side ran.

Gate: `d30_hash_gate.py`-style product identity on the SBND 16-event manifest and the PDHD/PDVD busy
sets, plus an A/B core-s census at `JOBS=1` (a 3 % claim is inside batch noise at `JOBS=6`).
Measure before believing — doc 30 §12.2's lesson.

## 9. Round 2 — `proj_pad_wire` / `proj_pad_time` for SBND: pre-authorised

C++ default −1 = OFF, present in `pdhd_track_fitting.json` and `pdvd_track_fitting.json` **only**.
`TrackFitting.cxx` is shared, so the absent key is what keeps SBND still; it is read by
`TaggerCheckNeutrino` as well as `TaggerCheckSTM`. On PDHD it bought **−28.8 % core-s and −56 % peak
RSS** with every physics product byte-identical — it attacks both the +19 % CPU and the 2.2 GiB tail
this flip introduces.

It trims the fitted-2-D-charge **display** to cells within 3 wires / 3 slices of a predicted cell;
only ~2.4 % of cells carry a prediction. SBND's consumers are `T_proj_data`, the calib dump's `proj`
block, and `pr85_panels2d.py`.

**The owner pre-authorised this for SBND on 2026-09-20**, the same trade accepted for PDHD/PDVD in
doc 30 round 3. Round 2 therefore measures **and** flips, on one explicit condition: only if the
per-tree gate shows every physics product byte-identical (the doc-30 bar: `T_rec_charge`, the tagger
trees, `mabc-pr.zip`, the non-`proj` calib dump), with `T_proj_data` and the `proj` block the only
things that move. Before/after `pr85_panels2d.py` panels are produced as the record of what the
display lost. **Not bundled into this round's commit**, although it lands in the same file — this
flip must stay gradeable on its own.

### 9.1 Round 3 — make SBND's second pass cheaper, not absent (doc 117 §10.3)

27 % of the post-flip `nuecc` job, and it earns it (1.7–2.1 pt of νeCC efficiency), so the target is
cost, not removal. It runs on 97–99 % of νe events but its transfer moves the pick on ~33 %.
`vertex_scoreboard.dual_chain.{agree,transferred,d,off_ms}` is already recorded per event in every
arm, so the census costs nothing. Lever: a default-OFF knob that skips the pass where it cannot
matter — **this one changes which events get the snap, so it needs grading against doc 107's
metrics, not only a byte-identical gate**, since `agree` does not provably imply rejection. Second
lead (unsized): `run_dual_chain_off_pass` duplicates the whole sequence including its own
`TrackFitting` to hand back one vertex position; PDVD's doc 28 round 3 already shipped the
analogous `a94ce32e`.

### 9.2 Round 4 — the structural one, all three detectors

Restricting `fill_fitted_charge_2d` to the cells a fit actually predicts (doc 30 §9 top row, §12.4:
40–45× ratio, 19.9 % of CPU) and dropping the per-cell `std::set<Cluster*>` where no consumer reads
it (20.5 % of live heap). Knob-required, changes which cells exist in every downstream product, a
round of its own. Listed so it is not forgotten; not scheduled.

**A cross-detector note this flip creates:** SBND now runs `charge_stepped`, PDVD does too
(doc pdvd/103), PDHD does (doc pdvd/108) — but the *retile* sampler flip is per detector and PDVD's
doc 102 had rejected it once on tagger cost. The three detectors are now on the same trajectory
family; they are **not** on the same display settings (`proj_pad_*` is PDHD/PDVD only until round 2).

---

## 10. Open items

1. **Doc 117 sec 10.2's independent-νe replication.** No longer a gate on the flip; still the
   cheapest way to convert a secondary signal into a primary verdict, and post-flip it is a
   validation of what production does. Keep it.
2. **Downstream retuning.** The uBooNE BDT operating points and doc 107's cut package were tuned
   against the pre-flip trajectory, and 10–13 % of the selected signal changed identity.
3. **`T_tagger` determinism was *not* tested here.** Sec 3.3 defect 2 means my first "finding" of
   ~150 unstable branches was an artifact; the corrected comparator says identical on the events
   gated. That is 62 events, not a determinism study.
4. **Disk.** The doc-118 gate arms are 318 MB (`work-r3{cv,nue,off}-d118flip` + `-d118fliprep`);
   the doc-117 arms are ~74 GB and the doc-116 arms ~95 GB. Every number from them is in the
   committed `products/` tables, so they are a retention decision. 522 G free.
5. **`pr-operating-point.jsonnet` currency.** It is generated, its header pins its sync to toolkit
   `0ad642235`, and the `scripts/compile-both.sh` acceptance test it names does not exist in this
   tree. Artifact 25 now detects a divergence, but nothing regenerates the wrapper; a round that
   flips a knob the wrapper *does* carry must regenerate it by hand.
6. **The `scripts/d118` drivers have no completeness guard.** The d117 fork refuses an incomplete
   arm; `stageB_flip.sh` does not, and a bad substitution silently aborted its nuecc loop after one
   of three sub-roots while still printing a clean summary. Carry the guard in if round 0 reuses it.
7. Doc 115/116/117's own open items carry: the off-beam gate count `f`, νeCC purity needing a full
   CV production, the `dvm()` CPA face applied to MC, and the missing completeness guard in
   `scripts/d115` and `scripts/d116`.

## 11. Files

**toolkit** (`apply-pointcloud`, commit `675fd266`): `cfg/pgrapher/experiment/sbnd/clus.jsonnet` (7 call-site lines +
2 rewritten comment blocks), `cfg/pgrapher/experiment/sbnd/sbnd_track_fitting.json` (2 keys + the
`_comment_d118_fit_lattice_knobs` block).

**wcp-porting-img** (`main`): this doc; `scripts/d118/{stageB_flip.sh,hash_gate.py,root_gate.py,
tf_key_gate.py,cfg_diff.sh}`; `docs/118_figs/{118_gate_hash.txt,118_gate_tfkeys.txt,
118_cfg_diff.txt,118_drift.txt}`; `ref/prod-2026-09-20/{consumers.sha256,prod_prjob.json,README.md}`;
`scripts/cfg/compile_consumers.sh` (+ steps (f) and (g): the three track-fitting hashes and the
LArSoft 1-step chain, sec 3.5) — the only edit to an existing file.

Untouched (M13): `docs/115_*`, `docs/116_*`, `docs/117_*`, `products/d115|d116|d117`,
`scripts/d115|d116|d117`, `ref/prod-2026-09-17b` and every earlier generation, `work/ql_labels/`,
`abtest/snap/`, and every other experiment's configs.
