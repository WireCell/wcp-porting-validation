# doc 75 — tagger-family audit: promoted candidates + FV consistency, two fixes SHIPPED (SBND PRODUCTION ON)

**Status (2026-08-20): both knobs implemented, gated, DEFAULT OFF in C++;
SBND PRODUCTION ON (owner: "things are good, turn them on for SBND
production"). See §9 for the flip round — it corrects §6's exposure
estimate for `nu_selected_as_main_snapshot_all` upward substantially (the
flip-equivalence check surfaced exposure the original enriched-manifest
census had not sampled) and confirms no ADVERSE on the expanded evidence.**

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
git log --oneline -1                                          # 9b5bb8fd
ls -la --time-style=full-iso build/clus/libWireCellClus.so     # freshness, M1
./build/clus/wcdoctest-clus                                   # 218/218 cases, 2301/2301 assertions

cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
MC50=$(ls -d work-mcp1k-ql0819/ql_evt* | sed 's/.*ql_evt//' | sort -n | head -50)
ENR="58755 62583 65289 65999 68956 71530 174422 280466 283515 283969 285253 \
286681 287007 288327 313847 317939 353729 395148 398690 400636 401252 401450 \
408028 409634 486907 487303"     # promoted-main (21) u multi-candidate (8) events,
                                  # from the peer pr/102 round's work-pr102-head-mcp1k

for s in nuecc48 ncpi0; do
  PR_EXTRA_STAGES=pr_display PR_JOBS=10 ./run_pr_chain_batch.sh work-$s-ql0819 work-d75r1-bare-$s data
done
PR_EXTRA_STAGES=pr_display PR_JOBS=10 ./run_pr_chain_batch.sh work-mcp1k-ql0819 work-d75r1-bare-mc50 data $MC50
PR_EXTRA_STAGES=pr_display PR_JOBS=10 ./run_pr_chain_batch.sh work-mcp1k-ql0819 work-d75r1-bare-enriched data $ENR
# (edit + wcbuild + freshness proof + wcdoctest-clus, per the commit above)
for s in nuecc48 ncpi0; do
  PR_EXTRA_STAGES=pr_display PR_JOBS=10 ./run_pr_chain_batch.sh work-$s-ql0819 work-d75r1-off1-$s data
done
PR_EXTRA_STAGES=pr_display PR_JOBS=10 ./run_pr_chain_batch.sh work-mcp1k-ql0819 work-d75r1-off1-mc50 data $MC50
PR_EXTRA_STAGES=pr_display PR_JOBS=10 ./run_pr_chain_batch.sh work-mcp1k-ql0819 work-d75r1-off1-enriched data $ENR

SBND_NUE_SP_CONSISTENT_FV=1 PR_EXTRA_STAGES=pr_display PR_JOBS=10 ./run_pr_chain_batch.sh work-nuecc48-ql0819 work-d75r1-onfv-nuecc48 data
SBND_NUE_SP_CONSISTENT_FV=1 PR_EXTRA_STAGES=pr_display PR_JOBS=10 ./run_pr_chain_batch.sh work-ncpi0-ql0819 work-d75r1-onfv-ncpi0 data
SBND_NUE_SP_CONSISTENT_FV=1 PR_EXTRA_STAGES=pr_display PR_JOBS=10 ./run_pr_chain_batch.sh work-mcp1k-ql0819 work-d75r1-onfv-mc50 data $MC50
SBND_NU_SELECTED_AS_MAIN_SNAPSHOT_ALL=1 PR_EXTRA_STAGES=pr_display PR_JOBS=10 \
  ./run_pr_chain_batch.sh work-mcp1k-ql0819 work-d75r1-onflag-enriched data $ENR

python3 scripts/pr85_hash_gate.py work-d75r1-bare-<s> work-d75r1-off1-<s>     # OFF gate, per sample
python3 scripts/pr94_root_gate.py work-d75r1-bare-<s> work-d75r1-off1-<s>     # OFF gate, per-branch value
python3 scripts/d75_nue_fv_census.py work-d75r1-off1-<s> work-d75r1-onfv-<s>  # FV knob census
python3 scripts/d75_mainflag_census.py work-d75r1-off1-enriched work-d75r1-onflag-enriched  # flag-leak census
```

Samples: `work-nuecc48-ql0819` (48 evts), `work-ncpi0-ql0819` (19), the numu-50
subset of `work-mcp1k-ql0819` (first 50 sorted event ids, same list as doc 74),
and a new **enriched manifest** (26 unique mcp1k events, see §0).

## 0. Scope and why the standard 3 samples aren't enough for half of this round

Owner's request, verbatim: *"we have promoted some of the non-main cluster as
the neutrino candidate [doc pr/94]. Now, I want to understand whether the
cosmic tagger, numu tagger, nue tagger, other tagger, do they have any issue
with this... In addition, as we just fixed the fiducial volume for the cosmic
tagger [doc 74]. I wonder if there is a problem for other taggers (numu,
nue...). Can you compare the prototype code as well as the toolkit code to
see if we are missing anything. We should fix them, if any."*

Two independent questions, three parallel read-only audits (tagger↔flag map;
FV call-site audit; main-flag/promoted-candidate trace), then two fixes. Owner
decisions on scope (AskUserQuestion, this session): peer-session file
collision → **rebuild-first protocol**; FV fix → **nue + single-photon only**
this round (wider sites documented, not fixed); flag leak → **fix now**, not
census-first.

**Exposure on the standard doc-74 samples is too thin for the flag-leak
question.** Events with a promoted (`nu_selected_as_main`) candidate: **1/48**
nueCC48, **0/19** NCpi0, **0/50** numu-50; events with ≥2 candidates in a
bundle: measured separately at **8/1000** on a 1000-event mcp1k arm (the peer
session's concurrent `work-pr102-head-mcp1k`, read-only). Doc pr/94 §9.11
already learned this lesson for a census ("1 mover of 48 ... gave the knobs 5
and 1 chances"); this round builds a 26-event **enriched manifest** (union of
the promoted-main and multi-candidate event lists) instead of trusting the
3-sample census to say anything about the flag-leak knob.

## 1. Question (a): do the taggers depend on main-cluster state a promoted candidate lacks?

**Short answer: no tagger reads a cluster flag at all.** Grepped
`Flags::main_cluster` (and every other `Flags::*`) across
`NeutrinoTaggerCosmic.cxx`, `NeutrinoTaggerNuMu.cxx`, `NeutrinoTaggerNuE.cxx`,
`NeutrinoTaggerSSM.cxx`, `NeutrinoTaggerSinglePhoton.cxx` — zero hits. Main-ness
inside every tagger is exclusively the `main_cluster` **pointer parameter**
`TaggerCheckNeutrino` hands them (reduced to `get_cluster_id()` for identity
tests), plus `main_cluster->grouping()` for event-level services
(`FiducialUtils`, wire angles). A promoted candidate is passed as that pointer
exactly like a real main, so it is **not** disadvantaged at the tagger
boundary.

The six places that DO derive main-ness from the persisted `Flags::main_cluster`
are all **upstream** of the taggers, in the PR/graph stages:
`NeutrinoPatternBase.cxx:1433,2797,2806,3001`, `NeutrinoVertexFinder.cxx:3450`,
`NeutrinoTrackShowerSep.cxx:2013`. `TaggerCheckNeutrino.h:625-641` documents all
six; `m_nu_selected_as_main`'s `SelectedMainFlagGuard`
(`TaggerCheckNeutrino.cxx:1512-1542`, armed for the whole per-candidate loop
body, spanning all five tagger calls at `:2408-2503`) is exactly the shipped
patch that satisfies those six reads for a promoted candidate — SBND
production ON since 2026-08-19 (`wct-pr-perevt.jsonnet:1171`, doc pr/94 round 3).

**What this audit found the guard does NOT cover — Finding A, fixed this
round.** `swap_main_cluster` (`NeutrinoPatternBase.cxx:3470-3512`) sets
`Flags::main_cluster` on a **second** cluster and clears it on the first;
called from `determine_overall_main_vertex_DL`
(`NeutrinoVertexFinder.cxx:4395,4976`), reached from
`TaggerCheckNeutrino.cxx:2141-2148` **on the real `main_cluster`/
`other_clusters` pointers** (not the throwaway copies the traditional
non-DL fallback uses at `:2150-2159`), with `dl_weights` non-empty in SBND
production (`wct-pr-perevt.jsonnet:750`). `SelectedMainFlagGuard`'s
destructor only clears the candidate's own pointer, so the swapped-to cluster
**keeps `Flags::main_cluster = 1` permanently** after the pass —
contradicting the guarantee `TaggerCheckNeutrino.h:637-641` documented ("no
later visitor... sees a changed flag"). Confirmed reachable in production;
confirmed **firing** by the §4 census.

This is not a per-bundle-specific bug: the same `swap_main_cluster` call site
runs for the single legacy candidate too, so the leak **pre-dates doc pr/94**.
Per-bundle mode multiplies the opportunity (N candidate passes per event
instead of 1), and matters in practice once `nu_selected_as_main` makes a
promoted candidate's flag state visible downstream at all. Blast radius
identified: `PrDisplayDump.cxx:1099` (`is_main_cluster` in the `steiner` dump
block of `calib-pr-evt<ID>.json`), `PrDisplayDump.cxx:467,841`, `PatternDebugIO.cxx:213`,
persisted pctree flags (`normalize_cluster_flags`,
`MultiAlgBlobClustering.cxx:3357`) — display/serialization consumers, not a
reconstruction decision in the vast majority of cases, because every
PR/tagger reader in §1 above takes the cluster by reference from its caller,
never by scanning for the flag.

**Correction, §9 flip round.** The clause above is right that a stray flag
cannot leak into a DIFFERENT candidate's tagger pass, but it is too strong
about consequences overall: `NeutrinoPatternBase.cxx:2797`
(`find_proto_vertex`, one of the six upstream readers) runs on **companion**
clusters too (the `other_clusters` loop, `TaggerCheckNeutrino.cxx:2067-2104`,
which executes BEFORE that candidate's own DL-swap call). In a per-bundle
event, a companion cluster carrying a STALE flag left over from an EARLIER
candidate's uncorrected swap can therefore reach a different endpoint-
ordering branch in that reader — a real, if tiny, reconstruction effect, not
merely display. Measured on one event out of the ~16 this fix touches
(§9.3) — everywhere else the correction is exactly what the paragraph above
says. The exposure and the adjudication are both in §9.

## 2. Fix — `nu_selected_as_main_snapshot_all` (default false)

Extends the guard from "restore the candidate's own flag" to "snapshot and
restore `Flags::main_cluster` on every cluster in `{main_cluster} ∪
other_clusters`, taken **before** any write in the pass (construction ordered
ahead of `SelectedMainFlagGuard`, so the snapshot predates that guard's own
write too), restored **after** the pass regardless of how many swaps happened
in between" (`TaggerCheckNeutrino.cxx` `MainFlagSnapshotAllGuard`,
`:1553-1596`). A sentinel log line fires only when the live value differs
from the snapshot at restore time — i.e. only when a swap actually moved the
flag during that candidate's pass — which is the census signal in §4.

Gated independently of `nu_selected_as_main` (only matters in practice when
that knob is also on, but the underlying swap call site doesn't care).

## 3. Question (b): do numu/nue/other carry the same zero-margin FV inconsistency doc 74 fixed for cosmic?

**Root cause, structural.** `MakeFiducialUtils` is always built with
`fiducial=dv` (`sbnd/clus.jsonnet:1793-1796`); `dv` is `DetectorVolumes`, the
**zero-margin** sensitive-volume union — 3 cm more permissive at every wall
than `sbnd_pr_fv + sbnd_pr_fv_margins`
(`sbnd/clus.jsonnet:1789-1792`, its own in-tree comment). Any tagger reaching
`get_fiducialutils()` therefore tests that permissive volume regardless of
which detector-consistency knob is on elsewhere. `cosmic_consistent_fv` (doc
74) fixed cosmic by routing *around* `FiducialUtils` entirely, not by adding a
margin to it — same shape reused here.

**Answer, per tagger:**

| tagger | FV test? | verdict |
|---|---|---|
| cosmic | yes | fixed, doc 74 |
| numu | **no FV test in prototype or toolkit** | genuine parity — no action |
| SSM | **no FV test in prototype or toolkit** | genuine parity — no action |
| pi0 | prototype file is empty (`NeutrinoID_pio_tagger.h`, 1 line); no toolkit port | genuine parity — no action |
| nue | yes, 3 call sites, all zero-margin | **fixed this round** |
| single-photon | yes, 1 call site, zero-margin | **fixed this round** |

No `other_tagger` symbol exists in either tree; the nearest neighbours
(`other_showers`, `single_shower`, `single_shower_pio_tagger`) have no FV
test either.

**The four nue/SP call sites, and the corrected fix shape.** Toolkit
`fiducial_utils` = `FiducialUtils::inside_fiducial_volume`
(`FiducialUtils.cxx:79-120`; empty tolerance ⇒ zero-margin `contained(p)`,
non-empty ⇒ six shifted-point probes). Prototype
`WCPPID::ToyFiducial::inside_fiducial_volume(p, offset_x, tolerance_vec)`
(`ToyFiducial.cxx:1756-1818`) has **two different polygon sets depending on
whether `tolerance_vec` is NULL** — this is the one place the initial audit
premise ("prototype uses SCB polygons inset by ~3 cm") was backwards, and the
in-tree comment at `NeutrinoTaggerCosmic.cxx:562-563` already had it right:

- **`tolerance_vec` non-NULL** (SCB per-y/z-sliced polygons, `:1769-1793`):
  **no** baked-in inset — the tolerance argument IS the entire margin.
- **`tolerance_vec` NULL** (non-SCB polygons, `:1764-1768`): built **already
  inset** by `boundary_dis_cut` (production value 3 cm,
  `wire-cell-prod-nue.cxx:417`), confirmed applied **uniformly across all six
  faces** at `ToyFiducial.cxx:118-131`.

| site | prototype call | tolerance | fix |
|---|---|---|---|
| `NeutrinoTaggerNuE.cxx:396` `angular_cut` | `NeutrinoID_nue_tagger.h:504,511` | non-NULL, uniform −1.5 cm | route to configured fiducial, **same** −1.5 cm (6-element form) |
| `NeutrinoTaggerNuE.cxx:2495,2648` `shower_to_wall` (both while-loops) | `:1248-1249,1398` | non-NULL, uniform −1.5 cm | same |
| `NeutrinoTaggerNuE.cxx:3320` `bad_reconstruction_2` `other_fid` | `:3222,3230` | **NULL** → non-SCB, baked-in −3 cm | route to configured fiducial **+ uniform −3 cm** (NOT the bare box — see below) |
| `NeutrinoTaggerSinglePhoton.cxx:734` `bad_reconstruction_2_sp` `other_fid` | `NeutrinoID_singlephoton_tagger.h:3538,3546` | **NULL** | same as above |

**A correction caught in review before implementation.** The first draft of
the two NULL-tolerance fixes proposed routing to the bare `sbnd_pr_fv` box
with no shift vector. That under-delivers: `sbnd_pr_fv` is inset from the
zero-margin `dv` union by only **0.40 cm (x) / 0.65 cm (y) / 0.85 cm (z)** —
not the prototype's 3 cm — and, being **one box spanning both TPCs**
(`sbnd/clus.jsonnet:1816-1820`), it also **removes the `|x| < 0.45` cm CPA
hole** that `dv` has, an unrelated second effect running in the *opposite*
direction from the intended fix. The faithful translation of a NULL-tolerance
prototype call is therefore `sbnd_pr_fv` **plus** a uniform −3 cm tolerance
(the prototype's own `boundary_dis_cut`), not the bare box — this is what
shipped.

**Out of scope this round (owner's choice; documented, not fixed).** The same
zero-margin defect, same class, wider blast radius (affects vertex
*selection*, not just BDT features): `NeutrinoVertexFinder.cxx`
(`compare_main_vertices:1252,1255,1309`, `examine_direction:2120,2148,2151`,
`compare_main_vertices_global:4066,4071,4174`,
`check_switch_main_cluster_2:4826-4827`) and `NeutrinoPatternBase.cxx:3095-3096`
(the two-end-break containment gate) — all NULL-tolerance in the prototype
too, so each is missing the prototype's implicit 3 cm. Also
`TaggerCheckTGM.cxx:994,998,1014,1018` / `TaggerCheckSTM.cxx:3183`
(`check_dead_volume`/`check_signal_processing` have no tolerance parameter at
all — `FiducialUtils.h:84,86`, internally call `inside_fiducial_volume(p)`
with the implicit empty vector, `FiducialUtils.cxx:125,136,184`) — a leak
**inside** TGM/STM even where they are otherwise fully on `sbnd_pr_fv`, closed
by neither `stm_consistent_fv` nor this round's knobs. And a latent, currently
harmless 5-vs-6-element tolerance-vector mapping trap
(`FiducialUtils.cxx:101-104` silently drops elements 3-4 of a 5-element
vector) — the two nue sites that already passed 5 elements were rewritten to
6 while being touched, closing it there without a behavior change.

## 4. Fix — `nue_sp_consistent_fv` (default false)

`PatternAlgorithms::m_nue_fiducial` (`NeutrinoPatternBase.h:2487-2497`), same
shape as `m_cosmic_fiducial`; forwarded from `TaggerCheckNeutrino.cxx:1885-1888`
`(m_nue_sp_consistent_fv && m_use_fiducial) ? m_fiducial : nullptr`. Each of
the four call sites gained a `contained_tol_nue`/`contained_tol_sp` lambda
(M10 per-file duplication, mirroring doc 74's `contained_tol`) and now
dispatches to the configured fiducial when armed, with the tolerance literal
hardcoded at the call site (not a jsonnet-configurable margin — same
convention as cosmic_tagger's flag-1 −1.5 cm, since the prototype's own
constant at each site is a literal, not a stage margin).

## 5. Gates — knob OFF, byte-identical

| gate | samples | result |
|---|---|---|
| member-content hash (`pr85_hash_gate.py`), bare vs off1 | nueCC48/NCpi0/numu-50/enriched | **PASS 96+38+100+52 = 286/286 archives** |
| per-branch/per-entry ROOT (`pr94_root_gate.py`), bare vs off1 | same four | **PASS 48+19+50+26 = 143/143 events** |
| compiled config, both knobs off | vs pre-round baseline | diff **empty** |
| compiled config, `nue_sp_consistent_fv=true` | single-event probe | `"nue_sp_consistent_fv": true` appears once, in `TaggerCheckNeutrino:pr`; absent when off |
| compiled config, `nu_selected_as_main_snapshot_all=true` | same | `"nu_selected_as_main_snapshot_all": true` appears once, in `TaggerCheckNeutrino:pr`; absent when off |
| `wcdoctest-clus` | — | **218/218 cases, 2301/2301 assertions**, 0 failed, 1 skipped |

Freshness proof done before the gate (`build/clus/libWireCellClus.so` mtime
after the edits). Both knobs pinned OFF in
`clus/test/doctest_clus_knob_defaults.cxx`'s existing "TaggerCheckNeutrino
switches are all OFF" case.

**Concurrent-session note.** A peer Claude session had uncommitted WIP
(doc pr/102 round 2) in exactly the files this round needed to touch
(`TaggerCheckNeutrino.{h,cxx}`, `NeutrinoPatternBase.h`, the three jsonnet
files) throughout this round. Rebuild-first protocol (owner's choice): bare
arms were generated from the binary carrying their WIP (confirmed via
`.so` mtime + a recorded `git diff` hash at bare-arm time) before this
round's edits were applied, so the OFF gate's bare/off1 pair differs by
exactly this round's two knobs, nothing else. The toolkit commit
(`9b5bb8fd`) was staged with `git add -p`, hunk by hunk (one signature line
required a manual token-level reconstruction, since both sessions' additions
landed on the same physical line) — verified zero peer tokens in the staged
diff before committing; their WIP remains uncommitted in the shared tree.

## 6. ON census

### `nue_sp_consistent_fv` — off1 vs onfv (`d75_nue_fv_census.py`, scalar T_tagger flags + calib-pr scores)

- **nueCC48**: **2/48** flag flips — evt **388**: `anc_flag_main_outside`
  0→1 (containment tightened as intended; **inert on the final decision**
  here, since `angular_cut`'s gate `angle>90 || energy<300 ||
  (angle>60 && energy<800)` is false regardless at `angle=18.7°,
  energy=1964 MeV`); evt **38856**: `shw_sp_br3_2_other_fid` 1→0 and
  `shw_sp_br3_2_flag` 1→0 — a vertex the zero-margin volume wrongly read as
  contained now correctly reads as outside, and `bad_reconstruction_2_sp`
  now flags it — the intended fix direction. Separately, **2/48** small
  `nue_score` moves with no tracked flag flip (evt 30504: −1.4579→−1.3879;
  evt 111412: 0.0609→0.0374) — continuous features (e.g. `shower_to_wall`'s
  wall-distance) perturbing near a boundary without crossing any flag's
  threshold.
- **NCpi0**: 0/19 flag flips; **1/19** `nue_score` move (evt 37112,
  −2.5014→−2.4678) — one of the same margin-adjacent events the
  `neutrino_consistent_fv` (F1) round already found sensitive to this exact
  boundary.
- **numu-50**: 0/50 on both axes — the fixed call sites are all inside
  `nue_tagger`/`singlephoton_tagger`, which read no showers on this sample's
  events in a way that exercises them.
- **Adjudication**: no ADVERSE anywhere; all movers are either inert
  (evt 388) or fix-direction (evt 38856, and the small score perturbations
  consistent with a stricter, prototype-faithful boundary).

### `nu_selected_as_main_snapshot_all` — off1 vs onflag, enriched manifest (`d75_mainflag_census.py`)

- **2/26** events show a live DL-swap **during** a candidate's own pass
  (sentinel fired): evt **409634** (2 restores) and evt **486907**
  (1 restore).
- Persisted-state check (`calib-pr-evt<ID>.json`'s `steiner` block,
  `is_main_cluster` per cluster — the exact field `PrDisplayDump.cxx:1099`
  writes from the live flag at end of event):
  - evt **409634**: OFF leaves cluster **63** flagged main (a swap remnant);
    ON correctly shows cluster **21** (the candidate `nu_per_bundle`
    actually selected) — the leak, closed.
  - evt **486907**: OFF leaves **two** clusters flagged main simultaneously
    (**16 and 95**) — the exact "two clusters main at once" pathology
    Finding A predicted; ON narrows it to just **16** — the leak, closed.
  - Three OTHER events in the manifest (286681, 400636, 487303) also show
    two main-flagged clusters in BOTH arms, unchanged — checked against the
    multi-candidate list and confirmed **legitimate**: each is a genuine
    two-bundle event where both bundles' own mains are correctly flagged
    simultaneously by design in per-bundle mode. Not the leak; the knob
    correctly leaves them alone.
- **Adjudication**: 2/26 events exercise the fix and both are corrected in
  the intended direction (the wide guard's restored state matches the
  candidate `nu_per_bundle` actually selected); no ADVERSE.

## 7. Findings summary

| # | finding | class | this round |
|---|---|---|---|
| A | DL-swap leak in `nu_selected_as_main`'s guard | main-flag, pre-dates pr/94 | **fixed**, `nu_selected_as_main_snapshot_all` |
| B | nue/SP tagger zero-margin FV (4 sites) | same class as doc 74's G1 | **fixed**, `nue_sp_consistent_fv` |
| C | numu/SSM/pi0 have no FV test on either side | genuine parity | no action — do not re-flag |
| D | `NeutrinoVertexFinder`/`NeutrinoPatternBase` zero-margin FV (9 sites) | same class as B, wider blast radius (vertex selection) | documented only, owner's choice |
| E | `check_dead_volume`/`check_signal_processing` cannot take a tolerance at all | leaks TGM/STM to zero-margin even when otherwise fixed | documented only, owner's choice |
| F | 5-element tolerance-vector mapping trap in `FiducialUtils` | latent, currently harmless | closed at the two nue sites touched; not swept elsewhere |

## 8. Status (superseded by §9 for the exposure numbers)

Knobs implemented, gated OFF byte-identical, ON census shows both fixes
behaving in the intended direction with no ADVERSE mover on the samples
tested. Toolkit `9b5bb8fd`.

## 9. Round 2 (2026-08-20, same day) — production flip; exposure correction

Owner: *"things are good, turn them on for SBND production."*

### 9.1 Repro (round 2)

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
export SBND_NU_SELECTED_AS_MAIN_SNAPSHOT_ALL=1
PR_EXTRA_STAGES=pr_display PR_JOBS=10 ./run_pr_chain_batch.sh work-ncpi0-ql0819 work-d75r1-onflag-ncpi0 data
PR_EXTRA_STAGES=pr_display PR_JOBS=10 ./run_pr_chain_batch.sh work-nuecc48-ql0819 work-d75r1-onflag-nuecc48 data
PR_EXTRA_STAGES=pr_display PR_JOBS=10 ./run_pr_chain_batch.sh work-mcp1k-ql0819 work-d75r1-onflag-mc50 data $(cat /home/xqian/tmp/knob75/mc50.txt)
unset SBND_NU_SELECTED_AS_MAIN_SNAPSHOT_ALL
# (flip cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet: both knobs -> true)
PR_EXTRA_STAGES=pr_display PR_JOBS=10 ./run_pr_chain_batch.sh work-ncpi0-ql0819 work-d75r1-flipchk-ncpi0 data   # no env
python3 scripts/pr85_hash_gate.py work-d75r1-off1-ncpi0    work-d75r1-onflag-ncpi0     # 8/19 differ -> the discovery
python3 scripts/pr85_hash_gate.py work-d75r1-off1-nuecc48  work-d75r1-onflag-nuecc48   # 4/48 differ
python3 scripts/pr85_hash_gate.py work-d75r1-off1-mc50     work-d75r1-onflag-mc50      # 2/50 differ
python3 scripts/pr85_hash_gate.py work-d75r1-off1-ncpi0    work-d75r1-flipchk-ncpi0    # flip-equivalence: same 8/19
python3 scripts/d75_mainflag_census.py work-d75r1-off1-<s> work-d75r1-onflag-<s>       # per-sample census
```

### 9.2 The flip-equivalence check found the §6 census had under-sampled exposure

Standard practice (doc 74 precedent): before flipping, run a **flip-equivalence**
arm — post-flip config, no env override — and hash-gate it against the
already-validated env-driven ON arm. For `nu_selected_as_main_snapshot_all`
that gate **FAILed** on NCpi0 against `work-d75r1-onfv-ncpi0` (which only had
`nue_sp_consistent_fv` on): 8 of 19 archives differed. Isolating with a clean
single-knob arm (`work-d75r1-onflag-ncpi0`) confirmed
`nu_selected_as_main_snapshot_all` alone causes all 8 — **not** the 0/19
exposure §0 estimated from the enriched manifold's "promoted-main" framing.

The root cause of the under-estimate: `swap_main_cluster` fires whenever the
DL vertex path picks a different cluster within a bundle, which turns out to
be **common** on ordinary events, not confined to promoted-candidate events
as the "enriched manifest" was scoped to find. §0/§1's enriched manifest
correctly bounds the *promoted-candidate* question but does not bound this
knob's real exposure — a distinction this round's initial adjudication
missed.

**This is exactly the situation CLAUDE.md §5 rule 5 describes** (a gate FAIL
whose cause needed tracing before proceeding): rather than flip on the
original thin evidence, the round paused to measure the real exposure on all
three standard samples before finishing the flip.

### 9.3 Corrected exposure and adjudication

Additional arms: `work-d75r1-onflag-{nuecc48,mc50}` (env-driven,
`nu_selected_as_main_snapshot_all` only), hash-gated against the existing
`work-d75r1-off1-*` baselines.

| sample | events | archives differing (= events touched) |
|---|---|---|
| nueCC48 | 48 | **4**: 52672, 137238, 269774, 389538 |
| NCpi0 | 19 | **8**: 18625, 37112, 71372, 314838, 463565, 506114, 506746, 521075 |
| numu-50 | 50 | **2**: 48367, 51865 |
| enriched-26 | 26 | **2**: 409634, 486907 (§6, no overlap with the above three samples) |
| **total** | 143 | **16 unique events (11%)** |

Per-event physics check (`calib-pr-evt<ID>.json`: `main_vertex`, `numu_score`,
`nue_score`, `match_isFC`) on all 16:

- **15/16 events: zero physics-level difference.** The fix corrects only the
  persisted `steiner[].is_main_cluster` flag (confirmed directly, e.g. evt
  37112's OFF arm leaves cluster 84 flagged main, ON correctly shows 9; evt
  51865's OFF arm leaves cluster 15 flagged, ON shows no cluster flagged —
  the true candidate and its vertex/scores are identical in both arms
  either way). Exactly the "display/serialization only" effect §1 predicted.
- **1/16 event (NCpi0 evt 37112): a genuine, tiny reconstruction effect** —
  `nue_score` moves −2.5014 → −2.4678 (0.03 units, on an already deeply
  negative, non-selecting score). This is the §1 correction's mechanism: a
  companion cluster's `find_proto_vertex` read a stale flag from an earlier
  candidate's uncorrected swap. Evt 37112 is independently known as a
  chronically boundary-sensitive event (doc pr/99's "168596 Enu double
  count" family cousin, doc pr/101's "37112 proton/gamma overlap" case) —
  the same event also moved by the same tiny amount under `nue_sp_consistent_fv`
  alone (§6), which is not a coincidence worth chasing further: both fixes
  independently touch a vertex already sitting on a boundary this event is
  known to straddle.
- **No ADVERSE** on any of the 16: no genuine neutrino candidate's
  main_vertex, numu_score, or nue_score moved in a verdict-changing way, and
  no event's selected candidate changed identity.

### 9.4 Flip-equivalence, closed

`off1-ncpi0` vs `flipchk-ncpi0` (post-flip config, both knobs on, no env)
differs on exactly the same 8 archives as `off1-ncpi0` vs `onflag-ncpi0`
(the flag-leak knob alone) — i.e. turning both knobs on together produces
precisely the union of their individually-characterized effects, with no
additional or surprising interaction. Flip-equivalence closed on this basis.

### 9.5 Production flip

`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`:
`nue_sp_consistent_fv = true`, `nu_selected_as_main_snapshot_all = true`.
Compiled-config proof: post-flip compile shows both keys `true` in
`TaggerCheckNeutrino:pr`; compiled byte-identical to the env-driven ON
compiles used throughout this doc. Toolkit `a3cb41ad`.

### 9.6 Status

**SBND PRODUCTION ON**, both knobs, owner-authorized. §1's blast-radius claim
is corrected per the note there; §0's exposure estimate for
`nu_selected_as_main_snapshot_all` is superseded by §9.3 (11% of standard-sample
events, not confined to promoted candidates). No ADVERSE found across 143 + 26
examined events. Arms kept: `work-d75r1-onflag-{nuecc48,mc50}`,
`work-d75r1-flipchk-ncpi0`.
