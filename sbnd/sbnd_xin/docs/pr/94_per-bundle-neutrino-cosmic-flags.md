# doc pr/94 — per-bundle T_tagger/T_kine rows: cosmic-tagged sibling bundles
# must not discard a co-bundled neutrino candidate (SBND run 18255)

**Status (2026-08-19): Phase 1 DONE — schema + plumbing SHIPPED, knob OFF
(`nu_per_bundle`, default false).** Byte-identical proven both at compile
time (compiled JSON diff empty vs pre-edit baseline) and at runtime (every
pre-existing `T_tagger`/`T_kine`/`T_bad_ch`/`T_proj`/`T_proj_data`/
`T_rec_charge`/`Trun` branch, on a real event, bit-for-bit identical between
OFF and ON). Knob ON books exactly the 12 new branches, all correctly at
their sentinel defaults (-1 scalars, empty vectors) — nothing populates them
yet, as designed. `wcdoctest-clus` 2215/2215. See §9 Phase 1 for full
results and Repro. Phases 2-6 (per-bundle selection logic, opening convicted
bundles, Bee, scale-up, SBND flip) are still plan-only, not started. Do not
treat any Phase 2-6 numbers below as implemented — they are measurements
against the *current* (pre-Phase-2) production chain, used to size the
problem and the fix.

Owner's framing, verbatim, of what "done" means: *"Now, I wonder how the
flags (including FC) are saved for this kind of cases? In the final result, I
think the right way is for each main activity, we should have a set of flag
to label it... For each bundle in coincidence with beam, we should have a set
of results (one neutrino, potentially multiple set of cosmic flags)... I
think we need the full per-bundle neutrino results incl. energy and BDT...
what we need is [that] the cosmic-related tagger (one per bundle) should be
able to handle the second activities, since the neutrino candidate can be the
second candidate."* And on validation: *"For the final validation, we should
have human verification as well... I will need to have bee links, and you
should ensure the root file are synced and good."*

## Repro

Numbers in §1 come from the existing (unmodified) production chain, run
before this plan existed, using the standard `run_pr_chain_batch.sh` pattern
documented in doc pr/93 §Repro:

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
ls -la --time-style=full-iso build/clus/libWireCellClus.so   # freshness proof, M1
./build/clus/wcdoctest-clus                                  # 2215/2215 pass

cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# The 2000-event mcp2k population arm used throughout this doc:
PR_JOBS=32 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh \
    work-cbr3-census-on work-pr83r4-m2kon data $(cat /home/xqian/tmp/mcp2k.ids)
# (work-pr83r4-m2kon predates this doc -- a pr/83-round-4 arm reused here for
# its population size; any equivalent 2000-event mcp2k arm on current
# production reproduces the same shape.)

# Headline counts (§1 table), from the arm's own artifacts:
cd work-pr83r4-m2kon
grep -h "main(s) evaluated" pr_evt*/wct_pr_evt*.log | \
    sed 's/.*us: \([0-9]*\) main(s) evaluated.*/\1/' | sort -n | uniq -c
grep -lh "TaggerCheckNeutrino: selected main cluster" pr_evt*/wct_pr_evt*.log | wc -l
grep -lh "cosmic-tagged.*skipping" pr_evt*/wct_pr_evt*.log | wc -l
awk 'NR>1 && $NF=="cosmic-tagged"{print $3}' nusel-events.tsv | sort > /home/xqian/tmp/ct.ids
for e in $(cat /home/xqian/tmp/ct.ids); do
    grep -q "selected main cluster" pr_evt$e/wct_pr_evt$e.log 2>/dev/null && echo $e
done | wc -l   # -> 68 mislabelled

# The evt 395148 case study (single bundle, two activities):
cd work-prod0819-mcp1k/pr_evt395148
grep -nE "cluster 21|cluster 10 |demoted main|OC53SKIP" wct_pr_evt395148.log
python3 -c "import uproot; f=uproot.open('tracking-pr.root'); \
    [print(k, f[k].num_entries) for k in f.keys()]"   # T_tagger;1 1, T_kine;1 1
```

## 1. Context

SBND event 18255/395148 exposed a structural gap. Its single in-beam bundle
(`matched_flash_gid=0`, t0 1.533 us) holds **two distinct physical
activities**: cluster 10, a 533 cm STM-tagged cosmic muon (the bundle's
`flag_main_cluster`), and cluster 21, a 198.9 cm contained neutrino
candidate — one of **14 demoted mains** in that same bundle.

The physics is right — cluster 21 is selected and fully reconstructed
(`cosmict_flag 0`, `numu_score 3.655`, `Enu 992.1 MeV`) — but the **output
cannot express it**. `T_tagger`/`T_kine` hold one entry per event, for the
one selected candidate; no branch records the gatekeeper TGM/STM/FC/LM
verdicts at all, and none records that a cosmic-tagged sibling shared the
bundle. Our hand-scan TSV therefore labels the event `cosmic-tagged`, so a
cut on it silently discards a good neutrino.

Measured on 2000 mcp2k events (`work-pr83r4-m2kon`):

| quantity | value |
|---|---|
| events with a selected candidate | 905 |
| events where a cosmic main was skipped | 812 |
| **events with BOTH** (the 395148 pattern) | **93** |
| events labelled `cosmic-tagged` that DID select a candidate | **68 / 629 (10.8 %)** |
| selected candidates with **no** gatekeeper flags (out-of-scope) | 43 / 905 (11 are > 10 cm, max 152 cm) |
| in-beam **bundles** per event | 0 -> 379, 1 -> 1575, 2 -> 46 (**max 2**) |
| main **activities** per event (in-window, tagger-evaluated) | median 2, max 11 (395148 has 14 demoted) |
| in-beam bundles currently **not opened** (cosmic-convicted) | 870 |
| selections arriving via `nu_fallback_demoted_mains` | 31 / 905 |

Shape: **few bundles per event (<= 2), many activities inside a bundle.**

Cluster 21 was selected through the **`nu_fallback_demoted_mains`** path
(`TaggerCheckNeutrino.cxx:1094` — *"no main-cluster candidate; selected
demoted main 21 ... of 14 demoted"*), not a bundle-veto length exemption. All
main-cluster candidates in that bundle were cosmic-skipped first; the
demoted fallback then supplied the neutrino.

### Bee: checked directly, "no change needed" does not hold

`MultiAlgBlobClustering::fill_bee_pf_tree` (`:1159-1165`) roots its whole BFS
at a single `tf->get_main_vertex()` and returns early if absent, and it reads
the **one** unnamed `grouping.get_track_fitting()` slot
(`Facade_Grouping.h:372-373`, a plain overwrite). With two bundles, the
second `set_track_fitting()` overwrites the first, so **one bundle's PF
would be lost entirely**, not merely flattened. The Bee `mc` layer format
itself is already a forest (verified: `0-mc.json` in `mabc-pr.zip` is a bare
JSON array — 3 roots for 395148), so the *format* needs nothing; the
*producer* needs a loop. There is also a real node-id collision: the
`pf_unique_node_ids` reissue path (`:1806-1814`) starts a per-call
`used_node_ids` set at `1000000`, which would collide across bundles.
Verified `Bee::ParticleTree::set_particles()` (`util/src/Bee.cxx:549-551`)
does a plain `m_data = particles_array` — an overwrite, not an append — so a
naive "call it twice" would also lose bundle 1's PF when bundle 2 is
written. So Bee gets a small, contained change (§8.6), not a redesign.

## 2. Goal

Per event, emit **one `T_tagger` / `T_kine` entry per in-beam-window bundle**
(0–2), each carrying that bundle's complete neutrino result — vertex,
kinematics, `Enu`, `nue_score`, `numu_score`. **At most one neutrino
candidate per bundle**; a bundle's entry is *effectively one event* from the
final-results point of view. Inside each entry, the **cosmic-related tagger
results are vectorised over that bundle's main activities**, so a cut on a
cosmic flag can never discard a sibling activity.

Bundles out of coincidence with the beam window are untouched. Other trees
may carry duplicate entries for the same event — that is acceptable (owner).

## 3. Full ROOT-tree inventory (the sync question, answered up front)

Grepped every ROOT-writing component type (`TFile::Open`/`new TTree` in
`root/src/*.cxx`, `clus/src/*.cxx`) against what is actually wired into any
SBND `clus.jsonnet`/`wct-pr-perevt.jsonnet` config. Exactly **two** visitor
types appear anywhere in the SBND PR configs — everything else found by that
grep (`CelltreeFrameSink`, `CelltreeSource`, `HistFrameSink`, `MagnifySink`,
`MagnifySource`, the PDHD/PDVD op-waveform/flash sources, `RootfileCreation`,
`SCEFieldTH3`, `UbooneGeomHelper`, `UbooneMagnifyTrackingVisitor`) is unused
by SBND, or (`UbooneMagnifyTrackingVisitor` specifically) wired into no
experiment config at all.

**File 1 — `tracking-pr.root`**, written by two visitors in sequence,
`RECREATE` then `UPDATE`:

| Tree | Writer | Fill() site | Entries/event today |
|---|---|---|---|
| `T_bad_ch` | `SbndPrMagnifyTrackingVisitor` | `:180`, inside a dead-channel loop | N (already multi-row) |
| `Trun` | ″ | `:210`, unconditional | 1 |
| `T_proj_data` | ″ | `:295`, one Fill of vector branches | 1 |
| `T_rec_charge` | ″ | `:506` (vertices) + `:589` (fit points) | N (already multi-row) |
| `T_proj` | ″ | `:128`, empty, "reader compatibility" | 1 |
| `T_tagger` | `UbooneTaggerOutputVisitor` | `:1092`, unconditional | 1 -> **becomes N (§4)** |
| `T_kine` | ″ | `:1123`, unconditional | 1 -> **becomes N (§4)** |

Only `Trun` carries run/subrun/event, and only once — but the filename is
already per-event (no templating; `SbndPrMagnifyTrackingVisitor` uses
`RECREATE` each event), so **no tree needs a run/subrun/event branch to
group back to an event**; the file already is the event. What's missing is a
*within-file* grouping key once `T_tagger`/`T_kine` go multi-row — covered
by §4's new `cluster_id`/`matched_flash_gid`/`nu_index` branches.

`T_bad_ch` and `T_rec_charge` are **already** multi-row per event (per dead
channel, per fitted 3-D point respectively) and already join back via
`cluster_id` — the owner's "duplicate entries OK" note applies directly to
these: if §8's Phase 3 opens more bundles, they just grow, no schema change
needed. `Trun`/`T_proj_data`/`T_proj` stay legitimately 1/event (job-level
metadata, not per-candidate) — §4 does not touch them.

**File 2 — `tracking-stm.root`** (`SbndMagnifyTrackingVisitor`,
`sbnd/clus.jsonnet:1979-1996`, named-slot input `"stm"`): same tree set
(`T_rec_charge`/`T_proj_data`/`T_bad_ch`/`Trun`), but **not produced by our
production runs today** — `run_pr_chain_batch.sh`'s header states `-stm-fit`
(which activates this stage) is deliberately omitted. Checked its source
slot: `TaggerCheckSTM.cxx:517-529` already accumulates **every**
STM-evaluated main cluster's fit into one merged `TrackFitting` handed to the
`"stm"` slot — i.e. it is already bundle/cluster-agnostic, not tied to "the
selected candidate." **No change needed here** even if this stage is ever
turned back on.

**Everything else** (`.time.meta`, `.log`, `mabc-pr.zip`,
`pctree-pr-evt<ID>.tar.gz`, `nusel-evt<ID>.tsv`, `calib-pr-evt<ID>.json`
under `PR_EXTRA_STAGES=pr_display`) is JSON/text/zip, not a ROOT tree — no
entry-count sync concept applies; §7 covers the tsv/json side directly.

**Net: the sync surface is exactly `T_tagger`/`T_kine`, and it's fully
scoped by §4.** No other tree in either file needs a schema or cardinality
change.

## 4. Design: T_tagger/T_kine become multi-entry rows, not vectorised branches

**Decision, with the owner's explicit sign-off to prefer whichever is
cleaner:** rows, not vectors. `wire-cell-prod-nue-port.cxx:2980-3013` — the
prototype's own `T_tagger` writer — already loops over `neutrino_vec` (its
in-beam bundles) and calls `Fill()` **inside** the loop: one row per bundle.
That is decisive porting-fidelity precedent, and it means the ~890 existing
scalar branches (`nue_score`, `numu_score`, the `cosmict_flag` family, every
`kine_*`) need **no change at all** — a row already is one candidate.
Vectorising instead would require touching every one of those 890 bindings
(`SCALAR_BR` hardcodes `/F`) for no fidelity benefit.

**Divergence from the prototype, deliberate and documented:** WCP fills
`T_kine` **outside** its bundle loop (`:3107`), so it silently keeps only the
*last* bundle's kinematics — indistinguishable from a bug, and not listed in
`porting_dictionary.md` as an intentional convention. Make `T_kine`
multi-entry too, `Fill()`d in lockstep with `T_tagger` inside the same
per-bundle loop iteration, so `T_tagger[i] <-> T_kine[i]` are positionally
synced by construction.

**New identity branches** (neither tree carries any today; only `Trun`
carries run/subrun/event, once, and the filename is already per-event so
no run/subrun/event branch is needed for grouping): `cluster_id/I` (the
selected main cluster, joins to `T_rec_charge.cluster_id`),
`matched_flash_gid/I`, `nu_index/I` (0..N-1 ordinal, documents the
`T_tagger[i]`<->`T_kine[i]` sync explicitly rather than leaving it implied).

**Vectorised per-activity cosmic block**, *inside* each row — the one thing
that is genuinely list-valued (N activities per bundle), one element per
main activity (`flag_main_cluster` or `demoted_main`) in that bundle:
`act_cluster_id[]`, `act_length_cm[]`, `act_is_selected[]`,
`act_is_demoted[]`, `act_tgm[]`, `act_stm[]`, `act_fc[]`, `act_lm[]`,
`act_evaluated[]`.

`act_evaluated[]` is load-bearing: `normalize_cluster_flags`
(`MultiAlgBlobClustering.cxx:100-131`) back-fills every missing flag with 0,
so after it runs **"exonerated" and "never evaluated" are
indistinguishable**. `lm_flag` is a bare scalar (no `flag_` prefix) with a
`-1` sentinel.

Mechanics: branch schema is hard-coded (~890 explicit `Branch()` calls);
each new field needs a member on `PR::TaggerInfo`/`KineInfo`
(`clus/inc/WireCellClus/NeutrinoTaggerInfo.h`), a fill site, and a
`SCALAR_BR()`/`VECTOR_BR()`/explicit `Branch()` line. `std::vector<float>`
and `<int>` already work (~84 such branches); **avoid nested vectors** (they
need a `root/dict/LinkDef.h` pragma and fail silently without one).
Precedent for an opt-in branch that keeps the tree byte-identical when off:
`m_neutrino_type_bitmask` (`UbooneTaggerOutputVisitor.h:46`,
`.cxx:29,81-82`, `clus.jsonnet:2442-2443`).

**Hazard:** `UbooneTaggerOutputVisitor.cxx:1126`'s `Write()` has no
`kOverwrite`. If the writer were ever invoked more than once per file, ROOT
creates invisible `T_tagger;2`/`T_kine;2` cycles that both `uproot` and
every existing gate silently resolve to the latest cycle of — i.e. a
duplicate-fill bug would go undetected. Keep the full per-bundle fill loop
**inside the one `visit()` call**; never call the visitor itself per bundle.

## 5. Design: TaggerCheckNeutrino per-bundle loop

Replace "select one main across the event" with "for each in-window
`matched_flash_gid`, select at most one candidate and run the chain".

- Enumerate in-window gids; within each, reuse today's rule (longest
  surviving main, then the `nu_fallback_demoted_mains` fallback) — scoped to
  the bundle.
- Bypass, in per-bundle mode only, the three cosmic vetoes that exist purely
  to pick a single event winner: per-main `m_nu_skip_cosmic` (`:983-996`),
  the `cosmic_gids` bundle veto (`:963-978`, `:997-1021`), and
  `m_skip_cosmic_companions` (`:1107-1118`).
- Everything inside the chain is **already per-candidate** (stack locals:
  `pattern_algos`, `pr_graph`, `tagger_info`, `kine_info`, all the
  accumulator maps) — including the DL/SCN vertex call (`:1663-1672`), which
  builds its cloud from the candidate's own `pr_graph`. Keep `main_cluster`
  bundle-local (the DL path may repoint it via `swap_main_cluster`).
- `acc_segment_id` restarts at 0 per candidate => carry it forward across
  bundles to avoid shower-id collisions.

**Plumbing N results out.** `m_track_fitter` is created **once in the
constructor** (`TaggerCheckNeutrino.h:23-26`) and never cleared;
`add_graph()` replaces `m_graph` while `sync_from_graph()` *accumulates*
clusters/blobs. So a naive loop on one fitter loses all but the last bundle.
**Chosen shape: one `TrackFitting` per bundle**, published as a vector
alongside the existing single slot (which keeps pointing at the first
bundle for backward compatibility). `Facade_Grouping` already has a
named-slot map (`:379-385`, used by `TaggerCheckSTM` with name `"stm"`) —
extend that rather than vectorising the 11 single-valued `TrackFitting`
result members. Cost to watch: `preload_clusters` -> `prepare_data()` /
`BuildGeometry()` runs per bundle — not in the existing timing table,
measure in the Phase-2 smoke step.

## 6. Design: downstream consumers become loops

Each currently assumes exactly one `TrackFitting`:

- `root/src/UbooneNumuBDTScorer.cxx:260-268`,
  `UbooneNueBDTScorer.cxx:607-618` — write back into
  `get_tagger_info_mutable()`; iterate.
- `root/src/UbooneTaggerOutputVisitor.cxx:50-58,74-76` — book against one
  `ti`/`ki` pair per bundle inside the loop, `Fill()` each => N fills, same
  file, same `visit()` call (§4 hazard).
- `root/src/SbndPrMagnifyTrackingVisitor.cxx:218,305-311` (and
  `UbooneMagnifyTrackingVisitor.cxx:180,260`) — duplicate entries per event
  are acceptable per owner; `T_rec_charge`/`T_proj_data` join back via
  `cluster_id`, which is already per-cluster there.
- `clus/src/PrDisplayDump.cxx:175-203` — singular `main_vertex`/`kine`/
  `tagger` keys, one JSON per event => make those arrays.

**Seven Python consumers need actual fixes** (not just re-baselining) — they
hard-index `array()[0]` and would silently keep reporting bundle 0 forever
otherwise: `sbnd_xin/pr_scores_table.py:192-206`,
`scripts/analysis/pr51/nuvtx_census.py:31-33`,
`scripts/analysis/pr74/pr74_pf_roots.py:61-64`,
`scripts/analysis/pr73/f3a_change_map.py:87-107` (15 sites),
`scripts/analysis/pr20/pr20_partI_pftree.py:32-36`,
`scripts/analysis/misc/ssm_tagger_ab.py:29,37,46`,
`scripts/analysis/misc/tagger_tree_ab.py:44`. Every full-array A/B identity
gate (`pr36_cmp.py`, `valfast/vf_tree_compare*.py`, `pr33_cmp.py`,
`misc/pr_arm_compare.py`, `ttag_cmp5.py`) just re-baselines once row count
changes — that's expected, not a bug to fix.

**Fidelity harness, currently unwired from automation, worth adopting as the
validation gate for this change:**
`root/apps/wire-cell-uboone-tagger-compare.cxx` already loops
`N_tagger`/`N_kine` independently (`:549-564`) and does a positional
prototype<->toolkit join (`:184-185`) — built for exactly this shape.

## 7. Design: Option A — hand-scan tooling (this repo)

`nusel-table.tsv` is one row per **qualifying bundle** (main-flagged +
in-scope, any time) — not per flash_grp. ~20 consumers parse it
(`pr_scores_table.py`, `pr3{2,4}_cmp.py`, `nusel_display/nusel_scan_viewer.py`,
census/retire scripts), so **do not change its row cardinality**.

- Add sidecar `nusel-mains-evt<ID>.tsv` + merged `nusel-mains.tsv`, keyed
  `(run, subrun, event, bundle_gid, main_id)` with per-activity
  TGM/STM/FC/LM/evaluated/selected.
- Fix `event_label` in `merge()` (`nusel_extract.py:672-687`) to consult the
  per-activity rows — that field is the one actually wrong (68/629).
- Reuse the existing regexes in `pr_scores_table.py:69-74` (`RE_SELECTED`,
  `RE_NOMAIN`, `RE_COSMIC_SKIP`); `nusel_extract.py` currently parses **no**
  `TaggerCheckNeutrino` line at all. Note logs use a Unicode arrow.

## 8. Design: remaining pieces (Bee, opening convicted bundles, runtime)

### 8.1 Opening every in-beam bundle

`ClusteringProtectBundle` withholds convicted bundles via `skip_convicted`
(C++ default true; `:201-217` gate 1, `:232-247` gate 3). Setting
`protect_skip_convicted=false` removes both, and gate 2 then admits every
in-window gid. Already plumbed (`wct-pr-perevt.jsonnet:534`,
`clus.jsonnet:1859`, currently `null`). Steiner needs nothing —
`CreateSteinerGraph` has no convicted test. Expose it in the runner as
`SBND_PROTECT_SKIP_CONVICTED` (not currently exposed).

### 8.2 Runtime

Post-selection chain is ~1.385 s median per candidate, of which **~1.15 s is
fixed SCN/DL inference** (69 % of the stage; doc pr/11 §444-454). Going from
905 to ~1670 candidates over 2000 events ~= +0.9 s/event against a
~20-24 s/event PR budget => **~+5 %**. Acceptable. If the Phase-2 smoke step
shows worse (e.g. `prepare_data` per bundle), fall back to skipping DL for
bundles below a length threshold.

### 8.3 Bee

Calling `fill_bee_pf_tree` once per bundle is the right shape, but not a
bare repeat of the existing call — three specific changes, all small:

1. **Stop resolving the fitter implicitly.** Today it reaches for the
   single unnamed `grouping.get_track_fitting()` slot; once §5's named-slot
   map holds N `TrackFitting`s, the function must take the specific one to
   render as a parameter (or a bundle index it looks up itself).
2. **Accumulate, don't overwrite.** Checked: `Bee::ParticleTree::set_particles()`
   (`util/src/Bee.cxx:549-551`) does a plain `m_data = particles_array`.
   Calling the function twice against the same `"mc"` tree today would have
   bundle 2 clobber bundle 1, not add to it. Fix: build each bundle's node
   list, prepend a synthetic root per bundle (`text:"nu (gid N)"`,
   `data.start` = that bundle's vertex, that bundle's roots nested under
   it), and call `set_particles()` **once**, after the per-bundle loop, on
   the concatenated array. Keeps the Bee layer set stable (still just
   `"mc"`).
3. **Kill the id collision.** `pf_unique_node_ids`'s `used_node_ids` set
   starts fresh at `1000000` per call (`:1806-1814`); hoist it to loop scope
   so bundle 2's reissued ids don't collide with bundle 1's.

## 9. Phased execution

Given the scope, build and validate in stages — each one a separate commit
with its own byte-identical-when-off gate, so a problem in a later phase
never puts an earlier phase's proof in doubt.

**Phase 1 — schema + plumbing, knob OFF only. DONE 2026-08-19.** Added the
C++ members (`NeutrinoTaggerInfo.h`: `cluster_id`/`matched_flash_gid`/
`nu_index` + the 9 `act_*` vectors on `TaggerInfo`), the knob member +
`configure()`/`default_configuration()` round-trip
(`UbooneTaggerOutputVisitor.{h,cxx}`), the 12 new branches booked only when
`m_nu_per_bundle` is true, and the jsonnet threading end to end
(`cfg/pgrapher/common/clus.jsonnet`'s `tagger_output(...)` builder;
`cfg/pgrapher/experiment/sbnd/clus.jsonnet`'s `clus_pr(...)` and its public
`pr(...)` wrapper, both signature + forwarded call; SBND
`wct-pr-perevt.jsonnet`'s top-level TLA, added as `nu_per_bundle = false,`
next to `neutrino_type_bitmask`). `Facade_Grouping.h`'s named-slot map
already supports what later phases need (`get_track_fitting(name)`/
`set_track_fitting(name, tf)`, used today by `TaggerCheckSTM`'s `"stm"`
slot) — no change needed in Phase 1. No behavior change: `TaggerCheckNeutrino`
still selects exactly one candidate per event; nothing populates the new
fields yet (every field reads its struct default — `-1` scalars, empty
vectors — regardless of knob state), so this phase is pure schema + booking
plumbing, as planned.

**Results:**
- `wcbuild` clean, no warnings on touched files; freshness proof done
  (`build/clus/libWireCellClus.so`, `build/root/libWireCellRoot.so`
  rebuilt after the edits).
- `./build/clus/wcdoctest-clus`: **2215/2215 pass** (unchanged from
  pre-Phase-1 baseline).
- Compiled-config proof, OFF: `wcsonnet` output on SBND evt 18255/395148
  diffed **byte-identical (empty diff)** against a pre-edit baseline compiled
  from the same commit via `git stash`.
- Compiled-config proof, ON: `--tla-code nu_per_bundle=true` — the
  `"nu_per_bundle"` key appears exactly once in the compiled JSON; absent
  when off.
- Runtime smoke on the same real event (18255/395148, `work-mcp1k-cb0805`
  pctree): OFF run's `T_tagger` has the pre-existing **1217 branches**; ON
  run has **1229** — exactly the 12 new ones
  (`cluster_id`, `matched_flash_gid`, `nu_index`, `act_cluster_id`,
  `act_length_cm`, `act_is_selected`, `act_is_demoted`, `act_tgm`,
  `act_stm`, `act_fc`, `act_lm`, `act_evaluated`), all at their sentinel
  defaults (`-1` / `[]`). **Every pre-existing `T_tagger` branch, every
  `T_kine` branch, and every other tree in the file (`T_bad_ch`, `T_proj`,
  `T_proj_data`, `T_rec_charge`, `Trun`) is bit-for-bit identical between
  the OFF and ON runs** — verified value-by-value with `uproot`, not just by
  schema.

**Repro:**
```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
bash -ic 'wcbuild' > /home/xqian/tmp/pr94_phase1_build.log 2>&1; echo rc=$?
ls -la --time-style=full-iso build/clus/libWireCellClus.so build/root/libWireCellRoot.so
./build/clus/wcdoctest-clus   # 2215/2215

export WIRECELL_PATH=$PWD/cfg:/nfs/data/1/xqian/toolkit-dev/wire-cell-data:/nfs/data/1/xqian/toolkit-dev/wire-cell-data/sbnd/photodet
export PYTHONPATH=$PWD/pyutil/python:/nfs/data/1/xqian/toolkit-dev/local/python:/nfs/data/1/xqian/toolkit-dev/wire-cell-python
QLDIR=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin/work-mcp1k-cb0805/ql_evt395148
PCT="$QLDIR/pctree-evt395148.tar.gz"
PIPELINE="switch_scope,unmerge_bundle,unmerge_assoc,steiner,fiducialutils,tagger_check_tgm,tagger_check_stm,tagger_check_fc,protect_bundle,steiner_refresh,tagger_check_neutrino,numu_bdt_scorer,nue_bdt_scorer,tracking_visitor,tagger_output"

# OFF (default) and ON (--tla-code nu_per_bundle=true) compiled configs,
# each with its own output_dir (a shared dir + RECREATE would silently let
# the second run clobber the first's tracking-pr.root):
mkdir -p /home/xqian/tmp/pr94_phase1/{off_run,on_run}
wcsonnet --tla-str "input=$PCT" --tla-code "anode_indices=[0,1]" \
  --tla-str "output_dir=/home/xqian/tmp/pr94_phase1/off_run" \
  --tla-code "run=18255" --tla-code "subrun=1" --tla-code "event=395148" \
  --tla-str "reality=data" \
  --tla-code "pipeline_names=[$(echo "$PIPELINE" | sed "s/[^,]\+/'&'/g")]" \
  --tla-str "save_tensors=/home/xqian/tmp/pr94_phase1/off_run/pctree-pr.tar.gz" \
  -o /home/xqian/tmp/pr94_phase1/off_run.json \
  cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet
wcsonnet --tla-str "input=$PCT" --tla-code "anode_indices=[0,1]" \
  --tla-str "output_dir=/home/xqian/tmp/pr94_phase1/on_run" \
  --tla-code "run=18255" --tla-code "subrun=1" --tla-code "event=395148" \
  --tla-str "reality=data" \
  --tla-code "pipeline_names=[$(echo "$PIPELINE" | sed "s/[^,]\+/'&'/g")]" \
  --tla-str "save_tensors=/home/xqian/tmp/pr94_phase1/on_run/pctree-pr.tar.gz" \
  --tla-code "nu_per_bundle=true" \
  -o /home/xqian/tmp/pr94_phase1/on_run.json \
  cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet

setarch x86_64 -R wire-cell -l stderr -l /home/xqian/tmp/pr94_phase1/off_run/wct.log:debug -L debug \
  -c /home/xqian/tmp/pr94_phase1/off_run.json
setarch x86_64 -R wire-cell -l stderr -l /home/xqian/tmp/pr94_phase1/on_run/wct.log:debug -L debug \
  -c /home/xqian/tmp/pr94_phase1/on_run.json

python3 -c "
import uproot
off = uproot.open('/home/xqian/tmp/pr94_phase1/off_run/tracking-pr.root')
on  = uproot.open('/home/xqian/tmp/pr94_phase1/on_run/tracking-pr.root')
tt_off, tt_on = off['T_tagger'], on['T_tagger']
print('branches off/on:', len(tt_off.keys()), len(tt_on.keys()))
print('new in ON:', sorted(set(tt_on.keys()) - set(tt_off.keys())))
"
```

**Phase 2 — per-bundle selection, knob ON, small sample only.** Turn on the
bundle loop in `TaggerCheckNeutrino` (§5), bypass the three cosmic vetoes in
per-bundle mode, wire the BDT scorers and `tagger_output` to iterate.
Validate on nueCC48 (48) + NCpi0 (19) first. Smoke-check 395148 specifically
against the expected `act_*` content (cluster 10 STM=1/FC=0, cluster 21
STM=0/FC=1/selected=1).

**Phase 3 — open convicted bundles.** `protect_skip_convicted=false` (§8.1)
is a separate, independently toggleable change — it changes what
`ClusteringProtectBundle` does to 870 in-beam bundles, on top of Phase 2's
selection change. Validate it as its own step so a regression is
attributable.

**Phase 4 — Bee (§8.3).** Independent of Phases 2-3's physics content; can
proceed once Phase 1's plumbing exists.

**Phase 5 — scale up + Option A.** Run mcp1k (1000) + mcp2k (2000). Fix the
7 Python consumers (§6) and the `nusel_extract.py` sidecar + `event_label`
fix (§7) in lockstep with the population arm, since they're what actually
reports the validation numbers.

**Phase 6 — SBND production flip.** Only after Phase 5's numbers are
reported and accepted — flip `wct-pr-perevt.jsonnet` ON for SBND, per the
owner's "default on after validation."

## 10. Verification

1. `wcbuild`, then freshness proof on `build/clus/libWireCellClus.so` (M1)
   and `./build/clus/wcdoctest-clus` (expect 2215/2215).
2. **Knob-OFF byte-identical gate**: `scripts/pr85_hash_gate.py` against the
   current production arm on all four samples; member-content hashes, never
   raw `cmp` (M2). Report labels.
3. **Compiled-config proof**: `wcsonnet` the SBND job, grep the new keys
   appear only when on (M6).
4. **Smoke on 395148**: expect 1 bundle entry, `act_*` vectors containing
   both cluster 10 (STM=1, FC=0) and cluster 21 (STM=0, FC=1, selected=1).
   Quote it.
5. **Knob-ON population arms** on nueCC48 (48), NCpi0 (19), mcp1k (1000),
   mcp2k (2000). New-branch comparison must be **per-branch value
   comparison** (existing branches unchanged, new ones present and
   populated) — a file hash cannot be the ON-arm gate since the schema
   changes.
6. Report movers: how many events gain a second entry, and how many of the
   68 mislabelled events are recovered.

### 10.1 Automated sync check — must PASS before any human review

Before spending human time, prove `T_tagger[i]` / `T_kine[i]` / the Bee `mc`
layer's per-bundle root node all refer to the *same* bundle, for every row
in the population arms. For each row `i`: read
`cluster_id`/`matched_flash_gid` (§4), cross-check `nu_x/nu_y/nu_z` against
(a) the corresponding `TaggerCheckNeutrino: selected main cluster N (t0 ...,
L ...)` log line for that `cluster_id`, and (b) the position of that
bundle's synthetic root node in the rebuilt Bee `mc.json` (§8.3). All three
must agree to floating-point tolerance. Script this once, run it over every
population arm, and report a pass count — this turns "I built N rows" into
"the rows are the right rows," and it is the load-bearing check the owner
asked for, not just the branch-value diff in step 5 above.

### 10.2 Human verification (owner review, Bee-based)

Automated gates can prove the plumbing is wired correctly; they cannot judge
whether the *reconstruction* is sensible for a real two-activity event.
Select concrete examples from the population arms for each structural case
and build Bee links for them:

| case | definition | example(s) already found in the 2000-evt mcp2k arm |
|---|---|---|
| **1 — single bundle, multiple cosmic-flagged activities, one candidate** | `n_inbeam_bundle==1`, >=2 evaluated main activities in that bundle | 395148 itself (14 demoted mains, 1 candidate); the other 92 events matching "events with BOTH" in §1's table |
| **2a — multiple bundles, both produce candidates** | `n_inbeam_bundle==2`, both bundles select a main | run 18255/1 evt 53881, 56521, 100322 (`nu-candidate` label today) — needs re-check under per-bundle logic to confirm both actually select |
| **2b — multiple bundles, only one produces a candidate** | `n_inbeam_bundle==2`, one bundle all-cosmic | evt 71681, 74128 (labelled `cosmic-tagged` today with 2 in-beam bundles — worth checking whether *either* bundle should have selected under per-bundle logic) |
| **2c — multiple bundles, neither produces a candidate** | `n_inbeam_bundle==2`, all activities in both bundles cosmic-tagged | to be identified from the population arm once Phase 2 runs (not distinguishable from today's single-winner logic/labels) |

Procedure: run Phase 2/3's knob-ON arm on nueCC48+NCpi0 first (cheap), pull
at least 2 events per case above (scan the full arm for case 2c, which
isn't identifiable under today's logic), build a Bee zip per
`make_pr_bee.py`'s existing pattern, get explicit owner sign-off case by
case before scaling to mcp1k/mcp2k. Record the decisions the way
`work/ql_labels/<tag>/` records other hand-scans (M13: fresh tag, never
overwrite).

## 11. Validation campaign / sample inventory

Samples (verified on disk; these are BNB **data** — `mcp1k`/`mcp2k` are
MCP2025C beam data, not truth-labelled numuCC):

| sample | ql_root | n | TLA |
|---|---|---|---|
| nueCC48 | `work-nuecc48-cb0805` | 48 | data |
| NCpi0 | `work-ncpi0-cb0805` | 19 | data |
| numuCC 1000 | `work-mcp1k-cb0805` | 1000 | data |
| numuCC 2000 | `work-cbr3-census-on` + explicit mcp2k id list | 2000 | data |

`work-mcp2k-cb0816` was retired 2026-08-17. `work-cbr3-census-on` (3000
`ql_evt` dirs = mcp1k 1000 + mcp2k 2000, verified disjoint by set
intersection) is the surviving Q/L root; it carries **no**
`.lineage_reality` stamp, so the runner's reality check is skipped rather
than satisfied — pass `data` explicitly and say so in any doc that uses it.
Also note `run_pr_chain_batch.sh` does **not** actually implement the M13
fresh-out_root refusal its header claims, so use a fresh tag every arm.

`PR_JOBS=32` (64 cores, 251 G RAM measured on the processing box, ~1.6 G/job
peak RSS). ~40 min per arm, ~8.5 G disk per arm (923 G free at plan time).

## 12. Deliverables (once execution begins)

- **toolkit** commit: `NeutrinoTaggerInfo.h`, `TaggerCheckNeutrino.{h,cxx}`,
  `Facade_Grouping.h`, `UbooneTaggerOutputVisitor.{h,cxx}`, both BDT
  scorers, `SbndPrMagnifyTrackingVisitor.cxx`, `PrDisplayDump.cxx`,
  `MultiAlgBlobClustering.cxx`, SBND `clus.jsonnet` + `wct-pr-perevt.jsonnet`.
- **wcp-porting-img** commit: `nusel_extract.py` sidecar + `event_label`
  fix, `run_pr_chain_batch.sh` env plumbing (`git add -f`, M9), this doc
  updated in place with each phase's results (never overwritten silently —
  append status blocks the way doc pr/93 does, per §Repro-style convention).
- Commit both, then push (owner asked for push), after each phase's gate
  passes — not held until the whole plan is done.
