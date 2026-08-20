# doc pr/94 — per-bundle T_tagger/T_kine rows: cosmic-tagged sibling bundles
# must not discard a co-bundled neutrino candidate (SBND run 18255)

**Status (2026-08-19): Phases 1, 2, 4 and 4b SHIPPED, knob still OFF
(`nu_per_bundle`, default false). Phase 3 validated and NOT recommended.
Phase 5 tooling done, population arms not run. Phase 6 (production flip) not
done -- it is blocked on the owner review below.**

**PHASE 6 DONE (2026-08-19): all four knobs are SBND PRODUCTION ON --
`nu_per_bundle`, `nu_selected_as_main`, `protect_open_convicted_bundles` and
`bee_flash_pred_min=0` (owner flip after the round-3 Bee scan). Flip-equivalence
PASS on 140 archives / 70 events plus the Q/L display; the pre-flip arm stays
reproducible via the tri-state env hooks. See §9.13. Production now emits one
`T_tagger`/`T_kine` row per in-beam bundle -- consumers reading one number per
event must use `scripts/pr94_rows.py primary_index()`. The §9.9 bookkeeping
defect (an activity reported under a bundle it was not matched to) is knowingly
carried into production.**

**ROUND 3 (2026-08-19, earlier the same day): the two owner-reported bugs from
the round-2 Bee scan are fixed, all three new knobs shipped default OFF -- §9.9 (evt
73038, the "matched to no flash" display filter -> `bee_flash_pred_min`),
§9.10 (evt 395148, the secondary activity missing the main cluster's treatment
-> `open_convicted_bundles` + `nu_selected_as_main`), §9.11 (gates: OFF
byte-identical 96/96 + 38/38 archives and 48/48 + 19/19 events; ON footprint on
nueCC48 = 1 mover of 48), §9.12 (repro). Review package `bee/pr94r3/`,
23 events, uploaded.**

Knob OFF is proven byte-identical after every phase: `pr85_hash_gate.py` PASS
on 96/96 + 38/38 archives and a new per-branch/per-entry ROOT gate PASS on
48/48 + 19/19 events, re-run after each of Phase 2, Phase 4 and Phase 4b, plus
a byte `cmp` on the `pr_display` JSON (which no other gate covers -- §9.5).
Knob ON reproduces the legacy candidate **bit-identically in 67/67 events** and
adds per-bundle rows beside it at no measurable runtime cost (-1.0 % / +0.3 %
wall, RSS unchanged). The §10.1 sync check passes on every row, and it has now
caught **three** real bugs no byte gate could (§9 Phase 4, §9.5).
`wcdoctest-clus` 2215/2215.

**Phase 4b (2026-08-19, owner Bee scan)** closes the last three consumers that
still described candidate 0 only: the Bee **point** layers
(`track_fit`/`shower_track`/`vertices`), Magnify's `T_rec_charge`/`T_proj_data`,
and the `pr_display` dump + viewer. See §9.5.

**Three things need the owner before this goes further**, all detailed in §9:
(i) per-bundle mode bypasses ONE of the three cosmic vetoes §5 named, not three
-- bypassing the per-main veto would emit a full neutrino result for every
convicted cosmic muon and would make §10.2's case 2c unreachable; (ii) whether
to patch `nusel_extract.py` in place so `nusel-events.tsv`'s `event_label` is
fixed at the source (§7) rather than only in the pr/94 sidecar; and (iii)
§10.2 case-by-case Bee sign-off, which gates the mcp1k/mcp2k arms and Phase 6.
The 2026-08-19 scan signed off on the *physics* of the nine review events; it
did not address (i) or (ii).

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

> **Correction (Phase 4b, 2026-08-19).** The sentence above is right about
> *schema* and wrong about *content*, and the distinction cost a review round.
> `T_rec_charge` and `T_proj_data` need no new branch — but both were written
> from `grouping.get_track_fitting()`, the **unnamed** slot, which in
> per-bundle mode is always candidate 0. So with the knob on they described one
> candidate while `T_tagger`/`T_kine` described N: a real sync gap, invisible to
> a schema audit because nothing about the branch list changes. Fixed in §9.5
> by looping the candidates *inside* each writer. Verified on NCpi0 evt 18625:
> `T_rec_charge` 705 -> 883 entries, `T_proj_data`'s per-cluster vectors gaining
> candidate 1's clusters, `T_tagger`/`T_kine` untouched at 2 rows each.
>
> **Answer to "how many trees are in the final root file": seven**, listed
> above, each at exactly one ROOT cycle (`;1`) — checked explicitly, because
> `UbooneTaggerOutputVisitor.cxx:1126`'s `Write()` carries no `kOverwrite` and a
> duplicated writer call would leave `T_tagger;2` cycles that `uproot` and every
> gate here silently resolve to the *latest* of. That hazard is why Phase 4b's
> candidate loops all live **inside** the writer functions: calling a writer
> once per candidate would have created exactly those hidden cycles.

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
  **Superseded by §9.5: do NOT make them arrays.** Every reader of this JSON —
  `pr_display_viewer.py` and any saved analysis — would break unconditionally,
  knob on or off, for a change that only ever matters when the knob is on. The
  shipped shape keeps all existing keys meaning candidate 0 and adds the extras
  under a new `candidates` array, which is the same key-suppression discipline
  the jsonnet knobs use.

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

> **Scoping error, corrected in Phase 4b (§9.5).** "Bee" here meant only the
> `mc` particle-flow layer. The **point** layers (`track_fit`, `shower_track`,
> `vertices`) come from two *different* functions —
> `fill_bee_points_from_pr_graph` and `fill_bee_vertices_from_pr_graph` — which
> have the same implicit-fitter problem as change 1 above and were not listed.
> The audit that produced this section went looking for callers of
> `fill_bee_pf_tree` rather than for readers of the unnamed slot; grepping the
> *symptom* (`get_track_fitting()` / `get_pr_graph()` with no argument) instead
> of the suspected *function* would have found all three at once, and is how
> §9.5 was scoped.

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

**Phase 2 — per-bundle selection. DONE 2026-08-19, knob still OFF.**
`TaggerCheckNeutrino::visit()` now builds a *candidate list* and runs the PR
chain once per candidate. The two legacy selection branches are **textually
untouched**: the beam-gate branch is gated `else if (!m_nu_per_bundle)` and the
new per-bundle branch is a sibling `else`, so with the knob off the code that
picks the single event-wide winner is byte-for-byte the pre-pr/94 code. The
~980-line post-selection body was wrapped in `for (nu_index ...)`, which is a
pure re-indent plus two mechanical substitutions -- `git diff -w` on the file
is **268 insertions / 39 deletions**, and all 39 deletions are the 35
`m_track_fitter` -> `track_fitter` renames plus the 4 intended structural lines.

Per-candidate state: `acc_segment_id` is hoisted above the loop (it restarts at
0 per candidate, so shower ids would otherwise collide between bundles); the
first candidate reuses the configured member fitter (hence the legacy path is
untouched) and later candidates get their own `TrackFitting` seeded from
`m_track_fitter->get_parameters()` -- `TrackFitting::sync_from_graph()`
*accumulates* clusters and blobs, so a shared fitter would leak bundle i-1's
charge into bundle i, and a freshly-preset one would silently drop the whole
`trackfitting_config` file. Each candidate publishes into a `"nu<i>"` named
slot; the unnamed slot keeps pointing at candidate 0. The four process-wide
audit blocks (PR30/31/32/33/36AUDIT) were deliberately left *outside* the loop,
so they stay one line per event and existing log parsers are unaffected.

Downstream consumers walk the `"nu<i>"` slots until one returns null, which is
empty in legacy mode -- so `UbooneNumuBDTScorer`, `UbooneNueBDTScorer` and the
Bee producer needed **no knob of their own** and collapse to exactly their
pre-pr/94 single-fitter code path. `UbooneTaggerOutputVisitor` keeps its knob
(it books branches) and fills `T_tagger`/`T_kine` **in the same loop iteration**,
so row i of each refers to the same bundle by construction.

**Divergence from this plan's §5, deliberate -- needs the owner's confirmation.**
§5 said to bypass *three* cosmic vetoes in per-bundle mode. Only **one** is
bypassed: the event-level `cosmic_gids` BUNDLE veto, which is the one that
actually discards a clean activity because a *sibling* was convicted -- the
395148 defect. The **per-main** veto is kept (a TGM/STM/LM-convicted activity is
not a neutrino candidate) and so is `skip_cosmic_companions`. Bypassing the
per-main veto would emit a full neutrino result -- vertex, Enu, BDT scores --
for every convicted cosmic muon, and would make this plan's own §10.2 case 2c
("multiple bundles, neither produces a candidate") unreachable by construction.
The two halves of the plan disagree; this is the reading that satisfies §10.2
and the owner's "at most one neutrino candidate per bundle". See §9.1 below for
the consequence this has for Phase 3.

**Phase 2 results** (nueCC48 48 + NCpi0 19, `PR_JOBS=32`):

| gate | result |
|---|---|
| `wcdoctest-clus` | 2215/2215 |
| compiled config, knob OFF, vs pre-Phase-2 baseline | diff **empty** |
| compiled config, knob ON | `nu_per_bundle` x2 (tagger_check_neutrino + tagger_output), `nu_per_bundle_demoted_acts` x1; absent when off |
| knob OFF, `pr85_hash_gate.py` vs `work-pr94p2-base-*` | **PASS 96/96 + 38/38** archives byte-identical |
| knob OFF, `pr94_root_gate.py` (every tree, branch, entry of `tracking-pr.root`) | **PASS 48/48 + 19/19** events |
| knob ON, primary row vs the legacy single row | **identical in 67/67 events** (vertex, `numu_score`, `nue_score`, `cosmict_flag`, `Enu`) |
| knob ON, §10.1 sync check | **PASS**, 53 + 23 rows |
| wall time ON vs OFF | nueCC48 **-1.0 %**, NCpi0 **+0.3 %** (median 25.0->24.5 s / 19.0->19.0 s); peak RSS unchanged at 1.56 G |

So per-bundle mode is **purely additive**: it reproduces the legacy answer
exactly and adds bundle rows next to it, at no measurable runtime cost. (§8.2
budgeted ~+5 %; the prediction was pessimistic because only ~10 % of events gain
a bundle and the extra bundles are mostly short shards.)

Row census: nueCC48 48 events -> **53 rows** (5 events gain a second bundle);
NCpi0 19 events -> **23 rows** (4 events gain). Of the extra rows, 8 are
"opened, nothing reconstructed" (selected activity 1.19-14.73 cm, no vertex, so
`fill_bee_pf_tree` correctly emits no particle flow) and **one is a genuine
second neutrino candidate** -- NCpi0 evt 18625 bundle gid 1000000, cluster 11,
Enu 1498.1 MeV. That is this plan's §10.2 **case 2a**, found in the wild.

**Smoke on 18255/395148** (§10 step 4) -- the motivating event, exactly as
predicted:

```
row 0: gid=0 selected_cluster=21 vtx=(-154.19,-62.46,181.99) Enu=992.1 numu=3.655 cosmict_flag=0
   cid    len_cm    sel  dem  TGM  STM  FC   LM
   10     508.5     0    0    0    1    0    0
   21     198.9     1    1    0    0    1    0
```

The row is now self-describing: the STM conviction is visibly attached to
cluster 10, *not* to the selected candidate 21, so a cut on a cosmic flag can no
longer discard this neutrino. `act_evaluated=1` on both is independently
confirmed by the taggers' own log lines ("TaggerCheckFC: beam_window_only
[0.200, 2.200) us: **2** main(s) evaluated, 19 out of window") -- the count
matches because `nu_per_bundle_demoted_acts` is wired straight from
`evaluate_demoted_mains` rather than from its own TLA, so the two cannot drift.

**Phase 3 — open convicted bundles. VALIDATED AND NOT RECOMMENDED; knob left at
its default (OFF).** `SBND_PROTECT_SKIP_CONVICTED=0` was run on top of Phase 2
on both small samples. Acceptance criterion, fixed **before** the numbers were
read: the primary row must stay bit-identical to the legacy result, since
anything else is a change to production physics for already-good events.

Result: it **fails**, and buys nothing.

| | nueCC48 | NCpi0 |
|---|---|---|
| rows (Phase 2 alone -> Phase 3) | 53 -> **53** | 23 -> **23** |
| rows with a vertex | 48 -> **48** | 20 -> **20** |
| primary row identical to legacy | **46 / 48** | 19 / 19 |

The two perturbed nueCC48 events are evt 10550 (`numu_score` -1.663 -> -1.379)
and evt 116962 (vertex moves 38 cm in z, `nue_score` -1.028 -> **-4.301**);
`ClusteringProtectBundle` lines in evt 116962 go 8 -> 16, i.e. the extra charge
really is entering the PR ensemble.

### 9.1 Why Phase 3 is inert here, and the fork it exposes

Every bundle Phase 3 opens is one the taggers **convicted** (the withheld-bundle
log lines are all `convicted TGM=1`). Phase 2 keeps the per-main cosmic veto, so
an all-convicted bundle yields no candidate and therefore **no row** -- while its
clusters still join the ensemble and perturb the reconstruction of the bundles
that do produce rows. Cost without benefit. Phase 3 only becomes meaningful if
the per-main veto is dropped too, which is the reading rejected above.

Three readings, for the owner to choose between -- this is not a call to make
silently:

- **(a) as shipped.** Per-main veto kept, Phase 3 off. An all-cosmic in-beam
  bundle produces *no row at all*, so the output does not record that the bundle
  existed. Costs nothing, changes nothing, matches "at most one neutrino
  candidate per bundle".
- **(b) the plan's literal §5.** Drop the per-main veto as well and turn Phase 3
  on. Every in-beam bundle gets a full row -- but that means a vertex, an Enu and
  BDT scores computed for a known cosmic muon, plus ~1.15 s of DL inference each,
  plus the 2/48 perturbation measured above.
- **(c) flags-only rows.** Emit a row for an all-cosmic in-beam bundle carrying
  the identity and the `act_*` block but **no** PR result (vertex 0, Enu 0,
  scores at defaults), by skipping the chain for those bundles. Complete bundle
  coverage -- which is what "for each bundle in coincidence with beam, we should
  have a set of results" literally asks for -- at near-zero cost and with no
  misleading physics. Not implemented; it is a genuine design decision.

Recommendation: **(c)**, with **(a)** as shipped in the meantime. Phase 3's
`SBND_PROTECT_SKIP_CONVICTED` env hook is wired into `run_pr_chain_batch.sh`
either way, so the arm can be re-run whenever this is settled.

**Phase 4 — Bee. DONE 2026-08-19.** `fill_bee_pf_tree` gained three optional
arguments, all defaulting to the pre-pr/94 behaviour (§8.3's three fixes):
render a caller-supplied `TrackFitting`; share the `pf_unique_node_ids` reissue
set across bundles so ids cannot collide between them; and append this bundle's
roots, wrapped in one synthetic node labelled `"nu <i> (gid G, cluster C)"`, to a
caller-owned array instead of calling `set_particles()` (a plain overwrite,
`Bee.cxx:549-551`, which would erase bundle i-1). The caller sets the
concatenation once, after the loop. The Bee `mc` layer stays a bare JSON forest,
so **no format change was needed** -- only the producer.

**One real bug, found by the §10.1 sync check and not by any byte gate.**
`fill_bee_pf_tree` also resolved its *graph* implicitly, via
`grouping.get_pr_graph()`, which is defined as `m_track_fitting->get_graph()`
(`Facade_Grouping.cxx:76-79`) -- the **unnamed** slot. Passing the right fitter
was therefore not enough: bundle i's vertex was being walked against bundle 0's
graph. It surfaced on NCpi0 evt 18625, whose second bundle reconstructed a real
1498 MeV candidate and emitted **no Bee node at all**. Fixed to `tf->get_graph()`;
an audit confirms these were the only two implicit unnamed-slot reads in the
function. This is exactly the failure the owner asked for ("you should ensure the
root file are synced and good") and it is invisible to a hash gate, because the
knob-off path never exercises it.

### 9.3 Candidate ordering, and what still reads only the primary candidate

Candidates are **ordered by their selected activity's length, longest first**
(ties by gid), not by gid. That is a correctness requirement, not cosmetics:
candidate 0 keeps the *unnamed* `TrackFitting` slot, and every consumer not
converted to walk the `"nu<i>"` slots still reads exactly that slot. Enumerating
by gid put a 1.7 cm shard from the low-gid drift side in slot 0 on nueCC48 evts
10550 / 234638 / 267597 -- and because a shard reconstructs no vertex, the three
Bee PR **point** layers (`track_fit-global`, `shower_track-global`,
`vertices-global`) came out *empty* for those events, while the real 18.5 /
114.5 / 127.2 cm candidate sat unused in slot 1. Caught by `make_pr_bee.py`
warning "has no ['track_fit-global', ...]" while building the review zips.

Longest-first is also the legacy selector's own rule, so slot 0 now holds the
same candidate the pre-pr/94 chain would have chosen, `nu_index == 0` means
"the primary candidate", and every unconverted consumer behaves exactly as it
did before.

**Known limitation at the time of the review round — CLOSED by Phase 4b
(§9.5).** These three described candidate 0 only:

- the three Bee PR point layers (`MultiAlgBlobClustering.cxx:3159-3170`, which
  read `gs[0]->get_pr_graph()`),
- `SbndPrMagnifyTrackingVisitor`'s `T_rec_charge` / `T_proj_data`,
- `PrDisplayDump`.

The judgement recorded here — that this was *"a separable follow-up rather than
an omission"*, because a second bundle's particle flow is still in the `mc`
forest with per-node coordinates — did not survive contact with the owner's
scan, which read the missing points as the defect they are. Kept as written
because the reasoning is instructive: "the information is recoverable from
another layer" is not the same as "the display is right", and the review is
about the display. All three are now per-candidate; see §9.5.

**Phase 5 — scale up + Option A. TOOLING DONE, POPULATION ARMS NOT RUN.** The
mcp1k (1000) and mcp2k (2000) arms are deliberately **not** run yet: §10.2
requires owner sign-off on the small-sample cases first. What is done:

- `scripts/pr94_rows.py` -- `primary_index()`, the shared "which row is THE
  candidate" helper. It reproduces the legacy meaning (longest selected main
  activity), so a single-bundle event reports exactly as before, and falls back
  to row 0 for pre-pr/94 and knob-off files.
- The five consumers that report one number per event and hard-indexed `[0]` now
  use it: `pr_scores_table.py` (which also gained `nu_row` / `n_nu_rows`),
  `pr51/nuvtx_census.py`, `pr74/pr74_pf_roots.py`, `pr20/pr20_partI_pftree.py`,
  `pr73/f3a_change_map.py` (14 sites). The remaining two named in §6,
  `misc/ssm_tagger_ab.py` and `misc/tagger_tree_ab.py`, print `[0]` only as a
  sample value beside a full-array comparison, so they are annotated rather than
  changed.
- `scripts/pr94_root_gate.py` -- per-branch, per-entry value gate on
  `tracking-pr.root` (the pr85 hash gate covers `mabc-pr.zip` and the pctree but
  not the ROOT file, which is precisely where this doc changes the schema).
  NaN maps to a sentinel: `T_rec_charge.reduced_chi2` legitimately carries NaN
  (2 of 962 entries on nueCC48 evt 389538 -- pre-existing, unrelated to pr/94,
  noted here and not touched), and `NaN != NaN` would make identical files
  compare as different.
- `scripts/pr94_sync_check.py` -- §10.1, described above.
- `scripts/pr94_mains_sidecar.py` -- §7's per-activity sidecar
  (`nusel-mains.tsv`) and the corrected event label
  (`nusel-events-pr94.tsv`), read from `T_tagger`'s `act_*` block.

**Deliberate deferral, owner's call.** §12 named `nusel_extract.py` itself as the
place for the sidecar and the `event_label` fix. It was **not** modified: its
`nusel-table.tsv` is parsed by roughly twenty scripts and the hand-scan viewer,
and changing its columns or its label logic in the same round as a C++ schema
change puts two unrelated risks in one commit. The consequence is real and worth
stating plainly: **`nusel-events.tsv` still carries the wrong `event_label`** --
the 68/629 miscount in §1 is *not* fixed by writing a second file next to it.
`nusel-events-pr94.tsv` carries the corrected label; migrating the consumers, or
patching `nusel_extract.py` in place, is a follow-up the owner should schedule.

### 9.2 Repro (Phases 2-4)

Every number in the Phase 2/3/4 blocks above comes from these commands. The
baseline arms are generated **before** any source edit, on purpose: Phase 2
refactors the knob-off path (the ~980-line body moves inside a loop), so a
single-event value diff is no longer adequate evidence -- the knob-off arm needs
the real manifest.

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
git log --oneline -1                       # must be the pre-Phase-2 commit
ls -la --time-style=full-iso build/clus/libWireCellClus.so   # freshness, M1

SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin; cd $SX
# 1. pre-edit baseline (knob off, old binary)
PR_JOBS=32 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr94p2-base-nuecc48 data
PR_JOBS=32 ./run_pr_chain_batch.sh work-ncpi0-cb0805   work-pr94p2-base-ncpi0   data

# ... apply Phases 2+4, then:
cd /nfs/data/1/xqian/toolkit-dev/toolkit
bash -ic 'wcbuild' > /home/xqian/tmp/b.log 2>&1; echo rc=$?
./build/clus/wcdoctest-clus                # 2215/2215
# the extraction is mechanical -- this is the proof:
git diff -w --stat clus/src/TaggerCheckNeutrino.cxx      # 268 ins / 39 del
git diff -w clus/src/TaggerCheckNeutrino.cxx | grep '^-' # all 39 accounted for

# compiled-config proofs (see the Phase 1 Repro for the full wcsonnet line);
# knob off must diff EMPTY against a `git stash`ed pre-edit compile, and
# knob on must show nu_per_bundle twice + nu_per_bundle_demoted_acts once.

cd $SX
# 2. knob-off recheck + knob-on arms (new binary)
for n in nuecc48 ncpi0; do
  q=work-$n-cb0805; [ $n = nuecc48 ] && q=work-nuecc48-cb0805
  PR_JOBS=32 ./run_pr_chain_batch.sh $q work-pr94g-off-$n data
  SBND_NU_PER_BUNDLE=1 PR_JOBS=32 ./run_pr_chain_batch.sh $q work-pr94g-on-$n data
done
# the motivating event lives in the mcp1k Q/L root, not in either sample:
SBND_NU_PER_BUNDLE=1 PR_JOBS=4 \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr94g-on-evt395148 data 395148

# 3. gates
for n in nuecc48 ncpi0; do
  python3 scripts/pr85_hash_gate.py work-pr94p2-base-$n work-pr94g-off-$n   # PASS
  python3 scripts/pr94_root_gate.py work-pr94p2-base-$n work-pr94g-off-$n   # PASS
  python3 scripts/pr94_sync_check.py work-pr94g-on-$n                       # PASS
  python3 scripts/pr94_mains_sidecar.py work-pr94g-on-$n
done

# 4. Phase 3, its own arm, judged against a criterion fixed in advance
#    (primary row must stay bit-identical to legacy -- it does not)
SBND_NU_PER_BUNDLE=1 SBND_PROTECT_SKIP_CONVICTED=0 PR_JOBS=32 \
  ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr94p3-open-nuecc48 data
```

Arm labels, for later re-checking: `work-pr94p2-base-{nuecc48,ncpi0}`
(pre-edit baseline), `work-pr94g-off-{nuecc48,ncpi0}` (knob off, final binary),
`work-pr94g-on-{nuecc48,ncpi0}` (knob on), `work-pr94g-on-evt395148`,
`work-pr94p3-open-{nuecc48,ncpi0}` (Phase 3). Phase 4b adds
`work-pr94i-{off,on}-{nuecc48,ncpi0}`, `work-pr94i-on-evt395148` and the three
`work-pr94h-disp-{base,off,on}` single-event `pr_display` arms. The
`work-pr94h-*` arms are the pre-`any_fitted` intermediates kept as the
before-side of §9.5's ON differential.

### 9.4 Review package (§10.2) -- built, NOT uploaded

Bee zips are under `sbnd_xin/bee/pr94/` (a fresh label, M13) and were uploaded
to the BNL server on 2026-08-19 at the owner's explicit request.  **One set
carries all nine scan events**, ordered by case:

> https://www.phy.bnl.gov/twister/bee/set/32a8f441-c3d8-4a80-a2a2-21c69a47c38a/event/list/

(`bee/pr94/pr94-all.zip`, index in `pr94-all.index.txt`.  The four per-case sets
uploaded first are superseded by this one and kept only as a record:
`1f6a89dc` evt395148, `3b678e3d` case2a, `f4cb4d8b` case2b, `700c45bd` case1.)

**Phase 4b rebuild — `bee/pr94/pr94b-all.zip`, BUILT, NOT UPLOADED.** The same
nine events in the same index order, rebuilt from the fixed arms
(`work-pr94i-on-*`), 4.6 MB, all nine layers on all nine events. This is what
shows the §9.5 fix: at index 1 (NCpi0 18625) the second neutrino now has
`track_fit` / `shower_track` / `vertices` points instead of a bare PF root. It
is deliberately **not** POSTed — the 2026-08-19 upload authorization was for
that scan round, and a new upload is a fresh outward-facing action
(CLAUDE.md escalation rule 6).

Scan sheet -- what each Bee index should show (`*SEL` = the bundle's selected
candidate; a trailing TGM/STM/LM is that activity's own conviction):

```
idx 0  evt 395148  1 row, 1 PF root   <-- the motivating event
   row 0 gid 0       cid 21  VTX  Enu  992.1 numu  3.655   acts: c10/508cm STM, c21/199cm *SEL

idx 1  evt 18625   2 rows, 2 PF roots <-- case 2a: TWO reconstructed candidates
   row 0 gid 1000000 cid 11  VTX  Enu 1498.1 numu  2.729   acts: c10/2cm, c11/119cm *SEL
   row 1 gid 0       cid 26  VTX  Enu  352.2 numu -1.175   acts: c2/1cm, c17/15cm *SEL, c26/12cm, c27/7cm TGM, c130/1cm

idx 2  evt 10550   2 rows, 1 PF root  <-- case 2b: 2 bundles, one reconstructs
   row 0 gid 1000002 cid 7   VTX  Enu  834.8 numu -1.663   acts: c7/19cm *SEL, c11/378cm TGM, + 7 shards
   row 1 gid 7       cid 5   no-vtx                        acts: c5/2cm *SEL, c6/1cm
idx 3  evt 234638  2 rows, 1 PF root
   row 0 gid 1000001 cid 10  VTX  Enu 1435.3 nue   4.301   acts: c10/114cm *SEL, c14/5cm TGM, + 9 shards
   row 1 gid 9       cid 4   no-vtx                        acts: c4/1cm *SEL
idx 4  evt 267597  2 rows, 1 PF root
   row 0 gid 1000000 cid 5   VTX  Enu 1866.0 nue   4.301   acts: c5/127cm *SEL, + 11 shards
   row 1 gid 5       cid 2   no-vtx                        acts: c2/3cm *SEL

idx 5  evt 116962  1 row    acts: c15/186cm TGM, c21/75cm *SEL, c22/31cm   Enu  837.3  <-- case 1
idx 6  evt 122660  1 row    acts: c9/49cm *SEL, c15/2cm TGM, + 6 shards     Enu 1624.4
idx 7  evt 360535  1 row    acts: c6/3cm, c7/101cm *SEL, c12/4cm            Enu 2259.9  (+1 all-cosmic bundle, no row)
idx 8  evt 444187  1 row    acts: c19/170cm *SEL, c23/2cm                   Enu 1075.2  (+1 all-cosmic bundle, no row)
```

The two things worth the closest look: **idx 1 (evt 18625)**, where the `mc`
layer now carries two neutrino roots and both should read as real neutrinos
rather than one being a cosmic fragment; and **idx 0/2/3/5/6**, where a TGM- or
STM-convicted activity shares a bundle with the selected candidate -- the
conviction must visibly belong to the *other* track.  The `no-vtx` rows are
bundles that were opened and reconstructed nothing (1-3 cm shards); they carry
no PF root by design, so a PF-root count below the row count is expected.

Case census over both small samples (67 events, 76 rows):

| case | n |
|---|---|
| 1 — single bundle, >= 2 evaluated activities | 49 |
| 2b — multiple bundles, one candidate | 8 |
| plain single bundle, single activity | 8 |
| 2b' — an all-cosmic bundle alongside a candidate (no row, see §9.1) | 2 |
| **2a — multiple bundles, BOTH candidates** | **1** |
| 2c — no candidate anywhere | 0 (needs the mcp2k arm; both these samples are neutrino-enriched) |

Per-activity tables for every event are in each ON arm's `nusel-mains.tsv`
(246 and 156 activity rows), written by `scripts/pr94_mains_sidecar.py`.

**Phase 6 — SBND production flip. NOT DONE, correctly blocked.**
`wct-pr-perevt.jsonnet` still has `nu_per_bundle = false`. It stays there until
Phase 5's population numbers exist and are accepted (§10.2), which needs the
small-sample human review first.

### 9.5 Phase 4b — the three consumers that still showed candidate 0 only

The 2026-08-19 owner Bee scan of the nine §9.4 events found the physics good
everywhere but one display defect: on NCpi0 evt 18625, *"for the second
neutrino, I do not see points for the track_fit and shower_track etc in Bee."*

**Symptom.** Candidate 1 appeared in the Bee `mc` (particle-flow) layer as a
correctly-labelled root with a full shower hierarchy, but contributed **zero
points** to `track_fit`, `shower_track` and `vertices`. Its vertex sat 352.7 cm
from the nearest point of every one of those layers.

**Root cause — the same unnamed-slot read, in two more places.**
`fill_bee_points_from_pr_graph` took its graph from `grouping.get_pr_graph()`
and its shower list from `grouping.get_track_fitting()`;
`fill_bee_vertices_from_pr_graph` did the same. `Grouping::get_pr_graph()` is
*defined* as `m_track_fitting->get_graph()` (`Facade_Grouping.cxx:76-79`) —
the unnamed slot, i.e. candidate 0 — so both functions re-emitted candidate 0's
trajectories once per candidate. This is the identical mechanism already fixed
once in `fill_bee_pf_tree` (§9, Phase 4); Phase 4 fixed the layer the sync check
covered and left the two it did not.

**Why the sync check missed it.** §10.1 checked the `mc` layer's roots and
nothing else, so a candidate with a root but no points passed. Check **E** now
covers the point layers.

**Fix.** Three consumers, all made per-candidate:

| consumer | change | hazard avoided |
|---|---|---|
| `fill_bee_points_from_pr_graph`, `fill_bee_vertices_from_pr_graph` | take `tf_in` + `do_reset`; graph and shower list from that fitter; caller loops the `"nu<i>"` slots | both **reset** their `Bee::Points` at entry — resetting per candidate would have left only the LAST candidate's points, a bug whose symptom ("candidate 1 now has points") looks exactly like the cure |
| `SbndPrMagnifyTrackingVisitor::write_t_rec_data` / `write_proj_data` | candidate loop **inside** each function; `T_proj_data`'s per-cluster maps merged before its single `Fill()` | looping *around* the functions would `new TTree(...)` per candidate, leaving `T_rec_charge;2` cycles that uproot silently resolves to the last of (§3 correction) |
| `PrDisplayDump` + `pr_display_viewer.py` | seven per-candidate dumps gain an optional fitter; the extras are emitted **additively** under a new top-level `candidates` array; viewer gains a "nu candidate" selector | making the seven existing keys into arrays (as §6 proposed) would have broken every reader of this JSON *unconditionally*, knob on or off |

Event-level blocks (`meta`, `steiner`, `dead`, `dqdx_ref`) are deliberately not
repeated per candidate.

**Exhaustive audit — grep the symptom, not the function.** Every
argument-less `get_track_fitting()` / `get_pr_graph()` in `clus/src` and
`root/src` was re-checked after the fix. What remains is, in full:

| site | verdict |
|---|---|
| `PrDisplayDump.cxx:175` | presence guard in `visit()` ("is this stage after tagger_check_neutrino?") — not a render |
| `MultiAlgBlobClustering.cxx:3189` | decides *whether* a PR-graph set is filled; candidate 0 always also publishes to the unnamed slot, so the decision is unchanged |
| `MultiAlgBlobClustering.cxx:3242` | `if (!tf) continue` guard ahead of the `nu<i>` loop |
| `SbndPrMagnifyTrackingVisitor.cxx:44` | the deliberate legacy fallback inside `collect_nu_fitters` |
| `UbooneTaggerOutputVisitor.cxx:53`, `UbooneNumuBDTScorer.cxx:260`, `UbooneNueBDTScorer.cxx:607` | the Phase 2 `if (fitters.empty()) fitters.push_back(tf)` fallback |
| `UbooneMagnifyTrackingVisitor.cxx:180,260` | wired into **no** experiment config (§3); left untouched per CLAUDE.md's rule on other experiments' production files |

No unconverted per-candidate reader is left in the SBND PR path.

**A gate bug found on the way, worth recording.** Check E first joined
T_tagger's `cluster_id` to the Bee layers' `cluster_id` and reported evt 10550
row 0 — a correctly-rendered *primary* candidate — as missing. Cluster ids are
re-issued by `enumerate_idents` after **every** visitor in MABC's main loop, so
the id `TaggerCheckNeutrino` recorded is a different **epoch** from the id the
Bee dump writes: on 10550 the selected activity is cluster 7 in `T_tagger` and
62 by the time the magnify visitor sees it. The join is now **positional** —
the candidate's vertex against the nearest point of each layer, which is
epoch-independent. Separation is better than an order of magnitude at both ends:
0.00-0.45 cm when rendered, 352.7 cm when not; tolerance 10 cm. (Same family as
doc 53's `real_cluster_id` epochs.)

**Gates.**

- Knob OFF byte-identical, final binary: `pr85_hash_gate` 96/96 + 38/38,
  `pr94_root_gate` 48/48 + 19/19.
- **`pr_display` has its own gate** — it is `PR_EXTRA_STAGES`-opt-in, so it is
  covered by *neither* hash gate (not in the arms) nor ROOT gate (not a ROOT
  file). Knob-OFF `calib-pr-evt18625.json` from the final binary is a byte
  `cmp` match against the same file built from a `git stash`ed pre-Phase-4b
  binary: 1 444 299 B identical. (`cmp` is legitimate here — plain JSON text,
  none of M2's archive timestamps.)
- **Knob-ON differential** vs the pre-Phase-4b ON arms: of 67 events, exactly
  **2 differ** — NCpi0 18625 and nueCC48 389538, both multi-candidate with a
  second *reconstructed* candidate. All 65 others, including the 7
  multi-candidate events whose second bundle reconstructed nothing, are
  identical. Only `T_proj_data` and `T_rec_charge` branches moved;
  `T_tagger`/`T_kine`/`T_bad_ch`/`Trun`/`T_proj` untouched. This is the sharp
  gate: single-candidate events *must* be identical, because with one candidate
  `nu0` **is** the unnamed slot and all three changes collapse to the old call.
- §10.1 sync check including new check E: PASS on all 76 rows of both ON arms;
  and it **FAILS** on the pre-Phase-4b arm at exactly evt 18625 row 1, which is
  what makes it a gate rather than a formality.
- `wcdoctest-clus` 2215/2215.

**Before/after on the reported event (NCpi0 18625, candidate 1, cluster 26):**

| layer | before | after |
|---|---|---|
| Bee `track_fit` | 705 pts, cluster 26 absent | 884 pts, present |
| Bee `shower_track` | 6223 pts, absent | 6903 pts, present |
| Bee `vertices` | 65 pts, absent | 98 pts, present |
| `T_rec_charge` | 705 entries | 883 entries |
| nearest point to candidate 1's vertex | 352.7 cm | 0.00 cm |

`pr_display` knob ON now emits `candidates[0..1]` carrying distinct vertices
(cluster 11 / cluster 26), distinct kinematics (`kine_reco_Enu` 1498.1 /
352.2 MeV) and distinct BDT scores (`numu_score` 2.729 / -1.175) — and
`top["kine"] == top["candidates"][0]["kine"]` and
`top["segments"] == top["candidates"][0]["segments"]` hold exactly, which is the
backward-compatibility guarantee proven rather than asserted.

**Repro (Phase 4b).**

```bash
cd sbnd_xin
# gate arms, final binary
for p in nuecc48:work-nuecc48-cb0805 ncpi0:work-ncpi0-cb0805; do
  n=${p%%:*}; q=${p##*:}
  PR_JOBS=16 ./run_pr_chain_batch.sh $q work-pr94i-off-$n data
  SBND_NU_PER_BUNDLE=1 PR_JOBS=16 ./run_pr_chain_batch.sh $q work-pr94i-on-$n data
  python3 scripts/pr85_hash_gate.py work-pr94g-off-$n work-pr94i-off-$n
  python3 scripts/pr94_root_gate.py work-pr94g-off-$n work-pr94i-off-$n
  python3 scripts/pr94_root_gate.py work-pr94g-on-$n  work-pr94i-on-$n   # DIFF set
  python3 scripts/pr94_sync_check.py work-pr94i-on-$n                    # incl. check E
done
# pr_display gate: baseline needs the pre-Phase-4b binary
git -C ../.. stash push -- clus/src/MultiAlgBlobClustering.cxx \
    clus/inc/WireCellClus/MultiAlgBlobClustering.h clus/src/PrDisplayDump.cxx \
    clus/inc/WireCellClus/PrDisplayDump.h root/src/SbndPrMagnifyTrackingVisitor.cxx
wcbuild && PR_JOBS=4 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-ncpi0-cb0805 work-pr94h-disp-base data 18625
git -C ../.. stash pop && wcbuild
PR_JOBS=4 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-ncpi0-cb0805 work-pr94h-disp-off data 18625
SBND_NU_PER_BUNDLE=1 PR_JOBS=4 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-ncpi0-cb0805 work-pr94h-disp-on data 18625
cmp work-pr94h-disp-base/pr_evt18625/calib-pr-evt18625.json \
    work-pr94h-disp-off/pr_evt18625/calib-pr-evt18625.json    # must be identical
```

### 9.6 Phase 5b — the dot guard (`nu_per_bundle_min_length`)

**Symptom.** The first Phase 5 population arm was reported here as "+143
gained neutrinos, validation passes". The owner rejected the number on
physics grounds without opening a file: *"107 MeV looks like a muon (mass 105
MeV) + some energy, so these must be dots like activities? I thought that we
have some cuts to avoid promoting dots like activities as neutrino
candidates?"*

Confirmed on the pre-guard mcp1k arm: of the 143 gained candidates, **87 sat
in the 100-149 MeV bin**, **143 of 155 seeds were under 5 cm**, and only **1
of the 97 non-cosmic-flagged ones** scored `numu_score > 0`. A dot fitted as
a muon at rest. The "+143" claim is **retracted**; it measured a defect.

**Root cause.** SBND runs `nu_skip_cosmic_bundle_min_length = 0`, i.e. the
legacy bundle veto removes *every* bundle-mate of a convicted main and the
chain reaches the real interaction through `nu_fallback_demoted_mains`.
Per-bundle mode drops that veto by design (§5), and nothing else in the chain
imposes a length floor — so every sub-cm shard inside a convicted bundle
became eligible to be "the neutrino" of that bundle.

**Fix.** New knob `nu_per_bundle_min_length` (cm; **C++ default 0** = the
Phase 2 behavior, no floor). SBND production value **15**. The floor applies
to an activity only when its bundle also holds a cosmic-tagged main — exactly
the bundles the legacy event-level veto emptied.

**Why that scoping is provably additive.** With
`nu_skip_cosmic_bundle_min_length = 0` both legacy emission sites reduce to
`!(0 > 0 && …)` = skip: the main loop (`:1048-1053`) and the demoted fallback
(`:1113-1120`). So legacy emits *nothing at all* from a convicted bundle, and
a floor scoped to convicted bundles **cannot remove a legacy row by
construction** — not merely "was not observed to".

#### The first cut of the guard was wrong, and the gate caught it

Scoping the floor by "does this bundle hold a cosmic-tagged main" is only
additive if "main" means the same thing it means in the legacy chain. It did
not. Legacy builds `cosmic_gids` (`TaggerCheckNeutrino.cxx:999-1011`) from
**`flag_main_cluster` activities only**; the first guard scanned `cand.acts`,
which also holds demoted mains. Bundles whose *only* cosmic-tagged activity
was demoted were therefore convicted by the guard and not by legacy, and the
floor then deleted rows legacy reports.

| arm | NCpi0 evt 114446 |
|---|---|
| OFF (legacy) | 1 row, cluster 21, `Enu` 126.4 MeV |
| ON, no guard | 1 row, cluster 21, `Enu` 126.4 MeV — identical |
| ON, first guard | **no `T_tagger` at all** |
| ON, fixed guard | 1 row, cluster 21, `Enu` 126.4 MeV — identical |

gid 0 there holds main 21 (10.9 cm, FC — the legacy selection) beside
**demoted** main 33 (2.4 cm, TGM). Cost on mcp1k: **8 events lost a vertex**
(169488, 171892, 279643, 281727, 283463, 389588, 391238, 398514).

Fix: `!a.is_demoted && (a.tgm || a.stm || a.lm > 0)` — an exact mirror of the
legacy `cosmic_gids` predicate. This is the M15 family: the guard and the
legacy veto must agree on the *scope word*, not just the flag test.

#### What the guard actually removes (mcp1k, 1000 events)

Splitting ON rows by index is the load-bearing decomposition, because row 0
is the legacy result and rows ≥ 1 are what pr/94 adds:

| rows | n | < 15 cm | with a vertex | < 15 cm **and** with a vertex |
|---|---|---|---|---|
| primary (row 0 — the legacy result) | 461 | 61 | 446 | 49 |
| **added (row ≥ 1 — what pr/94 introduces)** | **40** | 34 | **5** | **1** |

The legacy chain selects sub-15 cm activities routinely — 49 of its own
reported vertices are under 15 cm — so a floor applied to *its* row would be
a behavior change, not a fix.

Among the 40 added rows, whether a vertex reconstructs tracks length almost
perfectly, which is why only 5 of 40 produce one:

| added rows | < 15 cm | ≥ 15 cm |
|---|---|---|
| with a vertex | 1 | 4 |
| no vertex | 33 | 2 |

| evt | selected L | `Enu` | bundle convicted? |
|---|---|---|---|
| 400636 | 112.9 cm | 577.9 MeV | no |
| 65053 | 46.9 cm | 243.1 MeV | no |
| 286681 | 35.3 cm | 270.0 MeV | no |
| 487303 | 21.7 cm | 383.7 MeV | no |
| **391854** | **1.7 cm** | **108.9 MeV** | no |

The 35 no-vertex added rows (median 1.7 cm) select a sub-cm main and
reconstruct nothing (`Enu` 0), making no physics claim — but they are still
`T_tagger` rows.

**All five vertex-producing added rows sit in UNCONVICTED bundles**, so the
floor never applied to any of them: as scoped by bundle conviction,
`nu_per_bundle_min_length` did nothing for the added rows that matter, and
391854's 1.7 cm → 108.9 MeV dot — precisely the muon-at-rest signature the
owner flagged — survived. §9.7 rescopes the floor to close that.

#### Why an activity can exist with no vertex (owner question)

*"I thought that we will at least identify a vertex if we have some
activities."* The vertex is not a property of the cluster; it is derived from
a fitted **trajectory**, and `find_proto_vertex` refuses to start without one
(`NeutrinoPatternBase.cxx:2791-2794`):

```cpp
if (!cluster.has_pc("steiner_pc")) return false;
if (steiner_pc.size() < 2) return false;
```

A ~1.6 cm blob carries fewer than two Steiner points, so no segment is built,
`determine_main_vertex` returns nullptr, and the whole refinement block
(`if (final_main_vertex)`, `:1968`) is skipped — leaving `nu_x/y/z` at their
initialised zeros. Both candidates of mcp1k evt 62583 show it in one log:

| candidate | selected L | initial PR | Steiner graph | vertex |
|---|---|---|---|---|
| cluster 26 | 31.6 cm | **209.5 ms** | 186 vertices / 248 edges | (-82.1, 91.5, 327.4) |
| cluster 12 | 1.6 cm | **0.016 ms** | none built | (0, 0, 0) |

The row is still emitted because the per-bundle loop emits one row per bundle
that *selected* a candidate, and selection is on **length alone, before any
fitting**. "Has an activity" and "can fit a vertex to it" are different bars.

### 9.7 Phase 5b round 2 — the floor is scoped by row role, not bundle

Owner instruction: *"Prevent sub 15 cm sounds good to me."*

The obvious reading — drop added rows under 15 cm — is **wrong**, and mcp1k
says so in one table:

| evt | row | selected L | is it the legacy row? |
|---|---|---|---|
| 62583 | 1 | 1.6 cm | **yes** — 0 differing branches vs the OFF row |
| 391854 | 1 | 1.7 cm | no — genuinely added |

Same length, opposite required verdicts. No length rule, and no bundle-
conviction rule, separates them; only *"is this the legacy selection?"* does.

**Final rule.** The floor applies to every candidate **except the legacy
event-wide winner**, in any bundle. `legacy_main` is recomputed inside the
per-bundle branch as a side-effect-free duplicate of the legacy selector
(`:999-1135`; M10 — the production branch stays textually untouched). This
subsumes the round-1 convicted-bundle scoping, so it is one rule, not two,
and additivity becomes structural: **the row the legacy chain reports can
never be floored away.**

#### Additivity, measured rather than assumed

On the 3 mcp1k events whose primary row changed (62583, 174422, 280466) the
legacy row is preserved **byte-for-byte as row 1** — 0 differing branches on
every shared tagger/kine branch. pr/94 is strictly additive at the row-set
level; only the *primary designation* moves, to the better-reconstructed
candidate. `primary_index` returns 0 on all 1000 events.

#### Round-2 effect on the small arms

Every second row removed was a no-vertex dot; the one genuine second neutrino
survived:

| sample | removed second rows (all `Enu` 0) | kept |
|---|---|---|
| nueCC48 | 1.7, 1.4, 3.3, 1.2 cm | — |
| NCpi0 | 1.7, 7.9, 1.6 cm | **evt 18625, 15.3 cm → 352.2 MeV** |

**Caveat worth carrying forward:** evt 18625 — the only hand-validated
multi-bundle event (§9.5, owner-scanned) — clears the 15 cm floor by 0.3 cm.
A 20 cm floor would delete it. That argues against raising the floor.

395148 is unaffected: 1 row, cluster 21, `Enu` 992.1 MeV, `numu_score` 3.655,
with cluster 10's STM verdict recorded in its `act_*` slot.

#### Round-2 on mcp1k (1000 events)

| | round 1 | **round 2** | OFF |
|---|---|---|---|
| total rows | 501 | **470** | 461 |
| rows pr/94 adds | 40 | **9** | — |
| events gaining a first vertex | 4 | **3** | — |
| events LOSING a vertex | 0 | **0** | — |
| reconstructed vertices | 442 | **450** | 443 |

All nine added rows, and what each one is:

| evt | selected L | vertex | what it is |
|---|---|---|---|
| 400636 | 112.9 cm | 577.9 MeV | genuine second neutrino |
| 65053 | 46.9 cm | 243.1 MeV | genuine |
| 286681 | 35.3 cm | 270.0 MeV | genuine |
| 487303 | 21.7 cm | 383.7 MeV | genuine |
| 317939 | 16.8 cm | — | above the floor, no vertex |
| 409634 | 15.6 cm | — | above the floor, no vertex |
| 174422 | 10.3 cm | — | **preserved legacy row** (exempt) |
| 62583 | 1.6 cm | — | **preserved legacy row** (exempt) |
| 280466 | 1.3 cm | — | **preserved legacy row** (exempt) |

Sub-15 cm added rows carrying a vertex: **0** (round 1: 1). The only sub-15 cm
survivors are the three preserved legacy rows — the exemption demonstrating
itself on data rather than in principle. Dot promotion across the round:
**143 → 1 → 0**.

Dot promotion: **143 → 1**. Sub-15 cm selections inside a convicted bundle:
**0** (the floor is doing exactly its job where scoped).

`15` rather than `5`: on the small arms the guard removed activities at 1.2,
1.5, 1.8 and **14.7** cm; the 14.7 cm one is nueCC48 evt 389538's second
bundle (anode 1), which selected a 14.7 cm main and reconstructed `nu_x = 0`,
`Enu = 0`. A 5 cm floor keeps that zero-vertex row.

#### Correction: `matched_flash_gid = 1000000` is not a sentinel

Rows at gid 1000000 were briefly read here as a "no matched flash" pseudo-
bundle. They are not: `gid = anode_ident * kFlashGidStride + flash_row` with
`kFlashGidStride = 1000000` (`QLMatching.cxx:37,3693`), so gid 1000000 is
**anode 1, flash row 0** — a genuine second in-beam bundle, consistent with
§1's "max 2 bundles per event" census.

#### Phase 5b gates

| gate | arms | result |
|---|---|---|
| OFF hash (mabc + pctree) | `pr94i-off-*` vs `pr94j-off-*` | PASS 96/96, 38/38 |
| OFF ROOT, all branches | ″ | PASS 48/48, 19/19 |
| OFF unaffected by the guard edit | `pr94j-off-*` vs `pr94k-off-*` | PASS 96/96, 38/38, 48/48, 19/19 |
| primary-row gate (0 lost, 0 differing) | `pr94k-off-*` vs `pr94k-on-*` | PASS nueCC48 48/48, NCpi0 19/19 |
| sync check (A-E) | `pr94k-on-*` | PASS 52/52 rows, 23/23 rows |
| additivity on mcp1k | `pr94p5-off-mcp1k` vs `pr94k-on-mcp1k` | **0 events lost a vertex** |
| 395148 (the motivating event) | `pr94k-on-evt395148` | 1 row, cluster 21 selected, `act` = {10: STM, 198.9 cm 21: FC selected} |
| `wcdoctest-clus` | — | 2215/2215 |

### 9.8 Phase 5b round 2 — a second, DIFFERENT defect found during Bee scan (OPEN, not fixed)

Owner scan of the round-2 Bee package (`bee/pr94r5b2/`) flagged mcp2k evt
73038 (idx 3, NEW): *"it seems that your current code also recover the
activities that are not matched to the beam flash time... we only need to
consider the bundles with beam flash."*

**Not a length-floor issue** — cluster 24 (the promoted candidate, 26.5 cm)
clears the 15 cm floor regardless of scoping, so §9.7's round-2 fix does not
touch this. This is a separate, deeper defect in shared production
clustering code, confirmed by direct trace, NOT yet fixed.

**Wrong initial diagnosis, corrected.** First pass compared the Bee-display
remap script's (`make_pr_bee.py`) charge-fingerprint heuristic against
`op.json`'s `op_cluster_ids` and concluded cluster 24 was built from charge
that "never matched any flash." **That conclusion was wrong** — retracted
after a direct trace of the authoritative C++ scalars (below). The
display-heuristic disagreement is a real but separate, lower-priority
artifact (see "Loose end" below); it is not the mechanism.

**Confirmed root cause, by direct trace.** Added an env-gated diagnostic
(`WCT_FLASHT0_DEBUG=1`, byte-identical when unset — verified: nueCC48 96/96
hash + 48/48 ROOT PASS against the pre-instrumentation arm) in two places:

- `clustering_examine_bundles.cxx`, right before the flash-t0 merge
  (`ClusteringExamineBundles`, `assign_flash_t0_groups`/`merge_clusters`)
  — dumps every in-scope cluster's ident, `flash`/`matched_flash_gid`,
  `cluster_t0`, length, `main_cluster` flag, and flash-time group.
- `ClusteringUnmergeBundle.cxx`'s `mark_demoted_main` — dumps the
  `real_cluster_id` provenance of each split-off demoted main.

Re-ran the Q/L stage for evt 73038 in an isolated scratch workspace
(`/home/xqian/tmp/ql73038trace*`, reusing the existing imaging npz + opflash
tarballs read-only — `work-cbr3-census-on` itself was never touched,
`run_ql_evt.sh` does `rm -rf $QLDIR` unconditionally so this must never be
pointed at a production ql_root, M13). Result (Q/L-stage cluster idents, at
the exact merge point):

```
cluster 2  flash=14 matched_flash_gid=14      cluster_t0=0.967555us L=1.55cm   group=6
cluster 5  flash=14 matched_flash_gid=14      cluster_t0=0.967555us L=4.92cm   group=6
cluster 6  flash=14 matched_flash_gid=14      cluster_t0=0.967555us L=26.53cm  group=6  <- becomes the promoted candidate
cluster 24 flash=0  matched_flash_gid=1000000 cluster_t0=0.934404us L=208.45cm group=6  <- the cosmic muon
```

Cluster 6 (11 blobs, 26.53 cm — matches the final candidate's 26.5 cm) **is
genuinely, independently flash-matched** — to gid **14**, a weak 602-PE flash
on APA0, the *exact same flash the legacy chain's own single-event candidate
(cluster 1, 15.4 cm, reconstructs no vertex) also matched*. It is NOT
unmatched charge.

**The mechanism:** `ClusteringExamineBundles`'s flash-t0 merge
(`clustering_examine_bundles.cxx:130-253`) groups clusters whose
`cluster_t0` differ by less than `flash_t0_window` (default **80 ns**,
`TaggerCheckNeutrino.h`/`ClusteringFuncs.h` `flash_t0_window_{80*units::ns}`)
— **with no spatial/geometric check at all** (the code's own comment says so:
"this stage's edges are gated ONLY on shared flash time"). Cluster 6's time
(0.967555 us) and the muon's time (0.934404 us) differ by **33 ns** — under
the window — so the merge fuses them. `merge_clusters()`'s "longest
flash-bearing member" rule then makes the 616-blob muon the group's donor,
and cluster 6 **inherits the muon's gid=1000000** instead of keeping its own
gid=14 (`ClusteringUnmergeBundle.cxx`'s own comment: "separate()->from()
copied the merged cluster's flags and cluster_scalar... matched_flash_gid").
Found the identical pattern on a second bundle in the same event (a 1.45 cm
piece merged with a 352 cm cluster, times 30.8 ns apart) — not a one-off.

**Physical reading:** cluster 6/24 is very plausibly the *same small, real
activity* the legacy chain's own candidate (cluster 1, gid=14) also sees —
just a larger piece of it — mislabeled onto the bright cosmic muon's flash
purely because the merge ignores geometry. The bookkeeping is wrong; the
underlying charge may well be real.

**Why this is out of scope for tonight.** The flash-t0 merge is shared
production clustering code (used far beyond per-bundle mode — it runs
unconditionally whenever `use_flash_t0` is on, i.e. in the all-APA stage for
every SBND event). A correct fix needs a design decision (tighten the
window? add a spatial/geometric check? stop propagating gid onto demoted
members and instead report the piece's own true gid?) that affects
production reconstruction broadly, not just this knob. Owner instruction:
*"Let's commit the code and summarize the finding first, we will use a new
session to track down this bug, you can keep the flag off for now."*

**Loose end, not yet reconciled:** the Bee-display remap script
(`make_pr_bee.py`) independently flagged cluster 24 as
"NO-FLASH-MATCH (fallback: dominant)" against `op.json`'s `op_cluster_ids`.
Given the C++ trace above shows a genuine match (gid=14), `op.json`'s
per-flash `op_cluster_ids` list likely does not enumerate every genuine
match (a display-layer limitation), or there is a numbering/id gap between
the two data products this session did not close. Re-check when picking this
up: `op.json`'s flash i=16 (apa0, PE=602.6, t=0.967555us — exactly gid=14)
lists `cluster_ids=[4]`, one off from the img-global cluster (5) the Bee
remap found at the fitted vertex position — worth resolving alongside the
merge-window fix, not before it.

**State left for the next session:**
- `WCT_FLASHT0_DEBUG=1` diagnostic in `clustering_examine_bundles.cxx` and
  `ClusteringUnmergeBundle.cxx` — env-gated, verified byte-identical when
  unset, left in place (removable, but useful for the next trace).
- `nu_per_bundle` stays `false` everywhere in SBND production config
  (`wct-pr-perevt.jsonnet:1098`, `clus.jsonnet:1258,3039`) — **not flipped**.
- Scratch trace workspace: `/home/xqian/tmp/ql73038trace*` (not committed;
  regenerate from `work-cbr3-census-on/{evt,ql_evt}73038` if needed again).
- Round-2 length floor (§9.7) is independently complete, gated, and safe to
  ship on its own merits whenever the flash-attribution bug is resolved —
  the two are unrelated defects.

### 9.9 Round 3, bug 1 -- "this piece is shown as matched to no flash" (evt 73038)

**Symptom.** Owner Bee scan of the round-2 set (`d47318ad`, idx 3 = mcp2k evt
73038): the promoted candidate at `(-1.2, -34.7, 121.3)`, `real_cluster_id
24000`, is drawn as matched to **no flash** in `img-global` / the `op` layer,
yet the PR chain reconstructs it as that bundle's neutrino. *"I do not
understand what happened for this, or at what point this cluster was matched to
the beam flash."*

**Root cause -- a display filter, proven, not inferred.** The activity is
`img-global` cluster 5: 129 points, a cathode sliver at
`x[-1.5,-0.9] y[-42.4,-28.8] z[110.4,132.9]` cm, L 26.5 cm. It **is** matched,
to **APA0 flash gid 14** (t 0.9676 us, 602.6 PE) -- in the beam window -- but
its own predicted light is **3.6 PE**, and `fill_bee_flashes` dropped every
matched cluster under 100 PE (`MultiAlgBlobClustering.cxx:2896`, the legacy
`dump_light` value):

```
op-dump debug: cluster  5 matched_flash_gid=14      pred_tot=    3.644 PE L= 26.53 cm nblobs= 11 kept_by_display=false  <- the sliver
op-dump debug: cluster  2 matched_flash_gid=14      pred_tot=   18.465 PE L=  1.55 cm nblobs=  4 kept_by_display=false
op-dump debug: cluster  4 matched_flash_gid=14      pred_tot=  204.592 PE L= 15.94 cm nblobs= 22 kept_by_display=true
op-dump debug: cluster 25 matched_flash_gid=1000000 pred_tot=11093.789 PE L=208.45 cm nblobs=616 kept_by_display=true   <- the STM muon
```

(New env-gated probe `WCT_OPDUMP_DEBUG=1`, placed **inside** the op-dump loop so
the ids printed are the display's own epoch. That mattered: the QLMatching log
prints a per-run `ident` which is a different epoch -- apa1 log `ident 9` is
display cluster 25 -- so the artifacts on disk could not settle it and a probe
run was required. `op_pes_pred` row 16 sums to 204.59192, matching the logged
`total_pred_light` exactly, which is how the surviving row was identified.)

**Why it hid.** The filter is silent: `op.json` simply lists fewer clusters, and
both the Bee viewer's flash navigation and `make_pr_bee.py`'s remap read that
one array -- which is also why the remap reported "NO-FLASH-MATCH (fallback:
dominant)" for this cluster. **This closes the "loose end" §9.8 left open**
(`op` said 4, the remap found 5): 4 is the only member of that flash above
100 PE, 5 is the sliver, and both are genuine.

**Fix (owner decision: dumper only).** New knob `bee_flash_pred_min` on
`MultiAlgBlobClustering`, **C++ default 100** = the legacy filter =
byte-identical display. Set to 0 the `op` layer lists every genuine match.
Verified on evt 73038: flash row 16 `op_cluster_ids` **`[4]` -> `[2, 4, 5]`**,
and a member-content hash of `mabc-all-apa.zip` shows **`0-op.json` as the only
differing member**. Across the 23 review events the fix reveals **156** matches
the display was hiding, and the op rows, times, PE and `apa` are unchanged --
each row's cluster-id list is a strict **superset** of production's.

**What the round-3 PR knobs do to this row, measured.** The candidate's energy
drops from round 2's 204.3 MeV to 51.3 MeV, and it is **`nu_selected_as_main`**
that does it (both knobs fire on this event, so the attribution needed its own
arms):

| arm | selected | Enu | numu | vertex (cm) |
|---|---|---|---|---|
| round 2 (both round-3 knobs off) | c24 | 204.3 | -1.60 | (-1.8, -29.9, 112.8) |
| `nu_selected_as_main` only | c24 | **51.3** | -1.77 | (-1.7, -29.9, 113.2) |
| `open_convicted_bundles` only | c24 | 204.3 | -1.60 | (-1.8, -29.9, 112.8) |
| both (the ON arm) | c24 | **51.3** | -1.77 | (-1.7, -29.9, 113.2) |

**Say the quiet part out loud:** 51.3 MeV on a 26.5 cm activity is below the
muon mass, which is the same signature §9.6 records the owner rejecting a whole
round over. Its `numu_score` is -1.77, so the row fails any numu selection --
but the display fix does **not** make this row defensible, it only makes its
flash provenance visible. The piece that would stop the promotion is the
bookkeeping fix declined below.

**Explicitly NOT fixed this round (owner decision).** The second, separate
half of §9.8 stands: the sliver is *evaluated* inside the **muon's** bundle
because `ClusteringExamineBundles`' flash-t0 merge (80 ns window, no spatial
check, single-linkage chaining -- `ClusteringFuncs.cxx:617-624`) fused it with
the APA1 205 cm STM muon 33 ns away, `merge_clusters` kept only the longest
flash-bearing member's `matched_flash_gid` (`:368-378`, `:563-567`), and
`ClusteringUnmergeBundle` split it back out with the muon's gid inherited via
`separate()->from()`. No per-blob gid provenance is saved, so nothing
downstream can recover the true bundle. The owner chose the dumper fix alone
for this round; the bookkeeping fix (save/restore the pre-merge gid) and a
predicted-light floor on per-bundle candidates were both offered and declined.
`nu_per_bundle` therefore stays OFF in production.

### 9.10 Round 3, bug 2 -- the secondary activity does not get the main
cluster's treatment (evt 395148)

**Symptom.** Owner scan of `e1357e60` idx 5 (evt 395148): the secondary
(demoted-main) neutrino's `track_fit` point at `(-172.7, -131.5, 205.7)`,
`real_cluster_id 21016`, has ~zero dQ/dx and no charge under it. *"I feel this
is coming from a jumping of gaps... Apparently, for the secondary neutrino
activity, the Graph creation is different from the main activity."*

Measured on the round-2 arm: of 586 fitted points, **15 lie more than 3 cm from
any imaged charge in the event**, 8 beyond 5 cm, worst **8.40 cm**; 12 of them
are in segment `21016`. The excursion is **interior** to the segment -- both its
ends sit on charge (0.22 and 0.30 cm) -- so it is a chord across a void, not an
over-extended terminus.

**Root cause -- two independent mechanisms, both real.**

1. **The bundle is never opened for the second graph examination.**
   `ClusteringProtectBundle` (the uboone `Protect_Over_Clustering` port, whose
   whole job is splitting a cluster at graph-component boundaries) populates
   `beam_gids` from in-window mains, and `skip_convicted` makes a cosmic-tagged
   main *not open its bundle at all* (`:205-217`):
   `OC53SKIP main ident=10 nblobs=929 gid=0 t0=1.53us convicted TGM=0 STM=1 lm=0 -- bundle not opened`,
   then `split 0 bundle cluster(s) into 0 extra cluster(s)`. So when the main is
   a cosmic, **nothing in the bundle** is examined -- including the secondary
   the demoted-main fallback then selects as the neutrino. The code already
   separates "open the bundle" from "split this cluster": a per-member guard
   (`:234-248`) keeps a convicted cluster from ever being split even when an
   unconvicted mate opens the bundle.
2. **The selected candidate is not treated as a main.**
   `NeutrinoPatternBase.cxx:2797` re-derives main-ness from
   `Flags::main_cluster`, which `ClusteringUnmergeBundle` deliberately clears on
   a demoted main (`:404`), even though `TaggerCheckNeutrino.cxx:1854` passes
   that very cluster as the selected main. The candidate therefore silently
   loses the main-branch endpoint ordering honouring `flag_back_search`
   (`:1438-1451` vs the ascending-z `else` at `:1470-1478`),
   `main_cluster_initial_pair_vertices` (`:2805-2809`), `examine_vertices_3`
   (`:2962-2964`), `break_two_end_dqdx` (`:3001`), and -- read from the flag
   **directly, in two other files** -- `improve_vertex` +
   `fix_maps_shower_in_track_out` (`NeutrinoVertexFinder.cxx:3450`) and the
   main-cluster track/shower reclassification cut set
   (`NeutrinoTrackShowerSep.cxx:2013`). A cross-check written to catch exactly
   this (`NeutrinoPatternBase.cxx:1433-1438`) is dead code, because the pointer
   it compares is constructed from the same flag.

Graph *creation* is symmetric -- no `main_cluster` guard exists in any
`connect_graph*` / `find_graph` / Steiner entry point, and `steiner_gap_penalty`
is ON for SBND and applies to both. The defect is the main-cluster **treatment**,
not the builder.

**Fix -- two knobs, both C++ default false.**

| knob | component | effect when on |
|---|---|---|
| `open_convicted_bundles` | `ClusteringProtectBundle` | a convicted main still inserts its gid into `beam_gids` (log `OC94OPEN`), so the bundle's **unconvicted** members are examined and split; the convicted cluster itself is still never split |
| `nu_selected_as_main` | `TaggerCheckNeutrino` | the selected candidate carries `Flags::main_cluster` for the duration of **its own PR pass** and only that, via an exception-safe RAII guard around the per-candidate loop body -- no later visitor, bundle-veto set or output dump sees a changed flag |

`nu_selected_as_main` is deliberately **not** gated on `nu_per_bundle`: the
legacy `nu_fallback_demoted_mains` path (which is how 395148 selects cluster 21
in production today) has the identical defect.

**Verification -- which knob actually fixes the reported trail.** Distance from
every fitted point to the nearest imaged charge, evt 395148:

| arm | n | > 3 cm | > 5 cm | max | mean | worst segments |
|---|---|---|---|---|---|---|
| OFF (production) | 586 | **15** | 8 | **8.40 cm** | 0.54 | 21016 (12 pts, 8.4 cm) |
| `nu_selected_as_main` only | 594 | 19 | 11 | 8.58 cm | 0.58 | 21006 (14 pts, 8.6 cm) |
| `open_convicted_bundles` only | 503 | **0** | 0 | **0.90 cm** | 0.35 | -- |
| both (the ON arm) | 550 | **0** | 0 | **0.83 cm** | 0.34 | -- |

`open_convicted_bundles` is the fix for the owner's symptom, and the mechanism
is visible in the log:
`OC94OPEN main ident=10 ... -- bundle OPENED for its unconvicted members`,
`OC53SKIP member ident=10 ... -- never split`,
`split 1 bundle cluster(s) into 8 extra cluster(s)`. `nu_selected_as_main`
alone does **not** remove this particular trail -- expected, since
`examine_vertices_3` only touches degree-1 vertices
(`NeutrinoStructureExaminer.cxx:2780`) and this excursion is interior -- but it
is a genuine parity defect and it does change the fit (Enu 992.1 -> 877.4 alone,
1000.8 with both). Reported as measured rather than as predicted.

Remaining lever, **not** taken this round: `organize_segments_path`
(`TrackFitting.cxx:1781-1808`) fills any inter-blob jump longer than
`1.6 x low_dis_limit` with synthetic points at 0.6 cm spacing with **no** charge
or dead-region test. That is the mechanism that mints such points for main and
secondary alike; a support test there would be a third knob and a much wider
blast radius.

### 9.11 Round 3 -- gates, arms and the review package

**Knob-OFF byte-identical gate (new binary vs the pre-round-3 arms):**

| gate | arms | result |
|---|---|---|
| member-content hash (`pr85_hash_gate.py`) | `work-pr94n-off-nuecc48` vs `work-pr94r3-off-nuecc48` | **PASS 96/96** |
| per-branch/per-entry ROOT (`pr94_root_gate.py`) | ″ | **PASS 48/48** |
| member-content hash | `work-pr94m-off-ncpi0` vs `work-pr94r3-off-ncpi0` | **PASS 38/38** |
| per-branch/per-entry ROOT | ″ | **PASS 19/19** |
| compiled config, PR job | knobs off vs a HEAD-`cfg` compile | **diff EMPTY**; on shows `nu_selected_as_main` + `open_convicted_bundles` |
| compiled config, Q/L job | knob off vs HEAD-`cfg` | **diff EMPTY**; on shows `bee_flash_pred_min` |
| `wcdoctest-clus` | -- | 211 cases / **2214 assertions pass**, 0 failed, 1 skipped |

(Earlier sections of this doc quote 2215/2215. The one-assertion delta predates
round 3 -- it is not a regression from these knobs; 2214 is what the suite
reports at this commit, with 0 failures.)

`pr_display` is not separately gated this round: unlike Phase 4b, nothing in
`PrDisplayDump` was touched, and the two knobs that can change PR output are off
by construction on the gated arms.

**Knob-ON footprint on nueCC48 (48 events)** -- ON = `nu_per_bundle` +
`nu_per_bundle_min_length=15` + both round-3 knobs:

* tagger output produced on **48/48**, lost on **0**;
* a reconstructed vertex on **48/48** in both arms (0 lost, 0 gained);
* the primary row is identical on **47/48**; the single mover is evt **116962**
  (`anc_acc_forward_length`, `anc_acc_backward_length`, `anc_angle`), whose
  vertex moves 34 cm in z and whose `numu_score` goes 0.59 -> -0.50. It is
  index 22 of the review package and needs an owner verdict:

| arm | vertex (cm) | Enu | numu |
|---|---|---|---|
| OFF | (-72.3, -78.2, 121.7) | 837.3 | 0.59 |
| `nu_selected_as_main` only | (-73.0, -79.6, 118.4) | 684.4 | -0.07 |
| `open_convicted_bundles` only | (-69.7, -78.5, 159.7) | 1268.1 | 0.63 |
| both | (-69.4, -79.6, 156.2) | 877.8 | -0.50 |

**How much of nueCC48 actually exercised the knobs -- read the mover count with
this, not without it.** `open_convicted_bundles` can only bite in a bundle whose
main is cosmic-convicted, and `nu_selected_as_main` only when the selected
candidate is a demoted main. On nueCC48 that is a thin slice:

| | nueCC48 (48) | review set (22) |
|---|---|---|
| events where `OC94OPEN` fired (bundle opened) | **5** | 9 |
| events where `nu_selected_as_main` engaged | **1** | 9 |
| events with >= 1 protect_bundle split, OFF -> ON | 46 -> 47 | -- |
| extra clusters made by protect_bundle, OFF -> ON | 466 -> 487 | -- |

So "1 mover of 48" is a statement about a sample that gave the knobs 5 and 1
chances respectively, not a 48-event stability proof. The events where the
knobs really bite are the review set, which is what the owner scan covers.

Per owner instruction ("no need to go to full blown 1000 + 2000 numu events"),
the mcp1k/mcp2k population arms were **not** run this round.

**Post-review fix, log-only.** The first cut of `open_convicted_bundles`
incremented `n_convicted` before the early-continue, so an opened bundle's main
was counted twice (395148 logged `2 convicted main(s) skipped` for its single
convicted main). Moved inside the non-open branch; a census script parsing that
line would otherwise have read the wrong number. `n_convicted` feeds one debug
line and nothing else, so the uploaded ON arms (built with the pre-fix binary)
are unaffected -- and the OFF gate was re-run after the change:
`work-pr94r3-off-nuecc48` vs `work-pr94r3b-off-nuecc48` **PASS 96/96 archives
and 48/48 events**.

**Review package -- `bee/pr94r3/`, 23 events, UPLOADED 2026-08-19:**

> OFF https://www.phy.bnl.gov/twister/bee/set/0ee25b32-949f-4262-ba9d-6f73f0865d91/event/list/
> ON&nbsp; https://www.phy.bnl.gov/twister/bee/set/7414c7c1-c705-47df-9aca-07eb573ca4b6/event/list/

idx 0-20 are the round-2 scan set re-run with the round-3 fixes, idx 21 is
395148 (bug 2), idx 22 is 116962 (the nueCC48 mover). Per-event OFF -> ON
numbers and a "where to look" guide are in `bee/pr94r3/pr94r3.index.txt`. On
idx 8-20 the primary row is byte-identical to OFF and only the added second
bundle is new; on idx 0-7 every number shifted relative to round 2 -- those are
the events where the two round-3 knobs bite.

**The ON display root, and one honest caveat.** The Bee `op` layer for the ON
set comes from a Q/L re-run with `bee_flash_pred_min=0`
(`work-pr94r3-ql22`), which reproduces the production Q/L zip **byte-for-byte
except `0-op.json`** on 22 of the 23 events -- verified by member-content hash,
including against the untouched production arm. The 23rd, evt **65053**, also
shows a `clustering-global` difference: a 1005-point piece joins a different
neighbour (prod 7176+1005 / 561; re-run 7176 / 1005+561). That is **not** caused
by this knob -- three independent re-runs reproduce it identically, `img-global`
is unchanged, and the op layer is emitted pre-pipeline -- it is drift between
the ~2 week old `work-mcp1k-cb0805` Q/L arm and today's binary. Because both PR
arms consumed the **production** pctree, the display root
(`work-pr94r3-ql22disp`) is built as the production zip with its `op.json`
member replaced by the knob-on one: byte-identical to the knob-on run on all 22
other events, and consistent with the fitted pctree on 65053.

### 9.12 Round 3 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
wcbuild                                                       # then, M1:
ls -la --time-style=full-iso build/clus/libWireCellClus.so
./build/clus/wcdoctest-clus                                   # 2214/2214

SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin; cd $SX
# knob-OFF gate arms (new binary)
PR_JOBS=24 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr94r3-off-nuecc48 data
PR_JOBS=24 ./run_pr_chain_batch.sh work-ncpi0-cb0805   work-pr94r3-off-ncpi0   data
python3 scripts/pr85_hash_gate.py work-pr94n-off-nuecc48 work-pr94r3-off-nuecc48 --jobs 8
python3 scripts/pr94_root_gate.py work-pr94n-off-nuecc48 work-pr94r3-off-nuecc48
python3 scripts/pr85_hash_gate.py work-pr94m-off-ncpi0   work-pr94r3-off-ncpi0   --jobs 8
python3 scripts/pr94_root_gate.py work-pr94m-off-ncpi0   work-pr94r3-off-ncpi0

# knob-ON arms
export SBND_NU_PER_BUNDLE=1 SBND_NU_SELECTED_AS_MAIN=1 SBND_OPEN_CONVICTED_BUNDLES=1
PR_JOBS=8  ./run_pr_chain_batch.sh work-mcp1k-cb0805   work-pr94r3-on-mcp1k data \
    62583 174422 280466 65053 286681 400636 487303 395148
PR_JOBS=14 ./run_pr_chain_batch.sh work-cbr3-census-on work-pr94r3-on-mcp2k data \
    73038 78032 90751 167612 407798 68748 174661 175094 179054 179369 287638 351564 409624 410698
PR_JOBS=12 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr94r3-on-nuecc48 data
unset SBND_NU_SELECTED_AS_MAIN SBND_OPEN_CONVICTED_BUNDLES   # per-knob attribution arms

# bug 1: the probe that settled it, and the fix (isolated scratch root -- NEVER
# point run_ql_evt.sh at a production ql_evt dir, it rm -rf's it, M13)
WCT_OPDUMP_DEBUG=1 SBND_WORK_ROOT=/home/xqian/tmp/ql73038probe1 \
  SBND_INPUT_DIR=/home/xqian/tmp/ql73038trace_input ./run_ql_evt.sh data 1
SBND_QL_BEE_FLASH_PRED_MIN=0 SBND_WORK_ROOT=/home/xqian/tmp/ql73038fix1 \
  SBND_INPUT_DIR=/home/xqian/tmp/ql73038trace_input ./run_ql_evt.sh data 1

# bug 2: the gap metric quoted above (scripts/pr94r3_gap_metric.py)
python3 scripts/pr94r3_gap_metric.py work-pr94p5-off-mcp1k/pr_evt395148/mabc-pr.zip \
    work-pr94r3-a1only-395148/pr_evt395148/mabc-pr.zip \
    work-pr94r3-a2only-395148/pr_evt395148/mabc-pr.zip \
    work-pr94r3-on-mcp1k/pr_evt395148/mabc-pr.zip

# review package
python3 scripts/bee/make_pr_bee.py --allow-unevaluated \
    -q work-mcp1k-cb0805 -q work-cbr3-census-on -q work-nuecc48-cb0805 \
    -p work-pr94p5-off-mcp1k -p work-pr94p5-off-mcp2k -p work-pr94r3-off-nuecc48 \
    -o bee/pr94r3/pr94r3-off.zip $EVENTS
python3 scripts/bee/make_pr_bee.py --allow-unevaluated -q work-pr94r3-ql22disp \
    -p work-pr94r3-on-mcp1k -p work-pr94r3-on-mcp2k -p work-pr94r3-on-nuecc48 \
    -o bee/pr94r3/pr94r3-on.zip $EVENTS
```

### 9.13 Phase 6 -- SBND PRODUCTION FLIP (owner, 2026-08-19)

Owner decision after the round-3 Bee scan (`bee/pr94r3`): **all four knobs ON
for SBND production**, i.e. Phase 6 as well as the three round-3 fixes. The
owner was shown, and accepted, that this ships the evt 73038 row (51.3 MeV on a
26.5 cm activity, `numu_score` -1.77) with the §9.9 bookkeeping half still
unfixed.

| knob | file | pre-flip | production |
|---|---|---|---|
| `nu_per_bundle` | `wct-pr-perevt.jsonnet:1119` | false | **true** |
| `nu_per_bundle_min_length` | `:1133` | 15 | 15 (unchanged) |
| `nu_selected_as_main` | `:1146` | false | **true** |
| `protect_open_convicted_bundles` | `:548` | null (C++ false) | **true** |
| `bee_flash_pred_min` | `wct-clus-matching-perevt.jsonnet:345` | null (C++ 100) | **0** |

Config-only -- no rebuild, no C++ change. The escape hatches are now tri-state
so the pre-flip arm stays reproducible: `SBND_NU_PER_BUNDLE=0`,
`SBND_NU_SELECTED_AS_MAIN=0`, `SBND_OPEN_CONVICTED_BUNDLES=0`,
`SBND_QL_BEE_FLASH_PRED_MIN=100`.

**What production now emits that it did not before.** One `T_tagger`/`T_kine`
row per in-beam-window flash bundle instead of one per event. **Every consumer
that reads one number per event must go through `scripts/pr94_rows.py`
`primary_index()`** -- the five that hard-indexed `[0]` were converted in
Phase 5 (`pr_scores_table.py`, `pr51/nuvtx_census.py`, `pr74/pr74_pf_roots.py`,
`pr20/pr20_partI_pftree.py`, `pr73/f3a_change_map.py`). Row 0 is the longest
selected activity, i.e. the candidate the pre-pr/94 chain would itself have
chosen, so a single-bundle event reads exactly as before.

#### Flip gates

| gate | arms | result |
|---|---|---|
| flip-equivalence, nueCC48 (bare == env-forced ON) | `work-pr94r3-on-nuecc48` vs `work-pr94r3-flip-nuecc48` | **PASS 96/96 archives, 48/48 events** |
| flip-equivalence, mcp1k review events | `work-pr94r3-on-mcp1k` vs `work-pr94r3-flip-mcp1k` | **PASS 16/16, 8/8** |
| flip-equivalence, mcp2k review events | `work-pr94r3-on-mcp2k` vs `work-pr94r3-flip-mcp2k` | **PASS 28/28, 14/14** |
| flip-equivalence, Q/L display (bare == forced 0) | evt 73038 | **PASS**, 0 differing zip members; bare production op row for the 0.968 us APA0 beam flash = `[2, 4, 5]` |
| pre-flip escape hatch reproduces pre-flip production | `work-pr94r3b-off-nuecc48` vs `work-pr94r3-preflip-nuecc48` | **PASS 96/96, 48/48** |
| compiled config, bare == env-forced ON | -- | **diff EMPTY**; bare shows `nu_per_bundle: true` (x2), `nu_per_bundle_min_length: 15`, `nu_selected_as_main: true`, `open_convicted_bundles: true` |
| compiled config, escape hatch vs pre-flip | -- | one line: an explicit `open_convicted_bundles: false`, which **is** the C++ default -- behaviourally identical, same tri-state pattern as `protect_skip_convicted` |

Total flip-equivalence coverage: **140 archives / 70 events byte-identical**
plus the Q/L display, so the bare production chain provably reproduces the arms
the owner scanned.

#### What changes in production output, measured

* **nueCC48 (48 events)**: tagger output on 48/48, a reconstructed vertex on
  48/48, **0 lost**; the primary row is identical on **47/48**; evt **116962**
  moves (vertex +34 cm in z, `Enu` 837.3 -> 877.8, `numu_score` 0.59 -> -0.50).
  Coverage caveat from §9.11 stands: on this sample the bundle-opening knob
  fired on only 5/48 events and `nu_selected_as_main` on 1/48.
* **The 22 review events**: 8 events gain a first neutrino where production
  reconstructed none; 13 keep a byte-identical primary row and gain a second
  bundle; 395148's 17 cm trajectory excursion through empty space is gone
  (fit points > 3 cm from any charge 15 -> 0).
* **Not measured**: mcp1k/mcp2k at population scale -- the owner capped this
  round's validation at "bee links + nueCC". The row-count and label impact on
  `nusel-events.tsv` (§7's 68/629 miscount) is therefore still carried by the
  `nusel-events-pr94.tsv` sidecar rather than fixed at the source.

#### Repro

```bash
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin; cd $SX
# bare production (post-flip) == the env-forced ON arms
PR_JOBS=24 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr94r3-flip-nuecc48 data
python3 scripts/pr85_hash_gate.py work-pr94r3-on-nuecc48 work-pr94r3-flip-nuecc48 --jobs 8
python3 scripts/pr94_root_gate.py work-pr94r3-on-nuecc48 work-pr94r3-flip-nuecc48
# the pre-flip arm is still reproducible
SBND_NU_PER_BUNDLE=0 SBND_NU_SELECTED_AS_MAIN=0 SBND_OPEN_CONVICTED_BUNDLES=0 \
  PR_JOBS=24 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr94r3-preflip-nuecc48 data
python3 scripts/pr85_hash_gate.py work-pr94r3b-off-nuecc48 work-pr94r3-preflip-nuecc48 --jobs 8
```

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

**As implemented** (`scripts/pr94_sync_check.py`) the check cross-joins **five**
producers per row, not three — the two extra ones each caught a real bug:

| | producer | what it pins |
|---|---|---|
| A | `T_tagger[i]` | `cluster_id`, `matched_flash_gid`, `nu_index`, `nu_x/y/z` |
| B | `T_kine[i]` | the same three identity fields + `kine_nu_*_corr` |
| C | the log's **publish-time** `[nu_per_bundle] ROW i gid G cluster C` sentinel | that the row was stashed for the bundle it says. Joining on the *selection* line instead was the check's own first bug: `swap_main_cluster` can repoint `main_cluster` between selection and publish |
| D | `mabc-pr.zip` `0-mc.json` root labelled `nu <i> (gid G, cluster C)` | the particle-flow tree — this is what caught the Phase 4 graph bug |
| E | `mabc-pr.zip` **point layers** `track_fit`/`shower_track`/`vertices` | that the candidate's trajectories were actually drawn — added in Phase 4b (§9.5); D passing while E failed is precisely the defect the owner's scan reported |

Rows that reconstructed **no** vertex are exempt from D and E: they have no
particle flow and no points, by design, and demanding either was an earlier
false failure. E joins on **position, not `cluster_id`** — see §9.5 for the
`enumerate_idents` epoch trap that makes an id join wrong.

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

**As shipped (2026-08-19).** toolkit `apply-pointcloud`:

| commit | phase |
|---|---|
| `68952e5f` | Phase 1 — schema + branch booking |
| `7d1bbde6` | Phase 2 — per-bundle candidates, scorers, tagger output |
| `3f01ea90` | Phase 4 — Bee particle flow per candidate |
| `d5f87a13` | candidate ordering (longest-selected first) |
| *this round* | Phase 4b — Bee point layers, Magnify `T_rec_charge`/`T_proj_data`, `PrDisplayDump` + viewer |

wcp-porting-img `main`: `986c819` (harness), `91d6555` (doc), `9cf5c8e` +
`7b385b4` (Bee review links), *this round* (Phase 4b doc, sync check E, viewer).

`nusel_extract.py` is **not** patched — the corrected `event_label` lives in
`scripts/pr94_mains_sidecar.py`'s `nusel-events-pr94.tsv` instead, so
`nusel-table.tsv`'s row cardinality and its ~20 parsers are untouched. Patching
it in place is open decision (ii) in the status header.
