# pr/16 — Bundle-veto refinement study: can PR run on "the rest of the bundle" when the main is cosmic-tagged?

Status: §§1–9 STUDY (code read at `7f29d32d`); **§10 design A IMPLEMENTED**
(`nu_skip_cosmic_bundle_min_length`, owner-chosen 15 cm, SBND ON — NOT
bit-identical: it restores PR on evt 10550).  Owner question (2026-08-01,
follow-up to doc pr/3 §8.5): the bundle veto (`nu_skip_cosmic_bundle`,
45dae9d0) discards the whole flash bundle when any main is cosmic-tagged.  The proposed
refinement is to instead *filter out the cosmic-tagged cluster and use the
rest of the bundle for neutrino reconstruction* — possibly promoting an
associated cluster to be the new "main".  Before implementing, the owner asked
how such a promoted/surviving candidate would interact with neutrino vertex
identification, pattern recognition and energy reconstruction.

Everything below is a code trace at toolkit commit `7f29d32d`
(branch `apply-pointcloud`) plus score rows read from existing A/B arms.
No new runs were made.

## Repro block

```bash
# empirical rows quoted in sec. 2 (arms from doc pr/3 sec. 8.7, data, MCP2025C):
python3 -c "
import uproot
for p in ['work-nscbase-ab31/pr_evt52195',
          'work-nscoff-nuecc48/pr_evt271851',
          'work-nscoff-nuecc48/pr_evt10550']:
    f=uproot.open(f'{p}/tracking-pr.root')
    print(p, f['T_tagger'].arrays(['nue_score','numu_score','numu_cc_flag'],library='np'),
             f['T_kine'].arrays(['kine_reco_Enu','kine_nu_x_corr','kine_nu_y_corr','kine_nu_z_corr'],library='np'))"
# all file:line citations: git -C toolkit show 7f29d32d, files under clus/, match/, root/
```

## 1. The three designs on the table

- **A — guarded veto relaxation** (doc pr/3 §8.5's "size/assoc-count
  threshold"): keep the bundle veto, but let a *surviving un-tagged main*
  above a size threshold still get PR.  The cosmic-tagged main stays out.
- **B — promotion**: when the bundle's only main is cosmic-tagged, promote an
  associated cluster to "main" and run PR on it + the remaining members.
- **C — evaluate the associated clusters**: extend TGM/STM to non-main bundle
  members so a cosmic verdict *exists* on them, then filter the PR ensemble
  by verdict (the doc pr/3 §8.1 "fix the asymmetry" refinement).  C is what
  makes "use the rest" literally safe; A and B are selection policies.

## 2. Empirical anchor: the chain has already run in both regimes

The pre-veto (per-main) rule *was* design A with threshold zero: it skipped
cosmic-tagged mains and ran PR on the surviving main + associated companions.
Three events from the existing arms show both outcomes:

| event | effective main | big bundle-mate | vertex (x,y,z corr, cm) | kine_reco_Enu | verdict |
|---|---|---|---|---|---|
| 18255/52195 (`work-nscbase-ab31`) | shard 5: 1.3 cm, 5 pts | 400 cm **untagged cosmic** (associated 23) | (−51.6, **+199.4**, 344.5) — top wall, on the cosmic | 1062 MeV (cosmic charge), numu_cc_flag=1 | pathological |
| 18255/271851 (`work-nscoff-nuecc48`) | main 7: 13.0 cm, 26 associated | 253.9 cm TGM main 24 (excluded) | (−161.8, 19.3, 342.2) — interior | 153 MeV | sane |
| 18255/10550 (`work-nscoff-nuecc48`) | main 7: 18.5 cm, 28 associated | 380 cm TGM main 11 (excluded) | (−14.8, 59.1, 147.1) — interior | 388 MeV | sane |

The difference is not the machinery — it is whether the surviving ensemble
still contains an **un-vetoed cosmic**.  In 271851/10550 the cosmic was a
tagged *main* and was excluded; in 52195 the cosmic was an *associated*
cluster, carried no verdict, and PR faithfully reconstructed it as the
neutrino.

## 3. How "main" actually flows through the PR chain

### 3.1 The cosmic-tagged main is already filtered out

`TaggerCheckNeutrino` selects among `Flags::main_cluster` clusters only
(`TaggerCheckNeutrino.cxx:324`), skipping cosmic-tagged ones (`:338`), and
builds the companion set `other_clusters` from clusters that carry
`Flags::associated_cluster` **and** share the winner's `matched_flash_gid`
(`:368-369`).  A TGM/STM-tagged main is therefore *never* in the PR ensemble
when a runner-up main is selected — design A needs **no new filtering
mechanism**.  What no filter can currently remove is a cosmic that is not a
main: TGM/STM/FC iterate main-flagged clusters only (`TaggerCheckTGM.cxx:243`,
`TaggerCheckSTM.cxx:406`, `TaggerCheckFC.cxx:144`), so associated clusters
carry `flag_TGM = flag_STM = flag_FC = 0` *unevaluated*.

### 3.2 Below selection, "main" is 95 % positional…

Everything from `find_proto_vertex` (`TaggerCheckNeutrino.cxx:491`) through
the taggers (`:676-747`), `match_isFC` (`:757`) and `fill_kine_tree` (`:766`)
consumes the `Facade::Cluster*` the selection loop produced.
`NeutrinoKinematics.cxx` and `TrackFitting.cxx` contain **zero** reads of the
flag; the tagger/kine ROOT writer (`UbooneTaggerOutputVisitor.cxx`) never
reads it either.  Passing a different pointer changes the whole chain's
behavior coherently.

### 3.3 …but three sites re-derive it from the *flag*

1. `NeutrinoPatternBase.cxx:1646` — `find_proto_vertex` re-derives
   `main_cluster_ptr` from `get_flag(main_cluster)`; feeds
   `init_first_segment`'s start-point choice.  `init_first_segment:368-374`
   detects flag/pointer disagreement, WARNs, and lets the pointer win.
2. `NeutrinoPatternBase.cxx:1655` (+ `:1745,:1764`) — main-only
   `examine_structure_3` / `examine_vertices_3` refinements.
3. `NeutrinoVertexFinder.cxx:2507` — `improve_vertex` +
   `fix_maps_shower_in_track_out` inside `determine_main_vertex` run **only
   for flag-carrying clusters**.

So a pointer-only promotion silently loses the main-grade refinements; a
correct promotion must move the flag too (and demote the old main).

### 3.4 In-PR promotion already exists — with two live defects

`swap_main_cluster` (`NeutrinoPatternBase.cxx:2084-2105`) is the in-tree
precedent: it un-flags the old main, flags the new one, and fixes up
`other_clusters`.  It is called from `check_switch_main_cluster[_2]` and
`determine_overall_main_vertex` (`NeutrinoVertexFinder.cxx:3232,3275,3650`) —
i.e. **PR can already move the effective main onto a companion mid-flight**.
Two defects to know about:

- **By-value desync (traditional path)**: `determine_overall_main_vertex`
  takes `map_cluster_main_vertices` and `main_cluster` *by value*
  (`NeutrinoPatternBase.h:394`; the DL variant correctly takes
  `Cluster*&`).  After an internal swap the caller's `main_cluster` is stale:
  the chosen vertex is stored under the old main
  (`TaggerCheckNeutrino.cxx:592`), `improve_vertex(*main_cluster, …)` becomes
  a no-op (its loops skip nodes of other clusters,
  `NeutrinoVertexFinder.cxx:2052,2060`), the taggers see the old main, and
  `all_clusters` (`TaggerCheckNeutrino.cxx:669-671`) contains the old main
  **twice** (swap pushed it into `other_clusters`).
- **Tiny-main hazards**: a main with <2 Steiner terminals contributes no
  segments (`NeutrinoPatternBase.cxx:1640-1643`) and — unlike companions
  (`TaggerCheckNeutrino.cxx:541-543`) — has **no `init_point_segment`
  fallback**; `examine_main_vertices`' junk-companion pruning collapses to
  ~1 cm (`NeutrinoPatternBase.cxx:2112`); and the `kd_steiner_knn` at
  `NeutrinoPatternBase.cxx:2245` is unguarded against a missing
  `steiner_pc` PC (its guarded siblings at `:2160,:2175` check `has_pc`),
  so a degenerate main can throw.

### 3.5 Single-main assumptions elsewhere

`CreateSteinerGraph.cxx:109` (last-writer-wins main pick) and the
`mother_cluster_id` fill in `SbndPrMagnifyTrackingVisitor.cxx:451` /
`UbooneMagnifyTrackingVisitor.cxx:341` (first main found wins, `-1` if none)
assume one main; the tracking visitors run *after* `TaggerCheckNeutrino` and
would observe a mutated flag set.  Conversely `clustering_cathode_bundle_rescue`
(`:507-510,526-532`), `ClusteringUnmergeBundle` and the TGM/STM/FC loops are
multi-main-clean, and `clustering_cathode_bundle_rescue.cxx:526-532` is the
closest existing precedent for bundle-level promotion (promotes the longest
member of a main-less gid).

## 4. What a promoted associated cluster would carry

Traced through Q/L matching → `ClusteringUnmergeBundle` (the associated
clusters PR sees are made at PR-pipeline position 2, before the taggers at
positions 6-8):

| item | state on an associated cluster | consequence for promotion |
|---|---|---|
| `cluster_t0`, `flash`, `matched_flash_gid` | stamped on **all** bundle members (`QLMatching.cxx:3694-3706`), inherited verbatim through `Cluster::from` on split | flash association and beam-window test are sound |
| `x_t0cor` scope | `switch_scope` (position 1) corrects the *merged* parent with the bundle t0, then the split just moves corrected blobs (`clustering_switch_scope.cxx:69-78`) | geometry identical to a main's |
| `flag_TGM/STM/FC` | **never evaluated** (main-only tagger loops) | promotion is blind to cosmicness — the 52195 failure mode |
| `flashpred` PC | main-only (`QLMatching.cxx:3697`) | absent; nothing in the PR ladder reads it |
| `lm_flag` | stamped on all members (`QLMatching.cxx:3663`), **but** `merge_clusters` re-stamps only t0/flash/gid (`ClusteringFuncs.cxx:374-379`), so through merges `lm_flag` falls to `from()`'s add-if-absent copy in set order | side finding (§8): potentially order-dependent when members disagree |
| Steiner treatment | `CreateSteinerGraph.cxx:293` gives mains the heavy treatment, companions the light one — and steiner (position 4) has already run before any verdict exists | a cluster promoted inside `TaggerCheckNeutrino` runs the *main* PR ladder on a *companion-grade* Steiner cloud |

## 5. Vertex identification: no main privilege to lose

- The DL/SCN network's input cloud is **every vertex and segment point in the
  whole graph** — main and companions, no crop, no membership or fiducial
  gate (`NeutrinoVertexFinder.cxx:3309-3341`); candidates snap to the nearest
  graph vertex of *any* cluster (`:3378,:3472`).
- The rerank's main-cluster bonus does not protect a small main: `W_MAIN=2.0`
  vs a ≥60 cm companion's saturated `W_CLEN=2.0` (`:3535-3544`) — the DL
  score (×1000) decides, and geometry is a tie-break.  If a companion wins,
  the DL path swaps consistently (it takes `Cluster*&`).
- Per-cluster `determine_main_vertex` runs on the main *and every companion*
  (`TaggerCheckNeutrino.cxx:509,533,549`), each landing in
  `map_cluster_main_vertices`; fiducial volume enters only as a +0.5 scoring
  bonus, never as acceptance.

Net: the final vertex lands wherever the graph looks most vertex-like —
which is why 52195's landed on the cosmic's top-wall entry.  Vertexing will
neither malfunction on, nor be protected from, a promoted candidate: it
treats the ensemble almost symmetrically already (the asymmetries are the
flag-gated refinements of §3.3).

## 6. Energy reconstruction: cluster-agnostic, t0-neutral within a bundle

- `fill_kine_tree` takes **no cluster argument** — it BFS-walks the shared PR
  graph from `final_main_vertex` (`NeutrinoKinematics.cxx:171-257`) plus an
  unconditional sweep over all showers with `vtx_type<=3` (`:263-296`).
  Companions contribute either way; what changes with the vertex is the
  reachable set, wholesale.
- Charge maps are bundle-wide (`preload_clusters` over main + companions,
  `TaggerCheckNeutrino.cxx:409-413`).
- **No x-dependent corrections exist** in the energy path: `cal_corr_factor`
  returns 1.0 (`NeutrinoEnergyReco.cxx:14-35`); no lifetime, no
  position-dependent recombination.  And promotion within a bundle is
  **t0-neutral** — all members carry the same flash t0 — so kine gets no new
  x/t0 bias.  (Promoting across *bundles* would be different: the 2D charge
  match has a hard 0.6 cm cut, `NeutrinoEnergyReco.cxx:127`, and a wrong-t0
  frame shift silently zeroes all charge-based energies.)
- On "no vertex found", `T_kine` fills zeros, not −999 (`NeutrinoTaggerInfo.h:27-56`)
  — a failed reconstruction reads as a 0 MeV neutrino downstream (side
  finding, §8).

## 7. Assessment and recommendation

**Design A is safe and cheap.**  The chain is *proven* to handle
"surviving main + associated companions, cosmic main excluded" — that was
the pre-veto behavior, and on the two documented losses (271851, 10550) it
produced interior vertices and few-hundred-MeV energies.  A length threshold
cleanly separates the known populations: restore ≥13 cm mains, keep the
1.3 cm / 1.7 cm shards vetoed — any cut in ~2–13 cm works; **5 cm suggested**
(optionally AND'd with an associated-count minimum).  Implementation is a
second knob inside the existing `cosmic_gids` veto branch; default OFF,
byte-identical.  Residual risk: identical to the pre-veto baseline — an
untagged cosmic among the *associated* clusters can still steal the vertex
(52195-shape, but only when the surviving main also passes the size guard).

**Design B alone is unsound.**  Mechanically feasible (promotion = pointer +
flag move à la `swap_main_cluster`, before the first `find_proto_vertex`),
but with no verdict on associated clusters the promotion is blind, and the
longest associated member of a cosmic-tagged bundle is *frequently the
cosmic's partner* — 52195's T_kine row is precisely what that produces.  It
also requires fixing the §3.4 by-value desync, guarding the §3.4 kNN throw,
handling the Steiner-grade asymmetry (§4), and keeping `mother_cluster_id`
deterministic (§3.5).

**Design C is the principled enabler of "filter out the cosmic and use the
rest".**  `check_tgm(Cluster&)` is flag-agnostic — the main-only restriction
is one caller loop (`TaggerCheckTGM.cxx:242-256`) — and associated clusters
carry the t0 and corrected geometry the check needs (§4).  STM already
*reads* associated clusters as context (`TaggerCheckSTM.cxx:416`); it just
never writes verdicts on them.  With verdicts on associated clusters:
the companion loop can drop cosmic-tagged companions, `nu_skip_cosmic_bundle`
keeps harvesting mains only (unchanged semantics), and promotion (B) becomes
gated instead of blind.  Open physics question for the owner: TGM/STM cut
validity on *fragments* — an associated piece of a broken cosmic need not be
through-going by itself (52195's muon 23 is a complete crosser and would be
caught; a mid-detector fragment might not be).  Cost: extra tagger
evaluations per event, and a revalidation round since new verdicts change
which bundles the veto harvests only if we let them (they shouldn't, at
first).

**Recommended sequence**: implement A now (restores the doc pr/1 §2.2 losses
at zero new risk); take C as its own round if "use the rest" reconstruction
is wanted; consider B only on top of C.  Decision is the owner's.

## 8. Side findings (reported, not fixed — CLAUDE.md rule)

1. `determine_overall_main_vertex` by-value `main_cluster` /
   `map_cluster_main_vertices` desync after an internal swap, incl. the old
   main appearing twice in `all_clusters` (§3.4) — affects the *traditional*
   vertex path today, independent of any refinement.
2. Unguarded `kd_steiner_knn` on the main at `NeutrinoPatternBase.cxx:2245`
   (guarded siblings at `:2160,:2175`).
3. `lm_flag` not in `merge_clusters`' explicit re-stamp list → set-order
   inheritance through merges (§4); `clustering_cathode_bundle_rescue.cxx:506`
   shows the authors carry it explicitly where they knew it mattered.
4. `T_kine` zero-fill (not −999) on vertex-finding failure (§6).
5. `mother_cluster_id` first-main-wins map-order dependence in both tracking
   visitors (§3.5).
6. Stale comment `TaggerCheckNeutrino.cxx:271` credits the flags to
   `clustering_recovering_bundle`, which is defined but not wired into any
   pipeline; the actual producer chain is QLMatching →
   `ClusteringUnmergeBundle`.

## 9. Provenance (study half)

Code read at toolkit `7f29d32d` (apply-pointcloud).  Score rows from
`work-nscbase-ab31` (evt 52195) and `work-nscoff-nuecc48` (evts 271851,
10550) — the doc pr/3 §8.7 arms; no new runs, no labels created.

## 10. Design A implemented: `nu_skip_cosmic_bundle_min_length` (2026-08-01)

Owner decisions: implement A, drop C ("it does not make a lot of sense to
check associated clusters, many times dots"); threshold **15 cm**, not the
suggested 5 ("5 cm is too short").  Consequence accepted up front: 15 sits
above evt 271851's 13.0 cm surviving main, so of the two documented losses
only 10550 (18.5 cm) is restored.

### 10.0 Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
./wcb build --notests -p && ./wcb install --notests -p    # -> libWireCellClus.so md5 c4586ae4
./build/clus/wcdoctest-clus                                # 49 cases / 565 assertions

# compiled-config proofs (wcsonnet; note compiled JSON spells the key '"k" : v')
qlport/scripts/compile_ub_cfg.sh $PWD/cfg /home/xqian/tmp/ub_nbl.json      # cmp-identical to HEAD compile
sbnd_xin/compile_prjob_cfg.sh    $PWD/cfg /home/xqian/tmp/pr_nbl.json      # "nu_skip_cosmic_bundle_min_length" : 15 (once)
# knob-suppression proof: tree with the two SBND 15s sed'ed to 0 compiles
# cmp-identical to the HEAD compile (key absent in both).

# runtime arms (driver run_pr_chain_batch.sh; knob-off arm via
# tmp_run_pr_chain_nblhead.sh = same driver against a HEAD copy of cfg)
cd sbnd_xin
PR_JOBS=6 ./tmp_run_pr_chain_nblhead.sh work-mcp1kall-d59k work-nbloff-ab31    data $AB31   # knob off
PR_JOBS=6 ./run_pr_chain_batch.sh       work-mcp1kall-d59k work-nbl15-ab31     data $AB31   # knob on (15)
PR_JOBS=6 ./tmp_run_pr_chain_nblhead.sh work-nuecc48-nuf   work-nbloff-nuecc48 data
PR_JOBS=6 ./run_pr_chain_batch.sh       work-nuecc48-nuf   work-nbl15-nuecc48  data
PR_JOBS=1 ./run_pr_chain_batch.sh       work-mcp1kall-d59k work-nblrep-279256  data 279256  # determinism probe
python3 pr_arm_compare.py <armA> <armB> <evts...>   # hash_archive members + full T_tagger/T_kine rows
```

### 10.1 The knob

- C++: `TaggerCheckNeutrino` (`clus/src/TaggerCheckNeutrino.cxx`, veto branch
  ~:355): when `nu_skip_cosmic_bundle` would skip an in-window main because it
  shares `matched_flash_gid` with a cosmic-tagged main, a main whose length is
  `>= nu_skip_cosmic_bundle_min_length` (cm, C++ default **0** = veto
  everything, byte-identical) is **kept** instead, with an INFO log line
  (`kept (nu_skip_cosmic_bundle_min_length)`).  The cluster itself carries no
  cosmic verdict by construction (the per-main check at :338 ran first), and
  the tagged bundle-mate stays out of the PR ensemble (§3.1).
- jsonnet: threaded `common/clus.jsonnet` `tagger_check_neutrino(...)` with
  the key-suppression idiom (`> 0` emits); `sbnd/clus.jsonnet` sets **15** at
  the `clus_pr`/`pr` default sites and forwards through both call chains.

### 10.2 Gates (binary `c4586ae4`, freshness proof lib 14:55 > src 14:52)

| comparison | archives (mabc, pctree) | headline scores* |
|---|---|---|
| `work-rpg-ab31` vs `work-nbloff-ab31` (binary change, knob off) | 31/31, 31/31 | identical |
| `work-nbloff-ab31` vs `work-nbl15-ab31` (knob on, mcp30+52195) | 31/31, 31/31 — **no event moves** | identical |
| `work-nbloff-nuecc48` vs `work-nbl15-nuecc48` (knob on, nueCC48) | 47/48, 47/48 — **only 10550** | only 10550 (+3 FP-flicker Enu, ±0.0002 MeV: 269774, 422851, 447477) |

*headline = nue_score, numu_score, numu_cc_flag, cosmict_flag, kine_reco_Enu,
kine_nu_{x,y,z}_corr.

Knob-on behavior, from the logs:

- ab31: the four vetoed bundles' surviving mains are 1.7 / 4.0 / 3.1 / 3.7 cm
  — all < 15, all still skipped (52195 included).  No `kept` line fires.
- evt 10550: `in-window cluster 7 (t0 1.193 us, L 18.5 cm) ... kept
  (nu_skip_cosmic_bundle_min_length)` → `selected main cluster 7 (28
  associated)`.  The restored row **reproduces the pre-veto
  (`work-nscoff-nuecc48`) result exactly**: numu_score −1.595, cosmict_flag 1,
  kine_reco_Enu 387.6 MeV, vertex (−14.8, 59.1, 147.1) cm.
- evt 271851: `L 13.0 cm ... skipping` — below threshold, stays vetoed
  (owner-accepted trade of the 15 cm choice).

### 10.3 Side finding: pre-existing T_tagger detail-branch flicker

`pr_arm_compare.py` compares **every** branch, which the earlier rounds'
row checks did not.  Between any two runs (same binary, same config —
`work-nbl15-ab31` vs the `work-nblrep-279256` repeat proves it) a family of
per-candidate vector branches (`pio_2_v_*`, `shw_sp_pio_2_v_*`, `br3_6_v_*`,
`lol_2_v_*`) and the `numu_cc_1_*` scalars flicker: same value multisets,
permuted candidate order (one entry of `shw_sp_lol_1_v_angle` genuinely
changes on evt 166870).  Rate: 44/48 nueCC48 events, 3/31 mcp30 events —
and the **same magnitude between the pre-existing
`work-nscoff-nuecc48`/`work-nscon-nuecc48` arms (45/48)**, so this is not
introduced by this round.  The archives (mabc, pctree) and every headline
score are stable across the same run pairs.  Likely a pointer-order iteration
in the pi0/candidate enumeration that feeds these vectors (the pr/11 audit
fixed the `out_edges` instance of this class).  Reported, not fixed.

### 10.4 Labels

`work-nbloff-ab31`, `work-nbl15-ab31` (ql root `work-mcp1kall-d59k`, data);
`work-nbloff-nuecc48`, `work-nbl15-nuecc48` (ql root `work-nuecc48-nuf`,
data); `work-nblrep-279256` (determinism repeat).  Comparator:
`sbnd_xin/pr_arm_compare.py`.  Binaries: `55e5e621` (rpg arms) →
`c4586ae4` (this round).
