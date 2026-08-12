# doc pr/66 — evt 18255-10550: why the 1e1p nueCC is still welded to the TGM cosmic

Status: **investigation only, no fix implemented.** Corrects doc pr/18 §6/§9's
attribution of the re-fusion stage.

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
python3 scripts/analysis/pr66/oc66_layer_trace.py
```

Reads two existing arms directly, nothing regenerated: `work-nuecc48-cb0805/
ql_evt10550/mabc-all-apa.zip` (QL: `img-global` + `clustering-global`) and
`work-pr64r4-on48/pr_evt10550/mabc-pr.zip` (PR: `clustering-global`). Both are
current-production output (toolkit `b38127a0`, 2026-08-11 15:07).

## 1. Symptom

The owner scanned the fresh `nuecc48-prod0811` Bee set (index-aligned with the
pr/18-era `2f6ad762` set) and found evt 10550's 1e1p nueCC candidate essentially
missing — the particle-flow tree that ships in the current PR output is a
`pi0 119 MeV` decaying to `gamma 220 MeV -> e- 220 MeV` plus a scatter of small
protons/neutrons around vertex (60.83, 54.98, 165.92) cm, not a 1e1p tree. (This
tree is a `reality=data` event's *reconstructed* PF product, `data/0/0-mc.json`
in the Bee zip — not MC truth; the field name is inherited from the MC schema.)
The owner recalled having separated this event from its overlapping cosmic
before, and suspected a later code change re-merged them.

## 2. Root cause — real, but not where doc pr/18 said

**2.1 Not a regression.** Extracted the in-beam nusel row for 10550 from every
archived arm with a matching record (237 arms, `pr/11`→`pr/65` history) plus the
current one: all 238 show the identical signature
`main 11 | npts_main 12372 | len 374.1 cm | tgm 1 | label TGM`. No code change in
this repo's history has ever moved this row. (16 pre-pr/16 and pr/33-stale-binary
arms instead show a `main 7 | 18.5 cm` row — a different, unrelated Q/L bundle
assignment from before doc pr/16's veto-refinement round; not this bug.)

**2.2 The pr/18 fix still works — at the per-APA layer.** In today's `img-global`
(the Bee layer written at the end of the per-APA clustering chain, before the
all-APA merge and Q/L match) the owner's two points land in **separate**
clusters:

| cluster | npts | PCA length | drift-x extent | `iso_band_like()` |
|---|---:|---:|---:|---|
| 19 (nu candidate) | 2853 | 119.5 cm | 89.1 cm | false |
| 20 (cosmic band) | 9455 | 374.0 cm | 15.0 cm | true |

Closest approach 19↔20 = **0.31 cm** — the identical 0.31 cm touch doc pr/18 §2
measured at the pre-fix merged-cluster stage. 9455 + 2853 = **12308**, exactly
pr/18 §1's quoted pre-fix merged size. `protect_iso_band_xext` is doing its job:
the per-APA chain hands the Q/L stage two genuinely separate clusters.

**2.3 They are fused again by the time Q/L runs — in the all-APA merge chain,
not in `examine_bundles`.** In the QL `clustering-global` layer (post all-APA
merge, still pre-`examine_bundles`... or so doc pr/18 §6/§9 assumed) both points
are inside cluster 11 (12372 pts) — same fused size as before the pr/18 fix
existed. The decisive check: `examine_bundles` itself stamps a per-blob
`real_cluster_id` recording each blob's pre-merge cluster identity
(`clustering_examine_bundles.cxx`, `use_flash_t0` branch, `merge_clusters(...,
"real_cluster_id", ...)`). Inside QL cluster 11:

```
real_cluster_id breakdown: [(11, 12292), (15, 12), (16, 20), (17, 16), (18, 32)]
```

12292 + 12 + 20 + 16 + 32 = 12372. A cluster **already carrying id 11 at size
12292** — band (9455) + nu (2853) minus 16 pts of measurement noise between the
img-global and clustering-global samplings — existed **before** `examine_bundles`
ran. That stage only folded in four small members (12–32 pts each, the specks
`TaggerCheckTGM`'s log later calls "4 demoted main(s)"). `examine_bundles` is
innocent; the band and the nu were already one cluster when it started.

The actual re-fusion happens upstream, in the all-APA merge stages
(`cfg/pgrapher/experiment/sbnd/clus.jsonnet:463-468`: `extend`, `regular`×2,
`parallel_prolong`, `close`, `extend_loop`, then `cathode_connect` /
`cathode_bundle_rescue` before `examine_bundles` runs).

**2.4 Structural reason it sticks.** The per-APA chain runs the same family of
merge stages *and then* `ClusteringSeparate` + `ClusteringNeutrino` as repair
passes — `protect_iso_band_xext` lives inside `ClusteringNeutrino`
(`clus/src/clustering_neutrino.cxx:1039-1066`) and only runs there. The all-APA
chain runs the merge stages with **no repair pass and no iso-band guard of any
kind**. Anything the per-APA repair split apart and handed to Q/L as separate
clusters can be re-glued by the all-APA chain with nothing to stop it.

**2.5 Prime suspect, from a static read (not yet traced).**
`clustering_close.cxx:116` merges a pair when
`dis < length_cut && (length_1 >= 12 cm || length_2 >= 12 cm)`. The all-APA
instance uses `length_cut = 1.2 cm` (`clus.jsonnet:467`). Measured
`dis = 0.31 cm < 1.2 cm`, both lengths (374.0, 119.5 cm) clear 12 cm ⇒
`ClusteringClose:all` merges this pair. This is a static match, not a confirmed
first-fuser — `extend`/`regular`/`parallel_prolong` run earlier in the same
all-APA sequence and could fire first on a different pair-test; §5 below makes
the trace step 1 of any fix.

**2.6 No transitive bridge risk for this event.** Six `img-global` clusters map
into the fused QL cluster 11 (§2.3, `check 4` of the script): the band (c20,
9455 pts), the nu (c19, 2853 pts), and four specks — c10 (34 pts), c3 (20 pts),
c9 (16 of its 25 pts; the other 9 must land in a different QL cluster), c2
(12 pts). Sizes line up one-to-one with §2.3's four `real_cluster_id` specks
(18→32≈c10's 34, 16→20=c3's 20, 17→16=c9's 16, 15→12=c2's 12; the c10/18 gap is
resampling between the `img-global` and QL `clustering-global` point clouds,
same effect as the 12308-vs-12292+16 gap in §2.3) — i.e. these are the same four
specks, independently confirming §2.3's arithmetic. Pairwise closest approach:

|  | c20 | c19 | c10 | c3 | c9 | c2 |
|---|---:|---:|---:|---:|---:|---:|
| **c20** | – | 0.31 | 86.8 | 179.7 | 80.3 | 157.8 |
| **c19** | 0.31 | – | 11.0 | 112.5 | 44.0 | 137.3 |

No speck sits within reach of *both* c19 and c20 at any all-APA stage's cut
(the closest, c10, is 11.0 cm from c19 but 86.8 cm from c20 — both well outside
`close`'s 1.2 cm and every other stage's tighter cuts). A single pairwise veto
on the direct c19↔c20 edge is therefore geometrically sufficient for this
event; a group-level exclusion is not required here (though see §5.3's
fallback-within-the-fallback for other events).

**2.7 What the owner is actually looking at.** Mapping the 2853 nu-candidate
points from `img-global` cluster 19 into the PR `clustering-global` layer:

```
nu candidate (img c19, 2853 pts) lands on PR clusters:
  [(11, 1657), (48, 364), (62, 194), (49, 85), (47, 80), (63, 61), (43, 54), (26, 32), ...]
-> 1657/2853 pts (58%) remain inside PR main cluster 11 (TGM-tagged, nu_skip_cosmic'd)
-> 1196/2853 pts carved out by ClusteringUnmergeBundle into 28 small associated cluster(s)
```

`TaggerCheckNeutrino` skips cluster 11 outright
(`nu_skip_cosmic: L 378.0 cm, cosmic-tagged`), so the majority of the nu's
charge is invisible to particle-flow. PF instead runs on the 28 leftover
fragments `ClusteringUnmergeBundle` peeled off cluster 11's flash-group plus the
untouched small clusters — hence the reconstructed `pi0`/`gamma` tree the owner
is seeing instead of a 1e1p.

## 3. Correction to doc pr/18

Doc pr/18 §6 states: *"PR outcome: ... `examine_bundles` collapses the bundle
back into one 12372-pt cluster, and TGM still kills it."* §9's proposed
follow-ups — a per-member `nu_skip_cosmic_bundle` evaluation, or "a band-aware
member split at `unmerge_bundle`" — were both written on that premise.

§2.3 above shows the premise is wrong: `examine_bundles` receives a cluster that
is *already* fused (12292 of the 12372 points, under one `real_cluster_id`
before that stage runs); it only adds four small specks. The 12372-pt fusion pr/18
§6 attributed to `examine_bundles` was made earlier, in the all-APA merge
chain. Consequently:

- §9's `unmerge_bundle`-based follow-up cannot work: `unmerge_bundle` operates
  on the PR-stage cluster (already merged upstream at Q/L time, §2.7), and by
  the time it runs the nu's charge is already split between a TGM main and 28
  scattered associated fragments — there is no single "band-aware member" left
  to split out.
- A per-member `nu_skip_cosmic_bundle` evaluation (§9's other option) also
  cannot reach this case: the nu candidate was never a matched bundle *main*
  (the 374 cm band was — confirmed by `TaggerCheckTGM`'s
  `evaluate_demoted_mains: 4 demoted main(s) added`, which are the four
  12-32 pt specks, not the nu), so `save_bundle_main_provenance` /
  `restore_demoted_mains` has no record of it to restore.

Both pr/18 §9 follow-ups target the wrong layer. The fix has to be upstream, at
the all-APA merge itself (§5).

## 4. Why it hid

The in-beam nusel row for this event has never moved (§2.1) — every A/B gate
run across the pr/11→pr/65 campaign passed on 10550 by construction, because
none of them look inside a bundle's composition, only at its summary row
(main id, npts, length, flags). The pr/18 fix's own validation (§6 of that doc)
already observed "PR outcome: the nusel rows are LINE-IDENTICAL to the
baseline" and correctly flagged recovering the event as follow-up work (§9) —
it just misdiagnosed which stage to fix. Nothing downstream renders visibly
broken: one TGM-tagged bundle, a normal-looking (if physically wrong) PF tree,
no crash, no log warning. Only a hand-scan of the reconstructed PF tree against
the expected 1e1p topology exposes it.

## 5. Proposed fix (NOT implemented this round)

Two designs, in preference order. Both need §5.1's trace for understanding and
verification; only Design B (§5.3) needs it as a hard prerequisite for
*correctness* — see the trade-off note at the end of §5.2.

### 5.1 Trace (useful for both designs)

Per-stage Bee trace, the same tool doc pr/18 §2 used to name the original
re-fuser, this time on the all-APA chain:

```bash
SBND_WORK_ROOT=$PWD/work-oc66-tr10550 SBND_TRACE_BEE=1 ./run_ql_evt.sh data 4
```

Names which all-APA stage's pair test *first* connects the band and the nu.
§2.5's `ClusteringClose:all` match is a static, untraced prediction, not a
confirmed fuser — `extend`/`regular`/`parallel_prolong` run earlier in the
same sequence and could fire on a different pair-test first.

### 5.2 Design A (preferred) — provenance: honor the decision `protect_iso_band_xext` already made

Rather than re-deriving "is this a band vs. a drift-spanning cluster" a second
time at the all-APA stage, have the per-APA stage record that it *already
refused* this specific pair, and have every all-APA merge-pair test honor
that record instead of re-judging geometry.

**Architecture check (this corrects an earlier verbal claim in this
investigation — worth stating plainly since it changes what's buildable).**
Per-APA and all-APA clustering are **not** one continuous pass on one
in-memory tree. They are two separate `MultiAlgBlobClustering` pnodes,
joined by a `PointTreeMerging` pnode (`clus.jsonnet`'s `clus_all_apa`
function, `local pcmerging = g.pnode({type: 'PointTreeMerging', ...})`), and
the all-APA chain's *first* step, `ClusteringSwitchScope`
(`clus/src/clustering_switch_scope.cxx`), explicitly **destroys and rebuilds
every `Cluster` object** to apply the newly-available T0 correction
(`cluster->add_corrected_points(...)` then `live_grouping.separate(cluster,
..., true)` — the old `Cluster*` is invalid after this call). A plain
in-memory marker on a `Cluster` would not survive this rebuild, and neither
would an arbitrary custom-named per-blob array survive an *ordinary*
`merge_clusters()` call anywhere in the pipeline — `ClusteringFuncs.cxx`'s
`merge_clusters()` only forwards the small set of pcarrays named in its own
parameter list or its internal `carry_pairs` allowlist
(`{"assoc_cluster_id", "assoc_cluster_main"}` today), by explicit design
(doc 52 Stage 2 defect D4: *"until now nothing could CARRY an existing
per-blob array THROUGH a merge — so any provenance array died at the next of
the 12 `merge_clusters()` call sites... Handled here, for the whole codebase
at once, rather than per call site"*).

**The good news:** this exact survival problem already has a working,
precedented solution the codebase uses today for `real_cluster_id` /
`assoc_cluster_id` — and both of the relevant allowlists carry an explicit
invitation to extend them:

- `clustering_switch_scope.cxx`'s `carry_anames` array (currently
  `real_cluster_id`, `real_cluster_main`, `assoc_cluster_id`,
  `assoc_cluster_main`, `real_cluster_was_main`), with the comment *"Any
  future per-blob provenance array goes here and nowhere else."*
- `ClusteringFuncs.cxx::merge_clusters()`'s `carry_pairs` array, the doc-52
  fix for exactly "a provenance array written earlier must survive a later,
  unrelated merge that doesn't know about it."

**Design.** A new per-blob `"perblob"` int pcarray, e.g. `iso_band_refusal_id`:

1. **Write** — at the point `protect_iso_band_xext` currently refuses a merge
   and just `continue`s (`clustering_neutrino.cxx:1054-1066`), stamp every
   blob of `cluster1` with `+N` and every blob of `cluster2` with `-N`, where
   `N` is a small counter unique within that visitor invocation (no
   cross-event or cross-ident collision risk — the array only needs to
   distinguish refusal events within one QL run).
2. **Carry** — add `"iso_band_refusal_id"` to both allowlists above (two
   one-line additions, same pattern as the `real_cluster_id`/
   `assoc_cluster_id` precedent), so the marker survives `switch_scope`'s
   rebuild and any ordinary merge between the refusal point and wherever the
   all-APA chain re-tests the pair.
3. **Read** — a small shared helper (e.g.
   `bool refused_pair(Cluster* c1, Cluster* c2)`, checking whether `c1`
   carries some `+N` that `c2` carries as `-N` or vice versa), called right
   before `boost::add_edge(...)` in **every** all-APA merge-pair loop
   (`extend`, `regular`×2, `parallel_prolong`, `close`, `extend_loop`,
   `cathode_connect`, `cathode_bundle_rescue`) — not just the one the trace
   names. Marker line on refusal for greppability, same convention as the
   existing pr/18 markers.

**Why this doesn't need §5.1's trace as a hard prerequisite:** because the
check is a cheap, uniform veto (an int/set lookup, no geometry) applied at
every all-APA pair-test site, it is correct regardless of which stage
actually fires first — unlike Design B, which re-derives geometry and must be
placed at exactly the right stage to work. The trace remains valuable for
*understanding and verifying* the fix (confirming the marker fires where
expected, per §6), just not for correctness.

**Trade-off vs. Design B:** narrower and more precise — it only blocks the
*specific* pair `protect_iso_band_xext` already decided about, not a general
geometric class, so it cannot introduce a new false-positive veto on an
unrelated event. It does not duplicate `iso_band_like()`/`blob_center_xext()`
a second time. Its coverage is bounded by where `protect_iso_band_xext`
itself fires: an event whose band+non-band fusion happens *before*
`ClusteringSeparate`/`ClusteringNeutrino` ever run in the per-APA chain would
get no marker and no protection — Design B's geometric predicate has no such
gap, at the cost of needing the trace to place it correctly and being a
broader (if already-1000-event-validated) class of veto.

### 5.3 Design B (fallback) — re-derive the geometric predicate at the all-APA stage

Duplicate (not extract, M10) `iso_band_like()` and `blob_center_xext()`
(`clus/src/clustering_neutrino.cxx:83-106`) into the merge-pair loop of the
stage §5.1's trace names (prime suspect: `clustering_close.cxx:220-225`),
behind the same two config keys and the same semantics doc pr/18 already
validated: `protect_iso_band` (C++ default `false`) + `protect_iso_band_xext`
(C++ default `0`, read only when the first is `true`). Key omitted when off
⇒ byte-identical. Print the same style of unconditional marker line, gated on
`protect_iso_band_xext > 0`.

Wire it to the all-APA instance only: `cfg/pgrapher/experiment/sbnd/
clus.jsonnet:467`, inside `clus_all_apa` — a new `wct-clus-matching-perevt.
jsonnet` TLA arg (proposed name `bundle_iso_band_guard`, default `false` this
round) plus an `SBND_*` runner escape in `run_ql_evt.sh`, mirroring
`SBND_NU_ISO_GUARD`'s pattern. `cm.close()` (and the other candidate merge
stages) are also called from `clus_per_face` (the per-APA chain); threading
the same arg through that shared call site would change per-APA output and
alter what `ClusteringSeparate`/`ClusteringNeutrino` see downstream, breaking
byte-identity for every other detector using these shared `clus.jsonnet`
builders. The all-APA and per-APA call sites must take independent knobs.

**Fallback-within-the-fallback, documented but not currently needed (§2.6):**
if a future event's trace shows a bridging speck connecting the band and the
nu candidate through an intermediate member, a pairwise edge veto is
insufficient — the correct form becomes group-level: *exclude any non-band
member whose blob-center drift-x extent exceeds the knob from a flash-t0 (or
geometric-proximity) group that also contains a band-like member*, rather
than only refusing the direct edge.

## 6. Verification plan (for when the fix is implemented)

Applies to whichever design is chosen; Design A adds two items specific to the
carry-forward plumbing (marked below).

- **Knob-off byte-identity**: compiled-config `cmp` with the key absent; the
  `abtest` pdhd+pdvd clus-stage gate (`ab_compare.sh` OVERALL PASS); an SBND
  runtime hash check on 10550 with the new knob unset vs. today's
  `work-nuecc48-cb0805`/`work-pr64r4-on48` hashes. For Design A specifically:
  the two `carry_anames`/`carry_pairs` allowlist additions are unconditional
  code (no config gate), so their byte-identity proof is that the array is
  simply absent (no writer fires) when `protect_iso_band_xext` is off anywhere
  in the config — confirm via `wcdoctest-clus` and the same runtime hash check.
- **Compiled-config proof**: the new key present on the stage's `:all`
  instance, **absent** on its `:apa0` / `:apa1` instances.
- **Design A only — carry-forward proof**: dump `iso_band_refusal_id` from the
  QL `clustering-global` layer's `perblob` PC on a knob-on run of 10550 and
  confirm it is present and matches the `+N`/`-N` convention on the (still
  separate, pre-veto) band/nu clusters, i.e. the marker survived
  `PointTreeMerging` + `ClusteringSwitchScope`'s rebuild intact.
- `./build/clus/wcdoctest-clus` passes on both builds.
- **Knob-on acceptance — measured, not assumed** (doc pr/18 §6 is the
  precedent: a clustering-level split was achieved there and the PR rows still
  came out line-identical because a later stage re-fused; the same trap applies
  here unless checked explicitly):
  1. the nu candidate survives as its own cluster in the QL/PR
     `clustering-global` layers;
  2. it is **not itself** TGM-tagged — worth stating explicitly, since at
     119.5 cm length / 89.1 cm drift-x extent it is a genuinely drift-spanning
     object and could plausibly satisfy `TaggerCheckTGM`'s own criteria;
  3. `nu_skip_cosmic_bundle_min_length` keeps it (the same log line pr/3's
     doc already shows firing for this event's other in-beam member, cluster
     7, `L 18.5 cm ... kept`);
  4. the in-beam nusel row for 10550 changes from today's
     `main 11 | npts_main 12372 | len 374.1 cm | tgm 1 | TGM`;
  5. the reconstructed PF vertex lands on the nu candidate with a 1e1p-like
     tree, not today's `pi0 119 MeV` / `gamma 220 MeV -> e- 220 MeV`.
- Then the standard no-regression sweep (nueCC48 48-event + mcp1k 1000-event,
  doc pr/18 §7's precedent) **before** any owner flip decision.

## 7. Files

- wcp-porting-img: `scripts/analysis/pr66/oc66_layer_trace.py` (this doc's
  Repro), this doc. No toolkit files touched — investigation only.

## 8. Open questions for the owner

- **Design A vs. Design B** (§5.2 vs §5.3): recommend Design A (provenance) —
  narrower, no geometry re-derivation, doesn't need §5.1's trace for
  correctness — but it rides on two allowlist extensions
  (`clustering_switch_scope.cxx`'s `carry_anames`,
  `ClusteringFuncs.cxx::merge_clusters()`'s `carry_pairs`) that are shared,
  codebase-wide mechanisms also used by `real_cluster_id`/`assoc_cluster_id`;
  confirm the owner is comfortable extending shared infrastructure for a
  single-purpose marker before implementation.
- Confirm the proposed config-knob name/shape (§5.3, Design B only —
  Design A's marker carries no config knob of its own beyond
  `protect_iso_band_xext` gating whether it is ever written).
- §5.1's trace may surface a different first-fusing stage than
  `ClusteringClose:all`; for Design B that changes where the predicate is
  inserted, for Design A it only changes what the trace confirms (Design A's
  check runs at every all-APA merge-pair site regardless).
