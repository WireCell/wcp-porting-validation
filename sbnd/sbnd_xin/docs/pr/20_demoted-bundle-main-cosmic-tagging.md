# doc pr/20 — a Q/L bundle main demoted by the flash-group merge is never cosmic-tagged

**Status: DESIGN ONLY — no code written, no knob exists yet.** This document
records the diagnosis (fully evidenced, primary-source) and the proposed fix
(four default-OFF knobs). Nothing in the toolkit or in `sbnd_xin/` changed for
this doc.

Reproducing case: **SBND run 18255 / evt 59003**, Bee index 4 of the
`cath13-prod-20260801` set (`docs/pr/cath13-prod-20260801.index.txt`,
https://www.phy.bnl.gov/twister/bee/set/1e45d9e5-c5ad-485d-8ced-6934f3c866cf/event/list/).
Cluster **26006** in the Particle Flow display — a through-going cosmic drawn
as a 361 MeV gamma hanging off the neutrino candidate's muon.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# the two arms this doc reads (already on disk, produced 2026-08-01)
ls work-mcp1kall-cath13ql/ql_evt59003/   # Q/L arm  (mabc-all-apa.zip, pctree, log)
ls work-cath13pr/pr_evt59003/            # PR arm   (mabc-pr.zip, tracking-pr.root, log)

# the particle-flow tree Bee draws
unzip -p work-cath13pr/pr_evt59003/mabc-pr.zip data/0/0-mc.json

# the energy budget
python3 -c "import uproot; t=uproot.open('work-cath13pr/pr_evt59003/tracking-pr.root')['T_kine']; \
  print(t['kine_reco_Enu'].array(library='np'), t['kine_energy_particle'].array(library='np'), \
        t['kine_reco_add_energy'].array(library='np'))"

# the pre-merge provenance (per-blob arrays inside the Q/L pctree tarball)
mkdir -p /home/xqian/tmp/evt59003/pct2
tar xzf work-mcp1kall-cath13ql/ql_evt59003/pctree-evt59003.tar.gz -C /home/xqian/tmp/evt59003/pct2
grep -l assoc_cluster_main /home/xqian/tmp/evt59003/pct2/*_metadata.json
#   -> tensor 141 = real_cluster_id, 142 = real_cluster_main, 139 = assoc_cluster_main
```

## Symptom

The Bee Particle Flow tree for evt 59003 (`0-mc.json`):

```
mu-    732 MeV  (19000)   start (42.2, -15.4, 210.1)   end (-29.0, 0.01, 500.1)
└ gamma 361 MeV  (id 4)   start (-29.0, 0.01, 500.1)   end (-80.4, 105.2, 500.5)
  └ e-  361 MeV  (26006)
```

The `gamma` node's start→end **is** the conversion gap: **117.1 cm**, i.e.
6.5 conversion lengths (9/7·X₀ = 18 cm), survival probability ≈ 1.5 × 10⁻³.
Raw closest approach between the two objects' point clouds is **114.7 cm**
(cluster 19 at (-28.3, 2.7, 495.8) ↔ cluster 26 at (-80.5, 104.8, 500.9)).

Cluster 26 is plainly a through-going cosmic: it spans y 104.8 → **199.7**
(top face at y = 200) and z 459.5 → **501.1** (downstream wall at z = 500).

It is not a display artifact. `T_kine`:

```
kine_energy_particle = [732.206, 361.516, 0.598, 1.117, 1.448]
kine_reco_add_energy = 105.658
                sum  = 1202.543  ==  kine_reco_Enu = 1202.544
```

Every term is in the total, so the cosmic supplies **361.5 of 1202.5 MeV —
30 % of the reconstructed neutrino energy**. The event is tagged numu-CC
(`numu_score` 4.07; `nue_score` is the −15 sentinel). Whether the spurious
shower moved `numu_score` is untested.

## Root cause

### 1. Q/L matched two bundles to the beam flash

```
659: flash_bundles_map: flash id 10 ... cluster gidx 2 total_pred_light 5971.70
660: bundle_flags:      flash id 10 cluster ident 3 gidx 2 | at_x_boundary true  ... high_consistent true
661: flash_bundles_map: flash id 10 ... cluster gidx 3 total_pred_light 2472.19
662: bundle_flags:      flash id 10 cluster ident 4 gidx 3 | at_x_boundary false ... high_consistent false
641: after merge, meas pe 6351.78, pred pe 8443.89, ks_dis 0.056, chi2/ndf 2.18
```

5971.70 + 2472.19 = 8443.89. Their solo evaluations:

```
378: flash 10 and cluster 2, pred PE 5971.69, npts 2162, ks 0.083, chi2/ndf 1.23
379: flash 10 and cluster 3, pred PE 2472.19, npts 1155, ks 0.121, chi2/ndf 34.65
807: LM verdict: cluster 3 gidx 2 ... len 144.5 cm x_bnd true  -> lm=0
808: LM verdict: cluster 4 gidx 3 ... len 109.4 cm x_bnd false -> lm=0
```

`npts 1155` is exactly the number of points that later carry
`real_cluster_id == 26`. **The cosmic entered as a bundle main in its own
right.** Note what the merge accepted: prediction over-shoots measurement by
33 % and chi²/ndf worsens 1.23 → 2.18; only `ks_dis` improves.

### 2. The flash-group merge keeps one flag donor

`clus/src/ClusteringFuncs.cxx:209-215` — `Cluster::from()` copies each member's
flags in turn, so with `flags_from_longest` the merged cluster's flags are
re-applied from a single representative: the longest flash-bearing member.
144.5 cm beats 109.4 cm, so cluster 19's flags survive and cluster 26's
`flag_main_cluster` is discarded. The header comment
(`ClusteringFuncs.h:157`) names this exact case:

> The "main_cluster" flag cannot serve: a flash group can merge several
> bundles, each of whose mains carries it.

The information is destroyed at one line, `ClusteringFuncs.cxx:367`:

```cpp
orig_main[i] = (orig_id[i] == rep_ident) ? 1 : 0;
```

Every loser's `main_cluster` flag is in scope right there and is thrown away.

Provenance after the merge (read from the Q/L pctree tarball):

| rcid | nblobs | `real_cluster_main` | `assoc_cluster_main` |
|---|---|---|---|
| 19 | 374 | all 1 | {1: 363, 0: 11} |
| 26 | 143 | all 0 | all 1 |

At the Bee/point level the merged Q/L cluster 19 holds 5139 points splitting
{19: 3984, 26: 1155} — which is the `npts_main` in `nusel-table.tsv`.

### 3. The unmerge demotes everything it splits off

`clus/src/ClusteringUnmergeBundle.cxx:374-377`:

```cpp
for (auto& [gid, part] : splits) {
    part->set_flag(Flags::main_cluster, 0);
    part->set_flag(Flags::associated_cluster);
```

Deliberate and correct for its purpose ("that would make every fragment look
like a bundle main to STM/TGM/FC") — but blanket. It cannot tell a genuine
ex-bundle-main from a 5-point shard.

### 4. So no cosmic tagger ever examines it

All three gate on the flag (`TaggerCheckTGM.cxx:241`, `TaggerCheckSTM.cxx:406`,
`TaggerCheckFC.cxx`), and the PR log shows exactly one main evaluated:

```
TaggerCheckTGM: skipped 7 out-of-scope main cluster(s)
TaggerCheckTGM: beam_window_only [0.200, 2.200) us: 1 main(s) evaluated, 14 out of window
TaggerCheckTGM: cluster 19 → TGM=false
TaggerCheckSTM: cluster 19 → STM=0 TGM=0
TaggerCheckFC:  cluster 19 → FC=false
```

`nusel-table.tsv` has no row for cluster 26 — rows are per-main.

### 5. And then it is adopted into the neutrino

`TaggerCheckNeutrino.cxx:386-394` builds `other_clusters` as *every*
`associated_cluster` sharing the main's `matched_flash_gid`, with no cosmic
filter:

```
TaggerCheckNeutrino: selected main cluster 19 (t0 1.578 us, L 298.4 cm, 4 associated)
```

Those get trajectory-fit, and `shower_clustering_in_other_clusters`
(`NeutrinoShowerClustering.cxx:1295`) promotes any of them longer than 4 cm to
a shower hung off the nearest main-cluster vertex — **with no maximum
connection distance**.

**This is prototype behaviour, not a port defect.**
`prototype_base/pid/src/NeutrinoID_shower_clustering.h:1442` is the same
function with the same `< 4*units::cm` gate, the same
`min_dis > 0.8 * main_dis` fallback, and no distance ceiling. M15 applies: do
not "fix" this by divergence from the prototype.

## Why it hid

- The demotion is invisible in every per-main product: no `nusel-table.tsv`
  row, no tagger log line, no `T_tagger` entry.
- `nu_skip_cosmic_bundle` (45dae9d0, doc pr/3 §8) looks like it covers this and
  does not: it skips an in-window **main** sharing a bundle gid with a
  cosmic-tagged **main**. Here the cosmic is not a main, so the knob is
  structurally inapplicable.
- STM is not blind to cluster 26 — `TaggerCheckSTM.cxx:451-457` gathers it as a
  companion and `check_other_clusters()` examines it — but the companion test
  requires closest distance < 25 cm and the gap is 114.7 cm, so it contributes
  nothing and says nothing. A second, independent place in the code agrees the
  two objects are unrelated; nothing acts on that.
- The only surviving handle is `n_frag` in `nusel-table.tsv`, the count of
  distinct `real_cluster_id` fragments in a main. Verified against this event:

  | main | n_frag | rcid fragments |
  |---|---|---|
  | 16 | 1 | {16: 1631} |
  | **19** | **2** | **{19: 3984, 26: 1155}** |
  | 21 | 2 | {21: 577, 29: 355} |
  | 20 | 3 | {20: 579, 27: 12, 28: 38} |
  | 18 | 4 | {18: 61, 23: 6, 24: 27, 25: 6} |

  Any nu-candidate row with `n_frag > 1` is a merged main whose companions are
  unexaminable by the taggers.

## Fix (proposed — four knobs, all default OFF)

Design principle: **record the fact where it is destroyed, re-express it as a
separate flag, let the cosmic taggers see it, act on the verdict.**

### P1 — preserve the pre-merge main flag

`merge_clusters()` (`clus/src/ClusteringFuncs.cxx`) gains an optional
`orig_wasmain_aname`, default `""`. One line in the member loop beside the
existing `orig_id.resize(...)`, before `destroy_child(live)`:

```cpp
if (save_wasmain)
    orig_wasmain.resize(fresh_cluster.nchildren(), live->get_flag(Flags::main_cluster));
```

Same resize-fills-new-rows idiom as `orig_id`.

`clus/src/clustering_examine_bundles.cxx:153` is the sole call site; it passes
`"real_cluster_was_main"` when a new visitor knob `save_bundle_main_provenance`
is true, `""` otherwise.

**Fill-in is mandatory, not optional.** `aux/src/TensorDMpointtree.cxx:88-93`:

> `append()` keys the copy on the ACCUMULATED dataset: an array whose key is
> absent from the first-seen node's same-named PC is silently dropped (and a
> key the tail lacks throws). Same-named local PCs must therefore be
> key-homogeneous across nodes to round-trip. […] it cost a debugging session
> once (perblob/real_cluster_id, SBND flash merge).

So MABC must fill the new array on every cluster that lacks it, with the same
"absent provenance ⇒ own main ⇒ all 1" sentinel it already applies to
`real_cluster_main` (`MultiAlgBlobClustering.cxx:2615-2628`). Make the fill-in
**presence-triggered** (fill iff any cluster has the array) rather than adding
a second knob — two independent knobs would allow the throwing configuration.
Read-back is name-agnostic; there is no array whitelist.

`check_perblob_provenance` already validates per-key row counts at save time,
so a bug in the resize pattern trips a logged violation rather than corrupting
silently.

Empty name ⇒ array never created ⇒ no key-set change ⇒ byte-identical for
every caller (PDHD, PDVD, uBooNE, SBND).

### P2 — re-express it as a SEPARATE flag

`ClusteringUnmergeBundle`, knob `restore_demoted_mains` (default false): a
split part whose rows are all `real_cluster_was_main == 1` additionally gets a
new `Flags::demoted_main`. It keeps `associated_cluster`. It does **not** get
`main_cluster` back.

That is the load-bearing decision and it is not stylistic.
`nu_skip_cosmic_bundle` builds its veto set from main-flagged clusters
(`TaggerCheckNeutrino.cxx:311`):

```cpp
for (auto* cluster : grouping.children()) {
    if (!cluster->get_flag(Flags::main_cluster)) continue;
    ...
    if (cluster->get_flag(Flags::TGM) || ...) cosmic_gids.insert(gid);
}
```

Restore `main_cluster` on cluster 26 and then tag it TGM, and gid 10 lands in
`cosmic_gids` — main 19's survival then rests entirely on the SBND
`nu_skip_cosmic_bundle_min_length` = 15 cm guard (doc pr/16 §10). It would
survive at 298 cm, but the neutrino's existence would have been made contingent
on a guard that exists for a different purpose. A separate flag keeps the blast
radius at zero.

Same key-homogeneity trap one level up: when the knob is on, `demoted_main`
must be materialised (0/1) on **every** cluster for the `cluster_scalar` PC,
exactly as `QLMatching.cxx:1341` does for `main_cluster`/`associated_cluster`.

### P3 — let the cosmic taggers see them

`TaggerCheckTGM` **and** `TaggerCheckSTM` (and `TaggerCheckFC`, informational),
knob `evaluate_demoted_mains` (default false):

```cpp
if (!cluster->get_flag(Flags::main_cluster)
    && !(m_evaluate_demoted_mains && cluster->get_flag(Flags::demoted_main))) continue;
```

Scope filter and beam-window gate follow unchanged. No other prerequisite:
`CreateSteinerGraph` already builds graphs for these clusters — its beam-window
gate deliberately keeps companions (`CreateSteinerGraph.cxx:144`), and the PR
log confirms `kept 5 of 51 cluster(s) (1 in-window main(s), 1 flash group(s))`
= main 19 + its 4 associated. `separate()→from()` already copied `cluster_t0`
and `matched_flash_gid` onto every split part, so the beam-window gate passes.

Both taggers, not TGM alone: a demoted main is not a fragment, it is a Q/L
bundle main — exactly the population the TGM and STM cuts were tuned on.
Running both is nearly free because `TaggerCheckSTM` honours an upstream TGM
verdict ("a through-going muon is never an STM"), so a cluster like 26 that TGM
catches on geometry short-circuits STM.

Two implementation details:

- **Self-exclusion.** Companions are gathered by `matched_flash_gid`, which a
  demoted main shares, so it would appear in its own `associated_clusters`
  list. Needs an explicit skip.
- **Whose companions?** Under gid-keying, evaluating cluster 26 would hand it
  clusters 56/57/58 — but those are main 19's pieces (`unmerge_assoc`:
  `cluster 19: 374 blobs -> main 363 + 3 associated`), while `unmerge_bundle`
  gave cluster 26 no sub-pieces at all. The prototype's semantics is
  per-*bundle*; the gid is per flash *group*, too coarse once several bundles
  merge. P1's array fixes this as a bonus: a companion's bundle is identified
  by its `real_cluster_id`, so companion sets can be rebuilt per bundle
  instead of per gid — cluster 26 gets an empty list, main 19 keeps
  {56, 57, 58}.

### P4 — act on the verdict

`TaggerCheckNeutrino.cxx:386-394`, knob `skip_cosmic_companions` (default
false): when building `other_clusters`, skip a companion **if it is TGM- or
STM-tagged AND its length ≥ `cosmic_companion_min_length`**. Below that length
it stays in regardless of verdict, so a short mis-tagged neutrino daughter can
never be silently dropped and a bad tag on a fragment is bounded.

Same shape as `nu_skip_cosmic_bundle_min_length`, but a different question, so
it wants its own tuning rather than inheriting the 15 cm.

## Alternative considered and rejected: no new array

`assoc_cluster_main` already separates them on this event (table in §Root cause
2): rcid 19 → {1: 363, 0: 11}, rcid 26 → all 1. So "companion whose blobs are
all `assoc_cluster_main == 1`" picks out cluster 26 today with zero new state.

Rejected on semantics: that array records "was the main of its *isolated
grouping*", not "was the main of a matched *Q/L bundle*". Those coincide for
matched clusters. Whether the flash-time merge can ever group an unmatched
cluster (`t0 = -1e12`) into a beam bundle has **not** been checked — that is
where they would diverge. P1 records the flag that was actually lost, and says
so in its name.

## Verification plan

`clus/src/ClusteringFuncs.cxx` is linked by every detector, so knob-off needs
the full sweep.

| Gate | Scope |
|---|---|
| `abtest/ab_compare.sh <pre> <post>` | pdhd + pdvd, `events.txt` |
| `qlport/scripts/ab_check.sh` | uboone, gate 1 + gate 2 |
| manual `hash_archive.py` | SBND `mabc-all-apa.zip` + `mabc-pr.zip` |
| `./build/clus/wcdoctest-clus` | — |

Knob-on demonstration is this event: cluster 26 tagged TGM, dropped from
`other_clusters`, the `gamma 361 MeV` node gone from the PF tree,
`kine_reco_Enu` 1202.5 → ≈ 841 MeV.

**Metric to agree before shipping** — the `n_frag > 1` census over the
1000-event arm (`work-mcp1kall-u17on1kb`, read-only): how many nu-candidates
carry demoted mains, their length/charge, how many are TGM-taggable on
geometry, and the resulting `kine_reco_Enu` shift distribution. Hard
constraint: zero selection changes on the 48 nueCC events.

## Staging

P1 + P2 are inert bookkeeping — they record and re-express a fact and change no
verdict. They can land and be gate-proven ahead of P3 + P4, which is where the
physics moves.

## Open questions

1. Should Q/L have merged the two bundles at all? The merge over-predicts by
   33 % and worsens chi²/ndf; only `ks_dis` improves. Tightening the
   flash-group merge is an upstream alternative to this whole design and is
   **not** analysed here.
2. Whether the 361 MeV shower moved `numu_score` (needs an A/B).
3. Whether `assign_flash_t0_groups` can group unmatched clusters
   (`t0 = -1e12`) — see §Alternative.

## Related

- doc pr/3 §8 — `nu_skip_cosmic_bundle` (the per-main version of this veto)
- doc pr/12 — cathode-crossing neutrino PR; evt 59003 is one of its 13 cases
- doc pr/14 / pr/16 / pr/17 — cathode bundle rescue and the min-length guard
- doc 45 — unmerge bundle → main + associated
- doc 53 — `real_cluster_id` epochs
