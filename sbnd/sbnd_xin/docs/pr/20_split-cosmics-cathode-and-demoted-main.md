# doc pr/20 — one cosmic, two objects: a demoted bundle main, and a cathode crossing broken in two

Two independent ways a single cosmic becomes more than one object in the SBND
pattern-recognition chain. They were diagnosed in separate rounds and are
collected here because the second ends in the same place as the first: a cosmic
fragment that no cosmic tagger ever examined, adopted into the neutrino
candidate as a gamma hanging off its muon.

| | part | reproducing events | status |
|---|---|---|---|
| **I** | a Q/L bundle main demoted by the flash-group merge is never cosmic-tagged | 18255 / 59003 | design only — 4 knobs, all default OFF |
| **II** | a cathode-crossing track broken in two | 18259 / 169824, 18255 / 406796, 18255 / 315497 | diagnosis complete and reproduced at HEAD; fix design only — 2 SBND config lines + 2 new default-OFF passes |

No C++, jsonnet or runner changed for either part. The files this doc adds are
its two figures under `docs/pics/` and the three read-only analysis scripts
named in the Part II repro block.

A plan-review round (2026-08-02) went over both fix designs before any
implementation. Its inline additions are marked *(plan review)*; the combined
ordering of the two parts is the new §"Execution order" at the end.

## Part I — a Q/L bundle main demoted by the flash-group merge is never cosmic-tagged

**Status: DESIGN ONLY — no code written, no knob exists yet.** This part records
the diagnosis (fully evidenced, primary-source) and the proposed fix (four
default-OFF knobs).

Reproducing case: **SBND run 18255 / evt 59003**, Bee index 4 of the
`cath13-prod-20260801` set (`docs/pr/cath13-prod-20260801.index.txt`,
https://www.phy.bnl.gov/twister/bee/set/1e45d9e5-c5ad-485d-8ced-6934f3c866cf/event/list/).
Cluster **26006** in the Particle Flow display — a through-going cosmic drawn
as a 361 MeV gamma hanging off the neutrino candidate's muon.

### Repro

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

### Symptom

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

### Root cause

#### 1. Q/L matched two bundles to the beam flash

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

#### 2. The flash-group merge keeps one flag donor

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

#### 3. The unmerge demotes everything it splits off

`clus/src/ClusteringUnmergeBundle.cxx:374-377`:

```cpp
for (auto& [gid, part] : splits) {
    part->set_flag(Flags::main_cluster, 0);
    part->set_flag(Flags::associated_cluster);
```

Deliberate and correct for its purpose ("that would make every fragment look
like a bundle main to STM/TGM/FC") — but blanket. It cannot tell a genuine
ex-bundle-main from a 5-point shard.

#### 4. So no cosmic tagger ever examines it

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

#### 5. And then it is adopted into the neutrino

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

### Why it hid

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

### Fix (proposed — four knobs, all default OFF)

Design principle: **record the fact where it is destroyed, re-express it as a
separate flag, let the cosmic taggers see it, act on the verdict.**

#### P1 — preserve the pre-merge main flag

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

*(plan review)* Two notes against the source. First, `merge_clusters` already
has an `orig_main_aname` parameter, and its array name is
`real_cluster_main` — which marks only the rows of the one *representative*
donor member. The new array marks *every* member that carried the flag, and
its name `real_cluster_was_main` differs from the existing one by a single
word. The parameter comment must spell the distinction out, or the next
reader will conflate them the way this review nearly did. Second, the
carried-provenance-pair registry in the same function
(`assoc_cluster_id`/`assoc_cluster_main`) is not a shortcut here: it carries
arrays that exist on the members *before* the merge, while `was_main` must be
created *at* merge time from the live flag. The new parameter is needed.

#### P2 — re-express it as a SEPARATE flag

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

*(plan review)* Mechanically this is cheap: cluster flags are string-named
scalars, not an enum bitmask — `Flags::main_cluster` is an
`inline const std::string` in the `Flags` namespace
(`ClusteringFuncs.h:69`) — so `demoted_main` is one new string constant with
no persisted-format or flag-space concern. The materialisation obligation
above is the whole serialisation story.

#### P3 — let the cosmic taggers see them

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

#### P4 — act on the verdict

`TaggerCheckNeutrino.cxx:386-394`, knob `skip_cosmic_companions` (default
false): when building `other_clusters`, skip a companion **if it is TGM- or
STM-tagged AND its length ≥ `cosmic_companion_min_length`**. Below that length
it stays in regardless of verdict, so a short mis-tagged neutrino daughter can
never be silently dropped and a bad tag on a fragment is bounded.

Same shape as `nu_skip_cosmic_bundle_min_length`, but a different question, so
it wants its own tuning rather than inheriting the 15 cm.

*(plan review)* Where that tuning comes from: the `n_frag` census
(§Verification), **re-run after Part II's A1 lands**, with the companions
classified into (a) demoted ex-bundle-mains (`assoc_cluster_main` today, P1's
array once it exists) and (b) unjoined cathode-crosser halves — a straddle
test of the companion's points against x = 0. Class (b) belongs to Part II's
A1, not to P4: 315497 is in both parts of this doc precisely because it is a
class-(b) companion, and tuning `cosmic_companion_min_length` on the pre-A1
population would tune it on events A1 already fixes.

### Alternative considered and rejected: no new array

`assoc_cluster_main` already separates them on this event (table in §Root cause
2): rcid 19 → {1: 363, 0: 11}, rcid 26 → all 1. So "companion whose blobs are
all `assoc_cluster_main == 1`" picks out cluster 26 today with zero new state.

Rejected on semantics: that array records "was the main of its *isolated
grouping*", not "was the main of a matched *Q/L bundle*". Those coincide for
matched clusters. Whether the flash-time merge can ever group an unmatched
cluster (`t0 = -1e12`) into a beam bundle has **not** been checked — that is
where they would diverge. P1 records the flag that was actually lost, and says
so in its name.

### Verification plan

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

*(plan review)* Two additions to that bar. (1) The census runs **after**
Part II's A1 — A1 moves the unjoined crosser halves out of the companion
population, and the classification in P4 above keeps the two parts from
double-counting each other's events. (2) The ship/no-ship evidence for
P3 + P4 ON is a full **mcp1k verdict census**, not the 48 nueCC events alone.
That is the pr/19 precedent: the owner's decision there was made on 7/1000
verdict changes that only the 1000-event census surfaced. The sweep costs
~21 min at 8 jobs (`run_full1k_nusel.sh`) and the census + attribution
machinery is already committed (`oc19_census_mcp1k.py`; same-binary OFF/ON
rerun pairs for every changed event). Never present a knob for a ship
decision without it.

### Staging

P1 + P2 are inert bookkeeping — they record and re-express a fact and change no
verdict. They can land and be gate-proven ahead of P3 + P4, which is where the
physics moves.

### Open questions

1. Should Q/L have merged the two bundles at all? The merge over-predicts by
   33 % and worsens chi²/ndf; only `ks_dis` improves. Tightening the
   flash-group merge is an upstream alternative to this whole design and is
   **not** analysed here.
2. Whether the 361 MeV shower moved `numu_score` (needs an A/B).
3. Whether `assign_flash_t0_groups` can group unmatched clusters
   (`t0 = -1e12`) — see §Alternative.

## Part II — a cathode-crossing track broken in two

**Status: diagnosis COMPLETE and reproduced at HEAD; fix DESIGN ONLY.** No C++,
no jsonnet and no runner changed for Part II. Two of the four proposed changes
need only an SBND config line (the C++ knobs already exist and ship in PDHD);
the other two are new default-OFF passes in the pattern-recognition chain, which
today has no notion of the cathode plane at all.

Reported by the owner from the `cath13-prod-20260801` Bee scan:

> Some failures in the cathode for pattern recognitions, not connected, why???
> 18259-169824, 18255-406796, 18255-315497 single track broke to two cross
> cathode. Note, there are many cases, the tracks are merged to a cluster from
> the two sides of TPCs.

Both halves of that report are right, and they are **two different failures**.
Two of the three events are never merged into one cluster; the third *is* one
cluster across the two TPCs — the note's case — and the track still breaks.

### Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# Binary provenance.  This doc BUILT NOTHING: every number below comes from the
# installed library as found.  toolkit HEAD aff0ffde, tree clean (no modified
# tracked files), local/lib/libWireCellClus.so mtime Aug 1 19:55.  The property
# Part II actually needs is that the two tracers are in the loaded library:
strings /nfs/data/1/xqian/toolkit-dev/local/lib/libWireCellClus.so \
  | grep -c 'CATHODE_CONNECT_DEBUG\|CC_FEATURE_DUMP'      # -> 2

# (1) the connector's OWN per-pair numbers.  Both tracers are env-gated and
#     already in the shipped source (clustering_cathode_connect.cxx:279, :334).
export CATHODE_CONNECT_DEBUG=1 CC_FEATURE_DUMP=1
TAG=cathdbg1 ENTRIES="809 366 314" ./run_full1k_nusel.sh 3 3   # the 3 events
grep -h '\[cc\]\|\[ccx\]\|\[feat\]' work-mcp1kall-cathdbg1/.log_e{809,366,314}.log

# (2) the 500-event population the relaxations are judged on
TAG=ccfeat300  ./run_full1k_nusel.sh 300 8
TAG=ccfeat300b ENTRIES="$(seq 300 499 | tr '\n' ' ')" ./run_full1k_nusel.sh 200 6

# (3) PR chain (the nusel chain stops at tagger_check_fc and writes no
#     track_fit / mc Bee layer, so the graph symptom needs this arm)
./run_pr_chain_batch.sh work-mcp1kall-ccfeat300 work-ccfeat300pr data
./run_pr_chain_batch.sh work-mcp1kall-cathdbg1  work-cathdbg1pr  data

# (4) analysis (read-only; all four scripts committed beside this doc)
python3 feat_census.py work-mcp1kall-ccfeat300      # connector accept/reject census
python3 stub_census.py work-ccfeat300pr             # cathode-stub segment census
python3 pair_eyeball.py work-mcp1kall-ccfeat300 pairs.json newedges.png
python3 kink_probe.py                               # re-runs segment_search_kink's
                                                    # criteria on evt 169824 offline
```

`work-cath13ql/` and `work-cath13pr/` (the arm the Bee set was built from) give
the same three verdicts; the `cathdbg1` arm above is the re-run at HEAD and is
what every number below is quoted from.

### Symptom — three particle-flow trees

`0-mc.json` from `work-cathdbg1pr/pr_evt<ID>/mabc-pr.zip`:

```
evt 169824   ONE cosmic, THREE particles
  18005  mu-  173 MeV  ( 57.9,-142.8,114.9) -> (  2.4,-115.1,107.8)
    18007  pi+   33 MeV  (  2.4,-115.1,107.8) -> ( -2.0,-116.2,108.6)   <- 4.7 cm, spans x=0
      18008  mu-  290 MeV  ( -2.0,-116.2,108.6) -> (-104.9, -61.3,100.4)

evt 315497   the far half enters as a 706 MeV gamma
  24006  e-   362 MeV  ( -5.4, 196.6,164.3) -> ( -0.5, 199.0,257.2)
  1      gamma 706 MeV ( -5.4, 196.6,164.3) -> (  1.9, 194.3,257.2)
    16000  e-  706 MeV  (  1.9, 194.3,257.2) -> ( 22.6, 191.3,498.5)

evt 406796   the far half is simply absent
  9019   mu-  235 MeV  (-53.9,-154.8, 84.9) -> ( -2.3,-157.3,112.7)     <- stops at the cathode
  9018   proton 122 MeV
```

Energy consequence (`T_kine`, `tracking-pr.root`):

| evt | `kine_reco_Enu` | what the cosmic's cathode crossing contributes |
|---|---|---|
| 169824 | 1125.6 MeV | 173.2 + **33.4** + 290.2 = 496.8 MeV, as three particles |
| 315497 | 1068.5 MeV | **706.4 MeV = 66 %**, entering as a gamma — the Part I pathology |
| 406796 | 387.1 MeV | the far 33 cm half is dropped; a through-goer reads as contained |

### The measurement that splits the three into two classes

Q/L output `mabc-all-apa.zip` `0-clustering-global.json` carries both
`cluster_id` (the post-`examine_bundles` flash bucket) and `real_cluster_id`
(the pre-flash-merge geometric ident, doc 53). Grouping the points by each and
asking which groups have charge on both sides of x = 0:

| evt | halves share a `cluster_id`? | halves share a `real_cluster_id`? | closest approach | class |
|---|---|---|---|---|
| 169824 | yes (18) | **yes (18)** | joined | **B** |
| 406796 | yes (9) | **no — 9 (−x) / 17 (+x)** | 3.65 cm | **A** |
| 315497 | yes (16) | **no — 24 (−x) / 16 (+x)** | 3.28 cm | **A** |

**Sharing a `cluster_id` is not being joined.** `clustering_examine_bundles`
flash-collapses every cluster of one flash group into a single `cluster_id`
while recording the pre-merge ident per blob; `ClusteringUnmergeBundle` in the
PR job splits that back apart. So 406796 and 315497 enter pattern recognition
as **two clusters**, and the fit runs on each separately — which is why the
census in doc pr/12 recorded them as one bundle. (See §"Refinements to doc
pr/12" below.)

### Class A — the two halves are never geometrically joined

`ClusteringCathodeConnect` (`clus/src/clustering_cathode_connect.cxx`) is the
pass whose job this is. Its own tracer, on the two events:

```
evt 406796  [cc] c7<->c14 dis=3.65 p1=(-1.94,-157.0,112.9) p2=(1.31,-155.8,113.9)
                 apa=0/1 driftsep=3.25 len=67.5/32.6 t0=0.699/0.704us dt0=0.004us
            [feat] ttH=18.7 ttP=18.3 ccH=21.9 ccP=25.4
            [ccx] close_fallthrough -> reject

evt 315497  [cc] c13<->c6 dis=3.28 p1=(1.32,194.4,257.2) p2=(-1.01,196.7,257.5)
                 apa=1/0 driftsep=2.33 len=243.9/75.3 t0=0.746/0.742us dt0=0.004us
            [feat] ttH=27.0 ttP=1.7 ccH=85.4 ccP=89.1
            [ccx] close_fallthrough -> reject
```

Every hard gate passes on both: opposite TPCs, both tips inside
`cathode_x_cut = 5 cm`, drift separation inside `drift_cut = 8 cm`, same flash
to 4 ns, closest approach inside `dis_cut = 5 cm` so both are in the CLOSE
regime. What rejects them is the direction logic
(`clustering_cathode_connect.cxx:393-460`):

- the **primary** close test is `tt_hough < angle_cut (10°)` — the local
  charge-weighted Hough directions at the two closest points. 18.7° and 27.0°:
  both fail. At the dense cathode tip the local Hough is noisy — 315497's
  global cluster PCAs agree to **1.7°** while its local Houghs disagree by 27°.
- the **both-long PCA fallback** then requires `tt_pca < 10°` **and**
  `cc_pca < conn_far_cut (30°)`, where `cc_pca` is the angle between the
  tip-to-tip connection vector and the cluster axis. 315497 has `tt_pca = 1.7°`
  — as collinear as a pair can be — and is killed by `cc_pca = 89.1°` alone.
  406796 fails the collinearity term instead (`tt_pca = 18.3°`, a genuinely
  bent track) while its `cc_pca = 25.4°` passes.

![the two class-A pairs](../pics/pr20_cathode_pairs_targets.png)

**`cc_pca` is not a discriminator in the close regime — and the connector's own
accepted population proves it.** Over 500 SBND data events (entries 0-499,
`feat_census.py`), the connector saw 330 cross-APA candidate pairs and accepted
264. Of those, **183 were accepted by the primary Hough test, which never
consults `cc_pca` at all — and their `cc_pca` has median 37.8°, exceeding the
30° bound in 123 of 183 (67 %)**. Two thirds of the crossers the pass already
merges would be rejected if the `cc_pca` term were applied to them. It is
applied only to the pairs that fall through the Hough test, where it kills them.

The reason is geometric, and it is the same fact that produces class B: at a
3 cm tip separation the connection vector is the ~2-3 cm drift-x gap plus the
~1.1 cm transverse cathode offset (doc 14 / doc pr/12 §6, data median
`dyz = 1.08 cm`). It measures the cathode, not the track.

Reject-reason breakdown over those 500 events:

| verdict | count |
|---|---|
| ACCEPT `close_primary` (local Hough collinear) | 183 |
| ACCEPT `far_accept` | 76 |
| ACCEPT `close_bothlong` / `close_shortstub` | 3 / 2 |
| reject `close_fallthrough` (the pathology above) | **18** |
| reject `far_notcollinear` / `far_conn` | 14 / 8 |
| reject `tipx1` / `tipx2` (tip not at the cathode) | 10 / 6 |
| reject `close_shortstub` | 10 |

Of the 18 `close_fallthrough` rejects, 16 have both halves ≥ 25 cm (so the
both-long branch was reached) and **10 of those have `tt_pca < 10°`** — pairs
the connector itself believes are collinear and refuses only on `cc_pca`.

#### What the split costs downstream

`ClusteringUnmergeBundle` splits the flash-bucket back into the bundle main plus
`associated_cluster` parts (`ClusteringUnmergeBundle.cxx:374-377`), so:

- **315497** — the far half is an associated cluster of the candidate's flash
  group, and `NeutrinoShowerClustering`'s
  `shower_clustering_in_other_clusters` adopts it as a **706 MeV gamma**: two
  thirds of `kine_reco_Enu`. This is *exactly* the Part I symptom, reached by a
  different route: there the main flag was lost in the flash-group merge, here
  the halves were never merged at all. In both cases a cosmic fragment that no
  cosmic tagger examined lands inside the neutrino candidate.
- **406796** — the far half is not adopted, so a 100 cm through-going track is
  reconstructed as a 67 cm one that stops at the cathode, and reads as
  contained (`fc = 1`).

### Class B — one cluster, and the graph still breaks the track

169824's halves *are* one cluster (`real_cluster_id 18`, joined by the pr/14
cathode bundle rescue; the connector's own accept log for this event records two
accepted cross-cathode merges, at `ttH = 2.2` and `ttH = 1.9`). The
charge is a single continuous straight line through x = 0. The **fit** is broken:

![the class-B cathode stub](../pics/pr20_cathode_stub_169824.png)

The trajectory carries a 7-point, 4.67 cm segment `18007` whose two endpoints
sit at x = +2.42 and x = −2.04 — the tip-to-tip bridge across the cathode — and
it is a segment, so it is a particle: the `pi+ 33 MeV` between the two muons.

`PatternAlgorithms::examine_structure_3`
(`clus/src/NeutrinoStructureExaminer.cxx:359`, run for the main cluster only via
`NeutrinoPatternBase.cxx:1747`) exists to remove exactly this: at a degree-2
vertex it merges the two segments when `angle_10cm < 18°` and `angle_3cm < 27°`,
with the directions from `segment_cal_dir_3vector(sg, vtx, R)` =
(mean of *that segment's* fit points within R of the vertex) − vertex
(`PRSegmentFunctions.cxx:1130`).

Measured at both of 169824's cathode vertices (offline, same formula):

| quantity | value | gate |
|---|---|---|
| `angle_10cm`, stub `18007` vs neighbour `18005` | **43.36°** | < 18° — fails |
| `angle_10cm`, stub `18007` vs neighbour `18008` | **43.34°** | < 18° — fails |
| the two long neighbours `18005` vs `18008`, skipping the stub, R = 10 cm | **3.31°** | — |
| same, R = 15 / 20 cm | **2.70° / 2.21°** | — |

The stub's own direction is 43° off the track it belongs to, because it *is* the
tip-to-tip vector: 4.47 cm of drift-x and 1.12 cm of y, and the y step has the
**opposite sign** to the parent track's (the track runs to +y as x decreases;
the cathode offset does not). The two long arms either side agree to 2.7°.
Same physical fact as `cc_pca = 89°` in class A, one layer further down.

**Population.** `stub_census.py` looks for a segment whose end-to-end extent is
< 10 cm, whose endpoints straddle x = 0 with both |x| < 6 cm, and counts its
neighbours:

| arm | events | straddling short segments | with exactly one neighbour at each end (a real bridge) |
|---|---|---|---|
| `work-ccfeat300pr` (HEAD, entries 0-299) | 300 | 1 | **1** (evt 286400) |
| `work-mcp1kall-pr11v3` (doc pr/11) | 1000 | 28 | **1** (evt 286400) |
| `work-cathdbg1pr` (HEAD, the 3 events) | 3 | 1 | **1** (evt 169824) |

The other 27 in pr11v3 are isolated fragments with no neighbour at either end
(mostly dense sub-cm blobs), not bridges. Evt 286400 carries the identical
signature: a 6.27 cm / 9-point bridge between a 303.7 cm and a 30.5 cm segment,
neighbour-neighbour kink 4.36°, stub-vs-neighbour 37.71°/36.35°.

So class B is **rare today — roughly 1 event in 300 — and that is a consequence
of class A**: most crossers never become one cluster, so the graph never gets
the chance. doc pr/12 measured the joined population directly: 44 of 45 joined
crossers are fitted as a single segment, i.e. the stub is the ~2 % tail. Fixing
class A moves crossers into that population, so class B should be fixed with it,
not after it.

#### Where the break comes from — the PR chain does not know the cathode exists

The stub is not something the fitter drew badly; it is the product of a
deliberate **track-breaking** step. `PatternAlgorithms::break_segments`
(`NeutrinoPatternBase.cxx:880`, reached from `find_proto_vertex` when
`flag_break_track` is true — hard-coded `true` for the main cluster at
`TaggerCheckNeutrino.cxx:512`) walks each segment asking
`segment_search_kink` (`PRSegmentFunctions.cxx:191-352`) for a point where the
trajectory turns, and splits the segment there.

That kink finder scores each fit point with two angles:

- `refl_angle` — how much the trajectory turns at the point (minimised over six
  half-window sizes, 2 to 12 points), and
- `para_angle` — `|angle-to-drift − 90°|`, i.e. how far the local direction is
  from being **perpendicular to the drift axis**.

All four kink criteria (`PRSegmentFunctions.cxx:336-350`) require
`para_angle > 7.5-15°`. That term is a guard: a trajectory running
perpendicular to drift is isochronous, its apparent wiggles are an imaging
artefact, and it must not be broken on them.

**At the cathode that guard is exactly inverted.** The apparent crossing is
drift-x dominated — the two halves are separated by the drift gap, not by
track — so `para_angle` is wide open there, while the ~1 cm transverse cathode
mismatch supplies the turn. Re-running the finder's own arithmetic offline on
evt 169824's trajectory, reconstituted by concatenating `18005 + 18007 + 18008`
(`kink_probe.py`; a proxy — the pre-break fit is not bit-identical to the
concatenation of the post-break fits):

| i | x (cm) | `refl_angle` | `para_angle` | `sum_angles` | criteria met |
|---|---|---|---|---|---|
| 104 | 3.18 | 4.3 | 60.9 | 17.7 | — |
| 105 | 2.74 | 23.9 | 67.9 | 21.8 | — |
| **106** | **2.42** | **30.8** | **72.8** | **22.6** | **C1, C3** |
| 107 | 2.42 | 28.7 | 72.8 | 22.7 | C3 |
| 108-113 | 1.83 → −2.04 | 13.8 … 26.1 | 71-78 | 6.9-17.1 | — |
| **114** | **−2.04** | **27.4** | **71.1** | **17.1** | **C3** |
| 115 | −2.57 | 4.9 | 66.2 | 17.1 | — |

Over the whole 312-point, 163 cm trajectory the **only** indices that meet any
kink criterion are `3, 4` — the genuine junction at x = 56.7 cm where the muon
leaves the π⁺ — and `106, 107, 114`: the two cathode tips. `para_angle` sits at
61-78° through the crossing, so the isochronous guard never engages, and
`refl_angle` clears its thresholds by a hair (30.8 against C1's 30, 27.4 against
C3's 27).

And the PR chain has no way to know better. `grep -c cathode` over the
pattern-recognition sources:

| file | occurrences |
|---|---|
| `TaggerCheckSTM.cxx` | 43 |
| `TaggerCheckTGM.cxx` | 11 |
| `TaggerCheckNeutrino.cxx` / `TaggerCheckFC.cxx` | 1 / 1 |
| `NeutrinoStructureExaminer.cxx`, `TrackFitting*.cxx`, `PRSegment*.cxx`, `NeutrinoPatternBase.cxx` | **0** |

The **taggers** know where the cathode is; the graph builder, the kink finder
and the trajectory fitter do not. Nor can they ask: `IDetectorVolumes` exposes
`contained_by`, `inner_bounds` and `face_dirx` but has no cathode accessor — the
plane is only implicit, as the gap between two faces' `inner_bounds` (SBND:
apa0 sensitive volume ends at x = −0.45 cm, apa1 starts at +0.45 cm). The PR
sources that do use `dv` use it for dead-region and containment tests only.

How big is the mismatch it is breaking on? Extrapolating the +x half's 20 cm
near-cathode arm across the gap to the −x half's tip misses it by **3.74 cm**
transversely (3.71 cm computing it the other way). Not all of that is the
physical cathode offset: the extrapolation runs over the full 4.47 cm *apparent*
drift-x gap, of which only the ~1.5 cm cathode thickness is real travel, so the
genuine transverse mismatch is roughly 2 cm — the upper tail of doc pr/12 §6's
1.08 cm median, not an outlier.

### Fix

Four changes, plus one that belongs to calibration rather than to this doc.
**A1 and A2 are SBND config lines only — the C++ already exists and PDHD
already ships A1.** B0 and B1 are new C++, both default OFF; B0 is the one to
prefer, and B1 is its backstop.

#### A1 — enable `tip_touch_cut` for SBND (config only)

`clustering_cathode_connect.cxx:427,445` already implements "when the tips touch,
drop the `cc_pca` term and accept on PCA collinearity alone", gated on
`tip_touch_cut` (C++ default 0 = OFF). PDHD ships `tip_touch_cut = 3 cm`
(`cfg/pgrapher/experiment/pdhd/clus.jsonnet:618`); SBND never enabled it.

```jsonnet
// cfg/pgrapher/experiment/sbnd/clus.jsonnet:428
cm.cathode_connect(cathode_x_cut=5*wc.cm, drift_cut=8*wc.cm,
                   min_length_short=2*wc.cm, short_dir_len=25*wc.cm,
                   conn_short_cut=30.0,
                   tip_touch_cut=4*wc.cm,          // NEW (C++ default 0 = OFF)
                   flash_t0_window=800*wc.ns)
```

Analytical accept-log delta over the same 500 events (the shipped binary's own
per-pair numbers, predicate re-evaluated offline — no rebuild):

| `tip_touch_cut` | new edges / 500 events | 315497 recovered? |
|---|---|---|
| 2 cm | 0 | no |
| 3 cm (the PDHD value) | 4 | no |
| **4 cm** | **10** | **yes** |
| 5 cm | 10 | yes |

4 and 5 cm give the identical edge set, so 4 cm is not sitting on a cliff.

To be even-handed with A2 below: **4 cm is also a post-hoc value** — it is the
smallest of the four tried that recovers 315497, and the PDHD-shipped 3 cm does
not. The defence is the no-cliff evidence (4 and 5 cm are the same set, and the
next edge only appears when `dis_cut = 5 cm` itself would have to move) plus the
fact that the accept-log delta is enumerated rather than counted. It is a weaker
objection than A2's, not a different kind of objection.
Every one of the 10 new edges, eyeballed (`pair_eyeball.py`, figure below):

![the new edges](../pics/pr20_cathode_newedges.png)

| evt | rcids | dis | `tt_pca` | `tt_hough` | `cc_pca` | lengths | verdict by eye |
|---|---|---|---|---|---|---|---|
| 67394 | 20/24 | 2.23 | 6.3 | 10.8 | 56.9 | 71 / 53 | one straight track |
| 392354 | 15/21 | 2.36 | 2.0 | 29.6 | 79.6 | 388 / 84 | **ambiguous** — the two meet at ~70° locally |
| 71266 | 24/27 | 2.40 | 3.1 | 17.7 | 68.2 | 291 / 189 | **ambiguous** — far half is gap-fragmented |
| 71882 | 13/20 | 2.43 | 4.9 | 50.7 | 54.5 | 329 / 110 | **ambiguous** — partner is a diffuse EM region |
| 292450 | 21/27 | 3.06 | 1.7 | 13.9 | 57.3 | 165 / 107 | one straight track |
| 56519 | 14/19 | 3.20 | 2.4 | 11.0 | 38.6 | 65 / 62 | one straight track |
| **315497** | 24/16 | 3.28 | 1.7 | 27.0 | 89.1 | 132 / 244 | one object (the target) |
| 406752 | 18/22 | 3.52 | 2.3 | 11.6 | 51.8 | 416 / 40 | one straight track |
| 59025 | 21/20 | 3.59 | 3.5 | 16.1 | 33.8 | 221 / 184 | one straight track |
| 57575 | 15/27 | 3.60 | 1.4 | 13.2 | 35.4 | 197 / 204 | one straight track |

7 of 10 are unambiguous single tracks; **3 are ambiguous and are named here for
the owner to look at in Bee before this ships**. None is an obvious pair of
distinct parallel cosmics. The rate is 1 new merge per 50 events.

*The purity honesty note.* `tip_touch_cut` removes the `cc_pca` guard for pairs
inside the cut. What still has to hold for a merge: opposite TPC, both tips
within 5 cm of the cathode, drift depths within 8 cm, same flash within 800 ns,
closest approach under 4 cm, and cluster PCAs collinear within 10°. The
measurement above is what says the removed term was not carrying purity: the
already-accepted population violates it 67 % of the time.

#### A2 — `crosser_pca_angle` for the bent crosser (config only) — **owner decision**

406796 is a single, gently curving track (see the figure; the near-cathode arms
agree to 16.9° and the connection continues both, `cc_pca = 25.4°`) whose two
halves' global PCA axes differ by 18.3°. `tip_touch_cut` alone does not recover
it — the `tt_pca < 10°` bound does the rejecting. `crosser_pca_angle`
(`clustering_cathode_connect.cxx:442`, C++ default 0 ⇒ bound stays `angle_cut`)
raises that bound and is already offered by the common helper; PDVD's census
sets it from QL-pin truth, where real crossers reach `tt_pca ≈ 18` at p90 and
coincidences sit at p50 ≈ 24 (`protodunevd/clus.jsonnet:636-642`).

| `crosser_pca_angle` (with `tip_touch_cut = 4`) | new edges / 500 | 406796? | what else comes in |
|---|---|---|---|
| off (= 10°) | 10 | no | — |
| 15° | 12 | no | 315849 (`tt_pca` 10.3), 399702 (11.3) |
| **20°** | **13** | **yes** | + 406796 (18.3) |
| 25° | 15 | yes | + 405432 (23.9), 399702 c11 (23.7) — past PDVD's coincidence p50 |

**This is escalation rule 7 territory and is not decided here.** 406796 sits at
18.3°, right at PDVD's real-crosser p90, and a 20° bound is a value that admits
it. What argues for it: it adds only 3 edges in 500 events, two of which
(315849, 406796) look like single tracks by eye, and 25° is where the PDVD
truth study says coincidences begin. What argues against it: it is a bound
chosen after seeing the event it recovers. Recommend the owner look at the three
in Bee and decide.

#### B0 — do not create the break (cathode kink veto, new C++, default OFF)

The cheapest place to fix this is where the vertex is invented. In
`segment_search_kink`, **skip a candidate fit point whose |x − cathode_x| is
below `cathode_kink_xcut`** (C++ default 0 ⇒ no point is ever skipped ⇒
byte-identical; proposed SBND 5 cm). The loop already scans forward and takes
the first qualifying index, so skipping cathode indices lets it keep looking and
still find a genuine kink further along — it suppresses the cathode break
without suppressing the search.

Threading: `segment_search_kink` is a free function with defaulted trailing
arguments; add `double cathode_x = 0, double cathode_kink_xcut = 0` in the same
style, pass them from `break_segments` out of two new
`PatternAlgorithms` members, and set those in `TaggerCheckNeutrino::visit()`
exactly as `m_mip_dqdx_median` is set today
(`TaggerCheckNeutrino.cxx:73, 201, 459` — read the key, round-trip it in
`default_configuration()`, assign to `pattern_algos.m_*`).

*(plan review — two facts checked against source.)* `segment_search_kink`
has exactly **two** call sites in the tree, both inside `break_segments`
(`NeutrinoPatternBase.cxx:963`, `:1025`) — the veto cannot leak into STM or
any other consumer, and the threading above covers every caller. And the
skip must gate **only the four accept tests at index i**, leaving
`refl_angles` / `para_angles` and the windowed `sum_angles` untouched: the
loop breaks at the first qualifying index, so a vetoed cathode index simply
lets the scan continue, and a genuine kink elsewhere on the segment sees
arithmetic identical to today's.

Why prefer this to B1:

- **Nothing has to be undone.** No vertex, no stub segment, no splice, no refit
  — and no risk that the splice picks the wrong wcpts orientation.
- **It is strictly wider.** doc pr/12 §7 measured that 13 of 44 spanned
  crossers acquire a graph vertex within 3 cm of x = 0, and that the 0-3 cm band
  is the single most populated bin of the nearest-vertex distribution. Only the
  handful whose two vertices bracket a whole short segment produce the class-B
  stub that B1 can absorb; B0 removes the spurious vertex in all of them.
- **Its cut is on position, not on angle**, so it does not need a threshold
  tuned against the events it recovers.

What it gives up: a real kink that genuinely happens within 5 cm of the cathode
is no longer found. At SBND's cosmic rate that is a rare loss, and the trade is
explicit and measurable (gate B0-3 below).

`cathode_x` is a knob rather than a lookup because `IDetectorVolumes` has no
cathode accessor. Deriving it from the two faces' `inner_bounds` would be more
general and is worth doing if this pattern recurs — recorded as an open
question, not built here.

#### B1 — cathode-aware stub absorption (new C++, default OFF)

A pass placed immediately after `examine_structure_3`
(`NeutrinoPatternBase.cxx:1747`), or a branch inside it. *(plan review:
prefer the standalone pass — a branch inside `examine_structure_3` edits a
function every detector runs, and the knob-off gate then has to prove that
edit inert; a separate pass keeps the production function byte-identical at
the source level, the usual fork-not-modify shape.)* For every segment S:

1. S's end-to-end extent < `cathode_stub_max_len` (C++ default **0 ⇒ pass OFF**;
   proposed SBND 8 cm), and
2. S's two endpoints lie on opposite sides of `cathode_x` with both
   |x − cathode_x| < `cathode_stub_xcut` (SBND 4 cm), and
3. both of S's graph vertices have degree exactly 2 — S bridges two segments and
   nothing else. **This condition is proposed, not measured**: `stub_census.py`
   works from the Bee `track_fit` layer, so what it can test is the geometric
   proxy "exactly one other segment has a fit point near each endpoint". On
   169824 that proxy is robust — at tolerances of 0.5, 1.0 and 2.0 cm the only
   segments touching the two vertices are `18005`/`18007` and `18007`/`18008`
   (`0-vertices-global.json` confirms those two vertices exist at exactly those
   points and are the event's only vertices within 6 cm of the cathode) — but
   `boost::degree(vd, graph)` is what the implementation must actually test.

then compute the collinearity of the two **neighbours**, skipping S entirely:
`180° − angle(dir(nb1, v1, R), dir(nb2, v2, R))` with the same
`segment_cal_dir_3vector` and `R = cathode_stub_radius` (SBND 15 cm), and if it
is below `cathode_stub_angle` (SBND 15°) splice `nb1 + S + nb2` into one segment
— reusing `examine_structure_3`'s existing wcpts-orientation logic
(`NeutrinoStructureExaminer.cxx:428-470`) — drop the two vertices, and refit.

Measured margins on the two known cases: 169824 neighbour kink 2.70° at R = 15,
286400 2.92° at R = 15 — a 15° bound has a factor ~5 of headroom, and the test
never touches the stub's own direction, which is the quantity the cathode
corrupts.

Why not simply loosen `examine_structure_3`'s 18°/27° cuts: those are prototype
values that apply everywhere in the detector and would merge real kinks; the
cathode exemption is the narrow statement that is actually true.

B1 is worth having even with B0 in place: it catches a stub that arises some
other way (a bridge built by `find_other_segments` rather than by a kink break,
which this doc has not excluded). But B0 is the primary and B1 the backstop, not
the other way round.

Note the ordering against class A: the class-B repair alone touches ~1 event in
300 today. **A1 without B0 will raise the class-B rate**, because the ~10 newly
joined crossers per 500 events enter the population where the break can form
(~2 % of joined crossers become stubs, doc pr/12's 1 in 45; the spurious-vertex
rate is much higher, 13 in 44). Land B0 with, or before, A1.

#### B2 — remove the mismatch itself (calibration; not this doc's to make)

The physically right answer is that the two halves should line up. The cathode
crossing offset is measured (doc 14 / `project_cathode_crossing_offset`: ~1.5 cm
in drift-x and ~1 cm transverse in data, with a field-cage distortion map), and
applying it would shrink the turn the kink finder sees.

It is recorded here as the honest long-term direction and **not proposed as the
fix**, for two reasons. First, the margins say it would be fragile: `refl_angle`
at the two cathode tips is 30.8° and 27.4° against thresholds of 30° and 27°, so
a correction would have to remove nearly all of the ~2 cm transverse mismatch to
push the break below threshold, and any residual leaves it firing. Second, it
does nothing for class A: with the halves perfectly aligned the tip-to-tip
vector is still almost pure drift-x, so `cc_pca` stays near 90° for any track
that is not itself along the drift axis. A calibration improvement is welcome
and orthogonal; it is not a substitute for A1 or B0.

#### Considered and not proposed

- **Widening `flash_t0_window` again** — irrelevant here; all three pairs are
  same-flash to 4 ns.
- **Dropping `cc_pca` unconditionally in the close regime** — that is A1 with
  `tip_touch_cut = dis_cut`, i.e. 5 cm, which gives the same 10 edges on this
  sample but leaves no distance handle if a later sample shows over-merging.
- **Requiring the local Hough as well as the PCA inside the tip-touch branch**
  (`tt_pca < 10 && tt_hough < 20`) — it would exclude the two worst ambiguous
  pairs (392354 at 29.6°, 71882 at 50.7°), but it also excludes **315497**
  (27.0°), the event this is for. Rejected on that basis; recorded here because
  it is the first thing to try if the ambiguous merges turn out to be wrong.

### Validation plan

A1/A2 change SBND merging, so they are **NOT bit-identical and need
revalidation** — they are behaviour changes delivered as config, not knob-off
claims. B1 is new C++ and gets the full knob-off sweep.

**Scope proof, first, for A1/A2.** `git diff --stat` must show
`cfg/pgrapher/experiment/sbnd/clus.jsonnet` and nothing else. No
`cfg/pgrapher/common/` file moves ⇒ PDHD, PDVD, uBooNE cannot move ⇒ the
`abtest` and `qlport` gates are **not required** for A1/A2, and this must be
stated explicitly in the report rather than silently skipped.

| # | gate | pass criterion |
|---|---|---|
| A-1 | compiled-config proof: `wcsonnet` the SBND Q/L job, `grep tip_touch_cut` | key present with the new value; nothing else in the compiled JSON moves |
| A-2 | accept-log delta, 500 events, `CATHODE_CONNECT_DEBUG=1` before/after | the new-edge set equals the 10 (or 13) enumerated above, **enumerated event by event, not counted** |
| A-3 | collateral test: `hash_archive.py` on `mabc-all-apa.zip` for 20 events with **no** new edge | member-content identical — the relaxation touched nothing else |
| A-4 | `nusel-table.tsv` diff over the 500-event arm, before vs after | every changed row explained by a new edge; TGM/STM/FC label flips listed |
| A-5 | 48 nueCC events (`work-nuecc48-*`), full chain | **hard constraint: zero beam-label changes** (the pr/18 and pr/20 Part I bar) |
| A-6 | the 13 `cath13-prod` events re-run and re-uploaded as a **fresh** Bee set | 315497 (and, with A2, 406796) draw as one object; the other 11 unchanged |
| A-7 | owner eyeball of the 3 ambiguous merges (392354, 71266, 71882) and, if A2, of 315849 / 399702 | owner sign-off before default-ON |
| A-8 | *(plan review)* mcp1k nusel sweep + verdict census (`oc19_census_mcp1k.py` pattern), before vs after | every verdict change explained by a new edge, enumerated; the count is the owner's ship-decision evidence (pr/19 precedent: decided on 7/1000) |
| B0-0 | *(plan review — promoted from open question 3)* scratch build, evt 169824 only: flip the hard-coded `flag_break_track` (`TaggerCheckNeutrino.cxx:512`) to false | the stub and both cathode vertices disappear — the by-construction proof, run **before** implementing B0 |
| B0-1 | knob OFF byte-identical (B0 edits `PRSegmentFunctions.cxx`, which uBooNE links): `qlport/scripts/ab_check.sh` both gates, `abtest/ab_compare.sh` pdhd + pdvd, `hash_archive.py` on SBND `mabc-pr.zip` ×6 | all PASS with `cathode_kink_xcut` unset |
| B0-2 | compiled-config proof: `wcsonnet` the SBND PR job, `grep cathode_kink_xcut` | key present when on, absent when off |
| B0-3 | knob ON, 300-event PR arm: count graph vertices within 3 cm of x = 0 (the doc pr/12 §7 statistic) and segment counts elsewhere | the cathode vertices go away; on events with **no** cathode-band vertex at baseline, vertices identical; on events where a break was suppressed, off-cathode changes are allowed but must be **traceable to the affected track** *(plan review: suppressing a break re-runs the search on the longer surviving segment, so its later break positions can legitimately shift — the original "no vertex away from the cathode moves" criterion would fail on expected behaviour)* |
| B0-4 | knob ON, evt 169824 | segments `18005`/`18007`/`18008` become one; the PF tree has one muon; `kine_reco_Enu` recomputed and reported |
| B0-5 | knob ON, 48 nueCC events | **hard constraint: zero beam-label changes**; neutrino vertices unmoved |
| B0-6 | *(plan review)* determinism: knob ON, 3 events × 3 runs under `setarch x86_64 -R` | identical archives run-to-run — house bar for a pass that changes vertex selection |
| B0-7 | *(plan review)* knob ON, mcp1k nusel sweep + verdict census vs baseline | changes enumerated and attributed (same-binary OFF/ON rerun pairs), same bar as A-8 |
| B-1 | `./build/clus/wcdoctest-clus` | passes |
| B-2 | knob OFF byte-identical: `abtest/ab_compare.sh` (pdhd + pdvd, `events.txt`), `qlport/scripts/ab_check.sh` (uboone, both gates), `hash_archive.py` on SBND `mabc-pr.zip` ×6 | all PASS — B1 touches `clus/`, which every detector links |
| B-3 | knob ON, evts 169824 + 286400 | the stub segment is gone; the PF tree has one muon where it had three; `kine_reco_Enu` recomputed and reported |
| B-4 | knob ON, 300-event PR arm, `stub_census.py` + `nusel-table.tsv` diff | no segment count changes away from the cathode; label changes enumerated |
| B-5 | A1 + B1 together, 500-event arm | the newly joined crossers are fitted as single segments; the class-B rate does not rise |

Labels to report: `abtest/snap/{pre,post}_cathstub/`, and the analysis arms
`work-mcp1kall-ccfeat300{,b}`, `work-ccfeat300pr`, `work-cathdbg1pr` (all
created fresh for this doc; no existing label or `work/` tree was written into).

### Refinements to doc pr/12

1. **evt 406796 is not "the one real PR-side case".** doc pr/12 §6 reads the
   two halves as "the *same* in-beam bundle 9, i.e. `cathode_connect` did join
   the halves", and concludes the trajectory fit failed on a cluster it owned.
   Same *bundle* is not joined: they are `real_cluster_id` 9 and 17, sharing a
   `cluster_id` only through the flash-group merge, and `ClusteringUnmergeBundle`
   hands the PR chain two clusters. It is a class-A clustering case, and the fit
   behaved correctly on its input. That revises a shipped headline: on this
   sample there is **no** case of the PR fit failing to cross a cluster it owned.
2. **evt 315497** — pr/12 lists it under the "off-axis remainder" (far-side
   charge at d⊥ 5.0 cm, 0 points on the fit extrapolation). Both statements
   hold: pr/12 asked whether far-side charge lies along the *fitted*
   extrapolation, this doc asks whether the two clusters approach each other.
   They approach to 3.28 cm; the fit does not reach there because the far half
   is a different cluster.
3. **evt 169824** — the pr/12-style census calls it `spanned` with
   `seg_neg == seg_pos == 18007`, and the owner calls it broken. Both are
   correct measurements of different things: the nearest cross-cathode *fit
   point* pair lies inside the stub, so it has one `sub_cluster_id`, while Bee
   colours the particle flow by `real_cluster_id = cluster*1000 + segment`, so
   18008 / 18007 / 18005 draw as three particles. This is pr/12 §7's "13 of 44
   spanned candidates have a graph vertex within 3 cm of x = 0" seen from the
   display side.

### Open questions

1. Are the three ambiguous A1 merges (392354, 71266, 71882) one particle or
   two? They share a shape — a clean track meeting a fragmented or diffuse
   partner — that neither `tt_pca` nor `cc_pca` separates.
2. Is 315497 one EM shower crossing the cathode, or a track plus a shower? Both
   halves are broad in y (see the figure); the class-A verdict (one object) does
   not depend on the answer, but the 706 MeV energy assignment does.
3. **Confirm the origin by construction.** `flag_break_track` is hard-coded
   `true` at `TaggerCheckNeutrino.cxx:512`; flipping it to `false` for evt
   169824 alone should make the stub and both cathode vertices disappear. That
   is the one-line experiment that turns `kink_probe.py`'s offline
   re-evaluation into a direct proof, and it was not run here (it needs a C++
   edit in a shared tree). *(plan review: promoted to gate B0-0 — it runs
   before B0 is implemented, not after.)*
4. Should `IDetectorVolumes` grow a cathode accessor, so the PR chain can ask
   instead of being told? The plane is already derivable from two faces'
   `inner_bounds`; B0 and `ClusteringCathodeConnect` would both stop carrying a
   `cathode_x` knob.
5. Should the B1 stub absorption also run on non-main clusters?
   `examine_structure_3` is gated on `is_main_cluster`
   (`NeutrinoPatternBase.cxx:1745`), so a cosmic that is not the candidate keeps
   its stub — harmless for the PF display, but it changes segment counts that
   the cosmic taggers read.
6. Why is 169824's 285 cm through-going crosser tagged `fc = 1` (contained) with
   `tgm = 0`? It is the selected nu candidate at `numu_score 3.11`. Out of scope
   here, but it is the same family as Part I: a cosmic that no tagger caught.

## Execution order (plan review, 2026-08-02)

Each change above carries its own gates; what the doc had not fixed is the
order across the two parts. The review settles it:

0. **Owner decisions first, no build.** Everything class A needs from the
   owner can be decided from data already on disk: build **one fresh Bee set**
   holding the 3 ambiguous A1 merges (392354, 71266, 71882 — gate A-7), the
   A2 candidates (406796, 315849, 399702), and two or three of the clean A1
   merges for contrast; upload; get the A1 sign-off and the A2 yes/no before
   any code or config moves. In the same round run **B0-0** (the
   `flag_break_track = false` scratch experiment on 169824) — it is the
   by-construction proof B0's design rests on, and it costs one scratch build.
1. **B0 + A1 land in one round** (the ordering argument in §Fix: A1 without
   B0 raises the class-B rate). A2 rides along iff step 0 approved it.
   Ship evidence: gates A-8 / B0-7 (the mcp1k census) plus the nueCC48
   zero-change constraints (A-5 / B0-5).
2. **Part I P1 + P2 in parallel** — inert bookkeeping, gate-proven
   independently. Theirs is the expensive full-detector sweep, because
   `ClusteringFuncs.cxx` is linked by every detector.
3. **The `n_frag` census, post-A1** (Part I §Verification as amended):
   classify companions into demoted ex-mains vs unjoined crosser halves, set
   `cosmic_companion_min_length` from the class-(a) distribution.
4. **P3 + P4**, with the same mcp1k-census bar before any default flips.

The pr/19 lesson is budgeted throughout: a 1000-event nusel sweep is ~21 min
at 8 jobs, and its verdict census is the artifact the owner actually decides
on.

## Related

- doc pr/3 §8 — `nu_skip_cosmic_bundle` (the per-main version of this veto)
- doc pr/12 — cathode-crossing neutrino PR; evt 59003 is one of its 13 cases,
  and Part II §"Refinements" revises two of its §6-7 conclusions
- doc pr/14 / pr/16 / pr/17 — cathode bundle rescue and the min-length guard
- doc 45 — unmerge bundle → main + associated
- doc 53 — `real_cluster_id` epochs
- doc 14 / doc pr/12 §6 — the ~1.1 cm transverse cathode offset in data
- `clus/docs/cathode-crossing-clustering.md` — the `cathode_connect` pass
  and every earlier tuning round
