# doc sbnd_xin/120 — consolidate the detector configuration into `toolkit/cfg`, and the two-chain drift it exposed

**Scope.** Round A moves the PDHD and PDVD standalone jobs out of the working repo and into
`cfg/pgrapher/experiment/{pdhd,protodunevd}/`, byte-identically and with no value changed. Round B
addresses what scoping round A turned up: **SBND's two production chains are not running the same PR
operating point**, and have not been since 2026-09-14.

**Status.** Round A is **committed** (toolkit `c76b8cbe`, wcp `17381073`). Round B is written and
gated in the working tree — 24 of the 25 divergences closed, the local chain proven unmoved — but
**not committed**: three edits to the LArSoft entry point and its mirror were refused by the
session's permission classifier, and the 25th key is the physics one that needs an explicit go
anyway. See sec 4.4.

**Owner's ask, 2026-09-21**: *"For PDHD, PDVD, SBND, for the default configuration, I wonder if
things related to the configuration can be merged from the work directory ./pdvd ./pdhd ./sbnd_xin
to the ./toolkit/cfg/pgrapher/experiment part? After that, I would like to merge the
apply-pointcloud to the master branch."* Scope chosen the same day: relocate **and** consolidate the
SBND operating point; all `wct-*` entry points; master after the config work.

## 0. Repro

```bash
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# sec 2 -- the two-chain divergence (this is the finding, and the new gate)
python3 $SX/scripts/cfg/two_chain_gate.py --verbose

# sec 3 -- round A, the relocation gate.  Run before the move and after; the
# two output dirs must be byte-identical.
bash $SX/scripts/d120/compile_movers.sh ~/tmp/d120-before      # before
python3 $SX/scripts/d120/do_move.py --apply                    # the move
bash $SX/scripts/d120/compile_movers.sh ~/tmp/d120-after       # after
diff -rq ~/tmp/d120-before ~/tmp/d120-after

# the standing tripwire, before and after
python3 $SX/scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-21c
```

Toolkit at `253f1845`; no C++ changed in this round, so no rebuild and no freshness proof is needed
(M1 does not apply — nothing here is compiled).

---

## 1. What lived where, and why it mattered

SBND promoted its standalone jobs in-tree on 2026-07-27 (`docs/64_cfg-sync.md`) and left a 1-line
re-export shim in the work dir, so `sbnd_xin/wct-pr-perevt.jsonnet` is 1 209 bytes of header plus
one `import`. **PDHD and PDVD never did.** Before this round:

| | the detector's per-event PR job | where its operating point lived |
|---|---|---|
| SBND | `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` | in-tree, 535 TLA defaults, 107 `SBND PRODUCTION ON` comments |
| PDHD | `wcp-porting-img/pdhd/wct-pr-perevt.jsonnet` | **working repo**, 318 KB, 539 TLA defaults |
| PDVD | `wcp-porting-img/pdvd/wct-pr-perevt.jsonnet` | **working repo**, 342 KB, 545 TLA defaults |

The PDHD/PDVD runners pass nothing but per-event values — `pdhd/run_pr_evt.sh:233-242` names
`input`, `output_dir`, `run`, `subrun`, `event`, `trigger_offset_us`, `readout_window_ticks`,
`pipeline_names` and an escape-hatch variable, and PDVD adds only its two drift speeds and two
trigger offsets, all read from the Q/L job's own `.tlas` sidecar. So **every PDHD and PDVD
production knob value was already a jsonnet default**; it was simply a jsonnet default in the wrong
repository. Relocating it moves no value at all — which is what makes round A gradeable as an
identity.

---

## 2. The finding: SBND's two chains are not on the same operating point

### 2.1 What was measured

`scripts/cfg/two_chain_gate.py` compiles both production chains and compares their shared
components key by key:

* the **local 2-step chain** — `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` at the 15-stage
  production pipeline, which is what `run_pr_chain_batch.sh` runs;
* the **LArSoft 1-step chain** — `sbnd/wcls-img-clus-matching-xin.jsonnet` at
  `pr_operating_point=sync`, which is what `lar -c wcls-img-clus-matching-xin.fcl` runs.

`docs/120_figs/120_two_chain_pre.txt`. **25 keys are present in the local chain and absent from the
LArSoft chain. None is present-but-different; every one is an omission.**

| class | n | keys |
|---|--:|---|
| Bee particle-flow tree | 17 | `pf_conn4_near_candidate`, `pf_direct_when_touching`, `pf_drop_stray_satellites`, `pf_orphan_audit_only`, `pf_orphan_confident_track`, `pf_orphan_guard_freed`, `pf_orphan_near_cross_cluster`, `pf_orphan_track_parentage`, `pf_pdg_name_prototype_fallback`, `pf_pi0_node_per_id`, `pf_pseudo_gap_from_main`, `pf_shower_parent_precedence`, `pf_shower_vertex_barrier`, `pf_track_bridged_clusters`, `pf_track_main_cluster_only`, `pf_track_owns_loose_vertex`, `pf_unique_node_ids` |
| Bee display | 1 | `pseudo_shower_track_paint` |
| `tracking-pr.root` record | 6 | `fix_cluster_flags`, `provenance`, `rec_charge_provenance`, `nu_provenance` ×3 (`SbndPrMagnifyTrackingVisitor:pr`, `TaggerCheckNeutrino:pr`, `UbooneTaggerOutputVisitor:pr`) |
| **physics** | **1** | **`nu_bundle_flash_group`** |

`nu_bundle_flash_group` merges the two neutrino bundles one physical flash makes when their charge
touches (doc 109 rev 4 sec 9.7, owner-directed, commit `b93673ef`, 2026-09-18). Its C++ default is
`false` (`TaggerCheckNeutrino.h:813`), and the key is absent from the LArSoft compile — so **the
LArSoft chain is not merging them**, and the two chains can return different neutrino candidates on
an event where that merge fires.

### 2.2 The mechanism is omission, not staleness — and that distinction decides the fix

The natural theory is that `sbnd/pr-operating-point.jsonnet` — a hand-regenerated mirror of the PR
job's TLA defaults, pinned in its own header to toolkit `0ad642235` (2026-09-08) — has gone stale.
**It has not.** Two measurements:

1. Intersect the mirror's 242 knob names against every TLA default that changed in
   `wct-pr-perevt.jsonnet` between `0ad64223` and HEAD: **empty**.
2. Compare the mirror's value for each of its 242 knobs against that knob's current default in the
   job: **12 differences, all formatting** — `2.0` vs `2`, `15 * wc.cm` vs `150`.

The mirror is exact for everything it carries. The 25 divergent keys are ones it **never carried**:
they were added to the job after the last hand-run (`c203b400` 2026-09-14, `12798c4f` 2026-09-17,
`b93673ef` 2026-09-18, and the `pf_*` family earlier), and nothing re-ran the generator to pick up
*new* knobs. The generator cannot be re-run here in any case: `gen-pr-operating-point.py`,
`compile-both.sh` and `resync-operating-point.sh` **exist in neither repository** — doc 118 open
item 5 said so, and a `find` over both trees plus `git log --diff-filter=D` confirms they were never
committed.

This matters more than it looks. A currency check — "is the mirror up to date for the knobs it
carries?" — **passes today and would have passed every day since 09-08.** So would
`prod_cfg_gate`'s PASS 26/26 (`docs/120_figs/120_gate_pre.txt`), which proves the mirror's bytes are
unchanged, never that they are complete. Only a **chain-to-chain** comparison can see an omission,
because an omitted knob leaves no trace anywhere except in the other chain's compiled config. That
is why sec 4's gate is `two_chain_gate.py` and not a smarter mirror check.

`docs/8-build-and-run-both-chains.md` sec 7.1 records the same failure family once before: a
`pr-operating-point.jsonnet` regenerated against the 09-05 tree and never re-run cost **18 of 19
events**, with `Enu` off by up to 70 %, and was found only by diffing the compiled PR node.

### 2.3 What is NOT divergence

Four differences are structural and are declared, with the line that creates each, in the gate's
`STRUCTURAL` table rather than silently dropped:

* **the shared Bee zip.** The LArSoft chain passes `bee_sink`, so `save_deadarea` goes false
  (`clus.jsonnet:2825` — only one node may write dead area into a shared zip), the clustering layer
  is renamed `clustering-pr` (`:2833`), `bee_pf` gains `merge_metadata_key` / `merge_node_text` /
  `emit_empty` (`:2966-2972`) and dumps on the last stage instead of the tagger (`:2961`).
* **per-event identity** (`runNo`/`subRunNo`/`eventNo`, `rse_from_*`): LArSoft reads the art event.
* **output directory**: compared by basename, not ignored.
* **`dl_weights`**: `compile_prjob_cfg.sh` pins it empty because M4 keeps the SCN vertex out of byte
  gates. `two_chain_gate.py` deliberately does **not** pin it, so both sides carry the production
  weights; comparing an empty vertex source against a real one would have been meaningless. With the
  pin removed the key agrees, which is the check that the caveat was a caveat and not a defect.

---

## 3. Round A — the relocation

### 3.1 What moved: 14 files

`wcp-porting-img/pdhd/` → `cfg/pgrapher/experiment/pdhd/` (8): `wct-pr-perevt.jsonnet`,
`wct-clustering.jsonnet`, `wct-img-all.jsonnet`, `wct-sp-to-magnify.jsonnet`,
`wct-light-reco.jsonnet`, `wct-light-allpd-reco.jsonnet`, `wct-light-fullstream-reco.jsonnet`,
`wct-light-convert.jsonnet`.

`wcp-porting-img/pdvd/` → `cfg/pgrapher/experiment/protodunevd/` (6): `wct-pr-perevt.jsonnet`,
`wct-clustering.jsonnet`, `wct-nf-sp.jsonnet`, `wct-nf-sp-dnnroi.jsonnet`,
`wct-sp-to-magnify.jsonnet`, `wct-light-reco.jsonnet`.

Each work-dir path keeps a 1-line re-export shim in the doc-64 shape, so **no runner was edited**:
`run_pr_evt.sh` and `run_clus_evt.sh` `cd` into the work dir and name the file bare, and top-level
arguments bind through the `import` unchanged.

### 3.2 Three files held back, each for a reason found before the move

The plan called for 17 + 1. Three are **not** relocatable as-is, and forcing them would have been a
silent change:

1. **`pdhd/wct-nf-sp.jsonnet`** and **`pdhd/wct-nf-sp-dnnroi.jsonnet`** — the only two movers with an
   unqualified import, `import 'pdhd-coh-groups-preflip.jsonnet'`, which resolves against the
   *running directory*. That file is a deliberate work-dir-local override: its own header says the
   pre-flip coherent-noise grouping is *"selected at run time, per running directory, by the
   `coh_groups_preflip` toggle … when the local `.coh_preflip` sentinel file is present. **The
   toolkit cfg is left at the latest (post-flip) default, so colleagues are unaffected.**"*
   Relocating these two would either drag a running-directory override into the shared toolkit or
   break it — and **it would have passed the gate**, because jsonnet imports are lazy and
   `coh_groups_preflip` defaults to false, so the import is never forced at the compile point.
   Only a run with the sentinel present would have failed, on run 027409 frames, weeks later.
2. **`pdvd/wct-img-all.jsonnet`** — not a fork of the in-tree file in name only. Besides an added
   `output_dir` TLA it calls `img(nthreshold=[1e-6, 1e-6, 1e-6])` instead of the in-tree 3.6 σ, and
   `img_maker.per_anode(anode, "multi")` instead of the default tiling. Landing it over the in-tree
   file would change PDVD imaging for every in-tree caller; landing it beside would put two files of
   one name in the tree. The divergence is undocumented, so it is surfaced, not picked (§5 rule 4).

`sbnd/wcls-img-clus-matching-xin.jsonnet` is also deferred: it imports `pr-operating-point.jsonnet`
unqualified, so it can only move once sec 4 decides that file's fate.

### 3.3 Gate

`docs/120_figs/120_move_compiles.txt`. Two compile points per file — at the runner's own TLA set
**and** bare — because the production settings do not reach every branch: `pdhd/wct-pr-perevt.jsonnet`
has 539 TLAs and its runner passes nine.

| check | result |
|---|---|
| work-dir path, before vs after the move (17 compile points: 15 bare + 2 runner-TLA) | **byte-identical** |
| canonical in-tree path vs the same baseline (14 files) | **byte-identical** |
| `prod_cfg_gate.py --ref ref/prod-2026-09-21c`, before the first edit | PASS 26/26 (`120_gate_pre.txt`) |
| same, after the move | **PASS 26/26** (`120_gate_post_move.txt`) |

The last row is an independent confirmation: `abtest/compile_all_cfg.sh` lines 57-79 compile
`wct-pr-perevt.jsonnet` and `wct-clustering.jsonnet` for PDHD and PDVD **from the work-dir paths**,
so artifact (a) is now going through the new shims and still hashes the same.

---

## 4. Round B — one operating point for both chains

### 4.0 The design fork, and how it was settled

Sec 2's fix is to give both chains one source for the operating point. Two designs reach that, and
the difference is not cosmetic.

**(A) Move the operating point into `pr()`'s own defaults** in `sbnd/clus.jsonnet` and delete the
mirror. Verified tractable: `tcn_knobs` is a pass-through (`clus.jsonnet:2068` default `{}`,
consumed at `:2605` as `knobs=tcn_knobs + {…}` with the named args winning), so the bag is one
object default, not 218 edits, and the work is ~50 named-arg defaults.

**The cost, found while implementing it.** `pr()`'s defaults are not neutral — two things read them
as a record:

* `wcls-img-clus-matching-xin-preflip.fcl` selects `pr_operating_point: "preflip"`, an A/B arm
  documented as *"the pre-2026-08-29 PR operating point … Use to reproduce those campaigns."* It
  obtains that by calling `pr()` with almost nothing and letting the defaults supply the rest.
  Changing the defaults changes what that arm means. (It is worth recording that **the arm is
  already not what it claims**: every `pr()` default that moved since 2026-08-29 — the doc-118
  trajectory flip among them — has already moved it, silently. That is the same default-inheritance
  hazard doc 119 sec 11.8 had to pin three record scripts against.)
* `pdhd/pr.jsonnet:33-35` and `protodunevd/pr.jsonnet:20-23` both say the SBND-tuned `pr()` defaults
  *"are kept verbatim (they document those operating points)"*.

**(B) Keep `pr()`'s defaults untouched; put one operating-point file in-tree, have the LArSoft chain
apply it, and make `two_chain_gate.py` a standing artifact** so an omission fails a gate instead of
surviving for weeks. Smaller diff, no A/B arm re-meaning, but two lists remain — gated ones.

What (A) buys over (B) is that there is no second list to omit from. What (B) appeared to buy is
that nothing silently re-means a frozen arm — **and that turns out to be worth nothing, because the
arm is not frozen.** `preflip` reaches its values by *inheriting* `pr()`'s defaults, so every
default that moved since 2026-08-29 moved the arm with it: the doc-118 trajectory flip on 09-20 and
doc 119's `proj_pad` flip on 09-21, at least. There is no state of the tree in which that mode means
"the pre-2026-08-29 operating point". Designing around preserving it preserves an accident.

**(A) taken.** The arm gets a correction, not a rescue: its comment is rewritten to say that it
inherits production, and that reproducing the issue-16 / issue-18 campaigns means checking out the
tree at that date. Same for `pdhd/pr.jsonnet:33-35` and `protodunevd/pr.jsonnet:20-23`, which assert
that the SBND-tuned `pr()` defaults "are kept verbatim (they document those operating points)" — a
sentence that stops being true about SBND's file and is corrected in the same commit.

What makes (A) safe is that **every one of the ~50 default edits is checked by
`two_chain_gate.py`**, which compares against the local chain — the authority. A wrong value appears
as a divergence, not as silence.

### 4.1 What was changed

`cfg/pgrapher/experiment/sbnd/clus.jsonnet`: **44 named `pr()` argument defaults** and the
`tcn_knobs` default, which goes from `{}` to the **219-key production bag**. Values come from two
places and nowhere else — the mirror, for the 242 knobs it was already supplying to LArSoft (so
those do not change there at all), and `wct-pr-perevt.jsonnet`'s own TLA defaults for the knobs the
mirror never carried. The four `root_*` switches are the `pr()` arguments behind six of the
divergent keys: `clus.jsonnet:2698` `fix_cluster_flags`, `:2700` `rec_charge_provenance`, `:2726`
`nu_provenance`.

### 4.2 Result: 24 of 25 closed, and the 25th is the physics one

`docs/120_figs/120_two_chain_after_prdefaults.txt` — the gate goes from 25 differences to **one**:

```
## OPERATING-POINT DIFFERENCES (1)
   TaggerCheckNeutrino:pr    nu_bundle_flash_group    larsoft="<absent>"   local=true
```

It survives for a reason worth recording: the mirror passes **its own** `tcn_knobs` bag, and a
caller-supplied bag *replaces* the default wholesale rather than merging with it. So the 24 keys
that are named `pr()` arguments are fixed by the new defaults immediately, while the one key that
lives in the bag stays missing until the mirror stops being applied. The same replace-not-merge
semantics is what keeps the local chain byte-identical — one mechanism, both effects.

### 4.3 V6 — the local chain did not move

`docs/120_figs/120_v6_local_unmoved.txt`. `compile_prjob_cfg.sh` against the modified tree is
sha256-identical to `ref/prod-2026-09-21c/prod_prjob.json`. This is also the **empirical** proof of
the replace-not-merge semantics that 4.2 depends on: had the 219-key default merged into the local
job's bag, this comparison would have failed and named the key.

### 4.4 NOT DONE — three edits blocked, and round B is therefore not committed

The remaining work is one file plus its dependents, and the session's permission classifier refused
every edit to it as *"Modify Shared Resources"*:

1. `sbnd/wcls-img-clus-matching-xin.jsonnet` — drop `import 'pr-operating-point.jsonnet'`, collapse
   the three-way `pr_operating_point` switch to a single bare `clus_maker.pr(...)` call, make the
   two all-APA operating-point members (`save_bundle_main_provenance`, `bee_flash_pred_min`)
   unconditional, and correct the `preflip` comment per 4.0.
2. `sbnd/pr-operating-point.jsonnet` — delete.
3. `scripts/cfg/compile_consumers.sh` step (g) — repoint at the in-tree path and drop
   `wcp-porting-img/sbnd` from `WIRECELL_PATH`; then the entry point can be relocated as sec 3.2
   deferred.

A partial edit to (1) was applied and **reverted** so the file is not left in a half-state where the
comment says the switch is retired while the code still switches.

**Consequence: `clus.jsonnet`'s change is in the working tree and is NOT committed.** Committing it
alone would close 24 divergences while leaving a comment in the LArSoft entry point that describes
`preflip` as a frozen pre-2026-08-29 arm it no longer is — the documentation defect 4.0 exists to
avoid. Revert with `git -C toolkit checkout -- cfg/pgrapher/experiment/sbnd/clus.jsonnet`.

---

## 5. Files

**toolkit** (`apply-pointcloud`): 14 new files under `cfg/pgrapher/experiment/{pdhd,protodunevd}/`.

**wcp-porting-img** (`main`): the same 14 paths replaced by re-export shims;
`sbnd_xin/scripts/cfg/two_chain_gate.py` (new); `sbnd_xin/scripts/d120/{compile_movers.sh,do_move.py}`;
this doc and `docs/120_figs/`.

Untouched (M13): every `docs/11[5-9]_*`, `products/d11*`, `scripts/d11*`, `ref/prod-*`,
`work/ql_labels/`, `abtest/snap/`, and every other experiment's configs.

---

## 6. Open items

1. **The two-chain divergence is 24/25 closed in the working tree and 0/25 closed in git** (sec
   4.4). Round B is written but not committed, because three edits to the LArSoft entry point and
   its mirror were refused by the session's permission classifier.
2. **`nu_bundle_flash_group` needs an explicit go** before it reaches LArSoft production: it changes
   what that chain reconstructs, so §5 rule 1 applies even though closing the gap is a restoration
   rather than a new flip. It is also, by 4.2, the one key that cannot close until the mirror stops
   being applied — so the go and the unblocking are the same step.
3. **Two all-APA operating-point members are invisible to `two_chain_gate.py`**:
   `save_bundle_main_provenance` and `bee_flash_pred_min` live on `clus_all_apa()`, not `pr()`, and
   the local chain builds its all-APA stage in a different job
   (`wct-clus-matching-perevt.jsonnet`) that the gate does not compile. They agree today. Extending
   the gate to the clustering stage would close the same class of hole one level up.
4. **The `preflip` A/B arm is not what its comment says** (sec 4.0), and has not been since
   2026-08-29. The comment correction is part of the blocked edit.
4. **`pdvd/wct-img-all.jsonnet`** differs from the in-tree file in slicing threshold and tiling mode
   (sec 3.2). Which is PDVD's intended standalone imaging is an owner question.
5. **`abtest/compile_all_cfg.sh` is out of step with the runners it mirrors**: it pins
   `trackfitting_config` by a work-dir path and still spells out ~25 `tgm_*`/`stm_*` values that the
   runners deleted as byte-identical on the doc-68 flip. Reported, not fixed here (§5 tie-breaker).
6. **`sbnd/wcls-img-clus.jsonnet`** (24-line work-dir fork adding `std.extVar('img_config')`) and
   **`sbnd_xin/magnify-sinks.jsonnet`** (125-line divergent fork of the in-tree file of the same
   name) are two further consolidation candidates, neither touched here.
7. The three PR drivers now share **500 of ~540 TLA names** by hand-duplication, and for the first
   time all three sit in one directory tree. Unifying them is a separate, much larger decision.
