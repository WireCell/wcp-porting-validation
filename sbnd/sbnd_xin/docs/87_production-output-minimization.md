# doc 87 — production prep: master merge, and knobs for every output the next stage does not need (2026-09-01)

**Status:** merge landed (toolkit `f439a109`); SBND PR chain gated
**byte-identical on 308 events**.  Output knobs all **DEFAULT TO TODAY'S
BEHAVIOUR**; the operating point is unchanged and `prod_cfg_gate.py` PASSes
21/21 consumers untouched.  One knob, `save_in_scope`, is a *content* change and
ships DEFAULT OFF pending an owner flip (§4.6).

Owner ask, three parts:

> *"1. Merge the Master Branch to this branch and make sure it does not change
> the behaviour  2. In my local production, we have been saving a lot of extra
> files (e.g. Imaging --> QL stage), we saved the 2D measurement file
> (compressed?), this is not needed to run the later PR chain  3. During the PR
> chain, we saved the bee files, some calibration files, as well as the final
> pctree. These are useful for debugging purpose, but not exactly useful for
> production usage. I think we should have dedicated knobs to allow to turn them
> off … 1. in my local running, I still want them default on, 2. in the md file,
> please give instructions (after validation) that how to turn them off."*

and the rule that set the scope of part 2:

> *"there are some files there where we need them to do the next PR step, if we
> do not need them for PR, we should have a knob to turn them off?"*

So every QL artifact is classified by PR-necessity below, not just the one the
question named.

## Repro

```bash
cd wcp-porting-img/sbnd/sbnd_xin
S=<scratch>/doc87

# --- part 1: the merge ---------------------------------------------------
cp -a ../../../toolkit/.claude/skills $S/skills-premerge   # doc 69 sec 2 trap
cp -a /nfs/data/1/xqian/toolkit-dev/local/lib/. $S/lib-pre # pin BEFORE editing
cd ../../../toolkit && git merge origin/master
./wcb install --notests -p --targets=WireCellIface,WireCellAux,WireCellGen  # see sec 1.3
wcbuild
for p in util iface aux gen sio root img clus; do ./build/$p/wcdoctest-$p; echo "$p rc=$?"; done

# --- gates (all three parts use the SAME manifest) -----------------------
cd ../wcp-porting-img/sbnd/sbnd_xin
scripts/cfg/prod_cfg_gate.py                                  # 21/21 PASS
for s in nuecc48 ncpi0 mcp1k; do
  LD_LIBRARY_PATH=$S/lib-post PR_JOBS=16 \
    ./run_pr_chain_batch.sh work-$s-grp0825 work-87mrg-post-$s data \
      $(cat ref/prod-2026-09-01/gate308-$s.txt)
  python3 scripts/pr85_hash_gate.py work-77r4-rm11-$s work-87mrg-post-$s; echo rc=$?
  python3 scripts/pr94_root_gate.py work-77r4-rm11-$s work-87mrg-post-$s; echo rc=$?
done

# --- part 3: save_in_scope acceptance + characterization ------------------
echo "save_in_scope=true" > $S/tla.txt
PR_EXTRA_TLA=$S/tla.txt PR_JOBS=16 ./run_pr_chain_batch.sh \
    work-ncpi0-grp0825 work-87scope-on-ncpi0 data $(cat ref/prod-2026-09-01/gate308-ncpi0.txt)
python3 scripts/pr87_inscope_gate.py work-87scope-on-*        # 308/308 PASS
python3 scripts/pr87_root_tree_diff.py work-87tc-off-<s> work-87scope-on-<s>
./scripts/pr87_nusel_source_gate.sh work-87scope-on-<s> work-<s>-grp0825   # 308/308

# --- the knobs -----------------------------------------------------------
PR_JOBS=16 ./run_pr_chain_batch.sh work-<s>-grp0825 work-87knob-def-<s> data ...  # default
PR_MINIMAL_OUTPUT=1 ... work-87knob-min-<s>
SBND_PR_BEE=0 SBND_PR_PCTREE=0 ... work-87knob-sup-<s>
```

---

## 0. TL;DR for an operator

**Nothing changes unless you set something.**  Every knob below is unset by
default and an unset knob emits no TLA at all, so the compiled config and every
output byte are what they were.  Only the literal value `0` turns an output off,
so a typo (`SBND_PR_BEE=no`) fails **safe** — it keeps writing.

| set this | and you stop writing | GB / 1000 evt |
|---|---|---|
| `PR_MINIMAL_OUTPUT=1` | `mabc-pr.zip` + `pctree-pr-evt<ID>.tar.gz` + calib | **2.76** |
| `SBND_PR_BEE=0` | `pr_evt<ID>/mabc-pr.zip` | 0.24 |
| `SBND_PR_PCTREE=0` | `pr_evt<ID>/pctree-pr-evt<ID>.tar.gz` | 2.24 |
| `SBND_PR_CALIB=0` | `calib-pr-evt<ID>.json` (already opt-in) | 0.28 |
| `SBND_QL_PERFACE_BEE=0` | `ql_evt<ID>/mabc-apa{0,1}-face0.zip` | 0.24 |
| `SBND_QL_ALLAPA_BEE=0` | `ql_evt<ID>/mabc-all-apa.zip` | 0.47 |
| `SBND_QL_KEEP_ICLUSTER=0` | `evt<ID>/icluster-apa*.npz` (after Q/L succeeds) | **4.80** |

See §5 for the exact production lines and **what each one costs you**.

---

## 1. Part 1 — the master merge

Merge base `b806824f` (the second parent of `87ada3d5`), so this is the 4th
master→branch merge.  `git merge-tree --write-tree` reported **no conflicts** in
both directions before the merge was run, and the merge itself was clean.

**15 files, +840/−10.**  `clus/`, `img/`, `sigproc/` and `cfg/pgrapher/**` are
**not touched at all**.  The payload is new `ITrackSegment*` interfaces,
`SimpleTrackSegment*` aux impls, `Gen::TrackSegmentSampler` (a *simulation* depo
source), `CONTRIBUTING.md`, and tests.

### 1.1 The one behaviour-capable change, and why SBND is untouched

`e6fb7ef3` removes a spurious `units::cm` from `Gen::{Birks,Box}Recombination`'s
`operator()` and `dE()`.  The quenching term was inflated 10×, giving R ≈ 0.32
instead of the physical ≈ 0.70 for a MIP at 500 V/cm.  That is a **>2× shift in
every dQ/dx→dE/dx conversion routed through those two classes**, and upstream's
own message flags it: *"behavior change for any existing Birks/Box user (eg clus
dQ/dx energy estimation)"*.

SBND production does not route through them.  Checked against the **pinned
compiled config**, not inferred from the jsonnet:
`ref/prod-2026-09-01/prod_prjob.json` holds exactly one recombination node,
`"type": "PowerBoxRecombination"`, and both `"recombination_model"` keys read
`"PowerBoxRecombination:sbnd_power_recomb"`.  `BoxRecombination` appears **zero**
times as a type.  (`use_power_recomb=true` is the default in both
`sbnd/clus.jsonnet` and `wct-pr-perevt.jsonnet`, which leaves `sbnd_box_recomb`
an unreferenced jsonnet local that is never emitted.)  `PowerBoxRecombination`
is our own class and shares no helper with Box or Birks.

A cheap corroboration fell out of the build: `libWireCellClus.so` is
**byte-identical** before and after the merge (md5 `8c083812…` both sides);
only `libWireCellGen.so` differs.

### 1.2 Two upstream tests had to be repaired (toolkit `5d924ab9`)

Both are test-only and both are consequences of master, not of this branch.

- **`util/test/doctest_response.cxx` does not compile anywhere.**  `79c4043d`
  ("Fix compiler warning->error due to unused vars") commented out the
  `auto fr =` binding while leaving line 17 passing `fr` to
  `wire_region_average()`.  Restored the binding; kept the genuinely-unused
  `fravg` commented, which was that commit's actual intent.
- **`gen/test/doctest_powerbox_recombination.cxx` failed 8 of 8 sampled
  dE/dx.**  Our test asserts the identity *PowerBox with p=1, C=1, A=1 reduces
  to the plain Modified Box*, and mapped `B/(ρE)` into PowerBox's `k` under the
  OLD Box convention.  PowerBox builds `u` from dE/dx in MeV/cm; Box's quenching
  term is now the plain WCT-units ratio, so the mapping constant must carry the
  same `units::MeV/units::cm` = 0.1 factor master just fixed — exactly the 10×.
  The identity itself is intact.  Written symbolically so it cannot rot again.

**This did not spread.**  `wcdoctest-clus` is **234 cases / 2599 assertions
green**, including `doctest_pattern_recognition` and
`doctest_tagger_check_neutrino`, which bind `BoxRecombination:box_recomb`
explicitly and were the highest-probability breakage.

### 1.3 GOTCHA — `wcbuild` fails twice on a fresh merge, and re-running does not help

The first `wcbuild` after the merge dies with `undefined reference to 'vtable
for WireCell::Aux::SimpleTrackSegmentSet'` and
`ITrackSegmentSet::~ITrackSegmentSet()`.  The symbols exist in `build/`.  They
do not exist in `local/lib`, and **`wcdoctest-<pkg>` links the INSTALLED
library** — the same trap doc 77 §13.3 recorded for a changed *signature*, here
for *new* symbols.  `wcbuild` is `build && install`, so a test-link failure in
the build step means install never runs and the stale lib is never refreshed:
the failure is self-perpetuating.

*Rule:* `./wcb install --notests -p --targets=WireCellIface,WireCellAux,WireCellGen`
first, then `wcbuild`.  This is **not** M3's install race; re-running alone
never fixes it.

### 1.4 Merge gate — PASS

Post-merge arms `work-87mrg-post-*` (binary pinned to `$S/lib-post`) against
`work-77r4-rm11-*`, the arms doc 77 round 4 validated at `7542532f`.
**308 events** = 241 mcp1k + 48 nueCC + 19 NCπ⁰, `PR_JOBS=16`, all `rc=0`.

| sample | events | `pr85_hash_gate.py` | `pr94_root_gate.py` | sorted `nusel-table.tsv` |
|---|---|---|---|---|
| nueCC48 | 48 | PASS 96/96 byte-identical | PASS 48/48 | 0 lines |
| NCπ⁰ | 19 | PASS 38/38 byte-identical | PASS 19/19 | 0 lines |
| mcp1k | 241 | PASS 482/482 byte-identical | PASS 241/241 | 0 lines |

Plus `prod_cfg_gate.py`: **PASS, 21/21 consumers** (SBND pr/img/clus/QL, PDHD ×6,
PDVD ×5, uBooNE MABC, …).  This merge changes **zero** compiled config — the
previous one (doc 69) shifted 13 lines of SCE wiring.

### 1.5 Two things to report, not to fix here

- The `use_power_recomb=false` comment in `sbnd/clus.jsonnet` and
  `wct-pr-perevt.jsonnet` claims `false` restores *"the byte-identical pre-pr/10
  config"*.  **That sentence is false after this merge** — the off-arm now moves
  by the full ~2×.  Nothing pins it off (no gate, bats test or env var
  references `use_power_recomb`), so the exposure is ad-hoc runs plus the
  accuracy of those two comments.
- Any **uBooNE** chain that reaches `NeedRecombModel` without naming a model
  takes `ClusteringFuncsMixins.cxx:16`'s `"BoxRecombination"` default and would
  see the full shift.  No cfg in this tree does — only SBND instantiates
  `tagger_check_*` at all — but uBooNE is a frozen reference and this is the
  owner's call.

---

## 2. What the PR chain actually reads — measured, not assumed

Part 2's scope comes from the owner's rule, so every QL artifact is classified.
Consumers were found by grep across **both** repos, and corroborated against the
pinned compiled PR config, whose only file keys are `inname: "in.tar.gz"`,
`outname: "out.tar.gz"` and `bee_zip: "out/mabc-pr.zip"` — there is no
`icluster` or `.npz` reference in it at all.

| `ql_evt<ID>/` file | read by | verdict |
|---|---|---|
| `pctree-evt<ID>.tar.gz` | the PR job's `TensorFileSource` | **required** |
| `opflash_apa0.tar.gz` | shell only, to recover run/subrun | **required** |
| `mabc-all-apa.zip` | python only — `nusel_extract.py --qlbee`, an *optional* per-merge-component geometry cross-check that group mode already skips (`QLBEE=""`) | knob |
| `mabc-apa{0,1}-face0.zip` | **nothing, anywhere in either repo** | knob |
| `opflash_apa1.tar.gz` | nothing (only apa0 is opened) | staged *input* copy, folded into the icluster cleanup |
| `icluster-apa*.npz` (via `evt<ID>/`) | the **Q/L job**, and nothing downstream | knob (delete after Q/L) |

### 2.1 "2D measurement file" has no literal referent

Grep for `2dmeas`, `2d_measure`, `"2D measurement"` across `cfg/`, `clus/`,
`img/`, `aux/`, `sio/` and all of `sbnd_xin/`: **zero code or config hits** —
only prose in `docs/pr/{21,83,86,108,109}` and `TrackFitting.cxx`'s internal
`n_2d_measurements`.  It is an informal name, so the owner was asked and
answered **"both of them"**: the imaging→Q/L handoff (`icluster-*.npz`, whose
`'m'` nodes literally *are* the ICluster measurements — `ClusterArrays.cxx:130,
279-291`) and the per-APA-per-**face** Bee zips.

### 2.2 One thing deliberately NOT touched

`ctpc_a<A>f<F>p<P>` inside the pctree is 32 % of the tarball (~730 MB/1000 evt)
and *is* the WCP 2D charge map.  `Facade_Grouping.cxx:640,890,930` reads it back
and the PR job has no `PointTreeBuilding` to rebuild it, so dropping it would
break the chain.  Not a knob.

---

## 3. The three PR outputs are not symmetric

| output | state before this doc |
|---|---|
| `calib-pr-evt<ID>.json` | **already fully opt-in** — `PrDisplayDump` is instantiated only when `PR_EXTRA_STAGES=pr_display`; `doctest_clus_knob_defaults.cxx` already asserted it inert by default |
| `pctree-pr-evt<ID>.tar.gz` | a switch **existed** — `save_tensors=''` ⇒ `dump_mode:true` ⇒ `TensorFileSink.cxx:133-137` early-returns — but the runner always passed a path |
| `mabc-pr.zip` | **no off-path existed at all** |

*(Aside: the 461-of-1000 calib files in `prod0901` are not a bug.
`PrDisplayDump::visit` early-returns when no neutrino candidate produced a
TrackFitting.)*

### 3.1 The Bee guard — the only genuinely new C++ output path

`Bee::Sink::reset(store)` calls `Stream::output_filters` and `raise<IOError>`
when the result is empty, so an **empty `bee_zip` used to throw**; it was never
a legal value, let alone a disable.  `MultiAlgBlobClustering::configure` now
takes an empty `bee_zip` (with no `bee_sink`) to mean *write no Bee zip*: the
sink is never built, and `write_obj()` / `ensure_own_sink()` become no-ops.
`Sink::close()` was already a no-op on an unopened sink, so `finalize()` needed
nothing.  Default stays `"mabc.zip"`.

That makes the default **load-bearing in a new way** — if it ever became `""`,
every detector would silently stop writing `mabc*.zip` and SBND would lose
`nusel-evt<ID>.tsv` with it.  A zero-fires census cannot see that, so
`doctest_clus_knob_defaults.cxx` now pins it.

### 3.2 Two couplings, handled rather than discovered

- **`mabc-pr.zip` off used to mean no nusel table.**  `nusel_extract.py` called
  `parse_prbee()` unconditionally.  Solved by §4.
- **pctree off used to fail every event in group mode.**  The per-event verdict
  was `[ -s pctree-pr-evt<ID>.tar.gz ]`, whose own comment called that product
  *"unconditional in the group path"* — it no longer is.  It now asks for
  whichever product the run was configured to write, falling back to
  `tracking-pr.root`, which the visitor writes on every event and no knob
  suppresses.

---

## 4. `save_in_scope` — the in-scope set becomes a stored quantity

### 4.1 The question that produced it

Owner: *"if it is not recoverable, how does the flags known in the PR step?"*
and then *"Since it happens in the PR step, it must be stored inside the final
rootfile, right?"*

**The PR chain computes the flag and discards it.**  `switch_scope`, the first
pipeline stage, splits each incoming cluster on a per-blob filter over the
drift-corrected points and stamps
`new_cluster->set_scope_filter(correction_scope, id == 1)`
(`clustering_switch_scope.cxx:120-125`).  Every tagger's `require_in_scope` then
reads it back through `get_scope_filter()`, a plain lookup in the cluster's
in-memory map (`Facade_Cluster.cxx:105-112`).  Nothing serialized it.

`mabc-pr.zip` recorded it **implicitly**, by containing exactly the in-scope
clusters: `MultiAlgBlobClustering.cxx:2906-2923` gates the Bee clustering layer
on literally `cluster.get_scope_filter(scope)` with `filter == 1`.  That is why
`parse_prbee()` was the only reader that had it — and why turning the Bee zip
off silently killed the nusel table.

### 4.2 Why neither existing product could supply it

| candidate | why not |
|---|---|
| the pctree | evt 166738 has **77** clusters in `cluster_scalar` vs the Bee zip's **69**.  The 8 extras are all `flag_main_cluster==1` — and so are many *in*-scope clusters, so **no existing array separates them**.  There is no `flag_in_scope`. |
| `tracking-pr.root` as it stood | `T_tagger` exists on only **95 of 200** sampled mcp1k events (the rest hold just `Trun`/`T_proj`/`T_bad_ch`), because `UbooneTaggerOutputVisitor` runs only when a candidate is evaluated.  Where it does exist, `act_cluster_id` is **18** ids in a different id space, and includes id 2, which is not in the 69 at all. |
| the log | the taggers name **3** cluster ids on that event, not 69. |

### 4.3 Where it went, and why there

`SbndPrMagnifyTrackingVisitor` — which opens `tracking-pr.root` `RECREATE` and
**runs on every event** — gains a `T_cluster` tree beside `T_bad_ch`, `Trun`,
`T_proj_data`, `T_rec_charge`.  Not `UbooneTaggerOutputVisitor`: that is the one
that skips non-candidate events.  It is stage 14 of 15, after every tagger, with
only the output visitor between it and the Bee write.

One row per live cluster, emitted **sorted by ident** (never `children()`
order): `cluster_id, in_scope, is_main, is_associated, tgm, stm, fc, lm,
beam_flash, flash_id, flash_time_us, flash_pe, npoints, length_cm,
cluster_t0_us`.  `in_scope` is the *same call* the Bee layer is gated on, so the
set is the Bee set by construction rather than by resemblance.

### 4.4 Acceptance — the claim that cannot pass by accident

`scripts/pr87_inscope_gate.py` asserts
`set(T_cluster.cluster_id where in_scope==1) == parse_prbee(mabc-pr.zip)`
**exactly**:

| sample | events | match | mismatch |
|---|---|---|---|
| nueCC48 | 48 | 48 | 0 |
| NCπ⁰ | 19 | 19 | 0 |
| mcp1k | 241 | 241 | 0 |
| **total** | **308** | **308** | **0** |

And `scripts/pr87_nusel_source_gate.sh` regenerates `nusel-evt<ID>.tsv` from each
source in turn and requires the two **byte-identical**: **308 / 308, 0
differing.**  So `nusel_extract.py --prroot` is a drop-in for `--prbee`, which is
what makes `SBND_PR_BEE=0` survivable.

> **GOTCHA — `T_rec_charge.reduced_chi2` carries NaNs.**  A naive `!=`
> comparison reports every NaN row as a difference and makes a clean arm look
> broken; that is what a first pass here did, on 5 of 19 events, before the
> merge-arm pair showed the identical count.  `pr94_root_gate.py` already
> handles it; `scripts/pr87_root_tree_diff.py` compares with `equal_nan=True`
> and exists so a *characterized* difference can be recorded instead of a bare
> FAIL.

### 4.5 The knob-ON difference, characterized

`pr94_root_gate.py` correctly FAILs an ON-vs-OFF pair — the file gained a tree.
`scripts/pr87_root_tree_diff.py` says what actually moved:

| sample | events with every pre-existing tree identical | trees only in the ON arm | `mabc-pr.zip` + pctree |
|---|---|---|---|
| nueCC48 | 48 / 48 | `T_cluster` | PASS 96/96 byte-identical |
| NCπ⁰ | 19 / 19 | `T_cluster` | PASS 38/38 byte-identical |
| mcp1k | 241 / 241 | `T_cluster` | PASS 482/482 byte-identical |

### 4.6 Shipping status — DEFAULT OFF, awaiting an owner flip

`save_in_scope` **adds** a tree, so it is a content change: `tracking-pr.root`
stops being byte-identical to every arm recorded before doc 87.  It therefore
ships C++-default `false`, key-suppressed, and pinned by
`root/test/doctest_sbnd_pr_tracking_defaults.cxx`.  Knob-off gates: **616/616
archives byte-identical, 308/308 ROOT identical, `prod_cfg_gate.py` 21/21.**

**It is not flipped.**  The owner asked for the information to be in the ROOT
file, and the measurement says it can be — but flipping a default is a separate,
owner-recorded step (CLAUDE.md §5.1), and it should land in its own commit
quoting the request.  `SBND_PR_BEE=0` auto-enables it for the run that needs it,
so nothing is blocked meanwhile.

---

## 5. The knobs, and what each one costs you

### 5.1 Production lines

```bash
# residual disk minimal, NOTHING downstream lost -- the recommended line
PR_MINIMAL_OUTPUT=1 PR_JOBS=16 \
    ./run_pr_chain_batch.sh <ql_root> <out_root> data [evt ...]

# add the Q/L side (run at stage A, not stage B)
SBND_QL_PERFACE_BEE=0 SBND_QL_ALLAPA_BEE=0 SBND_QL_KEEP_ICLUSTER=0 \
    ./run_chain_group.sh <reco1.root> <out_root> data --size 16 --layout perevt

# suppress at SOURCE instead (no write at all; see the trade in 5.3)
SBND_PR_BEE=0 SBND_PR_PCTREE=0 PR_JOBS=16 \
    ./run_pr_chain_batch.sh <ql_root> <out_root> data [evt ...]
```

### 5.2 Two honest modes, and why the master switch is the recommended one

**`PR_MINIMAL_OUTPUT=1` does not suppress at source.**  It lets the job write the
Bee zip and the pctree, runs the per-event `nusel_extract.py` that needs them,
and *then* deletes them.  Residual disk equals suppression and the per-event peak
is one event's worth.  What it buys over suppression:

- `nusel-evt<ID>.tsv` keeps its **authoritative** pctree-derived `flag_TGM/STM/FC`
  instead of the tear-prone log fallback (the reason `parse_prtree_flags` exists —
  a torn spdlog line silently read no verdict on evt 287517);
- the group-mode per-event verdict keeps working off the product it was written
  against;
- it needs no `save_in_scope`, so it works on today's default build.

**Suppress-at-source** is for a chain that will never look at those files again.
It is genuinely cheaper (no write, no I/O), and with `save_in_scope` it loses
nothing measurable — but see §5.3.

### 5.3 What you give up

| you set | you lose |
|---|---|
| `PR_MINIMAL_OUTPUT=1` | the Bee event display, the calib dump, and any *later* re-analysis that wants the post-PR tree.  `nusel-evt<ID>.tsv`, `nusel-table.tsv` and `tracking-pr.root` are all intact. |
| `SBND_PR_BEE=0` | the Bee display; **and the in-scope set unless `save_in_scope` is on** — the runner therefore defaults it on for you (an explicit `SBND_PR_SAVE_IN_SCOPE` still wins) |
| `SBND_PR_PCTREE=0` | authoritative flags (nusel falls back to the log), any re-run of the PR chain from its own output, and `pr85_hash_gate.py` coverage of that product |
| `SBND_PR_CALIB=0` | every `scripts/pr1[2-4]x_*.py` census — they read the calib JSON and nothing else does |
| `SBND_QL_ALLAPA_BEE=0` | `nusel_extract.py --qlbee`, the optional per-merge-component geometry cross-check (group mode already runs without it) |
| `SBND_QL_PERFACE_BEE=0` | nothing measurable — no consumer exists in either repo |
| `SBND_QL_KEEP_ICLUSTER=0` | the ability to re-run **Q/L** without re-running imaging.  The PR chain is unaffected. |

> **An arm run with pctree and Bee suppressed is NOT A/B-gateable.**
> `pr85_hash_gate.py` compares exactly those two archives, so with both off there
> is nothing left for it to compare.  **Every future validation arm must run with
> the debug outputs ON.**  This is a property of minimal mode, not a defect —
> but it has to be said out loud, because a future round that quietly adopts the
> production line for a gate arm would get a vacuous PASS.

### 5.4 Failure mode is safe by construction

Every knob is read as `[ "${VAR:-1}" = 0 ]`.  Unset, empty, `no`, `false`, `off`
all leave the output ON.  Only the literal `0` disables.  An unset knob appends
**no TLA at all**, so the compiled config is byte-identical rather than
merely equivalent — `prod_cfg_gate.py` PASSes 21/21 with every knob in place.

---

## 6. Validation

All arms use the committed 308-event manifest
`ref/prod-2026-09-01/gate308-{mcp1k,nuecc48,ncpi0}.txt` (241 + 48 + 19), new
labels throughout (M13), binaries pinned to a snapshot for every arm (M1), and
every exit code taken directly, never through a pipe (M14).

### 6.1 Knobs at their defaults change nothing

`work-87knob-def-*` (all knobs unset, final binary and config) against
`work-87tc-off-*` (the pre-knob arm):

| sample | events | `pr85_hash_gate.py` | `pr94_root_gate.py` | sorted `nusel-table.tsv` |
|---|---|---|---|---|
| nueCC48 | 48 | PASS 96/96 | PASS 48/48 | 0 lines |
| NCπ⁰ | 19 | PASS 38/38 | PASS 19/19 | 0 lines |
| mcp1k | 241 | PASS 482/482 | PASS 241/241 | 0 lines |

`prod_cfg_gate.py`: **PASS 21/21** with every knob added.

### 6.2 The reduced modes keep the physics table exactly

`work-87knob-min-*` (`PR_MINIMAL_OUTPUT=1`) and `work-87knob-sup-*`
(`SBND_PR_BEE=0 SBND_PR_PCTREE=0`), both against `work-87knob-def-*`:

| arm | sample | `nusel-table.tsv` | per-event `nusel-evt<ID>.tsv` | `mabc-pr.zip` | pctree | calib |
|---|---|---|---|---|---|---|
| minimal | NCπ⁰ 19 | 0-line diff | 19/19 byte-identical | 0/19 | 0/19 | 0/19 |
| minimal | nueCC 48 | 0-line diff | 48/48 byte-identical | 0/48 | 0/48 | 0/48 |
| suppress | NCπ⁰ 19 | 0-line diff | 19/19 byte-identical | 0/19 | 0/19 | 0/19 |
| suppress | nueCC 48 | 0-line diff | 48/48 byte-identical | 0/48 | 0/48 | 0/48 |

Both leave exactly `nusel-evt<ID>.tsv`, `tracking-pr.root`, `rc.txt`,
`stdout.log`, `wct_pr_evt<ID>.log` — and **no `trash-pr.tar.gz`** (§6.5).
The minimal arm's `tracking-pr.root` is identical to the default arm's
(`pr94_root_gate.py` PASS 19/19); the suppressed arm's differs by `T_cluster`
alone, and *that tree's in-scope set still equals the default arm's Bee set on
19/19 events* — i.e. the set survives with the file that carried it deleted.

### 6.3 Part 2's decisive gate — the PR input is untouched

Same event (NCπ⁰ 37112), same binary, Q/L run twice with only the knobs
changed, compared by **member-content hash** (`hash_archive.py`; M2 — raw `cmp`
on a tarball is meaningless):

| artifact | knobs off | knobs on | shipped `grp0825` |
|---|---|---|---|
| `pctree-evt37112.tar.gz` | `29bbede1…` | `29bbede1…` | `29bbede1…` |
| `mabc-all-apa.zip` | `81d1eee1…` | *not written* | `81d1eee1…` |
| `mabc-apa{0,1}-face0.zip` | written | *not written* | written |

The pctree — the PR chain's **only** input — is byte-identical three ways,
including against the shipped production arm.  The knob-off run reproducing
`grp0825`'s Bee zip exactly is what makes the comparison non-vacuous: it proves
the reproduction is faithful before anything is switched off.

`SBND_QL_KEEP_ICLUSTER=0` is a runner deletion and was exercised on a synthetic
tree rather than on a real arm (M13): it removes exactly the `icluster-apa*.npz`
files **and the now-dangling `ql_evt<ID>/` symlinks into them**, leaving every
other product in place.

### 6.4 Group mode, and a gate that passes vacuously

Group mode (`PR_GROUP_SIZE=16`) needed its own arms because §3.2's verdict fix
lives there.  NCπ⁰ 19 events, `PR_JOBS=2`:

| arm | result |
|---|---|
| group, all defaults | vs the per-event default arm: `pr85_hash_gate` PASS 38/38, `pr94_root_gate` PASS 19/19, `nusel-table.tsv` 0-line diff — **doc 81's group/per-event equivalence still holds** |
| group, `SBND_PR_PCTREE=0` | **19/19 `rc=0`**, 0/19 pctree, 19/19 `mabc-pr.zip` kept and byte-identical, `nusel-table.tsv` 0-line diff, no stray `trash-pr.tar.gz` |
| group, `PR_MINIMAL_OUTPUT=1` | **19/19 `rc=0`**, 0/19 `mabc-pr.zip`, 0/19 pctree, no stray `trash-pr.tar.gz`, `nusel-table.tsv` 0-line diff, per-event tsv 19/19 byte-identical |

Before the fix, `SBND_PR_PCTREE=0` in group mode would have stamped `rc=1` on
**every** event and failed the group, because the per-event verdict was taken
from the very product the knob suppresses.

> **TRAP — `pr85_hash_gate.py` reports a suppressed arm as PASS.**  Run against
> the pctree-off arm it prints
> `# missing: 18625 21073 …` and then
> **`PASS all 0 archives byte-identical`** — it needs *both* archives per event,
> found neither pair complete, compared nothing, and passed.  That is a vacuous
> PASS, and citing it as evidence would be exactly the mistake §5.3 warns about.
> The honest comparison is the surviving artifact on its own: `hash_archive.py`
> on `mabc-pr.zip` gives **19/19 identical**.  This is the concrete reason a
> validation arm must never adopt the production line.

### 6.5 Unit tests

| binary | result |
|---|---|
| `wcdoctest-clus` | 235 cases / 2603 assertions, 0 failed |
| `wcdoctest-root` | 5 cases / 4035 assertions, 0 failed |
| `wcdoctest-gen` | 13 cases / 210 assertions, 0 failed |
| `wcdoctest-util` | 42570 assertions, 0 failed |
| `wcdoctest-aux` | 110738 assertions, 0 failed |
| `wcdoctest-iface` / `-sio` / `-img` / `-root` | all green |

Two new default pins: `bee_zip` must stay non-empty
(`doctest_clus_knob_defaults.cxx`) and `save_in_scope` must stay false
(`doctest_sbnd_pr_tracking_defaults.cxx` — new).

### 6.6 GOTCHA — the pctree knob leaves a file behind unless the runner sweeps it

`save_tensors=''` puts `TensorFileSink` in `dump_mode`, which writes nothing —
but `TensorFileSink::configure` opens the output stream **unconditionally** for a
non-templated `outname`, so the job default `trash-pr.tar.gz` is still *created*
in the process CWD.  The per-event path already deleted its copy; group mode's
CWD is `$OUTROOT`, shared by every concurrent group, so the group path now sweeps
it too.  Verified absent in every arm above.

---

## 7. What changed

**toolkit** (`apply-pointcloud`):

| commit | what |
|---|---|
| `f439a109` | the merge itself |
| `5d924ab9` | `util/test/doctest_response.cxx`, `gen/test/doctest_powerbox_recombination.cxx` — repair two upstream-broken tests (§1.2) |
| `d84352c7` | `SbndPrMagnifyTrackingVisitor` + `T_cluster`, `save_in_scope` TLA, `root/test/doctest_sbnd_pr_tracking_defaults.cxx` (new) — **DEFAULT OFF** |
| `b7c9aff4` | `MultiAlgBlobClustering` empty-`bee_zip` guard, `pr_bee` / `perface_bee` / `allapa_bee`, `bee_zip` default pinned — **all DEFAULT ON** |

No other experiment's config is touched.  No pre-existing C++ default is changed.

**wcp-porting-img** (`main`): this doc; `run_pr_chain_batch.sh` (the four PR
knobs, `--prroot`, the group verdict-file fix, the group `trash-pr` sweep);
`run_chain_group.sh` + `run_ql_evt.sh` (the two QL Bee knobs, the icluster
cleanup); `nusel_extract.py` (`parse_prroot_scope` + `--prroot`); and three new
gates —

| script | asserts |
|---|---|
| `scripts/pr87_inscope_gate.py` | `T_cluster` in-scope set **==** `parse_prbee(mabc-pr.zip)`, exactly |
| `scripts/pr87_nusel_source_gate.sh` | `--prbee` and `--prroot` give a **byte-identical** nusel tsv |
| `scripts/pr87_root_tree_diff.py` | *which* trees/branches moved, NaN-aware — so a knob-ON diff can be characterized instead of merely failing |

## 8. Next

1. **Flip `save_in_scope` ON** (§4.6).  It is measured, gated, and it is what the
   owner asked for; it needs only an explicit word, and should land in its own
   commit quoting it.  Until then `SBND_PR_BEE=0` turns it on per-run.
2. **Correct the two `use_power_recomb=false` comments** (§1.5) — they now
   promise a byte-identity that the merge removed.
3. **Decide on uBooNE and the Box fix** (§1.5).  Nothing in this tree is exposed,
   but the frozen reference's dQ/dx→dE/dx would move ~2× if anything ever ran it.
4. Optional: widen `T_cluster` consumers.  `nusel_extract.py` reads only
   `in_scope` today; the tree already carries the flags, flash and geometry that
   the rest of the nusel row is rebuilt from, so a later round could drop the
   `--pctree`/`--qlbee` dependencies too and make `tracking-pr.root` genuinely
   self-sufficient.

**Doc 87 status: parts 1-3 DONE, gated, and byte-identical at every default.**
