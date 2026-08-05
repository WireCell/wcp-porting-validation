# doc pr/37 — what docs pr/28–32 missed: a cross-audit residual round

**Owner instruction.** *"Can you take a look at
[pr/28, pr/29, pr/30, pr/31, pr/32] to see which part that we may missed in our
existing work. Here, I am looking for major bugs, non-determinstics, and other
crucial things. Please add a new md file to summarize these for one future
round. Please feel free to look at prototype code, toolkit code to check. No
need to modify the code for now."*

Each of the five documents ran its own owner filter and shipped its own fixes,
and each ends with a "what is NOT claimed" section. Four of the five name blocks
of code that were **never opened**. Nobody had read the five *together*. This
round does that, and it adds four things none of the five could have found from
the inside.

**No toolkit C++ or jsonnet is changed by this document.** Two measurements were
run on the already-installed binary; no build, no A/B gate, no knob.

**The one-line answer.** The largest thing missed is not a bug in any of the five
— it is that **the SBND production operating point does not execute the code
layer pr/32 spent its Tier-A effort verifying.** On 47 of 48 nueCC events the DL
vertex is accepted and `determine_overall_main_vertex` is never entered. That is
now measured (§1), it closes pr/32 loose end 1, and it demotes this round's
otherwise-headline source defect (§2) from *live* to *latent*.

### What this round found, ranked

| § | finding | class | status |
|---|---|---|---|
| **1** | SBND accepts the DL vertex **47/48**; the whole traditional overall-main-vertex layer — including `compare_main_vertices_global`, verified term-for-term in pr/32 §2.5 — runs **0 times**. The DL re-ranker's 4.0 acceptance floor has **never rejected** | **measured** | closes pr/32 loose end 1 |
| **2** | `determine_overall_main_vertex` takes its map and `main_cluster` **by value** while its DL twin one function above takes both **by reference**, so the main-cluster swap and the `examine_main_vertices` erases are discarded while `other_clusters` and the facade flag persist — an inconsistent state | source-verified, **reach 0/48** | latent; recommendation given |
| **3** | **doc pr/33's owner filter was never implemented** — 5 findings, 8 named knobs, zero code; the only *completed* round in that state (pr/36 is also at zero but was filtered 2026-08-04 and is expected to be — §3) | verified by grep | schedule |
| **4** | The prototype reference is branch `port` @ `53ca938`, **+5833/−989 over 26 files**, and pr/28–32 pin no SHA. Measured: the audited files' substantive diffs are 0–33 lines and all six live hunks triage to instrumentation or to **prototype bugs the toolkit is structurally immune to** | measured + source | citations **survive**; one-line process fix |
| **5** | Determinism source sweep at HEAD is clean, and the repeat-run **floor is 0** across 59 890 ROOT leaves + every archive | **measured** | pr/28's zero survives six knob rounds |
| **6** | Seven unaudited blocks the five docs name, ranked — `examine_structure_*` first, because it runs on SBND and *"can move vertices"* | inherited | schedule |
| **7** | The stale valfast/1000 baseline now carries **19 knobs ON plus one unconditional change**, derived from cfg — not the "ten" of pr/32 §11.8 | derived | schedule |

---

## Repro

```sh
# Trees and binary this document was written against
cd /nfs/data/1/xqian/toolkit-dev/toolkit && git rev-parse --short HEAD   # 29e8e452
ls -la --time-style=full-iso /nfs/data/1/xqian/toolkit-dev/local/lib/libWireCellClus.so
#   2026-08-04 19:47:37 -0700  -- installed BEFORE both runs below, unchanged after
# The prototype reference is a BRANCH, not a tag -- see sec.4:
cd /nfs/data/1/xqian/prototype-dev/wire-cell/pid
git rev-parse HEAD                      # 53ca93824306ea49fe480e8bc9e0a5dd678f81e5  (branch 'port')
git merge-base HEAD master              # a5fc0b9dc1a516c184d5063a8796619d588ab212

# --- sec.1 / sec.2: does the traditional overall-main-vertex layer run? -------
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
SBND_WCT_LOGLEVEL=trace PR_JOBS=5 \
  ./run_pr_chain_batch.sh work-nuecc48-prod0803 work-pr37-trace48 data
cd work-pr37-trace48
grep -l 'switching to DL vertex'          pr_evt*/wct_pr_evt*.log | wc -l   # 47
grep -l 'staying with traditional vertex' pr_evt*/wct_pr_evt*.log | wc -l   # 0
grep -h 'determine_overall_main_vertex timing'    pr_evt*/wct_pr_evt*.log | wc -l   # 0
grep -h 'determine_overall_main_vertex_DL timing' pr_evt*/wct_pr_evt*.log | wc -l   # 47
grep -h 'check_switch_main_cluster'       pr_evt*/wct_pr_evt*.log | wc -l   # 0
grep -h 'rerank selected cluster'         pr_evt*/wct_pr_evt*.log | wc -l   # 47
grep -h 'rerank rejected'                 pr_evt*/wct_pr_evt*.log | wc -l   # 0

# --- sec.5: the run-to-run floor at HEAD (same binary, no rebuild between) ----
setarch x86_64 -R env SBND_WCT_LOGLEVEL=debug PR_JOBS=5 \
  ./run_pr_chain_batch.sh work-nuecc48-prod0803 work-pr37-repeatA data
setarch x86_64 -R env SBND_WCT_LOGLEVEL=debug PR_JOBS=5 \
  ./run_pr_chain_batch.sh work-nuecc48-prod0803 work-pr37-repeatB data
python3 pr37_repeat_cmp.py work-pr37-repeatA work-pr37-repeatB

# --- sec.2: the by-value / by-reference split ---------------------------------
cd /nfs/data/1/xqian/toolkit-dev/toolkit
grep -n 'ClusterVertexMap [a-z]\|Facade::Cluster\* main_cluster' \
     clus/inc/WireCellClus/NeutrinoPatternBase.h      # ~10 by-value carriers
grep -n 'ClusterVertexMap& \|Facade::Cluster\*& ' \
     clus/inc/WireCellClus/NeutrinoPatternBase.h      # 3 by-reference
# prototype: every write to the member, in one grep
cd /nfs/data/1/xqian/prototype-dev/wire-cell/pid
grep -rn 'map_cluster_main_vertices' src/ inc/

# --- sec.4: is the prototype reference edited where the audits read it? -------
cd /nfs/data/1/xqian/prototype-dev/wire-cell/pid
git diff --numstat a5fc0b9 HEAD                       # 26 files, +5833/-989
# ... and the same diff with whitespace, comments and instrumentation removed:
git diff -U0 -w --ignore-blank-lines a5fc0b9 HEAD -- <file> \
  | grep -E '^[+-]' | grep -vE '^(\+\+\+|---)' | sed -E 's/^[+-][[:space:]]*//' \
  | grep -vE '^$|^(//|/\*|\*|\*/)' | grep -vE 'std::cout|std::endl|Clock::now|chrono'

# --- sec.7: the knob count, DERIVED from cfg at HEAD (never summed from docs) --
cd /nfs/data/1/xqian/toolkit-dev/toolkit
git show HEAD:cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet | grep -nE '^ {4}[a-z_0-9]+ *= *(true|false),'
# read at HEAD, not the working tree: a concurrent session was committing into
# this tree during the round (HEAD moved ed414bd4 -> 29e8e452 mid-run).
```

---

## §0 Method, and how far to trust each row

Same tier convention as pr/28 §3b, carried through pr/29–36.

* Every source claim below was read **in both trees myself**, at the anchors
  printed. Anchors were re-validated at `29e8e452` after the concurrent session
  moved HEAD mid-round (§8.3); the four load-bearing ones in §2 all still hold.
* Every count that came from a **run** says so and prints the grep that produced
  it. Nothing is inferred from "the code says it should".
* **Rows inherited from a P-list keep that list's both-readings posture** (M15,
  escalation rule 4). §2 does not: it is a toolkit-internal self-inconsistency
  with no prototype-vs-toolkit judgement in it, so it carries a recommendation.
* This document proposes **no patch** and changes no code.

What this round did **not** do: it did not open any of the seven unaudited blocks
in §6. Those are named and ranked, not read.

---

## §1 The headline — SBND does not run the layer pr/32 verified

**Symptom.** pr/32 §2.5 verifies `compare_main_vertices_global` term for term;
§2.2 and §2.3 do the same for `compare_main_vertices` and `calc_conflict_maps`.
pr/32 §7 loose end 1 then asks the question that decides what any of it is worth:

> *"How often does the DL re-ranker accept on SBND? This decides whether §2.5's
> global-ladder verification describes production or a fallback. **Nothing else
> in §4.3 can be judged without it.**"*

Nobody measured it. It is one trace-level run and no code change.

**Measured, 48 nueCC events at `29e8e452`** (`work-pr37-trace48`):

| | events |
|---|---|
| DL vertex **accepted** (`"switching to DL vertex"`) | **47** |
| DL **declined** (`"staying with traditional vertex"`) | **0** |
| `determine_overall_main_vertex` entered (its own `m_perf` timing line) | **0** |
| `determine_overall_main_vertex_DL` entered | 47 |
| DL re-rank **selected** a candidate | **47** |
| DL re-rank **rejected** (score below the 4.0 floor) | **0** |
| no PR at all | 1 — evt **116962**, as in doc pr/28 |

Two independent signals agree: the DL log line fires 47/47, and the traditional
function's own timing line — emitted unconditionally under `m_perf`, which SBND
has on — fires **zero** times.

**What this reclassifies.** `TaggerCheckNeutrino.cxx:731-745` runs the
traditional path only `if (!flag_dl_changed)`. So on the SBND operating point
these never execute:

| never entered on SBND | pr/32 status |
|---|---|
| `determine_overall_main_vertex` (`NeutrinoVertexFinder.cxx:3941`) | §0 in-scope, audited |
| `examine_main_vertices` (`NeutrinoPatternBase.cxx:2467`) — its only caller is `:3980` | audited |
| `check_switch_main_cluster` / `_2` (`:3355`, `:3449`) | **Tier B** |
| `compare_main_vertices_global` (`:3427` is its only caller) | **§2.5, Tier A, verified term for term** |

**What it does NOT reclassify** — stated because the easy over-reading is that
pr/32 audited dead code. It did not. These all run inside `determine_main_vertex`
(per cluster, `TaggerCheckNeutrino.cxx:662`), which the DL path *requires* to
have run first:

* `compare_main_vertices` (`:2803`), `compare_main_vertices_all_showers`
  (`:2763`), `examine_main_vertices_local` (`:2780`), `calc_conflict_maps`
  (`:1055`, reached from `compare_main_vertices` at `:859`);
* and pr/32's four shipped knobs are **engaged**, measured on the same run —
  evt 388 reports `f1_reads=104 f1_fit_valid=104 f1_mean_cm=0.5542 f2_gate=4
  f3_cand=50 f4_flagged=98`. They live in `improve_vertex` / `examine_direction`
  / `determine_main_vertex`, all of which run in both paths.

**Parity is not broken by any of this.** The prototype does the same thing:
`NeutrinoID.cxx:202-208` is `if (flag_dl_vtx){ if (!determine_overall_main_vertex_DL()) determine_overall_main_vertex(); }`,
and the prototype's DL body (`NeutrinoID_DL.h:134-176`) carries its own
`swap_main_cluster`, its own short/high-dQ/dx **proton retag** (`:144-147`) and
its own **long-muon cleanup** (`:159-166`) — all three faithfully ported to
`NeutrinoVertexFinder.cxx:3868`, `:3882-3901`, `:3907-3916`. Nothing is lost
relative to the prototype; what is lost is the *audit's* coverage of what
actually runs.

**Two consequences worth carrying forward.**

1. **The 4.0 acceptance floor has never rejected anything.** 47 selections, 0
   rejections. pr/32 §10.6 flagged that `W_MAIN`, `W_CLEN` and the floor "were
   not derived on SBND"; it is now measured that the floor is **not binding** on
   this manifest, i.e. it is an untested guard rather than a tuned one. That
   belongs with doc pr/2 gap G1, as pr/32 said.
2. **Anything that only fires in the traditional path is unvalidated on SBND by
   construction** — including §2's defect, and including any future fix aimed at
   `compare_main_vertices_global`. A round that wants to touch that layer must
   first turn DL off, and that is a different operating point.

---

## §2 The by-value boundary — `determine_overall_main_vertex` silently discards its own work

**Symptom.** Two adjacent functions in one file carry the same two arguments with
**opposite conventions**:

| | `map_cluster_main_vertices` | `main_cluster` | writes propagate? |
|---|---|---|---|
| `determine_overall_main_vertex_DL` (`NeutrinoVertexFinder.cxx:3488-3491`) | `ClusterVertexMap&` | `Facade::Cluster*&` | **yes** |
| `determine_overall_main_vertex` (`:3941`) | `ClusterVertexMap` *(by value)* | `Facade::Cluster*` *(by value)* | **no** |

**Root cause.** Inside the by-value one:

```cpp
// NeutrinoVertexFinder.cxx:3980  -- erases from the map, and can itself swap
examine_main_vertices(graph, map_cluster_main_vertices, main_cluster, other_clusters);
// :3986 (dev chain) / :3996 (frozen chain) -- assigns a LOCAL copy
main_cluster = check_switch_main_cluster  (graph, map_cluster_main_vertices, main_cluster, …);
main_cluster = check_switch_main_cluster_2(graph, temp_main_vertex, max_length_cluster, main_cluster, …);
```

and the function returns only a `VertexPtr`. `TaggerCheckNeutrino.cxx` declares
`Cluster* main_cluster` at `:345` and **never reassigns it after `:451`**.

What makes it a defect rather than dead plumbing is that the swap is only *half*
discarded. `swap_main_cluster` (`NeutrinoPatternBase.cxx:2444-2465`) mutates
state that outlives the copy:

* `other_clusters` is `std::vector<Facade::Cluster*>&` **by reference the whole
  way down** — the old main cluster is pushed in, the new one erased;
* `Facade::Flags::main_cluster` is cleared on the old facade and set on the new.

So after a swap the flag and `other_clusters` say cluster **B** is main while the
caller's pointer still says **A** — and A is now *also* inside `other_clusters`.
`:745` then files **B's vertex under A's key**
(`map_cluster_main_vertices[main_cluster] = final_main_vertex`) and `:767` runs
`improve_vertex(*pr_graph, *main_cluster, final_main_vertex, …)` on a mismatched
(cluster, vertex) pair.

The prototype has no such boundary: `map_cluster_main_vertices` and
`main_cluster` are `NeutrinoID` **members**, so
`check_switch_main_cluster` → `swap_main_cluster` (`NeutrinoID.cxx:972`) is
persistent for the whole downstream chain by construction.

**Reach — measured, and it is the reason this is not the headline.** §1 shows
`determine_overall_main_vertex` is entered **0 times in 48 events** at the SBND
operating point. Neither `check_switch_main_cluster` nor `_2` announced a switch
(0 occurrences of either trace line). **On SBND production this defect cannot
fire.** It becomes live the moment `dl_weights` is empty or the DL path declines
— i.e. in exactly the configuration a future audit of that layer would use.

**Why it hid.** pr/32 §0 puts `determine_overall_main_vertex` in its function map
(row 14) but classifies `check_switch_main_cluster{,_2}` **Tier B** — signature
and constants compared, bodies not read end to end. A parameter's `&` is
precisely the kind of thing a signature comparison passes over: the *names* and
*types* match the prototype's members; only the binding differs.

**Fix (recommended, not applied).** Change `determine_overall_main_vertex`'s two
parameters to `ClusterVertexMap&` and `Facade::Cluster*&`, matching its DL twin
one function above it. Byte-identical when it does not fire — which, per §1, is
every SBND event on this manifest — so the gate is cheap and the knob question
does not arise. **Do not** make this change without first re-reading §6 row 1:
the same round should decide whether that layer is worth keeping at all.

### §2.1 The class sweep — 10 by-value sites, exactly one is live

The split is a class, not one site: `NeutrinoPatternBase.h` has **3**
by-reference carriers against **~10** by-value ones. Each was resolved by asking
(a) does the prototype's body write the member, and (b) does anything downstream
read that write. The prototype's complete write list is five lines
(`grep -rn map_cluster_main_vertices src/ inc/`), which is what makes the sweep
short.

| toolkit site | prototype writes the member? | verdict |
|---|---|---|
| `determine_overall_main_vertex:808` | **yes** — via `examine_main_vertices` (`NeutrinoID.cxx:455`, `:514` erase) **and** `check_switch_main_cluster` (`:972` swap, via `swap_main_cluster`) | **LIVE defect**, unreachable at the SBND operating point (§1) |
| `examine_main_vertices:803` | yes | takes `Facade::Cluster*&` and `ClusterVertexMap&` — **correct**; its writes die one frame up, at the row above |
| `check_switch_main_cluster:806`, `_2:807` | yes | **correct as written** — each *returns* the new `main_cluster`; the loss is the caller's binding, not theirs |
| `deghost_segments:842` | **no** — `NeutrinoID_deghost.h:187-188` only *reads* the map; the erase at `:38` is in `deghosting()`, and the toolkit's `deghosting:843` is **by reference** | **inert** (a wasted copy) |
| `shower_clustering_in_other_clusters:830`, `examine_shower_1:831`, `examine_showers:832`, `id_pi0_with_vertex:833`, `id_pi0_without_vertex:834`, `shower_clustering_with_nv:835` | **no** — the only prototype reference in that file is a read-only iteration, `NeutrinoID_shower_clustering.h:1454` | **inert** (6 wasted copies) |

So the class collapses to one row. The six shower-clustering copies are a
performance smell, not a correctness one, and are **not** proposed for change:
a `ClusterVertexMap` is small and the copy is provably unobservable.

---

## §3 doc pr/33's owner filter was never implemented

Checked because §1 made it worth asking which rounds actually landed. Of the
eight audits that ran an owner filter, **seven produced toolkit code and one did
not**:

| doc | filter result | implementation section | knobs live in `sbnd/wct-pr-perevt.jsonnet` at HEAD |
|---|---|---|---|
| pr/29 | 12 → 5 | §11, §12 | 3 (`steiner_terminal_wire_tol=1`, `steiner_terminal_adjacent_slice`, `steiner_edge_charge_forward_dead_mix`) |
| pr/30 | 14 → 4 | §12 | 5 present, **1 ON** (`oov_prototype_parity`) |
| pr/31 | 15 → 9 | §11, §12 | 7 present, **5 ON** |
| pr/32 | 12 → 4 | §11 | **4 ON** |
| **pr/33** | **14 → 5, "8 knobs"** | **none — the doc stops at §10** | **0 — none of the names exists anywhere in `clus/`** |
| pr/34 | 14 → 5 | §11 | **5 ON** (`pf_*`) |
| pr/35 | 14 → 4 | §11 (`29e8e452`) | **1 ON** (`kine_shower_pdg_live`) + one unconditional change |
| pr/36 | 13 → 7 | none yet — filtered 2026-08-04, by design | 0 |

`grep -rn 'shower_pdg_from_start_segment\|shower_pdg_exact_muon_test\|shower_less_id_tiebreak' clus/`
returns nothing. pr/33's §11 knob table (`:958-962`) names the sites and the
change for each; no code was written. pr/36 is *expected* to be at zero — it was
filtered the same day. **pr/33 is the outlier**, and it is the largest single
block of accepted-but-unimplemented work in the series: five findings the owner
kept, in EM shower clustering, which is upstream of every shower quantity in
`T_kine` and the Bee mc tree.

Not proposed here. Named so it is scheduled rather than forgotten.

---

## §4 The prototype reference is a branch, and pr/28–32 never pinned it

**Symptom.** `prototype_base` → `prototype-dev/wire-cell`, submodule `pid` on
branch **`port`** at **`53ca938`**. Its merge-base with `master` is
**`a5fc0b9`**, and the branch carries **+5833/−989 across 26 files**. Every file
docs pr/28–32 cite is in that set. Docs pr/34, pr/35 and pr/36 name the commit;
**pr/28–32 do not** — their Repro blocks say only
`prototype_base -> …/wire-cell (package pid/, WCPPID)`.

**The alarm is narrow, and measuring it is what narrows it.** Filtering the diff
for whitespace, blank lines, comments and `std::cout` / `chrono` instrumentation
leaves the algorithm files nearly untouched:

| prototype file | raw diff | substantive lines | audited by |
|---|---|---|---|
| `PR3DCluster_path.h`, `…_trajectory_fit.h`, `…_multi_dQ_dx_fit.h`, `NeutrinoID_final_structure.h`, `ImprovePR3DCluster.cxx`, `ToyFiducial.cxx`, `CalcPoints.cxx`, `PR3DCluster_crawl.h` | +12/−5 … +153/−61 | **0** | pr/28, pr/29, pr/32 |
| `PR3DCluster_steiner.h` | +184/−110 | **2** | pr/29 D1/D3/D6/D12 |
| `PR3DCluster_graph.h` | +44/−27 | **2** | pr/29 D3 |
| `ProtoSegment.cxx` | +14/−6 | **3** | pr/31, pr/36 |
| `NeutrinoID_improve_vertex.h` | +99/−92 | **4** (brace-only) | pr/28 §3, pr/32 P3 |
| `PR3DCluster_multi_track_fitting.h` | +94/−224 | **9** | pr/28 §1, §3b |
| `PR3DCluster_dQ_dx_fit.h` | +201/−96 | **22** | pr/28 §4 |
| `NeutrinoID_track_shower.h` | +66/−12 | **33** | pr/31, pr/32 |
| `NeutrinoID_proto_vertex.h` | +217/−70 | **92** — every one `chrono` timing | pr/30 |
| `NeutrinoID.cxx` | +170/−34 | **114** — timing + `print_segs_info` + §4.1 item 6 | pr/32 |

**So the pr/28–32 citations survive.** The 92 and the 114 are instrumentation,
and `PR3DCluster_multi_track_fitting.h`'s −224 is deleted commented-out debris.
This is deliberately *not* written as "every prototype citation is suspect" — a
broad alarm here would be a wrong positive, and pr/36 §10.13 is the precedent for
what those cost.

### §4.1 The six candidate live edits, triaged

Each was opened. **Four are not live code at all, and two are real prototype bugs
the toolkit is structurally immune to** — so nothing in this section is a toolkit
defect.

| # | `port`-branch edit | what it actually is | toolkit |
|---|---|---|---|
| 1 | `PR3DCluster_graph.h:528` `if (ref_point_cloud != 0)` | guards a **`std::cout`** — not an algorithm change | n/a |
| 2 | `PR3DCluster_graph.h:1149` `if (edge.second)` | re-indentation of an existing pair of guards; the *new* one at `:1149` has a **commented-out body**, so the following `if (edge.second){…}` becomes it — inert (`if(A) if(A){X}` ≡ `if(A){X}`), but a dangling-`if` shape nobody should port | n/a |
| 3 | `PR3DCluster_steiner.h:585` `temp_map_new_old_indices = map_new_old_indices;` | populates a member whose **only reader is a commented-out line** in `apps/wire-cell-prod-stm-port.cxx:901` | n/a |
| 4 | `ProtoSegment.cxx:1286` `particle_score = 0;` | **a real upstream bug**: `do_track_pid`'s reset-before-return clears `flag_dir` and `particle_type` but left `particle_score` stale | **immune** — `segment_do_track_pid` (`PRSegmentFunctions.cxx:1454`) *returns* `tuple<bool,int,int,double>` instead of mutating members, so there is no member to leave stale |
| 5 | `ProtoSegment.cxx:1764` `if (fit_index_vec.size()==0) fit_index_vec.resize(…)` (un-commented) | **a real upstream bug**: `.at(nbreak_fit)` on a possibly-empty parallel vector throws `std::out_of_range` | **immune** — the toolkit keeps one `std::vector<PR::Fit> m_fits` (`PRSegment.h`), not six parallel vectors, so the desync cannot exist; the accessors bounds-check explicitly (`PRSegment.cxx:20-35`) |
| 6 | `NeutrinoID.cxx:362-369` null-guard + early return in `determine_overall_main_vertex` | **a real upstream bug**: `map_cluster_main_vertices[main_cluster]` with `operator[]` *inserted a null entry* and the function proceeded with `main_vertex == nullptr` | present — `NeutrinoVertexFinder.cxx:4005-4014` uses `find()` and returns `nullptr`. Independent corroboration for §2: someone hit trouble in this exact function from the other side |

**Fix (process, one line per doc).** Add the `pid` SHA to the Repro block of
pr/28, pr/29, pr/30, pr/31 and pr/32, as pr/34–36 already do. The audits are
sound; the provenance is undocumented, and undocumented provenance is what turns
a sound audit into an unreproducible one two years from now.

**Verification.** The triage above is a source read of six hunks; the
substantive-line counts are reproducible with the filter printed in the Repro
block.

---

## §5 Determinism — the sweep at HEAD is clean, and the floor is re-measured

pr/28 §10, §11, §14 and §15 did the heavy work here. Two things were checked
this round: whether the six knob rounds since have introduced anything new, and
whether the zero floor still holds at a HEAD those rounds have moved.

### §5.1 Source sweep at `29e8e452` — nothing new to fix

* **Raw BGL traversal.** Every live `boost::out_edges` / `boost::edges` /
  `boost::vertices` outside the `ordered_*` / `sorted_out_edges` helpers is
  either a helper definition (`PRGraphType.cxx:7,19`,
  `PRTrajectoryView.h:155`), a debug dumper (`PatternDebugIO.cxx:47`), or JSON
  export (`Facade_Util.cxx:675,695`). Everything else in the grep is a comment.
* **`std::unordered_set<SegmentPtr>` / `<VertexPtr>`** at `PRShower.cxx:495-585`
  — the class that bit round 9 via `TrajectoryView::edges()`. All three sites are
  **membership-only** (`insert().second`, `.size()`) and copy into an
  `IndexedSegmentSet` / `IndexedVertexSet` on return. Compliant.
* **`clustering_deghost.cxx:233`** — an argmax over
  `unordered_map<const Cluster*,int>`, with an explicit insertion-index
  tie-break and the reason in-code. Correct.
* **`SteinerGrapher.cxx:1010`** — a raw `boost::edges(base_graph)` feeding a
  strict-`<` per-terminal-pair best-edge selection, so a tie keeps the first edge
  seen. Not a nondeterminism: `Graphs.h:22-27` declares the edge list **`vecS`**,
  i.e. insertion-ordered. Recorded as a **fidelity** note — which edge wins a tie
  may differ from the prototype's (whose `setS` edge-descriptor order is
  address-keyed, pr/29 D6) — not as a determinism one.

### §5.2 The floor at HEAD

pr/28's zero floor was measured at `026a7501` / `397b1517`. Since then pr/31
§12, pr/32 §11, pr/33 (doc only), pr/34, pr/35 and pr/36 have landed, and **no
repeat-run identity check existed at current HEAD.** Two arms, same binary
(`libWireCellClus.so` mtime `19:47:37`, unchanged before, between and after both
arms), `setarch x86_64 -R` (M4), 48 nueCC events, `PR_JOBS=5`, both 48/48 `rc=0`:

| artifact class | compared | differ |
|---|---|---|
| `mabc-pr.zip` + `pctree-pr-evt<ID>.tar.gz` — member **content** hashes (`hash_archive.py`, never raw bytes — M2) | 96 | **0** |
| `nusel-evt<ID>.tsv` — exact bytes | 48 | **0** |
| `tracking-pr.root` — every leaf of every tree, exact-bit, via `uproot` | **59 890** | **0** |

**FLOOR = 0.** The zero pr/28 round 9 established survives all six knob rounds.
`work-pr37-repeatA` vs `work-pr37-repeatB`, comparator `pr37_repeat_cmp.py`
(committed alongside this doc).

**One methodological trap, recorded because it manufactured a regression.** The
comparator's first version hashed ragged leaves — the `vector<vector<T>>`
branches — with `repr()`. A numpy **object** array holding python objects reprs
as `<… at 0x7f…>`, i.e. a *heap address*, so two bit-identical runs hashed
differently. That reported **235 differing leaves on 47 of 48 events**, confined
to exactly the five nested branches of `T_proj_data` (`channel`, `time_slice`,
`charge`, `charge_err`, `charge_pred`) — which is precisely the family pr/28 §14
drove to zero, so it read as a plausible regression in a plausible place. Direct
value-by-value checks on evts 388, 10550 and 239794 showed **every one of the
13 821 / 4 466 / 13 709 per-hit values equal, in exact order**; the comparator
was wrong, not the toolkit. Fixed by descending to numeric leaves and hashing
their bytes (`_walk`), never `repr`. The rule this leaves behind: **a comparator
is an instrument and needs its own calibration** — an A-vs-A self-test does not
catch this class, because the same process yields the same addresses.

---

## §6 The unaudited blocks the five documents name, ranked

Not discoveries — the docs' own admissions, ranked once, with why each matters.

| # | block | named by | why it ranks here |
|---|---|---|---|
| 1 | **`examine_structure_*` / `NeutrinoStructureExaminer.cxx`** | pr/32 §0, §9 | ~700 lines per side, called from `determine_main_vertex` — which **does** run on SBND (§1) — and pr/32 says plainly it *"can move vertices before the final `examine_direction` sees them"*. It is the owner's original "vertex is a bit off" question and the largest wholly unread block in the live vertex chain |
| 2 | **Base-graph builders** — `Create_graph` / `Establish_close_connected_graph` ↔ `find_graph("ctpc_ref_pid")` | pr/29 §0, §13.2 | *"Everything in §4 is about inputs to the solver; the biggest input of all was not opened."* §4.1 items 1–2 land in these functions |
| 3 | **`improve_maps_no_dir_tracks`** | pr/31 §1, §9 | 331 vs 474 lines — the largest unread pair — and it contains **8 of P1's 11 sites**. P1's *guard* claim was verified per site; the surrounding branch conditions were not compared |
| 4 | **`examine_direction`'s PDG-reassignment ladder** (`:1232-1348`, `:1529-1548`) | pr/32 §9 | Tier B, 428 vs 632 lines, and `examine_direction` runs in **both** vertex paths, so unlike §1's casualties it is live on SBND. pr/32 says *"a divergence could be hiding there"* |
| 5 | **`clustering_points_segments`** (317 lines) and pr/30's `update_association` coordinate question | pr/31 §0; pr/30 §9 | the 2-D association machinery that P1's reach depends on; *"one instrumented run settles it"* and nobody ran it |
| 6 | **Downstream consumers of `steiner_graph` / `steiner_pc`** | pr/29 §13.2 | one was opened (`get_two_boundary_wcps(2)`) and found a deliberate, in-code-documented substitution. *"That one was found by looking; nobody has looked at the rest."* |
| 7 | **The retile step** (`RetileCluster` vs `Improve_PR3DCluster_2`) | pr/29 §0, §13.2 | D3's dismissal rests on an unaudited assumption about it. De-risked by §4 (`ImprovePR3DCluster.cxx` substantive diff = **0**) but still unread |

Also unclosed and cheap: pr/29's **D8 and D10 reaches** are dismissed as *forced*,
which is a statement about *why* the toolkit differs, not about *how much*.

---

## §7 The measurement debt — the count, derived

pr/32 §11.8 called the stale valfast/1000 baseline *"past the point where 'stale
baseline' is a footnote"* and put **ten** knobs on it. That number is now low,
and it must not be obtained by adding up the per-doc sections — per doc 68 the
SBND operating point lives **only** in cfg, and `clus.jsonnet`'s parameter
defaults are the *legacy/off* values, not the operating point. The derivation is
from `wct-pr-perevt.jsonnet` at HEAD:

| doc | knobs present | **ON at HEAD** |
|---|---|---|
| pr/29 | 3 | **3** — `steiner_terminal_wire_tol=1`, `steiner_terminal_adjacent_slice`, `steiner_edge_charge_forward_dead_mix` |
| pr/30 | 5 | **1** — `oov_prototype_parity` (`fit_exclusion`, `graph_endpoint_strict` false; two `null`) |
| pr/31 | 7 | **5** — `cont_muon_dir3_30cm`, `track_comp_empty_abstain`, `shower_topo_reset`, `reclass_preserve_4mom`, `dir_track_median_local` (`shower_topo_proto_dir`, `examine_showers_vertex_by_index` false) |
| pr/32 | 4 | **4** — `vertex_dir_use_fit_point`, `shower_traj_recheck_parity`, `main_vertex_require_descriptor`, `main_vertex_candidate_flag` |
| pr/33 | 0 | 0 — never implemented (§3) |
| pr/34 | 5 | **5** — the `pf_*` family |
| pr/35 | 1 | **1** — `kine_shower_pdg_live`, plus one **unconditional** change (the segment `cal_kine_charge` cache reuse) |
| | | **19 knobs ON, plus one unconditional change** |

Every one of them was gated on **48 nueCC events**, and several on a *different*
baseline from the others. Nineteen is not a footnote.

**How the table was built, so the number can be re-derived rather than trusted.**
The *total* is a mechanical count of boolean TLAs set away from the C++ default
in `wct-pr-perevt.jsonnet` at HEAD (the grep is in the Repro block). The
*per-doc split* is assigned **by knob name**, from each document's own §11/§12
implementation section — not by scanning the `// doc pr/NN` comments, which is
unreliable: a comment's attribution carries forward to every TLA below it until
the next one, so a naive scan credits pr/23 with 30 TLAs and pr/35 with 18. If a
re-run of the grep gives a different split, trust the knob names.

Carried forward from the five documents, one line each, so they are in one place:

* **pr/29 §13.1** — D1+D12 has never been separated from D2; the pi0 question
  (evt 388 lost one) is open; the 24-starved-clusters number's scaling is
  partially answered by §14 and not fully.
* **pr/28 §15.9** — round 8's two missing tests; **`work-tfix388-r9` must be
  hand-added to the next retire round's `PROTECTED` list** (there is no automatic
  protection).
* **pr/30 §7.1/§7.2** — the `walk_history` asymmetry between the two
  `proto_extend_point` calls; and the `init_first_segment` main-cluster
  flag-vs-pointer warning, which §2 now supplies a mechanism for.
* **pr/31 §7.3/§7.4** — `segment_cal_4mom`'s dead `MIP_dQdx` parameter;
  `kslike_compare` divides by zero on an all-zero window. **§7.1's anchor has
  drifted** — `PRSegmentFunctions.cxx:2028-2029` is now an `(apa,face)` angle
  cache, so that loose end must be re-anchored before it can be answered.
* **pr/31 §12.9, pr/32 §10.6** — F7/F8 and P2's SBND-tuning concern deliberately
  not moved.
* **pr/32 loose ends 5 and 6** — three coexisting formulations of "which end of
  the segment touches this vertex"; the `else if` that makes
  `angle_beam < 45 && max_angle < 70` unreachable in **both** trees (M15 applies).
* **The porting dictionary still has no section for the pr/30, pr/31 or pr/32
  stages** — three audits in a row. pr/29's was filled (§13.4); the others'
  divergences are undocumented by construction, which is what keeps escalation
  rule 4 binding on all of them.

---

## §8 Small, named, cheap

**§8.1 Raw `std::cout` on the production PR path.** Eight at HEAD — five in
`TaggerCheckNeutrino.cxx` (`:670`, `:774`, `:778`, `:788`, `:806`:
`"After first round of main cluster PR"`, `"After improve vertex:"`,
`"After shower clustering :"`, `"After examine direction: "`,
`"After shower clustering with NV: "`, each followed by a `print_segs_info`) and
three in `NeutrinoPatternBase.cxx`. CLAUDE.md §2 mandates `Aux::Logger` /
`SPDLOG_LOGGER_DEBUG`. These print unconditionally, at any log level, once per
event, and bypass the log-level plumbing every other diagnostic in this chain
respects. Reported, not fixed — an unrelated defect does not ride along in
another change.

**§8.2 Log lines tear mid-word** in the trace capture, e.g.
`PR31AUDIT … examine_showerusters=91>, 40562 points …` — two loggers interleaving
on one line. Known (`project_wct_log_line_tearing`); noted because it makes
counter lines unsafe to parse with a strict regex. Parse per-key, not per-line.

**§8.3 The tree moved under this round.** A concurrent session committed
`29e8e452` at 19:53:51, between the M1 arm's start and its finish, moving HEAD
from `ed414bd4`. The binary was installed at **19:47:37**, before both runs, and
did not change afterwards — so every number here is from one binary. All §2
anchors were re-validated at `29e8e452` and none had drifted. Recorded because
"same binary" is a premise of §5.2 and it deserves evidence rather than
assumption.

---

## §9 What is NOT claimed

* **The seven blocks in §6 were not opened.** They are ranked from what the five
  documents say about them, not from reading them.
* **§2's fix was not written, built, run or gated.** The recommendation is a
  source-level argument; "byte-identical when it does not fire" follows from §1's
  0/48, not from a gate.
* **§1 is 48 nueCC events, not valfast.** *"DL accepts on SBND"* is a statement
  about this manifest and this operating point. A different `dl_vtx_cut`, a
  different weights file, or cosmics rather than nueCC could all change it — and
  the traditional path is exactly the fallback that would then start running,
  carrying §2 with it.
* **§1 does not say pr/32 audited dead code.** Four of its Tier-A functions run
  in production and its four knobs are engaged; the reclassification is confined
  to the four rows tabulated.
* **§4 does not clear the `port` branch.** It shows the *audited* files' diffs
  are instrumentation plus six hunks, all six triaged. `wire-cell-prod-nue-port.cxx`
  (+3384, the app the SBND runners invoke) was **not** read, and the 26-file diff
  was not read line by line.
* **§4's immunity claims are structural, not measured.** Items 4 and 5 argue from
  the toolkit's data model; no event was run to demonstrate the prototype bug
  firing.
* **§5.1 is a source sweep, not an N-run identity test per site.** §5.2 is the
  measurement, and it covers the aggregate, not each container: a zero floor
  says no *observed* artifact moved, not that every pointer-ordered traversal is
  provably unobservable.
* **§5.2 is two runs, 48 events, one binary, one machine.** Two arms cannot
  bound a rare nondeterminism; pr/28 §14.5 found its residual only after a
  larger family was removed. The comparator is also **order-sensitive**, so a
  pure permutation would have been reported as a difference — moot at a floor of
  zero, but it is the right reading of the number.
* **§5.2 does not cover the display dump.** pr/28's zero was measured on
  `calib-pr-evt<ID>.json` (185 421 leaves); this round's arms do not emit it, so
  the two zeros are on overlapping but different artifact sets.
* **§7's count is knobs ON at one HEAD.** It is not a claim that all nineteen
  interact, nor that a valfast run would move any of them.
* **Nothing here is implemented.** No toolkit C++ or jsonnet was changed by this
  document; `git -C toolkit status --porcelain` carried only the concurrent
  session's files throughout.
