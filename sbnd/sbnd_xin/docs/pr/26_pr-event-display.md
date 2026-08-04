# pr/26 — SBND PR event display (SBND evt 18255/388)

A dedicated event display for validating and improving the neutrino
pattern-recognition (PR) code, ahead of the tuning campaign over the 572
valfast events. **No PR algorithm changes** in either stage.

* **Stage 1** (§1–§6) — read-only viewing plus the dump that feeds it:
  geometry, the PR graph, the Steiner skeleton, the six 2-D charge panels.
* **Stage 2** (§7) — the **particle flow**, clickable to highlight one particle
  in all nine panels, and the **event features** that decide selection
  (reco Enu, nue/numu/cosmic scores, per-particle energies).
* **Stage 3** (§8) — **the cosmic answer**: `cosmict_flag` and its ten per-test
  flags replace two fields that were not it, and 22 never-computed fields come
  off the panel. Read §8 before trusting any "cosmic" number here.

Two defects were found on the way and are written up in §5. §5.1 has since been
fixed (toolkit `4c02b679`); §5.2 remains reported-not-fixed, per CLAUDE.md's
rule on unrelated bugs found mid-task. A third — the single-photon tagger's
verdict being discarded — is reported in §8.2 on the same terms.

**Corrections in §8 to what §7 says**: `cosmic_flag`'s polarity is the opposite
of what §7 states, and `cosmict_score` is never computed at all.

## Repro

```bash
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
cd $SX

# 1. the display's input: the PR chain with the (default-OFF) pr_display stage.
#    ql_root is the owner's protected campaign -- READ ONLY; out_root is fresh.
#    Stage 1 arm: work-prdisp-388.  Stage 2 arm: work-prdisp-388-pf.
#    Stage 3 arm (what is served now) -- 7 events so the multi-event paths are
#    testable; evt 388 is the one displayed:
PR_EXTRA_STAGES=pr_display PR_JOBS=6 \
  ./run_pr_chain_batch.sh work-nuecc48-prod0803 work-prdisp-cosscan2 data \
      388 10550 111412 122660 137238 163543 172230
# -> work-prdisp-cosscan2/pr_evt388/calib-pr-evt388.json   (1.9 MB)
#    plus mabc-pr.zip beside it, whose data/0/0-mc.json is the particle flow

# 2. serve it.  Pass the path EXPLICITLY: serve_pr_display.sh's default glob is
#    ../work-prdisp-*/pr_evt*/calib-pr-evt*.json, and FOUR arms now yield the
#    label "evt388", so all but one would be silently shadowed.
./pr_display/serve_pr_display.sh 5017 work-prdisp-cosscan2/pr_evt388/calib-pr-evt388.json
# laptop:  ssh -L 5017:localhost:5017 wcgpu1.phy.bnl.gov
#          http://localhost:5017/pr_display_viewer
```

Toolkit `apply-pointcloud` (this round's commit); wcp-porting-img `main`.
Event 18255/388 is `valfast/events-nuecc48.txt:1`; its `main_id 23` bundle is
labelled `nu-candidate` by `nusel-evt388.tsv`.

---

## 1. Why a new dump rather than reading what exists

Three producers already carry pieces of what the display needs. None carries
the set, and one of them is broken:

| want | Bee `mabc-pr.zip` | `tracking-pr.root` | verdict |
|---|---|---|---|
| fit trajectories | `track_fit` layer, flat x/y/z/q arrays | `T_rec_charge` rows | **no segment grouping** in either, so no polylines |
| track/shower points | `shower_track` layer | — | ok |
| PR vertices | `vertices` layer | `flag_vertex` | ok |
| Steiner skeleton + terminals | — | — | **absent from both** |
| 2-D measured charge | — | `T_proj_data` | was **empty except `cluster_id`** when this display was designed — §5.1, fixed since |

So the display would need three readers, one of which returns nothing. Instead
the PR chain gains a `pr_display` stage writing the **union** into one
self-contained JSON per event — the same shape as QLMatching's `calib_dump`
that feeds `ql_scan`.

## 2. What ships

**`clus/{inc/WireCellClus,src}/PrDisplayDump.{h,cxx}`** — a new
`IEnsembleVisitor` + `IConfigurable`, registered as `PrDisplayDump`. Plain
JSON via `Persist::dump`, no ROOT (precedent: `clus/src/BeeSink.cxx`). It reads
and mutates nothing.

`calib-pr-evt<ID>.json`:

```jsonc
{
  "meta":         { runNo, subRunNo, eventNo, nch[3], base[3], nticks,
                    nticks_per_slice[{apa,face,n}], apa_faces[],
                    dQdx_scale, dQdx_offset, length_unit:"cm",
                    charge_transform:"none" },
  "segments":     [ {id, cluster_id, particle_id, flag_shower, dirsign,
                     is_main_cluster, shower_id,
                     points:[{x,y,z,dQ,dx,pu,pv,pw,pt,apa,face,reduced_chi2,rr}]} ],
  "vertices":     [ {id, cluster_id, is_main, degree, fit_distance, fit:{...}} ],
  "main_vertex":  {x,y,z,cluster_id} | null,
  "track_shower": {x[],y[],z[], flag_shower[], cluster_id[], particle_id[]},
  "steiner":      [ {cluster_id, is_main_cluster, x[],y[],z[], flag_terminal[]} ],
  "proj":         [ {apa,face,plane,nticks_per_slice,
                     wire[],slice[],charge[],charge_err[],charge_pred[],
                     flag[],cluster_id[]} ],
  "dead":         [ {apa,face,plane,wire, t0,t1 /*ticks*/, s0,s1 /*slices*/} ]
}
```

Sources: `grouping.get_pr_graph()` (segments, vertices, associated points),
`TrackFitting::get_fitted_charge_2d()` (2-D cells),
`cluster->get_pc("steiner_pc")` (skeleton + terminals),
`grouping.get_all_dead_chs()` (dead bands).

**Both Steiner toggles are free.** `steiner_pc` *already* carries a persisted
`flag_steiner_terminal` int column (`clus/src/SteinerGrapher.cxx:1036-1040`),
index-aligned with the reduced-graph vertices and present in the pctree
metadata. No clustering-stage change, no pctree schema change, no second gate.

**`cfg/pgrapher/experiment/sbnd/clus.jsonnet`** — one new `cm_by_name` entry
`pr_display`, next to `tracking_visitor`. Only instantiated when named in
`pipeline_names`, so with the name absent the compiled config is unchanged.

**`sbnd_xin/pr_display/`** — `pr_display_viewer.py`, `serve_pr_display.sh`
(port 5017), `README.md`. Structure follows `nusel_display/nusel_scan_viewer.py`
(which already builds the three X-Y/Y-Z/X-Z projections and is Bokeh 3.9).

**`sbnd_xin/run_pr_chain_batch.sh`** — `PR_EXTRA_STAGES`, empty by default,
appended to the pipeline string. Empty ⇒ the driver's behaviour and every
output are unchanged.

**Tests** — `clus/test/doctest_clus_knob_defaults.cxx` gains a `PrDisplayDump`
case pinning its factory registration and its five defaults. Revert-proven:
flipping `m_proj_charge_min` to 1 fails it.

### Deliberate divergences from the two existing producers

The PR-graph walk is a fork-by-duplication of
`MultiAlgBlobClustering::fill_bee_points_from_pr_graph` and
`SbndPrMagnifyTrackingVisitor::write_{proj_data,t_rec_data}` (M10 — both stay
untouched). Marked `DELTA` in the source:

- **segments stay grouped.** Both originals flatten every fit point into one
  row list, which is what loses the polyline.
- **`proj` is grouped by `(apa, face, plane)`, with the per-APA wire index**,
  because the six panels are per-TPC-per-plane. `write_proj_data` groups by
  cluster id and emits the concatenated global channel.
- **charges go out raw.** The Bee layers pre-bake `q*scale + offset`; both
  constants are recorded in `meta` instead, so dQ/dx stays available.
- **dead regions carry both ticks and slices.** The Magnify writer leaves that
  mismatch to its reader (`T_bad_ch` in ticks, `T_proj_data` in slices).

Kept from the originals because they are load-bearing: ordered-map
accumulation and sorted cluster-id selection (determinism — never iterate a
pointer-keyed container); `PR::ordered_nodes`/`ordered_edges` rather than raw
`boost::vertices`/`edges`; per-point `(apa,face)` from `PR::Fit::paf` with the
APA-0 fallback; and **fractional** `pu/pv/pw` (doc pr/7 §1).

## 3. The display

Row 1 — X-Y, Y-Z, X-Z with the active volume and cathode. Row 2 — six panels,
two columns (TPC 0 | TPC 1) × (T-U, T-V, T-W): fitted 2-D charge as a heat map
with the best-fit trajectory over it and dead bands shaded.

Layers, each a toggle: track fit · shower pts · track pts · **steiner** ·
**terminals** · vertices · dead. `zoom` reframes all nine panels to ±*half*
(default 30 cm) about a centre that starts at the neutrino vertex and can be
typed as (x, y, z). The 2-D panels follow the same centre via the fitted points
inside that sphere — the viewer loads no wire geometry, only the JSON.

## 4. Verification

| gate | result |
|---|---|
| **Compiled-config, SBND, `pr_display` absent** | `wcsonnet` output **byte-identical** to HEAD for the full production pipeline |
| **Compiled-config, SBND, `pr_display` present** | the `PrDisplayDump:pr` node appears, `output_filename=…/calib-pr-evt388.json` |
| **Compiled-config, uBooNE** (`qlport/scripts/compile_ub_cfg.sh`) | **byte-identical** |
| **PR output unmoved** — `hash_archive.py` member hashes vs the protected `work-vfnuecc48-prod0803/pr_evt388/` | `mabc-pr.zip` `c285c96d…` (7 members) **PASS**; `pctree-pr-evt388.tar.gz` `3ea67506…` (425 members) **PASS**; `nusel-evt388.tsv` identical |
| **Cross-check vs the independent Bee layers** | fit points 816 + vertices 127 = **943 = `track_fit` layer = `T_rec_charge` entries**; `track_shower` **7121 = 7121**, of which shower **7095 = 7095**; vertices **127 = 127**, main-flagged **1 = 1** |
| **Fit-vs-charge alignment** (pr/7 §2 method, in index space) | charge-weighted ⟨d⊥⟩ = +0.008 / −0.057 / −0.067 (TPC0 U/V/W) and −0.027 / +0.003 / +0.030 (TPC1 U/V/W) index units, rms ≈ 1 — the drawn track sits on the charge in every panel |
| **Determinism**, two runs under `setarch -R` | every section byte-identical **except `proj.charge_pred`** — §5.2 |
| `./build/clus/wcdoctest-clus` | 71 cases / 809 assertions **pass** |
| Freshness (M1) | `local/lib/libWireCellClus.so` newer than the last source edit before every claim above |
| Viewer | `py_compile` ok, `bash -n` ok, headless document build populates **28 of 28** CDS (28 440 rows), server returns 200 |
| Protected dirs (M13) | nothing written under `work-vfnuecc48-prod0803/` or `work-nuecc48-prod0803/` |

## 5. Two defects found on the way — 5.1 FIXED, 5.2 open

### 5.1 `T_proj_data` has only one branch — REGRESSION from upstream `5f684887`, **FIXED**

> **Corrected and fixed 2026-08-03.** An earlier revision of this section said
> the toolkit "generates no ROOT dictionaries at all" and that the breakage was
> longstanding. **Both are wrong.** It generated one, and this is a dated
> regression. Fixed by restoring `root/dict/LinkDef.h`; verification below.
> The superseded reasoning is kept only where it is still a useful warning.

**Symptom.** `tracking-pr.root` contains a `T_proj_data` tree with only
`cluster_id`. `channel`, `time_slice`, `charge`, `charge_err` and `charge_pred`
are absent, so the Magnify-tracking 2-D view has no measurement to draw.

**Root cause.** `TTree::Branch` refuses to create a branch for an STL collection
with no *compiled* `CollectionProxy`. Until recently `root/dict/LinkDef.h`
supplied exactly that, on its first three lines:

```cpp
#pragma link C++ class vector < vector<int> > +;
#pragma link C++ class vector < vector<float> > +;
#pragma link C++ class vector < vector<double> > +;
```

That file was **deleted by upstream commit `5f684887` ("moved to standalone
wire-cell-sbnd-reco1", Tue Jul 28 2026)**, which moved the SBND reco1 art-file
sources out of the toolkit and took `root/dict/` with them. The three `std`
pragmas were collateral: they serve `root/`'s own three Magnify `T_proj_data`
writers and have nothing to do with reco1. The commit reached
`apply-pointcloud` in the 2026-08-03 master merge.

The dated evidence is unambiguous:

| file | written | `T_proj_data` branches | vs. merge |
|---|---|---|---|
| `work-r1ql-f2/nusel_evt12/tracking-stm.root` | Jul 30 19:35 | **6**, with data | before |
| `work-vfnuecc48-prod0803/pr_evt388/tracking-pr.root` | Aug 3 12:00 | **1** | after |

Both were written by the same ROOT 6.32.02 (`fVersion 63202`), by
byte-identical branch code, from the same tree. The only variable is the
dictionary. ROOT says so itself, in `stdout.log`:

```
Error in <TTree::Branch>: The class requested (vector<vector<int> >) for the
branch "channel" is an instance of an stl collection and does not have a
compiled CollectionProxy. Please generate the dictionary for this collection
(vector<vector<int> >) to avoid to write corrupted data.
```

`cluster_id` survives because plain `vector<int>` is one of the collections
ROOT ships a proxy for — which is exactly why it is the one branch left.

**Why it hid.** `TTree::Branch` reports on ROOT's error stream and returns a
null branch **without throwing**, so the writer's very next line still logs
`wrote T_proj_data with 52 clusters` — a success message about a tree that lost
5 of its 6 branches. In the per-event runners that ROOT error lands in
`stdout.log` while every eye is on `wct_pr_evt<ID>.log`. And nothing reads the
tree: the valfast gate hashes `mabc-pr.zip` and the pctree tarball and compares
`T_tagger`/`T_kine` numerically, never `T_proj_data`.

**Blast radius.** Wider than the PR chain. All three Magnify tracking writers —
`SbndPrMagnifyTrackingVisitor`, `SbndMagnifyTrackingVisitor` (SBND
`stm_magnify`) and `UbooneMagnifyTrackingVisitor` — declare the same
`std::vector<std::vector<int>>` members and issue the same `Branch` calls, so
**every `tracking-*.root` written after the merge landed loses `T_proj_data`,
SBND and uBooNE alike**. (Inferred from the shared code path, verified directly
only on the PR file.) A grep confirms those three are the *only* nested-STL
`Branch` users in the tree, so nothing else is silently affected; the `float`
and `double` pragmas have no current consumer.

**The fix.** `root/dict/LinkDef.h` is restored, carrying **only** the three
`std` pragmas — deliberately not the reco1 mirror classes, which belong to the
standalone repo now. No build-file change was needed: both build systems
auto-detect the file (`waft/smplpkgs.py:512`, and `cmake/WCTPackage.cmake:114`
calling `wct_package_root_dict`), and `root/wscript_build` already carries
`ROOTSYS` in `use`. The file itself opens with a comment explaining why those
three pragmas exist, so the next merge has something to read.

The `GetCollectionProxy() != nullptr` trap is worth keeping in mind: an
*interpreted* class (`GetState() == 3`) returns a **non-null** proxy and still
cannot be branched. Only `kHasTClassInit` (`== 4`) carries a *compiled* one.
That false success signal is what made the earlier `GenerateDictionary` attempt
look like it had worked.

**Two things this is not.** Not `wire-cell`-specific: a bare `root -l -b -q`
macro branching `vector<vector<int>>` fails identically. Not the orphan rootmap:
stripping `local/lib` and `build/*` from `LD_LIBRARY_PATH` silences the
`recob::Wire` warning and the branch still fails.

**Verification of the fix** (owner-authorised 2026-08-03; landed as a behaviour
change, not knob-gated — a dictionary is a link-time property of
`libWireCellRoot`, and the change *restores* branches that were always intended):

```bash
cd toolkit && wcbuild                       # rc=0
root -l -b -q /home/xqian/tmp/vvload.C      # against local/lib/libWireCellRoot.so
#  -> STATE=4 (kHasTClassInit), branches created  (was STATE=3, NBRANCHES=0)
cd sbnd_xin && PR_JOBS=1 \
  ./run_pr_chain_batch.sh work-nuecc48-prod0803 work-prdict-388 data 388
```

| check | result |
|---|---|
| M1 freshness | `LinkDef.h` 17:32:23 → `WireCellRootDict.cxx` 17:32:36 → `libWireCellRoot.so` 17:33:04 |
| `T_proj_data` branches | **1 → 6**, all populated: 52 clusters, **13 821 cells** each in `channel`/`time_slice`/`charge`/`charge_err`/`charge_pred` |
| ROOT errors in `stdout.log` | `CollectionProxy` **0** (was 5); `no member named 'Wire'` **0** (was 5) |
| regenerated rootmap | 2756 B → **166 B**, `{ decls }` block now empty of larsoft/art |
| **physics unmoved** | `mabc-pr.zip` `c285c96d…` **7 members** and `pctree-pr-evt388.tar.gz` `3ea67506…` **425 members** hash-identical to the protected `work-vfnuecc48-prod0803/pr_evt388`; `nusel-evt388.tsv` identical |
| unit tests | `wcdoctest-clus` 71 cases / 809 assertions PASS |

Gate labels: `work-prdict-388/pr_evt388` vs `work-vfnuecc48-prod0803/pr_evt388`.
(When diffing `hash_archive.py` output, compare field 1 — the path column always
differs between arms.)

The ROOT cell count (13 821) and the display dump's (13 507) are **not expected
to match**, and neither is wrong. `T_proj_data` emits one row per *cluster tag*,
so a cell claimed by more than one cluster appears more than once — measured
here as **1023 duplicate rows, 12 798 unique `(channel, time_slice)` pairs**.
The dump instead keys cells by `(apa, face, plane)` + per-APA wire + slice, which
separates the two drift faces that share a global channel number. Compare the
two only after collapsing to a common key.

`wcdoctest-root` no longer exists: `root/test/` has no doctest sources. The
binary left in `build/` was a stale artifact of the reverted `ensure_stl_dict`
attempt and still ran a test for a deleted knob; it was removed and waf
correctly declines to relink it.

**Merge watch item.** A future master merge can delete `root/dict/LinkDef.h`
again exactly as silently as this one did. If it is restored, the file should
carry a comment saying *why* those three `std` pragmas exist (they are `root/`'s,
not reco1's), and it belongs on the post-merge check list — same failure mode as
the `CLAUDE.md` relocation.

**A wrong lead, recorded so it is not re-followed.** `tracking-stm.root` (STM
chain) has all six branches and its StreamerInfo record contains
`vector<vector<int> >`; `tracking-pr.root` has neither. The differentiating
variable *looked* like `tagger_output`, which reopens the file in `UPDATE` mode
after the tracking writer — a plausible story in which `TFile::WriteStreamerInfo`
rewrites the record from scratch and drops any class not touched in that
session. It is wrong. **Control: running the PR chain with `tagger_output`
removed from the pipeline still yields a one-branch `T_proj_data`.** The
missing streamer is a consequence of the branches never existing, not a cause —
and "the STM files simply predate whatever changed" turned out to be the whole
answer, once the *whatever* was identified as `5f684887`. A
`preserve_streamer_info` knob built on the streamer story was implemented,
tested, shown to change nothing, and removed. **Lesson: comparing StreamerInfo
lists measures the symptom. The date of the working file was the clue that
mattered.**

**A second dead end, also recorded.** Runtime dictionary generation —
`gInterpreter->GenerateDictionary("vector<vector<int> >","vector")` — is not
available here, so it is not an alternative to the LinkDef. It goes through
ACLiC, whose `g++` invocation fails with
`cc1plus: error: /wcwc/stage/root/spack-stage-root-6.32.02-…/include:
Permission denied` — ROOT's own build-time include path, left in its compiler
flags and no longer readable. It returns **rc = 0 regardless**, and
`TClass::GetCollectionProxy()` then hands back an *interpreted* proxy, so both
the return code and the obvious success check lie. Verified in a bare
`root -l -b -q` macro as well as in the job.

**The orphan rootmap, explained.** `build/root/libWireCellRoot.rootmap` and
`local/lib/{libWireCellRoot.rootmap,WireCellRootDict_rdict.pcm}` are dated
2026-07-30 and have no producer in the current build — they are **leftovers of
the deleted `root/dict/`**, not a mystery. Their forward-declaration block still
carries the reco1 larsoft/art classes whose headers left with `5f684887`, which
is why every ROOT session touching this tree prints `error: no member named
'Wire' in namespace 'recob'`. Restoring the LinkDef with only the three `std`
pragmas regenerates them without those declarations and retires that noise too.
They are stale but harmless: stripping them from `LD_LIBRARY_PATH` changes
nothing about the branch failure.

**This does not block the display**, which never reads `T_proj_data`.

### 5.2 `charge_pred` is not reproducible run to run

**Symptom.** Two runs of evt 18255/388 under `setarch x86_64 -R` produce
`calib-pr-evt388.json` files that differ in exactly one field:
`proj[].charge_pred`, on **922 of 13 507 cells (6.8 %)** in one pair and
**1379 of 13 507 (10.2 %)** in another. `wire`, `slice`, `charge`,
`charge_err`, `cluster_id`, `flag` and every other section of the dump —
segments, vertices, track_shower, steiner, dead, meta — are byte-identical.

**Root cause.** `TrackFitting::assemble_fitted_charge_2d`
(`clus/src/TrackFitting.cxx:1136-1152`) merges the per-cluster snapshots
last-writer-wins while iterating

```cpp
std::map<Facade::Cluster*, std::map<APAFacePlane, std::map<WireTime, FittedCharge2D>>>
    m_cluster_fitted_charge_2d;
```

— a **pointer-keyed container**, whose iteration order is not reproducible.
This is the exact pattern CLAUDE.md's determinism rule forbids. The code's own
comment notes that `charge`/`charge_err` "depend only on the readout, so the
overwrite is benign"; `pred_charge` is per-cluster, so on a cell claimed by two
clusters the winner varies.

**Blast radius: diagnostic only.** The merged map is read by exactly five call
sites — this dumper, the three Magnify tracking writers, and
`TaggerCheckSTM`'s `stm_fit` record. No tagger verdict, Bee layer or pctree
tensor depends on it, which is why no A/B gate has ever caught it.

**Not fixed here**, but the fix and its gate are both small. Sort the outer
loop of `assemble_fitted_charge_2d` by `cl->get_cluster_id()` and iterate that
order; `merge_fitted_charge_2d` (`:1154`) needs the same treatment if its input
is ever pointer-ordered.

**How to gate it — deliberately not the usual bar.** Do *not* ship this behind
a default-OFF knob and do *not* try to prove byte-identical-vs-before: today's
output disagrees **with itself** run to run, so there is no legacy path to
preserve. The gate is the inverse of the check that found the bug — two runs
under `setarch x86_64 -R` producing an identical `charge_pred`. Worth noting
that the measurement *confirmed* the existing source comment's premise:
`charge` and `charge_err` really are cluster-independent and were bit-identical
across runs. Only `charge_pred` moves.

Until then, do not read the display's per-cell measured-vs-predicted comparison
as a stable number. The caveat is stated in `pr_display/README.md` and in the
source.

## 6. Scope notes

- `work-vfnuecc48-prod0803` and `work-nuecc48-prod0803` are the owner's live
  campaign. Both were read only; all output went to the fresh
  `work-prdisp-388/`.
- The `pr_display` stage is read-only by construction, and the hash gate above
  proves it: an arm run with the stage is member-for-member identical to the
  protected arm run without it.
- Not in stage 1: hand-scan label saving, batch pre-rendering, and any change
  to the PR algorithms themselves.

---

# Stage 2 — particle flow and event features

Stage 1 showed *where* the charge and the fitted segments are. It could not
answer *which particle is this* or *why was this event selected*, which is what
the tuning campaign up to the neutrino vertex actually needs. Stage 2 adds both,
still read-only, still behind the same default-OFF `pr_display` stage.

## 7.1 The particle flow is READ, not rebuilt

The PR chain already produces a particle-flow tree:
`MultiAlgBlobClustering::fill_bee_pf_tree` writes it into the Bee zip as
`mabc-pr.zip::data/0/0-mc.json`, in the prototype's jsTree node format, and it
is what Bee shows. The display **reads that file** rather than deriving its own
tree — a second producer of the same quantity would eventually disagree with the
first, and the one that disagrees on a tuning display is the one that misleads.

The join works because the two producers already agree on ids:

```
fill_bee_pf_tree (mc.json)          PrDisplayDump (calib JSON)
  node id = cluster_id*1000           segment id = cluster_id*1000
            + seg->id()                            + seg->get_graph_index()
                        \                         /
       NeutrinoPatternBase.cxx:2170: seg->set_id(edge_bundle.index)
                       => the two are the same number
```

Verified on evt 388: `mc.json` node ids `23014 23032 81082 28053 23034` are all
present in the calib JSON's segment id set.

**What mc.json does not carry** is the rest of a shower. Its shower node names
only the shower's *start* segment, but the 789 MeV shower here spans 29. So the
dump gains one field — **`shower_id` on every segment**, `-1` when the segment
is in no shower, otherwise the *same encoding* mc.json puts on its shower node:

```
click "e-  789 MeV" (mc.json node 23014)
  -> segments with id == 23014            (1)
   ∪ segments with shower_id == 23014     (29)   -> highlight 29 segments
```

Shower membership comes from `shower->fill_sets(..., flag_exclude_start_segment
= false)`, not the per-segment flags: a segment absorbed into a shower from
another cluster may carry none of `kShowerTrajectory` / `kShowerTopology` /
`pdg == 11`. This is the same construction `dump_track_shower()` already used.

Three node kinds appear in the table:

| kind | id resolves to | highlight |
|---|---|---|
| `shower` | a `shower_id` group | every segment of the shower |
| `track` | a segment id | that segment |
| `gamma` | **nothing** | its children's segments, recursively |

The `gamma` rows are the pseudo-nodes `fill_bee_pf_tree` inserts between a
parent and an indirectly-connected shower (`start_connection_type` 2 or 3).
Their ids come from that function's own `next_id` counter (6 and 7 on evt 388)
and match no segment by construction — the display resolves them through their
children rather than showing an inert row.

A shower with `start_connection_type == 4` ("not clearly connected") is dropped
by `fill_bee_pf_tree` entirely, so it has a row in the dump's `showers` block
and **no PF node**. Evt 388 has one: id 27046, 4.8 MeV.

## 7.2 What the dump gained

`PrDisplayDump` grows three blocks and two fields. No new knob: the component is
only instantiated when `pr_display` is named in `pipeline_names`, so none of it
exists in a production job.

```jsonc
"segments": [ { …, "shower_id": 23014 | -1 } ],
"vertices": [ { …, "fit_distance": 0.224 } ],   // cm, |fit.point - wcpt.point|
"showers":  [ {id, shower_id, particle_id,
               kine_best, kine_range, kine_dQdx, kine_charge,   // MeV
               flag_kinematics, start:{x,y,z}, end:{x,y,z},
               start_connection_type, start_vertex_id,
               num_segments, num_main_segments, total_length,   // cm
               pio_id, pio_mass} ],
"kine":     { kine_reco_Enu, kine_reco_add_energy, kine_nu_{x,y,z}_corr,
              kine_energy_particle[], kine_particle_type[],
              kine_energy_info[], kine_energy_included[],
              kine_pio_* },                                     // MeV, cm
"tagger":   { weights: "uboone-trained -- UNCALIBRATED on SBND (doc pr/2 gap G1)",
              nue_score, numu_score, cosmict_score, cosmic_flag, match_isFC,
              …9 numu sub-scores, …30 nue sub-scores, photon_flag }
```

> **Superseded by §8.** That `tagger` block shipped 22 fields that no code ever
> assigns — they were displayed as `0.00` and read as physics answers. §8
> replaces it with the computed subset plus the cosmic tagger's ten per-test
> flags. The prose immediately below on `cosmic_flag` also had its **polarity
> backwards**; §8.1 is the corrected account.

`kine` is emitted verbatim — `NeutrinoKinematics.cxx` already divides by
`units::MeV` / `units::cm` before storing, so rescaling it here would be a bug.

**`fit_distance`** is `|fit().point − wcpt().point|` (`PRVertex.h:84`) — the gap
between a vertex's fitted position and its current Steiner seed point. Evt 388's
main vertex reads **0.22 cm**.

> **It is NOT "how far the 3-D vertex fit moved the vertex", and 0 does not mean
> the fit did not run.** Corrected 2026-08-03 (doc pr/28). Both halves fail:
> `TrackFitting::fit_point` moves `fit().point` for *every* vertex the vertex
> fit did **not** fix, and `MyFCN::UpdateInfo` re-snaps `wcpt()` to the nearest
> Steiner point for the ones it **did** — so the quantity is nonzero either way.
> Measured on evt 388: **127 of 127** vertices have `fit_distance > 0`,
> including degree-1 track ends that never reach `MyFCN` at all. For the main
> vertex the 0.22 cm is the residual between the `MyFCN` optimum and the Steiner
> node its seed was snapped to, not a fit displacement.
>
> The actual displacement is only in the trace log
> (`improve_vertex: ... fit_vertex done, vertex moved 0.693 cm`, evt 388 pass 1).
> **No artifact carries whether the vertex fit ran** — `PR::Fit::flag_fix`, the
> flag that answers it, is dumped nowhere (doc pr/27 §14).

### There is no `cosmic_score` — **and the two substitutes shown here were both wrong**

Stage 2 showed `cosmic_flag` and `cosmict_score` in its place. Both choices were
wrong, in different ways, and **§8 corrects them**:

| field | stage-2 claim | actually |
|---|---|---|
| `cosmic_flag` | "the cosmic tagger's own top-level boolean; 1 = cosmic-like" | **polarity inverted.** It is `!cosmict_flag_9`: 0 means cosmic-like. And it is a BDT *input feature*, not a verdict |
| `cosmict_score` | "the numu-BDT cosmic score" | **never computed**, in the toolkit or the prototype — a legacy slot on a dead code path |

The field that answers "did the cosmic tagger fire" is **`cosmict_flag`**. See
§8.

### The scores are UNCALIBRATED, and the dump says so

SBND books the **uBooNE-trained** weight XMLs (`sbnd/clus.jsonnet`, doc pr/2 gap
G1 — SBND retraining has not happened). The scores carry availability and
relative ranking, not a calibrated SBND number. The caveat is a string *inside*
`tagger.weights`, so it travels with the data and the panel prints it in red
under the scores; a viewer cannot show the number without it.

Ordering matters and is already right: `run_pr_chain_batch.sh` appends
`PR_EXTRA_STAGES` **last**, after `numu_bdt_scorer` and `nue_bdt_scorer`, both of
which write through `TrackFitting::get_tagger_info_mutable()`. Put `pr_display`
earlier and every score would read 0.

## 7.3 The display

A new row between the projections and the 2-D panels:

* **particle flow** — a `DataTable`, one row per mc.json node, indented by tree
  depth, with kind / id / KE / nseg / length. **nseg is what a click actually
  lights up**, so a gamma pseudo-node reports its children's count rather than 0.
  Clicking a row draws an **amber halo** under the selected segments in **all
  nine panels** — the three projections and each of the six 2-D panels, the
  latter split by the `(apa, face)` each fit point was recorded in (drawing a
  point with no recorded APA on APA 0 is the overlay bug doc pr/3 fixed). Amber
  because it has to stay legible both over the dark associated-point cloud and
  over the viridis charge cells.
* **selection / energy / topology chips** — nue_score, numu_score,
  cosmict_score, cosmic_flag, isFC; reco Enu and added energy; segment / shower /
  vertex counts, the neutrino vertex and its `fit_distance`. *(§8 replaces the
  two cosmic chips with one `cosmic` verdict chip plus a per-test table.)*
* **energy per particle** — the `kine_*` arrays as a table, with ✓ marking the
  entries actually summed into reco Enu (`kine_energy_included == 1`). Evt 388:
  13 particles, 4 counted, 2108 MeV.
* **BDT sub-scores** — all 39, behind a toggle so they do not dominate the page.

The `DataTable` carries the doc-58 fix: every column is always emitted (an empty
dict makes the client read 0 rows as 1), and `table.view.filter` flips between
two fixed `AllIndices()` instances after each `.data` assignment, because in
Bokeh 3.9 that view-change signal is the grid's *only* repaint channel.

## 7.4 Verification

| gate | result |
|---|---|
| **Compiled-config, production pipeline** | `PrDisplayDump` **absent** (0 occurrences); with `pr_display` appended it appears (2), and the two configs differ by **exactly** that inserted block — `diff` is a pure insertion, 633a634,653 |
| **Freshness (M1)** | `local/lib/libWireCellClus.so` 17:32:58 > `PrDisplayDump.cxx` 17:32:10 > `PrDisplayDump.h` 17:30:38 |
| **Dump regression** — new JSON vs the stage-1 arm, restricted to pre-existing keys | **identical**, with `proj[].charge_pred` excluded (§5.2; **464 / 13 507 = 3.4 %** of cells differ, inside the documented 6–10 % band) |
| **PF join** — every mc.json node → ≥ 1 segment | 7 rows: `23014`→29, `23032`→27, `6`(gamma)→1 via child, `81082`→1, `7`(gamma)→1 via child, `28053`→1, `23034`(track)→1. **No node resolves to zero** |
| **Scores read back from the served page** | nue_score **4.30**, numu_score **−2.48**, cosmict_score **0.00**, cosmic_flag **1**, reco Enu **2108 MeV** |
| **Click behaviour, headless chromium** | 7 table rows render; clicking row 1 reports "selected shower (id 23014) → 29 segment(s) highlighted"; the BDT toggle reveals `mipid_score`; **no JS errors** |
| `./build/clus/wcdoctest-clus` | 71 cases / 809 assertions **pass** |
| Protected dirs (M13) | fresh arm `work-prdisp-388-pf`; nothing written under `work-prdisp-388`, `work-nuecc48-prod0803` or `work-vfnuecc48-prod0803` |

**Not verified, stated rather than implied**: only one event is served, so the
`DataTable` *refresh-on-event-change* path — the doc-58 failure mode — is
exercised by construction (the filter flip runs on every `load()`) but has not
been observed across an actual event step. First multi-event serve should check
it, in single-row or empty-table mode (doc 58 GOTCHA 3: navigating a full table
passes even unfixed, because the surrounding reflow incidentally repaints).

> **Closed in §8.4** — the seven-event `work-prdisp-cosscan2` arm made the step
> observable, and it passes.

## 7.5 Scope

- No PF re-derivation in C++ — one producer, read by the display.
- No SBND retraining of the BDT weights (doc pr/2 gap G1); the scores are shown
  with the UNCALIBRATED label, not recalibrated.
- §5.2 (`charge_pred` pointer-order nondeterminism) stays reported-not-fixed.
- `pr_display` remains opt-in via `PR_EXTRA_STAGES`; no production config change.

---

# Stage 3 — what the cosmic answer actually is

*2026-08-03. Prompted by the owner reading the stage-2 panel: "since there is no
cosmic_score, you should remove it. I guess cosmic_flag may not be the one. What
is the score/flag for cosmics?"*

Both instincts were right. Stage 2 put two fields on the panel in place of the
`cosmic_score` the owner asked for; **neither was the cosmic answer**, and one of
them was described with its polarity inverted. This section establishes what the
toolkit really computes, removes everything it does not, and gives the display
the decomposition that makes a cosmic tag actionable.

## 8.1 The three things named "cosmic", and which one to read

### `cosmict_flag` — the verdict

`PatternAlgorithms::cosmic_tagger()` runs ten independent tests and ORs them
(`clus/src/NeutrinoTaggerCosmic.cxx:1342-1347`):

```cpp
bool flag_cosmic = flag_cosmic_1 || … || flag_cosmic_9 || flag_cosmic_10_save;
ti.cosmict_flag = flag_cosmic;
```

**This is the field to read.** It is the cosmic tagger's own answer, and it is
what the prototype feeds the numu BDT as `cosmict_flag`.

### `cosmic_flag` — not a verdict, and inverted

The owner's suspicion was exactly right. `cosmic_flag` is written in one place
only, inside the flag-9 block (`NeutrinoTaggerCosmic.cxx:1261-1264`; prototype
`NeutrinoID_cosmic_tagger.h:781-783`):

```cpp
if (flagp_cosmic) {                       // event looks like vertical cosmic tracks
    …
    if (/* neutrino-like shower at the main vertex */) {
        flagp_cosmic  = false;            // rescued
        ti.cosmic_flag = true;            // <-- TRUE means NOT cosmic
    } else {
        ti.cosmic_flag = false;           // <-- FALSE means cosmic
    }
}
if (flagp_cosmic) flag_cosmic_9 = true;
```

So `cosmic_flag == !cosmict_flag_9`, exactly. Three consequences:

1. **Its polarity is the opposite of its name.** 0 is the cosmic-like value.
   Stage 2's panel said "1 = cosmic-like". That was backwards.
2. **It covers one of the ten tests**, not the tagger.
3. **Its in-class default is 1** (`NeutrinoTaggerInfo.h:71`) and it is assigned
   only inside `if (flagp_cosmic)`, so a 1 is ambiguous between *never tested*
   and *tested and rescued*. `cosmic_filled` is the field that separates them,
   which is why it is now dumped beside it.

It is an input feature of the numu xgboost model (`m_xgb_vars[64]`), which is
its only real job.

### `cosmict_score` — never computed, anywhere

Not merely zero on this event — **never assigned**, in either codebase:

```
$ grep -rn "cosmict_score *=" prototype_base/
prototype_base/pid/src/NeutrinoID.cxx:3365:  tagger_info.cosmict_score = 0;   # init_tagger_info
```

That is the only write in the whole prototype, and the toolkit has none at all.
The field is a **legacy slot of the uBooNE ntuple schema on a dead code path**:
it belongs to `cal_numu_bdts()`, the pre-xgboost TMVA scorer, which **has no
caller** — the prototype selects `cal_numu_bdts_xgboost()` unconditionally
(`NeutrinoID.cxx:277`) and only that variant is ported. It survives in
`uboone_bdt_app`'s branch list because the offline ntuple schema is frozen.

*(Phrasing chosen deliberately: "legacy schema slot on a dead code path", not
"dead field" — the latter invites someone to restore it.)*

## 8.2 The same defect, 21 more times

Checking `cosmict_score` exposed the pattern, so the sweep was run over every
`_score` field the dump emitted:

| family | emitted by stage 2 | actually computed | dead |
|---|---|---|---|
| numu sub-scores | 9 | **4** — `cosmict_10`, `numu_1`, `numu_2`, `numu_3` | 5 (`cosmict_2_4`, `3_5`, `6`, `7`, `8`) |
| nue sub-scores | 30 | **15** — `br3_3/5/6`, `pio_2`, `stw_2/3/4`, `sig_1/2`, `lol_1/2`, `tro_1/2/4/5` | 15 |
| top level | `cosmict_score`, `photon_flag` | — | 2 |

The nue split has the same cause: `UbooneNueBDTScorer.cxx:1631-1645` fills
exactly the fifteen the xgboost model consumes; the other fifteen belong to
`cal_bdts()`, the prototype's `flag_bdt == 2` TMVA path, which is not ported.

**All 22 dead fields are removed from the dump.** A displayed `0.00` that no code
ever wrote is worse than an absent field: it reads as a physics answer. The C++
now carries the reason inline so the next person does not "restore" them.

> **`photon_flag` is different — it is a port gap, not a legacy slot.**
> `TaggerCheckNeutrino.cxx:813` calls `singlephoton_tagger()` and **discards its
> return value**, where the prototype does
> `if (flag_sp) tagger_info.photon_flag = true;` (`NeutrinoID.cxx:271`). The
> single-photon tagger's ~90 `shw_sp_*` features are filled; only its verdict is
> dropped. Reported here, **not fixed in this change** — it alters a TaggerInfo
> field that the uBooNE tagger ntuple writes, so it is a behavior change needing
> its own knob and gate.

## 8.3 What the dump and the display now carry

**Removed**: `cosmict_score`, `cosmict_2_4/3_5/6/7/8_score`, the 15 legacy nue
scores, `photon_flag`. **Added**:

```jsonc
"tagger": {
  weights, nue_score, numu_score, match_isFC,
  cosmict_flag,                       // THE VERDICT: OR of the ten tests
  cosmic_flag, cosmic_filled,         // == !cosmict_flag_9, and "did test 9 run"
  cosmict_flag_1 … cosmict_flag_9,    // which test fired
  cosmict_flag_10: [ … ],             // per-candidate: one entry per near-front-face vertex
  cosmict_flag_10_any,                // its OR (the tagger's own is a discarded local)
  cosmict_2_filled … cosmict_8_filled,// did each test EVALUATE
  …4 numu sub-scores, …15 nue sub-scores }
```

Two details that are easy to get wrong:

* **Test 10 is per-candidate, not per-event.** `cosmict_flag_10` is a
  `std::vector<float>` with one entry per vertex examined near the upstream
  face, and the tagger ORs it into `cosmict_flag` through `flag_cosmic_10_save`,
  a **local that is never stored**. Both the vector and a derived OR are
  emitted, because an *empty* vector ("no candidate was ever examined") is a
  different statement from an all-false one, and only the vector says which.
* **`*_filled` is what makes the panel honest.** Tests 2–8 fill their feature
  block only when their topology precondition is met. A 0 flag with `filled==0`
  means **not evaluated**; with `filled==1` it means **evaluated and did not
  fire**. Without this column every event on a neutrino-selected sample looks
  identical, and an inactive tagger is indistinguishable from a quiet one.

The display gains a **cosmic tagger** table: ten rows, `fired` and `ran`, rows
that never ran greyed out, each row's tooltip stating the actual cut. The four
selection chips become three — `nue_score`, `numu_score`, and a single `cosmic`
chip reading **TAGGED** (red) or **not tagged**.

| # | test | fires when |
|---|---|---|
| 1 | vertex outside FV | main vertex outside the FV shrunk by 1.5 cm |
| 2 | single muon, wrong dir. | muon at the vertex, ≤2 muon tracks, <40 cm shower, weak/steep direction or >40–60° off beam, downward-going |
| 3 | long-muon chain, wrong dir. | same test on a long-muon shower chain |
| 4 | muon exits, >100° | muon's far end outside FV, >100° from beam, no connected showers |
| 5 | long muon exits, >100° | same, for a chain |
| 6 | back-to-back secondary | 2nd muon track weak-direction, exits FV, >170° from the first |
| 7 | stopped muon + Michel | stopped µ with a Michel-like or near-back-to-back secondary, steep |
| 8 | muon + exiting back-track | µ >100 cm, one track >165° from it leaving FV, everything else <12 cm |
| 9 | vertical-track collection | cluster-PCA: most of the event's length vertical and reaching the top, not rescued by a neutrino-like shower |
| 10 | front-face beam-aligned | vertex outside FV within 15 cm of the upstream face, beam-aligned weak-direction track >10 cm |

## 8.4 Verification

The stage-2 no-A/B argument carries over unchanged and is not re-derived:
`PrDisplayDump` is instantiated only when `pr_display` is named in
`pipeline_names`, so production never builds it (§7.4, first row).

| gate | result |
|---|---|
| **Freshness (M1)** | `local/lib/libWireCellClus.so` **18:06:24** > `PrDisplayDump.cxx` **18:05:55** |
| `./build/clus/wcdoctest-clus` | 71 cases / 809 assertions **pass** |
| **Dead fields gone from the dump** | `cosmict_score`, `cosmict_2_4_score`, `mipid_score`, `photon_flag` all absent; `tagger` goes 47 → 37 keys |
| **Dead fields gone from the page** | headless: `cosmict_score`, `mipid_score`, `cme_anc_score`, `photon_flag`, `cosmic_flag` none present |
| **Cosmic table renders** | all ten rows, `fired`/`ran` columns, no JS errors |
| **The `ran` column earns its place** | over 7 events, two (**388**, **172230**) read `filled = 1010001` — tests 2, 4, 8 **ran and did not fire** — and five read `0000000`, never evaluated. Without it all seven look identical |
| **DataTable refresh across events — §7.4's open item, now CLOSED** | 7 events stepped with `next >`: **7 distinct row-sets**, row counts 3→13 (a 3-row table is doc 58 GOTCHA 3's discriminating case), and stepping back reproduces the first event's rows exactly |
| **Protected dirs (M13)** | fresh arms `work-prdisp-388-cos`, `work-prdisp-cosscan`, `work-prdisp-cosscan2`; nothing written under `work-prdisp-388`, `work-prdisp-388-pf`, `work-nuecc48-prod0803`, `work-vfnuecc48-prod0803` |

**Evt 388 reads `cosmict_flag = 0`** with all ten tests 0, `cosmic_filled = 0`,
`cosmict_flag_10` empty — so its `cosmic_flag = 1` is the *default*, not a
verdict. Exactly the ambiguity §8.1 describes, now visible on the panel.

**Stated, not implied — no firing case was found.** All **14** events tried
(388 plus 13 from `work-nuecc48-prod0803`) give `cosmict_flag = 0` with every
test 0. On a nueCC-*selected* sample that is the expected direction, and the
`ran` column proves the tagger is executing rather than silently inactive — but
**the FIRED rendering path itself has not been exercised on real data.** It
should be checked against a known cosmic before the flag decomposition is
trusted in a scan. This is reported, not tuned around (§5 rule 7).

## 8.5 Scope

- The 22 dead fields are removed from **this display's dump only**.
  `UbooneTaggerOutputVisitor`'s ROOT branch list is untouched — the offline
  ntuple schema is shared with uBooNE analysis code.
- `photon_flag`'s dropped verdict (§8.2) is **reported, not fixed**: it changes
  a written TaggerInfo field and needs its own knob and gate.
- No BDT retraining (doc pr/2 gap G1); no production config change;
  `pr_display` remains opt-in via `PR_EXTRA_STAGES`.

## 8.6 Repro

```bash
cd sbnd_xin
PR_EXTRA_STAGES=pr_display PR_JOBS=6 ./run_pr_chain_batch.sh \
    work-nuecc48-prod0803 work-prdisp-cosscan2 data 388 10550 111412 122660 137238 163543 172230
./pr_display/serve_pr_display.sh 5017 work-prdisp-cosscan2/pr_evt388/calib-pr-evt388.json

# the cosmic answer, per event
python3 -c "
import json,glob
for p in sorted(glob.glob('work-prdisp-cosscan2/pr_evt*/calib-pr-evt*.json')):
    t=json.load(open(p))['tagger']
    fl=''.join(str(int(t.get('cosmict_flag_%d'%i,0))) for i in range(1,10))+str(int(t['cosmict_flag_10_any']))
    fi=''.join(str(int(t['cosmict_%d_filled'%i])) for i in range(2,9))
    print('%-8s flag=%d fired=%s filled(2-8)=%s'%(p.split('evt')[-1].split('.')[0],t['cosmict_flag'],fl,fi))"
```
