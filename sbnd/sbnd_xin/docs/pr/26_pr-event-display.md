# pr/26 — SBND PR event display, stage 1 (SBND evt 18255/388)

A dedicated event display for validating and improving the neutrino
pattern-recognition (PR) code, ahead of the tuning campaign over the 572
valfast events. Stage 1 is **read-only viewing plus the dump that feeds it**:
no PR algorithm changes.

Two defects were found on the way and are written up in §5. Neither is fixed
here; both are reported, per CLAUDE.md's rule on unrelated bugs found mid-task.

## Repro

```bash
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
cd $SX

# 1. the display's input: the PR chain with the (default-OFF) pr_display stage.
#    ql_root is the owner's protected campaign -- READ ONLY; out_root is fresh.
PR_EXTRA_STAGES=pr_display PR_JOBS=1 \
  ./run_pr_chain_batch.sh work-nuecc48-prod0803 work-prdisp-388 data 388
# -> work-prdisp-388/pr_evt388/calib-pr-evt388.json   (1.9 MB)

# 2. serve it
./pr_display/serve_pr_display.sh 5017 work-prdisp-388/pr_evt*/calib-pr-evt*.json
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
                     is_main_cluster,
                     points:[{x,y,z,dQ,dx,pu,pv,pw,pt,apa,face,reduced_chi2,rr}]} ],
  "vertices":     [ {id, cluster_id, is_main, degree, fit:{...}} ],
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
