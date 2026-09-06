# 143 — `break_segment` leaves its new vertex clusterless: the SBND exposure, measured, and the fix

**Status: SHIPPED** (2026-09-06), unconditionally and on all three detectors,
on the owner's decision quoted in sec 5.  The measurement below was carried out
with a scaffolding knob `break_seg_stamp_cluster` (default OFF) so that OFF and
ON could be run from one binary; that knob was then **deleted**, and sec 7.1
proves the shipped unconditional code is byte-identical to its ON arm on all
3000 events.  Read sec 6 as the gate record of the scaffolding build and sec
7.1 as the bridge from it to what shipped.

Owner's question, after doc pdvd/48: *"When you create the STM Michel tagger
CheckSTM_Michel for PDVD, you said that you identified a bug for break_segment.
I wonder if there is any impact on the SBND chain (CheckNeutrino)?"*, then
*"Yes, add the knob and run the 1368-event arm please"*.

**Scope.** Six toolkit files, `clus` only: `break_segment` now stamps the
vertex it creates, and the four caller-side stamps that were compensating for
that omission are deleted.  No config key anywhere, on any detector — so there
is nothing to set, and nothing that can be left unset.  PDVD and PDHD get the
same fix; their production chains are `-stm`, which never reaches the changed
sites (sec 7.3).

## 0 Repro block

```bash
cd toolkit && ./wcb build --notests -p -k && ./wcb install --notests -p -k && ./wcb build -p
./build/clus/wcdoctest-clus -tc="clus pr break_segment stamps the new vertex"

# the population census that motivated the fix (no runs; reads existing dumps)
cd sbnd_xin
ls work-mcp2k-d97fvpr2/*/calib-pr-evt*.json work-mcp1k-d97fvpr2/*/calib-pr-evt*.json \
  | xargs -P 16 -n 25 python3 scripts/analysis/pr143/pr143_orphan_vertex_census.py \
  | awk 'NR==1 || $1!="event"' > docs/pr/pr143-orphan-census-off.tsv
# NOTE the awk: xargs starts one process per 25 dumps and EACH emits its own
# header, so the raw stream carries 54 header lines among the 1368 data rows.
# int() on them throws; a reader that skips bad rows silently undercounts.

# gate reference arm (pre-change binary), 26 mcp1k events
export LD_LIBRARY_PATH=/home/xqian/tmp/d47_libpin/new11
PR_JOBS=13 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-mcp1k-d97fv work-pr143-gateref data $(cat /home/xqian/tmp/d143/gate_events.txt)

# the two population arms (new binary), 3000 events each
export LD_LIBRARY_PATH=/home/xqian/tmp/d143_libpin/new
for QL in work-mcp1k-d97fv work-mcp2k-d97fv; do
  PR_JOBS=16 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh $QL work-pr143-off data
done
for QL in work-mcp1k-d97fv work-mcp2k-d97fv; do
  SBND_BREAK_SEG_STAMP_CLUSTER=1 PR_JOBS=16 PR_EXTRA_STAGES=pr_display \
    ./run_pr_chain_batch.sh $QL work-pr143-on data
done

python3 scripts/analysis/pr143/pr143_compare_arms.py work-pr143-off work-pr143-on
```

## 1 The defect

`PR::break_segment` (`clus/src/PRSegmentFunctions.cxx:1173`) creates a vertex
with `make_vertex(graph)` and copies the parent's cluster onto the two child
segments — but not onto the vertex.

It is the **only** vertex factory in the PR chain that does this.  There are
16 other direct `make_vertex` calls, and every one of them stamps the cluster
on the next line or two:

| file | calls | lines |
|---|---|---|
| `NeutrinoOtherSegments.cxx` | 5 | 597, 602, 619, 625, 1299 |
| `NeutrinoPatternBase.cxx` | 6 | 1487, 1490, 1841, 1897, 3140, 3144 |
| `NeutrinoVertexFinder.cxx` | 3 | 495, 747, 751 (the last two on a temporary local graph) |
| `NeutrinoStructureExaminer.cxx` | 2 | 934, 2668 |

and there is no back-fill pass anywhere — the 17 `vtx->cluster(...)` writes in
`clus/src` are all at a creation site, none is a sweep — so a vertex that
leaves `break_segment` clusterless stays clusterless for the rest of the
event.

Its four callers split two and two:

| caller | file:line | stamps the vertex? |
|---|---|---|
| `break_two_end_dqdx` | `NeutrinoPatternBase.cxx:3089` | yes, `:3097` (doc pr/48) |
| `snap_main_vertex_to_kink` | `NeutrinoVertexFinder.cxx:2857` | yes, `:2866` (doc pr/104) |
| `nv_bridge_track` | `NeutrinoShowerClustering.cxx:1919` | **no** |
| `shower_clustering_with_nv_from_vertices` | `NeutrinoShowerClustering.cxx:2246` | **no** |
| `CheckSTM_Michel` (PDVD/PDHD) | `CheckSTM_Michel.cxx:632` | yes, `:644` (doc pdvd/48) |

Both unstamped sites are live on SBND: `shower_nv_bridge_track = true` in
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`, and the from-vertices
break is legacy prototype code with no gate at all.

The two stamping callers each carry a comment naming the gotcha — the omission
was known twice and fixed twice at the call site, never at the factory.

## 2 How often, and where — measured on the 2026-09-02 production arms

`PrDisplayDump` writes `vertices[].cluster_id = vtx->cluster() ? id : 0`
(`PrDisplayDump.cxx:429`), and no SBND *segment* ever carries cluster id 0
(minimum over the sample is 1), so cluster_id 0 on a vertex means `cluster()`
is null.  The existing `work-mcp{1,2}k-d97fvpr2` calib dumps therefore answer
the question with no new runs.

| | |
|---|---|
| events processed | 3000 (mcp1k 1000 + mcp2k 2000, disjoint ids) |
| events with a PR graph (a dump) | 1368 |
| PR vertices in those | 30684 |
| **clusterless vertices** | **47, on 44 events** |
| … as a fraction | 3.2 % of dumped events, 1.5 % of all events, 0.15 % of vertices |
| per event | 1 or 2 |

**A census trap worth recording.**  The first pass of this table read only the
dump's top-level `vertices`/`segments` and reported *44 on 42*.  A dump produced
with `nu_per_bundle` repeats the seven per-candidate keys under a `candidates`
array (`PrDisplayDump.cxx:207-239`), and `candidates[0]` IS the top-level block
-- so a top-level-only reader silently drops every vertex of `candidates[1..]`,
which exist on 24 of the 1368 dumps.  The undercount was caught only because
sec 7's artefact comparison found 44 movers where the census claimed 42
candidates for movement, and the two extra events (179369, 409624) turned out
to carry their orphan in `candidates[1]`.  The committed script walks the
candidates array; the numbers above are the corrected ones, and they now match
the mover set exactly.

* The 43 degree-2 orphans each sit between two halves of one non-main
  cluster's segment.  Consistent with site 2246, though not a clean
  discriminator: a site-1919 break whose `nv_bridge_connect` then declines
  also leaves degree 2.
* The single degree-3 orphan is unambiguous — **event 95412**: a cluster-44
  muon split 0.60 cm / 93.63 cm (both pdg 13) plus a 2-point, 0.55 cm segment
  stamped to main cluster 12.  That third edge is `nv_bridge_connect`'s
  synthetic bridge, so `nv_bridge_track` fires once in 3000 events.
* **The null result, and the one that matters:** `main_vertex->cluster()` is
  non-null on 1368/1368 events, and no orphan's own material is in the main
  vertex's cluster (95412's single main-cluster incidence is the bridge, not
  the broken track).  Every consumer that asks *"is this vertex in the main
  cluster"* answers no for all 44 both before and after a stamp.

## 3 Why the PDVD failure modes cannot occur here

Both unstamped sites run inside `shower_clustering_with_nv`
(`TaggerCheckNeutrino.cxx:3194`) — **after** `determine_main_vertex` (`:2901`),
`main_vertex_graph_audit` (`:3089`) and `examine_direction` (`:3147`).

* `examine_direction` bails at `!vertex->cluster()`
  (`NeutrinoVertexFinder.cxx:1503`), but it is entered with
  `final_main_vertex`, which is always stamped.
* `fill_bee_pf_tree`'s `main_cluster = main_vertex->cluster()`
  (`MultiAlgBlobClustering.cxx:1337`) — the doc pdvd/48 mode that emptied 73 of
  574 PF roots — needs a null **main** vertex cluster: measured 0/1368.  Its
  `same_cluster` filter keys on *segment* clusters, which `break_segment` does
  stamp.
* `determine_main_vertex` can never select an orphan: its candidate loops
  filter `vtx->cluster() == &cluster`.

## 4 What is actually exposed

The loops that key on an *arbitrary* vertex's cluster, all downstream of
`:3194`:

* **`pi0_identification_sp`** (`NeutrinoTaggerSinglePhoton.cxx:2148-2152`) —
  the live one.  It walks `ordered_nodes(ctx.graph)`, i.e. every vertex.  An
  orphan is **not** skipped (null ≠ the shower vertex's cluster), then
  `cluster_acc_length.find(nullptr)` misses — the map is built under
  `if (sg1 && sg1->cluster())`, so a null key is never inserted — and
  `acc_length` reads 0.  The `acc_length > 0` clause then blocks both
  `flag_pi0_2` and every `shw_sp_pio_2_v_*` push_back for that vertex.
  `singlephoton_tagger` calls it at `:2601`; `TaggerCheckNeutrino.cxx:3423`
  calls that unconditionally.
* **`low_energy_overlapping_sp`** (`NeutrinoTaggerSinglePhoton.cxx:1937`) —
  found by the arms, not by this reading, and it corrects what this section
  claimed in its first draft.  Its loop IS guarded
  (`!vtx1->cluster() || vtx1->cluster() != sg->cluster()` → skip), and such a
  guard is safe only when the comparison is against the MAIN cluster.  Here
  `sg` is the shower's own start segment and the orphan's parent segment sits
  in exactly that cluster, so the stamp admits a vertex the guard used to drop
  and five `shw_sp_lol_1_v_*` features move.  Sec 2's null result retires the
  main-cluster loops, not every cluster-keyed loop.
* `low_energy_michel_sp` (`:2211`) and `mip_identification_sp` (`:1525`) are
  guarded the same way but keyed on a cluster the orphan is not in, so it is
  skipped either way.
* `bad_reconstruction_3_sp` (`:1010`), `NeutrinoKinematics.cxx:441`,
  `NeutrinoTaggerCosmic.cxx:1338` are main-cluster equality tests, and sec 2's
  null result says a stamp would assign a non-main id — no verdict can move.
* Display only: the Bee vertices layer (`MultiAlgBlobClustering.cxx:1226`) and
  the calib dump paint the orphan `cluster_id` 0, and its display id collapses
  from `cluster_id*1000 + graph_index` to `graph_index`.

## 5 The change

Owner decision 2026-09-06, after seeing sec 7's numbers: *"flip it on and delete
the two redundant caller-side stamps -- one correct factory instead of four
callers who each have to remember"*, and *"it should be also on for PDVD and
PDHD"*.  So this ships **unconditionally, with no knob at all**: a default-ON
knob nobody would ever set to OFF is the kind doc 77's rounds retire, and one
that could unstamp the other call sites would be a footgun.

`PR::break_segment` now ends its cluster block with

```c++
seg1->cluster(seg->cluster());
seg2->cluster(seg->cluster());
vtx->cluster(seg->cluster());     // <- new, doc pr/143
```

and **all four** caller-side stamps are deleted:

| caller | file:line (was) | why deleting it is a no-op |
|---|---|---|
| `break_two_end_dqdx` | `NeutrinoPatternBase.cxx:3097` | `cand` is selected under `sg->cluster() != &cluster -> continue` (`:3018`) |
| `snap_main_vertex_to_kink` | `NeutrinoVertexFinder.cxx:2866` | `arms[].seg` comes only from `incident`, built under `sg->cluster() == &cluster` (`:2669`) |
| `CheckSTM_Michel::anchor_vertex` | `CheckSTM_Michel.cxx:644` | `best` is drawn from `find_cluster_segments(g, cluster)`; the stamp was already dead behind `if (!nvtx->cluster())` once the factory ran first, and its comment had become the opposite of the truth |
| the two `NeutrinoShowerClustering` sites | `:1919`, `:2246` | never had one -- this is the behaviour change |

In each case the factory writes `seg->cluster()`, which those selection guards
make identical to the `&cluster` the caller used to write.  Proven by reading
above, and gated in sec 6.

Nothing else moves: no config key, no TLA, no runner env var, no jsonnet edit in
any detector.  `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` and
`sbnd_xin/run_pr_chain_batch.sh` are back at their pre-round content, so the
compiled config of every detector is byte-identical to HEAD.

**Where it actually bites.**  Only `pdhd`, `protodunevd` and `sbnd` bind
`TaggerCheckNeutrino`; uBooNE does not (the `Uboone*BDTScorer` classes are
consumed BY those three, which is a confusing name, not a binding).  Of the
three:

* **SBND** -- production runs the PR tail, so this is a production output
  change.  Sec 7 measures it.
* **PDVD and PDHD** -- production is `-stm`, which never runs the PR tail
  (gates A and C, byte-identical), and their `-nu` chain is `CheckSTM_Michel`,
  which never calls `shower_clustering_with_nv`.  The only chain of theirs that
  reaches the two unstamped sites is `-nu-legacy`, which is not production.
  What they gain is the fourth deletion above.

Tests: `clus/test/doctest_prsegment.cxx` "clus pr break_segment stamps the new
vertex" -- 3 subcases, 15 assertions: the vertex AND both children carry the
parent's cluster (so a "fix" that MOVES the stamp instead of adding one fails),
a clusterless parent stays clusterless rather than crashing, and `orient_split`
does not change who owns the vertex.  `./build/clus/wcdoctest-clus`: **323 cases
/ 23060 assertions SUCCESS**.

## 6 Proofs and gates

Gate record: `docs/pr/pr143-gate.txt` (pins, gates A–G, config proofs, tests,
and the effect census — re-checkable without this doc).

### 6.1 Binary pins

| pin | `libWireCellClus.so` | note |
|---|---|---|
| ref (shipped) | `a4ff5439991ee343f8e0c0dcda13128f` | `/home/xqian/tmp/d47_libpin/new11`, the doc pdhd/03 shipped build, toolkit `5d0b4e77` |
| new (this round) | `6aa917a541f10f1d91642e2bcd40727d` | `/home/xqian/tmp/d143_libpin/new` |

`libWireCellRoot.so` is `4a9efb7f52d22d93f2d9fcc9df394222` on both — untouched.
Freshness: `local/lib/libWireCellClus.so` 2026-09-06 05:53:26, last source edit
05:50:53.  Build recipe was the new-symbol one (`build -k`, `install -k`,
`build`) because the added parameter changes `break_segment`'s mangled name and
the doctest links against the installed library.

### 6.2 Config proofs

* **T0 — the knob-off compiled config is byte-identical.**  One event (mcp1k
  59929) compiled through the runner's own `wcsonnet` step twice: once with
  `PR_CFG_TREE` pointing at a symlink mirror of `toolkit/cfg` carrying
  `git show HEAD:.../sbnd/wct-pr-perevt.jsonnet`, once with the live tree.
  After normalising the arm-directory path (the two runs write to different
  out_roots) the compiled JSON is **IDENTICAL**, md5 `cebd93bb7b736a4212bda6a40810286a`.
* **T1 — the key appears only when asked.**  `work-pr143-on`'s
  `.wct-cfg-evt48301.json` carries `"break_seg_stamp_cluster": true` exactly
  once, in the `TaggerCheckNeutrino` node; `work-pr143-off`'s does not carry
  the key at all.

### 6.3 Byte-identity gates, scaffolding knob unset, ref pin vs new pin

| gate | chain | events | result |
|---|---|---|---|
| A | PDVD `-stm -stm-fit` (production) | 039252/2, 039349/23 | **PASS** 4/4 |
| B | PDVD `-nu-legacy` (`TaggerCheckNeutrino`) | 039252/2, 039349/23 | **PASS** 7/7 |
| C | PDHD `-stm -stm-fit` (production) | 029107/0, /6 | **PASS** 4/4 |
| D | PDHD `-nu-legacy` (`TaggerCheckNeutrino`) | 029107/0 | **PASS** 3/3 |
| E | SBND bare production (`run_d42_stmfit.sh`, `D42_NO_STMFIT=1`) | 284349, 285999, 286065 | **PASS** 9/9 |
| F | PDHD `-nu` = `CheckSTM_Michel` | 029107/0 | **PASS** 4/4 |

Compared per pair: `mabc-pr.zip` and `pctree-*.tar.gz` by member content hash,
`tracking-{pr,stm}.root` by per-tree hash over **every** tree the file holds
(`hash_root_trees.py --per-tree --trees <all>` — its default list is the SBND
triple and would silently skip `tracking-stm.root`'s trees), calib dumps with
the `*_ms` scoreboard timers stripped, TSVs by bytes.  A file present on one
side only is a FAIL — that is how gate F's first run caught a missing
`-stm-fit` on my side (re-run as `d143nu2`).

**B and D are the gates that carry weight**: they are the arms that actually
instantiate `TaggerCheckNeutrino`, the component the knob is wired into.  Gate
F passes for a weaker reason — `CheckSTM_Michel` calls `break_segment` with six
arguments, so `stamp_cluster` defaults false there and its own stamp at
`CheckSTM_Michel.cxx:644` still fires; the path is untouched rather than
exercised.  A and C and E prove the shared `clus` library carries no other
change.

The comparer was validated in both directions before use: a file against
itself reports SAME, and two different events report DIFF on 6 trees
(`pr143_pair_compare.py`, sec 0).

### 6.4 Tests

`clus/test/doctest_prsegment.cxx` "clus pr break_segment stamps the new
vertex" — 3 subcases, 15 assertions: the new vertex **and both children** carry
the parent's cluster (so a "fix" that *moves* the stamp from the children to
the vertex rather than adding one fails), a clusterless parent yields a
clusterless vertex rather than a crash, and `orient_split` does not change who
owns the vertex.  The test asserts the shipped unconditional behaviour, not the
scaffolding knob's.  `./build/clus/wcdoctest-clus`: **323 cases / 23060
assertions SUCCESS**.

## 7 What it changes, measured

Three arms on the final binary's behaviour, all 3000 SBND events
(`work-pr143-off` = today's production, `work-pr143-on` = the stamp via the
scaffolding knob, `work-pr143-final` = the shipped code with no knob at all).
Every arm 3000/3000 with `rc != 0` on zero events.

### 7.1 The shipped code IS the ON arm (flip-equivalence + deletion-inertness)

`work-pr143-final` vs `work-pr143-on`, 3000 events:

| class | result |
|---|---|
| `nusel-evt<ID>.tsv` | SAME 3000 |
| `tracking-pr.root` (per-tree, every tree) | SAME 3000 |
| `mabc-pr.zip` (member content) | SAME 3000 |
| `pctree-pr-evt<ID>.tar.gz` (member content) | SAME 3000 |
| `calib-pr-evt<ID>.json` (timers stripped) | SAME 1368 |
| `nusel-table.tsv`, `nusel-events.tsv` | SAME |
| `T_tagger` + `T_kine`, branch by branch | **0 / 3000 events move a leaf** |

That single comparison carries both halves of the change: the factory now
stamps the two `NeutrinoShowerClustering` vertices, **and** deleting the four
caller-side stamps changed nothing.  It has teeth --
`snap_main_vertex_to_kink` fires on 109 of every ~2138 SBND events and
`two_end_break = true` there -- so the deletions are exercised, not merely
unvisited.

### 7.2 SBND: what the fix actually moves

`work-pr143-off` vs `work-pr143-on`, 3000 events:

| class | result |
|---|---|
| clusterless vertices | **47 on 44 events → 0**, vertex total 30684 unchanged |
| `nusel-evt<ID>.tsv` | SAME 3000 |
| `nusel-table.tsv`, `nusel-events.tsv` | SAME |
| `pctree-pr-evt<ID>.tar.gz` | SAME 3000 |
| `tracking-pr.root` / `mabc-pr.zip` / calib dump | **DIFF on exactly the 44 orphan events**, SAME on the other 2956 |

The mover set equals the orphan set exactly -- no collateral.

Branch by branch over `T_tagger` (1229 branches) and `T_kine`, **18 of 3000
events move at least one leaf**:

| leaf | events |
|---|---|
| `shw_sp_pio_2_v_acc_length` | 17 |
| `pio_2_v_acc_length` | 7 |
| `shw_sp_lol_1_v_{angle,energy,flag,nseg,vtx_n_segs}` | 5 |
| `shw_sp_pio_2_v_{angle2,dis2,flag}` | 5 |
| `shw_sp_br3_*`, `shw_sp_br4_*` (9 leaves) | 1-2 (173495, 409624) |
| `shw_sp_n_good_showers`, `shw_sp_{n_,}br{3,4}_showers` | 1-2 (173495, 176986) |
| **`photon_flag`** | **1 (173495), 1 → 0** |

and, on the other side of the ledger:

* **no `T_kine` leaf moves at all** -- reconstructed neutrino energy and the
  pi0 kinematics are untouched;
* **all 111 score-like branches are identical on all 18 movers**, checked
  value by value, not merely absent from the diff -- `nue_score` and
  `numu_score` included;
* no `nusel` verdict changes anywhere.

The top entry is the mechanism sec 4 predicted: `pio_2_v_acc_length` is a real
nue-BDT input (`UbooneNueBDTScorer.cxx:215` registers it, `:1249` fills it per
candidate).  The vector LENGTH never changes -- the orphan's row was always
there, carrying `acc_length = 0` from the null-key lookup; the stamp gives it
the cluster's real accumulated length (e.g. 0.00 → 52.77 / 1.37 / 5.60 cm).
`pio_2_score` is the MINIMUM over candidates (`:1252`), and a candidate that
moves off zero without becoming the minimum leaves the reduction untouched --
which is why an input can move while every score stands still.

**The one verdict that moves is `photon_flag` 1 → 0 on event 173495.**  It is
the single-photon tagger's own verdict, it does not reach the nusel table, and
whether dropping that photon flag is right is a physics judgement no gate here
can make.  Flagged for a scan, not resolved.

### 7.3 PDVD: the same defect, ten times denser -- on a chain that is not production

`-nu-legacy` is the only PDVD chain that reaches the two unstamped sites.  Over
`pdvd/stm/events.txt` (120 events, ref pin `new11` vs the final binary):

| | |
|---|---|
| clusterless vertices (ref) | **52 on 37 of 119 dumped events -- 31 %** |
| `mabc-pr.zip` | DIFF 37 / SAME 83 |
| `tracking-pr.root` | DIFF 37 / SAME 83 |
| `calib-pr-evt<ID>.json` | DIFF 37 / SAME 82 |
| `tracking-stm.root` | **SAME 120** (the STM tagger runs before the PR tail) |

Movers equal orphan events exactly here too.  The rate is ten times SBND's
3.2 %, which is the number to remember when reading gates B and D: they PASS
on 3 hand-picked events that happen to contain no clusterless vertex, so they
prove the code compiles and runs, not that PDVD is unaffected.

**But PDVD production does not change**: production is `-stm`, which never runs
the PR tail (gate A, byte-identical), and the `-nu` chain is `CheckSTM_Michel`
(gate F, byte-identical).  Same for PDHD.

### 7.4 Gaps

* No PDHD population arm -- PDHD is covered by gates C/D/F (4+3+4 pairs) only.
  Its `-nu-legacy` rate is unmeasured; PDVD's 31 % suggests it is not small.
* `photon_flag` on 173495 is unscanned.
* The PDVD/PDHD `-nu-legacy` movers are unscanned; nothing downstream of them
  is production, which is why this was not treated as a blocker.

## 8 Status and what to do next

**Shipped unconditionally** (owner decision 2026-09-06).  There is no knob, so
the only way back is a revert.

What the owner should know before the next round touches this:

1. **`photon_flag` 1 → 0 on SBND 173495** is the one verdict that moved in 3000
   events.  Worth a Bee scan; it is the single case where the fix changed an
   answer rather than a feature.
2. **PDVD `-nu-legacy` moves on 31 % of events.**  If that chain is ever
   promoted back to production, this is the arm to re-read first.
3. **PDHD has no population number.**  A 30-event `-nu-legacy` arm on run
   029107 would close the last measurement gap cheaply.
4. **The census trap in sec 2 is general**: any script that reads a PDVD/SBND
   calib dump's top-level `vertices`/`segments` silently drops
   `candidates[1..]`.  `pr143_orphan_vertex_census.py` walks them; older
   per-round scripts do not.
