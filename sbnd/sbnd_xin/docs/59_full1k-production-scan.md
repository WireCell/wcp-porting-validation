# 59 — The full 1000-event MCP2025C data sample through the default production, and the hand-scan subset

**Yes.** The default SBND production nusel chain runs the whole 1000-event data
sample: **999/1000 succeeded**, total wall **~17 min** (two phases, 8- then
24-way; 4.5 h of serial per-event wall), peak RSS 405 MB median / 799 MB max,
8.3 GB of output. The **one failure is a deterministic crash** in
`TrackFitting::do_single_tracking` on evt 278794 (§6) — reported, not fixed here.

Of the 999 tabulated events, **639 (64 %)** have an in-beam bundle that is
STM-tagged or untagged: that is the hand-scan set served on **5011** (§4) and
uploaded to Bee in 7 chunks (§5). The rest are 154 TGM, 10 LM, 9 mixed and 187
with no in-beam bundle at all.

Nothing here is an A/B and nothing was tuned: no knob moved, no default changed,
no C++ or jsonnet was touched. The gate that matters is that this chain
reproduces the doc-56 production arm byte-for-byte (§2).

## Repro block

```bash
cd wcp-porting-img/sbnd/sbnd_xin

# 1. production: 1000 events into work-mcp1kall-d59k (8-way here; see §3)
./run_full1k_nusel.sh 1000 8          # per-event rc/wall/RSS in work-*/.status/<entry>
# a partial or interrupted run resumes by entry list, e.g.
# ENTRIES="373 374 ..." ./run_full1k_nusel.sh 0 24

# 2. the hand-scan subset + the Bee-index maps
python3 nusel_scan_filter.py -w work-mcp1kall-d59k \
    --census-out   scan-d59k/census.tsv \
    --events-out   scan-d59k/events.txt \
    --tsv-list-out scan-d59k/tsvs.txt \
    --chunk 100 --chunk-prefix scan-d59k/chunk

# 3. serve the subset on 5011 (ABSOLUTE paths -- doc 56 GOTCHA 1)
SB=$PWD
nusel_display/serve_nusel_scan.sh 5011 --tag s59k --charge-src pr \
    --prev $SB/work-mcp10-d56bw:d56bw --prev $SB/work-mcp1000-d56bw:d56bw \
    --prev $SB/work-mcp1000b-d56bw:d56bw \
    $(cat scan-d59k/tsvs.txt)

# 4. Bee: one upload per 100-event chunk (outward-facing -- owner asked for links)
./make_scan_bee.sh scan-d59k/bee work-mcp1kall-d59k scan-d59k/chunk-*.txt
```

Sample: `input_files_reco1/staged-mcp2025c-1000evt/e0..e999`, one single-event
sample dir per art entry of `data_MCP2025C_reco1_frameshift_first1000ev.root`
(**data**, runs 18255 subrun 1 with 850 entries and 18259 subrun 1 with 150; all
1000 event ids distinct). `entry_event_map.tsv` there is the entry →
run/subrun/event map.

## 1. What "the default production" is here

The chain is `run_nusel_evt.sh`: Q/L matching with a persisted point-cloud tree →
the PR tagger tail (`switch_scope` → `unmerge_bundle` → `unmerge_assoc` →
`steiner` → `fiducialutils` → `tagger_check_tgm` → `tagger_check_stm` →
`tagger_check_fc` → `stm_magnify`) → the per-bundle label table
(`nusel_extract.py`).

Flags are the **d55ton / d56bw production set verbatim** — the identical string
every doc-52…57 arm uses, kept in `run_full1k_nusel.sh` as `NUF`:

```
-chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm -main-pair-real
-fvx 2.5 -fvy 3 -stm-fit -mip 56000 -unmerge-assoc
```

Two of those this task itself requires, so they are not optional here: `-lm`
(without it there is no LM verdict to filter on) and `-stm-fit` (without it the
Bee `stm_fit` layer does not exist). The beam-window gate is on by default since
doc 56 (`[0.2, 2.2) us` on `cluster_t0`), which is also what makes a 1000-event
run affordable.

**Imaging is never regenerated** (M11/M13): `work-mcp1kall-d59k/evt<ID>` is an
absolute symlink into `work-mcp1000/evt<ID>`, the 1000-event imaging base from
the doc-37 coverage run (1000/1000 present, checked before launch). Everything
downstream *is* produced fresh in this root — the Q/L step (pctree,
`mabc-all-apa.zip`, calib dump) and the whole PR tail.

### The one plumbing gap that had to be closed

`run_nusel_evt.sh` forwards only `-save-pctree`, `-save-rcid`, `-lm`, `-calib` to
a Q/L step it launches — **not `-save-assoc`**. `-unmerge-assoc` needs the
isolated-grouping arrays, and without them the inner un-merge degrades to a
per-cluster *warning* and silently does nothing. Every earlier arm inherited its
pctree from an explicit `run_ql_evt.sh -save-assoc` run, so the gap never showed;
this is the first run that builds 1000 pctrees from scratch.
`run_full1k_nusel.sh` therefore **exports `SBND_SAVE_ASSOC=1`**, which
`run_ql_evt.sh` reads from the environment.

Because the failure mode is a warning and not an error it is checked per event,
not trusted: every `.status/<entry>` line carries `assoc=<mains>/<parts>` scraped
from the PR log's `<ClusteringUnmergeBundle:prassoc> unmerged …` line.

```
$ grep -l 'assoc=none' work-mcp1kall-d59k/.status/* | wc -l          ->   1  (the crashed event, §6)
$ grep -l 'assoc=0/0'  work-mcp1kall-d59k/.status/* | wc -l          ->   1  (evt408534)
$ LC_ALL=C grep -al 'prassoc> unmerged' */wct_nusel_evt*.log | wc -l  -> 999  (of 1000)
$ LC_ALL=C grep -al 'no flash-merge provenance' */wct_nusel*.log      ->   1 line, 1 event
```

So the inner un-merge ran with its arrays on all 999 events that completed. The
two singletons are both benign and were checked individually:

- **evt408534 `assoc=0/0`** — the arrays were there and there was simply nothing
  to undo: the *outer* un-merge on the same event split 6 mains into 8
  (`<ClusteringUnmergeBundle:pr> unmerged 6 … into 8`), and no
  isolated-grouping merge remained. No warning in its log.
- **evt281918 cluster 6** — the one `no blob marked real_cluster_main, using the
  proxy` → `no flash-merge provenance …; not split` pair. That is the documented
  switch_scope case (the representative member stranded out of scope), and the
  visitor deliberately **skips** rather than falling back to the connectivity
  proxy. One cluster of one event out of 999.

## 2. Gate: the chain reproduces the doc-56 production arm

Three of the 1000 events also sit in the doc-56 `d56bw` scan (entries 10, 12,
15 = evts 286681, 287517, 288067). Run through `run_full1k_nusel.sh` from a
**freshly built pctree** (d56bw symlinked d55ton's), their label tables are
**byte-identical** to the d56bw ones:

```
=== evt286681 === IDENTICAL
=== evt287517 === IDENTICAL
=== evt288067 === IDENTICAL
```

(`diff` on `nusel-evt<ID>.tsv`; copies kept in the session scratchpad as
`smoke_proof/{d56bw,d59k}-evt<ID>.tsv`.) The `prassoc` line agrees too —
evt287517: `unmerged 10 main cluster(s) into 35 associated cluster(s) (10
exact/provenance, 0 proxy/component)`, the same 10/35 d56bw logged.

That is the whole correctness argument for this production: same flags, same
inputs, same outputs as the arm the doc-56 scan was taken on.

Binary provenance: `local/lib/libWireCellClus.so` mtime 09:20 on 2026-07-26,
i.e. built with `efc2c281` (09:13, doc 57's MIP-scale thresholds); `HEAD` during
the run was `9f7dfd8e` (09:27), **comment-only**, so the running binary is
code-equivalent to HEAD. The byte-identity above is also independent
confirmation of doc 57's bit-identical claim, since `d56bw` was produced *before*
`efc2c281`.

Everything ran under `setarch x86_64 -R` (the SBND PR chain is
ASLR-non-deterministic at ±7 STM tags, doc 49 §4a). That is not required for a
production, but it makes this sample re-checkable the way the A/B arms are.

## 3. Cost

| | |
|---|---|
| events | 1000 (999 ok, 1 crash) |
| wall, phase 1 (8-way) | 373 events in 436 s |
| wall, phase 2 (24-way) | 627 events in 577 s |
| total wall | ~17 min including the ~2 min drain between phases |
| per-event wall | median 17 s, p90 23 s, max 199 s (sum 4.5 h) |
| peak RSS / event | median 405 MB, p90 477 MB, max 799 MB |
| output | 8.3 GB (≈8.5 MB/event: 3.2 MB pctree, 3.2 MB post-PR pctree, 1.9 MB calib dump, Bee zips, logs) |

The run started 8-way (load ~11–14 on 64 cores) and was moved to 24-way
mid-flight on the owner's word (load ~14–35, still ≈½ the box). **How to
re-parallelize safely:** kill the `xargs` dispatcher only, wait for the in-flight
workers to write their `.status/` lines, then relaunch with `ENTRIES=` the
remaining entries. Killing a *worker* mid-Q/L is the thing to avoid — it can
leave a truncated `pctree-evt<ID>.tar.gz`, and `run_nusel_evt.sh` reuses any
**non-empty** pctree, so the next run would silently consume a corrupt tree.

Per-event wall is dominated by the Q/L step, not the taggers: the doc-56
beam-window gate already cut the PR tail to ~3 s/event.

## 4. The hand-scan subset, and the 5011 display

The request: keep events with an in-beam bundle, drop TGM-tagged and LM-tagged
events, so what remains is STM-tagged or untagged. `nusel_scan_filter.py`
implements exactly that, deciding on the table's **`label`** column — not the raw
`tgm`/`stm`/`lm` columns, because `nusel_extract.label_of()` already applies the
beam window and the TGM > STM > LM priority, and the raw `lm` column is a 0/1/2
verdict code, not a boolean:

> An event is **kept** iff (a) it has ≥1 in-beam bundle — a row with `in_beam=1`
> whose label is not `no-bundle` (that synthetic row is an in-window *flash* that
> matched **no** qualifying bundle, i.e. nothing to scan) — and (b) no in-beam
> bundle of that event is labeled `TGM` or `LM`.

Census over the 999 tabulated events (`scan-d59k/census.tsv`, one row per event):

| verdict | events | share | |
|---|---:|---:|---|
| **keep** | **639** | 64.0 % | 151 with an STM-tagged in-beam bundle, 488 all-untagged |
| tgm | 154 | 15.4 % | every in-beam bundle TGM-tagged |
| lm | 10 | 1.0 % | every in-beam bundle LM-tagged |
| mixed | 9 | 0.9 % | a cosmic-tagged **and** a keepable in-beam bundle — dropped by (b) |
| no-inbeam-bundle | 187 | 18.7 % | no in-window bundle (in-window flash with no bundle, or no beam flash) |

The 9 **mixed** events are listed here because they are the only place the cut is
a judgement call — rule (b) drops an event if *any* in-beam bundle is
cosmic-tagged, so these leave the scan even though each also has a keepable
bundle. Say the word and they can be added back (they are 1.4 % of the kept set):

```
evt61681  TGM,nu-candidate      evt280281  STM,TGM
evt70562  nu-candidate,LM       evt282952  TGM,nu-candidate
evt169598 TGM,nu-candidate      evt288974  LM,nu-candidate
evt174928 nu-candidate,TGM      evt391854  TGM,nu-candidate
                                evt399382  TGM,nu-candidate
```

**Served on 5011**, scan tag **`s59k`** (a fresh tag — M13; the doc-56 `d56bw`
labels are untouched and its scan can be re-served any time):

- 639 events in the selector, `--charge-src pr` (the post-un-merge cluster the
  taggers actually saw, doc 50 Q1).
- The three `d56bw` roots are passed as `--prev` baselines: only ~20 of the 1000
  overlap the doc-56 draws, but for those the earlier scan labels carry over and
  a changed verdict is tinted amber.
- Verified headless (playwright chromium): 639 entries in the dropdown, the table
  repaints on every event step (the doc-58 fix), no JS error.
- Event ids are **not** contiguous and span both runs (evt48301 … evt493243).

The viewer is served from **explicit TSV paths**, not a work root, so no
symlink farm was built for the subset: `discover_events()` accepts TSV paths and
derives each event's work root from the path.

## 5. Bee links

7 uploads, 100 events each (39 in the last), built by `make_scan_bee.sh` →
`make_stmfit_bee.py`. Per event the Bee tree carries

- **`img-global`** — the per-APA post-Q/L clusters, raw x,
- **`clustering-global`** — the T0-corrected merged bundles (from the **Q/L**
  dump: `PR_LAYERS` takes only the fit layer from the PR zip, so this is the
  pre-un-merge clustering, matching the ids the `op` layer references),
- **`op`** — the flashes with `op_cluster_ids`, i.e. the **QLMatch** result, so
  Bee's flash `<`/`>` navigation with "show matching cluster" works,
- **`stm_fit-global`** — the STM track fits, `q = dQ*0.1 - 1000`, cluster ids
  remapped into `img-global`'s space so a fit appears together with the bundle
  its flash selects,
- the `channel-deadarea-apa{0,1}-face0` layers.

| chunk | events | zip | link |
|---|---:|---:|---|
| 00 | 100 | 43 M | <https://www.phy.bnl.gov/twister/bee/set/69bea0cf-22fe-40d6-95cd-c0b0e87c7e25/event/list/> |
| 01 | 100 | 42 M | <https://www.phy.bnl.gov/twister/bee/set/43a58f86-c1b6-4f11-92f9-7880db84b814/event/list/> |
| 02 | 100 | 40 M | <https://www.phy.bnl.gov/twister/bee/set/a5a1f26f-61bc-47b9-9bfd-4146fc111bb5/event/list/> |
| 03 | 100 | 39 M | <https://www.phy.bnl.gov/twister/bee/set/43782748-daa2-44d6-ac92-46d42983fe3c/event/list/> |
| 04 | 100 | 43 M | <https://www.phy.bnl.gov/twister/bee/set/f41676de-3c31-47ac-9372-a94d0e6bdfd2/event/list/> |
| 05 | 100 | 45 M | <https://www.phy.bnl.gov/twister/bee/set/99bf9909-d3af-469c-bd54-0ba3000a09c1/event/list/> |
| 06 | 39 | 16 M | <https://www.phy.bnl.gov/twister/bee/set/99560cb3-c650-403e-8872-a1f69d184057/event/list/> |

All seven return HTTP 200. **Bee identifies events by directory index, not event
id**, so each chunk ships its map: `scan-d59k/bee/chunk-NN.index.txt` — Bee
event `i` is line `i+1`. `make_stmfit_bee.py` also drops
`chunk-NN.stmid-map.txt` (PR cluster id → img cluster id) next to each zip;
that is the way back from a Bee color to the TSV / `tracking-stm.root` ids.

**330 of the 639 events carry a `stm_fit` layer**, and the other 309 legitimately
have none: the STM tagger only fits a main it does not skip first. The
`stmfit` column of the kept events' in-beam bundles reads `eval` 329,
`contained` 259, `nosteiner` 53, `postfit` 6, `midkink`/`shortfit`/`nexits` 1
each, `-` 2 — so the 329 `eval` rows are exactly the events with a fit layer.
`make_stmfit_bee.py`'s `has no ['stm_fit-global'] (was the run made with
-stm-fit?)` warning is therefore benign here; the 330 fits prove `-stm-fit` was
on.

## 6. The one failure: evt 278794 aborts in TrackFitting (NOT fixed here)

Entry 618 / **evt 278794** (run 18255 subrun 1) dies in the PR job:

```
[clus.NeutrinoPattern] visit: TaggerCheckSTM: beam_window_only [0.200, 2.200) us:
    1 main(s) evaluated, 7 out of window
terminate called after throwing an instance of 'std::runtime_error'
  what():  TrackFitting::do_single_tracking: inconsistent vector sizes for fit output!
```

`clus/src/TrackFitting.cxx:8566`. **Deterministic** — reran it once under the
same `setarch x86_64 -R` and it aborts at the same point. Because the Q/L step
already succeeded and its pctree survives, the repro is 1 second:

```bash
ENTRIES="618" ./run_full1k_nusel.sh 0 1     # rc=1, wall 1 s, reuses the pctree
```

Consequences, stated rather than hidden: the event has **no** `nusel-evt278794.tsv`
and so is absent from the 999-event census, from the 639-event scan set and from
the Bee uploads. It is 0.1 % of the sample. Per CLAUDE.md an unrelated bug found
mid-task is reported, not fixed in the same change — this one wants its own
write-up (the throw is a size mismatch between the fit's output vectors, i.e. a
real defect in `do_single_tracking`, not a configuration problem).

## 7. Files

New in `sbnd_xin/`:

| file | what |
|---|---|
| `run_full1k_nusel.sh` | the production driver (entry list, per-entry cwd, `.status/` rc+wall+RSS+assoc, resumable by `ENTRIES=`) |
| `nusel_scan_filter.py` | the census + subset selector (`--census-out/--events-out/--tsv-list-out/--chunk`) |
| `make_scan_bee.sh` | per-chunk Bee build + upload, with input pre-validation and the index map |
| `scan-d59k/{census,events,tsvs}.*`, `scan-d59k/chunk-*.txt` | the census and the kept-event / Bee-index lists |
| `docs/59_full1k-production-scan.md` | this doc |

Outputs (untracked): `work-mcp1kall-d59k/` 8.3 GB — `evt<ID>` symlinks into
`work-mcp1000`, plus `ql_evt<ID>/` and `nusel_evt<ID>/` per event, and
`.status/`, `.log_e<entry>.log`, `.time_e<entry>.meta`. `scan-d59k/bee/` holds
the 7 zips (268 MB), their `.url`, `.index.txt`, `.stmid-map.txt` and build logs.
`work-tags.md` gains the `work-mcp1kall-*` prefix and this root.

## Notes / leftovers

- **This is not an A/B and carries no gate label.** No knob, default, constant,
  C++ file or jsonnet file was touched; the only new code is runner/analysis
  scripts. §2 is the evidence that the chain is the doc-56 production one.
- The 187 `no-inbeam-bundle` events are *not* a defect — an in-window flash that
  matched no qualifying bundle is exactly what `nusel_extract` reports as the
  `no-bundle` row (and the beam-window population is ~1 bundle/event, doc 56).
  Whether that 18.7 % deserves its own look (flash with charge that Q/L did not
  match) is a separate question this doc does not answer.
- **GOTCHA — `grep` needs `-a` on these PR logs.** 3 of the 1000
  `wct_nusel_evt<ID>.log` files contain a byte sequence that is not valid UTF-8
  (a torn multi-byte character — the known non-atomic spdlog write), so GNU grep
  classifies the whole file as *binary* and silently reports **nothing**: plain
  `grep -c prassoc` returns rc=1 with no output on a file that has 18 matches.
  A first pass through this doc's own numbers came out as "996 of 999 events ran
  the inner un-merge" for exactly that reason. Any census over these logs must
  use `grep -a` (and `LC_ALL=C`); the runner's `.status` scraper uses `sed`,
  which is not affected.
- `work-mcp1kall-d59k/nusel_labels/d59ktest/` is mine, not a scan: the viewer was
  first verified against the partial output on port 5097 under that scratch tag,
  so the live `s59k` tag was never served from an unfinished sample. Inert (the
  viewer only reads the tag it is served); left in place.
- The doc-58 test run's scratch label tags
  `work-*-d56bw/nusel_labels/{tblrefresh,tblrefresh_before}/` are still there,
  deliberately (nothing under a labels dir is removed without the owner's word).
- Restarting 5011 for `s59k` ended the previous `d56bw` browser session; a stale
  tab shows up as `Token is expired` in the server log and needs a hard reload.
