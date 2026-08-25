# doc 82 — the two defects doc 81 left open: log tearing and Q/L non-determinism

Doc 81 shipped the group-mode re-baseline of the four SBND production samples
byte-identical, and recorded two residuals it deliberately did not chase:

* **§5 / §8.2** — 72 of 34835 `nusel-table.tsv` rows differ between arms, every
  one in a log-parsed column, because WCT log lines *tear*.
* **§7.1** — multi-event Q/L is run-to-run non-deterministic on marginal
  events; 7 of 3067 had to be re-run outside the group to reach the stage-A
  PASS.

Round 1 root-caused and **fixed** the log tear, and root-caused the Q/L flip as
far as "a read of memory the program never wrote".  **Round 2 retracts that
second conclusion** (§2c) — memcheck is clean in every line of WCT
reconstruction code, on runs landing in *both* states.  Round 2 does **not**
supply a replacement: it also proposed one (address reuse) and withdrew it when
the replicate came in.  Round 2 also tests event 99438 as
the separate thread doc 81 asked for (§2b, it is not separate), and fixes the
group-mode `rc=0` coverage defect (Part 4).

**Round 3 finds the mechanism and fixes it** (§2d):
`QLMatching::rescue_empty_flashes()` recorded which flash a cluster is
matched to by walking `std::map<Opflash*, …>` in raw heap-address order; when
a cluster had surviving bundles on two flashes, the winner of that overwrite
depended on the two `Opflash` objects' relative addresses.  Proven causally by
instrumenting the walk and catching a state-A and a state-B draw with
identical inputs differing only in the overwrite winner.  Fixed by using this
file's own existing stable order, `flash_iter_order()`, already used at every
other walk of the same map.  Gated: mcp1k 25-event + mcp2k 30-event manifests,
zero unexpected movers; every formerly-bistable event, repeated 12–25 times
post-fix, now returns exactly one answer.

Neither cause round 1 and round 2 chased was what the earlier write-ups
assumed, and round 1's own answer to the second was wrong too.  Every claim
below is labelled with the round that made it.

## Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin

# --- 1. the log tear, on any log written BEFORE the fix (read-only) -----
python3 scripts/multi/log_tear_scan.py work-nuecc48-grp0825/g0/wct_img.log
#   -> 5240 lines  70 torn  69/70 on a 4096 boundary  max staleness 93.7s
#   the same job re-run at the fixed binary reports 0 torn.

# --- 2. the bad read.  ~1 min per draw, and the two arms disagree ------
#     DETERMINISTICALLY, which is the whole finding.
MALLOC_PERTURB_=1   DRAWS=5 PRECOMPILE=1 ./scripts/multi/repro_ql_nondet.sh \
    work-mcp2k-grp0825 /home/xqian/tmp/d82/e99438-perturb1   99438
MALLOC_PERTURB_=170 DRAWS=5 PRECOMPILE=1 ./scripts/multi/repro_ql_nondet.sh \
    work-mcp2k-grp0825 /home/xqian/tmp/d82/e99438-perturb170 99438
#   -> =1   : 5/5 IDENTICAL to the reference
#   -> =170 : 5/5 identical to each other, 5/5 DIFFER from the reference

# --- 3. the original doc-81 pair, for the ~10%-flip version of it -------
DRAWS=10 ./scripts/multi/repro_ql_nondet.sh \
    work-mcp1k-grp0825 /home/xqian/tmp/d82/base-inproc 285993 286191

# --- ROUND 2 -----------------------------------------------------------
# 4. the cell round 1 never measured: 99438 alone and IN-PROCESS, which is
#    the configuration doc 81 actually failed in.  PRECOMPILE=0 is the point.
DRAWS=10 PRECOMPILE=0 ./scripts/multi/repro_ql_nondet.sh \
    work-mcp2k-grp0825 /home/xqian/tmp/d82r2/e99438-inproc 99438
#   -> 2 draws match the reference, 8 do NOT.  Doc 81 saw 3 of 3 in the 8-side.

# 5. the control: the other six, same configuration, 3 draws each.
./scripts/multi/run_e2.sh /home/xqian/tmp/d82r2   # 292643, 321101, 53793 also flip

# 6. how many distinct outcomes exist, over every draw of every arm.
python3 ./scripts/multi/state_census.py /home/xqian/tmp/d82r2
#   -> 1 or 2 states per event, never 3.

# 7. memcheck on the same event (~4 min; needs no rebuild).  "-" = no fill.
#    NOT ONE memcheck error in WCT reconstruction code, in either state.
#    Run each arm THREE times: a single pair here appears to make the freelist
#    volume decisive and it is not -- the event is bistable under valgrind too.
for i in 1 2 3; do
  ./scripts/multi/run_vg.sh /home/xqian/tmp/d82r2/fl20-$i  99438 - -  20000000
  ./scripts/multi/run_vg.sh /home/xqian/tmp/d82r2/fl500-$i 99438 - - 500000000
done
#   -> 20 MB : 2 A / 1 B     500 MB : 2 A / 1 B   (i.e. the knob does nothing)

# 8. permutation or genuinely different numbers?  (8% relative, so: different.)
python3 ./scripts/multi/numdiff.py \
    /home/xqian/tmp/d82r2/e99438-inproc/draw5/ql_evt99438/pctree-evt99438.tar.gz \
    /home/xqian/tmp/d82r2/e99438-inproc/draw1/ql_evt99438/pctree-evt99438.tar.gz

# --- ROUND 3 (post-fix; confirms the flip is gone) --------------------
# 9. the exact repro that caught the bug, 25 draws, fixed binary.
DRAWS=25 PRECOMPILE=0 ./scripts/multi/repro_ql_nondet.sh \
    work-mcp1k-grp0825 /home/xqian/tmp/d82r3/verify 285993 286191
#   -> 0 of 300 draw-vs-draw pairs differ (was ~1 in 9-10 pre-fix)

# 10. the mcp2k known-bistable events, 12 repeated draws.
DRAWS=12 PRECOMPILE=0 REF=work-mcp2k-grp0825 ./scripts/multi/repro_ql_nondet.sh \
    work-mcp2k-grp0825 /home/xqian/tmp/d82r3/verify2k \
    53793 99438 161043 321101 350816
#   -> 0 of 66 draw-vs-draw pairs differ; each event now has exactly one answer
```

The reproducer rebuilds a group from surviving per-event products, so it needs
no reco1 file and no re-imaging.

**Reference arm, changed 2026-08-25b.** Every number above was originally taken
against `work-<sample>-ql0819`, which the 08-25b retirement round released (doc
81 sec 10). The commands now name `work-<sample>-grp0825`, which is also the
reproducer's new default. The substitution is exact rather than approximate:
doc 81 sec 7 gated the two arms at 24536/24536 archives member-content
identical, and that gate was re-run from the frozen manifest against the
surviving arm on the day of the retirement -- 24536/24536 again
(`scripts/retire/verify_frozen_stagea_20260825b.py`). So none of the results
above change; only the path does.

## Part 1 — the log tear

### Symptom

`nusel_extract.py` has been hardened against torn log lines **four separate
times**: `VERDICT` reduced to `([tf01])\w*` after a torn `FC=fal` crashed
evt285999; the tagger name dropped from `RE_TGM`/`RE_FC` after a tear removed a
line's head; `RE_SKIP` reduced to a prefix match; `RE_STM_SKIP` relaxed in doc
76 round 3.  Doc 81 then hit a tear that cut the word `cluster` itself, which
no regex can recover.  The parser was being patched against a defect in the
toolkit.

### Root cause

`Aux::Logger::set_name()` (`aux/src/Logger.cxx:39`) takes a private-sink logger
so each component can carry its own pattern:

```cpp
log = Log::logger(lname, false); // uniqe sinks so we can set unique pattern
```

`Log::logger()` with `share_sinks=false` (`util/src/Logging.cxx:138-142`) re-runs
every entry of `common_sink_makers`, and the file maker
(`util/src/Logging.cxx:46-56`) constructs a **new sink object each time**:

```cpp
auto s = std::make_shared<sinktype>(fname);   // basic_file_sink_mt
```

`basic_file_sink` owns a `file_helper`, which owns a `std::FILE*` opened `"ab"`.
So a job with N configured components leaves **N independent 4096-byte stdio
buffers appending to one file**.  They fill and drain at different times, so one
component's buffer is flushed into the middle of another component's line.

### Why it hid

Three things made this look like something else.

* **It is not a thread race.**  Every sink WCT instantiates is already `_mt`
  (there is no `_st` sink in the tree), and both the Q/L and PR jobs run under
  `Pgrapher`, the single-threaded executor.  "Make the sinks thread-safe" would
  have been a no-op fix for a bug that is not there.
* **It reproduces exactly**, which reads as physics rather than formatting —
  `nusel_extract.py:211-215` records the tear as "DETERMINISTIC (a write-buffer
  boundary, not a race)".  It is deterministic because a buffer boundary is.
* **The `-l stderr` sinks are duplicated too** and never tear, because stderr is
  unbuffered.  So the same job shows the defect in one sink and not the other.

### Evidence

`scripts/multi/log_tear_scan.py` over four campaign logs:

| log | component loggers | lines | torn | on a 4096 boundary | max staleness |
|---|---:|---:|---:|---|---:|
| `work-nuecc48-grp0825/g0/wct_img.log` | 98 | 5240 | 70 | 69/70 | 93.7 s |
| `work-nuecc48-grp0825/g0/wct_ql.log` | 13 | 9734 | 12 | 12/12 | 85.1 s |
| `work-ncpi0-prod0825/wct_pr_g0.log` | 14 | 20267 | 11 | 11/11 | 133.7 s |
| `work-mcp2k-prod0825/wct_pr_g103.log` | — | 28888 | 11 | 11/11 | 28.1 s |
| **total** | | **64129** | **104** | **103/104** | |

Three independent confirmations in that table:

1. **103 of 104 splices sit on an exact multiple of 4096** (the last is one byte
   short of one, its `[` being the final byte of the preceding chunk).  A thread
   race has no reason to prefer page-sized offsets.
2. **The spliced-in fragment is up to 133 s older than the line it cuts.**  Only
   a second, independently buffered handle can emit bytes that stale.
3. **The tear count tracks the component count** — 98 loggers give 70 splices,
   13-14 loggers give 11-12 — because each component is one more buffer.

The pairing is consistent too: the *interrupted* writer is a file-scope
shared-sink logger (`clus/src/TaggerCheckSTM.cxx:32`,
`Log::logger("clus.NeutrinoPattern")`, default `share_sinks=true`), and the
*interrupting* one is always an `Aux::Logger` private-sink component
(`<CreateSteinerGraph:pr>`, `<TaggerCheckNeutrino:pr>`).

### Fix

`util/inc/WireCellUtil/SharedFileSink.h` + `util/src/SharedFileSink.cxx`: a
sink that keeps its **own formatter** — so the emitted line is byte-for-byte
what it was, and no log-parsing script or quoted doc changes — but routes every
write through **one shared `file_helper` and one shared mutex per path**,
memoized by filename.  `Log::add_file` constructs it instead of
`basic_file_sink_mt`; `common_sink_makers` still mints a sink object per logger,
and those objects now all find the same handle.

It derives from `base_sink<std::mutex>`, not `base_sink<null_mutex>`, and that
is deliberate: `basic_file_sink_mt` held a per-instance lock across `sink_it_()`
and spdlog's `pattern_formatter` mutates its own `cached_tm_` / `last_log_secs_`
inside `format()`, so dropping that lock would introduce a data race for any
caller sharing one sink across threads.  The per-instance lock is kept and the
shared per-file lock is taken inside it, around the write only.  Ordering is
always instance-then-shared and the shared lock is never held while an instance
lock is taken, so there is no deadlock.  `Pgrapher` jobs never contend either
one; `TbbFlow` would.

Two cheaper options were rejected:

* `Log::logger(lname, true)` in `Aux::Logger::set_name` — two lines, but the
  `set_pattern` immediately below then mutates the *shared* formatter, so the
  last component named wins for everybody.  Identity would have to move into
  `%n`, changing the prefix of every log line in every job.
* `flush_on` / per-message `fflush` — ends the tearing but leaves N handles open
  and costs a syscall per message per component.

*Recorded but not applied*: `-l stderr` with the shell redirecting `2>"$LOG"`
cannot tear at all (console sinks bind a process-static `ConsoleMutex` and
`fwrite`+`fflush` inside the lock).  It is a usable stopgap for any driver that
cannot take a rebuilt toolkit, at the cost of per-sink level control.

### Verification

Fail-first, on the binary that shipped doc 81:

```
util/test/doctest_logging.cxx:178: ERROR: CHECK( ntorn == 0 ) is NOT correct!
  values: CHECK( 31 == 0 )
```

with the reported example carrying the production signature exactly —
`…xxxxx|@| tear/1 msg 0 of logger 1…`.  After the fix, the same case reports
`6400 records, 0 spliced lines` and passes.

M1 freshness, taken against `toolkit/build/` **and** `local/lib`.  That matters
here: this environment's `LD_LIBRARY_PATH` puts `toolkit/build/*` ahead of
`local/lib`, and a live job was confirmed to map `libWireCellClus.so`,
`libWireCellAux.so` and `libWireCellImg.so` from `build/` — so a freshness proof
against `local/lib` alone checks the copy that is not loaded.

| check | result |
|---|---|
| `libWireCellUtil.so` (build/ and local/lib) 10:24:46 vs `Logging.cxx` 10:24:42 | newer |
| `nm -D` for `SharedFileSink` in both copies | 38 symbols each |
| `./build/util/wcdoctest-util` | 276 cases, 42570 assertions, **SUCCESS** |
| `./build/aux/wcdoctest-aux` | 19 cases, **SUCCESS** |

Real jobs, the same 16-event Q/L group before and after:

| log | lines | torn |
|---|---:|---:|
| before, `g46-inproc/draw1` | 10452 | 12 |
| before, `g46-precomp/draw1` | 10452 | 12 |
| **after, `g46-postfix/draw1`** | **10452** | **0** |
| **after, `g46-postfix/draw2`** | **10452** | **0** |
| **after, shipped build, `rcg-mtx/g0`** | **10452** | **0** |

The line counts are *identical* before and after, which is the point: no record
is lost, gained or reformatted — only the splices are gone.

Products, post-fix Q/L against the recorded per-event reference, 3 draws of the
same 16-event group:

| draw | vs `work-mcp1k-ql0819`, 16 events |
|---|---|
| `g46-postfix/draw1` | **IDENTICAL, 64/64 archives** |
| `g46-postfix/draw2` | **IDENTICAL, 64/64 archives** |
| `g46-postfix/draw3` | 62/64 — evt **286191** differs |

That third draw is not a regression, and the round has the instrument to say so
rather than argue it.  The fix replaces N file handles with one, which changes
the job's allocation history — and part 2 shows that is exactly what re-rolls a
bistable event.  So the question is not "did anything differ" but "did anything
NEW appear":

* the only event that moved in any draw is **286191**, already known bistable;
* it landed on **`eed0bc8753` = state B**, its known second state — not a third;
* the other 15 events are identical to `ql0819` in all three draws.

So the logging fix changes no physics.  It re-rolls an already-bistable event
onto one of the two values it already had, which is the part-2 defect showing
through, and the outcome-state hash is what separates the two cases.  Any future
log or allocation change carries the same caveat, and the same test settles it.

Full PR chain, one 16-event nueCC48 group at the fixed binary, gated against the
recorded `work-nuecc48-prod0825`:

| gate | result |
|---|---|
| `wct_pr_g0.log` tear scan | **31233 lines both, 10 torn → 0 torn** |
| `scripts/pr85_hash_gate.py` | **PASS** — 32 archives byte-identical, 0 missing/unpaired |
| `scripts/pr94_root_gate.py` | **PASS** — 16 identical, 0 differing, 0 skipped |
| `nusel-table.tsv` vs the recorded table | **178 rows, 0 differing** |

Both counters were read on each gate, not just the headline: `pr85` reports a
differing archive *set* as unpaired rather than FAIL, and `pr94` reports a
missing ROOT file as skipped, so a truncated arm can otherwise look clean.

The nusel comparison is against the **recorded** table, which is the check that
can see a *reordering* rather than only a tear: the fix does not merely remove
splices, it also puts a low-volume logger's line back in chronological position
instead of wherever its stale buffer happened to drain.  Zero differing rows
says no such reordering reaches the parsed columns on this sample.  It also
means these 16 events' 10 tears were ones `nusel_extract.py`'s existing
tolerance already absorbed — the fix recovered no row here, and that is worth
stating plainly rather than claiming a recovery the data does not show.

Performance was not measured under controlled conditions and no claim is made.
By construction the change adds no syscalls — buffering is unchanged, there are
simply fewer buffers — and it replaces N uncontended per-sink mutexes with one
shared mutex, which under `Pgrapher` is taken by a single thread anyway.

## Part 2 — the Q/L non-determinism

### Symptom

Doc 81 §7.1: the first stage-A gate failed on 7 of 3067 events (mcp1k 286191,
292643; mcp2k 53793, 99438, 161043, 321101, 350816), always in the Q/L products
and never in imaging.  Running the same 2-event group twice could give different
answers, so it was called run-to-run instability and left for a later round.

### The reproducer, which did not exist

Doc 81 assembled its 2-event group by hand and never saved it, and round 5 then
pruned the group scratch it lived in.  What survives is the **per-event split**
of those group archives, and the split is lossless — so a group can be rebuilt
from any subset of events with no reco1 dump and no re-imaging:

* `scripts/multi/merge_group_products.py` — the exact inverse of
  `split_group_products.py`.  Round-trip proven: merge two events, split back,
  **12/12 archives member-identical** to the originals.
* `scripts/multi/repro_ql_nondet.sh` — assembles the group, runs N draws,
  gates each draw against `work-<s>-ql0819` and every draw against every other.
* `scripts/multi/repro_cmp.py` — Q/L product comparison between two roots by
  member content (never archive bytes, CLAUDE.md M2).

Baseline, `285993 286191`, 10 draws, the driver exactly as doc 81 ran it:

| arm | draws matching `ql0819` |
|---|---|
| in-process jsonnet (what `run_chain_group.sh` does) | **9 of 10** |

Draw 5 is the odd one out and disagrees with all nine others.  Event 285993 is
untouched in every draw; only 286191 — one of the seven — moves.

### What actually differs

Comparing draw 1 against draw 5 for 286191, inside `pctree-evt286191.tar.gz`:

| what | count | nature |
|---|---:|---|
| `pointclouds/namedpcs/3d` arrays | 47 of 53 | **the same 39798 values in a different order** |
| the `live` pctree node array | 1 | **composition changed**: 3775 of 6523 entries, cluster sizes `…631…280…` vs `…627…284…` |
| remainder | 5 | not a pure permutation |

So two effects, and they are different in kind.  The point clouds are merely
**reordered** — no value changes, so this is not floating-point drift.  But the
node array shows **4 points moving between two clusters** while the cluster
count stays at 19.  A clustering decision genuinely flipped; it is not cosmetic.

### The comparator hypothesis — tested and FALSIFIED

`Facade::blob_less` (`clus/src/Facade_Blob.cxx:324`) and
`Facade::cluster_less` (`clus/src/Facade_Cluster.cxx:2725`) both end with an
explicit raw-pointer tie-break, defended in a comment as deliberate:

```cpp
    // ... Randomness is the better choice as we would have a better chance to
    // detect that in some future bug.
    return a < b;
```

Everything built on them is therefore content-order first, **address order on
ties**: `BlobLess`, `BlobSet`, `time_blob_map_t`, `const_blob_point_map_t`,
`cluster_less_functor`, `ClusterLess`, `sort_blobs()`, `sort_clusters()`.  That
is the CLAUDE.md §2 hazard hiding inside containers that look guarded, and it
was the obvious suspect.

It is not the cause here.  `blob_less` can only reach that tail when two
*distinct* blobs agree on wpid, npoints, charge, both slice bounds and all six
u/v/w wire-index bounds.  A census over the imaging blobs — keyed on every one
of those fields that the archives carry, so at least as permissive as the real
comparator — finds **no such pair anywhere**:

| sample | events | blobs | tied pairs |
|---|---:|---:|---:|
| nueCC48 | 48 | 295410 | **0** |
| NCpi0 | 19 | 110252 | **0** |
| the seven failing events | 7 | 41310 | **0** |
| **total** | **74** | **446972** | **0** |

The tie-break is unreachable on this data, including on every event that
actually moved.  **The knob this round was scoped to build is therefore not
indicated, and was not built** — shipping a default-OFF knob whose mechanism has
been disproven would be worse than shipping nothing.  The comparators remain a
real latent hazard and are recorded as such in §Latent below.

### It is a read of memory the program never wrote

> **SUPERSEDED BY ROUND 2 — read §2c before relying on this section.**  The
> observations below are reproducible and stand.  The *conclusion* does not:
> valgrind memcheck is clean in every line of WCT reconstruction code on runs
> landing in **both** states, and separating `--malloc-fill` from `--free-fill`
> shows the answer tracks neither — nor does any other valgrind knob tried.
> Round 2 offers no replacement mechanism.  The section is kept unedited as the
> record of how the round got there.


Three named hypotheses were tested and all three failed:

| hypothesis | test | result |
|---|---|---|
| pointer tie-break in `blob_less`/`cluster_less` | tie census, 446972 blobs, 74 events | **falsified** — zero ties |
| gojsonnet's in-process Go runtime | precompile the config, 16-event group, 10 draws each | **falsified** — 5/10 differ in-process, **9/10** precompiled |
| a thread race | `OPENBLAS_NUM_THREADS=1`, event alone, 10 draws | **falsified** — *fewer* threads is worse: 0/10 differ at default, **5/10** at one thread |

Every perturbation changes the *bias* and none removes the instability.  That is
the signature of a computation whose answer depends on the contents of memory it
has not written.  `MALLOC_PERTURB_` — which makes glibc fill freshly-allocated
memory with a chosen byte **and overwrite freed memory with its complement** —
settles it:

| event 99438, run alone, config precompiled | draws | outcome |
|---|---:|---|
| `MALLOC_PERTURB_=1` | 5 | **all 5 identical to `ql0819`** |
| `MALLOC_PERTURB_=170` | 5 | **all 5 identical to each other, all differ from `ql0819`** |

Each arm is internally stable **in this configuration** and the two arms
disagree, so the answer moves with the byte glibc uses to fill heap memory the
program itself never wrote.  That the fill byte *changes* the answer is the
finding, and it stands.

**Corrected in round 2:** the stronger reading — that the answer is a
deterministic *function* of the fill byte — does not generalise, and this
section originally implied it did.  Fixing `MALLOC_PERTURB_` pins some arms and
not others (§2b): 53793 precompiled pins 3/3 either way, 321101 precompiled at
`=1` still splits 2/1, and **every in-process arm splits** even with the byte
fixed.  Read the table below as one configuration, not as the general law.

**Which of the two kinds it is, is NOT settled here.**  `MALLOC_PERTURB_` fills
fresh allocations *and* poisons freed blocks, so this is consistent with either
a read of **uninitialised** memory or a **use-after-free**.  That distinction
decides the tool: MSan finds the first and is blind to the second; valgrind
memcheck or ASan find both.  This codebase has form for the second kind — doc 76
round 3's `TrackFitting` carried-state UAF, doc pr/102's ghost-drop UAF, doc
pr/97 — so the next round should reach for **valgrind memcheck or ASan**, not
MSan.  Nothing else in this section depends on which it turns out to be.

And the outcome is **binary**.  Hashing the pctree node array — the member whose
composition flips — over every draw of every configuration:

| event | runs | configurations | distinct outcomes |
|---|---:|---:|---:|
| 99438 | 50 | 6 | **2** |
| 286191 | 57 | 4 | **2** |

| event 99438 configuration | jsonnet | state A (= `ql0819`) | state B |
|---|---|---:|---:|
| alone, default threads | **precompiled** | 10 | 0 |
| alone, one BLAS thread | **precompiled** | 5 | 5 |
| alone, `MALLOC_PERTURB_=1` | **precompiled** | 5 | 0 |
| alone, `MALLOC_PERTURB_=170` | **precompiled** | 0 | 5 |
| 16-event group | in-process | 5 | 5 |
| 16-event group | **precompiled** | 1 | 9 |
| *round 2:* alone, default threads | in-process | 2 | 8 |
| *round 2:* alone, `MALLOC_PERTURB_=1` | in-process | 4 | 1 |
| *round 2:* alone, `MALLOC_PERTURB_=170` | in-process | 3 | 2 |

The `jsonnet` column was **missing in round 1 and it mattered**: every "alone"
row was precompiled, which is *not* how `run_chain_group.sh` ran, and the
omission is what made §2b's discrepancy look like a contradiction.

One binary decision flips.  `work-<s>-ql0819` is simply one of its two values —
not a "correct" answer the group mode failed to reproduce.  Every doc-81 §7.1
event that "converged" when re-run did so by landing on state A again, which is
why re-running worked.

*(Round 1 ended that paragraph with "and why 99438, whose bias is near 50/50 in
a group, needed several tries."  That was wrong: doc 81's tries were
**single-event**, not in a group, and it silently moved them onto a
configuration they never ran in.  §2b measures the configuration they actually
used — 8 of 10 draws land in state B — and the sentence is withdrawn.)*

This also explains, without any of them being separately true: why group
context matters (a different allocation history leaves different bytes behind),
why the driver matters (gojsonnet allocates and frees ~31 threads' worth of
heap), why thread count matters, and why `setarch -R` never helped — the defect
is about memory *contents*, not addresses.

**Not fixed in this round, and deliberately not tuned** (CLAUDE.md §5.7).  The
next round has a cheap, deterministic handle it did not have before: two
`MALLOC_PERTURB_` values that flip the answer on demand, a single event that
runs in about a minute on its own, and a binary outcome to bisect against.  A
valgrind memcheck or ASan run on that one event should name the line (not MSan
— see above, it may be a use-after-free).

**Relation to the accepted `match/` verdict.**  `QLMatching`'s build-to-build
residual (~7 marginal bundles at `m_strength_cutoff`) was investigated and
accepted as benign floating-point behaviour, with an explicit "don't sink more
builds into FP-pinning".  That verdict is **not** reopened here: this defect is
in `clus/`, upstream of matching, the differing arrays are exact permutations
rather than perturbed values, and the `MALLOC_PERTURB_` dependence is not an FP
signature.  Whether the `match/` residual has the same underlying cause is worth
a look in the next round, but nothing here settles it.

### Part 2b — event 99438, tested as its own thread

Doc 81 flagged 99438 as a **separate** problem from the other six: the others
converged when re-run as single-event groups, 99438 "did not converge in three
tries" and only matched via the legacy `run_ql_evt.sh` driver, so doc 81
recorded that it "differs *systematically* between the two drivers, not just
occasionally, which is a second thread for the audit round to pull."  Round 1
of this doc folded it into the general story without testing that.  This
section tests it, and the answer is **no — same mechanism, different bias**.

**The discrepancy that had to be resolved first.**  Doc 81: 99438 failed **3 of
3** as a single-event group.  Round 1 of this doc: "alone, default threads —
**10 state A / 0 state B**".  Same nominal configuration, 13 draws split 10-A /
3-B.  Reading the round-1 scratch arms settles it without a single new run:

| arm | jsonnet |
|---|---|
| `d82/e99438-{thr,1thr,perturb1,perturb170}` — *every* round-1 single-event arm | **precompiled** |
| doc 81's three tries, via `run_chain_group.sh` before this doc gave it the knob | **in-process** |

Round 1 never measured *99438 alone, in-process* — the one cell doc 81 was
actually in.  **E1** fills it: 10 draws, in-process, single-event.

| 99438 alone, in-process, 10 draws | state A (= `ql0819`) | state B |
|---|---:|---:|
| E1 | **2** | **8** |

At an 80 % bias toward state B, three tries all landing in B has probability
0.51 — *more likely than not*.  Doc 81 saw an ordinary run of a biased coin and
read it as a systematic difference.  And the "driver difference" is fully
accounted for without any second mechanism: **`run_ql_evt.sh:552` defaults
`SBND_PRECOMPILE_CFG=1` and `run_chain_group.sh` had no such path at all**, so
"legacy driver" and "group driver" were precompiled and in-process
respectively — 0/10 state B versus 8/10 on the same event (E3).

**The control that decides "distinct".**  Driver arms on 99438 alone cannot
show that 99438 is *special*; only the other six can.  **E2** runs each of them
in the configuration doc 81 actually used — alone, in-process, 3 draws:

| event | state A | state B | product that differs |
|---|---:|---:|---|
| mcp1k 286191 | 3 | 0 | — |
| mcp1k 292643 | 2 | **1** | `mabc-all-apa.zip` |
| mcp2k 53793 | 0 | **3** | `pctree` `clustering_tensor_53793_32_array.npy` |
| mcp2k 161043 | 3 | 0 | — |
| mcp2k 321101 | 1 | **2** | `mabc-all-apa.zip` |
| mcp2k 350816 | 3 | 0 | — |
| mcp2k 99438 *(E1, 10 draws)* | 2 | **8** | `mabc-all-apa.zip` |

Three of the six are bistable in exactly the configuration where doc 81
recorded them as "converged", and **53793 landed in the non-reference state 3
times out of 3** — by doc 81's own criterion *more* systematic than 99438, yet
it was written down as converged because the re-runs stopped at the first
match.  99438 is not distinguished.

**Verdict, against the rule written before the runs.**  One mechanism, not two:

* every arm's outcome set is `{A, B}` — an 84-draw census over all of round 2's
  arms (`state_census.py`, digesting member-content hashes of every Q/L product
  a draw wrote) finds **1 or 2 distinct states per event and never 3**;
* `MALLOC_PERTURB_` moves 99438 in the in-process arm as it does elsewhere, and
  53793 precompiled pins **3/3 state A at `=1` and 3/3 state B at `=170`** —
  the same fill-byte control, on a different event and a different product;
* the two drivers differ by `SBND_PRECOMPILE_CFG` and nothing semantic.  The
  group driver additionally passes `multi_event=true`, `evt_subdir`, `rse_map`
  and `save_tensors`, but at one event `multi_event=true` reproduces the legacy
  answer 10 times out of 10 (round 1's precompiled arm), so it is inert here.

**What round 2 did find, and it cuts the other way.**  Fixing the fill byte
does **not** always pin the outcome:

| arm | `MALLOC_PERTURB_=1` (A/B) | `=170` (A/B) |
|---|---|---|
| 99438, precompiled *(round 1)* | 5 / 0 | 0 / 5 |
| 53793, precompiled | 3 / 0 | 0 / 3 |
| 321101, precompiled | **2 / 1** | 0 / 3 |
| 99438, **in-process** | **4 / 1** | **3 / 2** |
| 53793, **in-process** | **1 / 2** | **2 / 1** |

Every in-process arm still splits with the byte held fixed.  That is consistent
with the diagnosis rather than against it: the fill byte fixes the *contents* of
never-written memory, but gojsonnet's Go runtime allocates and frees on ~31
extra threads concurrently with WCT, so *which* block a given request lands on
varies run to run — and a block the process itself wrote earlier carries real
data, not the fill byte.  Precompiled (7 threads, no concurrent allocator) the
sequence is reproducible and the byte usually decides; in-process it is not.

So the honest form of round 1's claim: **the fill byte demonstrably changes the
answer, which is what proves the read; it does not always determine it.**

### Part 2c — round 2 takes the diagnosis apart

Round 1 concluded the flip is **a read of heap memory the program never wrote**.
Round 2 tested that directly, and **it does not survive.**  Two independent
lines say so, and a third kills the obvious alternative.  Nothing here changes
what is *observed* — the flip is real, binary, and heap-dependent — only what it
is *caused by*.

**(i) memcheck is clean in every line of reconstruction code, on both states.**
`run_vg.sh` runs the same single event under valgrind memcheck, config
precompiled, in about four minutes and with no rebuild.  Seven runs.  The WCT
error inventory is *identical* in all of them:

| memcheck finding | count | where |
|---|---:|---|
| Conditional jump on uninitialised value | 98 | ROOT's `TFile::Open` path — `SCEFieldTH3::configure` is only the WCT *caller* |
| Invalid read of size 4 | 1 | ROOT's `DeleteChangesMemoryImpl`, which frees a block and reads it back **on purpose** |
| Invalid read of size 32 | 1 | Go's `indexbytebody` SIMD over-read of a 42-byte string, `Persist::Parser` at startup |
| **anything in `libWireCellClus` / `libWireCellImg` / `libWireCellAux`** | **0** | — |

All of it is third-party, all of it at configuration time, none of it per-point.
This is not a weak negative: detecting a conditional that depends on undefined
memory is precisely what memcheck does, and **one of the clean runs is a run
that produced the anomalous state** (below).  If the flip were decided by
reading never-written or freed memory, this is where it would appear.

**(ii) no valgrind allocator knob controls it — the event is bistable under
valgrind too.**  Under valgrind `--malloc-fill` and `--free-fill` are
*independent*, which glibc's single `MALLOC_PERTURB_` byte is not, so a 2×2
should separate "never written" from "freed":

| run | `--malloc-fill` | `--free-fill` | freelist | outcome |
|---|---|---|---|---|
| A2 | `0xfe` | `0x01` | 500 MB | state B |
| B2 | `0x55` | `0xaa` | 500 MB | state B |
| C2 | `0xfe` | `0xaa` | 500 MB | state B |
| D2 | `0x55` | `0x01` | 500 MB | state B |

The outcome tracks **neither** fill — four maximally different content
regimes, one answer.  A single pair run without fills then *appeared* to
implicate `--freelist-vol`, the knob controlling how long valgrind withholds
freed blocks from reuse (20 MB → state A, 500 MB → state B), and an earlier
draft of this section built an "address reuse is the channel" conclusion on it.

**That was one run per arm, and replicating it at n=3 destroys it:**

| freelist, no fills | state A | state B |
|---|---:|---:|
| 20 MB | 2 | 1 |
| 500 MB | 2 | 1 |

Identical.  The event is simply **bistable under valgrind as well**, at about
the same rate, and the 20 MB/500 MB pair was two draws of that coin.  *Recorded
because the wrong version was committed and pushed before the replicate ran —
n=1 per arm is not a lever, and the fill matrix's 4/4 is no longer strong
evidence either once the underlying rate is known to be ~1-in-3.*

So: no valgrind knob varied here controls the outcome.  Combined with (i), what
this rules out is a bad read that memcheck can see; it does **not** identify the
channel, and this round does not.

**(iii) it is not floating point either.**  Comparing the two states array by
array (`numdiff.py`) inside `pctree-evt99438.tar.gz`:

| | arrays |
|---|---:|
| byte-identical | 356 |
| **pure permutation** — same multiset, different order | 48 |
| **values genuinely differ** | 6 |

and the largest float difference is `max rel = 8.2e-2` over 28 of 31536 entries
— **8 %, not 8 ulp**.  Points really are assigned to different clusters.  A
rounding or SIMD-alignment story would show differences near 1e-16; this is a
decision flip, not drift.

**(iv) the order-dependence census comes back empty.**  Doc 81 proposed the
mechanism was an address-ordered walk of an unguarded pointer-keyed container,
and named five: `flash_t0_group` and `used_clusters` in
`clustering_{close,extend,cathode_connect,parallel_prolong,examine_bundles}.cxx`.
A sweep of `clus/`, `match/` and `img/` refutes all five — **every one is
lookup-only** (`.at()` / `.find()` / `.insert()`), never walked, so its order is
unobservable.  `used_clusters` does not exist at all in three of the five files.
Nor is there a surviving hit elsewhere: the containers that *are* iterated
(`TrackFitting.cxx:2397`, `retile_cluster`'s `blobs_to_remove` ×3,
`QLMatching.cxx:4667`, `clustering_deghost.cxx:262`, `NeutrinoDeghoster.cxx:61`)
are each order-neutral, most by an explicit tie-break or an immediate sort.
`img/` holds no pointer-keyed container at all.

### What survives, and what the next round should do

Round 2's contribution is subtraction.  Five mechanisms are now excluded by
measurement rather than argument:

| mechanism | how it died |
|---|---|
| `blob_less` / `cluster_less` pointer tie-break | 0 ties in 446972 blobs, 74 events (round 1) |
| a read of never-written or freed memory | memcheck clean in all WCT reconstruction code, on runs landing in **both** states |
| floating-point drift | differences are **8 % relative**, not ulps |
| address-ordered iteration of a pointer-keyed container | `clus/`+`match/`+`img/` census empty; doc 81's five suspects are lookup-only |
| a stale pointer key in the three caches this doc first nominated | all three are safe — checked, below |

The stale-key idea is worth stating carefully because an earlier draft of this
section nominated three specific sites and **all three are fine**:

* `Facade_Cluster.cxx:3698` and `NeutrinoPatternBase.cxx:952`
  (`unordered_map<const Blob*, double> blob_total_charge`) are **function-local**
  memos, built from `blob_with_point(i)` on a live cluster and destroyed with the
  call.  A key cannot outlive its object.
* the four mutable per-cluster memos at `QLMatching.h:922-955` are **cleared at
  the top of `operator()`**, per event, with a comment naming exactly this risk.

The *class* is not dead — the Q/L pipeline does destroy and create clusters
mid-run (`ClusteringSeparate`, `ClusteringDeghost`, `ClusteringProtectOverclustering`
all appear in the compiled pipeline), so address reuse within one job is real —
but no instance of it has been found, and nothing in this round's data points at
it specifically.  **The mechanism is unidentified.  Say so; do not adopt the
nearest surviving story.**

What the next round has that this one did not:

* **a shorter list.** The five rows above do not need re-running.  In
  particular, **do not run memcheck on this again** — seven runs, clean where it
  matters — and do not re-test doc 81's five container suspects.
* **a warning about n.** Both this round's wrong turn and doc 81's were the same
  error: a conclusion from one or three draws of a coin that lands ~1-in-3 the
  other way.  Anything claimed about this event needs N draws with N stated.
* **the handle, unchanged**: one event, ~1 min a draw, a binary outcome, and
  `MALLOC_PERTURB_` / precompile / thread count to shift the *bias* — no longer
  believed to *determine* the answer, but still enough to make either state
  appear on demand.
* the honest next step is probably not another environment knob but
  **bisection inside the pipeline**: the compiled pipeline is 16 named
  clustering stages plus matching, the outcome is binary, and dumping the
  pctree after each stage would localise the flip to one stage before anyone
  reads a line of its source.

**No fix is proposed and none should be guessed at** (CLAUDE.md §5.7).

### Part 2d — round 3 finds and fixes the mechanism

Round 2's own instruction was to localise by pctree bisection, not another
environment knob.  Round 3 did something narrower and cheaper first: the
per-APA clustering archives (`mabc-apa0-face0.zip`, `mabc-apa1-face0.zip`) for
draw1 (state A) and draw5 (state B) of the 286191 pair were **byte-identical
member-content hashes**, while `mabc-all-apa.zip` and `pctree-evt286191.tar.gz`
differed.  All 16 per-APA clustering stages are therefore innocent on this
event; the flip is born strictly downstream of them, in
`FlashTensorToOpticalPCs → QLMatching → PointTreeMerging → MABC(all-apa)`. That
already excludes the class round 2 spent most of its budget on.

**The site.** `match/src/QLMatching.cxx`, `QLMatching::rescue_empty_flashes()`
(§I, SBND production ON — `empty_rescue: true`,
`cfg/pgrapher/experiment/sbnd/qlmatching.jsonnet:289`). Before the rescue can
run it records where each cluster is currently matched:

```cpp
std::map<Cluster*, std::pair<Opflash*, double>> matched;
for (auto& kv : run.flash_bundles_map) {
    for (auto& b : kv.second) matched[b->get_main_cluster()] = {kv.first, metric(b)};
}
```

`run.flash_bundles_map` is `std::map<Opflash*, TimingTPCBundleSelection>` —
**heap-address order**.  When a cluster is the main cluster of surviving
bundles on *two* flashes — unremarkable; "one flash per cluster" is exactly
what this function goes on to *enforce*, not something guaranteed on entry —
the second bundle's `matched[C] = …` plain-overwrites the first, and *which*
flash's `(flash, metric)` pair survives depends on which `Opflash` object this
walk visits second, i.e. on the two objects' relative heap addresses. That
recorded pair is load-bearing a few lines later (`steal_bar =
mit->second.second`) in deciding whether a later rescue candidate can steal
the cluster. This is exactly the ten-other-call-sites-hardened, two-still-not
class this file already has a fix for: `flash_iter_order()`
(`QLMatching.cxx:4688`, flashes sorted by `get_flash_id()`) is used at every
other walk of `flash_bundles_map` in this file; this loop and its sibling
`rescue_empty_flashes_shared()` were the two that were not.

**Causal proof, not inference.** An env-gated probe
(`WCT_QLRESCUE_CENSUS=1`) logged, for every overwrite, the cluster ident and
the before/after flash id — no log call inside the metric loop itself, so it
could not perturb the allocation history under study. On mcp1k 286191, over
20 draws, the *count* of overwrites was identical in every draw regardless of
outcome state (`0 of 7`, `0 of 14`, `0 of 8`, `2 of 15`) — ruling out a naive
"more overwrites in state B" story. The *identity* was not:

```
state A (draw1, draw19): cluster 1 was flash 17, now overwritten by flash 21
                          cluster 3 was flash 6,  now overwritten by flash 3
state B (draw20):        cluster 1 was flash 21, now overwritten by flash 17
                          cluster 3 was flash 6,  now overwritten by flash 3
```

Same two clusters, same two flash pairs, same binary, same input — only
cluster 1's final owner flips (`matched[1]` = flash 21 in A, flash 17 in B),
tracking the two `Opflash` objects' relative heap addresses between runs.
Cluster 3 is unaffected because its overwrite is not a tie (flash 3 always
wins in the census, both states) — consistent with an address-order channel
that only matters when it is the sole thing deciding the outcome.

**Fix.** Walk `flash_iter_order(run.flash_bundles_map)` instead of the raw
map in `rescue_empty_flashes()`, and the identical raw walk in
`rescue_empty_flashes_shared()` (currently **inert everywhere** —
`empty_rescue_shared`/`m_shared_flash` combination is never set true by any
cfg in the tree today; fixed anyway because it is the same defect in the same
file, and it costs nothing to fix while looking at it). Unknobbed: this is a
determinism fix on an already-nondeterministic value, the same category as
doc 76 r3 and doc 81's `GridTiling` fix, not a new behavior.
`match/src/QLMatching.cxx`.

**Verification.**

| gate | result |
|---|---|
| `wcdoctest-match` | 36/36 assertions, 4/4 cases |
| repro (mcp1k 285993/286191, 25 draws, no fill/env tricks) | **0 of 300** draw-vs-draw pairs differ (was ~1 in 9–10 before the fix) |
| mcp1k 25-event manifest (incl. bistable 286191, 292643 + 23 controls) | **100/100** archives member-identical to `work-mcp1k-ql0819` |
| mcp2k 30-event manifest (incl. bistable 53793, 99438, 161043, 321101, 350816 + 25 controls), 1 draw | only **53793** differs from the reference (1/410 pctree members) — every control and every other listed event matches |
| mcp2k 5 bistable events, 12 repeated draws | **0 of 66** draw-vs-draw pairs differ for any of the 5 events — each has converged to a single answer. 53793's answer differs from `ql0819` on **all 12/12** draws (its own new fixed point, not a residual flip); 99438/161043/321101/350816 match `ql0819` on all 12/12 |

No control event moved in either manifest — the only mover anywhere is
53793, a member of the already-known-bistable set, and it is no longer
bistable. This is the owner-approved bar: fix unknobbed if the read is named,
gate on the standard manifest, require the only movers to be the
already-known bistable events. `53793`'s new answer legitimately differs from
its old `ql0819` archive because `ql0819` itself was recorded from one
arbitrary draw of a formerly-bistable event, before this fix existed — not
because the fix is wrong.

Freshness proof: `build/match/libWireCellMatch.so` and
`local/lib/libWireCellMatch.so` both `13:43:50`, source edit `13:43:11`
(both copies checked — `LD_LIBRARY_PATH` puts `build/` ahead of `local/lib`
here, doc 82's own gotcha).

**What this settles about round 2.** The "mechanism unidentified" verdict
(§2c/§2d) stands *for the memcheck-clean, five-container-excluded space it
searched* — QLMatching's `flash_bundles_map` walk is match-package code, not
in `clus/`, `img/` or the five suspects doc 81 named, so nothing round 2
excluded is contradicted. It was simply outside the search area: round 2's
census covered `clus/`+`match/`+`img/` for **iterated raw-pointer-keyed**
containers, and this one is keyed on the concrete `Opflash*`
class member typedef via `FlashBundlesMap`, walked with the ordinary
`for (auto& kv : map)` idiom rather than anything the earlier greps were
shaped to catch.

### Latent: the comparator tie-breaks

Not the cause of anything observed, but recorded because the census had to be
done to rule it out, and because the hazard is real the moment the data changes:

`Facade::blob_less` (`Facade_Blob.cxx:324`) and `Facade::cluster_less`
(`Facade_Cluster.cxx:2725`) end in `return a < b`, so `BlobSet`,
`time_blob_map_t`, `const_blob_point_map_t`, `sort_blobs()`, `sort_clusters()`
and every `cluster_less_functor` / `ClusterLess` container is address-ordered on
a tie.  Also unguarded and iterated: `retile_cluster.h:127`
`std::set<const Blob*> blobs_to_remove` — whose sibling `improvecluster_1.h:46`
returns a `std::vector` with an explicit "deterministic across runs regardless
of heap layout" comment, so this is a missed conversion, not a design choice.
Three arg-max loops at `clustering_connect.cxx:491/499/507` resolve ties by map
order, where `clustering_deghost.cxx:269-271` shows the correct pattern.

One trap for whoever fixes these: `Facade_Grouping.cxx:130-137` sorts
`clusters` and then iterates `children()`, so `sort_order == "size"` is a silent
no-op.  "Fixing" it *before* the comparator tie-breaks would make cluster
**idents** address-dependent, and idents are what the genuinely-safe
comparators (`ClusterIdentLess`, `ClusterPtrCmp`, `sortbysec`) rely on.

## Part 3 — `run_chain_group.sh` gets `SBND_PRECOMPILE_CFG`, default OFF

`run_chain_group.sh` was the only driver in the tree handing `wire-cell` a
`.jsonnet` rather than precompiled JSON, so gojsonnet ran in-process and left a
64-thread Go runtime alive for the whole job.  Measured on a live job: **38
threads in-process versus 7 precompiled.**  `run_ql_evt.sh:552` and
`run_pr_chain_batch.sh:1617/1751` both precompile, for the SIGSEGV doc pr/97 §5
found at ~120 s of process life — and a group job runs far longer than that.

Two other latent problems in the same code are closed by the same edit: the
three entry points were named *bare*, so they resolved through the **CWD** (the
runner silently required being invoked from `sbnd_xin/`), and wcsonnet and
wire-cell could in principle resolve different files.  Both now take one
absolute path string, used for both.  (The `sbnd_xin/` copies are thin
re-exports of the `cfg/pgrapher/experiment/sbnd/` modules; compiled with the
same TLAs they produce byte-identical JSON, checked directly.)

**It defaults OFF here**, unlike the other two drivers, and that is deliberate:
part 2 shows the Q/L answer on a bistable event is a function of the process's
allocation history, and precompiling changes exactly that.  Turning it on is a
stage-A behavior change that needs its own byte-identity gate on the standard
manifest, not a free win — so this round ships the capability, not the flip.

Verified on one 16-event group, `--from ql`, both ways:

| | rc | compiled JSON written | vs `work-mcp1k-ql0819` |
|---|---|---|---|
| knob OFF (default) | 0 | no — in-process, as before | 62/64: only **286191** differs |
| `SBND_PRECOMPILE_CFG=1` | 0 | yes | **IDENTICAL 64/64** |

ON versus OFF differ on 286191 alone, and it lands on `eed0bc8753`, its known
state B — the part-2 defect, not a runner regression.

## Part 4 — the rc=0 coverage defect in group mode

Neither "log related" nor "Q/L determinism", which is how it survived doc 81
and nearly survived this doc too: it is a **runner** defect, and it falls in the
gap between this round's two labels.  Doc 81 §7 recorded it in one line and left
it open.

### Symptom

`work-<s>-prod0825/pr_evt<ID>/rc.txt` says `rc=0` for every member of every
group that ran.  A coverage count taken from those files therefore always
reports 100 %, whatever actually happened.

### Root cause

`run_pr_chain_batch.sh`, in `process_group()`.  The **group's** own exit code is
handled correctly — at `:1772-1776` a non-zero wire-cell stamps the real rc into
every member's `rc.txt` and returns 1.  The defect is narrower than "always
rc=0" and worth stating precisely:

```bash
        echo "rc=0" > "$PRDIR/rc.txt"        # unconditional, inside the per-event loop
```

**Whenever the group process exits 0, every member is stamped `rc=0` — including
a member for which the job produced nothing.**  One wire-cell process now covers
16 events; it can complete and still leave a single event's products unwritten,
and that is exactly the case the file is silent about.  `nusel_extract.py`'s own
exit code is discarded on the next line too (`2>>"$PRDIR/stdout.log"` with no
check), so a member with a missing or unparsable table also reads as rc=0.

### Why it hid

Per-event mode never had the failure mode: one process, one event, and its rc
*is* the event's rc.  Group mode inherited the line unchanged, where the
identity no longer holds.  And it fails **silently and optimistically** — the
wrong answer is "everything is fine", which nothing downstream contradicts.

### Fix

Take the per-event verdict from the per-event product wire-cell was told to
write.  `save_tensors=$OUTROOT/pr_evt%1%/pctree-pr-evt%1%.tar.gz` is
unconditional in the group path, so its absence is unambiguous:

```bash
        if [ -s "$PRDIR/pctree-pr-evt${evt}.tar.gz" ]; then
            echo "rc=0" > "$PRDIR/rc.txt"; _NOK=$((_NOK+1))
        else
            echo "rc=1" > "$PRDIR/rc.txt"; _MISSING+=("$evt")
            echo "[group $GIDX] event $evt: NO per-event product though the group exited 0" >&2
        fi
```

and the group tail now reports the count rather than asserting it, returning
non-zero when a member is missing so the group lands in `BATCH_FAIL_LIST`
(`_runlib.sh:179`) instead of passing quietly.

### Verification

`smoke_rc.sh` lifts the decision block **out of the runner with `sed`** so the
test cannot drift from the source, then runs it over a real four-event group
reached through read-only symlinks, once complete and once with one member's
product removed:

```
[complete]    _NOK=4/4  missing=''        rc.txt: rc=0 rc=0 rc=0 rc=0
[onemissing]  _NOK=3/4  missing='166738'  rc.txt: rc=0 rc=1 rc=0 rc=0
              [group 0] event 166738: NO per-event product though the group exited 0
```

Existing arms are untouched — no `rc.txt` under any `work-*` was written, and
the change affects future runs only.

### The standing rule, which does not change

**Count group-mode coverage from real per-event product existence, never from
`rc.txt`.**  Doc 81 §8 already did exactly this, which is why its 3067-event
coverage numbers are sound despite the defect being live at the time.  The fix
makes `rc.txt` *agree* with that rule; it does not make it the authority.  Any
arm produced before this commit still has uniformly optimistic `rc.txt` files
and must be counted from products.

## What shipped, and what did not

| | status |
|---|---|
| `util` shared-handle file sink + fail-first doctest | **SHIPPED**, products gated byte-identical |
| `scripts/multi/log_tear_scan.py` | shipped |
| `scripts/multi/{merge_group_products.py, repro_ql_nondet.sh, repro_cmp.py}` | shipped |
| `run_chain_group.sh` `SBND_PRECOMPILE_CFG` | shipped **default OFF** — knob-off is byte-identical |
| `stable_facade_order` knob (round 1's approved scope) | **NOT built** — its mechanism was falsified |
| *round 2:* `run_pr_chain_batch.sh` per-event rc from product existence | **SHIPPED** — smoke-tested both ways, existing arms untouched |
| *round 2:* `scripts/multi/{run_e2.sh, run_vg.sh, state_census.py, smoke_rc.sh, numdiff.py}` | shipped |
| *round 2:* event 99438 as a separate thread | **TESTED, and it is not one** (§2b) |
| *round 3:* `match/src/QLMatching.cxx` — `flash_iter_order()` in `rescue_empty_flashes()` + `rescue_empty_flashes_shared()` | **SHIPPED, unknobbed** — the Q/L flip mechanism, found and fixed (§2d) |

Two honest retractions, recorded because the intermediate claims were made out
loud during the round:

1. "Precompiling the config fixes the Q/L non-determinism" — from the 2-event
   pair at 0/10 versus 1/10.  The 16-event group refutes it: **9/10** differ
   precompiled versus 5/10 in-process.  The 2-event result was small-N on an
   event whose flip rate is ~10%.
2. "The comparator pointer tie-break is the prime suspect" — carried from the
   plan.  446972 blobs say it is unreachable.

Round 2 adds two more, both retracting round 1 rather than an intermediate:

3. **"The Q/L flip is a read of memory the program never wrote."**  Memcheck is
   clean in every line of WCT reconstruction code on runs landing in *both*
   states, and `--malloc-fill` / `--free-fill` are inert.  §2c.
4. **"The defect is about memory contents, not addresses."**  Round 1's
   supporting `setarch -R` argument is still bad — `-R` removes the ASLR *base*
   and leaves relative heap layout untouched — but round 2 cannot supply the
   replacement.  An earlier draft of §2c, **committed and pushed as `f902e9b`**,
   claimed `--freelist-vol` identified address *reuse* as the channel.  That
   rested on one run per arm; replicating at n=3 gives **2 A / 1 B on both
   settings**, so the knob does nothing and the claim is withdrawn.  Round 2
   left the channel unidentified — **round 3 (§2d) identifies it**: not memory
   contents, and not free-list reuse either, but the iteration order of a live
   `std::map<Opflash*, …>` deciding a last-write-wins overwrite.  Round 2's
   instinct that it was address-shaped was right; both of its specific guesses
   at *which* address-shaped thing were wrong.
5. **"The stale pointer key is at `blob_total_charge` / `QLMatching.h:922-955`."**
   Same draft, same commit.  All three are safe: the two `blob_total_charge`
   memos are function-local, and the QL caches are cleared per event with a
   comment naming this exact risk.  The *class* of bug is not excluded; these
   instances are — the real instance is `QLMatching.cxx`'s
   `rescue_empty_flashes()`, a different container in the same file (§2d).

### Open, for the next round

* ~~Localise the flip to a pipeline stage before theorising about it again.~~
  **DONE, round 3 (§2d).** The per-APA clustering archives were the
  localisation: byte-identical between draw1/draw5, which put the flip
  strictly downstream of all 16 clustering stages, in matching. **Do not
  re-run memcheck** (seven runs, clean where it matters) and do not re-test
  doc 81's five container suspects (§2c iv) — neither is where this lived.
* **Two more address-ordered hazards found while hunting this one, not yet
  fixed, both PR-job (not the Q/L job that flips) so out of this round's gate
  scope:** `NeutrinoDeghoster.cxx:59-66` sorts a `map<Facade::Cluster*,double>`
  by length with **no ident tie-break** (`sortbysec`) after iterating the map
  raw at `:59` — a genuine, live address-order leak, not a laundered one.
  `clustering_neutrino.cxx:191` and `clustering_connect.cxx:198/746` sort
  clusters by `get_length()` alone, unlike every sibling length-sort in the
  same files, which all carry an `ident()` tie-break.
* **`SBND_PRECOMPILE_CFG=1` for stage A**, gated on the standard manifest.  It
  is worth doing on its own merits — it cuts the job from 38 threads to 7 and
  closes doc pr/97's SIGSEGV hazard, which is currently unmitigated in the only
  driver that does not precompile — but it is a behavior change on bistable
  events, not a free win.
* **The comparator tie-breaks and `Facade_Grouping.cxx:130-137`**, in the order
  given above.
* Whether `QLMatching`'s accepted "~7 marginal bundles" FP residual shares this
  cause.  Round 2 makes this *less* likely, not more: the two states here differ
  by **8 % relative**, not by ulps, so whatever this is, it is not the floating
  point that verdict was about.
* **The doc-81 §7.1 event list is not a list of unstable events.**  §2b shows
  292643, 321101 and 53793 flip in the very configuration doc 81 marked
  "converged", and 53793 landed in the non-reference state 3 times of 3.  A
  campaign that re-runs until an event matches records only the last draw.  Any
  future stability claim needs N draws per event, not one.

### Status flags

* The `util` logging change: **products byte-identical** — pr85 32/32, pr94
  16/16, nusel 178/178, Q/L 64/64 on two of three draws.
* The third Q/L draw: **NOT bit-identical, and explained** — one already-bistable
  event landed on its known second state.  Not a revalidation trigger for the
  logging change; it is the part-2 defect.
* `work-*-{grp,prod}0825` remain the current reference arms and are untouched.
* **Round 2 changed no C++ and no reconstruction output.**  The only code change
  is `run_pr_chain_batch.sh`'s per-event `rc.txt`, which is bookkeeping: it is
  written *after* wire-cell has finished and is read by nothing in the chain.
  No gate was needed and none is claimed.
* Round 2's runs are all in `/home/xqian/tmp/d82r2/` scratch.  Nothing under
  `work-*`, `abtest/snap/`, `sweep/` or any `decisions*` tree was written to,
  and no `rc.txt` in any existing arm was touched (CLAUDE.md M13).
* **Every `rc.txt` written before this commit is uniformly optimistic** and must
  not be used for coverage retrospectively — count from products (Part 4).
* **Round 3 changes reconstruction output on the formerly-bistable events, and
  only those.** `match/src/QLMatching.cxx` is a real C++ behavior change —
  gated: mcp1k 25-event manifest 100/100 identical, mcp2k 30-event manifest
  only the known-bistable 53793 differs, every formerly-bistable event
  converges to one answer under repeated post-fix draws (§2d table).  No
  control event moved in either manifest.  `wcdoctest-match` 36/36.  This is
  **NOT byte-identical** on the bistable events by design (that was the bug);
  it is byte-identical on every event whose Q/L answer never depended on
  `Opflash*` heap order.
* Round 3's runs are all in `/home/xqian/tmp/d82r3/` scratch.  Nothing under
  `work-*`, `abtest/snap/`, `sweep/` or any `decisions*` tree was written to.
