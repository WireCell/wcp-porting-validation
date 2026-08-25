# doc 82 — the two defects doc 81 left open: log tearing and Q/L non-determinism

Doc 81 shipped the group-mode re-baseline of the four SBND production samples
byte-identical, and recorded two residuals it deliberately did not chase:

* **§5 / §8.2** — 72 of 34835 `nusel-table.tsv` rows differ between arms, every
  one in a log-parsed column, because WCT log lines *tear*.
* **§7.1** — multi-event Q/L is run-to-run non-deterministic on marginal
  events; 7 of 3067 had to be re-run outside the group to reach the stage-A
  PASS.

This round root-causes both.  Neither cause was what the earlier write-ups
assumed.

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
#   -> =1   : 5/5 IDENTICAL to work-mcp2k-ql0819
#   -> =170 : 5/5 identical to each other, 5/5 DIFFER from ql0819

# --- 3. the original doc-81 pair, for the ~10%-flip version of it -------
DRAWS=10 ./scripts/multi/repro_ql_nondet.sh \
    work-mcp1k-grp0825 /home/xqian/tmp/d82/base-inproc 285993 286191
```

The reproducer rebuilds a group from surviving per-event products, so it needs
no reco1 file and no re-imaging.

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

Each arm is internally *perfectly* stable and the two arms disagree.  The
result is a deterministic function of the byte glibc uses to fill heap memory
the program itself never wrote.

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

| event 99438 configuration | state A (= `ql0819`) | state B |
|---|---:|---:|
| alone, default threads | 10 | 0 |
| alone, one BLAS thread | 5 | 5 |
| alone, `MALLOC_PERTURB_=1` | 5 | 0 |
| alone, `MALLOC_PERTURB_=170` | 0 | 5 |
| 16-event group, in-process | 5 | 5 |
| 16-event group, precompiled | 1 | 9 |

One binary decision flips.  `work-<s>-ql0819` is simply one of its two values —
not a "correct" answer the group mode failed to reproduce.  Every doc-81 §7.1
event that "converged" when re-run did so by landing on state A again, which is
why re-running worked and why 99438, whose bias is near 50/50 in a group,
needed several tries.

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

## What shipped, and what did not

| | status |
|---|---|
| `util` shared-handle file sink + fail-first doctest | **SHIPPED**, products gated byte-identical |
| `scripts/multi/log_tear_scan.py` | shipped |
| `scripts/multi/{merge_group_products.py, repro_ql_nondet.sh, repro_cmp.py}` | shipped |
| `run_chain_group.sh` `SBND_PRECOMPILE_CFG` | shipped **default OFF** — knob-off is byte-identical |
| `stable_facade_order` knob (the round's approved scope) | **NOT built** — its mechanism was falsified |
| the bad read itself (uninitialised *or* freed) | **NOT fixed** — reported, not tuned (CLAUDE.md §5.7) |

Two honest retractions, recorded because the intermediate claims were made out
loud during the round:

1. "Precompiling the config fixes the Q/L non-determinism" — from the 2-event
   pair at 0/10 versus 1/10.  The 16-event group refutes it: **9/10** differ
   precompiled versus 5/10 in-process.  The 2-event result was small-N on an
   event whose flip rate is ~10%.
2. "The comparator pointer tie-break is the prime suspect" — carried from the
   plan.  446972 blobs say it is unreachable.

### Open, for the next round

* **Find the bad read.**  Handle: `MALLOC_PERTURB_=1` versus `=170` on event
  99438 alone, ~1 min per run, binary outcome.  Use **valgrind memcheck or
  ASan** — it may be a use-after-free rather than an uninitialised read, and
  MSan would be blind to that case.
* **`SBND_PRECOMPILE_CFG=1` for stage A**, gated on the standard manifest.  It
  is worth doing on its own merits — it cuts the job from 38 threads to 7 and
  closes doc pr/97's SIGSEGV hazard, which is currently unmitigated in the only
  driver that does not precompile — but it is a behavior change on bistable
  events, not a free win.
* **The comparator tie-breaks and `Facade_Grouping.cxx:130-137`**, in the order
  given above.
* Whether `QLMatching`'s accepted "~7 marginal bundles" FP residual is in fact
  this same bad read.

### Status flags

* The `util` logging change: **products byte-identical** — pr85 32/32, pr94
  16/16, nusel 178/178, Q/L 64/64 on two of three draws.
* The third Q/L draw: **NOT bit-identical, and explained** — one already-bistable
  event landed on its known second state.  Not a revalidation trigger for the
  logging change; it is the part-2 defect.
* `work-*-{grp,prod}0825` remain the current reference arms and are untouched.
