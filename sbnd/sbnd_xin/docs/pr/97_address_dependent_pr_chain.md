# doc pr/97 — address-dependent behaviour in the PR chain

Status: **TWO DEFECTS FOUND, BOTH FIXED, GATED.** One ships behind a
default-OFF knob (`shower_nv_main_pi_init`), one ships unknobbed because the
legacy read is undefined behaviour with an unambiguous intended value; both
prove byte-identical on the standard nueCC48 + NC pi0 manifests. The
`18255-178410` crash from doc pr/95 is now **reproducible on demand at `-j 1`**
and is **not** a concurrency bug — that diagnosis is corrected here.

Origin: owner checklist item *"(non-determinstic, 178410, also tests)"*, i.e.
the two non-determinism items left open by doc pr/95 (§4 "not deterministic",
§7 "no determinism floor measured") plus the flapping unit-test total.

## Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
git log --oneline -1                       # f0e69780 at the start of this round
./wcb build -p && ./wcb install --notests -p        # rc=0 both

# 1. THE TEST SYMPTOM: the clus suite's assertion TOTAL is not stable
for i in $(seq 1 6); do ./build/clus/wcdoctest-clus 2>/dev/null \
    | grep -E '^\[doctest\] assertions:'; done
#   -> 2215, 2215, 2214, 2215, 2215, 2214   (0 failures either way)

# 2. bisect to the one test case whose assertion count moves
for i in 1 2 3; do ./build/clus/wcdoctest-clus --reporters=xml > /home/xqian/tmp/dt_xml_$i.xml; done
#   per-<TestCase> successes diff -> "pattern_recognition shower_clustering_with_nv [B]": 4,5,5

# 3. it is ADDRESS-dependent, not time- or cwd-dependent
TC='pattern_recognition shower_clustering_with_nv [B]'
for i in $(seq 1 15); do ./build/clus/wcdoctest-clus --test-case="$TC" 2>/dev/null \
    | awk '/^\[doctest\] assertions:/{print $3}'; done          # 4 5 5 5 5 5 5 4 5 ...
for i in $(seq 1 15); do setarch x86_64 -R ./build/clus/wcdoctest-clus --test-case="$TC" \
    2>/dev/null | awk '/^\[doctest\] assertions:/{print $3}'; done  # 4 x15, always

# 4. independent confirmation, and the second defect
valgrind --error-limit=no --track-origins=yes --num-callers=25 \
   --log-file=/home/xqian/tmp/vg_shower.log ./build/clus/wcdoctest-clus --test-case="$TC"
#   -> 16 errors / 10 contexts: 5 uninitialised-value contexts originating in
#      shower_clustering_with_nv_from_vertices (D1) + 3 "Invalid read of size 4"
#      in TrackFitting::get_channel_for_wire (D2) + 2 Go-runtime false positives

# 5. THE 178410 CRASH, at -j 1, ASLR off, ONLY the environment size varies
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
./pr97_layout_sweep.sh work-pr97-pad 0 16 64 256 1024 4096      # six fresh roots
#   pad16  -> rc=139  maxrss_kb=2310648   <-- crashes
#   others -> rc=0    maxrss_kb~=675000
#   ...but see sec 5: the padding is NOT the trigger, the rate is ~4 %/run.
#
# 5b. WHAT THE CRASH ACTUALLY IS (sec 5.4-5.6).  Run it under gdb until one dies:
#   ./pr97_gdb_loop.sh work-pr97g-r 48 6      # 2 captures in 48 runs
#   grep -l 'received signal SIG' work-pr97g-r*/.log.log   # NOT rc=139 under gdb
#   -> SIGSEGV on a libgojsonnet Go-runtime thread at PC=0x0, while WireCell's
#      main thread is healthy inside hough_transform.
#   thread census (the Go runtime is there only because the config is .jsonnet):
#   ls /proc/$(pgrep -f 'wire-cell.*perevt.jsonnet')/task | wc -l   # 65
#   ls /proc/$(pgrep -f 'wire-cell.*precompiled')/task    | wc -l   # 1
#   the arms:
#   ./pr97_json_loop.sh    work-pr97h-r 100 6   # config precompiled by wcsonnet
#   ./pr97_godebug_loop.sh work-pr97i-r  60 5   # GODEBUG=asyncpreemptoff=1
#   ./pr97_gogc_loop.sh    work-pr97j-r 100 5   # GOGC=off (kills the 120 s forced GC)

# 6. the gates (see sec 7)
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
PR_JOBS=6 ./run_pr_chain_batch.sh work-nuecc48-ql0819 work-pr97gate-nuecc48 data
PR_JOBS=3 ./run_pr_chain_batch.sh work-ncpi0-ql0819   work-pr97gate-ncpi0   data
# 7. the uBooNE leg (D2 lives in TrackFitting, which the uBooNE PR chain runs):
#    two labels from the SAME tree, with and without this round's 9 files
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/qlport/scripts
./sweep_5384.sh pr97_after 6            # fixes in
#   git stash push <the 9 files>; ./wcb build -p && ./wcb install --notests -p
./sweep_5384.sh pr97_before 6           # fixes out
./ab_check.sh pr97_after pr97_before    # ZIPS 35/35, TAGGER identical=35 diff=0
python3 scripts/pr85_hash_gate.py work-nuecc48-prod0819 work-pr97gate-nuecc48   # PASS 96/96
python3 scripts/pr85_hash_gate.py work-ncpi0-prod0819   work-pr97gate-ncpi0     # PASS 38/38
SBND_MAIN_PI_INIT=true PR_JOBS=6 ./run_pr_chain_batch.sh work-nuecc48-ql0819 \
    work-pr97on-nuecc48b data                                   # knob-on footprint
```

Arms: `work-pr97gate-{nuecc48,ncpi0}` (knob OFF, the byte-identical gate),
`work-pr97on-nuecc48b` (knob ON), `work-pr97-pad{0,16,64,256,1024,4096}` and
`work-pr97-{cor16,vgql}` (the 178410 layout probes). Q/L roots are the
single-epoch `work-{nuecc48,ncpi0,mcp2k}-ql0819` from doc pr/95.

---

## 1. Symptom: the unit-test total is not a constant

`./build/clus/wcdoctest-clus` reports **2215 assertions on some runs and 2214 on
others**, 0 failures either way. doc pr/94 §9.11 already noticed the 2214/2215
discrepancy and recorded it as "the one-assertion delta predates round 3". That
framing is now corrected: **it is not a per-commit constant at all** — the same
binary gives both numbers, and the run-to-run flap is the visible end of a
physics-level non-determinism in `shower_clustering_with_nv`.

Bisected with doctest's XML reporter (per-`TestCase` `successes=`), exactly one
case moves: `pattern_recognition shower_clustering_with_nv [B]`, 4 ↔ 5. Its
body spends **one CHECK per entry of `map_vertex_to_shower`** plus two fixed
CHECKs, so 4 assertions means 2 entries and 5 means 3 entries. The map size
itself is what flaps.

Two hypotheses were tested and killed before looking at code:

* **cwd** — the fixtures are named by relative path (`../clus/test/data/...`)
  and resolved through `Persist::resolve` + `WIRECELL_PATH`, so the binary
  finds them from any cwd; and the flap reproduces with cwd fixed.
* **time / load** — no; but **address-space layout is decisive**:
  15/15 runs under `setarch x86_64 -R` give 4, while with ASLR on the value
  moves. So the deciding input is an *address*.

## 2. Bisection: the divergence is inside shower clustering only

A temporary probe in `run_through()` fingerprinted the PR graph after each of
the 12 stages (node/edge counts, per-node graph index and fit point, per-edge
fit count, coordinate sums to 10 digits). Across a run that ends with 3 map
entries and one that ends with 2, **every one of the 12 fingerprints is
identical, node for node** — including the graph indices. The only difference
appears at the end:

| | 3-entry run | 2-entry run |
|---|---|---|
| showers | 3 | 3 |
| shower start segments (graph index) | 15, 17, 20 | 15, 17, 20 |
| shower start **vertices** | 14, 17, 24 (one each) | 4, 10 (1 and 2) |
| connection type | **3** | **2** |

Same graph, same three segments, different attachment vertices and a different
`connection_type`. conn-2 is minted in
`shower_clustering_with_nv_from_vertices` (`NeutrinoShowerClustering.cxx:1806`),
conn-3 in `shower_clustering_in_other_clusters` (`:2290`) — the *second* only
sees clusters the *first* left alone. So the question is why `from_vertices`
sometimes claims these three clusters and sometimes does not.

## 3. D1 — `main_pi` is read before it is written

`shower_clustering_with_nv_from_vertices` (`NeutrinoShowerClustering.cxx:1665`):

```cpp
cluster_point_info min_pi;
min_pi.cluster = cluster;
min_pi.min_angle = 90;          // explicit sentinels
min_pi.min_dis = 1e9;
min_pi.min_vertex = nullptr;

cluster_point_info main_pi;
main_pi.cluster = cluster;
main_pi.min_vertex = main_vertex;   // .min_angle / .min_dis / .min_point: NEVER SET
```

`cluster_point_info` (`:219`) is a plain aggregate with no initialisers.
`main_pi.min_angle` / `.min_dis` are filled **only** inside the vertex loop, and
only on the iteration where `vtx == main_vertex` (`:1731-1735`). That iteration
happens only if `main_vertex` is one of `main_cluster_vertices`, i.e. only if
the overall main vertex belongs to `main_cluster`. **When it does not, nothing
writes them** — and 20 lines later they are compared:

```cpp
if (main_pi.min_angle < min_pi.min_angle + 3 && main_pi.min_dis < min_pi.min_dis * 1.2 &&
    (min_pi.min_angle > 0.9 * main_pi.min_angle || vtx_dis < 1.5 * units::cm)) {
    map_cluster_pi[cluster] = main_pi;     // <-- decided by stale stack bytes
} else {
    map_cluster_pi[cluster] = min_pi;
}
```

A probe printing the raw values on fixture B, 8 runs:

```
SCDBG post main_pi cid=-1 seen=0 angle=4.6403157127207557e-310 dis=0 | min angle=48.95 dis=3433.1 v=4
SCDBG post main_pi cid=-1 seen=0 angle=4.6386440140953146e-310 dis=0 | ...
SCDBG post main_pi cid=-1 seen=0 angle=4.6624938770694718e-310 dis=0 | ...
SCDBG post main_pi cid=-1 seen=0 angle=4.6528771607831927e-310 dis=4.1503747253559418e+150 | ...   <-- the 2-entry run
```

`seen=0` on every cluster: the main vertex is never evaluated here. The garbage
`angle` is **a leftover pointer reinterpreted as a double** — 4.6e-310 is the
bit pattern `0x0000_7Fxx_xxxx_xxxx`, which is exactly why `setarch -R` freezes
the outcome and why ASLR unfreezes it. `dis` is usually 0 (so `main_pi` wins
with a fabricated 0 cm / 0 deg attachment to the main vertex) but sometimes a
huge stale double (so `min_pi` wins). One in ~6 runs flips.

**Valgrind agrees, independently**: five contexts of *"Conditional jump or move
depends on uninitialised value(s) … Uninitialised value was created by a stack
allocation at … shower_clustering_with_nv_from_vertices"*, two of them inside
`sortbydis` (`NeutrinoShowerClustering.cxx:228`) — the sort of `vec_pi` by
`min_dis`, i.e. the garbage propagating into the *ordering* of the whole
cluster list, not just one branch.

**The prototype has the same hole** — `NeutrinoID_shower_clustering.h:1071-1073`
sets only `.cluster` and `.min_vertex`, and `:1126` does the same comparison. So
this is an inherited defect, not a porting slip (M15 checked: nothing in
`porting_dictionary.md` blesses it).

### Blast radius when the read happens to say "prefer main_pi"

`min_dis = 0` and `min_angle ≈ 0` mean the cluster is recorded as touching the
main vertex at zero distance with a fabricated `min_point` — which then
* sorts that cluster to the front of `vec_pi` (`sortbydis` on `min_dis`),
* passes the Step-5 angle cuts unconditionally, and
* looks for a segment within 0.01 cm of a **garbage point**, which normally
  fails, so the cluster is silently dropped from `from_vertices` and picked up
  later as conn-3.

It is also copied into `min_pi` wholesale when no vertex qualified
(`:1746-1751`), so the garbage can reach `map_cluster_associated_vertex` too.

### The fix, and why it is a knob

`shower_nv_main_pi_init` (default **false** = legacy):

```cpp
if (m_shower_nv_main_pi_init) {
    main_pi.min_angle = 1e9;
    main_pi.min_dis = 1e9;
    main_pi.min_point.set(0, 0, 0);
}
```

With the sentinel, the first conjunct is false whenever the main vertex was
never evaluated, so the "no main-vertex information" case deterministically
prefers `min_pi`; when the main vertex *is* evaluated the loop overwrites the
sentinels and behaviour is unchanged by construction. Knob-off compiles the
statement away from the decision entirely.

It is a knob and not a straight fix because the legacy garbage is *usually*
`dis=0`, i.e. legacy usually takes the "prefer main_pi" branch — so turning
this on **changes reconstruction output on every event where the overall main
vertex lives outside `main_cluster`**. That is a physics change and the flip is
the owner's call, not a correctness cleanup.

## 4. D2 — use-after-free in the wire→channel cache

Found by the same valgrind pass, unrelated to D1 and much more likely to be
what a SIGSEGV looks like. `TrackFitting::get_channel_for_wire`
(`TrackFitting.cxx:632-645`, before this round):

```cpp
auto cold_it = m_cold_cache.find(wire_key);
if (cold_it != m_cold_cache.end()) {
    m_access_count[plane_key]++;
    if (m_access_count[plane_key] >= HOT_THRESHOLD) {
        cache_entire_plane(apa, face, plane);   // erases EVERY wire of this plane
    }                                           // from m_cold_cache -> frees cold_it's node
    return cold_it->second;                     // <-- read of freed memory
}
```

`cache_entire_plane` (`:715`) ends with an explicit *"Remove individual wire
entries from cold cache to save memory"* loop that does
`m_cold_cache.erase(wire_key)` for all wires of the plane, including the one
the caller is holding. Valgrind:

```
Invalid read of size 4
   at TrackFitting::get_channel_for_wire (TrackFitting.cxx:642)
   by TrackFitting::form_point_association (TrackFitting.cxx:2339)
   by TrackFitting::form_map (TrackFitting.cxx:3669)
   by TrackFitting::do_single_tracking (TrackFitting.cxx:8820)
 Address 0x59cfefe0 is 48 bytes inside a block of size 56 free'd
   by TrackFitting::cache_entire_plane (TrackFitting.cxx:715)
   by TrackFitting::get_channel_for_wire (TrackFitting.cxx:642)
```

It fires exactly once per (apa, face, plane) — the promotion is a one-shot — so
3 contexts on one small fixture, and it is *benign whenever the allocator has
not yet handed that 56-byte node to someone else*. When it has, this returns a
**garbage channel number** into `form_point_association`, i.e. into the
charge→trajectory association that drives every fit.

Fixed by reading the value before promoting (`const int cached_channel =
cold_it->second;`). **Not knobbed**: the legacy read is undefined behaviour and
the intended value is unambiguous — the same rule this tree already used for
the pr/82 segfault fix ("unknobbed fix OK only if undefined AND gated"). It is
gated by the byte-identical A/B in §7 instead.

## 5. 178410 — reproducible at `-j 1`; it is not a concurrency bug

doc pr/95 §4 recorded `rc=139` on 1 of 2000 mcp2k Q/L events, noted the
3.5× peak-RSS excursion (2403 MB vs 683 MB at the same wall time), observed
that a solo re-run in `work-probe178410a` succeeded, and concluded
*"a concurrency-dependent memory blow-up under 32-way load"*.

Two facts correct that reading.

1. **The inputs were never in question.** `work-probe178410a/evt178410` and
   `work-mcp2k-ql0819/evt178410` are the *same directory* — both are symlinks
   to `work-img-mcp2k/evt178410` (verified by inode). So the crashing run and
   the clean run consumed byte-identical imaging.
2. **It reproduces at `-j 1`, with no load and no ASLR.** Six fresh roots, one
   event each, every run under `setarch x86_64 -R` (as the driver always does),
   differing only in the length of one extra exported variable. *At the time I
   read this table as "the environment size is the trigger"; the repeat test
   further down shows that was wrong — the padding is irrelevant and these are
   samples of a ~5 %-per-run failure. The table is kept as measured.*

| padding bytes | rc | wall s | peak RSS kB |
|---|---|---|---|
| 0 | 0 | 134 | 672 632 |
| **16** | **139** | **129** | **2 310 648** |
| 64 | 0 | 144 | 676 396 |
| 256 | 0 | 133 | 682 252 |
| 1024 | 0 | 134 | 678 612 |
| 4096 | 0 | 134 | 678 608 |

(The original six roots were produced by an ad-hoc script; `pr97_layout_sweep.sh`
in this directory is that script, committed, and it is what produced the
10-padding repeat in the next subsection as `work-pr97b-pad*`.)

The `pad16` run reproduces the pr/95 signature to within noise (2311 MB vs
2403 MB, 129 s vs 132 s, apa0 `mabc` left 0 bytes and apa1 partial, the log
tail full of `Cluster::get_hull number of points is too large: 21608`). So the
32-way batch never mattered — the crash needs neither load nor a second
process. What it *does* need is answered below.

### Neither D1 nor D2 is the cause, and the crash is NOT fixed by this round


Checked rather than assumed. The Q/L job's own compiled config
(`cfg_work_out/sbnd_ql.json`) contains **no `TaggerCheck*` / PR-stage component
at all** — its 34 types are `MultiAlgBlobClustering`, the 18 `Clustering*`
visitors, `QLMatching`, `PointTreeBuilding`, the sources/sinks and geometry.
`TrackFitting` is only *consumed* from the grouping by MABC's Bee dumpers
("No TrackFitting in grouping ..."), never constructed there, and a **complete**
145 KB `-L debug` Q/L log for this event has **0 `clus.TrackFitting` lines**.
So D2 is not executed in the crashing process, and neither is D1
(`shower_clustering_with_nv` lives in `TaggerCheckNeutrino`).

The padding sweep was therefore repeated with the pr/97 binary (D2 fixed, D1
knob off), 10 paddings this time:

| padding bytes | rc | wall s | peak RSS kB |
|---|---|---|---|
| 0, 8, 16, 24, 32, 64, 96, 128, 192 | 0 | 131-134 | ~680 000 |
| **48** | **139** | **127** | **2 309 136** |

**The crash is still live at HEAD + these fixes**, with the signature unchanged
to within noise (2309 MB vs 2311 MB vs pr/95's 2403 MB). One crash in each of
these two sweeps — which, as the next subsection shows, is a per-run rate and
not a property of any particular padding.

### CORRECTION — it is not layout-determined either; it is ~5 % per run

The two sweeps above invited the reading "padding 16 / padding 48 is the fatal
layout". **That reading is wrong, and it was my own overclaim.** The test I had
not run is the obvious one: re-run *the same configuration* and see whether it
crashes every time. The ROOT path is part of the environment, so this needs the
same root string reused, with each run's products archived under a fresh name
first (`work-pr97e-pad14.run{1..5}`; nothing deleted or overwritten):

| repeat | `ulimit -c` | rc | wall s | peak RSS kB |
|---|---|---|---|---|
| 1 | 0 | 0 | 125 | 686 500 |
| 2 | 0 | 0 | 125 | 683 524 |
| **3** | 0 | **139** | 128 | **2 309 684** |
| 4 | unlimited | 0 | 124 | 681 400 |

Identical binary, identical inputs, identical environment down to the byte,
`setarch x86_64 -R`, one job on the box: **1 crash in 4**. So the environment
padding never *caused* anything — the sweeps were sampling a per-run
probability, and reading "1 of 6" as "this padding is special" was a
small-sample artifact.

Tally over every `-j 1` run of this event in this round:

| binary | runs | crashes | rate |
|---|---|---|---|
| pre-fix (`f0e69780`) | 7 | 1 (`pad16`) | 14 % |
| post-fix (`d57c750c`) | 61 | 3 (`pr97b-pad48`, `pr97e-pad14.run1`, `pr97e-pad14.run4`) | 4.9 % |

(Superseded by the 108-run inventory two subsections below, which adds 40 more
post-fix runs and settles the rate at 3.7 %.)

The two rates are the same within Poisson error on these counts, which is
expected: neither D1 nor D2 executes in this job (above).

This also reconciles with pr/95's 1-in-2000: the rate is a property of **this
event**, not of the sample. 178410 is the heavy one (21 608-point hulls against
a 10 000 cap); a few-percent chance on a handful of heavy events gives ~1
failure per 2000-event batch, and pr/95's re-run at `-j 1` succeeding was a 95 %
outcome, not evidence about concurrency.

### What is actually established

* the crash is **intermittent at ~4 % per run for this event** (4 in 108),
  reproducible by
  simply repeating the run (`pr97_layout_sweep.sh`, or any driver, `-j 1`);
* it is **not** concurrency (every run above is one job on an idle-ish box),
  **not** ASLR (M4 — all under `setarch -R`), **not** the environment, **not**
  the inputs, **not** D1 or D2, and **not** `ulimit -c`;
* it lives in the **clustering + Q/L matching** chain;
* the mechanism is *not yet named*, and the usual suspects are ruled out by
  inspection: `clus/src` contains **no** `std::thread` / TBB / OpenMP, **no**
  RNG (`gRandom`/`TRandom`/`mt19937`/`rand`), and **no** time-gated or
  deadline-driven behaviour, and the job runs `Pgrapher` (sequential) with
  "executing with 15 nodes". So a single-threaded, deterministic-input process
  is producing run-to-run divergence — which is exactly the signature of
  reading memory the program did not write, and is why D1's class of defect is
  still the leading hypothesis, just not D1 itself.

A full valgrind memcheck pass over the Q/L job on this event **completed** —
through a fork of the runner (`run_ql_evt_pr97vg.sh`, M10: the production script
is byte-untouched, root `work-pr97-vgql`). It ran the whole event to a good
1.1 MB `mabc-all-apa.zip` (~30× slowdown, 4.5 h) and reported **9 590 008
errors from 3878 contexts, of which ZERO are in WireCell algorithm code** —
every context is the gojsonnet Go runtime (memcheck cannot follow Go's stack
growth), ROOT streamer/`TStorage` noise, or configure-time `SCEFieldTH3`.

That is not an exoneration, for two reasons worth writing down:

* memcheck reports an indeterminate *use* whether or not the branch is taken,
  but only for code that actually executes, and this run did not crash;
* memcheck **cannot see an out-of-bounds access that stays inside a live heap
  block** — indexing a large `std::vector` / point cloud past `size()` but
  inside `capacity()`, or with a garbage index that happens to land in another
  live allocation, is invisible to it. (I argued here that the 3.5× RSS
  excursion made a bad index the shape to suspect. **Retracted** — §5.4 shows
  the excursion is ROOT's crash handler, and the fault is not in WireCell code
  at all.)

### The 40-run core-capture sweep: 0/40, and no core anywhere

`work-pr97f-r{1..40}` (fresh root per run, `ulimit -c unlimited`, 5-way
concurrency, D2-fixed binary) finished **40/40 rc=0**, wall 119-130 s, peak RSS
674-686 MB, `core=` empty on every run. No crash, so no backtrace.

Full inventory of every `-j 1` run of 178410 in this round, read back from the
per-run `.status` files rather than from memory
(`for d in work-pr97*; do cat $d/.status/*; done`):

| runs | crashes | rate | crashing dirs |
|---|---|---|---|
| 108 | 4 | **3.7 %** | `work-pr97-pad16`, `work-pr97b-pad48`, `work-pr97e-pad14.run1`, `work-pr97e-pad14.run4` |

Split by binary: pre-fix 7 runs / 1 crash, post-fix 101 runs / 3 crashes
(3.0 %). Same within Poisson error, as expected — neither D1 nor D2 runs in
this job. The `pr97e-pad14` root crashed **2 of its 5 runs** while `pad16`,
`pad24`, `pad32` and forty `pr97f` roots never crashed once; with a ~4 % rate
that clustering is unremarkable, and it is the last reason anyone might still
read the padding table as causal.

Why no core exists even under `ulimit -c unlimited`: `/proc/sys/kernel/core_pattern`
is the bare `core`, so the kernel writes into the *dying process's* CWD, and a
2.3 GB core there is easy to lose or to have silently fail. **For pr/98, do not
chase cores** — run the event under gdb directly, which needs no kernel or
`ulimit` cooperation:

```
until gdb -q -batch -ex run -ex bt -ex 'info locals' \
        --args wire-cell -c <precompiled>.json 2>&1 | tee /home/xqian/tmp/gdb-178410.log \
      | grep -q SIGSEGV; do :; done
```

At ~130 s/run and ~4 %, expect a hit inside ~25 runs (~1 h serial, less at
5-way). Precompile the config with `wcsonnet` first (M17).

**Naming the line was handed to pr/98 — and then answered in this round; the
rest of this section supersedes the paragraph below.**

*(Original plan, kept for the record:)* Reproduce by simply
repeating the run. The 40-run `ulimit -c unlimited` sweep produced no crash and
no core, and `core_pattern` makes core capture unreliable here — use the gdb
loop given above instead. ASan/UBSan
builds, or a `-D_GLIBCXX_ASSERTIONS` / `_GLIBCXX_DEBUG` build, would catch the
in-block overrun that memcheck cannot.

Note on the numbers: 139 is SIGSEGV, not the OOM killer (137), and a failed
allocation would throw `bad_alloc`. I read the 3.5× RSS as "a symptom of the
wrong branch being taken"; **that is retracted in §5.4** — the extra 1.7 GB is
the `gdb` that ROOT's `TUnixSystem::StackTrace()` spawns to print the
backtrace, and `timecmd.py` reports `getrusage(RUSAGE_CHILDREN).ru_maxrss`,
i.e. the peak over **all descendants**. Only crashing runs spawn it, which is
exactly why only they show 2.3-2.4 GB.


### 5.4 The gdb loop: the fault is not in WireCell code at all

Repro:

```
cd sbnd/sbnd_xin
./pr97_gdb_loop.sh work-pr97g-r 48 6      # 48 runs, 6 at a time, fresh root each (M13)
```

`pr97_gdb_loop.sh` inlines the `run_ql_batch.sh` worker for this one event and
calls `run_ql_evt_pr97gdb.sh` — `run_ql_evt.sh` with the wire-cell step wrapped
in `gdb -q -batch ... -ex run -ex "thread apply all bt" -ex "bt full"`, every
signal the Go runtime uses passed through silently (M10: the production runner
is byte-untouched).  Note the detector: under gdb a caught SIGSEGV leaves the
runner with **rc=0**, because gdb reaps the inferior — a crash in this arm is
found by grepping `received signal SIG`, never by `rc=139`.

**Two captures in 48 runs (4.2 %, matching the 4-in-108 baseline):**

| run | signal delivered to | its PC | thread 1 (the clustering) at that instant |
|---|---|---|---|
| `work-pr97g-r7` | Thread 34, LWP 466906 | **0x0**, empty stack | healthy — `__dynamic_cast` from `hough_transform` (`Facade_Cluster.cxx:1729`) |
| `work-pr97g-r32` | Thread 9, LWP 485373 | **0x0**, empty stack | healthy — `std::vector<double>::operator[](15451)` in `hough_transform` (`:1727`), a valid index into the 64 800-bin grid |

The faulting thread has jumped to address 0 and has no stack, so **nothing in
the dump names it directly**.  What identifies it is a thread census of a live
process (`ls /proc/<pid>/task | wc -l`):

| config handed to `wire-cell` | threads |
|---|---|
| `wct-clus-matching-perevt.jsonnet` (in-process gojsonnet) | **65** |
| the same config precompiled by `wcsonnet` to JSON | **1** |

Every non-main thread in this process comes from the Go runtime inside
`libgojsonnet.so`, and it stays alive for the whole job — in r7's dump, of 34
threads **32 were parked in `runtime.futex`** and one in `runtime.usleep` (Go's
`sysmon`).  The faulting thread is a non-main thread whose LWP falls inside the
startup-created range (r7: 466906, top of 466703-466906), so it is a Go runtime
thread.  The WireCell main thread is mid-computation with sane arguments in both
captures.

**This retracts the four ROOT backtraces quoted earlier in this section as
fault sites.** ROOT's
`TUnixSystem::StackTrace()` prints **thread 1's** stack, not the faulting
thread's, so those traces were a *time sample* of whatever the main thread
happened to be doing.  That is why their leaf frames were pure arithmetic
(`__ieee754_acos`, `boost::histogram::axis::regular::index`) on perfectly valid
arguments — a shape that never made sense as a segfault, and which I should have
treated as a contradiction rather than a clue.  `connect_graph_relaxed` appears
in all six traces simply because that is where this event spends its time.

It also explains the "3.5× RSS excursion" that this section and doc pr/95 §4 both read
as a memory blow-up: `StackTrace()` **spawns gdb**, and `timecmd.py` reports
`getrusage(RUSAGE_CHILDREN).ru_maxrss`, the peak over all descendants.  Measured
here, `gdb -batch -ex "thread apply all bt"` on this 55-thread process costs
**6.8 s and ~1.8 GB**.  Only crashing runs spawn it.  There is no memory blow-up.

### 5.5 The trigger is a ~120 s process-life timer

Peak RSS was never the tell; **wall time is**.  Across five crashes from
different sweeps, days and load levels:

| | wall s |
|---|---|
| the five crashing runs | **126, 127, 128, 128, 129** |
| clean runs, same arms | 119 … 142 |

A 3-second spread on the crashes against a 23-second spread on healthy runs is a
fixed *process-life* deadline, not an algorithm phase — and subtracting the
measured ~7 s stack-trace gdb puts every crash at **≈120 s of process life**.
Go's `sysmon` forces a garbage collection at exactly `forcegcperiod = 2 min`.

The internal consistency check: the `work-pr97f-r*` arm (40 runs, `ulimit -c
unlimited`) had the *shortest* walls of any arm, 119-130 s, and crashed **0/40** —
those runs barely reach the deadline before exiting.  The gdb arm, walls
135-142 s, crashed 2/48.

**The arms.** All against the contemporaneous control of 2/48 in the gdb arm,
each run in a fresh root (M13), same event, same binary:

| arm | root | runs | crashes | rate |
|---|---|---|---|---|
| control, `.jsonnet` config, under gdb | `work-pr97g-r*` | 48 | **2** | 4.2 % |
| control, `.jsonnet` + `GOTRACEBACK=crash` | `work-pr97k-r*` | 32 | **1** | 3.1 % |
| **config precompiled to JSON by `wcsonnet`** | `work-pr97h-r*` | **100** | **0** | — |
| `.jsonnet` + `GODEBUG=asyncpreemptoff=1` | `work-pr97i-r*` | 60 | 0 | — |
| `.jsonnet` + `GOGC=off` | `work-pr97j-r*` | 55+ | 0 | — |

Read these with the power in mind. Against a 4 % null, a clean 100-run arm is
p≈1.7 % — **the precompiled-JSON arm is the load-bearing evidence**, and it
says: remove the Go runtime from the process and the crash goes away. A clean
60-run arm is only p≈9 %, i.e. exactly as weak as "one padding out of ten
crashed" was, so `asyncpreemptoff` and `GOGC=off` **suggest and do not
establish** which Go subsystem is at fault; a null there refutes one mechanism
inside the runtime, not the runtime.

`GOTRACEBACK=crash` was meant to make the Go runtime print its own traceback,
which is the Go-side stack the gdb captures cannot give. It produced none:
**ROOT's SIGSEGV handler wins over Go's**, prints its own trace and exits. That
is worth knowing on its own — Go's protections for its own threads are not in
force in a process that has initialised ROOT.

### 5.6 What is established, and what is not

Established:

* the crash is **not in WireCell code**; the WireCell main thread is healthy at
  the moment of the fault, in both live captures;
* the faulting thread is a Go runtime thread from `libgojsonnet.so`, jumping to
  PC = 0x0;
* the Go runtime is present for the whole job — 64 extra threads, 130 s after the
  config finished compiling — purely because the config is handed to `wire-cell`
  as `.jsonnet` rather than precompiled JSON;
* the trigger is a ~120 s process-life deadline, matching Go's forced-GC period;
* the "memory blow-up" and the four ROOT backtraces are both artifacts of ROOT's
  crash handler (§5.4).

Not established:

* the exact Go-runtime path that lands at PC 0 — that needs a Go-side stack, and
  the Go thread has none by the time gdb stops it;
* whether anything WireCell does *provokes* it (a signal-handler interaction with
  ROOT's own handlers is the obvious suspect, since ROOT installs handlers over
  Go's and Go's preemption/GC signals then land in them);
* whether other detectors/jobs are exposed — every `wire-cell -c *.jsonnet` job
  that runs longer than ~120 s carries the same Go runtime, so in principle yes,
  and 178410 is not special beyond being long enough.

**No toolkit change is proposed from this round** — the finding is about how
jobs are launched, not about clustering. The toolkit-side fix is written up for
the maintainer in `97_upstream_brief.md`. What *was* deployed here, on the
owner's instruction, is §5.7.

### 5.7 Deployed: precompiled configs, and batches that fail loudly

Two changes in `wcp-porting-img`, both aimed at the immediate exposure:

**(a) Compile the config in a separate `wcsonnet` process** — `run_ql_evt.sh`
and `run_pr_chain_batch.sh` now collect their TLAs into one array, run
`wcsonnet` with exactly those arguments, and hand `wire-cell` the resulting
JSON. `SBND_PRECOMPILE_CFG=0` restores the legacy in-process path for A/B
archaeology. The Q/L job's process drops from **65 threads to 1**; the compile
costs **0.13 s** of the ~135 s job. The PR chain matters more than the Q/L job
here: it routinely runs far past the 120 s deadline.

**(b) Fail loudly on a failed event** — `run_ql_batch.sh` and
`run_pr_chain_batch.sh` now print a boxed banner naming every event whose output
is missing or truncated, and **exit non-zero**. `batch_summary()` in
`_runlib.sh` returns 0 as long as *any* event succeeded, which is how doc pr/95
lost evt 178410 out of 2000 and still read as a clean batch. `_runlib.sh` itself
is untouched (it is shared by many runners); the banner lives in the two
drivers. The banner names `rc=139` explicitly and points at this doc.

Gates for (a), knob ON vs the pre-change baseline arm `work-pr97gate-nuecc48`
(same binary `d57c750c`, same manifest, 48 nueCC48 events):

| gate | label | result |
|---|---|---|
| PR-chain member-content hashes (`scripts/pr85_hash_gate.py`) | `work-pr97L-prgate` | **PASS — 96/96 archives byte-identical** (`mabc-pr.zip` + `pctree`) |
| per-branch/per-entry ROOT compare (`scripts/pr94_root_gate.py`) | same | **PASS — 48 identical, 0 differing** |
| `nusel-table.tsv` / `nusel-events.tsv` | same | **identical** |
| Q/L job, member-content hashes | `work-pr97L-prod1` vs `work-pr97g-r1` | **identical** |
| crash rate with (a) on | `work-pr97h-r*` | **0 in 100** vs 2/48 control |

So (a) is byte-identical on the production manifest and removes the failure
mode. (b) changes only diagnostics and exit codes.

## 6. What changed

| file | change |
|---|---|
| `clus/src/TrackFitting.cxx` | D2: read the cached channel before the hot-cache promotion frees it. Unknobbed. |
| `clus/inc/WireCellClus/NeutrinoPatternBase.h` | `m_shower_nv_main_pi_init{false}` + rationale block |
| `clus/src/NeutrinoShowerClustering.cxx` | D1: gated sentinel init of `main_pi` |
| `clus/inc/WireCellClus/TaggerCheckNeutrino.h`, `clus/src/TaggerCheckNeutrino.cxx` | knob member, `configure()`, `default_configuration()`, hand-off to `pattern_algos` |
| `cfg/pgrapher/common/clus.jsonnet` | `tagger_check_neutrino(... shower_nv_main_pi_init=false)` + key-suppression |
| `cfg/pgrapher/experiment/sbnd/clus.jsonnet` | 2 default blocks + 2 pass-throughs, all `false` |
| `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` | TLA default `false` + pass-through |
| `clus/test/doctest_pattern_recognition.cxx` | fixed-count assertions in the flapping case; **new** `shower_clustering_with_nv main_pi determinism [B]` |
| `sbnd_xin/run_pr_chain_batch.sh` | `SBND_MAIN_PI_INIT` hook |
| `sbnd_xin/run_ql_evt_pr97vg.sh` | **new** valgrind fork of the Q/L runner (diagnostic only) |

## 7. Gates

| gate | arms / command | result |
|---|---|---|
| member-content hash, PR chain | `work-nuecc48-prod0819` vs `work-pr97gate-nuecc48` | **PASS 96/96 byte-identical** |
| member-content hash, PR chain | `work-ncpi0-prod0819` vs `work-pr97gate-ncpi0` | **PASS 38/38 byte-identical** |
| arm completion | both arms | 48/48 and 19/19 `rc=0` |
| compiled config, ALL 16 live jobs | `compile_all_cfg.sh` HEAD-cfg vs working-tree cfg, `cmp_cfg.sh` | **PASS**, normdiff 0 on every job (sbnd_pr/ql/clus/img, pdhd ×3, pdvd ×3, 5 sim jobs) |
| compiled config, knob on | `wcsonnet … -S shower_nv_main_pi_init=true` | key present (1 hit); absent when off (0 hits) |
| `wcdoctest-clus` | — | **212 cases / 2218 assertions, 0 failed**, 1 skipped, and the total is now STABLE run to run (12/12 identical; was 2214 ↔ 2215). The skip is a pre-existing deliberate `doctest::skip()` in `doctest_clustering_prototype.cxx:214`, unrelated. |
| member-content hash + tagger-compare, **uBooNE** PR chain | `qlport/scripts/ab_check.sh pr97_after pr97_before` (35-event `filelist`, the two labels built from the same tree with and without this round's 9 files) | **PASS**: 35/35 Bee zips content-identical, 35/35 `wire-cell-uboone-tagger-compare` logs identical, 0 non-rc0 either side |
| new determinism test | 8 runs ASLR on + 4 runs `setarch -R` | 12/12 pass, 4 assertions every time |
| freshness (M1) | `build/clus/libWireCellClus.so` 23:59 vs sources 23:56/23:57 | newer |

**Gate scope.** D2 is in `TrackFitting`, which instantiates only under the
PR-stage components (`TaggerCheck*`): the SBND `clus` and `ql` jobs' compiled
configs contain none of them and their logs carry no `clus.TrackFitting` line
(§5), so `abtest/events.txt` (img/clus) is not an affected manifest. The
affected chains are the **SBND PR chain** (nueCC48 + NC pi0 above) and the
**uBooNE PR chain** (`ab_check.sh`, above) — both gated. D1 is reached only
through `TaggerCheckNeutrino`, a subset of the same set.

The byte-identical PASSes are the interesting result for D2: the use-after-free
was benign on all 67 SBND + 35 uBooNE gated events — the freed node still held
the right bytes — which is exactly why this survived unnoticed.

## 8. Residuals and things deliberately not done

* **178410 is no longer a WireCell-code question** (§5.4-5.6): the fault is on a
  `libgojsonnet` Go-runtime thread at PC 0x0 while WireCell's main thread is
  healthy, triggered at ~120 s of process life. What is still open is the
  Go-side path, and whether ROOT's signal handlers (installed over Go's) are
  what turns Go's forced-GC/preemption signal into a jump to 0. That needs a
  Go-side stack — e.g. a build with `runtime-gdb.py` auto-load allowed
  (`add-auto-load-safe-path`), or `GOTRACEBACK=crash` so the Go runtime prints
  its own traceback before dying.
* **A workaround exists and is the owner's call, not shipped here**: precompile
  the job config with `wcsonnet` and hand `wire-cell` the JSON. That drops the
  process from 65 threads to 1 and is output-neutral (member-content hashes
  identical). It is a runner change in `wcp-porting-img`, and it would apply to
  every long-running `wire-cell -c *.jsonnet` job, not just this event.
* **`Cluster::get_hull(int max_points)` caches cap-insensitively**
  (`Facade_Cluster.cxx:2349`): the first caller's hull is returned to every
  later caller regardless of that caller's `max_points`. Reported, not touched
  (§5 tie-breaker: unrelated bug, different change).
* **The knob is OFF.** Turning `shower_nv_main_pi_init` on is a physics change
  (§3) and needs an owner verdict. First ON arm measured on nueCC48
  (`work-pr97on-nuecc48b` vs `work-pr97gate-nuecc48`, both 48/48 `rc=0`):

  | gate | result |
  |---|---|
  | member-content hash | **1 of 48 archives differ** — evt **52672** `mabc-pr.zip` |
  | per-branch/per-entry ROOT | 47 identical, **1 differing**: evt 52672, branches `T_kine.kine_energy_{included,info,particle}`, `kine_particle_type`, `kine_pio_angle`, `kine_pio_dis_1` |
  | `nusel-events.tsv` / `nusel-table.tsv` | **identical** — no `numu_score` / `nue_score` / vertex movement on this sample |

  So on nueCC48 the flip is a 1-event, kine-only move (a pi0 pairing and the
  particle-type/energy bookkeeping that follows from it) and touches no
  selection score. That is a *footprint*, not a verdict: the ON arm has not
  been hand-scanned and mcp1k/mcp2k have not been run. **Ships OFF.**
* **The pr/95 P3 determinism floor is still owed.** Unchanged by this round;
  the cross-layout ASLR-on leg recorded in `PROTECTED.txt` since 2026-08-05 is
  still not measured. What this round adds is the reason it matters: the tree
  has at least three address-dependent reads, so a layout change is a physics
  change today.
* The 2214/2215 line in doc pr/94 §9.11 should be read with §1 above: the
  delta is not a per-commit constant.
