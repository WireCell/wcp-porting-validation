# 60 — evt 278794 aborts in TrackFitting: the one-point fit path

The single failure of the doc-59 full-1000-event production (999/1000 ok).
Entry 618 / event **278794** aborts the whole `wire-cell` PR job with

```
terminate called after throwing an instance of 'std::runtime_error'
  what():  TrackFitting::do_single_tracking: inconsistent vector sizes for fit output!
```

`clus/src/TrackFitting.cxx:8566`, reached from `TaggerCheckSTM`.

**Status: FIXED.** Both guards of §5 and §4 are applied in
`clus/src/TrackFitting.cxx`; evt 278794 now completes with a full 8-bundle table
and an STM verdict on its in-beam bundle. §6 has the gates. The knob-off
question does not arise: there is no knob, because the change is byte-identical
on every event that completed before (§6 gate 1/2) and the only event it
changes is the one that used to abort.

**§0 — the chain is deterministic** (established first, before any A/B, on the
*unmodified* binary). Doc 49 §4a's "±7 STM tags" ASLR noise floor **does not
reproduce**: see §7. Every gate below is therefore a real comparison, not noise.

**One sentence:** a `search_other_tracks` candidate — a 4.7 cm, 5-point steiner
stub — is whittled down to a **single** fit point by the second
`trajectory_fit`; `dQ_dx_fit` then bails out at its `size() <= 1` guard leaving
`dQ/dx/reduced_chi2` empty while `pu/pv/pw/pt/paf` each hold one entry, and a
consistency check that **exists only in the toolkit** turns that into a fatal
throw. The prototype tolerates exactly this case by design (`if (dQ.size() > 1)`
— `PR3DCluster_pattern_recognition.h:262`).

---

## Repro

One second, from the doc-59 Q/L products (which survived the abort — only the
PR step dies). Runs in a **fresh** root, `work-mcp1kall-d59k` is untouched:

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

mkdir -p work-mcp1kall-d60crash/.status work-mcp1kall-d60crash/.cwd
# symlink the pctree TARBALL into a real local dir, never the ql_evt<ID> dir
# itself: run_nusel_evt.sh re-runs the Q/L step whenever that tarball is
# absent, and it writes into $SBND_WORK_ROOT/ql_evt<ID>/ -- through a directory
# symlink that lands inside the doc-59 record (M13).
mkdir -p work-mcp1kall-d60crash/ql_evt278794
ln -sfn "$PWD/work-mcp1kall-d59k/ql_evt278794/pctree-evt278794.tar.gz" \
        work-mcp1kall-d60crash/ql_evt278794/pctree-evt278794.tar.gz
TAG=d60crash ENTRIES="618" ./run_full1k_nusel.sh 0 1
# => rc=1 entry=618 evt=278794 ... wall_s=1
grep -a "inconsistent vector sizes" work-mcp1kall-d60crash/.log_e618.log
```

Reproduced 8/8 before the fix — 2 driver runs plus 6 under gdb. Both the driver
and gdb pin the address layout, so this was initially recorded as
"deterministic *under `setarch x86_64 -R`*, un-pinned behavior unknown". §7
closed that gap afterwards: the chain is deterministic either way, so the
1-in-1000 rate stands without an ASLR caveat.

### Debugger recipe used here

`wire-cell` is invoked deep inside `run_nusel_evt.sh`, so rather than
reconstructing its 75-argument command line, put a shim first on `PATH`:

```bash
S=<scratch>; mkdir -p $S/bin
cat > $S/bin/wire-cell <<EOF
#!/bin/bash
exec gdb -q -batch -x $S/gdb.cmds --args /nfs/data/1/xqian/toolkit-dev/toolkit/build/apps/wire-cell "\$@"
EOF
chmod +x $S/bin/wire-cell
PATH=$S/bin:$PATH TAG=d60crash ENTRIES="618" ./run_full1k_nusel.sh 0 1
```

gdb output lands in `work-mcp1kall-d60crash/.log_e618.log`. Three gotchas cost
real time and are worth writing down:

* **`set breakpoint pending on` is mandatory.** `libWireCellClus.so` is
  `dlopen`ed after `run`, so every breakpoint is pending at script time.
* **`break <mangled/demangled function name>` never resolves** for these
  symbols even though `nm -C` shows them; **`break TrackFitting.cxx:<line>`
  does**. Use file:line only.
* **`catch throw` alone is useless** — it fires first on a benign
  `spdlog::throw_spdlog_ex` during static init of `AnodeDumper.cxx`. Arm it
  only after a first breakpoint has been hit.
* At `-O2` the locals of `do_single_tracking` are optimized out and
  `p v.size()` fails with *"Cannot evaluate function -- may be inlined"*. Print
  the members by pointer arithmetic instead:
  `p this->dQ._M_impl._M_finish - this->dQ._M_impl._M_start`.

---

## 1. Symptom

The size check at `TrackFitting.cxx:8560-8567`:

```c++
size_t npoints = fine_tracking_path.size();
if (dQ.size() != npoints || dx.size() != npoints ||
    pu.size() != npoints || pv.size() != npoints ||
    pw.size() != npoints || pt.size() != npoints ||
    reduced_chi2.size() != npoints) {
    throw std::runtime_error("TrackFitting::do_single_tracking: inconsistent vector sizes for fit output!");
}
```

Measured at the throw (gdb, member pointer arithmetic):

| vector | size |
|---|---|
| `fine_tracking_path` (= `npoints`) | **1** |
| `dQ`, `dx`, `reduced_chi2` | **0** |
| `pu`, `pv`, `pw`, `pt`, `paf` | **1** |

So the mismatch is exactly `dQ/dx/reduced_chi2` empty against `npoints == 1`.

Call chain at the abort:

```
MultiAlgBlobClustering::operator()            MultiAlgBlobClustering.cxx:2344
 TaggerCheckSTM::visit                        TaggerCheckSTM.cxx:274
  TaggerCheckSTM::check_stm_conditions        TaggerCheckSTM.cxx:2596
   run_pass lambda                            TaggerCheckSTM.cxx:2580
    TaggerCheckSTM::search_other_tracks       TaggerCheckSTM.cxx:2117
     TrackFitting::do_single_tracking         TrackFitting.cxx:8566   <-- throw
```

`flag_dQ_dx_fit_reg=true, flag_dQ_dx_fit=true, cluster_filter=0x0` — the
`search_other_tracks` call site, not the main STM fit.

Blast radius: the exception is not caught anywhere, so `std::terminate` kills
the process. `nusel_evt278794/` is left with a 0-byte `mabc-pr.zip` and a 0-byte
`pctree-pr-evt278794.tar.gz`; `nusel_extract.py` then dies on `BadZipFile`, so
the event has **no label-table row at all**. That is why 278794 is absent from
the doc-59 census, the 639-event scan set and every Bee set.

## 2. Root cause

### 2a. What the segment is

`search_other_tracks` scans the steiner graph for charge not explained by the
already-fitted segments, and for each surviving component builds a candidate
segment from `shortest_path(special_A, special_B)`. The offending one
(`comp_idx=6`, `special_A=280`, `special_B=299`) is 5 steiner points:

```
p0 = (161.68, -45.96, 60.53) cm      p0->p1  0.600 cm
p1 = (161.68, -45.44, 60.83) cm      p1->p2  0.467 cm
p2 = (161.99, -45.27, 60.53) cm      p2->p3  3.038 cm
p3 = (159.80, -43.36, 61.43) cm      p3->p4  0.625 cm
p4 = (159.18, -43.36, 61.43) cm
                                     path 4.730 cm, chord 3.717 cm
```

A ~5 cm kinked stub. It clears `search_other_tracks`' quality cuts
(`max_dis_u/v/w = 14.5 / 44.0 / 34.3 cm`, `quality_check = true`) because those
cuts measure *2D separation from the existing fits*, not the candidate's own
length. Nothing upstream requires a candidate to be long enough to survive the
fitter.

### 2b. How 5 points become 1

Traced with breakpoints at `TrackFitting.cxx:1991` (`organize_ps_path` entry),
`:2071` (its tail) and `:4013` (`trajectory_fit`):

```
==== do_single_tracking ENTRY (fit_reg=1 fit=1)
    trajfit  pss=5  method=1          1st pass, 5 points in
    org_ps   ENTRY pts=3 eplim=3.0    <- 1st trajectory_fit dropped 2 points
    org_ps   tail  pts=1 psvec=2      <- rescued to 2 by the tail fallback
    trajfit  pss=1  method=2          <- 2nd trajectory_fit dropped one more
    org_ps   ENTRY pts=1 eplim=0.0
    org_ps   tail  pts=1 psvec=1      <- 1 point in, 1 point out
```

`trajectory_fit` shrinks its `pss_vec` in place (`skip_trajectory_point`,
`TrackFitting.cxx:4807`); on a 5 cm stub with a kink it discards almost
everything. `organize_ps_path` itself cannot collapse a path — it ends with
`if (pts.size() <= 1) pts = ps_vec;` (`:2070`) — but it cannot *rebuild* one
either, so a 1-point input gives a 1-point output.

`do_single_tracking` guards `ptss.size() <= 1` **after the first pass only**
(`:8348`). After the second pass it re-seeds the endpoints when
`pts.size() == 2` (`:8413-8424`) — but there is **no `pts.size() <= 1` guard**
before the 2D-projection loop at `:8496`. So one point flows through:

Note the projection loop is *inside* the `if (flag_2nd_tracking)` block opened
at `:8350` (the flag is a hard-coded `true`, so it always runs). That is why the
guard in §5 belongs inside the block and not after it: with the flag false,
`pu..paf` would stay empty while `ptss` still holds the first-pass path, which
trips the same throw from the other direction.

`pu/pv/pw/pt/paf` get one entry each, `fine_tracking_path` gets one pair, and

```c++
void TrackFitting::dQ_dx_fit(double, bool) {
    if (fine_tracking_path.size() <= 1) return;   // :6858 — before its own clear()
```

returns having written nothing. `dQ_dx_fill` has the identical guard at `:5606`,
so the `flag_dQ_dx == false` branch fails the same way.

### 2c. Why this is a port artifact, not a physics bug

Both `<= 1` early returns are faithful ports — the prototype has them verbatim:

```
prototype_base/pid/src/PR3DCluster_dQ_dx_fit.h:267   dQ_dx_fill   if (fine_tracking_path.size()<=1) return;
prototype_base/pid/src/PR3DCluster_dQ_dx_fit.h:369   dQ_dx_fit    if (fine_tracking_path.size()<=1) return;
```

and the prototype's `do_tracking` (`pid/src/PR3DCluster.cxx:33-265`) has the
same missing guard after its second pass. The difference is what happens next.
The prototype's `do_tracking` **ends** at `dQ_dx_fit`; it never assembles a
per-point structure, and its caller checks the empty-`dQ` case explicitly:

```c++
    do_tracking(ct_point_cloud, global_wc_map, flash_time);
    if (dQ.size() >1){                                     //  <-- tolerates dQ empty
      WCP::TrackInfo *track = new WCP::TrackInfo(fine_tracking_path, dQ, dx, pu, pv, pw, pt, reduced_chi2);
      fit_tracks.push_back(track);
    }
```
`prototype_base/pid/src/PR3DCluster_pattern_recognition.h:261-266`

The toolkit port turned that array-of-parallel-vectors into `std::vector<PR::Fit>`,
which *requires* the vectors to be parallel, and added the consistency check to
enforce it (`4dd81f9c`, 2025-08-14, "update code"). The toolkit caller keeps the
prototype's tolerance —
`if (new_segment->fits().size() > 1)` at `TaggerCheckSTM.cxx:2121` — but the
throw fires two frames earlier and it never gets to run.

`do_multi_tracking` has no equivalent check, so the failure mode is confined to
`do_single_tracking`.

## 3. Why it hid

* It needs `npoints == 1` **exactly**. `npoints == 0` passes the check
  (0 == 0 for every vector) and `npoints >= 2` is the normal path, so the
  window is one integer wide.
* Only `search_other_tracks` feeds `do_single_tracking` segments this short.
  The main STM fit works on a cluster-scale path; `search_other_tracks`
  candidates are leftovers, and their quality cuts constrain 2D *separation*,
  never length.
* It needs the fitter to discard 4 of 5 points, i.e. a stub that is both very
  short and kinked.
* Frequency: **1 event in 1000** on MCP2025C data (doc 59). Every earlier SBND
  campaign ran on 10- or 30-event manifests, so the whole apply-pointcloud
  effort to date had roughly a 3% chance of ever seeing it.

## 4. Secondary finding: an OOB read in organize_ps_path (FIXED)

`organize_ps_path` reads **out of bounds** when it is called with a single
point and `end_point_limit == 0` — which is exactly the final call at
`TrackFitting.cxx:8408`:

```c++
    } else {                                        // :2063, end_point_limit == 0
        WireCell::Point p1 = ps_vec.back();
        double dis1 = sqrt(pow(p1.x() - pts.back().x(), 2) + ...);   // pts is EMPTY here
        if (dis1 >= 0.45*units::cm) pts.push_back(p1);
    }
```

With `ps_vec.size() == 1` the beginning block pushes nothing (`dis1 == 0`), and
the middle loop takes its `continue` branch (`dis == 0 < low_dis_limit*0.8`), so
`pts` is empty at `:2065`. `pts.back()` on an empty vector is UB —
`*(_M_finish - 1)` reads the word before the allocation. The trace above shows
`pts == 1` at `:2071`, i.e. the `push_back` **did** execute, so the garbage
comparison happened to exceed 0.45 cm on this run.

It is benign *here* only because the very next line
(`if (pts.size() <= 1) pts = ps_vec;`) overwrites the result. It was still a real
OOB read on a live code path.

**Fix applied** — guard the branch instead of the deref:

```diff
-    } else {
+    } else if (!pts.empty()) {
```

**Provably output-identical**, which is why it ships without a knob and needs no
separate gate: the branch is entered only when `pts` is non-empty (unchanged
code path), and when `pts` *is* empty the old code could push at most one point
onto an empty vector, leaving `pts.size() == 1`, which the very next statement
(`if (pts.size() <= 1) pts = ps_vec;`) overwrites with `ps_vec` regardless. Both
versions therefore return `ps_vec`. The only thing removed is the UB.

Still open, same family, **not fixed here**: `examine_end_ps_vec` does
`ps_list.front()` on an empty list at `:1897` if it is ever called with an empty
`pts`. §5 argues that is unreachable today; it is a latent hole, not a live bug,
and belongs in its own change.

## 5. The fix (APPLIED)

Restore the prototype's tolerance by giving the second pass the guard the first
pass already has. In `do_single_tracking`, immediately before the 2D-projection
block at `TrackFitting.cxx:8496`:

```diff
             pts.push_back(p2);
             }
         }
 
+        // A degenerate candidate can be whittled down to a single point by the
+        // 2nd trajectory_fit (doc 60; SBND evt 278794, a 4.7 cm 5-point steiner
+        // stub from TaggerCheckSTM::search_other_tracks went 5 -> 3 -> 1).
+        // dQ_dx_fit and dQ_dx_fill then both bail out at their
+        // fine_tracking_path.size() <= 1 guard -- prototype
+        // (PR3DCluster_dQ_dx_fit.h lines 267, 369) does the same -- leaving
+        // dQ/dx/reduced_chi2 empty while pu..paf below would get one entry
+        // each, which trips the consistency check at the end of this function.
+        // The prototype carries the empty dQ and lets its caller drop the track
+        // (`if (dQ.size() > 1)`, PR3DCluster_pattern_recognition.h line 262);
+        // our caller already tests fits().size() > 1 (TaggerCheckSTM.cxx), so
+        // return with every output vector still cleared and let it filter.
+        // Mirrors the ptss.size() <= 1 guard after the 1st pass above.  Must
+        // sit inside the flag_2nd_tracking block: the projection loop is here,
+        // not after it.
+        if (pts.size() <= 1) return;
+
         // Generate 2D projections
         pu.clear();
```

Returning leaves every output vector in the cleared state set at `:8264-8272`,
so `segment->set_fits()` is never called, `fits()` stays empty, and
`search_other_tracks` drops the candidate — the prototype's outcome exactly.

**Why this needs no knob.** The guard can only fire on inputs that *previously
aborted the process*, so there is no legacy behavior to preserve — confirmed
empirically by the §6 gates (80 events, 206 STM tags, byte-identical). That claim
rests on `pts.size() == 0` being unreachable at `:8496`, because `npoints == 0`
does **not** throw today — all eight vectors are empty, the check passes, and
the function runs its tail (`segment->fits({})`,
`create_segment_point_cloud`/`create_segment_fit_point_cloud` with an empty
path, `m_cluster_filter = nullptr`). A `<= 1` guard would skip that tail, which
would be a real behavior change on a completing path.

It is unreachable, on this chain of facts:

* `:8348` returns unless `ptss.size() >= 2`;
* `organize_ps_path` cannot shrink below its `ps_vec`, and `ps_vec` falls back
  to the input when `examine_end_ps_vec` returns `<= 1` (`:1992`, `:2070`), so
  the first second-pass call returns `>= 2`;
* the only remaining way to reach `:8496` with zero points is
  `trajectory_fit(…, 2, …)` emptying `ptss` outright — and then the *next*
  statement, `organize_ps_path(segment, pts, low_dis_limit, 0)` at `:8408`,
  calls `examine_end_ps_vec`, which does `ps_list.front()` on an empty list at
  `:1897`. That is UB before `:8496` is ever reached.

So `<= 1` and `== 1` are equivalent in practice. If the owner wants the
no-behavior-change claim to hold on the letter rather than on that reachability
argument, use `== 1`. Either way the empty-`pts` UB at `:1897` is a separate
pre-existing hole, in the same family as §4.

Note also that returning here skips `m_cluster_filter = nullptr` at the end of
the function — as do all four existing early returns (`:8297`, `:8302`,
`:8310`, `:8348`), so this introduces no new asymmetry.

Alternative considered and rejected: make `dQ_dx_fit`/`dQ_dx_fill` `resize()` to
`npoints` instead of returning early. That diverges from the prototype, and it
manufactures a bogus one-point `PR::Fit` with `dQ = dx = 0` that the caller then
has to filter anyway.

## 6. Verification (run)

Build: `wcbuild` → `local/lib/libWireCellClus.so` mtime **11:50** vs
`clus/src/TrackFitting.cxx` **11:17** — freshness proof passes (M1).
Unit tests: `./build/clus/wcdoctest-clus` → **49/49 cases, 565/565 assertions,
rc=0**.

**1. The crash is gone.** Entry 618 / evt 278794, same production flag set:

```
rc=0                                      # was rc=1 / SIGABRT
grep -c "inconsistent vector sizes"  -> 0
grep -cE "terminate called|Aborted"  -> 0
```

and the event now produces real output rather than 0-byte stubs
(`mabc-pr.zip` 178 kB, `pctree-pr-evt278794.tar.gz` 1.65 MB,
`nusel-evt278794.tsv` with **8 bundle rows**). The in-beam bundle — main 10,
flash t = 0.695 µs, 1907 pts, 203.1 cm, the one whose `search_other_tracks` was
aborting — reaches a verdict instead of killing the job:

```
run    event   main_id  flash_time_us  in_beam  len_main_cm  tgm  stm  fc  stmfit  label
18255  278794  7        0.695          1        203.1        0    1    0   eval    STM
```

Root `work-mcp1kall-d60fixchk`.

**2. Byte-identical everywhere else.** Pre-fix vs post-fix binary, same Q/L
pctrees by symlink, same flags, compared by archive member content
(`mabc-pr.zip` + `pctree-pr-*.tar.gz`) **and** the per-bundle tsv:

| gate | events | STM tags | arms | result |
|---|---|---|---|---|
| 1 | 60 STM-tagged | 60 | `d60sr1` → `d60sfix` | **60/60 identical — PASS** |
| 2 | 20 (entries 0-19) | 5 | `d60nr1` → `d60nfix` | **20/20 identical — PASS** |

Row counts and tag counts match exactly as well (761 rows / 60 STM, and
233 rows / 5 STM). Reproduce with

```bash
./d60_ab_report.py work-mcp1kall-d60sr1 work-mcp1kall-d60sfix
./d60_ab_report.py work-mcp1kall-d60nr1 work-mcp1kall-d60nfix
```

The gate is meaningful only because §7 first established the chain is
deterministic, and only because the same script flags a genuinely different
config **60/60 DIFFERENT** (negative control, §7).

**Not done, and deliberately so:** the full 1000-event re-run. §7's determinism
result plus two byte-identical gates over 80 events (206 STM tags, both the
beam-window and all-bundle configs) make the remaining risk small, and the owner
scoped this down. If it is wanted later: `TAG=d60fix ./run_full1k_nusel.sh 1000
24`, then `./d60_ab_report.py work-mcp1kall-d60base work-mcp1kall-d60fix`
— but build the arm with **whole** `ql_evt<ID>/` symlinks, not pctree-only, or
every Bee zip will mismatch on `runNo` (§7 caveat).

## 7. Determinism of the Q/L-input → tagger chain (measured; doc 49 §4a retired)

Run **before** the fix, on the unmodified binary, because an A/B is meaningless
until the noise floor is known. Scope = the PR job: it reads the Q/L pctree and
runs `switch_scope → steiner → fiducialutils → tagger_check_tgm →
tagger_check_stm → tagger_check_fc`. Arms share one set of Q/L pctrees by
symlink, so the *only* variable is the run itself.

| arms | config | ASLR | N | STM tags | result |
|---|---|---|---|---|---|
| `d60base` vs `d59k` (≈1 h apart) | production | pinned `-R` | **431** | — | `pctree-pr` **431/431 identical**; `mabc-pr.zip` **430/430 identical** after normalizing `runNo/subRunNo` (see caveat) |
| `d60nr1` vs `d60nr2` | production | **active** | 20 | 5 | **20/20 identical** (archives + tsv) |
| `d60sr1` vs `d60sr2` | production, STM-tagged events only | **active** | 60 | 60 | **60/60 identical** |
| `d60bw1` vs `d60bw2` | **pre-doc-56** `-no-bwonly` | **active** | 60 | 146 STM / 256 TGM | **60/60 identical** |
| *negative control* `d60sr1` vs `d60bw1` | genuinely different config | — | 60 | — | **60/60 flagged DIFFERENT** — the harness is sensitive |

**Conclusion: the chain is run-to-run deterministic, with and without
`setarch x86_64 -R`.** ASLR was verified genuinely active for the un-pinned arms
(`/proc/sys/kernel/randomize_va_space` = 2, and the heap base varies between
plain `exec`s while `-R` pins it). 206 STM tags across 80 events moved by zero.

**Doc 49 §4a's "±7 STM tags out of ~44" noise floor does not reproduce.** Two
hypotheses were tested and both failed to explain it: the run-to-run fix
`309d41c7` (2026-07-10) *predates* doc 49 (2026-07-25) so it cannot be the
cause, and the doc-56 beam-window gate is not the cause either — the
`-no-bwonly` arm, which tags 146 STM + 256 TGM instead of 60, is also perfectly
stable. What actually changed between 2026-07-25 and now is not established
here; re-running doc 49's exact configuration at its own commit would be needed,
and that is out of scope. **Practical consequence: `setarch x86_64 -R` is no
longer required for A/B work on this chain** — it remains harmless and the
production driver still applies it.

### 7a. The claimed mechanism, tested directly — and a trap that may explain it

Doc 49 named a specific upstream cause: `CreateSteinerGraph` emitting different
steiner graphs on identical input, so clusters lose `steiner_pc` and
`check_stm_conditions` exits at `no steiner_pc`. Tested head-on across the
un-pinned `d60bw1`/`d60bw2` pair (the `-no-bwonly` arms exercise it hardest:
1112 `steiner terminal` + 1179 `no steiner_graph` lines):

```bash
extract() { grep -ao "create_steiner_tree: only [0-9]* steiner terminal[^,]*\|no steiner_graph for [a-z]* [0-9]*" "$1" | sort; }
# per event, compare extract(arm1 log) vs extract(arm2 log)
```

**60/60 events identical.** Which cluster/assoc loses its graph, and why, does
not move.

**But getting there took three tries, and the two failures are the trap.**
Comparing the raw log lines gave **60/60 "DIFF"**; stripping the `[hh:mm:ss.mmm]`
prefix still gave **54/60 "DIFF"**. Every one of those was an artifact:

* WCT log lines **tear** (non-atomic multi-line spdlog
  writes), so a `MABC timing: ... took 23.003219 ms` line — whose wall-clock
  number legitimately varies run to run — gets spliced *into the middle of* a
  steiner warning. The tear position and the timing text then differ while the
  warning itself is identical.
* Even after the leading timestamp is stripped, a spliced-in `.189]` fragment
  from another line survives inside the payload.

**Rule: never conclude non-determinism from a log diff on this chain.** Compare
the products (`hash_archive.py` over `mabc-pr.zip` / `pctree-pr-*.tar.gz`) or
extract the semantic substring with `grep -ao`. Note also doc 59's GOTCHA 2 —
`grep` without `-a` silently reports *nothing* on a log whose tear produced
invalid UTF-8, so a torn arm can appear to have **fewer** warning lines than its
twin.

**Hypothesis, not a conclusion:** those two failure modes together could produce
exactly doc 49's reported signature ("run A: no steiner_graph for main 1; run B:
main 1, main 11, assoc 15, 16, 19") without any real non-determinism — a torn
line in one arm hides whole entries from `grep`. That is worth checking before
doc 48 §4's retraction is treated as settled, since the retraction rests on the
±7 floor being real. Confirming it needs doc 49's arms re-measured at doc 49's
commit; **not done here, and not something to assume either way.**

**Caveat on `mabc-pr.zip`, worth knowing before trusting any Bee-zip diff.** The
Bee JSON embeds `runNo` / `subRunNo` / `eventNo`. `run_nusel_evt.sh` reads those
from the Q/L step's products, so an arm whose `ql_evt<ID>/` holds *only* the
pctree tarball silently gets `runNo="0", subRunNo="0"` and **every** Bee zip
mismatches while the physics is bit-identical. That is exactly what the first
431-event comparison reported (430 "DIFFERENT", all of them this field, with
`pctree-pr` matching everywhere). Symlink the *whole* `ql_evt<ID>/` directory
into an arm, not just the tarball, or normalize those three keys.

## 8. Open question for the owner

Is `search_other_tracks` supposed to propose a 4.7 cm candidate at all? The
quality cuts it applies (`number_not_faked`, `max_dis_*`) test how far the
candidate's points sit from the already-fitted segments in 2D; they never look
at the candidate's own length. A minimum path length there would remove the
class of input rather than the symptom, but it *would* change tagger behavior
and needs its own knob and its own A/B. §5 is the crash fix only.
