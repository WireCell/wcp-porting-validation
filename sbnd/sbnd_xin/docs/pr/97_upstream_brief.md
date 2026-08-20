# Technical brief: in-process gojsonnet leaves a Go runtime that kills long WCT jobs

**Status of this file.** These are the measurements only, written so they can be
handed to whoever fixes the toolkit side. The framing, the ask and any GitHub
Issue/PR text are the human author's to write — this file is deliberately not
a preamble. Numbers here are reproducible from doc pr/97 §5.4-5.6.

## The defect in one line

`wire-cell -c foo.jsonnet` evaluates jsonnet **in process** via `libgojsonnet`,
which starts a Go runtime whose ~ncores threads then live for the entire job;
at ~120 s of process life one of those threads jumps to PC `0x0` and the job
dies with SIGSEGV, at a rate of ~4 % per run.

## Where

* `apps/src/Main.cxx:270` — `Json::Value one = p.load(filename);`
* `util/src/Persist.cxx:438` — `.jsonnet` (or no extension) →
  `jsonnet_evaluate_file(m_jvm, ...)`, i.e. cgo into `libgojsonnet.so`.
* `jsonnet_destroy()` releases the VM but cannot unload the Go runtime; Go has
  no shutdown API, so the threads persist to process exit.

## Measurements

Thread census of a live process, `ls /proc/<pid>/task | wc -l`, same job, same
event, same binary — only how the config is supplied differs:

| config handed to `wire-cell` | threads |
|---|---|
| `.jsonnet` (in-process gojsonnet) | **65** |
| the same config precompiled by `wcsonnet` to `.json` | **1** |

So the Go threads are created on first jsonnet *evaluation*, not at library
load — which is what makes a subprocess fix clean (below).

Crash rate on one SBND event (18255-178410, a Q/L matching job, ~135 s wall),
all at `-j 1`-equivalent, ASLR pinned with `setarch x86_64 -R`:

| arm | runs | crashes |
|---|---|---|
| baseline, `.jsonnet` config | 108 | 4 (3.7 %) |
| same, under `gdb` (control for the arms below) | 48 | 2 (4.2 %) |
| `.jsonnet` + `GOTRACEBACK=crash` | 28+ | 1 |
| **config precompiled to `.json`** | 90+ | **0** |
| `.jsonnet` + `GODEBUG=asyncpreemptoff=1` | 60 | 0 |
| `.jsonnet` + `GOGC=off` | 50+ | 0 |

Two live `gdb` captures of the fault (`gdb -batch ... -ex run -ex "thread apply
all bt"`, all Go signals passed through):

| capture | signal delivered to | its PC | WCT's main thread at that instant |
|---|---|---|---|
| 1 | a non-main thread (LWP top of the startup range) | **0x0**, no stack | healthy — `__dynamic_cast` inside `Cluster::hough_transform` |
| 2 | a different non-main thread | **0x0**, no stack | healthy — `std::vector<double>::operator[](15451)`, a valid index |

In capture 1, of the 34 threads then alive, **32 were parked in
`runtime.futex`** and one in `runtime.usleep` (Go's `sysmon`). Every non-main
thread in this process comes from `libgojsonnet` (see the census above), so the
faulting thread is a Go runtime thread, and WCT's own thread is provably not
faulting.

**The trigger is a process-life deadline, not a code path.** Wall time of the
five crashing runs: **126, 127, 128, 128, 129 s**, against 119-142 s for
healthy runs of the same job. Subtracting ROOT's crash handler (measured: a
`gdb -batch -ex "thread apply all bt"` on this process costs 6.8 s) puts every
crash at **≈120 s of process life** — Go `sysmon`'s `forcegcperiod` is exactly
2 minutes.

Two side observations that cost us time and may cost the next person too:

* **ROOT's `TUnixSystem::StackTrace()` prints thread 1's stack, not the faulting
  thread's.** Four backtraces read as crashes inside our clustering code; they
  were time samples of the busiest thread. Their leaf frames were pure
  arithmetic (`__ieee754_acos`, `boost::histogram::axis::regular::index`) on
  valid arguments.
* **ROOT's SIGSEGV handler wins over Go's**, so `GOTRACEBACK=crash` yields no
  Go traceback at all — ROOT prints its trace and exits. Getting a Go-side
  stack will need ROOT's handler out of the way.
* A full valgrind memcheck pass over the crashing event completed with 3878
  error contexts, **zero of them in WCT algorithm code** — all Go runtime, ROOT
  streamers, or configure-time `SCEFieldTH3`.

## Suggested fix

Because the Go threads only appear on first evaluation, evaluating jsonnet in a
**forked child** keeps the parent free of a Go runtime entirely: `fork()`, child
calls the existing `Parser::load`, writes the JSON to a pipe, `_exit()`; parent
parses what it reads. No new dependency (POSIX `fork`/`pipe`), no change to the
jsonnet semantics, and it can ship behind a default-OFF flag so the legacy path
stays byte-identical. Cost measured here: **0.13 s** per job for a 56 KB
compiled config.

Open questions for whoever takes it:

* is the PC=0 jump a Go/cgo bug in its own right, or does it need ROOT's
  handlers to be installed over Go's? A build that keeps ROOT out of the process
  would separate these, and is the experiment we have not run;
* should the subprocess path become the default for `wire-cell`, given every
  job living past ~120 s is exposed, not just this event?

## What we deployed locally (not a toolkit change)

`wcp-porting-img` runners now compile with `wcsonnet` and hand `wire-cell` the
JSON (`SBND_PRECOMPILE_CFG=1`, default on; `=0` restores the legacy path), and
the batch drivers now fail loudly and non-zero on any failed event — a crashed
event used to be one quiet line and a 0-byte output that downstream consumers
read silently.
