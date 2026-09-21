# prod-2026-09-21c — the PR chain's allocator default becomes jemalloc (local running only)

## Why this generation exists

> "Let's flip for local running them."
> — the owner, 2026-09-21

Doc `sbnd_xin/docs/119_pr-profiling-rounds.md` §11.8. `run_pr_chain_batch.sh`'s allocator default
moves from `libtcmalloc_minimal` (round 1, §5) to **jemalloc 5.3.0**:

```
-SBND_TCMALLOC_LIB=${SBND_TCMALLOC_LIB:-/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4}
+SBND_PR_ALLOC_LIB=${SBND_PR_ALLOC_LIB:-${SBND_TCMALLOC_LIB:-/usr/lib/x86_64-linux-gnu/libjemalloc.so.2}}
+SBND_TCMALLOC_LIB=$SBND_PR_ALLOC_LIB
```

Previous generation `ref/prod-2026-09-21b` is kept. **Exactly one of the 26 artifacts moved** —
`runner_alloc.txt`. The other 25 are bit-identical, so **no compiled configuration changed and no
other detector is touched**.

## Read this before quoting the flip

**It is a tail-only change.** Round 5 measured four allocators through one invocation on the seven
doc-116 peak-RSS events, all read with `getrusage` CHILDREN high-water:

| | mean peak RSS | max | events > 2 GiB | in-job TICK |
|---|---:|---:|---:|---:|
| tcmalloc (was production) | 1.862 GiB | 2.156 | **3 of 7** | — |
| glibc | +1.4 % | +2.3 % | 4 of 7 | +18.8 % |
| **jemalloc 5.3.0** | **−18.8 %** | **−19.1 %** | **0 of 7** | **−4.0 %**, faster on 7 of 7 |
| tcmalloc + `RELEASE_RATE=10` | −4.4 % | −3.9 % | 1 of 7 | +0.4 % |

**And on the 62-event gate manifest it buys nothing** — +0.1 % mean RSS on nuecc, and 1.197 vs
1.197 GiB on the p50 event. **On jobs of a few seconds it costs CPU**: +6.6 % on `cv` (3 s jobs),
+4.5 % on beam-off (1 s), i.e. +0.20 s and +0.04 s per event against −3.2 s on an 82 s tail event.
Do not quote an average; there is nothing in it.

## Scope — what this generation does NOT cover

- **SBND's LArSoft production is NOT affected and is on glibc.** `lar -c
  wcls-img-clus-matching-xin.fcl` preloads nothing, `lar` links `libc.so.6`, and no art library
  pulls in an allocator. Round 1 never reached it either. Doc 119 §11.9 carries this as the
  campaign's largest open item.
- **Stage A is unchanged**: `run_clus_evt.sh` keeps `libtcmalloc_minimal`; jemalloc was never
  measured there.
- **`run_pr_evt.sh` is unchanged**, by design — the frozen `-stm` / `-tgm` A/B arms keep their
  exact process environment.

## Gate evidence

- **G1, the flipped default reproduces the measured arm** (`docs/119_figs/119_r6_gate_jdef.txt`):
  a fresh 62-event arm run with **no allocator environment at all**, gated against the measured
  `jem` arm at provenance-only allowance — **0 differences outside allowance** on all three
  samples. All 62 events `rc=0` and all 62 confirmed on jemalloc from the runner's own per-event
  log line, not inferred from the arm name.
- **Byte identity** (`119_r5_gate_jem.txt`): `lever_gate.py --vs r3 jem` PASS on the same 62
  events, with non-provenance differences **exactly** equal to the null pair's (10 / 24 / 1), so
  jemalloc moves nothing that two runs of one configuration do not.
- **G2, the tripwire caught this flip and named it** (`119_r6_gate_drift.txt`) — the first test of
  the hole closed in §11.4. Before that change this flip would have reported PASS 25/25.
- **G3, alias precedence, two-sided** (`119_r6_alias_precedence.txt`): with no allocator env a job
  loads `libjemalloc.so.2`; with round 1's `SBND_TCMALLOC_LIB` spelling it still loads
  `libtcmalloc_minimal.so.4`.
- Against this generation the tree is **PASS 26/26** (`119_r6_gate_post.txt`).

## The knob

Precedence: `SBND_PR_ALLOC_LIB` → `SBND_TCMALLOC_LIB` (round 1's spelling, still honoured) → the
jemalloc default. The `SBND_PR_TCMALLOC` on/off switch, the join-don't-assign rule and round 5's
missing-library warning are all unchanged.

**Escape hatch**, either spelling:

```bash
SBND_PR_ALLOC_LIB=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4   # back to round 1
SBND_PR_TCMALLOC=0                                                      # back to pre-round-1
```

**Record scripts now pin their allocator**, in the same commit as the flip:
`stageB_lever.sh LEVER=tcm`, `stageB_tail.sh ALLOC=prod` and `stageB_alloc.sh ALLOC=tcm|rel` all
name `libtcmalloc_minimal` explicitly instead of inheriting a default that no longer means what it
meant when those arms were produced. A script that does *not* pin inherits whatever the default is
on the day it runs — true since round 1, and the reason every arm now logs its own allocator.

## Reproduce

```bash
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
ALLOC=default JOBS=8 $SX/scripts/d119/stageB_alloc.sh nuecc        # and cv, off -- the G1 arm
python3 $SX/scripts/d119/lever_gate.py --vs jem jdef               # G1  -- PASS, 62 events
python3 $SX/scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-21b  # G2 -- DRIFT, named
python3 $SX/scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-21c  # PASS 26/26
```
