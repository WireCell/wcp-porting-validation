# prod-2026-09-21b — the PR runners' allocator block joins the consumer set

## Why this generation exists

Doc `sbnd_xin/docs/119_pr-profiling-rounds.md` §11. **No production operating point moved.** This
generation exists because the *set of things the tripwire watches* grew by one:

```
NEW       : runner_alloc.txt   (in the compiled set, absent from the reference)
REFRESHED prod-2026-09-21b -- 26 artifacts (1 added, 0 dropped).
```

The previous generation `ref/prod-2026-09-21` is kept. **All 25 of its hashes are bit-identical
here** — the only line that differs between the two manifests is the added one. That is the
blast-radius statement: round 5 changed what is watched, not what runs.

## The hole this closes

Doc 119 §6.4 named it and left it open: **round 1 shipped a production change no artifact could
see.** `SBND_PR_TCMALLOC` defaulting on (`run_pr_chain_batch.sh`, doc 119 §5) preloads
`libtcmalloc_minimal` into every PR job for −10.9 to −16.7 % in-job CPU. That is a production
operating-point change living in a **runner**, and no runner was among the 25 consumers, so
`prod_cfg_gate.py` reported PASS while it landed.

Round 4 turned the hypothetical into something measured. The runner's own preload branch is

```bash
if [ "${SBND_PR_TCMALLOC:-$SBND_PR_TCMALLOC_DEFAULT}" = 1 ] && [ -e "$SBND_TCMALLOC_LIB" ]; then
```

so if the library ever stops existing — a package upgrade renaming the soname is enough — the job
silently drops to glibc. Before round 5 there was **no log line, no config change and no gate
failure**: round 4 had to read `/proc/<pid>/maps` to establish which allocator a running job
actually had. A ~16 % regression could land on production with nothing anywhere recording it.

## What was added, and why it is *this* and not the whole runner

`runner_alloc.txt` is the **allocator decision**, extracted by pattern from the three SBND runners
that set `LD_PRELOAD` — `run_pr_chain_batch.sh`, `run_pr_evt.sh`, `run_clus_evt.sh` — with leading
whitespace and comments stripped and **no line numbers**:

```
grep -hE '(LD_PRELOAD|SBND_TCMALLOC_LIB|SBND_PR_TCMALLOC|TCMALLOC_SO|WCT_TCMALLOC|WC_PRELOAD|^PYLIB=)' \
    "$SX/$_r" | grep -vE '^[[:space:]]*#' | sed 's/^[[:space:]]*//'
```

§6.4's own condition was that the minimal form is the only acceptable one: these runners are large,
churn constantly with A/B scaffolding, and hashing them whole would make the tripwire noisy enough
to be ignored. Moving the block, re-indenting it or rewriting its comments does **not** fire.
Changing which library is preloaded, or deleting the round-5 warning that makes a fallback visible,
**does**. The extracted text is kept in full in this directory, so a drift is named and not merely
detected.

## A second hole, found while closing the first

`prod_cfg_gate.py` only ever looked up the names already in the reference manifest, and `--refresh`
rewrote it over `sorted(want)`. So **adding a consumer to `compile_consumers.sh` did nothing at
all**: the new artifact was compiled, ignored, and never adopted, and the gate went on reporting
PASS on the old set. Doc 118 added three fit JSONs and the LArSoft chain and got away with it only
because it built its reference generation by hand.

Fixed in the same change: the gate now computes what the compile **produced**, reports
`NEW : <name>` for anything the reference does not list, **fails** on it, and `--refresh` writes the
manifest from the produced set (reporting what it added and dropped). This generation is the first
one created through that path.

## Runner change that ships with it

`run_pr_chain_batch.sh` now logs the allocator it chose, on both the per-event and the group path:

```
[evt 11239] alloc: LD_PRELOAD=…/libpython3.11.so.1.0:/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
WARNING: [evt 11239] allocator preload requested but missing: … -- running on glibc malloc
```

Log-only: no product, no compiled config and no TLA moves. It is what makes an arm's allocator
readable from the arm itself, which round 5's four-arm comparison depends on.

## Reproduce

```bash
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
python3 $SX/scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-21    # NEW: runner_alloc.txt, rc=1
python3 $SX/scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-21b   # PASS 26/26
diff <(awk '{print $2}' $SX/ref/prod-2026-09-21/consumers.sha256) \
     <(awk '{print $2}' $SX/ref/prod-2026-09-21b/consumers.sha256)    # one added line
```

Records: `docs/119_figs/119_r5_gate_pre.txt` (PASS 25/25 at unmodified HEAD, before the first edit
of the round), `119_r5_gate_new.txt` (the NEW detection), `119_r5_gate_post.txt` (PASS 26/26).
