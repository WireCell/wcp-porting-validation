# doc pr/76 — the `prod0813` campaign: retire to 20 G, land pr/75, re-run PR on all five samples

Owner asked for four things, in this order: clean the `work*` tree back to about
20 G; re-process the five samples with the latest software, saving enough to
hand-scan the neutrino vertices; deliver three Bee links (48 nueCC, 19 NC π⁰,
1000 data); and serve the 48 nueCC events on port 5017. Data samples run as
`data`, MC as `sim`. 32 CPUs.

Status: **all four delivered.** Nothing in this round changes reconstruction —
the only code landing is the merge of an already-gated, default-OFF recording
knob, and that merge is proven byte-identical when off.

## Repro block

```bash
cd sbnd_xin

# 1. retire 74 G -> 20 G   (details: docs/work-tags.md "RETIREMENT ROUND 2026-08-13")
python3 scripts/retire/plan_20260813.py                            # 13-name KEEP, 6 asserts
RETIRE_JOBS=24 python3 scripts/retire/archive_records_20260813.py  # integrity PASS 388/388
CONFIRM=yes scripts/retire/retire_20260813.sh A                    # 388 dirs, 53 GiB

# 2. land pr/75 and gate it (toolkit)
git merge pr75-vertex-scoreboard        # 3 knob-append conflicts, keep both sides
wcbuild && ./build/clus/wcdoctest-clus  # 208/208
PR_JOBS=32                        ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-gate0813-head48    data  # pre-merge binary
PR_JOBS=32                        ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-gate0813-merged48  data
PR_JOBS=32 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-gate0813-on48  data
PR_JOBS=32                        ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-gate0813-repeat48  data  # A/A'
python3 scripts/retire/gate_cmp_arms.py work-gate0813-head48   work-gate0813-merged48
python3 scripts/retire/gate_cmp_arms.py work-gate0813-merged48 work-gate0813-on48
python3 scripts/retire/gate_cmp_arms.py work-gate0813-merged48 work-gate0813-repeat48

# 3. the campaign -- PR stage only, onto the surviving cb0805 Q/L pctrees
PR_JOBS=32 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-nuecc48-prod0813 data
PR_JOBS=32 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-ncpi0-cb0805   work-ncpi0-prod0813   data
PR_JOBS=32 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-mcp1k-cb0805   work-mcp1k-prod0813   data
PR_JOBS=32 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-r1qlmc-cb0805  work-r1qlmc-prod0813  sim
PR_JOBS=32 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-r2mc-cb0805    work-r2mc-prod0813    sim

# 4. census, then prune to PR-results-only
for s in nuecc48 ncpi0 mcp1k r1qlmc r2mc; do
  python3 pr_scores_table.py --root work-$s-prod0813 --sample $s \
      --out products/prod0813/$s-scores-prod0813.tsv --summary
  awk -F'\t' 'NR>1 && $15==1 {print $4}' products/prod0813/$s-scores-prod0813.tsv \
      | sort -n > products/prod0813/events-$s-prod0813.txt
done
python3 scripts/retire/prune_unevaluated.py --apply work-*-prod0813   # 0 QUARANTINE

# 5. three Bee sets
for s in nuecc48 ncpi0 mcp1k; do
  python3 scripts/bee/make_pr_bee.py -q work-$s-cb0805 -p work-$s-prod0813 \
      -o bee/prod0813/$s-prod0813.zip $(cat products/prod0813/events-$s-prod0813.txt)
  bash upload-to-bee.sh bee/prod0813/$s-prod0813.zip     # SERIALIZE: cookies.txt races
done

# 6. serve the hand scan
./pr_display/serve_pr_display.sh 5017 --scan-tag vtxscan-prod0813 \
    "work-nuecc48-prod0813/pr_evt*/calib-pr-evt*.json"
# ssh -L 5017:localhost:5017 wcgpu1.phy.bnl.gov ; http://localhost:5017/pr_display_viewer
```

## 1. Scope, and the one thing this campaign is NOT

Owner chose a **PR-stage-only** reprocessing. Imaging and Q/L are not
regenerated; the PR chain runs onto the Q/L pctrees the 2026-08-05 `cb0805`
campaign produced. Two consequences, both deliberate:

- **The campaign mixes two software epochs.** Q/L + tagger-tail products date
  from toolkit `a1ea3789` (2026-08-05); PR products date from `1e534c6f`
  (today). 82 clus/cfg commits separate them, and some of those (pr/66) touched
  Q/L. This is a scope limit, not a clean-slate reprocessing, and any statement
  of the form "prod0813 is current production end-to-end" is false.
- **The five `work-*-cb0805` hubs are now campaign INPUT, not a record layer.**
  They can no longer be thinned, and the 2026-08-13 retirement round runs no
  Phase 4 for exactly that reason.

Imaging is the one stage where reuse is provably free: `git log a1ea3789..HEAD
-- img/ sigproc/ aux/` is **empty**, so no imaging code has changed since those
npz files were written.

## 2. What "latest software" is

Toolkit **`1e534c6f`** = `80eeb592` (pr/73 round 2 flip, 09:44 today) + the
merge of `pr75-vertex-scoreboard`.

The installed `libWireCellClus.so` was already at 09:09, *earlier* than HEAD's
clus commit timestamp (09:32) — which looks like the stale-library trap (M1) and
is not. The edits were made 09:02–09:08 and built at 09:09:31; the commits were
made afterwards. Proven rather than argued: a fresh `wcbuild` recompiled **0
objects**, and no C++ source is newer than the installed library.

### Build fingerprint — one binary produced all 1090 events

```
local/lib/libWireCellClus.so   mtime 2026-08-13 20:02:48   md5 426ac7aba284752a62404d955b0f61a3
```

This matters more than usual here: the campaign ran **20:29 → 20:42 in a shared
tree with a concurrent session active**, and the freshness proof was taken at
20:02, *before* it. A rebuild by that session mid-campaign would have split the
1090 events across two binaries with nothing in the output to show it. Re-checked
after the last arm finished: mtime **unchanged** at 20:02:48, HEAD still
`1e534c6f`, working tree clean, and `wcbuild` recompiled **0 objects**. So the
provenance above holds for every event, not just the first.

> **Standing process item, still owed** (`docs/work-tags.md`, 2026-08-05): *"have
> the runner write the `libWireCellClus.so` md5 into each arm … would make the
> next occurrence detectable instead of inferable."* The five
> `work-*-prod0813` arms are now name-protected production references in
> `PROTECTED.txt` and **not one carries a build fingerprint** — the md5 above is
> recorded in this doc only. `scripts/runners/s4_nuecc48.sh:24` already computes
> it; writing it into the arm is the fix.

### The pr/75 merge

doc pr/75 shipped `vertex_scoreboard` on branch `pr75-vertex-scoreboard`
(`04b6e47d`, parented on `40651cb2`) because a concurrent session held
uncommitted work in every file it touched. Since then pr/74 rounds 3–4 and
pr/73 round 2 landed on the same files, so the fast-forward that doc
anticipated no longer exists. Three content conflicts, every one a site where
both sides append a knob to the same list — resolved by keeping both:

| file | conflict |
|---|---|
| `cfg/pgrapher/common/clus.jsonnet` | `tagger_check_neutrino()` signature |
| `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` | TLA declaration + pass-through |
| `clus/inc/WireCellClus/TaggerCheckNeutrino.h` | member block (`m_sgp_max_sep` + `m_vertex_scoreboard`) |

`clus.jsonnet` (sbnd), `NeutrinoPatternBase.h`, `NeutrinoVertexFinder.cxx`,
`TaggerCheckNeutrino.cxx` and `doctest_clus_knob_defaults.cxx` auto-merged.
Net +462 / −7.

## 3. Gates

All on the 48-event nueCC manifest (`work-nuecc48-cb0805`), DL on, `PR_JOBS=32`,
compared by archive **member content** via `abtest/hash_archive.py` — never raw
bytes (M2).

| gate | bar | result |
|---|---|---|
| `./build/clus/wcdoctest-clus` | pass | **208/208 cases, 2060/2060 assertions** |
| compiled config, knob off vs HEAD | byte-identical | **identical**, md5 `544dc81b379f6d657ab0ec0f43dc4171` |
| compiled config, knob on | key present | differs by **exactly one line**, `"vertex_scoreboard": true` |
| `head48` vs `merged48` | 0/48 | **0/48** — the merge changes nothing when off |
| `merged48` vs `on48` | 0/48 | **0/48 on reconstruction** (see §3.1) |
| `merged48` vs `repeat48` (A/A′) | 0/48 | **0/48**, every artifact incl. nusel |
| knob-on smoke | dump == code's own TRACE | **agrees to 4.5 × 10⁻⁴**, §3.2 |

Freshness proof (M1) done before every arm.

**The five gate arms are RETAINED** — `work-gate0813-{head48,merged48,on48,repeat48}`
(183–258 MB each) and `work-gate0813-trace10550` (5.5 MB). Deliberately: this
file's tables are the only summary of them, and a tag names a *config*, not a
build, so re-running `head48` today would not reproduce the pre-merge binary —
retiring them would permanently retire the ability to re-check any row above.
813 MB against 543 G free is not worth that. `trace10550` in particular cannot
be regenerated from the campaign arms at all: the `DL rerank cand` lines it
holds are below the `debug` level every production run uses. They are the first
thing to drop in a future round once §3's numbers stop mattering.

### 3.1 The knob-on gate's 2/48 nusel difference — a logging artifact, established not assumed

The `merged48` vs `on48` comparison initially reported 2 differing artifacts.
Both were `nusel-evt<ID>.tsv`; **`pctree` and `mabc` were byte-identical on all
48 events**. Chased to the end rather than waved off:

- Exactly one cell moves in each: `stmfit`, and **in opposite directions** —
  evt 52672 `eval`→`contained`, evt 219295 `contained`→`eval`. A systematic code
  change cannot flip a verdict both ways.
- `stmfit` is the only column in that row parsed purely from **log text**;
  `tgm`/`stm`/`fc` beside it come from the post-PR tree. Those are **identical in
  every arm** (`0/0/1`), and `fc=1` is precisely the condition that produces the
  "fully contained (Mid Point A)" STM-fit skip — so the tagger took the same
  branch each time.
- The mechanism, verbatim from the two logs — another thread's message spliced
  into the middle of the skip line, destroying the cluster-id token that
  `nusel_extract.py` joins on:

  ```
  evt 52672  off arm: "er_tree: only 0 steiner terminal(s) r 9 no STM fit: fully contained (Mid Point A)"
  evt 219295  on arm: "eck_stm_conditions: cluster 15 no STM fit[20:06:58.102] I [  clus  ] <CreateSteinerGraph:..."
  ```

- The tearing is **deterministic per pipeline**, not run-to-run noise: the A/A′
  repeat (`merged48` vs `repeat48`, identical pipeline, same binary, no rebuild
  between) reproduced every TSV exactly. Adding one visitor shifts thread
  scheduling, which moves where the writes tear. Same effect doc pr/75 recorded.

`scripts/retire/gate_cmp_arms.py` now reports **reconstruction** and
**log-derived** artifacts on separate lines, so this class can neither mask a
real regression nor manufacture a false one.

### 3.2 Knob-on smoke: the dump against the code's own TRACE lines

The `DL rerank cand` TRACE lines are below the `debug` level the runner uses, so
this needed one event re-run at `SBND_WCT_LOGLEVEL=trace` (evt 10550):

| voxel rank | cluster | vertex_id | TRACE TOTAL | dump total |
|---:|---:|---:|---:|---:|
| 0 | 28 | 28012 | 3.8280 | 3.8283 |
| 1 | 28 | 28013 | 3.8590 | 3.8588 |
| 2 | 33 | 33022 | 3.8500 | 3.8498 |
| 3 | 48 | 48052 | **5.9770** | **5.9771** |
| 4 | 36 | 36028 | 3.7330 | 3.7330 |

Worst disagreement across the totals **and all seven composite terms**:
**4.45 × 10⁻⁴** — exact to the precision TRACE prints. Also verified: the seven
terms sum to `total` on every row; `dl_winner` is unique and equals
`final_vertex_id` (48052); `dl_best_score` equals `max(total)`; and
`vertex_id // 1000` decodes to the cluster the TRACE line names, on all five.

## 4. The campaign

1090 events, **1090 rc=0, 0 failures, 0 timeouts**, ~13 min of wall clock at
32-way (nueCC48 74 s, NCpi0 28 s, r1qlmc 13 s, r2mc 15 s, mcp1k 10 min).
Peak loadavg 43 on 64 cores.

No `SBND_*` knob overrides were passed: empty env ⇒ no TLA ⇒ cfg default ⇒
current production (doc 68). `PR_EXTRA_STAGES=pr_display` alone is enough —
`run_pr_chain_batch.sh:175-177` turns `SBND_VERTEX_SCOREBOARD` on with it.

Each sample ran into a **fresh `out_root`**, not `out_root == ql_root` as doc 71
used, which keeps the `.lineage_reality` guard non-vacuous (that guard writes its
stamp before reading it, so it checks nothing when the roots are the same). The
guard therefore actually verified the data/sim split against the cb0805 arms'
recorded lineage: `data data data sim sim`.

### `nu_evaluated` census, against cb0805

| sample | mode | events | prod0813 | cb0805 | change |
|---|---|---:|---:|---:|---|
| nueCC48 | data | 48 | **47** | 47 | — |
| NC π⁰ | data | 19 | **19** | 19 | — |
| mcp1k | data | 1000 | **445** | 445 | −52613, −59723, +48895, +281953 |
| r1qlmc | sim | 10 | **4** | 4 | — |
| r2mc | sim | 13 | **5** | 6 | −37 |
| **total** | | **1090** | **520** | 521 | |

The mcp1k count is unchanged but the *set* is not: two events in, two out. This
is a fresh census at a moved operating point, not a gate — doc 71 §5 documents
the same behaviour across the previous flip.

nueCC48's one non-evaluated event is **116962**, the known case whose only
in-window cluster is cosmic-tagged, so the tagger skips it — correct, not a bug.

## 5. What "save only if they have PR results" cost, and the guard on it

Owner: *"you only need to save them, if they have PR results."* For an event
with no selected neutrino candidate, `prune_unevaluated.py` drops
`pctree-pr-evt*.tar.gz`, `tracking-pr.root`, `mabc-pr.zip` and
`calib-pr-evt*.json`, and **keeps the record layer** — `wct_pr_evt<ID>.log`,
`stdout.log`, `rc.txt`, `.time.meta`, `nusel-evt<ID>.tsv`. The log is kept
deliberately: `nu_evaluated` exists in no other artifact, so discarding it would
make the census unfalsifiable afterwards, and it costs ~150 KB against the ~8 MB
it replaces. Freed **1.37 GiB** over 570 events; all 520 evaluated events keep
every product, and all 1090 logs survive (verified).

### The three-marker rule, and why one grep would have destroyed a real result

Pruning on a single `grep` for "selected main cluster" is the obvious
implementation and is **unsafe here**. Every event must match exactly one of
three mutually exclusive markers — selection / no-main / cosmic-skip — and
anything matching none, or matching a non-selection marker while a
`calib-pr-evt*.json` exists, is **quarantined and never pruned**.

That guard fired, for a reason the plan did not predict:

> **WCT logs are frequently invalid UTF-8**, because tearing splices bytes
> mid-character. GNU grep in a UTF-8 locale then treats the file as binary and,
> for a plain match, prints **nothing** and exits 1 — no warning, and `grep -c`
> emits no count at all.
>
> ```
> $ grep -c 'selected main cluster' work-nuecc48-prod0813/pr_evt256587/wct_pr_evt256587.log
> (no output, rc=1)
> $ grep -a -c 'selected main cluster' .../wct_pr_evt256587.log
> 1
> ```
>
> Evt 256587 has a real selection — `selected main cluster 11 (t0 1.482 us,
> L 130.6 cm, 89 associated)` at byte 360702 — and a shell loop built on plain
> grep classified it as not-evaluated. Under `--apply` that would have deleted a
> genuine PR result, removing it from the Bee set and the hand scan with nothing
> left on disk to recompute from.

Python's `open(..., errors='replace')` has no such failure mode, which is why
`prune_unevaluated.py`, `pr_scores_table.py:99`, `nusel_extract.py:262` and
`make_pr_bee.py:83` all read that way — the production tools were never at risk.
**If you must grep a WCT log, pass `-a`.**

Result: **0 QUARANTINE across all 1090 events**.

`pr_scores_table.py`'s field-15 census and the three-marker rule agree exactly on
all five samples — but they are **not independent evidence**: both parse the same
log line and share the same tearing failure mode, so agreement between them
cannot rule tearing out. What makes 0 QUARANTINE meaningful is the
`calib-pr-evt*.json` presence cross-check, which comes from the filesystem rather
than the log: `PrDisplayDump` emits that dump off the selected main cluster, so a
dump present beside a "no selection" verdict is the signature of a torn selection
line, and no event exhibited it.

## 6. The three Bee sets

Built from the cb0805 Q/L root (`mabc-all-apa.zip`) × the prod0813 PR root
(`mabc-pr.zip`), filtered to `nu_evaluated=1`, 9 layers per event.

| set | events | zip | URL |
|---|---:|---:|---|
| 48 nueCC | 47 | 26.4 MB | `https://www.phy.bnl.gov/twister/bee/set/c0f0c371-689f-4604-9191-e99fc39b72a9/event/list/` |
| 19 NC π⁰ | 19 | 9.8 MB | `https://www.phy.bnl.gov/twister/bee/set/6c01e46a-e87c-4b85-a38b-3ee3014beba0/event/list/` |
| 1000 data | 445 | 205.6 MB | `https://www.phy.bnl.gov/twister/bee/set/41fa9f6d-a923-49d2-afce-e96712aba0ee/event/list/` |

All three verified live: `curl -k` returns 200 and the event-list page links
exactly 47 / 19 / 445 events. (Bare `curl` fails with an OpenSSL error here —
`upload-to-bee.sh` uses `curl -k`, and a check without it reports a misleading
`http=000`.)

**`bee_idx` alignment**: nueCC48 and NCpi0 are **identical to the prod0811 index
files**, so a scan done against those sets transfers by index. mcp1k is a
445-event set and is *not* index-comparable to the 50-event `mcp1k50` sets.
Bridge quality is unchanged (nueCC48 `NO-BRIDGE` 8→11 of 1855 rows,
`NO-FLASH-MATCH` 55→55; NCpi0 4→3 and 33→33).

The 205 MB upload — 8× the largest previously sent to twister from this tree —
went in a single POST in 22 s; the chunked fallback was not needed.

Records: `bee/prod0813/*.{zip,index.txt,prid-map.txt,url}` local;
`docs/pr/{nuecc48,ncpi0,mcp1k}-prod0813.{index,prid-map}.txt` and the five
`*-scores-prod0813.tsv` git-tracked.

## 7. The hand scan on 5017

```
ssh -L 5017:localhost:5017 wcgpu1.phy.bnl.gov
http://localhost:5017/pr_display_viewer
```

Scan tag **`vtxscan-prod0813`** — fresh, per M13. The pr/75 tags `vtxscan1` and
`uitest75` (one UI-test label each) are untouched; they live under
`sbnd_xin/vertex_labels/`, outside `work-*`, and were never at risk from the
retirement round. `vertex_labels/vtxscan-prod0813/` does not exist yet — the
viewer creates it on the first save, so the owner writes the first label. The
write path was verified through a throwaway tag using the same tmp+rename the
viewer uses, rather than by saving a label under the real tag and deleting it.

**47 events, not 48** — evt 116962 produces no calib dump because no main
cluster was selected. All 47 carry a filled scoreboard, and all 47 took route
`dl-rerank-accept`. Verified on the *served* document, not just the source: the
event selector offers 47 options and the scan panel is the fifth child of the
left column.

## 8. What is NOT established

- **Q/L products are from the 2026-08-05 epoch** (§1). This is not an
  end-to-end current-production reprocessing.
- **The determinism floor (P3) is only partly discharged.**
  `merged48` vs `repeat48` is a matched-layout A/A′ pair on the current binary,
  48/48 identical including the log-derived TSVs — and
  `run_pr_chain_batch.sh:1067` already runs every event under
  `setarch x86_64 -R`, so both arms are ASLR-disabled. **The cross-layout leg is
  still owed**: one ASLR-on arm compared against those two (the
  `pr37_a2_floor.sh` shape). Do not read "48/48" as the full floor.
- **No labels have been collected** and no operating point has moved. The scan
  exists; doc pr/52 §5.2's analysis — split failures by `route` and
  `not_a_candidate` — has not been done.
- **The oc56 truth table is not recomputable** until someone runs a fresh
  `PR_OC56_SCAN_DUMP=1` arm; the retirement round removed all three arms
  `oc56_truth.py` names (one of which was already stale). Owner-confirmed.
- The 445-event mcp1k Bee set is **not index-comparable** to the previous
  50-event mcp1k sets.
