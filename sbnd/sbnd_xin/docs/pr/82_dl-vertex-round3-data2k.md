# doc pr/82 — DL-vertex round 3: the 2000-event data sample, the harvest gap, and relabelling the three existing samples

**Status: PLAN. Nothing large-scale has been processed.** Four smoke tests were
run (§0b) and all pass; every number in §1 and §2 is measured, not projected from
another doc. No production flip, no knob change, no C++ or jsonnet edit is
proposed here.

This is the round pre-registered as "pr/82" in doc pr/81's `# NEXT ROUND`
section. It is written against the tree as it exists on 2026-08-16, which differs
from what pr/81 assumed in three ways that matter (§1.3).

---

## 0. Repro

Every number in §1 comes out of these. All read-only.

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# (a) the label inventory and the manual-pick count
python3 - <<'EOF'
import sys, collections; sys.path.insert(0, "vtx_rules"); import vtx_io
L = vtx_io.load_labels()
print(len(L), collections.Counter(l["tag"] for l in L))
print("manual rank-1 picks:", sum(1 for l in L if l["truth_vid"] is None))
print("source still resolves:", sum(1 for l in L if __import__("os").path.exists(l["source"])))
EOF

# (b) carry-forward of the 473 old labels onto the current-production arms
#     -> 449 / 15 / 9 at 1 cm / 1-3 cm / >3 cm, median 0.0000 cm, 380 bit-identical
python3 - <<'EOF'
import sys, os, json, collections; sys.path.insert(0, "vtx_rules"); import vtx_io
ARM = {"vtxscan-prod0813": "work-nuecc48-pr87ion3",
       "vtxscan-prod0813-ncpi0": "work-ncpi0-pr87ion3",
       "vtxscan-prod0813-mcp1k": "work-mcp1k-pr87ion3"}
stat, near, vid, score = collections.Counter(), [], collections.Counter(), 0
for l in [x for x in vtx_io.load_labels() if x["tag"] in ARM]:
    ev = l["eventNo"]
    d = json.load(open(f"{ARM[l['tag']]}/pr_evt{ev}/calib-pr-evt{ev}.json"))
    ds = [(vtx_io.dist(l["truth"], vtx_io.vertex_xyz(v)), v) for v in d["vertices"]]
    ds = [(x, v) for x, v in ds if x is not None]
    dmin, vbest = min(ds, key=lambda p: p[0]); near.append(dmin)
    stat["exact" if dmin <= 1 else "loose" if dmin <= 3 else "broken"] += 1
    if l["truth_vid"] is not None:
        vid["same" if vbest["id"] == l["truth_vid"] else "moved"] += 1
    dm = vtx_io.dist(l["truth"], vtx_io.xyz(d.get("main_vertex")))
    score += dm is not None and dm <= 1.0
near.sort()
print(stat, "zero:", sum(1 for x in near if x == 0.0), "median", near[len(near)//2], vid)
print("production correct@1cm:", score, "/", len(near))
EOF

# (c) per-stage wall time on the surviving mcp1k arms
find work-mcp1k-pr87ion3 -name .time.meta | xargs cat | awk -F= '/wall_s/{print $2}' | sort -n \
  | awk '{a[NR]=$1} END{print "PR median", a[int(NR/2)], "p90", a[int(0.9*NR)], "max", a[NR]}'
```

### 0b. Smoke tests run for this plan (all PASS, 2026-08-16)

| # | test | result |
|---|---|---|
| 1 | stage entry 0 of `…_2nd1k_part1.root` with `caf_offset_mode=product` | `rc=0`; `frames-dnn.tar.bz2` 1.73 MB + `opflash_apa{0,1}.tar.gz`; **run 18259 / subrun 1 / event 171099, `frame_apply_at_caf` = 2205.0 ns** |
| 2 | art branch-name grep on both new files | `sbnd::timing::FrameShiftInfo_frameshift__FRAMESHIFT.` (the **default** instance — no `-fsproduct`), and `recob::Wires_sptpc2d_dnnsp_Reco1.` / `ints_sptpc2d_badmasks_Reco1.` / `doubles_sptpc2d_wienersummary_Reco1.` / `recob::OpFlashs_opflashtpc{0,1}__Reco1.` (the data defaults — no `-mc`, no product TLAs) |
| 3 | one-event harvest-ON PR run, nuecc48 evt 10550, into a scratch out_root | `rc=0`, `wall_s=9`, RSS 1.5 GB; `vertex_scoreboard.harvest = true`, `hv_cloud` 189 pts / 81 vertex rows, `hv_global` 7 rows; log sentinel `dl_vtx_harvest on: cloud 189 pts (81 vertex rows)` |
| 4 | the pr/79 §10 ON==OFF gate in miniature | harvest-ON dump with all `hv_*` + `harvest` keys stripped is **JSON-canonically identical** to `work-nuecc48-pr87ion3`'s harvest-OFF dump for the same event, `main_vertex` included |

Reproduce 3+4:

```bash
SP=/home/xqian/tmp/pr82-smoke
PR_JOBS=1 PR_EXTRA_STAGES=pr_display SBND_DL_VTX_HARVEST=true SBND_DL_VTX_MIN_ACCEPT=10.0 \
  ./run_pr_chain_batch.sh work-nuecc48-cb0805 $SP/work-nuecc48-harvsmoke data 10550
# then strip hv_*/harvest from the new dump and compare to work-nuecc48-pr87ion3's
```

Test 4 is the load-bearing one: it proves that a harvest arm and the current
production arm have **identical vertex geometry**, which is what lets §4's
carry-forward run against the harvest arms without re-deriving §1 (see §4.0).

---

## 1. Where we actually stand

### 1.1 The old labels re-anchor onto the new PR almost perfectly

The 473 labels (`vtxscan-prod0813` 47 + `-ncpi0` 19 + `-mcp1k` 407) were taken on
the `prod0813` epoch at `dl_vtx_min_accept_score = 4.0`. Their `source` calib
dumps were deleted on 2026-08-16. But every one of those 473 events **already has
a fresh dump** on the surviving current-production arms
`work-{nuecc48,ncpi0,mcp1k}-pr87ion3` (toolkit `771f075b`, pr83/85/86 on).

`vtx_io.load_labels()` defines a label's truth as **the rank-1 pick's (x, y, z)**
(`vtx_rules/vtx_io.py:79`), and `correct()` as Euclidean distance ≤ 1 cm
(`:31`, `:88`). Neither definition references an arm. So the truth is portable.
Taking each label's `truth` and finding the nearest vertex in the *current* arm's
dump:

| bucket | n | % |
|---|---:|---:|
| ≤ 1 cm — carries forward | **449** | 94.9 |
| 1–3 cm — needs eyes | 15 | 3.2 |
| > 3 cm — broken, must re-scan | 9 | 1.9 |

median nearest-vertex distance **0.0000 cm**; **380/473 events have a
bit-identical vertex fit**; `vertex_id` is preserved on 449 and moved on 21
(exactly the pr/80 §11 F2 warning — `improve_vertex` refits after the choice).

Only **3 of 481** labels have a manual rank-1 pick (`vertex_id: null`), so
geometric carry-forward is defined for essentially the whole set. `confidence` is
null on 481/483 — the production scan carries no confidence tier at all, which is
why §5's auto-accept tiering cannot be back-fitted to it.

### 1.2 …but production still picks the wrong one on a quarter of them

Scoring the current arm's `main_vertex` against the same 473 truths:

```
prod0813 @ min_accept=4   322/473
pr87ion3 @ min_accept=10  358/473      52 fixed, 16 regressed, 99 wrong in both
```

**Two caveats, or this number misleads.**

1. The baseline 322 is **prod0813 at `min_accept=4`**, not the deployed `-ma10`
   arm — `prod_x/y/z` in `data/harv473/manifest.tsv` was inherited verbatim from
   `full473` by `build_dataset.py --inherit-manifest`. pr/79 measured the 4→10
   knob flip **alone** at +36/473 live. So the +36 net here is *the knob flip and
   pr83/85/86 combined*; the incremental effect of the PR advance is roughly
   **+16 net**, not +36.
2. The 358 coincides with pr/79's live `-ma10` headline. Per-event agreement with
   that arm is **not verifiable** — the arm is deleted. Treat the equality as a
   headline coincidence; do not claim the two arms were compared.

**Reconciling 449 and 358 — this is the round's headroom statement.** 449/473 is
*the correct vertex still exists in the PR graph*; 358/473 is *production picks
it*. The **91-event gap between available and chosen, measured on the current
binary**, is the ceiling for anything a better selector or a better net can win.
It is a cleaner statement than pr/79's k=20 selector gap because it is measured
after pr83/85/86 rather than before.

### 1.3 Three things pr/81 assumed that are no longer true

**(a) The DL features do not exist at the current binary — for any sample.**
`dl_vtx_harvest` defaults OFF (`toolkit/cfg/pgrapher/experiment/sbnd/
wct-pr-perevt.jsonnet:1617`, `clus.jsonnet:1437,2940`, key-suppressed so OFF is
byte-identical), and the surviving `work-*-pr87ion3` arms were run without it —
verified: zero `hv_*` keys in any of their 511 dumps. The harvest arms that did
have them (`work-*-ma10k20-harv2`) were retired on 2026-08-16 with the calib
class explicitly *dropped, not archived*.

What survives is the distilled snapshot `dl_vtx_training/data/` (20 MB):
`harv473` (473 × live `hv_cloud` xyz+q, truth, prod vertex), `full473` (rebuilt
clouds), `harv473-cands` / `full473-cands` (k=20 / k=5 candidate sidecars),
`k20feats-harv-20260815` (16-dim penultimate features). **Not** surviving:
`hv_global`, `hv_single_candidate_ids`, the per-row `hv_n_proton_in/out`,
`hv_z_prior`, `hv_conflicts`, `hv_reduced_chi2`, `hv_trad_main_vertex_id`, and
the recorded `voxels[]`/`rows[]` — none are in any npz, and `calib_guard.py`
needs them.

**(b) Every fine-tune checkpoint is gone.** `thin_dlruns_20260816.py` removed
2195 `.pth` files (58.8 GiB); `find dl_vtx_training -name '*.pth'` returns
nothing. Only `wire-cell-data/sbnd/scn_vtx/sbnd-vtx-ft2u-full473-e10-CP9.pth`
(the **rejected** ft2u arm) survives, referenced by no config. Every metrics
TSV/log/`config.json` was kept, so every number docs pr/77–81 quote is still
backed.

**Consequence: pr/81 Step 2 is dead as written.** The out-of-sample retest of
ft2u / hft1 / hr3-deploy needs both the deleted checkpoints and the deleted calib
dumps. Substitutes, in order of what they buy:

- **ft2u replay survives** — its CP9 is on disk, and once §3 rebuilds harvest
  arms `calib_guard.py` can replay it on the *new* labels. That is the single
  surviving out-of-sample methodology check, and it is worth running: if the
  guard's −57 verdict does not reproduce out-of-sample, the guard was overfit and
  everything downstream of it is in question (pr/81's own stop-and-report rule).
- **hft1 / hr3 cannot be replayed** — accept the loss and re-run the hr3 recipe
  from scratch on the new pool (§6 Step 4 does that anyway). Note in the round's
  results that the pre-registered Step-2 confirmation was only partially
  executable, and why.

**(c) `baselines.py` resolves nothing.** `deployed_dump_path()`
(`vtx_rules/baselines.py:38-49`) rewrites `-prod0813` → `-ma10`; those arms are
gone, so it returns `None` for all 483. B0 is dead, B2/B3/B3b are dead (they read
a dump), B1 still works (carried inside the label). Fixed in §4.4.

### 1.4 Free headroom already on disk

`work-mcp1k-pr87ion3` holds **445** calib dumps against **407** labels — **38
mcp1k events have a current-PR dump and have never been scanned.** They cost
nothing to add to the scan pool.

---

## 2. Task A — process the new 2000 data events

```
/nfs/data/1/yuhw/production-prep/add-frameshift-data-2nd-2k-2026-08-15/
  data_MCP2025C_reco1_frameshift_2nd1k_part1.root   3.88 GiB   1000 entries
  data_MCP2025C_reco1_frameshift_2nd1k_part2.root   3.90 GiB   1000 entries
```

Same production family as the existing `mcp1k` sample (`…_frameshift_first1000ev
.root`), just further entries. Proposed sample name **`mcp2k`** — collides with
none of the 18 surviving `work-*` roots.

### 2.1 The blocking pre-flight: event-ID uniqueness

`work/evt<ID>` keys on the bare event number, and event numbers are unique only
within `(run, subrun)`. `stage_all.sh:4-8` exists as a barrier for exactly this,
and `PROVENANCE.txt` records that the first 1k was collision-free **by luck, not
by construction**. This sample is riskier: it spans the same two runs (18255,
18259) as the first 1k, so a collision would be across samples as well as within.

A `FileIndex`-derived forecast says all 2000 triples are distinct and overlap
with the first 1k is exactly 0. **Do not gate on that forecast** — its
run/subrun column is unvalidated (the prose forecast for entry 0 of part1 said
run 18255; the staged metadata says run **18259**, event 171099, and the event ID
alone was right). The event ID for entry 0 is confirmed absent from the first-1k
map.

The gate is the staged output, not a forecast:

```bash
# after staging, rebuild the manifest from the authoritative per-event metadata
for i in $(seq 0 1999); do
  tar xzOf $STAGE/e$i/opflash_apa0.tar.gz --wildcards '*_metadata.json' \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["run"],d["subrun"],d["event"],d["frame_apply_at_caf"])'
done > entry_event_map.tsv
```

Then assert, and **stop if any fails**:

1. 2000 distinct `(run, subrun, event)` triples;
2. 2000 distinct bare event IDs (the stronger condition `work/evt<ID>` needs);
3. zero intersection of those IDs with `staged-mcp2025c-1000evt/entry_event_map.tsv`;
4. `frame_apply_at_caf` values are **not** all ≡ 0 mod 256 — that pattern is the
   `-caf auto` fallback signature and is how yuhw's bad ncpi0 extraction was
   caught (doc 71 §3). The first 1k spans 245–3322 ns; the smoke event is 2205.0.

**If assertion 2 fails**, the fix is not to rename events: it is to key the work
dirs by `run_subrun_event` for this sample, which changes every downstream path
and every label's join key. That is a stop-and-ask, not an in-flight decision.

### 2.2 The four stages, with measured cost

| stage | command | wall/event (measured on mcp1k) | disk (2000 evt) |
|---|---|---|---|
| stage | `wct-reco1-dump.jsonnet --tla-str entry=<i> --tla-str caf_offset_mode=product` — **one dir per entry**, not one combined archive | ~2 min per 1000 at 16-way | **2.6 GB** |
| imaging | `SBND_MAX_JOBS=1 SBND_WORK_ROOT=$PWD/work-img-mcp2k SBND_INPUT_DIR=$S/e{} ./run_img_evt.sh data 1` under `xargs -P 32` | median **7.0 s** | **14.6 GB** |
| Q/L + nusel tail | `ROOT=$PWD/work-mcp2k-<qltag> IMGBASE=$PWD/work-img-mcp2k ./run_full1k_nusel_2k.sh 2000 32` | median **25 s**, RSS 423 MB | **6.8–8.0 GB** |
| PR + harvest + display | `PR_JOBS=32 PR_EXTRA_STAGES=pr_display SBND_DL_VTX_HARVEST=true ./run_pr_chain_batch.sh work-mcp2k-<qltag> work-mcp2k-harv3 data` | median **20 s**, RSS 1.17 GB | **6.2 GB** |

**≈ 2 h wall at 32-way, ≈ 30–35 GB.** Against 535 GB free on `/nfs/data/1` disk
is not the constraint — but this roughly **doubles `sbnd_xin` from its
just-retired 23 G**, which the owner should see before it happens.

Per-entry staging is **mandatory, not stylistic**: `_runlib.sh:113` stages an
event by wildcard-extracting the whole `frames-dnn.tar.bz2`, so one 2000-event
archive makes staging quadratic (`PROVENANCE.txt:4-7`).

The three `SBND_MVGA_*` env knobs that doc pr/86 §15 used are **no longer
needed** — at `771f075b` they are SBND production defaults
(`wct-pr-perevt.jsonnet:1488,1490,1495`). A bare run is production.

### 2.3 Three script gaps to close before the campaign

| gap | file | fix |
|---|---|---|
| `STAGE` and `MAP` hardcoded to `staged-mcp2025c-1000evt` | `run_full1k_nusel.sh:58,60` | fork to `run_full1k_nusel_2k.sh`, or make both env-overridable (`ROOT`/`IMGBASE`/`TAG`/`ENTRIES` already are) |
| `SP=` points at a dead 2026-07 session scratchpad | `staged-mcp2025c-1000evt/stage_all.sh:14`, `process_all.sh:15` | `PROVENANCE.txt:16-18` already says edit `SP`/`RUNDIR` to a durable dir first |
| assumes a single input file | `stage_all.sh` | new: flat index `e0..e999` ← part1 entry *i*, `e1000..e1999` ← part2 entry *i−1000*. This two-file mapping has no precedent in the tree — write it down in the new `PROVENANCE.txt` |

### 2.4 Retention

Only `pr_evt*/calib-pr-evt*.json` is needed for scanning (~0.6 MB median on
data; the Bokeh viewer reads nothing else — `serve_pr_display.sh:46-49`). The
harvest arm's dumps are larger (`hv_cloud` etc.; pr/79 measured 1–7 MB on opt-in
arms). Budget ~1.3–3 GB of calib for 2000 events and treat the rest of the PR arm
(`pctree`, `mabc`, `tracking-pr.root`) as retirable once the round closes.

Write a `PROVENANCE.txt` for the new staged dir in the same shape as the first
1k's: source files, extract command, run/subrun census, uniqueness argument, caf
range, and the n/n imaging + Q/L results.

---

## 3. Task B — DL features for all four samples

**The finding, plainly: no sample has live DL features at the current binary.**
The harvest arms are gone (§1.3a) and `work-*-pr87ion3` was run harvest-OFF.

**The fix is cheap and serves two tasks at once.** The imaging hubs
(`work-img-*`) and the Q/L hubs (`work-*-cb0805`) survived the retirement round,
so this is a **PR-stage-only re-run** — no re-imaging, no re-Q/L:

```bash
for s in nuecc48 ncpi0 mcp1k; do
  PR_JOBS=24 PR_EXTRA_STAGES=pr_display SBND_DL_VTX_HARVEST=true \
    ./run_pr_chain_batch.sh work-$s-cb0805 work-$s-harv3 data
done
```

At the measured 20 s/event, that is **≈ 6 min for nuecc48+ncpi0, ≈ 15 min for
mcp1k** at `PR_JOBS=24` — tens of minutes, not a campaign. `SBND_VERTEX_SCOREBOARD`
does **not** need setting: `run_pr_chain_batch.sh:184` defaults it true whenever
`SBND_DL_VTX_HARVEST` is truthy. `SBND_DL_VTX_MIN_ACCEPT` does not need setting
either — 10.0 is the production default since pr/79.

The same pass writes the calib dumps the re-scan of §4 needs, so §3 and §4 share
one production. That is the reason §4 joins against the harvest arms rather than
against `pr87ion3`.

**Gates, exactly as pr/79 §10 did them:**

1. **ON == OFF.** Harvest-ON dump with `hv_*` and `harvest` stripped must equal
   the harvest-OFF dump. **Already demonstrated for one event** (§0b test 4,
   JSON-canonically identical including `main_vertex`); run it over the full
   arms as a batch gate.
2. **Bit-exact live reproduction.** `verify_harvest.py` — voxelize `hv_cloud`
   with the pyutil math, run CP24, require the recorded top-20 voxels to match
   positions **and** scores. This is what structurally closed the ft2u
   rebuilt-cloud trap; do not skip it.
3. **Calib count == event count** before the scan opens (pr/81 P1).

Then rebuild the training snapshot:

```bash
python3 dl_vtx_training/build_dataset.py --name harv3 \
    --tags <the new carried/scanned tags> \
    --harvest-roots <tag>=work-<sample>-harv3 …
```

**Keep `data/harv473` and `data/full473`.** They are the old-epoch comparison,
and the only thing that makes "how much did pr83/85/86 move the DL input cloud"
measurable at all. One anecdote already: evt 10550's live cloud is 189 pts at the
current binary and 189 pts in `harv473` — a full census is nearly free once the
arms exist.

---

## 4. Task C — relabelling the three existing samples

The owner's question was "PR has changed, so maybe we need to redo-scan and then
label?". §1.1 answers it quantitatively: **94.9% of the labels do not need a
human**, and the ones that do are identifiable in advance.

### 4.0 Which arm — stated up front

C1 joins against the **§3 harvest arms**, not `work-*-pr87ion3`, so that labels,
dumps and DL features all come from one arm (pr/80 §11: *"Labels and dumps must
come from the same arm"*). That makes §3 a hard prerequisite of §4.

§1's numbers were measured on `pr87ion3`. They transfer only if harvest-ON and
harvest-OFF produce identical vertex geometry — which §0b test 4 demonstrates for
one event and §3 gate 1 establishes for the full arms. **Re-run the §0(b) block
against the harvest arms and quote those as the operative figures**; keep §1's as
the planning-time measurement. If the two disagree anywhere, that is a
stop-and-report finding, not a rounding difference.

### 4.1 C1 — carry-forward (scripted, no human)

New `vtx_rules/carry_labels.py`. For each of the 473 labels: read `truth`, join
to the harvest arm's `vertices[]` by nearest-neighbour, bucket at the existing
`TOL`/`TOL_LOOSE` (1 cm / 3 cm), and for the ≤1 cm bucket write a label into a
**fresh tag** `vtxscan-harv3-{nuecc48,ncpi0,mcp1k}` (M13 — never into an existing
tag; the viewer refuses anyway without an explicit `--scan-tag`) carrying:

- `truth` verbatim (the human's answer, unchanged — this is the whole point);
- the **new** `vertex_id`, `arm`, `source`, and the current scoreboard columns;
- a new `carried_from` provenance field naming the old tag, the old arm, the old
  `vertex_id`, and the measured re-anchor distance.

The 1–3 cm and >3 cm buckets are **not** auto-carried. Nothing overwrites
`vertex_labels/vtxscan-prod0813*/` — those stay as the historical record.

Expected shape from §1.1: ~449 carried, 15 + 9 = 24 held back.

### 4.2 C2 — re-scan the deltas

The 24 non-exact events, **plus** the 38 mcp1k events that have a current dump
and no label (§1.4), through the pr/80 §11 machinery. Use
`selfscan.py prepare --only-manifest`, **not** `--dumps` — pr/80 §13 notes that
`--dumps` discards the label join key and disables `score`. Add the new tags to
`vtx_io.TAGS`.

### 4.3 C3 — validate the carry-forward

This is what turns "94.9% re-anchor" from an assumption into a measurement. Take
a stratified ~60 events **from the carried bucket** (sample × corrective ×
route), blind-scan them per pr/80 §11 step 2, and `score` against the carried
labels. The output is a carry-forward error rate. Pre-register the bar now:

- **≥ 95% agreement** → carry-forward accepted for the whole 449.
- **90–95%** → carry, but flag the disagreeing strata for full re-scan.
- **< 90%** → the carry-forward is not sound; fall back to a full 473 re-scan and
  say so.

Without C3 the round has no defence against a systematic carry error that
correlates with exactly the events the PR change touched.

### 4.4 C4 — the one-line fix and what it cannot restore

`vtx_rules/baselines.py:38` — point the arm rewrite at the harvest arms instead
of `-ma10`, and keep `check_deployed()` reading the operating point from
`vertex_scoreboard.dl_min_accept_score` rather than a directory name (it already
does). **B0's old 358/473 gate is not reproducible** — the `-ma10` arms are gone
and were archived without dumps. State that in the docstring so the next reader
does not spend an hour on it.

---

## 5. Task D — scanning the new 2000

pr/80 §11 verbatim. `prepare --dumps` (there is no truth, so `review` replaces
`score` — running `score` on unlabelled events is an error, not an option), N
blind subagent scanners each handed `vtx_rules/scan_prompt.md` unchanged, then
`review` buckets into `REVIEW FIRST` / `REVIEW` / `auto-accept` /
`no candidates`. The owner adjudicates `REVIEW FIRST` then `REVIEW`.

**Expected pool.** Dump yield on data is ~44.5% (445/1000 on mcp1k — only events
where PR selected a main cluster produce a dump at all), so expect **~890
scannable of the 2000**. Total label pool after this round: ~449 carried + ~24
re-scanned + 38 new-old + ~890 new ≈ **O(1400)**, which is the tripling pr/81
sized its gates for.

**Auto-accept must be calibrated on this sample before it is used at scale.**
pr/80's 95.5% precision at 36.7% coverage comes from 60 held-out events of a
*different* reconstruction epoch, and §11 says so explicitly. Pre-registered
prerequisite: the owner hand-scans **30–60** of the new events into a fresh tag,
`score` measures auto-accept precision on this sample, and only then does
auto-accept apply to the rest. If it lands below ~90%, every event goes to review.

**Throughput priors**, derived from the existing labels' `saved_utc` stamps (this
is a measurement of the owner's own past scanning, not a doc quote):

| tag | n | elapsed | median gap between saves |
|---|---:|---:|---:|
| `vtxscan-prod0813` (nueCC48) | 47 | 3.6 h | 12 s |
| `vtxscan-prod0813-ncpi0` | 19 | 1.0 h | 22 s |
| `vtxscan-prod0813-mcp1k` | 407 | **9.0 h** | 30 s |

≈ 30 s/event hands-on on a data sample; 407 events took ~9 h elapsed. A full
890-event hand scan is therefore ~20 h of the owner's time — which is the reason
the AI-fleet + adjudication split is the plan rather than a preference.

**Ports.** 5017 and 5018 are occupied right now by live `pr_display_viewer.py`
instances on `work-r{1qlmc,2mc}-prod0813`. Do not stop them; serve the new scan
on a free port (5019+, noting `overclustering_display` also defaults to 5018).

---

## 6. Training round 3

pr/81 Steps 1/3/4/5, carried forward with the composition made explicit.

**Step 1 — unbiased production measurement.** Score production on the *fresh*
labels alone. The existing 358/473 is measured on the events that steered every
knob decision since pr/33; the new sample has steered nothing. Report per-sample.
No decision hangs on it; it is the reference for everything after.

**Step 3 — fresh lockbox, before any new label is read.** Reserve 20–25% of the
new labels, stratified by sample × corrective × **data/MC**, seed recorded in the
manifest. This restores the three-tier gate (train/val → guard screen → lockbox →
live A/B) that has been running without a lockbox since pr/78. The old lockbox is
spent.

**P3 composition, stated because it changes the screen.** After this round the
pool is ~890 new data + 407 old data + 66 MC (47 nueCC + 19 NCpi0) — i.e.
**overwhelmingly detector data**. Every split stratifies by data/MC, and the
**data-only guard replay is the primary screen**: a net that helps MC and hurts
data must not be invisible behind an aggregate number.

**Step 4 — the dose question.** The corrective pool grows ~115 → ~350+. This is a
rerun of a committed pipeline, not a rebuild:

```
build_dataset.py --harvest-roots <new arms> --inherit-manifest <lockbox col>
train.py  --cand-softmax 1.0 --scale-anchor 1.0 --dense-weight 0.1 --lr0 1e-5   # the hr3 recipe
bash hr_guardsel.sh <arm>          # guard-in-loop checkpoint selection
calib_guard.py <deploy CP>         # full-manifest screen
# live A/B only if guard-positive; flip owner-gated as always
```

Start from hr3 — the only arm ever to pass the OOF gate. Known bias: fold-max
selection inflates OOF sums by ~+1, so marginal gate passes are noise until the
full-manifest screen.

**Step 5 — the two "real signal, can't use it" findings.** Rerun `cand_head.py`
unchanged at O(1400) (the 16-dim exact features overfit at O(362); triple the
pool may flip it) — a formulation clearing the pre-registered **≥ +5 anchored**
bar is a STOP-and-present-to-owner, since it is a new C++ inference path. And
re-run the `min_accept` sweep on the fresh labels, where ±2-point effects become
measurable.

---

## 7. What this costs

| item | cost |
|---|---|
| processing 2000 new events | ~2 h wall at 32-way, ~30–35 GB (doubles `sbnd_xin` from 23 G) |
| harvest arms for the 3 existing samples | ~25 min wall, ~7 GB (PR stage only) |
| C1 carry-forward | minutes, scripted |
| C2 delta re-scan (62 events) | ~30 min of scanning |
| C3 carry validation (60 events) | AI fleet + ~30 min owner adjudication |
| D new-sample scan (~890 events) | AI fleet; owner adjudicates the REVIEW piles + 30–60 calibration events |
| training round 3 | as pr/81 — CPU folds, no GPU (pr/78: GPU fold-procs 5× slower) |

## 8. What could go wrong

- **Event-ID collision** in the new sample or against the first 1k. Gated in
  §2.1; if it fires, stop — the fix changes every downstream path.
- **The carry-forward is systematically wrong on exactly the events pr83/85/86
  touched.** This is the failure mode C3 exists to catch. It is plausible: the
  9 "broken" events are not a random sample.
- **The auto-accept prior does not transfer** to a new reconstruction epoch —
  pr/80 §11 says so in as many words. Gated in §5.
- **The guard was overfit to the old 473.** If the ft2u replay does not reproduce
  its −57 out-of-sample, `calib_guard.py` cannot gate round 3 and the round stops
  to report (pr/81 Step 2's own stop condition).
- **`libWireCellSBNDReco1.so` is older than HEAD.** The plugin dates from
  2026-08-03; the last known-good use was 2026-08-05. Smoke test 1 exercised it
  successfully at HEAD, so this is noted rather than blocking — but if staging
  fails at scale, rebuild it first (recipe in `run_reco1_dump.sh:44-51`).
- **Disk.** 535 G free is ample, but the round leaves `sbnd_xin` at ~55 G. Plan
  the retirement of the PR arms' non-calib layer as part of closing the round,
  not as an afterthought.

## 9. Pre-registered gates

Written down now, before any new label is read.

1. §2.1 assertions 1–4 all pass before imaging starts.
2. §3 gates 1–3 pass before any dump enters a scan or a dataset.
3. §4.0 re-measurement on the harvest arms agrees with §1.1; disagreement is
   stop-and-report.
4. §4.3 carry-forward validation ≥ 95% → carry all; 90–95% → carry with flagged
   strata; < 90% → full re-scan.
5. §5 auto-accept calibration on ≥ 30 owner-scanned new events before
   auto-accept is applied at scale.
6. §6 lockbox reserved before the first new label enters a training pool; never
   read until a final candidate exists.
7. A learned-chooser formulation must clear **≥ +5 anchored** to be presented.
8. CP24 anchor stays exact after every `calib_guard.py` change.

## 10. Standing rules

- No production flip, no `wire-cell-data` commit, no Bee upload without the
  owner.
- `dl_vtx_training/runs/` and `data/` are never committed; scripts and docs only;
  `git add -f` for `*.sh`.
- M11 — the new sample is imaged and Q/L-matched by us, never fed from someone
  else's products.
- M13 — every new artifact gets a fresh tag/dir. `vertex_labels/vtxscan-prod0813*`
  is a historical record and is not modified by this round.
- M16 — scratch under `/home/xqian/tmp/`, never `/tmp`.
- Do not stop or restart the bokeh viewers on 5017/5018.

## 11. Files

| file | what |
|---|---|
| `input_files_reco1/staged-mcp2025c-2nd-2000evt/` | new staged sample (§2), with its own `PROVENANCE.txt` and `entry_event_map.tsv` |
| `work-img-mcp2k`, `work-mcp2k-<qltag>`, `work-mcp2k-harv3` | the new sample's three roots |
| `work-{nuecc48,ncpi0,mcp1k}-harv3` | harvest arms of the existing samples (§3) |
| `vtx_rules/carry_labels.py` | **new** — the §4.1 carry-forward |
| `vertex_labels/vtxscan-harv3-*` | **new tags** — carried + re-scanned labels |
| `dl_vtx_training/data/harv3` | the round-3 training snapshot |
| `dl_vtx_training/data/{harv473,full473,*-cands,k20feats*}` | **kept** — the old-epoch comparison |
