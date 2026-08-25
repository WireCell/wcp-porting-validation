# doc pr/82 — DL-vertex round 3: the 2000-event data sample, the harvest gap, and relabelling the three existing samples

**Status: §§2–4 EXECUTED 2026-08-16 — see §12 for what actually ran and what it
measured.** §§1–11 below are the plan as written before execution and are left
unedited except where §12 supersedes them; every pre-registered gate in §9 was
run and its verdict is recorded. §5 (the ~890-event scan of the new sample) and
§6 (training round 3) are **not** executed — the owner scoped this round to
processing, harvest, relabelling and the review display.

No production flip, no knob change, no C++ or jsonnet edit was made.

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

# (d) the harvest census behind sec 1.3(a):  511 dumps, 0 with harvest fields
python3 - <<'EOF'
import glob, json
for a in ("work-nuecc48-pr87ion3", "work-ncpi0-pr87ion3", "work-mcp1k-pr87ion3"):
    fs = sorted(glob.glob(f"{a}/pr_evt*/calib-pr-evt*.json"))
    h = sum(any(k.startswith("hv_") or k == "harvest"
                for k in (json.load(open(f)).get("vertex_scoreboard") or {})) for f in fs)
    print(a, len(fs), "dumps,", h, "with harvest fields")
EOF
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

> **PARTLY SUPERSEDED by §12.6.** The 358 and the 91-event gap below are measured
> against the **stale** prod0813-epoch truth. After the owner re-scanned the 24
> events the carry-forward declined, the current-epoch figures are
> **372/473 (78.6%)** and a **98-event** selector gap over **470** available.
> Use 358 only when comparing to pr/79; use 372 for any statement about the
> current arm.

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
byte-identical), and the surviving `work-*-pr87ion3` arms were run without it.
**Census (§0 block d): 511 dumps — 47 nuecc48 + 19 ncpi0 + 445 mcp1k — all 511
carry a `vertex_scoreboard`, and 0 carry any `hv_*` or `harvest` key.** The
harvest arms that did
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

**The gate deliberately runs *after* staging, not before.** The authoritative RSE
comes from the staged per-event metadata, so the 2.6 GB / ~4 min of staging is
spent before the assertions can fire. That is the first-1k precedent
(`stage_all.sh:4-8` calls itself "deliberately a barrier before imaging") and it
is cheap; the ordering is intentional, not an oversight. Imaging — the expensive
stage — is what the gate actually protects.

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

**Ports.** ~~5017 and 5018 are occupied right now by live `pr_display_viewer.py`
instances on `work-r{1qlmc,2mc}-prod0813`. Do not stop them; serve the new scan
on a free port (5019+, noting `overclustering_display` also defaults to 5018).~~

**SUPERSEDED 2026-08-16 by owner instruction (§12.4):** the execution round was
told to serve the review pile on **5017 specifically**, so the
`vtxscan-prod0813-mc` viewer that held it (pid 2871669, 8 labels saved, last
write 2026-08-15 10:07) was stopped. Its 8 labels are on disk and untouched; the
verbatim relaunch command is recorded in §12.4 so that scan can resume at any
time. **5018 was not touched.**

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
- ~~Do not stop or restart the bokeh viewers on 5017/5018.~~ **Amended
  2026-08-16:** 5017 was taken over on explicit owner instruction (§12.4); the
  rule still stands for **5018**, which is untouched and still serving
  `uitest-dqdx` on `work-r1qlmc-prod0813`.

## 11. Files

| file | what |
|---|---|
| `input_files_reco1/staged-mcp2025c-2nd-2000evt/` | new staged sample (§2), with its own `PROVENANCE.txt` and `entry_event_map.tsv` |
| `work-img-mcp2k`, `work-mcp2k-<qltag>`, `work-mcp2k-harv3` | the new sample's three roots |
| `work-{nuecc48,ncpi0,mcp1k}-harv3` | harvest arms of the existing samples (§3) |
| `vtx_rules/carry_labels.py` | **new** — the §4.1 carry-forward |
| `vertex_labels/vtxscan-harv3-*` | **new tags** — carried + re-scanned labels |
| `dl_vtx_training/data/harv3` | the round-3 training snapshot — **NOT built this round**; it belongs to §6, which the owner scoped out |
| `dl_vtx_training/data/{harv473,full473,*-cands,k20feats*}` | **kept** — the old-epoch comparison |

---

# 12. Execution log — 2026-08-16

Everything below actually ran. Owner scope for this round: process the new 2000
so its hand-scan and DL-feature products exist, produce the same for the three
existing samples, relabel those three from the existing scans, and serve the
events needing owner eyes on port 5017. §5's scan of the new sample and §6's
training round 3 are deliberately **not** in this round.

Owner authorised **32-way** parallelism explicitly, above CLAUDE.md §2's ~6-way
routine cap. M5's hazard is *truncated output*, not slowness, so every stage is
followed by a count-and-rc check rather than a trusted exit code (§12.5). Peak
load observed was **34.5** on a 64-core box; nothing was truncated.

## 12.1 Task B — the harvest arms (§3) — DONE

```bash
for s in nuecc48 ncpi0 mcp1k; do
  PR_JOBS=32 PR_EXTRA_STAGES=pr_display SBND_DL_VTX_HARVEST=true \
    ./run_pr_chain_batch.sh work-$s-cb0805 work-$s-harv3 data
done
```

`SBND_DL_VTX_MIN_ACCEPT` and `SBND_DL_VTX_TOP_K` are deliberately **not** set.
pr/79 §10a's harvest recipe passed `TOP_K=20`, and copying that would have been
a mistake here: a non-default `top_k` changes the rerank candidate set, so the
arm would no longer be production and gate 1 below could not run at all. Bare
defaults (10.0 / 5) are production since pr/79, and `run_pr_chain_batch.sh:184`
auto-defaults the scoreboard.

| arm | events | rc=0 | calib dumps | vs `pr87ion3` | wall | size |
|---|---:|---:|---:|---|---:|---:|
| `work-nuecc48-harv3` | 48 | 48 | **47** | 47 ✓ | 1m15s | 261 MB |
| `work-ncpi0-harv3` | 19 | 19 | **19** | 19 ✓ | 29s | 97 MB |
| `work-mcp1k-harv3` | 1000 | 1000 | **445** | 445 ✓ | 10m37s | 3.1 GB |

Total 12m21s for 1067 events, 3.5 GB. (The 48→47 and 1000→445 shortfalls are
not losses: only events where PR selects a main cluster emit a calib dump, and
the counts match the harvest-OFF reference arm exactly.)

### Gate 1 — ON == OFF: **PASS, 511/511**

New script `scripts/analysis/pr82/onoff_gate.py` (none existed —
`scripts/retire/gate_cmp_arms.py` deliberately *excludes* the calib JSON, and
pr/79 §10c / §0b test 4 were done by hand on a handful of events).

```
work-nuecc48-harv3 vs -pr87ion3 : common  47, identical-after-strip  47, DIFF 0
work-ncpi0-harv3   vs -pr87ion3 : common  19, identical-after-strip  19, DIFF 0
work-mcp1k-harv3   vs -pr87ion3 : common 445, identical-after-strip 445, DIFF 0
```

The strip is a **prefix rule** (`hv_*`, plus the literal `harvest`) applied
recursively, not a hand-listed key set, so a future harvest field cannot make
this gate fail for the wrong reason. Comparison is on canonical JSON, never file
bytes (M2). The gate also asserts each ON dump really carries `harvest: true` —
without that an inert knob would make it pass vacuously (the M1 shape).

**This is the load-bearing result of the round.** It establishes that
`dl_vtx_harvest` is recording-only at toolkit `771f075b`, which is what licenses
§4's carry-forward to join against the harvest arms.

### Gate 3 — bit-exact live reproduction (`verify_harvest.py`): **PASS**

| arm | dumps | verifiable | EXACT | not applicable |
|---|---:|---:|---:|---:|
| nuecc48 | 47 | 47 | **47** | 0 |
| ncpi0 | 19 | 19 | **19** | 0 |
| mcp1k | 445 | 429 | **429** | 16 |
| **total** | **511** | **495** | **495** | **16** |

(445 mcp1k dumps = 429 verifiable + 16 `dl-not-run`.)

Every voxel position **and** score reproduces from the recorded `hv_cloud`
through CP24. The 16 mcp1k exclusions are **not** failures: they carry
`dl_ran: false`, `route: dl-not-run`, zero rows, zero voxels and zero cloud
points — the DL vertex finder never ran, so there is nothing to reproduce.
`verify_harvest.py` has no case for that and reports it as `no hv keys in rows`;
worth a follow-up patch, but the harvest recipe is not implicated.

**None of the 16 is a labelled event.** All 473 labelled events, and all 449
carried ones, have a non-empty `hv_cloud` — checked explicitly, because a
labelled event with no cloud would silently drop a training row.

### Gate 4 — the §4.0 re-measurement on the harvest arms: **PASS, exact**

Re-running §0 block (b) against `-harv3` instead of `-pr87ion3`:

```
Counter({'exact': 449, 'loose': 15, 'broken': 9})  zero: 380  median 0.0
vertex_id: same 449, moved 21
production correct@1cm: 358 / 473
```

Identical in every digit to the planning-time measurement. §1's numbers
therefore stand as the operative figures; there is no second set to track.

## 12.2 Task C — relabelling (§4) — DONE

New `vtx_rules/carry_labels.py`. **449 of 473 labels carried**, into three fresh
tags; the 24 that did not are §12.4's review pile.

| tag | carried |
|---|---:|
| `vertex_labels/vtxscan-harv3-nuecc48` | 42 |
| `vertex_labels/vtxscan-harv3-ncpi0` | 19 |
| `vertex_labels/vtxscan-harv3-mcp1k` | 388 |

Of the 449: **380 bit-identical** vertex fits, **7 changed `vertex_id`** while
keeping the point. `vertex_labels/vtxscan-prod0813*/` is untouched (mtimes still
2026-08-14/15) and remains the historical record.

**Why the join is on position, not `vertex_id`.** Both directions were measured
on this data and the id-join is wrong both ways:

- 7 events keep the point and renumber the id — an id-join calls those *lost*
  and sends a human something with nothing to decide.
- **evt283040 keeps id 2000 while that id now names a point 117.5 cm away** — an
  id-join calls that *clean* and silently carries the label to the wrong place.
  That single event is the whole case. (It is in the review pile anyway: its
  nearest vertex is 1.34 cm from the click, past `TOL`.)

Position it is, with the id change recorded as provenance so an id-audit stays
possible. Every carried label gains a `carried_from` block: old tag, old arm,
old path, old `vertex_id`, the original `truth`, the re-anchor distance, and
`vertex_id_changed`.

### The one substantive finding: 63 carried picks have no scoreboard row

The original labels carried the board's `dl_score` / `trad_score` /
`dl_winner` / … columns on `picks[0]` for **470 of 473** events. Carried onto the
new arm the equivalent is **386 of 449**, and chasing the gap turned up
something real rather than a bug:

| how the row was found | n |
|---|---:|
| the final vertex is itself a scored row | 372 |
| the row the *original* label was built from (`improve_vertex` renumbered) | 5 |
| a row within 1 cm of the final position | 9 |
| **no row at all — the reranker never scored this vertex** | **63** |

**55 of those 63 are in the MAIN cluster.** So on 14% of the carried set the
human's answer is a vertex the current reranker did not even score — an
*admission* gap, the same class pr/78 §3 identified as `dl_vtx_top_k`-limited,
now measured post-pr83/85/86 and on the main cluster specifically.

**Is this just the selector gap seen twice?** Checked rather than assumed,
because the obvious reading — "unadmitted ⇒ production misses" — would make the
finding a restatement of §1.2 rather than a new target. Crossing `row_join`
against whether production picks the truth, over the 449 carried:

| | production right | production wrong |
|---|---:|---:|
| has a scored row (386) | 323 | 63 |
| **no row (63)** | **35** | **28** |

`P(production wrong | no-row) = 44.4%` vs `P(production wrong | has-row) =
16.3%` — a 2.7× enrichment, so the two gaps are **correlated but not the same
gap**, and both halves of that matter:

- **35 of the 63 unadmitted events, production still gets right.** Not being
  scored by the reranker is therefore not fatal — the traditional/anchored
  routes recover more than half of them. Admission is a real handicap, not a
  guaranteed loss.
- **Of the 91 production misses on the carried set, only 28 (31%) are
  admission failures; 63 (69%) are on vertices the reranker DID score and
  ranked below its winner.** So the selector gap remains the larger target,
  and the earlier framing of admission as "a better-defined target than the
  91-event gap" would have been wrong — it is a distinct and smaller one.

Worth pursuing as a separate lever (a `dl_vtx_top_k` widening changes only the
28), not as a replacement for selector work.

`carry_labels.py` records which of the four routes each label used
(`carried_from.row_join`) and leaves the `dl_*`/`trad_*` fields **absent** for
the 63 rather than writing zeros — a fabricated zero would turn an admission gap
into apparent data.

### §4.4 — `baselines.py` repaired

`deployed_dump_path()` now rewrites `-prod0813` → `-harv3` (was `-ma10`, deleted
2026-08-16 with the calib class dropped, so it had been returning `None` for all
483). Verified working:

```
half=all   labels=481   with a deployed arm=473
deployed operating points seen (min_accept, top_k): [(10.0, 5)]
B0 deployed main_vertex     answered 472   correct@1cm 358  75.8%
```

The operating point printing as exactly `(10.0, 5)` is the independent
confirmation that `-harv3` is the production point. **B0's 358 is a new
measurement of the current arm, not a reproduction of pr/79's 358 on `-ma10`** —
that arm's dumps are gone. The docstring says so, because the coincidence is
otherwise an invitation to an hour of confusion.

### Tag registration — deliberately not done

The `vtxscan-harv3-*` tags are **not** added to `vtx_io.TAGS`; a new
`vtx_io.TAGS_HARV3` holds them and callers pass it explicitly. The carried
labels cover the *same events* as the prod0813 tags, so a default holding both
would hand every unfiltered consumer (`baselines.py`, `selfscan.py score`,
`build_dataset.py`) ~922 labels with duplicate event keys and quietly wrong
denominators — with no error anywhere.

## 12.3 Task A — the new 2000 events (§2)

Staged dir `input_files_reco1/staged-mcp2025c-2nd-2000evt/`, sample `mcp2k`.

**Staging: 2000/2000 OK, 4m11s at 32-way**, via new `stage_all_2k.sh`. Flat
index `e0..e999` ← part1 entry *i*, `e1000..e1999` ← part2 entry *i−1000*, held
in one place in the script and written into `PROVENANCE.txt`.

### The uniqueness gate (§2.1): **PASS, all four assertions**

```
map rows: 2000 (expected 2000)
A1 distinct (run,subrun,event): 2000 / 2000  OK
A2 distinct bare event ids     : 2000 / 2000  OK
A3 overlap with first 1k       : 0            OK
A4 caf_ns range 245..3341, all==0 mod 256: False  OK
run/subrun census: [((18255,1), 1450), ((18259,1), 550)]
```

The caf range 245–3341 ns sits right on the first 1k's 245–3322, so the
`FrameShiftInfo` product read genuinely, and A4's `-caf auto` fallback signature
is absent.

**Two defects in this doc's own §2.1 map-builder snippet, both verified against
the existing 1k and both fixed in `check_uniqueness.py`:**

1. **It crashes.** `'*_metadata.json'` matches *two* archive members
   (`opflash_tensorset_<EVT>_metadata.json` and
   `opflash_tensor_<EVT>_0_metadata.json`, the latter just `{"name":"opflash"}`),
   so two JSON documents land on one stream:
   `json.decoder.JSONDecodeError: Extra data`.
2. **Wrong schema even if it ran.** It emits space-separated `run subrun event
   caf` with no entry column and no header, but `run_full1k_nusel.sh:66` is
   `awk -F'\t' '$1==e {print $4}'`. Every lookup would miss — and a missed
   lookup is **not an error**: the worker writes `rc=90 no-event-map-row` and
   `exit 0`, so a 2000/2000 silent skip reports as a completed batch.

The corrected builder reproduces the existing 1k map byte-for-byte.

### A third instance of the same silent-skip trap, in the runner fork

`run_full1k_nusel.sh:127` dispatches workers by **name**:
`xargs ... "$SBND_DIR/run_full1k_nusel.sh" --worker {}`. A straight copy to
`run_full1k_nusel_2k.sh` leaves that line pointing at the *original*, whose
`STAGE`/`MAP` are hardcoded to the 1000-event dir — so entries 1000–1999 would
each hit `rc=90 no-event-map-row`, exit 0, and half the sample would vanish with
the batch reporting success. Fixed, with the reason in a comment at the line.

Verified rather than assumed, with a two-entry smoke before the full run:
`ENTRIES="0 1999"` — entry **1999** specifically, because it is the only cheap
test that exercises the part2 half of the map. Both resolved
(`entry=0 evt=171099`, `entry=1999 evt=175931`), both `rc=0`, neither `rc=90`.
Entry 0's event id matches §0b smoke test 1 exactly.

### The measured chain

| stage | driver | wall | result | size |
|---|---|---:|---|---:|
| stage | `stage_all_2k.sh 2000 32` | 4m11s | **2000/2000 OK** | 2.6 GB |
| uniqueness | `check_uniqueness.py --build` | s | **4/4 assertions PASS** | — |
| imaging | `xargs -P 32 … run_img_evt.sh data 1` | 11m33s | **2000/2000**, 8000/8000 npz, 0 truncated | 13 GB |
| Q/L + nusel | `run_full1k_nusel_2k.sh 2000 32` | 27m03s | **2000/2000 `rc=0`**, 2000 pctree | 17 GB |
| PR + harvest | `run_pr_chain_batch.sh … work-mcp2k-harv3 data` | see below | see below | see below |

**Imaging completeness was checked on products, not directories.** The event dir
is created before the npz are written, so a killed or short job leaves a dir
behind and a `ls -d evt*` count reads full while the sample is truncated — which
is precisely M5's shape. Observed mid-run: 1306 dirs against 1300 completed
`icluster-apa0-active.npz`. The bar is **four npz per event** (`apa{0,1}` ×
`{active,masked}`, the layout of the reference `work-img-mcp1k` hub) plus a
`-size -1k` scan for truncation. Final: 8000/8000, zero undersized.

## 12.4 The review display — port 5017, 24 events

The owner asked for the events needing their eyes on **5017 specifically**. That
port was held by an in-progress MC scan; the owner confirmed the takeover before
it happened, so §5 and §10's "do not restart the viewers on 5017/5018" is
amended in place above rather than left standing against what this round did.

**What was stopped, and how to bring it back.** pid 2871669,
`--scan-tag vtxscan-prod0813-mc` over `work-r{1qlmc,2mc}-prod0813`, 8 labels
saved, last write 2026-08-15 10:07. Those 8 labels are on disk and untouched —
only the browser session was lost. Verbatim relaunch:

```bash
cd .../sbnd_xin && ./pr_display/serve_pr_display.sh 5017 \
  --scan-tag vtxscan-prod0813-mc \
  'work-r1qlmc-prod0813/pr_evt*/calib-pr-evt*.json' \
  'work-r2mc-prod0813/pr_evt*/calib-pr-evt*.json'
```

(the scratch copies `/home/xqian/tmp/pr82/{5017-relaunch.sh,5017-previous-cmdline.txt}`
were deleted in the 2026-08-25 tmp cleanup; the block above is the record). **5018 was not touched** and still serves
`uitest-dqdx`.

**What is now on 5017:**

```bash
mapfile -t DUMPS < /home/xqian/tmp/pr82/delta-dumps.txt   # 24, worst first
./pr_display/serve_pr_display.sh 5017 --scan-tag vtxscan-harv3-delta "${DUMPS[@]}"
```

`http://localhost:5017/pr_display_viewer` (HTTP 200, clean log). Events are
ordered **worst re-anchor first**, so the 16.6 cm case is the landing page.
Picks are written to `vertex_labels/vtxscan-harv3-delta/` — a **fresh** tag.

That freshness is not cosmetic: the viewer keys label files on event id alone
(`labels-evt<ID>.json`, no arm in the name), and an explicit `--scan-tag` is
treated as consent that *disables* the M13 write guard. Passing
`--scan-tag vtxscan-prod0813` here would have overwritten the owner's original
47 labels in place, with the guard that exists to prevent exactly that disarmed
by the flag itself.

`docs/pr/82-delta-scan.md` is the companion sheet: one row per event with the
re-anchor distance, old and new `vertex_id`, the distance from the old answer to
what production currently picks, the route, and the old truth coordinates — so
the question being asked is "where is the vertex now", not a cold re-scan.

Composition: **19 mcp1k + 5 nueCC48, 0 NCpi0** — all 19 NCpi0 labels carried
exactly. 9 are beyond 3 cm, 15 between 1 and 3 cm. At the owner's measured
~30 s/event this is ~15 minutes of scanning.

## 12.5 Completeness checks (the M5 mitigation for 32-way)

Every stage was followed by a count-and-rc check before the next started, since
M5's failure mode is silent truncation rather than a non-zero exit:

| stage | expected | observed |
|---|---|---|
| harvest arms | 1067 events | 1067 `rc=0`, dumps 47/19/445 = the `pr87ion3` counts |
| staging | 2000 entries | `2000 OK`, 2000 `e<i>/` dirs |
| uniqueness | 4 assertions | 4/4 PASS, run/subrun census 1450 + 550 = 2000 |

Peak load 37 on 64 cores; no stage was run concurrently with another
wire-cell stage. Exit codes captured as `cmd > log 2>&1; echo rc=$?`, never
through a pipe (M14).


## 12.6 The owner's 24-event scan — collected, and what it changed

All 24 were scanned and saved to `vertex_labels/vtxscan-harv3-delta/` the same
day. Classifying each decision against the label it replaced:

| what the owner did | n | reading |
|---|---:|---|
| re-picked the **same physical vertex** (moved exactly as far as the re-anchor) | **15** | the carry-forward was merely conservative — `improve_vertex` pushed the fit past 1 cm and nothing else changed |
| **manual** pick within 0.6–2.2 cm of the old answer, `not_a_candidate: true` | **3** | the old answer stands but the current graph has **no vertex there at all** — a reconstruction failure, and exactly what the >3 cm bucket exists to surface |
| chose a **materially different vertex** | **6** | the old label no longer describes the event |

### This retro-justifies TOL = 1.0 and the held-back loose bucket

Four of the six different-vertex events — **166870 (1.45 cm), 283040 (1.34),
72586 (1.89), 286353 (2.48)** — had a re-anchor **inside 3 cm**. A policy of
"carry anything under 3 cm" would have written four wrong labels silently, with
nothing downstream able to detect it. Two of those are not close calls: on
72586 and 286353 the owner's answer is **299.9 cm** and **258.7 cm** from the
old one — a different cluster entirely.

evt283040 is the event §12.2 flagged for the id-join: `vertex_id` 2000 survived
while naming a point 117.5 cm away. The owner's answer is 1.69 cm from the old
one and *not* the vertex nearest to it. So both defences were load-bearing —
position-join caught what id-join would have missed, and the 1 cm hold-back
caught what position-join alone would have carried.

### Two headline numbers move

**Production accuracy was understated.** §1.2's `358/473` is measured against
*stale* truth. On the corrected current-epoch set it is **372/473 = 78.6%** —
14 events where production was already right and the old label was the thing
that had gone out of date. Any comparison against pr/79's 358 must use 358, and
any statement about the current arm must use 372; they are not the same
quantity measured twice.

**The headroom statement, restated on corrected truth:**

```
vertex AVAILABLE within 1 cm : 470 / 473    (3 are not_a_candidate -- no vertex exists)
production CHOOSES it        : 372 / 473
selector gap                 :  98
```

So §1.2's "91-event gap" becomes **98**, and the denominator is now honest: on
**3** events the correct vertex is not in the graph at all, so no selector and
no net can ever win them. 98 is the real ceiling for selection work on this
sample; of it, 28 are the admission subset of §12.2 and the remaining ~70 are
vertices the reranker scored and ranked below its winner.

**The current-epoch label set is complete: 473 labels** — 449 carried + 24
re-scanned — all on `work-*-harv3`, all with a live `hv_cloud`, spread
`vtxscan-harv3-{nuecc48 42+5, ncpi0 19, mcp1k 388+19, delta 24}`. Pass them as
`vtx_io.TAGS_HARV3`, never mixed with the `vtxscan-prod0813*` originals.


## 12.7 Task A completed — `work-mcp2k-harv3`

| stage | wall | result | size |
|---|---:|---|---:|
| PR + harvest + display | 21m10s | **2000/2000 run, 1999 `rc=0`, 1 `rc=139`** | 6.0 GB |

**879 calib dumps** (43.95% yield, against the 44.5% mcp1k prior — only events
where PR selects a main cluster emit one). Every dump carries `harvest: true` at
the production operating point (`min_accept 10.0`, `top_k 5`); the first one
checked has a 340-point `hv_cloud`.

**Zero `PR_TIMEOUT` (rc=124).** Worth stating explicitly because
`run_pr_chain_batch.sh:1180-1183` records an MCP2025C event that once spun
8h17m and stalled a whole 1000-event batch; the 3600 s cap was never reached
here, so the 879 is a clean yield number and not "yield minus timeouts".

**Whole-sample totals: 2.6 GB staged + 13 GB imaging + 17 GB Q/L + 6.0 GB PR =
~38.6 GB**, 64 minutes of wall clock across four stages at 32-way.

### The one failure: evt54629 segfaults, deterministically, in production code

`rc=139` (SIGSEGV), zero-byte `mabc-pr.zip` and `pctree-pr-evt54629.tar.gz`,
29 s in, 2.3 GB RSS. **Re-running the single event reproduces it exactly** — not
a parallelism artifact, not an M5 truncation.

```
#6  D3Vector<double>::x ()                       util/inc/WireCellUtil/D3Vector.h:108
#8  TrackFitting::organize_ps_path (...)         clus/src/TrackFitting.cxx:2071
#9  TrackFitting::do_single_tracking (...)       clus/src/TrackFitting.cxx:8870
#10 PatternAlgorithms::find_other_segments (...) clus/src/NeutrinoOtherSegments.cxx:561
#11 PatternAlgorithms::find_proto_vertex (...)   clus/src/NeutrinoPatternBase.cxx:2885
#12 TaggerCheckNeutrino::visit
```

`TrackFitting.cxx:2065-2071` reads:

```cpp
std::vector<WireCell::Point> ps_vec = examine_end_ps_vec(segment, pts, true, true);
if (ps_vec.size() <= 1) ps_vec = pts;

pts.clear();
{
    WireCell::Point p1 = ps_vec.front();   // <-- line 2071
```

The guard rescues a degenerate `examine_end_ps_vec` result by falling back to
`pts`, but **never checks that `pts` is itself non-empty**. An empty `pts` gives
an empty `ps_vec`, and `.front()` on an empty vector is undefined behaviour.

### FIXED — on the owner's instruction, and gated

The paragraph that stood here said the fix belonged to its own round. The owner
asked for the protection directly ("can you provide a protection for this
one-event segfaults"), so it was written, gated, and shipped inside this round.
Recording the reversal rather than editing the reasoning away: the deferral was
the right *default*, not a finding.

**The trap is two-step, and neither step is wrong alone.** `examine_end_ps_vec`
*deliberately* returns an empty list when the whole path drains as face-invalid
— `TrackFitting.cxx:1985-1993` says so in its own comment: "returning an empty
list lets the caller (`organize_ps_path`) fall back to the original `pts` rather
than handing back an out-of-detector point". The `size() <= 1` fallback
implements exactly that contract. What neither side states is that the fallback
needs `pts` to be non-empty, and at the `:8870` call site `pts` was just rebuilt
from `ptss`, which can come back empty. Both empty ⇒ `ps_vec.front()` on nothing.

**The fix** (toolkit, `clus/src/TrackFitting.cxx`) is an early return in
`organize_ps_path` when `pts` is empty, plus the companion guard in
`examine_end_ps_vec` whose own `ps_list.front()`/`.back()` (`:1949`, `:1998`)
are unguarded for the same input. No knob. **A default-OFF knob is the rule for
a behaviour change; this is not one** — the early-out fires only on an input
where the current code has no defined behaviour at all, so there is no legacy
path to preserve. That is a claim about the *whole* sample, not about evt54629,
which is why it was gated rather than asserted.

**Gate — `gate_cmp_arms.py`, member-content hashing (M2), fixed vs pre-fix
binary:**

| arm | events | artifacts | differing |
|---|---:|---:|---:|
| `work-nuecc48-harv3` | 48 | 192 | **0** |
| `work-ncpi0-harv3` | 19 | 76 | **0** |
| `work-mcp1k-harv3` | 1000 | 4000 | **0** |
| **total** | **1067** | **4268** | **0** |

`GATE PASS -- 0/1000 events differ on reconstruction output, byte-identical by
member content`, and the non-gating `nusel-*.tsv` legs came back 0 differing
too. Freshness proof (M1): source last written 10:03:28, `local/lib/
libWireCellClus.so` installed 10:04:11, earliest arm-B event dir 10:13:13 — the
comparison arm ran under the fixed library, the reference arms under the
unfixed one.

**Reproducer-first test**, `clus/test/doctest_clus_organize_ps_path_empty.cxx`:
2 cases / 7 assertions, covering both `organize_ps_path` overloads (including
`end_point_limit = 0`, the `:8870` call site that crashed) and all four
`flag_start`/`flag_end` combinations. It is **revert-proven** in the strong
sense — the guards are reached before any member or the segment is touched, so
the deliberately-null `shared_ptr<PR::Segment>` means removing the fix crashes
the runner instead of failing politely. `./build/clus/wcdoctest-clus` 211/211.

**A discarded gate run, recorded so the number is not quietly reused.** The
first mcp1k leg was launched and *then* the library was rebuilt twice underneath
it (revert for the proof, restore) while it was still running. It spanned three
binaries and returned 11 unattributable `rc=1`. That is not evidence of
anything; it was discarded and re-run into a fresh directory
(`/home/xqian/tmp/pr82/gate-mcp1k2`) with no rebuild during. The nuecc48 and
ncpi0 legs finished before the first rebuild and are untouched by this.

Impact on this round: evt54629 was re-run into `work-mcp2k-harv3` in place
(owner: "no need to repeat the entire production, just add this event back"),
so the arm is **2000/2000 `rc=0`, 880 dumps**. Bee for that event alone:
<https://www.phy.bnl.gov/twister/bee/set/29e1933e-309e-4a8b-aca5-50259eb3d96d/event/list/>

## 12.8 Where this leaves the tree

**Disk.** `sbnd_xin` goes from the **23 G** yesterday's retirement round reached
to **~62 G**. The new sample is ~38.6 G of that and the harvest arms 3.5 G.

**CORRECTION (2026-08-16, same day).** An earlier draft of this section called
the PR arms' non-calib products "the retirable layer, when the owner wants it
back", on the reasoning that the scan and the DL training read only
`calib-pr-evt*.json`. **That was wrong and the advice is withdrawn.** Measured
composition of `work-mcp2k-harv3` (6.0 GB):

| class | files | size | consumer |
|---|---:|---:|---|
| `calib-pr-evt*.json` | 880 | 0.54 GB | the scan, `build_dataset.py`, the viewer |
| `pctree-pr-evt*.tar.gz` | 2000 | **4.42 GB** | **`gate_cmp_arms.py` GATING class**, pr36/37/38 comparisons |
| `mabc-pr.zip` | 2000 | 0.46 GB | **`gate_cmp_arms.py` GATING class**, and the Bee upload source |
| `tracking-pr.root` | 2000 | 0.24 GB | pr32/33/36/37 comparisons, `ttag_cmp5.py` |
| logs | 6000 | 0.19 GB | `nusel_extract.py` parses them for the label table |

The two largest classes are precisely the two inputs `gate_cmp_arms.py` gates a
byte-identity A/B on (`pctree` + `mabc` + `rc`). Deleting them would not free
"unused" space — it would permanently remove this arm's ability to be A/B-gated
again, which is the mechanism every claim in this tree rests on. That loss has
already been taken once knowingly: `PROTECTED.txt` records that dropping
`work-pr87-postflip-*` left its "42/42 archives == pr87ion3" claim alive only in
doc prose, not on disk.

It was also falsified immediately: the Bee link for evt54629 (§12.7) was
produced from `mabc-pr.zip` in that very layer, the same day the layer was
called retirable.

**Correct characterisation: `work-mcp2k-harv3` is live campaign input, not a
spent by-product, and nothing in it is safely retirable while this round is
open.** If disk pressure becomes real, the defensible candidates are the
*regenerable* upstream stages, and each costs the wall time to rebuild:
`work-mcp2k-cb0816` (17 GB, regenerable from imaging), `work-img-mcp2k` (13 GB,
regenerable from staging), staging itself (2.6 GB, regenerable from yuhw's two
art files). But "regenerable" is not "free", the Q/L hub is the input any PR
re-run needs — this round re-ran evt54629 through it — and the owner's own
standing rule from the last retire round is *keep the latest and the input to
achieve it*. Under that rule the whole `mcp2k` chain is KEEP. Any retirement
here is a fresh owner decision, not a follow-on from this round.

**Ready for §5/§6 whenever the owner wants them, and not started:**
- ~879 unscanned `mcp2k` calib dumps + 38 never-scanned `mcp1k` dumps (§1.4).
- A complete 473-label current-epoch set with live harvest features on every
  event, reachable as `vtx_io.TAGS_HARV3`.
- `build_dataset.py --harvest-roots <tag>=work-<sample>-harv3 --tags ...` will
  now resolve, which it could not this morning.

