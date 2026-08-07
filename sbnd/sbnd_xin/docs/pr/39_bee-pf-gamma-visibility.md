# doc pr/39 — Bee π0→γ→e⁻ display question: Bee config default (γ) + shower `end_point` fix, SBND PRODUCTION DEFAULT ON (SBND run 18345 evt 21073 + 8 sibling NCπ0 events)

## Repro block

```bash
# Finding 1 (Bee display): the source mc.json for the set the owner looked at
# (https://www.phy.bnl.gov/twister/bee/set/21b32701-828d-4ff9-a3b9-d7b8807e07b5/event/1/)
unzip -p /home/xqian/tmp/campaign0805/ncpi0-cb0805.zip data/1/1-mc.json > /home/xqian/tmp/evt1_mc.json
# then read wire-cell-bee3/events/static/js/bee/physics/mc.js:85-131 and store.js:32-37

# Finding 2 (PR) diagnosis: the |start-vtx| / |end-vtx| table below, generated
# from that mc.json plus the 8 other pi0 events in the same Bee set:
python3 - <<'PYEOF'
import json, math, zipfile
def dist(a,b): return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
z = zipfile.ZipFile('/home/xqian/tmp/campaign0805/ncpi0-cb0805.zip')
def find_pi0(nodes):
    for n in nodes:
        if n['text'].startswith('pi0'): return n
        r = find_pi0(n.get('children', []))
        if r: return r
    return None
def walk(nodes, vtx, counts):
    for n in nodes:
        t, dd = n.get('text',''), n.get('data',{})
        s, e = dd.get('start'), dd.get('end')
        if t.startswith('e-') and s and e:
            ds, de = dist(s,vtx), dist(e,vtx)
            counts[1 if de<1.0 else (0 if de>ds else 2)] += 1
        walk(n.get('children', []), vtx, counts)
for i in (1,3,4,8,10,12,15,16,17):
    d = json.loads(z.read(f"data/{i}/{i}-mc.json"))
    cg = json.loads(z.read(f"data/{i}/{i}-clustering-global.json"))
    vtx = tuple(find_pi0(d)['data']['start'])
    c = [0,0,0]; walk(d, vtx, c)
    print(i, "run", cg['runNo'], "evt", cg['eventNo'], "ok=%d rev=%d other=%d" % tuple(c))
PYEOF

# Fix: knob-off byte-identical gate + knob-on recovery, run against the
# 5-sample campaign's own ncpi0 Q/L root (M11), same 9 events:
cd sbnd_xin
PR_JOBS=6 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh \
    work-ncpi0-cb0805 work-pr39-off9 data 21073 56982 71372 142421 259542 314838 463565 506114 506746
SBND_SHOWER_ENDPOINT_EXCLUDE_START_VERTEX=1 PR_JOBS=6 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh \
    work-ncpi0-cb0805 work-pr39-on9  data 21073 56982 71372 142421 259542 314838 463565 506114 506746
python3 ../../abtest/hash_archive.py work-pr39-off9/pr_evt<ID>/{mabc-pr.zip,pctree-pr-evt<ID>.tar.gz}  # x9, vs a clean-HEAD-b25bdcf1 rebuild
```

## Symptom

Owner report, Bee set `21b32701-828d-4ff9-a3b9-d7b8807e07b5` event 1 — **SBND
run 18345, subrun 1, event 21073**, `ncpi0-cb0805` sample (confirmed from the
zip's own `data/1/1-clustering-global.json` `runNo`/`eventNo` fields; an
earlier draft of this doc mis-cited run 18255/evt 271851, an unrelated event
from a different sample looked at earlier in the same session — corrected
here): clicking the **π0** node shows a ball; clicking a **γ** node shows
nothing; clicking an **e⁻** node shows a ball + line. Question: is the
invisible γ a particle-flow bug (gamma not tagged to the root neutrino
vertex) or a wire-cell-bee3 display issue?

## Root cause

**Two independent things. Finding 1 is a display config default, not a bug.
Finding 2 is a real PR-side defect — fixed in this change.**

### Finding 1 — γ invisibility: Bee display default, not a bug

`wire-cell-bee3/events/static/js/bee/physics/mc.js` `selectionChanged()` draws
**only** from a clicked node's own `data.start`/`data.end` — it never touches
the separate point-cloud files (`real_cluster_id` does not appear anywhere in
`mc.js`), so a node's id is irrelevant to what renders. Two things happen for
every non-filtered node (`mc.js:120-131`):

* a `THREE.Line` from `start` to `end` (or a `traj_*` polyline, unused by this
  producer's node shape — see doc pr/26 / `clus/docs/bee_output.md`), and
* a fixed radius-2 `SphereGeometry` placed **unconditionally at `start`**.

But `mc.js:88-99` filters by a **substring match on the node's label** first:

```js
else if (node.text.indexOf("gamma") >= 0) {
    if (this.store.config.mc.showGamma) { material = gammaMaterial }
    else { continue }
}
```

and `store.js:33-37` defaults `showGamma` (and `showNeutron`, `showNeutrino`)
to **false**. That explains all three observations with no PR-side
involvement:

| clicked | label filtered? | geometry | what renders |
|---|---|---|---|
| `pi0  … MeV` | no | start == end (a genuine point particle, doc pr/39 unchanged) | zero-length line + sphere at `start` ⇒ **a ball** |
| `gamma  … MeV` | **yes → `continue`** | valid, non-degenerate | **nothing** |
| `e-  … MeV` | no | start ≠ end | line + sphere ⇒ **ball + line** |

**Fix (display-side, not attempted here):** load the set with
`?mc.showGamma=true` appended to the URL, or toggle "Monte Carlo → gamma" in
the dat.GUI panel. Nothing in the toolkit needs to change for this part; the
owner's stated hypothesis ("gamma not tagged to the root neutrino vertex") is
refuted below — every γ's `start` is exactly the neutrino vertex.

Corollary, reported not fixed: `MultiAlgBlobClustering::fill_bee_pf_tree`'s
synthetic pseudo-γ/π0 node ids (`next_id++`, `MultiAlgBlobClustering.cxx:1495`)
are small integers that overlap `clustering-global.json`'s `real_cluster_id`
range. Currently latent because Bee's mc-tree click path never joins node id
to point-cloud id (see above) — but it is a live hazard for any future
feature that does, and it is also an undocumented divergence from the
prototype, which assigns pseudo nodes real encoded ids (`NeutrinoID.cxx:1374`:
`sg1->get_cluster_id()*1000 + acc_segment_id`; π0 similarly at `:1328`).
`bee_output.md:239` documents the `cluster_id*1000+seg_id` scheme as
universal, which these nodes violate. Left unfixed — out of scope for this
change.

### Finding 2 — shower `end_point` collapses onto the neutrino vertex (FIXED, default OFF)

The PF **topology** is correct: every γ node's `start` is exactly the
neutrino vertex `(-32.99, 24.45, 363.23)` cm, nested `pi0 → gamma → e-` as
expected. But walking evt 21073's e⁻ leaves by distance to that vertex
(pre-fix):

| e⁻ node (id, KE) | \|start−vtx\| cm | \|end−vtx\| cm | |
|---|---:|---:|---|
| 59084, 108 MeV | 27.7 | 76.0 | OK — grows away |
| 54067, 16 MeV | 44.4 | 56.8 | OK — grows away |
| 11010, 1135 MeV | 0.0 | 99.7 | OK — starts at vtx, grows away |
| 63099, 18 MeV | 56.8 | **0.00** | **reversed** |
| 16037, 33 MeV | 57.3 | **0.00** | **reversed** |
| 35048, 32 MeV | 63.3 | **0.00** | **reversed** |
| 60088, 21 MeV | 81.0 | **0.00** | **reversed** |
| 57077, 41 MeV | 43.2 | **0.00** | **reversed** |

**5 of 8** showers in this one event had `end_point` sitting exactly on the
neutrino vertex — i.e. the display drew them running *backward*, toward the
vertex, instead of away from it. Across all 9 π0 events in the same Bee set
(script in the Repro block; "other" = neither clearly-away nor at-vertex,
ambiguous within the farthest-vertex geometry):

| run | evt | n showers | ok | reversed | other |
|---:|---:|---:|---:|---:|---:|
| 18345 | 21073 | 8 | 3 | 5 | 0 |
| 18255 | 56982 | 9 | 4 | 1 | 4 |
| 18255 | 71372 | 8 | 6 | 2 | 0 |
| 18255 | 142421 | 6 | 4 | 2 | 0 |
| 18345 | 259542 | 19 | 8 | 8 | 3 |
| 18255 | 314838 | 6 | 4 | 1 | 1 |
| 18255 | 463565 | 9 | 3 | 4 | 2 |
| 18255 | 506114 | 8 | 2 | 4 | 2 |
| 18255 | 506746 | 10 | 4 | 5 | 1 |
| **total** | | **83** | **38** | **32 (39%)** | **13** |

**Root cause — an unported prototype rule, one hop past where pr/38 round 2
fixed it.** `end_point` is computed as "the vertex in the shower's own vertex
set farthest from `start_point`" in three places in `clus/src/PRShower.cxx`
(single-segment branch, multi-segment branch, and
`calculate_kinematics_long_muon`'s muon-vertex search — pre-fix, all iterating
their own node set with no exclusion). The prototype computes the analogous
quantity in `WCShower.cxx:377-387` (and `:314`, `:294-315` for the long-muon
case) by iterating `map_vtx_segs` — and `WCShower::set_start_segment`
(`:541-548`) explicitly does `if (vtx == start_vertex) continue;` when
populating that map, so **the start vertex can never be picked as the
farthest vertex**. For a detached (`start_connection_type` 2 or 3) shower
whose `start_point` sits tens of cm from its own start vertex, the start
vertex — which sits at or near the parent's attachment point, i.e. often the
neutrino vertex itself for a first-generation γ — wins the farthest-distance
search unless it's excluded.

This is the *same* prototype rule doc pr/38 round 2 (commit `df40b2a4`)
established — but that fix covered only `Shower::fill_sets`'s
`exclude_start_vertex` parameter, which feeds `fill_bee_pf_tree`'s BFS
barrier. The `calculate_kinematics{,_long_muon}` farthest-vertex searches
that set `end_point` were never touched by that fix. **Undocumented
divergence**: `clus/docs/porting/porting_dictionary.md` does not list this
rule for `calculate_kinematics`; per CLAUDE.md M15 it was surfaced, and the
owner elected to fix it (this change).

**Blast radius.** `end_point` feeds the Bee PF node geometry drawn by
`fill_bee_pf_tree` (confirmed, this doc). It does **not** feed `init_dir` for
`start_connection_type` 2/3 showers — that computation
(`data.init_dir = (data.start_point - m_start_vertex->fit().point).norm()`)
uses `start_point` and `m_start_vertex` directly, never `end_point`, so shower
*direction* is unaffected by this fix. Whether `end_point` reaches PID or
energy reconstruction beyond the display was not traced further; the fix
ships as a knob specifically so no other output is disturbed by default.

## Why it hid

The Bee mc-tree display never surfaces `end_point` as a number, only as a
line endpoint — and per Finding 1, γ lines are hidden by default, so the
γ→(neutrino vertex) line was never visually inspected before now. The
5-of-8 rate in this one event and 39% across the set shows this was not
rare; it likely read as "showers converge near the vertex," which is
directionally true for a neutrino interaction, so a shortened/reversed
segment could pass a casual scan.

## SBND production default flip (2026-08-06)

Owner: *"since you fixed it, it should be turn on for SBND"*, after
reviewing the recovery numbers above (32/83 → 0/83 reversed showers).
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`:
`shower_endpoint_exclude_start_vertex = false` → `true` — cfg-only, no C++
change (`m_shower_endpoint_exclude_start_vertex{false}` is still the library
default; only the SBND operating point flips it), exactly the
`v3_extension_guard` idiom (doc pr/24 sec 19.1). `run_pr_chain_batch.sh`'s
`SBND_SHOWER_ENDPOINT_EXCLUDE_START_VERTEX` escape is rewritten to match
`SBND_V3_EXT_GUARD`'s post-flip shape: unset now means the cfg default (ON),
`=0` restores legacy, `=1` is a retained no-op.

**Gates:**

| # | check | result |
|---|---|---|
| G1 | compiled-config proof | bare `wcsonnet` compile adds exactly `"shower_endpoint_exclude_start_vertex": true`, and is byte-identical to the already-validated explicit-`true` compile (`work-pr39-on9`'s config); `--tla-code shower_endpoint_exclude_start_vertex=false` compiles byte-identical to the pre-flip JSON (`work-pr39-off9`'s config) |
| G2 | population-level, ON | **reused from the fix's own gate** (no C++ changed by this flip, so re-running is redundant with G1's identity proof): `work-pr39-on9`, 32/83 → 0/83 reversed showers, 9 events |
| G3 | population-level, OFF/legacy | **reused from the fix's own gate**: `work-pr39-off9` vs a clean-HEAD rebuild, 18/18 archives byte-identical, 9 events |
| G4 | unit tests | `./build/clus/wcdoctest-clus`: 95/95 (unaffected — cfg-only) |

**A live re-run of G2/G3 through `run_pr_chain_batch.sh` was attempted and
invalidated mid-gate**, worth recording: `local/lib/libWireCellClus.so` was
rebuilt by a second concurrent session (`clus/src/TrackFitting.{cxx,h}`,
uncommitted, unrelated to this change — feedback_concurrent_sessions_same_tree,
M1) between the two arms of the attempted re-run. Symptom: 3/9 events
(21073, 142421, 259542) lost their `pi0` node identically in *both* the
bare and the `=0` arm — a TrackFitting-driven shift in fitted trajectories
unrelated to this knob, not a knob defect (confirmed identical in both arms,
which this knob alone cannot produce). Discarded as gate evidence rather
than laundered into the numbers above; G1's pure-jsonnet identity proof plus
G2/G3's reuse of the fix commit's own clean-binary gates stand in its place.
No file belonging to the other session was touched, stashed, or rebuilt over.

## Fix

New default-OFF knob **`shower_endpoint_exclude_start_vertex`** (SBND
production default ON since the flip above), following
the `pf_shower_vertex_barrier`/`exclude_start_vertex` idiom (doc pr/38 round
2) exactly:

* `clus/inc/WireCellClus/PRShower.h`: `calculate_kinematics(...)` and
  `calculate_kinematics_long_muon(...)` each gain a trailing
  `bool exclude_start_vertex_from_endpoint = false` parameter.
* `clus/src/PRShower.cxx`: each of the three farthest-vertex searches gains
  `if (exclude_start_vertex_from_endpoint && vtx == m_start_vertex) continue;`
  immediately after its existing `if (!vtx) continue;` — single-segment branch
  (:1046), multi-segment branch (:1154), long-muon branch (:1292).
* `clus/inc/WireCellClus/NeutrinoPatternBase.h`: new
  `PatternAlgorithms::m_shower_endpoint_exclude_start_vertex{false}` member
  (the class that owns every `calculate_kinematics*` call site — both
  `NeutrinoShowerClustering.cxx` (10 sites) and `NeutrinoEnergyReco.cxx` (2
  sites) are `PatternAlgorithms::` member function bodies, so the member is
  already `this`-visible at all 12 without touching any call signature other
  than the trailing arg itself).
* `clus/inc/WireCellClus/TaggerCheckNeutrino.h` /
  `clus/src/TaggerCheckNeutrino.cxx`: mirror member, read via
  `get(config, "shower_endpoint_exclude_start_vertex", ...)` in `configure()`
  (and round-tripped in `default_configuration()`), copied into
  `pattern_algos.m_shower_endpoint_exclude_start_vertex` alongside every other
  `PatternAlgorithms` knob.
* jsonnet: `cfg/pgrapher/common/clus.jsonnet`'s `tagger_check_neutrino(...)`
  builder gains the param plus the key-suppression line
  `+ (if shower_endpoint_exclude_start_vertex then { shower_endpoint_exclude_start_vertex: true } else {})`;
  threaded through both `cfg/pgrapher/experiment/sbnd/clus.jsonnet` top-level
  functions and `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`, all
  defaulting `false` (**ships OFF**, pending owner review of the recovery
  numbers below before any SBND production-default flip, matching the
  two-step pattern doc pr/24 round 5 → round 6 used for
  `v3_extension_guard`).
* `sbnd_xin/run_pr_chain_batch.sh` (this repo): new
  `SBND_SHOWER_ENDPOINT_EXCLUDE_START_VERTEX=1` env hook, mirroring
  `SBND_V3_EXT_GUARD`, so the knob can be probed without a jsonnet edit.

### Gates

**Freshness proof**: `local/lib/libWireCellClus.so` mtime postdates every
edited source file (`ls -la` both, before any run).

**wcdoctest-clus**: `./build/clus/wcdoctest-clus` → 95/95 test cases, 987/987
assertions, 0 failed.

**Compiled-config proof**: `wcsonnet` on `wct-pr-perevt.jsonnet` with the
knob unset vs `-S shower_endpoint_exclude_start_vertex=true` — the key is
absent (0 occurrences) in the former, present (1) in the latter, and the two
compiled JSON documents differ by exactly that one key.

**Knob-off byte-identical gate.** Naively diffing against the pre-existing
`work-ncpi0-cb0805` campaign arm FAILED on 5/9 events — traced to *unrelated*
already-shipped commits between when that arm was built and current HEAD
(chiefly `v3_extension_guard`'s SBND-default flip, `b25bdcf1`), reproduced
identically by comparing that same arm against a `git stash`-clean rebuild of
HEAD with **zero** of this change's edits present. The correct pair is
HEAD-vs-patched-knob-off, holding the campaign lineage question out of it
entirely:
```
git stash push -- clus/ cfg/ && wcbuild   # true "before": HEAD b25bdcf1
run_pr_chain_batch.sh ... work-pr39-true-before9 ...   # 9 events
git stash pop && wcbuild                  # restore this change
run_pr_chain_batch.sh ... work-pr39-off9 ...            # knob unset (default)
```
Result: `work-pr39-true-before9` vs `work-pr39-off9`, `mabc-pr.zip` +
`pctree-pr-evt<ID>.tar.gz` (`hash_archive.py`, member-content hash) — **18/18
archives byte-identical** across all 9 events. Knob-off is a true no-op.

**Knob-on smoke run.** Same 9 events, `SBND_SHOWER_ENDPOINT_EXCLUDE_START_VERTEX=1`
→ `work-pr39-on9`; `mabc-pr.zip` hash differs from `work-pr39-off9` on every
event (the knob is live), and the distance-table script re-run against
`work-pr39-on9`'s `mc.json`s shows full recovery:

| arm | ok | reversed | other |
|---|---:|---:|---:|
| off (`work-pr39-off9`) | 38 | 32 | 13 |
| on (`work-pr39-on9`) | 74 | **0** | 9 |

Zero reversed showers across all 9 events with the knob on; most of the
ambiguous "other" cases resolve to "ok" too (13 → 9).

## Full-scale verification (2026-08-06): 48 nueCC + 19 NCπ0 events

The 9-event gate above (G2/G3) covers only a subset of the NCπ0 sample and
used the Bee `mc.json` distance table.  Owner asked to reprocess the **full**
`nueCC48` (48 events) and **full** `NCπ0` (19 events, not just the 9 used for
the fix gate) samples to confirm the fix at full population scale, with the
SBND production default (now ON, no env override needed) — and using a more
direct, sample-agnostic check than the `mc.json`/vertex-position heuristic:
read `calib-pr-evt<ID>.json`'s own `showers[]` + `vertices[]` directly (the
PR chain's `pr_display` stage output, not a Bee-display derivative), resolve
each shower's `start_vertex_id` to that vertex's fit position, and flag
`reversed` when `dist(end, start_vertex_pos) < 1e-6 cm` while
`dist(start, start_vertex_pos)` is not — i.e. directly testing whether
`end_point` collapsed onto the shower's own start vertex, the exact
pre-fix defect, regardless of whether that vertex happens to be the neutrino
vertex (so it also works for `nueCC48`, which has no π0/γ structure to
anchor a "distance to neutrino vertex" heuristic).

```bash
cd sbnd_xin
PR_JOBS=6 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh \
    work-ncpi0-cb0805  work-pr39-verify-ncpi0-19 data   # all 19 events
PR_JOBS=6 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh \
    work-nuecc48-cb0805 work-pr39-verify-nuecc48 data   # all 48 events
python3 check_shower_endpoint.py work-pr39-verify-ncpi0-19 work-pr39-verify-nuecc48
```

(`check_shower_endpoint.py` — new, ~50 lines, described above; committed
alongside this doc so the Repro block above is directly runnable.)

Both batches: `ok: 19/19` and `ok: 48/48`, zero failures
(`===== batch summary =====` in each run's log).

| sample | events | showers checked | ok | reversed |
|---|---:|---:|---:|---:|
| NCπ0 (all 19, `work-ncpi0-cb0805`) | 19 | 353 | 353 | **0** |
| nueCC (all 48, `work-nuecc48-cb0805`) | 48 | 670 | 670 | **0** |
| **total** | **67** | **1023** | **1023** | **0** |

**Zero reversed showers across all 1023 showers in all 67 events** — full
confirmation that the fix (now the SBND production default) eliminates the
defect at population scale, not just on the original 9-event/32-reversed
subsample. Per-event breakdown (all 67 rows, `ok`/`reversed`/`no_start_vertex`
columns) is committed at `docs/pr/39_verify_endpoint_check.log`; every single
row reads `reversed=0`.

**Binary-provenance caveat, stated transparently (same concurrent-session
situation as the flip section above):** at the time these two batches ran,
`local/lib/libWireCellClus.so` (mtime 09:44) already carried the *first* of
several uncommitted `TrackFitting.{cxx,h}` edits from a second live session
(feedback_concurrent_sessions_same_tree); a *further* edit to
`TrackFitting.{cxx,h}` plus `TrackFittingPresets.h`/`PatternDebugIO.cxx`
landed only afterward (mtime 09:54, confirmed by `ls -la` after both batches
completed) and was never built into the library used here — so a single,
internally-consistent binary was used across all 67 events, and no rebuild
happened mid-run. This is not a byte-identical A/B gate (no baseline
comparison is claimed), so partial staleness of an unrelated file wouldn't
invalidate it regardless; noted only for the record. The reversed/ok
determination itself is a pointer-equality-derived topological fact
(`vtx == m_start_vertex`) reflected as an exact position match, not a
floating-point fit-quality comparison, so it is insensitive to whatever
TrackFitting does to segment trajectories in any case. No file belonging to
the other session was touched, staged, or rebuilt over.

## Verification

- [x] Finding 1: mechanism confirmed by reading `mc.js`/`store.js`; owner to
      confirm live with `?mc.showGamma=true`.
- [x] Finding 2 diagnosis: distance table regenerated directly from
      `data/1/1-mc.json` and the 8 sibling π0 events by the Repro-block
      script (not hand-copied); run/evt numbers cross-checked against each
      event's own `clustering-global.json` metadata (the original draft's
      run/evt citation was wrong — corrected above).
- [x] Fix: knob-off byte-identical vs clean HEAD, 18/18 archives, 9 events.
- [x] Fix: knob-on smoke run, 32/83 → 0/83 reversed showers, quoted above.
- [x] Fix: full-scale confirmation, 48 nueCC + all 19 NCπ0 events (not just
      the 9-event fix subsample), 0/1023 reversed showers, quoted above.
- [x] `wcdoctest-clus` 95/95.
- [x] No iterated pointer-keyed containers introduced (the new check is a
      single pointer-equality test against the already-held `m_start_vertex`,
      not a new container).
- [x] Freshness proof done before each gate.
- [x] SBND production-default flip: compiled-config identity proof (G1);
      population-level ON/OFF gates reused from the fix's own clean-binary
      run (G2/G3, no C++ changed by the flip); a live re-run was attempted
      and invalidated by a concurrent session's unrelated rebuild, discarded
      rather than used, see "SBND production default flip" above.
- **Status: Finding 1 no toolkit action needed. Finding 2 FIXED, SBND
  PRODUCTION DEFAULT ON** since 2026-08-06 (owner: "turn it on for SBND"),
  confirmed at full population scale (0/1023 reversed showers, 67 events).
  Legacy behavior restorable via `SBND_SHOWER_ENDPOINT_EXCLUDE_START_VERTEX=0`
  or `--tla-code shower_endpoint_exclude_start_vertex=false`. Code lives in
  `toolkit` (fix `34fc09ca`, flip `2a432b82`); this doc + the
  `run_pr_chain_batch.sh` env hook live here.
