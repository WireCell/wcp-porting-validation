# doc pr/15 — separate() vertex veto: stop cutting a neutrino in two at its own vertex

Status: fix implemented (`vertex_veto`, C++ default OFF); SBND production default
ON (owner decision 2026-08-01).  Validation: this doc, §6–§7.

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# --- the founding event: run 18255 evt 56463 (mcp1k entry 599) ---
# per-step attribution of the split (trace_bee layers, fresh tag):
TAGDIR=$PWD/work-mcp1kall-cbr56463trace   # exists; new tag if re-running
SBND_INPUT_DIR=$PWD/input_files_reco1/staged-mcp2025c-1000evt/e599 \
    SBND_WORK_ROOT=$TAGDIR SBND_TRACE_BEE=1 ./run_ql_evt.sh data 1
# separation decision trace (SEPDBG lines on raw stdout -> batch log):
SBND_WORK_ROOT=... WCT_SEP_DEBUG=1 ./run_ql_evt.sh data 1

# --- the fix, ON (SBND default) and byte-identical escape ---
SBND_INPUT_DIR=... SBND_WORK_ROOT=$PWD/work-mcp1kall-cbr56463vvon2 ./run_ql_evt.sh data 1
SBND_INPUT_DIR=... SBND_WORK_ROOT=$PWD/work-mcp1kall-cbr56463vvoff SBND_SEP_VVETO=0 ./run_ql_evt.sh data 1

# --- no-regression sweep: full 1000-event mcp1k + all 48 nueCC48 ---
TAG=vveto1k ./run_full1k_nusel.sh 1000 6          # veto ON (production default)
python3 scripts/analysis/misc/vveto_sweep_compare.py --base work-mcp1kall-cbron1k --on work-mcp1kall-vveto1k \
    --out /home/xqian/tmp/vveto_sweep_mcp1k.tsv
python3 scripts/analysis/misc/vveto_sweep_compare.py --base work-nuecc48-cbron --on work-nuecc48-vveto \
    --out /home/xqian/tmp/vveto_sweep_nuecc.tsv
```

## 1. Symptom

SBND run 18255 evt 56463 (MC, mcp1k entry 599): the owner's scan found the
neutrino interaction split across two Bee img-layer clusters — cluster 18 at
(140.3, 174.6, 205.2) and cluster 20 at (115.3, 148.8, 204.3) — so only part
of the interaction survived into the beam-window PR evaluation.

## 2. Root cause — ClusteringSeparate's top-cosmic angle ladder

Per-step trace (`SBND_TRACE_BEE=1`, tag `work-mcp1kall-cbr56463trace`): through
`ClusteringExtendLoop` (tr07) the interaction is ONE 3316-pt, 261 cm TPC1
cluster; `ClusteringSeparate` (tr08) cuts it into 1947 + 1344 (+25) points and
nothing downstream rejoins them (the pieces still touch at 0.36 cm).

Decision trace (`WCT_SEP_DEBUG=1`, SEPDBG lines, cluster ident 57):

- `JudgeSeparateDec_2` (two-surface exit) correctly did NOT fire (nsurf=1);
- `JudgeSeparateDec_1` fired: the shower arm gives PCA ratio1 = 0.086 (above
  the angle-keyed thinness threshold) — "transverse structure present";
- the cluster touches the TOP wall (shower arm reaches y = 199.9) →
  `flag_top`, and the combined PCA main axis is 19.5° from beam-perpendicular
  → ladder rung "< 33° && length > 160 cm" → split.

`Separate_1` carves along the main track direction; the cut lands exactly at
the interaction vertex V ≈ (132.7, 167.3, 189.5) (raw): a straight muon-like
arm (1947 pts, vertex → cathode at x_raw = 1.8) vs a shower arm (1344 pts,
vertex → top wall).  This is the ported prototype cosmic-breaker heuristic
working as designed on the wrong victim: at the per-APA clustering stage there
is no timing information, and a nu with a long steep muon + a wall-touching
shower prong is geometrically indistinguishable from a top cosmic clipping
other activity — EXCEPT by the vertex topology (§3).

Downstream compounding (doc pr/12's absorbing-window defect is also active in
this event): the shower piece was Q/L-matched into the 11282-pt cosmic bundle
at −198.7 µs, the muon piece into the +5.746 µs wrong-window TPC1 flash; only
the muon's TPC0 continuation (img c11) sat in the beam bundle.

## 3. Fix — the vertex-veto (owner's discriminator)

Two crossing cosmics continue THROUGH their junction: at least one piece keeps
charge on both sides.  Two arms of one neutrino both END at the junction — the
junction IS the vertex — and two random coincident cosmics whose tips meet at
one interior point are geometrically rare.  So:

`veto_vertex_split` (clus/src/clustering_separate.cxx), a knob-gated
(`vertex_veto`, C++ default false ⇒ byte-identical) post-pass that runs after
all separation-family refinements:

1. the family's two longest survivors A, B (A ≥ 60 cm, B ≥ 30 cm), all other
   members together < 5% of the arm points;
2. arms touch: closest approach < 5 cm (Separate_2's relink scale); the
   junction J = midpoint of the closest pair;
3. endpoint test BOTH arms: npoints-weighted blob-center fraction behind J
   (along J→centroid, 5 cm slack) < 2%;
4. opening angle between the arms at J in (15°, 160°) — near-collinear pairs
   (one track cut in half) and hugging-parallel bands are NOT vertices;
5. all pass → merge every family member back into A (same
   take_children/destroy_child + scope-restore idiom as
   merge_collinear_members), log `Separate vertex_veto: rejoined V at ...`.

Measured on 56463 (SEPDBG `vveto` line): touch 0.357 cm, behind 0/0, opening
84.5°, frag 0.8% → VETO.  Control (junction forced mid-track on the same
event): behind-fraction reads 0.47 — the X/V discrimination is not marginal.

Config: `vertex_veto` arg on the common separate() factory
(cfg/pgrapher/common/clus.jsonnet, key-suppressed default false) → SBND
per-APA pipeline `sep_vertex_veto` (cfg/pgrapher/experiment/sbnd/clus.jsonnet
clus_per_face / per_apa, default TRUE for SBND) → TLA `sep_vertex_veto=true`
in wct-clus-matching-perevt.jsonnet.  Runner: `SBND_SEP_VVETO=0` forces the
pre-fix path (byte-identical), unset inherits production ON.
`per_volume` (LArSoft production entry) inherits the clus_per_face default ⇒ ON.

## 4. Why it hid

The split leaves the two pieces 0.36 cm apart, but separate runs LAST among
the merging-heavy per-APA stages precisely so its cuts survive; connect1 etc.
do not rejoin them.  In the nusel table the event still shows a plausible
"nu-candidate" (the TPC0 muon continuation + an unrelated TPC0 track in the
beam bundle), so nothing crashed or looked empty — it took the owner's
hand-scan of the Bee display to notice the interaction was in two colors.

## 5. Knob-off byte-identity (gates)

- Compiled config: `sep_vertex_veto=false` compile is byte-identical (`cmp`)
  to the pre-knob compile; the new default differs from the old compile by
  exactly two `"vertex_veto": true` keys (one per APA ClusteringSeparate).
- SBND runtime escape: 56463 QL with `SBND_SEP_VVETO=0` hash
  `d4467b1f…` == the pr/14 `cbron1k` sweep arm (across the binary change).
- abtest (pdhd+pdvd, events.txt, clus stage): A = `cbroff_new_clus`
  (pre-change binary), B = `post_vveto_clus` → `ab_compare.sh` OVERALL PASS.
  (Label `post_vveto` is VOID — an img-mode snapshot attempt that skipped on
  missing SP frames after the 2026-07-30 input retirement; superseded by
  `post_vveto_clus`.)
- qlport uboone MABC gate: base `cbroff_new_ub` vs `vveto_ub2` →
  ZIPS 35/35 content-identical, lib-mtime bracket LIBS_STABLE.
  (Label `vveto_ub` is VOID — 3 events crashed on a concurrent-session
  install race, "libWireCellClus.so: file too short".)
  Tagger gate 2 reads identical=2/diff=33 — the documented non-discriminating
  A/A noise (doc pr/2 §8, quoted in pr/14 §5.1); reproduced identical=2/diff=33
  running ab_check between the two pre-change baselines themselves.  ZIPS
  content-identity is the gate.
- `./build/clus/wcdoctest-clus`: 565/565 pass.
- Freshness proof: libWireCellClus.so 13:59 > clustering_separate.cxx 13:58
  (2026-08-01).

## 6. Demonstration on 56463 (veto ON)

QL (tag `work-mcp1kall-cbr56463vvon2`): `Separate vertex_veto: rejoined V at
(132.653, 166.753, 188.625) cm, arms 260.97 cm total, touch 0.357 cm, opening
84.5°, behind 0/0`.  The img layer now holds the interaction as ONE 3529-pt
cluster (was 1505 + 2024).  The sweep arm reproduces the demo tag exactly
(mabc-all-apa.zip member hash `58b832d6…` in both; baseline `d4467b1f…`).

PR-chain outcome (sweep arms, `.log_e599.log` nusel rows):

| arm | beam flash 1.185 µs | −198.7 µs cosmic bundle | +5.746 µs bundle |
|---|---|---|---|
| cbron1k (pre-veto) | 5537 pts, 546 cm, **TGM** | 11282 pts (holds shower arm) | 3560 pts (holds muon arm) |
| vveto1k (veto ON)  | 1547 pts, 173 cm, **contained → nu-candidate** | 9781 pts | 5526 pts (holds the rejoined 3529-pt nu) |

The veto fixes the SPLIT: the interaction is one cluster end-to-end.  The
pr/12 flash-reco defect still routes that joined cluster to the +5.746 µs
wrong-window flash, so the beam-window candidate is the muon's TPC0
continuation (now contained, no longer dragged into a 546 cm TGM by the
split pieces).  Net: the event flips from TGM-killed to a beam-window
nu-candidate; full recovery of the whole interaction into the beam bundle
additionally needs the flash-reco fix (upstream, out of scope — pr/12 §6).

## 7. No-regression sweep (1000-event mcp1k + 48 nueCC48)

Arms `work-mcp1kall-vveto1k` / `work-nuecc48-vveto` (veto ON, production
default stack incl. cathode rescue and 45dae9d0's nu_skip_cosmic_bundle) vs
the pr/14 `cbron1k` / `cbron` baselines.  Both arms rc=0; installed-lib
mtime bracket LIBS_STABLE (no concurrent-session install mid-sweep).
QL-stage products (mabc-all-apa.zip + pctree) compared per event
(`scripts/analysis/misc/vveto_sweep_compare.py`, TSVs `/home/xqian/tmp/vveto_sweep_{mcp1k,nuecc}.tsv`);
PR products are NOT compared arm-wide (45dae9d0 is PR-stage, so PR diffs are
expected), PR labels inspected for FIRED events only.

**nueCC48: 48/48 IDENTICAL, 0 firings.**  The veto never touches the 48
nue events — no efficiency risk on the signal sample.

**mcp1k: 991/1000 IDENTICAL, 7 firings (0.7%), 2 nondeterminism repeats.**
The only differing events besides the 7 firings are 292643 and 390182:
for each, a fresh single-event re-run with the SAME veto-ON binary
(tags `work-mcp1kall-vv{292643,390182}rr`) reproduces the BASELINE hash
exactly — the sweep-arm difference is the documented bimodal QL
nondeterminism (pr/14 §7, evt 286191 class), not a veto effect.
All 7 firings pass with
unambiguous V geometry (touch < 0.7 cm, behind-fraction ≤ 0.018 vs the 0.47
of the forced-X control, opening 57–116°):

| entry | evt | V (raw cm) | arms | touch | opening | behind | beam-window PR outcome |
|---|---|---|---|---|---|---|---|
| 599 | 56463 | (132.7, 166.8, 188.6) | 261 cm | 0.36 | 84.5° | 0/0 | **TGM → contained nu-candidate** (§6, the founding event) |
| 42 | 292533 | (−119.5, 127.1, 125.4) | 309 cm | 0.31 | 115.2° | 0/0 | unchanged (ident renumbering only) |
| 301 | 313847 | (−41.7, −2.1, 252.2) | 284 cm | 0.55 | 115.8° | 0/0 | unchanged (beam TGM stays TGM; +9 pts) |
| 412 | 59025 | (−164.9, −25.4, 134.9) | 237 cm | 0.68 | 84.4° | 0/0 | unchanged (nu-candidate stays, same rows) |
| 667 | 348889 | (−3.5, −191.7, 360.7) | 361 cm | 0.43 | 107.3° | 0.001/0 | unchanged (465 pts move between two out-of-beam cosmics) |
| 739 | 285795 | (−20.7, 199.3, 100.4) | 117 cm | 0.47 | 57.2° | 0/0 | unchanged (beam STM stays; out-of-beam redistribution) |
| 892 | 174306 | (230.5, −49.7, 354.6) | 237 cm | 0.47 | 79.5° | 0.013/0.018 | unchanged (657 pts move between two out-of-time cosmics) |

Bee sets for hand-check (7 events, index 0–6 in the table's order with
56463 = index 0; layers img-global / clustering-global / op; no PR point
layers exist for these events in either arm):

- baseline (cbron1k): https://www.phy.bnl.gov/twister/bee/set/b1feaa93-1fe6-44ac-88b3-02c9724b5d4f/event/list/
- veto ON (vveto1k):  https://www.phy.bnl.gov/twister/bee/set/881892b2-9d8e-4a88-b639-ab1033f89ab4/event/list/

Only the founding event's beam-window physics changes; the other six rejoin
out-of-beam activity (their beam-window rows are line-identical between the
arms up to cluster-ident renumbering).  Events 348889 / 174306 rejoin what
are presumably genuine cosmic tip-meetings near walls — the merged pieces
stay in out-of-time bundles, so downstream nu selection is unaffected; the
V candidates are available for owner hand-check on Bee.

Note the firing marker is a cout line: it lands in the harness stdout logs
(`.log_e<entry>.log` / `.batch_ql_evt<ID>.log`), not in
`ql_evt*/wct_ql_evt*.log` (wire-cell `-l` captures spdlog only) —
`scripts/analysis/misc/vveto_sweep_compare.py::firing_map` scans the right files.

## 8. Files

- toolkit: `clus/src/clustering_separate.cxx` (veto_vertex_split + knob),
  `cfg/pgrapher/common/clus.jsonnet`, `cfg/pgrapher/experiment/sbnd/clus.jsonnet`,
  `cfg/pgrapher/experiment/sbnd/wct-clus-matching-perevt.jsonnet`.
- wcp-porting-img: `run_ql_evt.sh` (SBND_SEP_VVETO), `scripts/analysis/misc/vveto_sweep_compare.py`,
  this doc.
