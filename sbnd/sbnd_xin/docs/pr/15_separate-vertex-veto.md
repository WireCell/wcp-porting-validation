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
python3 vveto_sweep_compare.py --base work-mcp1kall-cbron1k --on work-mcp1kall-vveto1k \
    --out /home/xqian/tmp/vveto_sweep_mcp1k.tsv
python3 vveto_sweep_compare.py --base work-nuecc48-cbron --on work-nuecc48-vveto \
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
  (pre-change binary), B = `post_vveto_clus` → PENDING §5-RESULT.
  (Label `post_vveto` is VOID — an img-mode snapshot attempt that skipped on
  missing SP frames; superseded by `post_vveto_clus`.)
- qlport uboone MABC two-gate: base `cbroff_new_ub` vs `vveto_ub` →
  PENDING §5-RESULT.
- `./build/clus/wcdoctest-clus`: 565/565 pass.
- Freshness proof: libWireCellClus.so 13:59 > clustering_separate.cxx 13:58
  (2026-08-01).

## 6. Demonstration on 56463 (veto ON)

QL (tag `work-mcp1kall-cbr56463vvon2`): `Separate vertex_veto: rejoined V at
(132.653, 166.753, 188.625) cm, arms 260.97 cm total, touch 0.357 cm, opening
84.5°, behind 0/0`.  The img layer now holds the interaction as ONE 3529-pt
cluster (was 1505 + 2024).  PR-chain outcome: §7-RESULT.

## 7. No-regression sweep (1000-event mcp1k + 48 nueCC48)

PENDING — arms `work-mcp1kall-vveto1k` (veto ON, production default stack
incl. cathode rescue and 45dae9d0's nu_skip_cosmic_bundle) vs the pr/14
`cbron1k` / `cbron` baselines.  QL-stage products (mabc-all-apa.zip + pctree)
compared per event; PR products are NOT compared arm-wide (45dae9d0 is
PR-stage, so PR diffs are expected), PR labels inspected for FIRED events only.

## 8. Files

- toolkit: `clus/src/clustering_separate.cxx` (veto_vertex_split + knob),
  `cfg/pgrapher/common/clus.jsonnet`, `cfg/pgrapher/experiment/sbnd/clus.jsonnet`,
  `cfg/pgrapher/experiment/sbnd/wct-clus-matching-perevt.jsonnet`.
- wcp-porting-img: `run_ql_evt.sh` (SBND_SEP_VVETO), `vveto_sweep_compare.py`,
  this doc.
