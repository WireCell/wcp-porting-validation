# doc pr/18 — neutrino-stage iso-band guard: stop re-gluing the nu candidate onto an isochronous cosmic band

Status: fix implemented (`protect_iso_band_xext`, C++ default 0 = OFF); SBND
production default ON (`nu_iso_band_guard`, owner decision 2026-08-01).
Validation: this doc §5–§7.

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
export SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-2025fall-48evt-fsprod
# evt 10550 = idx 4 of the nueCC48 sample (samples/lynn-nuecc-rse.csv line 22)

# --- attribution (per-stage Bee layers + separation decision trace) ---
SBND_WORK_ROOT=$PWD/work-nuecc48-iso10550tr  SBND_TRACE_BEE=1  ./run_ql_evt.sh data 4
SBND_WORK_ROOT=$PWD/work-nuecc48-iso10550dbg WCT_SEP_DEBUG=1   ./run_ql_evt.sh data 4

# --- the fix, ON (SBND default) and byte-identical escape ---
SBND_WORK_ROOT=$PWD/work-nuecc48-isog10550on                       ./run_ql_evt.sh data 4
SBND_WORK_ROOT=$PWD/work-nuecc48-isog10550off SBND_NU_ISO_GUARD=0  ./run_ql_evt.sh data 4

# --- no-regression sweeps (vs the pr/15 vveto baselines) ---
TAG=isog1k ./run_full1k_nusel.sh 1000 6
python3 iso_sweep_compare.py --base work-mcp1kall-vveto1k --on work-mcp1kall-isog1k \
    --out /home/xqian/tmp/isog_sweep_mcp1k.tsv --jobs 16
python3 iso_sweep_compare.py --base work-nuecc48-vveto --on work-nuecc48-isog \
    --out /home/xqian/tmp/isog_sweep_nuecc.tsv --jobs 12
```

## 1. Symptom

SBND run 18255 evt 10550 (nueCC48 data candidate, Bee set 2f6ad762 index 3):
an isochronous cosmic track is over-clustered with the neutrino candidate —
owner points (84.0, 79.2, 140.7) and (69.4, 57.4, 143.9) both in img-global
cluster 19 (12308 pts).  The merged object becomes the 374 cm TGM main of the
event's SOLE in-beam bundle (t0 1.193 µs), so the event is TGM-killed — one
of the 3/48 events at the doc pr/1 §2.2 loss floor.

## 2. Root cause — the neutrino stage re-glues what separate already split

Per-stage trace (`SBND_TRACE_BEE=1`, tag `work-nuecc48-iso10550tr`, APA1
layers; owner points matched by coordinate):

| stage | verdict |
|---|---|
| tr00 ClusteringPointed | **SAME cluster from the start** — the fusion is inherited from the blob graph (imaging connectivity), no merge stage creates it |
| tr01–tr07 | SAME (no stage splits it) |
| tr08 ClusteringSeparate | **SPLIT** (nu piece / band piece) |
| tr09 Connect1, tr10 Deghost, tr11 ExamineXBoundary, tr12 ProtectOverclustering | split SURVIVES |
| **tr13 ClusteringNeutrino** | **re-merged** |
| tr14–tr15 | stay merged → the final img-global cluster 19 |

The two tr12 pieces:

- **band** (9398 pts): 374 cm, drift-x extent **8.8 cm**, y-z footprint
  263×266 cm — the isochronous cosmic + its ghost fan.
  `iso_band_like()` = true.
- **nu candidate** (1657 pts): 55 cm, drift-x extent **45.6 cm** — a genuine
  drift-spanning object.  Its tip touches the band at **0.31 cm**.

Separate's own trace (`WCT_SEP_DEBUG=1`): the merged cluster (ident 103,
378 cm, 1253 blobs) fires `JudgeSeparateDec_1` (r1 = 0.0166, the isochronous
PCA signature), proceeds via the top-touch + "< 65° && > 360 cm" ladder rung,
and `Separate_1` carves the nu candidate out.  The pr/15 vertex veto
correctly KEEPS the split (`SEPDBG vveto … behind=0.365/0.062 … keep` — a
through-going X, not a V).

The re-merge happens in `clustering_neutrino.cxx`: the stage's extended-cloud
pair tests accept the pair, and the existing `protect_iso_band` veto
(PDVD-validated, band/non-band merges need a ≤ 6 cm genuine touch — and OFF
for SBND anyway) cannot discriminate here because the two really do touch at
0.31 cm.

## 3. Fix — `protect_iso_band_xext` (drift-extent veto)

Physics: an isochronous band lives in ONE narrow drift slab; it can never
legitimately claim charge that spans tens of cm of drift.  A touch, however
genuine, does not make a 45.6 cm drift-spanning object part of an 8.8 cm-wide
band.

`clus/src/clustering_neutrino.cxx`: new config key `protect_iso_band_xext`
(C++ default 0 = off ⇒ byte-identical; only read when `protect_iso_band` is
on).  Inside the existing band/non-band veto, after the 6 cm touch rule: if
the non-band partner's blob-center drift-x extent exceeds the knob, the merge
is refused regardless of touch, with an unconditional marker line
`Neutrino iso_band_guard: refused band/non-band merge, nonband xext ... cm, …`.

Both veto branches print a marker when the xext knob is active (so every
SBND firing is classifiable from the harness stdout logs):
`refused band/non-band far merge, gap ... cm` = the pre-existing 6 cm rule,
`refused band/non-band merge, nonband xext ... cm` = the new drift-extent
rule.  PDHD/PDVD run `protect_iso_band` WITHOUT the xext knob and keep their
exact stdout (marker gated on `protect_iso_band_xext > 0`).

Config: `neutrino()` factory arg `protect_iso_band_xext=null` (key-suppressed,
cfg/pgrapher/common/clus.jsonnet) → SBND `nu_iso_band_guard` arg
(cfg/pgrapher/experiment/sbnd/clus.jsonnet clus_per_face/per_apa, default
TRUE ⇒ `protect_iso_band=true, protect_iso_band_xext=20 cm`) → TLA in
wct-clus-matching-perevt.jsonnet.  Runner: `SBND_NU_ISO_GUARD=0` forces the
pre-fix path (byte-identical), unset inherits production ON.  `per_volume`
(LArSoft production entry) inherits the clus_per_face default ⇒ ON.

Note this also turns ON the pre-existing 6 cm far-veto (`protect_iso_band`)
for SBND — the two ship as one package under `nu_iso_band_guard`.

## 4. Why it hid

The clustering chain HAD already fixed this event once per run: separate
correctly cut the nu candidate off the band at tr08 and four subsequent
stages preserved the split.  The neutrino stage — whose job is gathering nu
fragments — undid it as its extended-cloud tests see only a 0.31 cm touch.
In the final Bee display nothing looks broken: one big cluster, one TGM, no
crash.  Only the owner's hand-scan (nu colored with the cosmic) exposed it.

## 5. Knob-off byte-identity (gates)

- Compiled config: `nu_iso_band_guard=false` compile is byte-identical
  (`cmp`) to the HEAD compile (both sessions' knobs key-suppressed); default
  compile adds exactly 2× `protect_iso_band` + 2× `protect_iso_band_xext`
  (one per APA).
- SBND runtime escape: evt 10550 QL with `SBND_NU_ISO_GUARD=0` hash
  `7bbc6fe5…` == `work-nuecc48-vveto` baseline (across the binary change);
  re-verified after the far-veto-marker rebuild (`work-nuecc48-isog2off`,
  same hash).
- abtest (pdhd+pdvd, events.txt, clus stage): A = `post_vveto_clus`
  (pr/15 binary), B = `post_isoguard_clus` → `ab_compare.sh` OVERALL PASS;
  re-run after the far-veto-marker rebuild as `post_isoguard2_clus` →
  OVERALL PASS.  (PDHD/PDVD run `protect_iso_band=true` WITHOUT the xext
  key ⇒ both C++ edits proven inert for them.)
- qlport uboone MABC: base `cbroff_new_ub` vs `isog_ub` → ZIPS 35/35
  content-identical, lib-mtime bracket LIBS_STABLE; re-run after the
  far-veto-marker rebuild as `isog2_ub` → ZIPS 35/35 content-identical.
  Tagger gate 2 reads identical=2/diff=33 — the documented
  non-discriminating A/A noise (doc pr/2 §8 via pr/14 §5.1, reproduced
  base-vs-base in pr/15 §5); ZIPS content-identity is the gate.
- `./build/clus/wcdoctest-clus`: 565/565 pass (both builds).
- Freshness proof: libWireCellClus.so + clustering_neutrino.cxx.2.o Aug 1
  16:07 > source edit 16:04 (xext build); 17:03 > 17:00 (far-veto-marker
  build).

## 6. Demonstration on 10550 (guard ON)

QL (tag `work-nuecc48-isog10550on`): marker
`Neutrino iso_band_guard: refused band/non-band merge, nonband xext 45.6 cm,
lens 57.73/375.33 cm, touch 0.358 cm`; img-global now holds the nu candidate
(c20) and the band (c19) as SEPARATE clusters.

PR outcome: the nusel rows are LINE-IDENTICAL to the baseline — the Q/L
matching still puts both clusters into the same beam-window bundle
(gid 1000002, t0 1.193 µs), `examine_bundles` collapses the bundle back into
one 12372-pt cluster, and TGM still kills it.  The clustering-level split is
achieved and does not regress anything, but recovering the EVENT additionally
needs the bundle level to evaluate the nu piece separately — that is
selection-policy territory (the pr/3 §8.5 / pr/14 / pr/16 machinery), flagged
as follow-up in §9, not smuggled into this clustering fix.

## 7. No-regression sweeps (1000-event mcp1k + 48-event nueCC48 data)

Arms `work-mcp1kall-isog1k` / `work-nuecc48-isog` (guard ON, production
stack, LIBS_STABLE bracket) vs the pr/15 baselines `work-mcp1kall-vveto1k` /
`work-nuecc48-vveto`.  Census `iso_sweep_compare.py`
(`/home/xqian/tmp/isog_sweep_{mcp1k,nuecc}.tsv`).  The first census pass
found 47 + 6 hash diffs WITHOUT the (xext) marker; root cause: the package
also enables the pre-existing 6 cm far-veto, whose refusal path had NO
marker.  Added the far-veto marker (§3, cout-only, xext-gated), re-ran every
diff event single-event into fresh tags `work-mcp1kall-isogrr` /
`work-nuecc48-isogrr2` with the marker binary: every re-run reproduces the
ON-arm member hashes exactly (deterministic) and every diff carries a marker.

**mcp1k (1000 events): 951 identical, 49 guard firings, 0 unexplained.**

- 2 × xext veto (the new rule): evt 350935 (nonband xext 44.7 cm, touch
  2.5 cm), evt 405432 (xext 20.9 cm, touch 5.2 cm).
- 47 × 6 cm far-veto: every marker shows a short band-like fragment
  (6–65 cm) refused a ≥ 6.3–72 cm gap merge with a long track — the
  extended-cloud prolongations these refusals kill are exactly the pr/18
  topology.  13 of the 47 change only pctree internal ordering (mabc
  content-identical; merge-order → ident renumbering); of the other 34,
  20 shift points between non-beam bundles.
- **Beam-window labels: ZERO changes across all 1000 events** (per-event
  in_beam label comparison, both arms).

**nueCC48 (48 real-nu data events): 41 identical, 7 guard firings, 0
unexplained — THE efficiency gate PASSES with a net gain:**

| evt | marker | nusel outcome |
|---|---|---|
| 10550 | xext veto (45.6 cm nonband, 0.36 cm touch) | the target: nu/band split in img-global; PR rows unchanged (§6) |
| 271851 | far-veto, gap 79.5 cm, lens 21.8/172.7 cm | **RECOVERED: in-beam TGM (230.9 cm, 14824 pts) → contained nu-candidate (159.8 cm, 10398 pts)**; the 172 cm cosmic piece + band re-match to their own flash, not-tagged |
| 30504 | 2 × far-veto (gaps 20.0/40.5 cm) | stays nu-candidate contained; beam main sheds a 57-pt band fragment (220.0 → 184.5 cm) |
| 42280, 437699, 444187 | 1 far-veto each | nusel rows line-identical (composition-only archive diffs) |
| 234638 | 1 far-veto | pctree ordering only (mabc content-identical) |

271851 and 10550 were 2 of the 3 nueCC48 events at the doc pr/1 §2.2
TGM-kill floor; the guard recovers 271851 outright and gives 10550 the
clustering-level prerequisite (§9 for the rest).  No real-nu event loses
charge to the guard: every firing strips cosmic-band material only.

## 8. Files

- toolkit: `clus/src/clustering_neutrino.cxx` (blob_center_xext +
  protect_iso_band_xext + marker), `cfg/pgrapher/common/clus.jsonnet`,
  `cfg/pgrapher/experiment/sbnd/clus.jsonnet`,
  `cfg/pgrapher/experiment/sbnd/wct-clus-matching-perevt.jsonnet`.
- wcp-porting-img: `run_ql_evt.sh` (SBND_NU_ISO_GUARD), `iso_sweep_compare.py`
  (fork of vveto_sweep_compare.py, M10), this doc.

## 9. Follow-up (out of scope here)

The clustering now hands the Q/L stage two clean objects, but the Q/L LASSO
matches both to the same beam flash (the touching pair shares visibility) and
the bundle collapse re-fuses them for the PR chain.  To actually recover
10550-class events the bundle level must evaluate the drift-spanning member
separately from the band member — candidate mechanisms: the pr/16
`nu_skip_cosmic_bundle_min_length` selection path on a per-member basis, or a
band-aware member split at `unmerge_bundle`.  Owner decision needed on which.
