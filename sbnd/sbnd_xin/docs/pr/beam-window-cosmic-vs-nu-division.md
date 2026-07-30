# Beam-window cosmic vs neutrino-candidate division on two SBND data samples

What fraction of in-beam-window bundles the cosmic taggers claim (TGM, STM, LM)
and what fraction survives as a neutrino candidate, measured on **both** SBND
real-data samples we have through the PR chain:

- **A — 1000 events, MCP2025C** (runs 18255/18259): **37.3 %** cosmic /
  **62.7 %** nu-candidate over 839 in-beam bundles.
- **B — 48 nueCC candidates** (lynnt's list, runs 18253..18409): **11.8 %**
  cosmic / **88.2 %** nu-candidate over 51 in-beam bundles, with
  **STM = 0 and LM = 0**.

B is new here: the 48-event sample had imaging and Q/L from doc 21 but had
**never been run through the PR tagger chain**, so no label table existed. A is
a census of the existing `work-stmcamp-d66fix` arm, not a new run.

Beam window is the doc-56 gate, `cluster_t0 ∈ [0.2, 2.2) µs`. Labels are
`nusel_extract.py`'s `label` column (priority TGM > STM > LM, in-beam only);
`fc` is orthogonal and never enters `label`.

**No code, no jsonnet and no default was changed by this document.** A is a
read-only census; B is a new run of the unmodified production chain into a fresh
tag.

## Repro block

```bash
cd wcp-porting-img/sbnd/sbnd_xin

# --- A. the 1000-event sample: census the existing arm (no run needed).
#     Second root is the cross-check; the script prints the per-event delta.
python3 bw_label_census.py work-stmcamp-d66fix work-mcp1kall-d59k

#     arm attribution (each pair prints "events differing"):
for p in "work-mcp1kall-d59k work-stmcamp-d66old" \
         "work-stmcamp-d66old work-stmcamp-d66new" \
         "work-stmcamp-d66new work-stmcamp-d66fixoff" \
         "work-stmcamp-d66fixoff work-stmcamp-d66fix"; do
    python3 bw_label_census.py $p | grep -E "^   (TGM|STM|LM|nu-cand)|events differing"
done

# --- B. the 48 nueCC candidates: the run this doc adds.
#     imaging is REUSED from work/evt<ID> (doc 21's 2026-07-21 campaign) by
#     symlink -- licensed by doc 21's identical frames-dnn.tar.bz2 member hash
#     46ff819f... between the fsprod and plain extractions (FrameShift touches
#     only timing products, so the charge side is byte-identical).
SB=$PWD; mkdir -p work-nuecc48-nuf
for e in $(awk -F, 'NR>1{print $3}' ../samples/lynn-nuecc-rse.csv | sort -u); do
    ln -sfn $SB/work/evt$e work-nuecc48-nuf/evt$e
done
find work-nuecc48-nuf -xtype l | wc -l      # must be 0

#     -caf product offsets (doc 21) + the production NUF flag set verbatim
#     (identical string to run_full1k_nusel.sh / run_perf54_nusel.sh).
export SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-2025fall-48evt-fsprod
export SBND_WORK_ROOT=$PWD/work-nuecc48-nuf
export SBND_SAVE_ASSOC=1                    # doc-59 GOTCHA 1: NOT forwarded otherwise
SBND_MAX_JOBS=6 ./run_nusel_evt.sh data \
    -chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm -main-pair-real \
    -fvx 2.5 -fvy 3 -stm-fit -mip 56000 -unmerge-assoc all

python3 nuecc48_detail.py  work-nuecc48-nuf   # per-event, names every bundle
python3 bw_label_census.py work-nuecc48-nuf   # same columns as A
```

## 1. Sample A — 1000 events, MCP2025C

Arm **`work-stmcamp-d66fix`**: HEAD's configuration (post-revert diffusion
DL 4.0 / DT 8.8 plus the doc-66 §12 STM cut package, toolkit `c0501d7e`,
default ON). **839 in-beam bundles** over 1000 events.

| label | n | % of in-beam bundles |
|---|---:|---:|
| TGM | 162 | 19.31 |
| STM | 139 | 16.57 |
| LM | 12 | 1.43 |
| **cosmic total** | **313** | **37.31** |
| **nu-candidate** | **526** | **62.69** |

`nu-candidate` fc split: **261 fc=0 / 265 fc=1** — the contained half is the
more neutrino-like population. Every TGM and STM bundle is fc=0 by
construction (STM skips fc=1 outright, `stmfit=contained`).

Outside that denominator: **86** in-beam `no-bundle` rows (a beam-window flash
that matched no qualifying main-flagged in-scope bundle — nothing to tag) and
**10 503** out-of-window bundle rows the doc-56 gate never evaluates.

Per event (n=1000):

| class | n | % |
|---|---:|---:|
| all in-beam bundles nu-candidate | 502 | 50.2 |
| no in-beam bundle at all | 187 | 18.7 |
| TGM only | 154 | 15.4 |
| STM, no TGM/LM | 138 | 13.8 |
| LM only | 10 | 1.0 |
| mixed (cosmic + keepable, same window) | 9 | 0.9 |

Cosmic-only events = 302 (30.2 %); the 9 mixed are what a hand scan has to
split.

### 1.1 Arm cross-check — only STM may move

The five 1000-event arms reuse the same pctrees, so the QL/clustering layer is
common and only the STM fit/cuts differ. That is a real validity test, and it
passes: **TGM = 162, LM = 12, in-beam no-bundle = 86 and no-in-beam-bundle =
187 are identical in all five arms**, and match doc 59's event counts
(154 TGM-only / 10 LM / 187 no-in-beam) exactly.

STM reconciles arm-by-arm with the published flip counts:

| step | STM bundles | events differing | cause |
|---|---:|---:|---|
| `work-mcp1kall-d59k` | 153 | — | doc 59 production |
| → `work-stmcamp-d66old` | 142 | 15 | doc-63 STM campaign (shipped default ON after d59k) |
| → `work-stmcamp-d66new` | 141 | 11 | doc-66 diffusion revert — **exactly its 11 flips** |
| → `work-stmcamp-d66fixoff` | 141 | **0** | the doc-66 §12 off-gate |
| → `work-stmcamp-d66fix` | 139 | **4** | doc-66 §12 cut package — **exactly its 4 flips** |

The d59k → d66old step is why a naive d59k-vs-d66fix diff shows 20 events, not
the 15 the doc-66 numbers alone would suggest: doc 63 landed in between.

## 2. Sample B — the 48 nueCC candidates (new)

Input list `wcp-porting-img/sbnd/samples/lynn-nuecc-rse.csv`: **48 real-data
nueCC candidate events**, 12 runs 18253..18409, 30 of them run 18255 (matches
doc 21's description of the sample). New tag **`work-nuecc48-nuf`**, 48/48
rc=0.

**This sample is disjoint from sample A** — A is runs 18255/18259 only, and the
event-number sets do not intersect at all, so B is not a slice of A's numbers
and had to be produced separately.

**51 in-beam bundles** over 48 events.

| label | n | % of in-beam bundles |
|---|---:|---:|
| TGM | 6 | 11.76 |
| **STM** | **0** | **0.00** |
| **LM** | **0** | **0.00** |
| **cosmic total** | **6** | **11.76** |
| **nu-candidate** | **45** | **88.24** |

`nu-candidate` fc split: **12 fc=0 / 33 fc=1** — 73 % fully contained, against
50 % in sample A.

Per event (n=48): **42 all-nu-candidate (87.5 %)**, 3 TGM-only (6.25 %),
3 mixed (6.25 %). **All 48 events have ≥1 in-beam bundle** — 0 with none,
against 18.7 % in sample A.

Exactly **1** in-beam `no-bundle` row: run 18255 evt 235435, APA1 flash gid
1000010 at t = 1.836 µs, 499.9 PE. That event still carries its own in-beam
nu-candidate (115.7 cm, fc=0, t = 1.438 µs), so a dim 500-PE APA1 flash
matching nothing is benign, not a lost candidate. Out-of-window bundle rows:
493.

### 2.1 Side by side

| | A: 1000 MCP2025C | B: 48 nueCC |
|---|---:|---:|
| in-beam bundles | 839 | 51 |
| cosmic (TGM+STM+LM) | 37.3 % | **11.8 %** |
| nu-candidate | 62.7 % | **88.2 %** |
| STM | 16.6 % | **0 %** |
| LM | 1.4 % | **0 %** |
| nu-candidate fc=1 | 50.4 % | **73.3 %** |
| events with no in-beam bundle | 18.7 % | **0 %** |

### 2.2 The 6 TGM bundles — all fc=0, all long

| run | event | main | len_cm | t0 (µs) | other in-beam candidate? |
|---|---|---:|---:|---:|---|
| 18255 | 10550 | 11 | 374.1 | 1.193 | **none** — sole in-beam bundle |
| 18259 | 116962 | 15 | 183.4 | 1.645 | **none** — sole in-beam bundle |
| 18255 | 271851 | 24 | 230.9 | 0.764 | **none** — sole in-beam bundle |
| 18255 | 360535 | 19 | 412.8 | 1.350 | yes — nu-candidate 122.9 cm, t=1.793 |
| 18255 | 389538 | 14 | 337.4 | 1.513 | yes — nu-candidate 221.1 cm, t=1.802 |
| 18253 | 444187 | 6 | 211.2 | 1.096 | yes — nu-candidate 174.2 cm, t=1.573 |

All six are one tagger's calls (TGM), all fc=0, all 183–413 cm.

The rate at which a cosmic tag leaves the event with **no** surviving in-beam
candidate is **3/48 = 6.3 %**. Treat that as a **floor, not the rate**: for the
three mixed events nothing in this measurement links either bundle to the nueCC
selection, so if the real neutrino is the TGM-tagged one the loss is as high as
**6/48 = 12.5 %**. **Quote the band 6–13 %**, and with n = 48 even that is
coarse — do not quote a point efficiency.

## 3. How to read each number

The two samples invert each other and must not be read the same way.

**Sample A: `nu-candidate` does not mean neutrino.** It means *untagged inside
the beam window*. Over 1000 SBND data events the true beam-ν rate is
negligible, and each event carries ~11 cosmic bundles of which ~0.8 land in the
2 µs window by chance. A's 62.7 % is therefore overwhelmingly cosmics the
taggers did not catch — it is a tagger-rejection measurement, not a neutrino
rate.

**Sample B is a positive control.** All 48 are externally selected neutrino
candidates, so `nu-candidate` is the *correct* outcome and the ideal result is
0 % cosmic. Every cosmic tag is a candidate neutrino-efficiency loss. The
notable result is **STM = 0 and LM = 0**: on a sample where every event holds a
neutrino candidate, the stopping-muon and light-mismatch taggers fired zero
times. That is the correct behavior, and a cleaner signal than the TGM number
since all 6 cosmic tags come from a single tagger.

### 3.1 Caveats

1. **B is candidates, not truth.** A TGM call on one of the 48 may be the
   tagger being right; this measurement cannot distinguish that from a loss.
2. **Run range / PMT mask (B).** B spans 12 runs against A's 18255/18259, and
   the static QL PMT mask is run-dependent (see memory: *QL data PMT mask is
   run-dependent*). LM fired 0 times in B so the confounder did not bite — but
   LM is correspondingly **untested** on B. If LM tags ever appear on this
   sample, check the mask before reading them as physics.
3. **A's STM is a known slight over-count.** Against doc 62's adjudicated
   72-bundle set with doc 66 §11.1a's two revised labels, this arm holds **2**
   residual wrong STM verdicts — `58755:21` (really a TGM-tagger problem) and
   `289295:15` — both false positives, deliberately left per doc 66 §12.3. The
   in-beam bundles outside the adjudicated set have unmeasured error.
4. **n = 48.** Always print counts beside percentages for B; a single event is
   2.1 %.

## 4. Validity of the sample-B run

- **48/48 rc=0**, all 48 tables present and non-empty.
- **Integrity checks (both samples): 0** in-beam rows with `tgm == -1` (an
  unevaluated in-window bundle would be a hole, not a nu-candidate) and **0**
  with `lm == -1` (LM knob actually on).
- **Freshness proof (M1).** `clus/src/TaggerCheckSTM.cxx` last edited
  2026-07-27 16:27:29 → `local/lib/libWireCellClus.so` installed 16:30:12; the
  installed lib contains the `michel_res_length_cut` / `proton_c_peak_max`
  strings; `clus/` and `cfg/` clean at toolkit `c0501d7e`.
- **Config identity.** The runner's `taggers (...)` echo line is
  **byte-identical** between `work-stmcamp-d66fix` and `work-nuecc48-nuf`, so
  A and B are directly comparable:
  ```
  (switch_scope,unmerge_bundle,unmerge_assoc,steiner,fiducialutils,
   tagger_check_tgm,tagger_check_stm,tagger_check_fc,stm_magnify,
   bw=[0.2,2.2] us bwonly=1, chord=1 mode=path rescue=1 rescue_chord=1
   main_pair=1/real fvz=5 fvzi=3 fvx=2.5 fvy=3 lm=1 stmfit=1 stmfv=1
   stmguards=1 stmpguard=1 stmcguard=1 stmafix=1 stmtguard=1 stmdguard=1
   stmvguard=1 stmd66cuts=1 unmerge=1/real)
  ```
- **`-unmerge-assoc` verified live** (doc-59 GOTCHA 1, whose failure mode is a
  silent per-cluster WARNING): logs show
  `all_flags=[associated_cluster,main_cluster]` and no degradation warning, so
  B's bundle definition matches A's.
- **Imaging provenance (M11).** B's imaging is our own: `work/evt<ID>`
  symlinked, all 48 dirs mtime 2026-07-21 = doc 21's campaign. Reuse across the
  fsprod boundary is licensed by doc 21's identical `frames-dnn.tar.bz2` member
  hash `46ff819f…`.
- **Flash offsets.** B ran `-caf product`; logs show non-256-ns-quantised
  offsets (e.g. 1948 ns, 2387 ns), confirming the authoritative
  `FrameShiftInfo::fFrameApplyAtCaf` and not the `auto` fallback.

### 4.1 Two runner traps hit while producing B

Both were caught before any number was computed; recording them because the
second is invisible in the output.

1. **`run_nusel_evt.sh` takes no flags by default.** A bare
   `./run_nusel_evt.sh data all` runs without `-chord -rescue -fvz …
   -unmerge-assoc`, i.e. not the production configuration. Use the `NUF`
   string from `run_full1k_nusel.sh` verbatim.
2. **`SBND_QL_LM` defaults to 0 in the runner** (`QL_LM="${SBND_QL_LM:-0}"`),
   while `run_full1k_nusel.sh`'s `NUF` passes `-lm`. Without it the Q/L step
   never stamps `lm_flag`, every in-beam row reads `lm = -1`, and the **LM
   category silently collapses into `nu-candidate`**. On sample B the final LM
   count is 0 anyway, so a `-lm`-less run would have produced the same visible
   table for the wrong reason. The `lm == -1` integrity check above is what
   distinguishes them — run it on any new arm.
   Note also that a re-run **cannot** repair this in place: the runner skips
   the Q/L step whenever a pctree already exists, so a `-lm` re-run in the same
   root reuses the LM-less pctree. It needs a fresh tag.

## 5. State left on disk

- **`work-nuecc48-nuf/`** — sample B's arm, 48 events, the tag this doc's
  numbers come from. `evt<ID>` are symlinks into `work/`; run `relink_tags.py`
  if `work/` ever moves.
- **`work-nuecc48-base/`** — an **abandoned** 17-event partial from the first
  attempt above (no NUF flags, no `-lm`). It is not a sibling arm and its
  numbers are not in this doc; safe to delete.
- Sample A's arms (`work-stmcamp-d66*`, `work-mcp1kall-d59k`) were **read
  only** — no existing tag, label dir or snapshot was written (M13).
- All `work*/` are gitignored, so neither tag is committed.
